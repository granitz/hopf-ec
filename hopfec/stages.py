"""Stage runners used by the CLI: participants/atlas resolution, time-series extraction, SC building."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .atlases import Atlas, select_atlas
from .bids import find_subjects, groups_dict, read_participants, select_participants, validate_participants
from .config import cfg_get
from .inputs.fmriprep import find_fmriprep_runs
from .inputs.halfpipe import find_halfpipe_bold, halfpipe_roots
from .timeseries import extract_run_timeseries, save_timeseries, timeseries_output_paths
from .utils import LOG, ensure_dir, save_json, save_matrix, sub_label, which


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(s))[:60]


# --------------------------------------------------------------------------- participants & atlas
def load_participants(cfg: dict, participant_labels: list[str] | None = None) -> pd.DataFrame:
    tsv = cfg_get(cfg, "paths.participants_tsv")
    bids_dir = cfg_get(cfg, "paths.bids_dir")
    if tsv and Path(tsv).exists():
        df = read_participants(tsv, cfg_get(cfg, "participants.id_column"), cfg_get(cfg, "participants.group_column"))
    else:
        subs = []
        for d in (bids_dir, cfg_get(cfg, "paths.fmriprep_dir"), cfg_get(cfg, "paths.halfpipe_dir"), cfg_get(cfg, "paths.output_dir")):
            if d and Path(d).exists():
                subs = find_subjects(d) or sorted(p.name for p in Path(d).glob("**/sub-*") if p.is_dir() and p.name.count("-") == 1)
                if subs:
                    break
        if not subs:
            raise FileNotFoundError("no participants.tsv and no sub-* folders found; set paths.participants_tsv")
        LOG.warning("no participants table: %d subjects from folders, all assigned to group 'all'", len(subs))
        df = pd.DataFrame({"participant_id": [sub_label(s) for s in subs], "group": "all"})
    df = select_participants(df, cfg_get(cfg, "participants.include"), cfg_get(cfg, "participants.exclude"))
    if participant_labels:
        df = select_participants(df, participant_labels)
    if bids_dir and Path(bids_dir).exists():
        v = validate_participants(df, find_subjects(bids_dir))
        if v["missing_in_bids"]:
            LOG.warning("%d participants of the table are not in the BIDS dataset: %s", len(v["missing_in_bids"]), v["missing_in_bids"][:5])
    return df


def get_atlas(cfg: dict) -> Atlas:
    return select_atlas(cfg_get(cfg, "paths.atlas_dir"), cfg_get(cfg, "atlas.name"), cfg_get(cfg, "atlas.file"),
                        cfg_get(cfg, "atlas.labels"), prefer_space=cfg_get(cfg, "input.space"))


# --------------------------------------------------------------------------- time series stage
def discover_bold_runs(cfg: dict, sub: str) -> list:
    src = cfg_get(cfg, "input.source", "auto")
    task, ses, run = cfg_get(cfg, "input.task", "rest"), cfg_get(cfg, "input.session"), cfg_get(cfg, "input.run")
    space, res, tr = cfg_get(cfg, "input.space"), cfg_get(cfg, "input.res"), cfg_get(cfg, "input.tr")
    bids_dir = cfg_get(cfg, "paths.bids_dir")
    runs = []
    if src in ("auto", "fmriprep") and cfg_get(cfg, "paths.fmriprep_dir"):
        runs = find_fmriprep_runs(cfg["paths"]["fmriprep_dir"], sub, task, space, res, ses, run, bids_dir, tr)
    if not runs and src in ("auto", "halfpipe") and cfg_get(cfg, "paths.halfpipe_dir"):
        runs = find_halfpipe_bold(cfg["paths"]["halfpipe_dir"], sub, cfg_get(cfg, "input.halfpipe_setting"), task, ses, run, bids_dir, tr, space, res)
    return runs


def _ts_job(run, atlas: Atlas, cfg: dict, out_dir: Path) -> dict:
    try:
        ts, info = extract_run_timeseries(run, atlas, cfg.get("postprocess", {}), cfg_get(cfg, "input.tr"))
        tsv, _ = timeseries_output_paths(out_dir, run.entities, run.sub, atlas)
        save_timeseries(ts, atlas.region_names, tsv, info)
        return {"sub": run.sub, "run": run.tag(), "output": str(tsv), "n_volumes": info["n_volumes"], "ok": True}
    except Exception as e:  # noqa: BLE001
        LOG.error("%s: %s", run.tag(), e)
        return {"sub": run.sub, "run": run.tag(), "ok": False, "error": str(e)}


def run_timeseries_stage(cfg: dict, participants: pd.DataFrame, atlas: Atlas, n_jobs: int = 1) -> list[dict]:
    out_dir = ensure_dir(cfg["paths"]["output_dir"])
    _write_dataset_description(out_dir)
    runs = []
    for sub in participants["participant_id"]:
        r = discover_bold_runs(cfg, sub)
        if not r:
            LOG.warning("%s: no preprocessed BOLD found (fmriprep_dir=%s, halfpipe_dir=%s)", sub, cfg_get(cfg, "paths.fmriprep_dir"), cfg_get(cfg, "paths.halfpipe_dir"))
        runs += r
    LOG.info("time-series stage: %d runs, atlas %s (%d parcels)", len(runs), atlas.name, atlas.n_parcels)
    results = Parallel(n_jobs=n_jobs)(delayed(_ts_job)(r, atlas, cfg, out_dir) for r in runs)
    save_json(out_dir / f"timeseries_atlas-{slug(atlas.name)}_log.json", results)
    n_ok = sum(1 for r in results if r["ok"])
    LOG.info("time-series stage done: %d/%d runs ok", n_ok, len(results))
    return results


def _write_dataset_description(out_dir: Path) -> None:
    p = out_dir / "dataset_description.json"
    if not p.exists():
        from . import __version__

        save_json(p, {"Name": "hopfec derivatives", "BIDSVersion": "1.9.0", "DatasetType": "derivative",
                      "GeneratedBy": [{"Name": "hopfec", "Version": __version__}]})


# --------------------------------------------------------------------------- SC stage
def run_normative_stage(cfg: dict, atlas: Atlas) -> dict:
    from .sc.normative import build_normative_sc

    conn = cfg_get(cfg, "paths.normative_connectome")
    if not conn:
        raise ValueError("paths.normative_connectome must point to dTOR_fibers_vox_2_mm.mat(.gz)")
    out_dir = ensure_dir(Path(cfg["paths"]["output_dir"]) / "sc")
    ncfg = dict(cfg_get(cfg, "sc.normative", {}))
    ncfg.setdefault("template_dir", cfg_get(cfg, "paths.template_dir"))
    res = build_normative_sc(atlas, conn, out_dir, ncfg, work_dir=Path(cfg["paths"]["work_dir"]) / "normative")
    return res["summary"]


def _find_raw_dwi(bids_dir: str | Path | None, sub: str) -> dict:
    out: dict = {}
    if not bids_dir:
        return out
    root = Path(bids_dir) / sub
    dwis = sorted(root.glob("**/dwi/*_dwi.nii.gz"))
    if not dwis:
        return out
    dwi = dwis[0]
    stem = dwi.name[: -len(".nii.gz")]
    out["dwi"] = dwi
    for ext, k in ((".bval", "bval"), (".bvec", "bvec"), (".json", "json")):
        c = dwi.with_name(stem + ext)
        if c.exists():
            out[k] = c
    for ext in (".bval", ".bvec"):
        if ext[1:] not in out:
            c = sorted(Path(bids_dir).glob(f"*dwi{ext}"))
            if c:
                out[ext[1:]] = c[0]
    t1 = sorted(root.glob("**/anat/*_T1w.nii.gz"))
    if t1:
        out["t1w"] = t1[0]
    if "json" in out:
        try:
            out["pe_dir"] = json.loads(out["json"].read_text()).get("PhaseEncodingDirection")
        except Exception:  # noqa: BLE001
            pass
    return out


def _template_t1(space: str | None, template_dir: str | None) -> Path | None:
    if template_dir:
        for pat in (f"*{space}*T1w*brain*", f"*{space}*T1w*", "MNI152_T1_2mm_brain.nii.gz", "MNI152_T1_1mm_brain.nii.gz"):
            hits = sorted(Path(template_dir).glob(pat))
            if hits:
                return hits[0]
    try:
        import templateflow.api as tf

        p = tf.get(space or "MNI152NLin2009cAsym", resolution=1, desc="brain", suffix="T1w", extension=".nii.gz")
        p = p[0] if isinstance(p, list) else p
        if p:
            return Path(p)
    except Exception as e:  # noqa: BLE001
        LOG.debug("templateflow T1w lookup failed: %s", e)
    import os

    fsl = Path(os.environ.get("FSLDIR", "")) / "data" / "standard" / "MNI152_T1_1mm_brain.nii.gz"
    return fsl if fsl.exists() else None


def run_tractography_stage(cfg: dict, participants: pd.DataFrame, atlas: Atlas, execute: bool | None = None, n_jobs: int = 1) -> list[dict]:
    from .sc.common import normalise_matrices, sc_summary
    from .sc.qsi import find_qsiprep_dwi
    from .sc.tractography import ants_warp_atlas_script, detect_backend, mrtrix_script, read_mrtrix_connectome, run_dipy_tractography, run_script, write_script

    tcfg = dict(cfg_get(cfg, "sc.tractography", {}))
    execute = bool(tcfg.get("execute", False)) if execute is None else execute
    backend = detect_backend(tcfg.get("backend", "auto"))
    out_dir = ensure_dir(cfg["paths"]["output_dir"])
    work = ensure_dir(Path(cfg["paths"]["work_dir"]) / "tractography")
    aslug = slug(atlas.name)
    results = []
    for sub in participants["participant_id"]:
        info: dict = {"sub": sub, "backend": backend}
        q = find_qsiprep_dwi(cfg["paths"]["qsiprep_dir"], sub) if cfg_get(cfg, "paths.qsiprep_dir") else {}
        raw = _find_raw_dwi(cfg_get(cfg, "paths.bids_dir"), sub) if not q else {}
        d = q or raw
        if not d.get("dwi"):
            info["error"] = "no DWI found (qsiprep or raw BIDS)"
            LOG.warning("%s: %s", sub, info["error"])
            results.append(info)
            continue
        preprocessed = bool(q)
        sdir = ensure_dir(work / sub)
        odir = ensure_dir(out_dir / sub / "dwi")
        base = odir / f"{sub}_atlas-{aslug}_desc-tractography"
        if backend == "mrtrix":
            atlas_dwi = sdir / "atlas_in_dwi.nii.gz"
            script = "#!/usr/bin/env bash\nset -euo pipefail\n"
            tpl = _template_t1(atlas.space, cfg_get(cfg, "paths.template_dir"))
            if which("antsRegistrationSyNQuick.sh") or which("antsApplyTransforms"):
                if d.get("t1w") and tpl:
                    script += ants_warp_atlas_script(atlas.image, tpl, Path(d["t1w"]), sdir, ref_image=Path(d.get("t1w")), existing_xfm=d.get("xfm_mni_to_anat"))
                else:
                    script += f"echo 'WARNING: no T1w/template for atlas registration; provide {atlas_dwi} yourself'\n"
            else:
                script += f"echo 'WARNING: ANTs not found; provide the atlas in DWI space at {atlas_dwi}'\n"
            script += mrtrix_script(Path(d["dwi"]), d.get("bval"), d.get("bvec"), sdir, atlas_dwi, tcfg, mask=d.get("mask"), t1w=d.get("t1w"),
                                    preprocessed=preprocessed, pe_dir=d.get("pe_dir"), grad_mrtrix=d.get("mrtrix_grad"), nthreads=max(1, int(cfg_get(cfg, "compute.threads_per_job", 1)) * 4))
            sp = write_script(script, sdir / f"{sub}_tractography.sh")
            info["script"] = str(sp)
            if execute:
                rc = run_script(sp)
                info["returncode"] = rc
                if rc == 0 and (sdir / "sc_count.csv").exists():
                    count = read_mrtrix_connectome(sdir / "sc_count.csv")
                    lenw = read_mrtrix_connectome(sdir / "sc_invlength.csv")
                    import nibabel as nib

                    lab = np.rint(np.asarray(nib.load(str(atlas_dwi)).dataobj)).astype(int)
                    vols = np.array([np.sum(lab == int(l)) for l in atlas.label_ids], float)
                    mats = normalise_matrices(count, lenw, vols)
                    for k, M in mats.items():
                        save_matrix(Path(str(base) + f"_weight-{k}_connectivity.tsv"), M, atlas.region_names)
                    save_json(Path(str(base) + "_connectivity.json"), {"backend": "mrtrix", **sc_summary(count, atlas.hemispheres), "script": str(sp)})
                    info["ok"] = True
            else:
                LOG.info("%s: MRtrix script written (not executed): %s", sub, sp)
        else:
            if not execute:
                info["note"] = "DIPY backend runs only with execute=True"
                results.append(info)
                continue
            tpl = _template_t1(atlas.space, cfg_get(cfg, "paths.template_dir"))
            try:
                res = run_dipy_tractography(Path(d["dwi"]), Path(d["bval"]), Path(d["bvec"]), sdir, atlas, tcfg, mask=d.get("mask"), t1w=d.get("t1w"),
                                            template_t1=tpl, atlas_in_dwi=(sdir / "atlas_in_dwi.nii.gz") if (sdir / "atlas_in_dwi.nii.gz").exists() else None)
                for k in ("count", "volnorm", "lencorr", "combined"):
                    save_matrix(Path(str(base) + f"_weight-{k}_connectivity.tsv"), res[k], atlas.region_names)
                save_json(Path(str(base) + "_connectivity.json"), {"backend": "dipy", **res["summary"], "n_streamlines": res["n_streamlines"]})
                info["ok"] = True
            except Exception as e:  # noqa: BLE001
                LOG.error("%s: DIPY tractography failed: %s", sub, e)
                info["error"] = str(e)
        results.append(info)
    save_json(out_dir / f"tractography_atlas-{aslug}_log.json", results)
    return results


def run_qsirecon_stage(cfg: dict, participants: pd.DataFrame, atlas: Atlas) -> list[dict]:
    """Copy the selected QSIRecon connectivity matrices into the derivatives folder (for inspection)."""
    from .pipeline import find_subject_sc

    out_dir = ensure_dir(cfg["paths"]["output_dir"])
    aslug = slug(atlas.name)
    res = []
    for sub in participants["participant_id"]:
        M, src = find_subject_sc({**cfg, "sc": {**cfg.get("sc", {}), "source": "qsirecon"}}, sub, atlas)
        if M is None:
            res.append({"sub": sub, "ok": False})
            continue
        p = ensure_dir(out_dir / sub / "dwi") / f"{sub}_atlas-{aslug}_desc-qsirecon_connectivity.tsv"
        save_matrix(p, M, atlas.region_names if M.shape[0] == atlas.n_parcels else None)
        res.append({"sub": sub, "ok": True, "source": src, "output": str(p)})
    return res


def summarize_inputs(cfg: dict, participants: pd.DataFrame) -> dict:
    """What the configured folders contain, per participant (for `hopfec inputs`)."""
    from .inputs.halfpipe import list_halfpipe_features
    from .pipeline import find_subject_sc, find_subject_timeseries

    atlas = None
    try:
        atlas = get_atlas(cfg)
    except Exception as e:  # noqa: BLE001
        LOG.warning("atlas not resolved: %s", e)
    out: dict = {"paths": cfg["paths"], "atlas": atlas.summary() if atlas else None, "participants": {}}
    if cfg_get(cfg, "paths.halfpipe_dir"):
        out["halfpipe"] = list_halfpipe_features(cfg["paths"]["halfpipe_dir"])
    for sub in participants["participant_id"]:
        entry = {"bold_runs": [r.tag() for r in discover_bold_runs(cfg, sub)]}
        if atlas:
            entry["timeseries"] = [str(f.path.name) for f in find_subject_timeseries(cfg, sub, atlas)]
            M, src = find_subject_sc(cfg, sub, atlas)
            entry["sc"] = src
        out["participants"][sub] = entry
    return out


# --------------------------------------------------------------------------- NDTE stage
def run_ndte_stage(cfg: dict, participants: pd.DataFrame, atlas: Atlas, n_jobs: int = 1) -> dict:
    """Empirical NDTE with surrogates per participant (+ group averages)."""
    from .models.ndte import ndte_with_surrogates
    from .nodes import determine_node_set, scan_node_validity
    from .pipeline import Subject, find_subject_timeseries, load_subject_empirical

    ncfg = dict(cfg_get(cfg, "model.ndte", {}) or {})
    out_dir = ensure_dir(cfg["paths"]["output_dir"])
    aslug = slug(atlas.name)
    N_full = atlas.n_parcels
    subs = []
    for _, r in participants.iterrows():
        s = Subject(sub=r["participant_id"], group=r.get("group"))
        s.files = find_subject_timeseries(cfg, s.sub, atlas)
        if s.files:
            subs.append(s)
    # model node set (parcels with valid data in every participant), as in the fit stage
    valid = Parallel(n_jobs=n_jobs)(delayed(scan_node_validity)(s.files, N_full) for s in subs)
    nodes = determine_node_set({s.sub: v for s, v in zip(subs, valid)}, atlas, cfg.get("nodes") or {})
    subs = [s for s in subs if s.sub not in nodes.excluded_participants]
    nodes.table().to_csv(out_dir / f"atlas-{aslug}_desc-nodes.tsv", sep="\t", index=False, float_format="%.4g")
    names = nodes.subset_atlas(atlas).region_names
    names_full = nodes.names

    def savem(path, M):
        return save_matrix(path, nodes.expand(M), names_full)

    LOG.info("NDTE stage: %d participants, %d/%d parcels, max_lag %s, %s surrogates", len(subs), nodes.n_kept, N_full, ncfg.get("max_lag", 10), ncfg.get("n_surrogates", 100))

    def _job(s: Subject):
        try:
            emp = load_subject_empirical(s.files, {**cfg, "model": {**cfg["model"], "ndte": {**ncfg, "enabled": False}}}, N_full, nodes=nodes)
            X = np.concatenate(emp["filtered_runs"], axis=0)
            res = ndte_with_surrogates(X, int(ncfg.get("max_lag", 10)), int(ncfg.get("n_surrogates", 100)), int(ncfg.get("seed", 0)),
                                       float(ncfg.get("fdr_q", 0.05)), n_jobs=1, p_source=str(ncfg.get("p_source", "kde")))
            d = ensure_dir(out_dir / s.sub / "func")
            base = d / f"{s.sub}_atlas-{aslug}"
            savem(Path(str(base) + "_desc-ndte_connectivity.tsv"), res["ndte"])
            if "z" in res:
                savem(Path(str(base) + "_desc-ndteZ_connectivity.tsv"), res["z"])
                savem(Path(str(base) + "_desc-ndteP_connectivity.tsv"), res["p_kde"])
                savem(Path(str(base) + "_desc-ndteSig_connectivity.tsv"), res["sig_fdr"].astype(int))
            pd.DataFrame({"region": names_full, "in_flow": nodes.expand_vec(res["in_flow"]), "out_flow": nodes.expand_vec(res["out_flow"]),
                          "total_flow": nodes.expand_vec(res["total_flow"])}).to_csv(str(base) + "_desc-ndteFlow.tsv", sep="\t", index=False, float_format="%.6g")
            save_json(Path(str(base) + "_desc-ndte.json"), {"n_volumes": int(X.shape[0]), "max_lag": res["max_lag"], "n_surrogates": res["n_surrogates"],
                                                            "n_significant": int(res["sig_fdr"].sum()) if "sig_fdr" in res else None, "p_source": res.get("p_source")})
            return {"sub": s.sub, "group": s.group, "ndte": res["ndte"], "z": res.get("z"), "sig": res.get("sig_fdr"), "ok": True}
        except Exception as e:  # noqa: BLE001
            LOG.error("%s: NDTE failed: %s", s.sub, e)
            return {"sub": s.sub, "group": s.group, "ok": False, "error": str(e)}

    results = Parallel(n_jobs=n_jobs)(delayed(_job)(s) for s in subs)
    ok = [r for r in results if r["ok"]]
    summary: dict = {"n_ok": len(ok), "n_failed": len(results) - len(ok), "groups": {}, "nodes": nodes.summary()}
    groups = {"all": ok}
    for r in ok:
        if r["group"]:
            groups.setdefault(r["group"], []).append(r)
    for g, rs in groups.items():
        if len(rs) < 1:
            continue
        gd = ensure_dir(out_dir / "ndte")
        base = gd / f"group-{slug(g)}_atlas-{aslug}"
        mean = np.mean([r["ndte"] for r in rs], 0)
        savem(Path(str(base) + "_desc-ndteMean_connectivity.tsv"), mean)
        if all(r["z"] is not None for r in rs):
            savem(Path(str(base) + "_desc-ndteZmean_connectivity.tsv"), np.mean([r["z"] for r in rs], 0))
            savem(Path(str(base) + "_desc-ndteSigFraction_connectivity.tsv"), np.mean([r["sig"] for r in rs], 0))
        flow_in, flow_out = mean.sum(1), mean.sum(0)
        pd.DataFrame({"region": names_full, "in_flow": nodes.expand_vec(flow_in), "out_flow": nodes.expand_vec(flow_out),
                      "total_flow": nodes.expand_vec(flow_in + flow_out)}).sort_values("total_flow", ascending=False).to_csv(str(base) + "_desc-ndteFlow.tsv", sep="\t", index=False, float_format="%.6g")
        summary["groups"][g] = {"n": len(rs), "top_regions": [names[i] for i in np.argsort(-(flow_in + flow_out))[:10]]}
    save_json(out_dir / f"ndte_summary_atlas-{aslug}.json", summary)
    return summary
