"""End-to-end effective-connectivity workflow.

    inputs (time series per participant)  ->  empirical FC / lagged correlation / spectra
    structural connectivity (per participant or shared)
    parameter search (G x a error surface)  ->  participant-level EC  ->  group-level EC
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .atlases import Atlas
from .config import cfg_get
from .inputs.common import TimeseriesFile
from .inputs.halfpipe import find_halfpipe_timeseries
from .models.gec import fit_gec, initial_ec, make_mask
from .models.hopf_linear import analytic_moments
from .models.hopf_nonlinear import simulated_moments
from .models.linear_gradient import fit_linear_gradient
from .models.nonlinear_fit import fit_nonlinear_surrogate, noise_floor
from .models.search import adaptive_search, error_surface
from .models.signal import average_spectra, empirical_moments, fcd_distribution, metastability, peak_frequencies
from .plotting import plot_ec_summary, plot_error_surface, plot_fit, plot_group_comparison
from .sc.common import prepare_sc
from .timeseries import load_timeseries
from .utils import LOG, ensure_dir, linspace_spec, load_matrix, save_json, save_matrix, sub_label

MODEL_TAGS = {"linear": "hopflinear", "nonlinear": "hopf"}


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(s))[:60]


@dataclass
class Subject:
    sub: str
    group: str | None = None
    files: list[TimeseriesFile] = field(default_factory=list)
    emp: dict | None = None
    SC: np.ndarray | None = None
    sc_source: str = ""
    errors: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- locating inputs
def find_subject_timeseries(cfg: dict, sub: str, atlas: Atlas) -> list[TimeseriesFile]:
    sub = sub_label(sub)
    src = cfg_get(cfg, "input.source", "auto")
    task = cfg_get(cfg, "input.task", "rest")
    ses = cfg_get(cfg, "input.session")
    run = cfg_get(cfg, "input.run")
    tr = cfg_get(cfg, "input.tr")
    bids_dir = cfg_get(cfg, "paths.bids_dir")
    out_dir = Path(cfg["paths"]["output_dir"])
    files: list[TimeseriesFile] = []
    glob_spec = cfg_get(cfg, "input.timeseries_glob")
    if src in ("timeseries", "auto") and glob_spec:
        pat = str(glob_spec).format(sub=sub, subject=sub.replace("sub-", ""))
        for p in sorted(Path("/").glob(pat.lstrip("/")) if pat.startswith("/") else Path(cfg["paths"]["root"]).glob(pat)):
            from .bids import parse_entities

            e = parse_entities(p.name)
            files.append(TimeseriesFile(sub=sub, path=p, task=e.get("task"), ses=e.get("ses"), run=e.get("run"), atlas=e.get("atlas"),
                                        tr=tr, source="timeseries", entities=e))
        if files or src == "timeseries":
            return files
    if src in ("auto", "fmriprep", "halfpipe"):
        pat = f"{sub}/**/func/{sub}*_atlas-{slug(atlas.name)}_desc-clean_timeseries.tsv"
        for p in sorted(out_dir.glob(pat)):
            from .bids import parse_entities

            e = parse_entities(p.name)
            if task and e.get("task") != task:
                continue
            if ses and e.get("ses") != ses:
                continue
            side = p.with_suffix(".json")
            this_tr = tr
            if this_tr is None and side.exists():
                this_tr = json.loads(side.read_text()).get("tr")
            files.append(TimeseriesFile(sub=sub, path=p, task=e.get("task"), ses=e.get("ses"), run=e.get("run"), atlas=e.get("atlas"),
                                        tr=this_tr, source="hopfec", sidecar=side if side.exists() else None, entities=e))
        if files:
            return files
    if src in ("auto", "halfpipe-timeseries") and cfg_get(cfg, "paths.halfpipe_dir"):
        files = find_halfpipe_timeseries(cfg["paths"]["halfpipe_dir"], sub, atlas=cfg_get(cfg, "input.halfpipe_atlas"),
                                         feature=cfg_get(cfg, "input.halfpipe_feature"), task=task, ses=ses, run=run, bids_dir=bids_dir, tr=tr)
    return files


def load_subject_empirical(files: list[TimeseriesFile], cfg: dict, n_expected: int, tr_override: float | None = None) -> dict:
    """Empirical statistics of one participant averaged over runs."""
    band = cfg_get(cfg, "model.filter_band", [0.008, 0.08])
    freq_band = cfg_get(cfg, "model.freq_band") or band
    tau = int(cfg_get(cfg, "model.tau_tr", 1))
    smooth = float(cfg_get(cfg, "model.spectrum_smoothing_hz", 0.01))
    fcd_cfg = cfg_get(cfg, "model.fcd", {})
    min_vol = int(cfg_get(cfg, "model.min_volumes", 50))
    fcs, covs, specs, fcds, metas, nvols, trs, used = [], [], [], [], [], [], [], []
    bad_nodes: set[int] = set()
    for f in files:
        X, _names = load_timeseries(f.path, n_expected)
        tr = tr_override or f.tr or cfg_get(cfg, "input.tr")
        if tr is None:
            raise ValueError(f"{f.path.name}: repetition time unknown; set input.tr")
        if X.shape[1] != n_expected:
            raise ValueError(f"{f.path.name}: {X.shape[1]} columns but the atlas has {n_expected} parcels")
        nan_cols = np.where(~np.isfinite(X).all(axis=0) | (np.nanstd(X, axis=0) == 0))[0]
        if len(nan_cols):
            bad_nodes.update(int(c) for c in nan_cols)
            X = np.nan_to_num(X)
        if X.shape[0] < min_vol:
            LOG.warning("%s: only %d volumes (< %d), run skipped", f.path.name, X.shape[0], min_vol)
            continue
        if X.shape[0] <= tau + 2:
            continue
        m = empirical_moments(X, float(tr), band, tau, freq_band, smooth)
        fcs.append(m["FC"])
        covs.append(m["COVtau"])
        specs.append(m["spectrum"])
        fcds.append(fcd_distribution(m["filtered"], int(fcd_cfg.get("window_tr", 30)), int(fcd_cfg.get("step_tr", 3))))
        metas.append(m["metastability"])
        nvols.append(m["n_volumes"])
        trs.append(float(tr))
        used.append(str(f.path))
    if not fcs:
        raise ValueError("no usable runs")
    if len(set(np.round(trs, 4))) > 1:
        LOG.warning("runs with different TR (%s); using the first for the model", trs)
    freqs, P = average_spectra(specs)
    f_peak, _ = peak_frequencies(None, trs[0], freq_band, smooth, spectrum=(freqs, P))
    return {"FC": np.mean(fcs, 0), "COVtau": np.mean(covs, 0), "f_peak": f_peak, "spectrum": (freqs, P),
            "fcd": np.concatenate(fcds), "metastability": float(np.mean(metas)), "n_volumes": int(np.mean(nvols)),
            "tr": trs[0], "n_runs": len(fcs), "files": used, "bad_nodes": sorted(bad_nodes)}


def group_empirical(subjects: list[Subject], freq_band, smooth: float) -> dict:
    emps = [s.emp for s in subjects if s.emp is not None]
    trs = sorted(set(round(e["tr"], 4) for e in emps))
    if len(trs) > 1:
        LOG.warning("participants have different TRs %s; group model uses the most common one", trs)
    tr = float(pd.Series([e["tr"] for e in emps]).mode().iloc[0])
    freqs, P = average_spectra([e["spectrum"] for e in emps])
    f_peak, _ = peak_frequencies(None, tr, freq_band, smooth, spectrum=(freqs, P))
    return {"FC": np.mean([e["FC"] for e in emps], 0), "COVtau": np.mean([e["COVtau"] for e in emps], 0), "f_peak": f_peak,
            "spectrum": (freqs, P), "fcd": np.concatenate([e["fcd"] for e in emps]), "metastability": float(np.mean([e["metastability"] for e in emps])),
            "n_volumes": int(np.mean([e["n_volumes"] for e in emps])), "tr": tr, "n_subjects": len(emps)}


# --------------------------------------------------------------------------- structural connectivity
def find_subject_sc(cfg: dict, sub: str, atlas: Atlas) -> tuple[np.ndarray | None, str]:
    """Raw SC for one participant according to sc.source (auto: normative -> tractography -> qsirecon -> file)."""
    src = cfg_get(cfg, "sc.source", "auto")
    out_dir = Path(cfg["paths"]["output_dir"])
    weight = cfg_get(cfg, "sc.weight", "combined")
    aslug = slug(atlas.name)
    order = [src] if src != "auto" else ["file", "tractography", "qsirecon", "normative"]
    for s in order:
        if s == "file" and cfg_get(cfg, "sc.file"):
            p = Path(str(cfg["sc"]["file"]).format(sub=sub, subject=sub.replace("sub-", "")))
            if p.exists():
                return load_matrix(p, cfg_get(cfg, "sc.qsirecon_key")), f"file:{p}"
            if src == "file":
                raise FileNotFoundError(p)
        if s == "normative":
            cands = sorted((out_dir / "sc").glob(f"atlas-{aslug}_desc-normative_weight-{weight}_connectivity.tsv"))
            if cands:
                return load_matrix(cands[0]), f"normative:{cands[0].name}"
        if s == "tractography":
            cands = sorted((out_dir / sub / "dwi").glob(f"{sub}_atlas-{aslug}_desc-tractography_weight-{weight}_connectivity.tsv")) if (out_dir / sub / "dwi").exists() else []
            if cands:
                return load_matrix(cands[0]), f"tractography:{cands[0].name}"
        if s == "qsirecon" and cfg_get(cfg, "paths.qsirecon_dir"):
            from .sc.qsi import find_qsirecon_connectivity, load_connectivity_mat

            files = find_qsirecon_connectivity(cfg["paths"]["qsirecon_dir"], sub, atlas_hint=atlas.name)
            for f in files:
                try:
                    if f.suffix == ".mat":
                        M, key, _ = load_connectivity_mat(f, cfg_get(cfg, "sc.qsirecon_key"), atlas.name)
                    else:
                        M = load_matrix(f)
                    if M.shape[0] == atlas.n_parcels:
                        return M, f"qsirecon:{f.name}"
                except Exception as e:  # noqa: BLE001
                    LOG.debug("qsirecon file %s not usable: %s", f, e)
    return None, "none"


# --------------------------------------------------------------------------- fitting helpers
def _model_cfg(cfg: dict) -> dict:
    return cfg.get("model", {})


def _omega(emp: dict) -> np.ndarray:
    return 2 * np.pi * np.asarray(emp["f_peak"], float)


def run_search(model: str, SC: np.ndarray, emp: dict, cfg: dict, out_base: Path, n_jobs: int, title: str = "") -> dict:
    """Adaptive (G x a) search: grid, border extension, refinement, interpolation, optional continuous optimisation."""
    m = _model_cfg(cfg)
    s = m.get("search", {})
    G_values = linspace_spec(s.get("G", {"start": 0.0, "stop": 3.0, "step": 0.1}))
    a_values = linspace_spec(s.get("a")) if s.get("a") is not None else None
    metric = s.get("metric", "fit_rmse")
    df, best, info = adaptive_search(model, SC, _omega(emp), emp, emp["tr"], G_values, a_values, m.get("filter_band"), search_cfg=s,
                                     tau_tr=int(m.get("tau_tr", 1)), beta=float(m.get("beta", 0.02)), dt=float(m.get("dt", 0.1)),
                                     n_sim=int(s.get("n_sim", 1)), seed=int(cfg_get(cfg, "model.nonlinear.seed", 0)),
                                     transient_s=float(m.get("transient_s", 100.0)), filter_consistent=bool(cfg_get(cfg, "model.linear.filter_consistent", True)),
                                     metric=metric, fcd_cfg=m.get("fcd", {}), n_jobs=n_jobs, use_numba=cfg_get(cfg, "model.nonlinear.use_numba", "auto"),
                                     default_a=_scalar_a(m))
    ensure_dir(out_base.parent)
    df.to_csv(str(out_base) + "_desc-errorsurface_table.tsv", sep="\t", index=False, float_format="%.6g")
    Gs, As, surf = error_surface(df, metric)
    pd.DataFrame(surf, index=[f"G={g:g}" for g in Gs], columns=[f"a={a:g}" for a in As]).to_csv(str(out_base) + f"_desc-errorsurface_{metric}.tsv", sep="\t", float_format="%.6g")
    np.savez(str(out_base) + "_desc-errorsurface.npz", G=Gs, a=As, surface=surf, metric=metric)
    for extra in ("fc_corr", "fcd_ks"):
        if extra in df.columns and extra != metric:
            _, _, s2 = error_surface(df, extra)
            pd.DataFrame(s2, index=[f"G={g:g}" for g in Gs], columns=[f"a={a:g}" for a in As]).to_csv(str(out_base) + f"_desc-errorsurface_{extra}.tsv", sep="\t", float_format="%.6g")
    refined = df[df["stage"] == "refined"] if "stage" in df.columns else None
    plot_error_surface(Gs, As, surf, metric, Path(str(out_base) + "_desc-errorsurface.png"), best, title=title,
                       extra_points=refined, used=(best["G"], best["a"]), grid_best=(best.get("G_grid", best["G"]), best.get("a_grid", best["a"])))
    save_json(Path(str(out_base) + "_desc-search.json"), {"best": best, "metric": metric, "G_values": Gs, "a_values": As, "n_points": len(df), **info})
    LOG.info("%s search (%s): G=%.4g a=%.4g via %s; %s=%.4g at the grid optimum%s", model, title, best["G"], best["a"], best.get("source", "grid"),
             metric, best[metric], "; WARNINGS: " + " | ".join(info["warnings"]) if info.get("warnings") else "")
    return best


def _scalar_a(m: dict) -> float:
    a = m.get("a", -0.02)
    return float(np.mean(a)) if isinstance(a, (list, tuple, np.ndarray)) else float(a)


def fit_one(model: str, emp: dict, SC: np.ndarray, G: float, a, cfg: dict, atlas: Atlas, C_init: np.ndarray | None = None,
            verbose: int = 0, omega_override: np.ndarray | None = None) -> dict:
    """Fit EC of one model to one set of empirical statistics.  Returns a dict of results.

    C_init / omega_override (e.g. from the linear fit) seed the non-linear model."""
    m = _model_cfg(cfg)
    N = SC.shape[0]
    tr = float(emp["tr"])
    tau = int(m.get("tau_tr", 1))
    beta = float(m.get("beta", 0.02))
    band = m.get("filter_band")
    sc_max = float(cfg_get(cfg, "sc.sc_max", 0.2))
    gcfg = m.get("gec", {})
    mask = make_mask(SC, N, gcfg.get("mask", "sc_plus_homotopic"), atlas.homotopic_pairs())
    omega = _omega(emp) if omega_override is None else np.asarray(omega_override, float)
    if model == "nonlinear" or cfg_get(cfg, "model.linear.method", "gradient") == "gec":
        if C_init is None and float(G) <= 0:
            raise ValueError("global coupling G <= 0 (uncoupled model): the GEC iteration cannot fit anything; set model.G > 0 or a search grid without 0")
    elif float(G) <= 0 and C_init is None:
        LOG.warning("G <= 0 from the search: the gradient fit starts from an all-zero coupling (poor initialisation)")
    t0 = time.time()
    out: dict = {"model": model, "G_search": float(G), "omega_source": "spectral_peaks" if omega_override is None else "linear_fit", "a": a if np.isscalar(a) else list(a), "tr": tr, "tau_tr": tau, "n_volumes": emp.get("n_volumes")}
    if model == "linear":
        lc = m.get("linear", {})
        method = lc.get("method", "gradient")
        C0 = (C_init if C_init is not None else initial_ec(SC, N, mask, gcfg.get("init", "sc"), sc_max))
        if method == "gradient":
            C0g = C0 * (G if C_init is None else 1.0)  # G absorbed into C
            res = fit_linear_gradient(emp["FC"], emp["COVtau"], C0g, mask, omega, tr, tau, a, beta, band if lc.get("filter_consistent", True) else None,
                                      w_fc=float(lc.get("w_fc", 1.0)), w_tau=float(lc.get("w_tau", 1.0)), lambda_sc=float(lc.get("lambda_sc", 0.0)),
                                      C_prior=C0g, lambda_l1=float(lc.get("lambda_l1", 0.0)), fit_a=lc.get("fit_a", "none"),
                                      fit_omega=bool(lc.get("fit_omega", True)), max_iter=int(lc.get("max_iter", 500)), verbose=verbose)
            EC = res.C
            out.update({"method": "gradient", "metrics": res.metrics, "history": res.history, "FC_sim": res.FC_sim, "COVtau_sim": res.COVtau_sim,
                        "a_fit": res.a.tolist(), "omega_fit": res.omega.tolist(), "n_iter": res.n_iter, "success": res.success, "message": res.message})
        else:
            filt = {"band": band} if lc.get("filter_consistent", True) else None
            res = fit_gec(emp["FC"], emp["COVtau"], C0, lambda C: analytic_moments(C, G, a, omega, tr, tau, beta, filt), mask, G=G,
                          eps_fc=float(gcfg.get("eps_fc", 1e-3)), eps_tau=float(gcfg.get("eps_tau", 1e-3)), max_iter=int(gcfg.get("max_iter", 2000)),
                          min_iter=int(gcfg.get("min_iter", 20)), patience=int(gcfg.get("patience", 50)),
                          normalize_max=sc_max if gcfg.get("normalize_max", True) else None, verbose=verbose)
            EC = G * res.C
            out.update({"method": "gec", "metrics": res.metrics, "history": res.history, "FC_sim": res.FC_sim, "COVtau_sim": res.COVtau_sim,
                        "n_iter": res.n_iter, "best_iter": res.best_iter, "converged": res.converged, "a_fit": None, "omega_fit": omega.tolist()})
    elif model == "nonlinear":
        nc = m.get("nonlinear", {})
        if C_init is not None:
            C0 = C_init / max(C_init.max(), 1e-12) * sc_max
            G_use = float(C_init.max() / sc_max)
        else:
            C0 = initial_ec(SC, N, mask, gcfg.get("init", "sc"), sc_max)
            G_use = float(G)
        n_vol = int(emp.get("n_volumes", 500))
        base_seed = int(nc.get("seed", 0))

        def sim(C, seed=base_seed):  # common random numbers: same seed at every iteration
            return simulated_moments(C, G_use, a, omega, tr, n_vol, band, tau, beta=beta, dt=float(m.get("dt", 0.1)),
                                     n_sim=int(nc.get("n_sim", 2)), seed=seed, transient_s=float(m.get("transient_s", 100.0)),
                                     use_numba=nc.get("use_numba", "auto"))

        nf = noise_floor(lambda C, sd: sim(C, sd), C0, emp["FC"], emp["COVtau"], n_seeds=int(nc.get("n_noise_seeds", 3))) if int(nc.get("n_noise_seeds", 3)) > 0 else {}
        nl_method = str(nc.get("method", "surrogate"))
        if nl_method == "surrogate":
            res = fit_nonlinear_surrogate(emp["FC"], emp["COVtau"], C0, sim, mask, G_use, a, omega, tr, tau, beta, band,
                                          filter_consistent=bool(cfg_get(cfg, "model.linear.filter_consistent", True)),
                                          max_iter=int(nc.get("max_iter", 300)), patience=int(nc.get("patience", 20)),
                                          step_frac=float(nc.get("step_frac", 0.05)), normalize_max=sc_max if gcfg.get("normalize_max", True) else None,
                                          verbose=verbose)
        else:
            res = fit_gec(emp["FC"], emp["COVtau"], C0, sim, mask, G=G_use, eps_fc=float(nc.get("eps_fc", 5e-4)), eps_tau=float(nc.get("eps_tau", 5e-4)),
                          max_iter=int(nc.get("max_iter", 300)), min_iter=int(gcfg.get("min_iter", 20)), patience=int(nc.get("patience", 20)),
                          normalize_max=sc_max if gcfg.get("normalize_max", True) else None, verbose=verbose,
                          accept_only_improving=bool(nc.get("accept_only_improving", True)))
        res.metrics.update(nf)
        if nf and (res.metrics.get("initial_fit_rmse", np.nan) - res.metrics["fit_rmse"]) < 2 * nf.get("noise_floor_sd", 0.0):
            res.metrics["improvement_below_noise"] = True
            LOG.warning("non-linear fit: improvement over the initialisation (%.4f -> %.4f) is within 2 SD of the simulation-noise floor (%.4f); increase nonlinear.n_sim",
                        res.metrics.get("initial_fit_rmse", np.nan), res.metrics["fit_rmse"], nf["noise_floor_sd"])
        else:
            res.metrics["improvement_below_noise"] = False
        EC = G_use * res.C
        out.update({"method": nl_method, "G_used": G_use, "metrics": res.metrics, "history": res.history, "FC_sim": res.FC_sim, "COVtau_sim": res.COVtau_sim,
                    "n_iter": res.n_iter, "best_iter": res.best_iter, "converged": res.converged, "a_fit": None, "omega_fit": omega.tolist()})
    else:
        raise ValueError(model)
    out["EC"] = EC
    out["EC_norm"] = EC / max(EC.max(), 1e-12) * sc_max
    out["G_eff"] = float(EC.max() / sc_max)
    out["mask"] = mask
    out["elapsed_s"] = time.time() - t0
    out["f_peak"] = np.asarray(emp["f_peak"]).tolist()
    return out


def save_fit(res: dict, base: Path, names: list[str], SC: np.ndarray | None, title: str) -> dict:
    ensure_dir(base.parent)
    save_matrix(Path(str(base) + "_desc-EC_connectivity.tsv"), res["EC"], names)
    save_matrix(Path(str(base) + "_desc-ECnorm_connectivity.tsv"), res["EC_norm"], names)
    if res.get("FC_sim") is not None:
        save_matrix(Path(str(base) + "_desc-modelFC_connectivity.tsv"), res["FC_sim"], names)
    meta = {k: v for k, v in res.items() if k not in ("EC", "EC_norm", "FC_sim", "COVtau_sim", "history", "mask")}
    meta["files"] = {"EC": str(base) + "_desc-EC_connectivity.tsv", "EC_norm": str(base) + "_desc-ECnorm_connectivity.tsv"}
    save_json(Path(str(base) + "_desc-fit.json"), meta)
    if res.get("history"):
        pd.DataFrame(res["history"]).to_csv(str(base) + "_desc-history.tsv", sep="\t", index=False, float_format="%.6g")
    try:
        if res.get("FC_sim") is not None:
            plot_fit(res["FC_emp"], res["FC_sim"], res.get("COVtau_emp"), res.get("COVtau_sim"), res.get("history"), Path(str(base) + "_desc-fit.png"), title)
        plot_ec_summary(res["EC"], SC, Path(str(base) + "_desc-EC.png"), title, names)
    except Exception as e:  # noqa: BLE001
        LOG.warning("figure failed for %s: %s", base.name, e)
    return meta


def _fit_participant_job(model: str, sub: Subject, G: float, a, cfg: dict, atlas: Atlas, base: Path, C_init: np.ndarray | None, title: str,
                         omega_init: np.ndarray | None = None) -> dict:
    res = fit_one(model, sub.emp, sub.SC, G, a, cfg, atlas, C_init=C_init, omega_override=omega_init)
    res["FC_emp"], res["COVtau_emp"] = sub.emp["FC"], sub.emp["COVtau"]
    res["participant_id"], res["group"], res["sc_source"] = sub.sub, sub.group, sub.sc_source
    meta = save_fit(res, base, atlas.region_names, sub.SC, title)
    row = {"participant_id": sub.sub, "group": sub.group, "model": model, "method": res.get("method"), "G": res.get("G_used", res["G_search"]),
           "G_eff": res["G_eff"], "n_volumes": sub.emp["n_volumes"], "n_runs": sub.emp.get("n_runs"), "elapsed_s": res["elapsed_s"], **res["metrics"]}
    return {"row": row, "EC": res["EC"], "EC_norm": res["EC_norm"], "omega": np.asarray(res["omega_fit"], float) if res.get("omega_fit") else None, "sub": sub.sub}


# --------------------------------------------------------------------------- main stage
def run_fit_stage(cfg: dict, participants: pd.DataFrame, atlas: Atlas, models: list[str] | None = None, n_jobs: int = 1,
                  participants_only: bool = False, group_only: bool = False, verbose: int = 0) -> dict:
    """Participant- and group-level EC for the selected models.  Returns a summary dict."""
    out_dir = ensure_dir(cfg["paths"]["output_dir"])
    m = _model_cfg(cfg)
    models = models or [k for k in ("linear", "nonlinear") if cfg_get(cfg, f"model.{k}.enabled", True)]
    aslug = slug(atlas.name)
    N = atlas.n_parcels
    names = atlas.region_names
    summary: dict = {"atlas": atlas.name, "n_parcels": N, "models": models, "participants": {}, "groups": {}}
    # ---- empirical statistics per participant (parallel)
    subs = [Subject(sub=r["participant_id"], group=r.get("group")) for _, r in participants.iterrows()]
    for s in subs:
        s.files = find_subject_timeseries(cfg, s.sub, atlas)
        if not s.files:
            s.errors.append("no time series found")
    ok = [s for s in subs if not s.errors]
    LOG.info("fit stage: %d participants with time series (%d without), models %s", len(ok), len(subs) - len(ok), models)
    emps = Parallel(n_jobs=n_jobs)(delayed(_safe_emp)(s.files, cfg, N) for s in ok)
    for s, e in zip(ok, emps):
        if isinstance(e, Exception):
            s.errors.append(f"empirical: {e}")
        else:
            s.emp = e
            base = ensure_dir(out_dir / s.sub / "func") / f"{s.sub}_atlas-{aslug}"
            save_matrix(Path(str(base) + "_desc-empiricalFC_connectivity.tsv"), e["FC"], names)
            save_matrix(Path(str(base) + "_desc-empiricalCOVtau_connectivity.tsv"), e["COVtau"], names)
            save_json(Path(str(base) + "_desc-empirical.json"), {k: v for k, v in e.items() if k not in ("FC", "COVtau", "spectrum", "fcd")})
    ok = [s for s in subs if not s.errors and s.emp is not None]
    # ---- structural connectivity
    for s in ok:
        raw, src = find_subject_sc(cfg, s.sub, atlas)
        if raw is None:
            s.errors.append("no structural connectivity (run `hopfec sc ...` or set sc.file)")
            continue
        if raw.shape != (N, N):
            s.errors.append(f"SC shape {raw.shape} != atlas ({N})")
            continue
        s.SC = prepare_sc(raw, cfg.get("sc", {}))
        s.sc_source = src
    ok = [s for s in subs if not s.errors and s.SC is not None]
    if not ok:
        raise RuntimeError("no participant is complete (time series + SC): " + "; ".join(f"{s.sub}: {s.errors}" for s in subs[:5]))
    for s in subs:
        summary["participants"][s.sub] = {"group": s.group, "n_files": len(s.files), "errors": s.errors, "sc_source": s.sc_source}
    freq_band = m.get("freq_band") or m.get("filter_band")
    smooth = float(m.get("spectrum_smoothing_hz", 0.01))
    groups = sorted(set(s.group for s in ok if s.group))
    emp_all = group_empirical(ok, freq_band, smooth)
    SC_group = prepare_sc(np.mean([s.SC for s in ok], 0), {**cfg.get("sc", {}), "symmetrize": False})
    save_matrix(out_dir / "sc" / f"group-all_atlas-{aslug}_desc-modelSC_connectivity.tsv", SC_group, names)
    a_cfg = m.get("a", -0.02)
    a = np.asarray(a_cfg, float) if isinstance(a_cfg, (list, tuple)) else float(a_cfg)
    linear_results: dict[str, dict] = {}          # sub -> {"EC", "omega"} from the linear model
    linear_group_results: dict[str, dict] = {}
    compare_on = str(cfg_get(cfg, "group.compare_on", "ECnorm"))
    for model in models:
        tag = MODEL_TAGS[model]
        mdir = ensure_dir(out_dir / "models" / f"model-{tag}")
        # ---- parameter search / global coupling
        G_fixed = m.get("G")
        best_group = None
        level = cfg_get(cfg, "model.search.level", "group")
        if G_fixed is None and cfg_get(cfg, "model.search.enabled", True) and not group_only:
            if level in ("group", "both"):
                best_group = run_search(model, SC_group, emp_all, cfg, mdir / "search" / f"group-all_atlas-{aslug}_model-{tag}", n_jobs, title=f"{model} model, all participants")
                if best_group.get("a") is not None and cfg_get(cfg, "model.search.a") is not None:
                    a = float(best_group["a"])
        G_group = float(G_fixed) if G_fixed is not None else (float(best_group["G"]) if best_group else 1.0)
        summary.setdefault("search", {})[model] = {"G_group": G_group, "a": a if np.isscalar(a) else list(a), "best": best_group}
        # ---- participant-level
        rows = []
        ECs: dict[str, np.ndarray] = {}
        ECns: dict[str, np.ndarray] = {}
        if not group_only:
            jobs = []
            for s in ok:
                G_s = G_group
                if G_fixed is None and level in ("participant", "both") and cfg_get(cfg, "model.search.enabled", True):
                    b = run_search(model, s.SC, s.emp, cfg, mdir / s.sub / f"{s.sub}_atlas-{aslug}_model-{tag}", n_jobs, title=f"{model} model, {s.sub}")
                    G_s = float(b["G"])
                use_lin = model == "nonlinear" and cfg_get(cfg, "model.nonlinear.init", "linear") == "linear" and s.sub in linear_results
                C_init = linear_results[s.sub]["EC"] if use_lin else None
                om_init = linear_results[s.sub].get("omega") if use_lin else None
                base = mdir / s.sub / f"{s.sub}_atlas-{aslug}_model-{tag}"
                jobs.append(delayed(_fit_participant_job)(model, s, G_s, a, cfg, atlas, base, C_init, f"{s.sub} {model} Hopf", om_init))
            results = Parallel(n_jobs=n_jobs)(jobs)
            for r in results:
                rows.append(r["row"])
                ECs[r["sub"]] = r["EC"]
                ECns[r["sub"]] = r["EC_norm"]
                if model == "linear":
                    linear_results[r["sub"]] = {"EC": r["EC"], "omega": r["omega"]}
        else:
            for s in ok:
                p = mdir / s.sub / f"{s.sub}_atlas-{aslug}_model-{tag}_desc-EC_connectivity.tsv"
                if p.exists():
                    ECs[s.sub] = load_matrix(p)
                    pn = p.with_name(p.name.replace("_desc-EC_", "_desc-ECnorm_"))
                    ECns[s.sub] = load_matrix(pn) if pn.exists() else ECs[s.sub] / max(ECs[s.sub].max(), 1e-12) * float(cfg_get(cfg, "sc.sc_max", 0.2))
                    fj = p.with_name(p.name.replace("_desc-EC_connectivity.tsv", "_desc-fit.json"))
                    meta = json.loads(fj.read_text()) if fj.exists() else {}
                    rows.append({"participant_id": s.sub, "group": s.group, "model": model, **meta.get("metrics", {})})
                    if model == "linear":
                        linear_results[s.sub] = {"EC": ECs[s.sub], "omega": np.asarray(meta["omega_fit"], float) if meta.get("omega_fit") else None}
        if rows:
            pd.DataFrame(rows).to_csv(mdir / f"participants_atlas-{aslug}_model-{tag}_fit.tsv", sep="\t", index=False, float_format="%.6g")
        # ---- group-level
        if participants_only:
            continue
        gcfg = cfg.get("group", {})
        group_sets = {"all": ok} if gcfg.get("pooled", True) else {}
        for g in groups:
            members = [s for s in ok if s.group == g]
            if len(members) >= int(gcfg.get("min_subjects", 2)):
                group_sets[g] = members
        grows = []
        for gname, members in group_sets.items():
            gbase = mdir / f"group-{slug(gname)}" / f"group-{slug(gname)}_atlas-{aslug}_model-{tag}"
            ensure_dir(gbase.parent)
            emp_g = emp_all if gname == "all" else group_empirical(members, freq_band, smooth)
            SC_g = SC_group if gname == "all" else prepare_sc(np.mean([s.SC for s in members], 0), {**cfg.get("sc", {}), "symmetrize": False})
            save_matrix(Path(str(gbase) + "_desc-empiricalFC_connectivity.tsv"), emp_g["FC"], names)
            save_matrix(Path(str(gbase) + "_desc-empiricalCOVtau_connectivity.tsv"), emp_g["COVtau"], names)
            entry: dict = {"n": len(members), "members": [s.sub for s in members]}
            if gcfg.get("fit_group_average", True):
                use_lin = model == "nonlinear" and cfg_get(cfg, "model.nonlinear.init", "linear") == "linear" and gname in linear_group_results
                C_init = linear_group_results[gname]["EC"] if use_lin else None
                om_init = linear_group_results[gname].get("omega") if use_lin else None
                res = fit_one(model, emp_g, SC_g, G_group, a, cfg, atlas, C_init=C_init, verbose=verbose, omega_override=om_init)
                res["FC_emp"], res["COVtau_emp"] = emp_g["FC"], emp_g["COVtau"]
                res["group"] = gname
                meta = save_fit(res, gbase, names, SC_g, f"group {gname} {model} Hopf (fit to group-average statistics)")
                entry["fit"] = {k: meta[k] for k in ("metrics", "G_eff", "elapsed_s", "method") if k in meta}
                grows.append({"group": gname, "n": len(members), "model": model, "kind": "fit_to_average", "G_eff": res["G_eff"], **res["metrics"]})
                if model == "linear":
                    linear_group_results[gname] = {"EC": res["EC"], "omega": np.asarray(res["omega_fit"], float) if res.get("omega_fit") else None}
            mem_ecs = [ECs[s.sub] for s in members if s.sub in ECs]
            mem_ecn = [ECns[s.sub] for s in members if s.sub in ECns]
            if gcfg.get("mean_of_participants", True) and mem_ecs:
                meanEC = np.mean(mem_ecs, 0)
                save_matrix(Path(str(gbase) + "_desc-meanEC_connectivity.tsv"), meanEC, names)
                save_matrix(Path(str(gbase) + "_desc-sdEC_connectivity.tsv"), np.std(mem_ecs, 0), names)
                save_matrix(Path(str(gbase) + "_desc-meanECnorm_connectivity.tsv"), np.mean(mem_ecn, 0), names)
                save_matrix(Path(str(gbase) + "_desc-sdECnorm_connectivity.tsv"), np.std(mem_ecn, 0), names)
                try:
                    plot_ec_summary(meanEC, SC_g, Path(str(gbase) + "_desc-meanEC.png"), f"group {gname}: mean of {len(mem_ecs)} participant ECs ({model})", names)
                except Exception as e:  # noqa: BLE001
                    LOG.warning("figure failed: %s", e)
                entry["mean_of_participants"] = {"n": len(mem_ecs)}
                grows.append({"group": gname, "n": len(mem_ecs), "model": model, "kind": "mean_of_participants"})
            summary["groups"].setdefault(model, {})[gname] = entry
        if grows:
            pd.DataFrame(grows).to_csv(mdir / f"groups_atlas-{aslug}_model-{tag}_fit.tsv", sep="\t", index=False, float_format="%.6g")
        # ---- group comparisons (participant-level ECs)
        if gcfg.get("compare", True) and len(groups) >= 2 and ECs:
            from .group import compare_groups

            comp_dir = ensure_dir(mdir / "comparisons")
            mask_any = np.zeros((N, N), bool)
            for s in ok:
                mask_any |= make_mask(s.SC, N, cfg_get(cfg, "model.gec.mask", "sc_plus_homotopic"), atlas.homotopic_pairs())
            src_mats = ECns if compare_on.lower() == "ecnorm" else ECs
            for i, ga in enumerate(groups):
                for gb in groups[i + 1:]:
                    A = [src_mats[s.sub] for s in ok if s.group == ga and s.sub in src_mats]
                    B = [src_mats[s.sub] for s in ok if s.group == gb and s.sub in src_mats]
                    if len(A) < 2 or len(B) < 2:
                        continue
                    c = compare_groups(A, B, mask_any, q=float(gcfg.get("fdr_q", 0.05)), n_perm=int(gcfg.get("n_perm", 0)))
                    cb = comp_dir / f"group-{slug(ga)}_vs_group-{slug(gb)}_atlas-{aslug}_model-{tag}"
                    save_matrix(Path(str(cb) + "_desc-tstat_connectivity.tsv"), c["t"], names)
                    save_matrix(Path(str(cb) + "_desc-pvalue_connectivity.tsv"), c["p"], names)
                    save_matrix(Path(str(cb) + "_desc-sigFDR_connectivity.tsv"), c["sig_fdr"].astype(int), names)
                    save_json(Path(str(cb) + "_desc-comparison.json"), {"compared_matrix": compare_on, **{k: v for k, v in c.items() if k in ("n_a", "n_b", "n_sig_edges", "q", "global", "n_sig_perm_fwe")}})
                    try:
                        plot_group_comparison(c["mean_a"], c["mean_b"], c["t"], c["sig_fdr"], (ga, gb), Path(str(cb) + "_desc-comparison.png"))
                    except Exception as e:  # noqa: BLE001
                        LOG.warning("figure failed: %s", e)
                    summary.setdefault("comparisons", {}).setdefault(model, {})[f"{ga}_vs_{gb}"] = {"compared_matrix": compare_on, "n_sig_edges_fdr": c["n_sig_edges"], "global": c["global"]}
    save_json(out_dir / f"fit_summary_atlas-{aslug}.json", summary)
    return summary


def _safe_emp(files, cfg, N):
    try:
        return load_subject_empirical(files, cfg, N)
    except Exception as e:  # noqa: BLE001
        return e
