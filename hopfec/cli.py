"""Command-line interface: ``hopfec <command> [options]``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .config import cfg_get, load_config, write_example_config
from .utils import LOG, setup_logging


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", help="YAML configuration file")
    p.add_argument("-v", "--verbose", action="count", default=1, help="-v info (default), -vv debug")
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--n-jobs", type=int, help="parallel workers (overrides compute.n_jobs)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value, e.g. --set model.a=-0.05 (dotted keys, YAML values)")
    p.add_argument("--participant-label", nargs="*", help="restrict to these participants (sub-XX or XX)")
    p.add_argument("--output-dir", help="override paths.output_dir")


def _overrides(args) -> dict:
    import yaml

    ov: dict = {}
    for item in args.set or []:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        cur = ov
        parts = k.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = yaml.safe_load(v)
    if getattr(args, "output_dir", None):
        ov.setdefault("paths", {})["output_dir"] = args.output_dir
    if getattr(args, "n_jobs", None) is not None:
        ov.setdefault("compute", {})["n_jobs"] = args.n_jobs
    return ov


def _setup(args, need_config: bool = True) -> dict:
    setup_logging(0 if args.quiet else args.verbose)
    if need_config and not args.config:
        raise SystemExit("a configuration file is required (-c config.yaml); create one with `hopfec init-config`")
    cfg = load_config(args.config, _overrides(args))
    return cfg


def _n_jobs(cfg: dict) -> int:
    n = int(cfg_get(cfg, "compute.n_jobs", 1) or 1)
    return n


# --------------------------------------------------------------------------- commands
def cmd_init_config(args):
    path = Path(args.path)
    if path.exists() and not args.force:
        raise SystemExit(f"{path} exists (use --force)")
    write_example_config(path)
    print(f"wrote {path}")


def cmd_atlases(args):
    from .atlases import atlases_table, scan_atlas_dir

    setup_logging(0 if args.quiet else args.verbose)
    d = args.atlas_dir
    if d is None:
        cfg = _setup(args)
        d = cfg_get(cfg, "paths.atlas_dir")
    if not d:
        raise SystemExit("give an atlas folder or set paths.atlas_dir in the config")
    atlases = scan_atlas_dir(d, args.pattern)
    if args.json:
        print(json.dumps([a.summary() for a in atlases], indent=2))
        return
    if not atlases:
        print(f"no atlases found in {d}")
        return
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(atlases_table(atlases).to_string(index=False))
    print(f"\n{len(atlases)} atlas(es) in {d}.  Select one with atlas.name (substring/regex) or atlas.file in the config.")


def cmd_participants(args):
    from .bids import find_subjects, read_participants, validate_participants

    setup_logging(0 if args.quiet else args.verbose)
    cfg = _setup(args, need_config=False) if args.config else None
    tsv = args.participants_tsv or (cfg_get(cfg, "paths.participants_tsv") if cfg else None)
    bids = args.bids_dir or (cfg_get(cfg, "paths.bids_dir") if cfg else None)
    if args.path:  # positional: a BIDS folder (uses its participants.tsv) or a table file
        p = Path(args.path)
        if p.is_dir():
            bids = bids or str(p)
            tsv = tsv or str(p / "participants.tsv")
        else:
            tsv = str(p)
    if not tsv:
        raise SystemExit("give a BIDS folder or table: hopfec participants <bids_dir|participants.tsv>, or --participants-tsv / a config")
    if not Path(tsv).exists():
        raise SystemExit(f"participants table not found: {tsv}")
    df = read_participants(tsv, args.id_column, args.group_column)
    print(df.to_string(index=False))
    if bids and Path(bids).exists():
        v = validate_participants(df, find_subjects(bids))
        print(json.dumps(v, indent=2))
    else:
        print(json.dumps({"groups": df.groupby("group")["participant_id"].count().to_dict()}, indent=2))


def cmd_inputs(args):
    from .stages import load_participants, summarize_inputs

    cfg = _setup(args)
    df = load_participants(cfg, args.participant_label)
    print(json.dumps(summarize_inputs(cfg, df), indent=2, default=str))


def cmd_fmriprep(args):
    from .preprocess.fmriprep import run_fmriprep
    from .stages import load_participants

    cfg = _setup(args)
    bids = cfg_get(cfg, "paths.bids_dir")
    if not bids:
        raise SystemExit("paths.bids_dir is required")
    out = cfg_get(cfg, "paths.fmriprep_dir") or str(Path(cfg_get(cfg, "paths.derivatives_dir") or Path(bids) / "derivatives") / "fmriprep")
    subs = None
    if args.participant_label or cfg_get(cfg, "participants.include"):
        subs = load_participants(cfg, args.participant_label)["participant_id"].tolist()
    rc = run_fmriprep(bids, out, subs, cfg.get("fmriprep", {}), work_dir=Path(cfg["paths"]["work_dir"]) / "fmriprep",
                      fs_license=cfg_get(cfg, "paths.fs_license") or None, runtime=args.runtime, dry_run=args.dry_run)
    sys.exit(rc)


def cmd_timeseries(args):
    from .stages import get_atlas, load_participants, run_timeseries_stage

    cfg = _setup(args)
    df = load_participants(cfg, args.participant_label)
    atlas = get_atlas(cfg)
    res = run_timeseries_stage(cfg, df, atlas, _n_jobs(cfg))
    n_ok = sum(1 for r in res if r.get("ok"))
    print(f"time series written for {n_ok}/{len(res)} runs -> {cfg['paths']['output_dir']}")


def cmd_sc(args):
    from .stages import get_atlas, load_participants, run_normative_stage, run_qsirecon_stage, run_tractography_stage

    cfg = _setup(args, need_config=False)   # everything can be given with --set / --connectome
    if args.kind != "orientation" and not (cfg_get(cfg, "atlas.file") or cfg_get(cfg, "paths.atlas_dir")):
        raise SystemExit("an atlas is required: -c config.yaml, or --set atlas.file=... (and atlas.labels=...)")
    if args.kind == "orientation":
        from .sc.normative import run_orientation_check

        conn = args.connectome or cfg_get(cfg, "paths.normative_connectome")
        if not conn:
            raise SystemExit("give --connectome FILE (dTOR_fibers_vox_2_mm.mat[.gz]) or set paths.normative_connectome")
        work = Path(cfg_get(cfg, "paths.work_dir") or "work") / "normative"
        ncfg = dict(cfg_get(cfg, "sc.normative", {}))
        ncfg.setdefault("template_dir", cfg_get(cfg, "paths.template_dir"))
        d = run_orientation_check(conn, work, ncfg, out_json=work / "orientation_check.json")
        print(json.dumps(d, indent=2, default=str))
        print(f"\n=> best convention: {d['best']}  (confident: {d['confident']}, margin in density-WM correlation {d['margin_wm_correlation']:.3f})")
        return
    atlas = get_atlas(cfg)
    if args.kind == "normative":
        if args.connectome:
            cfg["paths"]["normative_connectome"] = args.connectome
        if args.fiber_grid:
            cfg["sc"]["normative"]["fiber_grid"] = args.fiber_grid
        s = run_normative_stage(cfg, atlas)
        print(json.dumps({k: s[k] for k in s if k not in ("fiber_affine", "fiber_to_atlas_matrix", "orientation_diagnostics")}, indent=2, default=str))
        if s.get("orientation_diagnostics"):
            print("orientation diagnostics:", json.dumps(s["orientation_diagnostics"]["candidates"], indent=2))
        return
    df = load_participants(cfg, args.participant_label)
    if args.kind == "tractography":
        res = run_tractography_stage(cfg, df, atlas, execute=True if args.execute else None, n_jobs=_n_jobs(cfg))
    elif args.kind == "qsirecon":
        res = run_qsirecon_stage(cfg, df, atlas)
    else:
        raise SystemExit(args.kind)
    print(json.dumps(res, indent=2, default=str))


def cmd_ndte(args):
    from .stages import get_atlas, load_participants, run_ndte_stage

    cfg = _setup(args)
    df = load_participants(cfg, args.participant_label)
    atlas = get_atlas(cfg)
    if args.n_surrogates is not None:
        cfg["model"]["ndte"]["n_surrogates"] = args.n_surrogates
    if args.max_lag is not None:
        cfg["model"]["ndte"]["max_lag"] = args.max_lag
    print(json.dumps(run_ndte_stage(cfg, df, atlas, _n_jobs(cfg)), indent=2, default=str))


def cmd_fit(args):
    from .pipeline import run_fit_stage
    from .stages import get_atlas, load_participants

    cfg = _setup(args)
    df = load_participants(cfg, args.participant_label)
    atlas = get_atlas(cfg)
    models = None if args.model in (None, "both") else [args.model]
    if args.method:
        cfg["model"]["linear"]["method"] = args.method
    summary = run_fit_stage(cfg, df, atlas, models=models, n_jobs=_n_jobs(cfg), participants_only=args.participants_only,
                            group_only=args.group_only, verbose=args.verbose - 1)
    print(json.dumps({k: summary[k] for k in ("atlas", "n_parcels", "models", "search", "groups", "comparisons") if k in summary}, indent=2, default=str))
    bad = {s: v["errors"] for s, v in summary["participants"].items() if v["errors"]}
    if bad:
        print("participants with problems:", json.dumps(bad, indent=2))


def cmd_run(args):
    from .pipeline import run_fit_stage
    from .stages import get_atlas, load_participants, run_normative_stage, run_timeseries_stage, run_tractography_stage

    cfg = _setup(args)
    stages = args.stages or cfg.get("stages") or ["timeseries", "sc", "fit"]
    df = load_participants(cfg, args.participant_label)
    atlas = get_atlas(cfg)
    n_jobs = _n_jobs(cfg)
    for st in stages:
        LOG.info("=== stage %s ===", st)
        if st == "fmriprep":
            from .preprocess.fmriprep import run_fmriprep

            run_fmriprep(cfg["paths"]["bids_dir"], cfg_get(cfg, "paths.fmriprep_dir"), df["participant_id"].tolist(), cfg.get("fmriprep", {}),
                         work_dir=Path(cfg["paths"]["work_dir"]) / "fmriprep", fs_license=cfg_get(cfg, "paths.fs_license") or None)
        elif st == "timeseries":
            run_timeseries_stage(cfg, df, atlas, n_jobs)
        elif st == "sc":
            src = cfg_get(cfg, "sc.source", "auto")
            if src in ("normative", "auto") and cfg_get(cfg, "paths.normative_connectome"):
                run_normative_stage(cfg, atlas)
            elif src == "tractography":
                run_tractography_stage(cfg, df, atlas, n_jobs=n_jobs)
            else:
                LOG.info("sc stage: nothing to build (source=%s); SC will be read at fit time", src)
        elif st == "fit":
            run_fit_stage(cfg, df, atlas, n_jobs=n_jobs)
        elif st == "ndte":
            from .stages import run_ndte_stage

            run_ndte_stage(cfg, df, atlas, n_jobs)
        else:
            raise SystemExit(f"unknown stage {st}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hopfec", description="Whole-brain effective connectivity with linear and non-linear Hopf models.")
    p.add_argument("--version", action="version", version=f"hopfec {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init-config", help="write a commented example configuration")
    s.add_argument("path", nargs="?", default="hopfec.yaml")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_init_config)

    s = sub.add_parser("atlases", help="list the parcellation atlases available in a folder")
    s.add_argument("atlas_dir", nargs="?", help="folder with atlas NIfTIs (default: paths.atlas_dir of the config)")
    s.add_argument("--pattern", help="regex filter on file names")
    s.add_argument("--json", action="store_true")
    _add_common(s)
    s.set_defaults(func=cmd_atlases)

    s = sub.add_parser("participants", help="read/validate the participants table (id + group)")
    s.add_argument("path", nargs="?", help="BIDS folder (its participants.tsv is used) or a participants table (.tsv/.csv)")
    s.add_argument("--participants-tsv")
    s.add_argument("--bids-dir")
    s.add_argument("--id-column")
    s.add_argument("--group-column")
    _add_common(s)
    s.set_defaults(func=cmd_participants)

    s = sub.add_parser("inputs", help="summarise discovered inputs (BOLD runs, time series, SC) per participant")
    _add_common(s)
    s.set_defaults(func=cmd_inputs)

    s = sub.add_parser("fmriprep", help="run (or print) the fMRIPrep command for the BIDS dataset")
    s.add_argument("--dry-run", action="store_true", help="print the command only")
    s.add_argument("--runtime", choices=["auto", "docker", "apptainer", "singularity", "native", "fmriprep-docker"])
    _add_common(s)
    s.set_defaults(func=cmd_fmriprep)

    s = sub.add_parser("timeseries", help="post-process preprocessed BOLD (fMRIPrep/HALFpipe) and parcellate with the atlas")
    _add_common(s)
    s.set_defaults(func=cmd_timeseries)

    s = sub.add_parser("sc", help="structural connectivity: normative (dTOR-985), tractography (MRtrix3/DIPY) or qsirecon")
    s.add_argument("kind", choices=["normative", "tractography", "qsirecon", "orientation"],
                   help="orientation = RAS/LAS check of a fibers_vox file only (no atlas needed)")
    s.add_argument("--connectome", help="dTOR_fibers_vox_2_mm.mat(.gz) (normative)")
    s.add_argument("--fiber-grid", choices=["auto", "RAS", "LAS"], help="voxel-index convention of fibers_vox (normative)")
    s.add_argument("--execute", action="store_true", help="run the tractography (else only write scripts)")
    _add_common(s)
    s.set_defaults(func=cmd_sc)

    s = sub.add_parser("ndte", help="normalised directed transfer entropy (Deco et al. 2021) per participant with surrogates + group averages")
    s.add_argument("--n-surrogates", type=int)
    s.add_argument("--max-lag", type=int)
    _add_common(s)
    s.set_defaults(func=cmd_ndte)

    s = sub.add_parser("fit", help="fit effective connectivity (participant + group level, parameter search)")
    s.add_argument("--model", choices=["linear", "nonlinear", "both"], default=None)
    s.add_argument("--method", choices=["gradient", "gec"], help="linear model estimator")
    s.add_argument("--participants-only", action="store_true", help="skip group-level fits (e.g. per-subject cluster jobs)")
    s.add_argument("--group-only", action="store_true", help="only group-level outputs from existing participant fits")
    _add_common(s)
    s.set_defaults(func=cmd_fit)

    s = sub.add_parser("run", help="run the stages listed in the config (timeseries, sc, fit)")
    s.add_argument("--stages", nargs="*", choices=["fmriprep", "timeseries", "sc", "fit", "ndte"])
    _add_common(s)
    s.set_defaults(func=cmd_run)
    return p


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
