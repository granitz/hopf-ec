"""Configuration: defaults + YAML overrides + portable path resolution.

All paths in the ``paths`` section may contain ``${ENV_VAR}`` (or ``${ENV_VAR:-default}``),
``~`` and may be relative.  Relative paths are resolved against ``paths.root`` if given,
otherwise against the directory that contains the YAML file, so a project can be moved
to another machine/server without editing anything but (optionally) the environment.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from .utils import LOG, expand_env, expand_path

DEFAULTS: dict[str, Any] = {
    "paths": {
        "root": None,                 # project root; relative paths resolve against it
        "bids_dir": None,             # raw BIDS dataset
        "derivatives_dir": None,      # folder holding fmriprep/, halfpipe/, qsiprep/, qsirecon/
        "fmriprep_dir": None,         # override: <derivatives_dir>/fmriprep
        "halfpipe_dir": None,         # override: <derivatives_dir>/halfpipe (HALFpipe working dir or derivatives)
        "qsiprep_dir": None,          # override: <derivatives_dir>/qsiprep
        "qsirecon_dir": None,         # override: <derivatives_dir>/qsirecon*
        "atlas_dir": None,            # folder with parcellation NIfTIs (+ label tables)
        "participants_tsv": None,     # default: <bids_dir>/participants.tsv
        "output_dir": "derivatives/hopfec",
        "work_dir": "work",
        "normative_connectome": None,  # e.g. dTOR_fibers_vox_2_mm.mat(.gz)
        "template_dir": None,         # optional folder with MNI templates (else TemplateFlow / FSL)
        "fs_license": "${FS_LICENSE:-}",
    },
    "participants": {"id_column": None, "group_column": None, "include": None, "exclude": None},
    "input": {
        # auto: timeseries -> halfpipe-timeseries -> halfpipe -> fmriprep
        "source": "auto",
        "task": "rest",
        "session": None,
        "run": None,
        "space": "MNI152NLin2009cAsym",
        "res": "2",
        "tr": None,                   # seconds; overrides sidecars if set
        "halfpipe_feature": None,     # HALFpipe feature name whose atlas time series to use
        "halfpipe_setting": None,     # HALFpipe setting name for preprocessed BOLD
        "timeseries_glob": None,      # explicit glob for ready-made time series TSVs ({sub} placeholder)
    },
    "atlas": {"name": None, "file": None, "labels": None, "allow_space_mismatch": False},
    "fmriprep": {
        "runtime": "auto",            # auto | docker | apptainer | singularity | native
        "image": "nipreps/fmriprep:24.1.1",
        "sif": None,                  # path to fmriprep .sif for apptainer/singularity
        "output_spaces": ["MNI152NLin2009cAsym:res-2"],
        "nprocs": 8,
        "omp_nthreads": 4,
        "mem_mb": 16000,
        "fs_no_reconall": False,
        "skip_bids_validation": True,
        "extra_args": [],
        "templateflow_home": "${TEMPLATEFLOW_HOME:-}",
    },
    "postprocess": {
        "confounds": "acompcor",      # none | 24P | 36P | acompcor | acompcor+gsr | custom
        "custom_regex": [],
        "n_acompcor": 5,
        "acompcor_mask": "combined",  # combined | WM | CSF | separate
        "include_cosine": True,
        "global_signal": False,
        "motion_outliers": False,     # add fMRIPrep motion_outlier_XX spike regressors
        "scrub_fd": None,             # FD threshold (mm) for spike regressors (None = off)
        "band": [0.008, 0.08],        # Hz; null to disable filtering
        "detrend": True,
        "standardize": "zscore_sample",
        "dummy_scans": "auto",        # auto (non_steady_state columns) | int
        "smoothing_fwhm": None,
        "min_volumes": 100,
        "mask_strategy": "brain_mask",  # brain_mask | none
    },
    "sc": {
        "source": "auto",             # auto | file | qsirecon | tractography | normative
        "file": None,                 # single matrix used for all participants ({sub} placeholder allowed)
        "qsirecon_key": None,         # regex to select key inside qsirecon .mat (default heuristic)
        "weight": "combined",         # for normative/tractography: count | volnorm | lencorr | combined
        "normalize": "max",           # max (scale so that max = sc_max) | none
        "sc_max": 0.2,
        "symmetrize": True,
        "log_transform": False,
        "threshold": 0.0,             # zero entries below this fraction of the max after normalisation
        "normative": {
            "fiber_grid": "auto",     # auto | RAS | LAS | custom  (voxel-index convention of fibers_vox)
            "fiber_affine": None,     # custom 4x4 (0-indexed voxel -> mm) when fiber_grid == custom
            "count_mode": "touched",  # touched (every parcel a streamline passes) | endpoints
            "length_mode": "npoints", # npoints (n_points * voxel size, as in the original script) | polyline
            "cache": True,
            "cache_dtype": "float32",  # uint8/int16 write the cache compactly when coordinates are known integers
            "chunk_size": 100000,
            "warp_atlas": "auto",     # auto | never | always : warp atlas into fiber template space with ANTs+TemplateFlow
            "brain_mask": None,       # optional brain mask in the fiber grid for orientation diagnostics
        },
        "tractography": {
            "backend": "auto",        # auto | mrtrix | dipy
            "n_streamlines": 10000000,  # mrtrix -select
            "seeds_per_voxel": 2,     # dipy
            "step_size": 0.5,
            "max_angle": 30.0,
            "fa_threshold": 0.15,
            "min_length": 10.0,
            "max_length": 250.0,
            "sift2": True,
            "act": "auto",
            "endpoints_only": True,
            "execute": False,
        },
    },
    "model": {
        "filter_band": [0.008, 0.08],   # band-pass applied to empirical (and simulated) BOLD
        "freq_band": None,              # band for node peak-frequency estimation (default = filter_band)
        "spectrum_smoothing_hz": 0.01,
        "tau_tr": 1,                    # lag (in TRs) for the time-shifted covariance
        "a": -0.02,                     # bifurcation parameter (scalar or per-node list)
        "beta": 0.02,                   # noise amplitude
        "dt": 0.1,                      # integration step (s), adjusted so TR is an integer number of steps
        "G": None,                      # global coupling; None -> from parameter search
        "transient_s": 100.0,           # discarded initial simulation time (s)
        "min_volumes": 50,
        "search": {
            "enabled": True,
            "G": {"start": 0.0, "stop": 3.0, "step": 0.1},
            "a": None,                  # e.g. {"start": -0.2, "stop": 0.0, "num": 11} for a 2-D surface
            "metric": "fit_rmse",       # fc_corr | fc_rmse | fit_rmse | fcd_ks | combined
            "n_sim": 1,
            "level": "group",           # group | participant | both : whose FC the search is fitted on
            # border handling: an optimum on the edge of the grid extends the grid (same step) on that side
            "border_action": "extend",  # extend | warn | ignore
            "extend_factor": 0.5,       # fraction of the axis span added per extension
            "max_extensions": 2,
            "G_min": 0.0, "G_max": 10.0,    # hard caps for extension / continuous optimisation
            "a_min": -1.0, "a_max": 0.5,
            "allow_zero_G": False,      # G = 0 is evaluated but never selected (uncoupled model)
            "refine": True,             # coarse-to-fine pass (+- one coarse step, refine_points per axis)
            "refine_points": 7,
            "interpolate": True,        # parabolic interpolation through the optimum
            "continuous": False,        # continuous (G, a) optimisation after the grid stages
            "continuous_method": "auto",  # auto = L-BFGS-B with analytic gradient (linear, fit_rmse) else Powell
            "continuous_max_fev": 60,
        },
        "gec": {
            "eps_fc": 0.001,
            "eps_tau": 0.001,
            "max_iter": 2000,
            "min_iter": 20,
            "patience": 50,
            "mask": "sc_plus_homotopic",   # sc | sc_plus_homotopic | full
            "normalize_max": True,
            "init": "sc",                  # sc | zeros | uniform
        },
        "linear": {
            "enabled": True,
            "method": "gradient",       # gradient (exact-gradient L-BFGS, recommended) | gec (heuristic iteration)
            "filter_consistent": True,
            "fit_omega": True,          # co-estimate node frequencies (initialised from spectral peaks)
            "fit_a": "none",            # none | global | node
            "lambda_sc": 0.0,           # L2 pull towards the (scaled) structural prior
            "lambda_l1": 0.0,
            "w_fc": 1.0,
            "w_tau": 1.0,
            "max_iter": 500,
        },
        "nonlinear": {
            "enabled": True,
            "init": "linear",           # linear (start from the linear EC) | sc
            "method": "surrogate",      # surrogate (linear-adjoint gradient of the non-linear residual, accept-if-improving) | gec (heuristic)
            "accept_only_improving": True,  # gec: keep an update only if it lowers the error (common random numbers)
            "step_frac": 0.05,          # surrogate: initial largest coupling change as a fraction of max(C)
            "n_noise_seeds": 3,         # simulations to estimate the noise floor of the fitting error
            "n_sim": 4,
            "eps_fc": 0.0005,
            "eps_tau": 0.0005,
            "max_iter": 300,
            "patience": 20,
            "seed": 0,
            "use_numba": "auto",
        },
        "fcd": {"window_tr": 30, "step_tr": 3},
    },
    "group": {"fit_group_average": True, "mean_of_participants": True, "pooled": True, "min_subjects": 2, "compare": True,
              "compare_on": "ECnorm",    # ECnorm (max-normalised, scale-free) | EC (absolute coupling)
              "fdr_q": 0.05, "n_perm": 0},
    "compute": {"n_jobs": -1, "backend": "loky", "threads_per_job": 1},
    "stages": ["timeseries", "sc", "fit"],
}


def deep_update(base: dict, upd: Mapping | None) -> dict:
    out = copy.deepcopy(base)
    if not upd:
        return out
    for k, v in upd.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def cfg_get(cfg: Mapping, dotted: str, default=None):
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return default if cur is None else cur


def _expand_strings(obj):
    if isinstance(obj, str):
        return expand_env(obj)
    if isinstance(obj, list):
        return [_expand_strings(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand_strings(v) for k, v in obj.items()}
    return obj


# path-like settings outside the paths section that are resolved against paths.root as well
NESTED_PATH_KEYS = ["atlas.file", "atlas.labels", "sc.file", "fmriprep.sif", "fmriprep.templateflow_home",
                    "sc.normative.brain_mask", "sc.normative.wm_probseg", "sc.normative.template_dir"]


def _set_dotted(cfg: dict, dotted: str, value) -> None:
    cur = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def resolve_paths(cfg: dict, base_dir: str | os.PathLike | None) -> dict:
    """Expand env vars and make every entry of ``cfg['paths']`` (and NESTED_PATH_KEYS) absolute."""
    paths = dict(cfg.get("paths") or {})
    root = paths.get("root")
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    if root:
        root_p = expand_path(root, base)
    else:
        root_p = base
    paths["root"] = str(root_p)
    for k, v in list(paths.items()):
        if k == "root" or v in (None, ""):
            if v == "":
                paths[k] = None
            continue
        if not isinstance(v, str):
            continue
        paths[k] = str(expand_path(v, root_p))
    d = paths.get("derivatives_dir")
    if d:
        for key, sub in (("fmriprep_dir", "fmriprep"), ("halfpipe_dir", "halfpipe"), ("qsiprep_dir", "qsiprep")):
            if not paths.get(key) and (Path(d) / sub).exists():
                paths[key] = str(Path(d) / sub)
        if not paths.get("qsirecon_dir"):
            cands = sorted(Path(d).glob("qsirecon*"))
            if cands:
                paths["qsirecon_dir"] = str(cands[0])
    if not paths.get("participants_tsv") and paths.get("bids_dir"):
        cand = Path(paths["bids_dir"]) / "participants.tsv"
        if cand.exists():
            paths["participants_tsv"] = str(cand)
    cfg["paths"] = paths
    for key in NESTED_PATH_KEYS:
        v = cfg_get(cfg, key)
        if isinstance(v, str) and v.strip():
            _set_dotted(cfg, key, str(expand_path(v, root_p)))
    return cfg


def load_config(path: str | os.PathLike | None = None, overrides: Mapping | None = None) -> dict:
    """Load YAML config merged over DEFAULTS; ``overrides`` (nested dict) win over the file."""
    cfg = copy.deepcopy(DEFAULTS)
    base_dir = None
    if path is not None:
        path = Path(path)
        try:
            with open(path) as f:
                user = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            mark = getattr(e, "problem_mark", None)
            where = f" (line {mark.line + 1}, column {mark.column + 1})" if mark is not None else ""
            raise SystemExit(f"{path}: YAML syntax error{where}: {getattr(e, 'problem', e)}.\n"
                             "Hint: environment variables are set in the shell (export HOPFEC_BIDS=/path), not inside the YAML; "
                             "inside the YAML write `bids_dir: /path` or `bids_dir: ${HOPFEC_BIDS}`.") from None
        if not isinstance(user, dict):
            raise SystemExit(f"config {path} must be a mapping at top level (key: value pairs)")
        cfg = deep_update(cfg, user)
        base_dir = path.parent.resolve()
        cfg["_config_file"] = str(path.resolve())
    if overrides:
        cfg = deep_update(cfg, overrides)
    cfg = _expand_strings(cfg)
    cfg = resolve_paths(cfg, base_dir)
    return cfg


def dump_config(cfg: Mapping, path: str | os.PathLike) -> None:
    with open(path, "w") as f:
        yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, f, sort_keys=False)


def write_example_config(path: str | os.PathLike) -> None:
    """Write a fully commented example configuration."""
    with open(path, "w") as f:
        f.write(EXAMPLE_YAML)


EXAMPLE_YAML = """\
# hopfec configuration.  All paths may use ${ENV_VAR}, ${ENV_VAR:-default} and ~ ;
# relative paths are resolved against `paths.root` (default: the folder of this file),
# so the whole project can be moved between machines/servers.
paths:
  root: .
  bids_dir: ${HOPFEC_BIDS:-./bids}
  derivatives_dir: ${HOPFEC_DERIV:-./bids/derivatives}   # holds fmriprep/, halfpipe/, qsiprep/, qsirecon*/
  atlas_dir: ./atlases
  participants_tsv: null            # default <bids_dir>/participants.tsv (columns: participant_id, group)
  output_dir: ./derivatives/hopfec
  work_dir: ./work
  normative_connectome: ${DTOR_MAT:-}   # dTOR_fibers_vox_2_mm.mat(.gz) for template-based SC
  fs_license: ${FS_LICENSE:-}

participants:
  id_column: null                   # auto: participant_id / subject / sub
  group_column: null                # auto: group / diagnosis / dx (2nd column otherwise)

input:
  source: auto                      # auto | fmriprep | halfpipe | halfpipe-timeseries | timeseries
  task: rest
  space: MNI152NLin2009cAsym
  res: "2"
  tr: null                          # seconds; taken from sidecars if null
  halfpipe_feature: null            # e.g. corrMatrix -> uses *_feature-corrMatrix_atlas-*_timeseries.tsv

atlas:
  name: Schaefer2018_100Parcels7Networks   # substring/regex matched against files in atlas_dir
  file: null                               # or an explicit path
  labels: null                             # explicit label table (tsv/txt) if not auto-detected

fmriprep:
  runtime: auto                     # docker | apptainer | singularity | native
  image: nipreps/fmriprep:24.1.1
  sif: null
  output_spaces: [MNI152NLin2009cAsym:res-2]
  nprocs: 8
  mem_mb: 16000
  fs_no_reconall: false

postprocess:
  confounds: acompcor               # none | 24P | 36P | acompcor | acompcor+gsr
  n_acompcor: 5
  include_cosine: true
  band: [0.008, 0.08]
  dummy_scans: auto
  scrub_fd: null

sc:
  source: auto                      # auto | file | qsirecon | tractography | normative
  file: null
  weight: combined                  # normative/tractography weighting: count | volnorm | lencorr | combined
  normalize: max
  sc_max: 0.2
  symmetrize: true
  normative:
    fiber_grid: auto                # auto | RAS | LAS  (see docs/normative_sc.md)
    count_mode: touched             # touched | endpoints
    cache: true
  tractography:
    backend: auto                   # mrtrix (preferred) | dipy
    n_streamlines: 10000000
    execute: false                  # write the MRtrix script only, unless true

model:
  filter_band: [0.008, 0.08]
  tau_tr: 1
  a: -0.02
  beta: 0.02
  dt: 0.1
  G: null                           # fixed global coupling; null -> take the best value from the search
  search:
    enabled: true
    G: {start: 0.0, stop: 3.0, step: 0.1}
    a: null                         # {start: -0.2, stop: 0.0, num: 11} -> 2-D error surface
    metric: fit_rmse
    level: group                    # group | participant | both
    border_action: extend           # optimum on a grid edge: extend (same step, up to max_extensions) | warn | ignore
    G_max: 10.0                     # caps for the extension (G_min 0; a_min -1.0, a_max 0.5)
    refine: true                    # coarse-to-fine pass around the optimum, then parabolic interpolation
    interpolate: true
    continuous: false               # true: continuous (G, a) optimisation (analytic gradient for the linear model)
  gec:
    eps_fc: 0.001
    eps_tau: 0.001
    max_iter: 2000
    patience: 50
    mask: sc_plus_homotopic
  linear:
    enabled: true
    method: gradient                # gradient (exact gradient, L-BFGS-B) | gec (heuristic GEC iteration)
    filter_consistent: true
    fit_omega: true                 # co-estimate node frequencies (recommended)
    fit_a: none                     # none | global | node
    lambda_sc: 0.0                  # L2 pull towards the structural prior
  nonlinear:
    enabled: true
    init: linear                    # initialise from the linear EC (or: sc)
    method: surrogate               # surrogate (recommended) | gec (heuristic, accept-if-improving)
    n_sim: 4                        # raise if the fit reports improvement_below_noise
    eps_fc: 0.0005
    eps_tau: 0.0005
    max_iter: 300
    patience: 20

group:
  fit_group_average: true
  mean_of_participants: true
  compare: true
  compare_on: ECnorm                # edge-wise group tests on the max-normalised EC (or: EC)
  fdr_q: 0.05
  n_perm: 0                         # >0 adds a permutation (max-statistic FWE) test

compute:
  n_jobs: -1                        # joblib workers (-1 = all cores)
  threads_per_job: 1

stages: [timeseries, sc, fit]
"""
