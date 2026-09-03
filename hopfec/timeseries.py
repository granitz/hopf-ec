"""BOLD post-processing and parcellation -> region time series tables.

Pipeline per run (fMRIPrep-style inputs):
  1. drop non-steady-state volumes (dummy scans)
  2. parcellate: mean signal per atlas label (atlas resampled to the BOLD grid, nearest neighbour)
  3. nuisance regression with a chosen fMRIPrep confound set (24P, 36P, aCompCor [+GSR], custom)
     with the band-pass filter applied to data AND regressors before regression (Lindquist 2019)
  4. standardisation
Region-level and voxel-level regression are equivalent for OLS with identical regressors, so the
cheap order (parcellate first) is used.  HALFpipe *setting* outputs are already denoised and are
only parcellated (and optionally filtered).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img

from .atlases import Atlas
from .bids import build_name
from .inputs.common import BoldRun
from .models.signal import bandpass
from .utils import LOG, ensure_dir, save_json

MOTION6 = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]


def _expand(cols: list[str], derivatives=True, powers=True) -> list[str]:
    out = list(cols)
    if derivatives:
        out += [f"{c}_derivative1" for c in cols]
    if powers:
        out += [f"{c}_power2" for c in cols]
    if derivatives and powers:
        out += [f"{c}_derivative1_power2" for c in cols]
    return out


def select_confounds(df: pd.DataFrame, meta: dict | None, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Pick regressors from an fMRIPrep confounds table according to cfg['confounds'] strategy."""
    strategy = str(cfg.get("confounds", "acompcor")).lower()
    cols: list[str] = []
    info: dict = {"strategy": strategy}
    if strategy in ("none", "off", ""):
        return pd.DataFrame(index=df.index), info
    base = strategy.replace("+gsr", "")
    gsr = "+gsr" in strategy or bool(cfg.get("global_signal"))
    if base in ("24p", "acompcor", "36p"):
        cols += _expand(MOTION6)
    if base == "36p":
        cols += _expand(["csf", "white_matter", "global_signal"])
        gsr = False
    if base == "acompcor":
        n = int(cfg.get("n_acompcor", 5))
        acc = [c for c in df.columns if re.fullmatch(r"a_comp_cor_\d+", c)]
        chosen: list[str] = []
        which_mask = str(cfg.get("acompcor_mask", "combined"))
        if meta and acc:
            by_mask: dict[str, list[tuple[float, str]]] = {}
            for c in acc:
                m = meta.get(c, {})
                if not m.get("Retained", True):
                    continue
                by_mask.setdefault(str(m.get("Mask", "unknown")), []).append((-float(m.get("VarianceExplained", 0.0)), c))
            if which_mask == "separate":
                for mk in ("CSF", "WM"):
                    chosen += [c for _, c in sorted(by_mask.get(mk, []))[:n]]
            else:
                key = which_mask if which_mask in by_mask else ("combined" if "combined" in by_mask else next(iter(by_mask), None))
                if key:
                    chosen = [c for _, c in sorted(by_mask[key])[:n]]
                    info["acompcor_mask_used"] = key
        if not chosen:
            chosen = acc[:n]
        cols += chosen
        info["n_acompcor"] = len(chosen)
        if cfg.get("include_cosine", True):
            cols += [c for c in df.columns if re.fullmatch(r"cosine\d+", c)]
    if gsr:
        cols += _expand(["global_signal"]) if base == "24p" else ["global_signal"]
    for rx in cfg.get("custom_regex") or []:
        cols += [c for c in df.columns if re.fullmatch(rx, c)]
    if cfg.get("motion_outliers"):
        cols += [c for c in df.columns if re.fullmatch(r"motion_outlier\d+", c)]
    cols = [c for i, c in enumerate(cols) if c in df.columns and c not in cols[:i]]
    missing = [c for c in _expand(MOTION6) if c not in df.columns] if base in ("24p", "36p", "acompcor") else []
    if missing:
        LOG.warning("confounds table lacks %d expected columns (e.g. %s)", len(missing), missing[:3])
    X = df[cols].copy()
    X = X.fillna(0.0)  # derivatives have NaN in the first row
    info["n_regressors"] = int(X.shape[1])
    info["regressors"] = cols
    return X, info


def spike_regressors(df: pd.DataFrame, fd_threshold: float | None) -> tuple[np.ndarray | None, dict]:
    if not fd_threshold or "framewise_displacement" not in df.columns:
        return None, {}
    fd = pd.to_numeric(df["framewise_displacement"], errors="coerce").fillna(0.0).to_numpy()
    bad = np.where(fd > float(fd_threshold))[0]
    if len(bad) == 0:
        return None, {"n_spikes": 0}
    S = np.zeros((len(fd), len(bad)))
    S[bad, np.arange(len(bad))] = 1.0
    return S, {"n_spikes": int(len(bad)), "spike_fraction": float(len(bad) / len(fd))}


def n_dummy_scans(df: pd.DataFrame | None, cfg: dict) -> int:
    d = cfg.get("dummy_scans", "auto")
    if isinstance(d, (int, np.integer)) and not isinstance(d, bool):
        return int(d)
    if isinstance(d, str) and d.isdigit():
        return int(d)
    if df is None:
        return 0
    return int(sum(1 for c in df.columns if re.fullmatch(r"non_steady_state_outlier\d+", c)))


def parcellate(bold_img: nib.Nifti1Image, atlas: Atlas, mask_img: nib.Nifti1Image | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Mean time series per atlas label (T x N, columns in atlas.label_ids order) + voxel counts."""
    atl_img = atlas.load_img()
    if atl_img.shape[:3] != bold_img.shape[:3] or not np.allclose(atl_img.affine, bold_img.affine, atol=1e-3):
        atl_img = resample_to_img(atl_img, bold_img, interpolation="nearest", force_resample=True, copy_header=True)
    lab = np.rint(np.asanyarray(atl_img.dataobj)).astype(np.int64)
    lab[lab < 0] = 0
    if mask_img is not None:
        m = np.asanyarray(resample_to_img(mask_img, bold_img, interpolation="nearest", force_resample=True, copy_header=True).dataobj) > 0.5
        lab = lab * m
    data = np.asanyarray(bold_img.dataobj)
    T = data.shape[3]
    flat_lab = lab.ravel()
    sel = flat_lab > 0
    ids = atlas.label_ids
    pos = {int(l): k for k, l in enumerate(ids)}
    idx = np.array([pos.get(int(l), -1) for l in flat_lab[sel]])
    ok = idx >= 0
    idx = idx[ok]
    vox = np.where(sel)[0][ok]
    counts = np.bincount(idx, minlength=len(ids)).astype(float)
    ts = np.zeros((T, len(ids)))
    D = data.reshape(-1, T)[vox].astype(np.float64)  # (n_vox, T)
    nan_vox = ~np.isfinite(D).all(axis=1)
    if nan_vox.any():
        D[nan_vox] = 0.0
    for t in range(T):
        ts[t] = np.bincount(idx, weights=D[:, t], minlength=len(ids))
    with np.errstate(invalid="ignore", divide="ignore"):
        ts = ts / counts[None, :]
    ts[:, counts == 0] = np.nan
    return ts, counts


def clean_timeseries(ts: np.ndarray, tr: float, confounds: np.ndarray | None, band, detrend: bool = True,
                     standardize: str | bool = "zscore_sample", spikes: np.ndarray | None = None) -> np.ndarray:
    """Detrend, regress confounds (filtered like the data) and band-pass; then standardise."""
    x = np.array(ts, float)
    T = x.shape[0]
    x = np.nan_to_num(x)
    t = np.arange(T)
    # detrend (constant + linear) data and confounds
    Xd = np.column_stack([np.ones(T), t - t.mean()]) if detrend else np.ones((T, 1))
    beta_d = np.linalg.lstsq(Xd, x, rcond=None)[0]
    x = x - Xd @ beta_d
    regs = []
    if confounds is not None and confounds.shape[1] > 0:
        cf = np.array(confounds, float)
        cf = cf - np.linalg.lstsq(Xd, cf, rcond=None)[0].__rmatmul__(Xd) if False else cf - Xd @ np.linalg.lstsq(Xd, cf, rcond=None)[0]
        regs.append(cf)
    if band is not None:
        x = bandpass(x, tr, band)
        regs = [bandpass(r, tr, band) for r in regs]
    if spikes is not None:
        regs.append(np.asarray(spikes, float))
    if regs:
        R = np.column_stack(regs)
        keep = R.std(axis=0) > 0
        R = R[:, keep]
        if R.shape[1]:
            beta = np.linalg.lstsq(R, x, rcond=None)[0]
            x = x - R @ beta
    if standardize in ("zscore", "zscore_sample", True):
        sd = x.std(axis=0, ddof=1 if standardize == "zscore_sample" else 0)
        sd[sd == 0] = 1.0
        x = (x - x.mean(axis=0)) / sd
    elif standardize == "psc":
        mu = np.abs(ts).mean(axis=0)
        mu[mu == 0] = 1.0
        x = 100.0 * x / mu
    return x


def extract_run_timeseries(run: BoldRun, atlas: Atlas, cfg: dict, tr: float | None = None) -> tuple[np.ndarray, dict]:
    """Full post-processing of one run -> (T x N array, metadata dict)."""
    tr = tr or run.tr
    if tr is None:
        raise ValueError(f"{run.tag()}: repetition time unknown (set input.tr or provide a JSON sidecar)")
    bold = nib.load(str(run.bold))
    if len(bold.shape) != 4:
        raise ValueError(f"{run.bold} is not 4-D")
    conf_df, meta = None, None
    if run.confounds and not run.postprocessed:
        conf_df = pd.read_csv(run.confounds, sep="\t")
        if run.confounds_json:
            with open(run.confounds_json) as f:
                meta = json.load(f)
    n_dummy = n_dummy_scans(conf_df, cfg)
    if atlas.space and run.space and atlas.space.lower() != run.space.lower() and not cfg.get("allow_space_mismatch", False):
        raise ValueError(f"{run.tag()}: BOLD space {run.space} != atlas space {atlas.space} (set atlas.allow_space_mismatch to override)")
    mask_img = nib.load(str(run.mask)) if (run.mask and cfg.get("mask_strategy", "brain_mask") == "brain_mask") else None
    ts, counts = parcellate(bold, atlas, mask_img)
    info = {"source": str(run.bold), "source_type": run.source, "tr": float(tr), "n_volumes_raw": int(ts.shape[0]),
            "n_dummy": int(n_dummy), "atlas": atlas.name, "atlas_file": str(atlas.image), "n_parcels": int(len(atlas.label_ids)),
            "empty_parcels": [str(atlas.region_names[k]) for k in np.where(counts == 0)[0]],
            "voxels_per_parcel": counts.astype(int).tolist(), "postprocessed_input": bool(run.postprocessed)}
    ts = ts[n_dummy:]
    X, spikes = None, None
    if conf_df is not None:
        Xdf, cinfo = select_confounds(conf_df.iloc[n_dummy:].reset_index(drop=True), meta, cfg)
        info.update(cinfo)
        X = Xdf.to_numpy(float) if Xdf.shape[1] else None
        spikes, sinfo = spike_regressors(conf_df.iloc[n_dummy:].reset_index(drop=True), cfg.get("scrub_fd"))
        info.update(sinfo)
        if "framewise_displacement" in conf_df.columns:
            fd = pd.to_numeric(conf_df["framewise_displacement"].iloc[n_dummy:], errors="coerce")
            info["mean_fd"] = float(np.nanmean(fd))
    band = cfg.get("band", [0.008, 0.08])
    if run.postprocessed and not cfg.get("refilter_postprocessed", False):
        band_used = None
    else:
        band_used = band
    ts_clean = clean_timeseries(ts, tr, X, band_used, detrend=bool(cfg.get("detrend", True)),
                                standardize=cfg.get("standardize", "zscore_sample"), spikes=spikes)
    info["band"] = band_used
    info["n_volumes"] = int(ts_clean.shape[0])
    if ts_clean.shape[0] < int(cfg.get("min_volumes", 0) or 0):
        LOG.warning("%s: only %d volumes after dummy removal (< min_volumes)", run.tag(), ts_clean.shape[0])
    return ts_clean, info


def timeseries_output_paths(output_dir: Path, run_ents: dict, sub: str, atlas: Atlas) -> tuple[Path, Path]:
    ents = {"sub": sub.replace("sub-", ""), "ses": run_ents.get("ses"), "task": run_ents.get("task"), "run": run_ents.get("run"),
            "atlas": re.sub(r"[^A-Za-z0-9]", "", atlas.name)[:60], "desc": "clean"}
    d = ensure_dir(output_dir / sub / (f"ses-{ents['ses']}" if ents.get("ses") else "") / "func")
    tsv = d / build_name(ents, "timeseries", ".tsv")
    return tsv, tsv.with_suffix(".json")


def save_timeseries(ts: np.ndarray, names: list[str], tsv: Path, meta: dict) -> Path:
    ensure_dir(tsv.parent)
    pd.DataFrame(ts, columns=names).to_csv(tsv, sep="\t", index=False, float_format="%.6g")
    save_json(tsv.with_suffix(".json"), meta)
    return tsv


def load_timeseries(path: str | Path, n_expected: int | None = None) -> tuple[np.ndarray, list[str] | None]:
    """Load a time-series table (with or without header); returns (T x N, names or None)."""
    path = Path(path)
    with open(path) as f:
        first = f.readline()
    tokens = re.split(r"[\t,]", first.strip())
    header = False
    for t in tokens:
        try:
            float(t)
        except ValueError:
            header = True
            break
    sep = "," if path.suffix.lower() == ".csv" else "\t"
    df = pd.read_csv(path, sep=sep, header=0 if header else None)
    names = [str(c) for c in df.columns] if header else None
    X = df.to_numpy(float)
    if n_expected is not None and X.shape[1] != n_expected and X.shape[0] == n_expected:
        X = X.T
    return X, names
