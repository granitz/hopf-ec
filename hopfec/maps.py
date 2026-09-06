"""Parcel-wise brain maps (e.g. T1w/T2w myelin, receptor densities) for heterogeneous model parameters.

Sources
-------
* ``file``: a TSV/CSV with one value per parcel (columns ``index``/``name`` + a value column, or a single
  column in atlas label order), or an MNI-space NIfTI that is parcellated with the atlas (mean per parcel)
* ``neuromaps``: an annotation fetched with neuromaps (``source``, ``desc``, ``space``, ``den``/``res``).
  Volumetric (MNI152) annotations are parcellated with the atlas; surface annotations (fsLR/fsaverage)
  need a surface parcellation (``surface_labels``: a CIFTI .dlabel.nii or a pair of GIFTI label files
  whose parcel keys match the atlas labels) - subcortical parcels then get no value (NaN -> 0 after
  z-scoring, i.e. they stay at the homogeneous value).

The heterogeneous bifurcation parameter is a_j = a0 + sum_k beta_k z_kj with z the z-scored maps.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from .atlases import Atlas
from .utils import LOG


@dataclass
class ParcelMap:
    name: str
    values: np.ndarray            # raw parcel values (NaN where undefined)
    z: np.ndarray                 # z-scored (NaN -> 0)
    source: str
    n_missing: int = 0
    meta: dict = field(default_factory=dict)


def _zscore(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    z = np.zeros_like(v)
    if ok.sum() > 1 and np.nanstd(v[ok]) > 0:
        z[ok] = (v[ok] - v[ok].mean()) / v[ok].std()
    return z


def parcellate_volume(img: nib.Nifti1Image, atlas: Atlas, stat: str = "mean") -> np.ndarray:
    """Mean (or median) of a volumetric map inside every atlas parcel (map resampled to the atlas grid)."""
    from nilearn.image import resample_to_img

    atl_img = atlas.load_img()
    if img.shape[:3] != atl_img.shape[:3] or not np.allclose(img.affine, atl_img.affine, atol=1e-3):
        img = resample_to_img(img, atl_img, interpolation="continuous", force_resample=True, copy_header=True)
    data = np.asarray(img.dataobj, float)
    if data.ndim == 4:
        data = data[..., 0]
    lab = atlas.data()
    out = np.full(len(atlas.label_ids), np.nan)
    for k, l in enumerate(atlas.label_ids):
        m = (lab == int(l)) & np.isfinite(data) & (data != 0)
        if m.any():
            out[k] = np.median(data[m]) if stat == "median" else float(data[m].mean())
    return out


def read_parcel_table(path: Path, atlas: Atlas, value_column: str | None = None) -> np.ndarray:
    """One value per parcel from a table: matched by 'index' (label id) or 'name', else positional."""
    sep = "," if path.suffix.lower() == ".csv" else None
    first = path.read_text().strip().splitlines()[0]
    try:  # headerless list of numbers: one value per parcel (first column when several)
        [float(t) for t in re.split(r"[,\t ]+", first.strip())]
        arr = np.atleast_2d(np.genfromtxt(path, delimiter="," if sep else None, dtype=float))
        v = arr[:, 0] if arr.shape[0] >= arr.shape[1] else arr[0]
        if len(v) != atlas.n_parcels:
            raise ValueError(f"{path.name}: {len(v)} values for {atlas.n_parcels} parcels")
        return np.asarray(v, float)
    except ValueError as e:
        if "values for" in str(e):
            raise
    df = pd.read_csv(path, sep=sep, engine="python", na_values=["n/a", "NA", "nan", "NaN", ""])
    df.columns = [str(c) for c in df.columns]
    if df.shape[1] == 1:
        v = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
        if len(v) != atlas.n_parcels:
            raise ValueError(f"{path.name}: {len(v)} values for {atlas.n_parcels} parcels")
        return v
    cols_l = {c.lower(): c for c in df.columns}
    vcol = value_column or next((cols_l[c] for c in ("value", "map", "mean", "z", "score") if c in cols_l), None)
    if vcol is None:
        def _numeric(col: pd.Series) -> bool:  # every entry numeric or missing (n/a, NaN), at least one value
            v = pd.to_numeric(col, errors="coerce")
            return bool(v.notna().any() and (v.notna() | col.isna()).all())

        num = [c for c in df.columns if c.lower() not in ("index", "id", "label", "name", "region", "hemisphere", "network") and _numeric(df[c])]
        if not num:
            raise ValueError(f"{path.name}: no numeric value column found (give value_column)")
        vcol = num[-1]
    out = np.full(atlas.n_parcels, np.nan)
    if "index" in cols_l or "id" in cols_l or "label" in cols_l:
        idc = cols_l.get("index") or cols_l.get("id") or cols_l.get("label")
        lut = dict(zip(pd.to_numeric(df[idc], errors="coerce"), pd.to_numeric(df[vcol], errors="coerce")))
        for k, l in enumerate(atlas.label_ids):
            out[k] = lut.get(int(l), np.nan)
    elif "name" in cols_l:
        lut = dict(zip(df[cols_l["name"]].astype(str), pd.to_numeric(df[vcol], errors="coerce")))
        for k, nm in enumerate(atlas.region_names):
            out[k] = lut.get(nm, np.nan)
    else:
        v = pd.to_numeric(df[vcol], errors="coerce").to_numpy()
        if len(v) != atlas.n_parcels:
            raise ValueError(f"{path.name}: {len(v)} rows for {atlas.n_parcels} parcels")
        out = v
    return out


def _surface_parcellate(annotation, atlas: Atlas, surface_labels, space: str, den: str) -> np.ndarray:
    """Parcellate a surface annotation (neuromaps) with a CIFTI dlabel or GIFTI label files."""
    import neuromaps.images as nmi

    files = surface_labels if isinstance(surface_labels, (list, tuple)) else [surface_labels]
    out = np.full(atlas.n_parcels, np.nan)
    pos = {int(l): k for k, l in enumerate(atlas.label_ids)}
    ann = list(annotation) if isinstance(annotation, (list, tuple)) else [annotation]
    if len(files) == 1 and str(files[0]).endswith(".dlabel.nii"):
        cifti = nib.load(str(files[0]))
        labels = np.asarray(cifti.get_fdata()).ravel()
        bm = cifti.header.get_axis(1)
        hemi_data = {}
        for hemi_img in ann:
            g = nmi.load_gifti(hemi_img) if not hasattr(hemi_img, "agg_data") else hemi_img
            hemi_data.setdefault("L" if len(hemi_data) == 0 else "R", np.asarray(g.agg_data(), float))
        for name, slc, model in bm.iter_structures():
            key = "L" if "LEFT" in str(name) else "R" if "RIGHT" in str(name) else None
            if key is None or key not in hemi_data or not bool(np.all(model.surface_mask)):
                continue  # volume (subcortical) structures of the dlabel have no surface annotation
            vals = hemi_data[key][model.vertex]
            labs = labels[slc]
            for l in np.unique(labs):
                if int(l) in pos:
                    sel = labs == l
                    out[pos[int(l)]] = np.nanmean(vals[sel]) if sel.any() else np.nan
        return out
    # GIFTI label files (L, R)
    for hemi_img, lab_file in zip(ann, files):
        g = nmi.load_gifti(hemi_img) if not hasattr(hemi_img, "agg_data") else hemi_img
        vals = np.asarray(g.agg_data(), float)
        labs = np.asarray(nib.load(str(lab_file)).agg_data()).ravel()
        for l in np.unique(labs):
            if int(l) in pos and l != 0:
                sel = labs == l
                out[pos[int(l)]] = np.nanmean(vals[sel])
    return out


def load_parcel_map(spec: dict, atlas: Atlas) -> ParcelMap:
    """spec: {name, file | neuromaps: {source, desc, space, den|res}, value_column, surface_labels, stat}."""
    name = str(spec.get("name") or spec.get("desc") or (spec.get("neuromaps") or {}).get("desc") or Path(str(spec.get("file", "map"))).stem)
    if spec.get("file"):
        p = Path(str(spec["file"]))
        if not p.exists():
            raise FileNotFoundError(p)
        if p.suffix.lower() in (".tsv", ".csv", ".txt"):
            vals = read_parcel_table(p, atlas, spec.get("value_column"))
            src = f"table:{p.name}"
        else:
            vals = parcellate_volume(nib.load(str(p)), atlas, spec.get("stat", "mean"))
            src = f"volume:{p.name}"
    elif spec.get("neuromaps"):
        try:
            from neuromaps.datasets import fetch_annotation
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install neuromaps to use neuromaps annotations (or give the map as a file)") from e
        nm = dict(spec["neuromaps"])
        ann = fetch_annotation(source=nm.get("source"), desc=nm.get("desc"), space=nm.get("space"), den=nm.get("den"), res=nm.get("res"), return_single=True)
        space = str(nm.get("space", "")).lower()
        if space.startswith("mni"):
            vals = parcellate_volume(nib.load(str(ann)), atlas, spec.get("stat", "mean"))
        else:
            if not spec.get("surface_labels"):
                raise ValueError(f"map {name}: surface annotation ({space}) needs 'surface_labels' (CIFTI dlabel or GIFTI label files of the atlas)")
            vals = _surface_parcellate(ann, atlas, spec["surface_labels"], space, nm.get("den", ""))
        src = f"neuromaps:{nm.get('source')}/{nm.get('desc')}/{nm.get('space')}"
    else:
        raise ValueError(f"map {name}: give 'file' or 'neuromaps'")
    vals = np.asarray(vals, float)
    n_missing = int(np.sum(~np.isfinite(vals)))
    if n_missing:
        LOG.warning("map %s: %d/%d parcels without a value (kept at the homogeneous parameter)", name, n_missing, len(vals))
    return ParcelMap(name=name, values=vals, z=_zscore(vals) if spec.get("zscore", True) else np.nan_to_num(vals), source=src, n_missing=n_missing)


def load_maps(specs: list[dict], atlas: Atlas) -> list[ParcelMap]:
    return [load_parcel_map(sp, atlas) for sp in specs]


def hetero_a(a0: float, betas, maps: list[ParcelMap], clip: tuple[float, float] | None = (-0.9, 0.9)) -> np.ndarray:
    Z = np.stack([m.z for m in maps], axis=1) if maps else np.zeros((0, 0))
    b = np.atleast_1d(np.asarray(betas, float))
    a = a0 + (Z @ b if maps else 0.0)
    if clip is not None:
        a = np.clip(a, clip[0], clip[1])
    return a
