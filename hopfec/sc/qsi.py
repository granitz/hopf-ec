"""QSIPrep / QSIRecon derivatives: discover preprocessed DWI and ready-made connectivity matrices."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..bids import parse_entities
from ..utils import LOG, sub_label


def _is_hdf5(path: Path) -> bool:
    with open(path, "rb") as f:
        head = f.read(128)
    return b"MATLAB 7.3" in head or head[:8] == b"\x89HDF\r\n\x1a\n"


def list_mat_keys(path: str | Path) -> list[str]:
    path = Path(path)
    if _is_hdf5(path):
        import h5py

        with h5py.File(path, "r") as f:
            return [k for k in f.keys() if not k.startswith("#")]
    from scipy.io import whosmat

    return [name for name, _shape, _cls in whosmat(str(path))]


def _read_mat_var(path: Path, key: str):
    if _is_hdf5(path):
        import h5py

        with h5py.File(path, "r") as f:
            d = f[key]
            arr = np.array(d)
            if arr.dtype.kind in ("u", "i") and d.attrs.get("MATLAB_class", b"") in (b"char",):
                return "".join(chr(int(c)) for c in arr.ravel())
            return arr.T if arr.ndim == 2 else arr
    from scipy.io import loadmat

    m = loadmat(str(path), variable_names=[key], squeeze_me=True)
    return m[key]


def select_connectivity_key(keys: list[str], key_regex: str | None = None, atlas_hint: str | None = None) -> str:
    cands = [k for k in keys if "connectivity" in k.lower() or re.search(r"(count|sift|weight|fa|md)", k.lower())]
    cands = [k for k in cands if not re.search(r"(region|label|command|atlas_names)", k.lower())]
    if key_regex:
        rx = re.compile(key_regex, re.I)
        hits = [k for k in keys if rx.search(k)]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            raise KeyError(f"no matrix key matches {key_regex!r}; keys: {keys}")
        cands = hits
    if atlas_hint:
        n = re.sub(r"[^a-z0-9]", "", atlas_hint.lower())
        hits = [k for k in cands if n in re.sub(r"[^a-z0-9]", "", k.lower())]
        if hits:
            cands = hits
    prefs = [r"sift.*count", r"sift.*connectivity", r"radius\d+_count", r"count", r"sift"]
    for p in prefs:
        hits = [k for k in cands if re.search(p, k, re.I) and "length" not in k.lower()]
        if hits:
            return sorted(hits)[0]
    if cands:
        return sorted(cands)[0]
    raise KeyError(f"no connectivity matrix found among keys {keys}")


def load_connectivity_mat(path: str | Path, key: str | None = None, atlas_hint: str | None = None):
    """Load a connectivity matrix from a QSIRecon-style .mat -> (matrix, key, region_labels or None)."""
    path = Path(path)
    keys = list_mat_keys(path)
    chosen = key if (key in keys) else select_connectivity_key(keys, key, atlas_hint)
    M = np.asarray(_read_mat_var(path, chosen), float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"{path}:{chosen} is not a square matrix (shape {M.shape})")
    labels = None
    stem = chosen.split("_connectivity")[0]
    for cand in (f"{stem}_region_labels", "region_labels", f"{stem.split('_')[0]}_region_labels"):
        if cand in keys:
            try:
                lab = _read_mat_var(path, cand)
                labels = [str(x).strip() for x in np.atleast_1d(lab)]
            except Exception:  # noqa: BLE001
                labels = None
            break
    LOG.info("loaded %s [%s] %s", path.name, chosen, M.shape)
    return M, chosen, labels


def find_qsirecon_connectivity(qsirecon_dir: str | Path, sub: str, atlas_hint: str | None = None) -> list[Path]:
    """All connectivity files (.mat/.tsv/.csv) of one subject below a qsirecon* folder."""
    root = Path(qsirecon_dir)
    sub = sub_label(sub)
    files: list[Path] = []
    for pat in ("**/%s/**/*connectivity*.mat", "**/%s/**/*connectivity*.tsv", "**/%s/**/*connectivity*.csv",
                "%s/**/*connectivity*.mat", "%s/**/*connectivity*.tsv", "%s/**/*connectivity*.csv"):
        files += list(root.glob(pat % sub))
    files = sorted(set(files))
    if atlas_hint:
        n = re.sub(r"[^a-z0-9]", "", atlas_hint.lower())
        hits = [f for f in files if n in re.sub(r"[^a-z0-9]", "", f.name.lower()) or f.suffix == ".mat"]
        files = hits or files
    return files


def find_qsiprep_dwi(qsiprep_dir: str | Path, sub: str) -> dict:
    """Preprocessed DWI (+ bval/bvec/mask/T1w/MNI->anat transform) of one subject."""
    root = Path(qsiprep_dir)
    sub = sub_label(sub)
    out: dict = {}
    dwis = sorted(root.glob(f"{sub}/**/dwi/*_desc-preproc_dwi.nii.gz"))
    if not dwis:
        return out
    dwi = dwis[0]
    stem = dwi.name[: -len(".nii.gz")]
    out["dwi"] = dwi
    for ext, k in ((".bval", "bval"), (".bvec", "bvec"), (".b", "mrtrix_grad")):
        c = dwi.with_name(stem + ext)
        if c.exists():
            out[k] = c
    mask = sorted(dwi.parent.glob(f"{sub}*_desc-brain_mask.nii.gz"))
    if mask:
        out["mask"] = mask[0]
    anat = sorted(root.glob(f"{sub}/**/anat/*_desc-preproc_T1w.nii.gz"))
    if anat:
        out["t1w"] = anat[0]
    xfm = sorted(root.glob(f"{sub}/**/anat/*from-MNI152NLin2009cAsym_to-*_mode-image_xfm.h5"))
    if xfm:
        out["xfm_mni_to_anat"] = xfm[0]
    ents = parse_entities(dwi.name)
    out["space"] = ents.get("space")
    return out
