"""ROI-to-ROI structural connectivity from the dTOR-985 normative connectome (``dTOR_fibers_vox_2_mm.mat``).

Python port of the user's MATLAB ``build_roi_to_roi_*.m`` scripts, generalised:

* any atlas grid/space: streamline voxel indices are mapped to atlas voxels through the two affines
  (fiber grid -> world mm -> atlas voxel) instead of a hand-computed offset;
* the voxel-index convention of ``fibers_vox`` (1-indexed MATLAB indices into the 91x109x91
  MNI152 2 mm grid) is made explicit: ``RAS`` (x increases with the voxel index, as assumed by the
  original script) or ``LAS`` (FSL/SPM storage order, x decreasing).  ``auto`` decides with a
  data-driven check against template brain/white-matter maps;
* streaming over the .mat file (v7.3/HDF5 chunked; v7 fully loaded) with an optional compact
  cache, and vectorised pair accumulation (sparse incidence matrix products);
* identical outputs to the MATLAB version: raw counts, 1/length weights, volume normalisation and
  their combination, plus sanity checks, a figure and a MATLAB-compatible .mat file.
"""
from __future__ import annotations

import gzip
import json
import shutil
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from ..atlases import Atlas
from ..utils import LOG, ensure_dir, save_json, save_matrix, which
from .common import accumulate_pairs, endpoint_pairs, normalise_matrices, sc_summary

FIBER_SHAPE = (91, 109, 91)
VOXEL_MM = 2.0
# 0-indexed voxel -> mm for the MNI152 (NLin6Asym / FSL) 2 mm grid
FIBER_GRIDS = {
    "RAS": np.array([[2.0, 0, 0, -90.0], [0, 2.0, 0, -126.0], [0, 0, 2.0, -72.0], [0, 0, 0, 1.0]]),
    "LAS": np.array([[-2.0, 0, 0, 90.0], [0, 2.0, 0, -126.0], [0, 0, 2.0, -72.0], [0, 0, 0, 1.0]]),
}


# --------------------------------------------------------------------------- reading fibers
def _is_hdf5(path: Path) -> bool:
    with open(path, "rb") as f:
        head = f.read(128)
    return b"MATLAB 7.3" in head or head[:8] == b"\x89HDF\r\n\x1a\n"


def resolve_connectome_file(path: str | Path, work_dir: str | Path | None = None) -> Path:
    """Accept .mat or .mat.gz; gunzip once into work_dir (or next to the file) when needed."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".gz":
        target_dir = Path(work_dir) if work_dir else path.parent
        target = ensure_dir(target_dir) / path.name[:-3]
        if not target.exists() or target.stat().st_size == 0:
            LOG.info("decompressing %s -> %s (once)", path.name, target)
            with gzip.open(path, "rb") as fi, open(target, "wb") as fo:
                shutil.copyfileobj(fi, fo, length=64 * 1024 * 1024)
        return target
    return path


class FiberStore:
    """Iterate over streamlines (voxel coordinates, N x 3) of a fibers_vox .mat file, chunk by chunk.

    A compact cache (points + offsets, .npy) is written on the first full pass when cache_dir is
    given; later passes (other atlases, orientation checks) read the cache with memory mapping.
    """

    def __init__(self, mat_path: str | Path, var_name: str | None = None, cache_dir: str | Path | None = None):
        self.mat_path = Path(mat_path)
        self.var_name = var_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._cache_base = (self.cache_dir / (self.mat_path.stem + "_cache")) if self.cache_dir else None
        self.n_streamlines: int | None = None
        self.meta: dict = {}
        if self._cache_base and (self._cache_base.with_name(self._cache_base.name + "_offsets.npy")).exists():
            self.meta = json.loads((self._cache_base.with_name(self._cache_base.name + "_meta.json")).read_text())
            self.n_streamlines = int(self.meta["n_streamlines"])

    # ---- cache helpers
    def _cache_paths(self):
        b = self._cache_base
        return b.with_name(b.name + "_points.npy"), b.with_name(b.name + "_offsets.npy"), b.with_name(b.name + "_meta.json")

    def has_cache(self) -> bool:
        return bool(self._cache_base) and self._cache_paths()[1].exists()

    # ---- raw .mat iteration
    def _iter_mat(self, chunk_size: int):
        path = self.mat_path
        if _is_hdf5(path):
            import h5py

            with h5py.File(path, "r") as f:
                name = self.var_name
                if name is None:
                    cands = [k for k in f.keys() if not k.startswith("#") and isinstance(f[k], h5py.Dataset) and f[k].dtype.kind == "O"]
                    name = "fibers_vox" if "fibers_vox" in f else (cands[0] if cands else None)
                if name is None:
                    raise KeyError(f"no cell array found in {path}; keys: {list(f.keys())}")
                refs = f[name][()].ravel()
                self.n_streamlines = int(len(refs))
                for start in range(0, len(refs), chunk_size):
                    arrs = []
                    for r in refs[start : start + chunk_size]:
                        a = np.asarray(f[r][()])
                        if a.ndim == 2 and a.shape[0] == 3 and a.shape[1] != 3:
                            a = a.T  # MATLAB stores N x 3 as 3 x N in HDF5
                        elif a.ndim == 1:
                            a = a.reshape(-1, 3) if a.size % 3 == 0 else np.zeros((0, 3))
                        arrs.append(a.astype(np.float32, copy=False))
                    yield arrs
        else:
            from scipy.io import loadmat, whosmat

            name = self.var_name
            if name is None:
                names = [n for n, _s, c in whosmat(str(path)) if c == "cell"]
                name = "fibers_vox" if "fibers_vox" in names else (names[0] if names else None)
            if name is None:
                raise KeyError(f"no cell array in {path}")
            LOG.info("loading %s (MATLAB v5/v7 format: whole file in memory)...", path.name)
            cell = loadmat(str(path), variable_names=[name])[name].ravel()
            self.n_streamlines = int(len(cell))
            for start in range(0, len(cell), chunk_size):
                arrs = []
                for a in cell[start : start + chunk_size]:
                    a = np.asarray(a)
                    if a.ndim == 2 and a.shape[0] == 3 and a.shape[1] != 3:
                        a = a.T
                    elif a.ndim != 2:
                        a = a.reshape(-1, 3) if a.size % 3 == 0 else np.zeros((0, 3))
                    arrs.append(a.astype(np.float32, copy=False))
                yield arrs

    def iter_chunks(self, chunk_size: int = 100_000, write_cache: bool = True):
        """Yield (points (M x 3 float), offsets (n+1 int64)) per chunk of streamlines."""
        if self.has_cache():
            pts_p, off_p, _ = self._cache_paths()
            offsets = np.load(off_p)
            pts = np.load(pts_p, mmap_mode="r")
            self.n_streamlines = int(len(offsets) - 1)
            for start in range(0, self.n_streamlines, chunk_size):
                stop = min(start + chunk_size, self.n_streamlines)
                o = offsets[start : stop + 1]
                p = np.asarray(pts[o[0] : o[-1]], dtype=np.float32)
                yield p, (o - o[0]).astype(np.int64)
            return
        writer = _CacheWriter(*self._cache_paths()[:2]) if (write_cache and self._cache_base) else None
        t0 = time.time()
        done = 0
        for arrs in self._iter_mat(chunk_size):
            lengths = np.array([len(a) for a in arrs], np.int64)
            offsets = np.concatenate([[0], np.cumsum(lengths)])
            pts = np.concatenate(arrs, axis=0) if arrs else np.zeros((0, 3), np.float32)
            if writer:
                writer.append(pts, lengths)
            done += len(arrs)
            if self.n_streamlines:
                LOG.info("  read %d / %d streamlines (%.1f%%, %.0fs)", done, self.n_streamlines, 100 * done / self.n_streamlines, time.time() - t0)
            yield pts, offsets
        if writer:
            writer.close(self.n_streamlines or done, self._cache_paths()[2], str(self.mat_path))
            self.meta = json.loads(self._cache_paths()[2].read_text())


class _CacheWriter:
    """Append-only writer of concatenated points (smallest exact integer dtype) + offsets."""

    def __init__(self, pts_path: Path, off_path: Path):
        ensure_dir(pts_path.parent)
        self.pts_path, self.off_path = pts_path, off_path
        self.tmp = pts_path.with_suffix(".tmp")
        self.f = open(self.tmp, "wb")
        self.lengths: list[np.ndarray] = []
        self.n_points = 0
        self.integer = True
        self.vmin, self.vmax = np.inf, -np.inf

    def append(self, pts: np.ndarray, lengths: np.ndarray):
        if len(pts):
            if self.integer and not np.all(np.mod(pts, 1) == 0):
                self.integer = False
            self.vmin = min(self.vmin, float(pts.min()))
            self.vmax = max(self.vmax, float(pts.max()))
        pts.astype(np.float32).tofile(self.f)
        self.lengths.append(lengths)
        self.n_points += len(pts)

    def close(self, n_streamlines: int, meta_path: Path, source: str):
        self.f.close()
        lengths = np.concatenate(self.lengths) if self.lengths else np.zeros(0, np.int64)
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        raw = np.memmap(self.tmp, dtype=np.float32, mode="r", shape=(self.n_points, 3))
        if self.integer and self.vmin >= 0 and self.vmax <= 255:
            dtype = np.uint8
        elif self.integer and abs(self.vmin) < 32000 and self.vmax < 32000:
            dtype = np.int16
        else:
            dtype = np.float32
        out = np.lib.format.open_memmap(self.pts_path, mode="w+", dtype=dtype, shape=(self.n_points, 3))
        step = 50_000_000
        for s in range(0, self.n_points, step):
            out[s : s + step] = raw[s : s + step].astype(dtype)
        out.flush()
        del out, raw
        self.tmp.unlink()
        np.save(self.off_path, offsets)
        save_json(meta_path, {"n_streamlines": int(n_streamlines), "n_points": int(self.n_points), "dtype": np.dtype(dtype).name,
                              "integer_coordinates": bool(self.integer), "min": self.vmin, "max": self.vmax, "source": source})
        LOG.info("cache written: %d streamlines, %d points (%s)", n_streamlines, self.n_points, np.dtype(dtype).name)


# --------------------------------------------------------------------------- geometry
def fiber_affine(grid: str | np.ndarray | list) -> np.ndarray:
    if isinstance(grid, str):
        if grid.upper() not in FIBER_GRIDS:
            raise ValueError(f"unknown fiber grid {grid!r}; use RAS, LAS or a 4x4 affine")
        return FIBER_GRIDS[grid.upper()].copy()
    A = np.asarray(grid, float)
    if A.shape != (4, 4):
        raise ValueError("fiber_affine must be 4x4")
    return A


def fiber_to_atlas_matrix(A_fiber: np.ndarray, atlas_affine: np.ndarray) -> np.ndarray:
    """4x4 mapping 1-indexed fiber voxel coordinates -> 0-indexed atlas voxel coordinates."""
    shift = np.eye(4)
    shift[:3, 3] = -1.0  # MATLAB 1-indexed -> 0-indexed
    return np.linalg.inv(atlas_affine) @ A_fiber @ shift


def map_points(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a 4x4 matrix to N x 3 points and round to integer voxel indices."""
    v = pts.astype(np.float64) @ M[:3, :3].T + M[:3, 3]
    return np.rint(v).astype(np.int64)


def streamline_lengths(pts: np.ndarray, offsets: np.ndarray, mode: str, A_fiber: np.ndarray) -> np.ndarray:
    n_pts = np.diff(offsets)
    if mode == "polyline":
        world = pts.astype(np.float64) @ A_fiber[:3, :3].T
        seg = np.linalg.norm(np.diff(world, axis=0), axis=1)
        # segment k connects point k and k+1; exclude the joints between consecutive streamlines
        cs = np.concatenate([[0.0], np.cumsum(seg)])
        L = cs[np.maximum(offsets[1:] - 1, 0)] - cs[offsets[:-1]]
        return np.maximum(L, VOXEL_MM)
    return n_pts.astype(float) * float(np.abs(A_fiber[0, 0]))


# --------------------------------------------------------------------------- density & orientation
def streamline_density(store: FiberStore, chunk_size: int = 100_000, shape=FIBER_SHAPE) -> np.ndarray:
    dens = np.zeros(int(np.prod(shape)), np.int64)
    dims = np.array(shape)
    for pts, _off in store.iter_chunks(chunk_size):
        v = np.rint(pts).astype(np.int64) - 1
        ok = np.all(v >= 0, axis=1) & np.all(v < dims, axis=1)
        lin = np.ravel_multi_index((v[ok, 0], v[ok, 1], v[ok, 2]), shape)
        dens += np.bincount(lin, minlength=dens.size)
    return dens.reshape(shape)


def _template_maps(brain_mask: str | Path | None = None, wm_probseg: str | Path | None = None, template_dir: str | Path | None = None) -> dict:
    """Locate MNI152NLin6Asym 2 mm brain mask / WM probability / T1w (TemplateFlow cache, FSL or user files)."""
    maps: dict = {}
    if brain_mask:
        maps["brain_mask"] = Path(brain_mask)
    if wm_probseg:
        maps["wm"] = Path(wm_probseg)
    if template_dir:
        d = Path(template_dir)
        for k, pats in (("brain_mask", ["*NLin6Asym*res-02*brain_mask*", "MNI152_T1_2mm_brain_mask.nii.gz"]),
                        ("wm", ["*NLin6Asym*res-02*label-WM*probseg*"]), ("t1", ["*NLin6Asym*res-02_T1w*", "MNI152_T1_2mm.nii.gz"])):
            for p in pats:
                hits = sorted(d.glob(p))
                if hits and k not in maps:
                    maps[k] = hits[0]
    try:
        import templateflow.api as tf

        for k, kw in (("brain_mask", dict(desc="brain", suffix="mask")), ("wm", dict(label="WM", suffix="probseg")), ("t1", dict(suffix="T1w", desc=None))):
            if k in maps:
                continue
            try:
                p = tf.get("MNI152NLin6Asym", resolution=2, extension=".nii.gz", **kw)
                if isinstance(p, list):
                    p = p[0] if p else None
                if p:
                    maps[k] = Path(p)
            except Exception as e:  # noqa: BLE001
                LOG.debug("templateflow %s: %s", k, e)
    except Exception:  # noqa: BLE001
        pass
    fsl = Path(__import__("os").environ.get("FSLDIR", "")) / "data" / "standard"
    if "brain_mask" not in maps and (fsl / "MNI152_T1_2mm_brain_mask.nii.gz").exists():
        maps["brain_mask"] = fsl / "MNI152_T1_2mm_brain_mask.nii.gz"
    if "t1" not in maps and (fsl / "MNI152_T1_2mm_brain.nii.gz").exists():
        maps["t1"] = fsl / "MNI152_T1_2mm_brain.nii.gz"
    return maps


def _resample_to_fiber_grid(img_path: Path, A_fiber: np.ndarray, shape=FIBER_SHAPE, order: int = 0) -> np.ndarray:
    """Sample a template image at the world position of every fiber-grid voxel (nearest neighbour)."""
    img = nib.load(str(img_path))
    data = np.asarray(img.dataobj, float)
    if data.ndim == 4:
        data = data[..., 0]
    ii, jj, kk = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    vox = np.stack([ii.ravel(), jj.ravel(), kk.ravel(), np.ones(ii.size)], axis=0)
    world = A_fiber @ vox
    src = np.linalg.inv(img.affine) @ world
    s = np.rint(src[:3]).astype(np.int64)
    ok = np.all(s >= 0, axis=0) & np.all(s < np.array(data.shape)[:, None], axis=0)
    out = np.zeros(ii.size)
    out[ok] = data[s[0, ok], s[1, ok], s[2, ok]]
    return out.reshape(shape)


def diagnose_orientation(density: np.ndarray, maps: dict, candidates=("RAS", "LAS")) -> dict:
    """Score each voxel-index convention: fraction of streamline points inside the template brain
    mask and correlation of the streamline density with the template white-matter probability
    (or T1w intensity).  Both maps are asymmetric enough to tell a left-right flip apart."""
    res: dict = {"candidates": {}, "maps": {k: str(v) for k, v in maps.items()}}
    dens = density.astype(float)
    total = dens.sum()
    logd = np.log1p(dens)
    for cand in candidates:
        A = fiber_affine(cand)
        score: dict = {}
        if "brain_mask" in maps:
            bm = _resample_to_fiber_grid(maps["brain_mask"], A) > 0.5
            score["inside_brain_fraction"] = float(dens[bm].sum() / total) if total else float("nan")
        ref = maps.get("wm") or maps.get("t1")
        if ref:
            wm = _resample_to_fiber_grid(ref, A)
            sel = wm > 0 if "wm" in maps else np.ones_like(wm, bool)
            if "brain_mask" in maps:
                sel &= bm
            score["density_wm_correlation"] = float(np.corrcoef(logd[sel], wm[sel])[0, 1]) if sel.sum() > 10 else float("nan")
        res["candidates"][cand] = score
    # decision: WM correlation first, inside-brain fraction second
    best, best_key = None, (-np.inf, -np.inf)
    for cand, sc in res["candidates"].items():
        key = (sc.get("density_wm_correlation", -np.inf), sc.get("inside_brain_fraction", -np.inf))
        if key > best_key:
            best, best_key = cand, key
    res["best"] = best
    a, b = [res["candidates"][c].get("density_wm_correlation", np.nan) for c in candidates[:2]]
    res["margin_wm_correlation"] = float(abs(a - b)) if np.isfinite(a) and np.isfinite(b) else float("nan")
    res["confident"] = bool(np.isfinite(res["margin_wm_correlation"]) and res["margin_wm_correlation"] > 0.01)
    return res


# --------------------------------------------------------------------------- atlas warping
def warp_atlas_to_fiber_space(atlas: Atlas, work_dir: Path, mode: str = "auto") -> Atlas:
    """If the atlas is in MNI152NLin2009cAsym and ANTs + TemplateFlow transforms are available, warp it
    (nearest neighbour) into the MNI152NLin6Asym 2 mm grid of the fibers.  Otherwise return it unchanged
    (affine-only alignment, a few mm error between the two MNI templates)."""
    if mode == "never" or (atlas.space or "").lower() in ("mni152nlin6asym", "") :
        return atlas
    if (atlas.space or "").lower() != "mni152nlin2009casym":
        LOG.warning("atlas space %s: no transform to MNI152NLin6Asym available; using affine alignment only", atlas.space)
        return atlas
    ants = which("antsApplyTransforms")
    xfm = ref = None
    try:
        import templateflow.api as tf

        xfm = tf.get("MNI152NLin6Asym", **{"from": "MNI152NLin2009cAsym"}, mode="image", suffix="xfm", extension=".h5")
        ref = tf.get("MNI152NLin6Asym", resolution=2, suffix="T1w", desc=None, extension=".nii.gz")
        xfm = xfm[0] if isinstance(xfm, list) else xfm
        ref = ref[0] if isinstance(ref, list) else ref
    except Exception as e:  # noqa: BLE001
        LOG.debug("templateflow transform lookup failed: %s", e)
    if not (ants and xfm and ref):
        msg = "atlas is in MNI152NLin2009cAsym but ANTs/TemplateFlow transform unavailable: aligning by affine only (as in the original script)"
        if mode == "always":
            raise RuntimeError(msg)
        LOG.warning(msg)
        return atlas
    out = ensure_dir(work_dir) / (atlas.image.name.replace(".nii.gz", "").replace(".nii", "") + "_space-MNI152NLin6Asym.nii.gz")
    if not out.exists():
        import subprocess

        cmd = [ants, "-d", "3", "-i", str(atlas.image), "-r", str(ref), "-o", str(out), "-n", "GenericLabel", "-t", str(xfm)]
        LOG.info("warping atlas into MNI152NLin6Asym: %s", " ".join(cmd))
        subprocess.check_call(cmd)
    from ..atlases import atlas_from_file

    return atlas_from_file(out, labels=atlas.labels_file, name=atlas.name)


def run_orientation_check(connectome: str | Path, work_dir: str | Path, cfg: dict | None = None, out_json: str | Path | None = None) -> dict:
    """Stand-alone orientation check of a fibers_vox file (no atlas needed).

    Computes the streamline density in the fiber grid (writing the point cache for later SC builds)
    and scores the RAS / LAS voxel-index conventions against MNI152NLin6Asym template maps."""
    cfg = dict(cfg or {})
    work_dir = ensure_dir(work_dir)
    mat = resolve_connectome_file(connectome, work_dir)
    store = FiberStore(mat, cfg.get("var_name"), cache_dir=work_dir if cfg.get("cache", True) else None)
    t0 = time.time()
    dens = streamline_density(store, int(cfg.get("chunk_size", 100_000)))
    np.save(work_dir / "streamline_density_fibergrid.npy", dens)
    maps = _template_maps(cfg.get("brain_mask"), cfg.get("wm_probseg"), cfg.get("template_dir"))
    if not maps:
        raise RuntimeError("no template maps found (TemplateFlow cache, $FSLDIR or sc.normative.brain_mask/wm_probseg)")
    diag = diagnose_orientation(dens, maps)
    total = float(dens.sum())
    inside_grid = {"n_streamlines": store.n_streamlines, "n_points_in_grid": int(total),
                   "n_points_total": int(store.meta.get("n_points", total)) if store.meta else None,
                   "coordinate_range": [store.meta.get("min"), store.meta.get("max")] if store.meta else None,
                   "integer_coordinates": store.meta.get("integer_coordinates") if store.meta else None}
    # left/right asymmetry of the density itself (voxel index axis): informative but convention-free
    x_profile = dens.sum(axis=(1, 2))
    diag.update({"file": str(mat), "elapsed_s": time.time() - t0, "streamlines": inside_grid,
                 "density_x_profile_first_half_fraction": float(x_profile[: len(x_profile) // 2].sum() / max(total, 1))})
    if out_json:
        save_json(out_json, diag)
    return diag


# --------------------------------------------------------------------------- main builder
def build_normative_sc(atlas: Atlas, connectome: str | Path, out_dir: str | Path, cfg: dict | None = None,
                       work_dir: str | Path | None = None, atlas_grid_override: np.ndarray | None = None) -> dict:
    """Full pipeline: read fibers, decide orientation, accumulate ROI x ROI matrices, normalise, save."""
    cfg = dict(cfg or {})
    out_dir = ensure_dir(out_dir)
    work_dir = ensure_dir(work_dir or (Path(out_dir) / "work"))
    t0 = time.time()
    mat = resolve_connectome_file(connectome, work_dir)
    store = FiberStore(mat, cfg.get("var_name"), cache_dir=work_dir if cfg.get("cache", True) else None)
    chunk = int(cfg.get("chunk_size", 100_000))
    grid = cfg.get("fiber_grid", "auto")
    atlas_used = atlas
    if cfg.get("warp_atlas", "auto") != "never":
        atlas_used = warp_atlas_to_fiber_space(atlas, work_dir, cfg.get("warp_atlas", "auto"))
    # orientation
    diag = None
    if isinstance(grid, str) and grid.lower() == "auto":
        LOG.info("orientation check of fibers_vox against template maps ...")
        dens = streamline_density(store, chunk)
        np.save(work_dir / "streamline_density_fibergrid.npy", dens)
        maps = _template_maps(cfg.get("brain_mask"), cfg.get("wm_probseg"), cfg.get("template_dir"))
        if not maps:
            LOG.warning("no template maps found for the orientation check; falling back to RAS (original script convention)")
            grid = "RAS"
        else:
            diag = diagnose_orientation(dens, maps)
            grid = diag["best"] or "RAS"
            LOG.warning("fibers_vox voxel convention chosen automatically: %s  %s", grid, json.dumps(diag["candidates"]))
            if not diag["confident"]:
                LOG.warning("orientation evidence is weak (margin %.3f); set sc.normative.fiber_grid explicitly", diag["margin_wm_correlation"])
    A_fiber = fiber_affine(cfg.get("fiber_affine")) if (isinstance(grid, str) and grid.lower() == "custom") else fiber_affine(grid)
    atl_img = atlas_used.load_img()
    atl = atlas_used.data()
    dims = np.array(atl.shape)
    M = fiber_to_atlas_matrix(A_fiber, atl_img.affine)
    ids = atlas_used.label_ids
    n_roi = len(ids)
    pos = np.full(int(atl.max()) + 1, -1, np.int64)
    pos[ids] = np.arange(n_roi)
    roi_vol = np.bincount(atl.ravel(), minlength=len(pos))[ids].astype(float)
    LOG.info("fiber grid %s; fiber->atlas voxel matrix offset %s; %d ROIs", grid, np.round(M[:3, 3], 3).tolist(), n_roi)
    count = np.zeros((n_roi, n_roi))
    lenw = np.zeros((n_roi, n_roi))
    n_total = 0
    n_connecting = 0
    len_sum = 0.0
    n_len = 0
    count_mode = cfg.get("count_mode", "touched")
    length_mode = cfg.get("length_mode", "npoints")
    for pts, offsets in store.iter_chunks(chunk):
        n = len(offsets) - 1
        if n == 0:
            continue
        L = streamline_lengths(pts, offsets, length_mode, A_fiber)
        len_sum += L.sum()
        n_len += n
        inv_len = 1.0 / np.maximum(L, 1e-9)
        v = map_points(pts, M)
        inside = np.all(v >= 0, axis=1) & np.all(v < dims, axis=1)
        lab = np.zeros(len(pts), np.int64)
        lab[inside] = atl[v[inside, 0], v[inside, 1], v[inside, 2]]
        idx = np.where(lab > 0, pos[np.minimum(lab, len(pos) - 1)], -1)
        sid = np.repeat(np.arange(n), np.diff(offsets))
        if count_mode == "endpoints":
            first = idx[offsets[:-1]]
            last = idx[np.maximum(offsets[1:] - 1, 0)]
            c, w, nc = endpoint_pairs(first, last, n_roi, inv_len)
        else:
            keep = idx >= 0
            c, w, nc = accumulate_pairs(sid[keep], idx[keep], n, n_roi, inv_len)
        n_connecting += nc
        count += c
        lenw += w
        n_total += n
    mats = normalise_matrices(count, lenw, roi_vol)
    summary = sc_summary(count, atlas_used.hemispheres, list(atlas_used.label_table["type"]) if (atlas_used.label_table is not None and "type" in atlas_used.label_table) else None)
    summary.update({"n_streamlines": int(n_total), "n_connecting_streamlines": int(n_connecting), "mean_streamline_length_mm": float(len_sum / max(n_len, 1)),
                    "fiber_grid": grid if isinstance(grid, str) else "custom", "count_mode": count_mode, "length_mode": length_mode,
                    "elapsed_s": time.time() - t0, "atlas": atlas_used.name, "atlas_file": str(atlas_used.image),
                    "atlas_space": atlas_used.space, "fiber_affine": A_fiber.tolist(), "fiber_to_atlas_matrix": M.tolist(),
                    "orientation_diagnostics": diag})
    # save
    base = out_dir / f"atlas-{_slug(atlas.name)}_desc-normative"
    names = atlas_used.region_names
    for key in ("count", "volnorm", "lencorr", "combined"):
        save_matrix(Path(str(base) + f"_weight-{key}_connectivity.tsv"), mats[key], names)
    save_json(Path(str(base) + "_connectivity.json"), summary)
    try:
        from scipy.io import savemat

        savemat(str(base) + "_connectivity.mat", {
            "conn_matrix": mats["count"], "conn_matrix_volnorm": mats["volnorm"], "conn_matrix_lencorr": mats["lencorr"],
            "conn_matrix_combined": mats["combined"], "labels": np.asarray(ids, float), "nROIs": float(n_roi), "roi_vol": roi_vol,
            "roi_names": np.array(names, dtype=object), "roi_hemisphere": np.array(atlas_used.hemispheres, dtype=object),
            "roi_network": np.array(atlas_used.networks, dtype=object), "atlas_file": str(atlas_used.image),
            "fiber_affine": A_fiber, "fiber_grid": str(grid)}, do_compression=True)
    except Exception as e:  # noqa: BLE001
        LOG.warning("could not write .mat copy: %s", e)
    try:
        from ..plotting import plot_sc_panels

        plot_sc_panels(mats, Path(str(base) + "_connectivity.png"), title=f"dTOR normative SC - {atlas.name} ({grid})")
    except Exception as e:  # noqa: BLE001
        LOG.warning("figure failed: %s", e)
    LOG.info("normative SC done in %.0fs: %s", time.time() - t0, json.dumps({k: summary[k] for k in ("n_streamlines", "density", "empty_nodes", "LR_ratio") if k in summary}))
    return {"matrices": mats, "roi_vol": roi_vol, "summary": summary, "files": {"base": str(base)}, "atlas": atlas_used}


def _slug(s: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9]", "", s)[:60]
