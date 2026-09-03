"""Shared structural-connectivity utilities: streamline -> ROI accumulation, normalisation, checks."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..utils import LOG


def accumulate_pairs(sid: np.ndarray, label_idx: np.ndarray, n_streamlines: int, n_roi: int, weights: np.ndarray | None = None):
    """Given (streamline id, ROI index) hits, return (count, weighted) ROI x ROI matrices.

    Every unordered pair of distinct ROIs touched by the same streamline gets +1 (both directions)
    and +weight[streamline] (e.g. 1/length) - the 'touched' convention of the original dTOR script.
    Implemented as M^T M with M the sparse streamline x ROI incidence matrix.
    """
    if len(sid) == 0:
        z = np.zeros((n_roi, n_roi))
        return z, z.copy(), 0
    key = np.unique(sid.astype(np.int64) * n_roi + label_idx.astype(np.int64))
    s = key // n_roi
    r = key % n_roi
    M = sp.csr_matrix((np.ones(len(s)), (s, r)), shape=(n_streamlines, n_roi))
    n_connecting = int(np.sum(np.bincount(s, minlength=n_streamlines) >= 2))
    count = (M.T @ M).toarray()
    if weights is None:
        wmat = count.copy()
    else:
        W = sp.diags(np.asarray(weights, float))
        wmat = (M.T @ W @ M).toarray()
    np.fill_diagonal(count, 0.0)
    np.fill_diagonal(wmat, 0.0)
    return count, wmat, n_connecting


def endpoint_pairs(first_idx: np.ndarray, last_idx: np.ndarray, n_roi: int, weights: np.ndarray | None = None):
    """Endpoint convention: one count per streamline whose two end points fall in different ROIs."""
    ok = (first_idx >= 0) & (last_idx >= 0) & (first_idx != last_idx)
    a, b = first_idx[ok], last_idx[ok]
    w = np.ones(ok.sum()) if weights is None else np.asarray(weights, float)[ok]
    count = np.zeros((n_roi, n_roi))
    wmat = np.zeros((n_roi, n_roi))
    np.add.at(count, (a, b), 1.0)
    np.add.at(count, (b, a), 1.0)
    np.add.at(wmat, (a, b), w)
    np.add.at(wmat, (b, a), w)
    return count, wmat, int(ok.sum())


def normalise_matrices(count: np.ndarray, lenw: np.ndarray, roi_vol: np.ndarray) -> dict:
    """count / sqrt(vol_i vol_j), sum(1/len), and both corrections combined (diagonals zeroed)."""
    gv = np.sqrt(np.outer(roi_vol, roi_vol))
    gv[gv == 0] = np.inf
    volnorm = count / gv
    combined = lenw / gv
    for m in (volnorm, combined):
        np.fill_diagonal(m, 0.0)
    return {"count": count, "volnorm": volnorm, "lencorr": lenw, "combined": combined}


def sc_summary(count: np.ndarray, hemispheres: list[str] | None = None, types: list[str] | None = None) -> dict:
    n = count.shape[0]
    iu = np.triu_indices(n, 1)
    out = {
        "n_roi": int(n),
        "total_connections_upper": float(count[iu].sum()),
        "density": float(np.count_nonzero(count) / max(n * (n - 1), 1)),
        "max_edge": float(count.max()),
        "empty_nodes": int(np.sum(count.sum(axis=1) == 0)),
        "symmetric": bool(np.allclose(count, count.T)),
    }
    if hemispheres and any(hemispheres):
        h = np.array(hemispheres)
        t = np.array(types) if types else np.array([""] * n)
        cortex = np.array([("cort" in x.lower()) or (x == "") for x in t])
        lh = (h == "L") & cortex
        rh = (h == "R") & cortex
        if lh.any() and rh.any():
            l, r = float(count[lh].sum(axis=1).mean()), float(count[rh].sum(axis=1).mean())
            out.update({"mean_rowsum_L": l, "mean_rowsum_R": r, "LR_ratio": (l / r) if r else float("nan")})
        inter = float(count[np.ix_(h == "L", h == "R")].sum())
        intra = float(count[np.ix_(h == "L", h == "L")].sum() / 2 + count[np.ix_(h == "R", h == "R")].sum() / 2)
        out["interhemispheric_fraction"] = inter / (inter + intra) if (inter + intra) else float("nan")
    return out


def prepare_sc(SC: np.ndarray, cfg: dict) -> np.ndarray:
    """Symmetrise / log-transform / threshold / scale a raw SC matrix for modelling."""
    C = np.array(SC, float)
    C = np.nan_to_num(C)
    np.fill_diagonal(C, 0.0)
    if cfg.get("symmetrize", True):
        C = 0.5 * (C + C.T)
    if cfg.get("log_transform", False):
        C = np.log1p(C)
    C[C < 0] = 0.0
    thr = float(cfg.get("threshold", 0.0) or 0.0)
    mx = C.max()
    if thr > 0 and mx > 0:
        C[C < thr * mx] = 0.0
    if cfg.get("normalize", "max") == "max" and C.max() > 0:
        C = C / C.max() * float(cfg.get("sc_max", 0.2))
    if C.max() == 0:
        LOG.warning("structural connectivity matrix is all zeros")
    return C
