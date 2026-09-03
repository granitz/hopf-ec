"""Group-level summaries and comparisons of participant effective-connectivity matrices."""
from __future__ import annotations

import numpy as np
from scipy import stats


def fdr_bh(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg: boolean array of rejections at level q."""
    p = np.asarray(p, float).ravel()
    n = np.sum(np.isfinite(p))
    if n == 0:
        return np.zeros_like(p, bool)
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    ranked = p[order]
    thr = q * (np.arange(1, len(p) + 1) / n)
    ok = ranked <= thr
    out = np.zeros_like(p, bool)
    if ok.any():
        kmax = np.max(np.where(ok)[0])
        out[order[: kmax + 1]] = True
    return out


def compare_groups(ECs_a: list[np.ndarray], ECs_b: list[np.ndarray], mask: np.ndarray | None = None, q: float = 0.05,
                   n_perm: int = 0, seed: int = 0) -> dict:
    """Edge-wise Welch t-tests (FDR corrected) and optional permutation test on the group means."""
    A = np.stack(ECs_a)
    B = np.stack(ECs_b)
    N = A.shape[1]
    if mask is None:
        mask = ~np.eye(N, dtype=bool)
    t, p = stats.ttest_ind(A, B, axis=0, equal_var=False, nan_policy="omit")
    t = np.asarray(t, float)
    p = np.asarray(p, float)
    p[~mask] = np.nan
    sig = np.zeros((N, N), bool)
    sig_flat = fdr_bh(p[mask], q)
    sig[mask] = sig_flat
    out = {"mean_a": A.mean(0), "mean_b": B.mean(0), "t": t, "p": p, "sig_fdr": sig, "n_a": len(ECs_a), "n_b": len(ECs_b),
           "n_sig_edges": int(sig.sum()), "q": q}
    # global metrics per participant
    def _glob(E):
        return {"mean_strength": float(E[mask].mean()), "asymmetry": float(np.abs(E - E.T)[mask].mean() / (E[mask].mean() + 1e-12))}
    ga = [_glob(E) for E in ECs_a]
    gb = [_glob(E) for E in ECs_b]
    out["global"] = {}
    for k in ("mean_strength", "asymmetry"):
        x = [g[k] for g in ga]
        y = [g[k] for g in gb]
        tt = stats.ttest_ind(x, y, equal_var=False)
        out["global"][k] = {"mean_a": float(np.mean(x)), "mean_b": float(np.mean(y)), "t": float(tt.statistic), "p": float(tt.pvalue)}
    if n_perm and n_perm > 0:
        rng = np.random.default_rng(seed)
        allE = np.concatenate([A, B])
        obs = np.abs(A.mean(0) - B.mean(0))[mask]
        count = np.zeros(obs.shape)
        maxnull = []
        for _ in range(int(n_perm)):
            perm = rng.permutation(len(allE))
            d = np.abs(allE[perm[: len(A)]].mean(0) - allE[perm[len(A):]].mean(0))[mask]
            count += d >= obs
            maxnull.append(d.max())
        p_perm = np.full((N, N), np.nan)
        p_perm[mask] = (count + 1) / (n_perm + 1)
        p_fwe = np.full((N, N), np.nan)
        p_fwe[mask] = (np.sum(np.asarray(maxnull)[:, None] >= obs[None, :], axis=0) + 1) / (n_perm + 1)
        out["p_perm"] = p_perm
        out["p_perm_fwe"] = p_fwe
        out["n_sig_perm_fwe"] = int(np.nansum(p_fwe < 0.05))
    return out
