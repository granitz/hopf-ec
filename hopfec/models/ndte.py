"""Normalised directed transfer entropy (NDTE) - Deco, Vidaurre & Kringelbach, Nat Hum Behav 2021.

For target i and source k with L lags (present + L-1 past values of the target, L-1 past values of the
source), under the Gaussian assumption:

    TE(k -> i)   = log s2(i | own past) - log s2(i | own past, source past)      (2 x transfer entropy)
    I(i ; past)  = log var(i)           - log s2(i | own past, source past)      (present vs joint past)
    NDTE(i <- k) = TE / I(i ; past)                                             (in [0, 1])

s2(. | .) are conditional variances (Schur complements of the lagged covariance), which is exactly the
log-det formulation of the original MATLAB code (ndte_example_surrogates_fixlags_cs.m /
pair_granger_norm.m) but computed for all N^2 pairs at once.  Significance: circular-shift surrogates of
both series (independent offsets in [5 %, 95 %] of the length), z-scores, KDE/empirical p-values, FDR.

Convention: NDTE[i, k] is the flow from source k to target i (row = target), like the EC matrices.

For the linear Hopf model the same quantity follows analytically from the (filter-consistent) lagged
covariances c(k) = Cov(x(t+k), x(t)), k = 0..L-1, via a block-Toeplitz lagged covariance.
"""
from __future__ import annotations

import numpy as np

from ..utils import LOG


def lagged_covariance(X: np.ndarray, L: int) -> np.ndarray:
    """Sample covariance of the lagged vector [x_n(t-p)]_{n, p=0..L-1}, index n*L + p; t = L-1..T-1."""
    X = np.asarray(X, float)
    T, N = X.shape
    if T <= L + 5:
        raise ValueError("time series too short for the requested number of lags")
    M = T - (L - 1)
    D = np.empty((M, N, L))
    for p in range(L):
        D[:, :, p] = X[L - 1 - p : T - p]
    D = D.reshape(M, N * L)
    D = D - D.mean(axis=0, keepdims=True)
    return D.T @ D / (M - 1)


def toeplitz_lagged_covariance(c: list[np.ndarray]) -> np.ndarray:
    """Lagged covariance from model covariances c[k] = Cov(x(t+k), x(t)) (N x N), k = 0..L-1."""
    L = len(c)
    N = c[0].shape[0]
    S = np.empty((N * L, N * L))
    for p in range(L):
        for q in range(L):
            # Cov(x_n(t-p), x_m(t-q)) = c(q-p)[n, m] for q >= p, else c(p-q)[m, n]
            blk = c[q - p] if q >= p else c[p - q].T
            S[p::L, q::L] = blk
    return 0.5 * (S + S.T)


def ndte_from_lagged_cov(S: np.ndarray, N: int, L: int, eps: float = 1e-15) -> np.ndarray:
    """NDTE for all target/source pairs from a lagged covariance (index n*L + p)."""
    idx = lambda n, p: n * L + p  # noqa: E731
    out = np.zeros((N, N))
    Lp = L - 1
    for i in range(N):
        own = [idx(i, p) for p in range(1, L)]
        v_i = S[idx(i, 0), idx(i, 0)]
        A = S[np.ix_(own, own)]
        a = S[idx(i, 0), own]
        s2_own = v_i - a @ np.linalg.solve(A, a)
        # all sources at once: block system [[A, D_k], [D_k^T, B_k]]
        srcs = [k for k in range(N) if k != i]
        past = np.array([[idx(k, p) for p in range(1, L)] for k in srcs])       # (N-1, Lp)
        B = S[past[:, :, None], past[:, None, :]]                                  # (N-1, Lp, Lp)
        D = S[np.array(own)[None, :, None], past[:, None, :]]                     # (N-1, Lp, Lp)  own x source
        b = S[idx(i, 0), past]                                                     # (N-1, Lp)
        M = np.empty((len(srcs), 2 * Lp, 2 * Lp))
        M[:, :Lp, :Lp] = A
        M[:, :Lp, Lp:] = D
        M[:, Lp:, :Lp] = np.transpose(D, (0, 2, 1))
        M[:, Lp:, Lp:] = B
        rhs = np.concatenate([np.broadcast_to(a, (len(srcs), Lp)), b], axis=1)[:, :, None]
        sol = np.linalg.solve(M, rhs)[:, :, 0]
        s2_joint = v_i - np.einsum("kj,kj->k", rhs[:, :, 0], sol)
        s2_joint = np.clip(s2_joint, 1e-300, None)
        te = np.log(max(s2_own, 1e-300)) - np.log(s2_joint)
        info = np.log(max(v_i, 1e-300)) - np.log(s2_joint)
        val = te / np.where(info > 0, info, np.inf)
        val = np.where(val <= 0, eps, val)
        out[i, srcs] = val
    return out


def ndte(X: np.ndarray, max_lag: int = 10) -> np.ndarray:
    """NDTE matrix (row = target, column = source) of a (T x N) time-series array."""
    X = np.asarray(X, float)
    N = X.shape[1]
    return ndte_from_lagged_cov(lagged_covariance(X, max_lag), N, max_lag)


def ndte_linear_model(c: list[np.ndarray]) -> np.ndarray:
    """NDTE implied by a linear model with lagged covariances c[k], k = 0..L-1."""
    N = c[0].shape[0]
    return ndte_from_lagged_cov(toeplitz_lagged_covariance(c), N, len(c))


def circular_shift_surrogate(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independent circular shift of every node's series (offset in [5 %, 95 %] of the length)."""
    T = X.shape[0]
    lo, hi = int(np.ceil(0.05 * T)), int(np.ceil(0.95 * T))
    out = np.empty_like(X)
    for n in range(X.shape[1]):
        s = int(rng.integers(lo, hi + 1))
        out[:, n] = np.roll(X[:, n], -s)
    return out


def ndte_with_surrogates(X: np.ndarray, max_lag: int = 10, n_surrogates: int = 100, seed: int = 0, fdr_q: float = 0.05,
                         n_jobs: int = 1, p_source: str = "kde") -> dict:
    """NDTE + circular-shift surrogates: z-scores, p-values (KDE as in the original code, Gaussian tail,
    empirical), FDR mask based on `p_source` (kde | gauss | empirical), in/out flows."""
    from scipy.stats import gaussian_kde, norm

    from ..group import fdr_bh

    X = np.asarray(X, float)
    N = X.shape[1]
    val = ndte(X, max_lag)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=int(n_surrogates))
    if n_jobs != 1 and n_surrogates > 4:
        from joblib import Parallel, delayed

        sur = Parallel(n_jobs=n_jobs)(delayed(_surrogate_ndte)(X, max_lag, int(sd)) for sd in seeds)
    else:
        sur = [_surrogate_ndte(X, max_lag, int(sd)) for sd in seeds]
    S = np.stack(sur) if len(sur) else np.zeros((0, N, N))
    off = ~np.eye(N, dtype=bool)
    out = {"ndte": val, "n_surrogates": int(n_surrogates), "max_lag": int(max_lag)}
    if len(sur):
        mu, sd = S.mean(0), S.std(0)
        z = np.zeros((N, N))
        z[off] = (val[off] - mu[off]) / np.where(sd[off] > 0, sd[off], np.inf)
        p_emp = np.ones((N, N))
        p_emp[off] = (1 + np.sum(S[:, off] >= val[off][None, :], axis=0)) / (len(sur) + 1)
        p_kde = np.ones((N, N))
        for i in range(N):
            for k in range(N):
                if i == k:
                    continue
                s = S[:, i, k]
                if s.std() <= 0:
                    p_kde[i, k] = float(val[i, k] <= s.mean())
                    continue
                try:  # positive support by reflection at 0 (ksdensity 'Support','positive' analogue)
                    kde = gaussian_kde(np.concatenate([s, -s]))
                    cdf = 2 * kde.integrate_box_1d(0, val[i, k])
                    p_kde[i, k] = float(np.clip(1 - cdf, 0, 1))
                except Exception:  # noqa: BLE001
                    p_kde[i, k] = p_emp[i, k]
        p_gauss = np.ones((N, N))
        p_gauss[off] = norm.sf(z[off])
        p_use = {"kde": p_kde, "gauss": p_gauss, "empirical": p_emp}.get(p_source, p_kde)
        sig = np.zeros((N, N), bool)
        sig[off] = fdr_bh(p_use[off], fdr_q)
        out.update({"z": z, "p_kde": p_kde, "p_gauss": p_gauss, "p_empirical": p_emp, "sig_fdr": sig, "p_source": p_source,
                    "surrogate_mean": mu, "surrogate_sd": sd, "fdr_q": fdr_q})
        ndte_sig = np.where(sig, val, 0.0)
    else:
        ndte_sig = val
    out["in_flow"] = ndte_sig.sum(axis=1)     # into target i (sum over sources)
    out["out_flow"] = ndte_sig.sum(axis=0)    # out of source k (sum over targets)
    out["total_flow"] = out["in_flow"] + out["out_flow"]
    return out


def _surrogate_ndte(X, max_lag, seed):
    rng = np.random.default_rng(seed)
    return ndte(circular_shift_surrogate(X, rng), max_lag)


def ndte_similarity(ndte_emp: np.ndarray, ndte_model: np.ndarray) -> dict:
    N = ndte_emp.shape[0]
    off = ~np.eye(N, dtype=bool)
    a, b = ndte_emp[off], ndte_model[off]
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    return {"ndte_corr": corr, "ndte_rmse": float(np.sqrt(np.mean((a - b) ** 2)))}
