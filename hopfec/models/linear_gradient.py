"""Exact-gradient estimation of effective connectivity with the linear Hopf model.

The model moments (filter-consistent FC and lagged correlation, see hopf_linear.py) are computed
with the recursion c(k) = P c(k-1), P = expm(A TR), c(0) = Sigma (Lyapunov), and
S_filt(tau) = sum_m h2(m) c(tau - m).  The gradient of the loss

    L(C) = w_fc/2 sum_{i!=j} (FC_sim - FC_emp)^2 + w_tau/2 sum_ij (COVtau_sim - COVtau_emp)^2
           + lambda_sc/2 ||C - C_prior||^2 + lambda_l1 sum C

with respect to C (and optionally a, omega) is obtained with the adjoint method: back-propagation
through the recursion, the Frechet derivative of expm (adjoint) and the adjoint Lyapunov equation.
C is optimised with L-BFGS-B under C >= 0 on the allowed links.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import linalg as la
from scipy.optimize import minimize

from ..utils import LOG, corr_upper, offdiag_mask, rmse
from .gec import fit_error
from .hopf_linear import build_jacobian, lyapunov_covariance
from .signal import filter_autocorrelation


def _filter_weights(tr: float, band, order: int = 2, tol: float = 1e-6) -> np.ndarray:
    if band is None:
        return np.array([1.0])
    return filter_autocorrelation(tr, band, order, tol=tol)


def forward_moments(A: np.ndarray, beta: float, tr: float, tau: int, h2: np.ndarray, keep_cache: bool = True):
    """Filter-consistent covariance at lag 0 and lag tau (full 2N x 2N) via the lag recursion."""
    Sigma = lyapunov_covariance(A, beta)
    P = la.expm(A * tr)
    M = len(h2) - 1
    K = int(tau) + M
    c = [Sigma]
    for _ in range(K):
        c.append(P @ c[-1])
    S0 = h2[0] * c[0]
    for k in range(1, M + 1):
        S0 = S0 + h2[k] * (c[k] + c[k].T)
    St = np.zeros_like(Sigma)
    for k in range(0, K + 1):
        St = St + h2[abs(tau - k)] * c[k]
    for kp in range(1, M - tau + 1):
        St = St + h2[kp + tau] * c[kp].T
    cache = {"c": c, "P": P, "Sigma": Sigma, "K": K, "M": M} if keep_cache else None
    return S0, St, cache


def backward_moments(G_S0: np.ndarray, G_St: np.ndarray, A: np.ndarray, tr: float, tau: int, h2: np.ndarray, cache: dict) -> np.ndarray:
    """Adjoint: gradient of L wrt A given dL/dS0 and dL/dSt (full 2N x 2N)."""
    c, P, Sigma, K, M = cache["c"], cache["P"], cache["Sigma"], cache["K"], cache["M"]
    n = A.shape[0]
    G_c = [np.zeros((n, n)) for _ in range(K + 1)]
    G_c[0] += h2[0] * G_S0
    sym0 = G_S0 + G_S0.T
    for k in range(1, M + 1):
        G_c[k] += h2[k] * sym0
    for k in range(0, K + 1):
        G_c[k] += h2[abs(tau - k)] * G_St
    for kp in range(1, M - tau + 1):
        G_c[kp] += h2[kp + tau] * G_St.T
    lam = G_c[K].copy()
    G_P = np.zeros((n, n))
    for k in range(K, 0, -1):
        G_P += lam @ c[k - 1].T
        lam = G_c[k - 1] + P.T @ lam
    G_Sigma = 0.5 * (lam + lam.T)
    # through P = expm(A tr): adjoint of the Frechet derivative
    _, G_A_P = la.expm_frechet(A.T * tr, G_P, compute_expm=True)
    G_A = tr * G_A_P
    # through Sigma (Lyapunov): A^T Lam + Lam A = -G_Sigma  ->  dL/dA = 2 Lam Sigma
    Lam = la.solve_continuous_lyapunov(A.T, -G_Sigma)
    G_A = G_A + 2.0 * Lam @ Sigma
    return G_A


def loss_and_grad_moments(S0: np.ndarray, St: np.ndarray, N: int, FC_emp: np.ndarray, COV_emp: np.ndarray,
                          w_fc: float = 1.0, w_tau: float = 1.0, residuals: tuple | None = None):
    """Loss on the x-blocks and its gradient wrt the full S0 / St matrices.

    residuals=(dR, dCt) overrides the residuals of the linear moments (used to back-propagate the
    residuals of the *non-linear* model through the linear model's Jacobian = surrogate gradient)."""
    n = S0.shape[0]
    Sxx = S0[:N, :N]
    Txx = St[:N, :N]
    d = np.clip(np.diag(Sxx), 1e-300, None)
    sd = np.sqrt(d)
    denom = np.outer(sd, sd)
    R = Sxx / denom
    Ct = Txx / denom
    od = offdiag_mask(N)
    if residuals is None:
        dR = (R - FC_emp) * od
        dCt = Ct - COV_emp
    else:
        dR = np.asarray(residuals[0], float) * od
        dCt = np.asarray(residuals[1], float)
    loss = 0.5 * w_fc * np.sum(dR ** 2) + 0.5 * w_tau * np.sum(dCt ** 2)
    G_R = w_fc * dR
    G_Ct = w_tau * dCt
    G_Sxx = G_R / denom
    G_Txx = G_Ct / denom
    # diagonal of S0 enters through the normalisation of both R and Ct
    gdiag = -0.5 / d * ((G_R * R).sum(axis=1) + (G_R * R).sum(axis=0) + (G_Ct * Ct).sum(axis=1) + (G_Ct * Ct).sum(axis=0))
    G_Sxx = G_Sxx.copy()
    G_Sxx[np.arange(N), np.arange(N)] = gdiag
    G_S0 = np.zeros((n, n))
    G_S0[:N, :N] = G_Sxx
    G_St = np.zeros((n, n))
    G_St[:N, :N] = G_Txx
    return loss, G_S0, G_St, R, Ct


def forward_moments_multi(A: np.ndarray, beta: float, tr: float, taus: Sequence[int], h2: np.ndarray):
    """Filter-consistent covariance at lag 0 and at every lag in taus (full 2N x 2N); shared recursion."""
    taus = [int(t) for t in taus]
    Sigma = lyapunov_covariance(A, beta)
    P = la.expm(A * tr)
    M = len(h2) - 1
    K = max(taus) + M
    c = [Sigma]
    for _ in range(K):
        c.append(P @ c[-1])
    S0 = h2[0] * c[0]
    for k in range(1, M + 1):
        S0 = S0 + h2[k] * (c[k] + c[k].T)
    Sts = []
    for tau in taus:
        St = np.zeros_like(Sigma)
        for k in range(0, tau + M + 1):
            St = St + h2[abs(tau - k)] * c[k]
        for kp in range(1, M - tau + 1):
            St = St + h2[kp + tau] * c[kp].T
        Sts.append(St)
    return S0, Sts, {"c": c, "P": P, "Sigma": Sigma, "K": K, "M": M, "taus": taus}


def backward_moments_multi(G_S0: np.ndarray, G_Sts: Sequence[np.ndarray], A: np.ndarray, tr: float, h2: np.ndarray, cache: dict) -> np.ndarray:
    c, P, Sigma, K, M, taus = cache["c"], cache["P"], cache["Sigma"], cache["K"], cache["M"], cache["taus"]
    n = A.shape[0]
    G_c = [np.zeros((n, n)) for _ in range(K + 1)]
    G_c[0] += h2[0] * G_S0
    sym0 = G_S0 + G_S0.T
    for k in range(1, M + 1):
        G_c[k] += h2[k] * sym0
    for tau, G_St in zip(taus, G_Sts):
        for k in range(0, tau + M + 1):
            G_c[k] += h2[abs(tau - k)] * G_St
        for kp in range(1, M - tau + 1):
            G_c[kp] += h2[kp + tau] * G_St.T
    lam = G_c[K].copy()
    G_P = np.zeros((n, n))
    for k in range(K, 0, -1):
        G_P += lam @ c[k - 1].T
        lam = G_c[k - 1] + P.T @ lam
    G_Sigma = 0.5 * (lam + lam.T)
    _, G_A_P = la.expm_frechet(A.T * tr, G_P, compute_expm=True)
    G_A = tr * G_A_P
    Lam = la.solve_continuous_lyapunov(A.T, -G_Sigma)
    return G_A + 2.0 * Lam @ Sigma


def loss_and_grad_moments_multi(S0: np.ndarray, Sts: Sequence[np.ndarray], N: int, FC_emp: np.ndarray, COV_emps: Sequence[np.ndarray],
                                w_fc: float = 1.0, w_tau: float = 1.0):
    """Loss over FC and several lagged correlations; gradients wrt S0 and each St."""
    n = S0.shape[0]
    Sxx = S0[:N, :N]
    d = np.clip(np.diag(Sxx), 1e-300, None)
    sd = np.sqrt(d)
    denom = np.outer(sd, sd)
    R = Sxx / denom
    od = offdiag_mask(N)
    dR = (R - FC_emp) * od
    loss = 0.5 * w_fc * np.sum(dR ** 2)
    G_R = w_fc * dR
    gdiag = -0.5 / d * ((G_R * R).sum(axis=1) + (G_R * R).sum(axis=0))
    G_Sxx = G_R / denom
    G_Sts, Cts = [], []
    w_l = w_tau / max(len(Sts), 1)
    for St in Sts:
        Txx = St[:N, :N]
        Ct = Txx / denom
        dCt = Ct - COV_emps[len(Cts)]
        loss += 0.5 * w_l * np.sum(dCt ** 2)
        G_Ct = w_l * dCt
        gdiag = gdiag - 0.5 / d * ((G_Ct * Ct).sum(axis=1) + (G_Ct * Ct).sum(axis=0))
        G_St = np.zeros((n, n))
        G_St[:N, :N] = G_Ct / denom
        G_Sts.append(G_St)
        Cts.append(Ct)
    G_Sxx = G_Sxx.copy()
    G_Sxx[np.arange(N), np.arange(N)] = gdiag
    G_S0 = np.zeros((n, n))
    G_S0[:N, :N] = G_Sxx
    return loss, G_S0, G_Sts, R, Cts


def grads_to_params(G_A: np.ndarray, N: int, G: float):
    """Map dL/dA to dL/dC (N x N, i != j), dL/da (N,), dL/domega (N,)."""
    G11 = G_A[:N, :N] + G_A[N:, N:]
    gC = G * (G11 - np.diag(G11)[:, None])
    np.fill_diagonal(gC, 0.0)
    ga = np.diag(G11).copy()
    gw = -np.diag(G_A[:N, N:]) + np.diag(G_A[N:, :N])
    return gC, ga, gw


@dataclass
class GradientFitResult:
    C: np.ndarray
    a: np.ndarray
    omega: np.ndarray
    loss: float
    n_iter: int
    n_fev: int
    success: bool
    message: str
    history: list[dict] = field(default_factory=list)
    FC_sim: np.ndarray | None = None
    COVtau_sim: np.ndarray | None = None
    metrics: dict = field(default_factory=dict)
    elapsed_s: float = 0.0
    bound_hits: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"loss": self.loss, "n_iter": self.n_iter, "n_fev": self.n_fev, "success": self.success,
                "message": str(self.message), "elapsed_s": self.elapsed_s, **self.metrics}


UNSTABLE_LOSS_FACTOR = 10.0  # penalty (x initial loss) returned for an unstable linearisation during the EC fit


def fit_linear_gradient(FC_emp: np.ndarray, COVtau_emp: np.ndarray, C0: np.ndarray, mask: np.ndarray, omega: np.ndarray,
                        tr: float, tau_tr: int = 1, a=-0.02, beta: float = 0.02, band=None, filter_order: int = 2,
                        w_fc: float = 1.0, w_tau: float = 1.0, lambda_sc: float = 0.0, C_prior: np.ndarray | None = None,
                        lambda_l1: float = 0.0, fit_a: str = "none", fit_omega: bool = False, c_max: float | None = None,
                        max_iter: int = 500, tol: float = 1e-9, verbose: int = 0, log_every: int = 25,
                        a_bounds=(-2.0, -1e-3)) -> GradientFitResult:
    """L-BFGS-B fit of C (>= 0 on mask) [+ a, omega] of the linear Hopf model (G is absorbed in C)."""
    t0 = time.time()
    N = C0.shape[0]
    # several lags: tau_tr a list and COVtau_emp a matching list of matrices
    taus = [int(t) for t in (tau_tr if isinstance(tau_tr, (list, tuple, np.ndarray)) else [tau_tr])]
    COV_list = list(COVtau_emp) if (isinstance(COVtau_emp, (list, tuple)) or (isinstance(COVtau_emp, np.ndarray) and COVtau_emp.ndim == 3)) else [COVtau_emp]
    if len(COV_list) != len(taus):
        raise ValueError(f"{len(COV_list)} lagged matrices for {len(taus)} lags")
    mask = np.asarray(mask, bool).copy()
    np.fill_diagonal(mask, False)
    idx = np.where(mask)
    n_c = len(idx[0])
    a0 = np.broadcast_to(np.asarray(a, float), (N,)).copy()
    omega0 = np.asarray(omega, float).copy()
    h2 = _filter_weights(tr, band, filter_order)
    prior = C_prior if C_prior is not None else C0
    n_a = 0 if fit_a == "none" else (1 if fit_a == "global" else N)
    n_w = N if fit_omega else 0

    def unpack(theta):
        C = np.zeros((N, N))
        C[idx] = theta[:n_c]
        pos = n_c
        if n_a == 1:
            av = np.full(N, theta[pos])
            pos += 1
        elif n_a == N:
            av = theta[pos : pos + N]
            pos += N
        else:
            av = a0
        w = theta[pos : pos + N] if n_w else omega0
        return C, av, w

    history: list[dict] = []
    state = {"fev": 0, "n_unstable": 0, "loss0": None}

    def _max_real_eig(A: np.ndarray) -> float:
        return float(np.max(np.real(la.eigvals(A))))

    def _penalty(theta):
        # Unstable linearisation (some a_j crossed its coupled in-strength sum_k C_jk while C moved): the
        # model has no stationary covariance there.  A large finite loss with zero gradient makes the
        # L-BFGS-B line search backtrack instead of propagating NaNs into the Frechet derivative.
        state["n_unstable"] += 1
        return UNSTABLE_LOSS_FACTOR * float(state["loss0"] or 1.0), np.zeros_like(theta)

    def fun(theta):
        C, av, w = unpack(theta)
        A = build_jacobian(C, 1.0, av, w)
        if not np.isfinite(A).all() or _max_real_eig(A) >= 0.0:
            return _penalty(theta)
        S0, Sts, cache = forward_moments_multi(A, beta, tr, taus, h2)
        loss, G_S0, G_Sts, R, Cts = loss_and_grad_moments_multi(S0, Sts, N, FC_emp, COV_list, w_fc, w_tau)
        if not (np.isfinite(loss) and np.isfinite(S0).all()):
            return _penalty(theta)
        if state["loss0"] is None:
            state["loss0"] = float(loss)
        Ct = Cts[0]
        G_A = backward_moments_multi(G_S0, G_Sts, A, tr, h2, cache)
        gC, ga, gw = grads_to_params(G_A, N, 1.0)
        if lambda_sc > 0:
            dP = (C - prior) * mask
            loss += 0.5 * lambda_sc * np.sum(dP ** 2)
            gC = gC + lambda_sc * dP
        if lambda_l1 > 0:
            loss += lambda_l1 * np.sum(C[idx])
            gC = gC + lambda_l1 * mask
        g = np.empty_like(theta)
        g[:n_c] = gC[idx]
        pos = n_c
        if n_a == 1:
            g[pos] = ga.sum()
            pos += 1
        elif n_a == N:
            g[pos : pos + N] = ga
            pos += N
        if n_w:
            g[pos : pos + N] = gw
        state["fev"] += 1
        state["last"] = (R, Ct, loss)
        return loss, g

    def cb(theta):
        R, Ct, loss = state["last"]
        m = fit_error(FC_emp, R, COV_list[0], Ct)
        m["iter"] = len(history)
        m["loss"] = float(loss)
        history.append(m)
        if verbose and len(history) % log_every == 0:
            LOG.info("grad-fit it %4d  loss=%.5g  fit_rmse=%.4f  fc_corr=%.3f", len(history), loss, m["fit_rmse"], m["fc_corr"])

    theta0 = np.concatenate([C0[idx]] + ([np.array([a0.mean()])] if n_a == 1 else []) + ([a0] if n_a == N else []) + ([omega0] if n_w else []))
    C_0, av_0, w_0 = unpack(theta0)
    A0 = build_jacobian(C_0, 1.0, av_0, w_0)
    mre0 = _max_real_eig(A0)
    if not np.isfinite(mre0) or mre0 >= 0.0:
        raise ValueError(f"linear gradient fit: the linearisation is unstable at the starting point (max Re eig {mre0:.3g} >= 0; "
                         f"max a_j = {a0.max():.3g}, {int(np.sum(a0 > 0))} supercritical node(s)); lower a0 / beta or the coupling G")
    hi = c_max if c_max is not None else None
    bounds = [(0.0, hi)] * n_c + [tuple(a_bounds)] * n_a + [(1e-4, None)] * n_w
    res = minimize(fun, theta0, jac=True, method="L-BFGS-B", bounds=bounds, callback=cb,
                   options={"maxiter": int(max_iter), "ftol": tol, "gtol": 1e-10, "maxcor": 20})
    C, av, w = unpack(res.x)
    A = build_jacobian(C, 1.0, av, w)
    S0, Sts, _ = forward_moments_multi(A, beta, tr, taus, h2)
    _, _, _, R, Cts = loss_and_grad_moments_multi(S0, Sts, N, FC_emp, COV_list, w_fc, w_tau)
    Ct = Cts[0]
    out = GradientFitResult(C=C, a=av, omega=w, loss=float(res.fun), n_iter=int(res.nit), n_fev=int(state["fev"]),
                            success=bool(res.success), message=str(res.message), history=history, FC_sim=R, COVtau_sim=Ct,
                            elapsed_s=time.time() - t0)
    out.metrics = fit_error(FC_emp, R, COV_list[0], Ct)
    out.metrics["n_unstable_evaluations"] = int(state["n_unstable"])
    if state["n_unstable"]:
        LOG.warning("linear gradient fit: %d/%d evaluations hit an unstable linearisation and were rejected "
                    "(the a_j profile is close to the validity limit of the linear model)", state["n_unstable"], state["fev"] + state["n_unstable"])
    if len(taus) > 1:
        out.metrics["tau_rmse_all_lags"] = float(np.mean([rmse(COV_list[k], Cts[k]) for k in range(len(taus))]))
        out.COVtau_sim_all = Cts
    if history:
        out.metrics["initial_fit_rmse"] = history[0]["fit_rmse"]
        out.metrics["initial_fc_corr"] = history[0]["fc_corr"]
    # parameters that ended on a box constraint (validity guard: report, do not hide)
    tol = 1e-9
    hits = {"n_C_at_zero": int(np.sum(res.x[:n_c] <= tol)), "n_C_at_cmax": int(np.sum(res.x[:n_c] >= c_max - tol)) if c_max is not None else 0,
            "n_a_at_lower": 0, "n_a_at_upper": 0, "n_omega_at_lower": 0}
    if n_a:
        av_fit = res.x[n_c : n_c + n_a]
        hits["n_a_at_lower"] = int(np.sum(av_fit <= a_bounds[0] + tol))
        hits["n_a_at_upper"] = int(np.sum(av_fit >= a_bounds[1] - tol))
    if n_w:
        hits["n_omega_at_lower"] = int(np.sum(res.x[n_c + n_a :] <= 1e-4 + tol))
    out.bound_hits = hits
    for k in ("n_a_at_lower", "n_a_at_upper", "n_omega_at_lower", "n_C_at_cmax"):
        out.metrics[k] = hits[k]
    if hits["n_a_at_lower"] or hits["n_a_at_upper"] or hits["n_omega_at_lower"] or hits["n_C_at_cmax"]:
        LOG.warning("gradient fit: parameters at their bounds %s (a bounds %s, omega >= 1e-4, C <= c_max); widen a_bounds / c_max if unintended",
                    {k: v for k, v in hits.items() if v and k != "n_C_at_zero"}, a_bounds)
    return out
