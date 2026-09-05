"""Whittle (frequency-domain maximum) likelihood fit of the linear Hopf model.

For the TR-sampled linear model the cross-spectral density of x is the upper-left block of
S(f) = H(f) Q H(f)^H,  H(f) = (2 pi i f I - A)^-1,  Q = diag(beta_j^2) (both x and y blocks).
Given the periodogram matrices I(f) of the (filtered, standardised) data at the Fourier frequencies
inside the pass-band, the Whittle negative log-likelihood is

    L = sum_f [ log det S(f) + tr(S(f)^-1 I(f)) ],

which uses all lags at once (the FC / lagged-covariance fit uses lags 0 and tau only).  Gradients with
respect to C, a, omega and the per-node noise amplitudes are analytic:
    dL/dS = S^-1 - S^-1 I S^-1 ,   dL/dA = 2 Re[(H Q H^H P H)^T],   P = E_x^T (dL/dS) E_x .
Per-node noise amplitudes are fitted (default) so that the unit variances of z-scored data do not
constrain the coupling.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import linalg as la
from scipy.optimize import minimize

from ..utils import LOG
from .gec import fit_error
from .hopf_linear import analytic_moments, build_jacobian
from .linear_gradient import grads_to_params


def periodogram_matrices(ts: np.ndarray, tr: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided periodogram cross-spectral matrices I[f] = X(f) X(f)^H * tr / T  (freqs, F x N x N complex)."""
    x = np.asarray(ts, float)
    x = x - x.mean(axis=0, keepdims=True)
    T = x.shape[0]
    X = np.fft.rfft(x, axis=0)  # F x N
    freqs = np.fft.rfftfreq(T, d=tr)
    I = np.einsum("fi,fj->fij", X, np.conj(X)) * (tr / T)
    return freqs, I


def average_periodograms(items: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Average periodogram matrices over runs/participants (interpolated onto the coarsest common grid)."""
    grids = [f for f, _ in items]
    ref = max(grids, key=lambda g: (g[1] - g[0]) if len(g) > 1 else 0)
    acc = np.zeros((len(ref),) + items[0][1].shape[1:], complex)
    for f, I in items:
        if len(f) == len(ref) and np.allclose(f, ref):
            acc += I
        else:  # bin-average onto the reference grid
            idx = np.clip(np.searchsorted(ref, f), 0, len(ref) - 1)
            for k in range(len(ref)):
                sel = idx == k
                if sel.any():
                    acc[k] += I[sel].mean(axis=0)
    return ref, acc / len(items)


def band_selection(freqs: np.ndarray, band, shrink: float = 0.1) -> np.ndarray:
    lo, hi = band
    lo = 0.0 if lo is None else float(lo)
    hi = freqs[-1] if hi is None else float(hi)
    w = hi - lo
    sel = (freqs >= lo + shrink * w) & (freqs <= hi - shrink * w) & (freqs > 0)
    return np.where(sel)[0]


def filter_gain(freqs: np.ndarray, tr: float, band, order: int = 2) -> np.ndarray:
    """|B(f)|^2 of the zero-phase (filtfilt) Butterworth band-pass at the given frequencies (1 if no band)."""
    from scipy import signal as sps

    from .signal import butter_coeffs

    ba = butter_coeffs(tr, band, order)
    if ba is None:
        return np.ones(len(freqs))
    b, a = ba
    _, h = sps.freqz(b, a, worN=2 * np.pi * np.asarray(freqs) * tr)
    return np.abs(h) ** 4  # filtfilt applies the filter twice


def whittle_loss_grad(C: np.ndarray, a, omega: np.ndarray, q: np.ndarray, freqs: np.ndarray, I: np.ndarray, return_S: bool = False,
                      gain: np.ndarray | None = None, tr: float | None = None, n_alias: int = 0):
    """Whittle loss and gradients (dC, da, domega, dq) with q_j = beta_j^2 (noise variance of node j).

    gain[f] multiplies the model spectrum (filter response |B|^2 of the data's band-pass); n_alias > 0 adds the
    aliased images S(f + n/TR), n = -n_alias..n_alias, of the TR-sampled process."""
    N = C.shape[0]
    A = build_jacobian(C, 1.0, a, omega)
    Qd = np.concatenate([q, q])
    loss = 0.0
    G_A = np.zeros((2 * N, 2 * N))
    g_q = np.zeros(N)
    S_all = [] if return_S else None
    I2 = np.eye(2 * N)
    gain = np.ones(len(freqs)) if gain is None else gain
    shifts = [0.0] if (n_alias <= 0 or tr is None) else [n / tr for n in range(-int(n_alias), int(n_alias) + 1)]
    for f, If, gf in zip(freqs, I, gain):
        Hs, HQs = [], []
        S_full = np.zeros((2 * N, 2 * N), complex)
        for sh in shifts:
            M = 2j * np.pi * (f + sh) * I2 - A
            H = la.solve(M, I2)
            HQ = H * Qd[None, :]
            S_full += HQ @ H.conj().T
            Hs.append(H)
            HQs.append(HQ)
        S_full *= gf
        S = S_full[:N, :N]
        S = 0.5 * (S + S.conj().T)
        try:
            cf = la.cho_factor(S)
        except la.LinAlgError:
            S = S + 1e-12 * np.trace(S).real / N * np.eye(N)
            cf = la.cho_factor(S)
        Sinv = la.cho_solve(cf, np.eye(N))
        logdet = 2.0 * np.sum(np.log(np.diag(cf[0]).real))
        SinvI = Sinv @ If
        loss += logdet + np.trace(SinvI).real
        G_S = Sinv - SinvI @ Sinv  # dL/dS (Hermitian)
        P = np.zeros((2 * N, 2 * N), complex)
        P[:N, :N] = G_S * gf
        for H, HQ in zip(Hs, HQs):
            HHP = H.conj().T @ P          # H^H P
            G_A += 2.0 * np.real((HQ @ HHP @ H).T)
            # noise variances: dS_full/dq_j = H (e_j e_j^T + e_{N+j} e_{N+j}^T) H^H -> dL/dq_j = Re[(H^H P H)_{jj} + (H^H P H)_{N+j,N+j}]
            HPH = np.real(np.einsum("ij,ji->i", HHP, H))  # diag of H^H P H
            g_q += HPH[:N] + HPH[N:]
        if return_S:
            S_all.append(S)
    gC, ga, gw = grads_to_params(G_A, N, 1.0)
    out = (float(loss), gC, ga, gw, g_q)
    return out + (np.array(S_all),) if return_S else out


@dataclass
class WhittleFitResult:
    C: np.ndarray
    a: np.ndarray
    omega: np.ndarray
    beta: np.ndarray
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


def fit_linear_whittle(ts_list: list[np.ndarray] | None, tr: float, C0: np.ndarray, mask: np.ndarray, omega: np.ndarray, band,
                       a=-0.02, beta: float = 0.02, periodogram: tuple | None = None, fit_a: str = "none", fit_omega: bool = True,
                       fit_beta: str = "node", lambda_prior: float = 0.0, C_prior: np.ndarray | None = None, max_iter: int = 500,
                       verbose: int = 0, FC_emp: np.ndarray | None = None, COVtau_emp: np.ndarray | None = None, tau_tr: int = 1,
                       a_bounds=(-2.0, -1e-3), filter_gain_correction: bool = True, n_alias: int = 1) -> WhittleFitResult:
    """Maximum Whittle likelihood over C (>= 0 on mask) [+ a, omega, per-node noise]; periodogram from ts_list or given."""
    t0 = time.time()
    N = C0.shape[0]
    if periodogram is None:
        items = [periodogram_matrices(x, tr) for x in ts_list]
        freqs, I = average_periodograms(items)
    else:
        freqs, I = periodogram
    sel = band_selection(freqs, band)
    if len(sel) < 3:
        raise ValueError("too few Fourier frequencies inside the band for a Whittle fit (need longer runs)")
    fsel, Isel = freqs[sel], I[sel]
    gain = filter_gain(fsel, tr, band) if (filter_gain_correction and band is not None) else None
    mask = np.asarray(mask, bool).copy()
    np.fill_diagonal(mask, False)
    idx = np.where(mask)
    n_c = len(idx[0])
    a0 = np.broadcast_to(np.asarray(a, float), (N,)).copy()
    w0 = np.asarray(omega, float).copy()
    n_a = 0 if fit_a == "none" else (1 if fit_a == "global" else N)
    n_w = N if fit_omega else 0
    n_q = N if fit_beta == "node" else (1 if fit_beta == "global" else 0)
    q0 = np.full(N, float(beta) ** 2)
    prior = C_prior if C_prior is not None else C0

    def unpack(theta):
        C = np.zeros((N, N))
        C[idx] = theta[:n_c]
        pos = n_c
        if n_a == 1:
            av = np.full(N, theta[pos]); pos += 1
        elif n_a == N:
            av = theta[pos:pos + N]; pos += N
        else:
            av = a0
        if n_w:
            w = theta[pos:pos + N]; pos += N
        else:
            w = w0
        if n_q == N:
            q = np.exp(theta[pos:pos + N]); pos += N
        elif n_q == 1:
            q = np.full(N, np.exp(theta[pos])); pos += 1
        else:
            q = q0
        return C, av, w, q

    state = {"fev": 0}
    history: list[dict] = []

    def fun(theta):
        C, av, w, q = unpack(theta)
        loss, gC, ga, gw, gq = whittle_loss_grad(C, av, w, q, fsel, Isel, gain=gain, tr=tr, n_alias=n_alias)
        if lambda_prior > 0:
            d = (C - prior) * mask
            loss += 0.5 * lambda_prior * np.sum(d ** 2)
            gC = gC + lambda_prior * d
        g = np.empty_like(theta)
        g[:n_c] = gC[idx]
        pos = n_c
        if n_a == 1:
            g[pos] = ga.sum(); pos += 1
        elif n_a == N:
            g[pos:pos + N] = ga; pos += N
        if n_w:
            g[pos:pos + N] = gw; pos += N
        if n_q == N:
            g[pos:pos + N] = gq * q; pos += N      # d/d log q
        elif n_q == 1:
            g[pos] = float(np.sum(gq * q)); pos += 1
        state["fev"] += 1
        state["last_loss"] = loss
        return loss, g

    def cb(theta):
        history.append({"iter": len(history), "loss": float(state.get("last_loss", np.nan))})
        if verbose and len(history) % 25 == 0:
            LOG.info("whittle it %4d loss=%.6g", len(history), history[-1]["loss"])

    theta0 = np.concatenate([C0[idx]] + ([np.array([a0.mean()])] if n_a == 1 else []) + ([a0] if n_a == N else [])
                            + ([w0] if n_w else []) + ([np.log(q0)] if n_q == N else []) + ([np.array([np.log(q0[0])])] if n_q == 1 else []))
    bounds = [(0.0, None)] * n_c + [tuple(a_bounds)] * n_a + [(1e-4, None)] * n_w + [(np.log(1e-8), np.log(1e2))] * n_q
    res = minimize(fun, theta0, jac=True, method="L-BFGS-B", bounds=bounds, callback=cb,
                   options={"maxiter": int(max_iter), "ftol": 1e-12, "gtol": 1e-8, "maxcor": 20})
    C, av, w, q = unpack(res.x)
    out = WhittleFitResult(C=C, a=av, omega=w, beta=np.sqrt(q), loss=float(res.fun), n_iter=int(res.nit), n_fev=state["fev"],
                           success=bool(res.success), message=str(res.message), history=history, elapsed_s=time.time() - t0)
    # comparability with the moment fits: model FC / lagged correlation at the optimum (mean noise amplitude)
    FC_m, COV_m = analytic_moments(C, 1.0, av, w, tr, tau_tr, float(np.sqrt(q.mean())), filt={"band": band})
    out.FC_sim, out.COVtau_sim = FC_m, COV_m
    if FC_emp is not None and COVtau_emp is not None:
        out.metrics = fit_error(FC_emp, FC_m, COVtau_emp, COV_m)
    out.metrics["whittle_loss"] = float(res.fun)
    out.metrics["n_frequencies"] = int(len(sel))
    tol = 1e-9
    hits = {"n_a_at_lower": int(np.sum(av <= a_bounds[0] + tol)) if n_a else 0, "n_a_at_upper": int(np.sum(av >= a_bounds[1] - tol)) if n_a else 0,
            "n_omega_at_lower": int(np.sum(w <= 1e-4 + tol)) if n_w else 0}
    out.bound_hits = hits
    out.metrics.update(hits)
    return out
