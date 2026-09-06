"""Linear (Ornstein-Uhlenbeck) approximation of the Hopf whole-brain model.

Linearising  dz_j/dt = (a_j + i w_j - |z_j|^2) z_j + G sum_k C_jk (z_k - z_j) + beta eta_j
around z = 0 (valid while A is stable; all a_j < 0 suffices) gives dX = A X dt + beta dW with X = (x, y) and

    A = [[diag(a - G s) + G C,  -diag(w)],
         [diag(w),               diag(a - G s) + G C]],   s_j = sum_k C_jk.

The stationary covariance solves the Lyapunov equation A S + S A' + beta^2 I = 0 and the
lagged covariance is Cov(X(t+tau), X(t)) = expm(A tau) S.  Model FC and lagged correlation
are read from the x-blocks.  With ``filt`` given, the effect of the zero-phase band-pass filter
applied to the empirical BOLD is reproduced exactly for the TR-sampled process
(Ponce-Alvarez & Deco, Sci Rep 2024, extended here with filter-consistent moments).
"""
from __future__ import annotations

import numpy as np
from scipy import linalg as la

from .signal import filter_autocorrelation


def build_jacobian(C: np.ndarray, G: float, a, omega: np.ndarray) -> np.ndarray:
    C = np.asarray(C, float)
    N = C.shape[0]
    wC = G * C
    np.fill_diagonal(wC, 0.0)
    s = wC.sum(axis=1)
    a = np.broadcast_to(np.asarray(a, float), (N,))
    A11 = wC + np.diag(a - s)
    W = np.diag(np.asarray(omega, float))
    A = np.block([[A11, -W], [W, A11]])
    return A


def lyapunov_covariance(A: np.ndarray, beta: float) -> np.ndarray:
    n = A.shape[0]
    Q = (beta ** 2) * np.eye(n)
    S = la.solve_continuous_lyapunov(A, -Q)
    return 0.5 * (S + S.T)


def _corr_from_cov(Sxx: np.ndarray, Cxy: np.ndarray | None = None) -> np.ndarray:
    sd = np.sqrt(np.clip(np.diag(Sxx), 1e-300, None))
    M = Sxx if Cxy is None else Cxy
    R = M / np.outer(sd, sd)
    if Cxy is None:
        np.fill_diagonal(R, 1.0)
    return R


def analytic_moments(C: np.ndarray, G: float, a, omega: np.ndarray, tr: float, tau_tr: int = 1,
                     beta: float = 0.02, filt: dict | None = None, return_extra: bool = False):
    """Model FC and lagged correlation (lag = tau_tr * tr seconds) of the linear Hopf model.

    filt: None (no filtering) or {"band": [lo, hi], "order": 2} to reproduce the band-pass
    applied to the empirical data (filtfilt on TR-sampled signals).
    """
    N = C.shape[0]
    A = build_jacobian(C, G, a, omega)
    ev = None
    if filt is None or filt.get("band") is None:
        S = lyapunov_covariance(A, beta)
        P = la.expm(A * (tau_tr * tr))
        Stau = P @ S
        Sxx = S[:N, :N]
        FC = _corr_from_cov(Sxx)
        COV = _corr_from_cov(Sxx, Stau[:N, :N])
        extra = {"Sigma": S, "stable": bool(np.all(np.real(la.eigvals(A)) < 0))} if return_extra else None
    else:
        h2 = filter_autocorrelation(tr, filt["band"], filt.get("order", 2))
        S = lyapunov_covariance(A, beta)
        lam, V = la.eig(A)
        if np.any(np.real(lam) >= 0):
            # unstable linearisation: fall back to unfiltered moments to keep the fit going
            FC, COV = analytic_moments(C, G, a, omega, tr, tau_tr, beta, None)
            return (FC, COV, {"stable": False, "Sigma": S}) if return_extra else (FC, COV)
        Vinv = la.inv(V)
        m = np.arange(len(h2))
        # exponents e^{lam * k * tr} for k = 0..M+tau (k >= 0 only; magnitudes <= 1)
        M = len(h2) - 1
        kmax = M + int(tau_tr)
        E = np.exp(np.outer(lam, np.arange(kmax + 1) * tr))  # (2N, kmax+1)

        def filt_cov(tau: int) -> np.ndarray:
            # sum_{m} h2(|m|) c(tau - m) for m = -M..M ; c(k) = V diag(e^{lam k tr}) Vinv S  (k>=0), c(-k) = c(k)^T
            ms = np.arange(-M, M + 1)
            h = h2[np.abs(ms)]
            k = tau - ms
            pos = k >= 0
            Dp = (E[:, k[pos]] * h[pos]).sum(axis=1)          # (2N,)
            Dn = (E[:, -k[~pos]] * h[~pos]).sum(axis=1) if np.any(~pos) else np.zeros_like(lam)
            T1 = (V * Dp) @ Vinv @ S
            T2 = S @ ((V * Dn) @ Vinv).T
            out = T1 + T2
            return np.real(out)

        S0 = filt_cov(0)
        St = filt_cov(int(tau_tr))
        Sxx = 0.5 * (S0[:N, :N] + S0[:N, :N].T)
        FC = _corr_from_cov(Sxx)
        COV = _corr_from_cov(Sxx, St[:N, :N])
        extra = {"Sigma": S, "Sigma_filt": S0, "stable": True, "eig": lam} if return_extra else None
    FC = np.clip(FC, -1.0, 1.0)
    if return_extra:
        return FC, COV, extra
    return FC, COV


def analytic_power_spectrum(C, G, a, omega, freqs: np.ndarray, beta: float = 0.02) -> np.ndarray:
    """Power spectral density of x (per node) of the linear model at the given frequencies (Hz)."""
    N = C.shape[0]
    A = build_jacobian(C, G, a, omega)
    out = np.zeros((len(freqs), N))
    I = np.eye(2 * N)
    for i, f in enumerate(freqs):
        H = la.solve(1j * 2 * np.pi * f * I - A, I)
        S = (beta ** 2) * (H @ H.conj().T)
        out[i] = np.real(np.diag(S)[:N])
    return out


def linear_stability(C, G, a, omega) -> dict:
    A = build_jacobian(C, G, a, omega)
    ev = la.eigvals(A)
    return {"max_real_eig": float(np.max(np.real(ev))), "stable": bool(np.all(np.real(ev) < 0))}
