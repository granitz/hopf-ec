"""Non-linear Hopf whole-brain model: Euler-Maruyama simulation (numba-accelerated when available).

dx_j = [(a_j - x_j^2 - y_j^2) x_j - w_j y_j + G sum_k C_jk (x_k - x_j)] dt + beta dW
dy_j = [(a_j - x_j^2 - y_j^2) y_j + w_j x_j + G sum_k C_jk (y_k - y_j)] dt + beta dW
(Deco et al., Sci Rep 2017).  x is taken as the BOLD-like signal, sampled every TR.
"""
from __future__ import annotations

import numpy as np

from ..utils import LOG
from .signal import bandpass, functional_connectivity, lagged_correlation

try:  # optional acceleration
    import numba as _nb

    _HAVE_NUMBA = True
except Exception:  # noqa: BLE001
    _nb = None
    _HAVE_NUMBA = False


def have_numba() -> bool:
    return _HAVE_NUMBA


def _integrate_py(x, y, wC, s, a, omega, dt, dsig, noise, rec_idx, out, out_start, linear):
    N = x.shape[0]
    r = 0
    nrec = rec_idx.shape[0]
    for t in range(noise.shape[0]):
        cx = wC @ x - s * x
        cy = wC @ y - s * y
        if linear:
            r2 = 0.0
        else:
            r2 = x * x + y * y
        dx = (a - r2) * x - omega * y + cx
        dy = (a - r2) * y + omega * x + cy
        x += dt * dx + dsig * noise[t, :N]
        y += dt * dy + dsig * noise[t, N:]
        if r < nrec and rec_idx[r] == t:
            out[out_start + r, :] = x
            r += 1
    return r


if _HAVE_NUMBA:

    @_nb.njit(cache=True, fastmath=True)
    def _integrate_nb(x, y, wC, s, a, omega, dt, dsig, noise, rec_idx, out, out_start, linear):  # pragma: no cover
        N = x.shape[0]
        r = 0
        nrec = rec_idx.shape[0]
        cx = np.empty(N)
        cy = np.empty(N)
        for t in range(noise.shape[0]):
            for j in range(N):
                accx = 0.0
                accy = 0.0
                for k in range(N):
                    w = wC[j, k]
                    if w != 0.0:
                        accx += w * x[k]
                        accy += w * y[k]
                cx[j] = accx - s[j] * x[j]
                cy[j] = accy - s[j] * y[j]
            for j in range(N):
                if linear:
                    r2 = 0.0
                else:
                    r2 = x[j] * x[j] + y[j] * y[j]
                dx = (a[j] - r2) * x[j] - omega[j] * y[j] + cx[j]
                dy = (a[j] - r2) * y[j] + omega[j] * x[j] + cy[j]
                x[j] += dt * dx + dsig * noise[t, j]
                y[j] += dt * dy + dsig * noise[t, N + j]
            if r < nrec and rec_idx[r] == t:
                for j in range(N):
                    out[out_start + r, j] = x[j]
                r += 1
        return r


def simulate_hopf(C: np.ndarray, G: float, a, omega: np.ndarray, tr: float, n_vol: int, dt: float = 0.1,
                  beta: float = 0.02, seed: int | None = None, transient_s: float = 100.0, linear: bool = False,
                  use_numba: bool | str = "auto", chunk_steps: int = 4000, x0: np.ndarray | None = None) -> np.ndarray:
    """Simulate the Hopf model and return x sampled every TR, shape (n_vol, N).

    The integration step is adjusted so that one TR is an integer number of steps.
    """
    C = np.asarray(C, float)
    N = C.shape[0]
    wC = G * C
    np.fill_diagonal(wC, 0.0)
    s = wC.sum(axis=1)
    a = np.ascontiguousarray(np.broadcast_to(np.asarray(a, float), (N,)))
    omega = np.ascontiguousarray(np.asarray(omega, float))
    # Explicit Euler needs dt * (|a| + G s_j + w_j) well below 1 for accuracy/stability:
    # refine the step automatically (dt is treated as the maximum step) and log it.
    rate = float(np.max(np.abs(a) + s + np.abs(omega)))
    dt_max = dt if rate <= 0 else min(dt, 0.25 / rate)
    if dt_max < dt * 0.999:
        LOG.debug("Hopf simulation: reducing dt from %.3g to %.3g s (max rate %.3g 1/s)", dt, dt_max, rate)
    steps_per_tr = max(1, int(np.ceil(tr / dt_max - 1e-9)))
    dt_eff = tr / steps_per_tr
    n_trans = int(round(transient_s / dt_eff))
    total = n_trans + n_vol * steps_per_tr
    rec_all = n_trans + steps_per_tr * np.arange(n_vol) + (steps_per_tr - 1)
    rng = np.random.default_rng(seed)
    dsig = np.sqrt(dt_eff) * beta
    if x0 is None:
        x = 0.1 * rng.standard_normal(N)
        y = 0.1 * rng.standard_normal(N)
    else:
        x = np.array(x0[0], float)
        y = np.array(x0[1], float)
    out = np.empty((n_vol, N))
    use_nb = _HAVE_NUMBA if use_numba == "auto" else bool(use_numba and _HAVE_NUMBA)
    kernel = _integrate_nb if use_nb else _integrate_py
    done = 0
    start = 0
    while start < total:
        n = min(chunk_steps, total - start)
        noise = rng.standard_normal((n, 2 * N))
        sel = rec_all[(rec_all >= start) & (rec_all < start + n)] - start
        r = kernel(x, y, wC, s, a, omega, dt_eff, dsig, noise, sel.astype(np.int64), out, done, linear)
        done += r
        start += n
        if not np.all(np.isfinite(x)):
            raise FloatingPointError("Hopf simulation diverged (non-finite state); reduce dt or G")
    return out


def simulated_moments(C, G, a, omega, tr, n_vol, band, tau_tr, beta=0.02, dt=0.1, n_sim=1, seed=0,
                      transient_s=100.0, linear=False, use_numba="auto", return_ts=False):
    """Average FC and lagged correlation over n_sim simulations (band-pass filtered like the data)."""
    fcs, covs, tss = [], [], []
    for k in range(int(n_sim)):
        x = simulate_hopf(C, G, a, omega, tr, n_vol, dt=dt, beta=beta, seed=None if seed is None else seed + k,
                          transient_s=transient_s, linear=linear, use_numba=use_numba)
        xf = bandpass(x, tr, band)
        fcs.append(functional_connectivity(xf))
        covs.append(lagged_correlation(xf, tau_tr))
        if return_ts:
            tss.append(xf)
    FC = np.mean(fcs, axis=0)
    COV = np.mean(covs, axis=0)
    if return_ts:
        return FC, COV, tss
    return FC, COV
