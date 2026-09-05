"""Non-linear Hopf effective connectivity with a surrogate gradient.

The non-linear model is simulated (common random numbers: the same noise realisation at every
iteration, so that differences between iterations reflect the coupling and not the noise).  Its
residuals (FC_sim - FC_emp, COVtau_sim - COVtau_emp) are back-propagated through the *linear* model's
adjoint at the current coupling, which gives an approximate but genuine descent direction for the
non-linear loss.  Steps are accepted only if they lower the (common-random-number) error; the step
size adapts (x1.3 on success, x0.5 on rejection).
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from ..utils import LOG
from .gec import GECResult, _project, fit_error
from .hopf_linear import build_jacobian
from .linear_gradient import _filter_weights, backward_moments, forward_moments, grads_to_params, loss_and_grad_moments


def fit_nonlinear_surrogate(FC_emp: np.ndarray, COVtau_emp: np.ndarray, C0: np.ndarray, moments_fn: Callable[[np.ndarray], tuple],
                            mask: np.ndarray, G: float, a, omega: np.ndarray, tr: float, tau_tr: int = 1, beta: float = 0.02,
                            band=None, filter_consistent: bool = True, max_iter: int = 200, patience: int = 15,
                            step_frac: float = 0.05, step_max: float = 0.25, step_min: float = 1e-4, normalize_max: float | None = 0.2,
                            w_fc: float = 1.0, w_tau: float = 1.0, verbose: int = 0, log_every: int = 10) -> GECResult:
    """moments_fn(C) -> (FC_sim, COVtau_sim) of the non-linear model at coupling G*C (deterministic seed)."""
    t0 = time.time()
    N = C0.shape[0]
    a_vec = np.broadcast_to(np.asarray(a, float), (N,))
    h2 = _filter_weights(tr, band if filter_consistent else None)
    C = _project(C0, mask, normalize_max)
    FC_sim, COV_sim = moments_fn(C)
    m = fit_error(FC_emp, FC_sim, COVtau_emp, COV_sim)
    err = m["fit_rmse"]
    best_err, best_C, best_it, best_mom = err, C.copy(), 0, (FC_sim, COV_sim)
    history = [{**m, "iter": 0, "step_frac": step_frac, "C_sum": float(C.sum())}]
    stall = 0
    n_rej = 0
    it = 0
    while it < int(max_iter):
        # surrogate gradient of the non-linear residual through the linear Jacobian at W = G*C
        A = build_jacobian(G * C, 1.0, a_vec, omega)
        S0, St, cache = forward_moments(A, beta, tr, tau_tr, h2)
        _, G_S0, G_St, _, _ = loss_and_grad_moments(S0, St, N, FC_emp, COVtau_emp, w_fc, w_tau,
                                                    residuals=(FC_sim - FC_emp, COV_sim - COVtau_emp))
        G_A = backward_moments(G_S0, G_St, A, tr, tau_tr, h2, cache)
        gC, _, _ = grads_to_params(G_A, N, 1.0)
        gC = G * gC
        gC[~mask] = 0.0
        gmax = float(np.max(np.abs(gC)))
        if not np.isfinite(gmax) or gmax == 0:
            LOG.warning("surrogate fit: zero/non-finite gradient; stopping")
            break
        eta = step_frac * max(C.max(), 1e-9) / gmax          # largest change = step_frac of the largest coupling
        C_new = _project(C - eta * gC, mask, normalize_max)
        FC_new, COV_new = moments_fn(C_new)
        m_new = fit_error(FC_emp, FC_new, COVtau_emp, COV_new)
        it += 1
        if np.isfinite(m_new["fit_rmse"]) and m_new["fit_rmse"] < err:
            C, FC_sim, COV_sim, err = C_new, FC_new, COV_new, m_new["fit_rmse"]
            step_frac = min(step_frac * 1.3, step_max)
            history.append({**m_new, "iter": it, "step_frac": step_frac, "C_sum": float(C.sum())})
            if err < best_err - 1e-12:
                best_err, best_C, best_it, best_mom = err, C.copy(), it, (FC_sim, COV_sim)
                stall = 0
            else:
                stall += 1
        else:
            n_rej += 1
            step_frac *= 0.5
            stall += 1
            history.append({**m, "iter": it, "step_frac": step_frac, "C_sum": float(C.sum()), "rejected": 1})
            if step_frac < step_min:
                LOG.info("surrogate fit: step size exhausted after %d iterations (%d rejected)", it, n_rej)
                break
        if verbose and it % log_every == 0:
            LOG.info("surrogate it %4d  fit_rmse=%.4f  fc_corr=%.3f  step=%.3g  rejected=%d", it, err, history[-1]["fc_corr"], step_frac, n_rej)
        if stall >= patience:
            break
    res = GECResult(C=best_C, G=G, history=history, best_iter=best_it, n_iter=len(history), converged=(step_frac < step_min or stall >= patience),
                    elapsed_s=time.time() - t0)
    res.FC_sim, res.COVtau_sim = best_mom
    res.metrics = fit_error(FC_emp, res.FC_sim, COVtau_emp, res.COVtau_sim)
    res.metrics["initial_fit_rmse"] = history[0]["fit_rmse"]
    res.metrics["initial_fc_corr"] = history[0]["fc_corr"]
    res.metrics["n_rejected"] = n_rej
    return res


def noise_floor(moments_fn_seeded: Callable[[np.ndarray, int], tuple], C: np.ndarray, FC_emp, COV_emp, n_seeds: int = 3, seed0: int = 1000) -> dict:
    """SD of the fitting error across noise realisations at a fixed coupling (simulation-noise floor)."""
    errs = []
    for k in range(int(n_seeds)):
        FC, COV = moments_fn_seeded(C, seed0 + k)
        errs.append(fit_error(FC_emp, FC, COV_emp, COV)["fit_rmse"])
    return {"noise_floor_mean": float(np.mean(errs)), "noise_floor_sd": float(np.std(errs)), "noise_floor_n": int(n_seeds)}
