"""Generative effective connectivity (GEC) fitting shared by the linear and non-linear models.

Iterative heuristic gradient rule (Deco, Kringelbach et al.):

    C_ij <- C_ij + eps_fc (FC_emp_ij - FC_sim_ij) + eps_tau (COVtau_emp_ij - COVtau_sim_ij)

restricted to allowed links (structural links + homotopic pairs, or all), clipped at zero and
(optionally) rescaled so that max(C) stays fixed.  The best iterate (lowest fitting error) is
kept; iterations stop when the error has not improved for `patience` iterations.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..utils import LOG, corr_upper, offdiag_mask, rmse


def make_mask(SC: np.ndarray | None, N: int, mode: str = "sc_plus_homotopic",
              homotopic_pairs: Sequence[tuple[int, int]] | None = None) -> np.ndarray:
    """Boolean N x N mask of links allowed to change (diagonal always False)."""
    if mode == "full" or SC is None:
        mask = np.ones((N, N), bool)
    else:
        mask = np.asarray(SC) > 0
        if mode == "sc_plus_homotopic" and homotopic_pairs:
            for i, j in homotopic_pairs:
                mask[i, j] = True
                mask[j, i] = True
    mask = mask.copy()
    np.fill_diagonal(mask, False)
    return mask


def initial_ec(SC: np.ndarray | None, N: int, mask: np.ndarray, init: str = "sc", sc_max: float = 0.2) -> np.ndarray:
    if init == "sc" and SC is not None:
        C = np.array(SC, float)
        # give homotopic links absent from SC a small starting weight so they can grow
        C[(C == 0) & mask] = 0.1 * sc_max
    elif init == "uniform":
        C = np.full((N, N), 0.5 * sc_max)
    else:
        C = np.zeros((N, N))
    C[~mask] = 0.0
    np.fill_diagonal(C, 0.0)
    return C


@dataclass
class GECResult:
    C: np.ndarray
    G: float
    history: list[dict] = field(default_factory=list)
    FC_sim: np.ndarray | None = None
    COVtau_sim: np.ndarray | None = None
    best_iter: int = 0
    n_iter: int = 0
    converged: bool = False
    elapsed_s: float = 0.0
    metrics: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "G": self.G,
            "best_iter": self.best_iter,
            "n_iter": self.n_iter,
            "converged": self.converged,
            "elapsed_s": self.elapsed_s,
            **self.metrics,
        }


def fit_error(FC_emp, FC_sim, COV_emp, COV_sim) -> dict:
    N = FC_emp.shape[0]
    od = offdiag_mask(N)
    e_fc = rmse(FC_emp, FC_sim, od)
    e_tau = rmse(COV_emp, COV_sim)
    return {
        "fc_rmse": e_fc,
        "tau_rmse": e_tau,
        "fit_rmse": float(np.sqrt(0.5 * (e_fc ** 2 + e_tau ** 2))),
        "fc_corr": corr_upper(FC_emp, FC_sim),
        "tau_corr": float(np.corrcoef(COV_emp.ravel(), COV_sim.ravel())[0, 1]) if COV_sim.std() > 0 else float("nan"),
    }


def _project(C: np.ndarray, mask: np.ndarray, normalize_max: float | None) -> np.ndarray:
    C = np.array(C, float)
    C[~mask] = 0.0
    np.fill_diagonal(C, 0.0)
    C[C < 0] = 0.0
    if normalize_max:
        mx = C.max()
        if mx > 0:
            C *= normalize_max / mx
    return C


def fit_gec(FC_emp: np.ndarray, COVtau_emp: np.ndarray, C0: np.ndarray, moments_fn: Callable[[np.ndarray], tuple],
            mask: np.ndarray, G: float = 1.0, eps_fc: float = 1e-3, eps_tau: float = 1e-3, max_iter: int = 1000,
            min_iter: int = 20, patience: int = 50, normalize_max: float | None = 0.2, verbose: int = 0,
            callback: Callable[[int, dict], None] | None = None, log_every: int = 50, accept_only_improving: bool = False,
            eps_decay: float = 0.5, eps_min_frac: float = 1e-3) -> GECResult:
    """Run the GEC iteration.  moments_fn(C) -> (FC_sim, COVtau_sim).

    accept_only_improving: a proposed update is kept only if it lowers the fitting error (evaluated with the
    same moments_fn, i.e. with common random numbers for simulated moments); otherwise the step sizes are
    multiplied by eps_decay and the proposal repeated, until they fall below eps_min_frac of their start."""
    t0 = time.time()
    C = _project(C0, mask, normalize_max)
    best_err = np.inf
    best_C = C.copy()
    best_it = 0
    best_mom = None
    history: list[dict] = []
    stall = 0
    converged = False
    e_fc, e_tau = float(eps_fc), float(eps_tau)
    FC_sim, COV_sim = moments_fn(C)
    n_rejected = 0
    for it in range(int(max_iter)):
        if not (np.all(np.isfinite(FC_sim)) and np.all(np.isfinite(COV_sim))):
            LOG.warning("GEC: non-finite model moments at iteration %d; stopping", it)
            break
        m = fit_error(FC_emp, FC_sim, COVtau_emp, COV_sim)
        m["iter"] = it
        m["C_sum"] = float(C.sum())
        m["eps_fc"] = e_fc
        history.append(m)
        if callback:
            callback(it, m)
        if verbose and (it % log_every == 0):
            LOG.info("GEC it %4d  fit_rmse=%.4f  fc_corr=%.3f  fc_rmse=%.4f  tau_rmse=%.4f", it, m["fit_rmse"], m["fc_corr"], m["fc_rmse"], m["tau_rmse"])
        if m["fit_rmse"] < best_err - 1e-12:
            best_err = m["fit_rmse"]
            best_C = C.copy()
            best_it = it
            best_mom = (FC_sim, COV_sim)
            stall = 0
        else:
            stall += 1
            if stall >= patience and it >= min_iter:
                converged = True
                break
        # proposal
        C_new = _project(C + e_fc * (FC_emp - FC_sim) + e_tau * (COVtau_emp - COV_sim), mask, normalize_max)
        FC_new, COV_new = moments_fn(C_new)
        if accept_only_improving:
            err_new = fit_error(FC_emp, FC_new, COVtau_emp, COV_new)["fit_rmse"]
            if not np.isfinite(err_new) or err_new >= m["fit_rmse"]:
                n_rejected += 1
                e_fc *= eps_decay
                e_tau *= eps_decay
                if e_fc < eps_fc * eps_min_frac:
                    converged = True
                    LOG.info("GEC: step size exhausted after %d iterations (%d rejected proposals)", it + 1, n_rejected)
                    break
                continue  # same C, smaller steps
        C, FC_sim, COV_sim = C_new, FC_new, COV_new
    res = GECResult(C=best_C, G=G, history=history, best_iter=best_it, n_iter=len(history), converged=converged,
                    elapsed_s=time.time() - t0)
    if best_mom is not None:
        res.FC_sim, res.COVtau_sim = best_mom
        res.metrics = fit_error(FC_emp, res.FC_sim, COVtau_emp, res.COVtau_sim)
        res.metrics["initial_fit_rmse"] = history[0]["fit_rmse"] if history else float("nan")
        res.metrics["initial_fc_corr"] = history[0]["fc_corr"] if history else float("nan")
    res.metrics["n_rejected"] = n_rejected
    return res
