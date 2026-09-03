"""Parameter search (global coupling G x bifurcation a) with an error surface, run in parallel."""
from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..utils import LOG, corr_upper, offdiag_mask, rmse
from .gec import fit_error
from .hopf_linear import analytic_moments
from .hopf_nonlinear import simulate_hopf, simulated_moments
from .signal import bandpass, fcd_distribution, functional_connectivity, ks_distance, lagged_correlation, metastability

METRICS = ("fit_rmse", "fc_rmse", "fc_corr", "tau_rmse", "fcd_ks", "meta_diff", "combined")
LOWER_IS_BETTER = {"fit_rmse": True, "fc_rmse": True, "fc_corr": False, "tau_rmse": True, "fcd_ks": True,
                   "meta_diff": True, "combined": True}


def _eval_point(model: str, G: float, a, SC, omega, emp: dict, tr: float, band, tau_tr: int, beta: float, dt: float,
                n_sim: int, seed: int, transient_s: float, filter_consistent: bool, need_fcd: bool, fcd_cfg: dict,
                use_numba) -> dict:
    FC_emp, COV_emp = emp["FC"], emp["COVtau"]
    out = {"G": float(G), "a": float(a) if np.isscalar(a) else float(np.mean(a))}
    if model == "linear" and not need_fcd:
        FC, COV = analytic_moments(SC, G, a, omega, tr, tau_tr, beta, filt={"band": band} if filter_consistent else None)
        out.update(fit_error(FC_emp, FC, COV_emp, COV))
        return out
    n_vol = int(emp.get("n_volumes", 500))
    FC, COV, tss = simulated_moments(SC, G, a, omega, tr, n_vol, band, tau_tr, beta=beta, dt=dt, n_sim=n_sim, seed=seed,
                                     transient_s=transient_s, linear=(model == "linear"), use_numba=use_numba, return_ts=True)
    out.update(fit_error(FC_emp, FC, COV_emp, COV))
    if need_fcd:
        ks = []
        metas = []
        for xf in tss:
            ks.append(ks_distance(emp["fcd"], fcd_distribution(xf, fcd_cfg.get("window_tr", 30), fcd_cfg.get("step_tr", 3))))
            metas.append(metastability(xf))
        out["fcd_ks"] = float(np.mean(ks))
        out["meta_sim"] = float(np.mean(metas))
        out["meta_diff"] = float(abs(np.mean(metas) - emp.get("metastability", np.nan)))
    return out


def grid_search(model: str, SC: np.ndarray, omega: np.ndarray, emp: dict, tr: float, G_values: Sequence[float],
                a_values: Sequence | None, band, tau_tr: int = 1, beta: float = 0.02, dt: float = 0.1,
                n_sim: int = 1, seed: int = 0, transient_s: float = 100.0, filter_consistent: bool = True,
                metric: str = "fit_rmse", fcd_cfg: dict | None = None, n_jobs: int = 1, use_numba="auto",
                default_a=-0.02) -> tuple[pd.DataFrame, dict]:
    """Evaluate the homogeneous model (C = SC) on a grid of G (and a); return long table + best point.

    emp must contain FC, COVtau, n_volumes and, for fcd_ks / meta metrics, fcd and metastability.
    """
    fcd_cfg = fcd_cfg or {}
    need_fcd = metric in ("fcd_ks", "meta_diff", "combined")
    a_vals = list(a_values) if a_values is not None and len(a_values) else [default_a]
    points = list(itertools.product([float(g) for g in G_values], a_vals))
    LOG.info("%s model: parameter search over %d points (%d G x %d a), metric=%s, n_jobs=%s", model, len(points), len(G_values), len(a_vals), metric, n_jobs)
    rows = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_eval_point)(model, G, a, SC, omega, emp, tr, band, tau_tr, beta, dt, n_sim, seed, transient_s,
                             filter_consistent, need_fcd, fcd_cfg, use_numba)
        for G, a in points
    )
    df = pd.DataFrame(rows)
    if metric == "combined":
        df["combined"] = (1 - df["fc_corr"]) + df.get("fcd_ks", 0.0)
    best = select_best(df, metric)
    return df, best


def select_best(df: pd.DataFrame, metric: str) -> dict:
    if metric not in df.columns:
        raise KeyError(f"metric {metric!r} not in search results ({list(df.columns)})")
    col = df[metric].astype(float)
    idx = int(col.idxmin() if LOWER_IS_BETTER.get(metric, True) else col.idxmax())
    row = df.loc[idx].to_dict()
    row["metric"] = metric
    return row


def error_surface(df: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pivot the long table into (G_values, a_values, surface[G, a])."""
    Gs = np.array(sorted(df["G"].unique()))
    As = np.array(sorted(df["a"].unique()))
    surf = np.full((len(Gs), len(As)), np.nan)
    gi = {g: i for i, g in enumerate(Gs)}
    ai = {a: i for i, a in enumerate(As)}
    for _, r in df.iterrows():
        surf[gi[r["G"]], ai[r["a"]]] = r[metric]
    return Gs, As, surf
