"""Parameter search over the global coupling G (and bifurcation parameter a) of the homogeneous
model (C = SC), with

* validity guards (unstable linearisation of the linear model, G = 0),
* detection of optima on the border of the grid and automatic extension of the grid,
* coarse-to-fine refinement around the optimum with parabolic interpolation,
* optional continuous optimisation of (G, a): L-BFGS-B with analytic gradients for the linear
  model and the fit_rmse metric, derivative-free (Powell) otherwise.

All evaluations run in parallel with joblib.
"""
from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize

from ..utils import LOG
from .gec import fit_error
from .hopf_linear import analytic_moments, build_jacobian, linear_stability
from .hopf_nonlinear import simulated_moments
from .signal import fcd_distribution, ks_distance, metastability

METRICS = ("fit_rmse", "fc_rmse", "fc_corr", "tau_rmse", "fcd_ks", "meta_diff", "combined", "ndte_corr", "ndte_rmse")
LOWER_IS_BETTER = {"fit_rmse": True, "fc_rmse": True, "fc_corr": False, "tau_rmse": True, "fcd_ks": True,
                   "meta_diff": True, "combined": True, "ndte_corr": False, "ndte_rmse": True}
_NAN_METRICS = {k: float("nan") for k in ("fc_rmse", "tau_rmse", "fit_rmse", "fc_corr", "tau_corr", "fcd_ks", "meta_sim", "meta_diff", "ndte_corr", "ndte_rmse")}

SEARCH_DEFAULTS = {
    "border_action": "extend",   # extend | warn | ignore
    "extend_factor": 0.5,        # fraction of the axis span appended beyond the hit edge (same step)
    "max_extensions": 2,
    "G_min": 0.0, "G_max": 10.0,
    "a_min": -1.0, "a_max": 0.5,
    "allow_zero_G": False,       # G = 0 (uncoupled model) is evaluated but never selected
    "refine": True,              # second pass with a finer step around the optimum
    "refine_points": 7,          # points per axis in the refinement window (+- one coarse step)
    "interpolate": True,         # parabolic interpolation through the optimum and its neighbours
    "continuous": False,         # continuous optimisation of (G, a) after the grid stages
    "continuous_method": "auto", # auto: L-BFGS-B (linear model, fit_rmse) else Powell | pso (particle swarm)
    "continuous_max_fev": 60,
    "pso": {"n_particles": 20, "n_iter": 30, "stall_iter": 8},
}


def linear_validity(SC, G, a, omega, require_subcritical: bool = True) -> tuple[bool, str]:
    """Validity of a linear-model point.  Default: every node below its bifurcation (a_j < 0), the regime in which
    the linearisation of the Hopf model is meaningful; with require_subcritical=False only the stability of the
    coupled Jacobian is required (a node with a_j > 0 can be held stable by its coupling)."""
    a_vec = np.broadcast_to(np.asarray(a, float), (np.asarray(SC).shape[0],))
    if require_subcritical and np.any(a_vec >= 0):
        return False, (f"{int(np.sum(a_vec >= 0))} supercritical node(s) (a_j >= 0): the linear model is used below the bifurcation only "
                       "(model.linear.require_subcritical)")
    st = linear_stability(SC, G, a, omega)
    if not st["stable"]:
        return False, f"unstable linearisation (max Re eig {st['max_real_eig']:.3g} >= 0; needs a_j < G*sum_k C_jk)"
    return True, ""


# --------------------------------------------------------------------------- single point
def _eval_point(model: str, G: float, a, SC, omega, emp: dict, tr: float, band, tau_tr: int, beta: float, dt: float,
                n_sim: int, seed: int, transient_s: float, filter_consistent: bool, need_fcd: bool, fcd_cfg: dict,
                use_numba, a_fn=None, require_subcritical: bool = True) -> dict:
    FC_emp, COV_emp = emp["FC"], emp["COVtau"]
    out = {"G": float(G), "a": float(a) if np.isscalar(a) else float(np.mean(a)), "valid": True, "reason": ""}
    if a_fn is not None and np.isscalar(a):
        a = a_fn(float(a))  # heterogeneity: axis value (e.g. beta) -> node vector a_j
    if model == "linear":
        ok, reason = linear_validity(SC, G, a, omega, require_subcritical)
        if not ok:
            out.update(_NAN_METRICS)
            out["valid"] = False
            out["reason"] = reason
            return out
    need_ndte = "NDTE" in emp and emp["NDTE"] is not None
    if model == "linear" and not need_fcd:
        FC, COV = analytic_moments(SC, G, a, omega, tr, tau_tr, beta, filt={"band": band} if filter_consistent else None)
        out.update(fit_error(FC_emp, FC, COV_emp, COV))
        if need_ndte:
            out.update(linear_ndte_metrics(SC, G, a, omega, tr, beta, band if filter_consistent else None, emp["NDTE"], int(emp.get("ndte_max_lag", 10))))
        return out
    n_vol = int(emp.get("n_volumes", 500))
    FC, COV, tss = simulated_moments(SC, G, a, omega, tr, n_vol, band, tau_tr, beta=beta, dt=dt, n_sim=n_sim, seed=seed,
                                     transient_s=transient_s, linear=(model == "linear"), use_numba=use_numba, return_ts=True)
    out.update(fit_error(FC_emp, FC, COV_emp, COV))
    if need_ndte:
        from .ndte import ndte, ndte_similarity

        nd = np.mean([ndte(xf, int(emp.get("ndte_max_lag", 10))) for xf in tss], axis=0)
        out.update(ndte_similarity(emp["NDTE"], nd))
    if need_fcd:
        ks, metas = [], []
        for xf in tss:
            ks.append(ks_distance(emp["fcd"], fcd_distribution(xf, fcd_cfg.get("window_tr", 30), fcd_cfg.get("step_tr", 3))))
            metas.append(metastability(xf))
        out["fcd_ks"] = float(np.mean(ks))
        out["meta_sim"] = float(np.mean(metas))
        out["meta_diff"] = float(abs(np.mean(metas) - emp.get("metastability", np.nan)))
    return out


def linear_ndte_metrics(C, G, a, omega, tr, beta, band, ndte_emp, max_lag: int = 10) -> dict:
    """NDTE of the linear model (analytic, filter-consistent lagged covariances) vs the empirical NDTE."""
    from .linear_gradient import _filter_weights, forward_moments_multi
    from .ndte import ndte_linear_model, ndte_similarity

    N = C.shape[0]
    A = build_jacobian(C, G, a, omega)
    h2 = _filter_weights(tr, band)
    S0, Sts, _ = forward_moments_multi(A, beta, tr, list(range(1, int(max_lag))), h2)
    nd = ndte_linear_model([S0[:N, :N]] + [St[:N, :N] for St in Sts])
    return ndte_similarity(ndte_emp, nd)


def _finalize(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric == "combined":
        df["combined"] = (1 - df["fc_corr"]) + df.get("fcd_ks", 0.0)
    return df


def evaluate_points(model: str, points: Sequence[tuple[float, float]], SC, omega, emp, tr, band, tau_tr=1, beta=0.02,
                    dt=0.1, n_sim=1, seed=0, transient_s=100.0, filter_consistent=True, metric="fit_rmse", fcd_cfg=None,
                    n_jobs=1, use_numba="auto", stage: str = "coarse", a_fn=None, require_subcritical: bool = True) -> pd.DataFrame:
    fcd_cfg = fcd_cfg or {}
    need_fcd = metric in ("fcd_ks", "meta_diff", "combined")
    rows = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_eval_point)(model, G, a, SC, omega, emp, tr, band, tau_tr, beta, dt, n_sim, seed, transient_s,
                             filter_consistent, need_fcd, fcd_cfg, use_numba, a_fn, require_subcritical)
        for G, a in points
    )
    df = pd.DataFrame(rows)
    df["stage"] = stage
    return _finalize(df, metric)


def grid_search(model: str, SC: np.ndarray, omega: np.ndarray, emp: dict, tr: float, G_values: Sequence[float],
                a_values: Sequence | None, band, tau_tr: int = 1, beta: float = 0.02, dt: float = 0.1,
                n_sim: int = 1, seed: int = 0, transient_s: float = 100.0, filter_consistent: bool = True,
                metric: str = "fit_rmse", fcd_cfg: dict | None = None, n_jobs: int = 1, use_numba="auto",
                default_a=-0.02, allow_zero_G: bool = True) -> tuple[pd.DataFrame, dict]:
    """Plain single-pass grid evaluation of the homogeneous model; returns (long table, best point)."""
    a_vals = list(a_values) if a_values is not None and len(a_values) else [default_a]
    points = list(itertools.product([float(g) for g in G_values], a_vals))
    LOG.info("%s model: parameter search over %d points (%d G x %d a), metric=%s, n_jobs=%s", model, len(points), len(G_values), len(a_vals), metric, n_jobs)
    df = evaluate_points(model, points, SC, omega, emp, tr, band, tau_tr, beta, dt, n_sim, seed, transient_s, filter_consistent,
                         metric, fcd_cfg, n_jobs, use_numba)
    best = select_best(df, metric, allow_zero_G=allow_zero_G)
    return df, best


# --------------------------------------------------------------------------- selection & borders
def select_best(df: pd.DataFrame, metric: str, allow_zero_G: bool = False) -> dict:
    """Best valid point (NaN / invalid points ignored; G = 0 excluded unless allowed or nothing else is valid)."""
    if metric not in df.columns:
        raise KeyError(f"metric {metric!r} not in search results ({list(df.columns)})")
    col = df[metric].astype(float)
    ok = col.notna() & df.get("valid", pd.Series(True, index=df.index)).astype(bool)
    if not ok.any():
        raise RuntimeError("parameter search: no valid point (check model.a / the search ranges; the linear model needs a stable linearisation)")
    cand = ok & (df["G"] > 0) if not allow_zero_G else ok
    if not cand.any():
        LOG.warning("parameter search: only G = 0 (uncoupled model) is valid - the data cannot constrain the coupling")
        cand = ok
    sub = col[cand]
    idx = int(sub.idxmin() if LOWER_IS_BETTER.get(metric, True) else sub.idxmax())
    row = df.loc[idx].to_dict()
    row["metric"] = metric
    if not allow_zero_G and ok.any():
        full = col[ok]
        idx_all = int(full.idxmin() if LOWER_IS_BETTER.get(metric, True) else full.idxmax())
        if float(df.loc[idx_all, "G"]) == 0.0 and idx_all != idx:
            LOG.warning("parameter search: G = 0 gave the best %s but is excluded (uncoupled model); using G = %.3g", metric, row["G"])
            row["zero_G_was_best"] = True
    return row


def border_flags(best: dict, df: pd.DataFrame) -> dict:
    """Per axis: {'low': None|'edge'|'invalid', 'high': ...} - 'edge' = no evaluated value beyond the optimum,
    'invalid' = the neighbouring value on that side is invalid (unstable).  Plus summary booleans."""
    flags: dict = {}
    valid = df.get("valid", pd.Series(True, index=df.index)).astype(bool) & df[best["metric"]].notna()
    for axis in ("G", "a"):
        vals = np.array(sorted(df[axis].unique()))
        if len(vals) < 2:
            flags[axis] = {"low": None, "high": None}
            continue
        v = float(best[axis])
        lower = vals[vals < v - 1e-12]
        upper = vals[vals > v + 1e-12]
        side: dict = {}
        for name, neigh in (("low", lower[-1] if len(lower) else None), ("high", upper[0] if len(upper) else None)):
            if neigh is None:
                side[name] = "edge"
            elif not valid[np.isclose(df[axis], neigh)].any():
                side[name] = "invalid"
            else:
                side[name] = None
        flags[axis] = side
    flags["on_border"] = any(flags[ax][sd] == "edge" for ax in ("G", "a") for sd in ("low", "high"))
    flags["at_validity_limit"] = any(flags[ax][sd] == "invalid" for ax in ("G", "a") for sd in ("low", "high"))
    return flags


def _axis_step(vals: np.ndarray) -> float:
    d = np.diff(np.sort(vals))
    d = d[d > 1e-12]
    return float(np.min(d)) if len(d) else 0.0


def extend_axis(vals: np.ndarray, side: str, factor: float, lo_cap: float, hi_cap: float, step: float | None = None,
                span: float | None = None) -> np.ndarray:
    """New values beyond one edge with the (coarse) step, factor*span of them (at least one), within the caps."""
    vals = np.sort(np.asarray(vals, float))
    step = float(step) if step else _axis_step(vals)
    if step <= 0:
        return np.array([])
    span = float(span) if span is not None else float(vals[-1] - vals[0])
    n_new = max(1, int(round(factor * span / step)))
    if side == "high":
        new = vals[-1] + step * np.arange(1, n_new + 1)
        new = new[new <= hi_cap + 1e-12]
    else:
        new = vals[0] - step * np.arange(1, n_new + 1)
        new = new[new >= lo_cap - 1e-12]
    return np.round(new, 10)


def refine_axis(vals: np.ndarray, center: float, n_points: int, lo_cap: float, hi_cap: float, step: float | None = None) -> np.ndarray:
    """Fine values within +- one coarse *step* of *center* (caps respected), excluding already known values."""
    vals = np.sort(np.asarray(vals, float))
    step = float(step) if step else _axis_step(vals)
    if step <= 0 or n_points < 3:
        return np.array([])
    lo, hi = max(center - step, lo_cap), min(center + step, hi_cap)
    fine = np.linspace(lo, hi, int(n_points))
    fine = np.array([f for f in fine if not np.any(np.isclose(vals, f, atol=1e-9))])
    return np.round(fine, 10)


def parabolic_vertex(x: np.ndarray, y: np.ndarray, i: int, lower_is_better: bool = True):
    """Vertex of the parabola through points i-1, i, i+1 (unequal spacing).  None if not bracketed/convex."""
    if i <= 0 or i >= len(x) - 1:
        return None
    x0, x1, x2 = float(x[i - 1]), float(x[i]), float(x[i + 1])
    y0, y1, y2 = float(y[i - 1]), float(y[i]), float(y[i + 1])
    if not all(np.isfinite([y0, y1, y2])):
        return None
    denom = (x1 - x0) * (y1 - y2) - (x1 - x2) * (y1 - y0)
    if abs(denom) < 1e-18:
        return None
    xv = x1 - 0.5 * ((x1 - x0) ** 2 * (y1 - y2) - (x1 - x2) ** 2 * (y1 - y0)) / denom
    # curvature sign: fit y = c2 x^2 + ..., require minimum (or maximum) at the vertex
    c2 = ((y2 - y1) / (x2 - x1) - (y1 - y0) / (x1 - x0)) / (x2 - x0)
    if (lower_is_better and c2 <= 0) or (not lower_is_better and c2 >= 0):
        return None
    if not (x0 < xv < x2):
        return None
    yv = y1 - 0.25 * ((x1 - x0) * (y1 - y2) - (x1 - x2) * (y1 - y0)) ** 2 / denom / ((x1 - x0) * (x1 - x2)) if False else None
    return float(xv)


def _validated_used(model: str, used: dict, fallback: dict, SC, omega, a_fn, info: dict, require_subcritical: bool = True) -> dict:
    """The interpolated / continuous optimum is off the evaluated grid: for the linear model make sure its
    linearisation is stable, otherwise fall back to the (validated) grid optimum."""
    if model != "linear" or used.get("source") in ("grid", "refined_grid"):
        return used
    a = a_fn(float(used["a"])) if a_fn is not None else float(used["a"])
    ok, reason = linear_validity(SC, float(used["G"]), a, omega, require_subcritical)
    if ok:
        return used
    msg = (f"{used['source']} optimum (G={used['G']:.4g}, a={used['a']:.4g}) is invalid ({reason}); "
           f"using the grid optimum (G={fallback['G']:.4g}, a={fallback['a']:.4g}) instead")
    LOG.warning(msg)
    info["warnings"].append(msg)
    return {"G": float(fallback["G"]), "a": float(fallback["a"]), "source": f"refined_grid ({used['source']} point unstable)"}


def interpolate_optimum(df: pd.DataFrame, best: dict, metric: str) -> dict:
    """Axis-wise parabolic interpolation through the best point on the finest available grid."""
    out = {"G": float(best["G"]), "a": float(best["a"]), "interpolated_axes": []}
    lib = LOWER_IS_BETTER.get(metric, True)
    for axis, other in (("G", "a"), ("a", "G")):
        line = df[np.isclose(df[other], best[other]) & df[metric].notna()].sort_values(axis)
        if len(line) < 3:
            continue
        x = line[axis].to_numpy(float)
        y = line[metric].to_numpy(float)
        i = int(np.argmin(np.abs(x - float(best[axis]))))
        xv = parabolic_vertex(x, y, i, lib)
        if xv is not None:
            out[axis] = xv
            out["interpolated_axes"].append(axis)
    return out


# --------------------------------------------------------------------------- continuous optimisation
def continuous_search(model: str, SC, omega, emp, tr, G0: float, a0: float, band, fit_a: bool, metric: str,
                      bounds_G: tuple[float, float], bounds_a: tuple[float, float], tau_tr=1, beta=0.02, dt=0.1, n_sim=1,
                      seed=0, transient_s=100.0, filter_consistent=True, fcd_cfg=None, method: str = "auto",
                      max_fev: int = 60, use_numba="auto", a_fn=None, a_grad=None, pso_cfg: dict | None = None, require_subcritical: bool = True) -> dict:
    """Continuous optimisation of G (and a) starting from the grid optimum.

    Linear model + fit_rmse: exact loss/gradient (adjoint) with L-BFGS-B.  Otherwise Powell on the metric
    with a fixed simulation seed (deterministic objective)."""
    N = SC.shape[0]
    lo_G = max(float(bounds_G[0]), 1e-3)
    hi_G = float(bounds_G[1])
    lo_a, hi_a = float(bounds_a[0]), float(bounds_a[1])
    if model == "linear" and a_fn is None:
        hi_a = min(hi_a, -1e-3)
    to_vec = (lambda v: a_fn(float(v))) if a_fn is not None else (lambda v: float(v))
    grad_vec = a_grad if a_grad is not None else (lambda v: np.ones(N))   # d a_j / d(axis value)
    G0 = float(np.clip(G0, lo_G, hi_G))
    a0 = float(np.clip(a0, lo_a, hi_a))
    use_grad = (method == "auto" and model == "linear" and metric == "fit_rmse") or method == "lbfgs"
    x0 = np.array([G0, a0]) if fit_a else np.array([G0])
    bounds = [(lo_G, hi_G)] + ([(lo_a, hi_a)] if fit_a else [])
    nfev = {"n": 0}
    if method == "pso":
        from .pso import pso_minimize

        lib = LOWER_IS_BETTER.get(metric, True)
        pso_cfg = pso_cfg or {}

        def fun_pso(x):
            G = float(x[0])
            a = float(x[1]) if fit_a else a0
            r = _eval_point(model, G, a, SC, omega, emp, tr, band, tau_tr, beta, dt, n_sim, seed, transient_s, filter_consistent,
                            metric in ("fcd_ks", "meta_diff", "combined"), fcd_cfg or {}, use_numba, a_fn, require_subcritical)
            if metric == "combined":
                r["combined"] = (1 - r["fc_corr"]) + r.get("fcd_ks", 0.0)
            v = r.get(metric, np.nan)
            if not np.isfinite(v):
                return 1e6
            return v if lib else -v

        res = pso_minimize(fun_pso, bounds, n_particles=int(pso_cfg.get("n_particles", 20)), n_iter=int(pso_cfg.get("n_iter", 30)), x0=x0,
                           seed=seed, n_jobs=int(pso_cfg.get("n_jobs", 1)), stall_iter=int(pso_cfg.get("stall_iter", 8)))
        value = float(res["fun"] if lib else -res["fun"])
        G_opt = float(res["x"][0])
        a_opt = float(res["x"][1]) if fit_a else a0
        hits = {"G": "low" if np.isclose(G_opt, lo_G) else "high" if np.isclose(G_opt, hi_G) else None,
                "a": ("low" if np.isclose(a_opt, lo_a) else "high" if np.isclose(a_opt, hi_a) else None) if fit_a else None}
        return {"G": G_opt, "a": a_opt, metric: value, "method": "PSO (particle swarm)", "success": bool(res["converged"]), "n_fev": int(res["n_fev"]),
                "message": f"{res['n_iter']} swarm iterations", "bound_hits": hits, "history": res["history"]}
    if use_grad:
        from .linear_gradient import _filter_weights, backward_moments, forward_moments, grads_to_params, loss_and_grad_moments

        h2 = _filter_weights(tr, band if filter_consistent else None)
        # loss = N^2 * fit_rmse^2 (scaled so that gradients are O(1) for the optimiser's tolerances)
        scale = float(N * N)
        w_fc, w_tau = scale / (N * (N - 1)), 1.0

        def fun(x):
            G = float(x[0])
            av = float(x[1]) if fit_a else a0
            a_vec = to_vec(av)
            A = build_jacobian(G * SC, 1.0, a_vec, omega)
            if np.max(np.real(np.linalg.eigvals(A))) >= 0:   # outside the linear model's validity: large penalty
                nfev["n"] += 1
                return 1e6, np.zeros(len(x))
            S0, St, cache = forward_moments(A, beta, tr, tau_tr, h2)
            loss, G_S0, G_St, _, _ = loss_and_grad_moments(S0, St, N, emp["FC"], emp["COVtau"], w_fc, w_tau)
            G_A = backward_moments(G_S0, G_St, A, tr, tau_tr, h2, cache)
            gC, ga, _ = grads_to_params(G_A, N, 1.0)
            g = [float(np.sum(gC * SC))] + ([float(np.sum(ga * grad_vec(av)))] if fit_a else [])
            nfev["n"] += 1
            return loss, np.array(g)

        res = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 300, "maxfun": max(int(max_fev), 100), "ftol": 1e-14, "gtol": 1e-10})
        value = float(np.sqrt(max(res.fun, 0.0) / scale))  # back to fit_rmse
        used = "L-BFGS-B (analytic gradient)"
    else:
        lib = LOWER_IS_BETTER.get(metric, True)

        def fun(x):
            G = float(x[0])
            a = float(x[1]) if fit_a else a0
            r = _eval_point(model, G, a, SC, omega, emp, tr, band, tau_tr, beta, dt, n_sim, seed, transient_s, filter_consistent,
                            metric in ("fcd_ks", "meta_diff", "combined"), fcd_cfg or {}, use_numba, a_fn, require_subcritical)
            if metric == "combined":
                r["combined"] = (1 - r["fc_corr"]) + r.get("fcd_ks", 0.0)
            v = r.get(metric, np.nan)
            nfev["n"] += 1
            if not np.isfinite(v):
                return 1e6
            return v if lib else -v

        res = minimize(fun, x0, method="Powell", bounds=bounds, options={"maxfev": int(max_fev), "xtol": 1e-3, "ftol": 1e-4})
        value = float(res.fun if lib else -res.fun)
        used = "Powell (derivative-free)"
    G_opt = float(res.x[0])
    a_opt = float(res.x[1]) if fit_a else a0
    hits = {"G": "low" if np.isclose(G_opt, lo_G) else "high" if np.isclose(G_opt, hi_G) else None,
            "a": ("low" if np.isclose(a_opt, lo_a) else "high" if np.isclose(a_opt, hi_a) else None) if fit_a else None}
    if any(hits.values()):
        LOG.warning("continuous search: optimum at a bound (%s); consider widening search.G_max / a_min / a_max", hits)
    return {"G": G_opt, "a": a_opt, metric: value, "method": used, "success": bool(res.success), "n_fev": nfev["n"],
            "message": str(res.message), "bound_hits": hits}


# --------------------------------------------------------------------------- orchestration
def adaptive_search(model: str, SC, omega, emp, tr, G_values, a_values, band, search_cfg: dict | None = None, tau_tr=1,
                    beta=0.02, dt=0.1, n_sim=1, seed=0, transient_s=100.0, filter_consistent=True, metric="fit_rmse",
                    fcd_cfg=None, n_jobs=1, use_numba="auto", default_a=-0.02, a_fn=None, a_grad=None,
                    a_axis: str = "a", require_subcritical: bool = True) -> tuple[pd.DataFrame, dict, dict]:
    """Grid search + border handling + extension + refinement + interpolation [+ continuous optimisation].

    Returns (table of all evaluated points, best point actually to be used, info dict for reporting)."""
    s = {**SEARCH_DEFAULTS, **(search_cfg or {})}
    kw = dict(tau_tr=tau_tr, beta=beta, dt=dt, n_sim=n_sim, seed=seed, transient_s=transient_s, filter_consistent=filter_consistent,
              metric=metric, fcd_cfg=fcd_cfg, n_jobs=n_jobs, use_numba=use_numba, a_fn=a_fn, require_subcritical=require_subcritical)
    G_vals = np.round(np.array([float(g) for g in G_values]), 10)
    a_grid = a_values is not None and len(a_values) > 0
    a_vals = np.round(np.asarray(a_values, float), 10) if a_grid else np.array([float(default_a)])
    if model == "linear" and a_grid and a_fn is None and np.any(a_vals >= 0):
        LOG.warning("linear model: grid points with a >= 0 are %s", "outside the validity of the linearisation and marked invalid (model.linear.require_subcritical)"
                    if require_subcritical else "kept only where the coupled Jacobian is stable")
    info_axis = {"a_axis": a_axis}
    if 0.0 in G_vals and not s["allow_zero_G"]:
        LOG.info("G = 0 is on the grid: evaluated for the error surface but never selected (uncoupled model)")
    info: dict = {"metric": metric, "stages": [], "warnings": [], "extensions": []}
    points = list(itertools.product(G_vals, a_vals))
    LOG.info("%s model: coarse search over %d points (%d G x %d a), metric=%s", model, len(points), len(G_vals), len(a_vals), metric)
    df = evaluate_points(model, points, SC, omega, emp, tr, band, stage="coarse", **kw)
    info["stages"].append({"stage": "coarse", "n_points": len(points)})
    best = select_best(df, metric, allow_zero_G=s["allow_zero_G"])
    best_coarse = dict(best)
    caps = {"G": (float(s["G_min"]), float(s["G_max"])), "a": (float(s["a_min"]), float(s["a_max"]))}
    coarse_step = {"G": _axis_step(G_vals), "a": _axis_step(a_vals) if a_grid else 0.0}
    coarse_span = {"G": float(G_vals.max() - G_vals.min()), "a": float(a_vals.max() - a_vals.min()) if a_grid else 0.0}

    def _refine(df, best):
        fine_G = refine_axis(np.array(sorted(df["G"].unique())), float(best["G"]), int(s["refine_points"]), *caps["G"], step=coarse_step["G"]) if coarse_step["G"] > 0 else np.array([])
        fine_a = refine_axis(np.array(sorted(df["a"].unique())), float(best["a"]), int(s["refine_points"]), *caps["a"], step=coarse_step["a"]) if a_grid else np.array([])
        G_set = np.unique(np.concatenate([fine_G, [float(best["G"])]]))
        a_set = np.unique(np.concatenate([fine_a, [float(best["a"])]])) if a_grid else np.array([float(best["a"])])
        pts = [(g, a) for g, a in itertools.product(G_set, a_set) if not ((df["G"].sub(g).abs() < 1e-9) & (df["a"].sub(a).abs() < 1e-9)).any()]
        if not pts:
            return df, best
        LOG.info("refinement around G=%.3g a=%.3g: %d points", best["G"], best["a"], len(pts))
        df = pd.concat([df, evaluate_points(model, pts, SC, omega, emp, tr, band, stage="refined", **kw)], ignore_index=True)
        info["stages"].append({"stage": "refined", "n_points": len(pts)})
        return df, select_best(df, metric, allow_zero_G=s["allow_zero_G"])

    n_ext = 0
    while True:
        flags = border_flags(best, df)
        if not flags["on_border"] or s["border_action"] == "ignore":
            break
        if s["border_action"] != "extend" or n_ext >= int(s["max_extensions"]):
            msg = "optimum on the border of the search grid (" + ", ".join(f"{ax} {sd}" for ax in ("G", "a") for sd in ("low", "high") if flags[ax][sd] == "edge") + "); widen model.search.G / a or raise max_extensions"
            LOG.warning(msg)
            info["warnings"].append(msg)
            break
        new_points = []
        for axis in ("G", "a"):
            if axis == "a" and not a_grid:
                continue
            for side in ("low", "high"):
                if flags[axis][side] != "edge":
                    continue
                cur = np.array(sorted(df[axis].unique()))
                new = extend_axis(cur, side, float(s["extend_factor"]), *caps[axis], step=coarse_step[axis], span=coarse_span[axis])
                if len(new) == 0:
                    msg = f"optimum at the {side} cap of {axis} ({caps[axis]}); raise search.{axis}_max / lower search.{axis}_min if this is not expected"
                    LOG.warning(msg)
                    info["warnings"].append(msg)
                    continue
                other = "a" if axis == "G" else "G"
                other_vals = np.array(sorted(df[df["stage"] != "refined"][other].unique())) if "stage" in df.columns else np.array(sorted(df[other].unique()))
                new_points += list(itertools.product(new, other_vals) if axis == "G" else itertools.product(other_vals, new))
                info["extensions"].append({"axis": axis, "side": side, "values": new.tolist()})
        if not new_points:
            break
        n_ext += 1
        LOG.warning("optimum on the grid border: extending the grid (%d new points, extension %d/%d)", len(new_points), n_ext, int(s["max_extensions"]))
        df = pd.concat([df, evaluate_points(model, new_points, SC, omega, emp, tr, band, stage="extended", **kw)], ignore_index=True)
        info["stages"].append({"stage": "extended", "n_points": len(new_points)})
        best = select_best(df, metric, allow_zero_G=s["allow_zero_G"])
        if s["refine"]:
            # refine now so that a refined optimum landing on the outer edge can trigger a further extension
            df, best = _refine(df, best)
    if s["refine"] and not any(st["stage"] == "refined" for st in info["stages"]):
        df, best = _refine(df, best)
    flags = border_flags(best, df)
    info["border"] = flags
    if flags["at_validity_limit"]:
        info["warnings"].append("optimum next to invalid (unstable) grid points: limited by the model's validity, not by the range")
    if flags["on_border"] and not any("border" in w for w in info["warnings"]):
        msg = "optimum on the border of the evaluated range after extension/refinement; widen the range or raise max_extensions"
        LOG.warning(msg)
        info["warnings"].append(msg)
    best_refined = dict(best)
    # ---- interpolation
    used = {"G": float(best["G"]), "a": float(best["a"]), "source": "grid" if not s["refine"] else "refined_grid"}
    interp = None
    if s["interpolate"]:
        interp = interpolate_optimum(df, best, metric)
        if interp["interpolated_axes"]:
            used = _validated_used(model, {"G": interp["G"], "a": interp["a"], "source": "parabolic_interpolation"}, best_refined, SC, omega, a_fn, info, require_subcritical)
    # ---- continuous optimisation
    cont = None
    if s["continuous"]:
        cont = continuous_search(model, SC, omega, emp, tr, used["G"], used["a"], band, fit_a=a_grid, metric=metric,
                                 bounds_G=caps["G"], bounds_a=caps["a"], tau_tr=tau_tr, beta=beta, dt=dt, n_sim=n_sim, seed=seed,
                                 transient_s=transient_s, filter_consistent=filter_consistent, fcd_cfg=fcd_cfg,
                                 method=s["continuous_method"], max_fev=int(s["continuous_max_fev"]), use_numba=use_numba,
                                 a_fn=a_fn, a_grad=a_grad, pso_cfg={**s.get("pso", {}), "n_jobs": n_jobs}, require_subcritical=require_subcritical)
        better = (cont[metric] <= float(best[metric])) if LOWER_IS_BETTER.get(metric, True) else (cont[metric] >= float(best[metric]))
        if np.isfinite(cont[metric]) and (better or cont["success"]):
            used = _validated_used(model, {"G": cont["G"], "a": cont["a"], "source": f"continuous ({cont['method']})"}, best_refined, SC, omega, a_fn, info, require_subcritical)
        info["continuous"] = cont
    final = dict(best)
    final.update({"G": used["G"], "a": used["a"], "source": used["source"], "G_grid": best_refined["G"], "a_grid": best_refined["a"]})
    info.update(info_axis)
    info.update({"best_coarse": {k: best_coarse[k] for k in ("G", "a", metric)}, "best_refined": {k: best_refined[k] for k in ("G", "a", metric)},
                 "interpolated": interp, "used": used, "n_evaluated": int(len(df)), "n_invalid": int((~df["valid"].astype(bool)).sum()),
                 "zero_G_excluded": not s["allow_zero_G"]})
    LOG.info("search result: G=%.4g a=%.4g (%s); coarse best G=%.3g, %s=%.4g", used["G"], used["a"], used["source"], best_coarse["G"], metric, best_coarse[metric])
    return df, final, info


def error_surface(df: pd.DataFrame, metric: str, stages: Sequence[str] = ("coarse", "extended")) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pivot the (regular) coarse/extended part of the table into (G_values, a_values, surface[G, a])."""
    d = df[df["stage"].isin(stages)] if "stage" in df.columns else df
    if len(d) == 0:
        d = df
    Gs = np.array(sorted(d["G"].unique()))
    As = np.array(sorted(d["a"].unique()))
    surf = np.full((len(Gs), len(As)), np.nan)
    gi = {g: i for i, g in enumerate(Gs)}
    ai = {a: i for i, a in enumerate(As)}
    for _, r in d.iterrows():
        surf[gi[r["G"]], ai[r["a"]]] = r[metric]
    return Gs, As, surf
