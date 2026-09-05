"""Particle swarm optimisation (global, derivative-free) with parallel particle evaluation.

Standard constricted PSO (Clerc & Kennedy): w = 0.729, c1 = c2 = 1.49445, positions clipped to the
bounds, velocities limited to the box width.  Used for low-dimensional global parameters
(G, a or beta, noise, lag ...) with any fitting metric, including simulation-based ones
(common random numbers make the objective deterministic).
"""
from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np
from joblib import Parallel, delayed

from ..utils import LOG


def pso_minimize(fun: Callable[[np.ndarray], float], bounds: Sequence[tuple[float, float]], n_particles: int = 20, n_iter: int = 30,
                 x0: np.ndarray | None = None, spread: float = 0.1, w: float = 0.729, c1: float = 1.49445, c2: float = 1.49445,
                 seed: int = 0, n_jobs: int = 1, stall_iter: int = 8, tol: float = 1e-7, verbose: int = 0) -> dict:
    """Minimise fun(x) over the box `bounds`.  Returns {x, fun, history, n_fev, n_iter, converged}."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    d = len(bounds)
    width = hi - lo
    if x0 is not None:
        x0 = np.clip(np.asarray(x0, float), lo, hi)
        X = x0 + spread * width * rng.standard_normal((n_particles, d))
        X[0] = x0
    else:
        X = lo + width * rng.random((n_particles, d))
    X = np.clip(X, lo, hi)
    V = 0.1 * width * rng.standard_normal((n_particles, d))
    vmax = width

    def evaluate(P):
        if n_jobs != 1:
            return np.array(Parallel(n_jobs=n_jobs)(delayed(fun)(p) for p in P), float)
        return np.array([fun(p) for p in P], float)

    F = evaluate(X)
    F = np.where(np.isfinite(F), F, np.inf)
    pbest, pbest_f = X.copy(), F.copy()
    g = int(np.argmin(F))
    gbest, gbest_f = X[g].copy(), float(F[g])
    history = [{"iter": 0, "best": gbest_f, "mean": float(np.mean(F[np.isfinite(F)])) if np.isfinite(F).any() else float("nan")}]
    n_fev = n_particles
    stall = 0
    converged = False
    for it in range(1, int(n_iter) + 1):
        r1, r2 = rng.random((n_particles, d)), rng.random((n_particles, d))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        V = np.clip(V, -vmax, vmax)
        X = np.clip(X + V, lo, hi)
        F = evaluate(X)
        F = np.where(np.isfinite(F), F, np.inf)
        n_fev += n_particles
        better = F < pbest_f
        pbest[better], pbest_f[better] = X[better], F[better]
        g = int(np.argmin(pbest_f))
        if pbest_f[g] < gbest_f - tol:
            gbest, gbest_f = pbest[g].copy(), float(pbest_f[g])
            stall = 0
        else:
            stall += 1
        history.append({"iter": it, "best": gbest_f, "mean": float(np.mean(F[np.isfinite(F)])) if np.isfinite(F).any() else float("nan")})
        if verbose:
            LOG.info("PSO it %3d  best=%.6g  swarm mean=%.6g", it, gbest_f, history[-1]["mean"])
        if stall >= stall_iter:
            converged = True
            break
    return {"x": gbest, "fun": gbest_f, "history": history, "n_fev": n_fev, "n_iter": len(history) - 1, "converged": converged,
            "elapsed_s": time.time() - t0}
