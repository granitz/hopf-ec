"""Figures: connectivity matrices, error surfaces, fit diagnostics (matplotlib, headless)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .utils import ensure_dir, upper_tri  # noqa: E402


def _save(fig, path: Path):
    ensure_dir(Path(path).parent)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return Path(path)


def plot_matrix(M: np.ndarray, path: Path, title: str = "", log: bool = False, cmap: str = "viridis", vmin=None, vmax=None,
                symmetric_cmap: bool = False, labels: list[str] | None = None):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    data = np.log10(M + 1e-12) if log else M
    if symmetric_cmap:
        v = np.nanmax(np.abs(data)) if vmax is None else vmax
        im = ax.imshow(data, cmap="RdBu_r", vmin=-v, vmax=v)
    else:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title + (" (log10)" if log else ""))
    ax.set_xlabel("source region j")
    ax.set_ylabel("target region i")
    if labels and len(labels) <= 40:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, path)


def plot_sc_panels(mats: dict, path: Path, title: str = ""):
    keys = [k for k in ("count", "volnorm", "combined") if k in mats]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4.5))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        M = mats[k]
        im = ax.imshow(np.log10(M + (1 if k == "count" else 1e-12)), cmap="hot" if k == "count" else "viridis")
        ax.set_title({"count": "raw streamline count", "volnorm": "volume-normalised", "combined": "volume + length corrected"}[k] + " (log10)")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title)
    return _save(fig, path)


def plot_error_surface(Gs: np.ndarray, As: np.ndarray, surf: np.ndarray, metric: str, path: Path, best: dict | None = None,
                       title: str = ""):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if len(As) == 1:
        ax.plot(Gs, surf[:, 0], "o-")
        ax.set_xlabel("global coupling G")
        ax.set_ylabel(metric)
        if best:
            ax.axvline(best["G"], color="r", ls="--", label=f"best G = {best['G']:.3g}")
            ax.legend()
        ax.set_title(title or f"{metric} vs G (a = {As[0]:.3g})")
    else:
        im = ax.imshow(surf.T, origin="lower", aspect="auto", extent=[Gs[0], Gs[-1], As[0], As[-1]], cmap="viridis")
        ax.set_xlabel("global coupling G")
        ax.set_ylabel("bifurcation parameter a")
        fig.colorbar(im, ax=ax, label=metric)
        if best:
            ax.plot(best["G"], best["a"], "r*", ms=14, label=f"best (G={best['G']:.3g}, a={best['a']:.3g})")
            ax.legend(loc="upper right")
        ax.set_title(title or f"error surface: {metric}")
    return _save(fig, path)


def plot_fit(FC_emp: np.ndarray, FC_sim: np.ndarray, COV_emp: np.ndarray | None, COV_sim: np.ndarray | None, history: list[dict] | None,
             path: Path, title: str = ""):
    ncol = 3 if history else 2
    fig, axes = plt.subplots(1, ncol, figsize=(5 * ncol, 4.3))
    ax = axes[0]
    ax.scatter(upper_tri(FC_emp), upper_tri(FC_sim), s=6, alpha=0.5)
    lim = [min(FC_emp.min(), FC_sim.min()), 1.0]
    ax.plot(lim, lim, "k--", lw=0.8)
    r = np.corrcoef(upper_tri(FC_emp), upper_tri(FC_sim))[0, 1]
    ax.set_xlabel("empirical FC")
    ax.set_ylabel("model FC")
    ax.set_title(f"FC fit  r = {r:.3f}")
    ax = axes[1]
    if COV_emp is not None and COV_sim is not None:
        ax.scatter(COV_emp.ravel(), COV_sim.ravel(), s=6, alpha=0.5, color="tab:orange")
        lim = [min(COV_emp.min(), COV_sim.min()), max(COV_emp.max(), COV_sim.max())]
        ax.plot(lim, lim, "k--", lw=0.8)
        r2 = np.corrcoef(COV_emp.ravel(), COV_sim.ravel())[0, 1]
        ax.set_xlabel("empirical lagged corr")
        ax.set_ylabel("model lagged corr")
        ax.set_title(f"lagged-covariance fit  r = {r2:.3f}")
    if history:
        ax = axes[2]
        it = [h["iter"] for h in history]
        ax.plot(it, [h["fit_rmse"] for h in history], label="fit RMSE")
        ax.plot(it, [1 - h["fc_corr"] for h in history], label="1 - corr(FC)")
        ax.set_xlabel("iteration")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title("convergence")
    fig.suptitle(title)
    return _save(fig, path)


def plot_ec_summary(EC: np.ndarray, SC: np.ndarray | None, path: Path, title: str = "", names: list[str] | None = None):
    ncol = 3 if SC is not None else 2
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4.6))
    im = axes[0].imshow(EC, cmap="viridis")
    axes[0].set_title("effective connectivity (j -> i)")
    axes[0].set_xlabel("source j")
    axes[0].set_ylabel("target i")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    asym = EC - EC.T
    v = np.max(np.abs(asym)) or 1.0
    im = axes[1].imshow(asym, cmap="RdBu_r", vmin=-v, vmax=v)
    axes[1].set_title("asymmetry EC - EC^T")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    if SC is not None:
        axes[2].scatter(upper_tri(SC), upper_tri(0.5 * (EC + EC.T)), s=6, alpha=0.5)
        axes[2].set_xlabel("SC (upper triangle)")
        axes[2].set_ylabel("symmetric part of EC")
        m = np.triu(np.ones_like(SC, bool), 1) & (SC > 0)
        if m.sum() > 2:
            r = np.corrcoef(SC[m], (0.5 * (EC + EC.T))[m])[0, 1]
            axes[2].set_title(f"EC vs SC  r = {r:.3f}")
    fig.suptitle(title)
    return _save(fig, path)


def plot_group_comparison(mean_a: np.ndarray, mean_b: np.ndarray, tstat: np.ndarray, sig: np.ndarray, labels: tuple[str, str], path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    v = max(np.nanmax(mean_a), np.nanmax(mean_b))
    for ax, M, t in zip(axes[:2], (mean_a, mean_b), labels):
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=v)
        ax.set_title(f"mean EC: {t}")
        fig.colorbar(im, ax=ax, fraction=0.046)
    tv = np.nanmax(np.abs(tstat)) or 1.0
    im = axes[2].imshow(np.where(sig, tstat, np.nan), cmap="RdBu_r", vmin=-tv, vmax=tv)
    axes[2].imshow(np.where(sig, np.nan, tstat), cmap="Greys", alpha=0.25, vmin=-tv, vmax=tv)
    axes[2].set_title(f"t ({labels[0]} - {labels[1]}), significant edges coloured")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    return _save(fig, path)
