"""Model node set: parcels with valid data in every (or enough) participant(s).

Parcels without usable time series (NaN / constant columns, e.g. HALFpipe coverage < threshold, or parcels
outside the field of view) are removed from the modelled system: time series, SC, parcel maps and the atlas are
reduced to the kept nodes before anything is filtered, normalised or fitted, and every saved matrix / node
vector is expanded back to the full atlas (NaN rows/columns for the dropped parcels) so that outputs stay
aligned with the parcellation.

Config (``nodes:``):
  auto: true            drop parcels whose data are invalid in a participant (see min_coverage)
  exclude: []           parcel names or label ids that are always dropped
  min_coverage: 1.0     keep a parcel if it is valid in at least this fraction of participants; participants
                        lacking a kept parcel are excluded from the fit (1.0: a parcel invalid in any
                        participant is dropped for the whole cohort and nobody is excluded)
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .atlases import Atlas
from .utils import LOG


@dataclass
class NodeSet:
    n_total: int
    keep: np.ndarray                                   # bool (n_total,)
    names: list[str]                                   # full atlas
    label_ids: np.ndarray                              # full atlas
    reasons: dict[int, str] = field(default_factory=dict)          # dropped position -> reason
    coverage: np.ndarray | None = None                 # fraction of participants with valid data, per node
    excluded_participants: dict[str, list[str]] = field(default_factory=dict)  # sub -> kept nodes it lacks

    # ---- properties
    @property
    def idx(self) -> np.ndarray:
        return np.where(self.keep)[0]

    @property
    def n_kept(self) -> int:
        return int(self.keep.sum())

    @property
    def complete(self) -> bool:
        return bool(self.keep.all())

    @property
    def kept_names(self) -> list[str]:
        return [self.names[i] for i in self.idx]

    @property
    def dropped_names(self) -> list[str]:
        return [self.names[i] for i in np.where(~self.keep)[0]]

    # ---- reduce (full -> model); tolerant to arrays that are already reduced
    def reduce_vec(self, v) -> np.ndarray:
        v = np.asarray(v)
        if v.shape[0] == self.n_kept:
            return v
        if v.shape[0] != self.n_total:
            raise ValueError(f"vector of length {v.shape[0]}: expected {self.n_total} (atlas) or {self.n_kept} (model)")
        return v[self.idx]

    def reduce_mat(self, M) -> np.ndarray:
        M = np.asarray(M)
        if M.shape[:2] == (self.n_kept, self.n_kept):
            return M
        if M.shape[:2] != (self.n_total, self.n_total):
            raise ValueError(f"matrix {M.shape}: expected ({self.n_total}, {self.n_total}) or ({self.n_kept}, {self.n_kept})")
        return M[np.ix_(self.idx, self.idx)]

    def reduce_ts(self, X) -> np.ndarray:
        """Time series T x n_total -> T x n_kept."""
        X = np.asarray(X)
        if X.shape[1] == self.n_kept and X.shape[1] != self.n_total:
            return X
        if X.shape[1] != self.n_total:
            raise ValueError(f"time series with {X.shape[1]} columns: expected {self.n_total}")
        return X[:, self.idx]

    # ---- expand (model -> full atlas)
    def expand_vec(self, v, fill=np.nan) -> np.ndarray:
        v = np.asarray(v, float)
        if v.shape[0] == self.n_total:
            return v
        if v.shape[0] != self.n_kept:
            raise ValueError(f"vector of length {v.shape[0]}: expected {self.n_kept} (model)")
        out = np.full(self.n_total, fill, float)
        out[self.idx] = v
        return out

    def expand_mat(self, M, fill=np.nan) -> np.ndarray:
        M = np.asarray(M, float)
        if M.shape[:2] == (self.n_total, self.n_total):
            return M
        if M.shape[:2] != (self.n_kept, self.n_kept):
            raise ValueError(f"matrix {M.shape}: expected ({self.n_kept}, {self.n_kept}) (model)")
        out = np.full((self.n_total, self.n_total) + M.shape[2:], fill, float)
        out[np.ix_(self.idx, self.idx)] = M
        return out

    def expand_rows(self, M, fill=np.nan) -> np.ndarray:
        """Table with one row per node (n_kept x k) -> n_total x k."""
        M = np.asarray(M, float)
        if M.shape[0] == self.n_total:
            return M
        out = np.full((self.n_total,) + M.shape[1:], fill, float)
        out[self.idx] = M
        return out

    def expand(self, M, fill=np.nan) -> np.ndarray:
        M = np.asarray(M, float)
        return self.expand_vec(M, fill) if M.ndim == 1 else self.expand_mat(M, fill)

    # ---- atlas view of the kept nodes
    def subset_atlas(self, atlas: Atlas) -> Atlas:
        if self.complete:
            return atlas
        keep = self.keep
        table = atlas.label_table
        if table is not None and "index" in table.columns:
            kept_ids = set(int(i) for i in np.asarray(atlas.label_ids)[keep])
            table = table[table["index"].astype(int).isin(kept_ids)].reset_index(drop=True)
        return dataclasses.replace(atlas, label_ids=np.asarray(atlas.label_ids)[keep],
                                   region_names=[n for n, k in zip(atlas.region_names, keep) if k],
                                   hemispheres=[h for h, k in zip(atlas.hemispheres, keep) if k] if atlas.hemispheres else [],
                                   networks=[n for n, k in zip(atlas.networks, keep) if k] if atlas.networks else [],
                                   label_table=table)

    # ---- reporting
    def table(self) -> pd.DataFrame:
        return pd.DataFrame({"index": [int(i) for i in self.label_ids], "name": list(self.names), "kept": self.keep.astype(int),
                             "reason": [self.reasons.get(i, "") for i in range(self.n_total)],
                             "coverage": self.coverage if self.coverage is not None else np.ones(self.n_total)})

    def summary(self) -> dict:
        return {"n_atlas": self.n_total, "n_model": self.n_kept, "dropped": self.dropped_names,
                "reasons": {self.names[i]: r for i, r in sorted(self.reasons.items())},
                "excluded_participants": self.excluded_participants}


def identity_node_set(atlas: Atlas) -> NodeSet:
    n = atlas.n_parcels
    return NodeSet(n_total=n, keep=np.ones(n, bool), names=list(atlas.region_names), label_ids=np.asarray(atlas.label_ids))


# --------------------------------------------------------------------------- validity of the data
def node_validity(X: np.ndarray) -> np.ndarray:
    """Per column of a T x N time-series array: finite everywhere and not constant."""
    X = np.asarray(X, float)
    finite = np.isfinite(X).all(axis=0)
    sd = np.zeros(X.shape[1])
    if finite.any():
        sd[finite] = X[:, finite].std(axis=0)
    return finite & (sd > 0)


def scan_node_validity(files: Sequence, n_expected: int) -> np.ndarray:
    """Nodes with valid data in every run of a participant (files: objects with a ``.path``)."""
    from .timeseries import load_timeseries

    valid = np.ones(n_expected, bool)
    for f in files:
        X, _ = load_timeseries(Path(getattr(f, "path", f)), n_expected)
        if X.shape[1] != n_expected:
            continue  # reported later by the empirical-statistics loader
        valid &= node_validity(X)
    return valid


def _resolve_exclusions(exclude, atlas: Atlas) -> tuple[np.ndarray, list]:
    """Config exclusions (parcel names or label ids) -> boolean mask of positions; unknown entries returned."""
    mask = np.zeros(atlas.n_parcels, bool)
    unknown = []
    names = list(atlas.region_names)
    ids = [int(i) for i in atlas.label_ids]
    for item in exclude or []:
        if isinstance(item, (int, np.integer)) or (isinstance(item, str) and item.strip().lstrip("-").isdigit()):
            i = int(item)
            if i in ids:
                mask[ids.index(i)] = True
            else:
                unknown.append(item)
        elif isinstance(item, str):
            hits = [k for k, n in enumerate(names) if n == item]
            if hits:
                mask[hits] = True
            else:
                unknown.append(item)
        else:
            unknown.append(item)
    return mask, unknown


def determine_node_set(valid: dict[str, np.ndarray], atlas: Atlas, cfg_nodes: dict | None) -> NodeSet:
    """Common node set of a cohort from per-participant validity vectors and the ``nodes`` config."""
    cfg_nodes = cfg_nodes or {}
    n = atlas.n_parcels
    ns = identity_node_set(atlas)
    excl, unknown = _resolve_exclusions(cfg_nodes.get("exclude"), atlas)
    if unknown:
        LOG.warning("nodes.exclude: %d entr%s not found in the atlas (e.g. %s)", len(unknown), "y" if len(unknown) == 1 else "ies", unknown[:5])
    for i in np.where(excl)[0]:
        ns.reasons[int(i)] = "excluded (nodes.exclude)"
    keep = ~excl
    subs = list(valid)
    V = np.array([np.asarray(valid[s], bool) for s in subs]) if subs else np.ones((0, n), bool)
    coverage = V.mean(axis=0) if len(subs) else np.ones(n)
    ns.coverage = coverage
    if cfg_nodes.get("auto", True) and len(subs):
        min_cov = float(cfg_nodes.get("min_coverage", 1.0))
        low = (coverage < min_cov) & keep
        for i in np.where(low)[0]:
            n_bad = int((~V[:, i]).sum())
            ns.reasons[int(i)] = f"no valid data in {n_bad}/{len(subs)} participants"
        keep &= ~low
    ns.keep = keep
    # participants lacking a kept node cannot be modelled on this node set
    for k, s in enumerate(subs):
        missing = np.where(keep & ~V[k])[0]
        if len(missing):
            ns.excluded_participants[s] = [atlas.region_names[i] for i in missing]
    if not ns.complete:
        LOG.warning("model node set: %d/%d parcels (dropped: %s%s)", ns.n_kept, n, ", ".join(ns.dropped_names[:8]), "..." if ns.n_kept < n - 8 else "")
    if ns.excluded_participants:
        LOG.warning("%d participant(s) excluded: no valid data on kept node(s) (%s); lower nodes.min_coverage to drop the nodes instead",
                    len(ns.excluded_participants), "; ".join(f"{s}: {v[:3]}" for s, v in list(ns.excluded_participants.items())[:5]))
    if ns.n_kept < 2:
        raise RuntimeError(f"model node set has {ns.n_kept} parcel(s); check the time series / nodes config")
    return ns
