"""Small shared helpers: logging, JSON, matrices, paths."""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

LOG = logging.getLogger("hopfec")

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def setup_logging(verbosity: int = 1, logfile: str | os.PathLike | None = None) -> None:
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    root = logging.getLogger("hopfec")
    root.setLevel(level)
    if not root.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        root.addHandler(h)
    if logfile is not None:
        fh = logging.FileHandler(str(logfile))
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(fh)


def expand_env(s: str) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}`` (unset variables without default are left as-is)."""

    def _sub(m):
        name, default = m.group(1), m.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        LOG.warning("environment variable %s is not set (left unexpanded)", name)
        return m.group(0)

    return _ENV_RE.sub(_sub, os.path.expandvars(s) if "${" not in s else s)


def expand_path(p: str | os.PathLike, base: str | os.PathLike | None = None) -> Path:
    """Expand env vars and ``~``; make relative paths absolute with respect to *base* (default: cwd)."""
    s = expand_env(str(p))
    path = Path(os.path.expanduser(s))
    if not path.is_absolute():
        path = (Path(base) if base is not None else Path.cwd()) / path
    path = Path(os.path.normpath(str(path)))
    return path.resolve() if path.exists() else path


def ensure_dir(p: str | os.PathLike) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):  # noqa: D401
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(path: str | os.PathLike, obj) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=NumpyEncoder)
    return path


def load_json(path: str | os.PathLike):
    with open(path) as f:
        return json.load(f)


def save_matrix(path: str | os.PathLike, M: np.ndarray, labels: Sequence[str] | None = None) -> Path:
    """Save a 2-D matrix as TSV (header row = region labels when given) or .npy."""
    path = Path(path)
    ensure_dir(path.parent)
    M = np.asarray(M)
    if path.suffix == ".npy":
        np.save(path, M)
        return path
    sep = "," if path.suffix == ".csv" else "\t"
    if labels is not None and len(labels) == M.shape[1]:
        df = pd.DataFrame(M, columns=list(labels))
        df.to_csv(path, sep=sep, index=False, float_format="%.8g")
    else:
        np.savetxt(path, M, delimiter=sep, fmt="%.8g")
    return path


def load_matrix(path: str | os.PathLike, key: str | None = None) -> np.ndarray:
    """Load a matrix from .tsv/.csv/.txt (with or without header), .npy/.npz or MATLAB .mat."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".npy":
        return np.load(path)
    if suf == ".npz":
        z = np.load(path)
        k = key or (list(z.keys())[0])
        return z[k]
    if suf == ".mat":
        from .sc.qsi import load_connectivity_mat

        return load_connectivity_mat(path, key)[0]
    sep = "," if suf == ".csv" else None
    with open(path) as f:
        first = f.readline()
    tokens = re.split(r"[,\t ]+", first.strip())
    has_header = False
    for t in tokens:
        try:
            float(t)
        except ValueError:
            has_header = True
            break
    df = pd.read_csv(path, sep=sep, engine="python", header=0 if has_header else None)
    # drop a leading label column if present (non-numeric)
    for c in list(df.columns):
        if df[c].dtype == object:
            df = df.drop(columns=c)
    return df.to_numpy(dtype=float)


def upper_tri(M: np.ndarray, k: int = 1) -> np.ndarray:
    M = np.asarray(M)
    iu = np.triu_indices(M.shape[0], k=k)
    return M[iu]


def corr_upper(A: np.ndarray, B: np.ndarray) -> float:
    a, b = upper_tri(A), upper_tri(B)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rmse(A: np.ndarray, B: np.ndarray, mask: np.ndarray | None = None) -> float:
    d = np.asarray(A, float) - np.asarray(B, float)
    if mask is not None:
        d = d[mask]
    return float(np.sqrt(np.mean(d ** 2)))


def offdiag_mask(n: int) -> np.ndarray:
    return ~np.eye(n, dtype=bool)


def as_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return [x]


def sub_label(s: str) -> str:
    """Normalise 'sub-01' / '01' -> 'sub-01'."""
    s = str(s).strip()
    return s if s.startswith("sub-") else f"sub-{s}"


def which(cmd: str) -> str | None:
    from shutil import which as _which

    return _which(cmd)


def linspace_spec(spec) -> np.ndarray:
    """Turn a grid spec into values: list -> as is; dict {start, stop, num} or {start, stop, step}; scalar -> [scalar]."""
    if spec is None:
        return np.array([])
    if isinstance(spec, dict):
        if "values" in spec:
            return np.asarray(spec["values"], float)
        start, stop = float(spec["start"]), float(spec["stop"])
        if "num" in spec:
            return np.linspace(start, stop, int(spec["num"]))
        step = float(spec.get("step", 0.1))
        n = int(np.floor((stop - start) / step + 1e-9)) + 1
        return start + step * np.arange(n)
    if np.isscalar(spec):
        return np.array([float(spec)])
    return np.asarray(spec, float)
