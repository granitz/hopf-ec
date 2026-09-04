"""BIDS helpers: entity parsing, subject discovery, participants.tsv with groups."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .utils import LOG, sub_label

ENTITY_ORDER = ["sub", "ses", "task", "acq", "ce", "rec", "dir", "run", "echo", "space", "res", "den",
                "atlas", "seg", "feature", "setting", "model", "group", "hemi", "desc", "stat"]


def strip_ext(name: str) -> tuple[str, str]:
    """Return (stem, extension) handling double extensions such as .nii.gz / .tsv.gz."""
    name = Path(name).name
    for ext in (".nii.gz", ".tsv.gz", ".mat.gz", ".txt.gz"):
        if name.endswith(ext):
            return name[: -len(ext)], ext
    p = Path(name)
    return p.stem, p.suffix


def parse_entities(path: str | Path) -> dict:
    """Parse ``key-value`` entities + suffix/extension from a BIDS-style file name."""
    stem, ext = strip_ext(str(path))
    parts = stem.split("_")
    ents: dict = {}
    suffix = None
    for i, part in enumerate(parts):
        if "-" in part:
            k, v = part.split("-", 1)
            if re.fullmatch(r"[A-Za-z]+", k):
                ents[k] = v
                continue
        if i == len(parts) - 1:
            suffix = part
        else:
            ents.setdefault("_extra", []).append(part)
    ents["suffix"] = suffix
    ents["extension"] = ext
    return ents


def build_name(entities: dict, suffix: str, ext: str) -> str:
    parts = []
    for k in ENTITY_ORDER:
        v = entities.get(k)
        if v is not None and v != "":
            parts.append(f"{k}-{v}")
    for k, v in entities.items():
        if k not in ENTITY_ORDER and not k.startswith("_") and k not in ("suffix", "extension") and v not in (None, ""):
            parts.append(f"{k}-{v}")
    parts.append(suffix)
    return "_".join(parts) + ext


def find_subjects(bids_dir: str | Path) -> list[str]:
    bids_dir = Path(bids_dir)
    subs = sorted(p.name for p in bids_dir.glob("sub-*") if p.is_dir())
    return subs


def find_sessions(bids_dir: str | Path, sub: str) -> list[str]:
    return sorted(p.name for p in (Path(bids_dir) / sub_label(sub)).glob("ses-*") if p.is_dir())


def read_participants(path: str | Path, id_column: str | None = None, group_column: str | None = None) -> pd.DataFrame:
    """Read a participants table (TSV/CSV).

    First column (or ``id_column``) holds subject ids like ``sub-01`` (or ``01``); the second
    column (or ``group_column``) holds the group label.  Returns a DataFrame with normalised
    columns ``participant_id`` and ``group`` (plus any extra columns).
    """
    path = Path(path)
    sep = "," if path.suffix.lower() == ".csv" else "\t"
    df = pd.read_csv(path, sep=sep, dtype=str, comment=None, keep_default_na=True)
    df.columns = [str(c).strip() for c in df.columns]
    cols_l = {c.lower(): c for c in df.columns}
    if id_column is None:
        for cand in ("participant_id", "participant", "subjects", "subject_id", "subject", "sub", "id"):
            if cand in cols_l:
                id_column = cols_l[cand]
                break
        else:
            id_column = df.columns[0]
    if group_column is None:
        for cand in ("group", "diagnosis", "dx", "condition", "cohort"):
            if cand in cols_l:
                group_column = cols_l[cand]
                break
        else:
            others = [c for c in df.columns if c != id_column]
            if not others:
                raise ValueError(f"{path}: need a second column with the group label")
            group_column = others[0]
            LOG.warning("participants: using column %r as the group column", group_column)
    out = df.rename(columns={id_column: "participant_id", group_column: "group"})
    out["participant_id"] = out["participant_id"].map(lambda s: sub_label(str(s).strip()))
    out = out[out["group"].notna()]
    out["group"] = out["group"].astype(str).str.strip()
    out = out[~out["group"].str.lower().isin(["", "nan", "n/a", "na", "none", "<na>"])]
    if out["participant_id"].duplicated().any():
        dups = out.loc[out["participant_id"].duplicated(), "participant_id"].tolist()
        raise ValueError(f"{path}: duplicated participant ids {dups}")
    return out.reset_index(drop=True)


def validate_participants(df: pd.DataFrame, bids_subjects: Iterable[str]) -> dict:
    bids = set(bids_subjects)
    tsv = set(df["participant_id"])
    return {
        "n_tsv": len(tsv),
        "n_bids": len(bids),
        "missing_in_bids": sorted(tsv - bids),
        "missing_in_tsv": sorted(bids - tsv),
        "groups": df.groupby("group")["participant_id"].count().to_dict(),
    }


def select_participants(df: pd.DataFrame, include: Sequence[str] | None = None,
                        exclude: Sequence[str] | None = None) -> pd.DataFrame:
    out = df
    if include:
        inc = {sub_label(s) for s in include}
        out = out[out["participant_id"].isin(inc)]
    if exclude:
        exc = {sub_label(s) for s in exclude}
        out = out[~out["participant_id"].isin(exc)]
    return out.reset_index(drop=True)


def groups_dict(df: pd.DataFrame) -> dict[str, list[str]]:
    return {g: sorted(sub["participant_id"].tolist()) for g, sub in df.groupby("group")}
