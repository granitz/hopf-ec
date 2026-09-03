from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..bids import parse_entities
from ..utils import LOG


@dataclass
class BoldRun:
    """A preprocessed BOLD run in template space with its companions."""
    sub: str
    bold: Path
    task: str | None = None
    ses: str | None = None
    run: str | None = None
    space: str | None = None
    mask: Path | None = None
    confounds: Path | None = None
    confounds_json: Path | None = None
    tr: float | None = None
    source: str = "fmriprep"
    postprocessed: bool = False      # True when confounds/filtering were already applied (HALFpipe settings)
    entities: dict = field(default_factory=dict)

    def tag(self) -> str:
        parts = [self.sub]
        if self.ses:
            parts.append(f"ses-{self.ses}")
        if self.task:
            parts.append(f"task-{self.task}")
        if self.run:
            parts.append(f"run-{self.run}")
        return "_".join(parts)


@dataclass
class TimeseriesFile:
    """A ready-made parcellated time-series table (rows = volumes, columns = regions)."""
    sub: str
    path: Path
    task: str | None = None
    ses: str | None = None
    run: str | None = None
    atlas: str | None = None
    tr: float | None = None
    source: str = "timeseries"
    sidecar: Path | None = None
    entities: dict = field(default_factory=dict)

    def tag(self) -> str:
        parts = [self.sub]
        if self.ses:
            parts.append(f"ses-{self.ses}")
        if self.task:
            parts.append(f"task-{self.task}")
        if self.run:
            parts.append(f"run-{self.run}")
        return "_".join(parts)


def _read_json(p: Path) -> dict:
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        LOG.debug("could not read %s: %s", p, e)
        return {}


def read_tr_from_sidecars(path: Path, bids_dir: str | Path | None = None) -> float | None:
    """RepetitionTime from the file's own JSON sidecar, or from the raw BIDS bold sidecars."""
    path = Path(path)
    name = path.name
    for ext in (".nii.gz", ".tsv.gz", ".nii", ".tsv"):
        if name.endswith(ext):
            stem = name[: -len(ext)]
            break
    else:
        stem = path.stem
    side = path.with_name(stem + ".json")
    if side.exists():
        meta = _read_json(side)
        for k in ("RepetitionTime", "repetition_time", "TR", "tr"):
            if k in meta and meta[k]:
                return float(meta[k])
    if bids_dir:
        ents = parse_entities(name)
        sub, ses, task, run = ents.get("sub"), ents.get("ses"), ents.get("task"), ents.get("run")
        bids_dir = Path(bids_dir)
        cands = []
        if sub:
            base = bids_dir / f"sub-{sub}" / (f"ses-{ses}" if ses else "") / "func"
            pat = f"sub-{sub}" + (f"_ses-{ses}" if ses else "") + (f"_task-{task}" if task else "") + "*" + (f"_run-{run}" if run else "") + "*_bold.json"
            cands += sorted(base.glob(pat))
        if task:
            cands += sorted(bids_dir.glob(f"task-{task}*_bold.json"))
        for c in cands:
            meta = _read_json(c)
            if meta.get("RepetitionTime"):
                return float(meta["RepetitionTime"])
    return None
