"""Discovery of fMRIPrep derivatives (also used for HALFpipe's internal fmriprep folder)."""
from __future__ import annotations

from pathlib import Path

from ..bids import parse_entities
from ..utils import LOG, sub_label
from .common import BoldRun, read_tr_from_sidecars


def _norm_space(s: str | None) -> str:
    return (s or "").lower()


def find_fmriprep_runs(fmriprep_dir: str | Path, sub: str, task: str | None = "rest", space: str | None = "MNI152NLin2009cAsym",
                       res: str | None = "2", ses: str | None = None, run: str | None = None,
                       bids_dir: str | Path | None = None, tr: float | None = None) -> list[BoldRun]:
    """Return preprocessed BOLD runs of one subject matching task/space/res (+ mask, confounds, TR)."""
    fmriprep_dir = Path(fmriprep_dir)
    sub = sub_label(sub)
    subdir = fmriprep_dir / sub
    if not subdir.is_dir():
        return []
    bolds = sorted(subdir.glob("**/func/*_desc-preproc_bold.nii.gz"))
    out: list[BoldRun] = []
    for b in bolds:
        e = parse_entities(b.name)
        if task and e.get("task") != task:
            continue
        if ses and e.get("ses") != ses:
            continue
        if run and e.get("run") != run:
            continue
        if space and _norm_space(e.get("space")) != _norm_space(space):
            continue
        if res is not None and e.get("res") not in (None, str(res), f"{int(res):02d}" if str(res).isdigit() else str(res)):
            continue
        # companions share the entities without space/res/desc
        base_ents = {k: v for k, v in e.items() if k in ("sub", "ses", "task", "acq", "rec", "dir", "run", "echo")}
        prefix = "_".join(f"{k}-{base_ents[k]}" for k in ("sub", "ses", "task", "acq", "rec", "dir", "run", "echo") if k in base_ents)
        space_part = f"_space-{e['space']}" + (f"_res-{e['res']}" if e.get("res") else "") if e.get("space") else ""
        mask = b.with_name(f"{prefix}{space_part}_desc-brain_mask.nii.gz")
        if not mask.exists():
            mask = None
        conf = None
        for cand in (f"{prefix}_desc-confounds_timeseries.tsv", f"{prefix}_desc-confounds_regressors.tsv"):
            c = b.with_name(cand)
            if c.exists():
                conf = c
                break
        conf_json = conf.with_suffix(".json") if conf and conf.with_suffix(".json").exists() else None
        this_tr = tr or read_tr_from_sidecars(b, bids_dir)
        out.append(BoldRun(sub=sub, bold=b, task=e.get("task"), ses=e.get("ses"), run=e.get("run"), space=e.get("space"),
                           mask=mask, confounds=conf, confounds_json=conf_json, tr=this_tr, source="fmriprep", entities=e))
    if not out and bolds:
        LOG.warning("%s: %d preprocessed BOLD files found but none match task=%s space=%s res=%s", sub, len(bolds), task, space, res)
    return out


def fmriprep_subjects(fmriprep_dir: str | Path) -> list[str]:
    return sorted(p.name for p in Path(fmriprep_dir).glob("sub-*") if p.is_dir())
