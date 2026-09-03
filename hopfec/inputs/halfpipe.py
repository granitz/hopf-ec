"""Discovery of HALFpipe outputs.

HALFpipe (https://github.com/HALFpipe/HALFpipe) writes, below its working directory,
``derivatives/halfpipe/sub-<id>/func/`` with

* ``*_feature-<name>_atlas-<atlas>_timeseries.tsv``  parcellated, fully post-processed time series
  (feature type "atlas-based connectivity"), plus ``*_desc-correlation_matrix.tsv``;
* ``*_setting-<name>_bold.nii.gz``                     post-processed 4-D BOLD (confounds regressed,
  filtered, smoothed according to the setting);

and ``derivatives/fmriprep/`` with the plain fMRIPrep preprocessing (``*_desc-preproc_bold.nii.gz``
+ ``*_desc-confounds_timeseries.tsv``).  All three flavours are supported here; the folder given
may be the HALFpipe working directory, its ``derivatives`` folder or ``derivatives/halfpipe``.
"""
from __future__ import annotations

from pathlib import Path

from ..bids import parse_entities
from ..utils import LOG, sub_label
from .common import BoldRun, TimeseriesFile, read_tr_from_sidecars
from .fmriprep import find_fmriprep_runs


def halfpipe_roots(halfpipe_dir: str | Path) -> dict:
    """Locate the halfpipe and fmriprep derivative folders from any of the usual entry points."""
    d = Path(halfpipe_dir)
    roots = {"halfpipe": None, "fmriprep": None}
    cands_hp = [d, d / "derivatives" / "halfpipe", d / "halfpipe"]
    cands_fp = [d / "derivatives" / "fmriprep", d / "fmriprep", d.parent / "fmriprep"]
    for c in cands_hp:
        if c.is_dir() and any(c.glob("sub-*")) and (c.name == "halfpipe" or any(c.glob("sub-*/func/*_feature-*")) or any(c.glob("sub-*/func/*_setting-*"))):
            roots["halfpipe"] = c
            break
    for c in cands_fp:
        if c.is_dir() and any(c.glob("sub-*")):
            roots["fmriprep"] = c
            break
    return roots


def find_halfpipe_timeseries(halfpipe_dir: str | Path, sub: str, atlas: str | None = None, feature: str | None = None,
                             task: str | None = "rest", ses: str | None = None, run: str | None = None,
                             bids_dir: str | Path | None = None, tr: float | None = None) -> list[TimeseriesFile]:
    roots = halfpipe_roots(halfpipe_dir)
    root = roots["halfpipe"] or Path(halfpipe_dir)
    sub = sub_label(sub)
    files = sorted((root / sub).glob("**/*_feature-*_atlas-*_timeseries.tsv")) if (root / sub).is_dir() else []
    out = []
    for f in files:
        e = parse_entities(f.name)
        if task and e.get("task") != task:
            continue
        if ses and e.get("ses") != ses:
            continue
        if run and e.get("run") != run:
            continue
        if feature and e.get("feature") != feature:
            continue
        if atlas and (e.get("atlas") or "").lower() != atlas.lower():
            continue
        side = f.with_suffix(".json")
        out.append(TimeseriesFile(sub=sub, path=f, task=e.get("task"), ses=e.get("ses"), run=e.get("run"), atlas=e.get("atlas"),
                                  tr=tr or read_tr_from_sidecars(f, bids_dir), source="halfpipe-timeseries",
                                  sidecar=side if side.exists() else None, entities=e))
    return out


def list_halfpipe_features(halfpipe_dir: str | Path) -> dict:
    """Summary of (feature, atlas) combinations available in a HALFpipe output folder."""
    roots = halfpipe_roots(halfpipe_dir)
    root = roots["halfpipe"] or Path(halfpipe_dir)
    combos: dict[tuple, set] = {}
    for f in root.glob("sub-*/**/*_feature-*_atlas-*_timeseries.tsv"):
        e = parse_entities(f.name)
        combos.setdefault((e.get("feature"), e.get("atlas")), set()).add(f"sub-{e.get('sub')}")
    settings: dict[str, set] = {}
    for f in root.glob("sub-*/**/*_setting-*_bold.nii.gz"):
        e = parse_entities(f.name)
        settings.setdefault(e.get("setting"), set()).add(f"sub-{e.get('sub')}")
    return {"timeseries": {f"feature-{k[0]}_atlas-{k[1]}": len(v) for k, v in combos.items()},
            "settings": {k: len(v) for k, v in settings.items()},
            "fmriprep_dir": str(roots["fmriprep"]) if roots["fmriprep"] else None}


def find_halfpipe_bold(halfpipe_dir: str | Path, sub: str, setting: str | None = None, task: str | None = "rest",
                       ses: str | None = None, run: str | None = None, bids_dir: str | Path | None = None,
                       tr: float | None = None, space: str | None = None, res: str | None = None) -> list[BoldRun]:
    """Post-processed BOLD of a HALFpipe *setting* (already denoised) or, failing that, the fMRIPrep preproc BOLD."""
    roots = halfpipe_roots(halfpipe_dir)
    root = roots["halfpipe"] or Path(halfpipe_dir)
    sub = sub_label(sub)
    out: list[BoldRun] = []
    if (root / sub).is_dir():
        for b in sorted((root / sub).glob("**/*_setting-*_bold.nii.gz")):
            e = parse_entities(b.name)
            if task and e.get("task") != task:
                continue
            if ses and e.get("ses") != ses:
                continue
            if run and e.get("run") != run:
                continue
            if setting and e.get("setting") != setting:
                continue
            if space and e.get("space") and e["space"].lower() != space.lower():
                continue
            mask = None
            for cand in sorted(b.parent.glob(f"sub-{e.get('sub')}*_desc-brain_mask.nii.gz")):
                mask = cand
                break
            out.append(BoldRun(sub=sub, bold=b, task=e.get("task"), ses=e.get("ses"), run=e.get("run"), space=e.get("space") or space,
                               mask=mask, confounds=None, tr=tr or read_tr_from_sidecars(b, bids_dir), source="halfpipe",
                               postprocessed=True, entities=e))
    if out:
        return out
    if roots["fmriprep"]:
        return find_fmriprep_runs(roots["fmriprep"], sub, task=task, space=space, res=res, ses=ses, run=run, bids_dir=bids_dir, tr=tr)
    return []
