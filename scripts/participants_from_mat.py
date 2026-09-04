#!/usr/bin/env python
"""Build participants.tsv from MATLAB files that hold the keys 'subjects' and 'group'.

    python scripts/participants_from_mat.py data/PSIPET-ID_group-active_task-rest.mat data/PSIPET-ID_group-placebo_task-restvisual.mat ... \
        -o participants.tsv [--id-name participant_id|subjects] [--condition-in-id] [--pad 2]

* subjects: cell array of names / char matrix / numeric ids (MATLAB v5-v7 via scipy, v7.3 via h5py)
* group:    per-subject labels, a single label, or absent -> then the file name entity group-<label> is used
* the task is read from the file name entity task-<name>; one has_<task> column (1/0) per task found

hopfec reads the result directly (columns participant_id + group).  If the same subject occurs in more
than one group (cross-over design), the groups are joined with '+' and a warning is printed; with
--condition-in-id every (subject, group) pair becomes its own participant id instead (sub-01active ...).
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- MATLAB readers
def _to_str_list(v) -> list[str]:
    if isinstance(v, str):
        return [v.strip()]
    if isinstance(v, (bytes, np.bytes_)):
        return [v.decode(errors="replace").strip()]
    a = np.asarray(v)
    if a.dtype.kind in "US":
        return [str(x).strip() for x in a.ravel()]
    if a.dtype == object:
        out: list[str] = []
        for x in a.ravel():
            out += _to_str_list(x)
        return out
    if a.dtype.kind in "iuf":
        vals = a.ravel()
        return [str(int(x)) if float(x).is_integer() else str(x) for x in vals]
    raise TypeError(f"cannot interpret MATLAB value of dtype {a.dtype}")


def _h5_to_str_list(f, d) -> list[str]:
    import h5py

    if isinstance(d, h5py.Group):
        raise TypeError("MATLAB 'string' arrays cannot be decoded from v7.3 files; convert with cellstr() in MATLAB")
    a = np.asarray(d[()])
    if a.dtype.kind == "O":  # cell array: references
        out: list[str] = []
        for r in a.ravel():
            out += _h5_to_str_list(f, f[r])
        return out
    if "MATLAB_empty" in d.attrs and int(d.attrs["MATLAB_empty"]):
        return []
    cls = d.attrs.get("MATLAB_class", b"")
    cls = cls.decode() if isinstance(cls, bytes) else str(cls)
    if cls == "char" or (a.dtype.kind == "u" and a.dtype.itemsize == 2):
        m = a.T if a.ndim == 2 else a.reshape(1, -1)  # MATLAB char matrix is stored transposed
        return ["".join(chr(int(c)) for c in row).strip() for row in m]
    return _to_str_list(a)


def read_mat_strings(path: Path, key: str) -> list[str] | None:
    """Values of *key* as a list of strings, or None when the key is absent."""
    try:
        from scipy.io import loadmat

        m = loadmat(str(path), variable_names=[key], squeeze_me=True, chars_as_strings=True)
        if key not in m:
            return None
        return _to_str_list(m[key])
    except NotImplementedError:  # v7.3 / HDF5
        import h5py

        with h5py.File(path, "r") as f:
            if key not in f:
                return None
            return _h5_to_str_list(f, f[key])


# ----------------------------------------------------------------------------- table
def norm_subject(s: str, pad: int = 0) -> str:
    s = re.sub(r"\s+", "", str(s))
    s = re.sub(r"^(sub|subject|participant)[-_]?", "", s, flags=re.I)
    s = re.sub(r"[^A-Za-z0-9]", "", s)
    if pad and s.isdigit():
        s = s.zfill(pad)
    return f"sub-{s}"


def entity(name: str, key: str) -> str | None:
    m = re.search(rf"(?:^|_){key}-([A-Za-z0-9]+)", name)
    return m.group(1) if m else None


def build_table(files: list[Path], pad: int = 0) -> tuple[pd.DataFrame, list[str]]:
    rows: dict[str, dict] = OrderedDict()
    tasks: list[str] = []
    for f in files:
        f = Path(f)
        subjects = read_mat_strings(f, "subjects")
        if subjects is None:
            raise KeyError(f"{f.name}: no 'subjects' key")
        subjects = [s for s in subjects if s]
        grp_key = read_mat_strings(f, "group")
        grp_file = entity(f.name, "group")
        task = entity(f.name, "task") or "unknown"
        if task not in tasks:
            tasks.append(task)
        if grp_key and len(grp_key) == len(subjects) and len(set(grp_key)) > 1:
            groups = [g.strip() for g in grp_key]           # per-subject labels
        elif grp_key and len(set(g.strip() for g in grp_key)) == 1:
            g = grp_key[0].strip()
            if grp_file and g.lower() != grp_file.lower():
                print(f"WARNING {f.name}: 'group' key says {g!r}, file name says {grp_file!r}; using the key", file=sys.stderr)
            groups = [g] * len(subjects)
        elif grp_file:
            groups = [grp_file] * len(subjects)
        else:
            raise ValueError(f"{f.name}: no group in the file name (group-<label>) and no usable 'group' key")
        for s, g in zip(subjects, groups):
            sid = norm_subject(s, pad)
            r = rows.setdefault(sid, {"groups": [], "tasks": set(), "files": []})
            if g not in r["groups"]:
                r["groups"].append(g)
            r["tasks"].add(task)
            r["files"].append(f.name)
        print(f"{f.name}: {len(subjects)} subjects, group(s) {sorted(set(groups))}, task {task}", file=sys.stderr)
    return rows, tasks


def to_dataframe(rows: dict, tasks: list[str], id_name: str = "participant_id", condition_in_id: bool = False) -> pd.DataFrame:
    out = []
    multi = [s for s, r in rows.items() if len(r["groups"]) > 1]
    if multi and not condition_in_id:
        print(f"WARNING: {len(multi)} subject(s) occur in more than one group (cross-over?): e.g. {multi[:5]}; groups joined with '+'. "
              "Use --condition-in-id to make one participant id per (subject, group).", file=sys.stderr)
    for sid, r in rows.items():
        groups = r["groups"] if condition_in_id else ["+".join(r["groups"])]
        for g in groups:
            pid = f"{sid}{re.sub(r'[^A-Za-z0-9]', '', g)}" if condition_in_id and len(r["groups"]) > 1 else sid
            row = {id_name: pid, "group": g}
            for t in tasks:
                row[f"has_{t}"] = int(t in r["tasks"])
            row["n_files"] = len(r["files"])
            out.append(row)
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help=".mat files with keys 'subjects' and 'group'")
    ap.add_argument("-o", "--output", default="participants.tsv")
    ap.add_argument("--id-name", default="participant_id", help="name of the id column (participant_id is what BIDS/hopfec expect)")
    ap.add_argument("--condition-in-id", action="store_true", help="one participant id per (subject, group) for cross-over designs")
    ap.add_argument("--pad", type=int, default=0, help="zero-pad numeric subject ids to this width (e.g. 2 -> sub-01)")
    args = ap.parse_args(argv)
    rows, tasks = build_table([Path(f) for f in args.files], pad=args.pad)
    df = to_dataframe(rows, tasks, args.id_name, args.condition_in_id)
    df.to_csv(args.output, sep="\t", index=False)
    print(f"\nwrote {args.output}: {len(df)} rows, groups {df['group'].value_counts().to_dict()}, tasks {tasks}", file=sys.stderr)
    print(df.head(10).to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
