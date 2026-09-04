"""participants_from_mat.py on synthetic MATLAB files (v5 via scipy, v7.3 via h5py)."""
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import savemat

from hopfec.bids import read_participants

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "participants_from_mat.py"


def _v73(path: Path, subjects: list[str], group: str):
    with h5py.File(path, "w", userblock_size=512) as f:
        refs = f.create_group("#refs#")
        ds = []
        for i, s in enumerate(subjects):
            d = refs.create_dataset(f"s{i}", data=np.array([ord(c) for c in s], np.uint16).reshape(-1, 1))
            d.attrs["MATLAB_class"] = np.bytes_("char")
            ds.append(d.ref)
        c = f.create_dataset("subjects", data=np.array(ds, dtype=h5py.ref_dtype).reshape(-1, 1))
        c.attrs["MATLAB_class"] = np.bytes_("cell")
        g = f.create_dataset("group", data=np.array([ord(ch) for ch in group], np.uint16).reshape(-1, 1))
        g.attrs["MATLAB_class"] = np.bytes_("char")
    with open(path, "r+b") as f:
        f.write(b"MATLAB 7.3 MAT-file, Platform: synthetic" + b" " * 60)
        f.seek(124)
        f.write(b"\x00\x02IM")


def test_script(tmp_path):
    subs_a = np.array(["PS01", "PS02", "PS03"], dtype=object)
    savemat(tmp_path / "X_group-active_task-rest.mat", {"subjects": subs_a, "group": "active"})
    savemat(tmp_path / "X_group-active_task-restvisual.mat", {"subjects": np.array(["PS01", "PS03"], dtype=object), "group": "active"})
    _v73(tmp_path / "X_group-placebo_task-rest.mat", ["PS04", "PS05"], "placebo")
    _v73(tmp_path / "X_group-placebo_task-restvisual.mat", ["PS04"], "placebo")
    out = tmp_path / "participants.tsv"
    r = subprocess.run([sys.executable, str(SCRIPT), *sorted(str(p) for p in tmp_path.glob("*.mat")), "-o", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(out, sep="\t")
    assert list(df.columns) == ["participant_id", "group", "has_rest", "has_restvisual", "n_files"]
    assert df["participant_id"].tolist() == ["sub-PS01", "sub-PS02", "sub-PS03", "sub-PS04", "sub-PS05"]
    assert df["group"].tolist() == ["active"] * 3 + ["placebo"] * 2
    assert df["has_restvisual"].tolist() == [1, 0, 1, 1, 0] and df["has_rest"].tolist() == [1] * 5
    hp = read_participants(out)
    assert hp["group"].tolist() == df["group"].tolist()
    # cross-over: same subject in both groups
    savemat(tmp_path / "Y_group-placebo_task-rest.mat", {"subjects": np.array(["PS01"], dtype=object), "group": "placebo"})
    r = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "X_group-active_task-rest.mat"), str(tmp_path / "Y_group-placebo_task-rest.mat"), "-o", str(out), "--condition-in-id"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(out, sep="\t")
    assert "sub-PS01active" in df["participant_id"].tolist() and "sub-PS01placebo" in df["participant_id"].tolist()
