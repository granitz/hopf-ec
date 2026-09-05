import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from hopfec.atlases import atlas_from_file
from hopfec.maps import hetero_a, load_parcel_map, parcellate_volume, read_parcel_table


def _atlas(tmp_path):
    lab = np.zeros((8, 8, 4), np.int16)
    for k in range(4):
        lab[2 * k : 2 * k + 2, :, :] = k + 1
    nib.save(nib.Nifti1Image(lab, np.diag([2.0, 2.0, 2.0, 1.0])), tmp_path / "atlas_dseg.nii.gz")
    pd.DataFrame({"index": [1, 2, 3, 4], "name": ["A_L", "A_R", "B_L", "B_R"]}).to_csv(tmp_path / "atlas_dseg.tsv", sep="\t", index=False)
    return atlas_from_file(tmp_path / "atlas_dseg.nii.gz")


def test_tables_and_volume(tmp_path):
    atlas = _atlas(tmp_path)
    pd.DataFrame({"index": [4, 3, 2, 1], "myelin": [4.0, 3.0, 2.0, 1.0]}).to_csv(tmp_path / "m1.tsv", sep="\t", index=False)
    assert np.allclose(read_parcel_table(tmp_path / "m1.tsv", atlas), [1, 2, 3, 4])
    pd.DataFrame({"name": ["B_R", "A_L"], "value": [9.0, 1.0]}).to_csv(tmp_path / "m2.csv", index=False)
    v = read_parcel_table(tmp_path / "m2.csv", atlas)
    assert v[0] == 1.0 and v[3] == 9.0 and np.isnan(v[1])
    (tmp_path / "m3.txt").write_text("0.1\n0.2\n0.3\n0.4\n")
    assert np.allclose(read_parcel_table(tmp_path / "m3.txt", atlas), [0.1, 0.2, 0.3, 0.4])
    vol = np.zeros((8, 8, 4))
    for k in range(4):
        vol[2 * k : 2 * k + 2] = k + 10
    nib.save(nib.Nifti1Image(vol, np.diag([2.0, 2.0, 2.0, 1.0])), tmp_path / "map.nii.gz")
    assert np.allclose(parcellate_volume(nib.load(str(tmp_path / "map.nii.gz")), atlas), [10, 11, 12, 13])
    pm = load_parcel_map({"name": "m", "file": str(tmp_path / "map.nii.gz")}, atlas)
    assert abs(pm.z.mean()) < 1e-12 and abs(pm.z.std() - 1) < 1e-12
    a = hetero_a(-0.02, 0.01, [pm])
    assert a.shape == (4,) and np.argmax(a) == 3


def test_pipeline_heterogeneity(tmp_path):
    from tests.test_pipeline_cli import _make

    _make(tmp_path)
    N = 10
    pd.DataFrame({"index": range(1, N + 1), "value": np.linspace(0, 1, N)}).to_csv(tmp_path / "myelin.tsv", sep="\t", index=False)
    cfg = (tmp_path / "cfg.yaml").read_text() + (
        "  heterogeneity:\n    enabled: true\n    map: {name: myelin, file: ./myelin.tsv}\n    beta: {start: -0.02, stop: 0.02, num: 3}\n"
    )
    # the base config puts `model:` before `compute:`; append heterogeneity under model by rewriting
    txt = cfg.replace("compute:\n  n_jobs: 1\n", "").replace("  nonlinear:\n    n_sim: 1\n    max_iter: 5\n    patience: 3\n",
                                                             "  nonlinear:\n    n_sim: 1\n    max_iter: 5\n    patience: 3\n  heterogeneity:\n    enabled: true\n    map: {name: myelin, file: ./myelin.tsv}\n    beta: {start: -0.02, stop: 0.02, num: 3}\n")
    txt = txt.split("  heterogeneity:\n    enabled: true\n    map: {name: myelin, file: ./myelin.tsv}\n    beta: {start: -0.02, stop: 0.02, num: 3}\n", 1)
    txt = txt[0] + "  heterogeneity:\n    enabled: true\n    map: {name: myelin, file: ./myelin.tsv}\n    beta: {start: -0.02, stop: 0.02, num: 3}\n" + "compute:\n  n_jobs: 1\n"
    (tmp_path / "cfg_h.yaml").write_text(txt)
    r = subprocess.run([sys.executable, "-m", "hopfec.cli", "fit", "-c", str(tmp_path / "cfg_h.yaml"), "-q", "--model", "linear", "--set", "group.cross_validate=false", "--set", "model.linear.lambda_group=0.1"],
                       capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-3000:]
    out = tmp_path / "derivatives" / "hopfec"
    h = json.loads((out / "models" / "model-hopflinear" / "group-all_atlas-toydseg_model-hopflinear_desc-heterogeneity.json").read_text())
    assert h["map"] == "myelin" and "beta" in h and h["a_min"] <= h["a_max"]
    a_tab = pd.read_csv(out / "models" / "model-hopflinear" / "group-all_atlas-toydseg_model-hopflinear_desc-heterogeneity_a.tsv", sep="\t")
    assert list(a_tab.columns) == ["z", "a_j"] and len(a_tab) == N
    s = json.loads((out / "models" / "model-hopflinear" / "search" / "group-all_atlas-toydseg_model-hopflinear_desc-search.json").read_text())
    assert s["a_axis"] == "beta"
    fit = json.loads((out / "models" / "model-hopflinear" / "sub-01" / "sub-01_atlas-toydseg_model-hopflinear_desc-fit.json").read_text())
    assert isinstance(fit["a"], list) and len(fit["a"]) == N
