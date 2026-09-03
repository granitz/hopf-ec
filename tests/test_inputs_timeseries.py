import json

import nibabel as nib
import numpy as np
import pandas as pd

from hopfec.atlases import atlas_from_file
from hopfec.inputs.fmriprep import find_fmriprep_runs
from hopfec.inputs.halfpipe import find_halfpipe_timeseries, halfpipe_roots, list_halfpipe_features
from hopfec.preprocess.fmriprep import build_fmriprep_command
from hopfec.timeseries import clean_timeseries, extract_run_timeseries, load_timeseries, n_dummy_scans, select_confounds


def test_fmriprep_command(tmp_path):
    cmd = build_fmriprep_command(tmp_path / "bids", tmp_path / "out", ["sub-01", "02"], {"runtime": "docker", "image": "nipreps/fmriprep:24.1.1", "nprocs": 4}, fs_license=tmp_path / "lic.txt")
    assert cmd[:2] == ["docker", "run"] and "nipreps/fmriprep:24.1.1" in cmd and "--participant-label" in cmd and "01" in cmd and "02" in cmd
    cmd2 = build_fmriprep_command(tmp_path / "bids", tmp_path / "out", None, {"runtime": "apptainer", "sif": "/x/fmriprep.sif"})
    assert cmd2[0] == "apptainer" and "/x/fmriprep.sif" in cmd2


def test_confound_selection():
    T = 50
    cols = {}
    for c in ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", "csf", "white_matter", "global_signal"]:
        for suf in ["", "_derivative1", "_power2", "_derivative1_power2"]:
            cols[c + suf] = np.random.default_rng(0).standard_normal(T)
    for j in range(8):
        cols[f"a_comp_cor_{j:02d}"] = np.random.default_rng(j).standard_normal(T)
    cols["cosine00"] = np.ones(T)
    cols["framewise_displacement"] = np.abs(np.random.default_rng(1).standard_normal(T))
    df = pd.DataFrame(cols)
    meta = {f"a_comp_cor_{j:02d}": {"Mask": "combined" if j < 5 else "CSF", "VarianceExplained": 0.1 - 0.01 * j, "Retained": True} for j in range(8)}
    X, info = select_confounds(df, meta, {"confounds": "24P"})
    assert X.shape[1] == 24
    X, info = select_confounds(df, meta, {"confounds": "36P"})
    assert X.shape[1] == 36
    X, info = select_confounds(df, meta, {"confounds": "acompcor", "n_acompcor": 3})
    assert info["n_acompcor"] == 3 and X.shape[1] == 24 + 3 + 1 and info["acompcor_mask_used"] == "combined"
    X, info = select_confounds(df, None, {"confounds": "acompcor+gsr", "n_acompcor": 2})
    assert "global_signal" in info["regressors"]
    assert n_dummy_scans(pd.DataFrame({"non_steady_state_outlier00": [1], "non_steady_state_outlier01": [0]}), {"dummy_scans": "auto"}) == 2
    assert n_dummy_scans(df, {"dummy_scans": 3}) == 3


def test_clean_removes_confound_and_filters():
    rng = np.random.default_rng(0)
    T, tr = 300, 2.0
    conf = rng.standard_normal((T, 2))
    t = np.arange(T) * tr
    sig = np.sin(2 * np.pi * 0.03 * t)
    x = np.c_[sig + 2 * conf[:, 0], sig + 0.5 * conf[:, 1] + 0.01 * t]
    y = clean_timeseries(x, tr, conf, [0.008, 0.08], standardize="zscore_sample")
    from hopfec.models.signal import bandpass

    ref = bandpass(sig, tr, [0.008, 0.08])
    for j in range(2):
        assert np.corrcoef(y[:, j], ref)[0, 1] > 0.98


def _fake_fmriprep(tmp_path, sub="sub-01", T=60):
    d = tmp_path / "fmriprep" / sub / "func"
    d.mkdir(parents=True)
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    lab = np.zeros((6, 6, 6), np.int16)
    lab[:3] = 1
    lab[3:] = 2
    nib.save(nib.Nifti1Image(lab, aff), tmp_path / "atlas_dseg.nii.gz")
    rng = np.random.default_rng(0)
    bold = 100 + rng.standard_normal((6, 6, 6, T)).astype(np.float32)
    bold[:3] += 5 * np.sin(np.arange(T) / 5)
    stem = f"{sub}_task-rest_space-MNI152NLin2009cAsym_res-2"
    nib.save(nib.Nifti1Image(bold, aff), d / f"{stem}_desc-preproc_bold.nii.gz")
    (d / f"{stem}_desc-preproc_bold.json").write_text(json.dumps({"RepetitionTime": 2.0}))
    nib.save(nib.Nifti1Image(np.ones((6, 6, 6), np.uint8), aff), d / f"{stem}_desc-brain_mask.nii.gz")
    conf = pd.DataFrame({c: rng.standard_normal(T) for c in ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]})
    conf["non_steady_state_outlier00"] = np.r_[1, np.zeros(T - 1)]
    conf.to_csv(d / f"{sub}_task-rest_desc-confounds_timeseries.tsv", sep="\t", index=False)
    return tmp_path / "fmriprep"


def test_find_and_extract(tmp_path):
    fp = _fake_fmriprep(tmp_path)
    runs = find_fmriprep_runs(fp, "01", task="rest", space="MNI152NLin2009cAsym", res="2")
    assert len(runs) == 1 and runs[0].tr == 2.0 and runs[0].confounds is not None and runs[0].mask is not None
    assert find_fmriprep_runs(fp, "01", task="rest", space="T1w") == []
    atlas = atlas_from_file(tmp_path / "atlas_dseg.nii.gz")
    ts, info = extract_run_timeseries(runs[0], atlas, {"confounds": "24P", "band": [0.008, 0.08], "dummy_scans": "auto"})
    assert ts.shape == (59, 2) and info["n_dummy"] == 1 and info["n_regressors"] == 6  # only 6 motion columns exist


def test_halfpipe_discovery(tmp_path):
    hp = tmp_path / "work" / "derivatives" / "halfpipe" / "sub-01" / "func"
    hp.mkdir(parents=True)
    f = hp / "sub-01_task-rest_feature-corrMatrix_atlas-schaefer100_timeseries.tsv"
    np.savetxt(f, np.random.default_rng(0).standard_normal((30, 4)), delimiter="\t")
    f.with_suffix(".json").write_text(json.dumps({"RepetitionTime": 1.5}))
    roots = halfpipe_roots(tmp_path / "work")
    assert roots["halfpipe"] == tmp_path / "work" / "derivatives" / "halfpipe"
    ts = find_halfpipe_timeseries(tmp_path / "work", "sub-01", atlas="schaefer100")
    assert len(ts) == 1 and ts[0].tr == 1.5 and ts[0].source == "halfpipe-timeseries"
    assert list_halfpipe_features(tmp_path / "work")["timeseries"] == {"feature-corrMatrix_atlas-schaefer100": 1}
    X, names = load_timeseries(f)
    assert X.shape == (30, 4) and names is None
