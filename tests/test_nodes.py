import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from hopfec.atlases import atlas_from_file
from hopfec.nodes import NodeSet, determine_node_set, identity_node_set, node_validity


def _atlas(tmp_path):
    from tests.test_pipeline_cli import _make

    _make(tmp_path)
    return atlas_from_file(tmp_path / "atlases" / "toy_dseg.nii.gz")


def test_reduce_expand_roundtrip(tmp_path):
    atlas = _atlas(tmp_path)
    ns = identity_node_set(atlas)
    ns.keep[[2, 7]] = False
    assert ns.n_kept == 8 and ns.dropped_names == ["L_r2", "R_r2"]
    M = np.arange(100, dtype=float).reshape(10, 10)
    r = ns.reduce_mat(M)
    assert r.shape == (8, 8) and r[0, 0] == 0 and r[1, 1] == 11 and r[2, 2] == 33
    E = ns.expand_mat(r)
    assert E.shape == (10, 10) and np.isnan(E[2]).all() and np.isnan(E[:, 7]).all() and np.array_equal(E[np.ix_(ns.idx, ns.idx)], r)
    assert np.array_equal(ns.reduce_mat(E), r)  # tolerant to the full size
    v = np.arange(10, dtype=float)
    assert np.array_equal(ns.reduce_vec(v), [0, 1, 3, 4, 5, 6, 8, 9]) and np.isnan(ns.expand_vec(ns.reduce_vec(v))[2])
    X = np.random.default_rng(0).standard_normal((50, 10))
    assert ns.reduce_ts(X).shape == (50, 8) and np.array_equal(ns.reduce_ts(X)[:, 2], X[:, 3])
    sub = ns.subset_atlas(atlas)
    assert sub.n_parcels == 8 and sub.region_names == ns.kept_names and list(sub.label_ids) == [1, 2, 4, 5, 6, 7, 9, 10]
    assert len(sub.homotopic_pairs()) == 4 and atlas.n_parcels == 10  # original untouched
    t = ns.table()
    assert list(t.columns) == ["index", "name", "kept", "reason", "coverage"] and t["kept"].sum() == 8


def test_node_validity_and_determine(tmp_path):
    atlas = _atlas(tmp_path)
    X = np.random.default_rng(1).standard_normal((30, 10))
    X[:, 1] = np.nan
    X[5, 4] = np.nan
    X[:, 6] = 3.0
    v = node_validity(X)
    assert not v[1] and not v[4] and not v[6] and v[[0, 2, 3, 5, 7, 8, 9]].all()
    valid = {"sub-01": np.ones(10, bool), "sub-02": np.ones(10, bool), "sub-03": np.ones(10, bool)}
    valid["sub-02"][3] = False
    # default: a parcel invalid in any participant is dropped for everyone
    ns = determine_node_set(valid, atlas, {"exclude": ["R_r4", 1]})
    assert ns.n_kept == 7 and not ns.keep[[0, 3, 9]].any() and "nodes.exclude" in ns.reasons[9] and "1/3" in ns.reasons[3]
    assert ns.excluded_participants == {} and np.isclose(ns.coverage[3], 2 / 3)
    # min_coverage below 1: keep the parcel, exclude the participant lacking it
    ns2 = determine_node_set(valid, atlas, {"min_coverage": 0.5})
    assert ns2.n_kept == 10 and ns2.excluded_participants == {"sub-02": ["L_r3"]}
    # auto off: only explicit exclusions; participant with invalid kept node still excluded
    ns3 = determine_node_set(valid, atlas, {"auto": False})
    assert ns3.n_kept == 10 and "sub-02" in ns3.excluded_participants
    with pytest.raises(RuntimeError):
        determine_node_set({"sub-01": np.zeros(10, bool)}, atlas, {})


def test_pipeline_with_missing_parcel(tmp_path):
    """One participant has a NaN parcel: the parcel is dropped from the model, outputs stay on the full atlas."""
    from tests.test_pipeline_cli import _make

    _make(tmp_path)
    f = tmp_path / "derivatives" / "hopfec" / "sub-02" / "func" / "sub-02_task-rest_atlas-toydseg_desc-clean_timeseries.tsv"
    df = pd.read_csv(f, sep="\t")
    df["r3"] = np.nan
    df.to_csv(f, sep="\t", index=False)
    r = subprocess.run([sys.executable, "-m", "hopfec.cli", "fit", "-c", str(tmp_path / "cfg.yaml"), "-q", "--set", "group.cross_validate=false"],
                       capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-3000:]
    out = tmp_path / "derivatives" / "hopfec"
    nodes = pd.read_csv(out / "atlas-toydseg_desc-nodes.tsv", sep="\t")
    assert len(nodes) == 10 and nodes["kept"].sum() == 9 and nodes.loc[3, "name"] == "L_r3" and nodes.loc[3, "kept"] == 0 and "1/4" in nodes.loc[3, "reason"]
    summ = json.loads((out / "fit_summary_atlas-toydseg.json").read_text())
    assert summ["n_parcels"] == 10 and summ["n_model_nodes"] == 9 and summ["nodes"]["dropped"] == ["L_r3"] and summ["nodes"]["excluded_participants"] == {}
    for tag in ("hopflinear", "hopf"):
        ec = pd.read_csv(out / "models" / f"model-{tag}" / "sub-02" / f"sub-02_atlas-toydseg_model-{tag}_desc-EC_connectivity.tsv", sep="\t")
        assert ec.shape == (10, 10) and list(ec.columns)[3] == "L_r3"
        M = ec.to_numpy(float)
        assert np.isnan(M[3]).all() and np.isnan(M[:, 3]).all() and np.isfinite(np.delete(np.delete(M, 3, 0), 3, 1)).all()
        fit = json.loads((out / "models" / f"model-{tag}" / "sub-01" / f"sub-01_atlas-toydseg_model-{tag}_desc-fit.json").read_text())
        assert len(fit["omega_fit"]) == 10 and fit["omega_fit"][3] is None or np.isnan(fit["omega_fit"][3])
        assert fit["n_model_nodes"] == 9
        mean = pd.read_csv(out / "models" / f"model-{tag}" / "group-all" / f"group-all_atlas-toydseg_model-{tag}_desc-meanEC_connectivity.tsv", sep="\t").to_numpy(float)
        assert mean.shape == (10, 10) and np.isnan(mean[3]).all()
        t = pd.read_csv(out / "models" / f"model-{tag}" / "comparisons" / f"group-A_vs_group-B_atlas-toydseg_model-{tag}_desc-tstat_connectivity.tsv", sep="\t").to_numpy(float)
        assert t.shape == (10, 10) and np.isnan(t[3]).all()
    emp = pd.read_csv(out / "sub-02" / "func" / "sub-02_atlas-toydseg_desc-empiricalFC_connectivity.tsv", sep="\t").to_numpy(float)
    assert np.isnan(emp[3]).all() and np.isfinite(np.delete(np.delete(emp, 3, 0), 3, 1)).all()
    # group-only re-run reads the full-atlas files back
    r2 = subprocess.run([sys.executable, "-m", "hopfec.cli", "fit", "-c", str(tmp_path / "cfg.yaml"), "-q", "--group-only", "--model", "linear"], capture_output=True, text=True, cwd=tmp_path)
    assert r2.returncode == 0, r2.stderr[-2000:]
    # min_coverage < 1: keep the parcel, exclude sub-02
    r3 = subprocess.run([sys.executable, "-m", "hopfec.cli", "fit", "-c", str(tmp_path / "cfg.yaml"), "-q", "--model", "linear", "--set", "nodes.min_coverage=0.5",
                         "--set", "group.cross_validate=false"], capture_output=True, text=True, cwd=tmp_path)
    assert r3.returncode == 0, r3.stderr[-3000:]
    summ3 = json.loads((out / "fit_summary_atlas-toydseg.json").read_text())
    assert summ3["n_model_nodes"] == 10 and summ3["nodes"]["excluded_participants"] == {"sub-02": ["L_r3"]}
    assert "sub-02" in summ3["participants"] and summ3["participants"]["sub-02"]["errors"]
