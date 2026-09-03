import numpy as np
import nibabel as nib
import pandas as pd
from scipy.io import savemat

from hopfec.atlases import atlas_from_file
from hopfec.sc.common import accumulate_pairs, endpoint_pairs, prepare_sc, sc_summary
from hopfec.sc.normative import FiberStore, build_normative_sc, fiber_affine, fiber_to_atlas_matrix
from hopfec.sc.qsi import list_mat_keys, load_connectivity_mat, select_connectivity_key
from hopfec.sc.tractography import mrtrix_script, streamlines_to_connectivity


def _reference(fibers, atlas, labels, offset):
    n = len(labels)
    lm = {int(l): k for k, l in enumerate(labels)}
    conn = np.zeros((n, n))
    dims = np.array(atlas.shape)
    for coords in fibers:
        v = np.round(coords + offset).astype(int)
        inside = np.all(v >= 1, axis=1) & np.all(v <= dims, axis=1)
        v = v[inside] - 1
        hits = atlas[v[:, 0], v[:, 1], v[:, 2]] if len(v) else np.zeros(0, int)
        touched = np.unique(hits[hits > 0])
        for a in range(len(touched)):
            for b in range(a + 1, len(touched)):
                conn[lm[touched[a]], lm[touched[b]]] += 1
                conn[lm[touched[b]], lm[touched[a]]] += 1
    return conn


def test_accumulate_pairs_matches_loops():
    rng = np.random.default_rng(0)
    n_s, n_roi = 200, 6
    sid = np.repeat(np.arange(n_s), 5)
    lab = rng.integers(0, n_roi, len(sid))
    count, w, nconn = accumulate_pairs(sid, lab, n_s, n_roi, np.ones(n_s))
    ref = np.zeros((n_roi, n_roi))
    for s in range(n_s):
        t = np.unique(lab[sid == s])
        for a in range(len(t)):
            for b in range(a + 1, len(t)):
                ref[t[a], t[b]] += 1
                ref[t[b], t[a]] += 1
    assert np.array_equal(count, ref) and np.array_equal(w, ref)
    c2, _, n2 = endpoint_pairs(np.array([0, 1, 2, -1]), np.array([1, 1, 0, 0]), 3)
    assert c2[0, 1] == 1 and c2[2, 0] == 1 and n2 == 2


def test_normative_builder_matches_reference(tmp_path):
    rng = np.random.default_rng(1)
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    aff[:3, 3] = [-96.5, -132.5, -78.5]
    lab = np.zeros((97, 115, 97), np.int16)
    k = 1
    for i in range(20, 80, 15):
        for j in range(30, 100, 15):
            lab[i : i + 12, j : j + 12, 30:60] = k
            k += 1
    nib.save(nib.Nifti1Image(lab, aff), tmp_path / "atlas.nii.gz")
    atlas = atlas_from_file(tmp_path / "atlas.nii.gz")
    fibers = []
    for _ in range(400):
        L = rng.integers(5, 50)
        start = np.array([rng.integers(15, 75), rng.integers(25, 95), rng.integers(25, 60)], float)
        pts = np.clip(np.cumsum(np.vstack([start, rng.integers(-1, 2, (L, 3))]), 0)[:L], 1, [91, 109, 91])
        fibers.append(pts)
    cell = np.empty((len(fibers), 1), object)
    for i, f in enumerate(fibers):
        cell[i, 0] = f
    savemat(tmp_path / "fib.mat", {"fibers_vox": cell})
    ref = _reference(fibers, atlas.data(), atlas.label_ids, np.array([3.25, 3.25, 3.25]))
    res = build_normative_sc(atlas, tmp_path / "fib.mat", tmp_path / "out", {"fiber_grid": "RAS", "chunk_size": 50, "warp_atlas": "never"}, work_dir=tmp_path / "work")
    assert np.allclose(res["matrices"]["count"], ref)
    assert (tmp_path / "out").glob("*_weight-count_connectivity.tsv")
    M = fiber_to_atlas_matrix(fiber_affine("RAS"), aff)
    assert np.allclose(M[:3, 3], 2.25)  # = 3.25 - 1 (0-indexed)
    Ml = fiber_to_atlas_matrix(fiber_affine("LAS"), aff)
    assert np.isclose(Ml[0, 0], -1.0)
    # cache reused
    res2 = build_normative_sc(atlas, tmp_path / "fib.mat", tmp_path / "out", {"fiber_grid": "RAS", "chunk_size": 70, "warp_atlas": "never"}, work_dir=tmp_path / "work")
    assert np.allclose(res2["matrices"]["count"], ref)
    assert FiberStore(tmp_path / "fib.mat", cache_dir=tmp_path / "work").has_cache()


def test_prepare_sc_and_summary():
    M = np.array([[0, 3, 0], [3, 0, 1], [0, 1, 0]], float)
    P = prepare_sc(M, {"normalize": "max", "sc_max": 0.2, "symmetrize": True})
    assert P.max() == 0.2 and P[2, 1] == 0.2 / 3
    s = sc_summary(M, ["L", "R", "L"])
    assert s["symmetric"] and s["density"] > 0


def test_qsirecon_mat(tmp_path):
    M = np.random.default_rng(0).random((5, 5))
    savemat(tmp_path / "sub-01_connectivity.mat", {"schaefer100_sift_invnodevol_radius2_count_connectivity": M,
                                                     "schaefer100_sift_mean_length_connectivity": M * 2, "atlas_region_labels": np.array(["a", "b", "c", "d", "e"], object)})
    keys = list_mat_keys(tmp_path / "sub-01_connectivity.mat")
    assert select_connectivity_key(keys, atlas_hint="schaefer100") == "schaefer100_sift_invnodevol_radius2_count_connectivity"
    out, key, _ = load_connectivity_mat(tmp_path / "sub-01_connectivity.mat", "mean_length")
    assert key.endswith("mean_length_connectivity") and np.allclose(out, M * 2)


def test_streamlines_to_connectivity_and_mrtrix_script(tmp_path):
    lab = np.zeros((10, 10, 10), np.int16)
    lab[:3, :, :] = 1
    lab[7:, :, :] = 2
    img = nib.Nifti1Image(lab, np.eye(4))
    sl = [np.c_[np.linspace(0, 9, 20), np.full(20, 5.0), np.full(20, 5.0)], np.c_[np.full(5, 1.0), np.linspace(1, 3, 5), np.full(5, 5.0)]]
    res = streamlines_to_connectivity(sl, img, np.array([1, 2]), endpoints_only=True)
    assert res["count"][0, 1] == 1 and res["count"][1, 0] == 1 and res["n_connecting_streamlines"] == 1
    txt = mrtrix_script(tmp_path / "dwi.nii.gz", tmp_path / "b.bval", tmp_path / "b.bvec", tmp_path / "o", tmp_path / "atlas.nii.gz", {"n_streamlines": 1000}, preprocessed=False)
    assert "tckgen" in txt and "tcksift2" in txt and "tck2connectome" in txt and "-select 1000" in txt
