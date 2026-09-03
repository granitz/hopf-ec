import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from hopfec.atlases import atlas_from_file, infer_hemisphere, read_label_table, scan_atlas_dir, select_atlas
from hopfec.bids import build_name, parse_entities, read_participants, validate_participants
from hopfec.config import load_config


def test_config_paths_env_and_relative(tmp_path, monkeypatch):
    monkeypatch.setenv("MYBIDS", str(tmp_path / "bids"))
    (tmp_path / "bids").mkdir()
    (tmp_path / "bids" / "participants.tsv").write_text("participant_id\tgroup\nsub-01\tA\n")
    cfg_file = tmp_path / "proj" / "cfg.yaml"
    cfg_file.parent.mkdir()
    cfg_file.write_text("paths:\n  bids_dir: ${MYBIDS}\n  atlas_dir: ../atlases\n  output_dir: out\nmodel:\n  a: -0.05\n")
    cfg = load_config(cfg_file)
    assert cfg["paths"]["bids_dir"] == str(tmp_path / "bids")
    assert cfg["paths"]["atlas_dir"] == str(tmp_path / "atlases")
    assert cfg["paths"]["output_dir"] == str(tmp_path / "proj" / "out")
    assert cfg["paths"]["participants_tsv"] == str(tmp_path / "bids" / "participants.tsv")
    assert cfg["model"]["a"] == -0.05 and cfg["model"]["beta"] == 0.02  # override + default kept
    cfg2 = load_config(cfg_file, {"model": {"a": -0.1}})
    assert cfg2["model"]["a"] == -0.1


def test_entities_roundtrip():
    e = parse_entities("sub-01_ses-2_task-rest_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")
    assert e["sub"] == "01" and e["ses"] == "2" and e["space"] == "MNI152NLin2009cAsym" and e["suffix"] == "bold" and e["extension"] == ".nii.gz"
    assert build_name({"sub": "01", "task": "rest", "atlas": "X", "desc": "clean"}, "timeseries", ".tsv") == "sub-01_task-rest_atlas-X_desc-clean_timeseries.tsv"


def test_participants(tmp_path):
    p = tmp_path / "participants.tsv"
    p.write_text("participant_id\tgroup\tage\n01\tpatient\t30\nsub-02\tcontrol\t40\nsub-03\tn/a\t50\n")
    df = read_participants(p)
    assert df["participant_id"].tolist() == ["sub-01", "sub-02"]
    assert df["group"].tolist() == ["patient", "control"]
    v = validate_participants(df, ["sub-01", "sub-04"])
    assert v["missing_in_bids"] == ["sub-02"] and v["missing_in_tsv"] == ["sub-04"]


def test_hemisphere_and_label_tables(tmp_path):
    assert infer_hemisphere("7Networks_LH_Vis_1") == "L"
    assert infer_hemisphere("HIP-rh") == "R"
    assert infer_hemisphere("Thalamus") == ""
    lut = tmp_path / "lut.txt"
    lut.write_text("# comment\n1 Left-Thalamus 0 118 14 0\n2 Right-Thalamus 0 118 14 0\n")
    t = read_label_table(lut)
    assert t["name"].tolist() == ["Left-Thalamus", "Right-Thalamus"] and t["hemisphere"].tolist() == ["L", "R"]
    plain = tmp_path / "names.txt"
    plain.write_text("regionA_L\nregionA_R\n")
    t = read_label_table(plain)
    assert t["index"].tolist() == [1, 2]


def _make_atlas(d: Path, name: str, affine, shape=(10, 12, 10), n=4):
    lab = np.zeros(shape, np.int16)
    for k in range(n):
        lab[k * 2 : k * 2 + 2, :4, :4] = k + 1
    nib.save(nib.Nifti1Image(lab, affine), d / name)
    return lab


def test_scan_and_select(tmp_path):
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    _make_atlas(tmp_path, "atlas-Foo_desc-4parcels_space-MNI152NLin2009cAsym_dseg.nii.gz", aff)
    pd.DataFrame({"index": [1, 2, 3, 4], "name": ["A_LH", "A_RH", "B_LH", "B_RH"]}).to_csv(tmp_path / "atlas-Foo_desc-4parcels_dseg.tsv", sep="\t", index=False)
    aff6 = np.diag([2.0, 2.0, 2.0, 1.0])
    aff6[:3, 3] = [-90, -126, -72]
    lab = np.zeros((91, 109, 91), np.int16)
    lab[40:50, 50:60, 40:50] = 1
    lab[60:70, 50:60, 40:50] = 2
    nib.save(nib.Nifti1Image(lab, aff6), tmp_path / "mygrid.nii.gz")
    atlases = scan_atlas_dir(tmp_path)
    by = {a.image.name: a for a in atlases}
    foo = by["atlas-Foo_desc-4parcels_space-MNI152NLin2009cAsym_dseg.nii.gz"]
    assert foo.n_parcels == 4 and foo.space == "MNI152NLin2009cAsym" and foo.labels_file is not None
    assert foo.homotopic_pairs() == [(0, 1), (2, 3)]
    grid = by["mygrid.nii.gz"]
    assert grid.space == "MNI152NLin6Asym" and grid.space_source == "grid"
    assert select_atlas(tmp_path, "Foo 4parcels").name.startswith("atlas-Foo")
    assert select_atlas(tmp_path, "mygrid").n_parcels == 2
