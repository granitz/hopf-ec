"""End-to-end: synthetic time series + SC -> hopfec fit (both models, two groups) through the CLI."""
import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from hopfec.models.hopf_nonlinear import simulate_hopf


def _make(root: Path):
    rng = np.random.default_rng(5)
    N, TR, T = 10, 2.0, 200
    lab = np.zeros((10, 4, 4), np.int16)
    for k in range(N):
        lab[k, :, :] = k + 1
    (root / "atlases").mkdir(parents=True)
    nib.save(nib.Nifti1Image(lab, np.diag([2.0, 2.0, 2.0, 1.0])), root / "atlases" / "toy_dseg.nii.gz")
    pd.DataFrame({"index": range(1, N + 1), "name": [f"L_r{k}" for k in range(5)] + [f"R_r{k}" for k in range(5)]}).to_csv(root / "atlases" / "toy_dseg.tsv", sep="\t", index=False)
    C = rng.random((N, N)) * (rng.random((N, N)) < 0.4)
    np.fill_diagonal(C, 0)
    C *= 0.2 / C.max()
    np.savetxt(root / "sc.tsv", 0.5 * (C + C.T), delimiter="\t")
    omega = 2 * np.pi * rng.uniform(0.04, 0.07, N)
    subs = [("sub-01", "A"), ("sub-02", "A"), ("sub-03", "B"), ("sub-04", "B")]
    pd.DataFrame({"participant_id": [s for s, _ in subs], "group": [g for _, g in subs]}).to_csv(root / "participants.tsv", sep="\t", index=False)
    for k, (sub, g) in enumerate(subs):
        x = simulate_hopf(C, 1.0, -0.02, omega, TR, T, seed=k, transient_s=50)
        d = root / "derivatives" / "hopfec" / sub / "func"
        d.mkdir(parents=True)
        f = d / f"{sub}_task-rest_atlas-toydseg_desc-clean_timeseries.tsv"
        pd.DataFrame(x, columns=[f"r{j}" for j in range(N)]).to_csv(f, sep="\t", index=False)
        f.with_suffix(".json").write_text(json.dumps({"tr": TR}))
    (root / "cfg.yaml").write_text(
        "paths:\n  root: .\n  atlas_dir: ./atlases\n  participants_tsv: ./participants.tsv\n  output_dir: ./derivatives/hopfec\n"
        "atlas:\n  name: toy\ninput:\n  source: auto\n  task: rest\nsc:\n  source: file\n  file: ./sc.tsv\n"
        "model:\n  search:\n    G: {start: 0.5, stop: 1.5, step: 0.5}\n  linear:\n    max_iter: 100\n  nonlinear:\n    n_sim: 1\n    max_iter: 5\n    patience: 3\n"
        "compute:\n  n_jobs: 1\n")


def test_cli_fit(tmp_path):
    _make(tmp_path)
    r = subprocess.run([sys.executable, "-m", "hopfec.cli", "fit", "-c", str(tmp_path / "cfg.yaml"), "-q"], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-3000:]
    out = tmp_path / "derivatives" / "hopfec"
    summ = json.loads((out / "fit_summary_atlas-toydseg.json").read_text())
    assert summ["models"] == ["linear", "nonlinear"]
    for tag in ("hopflinear", "hopf"):
        assert (out / "models" / f"model-{tag}" / "sub-01" / f"sub-01_atlas-toydseg_model-{tag}_desc-EC_connectivity.tsv").exists()
        assert (out / "models" / f"model-{tag}" / "group-A" / f"group-A_atlas-toydseg_model-{tag}_desc-EC_connectivity.tsv").exists()
        assert (out / "models" / f"model-{tag}" / "group-all" / f"group-all_atlas-toydseg_model-{tag}_desc-meanEC_connectivity.tsv").exists()
        assert (out / "models" / f"model-{tag}" / "search" / f"group-all_atlas-toydseg_model-{tag}_desc-errorsurface.png").exists()
        assert (out / "models" / f"model-{tag}" / "comparisons" / f"group-A_vs_group-B_atlas-toydseg_model-{tag}_desc-tstat_connectivity.tsv").exists()
    fit = pd.read_csv(out / "models" / "model-hopflinear" / "participants_atlas-toydseg_model-hopflinear_fit.tsv", sep="\t")
    assert len(fit) == 4 and fit["fc_corr"].min() > 0.5
    # hierarchical fitting + cross-validation outputs
    assert (out / "models" / "model-hopflinear" / "cv_lambda_atlas-toydseg_model-hopflinear.tsv").exists()
    assert {"lambda_prior", "cv_fit_rmse", "cv_fc_corr", "cv_train_fit_rmse"} <= set(fit.columns) and fit["cv_fit_rmse"].notna().all()
    assert json.loads((out / "fit_summary_atlas-toydseg.json").read_text())["hierarchical"]["linear"]["lambda_group"] >= 0
    # group-only re-run works from the saved participant fits
    r2 = subprocess.run([sys.executable, "-m", "hopfec.cli", "fit", "-c", str(tmp_path / "cfg.yaml"), "-q", "--group-only", "--model", "linear"], capture_output=True, text=True, cwd=tmp_path)
    assert r2.returncode == 0, r2.stderr[-2000:]
