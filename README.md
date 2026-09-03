# hopfec — whole-brain effective connectivity with Hopf models

`hopfec` turns resting-state fMRI (BIDS) plus structural connectivity into **participant-level and
group-level effective connectivity (EC)** using the **linear Hopf whole-brain model** and the
**non-linear Hopf whole-brain model**.  It covers the whole chain:

| step | what | how |
|---|---|---|
| 1 | BIDS BOLD → preprocessed BOLD | fMRIPrep launched in Docker/Apptainer (`hopfec fmriprep`) |
| 1.2 | post-processing + parcellation | aCompCor/24P/36P (+GSR) nuisance regression with simultaneous band-pass, dummy-scan removal, optional scrubbing regressors (`hopfec timeseries`) |
| 2 | HALFpipe inputs | HALFpipe atlas time series (`*_feature-*_atlas-*_timeseries.tsv`), HALFpipe *setting* BOLD or its internal fMRIPrep folder are discovered automatically |
| 3 | structural connectivity from DWI | existing QSIPrep/QSIRecon connectivity matrices (3.1) or probabilistic tractography with MRtrix3 (iFOD2 + ACT + SIFT2) or DIPY (3.2) |
| 4 | atlas catalogue | `hopfec atlases <folder>` lists every parcellation (space, resolution, parcels, label table, homotopic pairs) |
| 5 | participants | `participants.tsv` with `participant_id` (sub-*) and `group` |
| 6–9 | effective connectivity | parameter search (G × a error surface), participant-level EC, group-level EC, model-fit metrics and figures; parallel with joblib / SLURM arrays |
| 10 | normative SC | ROI×ROI SC from the dTOR-985 normative connectome (`dTOR_fibers_vox_2_mm.mat`), Python and MATLAB |

Everything is driven by one YAML file whose paths may be relative, `~` or `${ENV_VAR}`, so a project
moves between machines/servers without code changes (`docs/configuration.md`).

## Install

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e ".[all]"           # numpy, scipy, nibabel, nilearn, joblib, PyYAML, matplotlib, h5py, numba, dipy, templateflow, pytest
pytest                            # ~30 s, synthetic data only (24 tests)
```

External tools (optional, detected at run time): Docker/Apptainer (fMRIPrep), MRtrix3 + FSL + ANTs
(tractography, atlas warping), MATLAB (companion scripts in `matlab/`).

## Quick start

```bash
hopfec init-config project.yaml          # commented template; edit paths/atlas/model
hopfec atlases /path/to/atlases          # (4) which parcellations are available
hopfec participants -c project.yaml      # (5) participant/group table vs BIDS
hopfec fmriprep -c project.yaml --dry-run   # (1.2) prints the fMRIPrep container command
hopfec timeseries -c project.yaml        # post-process + parcellate (fMRIPrep or HALFpipe outputs)
hopfec sc normative -c project.yaml --connectome dTOR_fibers_vox_2_mm.mat.gz   # (10) template SC
hopfec sc tractography -c project.yaml --execute                              # (3.2) MRtrix3/DIPY
hopfec fit -c project.yaml               # (6-9) linear + non-linear Hopf EC, search, groups
hopfec run -c project.yaml               # stages listed in the config
```

`hopfec inputs -c project.yaml` prints what was discovered for every participant (BOLD runs,
time series, SC source) before anything heavy runs.

## Outputs (`paths.output_dir`, BIDS-derivatives style)

```
derivatives/hopfec/
  sub-01/func/sub-01_task-rest_atlas-<A>_desc-clean_timeseries.tsv (+.json)   parcellated, denoised BOLD
  sub-01/func/sub-01_atlas-<A>_desc-empiricalFC|empiricalCOVtau_connectivity.tsv, *_desc-empirical.json
  sub-01/dwi/sub-01_atlas-<A>_desc-tractography_weight-<w>_connectivity.tsv    participant SC (tractography)
  sc/atlas-<A>_desc-normative_weight-{count,volnorm,lencorr,combined}_connectivity.tsv (+.mat, .json, .png)
  models/model-hopflinear/   (linear Hopf)          models/model-hopf/   (non-linear Hopf)
    search/group-all_..._desc-errorsurface.png|.tsv|.npz, *_desc-search.json       error surface, best (G, a)
    sub-01/sub-01_..._desc-EC_connectivity.tsv        participant EC  (entry [i, j] = coupling j -> i)
    sub-01/..._desc-ECnorm_connectivity.tsv           same, rescaled to max 0.2 (Deco convention)
    sub-01/..._desc-fit.json|png, *_desc-history.tsv  fit metrics (FC corr/RMSE, lagged-corr fit), convergence
    participants_..._fit.tsv                          one row per participant
    group-<G>/group-<G>_..._desc-EC_connectivity.tsv  group EC fitted to the group-average statistics
    group-<G>/..._desc-meanEC|sdEC_connectivity.tsv   mean / SD of participant ECs
    comparisons/group-A_vs_group-B_..._desc-tstat|pvalue|sigFDR_connectivity.tsv (+.png)
  fit_summary_atlas-<A>.json
```

## Methods in one paragraph

Empirical BOLD is band-pass filtered (default 0.008–0.08 Hz); each participant yields FC, the
time-shifted correlation `COVtau[i,j] = corr(x_i(t+tau), x_j(t))` and node peak frequencies.  The
Hopf model `dz_j = (a_j + i w_j − |z_j|²) z_j + G Σ_k C_jk (z_k − z_j) + β η` is fitted with the
structural connectivity as prior.  **Linear model**: the fixed-point linearisation is an
Ornstein–Uhlenbeck process whose FC and lagged covariance follow from the Lyapunov equation and
`expm(A tau)`; the band-pass filter applied to the data is reproduced analytically (filter-consistent
moments) and the coupling matrix (optionally node frequencies and `a`) is estimated by
**exact-gradient L-BFGS-B** (adjoint through Lyapunov/expm/filter) — or, optionally, with the
published heuristic GEC iteration.  **Non-linear model**: Euler–Maruyama simulations (numba) with the
GEC update rule `C_ij += ε_FC (FC_emp − FC_sim) + ε_τ (COVtau_emp − COVtau_sim)`, initialised from the
linear solution.  The global coupling `G` (and optionally `a`) come from a parallel grid search whose
error surface is saved.  Group EC is fitted to group-average statistics and also reported as the
mean of participant ECs; groups are compared edge-wise (FDR) on the max-normalised EC by default
(`group.compare_on`).  Details, references and the validation on synthetic ground truth:
`docs/methods.md`.

## Structural connectivity from the dTOR-985 normative connectome

`hopfec sc normative` (Python) and `matlab/build_normative_sc.m` (MATLAB) implement the same method as
the original `build_roi_to_roi_*.m` scripts, generalised to any atlas grid via the two affines and
with the voxel-index convention of `fibers_vox` made explicit (`sc.normative.fiber_grid: RAS | LAS |
auto`).  Both implementations were checked against a literal port of the original loops (exact match),
and `matlab/hopf_linear_moments.m` reproduces the Python linear-model moments to 1e-11.  **Read `docs/normative_sc.md`** — the original script assumed `RAS` (x increases with the
voxel index); the data-driven check on the actual file shows the FSL/SPM `LAS` storage order
(density-template correlation 0.774 vs 0.729), so RAS-based matrices are left-right mirrored.
`auto` runs that check and reports the evidence.

## Layout

```
hopfec/            Python package (CLI: hopfec)
  config.py bids.py atlases.py timeseries.py pipeline.py stages.py cli.py plotting.py group.py
  preprocess/fmriprep.py      inputs/{fmriprep,halfpipe}.py
  sc/{normative,qsi,tractography,common}.py
  models/{signal,hopf_linear,linear_gradient,hopf_nonlinear,gec,search}.py
matlab/            build_normative_sc.m, hopf_linear_moments.m, hopf_gec_linear.m
config/example.yaml   scripts/slurm_*.sh   tests/   docs/
```
