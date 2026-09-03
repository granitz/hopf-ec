# Structural connectivity from the dTOR-985 normative connectome

`dTOR_fibers_vox_2_mm.mat` (Elias et al., Sci Data 2024; Germann-lab/dTOR-985-Connectome) stores
~11.8 M streamlines as a MATLAB cell array `fibers_vox`, each an N×3 matrix of **1-indexed voxel
indices in the 91×109×91 MNI152 2 mm grid** (the authors' `dTOR_compute_fiber_weights.m` indexes an
SPM-read volume directly with these values, so they are integers).

## Method (identical to the original `build_roi_to_roi_*.m` scripts)

1. every point is mapped to the atlas grid: `fiber voxel (1-indexed) → mm → atlas voxel`, using the
   fiber-grid affine and the atlas NIfTI affine (`fiber_to_atlas_matrix`).  For the Schaefer/Tian
   atlas in MNI152NLin2009cAsym 2 mm this reproduces the original constant offset of 3.25 voxels;
2. a streamline touching parcels {p1..pk} adds 1 to every pair (both directions) — `count_mode:
   touched` — and 1/length to the length-weighted matrix (`length_mode: npoints` = n_points × 2 mm as
   in the original; `polyline` = true path length).  `count_mode: endpoints` gives the MRtrix-style
   endpoint convention;
3. normalisations: `volnorm = count / sqrt(vol_i vol_j)`, `lencorr = Σ 1/length`,
   `combined = lencorr / sqrt(vol_i vol_j)`; diagonals are zero;
4. sanity checks (density, empty nodes, symmetry, L/R row-sum ratio, interhemispheric fraction), a
   figure, TSV/CSV copies and a MATLAB `.mat` with the original variable names.

Python: `hopfec sc normative -c cfg.yaml [--connectome file] [--fiber-grid auto|RAS|LAS]`.
MATLAB: `matlab/build_normative_sc.m` (function with a `cfg` struct; same outputs).  Both were
validated against a literal port of the original loops on synthetic streamlines (exact match).

The Python version streams v7.3 (HDF5) files chunk by chunk and writes a compact cache
(`work/normative/*_cache_points.npy`, uint8 for integer coordinates) so that further atlases or
orientation checks take seconds.  MATLAB v7 files are loaded whole (as MATLAB does).

## The voxel-index convention (`sc.normative.fiber_grid`) — please read

The original script assumed that voxel index `i` increases with MNI `x` (`RAS`: mm = 2 (i−1) − 90).
FSL's `MNI152_T1_2mm` and SPM's canonical MNI152 2 mm images, however, store the x axis flipped
(`LAS`: mm = 92 − 2 i; voxel (1,1,1) is at x = +90 mm), and the dTOR scripts read the reference
volume with SPM (`spm_read_vols`) and index it directly with `fibers_vox`, while
`dTOR_create_trk.m` labels the voxel order `LAS`.  If the streamlines are in `LAS` storage order, the
`RAS` assumption mirrors the connectome left↔right; symmetric summary checks (L/R row-sum ratio) cannot
detect this.

`fiber_grid: auto` (default) computes the streamline density in the fiber grid and compares it with
the MNI152NLin6Asym brain mask and white-matter/T1w template under both conventions (the templates are
asymmetric: occipital/frontal petalia).  The convention with the higher density–white-matter
correlation is used; both scores are printed and stored in the JSON (`orientation_diagnostics`).  Set
`fiber_grid: RAS` to reproduce the original results exactly, or `LAS`/a custom 4×4 affine
(`fiber_affine`) when the convention is known.

## Template mismatch (MNI152NLin6Asym vs MNI152NLin2009cAsym)

The fibers live in the FSL/HCP MNI152NLin6Asym grid, many atlases in MNI152NLin2009cAsym.  Aligning the
two by affine only (the original approach) leaves errors of a few mm.  With ANTs
(`antsApplyTransforms`) and the TemplateFlow transform `tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym`
available, `warp_atlas: auto` warps the atlas (nearest neighbour) into the fiber grid first; otherwise
a warning is logged.  Simplest alternative: use the atlas release in MNI152NLin6Asym (Schaefer and Tian
are distributed in both spaces) — `hopfec atlases` shows the space of every file.
