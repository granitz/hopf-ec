# Configuration and portable paths

All settings live in one YAML file (`hopfec init-config` writes a commented template; every key has a
backward-compatible default in `hopfec/config.py`).  Values can be overridden on the command line:
`--set model.a=-0.05 --set sc.source=normative --output-dir /scratch/out`.

## Paths

```yaml
paths:
  root: .                                   # relative paths below resolve against this (default: folder of the YAML)
  bids_dir: ${HOPFEC_BIDS:-./bids}          # ${VAR} and ${VAR:-default} are expanded from the environment
  derivatives_dir: ./bids/derivatives       # fmriprep/, halfpipe/, qsiprep/, qsirecon*/ are found below it
  fmriprep_dir: null                        # explicit overrides when the layout differs
  halfpipe_dir: null                        # HALFpipe working dir, its derivatives/ or derivatives/halfpipe
  qsiprep_dir: null
  qsirecon_dir: null
  atlas_dir: ./atlases
  participants_tsv: null                    # default <bids_dir>/participants.tsv
  output_dir: ./derivatives/hopfec
  work_dir: ./work
  normative_connectome: ${DTOR_MAT:-}       # dTOR_fibers_vox_2_mm.mat or .mat.gz
  template_dir: null                        # optional folder with MNI templates (else TemplateFlow cache / $FSLDIR)
  fs_license: ${FS_LICENSE:-}
```

Path-like settings outside `paths` (`atlas.file`, `atlas.labels`, `sc.file`, `fmriprep.sif`,
`sc.normative.brain_mask/wm_probseg/template_dir`) are resolved the same way.

Moving a project to another server therefore only requires `export HOPFEC_BIDS=...` (or editing the
few absolute paths) — no code changes.  `hopfec inputs -c cfg.yaml` shows the resolved paths and what
was found for each participant.

## Inputs

* `input.source`: `auto` (own time series → HALFpipe time series → HALFpipe/fMRIPrep BOLD), `fmriprep`,
  `halfpipe`, `halfpipe-timeseries`, `timeseries` (`input.timeseries_glob`, `{sub}` placeholder).
* `input.task/space/res/session/run`, `input.tr` (overrides sidecars), `input.halfpipe_feature`,
  `input.halfpipe_setting`, `input.halfpipe_atlas`.
* `atlas.name` (substring / regex / space-insensitive tokens against the files in `atlas_dir`;
  the atlas whose template space equals `input.space` is preferred), or `atlas.file` + `atlas.labels`.

## Participants

`participants.tsv` (or `.csv`): first column participant ids (`sub-01` or `01`), second column group
(or name them with `participants.id_column` / `participants.group_column`).  Rows with an empty
group are ignored; `participants.include/exclude` and `--participant-label` subset the table.

## Compute

`compute.n_jobs` (joblib workers; `-1` = all cores).  Participant fits, parameter-search points and
time-series extraction run in parallel; each worker is single-threaded (loky limits BLAS threads).
For clusters use `scripts/slurm_array.sh` (one participant per task, `--participants-only`) followed by
`scripts/slurm_group.sh` (`--group-only`).

## Parameter search ranges

`model.search.G` / `model.search.a` accept `{start, stop, step}`, `{start, stop, num}` or `{values: [...]}`
(inclusive).  With `border_action: extend` (default) an optimum on a grid edge extends the grid on that
side (same step, `extend_factor` x span, up to `max_extensions`, capped by `G_min/G_max` and
`a_min/a_max`); `refine`/`refine_points` add a finer pass around the optimum; `interpolate` applies a
parabolic fit; `continuous` switches on a continuous (G, a) optimisation afterwards.  `G = 0` is never
selected unless `allow_zero_G: true`.  The chosen values and their provenance (`source`) are stored in
`*_desc-search.json`.
