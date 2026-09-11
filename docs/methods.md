# Methods

## Empirical statistics (per participant, averaged over runs)

* band-pass (2nd-order Butterworth, zero phase) `model.filter_band` (default 0.008–0.08 Hz);
* `FC = corr(x)`; lagged correlation `COVtau[i,j] = <x_i(t+τ) x_j(t)> / (σ_i σ_j)` with
  `τ = model.tau_tr` TRs (entry (i,j) > 0: j leads i);
* node frequencies `f_j` = peak of the Gaussian-smoothed power spectrum in `model.freq_band`;
  group spectra are averaged before taking the peak;
* FCD distribution (sliding windows) and metastability (SD of the Kuramoto order parameter) for the
  `fcd_ks` / `meta_diff` search metrics.

## Hopf whole-brain model (Deco et al. 2017)

```
dz_j/dt = (a_j + i ω_j − |z_j|²) z_j + G Σ_k C_jk (z_k − z_j) + β η_j(t),   x_j = Re z_j
```
`a = -0.02`, `β = 0.02`, `C` scaled to max 0.2, Euler–Maruyama with `dt ≤ 0.1 s` (the step is reduced
automatically so that `dt (|a| + G s_j + ω_j) ≤ 0.25`), sampled every TR after a transient, then
filtered like the data.  The numba kernel simulates ~10⁴ steps/s per 100 nodes.

## Linear Hopf model (Ponce-Alvarez & Deco 2024) and filter-consistent moments

Around `z = 0` (valid while the Jacobian `A` is stable, i.e. all its eigenvalues have negative real part;
all `a_j < 0` is sufficient, and a node with `a_j > 0` is admissible as long as its coupled in-strength
`G sum_k C_jk` exceeds `a_j`) the model is `dX = A X dt + β dW` with
`A = [[diag(a − G s) + G C, −diag(ω)], [diag(ω), diag(a − G s) + G C]]`.  The stationary covariance
solves `A Σ + Σ Aᵀ + β² I = 0` and `Cov(X(t+τ), X(t)) = expm(A τ) Σ`.  Because empirical BOLD is
band-pass filtered, the model's TR-sampled covariance sequence is convolved with the filter's
autocorrelation `h2` (impulse response of `filtfilt`): `Σ_filt(τ) = Σ_m h2(m) c(τ − m)`, `c(k) =
expm(A k TR) Σ` — computed via an eigendecomposition (fast, used in the parameter search) or via the
recursion `c(k) = P c(k−1)` (used for gradients).  Verified against long simulations (corr > 0.99).

## Estimating effective connectivity

**Exact-gradient fit (linear model, default, `model.linear.method: gradient`)**
Loss `L = ½ w_fc Σ_{i≠j}(FC_sim − FC_emp)² + ½ w_τ Σ_ij (COVtau_sim − COVtau_emp)² [+ λ_sc/2 ‖C − C_prior‖²
+ λ_l1 Σ C]`; gradients w.r.t. `C`, `a` and `ω` by the adjoint method (back-propagation through the lag
recursion, the adjoint Fréchet derivative of `expm`, and the adjoint Lyapunov equation); L-BFGS-B with
`C ≥ 0` on the allowed links (structural links + homotopic pairs by default, `model.gec.mask`).  `G` is
absorbed in `C`: the initial value is `G* · SC` from the search; the output `EC` is in these units and
`ECnorm` is rescaled to max 0.2 (`G_eff = max(EC)/0.2`).  Node frequencies are co-estimated by default
(`fit_omega: true`, initialised from spectral peaks) because errors in ω create spurious lagged
asymmetries.

**Heuristic GEC iteration (`method: gec`)**
`C_ij ← C_ij + ε_FC (FC_emp − FC_sim)_ij + ε_τ (COVtau_emp − COVtau_sim)_ij`, `C ≥ 0`, rescaled to
max 0.2, best iterate kept (Deco, Kringelbach et al. 2019–2021).  With `accept_only_improving`
(default for the non-linear model) an update is kept only if it lowers the error and the step sizes
are halved otherwise.

**Non-linear model (`model.nonlinear.method`)**
Simulations use *common random numbers* (the same noise realisation at every iteration), so that the
error trace reflects the coupling and not the simulation noise; the noise floor (SD of the error over
`n_noise_seeds` realisations at the initial coupling) is reported and a fit whose improvement stays
within 2 SD of it is flagged (`improvement_below_noise`; raise `n_sim`).  `surrogate` (default)
back-propagates the non-linear residuals through the linear model's adjoint at the current coupling
and takes accept-if-improving steps with an adaptive step size — a genuine descent direction for the
non-linear loss; `gec` is the heuristic above.  The non-linear fit is initialised from the linear EC
and node frequencies (`init: linear`), from the structural prior as in the published GEC fits (`init: sc`), or
from both (`init: both`: the better fit is kept and the metrics report the fit error of each start and the
correlation / relative difference of the two ECs, so that a dependence on the initialisation is visible).

### Validation on synthetic ground truth (N = 20, 8 "participants" × 600 volumes, TR 2 s)

| estimator | corr(EC, C_true) | corr of antisymmetric parts (direction) |
|---|---|---|
| structural prior (symmetrised truth) | 0.52 | – |
| heuristic GEC, linear moments | 0.43 | −0.07 |
| heuristic GEC, non-linear simulations | 0.42 | −0.08 |
| exact gradient, ω known | 0.67 | 0.63 |
| exact gradient, ω from spectral peaks | 0.27 | 0.13 |
| exact gradient, ω co-estimated (default) | **0.77** | **0.74** |
| exact gradient on noise-free model moments | 0.99 | 0.99 |

The heuristic rule improves the FC fit but does not recover the direction of coupling in this setting
(its lagged-covariance asymmetry is dominated by frequency differences); the exact-gradient linear
estimator does.  Reproduce with `tests/test_models.py::test_gradient_fit_recovers_truth` and the
scripts in the repository history.

## Hierarchical fitting, shrinkage and held-out validation

With `group.hierarchical: true` (default, linear model) the group-average statistics are fitted first;
every participant is then initialised from the EC of their own group (`hierarchical_prior: own`, or the
pooled `group-all` EC) and shrunk towards it with the penalty
`lambda_group * N^2 * ||C - C_group||^2 / ||C_group||^2` (relative, so lambda is comparable across
atlases).  `lambda_group: auto` chooses the weight by split-half cross-validation (odd/even runs, or
first/second half of a single run) on up to `cv_max_participants` participants over `lambda_grid`,
extending the grid upwards when the optimum is its largest value (`cv_lambda_*.tsv`).  With
`group.cross_validate: true` every participant also gets held-out metrics (`cv_fit_rmse`, `cv_fc_corr`,
`cv_train_fit_rmse` in the participants table): fit on one half, evaluate on the other.  On synthetic
ground truth this raised held-out FC correlation from 0.83 (no shrinkage) to 0.90 and the agreement of
the participant EC with the truth from 0.42 to 0.53–0.58.

## Alternative objectives for the linear model

* **Several lags** — `model.tau_tr: [1, 2, 3]`: the gradient fit matches FC and the lagged correlations
  at every listed lag (the search, GEC and non-linear fits use the first lag).  Lags beyond ~3 TRs add
  more noise than direction information.
* **Whittle likelihood** — `model.linear.method: whittle`: maximum likelihood on the cross-spectral
  matrices of the data inside the pass-band (all lags at once), with the data's filter response and
  the aliased spectral images of the TR-sampled process included, a fitted noise amplitude
  (`fit_beta: global | node | none`) and a white observation-noise floor (`fit_obs_noise`).  Analytic
  gradients w.r.t. C, a, omega, noise; `lambda_group` shrinkage applies as well.  On synthetic data with
  strong measurement noise it recovered the truth better than the moment fit (0.57 vs 0.41), without
  noise slightly worse (0.61 vs 0.70); per-node noise amplitudes overfit and are not the default.

## Heterogeneous bifurcation parameter from brain maps

`model.heterogeneity.enabled: true` with a parcel-wise map (TSV/CSV per parcel, an MNI-space NIfTI that is
parcellated with the atlas, or a `neuromaps` annotation: volumetric ones are parcellated directly,
surface ones need the atlas as CIFTI dlabel / GIFTI labels via `surface_labels`) makes the local
bifurcation parameter heterogeneous, `a_j = a0 + beta * z_j` with z the z-scored map (missing parcels stay
at a0).  The search axis for `a` becomes the map weight `beta` (grid `heterogeneity.beta`, border
extension, refinement, and the continuous optimiser with the chain-rule gradient sum_j (dL/da_j) z_j);
the resulting a_j profile is used in all subsequent fits (participants, groups, non-linear model) and saved
as `*_desc-heterogeneity_a.tsv` / `.json` (beta, G, range of a_j, number of supercritical nodes).  On a
synthetic system with a_j = -0.05 + 0.03 z_j the search recovered G and beta exactly.  Because the stability of
the linearisation depends on C through the diagonal `a_j - sum_k C_jk`, the EC gradient fit rejects any step
that makes the Jacobian unstable (large finite loss, zero gradient, so L-BFGS-B backtracks; the number of such
evaluations is reported), refuses to start from an unstable point, and an interpolated / continuous optimum
of the parameter search that turns out unstable is replaced by the validated grid optimum.  The default
`a0 = -0.1` leaves room for `beta * z_j` (z-scored maps span roughly +-2) before any node turns supercritical.  One map per run in
this version; several maps can be compared across runs (`fit_summary_*.json` records the map).

## Missing parcels

A parcel without usable data cannot be a node of the model: its coupling would shape the dynamics of every
other node while nothing constrains it.  The pipeline therefore fits the system of the parcels that have
valid time series in all participants (a common node set, so that participant ECs are estimates in the same
system and the hierarchical group-first fit and the group comparisons remain meaningful); SC normalisation and
the z-scoring of heterogeneity maps are done on the kept parcels.  Outputs are written on the full atlas with
NaN for the dropped parcels (`*_desc-nodes.tsv` documents the node set).  The alternative of keeping the
parcel as an uncoupled node was rejected: its frequency and bifurcation parameter would be undefined and its
simulated FC a fiction reported as numbers.

## Normalised directed transfer entropy (NDTE)

`hopfec ndte` computes, per participant (runs concatenated) and per group, the NDTE of Deco, Vidaurre &
Kringelbach (Nat Hum Behav 2021) - Gaussian transfer entropy from the past of a source to the present of a
target, conditioned on the target's own past and normalised by the information the joint past carries
about the present, with `max_lag` lags (default 10, as in the paper).  The implementation is a vectorised
equivalent of the original MATLAB (`ndte_example_surrogates_fixlags_cs.m`; identical to 1e-14 on test
data), with circular-shift surrogates (`n_surrogates`, default 100) giving z-scores, KDE and empirical
p-values and an FDR mask; in-, out- and total flow per region are reported (the basis of the
"functional rich club" / global-workspace ranking).  Convention: `NDTE[i, k]` = flow from k to i.

For the *linear* model the NDTE follows analytically from the filter-consistent lagged covariances
(block-Toeplitz lagged covariance -> Schur complements), so `ndte_corr` / `ndte_rmse` are available as
search metrics without simulation (r = 0.997 against long simulations), for the non-linear model they
are computed from the simulated BOLD.  The fitted ECs are additionally scored by `ndte_corr`
(`model.ndte.enabled: true`).

## Particle swarm optimisation

`model.search.continuous_method: pso` replaces the local optimiser by a constricted particle swarm
(`model.search.pso`: `n_particles`, `n_iter`, `stall_iter`) over (G, a or beta), evaluated in parallel with
common random numbers - the global, derivative-free scheme of `fitt_hopf_a_particleswarm.m`, applicable
to any metric including `ndte_corr` and `fcd_ks`.  Unlike the original script the swarm is not used for the
N^2 coupling entries (hopeless for a 20-particle swarm); the coupling is estimated with the gradient /
surrogate methods and validated against NDTE.

## Parameter search / error surface

`model.search.G` (and optionally `model.search.a`) define the coarse grid; the homogeneous model
`C = SC` is evaluated at every point in parallel (`joblib`); metrics: `fit_rmse` (default), `fc_corr`,
`fc_rmse`, `tau_rmse`, `fcd_ks` (KS distance of FCD distributions, needs simulation), `meta_diff`,
`combined`.  `search.level: group | participant | both` chooses whose FC is used.  The search then:

1. **validity guards** - linear-model points with a supercritical node (`a_j >= 0`, outside the regime in which
   the linearisation describes the Hopf model; `model.linear.require_subcritical: false` relaxes this to a
   stable coupled Jacobian) or an unstable linearisation are marked invalid (NaN); `G = 0` (uncoupled model) is evaluated for the surface but never selected
   (`allow_zero_G`); a selected optimum next to invalid points is flagged `at_validity_limit`;
2. **border detection + extension** - if the optimum lies on an edge of the grid the axis is extended
   beyond that edge with the same step (`extend_factor` x span, at most `max_extensions` times, within
   `G_min/G_max`, `a_min/a_max`) and the new points are evaluated; `border_action: warn` only reports;
3. **coarse-to-fine refinement** - `refine_points` values per axis within +- one coarse step of the
   optimum, then **parabolic interpolation** through the optimum and its neighbours (per axis, only if
   the parabola is convex and the vertex is bracketed);
4. **optional continuous optimisation** (`continuous: true`) - starting from the interpolated optimum,
   L-BFGS-B on (G[, a]) with the exact adjoint gradient of `fit_rmse²` for the linear model, or
   derivative-free Powell on any metric/model (fixed simulation seed, `continuous_max_fev`).

Outputs: `*_desc-errorsurface_table.tsv` (every evaluated point with `stage` and `valid`), regular
coarse/extended surface TSV/NPZ per metric, PNG with refinement points and the optimum actually used,
and `*_desc-search.json` (`best`, `border`, `extensions`, `best_coarse`, `best_refined`,
`interpolated`, `continuous`, `used`, `warnings`).  In the gradient fit, parameters that end on a box
constraint are counted (`n_a_at_lower/upper`, `n_omega_at_lower`, `n_C_at_cmax`) and logged.

## Group level

* fit to the group-average FC/COVtau/spectra with the group-average SC (`group.fit_group_average`);
* mean and SD of participant ECs (`group.mean_of_participants`);
* `group-all` pools everybody; per-group outputs for each label in `participants.tsv`;
* pairwise edge-wise Welch t-tests with Benjamini–Hochberg FDR (`group.compare`, `group.fdr_q`),
  optional permutation FWE (`group.n_perm`), plus global strength/asymmetry tests.

## References

* Deco G, Kringelbach ML, Jirsa VK, Ritter P (2017) The dynamics of resting fluctuations in the brain:
  metastability and its dynamical cortical core. *Sci Rep* 7:3095.
* Ponce-Alvarez A, Deco G (2024) The Hopf whole-brain model and its linear approximation. *Sci Rep*
  14:2615.
* Deco G, Cruzat J, Cabral J, Tagliazucchi E, Laufs H, Logothetis NK, Kringelbach ML (2019)
  Awakening: predicting external stimulation to force transitions between different brain states.
  *PNAS* 116:18088.
* Deco G, Sanz Perl Y, Vuust P, Tagliazucchi E, Kennedy H, Kringelbach ML (2021) Rare long-range
  cortical connections enhance human information processing. *Curr Biol* 31:4436.
* Elias GJB et al. (2024) A large normative connectome for exploring the tractographic correlates of
  focal brain interventions. *Sci Data* 11:353.
* Esteban O et al. (2019) fMRIPrep: a robust preprocessing pipeline for functional MRI. *Nat Methods*.
* Lindquist MA et al. (2019) Modular preprocessing pipelines can reintroduce artifacts into fMRI data.
  *Hum Brain Mapp* 40:2358.
