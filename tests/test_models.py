import numpy as np
import pytest

from hopfec.models.gec import fit_gec, initial_ec, make_mask
from hopfec.models.hopf_linear import analytic_moments, build_jacobian, linear_stability
from hopfec.models.hopf_nonlinear import simulate_hopf, simulated_moments
from hopfec.models.linear_gradient import _filter_weights, backward_moments, fit_linear_gradient, forward_moments, grads_to_params, loss_and_grad_moments
from hopfec.models.search import error_surface, grid_search
from hopfec.models.signal import bandpass, empirical_moments, fcd_distribution, lagged_correlation, metastability, peak_frequencies


def test_lagged_correlation_direction():
    rng = np.random.default_rng(0)
    T = 5000
    x = rng.standard_normal(T)
    y = np.r_[0.0, x[:-1]] + 0.1 * rng.standard_normal(T)  # y follows x by one sample: x leads y
    C = lagged_correlation(np.c_[x, y], 1)
    assert C[1, 0] > 0.8 and abs(C[0, 1]) < 0.1  # COV[i, j]: j leads i


def test_analytic_vs_simulation(small_system):
    s = small_system
    x = simulate_hopf(s["C"], 1.0, s["a"], s["omega"], s["tr"], 20000, seed=1, transient_s=100, linear=True)
    xf = bandpass(x, s["tr"], s["band"])
    FC_s = np.corrcoef(xf.T)
    COV_s = lagged_correlation(xf, 1)
    FC_a, COV_a = analytic_moments(s["C"], 1.0, s["a"], s["omega"], s["tr"], 1, s["beta"], filt={"band": s["band"]})
    iu = np.triu_indices(s["N"], 1)
    assert np.corrcoef(FC_a[iu], FC_s[iu])[0, 1] > 0.95
    assert np.corrcoef(COV_a.ravel(), COV_s.ravel())[0, 1] > 0.95
    assert linear_stability(s["C"], 1.0, s["a"], s["omega"])["stable"]


def test_forward_recursion_matches_eigen(small_system):
    s = small_system
    A = build_jacobian(s["C"], 1.0, s["a"], s["omega"])
    h2 = _filter_weights(s["tr"], s["band"])
    S0, St, _ = forward_moments(A, s["beta"], s["tr"], 1, h2)
    N = s["N"]
    sd = np.sqrt(np.diag(S0[:N, :N]))
    R = S0[:N, :N] / np.outer(sd, sd)
    Ct = St[:N, :N] / np.outer(sd, sd)
    FC_a, COV_a = analytic_moments(s["C"], 1.0, s["a"], s["omega"], s["tr"], 1, s["beta"], filt={"band": s["band"]})
    assert np.allclose(R, FC_a, atol=1e-8) and np.allclose(Ct, COV_a, atol=1e-8)


def test_gradient_finite_difference(small_system):
    s = small_system
    N = s["N"]
    rng = np.random.default_rng(3)
    FC_emp = rng.uniform(-0.2, 0.8, (N, N))
    FC_emp = 0.5 * (FC_emp + FC_emp.T)
    np.fill_diagonal(FC_emp, 1)
    COV_emp = rng.uniform(-0.2, 0.8, (N, N))
    h2 = _filter_weights(s["tr"], s["band"])
    a = np.full(N, s["a"])

    def L(C, a_, w):
        A = build_jacobian(C, 1.0, a_, w)
        S0, St, _ = forward_moments(A, s["beta"], s["tr"], 1, h2, keep_cache=False)
        return loss_and_grad_moments(S0, St, N, FC_emp, COV_emp)[0]

    A = build_jacobian(s["C"], 1.0, a, s["omega"])
    S0, St, cache = forward_moments(A, s["beta"], s["tr"], 1, h2)
    _, G_S0, G_St, _, _ = loss_and_grad_moments(S0, St, N, FC_emp, COV_emp)
    gC, ga, gw = grads_to_params(backward_moments(G_S0, G_St, A, s["tr"], 1, h2, cache), N, 1.0)
    eps = 1e-6
    for (i, j) in [(0, 1), (3, 2)]:
        Cp, Cm = s["C"].copy(), s["C"].copy()
        Cp[i, j] += eps
        Cm[i, j] -= eps
        fd = (L(Cp, a, s["omega"]) - L(Cm, a, s["omega"])) / (2 * eps)
        assert abs(gC[i, j] - fd) < 1e-5 * max(1, abs(fd))
    ap, am = a.copy(), a.copy()
    ap[2] += eps
    am[2] -= eps
    assert abs(ga[2] - (L(s["C"], ap, s["omega"]) - L(s["C"], am, s["omega"])) / (2 * eps)) < 1e-5 * max(1, abs(ga[2]))
    wp, wm = s["omega"].copy(), s["omega"].copy()
    wp[1] += eps
    wm[1] -= eps
    assert abs(gw[1] - (L(s["C"], a, wp) - L(s["C"], a, wm)) / (2 * eps)) < 1e-5 * max(1, abs(gw[1]))


def test_gradient_fit_recovers_truth(small_system):
    s = small_system
    N = s["N"]
    FC, COV = analytic_moments(s["C"], 1.0, s["a"], s["omega"], s["tr"], 1, s["beta"], filt={"band": s["band"]})
    mask = make_mask(0.5 * (s["C"] + s["C"].T), N, "sc")
    C0 = initial_ec(0.5 * (s["C"] + s["C"].T), N, mask, "sc", 0.2)
    r = fit_linear_gradient(FC, COV, C0, mask, s["omega"], s["tr"], 1, s["a"], s["beta"], s["band"], max_iter=1500)
    assert np.corrcoef(r.C[mask], s["C"][mask])[0, 1] > 0.95
    assert r.metrics["fc_corr"] > 0.99


def test_gec_heuristic_improves_fit(small_system):
    s = small_system
    N = s["N"]
    FC, COV = analytic_moments(s["C"], 1.0, s["a"], s["omega"], s["tr"], 1, s["beta"], filt={"band": s["band"]})
    SC = 0.5 * (s["C"] + s["C"].T)
    mask = make_mask(SC, N, "sc")
    C0 = initial_ec(SC, N, mask, "sc", 0.2)
    res = fit_gec(FC, COV, C0, lambda C: analytic_moments(C, 1.0, s["a"], s["omega"], s["tr"], 1, s["beta"], filt={"band": s["band"]}), mask,
                  eps_fc=0.002, eps_tau=0.002, max_iter=300, patience=50)
    assert res.metrics["fit_rmse"] < res.metrics["initial_fit_rmse"]
    assert res.C.max() == pytest.approx(0.2)


def test_nonlinear_simulation_and_search(small_system):
    s = small_system
    x = simulate_hopf(s["C"], 1.0, s["a"], s["omega"], s["tr"], 300, seed=2, transient_s=50)
    assert x.shape == (300, s["N"]) and np.all(np.isfinite(x))
    emp = empirical_moments(x, s["tr"], s["band"], 1)
    emp["fcd"] = fcd_distribution(emp["filtered"], 20, 2)
    df, best = grid_search("linear", 0.5 * (s["C"] + s["C"].T), s["omega"], emp, s["tr"], [0.5, 1.0, 1.5], None, s["band"], n_jobs=1)
    assert len(df) == 3 and "fit_rmse" in df and best["G"] in (0.5, 1.0, 1.5)
    df2, best2 = grid_search("nonlinear", 0.5 * (s["C"] + s["C"].T), s["omega"], emp, s["tr"], [1.0], [-0.05, -0.02], s["band"], n_jobs=1, metric="fcd_ks", fcd_cfg={"window_tr": 20, "step_tr": 2})
    Gs, As, surf = error_surface(df2, "fcd_ks")
    assert surf.shape == (1, 2) and np.all(np.isfinite(surf))
    FC, COV = simulated_moments(s["C"], 1.0, s["a"], s["omega"], s["tr"], 100, s["band"], 1, n_sim=2)
    assert FC.shape == (s["N"], s["N"]) and metastability(x) > 0


def test_peak_frequencies():
    tr = 1.0
    t = np.arange(600) * tr
    x = np.c_[np.sin(2 * np.pi * 0.05 * t), np.sin(2 * np.pi * 0.08 * t)] + 0.1 * np.random.default_rng(0).standard_normal((600, 2))
    f, _ = peak_frequencies(x, tr, [0.01, 0.2], smooth_hz=0.0)
    assert abs(f[0] - 0.05) < 0.004 and abs(f[1] - 0.08) < 0.004
