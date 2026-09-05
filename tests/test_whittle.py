import numpy as np
import pytest

from hopfec.models.gec import initial_ec, make_mask
from hopfec.models.hopf_nonlinear import simulate_hopf
from hopfec.models.signal import bandpass, functional_connectivity, lagged_correlation, peak_frequencies
from hopfec.models.whittle import band_selection, filter_gain, fit_linear_whittle, periodogram_matrices, whittle_loss_grad


def test_whittle_gradients(small_system):
    s = small_system
    x = simulate_hopf(s["C"], 1.0, s["a"], s["omega"], s["tr"], 400, seed=1, transient_s=50, linear=True)
    freqs, I = periodogram_matrices(bandpass(x, s["tr"], s["band"]), s["tr"])
    sel = band_selection(freqs, s["band"])
    fs, Is = freqs[sel], I[sel]
    gain = filter_gain(fs, s["tr"], s["band"])
    N = s["N"]
    q = np.full(N, s["beta"] ** 2)
    a = np.full(N, s["a"])
    loss, gC, ga, gw, gq = whittle_loss_grad(s["C"], a, s["omega"], q, fs, Is, gain=gain, tr=s["tr"], n_alias=1)

    def L(C, a_, w, q_):
        return whittle_loss_grad(C, a_, w, q_, fs, Is, gain=gain, tr=s["tr"], n_alias=1)[0]

    eps = 1e-6
    Cp, Cm = s["C"].copy(), s["C"].copy()
    Cp[0, 1] += eps
    Cm[0, 1] -= eps
    assert abs(gC[0, 1] - (L(Cp, a, s["omega"], q) - L(Cm, a, s["omega"], q)) / (2 * eps)) < 1e-5 * max(1, abs(gC[0, 1]))
    ap, am = a.copy(), a.copy()
    ap[2] += eps
    am[2] -= eps
    assert abs(ga[2] - (L(s["C"], ap, s["omega"], q) - L(s["C"], am, s["omega"], q)) / (2 * eps)) < 1e-5 * max(1, abs(ga[2]))
    qp, qm = q.copy(), q.copy()
    qp[1] += eps * 1e-3
    qm[1] -= eps * 1e-3
    fd = (L(s["C"], a, s["omega"], qp) - L(s["C"], a, s["omega"], qm)) / (2 * eps * 1e-3)
    assert abs(gq[1] - fd) < 1e-4 * max(1, abs(fd))


def test_whittle_recovers_direction():
    rng = np.random.default_rng(7)
    N, tr, band = 12, 2.0, [0.008, 0.08]
    Ctrue = rng.random((N, N)) * (rng.random((N, N)) < 0.4)
    np.fill_diagonal(Ctrue, 0)
    Ctrue *= 0.2 / Ctrue.max()
    SC = 0.5 * (Ctrue + Ctrue.T)
    SC *= 0.2 / SC.max()
    omega = 2 * np.pi * rng.uniform(0.04, 0.07, N)
    mask = make_mask(SC, N, "sc")
    C0 = initial_ec(SC, N, mask, "sc", 0.2)
    runs = [bandpass(simulate_hopf(Ctrue, 1.0, -0.02, omega, tr, 500, seed=k, transient_s=50), tr, band) for k in range(6)]
    FC = np.mean([functional_connectivity(r) for r in runs], 0)
    COV = np.mean([lagged_correlation(r, 1) for r in runs], 0)
    f_peak = np.mean([peak_frequencies(r, tr, band)[0] for r in runs], 0)
    r = fit_linear_whittle(runs, tr, C0, mask, 2 * np.pi * f_peak, band, fit_omega=True, fit_beta="global", max_iter=400, FC_emp=FC, COVtau_emp=COV)
    anti = lambda M: (M - M.T)[mask]  # noqa: E731
    assert np.corrcoef(r.C[mask], Ctrue[mask])[0, 1] > 0.6
    assert np.corrcoef(anti(r.C), anti(Ctrue))[0, 1] > 0.5
    assert r.metrics["fc_corr"] > 0.9 and "whittle_loss" in r.metrics
