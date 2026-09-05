import numpy as np

from hopfec.models.hopf_linear import build_jacobian
from hopfec.models.hopf_nonlinear import simulate_hopf
from hopfec.models.linear_gradient import _filter_weights, forward_moments_multi
from hopfec.models.ndte import ndte, ndte_linear_model, ndte_with_surrogates
from hopfec.models.pso import pso_minimize
from hopfec.models.signal import bandpass


def _ndte_ref(X, L):
    T, N = X.shape
    y_pre = np.zeros((N, T - (L - 1), L))
    for i in range(N):
        for p in range(L):
            y_pre[i, :, p] = X[L - 1 - p : T - p, i]
    out = np.zeros((N, N))
    ld = lambda M: np.linalg.slogdet(np.cov(M.T))[1]  # noqa: E731
    for i in range(N):
        y = y_pre[i]
        Hy1 = np.log(np.var(y[:, 0], ddof=1))
        Iy = ld(y) - ld(y[:, 1:])
        for k in range(N):
            if k == i:
                continue
            z = np.hstack([y, y_pre[k][:, 1:]])
            out[i, k] = (Iy - (ld(z) - ld(z[:, 1:]))) / (Hy1 + ld(z[:, 1:]) - ld(z))
    return out


def _var_data(seed=0, T=600):
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0, 0, 0, 0], [0.4, 0.5, 0, 0, 0], [0, 0, 0.5, 0.3, 0], [0, 0, 0, 0.5, 0], [0.2, 0, 0, 0, 0.5]])
    X = np.zeros((T, 5))
    for t in range(1, T):
        X[t] = A @ X[t - 1] + rng.standard_normal(5)
    return X, A


def test_ndte_matches_literal_port_and_direction():
    X, A = _var_data()
    v = ndte(X, 10)
    assert np.abs(v - _ndte_ref(X, 10)).max() < 1e-10
    assert v[1, 0] > 3 * v[0, 1] and v[2, 3] > 2 * v[3, 2]      # row = target, column = source
    res = ndte_with_surrogates(X, 10, n_surrogates=60, seed=1)
    assert res["z"][1, 0] > 4 and res["sig_fdr"][1, 0] and res["in_flow"].shape == (5,)
    assert (res["p_kde"] >= 0).all() and (res["p_kde"] <= 1).all()


def test_analytic_linear_ndte(small_system):
    s = small_system
    L = 8
    A = build_jacobian(s["C"], 1.0, s["a"], s["omega"])
    h2 = _filter_weights(s["tr"], s["band"])
    S0, Sts, _ = forward_moments_multi(A, s["beta"], s["tr"], list(range(1, L)), h2)
    N = s["N"]
    model = ndte_linear_model([S0[:N, :N]] + [St[:N, :N] for St in Sts])
    x = bandpass(simulate_hopf(s["C"], 1.0, s["a"], s["omega"], s["tr"], 20000, seed=3, transient_s=50, linear=True), s["tr"], s["band"])
    sim = ndte(x, L)
    off = ~np.eye(N, dtype=bool)
    assert np.corrcoef(model[off], sim[off])[0, 1] > 0.95


def test_pso():
    f = lambda p: (p[0] - 1.3) ** 2 + 5 * (p[1] + 0.4) ** 2  # noqa: E731
    r = pso_minimize(f, [(-3, 3), (-3, 3)], n_particles=12, n_iter=80, seed=0)
    assert abs(r["x"][0] - 1.3) < 0.02 and abs(r["x"][1] + 0.4) < 0.02 and r["n_fev"] > 0
