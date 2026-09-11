"""Border handling, extension, refinement/interpolation, validity guards and continuous optimisation."""
import numpy as np
import pytest

from hopfec.models.gec import initial_ec, make_mask
from hopfec.models.hopf_linear import analytic_moments
from hopfec.models.linear_gradient import fit_linear_gradient
from hopfec.models.search import (SEARCH_DEFAULTS, adaptive_search, border_flags, continuous_search, evaluate_points, extend_axis,
                                  interpolate_optimum, parabolic_vertex, refine_axis, select_best)


@pytest.fixture(scope="module")
def truth():
    rng = np.random.default_rng(11)
    N = 10
    C = rng.random((N, N)) * (rng.random((N, N)) < 0.6)
    np.fill_diagonal(C, 0)
    SC = 0.5 * (C + C.T)
    SC *= 0.2 / SC.max()
    omega = 2 * np.pi * rng.uniform(0.04, 0.07, N)
    G_true, a_true, tr, band = 2.5, -0.05, 2.0, [0.008, 0.08]
    FC, COV = analytic_moments(SC, G_true, a_true, omega, tr, 1, 0.02, filt={"band": band})
    emp = {"FC": FC, "COVtau": COV, "n_volumes": 400, "tr": tr}
    return dict(SC=SC, omega=omega, emp=emp, tr=tr, band=band, G_true=G_true, a_true=a_true)


def test_axis_helpers():
    vals = np.array([0.0, 0.5, 1.0])
    assert np.allclose(extend_axis(vals, "high", 0.5, 0.0, 10.0), [1.5])
    assert np.allclose(extend_axis(vals, "high", 2.0, 0.0, 1.6), [1.5])          # capped
    assert len(extend_axis(vals, "low", 0.5, 0.0, 10.0)) == 0                     # at the floor
    fine = refine_axis(vals, 0.5, 5, 0.0, 10.0)
    assert np.allclose(fine, [0.0, 0.25, 0.75, 1.0][1:3]) or set(np.round(fine, 3)) == {0.25, 0.75}
    x = np.array([1.0, 2.0, 3.0])
    assert abs(parabolic_vertex(x, (x - 2.3) ** 2, 1) - 2.3) < 1e-9
    assert parabolic_vertex(x, -((x - 2.3) ** 2), 1) is None                     # concave: no minimum


def test_border_extension_finds_optimum_beyond_initial_grid(truth):
    t = truth
    # budget of 2 extensions: the optimum (G = 2.5, error 0) is reached but sits on the outer edge -> flagged
    df, best, info = adaptive_search("linear", t["SC"], t["omega"], t["emp"], t["tr"], np.arange(0.0, 1.01, 0.25), None, t["band"],
                                     search_cfg={"refine": True, "interpolate": True, "continuous": False}, default_a=t["a_true"], n_jobs=1)
    assert info["extensions"], "grid should have been extended"
    assert info["best_refined"]["G"] > 1.0 and info["best_coarse"]["G"] == 1.0
    assert abs(best["G"] - t["G_true"]) < 0.06, best
    assert best["source"] in ("parabolic_interpolation", "refined_grid")
    assert info["border"]["on_border"] and any("border" in w for w in info["warnings"])
    # one more extension confirms the optimum is interior
    df, best, info = adaptive_search("linear", t["SC"], t["omega"], t["emp"], t["tr"], np.arange(0.0, 1.01, 0.25), None, t["band"],
                                     search_cfg={"refine": True, "interpolate": True, "continuous": False, "max_extensions": 3}, default_a=t["a_true"], n_jobs=1)
    assert abs(best["G"] - t["G_true"]) < 0.06 and not info["border"]["on_border"] and not info["warnings"]


def test_border_warn_only(truth):
    t = truth
    df, best, info = adaptive_search("linear", t["SC"], t["omega"], t["emp"], t["tr"], [0.0, 0.5, 1.0], None, t["band"],
                                     search_cfg={"border_action": "warn", "refine": False, "interpolate": False}, default_a=t["a_true"], n_jobs=1)
    assert best["G"] == 1.0 and info["border"]["G"]["high"] == "edge" and info["warnings"]


def test_validity_guard_and_zero_G(truth):
    t = truth
    df = evaluate_points("linear", [(0.0, -0.05), (0.5, -0.05), (0.5, 0.3)], t["SC"], t["omega"], t["emp"], t["tr"], t["band"])
    assert bool(df.loc[2, "valid"]) is False and np.isnan(df.loc[2, "fit_rmse"]) and "supercritical" in df.loc[2, "reason"]
    df_j = evaluate_points("linear", [(0.5, 0.3)], t["SC"], t["omega"], t["emp"], t["tr"], t["band"], require_subcritical=False)
    assert bool(df_j.loc[0, "valid"]) is False and "unstable" in df_j.loc[0, "reason"]   # Jacobian criterion
    best = select_best(df, "fit_rmse")
    assert best["G"] > 0
    flags = border_flags({"G": 0.5, "a": -0.05, "metric": "fit_rmse"}, df)
    assert flags["a"]["high"] == "invalid" and flags["a"]["low"] == "edge" and flags["at_validity_limit"]
    with pytest.raises(RuntimeError):
        select_best(df[df["valid"] == False], "fit_rmse")  # noqa: E712


def test_two_d_search_with_refinement(truth):
    t = truth
    df, best, info = adaptive_search("linear", t["SC"], t["omega"], t["emp"], t["tr"], [1.5, 2.0, 2.5, 3.0, 3.5], [-0.1, -0.05, -0.02, 0.0], t["band"],
                                     search_cfg={"refine": True, "refine_points": 5, "interpolate": True}, n_jobs=1)
    assert abs(best["G"] - t["G_true"]) < 0.15 and abs(best["a"] - t["a_true"]) < 0.02
    assert info["n_invalid"] >= 0 and any(s["stage"] == "refined" for s in info["stages"])


def test_continuous_optimisation(truth):
    t = truth
    res = continuous_search("linear", t["SC"], t["omega"], t["emp"], t["tr"], 2.0, -0.02, t["band"], fit_a=True, metric="fit_rmse",
                            bounds_G=(0.0, 10.0), bounds_a=(-1.0, 0.5))
    assert res["method"].startswith("L-BFGS-B") and abs(res["G"] - t["G_true"]) < 0.02 and abs(res["a"] - t["a_true"]) < 0.003
    res2 = continuous_search("linear", t["SC"], t["omega"], t["emp"], t["tr"], 2.0, t["a_true"], t["band"], fit_a=False, metric="fc_corr",
                             bounds_G=(0.5, 10.0), bounds_a=(-1.0, 0.5), max_fev=40)
    assert res2["method"].startswith("Powell") and abs(res2["G"] - t["G_true"]) < 0.2


def test_gradient_fit_reports_bound_hits(truth):
    t = truth
    N = t["SC"].shape[0]
    mask = make_mask(t["SC"], N, "sc")
    C0 = initial_ec(t["SC"], N, mask, "sc", 0.2) * 2.5
    r = fit_linear_gradient(t["emp"]["FC"], t["emp"]["COVtau"], C0, mask, t["omega"], t["tr"], 1, -0.02, 0.02, t["band"],
                            fit_a="global", a_bounds=(-0.03, -0.01), max_iter=50)
    assert "n_a_at_lower" in r.metrics and r.bound_hits["n_a_at_lower"] + r.bound_hits["n_a_at_upper"] >= 0
    # truth a = -0.05 lies below the lower bound -> the fitted a should sit on it
    assert r.bound_hits["n_a_at_lower"] == 1


def _a_near_limit(SC, G, omega, nodes, frac, base=-0.05):
    """a_j = base except on `nodes`, where a_j = t * (largest t keeping the linearisation stable), by bisection."""
    from hopfec.models.hopf_linear import linear_stability

    def a_of(t):
        a = np.full(SC.shape[0], base)
        a[nodes] = t
        return a

    lo, hi = 0.0, 5.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if linear_stability(SC, G, a_of(mid), omega)["stable"] else (lo, mid)
    return a_of(frac * lo), lo


def test_gradient_fit_stability_guard(truth):
    """a_j profile with supercritical nodes: the fit must not crash (unstable steps are rejected) and must refuse
    an unstable starting point with a clear message."""
    from hopfec.models.linear_gradient import fit_linear_gradient
    from hopfec.models.hopf_linear import linear_stability

    t = truth
    N = t["SC"].shape[0]
    C0 = t["G_true"] * t["SC"]
    top = np.argsort(C0.sum(axis=1))[-2:]
    a, t_lim = _a_near_limit(t["SC"], t["G_true"], t["omega"], top, 0.9)
    assert t_lim > 0 and a[top[0]] > 0 and linear_stability(t["SC"], t["G_true"], a, t["omega"])["stable"]
    mask = make_mask(t["SC"], N, "sc")
    r = fit_linear_gradient(t["emp"]["FC"], t["emp"]["COVtau"], C0, mask, t["omega"], t["tr"], 1, a, 0.02, t["band"], max_iter=60)
    assert np.isfinite(r.loss) and "n_unstable_evaluations" in r.metrics
    assert linear_stability(r.C, 1.0, a, t["omega"])["stable"]  # the accepted solution is always stable
    a_bad, _ = _a_near_limit(t["SC"], t["G_true"], t["omega"], top, 1.5)
    with pytest.raises(ValueError, match="unstable at the starting point"):
        fit_linear_gradient(t["emp"]["FC"], t["emp"]["COVtau"], C0, mask, t["omega"], t["tr"], 1, a_bad, 0.02, t["band"], max_iter=5)


def test_gradient_fit_penalty_path(truth):
    """Start so close to the validity limit that L-BFGS-B steps cross it: the penalty branch must fire and the
    returned solution must still be stable."""
    from hopfec.models.linear_gradient import fit_linear_gradient
    from hopfec.models.hopf_linear import linear_stability

    t = truth
    N = t["SC"].shape[0]
    C0 = t["G_true"] * t["SC"]
    a, _ = _a_near_limit(t["SC"], t["G_true"], t["omega"], np.arange(N), 1.0 - 1e-6)
    assert linear_stability(t["SC"], t["G_true"], a, t["omega"])["stable"]
    mask = make_mask(t["SC"], N, "sc")
    r = fit_linear_gradient(t["emp"]["FC"], t["emp"]["COVtau"], C0, mask, t["omega"], t["tr"], 1, a, 0.02, t["band"], max_iter=30)
    assert np.isfinite(r.loss) and r.metrics["n_unstable_evaluations"] > 0
    assert linear_stability(r.C, 1.0, a, t["omega"])["stable"]


def test_validated_used_falls_back_when_unstable(truth):
    from hopfec.models.search import _validated_used

    t = truth
    info = {"warnings": []}
    ok = {"G": 1.0, "a": -0.05, "source": "parabolic_interpolation"}
    assert _validated_used("linear", ok, {"G": 0.9, "a": -0.05}, t["SC"], t["omega"], None, info) is ok and not info["warnings"]
    bad = {"G": 1.0, "a": 0.5, "source": "parabolic_interpolation"}
    used = _validated_used("linear", bad, {"G": 0.9, "a": -0.05}, t["SC"], t["omega"], None, info)
    assert used["G"] == 0.9 and used["a"] == -0.05 and "unstable" in used["source"] and len(info["warnings"]) == 1
    # heterogeneity: the axis value is beta, a_fn maps it to the node vector
    z = np.linspace(-1, 1, t["SC"].shape[0])
    a_fn = lambda b: -0.1 + b * z  # noqa: E731
    assert _validated_used("linear", {"G": 1.0, "a": 0.02, "source": "continuous (lbfgs)"}, {"G": 1.0, "a": 0.0}, t["SC"], t["omega"], a_fn, info)["a"] == 0.02
    assert _validated_used("nonlinear", bad, {"G": 0.9, "a": -0.05}, t["SC"], t["omega"], None, info) is bad


def test_subcritical_validity(truth):
    from hopfec.models.hopf_linear import linear_stability
    from hopfec.models.search import linear_validity

    t = truth
    # small positive a at moderate G: the coupled Jacobian is stable, but the point is outside the linear model's regime
    a_pos = np.full(t["SC"].shape[0], -0.05)
    a_pos[0] = 0.01
    assert linear_stability(t["SC"], 1.0, a_pos, t["omega"])["stable"]
    ok, reason = linear_validity(t["SC"], 1.0, a_pos, t["omega"])
    assert not ok and "supercritical" in reason
    assert linear_validity(t["SC"], 1.0, a_pos, t["omega"], require_subcritical=False)[0]
    assert linear_validity(t["SC"], 1.0, -0.05, t["omega"])[0]
    df = evaluate_points("linear", [(1.0, -0.05), (1.0, 0.01)], t["SC"], t["omega"], t["emp"], t["tr"], t["band"])
    assert bool(df.loc[0, "valid"]) and not bool(df.loc[1, "valid"]) and "supercritical" in df.loc[1, "reason"]
    df2 = evaluate_points("linear", [(1.0, 0.01)], t["SC"], t["omega"], t["emp"], t["tr"], t["band"], require_subcritical=False,
                          a_fn=lambda v, a_pos=a_pos: a_pos)   # heterogeneous profile: one supercritical node, stable Jacobian
    assert bool(df2.loc[0, "valid"]) and np.isfinite(df2.loc[0, "fit_rmse"])
