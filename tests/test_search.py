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
    assert bool(df.loc[2, "valid"]) is False and np.isnan(df.loc[2, "fit_rmse"]) and "unstable" in df.loc[2, "reason"]
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
