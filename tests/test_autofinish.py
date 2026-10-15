"""Unit tests of sirilpipe.autofinish: the rules as functions, and a full run on a synthetic session."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_finish import synthetic_linear                      # noqa: E402

from sirilpipe import autofinish as af, finish, fitsio, post   # noqa: E402


def test_sky_aim_is_dark_grey_and_the_nebula_aim_is_a_midtone():
    assert 0.07 <= af.SKY_AIM <= 0.10 and 0.45 <= finish.NEBULA_AIM <= 0.65 and finish.BRIGHT_AIM > finish.NEBULA_AIM


def test_star_settings_by_density():
    assert af.star_settings(50)[2] == "sparse" and af.star_settings(300)[2] == "normal" and af.star_settings(2000)[2] == "dense"
    assert af.star_settings(50)[0] < af.star_settings(300)[0] < af.star_settings(2000)[0]
    assert af.star_settings(5000)[0] <= 0.40                                       # a dense field keeps its faint stars
    assert af.star_settings(2000)[1] >= 0.9


def test_gradient_residual_flags_a_tilted_sky():
    flat = np.full((200, 200), 0.02, np.float32) + np.random.default_rng(1).normal(0, 0.0005, (200, 200)).astype(np.float32)
    tilt = flat + np.linspace(0, 0.01, 200, dtype=np.float32)[None, :]
    slope = 8.0                                                   # display units per linear unit at the sky (what the GHS fit gives)
    g0, g1 = af.gradient_residual(flat, slope), af.gradient_residual(tilt, slope)
    assert g1["spread_display"] > g0["spread_display"] * 3
    assert g1["spread_display"] > af.GRADIENT_LIMIT > g0["spread_display"]


def test_stretch_slope_is_the_ghs_slope_at_the_sky():
    fit = finish.ghs_fit(0.06, 0.25, 0.085, 0.55, 4.0)
    info = {"ghs": fit, "white": 1e-3, "sky_in": 0.06}
    assert af.stretch_slope(info) == pytest.approx((0.55 - 0.085) / (0.25 - 0.06) / 1e-3, rel=1.5)     # same order as the secant slope
    assert af.stretch_slope({}) == 0.0


def test_score_prefers_neutral_unclipped_sky_at_target():
    good = {"sky_L": 0.16, "sky_C": 0.0, "clip_frac": 0.0}
    bad = {"sky_L": 0.3, "sky_C": 0.05, "clip_frac": 0.05}
    assert af.score(good, 0.16) == 1.0 and af.score(bad, 0.16) < 0.3


@pytest.fixture()
def session(tmp_path, monkeypatch):
    d = tmp_path / "s"
    d.mkdir()
    lin = synthetic_linear(h=500, w=700, nstars=120, blob=0.05)
    hdr = {"OBJECT": "NGC 1", "ROWORDER": "BOTTOM-UP"}
    fitsio.write_rgb(str(d / "stack.fits"), lin, hdr)
    starless = np.minimum(lin, np.float32(0.04))
    fitsio.write_rgb(str(d / "linear.fits"), lin, hdr)
    fitsio.write_rgb(str(d / "starless.fits"), starless, hdr)
    fitsio.write_rgb(str(d / "stars.fits"), np.maximum(lin - starless, 0), hdr)
    json.dump({"tool": "starnet2", "stage": "linear", "stars": {"count": 900, "per_megapixel": 450.0}},
              open(d / "starnet_meta.json", "w"))
    post._PREVIEW_CACHE.clear()
    return str(d)


def test_run_writes_report_with_a_measurement_per_decision(session):
    out = af.run(session, {"object": "NGC 1", "filter": "IRCUT"}, {"stretch": 0.3}, None, None)
    assert os.path.exists(os.path.join(session, "autofinish.json"))
    assert json.load(open(os.path.join(session, "autofinish.json")))["params"] == out["params"]
    p, rep = out["params"], out["report"]
    assert set(p) >= set(post.DEFAULTS)                                       # full parameter set for the sliders
    assert p["stretch"] == af.SKY_AIM and p["auto_stretch"] is False and p["black_point"] == 0.0
    assert p["saturation"] >= 1.0                                              # never desaturates
    assert p["star_reduction"] == pytest.approx(0.30) and p["star_intensity"] == pytest.approx(1.0)   # 450 stars / MP: normal
    assert p["hdr"] == 0.0 and p["contrast"] in finish.GHS_B_LADDER
    assert p["grade_temperature"] == 0 and p["grade_contrast"] == 0           # grade stays neutral
    steps = {s["step"]: s for s in rep["steps"]}
    assert {"stretch", "auto_stretch", "star_reduction", "star_intensity", "hdr", "contrast", "saturation"} <= set(steps)
    for s in rep["steps"]:
        assert s["why"] and s["measured"] is not None and "to" in s and "from" in s
    assert steps["stretch"]["from"] == 0.3 and steps["stretch"]["to"] == p["stretch"]
    assert 0 <= rep["score_before"] <= 1 and 0 <= rep["score_after"] <= 1
    assert "gradient" in rep and set(rep["gradient"]) >= {"spread_display", "note"}
    assert rep["class"].startswith("broadband")


def test_dual_band_run_sets_the_oiii_boost(session):
    out = af.run(session, {"object": "NGC 1", "filter": "Duo-Band", "is_dualband": True}, {}, None, None)
    assert out["report"]["class"].startswith("dual-band")
    assert af.OIII_BOOST_RANGE[0] <= out["params"]["oiii_boost"] <= af.OIII_BOOST_RANGE[1]
    assert any(s["step"] == "oiii_boost" for s in out["report"]["steps"]) or out["params"]["oiii_boost"] == 1.0


def test_progress_and_cancel_are_honoured(session):
    seen = []

    class Stop(Exception):
        pass

    def cp():
        if len(seen) >= 2:
            raise Stop()

    with pytest.raises(Stop):
        af.run(session, {}, {}, lambda i, n, m: seen.append(m), cp)
    assert seen and seen[0].startswith("Auto-finish")


def test_dense_star_field_gets_more_reduction(session):
    json.dump({"tool": "starnet2", "stage": "linear", "stars": {"count": 9000, "per_megapixel": 2500.0}},
              open(os.path.join(session, "starnet_meta.json"), "w"))
    out = af.run(session, {}, {}, None, None)
    assert out["params"]["star_reduction"] > 0.30 and out["params"]["star_intensity"] >= 0.9
