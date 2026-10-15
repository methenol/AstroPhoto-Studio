"""Tests of the professional finishing chain: published-formula identities, the NBN port against the script's own function, the GHS fit, the gates and autofinish's rules."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from sirilpipe import autofinish as af, finish, fitsio, gates, nbn, pro, siril

RNG = np.random.default_rng(3)


def test_star_stretch_is_the_midtones_transfer_function():
    """Seti Astro Star Stretch 3^a x / ((3^a - 1) x + 1) == Siril ``mtf 0 m 1`` with m = 1 / (3^a + 1) (a = 5: 0.004098)."""
    x = np.linspace(0, 1, 2001)
    for a in (0.0, 1.0, 3.0, 4.0, 5.0, 6.0, 8.0):
        f = 3.0 ** a * x / ((3.0 ** a - 1.0) * x + 1.0)
        assert np.abs(pro.mtf_balance(pro.star_stretch_m(a), x) - f).max() < 1e-9
    assert round(pro.star_stretch_m(5.0), 6) == 0.004098
    assert round(pro.star_stretch_m(4.0), 6) == 0.012195


@pytest.mark.docker
def test_star_stretch_mtf_in_siril_matches_the_formula(tmp_path):
    x = RNG.uniform(0, 0.3, (3, 200, 300)).astype(np.float32)
    fitsio.write_rgb(str(tmp_path / "a.fit"), x, {})
    m = pro.star_stretch_m(5.0)
    siril.SirilRunner(str(tmp_path)).run(f"load a\nmtf 0 {m:.6f} 1\nsave b\n", "mtf", expect=["b.fit"])
    got = fitsio.read_rgb(str(tmp_path / "b.fit"))[0]
    assert np.abs(got - 3.0 ** 5 * x / (242.0 * x + 1.0)).max() < 1e-4


def test_ghs_two_pass_fit_puts_the_background_on_the_target():
    h = {"mode": 0.0189, "left": 0.01888, "toe": 0.01883, "sigma": 2e-5, "sky": 0.01891}
    for target in (0.08, 0.10, 0.12):
        fit = pro.fit_ghs2(h, 0.01905, 0.0195, b1=10.0, b2=4.0, d2=1.0, target=target, use_hp=False)
        assert fit["sky2"] == pytest.approx(target, abs=1e-4)
        assert 0 < fit["d1"] <= 10
    hp = pro.fit_ghs2(h, 0.01905, 0.0195, b1=10.0, b2=4.0, d2=1.0, target=0.10, use_hp=True)
    assert hp["hp2"] < 1.0 and hp["sky2"] == pytest.approx(0.10, abs=1e-4)


def test_hist_stats_landmarks_of_a_noisy_sky():
    lum = (0.02 + RNG.normal(0, 1e-4, (400, 600))).astype(np.float32)
    lum[100:200, 100:300] += 5e-4
    h = pro._hist_stats(lum)
    assert abs(h["mode"] - 0.02) < 1e-4 and h["left"] < h["mode"] and h["toe"] < h["left"]


@pytest.mark.docker
def test_nbn_port_matches_the_scripts_own_function(tmp_path):
    """``nbn.normalise`` against ``process_image`` of NarrowbandNormalization.py run through community_worker.py (tolerance 1e-4)."""
    py, script = pro._community_python(), pro._community_script("processing/NarrowbandNormalization.py")
    if not py or not script:
        pytest.skip("the pyscript virtualenv / siril-scripts are not installed")
    yy, xx = np.mgrid[0:120, 0:160].astype(np.float32)
    neb = np.exp(-(((xx - 80) / 40) ** 2 + ((yy - 60) / 30) ** 2))
    ha = np.clip(0.12 + 0.5 * neb + RNG.normal(0, 0.01, xx.shape), 0, 1)
    oiii = np.clip(0.10 + 0.2 * neb + RNG.normal(0, 0.01, xx.shape), 0, 1)
    data = np.stack([ha, 0.5 * (ha + oiii), oiii], -1).astype(np.float32)
    np.save(tmp_path / "in.npy", data)
    for boost in (1.0, 1.7):
        prm = {"palette": "HOO", "lightness": "Off", "blend_mode": "Mode 1", "blend_amount": 0.6, "scnr": 0, "oiii_boost": boost, "sii_boost": 1, "shadow_point": 1,
               "highlight_reduction": 1, "brightness": 1}
        worker = os.path.join(os.path.dirname(pro.__file__), "community_worker.py")
        r = subprocess.run([py, worker, "--script", script, "--entry", "process_image", "--input", str(tmp_path / "in.npy"), "--output-prefix", str(tmp_path / "out"),
                            "--args", json.dumps([prm])], capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-500:]
        ref = np.load(tmp_path / "out_0.npy")
        mine = nbn.normalise(data, oiii_boost=boost)
        assert np.abs(mine - ref).max() <= 1e-4


def test_nbn_defaults_keep_ha_and_make_a_synthetic_green():
    d = np.stack([np.full((20, 20), 0.4), np.zeros((20, 20)), np.full((20, 20), 0.2)], -1).astype(np.float32)
    d[0, 0] = (0.1, 0, 0.1)
    out = nbn.normalise(d)
    assert np.allclose(out[..., 0], np.clip(d[..., 0], 0, 1), atol=1e-5)
    assert out[5, 5, 1] > 0


def _final(h=300, w=400, sky=0.10):
    img = np.full((3, h, w), sky, np.float32) + RNG.normal(0, 0.001, (3, h, w)).astype(np.float32)
    return np.clip(img, 0, 1)


def test_gates_pass_on_a_clean_image_and_flag_each_failure():
    g = gates.compute(_final())
    assert g["sky_median"]["pass"] and g["clipped"]["pass"] and g["white"]["pass"] and g["sky_chroma"]["pass"] and g["sky_band_structure"]["pass"]
    assert gates.compute(_final(sky=0.25))["sky_median"]["pass"] is False
    img = _final()
    img[:, :40, :40] = 1.0
    assert gates.compute(img)["clipped"]["pass"] is False and gates.compute(img)["white"]["pass"] is False
    tint = _final()
    tint[0] += 0.05
    assert gates.compute(tint)["sky_chroma"]["pass"] is False
    noisy = _final() + RNG.normal(0, 0.02, (3, 300, 400)).astype(np.float32)
    assert gates.compute(np.clip(noisy, 0, 1))["sky_pixel_noise"]["pass"] is False
    assert "gates:" in gates.format_gates(g)


def test_gates_count_round_coloured_stars():
    base = _final(h=400, w=600)
    yy, xx = np.mgrid[0:400, 0:600].astype(np.float32)
    stars = np.zeros((3, 400, 600), np.float32)
    n = 0
    for y in range(30, 380, 50):
        for x in range(30, 580, 60):
            g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.5 ** 2))
            for c, k in enumerate((0.8, 0.5, 0.3)):
                stars[c] += 0.7 * k * g
            n += 1
    final = 1 - (1 - base) * (1 - stars)
    g = gates.compute(final, starless=base, linear_star_count=n, star_level=0.05)
    assert g["stars_count"]["pass"] and g["stars_round"]["pass"] and g["stars_coloured"]["pass"]


def test_autofinish_rules_use_published_values_only():
    assert af.star_settings(50) == (0.4, "sparse") and af.star_settings(300)[0] == 0.5 and af.star_settings(2000)[0] == 0.6
    assert af.hdr_rule(0.0006, 0.0) and af.hdr_rule(0.0, 0.0006) and not af.hdr_rule(0.0004, 0.0004)
    assert af.nxt_strength(0.0) == 0.85 and af.nxt_strength(100.0) == 0.6 and 0.6 <= af.nxt_strength(6.0) <= 0.85
    assert af.local_contrast_amount(0.0) == 0.5 and af.local_contrast_amount(100.0) == pytest.approx(0.875, abs=0.006)
    assert af.satu_amount(0.5) == 0.25 and af.satu_amount(0.25) == 0.3 and af.satu_amount(0.1) == 0.5
    assert af.star_boost(0.2) == 1.0 and af.star_boost(0.08) == 1.5 and af.star_boost(0.01) == 2.0
    img = np.zeros((3, 50, 50), np.float32)
    img[1] = 0.3
    assert af.green_cast(img, None) > af.SCNR_CAST


def test_ghs_scalar_is_the_numpy_mirror_of_ght():
    assert pro.ghs_scalar(0.5, 5.0, 0.0, 0.5) == pytest.approx(0.5, abs=1e-6)       # D = 0: identity
    assert pro.ghs_scalar(0.019, 10.0, 3.0, 0.0189) == pytest.approx(float(finish.ghs(np.array([0.019]), 10.0, np.expm1(3.0), 0.0189)[0]), abs=1e-9)


@pytest.mark.docker
def test_the_chain_runs_in_siril_and_records_every_step(tmp_path):
    """The default chain on a synthetic dual-band stack: a picture after every step, the verbatim commands, the background on its target."""
    if not os.path.exists(pro.config.siril_bin()) and not __import__("shutil").which("siril-cli"):
        pytest.skip("no siril-cli")
    h, w = 400, 600
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    neb = np.exp(-(((xx - 300) / 90) ** 2 + ((yy - 200) / 60) ** 2))
    sl = np.stack([0.0189 + 6e-4 * neb, 0.0189 + 2e-4 * neb, 0.0189 + 2e-4 * neb]).astype(np.float32) + RNG.normal(0, 2e-5, (3, h, w)).astype(np.float32)
    st = np.zeros((3, h, w), np.float32)
    for y, x in ((100, 100), (300, 450), (150, 520)):
        g = 0.05 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.6 ** 2))
        st += g[None] * np.array([1.0, 0.8, 0.7], np.float32)[:, None, None]
    p = {"stretch_method": "ghs2", "stretch": 0.10, "contrast": 4.0, "hdr": 0.0, "saturation": 1.3, "local_contrast": 0.5, "star_reduction": 0.4, "star_intensity": 1.0,
         "star_saturation": 1.0, "oiii_boost": 1.0, "palette": "auto", "scnr": 0.0, "black_point": 0.0, "brightness": 0.0}
    fin = pro.Finisher(str(tmp_path), {"ROWORDER": "BOTTOM-UP"}, p, True, external=False)
    out, info = fin.run(np.clip(sl, 0, 1), st)
    names = [r["step"] for r in info["chain"]]
    assert names[0].startswith("starless") and "stretch pass 1 (GHS)" in names and "palette (hoo)" in names and "recombine + star reduction" in names
    assert all(r.get("picture") and os.path.exists(r["picture"]) for r in info["chain"] if r["status"] == "done")
    cmds = " ".join(c for r in info["chain"] for c in r["commands"])
    assert "ght -D=" in cmds and "linstretch -BP=" in cmds and "mtf 0 0.004098 1" in cmds and 'pm "~(~$rc_sl$*~$rc_st$)"' in cmds and "clahe 1.5" in cmds
    g = gates.compute(out, starless=info["starless"])
    assert g["sky_median"]["pass"] and g["clipped"]["pass"]
