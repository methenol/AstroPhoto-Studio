"""Tests of the professional finishing chain: published-formula identities, the NBN port against the script's own function, the GHS fit, the gates and autofinish's rules."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from sirilpipe import autofinish as af, finish, fitsio, gates, nbn, post, pro, siril

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
    from sirilpipe import stretch
    if not stretch.siril_available():
        pytest.skip("no siril-cli")
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
    assert af.saturation_factor(0.5, True) == (1.0, 0.45) and af.saturation_factor(0.1, True) == (4.0, 0.45) and af.saturation_factor(0.2, False) == (1.25, 0.25)
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


def test_the_chosen_palette_decides_and_the_oiii_boost_only_tunes_nbn(tmp_path, monkeypatch):
    """With an OIII boost set, every palette of the menu rendered as NBN (boost != 1 switched NBN on by itself): the preview stuck in NBN."""
    calls = []
    monkeypatch.setattr(pro.Finisher, "_palette_nbn", lambda self, arr, boost: calls.append(("nbn", boost)) or arr)
    monkeypatch.setattr(pro.Finisher, "_palette_hubble", lambda self, arr, pal: calls.append(("hubble", pal)) or arr)
    monkeypatch.setattr(pro.Finisher, "_palette_pm", lambda self, arr, pal: calls.append(("pm", pal)) or arr)
    monkeypatch.setattr(pro.Finisher, "ha_luminance", lambda self, out, ha: out)
    arr = RNG.uniform(0.05, 0.3, (3, 32, 32)).astype(np.float32)
    for pal, boost, want in (("hoo", 2.0, ("hubble", "hoo")), ("sho", 1.5, ("hubble", "sho")), ("foraxx", 2.0, ("pm", "foraxx")), ("nbn", 2.0, ("nbn", 2.0)), ("nbn", 1.0, ("nbn", 1.0))):
        calls.clear()
        fin = pro.Finisher(str(tmp_path), {}, {**post.DEFAULTS, "palette": pal, "oiii_boost": boost}, True, external=False)
        fin.colour(arr)
        assert calls == [want], (pal, boost, calls)


def _pedestal_case(gradient=0.0, rim=0.0, rim_px=60, wing=0.0, falloff=0):
    """A linear starless of 0.05 sky: an optional smooth ``gradient`` across the frame (a real flattened frame has one), an optional ``wing`` of nebulosity rising from the left
    edge, an optional ``rim`` pedestal on the degraded class, and a coverage map that calls that strip degraded - with an optional ``falloff`` px of gradual coverage ramp
    behind it, which is what a dithered mosaic margin leaves behind: pixels that are neither the degraded class nor the reference.  Returns what ``pro.sky_pedestal`` takes."""
    h, w = 600, 900
    rng = np.random.default_rng(11)
    x = np.arange(w)[None, :]
    base = np.stack([0.05 * (1.0 + gradient * x / w)] * 3) + rng.normal(0.0, 2e-4, (3, h, w))
    if wing:
        base += 0.05 * wing * np.clip(1.0 - x / 300.0, 0.0, 1.0)     # a nebula's wing rising from the left edge: ``wing`` of the sky there, gone by x = 300
    cov = np.ones((h, w))
    cov[:, :rim_px] = 0.35
    if falloff:
        cov[:, rim_px:rim_px + falloff] = np.linspace(0.35, 1.0, falloff)[None, :]
    if rim:
        base[:, cov < 0.7] += rim                                     # the pedestal belongs to the degraded class, wherever the coverage map puts it
    lum = finish.luminance(base.astype(np.float32))
    sm = finish._sky_mask(lum, edge=finish.EDGE_CROP)
    ref = max(float(np.median(base[c][sm])) for c in range(3))
    return base, ref, cov


def test_the_pedestal_levels_a_real_pedestal_on_the_degraded_rim():
    """The case the pedestal exists for (IC 1848): the low-coverage dithered rim carries a pedestal; taking its shape out must bring the rim down to the sky."""
    base, ref, cov = _pedestal_case(rim=0.00075)                          # 1.5 % of the sky, under the 2 % cap
    ped, info = pro.sky_pedestal(base, ref, cov)
    assert ped is not None, info
    before = float(np.median(base[0][:, :40])) - float(np.median(base[0][:, 200:600]))
    after = float(np.median((base[0] - ped[0])[:, :40])) - float(np.median((base[0] - ped[0])[:, 200:600]))
    assert before > 0.0004, before                                            # the pedestal is really there
    assert abs(after) < 0.35 * abs(before), (before, after, info)


def test_the_pedestal_takes_nothing_from_a_clean_frame_whose_rim_is_merely_low_coverage():
    """A frame that is already flat must come back near-untouched even where coverage is low.  The first version measured the degraded cells and the reference cells as two
    meshes interpolated separately and subtracted the fields; the degraded class holds few cells, so its field was mostly guesswork and the difference carried structure of
    the order of the cap over flat sky - on M 31 (2026-10-30) that turned a left rim sitting 0.9 % of the sky below the reference into a 24 % band after the stretch, whose
    slope at the sky is ~120.  Measured on this frame: the two-field version took 0.22 % of the sky, the plane reference now takes 0.06 %, and the estimator's own noise
    floor is 0.05 % (``SKY_PEDESTAL_EPS``) - the bound sits between the regression and the floor."""
    base, ref, cov = _pedestal_case(gradient=0.02)
    ped, info = pro.sky_pedestal(base, ref, cov)
    assert ped is None or float(np.abs(ped).max()) <= 0.001 * ref, info


def test_the_pedestal_takes_nothing_from_a_degraded_region_under_a_bright_nebula_wing():
    """The phantom that painted IC 1848's export with a bright red band down its left edge (2026-10-30).  The degraded region there lies *under* the nebula complex, and every
    reference but the region's own border reads the wing's rise as a pedestal: the version that fitted a plane through the well-covered cells took 0.86 % of the sky from that
    edge while the two classes' own medians agreed to 0.1 %, and a bare comparison of the sky levels either side of the border leaks 0.43 % of the sky from a wing this steep,
    because a wing is a slope and a pedestal is a jump.  A region whose sky is level with the pixels across its border stays exactly as the frames left it."""
    base, ref, cov = _pedestal_case(wing=0.08)
    ped, info = pro.sky_pedestal(base, ref, cov)
    assert ped is None or float(np.abs(ped).max()) <= 0.001 * ref, info


def test_a_degraded_margin_whose_reference_class_never_reaches_it_is_still_levelled():
    """NGC 1499 (2026-10-30): a bright band across the top and the bottom of the export while the starless was flat to the eye.  A dithered mosaic margin's coverage falls off
    gradually, so between the degraded class and the well-covered one sits a band of pixels that is neither, and the region shares no border with the reference at all - on that
    frame its two margins lay 20.2 and 17.5 px from the nearest reference pixel with ``SKY_PEDESTAL_BAND`` at 16, every shell came back empty, the verdict was "0 inside, 0
    outside", and the +0.4 % of the sky the margins carried came out of the stretch's ~120 slope at the sky as the band.  The frame outside the region's own class shares a real
    border with it, and the jump across that border is the same pedestal."""
    base, ref, cov = _pedestal_case(rim=0.00075, rim_px=90, falloff=60)
    ped, info = pro.sky_pedestal(base, ref, cov)
    assert ped is not None, info
    before = float(np.median(base[0][:, :40])) - float(np.median(base[0][:, 300:600]))
    after = float(np.median((base[0] - ped[0])[:, :40])) - float(np.median((base[0] - ped[0])[:, 300:600]))
    assert before > 0.0004 and abs(after) < 0.35 * abs(before), (before, after, info)
    assert any("outside its class" in str(v[1]) for v in info["verdicts"]), info                # the frame records that it had to fall back


def test_a_region_that_reaches_the_well_covered_frame_is_measured_against_that_and_nothing_else():
    """The fallback is only for a region that cannot reach the reference.  Where the classes touch, the reference stays the pixels every one of the frames built."""
    base, ref, cov = _pedestal_case(rim=0.00075)
    ped, info = pro.sky_pedestal(base, ref, cov)
    assert ped is not None, info
    assert all("well-covered" in str(v[1]) for v in info["verdicts"]), info


def test_the_pedestal_levels_a_real_pedestal_under_a_bright_nebula_wing():
    """The same hard setting with a pedestal that is really there: refusing the wing's slope must not cost the tool its edge, or the low-coverage rim of a frame under a
    nebula - IC 1848 again - keeps the band the stretch turns out of it."""
    base, ref, cov = _pedestal_case(wing=0.08, rim=0.00075)
    ped, info = pro.sky_pedestal(base, ref, cov)
    assert ped is not None, info
    # the step across the border, measured on equal bands either side of it so that the wing's own slope cancels: the jump is what has to go, the wing is what has to stay
    lo, hi = slice(50, 60), slice(60, 70)
    before = float(np.median(base[0][:, lo]) - np.median(base[0][:, hi]))
    after = float(np.median((base[0] - ped[0])[:, lo]) - np.median((base[0] - ped[0])[:, hi]))
    assert before > 0.0004, before
    assert abs(after) < 0.35 * abs(before), (before, after, info)
    assert float(np.abs(ped[0][:, 120:]).max()) <= 0.01 * float(np.abs(ped[0]).max()), info   # nothing beyond the degraded class is touched: that is the nebula's protection


def test_the_pedestal_leaves_the_structure_inside_a_degraded_region_alone():
    """What measuring only at the region's border, never inside it, is there for (IC 1848, 2026-10-30): a low-coverage area is not empty of sky structure, and a field fitted
    over its cells happily models that structure as a pedestal - it came back of both signs and +-2.2 % of the sky over 9.3 % of the frame while the region's sky was level with
    the reference's, and the stretch turned it into the rectangular patches the export showed.  A pedestal is one offset shared by the pixels of one origin, so structure that
    merely lies inside the region has to survive the tool that levels the region."""
    h, w = 600, 900
    rng = np.random.default_rng(7)
    base = np.stack([np.full((h, w), 0.05)] * 3).astype(np.float64) + rng.normal(0.0, 2e-4, (3, h, w))
    cov = np.ones((h, w))
    r0, r1, c0, c1 = 150, 450, 200, 700                                                    # the degraded region, well inside the frame
    cov[r0:r1, c0:c1] = 0.35
    half = (c0 + c1) // 2
    base[:, r0:r1, c0:half] += 0.001                                                       # +2 % of the sky of nebulosity inside it
    base[:, r0:r1, half:c1] -= 0.001                                                       # ... and its mirror, so the region carries no offset at all
    lum = finish.luminance(base.astype(np.float32))
    sm = finish._sky_mask(lum, edge=finish.EDGE_CROP)
    ref = max(float(np.median(base[c][sm])) for c in range(3))
    ped, info = pro.sky_pedestal(base, ref, cov)
    before = float(np.median(base[0, r0:r1, c0:half]) - np.median(base[0, r0:r1, half:c1]))
    assert before > 0.0015, before                                                         # the structure is really there
    if ped is not None:
        after = float(np.median((base[0] - ped[0])[r0:r1, c0:half]) - np.median((base[0] - ped[0])[r0:r1, half:c1]))
        assert abs(after - before) < 0.15 * before, (before, after, info)                 # the per-cell field took twice the whole contrast away


def _edge_case(step=0.0, depth=8, residue=0.15, sides=(0, 1), h=400, w=600, sky=0.05, degraded=None):
    """A linear starless of ``sky`` carrying ``step`` percent of the sky over the outer ``depth`` px of ``sides``, with ``residue`` of it carried one band deeper, the shape
    the restore stage leaves at the frame's own boundary (M 31, 2026-10-30: the outer 8 px sat -2.2 % of the sky under the interior and the next 56 px only -0.4 %).
    ``degraded`` marks those columns/rows as low coverage in a coverage map."""
    rng = np.random.default_rng(4)
    a = np.stack([np.full((h, w), sky)] * 3) + rng.normal(0.0, 1.2e-4, (3, h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    cov = np.ones((h, w))
    for side in sides:
        d = (yy if side == 0 else h - 1 - yy) if side < 2 else (xx if side == 2 else w - 1 - xx)
        profile = 1.0 * (d < depth) + residue * ((d >= depth) & (d < 2 * depth))
        for c in range(3):
            a[c] += step / 100.0 * sky * profile * (0.9 + 0.1 * c)
        if degraded:
            cov[{0: (slice(0, depth), slice(None)), 1: (slice(h - depth, h), slice(None)), 2: (slice(None), slice(0, depth)), 3: (slice(None), slice(w - depth, w))}[side]] = 0.3
    return a, cov


def _border(a, side, px=8):
    """Each channel's outermost ``px`` rows/columns as an offset from the middle of the frame."""
    h, w = a.shape[1], a.shape[2]
    sl = {0: (slice(0, px), slice(None)), 1: (slice(h - px, h), slice(None)), 2: (slice(None), slice(0, px)), 3: (slice(None), slice(w - px, w))}[side]
    m = min(8 * px, min(h, w) // 4)                                                  # the middle of the frame, kept a quarter of the frame away whatever `px` asks for
    inner = (slice(m, h - m), slice(m, w - m))
    return [float(np.median(a[c][sl]) - np.median(a[c][inner])) for c in range(a.shape[0])]


def test_the_ring_takes_the_step_the_restore_stage_leaves_at_the_border():
    """The case the tool exists for: a step of a few percent of the sky confined to the outer 8 px is invisible in the linear and a bright line down the edge of the export,
    because the stretch's slope at the sky is ~150.  Taking the measured profile out has to bring the border to the sky without touching the sky behind it."""
    a, cov = _edge_case(step=3.0)
    before = [max(abs(v) for v in _border(a, side)) for side in (0, 1)]
    assert before[0] > 0.0012, before                                                         # the step is really there
    field, info = pro.edge_ring(a, cov)
    assert field is not None, info
    corrected = a - field
    for side in (0, 1):
        assert max(abs(v) for v in _border(corrected, side)) < 0.001 * 0.05, (side, _border(corrected, side), info)
    assert max(abs(v) for v in _border(corrected, 2)) < 0.0002                                # the sides that carried nothing are left where they were


def test_the_ring_takes_the_frame_line_that_is_two_pixels_wide():
    """What measuring the ring in bands could not see (IC 1848, 2026-10-30): the artefact is +0.5 % of the sky over the outer 2 px - 2 px of an 8 px band, under the fraction
    a floor statistic can even notice - and 37.8 % of the display sky in the export, because the stretch's toe multiplies it.  Read line by line it is the largest deviation
    in the frame, and the correction is the profile itself, so the line goes and the sky behind it does not move."""
    a, cov = _edge_case(step=0.5, depth=2, residue=0.0)
    before = max(abs(v) for v in _border(a, 0, px=2))
    assert before > 0.00012, before                                                    # the line is really there: 0.5 % of the sky
    field, info = pro.edge_ring(a, cov)
    assert field is not None, info
    assert max(abs(v) for v in _border(a - field, 0, px=2)) < 0.001 * 0.05, (info, _border(a - field, 0, px=2))
    assert max(abs(v) for v in _border(a - field, 0, px=32)) < 0.001 * 0.05           # and nothing beyond the two lines was touched


def test_the_ring_leaves_a_deviation_that_continues_behind_the_border():
    """Confinement is the physical discriminator: nothing in the sky ends in the last pixels of a frame, so a deviation still standing ``EDGE_RING_DEPTH`` in is sky - or a
    gradient - and flattening it is what ``sky_pedestal`` established must never happen.  IC 1848's edges measure 0.1-0.4 % of the sky and reach past the ring: real faint
    nebulosity, and the tool has to leave every last bit of it."""
    a, cov = _edge_case(step=3.0, depth=24, residue=1.0)                                      # 3 % of the sky over the outer 48 px: no edge to it
    field, info = pro.edge_ring(a, cov)
    assert field is None, info


def test_the_ring_needs_the_frame_to_agree_on_a_small_step_but_not_on_a_large_one():
    """A deviation the size of faint sky is far more likely to be sky that ended near the edge than to be the frame's own boundary, so it needs a second side to agree.  A
    step of ``EDGE_RING_LONE`` of the sky or more is past anything sky leaves in its last pixels, and leaving it costs a visible line (IC 434, 2026-10-30: -3.8 % of the
    sky on the top border alone, which the first version refused to touch)."""
    small, cov = _edge_case(step=0.3, sides=(0,))
    assert pro.edge_ring(small, cov)[0] is None, pro.edge_ring(small, cov)[1]
    two, cov = _edge_case(step=0.3)
    assert pro.edge_ring(two, cov)[0] is not None, pro.edge_ring(two, cov)[1]
    large, cov = _edge_case(step=3.0, sides=(0,))
    assert pro.edge_ring(large, cov)[0] is not None, pro.edge_ring(large, cov)[1]


def test_the_ring_refuses_too_much_of_the_sky_rather_than_too_large_a_step():
    """Amplitude cannot tell a frame line from a nebula's edge - both can be 8 % of the sky - but the light they account for is not comparable: 8 % carried 8 px deep is a
    nebula's share of light, while 12.8 % in the outer line alone (IC 434's top border, measured) is unambiguously the frame's own.  So the bound is on the amount taken, and
    a line with no sky left in it at all (``EDGE_RING_MAX``) is missing data, which this tool must not paper over by inventing sky."""
    a, cov = _edge_case(step=8.0)                                                            # 8 % of the sky over the outer 8 px: 74 % of the sky along a line through the ring
    assert pro.edge_ring(a, cov)[0] is None, pro.edge_ring(a, cov)[1]
    thin, cov = _edge_case(step=12.0, depth=1, residue=0.0)                                  # IC 434's top border as measured
    field, info = pro.edge_ring(thin, cov)
    assert field is not None, info
    assert max(abs(v) for v in _border(thin - field, 0, px=1)) < 0.001 * 0.05, (info, _border(thin - field, 0, px=1))
    assert max(abs(v) for v in _border(thin - field, 0, px=8)) < 0.001 * 0.05                # the lines behind it were already level, and stay that way
    blank, cov = _edge_case(step=40.0, depth=1, residue=0.0)                                 # no sky left in that line at all: not this tool's to restore
    assert pro.edge_ring(blank, cov)[0] is None, pro.edge_ring(blank, cov)[1]


def test_the_ring_leaves_a_border_the_provenance_pedestal_has_already_levelled():
    """One owner per deviation: where the coverage map calls the border degraded and ``sky_pedestal`` has taken an offset there, that offset has a cause in the frames that
    contributed and the ring would only be guessing over what is left of it."""
    a, cov = _edge_case(step=2.0, degraded=True)                                             # both affected borders are also low coverage
    assert pro.edge_ring(a, cov)[0] is not None                                               # without the pedestal the ring is the only tool that can reach them
    ped = np.zeros_like(a)
    ped[:, :8, :] = 2.0 / 100.0 * 0.05                                                        # the pedestal's own field, over the degraded top border
    field, info = pro.edge_ring(a, cov, ped)
    assert field is not None, info
    assert max(abs(v) for v in _border(a - field, 1)) < 0.001 * 0.05, (info, _border(a - field, 1))   # the bottom is the ring's to take
    assert max(abs(v) for v in _border(a - field, 0)) > 0.0008, (info, _border(a - field, 0))          # the top is the pedestal's, and stands


def test_a_border_that_is_only_noisier_is_left_alone():
    """The low-coverage dither ring is both the noisiest and the least levelled part of a frame, and a sky *floor* reads low by a fixed multiple of sigma: IC 1848's rim carries
    1.6x the frame's noise, which is 0.2 % of the sky of offset the pixels do not have - 30 % of the display sky once the stretch's toe has multiplied it, in whichever
    direction the tool happens to be correcting.  Estimating the level with that bias taken out is what keeps the tool from pouring light into the rim it was meant to level,
    so a rim that differs from the interior only in its noise has to be left exactly where it is."""
    rng = np.random.default_rng(7)
    a = np.stack([np.full((400, 600), 0.05)] * 3) + rng.normal(0.0, 1.2e-4, (3, 400, 600))
    for c in range(3):
        a[c][:, :8] += rng.normal(0.0, 2.4e-4, (400, 8))                                     # the left and right rims: twice the noise, the same sky
        a[c][:, -8:] += rng.normal(0.0, 2.4e-4, (400, 8))
    field, info = pro.edge_ring(a)
    assert field is None, (info, [round(v, 6) for v in _border(a, 2, px=8)])


def _rim_case(step=2.0, depth=75, residue=0.0, chans=(1.0, 1.0, 1.0), filaments=0.0, fill=0.0, fill_level=300.0, sides=(0, 1), h=800, w=1200, sky=0.05, noise=1.2e-4, seed=4):
    """A linear starless of ``sky`` whose ``sides`` carry ``step`` percent of the sky across their outer ``depth`` px - the residue a gradient model leaves when it removed
    most of a border gradient but not all of it (NGC 1499: the sky's floor entered BGE -7.83 % below the frame's middle and left it -1.86 %) - with ``residue`` of it carried
    one band deeper, which is what a gradient the model never got hold of looks like.  ``chans`` scales it per channel, because the border is one object measured three
    times and a side whose channels disagree is not that object.  ``filaments`` (a fraction of the band) covers the border with bright ridges of real subject: they move the
    *median* of a border line a long way and its *sky floor* not at all, which is the difference this tool is built on.  ``fill`` (a fraction of the line's length) is the
    opposite case - a smooth plateau of ``fill_level`` percent of the sky lying across the border with no sky left under it, where a tool that measured a floor anyway would
    be inventing the border's level out of subject: bright enough to be believed, and the two cases - a wing the sky did not make, and a sky whose floor really did move -
    are what separates them.  ``noise`` is the sky's own scatter: it decides how tall a single line has to be before the witness stops reading that line as sky at all,
    which is the difference between a fluctuation of the border's step and a feature lying behind it."""
    rng = np.random.default_rng(seed)
    a = np.stack([np.full((h, w), sky)] * 3) + rng.normal(0.0, noise, (3, h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    for side in sides:
        d = (yy if side == 0 else h - 1 - yy) if side < 2 else (xx if side == 2 else w - 1 - xx)
        band = 1.0 * (d < depth) + residue * (d >= depth)                                             # the step the border itself made, on top of whatever the frame carries
        for c in range(3):
            a[c] += step / 100.0 * sky * chans[c] * band
        for k in range(int(filaments * 70)):                                                     # ridges lying along the border, with sky left between them
            cy = (k * 71) % max(1, depth)
            cx = (k * 211) % w
            a += (40.0 / 100.0 * sky) * np.exp(-(((yy - cy) / 7.0) ** 2 + ((xx - cx) / 11.0) ** 2))
        if fill:                                                                                 # no sky left on these lines at all: the witness is blind here, and blind is not level
            span = w if side < 2 else h
            along = xx if side < 2 else yy
            a += (fill_level / 100.0 * sky) * ((np.abs(along - span / 2.0) <= fill * span / 2.0) & (d < max(depth, 8)))
    return a


def _strip(a, side, d0, d1):
    """Each channel's band from d0 to d1 px in from that border, as an offset from the middle of the frame."""
    h, w = a.shape[1], a.shape[2]
    sl = {0: (slice(d0, d1), slice(None)), 1: (slice(h - d1, h - d0), slice(None)),
          2: (slice(None), slice(d0, d1)), 3: (slice(None), slice(w - d1, w - d0))}[side]
    m = min(8 * d1, min(h, w) // 4)
    inner = (slice(m, h - m), slice(m, w - m))
    return [float(np.median(a[c][sl]) - np.median(a[c][inner])) for c in range(a.shape[0])]


def test_the_rim_levels_the_band_the_gradient_model_left_at_the_border():
    """The case the tool exists for.  A border band sitting 2 % of the sky under the frame's own sky is nothing in the linear and a band across the export: the stretch's slope
    at the sky is 75-120, and NGC 1499's export measured its right edge at -51 % of the display sky for exactly that.  Taking the level out has to bring the border lines the
    export is read on to the frame's sky without disturbing the sky behind the band."""
    a = _rim_case(step=2.0, residue=0.15)
    band = max(64, min(512, int(round(min(a.shape[1:]) * pro.SKY_RIM_BAND))))
    before = [max(abs(v) for v in _border(a, side, px=2)) for side in (0, 1)]
    assert before[0] > 0.0008, before                                                        # 2 % of the sky over the band: really there
    field, info = pro.sky_rim(a)
    assert field is not None, info
    corrected = a - field
    for side in (0, 1):
        after = max(abs(v) for v in _border(corrected, side, px=2))
        assert after < 0.15 * before[0], (side, before, after, info)
        assert max(abs(v) for v in _border(corrected, side, px=2)) < 0.0002, (side, info)
        assert abs(_strip(corrected, side, band, 2 * band)[0] - _strip(a, side, band, 2 * band)[0]) < 1e-5, (side, info)   # the residue behind the band is left where it was
    assert max(abs(v) for v in _border(corrected, 2, px=2)) < 0.0002                          # the sides that carried nothing are untouched


def test_the_rim_leaves_a_border_as_level_as_the_frames_that_render_clean():
    """The population's floor, measured rather than aspired to.  Every real frame here sits 0.4-1.9 % below its own interior at the border on *all four sides* - that is the
    frame-scale gradient, and the strip behind the band carries it too, so it cancels.  What is left is the step the border actually made, and on the frames whose exports
    audit clean (M 42, M 45) that step is 0.00-0.02 % of the sky, while the borders the user sees are 0.2-1.0 %.  ``SKY_RIM_MIN`` sits between the two: below it nothing is
    taken, so the population the pipeline renders well is left exactly as it is."""
    a = _rim_case(step=0.02)
    field, info = pro.sky_rim(a)
    assert field is None, info
    assert any(s[0] == 0 and "level with the sky behind" in s[3] for s in info["sides"]), info


def test_the_rim_leaves_a_gradient_that_reaches_past_the_band():
    """The gate that keeps this tool out of the sky.  A step still standing at every depth the band reads, while the strip one band width behind has gone back to the sky, is
    not a boundary residue: it is a gradient the model never got hold of (M 31's left border enters BGE with its sky floor 51 % below the middle, because that is where the
    galaxy is).  A gradient that deep is the model's to remove, and a border tool that chased it would be drawing the sky flat across the frame - which is exactly what reading
    the border against the strip behind it, rather than against the middle, is there to make impossible."""
    a = _rim_case(step=2.0, depth=140, residue=0.0)                                                   # the step reaches past the band's inner edge while the strip behind it has
                                                                                                      # gone back to the sky
    field, info = pro.sky_rim(a)
    assert field is None, info
    assert any(s[0] == 0 and "reaches past the band" in s[3] for s in info["sides"]), info


def test_the_rim_leaves_a_deviation_confined_to_the_border_lines_to_the_ring():
    """Two instruments, one border.  A deviation standing only on the outer two lines is the frame line ``edge_ring`` was built for and has its own cause (a stage's boundary
    behaviour, not a gradient model's residue); reading it as a band would take the whole band's worth of sky on the strength of two lines."""
    a = _rim_case(step=2.0, depth=2, residue=0.0)
    field, info = pro.sky_rim(a)
    assert field is None, info
    # Which honest gate speaks depends on how much of the frame the raised band moves the frame's own spread with it: a deviation the witness will not read at all and one it
    # reads and finds nowhere in the band are both refusals, and both leave the line to ``edge_ring``, which owns a two-pixel claim about a border.
    assert any(s[0] in (0, 1) and any(k in s[3] for k in ("frame line", "does not stand", "too little", "fills", "more than a stage")) for s in info["sides"]), info


def test_the_rim_is_blind_to_its_subject_and_sharp_on_a_level_under_it():
    """The distinction the whole tool rests on, and the one its predecessors got wrong.  Bright subject lying on the border moves a line's *median* by tens of percent of the
    sky and its *sky floor* not at all, because the witness rejects filaments - so structure is refused while the level underneath it is still taken.  A tool that read the
    median instead poured light into exactly the rim it was meant to flatten (IC 1848's red band) or darkened a nebula's edge (M 31's 24 % band)."""
    quiet = _rim_case(step=0.0, filaments=0.45)
    field, info = pro.sky_rim(quiet)
    assert field is None, info                                                               # the subject alone is nothing to take
    assert any(s[0] == 0 and ("not a level" in s[3] or "level with the sky behind" in s[3]) for s in info["sides"]), info
    under = _rim_case(step=2.0, filaments=0.45)
    field, info = pro.sky_rim(under)
    assert field is not None, info                                                           # and its edge does not cost the tool its edge
    assert max(abs(v) for v in _border(under - field, 0, px=2)) < 0.0002, info


def test_the_rim_refuses_a_side_whose_channels_do_not_agree():
    """The border is one object measured three times.  A deviation the channels disagree about is not that object - it is something coloured sitting on the border, or noise -
    and taking the channels that passed would paint the border in the colour that was left out."""
    a = _rim_case(step=2.0, chans=(1.0, 1.0, 0.0))
    field, info = pro.sky_rim(a)
    assert field is None, info
    assert any(s[0] == 0 and s[2] is not None and "stays as it is" in s[3] for s in info["sides"]), info


def test_the_rim_refuses_a_border_the_subject_has_filled():
    """Blind is not the same as level.  Where the subject crowds the sky out of a line there is no floor to measure, and the tool that guessed in that situation is the one
    that darkened a nebula's rim on the strength of 194 sky pixels out of 5658.  Refusing costs a band on one frame; guessing costs the subject on any."""
    a = _rim_case(step=2.0, fill=0.9)
    field, info = pro.sky_rim(a)
    assert field is None, info                                                                                     # nothing is taken from a border whose sky has gone
    assert any(s[0] in (0, 1) and any(k in s[3] for k in ("fills", "too little", "more than a stage", "not a level")) for s in info["sides"]), info
    wing = _rim_case(step=0.0, fill=0.4, fill_level=2.0)                                                           # and a wing the sky did not make is not a level either
    field, info = pro.sky_rim(wing)
    assert field is None, info
    assert any(s[0] in (0, 1) and ("not a level" in s[3] or "not the sky's level" in s[3]) for s in info["sides"]), info


def test_the_rim_takes_a_band_down_the_side_as_well_as_one_across_it():
    """A band down the left edge is the same object as one across the top, but the fade along its border runs *down* the frame there, not across it - and getting that the wrong
    way round is not a mis-shaped correction, it fails outright on the assignment (``(h, 1)`` against ``(h,)``), which is how a tool that had only ever been seen taking
    horizontal bands reached a real left border."""
    a = _rim_case(step=2.0, sides=(2,))
    field, info = pro.sky_rim(a)
    assert field is not None, info
    assert any(s[0] == 2 and s[3] == "taken" for s in info["sides"]), info
    corrected = a - field
    assert max(abs(v) for v in _border(corrected, 2, px=2)) < 0.0002, info                                # the border is level with the sky behind it now
    assert max(abs(v) for v in _border(corrected, 0, px=2)) < 0.0002, info                                # and the sides that carried nothing are as they were


def _along(a, side):
    """Each pixel's depth in from ``side`` (0 top, 1 bottom, 2 left, 3 right)."""
    h, w = a.shape[1], a.shape[2]
    yy, xx = np.mgrid[0:h, 0:w]
    return (yy if side == 0 else h - 1 - yy) if side < 2 else (xx if side == 2 else w - 1 - xx)


def _lone_line(a, side, depth, percent, sky):
    """One line ``depth`` px in from ``side`` lifted ``percent`` of the sky, the depths either side of it left exactly as
    they were: a fluctuation of the border's own step, which reads high at one sampled depth and nowhere else."""
    return a + (percent / 100.0 * sky) * (_along(a, side) == depth)


def _plateau(a, side, d0, d1, percent, sky):
    """A second deviation lying ``d0`` to ``d1`` px in from ``side`` - what the subject or a frame-scale gradient looks
    like behind a border's own step, as opposed to one noisy line of it."""
    return a + (percent / 100.0 * sky) * ((_along(a, side) >= d0) & (_along(a, side) < d1))


def test_a_lone_line_beyond_the_step_does_not_cost_the_border_its_correction():
    """What C 20 paid for on 2026-10-31: its right border carried +0.2 % of the sky uniformly 48 px in, and the tool
    refused the whole side because one sample at 96 px stood a hair over its bar with the depths either side of it flat.
    The untaken offset went through a stretch whose slope at the sky is 75-120 and came back the export's bright edge, so a
    lone line has to be read as what it is - a fluctuation of the step in front of it - and the correction taken."""
    a = _lone_line(_rim_case(step=2.0, depth=64, sides=(0,)), 0, 96, 0.1, 0.05)          # just above the bar the line is judged on
    field, info = pro.sky_rim(a)
    assert field is not None, info                                                        # the border's own step is still taken
    assert any(s[0] == 0 and s[3] == "taken" for s in info["sides"]), info
    corrected = a - field
    assert max(abs(v) for v in _border(corrected, 0, px=2)) < 0.0002, info                 # level with the sky behind it now
    assert abs(_strip(corrected, 0, 96, 97)[0] - _strip(a, 0, 96, 97)[0]) < 1e-5, info      # and the lone line is left alone


def test_one_line_does_not_become_a_second_step_by_being_tall():
    """The contiguity half of the rule, on its own: C 20's vetoing sample stood at 40-60 % of its border's own height, so
    height alone cannot promote one line into a structure - a second step has to be there over more than a single sampled
    depth, or the deviation is the border's own step read noisily, and the frame keeps its band."""
    a = _lone_line(_rim_case(step=2.0, depth=64, sides=(0,), noise=4e-4), 0, 96, 1.2, 0.05)   # 60 % of the border's height, one line only
    field, info = pro.sky_rim(a)
    assert field is not None, info
    assert any(s[0] == 0 and s[3] == "taken" for s in info["sides"]), info


def test_a_second_step_of_its_own_height_behind_the_border_is_left_alone():
    """The refusal has to survive the relaxation, because what it refuses is the subject: a border carrying its own step
    with something of comparable height returning behind it over two sampled depths is a galaxy or a gradient, and
    levelling that would pour light into the very rim this tool exists to take out."""
    a = _plateau(_rim_case(step=2.0, depth=20, sides=(0,), noise=4e-4), 0, 48, 80, 1.2, 0.05)
    field, info = pro.sky_rim(a)
    assert field is None, info
    assert any(s[0] == 0 and "returns behind the border" in s[3] for s in info["sides"]), info


def test_a_frame_line_in_the_other_channels_does_not_veto_the_band_the_third_carries():
    """C 20's right border, exactly: R carried +0.3 % of the sky over 48 px while G and B carried only a two-pixel frame
    line - too shallow for this tool, and ``edge_ring`` declined to touch it.  Under the rule as it was, that line was a
    channel refusing the side, so the band shipped and the stretch made it a bright edge.  A channel whose own profile is
    a line carries no evidence about a band, and the band has to be taken from the channel that has one - while the
    channels that have none are left exactly as they were, or the correction would be painting the border's colour."""
    base = _rim_case(step=0.0, sides=(0,))
    d = _along(base, 0)
    a = base.copy()
    a[0] += 0.3 / 100.0 * 0.05 * (d < 48)                                              # the band, in one channel only
    a += 0.1 / 100.0 * 0.05 * (d < 2)                                                  # the frame line, in all three
    field, info = pro.sky_rim(a)
    assert field is not None, info
    assert [s[1] for s in info["sides"] if s[3] == "taken"] == [0], info                 # the band's channel, and only it
    before, after = _border(a, 0, px=8), _border(a - field, 0, px=8)                     # read past the line, on the band
    assert abs(before[0]) > 0.0001, before                                                # the band is really there
    assert abs(after[0]) < 0.2 * abs(before[0]), (before, after, info)                   # and it has gone
    edge_before, edge_after = _border(a, 0, px=1), _border(a - field, 0, px=1)
    for c in (1, 2):
        assert abs(edge_after[c] - edge_before[c]) < 1e-7, (c, edge_before, edge_after, info)   # no band, nothing taken


def _contrast_case(step=1.0, band=None, galaxy=0.0, h=400, w=600, sky=0.05, noise=0.0004, seed=3):
    """The finished frame's lightness: flat sky, one edge the background model left ``step`` noise sigmas
    of it bright, and optionally a galaxy running off that same edge.  One sigma sits deliberately between
    the two scales that matter here: far above the threshold the mask is built on (3 sigma of the *smoothed*
    scatter, a fraction of one noise sigma), which is what makes the step class that band as subject, and
    far below 3 sigma of the frame's own scatter, which is what keeps it a residue rather than a subject."""
    rng = np.random.default_rng(seed)
    lum = (sky + noise * rng.standard_normal((h, w))).astype(np.float32)
    band = int(pro.LC_EDGE_BAND * pro.CLAHE_SCALES[0]) if band is None else band
    lum[:, :band] += step * noise                                        # uniform: clears the frame's global threshold, sits at its own level
    if galaxy:
        y, x = np.mgrid[0:h, 0:w]
        lum += galaxy * np.exp(-((x / (0.35 * w)) ** 2 + ((y - h / 2) / (0.3 * h)) ** 2))
    return lum, band


def _contrast_masks(tmp_path, lum):
    """The mask as the step builds it, with and without the edge's own sky floor."""
    fin = pro.Finisher(str(tmp_path), {"ROWORDER": "BOTTOM-UP"}, {}, False)
    stats = pro._hist_stats(lum)
    mask, mi = pro.nebula_mask(lum, stats["mode"])
    floor, sides = pro.edge_sky_floor(lum, float(mi["threshold"]), int(pro.LC_EDGE_BAND * pro.CLAHE_SCALES[0]), stats)
    return fin.range_mask(lum, mask, mi), fin.range_mask(lum, mask, mi, floor=floor), sides


def test_clahe_gets_nothing_from_the_band_the_gradient_model_left_elevated(tmp_path):
    """The border's own defect (Sh2-171, LBN 429, 2026-10-30).  ``range_mask`` promises 0 on the sky and
    thresholds the lightness at one global value, so a band the background model left uniformly elevated
    clears it everywhere at once, is classed subject, and clahe - whose tile there has no pixels past the
    edge to judge it against - lifts it on top of a stretch whose slope is already ~100: this step took
    Sh2-171's left border from +161 % of the display sky to +321 % and left the interior alone, and LBN 429
    went +225 % to +324 %, which is what the export shows as a band.  Asking the band what *its* sky's top
    is takes the weight off it and nothing else: measured on those two frames the border's excess falls by
    80 % and 74 % while the step keeps 91-93 % of what it adds to the subject."""
    lum, band = _contrast_case()
    w0, w1, sides = _contrast_masks(tmp_path, lum)
    assert sides["left"].startswith("sky")
    assert float(w0[:, :8].mean()) > 0.5                                             # the elevated sky was the step's subject
    assert float(w1[:, :8].mean()) < 0.1                                             # now it is sky, which the mask is promised to ignore
    assert np.allclose(w1[band:lum.shape[0] - band, band:], w0[band:lum.shape[0] - band, band:], atol=1e-6)   # and the interior is what it was; only last bits move, from the floor being a float32 field where the threshold is a scalar


def test_clahe_keeps_the_subject_that_runs_off_the_edge(tmp_path):
    """M 31's refusal.  A band carrying the galaxy is not an elevated sky no matter how level it looks, and
    a floor taken from it would darken the nebula's edge - the failure that cost M 31 a 24 % band once
    already.  Where the band is not sky the mask has to be untouched, bit for bit, not merely gentler."""
    lum, _ = _contrast_case(galaxy=0.25)
    w0, w1, sides = _contrast_masks(tmp_path, lum)
    assert not sides["left"].startswith("sky")
    assert np.allclose(w0, w1, atol=1e-6)                                              # nothing was raised anywhere: the mask is the one the global threshold builds


def test_the_edge_floor_cannot_seam_at_the_inner_edge_of_the_band(tmp_path):
    """The ramp's whole purpose.  A floor that is the band's sky top for ``band`` px and the frame's global
    threshold one column later is a step in the mask, and a step in this mask is the hard inner edge the
    user sees on a band - the cure drawn as a new symptom.  One column's worth of the mask must not move
    more than the noise does."""
    lum, band = _contrast_case()
    h = lum.shape[0]
    w0, w1, _ = _contrast_masks(tmp_path, lum)
    taken = (w0 - w1).mean(axis=0)                                                   # what the floor alone removed, so the image's own step cannot masquerade as a seam
    assert abs(float(taken[band - 1]) - float(taken[band])) < 0.02
    assert float(taken[0]) > 0.3                                                     # and the ramp actually reaches the floor it was asked for


def test_chroma_denoise_takes_the_colour_when_the_denoiser_cannot(tmp_path):
    """C 20 (2026-10-31) delivered a sky whose red channel carried 1.64x the scatter of green and blue (0.0571
    against 0.0349 / 0.0347), with 35% red and 24% cyan speckled pixels and both sky-colour gates failed - while the
    run recorded chroma_denoise 0.95 as if it had been applied.  The amount only ever reached NXT (--dc) and Cosmic
    Clarity (--denoise-color); with both unavailable the fallback denoises RGB jointly and nothing touched the colour.
    The luminance may not move: the stretch, the star detection and the sky gates all read exactly that."""
    rng = np.random.default_rng(11)
    base = 0.08 + rng.normal(0, 2e-3, (120, 200))
    arr = np.clip(np.stack([base + rng.normal(0, 1.2e-2, (120, 200)),      # the red speckle: 6x the luminance noise
                            base + rng.normal(0, 2e-3, (120, 200)),
                            base + rng.normal(0, 1.2e-2, (120, 200))]), 0, 1).astype(np.float32)
    fin = pro.Finisher(str(tmp_path), {}, {**post.DEFAULTS, "chroma_denoise": 0.95}, False, external=False)
    out = fin.chroma_denoise(arr, why="test")
    assert float(np.abs(finish.luminance(out) - finish.luminance(arr)).max()) < 1e-5      # sum(w_k c'_k) = l exactly

    def spread(a):
        l = finish.luminance(a)
        return float(np.median([np.std(c - l) for c in a]))

    assert spread(out) < 0.5 * spread(arr)                                                # the colour noise goes down
    assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0
    rec = fin.records[-1]
    assert rec["step"] == "colour noise reduction"
    for k in ("sky_before", "sky_after"):
        assert set(rec["params"][k]) == {"spread", "a", "b", "a_median", "b_median"}
    b4, af = rec["params"]["sky_before"], rec["params"]["sky_after"]
    assert af["spread"] < 0.5 * b4["spread"]                                    # the speckle goes
    assert af["a"] < 0.5 * b4["a"] and af["b"] < 0.5 * b4["b"]                   # gates' own robust scatter
    # The re-centring is what keeps the sky's cast put (C 20's a* walked 5.42 -> 6.73 without it, see the docstring).
    # Its effect on the medians cannot be reproduced from a synthetic fixture, so what is pinned is the invariant that
    # makes it safe to apply at all: it may only move the colour, never the lightness.
    d = rec["params"]["sky_recentered"]
    assert d is not None and len(d) == 3
    # The record rounds the offsets to 5 dp, so the sum cannot be exact; it is this small only because the drift's
    # lightness component was projected out - left in, C 20 measured 2e-3 of it, three orders above this.
    assert abs(sum(wt * v for wt, v in zip(finish.LUM709, d))) < 1e-5          # sum(w_k d_k) = 0


def test_chroma_denoise_leaves_a_saturated_star_core_alone(tmp_path):
    """Star cores in a stretched picture sit exactly at the clip, and a uniform colour offset cannot be added there
    without moving the lightness: applying the re-centring everywhere cost 2.35e-3 of luminance on a synthetic core,
    which is the very invariant this step is built on.  A pixel that cannot take the offset is kept as it is."""
    rng = np.random.default_rng(5)
    H, W = 200, 340
    yy, xx = np.mgrid[0:H, 0:W]
    sky = 0.08 + rng.normal(0, 2e-3, (H, W))
    subj = (0.16 * np.exp(-(((xx - 170) ** 2) / (2 * 60.0 ** 2) + ((yy - 100) ** 2) / (2 * 45.0 ** 2)))
            + 0.55 * np.exp(-(((xx - 170) ** 2) / (2 * 12.0 ** 2) + ((yy - 100) ** 2) / (2 * 12.0 ** 2))))
    arr = np.clip(np.stack([sky + 1.9 * subj + rng.normal(0, 1.2e-2, (H, W)),
                            sky + 0.5 * subj + rng.normal(0, 2e-3, (H, W)),
                            sky + 0.6 * subj + rng.normal(0, 9e-3, (H, W))]), 0, 1).astype(np.float32)
    assert float(arr.max()) >= 1.0                                             # the case under test: a channel at the clip
    fin = pro.Finisher(str(tmp_path), {}, {**post.DEFAULTS, "chroma_denoise": 0.95}, False, external=False)
    out = fin.chroma_denoise(arr, why="test")
    d = fin.records[-1]["params"]["sky_recentered"]
    assert d is not None and max(abs(v) for v in d) > 1e-4                     # the offset really is being applied
    assert float(np.abs(finish.luminance(out) - finish.luminance(arr)).max()) < 1e-5


def _grainy_starless(h=160, w=200, sigma=0.03, seed=11):
    """A stretched starless whose sky is grainier than ``sky_pixel_noise`` allows, with a few stars in it so the star
    gates have something to judge.  The fixture the escalation loop is decided on."""
    rng = np.random.default_rng(seed)
    n = rng.normal(0, sigma, (h, w)).astype(np.float32)          # common-mode grain: the lightness is noisy, the colour is
    a = np.full((3, h, w), 0.10, np.float32) + n[None]           # clean sky, so sky_pixel_noise is the only gate it fails
    yy, xx = np.mgrid[0:h, 0:w]
    for cy, cx in ((40, 50), (90, 130), (125, 80)):
        a += (0.5 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.2 ** 2)))).astype(np.float32)[None]
    return np.clip(a, 0, 1)


def _fake_nxt(monkeypatch, cast_after_n_runs=None):
    """A stand-in for ``rc-astro nxt`` whose output noise falls as ``--di`` rises, the way the real tool's does.

    ``cast_after_n_runs``: every run past this many also paints the sky red - the candidates the loop is required to
    refuse, because lowering the noise is not permission to break a gate the incumbent had passed.  It keys on the run,
    not the amount, so the first pass (the incumbent) stays clean whatever amount it was asked for.
    """
    import cv2
    calls = []

    def run(product, src, dst, **kw):
        di = float(kw["di"])
        calls.append(round(di, 6))
        a = np.asarray(fitsio.read_rgb(src)[0], np.float32)
        sm = np.stack([cv2.GaussianBlur(c, (0, 0), 1.0) for c in a])
        out = np.clip(sm + (1.0 - di) * (a - sm), 0, 1)
        if cast_after_n_runs is not None and len(calls) > cast_after_n_runs:
            out[0] = np.clip(out[0] + 0.25, 0, 1)
        fitsio.write_rgb(dst, out, {})
        return {"flags": ["--di", f"{di:g}", "--dc", str(kw["dc"])]}

    monkeypatch.setattr(pro.tools.rcastro, "require", lambda product: None)
    monkeypatch.setattr(pro.tools.rcastro, "run", run)
    return calls


def _sky_gate_record(fin):
    return [r for r in fin.records if r["step"] == "noise reduction (sky gate)"][0]


def test_the_closing_step_escalates_until_the_delivered_sky_clears_its_gate(tmp_path, monkeypatch):
    """The chain denoised the starless, then raised its contrast, sharpened and saturated it, and shipped at
    ``sky_pixel_noise`` 0.01075 against the 0.008 gate - while the denoiser's own output had measured 0.00387, so nothing
    in the chain could see what the export saw.  The closing step measures the delivered picture the way the export
    measures it, and re-runs the tool the chain already chose until the gate is met, walking toward that tool's published
    ceiling and keeping the first amount that passes."""
    calls = _fake_nxt(monkeypatch)
    fin = pro.Finisher(str(tmp_path), {}, {**post.DEFAULTS, "luminance_denoise": 0.6}, True, external=True)
    arr = _grainy_starless()
    limit = float(gates.LIMITS["sky_pixel_noise"])
    assert fin._sky_noise_of(arr, arr) > limit                                 # the fixture really is over its gate
    fin.denoise(arr)
    assert calls == [0.6]                                                      # the requested amount, once, exactly as before
    sl, pic = fin._close_sky_noise(arr, compose=lambda x: x)
    rec = _sky_gate_record(fin)
    assert calls[1:] == sorted(set(calls[1:])) and len(calls) >= 3 and calls[-1] > 0.6      # escalated upward, no repeats
    assert rec["params"]["escalated"] is True and rec["params"]["requested"] == 0.6
    assert rec["params"]["used"] == pytest.approx(calls[-1], abs=1e-3)
    cleared = [a for a in rec["params"]["attempts"][1:] if a.get("sky_pixel_noise") is not None and a["sky_pixel_noise"] <= limit]
    assert cleared and rec["params"]["used"] == pytest.approx(cleared[0]["amount"], abs=1e-3)   # stopped at the first that passed
    assert fin._sky_noise_of(pic, sl) <= limit < fin._sky_noise_of(arr, arr)   # the sky that ships clears the gate


def test_the_closing_step_asks_the_tool_once_when_the_delivered_sky_already_passes(tmp_path, monkeypatch):
    """A sky already inside its gate must cost no second pass: on a licensed GPU tool a wasted escalation is real time
    and real texture taken off a picture that had already satisfied the gate."""
    calls = _fake_nxt(monkeypatch)
    fin = pro.Finisher(str(tmp_path), {}, {**post.DEFAULTS, "luminance_denoise": 0.6}, True, external=True)
    arr = _grainy_starless(sigma=0.0005)
    assert fin._sky_noise_of(arr, arr) <= float(gates.LIMITS["sky_pixel_noise"])
    fin.denoise(arr)
    sl, pic = fin._close_sky_noise(arr, compose=lambda x: x)
    assert calls == [0.6]
    rec = _sky_gate_record(fin)
    assert "escalated" not in rec["params"] and rec["params"]["used"] is None
    assert len(rec["params"]["attempts"]) == 1 and rec["params"]["attempts"][0]["amount"] == 0.6


def test_the_closing_step_refuses_a_candidate_that_breaks_another_gate(tmp_path, monkeypatch):
    """The step chases one number, so that number cannot be its own referee: a harder pass that lowers the delivered
    noise while newly failing a gate the incumbent passed is refused, the incumbent is what ships, and the refusal is
    legible in the record rather than silently discarded."""
    calls = _fake_nxt(monkeypatch, cast_after_n_runs=1)                   # the incumbent is clean; every harder pass is not
    fin = pro.Finisher(str(tmp_path), {}, {**post.DEFAULTS, "luminance_denoise": 0.6}, True, external=True)
    arr = _grainy_starless()
    fin.denoise(arr)
    sl, pic = fin._close_sky_noise(arr, compose=lambda x: x)
    rec = _sky_gate_record(fin)
    assert "sky_chroma" not in gates.compute(pic, starless=sl)["failed"]       # the shipped picture carries no cast
    assert rec["params"]["used"] is None and "escalated" not in rec["params"]  # the incumbent is what shipped
    refused = [a for a in rec["params"]["attempts"][1:] if "sky_chroma" in (a.get("failed") or [])]
    assert refused and refused[0]["kept"] is False and len(calls) > 1          # harder passes really were tried, and rejected
    assert fin._sky_noise_of(pic, sl) > float(gates.LIMITS["sky_pixel_noise"])  # noisy, and knowingly so


def test_chroma_denoise_off_is_the_array_itself(tmp_path):
    """chroma_denoise 0 must be no operation at all, and must say so in the chain rather than record a smoothing."""
    fin = pro.Finisher(str(tmp_path), {}, {**post.DEFAULTS, "chroma_denoise": 0.0}, False, external=False)
    arr = np.full((3, 40, 40), 0.1, np.float32)
    assert fin.chroma_denoise(arr, why="test") is arr
    assert fin.records[-1]["status"] == "skipped"


def test_the_step_record_sees_a_colour_cast_the_luminance_levels_cannot():
    """C 49 (2026-10-31): the export's sky is red by one 8-bit code on the nebula side and neutral on the other,
    yet every level the chain recorded - median, p99, p99.9, clipped, white, all of them luminance - was the same
    before and after the steps that did it, so the run left no witness of which step tinted the sky.  The record
    has to carry the sky per channel, in the gate's own sky samples, or a colour cast stays unattributable once
    the pictures are gone."""
    rng = np.random.default_rng(4)
    grad = 0.002 * np.indices((256, 256))[1] / 255.0                      # a floor slope, so the sky mask is real
    sky = (0.10 + grad + rng.normal(0, 1e-3, (3, 256, 256))).astype(np.float32)
    tinted = sky.copy()
    tinted[0] -= 2.0 / 255.0                                              # the sky's red, down by two codes
    a, b = pro.level_stats(sky), pro.level_stats(tinted)
    assert abs(a["median"] - b["median"]) < 0.003, (a["median"], b["median"])   # luminance barely notices
    assert abs(a["sky_channels"][0] - b["sky_channels"][0]) > 0.006, (a["sky_channels"], b["sky_channels"])
    assert a["sky_channels"][1:] == b["sky_channels"][1:]                 # and only the channel that changed


def test_a_layer_with_no_sky_records_no_sky_channels():
    """The stars stretch hands the star layer - black wherever the stars are not - to the same step recorder, and
    its darkest samples are the floor itself, so the channel ratios of them are not a colour measurement: the IC
    1848 chain replay (2026-11-01) recorded R-B = -1.000 at that step, a cast in a frame that has no sky to tint.
    A sky sample has to sit below the frame's own level to be one."""
    rng = np.random.default_rng(11)
    stars = np.zeros((3, 256, 256), np.float32)
    for y, x in rng.integers(8, 248, (60, 2)):
        stars[:, y, x] = rng.uniform(0.4, 0.9, 3)                        # points, and no background anywhere
    rec = pro.level_stats(stars)
    assert "sky_channels" not in rec, rec                                   # no cast invented for a skyless layer
    assert rec["max"] > 0.3, rec                                          # the guard drops nothing else: the points

    lit = stars + 0.05                                                    # now the frame has a floor to measure
    assert pro.level_stats(lit).get("sky_channels"), pro.level_stats(lit)


# --- SCNR: the step is judged on the sky it leaves, by the gate's own number --------------------------------------

def _rmgreen(monkeypatch, calls):
    """A stand-in for ``siril rmgreen 3 <amount>`` with the behaviour the chain recorded on M 31 (2026-11-01): the
    green it removes is estimated from the **whole frame** - which is why a subject's cast reaches the sky - and the
    light it lifts out of green comes back through red and blue with the lightness preserved (that sky went
    G -0.0183 while R and B each rose +0.0459: the ratio 0.7152 / (0.2126 + 0.0722) = 2.5112)."""
    def siril(self, tag, arr, lines, stage, *, out=None, extra=None):
        calls.append(lines[0])
        a = np.clip(np.nan_to_num(np.asarray(arr, np.float32)), 0.0, 1.0)
        taken = float(lines[0].split()[2]) * float(np.mean(a[1] - (a[0] + a[2]) / 2.0))
        res = a.copy()
        res[1] -= taken
        res[0] += 2.5112 * taken
        res[2] += 2.5112 * taken
        return np.clip(res, 0.0, 1.0).astype(np.float32)
    monkeypatch.setattr(pro.Finisher, "siril", siril)


def _neutral_sky_with_green_nebula(cast, fraction=0.3):
    """A stretched picture: a neutral sky with a little noise, and a nebula over ``fraction`` of the frame carrying
    the whole green cast - M 31's shape, where the cast belongs to the subject and not to the sky."""
    a = RNG.normal(0.10, 0.0005, (3, 120, 160)).astype(np.float32)
    a[1, :int(120 * fraction), :] += cast / fraction
    return np.clip(a, 0, 1).astype(np.float32)


def test_scnr_that_colours_the_sky_is_refused_and_its_pixels_handed_back(tmp_path, monkeypatch):
    """M 31 (Seestar, 2026-11-01): a green nebula bought SCNR 0.6 and Siril took that cast out of the **whole frame**,
    lightness preserved, so a sky that arrived neutral left magenta - the chain recorded the sky at |a*| 0.4 / |b*| 0.6
    going to 13.4 / 9.0, 6.7 x the ``sky_chroma`` gate - and the render was delivered with the cast in it.  A step
    whose business is the subject does not get to colour the sky."""
    calls: list[str] = []
    _rmgreen(monkeypatch, calls)
    fin = pro.Finisher(str(tmp_path), {}, {**post.DEFAULTS, "scnr": 0.6}, False, external=False)
    arr = _neutral_sky_with_green_nebula(0.03)
    assert max(fin._sky_chroma(arr).values()) < float(gates.LIMITS["sky_chroma"])      # the sky arrives inside its gate
    out = fin.colour(arr)
    rec = fin.records[-1]
    assert rec["tool"] == "siril rmgreen" and rec["status"] == "refused", rec
    assert out is arr                                                                  # the picture is the one the step was handed
    assert len(calls) == len(pro.SCNR_SKY_TRIES)                                         # weaker was tried, and still coloured the sky
    assert rec["params"]["sky_chroma_before"]["a"] < 1.0 < rec["params"]["sky_chroma_after"]["a"]


def test_scnr_is_kept_when_the_sky_stays_inside_its_gate(tmp_path, monkeypatch):
    """The refusal is for the step that takes the sky past the gate, not for SCNR: a subject cast small enough that
    the sky still lands inside ``sky_chroma`` ships with the tool's pixels, at the amount that was asked for, in one
    run."""
    calls: list[str] = []
    _rmgreen(monkeypatch, calls)
    fin = pro.Finisher(str(tmp_path), {}, {**post.DEFAULTS, "scnr": 0.6}, False, external=False)
    arr = _neutral_sky_with_green_nebula(0.004)
    out = fin.colour(arr)
    rec = fin.records[-1]
    assert rec["status"] == "done" and rec["params"]["amount"] == 0.6, rec
    assert calls == ["rmgreen 3 0.600"]
    assert out is not arr and float(np.median(out[1]) - np.median(arr[1])) < -1e-4      # the tool's green is the picture's


def test_scnr_ships_the_weakest_pass_that_leaves_the_sky_its_colour(tmp_path, monkeypatch):
    """The amount is the subject's, so the first candidate is not automatically the one that ships: the chain walks
    down and keeps the strongest pass whose sky is still inside the gate - refusing outright would take the subject's
    correction away with the sky's cast."""
    calls: list[str] = []
    _rmgreen(monkeypatch, calls)
    fin = pro.Finisher(str(tmp_path), {}, {**post.DEFAULTS, "scnr": 0.6}, False, external=False)
    arr = _neutral_sky_with_green_nebula(0.015)
    out = fin.colour(arr)
    rec = fin.records[-1]
    assert rec["status"] == "done" and rec["params"]["amount"] == 0.15, rec
    assert calls == ["rmgreen 3 0.600", "rmgreen 3 0.300", "rmgreen 3 0.150"]
    assert [t["amount"] for t in rec["params"]["tried"]] == [0.6, 0.3, 0.15]


# --- the stretched-image denoiser is judged on the sky it leaves, by the gate's own number -------------------------

def _red_shift_denoiser(calls, k=0.02):
    """A stand-in for a stretched-image denoiser with the behaviour ``output/.ab12/audit_exports.py --cast`` measured on
    IC 434 (2026-11-01): the harder it works, the further it walks the sky toward red (+31.5e-3 in (R-G)/(R+G) at GraXpert's
    full strength), here lightness-preserving - R up, G and B down by half each - in proportion to the strength it was given."""
    def run(a: float, a_arr: np.ndarray):
        calls.append(a)
        s = k * a
        res = np.asarray(a_arr, np.float32).copy()
        res[0] += s
        res[1] -= s / 2
        res[2] -= s / 2
        return np.clip(res, 0.0, 1.0).astype(np.float32), f"denoise strength={a:g}"
    return run


def _denoise_candidate():
    """A stretched starless: sky everywhere, neutral, with its own noise - the frame the denoiser is handed.  Its own
    generator, not the module's: with the shared ``RNG`` every later test's fixture depends on how many draws this file has."""
    return np.random.default_rng(11).normal(0.10, 5e-4, (3, 120, 160)).astype(np.float32)


def test_a_denoiser_that_colours_the_sky_is_refused_and_its_pixels_handed_back(tmp_path):
    """The step that denoises the stretched picture was the one colour-changing step of the chain with no refusal rule and no
    recorded before/after, and on the fleet it is the step that moves the sky's colour most wherever no palette runs.  It is
    now judged by the number the export is judged by, and a frame whose sky cannot trust the tool keeps its own pixels."""
    fin = pro.Finisher(str(tmp_path), {}, {**post.DEFAULTS}, False, external=False)
    arr = _denoise_candidate()
    assert max(fin._sky_chroma(arr).values()) < float(gates.LIMITS["sky_chroma"])          # the sky arrives inside its gate
    calls: list[float] = []
    out = fin._keep_denoise(_red_shift_denoiser(calls), arr, 0.7, tool="graxpert denoise", ceiling=0.8, start=time.time(),
                            params=lambda a: {"strength": a}, amounts=(0.7, 0.6), source="test stand-in")
    rec = fin.records[-1]
    assert rec["step"] == "noise reduction (stretched)" and rec["tool"] == "graxpert denoise" and rec["status"] == "refused", rec
    assert out is arr                                                                        # the picture is the one the step was handed
    assert calls == [0.7, 0.6]                                                               # every strength the tool documents was asked
    assert fin.denoiser is None                                                              # and the closing step must not re-ask a refused tool
    assert rec["params"]["sky_chroma_before"]["a"] < 1.0 < rec["params"]["sky_chroma_after"]["a"], rec["params"]


def test_a_denoiser_that_tints_one_quadrant_is_refused_though_it_smoothed_the_chroma(tmp_path):
    """``median |a*|`` over the sky is dominated by chroma *noise*, so a denoiser that smooths that noise while tinting
    part of the sky *improves* the only number the guard used to read - and the picture ships failing the gate nobody
    was policing: C 20 (2026-11-01) was delivered at |a*| 1.73 of 2.0 with its four quadrants' signed tints 1.97 apart
    of 1.5. The tint's direction is judged now, by the gate's own signed quadrant medians."""
    fin = pro.Finisher(str(tmp_path), {}, {**post.DEFAULTS}, False, external=False)
    arr = _denoise_candidate()
    s = 0.010                                                       # one quarter of the sky tinted red, lightness kept:
                                                                    # the C 20 case, spread 1.92 of 1.5 at |a*| 1.75 of 2.0

    def run(a: float, a_arr: np.ndarray):
        res = np.asarray(a_arr, np.float32).copy()
        res[0, :60, :80] += s
        res[1, :60, :80] -= s / 2
        res[2, :60, :80] -= s / 2
        return res, f"denoise strength={a:g}"

    out_tinted, _ = run(0.7, arr)
    meas = fin._sky_chroma(out_tinted)
    assert meas["spread"] > float(gates.LIMITS["sky_chroma_quadrants"]), meas      # the quadrant gate this picture fails
    assert max(meas["a"], meas["b"]) < float(gates.LIMITS["sky_chroma"]), meas     # and the chroma gate it passes - the old rule alone

    out = fin._keep_denoise(run, arr, 0.7, tool="graxpert denoise", ceiling=0.8, start=time.time(),
                            params=lambda a: {"strength": a}, amounts=(0.7,), source="test stand-in")
    rec = fin.records[-1]
    assert rec["status"] == "refused" and out is arr, rec
    assert rec["params"]["sky_chroma_before"]["spread"] < meas["spread"], rec["params"]
    assert fin.denoiser_refused["refused_for"] == "sky_chroma_quadrants", fin.denoiser_refused
    assert "quadrants" in rec["note"] and "sky_chroma_quadrants" in rec["note"], rec["note"]


def test_the_denoiser_ships_the_weakest_strength_that_leaves_the_sky_its_colour(tmp_path):
    """Refusing outright would take the noise reduction away with the cast: the step is asked again at the weakest strength
    that tool itself documents, and what ships is the pass whose sky is still inside the gate, recorded at that strength."""
    fin = pro.Finisher(str(tmp_path), {}, {**post.DEFAULTS}, False, external=False)
    arr = _denoise_candidate()
    calls: list[float] = []
    out = fin._keep_denoise(_red_shift_denoiser(calls), arr, 0.7, tool="rc-astro nxt", ceiling=1.0, start=time.time(),
                            params=lambda a: {"di": a}, amounts=(0.7, 0.35), source="test stand-in")
    rec = fin.records[-1]
    assert rec["status"] == "done" and rec["params"]["di"] == 0.35, rec
    assert calls == [0.7, 0.35]
    assert rec["commands"] == ["denoise strength=0.35"]                                      # the command of the pass that ships
    assert [t["amount"] for t in rec["params"]["tried"]] == [0.7, 0.35]                      # and both measurements recorded
    assert rec["params"]["tried"][0]["sky_chroma"]["a"] > rec["params"]["tried"][1]["sky_chroma"]["a"]
    assert fin.denoiser is not None and fin.denoiser[3] == 0.35                              # the closing step escalates from here
    assert out is not arr and float(np.median(out[0]) - np.median(arr[0])) > 1e-4             # the accepted pass is the picture


def test_a_refused_denoiser_is_named_as_the_reason_the_sky_gate_step_did_nothing(tmp_path):
    """Refusing the denoiser for colouring the sky also takes away the tool the sky-noise ladder escalates, so the sky can
    stay noisy - a consequence of the guard, which the chain has to state.  "Nothing was available to ask" would blame the
    machine for what was a measured decision about this frame's sky."""
    fin = pro.Finisher(str(tmp_path), {}, {**post.DEFAULTS}, False, external=True)
    arr = _denoise_candidate()
    fin._keep_denoise(_red_shift_denoiser([]), arr, 0.7, tool="graxpert denoise", ceiling=0.8, start=time.time(),
                      params=lambda a: {"strength": a}, amounts=(0.7,), source="test stand-in")
    assert fin.denoiser is None and fin.denoiser_refused is not None and fin.denoiser_refused["tool"] == "graxpert denoise"
    starless = _denoise_candidate()
    fin._close_sky_noise(starless, lambda sl: sl)
    rec = fin.records[-1]
    assert rec["step"] == "noise reduction (sky gate)" and rec["status"] == "skipped", rec
    assert "graxpert denoise" in rec["note"] and "refused" in rec["note"], rec["note"]
    assert rec["params"]["denoiser_refused"]["tool"] == "graxpert denoise", rec["params"]


def test_an_uneven_handed_denoiser_is_kept_at_the_strength_asked(tmp_path):
    """A step may leave the sky as coloured as it arrived: whose colour that is belongs to the calibration, not to this step.
    So an innocent denoiser runs once, at the requested strength, with the before/after recorded and nothing refused."""
    fin = pro.Finisher(str(tmp_path), {}, {**post.DEFAULTS}, False, external=False)
    arr = _denoise_candidate()
    calls: list[float] = []
    out = fin._keep_denoise(_red_shift_denoiser(calls, k=0.004), arr, 0.7, tool="cosmic clarity denoise", ceiling=1.0,
                            start=time.time(), params=lambda a: {"denoise_strength": a}, amounts=(0.7, 0.35),
                            source="test stand-in")
    rec = fin.records[-1]
    assert rec["status"] == "done" and rec["params"]["denoise_strength"] == 0.7, rec
    assert calls == [0.7]                                                                    # the second pass was never needed
    assert rec["params"]["sky_chroma_after"]["a"] < float(gates.LIMITS["sky_chroma"])


# --- the final midtones, under the sky_median gate ----------------------------------------------------------------------------

def _flat_sky_picture(sky=0.09):
    """A stretched picture that is sky everywhere but for one bright block, so the gate's sky samples are the flat field."""
    a = RNG.normal(sky, 5e-4, (3, 120, 160)).astype(np.float32)
    a[:, 10:40, 60:110] = 0.7
    return np.clip(a, 0, 1).astype(np.float32)


def _finalise(tmp_path, monkeypatch, **params):
    monkeypatch.setattr(pro.Finisher, "siril", lambda self, tag, arr, lines, stage, **kw: arr)
    fin = pro.Finisher(str(tmp_path), {}, {**post.DEFAULTS, **post.EXTRA_DEFAULTS, **params}, False, external=False)
    fin.finalise(_flat_sky_picture())
    rec = fin.records[-1]
    p = rec["params"]
    x0 = (p["background"] - p["BP"]) / max(1.0 - p["BP"], 1e-9)         # the sky the midtones are handed
    return rec, p, float(pro.mtf_balance(p["midtones"], np.float64(x0)))


def test_the_final_midtones_are_solved_back_when_the_brightness_takes_the_sky_over_its_gate(tmp_path, monkeypatch):
    """The sky's level is the picture's own spec (``sky_median`` 0.08 - 0.12) and the Midtones/brightness the chain is
    handed can put it above the window: M 31 (Seestar, 2026-11-01) left finalise with the sky at 0.134 and was
    delivered failing its own gate, nothing in the record saying which number had been missed.  The step now solves
    the same midtones the ``sky_levelling`` switch solves - to the edge of the window - and records what it moved."""
    rec, p, sky_after = _finalise(tmp_path, monkeypatch, brightness=0.5)
    assert p["sky_median_clamped"] == {"from": 0.2288, "to": 0.12, "brightness": 0.5}, p["sky_median_clamped"]
    assert abs(sky_after - float(gates.LIMITS["sky_median"][1])) < 1e-3, sky_after
    assert rec["commands"][0] == f"mtf 0 {p['midtones']:.4f} 1"          # the command is the solved value, not the slider's
    assert p["midtones"] > finish.brightness_to_mid(0.5)                  # and it brightened nothing: the sky came down


def test_the_final_midtones_are_left_alone_when_the_sky_is_already_inside_its_gate(tmp_path, monkeypatch):
    """The guard is a floor under the slider, not a replacement for it: a render whose sky already sits inside
    ``sky_median`` ships with the midtones it was given, byte for byte."""
    rec, p, _ = _finalise(tmp_path, monkeypatch, brightness=0.1135)
    assert p["sky_median_clamped"] is None
    assert p["midtones"] == round(finish.brightness_to_mid(0.1135), 5)


def test_sky_median_guard_off_leaves_the_owner_s_midtones_untouched(tmp_path, monkeypatch):
    """The look is the owner's to choose: the guard is an opt-out like the other measured guards, and off it must
    deliver the requested brightness even when that leaves the sky over its gate - with the failure left in the
    gates, where it is reported, rather than silently corrected."""
    rec, p, sky_after = _finalise(tmp_path, monkeypatch, brightness=0.5, sky_median_guard=False)
    assert p["sky_median_clamped"] is None
    assert p["midtones"] == round(finish.brightness_to_mid(0.5), 5)
    assert sky_after > float(gates.LIMITS["sky_median"][1])


# --- the stretch's black clip, judged on the sky it leaves --------------------------------------------------------------------

def _linear_sky_picture():
    """A linear frame: sky at 0.0125 (what the restoration hands the chain) with one bright block for a subject."""
    a = RNG.normal(0.0125, 2e-4, (3, 240, 320)).astype(np.float32)
    a[:, 20:70, 150:250] = 0.25
    return np.clip(a, 0, 1).astype(np.float32)


def _cells(a, spread, block=96):
    """Cell-by-cell multipliers of the size the clip leaves behind: the clip is a threshold at the sky minus sigma
    times the sky's own noise, so the cells whose sky is locally noisier lose more of their faintest pixels and come
    out darker.  C 20 (EXP_15, 2026-09-30) measured +25 % / -29.8 % with the clip and +-8 % without it."""
    h, w = a.shape[-2:]
    pat = np.clip(RNG.normal(1.0, spread, (h // block + 1, w // block + 1)), 0.35, 1.65)
    return a * np.kron(pat, np.ones((block, block)))[:h, :w]


def _stat_stretch(tmp_path, monkeypatch, spreads, fail_without_clip=False):
    """A stand-in for ``pyscript Statistical_Stretch.py`` that stretches to the requested median and applies the
    measured clip behaviour: with the black clip the sky leaves in cells, without it the sky leaves even."""
    calls: list[str] = []

    def siril(self, tag, arr, lines, stage, *, out=None, extra=None):
        line = lines[0]
        calls.append(line)
        if "Statistical_Stretch.py" not in line:
            return np.asarray(arr, np.float32)                 # the step's other commands (split/pm/rgbcomp) are not what is under test
        if fail_without_clip and "-noblackclip" in line:
            raise pro.SirilError("pyscript failed")
        a = np.clip(np.nan_to_num(np.asarray(arr, np.float32)), 0.0, 1.0)
        med = float(line.split("-median ")[1].split()[0])
        scale = med / max(float(np.quantile(finish.luminance(a), 0.25)), 1e-9)
        b = np.clip(a * scale, 0.0, 1.0)
        spread = spreads[1] if "-noblackclip" in line else spreads[0]
        return np.clip(_cells(b, spread), 0.0, 1.0).astype(np.float32)

    monkeypatch.setattr(pro.Finisher, "siril", siril)
    fin = pro.Finisher(str(tmp_path), {}, post.DEFAULTS, False, external=False)
    fin.stretch(_linear_sky_picture())
    return calls, next(r for r in fin.records if 'Statistical' in str(r.get('tool')))


def test_the_stretch_loses_its_black_clip_when_the_clip_is_what_unevens_the_sky(tmp_path, monkeypatch):
    """The fleet ledger's "wide rim offsets" (C 20 +36 %, M 31 -17 %, NGC 1499 +13/-12) are not in the sky and not
    in the linear: the frames arrive at this step flat to 0.1 % and leave with the sky-mask cells spread +-30 %.  The
    script offers no background option at all - the unevenness is its black clip, a threshold at the sky minus sigma
    times the sky's own noise, amplified by the toe's slope (75-120 at the sky).  Running the same command with
    ``-noblackclip`` measured +-8 %, so the step runs both and the sky decides."""
    calls, rec = _stat_stretch(tmp_path, monkeypatch, (0.30, 0.08))
    p = rec["params"]
    assert p["black_clip"] is False, p
    assert rec["commands"][0].endswith("-noblackclip"), rec["commands"]
    assert [t["black_clip"] for t in p["tried"]] == [True, False]
    lo, hi = p["tried"][1]["sky_lump"], p["tried"][0]["sky_lump"]
    assert hi > 20.0 and lo * pro.STRETCH_CLIP_GAIN < hi, p["tried"]      # the measure sees the cells the clip leaves


def test_the_stretch_keeps_its_black_clip_when_dropping_it_would_not_even_the_sky(tmp_path, monkeypatch):
    """The decision is a comparison, not a threshold, because unevenness can be the subject: a sky-mask cell that is
    bright because a galaxy crosses it measures the same with and without the clip, so the clip stays - a render
    loses its black point only when the black point was what cost it the sky.  (Both M 31 sessions in fact measured
    the clip as the cost and ship without it; this is the case where dropping it changes nothing.)"""
    calls, rec = _stat_stretch(tmp_path, monkeypatch, (0.21, 0.21))
    p = rec["params"]
    assert p["black_clip"] is True, p
    assert "-noblackclip" not in rec["commands"][0]
    assert abs(p["tried"][0]["sky_lump"] - p["tried"][1]["sky_lump"]) < 6.0, p["tried"]


def test_a_failing_clip_free_trial_ships_the_clipped_stretch_and_says_so(tmp_path, monkeypatch):
    """The clip-free run is a candidate, never a requirement: if the script fails on it, the render still ships with
    the clip and the failed trial is in the record rather than lost."""
    calls, rec = _stat_stretch(tmp_path, monkeypatch, (0.30, 0.08), fail_without_clip=True)
    p = rec["params"]
    assert p["black_clip"] is True and "-noblackclip" not in rec["commands"][0]
    assert p["tried"][1]["error"].startswith("pyscript failed"), p["tried"]


def test_a_stretch_whose_sky_is_already_flat_does_not_pay_for_a_second_run(tmp_path, monkeypatch):
    """The clip-free trial is a whole second run of the script, and it is adopted only when it comes back
    ``pro.STRETCH_CLIP_GAIN`` times more even: below ``pro.STAT_CLIP_FLOOR`` there is nothing that visible to win, so
    the script runs once and the record says the trial was not made, rather than looking like one that failed."""
    calls, rec = _stat_stretch(tmp_path, monkeypatch, (0.0005, 0.0005))
    stat = [c for c in calls if "Statistical_Stretch.py" in c]
    p = rec["params"]
    assert len(stat) == 1 and "-noblackclip" not in stat[0], calls
    assert p["black_clip"] is True, p
    assert len(p["tried"]) == 2 and p["tried"][1]["sky_lump"] is None, p["tried"]
    assert 0.0 < p["tried"][0]["sky_lump"] < pro.STAT_CLIP_FLOOR, p["tried"]
    assert p["tried"][1]["skipped"].startswith("the clipped sky already measures"), p["tried"]


# --- the sky's channel-to-channel tilt (the sky_chroma_quadrants family) ---------------------------------------------------------------

def _tilted_sky(tilt=0.045, nebula=0.0):
    """A linear frame whose quarters' sky channels have drifted apart - the C 20 case of 2026-11-01, where the restoration left the
    bottom-right quarter at b* 4.88 while the other three sat at 1.5-2.0 and the registered stack beneath them was neutral at 0.88.
    Built as smooth ramps on two channels (a tilt, not a step); ``nebula`` paints a bright block over the right of the frame, which is
    what Siril's degree-2 model mistook for sky on the real frame."""
    a = np.random.default_rng(29).normal(0.0125, 2e-4, (3, 240, 320)).astype(np.float32)   # its own generator: with the module's, every later
                                                                                            # fixture here would depend on how many draws this file has
    h, w = a.shape[1:]
    yy, xx = np.mgrid[0:h, 0:w]
    a[0] *= (1.0 + tilt * (xx / w - 0.5)).astype(np.float32)                       # red rising to the right
    a[2] *= (1.0 - tilt * (yy / h - 0.5) * 0.6).astype(np.float32)                  # blue falling downward
    if nebula:
        a[:, 20:h - 20, int(0.55 * w):] *= (1.0 + nebula)
    return np.maximum(a, 1e-6).astype(np.float32)


def test_the_tint_levens_quarters_whose_sky_channels_have_drifted_apart():
    """The defect: a frame that fails ``sky_chroma_quadrants`` on a cast nothing between the frame and the export reaches - the colour
    gradient step is refused above ``SKY_STEPS_MAX_COVERAGE``, the neutralisation's offsets are global, and the pedestal/rim tools are
    achromatic.  Adoption is judged by the gate's own reading, so that is what the test judges too."""
    a = _tilted_sky()
    before, _ = pro.sky_quadrant_spread(a)
    assert before > float(gates.LIMITS["sky_chroma_quadrants"]), before              # the fixture really is a sky the gate fails
    gain, info = pro.sky_tint(a)
    assert gain is not None, info
    after, _ = pro.sky_quadrant_spread(a * gain)
    assert after * pro.SKY_TINT_GAIN <= before, info                                 # better, by the margin the tool demands of itself
    assert info["spread_after"] == pytest.approx(after, abs=0.05)                     # and the record says what was actually measured


def test_the_tint_leaves_the_sky_its_own_colour():
    """Each channel keeps the level it arrived with: the frame's overall balance is ``neutralise``'s offsets, one step later, on purpose.
    The uncentred version of this field pulled C 20's registered stack from b* -2.9 to about 0 - the sky's colour removed rather than a
    residue, which is the sky's airglow and dust-reflected starlight the astrophotographer paid for."""
    a = _tilted_sky()
    gain, info = pro.sky_tint(a)
    assert gain is not None, info
    sm = gates.sky_mask(finish.luminance(pro.autostretch_display(a)))
    m0 = [float(np.median(a[c][sm])) for c in range(3)]
    m1 = [float(np.median((a * gain)[c][sm])) for c in range(3)]
    assert all(abs(m1[c] - m0[c]) / m0[c] < 0.01 for c in range(3)), (m0, m1)


def test_the_tint_leaves_a_frame_whose_quarters_already_agree():
    """A clean frame is left strictly alone: a tool that processed frames that need no processing would be indistinguishable from one
    that damages them, and every frame in the fleet passes through here."""
    a = _linear_sky_picture()
    assert pro.sky_quadrant_spread(a)[0] <= pro.SKY_TINT_MIN_SPREAD
    gain, info = pro.sky_tint(a)
    assert gain is None and "refused" in info, info


def test_the_tint_takes_nothing_from_a_nebula_sitting_in_one_quarter():
    """The hazard the colour-gradient refusal was written for (C 20, 2026-10-15: the degree-2 model "took the North America Nebula's own
    red"), reproduced here as a bright block over a quarter whose sky is level.  Measured on the real frame, running that step took out a
    model 10x the sky's actual tilt, anti-correlated with it in R (corr -0.23), and left the quarters **further** apart (a 1.89 -> 3.44);
    the subject is therefore never in the fit.  And the refusal must not cost the tool its edge: the same frame with a real sky tilt
    underneath the nebula still gets its sky levelled, with the nebula's own colour ratios untouched."""
    a = _tilted_sky(nebula=6.0)
    sky_only = _tilted_sky(tilt=0.0)                                                  # level sky, same nebula
    gain, info = pro.sky_tint(sky_only)
    assert gain is None, info                                                         # the red is the nebula's, and it stays
    before, _ = pro.sky_quadrant_spread(a)
    gain, info = pro.sky_tint(a)
    assert gain is not None, info                                                     # the cast under it is still corrected
    after, _ = pro.sky_quadrant_spread(a * gain)
    assert after * pro.SKY_TINT_GAIN <= before, info
    neb = slice(20, 220), slice(176, 320)
    r0 = float(np.median(a[0][neb]) / np.median(a[2][neb]))
    r1 = float(np.median((a * gain)[0][neb]) / np.median((a * gain)[2][neb]))
    assert max(info["gain_ptp_pct"]) < 6.0, info                                        # the correction stayed the sky's own size: the 7x-brighter red block never
                                                                                        # entered the fit (Siril's model, shown that same nebula, came back 10x the sky's
                                                                                        # tilt and left the quarters further apart)
    assert abs(r1 - r0) / r0 <= max(info["gain_ptp_pct"]) / 100.0 + 0.005, (r0, r1)      # what the subject's colour may move is the plane's own amplitude, which is
                                                                                        # capped - applying the field under a blurred sky mask instead would draw the seam
                                                                                        # this file's other tools exist to remove, at the stretch's slope of a hundred


def test_the_tint_refuses_a_tilt_that_is_the_sky_own_colour():
    """Past ``SKY_TINT_CAP`` the channel-to-channel difference is not a residue a plane should touch but the frame's colour - dual-band,
    dust, an unbalanced white - and correcting it would be white-balancing the sky without being asked."""
    gain, info = pro.sky_tint(_tilted_sky(tilt=0.6))
    assert gain is None and "own colour" in info["refused"], info


def test_the_neutralisation_records_the_tint_it_applied(tmp_path, monkeypatch):
    """Wired into the chain: the gain ships through ``neutralise``, which is where the sky's other spatial models are applied - immediately
    before the stretch, whose slope at the sky is hundreds and which is the only reason a 1 % tilt reaches the export at all."""
    monkeypatch.setattr(pro.Finisher, "siril", lambda self, tag, arr, lines, stage, **kw: arr)
    fin = pro.Finisher(str(tmp_path), {}, post.DEFAULTS, False, external=False)
    fin.neutralise(_tilted_sky())
    rec = fin.records[-1]
    assert rec["tool"].startswith("sky tint + "), rec["tool"]
    t = rec["params"]["tint"]
    assert t["spread_before"] > t["spread_after"] >= 0.0 and t["samples"] > pro.SKY_TINT_SAMPLES, t
