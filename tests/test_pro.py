"""Tests of the professional finishing chain: published-formula identities, the NBN port against the script's own function, the GHS fit, the gates and autofinish's rules."""
from __future__ import annotations

import json
import os
import subprocess
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
    assert af.STAR_REDUCTIONS == (0.6, 0.5, 0.4, 0.3, 0.2)
    assert af.hdr_rule(0.0006, 0.0) and af.hdr_rule(0.0, 0.0006) and not af.hdr_rule(0.0004, 0.0004)
    assert af.star_boost(0.2) == 1.0 and af.star_boost(0.08) == 1.5 and af.star_boost(0.01) == 2.0
    img = np.zeros((3, 50, 50), np.float32)
    img[1] = 0.3
    assert af.green_cast(img, None) > af.SCNR_CAST


def test_autofinish_searches_the_largest_passing_value():
    """The range search: the largest value that passes when passing is monotone, the top when it passes, the bottom (not passed) when nothing does."""
    v, ok = af.largest_passing(0.0, 1.0, lambda x: x <= 0.37)
    assert ok and 0.37 - 1.0 / 2 ** af.SEARCH_STEPS <= v <= 0.37
    assert af.largest_passing(0.0, 1.0, lambda x: True) == (1.0, True)
    assert af.largest_passing(0.0, 1.0, lambda x: False) == (0.0, False)


def test_autofinish_black_point_is_the_stf_shadows_clip_of_the_sky():
    rng = np.random.default_rng(3)
    img = np.clip(0.1 + 0.01 * rng.standard_normal((3, 300, 300)), 0, 1).astype(np.float32)
    bp, med, sig = af.black_point(img)
    assert med == pytest.approx(0.1, abs=0.002) and sig == pytest.approx(finish.mad_sigma(finish.luminance(img).ravel()), rel=0.05)
    assert bp == pytest.approx(med - af.STF_SHADOWS * sig, abs=1e-6)


def test_autofinish_gamut_fraction_counts_pinned_channels():
    img = np.full((3, 10, 10), 0.3, np.float32)
    img[1:, :5] = 0.0                                                   # half the pixels: pure red, green and blue at 0
    assert af.gamut_fraction(img, None) == pytest.approx(0.5)


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
        line = next(x for x in lines if x.startswith("rmgreen"))
        calls.append(line)
        a = np.clip(np.nan_to_num(np.asarray(arr, np.float32)), 0.0, 1.0)
        taken = float(line.split()[2]) * float(np.mean(a[1] - (a[0] + a[2]) / 2.0))
        res = a.copy()
        res[1] -= taken
        res[0] += 2.5112 * taken
        res[2] += 2.5112 * taken
        res = np.clip(res, 0.0, 1.0)
        m = (extra or {}).get("sc_m")
        return (a + m * (res - a) if m is not None else res).astype(np.float32)          # the chain's pm blend through the mask
    monkeypatch.setattr(pro.Finisher, "siril", siril)


def _neutral_sky_with_green_nebula(cast, fraction=0.3):
    """A stretched picture: a neutral sky with a little noise, and a nebula over ``fraction`` of the frame carrying
    the whole green cast - M 31's shape, where the cast belongs to the subject and not to the sky."""
    a = RNG.normal(0.10, 0.0005, (3, 120, 160)).astype(np.float32)
    a[1, :int(120 * fraction), :] += cast / fraction
    return np.clip(a, 0, 1).astype(np.float32)


def test_scnr_runs_under_the_range_mask_and_leaves_the_sky_alone(tmp_path, monkeypatch):
    """Siril's rmgreen acts on the whole frame, lightness preserved, so on a neutral sky the green it removes comes back as magenta: the
    subject's correction is blended in through the range mask, and the sky keeps its colour."""
    calls: list[str] = []
    _rmgreen(monkeypatch, calls)
    fin = pro.Finisher(str(tmp_path), {}, {**post.DEFAULTS, "scnr": 0.6}, False, external=False)
    arr = _neutral_sky_with_green_nebula(0.03)
    out = fin.colour(arr)
    rec = fin.records[-1]
    assert rec["tool"] == "siril rmgreen + pm (range mask)" and rec["status"] == "done" and rec["params"]["amount"] == 0.6, rec
    assert calls == ["rmgreen 3 0.600"]
    after = rec["params"]["sky_chroma_after"]
    assert max(after["a"], after["b"]) < float(gates.LIMITS["sky_chroma"]), rec["params"]


def test_the_stretched_denoiser_runs_as_asked_and_records_the_sky_colour(tmp_path):
    """The denoiser is the user's choice: it runs at the strength asked, its pixels ship, the sky's colour before / after is recorded, and the
    closing step may ask the same tool again."""
    fin = pro.Finisher(str(tmp_path), {}, {**post.DEFAULTS}, False, external=False)
    arr = _denoise_candidate()
    calls: list[float] = []
    out = fin._keep_denoise(_red_shift_denoiser(calls), arr, 0.7, tool="graxpert denoise", ceiling=0.8, start=time.time(),
                            params=lambda a: {"strength": a}, source="test stand-in")
    rec = fin.records[-1]
    assert rec["status"] == "done" and calls == [0.7] and out is not arr
    assert fin.denoiser[0] == "graxpert denoise" and fin.denoiser[3] == 0.7
    assert "sky_chroma_before" in rec["params"] and "sky_chroma_after" in rec["params"]


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


def test_the_statistical_stretch_runs_once_with_its_own_black_clip(tmp_path, monkeypatch):
    """The script's black clip is the stretch's black point: no clip-free retry substituted on a sky-evenness measure (a stretch
    that keeps the pedestal always measures more even in percent of the sky - it is simply flatter)."""
    calls: list[str] = []

    def siril(self, tag, arr, lines, stage, *, out=None, extra=None):
        calls.append(lines[0])
        a = np.clip(np.nan_to_num(np.asarray(arr, np.float32)), 0.0, 1.0)
        return a if "Statistical_Stretch.py" not in lines[0] else np.clip(a * 4.0, 0.0, 1.0).astype(np.float32)

    monkeypatch.setattr(pro.Finisher, "siril", siril)
    fin = pro.Finisher(str(tmp_path), {}, post.DEFAULTS, False, external=False)
    fin.stretch(_linear_sky_picture())
    stat = [c for c in calls if "Statistical_Stretch.py" in c]
    assert len(stat) == 1 and "-noblackclip" not in stat[0], calls
    rec = next(r for r in fin.records if "Statistical" in str(r.get("tool")))
    assert rec["params"]["black_clip"] is True and fin.info["stretch"]["black_clip"] is True


# --- the sky's channel-to-channel tilt (the sky_chroma_quadrants family) ---------------------------------------------------------------


                                                                                        # capped - applying the field under a blurred sky mask instead would draw the seam
                                                                                        # this file's other tools exist to remove, at the stretch's slope of a hundred




def test_autofinish_measures_the_picture_as_delivered():
    """The last step's levels: after the black point and the sky's midtones the sky sits at SKY_TARGET, and the highlights move up."""
    rng = np.random.default_rng(5)
    img = np.clip(0.2 + 0.01 * rng.standard_normal((3, 200, 200)), 0, 1).astype(np.float32)
    img[:, 80:120, 80:120] = 0.8
    out = af.delivered(img, 0.12)
    sky = float(np.median(finish.luminance(out)[gates.sky_mask(finish.luminance(out))]))
    assert sky == pytest.approx(pro.SKY_TARGET, abs=0.01) and float(out[:, 100, 100].mean()) > (0.8 - 0.12) / 0.88


def test_the_closing_step_lowers_the_sharpening_until_the_delivered_picture_passes_the_clipping_gate(tmp_path, monkeypatch):
    """Clipping that the sharpening adds to the delivered picture: the same tool weaker, the strongest amount that passes, recorded."""
    fin = pro.Finisher(str(tmp_path), {}, {**post.DEFAULTS, "cc_sharpen": "both", "cc_stellar_amount": 0.5}, False, external=True)

    def compose(self, sl, st):
        img = np.full((3, 100, 100), 0.3, np.float32)
        n = int(round(self.p["cc_stellar_amount"] * 40))                    # clipped pixels grow with the amount: 0.2 -> 8 px = 0.08 %
        img[:, :1, :n] = 1.0
        return img

    monkeypatch.setattr(pro.Finisher, "_compose", compose)
    out = fin._close_clipping(None, None, compose(fin, None, None))
    rec = fin.records[-1]
    assert rec["step"] == "clipping gate" and rec["params"]["used"]["cc_stellar_amount"] < 0.5
    assert float((out.max(0) >= 0.995).mean()) <= gates.LIMITS["clipped"]
    assert fin.p["cc_stellar_amount"] == rec["params"]["used"]["cc_stellar_amount"]


def test_the_star_gates_compare_the_picture_with_its_starless_in_the_same_levels(tmp_path):
    """The last step's black point and midtones go onto the starless the star gates measure against, or every star reads colourless."""
    fin = pro.Finisher(str(tmp_path), {}, post.DEFAULTS, False, external=False)
    sl = np.full((3, 8, 8), 0.2, np.float32)
    assert fin.delivered_starless(sl) is sl                              # nothing delivered yet: unchanged
    fin.final_levels = (0.05, 0.4)
    x = (0.2 - 0.05) / 0.95
    want = (0.4 - 1.0) * x / ((2 * 0.4 - 1.0) * x - 0.4)
    assert np.allclose(fin.delivered_starless(sl), want, atol=1e-6)


def test_the_preview_step_memo_reuses_a_step_only_while_the_settings_it_read_are_unchanged(tmp_path):
    """A step is taken from the memo only when every parameter it read has the value it had; its records, info and attributes are replayed and
    its output handed out as a copy (a later step may work in place)."""
    memo = pro.StepMemo()
    calls = []

    def step(fin):
        def fn(a):
            calls.append(1)
            fin.info["seen"] = fin.p.get("saturation")
            fin.final_levels = (0.0, 0.5)
            fin.record("step", "test", None, status="skipped")
            return a * float(fin.p["saturation"])
        return fn

    base = np.ones((3, 4, 4), np.float32)
    f1 = pro.Finisher(str(tmp_path), {}, {"saturation": 2.0, "contrast": 4}, False, external=False, keep_pictures=False, memo=memo)
    out1, k1 = f1._step("s", step(f1), base, "in")
    f2 = pro.Finisher(str(tmp_path), {}, {"saturation": 2.0, "contrast": 9}, False, external=False, keep_pictures=False, memo=memo)
    f2.record("first", "test", None, status="skipped")
    out2, k2 = f2._step("s", step(f2), base, "in")
    assert len(calls) == 1 and k1 == k2 and np.array_equal(out1, out2) and out1 is not out2      # contrast was not read: a hit
    assert f2.info["seen"] == 2.0 and f2.final_levels == (0.0, 0.5) and [r["n"] for r in f2.records] == [1, 2]
    out2 *= 0
    f3 = pro.Finisher(str(tmp_path), {}, {"saturation": 2.0}, False, external=False, keep_pictures=False, memo=memo)
    assert np.array_equal(f3._step("s", step(f3), base, "in")[0], out1)                         # the cached copy was not touched
    f4 = pro.Finisher(str(tmp_path), {}, {"saturation": 3.0}, False, external=False, keep_pictures=False, memo=memo)
    out4, k4 = f4._step("s", step(f4), base, "in")
    assert len(calls) == 2 and k4 != k1 and np.allclose(out4, 3.0)
    f5 = pro.Finisher(str(tmp_path), {}, {"saturation": 2.0}, False, external=False, keep_pictures=False, memo=memo)
    f5._step("s", step(f5), base, "other input")
    assert len(calls) == 3


def test_a_step_that_iterates_the_settings_depends_on_all_of_them(tmp_path):
    memo = pro.StepMemo()
    calls = []

    def fn_of(fin):
        def fn(a):
            calls.append(1)
            return a + len({**fin.p})
        return fn

    base = np.zeros((3, 2, 2), np.float32)
    for p in ({"a": 1, "b": 2}, {"a": 1, "b": 2}, {"a": 1, "b": 3}):
        fin = pro.Finisher(str(tmp_path), {}, p, False, external=False, keep_pictures=False, memo=memo)
        fin._step("s", fn_of(fin), base, "in")
    assert len(calls) == 2


def test_the_step_memo_is_bounded_by_bytes():
    memo = pro.StepMemo(limit_bytes=2 * 48)
    for i in range(3):
        memo.store({"in": "x", "name": str(i), "out": str(i), "reads": {}, "value": np.zeros(12, np.float32), "records": [], "info": {}, "attrs": {}})
    assert list(memo.entries) == ["1", "2"] and memo.size == 96


def test_a_memoised_preview_chain_renders_what_the_full_chain_renders(tmp_path, monkeypatch):
    """Moving one setting re-runs the chain from the first step that reads it; the picture equals the one rendered without the memo."""
    ran = []

    def stage(name, key, k):
        def f(self, a, *rest, **kw):
            ran.append(name)
            return (a * k + float(self.p.get(key, 0.0))).astype(np.float32)
        return f

    for i, (name, key) in enumerate((("prepare_linear", "background"), ("stretch_prepared", "stretch"), ("highlights", "hdr"), ("denoise", "luminance_denoise"), ("banding", "none"),
                                     ("colour", "palette_amount"), ("local_contrast", "local_contrast"), ("sharpen", "sharpen"),
                                     ("saturation", "saturation"), ("stars", "star_intensity"))):
        monkeypatch.setattr(pro.Finisher, name, stage(name, key, 0.9 + 0.01 * i))

    def close(self, sl, compose):
        ran.append("sky noise")
        return sl, compose(sl)
    monkeypatch.setattr(pro.Finisher, "_close_sky_noise", close)
    monkeypatch.setattr(pro.Finisher, "_compose", lambda self, sl, st: sl + (0 if st is None else st))
    base, stars = RNG.random((3, 6, 6)).astype(np.float32), RNG.random((3, 6, 6)).astype(np.float32)
    memo = pro.StepMemo()
    p0 = {"stretch": 0.1, "saturation": 1.2, "local_contrast": 0.5, "star_intensity": 1.0, "star_reduction": 0.4}

    def render(p, m):
        return pro.Finisher(str(tmp_path), {}, p, False, external=False, keep_pictures=False, memo=m, measure_steps=False).run(base, stars)[0]

    a = render(p0, memo)
    assert len(ran) == 11
    for change, rerun in (({}, []), ({"saturation": 1.5}, ["saturation", "sky noise"]), ({"star_intensity": 1.3}, ["stars", "sky noise"]),
                          ({"local_contrast": 0.7}, ["local_contrast", "sharpen", "saturation", "sky noise"]),
                          ({"stretch": 0.12}, ["stretch_prepared", "highlights", "denoise", "banding", "colour", "local_contrast", "sharpen",
                                               "saturation", "sky noise"])):
        ran.clear()
        p = {**p0, **change}
        got = render(p, memo)
        assert ran == rerun, change
        ran.clear()
        assert np.array_equal(got, render(p, None))
    assert np.array_equal(a, render(p0, None))
