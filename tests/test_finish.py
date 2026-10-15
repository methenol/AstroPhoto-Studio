"""Unit tests of sirilpipe.finish (stretch, palettes, stars, screen blend, numpy twins of Siril commands) and of the
export writers / render chain of sirilpipe.post, all on synthetic arrays."""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pytest

from sirilpipe import export, finish, fitsio, post

RNG = np.random.default_rng(7)


def synthetic_linear(h=240, w=320, sky=(0.02, 0.022, 0.025), noise=0.002, blob=0.08, nstars=25, seed=1):
    """A linear RGB frame: sky pedestal, a warm nebula blob, Gaussian stars."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.empty((3, h, w), np.float32)
    neb = np.exp(-(((xx - w * 0.45) / (w * 0.2)) ** 2 + ((yy - h * 0.5) / (h * 0.15)) ** 2))
    for c, col in enumerate((1.0, 0.45, 0.35)):
        img[c] = sky[c] + blob * col * neb + rng.normal(0, noise, (h, w))
    stars = np.zeros((h, w), np.float32)
    for _ in range(nstars):
        x, y, a = rng.uniform(5, w - 5), rng.uniform(5, h - 5), rng.uniform(0.05, 0.8)
        stars += a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.4 ** 2))
    for c, col in enumerate((1.0, 0.9, 0.8)):
        img[c] += stars * col
    return np.clip(img, 0, 1).astype(np.float32)


# ----------------------------------------------------------------------------- the stretch

@pytest.mark.parametrize("position", [0.0, 0.5, 1.0, 1.5, 2.0])
def test_curve_is_monotone_and_hits_the_target_at_the_median(position):
    m, t = 0.004, 0.16
    cur = finish.Curve(m, t, position)
    x = np.linspace(0, 1, 5001, dtype=np.float32)
    y = cur(x)
    assert abs(float(cur(np.array([m], np.float32))[0]) - t) < 2e-3
    assert np.all(np.diff(y) >= -1e-6)
    assert y[0] == pytest.approx(0, abs=1e-6) and y[-1] == pytest.approx(1, abs=1e-4)


def test_more_hdr_compresses_the_highlights_but_not_the_sky():
    m, t = 0.003, 0.16
    lo, hi = finish.Curve(m, t, 0.0), finish.Curve(m, t, 1.5)
    bright = np.array([0.2, 0.5], np.float32)
    assert np.all(hi(bright) < lo(bright))                       # more room in the cores
    sky = np.array([m], np.float32)
    assert abs(float(hi(sky)[0]) - float(lo(sky)[0])) < 3e-3


def test_rational_matches_statistical_stretch_formula():
    x = np.array([0.0, 0.01, 0.5, 1.0], np.float32)
    m, t = 0.02, 0.25
    expect = (m - 1) * t * x / (m * (t + x - 1) - t * x)
    assert np.allclose(finish.rational(x, m, t), expect, atol=1e-6)


def test_black_point_follows_statistical_stretch_rule():
    s = (0.1 + RNG.normal(0, 0.01, 100_000)).astype(np.float32)
    bp = finish.black_point(s, 5.0)
    med = float(np.median(s))
    lo = s[s <= med]
    noise = 1.4826 * float(np.median(np.abs(lo - np.median(lo))))         # Statistical_Stretch: lower-half MAD
    assert bp == pytest.approx(med - 5.0 * noise, abs=1e-6)
    assert 0.05 < bp < 0.1
    assert finish.black_point(np.full(1000, 0.995, np.float32), 5.0) <= 0.99


def test_stretch_rgb_median_target_neutral_sky_and_range():
    lin = synthetic_linear()
    out, info = finish.stretch_rgb(lin, 0.16, position=0.6, color_preservation=0.7)
    assert out.min() >= 0 and out.max() <= 1
    lum = finish.luminance(out)
    assert abs(float(np.median(lum)) - 0.16) < 0.02
    sky = [float(np.median(out[c][:20])) for c in range(3)]            # a corner of pure sky
    assert max(sky) - min(sky) < 0.03                                   # neutral background by construction
    assert info["target"] == 0.16


@pytest.mark.parametrize("cp", [0.0, 0.5, 1.0])
def test_stretch_is_monotone_in_brightness(cp):
    lin = synthetic_linear(noise=0.0)
    out, _ = finish.stretch_rgb(lin, 0.2, position=0.3, color_preservation=cp)
    a = finish.luminance(lin).ravel()
    b = finish.luminance(out).ravel()
    order = np.argsort(a)
    # brighter in -> not darker out (colour ratios may reorder pixels of equal luminance slightly)
    smooth = np.convolve(b[order], np.ones(2001) / 2001, mode="valid")
    assert np.all(np.diff(smooth) > -2e-3)


def test_stretch_channel_target():
    lin = synthetic_linear()
    out, info = finish.stretch_channel(lin[0], 0.2, position=0.0)
    assert abs(float(np.median(out)) - 0.2) < 0.02


def test_curve_position_mapping():
    assert finish.curve_position("ghs", 0.0) == 0.0
    assert finish.curve_position("asinh", 0.6) == pytest.approx(1.6)
    assert finish.curve_position("log", 1.5) == 2.0                      # clamped
    assert finish.curve_position("nonsense", 0.4) == 0.0


def test_adapted_target_small_objects_get_a_darker_sky():
    assert finish.adapted_target(0.16, 0.0, True) == pytest.approx(0.16 * 0.69)
    assert finish.adapted_target(0.16, 0.3, True) == pytest.approx(0.16)
    assert finish.adapted_target(0.16, 0.0, False) == 0.16


# ----------------------------------------------------------------------------- screen blend, stars

def test_screen_combine_is_bounded_and_symmetric():
    a = RNG.random((3, 50, 60), dtype=np.float32)
    b = RNG.random((3, 50, 60), dtype=np.float32)
    s = finish.screen(a, b)
    assert s.min() >= 0 and s.max() <= 1
    assert np.allclose(s, finish.screen(b, a))
    assert np.all(s >= np.maximum(a, b) - 1e-6)                          # never darker than either layer
    assert np.allclose(finish.screen(a, np.zeros_like(a)), a)            # a black layer changes nothing
    assert np.allclose(finish.screen(a, np.ones_like(a)), 1)


def test_unscreen_reconstructs_the_stars():
    starless = RNG.random((3, 40, 40), dtype=np.float32) * 0.6
    stars = RNG.random((3, 40, 40), dtype=np.float32) * 0.5
    full = finish.screen(starless, stars)
    assert np.allclose(finish.unscreen(full, starless), stars, atol=1e-4)


def test_star_stretch_is_the_seti_rational_and_bounded():
    lin = synthetic_linear(noise=0.0)
    g = np.maximum(lin[1] - 0.02, 0)
    stars = np.stack([0.3 * g * (g > 0.004)] * 3)                     # grey stars: the saturated-core whitening changes nothing
    out, info = finish.stretch_stars(stars, color_preservation=0.0, halo_suppress=0.0, amount=5.0)
    assert out.min() >= 0 and out.max() <= 1 and info["amount"] == 5.0
    x = np.clip(stars - info["black_point"], 0, None)
    k = 3.0 ** 5.0
    assert np.allclose(out, np.clip(k * x / ((k - 1) * x + 1), 0, 1), atol=1e-3)       # Seti Astro Star Stretch: k x / ((k - 1) x + 1)
    weak, _ = finish.stretch_stars(stars, halo_suppress=0.0, amount=2.0)
    assert weak.mean() < out.mean()


def test_star_stretch_amount_sends_the_median_star_to_the_aim():
    stars = np.zeros((3, 300, 300), np.float32)
    rng = np.random.default_rng(3)
    peaks = 10 ** rng.uniform(-3.3, -1.0, 60)
    for i, a in enumerate(peaks):
        stars[:, 20 + 4 * (i // 15) * 20 // 4 + 0, 10 + (i % 15) * 19] = a
    stars = np.stack([cv2.GaussianBlur(stars[c], (0, 0), 1.0) * 6.0 for c in range(3)])
    amount, info = finish.star_stretch_amount(stars, 1e-5)
    assert finish.STAR_STRETCH_RANGE[0] <= amount <= finish.STAR_STRETCH_RANGE[1]
    assert info["median_after"] == pytest.approx(finish.STAR_MEDIAN_AIM, abs=0.03)
    assert finish.star_stretch_amount(stars, 0.0) == (finish.STAR_STRETCH, {"stars": 0, "median_peak": None})   # no noise reference: the script's default


def test_render_stars_take_the_star_stretch_setting():
    lin = synthetic_linear(noise=0.001, nstars=15)
    starless = np.minimum(lin, np.float32(0.03))
    stars = np.maximum(lin - starless, 0)
    p = {**post.DEFAULTS, "star_reduction": 0.0, "star_stretch": 3.5}
    _, info = finish.render_layers(starless, stars, p, dual_band=False)
    assert info["stars"]["amount"] == 3.5


def test_core_chroma_keeps_lightness_and_takes_colour_in_the_highlights():
    main = np.zeros((3, 64, 64), np.float32) + np.array([0.9, 0.9, 0.9], np.float32)[:, None, None]
    main[:, :, :32] = 0.1
    alt = np.zeros_like(main) + np.array([0.7, 0.3, 0.3], np.float32)[:, None, None]
    out, info = finish.core_chroma(main, alt)
    assert np.allclose(finish.luminance(out)[:, 50:], finish.luminance(main)[:, 50:], atol=0.03)   # lightness of the main stretch
    assert (out[0, :, 50:] - out[1, :, 50:]).mean() > 0.1                                           # the red of the lower stretch arrives
    assert np.allclose(out[:, :, :10], main[:, :, :10])                                              # dark regions untouched


def test_count_stars_uses_the_sky_noise():
    stars = np.zeros((3, 200, 200), np.float32)
    for x, y in ((30, 40), (100, 100), (150, 60)):
        stars[:, y, x] = 0.2
    stars = np.stack([cv2.GaussianBlur(stars[c], (0, 0), 1.5) for c in range(3)])
    n, per_mp = finish.count_stars(stars, 0.001)
    assert n == 3 and per_mp == pytest.approx(3 / 0.04)
    assert finish.count_stars(stars, 0.0) == (0, 0.0)
    assert finish.count_stars(stars, 0.05)[0] == 0                          # noisier sky: nothing detectable


def test_halo_suppression_tightens_stars():
    lin = synthetic_linear(noise=0.0, nstars=10)
    stars = np.maximum(lin - 0.02, 0)
    loose, _ = finish.stretch_stars(stars, halo_suppress=0.0)
    tight, _ = finish.stretch_stars(stars, halo_suppress=1.0)
    assert (tight > 0.01).sum() < (loose > 0.01).sum()


# ----------------------------------------------------------------------------- palettes

def test_hoo_palette_channel_mapping():
    ha, o = np.full((8, 8), 0.7, np.float32), np.full((8, 8), 0.3, np.float32)
    rgb = finish.compose_palette(ha, o, "hoo")
    assert np.allclose(rgb[0], ha) and np.allclose(rgb[1], o) and np.allclose(rgb[2], o)


def test_hoo_warm_and_foraxx_formulas():
    ha = np.array([[0.2, 0.9]], np.float32)
    o = np.array([[0.8, 0.4]], np.float32)
    w = finish.compose_palette(ha, o, "hoo_warm")
    assert np.allclose(w[0], ha) and np.allclose(w[1], 0.35 * ha + 0.65 * o) and np.allclose(w[2], 0.9 * o + 0.1 * ha)
    f = finish.compose_palette(ha, o, "foraxx")
    p = ha * o
    ho = p ** (1 - p)
    assert np.allclose(f[0], ha) and np.allclose(f[2], o) and np.allclose(f[1], ho * ha + (1 - ho) * o, atol=1e-5)
    # the green of foraxx is a mix of the two lines: between them
    assert np.all(f[1] >= np.minimum(ha, o) - 1e-6) and np.all(f[1] <= np.maximum(ha, o) + 1e-6)


def test_extract_ha_oiii_renormalises_median_and_mad():
    lin = synthetic_linear()
    ha, oiii, info = finish.extract_ha_oiii(lin, unmix=False)
    assert np.allclose(ha, lin[0])
    ho, oo = finish.sample(ha), finish.sample(oiii)
    assert np.median(oo) == pytest.approx(np.median(ho), rel=0.02)
    mad = lambda x: np.median(np.abs(x - np.median(x)))
    assert mad(oo) == pytest.approx(mad(ho), rel=0.05)


def test_ha_leak_is_measured_and_capped():
    h, w = 200, 200
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ha = 0.05 + rng.normal(0, 0.002, (h, w)).astype(np.float32)
    for cx, cy in ((50, 50), (140, 60), (90, 150), (160, 160), (30, 140)):          # a few Ha-only blobs on flat sky
        ha += 0.3 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 12.0 ** 2))
    lin = np.stack([ha, 0.04 + 0.2 * ha, 0.04 + 0.2 * ha]).astype(np.float32)       # OIII channels hold 20 % of Ha
    _, _, info = finish.extract_ha_oiii(lin, unmix=True)
    assert 0.1 <= info["ha_leak_k"] <= finish.MAX_HA_LEAK


def test_dual_band_render_has_no_scnr_and_oiii_boost_works():
    lin = synthetic_linear()
    p = {**post.DEFAULTS, "palette": "hoo", "scnr": 1.0, "saturation": 1.0, "chroma_denoise": 0, "sharpen": 0, "local_contrast": 0,
         "star_reduction": 1.0, "black_point": 0.0}
    img, info = finish.render_layers(lin, None, p, dual_band=True)
    assert info["palette"] == "hoo"
    assert np.allclose(img[1], img[2], atol=1e-3)                          # HOO: G = B
    # SCNR 1.0 on broadband would clamp green to (R+B)/2; on dual-band it is not applied
    p2 = {**p, "oiii_boost": 2.0}
    img2, _ = finish.render_layers(lin, None, p2, dual_band=True)
    assert img2[1].mean() > img[1].mean()


def test_resolve_palette():
    assert finish.resolve_palette("auto", True) == "hoo"
    assert finish.resolve_palette("auto", False) == "natural"
    assert finish.resolve_palette("hoo", True) == "hoo"
    assert finish.resolve_palette("bogus", True) == "natural"
    assert finish.is_narrowband("Duo-Band") and finish.is_narrowband("LP") and not finish.is_narrowband("IRCUT")


# ----------------------------------------------------------------------------- numpy twins of Siril commands

def test_rmgreen_removes_green_excess_only():
    img = np.zeros((3, 4, 4), np.float32)
    img[0], img[1], img[2] = 0.3, 0.6, 0.3
    out = finish.rmgreen_np(img, 1.0)
    assert np.all(out[1] <= out[0] + 1e-3 + 0.2)                            # green pulled towards (R+B)/2
    assert out[1].mean() < img[1].mean()
    magenta = img.copy()
    magenta[1] = 0.1
    assert np.allclose(finish.rmgreen_np(magenta, 1.0)[1] / np.maximum(finish.luminance(finish.rmgreen_np(magenta, 1.0)), 1e-6),
                       magenta[1] / np.maximum(finish.luminance(magenta), 1e-6), atol=0.2)


def test_satu_raises_saturation():
    img = np.stack([np.full((6, 6), v, np.float32) for v in (0.6, 0.4, 0.3)])
    assert finish.mean_saturation(finish.satu_np(img, 0.5)) > finish.mean_saturation(img)
    assert np.allclose(finish.satu_np(img, 0.0), img)


def test_mtf_endpoints_and_direction():
    x = np.linspace(0, 1, 11, dtype=np.float32)[None, None, :].repeat(3, 0)
    y = finish.mtf_np(x, 0.0, 0.25, 1.0)
    assert y.min() == 0 and y.max() == pytest.approx(1.0)
    assert np.all(y >= x - 1e-6)                                            # mid < 0.5 brightens
    assert finish.brightness_to_mid(0.0) == 0.5
    assert finish.brightness_to_mid(0.5) < 0.5 < finish.brightness_to_mid(-0.5)


def test_final_color_numpy_skips_scnr_for_dual_band():
    img = np.stack([np.full((5, 5), v, np.float32) for v in (0.2, 0.7, 0.2)])
    p = {**post.DEFAULTS, "saturation": 1.0, "scnr": 0.8, "black_point": 0.0, "brightness": 0.0}
    assert np.allclose(finish.final_color_numpy(img, p, True), img, atol=1e-5)
    assert finish.final_color_numpy(img, p, False)[1].mean() < 0.7


# ----------------------------------------------------------------------------- detail tools and measurements

def test_detail_tools_keep_shape_and_range():
    img = finish.stretch_rgb(synthetic_linear(), 0.16)[0]
    for fn in (lambda a: finish.chroma_smooth(a, 0.8), lambda a: finish.sharpen(a, 0.5), lambda a: finish.local_contrast(a, 1.0)):
        out = fn(img)
        assert out.shape == img.shape and out.dtype == np.float32 and out.min() >= 0 and out.max() <= 1


def test_chroma_smooth_reduces_colour_noise_not_lightness():
    rng = np.random.default_rng(5)
    base = np.full((3, 200, 200), 0.3, np.float32)
    noisy = np.clip(base + rng.normal(0, 0.03, base.shape), 0, 1).astype(np.float32)
    out = finish.chroma_smooth(noisy, 1.0)
    chroma = lambda a: float(np.std(a[0] - a[1]))
    assert chroma(out) < chroma(noisy) * 0.7


def test_object_coverage_small_vs_large():
    img = synthetic_linear(blob=0.012, nstars=0)
    small = finish.object_coverage(finish.luminance(img))
    wide = synthetic_linear(h=240, w=320, blob=0.1, seed=2)
    yy, xx = np.mgrid[0:240, 0:320].astype(np.float32)
    wide += 0.08 * np.exp(-(((xx - 160) / 140) ** 2 + ((yy - 120) / 100) ** 2))[None] * np.array([1, 0.5, 0.4], np.float32)[:, None, None]
    assert 0 < small < 0.6
    assert finish.object_coverage(finish.luminance(np.clip(wide, 0, 1))) > small


def test_neutralize_background_and_star_balance():
    lin = synthetic_linear(sky=(0.03, 0.02, 0.05))
    out, info = finish.neutralize_background(lin)
    sky = [float(np.median(out[c][:20])) for c in range(3)]
    assert max(sky) - min(sky) < 0.003
    assert info["offsets"][2] > info["offsets"][1]
    rgb = synthetic_linear(sky=(0.02, 0.02, 0.02), nstars=60)
    rgb[0] *= 1.3
    _, det = finish.star_white_balance(rgb)
    assert det["stars"] > 0 and det["gains"][1] == 1.0


# ----------------------------------------------------------------------------- the render chain and the writers

@pytest.fixture()
def session(tmp_path, monkeypatch):
    d = tmp_path / "sess"
    d.mkdir()
    lin = synthetic_linear(h=600, w=800, nstars=80)
    hdr = {"OBJECT": "M 42", "ROWORDER": "BOTTOM-UP", "FILTER": "IRCUT"}
    fitsio.write_rgb(str(d / "stack.fits"), lin, hdr)
    fitsio.write_rgb(str(d / "linear.fits"), lin, hdr)
    json.dump({"object": "M 42", "filter": "IRCUT", "n_frames": 120, "total_exposure": 1800.0, "params": {}},
              open(d / "stack_meta.json", "w"))

    def no_siril(*a, **k):
        raise post.SirilError("no siril in the unit tests")

    monkeypatch.setattr(post, "_siril_image", no_siril)          # the numpy twins take over
    post._PREVIEW_CACHE.clear()
    return str(d)


def test_preview_is_a_jpeg_within_the_size_limit(session):
    b = post.preview_jpeg(session, {"object": "M 42"}, {}, 400)
    assert b[:3] == b"\xff\xd8\xff"
    im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    assert max(im.shape[:2]) <= 400


def test_preview_uses_the_star_layers_when_present(session):
    lin, hdr = fitsio.read_rgb(os.path.join(session, "linear.fits"))
    starless = np.minimum(lin, np.float32(0.03))
    fitsio.write_rgb(os.path.join(session, "starless.fits"), starless, hdr)
    fitsio.write_rgb(os.path.join(session, "stars.fits"), np.maximum(lin - starless, 0), hdr)
    json.dump({"tool": "starnet2", "stage": "linear"}, open(os.path.join(session, "starnet_meta.json"), "w"))
    post._PREVIEW_CACHE.clear()
    layers = post._Layers(session, post.DEFAULTS, 300)
    assert layers.separate and layers.stars is not None
    with_stars = post.preview_jpeg(session, {}, {"star_reduction": 0.0}, 300)
    no_stars = post.preview_jpeg(session, {}, {"star_reduction": 1.0}, 300)
    assert with_stars != no_stars


def test_export_writes_named_files_and_the_sidecar(session):
    files = post.export(session, {"object": "M 42", "n_frames": 120, "total_exposure": 1800.0, "filter": "IRCUT"},
                        {"saturation": 1.2, "crop": False, "grade_tint": 0.3, "mystery": 1}, {"quality": 90, "upscale": 1.5})
    assert set(files) == {"jpg", "tif", "json", "size"}
    assert os.path.basename(files["jpg"]).startswith("M42_") and files["jpg"].endswith(".jpg")
    e = finish.EDGE_CROP
    assert files["size"] == [int(round((800 - 2 * e) * 1.5)), int(round((600 - 2 * e) * 1.5))]   # the stack's rim trimmed, then x 1.5 Lanczos
    side = json.load(open(files["json"]))
    assert set(side) == {"params", "linear_info", "meta", "ignored_params"}
    assert side["params"]["saturation"] == 1.2 and side["params"]["scnr"] == post.DEFAULTS["scnr"]   # {**DEFAULTS, **params}
    assert side["ignored_params"] == ["crop", "grade_tint", "mystery"]
    assert side["meta"]["object"] == "M 42" and side["linear_info"]["upscaled"] == 1.5
    from PIL import Image
    im = Image.open(files["jpg"])
    assert im.size == tuple(files["size"]) and "M 42 | 120 x subs, 30.0 min | AstroPhoto Studio" in im.getexif()[0x010E]
    import tifffile
    t = tifffile.imread(files["tif"])
    assert t.dtype == np.uint16 and t.shape == (files["size"][1], files["size"][0], 3)


def test_export_without_tiff_and_default_name(session):
    files = post.export(session, {}, {}, {"tiff": False})
    assert "tif" not in files and os.path.basename(files["jpg"]).startswith("M42_")      # the object comes from stack_meta.json


def test_export_flips_bottom_up_rows(tmp_path):
    img = np.zeros((3, 10, 6), np.float32)
    img[:, 0, :] = 1.0                                                     # row 0 = the bottom row of a bottom-up file
    up = export.display_image(img, bottom_up=True)
    assert up[-1].min() == 1.0 and up[0].max() == 0.0
    assert export.display_image(img, bottom_up=False)[0].min() == 1.0
    assert export.bottom_up({"ROWORDER": "TOP-DOWN"}) is False and export.bottom_up({}) is True


def test_params_spec_only_lists_known_effective_keys():
    keys = [s["key"] for s in post.PARAM_SPEC]
    assert len(keys) == len(set(keys))
    assert set(keys) <= set(post.DEFAULTS)
    assert not set(keys) & set(post.IGNORED_KEYS)
    for s in post.PARAM_SPEC:
        assert s["type"] in ("bool", "range", "select") and s["group"] in ("Linear", "Stretch", "Colour", "Stars", "Detail", "Finish")
        if s["type"] == "range":
            assert s["min"] <= post.DEFAULTS[s["key"]] <= s["max"]
        if s["type"] == "select":
            assert post.DEFAULTS[s["key"]] in s["options"]
    for preset in post.PRESETS.values():
        assert set(preset) <= set(post.DEFAULTS)


def test_defaults_keep_the_contract_keys():
    d = post.DEFAULTS
    assert d["stretch"] == 0.09 and d["denoise"] == 0.95 and d["deconvolution"] == 0.7 and d["palette"] == "auto"
    assert d["saturation"] == 1.3 and d["scnr"] == 0.4 and d["star_reduction"] == 0.35 and d["star_intensity"] == 0.9
    assert len(d) == 47


def test_spcc_arguments_mapping():
    meta = {"filter": "Duo-Band"}
    args, desc = post.spcc_arguments({"TELESCOP": "DWARF 3", "FILTER": "Duo-Band"}, meta, post.DEFAULTS)
    assert "-oscsensor=Sony IMX678" in args and "-narrowband" in args and "-rwl=656.28" in args and "-gbw=30" in args
    args, _ = post.spcc_arguments({"CREATOR": "ZWO Seestar S50 Pro", "FILTER": "LP"}, {}, post.DEFAULTS)
    assert "-oscsensor=Sony IMX585" in args and "-oscfilter=ZWO Seestar LP" in args and "-narrowband" not in args
    args, _ = post.spcc_arguments({"CREATOR": "ZWO Seestar S50", "FILTER": "IRCUT"}, {}, post.DEFAULTS)
    assert "-oscsensor=ZWO Seestar S50" in args and "-oscfilter=UV/IR Block" in args
    args, _ = post.spcc_arguments({"INSTRUME": "ZWO ASI585MC Pro"}, {}, {**post.DEFAULTS, "spcc_white_ref": "g2v"})
    assert "-oscsensor=Sony IMX585" in args and any(a.startswith("-whiteref=Star") for a in args)
    args, _ = post.spcc_arguments({"TELESCOP": "DWARF 3"}, {}, {**post.DEFAULTS, "spcc_sensor": "Sony IMX999", "spcc_filter": "Baader UV/IR Cut"})
    assert "-oscsensor=Sony IMX999" in args and "-oscfilter=Baader UV/IR Cut" in args
    with pytest.raises(ValueError):
        post.spcc_arguments({"TELESCOP": "Mystery"}, {}, post.DEFAULTS)


def test_resolve_palette_foraxx_only_when_asked():
    assert finish.resolve_palette("foraxx", True) == "foraxx"


def test_star_color_balance_neutralises_the_median_star():
    stars = np.zeros((3, 120, 120), np.float32)
    rng = np.random.default_rng(3)
    for _ in range(40):
        y, x = rng.integers(10, 110, 2)
        stars[:, y, x] = np.array([0.05, 0.1, 0.09]) * rng.uniform(0.5, 3)      # cyan-ish: R at half of G
    stars = np.stack([cv2.GaussianBlur(stars[c], (0, 0), 1.2) for c in range(3)])
    out, info = finish.star_color_balance(stars)
    assert info["gains"][0] > 1.5 and info["gains"][1] == 1.0
    assert out.sum((1, 2))[0] == pytest.approx(out.sum((1, 2))[1], rel=0.2)


def test_reduce_stars_shrinks_and_one_is_starless():
    st = np.clip(np.random.default_rng(1).uniform(0, 1, (3, 20, 20)), 0, 1).astype(np.float32)
    none, g0 = finish.reduce_stars(st, 0.0, 1.0)
    assert np.allclose(none, st) and g0 == 1.0
    half, g = finish.reduce_stars(st, 0.5, 1.0)
    assert half.mean() < st.mean() and half.max() < 1.0 + 1e-6 and 0 < g < 1
    gone, _ = finish.reduce_stars(st, 1.0, 1.0)
    assert gone.max() == 0.0


def test_subtract_star_wings_reassigns_only_the_psf_wing():
    rng = np.random.default_rng(5)
    h = w = 700
    yy, xx = np.mgrid[:h, :w].astype(np.float32)

    def wing(cy, cx, amp):
        r = np.hypot(yy - cy, xx - cx)
        return (amp / (1.0 + (r / 25.0) ** 2.2)).astype(np.float32)

    sky = np.float32(0.02)
    stars_true = np.zeros((3, h, w), np.float32)
    for cy, cx, f in ((200, 200, 1.0), (450, 480, 0.8), (300, 560, 0.1)):
        core = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.5 ** 2)).astype(np.float32)
        stars_true += (f * core)[None]
    wings = sum(wing(cy, cx, 0.02 * f) for cy, cx, f in ((200, 200, 1.0), (450, 480, 0.8), (300, 560, 0.1)))
    starless_in = (sky + wings[None] + rng.normal(0, 5e-5, (3, h, w))).astype(np.float32)   # StarNet-like: wings kept
    lin = starless_in + stars_true
    stars_in = np.maximum(lin - starless_in, 0)
    out, info, stars_out = finish.subtract_star_wings(lin, starless_in, stars_in)
    assert info["stars"] >= 2
    ring = (np.hypot(yy - 200, xx - 200) > 40) & (np.hypot(yy - 200, xx - 200) < 120)
    assert starless_in[1][ring].mean() - sky > 1e-3                      # the wing is there before
    assert abs(out[1][ring].mean() - sky) < 4e-4                         # and mostly gone after
    assert np.allclose(lin, out + stars_out, atol=1e-5)                  # photons reassigned: starless + stars = linear
    assert out.min() >= 0


# ---- second dual-band quality round: highlight shoulder, faint stars -------------------------------------------

def test_hdr_shoulder_is_monotone_keeps_the_sky_and_lowers_the_core():
    x = np.concatenate([[0.0], np.geomspace(1e-8, 1.0, 20000)]).astype(np.float32)
    for m in (1e-3, 1e-2, 0.05):
        plain, full = finish.Curve(m, 0.25, 0.0), finish.Curve(m, 0.25, 0.0, shoulder=1.0)
        y = full(x)
        assert np.diff(y).min() >= -1e-6                                   # HDR_KNEE is the highest knee that stays monotone
        assert float(full(np.array([m], np.float32))[0]) == pytest.approx(0.25, abs=2e-3)   # the sky level does not move
        assert float(full(np.array([0.5], np.float32))[0]) < float(plain(np.array([0.5], np.float32))[0]) - 0.01


def test_curve_shoulder_only_for_the_rational_methods():
    assert finish.curve_shoulder("ghs", 0.6) == pytest.approx(0.5)
    assert finish.curve_shoulder("ghs", 5.0) == 1.0 and finish.curve_shoulder("asinh", 0.6) == 0.0


def test_reduce_stars_keeps_isolated_faint_stars():
    st = np.zeros((3, 80, 120), np.float32)
    st[:, 20, 20] = 0.06                                                    # faint isolated star
    st[:, 50:53, 90:93] = 0.9                                               # bright star with a wing
    st[:, 49:54, 89:94] = np.maximum(st[:, 49:54, 89:94], 0.1)
    out, _ = finish.reduce_stars(st, 0.4, 1.0)
    gain = 1.0 - 0.4 ** 3
    assert out[0, 20, 20] == pytest.approx(0.06 * gain, rel=0.02)           # untouched except for the fade
    assert out[0, 49, 89] < st[0, 49, 89] * gain                            # the bright star's footprint shrank


def test_stretch_stars_faint_star_not_cut_by_a_fixed_floor():
    st = np.zeros((3, 64, 64), np.float32)
    st[:, 32, 32] = 8e-4                                                    # M 27's faint stars peak at ~5e-4 .. 3e-3
    out, info = finish.stretch_stars(st, amount=5.0)
    assert out[:, 32, 32].max() > 0.15 and info["black_point"] < 1e-4


# --------------------------------------------------------------------------- GHS stretch of the starless layer (finish.ghs_*, starlet, nbn, flatten)

def _faint_nebula(h=160, w=240, seed=3):
    """Linear sky 0.02 with pixel noise 3e-4 and a Gaussian nebula of 6e-4 peak (the scale of the DWARF Duo-Band stacks: signal ~ 2 x noise)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    neb = 6e-4 * np.exp(-(((xx - w / 2) / 35) ** 2 + ((yy - h / 2) / 25) ** 2))
    return (0.02 + neb + rng.normal(0, 3e-4, (h, w))).astype(np.float32), neb


def test_starlet_reconstructs_and_denoise_lowers_the_noise_but_keeps_the_nebula():
    x, neb = _faint_nebula()
    planes, resid = finish.starlet(x, 5)
    assert np.allclose(sum(planes) + resid, x, atol=1e-5)
    dn, sig1 = finish.starlet_denoise(x, 1.0)
    assert sig1 == pytest.approx(3e-4, rel=0.25)
    assert (dn - (0.02 + neb)).std() < 0.6 * (x - (0.02 + neb)).std()                   # noise down by > 40 %
    assert dn[80, 120] - dn[10, 10] > 0.5 * 6e-4                                         # the nebula's peak survives
    assert np.array_equal(finish.starlet_denoise(x, 0.0)[0], x)


def test_ghs_is_monotone_normalised_and_steepest_at_its_symmetry_point():
    x = np.linspace(0, 1, 2001)
    for b in (0.5, 1.0, 2.0, 8.0):
        y = finish.ghs(x, b, 30.0, 0.2)
        assert y[0] == pytest.approx(0, abs=1e-6) and y[-1] == pytest.approx(1, abs=1e-6)
        assert (np.diff(y) >= -1e-7).all()
        assert abs(x[np.argmax(np.gradient(y, x))] - 0.2) < 0.01
    y = finish.ghs(np.array([0.2 - 0.1, 0.2, 0.2 + 0.1]), 4.0, 50.0, 0.2)                # odd symmetry about SP
    assert (y[2] - y[1]) == pytest.approx(y[1] - y[0], rel=1e-4)


def test_ghs_fit_meets_both_aims_with_the_highest_feasible_b():
    fit = finish.ghs_fit(0.06, 0.27, 0.085, 0.55, 13.0)
    assert fit["error"] <= finish.GHS_FIT_TOLERANCE
    t = finish.ghs(np.array([0.06, 0.27]), fit["b"], fit["D"], fit["sp"])
    assert t[0] == pytest.approx(0.085, abs=0.01) and t[1] == pytest.approx(0.55, abs=0.01)
    assert fit["b"] in finish.GHS_B_LADDER
    assert finish.ghs_fit(0.06, 0.27, 0.085, 0.55, 0.5)["b"] == 0.5                       # b_max caps the ladder


def test_nbn_oiii_equalises_the_average_level_and_keeps_the_floor():
    rng = np.random.default_rng(4)
    ha = np.clip(rng.normal(0.2, 0.1, (80, 80)), 0, 1).astype(np.float32)
    oiii = np.clip(rng.normal(0.1, 0.03, (80, 80)), 0, 1).astype(np.float32)
    out, e1 = finish.nbn_oiii(ha, oiii, 1.0)
    assert 0.01 <= e1 < 0.5 and out.mean() > oiii.mean()                                   # a weak OIII is lifted
    assert 0.01 < out.mean() - oiii.mean() < ha.mean() - oiii.mean()                       # towards the Ha level, not past it
    low = oiii < np.median(oiii)
    assert np.allclose(out[low], oiii[low], atol=1e-4)                                     # below the median nothing moves
    out2, _ = finish.nbn_oiii(ha, oiii, 0.6)
    assert out2.mean() < out.mean()                                                        # boost < 1 lifts less
    same, e_same = finish.nbn_oiii(ha, ha, 1.0)
    assert e_same == pytest.approx(0.5, abs=0.02) and np.allclose(same, ha, atol=0.02)


def test_flatten_sky_removes_a_tilt_but_not_the_nebula():
    x, neb = _faint_nebula(h=320, w=480)
    tilt = np.linspace(0, 2e-4, x.shape[1], dtype=np.float32)[None, :]
    y, info = finish.flatten_sky(x + tilt)
    sky = np.concatenate([y[:40].ravel(), y[-40:].ravel()])
    left, right = y[:, :60].mean(), y[:, -60:].mean()
    assert abs(left - right) < 0.5e-4                                                      # the 2e-4 tilt is gone
    assert y[160, 240] - np.median(y) > 3e-4                                               # the nebula is still there
    assert info["span"] > 1e-4


def test_ghs_stretch_puts_the_sky_and_the_nebula_on_their_aims_and_shows_the_nebula():
    x, neb = _faint_nebula(h=240, w=360)
    y, info = finish.ghs_stretch([x, x * 0.9], sky_aim=0.085, neb_aim=0.55, b_max=4.0, denoise=1.0)
    assert len(y) == 2 and y[0].min() >= 0 and y[0].max() <= 1
    sky = float(np.median(y[0][:30, :30]))
    # (the synthetic nebula is one bright Gaussian, 59 % of the white point at its 90th percentile: the aims cannot both be met, the denoised plane is clean enough
    # for the fit to see that; the sky then sits lower than the aim, never above the dark grey)
    assert 0.01 < sky < 0.2                                                                # dark sky
    assert y[0][120, 180] > 0.4 and y[0][120, 180] > sky + 0.3                             # the nebula stands out
    assert info["ghs"]["b"] in finish.GHS_B_LADDER and info["sky_in"] < info["nebula_in"]


def test_local_contrast_gates_the_sky_and_boosts_structure():
    rng = np.random.default_rng(5)
    base = np.full((3, 200, 300), 0.08, np.float32) + rng.normal(0, 0.002, (3, 200, 300)).astype(np.float32)
    yy, xx = np.mgrid[0:200, 0:300]
    structure = (0.25 * (np.sin(xx / 7.0) * np.sin(yy / 6.0) > 0.4)).astype(np.float32)
    img = np.clip(base + structure[None] * (xx > 150)[None], 0, 1)                         # right half carries the structure
    out = finish.local_contrast(img, 1.0)
    left, right = (out - img)[:, :, :120], (out - img)[:, :, 180:]
    assert np.abs(left).mean() < 0.35 * np.abs(right).mean()                               # the sky half is left alone
    assert out.std() > img.std() and np.array_equal(finish.local_contrast(img, 0.0), img)


def test_ghs_render_of_a_dual_band_stack_is_dark_grey_sky_and_visible_nebula():
    x, neb = _faint_nebula(h=200, w=300)
    lin = np.stack([x, 0.02 + 0.4 * (x - 0.02), 0.02 + 0.4 * (x - 0.02)]).astype(np.float32)
    p = {**post.DEFAULTS, "stretch": 0.085, "contrast": 4.0, "black_point": 0.0, "saturation": 1.0, "chroma_denoise": 0, "sharpen": 0, "local_contrast": 0,
         "star_reduction": 1.0, "palette": "hoo", "oiii_boost": 1.0}
    img, info = finish.render_layers(lin, None, p, dual_band=True)
    lum = finish.luminance(img)
    assert 0.01 < np.median(lum[:30, :30]) < 0.2 and lum[100, 150] > np.median(lum[:30, :30]) + 0.2
    assert info["palette_info"]["ghs"]["b"] in finish.GHS_B_LADDER and info["palette"] == "hoo"
    assert np.allclose(img[1], img[2], atol=1e-5)                                          # HOO: G = B = OIII


# --------------------------------------------------------------------------- noise-aware strengths, BayesShrink / bivariate denoise, Ha luminance, HDR compress, sky

def _noisy_object(amp, h=300, w=400, noise=3e-4, seed=11):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    neb = amp * np.exp(-(((xx - w / 2) / 60) ** 2 + ((yy - h / 2) / 45) ** 2))
    return (0.02 + neb + rng.normal(0, noise, (h, w))).astype(np.float32)


def test_object_snr_follows_the_signal_and_the_ladders_run_from_noisy_to_clean():
    weak, strong = finish.object_snr(_noisy_object(4e-4)), finish.object_snr(_noisy_object(8e-3))
    assert 0 < weak < finish.SNR_LOW < finish.SNR_HIGH < strong
    assert finish.bxt_sn(weak) == pytest.approx(0.3, abs=0.06) and finish.bxt_sn(strong) == 0.75     # RC-Astro: sharpen less on noisy data
    assert finish.nxt_dn(weak) > 0.85 and finish.nxt_dn(strong) == 0.6                                # NXT 0.6-0.9, noisier = stronger
    assert finish.graxpert_strength(weak) > 0.9 and finish.graxpert_strength(strong) == pytest.approx(0.6)
    assert finish.bxt_sn(0.0) == 0.3 and finish.nxt_dn(0.0) == 0.9
    assert finish.object_snr(np.full((100, 100), 0.02, np.float32)) == 0.0                           # no object, no signal-to-noise


def test_object_snr_ignores_stars():
    x = _noisy_object(4e-4)
    y = x.copy()
    for cy, cx in ((40, 50), (120, 300), (240, 90), (200, 200), (60, 330)):
        y[cy - 3:cy + 4, cx - 3:cx + 4] += 0.3 * np.exp(-(np.mgrid[-3:4, -3:4] ** 2).sum(0) / 3.0)
    assert finish.object_snr(y) == pytest.approx(finish.object_snr(x), rel=0.35)


def test_bivariate_shrink_removes_isolated_noise_outliers_but_keeps_an_extended_structure():
    rng = np.random.default_rng(2)
    x = (0.02 + rng.normal(0, 3e-4, (300, 300))).astype(np.float32)
    yy, xx = np.mgrid[0:300, 0:300]
    structure = 1.2e-3 * np.exp(-(((xx - 150) / 12.0) ** 2 + ((yy - 150) / 12.0) ** 2))      # 4 sigma at the centre, extended
    x += structure.astype(np.float32)
    hot = [(40, 40), (60, 240), (250, 60)]
    for y0, x0 in hot:
        x[y0, x0] += 1.5e-3                                                                      # an isolated 5 sigma pixel
    dn, _ = finish.starlet_denoise(x, 1.0)
    assert all(dn[y0, x0] - 0.02 < 0.6 * 1.5e-3 for y0, x0 in hot)                               # isolated: no support from the coarser scale
    assert dn[150, 150] - 0.02 > 0.5 * 1.2e-3                                                    # extended: kept
    # one noise reduction, not two: with the AI denoiser's flag only the fine scales are touched, the coarse residual (a smooth ramp) is untouched
    ramp = (0.02 + np.linspace(0, 1e-3, 300, dtype=np.float32)[None, :] * np.ones((300, 1), np.float32)).astype(np.float32)
    ramp += rng.normal(0, 3e-4, ramp.shape).astype(np.float32)
    coarse = finish.starlet(finish.starlet_denoise(ramp, 1.0, shrink_scales=3)[0], 6)[1]
    assert abs(float(coarse[:, 250:].mean() - coarse[:, :50].mean()) - 1e-3 * 0.8) < 3e-4


def test_wavelet_contrast_is_snr_gated():
    noise_only = _noisy_object(0.0, seed=5)
    flat = finish.starlet_denoise(noise_only, 1.0, contrast=0.0)[0]
    boosted = finish.starlet_denoise(noise_only, 1.0, contrast=2.0)[0]
    assert boosted.std() < 1.15 * flat.std()                                                      # noise is not amplified
    rng = np.random.default_rng(6)
    yy, xx = np.mgrid[0:300, 0:400]
    pattern = (4e-3 * (np.sin(xx / 6.0) * np.sin(yy / 6.0)) * ((xx > 150) & (xx < 260) & (yy > 100) & (yy < 200))).astype(np.float32)   # 13 sigma structure on the 4-16 px scales
    clean = (0.02 + pattern + rng.normal(0, 3e-4, pattern.shape)).astype(np.float32)
    a, b = finish.starlet_denoise(clean, 1.0, contrast=0.0)[0], finish.starlet_denoise(clean, 1.0, contrast=1.5)[0]
    assert b[100:200, 150:260].std() > 1.25 * a[100:200, 150:260].std()                           # the structure gets its gain


def test_hdr_compress_is_the_statistical_stretch_curve():
    y = np.linspace(0, 1, 11, dtype=np.float32)
    assert np.array_equal(finish.hdr_compress_highlights(y, 0.0), y)
    out = finish.hdr_compress_highlights(y, 0.5, 0.75)
    assert np.allclose(out[y <= 0.75], y[y <= 0.75]) and out[-1] == pytest.approx(1.0)               # below the knee nothing moves, white stays white
    assert (out[(y > 0.75) & (y < 1)] < y[(y > 0.75) & (y < 1)]).all() and (np.diff(out) >= 0).all()   # the cubic Hermite sags below the diagonal in between: room at the top
    t = (0.9 - 0.75) / 0.25
    expect = 0.75 + 0.25 * ((t ** 3 - 2 * t ** 2 + t) + (-2 * t ** 3 + 3 * t ** 2) + (t ** 3 - t ** 2) * (1 + 4 * 0.5))
    assert float(finish.hdr_compress_highlights(np.array([0.9], np.float32), 0.5, 0.75)[0]) == pytest.approx(expect, abs=1e-5)
    rgb = np.stack([np.full((4, 4), v, np.float32) for v in (0.95, 0.9, 0.85)])
    c = finish.hdr_compress(rgb, 1.2)
    assert finish.luminance(c)[0, 0] < finish.luminance(rgb)[0, 0] and c[0, 0, 0] > c[1, 0, 0] > c[2, 0, 0]     # darker, hue order kept
    assert np.array_equal(finish.hdr_compress(rgb, 0.0), rgb)


def test_ha_luminance_takes_the_lightness_from_ha_and_keeps_the_palette_colour():
    rng = np.random.default_rng(8)
    ha = np.clip(rng.uniform(0.1, 0.7, (40, 40)), 0, 1).astype(np.float32)
    o = np.full((40, 40), 0.15, np.float32)                                                       # smooth OIII
    rgb = finish.compose_palette(ha, o, "hoo")
    out = finish.ha_luminance(rgb, ha, 1.0)
    assert out.max() <= 1.0 and np.allclose(out.max(0), ha, atol=1e-4)                          # the lightness (brightest channel) is Ha's, nothing can overflow
    assert np.array_equal(finish.ha_luminance(rgb, ha, 0.0), rgb)
    hue_in = np.arctan2(np.sqrt(3) * (rgb[1] - rgb[2]), 2 * rgb[0] - rgb[1] - rgb[2])
    hue_out = np.arctan2(np.sqrt(3) * (out[1] - out[2]), 2 * out[0] - out[1] - out[2])
    assert np.abs(hue_in - hue_out).max() < 1e-3                                                  # G = B stays G = B: the palette's own hue
    # the OIII noise cannot reach the lightness: noisy OIII, same lightness
    noisy_o = (0.15 + rng.normal(0, 0.05, (40, 40))).clip(0, 1).astype(np.float32)
    light_noisy = finish.ha_luminance(finish.compose_palette(ha, noisy_o, "hoo"), ha, 1.0).max(0)
    assert np.abs(light_noisy - ha).max() < 1e-4


def test_colour_planes_share_a_sky_and_neutralize_display_sky_equalises_the_medians():
    rng = np.random.default_rng(9)
    h = np.clip(0.06 + 0.4 * np.exp(-(((np.mgrid[0:300, 0:400][1] - 200) / 50.0) ** 2)) + rng.normal(0, 0.01, (300, 400)), 0, 1).astype(np.float32)
    o = np.clip(0.11 + rng.normal(0, 0.01, (300, 400)), 0, 1).astype(np.float32)
    hc, oc, d = finish.colour_planes(h, o)
    assert abs(np.percentile(hc, 5) - np.percentile(oc, 5)) < 0.01 and d > 0.03
    assert oc.std() < 0.5 * o.std()                                                               # colour resolution: the OIII grain is averaged away
    rgb = np.stack([np.full((100, 100), 0.05, np.float32), np.full((100, 100), 0.08, np.float32), np.full((100, 100), 0.07, np.float32)])
    rgb[:, 40:60, 40:60] = 0.7
    out, off = finish.neutralize_display_sky(rgb)
    sky = out[:, :30, :30]
    assert np.allclose(sky[0], sky[1]) and np.allclose(sky[1], sky[2]) and off[0] == 0 and off[1] == pytest.approx(0.03, abs=1e-4)


def test_sky_lift_brings_the_sky_up_to_the_aim_and_never_darkens():
    rng = np.random.default_rng(10)
    img = np.clip(rng.normal(0.03, 0.002, (3, 120, 160)), 0, 1).astype(np.float32)
    img[:, 40:80, 60:120] = 0.8
    out, a = finish.sky_lift(img, 0.085)
    lum = cv2.GaussianBlur(finish.luminance(out), (0, 0), 2.0)
    assert abs(float(np.median(lum[lum <= np.percentile(lum, 45)])) - 0.085) < 0.006 and 0.04 < a < 0.07
    assert out.max() <= 1.0 and (out >= img).all()
    bright_sky, a2 = finish.sky_lift(np.full((3, 50, 50), 0.2, np.float32), 0.085)
    assert a2 == 0.0 and np.array_equal(bright_sky, np.full((3, 50, 50), 0.2, np.float32))
