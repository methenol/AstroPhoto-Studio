"""Unit tests of sirilpipe.finish (statistics, ghs mirror, restore-stage helpers) and of the export writers / render plumbing of sirilpipe.post, on synthetic arrays.

The finishing chain itself (Siril commands) is tested in test_pro.py; here ``pro.Finisher.run`` is replaced by a numpy stand-in."""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pytest

from sirilpipe import export, finish, fitsio, post, pro

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


def test_unscreen_reconstructs_the_stars():
    starless = RNG.random((3, 40, 40), dtype=np.float32) * 0.6
    stars = RNG.random((3, 40, 40), dtype=np.float32) * 0.5
    full = finish.screen(starless, stars)
    assert np.allclose(finish.unscreen(full, starless), stars, atol=1e-4)


def test_count_stars_uses_the_sky_noise():
    stars = np.zeros((3, 200, 200), np.float32)
    for x, y in ((30, 40), (100, 100), (150, 60)):
        stars[:, y, x] = 0.2
    stars = np.stack([cv2.GaussianBlur(stars[c], (0, 0), 1.5) for c in range(3)])
    n, per_mp = finish.count_stars(stars, 0.001)
    assert n == 3 and per_mp == pytest.approx(3 / 0.04)
    assert finish.count_stars(stars, 0.0) == (0, 0.0)
    assert finish.count_stars(stars, 0.05)[0] == 0                          # noisier sky: nothing detectable


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

    def stand_in(self, base, stars):
        """Numpy stand-in for the Siril chain: a plain stretch of the starless image, screen-blended stars when the star reduction asks for them."""
        out = np.clip(np.sqrt(np.clip(base, 0, 1)) * 3.0, 0, 1)
        if stars is not None and float(self.p.get("star_reduction", 0.4)) < 1.0:
            out = 1 - (1 - out) * (1 - np.clip(stars * 5, 0, 1))
        return out.astype(np.float32), {"chain": [], "starless": out}

    monkeypatch.setattr(pro.Finisher, "run", stand_in)
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
    assert d["stretch"] == 0.10 and d["denoise"] == 0.95 and d["deconvolution"] == 0.7 and d["palette"] == "auto" and d["stretch_method"] == "ghs2"
    assert d["saturation"] == 1.3 and d["scnr"] == 0.0 and d["star_reduction"] == 0.4 and d["star_intensity"] == 1.0 and d["hdr"] == 0.0
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


def test_ghs_is_monotone_normalised_and_steepest_at_its_symmetry_point():
    x = np.linspace(0, 1, 2001)
    for b in (0.5, 1.0, 2.0, 8.0):
        y = finish.ghs(x, b, 30.0, 0.2)
        assert y[0] == pytest.approx(0, abs=1e-6) and y[-1] == pytest.approx(1, abs=1e-6)
        assert (np.diff(y) >= -1e-7).all()
        assert abs(x[np.argmax(np.gradient(y, x))] - 0.2) < 0.01
    y = finish.ghs(np.array([0.2 - 0.1, 0.2, 0.2 + 0.1]), 4.0, 50.0, 0.2)                # odd symmetry about SP
    assert (y[2] - y[1]) == pytest.approx(y[1] - y[0], rel=1e-4)


def _noisy_object(amp, h=300, w=400, noise=3e-4, seed=11):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    neb = amp * np.exp(-(((xx - w / 2) / 60) ** 2 + ((yy - h / 2) / 45) ** 2))
    return (0.02 + neb + rng.normal(0, noise, (h, w))).astype(np.float32)


def test_object_snr_follows_the_signal_and_the_ladders_run_from_noisy_to_clean():
    weak, strong = finish.object_snr(_noisy_object(4e-4)), finish.object_snr(_noisy_object(8e-3))
    assert 0 < weak < finish.SNR_LOW < finish.SNR_HIGH < strong
    assert finish.bxt_sn(weak) == pytest.approx(0.5, abs=0.02) and finish.bxt_sn(strong) == 0.6        # research/pro-workflow.md A5: sn 0.5-0.6 on 1-2 h stacks, lower on noisy data
    assert 0.25 <= finish.bxt_ss(weak) <= finish.bxt_ss(strong) <= 0.35
    assert finish.nxt_dn(weak) > 0.75 and finish.nxt_dn(strong) == 0.5                                # NXT linear 0.5-0.8, noisier = stronger
    assert finish.graxpert_strength(weak) > 0.75 and finish.graxpert_strength(strong) == pytest.approx(0.6)
    assert finish.bxt_sn(0.0) == 0.5 and finish.nxt_dn(0.0) == 0.8
    assert finish.object_snr(np.full((100, 100), 0.02, np.float32)) == 0.0                           # no object, no signal-to-noise


def test_object_snr_ignores_stars():
    x = _noisy_object(4e-4)
    y = x.copy()
    for cy, cx in ((40, 50), (120, 300), (240, 90), (200, 200), (60, 330)):
        y[cy - 3:cy + 4, cx - 3:cx + 4] += 0.3 * np.exp(-(np.mgrid[-3:4, -3:4] ** 2).sum(0) / 3.0)
    assert finish.object_snr(y) == pytest.approx(finish.object_snr(x), rel=0.35)
