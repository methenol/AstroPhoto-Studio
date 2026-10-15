"""The stretch tools (stretch.py) and the noise-floor GHS fit (finish.ghs_fit_noise_floor); the tests marked ``docker`` run the real tools (siril-cli, the siril-scripts
Python scripts in the pyscript virtualenv) and are part of the image's integration run."""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from sirilpipe import finish, stretch


def _scene(seed=3, h=240, w=320):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    neb = 0.35 * np.exp(-(((xx - 160) / 40.0) ** 2 + ((yy - 120) / 30.0) ** 2))
    return np.clip(0.02 + 0.5 * neb * 0.1 + rng.normal(0, 0.0005, (h, w)), 0, 1).astype(np.float32)


def test_ghs_slope_family_is_the_ghs_harmonic_family():
    # slope (1 + b D t)^(-1 - 1/b) from SP: monotone decreasing both sides, normalised 0 -> 0 and 1 -> 1, LP / HP linear continuations
    x = np.linspace(0, 1, 2001)
    y = finish.ghs(x, 6.0, 20.0, 0.1, lp=0.02, hp=0.8)
    assert y[0] == pytest.approx(0.0, abs=1e-6) and y[-1] == pytest.approx(1.0, abs=1e-6) and (np.diff(y) > 0).all()
    s = np.diff(y) / np.diff(x)
    assert s[int(0.1 * 2000)] > s[int(0.5 * 2000)] > 0 and s[int(0.02 * 2000)] > s[int(0.01 * 2000)] - 1e-3    # slope peaks at SP, falls on both sides, constant below LP
    assert np.allclose(s[: int(0.019 * 2000)], s[0], rtol=1e-3) and np.allclose(s[int(0.81 * 2000):], s[-1], rtol=1e-3)


def test_noise_floor_fit_holds_the_sky_noise_within_the_limit_or_lowers_the_aim():
    sky, neb, bright, band = 0.0044, 0.0607, 0.2255, 0.0006
    fit = finish.ghs_fit_noise_floor(sky, neb, bright, band, 0.0425, (0.6, 0.5, 0.4, 0.35), 13.0, finish.SKY_NOISE_LIMIT, focus=0.03)
    assert fit["neb_aim"] in (0.6, 0.5, 0.4, 0.35) and fit["error"] <= finish.GHS_FIT_TOLERANCE * 3
    assert fit["noise_out"] <= finish.SKY_NOISE_LIMIT and fit["noise_limit_met"]
    assert fit["sp"] >= sky + finish.SP_ABOVE_NOISE * band - 1e-9                                        # SP above the noise it must not amplify
    # an impossible limit: the faintest aim and the lowest-noise curve, flagged
    hard = finish.ghs_fit_noise_floor(sky, neb, bright, band, 0.0425, (0.6, 0.35), 13.0, 1e-5, focus=0.03)
    assert hard["noise_limit_met"] is False and hard["neb_aim"] == 0.35


def test_large_scale_denoise_removes_sky_mottle_and_keeps_the_nebula():
    rng = np.random.default_rng(4)
    h, w = 512, 768
    sky = 0.02 + cv2_blur(rng.normal(0, 1, (h, w)).astype(np.float32), 20.0) * 4e-4
    yy, xx = np.mgrid[0:h, 0:w]
    neb = 4e-3 * np.exp(-(((xx - 500) / 60.0) ** 2 + ((yy - 250) / 50.0) ** 2))
    img = (sky + neb).astype(np.float32)
    out, info = finish.large_scale_denoise(img, 1.0)
    assert info["band_after"] < 0.6 * info["band_before"]
    assert abs(float(out[250, 500]) - float(img[250, 500])) < 0.35 * float(neb[250, 500])             # the nebula loses only the threshold


def cv2_blur(a, s):
    import cv2
    return cv2.GaussianBlur(a, (0, 0), s) * s * 3.0


def test_chroma_wavelet_denoise_removes_sky_blotches_not_the_nebula_hue():
    rng = np.random.default_rng(5)
    h, w = 384, 512
    g = np.full((h, w), 0.08, np.float32)
    blot = cv2_blur(rng.normal(0, 1, (h, w)).astype(np.float32), 12.0) * 0.004
    rgb = np.stack([g + blot, g - 0.5 * blot, g - 0.5 * blot]).astype(np.float32)
    rgb[0, 150:250, 300:420] += 0.5                                                                   # a red nebula
    out, info = finish.chroma_wavelet_denoise(rgb, 0.8)
    assert info["band_ab_after"] < 0.7 * info["band_ab_before"]
    assert out[0, 200, 360] - out[1, 200, 360] > 0.8 * (rgb[0, 200, 360] - rgb[1, 200, 360])         # the hue and saturation of the red region survive


def test_ratio_colour_keeps_a_core_colour_through_the_stretch_and_the_sky_neutral():
    h, w = 300, 400
    yy, xx = np.mgrid[0:h, 0:w]
    core = np.exp(-(((xx - 200) / 40.0) ** 2 + ((yy - 150) / 40.0) ** 2)).astype(np.float32)
    u_h = np.clip(0.02 + 0.8 * core, 0, 1).astype(np.float32)
    u_o = np.clip(0.02 + 0.25 * 0.8 * core, 0, 1).astype(np.float32)                                    # OIII = 0.25 x Ha in the core
    col, info, light = finish.ratio_colour(u_h, u_o, 1e-3, 0.0, "hoo", 0.0)
    assert info["oiii_to_ha"] > 0 and col.max(0).min() > 0.999
    assert col[0, 150, 200] > col[1, 150, 200] * 1.5                                                    # red core, not white
    assert abs(col[0, 5, 5] - col[1, 5, 5]) < 0.05                                                      # empty sky: grey


@pytest.mark.docker
def test_numpy_ghs_mirror_equals_siril_ght(tmp_path):
    if not stretch.siril_available():
        pytest.skip("no siril-cli")
    x = np.concatenate([np.linspace(0, 0.1, 500), np.linspace(0.1, 1, 500)]).astype(np.float32)
    plane = np.tile(x[None, :], (4, 1))
    st = stretch.SirilStretcher(str(tmp_path))
    for D, b, sp, lp, hp in ((2.0, 6.0, 0.1, 0.0, 1.0), (3.0, 10.0, 0.05, 0.02, 1.0), (4.0, 3.0, 0.3, 0.1, 0.8), (1.5, 0.0, 0.2, 0.0, 1.0)):
        fit = {"D": float(np.expm1(D)), "b": b, "sp": sp, "lp": lp, "hp": hp}
        out = st._run([plane], [[st.ght_line(fit)]], "mirror")[0][0]
        assert np.abs(out - finish.ghs(plane, b, fit["D"], sp, lp, hp)).max() < 2e-6


@pytest.mark.docker
@pytest.mark.parametrize("method", ["ghs", "autoghs", "modasinh", "autostretch", "statstretch", "hms"])
def test_every_stretch_tool_runs_headless_and_lands_the_sky(tmp_path, method):
    if not stretch.siril_available():
        pytest.skip("no siril-cli")
    if method in ("statstretch", "hms") and not (stretch.community_python() and stretch.community_script("processing/Statistical_Stretch.py")):
        pytest.skip("pyscript virtualenv / siril-scripts clone missing")
    img = _scene()
    u = np.clip((img - 0.0195) / 0.01 + 0.02, 0, 1).astype(np.float32)
    ctx = {"method": method, "sky": 0.02, "noise": 0.002, "band": 0.001, "neb": float(np.percentile(u, 90)), "bright": float(np.percentile(u, 99.5)), "focus": 0.05,
           "lum": u, "sky_aim": 0.0425, "neb_aims": (0.5, 0.4), "b_max": 13.0, "hdr": 0.0, "boost": 0.0}
    st = stretch.SirilStretcher(str(tmp_path))
    out, rec = st.stretch([u], ctx)
    assert rec["method"] == method and not rec.get("fallback"), rec
    assert 0.0 <= out[0].min() and out[0].max() <= 1.0 and rec["commands"]
    assert float(np.median(out[0])) > float(np.median(u))                                                # a stretch brightens
