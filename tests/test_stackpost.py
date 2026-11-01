"""stackpost.py: coverage crop, HDR blend, group placement (synthetic data with known answers)."""
from __future__ import annotations

import numpy as np

from sirilpipe import stackpost as sp


def test_largest_rectangle():
    m = np.zeros((20, 30), bool)
    m[3:15, 5:25] = True
    m[3:6, 5:9] = False                       # notch in a corner
    x0, y0, x1, y1 = sp.largest_rectangle(m)
    assert (x1 - x0) * (y1 - y0) >= 12 * 16 and m[y0:y1, x0:x1].all()


def _shifted_frames(n=30, w=400, h=300, step=(2.0, 1.0), rot=0.0):
    """Homographies (image -> reference) of frames drifting by `step` px per frame."""
    out = []
    for i in range(n):
        out.append([1, 0, step[0] * i, 0, 1, step[1] * i, 0, 0, 1])
    return out


def test_frame_coverage_and_crop_box_matches_common_area():
    hm = _shifted_frames()
    W, H = 400 + 58, 300 + 29
    cov, origin = sp.frame_coverage(hm, 400, 300, 1.0, (W, H), "H", down=2)
    assert cov.max() == len(hm) and cov.min() == 0
    stack = np.ones((3, H, W), np.float32)
    # data mask = union of frames
    valid = np.zeros((H, W), bool)
    for i, h in enumerate(hm):
        valid[int(h[5]):int(h[5]) + 300, int(h[2]):int(h[2]) + 400] = True
    stack[:, ~valid] = 0
    x, y, w, hh = sp.crop_box(valid, cov, 1.0, len(hm), down=2, margin=0)
    assert abs(w - (400 - 58)) <= 4 and abs(hh - (300 - 29)) <= 4   # threshold 1.0 = the area every frame covers
    x2, y2, w2, h2 = sp.crop_box(valid, cov, 0.3, len(hm), down=2, margin=0)
    assert w2 * h2 > w * hh                              # a lower threshold keeps more of the field


def test_auto_crop_validates_geometry_against_data_and_falls_back():
    hm = _shifted_frames()
    W, H = 458, 329
    valid = np.zeros((H, W), bool)
    for h in hm:
        valid[int(h[5]):int(h[5]) + 300, int(h[2]):int(h[2]) + 400] = True
    stack = np.where(valid, 0.5, 0.0).astype(np.float32)[None].repeat(3, 0)
    box, info = sp.auto_crop(stack, hm, (400, 300), 1.0, 0.3)
    assert info["method"] == "frame coverage" and info["iou"] > 0.97
    box2, info2 = sp.auto_crop(stack, [[1, 0, 100, 0, 1, 100, 0, 0, 1]] * 3, (400, 300), 1.0, 0.3)   # nonsense geometry
    assert info2["method"] == "data mask" and "did not match" in info2["note"]
    assert box2[2] > 300 and box2[3] > 200


def test_hdr_blend_uses_short_where_long_saturates_and_long_elsewhere():
    rng = np.random.default_rng(1)
    H, W = 200, 300
    truth = np.full((H, W), 20.0, np.float32)           # radiance ADU/s: faint nebula
    truth[80:120, 130:170] = 4000.0                     # bright core
    t_s, t_l = 15.0, 60.0
    sat = 60000.0

    def stack(t, n):
        sig = truth * t
        noisy = rng.normal(sig, np.sqrt(30.0**2 + 0.5 * np.clip(sig, 0, None)) / np.sqrt(n))
        return np.minimum(noisy, sat)
    short = np.stack([stack(t_s, 60)] * 3) / 65535.0
    long_ = np.stack([stack(t_l, 60)] * 3) / 65535.0
    long_[:, 80:120, 130:170] = sat / 65535.0           # clipped plateau (core saturated at 60 s)
    long_[:, 0:5, 0:30] = sat / 65535.0
    valid = np.ones((H, W), bool)
    out, rep = sp.hdr_blend([short.astype(np.float32), long_.astype(np.float32)], [t_s, t_l], [valid, valid])
    assert out.shape == short.shape and 0 <= out.min() and out.max() <= 1.0 + 1e-6
    # in units of the longest exposure: truth * 60 / max
    scale = out[1, 100, 150] / (4000.0 * t_l / 65535.0)
    nebula = out[1, 20:60, 50:100].mean()
    assert abs(nebula / (20.0 * t_l / 65535.0) / scale - 1.0) < 0.15
    core = out[1, 90:110, 140:160].mean()
    assert abs(core / (4000.0 * t_l / 65535.0) / scale - 1.0) < 0.15      # core from the SHORT stack, scaled
    assert rep["inputs"][1]["saturation"][1] is not None and rep["scale_to_exposure"] == 60.0


def test_hdr_blend_ignores_pixels_outside_a_footprint():
    H, W = 100, 100
    a = np.full((3, H, W), 0.2, np.float32)
    b = np.full((3, H, W), 0.8, np.float32)
    va = np.ones((H, W), bool)
    vb = np.zeros((H, W), bool)
    vb[:, :50] = True
    b[:, :, 50:] = 0
    out, _ = sp.hdr_blend([a, b], [10.0, 20.0], [va, vb])
    right = out[:, :, 60:].mean()
    left = out[:, :, :40].mean()
    assert right > 0 and abs(right / left - 1.0) < 2.0       # no black hole where the second stack has no data


def test_group_placement_paste_and_refine():
    rng = np.random.default_rng(2)
    big = rng.random((3, 300, 400)).astype(np.float32) ** 8
    big = (big * 5).astype(np.float32)
    a = big[:, 20:260, 30:350]
    b = big[:, 25:250, 40:340]                                  # same sky, other footprint
    canvas = (300, 400)
    pa = sp.paste(a, canvas, 30, 20)
    pb_wrong = sp.paste(b, canvas, 41, 24)                      # geometry off by (+1, -1)
    rx, ry = sp.refine_shift(pa, pb_wrong)
    pb = sp.paste(b, canvas, 41 + rx, 24 + ry)
    assert (rx, ry) == (-1, 1)
    m = (pa[1] > 0) & (pb[1] > 0)
    assert np.allclose(pa[1][m], pb[1][m])


def test_equalize_backgrounds():
    img = np.stack([np.full((50, 50), v, np.float32) for v in (0.2, 0.1, 0.05)])
    f = sp.equalize_backgrounds(img, np.ones((50, 50), bool))
    assert np.allclose(img[0], 0.1) and np.allclose(img[2], 0.1) and f[1] == 1.0


def test_noise_model_recovers_poisson_gaussian():
    rng = np.random.default_rng(3)
    lvl = np.tile(np.linspace(0.01, 0.5, 600, dtype=np.float32), (600, 1))
    a_true, b_true = 1e-6, 2e-4
    img = lvl + rng.normal(0, np.sqrt(a_true + b_true * lvl)).astype(np.float32)
    a, b = sp.noise_model(img)
    assert abs(b / b_true - 1) < 0.35 and a < 1e-5


def _sky_with_stars(rng, H, W, n=400, sky=20.0, top=2.3):
    """Radiance (ADU/s) of a flat sky plus Gaussian stars of 1-200 ADU/s peak."""
    img = np.full((H, W), sky, np.float32)
    ys, xs = rng.integers(8, H - 8, n), rng.integers(8, W - 8, n)
    fl = 10 ** rng.uniform(0.0, top, n)
    yy, xx = np.mgrid[-6:7, -6:7]
    prof = np.exp(-(xx**2 + yy**2) / (2 * 1.6**2)).astype(np.float32)
    for y, x, f in zip(ys, xs, fl):
        img[y - 6:y + 7, x - 6:x + 7] += f * prof
    return img


def test_hdr_blend_calibrates_pedestal_and_gain_so_single_group_areas_show_no_step():
    """A leftover pedestal / gain error in the short stack must not show as a step where only one group has data."""
    rng = np.random.default_rng(5)
    H, W = 240, 400
    truth = _sky_with_stars(rng, H, W)
    t_s, t_l = 15.0, 60.0
    long_ = (truth * t_l + rng.normal(0, 8, truth.shape)) / 65535.0
    short = (0.93 * truth * t_s + 9.0 + rng.normal(0, 4, truth.shape)) / 65535.0           # gain error + pedestal
    v_long = np.zeros((H, W), bool)
    v_long[:, :260] = True                                                                  # long covers the left
    v_short = np.zeros((H, W), bool)
    v_short[:, 140:] = True                                                                 # short covers the right
    long_ = np.where(v_long, long_, 0).astype(np.float32)
    short = np.where(v_short, short, 0).astype(np.float32)
    args = ([np.stack([short] * 3), np.stack([long_] * 3)], [t_s, t_l], [v_short, v_long])
    out, rep = sp.hdr_blend(*args)
    g, o, n = rep["inputs"][0]["gain_offset"][1]
    assert abs(g - 0.93) < 0.06 and abs(o - 9.0 / 65535.0 / t_s) < 4e-5 and n > 20000
    left, right = np.median(out[1, :, 20:100]), np.median(out[1, :, 300:380])               # single-group sky
    assert abs(left / right - 1.0) < 0.02
    out2, _ = sp.hdr_blend(*args, calibrate=False)
    assert abs(np.median(out2[1, :, 20:100]) / np.median(out2[1, :, 300:380]) - 1.0) > 0.025   # a step without it


def test_saturation_level_finds_the_stack_plateau_not_the_full_scale():
    rng = np.random.default_rng(7)
    img = rng.normal(0.003, 0.0005, (400, 400)).astype(np.float32)
    img[100:104, 100:150] = rng.normal(0.0715, 0.0004, (4, 50))         # clipped cores of a normalised stack
    img[10, 10] = 0.0732                                              # a hot pixel above the plateau
    assert abs(sp.saturation_level(img) - 0.0715) < 5e-4
    img2 = rng.normal(0.003, 0.0005, (400, 400)).astype(np.float32)
    img2[50, 50:60] = np.linspace(0.02, 0.05, 10)
    assert sp.saturation_level(img2) is None                          # bright but unclipped


def test_star_gain_is_psf_independent():
    rng = np.random.default_rng(8)
    H, W = 400, 500
    yy, xx = np.mgrid[-14:15, -14:15]

    def render(sig, gain, pos, fl):
        im = np.full((H, W), 5.0, np.float32)
        for (y, x), f in zip(pos, fl):
            im[y - 14:y + 15, x - 14:x + 15] += gain * f * np.exp(-(xx**2 + yy**2) / (2 * sig**2)) / (2 * np.pi * sig**2)
        return im + rng.normal(0, 0.05, im.shape).astype(np.float32)
    pos = [(int(y), int(x)) for y, x in zip(rng.integers(20, H - 20, 60), rng.integers(20, W - 20, 60))]
    fl = 10 ** rng.uniform(2.5, 3.5, 60)
    ref = render(2.2, 1.0, pos, fl)                                   # broad PSF (long night)
    img = render(1.4, 1.17, pos, fl)                                  # sharp PSF, gain 1.17
    g, n = sp.star_gain(ref, img, np.ones((H, W), bool), radius=9.0)
    assert abs(g - 1.17) < 0.02 and n >= 8


def test_hdr_blend_core_is_not_hollow_when_both_groups_clip_and_gain_differs():
    """Stars clipped in the long stack, clipped later in the short one (whose clip level is 1.15x the long one's in
    stack units: another normalisation): the blend must peak at the star centre."""
    rng = np.random.default_rng(9)
    H, W = 300, 300
    yy, xx = np.mgrid[0:H, 0:W]
    truth = np.full((H, W), 5.0, np.float32)
    for y, x, f in [(80, 80, 30000.0), (200, 150, 60000.0), (120, 230, 600.0), (230, 60, 1500.0)] + [
            (int(a), int(b), 40.0) for a, b in zip(rng.integers(10, 290, 80), rng.integers(10, 290, 80))]:
        truth += f * np.exp(-((yy - y)**2 + (xx - x)**2) / (2 * 1.8**2)) / (2 * np.pi * 1.8**2)
    t_s, t_l = 15.0, 60.0
    sat_l, sat_s = 0.06, 0.069
    long_ = np.minimum((truth * t_l / 65535.0 + rng.normal(0, 1e-5, truth.shape)) * 1.0, sat_l).astype(np.float32)
    short = np.minimum((1.15 * truth * t_s / 65535.0 + rng.normal(0, 1e-5, truth.shape)), sat_s).astype(np.float32)
    long_[long_ > 0.0535] = sat_l * np.where(long_[long_ > 0.0535] > 0.0535, rng.uniform(0.985, 1.0, long_[long_ > 0.0535].shape), 1)   # partial clip in the stack
    valid = np.ones((H, W), bool)
    out, rep = sp.hdr_blend([np.stack([short] * 3), np.stack([long_] * 3)], [t_s, t_l], [valid, valid], radius=8.0)
    assert abs(rep["inputs"][0]["gain_offset"][1][0] - 1.15) < 0.05
    assert abs(rep["inputs"][1]["saturation"][1] - sat_l) < 0.004 and abs(rep["inputs"][0]["saturation"][1] - sat_s) < 0.004
    for y, x in [(200, 150), (80, 80)]:
        prof = [out[1][(np.hypot(yy - y, xx - x) >= r - .5) & (np.hypot(yy - y, xx - x) < r + .5)].mean() for r in range(0, 8)]
        assert all(prof[i] >= prof[i + 1] * 0.97 for i in range(7)), prof     # monotone: no donut


def test_auto_crop_with_groups_keeps_only_area_every_group_covers():
    W, H = 400, 300
    hm_a = [[1, 0, 2.0 * i, 0, 1, 0, 0, 0, 1] for i in range(10)]            # group A drifts in x (0..18)
    hm_b = [[1, 0, 60.0 + 2.0 * i, 0, 1, 0, 0, 0, 1] for i in range(10)]     # group B starts 60 px further
    hm = hm_a + hm_b
    cw = int(W + 78)
    stack = np.ones((3, H, cw), np.float32)
    box, info = sp.auto_crop(stack, hm, (W, H), 1.0, 0.3, [hm_a, hm_b])
    box0, _ = sp.auto_crop(stack, hm, (W, H), 1.0, 0.3)
    assert info.get("groups_all_cover") and box[0] >= 60 - 4
    assert box0[0] < box[0]                                                   # the plain crop keeps the A-only strip


def _stars_img(shape, pos, fl, sig, shift=(0.0, 0.0), sky=5.0):
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    im = np.full(shape, sky, np.float32)
    for (y, x), f in zip(pos, fl):
        im += f * np.exp(-((yy - y - shift[1])**2 + (xx - x - shift[0])**2) / (2 * sig**2)) / (2 * np.pi * sig**2)
    return im


def test_psf_match_and_subpixel_alignment_make_groups_agree():
    rng = np.random.default_rng(11)
    H, W = 320, 420
    pos = [(int(y), int(x)) for y, x in zip(rng.integers(25, H - 25, 90), rng.integers(25, W - 25, 90))]
    fl = 10 ** rng.uniform(2.5, 3.3, 90)
    ref = _stars_img((H, W), pos, fl, 2.6)                                  # broad stars, long night
    img = _stars_img((H, W), pos, fl, 1.5, shift=(-1.3, 0.6))               # sharp, displaced by 1.3 / 0.6 px
    ref3, img3 = np.stack([ref] * 3) / 1e3, np.stack([img] * 3) / 1e3
    v = np.ones((H, W), bool)
    img3, vi, rep = sp.align_to_reference(ref3, img3, v, v, [None] * 3, [None] * 3)
    assert abs(rep[1]["dx"] + 1.3) < 0.1 and abs(rep[1]["dy"] - 0.6) < 0.1
    img3, vi, sig = sp.match_psf(ref3, img3, v, vi, [None] * 3, [None] * 3)
    assert abs(sig[1] - np.sqrt(2.6**2 - 1.5**2)) < 0.4
    m = vi & v
    m[:20] = m[-20:] = False
    m[:, :20] = m[:, -20:] = False
    d = (img3[1] - ref3[1])[m]
    assert np.abs(d).max() < 0.06 * ref3[1].max() and np.median(np.abs(d)) < 1e-3


def test_auto_crop_with_groups_uses_the_threshold_not_every_frame():
    """Two dithered groups: the kept area is where >= 30 % of each group's frames cover, not where every frame of both groups does
    (a fractional coverage with n_ref = 1 collapsed the threshold to 100 %: a two-night IC 1805 came out 855 x 444 px)."""
    W, H = 400, 300
    hm_a = [[1, 0, 4.0 * i, 0, 1, 3.0 * i, 0, 0, 1] for i in range(20)]        # drifts 0..76 px in x, 0..57 in y
    hm_b = [[1, 0, 4.0 * i, 0, 1, 3.0 * i, 0, 0, 1] for i in range(20)]
    cw, ch = W + 80, H + 60
    stack = np.ones((3, ch, cw), np.float32)
    box, info = sp.auto_crop(stack, hm_a + hm_b, (W, H), 1.0, 0.3, [hm_a, hm_b])
    assert info.get("groups_all_cover")
    assert box[2] >= W - 0.3 * 76 - 12 and box[3] >= H - 0.3 * 57 - 12       # at least the 30 %-coverage area, not the 100 % one (W - 76)
