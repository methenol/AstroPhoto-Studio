"""VeraLux ImageMM headless engine (sirilpipe/veralux_imagemm.py): the FFT operators against the script's padded
conv2d, the forward model's orientation, the findstar parser / coordinates and a small multi-frame restoration."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from sirilpipe import veralux_imagemm as V  # noqa: E402


def _script_conv(img, kernel, mode):
    """VeraLux_ImageMM.py ImageMMEngine.spatial_convolve: F.pad + F.conv2d (a cross-correlation)."""
    c = img.shape[1]
    kh, kw = kernel.shape[-2:]
    p = F.pad(img, (kw // 2, kw // 2, kh // 2, kh // 2), mode=mode)
    return F.conv2d(p, kernel.expand(c, 1, kh, kw), padding=0, groups=c)


@pytest.mark.parametrize("mode", ["reflect", "constant"])
def test_fft_operators_equal_the_scripts_padded_convolutions(mode):
    g = torch.Generator().manual_seed(1)
    x = torch.rand((2, 3, 40, 57), generator=g, dtype=torch.float64).float()
    k = torch.rand((9, 9), generator=g).float()
    conv = V.FFTConv(40, 57, 9, mode, "cpu")
    kh = conv.kernel_hat(k)
    # the script's spatial_correlation (conv2d with the flipped kernel) is the true convolution = our forward
    want_conv = _script_conv(x, torch.flip(k, (0, 1))[None, None], mode)
    assert torch.allclose(conv.convolve(conv.rfft(x), kh), want_conv, atol=1e-4)
    # the script's spatial_convolve (conv2d: a correlation) = our adjoint
    want_corr = _script_conv(x, k[None, None], mode)
    assert torch.allclose(conv.back(conv.correlate_hat(x, kh)), want_corr, atol=1e-4)


def test_forward_model_renders_the_measured_star_not_rotated():
    """A point source must come out as the kernel itself; the script's conv2d forward drew it rotated by 180 deg."""
    k = torch.zeros((7, 7))
    k[3, 3], k[3, 5], k[1, 3] = 1.0, 0.5, 0.25                # an asymmetric "PSF": light right of and above centre
    x = torch.zeros((1, 1, 21, 21))
    x[0, 0, 10, 10] = 1.0
    conv = V.FFTConv(21, 21, 7, "constant", "cpu")
    out = conv.convolve(conv.rfft(x), conv.kernel_hat(k))[0, 0]
    assert torch.allclose(out[7:14, 7:14], k, atol=1e-6)
    script = _script_conv(x, k[None, None], "constant")[0, 0]
    assert torch.allclose(script[7:14, 7:14], torch.flip(k, (0, 1)), atol=1e-6)


def test_parse_findstar_and_siril_coordinates(tmp_path):
    p = tmp_path / "s.lst"
    p.write_text("# 2 stars found using the following parameters:\n# sigma=1.00 roundness=0.50\n"
                 "# star#\tlayer\tB\tA\tbeta\tX\tY\tFWHMx [px]\tFWHMy [px]\n"
                 "1\t1\t0.0003\t0.11\t-1.00\t3062.00\t1288.09\t7.30\t5.62\n"
                 "2\t1\t0.0002\t0.07\t-1.00\t424.29\t1369.73\t8.00\t6.11\n")
    assert V.parse_findstar(str(p)) == [(3062.0, 1288.09), (424.29, 1369.73)]
    assert V.parse_findstar(str(tmp_path / "missing.lst")) == []
    assert V.siril_to_array(10.5, 90.5, 100) == (10.0, 9.0)


def test_sr_origin_matches_solve_sr_centring():
    """A kernel solved by solve_sr reproduces the native PSF through D; the convolution centres it on hr // 2:
    sr_origin is where frame pixel 0 sits on the latent grid."""
    assert V.sr_origin(1) == 0 and V.sr_origin(2) == 1 and V.sr_origin(3) == 1


def test_fast_len():
    assert V._fast_len(2187) == 2187 and V._fast_len(3828) == 3840 and V._fast_len(17) == 18


def _gauss(n, s, cx, cy):
    yy, xx = np.mgrid[:n, :n].astype(np.float64)
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))


def test_restoration_sharpens_stars_and_keeps_flux(tmp_path):
    """Eight noisy frames of point stars blurred by sigma 2 px: the restored stars are narrower than the frames'
    and keep their flux; pixels without data (NaN) are ignored."""
    rng = np.random.default_rng(3)
    n, N = 200, 8                                             # (81 x 81 PSF cut-outs: stars >= 45 px inside)
    stars = [(60.3, 65.7), (120.6, 70.2), (80.1, 120.4), (135.4, 135.9), (65.2, 145.5)]
    sky = 0.02
    st = V.FrameStore(str(tmp_path / "f.npy"), (N, 1, n, n))
    clean = np.full((n, n), sky)
    for cx, cy in stars:
        clean += 0.5 * _gauss(n, 2.0, cx, cy)
    for t in range(N):
        f = clean + rng.normal(0, 0.004, clean.shape)
        if t == 0:
            f[:, :6] = np.nan                                 # a frame edge without data
        st.data[t, 0] = f
        st.stars[t] = [(cx + rng.normal(0, 0.05), cy + rng.normal(0, 0.05)) for cx, cy in stars]
    eng = V.ImageMMEngine("cpu")
    out = eng.run_imagemm(st, iterations=20, log=lambda m: None)[0]
    assert np.isfinite(out).all() and out.shape == (n, n)
    w_in = [V_fwhm(clean, cx, cy) for cx, cy in stars]
    w_out = [V_fwhm(out, cx, cy) for cx, cy in stars]
    assert np.median(w_out) < 0.8 * np.median(w_in)
    for cx, cy in stars:
        ix, iy = int(round(cx)), int(round(cy))
        f_in = (clean[iy - 8:iy + 9, ix - 8:ix + 9] - sky).sum()
        f_out = (out[iy - 8:iy + 9, ix - 8:ix + 9] - np.median(out)).sum()
        assert abs(f_out / f_in - 1) < 0.1
    assert eng.last_run_info["psf_stars"][0] >= 4


def V_fwhm(img, cx, cy, r=7):
    ix, iy = int(round(cx)), int(round(cy))
    c = img[iy - r:iy + r + 1, ix - r:ix + r + 1]
    c = np.maximum(c - np.median(img), 0)
    return float(np.sqrt(V_fwhm_var(c)) * 2.3548)


def V_fwhm_var(c):
    yy, xx = np.mgrid[:c.shape[0], :c.shape[1]]
    s = c.sum()
    my, mx = (c * yy).sum() / s, (c * xx).sum() / s
    return ((c * ((yy - my) ** 2 + (xx - mx) ** 2)).sum() / s) / 2


def _donut(n=33, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n] - n // 2
    r = np.hypot(xx, yy)
    p = np.exp(-((r - 2.0) ** 2) / (2 * 1.6 ** 2)) + 0.02 * (1 + r ** 2 / 9.0) ** -2.5      # ring core + Moffat wing
    p = p / p.sum()
    return p + rng.normal(0, noise, p.shape)


def test_solve_keeps_the_psf_wings():
    """The script's Adam-on-ReLU fit lost every wing pixel (and a plain TV subgradient made them chatter): the
    projected solve with smooth TV keeps the light beyond 6 px and stays within the noise of the target."""
    t = _donut()
    yy, xx = np.mgrid[:33, :33] - 16
    wing = np.hypot(xx, yy) >= 6
    for noise in (0.0, 1e-4):
        f = torch.from_numpy(_donut(noise=noise, seed=3).astype(np.float32))
        h = V.PSFSolver("cpu").solve(f)[0, 0].numpy()
        assert abs(h.sum() - 1) < 1e-5 and h.min() >= 0
        assert np.sqrt(((h - t) ** 2).mean()) < 1.5e-4                 # within the noise of the truth
        if noise == 0:
            assert abs(h[wing].sum() / t[wing].sum() - 1) < 0.1          # the wings' light is kept


def test_radial_wings_remove_the_clipped_noise_floor():
    noisy = _donut(noise=1e-4, seed=2)
    clipped = np.maximum(noisy, 0)
    blended = np.maximum(V.PSFSolver._radial_wings(noisy), 0)
    yy, xx = np.mgrid[:33, :33] - 16
    far = np.hypot(xx, yy) >= 11
    truth = _donut()[far].sum()
    assert clipped[far].sum() > truth + 0.02                   # the script's clip: a floor of extra light
    assert abs(blended[far].sum() - truth) < 0.01                 # (ring medians: 4 x below the clipped floor)
    near = np.hypot(xx, yy) <= 3
    np.testing.assert_allclose(blended[near], noisy[near])     # the detected core is kept as measured


def _field(n, stars, sky, sigma, rng, noise):
    yy, xx = np.mgrid[:n, :n].astype(np.float64)
    img = np.full((n, n), sky)
    for cx, cy, a in stars:
        img += a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
    return img + rng.normal(0, noise, img.shape)


def test_lower_bound_stops_the_moat_round_bright_stars(tmp_path):
    rng = np.random.default_rng(5)
    stars = [(30.3, 30.6, 4.0), (70.2, 64.7, 3.0), (40.5, 75.4, 2.5), (72.6, 28.3, 3.5)]
    st = V.FrameStore(str(tmp_path / "f.npy"), (6, 1, 100, 100))
    for t in range(6):
        st.data[t, 0] = _field(100, stars, 0.3, 2.2, rng, 0.003)
        st.stars[t] = [(x, y) for x, y, _ in stars]
    out = {}
    for lb in (False, True):
        eng = V.ImageMMEngine("cpu")
        img = eng.run_imagemm(st, iterations=30, huber_delta=1e9, lower_bound=lb, per_channel_psf=False,
                              log=lambda m: None)[0]
        yy, xx = np.mgrid[:100, :100]
        r = np.hypot(xx - 30.3, yy - 30.6)
        out[lb] = img[(r > 4) & (r < 9)].min() - 0.3
    assert out[True] > out[False] and out[True] > -0.02


def test_local_bound_follows_the_nebula_under_the_stars():
    """IC 1805, 2026-10-21: a single level at the darkest sky let the moats dig into bright nebula (black discs round
    the stars after finishing); the local map sits at the nebula's level round a star and below a narrow dark lane."""
    yy, xx = np.mgrid[:200, :200]
    neb = 0.2 + 0.3 * (xx > 100)                             # bright nebula on the right half
    lane = (np.abs(xx - 150) <= 2)                             # a 5 px dark lane inside it
    img = neb - 0.25 * lane + 5.0 * np.exp(-((xx - 140) ** 2 + (yy - 60) ** 2) / 8.0)
    x = torch.from_numpy(img[None, None].astype(np.float32))
    b = V.ImageMMEngine._local_background(x, torch.ones_like(x, dtype=torch.bool))[0, 0].numpy()
    assert abs(b[60, 132] - 0.5) < 0.03                         # beside the star: the nebula, not the star
    assert b[100, 150] <= img[100, 150] + 0.01                  # never above the dark lane
    assert abs(b[100, 40] - 0.2) < 0.02


def test_star_saturated_in_every_frame_starts_from_its_clipped_values(tmp_path):
    """No frame has a valid core pixel: the core must start from the clipped data (the script's initial guess),
    not from 0 - a multiplicative update cannot grow a pixel that starts at 0."""
    rng = np.random.default_rng(6)
    st = V.FrameStore(str(tmp_path / "f.npy"), (4, 1, 64, 64))
    yy, xx = np.mgrid[:64, :64]
    core = np.hypot(xx - 32, yy - 32) <= 2
    for t in range(4):
        f = _field(64, [(32, 32, 5.0), (15.4, 16.2, 1.0), (48.6, 47.3, 1.0)], 0.2, 2.0, rng, 0.003)
        f[core] = np.minimum(f[core], 2.0)                      # clipped
        st.data[t, 0] = f
        st.sat[t] = core
        st.stars[t] = [(15.4, 16.2), (48.6, 47.3)]
    img = V.ImageMMEngine("cpu").run_imagemm(st, iterations=10, log=lambda m: None)[0]
    assert img[32, 32] > 2.0 and img[core].min() > 1.0


def _moffat(n, cx, cy, a, alpha=2.5, beta=2.2):
    yy, xx = np.mgrid[:n, :n].astype(np.float64)
    return a * (1.0 + ((xx - cx) ** 2 + (yy - cy) ** 2) / alpha ** 2) ** (-beta)


def test_radial_wing_profile_joins_saturated_stars_and_skips_a_bulge():
    """The extended PSF (Infante-Sainz et al. 2020): unsaturated stars give the profile near the centre, saturated stars
    (NaN cores) the far wing; a star on a galaxy's bulge (curved sky) must not lift it (M 31 2026-10-22: 1.7 x at 16 px)."""
    rng = np.random.default_rng(11)
    n = 600
    img = np.full((n, n), 0.1) + rng.normal(0, 0.002, (n, n))
    cat = []
    for i, (cx, cy) in enumerate([(70 + 115 * (k % 5) + 0.3, 70 + 115 * (k // 5) + 0.6) for k in range(25)]):
        a = 400.0 if i % 6 == 0 else 2.0
        img += _moffat(n, cx, cy, a)
        cat.append((cx, cy))
    yy, xx = np.mgrid[:n, :n]
    bulge = 0.4 * np.exp(-((xx - cat[7][0]) ** 2 + (yy - cat[7][1]) ** 2) / (2 * 25.0 ** 2))
    img += bulge
    sat = img > 5.0
    img[sat] = np.nan
    P, used = V.ImageMMEngine._radial_wing_profile(img, cat, bloom=0)
    true = (1.0 + np.arange(41) ** 2 / 2.5 ** 2) ** (-2.2)
    true /= true[8]
    for k in (10, 14, 20):
        assert abs(P[k] / true[k] - 1.0) < 0.15, (k, P[k], true[k])
    assert 0.6 < P[26] / true[26] < 1.2                    # (the sky ring at 36-40 px still holds some of the wing)
    assert used >= 10


def test_bloom_zone_next_to_a_saturated_core_is_missing_data(tmp_path):
    """Light spilled next to a full well (more than the PSF puts there) must not become a shell of light in the latent:
    the pixels within bloom_grow px of a frame's saturated pixels are left out of the fit."""
    rng = np.random.default_rng(12)
    N = 64
    yy, xx = np.mgrid[:N, :N]
    r = np.hypot(xx - 32, yy - 32)
    core = r <= 2.5
    st = V.FrameStore(str(tmp_path / "f.npy"), (4, 1, N, N))
    for t in range(4):
        f = _field(N, [(32, 32, 8.0), (14.4, 15.2, 1.0), (49.6, 48.3, 1.0), (15.3, 48.7, 1.0), (49.2, 14.6, 1.0)], 0.2, 1.6, rng, 0.003)
        f += 0.25 * ((r > 2.5) & (r <= 5.0))                       # the spill: a ring the PSF does not have
        f[core] = 2.0
        st.data[t, 0] = f
        st.sat[t] = core
        st.stars[t] = [(14.4, 15.2), (49.6, 48.3), (15.3, 48.7), (49.2, 14.6)]
    ring = (r > 3.5) & (r <= 6.0)
    out = {}
    for b in (0, 3):
        img = V.ImageMMEngine("cpu").run_imagemm(st, iterations=20, bloom_grow=b, saturated_photometry=False, psf_radial_wings=False,
                                                 psf_asymmetry=False, log=lambda m: None)[0]
        out[b] = float(np.median(img[ring]) - 0.2)
    assert out[3] < 0.5 * out[0], out


def test_saturated_star_is_a_point_at_its_wings_colour_and_a_galaxy_stays_extended():
    """A saturated star's core (no data: light and colour not identified) becomes a point at the latent's resolution
    with the colour of its unsaturated wings, on the background plane; the restoration further out is untouched; a
    galaxy nucleus (M 32) keeps the restoration's shape."""
    rng = np.random.default_rng(13)
    C, N = 3, 300
    yy, xx = np.mgrid[:N, :N].astype(np.float64)
    sky, noise = 0.1, 0.001
    lat = np.full((C, N, N), sky) + rng.normal(0, noise, (C, N, N))
    col = np.array([0.7, 1.0, 0.5])
    r1 = np.hypot(xx - 80, yy - 80)
    r2 = np.hypot(xx - 210, yy - 200)
    gal = 0.05 * np.exp(-r2 / 8.0)
    for c in range(C):
        lat[c] += (5.0 if c == 1 else 1.0) * (r1 <= 3) + col[c] * gal          # (the core's light per channel: free)
    core = (r1 <= 3) | (r2 <= 3)
    k = np.exp(-(np.mgrid[-20:21, -20:21] ** 2).sum(0) / (2 * 1.8 ** 2)); k /= k.sum()
    mean_img = np.full((C, N, N), sky)
    star = cv2.filter2D(np.where(r1 <= 0.5, 30.0, 0.0), -1, k)
    for c in range(C):
        mean_img[c] += np.minimum(col[c] * star, 1.0) + col[c] * cv2.filter2D(gal, -1, k)   # (clipped at 1)
    kmean = np.stack([k] * C).astype(np.float32)
    x = torch.from_numpy(lat.astype(np.float32))[None]
    out, info = V.ImageMMEngine._saturated_star_photometry(x, core, mean_img, kmean, ann=(4, 8))
    o = out[0].numpy().astype(np.float64)
    kinds = {(round(s[0] / 10), round(s[1] / 10)): s[2] for s in info["stars"]}
    assert kinds[(8, 8)] == "point" and kinds[(21, 20)] == "extended", info["stars"]
    far = (r1 >= 20) & (r1 <= 45)
    assert np.allclose(o[:, far], lat[:, far])                                         # untouched further out
    c0 = r1 <= 3
    ratio = [(o[c][c0] - sky).sum() / (o[1][c0] - sky).sum() for c in range(C)]
    assert abs(ratio[0] - 0.7) < 0.05 and abs(ratio[2] - 0.5) < 0.05                  # the wings' colour
    ring = (r1 >= 5) & (r1 <= 8)
    assert abs(np.median(o[1][ring]) - sky) < 0.01                                     # on the background
    g_ring = (r2 >= 8) & (r2 <= 16)
    assert abs(np.median(o[1][g_ring]) - np.median(lat[1][g_ring])) < 1e-3            # the galaxy: the restoration's


def test_bloom_zone_grows_with_a_large_saturated_patch():
    """Round a large saturated patch the missing zone reaches 0.75 x its equivalent radius (M 31's brightest star: the
    light beyond the PSF's out to 8-9 px); round a small one, bloom_grow px."""
    N = 200
    yy, xx = np.mgrid[:N, :N]
    sat = np.zeros((3, N, N), bool)
    big = np.hypot(xx - 60, yy - 60) <= 12.0
    small = np.hypot(xx - 150, yy - 150) <= 2.0
    sat[:, big] = True
    sat[1, small] = True
    z = V.ImageMMEngine._bloom_zone(sat, 4)
    r_big = np.hypot(xx - 60, yy - 60)
    r_small = np.hypot(xx - 150, yy - 150)
    assert z[0][(r_big > 12.5) & (r_big <= 20.5)].all() and not z[0][r_big > 22].any()        # 12 + 0.75 x 12 = 21
    assert z[1][(r_small > 2.5) & (r_small <= 6.0)].all() and not z[1][(r_small > 7.0) & (r_small < 40)].any()   # 2 + 4
    assert not z[0][r_small <= 7].any()                                                        # (per channel)
    assert not (z & sat).any()


def test_saturated_star_starts_at_the_light_of_its_unsaturated_wings():
    """The initial point of a saturated star carries the light PSF photometry of its unsaturated wings measures, not
    the clipped light (a lower bound): started at the clipped light the update kept the rest as a halo."""
    C, N = 3, 260
    yy, xx = np.mgrid[:N, :N].astype(np.float64)
    k = (1.0 + (np.mgrid[-40:41, -40:41] ** 2).sum(0) / 2.5 ** 2) ** -2.5
    k /= k.sum()
    flux = 4000.0
    star = cv2.filter2D(np.where(np.hypot(xx - 130, yy - 130) <= 0.5, flux, 0.0), -1, k)
    img = np.stack([0.1 + c_ * star for c_ in (0.6, 1.0, 0.5)])
    clip = 1.0
    core = (img[1] >= clip)
    img = np.minimum(img, clip)
    vf = np.ones((C, N, N))
    near = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    vf[:, near] = 0.0
    x = torch.from_numpy(img.astype(np.float32))[None]
    kmean = np.stack([k] * C).astype(np.float32)
    out, info = V.ImageMMEngine._point_initial(x, core, 1.0, [1e-3] * C, kmean=kmean, valid_frac=vf)
    o = out[0].numpy().astype(np.float64)
    r = np.hypot(xx - 130, yy - 130)
    got = [(o[c][r <= 6] - 0.1).sum() for c in range(C)]
    for c, c_ in enumerate((0.6, 1.0, 0.5)):
        assert abs(got[c] / (c_ * flux) - 1.0) < 0.15, (c, got[c], c_ * flux)
    clipped = (np.minimum(img[1], clip)[r <= 40] - 0.1).sum()
    assert got[1] > 2.0 * clipped                                     # (the clipped light: far below the star's)


def test_saturated_star_is_centred_on_its_missing_data():
    """A saturated star's centre is the centroid of its missing data (valid in under half the frames), not the
    plateau's: the plateau leans into the PSF's lobe."""
    R = 30
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float64)
    reg = np.hypot(xx + 1.0, yy) <= 5.0                                 # plateau centred 1 px to the left
    vf = np.where(np.hypot(xx - 0.4, yy + 0.3) <= 10.0, 0.2, 1.0)        # missing zone round the true centre
    cy, cx = V.ImageMMEngine._hole_centre(reg, vf, yy, xx, (0.0, -1.0))
    assert abs(cx - 0.4) < 0.15 and abs(cy + 0.3) < 0.15, (cx, cy)
    assert V.ImageMMEngine._hole_centre(np.zeros_like(reg), np.ones_like(vf), yy, xx, (1.0, 2.0)) == (1.0, 2.0)


def test_far_nonradial_measures_a_lopsided_halo_round_saturated_stars():
    """The PSF's lobe beyond 20 px (m = 1) from the saturated stars of the frames' mean: their halos brighter on one
    side by 15 % - the modulation recovered with its direction."""
    rng = np.random.default_rng(21)
    n = 520
    yy, xx = np.mgrid[:n, :n].astype(np.float64)
    img = np.full((n, n), 0.1) + rng.normal(0, 1e-4, (n, n))
    cat = []
    for (cx, cy) in [(130.3, 130.6), (390.4, 130.2), (130.7, 390.5), (390.1, 389.8)]:
        r = np.hypot(xx - cx, yy - cy)
        th = np.arctan2(yy - cy, xx - cx)
        img += 300.0 * (1.0 + r ** 2 / 3.0 ** 2) ** -1.6 * (1.0 + 0.15 * np.cos(th))
        img[r <= 6.0] = np.nan
        cat.append((cx, cy))
    out = V.ImageMMEngine._far_nonradial(img, cat, R=80)
    assert out is not None
    M, used, a1, ang, rings = out
    assert used == 4 and abs(a1 - 0.15) < 0.03 and abs(ang) < 10, (used, a1, ang)
    assert abs(M[80, 80 + 30] - 0.15) < 0.03 and abs(M[80, 80 - 30] + 0.15) < 0.03


def test_wing_chroma_follows_a_halo_that_reddens_outward():
    """Per channel, the far wing's factor on the channels' common wing: a halo whose red share grows outward (the
    field's brightest star on M 31) gives red a factor rising past 1 and blue one falling below it."""
    rng = np.random.default_rng(22)
    n = 400
    yy, xx = np.mgrid[:n, :n].astype(np.float64)
    ref = np.full((3, n, n), 0.1) + rng.normal(0, 1e-5, (3, n, n))
    cat = []
    for (cx, cy) in [(120.3, 120.6), (280.4, 280.2), (120.7, 280.5), (280.1, 119.8)]:
        r = np.hypot(xx - cx, yy - cy)
        prof = 500.0 * (1.0 + r ** 2 / 2.5 ** 2) ** -1.7
        tilt = np.clip((r - 14.0) / 20.0, 0.0, 1.0)
        for c, sgn in enumerate((+1.0, 0.0, -1.0)):
            ref[c] += prof * (1.0 + 0.2 * sgn * tilt)
        ref[:, r <= 9.0] = np.nan
        cat.append((cx, cy))
    ch = V.ImageMMEngine._wing_chroma(ref.astype(np.float32), cat, R=60)
    assert ch is not None
    assert ch[0][45] > 1.08 and ch[2][45] < 0.92 and abs(ch[1][45] - 1.0) < 0.06, (ch[0][45], ch[1][45], ch[2][45])
    assert np.allclose(ch[:, :10], 1.0)


def test_local_bound_is_not_held_up_over_a_dust_lane_under_the_sources():
    """Under a galaxy's bulge (all of it inside the stars' source masks) a dark lane narrower than a mesh cell: the
    bound follows the lane down, it is not held at the mesh's level (flat patches in the restoration)."""
    n = 192
    yy, xx = np.mgrid[:n, :n].astype(np.float64)
    img = 1.0 + 0.5 * np.exp(-((xx - 96) ** 2 + (yy - 96) ** 2) / (2 * 60.0 ** 2))
    lane = np.abs(xx - 100) <= 4
    img[lane] -= 0.3
    x = torch.from_numpy(img.astype(np.float32))[None, None]
    src = np.zeros((n, n), bool); src[40:150, 40:150] = True
    b = V.ImageMMEngine._local_background(x, torch.ones_like(x, dtype=torch.bool), src)[0, 0].numpy()
    assert np.all(b[60:130, 98:103] <= img[60:130, 98:103] + 1e-3)


# ----------------------------------------------------------------------------- several exposure times (HDR sets)

def test_transmission_is_per_second_of_exposure():
    """A 60 s frame of a 60 s + 120 s set holds half the light of a 120 s frame: not a cloud frame.  A 120 s frame at
    0.3 x the others' light still is one."""
    s = np.array([1.0, 1.02, 0.98, 1.0, 1.01, 0.47, 0.46, 0.48, 0.3])   # the 60 s frames under slightly thinner sky (0.94 x)
    exp = [120, 120, 120, 120, 120, 60, 60, 60, 120]
    assert set(V.transmission_drops(s, exp)) == {8}
    assert set(V.transmission_drops(s)) == {5, 6, 7, 8}              # without the exposure times every 60 s frame went


def test_photon_transfer_with_mixed_frame_scales():
    """Frames on the common scale y / s_t with shot variance slope x level / s_t: the fit recovers the slope whatever
    the mix of scales (unweighted, the 60 s frames' doubled variance biased it)."""
    rng = np.random.default_rng(5)
    n_px, slope = 20000, 2e-5
    level = rng.uniform(0.0, 0.2, n_px)
    ref = rng.random((100, 200)).astype(np.float32)                 # (the smooth-pixel selection needs a gradient to rank)
    pick = np.arange(n_px)
    sky = np.zeros((20, 1))

    def fit(scale, weighted=True):
        cube = (level[None] + rng.standard_normal((20, n_px)) * np.sqrt(slope * level[None] / scale[:, None]))[:, None].astype(np.float32)
        return V.ImageMMEngine._photon_transfer(cube, sky, ref, pick, scale if weighted else None)["slope"][0]
    one = fit(np.ones(20))                                         # (the 20-frame median-of-MAD estimate itself reads ~0.89 x)
    mixed = np.array([1.0] * 10 + [0.5] * 10)
    assert fit(mixed) == pytest.approx(one, rel=0.08)                # (two noise draws: ~5 % apart)
    assert fit(mixed, weighted=False) > 1.25 * one                 # pooled unweighted: ~1.4 x
