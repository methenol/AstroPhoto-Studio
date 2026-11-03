"""VeraLux ImageMM headless engine (sirilpipe/veralux_imagemm.py): the FFT operators against the script's padded
conv2d, the forward model's orientation, the findstar parser / coordinates and a small multi-frame restoration."""
from __future__ import annotations

import cv2
import math
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


def test_a_saturated_star_whose_wings_are_buried_in_noise_is_refused_not_repainted():
    """The wing ratio was accepted for being *positive*, so a channel whose annulus light never rose above the noise
    passed the guard with a ratio of ~4e-4 and its core was written at that ratio: C 49 (2026-10-31) put [0.0, 1.0,
    0.0] - that ratio, rounded - into every saturated core and drove red and blue below the sky, which is the green
    core the campaign was called in for.  A ratio with no measurement behind it now refuses the star and leaves the
    restoration's own core, which measures far closer to the stack than an invented colour does."""
    rng = np.random.default_rng(7)
    C, N = 3, 300
    yy, xx = np.mgrid[:N, :N].astype(np.float64)
    sky, noise = 0.1, 0.001
    lat = np.full((C, N, N), sky) + rng.normal(0, noise, (C, N, N))
    r1 = np.hypot(xx - 80, yy - 80)
    for c in range(C):
        lat[c] += 5.0 * (r1 <= 3)                                             # the core: full well, no data in any channel
    core = (r1 <= 3)
    k = np.exp(-(np.mgrid[-20:21, -20:21] ** 2).sum(0) / (2 * 1.8 ** 2)); k /= k.sum()
    star = cv2.filter2D(np.where(r1 <= 0.5, 30.0, 0.0), -1, k)
    mean_img = np.full((C, N, N), sky) + rng.normal(0, noise, (C, N, N))
    broad = np.exp(-(r1 / 6.0) ** 2)                                          # the light the wing annulus sees (r 9..13)
    mean_img[1] += np.minimum(star, 1.0) + 0.12 * broad                        # green's wings: 87 sigmas of that annulus
    mean_img[0] += 0.0041 * broad                                             # red and blue: 1.9 and 3.8 sigmas - a
    mean_img[2] += 0.0041 * broad                                             # positive ratio that was never measured
    kmean = np.stack([k] * C).astype(np.float32)
    x = torch.from_numpy(lat.astype(np.float32))[None]
    out, info = V.ImageMMEngine._saturated_star_photometry(x, core, mean_img, kmean, ann=(4, 8))
    o = out[0].numpy().astype(np.float64)
    assert info["measured"] == 0 and info["wing_below_noise"] == 1, info
    snr = info["stars"][0][-1]                                                # what each channel's colour was worth
    assert snr[1] > V.SAT_WING_SNR and snr[0] < V.SAT_WING_SNR and snr[2] < V.SAT_WING_SNR, info["stars"]
    assert np.allclose(o[:, r1 <= 8], lat[:, r1 <= 8])                        # not repainted at the noise's colour


def test_an_uncalibrated_wing_colour_is_the_light_and_not_the_kernel_weight():
    """``q_in`` / ``q_out`` divide each channel's annulus light by that channel's *own* kernel weight: the star's colour
    only once ``col_cal`` undoes it (fitted on real stars by this same estimator). With no calibration star in any ring
    band the widest kernel divides its channel away - C 49 (2026-10-31) had none, and its saturated stars' red and blue
    wings carried 12.7 and 26.4 sigmas of light (R/G 0.45, B/G 0.94, an ordinary star) while the ratio read [0.0, 1.0,
    0.0] and wrote both channels below the sky. Uncalibrated, the colour is the light: the annulus is the same geometry
    in every channel and a star's profile the same shape, so light compared to light needs no kernel to undo."""
    rng = np.random.default_rng(11)
    C, N = 3, 300
    yy, xx = np.mgrid[:N, :N].astype(np.float64)
    sky, noise = 0.1, 0.001
    lat = np.full((C, N, N), sky) + rng.normal(0, noise, (C, N, N))
    r1 = np.hypot(xx - 80, yy - 80)
    for c in range(C):
        lat[c] += 5.0 * (r1 <= 3)
    core = (r1 <= 3)
    k = np.exp(-(np.mgrid[-20:21, -20:21] ** 2).sum(0) / (2 * 1.8 ** 2)); k /= k.sum()
    star = cv2.filter2D(np.where(r1 <= 0.5, 30.0, 0.0), -1, k)
    broad = np.exp(-(r1 / 6.0) ** 2)
    mean_img = np.full((C, N, N), sky) + rng.normal(0, noise, (C, N, N))
    for c in range(C):
        mean_img[c] += np.minimum(star, 1.0) + 0.12 * broad                   # the same light everywhere: a white star
    ks = []
    for sig in (4.0, 1.8, 1.8):                                               # the frames' PSF, red's far the widest
        kk = np.exp(-(np.mgrid[-20:21, -20:21] ** 2).sum(0) / (2 * sig ** 2)); ks.append(kk / kk.sum())
    kmean = np.stack(ks).astype(np.float32)
    x = torch.from_numpy(lat.astype(np.float32))[None]
    out, info = V.ImageMMEngine._saturated_star_photometry(x, core, mean_img, kmean, ann=(4, 8))
    o = out[0].numpy().astype(np.float64)
    assert info["measured"] == 1 and info["wing_below_noise"] == 0, info
    s = info["stars"][0]
    assert s[5][0] < 0.05, s                                                  # the kernel-normalised ratio crushes red
    assert s[7][0] > 0.5, s                                                   # the light does not: that is what the core got
    c0 = r1 <= 3
    ratio = [(o[c][c0] - sky).sum() / (o[1][c0] - sky).sum() for c in range(C)]
    assert abs(ratio[0] - 1.0) < 0.2 and abs(ratio[1] - 1.0) < 0.02 and abs(ratio[2] - 1.0) < 0.2, ratio


def test_a_calibrated_wing_colour_undoes_the_kernel_weight_using_real_stars():
    """The same estimator with stars to calibrate on: ``col_cal`` is each ring band's measured colour divided by what
    this same estimator returned for it, fitted on the bright unsaturated catalogue stars, so multiplying by it cancels
    each channel's kernel weight. Here red's wider PSF (C 49's measured FWHM 7.42 px against green's 6.33) leaves the
    ratio 2.8 x low in red and 2.7 x low in blue; the calibration lifts both back toward the light and the delivered
    core carries exactly what the engine recorded -- the invariant the green cores violated. It is an aperture
    correction and not an identity: the aperture it calibrates against (6 px) and the annulus it corrects (10 px) see
    differently shaped channel profiles, so it still lands ~40 % under the star's colour, which is why the wing's own
    ``snr`` -- the light, needing no correction -- is recorded beside it."""
    rng = np.random.default_rng(7)
    C, N = 3, 700
    sky, noise = 0.1, 0.001
    SIG = (3.15, 2.69, 3.05)                                                   # C 49's FWHM 7.42 / 6.33 / 7.18 px

    def gauss(sig):
        g = np.mgrid[-30:31, -30:31] ** 2
        k = np.exp(-g.sum(0) / (2 * sig * sig))
        return k / k.sum()

    ks = [gauss(s) for s in SIG]
    wings = gauss(5.0)                                                          # the extended PSF: one shape, all channels
    lat = np.full((C, N, N), sky) + rng.normal(0, noise, (C, N, N))
    mean_img = np.full((C, N, N), sky) + rng.normal(0, noise, (C, N, N))

    def star(img, sx, sy, amps):
        d = np.zeros((N, N)); d[sy, sx] = 1.0
        for c in range(C):
            img[c] += amps[c] * (0.7 * cv2.filter2D(d, -1, ks[c]) + 0.3 * cv2.filter2D(d, -1, wings))

    sat_col, cal_col = (0.45, 1.0, 0.94), (0.60, 1.0, 0.80)                   # C 49's star; the calibration stars' colour
    star(mean_img, 200, 200, [8.0 * v for v in sat_col])
    yy, xx = np.mgrid[:N, :N]
    r1 = np.hypot(xx - 200, yy - 200)
    for c in range(C):
        lat[c] += 5.0 * (r1 <= 3)
    core = (r1 <= 3)
    catalog = []
    for gx in (350, 450, 550):                                                 # nine, 100 px apart: no ring sees a neighbour
        for gy in (350, 450, 550):
            star(mean_img, gx, gy, [5.0 * v for v in cal_col])                 # bright enough for the calibration's floor
            catalog.append((float(gx), float(gy)))
    kmean = np.stack(ks).astype(np.float32)
    x = torch.from_numpy(lat.astype(np.float32))[None]
    out, info = V.ImageMMEngine._saturated_star_photometry(x, core, mean_img, kmean, ann=(4, 8), catalog=catalog)
    o = out[0].numpy().astype(np.float64)
    assert info["measured"] == 1 and info["wing_below_noise"] == 0, info
    assert info["calibration_bands"] >= 1, info
    s = info["stars"][0]
    q, col, truth = np.array(s[5]), np.array(s[7]), np.array(sat_col)
    assert q[0] < 0.6 * truth[0] and q[2] < 0.6 * truth[2], s                  # the kernel weight buries both channels
    assert col[0] > 1.4 * q[0] and col[2] > 1.4 * q[2], s                      # the calibration lifts them back
    assert np.abs(col - truth)[[0, 2]].max() < np.abs(q - truth)[[0, 2]].max(), s
    c0 = r1 <= 3
    ratio = np.array([(o[c][c0] - sky).sum() for c in range(C)])
    assert np.allclose(ratio / ratio[1], col, atol=0.02), (ratio.tolist(), col.tolist())


def test_a_core_is_never_left_dimmer_than_the_frames_say_the_star_is():
    """``amps`` is the model's TOTAL light -- it multiplies a PSF normalised to unit light over the region -- and it is
    taken from ``Lg``, the restoration's own light inside the censored region. So when the restoration has crushed that
    region, the star lands short everywhere at once: C 49 (2026-10-31) delivered 0.34 of the stack's aperture flux at
    every radius and, because ``post.py`` keeps ImageMM's pixels outside the stack's fitted footprint, a dark annulus at
    r 6-12 in ``linear.fits`` -- while the frames' mean image carries a PSF-consistent halo out there. The frames are the
    only unbiased photometry a saturated star has, so a rewritten core may not be quieter than the star they show: it has
    to sit within a factor of two of the flux the mean image holds for the same aperture."""
    rng = np.random.default_rng(5)
    C, N = 3, 700
    sky, noise = 0.1, 1e-5                                                   # quiet: a crushed core must not read as extended

    def gauss(sig):
        g = np.mgrid[-30:31, -30:31] ** 2
        k = np.exp(-g.sum(0) / (2 * sig * sig))
        return k / k.sum()

    ks = [gauss(s) for s in (3.15, 2.69, 3.05)]                             # C 49's per-channel FWHM
    wings = gauss(5.0)                                                       # the extended PSF, one shape per channel
    lat = np.full((C, N, N), sky) + rng.normal(0, noise, (C, N, N))
    d = np.zeros((N, N)); d[200, 200] = 1.0
    col_true = (0.45, 1.0, 0.94)
    mean_img = np.full((C, N, N), sky) + rng.normal(0, noise, (C, N, N))
    for c in range(C):
        mean_img[c] += 8.0 * col_true[c] * (0.7 * cv2.filter2D(d, -1, ks[c]) + 0.3 * cv2.filter2D(d, -1, wings))
    r1 = np.hypot(*[a - 200 for a in np.mgrid[:N, :N]])
    for c in range(C):                                                       # the restoration crushed the core: what
        lat[c] += 0.01 * (r1 <= 3)                                           # Lg would happily accept as the star
    core = (r1 <= 3)
    x = torch.from_numpy(lat.astype(np.float32))[None]
    out, info = V.ImageMMEngine._saturated_star_photometry(x, core, mean_img, np.stack(ks).astype(np.float32), ann=(4, 8))
    assert info["measured"] == 1, info                                       # the wings are significant: it is rewritten
    o = out[0].numpy().astype(np.float64)
    a_out = [(o[c][core] - sky).sum() for c in range(C)]
    a_ref = [(mean_img[c][core] - sky).sum() for c in range(C)]
    a = info["amplitudes"]["200,200"]
    assert a["Lg_wing"] > 4.0 * a["Lg_region"], a                             # the region's light was a fraction of the star
    assert min(a_out) > 0.5 * min(a_ref), (a_out, a_ref)                       # never quieter than the frames by 2x
    assert max(float(o[c][core].max()) for c in range(C)) > 0.5 * max(float(mean_img[c][core].max()) for c in range(C))


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


def test_the_starting_point_of_a_saturated_star_is_its_psf_not_a_delta():
    """The complaint against the ImageMM renders is the star's *shape*, not its light - measured on C 20 (2026-11-01)
    the rewritten stars carry 68.5 % of their r<12 light inside r<2 where the stack carries 10.7 %, and an eighth of the
    stack's share at r4-6 (hot point, missing halo, colour crowns after finishing).  The cause is the shape the initial
    guess deposits the measured light with: a 1 px Gaussian leaves nothing for the restoration to refine against, while
    the same run has already least-squared the wings against the frames' own kernel.  ``point_kernel`` (change 34)
    deposits through that kernel: the light must be untouched, the shape must move out of the core."""
    C, N = 3, 260
    yy, xx = np.mgrid[:N, :N].astype(np.float64)
    k = (1.0 + (np.mgrid[-40:41, -40:41] ** 2).sum(0) / 2.5 ** 2) ** -2.5
    k /= k.sum()
    flux = 4000.0
    star = cv2.filter2D(np.where(np.hypot(xx - 130, yy - 130) <= 0.5, flux, 0.0), -1, k)
    img = np.stack([0.1 + c_ * star for c_ in (0.6, 1.0, 0.5)])
    core = (img[1] >= 1.0)
    img = np.minimum(img, 1.0)
    vf = np.ones((C, N, N))
    vf[:, cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0] = 0.0
    x = torch.from_numpy(img.astype(np.float32))[None]
    kmean = np.stack([k] * C).astype(np.float32)
    r = np.hypot(xx - 130, yy - 130)

    def shapes(kernel):
        out, info = V.ImageMMEngine._point_initial(x, core, 1.0, [1e-3] * C, kmean=kmean, valid_frac=vf, kernel=kernel)
        g = out[0].numpy().astype(np.float64)[1] - 0.1                     # green: the channel the star's photometry is anchored on
        return float(g[r <= 2].sum() / g[r <= 12].sum()), float(g[r <= 12].sum()), info

    delta, tot_delta, info_delta = shapes(False)
    psf, tot_psf, info_psf = shapes(True)
    kg = cv2.GaussianBlur(k, (0, 0), 1.0)                                             # the frames' PSF at the latent's resolution - the shape the
    rr = np.hypot(*np.mgrid[-40:41, -40:41])                                          # deposit is now claimed to reproduce
    want = float(kg[rr <= 2].sum() / kg[rr <= 12].sum())
    assert delta - psf > 0.3, (delta, psf)                                            # as shipped, the star's light sat in its core; through the
    assert abs(psf - want) < 0.08, (psf, want)                                        # kernel it sits where the optics put it
    assert abs(tot_psf / tot_delta - 1.0) < 0.05, (tot_delta, tot_psf)       # and moving it cost no light
    # the record says which shape each star got, in the defect's own units: 1 - e^-2 = 0.865 is the fraction of a unit-sum
    # Gaussian of the latent's own resolution (sigma 1) that lands inside r<2 - the whole star in one resolution element -
    # where the kernel deposits the fraction the frames' optics measure.  A restoration's starting point is only believable
    # by the before/after measurement of what it did to the star.
    assert abs(info_delta["stars"][0][5][1] - (1.0 - math.exp(-2.0))) < 0.02, info_delta["stars"]
    assert abs(info_psf["stars"][0][5][1] - want) < 0.05, info_psf["stars"]


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


def test_the_core_rewrite_gives_back_the_light_the_restoration_had():
    """A rewritten saturated core may not leave the region quieter than the restoration found it.  ``amps`` is decided
    on green -- ``Lg``, or green's edge fit -- and shared by every channel through the wing colour, so a star whose
    channels the fit left level is written at the wings' colour ratios and loses the difference.  On C 49 (2026-10-31)
    that is exactly what happened: ``from_wings / amps`` was 0.68 identically for every star and channel, i.e. ``Lg``
    always took the restoration's own region light and the wing anchor never fired, only 1 of 43 stars had a
    significant edge fit, and the step that is supposed to restore the star's photometry delivered 0.403 / 0.754 /
    0.557 of the stack's aperture flux against 0.645 / 0.793 / 0.741 with it off.  The wings keep the colour -- what
    they cannot do is pay for the light -- so the total is given back with the colour ratios intact."""
    rng = np.random.default_rng(7)
    C, N = 3, 700
    sky, noise = 0.1, 1e-5

    def gauss(sig):
        g = np.mgrid[-30:31, -30:31] ** 2
        k = np.exp(-g.sum(0) / (2 * sig * sig))
        return k / k.sum()

    ks = [gauss(s) for s in (3.15, 2.69, 3.05)]
    wings = gauss(5.0)
    col_true = (0.45, 1.0, 0.94)
    d = np.zeros((N, N)); d[200, 200] = 1.0
    mean_img = np.full((C, N, N), sky) + rng.normal(0, noise, (C, N, N))
    for c in range(C):                                                    # the frames: a red-faint star, significant wings
        mean_img[c] += 1.0 * col_true[c] * (0.7 * cv2.filter2D(d, -1, ks[c]) + 0.3 * cv2.filter2D(d, -1, wings))
    lat = np.full((C, N, N), sky) + rng.normal(0, noise, (C, N, N))
    blob = cv2.filter2D(d, -1, gauss(3.0))
    for c in range(C):                                                    # the restoration: level across the channels,
        lat[c] += 16.0 * blob                                             # and carrying more there than the wings imply
    r1 = np.hypot(*[a - 200 for a in np.mgrid[:N, :N]])
    core = (r1 <= 3)
    x = torch.from_numpy(lat.astype(np.float32))[None]
    out, info = V.ImageMMEngine._saturated_star_photometry(x, core, mean_img, np.stack(ks).astype(np.float32), ann=(4, 8))
    assert info["measured"] == 1, info                                    # the wings are significant: the star is rewritten
    o = out[0].numpy().astype(np.float64)
    a = info["amplitudes"]["200,200"]
    assert a["Lg_region"] > a["Lg_wing"] and a["Lg"] == a["Lg_region"], a   # the case C 49 was in: the anchor is the region
    tot_in = sum(float((lat[c][core] - sky).sum()) for c in range(C))
    tot_out = sum(float((o[c][core] - sky).sum()) for c in range(C))
    assert sum(a["amps_pre_floor"]) < sum(a["region_light"]), a           # the case: the model's total was short of the region's
    assert sum(a["amps"]) > sum(a["amps_pre_floor"]), a                    # ... and it was given back
    assert tot_out >= 0.92 * tot_in, (tot_out, tot_in)                     # the region is not left quieter than it was
    pre = np.array(a["amps_pre_floor"], float)
    post = np.array(a["amps"], float)
    assert post / post[1] == pytest.approx(pre / pre[1], rel=1e-3)       # given back at the wings' colour, not level


def _nucleus_store(tmp_path, n_frames=4, N=260):
    """Frames of a galaxy whose nucleus is clipped in every frame (M 31 on a Seestar, 60 s gain 200), with stars for the PSF
    (41 px cut-outs: 45 px from the edges, clear of the galaxy)."""
    rng = np.random.default_rng(21)
    yy, xx = np.mgrid[:N, :N].astype(np.float64)
    gal = 3.0 * np.exp(-np.hypot(xx - 130, yy - 130) / 9.0)
    stars = [(45.4, 45.2, 1.0), (215.6, 45.3, 1.0), (45.2, 214.7, 1.0), (214.3, 215.4, 1.0), (130.5, 45.6, 0.8), (45.3, 130.7, 0.9),
             (215.6, 130.4, 0.9), (130.4, 215.6, 0.8), (88.7, 46.2, 0.9), (172.4, 214.4, 0.8)]
    clip = 1.5
    st = V.FrameStore(str(tmp_path / "f.npy"), (n_frames, 1, N, N))
    for t in range(n_frames):
        f = _field(N, stars, 0.2, 1.8, rng, 0.003) + cv2.GaussianBlur(gal, (0, 0), 1.8)
        sat = f >= clip
        f[sat] = clip
        st.data[t, 0] = f
        st.sat[t] = sat
        st.stars[t] = [(x, y) for x, y, _ in stars]
    return st, sat, clip


def test_a_galaxy_nucleus_saturated_in_every_frame_is_censored_not_drained(tmp_path):
    """Change 35: a nucleus clipped in every frame is told from a star by the light beyond its core in the rings that hold data (the
    no-data zone round it is 0 in the initial guess and read as "no light beyond the core": a point of 9000 x the clip level), and
    is censored data, at least the clip level, rather than missing data (the Seestar M 31's nucleus: a disc at the sky level,
    red and black after finishing)."""
    st, sat, clip = _nucleus_store(tmp_path)
    eng = V.ImageMMEngine("cpu")
    img = eng.run_imagemm(st, iterations=20, log=lambda m: None)[0]
    assert sat.sum() > 50
    core = float(np.median(img[sat]))
    assert 0.9 * clip < core < 4.0 * clip, core                         # (the old classification made it a "star" at 9000 x the clip)
    assert eng.last_run_info["censored_extended_px"] >= sat.sum()
    assert float(img[sat].min()) >= 0.98 * clip, float(img[sat].min())    # censored: at least the clip level, everywhere in the core


def test_a_channel_keeps_its_data_round_a_patch_only_another_channel_clipped():
    """The Seestar M 31's nucleus clips in red to 14 px and never in green: green's measurements there are data."""
    N = 200
    yy, xx = np.mgrid[:N, :N]
    sat = np.zeros((3, N, N), bool)
    r = np.hypot(xx - 100, yy - 100)
    sat[0, r <= 14.0] = True
    sat[2, r <= 4.0] = True
    z = V.ImageMMEngine._bloom_zone(sat, 4)
    assert not z[1].any()
    assert z[0][(r > 14.5) & (r <= 23.0)].all() and not z[0][r > 26].any()                     # 14 + 0.75 x 14
    assert z[2][(r > 4.5) & (r <= 8.0)].all() and not z[2][r > 9.5].any()                      # blue: its own 4 px core


def test_a_galaxy_nucleus_has_no_bloom_zone():
    N = 200
    yy, xx = np.mgrid[:N, :N]
    sat = np.zeros((3, N, N), bool)
    big = np.hypot(xx - 60, yy - 60) <= 12.0
    sat[:, big] = True
    small = np.hypot(xx - 150, yy - 150) <= 2.0
    sat[1, small] = True
    z = V.ImageMMEngine._bloom_zone(sat, 4, extended=big)
    r_small = np.hypot(xx - 150, yy - 150)
    assert not z[:, np.hypot(xx - 60, yy - 60) < 40].any()                                     # the nucleus: data to its edge
    assert z[1][(r_small > 2.5) & (r_small <= 6.0)].all()                                       # a star keeps its zone


def test_point_initial_tells_a_nucleus_from_a_star_without_moving_anything(tmp_path):
    C, N = 1, 300
    yy, xx = np.mgrid[:N, :N].astype(np.float64)
    gal = 0.1 + 3.0 * np.exp(-np.hypot(xx - 150, yy - 150) / 9.0)
    img = np.minimum(gal, 1.5)[None]
    core = gal >= 1.5
    x = torch.from_numpy(img.astype(np.float32))[None]
    ext = np.zeros(core.shape, bool)
    out, _ = V.ImageMMEngine._point_initial(x, core, 1.0, [1e-3] * C, extended=ext, move=False)
    assert ext[core].all() and torch.equal(out, x)
