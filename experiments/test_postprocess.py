"""Checks of the display stage (astrophoto/postprocess.py) on synthetic images.

    python experiments/test_postprocess.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from astrophoto.postprocess import extract_ha_oiii, luminance, neutralize_star_halos  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def halo_scene(H=1800, W=1800, seed=0):
    """A bright star whose halo (r^-3, out to ~100 px) is 25 % stronger in G and B than in R -
    the dual-band filter halo of the Seestar - with a red knot inside the halo, faint stars and
    sky noise, as a linear, background-neutralised image (0..1)."""
    rs = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :W].astype(np.float64)
    img = np.full((H, W, 3), 2e-4) + rs.normal(size=(H, W, 3)) * 3e-5
    cy, cx = H / 2, W / 2
    r = np.hypot(yy - cy, xx - cx)
    core = 0.9 * np.exp(-r ** 2 / (2 * 1.8 ** 2))
    halo = 2e-2 * (1 + (r / 6.0) ** 2) ** -1.5
    for c, k in enumerate((1.0, 1.25, 1.25)):
        img[..., c] += core + k * halo
    for _ in range(1200):                                          # field stars
        y, x = rs.uniform(20, H - 20, 2)
        f = 10 ** rs.uniform(-3.5, -1.5)
        img += (f * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.8 ** 2)))[..., None]
    knot = 3e-3 * np.exp(-((yy - cy - 45) ** 2 + (xx - cx) ** 2) / (2 * 3.0 ** 2))
    img[..., 0] += knot                                             # an Ha knot inside the halo
    return img.astype(np.float32), (cy, cx), (cy + 45, cx), halo


def test_halo_neutral():
    img, (cy, cx), (ky, kx), halo = halo_scene()
    out, W = neutralize_star_halos(img, 0.96, noise_ref=3e-5)
    H, Wd = img.shape[:2]
    yy, xx = np.mgrid[:H, :Wd]
    r = np.hypot(yy - cy, xx - cx)
    # the visible halo: its excess at least twice the pixel noise (12 - 40 px)
    ring = (r > 12) & (r < 40) & (np.hypot(yy - ky, xx - kx) > 12)
    col = lambda z: np.median(z[ring] - np.median(z[r > 500], 0), 0)
    before, after = col(img), col(out)
    gb_r = lambda c: (0.5 * (c[1] + c[2])) / c[0]
    knot = np.hypot(yy - ky, xx - kx) < 3
    kb = (img[knot][:, 0] - img[knot][:, 1]).mean()
    ka = (out[knot][:, 0] - out[knot][:, 1]).mean()
    check("halo colour neutralised out to the halo's measured reach",
          abs(gb_r(after) - 1) < 0.03 and abs(gb_r(before) - 1.25) < 0.05,
          f"(G+B)/2 / R in the halo 12-40 px: before {gb_r(before):.3f}, after {gb_r(after):.3f}")
    check("a red knot inside the halo keeps its colour", ka > 0.9 * kb, f"R - G at the knot: before {kb:.2e}, after {ka:.2e}")
    ha, oiii = extract_ha_oiii(out, unmix=True, neutral=W)
    ring2 = ring & (r < 40)
    hx = np.median(ha[ring2]) - np.median(ha[r > 500])
    ox = np.median(oiii[ring2]) - np.median(oiii[r > 500])
    ha0, o0 = extract_ha_oiii(out, unmix=True)
    ox0 = np.median(o0[ring2]) - np.median(o0[r > 500])
    check("palette: the neutral halo stays neutral (no OIII gain / unmixing on it)", abs(ox / hx - 1) < 0.1,
          f"OIII / Ha excess in the halo {ox / hx:.3f} (without the protection {ox0 / hx:.3f})")


def test_stretch_scale():
    """A restoration's sky is exactly the pedestal over much of the field (MAD 0): the stretch
    must stay sane there, and a full-resolution render must get the same curve as a downsampled
    preview (NGC 6960: D = 6.7e5 at full resolution against 300 in the preview)."""
    import cv2
    import astrophoto.postprocess as P
    rs = np.random.default_rng(3)
    H, W = 2400, 1600
    ped, noise = 3e-3, 1e-3                                    # pedestal = 3 x the coadd's noise
    L = np.full((H, W), ped, np.float32)
    sp = rs.random((H, W)) < 0.15                              # sparse positive speckle
    L[sp] += rs.exponential(2e-3, sp.sum()).astype(np.float32)
    yy, xx = np.mgrid[:H, :W]
    L += (0.02 * np.exp(-((xx - 800) ** 2) / (2 * 150 ** 2))).astype(np.float32)   # a nebula band
    res = []
    for f in (1.0, 0.625):
        Lf = L if f == 1 else cv2.resize(L, (int(W * f), int(H * f)), interpolation=cv2.INTER_AREA)
        fp = min(1.0, P._PROXY_SIZE / max(Lf.shape))
        res.append(P.solve_stretch(P._stretch_proxy(Lf), 0.12, 2.0, noise * f * fp))
    (bp1, D1, _), (bp2, D2, _) = res
    check("stretch: sane on an exactly flat restored sky, same curve at full and preview scale",
          D1 < 5e3 and abs(np.log(D1 / D2)) < 0.15 and abs(bp1 - bp2) < 0.1 * noise,
          f"full: bp {bp1:.5f} D {D1:.1f}; preview: bp {bp2:.5f} D {D2:.1f}")


def test_restored_denoise():
    """The display's starlet shrinkage on a restoration: with the restoration's own noise
    realisation (the N2N residual) it removes that noise and keeps restored detail far below the
    coadd's per-pixel noise, which the coadd-calibrated shrinkage erased."""
    import astrophoto.postprocess as P
    rs = np.random.default_rng(5)
    H = W = 512
    yy, xx = np.mgrid[:H, :W]
    # fine filaments: amplitude 1/4 of the coadd noise, width ~1.5 px, on a faint nebula
    fil = np.zeros((H, W), np.float32)
    for _ in range(40):
        y, x, t = rs.uniform(0, H), rs.uniform(0, W), rs.uniform(0, np.pi)
        d = np.abs((yy - y) * np.cos(t) - (xx - x) * np.sin(t))
        fil += np.exp(-d ** 2 / (2 * 1.5 ** 2)) * (np.hypot(yy - y, xx - x) < 60)
    nref, rnoise = 4e-3, 1.5e-4                    # coadd per-pixel noise; the restoration's residual noise
    sig = 2e-3 + 0.25 * nref * np.minimum(fil, 1)
    img = sig + rs.normal(size=(H, W)) * rnoise
    resid = (rs.normal(size=(H, W)) * rnoise).astype(np.float32)
    bp, D, b, sp, cp = 1e-3, 300.0, 2.0, 0.0, 0.7
    st = lambda z: P.rgb_to_oklab(P.apply_stretch(np.repeat(z[..., None], 3, -1).astype(np.float32), bp, D, b, sp, cp))
    lab = st(img)
    truth = st(sig)[..., 0]
    bgl = float(np.median(img))
    noise_L = P.rgb_to_oklab(P.apply_stretch(np.repeat(np.clip(bgl + resid, 0, 1)[..., None], 3, -1), bp, D, b, sp, cp))[..., 0]
    rsn = np.random.default_rng(0)
    patch = np.clip(bgl + rsn.normal(size=(256, 256)).astype(np.float32) * nref, 0, 1)
    pl_ = P.rgb_to_oklab(np.repeat(P.apply_stretch(patch[..., None].repeat(3, -1), bp, D, b, sp, cp)[..., :1], 3, -1))[..., 0]
    floors = np.array([P.mad_sigma(dd) for dd in P.atrous(pl_, 4)[0]])
    new = P._luminance_denoise_lab(lab.copy(), 0.8, None, noise_L)[..., 0]
    old = P._luminance_denoise_lab(lab.copy(), 0.8, floors)[..., 0]
    fine = lambda z: sum(P.atrous(z, 2)[0])                 # scales 1-2 (~1-4 px)
    kept = lambda z: float(np.sum(fine(z) * fine(truth)) / np.sum(fine(truth) ** 2))
    err = lambda z: float(np.sqrt(np.mean((z - truth) ** 2)))
    check("restored display denoise: fine detail below the coadd noise survives",
          kept(new) > 3 * kept(old) and err(new) < 0.8 * err(old),
          f"fine-scale detail kept: {kept(new):.2f} (coadd-calibrated: {kept(old):.2f}); rms error vs truth "
          f"{err(new):.2e} (coadd-calibrated: {err(old):.2e})")
    check("restored display denoise: the residual noise is still reduced", err(new) < 0.9 * err(lab[..., 0]),
          f"rms error vs truth: input {err(lab[..., 0]):.2e}, output {err(new):.2e}")


def test_stretch_methods():
    """Every stretch algorithm is a monotone curve through (0, 0) and (1, 1), and the solver puts
    the background on the target level whichever is chosen."""
    import astrophoto.postprocess as P
    rs = np.random.default_rng(7)
    L = (2e-3 + rs.normal(size=(800, 800)) * 2e-4).astype(np.float32)
    L[300:500, 300:500] += 0.02                                          # some signal
    for m in P.STRETCH_METHODS:
        bp, D, sp = P.solve_stretch(L, 0.15, 2.0, method=m)
        x = np.linspace(0, 1, 1001)
        y = P.tone(x, D, 2.0, sp, m)
        bgv = float(np.median(P.tone_fast(np.clip((L - bp) / (1 - bp), 0, 1), D, 2.0, sp, m)))
        check(f"stretch {m}: monotone 0->0, 1->1, background on target",
              np.all(np.diff(y) >= -1e-6) and abs(y[0]) < 1e-5 and abs(y[-1] - 1) < 1e-4 and abs(bgv - 0.15) < 0.01,
              f"D {D:.3g}, background {bgv:.3f}")


def test_spcc_detection():
    """Sensor and filter detection are generic: from the camera name, a camera model number that
    differs from the sensor's, the sensor geometry, and the FILTER header - never one camera."""
    from astrophoto.spcc import detect_filter, detect_sensor
    avail = ["Sony_IMX585", "Sony_IMX571", "Sony_IMX462", "Sony_IMX662", "Sony_IMX294", "ZWO_Seestar_S50",
             "ZWO_Seestar_S30", "Canon_EOS600D"]
    cases = [("ZWO ASI585MC", 3840, 2160, 2.9, "Sony_IMX585"), ("ZWO ASI2600MC Pro", 6248, 4176, 3.76, "Sony_IMX571"),
             ("Seestar S50", 1080, 1920, 2.9, "ZWO_Seestar_S50"), ("Seestar S30", 1080, 1920, 2.9, "ZWO_Seestar_S30"),
             ("Seestar S50", 2160, 3840, 2.9, "Sony_IMX585"), ("", 4144, 2822, 4.63, "Sony_IMX294"),
             ("Canon EOS 600D", 5184, 3456, 4.3, "Canon_EOS600D"), ("", 1920, 1080, 2.9, None),
             ("Mystery cam", 1000, 1000, 5.0, None)]
    for inst, w, h, px, want in cases:
        got, how = detect_sensor(inst, w, h, px, avail)
        check(f"SPCC sensor: {inst or 'no name'} {w}x{h} {px} um -> {want}", got == want, f"got {got} ({how})")
    favail = ["UV-IR-Block", "ZWO_Seestar_LP", "OPTOLONG_L-PRO_Light_Pollution", "No_filter", "Baader_UHC-S"]
    for f, inst, want in [("IRCUT", "Seestar S50", "UV-IR-Block"), ("LP", "Seestar S50", "ZWO_Seestar_LP"),
                          ("L-Pro", "ZWO ASI585MC", "OPTOLONG_L-PRO_Light_Pollution"), ("", "ZWO ASI585MC", "UV-IR-Block"),
                          ("UHC-S", "", "Baader_UHC-S")]:
        got, how = detect_filter(f, inst, favail)
        check(f"SPCC filter: {f or 'none'} ({inst or 'no camera'}) -> {want}", got == want, f"got {got} ({how})")


def test_spcc_physics():
    """CCM89 extinction: A_V at V (550 nm) is 1 and A_B / A_V ~ 1.32; reddening makes a star redder
    through any R/G/B response."""
    from astrophoto import spcc
    a = spcc.ccm89(np.array([550.0, 440.0]))
    check("SPCC: CCM89 extinction law", abs(a[0] - 1) < 0.02 and abs(a[1] - 1.32) < 0.04, f"A550 {a[0]:.3f}, A440 {a[1]:.3f}")
    resp = np.stack([np.exp(-((spcc.WL - c) / 40) ** 2) for c in (610, 530, 460)])
    flat = np.ones_like(spcc.WL)
    c0 = spcc._counts(flat, resp)
    c1 = spcc._counts(flat * 10 ** (-0.4 * 2.0 * spcc.ccm89(spcc.WL)), resp)
    check("SPCC: reddening raises R/G and lowers B/G", (c1[0] / c1[1]) > (c0[0] / c0[1]) and (c1[2] / c1[1]) < (c0[2] / c0[1]))


def test_photometric_gains():
    """Stars with known predicted colours seen through unknown channel gains: the photometric
    calibration recovers the gains (the inverse of the camera's) to 1 %, though the stars' colours
    differ widely from white and from each other."""
    from astrophoto.spcc import photometric_gains
    rs = np.random.default_rng(11)
    H, W, N = 1500, 1500, 400
    x, y = rs.uniform(40, W - 40, N), rs.uniform(40, H - 40, N)
    g = rs.uniform(9, 14, N)
    expected = np.stack([10 ** rs.normal(0.1, 0.12, N), np.ones(N), 10 ** rs.normal(-0.15, 0.15, N)], 1)  # reddish field
    cam = np.array([0.55, 1.0, 0.8])                          # the camera's response relative to the calibrated colours
    img = np.full((H, W, 3), 1e-3, np.float32) + rs.normal(size=(H, W, 3)).astype(np.float32) * 1e-5
    yy, xx = np.mgrid[-12:13, -12:13]
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2 * 1.6 ** 2))
    psf /= psf.sum()
    for i in range(N):
        f = 10 ** (-0.4 * (g[i] - 9)) * 0.5
        xi, yi = int(x[i]), int(y[i])
        for c in range(3):
            img[yi - 12:yi + 13, xi - 12:xi + 13, c] += (f * expected[i, c] * cam[c] * psf).astype(np.float32)
    stars = {"x": np.floor(x), "y": np.floor(y), "g": g, "expected": expected}
    gains, info = photometric_gains(img, stars, np.zeros((H, W), bool), noise=1e-5)
    want = 1 / cam
    check("SPCC: gains recovered from catalogue colours", gains is not None and np.allclose(gains, want, rtol=0.01),
          f"gains {None if gains is None else np.round(gains, 4)}, want {np.round(want, 4)}; {info}")


def test_star_reduction_starless():
    """With a starless image given (the AI star remover's), star reduction 1 renders it without
    the stars, 0 keeps every star, and the sky away from stars is the same either way."""
    import astrophoto.postprocess as P
    rs = np.random.default_rng(13)
    H = W = 600
    yy, xx = np.mgrid[:H, :W]
    bg = np.full((H, W, 3), 2e-3, np.float32) + rs.normal(size=(H, W, 3)).astype(np.float32) * 1e-4
    bg[..., 0] += (0.01 * np.exp(-((xx - 300) ** 2 + (yy - 300) ** 2) / (2 * 120 ** 2))).astype(np.float32)
    stars = np.zeros_like(bg)
    pos = rs.uniform(30, W - 30, (60, 2))
    for x, y in pos:
        stars += (rs.uniform(0.01, 0.3) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.5 ** 2)))[..., None].astype(np.float32)
    lin = bg + stars
    base = {"_noise_ref": 1e-4, "_starless": bg, "sharpen": 0, "local_contrast": 0}
    out1 = P.nonlinear_stage(lin, {**base, "star_reduction": 1.0})
    out0 = P.nonlinear_stage(lin, {**base, "star_reduction": 0.0})
    ref = P.nonlinear_stage(bg, {**base, "star_reduction": 1.0})
    at = lambda im: np.mean([P.luminance(im)[int(y), int(x)] for x, y in pos])
    far = np.ones((H, W), bool)
    for x, y in pos:
        far &= np.hypot(xx - x, yy - y) > 10
    check("star reduction 1 = starless, 0 = every star",
          abs(at(out1) - at(ref)) < 0.02 and at(out0) > at(out1) + 0.2 and
          np.abs(P.luminance(out1)[far] - P.luminance(out0)[far]).mean() < 2e-3,
          f"at the stars: r=1 {at(out1):.3f} (starless {at(ref):.3f}), r=0 {at(out0):.3f}")


if __name__ == "__main__":
    t0 = time.time()
    ALL = [test_halo_neutral, test_stretch_scale, test_restored_denoise, test_stretch_methods, test_spcc_detection,
           test_spcc_physics, test_photometric_gains, test_star_reduction_starless]
    chosen = [f for f in ALL if not sys.argv[1:] or f.__name__ in sys.argv[1:]]
    for f in chosen:
        f()
    print(f"\n{len(FAILS)} failed: {FAILS}  ({time.time() - t0:.0f}s)")
