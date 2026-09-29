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


if __name__ == "__main__":
    t0 = time.time()
    ALL = [test_halo_neutral, test_stretch_scale]
    chosen = [f for f in ALL if not sys.argv[1:] or f.__name__ in sys.argv[1:]]
    for f in chosen:
        f()
    print(f"\n{len(FAILS)} failed: {FAILS}  ({time.time() - t0:.0f}s)")
