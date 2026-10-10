"""Checks of the deconvolution network's target-resolution pieces in astrophoto/denoise.py:
the target Gaussian, the split of a PSF into s * r (split_target) and the simulated extended
sources of the source term (_StarTerm._extended).

    python experiments/test_denoise.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from astrophoto import denoise as D  # noqa: E402
from astrophoto.exposures import moffat_image  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def centroid(img):
    n = img.shape[-1]
    yy, xx = np.mgrid[:n, :n] - n // 2
    return float((img * yy).sum() / img.sum()), float((img * xx).sum() / img.sum())


def conv(z, k):
    """True convolution z * k, same size (cv2.filter2D correlates: flip the kernel)."""
    return cv2.filter2D(z.astype(np.float64), -1, k[::-1, ::-1].astype(np.float64).copy(), borderType=cv2.BORDER_CONSTANT)


# --- the target Gaussian: unit sum, centred where asked
g = D.gauss_kernel(3.0, 31, 0.3, -0.2)
cy, cx = centroid(g)
check("gauss_kernel unit sum and sub-pixel centre", abs(g.sum() - 1) < 1e-5 and abs(cy - 0.3) < 1e-3 and abs(cx + 0.2) < 1e-3,
      f"sum {g.sum():.6f} centre ({cy:.4f}, {cx:.4f})")

# --- split_target on an asymmetric (elongated, off-centre halo) PSF: s >= 0 and s * r = K
n = 61
core = moffat_image([1.0, 0.0, 0.0, 4.5, 3.0, 0.6, 2.8], n)
halo = moffat_image([0.04, 1.5, -1.0, 9.0, 8.0, 0.0, 2.2], n)
K = (core + halo).astype(np.float64)
K /= K.sum()
for fw in (2.0, 3.0):
    s, err = D.split_target(K[None], fw)
    s = s[0].astype(np.float64)
    back = conv(s, D.gauss_kernel(fw, n))
    rr = np.hypot(*(np.mgrid[:n, :n] - n // 2))
    core_err = np.sqrt(((back - K)[rr < 8] ** 2).mean()) / K.max()
    check(f"split_target fwhm {fw}: s * r reproduces the PSF", core_err < 0.01 and err < 0.05,
          f"core rms {core_err:.4f} of peak, total rel {err:.4f}")
    # the asymmetry stays in s (not symmetrised away): same centroid offset as K
    dk, ds = np.array(centroid(K)), np.array(centroid(s))
    check(f"split_target fwhm {fw}: keeps the PSF's asymmetry", np.abs(dk - ds).max() < 0.05,
          f"centroid K {np.round(dk, 3)} s {np.round(ds, 3)}")


# --- the Moffat target: round, unit sum, a halo kept, and split exactly
t = D.moffat_target(3.0)
r = D.target_kernel(t, n)
rr = np.hypot(*(np.mgrid[:n, :n] - n // 2))
prof = lambda z, a: float(z[(rr >= a - 0.5) & (rr < a + 0.5)].mean() / z.max())
half = [a for a in np.arange(0, 10, 0.05) if np.interp(a, [0, 1, 2, 3, 4], [prof(r, q) for q in range(5)]) < 0.5][0]
check("moffat_target: FWHM as asked", abs(2 * half - 3.0) < 0.35, f"FWHM {2 * half:.2f} px")
check("moffat_target: keeps a halo (Gaussian: none)", prof(r, 6) > 5 * prof(D.gauss_kernel(3.0, n), 6) + 1e-3,
      f"6 px: {prof(r, 6):.4f} of peak")
s, err = D.split_target(K[None], t)
back = conv(s[0].astype(np.float64), r)
core_err = np.sqrt(((back - K)[rr < 8] ** 2).mean()) / K.max()
check("split_target with the Moffat target reproduces the PSF", core_err < 0.01, f"core rms {core_err:.4f} of peak")

# --- simulated extended sources: input e * psf and target e * r hold the same flux at the same place
class _Stab:
    k, sigma, bg = 1.0, np.ones(3, np.float32), np.zeros(3, np.float32)


class _Net:
    def to(self, d):
        return self

    def eval(self):
        return self

    def parameters(self):
        return []


Kc = np.stack([K.astype(np.float32)] * 3)
st = D._StarTerm({"net": _Net(), "flux": (10.0, 1000.0), "colours": np.ones((1, 3)), "c1": np.ones(3), "c0": np.ones(3),
                  "psf_fwhm": 9.0}, _Stab(), Kc[:, None, None], lambda K_, y, x: K_[:, 0, 0], 3.0, 3,
                 np.random.default_rng(0), "cpu")
worst_f, worst_c = 0.0, 0.0
for i in range(20):
    a_, t_ = st._extended(Kc, 0.25, -0.4)
    fa, ft = a_.sum((1, 2)), t_.sum((1, 2))
    worst_f = max(worst_f, float(np.abs(fa / ft - 1).max()))
    worst_c = max(worst_c, float(np.abs(np.array(centroid(a_.mean(0))) - np.array(centroid(t_.mean(0))) - np.array(centroid(K))).max()))
check("extended sources: input and target fluxes agree", worst_f < 0.01, f"worst {worst_f:.4f}")
check("extended sources: target centred where the PSF puts the light", worst_c < 0.05, f"worst {worst_c:.4f} px")

print(f"\n{len(FAILS)} failure(s)" + (": " + ", ".join(FAILS) if FAILS else ""))
sys.exit(1 if FAILS else 0)
