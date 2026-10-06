"""Shared benchmark harness for denoising / deconvolution experiments.

Ground truth never exists for real astro data, but the two half-stacks give an
unbiased referee: a method sees ONLY half A (and is trained only on the
training bands), then its estimate x̂ is compared with half B on held-out bands.
Because B's noise is independent of everything the method saw,

    E |x̂ - B|² = E |x̂ - x|² + σ_B²

so subtracting B's (measured) noise variance gives the true error of x̂.
For deconvolution the same holds after re-blurring: E|k*x̂ - B|² - σ_B².
Scores are reported relative to the noise of a single half-stack
("error / σ²"; 1.0 = no better than the raw half, lower is better) and as dB.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from astropy.io import fits  # noqa: E402

DATASETS = {
    # stacks present in output/ (October 2026); a name that is not listed here is taken
    # as a path to a stack directory (half_a.fits, half_b.fits, stack.fits, coverage.fits)
    "M42": "output/DWARF_RAW_TELE_M_42_EXP_15_GAIN_60_2026-10-04-04-49-20-108+1-debc1b",
    "M31": "output/M_31_sub-5f1bec",
    "NGC7000": "output/NGC_7000_North_American_Nebula-75f7ec",
    "C33": "output/DWARF_RAW_TELE_C_33_EXP_15_GAIN_60_2026-09-30-22-11-12-896+1-428802",
    # earlier benchmarks (results_*.json), stacks no longer in output/
    "IC5070": "output/IC_5070_sub-a9642d",
    "M27": "output/M_27_sub-2c99a3",
    "IC5070g": "experiments/cache/stacks/IC_5070_sub-a9642d",
    "M27g": "experiments/cache/stacks/M_27_sub-2c99a3",
}
BAND = 256          # width of train / test bands
CROP = 1536         # benchmark crop (per side) at 1x; scaled for drizzled stacks


def _fits(path):
    d = fits.getdata(path).astype(np.float32)
    return np.ascontiguousarray(np.moveaxis(d, 0, -1)) if d.ndim == 3 and d.shape[0] == 3 else d


def load(name: str, crop: int | None = None):
    """Half stacks, full stack and coverage cropped to the deepest square region."""
    d = DATASETS.get(name, name)
    d = d if os.path.isabs(d) else os.path.join(ROOT, d)
    meta = json.load(open(os.path.join(d, "stack_meta.json")))
    cov = _fits(os.path.join(d, "coverage.fits"))
    s = int((crop or CROP) * meta.get("scale", 1.0) ** 0.5)   # 2x stacks: bigger crop, same sky fraction-ish
    s -= s % BAND
    import cv2
    c = cv2.blur(cov, (s // 4, s // 4))
    c[: s // 2], c[-s // 2:], c[:, : s // 2], c[:, -s // 2:] = 0, 0, 0, 0
    cy, cx = np.unravel_index(np.argmax(c), c.shape)
    y0, x0 = cy - s // 2, cx - s // 2
    sl = (slice(y0, y0 + s), slice(x0, x0 + s))
    a = _fits(os.path.join(d, "half_a.fits"))[sl].copy()
    b = _fits(os.path.join(d, "half_b.fits"))[sl].copy()
    full = _fits(os.path.join(d, "stack.fits"))[sl].copy()
    return dict(name=name, dir=d, sl=sl, a=a, b=b, full=full, cov=cov[sl].copy(), sat=meta["saturation"],
                scale=meta.get("scale", 1.0), filter=meta.get("filter", ""))


def split_masks(shape):
    """Vertical bands: every third band is held out for testing."""
    h, w = shape[:2]
    band = (np.arange(w) // BAND) % 3 == 1
    test = np.broadcast_to(band[None, :], (h, w)).copy()
    return ~test, test


class NoiseModel:
    """Local noise variance of ONE half stack, measured from the half-stack difference.

    var(p) = Gaussian-window mean of (A-B)^2 / 2.  A ~20 px window averages
    hundreds of pixels, so its correlation with any single pixel of B is negligible.
    """

    def __init__(self, a, b, signal=None, sigma=10.0):
        import cv2
        d2 = 0.5 * (a - b) ** 2
        v = cv2.GaussianBlur(d2, (0, 0), sigma)
        self.v = np.maximum(v, np.percentile(v, 1, axis=(0, 1)) * 0.5).astype(np.float32)

    def var(self, signal=None):
        return self.v


def score(xhat, bench, blur=None, key="test"):
    """Unbiased error of x̂ on the held-out bands, in units of one half-stack's noise variance.

    ``blur``: optional callable applied to x̂ first (the forward model k*x̂ for deconvolution).
    Returns dict with linear-domain (inverse-variance weighted) and stretched-domain scores.
    """
    a, b = bench["a"], bench["b"]
    test = bench[key]
    pred = blur(xhat) if blur is not None else xhat
    var = bench["nm"].var(bench["ref"])
    ok = test & bench["valid"]
    e_lin = ((pred - b) ** 2 / var)[ok].mean() - 1.0
    # stretched domain (what the eye sees after an asinh-type stretch): faint signal matters
    st = bench["stab"]
    ps, bs, as_ = st.fwd(pred), st.fwd(b), st.fwd(a)
    noise_s = 0.5 * ((as_ - bs) ** 2)[ok].mean()
    e_str = ((ps - bs) ** 2)[ok].mean() - noise_s
    return {"lin": float(e_lin), "lin_db": float(-10 * np.log10(max(e_lin, 1e-6))),
            "str": float(e_str / noise_s), "str_db": float(-10 * np.log10(max(e_str / noise_s, 1e-6)))}


def prepare(name: str, crop: int | None = None):
    from astrophoto.denoise import Stabiliser
    from scipy.ndimage import binary_erosion
    bench = load(name, crop)
    train, test = split_masks(bench["a"].shape)
    good = bench["cov"] >= 0.6 * np.percentile(bench["cov"], 90)
    unsat = bench["full"].max(-1) < 0.5 * bench["sat"]
    unsat = binary_erosion(unsat, iterations=6)
    bench["train"], bench["test"] = train & good, test
    bench["valid"] = good & unsat
    bench["stab"] = Stabiliser(bench["a"], bench["b"])
    # smooth signal estimate for the variance model (full stack, lightly blurred)
    import cv2
    bench["ref"] = cv2.GaussianBlur(bench["full"], (0, 0), 1.5)
    bench["nm"] = NoiseModel(bench["a"], bench["b"], bench["ref"])
    return bench


# ----------------------------------------------------------------------------- noise law, synthetic twin
def variance_law(a, b, full, nbins=24):
    """Per-channel noise variance of one half as a function of signal level above the sky,
    var = c0 + c1 * max(level - sky, 0), from (A-B)²/2 against the full stack's level.

    c0 is the robust variance at the sky level; c1 the slope fitted to the binned robust
    variances between the sky and sky + 4 sigma.  Beyond that the half-stack difference is
    dominated by structure (registration and seeing differences between the halves, defects
    left in one half), not by pixel noise, so the slope is an upper bound on the photon term.
    Returns (c0, c1, sky), arrays of shape (3,)."""
    c0, c1, skys = np.zeros(3, np.float32), np.zeros(3, np.float32), np.zeros(3, np.float32)
    for c in range(3):
        lv, d2 = full[..., c].ravel()[::7], (0.5 * (a[..., c] - b[..., c]) ** 2).ravel()[::7]
        sky = float(np.median(lv))
        v0 = float(np.median(d2[np.abs(lv - sky) < 0.5 * np.sqrt(np.median(d2) / 0.4549)]) / 0.4549)
        sg = np.sqrt(v0)
        edges = np.linspace(sky, sky + 4 * sg, nbins + 1)
        xs, ys = [], []
        for i in range(nbins):
            m = (lv >= edges[i]) & (lv < edges[i + 1])
            if m.sum() > 200:
                xs.append(float(np.median(lv[m])) - sky)
                ys.append(float(np.median(d2[m])) / 0.4549 - v0)   # median of a chi²_1 variate = 0.4549 x variance
        slope = float(np.dot(xs, ys) / max(np.dot(xs, xs), 1e-9)) if xs else 0.0
        c0[c], c1[c], skys[c] = v0, max(slope, 0.0), sky
    return c0, c1, skys


def synthetic(bench, seed=0, blur=1.0):
    """Synthetic twin of a benchmark with exact truth: truth = the full stack lightly smoothed
    (its own noise is 1/sqrt(2) of a half's and the smoothing takes most of the rest), and two
    halves = truth + independent Gaussian noise following the measured variance law of the
    data (``variance_law``).  Everything else (masks, stabiliser, noise model) is rebuilt from
    the synthetic halves, so a method sees exactly what it would see on the real data."""
    import cv2
    from astrophoto.denoise import Stabiliser
    rng = np.random.default_rng(seed)
    c0, c1, sky = variance_law(bench["a"], bench["b"], bench["full"])
    truth = cv2.GaussianBlur(bench["full"], (0, 0), blur).astype(np.float32)
    var = np.maximum(c0 + c1 * np.maximum(truth - sky, 0.0), 1e-6).astype(np.float32)
    sd = np.sqrt(var)
    a = (truth + sd * rng.standard_normal(truth.shape, dtype=np.float32)).astype(np.float32)
    b = (truth + sd * rng.standard_normal(truth.shape, dtype=np.float32)).astype(np.float32)
    syn = dict(bench)
    syn.update(a=a, b=b, full=0.5 * (a + b), truth=truth, var_true=var, law=(c0, c1, sky),
               name=bench["name"] + "-syn", synthetic=True)
    syn["stab"] = Stabiliser(a, b)
    syn["ref"] = cv2.GaussianBlur(syn["full"], (0, 0), 1.5)
    syn["nm"] = NoiseModel(a, b, syn["ref"])
    return syn


# ----------------------------------------------------------------------------- bias, detection, photometry
def half_sigma(bench):
    """Sky noise of one half per channel (the stabiliser's robust estimate)."""
    return bench["stab"].sigma


def bias_metrics(est, bench, key="test", edges=(0, 1, 2, 4, 8, 16, 32, 64, 128, 1e9)):
    """Exact-truth metrics (synthetic benches): rms error, faint-region error and the signed
    bias of ``est`` by true signal level above the sky, in units of one half's sky sigma.

    * rmse_sigma: rms(est - truth) on valid test pixels / sigma
    * faint_rmse_sigma, faint_bias_sigma: same on pixels whose true signal is < 4 sigma
      above the sky (faint emission and sky): bias here is what a stretch shows
    * bias_by_level: mean(est - truth)/sigma for signal in [edges[i], edges[i+1]) sigma above the
      sky, with the pixel count: a Jensen-gap bias shows up as a level-dependent offset
    """
    t, sg = bench["truth"], half_sigma(bench)
    ok = bench[key] & bench["valid"]
    sky = np.median(t[ok], axis=0)
    out, bins = {}, []
    lvl = ((t - sky) / sg).mean(-1)                         # luminance level in sigma
    d = (est - t) / sg
    out["rmse_sigma"] = float(np.sqrt((d[ok] ** 2).mean()))
    faint = ok & (lvl < 4)
    out["faint_rmse_sigma"] = float(np.sqrt((d[faint] ** 2).mean()))
    out["faint_bias_sigma"] = float(d[faint].mean())
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ok & (lvl >= lo) & (lvl < hi)
        if m.sum() >= 50:
            bins.append({"lo": lo, "hi": hi if hi < 1e8 else None, "n": int(m.sum()),
                         "bias": round(float(d[m].mean()), 4), "rms": round(float(np.sqrt((d[m] ** 2).mean())), 4)})
    out["bias_by_level"] = bins
    return out


def _lum(img):
    return np.ascontiguousarray(img.mean(-1), np.float32)


def _detect(L, thresh_abs, minarea=4):
    """sep sources of a luminance image above an ABSOLUTE threshold (after a 64 px mesh background)."""
    import sep
    bkg = sep.Background(L, bw=64, bh=64)
    sub = L - bkg.back()
    objs = sep.extract(sub, thresh_abs, minarea=minarea)
    return objs, sub


def score_region(bench):
    """Pixels the catalogue scores use: valid test bands on real data (the reference is half B,
    which the network was trained towards on the training bands), all valid pixels on a
    synthetic bench (the reference is the truth, which no method ever sees)."""
    return bench["valid"] if bench.get("synthetic") else (bench["test"] & bench["valid"])


def reference_catalog(bench, thresh_sigma=4.0, faint_peak=8.0, minarea=4):
    """Reference sources for the detection / photometry scores.

    Synthetic: sources of the noiseless truth with peak >= 1.5 sigma (one half's sky noise).
    Real: sources of half B above ``thresh_sigma`` x sigma; B's noise is independent of
    everything a method sees, so a noise peak of A is never confirmed by the reference.
    Returns dict(x, y, flux, peak_sigma, faint) inside ``score_region`` away from the
    borders; ``faint``: peak below ``faint_peak`` sigma in the reference image."""
    sg = float(half_sigma(bench).mean())
    ok = score_region(bench)
    if bench.get("synthetic"):
        objs, sub = _detect(_lum(bench["truth"]), 1.5 * sg, minarea)
    else:
        objs, sub = _detect(_lum(bench["b"]), thresh_sigma * sg, minarea)
    peak = objs["peak"] / sg
    h, w = ok.shape
    ix, iy = np.clip(np.round(objs["x"]).astype(int), 0, w - 1), np.clip(np.round(objs["y"]).astype(int), 0, h - 1)
    keep = ok[iy, ix] & (objs["x"] > 8) & (objs["x"] < w - 9) & (objs["y"] > 8) & (objs["y"] < h - 9)
    return {"x": objs["x"][keep], "y": objs["y"][keep], "flux": objs["flux"][keep], "peak_sigma": peak[keep],
            "faint": (peak < faint_peak)[keep]}


def detection_metrics(est, bench, ref, thresh_sigma=2.0, match_r=3.0):
    """AstroSURE-style (arXiv:2604.16793) detection rate / false-alarm rate of ``est``.

    Sources are extracted from ``est`` above the SAME absolute threshold for every method,
    ``thresh_sigma`` x one half's sky noise (a denoised image is not allowed to lower the bar
    by its own, smaller rms).  DR = fraction of reference sources with a detection within
    ``match_r`` px; FAR = fraction of detections with no reference source within ``match_r``.
    Both over ``score_region``; ``*_faint`` for the faint reference subset."""
    from scipy.spatial import cKDTree
    sg = float(half_sigma(bench).mean())
    ok = score_region(bench)
    h, w = ok.shape
    objs, _ = _detect(_lum(est), thresh_sigma * sg, minarea=4)
    ix, iy = np.clip(np.round(objs["x"]).astype(int), 0, w - 1), np.clip(np.round(objs["y"]).astype(int), 0, h - 1)
    keep = ok[iy, ix] & (objs["x"] > 8) & (objs["x"] < w - 9) & (objs["y"] > 8) & (objs["y"] < h - 9)
    dx, dy = objs["x"][keep], objs["y"][keep]
    out = {"n_det": int(len(dx)), "n_ref": int(len(ref["x"])), "n_ref_faint": int(ref["faint"].sum())}
    if len(dx) == 0 or len(ref["x"]) == 0:
        out.update(dr=0.0, dr_faint=0.0, far=1.0 if len(dx) else 0.0)
        return out
    dt = cKDTree(np.stack([dx, dy], 1))
    d_ref, _ = dt.query(np.stack([ref["x"], ref["y"]], 1))
    hit = d_ref <= match_r
    rt = cKDTree(np.stack([ref["x"], ref["y"]], 1))
    d_det, _ = rt.query(np.stack([dx, dy], 1))
    out["dr"] = float(hit.mean())
    out["dr_faint"] = float(hit[ref["faint"]].mean()) if ref["faint"].any() else None
    out["far"] = float((d_det > match_r).mean())
    return out


def flux_metrics(est, bench, ref, r=3.0, ann=(6.0, 9.0)):
    """STAR-style (arXiv:2507.16385) flux error: circular-aperture flux (radius ``r`` px, local
    annulus background) of the reference sources on ``est`` against the reference image
    (the truth, or half B whose noise is independent of the method's input).  Reports the
    median relative error |F/F_ref - 1| and the median signed bias F/F_ref - 1, over all and
    over the faint reference sources.  On real data the raw-A row sets the noise floor."""
    import sep
    sg = float(half_sigma(bench).mean())
    refimg = _lum(bench["truth"]) if bench.get("synthetic") else _lum(bench["b"])
    x, y = ref["x"], ref["y"]
    if len(x) == 0:
        return {}
    fe, _, _ = sep.sum_circle(_lum(est), x, y, r, bkgann=ann, subpix=5)
    fr, _, _ = sep.sum_circle(refimg, x, y, r, bkgann=ann, subpix=5)
    pos = fr > 3 * sg * np.sqrt(np.pi * r * r)        # reference flux at least 3 sigma in the aperture
    rel = fe[pos] / fr[pos] - 1
    out = {"n_phot": int(pos.sum())}
    if pos.sum() >= 5:
        out["flux_err"] = float(np.median(np.abs(rel)))
        out["flux_bias"] = float(np.median(rel))
        f = ref["faint"][pos]
        if f.sum() >= 5:
            out["flux_err_faint"] = float(np.median(np.abs(rel[f])))
            out["flux_bias_faint"] = float(np.median(rel[f]))
    return out


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.dt = time.time() - self.t
