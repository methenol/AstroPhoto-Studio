"""Post-integration processing: linear stage + non-linear (display) stage.

Linear stage (full resolution, cached):
    auto-crop -> gradient/background extraction -> background neutralisation
    -> star-based white balance -> ML denoise blend -> self-supervised
    deconvolution network blend (or PSF-measured Richardson-Lucy fallback)
Non-linear stage (fast; runs on a preview scale or full resolution):
    star separation (mask + inpainting) -> auto-solved Generalized Hyperbolic
    Stretch -> narrowband palette (HOO / Foraxx dynamic) or colour-preserving
    RGB -> wavelet local contrast -> perceptual (OKLab) saturation -> SCNR
    -> star reduction and screen recombination -> final sharpening / curves
"""
from __future__ import annotations

import cv2
import numpy as np
import sep
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import maximum_filter, minimum_filter

DEFAULTS = {
    # linear stage
    "crop": True,
    "crop_threshold": 0.3,
    "background": True,
    "bg_method": "auto",        # auto | poly | rbf
    "bg_degree": 2,
    "white_balance": "auto",    # auto (spectrophotometric, Gaia) | stars | background | none
    "spcc_sensor": "auto",      # camera sensor curve (Siril SPCC database) - auto: from the FITS headers
    "spcc_filter": "auto",      # filter curve - auto: from the FILTER header
    "spcc_white_ref": "average_spiral_galaxy",  # white reference: average_spiral_galaxy | g2v
    "denoise": 0.95,            # blend with Noise2Noise result (0..1)
    "deconvolution": 0.7,       # 0..1 strength (AI deconvolution blend, or Richardson-Lucy fallback)
    "restored_resolution": 1.25,  # ImageMM: shown as the sky seen through a Gaussian g_sigma, sigma this
                                  # factor x the Eq. 11 sigma_0 (the restoration already is the sky through
                                  # g_sigma_0).  Finer than that is the latent's speckle, whose zero pixels
                                  # showed as dark pits at full resolution (1.25: 2133 -> 420 pits on
                                  # NGC 6960 at r = 1, stars still sharp)
    # non-linear stage
    "palette": "auto",          # auto | natural | hoo | foraxx | hoo_warm
    "oiii_boost": 1.0,
    "synthetic_luminance": 0.8, # narrowband: lightness from all-channel stretch (0..1)
    "stretch_method": "ghs",    # ghs | asinh | mtf (histogram transformation) | log - see STRETCH_METHODS
    "stretch": 0.16,            # target background brightness (0..1)
    "auto_stretch": True,       # scale the stretch by how much of the frame holds signal
    "hdr": 0.6,                 # compress bright large-scale structures (0..1)
    "contrast": 2.0,            # GHS local-intensity b (focus of contrast)
    "color_preservation": 0.7,  # 0 = per-channel stretch, 1 = luminance-ratio
    "star_separation": True,
    "star_removal": "auto",     # auto (the AI star remover once trained, else classic) | ai | classic
    "star_reduction": 0.35,
    "star_intensity": 0.9,
    "star_saturation": 1.0,
    "halo_suppress": 0.6,
    "star_color_preservation": 0.5,
    "saturation": 1.5,
    "chroma_denoise": 0.8,
    "luminance_denoise": 0.6,   # post-stretch wavelet shrinkage (0..1)      # OKLab chrominance smoothing (0..1)
    "oiii_unmix": True,         # remove Ha leakage/continuum from the OIII channel
    "local_contrast": 0.5,
    "scnr": 0.4,
    "black_point": 0.02,
    "brightness": 0.0,          # -1..1 midtone curve
    "sharpen": 0.25,
}

# Constants of star detection, star masks and gradient removal.  They are not processing settings
# (no slider): the experiment lab (lab/tasks.py: "Star detection & separation", "Gradient removal")
# tunes them - a better value found there is changed here, for every dataset.
STAR_DETECT = {
    "detect_sigma": 4.0,      # detection threshold, x the background rms
    "tophat_sigma": 8.0,      # threshold of the top-hat pass (stars on bright extended light)
    "compact_fwhm": 3.0,      # a bright object wider than this x the stars' FWHM is extended (not a star)
    "max_elongation": 2.0,    # a / b above this: not a star
    "concentration": 3.0,     # flux / (2 pi sigma^2 peak) above this: a galaxy nucleus or a knot
    "deblend_cont": 0.002,    # sep's deblending contrast
}
STAR_MASK = {
    "grow": 1.0,              # every mask radius x this
    "radius_scale": 1.35,     # radius x where a star's Gaussian core falls to the noise
    "min_radius_fwhm": 1.0,   # never below this x the stars' FWHM
    "halo_floor": 0.002,      # a bright star's halo ends below this fraction of its peak
    "halo_margin": 1.15,      # mask radius x the measured halo radius
    "grain": 1.8,             # inpainting: synthetic grain, x the local texture noise
}
BACKGROUND = {
    "grid": 24,               # sample tiles across the frame
    "min_sky_fraction": 0.4,  # a tile needs this much star-free area to be a sample
    "start_percentile": 75.0,  # the first fit uses the samples up to this luminance percentile
    "clip_high": 1.5,         # then keeps samples less than this x the residual rms above the model...
    "clip_low": 4.0,          # ...and less than this x below it
}


# ============================================================ helpers

def luminance(img: np.ndarray) -> np.ndarray:
    return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(np.float32)


def mad_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else 0.0


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / max(e1 - e0, 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def atrous(img: np.ndarray, levels: int):
    """Starlet (B3-spline a trous) wavelet decomposition. Returns (details[], residual)."""
    k1 = np.array([1, 4, 6, 4, 1], np.float32) / 16
    cur = img.astype(np.float32)
    details = []
    for j in range(levels):
        step = 2 ** j
        k = np.zeros(4 * step + 1, np.float32)
        k[::step] = k1
        nxt = cv2.sepFilter2D(cur, -1, k, k, borderType=cv2.BORDER_REFLECT)
        details.append(cur - nxt)
        cur = nxt
    return details, cur


# ------------------------------------------------------------- OKLab

def _srgb_to_linear_exact(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb_exact(c):
    c = np.maximum(c, 0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


_UG = np.linspace(0, 1, 8193, dtype=np.float64) ** 2
_LUT_S2L = _srgb_to_linear_exact(_UG).astype(np.float32)
_LUT_L2S = _linear_to_srgb_exact(_UG).astype(np.float32)


def _srgb_to_linear(c):
    return lut_sqrt(c, _LUT_S2L)


def _linear_to_srgb(c):
    return lut_sqrt(c, _LUT_L2S)


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]], np.float32)
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]], np.float32)
_M2i = np.linalg.inv(_M2.astype(np.float64)).astype(np.float32)
_M1i = np.linalg.inv(_M1.astype(np.float64)).astype(np.float32)


def rgb_to_oklab(rgb):
    """sRGB (0..1, display-referred) -> OKLab (Ottosson 2020), float32 throughout."""
    lin = _srgb_to_linear(rgb).astype(np.float32)
    lms = np.cbrt(cv2.transform(lin, _M1))
    return cv2.transform(lms.astype(np.float32), _M2)


def _oklab_to_linear(lab):
    lms = cv2.transform(np.ascontiguousarray(lab, np.float32), _M2i)
    return cv2.transform(lms * lms * lms, _M1i)


def oklab_to_rgb(lab, gamut_map: bool = True):
    """OKLab -> sRGB.  Out-of-gamut colours have their chroma reduced (hue and
    lightness kept) instead of being hard-clipped, which would flatten saturated
    regions into solid blotches."""
    lin = _oklab_to_linear(lab)
    if gamut_map:
        bad = ((lin < -1e-4) | (lin > 1 + 1e-4)).any(-1)
        if bad.any():
            sub = lab[bad].astype(np.float32)
            lo = np.zeros(len(sub), np.float32)
            hi = np.ones(len(sub), np.float32)
            for _ in range(10):  # bisection on chroma scale
                mid = (lo + hi) / 2
                t = sub.copy()
                t[:, 1:] *= mid[:, None]
                ll = _oklab_to_linear(t[None])[0]
                ok = ((ll >= -1e-4) & (ll <= 1 + 1e-4)).all(-1)
                lo = np.where(ok, mid, lo)
                hi = np.where(ok, hi, mid)
            sub[:, 1:] *= lo[:, None]
            lin[bad] = _oklab_to_linear(sub[None])[0]
    return np.clip(_linear_to_srgb(lin), 0, 1).astype(np.float32)


# ============================================================ linear stage

def auto_crop_box(coverage: np.ndarray, threshold: float = 0.5, min_aspect: float = 0.5,
                  max_aspect: float = 2.0):
    """Largest axis-aligned rectangle whose pixels all have coverage >= threshold.

    Searches a range of aspect ratios on a down-sampled bad-pixel map with an
    integral image (vectorised sliding window).  Among equal areas the more
    central rectangle wins.  Aspect limits avoid degenerate thin slivers.
    """
    ref = np.percentile(coverage[coverage > 0], 90) if (coverage > 0).any() else 1.0
    h, w = coverage.shape
    f = max(1, int(round(max(h, w) / 480)))
    small = cv2.resize((coverage >= threshold * ref).astype(np.float32), (w // f, h // f), interpolation=cv2.INTER_AREA)
    bad = (small < 0.999).astype(np.int32)
    sh, sw = bad.shape
    ii = np.pad(bad.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    # the target sits where the stack is deepest: coverage-weighted centroid
    cs = cv2.resize(coverage.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA) ** 4
    gy, gx = np.mgrid[0:sh, 0:sw]
    cy0, cx0 = (cs * gy).sum() / cs.sum(), (cs * gx).sum() / cs.sum()
    half_diag = 0.5 * np.hypot(sh, sw)
    best = None
    for aspect in np.geomspace(min_aspect, max_aspect, 25):  # height / width
        lo, hi = 1, min(sw, int(sh / aspect))
        found = None
        while lo <= hi:
            rw = (lo + hi) // 2
            rh = max(1, int(round(rw * aspect)))
            if rh > sh:
                hi = rw - 1
                continue
            cnt = ii[rh:, rw:] - ii[:-rh, rw:] - ii[rh:, :-rw] + ii[:-rh, :-rw]
            ys, xs = np.nonzero(cnt == 0)
            if len(ys):
                # most central valid placement
                d = (ys + rh / 2 - cy0) ** 2 + (xs + rw / 2 - cx0) ** 2
                k = int(np.argmin(d))
                found = (ys[k], xs[k], rh, rw, d[k])
                lo = rw + 1
            else:
                hi = rw - 1
        if found:
            y, x, rh, rw, d = found
            # area, penalised by how far the rectangle's centre is from the target
            key = rh * rw * (1 - 0.9 * min(1.0, np.sqrt(d) / half_diag))
            if best is None or key > best[0]:
                best = (key, (y, x, rh, rw))
    if best is None:
        return 0, h, 0, w
    y, x, rh, rw = best[1]
    # map back to full resolution, shrinking by one block for safety
    y0, x0 = (y + 1) * f if y > 0 else 0, (x + 1) * f if x > 0 else 0
    y1, x1 = min(h, (y + rh - 1) * f), min(w, (x + rw - 1) * f)
    return int(y0), int(y1), int(x0), int(x1)


def star_mask_simple(L: np.ndarray, k: float = 4.0, grow: int = 3) -> np.ndarray:
    """Fast star mask: small-scale positive structure, dilated."""
    small = L - cv2.medianBlur(np.ascontiguousarray(L, np.float32), 5)
    s = mad_sigma(small[::4, ::4])
    m = (small > k * s).astype(np.uint8)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    return m.astype(bool)


def background_model(img: np.ndarray, method: str = "auto", degree: int = 2, grid: int | None = None,
                     progress=None) -> tuple[np.ndarray, dict]:
    """Model sky background (light pollution gradient + vignetting residual).

    Samples a tile grid (stars masked), then iteratively fits a smooth surface
    while rejecting tiles that sit *above* the model (nebulosity, galaxies) –
    a lower-envelope fit that behaves like a careful DBE sample placement.
    """
    h, w, _ = img.shape
    B = BACKGROUND
    grid = int(grid or B["grid"])
    L = luminance(img)
    smask = star_mask_simple(L)
    nx = grid
    ny = max(4, int(round(grid * h / w)))
    ys = np.linspace(0, h, ny + 1).astype(int)
    xs = np.linspace(0, w, nx + 1).astype(int)
    pts, vals = [], []
    for i in range(ny):
        for j in range(nx):
            tile = img[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            tm = ~smask[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            if tm.mean() < B["min_sky_fraction"]:
                continue
            px = tile[tm]
            med = np.median(px, axis=0)
            # one sigma-clip round inside the tile
            sd = 1.4826 * np.median(np.abs(px - med), axis=0) + 1e-9
            ok = (np.abs(px - med) < 2.5 * sd).all(1)
            if ok.sum() > 20:
                med = np.median(px[ok], axis=0)
            pts.append(((ys[i] + ys[i + 1]) / 2 / h * 2 - 1, (xs[j] + xs[j + 1]) / 2 / w * 2 - 1))
            vals.append(med)
    pts = np.array(pts)
    vals = np.array(vals)
    if len(vals) < 6:
        # no (or too few) star-free sky tiles to fit: remove nothing rather than guess
        return np.zeros_like(img, dtype=np.float32), {"samples": int(len(vals)), "total": int(ny * nx),
                                                      "method": "none", "degree": 0,
                                                      "note": "too few star-free sky tiles: no gradient removed"}
    lum = vals @ np.array([0.2126, 0.7152, 0.0722])
    sel = lum <= np.percentile(lum, B["start_percentile"])
    if method == "auto":
        # the polynomial unless it cannot follow the sky: a vignetting rim, patchy residual glow
        # (C 33: bright edges and corners).  Both are fitted to the same lower-envelope samples and
        # the one whose sky residuals are clearly smaller is used
        res = {}
        for m_ in ("poly", "rbf"):
            bg_, inf_ = background_model(img, m_, degree, grid)
            sky_ = img - bg_
            tiles_, edge_ = [], []
            for i in range(ny):
                for j in range(nx):
                    tm_ = ~smask[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
                    if tm_.mean() >= B["min_sky_fraction"]:
                        tiles_.append(np.median(sky_[ys[i]:ys[i + 1], xs[j]:xs[j + 1]][tm_], axis=0))
                        edge_.append(i in (0, ny - 1) or j in (0, nx - 1))
            T_, e_ = np.array(tiles_), np.array(edge_)
            # per colour (a rim is often in one channel: C 33's red border): the sky tiles' scatter
            # (nebula tiles, far above, are cut robustly) plus the border's offset from the interior
            score = 0.0
            for c_ in range(T_.shape[1]):
                t_ = T_[:, c_]
                ok_ = t_ <= np.median(t_) + 3 * mad_sigma(t_)
                rim = (abs(np.median(t_[ok_ & e_]) - np.median(t_[ok_ & ~e_]))
                       if (ok_ & e_).any() and (ok_ & ~e_).any() else 0.0)
                score = max(score, mad_sigma(t_[ok_]) + rim)
            res[m_] = (score, bg_, inf_)
        pick = "rbf" if res["rbf"][0] <= res["poly"][0] else "poly"
        bg_, inf_ = res[pick][1], res[pick][2]
        inf_ = {**inf_, "auto": f"{pick} (sky residual poly {res['poly'][0]:.3g}, rbf {res['rbf'][0]:.3g})"}
        return bg_, inf_

    def fit(sel_, deg):
        if method == "rbf":
            model = RBFInterpolator(pts[sel_], vals[sel_], kernel="thin_plate_spline",
                                    smoothing=float(sel_.sum()) * 2.0)
            return model
        X = np.stack([pts[:, 1] ** a * pts[:, 0] ** b for a in range(deg + 1) for b in range(deg + 1 - a)], 1)
        beta, *_ = np.linalg.lstsq(X[sel_], vals[sel_], rcond=None)
        return lambda p: np.stack([p[:, 1] ** a * p[:, 0] ** b for a in range(deg + 1)
                                   for b in range(deg + 1 - a)], 1) @ beta

    deg = degree
    for _ in range(8):
        model = fit(sel, deg)
        pred = model(pts) @ np.array([0.2126, 0.7152, 0.0722])
        r = lum - pred
        s = mad_sigma(r[sel]) + 1e-9
        new_sel = (r < B["clip_high"] * s) & (r > -B["clip_low"] * s)
        if new_sel.sum() < 12:
            break
        if np.array_equal(new_sel, sel):
            break
        sel = new_sel
    # nebula-dominated fields: protect large-scale structure with a lower-order model
    if sel.mean() < 0.35 and deg > 1 and method == "poly":
        deg = 1
        model = fit(sel, deg)
    info = {"samples": int(sel.sum()), "total": int(len(sel)), "method": method, "degree": deg}
    lh, lw = max(8, h // 32), max(8, w // 32)
    gy, gx = np.mgrid[0:lh, 0:lw]
    grid_pts = np.stack([(gy + 0.5) / lh * 2 - 1, (gx + 0.5) / lw * 2 - 1], -1).reshape(-1, 2)
    low = model(grid_pts).reshape(lh, lw, 3).astype(np.float32)
    bg = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    return bg, info


def _extract(data: np.ndarray, thresh: float, err: float, **kw):
    """sep.extract that raises its threshold instead of failing when a very crowded field
    overflows the deblending limits (as analysis.measure_stars does)."""
    for k in range(4):
        try:
            return sep.extract(data, thresh * 2 ** k, err=err, **kw)
        except Exception as e:
            if "overflow" not in str(e) or k == 3:
                raise


def measure_star_colors(img: np.ndarray, sat: float, noise_floor: float = 0.0) -> np.ndarray | None:
    """Aperture photometry of unsaturated stars in each channel -> (N, 3) fluxes.
    ``noise_floor``: per-pixel noise of the data the image came from (for a restoration,
    whose own sky noise is ~0, the coadd's): detection significance is relative to the
    larger of it and the image's own background rms."""
    L = np.ascontiguousarray(luminance(img))
    bkg = sep.Background(L, bw=64, bh=64)
    objs = _extract(L - bkg.back(), 8.0, max(bkg.globalrms, noise_floor), minarea=5)
    if len(objs) < 10:
        return None
    ok = (objs["peak"] + np.median(bkg.back()) < 0.8 * sat) & (objs["flag"] == 0) & (objs["a"] / np.maximum(objs["b"], 1e-3) < 1.6)
    objs = objs[ok]
    if len(objs) < 10:
        return None
    # The aperture must hold the whole star in EVERY channel: refractors focus colours
    # differently (blue/violet halos), so an aperture sized to the luminance core
    # under-measures the widest channel and the white balance over-boosts it.
    subs = []
    r50 = []
    for c in range(3):
        ch = np.ascontiguousarray(img[..., c])
        sub = ch - sep.Background(ch, bw=64, bh=64).back()
        subs.append(sub)
        rr, _ = sep.flux_radius(sub, objs["x"], objs["y"], 6.0 * objs["a"], 0.5, subpix=5)
        r50.append(np.nanmedian(rr[np.isfinite(rr) & (rr > 0)]) if np.isfinite(rr).any() else np.nan)
    r = float(np.clip(5.0 * np.nanmax(r50), 3.0 * np.median(objs["a"]) + 1, 40.0))
    xy = np.stack([objs["x"], objs["y"]], 1)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(xy).query(xy, k=2)
    iso = d[:, 1] > r + 12
    if iso.sum() >= 10:
        objs = objs[iso]
    fl = []
    for sub in subs:
        f, _, _ = sep.sum_circle(sub, objs["x"], objs["y"], r, bkgann=(r + 4, r + 10))
        fl.append(f)
    fl = np.stack(fl, 1)
    return fl[(fl > 0).all(1)]


def estimate_psf(L: np.ndarray, sat: float, max_stars: int = 150, max_half: int = 20,
                 edge_sub: bool = True, bkg_box: int = 64, clip: bool = True) -> tuple[np.ndarray, float]:
    """Empirical PSF from isolated, unsaturated stars (sub-pixel re-centred median),
    out to 3 FWHM or ``max_half`` pixels.  ``edge_sub`` removes each cutout's border
    median (robust, but it also removes the PSF's faint wings); ``clip=False`` keeps
    the (noisy, slightly negative) far wings for model fitting."""
    Lc = np.ascontiguousarray(L, np.float32)
    bkg = sep.Background(Lc, bw=bkg_box, bh=bkg_box)
    sub = Lc - bkg.back()
    objs = sep.extract(sub, 10.0, err=bkg.globalrms, minarea=5)
    if len(objs) < 5:
        return None, float("nan")
    fwhm = 2 * sep.flux_radius(sub, objs["x"], objs["y"], 6 * objs["a"], 0.5, subpix=5)[0]
    fw = float(np.median(fwhm[np.isfinite(fwhm)]))
    half = int(np.clip(np.ceil(3.0 * fw), 5, max_half))
    good = ((objs["peak"] < 0.5 * sat) & (objs["flag"] == 0) &
            (objs["peak"] > 30 * bkg.globalrms) & (objs["a"] / np.maximum(objs["b"], 1e-3) < 1.4))
    xy = np.stack([objs["x"], objs["y"]], 1)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(xy).query(xy, k=2)
    h, w = L.shape
    inside = ((objs["x"] >= half + 2) & (objs["y"] >= half + 2) & (objs["x"] < w - half - 3) & (objs["y"] < h - half - 3))
    # isolation: no other 10-sigma detection within 3 cut-out half-widths; in crowded (Milky Way)
    # fields, and on the crops the experiment lab trains on, that can leave fewer than the 5 stars
    # needed, so it is relaxed step by step - the per-pixel median of the cut-outs rejects a
    # neighbour that appears in a minority of them
    for iso in (3.0, 2.0, 1.5):
        ok = good & inside & (d[:, 1] > iso * half)
        if ok.sum() >= 15:
            break
    idx = np.nonzero(ok)[0]
    idx = idx[np.argsort(-objs["peak"][idx])][:max_stars]
    cuts = []
    for i in idx:
        x, y = objs["x"][i], objs["y"][i]
        xi, yi = int(round(x)), int(round(y))
        if xi - half - 2 < 0 or yi - half - 2 < 0 or xi + half + 3 > w or yi + half + 3 > h:
            continue
        c = sub[yi - half - 2: yi + half + 3, xi - half - 2: xi + half + 3].astype(np.float32)
        M = np.float32([[1, 0, xi - x], [0, 1, yi - y]])
        c = cv2.warpAffine(c, M, (c.shape[1], c.shape[0]), flags=cv2.INTER_CUBIC)[2:-2, 2:-2]
        if edge_sub:
            edge = np.concatenate([c[0], c[-1], c[:, 0], c[:, -1]])
            c = c - np.median(edge)
        s = c.sum()
        if s > 0:
            cuts.append(c / s)
    if len(cuts) < 5:
        return None, fw
    psf = np.median(np.stack(cuts), 0)
    psf = (psf + np.rot90(psf, 1) + np.rot90(psf, 2) + np.rot90(psf, 3)) / 4
    psf = (psf + psf[::-1]) / 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    psf *= (xx ** 2 + yy ** 2) <= half ** 2
    if clip:
        psf = np.clip(psf, 0, None)
    return (psf / psf.sum()).astype(np.float32), fw


def deconvolve(img: np.ndarray, strength: float, sat: float, noise_ref: float | None = None,
               progress=None) -> tuple[np.ndarray, dict]:
    """Luminance Richardson-Lucy with TV regularisation, deringing and SNR masking.

    ``noise_ref`` is the per-pixel noise of the *un-denoised* stack; the SNR
    mask uses it so deconvolution only acts where the original data genuinely
    supports it (stars, bright structure) and never re-amplifies residual noise.
    """
    if strength <= 0:
        return img, {}
    L = luminance(img)
    psf, fw = estimate_psf(L, sat)
    if psf is None:
        return img, {"skipped": "not enough stars for PSF"}
    iters = int(round(5 + 25 * strength))
    lam = 0.004
    eps = 1e-6
    bg = np.median(L)
    sig = mad_sigma(L[::4, ::4] - cv2.GaussianBlur(L, (0, 0), 3)[::4, ::4])
    off = max(0.0, -float(L.min())) + 10 * sig + 1e-4
    obs = L + off
    est = obs.copy()
    psf_flip = psf[::-1, ::-1].copy()
    for i in range(iters):
        conv = cv2.filter2D(est, -1, psf_flip, borderType=cv2.BORDER_REFLECT)
        ratio = obs / np.maximum(conv, eps)
        corr = cv2.filter2D(ratio, -1, psf, borderType=cv2.BORDER_REFLECT)
        # total-variation regularisation (Dey et al. 2006)
        gy, gx = np.gradient(est)
        nrm = np.sqrt(gx * gx + gy * gy) + 1e-3 * (sig + 1e-6)
        div = np.gradient(gx / nrm, axis=1) + np.gradient(gy / nrm, axis=0)
        est = est * corr / np.maximum(1 - lam * div, 0.2)
        if progress:
            progress(i + 1, iters, f"Deconvolution {i + 1}/{iters}")
    est -= off
    # deringing: never go darker than the local minimum of the observation, and never
    # remove more than 20% of any pixel (dark rings around stars are RL's signature artefact)
    lo = minimum_filter(L, size=psf.shape[0])
    est = np.maximum(est, lo - 0.5 * sig)
    pos = np.maximum(L - bg, 0)
    est = np.maximum(est, bg + 0.8 * pos - (L - bg - pos))
    # only sharpen where there is signal (avoid lifting noise in the background)
    ref_sig = max(noise_ref if noise_ref else sig, 1e-9)
    snr = (cv2.GaussianBlur(L, (0, 0), 1.5) - bg) / ref_sig
    m = smoothstep(10, 40, snr)
    # keep saturated star cores untouched; taper strength around very bright stars
    satm = maximum_filter((L > 0.8 * sat).astype(np.float32), size=psf.shape[0] * 2)
    bright = cv2.GaussianBlur(maximum_filter((snr > 400).astype(np.float32), size=psf.shape[0]), (0, 0), psf.shape[0] / 3)
    m = m * (1 - satm) * (1 - 0.6 * np.clip(bright, 0, 1))
    Lnew = L + m * (est - L) * min(1.0, 0.5 + strength)
    ratio = (np.maximum(Lnew, 0) + 1e-6) / (np.maximum(L, 0) + 1e-6)
    ratio = np.clip(ratio, 0.2, 5)
    out = img * ratio[..., None]
    return out.astype(np.float32), {"psf_fwhm": fw, "iterations": iters}


def linear_stage(stack: np.ndarray, coverage: np.ndarray | None, denoised: np.ndarray | None,
                 params: dict, sat: float, progress=None, sharp: np.ndarray | None = None,
                 restored: bool = False, clip_ref: np.ndarray | None = None,
                 ref_stars: dict | None = None) -> tuple[np.ndarray, dict]:
    """``sharp``: output of the self-supervised deconvolution network (denoise.n2n_restore).
    When present, "deconvolution" blends towards it; otherwise Richardson-Lucy is used.

    ``restored``: ``stack`` is already a restoration (ImageMM's latent image): no denoise
    blend and no further deconvolution; ``clip_ref`` is then the original coadd on the same
    grid, which tells where the data were saturated (a restored star core may legitimately
    exceed the sensor's white level), and the output is normalised by the larger of the
    white level and the image maximum so restored cores are not clipped.

    ``ref_stars``: catalogue stars for the spectrophotometric white balance ("auto"; spcc.py),
    positions in ``stack`` pixels, or {"error": why there are none}."""
    p = {**DEFAULTS, **(params or {})}
    info = {}
    img = stack
    # per-pixel noise of the raw stack (for a restoration: of the coadd it was restored from)
    Ls = luminance((clip_ref if (restored and clip_ref is not None) else stack)[::2, ::2])
    noise_ref = mad_sigma(Ls - cv2.GaussianBlur(Ls, (0, 0), 1.5)) * 1.6
    if denoised is not None and p["denoise"] > 0:
        img = stack + float(p["denoise"]) * (denoised - stack)
    ai_deconv = sharp is not None and denoised is not None and p["deconvolution"] > 0
    if ai_deconv:
        img = img + float(p["deconvolution"]) * (sharp - denoised)
    if p["crop"] and coverage is not None:
        y0, y1, x0, x1 = auto_crop_box(coverage, p["crop_threshold"])
        img = img[y0:y1, x0:x1]
        info["crop"] = [int(y0), int(y1), int(x0), int(x1)]
    img = np.ascontiguousarray(img, np.float32)
    if progress:
        progress(1, 4, "Background extraction")
    if restored:
        # the restoration is already background-subtracted (the ImageMM exposures had the
        # reference sky model and their own smooth deviations removed) and its zero point is
        # the sky: no gradient model and no median subtraction (in a nebula-filled field the
        # median is nebula).  The display pedestal is set from the coadd's sky noise, as for
        # a stack after neutralisation.
        info["background"] = {"method": "subtracted before the restoration"}
        ref = clip_ref if clip_ref is not None else img
        if info.get("crop") and clip_ref is not None:
            y0, y1, x0, x1 = info["crop"]
            ref = ref[y0:y1, x0:x1]
        ref = ref[::2, ::2]
        ped = 3 * max(mad_sigma(ref[..., c] - cv2.GaussianBlur(np.ascontiguousarray(ref[..., c]), (0, 0), 1.5)) * 1.6
                      for c in range(3))
    else:
        if p["background"]:
            bg, binfo = background_model(img, p["bg_method"], int(p["bg_degree"]))
            info["background"] = binfo
            img = img - bg
        else:
            img = img - np.array([np.median(img[..., c][::4, ::4]) for c in range(3)], np.float32)
        # background neutralisation: equal, small pedestal in every channel
        s_bg = [mad_sigma(img[..., c][::4, ::4]) for c in range(3)]
        ped = 3 * max(s_bg)
        for c in range(3):
            img[..., c] -= np.median(img[..., c][::4, ::4])
    img += ped
    if progress:
        progress(2, 4, "Colour calibration")
    # clipped highlights carry no colour information (see below); also kept out of the photometry
    clip_src = clip_ref if (restored and clip_ref is not None) else stack
    if info.get("crop"):
        y0, y1, x0, x1 = info["crop"]
        clip_src = clip_src[y0:y1, x0:x1]
    clipped = (clip_src.max(-1) >= 0.80 * sat).astype(np.float32)
    gains = np.ones(3, np.float32)
    wb = p["white_balance"]
    cal = {"method": wb}
    calibrated = False
    if wb in ("auto", "stars"):
        wb_src = img
        if sharp is not None and denoised is not None:
            # measure star colours on the fully deconvolved image: each channel's halo
            # light is back in the star core, so the photometry is complete in every
            # colour and the white balance does not depend on the deconvolution slider
            extra = (1.0 - (float(p["deconvolution"]) if ai_deconv else 0.0)) * (sharp - denoised)
            if info.get("crop"):
                y0, y1, x0, x1 = info["crop"]
                extra = extra[y0:y1, x0:x1]
            wb_src = img + extra
        if wb == "auto":
            # spectrophotometric: catalogue stars' predicted colours in this camera (spcc.py)
            g_ = None
            if ref_stars and ref_stars.get("x") is not None:
                from .spcc import photometric_gains
                y0, _, x0, _ = info.get("crop") or [0, 0, 0, 0]
                st_ = {**ref_stars, "x": np.asarray(ref_stars["x"]) - x0, "y": np.asarray(ref_stars["y"]) - y0}
                g_, cinfo = photometric_gains(wb_src, st_, clipped > 0, noise=noise_ref)
                cal.update(cinfo)
                cal.update({k: v for k, v in (ref_stars.get("meta") or {}).items() if k != "db_offline"})
            else:
                cal["note"] = (ref_stars or {}).get("error") or \
                    "the stack is not plate-solved (Explore tab): no catalogue colours"
            if g_ is not None:
                gains, calibrated = g_, True
                cal["method"] = "spectrophotometric (Gaia DR3)"
            else:
                cal["method"] = "stars (average star = white): photometric calibration unavailable"
        if not calibrated:
            fl = measure_star_colors(wb_src, sat, noise_floor=noise_ref if restored else 0.0)
            if fl is not None and len(fl) >= 10:
                rg = np.median(fl[:, 0] / fl[:, 1])
                bg_ = np.median(fl[:, 2] / fl[:, 1])
                gains = np.array([1 / rg, 1.0, 1 / bg_], np.float32)
                gains = np.clip(gains, 0.25, 4)
        del wb_src
    if (gains != 1).any():
        img = (img - ped) * gains + ped
    info["wb_gains"] = gains.tolist()
    info["color_calibration"] = cal
    # Once white balance scales the channels differently, clipped pixels turn blue/purple/cyan
    # ("coloured blooming"): render every pixel that was clipped in ANY channel neutral (max channel).
    info["clipped_pixels"] = int(clipped.sum())
    if clipped.any():
        soft = cv2.GaussianBlur(cv2.dilate(clipped, np.ones((5, 5), np.uint8)), (0, 0), 2.0)
        soft = np.clip(soft * 1.5, 0, 1)[..., None]
        img = img * (1 - soft) + img.max(-1, keepdims=True) * soft
    if progress:
        progress(3, 4, "Deconvolution")
    if restored:
        dinfo = {"method": "restored (ImageMM)"}
    elif ai_deconv:
        dinfo = {"method": "Noise2Noise deconvolution network", "strength": float(p["deconvolution"])}
    else:
        img, dinfo = deconvolve(img, float(p["deconvolution"]), sat, noise_ref=noise_ref)
    info["deconvolution"] = dinfo
    # normalise to [0, 1] against the sensor's white level (a restoration may exceed it)
    if restored:
        sat = max(sat, float(img.max()))
        info["white_level"] = sat
    img = img / sat
    info["pedestal"] = ped / sat
    info["noise_ref"] = float(noise_ref / sat)  # per-pixel noise of the un-denoised stack
    return img.astype(np.float32), info


# ============================================================ non-linear stage

def detect_stars_for_mask(L: np.ndarray, px_scale: float = 1.0, noise_floor: float = 0.0,
                          detect_floor: float | None = None):
    """``noise_floor``: per-pixel noise of the original data at this scale; detection
    significance is relative to the larger of it and the image's own background rms (after
    a restoration or ML denoising the image's own noise no longer describes the data).
    ``detect_floor``: a different floor for the detection threshold alone (the PSF width and
    the returned rms still use ``noise_floor``).  Returns (objects, rms, fwhm)."""
    D = STAR_DETECT
    Lc = np.ascontiguousarray(L, np.float32)
    # a first coarse pass measures the PSF; the background mesh then scales with it so
    # it follows extended light (galaxy discs, nebula ridges) and stars on top of it
    # separate cleanly instead of merging into one large blob
    b0 = sep.Background(Lc, bw=64, bh=64)
    o0 = _extract(Lc - b0.back(), 10.0, max(b0.globalrms, noise_floor), minarea=5)
    fw0 = 3.0
    if len(o0) > 10:
        f0 = 2 * sep.flux_radius(Lc - b0.back(), o0["x"], o0["y"], 6 * o0["a"], 0.5, subpix=5)[0]
        fw0 = float(np.nanmedian(f0[(o0["flag"] == 0) & (o0["a"] / np.maximum(o0["b"], 1e-3) < 1.5)])) if len(f0) else 3.0
    mesh = int(np.clip(4.5 * fw0, 12, 64))
    bkg = sep.Background(Lc, bw=mesh, bh=mesh, fw=3, fh=3)
    sub = Lc - bkg.back()
    rms = max(bkg.globalrms, noise_floor)
    rms_det = max(bkg.globalrms, noise_floor if detect_floor is None else detect_floor)
    objs = _extract(sub, D["detect_sigma"], rms_det, minarea=3, deblend_cont=D["deblend_cont"])
    if len(objs) == 0:
        return objs, rms, 3.0
    fw = 2 * sep.flux_radius(sub, objs["x"], objs["y"], 6 * objs["a"], 0.5, subpix=5)[0]
    fw_med = float(np.nanmedian(fw[(objs["peak"] > 20 * rms)])) if (objs["peak"] > 20 * rms).any() else float(np.nanmedian(fw))
    elong = objs["a"] / np.maximum(objs["b"], 1e-3)
    # stars: compact & round.  Big objects only count when they are *saturated*
    # (bloated bright stars); extended bright objects – galaxy cores, compact
    # companions (M32, M110), planetary nebulae – are left in the starless layer.
    back_med = float(np.median(bkg.back()))
    # data are normalised to the sensor's full well, so "near saturation" is absolute
    sat_peak = objs["peak"] + back_med > min(0.85 * float(np.max(Lc)), 0.5)
    # the half-light size is only measured well on bright objects: for a faint one in a crowded field
    # the aperture is ruled by its neighbours and the background residual (C 33: compact, round faint
    # stars measured 3.4-8x the stars' FWHM and were rejected as extended - 2700 of them stayed in the
    # starless image).  Faint objects are judged by shape and concentration (below) alone.
    compact = ((fw < D["compact_fwhm"] * fw_med) | (objs["peak"] < 20 * rms)) & (elong < D["max_elongation"])
    star = compact | (sat_peak & (elong < 1.5))
    # concentration index: for a point source total flux ~ 2*pi*sigma^2*peak with the
    # image PSF; galaxy nuclei / compact galaxies / knots carry far more flux than
    # their peak implies.  (Saturated stars are exempt – their peak is clipped.)
    sig_psf = max(fw_med, 1.0) / 2.3548
    conc = objs["flux"] / np.maximum(2 * np.pi * sig_psf ** 2 * objs["peak"], 1e-12)
    # "saturated" objects must also *look* like saturated stars: a small flat top with
    # a steep fall-off.  Bright extended objects (planetary nebulae, galaxy cores)
    # stay bright far beyond their brightest region and are rejected here.
    h, w = Lc.shape
    sat_star = np.zeros(len(objs), bool)
    for i in np.nonzero(sat_peak)[0]:
        x, y = float(objs["x"][i]), float(objs["y"][i])
        R = int(min(0.05 * max(h, w), max(20, 8 * np.sqrt(objs["npix"][i] / np.pi))))
        y0, y1, x0, x1 = max(0, int(y) - R), min(h, int(y) + R + 1), max(0, int(x) - R), min(w, int(x) + R + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rad = np.hypot(xx - x, yy - y).astype(np.int32)
        cut = sub[y0:y1, x0:x1]
        prof = np.bincount(rad.ravel(), cut.ravel()) / np.maximum(np.bincount(rad.ravel()), 1)
        pk = prof[:3].max()
        if pk <= 0:
            continue
        flat = np.nonzero(prof < 0.8 * pk)[0]
        r_flat = int(flat[0]) if len(flat) else len(prof)
        probe = int(r_flat + 3 * fw_med)
        sat_star[i] = probe < len(prof) and prof[probe] < 0.12 * pk
    star &= (conc < D["concentration"]) | sat_star
    star &= ~(sat_peak & ~sat_star & ~((fw < D["compact_fwhm"] * fw_med) & (conc < D["concentration"])))
    found = objs[star]
    # stars on bright extended light (a planetary nebula's lobes, a galaxy disc): the mesh background
    # cannot follow structure of a few tens of pixels, so such a star merges with the object into one
    # extended source and is rejected - it then stays in the starless layer, where the chroma noise
    # reduction spreads its neutral colour into a grey disc over the nebula (M 76).  A white top-hat
    # (the image minus its opening by a disc a few FWHM across) keeps only structure smaller than a
    # star: compact, round, significant peaks there that are not already stars are added.
    k = max(5, int(round(3 * fw_med)) | 1)
    th = Lc - cv2.morphologyEx(Lc, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    try:
        extra = _extract(np.ascontiguousarray(th), D["tophat_sigma"], rms_det, minarea=3,
                         deblend_cont=D["deblend_cont"])
    except Exception:
        extra = []
    if len(extra):
        fe = 2 * sep.flux_radius(th, extra["x"], extra["y"], 6 * extra["a"], 0.5, subpix=5)[0]
        # (as above: the size is only trusted on bright objects; and blended / merged detections are
        # kept - on a star-dense nebula nearly every star touches a neighbour, and requiring flag 0
        # left all of them on the Veil's filaments)
        ok = (((fe < 1.6 * fw_med) | (extra["peak"] < 20 * rms)) & (extra["a"] / np.maximum(extra["b"], 1e-3) < 1.6)
              & ((extra["flag"] & ~0x03) == 0))
        if len(found) and ok.any():
            from scipy.spatial import cKDTree
            d, _ = cKDTree(np.stack([found["x"], found["y"]], 1)).query(np.stack([extra["x"], extra["y"]], 1))
            ok &= d > 1.5 * fw_med
        if ok.any():
            found = np.concatenate([found, extra[ok]])
    return found, rms, fw_med


def star_mask(L: np.ndarray, px_scale: float = 1.0, grow: float = 1.0, rgb: np.ndarray | None = None,
              noise_ref: float | None = None, detect_L: np.ndarray | None = None,
              restored_sigma: float | None = None) -> np.ndarray:
    """Soft star mask built from photometric star profiles (Gaussian + wings).

    A restoration (ImageMM) is different: its stars are points seen through the display
    Gaussian g_sigma (``restored_sigma``, in pixels of ``L``), with no wings or halos left
    (they were deconvolved into the cores), and its sky noise is sparse positive speckle that
    looks like faint stars.  Stars are then found on ``detect_L``, the coadd the restoration
    was made from (same grid and scale; honest Gaussian noise, so speckle is never taken for
    a star), and each mask extends to where the star's Gaussian falls to the noise floor."""
    h, w = L.shape
    mask = np.zeros((h, w), np.float32)
    if restored_sigma is not None and detect_L is not None:
        objs, rms, _ = detect_stars_for_mask(detect_L, px_scale, noise_ref or 0.0)
        if len(objs) == 0:
            return mask
        sig = max(float(restored_sigma), 0.5)
        peak = np.maximum(objs["flux"], 0) / (2 * np.pi * sig ** 2)       # the restoration conserves flux
        ratio = np.maximum(peak / max(rms, noise_ref or 0.0, 1e-12), 1.01)
        r = sig * np.sqrt(2 * np.log(ratio)) * STAR_MASK["radius_scale"] * grow + 1.0
        r = np.clip(r, 1.5, 0.02 * max(h, w))
        for x, y, rr in zip(objs["x"], objs["y"], r):
            cv2.circle(mask, (int(round(x)), int(round(y))), int(np.ceil(rr)), 1.0, -1, lineType=cv2.LINE_AA)
        star_mask.last_expanded = 0
        return np.clip(mask, 0, 1)
    # detection at the image's own noise: on denoised data a faint star is obvious at a peak of 1-4x
    # the raw stack's per-pixel noise, and with that noise as the detection floor half the stars of
    # a Milky Way field (C 33: 6965 of ~14000) were never masked - they stayed in the starless
    # image, and in the AI star remover's training targets, so it learnt to keep them as well.
    # Their sizes are still measured against the raw stack's noise (below).
    objs, rms, fw = detect_stars_for_mask(L, px_scale, noise_ref or 0.0, detect_floor=0.0)
    if len(objs) == 0:
        return mask
    sigma = max(fw, 1.0) / 2.3548
    peak = np.maximum(objs["peak"], objs["flux"] / (2 * np.pi * sigma ** 2))
    # size stars against the noise of the *original* stack when known: after ML
    # denoising the residual noise is tiny, so every faint star would look high-SNR
    nfloor = max(rms, noise_ref or 0.0)
    ratio = np.maximum(peak / nfloor, 1.01)
    # radius where the (Gaussian-core) profile falls to the noise, but never below
    # 0.5% of the star's own peak: on denoised data the noise floor is tiny and a
    # pure noise criterion balloons the mask (dense Milky Way fields -> 50% masked).
    # Bright stars' real halos are measured from their profiles below.
    r = sigma * np.sqrt(2 * np.log(np.minimum(ratio, 200.0)))
    r = r * STAR_MASK["radius_scale"] * grow + 1.0
    # never below one FWHM: a faint star's wings hold half its light, and a core-only disc left
    # them in the starless image as a ring of dots
    r = np.clip(r, max(1.5, STAR_MASK["min_radius_fwhm"] * fw * grow), 0.08 * max(h, w))
    # and never inside the star's own measured extent (3 x its second-moment size, i.e. ~3 sigma of
    # its profile): faint stars on denoised data are wider than the bright stars' FWHM, so a disc
    # sized from that FWHM left their wings outside, and the inpainting filled the disc back up
    # from them (a dot in every hole)
    r = np.maximum(r, np.minimum(3.0 * objs["a"] * grow + 1.0, 4.0 * fw * grow + 1.0))
    # bright stars: measure the real extent of the halo from the radial profile
    # (chromatic/scattering halos of small refractors are far wider than a Gaussian)
    # Only the brightest stars carry significant halos: top 2% by peak, or saturated.
    # Halos of small refractors are often coloured (blue/violet rings) and nearly
    # invisible in luminance, so the profile is measured per colour channel and
    # normalised by the *local* texture noise of the surrounding annulus.
    chans = rgb if rgb is not None else L[..., None]
    chans = cv2.GaussianBlur(np.ascontiguousarray(chans, np.float32), (0, 0), 1.0)
    if chans.ndim == 2:
        chans = chans[..., None]
    lmax = float(L.max())
    # candidates: stars whose peak reaches a sizeable fraction of full well (true halo
    # producers) – a brightness *rank* would pick hundreds of ordinary stars in a
    # dense Milky Way field
    peak_abs = objs["peak"] + float(np.median(L[::4, ::4]))
    peak_thr = max(np.percentile(ratio, 98), 150)
    cand = (ratio >= peak_thr) & (peak_abs >= 0.15 * lmax)
    rmax_all = int(0.04 * max(h, w))
    expanded = 0
    for i in np.nonzero(cand)[0][np.argsort(-objs["peak"][cand])][:200]:
        x, y = objs["x"][i], objs["y"][i]
        xi, yi = int(round(x)), int(round(y))
        R = int(min(rmax_all, max(40, 3 * r[i])))
        y0, y1, x0, x1 = max(0, yi - R), min(h, yi + R + 1), max(0, xi - R), min(w, xi + R + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rad = np.hypot(xx - x, yy - y).astype(np.int32)
        ok = rad < R
        ring_out = ok & (rad >= int(0.8 * R))
        if ring_out.sum() < 30:
            continue
        cnt = np.maximum(np.bincount(rad[ok], minlength=R), 1)
        lim = int(0.8 * R)
        below_all = np.ones(R, bool)
        core_frac = 0.0
        for c in range(chans.shape[2]):
            cut = chans[y0:y1, x0:x1, c]
            prof = np.bincount(rad[ok], weights=cut[ok], minlength=R) / cnt
            outer = cut[ring_out]
            bgc = np.median(outer)
            pix_noise = 1.4826 * np.median(np.abs(outer - bgc)) + 1e-9
            amp = max(prof[0] - bgc, 1e-9)
            # the halo ends where the channel falls below 0.2% of the star's own peak
            # (or the local pixel noise) – relative, so it works on any background
            thr = bgc + max(STAR_MASK["halo_floor"] * amp, 1.0 * pix_noise)
            below_all &= prof <= thr
            k2 = min(R - 1, int(round(2 * fw)) + 1)
            core_frac = max(core_frac, (prof[k2] - bgc) / amp)
        below = np.nonzero(below_all[:lim])[0]
        if not len(below):
            continue  # never falls to the local sky -> extended object
        edge = int(below[0])
        saturated = L[yi, xi] > 0.85 * lmax if 0 <= yi < h and 0 <= xi < w else False
        if not saturated and core_frac > 0.08:
            continue  # not point-like (galaxy nucleus, nebula knot)
        new_r = edge * STAR_MASK["halo_margin"] * grow + 2
        if new_r > r[i]:
            r[i] = new_r
            expanded += 1
    star_mask.last_expanded = expanded
    for x, y, rr in zip(objs["x"], objs["y"], r):
        cv2.circle(mask, (int(round(x)), int(round(y))), int(np.ceil(rr)), 1.0, -1, lineType=cv2.LINE_AA)
    return np.clip(mask, 0, 1)


def _halo_radius(L: np.ndarray, x: float, y: float, r_min: float, r_max: int, nsig: float = 3.0,
                 floor: float = 0.0) -> float:
    """Radius where a star's halo meets the local background: the ring *medians* of ``L``
    (robust to the other stars in a ring) minus the median of the outer annulus [0.8, 1] r_max,
    first below the larger of ``nsig`` x the standard error of a ring median and ``floor`` (light
    below it is invisible: on denoised data the ring medians are so precise that a halo falling
    as r^-3 stays "significant" out to any radius), beyond ``r_min``.  Returns r_max when the
    halo never meets the background (an extended object)."""
    from scipy import ndimage
    h, w = L.shape
    xi, yi = int(round(x)), int(round(y))
    y0, y1, x0, x1 = max(0, yi - r_max), min(h, yi + r_max + 1), max(0, xi - r_max), min(w, xi + r_max + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rad = np.hypot(xx - x, yy - y).astype(np.int32)
    cut = L[y0:y1, x0:x1]
    ok = rad < r_max
    outer = cut[ok & (rad >= int(0.8 * r_max))]
    if outer.size < 50:
        return float(r_max)
    bg = float(np.median(outer))
    sig = 1.4826 * float(np.median(np.abs(outer - bg))) + 1e-12
    idx = np.arange(int(np.ceil(r_min)), int(0.8 * r_max))
    if not len(idx):
        return float(r_max)
    med = ndimage.median(cut, labels=np.where(ok, rad, -1), index=idx)
    cnt = np.bincount(rad[ok], minlength=r_max)[idx]
    se = 1.2533 * sig / np.sqrt(np.maximum(cnt, 1))        # standard error of a median
    below = np.nonzero(np.asarray(med) - bg < np.maximum(nsig * se, floor))[0]
    return float(idx[below[0]]) if len(below) else float(r_max)


def neutralize_star_halos(lin: np.ndarray, strength: float, px_scale: float = 1.0,
                          noise_ref: float | None = None, detect_L: np.ndarray | None = None,
                          core_sigma: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Remove coloured (blue/violet/cyan/green) halos around bright stars, in linear data.

    Small refractors focus blue/violet (and the OIII band) slightly differently, and dual-band
    filters scatter a wide halo mostly into the OIII (green and blue) pixels, so bright stars
    get coloured rings/halos.  Around each bright star, the halo light *in excess of the local
    background* is desaturated towards neutral, ramping in outside the star's core so the
    core keeps its true colour.  Works regardless of which layer the halo later ends up in.

    The halo's reach is measured on the star's own radial profile (``_halo_radius``).  It used
    to be a formula of the star's FWHM, which after a deconvolution or restoration is a
    fraction of the optical halo's size: on M 76 (Seestar, LP filter) the brightest field star
    measured FWHM 2.0 px, reach 20 px, while its teal halo (G and B 20 % above R) extends to
    ~80 px.  ``detect_L``: the image to find stars on (for a restoration, the coadd it was made
    from - its speckle would pass for stars); ``core_sigma``: Gaussian sigma of a restoration's
    star cores (the core radius is then set from it).
    Returns (corrected image, halo weight map in [0, 1]: how far each pixel was neutralised -
    the palette uses it to keep the halo neutral, see ``extract_ha_oiii``)."""
    W = np.zeros(lin.shape[:2], np.float32)
    if strength <= 0:
        return lin, W
    L = luminance(lin)
    objs, rms, fw = detect_stars_for_mask(detect_L if detect_L is not None else L, px_scale, noise_ref or 0.0)
    if len(objs) == 0:
        return lin, W
    h, w = L.shape
    ratio = objs["peak"] / max(rms, 1e-12)
    sel = np.nonzero(ratio >= max(np.percentile(ratio, 97), 100))[0]
    sel = sel[np.argsort(-objs["flux"][sel])][:300]
    out = lin.copy()
    r_core = max(1.5, 1.2 * fw) if core_sigma is None else max(1.5, 2.5 * float(core_sigma))
    r_cap = int(np.clip(0.06 * max(h, w), 40, 400))
    for i in sel:
        x, y = float(objs["x"][i]), float(objs["y"][i])
        # measured reach, never below the former brightness-based estimate
        Rf = float(np.clip(fw * (2.5 + 2.2 * np.log10(ratio[i])), 12, 0.035 * max(h, w)))
        hr = _halo_radius(L, x, y, 2 * r_core, r_cap, floor=0.5 * (noise_ref or 0.0))
        if hr >= int(0.8 * r_cap) - 1:
            # the profile never meets the background: the star sits on an extended object (a nebula,
            # a galaxy, a planetary nebula), whose own light the ring medians measure - neutralising
            # it greyed a disc of the object around every star on it (M 76's lobes)
            continue
        Rh = max(Rf, 1.1 * hr)
        R = int(np.ceil(Rh * 1.35))
        xi, yi = int(round(x)), int(round(y))
        y0, y1, x0, x1 = max(0, yi - R), min(h, yi + R + 1), max(0, xi - R), min(w, xi + R + 1)
        if y1 - y0 < 8 or x1 - x0 < 8:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rad = np.hypot(xx - x, yy - y)
        patch = out[y0:y1, x0:x1]
        ring_out = (rad >= Rh * 1.15) & (rad < R)
        if ring_out.sum() < 20:
            continue
        bg = np.median(patch[ring_out], axis=0)
        wgt = np.clip((rad - r_core) / r_core, 0, 1) * np.clip((R - rad) / max(R - Rh, 1e-6), 0, 1)
        wgt = strength * wgt
        W[y0:y1, x0:x1] = np.maximum(W[y0:y1, x0:x1], wgt)
        # the halo is the star's azimuthally symmetric light: per channel, the ring medians minus the
        # local background (robust to the stars, knots and nebula texture inside the rings).  Only
        # its colour - each channel's departure from the halo's luminance - is removed, so everything
        # else in the rings (a red Ha knot, a neighbouring star) keeps its own colour.
        ri = rad.astype(np.int32)
        lab_ = np.where(ri < R, ri, -1)
        from scipy import ndimage
        prof = np.stack([np.asarray(ndimage.median(patch[..., c], labels=lab_, index=np.arange(R))) - bg[c]
                         for c in range(patch.shape[-1])], -1)                      # (R, C)
        prof = np.maximum(prof, 0)
        # the halos to fix carry *extra* green / blue light (a dual-band filter's OIII scatter, a
        # refractor's blue focus): remove the G and B excess over R, nothing else.  Pulling every
        # channel to the halo's luminance also "neutralised" red light - a red nebula or the reddish
        # sky under a wide profile - by adding G and B: a grey disc ~140 px across round NGC 6960's
        # brightest star at full resolution, where the profile reaches 6 % of the frame
        ex = np.zeros_like(prof)
        ex[:, 1:] = np.maximum(prof[:, 1:] - prof[:, :1], 0)
        excess = ex[np.minimum(ri, R - 1)]                                           # colour of the halo
        out[y0:y1, x0:x1] = patch - wgt[..., None] * excess
    return out.astype(np.float32), W


def pushpull_fill(img: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Fill low-weight regions by pyramid push-pull (normalised convolution).

    Smooth, artefact-free interpolation from the hole boundary inward; works
    for arbitrarily large holes (big star halos) and any value range.
    """
    levels = []
    cur_v = img * weight[..., None]
    cur_w = weight.astype(np.float32)
    while min(cur_w.shape) > 8:
        levels.append((cur_v, cur_w))
        h, w = cur_w.shape
        cur_v = cv2.resize(cur_v, ((w + 1) // 2, (h + 1) // 2), interpolation=cv2.INTER_AREA)
        cur_w = cv2.resize(cur_w, ((w + 1) // 2, (h + 1) // 2), interpolation=cv2.INTER_AREA)
        if cur_v.ndim == 2:
            cur_v = cur_v[..., None]
    filled = cur_v / np.maximum(cur_w, 1e-6)[..., None]
    for v, w in reversed(levels):
        up = cv2.resize(filled, (w.shape[1], w.shape[0]), interpolation=cv2.INTER_LINEAR)
        if up.ndim == 2:
            up = up[..., None]
        wn = np.clip(w * 4, 0, 1)[..., None]  # a partially-covered cell trusts its own data
        own = v / np.maximum(w, 1e-6)[..., None]
        filled = wn * own + (1 - wn) * up
    return filled.astype(np.float32)


def inpaint_stars(img: np.ndarray, mask: np.ndarray, seed: int = 0) -> np.ndarray:
    """Remove stars: push-pull fill of the masked profiles plus matched synthetic grain."""
    hard = (mask > 0.3).astype(np.float32)
    fill = pushpull_fill(img, 1 - hard)
    # re-introduce noise of the local texture so filled discs don't look "plastic"
    resid = img - cv2.GaussianBlur(img, (0, 0), 1.5)
    sig = np.array([mad_sigma(resid[..., c][::3, ::3][hard[::3, ::3] < 0.5]) for c in range(img.shape[2])], np.float32)
    rng = np.random.default_rng(seed)
    noise = cv2.GaussianBlur(rng.standard_normal(img.shape).astype(np.float32), (0, 0), 0.7) * STAR_MASK["grain"] * sig
    fill = fill + noise
    soft = cv2.GaussianBlur(hard, (0, 0), 1.2)[..., None]
    return (img * (1 - soft) + fill * soft).astype(np.float32)


def classic_star_separation(lin: np.ndarray, px_scale: float = 1.0, noise_ref: float | None = None,
                            detect_L: np.ndarray | None = None, restored_sigma: float | None = None):
    """Star removal without the network: photometric star mask (``star_mask``) and push-pull
    inpainting.  Arguments at the scale of ``lin``.  Returns (mask, starless)."""
    smask = star_mask(luminance(lin), px_scale, grow=STAR_MASK["grow"], rgb=lin, noise_ref=noise_ref,
                      detect_L=detect_L, restored_sigma=restored_sigma)
    return smask, inpaint_stars(lin, smask)


def ghs(x: np.ndarray, D: float, b: float, SP: float) -> np.ndarray:
    """Generalized Hyperbolic Stretch (Payne & Cranfield 2021) with symmetry point SP."""
    def T0(u):
        u = np.maximum(u, 0)
        if abs(b) < 1e-6:
            return 1 - np.exp(-D * u)
        if b > 0:
            return 1 - np.power(1 + D * b * u, -1 / b)
        if abs(b + 1) < 1e-6:
            return np.log1p(D * u)
        return (1 - np.power(1 - D * b * u, (b + 1) / b)) / (D * (b + 1))
    x = np.asarray(x, np.float32)
    t = np.where(x >= SP, T0(x - SP), -T0(SP - x))
    t0 = -T0(np.float32(SP))
    t1 = T0(np.float32(1 - SP))
    return ((t - t0) / (t1 - t0)).astype(np.float32)


_U = np.linspace(0, 1, 8193, dtype=np.float64)


def lut_sqrt(x: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Evaluate a 1-D curve tabulated on the sqrt grid ``_U`` (linear interpolation)."""
    n = len(lut) - 1
    u = np.sqrt(np.clip(x, 0, 1), dtype=np.float32) * np.float32(n)
    i = np.minimum(u.astype(np.int32), n - 1)
    f = u - i
    lo = lut[i]
    return lo + (lut[i + 1] - lo) * f


def ghs_fast(x: np.ndarray, D: float, b: float, SP: float) -> np.ndarray:
    """GHS via a lookup table sampled on a sqrt(x) grid (dense near black where the
    curve is steepest). Max error ~1e-5, ~10x faster than direct evaluation."""
    lut = ghs(_U ** 2, D, b, SP).astype(np.float32)
    return lut_sqrt(x, lut).astype(np.float32)


# Stretch algorithms (as in Siril, https://siril.readthedocs.io/en/stable/processing/stretching.html).
# Each is a one-parameter family in its strength D >= 0 (D = 0: identity), so one solver finds the
# D that puts the background on the target level whatever the curve:
#   ghs    Generalized Hyperbolic Stretch (Payne & Cranfield 2021): b ("contrast") sets where the
#          contrast is focused, the symmetry point SP sits on the background
#   asinh  arcsinh(D x) / arcsinh(D): logarithmic in the highlights, linear in the shadows -
#          keeps star colour and bright cores well
#   mtf    histogram transformation / autostretch: the midtones transfer function with midtone
#          balance m = 0.5 / (1 + D) (PixInsight's STF, Siril's histogram tool)
#   log    log(1 + D x) / log(1 + D): the classic logarithmic stretch
STRETCH_METHODS = ("ghs", "asinh", "mtf", "log")


def tone(x: np.ndarray, D: float, b: float, SP: float, method: str = "ghs") -> np.ndarray:
    """The stretch curve of ``method`` with strength ``D`` (x, output in 0..1)."""
    if method == "ghs":
        return ghs(x, D, b, SP)
    x = np.clip(np.asarray(x, np.float64), 0, 1)
    if D < 1e-9:
        return x.astype(np.float32)
    if method == "asinh":
        return (np.arcsinh(D * x) / np.arcsinh(D)).astype(np.float32)
    if method == "log":
        return (np.log1p(D * x) / np.log1p(D)).astype(np.float32)
    if method == "mtf":
        m = 0.5 / (1.0 + D)
        return (((m - 1) * x) / ((2 * m - 1) * x - m)).astype(np.float32)
    raise ValueError(f"unknown stretch method {method!r}")


def tone_fast(x: np.ndarray, D: float, b: float, SP: float, method: str = "ghs") -> np.ndarray:
    """``tone`` via a lookup table on the sqrt grid (see ghs_fast)."""
    lut = tone(_U ** 2, D, b, SP, method).astype(np.float32)
    return lut_sqrt(x, lut).astype(np.float32)


_PROXY_SIZE = 1400


def _stretch_proxy(L: np.ndarray, size: int | None = None) -> np.ndarray:
    """The image the stretch is solved on: area-averaged to at most ``size`` px (the fast
    preview's scale) whatever the render resolution.  The black point and strength come from
    the median and MAD of the pixels, and per-pixel noise depends on the pixel scale - on a
    restoration, whose sky is pixel-scale speckle, the full-resolution render got a lower black
    point and a lifted, grainy, washed-out background compared with the "Detailed" preview
    (NGC 6960).  Solved at one scale, every render size gets the same curve."""
    h, w = L.shape[:2]
    f = (size or _PROXY_SIZE) / max(h, w)
    if f >= 1:
        return L
    return cv2.resize(np.ascontiguousarray(L, np.float32), (max(1, int(round(w * f))), max(1, int(round(h * f)))),
                      interpolation=cv2.INTER_AREA)


def solve_stretch(L: np.ndarray, target: float, b: float, noise_floor: float = 0.0,
                  method: str = "ghs") -> tuple[float, float, float]:
    """Find black point and GHS strength D so the background median lands on ``target``.
    ``noise_floor``: the data's real per-pixel noise at this scale; the background's own MAD is
    never taken below it.  A restoration's sky is exactly 0 over much of the field (ImageMM drives
    it there, and at the paper's g_sigma the display adds no blur), so its MAD can be 0: the black
    point then sat on the sky itself and the strength ran away (NGC 6960: D = 6.7e5 against 300,
    a grey, speckled, washed-out full-resolution render)."""
    sample = L[::4, ::4].ravel()
    med = float(np.median(sample))
    sig = max(mad_sigma(sample), float(noise_floor))
    bp = max(0.0, med - 2.8 * sig)
    x = np.clip((sample - bp) / (1 - bp), 0, 1)
    sp = float(np.median(x))
    lo, hi = 0.0, 18.0  # log(D) search
    for _ in range(40):
        mid = (lo + hi) / 2
        v = float(np.median(tone(x, np.exp(mid) - 1, b, sp, method)))
        if v < target:
            lo = mid
        else:
            hi = mid
    return bp, float(np.exp((lo + hi) / 2) - 1), sp


def apply_stretch(img: np.ndarray, bp, D, b, sp, color_preservation: float, method: str = "ghs") -> np.ndarray:
    x = np.clip((img - bp) / (1 - bp), 0, 1)
    per = tone_fast(x, D, b, sp, method)
    if color_preservation <= 0:
        return per
    L = luminance(x)
    Ls = tone_fast(L, D, b, sp, method)
    ratio = Ls / np.maximum(L, 1e-6)
    cp = x * ratio[..., None]
    # luminance-preserving gamut mapping: when a channel exceeds 1, desaturate towards
    # white just enough to fit (dividing by the max would darken saturated highlights
    # and create dark rings around coloured star halos)
    mx = cp.max(-1, keepdims=True)
    Lsn = Ls[..., None]
    k = np.clip((1 - Lsn) / np.maximum(mx - Lsn, 1e-6), 0, 1)
    cp = np.where(mx > 1, Lsn + (cp - Lsn) * k, cp)
    return (color_preservation * cp + (1 - color_preservation) * per).astype(np.float32)


def stretch_mono(ch: np.ndarray, target: float, b: float, method: str = "ghs") -> np.ndarray:
    bp, D, sp = solve_stretch(ch, target, b, method=method)
    return tone(np.clip((ch - bp) / (1 - bp), 0, 1), D, b, sp, method)


def palette_compose(ha: np.ndarray, oiii: np.ndarray, palette: str) -> np.ndarray:
    """Build a colour image from stretched Ha and OIII (0..1) using a narrowband palette."""
    if palette == "foraxx":
        ho = np.power(np.clip(ha * oiii, 1e-6, 1), 1 - ha * oiii)
        g = ho * ha + (1 - ho) * oiii
        return np.stack([ha, g, oiii], -1)
    if palette == "hoo_warm":
        # Ha -> red/orange, OIII -> teal; a popular "natural-ish" bicolour
        r = ha
        g = 0.35 * ha + 0.65 * oiii
        b = oiii * 0.9 + 0.1 * ha
        return np.stack([r, g, b], -1)
    return np.stack([ha, oiii, oiii], -1)  # HOO


def ha_oiii_params(lin: np.ndarray, unmix: bool = True) -> tuple[float, float]:
    """The two data-derived constants of ``extract_ha_oiii``: the Ha leakage into the OIII
    pixels, k (0 without ``unmix``), and the gain that fits OIII's signal scale to Ha's."""
    ha = lin[..., 0].astype(np.float32)
    oiii = (0.5 * (lin[..., 1] + lin[..., 2])).astype(np.float32)
    hb, ob = np.median(ha[::4, ::4]), np.median(oiii[::4, ::4])
    hs = mad_sigma(ha[::4, ::4] - cv2.GaussianBlur(ha, (0, 0), 2)[::4, ::4])
    k = 0.0
    if unmix:
        hsm = cv2.GaussianBlur(ha, (0, 0), 3) - hb
        osm = cv2.GaussianBlur(oiii, (0, 0), 3) - ob
        sel = hsm[::2, ::2] > max(8 * hs, np.percentile(hsm[::2, ::2], 80))
        if sel.sum() > 500:
            ratio = osm[::2, ::2][sel] / hsm[::2, ::2][sel]
            # sensor leakage is a small physical constant: the lower envelope overestimates
            # it where genuine Ha-only emission is bright (e.g. M27's "ears"), so cap it
            k = float(np.clip(np.percentile(ratio, 5), 0, 0.3))
            unmixed = oiii - k * (ha - hb)
            oiii = np.maximum(unmixed, ob + 0.3 * (oiii - ob))
            ob = np.median(oiii[::4, ::4])
    # linear fit of OIII signal scale to Ha (robust upper-range statistics)
    hr = np.percentile(ha[::4, ::4], 99.0) - hb
    orr = np.percentile(oiii[::4, ::4], 99.0) - ob
    return k, float(np.clip(hr / max(orr, 1e-6), 1.0, 2.0))


def extract_ha_oiii(lin: np.ndarray, unmix: bool = True, boost: float = 1.0, neutral: np.ndarray | None = None,
                    params: tuple[float, float] | None = None, continuum: float | None = None):
    """Split dual-band OSC data into Ha (red pixels) and OIII (green+blue pixels).

    Colour filters on a Bayer sensor are not perfectly selective: green/blue
    pixels also record some Ha (and broadband starlight/continuum).  With
    ``unmix`` the leakage coefficient k is estimated from the data as the
    *lower envelope* of OIII/Ha over high-SNR Ha pixels (regions that are
    pure Ha constrain k), and k*Ha is subtracted (never more than 70 % of the OIII
    signal).  OIII is then linearly fitted to Ha's signal scale so both lines use one
    stretch.

    ``params``: (k, gain) measured elsewhere (``ha_oiii_params``).  A restoration must use the
    coadd's: its OIII is sparse non-negative speckle, often exactly 0, so the 5th percentile of
    OIII / Ha - the leakage - comes out 0, the Ha leakage stays in the OIII pixels and the Ha
    nebula renders grey (IC 405: output R/G 1.32 against the coadd's 1.68).

    ``neutral`` (H, W) in [0, 1]: light that must stay neutral (star halos, made neutral by
    ``neutralize_star_halos``) is exempt from the unmixing and the OIII gain in proportion:
    either would turn it red or teal again - the OIII gain (up to 2x) is what painted the
    halos of bright stars teal / green in the HOO and Foraxx palettes.
    """
    k, gain = params if params is not None else ha_oiii_params(lin, unmix)
    if not unmix:
        k = 0.0
    ha = lin[..., 0].astype(np.float32)
    oiii = (0.5 * (lin[..., 1] + lin[..., 2])).astype(np.float32)
    hb, ob = np.median(ha[::4, ::4]), np.median(oiii[::4, ::4])
    if k > 0:
        unmixed = oiii - k * (ha - hb) * (1 if neutral is None else (1 - neutral))
        oiii = np.maximum(unmixed, ob + 0.3 * (oiii - ob))
        ob = np.median(oiii[::4, ::4])
    gain = gain * boost
    g = gain if neutral is None else 1 + (gain - 1) * (1 - neutral)
    if continuum is not None and continuum > 0:
        # ``continuum``: OIII/Ha of continuum light (the stars', unmixed, before the gain).  Each
        # pixel's light above the sky is split into continuum (as much as both lines allow), an Ha
        # line and an OIII line; the continuum goes into both lines equally - white - and only the
        # OIII *line* gets the gain.  Applied to everything, the gain turned every star, every star
        # the removal left and the Milky Way's unresolved star clouds teal.
        h_, o_ = ha - hb, oiii - ob
        C = np.clip(np.minimum(h_, o_ / continuum), 0, None)
        oiii = hb + C + (o_ - continuum * C) * g
    else:
        oiii = hb + (oiii - ob) * g
    return ha, oiii.astype(np.float32)


def _local_sky(x: np.ndarray, px_scale: float, window: float = 96.0) -> np.ndarray:
    """Robust local sky level of a line map: a median over ~``window`` px (scaled), so stars and
    filaments do not lift it but the large-scale background does.  96 px: the sky's residual cloud
    glow comes in patches 50-200 px across, and against a wider window every one of them counted as
    line emission and was painted red or teal; the Veil's filaments are far narrower."""
    from scipy.ndimage import median_filter, percentile_filter
    f = 8
    h, w = x.shape
    small = cv2.resize(x, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    k = max(5, int(round(window * px_scale / f)) | 1)
    # the window's median where it is sky, its lower envelope where a nebula fills part of it: there
    # the median is nebula, and its diffuse light measured as "sky" was rendered grey between red
    # clumps.  (The envelope everywhere coloured the plain sky again: sky sits above its own 20th
    # percentile.)  A window is "nebula" when median - envelope exceeds what plain sky gives.
    med = median_filter(small, size=k, mode="reflect")
    env = percentile_filter(small, 20, size=k, mode="reflect")
    d0 = 1.5 * float(np.median(med - env))
    sky = np.minimum(med, env + d0)
    sky = cv2.GaussianBlur(sky, (0, 0), k / 4)
    return cv2.resize(sky, (w, h), interpolation=cv2.INTER_LINEAR)


def _line_significance(x: np.ndarray, sigma: float, sky: np.ndarray) -> np.ndarray:
    """Signal of a linear line map above its local sky, in units of its own noise after a
    Gaussian blur of ``sigma`` px (noise from the fine-scale residual of the faint pixels).
    Measured against the *local* sky: against one global level, every large-scale patch of
    background (a residual of cloud glow a few ADU strong) counted as line emission, and the
    palette painted the sky in red and teal blotches."""
    xb = cv2.GaussianBlur(x, (0, 0), sigma)
    d = xb - sky
    faint = d[::4, ::4] <= 0
    resid = (xb - cv2.GaussianBlur(xb, (0, 0), 2 * sigma))[::4, ::4][faint]
    return d / max(mad_sigma(resid), 1e-12)


def _palette_chroma(ha: np.ndarray, oiii: np.ndarray, palette: str, line_s, px_scale: float) -> np.ndarray:
    """OKLab a/b of the narrowband palette from noise-suppressed line maps (see nonlinear_stage)."""
    s_fine, s_broad = max(1.0, 1.5 * px_scale), max(3.0, 8.0 * px_scale)
    sky_h, sky_o = _local_sky(ha, px_scale), _local_sky(oiii, px_scale)

    def chroma(sig):
        hb, ob = cv2.GaussianBlur(ha, (0, 0), sig), cv2.GaussianBlur(oiii, (0, 0), sig)
        lab = rgb_to_oklab(palette_compose(line_s(hb), line_s(ob), palette))
        z = np.maximum(_line_significance(ha, sig, sky_h), _line_significance(oiii, sig, sky_o))
        return lab[..., 1:], z

    c_fine, z_fine = chroma(s_fine)
    c_broad, z_broad = chroma(s_broad)
    # (a few sigma: at 1.5 the sky's own residual texture passed for emission, a red/teal speckle)
    w_fine = smoothstep(4.0, 7.0, z_fine)[..., None]
    w_any = np.maximum(smoothstep(2.5, 5.0, z_broad), w_fine[..., 0])[..., None]
    return ((c_fine * w_fine + c_broad * (1 - w_fine)) * w_any).astype(np.float32)


def _chroma_denoise_lab(lab: np.ndarray, amount: float, px_scale: float) -> np.ndarray:
    """Smooth chrominance (OKLab a/b) while keeping luminance detail untouched."""
    if amount <= 0:
        return lab
    sigma = max(0.5, 4.0 * amount * px_scale)
    L = lab[..., 0]
    for c in (1, 2):
        ch = lab[..., c]
        blur = cv2.GaussianBlur(ch, (0, 0), sigma)
        blur2 = cv2.GaussianBlur(ch, (0, 0), sigma * 2.5)
        # fainter areas (lower SNR) get the broader smoothing
        w = smoothstep(np.median(L[::4, ::4]), np.median(L[::4, ::4]) + 0.25, cv2.GaussianBlur(L, (0, 0), 3))
        lab[..., c] = ch * (1 - amount) + amount * (w * blur + (1 - w) * blur2)
    return lab

def chroma_denoise(rgb: np.ndarray, amount: float, px_scale: float) -> np.ndarray:
    return oklab_to_rgb(_chroma_denoise_lab(rgb_to_oklab(rgb), amount, px_scale))


def _luminance_denoise_lab(lab: np.ndarray, amount: float, floors: np.ndarray | None = None,
                           noise_L: np.ndarray | None = None) -> np.ndarray:
    """Noise-adaptive starlet shrinkage of OKLab lightness after the stretch.

    Per-scale noise is estimated robustly (MAD) on the faintest pixels, where
    the stretch amplifies noise most; thresholds fade out in bright regions so
    that high-SNR detail is untouched.
    ``noise_L``: the lightness of a realisation of the image's own noise (same shape, through the
    same stretch); its per-scale MAD on those pixels is then the noise, as it is.  The image's own
    statistics cannot tell noise from signal on a restoration: its fine scales are the restored
    detail, far below the coadd's noise (NGC 6960: MAD 1.2 against 17.7 at the first scale), and
    shrinking at either level erased it.
    """
    if amount <= 0:
        return lab
    L = lab[..., 0]
    det, res = atrous(L, 4)
    Lb = cv2.GaussianBlur(L, (0, 0), 3)
    bg = np.median(Lb[::4, ::4])
    faint = Lb[::2, ::2] < np.percentile(Lb[::2, ::2], 40)
    w = 1 - 0.75 * smoothstep(bg, bg + 0.35, Lb)  # 1 in background, 0.25 in bright areas
    k = np.array([2.2, 1.6, 1.0, 0.5]) * amount
    ndet = atrous(noise_L, 4)[0] if noise_L is not None else None
    out = res.copy()
    for j, d in enumerate(det):
        if ndet is not None:
            sig = mad_sigma(ndet[j][::2, ::2][faint])
        else:
            sig = mad_sigma(d[::2, ::2][faint])
        if ndet is None and floors is not None:   # the data's real noise at this scale (see nonlinear_stage)
            sig = max(sig, float(floors[j]))
        t = k[j] * sig * w
        out += np.sign(d) * np.maximum(np.abs(d) - t, 0)
    lab[..., 0] = np.clip(out, 0, 1)
    return lab

def luminance_denoise(rgb: np.ndarray, amount: float) -> np.ndarray:
    return oklab_to_rgb(_luminance_denoise_lab(rgb_to_oklab(rgb), amount))


def neutralize_background(rgb: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Remove any colour cast of the darkest (sky) pixels after stretching/palette mapping."""
    L = luminance(rgb)
    Lb = cv2.GaussianBlur(L, (0, 0), 4)[::4, ::4]
    sel = Lb <= np.percentile(Lb, 20)
    sub = cv2.GaussianBlur(rgb, (0, 0), 4)[::4, ::4][sel]
    med = np.median(sub, axis=0)
    shift = (med - med.mean()) * strength
    # apply the offset fully in the background, fading out towards bright signal
    w = 1 - smoothstep(med.mean(), med.mean() + 0.35, L)
    return np.clip(rgb - shift[None, None, :] * w[..., None], 0, 1).astype(np.float32)


def _local_contrast_lab(lab: np.ndarray, amount: float, px_scale: float) -> np.ndarray:
    if amount <= 0:
        return lab
    L = lab[..., 0]
    levels = 7
    det, res = atrous(L, levels)
    bg = np.median(L[::4, ::4])
    sig = mad_sigma(det[0][::4, ::4])
    mask = smoothstep(bg + 2 * sig, bg + 0.15, cv2.GaussianBlur(L, (0, 0), 4))
    # boost medium/large structures (scaled for the working resolution)
    first = max(1, int(round(2 + np.log2(max(px_scale, 1e-3)))))
    boost = np.zeros(levels)
    boost[first:first + 4] = amount * np.array([0.35, 0.6, 0.6, 0.4])
    newL = res.copy()
    for j, d in enumerate(det):
        newL += d * (1 + boost[j] * mask)
    lab[..., 0] = np.clip(newL, 0, 1)
    return lab

def local_contrast(rgb: np.ndarray, amount: float, px_scale: float) -> np.ndarray:
    return oklab_to_rgb(_local_contrast_lab(rgb_to_oklab(rgb), amount, px_scale))


def _saturate_lab(lab: np.ndarray, amount: float) -> np.ndarray:
    L = lab[..., 0]
    bg = np.median(L[::4, ::4])
    m = smoothstep(bg, bg + 0.12, cv2.GaussianBlur(L, (0, 0), 2))
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    # vibrance-like: weaker colours get a larger lift, background is protected
    gain = 1 + (amount - 1) * m * (1.2 - np.clip(chroma / 0.25, 0, 1) * 0.5)
    lab[..., 1] *= gain
    lab[..., 2] *= gain
    return lab

def saturate(rgb: np.ndarray, amount: float) -> np.ndarray:
    return oklab_to_rgb(_saturate_lab(rgb_to_oklab(rgb), amount))


def scnr(rgb: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return rgb
    g = rgb[..., 1]
    neutral = np.minimum(g, (rgb[..., 0] + rgb[..., 2]) / 2)
    out = rgb.copy()
    out[..., 1] = g + amount * (neutral - g)
    return out


def curves(rgb: np.ndarray, black: float, brightness: float) -> np.ndarray:
    x = np.clip((rgb - black) / max(1 - black, 1e-3), 0, 1)
    if abs(brightness) > 1e-3:
        m = 0.5 - 0.35 * brightness  # MTF midtone balance
        x = ((m - 1) * x) / ((2 * m - 1) * x - m)
    return np.clip(x, 0, 1).astype(np.float32)


def sharpen(rgb: np.ndarray, amount: float, px_scale: float) -> np.ndarray:
    if amount <= 0:
        return rgb
    lab = rgb_to_oklab(rgb)
    L = lab[..., 0]
    s = max(0.6, 1.2 * px_scale)
    blur = cv2.GaussianBlur(L, (0, 0), s)
    detail = L - blur
    m = smoothstep(np.median(L[::4, ::4]) + 0.02, np.median(L[::4, ::4]) + 0.2, blur)
    lab[..., 0] = np.clip(L + amount * 1.5 * detail * m, 0, 1)
    return oklab_to_rgb(lab)


def is_narrowband(filter_name: str) -> bool:
    """A dual-band (Ha + OIII) filter: Seestar LP, DWARF Duo-Band, L-eNhance / L-eXtreme, ..."""
    import re
    n = re.sub(r"[^A-Z0-9]", "", (filter_name or "").upper())
    return n in {"LP", "DUO", "DUAL", "DUOBAND", "DUALBAND", "LENHANCE", "LEXTREME", "LULTIMATE", "HOO"}


_SEP_CACHE: dict = {}


def nonlinear_stage(lin: np.ndarray, params: dict, filter_name: str = "", px_scale: float = 1.0,
                    progress=None, return_layers: bool = False):
    """Everything after the linear stage. ``lin`` is normalised linear RGB (0..1)."""
    p = {**DEFAULTS, **(params or {})}
    palette = p["palette"]
    if palette == "auto":
        palette = "foraxx" if is_narrowband(filter_name) else "natural"
    target = float(p["stretch"])
    b = float(p["contrast"])
    step = [0]

    def tick(msg):
        step[0] += 1
        if progress:
            progress(step[0], 8, msg)

    # --- star separation (in linear space)
    tick("Separating stars")
    nref = p.get("_noise_ref")
    # a restoration (ImageMM) has no star halos: the halo light was deconvolved into the cores
    restored_sigma = p.get("_restored_sigma")
    detect_ref = p.get("_detect_ref")
    # (a restoration's stars are points, but the far halo - beyond its PSF kernels - is still there)
    lin, halo_w = neutralize_star_halos(lin, min(1.0, 1.6 * float(p["halo_suppress"])), px_scale,
                                        noise_ref=(nref * px_scale) if nref else None,
                                        detect_L=luminance(detect_ref) if detect_ref is not None else None,
                                        core_sigma=(restored_sigma * px_scale) if restored_sigma is not None else None)
    ml_starless = p.get("_starless")          # the AI star remover's starless image (starnet.py), same grid
    if ml_starless is not None and ml_starless.shape != lin.shape:
        ml_starless = None
    if p["star_separation"]:
        key = (lin.shape, float(px_scale), float(lin[::97, ::89].sum()),
               None if ml_starless is None else float(ml_starless[::97, ::89].sum()))
        if key in _SEP_CACHE:
            smask, starless, stars_lin = _SEP_CACHE[key]
        else:
            nref = p.get("_noise_ref")
            # (never below the data's real noise: a restoration's sky is exactly flat, its MAD 0, and at
            # full resolution every speckle then passed into the star layer)
            nsig = np.array([max(mad_sigma((lin[..., c] - cv2.GaussianBlur(lin[..., c], (0, 0), 1.5))[::3, ::3]),
                                 float(nref or 0.0) * px_scale * 0.5) for c in range(3)], np.float32)
            if ml_starless is not None:
                # the network removed stars *and* their halos (and any dark ring round them) and kept the
                # sky's own noise and texture, so the star layer is simply the difference: no threshold
                # (which clipped the faint wings), and the mask is wherever that layer rises above the noise
                starless = ml_starless
                stars_lin = np.maximum(lin - starless, 0)
                sl = luminance(stars_lin)
                smask = cv2.dilate((sl > 0.5 * float(nsig.mean())).astype(np.uint8),
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(np.float32)
                smask = cv2.morphologyEx(smask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            else:
                smask, starless = classic_star_separation(
                    lin, px_scale, noise_ref=(nref * px_scale) if nref else None,
                    detect_L=luminance(detect_ref) if detect_ref is not None else None,
                    restored_sigma=(restored_sigma * px_scale) if restored_sigma is not None else None)
                # linear star layer, soft-thresholded above the noise floor and confined to the mask
                diff = lin - starless
                soft = cv2.GaussianBlur(smask, (0, 0), 1.0)[..., None]
                stars_lin = np.maximum(diff - 2.5 * nsig, 0) * soft
            if len(_SEP_CACHE) >= 3:
                _SEP_CACHE.pop(next(iter(_SEP_CACHE)))
            _SEP_CACHE[key] = (smask, starless, stars_lin)
    else:
        smask = None
        starless = lin

    # --- stretch; the stretch curve is solved on the starless luminance so
    #     nebulosity (not stars) drives it, and reused for the star layer.
    tick("Stretching (GHS)")
    Lsl = luminance(starless)
    # adaptive strength: how bright the sky is pushed scales with how much of the frame
    # holds extended signal. Frame-filling nebulae keep the full target; small targets
    # in empty/star-dense fields get a darker, cleaner sky.
    if p.get("auto_stretch", True):
        # amount of large-scale structure (starless, heavily smoothed) in units of the
        # original per-pixel noise: ~2 for a frame-filling nebula, ~4 for a big galaxy,
        # ~0.5 for a small planetary nebula in an empty/star-dense field
        nref = p.get("_noise_ref")
        sm = cv2.GaussianBlur(Lsl, (0, 0), max(3.0, 12 * px_scale))[::4, ::4]
        unit = nref if nref else max(mad_sigma(Lsl[::4, ::4]), 1e-9)
        structure = float((np.percentile(sm, 90) - np.percentile(sm, 10)) / max(unit, 1e-12))
        target = target * (0.55 + 0.45 * np.clip((structure - 0.5) / 1.0, 0, 1))
    # the stretch is solved at one fixed scale (_stretch_proxy), with the data's real noise at that
    # scale as the floor of the background's (the coadd's per-pixel noise, averaged down with it)
    f_proxy = min(1.0, _PROXY_SIZE / max(Lsl.shape[:2]))
    nfloor = float(p.get("_noise_ref") or 0.0) * px_scale * f_proxy
    sm = p.get("stretch_method", "ghs")
    bp, D, sp = solve_stretch(_stretch_proxy(Lsl), target, b, nfloor, sm)
    # HDR: compress large-scale brightness above a knee (linear, multiplicative, so local
    # detail and colour ratios survive) – keeps bright compact objects (planetary
    # nebulae, galaxy cores, M42-type cores) from burning out.
    hdr = float(p["hdr"])
    if hdr > 0:
        grid = np.linspace(0, 1, 4097, dtype=np.float32) ** 3
        knee = 0.6
        yv = tone_fast(np.clip((grid - bp) / (1 - bp), 0, 1), D, b, sp, sm)
        x0 = float(grid[min(np.searchsorted(yv, knee), len(grid) - 1)])
        sig_b = max(4.0, 0.008 * max(Lsl.shape))
        # the brightness of *extended* structure only: stars (whatever is left of them in the starless
        # layer, or all of them without star separation) are removed by a morphological opening
        # first.  Otherwise a star's core drives the compression of its surroundings: a deconvolved
        # or restored star, its light packed into a few pixels, darkens a disc ~2 sig_b across
        k_open = max(5, int(round(1.5 * sig_b)) | 1)
        opened = cv2.morphologyEx(np.ascontiguousarray(Lsl, np.float32), cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open)))
        B = cv2.GaussianBlur(opened, (0, 0), sig_b)
        g = np.power(1 + np.maximum(B / max(x0, 1e-9) - 1, 0), -hdr).astype(np.float32)[..., None]
        if (g < 0.999).any():
            starless = starless * g + bp * (1 - g)
            if p["star_separation"]:
                stars_lin = stars_lin * g
            Lsl = luminance(starless)
    cp = float(p["color_preservation"])
    sl_s = apply_stretch(starless, bp, D, b, sp, cp, sm)

    tick("Palette")
    if palette in ("hoo", "foraxx", "hoo_warm"):
        # a restoration's own statistics are speckle: the leakage and the OIII scale come from its coadd
        hp = ha_oiii_params(detect_ref if detect_ref is not None else starless, bool(p["oiii_unmix"]))
        # the continuum's OIII/Ha: the stars', unmixed like the lines and before the gain
        cont = None
        if p["star_separation"]:
            sh_, so_ = extract_ha_oiii(stars_lin, unmix=bool(p["oiii_unmix"]), boost=1.0, params=(hp[0], 1.0))
            sh_, so_ = sh_ - np.median(sh_[::4, ::4]), so_ - np.median(so_[::4, ::4])
            sel_ = luminance(stars_lin) > np.percentile(luminance(stars_lin)[::4, ::4], 99)
            if sel_.sum() > 200 and float(np.sum(sh_[sel_])) > 0:
                cont = float(np.sum(so_[sel_]) / np.sum(sh_[sel_]))
        ha, oiii = extract_ha_oiii(starless, unmix=bool(p["oiii_unmix"]), boost=float(p["oiii_boost"]), neutral=halo_w,
                                   params=hp, continuum=cont)
        # identical stretch for both lines preserves their relative signal/noise
        bpn, Dn, spn = solve_stretch(_stretch_proxy(ha), target, b, nfloor, sm)
        line_s = lambda x: tone_fast(np.clip((x - bpn) / (1 - bpn), 0, 1), Dn, b, spn, sm)
        pal = palette_compose(line_s(ha), line_s(oiii), palette)
        # Colour comes from noise-suppressed line maps.  Built pixel by pixel, the palette turned the
        # independent noise of the Ha and OIII pixels into red / cyan colour wherever there is no line
        # signal - the whole sky - and the synthetic luminance below kept that chroma at full strength.
        # Chroma is taken at a fine scale where a line is significant there, at a broad scale where it
        # is only significant when averaged, and is neutral where neither line is significant at all.
        lab_c = _palette_chroma(ha, oiii, palette, line_s, px_scale)
        lab_p = rgb_to_oklab(pal)
        lab_p[..., 1:] = lab_c
        # LRGB-style: palette provides chrominance, the stretched all-channel image
        # provides lightness (synthetic luminance = best SNR, perceptually balanced)
        lm = float(p["synthetic_luminance"])
        if lm > 0:
            lab_l = rgb_to_oklab(sl_s)
            lab_p[..., 0] = (1 - lm) * lab_p[..., 0] + lm * lab_l[..., 0]
        # keep hue & chroma of the palette relative to the new lightness
        sl_s = oklab_to_rgb(lab_p)
    sl_s = neutralize_background(sl_s)
    tick("Local contrast")
    # the whole finishing chain runs in a single OKLab session (one conversion each way)
    lab = rgb_to_oklab(sl_s)
    # noise floors of the starlet scales after the stretch: Gaussian noise of the data's real per-pixel
    # level (the coadd's, at this scale) on the background, put through the same stretch.  A
    # restoration's sky is exactly flat over much of the field, so the shrinkage measured 0 noise
    # there at full resolution and left all the speckle in
    # With a restoration's own noise realisation (the N2N pass's residual, same grid) that goes through
    # the stretch instead: the coadd's noise is not the restoration's
    floors, noise_L = None, None
    resid = p.get("_noise_resid")
    bgl = float(np.median(Lsl[::4, ::4]))
    if resid is not None and resid.shape[:2] == lab.shape[:2] and float(p["luminance_denoise"]) > 0:
        nl = np.clip(bgl + luminance(resid), 0, 1)
        noise_L = rgb_to_oklab(np.repeat(apply_stretch(nl[..., None].repeat(3, -1), bp, D, b, sp, cp, sm)[..., :1], 3, -1))[..., 0]
    elif nref:
        rs_ = np.random.default_rng(0)
        patch = np.clip(bgl + rs_.normal(size=(256, 256)).astype(np.float32) * float(nref) * px_scale, 0, 1)
        pl_ = rgb_to_oklab(np.repeat(apply_stretch(patch[..., None].repeat(3, -1), bp, D, b, sp, cp, sm)[..., :1], 3, -1))[..., 0]
        floors = np.array([mad_sigma(dd) for dd in atrous(pl_, 4)[0]])
    lab = _luminance_denoise_lab(lab, float(p["luminance_denoise"]), floors, noise_L)
    lab = _local_contrast_lab(lab, float(p["local_contrast"]), px_scale)
    tick("Colour")
    lab = _chroma_denoise_lab(lab, float(p["chroma_denoise"]), px_scale)
    lab = _saturate_lab(lab, float(p["saturation"]))
    sl_s = oklab_to_rgb(lab)
    sl_s = scnr(sl_s, float(p["scnr"]) if palette in ("natural", "hoo") else float(p["scnr"]) * 0.5)

    tick("Stars")
    if p["star_separation"]:
        # stars use far less colour preservation than the nebula: bright stars (and
        # their chromatic halos) render white-ish like a linked stretch, instead of
        # blooming into saturated blue/violet discs; faint stars keep their colour.
        # Star brightness = luminance difference of the stretch with/without the star;
        # star colour = the star's own *linear* colour.  (Differencing RGB directly
        # tints stars on bright backgrounds, e.g. blue stars over a warm galaxy core,
        # because the stretch compresses each channel by a different amount.)
        cps = float(p["star_color_preservation"])
        # pure luminance stretch for the star brightness (no gamut effects)
        Ls0 = luminance(np.clip((starless - bp) / (1 - bp), 0, None))
        Ls1 = luminance(np.clip((starless + stars_lin - bp) / (1 - bp), 0, None))
        # "unscreen": the star layer that screen-blending reconstructs exactly
        T0, T1 = tone_fast(Ls0, D, b, sp, sm), tone_fast(Ls1, D, b, sp, sm)
        dL = np.clip(1 - (1 - T1) / np.maximum(1 - T0, 1e-4), 0, 1)[..., None]
        sl_lum = luminance(stars_lin)[..., None]
        chroma = stars_lin / np.maximum(sl_lum, 1e-9)
        chroma = np.where(sl_lum > 1e-9, chroma, 1.0)
        chroma = 1 + (np.clip(chroma, 0, 3) - 1) * cps     # 0 -> white stars, 1 -> full colour
        stars = np.clip(dL * chroma, 0, 1)
        red = float(p["star_reduction"])
        if red > 0:
            k = max(3, int(round(3 * px_scale)) | 1)
            er = cv2.erode(stars, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            # shrink: erosion blended where the star layer is faint (halo), keep cores
            stars = stars * (1 - red) + er * red
        # how far into its star's halo each pixel is: 0 at the core peak, -> 1 in the faint wings.
        # Each pixel is compared with the core peak of *its own* star (connected component of the
        # star mask), so ring-shaped chromatic halos are caught too
        sl_ = luminance(stars)
        n_lab, labels = cv2.connectedComponents((smask > 0.3).astype(np.uint8), connectivity=8)
        comp_max = np.zeros(n_lab, np.float32)
        np.maximum.at(comp_max, labels.ravel(), sl_.ravel())
        peak = comp_max[labels]
        k = max(5, int(round(15 * px_scale)) | 1)
        peak = np.where(labels > 0, peak, maximum_filter(sl_, size=k))
        halo = np.clip(1 - sl_ / np.maximum(peak, 1e-6), 0, 1)[..., None]
        # the colour star light is desaturated *towards*: neutral at the core, the colour of the
        # nebula underneath in the wings.  Desaturating the wings to grey painted a grey disc round
        # every star on coloured nebulosity (M 76's lobes); on dark sky the two are the same
        bg_rgb = cv2.GaussianBlur(sl_s, (0, 0), max(1.0, 2.0 * px_scale))
        bgc = np.clip(bg_rgb / np.maximum(luminance(bg_rgb)[..., None], 1e-4), 0, 3)
        tint = 1 + (bgc - 1) * halo                          # multiplies the luminance
        # halo suppression: pixels far below their star's local peak are halo – dim and
        # desaturate them (tames the OIII/blue bloat typical of small refractors)
        hs = float(p["halo_suppress"])
        if hs > 0:
            ls = sl_[..., None]
            # chromatic halo removal: refractors focus blue/violet differently, leaving
            # blue/purple rings. Pull blue down to the brighter of red/green, weighted
            # by how far into the halo the pixel is (cores keep their true colour).
            hw = np.clip(halo[..., 0] * 1.4, 0, 1) * hs
            rg = np.maximum(stars[..., 0], stars[..., 1])
            stars[..., 2] = stars[..., 2] - hw * np.maximum(stars[..., 2] - rg, 0)
            # purple (red+blue without green) -> neutral
            stars[..., 0] = stars[..., 0] - hw * np.maximum(stars[..., 0] - np.maximum(stars[..., 1], stars[..., 2]), 0) * 0.5
            sl_ = luminance(stars)
            tgt = sl_[..., None] * tint
            stars = tgt + (stars - tgt) * (1 - hs * halo)
            stars = stars * (1 - 0.6 * hs * halo ** 1.5)
        # star reduction beyond 0.5 also fades the star layer out: 1 = the starless image (with the AI
        # star remover a clean one; with the classic separation, the inpainted one)
        stars = stars * float(p["star_intensity"]) * (1.0 - float(smoothstep(0.5, 1.0, red)))
        if abs(p["star_saturation"] - 1) > 1e-3:
            ls = luminance(stars)[..., None]
            stars = np.clip(ls + (stars - ls) * float(p["star_saturation"]), 0, 1)
        if palette in ("hoo", "foraxx", "hoo_warm"):
            # dual-band star colours are unnatural (magenta); tame the green/magenta axis
            stars = scnr(stars, 0.5)
            tgt = luminance(stars)[..., None] * tint
            stars = np.clip(tgt + (stars - tgt) * 0.5, 0, 1)
        out = 1 - (1 - sl_s) * (1 - stars)  # screen blend
    else:
        out = sl_s
    tick("Finishing")
    out = curves(out, float(p["black_point"]), float(p["brightness"]))
    out = sharpen(out, float(p["sharpen"]), px_scale)
    out = np.clip(out, 0, 1).astype(np.float32)
    if return_layers:
        return out, {"starless": sl_s, "mask": smask, "palette": palette}
    return out
