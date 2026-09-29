"""Exposures for ImageMM: the data products of Sec. 2 of arXiv:2501.03002.

ImageMM needs, for every exposure t: the coregistered, background-subtracted
image y(t), per-pixel variances v(t), a binary mask m(t) and the PSF f(t)
measured from the exposure's stars.  This module derives them from the raw
Seestar subs.

Each step is linear in the sky signal, so the paper's model y = f * x + noise
holds for the prepared exposure, with f the PSF measured on it:

* Demosaic: bilinear interpolation (a fixed linear filter per colour).  An
  edge-aware demosaic is non-linear and would break the convolution model.
* Registration: the analysis stage's similarity transform and 3rd-order
  distortion polynomial (reference -> frame coordinates), evaluated exactly per
  pixel, then refined: windowed centroids (SExtractor WINPOS, Bertin & Arnouts
  1996) of reference-catalogue stars measured on the registered exposure give
  residual offsets, a robust 3rd-order polynomial of those offsets is composed
  into the mapping.  Resampling: Lanczos-4 (linear).
* Photometric scale: per colour (extinction is colour dependent), the median
  ratio of aperture fluxes of isolated, unsaturated stars in the exposure and in
  the reference coadd, with local annulus backgrounds and apertures of 3 FWHM so
  the ratio is independent of the seeing.
* Background: the scaled exposure minus the reference leaves the exposure's own
  smooth sky deviation, fitted with a robust 2nd-order surface; the reference's
  sky model is then subtracted too, so the latent sky is 0 as ImageMM assumes.
* Variances: photon-transfer curve (Janesick 2007) of the registered data,
  var = c0 + c1 * level in ADU per channel (read + sky noise, shot noise), fitted
  on differences of consecutive exposures in star-free pixels, binned by level,
  with robust (MAD) variances.  It is measured after demosaicing and resampling,
  so their effect on the per-pixel variance is included.
* Masks: outside the footprint of the Lanczos support, saturated and defective
  raw pixels (spread by the demosaic and resampling supports), and obstructed
  parts of the frame.
* PSFs: per exposure and colour, an empirical PSF: cut-outs of isolated,
  unsaturated reference stars around their *reference* positions (so the PSF
  also carries the exposure's residual registration), shifted onto the pixel
  grid with an exact Fourier shift, normalised by aperture flux, combined with a
  per-pixel sigma-clipped mean, negative wings set to 0 and normalised.
"""
from __future__ import annotations

import math
import time
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import cv2
import numpy as np
import sep

from .analysis import poly_eval, poly_terms
from .frames import cfa_masks, fix_defects, read_raw


# ----------------------------------------------------------------- demosaic
_K_RB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4
_K_G = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 4


def bilinear_demosaic(raw: np.ndarray, pattern: str, origin: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Bilinear demosaic of a CFA frame (or of a window starting at ``origin`` of it)."""
    oy, ox = origin
    masks = cfa_masks(pattern, (raw.shape[0] + 2, raw.shape[1] + 2))[oy % 2: oy % 2 + raw.shape[0],
                                                                       ox % 2: ox % 2 + raw.shape[1]]
    out = np.empty(raw.shape + (3,), np.float32)
    for c, k in ((0, _K_RB), (1, _K_G), (2, _K_RB)):
        out[..., c] = cv2.filter2D(raw * masks[..., c], -1, k, borderType=cv2.BORDER_CONSTANT)
        # at the frame border the neighbour sum is incomplete: normalise by the weights present
        norm = cv2.filter2D(masks[..., c], -1, k, borderType=cv2.BORDER_CONSTANT)
        out[..., c] /= np.maximum(norm, 1e-6)
    return out


# ----------------------------------------------------------------- geometry
def source_coords(fr: dict, W0: int, H0: int, xs: np.ndarray, ys: np.ndarray,
                  refine: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Frame (source) pixel coordinates of reference-grid points (xs, ys).

    ``refine``: residual offsets d(p) (reference coordinates) from ``refine_registration``;
    the frame content seen at reference p + d(p) belongs at p, so the mapping is
    evaluated at p + d(p)."""
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    if refine is not None:
        dx, dy = poly_eval(xs / W0 * 2 - 1, ys / H0 * 2 - 1, refine["deg"], refine["coefs"][0], refine["coefs"][1])
        xs, ys = xs + dx, ys + dy
    dist = fr.get("distortion")
    if dist is not None:
        return tuple(poly_eval(xs / W0 * 2 - 1, ys / H0 * 2 - 1, dist["deg"], dist["coefs"][0], dist["coefs"][1]))
    A = np.vstack([np.asarray(fr["transform"], np.float64), [0, 0, 1]])
    Ai = np.linalg.inv(A)[:2]
    return Ai[0, 0] * xs + Ai[0, 1] * ys + Ai[0, 2], Ai[1, 0] * xs + Ai[1, 1] * ys + Ai[1, 2]


def window_maps(fr, W0, H0, y0, y1, x0, x1, refine=None, rows: int = 128):
    """Float32 remap maps for the reference-grid window [y0, y1) x [x0, x1), evaluated
    exactly at every pixel (in blocks of ``rows`` rows to bound memory)."""
    mx = np.empty((y1 - y0, x1 - x0), np.float32)
    my = np.empty_like(mx)
    xs = np.arange(x0, x1, dtype=np.float64)
    for a in range(y0, y1, rows):
        b = min(a + rows, y1)
        xx, yy = np.meshgrid(xs, np.arange(a, b, dtype=np.float64))
        sx, sy = source_coords(fr, W0, H0, xx.ravel(), yy.ravel(), refine)
        mx[a - y0:b - y0] = sx.reshape(xx.shape)
        my[a - y0:b - y0] = sy.reshape(xx.shape)
    return mx, my


LANCZOS_R = 4          # half-width of the Lanczos-4 kernel (8 taps)


def warp_window(raw: np.ndarray, sat: np.ndarray, rep: np.ndarray, pattern: str, mx: np.ndarray, my: np.ndarray,
                rep_tol: float = 0.5):
    """Demosaic the part of the raw frame the window needs and resample it.

    ``sat``: saturated raw pixels (their values are wrong by an unknown amount: every
    output pixel whose demosaic + Lanczos support touches one is masked); ``rep``:
    defective (hot) raw pixels already repaired from the median of their 8 same-colour
    neighbours, a good estimate on smooth sky (error ~0.45 sigma): an output pixel is
    masked only where they carry more than ``rep_tol`` of its interpolation weight (it is
    then mostly an interpolation, not a measurement); the weights are the bilinear
    demosaic weights of the defects carried through the resampling.
    Returns (rgb, valid, hard) on the window: ``hard`` = full Lanczos support inside the
    frame and no saturated pixel in the support (for star measurements, where repaired
    pixels are acceptable estimates); ``valid`` = hard and not dominated by repaired pixels
    (the ImageMM mask m(t))."""
    H, W = raw.shape
    m = 3 + LANCZOS_R
    sx0 = int(max(0, math.floor(np.nanmin(mx)) - m)) // 2 * 2
    sy0 = int(max(0, math.floor(np.nanmin(my)) - m)) // 2 * 2
    sx1 = int(min(W, math.ceil(np.nanmax(mx)) + m + 1))
    sy1 = int(min(H, math.ceil(np.nanmax(my)) + m + 1))
    if sx1 <= sx0 + 2 * m or sy1 <= sy0 + 2 * m:
        return None, None, None
    rgb = bilinear_demosaic(raw[sy0:sy1, sx0:sx1], pattern, (sy0, sx0))
    lx, ly = mx - sx0, my - sy0
    out = cv2.remap(rgb, lx, ly, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    inside = ((mx >= 1 + LANCZOS_R) & (mx <= W - 2 - LANCZOS_R) &
              (my >= 1 + LANCZOS_R) & (my <= H - 2 - LANCZOS_R))
    valid = inside
    s_ = sat[sy0:sy1, sx0:sx1]
    if s_.any():
        b = cv2.dilate(s_.astype(np.uint8), np.ones((3, 3), np.uint8))                    # demosaic support
        b = cv2.dilate(b, np.ones((2 * LANCZOS_R + 2,) * 2, np.uint8))                     # Lanczos support
        valid = valid & (cv2.remap(b, lx, ly, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1) == 0)
    hard = valid.copy()
    r_ = rep[sy0:sy1, sx0:sx1]
    if r_.any():
        wdem = bilinear_demosaic(r_.astype(np.float32), pattern, (sy0, sx0))                # demosaic weight of defects
        wout = cv2.remap(wdem, lx, ly, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = valid & (wout.max(-1) <= rep_tol)
    return out, valid, hard


# ----------------------------------------------------------------- stars
def star_catalog(img: np.ndarray, sat: float, fwhm: float, thresh: float = 10.0):
    """Sources of a background-subtracted RGB coadd (luminance): WINPOS centroids, flux,
    peak, flags, and for each the distance to its nearest detected neighbour (3-sigma
    detections) and the flux of the brightest neighbour within 8 FWHM relative to it, so
    that every measurement can apply its own isolation criterion (``select``).  All
    3-sigma detections are kept for star masks."""
    from scipy.spatial import cKDTree
    L = np.ascontiguousarray(img.mean(-1), np.float32)
    bkg = sep.Background(L, bw=64, bh=64)
    sub = L - bkg.back()
    rms = bkg.globalrms
    allobj = sep.extract(sub, 3.0, err=rms, minarea=5)
    objs = sep.extract(sub, thresh, err=rms, minarea=5)
    wx, wy, wflag = sep.winpos(sub, objs["x"], objs["y"], np.full(len(objs), fwhm / 2.3548))
    ix = np.clip(np.round(wx).astype(int), 0, img.shape[1] - 1)
    iy = np.clip(np.round(wy).astype(int), 0, img.shape[0] - 1)
    peak_rgb = img[iy, ix].max(-1)
    tree = cKDTree(np.stack([allobj["x"], allobj["y"]], 1))
    d, j = tree.query(np.stack([objs["x"], objs["y"]], 1), k=2)
    nn = d[:, 1]                                        # [:, 0] is the source itself
    near = tree.query_ball_point(np.stack([objs["x"], objs["y"]], 1), 8 * fwhm)
    nflux = np.array([max([allobj["flux"][q] for q in lst if np.hypot(allobj["x"][q] - x0, allobj["y"][q] - y0) > 1.0],
                          default=0.0) for lst, x0, y0 in zip(near, objs["x"], objs["y"])])
    good = (objs["flag"] == 0) & (wflag == 0) & (peak_rgb < 0.5 * sat)
    return {"x": wx[good], "y": wy[good], "flux": objs["flux"][good], "peak": objs["peak"][good],
            "nn": nn[good], "nflux": nflux[good] / np.maximum(objs["flux"][good], 1e-12),
            "all_x": allobj["x"], "all_y": allobj["y"], "all_a": allobj["a"], "all_flux": allobj["flux"],
            "rms": rms}


def select(cat: dict, min_dist: float, max_nflux: float = np.inf) -> dict:
    """Catalogue subset whose nearest neighbour is farther than ``min_dist`` pixels, or whose
    neighbours within 8 FWHM are all fainter than ``max_nflux`` x the star."""
    ok = (cat["nn"] > min_dist) | (cat["nflux"] < max_nflux)
    return {k: (v[ok] if isinstance(v, np.ndarray) and len(v) == len(ok) and not k.startswith("all_") else v)
            for k, v in cat.items()}


def star_mask(shape, cat, fwhm: float, grow: float = 3.0) -> np.ndarray:
    """Pixels within ``grow`` FWHM (and 3 isophotal semi-axes) of any detected source."""
    m = np.zeros(shape, np.uint8)
    for x, y, a in zip(cat["all_x"], cat["all_y"], cat["all_a"]):
        cv2.circle(m, (int(round(x)), int(round(y))), int(math.ceil(max(grow * fwhm, 3 * a))), 1, -1)
    return m.astype(bool)


def refine_registration(L: np.ndarray, hard: np.ndarray, cat: dict, fwhm: float, W0: int, H0: int,
                        max_deg: int = 3, min_snr: float = 20.0) -> dict | None:
    """Residual registration offsets of a registered exposure (luminance L): WINPOS
    centroids of catalogue stars minus their reference positions, for stars with an
    exposure SNR >= ``min_snr`` and no neighbour within 3 FWHM.  Offsets are fitted by
    weighted least squares (weights SNR^2: a centroid's variance scales as 1/SNR^2),
    3-sigma clipped, with a polynomial whose degree (0 .. max_deg) minimises the AIC."""
    bkg = sep.Background(np.ascontiguousarray(L, np.float32), bw=64, bh=64)
    sub = np.ascontiguousarray(L - bkg.back(), np.float32)
    c = select(cat, 3 * fwhm)
    x, y = c["x"], c["y"]
    ix, iy = np.round(x).astype(int), np.round(y).astype(int)
    r = int(math.ceil(3 * fwhm))
    ok = (ix > r) & (iy > r) & (ix < L.shape[1] - r - 1) & (iy < L.shape[0] - r - 1)
    k = max(1, int(round(fwhm)))
    ok[ok] &= np.array([hard[j - k:j + k + 1, i - k:i + k + 1].all() for i, j in zip(ix[ok], iy[ok])], bool)
    if ok.sum() < 10:
        return None
    x, y = x[ok], y[ok]
    fl, fe, _ = sep.sum_circle(sub, x, y, 1.5 * fwhm, err=bkg.globalrms, mask=~hard)
    snr = fl / np.maximum(fe, 1e-12)
    s = snr >= min_snr
    if s.sum() < 10:
        return None
    x, y, snr = x[s], y[s], snr[s]
    wx, wy, flg = sep.winpos(sub, x, y, np.full(len(x), fwhm / 2.3548), mask=~hard)
    g = flg == 0
    x, y, snr, dx, dy = x[g], y[g], snr[g], (wx - x)[g], (wy - y)[g]
    w = snr ** 2
    u, v = x / W0 * 2 - 1, y / H0 * 2 - 1
    best = None
    for deg in range(0, max_deg + 1):
        Tm = poly_terms(u, v, deg)
        p = Tm.shape[1]
        if len(x) < 3 * p:
            break
        keep = np.ones(len(x), bool)
        for _ in range(10):
            sw = np.sqrt(w[keep])[:, None]
            cx = np.linalg.lstsq(Tm[keep] * sw, dx[keep] * sw[:, 0], rcond=None)[0]
            cy = np.linalg.lstsq(Tm[keep] * sw, dy[keep] * sw[:, 0], rcond=None)[0]
            res2 = ((Tm @ cx - dx) ** 2 + (Tm @ cy - dy) ** 2) * w
            s2 = np.sum(res2[keep]) / max(2 * (keep.sum() - p), 1)
            new = res2 < 9 * 2 * s2
            if (new == keep).all():
                break
            keep = new
        best = best or []
        best.append((deg, cx, cy, keep, s2, p))
    if not best:
        return None
    # AIC with the noise scale of the most flexible model (sigma^2 per unit weight)
    s2_ref, kp_ref = best[-1][4], best[-1][3]          # common inlier set and noise scale
    aic = [np.sum((((poly_terms(u, v, d) @ cx - dx) ** 2 + (poly_terms(u, v, d) @ cy - dy) ** 2) * w)[kp_ref]) / s2_ref
           + 2 * 2 * p for d, cx, cy, _, _, p in best]
    deg, cx, cy, keep, s2, p = best[int(np.argmin(aic))]
    Tm = poly_terms(u, v, deg)
    res = np.hypot(Tm @ cx - dx, Tm @ cy - dy)
    ww = w[keep] / w[keep].sum()
    return {"coefs": np.stack([cx, cy]), "deg": deg, "n": int(keep.sum()),
            "rms_before": float(np.sqrt(np.sum(ww * (dx[keep] ** 2 + dy[keep] ** 2)))),
            "rms_after": float(np.sqrt(np.sum(ww * res[keep] ** 2))),
            "noise_rms": float(np.sqrt(2 * s2 / np.mean(w[keep])))}


def compose_refine(a: dict | None, b: dict | None, W0: int, H0: int) -> dict | None:
    """Offsets of two successive refinements (a applied first): d(p) = a(p) + b(p) to first
    order in these sub-pixel offsets; polynomials of different degree are summed term by
    term (poly_terms orders the terms x^i y^j by i, then j)."""
    if a is None:
        return b
    if b is None:
        return a
    deg = max(a["deg"], b["deg"])
    full = [(i, j) for i in range(deg + 1) for j in range(deg + 1 - i)]

    def expand(r):
        own = [(i, j) for i in range(r["deg"] + 1) for j in range(r["deg"] + 1 - i)]
        out = np.zeros((2, len(full)))
        for q, t in enumerate(own):
            out[:, full.index(t)] = r["coefs"][:, q]
        return out
    return {"coefs": expand(a) + expand(b), "deg": deg, "n": b["n"], "rms_before": a["rms_before"],
            "rms_after": b["rms_after"], "noise_rms": b.get("noise_rms")}


def aperture_ratio(img: np.ndarray, ref: np.ndarray, hard: np.ndarray, cat: dict, radius: float,
                   min_snr: float = 50.0, min_stars: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel photometric scale img / ref: aperture fluxes (radius ~3 FWHM, so the
    ratio does not depend on the seeing) with local annulus backgrounds, of stars with no
    neighbour brighter than 1 % of them inside aperture + annulus.

    The scale is the weighted least-squares slope of f_img = T f_ref over the stars,
    T = sum(f_img f_ref / s^2) / sum(f_ref^2 / s^2), with s the exposure's aperture error
    from its background noise and 3-sigma clipping of the normalised residuals.  The
    reference is a deep coadd (its flux errors are small next to the exposure's), so the
    slope is unbiased and each star counts by its SNR; no star needs a high SNR of its own
    (a per-star ratio with weights from |f_img| would favour stars that came out faint, and
    a per-star SNR cut left the OIII-only blue channel of dual-band data, where a 60 s sub
    has almost no star at SNR 50, without any scale).  ``min_snr``: the stars' combined SNR
    in the exposure, sqrt(sum (f_ref T / s)^2), must reach it.
    Returns (scale, standard error) per channel."""
    c = select(cat, radius + 8, max_nflux=0.01)
    x, y = c["x"], c["y"]
    ix, iy = np.round(x).astype(int), np.round(y).astype(int)
    R = int(math.ceil(radius + 8))
    ok = (ix > R) & (iy > R) & (ix < img.shape[1] - R - 1) & (iy < img.shape[0] - R - 1)
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    rr = np.hypot(yy, xx)
    ap, ann = rr <= radius, (rr >= radius + 3) & (rr <= radius + 8)
    for j in np.nonzero(ok)[0]:
        w = hard[iy[j] - R:iy[j] + R + 1, ix[j] - R:ix[j] + R + 1]
        ok[j] = w[ap].all() and w[ann].mean() >= 0.5
    out, err = np.full(img.shape[2], np.nan), np.full(img.shape[2], np.nan)
    if ok.sum() < 5:
        return out, err
    x, y = x[ok], y[ok]
    for ch in range(img.shape[2]):
        a = np.ascontiguousarray(img[..., ch], np.float32)
        b = np.ascontiguousarray(ref[..., ch], np.float32)
        ea = sep.Background(a, bw=64, bh=64, mask=~hard).globalrms
        eb = sep.Background(b, bw=64, bh=64).globalrms
        fa, sa, _ = sep.sum_circle(a, x, y, radius, err=ea, bkgann=(radius + 3, radius + 8), mask=~hard, gain=None)
        fb, sb, _ = sep.sum_circle(b, x, y, radius, err=eb, bkgann=(radius + 3, radius + 8))
        # stars measured on the reference at SNR >= 10 (their reference flux is then a regressor
        # with <= 10 % error; the reference is the coadd of all subs, so most are far better)
        good = np.isfinite(fa) & np.isfinite(fb) & (fb > 10 * np.maximum(sb, 1e-12)) & (sa > 0)
        fa_, fb_, sa_ = fa[good], fb[good], sa[good]
        if len(fa_) < min_stars:
            continue
        keep = np.ones(len(fa_), bool)
        for _ in range(10):
            w = 1 / sa_[keep] ** 2
            T = np.sum(w * fa_[keep] * fb_[keep]) / np.sum(w * fb_[keep] ** 2)
            z = (fa_ - T * fb_) / sa_
            s = max(1.0, float(np.sqrt(np.sum(z[keep] ** 2) / max(keep.sum() - 1, 1))))
            new = np.abs(z) <= 3 * s
            if (new == keep).all() or new.sum() < min_stars:
                break
            keep = new
        w = 1 / sa_[keep] ** 2
        T = np.sum(w * fa_[keep] * fb_[keep]) / np.sum(w * fb_[keep] ** 2)
        z = (fa_[keep] - T * fb_[keep]) / sa_[keep]
        chi2 = float(np.sum(z ** 2) / max(keep.sum() - 1, 1))
        if not T > 0 or T * np.sqrt(np.sum(w * fb_[keep] ** 2)) < min_snr:
            continue
        out[ch] = T
        err[ch] = np.sqrt(max(chi2, 1.0) / np.sum(w * fb_[keep] ** 2))
    return out, err


def fourier_shift(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Exact (band-limited) sub-pixel shift of a 2-D array by (dy, dx)."""
    h, w = img.shape
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.rfftfreq(w)[None, :]
    return np.fft.irfft2(np.fft.rfft2(img) * np.exp(-2j * np.pi * (ky * dy + kx * dx)), s=(h, w))


def _psf_star_table(cat: dict, fwhm: float, half: int) -> list:
    """PSF-star candidates of a reference catalogue, brightest first: (index, integer centre,
    neighbour mask of the cut-out).  Depends only on the catalogue, the FWHM and the cut-out
    size, so it is built once and shared by every exposure and channel."""
    from scipy.spatial import cKDTree
    tables = cat.setdefault("_psf_tables", {})
    key = (len(cat["x"]), round(float(fwhm), 3), int(half))
    if key in tables:
        return tables[key]
    rc = max(3.0, 2.0 * fwhm)
    pad = half + 8
    yy, xx = np.mgrid[-pad:pad + 1, -pad:pad + 1]
    x, y, f = np.asarray(cat["x"], float), np.asarray(cat["y"], float), np.asarray(cat["flux"], float)
    if "all_x" in cat and len(cat["all_x"]):
        ax, ay = np.asarray(cat["all_x"], float), np.asarray(cat["all_y"], float)
        af = np.asarray(cat["all_flux"], float)
        beta = 2.5
        alpha = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
        peak = np.maximum(af, 0) * (beta - 1) / (np.pi * alpha ** 2)
        thr = 0.5 * float(cat.get("rms", 0.0) or 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_light = np.where(peak > thr, alpha * np.sqrt(np.maximum((peak / max(thr, 1e-30)) ** (1 / beta) - 1, 0)),
                               0.0)
        rn = np.maximum(r_light, 3 * np.asarray(cat["all_a"], float))
        tree = cKDTree(np.stack([ax, ay], 1))
        reach = pad * np.sqrt(2) + rn.max()
    else:                                                # no neighbour list: nearest-neighbour distances only
        tree = None
    table = []
    for i in np.argsort(-f):
        ix, iy = int(round(x[i])), int(round(y[i]))
        m = np.zeros(yy.shape, bool)
        if tree is not None:
            nb = [j for j in tree.query_ball_point([x[i], y[i]], reach) if np.hypot(ax[j] - x[i], ay[j] - y[i]) > 1.0]
            if nb:
                d = np.hypot(ax[nb] - x[i], ay[nb] - y[i])
                if np.any(d <= rc + rn[nb]):             # a neighbour's light reaches the core
                    continue
                for j in nb:
                    m |= np.hypot(yy - (ay[j] - iy), xx - (ax[j] - ix)) <= rn[j]
        elif "nn" in cat and cat["nn"][i] <= rc + 3 * fwhm:
            continue
        table.append((int(i), ix, iy, m))
    tables[key] = table
    return table


def _psf_cutouts(img: np.ndarray, valid: np.ndarray, cat: dict, half: int, fwhm: float, max_stars: int):
    """Flux-normalised, centred PSF-star cut-outs of one background-subtracted channel (see
    ``empirical_psf``).  Returns (cut-outs (N, n, n), weights (N, n, n), per-cut-out noise in
    normalised units (N,), star x (N,), star y (N,))."""
    rc = max(3.0, 2.0 * fwhm)                            # core aperture (normalisation)
    pad = half + 8
    yy, xx = np.mgrid[-pad:pad + 1, -pad:pad + 1]
    x, y = np.asarray(cat["x"], float), np.asarray(cat["y"], float)
    core = np.hypot(yy, xx)[8:-8, 8:-8] <= rc
    cuts, wts, noises, px, py = [], [], [], [], []
    for i, ix, iy, nmask in _psf_star_table(cat, fwhm, half):
        if len(cuts) >= max_stars:
            break
        if ix - pad < 0 or iy - pad < 0 or ix + pad + 1 > img.shape[1] or iy + pad + 1 > img.shape[0]:
            continue
        m = nmask | ~valid[iy - pad:iy + pad + 1, ix - pad:ix + pad + 1]
        rr = np.hypot(yy - (y[i] - iy), xx - (x[i] - ix))
        if m[rr <= rc + 2].any():
            continue
        ann = (rr > half + 2) & (rr <= half + 8) & ~m
        if ann.sum() < 20:
            continue
        cc = img[iy - pad:iy + pad + 1, ix - pad:ix + pad + 1].astype(np.float64)
        bg = np.median(cc[ann])
        noise = 1.4826 * np.median(np.abs(cc[ann] - bg))                      # this cut-out's pixel noise
        cc = np.where(m, 0.0, cc - bg)                                        # local residual sky; neighbours out
        cc = fourier_shift(cc, -(y[i] - iy), -(x[i] - ix))[8:-8, 8:-8]       # star centre -> pixel centre
        mk = fourier_shift(m.astype(np.float64), -(y[i] - iy), -(x[i] - ix))[8:-8, 8:-8] > 0.02
        flux = cc[core].sum()
        if flux <= 0:
            continue
        cuts.append(cc / flux)
        wts.append(np.where(mk, 0.0, flux ** 2))
        noises.append(noise / flux)                                          # in units of the normalised cut-out
        px.append(x[i])
        py.append(y[i])
    n = 2 * half + 1
    if not cuts:
        z = np.zeros((0, n, n))
        return z, z, np.zeros(0), np.zeros(0), np.zeros(0)
    return np.stack(cuts), np.stack(wts), np.asarray(noises), np.asarray(px), np.asarray(py)


def _point_sources(S: np.ndarray, noises: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    """Cut-outs that show a point source (see ``empirical_psf``): the concentration c (mean of
    the normalised cut-out within 1.5 px of the centre) must agree with its expectation within
    4 x (its noise combined with the genuine star-to-star spread of the brightest fifth).  The
    expectation is the median, or - with a basis ``B`` (N, P) of the stars' field positions - a
    robust polynomial fit over the field, since a field-dependent PSF changes c from the centre
    to the corners of the field by more than the noise of a bright star."""
    R0 = np.hypot(*np.mgrid[:S.shape[1], :S.shape[2]] - S.shape[1] // 2)
    centre = R0 < 1.5
    c = S[:, centre].mean(1)
    e_noise = np.asarray(noises) / np.sqrt(centre.sum())
    if B is None:
        expect = np.full(len(c), float(np.median(c)))
    else:
        keep = np.ones(len(c), bool)
        for _ in range(10):                              # iteratively reweighted (clipped) fit
            coef = np.linalg.lstsq(B[keep], c[keep], rcond=None)[0]
            res = c - B @ coef
            s = 1.4826 * np.median(np.abs(res[keep]))
            new = np.abs(res) <= 3 * max(s, 1e-12) + 4 * e_noise
            if (new == keep).all() or new.sum() < B.shape[1] + 3:
                break
            keep = new
        expect = B @ coef
    fl = 1 / np.maximum(e_noise, 1e-30)                  # ~ flux / noise
    bright = fl >= np.percentile(fl, 80)
    d = c - expect
    s_int = 1.4826 * float(np.median(np.abs(d[bright] - np.median(d[bright])))) if bright.sum() >= 5 else 0.0
    return np.abs(d) <= 4 * np.sqrt(e_noise ** 2 + s_int ** 2)


def _support_radius(mu: np.ndarray, se: np.ndarray, half: int, nsig: float) -> float:
    """First 1-px annulus (beyond 2 FWHM of ``mu``) whose mean is below ``nsig`` standard errors."""
    n = mu.shape[0]
    rr = np.hypot(*np.mgrid[:n, :n] - n // 2)
    fw = 2 * np.sqrt(max((mu * rr ** 2).sum() / max(mu.sum(), 1e-12), 0) / 2) * 1.1774   # 2nd-moment FWHM
    for k in range(1, half + 1):
        ring = (rr >= k - 0.5) & (rr < k + 0.5) & np.isfinite(se)
        if not ring.any():
            continue
        m_ = mu[ring].mean()
        e_ = np.sqrt((se[ring] ** 2).sum()) / ring.sum()
        if k >= 2 * fw and m_ < nsig * e_:
            return k - 0.5
    return float(half)


def _finish_psf(mu: np.ndarray, rsup: float) -> np.ndarray:
    """Cut at the support radius, negative pixels to 0, unit sum."""
    n = mu.shape[-1]
    rr = np.hypot(*np.mgrid[:n, :n] - n // 2)
    psf = np.clip(np.where(rr <= rsup, mu, 0.0), 0, None)
    return (psf / np.maximum(psf.sum((-2, -1), keepdims=True), 1e-300)).astype(np.float32)


def empirical_psf(img: np.ndarray, valid: np.ndarray, cat: dict, half: int, clip: float = 3.0,
                  min_stars: int = 10, return_error: bool = False, nsig: float = 2.0, fwhm: float | None = None,
                  max_stars: int = 400):
    """Empirical PSF of one background-subtracted channel.

    Crowded fields (the Milky Way: M 27's median nearest neighbour is ~11 px) have almost no
    stars without a neighbour inside a PSF cut-out, so neighbours are handled as crowded-field
    PSF builders do (DAOPHOT; Stetson 1987): every other detection near a star is masked out
    to the radius where its light falls below half the reference coadd's per-pixel sky noise
    (from its flux, the FWHM and a Moffat profile with beta = 2.5, heavier-winged than the
    subs' measured beta ~ 2.8, so the masks err large), and at least 3 isophotal semi-axes;
    a catalogue star is a PSF star when no mask reaches its core aperture (radius 2 FWHM).
    Masked pixels get zero weight.

    Each cut-out: local residual sky from the unmasked annulus median, masked pixels zeroed,
    shifted onto the pixel grid by an exact Fourier shift (the mask is shifted with it), and
    normalised by its core flux (radius 2 FWHM, unmasked by construction).  The cut-outs are
    combined by a per-pixel sigma-clipped weighted mean with weights core flux^2 on unmasked
    pixels (a flux-normalised cut-out of a sky-limited star has variance sigma_sky^2 /
    flux^2).

    The mean is noisy in the far wings, and clipping its negative pixels to zero before
    normalising would add a positive pedestal there (on M 27 subs it put 20-40 % of the
    flux beyond 2 FWHM).  So the kernel is cut at the support radius where the azimuthally
    averaged profile is no longer significant (< ``nsig`` times its standard error), at
    least 2 FWHM; only then are the few remaining negative pixels set to 0 and the kernel
    normalised to unit sum.  ``return_error``: also return (unclipped mean, per-pixel
    standard error, support radius) for model fitting.

    This is one PSF for the stars of the whole image; ``empirical_psf_field`` models its
    variation over the field.
    Returns (PSF of (2 half + 1)^2 pixels with unit sum, number of stars used[, extras])."""
    fwhm = float(fwhm) if fwhm else half / 3.5
    # rounded up to 0.1 px: the neighbour masks and the core are then never smaller than for the
    # exact FWHM, and the star table is built once per 0.1 px of seeing instead of once per sub
    fwhm = math.ceil(fwhm * 10 - 1e-9) / 10
    S, w, noises, _, _ = _psf_cutouts(img, valid, cat, half, fwhm, max_stars)
    if len(S) < min_stars:
        return (None, len(S), None) if return_error else (None, len(S))
    # Every cut-out must show the same PSF before the flux^2-weighted mean: the weights are the
    # inverse variances *of a point source's* estimate of the PSF, and a bright non-point source (a
    # galaxy or cluster core, a blend) gets a large weight while its profile is not the PSF.  On
    # M 31 one such cut-out carried 24 % of the weight (its core 4.5x less concentrated than the
    # median cut-out's, its wings 50x stronger) and the PSF wings came out ~16x too strong, so the
    # restoration zeroed a disc around every star to compensate; the per-pixel sigma clipping could
    # not catch it, being centred on the weighted mean it dominated (``_point_sources``).
    point = _point_sources(S, noises)
    if point.sum() < min_stars:
        return (None, int(point.sum()), None) if return_error else (None, int(point.sum()))
    S, w = S[point], w[point]
    keep = w > 0
    for _ in range(10):
        sw = (w * keep).sum(0)
        mu = (S * w * keep).sum(0) / np.maximum(sw, 1e-300)
        # weighted scatter -> standard error of the weighted mean
        var = (w * keep * (S - mu) ** 2).sum(0) / np.maximum(sw, 1e-300)
        neff = sw ** 2 / np.maximum((w ** 2 * keep).sum(0), 1e-300)
        sd = np.sqrt(var * neff / np.maximum(neff - 1, 1))
        new = (w > 0) & (np.abs(S - mu) <= clip * np.maximum(sd, 1e-12))
        if (new == keep).all():
            break
        keep = new
    se = np.where(neff > 1, sd / np.sqrt(np.maximum(neff, 1)), np.inf)
    rsup = _support_radius(mu, se, half, nsig)
    psf = _finish_psf(mu, rsup)
    if return_error:
        n = mu.shape[0]
        rr = np.hypot(*np.mgrid[:n, :n] - n // 2)
        tot = max(float(np.clip(np.where(rr <= rsup, mu, 0), 0, None).sum()), 1e-300)
        se_out = np.where(np.isfinite(se), se, 1e30 * tot)
        return psf, len(S), {"mean": (mu / tot).astype(np.float32), "se": (se_out / tot).astype(np.float32),
                             "support": rsup}
    return psf, len(S)


def psf_nodes(H: int, W: int, spacing: float = 900.0) -> tuple[np.ndarray, np.ndarray]:
    """Node positions (reference-grid rows, columns) of the field PSF grid: evenly spaced from
    edge to edge, at most ``spacing`` pixels apart, at least 2 per axis."""
    ny = max(2, int(math.ceil((H - 1) / spacing)) + 1)
    nx = max(2, int(math.ceil((W - 1) / spacing)) + 1)
    return np.linspace(0, H - 1, ny), np.linspace(0, W - 1, nx)


def empirical_psf_field(img: np.ndarray, valid: np.ndarray, cat: dict, half: int, fwhm: float,
                        nodes: tuple[np.ndarray, np.ndarray], deg: int = 2, clip: float = 3.0,
                        min_stars: int = 10, nsig: float = 2.0, max_stars: int = 1000) -> dict | None:
    """Field-dependent empirical PSF of one background-subtracted channel (as PSFEx, Bertin
    2011, and the HSC pipeline model it): every PSF pixel is a polynomial of degree ``deg`` in
    the field position, fitted by weighted least squares to the cut-outs of ``empirical_psf``
    (same selection, centring, normalisation, flux^2 weights and per-pixel 3-sigma clipping,
    now about the local model instead of the field mean).

    Why: the Seestar's field curvature and off-axis aberrations change the star FWHM by ~40 %
    from the centre to the top of the frame (IC 405, NGC 6960: 4.2 px -> 5.8 px).  ImageMM's
    model y = f * x assumes the PSF of the cutout it restores (the paper takes the HSC PSF at
    each cutout's position); one field-averaged PSF is too broad at the centre - the latent
    then carves a dark ring around every bright star to cancel the excess wings - and too
    narrow at the edges.

    The degree drops while there are fewer than 12 stars per polynomial term.  The support
    radius comes from the field-mean PSF and its standard error (the noise of the wings sets
    it, and it must be one radius for the whole field).  The model is evaluated at ``nodes``
    (rows, columns of the reference grid; ``psf_nodes``), each cut at the support radius,
    negative pixels set to 0, unit sum; kernels between nodes are interpolated bilinearly.
    Returns None with fewer than ``min_stars`` point-source cut-outs, else {"nodes":
    (ny, nx, n, n) kernels, "mean": (ny, nx, n, n) unclipped models with unit sum inside the
    support, "se": (n, n) standard error of the field mean on that scale, "support", "deg",
    "n_stars", "psf": field-mean kernel}."""
    fwhm = math.ceil(float(fwhm) * 10 - 1e-9) / 10
    S, w, noises, px, py = _psf_cutouts(img, valid, cat, half, fwhm, max_stars)
    if len(S) < min_stars:
        return None
    H, W = img.shape
    uv = lambda xx, yy: (np.asarray(xx, float) / max(W - 1, 1) * 2 - 1, np.asarray(yy, float) / max(H - 1, 1) * 2 - 1)
    P = lambda d: (d + 1) * (d + 2) // 2
    while deg > 0 and len(S) < 12 * P(deg):
        deg -= 1
    B = poly_terms(*uv(px, py), deg)
    point = _point_sources(S, noises, B if deg > 0 else None)
    if point.sum() < min_stars:
        return None
    S, w, px, py = S[point], w[point], px[point], py[point]
    while deg > 0 and len(S) < 12 * P(deg):
        deg -= 1
    B = poly_terms(*uv(px, py), deg)
    N, n, _ = S.shape
    Pn = B.shape[1]
    Sf, wf = S.reshape(N, -1), w.reshape(N, -1)
    keep = wf > 0
    ridge = np.zeros((Pn, Pn))
    ridge[np.arange(1, Pn), np.arange(1, Pn)] = 1e-6       # keeps pixels seen by few stars solvable
    for _ in range(10):
        ww = wf * keep                                     # (N, n^2)
        A = np.einsum("ip,iq,ij->jpq", B, B, ww)           # normal equations per pixel
        A += ridge[None] * np.maximum(np.trace(A, axis1=1, axis2=2), 1e-300)[:, None, None]
        rhs = np.einsum("ip,ij->jp", B, ww * Sf)
        coef = np.linalg.solve(A, rhs[..., None])[..., 0]  # (n^2, P)
        res = Sf - B @ coef.T
        sw = ww.sum(0)
        var = (ww * res ** 2).sum(0) / np.maximum(sw, 1e-300)
        neff = sw ** 2 / np.maximum((ww ** 2).sum(0), 1e-300)
        sd = np.sqrt(var * neff / np.maximum(neff - Pn, 1))
        new = (wf > 0) & (np.abs(res) <= clip * np.maximum(sd, 1e-12))
        if (new == keep).all():
            break
        keep = new
    coef = coef.T.reshape(Pn, n, n)
    se = np.where(neff > Pn, sd / np.sqrt(np.maximum(neff, 1)), np.inf).reshape(n, n)
    # field mean of the model: the mean of the polynomial over the (uniformly covered) field
    gy, gx = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9), indexing="ij")
    mu_bar = np.tensordot(poly_terms(gx.ravel(), gy.ravel(), deg).mean(0), coef, 1)
    rsup = _support_radius(mu_bar, se, half, nsig)
    rr = np.hypot(*np.mgrid[:n, :n] - n // 2)
    inside = rr <= rsup
    ny_, nx_ = len(nodes[0]), len(nodes[1])
    ny, nx = np.meshgrid(nodes[0], nodes[1], indexing="ij")
    Bn = poly_terms(*uv(nx.ravel(), ny.ravel()), deg)                 # (ny*nx, P)
    mu_n = np.tensordot(Bn, coef, 1)                                    # (ny*nx, n, n)
    tot = np.maximum(np.clip(np.where(inside, mu_n, 0), 0, None).sum((1, 2)), 1e-300)
    tot_bar = max(float(np.clip(np.where(inside, mu_bar, 0), 0, None).sum()), 1e-300)
    return {"nodes": _finish_psf(mu_n, rsup).reshape(ny_, nx_, n, n),
            "mean": (mu_n / tot[:, None, None]).astype(np.float32).reshape(ny_, nx_, n, n),
            "se": np.where(np.isfinite(se), se / tot_bar, 1e30).astype(np.float32),
            "support": rsup, "deg": deg, "n_stars": int(N), "psf": _finish_psf(mu_bar, rsup)}


def hybrid_psf(sub: dict, ref: dict, fwhm: float, r_fit: float = 2.5,
               scales: np.ndarray = np.round(np.arange(0.60, 1.80001, 0.02), 2)) -> dict:
    """Per-exposure PSF field: the deep reference coadd's field PSF, radially scaled to the
    exposure's seeing.

    A single sub cannot measure its own PSF shape: its per-pixel noise is comparable to the PSF
    beyond ~1 FWHM, and a kernel must be non-negative, so the noise clipped at 0 inflates the
    profile.  On M 31's 10 s subs the sub kernels averaged 0.6-0.7x the coadd PSF's peak and 2-5x
    its value at 5-10 px; on IC 405's 60 s subs the wings alone were ~6 % of the flux too strong.
    ImageMM, which cannot put negative flux on the sky, then emptied a moat round every star
    (the dark rings, 2 % of a star's peak at 3-9 px on M 31, deepest in R and B).

    The coadd averages every sub, so its shape - core, field-dependent aberrations and wings - is
    measured ~sqrt(n) times better.  What differs between subs is mainly the seeing, a single
    number: at every node and channel the exposure's kernel is the coadd's, radially scaled by s
    (K_s(x) = K(x / s) / s^2, renormalised), with s and an amplitude fitted by weighted least
    squares to the exposure's own unclipped mean profile inside ``r_fit`` FWHM (its noise is
    zero-mean there, so the fit is unbiased; no clipping is involved) - one s per exposure and
    channel, the median over the nodes.
    Returns ``sub`` with "nodes", "mean", "psf" replaced and "scale" (ny, nx) added."""
    Kr = ref["nodes"]                                     # (ny, nx, nr, nr), clipped, unit sum
    ms, se = sub["mean"], sub["se"]                       # (ny, nx, n, n) unclipped; (n, n)
    ny, nx, n, _ = ms.shape
    nr = Kr.shape[-1]
    N = max(n, int(np.ceil(nr * scales.max())) | 1)
    pad = lambda z, m: np.pad(z, [(0, 0)] * (z.ndim - 2) + [((N - m) // 2, (N - m) // 2)] * 2)
    Kr, ms = pad(Kr, nr), pad(ms, n)
    se = np.pad(se, (N - n) // 2, constant_values=1e30)
    c = N // 2
    Ks = np.empty((len(scales), ny, nx, N, N), np.float32)
    for q, sc in enumerate(scales):
        M_ = np.float32([[1 / sc, 0, c - c / sc], [0, 1 / sc, c - c / sc]])       # dst x <- src c + (x - c) / s
        for i in range(ny):
            for j in range(nx):
                k = cv2.warpAffine(Kr[i, j].astype(np.float32), M_, (N, N), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                k = np.clip(k, 0, None)
                Ks[q, i, j] = k / max(float(k.sum()), 1e-30)
    rr = np.hypot(*np.mgrid[:N, :N] - c)
    fit = (rr <= r_fit * fwhm) & (se < 1e29)
    w = 1.0 / np.maximum(se[fit], 1e-30) ** 2
    A = Ks[..., fit]                                      # (S, ny, nx, P)
    m = ms[..., fit][None]                                # (1, ny, nx, P)
    amp = (w * A * m).sum(-1) / np.maximum((w * A * A).sum(-1), 1e-300)
    chi = (w * (m - amp[..., None] * A) ** 2).sum(-1)    # (S, ny, nx)
    best = np.argmin(chi, 0)                              # per node
    # the seeing is one number per exposure (the field dependence is already in the coadd's PSF);
    # single nodes fit badly where the exposure has few PSF stars - on a galaxy disc, the sub's own
    # field model is an extrapolation and its fit ran to the bounds (M 31: 20 % of the nodes) - so
    # the scale is the median of the node fits that stay inside the range, used at every node
    inner = (best > 0) & (best < len(scales) - 1)
    q = int(np.round(np.median(best[inner]))) if inner.any() else int(np.argmin(np.abs(scales - 1.0)))
    best = np.full_like(best, q)
    K = np.take_along_axis(Ks, best[None, ..., None, None], 0)[0]
    # only as large as the scaled support (the range of s left room for up to 1.8 x): the cost of
    # the restoration grows with the square of the kernel size
    h_ = min(c, int(np.ceil(float(ref["support"]) * scales[q])) + 1)
    K = K[..., c - h_:c + h_ + 1, c - h_:c + h_ + 1]
    K = K / np.maximum(K.sum((-2, -1), keepdims=True), 1e-30)
    se = se[c - h_:c + h_ + 1, c - h_:c + h_ + 1]
    out = dict(sub)
    out.update({"nodes": K.astype(np.float32), "mean": K.astype(np.float32), "psf": K.mean((0, 1)).astype(np.float32),
                "scale": scales[best].astype(np.float32), "support": float(ref["support"]),
                "se": np.where(se < 1e29, se, 1e30).astype(np.float32)})
    return out


def interp_nodes(K: np.ndarray, nodes: tuple[np.ndarray, np.ndarray], y: float, x: float) -> np.ndarray:
    """Bilinear interpolation of node kernels K (..., ny, nx, k, k) at reference position (y, x)
    (clamped to the node grid).  A convex combination of unit-sum kernels has unit sum."""
    ys, xs = nodes
    fy = float(np.clip(np.interp(y, ys, np.arange(len(ys))), 0, len(ys) - 1))
    fx = float(np.clip(np.interp(x, xs, np.arange(len(xs))), 0, len(xs) - 1))
    iy, ix = min(int(fy), len(ys) - 2), min(int(fx), len(xs) - 2)
    ty, tx = fy - iy, fx - ix
    return ((1 - ty) * (1 - tx) * K[..., iy, ix, :, :] + (1 - ty) * tx * K[..., iy, ix + 1, :, :]
            + ty * (1 - tx) * K[..., iy + 1, ix, :, :] + ty * tx * K[..., iy + 1, ix + 1, :, :])


# ----------------------------------------------------------------- Moffat model
def moffat_image(p, size: int, sub: int = 9) -> np.ndarray:
    """Elliptical Moffat profile (Moffat 1969; elliptical form as in Trujillo et al. 2001)
    I = A [1 + (x'/a1)^2 + (y'/a2)^2]^-beta, (x', y') rotated by theta about (x0, y0)
    relative to the centre pixel, integrated over each pixel (sub x sub samples)."""
    A, x0, y0, a1, a2, theta, beta = p
    c = size // 2
    o = (np.arange(size * sub) + 0.5) / sub - 0.5 - c
    yy, xx = np.meshgrid(o - y0, o - x0, indexing="ij")
    ct, st = np.cos(theta), np.sin(theta)
    u, v = ct * xx + st * yy, -st * xx + ct * yy
    img = A * (1 + (u / a1) ** 2 + (v / a2) ** 2) ** (-beta)
    return img.reshape(size, sub, size, sub).mean((1, 3))


def fit_moffat(psf: np.ndarray, err: np.ndarray, ee: float = 0.995, max_half: int | None = None) -> tuple[np.ndarray, dict]:
    """Weighted least-squares fit of the pixel-integrated elliptical Moffat to an empirical
    PSF (the unclipped mean) with per-pixel standard errors ``err``.

    The model kernel extends to the radius holding ``ee`` of its analytic flux
    (enclosed energy within the ellipse of scaled radius R: 1 - (1 + R^2)^(1 - beta)), capped
    at ``max_half`` (default: twice the fitted cut-out's half-size); it is normalised to unit
    sum over that extent and the enclosed fraction is reported (``ee_kernel``)."""
    from scipy.optimize import least_squares
    n = psf.shape[0]
    c = n // 2
    max_half = max_half or 2 * c
    yy, xx = np.mgrid[:n, :n] - c
    m0 = max(psf.sum(), 1e-12)
    sx = np.sqrt(max((psf * xx ** 2).sum() / m0, 0.25))
    fw = 2.3548 * sx
    p0 = [psf.max(), 0.0, 0.0, fw, fw, 0.0, 2.5]
    w = 1 / np.maximum(err, 1e-6 * psf.max())

    def resid(p):
        return ((moffat_image(p, n) - psf) * w).ravel()

    lo = [0, -2, -2, 0.1, 0.1, -np.pi, 1.01]
    hi = [np.inf, 2, 2, 10 * n, 10 * n, np.pi, 30]
    sol = least_squares(resid, p0, bounds=(lo, hi), x_scale="jac")
    A, x0, y0, a1, a2, th, beta = sol.x
    R = np.sqrt(np.exp(np.log(1 - ee) / (1 - beta)) - 1) if beta > 1.0001 else np.inf
    half = int(min(max_half, np.ceil(R * max(a1, a2) + max(abs(x0), abs(y0)))))
    model = moffat_image(sol.x, 2 * half + 1)
    total = np.pi * a1 * a2 * A / (beta - 1)
    info = {"A": A, "x0": x0, "y0": y0, "alpha1": a1, "alpha2": a2, "theta": th, "beta": beta,
            "fwhm1": 2 * a1 * np.sqrt(2 ** (1 / beta) - 1), "fwhm2": 2 * a2 * np.sqrt(2 ** (1 / beta) - 1),
            "chi2_red": float(np.sum(sol.fun ** 2) / max(sol.fun.size - 7, 1)), "half": half,
            "ee_kernel": float(model.sum() / total)}
    return (model / model.sum()).astype(np.float32), info


# ----------------------------------------------------------------- the exposure set
def _raw_cache_budget() -> float:
    """Bytes of RAM the raw-frame cache of ExposureSet may use."""
    env = os.environ.get("ASTROPHOTO_RAW_CACHE_GB")
    if env:
        return float(env) * 2 ** 30
    try:
        return 0.35 * os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 4 * 2 ** 30


_PREP_SET = None
_PREP_COUNTER = None


def _prep_worker_init(state: bytes, counter=None):
    """Worker process of ExposureSet.prepare: the set (reference, catalogue with its PSF-star
    tables, masks) once.  One thread per process: the processes already use every core, and
    OpenCV's own thread pool in each of them would oversubscribe the CPU."""
    import pickle
    global _PREP_SET, _PREP_COUNTER
    _PREP_COUNTER = counter
    cv2.setNumThreads(1)
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    _PREP_SET = pickle.loads(state)


def _prep_worker_run(k0: int, k1: int):
    def tick():
        if _PREP_COUNTER is not None:
            with _PREP_COUNTER.get_lock():
                _PREP_COUNTER.value += 1
    return _PREP_SET.prepare_run(k0, k1, tick=tick)


class ExposureSet:
    """The prepared exposures of one session (see module docstring).

    ``prepare()`` makes one pass over the subs (registration refinement, photometric
    scale, background, PSFs, photon-transfer statistics); ``window()`` then yields
    y(t), v(t), m(t) of every exposure for any window of the reference grid."""

    def __init__(self, infos, analysis, defects, ref: np.ndarray, sat: float, workers: int | None = None):
        frames = analysis["frames"]
        self.items = [(info, fr) for info, fr in zip(infos, frames) if fr["accepted"] and fr["weight"] > 0]
        self.info0 = infos[0]
        self.W0, self.H0 = infos[0].width, infos[0].height
        self.pattern = infos[0].bayer
        self.defects = defects if defects is not None else np.zeros((self.H0, self.W0), bool)
        self.sat = float(sat)                     # saturation in bias-subtracted ADU
        self.ref = ref.astype(np.float32)         # reference coadd on the reference (1x) grid
        # threads for window extraction (I/O, OpenCV and NumPy release the interpreter lock)
        from .resources import workers_for
        frame = 4.0 * self.W0 * self.H0
        self.workers = workers or workers_for(40 * frame, "ASTROPHOTO_PREP_RAM_GB")
        self._worker_bytes = 30 * frame          # per preparation process: one sub's working arrays (~24 frames) + the shared state (~7)
        fw = np.array([fr["fwhm"] for _, fr in self.items], float)
        self.fwhm_max = float(np.nanmax(fw))
        self.fwhm_med = float(np.nanmedian(fw))
        self.params: list[dict] = []
        self.ptc: dict | None = None
        self.nodes = psf_nodes(self.H0, self.W0)      # field PSF grid (empirical_psf_field)

    # ------------------------------------------------------------- per frame
    def _raw(self, info):
        """Bias-subtracted raw frame with defects repaired (same-colour neighbour median),
        its saturated pixels, and the repaired ones.

        A restoration reads every sub once per cutout, and reading a sub (~0.4 s, most of it
        FITS I/O from the image folder) costs 40x more than demosaicing and warping the cutout:
        IC 405 spent over 2 h of a restoration reading.  So the frames are kept in memory
        (float32: frames without a black level in their header, as the ASI585MC's, reach 65 520,
        beyond float16's 65 504) within a budget of 35 % of the RAM (ASTROPHOTO_RAW_CACHE_GB
        overrides it).  Frames beyond the budget are read every time
        (no eviction: the restoration visits the subs in the same order for every cutout, so an
        LRU cache smaller than the session would never hit)."""
        cache = self.__dict__.setdefault("_raw_cache", {})
        hit = cache.get(info.path)
        if hit is not None:
            return hit[0].copy(), hit[1], self.defects
        raw = read_raw(info.path, info.bias)
        sat = raw >= 0.95 * self.sat
        raw = fix_defects(raw, self.defects)
        need = raw.size * 5                                  # float32 frame + saturation mask
        if self._raw_cache_used() + need <= _raw_cache_budget():
            cache[info.path] = (raw.astype(np.float32, copy=True), sat)
        return raw, sat, self.defects

    def _raw_cache_used(self) -> int:
        return sum(r.nbytes + m.nbytes for r, m in self.__dict__.get("_raw_cache", {}).values())

    def __getstate__(self):
        # worker processes and pickles never carry the frame cache
        return {k: v for k, v in self.__dict__.items() if k != "_raw_cache"}

    def _obstruction(self, fr, y0, y1, x0, x1):
        tm = fr.get("tile_mask")
        if tm is None or (tm >= 0.5).all():
            return None
        full = cv2.resize(tm.astype(np.float32), (self.W0, self.H0), interpolation=cv2.INTER_LINEAR)
        return full[y0:y1, x0:x1] >= 0.5

    def register(self, k: int, refine=None):
        info, fr = self.items[k]
        raw, sat, rep = self._raw(info)
        mx, my = window_maps(fr, self.W0, self.H0, 0, self.H0, 0, self.W0, refine)
        y, valid, hard = warp_window(raw, sat, rep, self.pattern, mx, my)
        obs = self._obstruction(fr, 0, self.H0, 0, self.W0)
        if obs is not None:
            valid &= obs
            hard &= obs
        return y, valid, hard

    def prepare_one(self, k: int) -> dict:
        from .stacking import eval_surface, fit_smooth_surface
        info, fr = self.items[k]
        fwhm = float(fr["fwhm"]) if np.isfinite(fr["fwhm"]) else self.fwhm_max
        refine = None
        y, valid, hard = self.register(k)
        for _ in range(3):                                   # re-measure after each correction
            d = refine_registration(y.mean(-1), hard, self.cat, fwhm, self.W0, self.H0)
            if d is None:
                break
            refine = compose_refine(refine, d, self.W0, self.H0)
            y, valid, hard = self.register(k, refine)
            if d["rms_before"] < 1.5 * d["noise_rms"]:      # the offsets are already at the noise level
                break
        final = refine_registration(y.mean(-1), hard, self.cat, fwhm, self.W0, self.H0)
        T, T_err = aperture_ratio(y, self.ref, hard, self.cat, radius=3.0 * max(fwhm, self.fwhm_ref))
        if not np.all(np.isfinite(T)):
            # no photometric scale in some channel (too few isolated stars at SNR >= 50): the
            # exposure cannot be placed in the model; usable() leaves it out
            return {"refine": refine, "residual": final, "T": T, "T_err": T_err, "surf": None, "psf": [None] * 3,
                    "psf_field": [None] * 3, "psf_stars": [0, 0, 0], "fwhm": fwhm, "valid_frac": float(valid.mean()),
                    "_y": None}
        e = y / T[None, None, :] - self.ref
        coefs = fit_smooth_surface(e, valid & ~self.smask, deg=2)
        surf = eval_surface(coefs, self.H0, self.W0)
        ybs = y / T[None, None, :] - surf - self.sky_ref
        half = int(math.ceil(3.5 * max(fwhm, self.fwhm_ref)))
        fe = max(fwhm, self.fwhm_ref)
        fields = [empirical_psf_field(ybs[..., c], hard, self.cat, half, fe, self.nodes) for c in range(3)]
        fields = [hybrid_psf(f, r, fe) if f is not None and r is not None else f for f, r in zip(fields, self.psf_ref)]
        return {"refine": refine, "residual": final, "T": T, "T_err": T_err, "surf": coefs,
                "psf": [f["psf"] if f else None for f in fields], "psf_field": fields,
                "psf_stars": [f["n_stars"] if f else 0 for f in fields], "fwhm": fwhm,
                "valid_frac": float(valid.mean()), "_y": y, "_valid": valid, "_surf": surf}

    # ------------------------------------------------------------- the pass
    def prepare(self, progress=None, cancel=None):
        from .postprocess import background_model
        say = (lambda msg: progress(0, len(self.items), msg)) if progress else (lambda msg: None)
        say("ImageMM: reference sky model and star catalogue")
        self.sky_ref, self.sky_info = background_model(self.ref, "poly", 2)
        ref_bs = self.ref - self.sky_ref
        self.fwhm_ref = self.fwhm_med
        self.cat = star_catalog(ref_bs, self.sat, self.fwhm_ref)
        self.smask = star_mask(self.ref.shape[:2], self.cat, self.fwhm_ref)
        # the coadd's own field PSF: the high-SNR wings of every exposure's PSF (hybrid_psf)
        say("ImageMM: PSF of the reference coadd")
        half_ref = int(math.ceil(3.5 * self.fwhm_max))
        self.psf_ref = [empirical_psf_field(ref_bs[..., c], np.ones(ref_bs.shape[:2], bool), self.cat, half_ref,
                                            self.fwhm_ref, self.nodes) for c in range(3)]
        n = len(self.items)
        self.params = [None] * n
        acc = {"X": [], "v": [], "n": []}        # photon-transfer bins
        # Worker processes, each preparing a run of consecutive subs and pairing neighbours itself
        # (photon transfer), so only the small per-sub parameters and the pair bins travel back:
        # shipping every prepared frame (~200 MB) to one process for the pairing left the workers
        # blocked (measured on M 27: 20-60 % CPU each).  Processes, not threads: much of prepare_one
        # is Python holding the interpreter lock (1 thread 4.2 s per sub, 4 threads 3.5 s, 8 threads
        # 3.8 s).  Each run also prepares the sub before it, so the pairs are exactly those of a
        # serial pass, merged in sub order.
        from concurrent.futures import ProcessPoolExecutor
        from .resources import workers_for
        nproc = min(n, workers_for(self._worker_bytes, "ASTROPHOTO_PREP_RAM_GB"))
        run = int(np.clip(n // max(3 * nproc, 1), 4, 16)) if nproc > 1 else n
        chunks = [(k0, min(n, k0 + run)) for k0 in range(0, n, run)]
        if nproc > 1:
            import pickle
            keys = set()
            for _, fr in self.items:          # the PSF-star tables, built once here (as prepare_one keys them)
                fe = max(float(fr["fwhm"]) if np.isfinite(fr["fwhm"]) else self.fwhm_max, self.fwhm_ref)
                keys.add((math.ceil(fe * 10 - 1e-9) / 10, int(math.ceil(3.5 * fe))))
            for i, (fq, half) in enumerate(sorted(keys)):
                say(f"ImageMM: PSF-star table {i + 1}/{len(keys)} (FWHM {fq:.1f} px)")
                _psf_star_table(self.cat, fq, half)
            say(f"ImageMM: starting {nproc} worker processes")
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            counter = ctx.Value("i", 0)            # subs prepared so far, over all workers
            ex = ProcessPoolExecutor(max_workers=nproc, mp_context=ctx, initializer=_prep_worker_init,
                                     initargs=(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL), counter))
            submit = lambda c: ex.submit(_prep_worker_run, *c)
        else:
            import threading as _th
            counter = type("C", (), {"value": 0, "get_lock": lambda self_: _th.Lock()})()
            ex = ThreadPoolExecutor(max_workers=1)
            submit = lambda c: ex.submit(self.prepare_run, *c, tick=lambda: setattr(counter, "value", counter.value + 1))
        with ex:
            futs = [submit(c) for c in chunks[:2 * max(nproc, 1)]]
            nxt = len(futs)
            for ci, (k0, k1) in enumerate(chunks):
                if cancel and cancel():
                    for f in futs[ci:]:
                        if f is not None:
                            f.cancel()
                    raise RuntimeError("cancelled")
                # report every sub as the workers finish it (runs return whole), and honour a pause
                # or cancel while waiting
                while True:
                    try:
                        params, own = futs[ci].result(timeout=1.0)
                        break
                    except TimeoutError:
                        if cancel and cancel():
                            for f in futs[ci:]:
                                if f is not None:
                                    f.cancel()
                            raise RuntimeError("cancelled")
                        if progress:
                            done = min(counter.value, n)
                            progress(done, n, f"ImageMM: preparing exposures {done}/{n}")
                futs[ci] = None
                if nxt < len(chunks):
                    futs.append(submit(chunks[nxt]))
                    nxt += 1
                self.params[k0:k1] = params
                for kk in acc:
                    acc[kk].extend(own[kk])
                if progress:
                    progress(k1, n, f"ImageMM: preparing exposures {k1}/{n}")
        self.ptc = self._ptc_fit(acc)
        return self

    def prepare_run(self, k0: int, k1: int, tick=None):
        """prepare_one for subs k0 .. k1-1 and the photon-transfer bins of each consecutive usable
        pair (k-1, k) with k in [k0, k1), as a serial pass forms them.  Returns (their parameters,
        the bins)."""
        acc = {"X": [], "v": [], "n": []}
        prev = None
        if k0 > 0:
            q = self.prepare_one(k0 - 1)
            if q["_y"] is not None:
                prev = {kk: q[kk] for kk in ("_y", "_valid", "_surf", "T")}
        params = []
        for k in range(k0, k1):
            p = self.prepare_one(k)
            if tick:
                tick()
            if p["_y"] is None:                  # not usable: no photometric scale
                prev = None
            else:
                if prev is not None:
                    self._ptc_pair(prev, p, acc)
                prev = {kk: p[kk] for kk in ("_y", "_valid", "_surf", "T")}
            params.append({kk: v for kk, v in p.items() if not kk.startswith("_")})
        return params, acc

    # ------------------------------------------------------------- photon transfer
    def _ptc_pair(self, a: dict, b: dict, acc: dict, nbins: int = 32):
        """Variance of the difference of two consecutive exposures (both scaled to the
        reference and background-matched) in star-free pixels, binned by level.

        var(y/T) = (c0 + c1 L)/T^2 with L = T * l the expected exposure level (ADU), l the
        reference level plus the exposure's sky deviation, so
        var(diff) = c0 (1/Ta^2 + 1/Tb^2) + c1 (la/Ta + lb/Tb)."""
        ok = a["_valid"] & b["_valid"] & ~self.smask
        for c in range(3):
            Ta, Tb = a["T"][c], b["T"][c]
            la = (self.ref[..., c] + a["_surf"][..., c])[ok]
            lb = (self.ref[..., c] + b["_surf"][..., c])[ok]
            dv = ((a["_y"][..., c][ok] / Ta - a["_surf"][..., c][ok]) - (b["_y"][..., c][ok] / Tb - b["_surf"][..., c][ok]))
            lv = 0.5 * (la + lb)
            edges = np.unique(np.quantile(lv, np.linspace(0, 1, nbins + 1)))
            # bin j holds edges[j] <= level < edges[j + 1] (the last bin also its upper edge): with the
            # pixels sorted by level once, every bin is a contiguous slice
            order = np.argsort(lv, kind="stable")
            ls = lv[order]
            bounds = np.searchsorted(ls, edges, side="left")
            bounds[-1] = len(ls)
            for j in range(len(edges) - 1):
                sl = order[bounds[j]:bounds[j + 1]]
                if len(sl) < 500:
                    continue
                dd = dv[sl]
                var = (1.4826 * np.median(np.abs(dd - np.median(dd)))) ** 2
                l_a, l_b = np.median(la[sl]), np.median(lb[sl])
                acc["X"].append((c, 1 / Ta ** 2 + 1 / Tb ** 2, l_a / Ta + l_b / Tb))
                acc["v"].append(var)
                acc["n"].append(int(len(sl)))

    @staticmethod
    def _ptc_fit(acc: dict) -> dict:
        from scipy.optimize import nnls
        if not acc["X"]:
            raise RuntimeError("photon-transfer calibration impossible: no two consecutive usable exposures "
                               "(each needs a photometric scale in every channel, from isolated stars with a combined "
                               "SNR >= 50) with enough star-free pixels")
        X = np.array(acc["X"], float)
        v = np.array(acc["v"], float)
        n = np.array(acc["n"], float)
        c0, c1, red = np.zeros(3), np.zeros(3), np.zeros(3)
        for c in range(3):
            s = X[:, 0] == c
            if s.sum() < 2:
                raise RuntimeError(f"photon-transfer calibration impossible in channel {c}: too few level bins")
            A = X[s, 1:]
            y = v[s]
            w = np.sqrt(n[s] / 2) / np.maximum(y, 1e-12)        # sd of a variance estimate ~ var sqrt(2/n)
            coef = nnls(A * w[:, None], y * w)[0]              # c0 (read noise^2), c1 (1/gain) >= 0
            for _ in range(3):                                 # reweight with the model variance
                mdl = A @ coef
                w = np.sqrt(n[s] / 2) / np.maximum(mdl, 1e-12)
                coef = nnls(A * w[:, None], y * w)[0]
            c0[c], c1[c] = coef
            mdl = A @ coef
            red[c] = float(np.mean(((y - mdl) * w) ** 2))
        return {"c0": c0, "c1": c1, "chi2_red": red, "bins": int(len(v))}

    # ------------------------------------------------------------- persistence
    _STATE = ("params", "ptc", "sky_ref", "sky_info", "fwhm_ref", "cat", "smask", "nodes", "psf_ref")
    # bumped whenever the preparation changes what it produces; an older cache is prepared again
    # (2: crowded-field PSF stars - with 1, most subs of a Milky Way field had no PSF;
    #  3: PSF cut-outs must agree with each other - with 2, bright non-point sources set the wings)
    #  4: field-dependent PSFs (empirical_psf_field) with the coadd's wings (hybrid_psf), least-squares
    #     photometric scales)
    #  7: per-exposure PSFs = the coadd's field PSF scaled to the exposure's seeing (hybrid_psf)
    PREP_VERSION = 7

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            state = {k: getattr(self, k) for k in self._STATE}
            state["cat"] = {k: v for k, v in state["cat"].items() if not k.startswith("_")}   # rebuilt on demand
            pickle.dump({"names": [info.name for info, _ in self.items], "prep_version": self.PREP_VERSION,
                         **state}, f)

    def load(self, path: str) -> "ExposureSet":
        import pickle
        with open(path, "rb") as f:
            st = pickle.load(f)
        if st["names"] != [info.name for info, _ in self.items]:
            raise RuntimeError("the prepared exposures belong to a different frame selection")
        if st.get("prep_version", 1) != self.PREP_VERSION:
            raise RuntimeError("the prepared exposures were made by an older version of the preparation")
        for k in self._STATE:
            setattr(self, k, st[k])
        return self

    def usable(self) -> list[int]:
        """Exposures with a PSF in every channel (the model needs f(t))."""
        return [k for k, p in enumerate(self.params) if p is not None and all(q is not None for q in p["psf_field"])
                and np.all(np.isfinite(p["T"]))]

    def moffat_psfs(self):
        """Fit the Moffat model (``fit_moffat``) to every exposure's empirical PSF model at every
        node of the field grid."""
        from concurrent.futures import ThreadPoolExecutor as _TP
        todo = [p for p in self.params if p is not None and "psf_moffat_nodes" not in p
                and all(f is not None for f in p["psf_field"])]

        def fit(p):
            ks, infos = [], []
            for f in p["psf_field"]:
                ny, nx, n, _ = f["mean"].shape
                fits_ = [fit_moffat(f["mean"][i, j], f["se"]) for i in range(ny) for j in range(nx)]
                size = max(q[0].shape[0] for q in fits_)
                K = np.zeros((ny * nx, size, size), np.float32)
                for q, (mdl, _) in enumerate(fits_):
                    o = (size - mdl.shape[0]) // 2
                    K[q, o:o + mdl.shape[0], o:o + mdl.shape[0]] = mdl
                ks.append(K.reshape(ny, nx, size, size))
                infos.append([inf for _, inf in fits_])
            return ks, infos
        with _TP(max_workers=max(1, os.cpu_count() or 1)) as ex:
            for p, (ks, infos) in zip(todo, ex.map(fit, todo)):
                p["psf_moffat_nodes"], p["moffat"] = ks, infos

    def node_kernels(self, idx: list[int], model: str = "empirical") -> np.ndarray:
        """(n, 3, ny, nx, ks, ks) PSFs of the chosen exposures at the nodes of the field grid
        ``self.nodes``, zero-padded to a common odd size (zero padding does not change a
        convolution).  model: "empirical" (measured, the paper's input) or "moffat" (the fitted
        models, see ``moffat_psfs``)."""
        if model == "moffat":
            self.moffat_psfs()
            get = lambda p, c: p["psf_moffat_nodes"][c]
        else:
            get = lambda p, c: p["psf_field"][c]["nodes"]
        ks = max(get(self.params[k], c).shape[-1] for k in idx for c in range(3))
        ny, nx = len(self.nodes[0]), len(self.nodes[1])
        out = np.zeros((len(idx), 3, ny, nx, ks, ks), np.float32)
        for a, k in enumerate(idx):
            for c in range(3):
                q = get(self.params[k], c)
                o = (ks - q.shape[-1]) // 2
                out[a, c, :, :, o:o + q.shape[-1], o:o + q.shape[-1]] = q
        return out

    def kernels(self, idx: list[int], model: str = "empirical", at: tuple[float, float] | None = None) -> np.ndarray:
        """(n, 3, ks, ks) PSFs of the chosen exposures at reference position ``at`` = (row, column)
        (bilinear between the field-grid nodes), or - ``at`` None - averaged over the field."""
        K = self.node_kernels(idx, model)
        if at is None:
            return K.mean((2, 3))
        return interp_nodes(K, self.nodes, *at).astype(np.float32)

    # ------------------------------------------------------------- data for a window
    def group_coadds(self, idx: list[int], n_groups: int, model: str = "empirical", progress=None,
                     cancel=None) -> dict:
        """Full-field seeing-group coadds of the exposures ``idx`` (see imagemm.coadd_groups):
        exposures sorted by PSF width into ``n_groups`` equal-count groups, each reduced to
            y_g = sum w y / sum w,  v_g = 1 / sum w,  m_g = [sum w > 0],  w = m / v,
        with PSF f_g = sum_t W_t f_t / sum_t W_t (W_t: field-mean weight per channel).
        One pass over the exposures.  Returns {"y", "v", "m": (G, H, W, 3), "kernels":
        (G, 3, ks, ks), "groups": member lists, "fwhm": per group}."""
        from .imagemm import _fwhm, seeing_groups
        K = self.kernels(idx, model)
        groups = seeing_groups(K, n_groups)
        G = len(groups)
        of = {int(t): g for g, members in enumerate(groups) for t in members}
        H, W = self.H0, self.W0
        S = np.zeros((G, H, W, 3), np.float32)
        Wsum = np.zeros((G, H, W, 3), np.float32)
        Wbar = np.zeros((len(idx), 3))
        with ThreadPoolExecutor(max_workers=max(1, self.workers // 2)) as ex:
            futs = {ex.submit(self.window, k, 0, H, 0, W): a for a, k in enumerate(idx[:self.workers])}
            nxt = min(len(idx), self.workers)
            done = 0
            while futs:
                fut = next(iter(futs))
                a = futs.pop(fut)
                y, v, m = fut.result()
                if nxt < len(idx):
                    futs[ex.submit(self.window, idx[nxt], 0, H, 0, W)] = nxt
                    nxt += 1
                w = np.where(m > 0, 1 / np.maximum(v, 1e-30), 0).astype(np.float32)
                g = of[a]
                S[g] += w * y
                Wsum[g] += w
                Wbar[a] = w.reshape(-1, 3).mean(0)
                done += 1
                if cancel and cancel():
                    raise RuntimeError("cancelled")
                if progress:
                    progress(done, len(idx), f"Seeing-group coadds {done}/{len(idx)}")
        Kg = np.stack([(K[g] * Wbar[g][:, :, None, None]).sum(0) / np.maximum(Wbar[g].sum(0), 1e-30)[:, None, None]
                       for g in groups]).astype(np.float32)
        ok = Wsum > 0
        Y = np.where(ok, S / np.maximum(Wsum, 1e-30), 0).astype(np.float32)
        V = np.where(ok, 1 / np.maximum(Wsum, 1e-30), 1).astype(np.float32)
        return {"y": Y, "v": V, "m": ok.astype(np.float32), "kernels": Kg,
                "groups": [[idx[int(t)] for t in g] for g in groups],
                "fwhm": [float(_fwhm(k.mean(0))) for k in Kg]}

    def windows(self, idx: list[int], y0: int, y1: int, x0: int, x1: int):
        """Stacked y, v, m of the chosen exposures on a window: (n, 3, h, w) float32 each."""
        n, h, w = len(idx), y1 - y0, x1 - x0
        Y = np.empty((n, 3, h, w), np.float32)
        V = np.empty_like(Y)
        Mk = np.empty_like(Y)
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            for a, (yy, vv, mm) in enumerate(ex.map(lambda k: self.window(k, y0, y1, x0, x1), idx)):
                Y[a], V[a], Mk[a] = (np.moveaxis(z, -1, 0) for z in (yy, vv, mm))
        return Y, V, Mk

    def window(self, k: int, y0: int, y1: int, x0: int, x1: int):
        """y(t), v(t), m(t) of exposure k on the reference-grid window (float32, HxWx3)."""
        from .stacking import eval_surface
        info, fr = self.items[k]
        p = self.params[k]
        raw, sat, rep = self._raw(info)
        mx, my = window_maps(fr, self.W0, self.H0, y0, y1, x0, x1, p["refine"])
        y, valid, _ = warp_window(raw, sat, rep, self.pattern, mx, my)
        h, w = y1 - y0, x1 - x0
        if y is None:
            z = np.zeros((h, w, 3), np.float32)
            return z, np.ones_like(z), z
        obs = self._obstruction(fr, y0, y1, x0, x1)
        if obs is not None:
            valid &= obs
        surf = eval_surface(p["surf"], self.H0, self.W0)[y0:y1, x0:x1]
        T = p["T"][None, None, :]
        ybs = y / T - surf - self.sky_ref[y0:y1, x0:x1]
        level = np.maximum(T * (self.ref[y0:y1, x0:x1] + surf), 0)            # expected exposure level, ADU
        var = (self.ptc["c0"][None, None, :] + self.ptc["c1"][None, None, :] * level) / T ** 2
        m = np.repeat(valid[..., None], 3, -1).astype(np.float32)
        return ybs.astype(np.float32), var.astype(np.float32), m
