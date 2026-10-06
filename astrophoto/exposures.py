"""Exposures for ImageMM: the data products of Sec. 2 of arXiv:2501.03002.

ImageMM needs, for every exposure t: the coregistered, background-subtracted
image y(t), per-pixel variances v(t), a binary mask m(t) and the PSF f(t)
measured from the exposure's stars.  This module derives them from the raw
(calibrated) subs.

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
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import cv2
import numpy as np
import sep

from .analysis import poly_eval, poly_terms
from .frames import cfa_masks, fix_defects, read_frame


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
    me = np.where(d[:, 0] <= max(1.0, fwhm), j[:, 0], -1)  # the source's own 3-sigma detection (see _psf_star_table)
    nflux = np.array([max([allobj["flux"][q] for q in lst if q != s], default=0.0) for lst, s in zip(near, me)])
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
    key = (len(cat["x"]), round(float(fwhm), 3), int(half), "self-fwhm")   # (tables cached before the self-match fix are not reused)
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
        tree = self_tree = cKDTree(np.stack([ax, ay], 1))
        reach = pad * np.sqrt(2) + rn.max()
    else:                                                # no neighbour list: nearest-neighbour distances only
        tree = None
    table = []
    for i in np.argsort(-f):
        ix, iy = int(round(x[i])), int(round(y[i]))
        m = np.zeros(yy.shape, bool)
        if tree is not None:
            # the star's own 3-sigma detection: the nearest one within a FWHM.  Its isophotal barycentre is not
            # the windowed centroid (cat x, y): on a bright DWARF 3 star the halo, off-centre by 1-2 px, pulls it
            # 1.0-1.1 px away, and a fixed 1 px match made every bright star its own bright neighbour - the
            # PSF was then measured from faint stars only (71 of 71 bright unsaturated stars left out on M 42)
            d0, j0 = self_tree.query([x[i], y[i]])
            me = j0 if d0 <= max(1.0, fwhm) else -1
            nb = [j for j in tree.query_ball_point([x[i], y[i]], reach) if j != me]
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


def _finish_psf(mu: np.ndarray, rsup: float, fwhm: float | None = None, harmonics: int = 4) -> np.ndarray:
    """Cut at the support radius, negative pixels to 0, unit sum.  With ``fwhm``, the wings beyond
    2 FWHM are first replaced by their low-order angular expansion: in each 1-px ring the least-squares
    fit of a_0 + sum_{m<=harmonics} (a_m cos m theta + b_m sin m theta), the coefficients interpolated
    linearly in r at each pixel.  Clipped pixel by pixel, the zero-mean noise of the far wings becomes
    a positive pedestal (on M 42, DWARF 3, the kernel held 2.5-3x the measured wing at 12-18 px and
    ImageMM took that much light out of the sky round every bright star); a ring's few coefficients
    average tens to hundreds of pixels, so they are near noise-free and are clipped at 0 without that
    bias.  The expansion keeps the wings' real asymmetry - the DWARF 3's halo is centred 1-2 px off the
    core: an azimuthal average (m = 0 alone) erased it, and the restoration put the missing halo light
    into the latent as an arc beside each bright star, with a dark gap opposite."""
    n = mu.shape[-1]
    yy, xx = np.mgrid[:n, :n] - n // 2
    rr = np.hypot(yy, xx)
    th = np.arctan2(yy, xx)
    mu = np.asarray(mu, np.float64)
    if fwhm:
        flat = mu.reshape(-1, n * n)
        ri = np.minimum(rr.astype(int), n).ravel()
        cols = [np.ones(n * n)] + [f(m * th.ravel()) for m in range(1, harmonics + 1) for f in (np.cos, np.sin)]
        Bm = np.stack(cols, 1)                                       # (n^2, 1 + 2 harmonics)
        nb = int(ri.max()) + 1
        coef = np.full((nb, Bm.shape[1], len(flat)), np.nan)
        rmean = np.full(nb, np.nan)
        for k in range(nb):
            q = ri == k
            if not q.any():
                continue
            rmean[k] = rr.ravel()[q].mean()
            mm = Bm.shape[1] if q.sum() >= 4 * Bm.shape[1] else 1       # small rings: the mean only
            coef[k, :mm] = np.linalg.lstsq(Bm[q, :mm], flat[:, q].T, rcond=None)[0]
            coef[k, mm:] = 0.0
        ok = np.isfinite(rmean)
        r_ = rr.ravel()
        C = np.stack([np.stack([np.interp(r_, rmean[ok], coef[ok, j, i]) for j in range(Bm.shape[1])], 1)
                      for i in range(len(flat))])                     # (kernels, n^2, terms)
        wing = (C * Bm[None]).sum(-1).reshape(mu.shape)
        mu = np.where(rr > 2 * float(fwhm), np.maximum(wing, 0.0), mu)
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
    psf = _finish_psf(mu, rsup, fwhm)
    if return_error:
        n = mu.shape[0]
        rr = np.hypot(*np.mgrid[:n, :n] - n // 2)
        tot = max(float(np.clip(np.where(rr <= rsup, mu, 0), 0, None).sum()), 1e-300)
        se_out = np.where(np.isfinite(se), se, 1e30 * tot)
        return psf, len(S), {"mean": (mu / tot).astype(np.float32), "se": (se_out / tot).astype(np.float32),
                             "support": rsup}
    return psf, len(S)


PSF_FIELD_DEG = 3            # degree of the field PSF polynomial (empirical_psf_field)
PSF_NODE_SPACING = 450.0     # px between the field PSF nodes.  M 42's cluster (DWARF 3, lower part of the
                             # frame): degree 2 on 900 px nodes missed the stars' coma by 2.7 % of the core
                             # flux (light above the core against below), and the restoration grew a small
                             # crescent above every star; degree 3 on 450 px: 1.5 %


def psf_nodes(H: int, W: int, spacing: float = PSF_NODE_SPACING) -> tuple[np.ndarray, np.ndarray]:
    """Node positions (reference-grid rows, columns) of the field PSF grid: evenly spaced from
    edge to edge, at most ``spacing`` pixels apart, at least 2 per axis."""
    ny = max(2, int(math.ceil((H - 1) / spacing)) + 1)
    nx = max(2, int(math.ceil((W - 1) / spacing)) + 1)
    return np.linspace(0, H - 1, ny), np.linspace(0, W - 1, nx)


def empirical_psf_field(img: np.ndarray, valid: np.ndarray, cat: dict, half: int, fwhm: float,
                        nodes: tuple[np.ndarray, np.ndarray], deg: int = PSF_FIELD_DEG, clip: float = 3.0,
                        min_stars: int = 10, nsig: float = 2.0, max_stars: int = 1000, cutouts=None) -> dict | None:
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
    S, w, noises, px, py = cutouts if cutouts is not None else _psf_cutouts(img, valid, cat, half, fwhm, max_stars)
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
        Ainv = np.linalg.inv(A)                            # (n^2, P, P)
        coef = np.einsum("jpq,jq->jp", Ainv, rhs)          # (n^2, P)
        model = B @ coef.T
        # leave-one-out residuals, res / (1 - leverage): with the bright stars admitted a pixel is
        # fitted by few effective stars and nearly interpolated, and its plain residual says
        # nothing about its scatter (the clipping threshold collapsed round the brightest stars)
        lev = ww * np.einsum("ip,jpq,iq->ij", B, Ainv, B)
        res = (Sf - model) / np.clip(1.0 - lev, 0.05, None)
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
    rr = np.hypot(*np.mgrid[:n, :n] - n // 2)
    ny_, nx_ = len(nodes[0]), len(nodes[1])
    ny, nx = np.meshgrid(nodes[0], nodes[1], indexing="ij")
    Bn = poly_terms(*uv(nx.ravel(), ny.ravel()), deg)                 # (ny*nx, P)
    mu_n = np.tensordot(Bn, coef, 1)                                    # (ny*nx, n, n)
    # the wings beyond 2 FWHM: _finish_psf's low-order angular expansion of this model (the
    # azimuthally averaged power-law tail of the median ring means, _robust_wings, made up for a PSF
    # measured on faint stars alone - see _psf_star_table - and erased the halo's real asymmetry)
    rsup = _support_radius(mu_bar, se, half, nsig)
    inside = rr <= rsup
    tot = np.maximum(np.clip(np.where(inside, mu_n, 0), 0, None).sum((1, 2)), 1e-300)
    tot_bar = max(float(np.clip(np.where(inside, mu_bar, 0), 0, None).sum()), 1e-300)
    return {"nodes": _finish_psf(mu_n, rsup, fwhm).reshape(ny_, nx_, n, n),
            "mean": (mu_n / tot[:, None, None]).astype(np.float32).reshape(ny_, nx_, n, n),
            "se": np.where(np.isfinite(se), se / tot_bar, 1e30).astype(np.float32),
            "support": rsup, "deg": deg, "n_stars": int(N), "psf": _finish_psf(mu_bar, rsup, fwhm),
            "wings_averaged": True, "wings": "harmonic", "field": [deg, len(nodes[0]), len(nodes[1])]}


def _fit_size(a: np.ndarray, n: int, fill: float = 1e30) -> np.ndarray:
    """A centred (m, m) array cropped or padded (with ``fill``) to (n, n)."""
    m = a.shape[-1]
    if m >= n:
        o = (m - n) // 2
        return a[o:o + n, o:o + n]
    o = (n - m) // 2
    return np.pad(a, o, constant_values=fill)


def warp_psf(K: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Kernels K (..., N, N) under the linear map A (2 x 2, (y, x) order) about their centre:
    K_A(p) = K(A^-1 p) / |det A|, clipped at 0 and renormalised to unit sum (same N)."""
    N = K.shape[-1]
    c = (N - 1) / 2
    Ai = np.linalg.inv(np.asarray(A, np.float64))
    # cv2 works in (x, y): dst (x, y) <- src = Ai (p - c) + c
    Mxy = np.array([[Ai[1, 1], Ai[1, 0]], [Ai[0, 1], Ai[0, 0]]])
    off = np.array([c, c]) - Mxy @ np.array([c, c])
    M_ = np.float32(np.c_[Mxy, off])
    flat = K.reshape(-1, N, N)
    out = np.empty_like(flat, dtype=np.float32)
    for i, k in enumerate(flat):
        k = np.clip(cv2.warpAffine(k.astype(np.float32), M_, (N, N), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP), 0, None)
        out[i] = k / max(float(k.sum()), 1e-30)
    return out.reshape(K.shape)


def _weighted_moments(K: np.ndarray, fwhm: float, valid: np.ndarray | None = None) -> np.ndarray:
    """Second-moment matrix (y, x) of a centred kernel with a Gaussian weight of sigma = FWHM (adaptive
    moments: the weight keeps the noise of the far wings out)."""
    N = K.shape[-1]
    Y, X = np.mgrid[:N, :N] - (N - 1) / 2
    w = np.exp(-(Y ** 2 + X ** 2) / (2 * fwhm ** 2))
    if valid is not None:
        w = w * valid
    k = K * w
    t = max(float(k.sum()), 1e-30)
    my, mx = (k * Y).sum() / t, (k * X).sum() / t
    return np.array([[(k * (Y - my) ** 2).sum(), (k * (Y - my) * (X - mx)).sum()],
                     [(k * (Y - my) * (X - mx)).sum(), (k * (X - mx) ** 2).sum()]]) / t


def _sqrtm(M: np.ndarray) -> np.ndarray:
    v, U = np.linalg.eigh(M)
    return (U * np.sqrt(np.maximum(v, 1e-12))) @ U.T


def scale_psf(K: np.ndarray, sc: float) -> np.ndarray:
    """Kernels K (..., N, N) radially scaled by ``sc`` about their centre: K_s(x) = K(x / s) / s^2,
    clipped at 0 and renormalised to unit sum (same N; pad K first to leave room)."""
    N = K.shape[-1]
    c = N // 2
    M_ = np.float32([[1 / sc, 0, c - c / sc], [0, 1 / sc, c - c / sc]])          # dst x <- src c + (x - c) / s
    flat = K.reshape(-1, N, N)
    out = np.empty_like(flat, dtype=np.float32)
    for i, k in enumerate(flat):
        k = np.clip(cv2.warpAffine(k.astype(np.float32), M_, (N, N), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP), 0, None)
        out[i] = k / max(float(k.sum()), 1e-30)
    return out.reshape(K.shape)


def fit_seeing_scale(cutouts, ref: dict, nodes, fwhm: float, scales: np.ndarray, shape: np.ndarray | None = None,
                     r_fit: float = 2.5, max_stars: int = 80) -> tuple[float, float]:
    """The seeing of one exposure and channel relative to the coadd: the radial scale s of the coadd's
    field PSF (``ref``, at each star's position; mapped by ``shape`` first) that best fits the
    exposure's own PSF-star cut-outs (``_psf_cutouts``: centred, flux-normalised, neighbours
    masked).  Per star an amplitude, a constant and a sub-pixel shift (linearised) are free; the
    scale is the flux^2-weighted median of the stars' best scales over the brightest ``max_stars``
    (one contaminated star cannot move it).  Returns (s, summed chi^2 at s).

    Why not the exposure's own field model (empirical_psf_field): a single sub's model is fitted by
    its few bright stars and nearly interpolates them, so it has no honest standard error to weight
    a fit with, and before the bright stars were admitted it was the faint stars' blurred stack -
    M 42's kernels followed the seeing measured on bright stars with a slope of 0.69."""
    S, w, noises, px, py = cutouts
    N, n, _ = S.shape
    if N == 0:
        return 1.0, np.inf
    order = np.argsort(-w.reshape(N, -1).max(1))[:max_stars]
    Kr = np.asarray(ref["nodes"], np.float32)
    if shape is not None:
        Kr = warp_psf(Kr, shape)
    nr = Kr.shape[-1]
    M = max(n, int(np.ceil(nr * scales.max())) | 1)
    pad = lambda z, m: np.pad(z, [(0, 0)] * (z.ndim - 2) + [((M - m) // 2, (M - m) // 2)] * 2)
    Kr = pad(Kr, nr)
    c = M // 2
    h = n // 2
    rr = np.hypot(*np.mgrid[:n, :n] - h)
    fit = rr <= r_fit * fwhm
    best, wt, chis = [], [], []
    for i in order:
        K0 = interp_nodes(Kr, nodes, float(py[i]), float(px[i]))              # (M, M)
        m = (w[i] > 0) & fit
        if m.sum() < 20:
            continue
        y = S[i][m]
        # the shift, linearised about the unscaled kernel: S ~ a K + a dy dK/dy + a dx dK/dx + b
        k1 = K0[c - h:c + h + 1, c - h:c + h + 1]
        gy, gx = np.gradient(k1)
        A = np.stack([k1[m], gy[m], gx[m], np.ones(m.sum())], 1)
        sol = np.linalg.lstsq(A, y, rcond=None)[0]
        dy, dx = (sol[1] / sol[0], sol[2] / sol[0]) if sol[0] > 0 else (0.0, 0.0)
        dy, dx = float(np.clip(dy, -1, 1)), float(np.clip(dx, -1, 1))
        Ks = fourier_shift(K0, dy, dx) if (abs(dy) > 1e-3 or abs(dx) > 1e-3) else K0
        chi = np.empty(len(scales))
        for j, sc in enumerate(scales):
            k = scale_psf(Ks[None], sc)[0][c - h:c + h + 1, c - h:c + h + 1]
            A = np.stack([k[m], np.ones(m.sum())], 1)
            r_ = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
            chi[j] = float((r_ ** 2).sum())
        best.append(float(scales[int(np.argmin(chi))]))
        wt.append(float(w[i].max()))
        chis.append(chi)
    if not best:
        return 1.0, np.inf
    best, wt = np.asarray(best), np.asarray(wt)
    o = np.argsort(best)
    cw = np.cumsum(wt[o])
    s_ = float(best[o][int(np.searchsorted(cw, 0.5 * cw[-1]))])
    chi_tot = float((np.asarray(chis) * wt[:, None]).sum(0)[int(np.argmin(np.abs(scales - s_)))])
    return s_, chi_tot


def hybrid_psf(sub: dict, ref: dict, fwhm: float, r_fit: float = 2.5,
               scales: np.ndarray = np.round(np.arange(0.60, 1.80001, 0.02), 2), cutouts=None, nodes=None) -> dict:
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
    if cutouts is not None and nodes is not None:
        return _hybrid_from_cutouts(sub, ref, fwhm, scales, cutouts, nodes)
    N = max(n, int(np.ceil(nr * scales.max())) | 1)
    pad = lambda z, m: np.pad(z, [(0, 0)] * (z.ndim - 2) + [((N - m) // 2, (N - m) // 2)] * 2)
    Kr, ms = pad(Kr, nr), pad(ms, n)
    se = np.pad(se, (N - n) // 2, constant_values=1e30)
    c = N // 2
    Ks = np.stack([scale_psf(Kr, sc) for sc in scales])
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
    # the exposure's own shape: its stars are elongated by wind and tracking in a direction of their
    # own (M 42, DWARF 3: a/b 1.12 - 1.30, median 1.18), which no scaled copy of the round-averaged
    # coadd PSF has - ImageMM, fitting every exposure's elongated core with a round kernel, emptied
    # the pixels round each star.  The map A = M_t^1/2 M_c^-1/2 between the second moments of the
    # exposure's mean star (field mean of its unclipped model, Gaussian-weighted: the noise averages
    # out) and of the scaled coadd PSF, its eigenvalues limited to 0.75 - 1.33; the overall size then
    # fitted again with the shape fixed
    valid = (se < 1e29).astype(np.float64)
    Mt = _weighted_moments(ms.mean((0, 1)), fwhm, valid)
    Mc = _weighted_moments(Ks[q].mean((0, 1)), fwhm, valid)
    shape = np.eye(2)
    if np.all(np.linalg.eigvalsh(Mt) > 0) and np.all(np.linalg.eigvalsh(Mc) > 0):
        A_ = _sqrtm(Mt) @ np.linalg.inv(_sqrtm(Mc))
        U, sv, Vt = np.linalg.svd(A_)
        sv = np.clip(sv / np.sqrt(sv[0] * sv[1]), 0.75, 1.33)          # shape only: unit determinant
        shape = (U * sv) @ Vt
    Kq = warp_psf(Kr, shape)
    Ks2 = np.stack([scale_psf(Kq, sc) for sc in scales])
    chi2 = (w * (m - ((w * Ks2[..., fit] * m).sum(-1) / np.maximum((w * Ks2[..., fit] ** 2).sum(-1), 1e-300))[..., None]
                 * Ks2[..., fit]) ** 2).sum(-1)
    b2 = np.argmin(chi2, 0)
    in2 = (b2 > 0) & (b2 < len(scales) - 1)
    q2 = int(np.round(np.median(b2[in2]))) if in2.any() else q
    if chi2[q2].sum() < chi[q].sum():                    # keep the shape only where it fits better
        q, Ks = q2, Ks2
    else:
        shape = np.eye(2)
    best = np.full_like(best, q)
    K = np.take_along_axis(Ks, best[None, ..., None, None], 0)[0]
    # only as large as the scaled support (the range of s left room for up to 1.8 x): the cost of
    # the restoration grows with the square of the kernel size
    h_ = min(c, int(np.ceil(float(ref["support"]) * scales[q] * max(1.0, float(np.abs(shape).max())))) + 1)
    K = K[..., c - h_:c + h_ + 1, c - h_:c + h_ + 1]
    K = K / np.maximum(K.sum((-2, -1), keepdims=True), 1e-30)
    se = se[c - h_:c + h_ + 1, c - h_:c + h_ + 1]
    out = dict(sub)
    out.update({"nodes": K.astype(np.float32), "mean": K.astype(np.float32), "psf": K.mean((0, 1)).astype(np.float32),
                "scale": scales[best].astype(np.float32), "support": float(ref["support"]),
                "shape": shape.astype(np.float32), "own_mean": ms.mean((0, 1)).astype(np.float32),
                "se": np.where(se < 1e29, se, 1e30).astype(np.float32)})
    return out


def _hybrid_from_cutouts(sub: dict, ref: dict, fwhm: float, scales: np.ndarray, cutouts, nodes) -> dict:
    """hybrid_psf with the seeing scale and the shape measured on the exposure's PSF-star cut-outs
    (fit_seeing_scale) instead of on its own field model."""
    Kr = np.asarray(ref["nodes"], np.float32)
    ny, nx, nr, _ = Kr.shape
    S, w, _, _, _ = cutouts
    N, n, _ = S.shape
    s1, chi1 = fit_seeing_scale(cutouts, ref, nodes, fwhm, scales)
    # the exposure's own shape (see hybrid_psf): second moments of its flux^2-weighted mean star
    # against the scaled coadd PSF's, eigenvalues 0.75 - 1.33, unit determinant
    wt = w.reshape(N, -1).max(1)
    Sm = np.tensordot(wt / max(wt.sum(), 1e-30), np.clip(S, 0, None), 1)
    Mt = _weighted_moments(Sm, fwhm)
    M = max(n, int(np.ceil(nr * scales.max())) | 1)
    pad = lambda z, m: np.pad(z, [(0, 0)] * (z.ndim - 2) + [((M - m) // 2, (M - m) // 2)] * 2)
    Kp = pad(Kr, nr)
    Mc = _weighted_moments(scale_psf(Kp, s1).mean((0, 1)), fwhm)
    shape = np.eye(2)
    if np.all(np.linalg.eigvalsh(Mt) > 0) and np.all(np.linalg.eigvalsh(Mc) > 0):
        A_ = _sqrtm(Mt) @ np.linalg.inv(_sqrtm(Mc))
        U, sv, Vt = np.linalg.svd(A_)
        sv = np.clip(sv / np.sqrt(sv[0] * sv[1]), 0.75, 1.33)
        shape = (U * sv) @ Vt
    s_, chi_ = s1, chi1
    if not np.allclose(shape, np.eye(2)):
        s2, chi2 = fit_seeing_scale(cutouts, ref, nodes, fwhm, scales, shape=shape)
        if chi2 < chi1:
            s_, chi_ = s2, chi2
        else:
            shape = np.eye(2)
    K = scale_psf(warp_psf(Kp, shape) if not np.allclose(shape, np.eye(2)) else Kp, s_)
    c = M // 2
    h_ = min(c, int(np.ceil(float(ref["support"]) * s_ * max(1.0, float(np.abs(shape).max())))) + 1)
    K = K[..., c - h_:c + h_ + 1, c - h_:c + h_ + 1]
    K = K / np.maximum(K.sum((-2, -1), keepdims=True), 1e-30)
    out = dict(sub)
    out.update({"nodes": K.astype(np.float32), "mean": K.astype(np.float32), "psf": K.mean((0, 1)).astype(np.float32),
                "scale": np.full((ny, nx), s_, np.float32), "support": float(ref["support"]),
                "shape": shape.astype(np.float32), "own_mean": Sm.astype(np.float32),
                "se": np.full((2 * h_ + 1, 2 * h_ + 1), 1e30, np.float32), "scale_chi": float(chi_)})
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
        from .resources import total_ram_bytes            # a container's memory limit, not the host's
        return 0.35 * total_ram_bytes()
    except (ValueError, OSError, AttributeError):
        return 4 * 2 ** 30


_PREP_SET = None
_PREP_COUNTER = None


def _compact_cutouts(cu, n_keep: int):
    """The brightest ``n_keep`` cut-outs of ``_psf_cutouts`` as (S float32, flux, mask, noises, px, py)."""
    S, w, noises, px, py = cu
    if len(S) == 0:
        return None
    fl = w.reshape(len(S), -1).max(1)
    o = np.argsort(-fl)[:n_keep]
    return (S[o].astype(np.float32), np.sqrt(fl[o]).astype(np.float32), (w[o] > 0), np.asarray(noises)[o], px[o], py[o])


def _expand_cutouts(cc):
    """Inverse of _compact_cutouts: (S, w, noises, px, py) with w = flux^2 on the unmasked pixels."""
    S, fl, mask, noises, px, py = cc
    return S.astype(np.float64), (fl.astype(np.float64) ** 2)[:, None, None] * mask, noises, px, py


def _gauss(sigma: float, n: int) -> np.ndarray:
    """Sampled, unit-sum Gaussian of width ``sigma`` on an n x n grid (n odd)."""
    yy, xx = np.mgrid[:n, :n] - n // 2
    g = np.exp(-(yy ** 2 + xx ** 2) / (2 * max(float(sigma), 1e-3) ** 2))
    return g / g.sum()


def seeing_kernel(Q: np.ndarray, sigma: float, shape: np.ndarray | None = None) -> np.ndarray:
    """One exposure's PSF from the seeing-free PSF ``Q`` (..., n, n): G(sigma) * Q, then - with a
    ``shape`` (2 x 2) - mapped by it (warp_psf); non-negative, unit sum, same n."""
    n = Q.shape[-1]
    g = _gauss(sigma, n).astype(np.float32)
    flat = Q.reshape(-1, n, n)
    out = np.stack([cv2.filter2D(k.astype(np.float32), -1, g, borderType=cv2.BORDER_CONSTANT) for k in flat]).reshape(Q.shape)
    if shape is not None and not np.allclose(shape, np.eye(2)):
        out = warp_psf(out, shape)
    out = np.clip(out, 0, None)
    return (out / np.maximum(out.sum((-2, -1), keepdims=True), 1e-30)).astype(np.float32)


def fit_seeing_sigma(cutouts, Q_nodes: np.ndarray, nodes, fwhm: float, shape: np.ndarray | None = None,
                     bounds: tuple[float, float] = (0.15, 3.0), max_stars: int = 25, r_fit: float = 2.5,
                     tol: float = 0.01) -> tuple[float, float]:
    """The seeing of one exposure and channel: the Gaussian width sigma for which G(sigma) * Q, Q the
    seeing-free field PSF (``Q_nodes``, (ny, nx, n, n), at each star's position), best fits the
    exposure's brightest ``max_stars`` PSF-star cut-outs (per star an amplitude, a constant and a
    linearised sub-pixel shift are free; pixels within ``r_fit`` FWHM; stars weighted by 1 / noise^2).
    Golden-section search in log sigma (the summed chi^2 is unimodal).  Returns (sigma, chi^2).

    Why a Gaussian on a seeing-free PSF rather than the coadd's PSF radially scaled: the coadd's PSF is
    the mean over exposures of different seeing, and a mixture of Gaussians has a sharper peak and
    broader shoulders than any one of them - no scaling of it fits a single exposure (M 42, DWARF 3:
    0.4-0.6 % of a star's flux too much at 4-6 px for every exposure, which ImageMM removed from the
    sky as a dark ring round every bright star).  With Q the coadd's PSF deconvolved by the mean of
    the exposures' Gaussians (ExposureSet._fit_exposure_kernels), the mean of the exposures' kernels
    is the coadd's PSF by construction, and each exposure's shoulder is its own."""
    S, w, noises, px, py = cutouts
    N, n, _ = S.shape
    if N == 0:
        return float(np.sqrt(bounds[0] * bounds[1])), np.inf
    fl = w.reshape(N, -1).max(1)
    order = np.argsort(-fl)[:max_stars]
    nq = Q_nodes.shape[-1]
    c, h = nq // 2, n // 2
    rr = np.hypot(*np.mgrid[:n, :n] - h)
    fit = rr <= r_fit * fwhm
    stars = []
    for i in order:
        m = (w[i] > 0) & fit
        if m.sum() < 20:
            continue
        Qi = interp_nodes(Q_nodes, nodes, float(py[i]), float(px[i]))
        if shape is not None and not np.allclose(shape, np.eye(2)):
            Qi = warp_psf(Qi[None], shape)[0]
        stars.append((Qi.astype(np.float32), m, S[i][m], 1.0 / max(float(noises[i]), 1e-12) ** 2))
    if not stars:
        return float(np.sqrt(bounds[0] * bounds[1])), np.inf

    def chi2(log_sig):
        g = _gauss(math.exp(log_sig), nq).astype(np.float32)
        tot = 0.0
        for Qi, m, y, wt in stars:
            k = cv2.filter2D(Qi, -1, g, borderType=cv2.BORDER_CONSTANT)[c - h:c + h + 1, c - h:c + h + 1]
            k = k / max(float(k.sum()), 1e-30)
            gy, gx = np.gradient(k)
            A = np.stack([k[m], gy[m], gx[m], np.ones(int(m.sum()))], 1)
            sol, res, *_ = np.linalg.lstsq(A, y, rcond=None)
            r_ = y - A @ sol
            tot += wt * float(r_ @ r_)
        return tot

    a, b = math.log(bounds[0]), math.log(bounds[1])
    gr = (math.sqrt(5) - 1) / 2
    x1, x2 = b - gr * (b - a), a + gr * (b - a)
    f1, f2 = chi2(x1), chi2(x2)
    while (b - a) > math.log(1 + tol):
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - gr * (b - a)
            f1 = chi2(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + gr * (b - a)
            f2 = chi2(x2)
    x = x1 if f1 < f2 else x2
    return float(math.exp(x)), float(min(f1, f2))


def _sigma_task(args):
    """One exposure's seeing per channel (fit_seeing_sigma) against the current Q."""
    cuts, Qs, nodes, fwhm = args
    out = []
    for cc, Q in zip(cuts, Qs):
        if cc is None or Q is None:
            out.append((np.nan, np.inf))
        else:
            out.append(fit_seeing_sigma(_expand_cutouts(cc), Q, nodes, fwhm))
    return out


def _kernel_task(args):
    """One exposure's final kernels per channel: G(sigma) * Q at every node, its own shape (elongation)
    where that fits its cut-outs better, cut to the support, plus the summary entries."""
    fields, refs, fwhm, cuts, sigmas, nodes = args
    out = []
    for f, r, cc, sig in zip(fields, refs, cuts, sigmas):
        if f is None or r is None or cc is None or not np.isfinite(sig):
            out.append(f)
            continue
        Q = np.asarray(r["Q"], np.float32)
        ny, nx, nq, _ = Q.shape
        cu = _expand_cutouts(cc)
        S, w = cu[0], cu[1]
        # the exposure's shape: second moments of its flux^2-weighted mean star against the model's
        wt = w.reshape(len(S), -1).max(1)
        Sm = np.tensordot(wt / max(wt.sum(), 1e-30), np.clip(S, 0, None), 1)
        Km = seeing_kernel(Q.mean((0, 1)), sig)
        shape = np.eye(2)
        Mt, Mc = _weighted_moments(_fit_size(Sm, nq, 0.0), fwhm), _weighted_moments(Km, fwhm)
        if np.all(np.linalg.eigvalsh(Mt) > 0) and np.all(np.linalg.eigvalsh(Mc) > 0):
            A_ = _sqrtm(Mt) @ np.linalg.inv(_sqrtm(Mc))
            U, sv, Vt = np.linalg.svd(A_)
            sv = np.clip(sv / np.sqrt(sv[0] * sv[1]), 0.75, 1.33)
            shape = (U * sv) @ Vt
        sig_, chi_ = sig, None
        if not np.allclose(shape, np.eye(2)):
            _, chi0 = fit_seeing_sigma(cu, Q, nodes, fwhm, bounds=(sig * 0.999, sig * 1.001))
            sig1, chi1 = fit_seeing_sigma(cu, Q, nodes, fwhm, shape=shape)
            if chi1 < chi0:
                sig_, chi_ = sig1, chi1
            else:
                shape = np.eye(2)
        K = seeing_kernel(Q, sig_, shape)
        c = nq // 2
        h_ = min(c, int(math.ceil(float(r["support"]) + 3 * sig_ * max(1.0, float(np.abs(shape).max())))) + 1)
        K = K[..., c - h_:c + h_ + 1, c - h_:c + h_ + 1]
        K = K / np.maximum(K.sum((-2, -1), keepdims=True), 1e-30)
        g = {k: v for k, v in f.items() if k not in ("mean", "se", "nodes")}    # the exposure's own field model, summarised
        g.update({"nodes": K.astype(np.float32), "psf": K.mean((0, 1)).astype(np.float32), "own_psf": f.get("psf"),
                  "sigma": float(sig_), "shape": shape.astype(np.float32), "support": float(r["support"]),
                  "scale": np.ones((ny, nx), np.float32), "model": "mixture", "own_mean": Sm.astype(np.float32)})
        out.append(g)
    return out


def _hybrid_task(args):
    """ExposureSet.prepare's second stage for one exposure: its kernels from the coadd's PSF and its
    own cut-outs (hybrid_psf), per channel."""
    fields, refs, fwhm, cuts, nodes = args
    out = []
    for f, r, cc in zip(fields, refs, cuts):
        if f is None or r is None or cc is None:
            out.append(f)
        else:
            out.append(hybrid_psf(f, r, fwhm, cutouts=_expand_cutouts(cc), nodes=nodes))
    return out


def _prep_worker_init(state: bytes, counter=None, acc=None):
    """Worker process of ExposureSet.prepare: the set (reference, catalogue with its PSF-star
    tables, masks) once.  One thread per process: the processes already use every core, and
    OpenCV's own thread pool in each of them would oversubscribe the CPU.  The worker keeps only
    the sub it is preparing in its raw-frame cache (prepare_one registers it up to four times):
    a worker visits each sub once, and one cache of 35 % of the RAM per worker ran a 32 GB
    machine out of memory."""
    import pickle
    from .resources import exit_with_parent
    global _PREP_SET, _PREP_COUNTER
    exit_with_parent()
    _PREP_COUNTER = counter
    cv2.setNumThreads(1)
    if "torch" in sys.modules:           # the preparation never uses torch: importing it costs ~450 MB
        sys.modules["torch"].set_num_threads(1)
    _PREP_SET = pickle.loads(state)
    _PREP_SET._raw_cache_frames = 1
    if acc is not None:                   # the exposures' coadd, summed in shared memory (ExposureSet.prepare)
        from multiprocessing import shared_memory
        global _PREP_SHM
        name_s, name_w, shape, lock = acc
        _PREP_SHM = (shared_memory.SharedMemory(name=name_s), shared_memory.SharedMemory(name=name_w))
        _PREP_SET._acc = (np.ndarray(shape + (3,), np.float32, buffer=_PREP_SHM[0].buf),
                          np.ndarray(shape, np.float32, buffer=_PREP_SHM[1].buf), lock)


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
        self.coadd = None                         # the exposures' own coadd (prepare), sky included
        self.coverage = None                      # exposures with a valid pixel, per pixel (prepare)
        self._acc = None
        # threads for window extraction (I/O, OpenCV and NumPy release the interpreter lock)
        from .resources import workers_for
        frame = 4.0 * self.W0 * self.H0
        self.workers = workers or workers_for(40 * frame, "ASTROPHOTO_PREP_RAM_GB")
        # per preparation process: one sub's working arrays, the shared state (reference, its sky
        # model and the catalogue, ~9 frames, held twice: unpickled and as the pool's initargs) and
        # the libraries.  Measured on M 31 (3840x2160): 2.1 GB peak = 63 frames.
        self._worker_bytes = 64 * frame
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
        raw = read_frame(info)
        sat = raw >= 0.95 * self.sat
        raw = fix_defects(raw, self.defects)
        limit = self.__dict__.get("_raw_cache_frames")      # a preparation worker: the current sub only
        if limit is not None and len(cache) >= limit:
            cache.clear()
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
        fe_ = math.ceil(fe * 10 - 1e-9) / 10                 # (empirical_psf_field rounds the FWHM the same way)
        cuts = [_psf_cutouts(ybs[..., c], hard, self.cat, half, fe_, 1000) for c in range(3)]
        fields = [empirical_psf_field(ybs[..., c], hard, self.cat, half, fe, self.nodes, cutouts=cu) for c, cu in enumerate(cuts)]
        # the brightest PSF stars' cut-outs, kept for the seeing fit against the coadd's PSF once every
        # exposure has gone into that coadd (prepare); compact: flux and mask instead of the weight map
        return {"refine": refine, "residual": final, "T": T, "T_err": T_err, "surf": coefs,
                "psf": [f["psf"] if f else None for f in fields], "psf_field": fields,
                "psf_stars": [f["n_stars"] if f else 0 for f in fields], "fwhm": fwhm,
                "cuts": [_compact_cutouts(cu, 60) for cu in cuts],
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
        # The reference PSF is measured after the pass, on the plain mean of the registered exposures
        # as window() delivers them (summed below): the pipeline's stack (self.ref) is resampled and
        # combined differently and its stars differ from the exposures' (M 42, DWARF 3: 3 % of a star's
        # flux at 2-4 px in green), so a PSF measured on it was the wrong PSF for the data ImageMM fits -
        # the latent emptied a ring round every bright star (green only) to make up the difference.
        self.psf_ref = [None] * 3
        half_ref = int(math.ceil(3.5 * self.fwhm_max))
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
            state = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
            from multiprocessing import shared_memory
            shm = (shared_memory.SharedMemory(create=True, size=self.H0 * self.W0 * 12),
                   shared_memory.SharedMemory(create=True, size=self.H0 * self.W0 * 4))
            acc_S = np.ndarray((self.H0, self.W0, 3), np.float32, buffer=shm[0].buf)
            acc_W = np.ndarray((self.H0, self.W0), np.float32, buffer=shm[1].buf)
            acc_S[:] = 0
            acc_W[:] = 0
            lock = ctx.Lock()
            ex = ProcessPoolExecutor(max_workers=nproc, mp_context=ctx, initializer=_prep_worker_init,
                                     initargs=(state, counter, (shm[0].name, shm[1].name, (self.H0, self.W0), lock)))
            submit = lambda c: ex.submit(_prep_worker_run, *c)
        else:
            import threading as _th
            shm = None
            acc_S = np.zeros((self.H0, self.W0, 3), np.float32)
            acc_W = np.zeros((self.H0, self.W0), np.float32)
            self._acc = (acc_S, acc_W, _th.Lock())
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
        # the exposures' coadd (sky included, as self.ref), the pipeline's stack where no exposure covers
        seen = acc_W > 0
        self.coadd = np.where(seen[..., None], acc_S / np.maximum(acc_W, 1)[..., None], self.ref).astype(np.float32)
        self.coverage = acc_W.astype(np.float32).copy()   # exposures with a valid pixel, per pixel
        self._acc = None
        if shm is not None:
            for m in shm:
                m.close()
                m.unlink()
        self.ptc = self._ptc_fit(acc)
        say("ImageMM: PSF of the exposures' coadd")
        cbs = self.coadd - self.sky_ref
        psf_valid = seen
        self.psf_ref = [empirical_psf_field(cbs[..., c], psf_valid, self.cat, half_ref, self.fwhm_ref, self.nodes)
                        for c in range(3)]
        for r in self.psf_ref:
            if r is not None:
                r["source"] = "subs"
        self._fit_exposure_kernels(nproc, progress, cancel)
        return self

    def _fit_exposure_kernels(self, nproc: int, progress=None, cancel=None, max_rounds: int = 12, tol: float = 0.01):
        """Second stage of prepare: the seeing-mixture PSF model.  Every exposure's kernel is
        G(sigma_t) * Q, Q the seeing-free field PSF (per channel and node) and sigma_t its seeing
        (fit_seeing_sigma on its retained PSF-star cut-outs).  The coadd's PSF is the mean over the
        exposures of their kernels, so Q = f_coadd deconvolved by mean_t G(sigma_t) (Eq. 11's solver,
        refine_psfs, with that mixture as the kernel) - the two are found by alternation, starting from
        Q = f_coadd, until the median seeing changes by less than ``tol`` px.  Then every exposure's
        kernels are built (and its elongation fitted, _kernel_task) and its cut-outs dropped."""
        from concurrent.futures import ProcessPoolExecutor
        from .imagemm import refine_psfs
        todo = [k for k, p in enumerate(self.params) if p is not None and p.get("cuts") is not None
                and any(f is not None for f in p["psf_field"])]
        say = (lambda msg: progress(0, 1, msg)) if progress else (lambda msg: None)
        if not todo or all(r is None for r in self.psf_ref):
            for p in self.params:
                if p is not None:
                    p.pop("cuts", None)
            return
        fwhm_of = lambda k: max(float(self.params[k]["fwhm"]), self.fwhm_ref)
        f_nodes = [None if r is None else np.asarray(r["nodes"], np.float32) for r in self.psf_ref]
        Q = [None if f is None else f.copy() for f in f_nodes]
        sig = np.full((len(self.params), 3), np.nan)

        def run(fn, args_of):
            if nproc > 1 and len(todo) > 1:
                import multiprocessing as mp
                with ProcessPoolExecutor(max_workers=nproc, mp_context=mp.get_context("spawn")) as ex:
                    return list(ex.map(fn, (args_of(k) for k in todo), chunksize=4))
            return [fn(args_of(k)) for k in todo]

        for rnd in range(max_rounds):
            say(f"ImageMM: exposure seeing, round {rnd + 1}")
            res = run(_sigma_task, lambda k: (self.params[k]["cuts"], Q, self.nodes, fwhm_of(k)))
            new = sig.copy()
            for k, r in zip(todo, res):
                new[k] = [v[0] for v in r]
            ok = np.isfinite(new) & np.isfinite(sig)
            step = float(np.median(np.abs(new - sig)[ok])) if ok.any() else np.inf
            sig = new
            if cancel and cancel():
                raise RuntimeError("cancelled")
            if step < tol:
                break
            # Q per channel: the coadd's PSF deconvolved by the exposures' mean Gaussian
            for c in range(3):
                if f_nodes[c] is None:
                    continue
                sc = sig[:, c][np.isfinite(sig[:, c])]
                if len(sc) == 0:
                    continue
                n = f_nodes[c].shape[-1]
                mix = np.mean([_gauss(v, 2 * n - 1) for v in sc], 0)
                h, _ = refine_psfs(f_nodes[c].reshape(-1, n, n), 1, 1.0, g=mix)
                Q[c] = np.clip(h.reshape(f_nodes[c].shape), 0, None).astype(np.float32)
        for c, r in enumerate(self.psf_ref):
            if r is not None:
                r.update({"Q": Q[c], "model": "mixture", "sigma_median": float(np.nanmedian(sig[:, c])),
                          "rounds": rnd + 1, "sigma_step": step})
        say("ImageMM: exposure kernels")
        res = run(_kernel_task, lambda k: (self.params[k]["psf_field"], self.psf_ref, fwhm_of(k), self.params[k]["cuts"],
                                           list(sig[k]), self.nodes))
        for k, fields in zip(todo, res):
            self.params[k]["psf_field"] = fields
        for p in self.params:
            if p is not None:
                p.pop("cuts", None)
                p["psf"] = [f["psf"] if f else None for f in p["psf_field"]]

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
                if getattr(self, "_acc", None) is not None:
                    S_, W_, lock = self._acc
                    yb = (p["_y"] / p["T"][None, None, :] - p["_surf"]).astype(np.float32)
                    v = p["_valid"].astype(np.float32)
                    with lock:
                        S_ += yb * v[..., None]
                        W_ += v
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
    _STATE = ("params", "ptc", "sky_ref", "sky_info", "fwhm_ref", "cat", "smask", "nodes", "psf_ref", "coadd", "coverage")
    # bumped whenever the preparation changes what it produces; an older cache is prepared again
    # (2: crowded-field PSF stars - with 1, most subs of a Milky Way field had no PSF;
    #  3: PSF cut-outs must agree with each other - with 2, bright non-point sources set the wings)
    #  4: field-dependent PSFs (empirical_psf_field) with the coadd's wings (hybrid_psf), least-squares
    #     photometric scales)
    #  7: per-exposure PSFs = the coadd's field PSF scaled to the exposure's seeing (hybrid_psf)
    #  8: ... and mapped to the exposure's own shape (elongation and its direction)
    #  9: PSF stars include the bright ones (_psf_star_table's self-match): the exposures' seeing scales
    #     were fitted on faint stars only (M 42: slope 0.69 against the scale measured on bright stars)
    # 10: the exposures' seeing scale and shape fitted on their PSF-star cut-outs (fit_seeing_scale)
    # 11: the reference PSF measured on the exposures' own coadd, not the pipeline's stack
    # 12: seeing-mixture kernels, G(sigma_t) * Q (fit_seeing_sigma, _fit_exposure_kernels)
    # 13: (withdrawn: PSF stars restricted by coverage left too few)
    # 14: the exposures' coverage kept in the state
    PREP_VERSION = 14
    SEEING_NORM = "channel"    # normalise_seeing: channel | shared | off (ASTROPHOTO_SEEING_NORM overrides)

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
            setattr(self, k, st.get(k))              # (entries added later, e.g. "coverage", may be missing)
        return self

    def normalise_seeing(self, lams: np.ndarray = np.round(np.arange(0.80, 1.50001, 0.01), 2)) -> list[float]:
        """Make the exposures' PSFs consistent with the coadd's: the coadd is the mean of the registered
        exposures, so its PSF is the mean of theirs.  ``hybrid_psf`` scales the coadd's PSF to each
        exposure's seeing, fitted to the exposure's own mean star profile, and that profile is
        broadened by the noise of a short sub (centroiding the noisy PSF stars before averaging):
        on M 42 (DWARF 3, 15 s subs) the scale was 1.09 for the median sub - broader than the coadd
        they make up - the mean of the exposures' PSFs held 5-11 % less light in the core and 3-8 %
        more in the wings than the coadd's, and ImageMM, explaining each star's wings as the blur of
        a too-compact core and unable to put negative light on the sky, emptied a disc round every
        star to fit the data.

        So per channel the exposures keep their relative seeing (s_t / s_u, which the fits measure well)
        but share one factor lambda >= 1, the one for which the mean of their kernels K(s_t / lambda) best
        matches the coadd's (least squares over the coadd PSF's support, field means), and every
        exposure's node kernels are rebuilt from the coadd's at s_t / lambda.  Per channel, because the
        bias differs between them (M 42: 1.22 / 1.13 / 1.31; one shared 1.21 left green kernels too
        narrow, and a green halo round every star).  lambda < 1 is not applied (see below; per channel
        and unclamped, M 31's 0.89 / 0.94 / 0.88 made its stars 7 % less red than the coadd's).
        ``SEEING_NORM`` (or the environment variable ASTROPHOTO_SEEING_NORM): "channel" (default),
        "shared" (one factor for the three) or "off" (lambda = 1, the kernels still rebuilt from the
        coadd's PSF).  Done once per mode (recorded per channel as
        "seeing_norm"); returns the factors."""
        out = []
        idx = [k for k, p in enumerate(self.params) if p is not None]
        if self.psf_ref and all(r is None or r.get("model") == "mixture" for r in self.psf_ref):
            # the seeing-mixture model (prepare): the mean of the exposures' kernels is the coadd's PSF
            # by construction, there is no factor to fit
            self.seeing_norm = [1.0, 1.0, 1.0]
            return self.seeing_norm
        # a preparation made before the robust wings (_robust_wings): the coadd's PSF is measured
        # again (a minute; the exposures need no new preparation), and every exposure's kernels below
        # are rebuilt from it
        nodes_now = psf_nodes(self.H0, self.W0)
        if self.psf_ref and any(r is not None and (r.get("wings") != "harmonic" or "field" not in r
                                                  or r.get("source") != "subs"
                                                  or len(self.nodes[0]) != len(nodes_now[0])
                                                  or len(self.nodes[1]) != len(nodes_now[1])) for r in self.psf_ref) \
                and getattr(self, "coadd", None) is not None:
            self.nodes = nodes_now                        # (the exposures' kernels below follow the new grid)
            ref_bs = self.coadd - self.sky_ref
            half_ref = max(int(np.asarray(r["nodes"]).shape[-1]) // 2 for r in self.psf_ref if r is not None)
            cov = getattr(self, "coverage", None)
            psf_valid = cov > 0 if cov is not None else np.ones(ref_bs.shape[:2], bool)
            self.psf_ref = [empirical_psf_field(ref_bs[..., c], psf_valid, self.cat, half_ref,
                                                self.fwhm_ref, self.nodes) for c in range(3)]
            for r in self.psf_ref:
                if r is not None:
                    r["source"] = "subs"
            for p in self.params:                         # the exposures' kernels are rebuilt below
                if p is not None:
                    for f in p["psf_field"]:
                        if f is not None:
                            f.pop("seeing_norm", None)
        mode = os.environ.get("ASTROPHOTO_SEEING_NORM", self.SEEING_NORM)
        plan = []                                         # per channel: (subs, st, scaled, N, sup, errors)
        for c in range(3):
            ref = self.psf_ref[c] if self.psf_ref else None
            subs = [k for k in idx if self.params[k]["psf_field"][c] is not None and "scale" in self.params[k]["psf_field"][c]]
            if ref is None or len(subs) < 3:
                plan.append(None)
                continue
            if all(self.params[k]["psf_field"][c].get("seeing_norm") is not None
                   and self.params[k]["psf_field"][c].get("seeing_mode") == mode for k in subs):
                plan.append(float(self.params[subs[0]]["psf_field"][c]["seeing_norm"]))
                continue
            Kr = np.asarray(ref["nodes"], np.float32)                     # (ny, nx, nr, nr), unit sum
            nr = Kr.shape[-1]
            # (the fitted scales: "scale" holds the normalised one once this has run)
            st = np.array([float(np.median(self.params[k]["psf_field"][c].get("scale_fit", self.params[k]["psf_field"][c]["scale"])))
                           for k in subs])
            N = int(np.ceil(nr * max(st.max() / lams.min(), 1.0))) | 1
            o = (N - nr) // 2
            Kp = np.pad(Kr, [(0, 0), (0, 0), (o, o), (o, o)])
            Km = Kp.mean((0, 1))                                          # the coadd's field-mean PSF
            cache: dict = {}

            def scaled(sc, Kp=Kp, cache=cache):
                sc = round(float(sc), 3)
                if sc not in cache:
                    cache[sc] = scale_psf(Kp, sc)
                return cache[sc]

            def mean_kernel(lam, st=st, scaled=scaled):
                q, n = np.unique(np.round(st / lam, 3), return_counts=True)
                return sum(nn * scaled(sc).mean((0, 1)) for sc, nn in zip(q, n)) / len(st)
            err = np.array([float(((mean_kernel(l) - Km) ** 2).sum()) / max(float((Km ** 2).sum()), 1e-30) for l in lams]) \
                if mode != "off" else None
            plan.append((subs, st, scaled, N, float(ref["support"]), err))
        # the factor: per channel, one shared by the channels (their relative widths - the colour of a
        # restored star - kept as measured; per channel, M 31's restored stars came out 7 % less red in
        # aperture photometry than the coadd), or none (the kernels still rebuilt from the coadd's PSF)
        errs = [q[5] for q in plan if isinstance(q, tuple)]
        # Only lambda > 1 is applied: it undoes the broadening noise gives a short sub's fitted seeing.
        # lambda < 1 means the exposures are sharper than the coadd they make up - registration jitter
        # broadens the coadd, and the exposures' kernels rightly leave it out (M 31: 0.91, and applying
        # it deepened the rings round its stars: -4 ADU at 4-12 px against -1 with lambda 1)
        lam_shared = max(float(lams[int(np.argmin(np.sum(errs, 0)))]), 1.0) if mode == "shared" and errs else 1.0
        for c, q in enumerate(plan):
            if q is None:
                out.append(1.0)
                continue
            if not isinstance(q, tuple):
                out.append(q)
                continue
            subs, st, scaled, N, sup, err = q
            lam = max(float(lams[int(np.argmin(err))]), 1.0) if mode == "channel" else lam_shared
            for k in subs:
                f = self.params[k]["psf_field"][c]
                fit = f.get("scale_fit", f["scale"])
                sc = float(np.median(fit)) / lam
                K = scaled(sc)
                if f.get("shape") is not None and not np.allclose(f["shape"], np.eye(2)):
                    K = warp_psf(K, np.asarray(f["shape"], np.float64))
                h = min(N // 2, int(np.ceil(sup * sc)) + 1)
                K = K[..., N // 2 - h:N // 2 + h + 1, N // 2 - h:N // 2 + h + 1]
                K = K / np.maximum(K.sum((-2, -1), keepdims=True), 1e-30)
                f2 = dict(f)
                f2.update({"nodes": K.astype(np.float32), "mean": K.astype(np.float32), "psf": K.mean((0, 1)).astype(np.float32),
                           "scale": np.full_like(np.asarray(f["scale"], np.float32), sc), "seeing_norm": lam,
                           "seeing_mode": mode, "scale_fit": fit,
                           "se": _fit_size(np.asarray(f["se"], np.float32), K.shape[-1])})
                self.params[k]["psf_field"][c] = f2
                self.params[k]["psf"][c] = f2["psf"]
            out.append(lam)
        self.seeing_norm = out
        return out

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
