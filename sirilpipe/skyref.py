"""Gradient removal against a calibrated sky survey (the idea of PixInsight's MARS database and
MultiscaleGradientCorrection: the reference is observational data, not "free sky" samples).

Gradients (light pollution, moonlight, airglow, residual vignetting) are smooth.  Sampling the sky
between objects cannot tell them from faint emission that fills the field: on C 20 (the North
America Nebula) a degree-2 model took the nebula's diffuse body, ~20 % of its large-scale
contrast.  A survey with linear, background-calibrated intensities shows what the sky really holds
there.  The image is fitted, per colour channel, as a linear mix of the survey's maps plus a smooth
polynomial (star-masked block medians, Huber-robust); only the polynomial - what the survey cannot
explain - is the gradient.  The mix also gives the emission each block holds, which sets the sky's
zero point (``emission`` in the result).

Reference: NSNS DR0.2, the Northern Sky Narrowband Survey (http://www.simg.de/nebulae3/dr0_2):
H-alpha calibrated in Rayleighs against WHAM, [OIII], and the star-subtracted visual continuum,
~10" resolution, declination -16 to +76 degrees; CC BY-NC-SA 4.0, doi:10.3847/2515-5172/adfec7.
Served as HiPS by CDS (hips2fits) and resampled onto the stack's plate solution.  Its [OIII] and
continuum maps are background-filtered above ~3 degrees: on wider fields, structure larger than
that is not in the reference and is taken for gradient.
"""
from __future__ import annotations

import io
import json
import math
import urllib.parse
import urllib.request

import cv2
import numpy as np

SURVEYS = [("halpha", "simg.de/P/NSNS/DR0_2/halpha"),
           ("oiii", "simg.de/P/NSNS/DR0_2/oiii"),
           ("continuum", "simg.de/P/NSNS/DR0_2/vc")]
# the same HiPS on the survey's own server (its properties' hips_service_url): read directly when CDS hips2fits does not answer
HIPS_SERVICE = {"halpha": "http://www.simg.de/nebulae3/dr0_2/halpha", "oiii": "http://www.simg.de/nebulae3/dr0_2/oiii",
                "continuum": "http://www.simg.de/nebulae3/dr0_2/vc"}
HIPS_ORDER_MAX = 6       # the survey's hips_order (tiles), 512 px tiles: pixel order 15
DIRECT_RETRY_PAUSE = 10.0  # s before the one retry of the survey's own server after a network failure
NAME = "NSNS DR0.2"
CREDIT = "NSNS DR0.2 (Northern Sky Narrowband Survey, simg.de; CC BY-NC-SA 4.0, doi:10.3847/2515-5172/adfec7)"
# The gradient scale: light pollution and airglow vary over degrees, more than any field, so once the survey has explained the emission the
# gradient is what stays smooth across the whole frame - the frame's long side (``fit``'s default).  A smaller scale (MultiscaleGradientCorrection's
# 1024 px default) lets the model follow structure the survey does not hold (its [OIII] and continuum are background-filtered above ~3 degrees,
# reflection nebulosity is not in its line maps): the model then carries the field's own light and subtracting it lays bands across the frame.
GRADIENT_CLIP = 3.0      # robust sigma: blocks whose residual stands further from the smooth model are not sky (sigma clipping, as background samples)
HIPS2FITS = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
FACTOR = 8               # reference pixels: FACTOR x FACTOR stack pixels (~20" for a DWARF / Seestar)
FETCH_VERSION = 2        # part of the skyref.npz cache key: 2 = the reduced WCS with the pixel scale (caches of 1 hold a radially smeared sky)


def reduced_wcs(wcs, factor: int):
    """The stack's TAN projection on a grid ``factor`` times coarser (SIP dropped: sub-pixel there)."""
    w = wcs.deepcopy()
    w.sip = None
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [(wcs.wcs.crpix[0] - 0.5) / factor + 0.5, (wcs.wcs.crpix[1] - 0.5) / factor + 0.5]
    # degrees per pixel = PC x CDELT (or CD): get_pc() alone is the scale only for a CD-matrix header; with PC + CDELT
    # (Siril 1.4 plate solutions) it asked hips2fits for ~6 degrees per reduced pixel and the "survey" was the whole sky smeared radially
    w.wcs.pc = wcs.pixel_scale_matrix * factor
    w.wcs.cdelt = [1.0, 1.0]
    return w


def fetch(wcs, shape, factor: int = FACTOR, timeout: float = 120.0) -> tuple[np.ndarray, str]:
    """(maps, source): the survey's maps on the stack's grid reduced by ``factor`` - (h, w, len(SURVEYS)) float32, NaN outside the survey's
    coverage - and where they came from.  CDS hips2fits first; when it does not answer, the same HiPS read from the survey's own server and
    reprojected here (``fetch_direct``).  Raises when neither source answers."""
    try:
        return fetch_hips2fits(wcs, shape, factor, timeout), "CDS hips2fits"
    except Exception as e:                                             # noqa: BLE001  CDS unreachable / timed out
        try:
            try:
                return fetch_direct(wcs, shape, factor), "the survey's own HiPS server (simg.de)"
            except OSError:                                            # a transient network / name-lookup failure: once more after a pause
                import time
                time.sleep(DIRECT_RETRY_PAUSE)
                return fetch_direct(wcs, shape, factor), "the survey's own HiPS server (simg.de)"
        except Exception as e2:                                        # noqa: BLE001
            raise RuntimeError(f"CDS hips2fits: {type(e).__name__}: {str(e)[:120]}; the survey's own HiPS server: {type(e2).__name__}: {str(e2)[:120]}") from e2


def fetch_direct(wcs, shape, factor: int = FACTOR) -> np.ndarray:
    """``fetch`` from the survey's own HiPS (HIPS_SERVICE): the tiles covering the field at the coarsest order whose pixels still resolve the
    reduced grid, as one tile-aligned cutout of the HPX image (``reproject.hips.hips_as_dask_array``; a cutout that splits a tile comes back
    as whole tiles, so the cut follows the tile grid), reprojected onto the reduced TAN grid bilinearly (hips2fits' own sampling), NaN outside
    the survey's coverage."""
    from reproject import reproject_interp
    from reproject.hips import hips_as_dask_array
    H, W = shape[:2]
    h, w = int(math.ceil(H / factor)), int(math.ceil(W / factor))
    out_wcs = reduced_wcs(wcs, factor)
    step = float(np.sqrt(abs(np.linalg.det(out_wcs.pixel_scale_matrix))))          # degrees per reduced pixel
    level = HIPS_ORDER_MAX
    while level > 0 and math.degrees(math.sqrt(4 * math.pi / (12 * 4 ** (level - 1 + 9)))) <= step:
        level -= 1                                                     # the HiPS pixel at order level + 9 stays at or below the reduced pixel
    yy, xx = np.mgrid[0:h:max(1, h // 16), 0:w:max(1, w // 16)]
    xs, ys = np.r_[xx.ravel(), 0, w - 1, 0, w - 1], np.r_[yy.ravel(), 0, 0, h - 1, h - 1]
    sky = out_wcs.pixel_to_world(xs, ys)
    out = []
    for name, _ in SURVEYS:
        arr, hw = hips_as_dask_array(HIPS_SERVICE[name], level=level)
        px, py = hw.world_to_pixel(sky)
        T = int(arr.chunksize[0])
        y0, y1 = int(np.floor(np.nanmin(py))) // T * T, -(-(int(np.ceil(np.nanmax(py))) + 2) // T) * T
        x0, x1 = int(np.floor(np.nanmin(px))) // T * T, -(-(int(np.ceil(np.nanmax(px))) + 2) // T) * T
        y0, x0 = max(y0, 0), max(x0, 0)
        cut = np.asarray(arr[y0:y1, x0:x1].compute(), np.float32)
        a, _fp = reproject_interp((cut, hw[y0:y0 + cut.shape[0], x0:x0 + cut.shape[1]]), out_wcs, shape_out=(h, w), order="bilinear")
        out.append(np.asarray(a, np.float32))
    return np.stack(out, -1)


def fetch_hips2fits(wcs, shape, factor: int = FACTOR, timeout: float = 120.0) -> np.ndarray:
    """``fetch`` from CDS hips2fits."""
    from astropy.io import fits
    H, W = shape[:2]
    h, w = int(math.ceil(H / factor)), int(math.ceil(W / factor))
    hdr = reduced_wcs(wcs, factor).to_header()
    d = {k: hdr[k] for k in hdr}
    d.update(NAXIS=2, NAXIS1=w, NAXIS2=h)
    out = []
    for _, hips in SURVEYS:
        url = HIPS2FITS + "?" + urllib.parse.urlencode({"hips": hips, "wcs": json.dumps(d), "format": "fits"})
        req = urllib.request.Request(url, headers={"User-Agent": "AstroPhoto-Studio"})
        data = urllib.request.urlopen(req, timeout=timeout).read()
        a = np.asarray(fits.getdata(io.BytesIO(data)), np.float32)
        if a.shape != (h, w):
            raise RuntimeError(f"{hips}: got a {a.shape} map, expected {(h, w)}")
        out.append(a)
    return np.stack(out, -1)


def _nan_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing that ignores NaNs (normalised convolution); NaN where nothing is near."""
    ok = np.isfinite(a)
    v = cv2.GaussianBlur(np.where(ok, a, 0).astype(np.float32), (0, 0), sigma)
    wgt = cv2.GaussianBlur(ok.astype(np.float32), (0, 0), sigma)
    return np.where(wgt > 0.2, v / np.maximum(wgt, 1e-6), np.nan)


def _local_linear(a: np.ndarray, sigma: float) -> np.ndarray:
    """Local linear (LOESS-type) Gaussian kernel regression of ``a`` (NaN = no sample): at every point the weighted least-squares plane of
    its neighbourhood, evaluated there.  A weighted mean is biased toward the interior wherever the kernel meets the frame's edge (a steep
    gradient into a corner comes out flattened); the local plane has no first-order boundary bias (Cleveland 1979, Fan & Gijbels 1996)."""
    ok = np.isfinite(a)
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    yy, xx = (yy - h / 2) / max(sigma, 1.0), (xx - w / 2) / max(sigma, 1.0)      # scaled for conditioning
    g = lambda v: cv2.GaussianBlur(np.where(ok, v, 0.0).astype(np.float64), (0, 0), sigma, borderType=cv2.BORDER_CONSTANT)  # noqa: E731
    one, v = np.ones_like(xx), np.where(ok, a, 0.0)
    S = {k: g(f) for k, f in (("1", one), ("x", xx), ("y", yy), ("xx", xx * xx), ("xy", xx * yy), ("yy", yy * yy))}
    T = {k: g(f * v) for k, f in (("1", one), ("x", xx), ("y", yy))}
    M = np.stack([np.stack([S["1"], S["x"], S["y"]], -1), np.stack([S["x"], S["xx"], S["xy"]], -1), np.stack([S["y"], S["xy"], S["yy"]], -1)], -2)
    rhs = np.stack([T["1"], T["x"], T["y"]], -1)[..., None]
    M = M + np.eye(3) * 1e-9 * np.maximum(S["1"], 1e-12)[..., None, None]
    beta = np.linalg.solve(M, rhs)[..., 0]
    out = beta[..., 0] + beta[..., 1] * xx + beta[..., 2] * yy
    return np.where(S["1"] > 1e-6, out, np.nan).astype(np.float32)


def fit(img: np.ndarray, refs: np.ndarray, px_per_ref: float, origin=(0, 0), smooth: float = 1.5, min_coverage: float = 0.6,
        star_mask=None, source: str = "", gradient_scale: float | None = None) -> tuple[np.ndarray, dict]:
    """Gradient model of ``img`` (h, w, 3) against the survey maps ``refs`` (on a grid of ``px_per_ref`` image pixels per reference pixel,
    ``origin``: the image's (y, x) offset on the grid the references were made for, e.g. a crop), as MultiscaleGradientCorrection does it:
    the survey's maps are scaled to the image on the structure below the gradient scale (where a gradient has no power), and the gradient is the large-scale
    component of the image minus the scaled survey - local linear Gaussian kernel regression at ``gradient_scale`` px (default: the frame's long
    side; no edge bias),
    sigma-clipped like background samples.
    Local, so every part of the frame - a corner too - is modelled from the data around it; a global polynomial fitted over the whole field
    rings (alternating diagonal gradients across the frame) and overshoots in the corners where few blocks hold it.  Returns (model on the image
    grid, info); raises ValueError when too little of the field is covered by usable reference data."""
    from .finish import luminance_hwc as luminance, mad_sigma
    star_mask_simple = _star_mask_simple
    h, w = img.shape[:2]
    b = max(2, int(round(px_per_ref)))
    hb, wb = h // b, w // b
    if hb < 8 or wb < 8:
        raise ValueError("the image is too small for the survey's resolution")
    sm = star_mask_simple(luminance(img)) if star_mask is None else star_mask
    blk = img[:hb * b, :wb * b].reshape(hb, b, wb, b, 3).transpose(0, 2, 1, 3, 4).reshape(hb, wb, b * b, 3)
    okb = ~sm[:hb * b, :wb * b].reshape(hb, b, wb, b).transpose(0, 2, 1, 3).reshape(hb, wb, b * b)
    D = np.stack([np.nanmedian(np.where(okb, blk[..., c], np.nan), axis=-1) for c in range(3)], -1)
    del blk
    # the references at the blocks' centres
    yc = (np.arange(hb) + 0.5) * b - 0.5 + origin[0]
    xc = (np.arange(wb) + 0.5) * b - 0.5 + origin[1]
    my, mx = np.meshgrid((yc + 0.5) / px_per_ref - 0.5, (xc + 0.5) / px_per_ref - 0.5, indexing="ij")
    R = np.stack([cv2.remap(refs[..., k].astype(np.float32), mx.astype(np.float32), my.astype(np.float32),
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))
                  for k in range(refs.shape[-1])], -1)
    # matched resolution: both seen through the same Gaussian
    # each map's pixel noise, from its own neighbour differences before the matching blur (MAD of horizontal differences / sqrt 2), then as the
    # blur leaves it (a Gaussian of sigma s blocks averages ~4 pi s^2 independent pixels): the regressors' noise for the errors-in-variables fit
    noise_r = []
    for k in range(R.shape[-1]):
        dk = np.diff(R[..., k], axis=1)
        dk = dk[np.isfinite(dk)]
        noise_r.append((1.4826 * float(np.median(np.abs(dk - np.median(dk)))) / np.sqrt(2.0) if dk.size else 0.0) / np.sqrt(max(1.0, 4 * np.pi * smooth ** 2)))
    R = np.stack([_nan_blur(R[..., k], smooth) for k in range(R.shape[-1])], -1)
    D = np.stack([_nan_blur(D[..., c], smooth) for c in range(3)], -1)
    usable = (okb.mean(-1) > 0.3) & np.isfinite(R).all(-1) & np.isfinite(D).all(-1)
    cover = float(usable.mean())
    if cover < min_coverage:
        raise ValueError(f"only {cover * 100:.0f} % of the field has survey data ({NAME} covers declination "
                         "-16 to +76 degrees)")
    # columns: the survey maps (scaled to unit spread for conditioning) and the polynomial
    sd = np.array([np.std(R[..., k][usable]) or 1.0 for k in range(R.shape[-1])])
    from scipy.optimize import nnls
    nk = R.shape[-1]
    gradient_scale = float(max(h, w)) if gradient_scale is None else float(gradient_scale)
    sig_b = 0.5 * gradient_scale / b
    # the survey's scale factors from the structure below the gradient scale: a gradient has no power there, so image and survey high-passed
    # at that scale relate by the survey's scale factors alone (fitting the maps beside a smooth model lets the maps take part of a gradient's
    # shape whenever that model cannot follow it).  Huber IRLS (1.5 robust sigma) per channel, non-negative least squares.
    U = np.where(usable[..., None], 1.0, np.nan).astype(np.float32)
    Rh = np.stack([R[..., k] - _local_linear(R[..., k] * U[..., 0], sig_b) for k in range(nk)], -1)
    # the fit uses star-free blocks only, beyond the smoothing kernel's reach (3 sigma) of any masked star pixel: the survey's maps carry
    # residue at bright stars and a star's halo is in the image's block medians, and the two correlate without being emission
    reach = 2 * int(math.ceil(3 * smooth)) + 1
    clean = usable & (cv2.erode((okb.mean(-1) >= 1.0).astype(np.uint8), np.ones((reach, reach), np.uint8)) > 0)
    fit_on = clean if clean.sum() >= max(50, 0.1 * usable.sum()) else usable
    A = (Rh / sd)[fit_on]
    nv = (np.asarray(noise_r) / sd) ** 2                               # the regressors' noise variance, in the scaled units of A
    coefs, scatter = [], []
    for c in range(3):
        y = (D[..., c] - _local_linear(D[..., c] * U[..., 0], sig_b))[fit_on]
        ok = np.isfinite(y) & np.isfinite(A).all(-1)
        Ac, yc = A[ok], y[ok]
        wgt = np.ones(len(yc))
        for _ in range(15):
            # errors in variables: noise in a regressor biases a least-squares slope toward 0 (attenuation), so its noise variance is taken out
            # of the normal equations (G = A'WA - sum(W) diag(noise^2)); solved non-negatively (emission only adds light) through G's Cholesky factor
            w2 = wgt ** 2
            G = (Ac * w2[:, None]).T @ Ac - float(w2.sum()) * np.diag(nv)
            rhs = (Ac * w2[:, None]).T @ yc
            try:
                Lc = np.linalg.cholesky(G)
                beta, _rn = nnls(Lc.T, np.linalg.solve(Lc, rhs))
            except np.linalg.LinAlgError:                             # the maps' noise exceeds their signal here: the plain fit
                beta, _rn = nnls(Ac * wgt[:, None], yc * wgt)
            r = yc - Ac @ beta
            sc = 1.4826 * float(np.median(np.abs(r))) + 1e-12
            wgt = np.sqrt(np.minimum(1.0, 1.5 * sc / np.maximum(np.abs(r), 1e-12)))
        coefs.append(beta / sd)
        scatter.append(sc)
    # the gradient: the large-scale component of what the scaled survey does not explain, per channel; blocks with no reference, or whose
    # residual is object light the survey lacks (a star's halo), are left out and the smoothing fills them from their surroundings
    lows = []
    for c in range(3):
        r = np.where(usable, D[..., c] - np.nan_to_num(R) @ coefs[c], np.nan).astype(np.float32)
        m = _local_linear(r, sig_b)
        for _ in range(3):
            dev = r - m
            sc = 1.4826 * float(np.nanmedian(np.abs(dev))) + 1e-12
            m = _local_linear(np.where(np.abs(dev) <= GRADIENT_CLIP * sc, r, np.nan).astype(np.float32), sig_b)
        lows.append(np.nan_to_num(m, nan=float(np.nanmedian(m))))
    low = np.stack(lows, -1).astype(np.float32)
    bg = cv2.resize(low, (wb * b, hb * b), interpolation=cv2.INTER_CUBIC)
    if bg.shape[:2] != (h, w):
        bg = cv2.copyMakeBorder(bg, 0, h - bg.shape[0], 0, w - bg.shape[1], cv2.BORDER_REPLICATE)
    # the emission the survey predicts per block (luminance), for the sky's zero point
    E = np.stack([np.nan_to_num(R) @ coefs[c] for c in range(3)], -1) @ np.array([0.2126, 0.7152, 0.0722])
    E = np.where(usable, E, np.nan).astype(np.float32)
    noise = mad_sigma((luminance(img) - cv2.GaussianBlur(luminance(img), (0, 0), 1.5))[::3, ::3])
    info = {"method": f"reference ({NAME})", "gradient_scale_px": float(gradient_scale), "coverage": round(cover, 3),
            "fit_blocks": int(fit_on.sum()), "star_free_blocks": int(clean.sum()),
            "survey_coefficients": {name: [float(coefs[c][k]) for c in range(3)] for k, (name, _) in enumerate(SURVEYS)},
            "gradient_range": float(np.ptp(low @ np.array([0.2126, 0.7152, 0.0722])) / max(noise, 1e-12)),
            "fit_scatter": [float(s) for s in scatter], "credit": CREDIT + (f", via {source}" if source else ""),
            "_emission": E, "_emission_block": b}
    return bg, info


def emission(refs: np.ndarray, info: dict, shape, factor: int = FACTOR) -> np.ndarray:
    """The emission the survey predicts on the image grid, (H, W, 3): its maps (on the grid reduced by ``factor``) times ``fit``'s scale factors."""
    H, W = shape[:2]
    hr, wr = refs.shape[:2]
    up = cv2.resize(np.nan_to_num(refs).astype(np.float32), (wr * factor, hr * factor), interpolation=cv2.INTER_CUBIC)[:H, :W]
    k = info["survey_coefficients"]
    return np.stack([sum(up[..., i] * float(k[name][c]) for i, (name, _) in enumerate(SURVEYS)) for c in range(3)], -1).astype(np.float32)


def bright_stars(wcs, shape, factor: int = FACTOR, path: str | None = None) -> list[tuple[str, float, float, float]]:
    """The catalogued bright stars (Siril's stars.csv) whose halo can reach the field: (name, magnitude, x, y) on the grid reduced by ``factor``,
    within HALO_R_MAX reference pixels of it.  Empty when the catalogue is not there."""
    import csv
    import os
    from . import config
    path = path or os.path.join(config.SIRIL_CATALOGUE, "stars.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("ra") and r.get("dec")]
    if not rows:
        return []
    ra = np.array([float(r["ra"]) for r in rows])
    dec = np.array([float(r["dec"]) for r in rows])
    # only stars on the field's side of the sky (the TAN projection maps the far hemisphere onto the plane too)
    c0 = wcs.wcs.crval
    cosd = (np.sin(np.radians(dec)) * np.sin(np.radians(c0[1])) +
            np.cos(np.radians(dec)) * np.cos(np.radians(c0[1])) * np.cos(np.radians(ra - c0[0])))
    near = cosd > np.cos(np.radians(20.0))
    if not near.any():
        return []
    x, y = wcs.world_to_pixel_values(ra[near], dec[near])
    gx, gy = (np.asarray(x) + 0.5) / factor - 0.5, (np.asarray(y) + 0.5) / factor - 0.5
    h, w = int(math.ceil(shape[0] / factor)), int(math.ceil(shape[1] / factor))
    keep = (gx > -HALO_R_MAX) & (gx < w + HALO_R_MAX) & (gy > -HALO_R_MAX) & (gy < h + HALO_R_MAX)
    sel = [r for r, n in zip(rows, near) if n]
    return [(sel[i]["name"], float(sel[i]["mag"] or 99), float(gx[i]), float(gy[i])) for i in np.nonzero(keep)[0]]


# A survey's maps carry its own telescopes' scattered light around bright stars (NSNS [OIII] around zeta Ori: 370x the map's median at the
# star, falling as r^-1.9 out to ~40 reference pixels).  Scaled to the image as emission, that light is subtracted from the gradient model's
# input, the model dips there and the correction leaves the survey's halo in the picture as a coloured glow.  A halo is the star's aureole:
# a core and a power-law fall-off, ~r^-2 (Racine 1996, PASP 108, 699; King 1971, PASP 83, 199 - r^-2 to r^-3 far out); emission around a
# star is not.  So per star and map: the annulus medians about the star (robust to emission covering part of an annulus), and the halo is
# taken out only where the profile peaks at the star and falls off as background + A r^-alpha, alpha within HALO_ALPHA, significantly better
# than a flat background (HALO_DCHI2); a plateau or a rise (an HII region around its ionising star) is left as it is.
HALO_R_MAX = 128         # reference pixels: the widest halo considered (~2500" at 20"/px)
HALO_ALPHA = (1.5, 4.0)  # the aureole's power-law index range accepted (the literature's r^-2 .. r^-3, with room for the fit's spread);
                         # a fit that ends on either bound is no aureole (a nebula's slow decline away from a star fits r^-1)
HALO_DCHI2 = 25.0        # the halo model's chi^2 improvement over a flat background required (~5 sigma)


def _halo_profile(m: np.ndarray, r: np.ndarray, edges: np.ndarray):
    """Annulus medians, their standard errors (1.2533 x MAD-sigma / sqrt n) and pixel counts of map ``m`` (NaN = no data) in the rings ``edges``."""
    ok = np.isfinite(m)
    med, err, cnt = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        v = m[ok & (r >= a) & (r < b)]
        if v.size < 8:
            med.append(np.nan); err.append(np.nan); cnt.append(v.size)
            continue
        q = float(np.median(v))
        med.append(q)
        err.append(1.2533 * 1.4826 * float(np.median(np.abs(v - q))) / math.sqrt(v.size) + 1e-12)
        cnt.append(v.size)
    return np.array(med), np.array(err), np.array(cnt)


def remove_star_halos(refs: np.ndarray, stars) -> tuple[np.ndarray, list[dict]]:
    """``refs`` (h, w, k) without the bright ``stars``' halos (see HALO_*): returns (maps, one record per halo removed)."""
    from scipy.optimize import least_squares
    out = refs.astype(np.float32).copy()
    h, w = out.shape[:2]
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    edges = np.unique(np.round(np.geomspace(1.0, HALO_R_MAX, 24)).astype(int))
    edges = np.r_[0, edges]
    rc = np.sqrt(np.maximum(edges[:-1], 0.5) * edges[1:])                     # each ring's representative radius
    done = []
    for name, mag, gx, gy in sorted(stars, key=lambda s: s[1]):
        r = np.hypot(Y - gy, X - gx)
        if not (r < HALO_R_MAX).any():
            continue
        for k in range(out.shape[-1]):
            med, err, _ = _halo_profile(out[..., k], r, edges)
            v = np.isfinite(med)
            if v.sum() < 6:
                continue
            first = int(np.argmax(v))
            far = float(np.median(med[v][-3:]))
            # it peaks at the star: the innermost ring with data holds the profile's maximum (within its error)
            if med[first] + 3 * err[first] < np.nanmax(med):
                continue
            # the core's edge: the first ring where the excess has fallen below half the peak's; the power law is fitted from there out
            exc0 = med[first] - far
            if exc0 <= 3 * err[first]:
                continue
            below = np.nonzero(v & (np.arange(len(med)) > first) & (med - far < 0.5 * exc0))[0]
            if below.size == 0:
                continue
            j0 = int(below[0])
            sel = v & (np.arange(len(med)) >= j0)
            if sel.sum() < 4:
                continue
            rr, mm, ee = rc[sel], med[sel], err[sel]
            # robust (soft L1 beyond 3 standard errors): a ring that crosses other emission is an outlier to both models, not a reason to
            # reject the aureole or to bend it
            null = least_squares(lambda p: (p[0] - mm) / ee, x0=[far], loss="soft_l1", f_scale=3.0)
            fit = least_squares(lambda p: (p[0] + np.exp(p[1]) * rr ** -p[2] - mm) / ee,
                                x0=[far, math.log(max(exc0, 1e-12)) + 2 * math.log(max(rc[j0], 1.0)), 2.0],
                                bounds=([-np.inf, -np.inf, HALO_ALPHA[0]], [np.inf, np.inf, HALO_ALPHA[1]]), loss="soft_l1", f_scale=3.0)
            b, A, alpha = float(fit.x[0]), float(math.exp(fit.x[1])), float(fit.x[2])
            dchi2 = 2.0 * (float(null.cost) - float(fit.cost))
            if dchi2 < HALO_DCHI2 or not (HALO_ALPHA[0] + 0.02 < alpha < HALO_ALPHA[1] - 0.02):
                continue
            # the halo: the fitted aureole beyond the core's edge, out to where it falls below the ring's error; inside the core (the star's
            # own residue, brighter than any ring median shows) everything above the background.  Halo light only adds, so nothing is taken
            # below the fitted background: a ring median raised by emission on one side would otherwise dig a hole on the other
            past = np.nonzero((rc >= rc[j0]) & (A * rc ** -alpha < np.nan_to_num(err, nan=np.inf)))[0]
            r_end = float(edges[past[0]]) if past.size else float(HALO_R_MAX)
            m = out[..., k]
            above = np.maximum(np.nan_to_num(m - b), 0.0)
            aureole = (A * np.maximum(r, rc[j0]) ** -alpha).astype(np.float32)
            halo = np.where(r < rc[j0], above, np.minimum(aureole, above))
            halo[r >= r_end] = 0.0
            out[..., k] = m - halo
            done.append({"star": name, "mag": round(mag, 2), "map": k, "alpha": round(alpha, 2), "radius_px": round(r_end, 1),
                         "core_excess": round(float(exc0), 4), "dchi2": round(dchi2, 1)})
    return out, done



def _whittaker(y: np.ndarray, se: np.ndarray) -> np.ndarray:
    """``y`` (with standard errors ``se``) smoothed by the Whittaker smoother - weighted least squares with a second-difference penalty
    (Eilers 2003, Anal. Chem. 75, 3631) - its smoothness chosen by the unbiased risk estimate (Mallows' Cp / UBRE: the noise is known, from
    ``se``; Craven & Wahba 1979) over a log grid: the data decide how much of the profile is structure, and noise alone gives an (almost)
    straight line."""
    y = np.asarray(y, np.float64)
    n = y.size
    if n < 6:
        return np.zeros_like(y)
    w = 1.0 / np.maximum(np.asarray(se, np.float64), 1e-30) ** 2
    D = np.diff(np.eye(n), 2, axis=0)
    P = D.T @ D
    Wd = np.diag(w)
    best = None
    from scipy.linalg import cho_factor, cho_solve
    for lam in np.logspace(-2, 10, 25) * float(np.median(w)):
        Hm = cho_solve(cho_factor(Wd + lam * P), Wd)                  # the hat matrix: z = H y (the system is symmetric positive definite)
        z = Hm @ y
        risk = float(np.sum(w * (y - z) ** 2)) + 2.0 * float(np.trace(Hm)) - n
        if best is None or risk < best[0]:
            best = (risk, z)
    return best[1]


def line_profiles(resid: np.ndarray, star_mask: np.ndarray, block: int = FACTOR) -> tuple[np.ndarray, np.ndarray, dict]:
    """Light that varies along the sensor's rows and columns in what the survey cannot explain (``resid`` (H, W, 3) = the corrected image minus
    the survey's emission): per block row (and column) the median over the sky blocks of the line (star-free, sigma-clipped at GRADIENT_CLIP),
    so structure covering less than half a line does not move it - Siril's fixbanding's estimator, at the scale of blocks.  A sensor's edge
    fall-off (no flat) and wide row bands stay fixed to the rows while the frames are registered with little rotation; a smooth gradient model
    cannot follow a ramp a few hundred pixels wide at the frame's edge.  Each profile is smoothed with weights from its lines' standard errors
    and a smoothness chosen by the unbiased risk estimate (``_whittaker``), so no smoothing scale is picked by hand and lines that differ
    from the field by their noise only are not taken for a pattern.  Returns (rows (H, 3), columns (W, 3), info)."""
    H, W = resid.shape[:2]
    b = max(2, int(block))
    hb, wb = H // b, W // b
    okb = ~star_mask[:hb * b, :wb * b].reshape(hb, b, wb, b).transpose(0, 2, 1, 3).reshape(hb, wb, b * b)
    rows, cols, amp = np.zeros((H, 3), np.float32), np.zeros((W, 3), np.float32), []
    yc, xc = (np.arange(hb) + 0.5) * b - 0.5, (np.arange(wb) + 0.5) * b - 0.5
    for c in range(3):
        x = resid[:hb * b, :wb * b, c].reshape(hb, b, wb, b).transpose(0, 2, 1, 3).reshape(hb, wb, b * b)
        D = np.nanmedian(np.where(okb, x, np.nan), axis=-1)
        D = np.where(okb.mean(-1) > 0.3, D, np.nan)
        med = float(np.nanmedian(D))
        sc = 1.4826 * float(np.nanmedian(np.abs(D - med))) + 1e-12
        D = np.where(np.abs(D - med) <= GRADIENT_CLIP * sc, D - med, np.nan)
        prof = {}
        for axis in (1, 0, 1, 0):                                      # rows, columns, then both again: each without the other's part
            other = prof.get(1 - axis)
            Dc = D if other is None else D - (other[None, :] if axis == 1 else other[:, None])
            p = np.nanmedian(Dc, axis=axis)
            n = np.isfinite(Dc).sum(axis=axis)
            se = 1.2533 * sc / np.sqrt(np.maximum(n, 1))
            p = np.where(n >= 8, np.nan_to_num(p), 0.0)
            p = _whittaker(p, se)
            prof[axis] = (p - np.median(p)).astype(np.float32)
        rows[:, c] = np.interp(np.arange(H), yc, prof[1])
        cols[:, c] = np.interp(np.arange(W), xc, prof[0])
        amp.append([float(np.ptp(prof[1])), float(np.ptp(prof[0]))])
    return rows, cols, {"row_range": [round(a[0], 7) for a in amp], "column_range": [round(a[1], 7) for a in amp], "block_px": b}


def _star_mask_simple(lum: np.ndarray) -> np.ndarray:
    """Stars and their halos: pixels more than 4 sigma above a 15 px median-filtered sky, dilated by 5 px (the mask the block medians exclude)."""
    from scipy.ndimage import median_filter
    sm = median_filter(lum.astype(np.float32), size=15)
    hp = lum - sm
    sig = 1.4826 * float(np.median(np.abs(hp - np.median(hp)))) + 1e-12
    m = (hp > 4.0 * sig).astype(np.uint8)
    return cv2.dilate(m, np.ones((11, 11), np.uint8)).astype(bool)


def sky_zero_mask(info: dict, shape) -> np.ndarray | None:
    """Pixels where the survey predicts the least emission (lowest quarter): the sky's zero point."""
    E = info.get("_emission")
    if E is None:
        return None
    ok = np.isfinite(E)
    if ok.sum() < 20:
        return None
    m = (E <= np.percentile(E[ok], 25)) & ok
    b = int(info.get("_emission_block", FACTOR))
    big = np.zeros(shape[:2], bool)
    up = np.repeat(np.repeat(m, b, 0), b, 1)[:shape[0], :shape[1]]
    big[:up.shape[0], :up.shape[1]] = up
    return big
