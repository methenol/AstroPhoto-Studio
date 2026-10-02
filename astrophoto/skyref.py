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
NAME = "NSNS DR0.2"
CREDIT = ("NSNS DR0.2 (Northern Sky Narrowband Survey, simg.de; CC BY-NC-SA 4.0, doi:10.3847/2515-5172/adfec7), "
          "via CDS hips2fits")
HIPS2FITS = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
FACTOR = 8               # reference pixels: FACTOR x FACTOR stack pixels (~20" for a DWARF / Seestar)


def reduced_wcs(wcs, factor: int):
    """The stack's TAN projection on a grid ``factor`` times coarser (SIP dropped: sub-pixel there)."""
    w = wcs.deepcopy()
    w.sip = None
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [(wcs.wcs.crpix[0] - 0.5) / factor + 0.5, (wcs.wcs.crpix[1] - 0.5) / factor + 0.5]
    w.wcs.pc = wcs.wcs.get_pc() * factor
    w.wcs.cdelt = [1.0, 1.0]
    return w


def fetch(wcs, shape, factor: int = FACTOR, timeout: float = 120.0) -> np.ndarray:
    """The survey's maps on the stack's grid reduced by ``factor``: (h, w, len(SURVEYS)) float32, NaN
    outside the survey's coverage.  Raises on a network failure."""
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


def fit(img: np.ndarray, refs: np.ndarray, px_per_ref: float, origin=(0, 0), degree: int = 3,
        smooth: float = 1.5, min_coverage: float = 0.6, star_mask=None,
        clip_high: float = 2.5) -> tuple[np.ndarray, dict]:
    """Gradient model of ``img`` (h, w, 3) against the survey maps ``refs`` (on a grid of
    ``px_per_ref`` image pixels per reference pixel, ``origin``: the image's (y, x) offset on the
    grid the references were made for, e.g. a crop).  Returns (model on the image grid, info);
    raises ValueError when too little of the field is covered by usable reference data."""
    from .postprocess import luminance, mad_sigma, star_mask_simple
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
    R = np.stack([_nan_blur(R[..., k], smooth) for k in range(R.shape[-1])], -1)
    D = np.stack([_nan_blur(D[..., c], smooth) for c in range(3)], -1)
    usable = (okb.mean(-1) > 0.3) & np.isfinite(R).all(-1) & np.isfinite(D).all(-1)
    cover = float(usable.mean())
    if cover < min_coverage:
        raise ValueError(f"only {cover * 100:.0f} % of the field has survey data ({NAME} covers declination "
                         "-16 to +76 degrees)")
    # columns: the survey maps (scaled to unit spread for conditioning) and the polynomial
    sd = np.array([np.std(R[..., k][usable]) or 1.0 for k in range(R.shape[-1])])
    yy, xx = np.mgrid[0:hb, 0:wb]
    X, Y = (xx + 0.5) / wb * 2 - 1, (yy + 0.5) / hb * 2 - 1
    terms = [(a, c) for a in range(degree + 1) for c in range(degree + 1 - a)]
    P = np.stack([X ** a * Y ** c for a, c in terms], -1)
    A = np.concatenate([R / sd, P], -1)[usable]
    nk = R.shape[-1]
    coefs, polys, scatter = [], [], []
    for c in range(3):
        y = D[..., c][usable]
        wgt = np.ones(len(y))
        for _ in range(15):
            beta, *_ = np.linalg.lstsq(A * wgt[:, None], y * wgt, rcond=None)
            r = y - A @ beta
            s = 1.4826 * float(np.median(np.abs(r))) + 1e-12
            # Huber (1.5 robust sigma), and a lower envelope: light well above survey + gradient is
            # object light the survey lacks (a bright star's halo - its maps are star-subtracted -, a
            # core it resolves less well), never gradient; left in, the polynomial bent under it (NGC 281:
            # +20 % of the nebula's signal taken with a bright star's halo at the field's edge)
            wgt = np.sqrt(np.minimum(1.0, 1.5 * s / np.maximum(np.abs(r), 1e-12)))
            wgt[r > clip_high * s] = 0.0
        coefs.append(beta[:nk] / sd)
        polys.append(beta[nk:])
        scatter.append(s)
    # the model on the image grid: evaluated coarsely, then smoothly resized (as background_model)
    lh, lw = max(8, h // 32), max(8, w // 32)
    gy, gx = np.mgrid[0:lh, 0:lw]
    GX, GY = (gx + 0.5) / lw * 2 - 1, (gy + 0.5) / lh * 2 - 1
    PG = np.stack([GX ** a * GY ** c for a, c in terms], -1)
    low = np.stack([PG @ polys[c] for c in range(3)], -1).astype(np.float32)
    bg = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    # the emission the survey predicts per block (luminance), for the sky's zero point
    E = np.stack([np.nan_to_num(R) @ coefs[c] for c in range(3)], -1) @ np.array([0.2126, 0.7152, 0.0722])
    E = np.where(usable, E, np.nan).astype(np.float32)
    noise = mad_sigma((luminance(img) - cv2.GaussianBlur(luminance(img), (0, 0), 1.5))[::3, ::3])
    info = {"method": f"reference ({NAME})", "degree": int(degree), "coverage": round(cover, 3),
            "survey_coefficients": {name: [float(coefs[c][k]) for c in range(3)] for k, (name, _) in enumerate(SURVEYS)},
            "gradient_range": float(np.ptp(low @ np.array([0.2126, 0.7152, 0.0722])) / max(noise, 1e-12)),
            "fit_scatter": [float(s) for s in scatter], "credit": CREDIT,
            "_emission": E, "_emission_block": b}
    return bg, info


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
