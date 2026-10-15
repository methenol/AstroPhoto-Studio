"""Quality gates of a finished picture (research/FINISH-SPEC.md), measured on the display image the exporter writes and printed with every export.

All values are on the final (3, H, W) image in [0, 1]; "sky" = the darkest 45 % of the 64 px smoothing of the lightness (``finish._sky_mask``: the nebula and the stars are
brighter than the sky at every large scale).  The limits are the specification's:

=======================  ============================================================================================================
sky_median               0.08 - 0.12   median lightness (luminance) of the sky; Brecher's five finished backgrounds are 0.09 - 0.11 (research/pro-workflow.md section 0)
clipped                  < 0.05 %      pixels with any channel >= 0.995 (the pros quote < 0.1 %, Siril / fas37; the specification is stricter)
white                    < 0.05 %      pixels with Lab lightness >= 0.97 outside star cores (cores = the stars' footprint, dilated)
sky_chroma               <= 2          median |a*| and |b*| (Lab) over the sky
sky_chroma_quadrants     <= 1.5        spread (max - min) of the four quadrants' median a* and of b*
sky_band_structure       <= 0.008      std of the lightness' 32-128 px band (G(10.7) - G(42.7)) over the sky, display units
sky_pixel_noise          <= 0.008      robust sigma of the lightness' pixel-level scatter (L - G(2)) over the sky, display units
stars_count              within 20 %   detected stars (local maxima >= 5 sigma) of the star layer of the picture against the linear star layer's count
stars_round              <= 0.25       median ellipticity (1 - minor / major axis of the second moments) of the 300 brightest unsaturated stars
stars_coloured           >= 0.10       median HSV saturation at the stars' peaks
=======================  ============================================================================================================
"""
from __future__ import annotations

import cv2
import numpy as np

from . import finish

LIMITS = {"sky_median": (0.08, 0.12), "clipped": 0.0005, "white": 0.0005, "sky_chroma": 2.0, "sky_chroma_quadrants": 1.5, "sky_band_structure": 0.008,
          "sky_pixel_noise": 0.008, "stars_count": 0.20, "stars_round": 0.25, "stars_coloured": 0.10}
CLIP_LEVEL, WHITE_LEVEL = 0.995, 0.97
STAR_SIGMA = 5.0


def _lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(np.moveaxis(np.clip(img, 0, 1), 0, -1), dtype=np.float32), cv2.COLOR_RGB2Lab)


def sky_mask(lum: np.ndarray) -> np.ndarray:
    return finish._sky_mask(lum, edge=finish.EDGE_CROP)


def star_layer(final: np.ndarray, starless: np.ndarray | None) -> np.ndarray | None:
    """The stars of a screen blend, ``final = 1 - (1 - starless)(1 - stars)`` -> ``stars`` (None when the picture has no separate layers)."""
    if starless is None:
        return None
    return np.clip(finish.unscreen(final, starless), 0.0, 1.0)


def _peaks(plane: np.ndarray, level: float) -> np.ndarray:
    sm = cv2.GaussianBlur(np.ascontiguousarray(plane, dtype=np.float32), (0, 0), 1.0)
    dil = cv2.dilate(sm, np.ones((7, 7), np.uint8))
    return (sm == dil) & (sm > level)


def _ellipticity(win: np.ndarray) -> float:
    w = win - win.min()
    s = float(w.sum())
    if s <= 0:
        return 1.0
    yy, xx = np.mgrid[0:w.shape[0], 0:w.shape[1]].astype(np.float32)
    cx, cy = float((w * xx).sum() / s), float((w * yy).sum() / s)
    mxx, myy = float((w * (xx - cx) ** 2).sum() / s), float((w * (yy - cy) ** 2).sum() / s)
    mxy = float((w * (xx - cx) * (yy - cy)).sum() / s)
    tr, det = mxx + myy, mxx * myy - mxy ** 2
    disc = max(tr * tr / 4 - det, 0.0) ** 0.5
    l1, l2 = tr / 2 + disc, max(tr / 2 - disc, 1e-9)
    return float(1.0 - (l2 / max(l1, 1e-9)) ** 0.5)


def measure_stars(stars: np.ndarray, noise: float) -> dict:
    """Count, roundness and colour of the stars of a (3, H, W) star layer (display or linear) whose sky pixel noise is ``noise`` (same units)."""
    lum = stars.max(0)
    pk = _peaks(lum, STAR_SIGMA * max(noise, 1e-9))
    ys, xs = np.nonzero(pk)
    out = {"count": int(len(ys)), "ellipticity": None, "saturation": None}
    if len(ys) == 0:
        return out
    h, w = lum.shape
    vals = lum[ys, xs]
    order = np.argsort(-vals)
    ell, sat = [], []
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if vals[i] >= 0.98 or y < 10 or x < 10 or y >= h - 10 or x >= w - 10:
            continue
        if len(ell) < 300:
            ell.append(_ellipticity(lum[y - 10:y + 11, x - 10:x + 11]))
        if len(sat) < 300:
            px = stars[:, y - 4:y + 5, x - 4:x + 5].sum(axis=(1, 2))          # flux colour of the star (a clipped core is white, its wings carry the colour)
            sat.append(float((px.max() - px.min()) / max(px.max(), 1e-9)))
        if len(ell) >= 300 and len(sat) >= 300:
            break
    out["ellipticity"] = round(float(np.median(ell)), 3) if ell else None
    out["saturation"] = round(float(np.median(sat)), 3) if sat else None
    return out


def compute(img: np.ndarray, *, starless: np.ndarray | None = None, linear_star_count: int | None = None, star_level: float | None = None) -> dict:
    """The gates of the final picture ``img`` (3, H, W).  ``starless`` = the stretched starless picture the stars were screened onto (for the star layer's measurements),
    ``linear_star_count`` = stars of the linear star layer at 5 sigma (``finish.count_stars``), ``star_level`` = the display level of a 5 sigma linear
    star after the star stretch (the detection threshold of the picture's star layer; default 5 sigma of the picture's own sky pixel noise)."""
    img = np.clip(np.nan_to_num(img), 0.0, 1.0).astype(np.float32)
    lum = finish.luminance(img)
    sm = sky_mask(lum)
    lab = _lab(img)
    out: dict = {}

    def gate(name, value, ok, limit=None, **extra):
        out[name] = {"value": value, "limit": LIMITS.get(name) if limit is None else limit, "pass": bool(ok), **extra}

    sky = float(np.median(lum[sm])) if sm.sum() > 100 else float(np.median(lum))
    lo, hi = LIMITS["sky_median"]
    gate("sky_median", round(sky, 4), lo <= sky <= hi)
    clipped = float((img.max(0) >= CLIP_LEVEL).mean())
    gate("clipped", round(clipped, 6), clipped < LIMITS["clipped"])
    # stars: layer of the picture, footprint of the star cores
    stars = star_layer(img, starless)
    cores = np.zeros(lum.shape, bool)
    if stars is not None:
        cores = cv2.dilate((stars.max(0) > 0.05).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    light = lab[..., 0] / 100.0
    white = float(((light >= WHITE_LEVEL) & ~cores).mean())
    gate("white", round(white, 6), white < LIMITS["white"])
    if sm.sum() > 100:
        a, b = lab[..., 1], lab[..., 2]
        ca, cb = float(np.median(np.abs(a[sm]))), float(np.median(np.abs(b[sm])))
        gate("sky_chroma", {"a": round(ca, 2), "b": round(cb, 2)}, max(ca, cb) <= LIMITS["sky_chroma"])
        h, w = lum.shape
        quads = [(slice(0, h // 2), slice(0, w // 2)), (slice(0, h // 2), slice(w // 2, w)), (slice(h // 2, h), slice(0, w // 2)), (slice(h // 2, h), slice(w // 2, w))]
        spread = {"a": [], "b": []}
        for ys, xs in quads:
            m = sm[ys, xs]
            if m.sum() > 100:
                spread["a"].append(float(np.median(a[ys, xs][m])))
                spread["b"].append(float(np.median(b[ys, xs][m])))
        sa = (max(spread["a"]) - min(spread["a"])) if spread["a"] else 0.0
        sb = (max(spread["b"]) - min(spread["b"])) if spread["b"] else 0.0
        gate("sky_chroma_quadrants", {"a": round(sa, 2), "b": round(sb, 2)}, max(sa, sb) <= LIMITS["sky_chroma_quadrants"])
        band = cv2.GaussianBlur(lum, (0, 0), finish.BAND_LO) - cv2.GaussianBlur(lum, (0, 0), finish.BAND_HI)
        bs = float(band[sm].std())
        gate("sky_band_structure", round(bs, 5), bs <= LIMITS["sky_band_structure"])
        hp = lum - cv2.GaussianBlur(lum, (0, 0), 2.0)
        pn = finish.mad_sigma(hp[sm][::7])
        gate("sky_pixel_noise", round(pn, 5), pn <= LIMITS["sky_pixel_noise"])
    else:
        for n in ("sky_chroma", "sky_chroma_quadrants", "sky_band_structure", "sky_pixel_noise"):
            gate(n, None, False)
    if stars is not None:
        hp = lum - cv2.GaussianBlur(lum, (0, 0), 2.0)
        noise = finish.mad_sigma(hp[sm][::7]) if sm.sum() > 100 else 0.005
        ms = measure_stars(stars, (star_level / STAR_SIGMA) if star_level else noise)
        if linear_star_count:
            dev = abs(ms["count"] - linear_star_count) / max(linear_star_count, 1)
            gate("stars_count", {"final": ms["count"], "linear": int(linear_star_count), "deviation": round(dev, 3)}, dev <= LIMITS["stars_count"])
        else:
            gate("stars_count", {"final": ms["count"], "linear": None}, ms["count"] > 0)
        gate("stars_round", ms["ellipticity"], ms["ellipticity"] is not None and ms["ellipticity"] <= LIMITS["stars_round"])
        gate("stars_coloured", ms["saturation"], ms["saturation"] is not None and ms["saturation"] >= LIMITS["stars_coloured"])
    out["passed"] = all(v["pass"] for k, v in out.items() if isinstance(v, dict))
    out["failed"] = [k for k, v in out.items() if isinstance(v, dict) and not v["pass"]]
    return out


def format_gates(g: dict) -> str:
    """The table the exporter prints."""
    rows = []
    for k, v in g.items():
        if not isinstance(v, dict):
            continue
        lim = v.get("limit")
        rows.append(f"  {'PASS' if v['pass'] else 'FAIL'}  {k:<22} {v['value']}   (limit {lim})")
    return "gates:\n" + "\n".join(rows) + f"\n  => {'all gates pass' if g.get('passed') else 'FAILED: ' + ', '.join(g.get('failed', []))}"
