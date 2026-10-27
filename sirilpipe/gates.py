"""Quality gates of a finished picture (research/FINISH-SPEC.md), measured on the display image the exporter writes and printed with every export.

All values are on the final (3, H, W) image in [0, 1]; "sky" = the darkest 45 % of the 64 px smoothing of the lightness (``finish._sky_mask``: the nebula and the stars are
brighter than the sky at every large scale).  The limits are the specification's:

=======================  ============================================================================================================
sky_median               0.08 - 0.12   median lightness (luminance) of the sky; Brecher's five finished backgrounds are 0.09 - 0.11 (research/pro-workflow.md section 0)
clipped                  < 0.05 %      pixels with any channel >= 0.995 (the pros quote < 0.1 %, Siril / fas37; the specification is stricter)
white                    < 0.05 %      pixels with Lab lightness >= 0.97 outside star cores (cores = the stars' footprint, dilated)
sky_chroma               <= 2          median |a*| and |b*| (Lab) over the sky
sky_chroma_quadrants     <= 1.5        spread (max - min) of the four quadrants' median a* and of b*
sky_band_structure       <= 0.008      std of the lightness' 32-128 px band (G(10.7) - G(42.7)) over the sky, display units, stars excluded
sky_pixel_noise          <= 0.008      robust sigma of the lightness' pixel-level scatter (L - G(2)) over the sky, display units, stars excluded
stars_count              within 20 %   of the linear star layer's 5 sigma stars, the share still a 5 sigma detection at their positions against the picture's
                                       sky pixel noise (and above the CSF visibility threshold): at most 20 % lost (without the linear layer: the counts)
stars_round              <= 0.25       median ellipticity (1 - minor / major axis of the second moments) of the 300 brightest unsaturated stars
stars_coloured           >= 0.10       median HSV saturation at the stars' peaks
stars_physical           <= data + 5 % fraction of the stars measured at colour S/N >= 60 (in the linear data) whose flux colour in the picture is off the
                                       blackbody locus, at most the linear star layer's own fraction + 5 points (0.10 without the linear layer) (green above
                                       both red and blue, or below both, by more than 5 %): stars radiate nearly as blackbodies, whose sRGB colours
                                       have green between red and blue - a green or magenta star is chroma noise or a hue shift (M 31: physical
                                       renderings 3-4 %, a Bayer-drizzled stack 30 %, colour preservation forcing the colour gate 44 %)
=======================  ============================================================================================================
"""
from __future__ import annotations

import cv2
import numpy as np

from . import finish

LIMITS = {"sky_median": (0.08, 0.12), "clipped": 0.0005, "white": 0.0005, "sky_chroma": 2.0, "sky_chroma_quadrants": 1.5, "sky_band_structure": 0.008,
          "sky_pixel_noise": 0.008, "stars_count": 0.20, "stars_round": 0.25, "stars_coloured": 0.10,
          "stars_physical": 0.10}
STAR_COLOUR_SNR, LOCUS_TOL, PHYSICAL_MARGIN = 60.0, 0.05, 0.05
# a star is visible when its display peak stands above the starless picture by the CSF visibility threshold at 16 cycles / degree - TMQI.m's
# threshold for the finished image (lab.objectives.structural_fidelity): 128 / (1.4 CSF(16)) on 0-255
_CSF16 = 100.0 * 2.6 * (0.0192 + 0.114 * 16) * np.exp(-(0.114 * 16) ** 1.1)
STAR_VISIBLE = 128.0 / (1.4 * _CSF16) / 255.0
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
    lum = stars.mean(0)                                                       # colour-invariant detection, as finish.count_stars
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


def sky_luminance(lum: np.ndarray, sm: np.ndarray, starless: np.ndarray | None) -> np.ndarray:
    """The luminance the sky's own structure is measured on (sky_band_structure, sky_pixel_noise): the starless picture when the stars were screened
    on separately, else ``lum`` with its stars replaced by the local background (normalised convolution of the star-masked picture, sigma 8 px).
    The 32-128 px band of a star field is the stars' light, not the sky's blobs: on M 31 the gate read 0.011 on the finished picture, the star
    field's band-pass, and no finishing setting could bring it to its limit (0.008)."""
    if starless is not None:
        return finish.luminance(np.clip(np.nan_to_num(starless), 0.0, 1.0).astype(np.float32))
    hp = lum - cv2.GaussianBlur(lum, (0, 0), 8.0)
    nz = finish.mad_sigma(hp[sm][::7]) if sm.sum() > 100 else 0.0
    if not nz or nz <= 0:
        return lum
    keep = (cv2.dilate((hp > 5.0 * nz).astype(np.uint8), np.ones((7, 7), np.uint8)) == 0).astype(np.float32)
    bg = cv2.GaussianBlur(lum * keep, (0, 0), 8.0) / np.maximum(cv2.GaussianBlur(keep, (0, 0), 8.0), 1e-6)
    return np.where(keep > 0, lum, bg).astype(np.float32)


def off_locus(img: np.ndarray, linear_stars: np.ndarray | None = None, linear_noise: float | None = None) -> tuple[float | None, int]:
    """The fraction of the picture's stars measured at colour S/N >= STAR_COLOUR_SNR whose colour is off the blackbody locus (stars_physical), and
    how many were judged: every star detected at 8 sigma on the picture's luminance (unsaturated, compact), its colour the aperture flux of each
    channel (radius 3 px) over the local background (annulus 6-10 px) - the colour the viewer sees.  Measured on the star layer of the 500
    brightest instead, M 31 read 6 % where 44 % of the picture's stars were green or magenta."""
    import sep
    a = np.ascontiguousarray(np.moveaxis(img, 0, -1), dtype=np.float32)
    L = np.ascontiguousarray(a.mean(-1))
    if linear_stars is not None and linear_noise and linear_stars.shape[1:] == L.shape:
        # the stars judged are fixed by the data, not by the finish: the linear layer's stars whose colour the data measures at S/N >= 60 -
        # selected in the picture, a star reduction that dimmed the noisy faint stars below the cut took them out of the judged set and passed
        ys, xs = finish.star_positions(linear_stars, float(linear_noise))
        if len(ys) == 0:
            return None, 0
        x, y = xs.astype(np.float64), ys.astype(np.float64)
        fl = np.stack([sep.sum_circle(np.ascontiguousarray(linear_stars[c], dtype=np.float32), x, y, 3.0, bkgann=(6, 10), err=float(linear_noise))[:2] for c in range(3)])
        sel = (fl[:, 0].min(0) > 0) & ((fl[:, 0] / np.maximum(fl[:, 1], 1e-12)).min(0) >= STAR_COLOUR_SNR)
        x, y = x[sel], y[sel]
        if len(x) < 20:
            return None, int(len(x))
        flux = np.stack([sep.sum_circle(np.ascontiguousarray(a[..., c]), x, y, 3.0, bkgann=(6, 10))[0] for c in range(3)])
        ok = flux.min(0) > 0
    else:
        try:
            bk = sep.Background(L, bw=32, bh=32)
            objs = sep.extract(L - bk.back(), 8.0, err=bk.globalrms, minarea=4)
        except Exception:                                                    # noqa: BLE001
            return None, 0
        objs = objs[(objs["peak"] < 0.95) & (objs["a"] < 6)]
        if len(objs) == 0:
            return None, 0
        f = np.stack([sep.sum_circle(np.ascontiguousarray(a[..., c]), objs["x"], objs["y"], 3.0, bkgann=(6, 10), err=bk.globalrms)[:2] for c in range(3)])
        flux, err = f[:, 0], f[:, 1]
        ok = (flux.min(0) > 0) & ((flux / np.maximum(err, 1e-12)).min(0) >= STAR_COLOUR_SNR)
    if ok.sum() < 20:
        return None, int(ok.sum())
    r, g, b = flux[:, ok]
    off = ~((np.minimum(r, b) * (1 - LOCUS_TOL) <= g) & (g <= np.maximum(r, b) * (1 + LOCUS_TOL)))
    return round(float(off.mean()), 3), int(ok.sum())


def display_level(stars_lin: np.ndarray, stars_disp: np.ndarray, value: float) -> float | None:
    """The display value the finishing chain gave the star layer's pixels whose linear value is ``value`` (channel means; the median over the
    pixels within 10 % of it): the detection threshold of the picture's star layer, measured on the render itself.  Predicted from the star
    stretch alone it ignored the colour boost, halo suppression, star reduction and screen inversion - at a star stretch of 3 the predicted level
    counted 4.7 x the linear stars on M 31."""
    lin = np.asarray(stars_lin, np.float32).mean(0)
    disp = np.asarray(stars_disp, np.float32).mean(0)
    if lin.shape != disp.shape or value <= 0:
        return None
    sel = (lin >= 0.9 * value) & (lin <= 1.1 * value)
    if sel.sum() < 50:
        return None
    return float(np.median(disp[sel]))


def compute(img: np.ndarray, *, starless: np.ndarray | None = None, linear_star_count: int | None = None, star_level: float | None = None,
            linear_stars: np.ndarray | None = None, linear_noise: float | None = None, linear_base: np.ndarray | None = None) -> dict:
    """``linear_stars`` / ``linear_noise``: the linear star layer and the starless sky noise - then stars_count is the share of the linear layer's
    5 sigma stars the picture still shows (at their positions, above the visibility threshold), within 20 %; otherwise the count comparison."""
    """The gates of the final picture ``img`` (3, H, W).  ``starless`` = the stretched starless picture the stars were screened onto (for the star layer's measurements),
    ``linear_star_count`` = stars of the linear star layer at 5 sigma (``finish.count_stars``), ``star_level`` = the display level of a 5 sigma linear
    star after the star stretch (the detection threshold of the picture's star layer; default 5 sigma of the picture's own sky pixel noise)."""
    img = np.clip(np.nan_to_num(img), 0.0, 1.0).astype(np.float32)
    lum = finish.luminance(img)
    sm = sky_mask(lum)
    lab = _lab(img)
    out: dict = {}
    pn_sky = None

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
        sl = sky_luminance(lum, sm, starless)
        band = cv2.GaussianBlur(sl, (0, 0), finish.BAND_LO) - cv2.GaussianBlur(sl, (0, 0), finish.BAND_HI)
        bs = float(band[sm].std())
        gate("sky_band_structure", round(bs, 5), bs <= LIMITS["sky_band_structure"])
        hp = sl - cv2.GaussianBlur(sl, (0, 0), 2.0)
        pn = finish.mad_sigma(hp[sm][::7])
        pn_sky = pn
        gate("sky_pixel_noise", round(pn, 5), pn <= LIMITS["sky_pixel_noise"])
    else:
        for n in ("sky_chroma", "sky_chroma_quadrants", "sky_band_structure", "sky_pixel_noise"):
            gate(n, None, False)
    if stars is not None:
        hp = lum - cv2.GaussianBlur(lum, (0, 0), 2.0)
        noise = finish.mad_sigma(hp[sm][::7]) if sm.sum() > 100 else 0.005
        # the detection threshold: the display level of a 5 sigma linear star after the star stretch, but never below 5 sigma of the noise the
        # picture's star layer itself carries (after the denoisers and the screen inversion) - at a gentler star stretch the predicted level fell
        # under it and the count read noise peaks: 3.6 x the linear star count on M 31 (star stretch 3)
        sl_noise = finish.mad_sigma((stars.mean(0) - cv2.GaussianBlur(stars.mean(0), (0, 0), 2.0))[sm][::7]) if sm.sum() > 100 else 0.0
        ms = measure_stars(stars, max((star_level / STAR_SIGMA) if star_level else noise, sl_noise))
        if linear_stars is not None and linear_noise and stars.shape[1:] == linear_stars.shape[1:]:
            # the stars the finish keeps: every 5 sigma star of the linear layer, visible at its position in the picture's star layer.  Counting the
            # picture's peaks above a level predicted from the star stretch (or read off the render's star wings, which halo suppression darkens)
            # had counted 2 - 4.7 x the linear stars at a gentler star stretch on M 31 - the gate is about stars lost, not about extra peaks
            ys, xs = finish.star_positions(linear_stars, float(linear_noise))
            if len(ys):
                # kept = still a 5 sigma detection against the picture's own sky noise (and visible): what was detectable in the data stays
                # detectable in the picture.  The visibility threshold alone (1.3 / 255) let a star reduction of 0.6 pass that had visibly
                # taken most of M 31's faint stars
                thr = max(STAR_VISIBLE, STAR_SIGMA * float(pn_sky or 0.0))
                peak = cv2.dilate(np.ascontiguousarray(stars.mean(0)), np.ones((3, 3), np.uint8))[ys, xs]
                kept = float(np.mean(peak >= thr))
                gate("stars_count", {"kept": round(kept, 3), "linear": int(len(ys)), "deviation": round(1.0 - kept, 3)}, 1.0 - kept <= LIMITS["stars_count"])
            else:
                gate("stars_count", {"kept": None, "linear": 0}, True)
        elif linear_star_count:
            dev = abs(ms["count"] - linear_star_count) / max(linear_star_count, 1)
            gate("stars_count", {"final": ms["count"], "linear": int(linear_star_count), "deviation": round(dev, 3)}, dev <= LIMITS["stars_count"])
        else:
            gate("stars_count", {"final": ms["count"], "linear": None}, ms["count"] > 0)
        gate("stars_round", ms["ellipticity"], ms["ellipticity"] is not None and ms["ellipticity"] <= LIMITS["stars_round"])
        gate("stars_coloured", ms["saturation"], ms["saturation"] is not None and ms["saturation"] >= LIMITS["stars_coloured"])
    off, n_judged = off_locus(img, linear_stars, linear_noise)
    lim = LIMITS["stars_physical"]
    off_lin = None
    if linear_stars is not None and linear_noise:
        # no worse than the data: the calibrated linear star layer itself has stars off the locus (calibration and measurement; M 31: 11 % of
        # the stars measured at S/N >= 60), so the limit is its fraction plus 5 points - the finish must not push stars off the locus
        # measured on the linear picture (starless + stars) exactly as on the finished one: aperture colours over the local background - the
        # bare star layer (11 % on M 31) was no like-for-like reference for a measurement over the galaxy's light (the linear picture: 16 %)
        ref = linear_stars if linear_base is None or linear_base.shape != linear_stars.shape else linear_base + linear_stars
        off_lin, _ = off_locus(np.ascontiguousarray(ref, dtype=np.float32), linear_stars, linear_noise)
        if off_lin is not None:
            lim = round(off_lin + PHYSICAL_MARGIN, 3)
    gate("stars_physical", off, off is None or off <= lim, limit=lim, stars_judged=n_judged, linear=off_lin)
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
