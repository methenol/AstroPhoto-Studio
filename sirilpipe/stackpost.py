"""Numpy helpers applied to the Siril stack: coverage-based auto-crop and the HDR blend of exposure groups.

No synthetic light anywhere: the crop only removes rows/columns, the HDR blend only combines measured values
(inverse-variance weights; saturated pixels get zero weight).
"""
from __future__ import annotations

import math

import cv2
import numpy as np


# ======================================================================= coverage / crop
CONVENTIONS = ("FHF", "inv", "H", "FinvF")      # tried in this order; the best IoU against the data wins


def frame_coverage(H_list: list[list[float]], width: int, height: int, scale: float, canvas: tuple[int, int],
                   convention: str = "H", down: int = 4,
                   origin: tuple[float, float] | None = None) -> tuple[np.ndarray, tuple[float, float]]:
    """How many registered frames cover each (down-sampled) pixel of the -framing=max canvas.

    ``H_list``: 3x3 homographies (row-major, Siril's image -> reference); ``width/height``: sub size in pixels;
    ``canvas``: (width, height) of the stack.  The canvas origin is the minimum corner of all projected frames
    (what ``-framing=max`` + ``-maximize`` builds).  ``convention``: ``"FHF"`` = Siril's stored homography acts on
    y-flipped (top-down) pixel coordinates, i.e. ``F H F`` with ``F: y -> height - y`` (this reproduced the canvas of
    a rotating Seestar stack to 0.002 IoU and 0.3 px extent); ``"H"``, ``"inv"``, ``"FinvF"`` are the alternatives
    the callers also try, since Siril's orientation is not documented."""
    corners = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]], dtype=np.float64).T
    F = np.array([[1, 0, 0], [0, -1, height], [0, 0, 1]], dtype=np.float64)
    polys = []
    for h in H_list:
        M = np.asarray(h, dtype=np.float64).reshape(3, 3)
        if convention == "inv":
            M = np.linalg.inv(M)
        elif convention == "FHF":
            M = F @ M @ F
        elif convention == "FinvF":
            M = F @ np.linalg.inv(M) @ F
        p = M @ corners
        p = (p[:2] / p[2]) * scale
        polys.append(p.T)
    allp = np.concatenate(polys)
    x0, y0 = allp.min(axis=0) if origin is None else origin
    cw, ch = canvas
    cov = np.zeros((max(1, math.ceil(ch / down)), max(1, math.ceil(cw / down))), np.uint16)
    tmp = np.zeros_like(cov, dtype=np.uint8)
    for poly in polys:
        pts = np.round((poly - [x0, y0]) / down).astype(np.int32)
        tmp[:] = 0
        cv2.fillConvexPoly(tmp, pts, 1)
        cov += tmp
    return cov, (float(x0), float(y0))


def largest_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Largest axis-aligned all-True rectangle of a boolean mask: (x0, y0, x1, y1) with exclusive ends.
    Classic histogram/stack algorithm, O(H*W)."""
    h, w = mask.shape
    heights = np.zeros(w, np.int32)
    best = (0, 0, 0, 0, 0)               # area, x0, y0, x1, y1
    for y in range(h):
        heights = np.where(mask[y], heights + 1, 0)
        stack: list[tuple[int, int]] = []        # (start index, height)
        for x in range(w + 1):
            cur = int(heights[x]) if x < w else 0
            start = x
            while stack and stack[-1][1] >= cur:
                s, hh = stack.pop()
                area = hh * (x - s)
                if area > best[0]:
                    best = (area, s, y - hh + 1, x, y + 1)
                start = s
            stack.append((start, cur))
    return best[1], best[2], best[3], best[4]


def crop_box(valid: np.ndarray, coverage: np.ndarray | None, threshold: float, n_frames: int,
             down: int = 4, margin: int = 2) -> tuple[int, int, int, int]:
    """The crop rectangle (x, y, w, h) in stack pixels: the largest rectangle inside the region covered by at
    least ``threshold`` x ``n_frames`` frames (when ``coverage`` is given) and containing valid data only,
    shrunk by ``margin`` px (interpolation/drizzle edge)."""
    H, W = valid.shape
    small_valid = cv2.resize(valid.astype(np.uint8), (math.ceil(W / down), math.ceil(H / down)),
                             interpolation=cv2.INTER_AREA) >= 1
    # INTER_AREA gives < 1 for blocks touching invalid pixels: keep only fully valid blocks
    m = small_valid
    if coverage is not None:
        c = coverage
        if c.shape != m.shape:
            c = cv2.resize(c.astype(np.float32), (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = m & (c >= max(1.0, threshold * n_frames))
    x0, y0, x1, y1 = largest_rectangle(m)
    x0, y0, x1, y1 = x0 * down, y0 * down, min(W, x1 * down), min(H, y1 * down)
    # refine at full resolution: shrink until the border rows/columns are all valid
    while y1 > y0 and not valid[y0, x0:x1].all():
        y0 += 1
    while y1 > y0 and not valid[y1 - 1, x0:x1].all():
        y1 -= 1
    while x1 > x0 and not valid[y0:y1, x0].all():
        x0 += 1
    while x1 > x0 and not valid[y0:y1, x1 - 1].all():
        x1 -= 1
    x0, y0, x1, y1 = x0 + margin, y0 + margin, x1 - margin, y1 - margin
    if x1 - x0 < 16 or y1 - y0 < 16:
        return 0, 0, W, H
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def valid_mask(stack: np.ndarray) -> np.ndarray:
    """Pixels with data in every channel (the -maximize canvas is exactly 0 outside the frames)."""
    return (stack > 0).all(axis=0) & np.isfinite(stack).all(axis=0)


def auto_crop(stack: np.ndarray, hmats: list[list[float]] | None, sub_size: tuple[int, int], scale: float,
              threshold: float, group_hmats: list[list[list[float]]] | None = None,
              footprint: np.ndarray | None = None) -> tuple[tuple[int, int, int, int], dict]:
    """Crop box for a -framing=max stack plus a report of how the coverage was obtained.

    Coverage by geometry (projected frame outlines, validated against the data mask with both homography
    conventions; IoU >= 0.95 required); otherwise the data mask alone (any frame covers = threshold ignored,
    reported).  ``footprint``: HDR, the area every exposure group has data in (the crop stays inside it, whichever
    way the coverage was obtained)."""
    valid = valid_mask(stack)
    H, W = valid.shape
    geo_valid = valid
    if footprint is not None and footprint.shape == valid.shape:
        valid = valid & footprint
    info: dict = {"method": "data mask"}
    cov = None
    if hmats:
        best = (0.0, None, None)
        for conv in CONVENTIONS:
            try:
                c, _ = frame_coverage(hmats, sub_size[0], sub_size[1], scale, (W, H), conv)
            except Exception:
                continue
            vs = cv2.resize(geo_valid.astype(np.uint8), (c.shape[1], c.shape[0]), interpolation=cv2.INTER_AREA) > 0
            inter = float(((c > 0) & vs).sum())
            union = float(((c > 0) | vs).sum())
            iou = inter / union if union else 0.0
            if iou > best[0]:
                best = (iou, conv, c)
        if best[1] and best[0] >= 0.95:
            cov = best[2]
            info = {"method": "frame coverage", "convention": best[1], "iou": round(best[0], 4),
                    "threshold": threshold}
        else:
            info["note"] = f"projected frame outlines did not match the data (IoU {best[0]:.2f}); data mask used"
    n_ref = len(hmats) if hmats else 1
    if cov is not None and group_hmats and len(group_hmats) > 1:
        # HDR: EVERY exposure group must cover the kept area (else a single-group strip shows as a seam):
        # coverage = the smallest per-group fraction of its frames, threshold applied to that fraction
        _, origin = frame_coverage(hmats, sub_size[0], sub_size[1], scale, (W, H), info["convention"])
        frac = None
        for hg in group_hmats:
            cg, _ = frame_coverage(hg, sub_size[0], sub_size[1], scale, (W, H), info["convention"], origin=origin)
            f = cg.astype(np.float32) / max(len(hg), 1)
            frac = f if frac is None else np.minimum(frac, f)
        cov, n_ref = frac, 1
        info["groups_all_cover"] = True
    box = crop_box(valid, cov, threshold, n_ref)
    return box, info


# ======================================================================= HDR blend
def noise_model(plane: np.ndarray, bins: int = 24) -> tuple[float, float]:
    """Poisson-Gaussian noise level function ``var(v) = a + b * v`` of one stack plane, measured from the image
    itself: the residual of a small Gaussian blur, robust (MAD) variance per intensity bin, weighted linear fit.
    (HDR_Blender builds the same kind of model from Siril's bgnoise + median; the stack's own residuals avoid
    needing the sensor gain.)  ``a`` is floored (HDR_Blender's VAR_FLOOR idea)."""
    g = np.ascontiguousarray(plane[::2, ::2] if min(plane.shape) > 1000 else plane, dtype=np.float32)
    sm = cv2.GaussianBlur(g, (0, 0), 1.5)
    res = (g - sm).ravel()
    lv = sm.ravel()
    ok = lv > 0
    res, lv = res[ok], lv[ok]
    if res.size < 1000:
        return 1e-10, 0.0
    qs = np.quantile(lv, np.linspace(0.0, 0.97, bins + 1))
    xs, vs, ws = [], [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        sel = (lv >= lo) & (lv < hi)
        if sel.sum() < 200:
            continue
        r = res[sel]
        mad = np.median(np.abs(r - np.median(r))) * 1.4826
        xs.append(float(np.median(lv[sel])))
        vs.append(float(mad) ** 2)
        ws.append(float(sel.sum()))
    if len(xs) < 3:
        return (float(np.median(vs)) if vs else 1e-10), 0.0
    A = np.vstack([np.ones(len(xs)), xs]).T
    W = np.sqrt(np.array(ws))
    sol, *_ = np.linalg.lstsq(A * W[:, None], np.array(vs) * W, rcond=None)
    return max(float(sol[0]), 1e-12), float(max(sol[1], 0.0))


def saturation_level(plane: np.ndarray, valid: np.ndarray | None = None, min_pixels: int = 20) -> float | None:
    """Level of the clipped plateau of ONE stack plane (None when nothing is clipped).

    A stack of many clipped subs does not clip at the ADC full scale: every sub is normalised (``-norm=addscale``,
    different per group and per channel: 0.98-1.15 of the full scale on DWARF 3) and divided by the flat, so the
    plateau sits at the mean clip level of the subs, found here as the spike of the histogram's top end: the
    highest histogram bin (0.5 % of the maximum wide) that holds ``min_pixels`` or more pixels and far more than the
    continuum below it; the plateau is the median of the pixels around that spike."""
    v = plane[valid] if valid is not None else plane.ravel()
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < min_pixels:
        return None
    top = float(np.quantile(v, 0.99999)) if v.size > 100000 else float(v.max())
    lo = 0.4 * top
    v = v[(v > lo) & (v <= top * 1.001)]
    if v.size < min_pixels:
        return None
    width = top / 200.0
    edges = np.arange(lo, top * 1.001 + width, width)
    h, _ = np.histogram(v, bins=edges)
    cont = max(1.0, float(np.median(h[: max(3, len(h) // 2)])))
    cand = np.nonzero((h >= min_pixels) & (h >= 6.0 * cont))[0]
    if cand.size == 0:
        return None
    k = int(cand.max())
    # climb to the local maximum of the spike (the neighbouring bins may hold the bulk of the plateau)
    while k + 1 < len(h) and h[k + 1] > h[k]:
        k += 1
    while k > 0 and h[k - 1] > h[k]:
        k -= 1
    sel = v[(v >= edges[max(k - 1, 0)]) & (v < edges[min(k + 2, len(edges) - 1)])]
    return float(np.median(sel)) if sel.size >= min_pixels else None


def star_gain(ref: np.ndarray, img: np.ndarray, valid: np.ndarray, sat_ref: float | None = None,
              sat_img: float | None = None, radius: float = 8.0, min_stars: int = 8) -> tuple[float, int] | None:
    """Gain ``g`` (``img ~ g * ref``) from aperture photometry of stars unsaturated in BOTH radiance planes:
    the median of the flux ratios ``flux_img / flux_ref`` (flux = sum within ``radius`` minus the median of the
    annulus around it), None when fewer than ``min_stars`` stars qualify.

    Pixel ratios of blurred images are biased by the groups' different PSFs (night 1 vs night 2 seeing); the flux
    inside an aperture of ~3 FWHM is PSF independent.  ``sat_*`` are the planes' clip levels (same units as the
    planes): a star whose aperture reaches 0.8 of either is skipped."""
    from scipy import ndimage as ndi
    r = int(math.ceil(radius))
    ann_in, ann_out = radius + 2.0, radius + 4.5
    pad = int(math.ceil(ann_out)) + 1
    sm = cv2.GaussianBlur(np.ascontiguousarray(ref, np.float32), (0, 0), 1.5)
    inner = ndi.binary_erosion(valid, iterations=pad + 2)
    if not inner.any():
        return None
    sky = float(np.median(sm[valid][::7]))
    sig = 1.4826 * float(np.median(np.abs(sm[valid][::7] - sky)))
    top = 0.8 * sat_ref if sat_ref else float(np.quantile(sm[inner], 0.9999))
    mx = ndi.maximum_filter(sm, size=2 * pad + 1)
    pk = (sm == mx) & inner & (sm > sky + 30.0 * sig) & (sm < top)
    ys, xs = np.nonzero(pk)
    if ys.size < min_stars:
        return None
    if ys.size > 600:
        keep = np.argsort(-sm[ys, xs])[:600]
        ys, xs = ys[keep], xs[keep]
    yy, xx = np.mgrid[-pad:pad + 1, -pad:pad + 1]
    rr = np.hypot(yy, xx)
    ap, an = rr <= radius, (rr >= ann_in) & (rr <= ann_out)
    ratios = []
    for y, x in zip(ys, xs):
        wr = ref[y - pad:y + pad + 1, x - pad:x + pad + 1]
        wi = img[y - pad:y + pad + 1, x - pad:x + pad + 1]
        if wr[ap].max() >= (top if sat_ref else np.inf) or (sat_img and wi[ap].max() >= 0.8 * sat_img):
            continue
        fr = float(wr[ap].sum() - ap.sum() * np.median(wr[an]))
        fi = float(wi[ap].sum() - ap.sum() * np.median(wi[an]))
        if fr > 0:
            ratios.append(fi / fr)
    if len(ratios) < min_stars:
        return None
    return float(np.median(ratios)), len(ratios)


def calibrate_to_reference(ref: np.ndarray, img: np.ndarray, valid: np.ndarray, sat_ref: float | None = None,
                           sat_img: float | None = None, radius: float = 8.0,
                           n_sigma: float = 8.0) -> tuple[float, float, int]:
    """Gain ``a`` and offset ``b`` with ``img ~ a * ref + b`` over the pixels both radiance planes cover (HDR_Blender's
    optional "calibrate each exposure against the longest": it removes scale errors, e.g. a different flat,
    normalisation or transparency, and pedestal offsets, e.g. a dark of another temperature, which would otherwise
    show as a brightness step wherever only one exposure group has data, and as ringed star cores where the
    groups are exchanged).

    Gain: aperture photometry of the stars unsaturated in both planes (``star_gain``); without enough stars, the
    median of ``(img - sky_img) / (ref - sky_ref)`` over lightly blurred pixels standing ``n_sigma`` noise-widths
    above the sky in ``ref`` and below 60 % of ``sat_ref`` (a plain least-squares line is useless: the sky occupies a
    few noise-widths and the regression dilutes the slope to 0.4-0.7).  Offset ``b``: makes the sky levels (medians
    of the overlap; the field is mostly sky) agree.  Returns (a, b, n_pixels); (1, sky difference, n) when no
    gain can be measured, (1, 0, n) when the overlap is too small."""
    n = int(valid.sum())
    if n < 20000:
        return 1.0, 0.0, n
    r = cv2.GaussianBlur(np.ascontiguousarray(ref, np.float32), (0, 0), 2.0)[valid]
    m = cv2.GaussianBlur(np.ascontiguousarray(img, np.float32), (0, 0), 2.0)[valid]
    r0, m0 = float(np.median(r)), float(np.median(m))
    sg = star_gain(ref, img, valid, sat_ref, sat_img, radius)
    if sg is not None and 0.5 < sg[0] < 2.0:
        return sg[0], m0 - sg[0] * r0, n
    sig = 1.4826 * float(np.median(np.abs(r[r <= np.quantile(r, 0.6)] - r0)))
    top = 0.6 * sat_ref if sat_ref else float(np.quantile(r, 0.9999))
    sel = (r > r0 + n_sigma * sig) & (r < top)
    if int(sel.sum()) < 1000:
        return 1.0, m0 - r0, n
    g = float(np.median((m[sel] - m0) / (r[sel] - r0)))
    if not (0.5 < g < 2.0):                           # nonsense (structure mismatch): do not apply
        return 1.0, m0 - r0, n
    return g, m0 - g * r0, n


SAT_TAPER = (0.85, 0.97)      # weight tapers from 0.85 to 0.97 of the measured clip level (zero above)


def hdr_blend(stacks: list[np.ndarray], exptimes: list[float], valids: list[np.ndarray],
              calibrate: bool = True, radius: float = 8.0, saturation: list | None = None) -> tuple[np.ndarray, dict]:
    """Inverse-variance MLE blend of registered stacks of different exposures (A. Knagg-Baugh's HDR_Blender.py,
    workflows.md 9.2), per channel, all in the calibrated radiance (ADU/s) of the longest exposure:

    * radiance = stack / exposure; every shorter exposure is calibrated to the longest (gain from star photometry,
      offset from the sky, ``calibrate_to_reference``);
    * weight = taper / variance, the variance from the Poisson-Gaussian noise model of each stack
      (``noise_model``), 0 outside the stack's footprint (``valids``);
    * taper = cosine, 1 -> 0 between 0.85 and 0.97 of the group's MEASURED clip level (``saturation_level``, per
      group and channel: the normalised stacks clip at 0.98-1.15 of the ADC full scale, and a stack of clipped subs is
      already non-linear below its plateau).  A longer exposure also reads the shorter ones as witnesses: wherever
      a shorter exposure measures a radiance that would clip the longer one, the longer one is out (its stacked
      value is a partial clip, not a measurement), however low its own value looks;
    * pixels with no weight anywhere (clipped in every input) take the shortest exposure's measured radiance (the
      HDR_Blender rule; a lower bound, nothing invented).
    The result is scaled back to the longest exposure's units and divided by its maximum, so it stays in [0, 1].
    ``saturation``: per group and channel clip levels measured BEFORE any resampling or PSF matching (which smear the
    plateau); measured here when omitted.  Every value comes from the inputs."""
    shape = stacks[0].shape
    n_ch = shape[0]
    report: dict = {"inputs": []}
    order = list(np.argsort(exptimes, kind="stable"))              # shortest first
    ref = int(np.argmax(exptimes))
    both = [np.logical_and(valids[ref], v) for v in valids]
    rad = [np.zeros(shape, np.float32) for _ in stacks]
    sat_rad = [[None] * n_ch for _ in stacks]                       # clip level in calibrated radiance
    for j, (s, t, vm) in enumerate(zip(stacks, exptimes, valids)):
        info = {"exptime": float(t), "noise": [], "saturation": [], "gain_offset": []}
        for c in range(n_ch):
            sat = saturation[j][c] if saturation else saturation_level(s[c], vm)
            info["saturation"].append(sat)
            r = (s[c] / t).astype(np.float32)
            g, o = 1.0, 0.0
            if calibrate and j != ref:
                sat_ref = saturation[ref][c] if saturation else saturation_level(stacks[ref][c], valids[ref])
                g, o, npx = calibrate_to_reference(stacks[ref][c] / exptimes[ref], r, both[j],
                                                   (sat_ref / exptimes[ref]) if sat_ref else None,
                                                   (sat / t) if sat else None, radius)
                r = ((r - o) / g).astype(np.float32)
                info["gain_offset"].append([g, o, npx])
            rad[j][c] = r
            sat_rad[j][c] = ((sat / t - o) / g) if sat else None
        report["inputs"].append(info)
    acc = np.zeros(shape, np.float32)
    wsum = np.zeros(shape, np.float32)
    for j in order:                                                  # shortest first: its radiance witnesses for the longer ones
        t, vm = exptimes[j], valids[j]
        shorter = [k for k in order if exptimes[k] < t]
        for c in range(n_ch):
            plane = stacks[j][c]
            a, b = noise_model(np.where(vm, plane, 0.0))
            var = np.maximum((a + b * np.clip(plane, 0, None)) / (t * t), 1e-18)
            w = np.where(vm, 1.0 / var, 0.0).astype(np.float32)
            sr = sat_rad[j][c]
            if sr is not None:
                x = rad[j][c]
                for k in shorter:
                    x = np.where(valids[k], np.maximum(x, rad[k][c]), x)
                lo, hi = sr * SAT_TAPER[0], sr * SAT_TAPER[1]
                u = np.clip((x - lo) / max(hi - lo, 1e-30), 0.0, 1.0)
                w *= (0.5 * (1.0 + np.cos(np.pi * u))).astype(np.float32)
            acc[c] += w * rad[j][c]
            wsum[c] += w
            report["inputs"][j]["noise"].append([a, b])
    blend = np.where(wsum > 0, acc / np.maximum(wsum, 1e-30), 0.0).astype(np.float32)
    any_valid = np.any(np.stack(valids), axis=0)
    none = (wsum <= 0) & any_valid[None]
    if none.any():
        for c in range(n_ch):
            fill = np.zeros(shape[1:], np.float32)
            for j in reversed(order):                               # the shortest valid exposure wins
                fill = np.where(valids[j], rad[j][c], fill)
            blend[c][none[c]] = fill[none[c]]
        report["unweighted_pixels"] = int(none[0].sum())
    t_long = float(max(exptimes))
    out = blend * t_long
    top = float(out.max())
    if top > 0:
        out /= top
    report["scale_to_exposure"] = t_long
    report["divided_by"] = top
    return out.astype(np.float32), report


def equalize_backgrounds(img: np.ndarray, valid: np.ndarray) -> list[float]:
    """Scale R and B so that the channel background medians equal the green one (Siril's ``-rgb_equal`` idea:
    colour balance of the sky background before any photometric calibration).  Returns the factors."""
    meds = [float(np.median(img[c][valid])) for c in range(img.shape[0])]
    ref = meds[1] if img.shape[0] == 3 else meds[0]
    f = [ref / m if m > 0 else 1.0 for m in meds]
    for c in range(img.shape[0]):
        img[c] *= f[c]
    return f


def group_offset(stack: np.ndarray, hmats_group: list, hmats_all: list, sub_size: tuple[int, int], scale: float,
                 convention: str | None = None) -> tuple[int, int, str]:
    """Pixel offset (dx, dy) of an exposure-group stack's canvas inside the canvas of all frames, from the projected
    frame outlines (origin = minimum corner), with the homography convention validated on the stack's own data."""
    valid = valid_mask(stack)
    H, W = valid.shape
    best = (-1.0, convention)
    if convention is None:
        for conv in CONVENTIONS:
            c, _ = frame_coverage(hmats_group, sub_size[0], sub_size[1], scale, (W, H), conv)
            vs = cv2.resize(valid.astype(np.uint8), (c.shape[1], c.shape[0]), interpolation=cv2.INTER_AREA) > 0
            iou = float(((c > 0) & vs).sum()) / max(1.0, float(((c > 0) | vs).sum()))
            if iou > best[0]:
                best = (iou, conv)
        convention = best[1]
    _, (gx, gy) = frame_coverage(hmats_group, sub_size[0], sub_size[1], scale, (W, H), convention)
    _, (ax, ay) = frame_coverage(hmats_all, sub_size[0], sub_size[1], scale, (W, H), convention)
    return int(round(gx - ax)), int(round(gy - ay)), convention


def refine_shift(ref: np.ndarray, img: np.ndarray, max_shift: int = 3, win: int = 1024) -> tuple[int, int]:
    """Integer correction (dx, dy) to ADD to the placement offset of ``img`` so that it matches ``ref`` (both on the
    same canvas): exhaustive search of the normalised cross-correlation of a central window of the green planes over
    +-``max_shift`` px.  Drizzled stacks of one registration share a pixel grid, so the true residual is an integer;
    a search beats phase correlation here because the peak is exactly on the grid.  (0, 0) when the overlap is too
    small or the best shift is not clearly better than no shift."""
    a = np.asarray(ref[1], np.float32)
    b = np.asarray(img[1], np.float32)
    common = (a > 0) & (b > 0)
    ys, xs = np.nonzero(common[::8, ::8])
    if ys.size < 500:
        return 0, 0
    cy, cx = int(np.median(ys)) * 8, int(np.median(xs)) * 8
    m = max_shift
    h = min(win, a.shape[0] - 2 * m - 2, b.shape[0] - 2 * m - 2)
    w = min(win, a.shape[1] - 2 * m - 2)
    y0 = int(np.clip(cy - h // 2, m + 1, a.shape[0] - h - m - 1))
    x0 = int(np.clip(cx - w // 2, m + 1, a.shape[1] - w - m - 1))
    if h < 64 or w < 64:
        return 0, 0
    ref_p = a[y0:y0 + h, x0:x0 + w].astype(np.float64)
    ref_p = (ref_p - ref_p.mean()) / (ref_p.std() + 1e-12)
    scores = {}
    for dy in range(-m, m + 1):
        for dx in range(-m, m + 1):
            q = b[y0 - dy:y0 - dy + h, x0 - dx:x0 - dx + w].astype(np.float64)
            q = (q - q.mean()) / (q.std() + 1e-12)
            scores[(dx, dy)] = float((ref_p * q).mean())
    best = max(scores, key=scores.get)
    if best != (0, 0) and scores[best] < scores[(0, 0)] + 0.02:
        return 0, 0
    return best


def star_offsets(ref: np.ndarray, img: np.ndarray, valid: np.ndarray, sat_ref: float | None = None,
                 sat_img: float | None = None, sigma: float = 3.0, max_stars: int = 400) -> np.ndarray:
    """Centroid offsets of the stars both planes measure (unsaturated in both): rows ``(x, y, dx, dy)`` with
    ``(dx, dy)`` = position in ``img`` minus position in ``ref`` (Gaussian-windowed centroids of the
    background-subtracted 13x13 neighbourhoods)."""
    from scipy import ndimage as ndi
    pad = 12
    sm = cv2.GaussianBlur(np.ascontiguousarray(ref, np.float32), (0, 0), 1.2)
    inner = ndi.binary_erosion(valid, iterations=pad + 3)
    if not inner.any():
        return np.zeros((0, 4))
    v = sm[valid][::7]
    sky = float(np.median(v))
    sig = 1.4826 * float(np.median(np.abs(v - sky)))
    top = 0.8 * sat_ref if sat_ref else float(np.quantile(sm[inner], 0.9999))
    pk = (sm == ndi.maximum_filter(sm, size=21)) & inner & (sm > sky + 30.0 * sig) & (sm < top)
    ys, xs = np.nonzero(pk)
    keep = np.argsort(-sm[ys, xs])[:max_stars]
    ys, xs = ys[keep], xs[keep]
    yy, xx = np.mgrid[-6:7, -6:7]
    w = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    rows = []
    for y, x in zip(ys, xs):
        a = ref[y - pad:y + pad + 1, x - pad:x + pad + 1]
        b = img[y - pad:y + pad + 1, x - pad:x + pad + 1]
        if (sat_img and b.max() >= 0.8 * sat_img) or (sat_ref and a.max() >= 0.8 * sat_ref):
            continue
        a = a[6:19, 6:19] - np.median(a)
        b = b[6:19, 6:19] - np.median(b)
        sa, sb = float((a * w).sum()), float((b * w).sum())
        if sa <= 0 or sb <= 0:
            continue
        rows.append((x, y, float((b * w * xx).sum()) / sb - float((a * w * xx).sum()) / sa,
                     float((b * w * yy).sum()) / sb - float((a * w * yy).sum()) / sa))
    return np.array(rows, float).reshape(-1, 4)


def fit_shift_field(offs: np.ndarray, min_stars: int = 30) -> np.ndarray | None:
    """Robust affine field ``d(x, y) = M @ (x, y, 1)`` (2x3) through the star offsets (iterated least squares with
    3-sigma MAD rejection); None when too few stars or the fit is no better than a constant."""
    if len(offs) < min_stars:
        return None
    A = np.c_[offs[:, 0], offs[:, 1], np.ones(len(offs))]
    keep = np.ones(len(offs), bool)
    M = np.zeros((2, 3))
    for _ in range(4):
        for k in (0, 1):
            M[k] = np.linalg.lstsq(A[keep], offs[keep, 2 + k], rcond=None)[0]
        res = offs[:, 2:4] - A @ M.T
        mad = 1.4826 * np.median(np.abs(res - np.median(res[keep], axis=0)), axis=0) + 1e-3
        keep = (np.abs(res) < 3.0 * mad).all(axis=1)
        if keep.sum() < min_stars:
            return None
    return M


def align_to_reference(ref: np.ndarray, img: np.ndarray, valid_ref: np.ndarray, valid_img: np.ndarray,
                       sats_ref: list, sats_img: list) -> tuple[np.ndarray, np.ndarray, list]:
    """Sub-pixel registration of one exposure-group stack onto another that shares its pixel grid up to whole pixels:
    the same stars are not at the same place (night 2 vs night 1: atmospheric dispersion differs per channel by up to
    0.3 px, plus ~1 px of pointing and a small affine term) and the mismatch shows as a coloured crescent on every star
    where the groups are exchanged.  Per channel: star centroid offsets (``star_offsets``), robust affine field
    (``fit_shift_field``), cubic resampling of ``img`` onto ``ref``'s grid.  Returns (img, valid, report); the plane
    is unchanged when a channel has too few stars or a field below 0.05 px."""
    out = img.copy()
    H, W = img.shape[1:]
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
    both = valid_ref & valid_img
    rep = []
    moved = False
    for c in range(img.shape[0]):
        M_tot = np.zeros((2, 3))
        cur = img[c]
        n_st = 0
        for _ in range(4):
            # a windowed centroid under-measures a shift (by ~0.7 for a 1 px offset): iterate on the resampled plane
            offs = star_offsets(ref[c], cur, both, sats_ref[c], sats_img[c])
            M = fit_shift_field(offs)
            if M is None:
                break
            n_st = len(offs)
            M_tot += M
            if np.hypot(M[0, 2], M[1, 2]) < 0.03 and np.abs(M[:, :2]).max() * max(H, W) < 0.03:
                break
            mx = (gx + M_tot[0, 0] * gx + M_tot[0, 1] * gy + M_tot[0, 2]).astype(np.float32)   # sample img where its star sits
            my = (gy + M_tot[1, 0] * gx + M_tot[1, 1] * gy + M_tot[1, 2]).astype(np.float32)
            cur = cv2.remap(np.ascontiguousarray(img[c]), mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        centre = M_tot @ np.array([W / 2.0, H / 2.0, 1.0])
        if not n_st:
            rep.append(None)
            continue
        rep.append({"stars": int(n_st), "dx": float(centre[0]), "dy": float(centre[1]), "affine": M_tot.round(8).tolist()})
        if cur is not img[c]:
            out[c] = cur
            moved = True
    if not moved:
        return img, valid_img, rep
    vm = cv2.erode(valid_img.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0       # interpolation reaches 3 px out
    out *= vm[None]
    return out, vm & valid_mask(out), rep


def psf_match_sigma(ref: np.ndarray, img: np.ndarray, valid: np.ndarray, sat_ref: float | None = None,
                    sat_img: float | None = None, radius: int = 10, max_sigma: float = 6.0) -> float:
    """Width (px) of the Gaussian that, convolved into ``img``, makes its stars look like ``ref``'s (0 when ``img`` is
    not the sharper one): minimum over sigma of the squared difference of the flux-normalised, background-subtracted
    star patches (unsaturated in both planes) of the two stacks."""
    from scipy import ndimage as ndi
    pad = radius + 4
    sm = cv2.GaussianBlur(np.ascontiguousarray(ref, np.float32), (0, 0), 1.5)
    inner = ndi.binary_erosion(valid, iterations=pad + 3)
    if not inner.any():
        return 0.0
    v = sm[valid][::7]
    sky = float(np.median(v))
    sig = 1.4826 * float(np.median(np.abs(v - sky)))
    top = 0.6 * sat_ref if sat_ref else float(np.quantile(sm[inner], 0.9999))
    pk = (sm == ndi.maximum_filter(sm, size=2 * pad + 1)) & inner & (sm > sky + 30.0 * sig) & (sm < top)
    ys, xs = np.nonzero(pk)
    if ys.size < 8:
        return 0.0
    keep = np.argsort(-sm[ys, xs])[:150]
    ys, xs = ys[keep], xs[keep]
    yy, xx = np.mgrid[-pad:pad + 1, -pad:pad + 1]
    rr = np.hypot(yy, xx)
    ap, an = rr <= radius, (rr >= radius + 1.5) & (rr <= pad)
    inner_m = rr <= radius
    sigmas = np.arange(0.0, max_sigma + 1e-6, 0.25)
    cost = np.zeros(len(sigmas))
    n = 0
    for y, x in zip(ys, xs):
        a = ref[y - pad:y + pad + 1, x - pad:x + pad + 1].astype(np.float64)
        b = img[y - pad:y + pad + 1, x - pad:x + pad + 1].astype(np.float64)
        if (sat_ref and a.max() >= 0.8 * sat_ref) or (sat_img and b.max() >= 0.8 * sat_img):
            continue
        a = a - np.median(a[an])
        fa = float(a[ap].sum())
        if fa <= 0:
            continue
        a = a / fa
        for k, sg in enumerate(sigmas):
            bb = cv2.GaussianBlur(b, (0, 0), sg) if sg > 0 else b
            bb = bb - np.median(bb[an])
            fb = float(bb[ap].sum())
            if fb <= 0:
                cost[k] += 1e3
                continue
            cost[k] += float(((bb / fb - a)[inner_m] ** 2).sum())
        n += 1
    if n < 8:
        return 0.0
    k = int(np.argmin(cost))
    if k in (0, len(sigmas) - 1):
        return float(sigmas[k])
    c0, c1, c2 = cost[k - 1], cost[k], cost[k + 1]
    den = c0 - 2 * c1 + c2
    off = 0.5 * (c0 - c2) / den if den > 0 else 0.0
    return float(max(0.0, sigmas[k] + 0.25 * np.clip(off, -1, 1)))


def match_psf(ref: np.ndarray, img: np.ndarray, valid_ref: np.ndarray, valid_img: np.ndarray,
              sats_ref: list, sats_img: list) -> tuple[np.ndarray, np.ndarray, list]:
    """Convolve the sharper exposure group with the Gaussian that gives it the other group's stellar profile
    (``psf_match_sigma``, per channel).  Stars of two nights (15 s vs 60 s: seeing, guiding, focus) differ in width by
    a factor 1.5-2; where the longer group clips and the shorter one fills the core, the sharp star sits inside the
    broad one's clipped disc and leaves a dark ring (hollow 'donut' stars).  Matching the PSFs first (the standard
    homogenisation of multi-epoch coadds) makes both groups measure the same profile; flux is conserved, no value is
    invented.  Returns (img, valid, sigmas)."""
    out = img.copy()
    both = valid_ref & valid_img
    sig = []
    for c in range(img.shape[0]):
        sg = psf_match_sigma(ref[c], img[c], both, sats_ref[c], sats_img[c])
        sig.append(round(sg, 3))
        if sg >= 0.3:
            out[c] = cv2.GaussianBlur(np.ascontiguousarray(img[c]), (0, 0), sg)
    k = int(math.ceil(3.0 * max(sig))) if max(sig) >= 0.3 else 0
    if k == 0:
        return img, valid_img, sig
    vm = cv2.erode(valid_img.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
    out *= vm[None]
    return out, vm & valid_mask(out), sig


def paste(stack: np.ndarray, canvas_hw: tuple[int, int], dx: int, dy: int) -> np.ndarray:
    """Place ``stack`` (3, h, w) at offset (dx, dy) on a zero canvas of ``canvas_hw`` (clipped)."""
    H, W = canvas_hw
    out = np.zeros((stack.shape[0], H, W), np.float32)
    h, w = stack.shape[1:]
    x0, y0 = max(dx, 0), max(dy, 0)
    x1, y1 = min(dx + w, W), min(dy + h, H)
    if x1 > x0 and y1 > y0:
        out[:, y0:y1, x0:x1] = stack[:, y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return out
