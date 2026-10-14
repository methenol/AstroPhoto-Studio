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
                   convention: str = "H", down: int = 4) -> tuple[np.ndarray, tuple[float, float]]:
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
    x0, y0 = allp.min(axis=0)
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
              threshold: float) -> tuple[tuple[int, int, int, int], dict]:
    """Crop box for a -framing=max stack plus a report of how the coverage was obtained.

    Coverage by geometry (projected frame outlines, validated against the data mask with both homography
    conventions; IoU >= 0.95 required); otherwise the data mask alone (any frame covers = threshold ignored,
    reported)."""
    valid = valid_mask(stack)
    H, W = valid.shape
    info: dict = {"method": "data mask"}
    cov = None
    if hmats:
        best = (0.0, None, None)
        for conv in CONVENTIONS:
            try:
                c, _ = frame_coverage(hmats, sub_size[0], sub_size[1], scale, (W, H), conv)
            except Exception:
                continue
            vs = cv2.resize(valid.astype(np.uint8), (c.shape[1], c.shape[0]), interpolation=cv2.INTER_AREA) > 0
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
    box = crop_box(valid, cov, threshold, len(hmats) if hmats else 1)
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


def saturation_level(plane: np.ndarray, valid: np.ndarray | None = None) -> float | None:
    """Level of the clipped plateau at the top of a stack's histogram (None when nothing is clipped): a value at
    the maximum reached by a handful of pixels is a saturated plateau."""
    v = plane[valid] if valid is not None else plane.ravel()
    if v.size == 0:
        return None
    top = float(v.max())
    if top <= 0:
        return None
    return top if int((v >= top * 0.999).sum()) >= 20 else None


def calibrate_to_reference(ref: np.ndarray, img: np.ndarray, valid: np.ndarray, sat_ref: float | None = None,
                           n_sigma: float = 8.0) -> tuple[float, float, int]:
    """Gain ``a`` and offset ``b`` with ``img ~ a * ref + b`` over the pixels both radiance planes cover (HDR_Blender's
    optional "calibrate each exposure against the longest": it removes scale errors, e.g. a different flat or
    normalisation, and pedestal offsets, e.g. a dark of another temperature, which would otherwise show as a
    brightness step wherever only one exposure group has data).

    Estimator (a plain least-squares line is useless here: the sky occupies a few noise-widths of ``ref`` and the
    regression dilutes the slope to 0.4-0.7 on real data): both planes are lightly blurred; the sky level of each is
    its median (the field is mostly sky); the gain is the median of ``(img - sky_img) / (ref - sky_ref)`` over pixels
    that stand ``n_sigma`` noise-widths above the sky in ``ref`` and below 60 % of ``sat_ref`` (clipped star cores
    excluded); ``b`` then makes the sky levels agree.  Returns (a, b, n_pixels); (1, sky difference, n) when fewer than
    1000 pixels qualify, and (1, 0, n) when the overlap is too small."""
    n = int(valid.sum())
    if n < 20000:
        return 1.0, 0.0, n
    r = cv2.GaussianBlur(np.ascontiguousarray(ref, np.float32), (0, 0), 2.0)[valid]
    m = cv2.GaussianBlur(np.ascontiguousarray(img, np.float32), (0, 0), 2.0)[valid]
    r0, m0 = float(np.median(r)), float(np.median(m))
    sig = 1.4826 * float(np.median(np.abs(r[r <= np.quantile(r, 0.6)] - r0)))
    top = 0.6 * sat_ref if sat_ref else float(np.quantile(r, 0.9999))
    sel = (r > r0 + n_sigma * sig) & (r < top)
    if int(sel.sum()) < 1000:
        return 1.0, m0 - r0, n
    ratio = (m[sel] - m0) / (r[sel] - r0)
    g = float(np.median(ratio))
    if not (0.5 < g < 2.0):                           # nonsense (structure mismatch): do not apply
        return 1.0, m0 - r0, n
    return g, m0 - g * r0, n


SAT_TAPER = (0.75, 0.95)      # weight tapers from 0.75 to 0.95 of the ADC full scale (zero above)


def full_scale_norm(raw_max: float) -> float:
    """ADC full scale in Siril's normalised units (raw / 65535) from the largest raw value seen in some subs: the
    next 2^k - 1 (DWARF 3: 4095 -> 0.0625; a 16-bit camera: 65535 -> 1.0)."""
    k = max(10, int(np.ceil(np.log2(raw_max + 1.0))))
    return float((2 ** k - 1) / 65535.0)


def hdr_blend(stacks: list[np.ndarray], exptimes: list[float], valids: list[np.ndarray],
              margin: float = 0.05, calibrate: bool = True, full_scale: float | None = None) -> tuple[np.ndarray, dict]:
    """Inverse-variance MLE blend of registered stacks of different exposures (A. Knagg-Baugh's HDR_Blender.py,
    workflows.md 9.2), per channel:

    * radiance = stack / exposure (ADU/s); shorter exposures are first calibrated to the longest one (gain + offset,
      ``calibrate_to_reference``);
    * weight = 1 / variance, variance from the Poisson-Gaussian noise model of each stack (``noise_model``);
    * weight 0 outside the stack's footprint (``valids``) and, through a cosine taper, near saturation: with
      ``full_scale`` (the ADC full scale in stack units) from 0.75 to 0.95 of it (a stack of clipped subs is already
      non-linear below the clip: the clipped values average with unclipped ones, and calibration moves the clip
      level by the flat and the dark pedestal), else at the plateau detected at the top of the stack's histogram
      (``margin``);
    * pixels with no weight anywhere (saturated in every input) take the shortest exposure's measured radiance
      (a lower bound, nothing invented).
    The result is scaled back to the longest exposure's units and divided by its maximum, so it stays in [0, 1].
    Every value comes from the inputs."""
    shape = stacks[0].shape
    n_ch = shape[0]
    acc = np.zeros(shape, np.float32)
    wsum = np.zeros(shape, np.float32)
    report: dict = {"inputs": []}
    sats = [[None] * n_ch for _ in stacks]
    ref = int(np.argmax(exptimes))
    both = [np.logical_and(valids[ref], v) for v in valids]
    rad_all = []
    for j, (s, t, vm) in enumerate(zip(stacks, exptimes, valids)):
        info = {"exptime": float(t), "noise": [], "saturation": [], "gain_offset": []}
        rad = (s / t).astype(np.float32)
        for c in range(n_ch):
            plane = s[c]
            sat = full_scale if full_scale else saturation_level(plane, vm)
            sats[j][c] = sat
            if calibrate and j != ref:
                sat_ref = (full_scale * SAT_TAPER[0]) if full_scale else saturation_level(stacks[ref][c], valids[ref])
                g, o, npx = calibrate_to_reference(stacks[ref][c] / exptimes[ref], rad[c], both[j],
                                                   (sat_ref / exptimes[ref]) if sat_ref else None)
                rad[c] = ((rad[c] - o) / g).astype(np.float32)
                info["gain_offset"].append([g, o, npx])
            info["saturation"].append(sat)
        rad_all.append(rad)
        report["inputs"].append(info)
    for j, (s, t, vm) in enumerate(zip(stacks, exptimes, valids)):
        for c in range(n_ch):
            plane = s[c]
            a, b = noise_model(np.where(vm, plane, 0.0))
            var = np.maximum((a + b * np.clip(plane, 0, None)) / (t * t), 1e-18)
            w = np.where(vm, 1.0 / var, 0.0).astype(np.float32)
            sat = sats[j][c]
            if sat is not None:
                lo, hi = (sat * SAT_TAPER[0], sat * SAT_TAPER[1]) if full_scale else (sat * (1.0 - margin), sat * 0.999)
                x = np.clip((plane - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
                w *= (0.5 * (1.0 + np.cos(np.pi * x))).astype(np.float32)
            acc[c] += w * rad_all[j][c]
            wsum[c] += w
            report["inputs"][j]["noise"].append([a, b])
    blend = np.where(wsum > 0, acc / np.maximum(wsum, 1e-30), 0.0).astype(np.float32)
    any_valid = np.any(np.stack(valids), axis=0)
    none = (wsum <= 0) & any_valid[None]
    if none.any():
        short = int(np.argmin(exptimes))
        for c in range(n_ch):
            fill = np.where(valids[short], rad_all[short][c], np.max([r[c] for r in rad_all], axis=0))
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
