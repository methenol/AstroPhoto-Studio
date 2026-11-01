"""Image statistics and the numpy helpers the post-stack stages share (the finishing chain itself is ``pro.py``).

* statistics: ``luminance``, ``sample``, ``mad_sigma``, ``lower_half_sigma``, ``object_snr`` (per-pixel signal-to-noise of the object: the strengths of BXT / NXT follow it),
  ``object_coverage``, ``count_stars``, ``mean_saturation``;
* RC-Astro / GraXpert strengths by signal-to-noise (``bxt_ss``, ``bxt_sn``, ``nxt_dn``, ``graxpert_strength``: the published ranges of research/pro-workflow.md);
* ``ghs``: the numpy mirror of Siril's ``ght`` (identical to 1e-6, tests/test_stretch.py) used by ``pro.fit_ghs2`` to solve the one free parameter of the stretch on scalars;
* ``neutralize_background`` / ``star_white_balance`` / ``rmgreen_np`` (restore-stage fallbacks when SPCC cannot run), ``subtract_star_wings`` (StarNet2's scattered light),
  ``screen`` / ``unscreen`` (the screen blend and its inverse, for the gates).
"""
from __future__ import annotations

import cv2
import numpy as np

# Rec. 709 luminance weights (Statistical_Stretch's luminance mode uses the same ones).
LUM709 = (0.2126, 0.7152, 0.0722)


# --------------------------------------------------------------------------- statistics

def luminance(rgb: np.ndarray, weights=LUM709) -> np.ndarray:
    """Weighted luminance of a (3, H, W) array."""
    return rgb[0] * np.float32(weights[0]) + rgb[1] * np.float32(weights[1]) + rgb[2] * np.float32(weights[2])


def luminance_hwc(img_hwc: np.ndarray) -> np.ndarray:
    """Rec. 709 luminance of an (H, W, 3) image."""
    return (0.2126 * img_hwc[..., 0] + 0.7152 * img_hwc[..., 1] + 0.0722 * img_hwc[..., 2]).astype(np.float32)


def sample(a: np.ndarray, n: int = 500_000, valid_only: bool = True) -> np.ndarray:
    """A strided sample of ``a`` (same spatial positions for every plane of equal shape).  Exact zeros
    (the empty border of a stack that was not cropped) are dropped when they are more than 0.1 % of
    the sample, so they cannot drag the median to 0."""
    flat = np.asarray(a).reshape(-1)
    s = flat[:: max(1, flat.size // n)]
    if valid_only and s.size and (s == 0).mean() > 0.001 and (s != 0).sum() > 1000:
        s = s[s != 0]
    return s


def mad_sigma(x: np.ndarray) -> float:
    """Robust sigma = 1.4826 x median absolute deviation."""
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def lower_half_sigma(x: np.ndarray) -> float:
    """Statistical_Stretch's noise estimate: 1.4826 x MAD of the values at or below the median (the
    nebula and the stars sit in the upper half, so the lower half is the sky's own scatter)."""
    x = np.asarray(x, np.float32).reshape(-1)
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    lo = x[x <= med]
    if lo.size < 16:
        return float(1.4826 * np.median(np.abs(x - med)))
    return float(1.4826 * np.median(np.abs(lo - np.median(lo))))


# Noise-aware strengths.  RC-Astro's guidance (BlurXTerminator / NoiseXTerminator manuals, rc-astro.com): non-stellar sharpening has to come down on noisy data
# (it turns noise into texture and rings), NoiseXTerminator wants 0.6-0.9 and the noisier the data the higher; GraXpert's AI denoise is the same kind of tool.  The
# signal-to-noise they key on is measured here, per pixel, on the object: the 90th percentile of the (1.5 px smoothed) luminance above the sky over the pixel noise
# (MAD of the 5 px high-pass), over the pixels that stand 3 sigma of the smoothed luminance above the sky and are not stars (compact excess over a 4 px
# smoothing: more than 5 sigma of pixel noise, dilated 15 px; a nebula is smooth and stays in).  Calibration of the ladder on this statistic: IC 1805, Sh2-132 / 157 / 171
# (1-3 h of 30 s DWARF 3 frames, the nebula barely above the noise per pixel) read 1.5-3, M 42 / M 31 / M 45 4-8, so "noisy" = 3 and below (the pixel signal is
# smaller than the pixel noise) and "clean" = 15 and above (the signal stands 5 sigma above the noise even after the 1.5 px smoothing).
SNR_LOW, SNR_HIGH = 3.0, 15.0


def object_snr(lum: np.ndarray) -> float:
    """Per-pixel signal-to-noise of the object in a linear luminance plane (see ``SNR_LOW``); 0 when there is no object."""
    lum = np.ascontiguousarray(lum, dtype=np.float32)
    hp = lum - cv2.GaussianBlur(lum, (0, 0), 5.0)
    noise = float(1.4826 * np.median(np.abs(hp - np.median(hp))))
    sm = cv2.GaussianBlur(lum, (0, 0), 1.5)
    s = sample(sm, valid_only=True)
    sky = float(np.median(s[s <= np.percentile(s, 40)]))
    res = sm - cv2.GaussianBlur(sm, (0, 0), 10.0)
    ns = float(1.4826 * np.median(np.abs(res[:: max(1, res.shape[0] // 700), :: max(1, res.shape[1] // 700)])))
    s1 = cv2.GaussianBlur(lum, (0, 0), 1.0)
    stars = cv2.dilate(((s1 - cv2.GaussianBlur(lum, (0, 0), 4.0)) > 5.0 * noise).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
    obj = (sm > sky + 3.0 * ns) & ~stars
    if noise <= 0 or obj.sum() < 500:
        return 0.0
    return float(np.percentile((sm[obj] - sky) / noise, 90))


def snr_ladder(snr: float, low_value: float, high_value: float) -> float:
    """``low_value`` at ``SNR_LOW`` and below, ``high_value`` at ``SNR_HIGH`` and above, linear in log(snr) between."""
    t = float(np.clip(np.log(max(snr, 1e-6) / SNR_LOW) / np.log(SNR_HIGH / SNR_LOW), 0.0, 1.0))
    return low_value + (high_value - low_value) * t


BXT_SS = (0.25, 0.35)             # Sharpen Stars for 1-2 h smart-scope stacks (research/pro-workflow.md A5: Brecher 0.5 on 15-19 h, AstroPipe 0.25 on smart-scope data)
BXT_SN = (0.50, 0.60)             # Sharpen Nonstellar for the same stacks (Brecher 0.9 on 15-19 h; AstroPipe 0.60); the lower end on the noisiest data (RC-Astro: sharpen less on noisy data)
NXT_LINEAR_DN = (0.80, 0.50)      # NXT --dn on the linear data (pro-workflow.md A6: Brecher 0.9-0.95, AstroPipe 0.25, Astrobackyard 0.5; recommendation 0.5-0.8): 0.8 on the noisiest data
GRAXPERT_DN = (0.80, 0.60)        # GraXpert denoise strength where NXT is unavailable (pro-workflow.md A10: 0.6-0.8)


def bxt_sn(snr: float) -> float:
    """BlurXTerminator non-stellar sharpening within ``BXT_SN`` (0.5 on noisy data, 0.6 on clean data)."""
    return round(snr_ladder(snr, *BXT_SN), 3)


def bxt_ss(snr: float) -> float:
    """BlurXTerminator stellar sharpening within ``BXT_SS``."""
    return round(snr_ladder(snr, *BXT_SS), 3)


def nxt_dn(snr: float) -> float:
    """NoiseXTerminator ``--dn`` on the linear data: ``NXT_LINEAR_DN`` (0.8 on noisy data, 0.5 on clean data)."""
    return round(snr_ladder(snr, *NXT_LINEAR_DN), 3)


def graxpert_strength(snr: float) -> float:
    """GraXpert AI denoise strength (the stand-in for NXT): ``GRAXPERT_DN``."""
    return round(snr_ladder(snr, *GRAXPERT_DN), 3)



# --------------------------------------------------------------------------- the stretch curves
















# --------------------------------------------------------------------------- dual-band palette





# --------------------------------------------------------------------------- star wings left by StarNet

WING_MIN_PEAK = 0.1          # only stars at least this bright (linear, 0..1) leave a wing above the noise: ~ 1e-3 at 30 px
WING_MAX_STARS = 60
WING_CORE = 0.05             # star-layer level (linear) that marks a core to keep out of the rings
WING_RING = 6                # annulus width (px) of the radial profile
WING_TEXTURE = 5e-4          # quadrant medians of the sky differ by this much from large-scale structure alone
WING_END_ABOVE_SKY = 1e-4    # a profile whose outer rings stand more than this (linear) above the image sky ends in a nebula, not in the sky
WING_SYMMETRY = 0.5          # quadrant medians may differ by this fraction of the wing (or 3e-4, the sky texture) before the profile stops


def _ring_profile(win: np.ndarray, rr: np.ndarray, quad: np.ndarray, ok: np.ndarray, sky0: np.ndarray, rmax: int):
    """Azimuthal median profile of the layer ``win`` (3, h, w) around the window centre, in rings of ``WING_RING`` px, followed
    outward while it falls and stays circularly symmetric.  Returns ``(ring_start_radii, profile (3, n))``."""
    prof: list[np.ndarray] = []
    used: list[int] = []
    for r0 in range(10, rmax, WING_RING):
        sel = ok & (rr >= r0) & (rr < r0 + WING_RING)
        if sel.sum() < 200:
            if prof:
                break
            continue                                       # the core is still masked: start further out
        lvl = np.array([np.median(win[c][sel]) for c in range(3)], np.float32)
        spread = max(float(np.ptp([np.median(win[c][sel & (quad == k)]) for k in range(4)])) for c in range(3))
        wing = float((lvl - sky0).max())
        if prof and (lvl.mean() > prof[-1].mean() + 4e-5 or spread > max(WING_SYMMETRY * wing, WING_TEXTURE)):
            break
        prof.append(lvl)
        used.append(r0)
        if wing < 3e-5:                                       # the profile has reached the sky
            break
    if len(prof) < 4:
        return [], np.zeros((3, 0), np.float32)
    raw = np.stack(prof, 1)
    p = np.minimum.accumulate(raw, axis=1)                                  # monotone falling outward
    # the level the profile ends in: the median of its outermost rings.  (It used to be capped at the image's sky median + 3e-5, which made every nebula glow
    # around a star count as wing: on IC 1805 the profiles of 18 stars took 213 flux units out of the Heart's cavities and left black discs of up to 450 px.)
    ref = np.median(raw[:, -5:], axis=1)
    if float((ref - sky0).max()) > WING_END_ABOVE_SKY:       # the profile never reached the sky (it ends on a nebula): that glow is not wing, subtract nothing
        return [], np.zeros((3, 0), np.float32)
    model = np.clip(p - ref[:, None], 0, None)
    # a profile that stopped on an asymmetry still has a wing left at its last ring: continue it as the power law of its last
    # rings (log-log slope, wings fall at least as r^-1) down to nothing, so no plateau is left inside the disc
    r_end = used[-1] + WING_RING / 2
    extra_r = np.arange(used[-1] + WING_RING, rmax, WING_RING)
    if len(extra_r) and model[:, -1].max() > 3e-5:
        rc = np.array(used, np.float32) + WING_RING / 2
        k = max(3, len(used) // 4)
        tail = []
        for c in range(3):
            y = np.maximum(model[c, -k:], 1e-7)
            a = float(np.clip(-np.polyfit(np.log(rc[-k:]), np.log(y), 1)[0], 1.0, 4.0))
            tail.append(model[c, -1] * ((extra_r + WING_RING / 2) / r_end) ** (-a))
        used = used + [int(r) for r in extra_r]
        model = np.concatenate([model, np.stack(tail)], axis=1)
    return used, model


def subtract_star_wings(lin: np.ndarray, starless: np.ndarray, stars: np.ndarray) -> tuple[np.ndarray, dict, np.ndarray]:
    """Move the scattered-light wings of bright stars from the starless layer to the star layer.

    StarNet2 removes the core and the first rings of a star but leaves its wing (measured on M 42: iota Ori keeps a
    smooth glow of 1.6e-3 at 30 px falling to 2e-4 at 140 px in the starless layer, which the stretch then blows up into a
    cyan disc); StarXTerminator, trained on such wings, takes them out.  The wing is the star's own light, so the photons
    are *reassigned*, never invented: for each bright star the azimuthal median profile of the starless layer is measured
    in rings outward and followed while it is (a) falling and (b) circularly symmetric (the four quadrant medians agree
    to ``WING_SYMMETRY`` of the wing: a nebula, a neighbouring star or a gradient breaks the symmetry and ends the
    profile there); the profile relative to the sky it ends in is removed from the starless layer as a smooth model (the noise
    of the layer is untouched) and so lands in the star layer (``lin - starless``).  Growth-curve / PSF-wing modelling as in
    stellar photometry.  (A flux-scaled template shared by all stars was tried and rejected: the wing's colour differs from
    star to star, which carved a dark hole around the neighbours of iota Ori.)  Returns ``(starless, info, stars)``."""
    h, w = starless.shape[1:]
    lum = cv2.GaussianBlur(np.ascontiguousarray(stars.max(0)), (0, 0), 1.0)
    dil = cv2.dilate(lum, np.ones((15, 15), np.uint8))
    ys, xs = np.nonzero((lum == dil) & (lum > WING_MIN_PEAK))
    order = np.argsort(-lum[ys, xs])[:WING_MAX_STARS]
    ys, xs = ys[order], xs[order]
    sky0 = np.array([float(np.median(sample(starless[c]))) for c in range(3)], np.float32)
    star_mask = cv2.dilate((lum > WING_CORE).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0   # other stars' cores are not part of the sky ring
    out = starless.copy()
    moved, used_n, radii = 0.0, 0, []
    for cy, cx in zip(ys, xs):
        rmax = int(min(450, cy, cx, h - 1 - cy, w - 1 - cx))
        if rmax < 40:
            continue
        y0, y1, x0, x1 = cy - rmax, cy + rmax + 1, cx - rmax, cx + rmax + 1
        yy, xx = np.mgrid[y0 - cy:y1 - cy, x0 - cx:x1 - cx]
        rr = np.hypot(yy, xx)
        quad = (yy >= 0).astype(int) * 2 + (xx >= 0).astype(int)
        win = out[:, y0:y1, x0:x1]
        used, model = _ring_profile(win, rr, quad, ~star_mask[y0:y1, x0:x1], sky0, rmax)
        if not used or model.max() <= 1e-5:
            continue
        rc = np.array(used, np.float32) + WING_RING / 2
        full = np.stack([np.interp(rr, rc, model[c], left=model[c, 0], right=0.0) for c in range(3)]).astype(np.float32)
        sub = np.minimum(full, np.maximum(win, 0))
        win -= sub
        moved += float(sub.sum())
        used_n += 1
        radii.append(int(used[-1]))
    return out, {"stars": used_n, "flux_moved": round(moved, 3), "radius_px": radii[:12]}, np.maximum(lin - out, 0).astype(np.float32)


# --------------------------------------------------------------------------- stars


def unscreen(full: np.ndarray, starless: np.ndarray) -> np.ndarray:
    """The star layer ``s`` with ``screen(starless, s) == full``: ``(full - starless) / (1 - starless)``
    (what ``starnet2 -n`` / ``sxt --unscreen`` produce; for stretched data only)."""
    return np.clip((full - starless) / np.maximum(1.0 - starless, 1e-4), 0.0, 1.0).astype(np.float32)











# --------------------------------------------------------------------------- numpy twins of Siril commands


def rmgreen_np(rgb: np.ndarray, amount: float) -> np.ndarray:
    """``rmgreen 0 amount``: average-neutral SCNR, lightness preserved: G -> min(G, (R+B)/2) by ``amount``."""
    if amount <= 0:
        return rgb
    out = rgb.copy()
    neutral = 0.5 * (rgb[0] + rgb[2])
    out[1] = rgb[1] * (1 - amount) + np.minimum(rgb[1], neutral) * amount
    lo, ln = luminance(rgb), luminance(out)
    return np.clip(out * (lo / np.maximum(ln, 1e-6))[None], 0, 1)



def brightness_to_mid(brightness: float) -> float:
    """Midtones slider (-1..1, 0 neutral) -> the ``mid`` of ``mtf``: m = 0.5 x 4^-b (positive brightens)."""
    return float(np.clip(0.5 * 4.0 ** (-float(brightness)), 0.05, 0.95))


# --------------------------------------------------------------------------- detail tools (HWC float32 in/out)



                                                       # scales 1-3: Brecher's MultiscaleLinearTransform chrominance thresholds 5.0 / 4 / 3 (astrodoc.ca IC 417 SHO),
                                                       # beyond: 3 (the large-scale colour mottle of the stack: the user's blotches), as TGVDenoise / MLT chrominance does at every scale






# --------------------------------------------------------------------------- measurements for auto-finish

def object_coverage(lum: np.ndarray, sigma_k: float = 3.0) -> float:
    """Fraction of the frame holding signal: pixels of the (smoothed) luminance above the sky by ``sigma_k`` x
    its noise.  The sky level is the 20th percentile (not the median: an object that fills the frame lifts
    the median) and the noise the scatter of the darkest 40 %.  Used to choose the target median (a small object
    on empty sky gets a darker sky than a nebula that fills the frame)."""
    step = max(1, int(round(max(lum.shape) / 1200)))
    small = cv2.resize(np.ascontiguousarray(lum), (lum.shape[1] // step, lum.shape[0] // step),
                       interpolation=cv2.INTER_AREA) if step > 1 else lum
    small = cv2.GaussianBlur(small, (0, 0), 2.0)
    s = small[small > 0] if (small > 0).mean() > 0.5 else small.reshape(-1)
    sky = float(np.percentile(s, 20))
    dark = s[s <= np.percentile(s, 40)]
    sig = 1.4826 * float(np.median(np.abs(dark - np.median(dark)))) if dark.size > 16 else 0.0
    return float((s > sky + sigma_k * sig).mean()) if sig > 0 else 0.0


def star_positions(stars_lin: np.ndarray, noise_sigma: float, min_sigma: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """(ys, xs) of the stars count_stars counts."""
    if noise_sigma <= 0:
        return np.zeros(0, int), np.zeros(0, int)
    lum = cv2.GaussianBlur(np.ascontiguousarray(stars_lin.mean(0)), (0, 0), 1.0)
    dil = cv2.dilate(lum, np.ones((7, 7), np.uint8))
    return np.nonzero((lum == dil) & (lum > min_sigma * noise_sigma))


def count_stars(stars_lin: np.ndarray, noise_sigma: float, min_sigma: float = 5.0) -> tuple[int, float]:
    """Detectable stars: local maxima of the (lightly smoothed) star layer above ``min_sigma`` x the sky's pixel noise
    ``noise_sigma`` (of the starless layer), and their number per megapixel."""
    if noise_sigma <= 0:
        return 0, 0.0
    # the channel mean, as the picture's count (gates.measure_stars): detection must not depend on colour - on the maximum channel a colour boost of
    # 2 lifted faint stars' chroma noise over the threshold and the picture counted 4.7 x the linear stars (M 31)
    lum = cv2.GaussianBlur(np.ascontiguousarray(stars_lin.mean(0)), (0, 0), 1.0)
    dil = cv2.dilate(lum, np.ones((7, 7), np.uint8))
    n = int(((lum == dil) & (lum > min_sigma * noise_sigma)).sum())
    return n, n / (lum.size / 1e6)



def neutralize_background(rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    """Background neutralisation for the linear data when SPCC could not run: the sky (darkest 35 % of the
    smoothed luminance, nebulosity and stars excluded) of each channel is shifted to the common level of the
    darkest channel, so the sky is grey.  Offsets only, no gains."""
    lum = luminance(rgb)
    step = max(1, int(round(max(lum.shape) / 1000)))
    sm = cv2.GaussianBlur(np.ascontiguousarray(lum[::step, ::step]), (0, 0), 1.5)
    valid = sm > 0
    thr = np.percentile(sm[valid], 35) if valid.any() else 0
    sky = valid & (sm <= thr)
    meds = [float(np.median(np.ascontiguousarray(rgb[c, ::step, ::step])[sky])) for c in range(3)]
    ref = min(meds)
    out = np.empty_like(rgb)
    for c in range(3):
        out[c] = np.maximum(rgb[c] - np.float32(meds[c] - ref), 0)
    return out, {"sky_medians": [round(m, 6) for m in meds], "offsets": [round(m - ref, 6) for m in meds]}


def star_white_balance(rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    """Colour balance from the stars themselves ("stars" white balance): unsaturated star pixels (the bright
    top-hat residue below 85 % of full scale) are on average sun-like, so the channel gains are chosen to make
    their mean colour neutral.  Gains are normalised to G = 1."""
    lum = luminance(rgb)
    step = max(1, int(round(max(lum.shape) / 2000)))
    l = np.ascontiguousarray(lum[::step, ::step])
    bg = cv2.GaussianBlur(cv2.morphologyEx(l, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)), (0, 0), 3)
    th = l - bg
    sig = mad_sigma(th)
    sel = (th > 12 * sig) & (l < 0.85 * float(l.max()))
    if sel.sum() < 30:
        return rgb, {"gains": [1, 1, 1], "stars": int(sel.sum())}
    cols = [np.ascontiguousarray(rgb[c, ::step, ::step]) for c in range(3)]
    flux = [float((cols[c] - cv2.GaussianBlur(cv2.morphologyEx(cols[c], cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)),
                                               (0, 0), 3))[sel].sum()) for c in range(3)]
    gains = [flux[1] / max(f, 1e-9) for f in flux]
    out = np.stack([rgb[c] * np.float32(gains[c]) for c in range(3)])
    return out, {"gains": [round(g, 4) for g in gains], "stars": int(sel.sum())}


EDGE_CROP = 24                    # px trimmed from every side of the stack before the render (post._Layers): the rim holds fewer dithered frames, measured 25 % noisier
                                  # the output range sits 0.4 % above the black point of a white-scaled plane: the slope below it would have to exceed the slope above it
_STARLET_NOISE: dict = {}










def _ghs_g(u: np.ndarray, b: float, d: float) -> np.ndarray:
    """Integral of the GHS slope ``(1 + b D t)^(-1 - 1/b)`` from 0 to ``u``: ``(1 - (1 + b D u)^(-1/b)) / D`` (b = 0: ``(1 - exp(-D u)) / D``).  Payne & Cranfield's harmonic
    family, ``T(x) = 1 - (1 + b D x)^(-1/b)`` (Siril ``ght`` is checked against it in tests/test_finish.py)."""
    u = np.asarray(u, np.float64)
    if abs(b) < 1e-6:
        return (1.0 - np.exp(-d * u)) / d
    return (1.0 - (1.0 + b * d * u) ** (-1.0 / b)) / d


def _ghs_slope(t: np.ndarray, b: float, d: float) -> np.ndarray:
    """The GHS slope at distance ``t`` from the symmetry point, ``(1 + b D t)^(-1 - 1/b)`` (b = 0: ``exp(-D t)``)."""
    t = np.asarray(t, np.float64)
    return np.exp(-d * t) if abs(b) < 1e-6 else (1.0 + b * d * t) ** (-1.0 - 1.0 / b)


def _ghs_t(x, b: float, d, sp, lp=0.0, hp=1.0):
    """The normalised GHS ``T(x)`` (see ``ghs``); ``x``, ``d``, ``sp``, ``lp``, ``hp`` broadcast against each other (``b`` is a scalar): used by the fit on parameter grids."""
    x = np.asarray(x, np.float64)

    def raw(v):
        core = np.clip(v, lp, hp)
        f = np.where(core >= sp, _ghs_g(np.abs(core - sp), b, d), -_ghs_g(np.abs(core - sp), b, d))
        return f + np.where(v < lp, _ghs_slope(sp - lp, b, d) * (v - lp), 0.0) + np.where(v > hp, _ghs_slope(hp - sp, b, d) * (v - hp), 0.0)
    r0, r1 = raw(np.float64(0.0)), raw(np.float64(1.0))
    return (raw(x) - r0) / (r1 - r0)


def ghs(x: np.ndarray, b: float, d: float, sp: float, lp: float = 0.0, hp: float = 1.0) -> np.ndarray:
    """Generalised Hyperbolic Stretch (Payne & Cranfield 2021, https://ghsastro.co.uk) of ``x`` in [0, 1] with local intensity ``b``, stretch factor ``d`` (the
    factor itself; Siril's ``-D`` is ``ln(d + 1)``), symmetry point ``sp``, shadow protection ``lp`` and highlight protection ``hp``: between LP and HP the slope peaks at
    SP and falls off as ``(1 + b D |x - SP|)^(-1 - 1/b)`` on both sides (odd about SP); below LP and above HP the transform is the straight line that continues the curve
    with its slope at LP / HP; normalised to 0 -> 0, 1 -> 1.  This is the numpy mirror of Siril's ``ght`` (checked against it to 1e-6 in tests/test_finish.py, docker
    marker): it is used to FIT the parameters; the pixels are stretched by Siril (``stretch.py``) and only by this function when Siril is not available."""
    lp, hp = float(np.clip(lp, 0.0, sp)), float(np.clip(hp, sp, 1.0))
    return np.asarray(_ghs_t(x, float(b), float(d), float(sp), lp, hp), np.float32)









BAND_LO, BAND_HI = 10.7, 42.7     # gaussian sigmas (px) of the 32-128 px band the user's "watercolour" blobs live in (size ~ 3 sigma)
BAND_SKY_FRACTION = 0.45          # the darkest 45 % of the 64 px smoothed plane are the sky samples of the band-noise estimates


def _sky_mask(plane: np.ndarray, fraction: float = BAND_SKY_FRACTION, edge: int = 0) -> np.ndarray:
    """The sky samples of a plane: the darkest ``fraction`` of its 64 px smoothing (the nebula and the stars are brighter than the sky at every large scale)."""
    sm = cv2.GaussianBlur(np.ascontiguousarray(plane, dtype=np.float32), (0, 0), 64.0)
    inner = sm[edge:-edge, edge:-edge] if edge and min(sm.shape) > 4 * edge else sm
    m = sm <= np.percentile(inner[::2, ::2], fraction * 100)
    if edge and min(sm.shape) > 4 * edge:
        m[:edge] = m[-edge:] = False
        m[:, :edge] = m[:, -edge:] = False
    return m





# --------------------------------------------------------------------------- the whole non-linear render

def is_narrowband(filter_name: str | None) -> bool:
    """A dual-band (Ha + OIII) filter: Seestar LP, DWARF Duo-Band, L-eNhance / L-eXtreme, ..."""
    import re
    n = re.sub(r"[^A-Z0-9]", "", (filter_name or "").upper())
    return n in {"LP", "DUO", "DUAL", "DUOBAND", "DUALBAND", "LENHANCE", "LEXTREME", "LULTIMATE", "HOO"}




                                  # than the Ha luminance (LRGB principle: the eye resolves lightness, not colour; chroma is binned 3-4x in video and JPEG alike); the OIII
                                  # noise left after the denoise is averaged over ~(2 sigma)^2 pixels, so the sky colour has no blotches

















def mean_saturation(rgb: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean HSV saturation (over the masked pixels)."""
    step = max(1, int(round(max(rgb.shape[1:]) / 1200)))
    hwc = np.ascontiguousarray(np.moveaxis(rgb[:, ::step, ::step], 0, -1), dtype=np.float32)
    s = cv2.cvtColor(hwc, cv2.COLOR_RGB2HSV)[..., 1]
    if mask is not None:
        m = mask[::step, ::step]
        if m.sum() > 50:
            return float(s[m].mean())
    return float(s.mean())



def screen(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Screen blend ``1 - (1-a)(1-b)`` = ``a + b - ab``: never above 1, the recombination RC-Astro and Brecher prescribe for two stretched layers."""
    return (a + b - a * b).astype(np.float32)


# =========================================================================== OKLab and the colour grade (the application's own finishing controls, restored)

def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0).astype(np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0).astype(np.float32)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055).astype(np.float32)


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566], [0.0883024619, 0.2817188376, 0.6299787005]], np.float32)
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099], [0.0259040371, 0.7827717662, -0.8086757660]], np.float32)
_M2i = np.linalg.inv(_M2.astype(np.float64)).astype(np.float32)
_M1i = np.linalg.inv(_M1.astype(np.float64)).astype(np.float32)


def rgb_to_oklab(rgb_hwc: np.ndarray) -> np.ndarray:
    """sRGB (H, W, 3) in 0..1 -> OKLab (Ottosson 2020)."""
    lin = _srgb_to_linear(rgb_hwc)
    lms = np.cbrt(cv2.transform(lin, _M1))
    return cv2.transform(lms.astype(np.float32), _M2)


def _oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    lms = cv2.transform(np.ascontiguousarray(lab, np.float32), _M2i)
    return cv2.transform(lms * lms * lms, _M1i)


def oklab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """OKLab -> sRGB (H, W, 3); out-of-gamut colours get their chroma reduced (hue and lightness kept) instead of a hard clip."""
    lin = _oklab_to_linear(lab)
    bad = ((lin < -1e-4) | (lin > 1 + 1e-4)).any(-1)
    if bad.any():
        sub = lab[bad].astype(np.float32)
        lo, hi = np.zeros(len(sub), np.float32), np.ones(len(sub), np.float32)
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            t = sub.copy()
            t[:, 1:] *= mid[:, None]
            ll = _oklab_to_linear(t[None])[0]
            ok = ((ll >= -1e-4) & (ll <= 1 + 1e-4)).all(-1)
            lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
        sub[:, 1:] *= lo[:, None]
        lin[bad] = _oklab_to_linear(sub[None])[0]
    return np.clip(_linear_to_srgb(lin), 0, 1).astype(np.float32)


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


GRADE_KEYS = ("grade_temperature", "grade_tint", "grade_warm_hue", "grade_warm_sat", "grade_cool_hue", "grade_cool_sat", "grade_contrast")
GRADE_NEUTRAL = {"grade_temperature": 0.0, "grade_tint": 0.0, "grade_warm_hue": 0.0, "grade_warm_sat": 1.0, "grade_cool_hue": 0.0, "grade_cool_sat": 1.0, "grade_contrast": 0.0}
_WARM_HUE, _COOL_HUE = np.deg2rad(40.0), np.deg2rad(220.0)        # OKLab hue: red ~25 deg, yellow ~110, cyan ~195, blue ~265


def grade_neutral(p: dict) -> bool:
    return float(p.get("grade_amount", 1.0)) <= 0 or all(abs(float(p.get(k, v)) - v) < 1e-4 for k, v in GRADE_NEUTRAL.items())


def color_grade(rgb: np.ndarray, p: dict) -> np.ndarray:
    """The colour grade of the finished image (3, H, W) in OKLab, as the application has always offered it: white balance of the object (temperature, tint), hue
    rotation (up to 25 deg) and chroma gain of the warm and of the cool hue sectors (raised cosines 180 deg wide centred on red-orange and on cyan-blue: every hue moves
    smoothly, greys not at all), and an S-curve of the object's lightness with its pivot at the object's median.  The sky (the darkest 15 %) is protected, so a neutral
    background stays neutral; ``grade_amount`` blends the result."""
    if grade_neutral(p):
        return rgb
    hwc = np.ascontiguousarray(np.moveaxis(np.clip(rgb, 0, 1), 0, -1), np.float32)
    lab = rgb_to_oklab(hwc)
    L = lab[..., 0]
    Ls = cv2.GaussianBlur(L, (0, 0), max(1.5, 0.002 * max(L.shape)))
    sky = float(np.percentile(Ls[::4, ::4], 15))
    obj = smoothstep(sky + 0.02, sky + 0.25, Ls).astype(np.float32)
    h = np.arctan2(lab[..., 2], lab[..., 1])
    C = np.hypot(lab[..., 1], lab[..., 2])
    w_warm = (np.clip(np.cos(h - _WARM_HUE), 0, 1) ** 1.5).astype(np.float32)
    w_cool = (np.clip(np.cos(h - _COOL_HUE), 0, 1) ** 1.5).astype(np.float32)
    h = h + np.deg2rad(25.0) * (float(p.get("grade_warm_hue", 0.0)) * w_warm + float(p.get("grade_cool_hue", 0.0)) * w_cool)
    gain = 1 + ((float(p.get("grade_warm_sat", 1.0)) - 1) * w_warm + (float(p.get("grade_cool_sat", 1.0)) - 1) * w_cool) * obj
    C = C * np.maximum(gain, 0)
    a, b = C * np.cos(h), C * np.sin(h)
    a = a + 0.025 * float(p.get("grade_tint", 0.0)) * obj
    b = b + 0.025 * float(p.get("grade_temperature", 0.0)) * obj
    k = float(np.clip(p.get("grade_contrast", 0.0), -1, 1)) * 1.5
    if abs(k) > 1e-4:                                             # S-curve above the sky: y = x + k x (1 - x)(x - pivot), monotone for |k| <= 1.5 with the pivot in [0.25, 0.6]
        x = np.clip((L - sky) / max(1 - sky, 1e-3), 0, 1)
        sel = obj[::4, ::4] > 0.5
        piv = float(np.clip(np.median(x[::4, ::4][sel]) if sel.sum() > 100 else 0.4, 0.25, 0.6))
        lab[..., 0] = np.where(L > sky, sky + (1 - sky) * (x + k * x * (1 - x) * (x - piv)), L)
    lab[..., 1], lab[..., 2] = a, b
    out = oklab_to_rgb(lab)
    amt = float(np.clip(p.get("grade_amount", 1.0), 0, 1))
    res = hwc + (out - hwc) * amt if amt < 1 else out
    return np.ascontiguousarray(np.moveaxis(res, -1, 0), np.float32)
