"""Non-linear finishing math in numpy: the parts Siril has no command for, and the preview path.

What lives here (all on ``(3, H, W)`` float32 arrays, rows in the FITS row order of the input):

* the **statistical stretch** of Seti Astro's ``Statistical_Stretch.py`` (ported by C. Richard into
  siril-scripts): black point ``median - sigma * noise`` (noise = 1.4826 x MAD of the lower half),
  then the rational midtones function that sends the median to the target; extended with the
  asinh and log families (each solved to hit the same median) so ``hdr`` and ``stretch_method``
  can blend towards more highlight compression without moving the sky;
* dual-band palettes (HOO / foraxx / hoo_warm) composed from the Ha and OIII channels of an OSC
  RGB stack, with OSC_Extract_HaOIII's median/MAD renormalisation of OIII to Ha;
* the star layer stretch (asinh) and the screen recombination ``1 - (1-a)(1-b)``;
* numpy equivalents of the Siril commands the export uses (``satu``, ``rmgreen``, ``mtf``) so a
  preview and a Siril-less environment give the same picture to within rounding;
* small detail tools (chroma smoothing, unsharp mask, CLAHE blend) and the image statistics the
  auto-finish rules read.

Why numpy and not Siril for the stretch: the statistic that drives it (median of the starless
layer) must be measured on the layer being stretched, per channel or on a luminance, and the same
code has to run in the web UI's fast preview.
"""
from __future__ import annotations

import cv2
import numpy as np

# Rec. 709 luminance weights (Statistical_Stretch's luminance mode uses the same ones).
LUM709 = (0.2126, 0.7152, 0.0722)

# Statistical_Stretch.py defaults (siril-scripts/processing/Statistical_Stretch.py): sigma for the
# black point, target median clamp.
STAT_SIGMA = 5.0
TARGET_RANGE = (0.01, 0.99)

# Foraxx-type and bicolour palettes come from the project's earlier postprocess.py (they are the
# palettes the UI always offered); the HOO mapping is Hubble_Palette_from_Dual-Band_OSC.py's
# ``rgbcomp HA OIII OIII`` (workflows.md A.9).
PALETTES = ("natural", "hoo", "foraxx", "hoo_warm")
# Seti Astro "Star Stretch" (Franklin Marek's PixInsight script, setiastro.com/pjsr-scripts): the linear star layer goes through the
# midtones-type rational ``k x / ((k - 1) x + 1)`` with ``k = 3^amount``, amount 0..8, default 5 (workflows.md 7.5).  The amount is a
# setting (``star_stretch``); autofinish measures it (star_stretch_amount).
STAR_STRETCH = 5.0
STAR_STRETCH_RANGE = (2.0, 8.0)
# Autofinish aim: the median detected star (peak above 5 sigma of the sky) lands at this display level of the star layer, the way the
# telescope's own stacks show it (measured on M 42: rank 51-550 stars sit at 0.32 above the sky); the brightest unsaturated ones then
# approach 1 by the rational's own shoulder.
STAR_MEDIAN_AIM = 0.3
STAR_DETECT_SIGMA = 5.0
REDUCE_NEIGHBOURHOOD = 15         # px: a star layer pixel is "near a bright star" when a pixel >= REDUCE_BRIGHT lies within this box
REDUCE_BRIGHT = 0.3
HALO_KERNEL = 4.0            # opening disc diameter = this x the stars' FWHM (suppress_halos)
# Black point of the linear star layer before its asinh stretch: 2.5 sigma of the layer's own noise (the 0.001-0.003 of Under South West Skies,
# workflows.md 7.5, is for a stack with ~1e-3 noise; as a fixed floor 4e-4 it cut every star below ~6 sigma on M 27 / M 42, whose star layer has
# a noise of 1e-6).  The floor is only a numerical epsilon; the PSF wings are handled by suppress_halos.
STAR_BP_MIN = 1e-5
# Highlight chroma of a dual-band picture (Seti Astro HDR: "a lower stretch of the same data, blended through a luminosity mask"): both
# lines stretched to the 0.25 target sit near 1 together in a bright core, and HOO's red / teal collapses into white (M 42: the core's
# saturation was 0.02).  The same Ha and OIII stretched to this lower median keep their ratio there; the lower stretch contributes its
# COLOUR only (its lightness is set to the main stretch's), through a mask that ramps over the main picture's smoothed lightness.
CORE_TARGET = 0.04
CORE_MASK = (0.45, 0.75)              # smoothed lightness of the main stretch where the mask goes 0 -> 1
CORE_MASK_SIGMA = 8.0                 # px, smoothing of the lightness the mask is taken from (a mask of the structure, not of the noise)
# "Improved" OIII of Hubble_Palette_from_Dual-Band_OSC.py: (G + B - 0.3 R) / 2 -> k = 0.3
MAX_HA_LEAK = 0.3


# --------------------------------------------------------------------------- statistics

def luminance(rgb: np.ndarray, weights=LUM709) -> np.ndarray:
    """Weighted luminance of a (3, H, W) array."""
    return rgb[0] * np.float32(weights[0]) + rgb[1] * np.float32(weights[1]) + rgb[2] * np.float32(weights[2])


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


def bxt_sn(snr: float) -> float:
    """BlurXTerminator non-stellar sharpening: 0.3 on noisy data (SNR <= 10), 0.75 on clean data (>= 50); RC-Astro: lower on noisy data."""
    return round(snr_ladder(snr, 0.30, 0.75), 3)


def nxt_dn(snr: float) -> float:
    """NoiseXTerminator ``--dn``: 0.9 on noisy data, 0.6 on clean data (RC-Astro's 0.6-0.9)."""
    return round(snr_ladder(snr, 0.90, 0.60), 3)


def graxpert_strength(snr: float) -> float:
    """GraXpert AI denoise strength (the stand-in for NXT): 1.0 on noisy data, 0.6 on clean data."""
    return round(snr_ladder(snr, 1.0, 0.6), 3)


def black_point(ch_sample: np.ndarray, sigma: float = STAT_SIGMA) -> float:
    """``max(min, median - sigma * noise)`` capped at 0.99 (Statistical_Stretch ``_compute_blackpoint_sigma``)."""
    med = float(np.median(ch_sample))
    bp = max(float(ch_sample.min()), med - sigma * lower_half_sigma(ch_sample))
    return float(min(bp, 0.99))


# --------------------------------------------------------------------------- the stretch curves

def rational(x: np.ndarray, m: float, t: float) -> np.ndarray:
    """Statistical_Stretch's midtones function: sends the median ``m`` to ``t`` and 0 -> 0, 1 -> 1.
    ``(m-1) t x / (m (t + x - 1) - t x)``."""
    x = np.asarray(x, np.float32)
    den = np.float32(m) * (np.float32(t) + x - 1.0) - np.float32(t) * x
    den = np.where(np.abs(den) < 1e-12, np.float32(-1e-12), den)
    return (np.float32((m - 1.0) * t) * x) / den


def _solve_beta(kind: str, m: float, t: float) -> float:
    """beta such that f(m) = t for f = asinh(beta x)/asinh(beta) or log1p(beta x)/log1p(beta)."""
    g = np.arcsinh if kind == "asinh" else np.log1p
    lo, hi = -7.0, 9.0                                   # log10 beta
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        b = 10.0 ** mid
        if g(b * m) / g(b) < t:
            lo = mid
        else:
            hi = mid
    return 10.0 ** (0.5 * (lo + hi))


# Highlight shoulder of the ``hdr`` setting ("HDR via a masked lower stretch", the community recipe for M 42 / M 31 cores; the Statistical_Stretch
# HDR compress, amount 0.15 / knee 0.75, is the same idea).  The plain rational stretch of a faint nebula (median ~ 1e-3) already reaches
# 0.5 at three times the sky, and from there every brighter pixel is squeezed into the last 0.1 of the range (a white core).  Above ``HDR_KNEE``
# of the plain stretch the picture is blended smoothly (smoothstep) into the log stretch of the same median: the sky and the faint nebula keep
# the full midtone contrast, only the bright end gets room.  0.4 is the highest knee for which the blend stays monotone at full strength
# (tests/test_finish.py checks it).  ``hdr`` = 1.2 is full strength.
HDR_KNEE = 0.4
HDR_FULL = 1.2


class Curve:
    """A monotone map [0,1] -> [0,1] with ``curve(m) == t``: the rational midtones function (position 0),
    asinh (1) or log (2), or a blend of two neighbours.  Every member passes through (m, t), so blending
    them never moves the sky level; it only moves the highlights down (more room in bright cores).
    ``shoulder`` (0..1) additionally rolls the *bright end only* of the rational curve into the log curve (see ``HDR_KNEE``)."""

    def __init__(self, m: float, t: float, position: float = 0.0, shoulder: float = 0.0):
        self.m, self.t = float(m), float(t)
        self.shoulder = float(np.clip(shoulder, 0.0, 1.0))
        self.position = float(np.clip(position, 0.0, 2.0))
        self.exact = t > m * 1.0001 and m > 0                # asinh / log can only brighten
        pos = self.position if self.exact else 0.0
        if pos <= 1.0:
            self.a, self.b, self.w = "mtf", "asinh", pos
        else:
            self.a, self.b, self.w = "asinh", "log", pos - 1.0
        self.beta = {k: _solve_beta(k, self.m, self.t) for k in {self.a, self.b} if k != "mtf"
                     and (self.w > 0 or k == self.a)}
        self._shoulder_on = self.exact and self.shoulder > 0 and self.position <= 0.0
        if self._shoulder_on:
            self.beta["log"] = _solve_beta("log", self.m, self.t)

    def _one(self, kind: str, x: np.ndarray) -> np.ndarray:
        if kind == "mtf":
            return rational(x, self.m, self.t)
        b = np.float32(self.beta[kind])
        if kind == "asinh":
            return np.arcsinh(b * x) / np.float32(np.arcsinh(b))
        return np.log1p(b * x) / np.float32(np.log1p(b))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self._shoulder_on:
            u = self._one("mtf", x)
            wgt = np.clip((u - np.float32(HDR_KNEE)) / np.float32(1.0 - HDR_KNEE), 0.0, 1.0)
            wgt = wgt * wgt * (3.0 - 2.0 * wgt) * np.float32(self.shoulder)
            return u - wgt * (u - self._one("log", x))
        if self.w <= 0:
            return self._one(self.a, x)
        if self.w >= 1:
            return self._one(self.b, x)
        return (1.0 - self.w) * self._one(self.a, x) + self.w * self._one(self.b, x)

    def info(self) -> dict:
        return {"median_in": round(self.m, 6), "target": self.t, "position": round(self.position, 3), "shoulder": round(self.shoulder, 3),
                "beta": {k: round(float(v), 3) for k, v in self.beta.items()}}


# HDR highlight compress of the GHS picture: Statistical_Stretch.py's ``hdr_compress_color_luminance`` (Seti Astro / C. Richard, siril-scripts), ported as is: above
# the knee the Rec. 709 lightness follows a cubic Hermite from (knee, knee) to (1, 1) that starts with slope 1 and ends with slope ``1 + 4 amount`` (the white end
# gets room: bright cores keep a gradation instead of a flat plateau), and the colour is carried by the ratio new / old lightness with the script's soft knee 0.25
# (``s / (1 + 0.25 (s - 1))``: a darkened highlight keeps its saturation).  ``hdr`` (0..1.5 in the settings) maps to the script's 0..1 amount.
HDR_KNEE_GHS = 0.75
HDR_SOFT_KNEE = 0.25
HDR_SETTING_MAX = 1.5


def hdr_compress_highlights(x: np.ndarray, amount: float, knee: float = HDR_KNEE_GHS) -> np.ndarray:
    """Statistical_Stretch.hdr_compress_highlights on a lightness plane."""
    a = float(np.clip(amount, 0.0, 1.0))
    y = np.asarray(x, np.float32)
    if a <= 0.0:
        return y
    k = float(np.clip(knee, 0.0, 0.99))
    hi = y > k
    if not hi.any():
        return np.clip(y, 0.0, 1.0)
    t = np.clip((y[hi] - k) / (1.0 - k), 0.0, 1.0)
    m1 = float(np.clip(1.0 + 4.0 * a, 1.0, 5.0))
    t2 = t * t
    t3 = t2 * t
    f = (t3 - 2.0 * t2 + t) + (-2.0 * t3 + 3.0 * t2) + (t3 - t2) * m1
    out = y.copy()
    out[hi] = k + (1.0 - k) * np.clip(f, 0.0, 1.0)
    return np.clip(out, 0.0, 1.0)


def hdr_compress(rgb: np.ndarray, setting: float, knee: float = HDR_KNEE_GHS) -> np.ndarray:
    """``hdr_compress_color_luminance`` of a (3, H, W) picture; ``setting`` is the ``hdr`` parameter (0 = off)."""
    a = float(np.clip(float(setting) / HDR_SETTING_MAX, 0.0, 1.0))
    if a <= 0.0:
        return rgb
    out = np.empty_like(rgb, dtype=np.float32)
    for r0, r1 in _bands(rgb.shape[1]):
        b = rgb[:, r0:r1]
        y = luminance(b)
        sc = hdr_compress_highlights(y, a, knee) / (y + np.float32(1e-6))
        sc = sc / (1.0 + np.float32(HDR_SOFT_KNEE) * (sc - 1.0))
        out[:, r0:r1] = np.clip(b * sc[None], 0.0, 1.0)
    return out


def sky_lift(rgb: np.ndarray, aim: float) -> tuple[np.ndarray, float]:
    """Output black level: the display level of the sky (median of the darkest 45 % of the 2 px smoothed lightness) is brought up to ``aim`` with
    ``y' = a + (1 - a) y`` (Levels' output black slider; never darkens).  The GHS puts the *mode* of the plane on the sky aim, but a frame whose Ha plateau covers
    the field has dark lanes well below the mode (IC 1805: the darkest 5 % at 0.004, pure black); the user's sky aim 0.07-0.10 is for the sky, not for the mode."""
    lum = cv2.GaussianBlur(np.ascontiguousarray(luminance(rgb)), (0, 0), 2.0)
    smp = sample(lum, valid_only=False)
    m = float(np.median(smp[smp <= np.percentile(smp, 45)]))
    a = float(np.clip((aim - m) / max(1.0 - m, 1e-6), 0.0, 0.5))
    if a <= 1e-4:
        return rgb, 0.0
    return (np.float32(a) + np.float32(1.0 - a) * rgb).astype(np.float32), a


def curve_position(stretch_method: str, hdr: float) -> float:
    """Where on the mtf -> asinh -> log axis the stretch sits: the method picks the base; for the asinh / log methods ``hdr`` (0..1.5)
    moves it towards stronger highlight compression.  The statistical (ghs / mtf) stretch keeps the rational body and takes its
    highlight protection as a shoulder instead (``curve_shoulder``)."""
    base = {"ghs": 0.0, "mtf": 0.0, "asinh": 1.0, "log": 2.0}.get(str(stretch_method), 0.0)
    return float(np.clip(base + float(hdr), 0.0, 2.0)) if base > 0 else 0.0


def curve_shoulder(stretch_method: str, hdr: float) -> float:
    """Strength 0..1 of the highlight shoulder (``hdr`` / HDR_FULL) for the rational (ghs / mtf) stretch, 0 for asinh / log."""
    if str(stretch_method) in ("asinh", "log"):
        return 0.0
    return float(np.clip(float(hdr) / HDR_FULL, 0.0, 1.0))


def _bands(h: int, rows: int = 512):
    for r in range(0, h, rows):
        yield r, min(h, r + rows)


GAMUT_DARKEN = 0.5      # share of the overflow resolved by darkening (hue and saturation kept) instead of pulling towards the lightness (desaturating): a rich red stays red


def _gamut_map(cp: np.ndarray, ls: np.ndarray, darken: float = 0.0) -> np.ndarray:
    """When a channel exceeds 1 pull the pixel towards its luminance just enough to fit (dividing by the maximum would darken saturated highlights and ring coloured
    star halos).  ``darken`` (0..1) resolves that share of the overflow the other way, by scaling the pixel down (hue and saturation kept, lightness drops): a bright
    Ha-red nebula with a plain pull went salmon-pink with a flat red channel (NGC 1499: 13 % of the pixels at R = 1), a half-and-half mapping keeps its colour."""
    mx = cp.max(0)
    over = mx > 1.0
    if not over.any():
        return cp
    k = np.clip((1.0 - ls) / np.maximum(mx - ls, 1e-6), 0.0, 1.0)
    pulled = ls[None] + (cp - ls[None]) * k[None]
    if darken > 0:
        d = float(np.clip(darken, 0.0, 1.0))
        scaled = cp / np.maximum(mx, 1e-6)[None]
        pulled = (1.0 - d) * pulled + d * scaled
    return np.where(over[None], pulled, cp)


def stretch_rgb(rgb: np.ndarray, target: float, *, position: float = 0.0, shoulder: float = 0.0, color_preservation: float = 0.7,
                sigma: float = STAT_SIGMA, weights=LUM709) -> tuple[np.ndarray, dict]:
    """Statistically stretch ``rgb`` so its median lands on ``target``.

    ``color_preservation`` blends two routes: 0 = each channel stretched on its own (backgrounds
    become exactly equal: a neutral sky by construction, colours drift towards white in bright
    areas), 1 = one stretch of the luminance applied as a ratio (hue and saturation of the linear
    data survive, sky neutral because the channel backgrounds are first brought to a common level)."""
    target = float(np.clip(target, *TARGET_RANGE))
    cp = float(np.clip(color_preservation, 0.0, 1.0))
    _, h, w = rgb.shape
    smp = [sample(rgb[c]) for c in range(3)]
    n = min(len(s) for s in smp)
    smp = [s[:n] for s in smp]
    med = [float(np.median(s)) for s in smp]
    bps = [black_point(s, sigma) for s in smp]
    info: dict = {"target": target, "sigma": sigma, "color_preservation": cp, "medians": [round(m, 6) for m in med]}
    per_curves = None
    if cp < 1.0:
        per_curves = [Curve(max((med[c] - bps[c]) / (1 - bps[c]), 1e-6), target, position, shoulder) for c in range(3)]
        info["black_points"] = [round(b, 6) for b in bps]
        info["per_channel"] = per_curves[1].info()
    lum_curve = off = None
    if cp > 0.0:
        s_common = sigma * float(np.mean([lower_half_sigma(s) for s in smp]))
        off = [min(med[c] - s_common, 0.99) for c in range(3)]
        xs = [np.clip((smp[c] - off[c]) / (1 - off[c]), 0, None) for c in range(3)]
        ls = weights[0] * xs[0] + weights[1] * xs[1] + weights[2] * xs[2]
        lum_curve = Curve(max(float(np.median(ls)), 1e-6), target, position, shoulder)
        info["luminance"] = lum_curve.info()
    out = np.empty_like(rgb, dtype=np.float32)
    for r0, r1 in _bands(h):
        b = rgb[:, r0:r1]
        per = ratio = None
        if per_curves:
            per = np.stack([per_curves[c](np.clip((b[c] - bps[c]) / (1 - bps[c]), 0, None)) for c in range(3)])
        if lum_curve:
            x = np.stack([np.clip((b[c] - off[c]) / (1 - off[c]), 0, None) for c in range(3)])
            lx = x[0] * np.float32(weights[0]) + x[1] * np.float32(weights[1]) + x[2] * np.float32(weights[2])
            lsx = np.clip(lum_curve(lx), 0, 1)
            ratio = _gamut_map(x * (lsx / np.maximum(lx, np.float32(1e-6)))[None], lsx)
        if per is None:
            res = ratio
        elif ratio is None:
            res = per
        else:
            res = cp * ratio + (1.0 - cp) * per
        out[:, r0:r1] = np.clip(res, 0.0, 1.0)
    return out, info


def stretch_channel(ch: np.ndarray, target: float, *, position: float = 0.0, shoulder: float = 0.0, sigma: float = STAT_SIGMA) -> tuple[np.ndarray, dict]:
    """The same statistical stretch for one plane (narrowband channels are stretched one by one so
    their medians match, the way astroaf.space / Siril's HOO tutorial do before the colour mix)."""
    target = float(np.clip(target, *TARGET_RANGE))
    s = sample(ch)
    bp = black_point(s, sigma)
    cur = Curve(max((float(np.median(s)) - bp) / (1 - bp), 1e-6), target, position, shoulder)
    out = np.empty_like(ch, dtype=np.float32)
    for r0, r1 in _bands(ch.shape[0]):
        out[r0:r1] = np.clip(cur(np.clip((ch[r0:r1] - bp) / (1 - bp), 0, None)), 0, 1)
    return out, {"black_point": round(bp, 6), **cur.info()}


# --------------------------------------------------------------------------- dual-band palette

def extract_ha_oiii(lin: np.ndarray, unmix: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """Ha and OIII from the linear RGB of a dual-band OSC stack: Ha = R, OIII = (G + B) / 2
    (Hubble_Palette_from_Dual-Band_OSC.py "Classic"), optionally minus the Ha that leaks into the
    G and B pixels, then OIII is matched to Ha with OSC_Extract_HaOIII's median / MAD renormalisation:
    ``O' = median(H) + (O - median(O)) mad(H) / mad(O)`` (workflows.md section 8).  (The OIII *boost* is applied
    after the stretch: every channel is stretched to the same sky median, which cancels any linear gain.)

    With ``unmix`` the Ha that leaks into the G and B pixels is removed with the fixed constant of the "Improved" formula
    of Hubble_Palette_from_Dual-Band_OSC.py, ``OIII = (G + B - 0.3 R) / 2`` (MAX_HA_LEAK; the sky level is kept).  A leak
    measured from the data (lower envelope of OIII / Ha, capped at 0.3 of the *mean* G, B = twice the formula) took the blue out
    of the whole M 42 core, so it is not used."""
    ha = lin[0].astype(np.float32)
    oiii = 0.5 * (lin[1] + lin[2])
    info: dict = {"unmix": bool(unmix)}
    hb, ob = float(np.median(sample(ha))), float(np.median(sample(oiii)))
    k = 0.0
    if unmix:
        k = 0.5 * MAX_HA_LEAK
        oiii = oiii - k * (ha - hb)
    info["ha_leak_k"] = round(k, 4)
    ho = sample(ha)
    oo = sample(oiii)
    mad_h = max(float(np.median(np.abs(ho - np.median(ho)))), 1e-9)
    mad_o = max(float(np.median(np.abs(oo - np.median(oo)))), 1e-9)
    gain = mad_h / mad_o
    med_h, med_o = float(np.median(ho)), float(np.median(oo))
    oiii = (med_h + (oiii - med_o) * gain).astype(np.float32)
    info.update(gain=round(gain, 4), median_ha=round(med_h, 6), mad_ha=round(mad_h, 6))
    return ha, np.maximum(oiii, 0).astype(np.float32), info


def compose_palette(ha: np.ndarray, oiii: np.ndarray, palette: str) -> np.ndarray:
    """RGB (3, H, W) from stretched Ha and OIII (0..1).  The palette's own colour is kept as is:
    nothing is desaturated by how significant a line is (project rule)."""
    if palette == "foraxx":
        # Foraxx dynamic palette: green is a Ha/OIII mix weighted by (Ha*OIII)^(1-Ha*OIII)
        p = np.clip(ha * oiii, 1e-6, 1.0)
        ho = np.power(p, 1.0 - p)
        return np.stack([ha, ho * ha + (1.0 - ho) * oiii, oiii])
    if palette == "hoo_warm":
        # Ha -> red/orange, OIII -> teal; the "natural-ish" bicolour (G = .35 Ha + .65 OIII)
        return np.stack([ha, 0.35 * ha + 0.65 * oiii, 0.9 * oiii + 0.1 * ha])
    return np.stack([ha, oiii, oiii])                        # HOO


def core_chroma(main: np.ndarray, alt: np.ndarray) -> tuple[np.ndarray, dict]:
    """Blend the colour of ``alt`` (the lower stretch) into the bright regions of ``main``, keeping ``main``'s lightness: where the
    mask (smoothstep of the smoothed lightness of ``main`` over CORE_MASK) is 1 the pixel is ``alt`` scaled to ``main``'s lightness."""
    lm, la = luminance(main), luminance(alt)
    sm = cv2.GaussianBlur(np.ascontiguousarray(lm), (0, 0), CORE_MASK_SIGMA)
    w = np.clip((sm - CORE_MASK[0]) / (CORE_MASK[1] - CORE_MASK[0]), 0.0, 1.0)
    w = (w * w * (3.0 - 2.0 * w)).astype(np.float32)
    out = np.empty_like(main, dtype=np.float32)
    for r0, r1 in _bands(main.shape[1]):
        a = alt[:, r0:r1] * (lm[r0:r1] / np.maximum(la[r0:r1], 1e-4))[None]
        a = _gamut_map(a, np.clip(lm[r0:r1], 0, 1))
        out[:, r0:r1] = main[:, r0:r1] * (1.0 - w[None, r0:r1]) + a * w[None, r0:r1]
    return np.clip(out, 0, 1), {"mask_px": int((w > 0.5).sum()), "target": CORE_TARGET}


def synthetic_luminance(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Narrowband colour images look dark where OIII (blue/teal) dominates, because Rec.709 luminance
    barely counts blue.  Replace the lightness by the all-channel mean by ``amount`` (0..1): each
    pixel is scaled so its Rec.709 lightness moves towards the channels' mean."""
    amount = float(np.clip(amount, 0, 1))
    if amount <= 0:
        return rgb
    out = np.empty_like(rgb)
    for r0, r1 in _bands(rgb.shape[1]):
        b = rgb[:, r0:r1]
        lc = luminance(b)
        lt = b.mean(0)
        scale = (1.0 - amount) + amount * lt / np.maximum(lc, 1e-6)
        scale = np.clip(scale, 0.0, 4.0)
        c = b * scale[None]
        out[:, r0:r1] = _gamut_map(c, np.clip(lt * amount + lc * (1 - amount), 0, 1))
    return np.clip(out, 0, 1)


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

def screen(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Screen blend ``1 - (1-a)(1-b)`` = ``a + b - ab``: never above 1, the recombination the
    RC-Astro and Siril guides prescribe for two stretched layers (workflows.md 7.5)."""
    out = np.empty_like(a, dtype=np.float32)
    for r0, r1 in _bands(a.shape[1]):
        out[:, r0:r1] = 1.0 - (1.0 - a[:, r0:r1]) * (1.0 - b[:, r0:r1])
    return out


def unscreen(full: np.ndarray, starless: np.ndarray) -> np.ndarray:
    """The star layer ``s`` with ``screen(starless, s) == full``: ``(full - starless) / (1 - starless)``
    (what ``starnet2 -n`` / ``sxt --unscreen`` produce; for stretched data only)."""
    return np.clip((full - starless) / np.maximum(1.0 - starless, 1e-4), 0.0, 1.0).astype(np.float32)


def star_noise_floor(stars: np.ndarray) -> float:
    """sigma of the residual noise inside a star layer ``linear - starless`` (clipped at 0): the stars
    are a tiny fraction of the pixels, so the median of the positive values is the half-normal
    median ``0.6745 sigma`` of the noise."""
    pos = sample(stars.max(0), valid_only=False)
    pos = pos[pos > 0]
    return float(np.median(pos) / 0.6745) if pos.size > 200 else 0.0


def star_color_balance(stars: np.ndarray, noise_floor: float = 0.0) -> tuple[np.ndarray, dict]:
    """Channel gains for the LINEAR star layer of a dual-band image so the average star is neutral.

    A Duo-Band / LP filter passes only Ha (R) and OIII (G, B): SPCC's narrowband mode calibrates the lines, not the stars'
    broadband colour, and measured on M 42 every star came out cyan (flux R : G : B = 0.5 : 1 : 0.9).  The same
    remedy as PCC with the average star colour (Siril ``pcc`` / SPCC with a G2V white reference): the median
    colour of the unsaturated stars (each measured in a 7 x 7 aperture on the star layer, which is already
    sky-free) is taken as white; gains are G = 1 and applied to the star layer only, never to the nebula.
    Stars keep their relative colours (hot stars stay bluer than the median)."""
    lum = cv2.GaussianBlur(np.ascontiguousarray(stars.max(0)), (0, 0), 1.0)
    dil = cv2.dilate(lum, np.ones((9, 9), np.uint8))
    lo = max(6.0 * noise_floor, 3e-3)
    ys, xs = np.nonzero((lum == dil) & (lum > lo) & (lum < 0.5))
    if len(ys) < 12:
        return stars, {"gains": [1.0, 1.0, 1.0], "stars": int(len(ys))}
    h, w = lum.shape
    keep = (ys > 3) & (xs > 3) & (ys < h - 4) & (xs < w - 4)
    ys, xs = ys[keep], xs[keep]
    ap = np.stack([np.array([stars[c, y - 3:y + 4, x - 3:x + 4].sum() for y, x in zip(ys, xs)]) for c in range(3)], 1)
    ok = ap[:, 1] > 0
    r = np.median(ap[ok, 0] / ap[ok, 1]) if ok.sum() >= 12 else 1.0
    b = np.median(ap[ok, 2] / ap[ok, 1]) if ok.sum() >= 12 else 1.0
    gains = np.array([1.0 / max(r, 0.05), 1.0, 1.0 / max(b, 0.05)], np.float32)
    gains = np.clip(gains, 0.25, 4.0)
    out = stars * gains[:, None, None]
    return out.astype(np.float32), {"gains": [round(float(g), 3) for g in gains], "stars": int(ok.sum())}


def clipped_core_weight(stars: np.ndarray) -> np.ndarray | None:
    """Weight (H, W) in 0..1 of the pixels where every channel of the linear star layer sits at its own top: a saturated core.
    After colour calibration each channel clips at a different level (SPCC scaled G by 0.52 on the M 42 stack, so the clip is
    R 0.92 / G 0.53 / B 0.89), and a core that clips at those levels has no colour information left: it came out magenta.
    The levels are measured (channel maxima, accepted only when at least 3 pixels sit within 10 % of them: a real pile-up),
    the weight is the smooth ramp 0.75 -> 0.95 of the lowest channel / level.  None when no channel clips."""
    top = [float(stars[c].max()) for c in range(3)]
    if min(top) <= 0.05:
        return None
    for c in range(3):
        if int((stars[c] >= 0.9 * top[c]).sum()) < 3:
            return None
    frac = np.min(np.stack([stars[c] / top[c] for c in range(3)]), axis=0)
    wgt = np.clip((frac - 0.75) / 0.2, 0.0, 1.0)
    return wgt if wgt.max() > 0 else None


def suppress_halos(stars: np.ndarray, fwhm_px: float, amount: float) -> tuple[np.ndarray, float]:
    """Remove ``amount`` (0..1) of the smooth wing baseline of every star in the LINEAR star layer.

    A star's scattered-light wing is the slowly varying part of the layer; the star itself is the narrow peak on top.  The
    baseline is the morphological opening of the luminance with a disc of ``HALO_KERNEL x FWHM`` (it removes everything
    narrower than the disc: a white top-hat is ``x - opening``), smoothed.  Cutting it at a constant black point instead left
    every bright star inside a flat disc with a hard rim after the asinh stretch (measured on M 31 / M 45 / M 42); taking
    the baseline out *proportionally* keeps the rim soft and leaves the faint stars (narrower than the disc) alone.  The
    light is only left out of the display, never moved or invented.  Returns the layer and the disc diameter in px."""
    amount = float(np.clip(amount, 0.0, 1.0))
    k = int(max(9, round(HALO_KERNEL * max(float(fwhm_px), 1.5))))
    k += 1 - k % 2
    if amount <= 0:
        return stars, k
    lum = np.ascontiguousarray(stars.max(0))
    base = cv2.morphologyEx(lum, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    base = np.minimum(cv2.GaussianBlur(base, (0, 0), k / 4.0), lum)
    frac = np.clip(amount * base / np.maximum(lum, 1e-9), 0.0, 1.0)
    return (stars * (1.0 - frac)[None]).astype(np.float32), k


def star_peaks(stars: np.ndarray, noise_sigma: float, min_sigma: float = STAR_DETECT_SIGMA) -> np.ndarray:
    """Peak levels (linear) of the detectable stars of a LINEAR star layer: local maxima of the lightly smoothed luminance
    above ``min_sigma`` x the sky's pixel noise ``noise_sigma`` (the detection of ``count_stars``), brightest first."""
    if noise_sigma <= 0:
        return np.zeros(0, np.float32)
    lum = cv2.GaussianBlur(np.ascontiguousarray(stars.max(0)), (0, 0), 1.0)
    dil = cv2.dilate(lum, np.ones((7, 7), np.uint8))
    return np.sort(lum[(lum == dil) & (lum > min_sigma * noise_sigma)])[::-1]


def star_stretch_amount(stars: np.ndarray, noise_sigma: float, aim: float = STAR_MEDIAN_AIM) -> tuple[float, dict]:
    """The Star Stretch amount that sends the median detected star to ``aim``: ``k x / ((k - 1) x + 1) = aim`` gives
    ``k = aim (1 - x) / (x (1 - aim))`` and the amount is ``log3 k``, kept inside the script's useful range.  With fewer than
    20 detectable stars the script's default (5) stands.  Also reports where the brightest unsaturated star (the 98th
    percentile of the peaks below 0.9) lands."""
    pk = star_peaks(stars, noise_sigma)
    if pk.size < 20:
        return STAR_STRETCH, {"stars": int(pk.size), "median_peak": None}
    med = float(np.median(pk))
    k = aim * (1.0 - med) / (max(med, 1e-7) * (1.0 - aim))
    amount = float(np.clip(np.log(max(k, 1.0)) / np.log(3.0), *STAR_STRETCH_RANGE))
    kk = 3.0 ** amount
    unsat = pk[pk < 0.9]
    top = float(np.percentile(unsat, 98)) if unsat.size else float(pk[0])
    return round(amount, 2), {"stars": int(pk.size), "median_peak": round(med, 6), "median_after": round(kk * med / ((kk - 1) * med + 1), 3),
                              "bright_peak": round(top, 4), "bright_after": round(kk * top / ((kk - 1) * top + 1), 3)}


def stretch_stars(stars: np.ndarray, *, color_preservation: float = 0.5, halo_suppress: float = 0.6,
                  amount: float = STAR_STRETCH, fwhm_px: float = 3.0) -> tuple[np.ndarray, dict]:
    """Seti Astro Star Stretch of a linear star layer: ``k x / ((k - 1) x + 1)``, ``k = 3^amount`` (workflows.md 7.5), after a black
    point of 2.5 sigma of the layer's own noise (at least STAR_BP_MIN, so no grey fuzz) and ``suppress_halos``.  ``color_preservation``
    mixes the per-channel result with the one that applies the luminance's gain to all channels (the star's colour survives the
    stretch; Siril's "independent vs. human" choice).  Saturated cores (every channel at its clip level) go to white, brightness
    kept.  The star layer is then scaled (reduce_stars: reduction, intensity) and screen-blended onto the starless picture."""
    stars_full = stars
    stars, halo_k = suppress_halos(stars, fwhm_px, halo_suppress)
    sig = star_noise_floor(stars)
    bp = float(max(2.5 * sig, STAR_BP_MIN))
    amount = float(np.clip(amount, 0.0, 12.0))
    k = np.float32(3.0 ** amount)
    km1 = k - np.float32(1.0)
    cp = float(np.clip(color_preservation, 0, 1))
    out = np.empty_like(stars, dtype=np.float32)
    for r0, r1 in _bands(stars.shape[1]):
        x = np.clip(stars[:, r0:r1] - np.float32(bp), 0, None)
        per = k * x / (km1 * x + np.float32(1.0))
        if cp > 0:
            lx = np.maximum(x.max(0), np.float32(1e-9))
            lsx = k * lx / (km1 * lx + np.float32(1.0))
            res = cp * x * (lsx / lx)[None] + (1 - cp) * per
        else:
            res = per
        out[:, r0:r1] = np.clip(res, 0, 1)
    wc = clipped_core_weight(stars_full)
    n_core = 0
    if wc is not None:                                    # saturated cores go to white (their brightness is kept)
        n_core = int((wc > 0.5).sum())
        grey = out.mean(0)
        out = (out * (1.0 - wc[None]) + grey[None] * wc[None]).astype(np.float32)
    return out, {"amount": round(amount, 2), "black_point": round(bp, 6), "noise_sigma": round(sig, 6), "halo_kernel_px": halo_k,
                 "clipped_core_px": n_core}


def reduce_stars(st: np.ndarray, reduction: float, intensity: float = 1.0) -> tuple[np.ndarray, float]:
    """Star reduction of a STRETCHED star layer: ``reduction`` 0..1 (1 = no stars).  Reducing means smaller stars, not
    dimmer ones everywhere: the layer's black point is raised by ``0.12 x reduction`` (the footprints shrink, the
    cores stay: "raise the black point on the star layer to shrink the stars", Under South West Skies, workflows.md 7.5)
    and the layer fades out as ``1 - reduction^3`` so that 1.0 really is starless.  ``intensity`` multiplies last.
    The raised black point acts where it shrinks something: around bright stars (a pixel >= ``REDUCE_BRIGHT`` within
    ``REDUCE_NEIGHBOURHOOD`` px, smooth ramp); an isolated faint star is a core without a footprint and keeps its level
    (a black point of 0.03-0.05 on the whole layer deleted every star with a stretched peak below ~0.08)."""
    r = float(np.clip(reduction, 0.0, 1.0))
    bp = 0.12 * r
    gain = (1.0 - r ** 3) * float(intensity)
    if bp > 0:
        near = cv2.dilate(np.ascontiguousarray(st.max(0)), np.ones((REDUCE_NEIGHBOURHOOD, REDUCE_NEIGHBOURHOOD), np.uint8))
        wgt = np.clip((near - np.float32(0.5 * REDUCE_BRIGHT)) / np.float32(0.5 * REDUCE_BRIGHT), 0.0, 1.0)
        wgt = cv2.GaussianBlur(wgt, (0, 0), 2.0)
        b = np.float32(bp) * wgt[None]
        out = np.clip((st - b) / (1.0 - b), 0, 1)
    else:
        out = st
    return np.clip(out * np.float32(gain), 0, 1).astype(np.float32), gain


def scale_saturation(rgb: np.ndarray, factor: float) -> np.ndarray:
    """Multiply the HSV saturation of an RGB layer by ``factor`` (star colour)."""
    if abs(factor - 1.0) < 1e-3:
        return rgb
    hwc = np.ascontiguousarray(np.moveaxis(rgb, 0, -1))
    hsv = cv2.cvtColor(hwc, cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * np.float32(factor), 0, 1)
    return np.moveaxis(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), -1, 0)


# --------------------------------------------------------------------------- numpy twins of Siril commands

def satu_np(rgb: np.ndarray, amount: float) -> np.ndarray:
    """``satu amount``: HSL saturation x (1 + amount)."""
    if abs(amount) < 1e-3:
        return rgb
    hwc = np.ascontiguousarray(np.moveaxis(rgb, 0, -1))
    hls = cv2.cvtColor(hwc, cv2.COLOR_RGB2HLS)
    hls[..., 2] = np.clip(hls[..., 2] * np.float32(1.0 + amount), 0, 1)
    return np.clip(np.moveaxis(cv2.cvtColor(hls, cv2.COLOR_HLS2RGB), -1, 0), 0, 1)


def rmgreen_np(rgb: np.ndarray, amount: float) -> np.ndarray:
    """``rmgreen 0 amount``: average-neutral SCNR, lightness preserved: G -> min(G, (R+B)/2) by ``amount``."""
    if amount <= 0:
        return rgb
    out = rgb.copy()
    neutral = 0.5 * (rgb[0] + rgb[2])
    out[1] = rgb[1] * (1 - amount) + np.minimum(rgb[1], neutral) * amount
    lo, ln = luminance(rgb), luminance(out)
    return np.clip(out * (lo / np.maximum(ln, 1e-6))[None], 0, 1)


def mtf_np(rgb: np.ndarray, low: float, mid: float, high: float = 1.0) -> np.ndarray:
    """``mtf low mid high``: x -> MTF((x - low) / (high - low), mid), MTF(m, x) = (m-1) x / ((2m-1) x - m)."""
    x = np.clip((rgb - low) / max(high - low, 1e-6), 0, 1)
    if abs(mid - 0.5) < 1e-4:
        return x.astype(np.float32)
    return (((mid - 1.0) * x) / ((2 * mid - 1.0) * x - mid)).astype(np.float32)


def brightness_to_mid(brightness: float) -> float:
    """Midtones slider (-1..1, 0 neutral) -> the ``mid`` of ``mtf``: m = 0.5 x 4^-b (positive brightens)."""
    return float(np.clip(0.5 * 4.0 ** (-float(brightness)), 0.05, 0.95))


# --------------------------------------------------------------------------- detail tools (HWC float32 in/out)

def to_hwc(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.moveaxis(rgb, 0, -1))


def to_chw(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.moveaxis(img, -1, 0))


def chroma_smooth(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Smooth the colour noise (Lab a/b planes) of a starless layer; the lightness plane is untouched.
    The planes are blurred at reduced resolution (cheap, equals a wide gaussian); sigma grows with
    ``amount`` (0..1) and the image size (a 2x drizzle has noise twice as wide)."""
    amount = float(np.clip(amount, 0, 1))
    if amount <= 0:
        return rgb
    hwc = to_hwc(rgb)
    lab = cv2.cvtColor(hwc, cv2.COLOR_RGB2Lab)
    h, w = lab.shape[:2]
    f = 4 if max(h, w) > 3000 else 2
    sigma = (0.8 + 2.2 * amount) * max(1.0, max(h, w) / 3840.0 * 2.0)
    for c in (1, 2):
        small = cv2.resize(lab[..., c], (w // f, h // f), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigma / f * 2.0)
        lab[..., c] = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return to_chw(np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1))


SKY_CHROMA_RAMP = 0.0            # display-lightness span above the sky over which the chroma comes back to full strength (smoothstep)
CHROMA_K = (5.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0)   # soft thresholds of the chrominance planes' starlet scales 1-128 px, in sigmas of the scale measured in the sky;
                                                       # scales 1-3: Brecher's MultiscaleLinearTransform chrominance thresholds 5.0 / 4 / 3 (astrodoc.ca IC 417 SHO),
                                                       # beyond: 3 (the large-scale colour mottle of the stack: the user's blotches), as TGVDenoise / MLT chrominance does at every scale


def chroma_wavelet_denoise(rgb: np.ndarray, amount: float) -> tuple[np.ndarray, dict]:
    """Chrominance noise reduction at every scale from 1 to 128 px (PixInsight MultiscaleLinearTransform / TGVDenoise "chrominance" practice, Brecher IC 417 SHO;
    Delsaert: reduce the chrominance noise BEFORE raising the saturation): the Lab a* and b* planes of the picture are decomposed in starlet scales and every
    coefficient is soft-thresholded at ``CHROMA_K[j] x amount / 0.8`` x sigma_j, with sigma_j the robust (MAD) noise of the scale measured on the SKY samples (the darkest
    45 % of the 32 px smoothed lightness, where only noise lives).  A colour structure of the nebula stands many sigmas above that and keeps its hue and saturation (it
    loses the threshold only); the blotches of the sky are 1-2 sigma and vanish.  The lightness plane is untouched.  Nothing is weighted by the significance of a line:
    the threshold is the one of the noise, the same everywhere."""
    amount = float(np.clip(amount, 0.0, 1.5))
    info: dict = {"amount": amount}
    if amount <= 0:
        return rgb, info
    hwc = to_hwc(rgb)
    lab = cv2.cvtColor(hwc, cv2.COLOR_RGB2Lab)
    lum = lab[..., 0]
    mask = _sky_mask(lum, edge=EDGE_CROP)
    sig_all = []
    before = float(np.sqrt(np.mean([(cv2.GaussianBlur(lab[..., c], (0, 0), BAND_LO) - cv2.GaussianBlur(lab[..., c], (0, 0), BAND_HI))[mask].var() for c in (1, 2)])))
    for c in (1, 2):
        planes, res = starlet(lab[..., c], len(CHROMA_K))
        out = res.copy()
        sg = []
        for j, w in enumerate(planes):
            sj = float(1.4826 * np.median(np.abs(w[mask] - np.median(w[mask])))) if mask.sum() > 100 else 0.0
            t = np.float32(CHROMA_K[j] * amount / 0.8 * sj)
            out += np.sign(w) * np.maximum(np.abs(w) - t, 0.0)
            sg.append(round(sj, 4))
        lab[..., c] = out
        sig_all.append(sg)
    after = float(np.sqrt(np.mean([(cv2.GaussianBlur(lab[..., c], (0, 0), BAND_LO) - cv2.GaussianBlur(lab[..., c], (0, 0), BAND_HI))[mask].var() for c in (1, 2)])))
    info.update(sigma_sky_a=sig_all[0], sigma_sky_b=sig_all[1], k=[round(k * amount / 0.8, 2) for k in CHROMA_K], band_ab_before=round(before, 3), band_ab_after=round(after, 3))
    # the colour of the dark sky: chroma follows the lightness down to the sky (a photographic "saturation mask by luminance": the shadows of every colour image lose chroma;
    # a smooth function of the 8 px smoothed lightness only, never of how significant a line is): w = smoothstep((L - sky) / ramp) with the sky = the median of the darkest
    # 45 % and the ramp ``SKY_CHROMA_RAMP`` x amount / 0.8 (Lab L units / 100)
    if SKY_CHROMA_RAMP <= 0:
        return to_chw(np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)), info
    sm = cv2.GaussianBlur(lum, (0, 0), 8.0) / 100.0
    smp = sm[::3, ::3].reshape(-1)
    sky = float(np.median(smp[smp <= np.percentile(smp, 45)]))
    ramp = max(SKY_CHROMA_RAMP * amount / 0.8, 1e-3)
    t = np.clip((sm - np.float32(sky)) / np.float32(ramp), 0.0, 1.0)
    w = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
    lab[..., 1] *= w
    lab[..., 2] *= w
    info.update(sky_L=round(sky, 4), sky_chroma_ramp=round(ramp, 4))
    return to_chw(np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)), info


def sharpen(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Unsharp mask on the lightness (Lab L) with a soft threshold at the image's noise, so the sky's
    grain is not sharpened; ``amount`` 0..1.5."""
    amount = float(amount)
    if amount <= 0:
        return rgb
    hwc = to_hwc(rgb)
    lab = cv2.cvtColor(hwc, cv2.COLOR_RGB2Lab)
    ll = lab[..., 0]
    h, w = ll.shape
    sigma = 1.2 * max(1.0, max(h, w) / 3840.0)
    detail = ll - cv2.GaussianBlur(ll, (0, 0), sigma)
    thr = 1.0 * mad_sigma(detail[:: max(1, h // 400), :: max(1, w // 400)])
    soft = np.sign(detail) * np.maximum(np.abs(detail) - thr, 0.0)
    lab[..., 0] = np.clip(ll + np.float32(amount * 0.8) * soft, 0, 100)
    return to_chw(np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1))


REVELA_SHADOW = 33                 # VeraLux Revela's default "shadow authority": the signal gate opens 33 * 0.12 - 3 = 0.96 sigma above the median lightness
REVELA_TEXTURE = 0.35              # texture (scales 1-2) gain per unit of ``amount``: small, those scales hold the noise left after the denoise


def local_contrast(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Local contrast / texture on the lightness with VeraLux Revela's engine (R. Paterniti 2025, GPL, siril-scripts/VeraLux/VeraLux_Revela.py): Lab L is split
    into six a trous B3-spline planes, the medium scales 3-5 (4-16 px: the body of the nebula, "structure") are multiplied by ``1 + amount`` and the fine
    scales 1-2 ("texture") by ``1 + 1.5 x REVELA_TEXTURE x amount``, both through a signal mask (``(L - (median + t sigma)) / (2 sigma)`` clipped to 0..1, t = 0.96)
    so the sky and its noise keep their gain of 1; the colour (a, b) is untouched.  ``amount`` 0..2 (0 = off)."""
    amount = float(np.clip(amount, 0.0, 2.0))
    if amount <= 0:
        return rgb
    lab = cv2.cvtColor(to_hwc(rgb), cv2.COLOR_RGB2Lab)
    ll = np.ascontiguousarray(lab[..., 0])
    planes, resid = starlet(ll, 6)
    ln = ll / 100.0
    med = float(np.median(ln[:: max(1, ln.shape[0] // 700), :: max(1, ln.shape[1] // 700)]))
    sub = ln[:: max(1, ln.shape[0] // 700), :: max(1, ln.shape[1] // 700)]
    sigma = max(1.4826 * float(np.median(np.abs(sub - med))), 1e-6)
    thr = REVELA_SHADOW * 0.12 - 3.0
    mask = np.clip((ln - (med + thr * sigma)) / (2.0 * sigma), 0.0, 1.0)
    mask = cv2.blur(mask, (3, 3), borderType=cv2.BORDER_REFLECT)
    t_gain = 1.0 + 1.5 * REVELA_TEXTURE * amount
    s_gain = 1.0 + amount
    for j in (0, 1):
        planes[j] = planes[j] * (1.0 + (t_gain - 1.0) * mask)
    for j in (2, 3, 4):
        planes[j] = planes[j] * (1.0 + (s_gain - 1.0) * mask)
    lab[..., 0] = np.clip(resid + sum(planes), 0.0, 100.0)
    return to_chw(np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1))


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


def count_stars(stars_lin: np.ndarray, noise_sigma: float, min_sigma: float = 5.0) -> tuple[int, float]:
    """Detectable stars: local maxima of the (lightly smoothed) star layer above ``min_sigma`` x the sky's pixel noise
    ``noise_sigma`` (of the starless layer), and their number per megapixel."""
    if noise_sigma <= 0:
        return 0, 0.0
    lum = cv2.GaussianBlur(np.ascontiguousarray(stars_lin.max(0)), (0, 0), 1.0)
    dil = cv2.dilate(lum, np.ones((7, 7), np.uint8))
    n = int(((lum == dil) & (lum > min_sigma * noise_sigma)).sum())
    return n, n / (lum.size / 1e6)


def mean_saturation(rgb: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean HSV saturation (over the masked pixels)."""
    step = max(1, int(round(max(rgb.shape[1:]) / 1200)))
    hwc = to_hwc(rgb[:, ::step, ::step])
    s = cv2.cvtColor(hwc, cv2.COLOR_RGB2HSV)[..., 1]
    if mask is not None:
        m = mask[::step, ::step]
        if m.sum() > 50:
            return float(s[m].mean())
    return float(s.mean())


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


# --------------------------------------------------------------------------- the GHS stretch of the starless layer ("stretch where the signal is")
#
# Why the old route (statistical stretch to a 0.25 median, then a black point) left faint nebulae flat: its black point sits at ``median - 5 sigma`` of the
# PIXEL noise, which on a 1-hour stack of a faint emission nebula is 5-10x wider than the whole nebula signal (IC 1805: sigma 2.2e-4, nebula 1e-4 .. 5e-4
# above the sky), so the sky and the nebula both land in the lowest few percent of the stretch's input range.  What the community does instead:
#   * denoise the linear starless layer first, so the black point can sit at the sky (Under South West Skies / Seti Astro / VeraLux workflows, workflows.md 7.7);
#   * anchor the black point at the low percentile of the (smoothed) channel, not at a pixel-noise multiple (VeraLux HyperMetric Stretch ``calculate_anchor``:
#     0.5th percentile of each channel; GPL, siril-scripts/VeraLux);
#   * stretch with the Generalised Hyperbolic Stretch (Payne & Cranfield 2021, ghsastro.co.uk; Siril ``ght``) whose symmetry point SP is put where the faint
#     nebula is and whose local intensity b concentrates the contrast around it ("the stretch is where the signal is").  Siril's ght needs D, SP by hand; here
#     they are solved so the sky lands on ``SKY_AIM`` and the 90th percentile of the nebula on ``NEBULA_AIM`` (display levels of an sRGB picture);
#   * OIII is normalised to Ha on the STRETCHED data with the Blanshan / Cranfield narrowband normalisation (NarrowbandNormalization.py by Cuiv, siril-scripts,
#     GPL), whose boost defaults to 1.0: no OIII multiplier that amplifies noise.
SKY_AIM = 0.085                   # display level of the sky (the user's aim 0.07-0.10; Under SW Skies / VeraLux target backgrounds are 0.10-0.20 for a screen with gamma)
NEBULA_AIM = 0.60                 # display level of the nebula's 90th percentile (aim: nebula p90 >= 0.45 measured on the final picture; the fit's "signal" mask is stricter than the picture's, so the aim is higher)
ANCHOR_PERCENTILE = 1.0           # black point = this percentile of the smoothed, denoised channel (VeraLux uses 0.5 on the raw channel; the denoised one has no noise tail)
ANCHOR_SMOOTH = 1.5               # px, gaussian used for the percentile statistics
WHITE_PERCENTILE = 99.999        # of the denoised, smoothed starless plane (a lower one clipped IC 1805's core, 0.2 % of the frame, to a flat white plateau)
GHS_B_LADDER = (13.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.5)
BRIGHT_AIM = 0.95                 # the signal's 99.5th percentile stays at or below this display level (headroom below clipping)
GHS_FIT_TOLERANCE = 1e-4          # summed squared error of the two aims below which a b is "met"
GHS_SKY_SHARE = 0.5               # the stretch puts the sky at this share of the sky aim; the output black level (sky_lift: Levels' output slider) brings it up to the aim.  A sky at 8.5 % of
                                  # the output range sits 0.4 % above the black point of a white-scaled plane: the slope below it would have to exceed the slope above it
NEBULA_AIMS = (0.60, 0.55, 0.50, 0.45, 0.40, 0.35)  # nebula p90 display levels tried in turn (the brightest whose stretched sky noise stays within ``SKY_NOISE_LIMIT``)
SKY_NOISE_LIMIT = 0.01            # display units: noise in the 32-128 px band of the sky (the user's "watercolour" blobs are this band)
NOISE_LP_MODES = (0.0, 1.0)       # shadow protection candidates of the noise-floor fit, as fractions of the sky level

# Starlet (a trous B3-spline) shrinkage, Starck & Murtagh "Astronomical Image and Data Analysis" 2002 / Starck, Murtagh, Bijaoui 1998 (the noise reduction of
# PixInsight's MultiscaleLinearTransform and Siril's wavelet tools), with the *locally adaptive* threshold of BayesShrink (Chang, Yu & Vetterli 2000, "Adaptive
# wavelet thresholding for image denoising and compression", IEEE Trans. Image Process. 9(9)): a coefficient w of scale j is soft-thresholded at
# ``T = sigma_n^2 / sigma_x`` with sigma_n the noise of that scale (MAD of the plane, measured: the DWARF stacks' noise is spatially correlated, so white-noise
# sigmas are 1.5-3x too small) and ``sigma_x = sqrt(max(E[w^2] - sigma_n^2, 0))`` the local signal deviation, E taken over a window of at least
# ``BAYES_MIN_WINDOW`` px (one noise pixel inside a 5 x 5 window looked like signal and survived as a dark or bright speck: measured on IC 1805).  Where there is
# no signal (sigma_x -> 0) the threshold is infinite and the sky comes out smooth; where a structure stands above the noise T falls to nothing and it keeps its
# amplitude.  This replaces the fixed K x sigma thresholds (4 .. 1) whose residual noise blobs the stretch then magnified into watercolour blotches.
BAYES_MIN_WINDOW = 15
BIVARIATE_C = float(np.sqrt(3.0))    # Sendur & Selesnick's MAP constant
COARSE_GAIN = 0.0                 # share of the threshold applied on the scales beyond ``shrink_scales``
BAYES_WINDOW_SCALE = 4                # window = this x 2^j px (the support of the scale)
_B3 = np.array([1, 4, 6, 4, 1], np.float32) / 16.0
_STARLET_NOISE: dict = {}


def starlet(x: np.ndarray, scales: int) -> tuple[list[np.ndarray], np.ndarray]:
    """A trous decomposition with the B3-spline kernel: ``(wavelet planes, smooth residual)``, x = sum(planes) + residual."""
    c = np.ascontiguousarray(x, dtype=np.float32)
    planes = []
    for j in range(scales):
        k = np.zeros(4 * 2 ** j + 1, np.float32)
        k[:: 2 ** j] = _B3
        s = cv2.sepFilter2D(c, -1, k, k, borderType=cv2.BORDER_REFLECT)
        planes.append(c - s)
        c = s
    return planes, c


def _starlet_noise_scale(scales: int) -> list[float]:
    """Std of each wavelet plane of unit-variance white noise (a fixed seed: deterministic)."""
    if scales not in _STARLET_NOISE:
        n = np.random.default_rng(1).standard_normal((768, 768)).astype(np.float32)
        pl, _ = starlet(n, scales)
        _STARLET_NOISE[scales] = [float(p[48:-48, 48:-48].std()) for p in pl]
    return _STARLET_NOISE[scales]


COLD_SIGMA = 1.5                  # cold-pixel rejection: values more than this many pixel sigmas below the local background are clipped to it


def _cold_clip(x: np.ndarray) -> np.ndarray:
    """Cold-pixel rejection before the wavelet shrinkage.  StarXTerminator / StarNet leave over-subtracted dips (negative rings) where they removed a star;
    nothing in the sky is darker than its local background by more than the noise, so values below ``background - COLD_SIGMA x sigma`` are clipped to that
    floor (a floor at the noise edge, never above the background: no light is added).  Background = a wide median-like gaussian of the plane, sigma from the
    MAD of the finest a trous plane."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    small = cv2.resize(x, (max(8, x.shape[1] // 16), max(8, x.shape[0] // 16)), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.GaussianBlur(small, (0, 0), 2.0), (x.shape[1], x.shape[0]), interpolation=cv2.INTER_LINEAR)
    w0, _ = starlet(x, 1)
    sub = w0[0][:: max(1, x.shape[0] // 1200), :: max(1, x.shape[1] // 1200)]
    sig = float(1.4826 * np.median(np.abs(sub - np.median(sub))) / _starlet_noise_scale(1)[0])
    return np.maximum(x, bg - np.float32(COLD_SIGMA * sig))


def scale_noise(planes: list[np.ndarray]) -> list[float]:
    """Noise sigma of every wavelet plane: 1.4826 x MAD (robust to the nebula, which covers the minority of the pixels)."""
    out = []
    for w in planes:
        sub = w[:: max(1, w.shape[0] // 1200), :: max(1, w.shape[1] // 1200)]
        out.append(float(1.4826 * np.median(np.abs(sub - np.median(sub)))))
    return out


def starlet_denoise(x: np.ndarray, amount: float = 1.0, scales: int = 6, contrast: float = 0.0, shrink_scales: int | None = None) -> tuple[np.ndarray, float]:
    """BayesShrink of the starlet planes of one linear plane at ``amount`` x T (see the block comment above); returns ``(denoised, sigma_1)`` with ``sigma_1`` the
    pixel-noise estimate (MAD of the first plane / its white-noise factor).  ``amount`` <= 0 returns the input.

    ``contrast`` (0..2) is the SNR-gated wavelet gain (the noise-aware multiscale sharpening of PixInsight's MultiscaleLinearTransform "bias" with a noise
    threshold; Revela's structure gain 1 + amount, VeraLux 2025): a surviving coefficient is multiplied by ``1 + contrast x s`` on the scales 2-5 (2-16 px) with
    ``s`` the smoothstep of the coefficient's local signal-to-noise ``sigma_x / sigma_n`` from 3 to 6: a scale is never amplified where its signal does not stand
    5 x above its noise (measured per scale), so noise cannot be sharpened into blobs."""
    if amount <= 0:
        return np.asarray(x, np.float32), 0.0
    x = _cold_clip(x)
    planes, out = starlet(x, scales)
    cf = _starlet_noise_scale(scales)
    sig = scale_noise(planes)
    out = out.copy()
    for j, w in enumerate(planes):
        k = int(max(BAYES_MIN_WINDOW, BAYES_WINDOW_SCALE * 2 ** j)) | 1
        v = cv2.boxFilter(w * w, -1, (k, k), borderType=cv2.BORDER_REFLECT)
        sx = np.sqrt(np.maximum(v - np.float32(sig[j] ** 2), 0.0))
        g = 1.0 if (shrink_scales is None or j < shrink_scales) else COARSE_GAIN        # beyond ``shrink_scales`` only a fraction of the threshold
        if g <= 0:
            shrunk = w
        elif j + 1 < len(planes):                         # bivariate shrinkage (Sendur & Selesnick 2002): the parent scale supports or condemns the coefficient
            par = planes[j + 1] * np.float32(sig[j] / max(sig[j + 1], 1e-12))
            r = np.sqrt(w * w + par * par)
            t = np.float32(BIVARIATE_C * amount * g * sig[j] ** 2) / np.maximum(sx, np.float32(1e-12))
            shrunk = w * np.maximum(r - t, 0.0) / np.maximum(r, np.float32(1e-12))
        else:
            t = np.float32(amount * g * sig[j] ** 2) / np.maximum(sx, np.float32(1e-12))
            shrunk = np.sign(w) * np.maximum(np.abs(w) - t, 0.0)
        if contrast > 0 and 1 <= j <= 4:
            g = np.clip((sx / np.float32(max(sig[j], 1e-12)) - 3.0) / 3.0, 0.0, 1.0)
            shrunk = shrunk * (1.0 + np.float32(contrast) * g * g * (3.0 - 2.0 * g))
        out += shrunk
    return out, sig[0] / cf[0]


FLATTEN_DEGREE = 2                # polynomial degree of the second background pass (Siril ``subsky`` poly, degree 1-4)
FLATTEN_DEGREE_BROADBAND = 3      # broadband RGB planes: the colour gradients of the corners (an amber glow in NGC 1499's corner) need one more order
FLATTEN_SKY_FRACTION = 0.40       # the darkest 40 % of the (model-corrected) smoothed plane are the sky samples


def flatten_sky(plane: np.ndarray, degree: int | None = None) -> tuple[np.ndarray, dict]:
    """Second, light background-extraction pass on a (denoised) linear plane before the stretch: a 2-D polynomial of ``degree`` fitted to the sky samples
    (the darkest 40 % of the plane reduced 16x, re-selected over three iterations on the residual to the model so the nebula is never fitted) and subtracted,
    keeping the sky median (offsets only, nothing added).  This is Siril's ``subsky`` polynomial model / the DBE-before-stretch step of every workflow in
    workflows.md 7.7: GraXpert's residual on the DWARF stacks is only 0.2 % of the sky level, but the stretch's window is 5 % of it, so a 0.2 % tilt is a
    0.06 display-unit drift of the sky across the frame (measured on IC 1805)."""
    degree = int(FLATTEN_DEGREE if degree is None else degree)
    h, w = plane.shape
    f = 16
    small = cv2.GaussianBlur(cv2.resize(np.ascontiguousarray(plane, dtype=np.float32), (max(8, w // f), max(8, h // f)), interpolation=cv2.INTER_AREA), (0, 0), 1.0)
    sh, sw = small.shape
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float64)
    xn, yn = xx / max(sw - 1, 1) * 2 - 1, yy / max(sh - 1, 1) * 2 - 1
    terms = [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]
    basis = np.stack([(xn ** i) * (yn ** j) for i, j in terms], -1)
    z = small.astype(np.float64)
    resid = z - np.median(z)
    coef = np.zeros(len(terms))
    for _ in range(3):
        sel = resid <= np.percentile(resid, FLATTEN_SKY_FRACTION * 100)
        coef, *_ = np.linalg.lstsq(basis[sel], z[sel], rcond=None)
        resid = z - basis @ coef
    model_small = (basis @ coef)
    sky_med = float(np.median(model_small[sel]))
    up = cv2.resize((model_small - sky_med).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    span = float(model_small.max() - model_small.min())
    return np.ascontiguousarray(plane, dtype=np.float32) - up, {"degree": degree, "span": span, "span_rel": span / max(sky_med, 1e-9)}


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


def ghs_slope_at(x: float, b: float, d: float, sp: float, lp: float = 0.0, hp: float = 1.0, h: float = 1e-4) -> float:
    """Output units per input unit of the GHS at ``x`` (central difference)."""
    lo, hi = max(x - h, 0.0), min(x + h, 1.0)
    t = _ghs_t(np.array([lo, hi]), float(b), float(d), float(sp), lp, hp)
    return float((t[1] - t[0]) / (hi - lo))


def ghs_fit(sky: float, neb: float, sky_aim: float, neb_aim: float, b_max: float, bright: float | None = None, noise: float | None = None,
            lp_modes: tuple = (0.0,), sp_min: float = 0.0, hp: float = 1.0, focus: float | None = None, noise_limit: float | None = None) -> dict:
    """D, SP, LP of the GHS that sends ``sky`` to ``sky_aim`` and ``neb`` to ``neb_aim`` (grid search over log D 0.1 .. 1e6 and SP, then a finer grid around the best),
    for every local intensity ``b <= b_max`` of ``GHS_B_LADDER`` and every shadow protection of ``lp_modes`` (fractions of the sky level: 0 = none, 1 = LP at the sky: the
    shadows below the sky stay linear).  Without ``noise`` the highest b that meets ``GHS_FIT_TOLERANCE`` wins (the legacy rule).  With ``noise`` (the sky's large-scale noise
    in input units) the candidate that meets the aims with the LOWEST slope at the sky wins: the noise the stretch puts in the sky is ``noise x slope``, and the curve
    of the GHS has a free shape (b, SP, LP) after the two aims are met - Payne & Cranfield / Cranfield's guidance, "SP above the noise you want to hide, LP to keep the
    shadows".  ``bright`` (the 99.5th percentile of the signal) must not be sent above ``BRIGHT_AIM`` (a one-sided penalty: the cores keep a slope)."""
    sky, neb = float(np.clip(sky, 1e-4, 0.98)), float(np.clip(neb, 1e-4, 0.99))
    neb = max(neb, sky * 1.05)
    cands = []
    for b in [v for v in GHS_B_LADDER if v <= b_max] or [min(GHS_B_LADDER)]:
        for lpm in lp_modes:
            lp = float(lpm) * sky
            lds = np.linspace(-1.0, 6.0, 71)
            sps = np.linspace(max(0.005, lp, sp_min), min(0.9, max(neb, sp_min * 1.5)), 89)
            best = None
            for rnd in range(2):
                d = (10.0 ** lds)[:, None]
                s_ = sps[None, :]
                err = (_ghs_t(sky, b, d, s_, lp, hp) - sky_aim) ** 2 + (_ghs_t(neb, b, d, s_, lp, hp) - neb_aim) ** 2
                if bright is not None:
                    err = err + np.maximum(_ghs_t(max(bright, neb), b, d, s_, lp, hp) - BRIGHT_AIM, 0.0) ** 2
                i, k = np.unravel_index(int(np.nanargmin(err)), err.shape)
                best = (float(err[i, k]), float(lds[i]), float(sps[k]))
                if rnd == 0:
                    lds = np.linspace(best[1] - 0.15, best[1] + 0.15, 41)
                    sps = np.linspace(max(0.005, lp, sp_min, best[2] - 0.012), min(0.95, best[2] + 0.012), 41)
            c = {"b": b, "D": 10.0 ** best[1], "sp": best[2], "lp": lp, "hp": hp, "error": best[0]}
            c["slope_sky"] = ghs_slope_at(sky, b, c["D"], c["sp"], lp, hp)
            if noise is not None:
                c["noise_out"] = float(noise) * c["slope_sky"]
            if focus is not None:
                c["slope_focus"] = ghs_slope_at(focus, b, c["D"], c["sp"], lp, hp)
            # the steepest part of the curve between the sky and the nebula's 90th percentile, against the straight line between the two aims (the secant): a curve whose
            # slope is several times the secant's makes a ragged boundary of whatever noise sits at that level (M 45's nebula edge with b = 13)
            gx = np.linspace(sky, neb, 48)
            gt = _ghs_t(gx, b, c["D"], c["sp"], lp, hp)
            c["slope_max"] = float(np.max(np.diff(gt) / np.diff(gx)))
            c["slope_ratio"] = c["slope_max"] / max((neb_aim - sky_aim) / max(neb - sky, 1e-9), 1e-9)
            cands.append(c)
            if noise is None and c["error"] <= GHS_FIT_TOLERANCE and lpm == lp_modes[0]:
                return c
    met = [c for c in cands if c["error"] <= GHS_FIT_TOLERANCE]
    if noise is not None and met:
        quiet = [c for c in met if noise_limit is None or c["noise_out"] <= noise_limit]
        smooth = [c for c in quiet if c["slope_ratio"] <= MAX_SLOPE_RATIO]
        quiet = smooth or ([min(quiet, key=lambda c: c["slope_ratio"])] if quiet else [])
        if quiet and focus is not None:
            return max(quiet, key=lambda c: c["slope_focus"])            # the noise limit is met: the most contrast where the signal sits (Cranfield: SP / b where you want contrast)
        return min(met, key=lambda c: c["noise_out"])
    return min(met or cands, key=lambda c: c["error"])


HP_AT_BRIGHT = True               # highlight protection at the 99.5th percentile of the signal: the cores above it stay on a straight line (Siril GHS tutorial: "almost always some HP")
MAX_SLOPE_RATIO = 2.2             # the steepest slope of the curve between sky and nebula p90 is at most this many times the secant between the two aims
SP_ABOVE_NOISE = 2.0              # the symmetry point sits at least this many (32-128 px band) noise sigmas above the sky (Cranfield: SP "above the level of noise you want to hide")


def ghs_fit_noise_floor(sky: float, neb: float, bright: float | None, noise: float, sky_aim: float, neb_aims: tuple, b_max: float,
                        noise_limit: float, focus: float | None = None) -> dict:
    """``ghs_fit`` with the noise floor held down: the brightest of ``neb_aims`` for which the stretch's noise in the sky (``noise`` x the slope at the sky, display
    units) stays within ``noise_limit``; when none does, the faintest aim with its lowest-noise curve ("accept a fainter nebula rather than mottling")."""
    last = None
    for aim in neb_aims:
        hp = float(np.clip(bright, 0.3, 1.0)) if (bright is not None and HP_AT_BRIGHT) else 1.0
        c = ghs_fit(sky, neb, sky_aim, aim, b_max, bright, noise=noise, lp_modes=NOISE_LP_MODES, sp_min=sky + SP_ABOVE_NOISE * noise, hp=hp, focus=focus,
                    noise_limit=noise_limit)
        c["neb_aim"] = aim
        last = c
        if c["error"] <= GHS_FIT_TOLERANCE and c["noise_out"] <= noise_limit:
            c["noise_limit_met"] = True
            return c
    last["noise_limit_met"] = False
    return last


def nbn_oiii(ha: np.ndarray, oiii: np.ndarray, boost: float = 1.0) -> tuple[np.ndarray, float]:
    """Narrowband normalisation of the stretched OIII to the stretched Ha (Bill Blanshan & Mike Cranfield's NarrowbandNormalization as ported in
    siril-scripts/processing/NarrowbandNormalization.py by Y. Dutertre, GPL; shadow point 1 = the median): OIII is rescaled from its median to 1, sent through a
    midtones function whose strength makes the *average level above the median* of OIII equal Ha's (``boost`` = strength, 1 = equal), and screen-blended with the
    part below the median.  Returns ``(normalised OIII, midtones strength)``."""
    def stats(ch):
        s = sample(ch)
        mn, med = float(s.min()), float(np.median(s))
        mean = float(np.mean(s, dtype=np.float64))
        adev = float(np.mean(np.abs(s - mean), dtype=np.float64))
        return mn + (med - mn), adev / 1.2533 + mean - med        # M (shadow point 1 -> median), E0
    m_h, e_h = stats(ha)
    m_o, e_o = stats(oiii)
    den = max(1.0 - m_o, 1e-9)
    a_o, a_h = e_o / den, e_h / den
    d = a_o - 2.0 * a_o * a_h + a_h
    if abs(d) < 1e-9:
        d = 1e-9
    e_full = float(np.clip(a_o * (1.0 - a_h) / d, 0.01, 0.99))                       # the midtones balance that equalises the two lines' levels (Blanshan / Cranfield)
    # ``boost`` = the strength of that normalisation: 1 = equal levels, 0 = none (midtones balance 0.5 = identity), in between the geometric interpolation of the balance; 2 = twice
    # as strong.  (A divisor of the balance made 0.5 still triple the faint OIII of IC 1805: the normalisation of a line 3x weaker than Ha is a strong operation)
    e1 = float(np.clip(e_full ** max(float(boost), 0.0) * 0.5 ** (1.0 - max(float(boost), 0.0)), 0.01, 0.99))
    resc = np.clip((oiii - np.float32(m_o)) / np.float32(max(1.0 - m_o, 1e-6)), 0.0, 1.0)
    st = ((e1 - 1.0) * resc) / ((2.0 * e1 - 1.0) * resc - e1) if abs(e1 - 0.5) > 1e-9 else resc
    fl = np.minimum(oiii, np.float32(m_o))
    return np.clip(1.0 - (1.0 - st) * (1.0 - fl), 0.0, 1.0).astype(np.float32), e1


LUMA_RATIO_SMOOTH = 3.0           # px (minimum), gaussian of the planes the colour ratios are taken from: ``COLOR_SIGMA_FRAC`` x the long side, as for the dual-band colour
SIGNAL_SIGMA = 4.0                # "signal" = more than this many low-frequency sigmas above the sky mode (3 let the plateau of noise clouds count as nebula)


def _hist_mode(x: np.ndarray, lo: float | None = None, hi: float | None = None, bins: int = 512) -> float:
    """Most frequent level of ``x`` (peak of a 5-bin smoothed histogram between the 0.5th and 99th percentile)."""
    x = np.asarray(x).reshape(-1)
    lo = float(np.percentile(x, 0.5)) if lo is None else lo
    hi = float(np.percentile(x, 99.0)) if hi is None else hi
    h, e = np.histogram(x, bins=bins, range=(lo, max(hi, lo + 1e-12)))
    h = np.convolve(h.astype(np.float64), np.ones(5) / 5.0, mode="same")
    k = int(np.argmax(h))
    return float(0.5 * (e[k] + e[k + 1]))


def _smooth_sample(x: np.ndarray, sigma: float = ANCHOR_SMOOTH, step: int = 3) -> np.ndarray:
    return cv2.GaussianBlur(np.ascontiguousarray(x, dtype=np.float32), (0, 0), sigma)[::step, ::step]


LARGE_SCALES = (3, 4, 5, 6, 7)    # starlet scales (8, 16, 32, 64, 128 px) of the large-scale noise pass
LARGE_SCALE_K = 2.0               # soft threshold of that pass, in sigmas of the scale's coefficients in the sky (measured, not assumed)
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


def band_noise(plane: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Noise of a plane in the 32-128 px band (std of ``G(BAND_LO) - G(BAND_HI)``) over its sky samples."""
    pl = np.ascontiguousarray(plane, dtype=np.float32)
    band = cv2.GaussianBlur(pl, (0, 0), BAND_LO) - cv2.GaussianBlur(pl, (0, 0), BAND_HI)
    m = _sky_mask(pl, edge=EDGE_CROP) if mask is None else mask
    return float(band[m].std()) if m.sum() > 100 else float(band.std())


def large_scale_denoise(plane: np.ndarray, amount: float = 1.0, residual_k: float = 0.0) -> tuple[np.ndarray, dict]:
    """Soft-threshold the 16-128 px starlet coefficients of a linear plane at ``amount`` x ``LARGE_SCALE_K`` sigma of the scale, sigma measured on the SKY samples
    (the darkest 45 % at 64 px) where the nebula is absent: what Ron Brecher does with MultiscaleLinearTransform "on background areas with a mask protecting bright
    structure" (astrodoc.ca IC 417 SHO) and what Delsaert recommends for the large-scale noise of a stack, in the coefficient domain.  A coefficient of a real
    structure stands many sigmas above the sky's noise and loses only the threshold; the faint 32-128 px mottle of the stack (1-3 sigma) goes to zero.  The sky mask
    comes from the plane itself, so the threshold follows the data (nothing is assumed about the noise).  With ``residual_k`` > 0 the smooth residual (structure beyond
    128 px: what the background extraction and the flats leave of a channel-to-channel gradient, 0.1-0.2 % of the sky level on the DWARF stacks) is soft-thresholded about
    its median at ``residual_k`` x its robust spread over the whole plane: a colour plane (the OIII of a dual-band stack) then carries a 128+ px structure only where it
    stands out of that systematic floor."""
    info: dict = {"k": round(float(LARGE_SCALE_K * amount), 3)}
    if amount <= 0:
        return np.asarray(plane, np.float32), info
    nscales = max(LARGE_SCALES) + 1
    planes, res = starlet(np.asarray(plane, np.float32), nscales)
    mask = _sky_mask(plane, edge=EDGE_CROP)
    out = res.copy()
    if residual_k > 0:
        med = float(np.median(res[::4, ::4]))
        d = res - np.float32(med)
        sd = float(1.4826 * np.median(np.abs(d[::4, ::4] - np.median(d[::4, ::4]))))
        out = np.float32(med) + np.sign(d) * np.maximum(np.abs(d) - np.float32(residual_k * sd), 0.0)
        info["residual_sigma"] = round(sd, 9)
        info["residual_k"] = residual_k
    sig = []
    for j, w in enumerate(planes):
        if j in LARGE_SCALES:
            sj = float(1.4826 * np.median(np.abs(w[mask] - np.median(w[mask])))) if mask.sum() > 100 else 0.0
            t = np.float32(LARGE_SCALE_K * amount * sj)
            w = np.sign(w) * np.maximum(np.abs(w) - t, 0.0)
            sig.append(round(sj, 9))
        out += w
    info["sigma_sky"] = sig
    info["band_before"] = round(band_noise(plane, mask), 9)
    info["band_after"] = round(band_noise(out, mask), 9)
    return out.astype(np.float32), info


def ghs_stretch(planes: list[np.ndarray], *, sky_aim: float, neb_aim: float | tuple | None = None, b_max: float, denoise, mix=None, luma_ratio: bool = False,
                contrast: float = 0.0, shrink_scales: int | None = None, stretcher=None, method: str = "ghs", hdr: float = 0.0, boost: float = 0.0,
                large_scale: float = 1.0, residual_k=0.0, compose=None) -> tuple[list[np.ndarray], dict]:
    """The "where the signal is" stretch of the linear planes ``planes`` (two: Ha, OIII of a dual-band stack, or three: R, G, B).

    1. every plane is denoised with ``starlet_denoise(amount = denoise)`` (a number, or one per plane; fine scales: BayesShrink) and ``large_scale_denoise`` (16-128 px:
       the mottle of the stack, thresholds measured in the sky), and, with ``contrast`` > 0, sharpened where its signal-to-noise allows it;
    2. the black point of each plane is the mode of the smoothed plane, one common white point (the largest ``WHITE_PERCENTILE`` of the planes above their floors)
       keeps the planes' ratios: ``u = (x - black) / white``;
    3. the statistics (sky, sky noise in the 32-128 px band, nebula 90th percentile = pixels 4 sigma above the sky) go to the stretch tool ``stretcher`` (``stretch.SirilStretcher``:
       Siril ``ght`` / ``autoghs`` / ``modasinh`` / ``autostretch`` or Statistical_Stretch.py, ``method``), which fits its parameters to the aims (the sky on ``sky_aim``,
       the nebula on the brightest of ``neb_aim`` whose stretched sky noise stays within ``SKY_NOISE_LIMIT``) and stretches every plane with the same transform;
       without a ``stretcher`` the numpy mirror of ``ght`` does it.
    ``mix(planes)`` returns the display luminance of the planes (default: the mean).  Returns the stretched planes and the record of what was done."""
    amounts = list(denoise) if isinstance(denoise, (list, tuple)) else [float(denoise)] * len(planes)
    info: dict = {"denoise": [round(float(v), 3) for v in amounts] if len(set(amounts)) > 1 else round(float(amounts[0]), 3)}
    dn, sig = [], []
    flat, large = [], []
    rks = list(residual_k) if isinstance(residual_k, (list, tuple)) else [float(residual_k)] * len(planes)
    for pl, am, rk in zip(planes, amounts, rks):
        a, s1 = starlet_denoise(pl, float(am), contrast=contrast, shrink_scales=shrink_scales)
        a, li = large_scale_denoise(a, float(am) * float(large_scale) if am > 0 else 0.0, float(rk)) if large_scale > 0 else (a, {})
        a, fi = flatten_sky(a, FLATTEN_DEGREE_BROADBAND if luma_ratio else None)
        dn.append(a)
        sig.append(s1)
        flat.append(round(fi["span"], 7))
        large.append(li)
    info["flatten_span"] = flat
    info["large_scale"] = large
    info["noise_sigma"] = [round(float(s), 7) for s in sig]
    sm = [_smooth_sample(a) for a in dn]
    # black points: the sky of every plane (the mode of its smoothed histogram: the sky is the most frequent level; VeraLux's adaptive anchor also reads the
    # histogram peak) is moved to the same level S0 of the white-point scale, so the sky is neutral by construction (offsets only, no gains; the
    # finish.neutralize_background rule); S0 = how far the sky sits above the plane's anchor percentile on average (VeraLux calculate_anchor)
    skies = [_hist_mode(s) for s in sm]
    floors = [float(np.percentile(s, ANCHOR_PERCENTILE)) for s in sm]
    white = max(max(float(np.percentile(s - f, WHITE_PERCENTILE)) for s, f in zip(sm, floors)), 1e-9)
    s0 = float(np.mean([(k - f) for k, f in zip(skies, floors)])) / white
    bps = [k - s0 * white for k in skies]
    mixf = mix or (lambda ps: sum(ps) / len(ps))
    lraw = mixf([(s - b) / white for s, b in zip(sm, bps)])
    ls = np.clip(lraw, 0.0, 1.0)
    # sky level = the mode; the low-frequency noise sigma from the values BELOW the mode (a half Gaussian: the nebula only adds to the upper side)
    sky = _hist_mode(lraw, lo=float(np.percentile(lraw, 0.5)), hi=float(np.percentile(lraw, 99.0)))
    below = lraw[lraw < sky]
    noise = float(np.sqrt(np.mean((below - sky) ** 2))) if below.size > 50 else 1e-3
    sig_mask = ls > sky + SIGNAL_SIGMA * max(noise, 1e-4)
    neb = float(np.percentile(ls[sig_mask], 90)) if sig_mask.sum() > 500 else float(np.percentile(ls, 99.0))
    bright = float(np.percentile(ls[sig_mask], 99.5)) if sig_mask.sum() > 500 else None
    # the sky's noise in the 32-128 px band, in the same units as ``lraw`` (the lightness the fit looks at): the dominant noise after the fine scales are gone
    mix_full = mixf([(a - np.float32(b0)) / np.float32(white) for a, b0 in zip(dn, bps)])
    band = band_noise(mix_full)
    aims = tuple(neb_aim) if isinstance(neb_aim, (tuple, list)) else ((float(neb_aim),) if neb_aim is not None else NEBULA_AIMS)
    if stretcher is None:
        from .stretch import SirilStretcher
        import tempfile
        stretcher = SirilStretcher(tempfile.mkdtemp(prefix="stretch_"), allow=False)
    focus = float(np.median(ls[sig_mask])) if sig_mask.sum() > 500 else None          # where the signal sits: the stretch's contrast is asked for here
    ctx = {"method": method, "sky": sky, "noise": noise, "band": band, "neb": neb, "bright": bright, "focus": focus, "lum": mix_full, "sky_aim": sky_aim * GHS_SKY_SHARE, "neb_aims": aims,
           "b_max": b_max, "hdr": hdr, "boost": boost}
    info.update(anchors=[round(b, 7) for b in bps], sky_offset=round(s0, 4), white=round(white, 7), sky_in=round(sky, 4), noise_in=round(noise, 5), band_in=round(band, 6),
                nebula_in=round(neb, 4), bright_in=None if bright is None else round(bright, 4), signal_frac=round(float(sig_mask.mean()), 4))
    if luma_ratio and len(dn) == 3:
        # broadband colour: ONE stretch of the luminance, the colour carried as the (smoothed) ratio of every plane to the luminance (VeraLux HyperMetric Stretch's
        # vector colour preservation, ratio = channel / lightness of the anchored data; the lightness is the maximum channel, not Rec. 709 luminance: a red nebula
        # at lightness 0.4 needs R = 1.9 in luminance terms, every such pixel was gamut-mapped to the same R = 1 and the nebula came out a flat salmon plateau, NGC 1499): independent per-channel curves turn the planes' own noise and
        # calibration residuals into rainbow blotches (NGC 1499 before BGE / SPCC: colourfulness 0.58)
        u = [np.clip((a - np.float32(b0)) / np.float32(white), 0.0, None) for a, b0 in zip(dn, bps)]
        sg_c = max(LUMA_RATIO_SMOOTH, COLOR_SIGMA_FRAC * max(dn[0].shape))
        us = [cv2.GaussianBlur(np.ascontiguousarray(x), (0, 0), sg_c) for x in u]
        lu = np.clip(np.max(np.stack(u), axis=0), 0.0, 1.0)             # the lightness is the brightest channel (HSV value): a saturated colour keeps its gradation
        (ls_full,), srec = stretcher.stretch([lu], ctx)
        info["ghs"] = srec.get("ghs_fit")
        info["stretch_record"] = srec
        out = [np.empty_like(dn[0], dtype=np.float32) for _ in range(3)]
        for r0, r1 in _bands(dn[0].shape[0]):
            lus = np.maximum(np.max(np.stack([x[r0:r1] for x in us]), axis=0), 1e-4)
            ls_ = np.clip(ls_full[r0:r1], 0.0, 1.0)
            cols = np.stack([np.clip(x[r0:r1] / lus, 0.0, 4.0) * ls_ for x in us])
            cols = _gamut_map(cols, ls_, GAMUT_DARKEN)
            for c in range(3):
                out[c][r0:r1] = np.clip(cols[c], 0.0, 1.0)
        return out, info
    us = [np.clip((a - np.float32(b0)) / np.float32(white), 0.0, 1.0) for a, b0 in zip(dn, bps)]
    if compose is not None:
        # ``compose(us)`` -> (the mono lightness plane to stretch, extra): ONE stretch of one plane by the chosen tool (every community stretch tool is a lightness tool: Siril ght /
        # autoghs / modasinh, Statistical_Stretch, VeraLux HyperMetric Stretch all run on a mono image here), the colour travels as ratios (``extra``)
        light_u, extra = compose(us, info)
        ctx["lum"] = light_u
        (lt,), srec = stretcher.stretch([light_u], ctx)
        info["ghs"] = srec.get("ghs_fit")
        info["stretch_record"] = srec
        info["_extra"] = extra
        return [np.clip(lt, 0, 1).astype(np.float32)], info
    out, srec = stretcher.stretch(us, ctx)
    info["ghs"] = srec.get("ghs_fit")
    info["stretch_record"] = srec
    info["_u"] = us
    return [np.clip(o, 0, 1).astype(np.float32) for o in out], info


# --------------------------------------------------------------------------- the whole non-linear render

def is_narrowband(filter_name: str | None) -> bool:
    """A dual-band (Ha + OIII) filter: Seestar LP, DWARF Duo-Band, L-eNhance / L-eXtreme, ..."""
    import re
    n = re.sub(r"[^A-Z0-9]", "", (filter_name or "").upper())
    return n in {"LP", "DUO", "DUAL", "DUOBAND", "DUALBAND", "LENHANCE", "LEXTREME", "LULTIMATE", "HOO"}


def resolve_palette(palette: str, dual_band: bool) -> str:
    """``auto`` = natural for broadband, HOO for dual-band filters (the old DEFAULTS semantics); foraxx / hoo_warm only when asked."""
    if palette in (None, "", "auto"):
        return "hoo" if dual_band else "natural"
    return palette if palette in PALETTES else "natural"


def adapted_target(stretch: float, coverage: float, adapt: bool) -> float:
    """Target median for the sky.  ``auto_stretch``: an object that fills less than 30 % of the frame gets a
    darker sky, down to 0.11 / 0.16 of the setting (DESIGN section 4: 0.16 -> 0.11 for small objects;
    Under South West Skies / Statistical_Stretch guidance: 0.25 full frame, ~0.11 when the target does not
    fill it)."""
    if not adapt:
        return float(stretch)
    return float(stretch) * float(np.clip(0.69 + 0.31 * coverage / 0.3, 0.69, 1.0))


def final_color_numpy(img: np.ndarray, p: dict, dual_band: bool) -> np.ndarray:
    """satu / rmgreen in numpy (the preview path and the fallback of the Siril step).  Applied to the starless layer only:
    the stars have their own colour control (``star_saturation``), and a saturation of 2 on a teal star turns it
    into a green dot (seen on M 42 when the saturation was applied after the screen blend)."""
    img = satu_np(img, float(p["saturation"]) - 1.0)
    if not dual_band and float(p["scnr"]) > 0:                # no SCNR on dual-band: the green IS the OIII
        img = rmgreen_np(img, float(p["scnr"]))
    return np.clip(img, 0, 1)


def levels_numpy(img: np.ndarray, p: dict) -> np.ndarray:
    """Black point + midtones on the finished picture (Siril ``mtf low mid 1``)."""
    bp, mid = float(p["black_point"]), brightness_to_mid(float(p["brightness"]))
    if bp > 0 or abs(mid - 0.5) > 1e-4:
        img = mtf_np(img, bp, mid, 1.0)
    return np.clip(img, 0, 1)


SKY_PERCENTILE = 0.5               # the darkest sky of the colour planes
AI_RESIDUAL_GAIN = 2.0              # the grain an AI denoiser leaves is spatially correlated (3-8 px): twice the BayesShrink threshold removes it, 1x left a crisp mottle on IC 1805,
                                   # 4 scales instead of 3 turned the nebula to watercolour
FINE_SCALES = 3                   # starlet scales 1-3 (1-8 px) = the grain an AI denoiser leaves behind
COLOR_SIGMA_OIII_FRAC = 0.006     # the OIII colour plane's resolution = this x the long side (23 px on a 3800 px stack): its noise is averaged over ~(2 sigma)^2 pixels, the cores are 100+ px
OIII_FLAT_FRAC = 0.08             # window of the OIII background model's median = this x the long side (300 px on a 3800 px stack)
OIII_COLOUR_K = 2.5               # soft threshold of the colour-resolution OIII, in sigmas of its sky (measured below the sky mode); what stands above it is OIII
OIII_RESIDUAL_K = 1.5             # the OIII plane's 128+ px structure is soft-thresholded at this many robust sigmas of that structure (large_scale_denoise ``residual_k``)
OIII_DENOISE = 1.5                # the OIII plane is denoised 1.5 x harder than Ha: its signal is the weaker (IC 1805: 2-3 x lower SNR) and it only carries colour
COLOR_SIGMA_FRAC = 0.003          # colour resolution = this x the long side (11 px on a 3800 px DWARF stack): the colour (Ha : OIII ratio) is carried at lower resolution
                                  # than the Ha luminance (LRGB principle: the eye resolves lightness, not colour; chroma is binned 3-4x in video and JPEG alike); the OIII
                                  # noise left after the denoise is averaged over ~(2 sigma)^2 pixels, so the sky colour has no blotches


SKY_NEUTRAL_FRACTION = 30.0        # the darkest 30 % (smoothed lightness) are the sky samples of the display-level background neutralisation


def neutralize_display_sky(rgb: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Background neutralisation of the display picture: the per-channel medians of the sky samples (darkest ``SKY_NEUTRAL_FRACTION`` % of the 3 px smoothed
    lightness) are brought to the lowest of them by subtracting the difference from the whole channel (offsets only, what Siril's / PixInsight's background
    neutralisation does on linear data; a constant per channel, so no colour is faded by how significant a line is).  The colour planes were neutralised on their own
    darkest 5 % (``colour_planes``), but the composed picture's sky still sat 17 % teal on IC 1805 (the dark sky of the Ha plane is not the dark tail of the OIII plane)."""
    lum = cv2.GaussianBlur(np.ascontiguousarray(luminance(rgb)), (0, 0), 3.0)
    sm = lum[::2, ::2]
    thr = np.percentile(sm, SKY_NEUTRAL_FRACTION)
    mask = lum <= thr
    med = np.array([float(np.median(rgb[c][mask])) for c in range(3)], np.float32)
    off = med - med.min()
    return np.clip(rgb - off[:, None, None], 0.0, 1.0), [round(float(o), 4) for o in off]


def ha_luminance(rgb: np.ndarray, ha: np.ndarray, amount: float) -> np.ndarray:
    """"HOO with Ha as luminance" (community LHOO / synthetic-luminance recipe for dual-band OSC): the colour is the HOO palette's own, its lightness (the brightest
    channel) moves towards ``ha`` (the stretched Ha, or its signal-to-noise weighted mix with the OIII: the high-SNR lightness) by ``amount`` (0..1), so the noise of the
    OIII plane cannot reach the picture's lightness.  Every pixel is scaled so that its brightest channel is ``amount x ha + (1 - amount) x its own``: no channel can
    exceed 1, hue and saturation are kept (a Rec. 709 lightness needed R = 1.9 for a red pixel of lightness 0.4, and flattened every Ha-red region to R = 1)."""
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0:
        return rgb
    out = np.empty_like(rgb)
    for r0, r1 in _bands(rgb.shape[1]):
        b = rgb[:, r0:r1]
        mx = b.max(0)
        lt = np.clip(amount * ha[r0:r1] + (1.0 - amount) * mx, 0.0, 1.0)
        out[:, r0:r1] = b * (lt / np.maximum(mx, 1e-6))[None]
    return np.clip(out, 0.0, 1.0)


def denoise_settings(p: dict) -> tuple[float, int | None]:
    """(wavelet amount, scales shrunk) of the render's own noise reduction: ``luminance_denoise`` / 0.6 x the BayesShrink threshold on every scale.  When the restore
    stage already ran an AI denoiser (``_denoised_linear``) only the grain it leaves on the fine scales (``FINE_SCALES``: 1-8 px, spatially correlated) is shrunk, at
    ``AI_RESIDUAL_GAIN`` x the threshold, and the coarse scales are left alone: one noise reduction followed by a cleaning, not two reductions (a full-strength wavelet
    pass on top of the AI denoiser wiped the Ha plateau and the filaments of IC 1805 into watercolour and left specks)."""
    amount = float(np.clip(float(p.get("luminance_denoise", 0.6)) / 0.6, 0.0, 2.0))
    if p.get("_denoised_linear"):
        return amount * AI_RESIDUAL_GAIN, FINE_SCALES
    return amount, None


STRETCH_METHODS = ("ghs", "autoghs", "modasinh", "autostretch", "statstretch", "hms")     # = stretch.METHODS (Siril commands / Statistical_Stretch.py); mtf, asinh, log = the legacy numpy statistical stretches


def uses_siril_stretch(p: dict) -> bool:
    return str(p.get("stretch_method", "ghs")) in STRETCH_METHODS


def _stretch_kwargs(p: dict) -> dict:
    return {"method": str(p.get("stretch_method", "ghs")) if uses_siril_stretch(p) else "ghs", "hdr": float(p.get("hdr", 0.0)), "boost": float(p.get("stretch_boost", 0.0)),
            "large_scale": float(np.clip(float(p.get("luminance_denoise", 0.6)) / 0.6, 0.0, 2.0))}


def dual_band_planes(base: np.ndarray, p: dict, stretcher=None):
    """Ha and OIII of a dual-band starless layer, denoised once (OIII harder), the lightness = the Ha (with the OIII's share where it is as clean as the Ha) stretched by ONE tool
    (``stretcher``: Siril ght / Statistical_Stretch / VeraLux HyperMetric Stretch ..., fitted on the Ha's sky and nebula), the colour = the smoothed ratio of the linear lines
    (``ratio_colour``).  Returns ``(stretched lightness, colour (3, H, W) of unit maximum, ghs info, extraction + colour info)``, display units."""
    amount, shrink = denoise_settings(p)
    ha, oiii, pinfo = extract_ha_oiii(base, bool(p.get("oiii_unmix", True)))
    e = EDGE_CROP if min(ha.shape) > 8 * EDGE_CROP else 0
    snr_h, snr_o = (object_snr(x[e:-e, e:-e] if e else x) for x in (ha, oiii))
    share = snr_o / max(snr_h + snr_o, 1e-6)
    pinfo.update(snr_ha=round(snr_h, 2), snr_oiii=round(snr_o, 2), oiii_lightness_share=round(share, 3))
    boost = float(p.get("oiii_boost", 1.0))
    palette = resolve_palette(p.get("palette"), True)

    def compose(us, ginfo_so_far):
        col, cinfo, light_u = ratio_colour(us[0], us[1], float(ginfo_so_far.get("noise_in", 1e-3)), boost, palette, share)
        return light_u, (col, cinfo)

    (light_st,), ginfo = ghs_stretch([ha, oiii], sky_aim=float(np.clip(p.get("stretch", SKY_AIM), 0.02, 0.4)),
                                     b_max=float(np.clip(p.get("contrast", 2.0), 0.25, 13.0)), denoise=[amount, amount * OIII_DENOISE],
                                     mix=lambda ps: ps[0], contrast=float(np.clip(p.get("local_contrast", 0.0), 0.0, 2.0)), shrink_scales=shrink,
                                     stretcher=stretcher, residual_k=[0.0, OIII_RESIDUAL_K], compose=compose, **_stretch_kwargs(p))
    col, cinfo = ginfo.pop("_extra")
    pinfo.update(cinfo, oiii_boost=boost, oiii_threshold_k=OIII_COLOUR_K)
    return light_st, col, ginfo, pinfo


def oiii_background(plane: np.ndarray) -> np.ndarray:
    """Large-scale background of a colour plane whose targets are compact: the running MEDIAN over a window of ``OIII_FLAT_FRAC`` x the long side (on the plane reduced
    16x; a median ignores any structure covering less than half the window, the OIII cores, and follows a rise of the plane towards the frame's edge), smoothed."""
    from scipy.ndimage import median_filter
    h, w = plane.shape
    f = 16
    small = cv2.resize(np.ascontiguousarray(plane, dtype=np.float32), (max(8, w // f), max(8, h // f)), interpolation=cv2.INTER_AREA)
    k = int(max(3, round(OIII_FLAT_FRAC * max(h, w) / f))) | 1
    m = median_filter(small, size=k, mode="reflect")
    m = cv2.GaussianBlur(m, (0, 0), max(1.0, k / 4.0))
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)


def colour_planes(h_st: np.ndarray, o_st: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """The stretched Ha and OIII at colour resolution (``COLOR_SIGMA``), the OIII's SKY brought to the level of the Ha's empty sky (background neutralisation of the
    colour, the offsets-only rule of Siril's / PixInsight's background neutralisation): Ha has a faint haze over a large part of the frame (its mode is the haze, 41 % of
    the frame on IC 1805) and dark lanes well below it, while the OIII is flat there, so the mode of the OIII (its sky) sits ``d_sky`` above the darkest ``SKY_PERCENTILE`` %
    of the Ha (its empty sky): subtracting ``d_sky`` makes the empty sky colourless and leaves the haze red, instead of the teal that equal modes (or equal 5th percentiles
    of two planes with different tails) painted over the dark lanes of IC 1805.  Returns ``(Ha, OIII, d_sky, sigma of the OIII sky)``."""
    sg = max(4.0, COLOR_SIGMA_FRAC * max(h_st.shape))
    h_c = cv2.GaussianBlur(np.ascontiguousarray(h_st), (0, 0), sg)
    o_c = cv2.GaussianBlur(np.ascontiguousarray(o_st), (0, 0), max(sg, COLOR_SIGMA_OIII_FRAC * max(h_st.shape)))      # the weak line's colour is the most smoothed one
    hs, os_ = h_c[::4, ::4], o_c[::4, ::4]
    o_mode = _hist_mode(os_)
    h_floor = float(np.percentile(hs, SKY_PERCENTILE))
    # the noise of the colour-resolution OIII from the values BELOW its sky mode (a half Gaussian: the nebula only adds above), and the soft threshold that keeps what stands
    # OIII_COLOUR_K sigmas above the sky (BayesShrink / MLT practice: what is within the noise is not colour); the Ha is not thresholded (its haze is the signal)
    below = os_[os_ < o_mode]
    sig_o = float(np.sqrt(np.mean((below - o_mode) ** 2))) if below.size > 50 else 0.0
    # the OIII's own large-scale background (what the flats / the background extraction leave of a channel-to-channel gradient, 0.03-0.1 % of the sky on IC 1805, which the
    # stretch turns into teal patches of 0.05 display units over a third of the frame): a heavy gaussian of the plane with its signal (above the sky + 3 sigma) clipped away,
    # i.e. a dynamic-background-extraction model of a plane whose targets are compact (the OIII cores); subtracted about the sky mode
    o_c = o_c - (oiii_background(o_c) - np.float32(o_mode))
    thr = np.float32(OIII_COLOUR_K * sig_o)
    e = o_c - np.float32(o_mode)
    e = np.where(e > 0, e * e * e / (e * e + thr * thr + 1e-12), 0.0).astype(np.float32)        # smooth shrinkage e^3 / (e^2 + T^2): ~0 within the noise, ~e above it, no ragged islands
    o_out = np.clip(np.float32(h_floor) + e, 0.0, 1.0)
    return h_c, o_out, float(o_mode - h_floor), float(sig_o)


RATIO_OBJECT_PERCENTILE = 80.0    # the "object" of the normalisation statistic = the brightest 20 % of the Ha at colour resolution
RATIO_EPS_K = 1.0                 # colour-vector floor (the sky grey): this many pixel-noise sigmas (u units) added to both lines before the ratio is taken
RATIO_GAIN_MAX = 20.0


def ratio_colour(u_h: np.ndarray, u_o: np.ndarray, noise_u: float, boost: float, palette: str, share: float = 0.0) -> tuple[np.ndarray, dict, np.ndarray]:
    """The colour of the dual-band picture as the smoothed ratio of the two lines in the LINEAR domain (the lightness is the Ha's own, ``ghs_main``): ``(R, G, B)`` of unit
    maximum channel.  The two lines' excess over their empty sky are taken at colour resolution (Ha: over its floor, the darkest 0.5 %; OIII: over its own large-scale
    background (``oiii_background``) and soft-thresholded at ``OIII_COLOUR_K`` sigmas of its sky), the OIII is scaled by the Blanshan / Cranfield narrowband
    normalisation (NarrowbandNormalization.py: the gain that equalises the two lines' mean deviation over the object, ``boost`` = its strength: 1 = equal, 0 = none, the
    geometric interpolation between), and the palette is composed from ``(Ha + eps, gain x OIII + eps)`` and divided by its maximum channel: ``eps`` (``RATIO_EPS_K`` x the
    pixel noise) makes the ratio of two lines that are both noise a neutral grey.  A ratio is invariant under the stretch, so a bright core keeps the colour it has (stretching
    the lines separately, even at the same curve, drives every core towards white: IC 1805, R:G:B = 1 : 0.27 : 0.23 in the linear data came out 0.85 : 0.75 : 0.75)."""
    sg_h = max(4.0, COLOR_SIGMA_FRAC * max(u_h.shape))
    sg_o = max(sg_h, COLOR_SIGMA_OIII_FRAC * max(u_h.shape))
    h_c = cv2.GaussianBlur(np.ascontiguousarray(u_h, dtype=np.float32), (0, 0), sg_h)
    o_c = cv2.GaussianBlur(np.ascontiguousarray(u_o, dtype=np.float32), (0, 0), sg_o)
    hs, os_ = h_c[::4, ::4], o_c[::4, ::4]
    eh = np.maximum(h_c - np.float32(np.percentile(hs, SKY_PERCENTILE)), 0.0)
    o_mode = _hist_mode(os_)
    below = os_[os_ < o_mode]
    sig_o = float(np.sqrt(np.mean((below - o_mode) ** 2))) if below.size > 50 else 0.0
    e = o_c - oiii_background(o_c)
    thr = np.float32(OIII_COLOUR_K * sig_o)
    eo = np.where(e > 0, e * e * e / (e * e + thr * thr + 1e-12), 0.0).astype(np.float32)
    obj = eh > np.percentile(eh[::4, ::4], RATIO_OBJECT_PERCENTILE)
    e_h = float(eh[obj].mean()) if obj.any() else 1e-9
    e_o = float(eo[obj].mean()) if obj.any() else 0.0
    r0 = e_o / max(e_h, 1e-12)
    gain_full = float(np.clip(1.0 / max(r0, 1e-6), 1.0, RATIO_GAIN_MAX))
    gain = gain_full ** float(np.clip(boost, 0.0, 2.0))
    eps = np.float32(RATIO_EPS_K * max(noise_u, 1e-6))
    comp = compose_palette(eh + eps, np.minimum(gain * eo + eps, 4.0 * (eh + eps) + 1.0), palette)
    col = comp / np.maximum(comp.max(0, keepdims=True), 1e-6)
    # the synthetic luminance (linear): the Ha's own pixels, whose colour-resolution part is replaced by the signal-to-noise weighted mean of the Ha and the normalised OIII
    # (``share`` = the OIII's weight, its SNR over the sum of both: Ha alone when the OIII is noise, both where the OIII is as clean as the Ha), so the OIII's noise never
    # reaches the lightness and an OIII-bright core (M 42's teal haloes) does not go black on pure Ha
    h_floor_v = float(np.percentile(hs, SKY_PERCENTILE))
    light_u = np.clip(u_h - h_c + (1.0 - share) * h_c + share * (np.float32(h_floor_v) + gain * eo), 0.0, 1.0).astype(np.float32)
    return col.astype(np.float32), {"oiii_to_ha": round(r0, 4), "gain_full": round(gain_full, 3), "gain": round(gain, 3), "boost": float(boost), "eps": round(float(eps), 6),
                                    "oiii_sky_sigma_u": round(sig_o, 6), "colour_sigma_px": [round(sg_h, 1), round(sg_o, 1)], "object_frac": round(float(obj.mean()), 3)}, light_u


def ghs_main(base: np.ndarray, p: dict, dual_band: bool, palette: str, stretcher=None) -> tuple[np.ndarray, dict]:
    """The starless layer's display picture by ``ghs_stretch``.

    Dual-band -> ``dual_band_planes`` (Ha / OIII denoised once, one GHS fitted on Ha), the colour at colour resolution (``colour_planes``: LRGB principle, the eye
    resolves lightness, not colour), the OIII normalised to Ha with the Blanshan / Cranfield narrowband normalisation (``nbn_oiii``), composed in the palette, and
    the lightness taken from the full-resolution Ha (``ha_luminance``; ``synthetic_luminance`` = how much), so the noise of the OIII plane cannot reach the picture.
    Broadband / natural -> R, G, B stretched with one GHS (common white point, own black points: a neutral sky and the stars' colours kept), colour carried as the
    smoothed ratio to the luminance.
    Setting meanings: ``stretch`` = the sky's display level, ``contrast`` = the highest GHS local intensity b tried, ``luminance_denoise`` scales the wavelet
    denoise (0 = off), ``local_contrast`` = the SNR-gated wavelet gain."""
    info: dict = {"denoised_by_restore": bool(p.get("_denoised_linear"))}
    if dual_band and palette != "natural":
        light, col, ginfo, pinfo = dual_band_planes(base, p, stretcher)
        main = np.clip(col * light[None], 0.0, 1.0).astype(np.float32)
        main, off = neutralize_display_sky(main)
        pinfo["sky_neutralised_by"] = off
        info["palette_info"] = {**pinfo, **ginfo}
    else:
        amount, shrink = denoise_settings(p)
        pl, ginfo = ghs_stretch([base[0], base[1], base[2]], sky_aim=float(np.clip(p.get("stretch", SKY_AIM), 0.02, 0.4)),
                                b_max=float(np.clip(p.get("contrast", 2.0), 0.25, 13.0)), denoise=amount, contrast=float(np.clip(p.get("local_contrast", 0.0), 0.0, 2.0)),
                                mix=lambda ps: np.max(np.stack(ps), axis=0), luma_ratio=True, shrink_scales=shrink,
                                stretcher=stretcher, **_stretch_kwargs(p))
        main, off = neutralize_display_sky(np.stack(pl))              # background neutralisation of the picture's sky (offsets only): no colour cast of the sky
        ginfo["sky_neutralised_by"] = off
        info["stretch"] = ginfo
    info["ghs"] = ginfo
    return np.clip(main, 0, 1).astype(np.float32), info


def render_layers(base: np.ndarray, stars: np.ndarray | None, p: dict, *, dual_band: bool, pre_stretched: bool = False,
                  stars_stretched: bool = False, hooks: dict | None = None, tick=None) -> tuple[np.ndarray, dict]:
    """Linear (starless or complete) RGB ``base`` + optional linear star layer -> display RGB (3, H, W) in [0,1].

    The order is the one of DESIGN section 4 "Export": wavelet denoise -> GHS stretch of the starless layer (``ghs_main``; legacy mtf / asinh / log
    methods: the statistical stretch) -> [second denoise: NXT] -> chroma smoothing, sharpening, local contrast -> saturation / SCNR
    (starless layer only) -> star layer (asinh stretch, reduction, intensity, saturation) -> screen recombination ->
    black point / midtones.  ``hooks`` replace steps by the Siril / RC-Astro versions in a real export:
    ``contrast(img, target, p)``, ``denoise(img, p)``, ``final_color(img, p, dual_band)``; each returns the
    image or None to keep the numpy result.  ``tick(name)`` reports step starts."""
    hooks = hooks or {}
    info: dict = {}

    def step(name):
        if tick:
            tick(name)

    palette = resolve_palette(p.get("palette"), dual_band)
    use_palette = dual_band and palette != "natural"
    info["palette"] = palette if dual_band else "natural"
    pos = curve_position(p.get("stretch_method", "ghs"), float(p.get("hdr", 0.6)))
    shd = curve_shoulder(p.get("stretch_method", "ghs"), float(p.get("hdr", 0.6)))
    cov = object_coverage(luminance(base)) if (not pre_stretched and p.get("auto_stretch", True)) else 0.0
    target = adapted_target(float(p["stretch"]), cov, bool(p.get("auto_stretch", True)) and not pre_stretched)
    info.update(coverage=round(cov, 4), target_median=round(target, 4), curve_position=round(pos, 3), curve_shoulder=round(shd, 3))
    step("Stretching")
    if not pre_stretched and uses_siril_stretch(p):
        main, ginfo = ghs_main(base, p, dual_band, palette, hooks.get("stretcher"))
        info.update(ginfo)
        info["target_median"] = float(p.get("stretch", SKY_AIM))
    elif pre_stretched:
        main = base
        if use_palette:
            main = compose_palette(base[0], 0.5 * (base[1] + base[2]), palette)
    elif use_palette:
        ha, oiii, pinfo = extract_ha_oiii(base, bool(p.get("oiii_unmix", True)))
        ha_st, hi = stretch_channel(ha, target, position=pos, shoulder=shd)
        o_st, oi = stretch_channel(oiii, target, position=pos, shoulder=shd)
        boost = float(p.get("oiii_boost", 1.0))
        if abs(boost - 1.0) > 1e-3:                       # boost the OIII signal above the sky (stretched domain)
            o_st = np.clip(target + (o_st - target) * np.float32(boost), 0, 1)
        pinfo["oiii_boost"] = boost
        main = synthetic_luminance(compose_palette(ha_st, o_st, palette), float(p.get("synthetic_luminance", 0.8)))
        info["palette_info"] = {**pinfo, "ha": hi, "oiii": oi}
        if CORE_TARGET < target:
            ha_lo, _ = stretch_channel(ha, CORE_TARGET, position=pos, shoulder=shd)
            o_lo, _ = stretch_channel(oiii, CORE_TARGET, position=pos, shoulder=shd)
            if abs(boost - 1.0) > 1e-3:
                o_lo = np.clip(CORE_TARGET + (o_lo - CORE_TARGET) * np.float32(boost), 0, 1)
            alt = synthetic_luminance(compose_palette(ha_lo, o_lo, palette), float(p.get("synthetic_luminance", 0.8)))
            main, info["core_chroma"] = core_chroma(main, alt)
            del ha_lo, o_lo, alt
        del ha, oiii, ha_st, o_st
    else:
        main, sinfo = stretch_rgb(base, target, position=pos, shoulder=shd, color_preservation=float(p.get("color_preservation", 0.7)))
        info["stretch"] = sinfo
    if float(p.get("luminance_denoise", 0)) > 0 and "denoise" in hooks and (pre_stretched or not uses_siril_stretch(p)):
        # the second (stretched-image) noise reduction of the legacy stretches; the GHS render has had its one noise reduction (restore: AI denoiser on the linear
        # data, then the wavelet pass above): NXT 0.8 on top of that is what smeared the nebula into plastic
        step("Noise reduction")
        res = hooks["denoise"](main, p)
        main = main if res is None else res
    step("Detail")
    main, cinfo = chroma_wavelet_denoise(main, float(p.get("chroma_denoise", 0.0)))
    info["chroma_denoise"] = cinfo
    main = sharpen(main, float(p.get("sharpen", 0.0)))
    if not uses_siril_stretch(p) or pre_stretched:
        main = local_contrast(main, float(p.get("local_contrast", 0.0)))
    step("Colour")
    fc = hooks.get("final_color")
    res = fc(main, p, dual_band) if fc else None
    main = res if res is not None else final_color_numpy(main, p, dual_band)
    if uses_siril_stretch(p) and not pre_stretched and str(p.get("stretch_method")) != "statstretch":      # Statistical_Stretch.py has its own HDR compress (-hdr)
        main = hdr_compress(main, float(p.get("hdr", 0.0)))          # the GHS has no highlight shoulder of its own; the legacy curves use ``curve_shoulder``
    bpt = float(p.get("black_point", 0.0))
    if bpt > 0:                                           # black point on the starless layer only: the stars sit on top of the sky afterwards
        main = np.clip((main - np.float32(bpt)) / np.float32(1.0 - bpt), 0, 1)   # (a black point after the blend cut every star that is not far above the sky)
    if uses_siril_stretch(p) and not pre_stretched:
        main, lift = sky_lift(main, float(p.get("stretch", SKY_AIM)))
        info["sky_lift"] = round(lift, 4)
    out = main
    if stars is not None and float(p.get("star_reduction", 0.0)) < 1.0:
        step("Stars")
        if stars_stretched:
            st, stinfo = stars, {"stretched_by": "unscreen"}
        else:
            if dual_band and p.get("star_balance", True):
                stars, binfo = star_color_balance(stars, star_noise_floor(stars))
                info["star_balance"] = binfo
            st, stinfo = stretch_stars(stars, color_preservation=float(p.get("star_color_preservation", 0.5)),
                                       halo_suppress=float(p.get("halo_suppress", 0.6)), amount=float(p.get("star_stretch", STAR_STRETCH)),
                                       fwhm_px=float(p.get("_fwhm_px") or 3.0))
        st = scale_saturation(st, float(p.get("star_saturation", 1.0)))
        st, gain = reduce_stars(st, float(p.get("star_reduction", 0.0)), float(p.get("star_intensity", 1.0)))
        out = screen(main, st)
        info["stars"] = {**stinfo, "gain": round(gain, 4)}
        del st
    out = levels_numpy(out, {**p, "black_point": 0.0})    # midtones (brightness) on the finished picture
    return np.clip(out, 0, 1).astype(np.float32), info
