"""The finishing chain of the default render: the community "professional workflow" (research/pro-workflow.md, Chain A broadband / Chain B dual-band), every step a named
tool with its published parameters, an intermediate picture saved after each step, the exact command of each step recorded.

Input: the LINEAR starless image and the LINEAR star image (``post.remove_stars``: SXT on linear data, no unscreen), after the linear stage of ``post.restore`` (GraXpert BGE ->
BXT correct only -> plate solve -> SPCC [narrowband mode for dual-band] -> BXT sharpen 0.25-0.35 / 0.5-0.6 -> NXT linear 0.5-0.8).  Output: the display image (3, H, W).

=====  ==============================================================================================================================================================
step   tool and published parameters (source)
=====  ==============================================================================================================================================================
1      starless stretch, background -> 0.10 (Brecher, five finished images 0.09-0.11):  ``stretch_method=ghs2``: GHS two-pass recipe by Siril ``ght`` (Siril GHS tutorial
       https://siril.org/tutorials/ghs/, ghsastro https://www.ghsastro.co.uk/doc/tools/GeneralizedHyperbolicStretch/GeneralizedHyperbolicStretch.html): pass 1 SP just
       left of the histogram peak (the histogram's left half-maximum), b 10, LP 0, HP 1, D raised until the background reaches 0.10 after the black point and pass 2;
       ``linstretch -BP`` at the histogram toe; pass 2 SP on the subject's flat area (median of the nebula mask), b 4 (3-5), D 1 (small), LP = the background, HP < 1
       when ``hdr`` is on.  Alternatives: ``statstretch`` = ``pyscript Statistical_Stretch.py -median T -sigma 5 -linked`` (Seti Astro; T 0.10 for a small target, 0.25
       when the target fills the frame), ``autoghs``, ``modasinh``, ``autostretch``, ``hms`` (one parameter raised until the sky reaches 0.10).
2      highlights: GHS HP < 1 in pass 2 / Statistical Stretch ``-hdr -hdramount 0.15 -hdrknee 0.75`` / Wavescale HDR (``compute_wavescale_hdr``: 5 scales, compression 1.5,
       mask gamma 5.0, Seti Astro, HDR_multiscale.py), switched on by ``hdr`` (autofinish: clipped > 0.05 % or lightness >= 0.97 > 0.05 %).
3b     ``cc_sharpen``: Cosmic Clarity Sharpen (Both / Stellar Only / Non-Stellar Only, the script's amounts) on the stretched image after the recombination (step 7).
3      NoiseXTerminator on the stretched starless (RC-Astro / Brecher ``--dn`` 0.6-0.85) with its own colour separation (``--di`` / ``--dc`` at 0.95); fallback GraXpert
       denoise 0.6-0.8.  ``cc_denoise``: Cosmic Clarity Denoise instead (luminance / colour strengths = the same two settings; model ``cc_denoise_model``).  (This replaces any chroma shrinkage of our own.)
4      palette (dual-band): Hubble_Palette_from_Dual-Band_OSC.py itself (siril-scripts, run headless by ``hubble_palette_worker.py``): HOO (default; Classic Ha = R,
       OIII = (G + B) / 2, or Improved OIII = (G + B - 0.3 R) / 2) and the Hubble-like SHO / HSO / OSH / OHS / HOS with its synthetic SII, formula preset
       ``hubble_formula`` (Classic / Improved / Advanced / NonLinear S2); Foraxx (The Coldest Nights) and HOO warm by Siril ``pm``; NBN (Blanshan / Cranfield,
       Cuiv's script: ``nbn.py``) as the palette ``nbn``, with its OIII boost ``oiii_boost``; SCNR (``scnr``) after the Ha-in-green palettes SHO / OHS.  Broadband: ``rmgreen 3 <scnr>`` (SCNR).
5      local contrast under a background mask, after the noise reduction: Siril ``clahe`` (contrast limit 1.5 = Brecher's LHE limit), 150 px then 40 px tiles
       (Brecher M51 / IC 1848 / Sh2-140 scales), blended at 0.4 x ``local_contrast`` (0.5 -> 0.2 = the lowest published strengths 0.18-0.2).
6      saturation: ``ght -sat`` (Siril's GHS saturation stretch, ghsastro) with D fitted so the subject's mean HSL saturation is multiplied by the ``saturation`` setting, under the
       range mask, after the chroma noise reduction (``satu``'s linear gain, gated at median + sigma, left faint subjects grey: C 33 2026-10-15).
7      stars: Seti Astro Star Stretch = ``mtf 0 1/(3^a+1) 1`` on the linear stars (a = 5: ``mtf 0 0.004098 1``); dual-band: invert, ``rmgreen``, invert (Brecher, IC 1848);
       reduction by the DSA star-reduction formula (siril-scripts DSA-Star_Reduction.py, Rich Stevenson: value 0.2 default); recombination by screen
       ``pm "~(~$starless$*~$stars$)"`` (RC-Astro, Brecher).
8      black point at the histogram toe (``linstretch -BP``), ``icc_assign sRGB`` + ``icc_convert_to sRGB relative`` (Brecher: ICC to sRGB, relative colorimetric).
=====  ==============================================================================================================================================================

The preview runs this class without the external AI steps (``external=False``: step 3 is skipped) on the same data.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import subprocess
import threading
import time

import cv2
import numpy as np

from . import config, export as export_mod, finish, fitsio, gates, nbn, tools, workflow
from .siril import SirilError, SirilRunner
from .tools import ToolError, ToolUnavailable
from .util import Cancelled

# --- published values (source in the module docstring and at each use)
SKY_BAND = (0.09, 0.11)            # Brecher's finished backgrounds
SKY_TARGET = 0.10                  # background after the stretch (Brecher: 0.10 / 0.11 / 0.09 / 0.11 / 0.10)
GHS1_B = 10.0                      # pass 1 local intensity: b 15 -> 10 (ghsastro "perhaps around 10")
GHS2_B = 4.0                       # pass 2: b 3-5 (ghsastro) / 2-6 (Siril); the ``contrast`` setting
GHS2_D = 2.5                       # pass 2 D (Siril -D scale 0-10): the ``ghs_d2`` setting's starting value; autofinish searches it on the image
TOE_PERCENTILE = 0.1               # "linear black point just left of the histogram rise": the level below which 0.1 % of the pixels lie (the clipping limit of Siril / fas37, VeraLux)
HP_PERCENTILE = 99.5               # HP < 1 (highlight protection) starts at this percentile of the subject
STAT_SMALL, STAT_FILLING = 0.10, 0.25   # Statistical_Stretch median target: small target (pro-workflow.md A8 route 2 / SW Skies) / frame-filling target (script default)
STAT_FILLING_COVERAGE = 0.5        # the object covers at least this fraction of the frame
STAT_SIGMA, STAT_HDR_AMOUNT, STAT_HDR_KNEE = 5.0, 0.15, 0.75
WAVESCALE = (5, 1.5, 5.0)          # compute_wavescale_hdr: scales, compression, mask gamma (HDR_multiscale.py defaults / SASpro)
NXT_DC = 0.95                      # NXT colour denoise (manual: colour 90-100 %)
NXT_DN = (0.60, 0.85)              # NXT --dn on stretched starless (RC-Astro / Brecher); 1.0 is the tool's own ceiling, not a tuned value
CLIP_CLOSE_STEPS = 4               # bisection steps of the closing clipping step (1/16 of the asked sharpening amount)
DENOISE_ESCALATION = 3             # extra runs of the stretched-image denoiser, at most, when the sky it leaves is noisier than
                                   # sky_pixel_noise allows: the ladder from the requested amount to the tool's ceiling in this
                                   # many steps, so a frame that cannot be denoised clean costs a bounded amount of time and is
                                   # then left alone with every attempt recorded
CHROMA_SIGMA_MAX = 2.0                 # px, the chroma kernel at ``chroma_denoise`` 1.0: our mapping, because NXT's --dc and
                                       # Cosmic Clarity's --denoise-color publish no kernel scale (manual: colour 90-100 %)
CLAHE_CLIP = 1.5                   # Brecher's LHE contrast limit
CLAHE_SCALES = (150, 40)           # px, Brecher's LHE scales (M51: 150 then 40; IC 1848 40 then 150)
LC_WEAKEST = 0.5                   # ``local_contrast`` at Brecher's lowest published LHE strength (x LC_STRENGTH = 0.2): the closing clipping step's floor
LC_STRENGTH = 0.4                  # blend per unit of ``local_contrast`` (0.5 -> 0.2, the lowest published strength)
SAT_B, SAT_SP = 0.0, 0.0           # GHS saturation stretch (ght -sat): the plain hyperbolic curve on the saturation channel, strongest at 0 (grey gains most, saturated pixels are compressed, nothing clips)
SAT_D_MAX = 10.0                   # Siril ght -D range (0-10): the fit's bound
SAT_FACTOR_MAX = 4.0               # the ``saturation`` setting's range: the subject's mean saturation x 1 .. x 4
SAT_MEAN_MAX = 0.9                 # the fit's target never exceeds this mean saturation
STAR_STRETCH = 5.0                 # Seti Astro Star Stretch amount (0-8, default 5)
MASK_SIGMA = 8.0                   # px, smoothing of the lightness the nebula / local contrast masks are taken from
MASK_K = 3.0                       # the subject = more than 3 sigma of the smoothed sky scatter above the sky (finish.object_coverage)
ALT_METHODS = ("autoghs", "modasinh", "autostretch", "hms")
# Hubble_Palette_from_Dual-Band_OSC.py (siril-scripts processing/, Carlo Mollicone - AstroBOH, v2.0.4 read): the dual-band palette step runs the script itself
# (hubble_palette_worker.py); its palettes (rgbcomp R G B of Ha / synthetic SII / OIII) and formula presets (Ha, OIII and SII from the OSC's R, G, B by pm)
HUBBLE_SCRIPT = "processing/Hubble_Palette_from_Dual-Band_OSC.py"
HUBBLE_PALETTES = {"hoo": "HOO", "hoo_improved": "HOO", "sho": "SHO", "hso": "HSO", "osh": "OSH", "ohs": "OHS", "hos": "HOS"}
HUBBLE_MAPPING = {"HSO": ("Ha", "S2_synt", "Oiii"), "SHO": ("S2_synt", "Ha", "Oiii"), "OSH": ("Oiii", "S2_synt", "Ha"), "OHS": ("Oiii", "Ha", "S2_synt"),
                  "HOS": ("Ha", "Oiii", "S2_synt"), "HOO": ("Ha", "Oiii", "Oiii")}
HUBBLE_PRESETS = {"classic": "Classic", "improved": "Improved", "advanced": "Advanced", "nonlinear_s2": "NonLinear S2"}
HUBBLE_FORMULAS = {      # the script's preset_formulas, verbatim (shown in the sidecar; the fallback runs them when the script itself cannot)
    "Classic": {"HA": "R", "OIII": "(G + B) * 0.5", "S2": "(HA + OIII) * 0.5"},
    "Improved": {"HA": "R", "OIII": "(G + B - 0.3 * R) * 0.5", "S2": "R * 0.3"},
    "Advanced": {"HA": "R", "OIII": "max((G + B - 0.3 * R) * 0.5, 0)", "S2": "(R * 0.4 + ((G + B) * 0.1)) * 0.6"},
    "NonLinear S2": {"HA": "R", "OIII": "(G + B) * 0.5", "S2": "pow(R, 1.4) * 0.3"},
}
HA_GREEN_PALETTES = ("sho", "ohs")
CC_SHARPEN_AMOUNT = 0.5            # Cosmic Clarity Sharpen stellar / non-stellar amount: CosmicClarity_Native.py's GUI and CLI default
  # Ha in the green channel: the palettes NarrowbandNormalization.py applies SCNR to
SHARPEN_SIGMA = 1.0                # px, Siril unsharp sigma for the final sharpening (PixInsight UnsharpMask default 2.0 px std dev at 1x; 1.0 for 2.3-2.8"/px smart-scope data)
SHARPEN_GAIN = 0.5                 # unsharp amount per unit of ``sharpen`` (0.25 -> 0.125: Brecher's MMT sharpening 0.03-0.06 per layer is the lower end)
UNPURPLE_THRESH = 0.4              # Siril unpurple threshold (its default)
SKY_LUMP_CELL = 128                 # px, the cell the sky's unevenness is measured on (recorded with the stretch)
FRAME_FILLING = 0.5                # the object covers at least this fraction of the frame: ``auto_stretch`` raises the background to the top of Brecher's band


def ghs_colour_model(color_preservation: float) -> str:
    """Siril ``ght`` colour stretch model from the ``color_preservation`` setting (Siril docs, ght: ``-independent`` stretches each channel, ``-even`` an even-weighted
    luminance, ``-human`` the human-weighted luminance; the luminance models keep the colour ratios, i.e. preserve saturation).  0-0.33 independent, 0.33-0.66 even, above human."""
    c = float(np.clip(color_preservation, 0.0, 1.0))
    return "-independent" if c < 0.33 else ("-even" if c < 0.66 else "-human")


# =========================================================================== measurements

def _hist_stats(lum: np.ndarray) -> dict:
    """The histogram landmarks of a linear lightness plane: ``mode`` (peak), ``left`` (its left half-maximum = "just left of the peak"), ``toe`` (0.1 percentile),
    ``sigma`` (lower-half sigma), ``sky`` (median of the darkest 45 % of the 64 px smoothing = the background the gates and Brecher's 0.10 refer to)."""
    x = finish.sample(lum)
    lo, hi = float(np.percentile(x, 0.05)), float(np.percentile(x, 99.5))
    h, e = np.histogram(x, bins=1024, range=(lo, max(hi, lo + 1e-9)))
    hs = np.convolve(h.astype(np.float64), np.ones(9) / 9.0, mode="same")
    k = int(np.argmax(hs))
    j = k
    while j > 0 and hs[j] > hs[k] / 2.0:
        j -= 1
    c = 0.5 * (e[:-1] + e[1:])
    sm = finish._sky_mask(lum, edge=finish.EDGE_CROP)
    sky = float(np.median(lum[sm])) if sm.sum() > 100 else float(c[k])
    return {"mode": float(c[k]), "left": float(c[j]), "toe": float(np.percentile(x, TOE_PERCENTILE)), "sigma": finish.lower_half_sigma(x), "sky": sky}


def sky_lump(arr: np.ndarray) -> float | None:
    """How uneven the **sky** is across the frame: the largest deviation, in percent of the sky, of any
    ``SKY_LUMP_CELL`` cell's lower quartile from the sky's own.  Measured over ``gates.sky_mask`` - the sample the
    export is gated on - so a nebula crossing a cell is not counted as an uneven sky, and reduced to one number so a
    step's verdict fits in its record.  ``None`` when the frame holds too little sky to judge."""
    a = np.clip(np.nan_to_num(np.asarray(arr, np.float32)), 0.0, 1.0)
    if a.ndim != 3 or a.shape[0] not in (3, 4):
        return None
    lum = finish.luminance(a)
    sm = gates.sky_mask(lum)
    if sm.sum() <= 100:
        return None
    h, w = lum.shape
    cell = SKY_LUMP_CELL if min(h, w) >= 5 * SKY_LUMP_CELL else max(16, min(h, w) // 5)
    vals = []
    for i in range(0, w, cell):
        sl = (slice(None), slice(i, min(i + cell, w)))
        v = lum[sl][sm[sl]]
        if v.size > 200:
            vals.append(float(np.quantile(v, 0.25)))
    for i in range(0, h, cell):
        sl = (slice(i, min(i + cell, h)), slice(None))
        v = lum[sl][sm[sl]]
        if v.size > 200:
            vals.append(float(np.quantile(v, 0.25)))
    sky = float(np.quantile(lum[sm], 0.25))
    if len(vals) < 4 or not sky > 1e-4:
        return None
    return round(max(100.0 * (max(vals) - sky) / sky, 100.0 * (sky - min(vals)) / sky), 1)


def nebula_mask(lum: np.ndarray, mode: float) -> tuple[np.ndarray, dict]:
    """The subject: pixels of the smoothed lightness more than ``MASK_K`` sigma (scatter of the darkest 40 % of the smoothed plane) above the sky mode
    (finish.object_coverage's definition)."""
    sm = cv2.GaussianBlur(np.ascontiguousarray(lum, dtype=np.float32), (0, 0), MASK_SIGMA)
    s = finish.sample(sm)
    dark = s[s <= np.percentile(s, 40)]
    sig = finish.mad_sigma(dark) if dark.size > 16 else 0.0
    thr = mode + MASK_K * sig
    m = sm > thr
    if m.mean() < 0.005:                                    # nothing stands out: the brightest 5 % of the smoothed plane
        thr = float(np.percentile(s, 95))
        m = sm > thr
    return m, {"threshold": thr, "coverage": float(m.mean()), "sigma_smoothed": sig}


def _sigma(v: np.ndarray) -> float:
    """A class of pixels' own noise: 1.4826 x its median absolute deviation, robust to the nebulosity a cell or a border can carry."""
    return (1.4826 * float(np.median(np.abs(v - np.median(v))))) or 1e-12


def level_stats(arr: np.ndarray) -> dict:
    lum = finish.luminance(arr)
    x = finish.sample(lum)
    out = {"median": round(float(np.median(x)), 5), "p99": round(float(np.percentile(x, 99)), 5),
           "p99.9": round(float(np.percentile(x, 99.9)), 5),
           "max": round(float(arr.max()), 5), "clipped": round(float((arr.max(0) >= 0.995).mean()), 6),
           "white": round(float((lum >= 0.97).mean()), 6)}
    # A colour cast is invisible to every number above: a sky sitting at R 4626 / B 4369 codes and a sky at
    # R = B = 5654 give the same luminance median to three places, and the chain record is the only witness
    # left, after the run, of which step tinted the sky (C 49 2026-10-31: the export's sky is red by one
    # 8-bit code on the nebula side and neutral on the other, and no recorded level moved).  The sky
    # samples are the gate's own mask, so these numbers are comparable with sky_chroma and
    # sky_chroma_quadrants in the same export record.
    a = np.ascontiguousarray(arr, np.float32)
    if a.ndim == 3 and a.shape[0] in (3, 4):
        sm = gates.sky_mask(lum)
        # A layer with no sky in it - the star layer handed to `stars stretch` - is black wherever the stars are
        # not, so its darkest samples are simply the floor and the ratios of its channels are not a colour
        # measurement at all: the IC 1848 chain replay (2026-11-01) recorded R-B = -1.000 at that step, a cast in
        # a frame that has no sky to tint, and the next reader would chase something that was never in the
        # picture.  The frame's own median says whether the layer carries light over most of its area; a layer
        # that sits at black outside a few points has no sky sample to report, so the key is left out and the
        # record says nothing was measured rather than inventing a cast.
        if sm.sum() <= 100:                                              # too few pixels for a mask to mean anything
            out["sky_channels"] = [round(float(np.median(finish.sample(a[c]))), 5) for c in range(3)]
        elif float(np.median(lum)) > 0.0:
            out["sky_channels"] = [round(float(np.median(a[c][sm])), 5) for c in range(3)]
    return out


def autostretch_display(arr: np.ndarray) -> np.ndarray:
    """Siril's ``autostretch -linked`` (shadows clipped at median - 2.8 sigma, midtones balance sending the median to 0.25) of a linear image, for the step pictures."""
    lum = finish.luminance(arr)
    s = finish.sample(lum)
    med = float(np.median(s))
    sig = finish.mad_sigma(s)
    c0 = max(0.0, med - 2.8 * sig)
    x = (med - c0) / max(1.0 - c0, 1e-9)
    t = 0.25
    m = x * (t - 1.0) / (2.0 * t * x - t - x) if x > 0 else 0.5
    u = np.clip((arr - c0) / max(1.0 - c0, 1e-9), 0.0, 1.0)
    den = (2.0 * m - 1.0) * u - m
    return np.clip((m - 1.0) * u / np.where(np.abs(den) < 1e-9, -1e-9, den), 0.0, 1.0).astype(np.float32)


def mtf_balance(m: float, x: np.ndarray) -> np.ndarray:
    """Siril / PixInsight midtones transfer function ``mtf m`` (``(m - 1) x / ((2 m - 1) x - m)``)."""
    den = (2.0 * m - 1.0) * x - m
    return (m - 1.0) * x / den


def star_stretch_m(a: float) -> float:
    """Seti Astro Star Stretch ``out = 3^a x / ((3^a - 1) x + 1)`` is the midtones transfer function with m = 1 / (3^a + 1) (SASpro numba_utils.applyPixelMath_numba;
    algebra checked in tests/test_pro.py): amount 5 -> 0.004098."""
    return 1.0 / (3.0 ** float(a) + 1.0)


def ghs_scalar(x: float, b: float, d_siril: float, sp: float, lp: float = 0.0, hp: float = 1.0) -> float:
    """One value through Siril's ``ght`` (numpy mirror ``finish.ghs``, identical to 1e-6): the parameter search below runs on scalars (the stretch is monotone: the mode,
    toe and median of the stretched image are the stretch of the mode, toe and median)."""
    if d_siril < 1e-9:
        return float(x)
    return float(finish.ghs(np.array([x], np.float64), b, float(np.expm1(d_siril)), sp, lp, hp)[0])


def hsl_saturation(arr: np.ndarray) -> np.ndarray:
    """HSL saturation of a CHW image: what Siril's ``ght -sat`` stretches (verified: the numpy GHS of this plane reproduces ``ght -sat`` to 1e-6, lightness unchanged)."""
    mx, mn = arr.max(0), arr.min(0)
    l, d = 0.5 * (mx + mn), mx - mn
    return np.where(d > 1e-6, d / np.maximum(1.0 - np.abs(2.0 * l - 1.0), 1e-6), 0.0).astype(np.float32)


def fit_saturation_d(s_subject: np.ndarray, factor: float, weight: np.ndarray | None = None) -> tuple[float, float]:
    """(Siril -D, target): D of ``ght -D -B=SAT_B -SP=SAT_SP -sat`` such that the mean of the subject's saturation values, after the blend by the range mask
    ``weight`` (s + w (ghs(s) - s)), becomes ``factor`` x the present mean (capped at ``SAT_MEAN_MAX``); bisection on the numpy mirror.  D 0 when already there."""
    s = np.asarray(s_subject, np.float64)
    w = np.ones_like(s) if weight is None else np.asarray(weight, np.float64)
    now = float(s.mean())
    target = min(SAT_MEAN_MAX, now * factor)
    if target <= now + 1e-4:
        return 0.0, target
    mean_at = lambda d: float((s + w * (finish.ghs(s, SAT_B, float(np.expm1(d)), SAT_SP, 0.0, 1.0) - s)).mean())      # noqa: E731
    lo, hi = 0.0, SAT_D_MAX
    if mean_at(hi) <= target:
        return hi, target
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if mean_at(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), target


def fit_ghs2(h: dict, neb_med: float, neb_hi: float, *, b1: float, b2: float, d2: float, target: float, use_hp: bool) -> dict:
    """D of pass 1: raised until the background (``h['sky']``) of the image after pass 1, the black point at the toe and pass 2 sits at ``target``.  Bisection on the scalar
    chain (the one free parameter; SP, LP, HP, b of both passes follow the published rules from the histogram and the nebula mask)."""
    sp1, mode, toe = h["left"], h["sky"], h["toe"]

    def chain(d1: float) -> dict:
        f1 = lambda v: ghs_scalar(v, b1, d1, sp1)                       # noqa: E731
        bp = f1(toe)
        g = lambda y: max((y - bp) / max(1.0 - bp, 1e-9), 0.0)          # noqa: E731  linstretch -BP
        sky1, sp2 = g(f1(mode)), g(f1(neb_med))
        hp2 = g(f1(neb_hi)) if use_hp else 1.0
        lp2 = min(sky1, sp2)
        hp2 = max(hp2, sp2)
        sky2 = ghs_scalar(sky1, b2, d2, sp2, lp2, hp2)
        return {"d1": d1, "bp": bp, "sky1": sky1, "sp2": sp2, "lp2": lp2, "hp2": hp2, "sky2": sky2}
    lo, hi = 0.0, 10.0
    if chain(hi)["sky2"] < target:                                       # unreachable: the strongest stretch Siril allows
        return {**chain(hi), "note": "D at its maximum (10): the background stays below the target"}
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if chain(mid)["sky2"] < target:
            lo = mid
        else:
            hi = mid
    return chain(0.5 * (lo + hi))


# =========================================================================== the chain

def mtf_curve(x: np.ndarray, m: float) -> np.ndarray:
    """Siril's midtones transfer function, mtf(x; m) = (m - 1) x / ((2m - 1) x - m), on [0, 1]."""
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.0, 0.0, np.where(x >= 1.0, 1.0, (m - 1.0) * x / ((2.0 * m - 1.0) * x - m))).astype(np.float32)


def _chroma_ray(lum: np.ndarray, off: np.ndarray) -> np.ndarray:
    """``lum + off`` taken along the chroma ray, shortened rather than clamped so the lightness cannot move.

    ``Finisher.chroma_denoise`` adds a per-channel offset to every pixel, and wherever a channel already sits at the
    top or the bottom of the range that sum leaves it.  Clamping those pixels back is precisely what breaks the
    invariant the step is built on - it changes the lightness (measured 2.38e-3 on a synthetic star core, where the
    chroma blur pulls colour in from the brighter neighbourhood and pushes the pinned channel over the top).
    Shortening instead - taking the largest multiple of the offset that keeps all three channels in range - keeps
    sum(w_k out_k) = lum by construction, and a pixel that cannot take the colour simply takes less of it.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        down = np.where(off < 0, lum / np.maximum(-off, 1e-12), np.inf)
        up = np.where(off > 0, (1.0 - lum) / np.maximum(off, 1e-12), np.inf)
    a = np.minimum(1.0, np.nanmin(np.minimum(down, up), axis=0))
    return lum + a * off


_ALL = object()                    # a step iterated the whole parameter dict: every parameter is one of its inputs
_MISSING = object()


class _ReadLog(dict):
    """The finishing parameters, noting which keys a step reads (``reads``: None = not recording) and whether it wrote any (``wrote``).

    Reads through ``[]`` / ``get`` / ``in`` record the key; iterating, copying or comparing the dict records every key (``_ALL``), so
    ``{**p}`` / ``dict(p)`` / ``json.dumps(p)`` inside a step make the whole dict its input rather than escaping the log."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.reads: set | None = None
        self.wrote = False

    def _note(self, key) -> None:
        if self.reads is not None:
            self.reads.add(key)

    def __getitem__(self, key):
        self._note(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._note(key)
        return super().get(key, default)

    def __contains__(self, key):
        self._note(key)
        return super().__contains__(key)

    def __iter__(self):
        self._note(_ALL)
        return super().__iter__()

    def keys(self):
        self._note(_ALL)
        return super().keys()

    def items(self):
        self._note(_ALL)
        return super().items()

    def values(self):
        self._note(_ALL)
        return super().values()

    def __len__(self):
        self._note(_ALL)
        return super().__len__()

    def copy(self):
        self._note(_ALL)
        return dict(super().items())

    def __eq__(self, other):
        self._note(_ALL)
        return super().__eq__(other)

    __hash__ = None

    def _write(self):
        self.wrote = True

    def __setitem__(self, key, value):
        self._write()
        super().__setitem__(key, value)

    def __delitem__(self, key):
        self._write()
        super().__delitem__(key)

    def update(self, *a, **k):
        self._write()
        super().update(*a, **k)

    def pop(self, *a):
        self._write()
        return super().pop(*a)

    def setdefault(self, key, default=None):
        self._write()
        return super().setdefault(key, default)


def _copy_value(v):
    if isinstance(v, np.ndarray):
        return v.copy()
    if isinstance(v, tuple):
        return tuple(_copy_value(x) for x in v)
    return v


def _nbytes(v) -> int:
    if isinstance(v, np.ndarray):
        return int(v.nbytes)
    if isinstance(v, tuple):
        return sum(_nbytes(x) for x in v)
    return 0


class StepMemo:
    """The outputs of the finishing chain's steps for one set of loaded layers, so a preview re-runs only the steps a changed setting reaches.

    A step's entry is keyed by the key of its input (the chain of steps and settings that produced it) and holds the parameter values the step
    read while it ran (``_ReadLog``), its output, the records it appended and the attributes and ``info`` entries it set.  A later render takes
    the entry only when every parameter the step read has the same value now, so the picture is the one the full chain would give: the steps
    are deterministic functions of their input array and those parameters.  Bounded by bytes (least recently used dropped first):
    ``ASTROPHOTO_PREVIEW_MEMO_MB``, default 3072."""

    def __init__(self, limit_bytes: int | None = None):
        if limit_bytes is None:
            limit_bytes = int(float(os.environ.get("ASTROPHOTO_PREVIEW_MEMO_MB", 3072) or 0) * 2**20)
        self.limit = limit_bytes
        self.entries: "collections.OrderedDict[str, dict]" = collections.OrderedDict()
        self.size = 0
        self.lock = threading.Lock()
        self.hits = self.misses = 0

    def find(self, key: str, name: str, p: dict) -> dict | None:
        with self.lock:
            for k in [k for k, e in self.entries.items() if e["in"] == key and e["name"] == name]:
                e = self.entries[k]
                if all(dict.get(p, r, _MISSING) == v for r, v in e["reads"].items()):
                    self.entries.move_to_end(k)
                    self.hits += 1
                    return e
            self.misses += 1
            return None

    def store(self, entry: dict) -> None:
        n = _nbytes(entry["value"])
        if n > self.limit:
            return
        with self.lock:
            old = self.entries.pop(entry["out"], None)
            if old is not None:
                self.size -= old["bytes"]
            entry["bytes"] = n
            self.entries[entry["out"]] = entry
            self.size += n
            while self.size > self.limit and self.entries:
                _, e = self.entries.popitem(last=False)
                self.size -= e["bytes"]


class Finisher:
    """One run of the chain.  ``external`` = the AI steps (NoiseXTerminator / GraXpert on the stretched image) run; the preview passes False.
    ``memo``: a ``StepMemo`` the steps' outputs are taken from and kept in (the preview); ``measure_steps`` False leaves the levels out of the
    chain records (nothing but the export's sidecar reads them)."""

    def __init__(self, work: str, header, p: dict, dual: bool, *, external: bool = True, progress=None, checkpoint=None, tick=None, device: str = "auto",
                 steps=None, keep_pictures: bool = True, coverage: np.ndarray | None = None, memo: StepMemo | None = None, measure_steps: bool = True):
        self.memo = memo if steps is None else None             # the export's _Steps log every step as it runs: never replayed
        self.measure_steps = measure_steps
        if self.memo is not None:
            p = _ReadLog(p)
        self.work, self.header, self.p, self.dual, self.external = work, header, p, dual, external
        self.progress, self.checkpoint, self.tick, self.device, self.steps = progress, checkpoint, tick, device, steps
        self.coverage = coverage                            # the stacking coverage map (frames per pixel), or None
        self.denoiser = None                                 # (tool, run(amount, array), ceiling, requested) once denoise() has run its tool; the closing
                                                             # step (run -> _close_sky_noise) re-runs that same tool on the starless the contrast steps left
                                                                   # step must then say the tool was refused for colouring, not that none existed
        self.cwd = os.path.join(work, "siril")
        self.pdir = os.path.join(work, "steps")
        os.makedirs(self.cwd, exist_ok=True)
        if keep_pictures:
            os.makedirs(self.pdir, exist_ok=True)
        self.keep = keep_pictures
        self.records: list[dict] = []
        self.n = 0
        self.bottom_up = export_mod.bottom_up(header)
        self.info: dict = {}

    # ---- plumbing -------------------------------------------------------------------------------------------------------
    def _check(self) -> None:
        if self.checkpoint:
            self.checkpoint()

    def _say(self, msg: str) -> None:
        self._check()
        if self.tick:
            self.tick(msg)

    def siril(self, tag: str, arr: np.ndarray | None, lines: list[str], stage: str, *, out: str | None = None, extra: dict | None = None) -> np.ndarray:
        """Run ``load <tag>_in; <lines>; save <tag>_out`` in Siril (``extra``: more input images {name: array}, written before) and return the output image.  When ``out``
        is given the lines write that file themselves."""
        names = []
        fin = os.path.join(self.cwd, f"{tag}_in.fit")
        if arr is not None:
            fitsio.write_rgb(fin, arr, self.header)
            names.append(fin)
        for n, a in (extra or {}).items():
            path = os.path.join(self.cwd, f"{n}.fit")
            fitsio.write_rgb(path, a, self.header)
            names.append(path)
        outname = out or f"{tag}_out"
        fout = os.path.join(self.cwd, f"{outname}.fit")
        if os.path.exists(fout):
            os.remove(fout)
        script = (f"load {tag}_in\n" if arr is not None else "") + "\n".join(lines) + "\n" + ("" if out else f"save {outname}\n")
        runner = SirilRunner(self.cwd, progress=(lambda i, n, m: self.progress(m) if self.progress and m else None), checkpoint=self.checkpoint)
        try:
            runner.run(script, stage, expect=[f"{outname}.fit"])
            res, hdr = fitsio.read_rgb(fout)
            if export_mod.bottom_up(hdr) != self.bottom_up:
                res = np.ascontiguousarray(res[:, ::-1])
            return np.clip(np.nan_to_num(res), 0.0, 1.0).astype(np.float32)
        finally:
            for f in names + [fout]:
                if os.path.exists(f):
                    os.remove(f)

    def mono(self, name: str, plane: np.ndarray) -> str:
        """Write a single-plane FITS ``<name>.fit`` into the Siril working folder (``rgbcomp -lum=`` needs a MONOCHROME luminance image); returns its path."""
        from astropy.io import fits as _fits
        path = os.path.join(self.cwd, f"{name}.fit")
        hdr = _fits.Header()
        try:
            for k in ("ROWORDER",):
                if k in self.header:
                    hdr[k] = self.header[k]
        except Exception:                                             # noqa: BLE001
            pass
        _fits.PrimaryHDU(np.ascontiguousarray(plane, np.float32), header=hdr).writeto(path, overwrite=True)
        return path

    def picture(self, name: str, arr: np.ndarray, linear: bool = False) -> str | None:
        """``steps/NN_<name>.png``: the picture after a step (linear steps through a linked autostretch)."""
        if not self.keep:
            return None
        img = autostretch_display(arr) if linear else arr
        disp = export_mod.display_image(img, self.bottom_up)
        path = os.path.join(self.pdir, f"{self.n:02d}_{name}.png")
        cv2.imwrite(path, cv2.cvtColor((disp * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
        return path

    def record(self, step: str, tool: str, arr: np.ndarray | None, *, commands=None, params=None, seconds: float = 0.0, linear: bool = False, status: str = "done",
               note: str = "", source: str = "") -> dict:
        self.n += 1
        rec = {"n": self.n, "step": step, "tool": tool, "status": status, "seconds": round(seconds, 1), "commands": list(commands or []), "params": params or {}}
        if note:
            rec["note"] = note
        if source:
            rec["source"] = source
        if arr is not None and status == "done":
            rec["picture"] = self.picture(re.sub(r"[^a-z0-9]+", "_", step.lower()).strip("_"), arr, linear)
            if self.measure_steps:
                rec["levels"] = level_stats(arr)
        self.records.append(rec)
        if self.steps is not None:
            self.steps.add(step, tool, status, seconds, note=note, commands=rec["commands"], **(params or {}))
        return rec

    # ---- 0a. colour gradient (linear) ----------------------------------------------------------------------------------------
    def colour_gradient(self, base: np.ndarray) -> np.ndarray:
        """The channel-to-channel gradient the restore stage's background extraction leaves (the stretch multiplies it by hundreds): Siril ``subsky 2 -samples=20 -tolerance=1.0`` (the command's polynomial degree 2 / default samples and tolerance, pro-workflow.md A2) on each
        channel, with the MEAN of the three channels' models added back so that only the colour part of the gradient is removed and the brightness of the faint nebula is left alone."""
        deg = 2                              # subsky's polynomial degree 2 (pro-workflow.md A2)
        t0 = time.time()
        lines = ["split cg_r cg_g cg_b", "load cg_in", f"subsky {deg} -samples=20 -tolerance=1.0", "split cg_sr cg_sg cg_sb"]
        for ch in "rgb":
            lines += [f'pm "$cg_{ch}$-$cg_s{ch}$+0.5"', f"save cg_m{ch}"]
        for ch in "rgb":
            lines += [f'pm "$cg_s{ch}$+($cg_mr$+$cg_mg$+$cg_mb$)/3-0.5"', f"save cg_o{ch}"]
        lines += ["rgbcomp cg_or cg_og cg_ob -out=cg_out"]
        try:
            out = self.siril("cg", base, lines, "Colour gradient", out="cg_out")
        except SirilError as e:                              # e.g. "removing the gradient on negative images is not supported" (subsky refuses frames it takes for negatives)
            self.record("colour gradient (linear)", "siril subsky + pm", None, status="failed", commands=lines, note=str(e).splitlines()[0][:200])
            return base
        finally:
            for f in [f"cg_{c}" for c in ("r", "g", "b", "sr", "sg", "sb", "mr", "mg", "mb", "or", "og", "ob")]:
                fp = os.path.join(self.cwd, f + ".fit")
                if os.path.exists(fp):
                    os.remove(fp)
        self.record("colour gradient (linear)", "siril subsky + pm", out, commands=lines, seconds=time.time() - t0, linear=True,
                    params={"degree": deg, "samples": 20, "tolerance": 1.0},
                    source="Siril subsky (defaults: samples 20, tolerance 1.0) per channel, mean model added back = colour gradient only; pro-workflow.md A2 (gradient removal before anything that measures the sky)")
        return out

    # ---- 0. background neutralisation (linear) -----------------------------------------------------------------------------
    def neutralise(self, base: np.ndarray) -> np.ndarray:
        """Equal sky level in R, G and B immediately before the stretch: per-channel OFFSETS (no gains) that bring the median of the sky of each channel to the highest one
        (Siril's SPCC ``-bgtol`` background neutralisation / PixInsight BackgroundNeutralization, offsets only).  The stretch multiplies the linear differences by its slope at the sky,
        so the channel-to-channel differences the restoration tools leave after SPCC would otherwise become a colour cast of the sky."""
        lum = finish.luminance(base)
        sm = finish._sky_mask(lum, edge=finish.EDGE_CROP)
        med = [float(np.median(base[c][sm])) for c in range(3)]
        ref = max(med)
        off = [ref - m for m in med]
        t0 = time.time()
        if max(off) < 1e-7:
            self.record("background neutralisation (linear)", "none", None, status="skipped", note="the sky is already neutral", params={"sky_median": med})
            return base
        lines = ["split bn_r bn_g bn_b"]
        for c, ch in enumerate("rgb"):
            lines += [f'pm "$bn_{ch}$+{off[c]:.8f}"', f"save bn_{ch}o"] if off[c] > 0 else [f'pm "$bn_{ch}$"', f"save bn_{ch}o"]
        lines += ["rgbcomp bn_ro bn_go bn_bo -out=bn_out"]
        out = self.siril("bn", base, lines, "Background neutralisation", out="bn_out")
        for f in ("bn_r", "bn_g", "bn_b", "bn_ro", "bn_go", "bn_bo"):
            fp = os.path.join(self.cwd, f + ".fit")
            if os.path.exists(fp):
                os.remove(fp)
        self.record("background neutralisation (linear)", "siril split / pm (offsets) / rgbcomp", out,
                    commands=lines, seconds=time.time() - t0, linear=True,
                    params={"sky_median_before": [round(m, 7) for m in med], "offsets": [round(o, 8) for o in off]},
                    source="Siril SPCC -bgtol / PixInsight BackgroundNeutralization: per-channel offsets so that the sky has equal levels (no gains)")
        return out

    # ---- 0b. banding (second line of defence) --------------------------------------------------------------------------------
    def banding(self, arr: np.ndarray, where: str, linear: bool, noise_from: np.ndarray | None = None) -> np.ndarray:
        """Siril ``fixbanding 1 1`` on the whole image, only when the row-banding ratio (workflow.image_banding_ratio: row-profile std over the white-noise expectation) trips the
        same gate as the sub-level pass.  Siril docs (https://siril.readthedocs.io/en/stable/Commands.html#fixbanding): "Tries to remove the horizontal or vertical banding in the loaded
        image ... Values of 1 and 1 are often good enough" (community practice on stacks too).  Run on the linear starless image before the stretch (which multiplies the stripes by its slope
        at the sky) and on the stretched starless after the denoise (a denoiser keeps row-correlated structure while it removes the white noise, so the stripes can emerge there)."""
        name = f"banding ({where})"
        before = workflow.image_banding_ratio(arr, noise_from)
        ref = "its own pixel noise" if noise_from is None else "the pixel noise before the denoiser"
        if not before["banded"]:
            rec = self.record(name, "none", None, status="skipped", note=f"row banding below the gate (measured against {ref})",
                              params={"ratio_before": before["ratio"], "needed": before["needed"]})
            rec.update(ratio_before=before["ratio"], ratio_after=None, needed=before["needed"], command=None)
            return arr
        t0 = time.time()
        line = f"fixbanding {workflow.BANDING_AMOUNT:g} {workflow.BANDING_SIGMA:g}"
        out = self.siril("bnd", arr, [line], "Banding")
        after = workflow.image_banding_ratio(out, noise_from)
        if not (after["ratio"] is not None and after["ratio"] < before["ratio"]):
            # measured, not assumed: on the stretched IC 1805 starless the tool raised the ratio (2.35 -> 2.70: its sigma-1 highlight protection acts on a non-linear noise level)
            rec = self.record(name, "siril fixbanding", None, status="skipped", commands=[line], seconds=time.time() - t0,
                              note="fixbanding did not lower the measured ratio: result discarded",
                              params={"ratio_before": before["ratio"], "ratio_after": after["ratio"], "needed": before["needed"]})
            rec.update(ratio_before=before["ratio"], ratio_after=after["ratio"], needed=before["needed"], command=line)
            return arr
        rec = self.record(name, "siril fixbanding", out, commands=[line], seconds=time.time() - t0, linear=linear,
                          params={"ratio_before": before["ratio"], "ratio_after": after["ratio"], "needed": before["needed"]},
                          source="Siril docs, fixbanding: amount 1, sigma 1 'often good enough'; gated by the measured row ratio > 1 + 5/sqrt(2 n_rows)")
        rec.update(ratio_before=before["ratio"], ratio_after=after["ratio"], needed=before["needed"], command=line)
        return out

    # ---- 1. stretch -----------------------------------------------------------------------------------------------------
    def subject_coverage(self, base: np.ndarray) -> float:
        lum = finish.luminance(base)
        return float(nebula_mask(lum, _hist_stats(lum)["mode"])[1]["coverage"])

    def stretch(self, base: np.ndarray) -> np.ndarray:
        return self.stretch_prepared(self.prepare_linear(base))

    def prepare_linear(self, base: np.ndarray) -> np.ndarray:
        """The linear steps before the stretch: colour gradient, background neutralisation, banding (none of them reads a stretch setting)."""
        self._say("Stretching")
        cov = self.subject_coverage(base)
        self.info["subject_coverage"] = round(cov, 3)
        if str(self.p.get("background", True)).lower() not in ("false", "0", "", "none"):
            base = self.colour_gradient(base)                   # Siril subsky's own sample rejection (-tolerance) picks the sky samples
        else:
            self.record("colour gradient (linear)", "none", None, status="skipped", note="background = off")
        base = self.neutralise(base)
        return self.banding(base, "linear starless", True)

    def stretch_prepared(self, base: np.ndarray) -> np.ndarray:
        method = str(self.p.get("stretch_method", "ghs2"))
        method = "ghs2" if method == "ghs" else method
        if method == "statstretch":
            return self._stretch_stat(base)
        if method in ALT_METHODS:
            return self._stretch_alt(base, method)
        return self._stretch_ghs2(base)

    def _stretch_ghs2(self, base: np.ndarray) -> np.ndarray:
        p = self.p
        lum = finish.luminance(base)
        h = _hist_stats(lum)
        mask, mi = nebula_mask(lum, h["mode"])
        sel = lum[mask]
        neb_med = float(np.median(sel)) if sel.size else h["mode"]
        neb_hi = float(np.percentile(sel, HP_PERCENTILE)) if sel.size else h["mode"]
        hdr_on = float(p.get("hdr", 0.0)) > 0 and str(p.get("hdr_method", "auto")) != "wavescale"
        b2 = float(np.clip(p.get("contrast", GHS2_B), 0.25, 13.0))
        d2 = float(np.clip(p.get("ghs_d2") or GHS2_D, 0.05, 10.0))
        target = float(np.clip(p.get("stretch", SKY_TARGET), 0.02, 0.4))
        if p.get("auto_stretch", True) and mi["coverage"] >= FRAME_FILLING:
            target = max(target, SKY_BAND[1])                    # a frame-filling object: the background at the top of Brecher's band (0.11), the midtones get the room
        model = ghs_colour_model(float(p.get("color_preservation", 0.7)))
        fit = fit_ghs2(h, neb_med, neb_hi, b1=GHS1_B, b2=b2, d2=d2, target=target, use_hp=hdr_on)
        l1 = [f"ght -D={fit['d1']:.6f} -B={GHS1_B:g} -SP={h['left']:.6f} -LP=0 -HP=1 -clipmode=clip {model}", f"linstretch -BP={fit['bp']:.6f}"]
        t0 = time.time()
        a1 = self.siril("ghs1", base, l1, "Stretch pass 1")
        self.record("stretch pass 1 (GHS)", "siril ght + linstretch", a1, commands=l1, seconds=time.time() - t0,
                    params={"SP": round(h["left"], 6), "b": GHS1_B, "D_siril": round(fit["d1"], 4), "D": round(float(np.expm1(fit["d1"])), 2), "LP": 0, "HP": 1, "BP": round(fit["bp"], 6), "colour_model": model,
                            "histogram": {k: round(v, 7) for k, v in h.items()}, "background_after_pass_1": round(fit["sky1"], 4), "target_background": target, "fit_note": fit.get("note")},
                    source="Siril GHS tutorial + ghsastro: SP left of the histogram peak, b 15 -> 10, D raised until the background reaches the Brecher level (0.10), linstretch -BP at the histogram toe")
        # pass 2 from the measured image
        lum1 = finish.luminance(a1)
        h1 = _hist_stats(lum1)
        sel1 = lum1[mask]
        sp2 = float(np.median(sel1)) if sel1.size else h1["mode"]
        hp2 = float(np.percentile(sel1, HP_PERCENTILE)) if (hdr_on and sel1.size) else 1.0
        lp2 = min(h1["sky"], sp2)
        hp2 = max(hp2, sp2 + 1e-4) if hdr_on else 1.0
        l2 = [f"ght -D={d2:.6f} -B={b2:g} -SP={sp2:.6f} -LP={lp2:.6f} -HP={hp2:.6f} -clipmode=clip {model}"]
        t0 = time.time()
        a2 = self.siril("ghs2", a1, l2, "Stretch pass 2")
        self.record("stretch pass 2 (GHS)", "siril ght", a2, commands=l2, seconds=time.time() - t0,
                    params={"SP": round(sp2, 5), "b": b2, "D_siril": d2, "LP": round(lp2, 5), "HP": round(hp2, 5), "subject_coverage": round(mi["coverage"], 3),
                            "background_before": round(h1["sky"], 4)},
                    source="Siril GHS tutorial + ghsastro: SP on a flat area of the subject (median of the nebula mask), b 3-5, D small, LP raised to hold the background, HP < 1 to protect cores")
        self.info["stretch"] = {"method": "ghs2", "pass1": fit, "histogram": h, "subject": mi}
        return a2

    def _stretch_stat(self, base: np.ndarray) -> np.ndarray:
        p = self.p
        lum = finish.luminance(base)
        cov = finish.object_coverage(lum)
        t = STAT_FILLING if cov >= STAT_FILLING_COVERAGE else float(np.clip(p.get("stretch", STAT_SMALL), 0.01, 0.99))
        line = f"pyscript Statistical_Stretch.py -median {t:.4f} -sigma {STAT_SIGMA:g} -linked"
        if float(p.get("stretch_boost", 0.0)) > 0:
            line += f" -boost {float(p['stretch_boost']):.3f}"
        hdr = float(p.get("hdr", 0.0)) > 0 and str(p.get("hdr_method", "auto")) != "wavescale"
        if hdr:
            line += f" -hdr -hdramount {STAT_HDR_AMOUNT * min(float(p['hdr']), 1.0):.4f} -hdrknee {STAT_HDR_KNEE:g}"
        t0 = time.time()
        out = self.siril("stat", base, [line], "Statistical stretch")
        if float(np.abs(out - base).max()) < 1e-6:
            raise SirilError("Statistical_Stretch.py left the image unchanged (pyscript failed)")
        seconds = time.time() - t0
        # The script's own black clip (median - sigma x its noise estimate) is its black point and runs as published.  An automatic
        # clip-free retry used to ship whenever its sky measured "more even" in percent of the sky - but a stretch that keeps the
        # linear pedestal compresses everything near the sky, so it always measures more even: IC 434 (2026-11-01) shipped it at a
        # quarter of the subject contrast with the sky as even in units of its own noise (0.37 vs 0.36 sigma).
        lump = sky_lump(out)
        self.record("stretch (Statistical Stretch)", "siril pyscript Statistical_Stretch.py", out, commands=[line], seconds=seconds,
                    params={"median": t, "sigma": STAT_SIGMA, "coverage": round(cov, 3), "hdr": hdr,
                            "black_clip": True, "sky_lump": lump},
                    source="Seti Astro Statistical Stretch (Siril port, C. Richard): median -> target; 0.10 for a small target, 0.25 (script default) when the target fills the "
                           "frame (pro-workflow.md A8); the script's black clip at -sigma (its default) is the stretch's black point")
        self.info["stretch"] = {"method": "statstretch", "median": t, "coverage": cov, "black_clip": True, "sky_lump": lump}
        return out

    def _stretch_alt(self, base: np.ndarray, method: str) -> np.ndarray:
        from . import stretch as stretch_mod
        t0 = time.time()
        lum = finish.luminance(base)
        h0 = _hist_stats(lum)
        # black point first (= ``linstretch -BP`` at the histogram toe, what the GHS recipe does after its first pass): the linear stack carries a pedestal
        # (IC 1805 2026-10-16: median 0.05 with the nebula 0.001 above it); a stretch whose symmetry point sits at that median only maps the pedestal to 0.10
        # and leaves the nebula at the sky level (autoghs / modasinh showed stars only)
        bp = float(np.clip(h0["toe"], 0.0, 0.5))
        base = np.clip((base - bp) / max(1.0 - bp, 1e-6), 0.0, 1.0).astype(np.float32)
        lum = finish.luminance(base)
        h = _hist_stats(lum)
        st = stretch_mod.SirilStretcher(os.path.join(self.work, "stretch"), self.steps)
        planes, rec = st.stretch_to_sky([base[0], base[1], base[2]], {"lum": lum, "sky": h["mode"], "noise": max(h["sigma"], 1e-6), "sky_aim": float(self.p.get("stretch", SKY_TARGET))}, method)
        out = np.stack(planes)
        self.record(f"stretch ({method})", rec["tool"], out, commands=[f"linstretch -BP={bp:.6f}"] + list(rec["commands"]), seconds=time.time() - t0,
                    params={"value": rec["value"], "trial_fit": rec["trial_fit"], "target_background": float(self.p.get("stretch", SKY_TARGET)), "black_point": round(bp, 6),
                            "sky_after_black_point": round(float(h["mode"]), 6)},
                    source="linstretch -BP at the histogram toe (the pedestal of the linear stack), then the tool's published command with its one free parameter raised until the background reaches 0.10")
        self.info["stretch"] = {"method": method, "value": rec["value"]}
        return out

    # ---- 2. highlights --------------------------------------------------------------------------------------------------
    def highlights(self, arr: np.ndarray) -> np.ndarray:
        p = self.p
        stats = level_stats(arr)
        hdr = float(p.get("hdr", 0.0))
        method = str(p.get("stretch_method", "ghs2"))
        native = "ghs2" if method == "ghs" else method
        mode = str(p.get("hdr_method", "auto"))
        use_wave = hdr > 0 and (mode == "wavescale" or (mode == "auto" and native in ALT_METHODS))
        self.info["highlights"] = {"clipped": stats["clipped"], "white": stats["white"], "hdr": hdr, "route": "wavescale" if use_wave else (
            {"ghs2": "GHS HP < 1 (pass 2)", "statstretch": "Statistical Stretch -hdr"}.get(native, "none") if hdr > 0 else "off")}
        if not use_wave:
            self.record("highlights", self.info["highlights"]["route"] if hdr > 0 else "none (hdr off)", None, params=self.info["highlights"],
                        status="done" if hdr > 0 else "skipped", note="the highlight protection is part of the stretch step" if hdr > 0 else "hdr=0")
            self.records[-1]["levels"] = stats
            return arr
        self._say("Highlights")
        n_s, comp, gam = WAVESCALE
        comp = comp * min(hdr, 1.5)
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "community_worker.py")
        py, script = _community_python(), _community_script("processing/HDR_multiscale.py")
        if not py or not script:
            self.record("highlights (Wavescale HDR)", "none", None, status="unavailable", note="the pyscript virtualenv or HDR_multiscale.py is missing")
            return arr
        fin = os.path.join(self.cwd, "hdr_in.npy")
        prefix = os.path.join(self.cwd, "hdr_out")
        np.save(fin, np.ascontiguousarray(np.moveaxis(arr, 0, -1)))
        args = json.dumps([n_s, comp, gam])
        cmd = [py, worker, "--script", script, "--entry", "compute_wavescale_hdr", "--input", fin, "--output-prefix", prefix, "--args", args, "--kwargs", json.dumps([{"dim_gamma": 1.0}])]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, env=config.siril_env({"QT_QPA_PLATFORM": "offscreen"}), timeout=1800)
        try:
            if r.returncode != 0:
                raise SirilError("Wavescale HDR worker failed: " + (r.stderr or r.stdout).strip().splitlines()[-1][:300])
            out = np.moveaxis(np.load(f"{prefix}_0.npy"), -1, 0)
        finally:
            for f in (fin, f"{prefix}_0.npy"):
                if os.path.exists(f):
                    os.remove(f)
        out = np.clip(np.nan_to_num(out), 0, 1).astype(np.float32)
        self.record("highlights (Wavescale HDR)", "community_worker compute_wavescale_hdr (Seti Astro, HDR_multiscale.py)", out, commands=[f"community_worker.py --script processing/HDR_multiscale.py "
                    f"--entry compute_wavescale_hdr --args '{args}' --kwargs '[{{\"dim_gamma\": 1.0}}]'"], seconds=time.time() - t0,
                    params={"scales": n_s, "compression": comp, "mask_gamma": gam, "clipped_before": stats["clipped"]},
                    source="Seti Astro Wavescale HDR defaults 5 scales / compression 1.5 / mask gamma 5.0 (pro-workflow.md A9); the script's final dimming curve (gamma 1 + 0.2 x scales = 2) is switched off "
                           "(dim_gamma 1): it sends the background of 0.10 to 0.01, and the background is fixed by the stretch")
        return out

    # ---- 3. noise reduction of the stretched starless ----------------------------------------------------------------------
    def _sky_noise_of(self, pic: np.ndarray, starless: np.ndarray) -> float:
        """The delivered picture's ``sky_pixel_noise``, computed the way the export computes it: the luminance of the
        picture being handed out, the sky mask on that luminance, and the **starless** as the plane the sky is read from
        (``gates.sky_luminance``).  The picture alone is a different and much lower number - the first version of this
        step read 0.00387 on the denoiser's output while the export reported 0.01075, and so never fired.
        """
        lum = finish.luminance(pic)
        return gates.pixel_noise(lum, gates.sky_mask(lum.astype(np.float32)), self.delivered_starless(starless))

    def delivered_starless(self, starless: np.ndarray | None) -> np.ndarray | None:
        """``starless`` through the last step's levels (``final_levels``: the black point, then the midtones), so that the star gates, which
        measure a star's light against the starless it was screened onto, compare the two in the same levels.  Before the black point and the
        sky's midtones existed the two were the same; with them, the undelivered starless made every star read colourless."""
        lv = getattr(self, "final_levels", None)
        if starless is None or lv is None:
            return starless
        bp, mid = lv
        x = np.clip((np.asarray(starless, np.float32) - bp) / max(1.0 - bp, 1e-9), 0.0, 1.0)
        return x if abs(mid - 0.5) <= 1e-4 else mtf_balance(mid, x).astype(np.float32)

    def _failed_of(self, pic: np.ndarray, starless: np.ndarray) -> list:
        """Which gates the delivered picture fails, for the comparison that decides whether a harder pass is allowed.
        The export hands ``gates.compute`` the linear-side measurements as well; here incumbent and candidate are judged
        by the same call, which is what makes the comparison fair rather than absolute.
        """
        return sorted(k for k, v in gates.compute(pic, starless=self.delivered_starless(starless)).items() if isinstance(v, dict) and not v["pass"])

    def _close_sky_noise(self, starless: np.ndarray, compose):
        """Denoise the sky the delivered picture actually carries, not the one the denoiser left behind.

        The chain denoises the stretched starless and then raises its local contrast, sharpens and saturates it, and the
        composite finally goes through a black point and a midtone curve - every one of which multiplies the sky's
        scatter.  Measured on C 20 (2026-10-31): the denoiser returned a sky at 0.00387, the export reported 0.01075
        against the 0.008 gate, and the same tool at its own ceiling delivered 0.00487 with all twelve gates passing.
        Asking the tool once and never looking again is how a frame denoised too *weakly* was indistinguishable, in the
        record, from a frame denoised at all.

        So this runs after the composition, on the array the export will measure, and re-runs the tool the chain already
        chose on the starless that produced it - recomposing for every candidate, so the number compared is the number
        delivered.  Three rules keep it from becoming a knob with one dataset's name on it:

        * the statistic is the gate's own (``gates.pixel_noise``), the limit is ``gates.LIMITS`` and the ceiling is the
          tool's published maximum - nothing here is tuned to a frame;
        * a candidate is kept only when it lowers the delivered noise **and** fails no gate the incumbent had not
          already failed, both judged by the same ``gates.compute`` call, so chasing one number cannot buy a failure in
          another;
        * at most ``DENOISE_ESCALATION`` extra tool runs, the ladder reaching the ceiling, every attempt recorded, and a
          refused candidate's chain records discarded so the log describes the picture that shipped.

        ``compose(starless)`` renders the delivered picture from a starless.
        """
        t0 = time.time()
        out = compose(starless)
        limit = float(gates.LIMITS["sky_pixel_noise"])
        if not self.external and not self.measure_steps:
            self.record("noise reduction (sky gate)", "none", out, status="skipped", note="preview: no AI steps")
            return starless, out
        noise = self._sky_noise_of(out, starless)
        if not self.external or self.denoiser is None:
            why = "preview: no AI steps" if not self.external else "nothing was available to ask"
            self.record("noise reduction (sky gate)", "none", out, status="skipped", seconds=time.time() - t0,
                        params={"sky_pixel_noise": round(noise, 5), "gate": limit},
                        note=f"the sky the delivered picture carries measured {noise:.5f} against its {limit} gate, and no denoiser ran "
                             f"({why})")
            return starless, out
        tool, run, ceiling, requested = self.denoiser
        attempts = [{"amount": round(float(requested), 3), "sky_pixel_noise": round(noise, 5)}]
        used = used_cmd = None
        if noise > limit:
            base_failed = self._failed_of(out, starless)
            for step in np.unique(np.round(np.linspace(requested, ceiling, DENOISE_ESCALATION), 6)):
                a = float(step)
                self._say(f"Noise reduction (the delivered sky measured {noise:.5f} against its {limit} gate: {tool} again at {a:g})")
                saved, self.records, accepted = self.records, [], False
                try:
                    cand, cmd = run(a, starless)                    # the same tool, on the starless as the chain has it now
                    cand_out = compose(cand)
                    cn = self._sky_noise_of(cand_out, cand)
                    failed = self._failed_of(cand_out, cand)
                    accepted = cn < noise and set(failed) <= set(base_failed)
                    attempts.append({"amount": round(a, 3), "sky_pixel_noise": round(cn, 5), "failed": failed, "kept": accepted})
                    if accepted:
                        starless, out, noise, used, used_cmd, base_failed = cand, cand_out, cn, a, cmd, failed
                        saved.extend(self.records)                  # only the pass that shipped enters the chain log
                except (ToolUnavailable, ToolError) as e:            # the tool refuses to go further: keep what is good
                    attempts.append({"amount": round(a, 3), "error": str(e)[:120]})
                finally:
                    self.records = saved
                if "error" in attempts[-1] or noise <= limit:
                    break
        self._sky_noise_used = used                                          # the closing clipping step re-renders with the same escalation
        info = {"sky_pixel_noise": round(noise, 5), "gate": limit, "requested": round(float(requested), 3),
                "used": used, "attempts": attempts}
        if used is not None:
            info["escalated"] = True
        self.record("noise reduction (sky gate)", tool, out, commands=used_cmd, seconds=time.time() - t0, params=info,
                    note=f"the sky the delivered picture carries measured {noise:.5f} against its {limit} gate"
                         + (f"; {tool} was re-run on the starless and the picture recomposed, kept at {used:g}" if used is not None
                            else "; the picture was already inside its gate"))
        return starless, out

    def _keep_denoise(self, run, arr: np.ndarray, amount: float, *, tool: str, ceiling: float, params, start: float,
                      source: str, note: str = "", amounts: tuple | None = None) -> np.ndarray:
        """Run the stretched-image denoiser at the strength asked and keep its result; the sky's colour before and after (``_sky_chroma``: the gates'
        median |a*| / |b*| and the quadrants' spread) is recorded.  ``self.denoiser`` is set so the closing step (``_close_sky_noise``) can run the same
        tool harder while the delivered sky fails the sky_pixel_noise gate.  ``params`` is called with the amount, after the tool ran."""
        a = round(float(amount), 3)
        if a <= 0:
            return arr
        before = self._sky_chroma(arr)
        out, cmd = run(a, arr)
        after = self._sky_chroma(out)
        self.denoiser = (tool, run, ceiling, a)
        self.record("noise reduction (stretched)", tool, out, commands=[cmd] if isinstance(cmd, str) else list(cmd), seconds=time.time() - start,
                    params={**params(a), "sky_chroma_before": before, "sky_chroma_after": after}, note=note, source=source)
        return out

    def denoise(self, arr: np.ndarray) -> np.ndarray:
        p = self.p
        if not self.external:
            self.record("noise reduction (stretched)", "none", None, status="skipped", note="preview: no AI steps (the export runs NoiseXTerminator here)")
            return arr
        self._say("Noise reduction")
        dn = float(np.clip(p.get("luminance_denoise", 0.7), NXT_DN[0] - 0.1, 1.0))
        dc = float(np.clip(p.get("chroma_denoise", NXT_DC), 0.0, 1.0))
        t0 = time.time()
        if self.p.get("cc_denoise"):
            # Cosmic Clarity Denoise (``cc_denoise``) instead of NoiseXTerminator on the stretched starless - its models' own domain (it denoises linear data through a
            # temporary stretch, which on this pipeline's linear data came back with 2.2 x the flux round the stars; here it keeps it, 1.00).  Luminance strength =
            # ``luminance_denoise``, colour strength = ``chroma_denoise`` (the two amounts NXT takes as --di / --dc); model = ``cc_denoise_model``, the script's
            # Full / Lite / Walking Noise variants (--lite / --walking).  Chosen and unable to run: recorded, no other denoiser in its place.
            model = str(self.p.get("cc_denoise_model") or "full").lower()
            extra = {"lite": ["--lite"], "walking": ["--walking"]}.get(model, [])
            meta: dict = {}

            def run(a: float, a_arr: np.ndarray):
                cli = ["--denoise-mode", "full", "--denoise-strength", f"{a:g}", "--denoise-color", f"{dc:g}"] + extra
                o, m = tools.cosmic_clarity.run("denoise", a_arr, cli, work=self.cwd, device=self.device, checkpoint=self.checkpoint,
                                                progress=(lambda i, n, msg: self.progress(msg) if self.progress else None))
                meta["backend"] = m.get("backend")
                return np.clip(o, 0, 1), m["command"]

            try:
                return self._keep_denoise(run, arr, dn, tool="cosmic clarity denoise", ceiling=1.0, start=t0,
                                          amounts=(dn, max(dn * 0.5, 0.1)),
                                          params=lambda a: {"denoise_strength": a, "denoise_color": dc, "model": model,
                                                            "backend": meta.get("backend"), "device": self.device},
                                          source=f"SetiAstro Cosmic Clarity AI4 denoise, {model} model (siril-scripts CosmicClarity_Native.py, full mode: mono model on the luminance, colour model on the chroma).  "
                                                 "The sky it leaves is measured again after the contrast steps, in noise reduction (sky gate)")
            except Cancelled:
                raise
            except (ToolUnavailable, ToolError) as e:
                self.record("noise reduction (stretched)", "cosmic clarity denoise", None, status="unavailable", note=str(e)[:200])
                return arr
        src, dst = os.path.join(self.cwd, "nxt_in.fits"), os.path.join(self.cwd, "nxt_out.fits")

        def tidy():                                               # each pass writes its own pair and leaves no residue behind
            for f in (src, dst):
                if os.path.exists(f):
                    os.remove(f)

        prog = lambda i, n, msg: self.progress(msg) if self.progress else None
        try:
            tools.rcastro.require("nxt")

            def run(a: float, a_arr: np.ndarray):
                tidy()
                fitsio.write_rgb(src, a_arr, self.header)
                try:
                    m = tools.rcastro.run("nxt", src, dst, device=self.device, progress=prog, checkpoint=self.checkpoint,
                                          csep=True, di=a, dc=dc)
                    return np.clip(fitsio.read_rgb(dst)[0], 0, 1), ["rc-astro nxt " + " ".join(m.get("flags") or [])]
                finally:
                    tidy()

            return self._keep_denoise(run, arr, dn, tool="rc-astro nxt", ceiling=1.0, start=t0,
                                      amounts=(dn, max(dn * 0.5, NXT_DN[0])),
                                      params=lambda a: {"di": a, "dc": dc},
                                      source="RC-Astro / Brecher: NXT on the stretched starless, amount 0.6-0.85; colour separation --dc 0.9-1.0.  The "
                                             "sky it leaves is measured again after the contrast steps, in noise reduction (sky gate)")
        except Cancelled:
            raise
        except (ToolUnavailable, ToolError) as e:
            self.record("noise reduction (stretched)", "rc-astro nxt", None, status="unavailable", note=str(e)[:200])
        try:
            strength = float(np.clip(dn, 0.6, 0.8))                  # GraXpert's own documented range for denoising
            gx: dict = {}

            def run(a: float, a_arr: np.ndarray):
                tidy()
                fitsio.write_rgb(src, a_arr, self.header)
                try:
                    m = tools.graxpert.denoise(src, dst, strength=a, gpu=self.device, progress=prog, checkpoint=self.checkpoint)
                    gx["model"] = m.get("model")
                    return np.clip(fitsio.read_rgb(dst)[0], 0, 1), [f"graxpert denoising strength={a}"]
                finally:
                    tidy()

            out = self._keep_denoise(run, arr, strength, tool="graxpert denoise", ceiling=0.8, start=t0,
                                     amounts=(strength, 0.6),
                                     params=lambda a: {"strength": a, "model": gx.get("model")},
                                     note="fallback: NoiseXTerminator unavailable",
                                     source="pro-workflow.md A10: GraXpert 0.6-0.8.  The sky it leaves is measured again after the contrast steps, "
                                            "in noise reduction (sky gate)")
            return self.chroma_denoise(out, why="the denoiser that ran takes no colour amount: GraXpert denoises RGB jointly, "
                                                 "and the ``chroma_denoise`` setting only reaches NXT (--dc) / Cosmic Clarity "
                                                 "(--denoise-color), both of which were unavailable")
        except Cancelled:
            raise
        except (ToolUnavailable, ToolError) as e:
            self.record("noise reduction (stretched)", "none", None, status="unavailable", note=str(e)[:200])
            return self.chroma_denoise(arr, why="no denoiser ran at all, so the colour was certainly left untouched")
        finally:
            tidy()

    def chroma_denoise(self, arr: np.ndarray, why: str) -> np.ndarray:
        """Smooth the colour when the denoiser that actually ran could not be handed the colour amount.

        ``chroma_denoise`` is the user's colour-noise control (post.py PARAM_SPEC: "Colour noise reduction"), but the amount
        is only ever passed to NoiseXTerminator (``--dc``) and to Cosmic Clarity (``--denoise-color``).  With both unavailable
        - which is every frame on a machine without the RC-Astro licence, and every frame the user routes to GraXpert -
        colour: C 20 (2026-10-31) delivered a sky whose red channel carried 1.64x the scatter of the other two (sky std
        0.0571 against 0.0349 and 0.0347), with 35% red and 24% cyan speckled pixels and both sky-colour gates failed, while
        the run recorded chroma_denoise 0.95 as though it had been applied.

        The split is luminance-preserving by construction: with l = sum(w_k c_k) and sum w_k = 1, replacing c_k by
        l + blur(c_k - l) gives sum(w_k c'_k) = l + blur(sum w_k (c_k - l)) = l + blur(0) = l.  The stretch, the star detection
        and the sky gates therefore see exactly the luminance they saw before, and only the colour moves - which is the whole
        point, since the grain this removes is invisible in luminance and unmistakable in colour.

        The sky's own colour is preserved as exactly as its lightness is.  Smoothing regresses each channel toward its
        neighbourhood mean, and because the sky mask keeps the *darkest* pixels that regression is one-sided: measured
        on C 20 (2026-10-31) the first version of this step halved the speckle yet walked the sky's median a* from 5.42
        to 6.73 and its b* from -2.17 to -3.44, moving both colour gates the wrong way - and they judge those medians
        (``gates.sky_chroma`` is the sky's median |a*| and |b*|), not the grain.  The drift is therefore undone below,
        less its lightness component, so neither the lightness nor the sky's cast moves and only the grain goes.
        """
        name = "colour noise reduction"
        amount = float(np.clip(self.p.get("chroma_denoise", NXT_DC), 0.0, 1.0))
        if amount <= 0:
            self.record(name, "none", None, status="skipped", note="chroma_denoise=0")
            return arr

        def measure(a: np.ndarray, m) -> dict | None:
            """The quantities that decide this step's verdict, all over one shared sky region: the per-channel scatter
            of the colour (channel minus luminance), which is where the red/cyan speckle lives, and what ``gates``
            actually judges - ``sky_chroma`` is the sky's median |a*| and |b*| (a colour cast, not a noise measure)
            and ``sky_chroma_quadrants`` the spread of the four quadrants' medians.  Recording only the first let this
            step report a halved speckle while both gates moved the wrong way."""
            if m is None:
                return None
            lab = cv2.cvtColor(np.ascontiguousarray(np.moveaxis(np.clip(a, 0, 1), 0, -1), np.float32),
                               cv2.COLOR_RGB2Lab)
            l = finish.luminance(a)
            am, bm = lab[..., 1][m], lab[..., 2][m]
            return {"spread": round(float(np.median([float(np.std(c[m])) for c in a - l])), 5),
                    "a": round(float(finish.mad_sigma(am)), 4), "b": round(float(finish.mad_sigma(bm)), 4),
                    "a_median": round(float(np.median(am)), 4), "b_median": round(float(np.median(bm)), 4)}

        sigma = CHROMA_SIGMA_MAX * amount
        t0 = time.time()
        lum = finish.luminance(arr)
        try:
            mask = finish._sky_mask(lum.astype(np.float32))
            mask = mask if int(np.count_nonzero(mask)) >= 1000 else None
        except Exception:                                             # noqa: BLE001
            mask = None
        sm = np.stack([cv2.GaussianBlur((c - lum).astype(np.float32), (0, 0), sigma) for c in arr])
        out = _chroma_ray(lum, sm).astype(np.float32)
        before = measure(arr, mask)
        recentered = None
        if mask is not None:
            # Smoothing regresses every channel toward its neighbourhood mean, and because the sky mask keeps the
            # *darkest* pixels that regression is one-sided: measured on C 20 (2026-10-31) the sky's a* median walked
            # 5.42 -> 6.73 and its b* -2.17 -> -3.44 while the speckle halved, which pushed both colour gates the
            # wrong way, because they judge those medians.  Undo the drift, less its lightness component: with
            # sum(w_k d_k) = 0 the correction cannot move the lightness either, so the sky the gates measure is the
            # sky that arrived and only the grain has gone.
            w = np.asarray(finish.LUM709, dtype=np.float64)
            d = np.array([np.median(arr[k][mask] - lum[mask]) - np.median(out[k][mask] - lum[mask]) for k in range(3)])
            d = d - float(d @ w / (w @ w)) * w
            out = _chroma_ray(lum, out - lum + d[:, None, None]).astype(np.float32)
            recentered = [round(float(v), 5) for v in d]
        self.record(name, "gaussian on the channels' difference from the Rec.709 luminance", out,
                    commands=[f"per channel: c = lum + gauss(c - lum, sigma={sigma:.2f} px)"
                              + (f"; sky re-centred by {recentered}" if recentered else "")],
                    seconds=time.time() - t0,
                    params={"chroma_denoise": round(amount, 3), "sigma_px": round(sigma, 3),
                            "sky_before": before, "sky_after": measure(out, mask), "sky_recentered": recentered},
                    note=why,
                    source="the fallback for NXT --dc / Cosmic Clarity --denoise-color, whose kernel scale is not "
                           "published; the split leaves sum(w_k c'_k) = l and the re-centring leaves sum(w_k d_k) = 0, "
                           "so the luminance the gates measure is untouched")
        return out

    # ---- 3b. Cosmic Clarity Sharpen on the stretched image (cc_sharpen) -----------------------------------------------------------------------
    def cc_sharpen(self, arr: np.ndarray) -> np.ndarray:
        """Cosmic Clarity Sharpen (siril-scripts CosmicClarity_Native.py) on the stretched, recombined image (``cc_sharpen`` both / stellar / nonstellar = the script's
        Both / Stellar Only / Non-Stellar Only, amounts ``cc_stellar_amount`` / ``cc_nonstellar_amount``, PSF measured per tile by its auto-PSF).  On the stretched image
        the model keeps the flux (0.96 on Sh2-131); on this pipeline's linear data the script's own stretch / unstretch round trip came back with 6.2 x the flux
        (bloated stars), so it is not offered there.  Independent of the linear deconvolution (``deconv_method``): with BlurXTerminator / GraXpert it runs in
        addition, with deconv_method none instead.  Skipped in the preview (an AI step, like the noise reduction)."""
        mode = {"both": "Both", "stellar": "Stellar Only", "nonstellar": "Non-Stellar Only"}.get(str(self.p.get("cc_sharpen") or "off").lower())
        if mode is None:
            return arr
        if not self.external:
            self.record("sharpening (Cosmic Clarity)", "none", None, status="skipped", note="preview: no AI steps")
            return arr
        self._say("Sharpening")
        st_amt = round(float(np.clip(self.p.get("cc_stellar_amount", CC_SHARPEN_AMOUNT), 0.0, 1.0)), 3)
        ns_amt = round(float(np.clip(self.p.get("cc_nonstellar_amount", CC_SHARPEN_AMOUNT), 0.0, 1.0)), 3)
        cli = ["--sharpening-mode", mode, "--stellar-amount", f"{st_amt:g}", "--nonstellar-amount", f"{ns_amt:g}"]
        t0 = time.time()
        try:
            out, m = tools.cosmic_clarity.run("sharpen", arr, cli, work=self.cwd, device=self.device, checkpoint=self.checkpoint,
                                              progress=(lambda i, n, msg: self.progress(msg) if self.progress else None))
            out = np.clip(out, 0, 1)
            self.record("sharpening (Cosmic Clarity)", "cosmic clarity sharpen", out, commands=[m["command"]], seconds=time.time() - t0,
                        params={"mode": mode, "stellar_amount": st_amt, "nonstellar_amount": ns_amt, "auto_psf": True, "backend": m.get("backend"), "device": m.get("device")},
                        source="SetiAstro Cosmic Clarity AI4 sharpen (siril-scripts CosmicClarity_Native.py; its default amounts are 0.5, auto PSF)")
            return out
        except Cancelled:
            raise
        except (ToolUnavailable, ToolError) as e:
            self.record("sharpening (Cosmic Clarity)", "cosmic clarity sharpen", None, status="unavailable", note=str(e)[:200])
            return arr

    # ---- 4. colour: palette / SCNR -----------------------------------------------------------------------------------------
    def palette_name(self) -> str:
        pal = str(self.p.get("palette", "auto"))
        if pal in ("", "auto"):
            pal = ("hoo_improved" if self.p.get("oiii_unmix", True) else "hoo") if self.dual else "natural"
        return pal

    def hubble_preset(self, pal: str) -> str:
        """The Hubble palette script's formula preset: ``hubble_formula`` (classic / improved / advanced / nonlinear_s2); auto = Improved for ``hoo_improved`` or with
        ``oiii_unmix`` (OIII cleaned of the Ha leak), else Classic (the script's default)."""
        f = str(self.p.get("hubble_formula", "auto") or "auto").lower()
        if f in HUBBLE_PRESETS:
            return HUBBLE_PRESETS[f]
        return "Improved" if (pal == "hoo_improved" or self.p.get("oiii_unmix", False)) else "Classic"

    def _sky_chroma(self, arr: np.ndarray) -> dict | None:
        """The sky's colour as the gates measure it, so a step is judged by the numbers the export is judged by and
        cannot satisfy its own check while failing the picture's.

        Two quantities, because the two colour gates ask different questions.  ``a``/``b`` are ``median |a*|`` and
        ``median |b*|`` over ``gates.sky_mask`` -- what ``sky_chroma`` gates -- and those are dominated by chroma
        *noise*: a perfectly grey sky that carries noise scores high on them and a denoiser lowers them, so they
        cannot see a tint at all.  ``spread`` is the spread of the four quadrants' **signed** median tints, which is
        what ``sky_chroma_quadrants`` gates.  A step that smooths chroma noise while tinting the sky improves the
        first and fails the second -- C 20 (2026-11-01) was delivered at |a*| 1.73 of 2.0 with the quadrants 1.97
        apart of 1.5 -- so a step that judges colour is asked both.
        """
        a = np.clip(np.nan_to_num(np.asarray(arr, np.float32)), 0.0, 1.0)
        if a.ndim != 3 or a.shape[0] not in (3, 4):
            return None
        sm = gates.sky_mask(finish.luminance(a))
        if sm.sum() <= 100:
            return None
        lab = gates._lab(a)
        qs = gates.quadrant_spread(lab, sm)
        return {"a": round(float(np.median(np.abs(lab[..., 1][sm]))), 2),
                "b": round(float(np.median(np.abs(lab[..., 2][sm]))), 2),
                "net_a": round(float(np.median(lab[..., 1][sm])), 2),
                "net_b": round(float(np.median(lab[..., 2][sm])), 2),
                "spread": round(max(qs["spread_a"], qs["spread_b"]), 2)}

    def _scnr(self, arr: np.ndarray, scnr: float, *, source: str) -> np.ndarray:
        """``rmgreen 3 <amount>`` (Siril's SCNR) under the background-protecting range mask (``range_mask``, as the saturation step): the amount is
        decided on the subject, and Siril's rmgreen acts on the whole frame and preserves lightness, so on the sky it would put the green it lifts
        out back through red and blue (a magenta sky).  The sky's colour before and after is recorded."""
        lum = finish.luminance(arr)
        h = _hist_stats(lum)
        mask, mi = nebula_mask(lum, h["mode"])
        w = self.range_mask(lum, mask, mi)
        before = self._sky_chroma(arr)
        lines = ["save sc_a", f"rmgreen 3 {scnr:.3f}", "save sc_b", 'pm "$sc_a$+$sc_m$*($sc_b$-$sc_a$)"']
        t0 = time.time()
        out = self.siril("scnr", arr, lines, "SCNR", extra={"sc_m": np.stack([w, w, w])})
        for f in ("sc_a", "sc_b"):
            fp = os.path.join(self.cwd, f + ".fit")
            if os.path.exists(fp):
                os.remove(fp)
        self.record("SCNR", "siril rmgreen + pm (range mask)", out, commands=lines, seconds=time.time() - t0,
                    params={"type": 3, "amount": scnr, "sky_chroma_before": before, "sky_chroma_after": self._sky_chroma(out)}, source=source)
        return out

    def colour(self, arr: np.ndarray) -> np.ndarray:
        self._say("Colour")
        p = self.p
        pal = self.palette_name()
        boost = float(p.get("oiii_boost", 1.0))
        if self.dual and pal != "natural":
            ha = arr[0].copy()
            # the palette chosen decides; the OIII boost is NBN's own setting.  It used to switch NBN on by itself (boost != 1), so with a boost set
            # every other palette of the menu rendered as NBN - the preview "stuck" in NBN whatever palette was picked
            if pal == "nbn":
                out = self._palette_nbn(arr, boost)
            elif pal in HUBBLE_PALETTES:
                out = self._palette_hubble(arr, pal)
            else:
                out = self._palette_pm(arr, pal)
            scnr = float(np.clip(p.get("scnr", 0.0), 0.0, 1.0))
            if pal in HA_GREEN_PALETTES and scnr > 0:
                out = self._scnr(out, scnr, source="Siril rmgreen on an Ha-in-green palette (NarrowbandNormalization.py applies SCNR to SHO-type palettes)")
            return self.ha_luminance(out, ha)
        scnr = float(np.clip(p.get("scnr", 0.0), 0.0, 1.0))
        if not self.dual and scnr > 0:
            return self._scnr(arr, scnr, source="Siril rmgreen (PixInsight SCNR), additive mask; broadband only (pro-workflow.md A13)")
        self.record("palette / SCNR", "none", None, status="skipped", note="dual-band natural colour: no SCNR (green is OIII)" if self.dual else "scnr=0")
        return arr

    def ha_luminance(self, arr: np.ndarray, ha: np.ndarray) -> np.ndarray:
        """``synthetic_luminance`` s (dual-band): the lightness of the palette image becomes (1 - s) x its own + s x the stretched Ha (R) plane, composed with Siril
        ``rgbcomp -lum=``: Ha carries most of the signal of a dual-band OSC stack, so the OIII channel's noise cannot reach the lightness (pro-workflow.md B "Lightness: Ha";
        NarrowbandNormalization.py's Lightness option; Cloudy Nights 'use Ha as luminance')."""
        s_ = float(np.clip(self.p.get("synthetic_luminance", 0.5), 0.0, 1.0))
        if not self.dual or s_ <= 1e-3:
            return arr
        t0 = time.time()
        lum = finish.luminance(arr)
        newl = np.clip((1.0 - s_) * lum + s_ * ha, 0.0, 1.0).astype(np.float32)
        lines = ["rgbcomp -lum=hl_l hl_in -out=hl_out"]
        lp = self.mono("hl_l", newl)
        try:
            out = self.siril("hl", arr, lines, "Ha luminance", out="hl_out")
        finally:
            if os.path.exists(lp):
                os.remove(lp)
        self.record("Ha luminance", "siril rgbcomp -lum", out, commands=lines, seconds=time.time() - t0, params={"synthetic_luminance": s_},
                    source="rgbcomp -lum (Siril LRGB composition): lightness = (1 - s) image + s Ha; pro-workflow.md Chain B (NBN Lightness: Ha)")
        return out

    def _palette_pm(self, arr: np.ndarray, pal: str) -> np.ndarray:
        t0 = time.time()
        o_classic = "$pr_g$*0.5+$pr_b$*0.5"
        if pal == "hoo_improved":
            o_expr, src = "($pr_g$+$pr_b$-0.3*$pr_r$)*0.5", "Hubble_Palette_from_Dual-Band_OSC.py Improved: OIII = (G + B - 0.3 R) / 2"
        else:
            o_expr, src = o_classic, "Hubble_Palette_from_Dual-Band_OSC.py Classic: Ha = R, OIII = (G + B) / 2"
        lines = ["split pr_r pr_g pr_b", f'pm "{o_expr}"', "save pr_o"]
        if pal == "foraxx":
            ho = "(($pr_r$*$pr_o$)^(~($pr_r$*$pr_o$)))"
            lines += [f'pm "{ho}*$pr_r$+(~{ho})*$pr_o$"', "save pr_gg", "rgbcomp pr_r pr_gg pr_o -out=pr_out"]
            src = "Foraxx dynamic palette (The Coldest Nights, theatrus/foraxx-palette-utility): G = ho*Ha + ~ho*OIII, ho = (Ha*OIII)^~(Ha*OIII), R = Ha, B = OIII (no SII)"
        elif pal == "hoo_warm":
            lines += ['pm "0.9*$pr_o$+0.1*$pr_r$"', "save pr_gg", "rgbcomp pr_r pr_gg pr_o -out=pr_out"]
            src = "Siril PixelMath variant (origin-astro.wheep.co.uk): R = Ha, G = 0.9 OIII + 0.1 Ha, B = OIII"
        else:
            lines += ["rgbcomp pr_r pr_o pr_o -out=pr_out"]
        out = self.siril("pr", arr, lines, "Palette", out="pr_out")
        for f in ("pr_r", "pr_g", "pr_b", "pr_o", "pr_gg"):
            fp = os.path.join(self.cwd, f + ".fit")
            if os.path.exists(fp):
                os.remove(fp)
        self.record(f"palette ({pal})", "siril split / pm / rgbcomp", out, commands=lines, seconds=time.time() - t0, params={"palette": pal}, source=src)
        return out

    def _palette_hubble(self, arr: np.ndarray, pal: str) -> np.ndarray:
        """The dual-band palette by Hubble_Palette_from_Dual-Band_OSC.py itself (siril-scripts, Carlo Mollicone - AstroBOH), run headless in Siril by
        ``hubble_palette_worker.py``: the starless image is loaded, the script splits it, derives Ha / OIII / synthetic SII with its formula preset (``pm``) and composes
        the palette with ``rgbcomp``; its result file is the output.  When the script cannot run (no pyscript environment, Qt missing, ...) the step is recorded as
        failed and the image is left as it is."""
        t0 = time.time()
        letters, preset = HUBBLE_PALETTES[pal], self.hubble_preset(pal)
        forms = HUBBLE_FORMULAS[preset]
        script = _community_script(HUBBLE_SCRIPT)
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hubble_palette_worker.py")
        result = f"hp_in_result_{letters}_{preset}"
        line = f"pyscript {worker} --script {script} --palette {letters} --preset {preset.replace(' ', '_')}"
        params = {"palette": letters, "preset": preset, "Ha": forms["HA"], "OIII": forms["OIII"], "SII (synthetic)": forms["S2"],
                  "rgb": " / ".join(HUBBLE_MAPPING[letters])}
        src = (f"Hubble_Palette_from_Dual-Band_OSC.py v2.0.4 (Carlo Mollicone - AstroBOH, siril-scripts): {letters} = R {HUBBLE_MAPPING[letters][0]}, "
               f"G {HUBBLE_MAPPING[letters][1]}, B {HUBBLE_MAPPING[letters][2]}; preset {preset}: Ha = {forms['HA']}, OIII = {forms['OIII']}, SII = {forms['S2']}")
        try:
            if not script:
                raise SirilError(f"siril-scripts' {HUBBLE_SCRIPT} is missing")
            out = self.siril("hp", arr, [line], "Hubble palette", out=result)
            self.record(f"palette ({pal})", "siril pyscript Hubble_Palette_from_Dual-Band_OSC.py", out, commands=[line], seconds=time.time() - t0,
                        params=params, source=src)
            return out
        except (SirilError, OSError) as e:
            # the palette chosen is the script's: when it cannot run, that is recorded and the image keeps its colours (nothing else runs in its place)
            self.record(f"palette ({pal})", "siril pyscript Hubble_Palette_from_Dual-Band_OSC.py", None, status="failed", commands=[line], params=params,
                        note=str(e).splitlines()[0][:200], source=src)
            return arr
        finally:
            import glob as _glob
            for pat in ("Ha.fit", "Oiii.fit", "S2_synt.fit", "Temporary_Image_*.fit", "hp_in_result_*.fit", "hp_?.fit", "hpf_*.fit"):
                for f in _glob.glob(os.path.join(self.cwd, pat)):
                    try:
                        os.remove(f)
                    except OSError:
                        pass

    def _palette_nbn(self, arr: np.ndarray, boost: float) -> np.ndarray:
        t0 = time.time()
        data = np.stack([arr[0], 0.5 * (arr[1] + arr[2]), 0.5 * (arr[1] + arr[2])], axis=-1)
        prm = {"oiii_boost": boost}
        out = np.moveaxis(nbn.normalise(data, **prm), -1, 0).astype(np.float32)
        info = nbn.info(data, **prm)
        self.record("palette (NBN HOO)", "port of NarrowbandNormalization.py process_image (Cuiv; Blanshan / Cranfield)", out, commands=["nbn.normalise(Ha=R, OIII=(G+B)/2, " + json.dumps({**nbn.DEFAULTS, **prm}) + ")"],
                    seconds=time.time() - t0, params={**nbn.DEFAULTS, **prm, "derived": info},
                    source="NarrowbandNormalization.py v1.0.5 defaults (Mode 1 blend 0.6, shadow 1.0, highlight 1.0, brightness 1.0, no SCNR); port validated to 1e-4 against the script's own function")
        return out

    # ---- 5. local contrast -----------------------------------------------------------------------------------------------
    def local_contrast(self, arr: np.ndarray, sky: float | None = None) -> np.ndarray:
        amount = float(np.clip(self.p.get("local_contrast", 0.5), 0.0, 2.0))
        if amount <= 0:
            self.record("local contrast", "none", None, status="skipped", note="local_contrast=0")
            return arr
        self._say("Detail")
        lum = finish.luminance(arr)
        h = _hist_stats(lum)
        mask, mi = nebula_mask(lum, h["mode"])
        lo, hi = float(mi["threshold"]), float(np.percentile(cv2.GaussianBlur(np.ascontiguousarray(lum), (0, 0), MASK_SIGMA)[mask], 95)) if mask.any() else float(mi["threshold"]) + 0.1
        w = self.range_mask(lum, mask, mi)
        k = LC_STRENGTH * amount
        longside = max(arr.shape[1:])
        grids = [max(2, int(round(longside / s))) for s in CLAHE_SCALES]
        lines = ["save lc_a", f"clahe {CLAHE_CLIP:g} {grids[0]}", "save lc_b", f'pm "$lc_a$+{k:.4f}*$lc_m$*($lc_b$-$lc_a$)"', "save lc_c",
                 f"clahe {CLAHE_CLIP:g} {grids[1]}", "save lc_d", f'pm "$lc_c$+{k:.4f}*$lc_m$*($lc_d$-$lc_c$)"']
        t0 = time.time()
        m3 = np.stack([w, w, w])
        out = self.siril("lc", arr, lines, "Local contrast", extra={"lc_m": m3})
        for f in ("lc_a", "lc_b", "lc_c", "lc_d"):
            fp = os.path.join(self.cwd, f + ".fit")
            if os.path.exists(fp):
                os.remove(fp)
        self.record("local contrast", "siril clahe + pm (background mask)", out, commands=lines, seconds=time.time() - t0,
                    params={"clip_limit": CLAHE_CLIP, "tile_px": list(CLAHE_SCALES), "grid": grids, "strength": round(k, 3), "mask_low": round(lo, 4), "mask_high": round(hi, 4)},
                    source="Brecher LHE (astrodoc.ca M51 / IC 1848 / Sh2-140): contrast limit 1.5, scales 150 and 40 px, strength 0.2-0.35 under a background-protecting mask, after NR; "
                           "the mapping of LHE to Siril clahe is not published (pro-workflow.md A11)")
        return out

    def sharpen(self, arr: np.ndarray) -> np.ndarray:
        amount = float(np.clip(self.p.get("sharpen", 0.0), 0.0, 1.5))
        if amount <= 0:
            self.record("sharpening", "none", None, status="skipped", note="sharpen=0")
            return arr
        self._say("Detail")
        lum = finish.luminance(arr)
        h = _hist_stats(lum)
        mask, mi = nebula_mask(lum, h["mode"])
        sm = cv2.GaussianBlur(np.ascontiguousarray(lum), (0, 0), MASK_SIGMA)
        lo = float(mi["threshold"])
        hi = float(np.percentile(sm[mask], 95)) if mask.any() else lo + 0.1
        w = np.clip((sm - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        w = (w * w * (3.0 - 2.0 * w)).astype(np.float32)
        k = SHARPEN_GAIN * amount
        lines = ["save sh_a", f"unsharp {SHARPEN_SIGMA:g} {k:.4f}", "save sh_b", 'pm "$sh_a$+$sh_m$*($sh_b$-$sh_a$)"']
        t0 = time.time()
        out = self.siril("sh", arr, lines, "Sharpening", extra={"sh_m": np.stack([w, w, w])})
        for f in ("sh_a", "sh_b"):
            fp = os.path.join(self.cwd, f + ".fit")
            if os.path.exists(fp):
                os.remove(fp)
        self.record("sharpening", "siril unsharp + pm (subject mask)", out, commands=lines, seconds=time.time() - t0, params={"sigma": SHARPEN_SIGMA, "amount": round(k, 4)},
                    source="Siril unsharp (Gaussian unsharp mask) under the subject mask after the noise reduction; Brecher: masked sharpening after NR (pro-workflow.md A11)")
        return out

    # ---- 6. saturation ------------------------------------------------------------------------------------------------------
    def saturation(self, arr: np.ndarray) -> np.ndarray:
        """Siril's GHS saturation stretch (``ght -sat``: the hyperbolic stretch applied to the HSL saturation, lightness kept) with D fitted so that the subject's
        mean saturation is multiplied by the ``saturation`` setting, under the range mask (the sky's chroma noise is not raised).  Replaces ``satu``: its gain is
        linear and gated at median + sigma, so on a faint subject (C 33 2026-10-15) +45 % came out as +17 % and the filaments stayed grey."""
        k = float(np.clip(self.p.get("saturation", 1.45), 1.0, SAT_FACTOR_MAX))
        if k - 1.0 < 1e-3:
            self.record("saturation", "none", None, status="skipped", note="saturation=1")
            return arr
        self._say("Colour")
        lum = finish.luminance(arr)
        h = _hist_stats(lum)
        mask, mi = nebula_mask(lum, h["mode"])
        w = self.range_mask(lum, mask, mi)
        s = hsl_saturation(arr)
        use = mask if mask.sum() > 50 else np.ones(mask.shape, bool)
        step = max(1, int(use.sum()) // 200000)
        sub, wsub = s[use][::step], w[use][::step]
        before = float(sub.mean())
        d, target = fit_saturation_d(sub, k, wsub)
        capped = d >= SAT_D_MAX - 1e-9
        if d <= 0:
            self.record("saturation", "none", None, status="skipped", params={"subject_saturation": round(before, 4), "factor": k},
                        note=f"the subject's mean saturation {before:.3f} already reaches {target:.3f}")
            return arr
        lines = ["save sa_a", f"ght -D={d:.6f} -B={SAT_B:g} -SP={SAT_SP:g} -LP=0 -HP=1 -sat", "save sa_b", 'pm "$sa_a$+$sa_m$*($sa_b$-$sa_a$)"']
        t0 = time.time()
        out = self.siril("sa", arr, lines, "Saturation", extra={"sa_m": np.stack([w, w, w])})
        for f in ("sa_a", "sa_b"):
            fp = os.path.join(self.cwd, f + ".fit")
            if os.path.exists(fp):
                fpr = fp
                os.remove(fpr)
        after = float(hsl_saturation(out)[mask].mean()) if mask.sum() > 50 else float(hsl_saturation(out).mean())
        self.record("saturation", "siril ght -sat + pm (range mask)", out, commands=lines, seconds=time.time() - t0,
                    params={"D_siril": round(d, 4), "D": round(float(np.expm1(d)), 2), "b": SAT_B, "SP": SAT_SP, "factor": k, "subject_saturation_before": round(before, 4),
                            "target": round(target, 4), "subject_saturation_after": round(after, 4), "subject_coverage": round(mi["coverage"], 3), "D_capped": capped},
                    note=f"D at its cap {SAT_D_MAX:g} (the target {target:.3f} would need a flattening stretch)" if capped else "",
                    source="Siril ght -sat (GHS saturation mode, ghsastro: the stretch on the saturation channel with the lightness restored); D fitted on the subject's "
                           "saturation histogram (the numpy GHS mirror reproduces Siril's -sat on HSL saturation to 1e-6, C 33 2026-10-15); under the range mask as Brecher's "
                           "saturation under a background-protecting mask")
        return out

    def range_mask(self, lum: np.ndarray, mask: np.ndarray, mi: dict, floor: np.ndarray | None = None) -> np.ndarray:
        """Range mask (PixInsight RangeSelection + blur): 0 on the sky, 1 on the subject, smooth between the sky floor and the subject's 95th percentile.
        ``floor``: a per-pixel sky floor in place of the nebula mask's single global threshold; see ``edge_sky_floor``."""
        sm = cv2.GaussianBlur(np.ascontiguousarray(lum), (0, 0), MASK_SIGMA)
        lo = float(mi["threshold"]) if floor is None else floor
        hi = float(np.percentile(sm[mask], 95)) if mask.any() else float(mi["threshold"]) + 0.1
        w = np.clip((sm - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
        return (w * w * (3.0 - 2.0 * w)).astype(np.float32)

    # ---- 7. stars -----------------------------------------------------------------------------------------------------------
    def stars(self, stars: np.ndarray) -> np.ndarray:
        p = self.p
        self._say("Stars")
        a = float(np.clip(p.get("star_stretch", STAR_STRETCH), 0.0, 8.0))
        m = star_stretch_m(a)
        lines = [f"mtf 0 {m:.6f} 1"]
        params = {"amount": a, "m": round(m, 6)}
        if self.dual:
            lines += ['pm "~$T"', "rmgreen 0", 'pm "~$T"']
            params["magenta_fix"] = "invert, rmgreen, invert"
        inten = float(p.get("star_intensity", 1.0))
        if abs(inten - 1.0) > 1e-3:
            lines += [f'pm "$T*{inten:.4f}"']
        sat = float(p.get("star_saturation", 1.0))
        if abs(sat - 1.0) > 1e-3:
            lines += [f"satu {sat - 1.0:.3f} 0"]
        t0 = time.time()
        out = self.siril("stars", stars, lines, "Star stretch")
        self.record("stars stretch", "siril mtf (Seti Astro Star Stretch)", out, commands=lines, seconds=time.time() - t0, params=params,
                    source="Seti Astro Star Stretch = mtf with m = 1/(3^a+1), a = 5 (SASpro star_stretch.py / numba_utils); dual-band magenta: invert, SCNR, invert (Brecher, astrodoc.ca IC 1848)")
        c = float(np.clip(p.get("star_color_preservation", 0.5), 0.0, 1.0))
        if c > 1e-3:
            # the colour-preserving stretch of Lupton et al. 2004 (PASP 116, 133; PixInsight's ArcsinhStretch): every pixel's RGB multiplied by one
            # factor F(I) / I of its intensity I (the channel mean), F the star stretch's own curve (mtf with m) times the intensity setting, then
            # clipped by dividing by the largest channel - the channel ratios, and so the order of R, G and B, stay the linear ones.  Blended with the
            # per-channel stretch by c.  (A Siril rgbcomp -lum composition in L*a*b* kept each pixel's a* / b* while replacing its lightness: the
            # dim wings took the core's chroma, out of gamut, and gamut clipping shifted their hue - 45 % of M 31's well-measured stars off the
            # blackbody locus at c = 0.5)
            t0 = time.time()
            x = np.clip(np.nan_to_num(stars.astype(np.float32)), 0.0, None)
            I = x.mean(0)
            F = mtf_curve(np.clip(I, 0.0, 1.0), m) * inten
            k = np.where(I > 1e-12, F / np.maximum(I, 1e-12), 0.0)
            lup = x * k[None]
            mx = lup.max(0)
            lup = np.where(mx > 1.0, lup / np.maximum(mx, 1e-12), lup)[...]
            out2 = np.clip((1.0 - c) * out + c * lup, 0.0, 1.0).astype(np.float32)
            self.record("stars colour preservation", "Lupton et al. 2004 colour-preserving stretch", out2, seconds=time.time() - t0, params={"star_color_preservation": c},
                        source="RGB scaled by F(I)/I, clipped by the largest channel (Lupton, Blanton, Fekete et al. 2004, PASP 116, 133; PixInsight ArcsinhStretch)")
            return out2
        return out

    def halos(self, arr: np.ndarray) -> np.ndarray:
        h = float(np.clip(self.p.get("halo_suppress", 0.0), 0.0, 1.0))
        if h <= 1e-3:
            self.record("halo suppression", "none", None, status="skipped", note="halo_suppress=0")
            return arr
        line = f"unpurple -starmask -blue={h:.3f} -thresh={UNPURPLE_THRESH:g}"
        t0 = time.time()
        try:
            out = self.siril("up", arr, [line], "Halo suppression")
        except SirilError as e:
            self.record("halo suppression", "siril unpurple", None, status="failed", commands=[line], note=str(e).splitlines()[0][:200])
            return arr
        self.record("halo suppression", "siril unpurple", out, commands=[line], seconds=time.time() - t0, params={"blue": h, "thresh": UNPURPLE_THRESH},
                    source="Siril unpurple (blue / purple fringes and halos around bright stars, star mask), the amount from halo_suppress")
        return out

    def grade(self, arr: np.ndarray) -> np.ndarray:
        if finish.grade_neutral(self.p):
            self.record("colour grade", "none", None, status="skipped", note="neutral grade")
            return arr
        t0 = time.time()
        out = finish.color_grade(arr, self.p)
        keys = ("grade_amount",) + finish.GRADE_KEYS
        self.record("colour grade", "finish.color_grade (OKLab)", out, seconds=time.time() - t0, params={k: self.p.get(k) for k in keys},
                    source="the application's colour grade: object white balance (temperature, tint), warm / cool hue rotation and chroma, S-curve of the object's lightness, sky protected")
        return out

    def recombine(self, starless: np.ndarray, stars: np.ndarray | None) -> np.ndarray:
        s = float(np.clip(self.p.get("star_reduction", 0.4), 0.0, 1.0))
        if stars is None or s >= 1.0:
            self.record("recombine", "none", None, status="skipped", note="no star layer (starless render)" if stars is None else "star_reduction=1: starless")
            return starless
        v = 0.5 * (1.0 - s)
        lines = ['pm "~(~$rc_sl$*~$rc_st$)"', "save rc_full"]
        if s > 0:
            lines += [f'pm "~((~mtf(~{v:.4f},$rc_full$)/~mtf(~{v:.4f},$rc_sl$))*~$rc_sl$)"']
        t0 = time.time()
        out = self.siril("rc", None, ["load rc_sl"] + lines, "Recombine", extra={"rc_sl": starless, "rc_st": stars})
        fp = os.path.join(self.cwd, "rc_full.fit")
        if os.path.exists(fp):
            os.remove(fp)
        self.record("recombine + star reduction", "siril pm (screen; DSA star reduction)", out, commands=lines, seconds=time.time() - t0,
                    params={"star_reduction": s, "dsa_value": round(v, 4)},
                    source="screen blend ~(~starless * ~stars) (RC-Astro, Brecher op_screen); star reduction = siril-scripts DSA-Star_Reduction.py pm formula, value 0.2 default (0.5 = none); "
                           "Brecher StarReduction strength 0.4")
        return out

    # ---- 8. final black point, ICC ---------------------------------------------------------------------------------------------
    def finalise(self, arr: np.ndarray) -> np.ndarray:
        self._say("Exporting")
        p = self.p
        lum = finish.luminance(arr)
        x = finish.sample(lum)
        toe = float(np.percentile(x, TOE_PERCENTILE))
        sm = finish._sky_mask(lum, edge=finish.EDGE_CROP)
        sky = float(np.median(lum[sm])) if sm.sum() > 100 else float(np.median(x))
        # the black point goes to the histogram toe, but never into the background body: Brecher's backgrounds are 0.09-0.11, so it may take the background down to 0.09 at most
        applied = toe if (sky - toe) / max(1.0 - toe, 1e-9) >= SKY_BAND[0] else 0.0
        bp = max(applied, 0.0) + float(p.get("black_point", 0.0))
        lines = []
        if bp > 1e-5:
            lines.append(f"linstretch -BP={bp:.6f}")
        mid = finish.brightness_to_mid(float(p.get("brightness", 0.0)))
        level = None
        if str(p.get("sky_levelling", False)).lower() not in ("false", "0", "", "none"):
            # the final sky level: the midtones that take the sky (after the black point) exactly to sky_level - mtf(x; m) = (m-1)x / ((2m-1)x - m)
            # solved for m - the way a finisher sets the sky last (Brecher's backgrounds 0.09-0.11; the sky_median gate 0.08-0.12)
            y = float(np.clip(float(p.get("sky_level", 0.10)), 0.02, 0.5))
            x = (sky - bp) / max(1.0 - bp, 1e-9)
            if 1e-5 < x < 0.999:
                mid = float(np.clip(x * (y - 1.0) / (2.0 * x * y - x - y), 0.02, 0.98))
                level = y
        clamped = None
        if level is None and str(p.get("sky_median_guard", True)).lower() not in ("false", "0", "", "none"):
            # The sky's level is the picture's own spec (``sky_median`` 0.08 - 0.12), and the brightness the chain was
            # handed can take it above the window: M 31 (Seestar, 2026-11-01) left finalise with the sky at 0.134 and
            # was delivered failing its own gate, with nothing in the record saying which number was missed.  The same
            # closed-form solve as ``sky_levelling`` above, to the edge of the window rather than to ``sky_level``, so
            # a render whose sky is already inside it is never touched.  Only the bright side is corrected: the one
            # this campaign measured.  Lifting a frame-filling subject to a number that is not measuring sky is a
            # different decision and no dataset has asked for it.
            lo, hi = gates.LIMITS["sky_median"]
            x0 = (sky - bp) / max(1.0 - bp, 1e-9)
            if 1e-5 < x0 < 0.999:
                y0 = float(mtf_balance(mid, np.float64(x0)))
                if y0 > hi:
                    mid = float(np.clip(x0 * (hi - 1.0) / (2.0 * x0 * hi - x0 - hi), 0.02, 0.98))
                    clamped = {"from": round(y0, 4), "to": hi, "brightness": round(float(p.get("brightness", 0.0)), 4)}
        if abs(mid - 0.5) > 1e-4:
            lines.append(f"mtf 0 {mid:.4f} 1")
        self.final_levels = (float(bp), float(mid))                          # the starless the star gates compare against goes through the same levels
        lines += ["icc_assign sRGB", "icc_convert_to sRGB relative"]
        t0 = time.time()
        out = self.siril("fin", arr, lines + ["save finish"], "Final", out="finish")
        self.record("final black point + ICC", "siril linstretch / icc_convert_to", out, commands=lines, seconds=time.time() - t0,
                    params={"BP": round(bp, 6), "toe": round(toe, 5), "toe_percentile": TOE_PERCENTILE, "background": round(sky, 4), "icc": "sRGB, relative colorimetric",
                            "midtones": round(mid, 5), "sky_level": level, "sky_median_clamped": clamped},
                    source="Brecher: black point at the toe of the histogram, ICC to sRGB IEC61966-2.1 relative colorimetric (pro-workflow.md A16)")
        return out

    # ---- the whole chain -----------------------------------------------------------------------------------------------------
    _STATE_SKIP = ("records", "info", "n", "p", "memo")

    def _step(self, name: str, fn, arg, key):
        """``fn(arg)`` through the step memo: ``(output, key of the output)``.  ``key`` names the input (None = no memo for this run).  On a hit the
        step's records, ``info`` entries and attributes are replayed and a copy of its output returned (later steps may work in place)."""
        p = self.p
        if self.memo is None or key is None or not isinstance(p, _ReadLog) or p.wrote:
            return fn(arg), None
        hit = self.memo.find(key, name, p)
        if hit is not None:
            for r in hit["records"]:
                self.n += 1
                self.records.append({**r, "n": self.n})
            self.info.update(hit["info"])
            vars(self).update(hit["attrs"])
            return _copy_value(hit["value"]), hit["out"]
        before, info0, nrec = dict(vars(self)), dict(self.info), len(self.records)
        p.reads = set()
        try:
            value = fn(arg)
            reads = p.reads
        finally:
            p.reads = None
        if p.wrote:
            return value, None
        reads = dict(dict.items(p)) if _ALL in reads else {r: dict.get(p, r, _MISSING) for r in reads}
        out = hashlib.sha1(repr((key, name, sorted((repr(k), repr(v)) for k, v in reads.items()))).encode()).hexdigest()
        self.memo.store({"in": key, "name": name, "out": out, "reads": reads, "value": _copy_value(value),
                         "records": [dict(r) for r in self.records[nrec:]],
                         "info": {k: v for k, v in self.info.items() if k not in info0 or info0[k] is not v},
                         "attrs": {k: v for k, v in vars(self).items() if k not in self._STATE_SKIP and (k not in before or before[k] is not v)}})
        return value, out

    def run(self, base: np.ndarray, stars: np.ndarray | None) -> tuple[np.ndarray, dict]:
        self.n = 0
        self.record("starless, linear", "input (post.restore -> post.remove_stars)", base, linear=True, params={"stars_layer": stars is not None},
                    note="shown through a linked autostretch")
        k = repr(("starless", self.dual)) if self.memo is not None else None
        s, k = self._step("linear", self.prepare_linear, base, k)
        s, k = self._step("stretch", self.stretch_prepared, s, k)
        s, k = self._step("highlights", self.highlights, s, k)
        pre = s
        s, k = self._step("denoise", self.denoise, s, k)
        s, k = self._step("banding", lambda a: self.banding(a, "stretched starless", False, noise_from=pre), s, k)
        s, k = self._step("colour", self.colour, s, k)
        self._pre_local_contrast = s
        s, k = self._step("local contrast", self.local_contrast, s, k)
        s, k = self._step("sharpen", self.sharpen, s, k)
        self._pre_saturation = s
        s, k = self._step("saturation", self.saturation, s, k)
        t, kt = None, "no stars"
        if stars is not None and float(self.p.get("star_reduction", 0.4)) < 1.0:
            t, kt = self._step("stars", self.stars, stars, repr(("stars", self.dual)) if self.memo is not None else None)
        # the gate measures the delivered picture, so the loop runs after the composition and recomposes every candidate it tries
        (s, out), _ = self._step("sky noise", lambda sl: self._close_sky_noise(sl, lambda x: self._compose(x, t)), s,
                                 None if k is None or kt is None else repr((k, kt)))
        out = self._close_clipping(s, t, out)
        self.info.update(chain=self.records, starless=s, starless_delivered=self.delivered_starless(s))
        return out, self.info

    def _close_clipping(self, starless: np.ndarray, stars: np.ndarray | None, out: np.ndarray) -> np.ndarray:
        """The clipping gates on the delivered picture, as ``_close_sky_noise`` does for the sky's noise.  Each search before the delivery takes
        the largest setting its own measure allows, so the steps after it (the escalated denoiser, the sharpening on the composed picture) can
        carry the picture over the gate, several steps sharing in it.  When the delivered picture has more clipped or white pixels than the gates
        allow, the intensities that can be re-run on it and raise its brightest pixels - Cosmic Clarity Sharpen's stellar amount, the local
        contrast and the saturation stretch's factor - are scaled together, one common fraction of the way from their weakest to the settings,
        and the fraction bisected (CLIP_CLOSE_STEPS + 1 trials, two-sided: a candidate that clips goes weaker, one that fails a gate the delivered
        picture passes goes stronger) to the strongest that passes; the picture is recomposed.  Otherwise nothing is changed and that is recorded."""
        lim_c, lim_w = float(gates.LIMITS["clipped"]), float(gates.LIMITS["white"])

        def fails(img: np.ndarray) -> tuple[bool, dict]:
            st = level_stats(img)
            return st["clipped"] > lim_c or st["white"] > lim_w, {"clipped": st["clipped"], "white": st["white"]}

        if not self.external:
            return out
        bad, st0 = fails(out)
        if not bad:
            return out
        base_failed = set(self._failed_of(out, starless)) - {"clipped", "white"}
        new_fails = lambda img, sl: sorted(set(self._failed_of(img, sl)) - {"clipped", "white"} - base_failed)   # noqa: E731  a gate passing now must keep passing
        t0 = time.time()
        pre = getattr(self, "_pre_saturation", None)
        knobs = []
        if str(self.p.get("cc_sharpen") or "off").lower() in ("both", "stellar"):
            knobs.append(("cc_stellar_amount", float(np.clip(self.p.get("cc_stellar_amount", CC_SHARPEN_AMOUNT), 0.0, 1.0)), 0.0))
        pre_lc = getattr(self, "_pre_local_contrast", None)
        if pre_lc is not None and float(self.p.get("local_contrast", 0.0)) > LC_WEAKEST:
            knobs.append(("local_contrast", float(self.p.get("local_contrast", 0.0)), LC_WEAKEST))
        if pre is not None and float(self.p.get("saturation", 1.0)) > 1.0:
            knobs.append(("saturation", float(np.clip(self.p.get("saturation", 1.0), 1.0, SAT_FACTOR_MAX)), 1.0))
        saved_p, saved_rec = dict(self.p), self.records

        if not knobs:
            return out

        def render(frac: float) -> tuple[np.ndarray, np.ndarray]:
            vals = {k: round(w + frac * (a - w), 3) for k, a, w in knobs}
            self.p = {**saved_p, **vals}
            self.records = []
            if "local_contrast" in vals:
                sl = self.saturation(self.sharpen(self.local_contrast(pre_lc)))
            elif "saturation" in vals:
                sl = self.saturation(pre)
            else:
                sl = starless
            used = getattr(self, "_sky_noise_used", None)
            if sl is not starless and used is not None and self.denoiser is not None:
                sl, _cmd = self.denoiser[1](used, sl)                       # the starless as the chain has it: with the sky-noise step's escalation
            return self._compose(sl, stars), sl

        attempts: list[dict] = [{"fraction": 1.0, **{k: a for k, a, _ in knobs}, **st0}]
        try:
            # two-sided: clipping wants weaker settings, the star gates stronger ones; a candidate that clips goes weaker, one that passes the
            # clipping gates but fails another gate goes stronger, one that passes both is kept and a stronger one tried
            lo, hi, best, best_out = 0.0, 1.0, None, None
            for _ in range(CLIP_CLOSE_STEPS + 1):
                f = round(0.5 * (lo + hi), 4)
                cand, sl = render(f)
                clips, stc = fails(cand)
                nf = [] if clips else new_fails(cand, sl)
                rej = bool(nf)
                attempts.append({"fraction": f, **stc, "new_failures": nf})
                if clips:
                    hi = f
                elif rej:
                    lo = f
                else:
                    lo, best, best_out = f, f, cand
        finally:
            self.p, self.records = saved_p, saved_rec
        if best is None:
            self.record("clipping gate", "none", None, status="skipped", seconds=time.time() - t0, params={"attempts": attempts},
                        note="no common fraction of the sharpening, local contrast and saturation brings the delivered picture inside the clipping gates "
                             "without failing another gate: kept as composed")
            return out
        used = {k: round(w + best * (a - w), 3) for k, a, w in knobs}
        self.p.update(used)
        self.record("clipping gate", "+".join(k for k, _, _ in knobs), best_out, seconds=time.time() - t0,
                    params={"fraction": best, "asked": {k: a for k, a, _ in knobs}, "used": used, "attempts": attempts, "limits": {"clipped": lim_c, "white": lim_w}},
                    note=f"the delivered picture had {st0['clipped'] * 100:.3f} % clipped / {st0['white'] * 100:.3f} % white pixels; the finishing intensities "
                         f"scaled to {best:g} of the way from their weakest to the settings")
        return best_out

    def _compose(self, starless: np.ndarray, stars: np.ndarray | None) -> np.ndarray:
        """The delivered picture from a starless and a star layer: screen the two, tame the halos, sharpen the whole,
        grade it, then the black point and the ICC conversion.  Factored out of ``run`` because the closing sky-noise
        step has to render a candidate exactly this way to be comparable with the picture it is offered in place of.
        """
        out = self.recombine(starless, stars)
        if stars is not None:
            out = self.halos(out)
        out = self.cc_sharpen(out)
        out = self.grade(out)
        return self.finalise(out)


def _community_python() -> str | None:
    p = os.path.join(config.home_dir(), ".local", "share", "siril", "venv", "bin", "python")
    return p if os.path.exists(p) else None


def _community_script(rel: str) -> str | None:
    p = os.path.join(config.home_dir(), ".local", "share", "siril-scripts", rel)
    return p if os.path.exists(p) else None
