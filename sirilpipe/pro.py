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

import json
import math
import os
import re
import subprocess
import time
from statistics import NormalDist

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
GHS2_D = 2.5                       # pass 2 D (Siril -D scale 0-10; the ``ghs_d2`` setting): 2.5 measured on IC 1805 / M 42 / M 31 2026-10-15 (1 left the nebula flat, 4 posterised it)
GHS2_D_FILLING = 1.5               # pass 2 D when the subject covers more than GHS2_FILLING_COVERAGE: M 31's core blew at 2.5 and the frame-filling IC 1805 (2026-10-16) became a red wall
GHS2_FILLING_COVERAGE = 0.3        # = SKY_STEPS_MAX_COVERAGE
TOE_PERCENTILE = 0.1               # "linear black point just left of the histogram rise": the level below which 0.1 % of the pixels lie (the clipping limit of Siril / fas37, VeraLux)
HP_PERCENTILE = 99.5               # HP < 1 (highlight protection) starts at this percentile of the subject
STAT_SMALL, STAT_FILLING = 0.10, 0.25   # Statistical_Stretch median target: small target (pro-workflow.md A8 route 2 / SW Skies) / frame-filling target (script default)
STAT_FILLING_COVERAGE = 0.5        # the object covers at least this fraction of the frame
STAT_SIGMA, STAT_HDR_AMOUNT, STAT_HDR_KNEE = 5.0, 0.15, 0.75
WAVESCALE = (5, 1.5, 5.0)          # compute_wavescale_hdr: scales, compression, mask gamma (HDR_multiscale.py defaults / SASpro)
NXT_DC = 0.95                      # NXT colour denoise (manual: colour 90-100 %)
NXT_DN = (0.60, 0.85)              # NXT --dn on stretched starless (RC-Astro / Brecher); 1.0 is the tool's own ceiling, not a tuned value
DENOISE_ESCALATION = 3             # extra runs of the stretched-image denoiser, at most, when the sky it leaves is noisier than
                                   # sky_pixel_noise allows: the ladder from the requested amount to the tool's ceiling in this
                                   # many steps, so a frame that cannot be denoised clean costs a bounded amount of time and is
                                   # then left alone with every attempt recorded
CHROMA_SIGMA_MAX = 2.0                 # px, the chroma kernel at ``chroma_denoise`` 1.0: our mapping, because NXT's --dc and
                                       # Cosmic Clarity's --denoise-color publish no kernel scale (manual: colour 90-100 %)
CLAHE_CLIP = 1.5                   # Brecher's LHE contrast limit
CLAHE_SCALES = (150, 40)           # px, Brecher's LHE scales (M51: 150 then 40; IC 1848 40 then 150)
LC_STRENGTH = 0.4                  # blend per unit of ``local_contrast`` (0.5 -> 0.2, the lowest published strength)
LC_EDGE_BAND = 1.0                   # px, in units of CLAHE_SCALES[0]: how deep the frame's edge is asked what *its* sky's top is
LC_EDGE_MAX_EXCESS = 3.0               # x the frame's own sky scatter: a band lying higher above the frame's sky than this is subject, not a model residue, and keeps the global threshold
SAT_B, SAT_SP = 0.0, 0.0           # GHS saturation stretch (ght -sat): the plain hyperbolic curve on the saturation channel, strongest at 0 (grey gains most, saturated pixels are compressed, nothing clips)
SAT_D_MAX = 1.5                    # Siril -D scale (GHS D 3.5), the fit's cap: measured 2026-10-16 on C 33 / IC 1805, 1.5 vivid and still textured, 2.0 flattens IC 1805, 6.0 posterised both
SAT_FACTOR_MAX = 4.0               # the ``saturation`` setting: the subject's mean saturation x 1 .. x 4 (C 33: 0.09 after the palette, the HOO target 0.45)
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
SCNR_SKY_TRIES = (1.0, 0.5, 0.25)   # an SCNR that colours the sky is retried at these fractions of the requested amount before it is refused
SCNR_SKY_WORSE = 2.0                # how much more coloured than it arrived the sky may leave SCNR looking: past the gate's own limit *and* past
                                    # this multiple of what it came with, the step is the thing that coloured it
DENOISE_SKY_WORSE = 2.0             # how much more coloured than it arrived the sky may leave the stretched-image denoiser looking: past the gate's own
                                    # limit *and* past this multiple of what it came with, the step is the thing that coloured it (SCNR's rule, the same
                                    # evidence class; the denoiser was the one colour-changing step of the chain with no such rule)
SKY_LUMP_CELL = 128                 # px, the cell the sky's unevenness is measured on: the scale at which the stretch's black clip leaves its steps
STRETCH_CLIP_GAIN = 1.4             # the clip-free stretch ships only when it leaves the sky this times more even than the clipped one
STAT_CLIP_FLOOR = 1.0               # % of sky: below this the clipped stretch's sky is already flat, and the clip-free trial - which costs a whole second run of the
                                    # script - could only win a fraction of a percent nobody can see, so it is not made
FRAME_FILLING = 0.5                # the object covers at least this fraction of the frame: ``auto_stretch`` raises the background to the top of Brecher's band
SKY_STEPS_MAX_COVERAGE = 0.3       # above this subject coverage the frame has no sky to sample: the colour-gradient step is skipped (C 20: the degree-2
                                   # per-channel model took the North America Nebula's own red, 2026-10-15); DBE needs true sky samples.  It gates the global
                                   # polynomial only - see Finisher.stretch for why the neutralisation runs whatever the coverage


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


SKY_PEDESTAL_CAP = 0.02              # the correction never exceeds 2 % of the sky, and only where the coverage map says the pixels are degraded
SKY_PEDESTAL_EPS = 0.0005            # below 0.05 % of the sky there is nothing worth taking away
SKY_PEDESTAL_BAND = 16               # px on each side of a region's own border that are compared - the region's skin against the frame's skin across that same boundary - so a
                                     # smooth gradient across the frame falls out of the difference and only a step at the class boundary, which is what a pedestal is, can
                                     # survive it
SKY_PEDESTAL_MIN_PIXELS = 1024       # each of the two skins needs this many of the frame's sky pixels before its level is believed
SKY_PEDESTAL_MIN_SIGMA = 3.0         # ... and the step between them has to stand this many standard errors of the two estimates clear of zero: under that it is the
                                     # noise of two different parts of the frame, not a pedestal
SKY_PEDESTAL_WITNESS = 12.0          # sigma (px) of the plane the witness is taken against: stars and filaments are rejected, a large-scale level offset is not - it is the
                                     # very thing being measured, and it may not disqualify the pixels that carry it
SKY_PEDESTAL_WITNESS_K = 4.0         # ... within this many pixel-level sigma of that plane
SKY_COVERAGE_LOW = 0.7               # a pixel is degraded below this fraction of the frame's median coverage: only its sky is re-levelled
SKY_COVERAGE_HIGH = 0.9              # the reference sky is measured at or above this fraction, so the reference is always the well-dithered part of the frame
EDGE_RING = 64                       # the depth of the frame ring measured against the interior, line by line - the boundary behaviour of the stages before the stretch lives
                                     # in its first pixels, see edge_ring
EDGE_RING_DEPTH = 16                 # the deepest a boundary artefact reaches, and therefore the control region: the frame line measured 2 px deep on IC 1848 and 8 px on
                                     # M 31, while a deviation still standing this far in is sky (a nebula's edge, a gradient) and the tool leaves every last bit of it
EDGE_RING_MIN_SIGMA = 3.0            # a line's offset must stand this many standard errors of its own median off the interior to be believed
EDGE_RING_MIN = 0.001                # ... and be at least this fraction of the sky, or there is nothing worth taking (the lines that reach the export start at 0.1 %)
EDGE_RING_BUDGET = 0.32               # the light the ring may take back along one line through it, summed over the ring's lines, as a fraction of the sky: amplitude on its own
                                      # cannot tell a boundary artefact from a nebula, so what is bounded is the amount taken.  Measured on the frames - the excess a frame line
                                      # carries is 4 %-px on M 31 and 9 on IC 434 (percent of the sky, summed over the depth) and a real processing step at 3 % over 8 px is 27
                                      # - while a nebula's edge confined to the outer half of the ring reaches 36, and that is what this bound exists to leave alone
EDGE_RING_MAX = 0.30                  # ... and no single line by more than this fraction of the sky: past that the line has no sky in it to level, and supplying one would be
                                      # inventing what the frames never recorded
EDGE_RING_SHARP = 0.35               # the lines behind the border may carry at most this fraction of its deviation: one that goes on behind the border is sky, and sky stays
EDGE_RING_MIN_SIDES = 2              # a deviation smaller than the one below has to appear on this many sides: the boundary is a property of the frame, and a lone offset the
                                     # size of faint sky is far more likely to be sky than to be the frame's own edge
EDGE_RING_LONE = 0.01                # ... unless one side carries at least this fraction of the sky: a step that size confined to the border band is no sky feature that happened
                                     # to end near the edge, and the stretch's toe turns it into a line across the export (IC 434's top border measures -3.8 % of the sky)
SKY_RIM_REPEAT = 2        # a depth beyond the run has to stand again this many times in a row to be a second step at all: one line reaching the bar is a fluctuation of the
                          # first, not a structure behind it (C 20's right border lost its correction to exactly one such sample, at 96 px)
SKY_RIM_ECHO = 0.5        # ... and it has to carry this fraction of the border's own deviation while it does: sky lying behind a pedestal sits well below it, and a deviation
                          # half the border's height that returns over two depths is the subject or the gradient coming back, which is what this refuses


def _sigma(v: np.ndarray) -> float:
    """A class of pixels' own noise: 1.4826 x its median absolute deviation, robust to the nebulosity a cell or a border can carry."""
    return (1.4826 * float(np.median(np.abs(v - np.median(v))))) or 1e-12

def edge_sky_floor(lum: np.ndarray, lo: float, band: int, stats: dict) -> tuple[np.ndarray, dict]:
    """Raise the mask's sky floor near each edge to that edge's own sky top, where the band is sky.

    ``range_mask`` thresholds the smoothed lightness at one global value - the sky mode plus ``MASK_K`` sigma of the sky scatter - which is the wrong
    reference at the frame's edge.  A band the background model left uniformly elevated clears that threshold everywhere at once, is classed subject,
    and clahe - whose tile there has no pixels past the edge to judge it against - lifts it again on top of a stretch whose slope is already ~100.
    Measured on Sh2-171 (2026-10-30): this step took the left border from +161 % of the display sky to +321 % while leaving the interior alone, and the
    export came out with a band across the frame.  Each edge is therefore asked for its own sky top - its median plus ``MASK_K`` sigma, the definition
    ``nebula_mask`` uses - and the floor is raised to it, ramped linearly to nothing at ``band`` px so the mask cannot seam there.  A band lying more
    than ``LC_EDGE_MAX_EXCESS`` of the frame's own sky scatter above that sky is not a residue the model left but the subject itself - M 31's galaxy
    running off the edge - and is left exactly as the global threshold leaves it.  Returns the floor and, per side, what it was judged to be.
    """
    h, w = lum.shape
    floor = np.full((h, w), lo, np.float32)
    sm = cv2.GaussianBlur(np.ascontiguousarray(lum), (0, 0), MASK_SIGMA)
    sides = {}
    for name, strip, ramp in (("top", sm[:band], np.linspace(1.0, 0.0, band)[:, None]),
                              ("bottom", sm[h - band:], np.linspace(0.0, 1.0, band)[:, None]),
                              ("left", sm[:, :band], np.linspace(1.0, 0.0, band)[None, :]),
                              ("right", sm[:, w - band:], np.linspace(0.0, 1.0, band)[None, :])):
        med = float(np.median(strip))
        if med - float(stats["sky"]) > LC_EDGE_MAX_EXCESS * max(float(stats["sigma"]), 1e-6):
            sides[name] = "subject"                                      # far above the frame's own sky: the subject lying on the border, not a residue the model left
            continue
        top = med + MASK_K * max(_sigma(strip), 1e-6)                    # the top of this band's own sky population
        frac = float(np.mean(strip <= top))
        if top <= lo:
            sides[name] = "clean"
            continue
        sel = ((slice(0, band), slice(None)) if name == "top" else (slice(h - band, h), slice(None)) if name == "bottom"
               else (slice(None), slice(0, band)) if name == "left" else (slice(None), slice(w - band, w)))
        floor[sel] = np.maximum(floor[sel], lo + ramp.astype(np.float32) * (top - lo))
        sides[name] = f"sky {round(100 * frac)} %"
    return floor, sides

def _sky_witness(lum: np.ndarray) -> np.ndarray:
    """The pixels this tool may consult: everything within ``SKY_PEDESTAL_WITNESS_K`` sigma of the plane a ``SKY_PEDESTAL_WITNESS`` px smoothing leaves - stars and
    filaments out, and **blind to large-scale level differences**, which is what a pedestal is.  ``finish._sky_mask`` cannot serve here: it thresholds a 64 px smoothing at
    its darkest 45 %, so a rim that sits a percent of the sky bright is declared not-sky and the tool is left without a witness exactly where its subject is (measured on
    the pedestal's own test frame: 0 sky pixels on both sides of the boundary it exists to compare)."""
    big = cv2.GaussianBlur(np.ascontiguousarray(lum, np.float32), (0, 0), SKY_PEDESTAL_WITNESS)
    d = np.asarray(lum, np.float32) - big
    sig = 1.4826 * float(np.median(np.abs(d - np.median(d))))                                    # the pixel-level scatter; the subject lies entirely on the bright side
    return d <= SKY_PEDESTAL_WITNESS_K * max(sig, 1e-9)


def _boundary_step(v: np.ndarray, region: np.ndarray, high: np.ndarray, sky: np.ndarray, din: np.ndarray, dout: np.ndarray,
                   band: int) -> tuple[float | None, float, int, int]:
    """The level difference across the border a degraded region shares with the well-covered frame, by **regression discontinuity**: the median of the sky in every 1 px shell of
    that border, on both sides of it, fitted against signed distance from the border with terms for the level, the local slope and the local curvature - the pedestal being the
    **jump at the border**, and nothing else.  The two shape terms are the point of the fit: a nebula's wing or a gradient rising past the region is a slope, and comparing the
    two sides' levels without a term for it reads that rise as a pedestal (measured: a wing of 8 % of the sky over 300 px leaks 0.43 % of the sky as one, and the stretch's slope
    turns it into exactly the band this tool exists to remove).  The shells are symmetric, so curvature cannot be confounded with the jump; each shell enters with the weight of
    its own median - 1.253 sigma over the root of its count - and those weights are what the reported standard error of the jump is made of.  Stars, filaments and unrecorded
    pixels are already out of ``sky``.  Returns the jump, its standard error, and the sky population of each side."""
    t, y, w, n_in, n_out = [], [], [], 0, 0
    for k in range(1, int(band) + 1):
        for sign, side, d in ((-1.0, region, din), (1.0, high, dout)):
            s = v[side & sky & (d >= k - 0.5) & (d < k + 0.5)]
            if s.size < 64:
                continue                                                                        # a shell too thin to have a level of its own: it is left out of the fit, never averaged in
            sig = _sigma(s)
            t.append(sign * (k - 0.5))
            y.append(float(np.median(s)))
            w.append(s.size / (1.5708 * sig * sig))                                             # the inverse variance of that median
            if sign < 0:
                n_in += int(s.size)
            else:
                n_out += int(s.size)
    if n_in < SKY_PEDESTAL_MIN_PIXELS or n_out < SKY_PEDESTAL_MIN_PIXELS or len(t) < 6:
        return None, 0.0, n_in, n_out
    t, y, W = np.asarray(t), np.asarray(y), np.asarray(w)
    X = np.column_stack([np.ones_like(t), t, np.square(t), (t < 0).astype(np.float64)])         # level, slope, curvature, and the jump at the border
    A = X.T @ (X * W[:, None])
    if np.linalg.matrix_rank(A) < X.shape[1]:
        return None, 0.0, n_in, n_out
    Ainv = np.linalg.inv(A)
    return float((Ainv @ (X.T @ (y * W)))[3]), float(math.sqrt(max(float(Ainv[3, 3]), 0.0))), n_in, n_out


def sky_pedestal(base: np.ndarray, ref: float, cov: np.ndarray | None = None) -> tuple[np.ndarray | None, dict]:
    """The sky's slowly varying pedestal, measured as a **step between provenance classes at their shared border**.

    Why the stretch cares: its slope at the sky is ~120 at ``-median 0.1 -sigma 5`` (the black point sits five noise-sigma below the sky, so the rescaled median is 0.00075
    and the map has to lift that to 0.10).  A pedestal of 0.47 % of the sky at the frame border - what the low-coverage dithered rim of a mosaic carries - therefore becomes
    the 40-60 % band the export shows, and ``neutralise``'s one offset per channel cannot reach a step that varies across the frame.

    What a pedestal is: the level one set of frames leaves behind.  It is shared by every pixel of one origin, and it shows as a **step** where that origin meets the pixels the
    other frames built - so that is where it is measured, and as a step and nothing else: ``_boundary_step`` fits the sky's level in 1 px shells on both sides of the border the
    region shares with the well-covered frame - or, when a gradual coverage falloff leaves the two classes too far apart to share a border, with the frame immediately outside
    the region's own class - granting that level its own slope and curvature, and the pedestal is the jump at the border alone.  A real one is on the order of
    half a percent of the sky; a jump the tool cannot tell from the noise of the two classes (``SKY_PEDESTAL_MIN_SIGMA`` standard errors, never less than
    ``SKY_PEDESTAL_EPS``) is refused, because everything this tool does after measuring is irreversible.

    The witness is the frame's sky at its own scale, ``_sky_witness``: stars and filaments are excluded, while a large-scale offset cannot exclude the pixels carrying it.

    Why the reference has to be adjacent: the previous version fitted a plane through the well-covered cells of the whole frame and measured the degraded cells against it,
    which asks that plane to be *extrapolated* into the region - and the regions are exactly where the coverage ran out, at the frame's edges.  Measured on IC 1848
    (2026-10-30) it took +0.86 % of the sky along the frame's left edge, where the two classes' own medians differ by 0.1 %: the region lies under the nebula complex, the
    ceiling rejected every cell the nebula touched, and what was left of the fit was no longer the sky.  The stretch's slope sprayed that correction out as a bright red band
    down the export's left edge - the rectangular patch this tool exists to remove, remade by the tool itself.  Comparing a region against the frame immediately around it
    cannot be fooled that way: whatever gradient or nebulosity lies near the region sits on both sides of the comparison.

    One number per region (the largest channel's step), applied to **exactly the pixels the coverage map calls degraded** and nowhere else: a pedestal is a property of the
    provenance class, so the correction has to have that class's own shape, and a correction faded out at the region's edge leaves half the step standing precisely where the
    step is (measured: a correction faded over 16 px took 53 % of a 1.5 %-of-sky pedestal off and left the rest at the border, which is the line the export shows).  Interior
    nebula sits at exactly zero correction either way, and what guards it against a wrong measurement is the refusal gates above, not a taper that also disarms the tool.
    Without a coverage map nothing is modelled: guessing here costs nebulosity."""
    if cov is None:
        return None, {"cells": 0, "note": "no coverage map: the pedestal is not modelled"}
    a = np.asarray(base, np.float64)
    ch, h, w = a.shape
    c = np.asarray(cov, np.float64)
    if c.shape != (h, w):
        c = cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA)
    cmed = float(np.median(c))
    if cmed <= 0:
        return None, {"cells": 0, "note": "coverage map empty"}
    low = c < SKY_COVERAGE_LOW * cmed
    high = c >= SKY_COVERAGE_HIGH * cmed
    mid = ~low                           # everything the degraded class is not: the reference for a region whose border never reaches the well-covered class
    if low.mean() < 0.002 or high.mean() < 0.2:
        return None, {"cells": 0, "note": f"coverage uniform: {low.mean() * 100:.2f} % degraded, {high.mean() * 100:.0f} % reference"}
    from . import finish as _finish                                                              # late: finish imports this module for the render itself
    sky = _sky_witness(_finish.luminance(np.ascontiguousarray(a, np.float32)))                   # stars and filaments out; a pedestal may not disqualify its own pixels
    lab = cv2.connectedComponents(np.ascontiguousarray(low, np.uint8), connectivity=8)[1]
    regions = [int(L) for L in np.unique(lab) if L > 0]
    if not regions:
        return None, {"cells": 0, "note": "no degraded region big enough to measure"}
    din_ref = cv2.distanceTransform(np.ascontiguousarray(1 - high.astype(np.uint8), np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)      # every pixel's distance to the well-covered frame
    din_out = cv2.distanceTransform(np.ascontiguousarray(1 - mid.astype(np.uint8), np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)       # ... and to everything the degraded class is not
    v = np.zeros((ch, h, w), np.float64)
    offsets: list[float] = []
    verdicts: list[list] = []
    for L in regions:
        region = lab == L
        dout = cv2.distanceTransform(np.ascontiguousarray(1 - region.astype(np.uint8), np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)   # and to this region
        # A mosaic margin whose coverage falls off gradually leaves a band of pixels that are neither class, and then the region shares no border with the reference at all.
        # Measured on NGC 1499 (2026-10-30): its degraded top and bottom margins lay 20.2 and 17.5 px from the reference class with the band 16 px wide, every shell came back
        # empty, the verdict was "0 inside, 0 outside", and the +0.4 % of the sky those margins carried became the band across the export - the stretch's slope at the sky is
        # ~120.  Such a region is measured against the frame immediately outside its own class, which does share a border with it; the well-covered class stays the reference
        # whenever it reaches, because it is the one built by every frame.
        measured, why, used = None, None, None
        for cls, din, name in ((high, din_ref, "the well-covered frame"), (mid, din_out, "the frame outside its class")):
            steps, why = [], None
            for c_ in range(ch):
                step, se, n_in, n_out = _boundary_step(a[c_], region, cls, sky, din, dout, int(SKY_PEDESTAL_BAND))
                if step is None:
                    why = f"too few sky pixels at its border to compare ({n_in} inside, {n_out} outside)"
                    break
                if abs(step) > max(SKY_PEDESTAL_EPS * ref, SKY_PEDESTAL_MIN_SIGMA * se):
                    steps.append((c_, step))
            if why is None:
                measured, used = steps, name
                break
        if not measured:
            verdicts.append([L, why if used is None else f"level with {used} at its border"])
            continue
        for c_, step in measured:
            v[c_][region] = step
        off = max(abs(s) for _, s in measured)
        offsets.append(round(off, 7))
        verdicts.append([L, f"{off / ref * 100:.3f} % of the sky, across its border with {used}"])
    if not offsets:
        return None, {"cells": len(regions), "peak": 0.0, "cap": round(SKY_PEDESTAL_CAP * ref, 7), "degraded": round(float(low.mean()), 5),
                      "regions": regions, "offsets": [], "verdicts": verdicts, "note": "no degraded region differs from the frame around it"}
    cap = SKY_PEDESTAL_CAP * ref
    total = np.clip(v, -cap, cap)                                                                       # the class's own shape, unblurred: see the docstring
    peak = float(np.abs(total).max())
    if peak <= SKY_PEDESTAL_EPS * ref:
        return None, {"cells": len(regions), "peak": round(peak, 7), "offsets": offsets, "verdicts": verdicts}
    return total.astype(np.float32), {"cells": len(regions), "band_px": int(SKY_PEDESTAL_BAND), "peak": round(peak, 7), "cap": round(cap, 7),
                                      "degraded": round(float(low.mean()), 5), "regions": regions, "offsets": offsets, "verdicts": verdicts}



SKY_FLOOR_BIAS = abs(NormalDist().inv_cdf(0.5 * float(finish.BAND_SKY_FRACTION)))   # the median of the darkest f of a Gaussian sits this many sigma under its mean (0.755 at f = 0.45)


def sky_floor(values: np.ndarray, fraction: float = finish.BAND_SKY_FRACTION) -> tuple[float, float]:
    """The sky's level in a class of pixels, and the standard error of that estimate: the median of the darkest ``fraction`` of them - ``finish._sky_mask``'s definition of the
    sky, applied to whatever class is being measured - put back on the class's own mean by the bias that statistic carries.  The correction is what makes two classes of
    *unequal noise* comparable: an uncorrected floor reads low by ``SKY_FLOOR_BIAS`` x sigma, so a region whose sigma is 1.6 x the frame's - the low-coverage dithered rim of
    IC 1848, measured - is taken for 0.2 % of the sky darker than its surroundings without being darker at all, and a tool that levelled on the raw floor would pour light
    into exactly the rim it was meant to flatten (and darken an ordinary border by the same amount, which is 30 % of the display sky once the stretch's toe has multiplied it).
    """
    v = np.asarray(values, np.float64).ravel()
    sigma = _sigma(v)                                                                   # the class's own noise, robust to the nebulosity a border can carry
    k = max(1, int(fraction * v.size))
    sel = np.partition(v, k)[:k]
    return float(np.median(sel)) + SKY_FLOOR_BIAS * sigma, 1.253 * sigma / math.sqrt(sel.size)


def edge_ring(a: np.ndarray, cov: np.ndarray | None = None, ped: np.ndarray | None = None) -> tuple[np.ndarray | None, dict]:
    """Level the frame's own border against the interior: the boundary behaviour of every stage before the stretch, which the stretch's slope at the sky turns into a line.

    Measured on IC 1848 (2026-10-30), whose export showed a bright strip down the left edge and a dark one across the bottom while its starless is flat to the eye: the outer
    2 px of the frame sit +0.5 % of the sky above the interior - nothing at all in the linear, and 37.8 % of the display sky in the export, which is the step trace's own
    reading across the stretch (0.3 % before the Statistical Stretch, 37.8 % after it).  The stretch's toe multiplies a deviation at the sky by a hundred and fifty, so a
    border the eye cannot see in the linear is a frame line in the export, and nothing downstream of the stretch can take it out again.

    Each side is measured **line by line**: one median per row / column, read against the interior, for the outer ``EDGE_RING`` px.  Not a statistic over a band, however
    thin - the sky's floor is the mean of the darkest fraction of its sample, so anything occupying less than that fraction of the band is invisible in it, and the frame line
    is 2 px of an 8 px band on IC 1848: the first version of this tool measured nothing there and left the line standing.  A **median**, and not ``sky_floor``, because a line
    is a one-dimensional sample that a minority of pixels can contaminate from either side, and a floor is the statistic least able to resist it: measured on IC 1848
    (2026-10-30), its last row sits level with the interior by median (99.6-99.9 % of the sky) and 5.8 % of the sky under the interior by the floor statistic, and a tool that
    corrects that floor pours light into the border - the bright frame line down the export.  The median is also unbiased where the border is the noisier part of the frame,
    which a floor is not: a low-coverage rim reads low by a fixed multiple of its own sigma.  A border that crosses a star field or a galaxy needs no special rejection
    either: what is measured is the border's excess over the ring behind it, and subject that is not confined to the border cancels in that difference.

    Four things decide whether a border deviation is an artefact and may be taken.  **Ownership**: a ring the coverage map calls degraded *and* the provenance pedestal above
    has already levelled belongs to that tool - its offset has a cause in the frames that contributed, and this one would only be guessing over it.  **Confinement**: what is
    measured is the border's **excess over the ring behind it** - the median of the ring's inner quarter is the frame's own broad level at that side, sky or gradient, and not
    this tool's to take.  M 31 (2026-10-30), whose export shows a dark line across the bottom edge, measures -3.6 % of the sky at the border over a plateau of -0.9 % that
    reaches past the ring: the 2.7 % of excess above the plateau is the boundary's doing (a deconvolution reaching its reflect condition, the calibration pedestal, a gradient
    model extrapolated past its last sample), the plateau is the frame, and the plateau stays.  The excess in turn has to have died out by ``EDGE_RING_DEPTH`` px in - a
    deviation still standing there is sky, and IC 1848's edges carry 0.1-0.4 % that reaches past the ring, which is faint nebulosity, and flattening that is what
    ``sky_pedestal`` exists to refuse.  **Scale**: how big a deviation may be taken is decided by the *light* it accounts for (``EDGE_RING_BUDGET``) rather than by its
    amplitude, because amplitude cannot tell the two cases apart - 12.8 % of the sky in the outer line alone (IC 434's top) is the frame's own edge, while the same 8 % carried
    8 px deep is a nebula's edge and stays.  **Consent of the frame**: the boundary is a property of the frame, so a small deviation has to show on
    ``EDGE_RING_MIN_SIDES`` sides; a lone side is only believed from ``EDGE_RING_LONE`` of the sky up, far past anything sky itself leaves in its last pixels.

    The correction is the measured excess itself, so the artefact is taken out at the depth it has - one line on IC 434 and M 31, two on IC 1848 - rather than at a depth
    assumed here, and the sky the side was carrying underneath it is left exactly where the frames put it."""
    a = np.asarray(a, np.float64)
    ch, h, w = a.shape
    ring = int(min(EDGE_RING, h // 4, w // 4))
    if ring < 2 * EDGE_RING_DEPTH:
        return None, {"note": "frame too small for a ring"}
    keep = np.ones((h, w), bool)
    keep[:ring, :] = keep[h - ring:, :] = keep[:, :ring] = keep[:, w - ring:] = False
    if int(keep.sum()) < 4096:
        return None, {"note": "too little interior to measure the sky on"}
    # the interior's sky is measured exactly as every border line is - the same median on the same class of pixels - because the two are subtracted from each other, and two
    # statistics that answer different questions on the same pixels make an offset the border cannot be blamed for (see above: IC 1848's last row, level by median and 5.8 %
    # of the sky under the interior by floor)
    interior = np.array([float(np.median(a[c][keep])) for c in range(ch)])
    degraded = None
    if cov is not None:                                                              # a degraded rim is the provenance pedestal's to level, not this tool's
        c = np.asarray(cov, np.float64)
        if c.shape != (h, w):
            c = cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA)
        degraded = c < SKY_COVERAGE_LOW * float(np.median(c))

    def line_slice(side, d):                                                          # 0 top, 1 bottom, 2 left, 3 right; the line d px in from that border
        if side == 0:
            return slice(d, d + 1), slice(None)
        if side == 1:
            return slice(h - 1 - d, h - d), slice(None)
        return (slice(None), slice(w - 1 - d, w - d)) if side == 3 else (slice(None), slice(d, d + 1))

    num, den, table, qual, cand, strength = np.zeros((ch, h, w)), np.zeros((ch, h, w)), [], set(), [], np.zeros(4)
    ped_mag = np.abs(np.asarray(ped, np.float64)).max(0) if ped is not None else None
    for side in range(4):
        sl0 = line_slice(side, 0)
        if degraded is not None and ped_mag is not None and degraded[sl0].mean() > 0.5 and ped_mag[sl0].max() > EDGE_RING_MIN * float(interior.mean()):
            table.append((side, -1, "degraded and already levelled by the provenance pedestal"))
            continue                                                                # that side's offset has a cause in the frames that contributed and the tool above owns it
        prof = np.zeros((ch, ring))
        for d in range(ring):
            sl = line_slice(side, d)
            for c in range(ch):
                v = a[c][sl]
                if v.size < 256:
                    continue
                se = 1.253 * _sigma(v) / math.sqrt(v.size)                              # the standard error of the line's own median, which is what the gate below is stated in
                x = float(np.median(v)) - interior[c]
                prof[c, d] = x if abs(x) > EDGE_RING_MIN_SIGMA * se else 0.0
        for c in range(ch):
            behind = float(np.median(prof[c, ring - ring // 4:]))                  # what the ring behind the border itself carries - the frame's own broad level at this side,
                                                                                    # which is sky or a gradient and is not this tool's to take: M 31's bottom line (2026-10-30) sits
                                                                                    # -3.6 % of the sky under the interior over a plateau of -0.9 % that reaches past the ring, and only
                                                                                    # the 2.7 % of excess above that plateau is the boundary's doing
            x = prof[c] - behind
            edge = abs(x[0])                                                       # the border line's excess is the whole candidate: a boundary artefact lives there
            control = float(np.abs(x[EDGE_RING_DEPTH:]).max())                     # and it has to have died out by EDGE_RING_DEPTH px in, or the deviation is sky
            light = float(np.abs(x).sum())                                         # the light the correction would take back along a line through the ring
            why = None
            if edge < EDGE_RING_MIN * interior[c]:
                continue                                                           # the border sits level with the ring behind it: nothing here to take
            if edge > EDGE_RING_MAX * interior[c]:
                why = "missing data, not a boundary artefact"                      # the line holds no sky to level; supplying one is not this tool's job
            elif light > EDGE_RING_BUDGET * interior[c]:
                why = "too much of the sky's light to be a boundary artefact"      # this large at this depth is a nebula's edge, and a nebula's edge is not an artefact
            elif control > EDGE_RING_SHARP * edge:
                why = "the deviation goes on behind the border"                    # that is sky, or a gradient, and sky stays
            if why is not None:
                table.append((side, c, round(edge / interior[c], 5), round(light / interior[c], 3), why))
                continue
            cand.append((side, c, x))                                              # the excess only: the broad level the side was carrying stays exactly where it was
            qual.add(side)
            strength[side] = max(strength[side], edge / interior[c])              # how far the side's strongest channel stands above the ring behind it
    lone = float(strength[next(iter(qual))]) if len(qual) == 1 else 0.0
    if not qual or (len(qual) < EDGE_RING_MIN_SIDES and lone < EDGE_RING_LONE):
        # the boundary is a property of the frame, so a deviation the size of faint sky needs a second side to agree with it before anything is taken
        return None, {"ring": ring, "sides": table, "qualifying": sorted(qual), "lone": round(lone, 5),
                      "note": "the border's profile is not on enough sides to be the frame's own"}
    for side, c, prof in cand:
        for d in range(ring):
            if prof[d] == 0.0:
                continue
            sl = line_slice(side, d)
            num[c][sl] += prof[d]                                                   # a corner belongs to two sides: their offsets are averaged, never added
            den[c][sl] += 1.0
    if not den.any():
        return None, {"ring": ring, "sides": table, "note": "the border carries no artefact"}
    field = num / np.maximum(den, 1.0)
    return field.astype(np.float32), {"ring": ring, "sides": table, "peak": round(float(np.abs(field).max()), 7)}


SKY_RIM_BAND = 0.125      # the outer eighth of the short side, capped at 512 px, is as deep as a stage's boundary behaviour reaches.  Measured on the frames that still show a band:
                          # NGC 1499's right border sits -2.04 % of the sky at its line and -0.23 % by 256 px, IC 434's left +1.10 % at its line and -0.32 % by 128 px.  A deviation
                          # still standing deeper than this band is not a boundary residue and this tool has no business taking it (SKY_RIM_BEHIND)
SKY_RIM_DEPTHS = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
                          # how deep into the border the sky's floor is read.  The *profile* is what tells the three things a border can be apart: a step that dies within tens
                          # of pixels (LBN 429's bottom, +0.23 % of the sky at 10 px and gone by 40, which the export showed as +22 %), a step that reaches a few hundred
                          # (IC 434's left, +1.02 % above what lies 128 px behind it, shown as +25 %), and a deviation that never dies (M 31's left, which is a galaxy and stays
                          # exactly where it is)
SKY_RIM_MIN = 0.0007      # the smallest step worth taking back: 0.07 % of the sky.  The frames that already render clean deviate at their border by no more than ~5 % of the
                          # display sky, and the stretch's slope at the sky here is 75-120, so a step this small cannot reach the export; the bands that started this measure
                          # 0.2-1.0 %, which is why the floor used to sit at 0.35 % was itself the bug - it refused exactly what the user was seeing
SKY_RIM_MIN_DEPTH = 3     # a deviation standing on no more than the outer two pixels is a frame *line*, not a band: it has a different cause (a stage writing its boundary
                          # condition into the edge) and its own instrument, ``edge_ring``, which is allowed to touch two pixels and no more.  A step this shallow is left to the
                          # ring rather than taken as a band's worth of sky; the residues this tool exists for measure 30-150 px deep on the frames that have them
SKY_RIM_CAP = 0.04        # the ceiling on what one side may hand back: no stage displaces the sky's floor at a border by more than 4 % of the sky (the largest measured anywhere
                          # here is 2.04 %), and a deviation bigger than that is a gradient the model never got hold of - the model's to fix, not this tool's to bury
SKY_RIM_SIGMA = 3.0       # a floor has to clear 3 of its own standard errors before it is called a deviation at all
SKY_RIM_KEEP = 0.55       # and the witness has to keep on at least this *fraction of what it keeps in the middle of the frame* on every line it measures, counted about that
                          # line's own sky: a border whose extra light is subject rather than sky holds two populations where the sky holds one, and spreads wider about its
                          # own centre than the frame's sky does.  Measured here: a sky displaced 2 % reads 0.85-0.88 of the middle's share and a border under 45 % filaments
                          # still reads 0.63, while a 2 %-of-sky wing over 40 % of the line reads 0.35-0.39 - the two are not close
SKY_RIM_TOL = 1.5         # how far from a line's own sky a kept pixel may sit to count: the sky's mass interval at the frame's noise, so the comparison is with the frame's
                          # own spread and not with an absolute width that no two frames share
SKY_RIM_MIN_SKY = 256     # the fewest sky pixels a line, or the strip behind the band, may offer for its floor to be read at all: the same number the ring's own statistic
                          # asks of a border, and on the smallest frames here (a line of ~1100 witness pixels) still four times it.  Below this the line is refused, not guessed
SKY_RIM_BEHIND = 2        # the reference is the sky's floor in this same side's strip, one band width out to two - not the middle of the frame.  Against the middle, a border
                          # under a frame-scale gradient looks like whatever the gradient makes it: every real frame here sits 0.4-1.9 % below its own interior at the border on
                          # *all four sides*, which is the gradient, not the border, and a tool that levelled to the middle would draw the sky flat across the whole frame.
                          # Against the strip behind it the gradient cancels and only the step the border itself made is left.  The middle keeps one job: saying whether the
                          # deviation died out inside the band, which no frame-scale gradient does


def sky_rim(a: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """Level the sky's floor in the frame's outer band against the strip of sky lying just behind it: the residue the stages before the stretch leave at the border, taken back before the slope at the sky turns it into a band.

    Three tools stand between here and the export and none of them can do this.  ``edge_ring`` may touch two pixels and no more, because a wider claim about a border is a
    claim about the sky; ``sky_pedestal`` levels a region only where its provenance class differs from the class around it, and a gradient model's residue carries no
    provenance; ``border_debt`` may take back only what the background step itself changed, and by construction leaves whatever the frames brought - which is exactly what a
    model that removed 78 % of a border gradient (NGC 1499: -7.83 % entering, -1.86 % leaving) did not remove.  The residue is a processing artifact all the same: the sky does
    not step 2 % at the edge of a frame, and the stretch's slope at the sky (75-120) multiplies those 2 % into the -60 % band the export showed.

    **The subject is never in this measurement, and that is not an assumption.**  What is compared is the *sky's floor* on each line - ``sky_floor`` over the pixels
    ``_sky_witness`` kept, stars and filaments out - and not the line's median: the median of a border line on a nebula-dominated frame measures where the subject is, which is
    how a tool comes to confuse the two and pour light into a rim.  Because a level error is removed by a constant offset under a taper, every pixel under the band moves by the
    same amount at its own depth, so the subject's contrast against its own sky is preserved exactly; and where the subject crowds the sky out of a line there are not
    ``SKY_RIM_MIN_SKY`` pixels the witness will accept as sky left on it, and the side is refused rather than guessed at - the rim that started this offered 194 accepted
    pixels out of 5658.  M 31 is the standing proof of both halves: its left border, where the galaxy lies, measures no larger a floor offset than its clean sky sides do, and
    this tool leaves all four of them alone.

    A side is taken only when all of this holds, in every channel or none:

    * every sampled line offers at least ``SKY_RIM_MIN_SKY`` pixels the witness accepts as sky - a line the subject has filled has no floor to read, and reading one there is
      guessing - and the pixels it does accept on that line sit as tightly about the line's own sky as the middle's do about theirs (``SKY_RIM_KEEP``, ``SKY_RIM_TOL``).  Light
      lying across a border that the sky did not make cannot be told from a displaced sky by its level alone, and this is what tells them apart: the one adds a second
      population, the other moves the first;
    * the deviation is a step and not a ramp.  It stands ``SKY_RIM_MIN`` of the sky and ``SKY_RIM_SIGMA`` standard errors clear of the strip behind the band from the very edge
      in, in one direction, with no gap in the depths it stands at, and it has died out by the deepest line read inside the band.  One still standing there is a frame-scale
      gradient, and levelling a border against a gradient would draw the sky flat across the whole frame: that is the gradient model's to remove, and sky stays;
    * it is under ``SKY_RIM_CAP`` of the sky.

    What is taken is the measured profile itself - the offset read at each depth, joined between the samples and faded to nothing with a cos^2 over as much room again beyond the
    depth where it last stood - so the correction ends where the deviation ends and cannot leave a step of its own, rather than reaching the full band width whether the
    deviation is 20 px deep or 400.  It is faded out along the border over a quarter of a band at each end, so the corners - where two sides overlap - average the two sides'
    offsets instead of adding them (the ring's discipline).

    Returns ``(field, info)``; subtract the field.  ``info["sides"]`` carries every side and channel this tool looked at: how far its floor stood from the strip behind the
    band, as a percentage of that sky, and either what was taken or why nothing was.
    """
    a = np.asarray(a, np.float32)
    ch, h, w = a.shape
    lum = finish.luminance(a)
    keep = _sky_witness(lum)
    band = max(64, min(512, int(round(min(h, w) * SKY_RIM_BAND))))
    cut = max(16, int(0.05 * min(h, w)))                                                              # the corners are where two sides' ramps overlap: neither side's evidence
    m = max(140, min(h, w) // 8)
    mid = np.zeros((h, w), bool)
    mid[m:h - m, m:w - m] = True
    ref_sel = keep & mid
    if int(ref_sel.sum()) < 4 * SKY_RIM_MIN_SKY:
        return None, {"band_px": band, "sides": [], "note": "the middle of the frame offers no sky to measure a border against"}
    ref, ref_se = sky_floor(a[:, ref_sel].reshape(-1))
    ref_lum = float(np.median(lum[ref_sel]))                                                         # the middle's sky, on the plane every side is compared on
    sig = max(float(_sigma(lum[mid])), 1e-5 * max(1.0, abs(ref_lum)))                                # the frame's sky spread: stars and nebula are outliers to it, and a frame
                                                                                                      # with no noise at all has nothing for a subject to hide behind
    ref_share = float(np.mean(np.abs(lum[ref_sel] - ref_lum) <= SKY_RIM_TOL * sig))                  # how tightly this frame's sky sits about its own level: the yardstick,
                                                                                                      # measured on the witness's pixels exactly as a border line is
    ref = float(ref)
    if ref <= 0.0:
        return None, {"band_px": band, "sides": [], "note": "the frame's own sky is not a level anything can be measured against"}
    depths = tuple(d for d in SKY_RIM_DEPTHS if d < band)                                             # the profile is read across the band, and the band is never shallower

    def pixels(side, d0, d1):
        """The pixels from d0 to d1 in from that border, with the ends of each line cut back, as one sample per channel with the witness's share of them alongside."""
        if side == 0:
            v, s = a[:, d0:d1, cut:w - cut], keep[d0:d1, cut:w - cut]
        elif side == 1:
            v, s = a[:, h - d1:h - d0, cut:w - cut], keep[h - d1:h - d0, cut:w - cut]
        elif side == 2:
            v, s = a[:, cut:h - cut, d0:d1], keep[cut:h - cut, d0:d1]
        else:
            v, s = a[:, cut:h - cut, w - d1:w - d0], keep[cut:h - cut, w - d1:w - d0]
        return v.reshape(ch, -1), s.reshape(-1)

    def sky_share(side, d):
        """How tightly the sky the witness accepts on the line d in from that border sits about its own level, against the same figure for the middle of the frame.

        A level error moves the sky; a subject lying across the border adds a second population to it.  Both raise the line's floor and, once the line's own centre is used,
        both sit at the level the tool would be told to remove - so the level cannot separate them and the shape can.  Counted over the pixels the witness accepted, on both
        sides of the comparison, because that is what keeps the two cases apart: filaments are rejected there and cost the line nothing, while a smooth plateau of a couple of
        percent of the sky is accepted along with the sky and halves the line's share.  A border whose sky has gone entirely is not this gate's to catch - its centre is the
        subject's own level, its share looks perfect, and ``SKY_RIM_CAP`` is what refuses it."""
        if side == 0:
            sl = (slice(d, d + 1), slice(cut, w - cut))
        elif side == 1:
            sl = (slice(h - 1 - d, h - d), slice(cut, w - cut))
        elif side == 2:
            sl = (slice(cut, h - cut), slice(d, d + 1))
        else:
            sl = (slice(cut, h - cut), slice(w - 1 - d, w - d))
        s = lum[sl][keep[sl]]
        if s.size < SKY_RIM_MIN_SKY:
            return 0.0                                                                                 # no sky accepted here at all is not a line to measure a level on
        return float(np.mean(np.abs(s - float(np.median(s))) <= SKY_RIM_TOL * sig))

    def place(side, d):
        """Where the line d in from that border lives - its whole length, because a corner belongs to two sides and their offsets have to be averaged there, never added.  The
        fade along the border (``ramp``) is what keeps each side's claim to the length it actually measured."""
        if side == 0:
            return slice(d, d + 1), slice(None)
        if side == 1:
            return slice(h - 1 - d, h - d), slice(None)
        if side == 2:
            return slice(None), slice(d, d + 1)
        return slice(None), slice(w - 1 - d, w - d)

    num = np.zeros_like(a, np.float64)
    den = np.zeros_like(a, np.float64)
    table = []
    taken = []
    for side in range(4):
        n = w if side < 2 else h
        tail = max(1, band // 4)
        ramp = np.ones(n, np.float64)
        fade = 0.5 * (1.0 + np.cos(np.pi * np.arange(tail) / tail))                                   # the corner belongs to two sides: fade out before it is reached
        ramp[:tail] = fade
        ramp[n - tail:] = fade[::-1]
        vb, sb = pixels(side, band, SKY_RIM_BEHIND * band)
        if int(sb.sum()) < SKY_RIM_MIN_SKY:
            for c in range(ch):
                table.append((side, c, None, "too little sky behind this border to know what it should be"))
            continue
        back, back_se = [], []
        for c in range(ch):
            fb, seb = sky_floor(vb[c][sb])
            back.append(float(fb))
            back_se.append(float(seb))
        if min(back) <= 0.0:
            for c in range(ch):
                table.append((side, c, None, "the sky behind this border is not a level anything can be measured against"))
            continue
        outside, shallow = [], []                                                       # the channels that carry a band, and the ones whose own profile is only a frame line
        for c in range(ch):
            offs, ses, cov = [], [], 1.0
            for d in depths:
                v, s = pixels(side, d, d + 1)
                cov = min(cov, sky_share(side, d))
                if int(s.sum()) < SKY_RIM_MIN_SKY:
                    offs = None
                    break
                fl, se = sky_floor(v[c][s])
                offs.append(float(fl) - back[c])                                                       # the border's own step, with the gradient behind it cancelled
                ses.append(float(np.hypot(se, back_se[c])))
            if offs is None:
                table.append((side, c, None, "the witness keeps too little of this side's lines to measure it"))
                continue
            sign = 1.0 if offs[0] >= 0.0 else -1.0                                                    # the border deviates one way or it does not deviate: a profile that turns
                                                                                                      # round inside the band is not the step the edge was set to
            stood = [o * sign > max(SKY_RIM_MIN * back[c], SKY_RIM_SIGMA * se) for o, se in zip(offs, ses)]
            how = round(offs[0] / back[c], 5)
            if cov < SKY_RIM_KEEP * ref_share:
                table.append((side, c, how, "the subject lies across this side's lines; its light is not the sky's level"))
                continue
            if not stood[0]:
                table.append((side, c, how, "level with the sky behind the border"))
                continue
            last = 0
            while last + 1 < len(stood) and stood[last + 1]:
                last += 1
            if depths[last] < SKY_RIM_MIN_DEPTH:
                shallow.append(c)
                table.append((side, c, how, "a deviation this shallow is a frame line, and the ring's to take"))
                continue
            # What has to kill the side is a second step, never a noisy sample of the first.  The first version refused on
            # *any* standing depth beyond the run, and C 20's right border paid for it: +0.2 % of the sky lying uniformly
            # 48 px in, vetoed by one sample at 96 px that stood a hair over the bar with the depths either side of it
            # flat, so the untaken offset went through the stretch (slope 75-120 at the sky) and came out the export's
            # bright edge.  A deviation that really returns behind the border - the nebula, the galaxy, the gradient this
            # rule exists to leave alone - returns over SKY_RIM_REPEAT depths in a row and carries SKY_RIM_ECHO of the
            # border's own height; a single line above the bar neither does nor needs to.
            height = max(abs(o) for o in offs[:last + 1])
            echo = 0
            for i in range(last + 1, len(stood)):
                echo = echo + 1 if stood[i] and abs(offs[i]) >= SKY_RIM_ECHO * height else 0
                if echo >= SKY_RIM_REPEAT:
                    break
            if echo >= SKY_RIM_REPEAT:
                table.append((side, c, how, "the deviation is not a step: a second one of its own height returns behind the border"))
                continue
            if last == len(depths) - 1:
                table.append((side, c, how, "the gradient reaches past the band: that is the model's to remove, and sky stays"))
                continue
            amplitude = height                                                                # the same measurement the returning deviation was judged against above
            if amplitude > SKY_RIM_CAP * back[c]:
                table.append((side, c, how, "more than a stage can have displaced at a border"))
                continue
            outside.append((side, c, offs, last, amplitude))
        if not outside:
            continue
        # The border is one object measured three times, and handing back the channels that passed while leaving the one
        # that did not paints the border in the colour that was left out - so a side is taken whole or not at all.  What
        # is *not* a disagreement is a channel whose own profile is nothing but a frame line: two pixels deep, too
        # shallow for this tool and ``edge_ring``'s own claim, which it declined.  Such a channel carries no evidence
        # about a band either way, and refusing on it is how C 20's right border kept +0.3 % of its sky over 48 px - the
        # R channel held hostage by G and B carrying a two-pixel line.  A channel that was read and found level, or found
        # not to be a step at all, is a disagreement, and the side stays whole.
        channels = {o[1] for o in outside}
        if any(c not in channels and c not in shallow for c in range(ch)):
            reason = "a channel of this side is refused, so the side stays as it is"
            for side_, c, _, _, amplitude in outside:
                table.append((side_, c, round(amplitude / back[c], 5), reason))
            continue
        for side_, c, offs, last, amplitude in outside:
            table.append((side_, c, round(amplitude / back[c], 5), "taken"))
            ds = list(depths[:last + 1])
            prof = np.interp(np.arange(band), ds, offs[:last + 1])                                     # the step as it was measured, joined between the depths it was read at
            end = min(band - 1, ds[-1] + max(4, ds[-1]))                                               # as much room again as the step is deep to fade out in
            taper = np.ones(band, np.float64)
            inner = np.arange(ds[-1] + 1, end + 1)
            taper[inner] = 0.5 * (1.0 + np.cos(np.pi * (inner - ds[-1]) / max(1, end - ds[-1])))      # to nothing, with a zero slope, where the deviation has gone
            prof = prof * taper
            along = ramp[:, None] if side_ >= 2 else ramp                                             # a left/right strip is (h, 1): the fade along the border runs down the rows
                                                                                                      # there, not across them
            for d in range(band):
                t = float(prof[d])
                if t == 0.0:
                    continue
                rows, cols = place(side_, d)
                num[c][rows, cols] += t * along
                den[c][rows, cols] += along
                taken.append((side_, c))
    if not den.any():
        return None, {"band_px": band, "depths": list(depths), "sides": table, "note": "no border differs from the sky behind it"}
    field = num / np.maximum(den, 1e-9)
    return field.astype(np.float32), {"band_px": band, "depths": list(depths), "sides": table, "taken": taken,
                                      "peak": round(float(np.abs(field).max()), 7), "sky": round(ref, 7)}

SKY_TINT_SAMPLES = 20000      # the fewest sky pixels (subject excluded) a plane may be fitted on: below it the plane is the noise of one part of the frame
SKY_TINT_BUFFER = 9           # px of sky thrown away around the subject, because the subject's edge is not sky either.  The subject is ``nebula_mask``'s own - the same
                              # definition ``subject_coverage`` uses to refuse the colour-gradient step - and not a high-pass threshold of mine: my first version excluded
                              # anything standing 3 sigma above an 8 px high-pass of the *linear* light and dilated it 15 px, which on C 20's starless left 10.9k of 2.3M sky
                              # pixels - a noise-relative threshold on linear data flags a sprinkling, and 15 px of dilation around a sprinkling is the whole frame
SKY_TINT_MIN_SPREAD = 1.0     # the gate's own units: below this the quarters already agree (a delivery is failed for 1.5), so the frame is left strictly alone - a tool that
                              # touched clean frames would be processing, not correcting
SKY_TINT_CAP = 0.10           # the largest peak-to-peak gain allowed in one channel: past 10 % the colour is the sky's own (dual-band, dust, an unbalanced white) and a plane
                              # fitted on the sky has no business touching it - that is a calibration question, not a residue
SKY_TINT_GAIN = 1.2           # the correction ships only when it leaves the gate's own quadrant spread this times smaller - a comparison, not a threshold, in the spirit of
                              # ``STRETCH_CLIP_GAIN``: what is refused is a fit that does not measurably help the thing the delivery is judged on
SKY_TINT_CLIP = 3.0           # sigma clips of the plane fit: the sky is the bulk of the frame and the subject must not drag the plane, so the sample is trimmed from both tails


def sky_quadrant_spread(a: np.ndarray) -> tuple[float, dict]:
    """What ``gates.quadrant_spread`` will find, read on the frame's own display transform: the worse of the a* and b* spreads between the four quarters of the sky, with the
    measurement behind it.  This is the number a delivery is failed for - read here, where it can still be corrected, rather than in the sidecar after the export.  It is taken
    through ``autostretch_display`` (the transform the chain's own step pictures go through) rather than through the whole downstream chain: the delivered gate is the final
    judge, and the two numbers this records are what the tool saw.
    """
    disp = autostretch_display(a)
    sm = gates.sky_mask(finish.luminance(disp))
    if sm.sum() < SKY_TINT_SAMPLES:
        return 0.0, {"refused": "no sky to measure on the display transform"}
    qs = gates.quadrant_spread(gates._lab(disp), sm)
    return max(float(qs["spread_a"]), float(qs["spread_b"])), qs


def sky_tint(a: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """Take out the sky's slowly varying **channel-to-channel** tilt: what the refused ``colour_gradient`` step would have done, fitted on sky alone.

    Why it exists (C 20, 2026-11-01, one fixed display transform, the sky taken with the subject excluded, measured on the session's own checkpoints): the registered stack's
    quadrant tints are neutral - ``gates.quadrant_spread`` reads b* 0.88 - and they grow through the stages after it: restoration 1.55, linear 1.98, starless 3.41, with the
    bottom-right quadrant's b* at 4.88 while the other three sit at 1.5-2.0.  The delivery then fails ``sky_chroma_quadrants`` on a cast that was never in the data.  Nothing
    between the frame and the export reaches it: ``colour_gradient`` is refused above ``SKY_STEPS_MAX_COVERAGE`` (this frame's subject covers 34 %), ``neutralise``'s offsets
    are per channel but *global*, and ``sky_pedestal``/``sky_rim`` are spatial but *achromatic*.

    Why the fit may only look at sky, and only at degree 1: the refusal it stands in for was written for a real hazard - on C 20 (2026-10-15) Siril's degree-2 model "took the
    North America Nebula's own red".  Running that step on this frame (``output/.ab12/subsky_probe.py``) takes out a model 10x the size of the sky's actual tilt (2.35 % and
    1.96 % of the sky in R and G where the sky only tilts 0.25 % and 0.43 %), anti-correlated with it in R (corr -0.23), and the quadrant tints get **worse**: a 1.89 -> 3.44.
    So the replacement is never shown the subject, and stays linear because degree 2 and 3 are what ate the nebula.

    Three things this measures rather than assumes, each learned from it being wrong on these same pixels:

    * it fits in the **log**, because colour is a channel *ratio* and the stretch carries ratios to the picture; the first version fitted absolute differences, and on the
      starless that raised the gate's a* spread from 1.89 to 3.59 while its own metric said it had halved the tilt;
    * it **centres** each channel's field on its own sky, so the frame keeps the colour it arrived with (``neutralise``'s offsets own the overall balance, one step later, on
      purpose); uncentred, this pulled C 20's registered stack from b* -2.9 to about 0 - the sky's colour removed, not a residue;
    * it **adopts on the gate's own reading** (``sky_quadrant_spread``) rather than on a proxy of it: measured on the real checkpoints, a metric of percent-of-sky tints said
      the correction improved the restoration frame 1.4x while the gate's number said the opposite, and only the gate's number agreed with what the export does with it.

    The gain is applied to the whole frame rather than under a sky mask: a plane cannot draw an edge, whereas a smooth field multiplied by a blurred mask does - at the
    nebula's boundary, where the stretch's slope at the sky is a hundred - which is the artifact class this file's other tools exist to remove.  Returns the field to
    *multiply* by, or None with the reason it refused.
    """
    lum = finish.luminance(a)
    before, qs0 = sky_quadrant_spread(a)
    if before <= SKY_TINT_MIN_SPREAD:
        return None, {"refused": "the quarters' sky tints already agree", "spread": round(before, 2),
                      "limit": gates.LIMITS["sky_chroma_quadrants"]}
    sm = finish._sky_mask(lum, edge=finish.EDGE_CROP)
    if sm.sum() < SKY_TINT_SAMPLES:
        return None, {"refused": f"the frame has no sky to measure ({int(sm.sum())} pixels)"}
    subject = nebula_mask(lum, _hist_stats(lum)["mode"])[0]                          # the project's own subject definition - the one ``subject_coverage`` refuses the
    structure = cv2.dilate(subject.astype(np.uint8), np.ones((SKY_TINT_BUFFER, SKY_TINT_BUFFER), np.uint8)) > 0   # gradient on - buffered, because its edge is not sky
    samples = sm & ~structure
    if samples.sum() < SKY_TINT_SAMPLES:
        return None, {"refused": f"only {int(samples.sum())} sky pixels survive the subject", "sky": round(float(sm.mean()), 3)}
    ys, xs = np.nonzero(samples)
    stride = max(1, len(ys) // 60000)                                                # a plane has three parameters; 60 k samples is twenty thousand times over-determined,
                                                                                     # and this fit's cost is paid by every preview (it measured 7 s a frame at 200 k)
    yi, xi = ys[::stride], xs[::stride]
    A = np.stack([np.ones(len(yi)), yi.astype(np.float64), xi.astype(np.float64)], -1)
    yy, xx = np.arange(lum.shape[0], dtype=np.float64), np.arange(lum.shape[1], dtype=np.float64)   # evaluated by row/column below, not as two full grids
    loga = np.log(np.maximum(a, 1e-12))                                              # colour is a ratio: fit the channels' log levels, and the difference of those
    planes = np.empty((3,) + lum.shape, np.float64)
    for c in range(3):
        v = loga[c][yi, xi].astype(np.float64)
        keep = np.ones(v.shape, bool)
        coef = np.zeros(3)
        for _ in range(3):
            coef = np.linalg.lstsq(A[keep], v[keep], rcond=None)[0]
            r = v - A @ coef
            keep = np.abs(r - np.median(r)) <= SKY_TINT_CLIP * _sigma(r)
        planes[c] = coef[0] + coef[1] * yy[:, None] + coef[2] * xx[None, :]
    field = planes - planes.mean(0)                                                   # the channel-to-channel part: the mean of the three is added back, so the frame's
    for c in range(3):                                                                # luminance gradient is left exactly where the frames put it
        field[c] -= float(np.mean(field[c][samples]))                                 # ... and only its SPATIAL part: each channel keeps the sky level it arrived with
    gain = np.exp(-field).astype(np.float32)
    ptp = [float(np.ptp(gain[c])) for c in range(3)]
    if max(ptp) < 1e-4:
        return None, {"refused": "no tilt to take", "spread": round(before, 2)}
    if max(ptp) > SKY_TINT_CAP:
        return None, {"refused": f"a {100.0 * max(ptp):.1f} % tilt in one channel is the sky's own colour, not a residue",
                      "gain_ptp_pct": [round(100.0 * p, 2) for p in ptp]}
    after, qs1 = sky_quadrant_spread(a * gain)
    if after * SKY_TINT_GAIN >= before:
        return None, {"refused": f"the fit leaves the gate's quadrant spread at {after:.2f} (was {before:.2f})",
                      "gain_ptp_pct": [round(100.0 * p, 2) for p in ptp],
                      "spread_before": round(before, 2), "spread_after": round(after, 2)}
    return gain, {"samples": int(samples.sum()), "sky": round(float(sm.mean()), 3),
                  "gain_ptp_pct": [round(100.0 * p, 2) for p in ptp],
                  "spread_before": round(before, 2), "spread_after": round(after, 2),
                  "medians_b_before": [round(float(v), 2) for v in qs0.get("median_b", [])],
                  "medians_b_after": [round(float(v), 2) for v in qs1.get("median_b", [])]}


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


class Finisher:
    """One run of the chain.  ``external`` = the AI steps (NoiseXTerminator / GraXpert on the stretched image) run; the preview passes False."""

    def __init__(self, work: str, header, p: dict, dual: bool, *, external: bool = True, progress=None, checkpoint=None, tick=None, device: str = "auto",
                 steps=None, keep_pictures: bool = True, coverage: np.ndarray | None = None):
        self.work, self.header, self.p, self.dual, self.external = work, header, p, dual, external
        self.progress, self.checkpoint, self.tick, self.device, self.steps = progress, checkpoint, tick, device, steps
        self.coverage = coverage                            # the stacking coverage map (frames per pixel), or None: see sky_pedestal
        self.denoiser = None                                 # (tool, run(amount, array), ceiling, requested) once denoise() has run its tool; the closing
                                                             # step (run -> _close_sky_noise) re-runs that same tool on the starless the contrast steps left
        self.denoiser_refused = None                               # {"tool", "sky_chroma", "limit"} when _keep_denoise refused the step: the closing
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
            rec["levels"] = level_stats(arr)
        self.records.append(rec)
        if self.steps is not None:
            self.steps.add(step, tool, status, seconds, note=note, commands=rec["commands"], **(params or {}))
        return rec

    # ---- 0a. colour gradient (linear) ----------------------------------------------------------------------------------------
    def colour_gradient(self, base: np.ndarray) -> np.ndarray:
        """The channel-to-channel gradient the restore stage's background extraction leaves (R - G varies by +-8e-5 across IC 1805's starless layer, 0.4 % of the sky; the stretch
        multiplies it by hundreds): Siril ``subsky 2 -samples=20 -tolerance=1.0`` (the command's polynomial degree 2 / default samples and tolerance, pro-workflow.md A2) on each
        channel, with the MEAN of the three channels' models added back so that only the colour part of the gradient is removed and the brightness of the faint nebula is left alone."""
        deg = 2 if self.dual else 3          # broadband colour gradients (an amber / green glow in NGC 1499's corners) need one more order; Siril allows 1-4
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
        (Siril's SPCC ``-bgtol`` background neutralisation / PixInsight BackgroundNeutralization, offsets only).  The stretch multiplies the linear differences by its slope at the sky
        (hundreds on a stack whose nebula is 1-5 % above the sky): the 0.3 % channel-to-channel differences BXT, NXT and SXT leave after SPCC (R 0.018865, G 0.018911, B 0.018919 on
        IC 1805's starless layer) would otherwise become a 0.1 display-unit colour cast of the sky.  The same slope turns a pedestal that varies across the frame (the low-coverage
        dithered rim, measured 0.3 % of the sky over ~300 px in IC 1848, 40-60 % of the display sky after the stretch) into a band, so the sky's slowly varying pedestal is modelled
        and removed before it, measured at the border where the provenance classes meet - see ``sky_pedestal``.  The frame's own border is levelled in the same step - see
        ``edge_ring``: the outer 2 px of IC 1848 sit +0.5 % of the sky above the interior, which is nothing in the linear and a bright line down the edge of the export once
        the stretch has multiplied it.  And so is the band behind that line: ``sky_rim`` compares the *sky's floor* on each border line with the sky's floor in the middle of the
        frame, which is the only comparison that cannot mistake a nebula for an artifact, and takes back the residue the gradient model left there while it is still 1-2 % of the
        sky rather than 50 % of the display sky."""
        lum = finish.luminance(base)
        sm = finish._sky_mask(lum, edge=finish.EDGE_CROP)
        ref0 = max(float(np.median(base[c][sm])) for c in range(3))
        ped, ped_info = sky_pedestal(base, ref0, self.coverage)
        if ped is not None:                                              # the spatial half: one offset per channel cannot reach a pedestal that varies across the frame
            base = np.maximum(base - ped, 0.0).astype(np.float32)
        ring, ring_info = edge_ring(base, self.coverage, ped)                      # and the frame's own border: the +0.5 % of the sky over its outer 2 px is nothing in the
        if ring is not None:                                             # linear, and a bright line down the edge of the export once the stretch's toe has multiplied it
            base = np.maximum(base - ring, 0.0).astype(np.float32)
        rim, rim_info = sky_rim(base)                                                    # and the band behind the ring's line: what the gradient model left in the outer eighth of
        if rim is not None:                                                              # the frame, 1-2 % of the sky in the linear and a band across the export after the toe
            base = np.maximum(base - rim, 0.0).astype(np.float32)
        gain, tint_info = sky_tint(base)                                                       # and the colour half of the same problem: quarters of the sky carrying different
        if gain is not None:                                      # channel-to-channel levels, which no global offset can reach and the stretch's slope delivers as a cast
            base = (base * gain).astype(np.float32)
        if ped is not None or ring is not None or rim is not None or gain is not None:
            lum = finish.luminance(base)
            sm = finish._sky_mask(lum, edge=finish.EDGE_CROP)
        med = [float(np.median(base[c][sm])) for c in range(3)]
        ref = max(med)
        off = [ref - m for m in med]
        t0 = time.time()
        if max(off) < 1e-7 and ped is None and ring is None and rim is None and gain is None:
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
        self.record("background neutralisation (linear)",
                    ("sky tint + " if gain is not None else "") + ("provenance sky pedestal + " if ped is not None else "") + ("frame ring + " if ring is not None else "") + ("border sky rim + " if rim is not None else "")
                    + "siril split / pm (offsets) / rgbcomp", out,
                    commands=lines, seconds=time.time() - t0, linear=True,
                    params={"sky_median_before": [round(m, 7) for m in med], "offsets": [round(o, 8) for o in off], "pedestal": ped_info, "ring": ring_info, "rim": rim_info, "tint": tint_info},
                    source="Siril SPCC -bgtol / PixInsight BackgroundNeutralization: per-channel offsets so that the sky has equal levels (no gains), plus one offset per"
                           " low-coverage provenance region for the sky's slowly varying pedestal, a measured profile across the frame's outer ring, and the sky's floor on the"
                           " border lines levelled against the frame's own where the gradient model left a residue, and a degree-1 per-channel plane fitted on sky alone for the"
                           " quarters' channel-to-channel tilt the refused colour-gradient step would have taken; needed because the stretch's slope at the sky is hundreds")
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
        method = str(self.p.get("stretch_method", "ghs2"))
        method = "ghs2" if method == "ghs" else method
        self._say("Stretching")
        cov = self.subject_coverage(base)
        self.info["subject_coverage"] = round(cov, 3)
        if cov <= SKY_STEPS_MAX_COVERAGE:
            base = self.colour_gradient(base)
        else:
            self.record("colour gradient (linear)", "none", None, status="skipped", note=f"the subject covers {cov * 100:.0f} % of the frame (> {SKY_STEPS_MAX_COVERAGE * 100:.0f} %): no sky to sample, the nebula's own colour would be taken")
        # neutralisation is NOT gated by the coverage.  The gate guards a specific hazard (C 20: the degree-2 per-channel model of ``colour_gradient`` took the North America
        # Nebula's own red), and a global polynomial is indeed unsafe on a frame-filling subject; offsets relative to the sky samples and a mesh capped at 2 % of the sky are not.
        # Skipping it is the worse failure: the stretch's slope at the sky is ~120, so every sky difference left here is amplified a hundredfold, and the coverage figure cannot
        # be trusted for this decision anyway - its threshold is mode + 3 sigma of the *smoothed sky's own scatter* (IC 1848: 0.4 % of the sky), so a well-flattened frame of
        # faint diffuse nebula reads "40 % subject" while 41 % of its pixels still lie below the mode.
        base = self.neutralise(base)
        base = self.banding(base, "linear starless", True)
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
        d2_default = GHS2_D if mi["coverage"] < GHS2_FILLING_COVERAGE else GHS2_D_FILLING     # autofinish's rule, also without autofinish (a plain export)
        d2 = float(np.clip(p.get("ghs_d2") or d2_default, 0.05, 10.0))
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
        lump = sky_lump(out)
        tried = [{"command": line, "black_clip": True, "sky_lump": lump}]
        # The script's black clip is a threshold at the sky minus ``sigma`` times the sky's own noise, so wherever the sky is
        # locally noisier more pixels fall under it - and the toe of the curve is steep enough (75-120 at the sky) to turn that
        # into a lightness step.  C 20 (EXP_15, 2026-09-30) hands this step a sky flat to 0.1 % and takes back a sky whose
        # sky-mask cells spread +25 % / -29.8 %: that is the "wide rim offset" the fleet ledger recorded on C 20, M 31 and
        # NGC 1499, none of which the linear frame carries.  The same step without the clip measures +-8 %.  Which of the two
        # ships is decided on the pixels rather than on a threshold this file would have to invent: unevenness that *is* the
        # subject (M 31's envelope measures +21 % with a clip that is doing nothing) is not improved by dropping the clip and
        # keeps it, so a render loses its black point only when the black point was what cost it the sky.
        line_nc = f"{line} -noblackclip"
        out_nc = lump_nc = None
        if lump is not None and lump < STAT_CLIP_FLOOR:
            # Nothing to win: the clip-free run ships only when it is STRETCH_CLIP_GAIN times more even, and a sky that
            # already measures flat under the clip cannot gain a difference anyone can see - while the second run of
            # the script costs as much as the first.  The record says the trial was not made, not that it failed.
            tried.append({"command": line_nc, "black_clip": False, "sky_lump": None,
                          "skipped": f"the clipped sky already measures {lump} % of sky"})
        else:
            try:
                t1 = time.time()
                out_nc = self.siril("stat", base, [line_nc], "Statistical stretch")
                seconds += time.time() - t1
                lump_nc = sky_lump(out_nc)
            except SirilError as exc:                               # the clip-free run is a candidate, never a requirement
                out_nc, lump_nc = None, None
                tried.append({"command": line_nc, "black_clip": False, "sky_lump": None, "error": str(exc)[:120]})
        if lump_nc is not None:
            tried.append({"command": line_nc, "black_clip": False, "sky_lump": lump_nc})
        if lump is not None and lump_nc is not None and lump_nc * STRETCH_CLIP_GAIN < lump:
            out, line, lump = out_nc, line_nc, lump_nc
        self.record("stretch (Statistical Stretch)", "siril pyscript Statistical_Stretch.py", out, commands=[line], seconds=seconds,
                    params={"median": t, "sigma": STAT_SIGMA, "coverage": round(cov, 3), "hdr": hdr,
                            "black_clip": "-noblackclip" not in line, "sky_lump": lump, "tried": tried},
                    source="Seti Astro Statistical Stretch (Siril port, C. Richard): median -> target; 0.10 for a small target, 0.25 (script default) when the target fills the "
                           "frame (pro-workflow.md A8); the black clip is kept unless dropping it measurably evens the sky, the clip being a threshold at the sky's own noise")
        self.info["stretch"] = {"method": "statstretch", "median": t, "coverage": cov, "black_clip": "-noblackclip" not in line, "sky_lump": lump}
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
        return gates.pixel_noise(lum, gates.sky_mask(lum.astype(np.float32)), starless)

    def _failed_of(self, pic: np.ndarray, starless: np.ndarray) -> list:
        """Which gates the delivered picture fails, for the comparison that decides whether a harder pass is allowed.
        The export hands ``gates.compute`` the linear-side measurements as well; here incumbent and candidate are judged
        by the same call, which is what makes the comparison fair rather than absolute.
        """
        return sorted(k for k, v in gates.compute(pic, starless=starless).items() if isinstance(v, dict) and not v["pass"])

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
        noise = self._sky_noise_of(out, starless)
        if not self.external or self.denoiser is None:
            why = ("preview: no AI steps" if not self.external else
                   (f"{self.denoiser_refused['tool']} was refused at every strength it was asked "
                    f"({', '.join(str(a) for a in self.denoiser_refused['asked'])}) for leaving the sky at |a*| "
                    f"{self.denoiser_refused['sky_chroma']['a']} / |b*| {self.denoiser_refused['sky_chroma']['b']} over its "
                    f"{self.denoiser_refused['limit']} gate, and escalating a tool this frame's sky cannot trust would deepen "
                    "the cast" if self.denoiser_refused else "nothing was available to ask"))
            self.record("noise reduction (sky gate)", "none", out, status="skipped", seconds=time.time() - t0,
                        params={"sky_pixel_noise": round(noise, 5), "gate": limit, "denoiser_refused": self.denoiser_refused},
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
        """Run the stretched-image denoiser and keep its pixels only while it leaves the sky inside the gate's own colour limit.

        Nothing ever judged this step's colour.  ``output/.ab12/audit_exports.py --cast`` -- which reads the per-step
        ``sky_channels`` the chain already records, over the fleet's exports -- found the denoiser to be the step that moves
        the sky's colour most wherever no palette runs: GraXpert's noise reduction on the stretched starless walked IC 434's
        sky by +31.5e-3 in ``(R-G)/(R+G)``, the same step that amplifies banding, and the only colour-changing step of the
        chain with neither a measured refusal rule nor a recorded before/after, while SCNR, the stretch's black clip and the
        border tools all had one.  So it is judged by the numbers the export is judged by -- the sky's median ``|a*|``
        and ``|b*|`` over ``gates.sky_mask`` *and* the spread of its four quadrants' signed tints, before and after
        (``_sky_chroma``).  The second one is not decoration: ``|a*|`` is dominated by chroma noise, so a denoiser
        *improves* it and no rule written on it alone can refuse a tint -- which is how C 20 (2026-11-01) shipped
        with ``sky_chroma`` at 1.73 of 2.0 and ``sky_chroma_quadrants`` at 1.97 of 1.5.  A step may leave the sky as
        coloured as it arrived (whose colour that is belongs to the calibration, not to this step); the step that
        takes either number past its gate *and* past ``DENOISE_SKY_WORSE`` times what it came with is the thing that
        coloured it, so it is asked again at the weakest strength that tool itself documents, and then refused: the
        pixels handed back, both measurements recorded, and ``self.denoiser`` cleared, because the closing step
        (``run`` -> ``_close_sky_noise``) escalating a tool this frame's sky cannot trust would only deepen the cast.

        ``params`` is called with the amount that was accepted, *after* the tool ran, so it can record what the tool reported.
        """
        lim = float(gates.LIMITS["sky_chroma"])
        lim_q = float(gates.LIMITS["sky_chroma_quadrants"])
        before = self._sky_chroma(arr)
        cands = [a for a in dict.fromkeys(round(float(x), 3) for x in (amounts if amounts is not None else (amount,))) if a > 0]
        tried: list[dict] = []
        for a in cands:
            out, cmd = run(a, arr)
            after = self._sky_chroma(out)
            tried.append({"amount": a, "sky_chroma": after})
            verdict = None
            if before is not None and after is not None:
                chroma = (max(after["a"], after["b"]) > lim
                          and max(after["a"], after["b"]) > DENOISE_SKY_WORSE * max(before["a"], before["b"]))
                cast = (after["spread"] > lim_q
                        and after["spread"] > DENOISE_SKY_WORSE * before["spread"])
                verdict = "sky_chroma" if chroma else ("sky_chroma_quadrants" if cast else None)
            tried[-1]["refused_for"] = verdict                             # the sidecar says which rule each amount tripped
            if verdict is None:
                self.denoiser, self.denoiser_refused = (tool, run, ceiling, a), None
                self.record("noise reduction (stretched)", tool, out, commands=[cmd] if isinstance(cmd, str) else list(cmd),
                            seconds=time.time() - start,
                            params={**params(a), "sky_chroma_before": before, "sky_chroma_after": after, "tried": tried, "limit": lim},
                            note=note, source=source)
                return out
        if not cands:
            return arr
        last = tried[-1]["sky_chroma"]
        why = tried[-1].get("refused_for") or "sky_chroma"
        self.denoiser, self.denoiser_refused = None, {"tool": tool, "sky_chroma": last, "limit": lim,
                                                      "spread": last["spread"], "spread_limit": lim_q,
                                                      "refused_for": why, "asked": cands}
        self.record("noise reduction (stretched)", tool, None, seconds=time.time() - start, status="refused",
                    params={**params(cands[-1]), "sky_chroma_before": before, "sky_chroma_after": last,
                            "tried": tried, "limit": lim, "spread_limit": lim_q},
                    note=f"{tool} left the sky at |a*| {last['a']} / |b*| {last['b']} (the gate's {lim}) with its four "
                         f"quadrants' signed tints {last['spread']} apart (the gate's {lim_q}), where it arrived at "
                         f"{before['a']} / {before['b']} and {before['spread']}; every strength asked left the sky "
                         f"coloured that way ({why}), so the step's pixels were handed back",
                    source=source)
        return arr

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
        """``rmgreen 3 <amount>`` (Siril's SCNR), kept only while it leaves the sky no more coloured than it found it.

        The amount is decided on the **subject** (``autofinish``: median G - (R + B) / 2 over the subject mask, above
        ``SCNR_CAST``) and Siril reduces green over the whole frame, so a subject's cast is subtracted from the sky as
        well -- and rmgreen preserves lightness, so the green it lifts out comes back through red and blue.  A neutral
        sky therefore leaves it magenta: M 31 (Seestar, 2026-11-01) went from a sky at |a*| 0.4 / |b*| 0.6 to
        13.4 / 9.0 -- 6.7 x the gate's own limit, ``sky_chroma`` failing -- with a green nebula as the only reason the
        step ran, and the render was delivered with the cast in it.  A step may leave the sky as coloured as it
        arrived (whose colour that is belongs to the calibration, not to this step); the step that *takes* it past the
        gate is retried weaker -- a pass that reaches the gate keeps the subject its correction -- and then refused,
        the pixels handed back and both measurements recorded.
        """
        lim = float(gates.LIMITS["sky_chroma"])
        before = self._sky_chroma(arr)
        tried: list[dict] = []
        total = 0.0
        last: dict | None = None
        for scale in SCNR_SKY_TRIES:
            amount = round(scnr * scale, 3)
            if amount <= 0:
                break
            line = f"rmgreen 3 {amount:.3f}"
            t0 = time.time()
            out = self.siril("scnr", arr, [line], "SCNR")
            total += time.time() - t0
            after = self._sky_chroma(out)
            tried.append({"amount": amount, "sky_chroma": after})
            last = after
            if before is None or after is None or not (max(after["a"], after["b"]) > lim
                                                       and max(after["a"], after["b"]) > SCNR_SKY_WORSE * max(before["a"], before["b"])):
                self.record("SCNR", "siril rmgreen", out, commands=[line], seconds=total,
                            params={"type": 3, "amount": amount, "sky_chroma_before": before, "sky_chroma_after": after,
                                    "tried": tried, "limit": lim}, source=source)
                return out
        if not tried:
            return arr
        self.record("SCNR", "siril rmgreen", None, commands=[f"rmgreen 3 {t['amount']:.3f}" for t in tried], seconds=total,
                    status="refused", params={"type": 3, "amount": scnr, "sky_chroma_before": before, "sky_chroma_after": last,
                                               "tried": tried, "limit": lim},
                    note=f"rmgreen left the sky at |a*| {last['a']} / |b*| {last['b']} over the gate's {lim}, where it arrived at "
                         f"{before['a']} / {before['b']}; retried weaker {len(tried) - 1} time(s) and every pass coloured the sky, "
                         "so the step's pixels were handed back", source=source)
        return arr

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
        band = int(round(LC_EDGE_BAND * CLAHE_SCALES[0]))
        floor, sides = edge_sky_floor(lum, lo, band, h) if 4 <= band <= min(arr.shape[1:]) // 2 else (None, {})
        w = self.range_mask(lum, mask, mi, floor=floor)
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
                    params={"clip_limit": CLAHE_CLIP, "tile_px": list(CLAHE_SCALES), "grid": grids, "strength": round(k, 3), "mask_low": round(lo, 4), "mask_high": round(hi, 4),
                            "edge_band_px": band if floor is not None else 0, "edge_sides": sides},
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
        lines += ["icc_assign sRGB", "icc_convert_to sRGB relative"]
        t0 = time.time()
        out = self.siril("fin", arr, lines + ["save finish"], "Final", out="finish")
        self.record("final black point + ICC", "siril linstretch / icc_convert_to", out, commands=lines, seconds=time.time() - t0,
                    params={"BP": round(bp, 6), "toe": round(toe, 5), "toe_percentile": TOE_PERCENTILE, "background": round(sky, 4), "icc": "sRGB, relative colorimetric",
                            "midtones": round(mid, 5), "sky_level": level, "sky_median_clamped": clamped},
                    source="Brecher: black point at the toe of the histogram, ICC to sRGB IEC61966-2.1 relative colorimetric (pro-workflow.md A16)")
        return out

    # ---- the whole chain -----------------------------------------------------------------------------------------------------
    def run(self, base: np.ndarray, stars: np.ndarray | None) -> tuple[np.ndarray, dict]:
        self.n = 0
        self.record("starless, linear", "input (post.restore -> post.remove_stars)", base, linear=True, params={"stars_layer": stars is not None},
                    note="shown through a linked autostretch")
        s = self.stretch(base)
        s = self.highlights(s)
        pre = s
        s = self.denoise(s)
        s = self.banding(s, "stretched starless", False, noise_from=pre)
        s = self.colour(s)
        s = self.local_contrast(s)
        s = self.sharpen(s)
        s = self.saturation(s)
        t = None
        if stars is not None and float(self.p.get("star_reduction", 0.4)) < 1.0:
            t = self.stars(stars)
        s, out = self._close_sky_noise(s, lambda sl: self._compose(sl, t))   # the gate measures the delivered picture, so the loop runs after the
        self.info.update(chain=self.records, starless=s)                     # composition and recomposes every candidate it tries
        return out, self.info

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
