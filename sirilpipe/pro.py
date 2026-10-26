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
import os
import re
import subprocess
import time

import cv2
import numpy as np

from . import config, export as export_mod, finish, fitsio, nbn, tools, workflow
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
NXT_DN = (0.60, 0.85)              # NXT --dn on stretched starless (RC-Astro / Brecher)
CLAHE_CLIP = 1.5                   # Brecher's LHE contrast limit
CLAHE_SCALES = (150, 40)           # px, Brecher's LHE scales (M51: 150 then 40; IC 1848 40 then 150)
LC_STRENGTH = 0.4                  # blend per unit of ``local_contrast`` (0.5 -> 0.2, the lowest published strength)
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
FRAME_FILLING = 0.5                # the object covers at least this fraction of the frame: ``auto_stretch`` raises the background to the top of Brecher's band
SKY_STEPS_MAX_COVERAGE = 0.3       # above this subject coverage the frame has no sky to sample: the colour-gradient and neutralisation steps are skipped (C 20: the degree-2
                                   # per-channel model took the North America Nebula's own red, 2026-10-15); DBE / BackgroundNeutralization need true sky samples


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


def level_stats(arr: np.ndarray) -> dict:
    lum = finish.luminance(arr)
    x = finish.sample(lum)
    return {"median": round(float(np.median(x)), 5), "p99": round(float(np.percentile(x, 99)), 5), "p99.9": round(float(np.percentile(x, 99.9)), 5),
            "max": round(float(arr.max()), 5), "clipped": round(float((arr.max(0) >= 0.995).mean()), 6), "white": round(float((lum >= 0.97).mean()), 6)}


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

class Finisher:
    """One run of the chain.  ``external`` = the AI steps (NoiseXTerminator / GraXpert on the stretched image) run; the preview passes False."""

    def __init__(self, work: str, header, p: dict, dual: bool, *, external: bool = True, progress=None, checkpoint=None, tick=None, device: str = "auto",
                 steps=None, keep_pictures: bool = True):
        self.work, self.header, self.p, self.dual, self.external = work, header, p, dual, external
        self.progress, self.checkpoint, self.tick, self.device, self.steps = progress, checkpoint, tick, device, steps
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
        IC 1805's starless layer) would otherwise become a 0.1 display-unit colour cast of the sky."""
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
        self.record("background neutralisation (linear)", "siril split / pm (offsets) / rgbcomp", out, commands=lines, seconds=time.time() - t0, linear=True,
                    params={"sky_median_before": [round(m, 7) for m in med], "offsets": [round(o, 8) for o in off]},
                    source="Siril SPCC -bgtol / PixInsight BackgroundNeutralization: per-channel offsets so that the sky has equal levels (no gains); needed because the stretch's slope at the sky is hundreds")
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
            base = self.neutralise(self.colour_gradient(base))
        else:
            self.record("colour gradient (linear)", "none", None, status="skipped", note=f"the subject covers {cov * 100:.0f} % of the frame (> {SKY_STEPS_MAX_COVERAGE * 100:.0f} %): no sky to sample, the nebula's own colour would be taken")
            self.record("background neutralisation (linear)", "none", None, status="skipped", note="no sky samples on a frame-filling subject (SPCC's background neutralisation stands)")
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
        self.record("stretch (Statistical Stretch)", "siril pyscript Statistical_Stretch.py", out, commands=[line], seconds=time.time() - t0,
                    params={"median": t, "sigma": STAT_SIGMA, "coverage": round(cov, 3), "hdr": hdr},
                    source="Seti Astro Statistical Stretch (Siril port, C. Richard): median -> target; 0.10 for a small target, 0.25 (script default) when the target fills the frame (pro-workflow.md A8)")
        self.info["stretch"] = {"method": "statstretch", "median": t, "coverage": cov}
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
            cli = ["--denoise-mode", "full", "--denoise-strength", f"{dn:g}", "--denoise-color", f"{dc:g}"] + {"lite": ["--lite"], "walking": ["--walking"]}.get(model, [])
            try:
                out, m = tools.cosmic_clarity.run("denoise", arr, cli, work=self.cwd, device=self.device, checkpoint=self.checkpoint,
                                                  progress=(lambda i, n, msg: self.progress(msg) if self.progress else None))
                out = np.clip(out, 0, 1)
                self.record("noise reduction (stretched)", "cosmic clarity denoise", out, commands=[m["command"]], seconds=time.time() - t0,
                            params={"denoise_strength": dn, "denoise_color": dc, "model": model, "backend": m.get("backend"), "device": m.get("device")},
                            source=f"SetiAstro Cosmic Clarity AI4 denoise, {model} model (siril-scripts CosmicClarity_Native.py, full mode: mono model on the luminance, colour model on the chroma)")
                return out
            except Cancelled:
                raise
            except (ToolUnavailable, ToolError) as e:
                self.record("noise reduction (stretched)", "cosmic clarity denoise", None, status="unavailable", note=str(e)[:200])
                return arr
        src, dst = os.path.join(self.cwd, "nxt_in.fits"), os.path.join(self.cwd, "nxt_out.fits")
        fitsio.write_rgb(src, arr, self.header)
        try:
            tools.rcastro.require("nxt")
            m = tools.rcastro.run("nxt", src, dst, device=self.device, progress=(lambda i, n, msg: self.progress(msg) if self.progress else None), checkpoint=self.checkpoint,
                                  csep=True, di=dn, dc=dc)
            out = np.clip(fitsio.read_rgb(dst)[0], 0, 1)
            self.record("noise reduction (stretched)", "rc-astro nxt", out, commands=["rc-astro nxt " + " ".join(m.get("flags") or [])], seconds=time.time() - t0,
                        params={"di": dn, "dc": dc, "flags": m.get("flags")}, source="RC-Astro / Brecher: NXT on the stretched starless, amount 0.6-0.85; colour separation --dc 0.9-1.0")
            return out
        except Cancelled:
            raise
        except (ToolUnavailable, ToolError) as e:
            self.record("noise reduction (stretched)", "rc-astro nxt", None, status="unavailable", note=str(e)[:200])
        try:
            strength = float(np.clip(dn, 0.6, 0.8))
            m = tools.graxpert.denoise(src, dst, strength=strength, gpu=self.device, progress=(lambda i, n, msg: self.progress(msg) if self.progress else None), checkpoint=self.checkpoint)
            out = np.clip(fitsio.read_rgb(dst)[0], 0, 1)
            self.record("noise reduction (stretched)", "graxpert denoise", out, commands=[f"graxpert denoising strength={strength}"], seconds=time.time() - t0,
                        params={"strength": strength, "model": m.get("model")}, note="fallback: NoiseXTerminator unavailable", source="pro-workflow.md A10: GraXpert 0.6-0.8")
            return out
        except Cancelled:
            raise
        except (ToolUnavailable, ToolError) as e:
            self.record("noise reduction (stretched)", "none", None, status="unavailable", note=str(e)[:200])
            return arr
        finally:
            for f in (src, dst):
                if os.path.exists(f):
                    os.remove(f)

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
                line = f"rmgreen 3 {scnr:.3f}"
                t0 = time.time()
                out = self.siril("scnr", out, [line], "SCNR")
                self.record("SCNR", "siril rmgreen", out, commands=[line], seconds=time.time() - t0, params={"type": 3, "amount": scnr},
                            source="Siril rmgreen on an Ha-in-green palette (NarrowbandNormalization.py applies SCNR to SHO-type palettes)")
            return self.ha_luminance(out, ha)
        scnr = float(np.clip(p.get("scnr", 0.0), 0.0, 1.0))
        if not self.dual and scnr > 0:
            line = f"rmgreen 3 {scnr:.3f}"
            t0 = time.time()
            out = self.siril("scnr", arr, [line], "SCNR")
            self.record("SCNR", "siril rmgreen", out, commands=[line], seconds=time.time() - t0, params={"type": 3, "amount": scnr},
                        source="Siril rmgreen (PixInsight SCNR), additive mask; broadband only (pro-workflow.md A13)")
            return out
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
        w = self.range_mask(lum, mask, mi)
        lo, hi = float(mi["threshold"]), float(np.percentile(cv2.GaussianBlur(np.ascontiguousarray(lum), (0, 0), MASK_SIGMA)[mask], 95)) if mask.any() else float(mi["threshold"]) + 0.1
        k = LC_STRENGTH * amount
        longside = max(arr.shape[1:])
        grids = [max(2, int(round(longside / s))) for s in CLAHE_SCALES]
        lines = ["save lc_a", f"clahe {CLAHE_CLIP:g} {grids[0]}", "save lc_b", f'pm "$lc_a$+{k:.4f}*$lc_m$*($lc_b$-$lc_a$)"', "save lc_c",
                 f"clahe {CLAHE_CLIP:g} {grids[1]}", "save lc_d", f'pm "$lc_c$+{k:.4f}*$lc_m$*($lc_d$-$lc_c$)"']
        t0 = time.time()
        out = self.siril("lc", arr, lines, "Local contrast", extra={"lc_m": np.stack([w, w, w])})
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

    def range_mask(self, lum: np.ndarray, mask: np.ndarray, mi: dict) -> np.ndarray:
        """Range mask (PixInsight RangeSelection + blur): 0 on the sky, 1 on the subject, smooth between the nebula mask's threshold and the subject's 95th percentile."""
        sm = cv2.GaussianBlur(np.ascontiguousarray(lum), (0, 0), MASK_SIGMA)
        lo = float(mi["threshold"])
        hi = float(np.percentile(sm[mask], 95)) if mask.any() else lo + 0.1
        w = np.clip((sm - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
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
            # the colour-preserving star stretch (arcsinh-style): the stretched LIGHTNESS with the linear colour ratios, composed by Siril rgbcomp -lum, blended by c
            t0 = time.time()
            lum = finish.luminance(out)
            lines2 = ["rgbcomp -lum=sc_l sc_c -out=sc_lrgb", "load sc_lrgb", f'pm "{1.0 - c:.4f}*$sc_in$+{c:.4f}*$sc_lrgb$"']
            colour = np.clip(stars / np.maximum(stars.max(0, keepdims=True), 1e-9), 0, 1).astype(np.float32)
            lp = self.mono("sc_l", lum)
            try:
                out2 = self.siril("sc", out, lines2, "Star colour", extra={"sc_c": colour})
            finally:
                for fp in (lp, os.path.join(self.cwd, "sc_lrgb.fit")):
                    if os.path.exists(fp):
                        os.remove(fp)
            self.record("stars colour preservation", "siril rgbcomp -lum / pm", out2, commands=lines2, seconds=time.time() - t0, params={"star_color_preservation": c},
                        source="LRGB composition of the stretched lightness with the linear colour ratios (the arcsinh / colour-preserving star stretch: PixInsight ArcsinhStretch, Seti Star Stretch colour boost)")
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
        if abs(mid - 0.5) > 1e-4:
            lines.append(f"mtf 0 {mid:.4f} 1")
        lines += ["icc_assign sRGB", "icc_convert_to sRGB relative"]
        t0 = time.time()
        out = self.siril("fin", arr, lines + ["save finish"], "Final", out="finish")
        self.record("final black point + ICC", "siril linstretch / icc_convert_to", out, commands=lines, seconds=time.time() - t0,
                    params={"BP": round(bp, 6), "toe": round(toe, 5), "toe_percentile": TOE_PERCENTILE, "background": round(sky, 4), "icc": "sRGB, relative colorimetric",
                            "midtones": round(mid, 5), "sky_level": level},
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
        starless_display = s
        t = None
        if stars is not None and float(self.p.get("star_reduction", 0.4)) < 1.0:
            t = self.stars(stars)
        out = self.recombine(s, t)
        if t is not None:
            out = self.halos(out)
        out = self.cc_sharpen(out)
        out = self.grade(out)
        out = self.finalise(out)
        self.info.update(chain=self.records, starless=starless_display)
        return out, self.info


def _community_python() -> str | None:
    p = os.path.join(config.home_dir(), ".local", "share", "siril", "venv", "bin", "python")
    return p if os.path.exists(p) else None


def _community_script(rel: str) -> str | None:
    p = os.path.join(config.home_dir(), ".local", "share", "siril-scripts", rel)
    return p if os.path.exists(p) else None
