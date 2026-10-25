##############################################
# VeraLux — ImageMM
# Hybrid Restoration & Super-Resolution Engine
# Author: Riccardo Paterniti (2026)
# Contact: info@veralux.space
##############################################
# (c) 2026 Riccardo Paterniti
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Headless pipeline edition of VeraLux_ImageMM.py v1.0.0, branch IMM_update of
# https://gitlab.com/RikyPate/vera-lux-siril-scripts (VeraLux/VeraLux_ImageMM.py), modified for sirilpipe.
"""VeraLux ImageMM without its Qt window: the restoration engine of the script, run by ``sirilpipe.imagemm_stage``.

Kept from the script (same classes, method names, objectives and defaults): ``PSFSolver.solve`` / ``solve_sr`` (the
fits of Eq. 2 / Eq. 11, h >= 0, with the micro-TV term), ``ImageMMEngine`` (Huber weights with delta 0.05, weights m / v
with the MAD variance floor of the practical mode, update clipping kappa (Eq. 9), the Appendix-C stopping rule over
the effective mask mu, average-pool D and its adjoint, practical / formal modes, 20 iterations), the ``findstar``
list parser and the GUI defaults (``DEFAULTS``).  The script's pre-processing tab (calibrate -debayer, seqsubsky 1,
register) and its findstar loop are run by the stage as Siril scripts.

Changed for the pipeline.  Each was found on real data - DWARF 3 IC 1805 (245 x 3796 x 2152 RGB frames, a slightly
defocused, ring-shaped PSF) and M 42 (2026-10-04, 155 frames) - and is marked ``PIPELINE (change n)`` in the code.
1-6 make the script run on the pipeline's frames; 7-12 are what it took for the result to be clean (crops judged
against stack.fits: star FWHM, ring depth round stars, band-pass sky noise).  Options 7, 11 and the model variance are
switches (``run_imagemm`` arguments, ``imagemm_*`` settings), on by default.

1. Frames stream from a disk store (``FrameStore``) instead of RAM: the script caches every frame three times
   (data, mask, variance: 300 MB per frame here, 73 GB for the night).  Only the data and a saturation flag are
   stored; mask and variance are computed on the GPU, as ``build_mask_and_variance`` does.
2. The convolutions run through the FFT, exactly equal to the script's padded convolution (reflect padding in the
   practical mode, zeros in the formal mode, the centre kept), and the forward operator is a convolution.  The
   script's ``spatial_convolve`` is ``F.conv2d``, a cross-correlation: its forward model drew every star as the
   measured PSF rotated by 180 degrees (and the adjoint was the convolution).  Here a point source renders as the
   measured star (``tests/test_veralux_imagemm.py``).
3. Star positions: Siril's findstar X / Y are pixel-edge coordinates with Y counted from the other end of the FITS
   rows (measured on synthetic stars: array column = X - 0.5, row = height - Y - 0.5).  The script cut its stars at
   ``data[int(Y), int(X)]`` - the vertically mirrored position, mostly empty sky.  ``siril_to_array`` converts.
4. Mask m: the script excluded pixels <= 0 after the background subtraction ("registration borders").  On a faint
   sky that also drops the low half of the noise (2-11 % of the sky pixels of IC 1805 frames are negative after
   ``seqsubsky``), which biases the faint sky up.  Pixels without data are NaN in the store; everything else is
   valid.  ``y = max(y, 0)`` is kept as in the script.
5. Saturation: the script masks ``y >= 0.995 * level`` with the level from BITPIX (16 / 8 bit) - None for Siril's
   32-bit float frames, and wrong for 12-bit sensors in 16-bit files.  The stage finds the sensor ceiling and the
   saturated pixels on the raw frames, keeps those still saturated after calibration (the rest are hot pixels) and
   lets Siril register that mask with the frames' own transforms.  Saturated stars are not used for the PSF.
6. Initial guess: the mean of the frames over their valid pixels (the script's comment), where pixels without data
   had counted as 0; where no frame has a valid sample (a star saturated all night) the frames' clipped values, as
   the script's guess - a multiplicative update cannot grow a pixel that starts at 0.
7. Per-channel PSFs (``per_channel_psf``): the script measures green and repeats it; a debayered OSC frame's R and
   B, sampled at 1/4 density, are wider (IC 1805: R wings 1.6-2 x G at 6-8 px).
8. The PSF fits: the script ran Adam (lr 0.005) on ReLU(h); the wing pixels (~1e-4) went negative in the first steps,
   where ReLU's gradient vanishes, and never came back - the kernels lost their wings beyond 6 px and had cores 27 %
   too bright, which rang round every star.  Same objective, solved by projected FISTA with a smooth (Charbonnier) TV
   (``PSFSolver._fit``).
9. Stacked PSF wings: beyond ~10 px a pixel of the median stack is noise, and the script's ``max(stack, 0)`` made it a
   floor holding 4-5 % of the kernel's light; such pixels are blended into their ring's mean (``_radial_wings``).
10. Star cut-outs are aligned to findstar's sub-pixel centre (bicubic) before the median, as the script's docstring
   says ("aligns and stacks"); it cut them at the integer pixel.
11. Lower bound (``lower_bound``): the Lower Bound Constraint Algorithm (Lanteri, Roche & Aime 2002; Roche et al.
   2003): x >= the local background less 3 noise sigma instead of x >= 0, which removes the moat the updates dig
   round every star (IC 1805: -1.4 % of the restored peak -> -0.2 %).  "local" (default): min(36 px median, 5 x 5
   minimum of the 1 px smoothed image), which follows the nebula; "global", one level per channel at the darkest
   sky, let the moats dig into bright nebula - black discs round the stars after finishing (full IC 1805 run).
12. Saturated pixels are censored data (Whyte, Sivic & Zisserman 2014): in the update while the model is below the
   clipped value, no information above it.

13. Reference-centred PSFs and per-frame photometry: each frame's kernel from cut-outs at the stars' positions on the
   frames' mean (a frame's residual registration stays in its kernel), each frame's transmission s_t.
14. Noise: the photon transfer measured on the frames (var = floor + gain x signal), Huber on the standardised
   residual (the paper's), not the script's absolute delta 0.05.
15. PSF support 81 px with neighbours masked, aperture-normalised cut-outs, the support ending where the stacked PSF
   stops being detected (``_detected_radius``), no TV in the fit.
21. Frame selection: transmission < 0.5 x the median, no measurable PSF, or a fit far outside the noise (chi2 outlier
   rule at iteration 10) - M 31 2026-10-21: 33 of 299 frames, at chi2 28-261 against 0.5-1.
23. Relative normalisation: y' = (y - plane_t) / s_t against the frames' mean (the script's per-frame seqsubsky took
   part of large targets - M 31's outer disk); seqsubsky is an option of the stage, off.
24. Saturated cores as point sources (flux kept) at iteration 10: the data say only "at least the clip level" there,
   and the update had kept flat discs (option, off: superseded by 25).
25. Saturated cores after the restoration (``_saturated_star_photometry``): a star gets local deringing support (the
   frames' own mean in its censored region, blended in over 5 px - every rewrite of the star inside the restoration made
   its own artefact); a galaxy nucleus (light beyond the core > 0.1 x the core's) keeps its restored shape at the colour
   of its unsaturated wings (PSF photometry).
26. Mean-image wings joined to frames' kernels (``psf_wings``, off: galaxy light in the wings on M 31).
27. The PSF's static non-radial part (m = 1, 2) from the frames' mean (``psf_asymmetry``): the optics' one-sided wing
   (M 31 dipole 0.33) that single-frame stacks average away - a crown on one side of every bright star.
28. The PSF's radial wing beyond 5 px (``psf_radial_wings``, ``_radial_wing_profile``): an extended PSF joined from
   unsaturated and saturated stars (Infante-Sainz, Trujillo & Roman 2020), inverse-variance weighted median per ring,
   stars on a curved sky left out.
29. Bloom zone (``bloom_grow``, ``_bloom_zone``): pixels within 4 px of a frame's saturated pixels are missing data,
   not censored - saturated stars carry 17-37 % more light at 4-7 px than the unsaturated stars' PSF (M 31), and the
   restoration built a shell of it round every core that the core step then cut into a dark gap and a ring; round a
   large saturated patch the zone grows to 0.75 x its equivalent radius (change 33).
30. Saturated stars in the initial guess as points (``point_init``) at the light PSF photometry of their unsaturated
   wings measures (at least the clipped light): started from the clipped light the update never grew the point and
   kept the difference as a halo.
31. Optional PSF wing calibration on the bright stars (``run_calibrated``): measured on the restoration as fitted, where
   the frames' data are dense.  Not used by the stage: in a crop on a galaxy's bulge it follows the bulge, not the PSF.
32. The extended PSF to 80 px (``FAR_WING_R``), joined from unsaturated and saturated stars; per channel its colour
   (``_wing_chroma``: the field's brightest star's halo reddens outward) and its lobe beyond 20 px
   (``_far_nonradial``, m = 1, 2 from the saturated stars).
33. Saturated cores after the restoration (``_saturated_star_photometry``, extends 25): a star where the frames' data
   are sparse is its restored-image PSF at its wings' colour on a floor fitted 4-20 px outside; in the join the
   restoration's own structure (less its smooth part per ring) is kept, and the PSF's tail fades on beyond.  The
   lower bound takes the 5 x 5 minimum of the image everywhere (a dust lane under the bulge was held flat).
The lower bound's margin is 1 sigma (``bound_sigma``): 3 sigma let rings of -1.2 sigma form round bright stars in R / B,
0 sigma clipped the sky noise.  The input frames are the stack's own Bayer drizzle when the stack is one (the stage):
the script's debayer mixed the colours (ImageMM R/G = 0.32 + 1.28 x the stack's).

Variance (``variance_from = "model"``): the practical mode's v = floor + y uses each single frame's noisy y; on a faint
sky the weights then swing ~10 x with the noise itself and anti-correlate with it, which raised the band-pass noise
to 1.5-2.2 x the stack's.  v is evaluated at the model prediction instead (the Poisson / IRLS convention), which keeps
the update Richardson-Lucy-like (IC 1805: 1.14-1.35 x at the same sharpness).
"""
from __future__ import annotations

import gc
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

VERSION = "1.0.0"
SOURCE = "https://gitlab.com/RikyPate/vera-lux-siril-scripts/-/blob/IMM_update/VeraLux/VeraLux_ImageMM.py"

# the GUI's default values (VeraLux_ImageMM.py ImageMMGUI.init_ui / run_engine)
# (iterations and latent_sigma: the pipeline's values - the script's are 20 and 0.45.  The paper's Appendix-C stopping
#  rule ends the restoration (M 31 2026-10-21: at 92 iterations, faint stars then at 2.42 px FWHM against the latent's
#  2.35 px limit; at 20 / 60 iterations they were unconverged, 2.5-3.4 px); iterations is only the cap.  A latent sigma
#  under the paper's 1 px gave no sharper converged faint stars worth its cost: green cores 7.6 % (0.7) to 24 % (0.45)
#  too faint against the frames)
DEFAULTS = {"iterations": 150, "kappa": 2.0, "huber_delta": 0.05, "mu": 0.1, "stop_epsilon": 1e-5,
            "saturation_level": None, "shot_noise_scale": 1.0, "read_noise": 0.0, "upscale_factor": 1,
            "psf_sigma": 1.0, "formal_mode": False,
            # pipeline options (not in the script), on by default: each fixed a measured defect (module docstring)
            "variance_from": "model", "per_channel_psf": True, "latent_sigma": 1.0, "lower_bound": "local",
            "reference_psf": True, "photometry": True, "noise_model": "ptc", "huber_sigma": 2.0,
            "halo": False, "saturated_points": False, "saturated_photometry": True,
            # (change 26, off: on M 31 the mean image's wings carried galaxy light - diffuse residual 0.02 -> 0.17)
            "psf_wings": False, "psf_asymmetry": True, "psf_radial_wings": True,
            # the lower bound's margin below the local background, in the latent's pixel sigma (change 11): M 31 rings at
            # 5-10 px round the 12 brightest stars -1.17 / -0.34 / -1.04 sigma (R / G / B) at 3, -0.02 / +0.45 / -0.07 at
            # 1, +0.5 / +1.1 / +0.5 at 0 (a halo, and a clipped noise floor: sky noise 0.8 x the stack's)
            "bound_sigma": 1.0,
            # pixels within this many px of a frame's saturated pixels (same channel) are missing data, not censored: the
            # saturated stars hold 17-37 % more light at 4-7 px than the PSF of the unsaturated ones (charge spill /
            # non-linearity next to a full well; M 31 2026-10-22), and the restoration built a shell of it round every core
            "bloom_grow": 4,
            # saturated stars start as points (change 30): the starting halo stayed where the censored cores left it free
            "point_init": True,
            # saturated pixels are missing data (the paper's mask), not censored: next to a full well the detector is
            # non-linear (charge spill to 8-10 px round the brightest stars, M 31), and "at least the clip level" there
            # forced the star's light up until its wings over-predicted 6-18 px - a moat held at the lower bound
            "censor_saturated": False}
SATURATION_FRAC = 0.995          # build_mask_and_variance(saturation_frac=0.995)
# PSF cut-outs: the script's crop_size 31 (33 x 33).  PIPELINE (change 15): 79 (81 x 81, radius 40 px).  IC 1805: 6 % of a
# star's light lies beyond 16 px (93.8 % within 16, 96.2 % within 39, 99 isolated stars of the stack); a unit-sum 33 px
# kernel made the model's stars 6 % too bright inside 16 px and dark outside, and the restoration answered with a moat
# round every core and a halo disc beyond it.
CROP_SIZE = 79
TV_EPS = 1e-4
BG_ANNULUS = (25.0, 38.0)        # px: a PSF cut-out's background (change 19)
AP_RADIUS = 8.0                  # px: its normalising aperture flux
MIN_STAR_SNR = 20.0              # aperture SNR a star needs to enter the PSF stack
WING_JOIN = 5.0                  # px: a frame's kernel is its own inside, the mean image's radial wing beyond (change 28)
SAT_BLEND = 3.0              # px over which a saturated star's photometric core joins the restoration
SPARSE_FRAC = 0.95               # a pixel valid in fewer of the frames than this is the star's model (change 25): the frames that
                                 # still see it there are the poorer-seeing ones (a smaller saturated core) - a biased sample
TAIL_FLOOR = 0.01                # x the restoration's pixel noise: a saturated star's tail is carried out to this level (change 33)
BLOOM_SCALE = 0.75               # x a saturated patch's equivalent radius: its bloom zone when that exceeds bloom_grow (change 33)
FAR_WING_R = 80                  # px: the PSF's measured wing reaches this far; kernels are (2 x 80 + 1)^2 (change 32)
MAX_WING_HOLE_PX = 2000          # a saturated hole larger than this does not enter the extended PSF (change 28; the field's brightest star carries the far wing)
MAX_SAT_CORE_PX = 2000           # a saturated core larger than this is not a star (change 25; the field's brightest star, 500+ px, was skipped at 400 - nuclei are told apart by the extended test)
HALO_R = 110                     # px: the optics' halo measured on the frames' mean image (change 18)
FRAME_REJECT_ITER = 10           # the iteration after which frames that do not fit are dropped (change 22)
MIN_TRANSMISSION = 0.5          # a frame at under this x the median transmission is not used (change 21)
MIN_PSF_STARS = 5                # a channel's PSF stack needs this many stars, else green's is used
NEIGHBOUR_R = 12                 # px masked round every other star inside a PSF cut-out                    # Charbonnier TV smoothing of the PSF fit: the wing pixels' noise (~1e-4 of a unit-sum kernel)


def _torch():
    import torch                                         # noqa: WPS433 (optional, large: imported when used)
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    return torch, nn, F, optim


# ------------------------------------------------------------------------------
#  FINDSTAR (ImageMMWorker.run's parser)
# ------------------------------------------------------------------------------

def parse_findstar(path: str) -> list[tuple[float, float]]:
    """(X, Y) of every star of a Siril ``findstar`` list: the X / Y columns from the header line, else columns 5 / 6
    (VeraLux's parser, unchanged)."""
    coords: list[tuple[float, float]] = []
    if not os.path.exists(path):
        return coords
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    x_col_idx = y_col_idx = -1
    for line in lines:
        line = line.strip()
        if not line.startswith('#'):
            continue
        parts_upper = [p.upper() for p in line.lstrip('#').strip().split()]
        if 'X' in parts_upper and 'Y' in parts_upper:
            x_col_idx, y_col_idx = parts_upper.index('X'), parts_upper.index('Y')
            break
    if x_col_idx != -1 and y_col_idx != -1:
        max_idx = max(x_col_idx, y_col_idx)
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) > max_idx:
                try:
                    coords.append((float(parts[x_col_idx]), float(parts[y_col_idx])))
                except ValueError:
                    continue
    else:
        for line in lines:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 7:
                try:
                    coords.append((float(parts[5]), float(parts[6])))
                except ValueError:
                    pass
    return coords


def siril_to_array(x: float, y: float, height: int) -> tuple[float, float]:
    """(column, row) of the FITS data array (as numpy / astropy read it) of findstar's (X, Y) on an image of
    ``height`` rows.  PIPELINE (change 3): measured with Gaussian stars at known array positions -
    X - column = 0.500 (sd 0.003), (height - Y) - row = 0.500 (sd 0.002)."""
    return x - 0.5, height - y - 0.5


# ------------------------------------------------------------------------------
#  FRAME STORE (PIPELINE, change 1)
# ------------------------------------------------------------------------------

def transmission_drops(scale_t: np.ndarray, exptimes=None) -> dict[int, str]:
    """{frame index: why} of the frames not used (change 21): no photometry, or a transmission - the frame's star flux
    ratio s_t per second of exposure - under MIN_TRANSMISSION x the median frame's."""
    s_ = np.asarray(scale_t, np.float64)
    exp_t = np.array([float(e) if e and float(e) > 0 else 1.0 for e in (exptimes if exptimes is not None else [1.0] * len(s_))], np.float64)
    trans = s_ / exp_t
    med = float(np.nanmedian(trans))
    out: dict[int, str] = {}
    for i in range(len(s_)):
        if not np.isfinite(s_[i]):
            out[i] = "photometry: fewer than 5 stars measured"
        elif trans[i] < MIN_TRANSMISSION * med:
            out[i] = f"transmission {trans[i] / med:.2f} of the median"
    return out


class FrameStore:
    """The prepared frames: one float32 ``.npy`` memmap (N, C, H, W) on disk, NaN where a frame has no data, a uint8
    memmap (N, H, W) flagging its saturated pixels (their values are kept: the initial image uses them, as the
    script's does), plus per-frame star positions (array coordinates of the store) and names."""

    def __init__(self, path: str, shape: tuple[int, int, int, int] | None = None):
        self.path = path
        sat_path = path[:-4] + "_sat.npy"
        if shape is None:
            self.data = np.load(path, mmap_mode="r")
            self.sat = np.load(sat_path, mmap_mode="r") if os.path.exists(sat_path) else None
        else:
            self.data = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=tuple(int(v) for v in shape))
            self.sat = np.lib.format.open_memmap(sat_path, mode="w+", dtype=np.uint8, shape=(int(shape[0]),) + tuple(int(v) for v in shape[2:]))
        self.stars: list[list[tuple[float, float]]] = [[] for _ in range(self.data.shape[0])]
        self.names: list[str] = [""] * self.data.shape[0]
        # each frame's exposure time (s; the stage sets them from the frames' headers): an HDR set's 60 s frame holds half a
        # 120 s frame's light without any loss of transmission
        self.exptimes: list[float] = [1.0] * self.data.shape[0]

    def __len__(self) -> int:
        return int(self.data.shape[0])

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.data.shape[1:])

    def frame(self, i: int) -> np.ndarray:
        return np.array(self.data[i], dtype=np.float32, copy=True)

    def saturated(self, i: int) -> np.ndarray:
        """(C, H, W) bool: the pixels of each channel that are censored (bit c of the stored byte; a byte without
        per-channel bits - an older store - marks every channel)."""
        C = self.data.shape[1]
        if self.sat is None:
            return np.zeros(self.data.shape[1:], bool)
        b = np.array(self.sat[i], dtype=np.uint8)
        if C == 3:
            return np.stack([(b & 1) > 0, (b & 2) > 0, (b & 4) > 0])
        return np.repeat((b > 0)[None], C, 0)

    def flush(self) -> None:
        for a in (self.data, self.sat):
            if hasattr(a, "flush"):
                a.flush()


# ------------------------------------------------------------------------------
#  PSF SOLVER
# ------------------------------------------------------------------------------

class PSFSolver:
    def __init__(self, device='cpu', kernel_size=25, tv_weight=1e-7):
        torch, _, _, _ = _torch()
        self.device = torch.device(device)
        self.k_size = kernel_size
        if self.k_size % 2 == 0:
            self.k_size += 1
        self.tv_weight = float(tv_weight)

    def extract_and_stack_stars(self, raw_data, star_coords, crop_size=CROP_SIZE):
        """Extracts stars from loaded raw_data, aligns and stacks them.  Returns (PSF tensor, stars used).

        PIPELINE: ``star_coords`` are (column, row) of the array (``siril_to_array``, change 3), cut at the nearest
        pixel; a cut-out with a pixel without data (NaN: saturated, outside the frame) is skipped (change 5)."""
        torch, _, _, _ = _torch()
        if raw_data.ndim == 3:
            data_2d = raw_data[1] if raw_data.shape[0] > 1 else raw_data[0]
        else:
            data_2d = raw_data

        h_img, w_img = data_2d.shape
        crops = []
        ann_mask = ap_mask = None
        margin = crop_size // 2 + 1

        for (x, y) in star_coords:
            ix, iy = int(round(x)), int(round(y))
            # (2 px more each side for the sub-pixel shift)
            if (margin + 2 < ix < w_img - margin - 2) and (margin + 2 < iy < h_img - margin - 2):
                big = data_2d[iy - margin - 2:iy + margin + 3, ix - margin - 2:ix + margin + 3]
                if not np.isfinite(big).all():
                    continue
                # PIPELINE (change 10): aligned to the star's fitted centre (findstar's sub-pixel position) by a
                # bicubic shift, as the docstring says ("aligns and stacks"); the script cut at the integer pixel and
                # the +-0.5 px scatter blurred the stacked PSF (IC 1805's donut: its ring 13-15 % of the peak too faint)
                m_ = np.float32([[1, 0, x - ix], [0, 1, y - iy]])
                big = cv2.warpAffine(np.ascontiguousarray(big, np.float32), m_, (big.shape[1], big.shape[0]),
                                     flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
                cutout = big[2:-2, 2:-2].copy()
                # PIPELINE (change 17): the other stars inside the box are masked (an empirical PSF's standard
                # practice): with 81 px boxes in a cluster most cut-outs hold neighbours, which lifted the kernel's
                # outer wings and the restoration dug a moat round the bright stars to match (IC 1805)
                n_ = cutout.shape[0]
                for (ox, oy) in star_coords:
                    dx_, dy_ = ox - x, oy - y
                    if (dx_ or dy_) and abs(dx_) < margin + NEIGHBOUR_R and abs(dy_) < margin + NEIGHBOUR_R:
                        cy_, cx_ = margin + dy_, margin + dx_
                        yy_, xx_ = np.ogrid[:n_, :n_]
                        cutout[(yy_ - cy_) ** 2 + (xx_ - cx_) ** 2 <= NEIGHBOUR_R ** 2] = np.nan
                # PIPELINE (change 19): background from an annulus and normalisation by the aperture flux (the
                # empirical-PSF practice, e.g. photutils EPSFBuilder); the script subtracted the box edge's median and
                # divided by the sum over the whole box, where a faint star's sum is mostly background error - its
                # cut-out entered the median with a random scale, which lifted the kernel's wings 2-3 x above the stars'
                # (IC 1805, 10-18 px) and the restoration dug a moat round every star to match
                if ann_mask is None:
                    yy0, xx0 = np.mgrid[:cutout.shape[0], :cutout.shape[1]] - margin
                    r0 = np.hypot(yy0, xx0)
                    ann_mask, ap_mask = (r0 >= BG_ANNULUS[0]) & (r0 <= BG_ANNULUS[1]), r0 <= AP_RADIUS
                bgv = cutout[ann_mask]
                bgv = bgv[np.isfinite(bgv)]
                if bgv.size < 50:
                    continue
                bg = np.median(bgv)
                sig = 1.4826 * np.median(np.abs(bgv - bg))
                cutout = cutout - bg
                ap = cutout[ap_mask]
                fin = np.isfinite(ap)
                # (a Bayer-drizzled frame misses scattered pixels - no drop reached them: up to 10 % of the aperture
                # is allowed, its flux the mean of the rest; a saturated core, masked as a whole, is not)
                core = cutout[margin - 2:margin + 3, margin - 2:margin + 3]
                if fin.mean() < 0.9 or np.isfinite(core).mean() < 0.7:
                    continue
                flux = float(np.mean(ap[fin]) * ap.size)
                if flux <= 0 or flux < MIN_STAR_SNR * sig * np.sqrt(ap.size):
                    continue
                crops.append(cutout / flux)

        if not crops:
            return self.generate_gaussian_prior(sigma=2.0, size=2 * margin + 1).squeeze(), 0

        stack = np.nanmedian(np.array(crops), axis=0)
        stack = np.where(np.isfinite(stack), stack, 0.0)
        r_s = self._detected_radius(stack, len(crops))
        self.last_radius = r_s
        stack = self._radial_wings(stack)
        stack = self._round_support(stack, r_s)
        stack = np.maximum(stack, 0)
        stack /= np.sum(stack)
        return torch.from_numpy(stack.astype(np.float32)).float().to(self.device), len(crops)

    @staticmethod
    def _radial_wings(stack: np.ndarray) -> np.ndarray:
        """PIPELINE (change 9): the stacked PSF where single pixels are noise, replaced by its azimuthal mean.

        Beyond ~10 px a pixel of the median stack is mostly noise (IC 1805, 40-50 stars: sd 1.0e-4 per pixel against
        a profile of 2-7e-5), and the script's ``np.maximum(stack, 0)`` turns that noise into a positive floor: 4-5 %
        of the kernel's light spread evenly over the 33 x 33 box, which no star has - the restoration then dug a moat
        round every star to match it.  Each pixel is blended towards the mean of its ring (the same radius, ~2 pi r
        pixels, unbiased) by its signal-to-noise: kept where the ring mean is >= 3 sigma, the ring mean where it is
        <= 1 sigma (sigma = the robust scatter about the ring means in the outer annulus)."""
        n = stack.shape[0]
        c = n // 2
        yy, xx = np.mgrid[:n, :n]
        rr = np.hypot(yy - c, xx - c)
        rb = np.rint(rr).astype(int)
        # (the ring's median: a neighbour left in the stack must not lift its ring)
        prof = np.array([np.median(stack[rb == k]) if np.any(rb == k) else 0.0 for k in range(int(rb.max()) + 1)])
        ring = prof[rb]
        outer = rr >= c - 3
        dev = (stack - ring)[outer]
        sigma = 1.4826 * float(np.median(np.abs(dev - np.median(dev))))
        if not np.isfinite(sigma) or sigma <= 0:
            return stack
        w = np.clip((ring / sigma - 1.0) / 2.0, 0.0, 1.0)
        return w * stack + (1.0 - w) * ring

    @staticmethod
    def _detected_radius(stack: np.ndarray, n_stars: int, k: float = 3.0, min_r: int = 8) -> float:
        """The radius out to which the stacked PSF is detected: the first ring (beyond ``min_r``) of three in a row
        whose median is below ``k`` x its standard error (1.2533 x the ring's robust scatter / sqrt(pixels))."""
        n = stack.shape[0]
        c = n // 2
        yy, xx = np.mgrid[:n, :n]
        rb = np.rint(np.hypot(yy - c, xx - c)).astype(int)
        low = 0
        for r_ in range(min_r, c + 1):
            v = stack[rb == r_]
            med = np.median(v)
            se = 1.2533 * 1.4826 * np.median(np.abs(v - med)) / np.sqrt(v.size)
            low = low + 1 if med < k * se else 0
            if low == 3:
                return float(r_ - 2)
        return float(c)

    @staticmethod
    def _round_support(stack: np.ndarray, r_s: float | None = None, taper: float = 4.0) -> np.ndarray:
        """PIPELINE (change 16): the kernel ends where the PSF is no longer detected.  Beyond that radius the stack
        holds only a residual floor (neighbours, nebula, background error; IC 1805: ~0.001 of the peak over 20-40 px,
        5 % of a star's light), which made the model put light in a wide annulus round every star - a square at first
        (81 px box), then a moat at 20-35 px after the restoration matched it.  The profile is tapered to 0 over
        ``taper`` px from the detected radius ``r_s`` (``_detected_radius``; the box's inscribed circle without one)."""
        n = stack.shape[0]
        c = n // 2
        yy, xx = np.mgrid[:n, :n]
        rr = np.hypot(yy - c, xx - c)
        R = float(c) if r_s is None else min(float(c), float(r_s) + taper)
        w = np.clip((R - rr) / taper, 0.0, 1.0)
        w = 0.5 - 0.5 * np.cos(np.pi * w)
        return stack * w

    def generate_gaussian_prior(self, sigma=0.5, size=None):
        """Generates g_sigma. If size is None, uses self.k_size."""
        torch, _, _, _ = _torch()
        k = size if size is not None else self.k_size
        if k % 2 == 0:
            k += 1

        coords = torch.arange(k, dtype=torch.float32, device=self.device)
        coords -= (k - 1) / 2.0
        y, x = torch.meshgrid(coords, coords, indexing='ij')

        g = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        g /= torch.sum(g)
        return g.unsqueeze(0).unsqueeze(0)

    def _fit(self, forward, target, h0, iterations, tv_weight):
        """argmin_{h >= 0} mse(forward(h), target) + tv_weight * TV(h), normalised to unit sum.

        PIPELINE (change 8): the script minimises the same objective with Adam (lr 0.005) on ReLU(h).  The PSF's
        wing pixels are ~1e-4 of a unit-sum kernel: Adam's first steps (~ +-lr) push them below zero, where ReLU's
        gradient vanishes, and they never come back (IC 1805, frame 0: the kernel kept 0.38 / 0.12 / 0 x 1e-3 of the
        observed 1.78 / 0.82 / 0.35 x 1e-3 at 6 / 7 / 8 px and a core 27 % too bright, residual 5 % of the peak with
        400 or 4000 steps), which rang round every star.  The problem is a convex non-negative least squares: solved
        here by projected FISTA (Beck & Teboulle 2009) with the step 1 / L, L from power iteration on the operator.
        TV is the smooth (Charbonnier) form sqrt(d^2 + eps^2), eps = TV_EPS, its Lipschitz bound 8 tv_weight / eps
        added to L: as a plain (sub)gradient step the |d| term made the kernel's wings chatter (a synthetic donut:
        0 to 500 x the true wing at 10-14 px)."""
        torch, _, F, _ = _torch()

        eps = TV_EPS

        def objective(h):
            dx, dy = h[..., :, :-1] - h[..., :, 1:], h[..., :-1, :] - h[..., 1:, :]
            tv = torch.sum(torch.sqrt(dx * dx + eps * eps)) + torch.sum(torch.sqrt(dy * dy + eps * eps))
            return F.mse_loss(forward(h), target) + tv_weight * tv

        def grad(h):
            h = h.detach().requires_grad_(True)
            g, = torch.autograd.grad(objective(h), h)
            return g

        gen = torch.Generator(device="cpu").manual_seed(0)
        v = torch.rand(h0.shape, generator=gen).to(h0.device)
        lip = 1.0
        for _ in range(30):                       # largest eigenvalue of A^T A
            v = v.detach().requires_grad_(True)
            av, = torch.autograd.grad(0.5 * torch.sum(forward(v) ** 2), v)
            lip = float(av.detach().norm() / v.detach().norm())
            v = (av / av.norm()).detach()
        step = 1.0 / (1.05 * lip * 2.0 / target.numel() + 8.0 * tv_weight / eps)
        h = torch.clamp(h0.detach().clone(), min=0.0)
        h_prev, t = h.clone(), 1.0
        for _ in range(iterations):
            t_next = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
            y = h + ((t - 1.0) / t_next) * (h - h_prev)
            h_prev = h
            h = torch.clamp(y - step * grad(y), min=0.0)
            t = t_next
        h = h.detach()
        return h / torch.sum(h, dim=(-2, -1), keepdim=True)

    def solve(self, observed_psf_tensor, sigma=0.45, iterations=400):
        """Native resolution solver (Eq. 2 context)."""
        torch, nn, F, optim = _torch()
        if observed_psf_tensor.ndim == 2:
            target = observed_psf_tensor.unsqueeze(0).unsqueeze(0)
        elif observed_psf_tensor.ndim == 3:                          # (C, k, k): the channels solved together
            target = observed_psf_tensor.unsqueeze(0)
        else:
            target = observed_psf_tensor

        g = self.generate_gaussian_prior(sigma)
        nc = target.shape[1]
        if nc > 1:
            g = g.repeat(nc, 1, 1, 1)
        pad = target.shape[-1] // 2

        def forward(h):
            modelled_psf = F.conv2d(h, g, padding=pad, groups=nc)
            if modelled_psf.shape[-1] != target.shape[-1]:
                diff = modelled_psf.shape[-1] - target.shape[-1]
                p1 = diff // 2
                p2 = diff - p1
                modelled_psf = modelled_psf[:, :, p1:-p2, p1:-p2]
            return modelled_psf

        # (the MSE is a mean over all channels' pixels: TV / nc keeps a batched solve equal to separate ones)
        return self._fit(forward, target, target, iterations, self.tv_weight / nc)

    def solve_sr(self, native_psf, r, sigma=1.0, iterations=500):
        """
        Solves Eq. 11 for Super-Resolution PSF.
        h = argmin || f - D(h * g_sigma) ||^2
        """
        torch, nn, F, optim = _torch()
        if native_psf.ndim == 2:
            target = native_psf.unsqueeze(0).unsqueeze(0)
        elif native_psf.ndim == 3:
            target = native_psf.unsqueeze(0)
        else:
            target = native_psf

        native_size = target.shape[-1]
        hr_size = native_size * r
        if hr_size % 2 == 0:
            hr_size += 1

        h_init = F.interpolate(target, size=(hr_size, hr_size), mode='bilinear', align_corners=False)

        g_size = int(math.ceil(4 * sigma)) * 2 + 1
        g = self.generate_gaussian_prior(sigma, size=g_size)
        if target.shape[1] > 1:
            g = g.repeat(target.shape[1], 1, 1, 1)
        pad_g = g_size // 2

        def forward(h):
            blurred_hr = F.conv2d(h, g, padding=pad_g, groups=target.shape[1])
            predicted_lr = F.avg_pool2d(blurred_hr, kernel_size=r, stride=r)
            if predicted_lr.shape[-1] != target.shape[-1]:
                diff = predicted_lr.shape[-1] - target.shape[-1]
                if diff > 0:
                    p1 = diff // 2
                    predicted_lr = predicted_lr[:, :, p1: p1 + target.shape[-1], p1: p1 + target.shape[-1]]
            return predicted_lr

        return self._fit(forward, target, h_init, iterations, 0.0)


def sr_origin(r: int, native_size: int = 2 * (CROP_SIZE // 2 + 1) + 1) -> int:
    """t of the latent grid: latent pixel u (super-resolved by r) lies at frame pixel (u - t) / r.

    ``solve_sr`` fits the r x kernel through ``avg_pool2d`` (frame pixel j = latent pixels r j .. r j + r - 1), so its
    light is centred on frame pixel c = native_size // 2, i.e. on latent coordinate r c + (r - 1) / 2, while the
    convolution puts the kernel's array centre o = hr_size // 2 on the source pixel: a point at latent u reaches frame
    coordinate (u + r c - o) / r.  t = o - r c (1 for r = 2 and 3, 0 at r = 1)."""
    if r <= 1:
        return 0
    hr = native_size * r
    hr += 1 if hr % 2 == 0 else 0
    return hr // 2 - r * (native_size // 2)


# ------------------------------------------------------------------------------
#  FFT CONVOLUTION (PIPELINE, change 2)
# ------------------------------------------------------------------------------

def _fast_len(n: int) -> int:
    """The smallest 2^a 3^b 5^c >= n (fast cuFFT / pocketfft sizes)."""
    best = 1 << (int(n) - 1).bit_length()
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            p = p35
            while p < n:
                p *= 2
            best = min(best, p)
            p35 *= 3
        p5 *= 5
    return best


class FFTConv:
    """Linear convolution / correlation of (B, C, H, W) images with a (k, k) kernel via the FFT, with the script's
    borders: the image is padded by k // 2 (``reflect`` practical, ``constant`` formal) and the centre is kept, which
    is exactly ``F.pad`` + a valid convolution (the padded size is never wrapped).  Both operators are linear in the
    image, so the adjoint terms of many frames add up in Fourier space and are transformed back once."""

    def __init__(self, H: int, W: int, k: int, mode: str, device):
        self.H, self.W, self.R = int(H), int(W), int(k) // 2
        self.k = 2 * self.R + 1
        self.mode = mode
        self.device = device
        self.L = (_fast_len(self.H + 2 * self.R), _fast_len(self.W + 2 * self.R))

    def rfft(self, a):
        torch, _, F, _ = _torch()
        R = self.R
        if R:
            mode = self.mode
            if mode == "reflect" and (R >= a.shape[-1] or R >= a.shape[-2]):
                mode = "replicate"
            a = F.pad(a, (R, R, R, R), mode=mode)
        return torch.fft.rfft2(a, s=self.L)

    def kernel_hat(self, kernel):
        """rfft of a (k, k) kernel (or (..., k, k)) placed with its centre on the origin."""
        torch, _, _, _ = _torch()
        k = kernel.shape[-1]
        if k != self.k:
            raise ValueError(f"kernel {k} x {k}, the operator was built for {self.k}")
        z = torch.zeros(kernel.shape[:-2] + self.L, dtype=torch.float32, device=kernel.device)
        z[..., :k, :k] = kernel
        z = torch.roll(z, shifts=(-self.R, -self.R), dims=(-2, -1))
        return torch.fft.rfft2(z)

    def back(self, spec):
        torch, _, _, _ = _torch()
        out = torch.fft.irfft2(spec, s=self.L)
        return out[..., self.R:self.R + self.H, self.R:self.R + self.W]

    def convolve(self, a_hat, k_hat):
        """h * x: a point source at p renders the kernel centred on p (the measured star)."""
        return self.back(a_hat * k_hat)

    def correlate_hat(self, a, k_hat):
        """The Fourier transform of the adjoint (correlation) term of ``a``, to be summed and passed to ``back``."""
        return self.rfft(a) * k_hat.conj()


# ------------------------------------------------------------------------------
#  IMAGEMM MATHEMATICAL CORE
# ------------------------------------------------------------------------------

class ImageMMEngine:
    def __init__(self, device_name='cpu', formal_mode=False):
        torch, _, _, _ = _torch()
        self.device = torch.device(device_name)
        self.formal_mode = bool(formal_mode)
        # PIPELINE (change 20): no TV in the PSF fit.  The script's practical-mode 1e-7 is not "micro" next to the MSE
        # of a unit-sum 81 x 81 kernel: it smoothed the core's edge out into the wings (IC 1805: 0.0062 -> 0.0085 of the
        # peak at 10 px, the stars 0.0051); the wing noise is handled by _radial_wings
        tvw = 0.0
        self.psf_solver = PSFSolver(device=device_name, tv_weight=tvw)
        self.last_run_info: dict = {}

    def downsample(self, hr_tensor, r):
        """Operator D: Average Pooling (Paper-Faithful)."""
        _, _, F, _ = _torch()
        if r <= 1:
            return hr_tensor
        return F.avg_pool2d(hr_tensor, kernel_size=r, stride=r)

    def upsample_adjoint(self, lr_tensor, r):
        """Operator D^T: scaled repeat (adjoint of average pooling)."""
        if r <= 1:
            return lr_tensor
        b, c, h, w = lr_tensor.shape
        out = lr_tensor.reshape(b, c, h, 1, w, 1).expand(b, c, h, r, w, r).reshape(b, c, h * r, w * r)
        return out / (r * r)

    def huber_weight(self, residual, delta=0.05):
        """Robust Huber Weights."""
        torch, _, _, _ = _torch()
        abs_r = torch.abs(residual)
        return torch.where(abs_r <= delta, torch.ones_like(abs_r), delta / (abs_r + 1e-9))

    @staticmethod
    def _robust_bg_sigma(img_np):
        """Robust background sigma estimate via MAD, ignoring black borders (0)."""
        arr = img_np
        if arr.ndim == 3:
            arr = arr[1] if arr.shape[0] > 1 else arr[0]
        valid_pixels = arr[arr > 1e-7]
        if valid_pixels.size == 0:
            return 1e-6
        med = np.median(valid_pixels)
        mad = np.median(np.abs(valid_pixels - med))
        return float(max(1.4826 * mad, 1e-6))

    def _mask_and_norm(self, d, norm_factor, sat_norm, saturated=None, idx=None):
        """build_mask_and_variance's mask and d_norm on the GPU.  PIPELINE (changes 4, 5): m = the finite pixels of
        the store (NaN = no data / saturated); the explicit saturation level, when given, as in the script."""
        torch, _, _, _ = _torch()
        finite = torch.isfinite(d)
        d_norm = torch.clamp(torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0), min=0.0) / float(max(norm_factor, 1e-6))
        if idx is not None and getattr(self, "_norm", None) is not None:
            scale_t = self._norm[1]
            pl = torch.tensor(self._planes[idx], dtype=d_norm.dtype, device=d_norm.device)
            X, Y = self._grid
            P = pl[:, 0, None, None] + pl[:, 1, None, None] * X[None] + pl[:, 2, None, None] * Y[None]
            d_norm = torch.clamp((d_norm - P[None]) / float(scale_t[idx]), min=0.0)
        m = finite
        if saturated is not None:
            m = m & ~saturated
        if sat_norm is not None:
            m = m & (d_norm < sat_norm)
        return m.to(d_norm.dtype), d_norm

    @staticmethod
    def _halo_profile(ref: np.ndarray, catalog, R: int = HALO_R, iso: float = 40.0, mask_r: float = 12.0):
        """(fraction of a star's light per 1 px ring, r = 0..R; stars used) from the brightest isolated catalogue stars of
        the frames' mean image: other stars masked (``mask_r``), sky the median at R-10..R, each star normalised to its
        total within R, the median over stars per ring.  None with fewer than 3 stars."""
        H, W = ref.shape
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        rr = np.hypot(yy, xx)
        rb = np.rint(rr).astype(int)
        cnt = np.bincount(rb.ravel(), minlength=R + 2)[:R + 1]
        pts = [(x, y) for x, y in catalog if R + 2 < x < W - R - 2 and R + 2 < y < H - R - 2]
        bright = []
        for x, y in pts:
            ix, iy = int(round(x)), int(round(y))
            c = ref[iy - 6:iy + 7, ix - 6:ix + 7]
            if np.isfinite(c).all():
                bright.append((float(c.sum()), x, y))
        bright.sort(reverse=True)
        profs = []
        for _, x, y in bright[:60]:
            if any(0 < (x - a) ** 2 + (y - b) ** 2 < iso ** 2 for a, b in catalog):
                continue
            ix, iy = int(round(x)), int(round(y))
            c = ref[iy - R:iy + R + 1, ix - R:ix + R + 1].astype(np.float64).copy()
            for a, b in catalog:
                if (a, b) != (x, y) and abs(a - x) < R + mask_r and abs(b - y) < R + mask_r:
                    c[(yy - (b - y)) ** 2 + (xx - (a - x)) ** 2 <= mask_r ** 2] = np.nan
            sky = np.nanmedian(c[(rr >= R - 10) & (rr <= R)])
            c = c - sky
            if not np.isfinite(c[rr <= 6]).all():
                continue
            ring = np.array([np.nanmean(c[rb == k]) if np.isfinite(c[rb == k]).any() else np.nan for k in range(R + 1)]) * cnt
            tot = np.nansum(ring)
            if tot > 0:
                profs.append(ring / tot)
            if len(profs) >= 25:
                break
        if len(profs) < 3:
            return None
        P = np.nanmedian(np.array(profs), 0)
        P = np.where(np.isfinite(P), np.maximum(P, 0.0), 0.0)
        return P / P.sum(), len(profs)

    @staticmethod
    def _attach_halo(k, prof: np.ndarray, r_in: float = 30.0, r_out: float = 40.0):
        """The frame's measured kernel (centre, < r_out) joined to the static halo profile ``prof`` (light per ring, out
        to len(prof) - 1 px): w k_scaled + (1 - w) halo, w 1 inside r_in and a raised cosine to 0 at r_out, the core
        scaled to the halo's light inside r_in; unit sum."""
        torch, _, _, _ = _torch()
        kk = k[0, 0] if k.ndim == 4 else k
        kn = kk.detach().cpu().numpy().astype(np.float64)
        R = len(prof) - 1
        n = 2 * R + 1
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        rr = np.hypot(yy, xx)
        rb = np.rint(rr).astype(int)
        cnt = np.bincount(rb.ravel(), minlength=R + 1)[:R + 1]
        halo = np.where(rb <= R, prof[np.minimum(rb, R)] / np.maximum(cnt[np.minimum(rb, R)], 1), 0.0)
        core = np.zeros((n, n))
        h_ = kn.shape[0] // 2
        core[R - h_:R + h_ + 1, R - h_:R + h_ + 1] = kn
        w = np.clip((r_out - rr) / (r_out - r_in), 0.0, 1.0)
        w = 0.5 - 0.5 * np.cos(np.pi * w)
        inner = rr <= r_in
        core *= halo[inner].sum() / max(core[inner].sum(), 1e-12)
        K = np.maximum(w * core + (1.0 - w) * halo, 0.0)
        K /= K.sum()
        out = torch.from_numpy(K.astype(np.float32)).to(kk.device)
        return out[None, None] if k.ndim == 4 else out

    @staticmethod
    def _plane_fit(r: np.ndarray, tile: int = 64) -> np.ndarray:
        """(a, b, c) of the plane a + b X + c Y (X, Y: column / row offsets from the centre in units of the width /
        height) through the medians of ``tile`` x ``tile`` tiles of ``r`` (at least half finite), one 3-sigma clip."""
        H, W = r.shape
        xs, ys, vs = [], [], []
        for y0 in range(0, H - tile // 2, tile):
            for x0 in range(0, W - tile // 2, tile):
                t = r[y0:y0 + tile, x0:x0 + tile]
                f = np.isfinite(t)
                if f.mean() < 0.2:                     # (the R / B planes of an undithered Bayer drizzle: 25 % data)
                    continue
                vs.append(float(np.median(t[f])))
                xs.append((x0 + t.shape[1] / 2 - W / 2) / W)
                ys.append((y0 + t.shape[0] / 2 - H / 2) / H)
        if len(vs) < 3:
            return np.array([float(np.nanmedian(r)) if np.isfinite(r).any() else 0.0, 0.0, 0.0])
        A = np.c_[np.ones(len(vs)), xs, ys]
        v = np.array(vs)
        k = np.linalg.lstsq(A, v, rcond=None)[0]
        d = v - A @ k
        keep = np.abs(d) <= 3 * 1.4826 * np.median(np.abs(d)) + 1e-12
        if keep.sum() >= 3:
            k = np.linalg.lstsq(A[keep], v[keep], rcond=None)[0]
        return k

    @staticmethod
    def _photon_transfer(cube: np.ndarray, sky_t: np.ndarray, ref_img: np.ndarray, pick: np.ndarray, scale_t: np.ndarray | None = None) -> dict | None:
        """var = a + slope x (level - sky) per channel from the frames themselves: at a fixed sample of pixels, the
        temporal median (level) and robust scatter (variance) over the frames, each frame's sky removed, on smooth
        pixels (the mean image's gradient in the lower half: no star edges, whose seeing changes are not shot noise);
        a robust line through 15 level bins.  IC 1805 (2026-10-21): slope 1.7-1.9e-5 in [0, 1] units = ~0.85 e-/DN.

        ``scale_t``: each frame's scale s_t onto the common one (cube = y / s_t).  A frame's shot variance there is
        slope x level / s_t (the run's variance model divides by s_t the same way), so each frame's deviation from the
        median is weighted by sqrt(s_t) and every frame measures the same slope: pooled unweighted, an HDR set's
        60 s + 120 s frames (s_t 0.5 / 1) mixed two variances into one slope."""
        N, C, n = cube.shape
        w_t = np.sqrt(np.clip(np.asarray(scale_t, np.float64), 1e-6, None)).astype(np.float32)[:, None] if scale_t is not None else np.ones((N, 1), np.float32)
        H, W = ref_img.shape
        g = cv2.GaussianBlur(np.nan_to_num(ref_img), (0, 0), 1.0)
        grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)).ravel()[pick]
        smooth = grad < np.nanpercentile(grad, 50)
        slopes, inter = [], []
        for c in range(C):
            x = cube[:, c] - sky_t[:, c][:, None].astype(np.float32)
            med = np.nanmedian(x, 0)
            var = (1.4826 * np.nanmedian(np.abs((x - med[None]) * w_t), 0)) ** 2
            ok = smooth & np.isfinite(med) & np.isfinite(var) & (np.sum(np.isfinite(x), 0) >= max(5, N // 2))
            if ok.sum() < 500:
                return None
            lv, vv = med[ok], var[ok]
            edges = np.percentile(lv, np.linspace(1, 99.5, 16))
            bx, by = [], []
            for a, b in zip(edges[:-1], edges[1:]):
                sel = (lv >= a) & (lv < b)
                if sel.sum() > 30:
                    bx.append(np.median(lv[sel])); by.append(np.median(vv[sel]))
            if len(bx) < 4:
                return None
            k = np.polyfit(bx, by, 1)
            slopes.append(float(max(k[0], 0.0)))
            inter.append(float(k[1]))
        return {"slope": slopes, "intercept": inter}

    @staticmethod
    def _reference_catalog(ref: np.ndarray, store, m: int = 7, isolation: float = 6.0) -> list[tuple[float, float]]:
        """Star positions on the latent grid: findstar's stars (of the frame with the most) re-centroided on the
        frames' mean image (an iterated Gaussian-weighted centroid, sigma 3 px: robust to a ring-shaped PSF), with no
        other star within ``isolation`` px.  Saturated stars stay in: each frame's saturation flags drop their cut-outs."""
        H, W = ref.shape
        src = max(store.stars, key=len) if len(store) else []
        yy, xx = np.mgrid[-m:m + 1, -m:m + 1]
        out = []
        for x, y in src:
            ix, iy = int(round(x)), int(round(y))
            if not (m + 12 < ix < W - m - 12 and m + 12 < iy < H - m - 12):
                continue
            c = ref[iy - m:iy + m + 1, ix - m:ix + m + 1].astype(np.float64)
            if not np.isfinite(c).all():
                continue
            w = np.maximum(c - np.median(ref[iy - m - 10:iy + m + 11, ix - m - 10:ix + m + 11]), 0)
            if w.sum() <= 0:
                continue
            cx = cy = 0.0
            for _ in range(5):
                wt = w * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 18.0)
                st_ = wt.sum()
                if st_ <= 0:
                    break
                cx, cy = float((wt * xx).sum() / st_), float((wt * yy).sum() / st_)
            if abs(cx) < 2 and abs(cy) < 2:
                out.append((ix + cx, iy + cy))
        keep = []
        for i, (x, y) in enumerate(out):
            if all((x - a) ** 2 + (y - b) ** 2 > isolation ** 2 for j, (a, b) in enumerate(out) if j != i):
                keep.append((x, y))
        return keep

    @staticmethod
    def _aperture_fluxes(img: np.ndarray, catalog, r_ap: float = 8.0, r_in: float = 12.0, r_out: float = 18.0) -> np.ndarray:
        """Aperture photometry at the catalogue positions (sky: the annulus median); NaN where a pixel is missing."""
        H, W = img.shape
        R = int(np.ceil(r_out))
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        out = np.full(len(catalog), np.nan)
        for k, (x, y) in enumerate(catalog):
            ix, iy = int(round(x)), int(round(y))
            if not (R < ix < W - R - 1 and R < iy < H - R - 1):
                continue
            c = img[iy - R:iy + R + 1, ix - R:ix + R + 1]
            rr = np.hypot(xx - (x - ix), yy - (y - iy))
            ap, an = rr <= r_ap, (rr >= r_in) & (rr <= r_out)
            fin = np.isfinite(c[ap])
            if fin.mean() < 0.9:                       # (scattered drizzle holes: up to 10 %, the mean of the rest)
                continue
            sky = np.nanmedian(c[an])
            out[k] = float(np.mean(c[ap][fin] - sky) * ap.sum())
        return out

    @staticmethod
    def _nonradial_psf(ref: np.ndarray, catalog, n: int = 81, min_snr: float = 100.0, mask_r: float = 8.0,
                       r_max: float = 16.0, taper: float = 4.0, r_in: float = 4.0, r_full: float = 7.0):
        """(the non-radial part of the PSF, (n, n), zero sum in every 1 px ring; stars; dipole at 5-15 px; its angle) from
        the bright unsaturated stars of the frames' mean image ``ref`` (NaN where a frame was saturated / missing): each
        cut-out sub-pixel centred, its background the median at 25-38 px, other stars masked (``mask_r``), normalised by
        its 8 px aperture flux; the flux-weighted mean over the stars with aperture SNR >= ``min_snr``, then minus its ring
        means, tapered to 0 from ``r_max`` (beyond, the mean image's light is mostly other sources).  None with < 5 stars."""
        H, W = ref.shape
        m = n // 2
        yy, xx = np.mgrid[:n, :n] - m
        rr = np.hypot(yy, xx)
        rb = np.rint(rr).astype(int)
        ann, ap = (rr >= BG_ANNULUS[0]) & (rr <= BG_ANNULUS[1]), rr <= AP_RADIUS
        acc = np.zeros((n, n)); wsum = np.zeros((n, n)); used = 0
        for (x, y) in catalog:
            ix, iy = int(round(x)), int(round(y))
            if not (m + 3 < ix < W - m - 3 and m + 3 < iy < H - m - 3):
                continue
            big = ref[iy - m - 2:iy + m + 3, ix - m - 2:ix + m + 3].astype(np.float32)
            if not np.isfinite(big[m - 6:m + 11, m - 6:m + 11]).all():
                continue                                   # (a saturated or incomplete core)
            fin = np.isfinite(big)
            sh = np.float32([[1, 0, x - ix], [0, 1, y - iy]])
            cut = cv2.warpAffine(np.where(fin, big, 0).astype(np.float32), sh, (big.shape[1], big.shape[0]), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)[2:-2, 2:-2]
            ok = cv2.warpAffine(fin.astype(np.float32), sh, (big.shape[1], big.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)[2:-2, 2:-2] > 0.99
            for (ox, oy) in catalog:
                dx_, dy_ = ox - x, oy - y
                if (dx_ or dy_) and abs(dx_) < m + mask_r and abs(dy_) < m + mask_r:
                    ok &= (yy - dy_) ** 2 + (xx - dx_) ** 2 > mask_r ** 2
            bgv = cut[ann & ok]
            if bgv.size < 50:
                continue
            bg = np.median(bgv)
            sig = 1.4826 * np.median(np.abs(bgv - bg))
            c_ = cut - bg
            flux = float(c_[ap].sum())
            if flux <= 0 or flux < min_snr * sig * np.sqrt(ap.sum()):
                continue
            acc += np.where(ok, c_, 0.0)                   # (flux-weighted mean of the unit-flux profiles)
            wsum += np.where(ok, flux, 0.0)
            used += 1
        if used < 5:
            return None
        A = np.where(wsum > 0, acc / np.maximum(wsum, 1e-30), 0.0)
        th = np.arctan2(yy, xx)
        # only the low angular orders (m = 1, 2) of each ring, smoothed in radius: the optics' lopsided lobe.  The full
        # residual of 30-40 stars also held their faint neighbours and noise, and every lump of it came back inverted
        # round every bright star (dark blobs on one side, M 31 2026-10-22)
        kmax = int(r_max + taper) + 1
        coef = np.zeros((kmax + 1, 4))
        for k in range(1, kmax + 1):
            sel = rb == k
            if sel.sum() < 6:
                continue
            t_, v_ = th[sel], A[sel] - A[sel].mean()
            B = np.c_[np.cos(t_), np.sin(t_), np.cos(2 * t_), np.sin(2 * t_)]
            coef[k] = np.linalg.lstsq(B, v_, rcond=None)[0]
        from scipy.ndimage import gaussian_filter1d
        coef = gaussian_filter1d(coef, 1.5, axis=0, mode="nearest")
        coef[0] = 0.0
        rr_c = np.clip(rr, 0, kmax)
        ci = np.stack([np.interp(rr_c.ravel(), np.arange(kmax + 1), coef[:, j]).reshape(rr.shape) for j in range(4)])
        N_ = ci[0] * np.cos(th) + ci[1] * np.sin(th) + ci[2] * np.cos(2 * th) + ci[3] * np.sin(2 * th)
        w = np.clip((r_max + taper - rr) / taper, 0.0, 1.0)
        # (inside r_in the frame's own kernel is measured at high SNR, its asymmetry included: the static part only fills
        # the wings the frame's stack made radial - added in the core too it overcorrected, a dark arc on the far side)
        wi = np.clip((rr - r_in) / (r_full - r_in), 0.0, 1.0)
        N_ = N_ * (0.5 - 0.5 * np.cos(np.pi * w)) * (0.5 - 0.5 * np.cos(np.pi * wi)) * (rr <= r_max + taper)
        a5 = (rr >= 5) & (rr <= 15)
        zz = (A[a5] * np.exp(1j * th[a5])).sum()
        dip = abs(zz) / max(np.abs(A[a5]).sum(), 1e-30)
        return N_, used, dip, float(np.degrees(np.angle(zz)))

    @staticmethod
    def _radial_wing_profile(ref: np.ndarray, catalog, R: int = 40, sky_w: int = 4, min_snr: float = 5.0, mask_r: float = 6.0,
                             bloom: int = 0, r_norm: int = 8, fit_w: int = 5, flat: float = 0.005):
        """(the PSF's mean light per pixel at radius r = 0..R, 1 at ``r_norm``; stars) from the frames' mean image ``ref``
        (NaN where a frame was saturated / missing): an extended PSF joined from stars of different brightness (Infante-
        Sainz, Trujillo & Roman 2020, MNRAS 491, 5317).  Per star (other catalogue stars masked) the median of each 1 px
        ring minus the sky at R - ``sky_w`` .. R, with its standard error.  Unsaturated stars give the profile out from the
        centre, normalised at ``r_norm``; a saturated star only beyond its saturated core + ``bloom`` px (charge spill), scaled
        to the profile over its first ``fit_w`` valid rings.  Each ring is the inverse-variance weighted mean over the stars
        (3-sigma clipped): far out only the bright stars measure the wing.  Stars on a curved sky (the sky's median changes
        between R - sky_w .. R and the ring 2 sky_w further out by more than 4 standard errors and ``flat`` x the star's light
        at ``r_norm``; a galaxy's bulge under a star lifted the far wing 1.7 x - M 31 2026-10-22, the kernel's excess at
        16-18 px left a disc edge round every bright star) are left out.
        None with fewer than 5 unsaturated stars."""
        H, W = ref.shape
        Rx = R + 2 * sky_w                                              # (a second sky ring outside the first)
        yy, xx = np.mgrid[-Rx:Rx + 1, -Rx:Rx + 1]
        rr = np.hypot(yy, xx)
        rb = np.rint(rr).astype(int)
        skym = (rb >= R - sky_w) & (rb <= R)
        sky2 = (rb >= Rx - sky_w) & (rb <= Rx)
        ring_n = np.bincount(rb.ravel(), minlength=Rx + 1)[:R + 1]
        uns, sats = [], []
        for (x, y) in catalog:
            ix, iy = int(round(x)), int(round(y))
            if not (Rx < ix < W - Rx - 1 and Rx < iy < H - Rx - 1):
                continue
            w = ref[iy - Rx:iy + Rx + 1, ix - Rx:ix + Rx + 1].astype(np.float64).copy()
            hole = ~np.isfinite(w) & (rr <= 15)
            r_ex = float(rr[hole].max()) + bloom if hole.any() else -1.0
            if hole.sum() > MAX_WING_HOLE_PX:                        # (a galaxy nucleus, not a star)
                continue
            for (ox, oy) in catalog:
                dx_, dy_ = ox - x, oy - y
                if (dx_ or dy_) and abs(dx_) < Rx + mask_r and abs(dy_) < Rx + mask_r:
                    w[(yy - dy_) ** 2 + (xx - dx_) ** 2 <= mask_r ** 2] = np.nan
            sk, sk2 = w[skym], w[sky2]
            sk, sk2 = sk[np.isfinite(sk)], sk2[np.isfinite(sk2)]
            if sk.size < 30 or sk2.size < 30:
                continue
            sky = np.median(sk)
            sd = 1.4826 * np.median(np.abs(sk - sky))
            # (a plane does not move a ring's median; a bump under the star - curvature - does, and shows as a change of the
            # sky's median between the two outer rings: allowed within 4 standard errors, or below ``flat`` of the star)
            om = [sky - np.median(sk2)]
            tol = 4.0 * 1.2533 * sd * np.sqrt(1.0 / sk.size + 1.0 / sk2.size)
            p = np.full(R + 1, np.nan)
            se = np.full(R + 1, np.nan)
            for k in range(R + 1):
                if k <= r_ex:
                    continue
                v = w[rb == k]
                v = v[np.isfinite(v)]
                if v.size >= max(1, 0.5 * ring_n[k]):
                    p[k] = np.median(v) - sky
                    se[k] = 1.2533 * sd / np.sqrt(v.size)
            if r_ex < 0:
                if not np.isfinite(p[r_norm]) or p[r_norm] <= min_snr * se[r_norm] or abs(om[0]) > max(tol, flat * p[r_norm]):
                    continue
                uns.append((p / p[r_norm], se / p[r_norm]))
            else:
                sats.append((p, se, r_ex, om, tol))

        def wmean(items):
            # the inverse-variance WEIGHTED MEDIAN per ring: far out one or two bright stars carry most of the weight, and a
            # weighted mean followed whichever of them had a neighbour or a sky lump (bumps of 1.5-2 x at 14-24 px)
            out = np.full(R + 1, np.nan)
            for k in range(R + 1):
                v = np.array([q[k] for q, _ in items])
                s_ = np.array([e[k] for _, e in items])
                ok = np.isfinite(v) & np.isfinite(s_) & (s_ > 0)
                if ok.sum() < 3:
                    continue
                o_ = np.argsort(v[ok])
                vv, wt = v[ok][o_], (1.0 / s_[ok] ** 2)[o_]
                cw = np.cumsum(wt) - 0.5 * wt
                out[k] = float(np.interp(0.5 * wt.sum(), cw, vv))
            return out

        if len(uns) < 5:
            return None
        P = wmean(uns)
        joined = []
        for _ in range(3):
            joined = []
            for p, se, r_ex, om, tol in sats:
                ks = [k for k in range(int(np.floor(r_ex)) + 1, R + 1) if np.isfinite(p[k]) and np.isfinite(P[k]) and P[k] > 0][:fit_w]
                if len(ks) < 3 or np.sum(p[ks] / se[ks]) < 20:
                    continue
                a = float(np.sum(p[ks] * P[ks] / se[ks] ** 2) / np.sum(P[ks] ** 2 / se[ks] ** 2))
                if a <= 0 or abs(om[0]) > max(tol, flat * a):            # (the flat-sky test on the scaled star)
                    continue
                joined.append((p / a, se / a))
            P = wmean(uns + joined)
        P = np.where(np.isfinite(P), np.maximum(P, 0.0), 0.0)
        P = P / max(P[r_norm], 1e-30)
        # beyond r_norm smoothed in log over 1.5 px (the wing is smooth; what is left is the stars' noise); the rings inside
        # are the smoothing's left edge only - smoothed with them, the normalising ring lost 25 % and every wing was 1.3 x
        from scipy.ndimage import gaussian_filter1d
        lo = max(0, r_norm - 4)
        sm = np.exp(gaussian_filter1d(np.log(np.maximum(P[lo:R - sky_w], 1e-6)), 1.5, mode="nearest"))
        P[r_norm + 1:R - sky_w] = sm[r_norm + 1 - lo:]
        P[R - sky_w:] = 0.0                                             # (the sky ring itself)
        return P, len(uns) + len(joined)

    @staticmethod
    def _wing_chroma(ref: np.ndarray, catalog, R: int = 40, sky_w: int = 4, bloom: int = 0, j: tuple = (10, 14),
                     fit_w: int = 5, mask_r: float = 6.0, max_se: float = 0.04):
        """PIPELINE (change 33): the far wing's colour.  (C, R + 1) per channel the factor on the channels' common wing,
        1 over the join ``j``: each star's share of the light in the channel per 1 px ring (ring medians minus the sky at
        R - ``sky_w`` .. R, the frames' mean ``ref`` (C, H, W)) over its share at ``j`` - unsaturated stars directly, a
        saturated star beyond its core + ``bloom`` px scaled to the unsaturated stars' over its first ``fit_w`` valid rings
        (the join of ``_radial_wing_profile``, on colour; with the unsaturated stars too faint out there, to 1 - the join's
        share - at first) - the inverse-variance weighted median per ring, smoothed in
        log r over 2 px, held beyond the last ring measured to ``max_se``.  Shares, so the stars' unknown light cancels.
        The optics' halo is not grey: the field's brightest star's halo reddens outward (M 31 2026-10-24: R / G / B shares
        x 1.12 / 1.08 / 0.85 from 18 to 30 px); on the common wing the restoration kept R and G's excess as a red-green
        halo to 40 px and dug B below the sky.  None without enough stars."""
        C, H, W = ref.shape
        if C < 2:
            return None
        Rx = R
        yy, xx = np.mgrid[-Rx:Rx + 1, -Rx:Rx + 1]
        rr = np.hypot(yy, xx)
        rb = np.rint(rr).astype(int)
        skym = (rb >= R - sky_w) & (rb <= R)
        ring_n = np.bincount(rb.ravel(), minlength=R + 1)[:R + 1]
        uns, sats = [], []
        for (x, y) in catalog:
            ix, iy = int(round(x)), int(round(y))
            if not (Rx < ix < W - Rx - 1 and Rx < iy < H - Rx - 1):
                continue
            w = ref[:, iy - Rx:iy + Rx + 1, ix - Rx:ix + Rx + 1].astype(np.float64).copy()
            fin = np.isfinite(w).all(0)
            hole = ~fin & (rr <= 15)
            if hole.sum() > MAX_WING_HOLE_PX:
                continue
            r_ex = float(rr[hole].max()) + bloom if hole.any() else -1.0
            for (ox, oy) in catalog:
                dx_, dy_ = ox - x, oy - y
                if (dx_ or dy_) and abs(dx_) < Rx + mask_r and abs(dy_) < Rx + mask_r:
                    fin &= (yy - dy_) ** 2 + (xx - dx_) ** 2 > mask_r ** 2
            if (fin & skym).sum() < 30:
                continue
            p = np.full((C, R + 1), np.nan)
            se = np.full((C, R + 1), np.nan)
            for c in range(C):
                sk = w[c][fin & skym]
                sky = np.median(sk); sd = 1.4826 * np.median(np.abs(sk - sky))
                for k in range(R - sky_w):
                    if k <= r_ex:
                        continue
                    v = w[c][fin & (rb == k)]
                    if v.size >= max(1, 0.5 * ring_n[k]):
                        p[c, k] = np.median(v) - sky
                        se[c, k] = 1.2533 * sd / np.sqrt(v.size)
            tot = p.sum(0)
            with np.errstate(divide="ignore", invalid="ignore"):
                f = p / tot[None]
                fse = np.abs(f) * np.sqrt((se / p) ** 2 + (np.sqrt((se ** 2).sum(0)) / tot)[None] ** 2)
            bad = ~(tot > 0) | ~np.isfinite(fse).all(0)
            f[:, bad] = np.nan
            if r_ex < 0:
                jm = slice(j[0], j[1] + 1)
                f0 = np.nanmean(f[:, jm], 1)
                if not np.isfinite(f0).all() or (f0 <= 0).any() or np.nanmax(fse[:, jm] / np.abs(f[:, jm])) > 0.1:
                    continue
                uns.append((f / f0[:, None], fse / f0[:, None]))
            else:
                sats.append((f, fse, r_ex))

        def wmed(items):
            out = np.full((C, R + 1), np.nan); err = np.full((C, R + 1), np.inf)
            for c in range(C):
                for k in range(R + 1):
                    v = np.array([q[c, k] for q, _ in items]); s_ = np.array([e[c, k] for _, e in items])
                    ok = np.isfinite(v) & np.isfinite(s_) & (s_ > 0)
                    if ok.sum() < 1 or (ok.sum() < 3 and k <= j[1]):
                        continue
                    o_ = np.argsort(v[ok]); vv, wt = v[ok][o_], (1.0 / s_[ok] ** 2)[o_]
                    cw = np.cumsum(wt) - 0.5 * wt
                    out[c, k] = float(np.interp(0.5 * wt.sum(), cw, vv)); err[c, k] = 1.0 / math.sqrt(wt.sum())
            return out, err

        if not sats:
            return None
        # (unsaturated stars are too faint beyond ~12 px to measure a share; the saturated stars carry it, each joined
        # over its first rings - to 1 at first: the share is taken as the join's out to the first ring measured)
        X = np.ones((C, R + 1)); E = np.full((C, R + 1), np.inf)
        if len(uns) >= 3:
            Xu, Eu = wmed(uns)
            X = np.where(np.isfinite(Xu) & (Eu <= max_se), Xu, X)
        for _ in range(3):
            joined = []
            for f, fse, r_ex in sats:
                ks = [k for k in range(int(np.floor(r_ex)) + 1, R + 1) if np.isfinite(f[:, k]).all() and np.isfinite(X[:, k]).all()][:fit_w]
                if len(ks) < 3:
                    continue
                if np.max(fse[:, ks] / np.abs(f[:, ks])) > 0.1:
                    continue
                a = np.array([np.sum(f[c, ks] * X[c, ks] / fse[c, ks] ** 2) / np.sum(X[c, ks] ** 2 / fse[c, ks] ** 2) for c in range(C)])
                if not np.isfinite(a).all() or (a <= 0).any():
                    continue
                joined.append((f / a[:, None], fse / a[:, None]))
            if not joined:
                return None
            Xn, E = wmed(uns + joined)
            X = np.where(np.isfinite(Xn), Xn, 1.0)
        from scipy.ndimage import gaussian_filter1d
        out = np.ones((C, R + 1))
        for c in range(C):
            good = np.isfinite(X[c]) & (E[c] <= max_se)
            good[:j[1] + 1] = True
            last = int(np.nonzero(good)[0].max())
            xc = np.where(good, X[c], np.nan)
            xc[:j[1] + 1] = 1.0
            ks_ = np.arange(R + 1)
            xc[:last + 1] = np.interp(ks_[:last + 1], ks_[:last + 1][np.isfinite(xc[:last + 1])], xc[:last + 1][np.isfinite(xc[:last + 1])])
            xc[last + 1:] = xc[last]
            lx = gaussian_filter1d(np.log(np.clip(xc, 0.2, 5.0)), 2.0, mode="nearest")
            out[c] = np.exp(lx)
            t_ = np.clip((ks_ - j[0]) / float(j[1] - j[0]), 0.0, 1.0)
            out[c] = 1.0 + (0.5 - 0.5 * np.cos(np.pi * t_)) * (out[c] - 1.0)
        return out

    @staticmethod
    def _apply_radial_wing(t, P: np.ndarray, r_join: float = WING_JOIN, blend: float = 2.0):
        """The frame's stacked PSF ``t`` (torch (k, k)) with its radial profile beyond ``r_join`` replaced by the measured
        wing ``P`` (light per pixel by radius) scaled to the frame's ring mean at ``r_join``: w t + (1 - w) a P(r), w 1 inside
        r_join and a raised cosine to 0 at r_join + blend; beyond len(P) - 1 px 0; unit sum."""
        import torch
        tn = t.detach().cpu().numpy().astype(np.float64)
        n = tn.shape[0]
        c = n // 2
        yy, xx = np.mgrid[:n, :n] - c
        rr = np.hypot(yy, xx)
        rb = np.rint(rr).astype(int)
        ring = rb == int(round(r_join))
        pj = float(np.interp(r_join, np.arange(len(P)), P))
        if pj <= 0 or not ring.any():
            return t
        a = float(tn[ring].mean()) / pj
        if not np.isfinite(a) or a <= 0:
            return t
        Pr = np.interp(rr, np.arange(len(P)), P, right=0.0)
        w = np.clip((r_join + blend - rr) / blend, 0.0, 1.0)
        w = 0.5 - 0.5 * np.cos(np.pi * w)
        out = np.maximum(w * tn + (1.0 - w) * a * Pr, 0.0)
        out /= out.sum()
        return torch.from_numpy(out.astype(np.float32)).to(t.device)

    @staticmethod
    def _attach_far_wing(t, P: np.ndarray, r_join: float = 34.0, blend: float = 4.0):
        """PIPELINE (change 32): a solved kernel ``t`` (torch (k, k), radius k // 2 = 40) extended to the radius of the
        measured wing ``P`` (light per pixel by radius, ``_radial_wing_profile``): beyond ``r_join`` the wing, scaled to
        the kernel's ring at ``r_join``, joined over ``blend`` px; unit sum.  The field's brightest star (M 31) carries
        its PSF's light to ~80 px (relative to 12 px: 0.010 at 40, 0.003 at 60, 0.0004 at 80) - beyond a 40 px kernel it
        stayed in the sky as a 60 px glow."""
        import torch
        Rp = len(P) - 1
        tn = t.detach().cpu().numpy().astype(np.float64)
        k = tn.shape[0]
        c = k // 2
        if Rp <= c:
            return t
        n = 2 * Rp + 1
        out = np.zeros((n, n))
        out[Rp - c:Rp + c + 1, Rp - c:Rp + c + 1] = tn
        yy, xx = np.mgrid[:n, :n] - Rp
        rr = np.hypot(yy, xx)
        ring = (np.rint(rr) == int(round(r_join))) & (rr <= c)
        pj = float(np.interp(r_join, np.arange(len(P)), P))
        if pj <= 0 or not ring.any():
            return t
        a = float(out[ring].mean()) / pj
        if not np.isfinite(a) or a <= 0:
            return t
        Pr = np.interp(rr, np.arange(len(P)), P, right=0.0)
        w = np.clip((r_join + blend - rr) / blend, 0.0, 1.0)
        w = 0.5 - 0.5 * np.cos(np.pi * w)
        out = np.maximum(w * out + (1.0 - w) * a * Pr, 0.0)
        out /= out.sum()
        return torch.from_numpy(out.astype(np.float32)).to(t.device)

    @staticmethod
    def _far_nonradial(ref: np.ndarray, catalog, R: int = 80, r0: float = 20.0, taper: float = 4.0, bloom: int = 0,
                       mask_r: float = 6.0, max_se: float = 0.03, smooth: float = 3.0):
        """PIPELINE (change 33): the PSF's non-radial part beyond ``r0`` px, as a relative modulation (n, n), n = 2 R + 1,
        zero mean in every ring, of the radial wing - from the saturated stars of the frames' mean ``ref`` (the channels'
        sum; NaN where a frame saturated), the only stars whose light reaches out there.  Per star (centred on its
        saturated hole, as the initial guess places it; other catalogue stars masked; a robust plane through the R .. R
        + 10 px annulus removed) and 1 px ring beyond its hole + ``bloom`` px: the ring's m = 1, 2 Fourier terms over its
        mean (least squares, 3-sigma clipped), with their standard errors; per ring and term the inverse-variance
        weighted mean over the stars, smoothed over ``smooth`` px in radius; kept where the standard error is under
        ``max_se``, held inward from the first such ring, raised in from ``r0`` - ``taper`` and tapered to 0 over
        ``taper`` px past the last.  The
        unsaturated stars' static part (``_nonradial_psf``) ends at 20 px; beyond, the field's brightest star's halo is
        lopsided by +-12-16 % at 22-46 px in all three channels at the star's colour, growing outward - not a centring
        error (that falls as 1 / r) - and on a radial wing the restoration kept the bright side as a glow and cut the
        other below the sky (M 31 2026-10-24).  None without a measured ring."""
        from scipy.ndimage import label, gaussian_filter1d
        H, W = ref.shape
        Rw = R + 10
        yy, xx = np.mgrid[-Rw:Rw + 1, -Rw:Rw + 1].astype(np.float64)
        rr = np.hypot(yy, xx)
        bgm = (rr >= R) & (rr <= Rw)
        hole_all = ~np.isfinite(ref)
        lab, _ = label(hole_all)
        num = np.zeros((R + 1, 4)); den = np.zeros((R + 1, 4))
        used = 0
        for (x, y) in catalog:
            ix, iy = int(round(x)), int(round(y))
            if not (Rw < ix < W - Rw - 1 and Rw < iy < H - Rw - 1):
                continue
            lw = lab[iy - Rw:iy + Rw + 1, ix - Rw:ix + Rw + 1]
            q = lw[Rw, Rw]
            if q == 0:
                continue                                      # (not saturated)
            hole = lw == q
            if hole.sum() > MAX_WING_HOLE_PX:
                continue
            hy, hx = np.nonzero(hole)
            cy_, cx_ = hy.mean() - Rw, hx.mean() - Rw
            w = ref[iy - Rw:iy + Rw + 1, ix - Rw:ix + Rw + 1].astype(np.float64)
            ok = np.isfinite(w)
            for (ox, oy) in catalog:
                dx_, dy_ = ox - x, oy - y
                if (dx_ or dy_) and abs(dx_) < Rw + mask_r and abs(dy_) < Rw + mask_r:
                    ok &= (yy - dy_) ** 2 + (xx - dx_) ** 2 > mask_r ** 2
            if (bgm & ok).sum() < 100:
                continue
            pl = ImageMMEngine._robust_plane(w[bgm & ok], yy[bgm & ok], xx[bgm & ok])
            ab = w - (pl[0] + pl[1] * xx + pl[2] * yy)
            r_ = np.hypot(yy - cy_, xx - cx_)
            rb_ = np.rint(r_).astype(int)
            th = np.arctan2(yy - cy_, xx - cx_)
            r_ex = float(r_[hole].max()) + bloom
            got = False
            for k in range(int(max(r0 - taper, r_ex + 1)), R + 1):
                sel = ok & (rb_ == k)
                if sel.sum() < 30:
                    continue
                v, t_ = ab[sel], th[sel]
                B = np.c_[np.ones_like(t_), np.cos(t_), np.sin(t_), np.cos(2 * t_), np.sin(2 * t_)]
                keep = np.ones(v.size, bool)
                for _ in range(2):
                    cf = np.linalg.lstsq(B[keep], v[keep], rcond=None)[0]
                    res = v - B @ cf
                    sd = 1.4826 * np.median(np.abs(res[keep]))
                    keep = np.abs(res) <= 3.0 * max(sd, 1e-30)
                if cf[0] <= 0 or keep.sum() < 20:
                    continue
                se = sd / math.sqrt(keep.sum() / 2.0) / cf[0]
                if cf[0] < 5.0 * sd / math.sqrt(keep.sum()):
                    continue
                num[k] += (cf[1:] / cf[0]) / se ** 2
                den[k] += 1.0 / se ** 2
                got = True
            used += int(got)
        if used == 0 or not (den > 0).any():
            return None
        ks = np.arange(R + 1)
        coef = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)
        err = np.where(den > 0, 1.0 / np.sqrt(np.maximum(den, 1e-30)), np.inf)
        good = (err[:, 0] <= max_se) & (ks >= r0 - taper)
        if not good.any():
            return None
        last = int(ks[good].max())
        first = int(ks[good].min())
        wts = np.where(good, 1.0, 0.0)
        cs = np.stack([gaussian_filter1d(coef[:, j] * wts, smooth, mode="nearest") /
                       np.maximum(gaussian_filter1d(wts, smooth, mode="nearest"), 1e-6) for j in range(4)], 1)
        cs[:first] = cs[first]                          # (held inward to r0: the brightest star's hole hides its start)
        env = np.clip((ks - (r0 - taper)) / taper, 0.0, 1.0) * np.clip((last + taper - ks) / taper, 0.0, 1.0)
        env = 0.5 - 0.5 * np.cos(np.pi * env)
        cs *= env[:, None]
        n = 2 * R + 1
        y2, x2 = np.mgrid[:n, :n] - R
        r2 = np.hypot(y2, x2); t2 = np.arctan2(y2, x2)
        ci = [np.interp(r2, ks, cs[:, j], right=0.0) for j in range(4)]
        M = ci[0] * np.cos(t2) + ci[1] * np.sin(t2) + ci[2] * np.cos(2 * t2) + ci[3] * np.sin(2 * t2)
        a1 = np.hypot(cs[:, 0], cs[:, 1])
        return M, used, float(a1[first:last + 1].max()) if last >= first else 0.0, float(np.degrees(np.arctan2(cs[(first + last) // 2, 1], cs[(first + last) // 2, 0]))), (first, last)

    @staticmethod
    def _modulate(t, M: np.ndarray):
        """A kernel ``t`` (torch (k, k)) times (1 + ``M``) (``_far_nonradial``, centred; cropped or padded to k),
        clipped at 0, unit sum."""
        import torch
        tn = t.detach().cpu().numpy().astype(np.float64)
        k, n = tn.shape[0], M.shape[0]
        if n >= k:
            o = (n - k) // 2
            Mk = M[o:o + k, o:o + k]
        else:
            o = (k - n) // 2
            Mk = np.zeros((k, k)); Mk[o:o + n, o:o + n] = M
        out = np.maximum(tn * (1.0 + Mk), 0.0)
        out /= out.sum()
        return torch.from_numpy(out.astype(np.float32)).to(t.device)

    @staticmethod
    def _add_nonradial(t, N_: np.ndarray):
        """The frame's stacked PSF ``t`` (torch (k, k)) plus the static non-radial part ``N_`` (same size), clipped at 0,
        unit sum."""
        import torch
        tn = t.detach().cpu().numpy().astype(np.float64)
        if tn.shape != N_.shape:
            return t
        out = np.maximum(tn + N_, 0.0)
        out /= out.sum()
        return torch.from_numpy(out.astype(np.float32)).to(t.device)

    @staticmethod
    def _join_wings(t, wing: np.ndarray, r_f: float, r_m: float, join: float = 4.0):
        """A frame's stacked PSF ``t`` (torch, (k, k)) with the mean image's ``wing`` beyond the frame's detected radius
        ``r_f``: w t + (1 - w) a wing, w 1 inside r_f - join and a raised cosine to 0 at r_f, a = the frame's light over
        the light of ``wing`` in the ring r_f - join .. r_f; unit sum.  Unchanged when the mean image does not reach
        further than the frame."""
        import torch
        if r_m <= r_f + 1:
            return t
        tn = t.detach().cpu().numpy().astype(np.float64)
        n = tn.shape[0]
        c = n // 2
        yy, xx = np.mgrid[:n, :n]
        rr = np.hypot(yy - c, xx - c)
        ring = (rr >= r_f - join) & (rr <= r_f)
        a = float(tn[ring].sum()) / max(float(wing[ring].sum()), 1e-12)
        if not np.isfinite(a) or a <= 0:
            return t
        w = np.clip((r_f - rr) / join, 0.0, 1.0)
        w = 0.5 - 0.5 * np.cos(np.pi * w)
        out = np.maximum(w * tn + (1.0 - w) * a * wing, 0.0)
        out /= out.sum()
        return torch.from_numpy(out.astype(np.float32)).to(t.device)

    @staticmethod
    def _restored_psf(x, catalog, core, R: int = 20, min_snr: float = 200.0, max_stars: int = 60, R_ext: int = 48):
        """(C, 2 R_ext + 1, 2 R_ext + 1) the restored image's own PSF per channel - how a star looks after the restoration - from the
        brightest unsaturated, isolated catalogue stars of ``x``: cut-outs at the green centroid (sub-pixel shift), sky
        the median at R - 3 .. R, each divided by its light within R, the median per pixel.  None with fewer than 5."""
        from scipy.ndimage import shift as _nd_shift
        C, H, W = x.shape
        g_ = 1 if C > 1 else 0
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        rr = np.hypot(yy, xx)
        skym = rr >= R - 3
        near = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * R + 1, 2 * R + 1))) > 0
        cand = []
        for (cx, cy) in catalog:
            ix, iy = int(round(cx)), int(round(cy))
            if not (R + 2 < ix < W - R - 3 and R + 2 < iy < H - R - 3) or near[iy, ix]:
                continue
            if any(0 < (ox - cx) ** 2 + (oy - cy) ** 2 < (R + 2) ** 2 for ox, oy in catalog):
                continue
            w = x[:, iy - R:iy + R + 1, ix - R:ix + R + 1].astype(np.float64)
            if not np.isfinite(w).all():
                continue
            sky = np.median(w[g_][skym]); sd = 1.4826 * np.median(np.abs(w[g_][skym] - sky))
            f = float((w[g_] - sky)[rr <= R].sum())
            if sd <= 0 or f < min_snr * sd * math.sqrt(np.pi * R * R):
                continue
            cand.append((f, ix, iy, w))
        cand.sort(key=lambda t: -t[0])
        cuts = []
        for f, ix, iy, w in cand[:max_stars]:
            a = np.clip(w[g_] - np.median(w[g_][skym]), 0, None) * (rr <= 3)
            if a.sum() <= 0:
                continue
            dy, dx = float((a * yy).sum() / a.sum()), float((a * xx).sum() / a.sum())
            cut = []
            for c in range(C):
                v = w[c] - np.median(w[c][skym])
                v = _nd_shift(v, (-dy, -dx), order=3, mode="nearest")
                tot = float(v[rr <= R].sum())
                if tot <= 0:
                    break
                cut.append(v / tot)
            if len(cut) == C:
                cuts.append(np.stack(cut))
        if len(cuts) < 5:
            return None
        cuts = np.array(cuts)
        psi = np.median(cuts, 0)
        # (beyond 3 px each pixel of the median is the stars' noise - scaled by the field's brightest star it was coloured
        # speckle: there the profile is the ring median, smoothed in radius; the 2-D shape is kept in the core.  The profile
        # is measured out to where it falls below 2 x its standard error (the stars' scatter of their ring medians) for 2
        # rings and falls monotonically, as a star's does: its noisy tail, x the field's brightest star (~1e3 x the
        # measured stars' light), was a dark and a bright blue ring at 16 / 20 px (M 31 2026-10-24))
        from scipy.ndimage import gaussian_filter1d
        rb_ = np.rint(rr).astype(int)
        ks_ = np.arange(R + 1)
        profs, r_ts = [], []
        for c in range(C):
            per = np.array([[np.median(ct[c][rb_ == k]) for k in range(R + 1)] for ct in cuts])
            prof = np.median(per, 0)
            se_ = 1.2533 * 1.4826 * np.median(np.abs(per - prof[None]), 0) / math.sqrt(len(cuts))
            r_t, low = R, 0
            for k in range(4, R + 1):
                low = low + 1 if prof[k] < 2.0 * se_[k] else 0
                if low == 2:
                    r_t = k - 1
                    break
            prof = np.clip(gaussian_filter1d(prof, 1.0, mode="nearest"), 0, None)
            profs.append(np.minimum.accumulate(prof)); r_ts.append(r_t)
        # beyond where it is measured, each channel's profile continues as the channels' common power law (fitted on
        # green over the 6 rings before its limit, slope -2 .. -6) out to R_ext: tapered to 0 there, x a saturated star's
        # light it ended in an edge at 12-14 px after the finish; per channel slopes coloured the tail (a magenta ring)
        g_ = 1 if C > 1 else 0
        r_g = r_ts[g_]
        kf = np.arange(max(4, r_g - 6), r_g + 1)
        okf = profs[g_][kf] > 0
        slope = float(np.clip(np.polyfit(np.log(kf[okf]), np.log(profs[g_][kf][okf]), 1)[0], -6.0, -2.0)) if okf.sum() >= 3 else -3.0
        yx, xx_ = np.mgrid[-R_ext:R_ext + 1, -R_ext:R_ext + 1]
        rx = np.hypot(yx, xx_)
        kx = np.arange(R_ext + 1)
        out = np.zeros((C, 2 * R_ext + 1, 2 * R_ext + 1))
        for c in range(C):
            r_t = min(r_ts[c], r_g)
            pe = np.interp(kx, ks_, profs[c])
            if profs[c][r_t] > 0:
                pe[r_t:] = profs[c][r_t] * (kx[r_t:] / float(r_t)) ** slope
            core_c = np.zeros_like(rx); core_c[R_ext - R:R_ext + R + 1, R_ext - R:R_ext + R + 1] = psi[c]
            w_ = np.clip((rx - 3.0) / 2.0, 0.0, 1.0)
            out[c] = (1.0 - w_) * core_c + w_ * np.interp(rx, kx, pe, right=0.0)
        out = np.where(rx[None] <= R_ext, np.clip(out, 0, None), 0.0)
        return out / out.sum(axis=(1, 2), keepdims=True)

    @staticmethod
    def _hole_centre(reg, vf_g, yy, xx, c0, frac: float = 0.5):
        """PIPELINE (change 33): a saturated star's centre - the centroid of its missing data (the pixels valid in under
        ``frac`` of the frames, connected to its core ``reg``), the centre ``_far_nonradial`` measures the PSF's lobe
        round.  The saturated plateau's own light-weighted centre leans into the PSF's inner lobe: for M 31's brightest
        star 0.7 px off that centre, and the restoration's point there left a dipole of +3 / -1 sigma at 24-32 px - a
        cyan glow on one side, a dark hollow on the other.  ``c0`` when there is no such region."""
        from scipy.ndimage import label
        lz, n_ = label((vf_g < frac) | reg)
        ids = np.unique(lz[reg & (lz > 0)])
        hole = np.isin(lz, ids[ids > 0])
        if hole.sum() < 5:
            return c0
        return float(yy[hole].mean()), float(xx[hole].mean())

    @staticmethod
    def _robust_plane(v, yy, xx, it: int = 3):
        """Robust plane a + b x + c y through the values ``v`` at (``yy``, ``xx``): least squares, 3-sigma clipped."""
        keep = np.isfinite(v)
        coef = np.array([np.nanmedian(v) if keep.any() else 0.0, 0.0, 0.0])
        for _ in range(it):
            if keep.sum() < 10:
                break
            A = np.c_[np.ones(keep.sum()), xx[keep], yy[keep]]
            coef = np.linalg.lstsq(A, v[keep], rcond=None)[0]
            r_ = v - (coef[0] + coef[1] * xx + coef[2] * yy)
            sd = 1.4826 * np.nanmedian(np.abs(r_[keep]))
            keep = np.isfinite(v) & (np.abs(r_) <= 3.0 * max(sd, 1e-30))
        return coef

    @staticmethod
    def _saturated_star_photometry(x, core, mean_img, kmean, grow: int = 2, ann: tuple = (4, 8), sky: tuple = (18, 26),
                                   sigma: float = 1.0, R_win: int = 104, bg_ring: tuple = (90, 104), ext_max: float = 0.1, valid_frac=None,
                                   inner_px: float = 2.0, blend_px: float = SAT_BLEND, psi=None, catalog=None):
        """PIPELINE (changes 25, 33): saturated stars as the restored image's PSF at their wings' colour; saturated galaxy
        nuclei with their colour.

        Next to a saturated core the PSF model does not fit the data (M 31 2026-10-22: 2.3-3.8 x the PSF at 5-6 px against
        a point fitted at 9-25 px; charge spill near a full well) and inside it the data say only "at least the clip
        level": where the frames' data are sparse - valid in under SPARSE_FRAC of them - the restoration rests on few
        frames or none, and its light there is not to be trusted (green cores, ragged blobs, coloured rings).  Per
        saturated star (pixels saturated in at least half the frames covering them, grown by ``grow`` px) that region,
        grown by 1 px, is the star as every star of the field looks after the restoration - the restored image's own PSF
        ``psi`` (``_restored_psf``) at the star's light (green's light of the restoration in the region, or where the model
        is still bright at the region's edge, fitted to the restoration just outside) and each channel at the colour of
        its unsaturated wings - on a floor (the background plane at ``bg_ring`` px plus a robust plane fitted 4-20 px
        outside the region), joined over 30 % of the region's radius (the restoration's own structure kept there, less
        its smooth part per ring), the PSF's tail fading on beyond to TAIL_FLOOR x the restoration's noise.  (Every
        rewrite of the star that ended at the region's edge made its own artefact: a point on a flat floor - rings after
        StarNet; the frames' own mean - stars twice BXT's size; a noiseless join - a disc round every star.)  The wing
        colour is F_c / F_g, F_c = the
        frames' mean over the wing annulus ``ann`` px outside the grown core (beyond the bloom zone; local sky from the
        ``sky`` ring; per 1 px ring, pixels off the ring's green median by 3 robust sigma - neighbours - left out) / the
        channel's mean kernel's light there, calibrated on unsaturated stars of known colour.  The window is ``R_win`` px,
        padded with "no data" at the image's edge; the radial profile (ring medians above the background plane, to where
        green's falls below half a ring's standard error for 3 rings) measures the light beyond the core.

        An EXTENDED source - a galaxy nucleus: light beyond the core over ``ext_max`` x the core's (M 31's 23 saturated
        stars in two regions 0.000-0.023, the field's brightest included; M 32 0.53, M 31's nucleus 2.9) - keeps its
        restored shape and gets the colour only: as a point M 32 became a dot in a dark ring.  Near the image's edge the window is padded with "no data"."""
        import torch
        from scipy.ndimage import label
        C, H, W = mean_img.shape
        g_ = 1 if C > 1 else 0
        grown = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))) > 0
        lab, n = label(grown)
        xc = x[0].detach().cpu().numpy().astype(np.float64)
        kc = kmean.shape[-1] // 2
        R = int(max(R_win, bg_ring[1] + 1, sky[1] + 1))
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float64)
        rr0 = np.hypot(yy, xx)
        kws = []
        for c in range(C):
            k = kmean[c if kmean.shape[0] > 1 else 0]
            kw = np.zeros((2 * R + 1, 2 * R + 1))
            h_ = min(kc, R)
            kw[R - h_:R + h_ + 1, R - h_:R + h_ + 1] = k[kc - h_:kc + h_ + 1, kc - h_:kc + h_ + 1]
            kws.append(kw)
        # the latent's pixel noise per channel (3 x 3 high-pass of the whole image, MAD)
        noise = []
        for c in range(C):
            hp = xc[c] - cv2.blur(xc[c], (3, 3))
            noise.append(1.4826 * float(np.median(np.abs(hp - np.median(hp)))) / math.sqrt(8.0 / 9.0))
        done, ratios, dbg = 0, [], []
        skipped = extended = 0
        # the wing-colour estimator calibrated on stars of known colour (an aperture correction per band): for the bright
        # unsaturated catalogue stars, their colour within 6 px over their colour by the same estimator in 4 px rings
        # centred 6 .. 30 px out; the median per channel and ring.  Uncalibrated, a channel's kernel wing error went
        # straight into the core's colour (M 31: green-tinged cores, R / G and B / G both low - no star's spectrum)
        col_cal = {}
        if catalog:
            near_sat = cv2.dilate(grown.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))) > 0
            bands = list(range(6, 31, 2))
            acc_ = {b_: [] for b_ in bands}
            for (sx_, sy_) in catalog:
                ix_, iy_ = int(round(sx_)), int(round(sy_))
                if not (R < ix_ < W - R - 1 and R < iy_ < H - R - 1) or near_sat[iy_, ix_]:
                    continue
                mw_ = mean_img[:, iy_ - R:iy_ + R + 1, ix_ - R:ix_ + R + 1].astype(np.float64)
                if not np.isfinite(mw_[:, rr0 <= 34]).all():
                    continue
                bgv = np.array([np.median(mw_[c][(rr0 >= 36) & (rr0 <= 44)]) for c in range(C)])
                sdv = 1.4826 * np.median(np.abs(mw_[g_][(rr0 >= 36) & (rr0 <= 44)] - bgv[g_]))
                ap_ = np.array([(mw_[c] - bgv[c])[rr0 <= 6].sum() for c in range(C)])
                if ap_[g_] < 2000.0 * sdv or (ap_ <= 0).any():
                    continue
                for b_ in bands:
                    ring_ = (rr0 >= b_ - 2) & (rr0 < b_ + 2)
                    Fb = np.array([(mw_[c] - bgv[c])[ring_].sum() / max(float(kws[c][ring_].sum()), 1e-30) for c in range(C)])
                    if (Fb > 0).all():
                        acc_[b_].append((ap_ / ap_[g_]) / (Fb / Fb[g_]))
            for b_ in bands:
                if len(acc_[b_]) >= 8:
                    col_cal[b_] = np.median(np.array(acc_[b_]), 0)
        # (padded by R with "no data": a star near the edge gets the same treatment)
        xp = np.pad(xc, ((0, 0), (R, R), (R, R)), constant_values=np.nan)
        mp = np.pad(mean_img.astype(np.float64), ((0, 0), (R, R), (R, R)), constant_values=np.nan)
        lp = np.pad(lab, ((R, R), (R, R)), constant_values=-1)
        vf = np.pad(valid_frac if valid_frac is not None else np.ones((C, H, W)), ((0, 0), (R, R), (R, R)), constant_values=0.0)
        for q in range(1, n + 1):
            ys, xs = np.nonzero(lab == q)
            if len(ys) > MAX_SAT_CORE_PX:                     # (a galaxy nucleus, not a star)
                skipped += 1
                continue
            iy, ix = int(round(float(ys.mean()))) + R, int(round(float(xs.mean()))) + R
            lw = lp[iy - R:iy + R + 1, ix - R:ix + R + 1]
            reg, free = lw == q, lw == 0
            dist = cv2.distanceTransform((~reg).astype(np.uint8), cv2.DIST_L2, 3)
            annm = (dist > ann[0]) & (dist <= ann[1]) & free
            skym = (rr0 >= sky[0]) & (rr0 <= sky[1]) & free
            if annm.sum() < 20 or skym.sum() < 20:
                continue
            win = xp[:, iy - R:iy + R + 1, ix - R:ix + R + 1]
            have = np.isfinite(win).all(0)
            free = free & have
            # centre: green's light-weighted core
            gwin = win[g_]
            core_w = np.nan_to_num(np.clip(gwin - np.nanmedian(gwin[(rr0 >= bg_ring[0]) & (rr0 <= bg_ring[1])]), 0, None)) * (dist <= 0)
            if core_w.sum() <= 0:
                dbg.append((ix - R, iy - R, "no core light")); continue
            ccy = float((core_w * yy).sum() / core_w.sum()); ccx = float((core_w * xx).sum() / core_w.sum())
            if valid_frac is not None:
                ccy, ccx = ImageMMEngine._hole_centre(reg, vf[g_, iy - R:iy + R + 1, ix - R:ix + R + 1], yy, xx, (ccy, ccx))
            rr = np.hypot(yy - ccy, xx - ccx)
            rb = np.rint(rr).astype(int)
            bgm = (rr >= bg_ring[0]) & (rr <= bg_ring[1]) & free
            plane = np.stack([(lambda cf: cf[0] + cf[1] * xx + cf[2] * yy)(ImageMMEngine._robust_plane(win[c][bgm], yy[bgm], xx[bgm]))
                              for c in range(C)])
            above = win - plane
            # radial light per channel; its extent from green
            r_in = float(rr[reg].max()) + 2.0 + SAT_BLEND
            nr = bg_ring[0]
            prof = np.zeros((C, nr + 1))
            se = np.zeros(nr + 1)
            for k in range(nr + 1):
                sel = (rb == k) & (free | reg)
                if sel.sum() == 0:
                    continue
                for c in range(C):
                    prof[c, k] = np.median(above[c][sel])
                se[k] = 1.2533 * noise[g_] / math.sqrt(sel.sum())
            r_out, low = nr, 0
            for k in range(int(math.ceil(r_in)), nr + 1):
                low = low + 1 if prof[g_, k] < 0.5 * se[k] else 0
                if low == 3:
                    r_out = k - 2
                    break
            ks = np.arange(nr + 1)
            tap = np.clip((r_out + 1.5 - ks) / 3.0, 0.0, 1.0)            # (to 0 over 3 px at r_out)
            prof = prof * tap[None]
            area = np.array([((rb == k) & (free | reg)).sum() for k in range(nr + 1)], np.float64)
            L_core = float(above[g_][(rr <= r_in) & (free | reg)].sum())          # (the core's own pixels: a ring median misjudges it)
            L_ext = float((np.clip(prof[g_], 0, None) * area)[ks > r_in].sum())
            is_ext = L_core <= 0 or L_ext > ext_max * L_core
            # colour from the wing annulus
            mw = mp[:, iy - R:iy + R + 1, ix - R:ix + R + 1]
            clean = annm & np.isfinite(mw).all(0)
            dr = np.rint(dist).astype(int)
            for k_ in range(ann[0] + 1, ann[1] + 1):
                rg_ = clean & (dr == k_)
                if rg_.sum() < 5:
                    continue
                v_ = mw[g_][rg_]
                md = np.median(v_); sd = 1.4826 * np.median(np.abs(v_ - md))
                clean[rg_] = np.abs(v_ - md) <= 3.0 * max(sd, 1e-12)
            a_in, a_out = clean & (dist <= (ann[0] + ann[1]) / 2), clean & (dist > (ann[0] + ann[1]) / 2)
            Fi, Fo = np.zeros(C), np.zeros(C)
            for c in range(C):
                d_ = mw[c] - np.nanmedian(mw[c][skym])
                Fi[c] = d_[a_in].sum() / max(float(kws[c][a_in].sum()), 1e-12)
                Fo[c] = d_[a_out].sum() / max(float(kws[c][a_out].sum()), 1e-12)
            with np.errstate(divide="ignore", invalid="ignore"):
                q_in, q_out = Fi / Fi[g_], Fo / Fo[g_]
            ok_ = bool(np.all(np.isfinite(q_in) & np.isfinite(q_out) & (q_in > 0) & (q_out > 0)))
            dbg.append((ix - R, iy - R, "extended" if is_ext else "point", round(r_out, 1), round(L_ext / max(L_core, 1e-30), 3),
                        [round(float(v), 3) for v in q_in], [round(float(v), 3) for v in q_out]))
            if not ok_:
                skipped += 1
                continue
            col = 0.5 * (q_in + q_out)
            if col_cal:
                r_ann = float(np.mean(rr0[annm]))
                b_near = min(col_cal, key=lambda b_: abs(b_ - r_ann))
                col = col * col_cal[b_near]
            inner = dist <= inner_px
            d_in = cv2.distanceTransform((~inner).astype(np.uint8), cv2.DIST_L2, 3)
            wgt = np.where(inner, 1.0, np.clip(1.0 - d_in / blend_px, 0.0, 1.0))
            wgt = np.where(inner, 1.0, 0.5 + 0.5 * np.cos(np.pi * (1.0 - wgt))) * (wgt > 0)
            if is_ext:
                # keep the restored shape: green's light above the plane, each channel at its measured colour
                extended += 1
                for c in range(C):
                    new = plane[c] + col[c] * np.clip(above[g_], 0, None)
                    win[c] = np.where(have, wgt * new + (1.0 - wgt) * win[c], win[c])
            else:
                # a star: a point at the latent's resolution g_sigma (Eq. 11), centred on the green core's light, on the
                # background plane, with green's restored light in the region and each channel at the colour of the
                # unsaturated wings (PSF photometry: inside the region the frames have no data - saturated or bloom - so
                # each channel's light there is not identified by the restoration).  Keeping the region's own pixels (their
                # few frames' data) gave rainbow cores; the star's forward model there (the frames' PSF) a hard-edged
                # disc (2026-10-23/24)
                fl = plane
                if valid_frac is not None:
                    # (the region is where the frames' data are sparse - valid in under SPARSE_FRAC of them - grown by 1 px and
                    # joined over 30 % of its radius: just outside the bare no-data core the restoration rests on a few frames and each
                    # channel's noise there is its own - a ring of coloured speckle round the field's brightest star)
                    vw_ = vf[g_, iy - R:iy + R + 1, ix - R:ix + R + 1]
                    sparse = ((vw_ < SPARSE_FRAC) & (rr <= bg_ring[0] - 8)) | reg
                    lz_, nz2 = label(sparse)
                    keep_ = np.unique(lz_[reg & (lz_ > 0)])
                    sparse = np.isin(lz_, keep_[keep_ > 0]) | reg
                    sparse = cv2.dilate(sparse.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
                    d_s = cv2.distanceTransform((~sparse).astype(np.uint8), cv2.DIST_L2, 3)
                    # (the join scales with the region: over 3 px a 20 px region met the restoration's falling halo with a
                    # kink - a ring round the field's brightest star)
                    jb = max(3.0, 0.3 * float(rr[sparse].max()) if sparse.any() else 3.0)
                    ws = np.where(sparse, 1.0, np.clip(1.0 - d_s / jb, 0.0, 1.0))
                    wgt = np.where(sparse, 1.0, 0.5 - 0.5 * np.cos(np.pi * ws)) * (ws > 0)
                Lg = float((np.clip(win[g_] - fl[g_], 0.0, None) * wgt)[have].sum())
                from scipy.ndimage import shift as _nd_shift
                Sks = []
                for c in range(C):
                    if psi is not None:
                        # (the star as every star of the field looks after the restoration: the restored image's own PSF)
                        Rp = psi.shape[-1] // 2
                        Sk = np.zeros_like(rr)
                        Sk[R - Rp:R + Rp + 1, R - Rp:R + Rp + 1] = psi[c]
                        Sk = np.maximum(_nd_shift(Sk, (ccy, ccx), order=3, mode="constant"), 0.0)
                    else:
                        Sk = np.exp(-(rr ** 2) / (2.0 * sigma * sigma))
                    Sks.append(Sk / max(float((Sk * wgt).sum()), 1e-30))
                amps = [col[c] * Lg for c in range(C)]
                # where the model is still bright at the region's edge, its amplitude from the restoration just outside -
                # data there - by least squares over a 3 px ring (sources clipped): a continuous join.  Decided on green; a
                # channel without a significant fit of its own takes green's x the wing colour (decided per channel, the
                # channels mixed two methods - a green core)
                edge_ = (wgt == 0) & have & free & (cv2.distanceTransform((wgt == 0).astype(np.uint8), cv2.DIST_L2, 3) <= 3)
                if psi is not None and edge_.sum() >= 30:
                    fits = []
                    for c in range(C):
                        dv = (win[c] - fl[c])[edge_]; sv = Sks[c][edge_]
                        md_ = np.median(dv); sd_ = 1.4826 * np.median(np.abs(dv - md_))
                        okv = np.abs(dv - md_) <= 4 * max(sd_, 1e-30)
                        if okv.sum() < 20 or (sv[okv] ** 2).sum() <= 0:
                            fits.append((0.0, 0.0)); continue
                        a_e = float((dv[okv] * sv[okv]).sum() / (sv[okv] ** 2).sum())
                        fits.append((a_e, a_e * float(sv[okv].mean()) / max(sd_ / math.sqrt(okv.sum()), 1e-30)))
                    if fits[g_][0] > 0 and fits[g_][1] > 3.0:
                        amps = [fits[c][0] if (fits[c][0] > 0 and fits[c][1] > 3.0) else col[c] * fits[g_][0] for c in range(C)]
                # (the plane at 90-104 px misses a galaxy's curvature at 10 px: the floor takes the level the restoration has
                # round the region, less the model - an outline round the region otherwise)
                d_out = cv2.distanceTransform((wgt == 0).astype(np.uint8), cv2.DIST_L2, 3)
                ring_out = (wgt == 0) & have & free & (d_out > 4) & (d_out <= 20)
                # (a robust plane through an annulus 4-20 px outside the region, not the median of the 3 px just outside:
                # a constant across a 20 px region was a flat disc with a step where the surroundings slope, and the ring
                # just outside carries the restoration's edge residue - tilted by it, the plane cut a dark hollow into
                # one side of the field's brightest star)
                offs = [(lambda cf: cf[0] + cf[1] * xx + cf[2] * yy)(ImageMMEngine._robust_plane(
                            (win[c] - fl[c] - amps[c] * Sks[c])[ring_out], yy[ring_out], xx[ring_out]))
                        if ring_out.sum() >= 20 else 0.0 for c in range(C)]
                # (in the join, where the frames' data are dense, the restoration's own structure - its noise, neighbouring
                # stars - is kept and only its smooth part blended into the model: blended whole, the join was a
                # noiseless annulus, 0.3-0.6 x the sky's noise at 10-14 px, a smooth disc round every saturated star after
                # the finish (M 31 2026-10-24; BXT's stars sit in the sky's noise).  Inside the sparse zone the restoration
                # rests on few frames - its texture there, kept, was coloured speckle on the stars)
                # (the restoration less its smooth part per 1 px ring round the star - the ring's m = 0, 1, 2 terms - and
                # not less a blur: a blur's high-pass kept a neighbouring star's core and cut a dark ring round it, a
                # black dot beside it after finishing)
                dense_ = ~sparse if valid_frac is not None else (wgt < 1.0)
                tex = np.zeros((C,) + rr.shape)
                zone = (wgt > 0) & dense_ & have
                if zone.any():
                    th_ = np.arctan2(yy - ccy, xx - ccx)
                    for k_ in range(int(rr[zone].min()), int(np.ceil(rr[zone].max())) + 1):
                        ring_ = (rb == k_) & have & dense_
                        if ring_.sum() < 16 or not (ring_ & zone).any():
                            continue
                        t_ = th_[ring_]
                        B_ = np.c_[np.ones(t_.size), np.cos(t_), np.sin(t_), np.cos(2 * t_), np.sin(2 * t_)]
                        for c in range(C):
                            v_ = win[c][ring_]
                            kp = np.ones(v_.size, bool)
                            for _ in range(3):
                                cf_ = np.linalg.lstsq(B_[kp], v_[kp], rcond=None)[0]
                                res_ = v_ - B_ @ cf_
                                sd_ = 1.4826 * np.median(np.abs(res_[kp]))
                                kp = np.abs(res_) <= 2.5 * max(sd_, 1e-30)
                            tex[c][ring_] = res_
                # the star's tail beyond the region: as every star of the field looks after the restoration (the restored
                # PSF's tail, its light within 0.3 x the restoration's noise per pixel there) out to where it falls to
                # TAIL_FLOOR x that noise, the star's light shared over it - the restoration's
                # sky, noise and neighbours there kept.  Stopped at the region's edge it was a step from ~10 sigma to
                # the sky after the finish's noise reduction, which StarNet cut into a dark moat (M 31 2026-10-24)
                tail_w = np.zeros_like(rr)
                if psi is not None and valid_frac is not None and sparse.any():
                    out_ = (wgt < 1.0) & have
                    r_f = None
                    mg = amps[g_] * Sks[g_]
                    for k_ in range(int(rr[sparse].max()) + 1, R_win):
                        sel_ = (rb == k_) & out_
                        if sel_.sum() and float(np.median(mg[sel_])) < TAIL_FLOOR * noise[g_]:
                            r_f = float(k_)
                            break
                    if r_f is not None:
                        # (whole while the model is over 0.3 x the noise, then fading in log brightness to TAIL_FLOOR x it:
                        # faded by the distance from the region it lost half its light within 10 px - a step at the edge)
                        lvl = np.log(np.maximum(mg, 1e-30) / (TAIL_FLOOR * noise[g_])) / math.log(0.3 / TAIL_FLOOR)
                        w_f = np.clip(lvl, 0.0, 1.0)
                        w_f = 0.5 - 0.5 * np.cos(np.pi * w_f)
                        tail_w = np.where(sparse, 0.0, (1.0 - wgt) * w_f) * (rr <= r_f) * have
                        for c in range(C):
                            amps[c] *= float((Sks[c] * wgt).sum()) / max(float((Sks[c] * (wgt + tail_w)).sum()), 1e-30)
                # (the texture comes in over the join from the sparse zone's edge: right at it the restoration's pile-up of
                # the frames' last unsaturated pixels varies round the ring per channel - coloured specks)
                tex_w = np.clip(d_s / max(3.0, jb), 0.0, 1.0) if valid_frac is not None else 1.0
                for c in range(C):
                    new = fl[c] + offs[c] + amps[c] * Sks[c] * (wgt > 0)
                    win[c] = np.where(have, wgt * (new + np.where(zone, tex[c] * tex_w, 0.0)) + (1.0 - wgt) * win[c] + tail_w * amps[c] * Sks[c], win[c])
            ratios.append(list(col))
            done += 1
        xc = xp[:, R:R + H, R:R + W]
        out = torch.from_numpy(xc.astype(np.float32)).to(x.device)[None]
        return out, {"cores": int(n), "measured": done, "skipped": skipped, "extended": extended,
                     "colour_ratio_median": [round(float(v), 3) for v in np.median(ratios, 0)] if ratios else None, "stars": dbg}

    @staticmethod
    def _point_initial(x, core, sigma: float, noise, grow: int = 2, R: int = 104, bg_ring: tuple = (90, 104), ext_max: float = 0.5, valid=None,
                       kmean=None, valid_frac=None, fit_w: int = 30):
        """PIPELINE (change 30): each saturated star in the initial guess as the point it is.

        The initial guess is the frames' mean: a saturated star there is its blurred halo round a clipped core.  With
        the core censored the data say only "at least the clip level" there, so a point and a point + halo explain the
        unsaturated wings about equally well, and the multiplicative update - which keeps a pixel's light where it starts
        when the data do not tell otherwise - left the starting halo in place: a flat halo of +2-3 sigma to ~20 px in the
        censored channels round the brightest stars, coloured glows after finishing (M 31 2026-10-23; the blue core,
        unsaturated, was fitted and had none; 400 iterations did not move it).  Per core (pixels saturated in at least
        half the frames covering them, grown by ``grow`` px): each channel's light above a robust plane at ``bg_ring``
        px, radially symmetric part (ring medians, out to where green's falls below half a ring's standard error for 3
        rings) moved into a point at the latent's resolution g_sigma on the green core's light-weighted centre; the
        rest of the guess (neighbours, the galaxy, the noise) is kept; inside the censored region the guess is the plane
        and the point.  Light is conserved; the restoration then fits it.  An extended core (light beyond the core over
        ``ext_max`` x the core's in the guess - a galaxy nucleus) is left as it is."""
        import torch
        from scipy.ndimage import label
        C = x.shape[1]
        H, W = core.shape
        g_ = 1 if C > 1 else 0
        grown = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))) > 0
        lab, n = label(grown)
        xc = x[0].detach().cpu().numpy().astype(np.float64)
        xp = np.pad(xc, ((0, 0), (R, R), (R, R)), constant_values=np.nan)
        lp = np.pad(lab, ((R, R), (R, R)), constant_values=-1)
        vp = np.pad(valid if valid is not None else np.ones_like(xc, bool), ((0, 0), (R, R), (R, R)), constant_values=False)
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float64)
        keff = None
        vfp = np.pad(valid_frac, ((0, 0), (R, R), (R, R)), constant_values=0.0) if valid_frac is not None else None
        if kmean is not None and valid_frac is not None:
            # (each channel's star image: the mean kernel * g_sigma, centred in a window)
            kc = kmean.shape[-1] // 2
            h_ = min(kc, R)
            g1 = np.exp(-(np.arange(-4, 5) ** 2) / (2.0 * sigma * sigma)); g1 /= g1.sum()
            keff = np.zeros((C, 2 * R + 1, 2 * R + 1))
            for c in range(C):
                keff[c, R - h_:R + h_ + 1, R - h_:R + h_ + 1] = kmean[c if kmean.shape[0] > 1 else 0][kc - h_:kc + h_ + 1, kc - h_:kc + h_ + 1]
                keff[c] = cv2.sepFilter2D(keff[c], -1, g1, g1, borderType=cv2.BORDER_CONSTANT)
        info = []
        done = 0
        for q in range(1, n + 1):
            ys, xs = np.nonzero(lab == q)
            if len(ys) > MAX_SAT_CORE_PX:
                continue
            iy, ix = int(round(float(ys.mean()))) + R, int(round(float(xs.mean()))) + R
            win = xp[:, iy - R:iy + R + 1, ix - R:ix + R + 1]
            lw = lp[iy - R:iy + R + 1, ix - R:ix + R + 1]
            reg = lw == q
            have = np.isfinite(win).all(0)
            free = (lw == 0) & have
            gw = np.nan_to_num(win[g_])
            rr0 = np.hypot(yy, xx)
            bgm0 = (rr0 >= bg_ring[0]) & (rr0 <= bg_ring[1]) & free
            if bgm0.sum() < 50:
                continue
            cw = np.clip(gw - np.median(gw[bgm0]), 0, None) * reg
            if cw.sum() <= 0:
                continue
            ccy = float((cw * yy).sum() / cw.sum()); ccx = float((cw * xx).sum() / cw.sum())
            if valid_frac is not None:
                ccy, ccx = ImageMMEngine._hole_centre(reg, vfp[g_, iy - R:iy + R + 1, ix - R:ix + R + 1], yy, xx, (ccy, ccx))
            rr = np.hypot(yy - ccy, xx - ccx)
            rb = np.rint(rr).astype(int)
            bgm = (rr >= bg_ring[0]) & (rr <= bg_ring[1]) & free
            plane = np.stack([(lambda cf: cf[0] + cf[1] * xx + cf[2] * yy)(ImageMMEngine._robust_plane(win[c][bgm], yy[bgm], xx[bgm])) for c in range(C)])
            above = win - plane
            nr = bg_ring[0]
            prof = np.zeros((C, nr + 1))
            se = np.zeros(nr + 1)
            for k in range(nr + 1):
                sel = (rb == k) & (free | reg)
                if sel.sum() == 0:
                    continue
                for c in range(C):
                    prof[c, k] = np.median(above[c][sel])
                se[k] = 1.2533 * float(noise[g_]) / math.sqrt(sel.sum())
            r_in = float(rr[reg].max()) + 1.0
            r_out, low = nr, 0
            for k in range(int(math.ceil(r_in)), nr + 1):
                low = low + 1 if prof[g_, k] < 0.5 * se[k] else 0
                if low == 3:
                    r_out = k - 2
                    break
            ks = np.arange(nr + 1)
            prof = np.clip(prof, 0, None) * np.clip((r_out + 1.5 - ks) / 3.0, 0.0, 1.0)[None]
            area = np.array([((rb == k) & (free | reg)).sum() for k in range(nr + 1)], np.float64)
            inside = (rr <= r_in) & (free | reg)
            L_core = [float(np.clip(above[c][inside], 0, None).sum()) for c in range(C)]
            L_ext = [float((prof[c] * area)[ks > r_in].sum()) for c in range(C)]
            ratio = L_ext[g_] / max(L_core[g_], 1e-30)
            if L_core[g_] <= 0 or ratio > ext_max:
                info.append((ix - R, iy - R, round(r_out, 1), round(ratio, 3)))
                continue
            Mr = np.stack([np.interp(rr, ks, prof[c], right=0.0) for c in range(C)])
            S = np.exp(-(rr ** 2) / (2.0 * sigma * sigma))
            S /= S.sum()
            # the star's light: the clipped data's light is only a lower bound (the core holds more than the clip
            # level); PSF photometry of the unsaturated wings measures it - per channel, the ring medians above the
            # plane against the star image's (mean kernel * g_sigma, at the core's centre) over the ``fit_w`` px beyond
            # the zone where under 95 % of the frames have data, least squares (ring medians: neighbours).  Started
            # from the clipped light, the multiplicative update - no data in the core - never grew the point to it and
            # kept the difference as a halo (M 31 2026-10-24: the field's brightest star at 0.80 of its wings' light, a
            # red-green halo of 6-7 % of the data's to 40 px; calibrating the wing up to explain it made every other
            # star's wing 1.3-1.6 x too heavy - dark moats)
            Ltot = [L_core[c] + L_ext[c] for c in range(C)]
            Fw = [None] * C
            if keff is not None:
                vg = vfp[g_, iy - R:iy + R + 1, ix - R:ix + R + 1]
                sp_, _ = label((vg < 0.95) | reg)
                ids_ = np.unique(sp_[reg & (sp_ > 0)])
                spz = np.isin(sp_, ids_[ids_ > 0]) | reg
                r_d = float(rr[spz].max()) + 1.0
                r_f = max(r_d + 6.0, min(float(r_out), r_d + fit_w))
                from scipy.ndimage import shift as _nd_shift
                for c in range(C):
                    Kc = _nd_shift(keff[c], (ccy, ccx), order=3, mode="constant")
                    dense = (vfp[c, iy - R:iy + R + 1, ix - R:ix + R + 1] >= 0.95) & free & ~spz
                    num = den = 0.0
                    for k in range(int(math.ceil(r_d)), int(r_f) + 1):
                        sel = dense & (rb == k)
                        if sel.sum() < 8:
                            continue
                        kk = float(np.median(Kc[sel]))
                        num += sel.sum() * float(np.median(above[c][sel])) * kk
                        den += sel.sum() * kk * kk
                    if den > 0 and num > 0:
                        Fw[c] = num / den
            Fs = [max(Ltot[c], Fw[c]) if Fw[c] is not None else Ltot[c] for c in range(C)]
            info.append((ix - R, iy - R, round(r_out, 1), round(ratio, 3), [round(Fs[c] / max(Ltot[c], 1e-30), 3) for c in range(C)]))
            for c in range(C):
                rest = np.where(rr <= r_in, 0.0, above[c] - Mr[c])            # (inside the censored region: no data)
                new = plane[c] + rest + Fs[c] * S
                win[c] = np.where(have & (rr <= r_out + 2), np.maximum(new, 1e-6), win[c])
            done += 1
        xc = xp[:, R:R + H, R:R + W]
        return torch.from_numpy(xc.astype(np.float32)).to(x.device)[None], {"cores": int(n), "points": done, "stars": info}

    @staticmethod
    def _saturated_point_sources(x, core, lb, sigma: float, r: int = 1, grow: int = 3, ring: int = 4):
        """PIPELINE (change 24): each saturated core of the latent as the point source it is.

        In a core saturated in (most of) the frames the data say only "at least the clip level"; the star's light is
        fixed by its unsaturated wings, its distribution inside the core by nothing - and the multiplicative update,
        started from the frames' clipped values, kept it a flat disc (M 31 2026-10-22: 5 px plateaus at the clip level).
        A star is a point: its light in the latent is the latent's own resolution element, the Gaussian g_sigma (the
        latent is the sky seen through g_sigma, Eq. 11).  Per core (grown by ``grow`` px): each channel's light above
        the local background (median of the ``ring`` px round it) is put into g_sigma at the light-weighted centroid
        (green); the total is unchanged, and the following iterations refine it against the wings with the saturated
        pixels still censored.  The point blurred by the PSF exceeds the clip level where the frames saturate."""
        import torch
        from scipy.ndimage import label
        C = x.shape[1]
        H, W = core.shape
        if not core.any():
            return x, {"cores": 0}
        grown = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))) > 0
        lab, n = label(grown)
        xc = x[0].detach().cpu().numpy().astype(np.float64)
        lbn = torch.broadcast_to(lb, x.shape)[0].detach().cpu().numpy().astype(np.float64)
        done = 0
        for q in range(1, n + 1):
            reg = lab == q
            ys, xs = np.nonzero(reg)
            y0, y1, x0, x1 = max(0, ys.min() - ring - 2), min(H, ys.max() + ring + 3), max(0, xs.min() - ring - 2), min(W, xs.max() + ring + 3)
            rg = reg[y0:y1, x0:x1]
            rn = (cv2.dilate(rg.astype(np.uint8), np.ones((2 * ring + 1, 2 * ring + 1), np.uint8)) > 0) & ~(lab[y0:y1, x0:x1] > 0)
            if rn.sum() < 8:
                continue
            g_ = 1 if C > 1 else 0
            win = xc[:, y0 * r:y1 * r, x0 * r:x1 * r]
            flo = lbn[:, y0 * r:y1 * r, x0 * r:x1 * r]
            rgh = np.kron(rg, np.ones((r, r), bool)) if r > 1 else rg
            rnh = np.kron(rn, np.ones((r, r), bool)) if r > 1 else rn
            bg = np.array([np.median(win[c][rnh]) for c in range(C)])
            wgt = np.clip(win[g_] - bg[g_], 0, None) * rgh
            if wgt.sum() <= 0:
                continue
            yy, xx = np.mgrid[:win.shape[1], :win.shape[2]]
            cy, cx = (wgt * yy).sum() / wgt.sum(), (wgt * xx).sum() / wgt.sum()
            s_ = float(sigma) * r
            G = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s_ * s_)) * rgh
            if G.sum() <= 0:
                continue
            G /= G.sum()
            for c in range(C):
                F = float(np.clip(win[c] - bg[c], 0, None)[rgh].sum())
                new = bg[c] + F * G
                win[c] = np.where(rgh, np.maximum(new, flo[c]), win[c])
            xc[:, y0 * r:y1 * r, x0 * r:x1 * r] = win
            done += 1
        out = torch.from_numpy(xc.astype(np.float32)).to(x.device)[None]
        return out, {"cores": int(n), "made_points": done}

    @staticmethod
    def _bloom_zone(sat, grow: int, scale: float = BLOOM_SCALE):
        """PIPELINE (changes 29, 33): (C, H, W) the pixels next to a frame's saturated pixels ``sat`` (C, H, W) that are
        left out as data the linear model does not describe (the saturation / bleed masks of survey pipelines, grown):
        per channel within ``grow`` px of its saturated pixels, and round a saturated patch (the channels' union) of
        equivalent radius r_eq within ``scale`` x r_eq px of it where that is further.  The light next to a saturated
        core exceeds the PSF's out to a distance that grows with the core: 2-5 px for cores of 2-8 px (M 31, against
        the restoration, by distance from the core), 8-9 px for the field's brightest star (r_eq ~11; per frame against
        its own wing fitted 10+ px out: G +0.33 / +0.14 / +0.04 at 5 / 7 / 9 px, B +0.54 / +0.21 / +0.07; at a fixed
        radius it falls with the distance from the saturated pixels).  With a fixed 4 px zone that light pulled the
        restoration's point to 0.77 of its wings' light and stayed as a ring at 12-16 px and a red-green halo to 40 px;
        modelled as a feature of the PSF instead (fitted on that star's frames) the ring and halo stayed."""
        from scipy.ndimage import label, find_objects
        k_ = 2 * int(grow) + 1
        se_ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_, k_))
        near = np.stack([cv2.dilate(sat[c].astype(np.uint8), se_) > 0 for c in range(sat.shape[0])])
        anys = sat.any(0)
        r_big = (grow + 1) / max(scale, 1e-6)
        if anys.sum() > np.pi * r_big * r_big:
            lab, n = label(anys)
            H, W = anys.shape
            for q, sl in enumerate(find_objects(lab), start=1):
                if sl is None:
                    continue
                area = int((lab[sl] == q).sum())
                g2 = int(round(scale * math.sqrt(area / math.pi)))
                if g2 <= grow:
                    continue
                y0, y1 = max(0, sl[0].start - g2), min(H, sl[0].stop + g2)
                x0, x1 = max(0, sl[1].start - g2), min(W, sl[1].stop + g2)
                se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g2 + 1, 2 * g2 + 1))
                z = cv2.dilate((lab[y0:y1, x0:x1] == q).astype(np.uint8), se2) > 0
                near[:, y0:y1, x0:x1] |= z[None]
        return near & ~sat

    @staticmethod
    def _local_background(x_lr, valid, sources=None, box: int = 32):
        """The background under the sources, per channel (1, C, H, W), for the lower bound: the source-masked mesh
        background of SExtractor (Bertin & Arnouts 1996; photutils Background2D) - per ``box`` x ``box`` px cell the
        3-sigma clipped median of the pixels outside ``sources`` ((H, W) bool, ``_source_mask``), a cell with less than
        half of them free filled from its neighbours, the mesh median-filtered over 3 x 3 cells and interpolated
        bicubically - and the smaller of that and the 5 x 5 minimum of the image smoothed over 1 px (the bottom of a dark
        lane narrower than a cell).

        The bound had been taken from the image itself (min of a 36 px median and that 5 x 5 minimum): inside a bright
        star's blurred halo both are the halo, and the restoration could not take that light into the star - a flat
        plateau to ~20 px round every bright star, coloured glows after finishing (M 31 2026-10-23: +0.5-1.5 sigma at
        11-21 px with the bound, ~0 without)."""
        import torch
        from scipy.ndimage import median_filter
        _, C, H, W = x_lr.shape
        out = torch.empty_like(x_lr)
        a = x_lr[0].detach().cpu().numpy()
        v = valid[0].detach().cpu().numpy()
        ny, nx = max(1, int(math.ceil(H / box))), max(1, int(math.ceil(W / box)))
        for c in range(C):
            img = np.where(v[c], a[c], np.nan).astype(np.float32)
            free = np.isfinite(img) & (~sources if sources is not None else True)
            mesh = np.full((ny, nx), np.nan, np.float64)
            for j in range(ny):
                for i in range(nx):
                    cell = img[j * box:(j + 1) * box, i * box:(i + 1) * box]
                    fm = free[j * box:(j + 1) * box, i * box:(i + 1) * box]
                    if fm.sum() < 0.5 * fm.size:
                        continue
                    q = cell[fm].astype(np.float64)
                    for _ in range(3):
                        md = np.median(q); sd = 1.4826 * np.median(np.abs(q - md))
                        keep = np.abs(q - md) <= 3.0 * max(sd, 1e-30)
                        if keep.all():
                            break
                        q = q[keep]
                    mesh[j, i] = np.median(q)
            if not np.isfinite(mesh).any():
                mesh[:] = np.nanmedian(img) if np.isfinite(img).any() else 0.0
            while not np.isfinite(mesh).all():                   # (empty cells from their neighbours, outward)
                pad = np.pad(mesh, 1, constant_values=np.nan)
                nb = np.stack([pad[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
                with np.errstate(all="ignore"):
                    fillv = np.nanmedian(nb, 0)
                mesh = np.where(np.isfinite(mesh), mesh, fillv)
            mesh = median_filter(mesh, size=3, mode="nearest")
            # (cell centres at (k + 0.5) box: cubic interpolation onto the pixel grid)
            yy = (np.arange(H) + 0.5) / box - 0.5
            xx = (np.arange(W) + 0.5) / box - 0.5
            from scipy.ndimage import map_coordinates
            gy, gx = np.meshgrid(np.clip(yy, 0, ny - 1), np.clip(xx, 0, nx - 1), indexing="ij")
            bgc = map_coordinates(mesh, [gy, gx], order=3, mode="nearest").astype(np.float32)
            fill = np.nanmedian(img) if np.isfinite(img).any() else 0.0
            g1 = cv2.erode(cv2.GaussianBlur(np.where(np.isfinite(img), img, fill).astype(np.float32), (0, 0), 1.0), np.ones((5, 5), np.uint8))
            # (everywhere, the sources too: in a star's halo the 5 x 5 minimum is the halo, above the mesh, so the mesh
            # stands; but under a galaxy's bulge - all of it inside the stars' masks - a dust lane narrower than a cell
            # was held up at the mesh's level, flat patches in the restoration, M 31 2026-10-24)
            lvl = np.minimum(bgc, g1)
            out[0, c] = torch.from_numpy(lvl.astype(np.float32)).to(x_lr.device)
        return out

    @staticmethod
    def _source_mask(img, points, noise: float, thresh: float = 0.5, r_min: float = 3.0, r_max: int = 80, sky: tuple = (86, 96)):
        """(H, W) bool: each source at ``points`` ((x, y)) out to where its light in ``img`` (H, W) - the median of each
        1 px ring less the median at ``sky`` px - first falls below ``thresh`` x ``noise`` (the image's pixel sigma) for
        2 rings in a row; at least ``r_min``, at most ``r_max`` px (the segmentation-and-dilation source mask of the
        background estimation, sized by each source's own measured profile)."""
        H, W = img.shape
        R = sky[1]
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        rb = np.rint(np.hypot(yy, xx)).astype(int)
        skym = (rb >= sky[0]) & (rb <= sky[1])
        pad = np.pad(img.astype(np.float64), R, constant_values=np.nan)
        out = np.zeros((H, W), bool)
        rings = [rb == k for k in range(r_max + 1)]
        for (x, y) in points:
            ix, iy = int(round(x)), int(round(y))
            if not (0 <= ix < W and 0 <= iy < H):
                continue
            w = pad[iy:iy + 2 * R + 1, ix:ix + 2 * R + 1]
            s0 = np.nanmedian(w[skym])
            rad, low = float(r_max), 0
            for k in range(int(r_min), r_max + 1):
                v = w[rings[k]]
                v = v[np.isfinite(v)]
                if v.size == 0:
                    continue
                low = low + 1 if np.median(v) - s0 < thresh * noise else 0
                if low == 2:
                    rad = float(k - 1)
                    break
            rr_ = int(math.ceil(rad))
            ya, yb, xa, xb = max(0, iy - rr_), min(H, iy + rr_ + 1), max(0, ix - rr_), min(W, ix + rr_ + 1)
            gy, gx = np.mgrid[ya:yb, xa:xb]
            out[ya:yb, xa:xb] |= (gy - iy) ** 2 + (gx - ix) ** 2 <= rad * rad
        return out

    def _variance(self, d_norm, var_floor, shot_noise_scale, read_noise):
        torch, _, _, _ = _torch()
        if self.formal_mode:
            v = (shot_noise_scale * d_norm) + (read_noise * read_noise)
        else:
            v = var_floor + (shot_noise_scale * d_norm) + (read_noise * read_noise)
        return torch.clamp(v, min=1e-10)

    def measure_wing_correction(self, store: FrameStore, x, R: int = 80, bg_ring: tuple = (90, 104), r0: float = 6.0,
                                r1: float = 9.0):
        """PIPELINE (change 31): the PSF wing calibrated on the bright stars.  Round a bright star the data cannot tell
        a wing error from light round the star: a wing too heavy is answered by a moat that keeps the fit, one too light
        by a halo (M 31 2026-10-23: the forward model fitted to +-0.06 sigma per frame beyond 10 px, while the
        restoration held -0.7 sigma moats and +1 sigma rings round the brightest stars).  The calibration takes what PSF
        modelling on stars always takes - an isolated star sits on a smooth sky - and gives the radially symmetric light
        the restoration ``x`` leaves round each saturated star (ring medians above a robust plane at ``bg_ring`` px, out
        to R) back to the PSF: with e(r) that light per pixel, F the star's light and K(r) the frames' mean kernel's light
        per pixel at r, the wing factor is 1 + the least-squares delta over the stars, sum e F K / sum (F K)^2, smoothed over 1.5 px,
        clipped to 0.5 .. 2 per pass, 1 inside ``r0`` and joined in by ``r1`` px (inside, each frame's own measured core).  {channel:
        factors 0..R} or None."""
        from scipy.ndimage import label, gaussian_filter1d
        info = self.last_run_info
        core = getattr(self, "_sat_core", None)
        if core is None or not core.any():
            return None
        # (measured on the restoration as fitted - the saturated cores rewritten afterwards are the star's model, not
        # light the fit left - and only where the frames' data are not sparse: inside, the latent rests on a few frames
        # or none.  Measured on the rewritten cores, the model's own light at 8-20 px counted as a wing error, and the
        # field's brightest star kept a red-green halo at 18-40 px, M 31 2026-10-24)
        xf = getattr(self, "_x_fitted", None)
        xc = np.asarray(xf if xf is not None and np.shape(xf) == np.shape(x) else x, np.float64)
        C, H, W = xc.shape
        vfr = getattr(self, "_vfrac", None)
        dense = (np.asarray(vfr) >= SPARSE_FRAC) if vfr is not None and np.shape(vfr) == (C, H, W) else np.ones((C, H, W), bool)
        dense = np.stack([cv2.erode(d_.astype(np.uint8), np.ones((3, 3), np.uint8), borderValue=1) > 0 for d_ in dense])
        km = np.mean([info["kernels"][t] for t in info["frames_used"]], 0).astype(np.float64)
        kc = km.shape[-1] // 2
        rk = np.rint(np.hypot(*(np.mgrid[:km.shape[-1], :km.shape[-1]] - kc))).astype(int)
        K = [np.array([km[c if km.shape[0] > 1 else 0][rk == q].mean() for q in range(R + 1)]) for c in range(C)]
        lab, n = label(core)
        Rw = bg_ring[1]
        yy, xx = np.mgrid[-Rw:Rw + 1, -Rw:Rw + 1].astype(np.float64)
        rr = np.hypot(yy, xx); rb = np.rint(rr).astype(int); th = np.arctan2(yy, xx)
        bgm = (rr >= bg_ring[0]) & (rr <= bg_ring[1])
        ext = {(int(s_[0]), int(s_[1])) for s_ in (info.get("saturated_photometry") or {}).get("stars", []) if len(s_) > 2 and s_[2] == "extended"}
        ratios = [[] for _ in range(C)]
        used = 0
        for q in range(1, n + 1):
            ys, xs = np.nonzero(lab == q)
            if len(ys) > MAX_SAT_CORE_PX:
                continue
            cy, cx = int(round(float(ys.mean()))), int(round(float(xs.mean())))
            if not (Rw <= cx < W - Rw and Rw <= cy < H - Rw):
                continue
            if any(abs(cx - e[0]) <= 3 and abs(cy - e[1]) <= 3 for e in ext):
                continue
            other = (lab[cy - Rw:cy + Rw + 1, cx - Rw:cx + Rw + 1] > 0) & (lab[cy - Rw:cy + Rw + 1, cx - Rw:cx + Rw + 1] != q)
            for c in range(C):
                w = xc[c, cy - Rw:cy + Rw + 1, cx - Rw:cx + Rw + 1]
                ok = np.isfinite(w) & ~other
                okd = ok & dense[c, cy - Rw:cy + Rw + 1, cx - Rw:cx + Rw + 1]
                pl = ImageMMEngine._robust_plane(w[bgm & ok], yy[bgm & ok], xx[bgm & ok])
                ab = w - (pl[0] + pl[1] * xx + pl[2] * yy)
                e = np.array([np.median(ab[(rb == k) & okd]) if ((rb == k) & okd).sum() >= 8 else np.nan for k in range(R + 1)])
                F = float(ab[(rr <= R) & ok].sum())
                if F <= 0:
                    continue
                ratios[c].append((e, F * K[c]))
            used += 1
        if used < 2:
            return None
        out = {}
        ks = np.arange(R + 1)
        for c in range(C):
            # (least squares over the stars per ring: delta = sum e FK / sum (FK)^2 - the stars carry weight by the light
            # the model puts there; the median of e / FK let the faint stars' noise set the far wing)
            E = np.array([v[0] for v in ratios[c]]); M = np.array([v[1] for v in ratios[c]])
            M = np.where(np.isfinite(E), M, 0.0); E = np.nan_to_num(E)          # (a star's sparse rings carry no weight)
            qv = (E * M).sum(0) / np.maximum((M * M).sum(0), 1e-60)
            f = np.clip(1.0 + gaussian_filter1d(qv, 1.5, mode="nearest"), 0.5, 2.0)
            w_ = np.clip((ks - r0) / max(r1 - r0, 1e-6), 0.0, 1.0)
            out[c] = list((1.0 + w_ * (f - 1.0)).astype(float))
        self.last_run_info["wing_correction"] = {int(c): [round(v, 3) for v in out[c][::2]] for c in out if not isinstance(c, tuple)}
        self.last_run_info["wing_correction_stars"] = used
        return out

    def run_calibrated(self, store: FrameStore, wing_calibration: bool = True, calibration_iterations: int = 80,
                       calibration_passes: int = 2, log=print, **kw):
        """``run_imagemm`` with the PSF wing calibrated on the bright stars (``measure_wing_correction``): restorations of
        ``calibration_iterations``, each correcting the wing by the light left round the stars (a moat held at the lower
        bound shows only part of its depth: the factors compound over ``calibration_passes``), then the restoration."""
        if not wing_calibration:
            return self.run_imagemm(store, log=log, **kw)
        kw1 = dict(kw)
        kw1["iterations"] = min(int(kw.get("iterations", 150)), int(calibration_iterations))
        corr = None
        hist = []
        for p_ in range(int(calibration_passes)):
            log(f"ImageMM: PSF wing calibration {p_ + 1}/{calibration_passes}")
            x1 = self.run_imagemm(store, log=log, wing_correction=corr, **kw1)
            c1 = self.measure_wing_correction(store, x1)
            if c1 is None:
                break
            hist.append(self.last_run_info.get("wing_correction"))
            new_ = {}
            for c in c1:
                if isinstance(c, tuple):
                    new_[c] = np.asarray(c1[c]) + (np.asarray(corr[c]) if corr and c in corr else 0.0)
                else:
                    new_[c] = list(np.asarray(c1[c]) * (np.asarray(corr[c]) if corr and c in corr else 1.0))
            corr = new_
        x = self.run_imagemm(store, log=log, wing_correction=corr, **kw)
        self.last_run_info["wing_correction"] = {int(c): [round(float(v), 3) for v in corr[c][::2]] for c in corr if not isinstance(c, tuple)} if corr else None
        self.last_run_info["wing_correction_passes"] = hist
        return x

    def run_imagemm(self, store: FrameStore, iterations=20, upscale_factor=1,
                    huber_delta=0.05, kappa=2.0, mu=0.1, stop_epsilon=1e-5,
                    shot_noise_scale=1.0, read_noise=0.0, saturation_level=None,
                    psf_sigma=1.0, variance_from="model", per_channel_psf=True, latent_sigma=1.0, lower_bound="local", reference_psf=True, photometry=True, noise_model="ptc", huber_sigma=2.0, halo=False, saturated_points=False, saturated_photometry=True, psf_wings=False, psf_asymmetry=True, psf_radial_wings=True, bound_sigma=1.0, bloom_grow=4, point_init=True, censor_saturated=False, wing_correction=None, callback=None, cancel=None, log=print):
        """The script's ``run_imagemm`` on a ``FrameStore`` (stars from ``store.stars``).  Returns the restored image
        (C, r H, r W) float32 in the frames' units; ``self.last_run_info`` says how it went."""
        torch, _, F, _ = _torch()
        self._norm = None
        r = int(upscale_factor)
        n_imgs = len(store)
        channels, height, width = store.shape
        dev = self.device
        say = callback or (lambda i, n, m: None)
        check = cancel or (lambda: None)

        self.last_run_info = {
            'converged': False, 'iters_done': 0, 'iters_max': int(iterations),
            'mean_diff': None, 'stop_epsilon': float(stop_epsilon),
            'mu': float(mu), 'kappa': float(kappa), 'data_mode': "MONO" if channels == 1 else "RGB", 'r': r,
            'mean_diff_history': [],
        }

        # 1. Analyze First Image & Init
        raw = store.frame(0)
        norm_ref = raw if raw.ndim == 2 else raw[1 if channels > 1 else 0]
        valid_ref = norm_ref[np.isfinite(norm_ref) & (norm_ref > 1e-7)]
        if valid_ref.size == 0:
            valid_ref = norm_ref[np.isfinite(norm_ref)]
        norm_factor = float(np.percentile(valid_ref, 99.5)) if valid_ref.size else 1.0
        if norm_factor <= 1e-5:
            norm_factor = 1.0
        sat_norm = None
        if saturation_level:
            sat_norm = (float(saturation_level) * SATURATION_FRAC) / float(max(norm_factor, 1e-6))
        log(f"Data: {channels}x{width}x{height} | Norm: {norm_factor:.6f} | Mode: {self.last_run_info['data_mode']} | SR: x{r}")
        self.last_run_info['norm_factor'] = norm_factor

        # frames are read by a helper thread while the GPU works on the previous one
        pool = ThreadPoolExecutor(max_workers=1)
        lock = threading.Lock()

        def fetch(i):
            with lock:
                fa, fb = store.frame(i), store.saturated(i)
                if bloom_grow and fb.any():
                    fa = np.where(self._bloom_zone(fb, int(bloom_grow)), np.nan, fa).astype(np.float32)
                a, b = torch.from_numpy(fa), torch.from_numpy(fb)
                return (a.pin_memory(), b.pin_memory()) if dev.type == "cuda" else (a, b)

        def frames(which=None):
            idx = list(range(n_imgs)) if which is None else list(which)
            if not idx:
                return
            fut = pool.submit(fetch, idx[0])
            for j, i in enumerate(idx):
                a = fut.result()
                if j + 1 < len(idx):
                    fut = pool.submit(fetch, idx[j + 1])
                yield i, a[0].to(dev, non_blocking=True).unsqueeze(0), a[1].to(dev, non_blocking=True)[None]   # sat: (1, C, H, W)

        # 2. PSF Phase + initial guess
        # PIPELINE (change 13, reference_psf / photometry): a multi-frame model needs each frame's kernel to say where a
        # point of the LATENT grid lands in that frame.  The script centres every frame's PSF on that frame's own star
        # positions, which removes the frame's residual registration (IC 1805: per-frame offsets -0.19..+0.21 px p5-p95)
        # and turns it into blur the model does not know about; and it has no photometric scale, so frames 10-20 %
        # dimmer (transparency) disagree on every star.  Pass A: the frames' mean and sky levels; a reference star
        # catalogue centroided on that mean.  Pass B: each frame's flux scale s_t (median star flux ratio) and sky b_t,
        # y -> (y - b_t) / s_t + b_ref (the stack's additive + scale normalisation), and its PSF from cut-outs centred
        # on the reference positions.
        log(f"Loading dataset ({n_imgs} frames) and Solving PSFs...")
        var_floors = []
        sky_t = np.zeros((n_imgs, channels), np.float64)
        scale_t = np.ones(n_imgs, np.float64)
        acc = np.zeros((channels, height, width), np.float32)
        cnt = np.zeros((channels, height, width), np.float32)
        satc = np.zeros((height, width), np.int32)
        rng = np.random.default_rng(0)
        n_s = min(height * width, 60_000)
        pick = rng.choice(height * width, n_s, replace=False)
        cube = np.full((n_imgs, channels, n_s), np.nan, np.float32)      # the photon-transfer sample
        for idx, d, sat in frames():                             # pass A
            check()
            d_np = d[0].cpu().numpy().copy()                    # (on the CPU .numpy() shares the tensor's memory)
            s_np = sat[0].cpu().numpy()
            d_np[s_np] = np.nan
            satc += s_np.any(0)
            for c in range(channels):
                v = d_np[c][np.isfinite(d_np[c])]
                sky_t[idx, c] = float(np.median(v[:: max(1, v.size // 500_000)])) / norm_factor if v.size else 0.0
            cube[idx] = d_np.reshape(channels, -1)[:, pick] / norm_factor
            ok = np.isfinite(d_np)
            acc[ok] += d_np[ok] / norm_factor
            cnt[ok] += 1
            if not self.formal_mode:
                sigma_bg = self._robust_bg_sigma(np.where(np.isfinite(d_np), np.maximum(d_np, 0.0), 0.0) / norm_factor)
                var_floors.append(sigma_bg * sigma_bg)
            else:
                var_floors.append(0.0)
            say(idx + 1, 2 * n_imgs, f"frame statistics {idx + 1}/{n_imgs}")
        ref_all = (acc / np.maximum(cnt, 1)).astype(np.float32)
        ref_img = ref_all[1 if channels > 1 else 0]
        # (a pixel missing from some frame - saturated, or off a frame's edge - reads low in the mean: NaN for the
        # halo measurement, or a saturated star's empty core inflated its halo fraction ~10 x: dark 110 px discs)
        # (a channel's pixel is measured where at least half the frames cover it - Bayer-drizzle holes vary per frame and
        # channel - and no frame saturated it: a saturated core reads low in the mean)
        full = (cnt >= 0.5 * cnt.reshape(channels, -1).max(1)[:, None, None]) & (satc[None] == 0)
        ref_full = np.where(full, ref_all, np.nan).astype(np.float32)
        del satc
        del acc, cnt
        sky_ref = np.median(sky_t, 0)
        catalog = self._reference_catalog(ref_img, store) if (reference_psf or photometry) else []
        self.last_run_info['reference_stars'] = len(catalog)
        self.last_run_info['reference_catalog'] = [(round(float(a), 2), round(float(b), 2)) for a, b in catalog]
        # PIPELINE (change 18): the optics' static halo.  IC 1805: 90-91 % of a star's light lies within 40 px, the rest
        # out to ~110 px (stack, 34 isolated stars).  A kernel ending at 40 px left that light to the latent, which grew a
        # halo just outside the kernel's reach - a moat round every bright star by contrast.  The halo is the same in
        # every frame (scattered light), too faint to measure per frame: its radial profile per channel is measured once
        # on the frames' mean image and joined to each frame's measured core (30-40 px blend).
        halos = [self._halo_profile(ref_full[c], catalog) for c in range(channels)] if halo else [None] * channels
        self.last_run_info['halo'] = [None if h_ is None else {"stars": h_[1], "light_beyond_40px": round(float(h_[0][41:].sum()), 4)}
                                      for h_ in halos]
        ref_flux = self._aperture_fluxes(ref_img, catalog) if photometry else None
        # PIPELINE (change 26): the PSF's wings from the frames' mean image.  A single frame has too few stars in R and B
        # to detect the wings (M 31: 15-28 stars; the kernel ended at ~10 px holding 100 % of the light, where the mean
        # image's 74 stars show 95 % within 10 px and the rest out to 25 px).  Each frame's kernel keeps its own core out
        # to its detected radius and takes its wings from the mean image's PSF of the same channel (joined over 4 px,
        # the wing scaled to the frame's light in the join).
        # PIPELINE (change 27): the PSF's non-radial part.  The optics put a fixed one-sided wing on every star (M 31 /
        # DWARF 3: dipole 0.33 at 5-15 px, the same direction in every frame); a single frame's PSF stack is too noisy
        # out there and its wings are made radial (_radial_wings), so the kernels had 0.08-0.17 - the restoration kept
        # the rest as sky light, a fan on one side of every bright star and a hole on the other (dipole 0.63 against
        # the data's 0.36).  It is measured once per channel on the frames' mean from the bright unsaturated stars and
        # added to every frame's kernel (zero sum per ring: the kernel's radial profile - its seeing - is the frame's).
        # PIPELINE (change 28): the PSF's radial wing beyond the seeing core.  A single frame's PSF stack cannot measure it
        # beyond ~5 px (M 31: frame 100's kernel 1.84 / 1.17 / 0.49 at 6 / 7 / 9 px relative to 8 px, the stars 2.9 / 1.6
        # / 0.64; lumps of noise at 12-13 px): the restoration made up the difference as a ring of light round every bright
        # star.  The wing is the optics' (the same for saturated and unsaturated stars): its radial profile per channel
        # is measured on the frames' mean (median over the bright stars of their ring medians), and every frame's kernel
        # takes it beyond WING_JOIN px, scaled to the frame's own kernel at the join.
        # The far wing (the optics' scattered light) is measured once on the channels' sum - achromatic to first order, and
        # R / B alone have too few bright stars: their wings ended at ~20 px where green's reached 35, and the restoration
        # took more of green's halo into the core than R's / B's - magenta halos round the bright stars (M 31 2026-10-23).
        # Each channel keeps its own profile to 10 px (seeing, debayer) and joins the common wing over 10-14 px.
        wings = {}
        w_all = None
        chroma = None
        if psf_radial_wings and reference_psf and len(catalog) >= 5:
            w_all = self._radial_wing_profile(ref_full.sum(0) if channels > 1 else ref_full[0], catalog, R=FAR_WING_R, bloom=int(bloom_grow))
            chroma = self._wing_chroma(ref_full, catalog, R=FAR_WING_R, bloom=int(bloom_grow)) if channels > 1 else None
            self.last_run_info['psf_wing_chroma'] = None if chroma is None else {int(c): [round(float(v), 3) for v in chroma[c][::5]] for c in range(channels)}
            for c in range(channels):
                w_ = self._radial_wing_profile(ref_full[c], catalog, R=FAR_WING_R, bloom=int(bloom_grow))
                if w_ is None:
                    w_ = w_all if (w_all is None or chroma is None) else (w_all[0] * chroma[c][:len(w_all[0])], w_all[1])
                elif w_all is not None and channels > 1:
                    Pc, Pa = w_[0].copy(), w_all[0]
                    j0, j1 = 10, 14
                    a_ = float(Pc[j0:j1 + 1].sum()) / max(float(Pa[j0:j1 + 1].sum()), 1e-30)
                    rr_ = np.arange(len(Pc))
                    t_ = np.clip((rr_ - j0) / float(j1 - j0), 0.0, 1.0)
                    t_ = 0.5 - 0.5 * np.cos(np.pi * t_)
                    Pa_c = Pa * (chroma[c][:len(Pa)] if chroma is not None else 1.0)
                    w_ = ((1.0 - t_) * Pc + t_ * a_ * Pa_c, w_[1])
                if w_ is not None:
                    wings[c] = w_
        if wing_correction:
            # (the PSF wing calibrated on the bright stars' residuals of a first restoration - measure_wing_correction)
            for c_, corr in wing_correction.items():
                if isinstance(c_, tuple):
                    continue
                c_ = int(c_)
                if c_ in wings:
                    P_, n_ = wings[c_]
                    cr = np.ones(len(P_))
                    m_ = min(len(cr), len(corr))
                    cr[:m_] = np.asarray(corr[:m_], float)
                    wings[c_] = (P_ * cr, n_)
        self.last_run_info['psf_radial_wings'] = {int(c): {"stars": int(v[1]), "profile": [round(float(q), 5) for q in v[0]]}
                                                  for c, v in wings.items()}
        asym = {}
        if psf_asymmetry and reference_psf and len(catalog) >= 5:
            for c in range(channels):
                a_ = self._nonradial_psf(ref_full[c], catalog)
                if a_ is not None:
                    asym[c] = a_
        self.last_run_info['psf_asymmetry'] = {int(c): {"stars": int(v[1]), "dipole": round(float(v[2]), 3), "angle_deg": round(float(v[3]), 1)}
                                               for c, v in asym.items()}
        far_asym = {}
        if psf_asymmetry and psf_radial_wings and reference_psf and len(catalog) >= 5:
            # (per channel: the lobe is not grey - M 31's brightest star, m1 0.13 / 0.15 / 0.22 in R / G / B at 26-40 px;
            # the channels' common lobe left a cyan glow on its bright side)
            fa_all = self._far_nonradial(ref_full.sum(0) if channels > 1 else ref_full[0], catalog, R=FAR_WING_R, bloom=int(bloom_grow))
            for c in range(channels):
                fa_c = self._far_nonradial(ref_full[c], catalog, R=FAR_WING_R, bloom=int(bloom_grow)) if channels > 1 else None
                if fa_c is not None or fa_all is not None:
                    far_asym[c] = fa_c if fa_c is not None else fa_all
        self.last_run_info['psf_far_asymmetry'] = {int(c): {"stars": int(v[1]), "m1_max": round(v[2], 3), "angle_deg": round(v[3], 1), "rings": list(v[4])}
                                                   for c, v in far_asym.items()} or None
        mean_psf = {}
        if psf_wings and reference_psf and len(catalog) >= 5:
            for c in range(channels):
                # (pixels missing from any frame - saturated cores - are NaN: their cut-outs are skipped)
                t_, n_ = self.psf_solver.extract_and_stack_stars(ref_full[c], catalog)
                if n_ >= MIN_PSF_STARS:
                    mean_psf[c] = (t_.detach().cpu().numpy().astype(np.float64), float(self.psf_solver.last_radius))
        self.last_run_info['mean_psf_radius'] = {int(c): v[1] for c, v in mean_psf.items()}
        del ref_full
        # PIPELINE (change 21): frame selection.  Pass P measures each frame's transmission (median star flux against
        # the mean image, unclamped - the old clamp at 0.2 rescaled frames dimmer than that wrongly).  A frame at under
        # MIN_TRANSMISSION x the median (cloud, dawn) or whose PSF cannot be measured is not used: ImageMM fits every frame
        # it is given and has no rejection like the stack's (M 31 2026-10-21: the night's last 34 frames, transmission
        # 0.05-0.6, fitted at chi2 28-261 against 0.5-1 for the others, coloured the latent and lost its faint disk).
        active = list(range(n_imgs))
        dropped = {}
        # PIPELINE (change 23, relative normalisation): each frame is put on the frames' mean image M,
        # y' = (y - P_t) / s_t with s_t its transmission and P_t a plane per channel fitted to tile medians of
        # y - s_t M (the target cancels there; the frame's own sky level and gradient remain).  The script subtracts an
        # absolute sky model per frame (seqsubsky): on a large target that model takes part of the target's own light,
        # differently in every frame (M 31: the faint outer disk lost, star / diffuse flux 1.05-1.21 against the
        # stack).  The stack normalises the same way (offset + scale against the reference) and leaves the absolute
        # gradient to the gradient removal after it.
        planes = np.zeros((n_imgs, channels, 3), np.float64)
        Mn = ref_all
        tile = 64
        for idx, d, sat in frames():                             # pass P
            check()
            dn = d[0].cpu().numpy().copy() / norm_factor
            dn[sat[0].cpu().numpy()] = np.nan
            if photometry and len(catalog) >= 5:
                f_t = self._aperture_fluxes(dn[1 if channels > 1 else 0], catalog)
                ok = np.isfinite(f_t) & np.isfinite(ref_flux) & (ref_flux > 0)
                scale_t[idx] = float(np.median(f_t[ok] / ref_flux[ok])) if ok.sum() >= 5 else np.nan
            s_ = scale_t[idx] if np.isfinite(scale_t[idx]) and scale_t[idx] > 0 else 1.0
            for c in range(channels):
                planes[idx, c] = self._plane_fit(dn[c] - s_ * Mn[c], tile)
            say(idx + 1, n_imgs, f"photometry {idx + 1}/{n_imgs}")
        if photometry and len(catalog) >= 5:
            # transmission = the frame's light per second of exposure against the median frame's: with several exposure times
            # (HDR) the star flux ratio s_t alone made every 60 s frame of a 60 s + 120 s set a "transmission 0.50" cloud frame
            dropped.update(transmission_drops(scale_t, getattr(store, "exptimes", None)))
            scale_t = np.where(np.isfinite(scale_t) & (scale_t > 0), scale_t, 1.0)
            active = [i for i in range(n_imgs) if i not in dropped]
        # the sky level of the common scale: the mean image's 5th percentile per channel (signal above it is shot noise)
        sky_ref = np.array([float(np.nanpercentile(Mn[c][::4, ::4], 5)) for c in range(channels)])
        self._norm = (sky_t, scale_t, sky_ref)
        self._planes = planes
        gy, gx = np.mgrid[:height, :width]
        self._grid = (torch.from_numpy(((gx - width / 2) / width).astype(np.float32)).to(dev),
                      torch.from_numpy(((gy - height / 2) / height).astype(np.float32)).to(dev))
        self.last_run_info['planes_rms_over_sky_rgb'] = None
        # the photon transfer of the frames used, each frame's sample on the common scale
        if noise_model == "ptc":
            py_, px_ = np.divmod(pick, width)
            xs_, ys_ = (px_ - width / 2) / width, (py_ - height / 2) / height
            pl = planes[active][:, :, 0:1] + planes[active][:, :, 1:2] * xs_[None, None] + planes[active][:, :, 2:3] * ys_[None, None]
            sub = (cube[active] - pl) / scale_t[active][:, None, None]
            ptc = self._photon_transfer(sub.astype(np.float32), np.zeros((len(active), channels)), ref_img, pick, scale_t[active])
        else:
            ptc = None
        del cube
        self.last_run_info['photon_transfer'] = ptc

        x_accum = torch.zeros((1, channels, height, width), dtype=torch.float32, device=dev)
        m_sum = torch.zeros_like(x_accum)
        all_accum = torch.zeros((1, channels, height, width), dtype=torch.float32, device=dev)
        all_sum = torch.zeros_like(all_accum)
        sat_cnt = torch.zeros((height, width), dtype=torch.float32, device=dev)
        cov_cnt = torch.zeros_like(sat_cnt)
        kernel_list_hr = {}
        psf_stars = {}
        for idx, d, sat in frames(list(active)):                 # pass B
            check()
            m, d_norm = self._mask_and_norm(d, norm_factor, sat_norm, sat, idx)
            # the PSF from the normalised frame, saturated pixels excluded
            d_np = (d_norm[0].cpu().numpy() * norm_factor).copy()
            d_np[~np.isfinite(d[0].cpu().numpy())] = np.nan
            d_np[sat[0].cpu().numpy()] = np.nan
            coords = [(cx, cy) for cx, cy in catalog] if reference_psf and len(catalog) >= 5 else store.stars[idx]
            # PIPELINE (change 7): one kernel per channel when per_channel_psf (the script measures green and repeats
            # it); a debayered OSC frame's R and B, sampled at 1/4 density, have wider PSFs than G
            ks = []
            # (green first: a channel with too few stars of its own uses green's measured stack)
            order = ([1] + [c for c in range(channels) if c != 1]) if (per_channel_psf and channels > 1) else (range(channels) if per_channel_psf else [None])
            stacked, n_green = {}, 0
            for c in order:
                plane = d_np if c is None else d_np[c]
                target_psf, n_c = self.psf_solver.extract_and_stack_stars(plane, coords)
                cm = 1 if c is None and channels > 1 else (0 if c is None else c)
                if cm in mean_psf and n_c > 0:
                    target_psf = self._join_wings(target_psf, mean_psf[cm][0], float(self.psf_solver.last_radius), mean_psf[cm][1])
                if c == 1 or c is None or channels == 1:
                    n_green = n_c
                elif n_c < MIN_PSF_STARS and n_green >= MIN_PSF_STARS:
                    target_psf, n_c = stacked[1], n_green
                stacked[c] = target_psf
            n_used = n_green
            # (the channels' fits are independent problems on the same operator: one batched solve)
            chans = list(range(channels)) if per_channel_psf else [None]
            for c in chans:
                ca = (1 if channels > 1 else 0) if c is None else c
                if ca in wings:
                    stacked[c] = self._apply_radial_wing(stacked[c], wings[ca][0])
                if ca in asym:
                    stacked[c] = self._add_nonradial(stacked[c], asym[ca][0])
            solved_all = self.psf_solver.solve(torch.stack([stacked[c] for c in chans]), sigma=latent_sigma)
            for j_, c in enumerate(chans):
                ca_w = (1 if channels > 1 else 0) if c is None else c
                solved_k = solved_all[:, j_:j_ + 1]
                hc = halos[c if c is not None else (1 if channels > 1 else 0)]
                if hc is not None:
                    solved_k = self._attach_halo(solved_k, hc[0])
                k_hr = self.psf_solver.solve_sr(solved_k, r, sigma=psf_sigma) if r > 1 else solved_k
                if ca_w in wings and r == 1:
                    k_hr = self._attach_far_wing(k_hr[0, 0], wings[ca_w][0])[None, None]
                    if ca_w in far_asym:
                        k_hr = self._modulate(k_hr[0, 0], far_asym[ca_w][0])[None, None]
                ks.append(k_hr[0, 0])
            psf_stars[idx] = n_used
            if n_used < MIN_PSF_STARS and (reference_psf and len(catalog) >= 5):
                dropped[idx] = f"PSF: {n_used} stars"
                active.remove(idx)
                del m, d_norm, d
                continue
            kernel_list_hr[idx] = torch.stack(ks).contiguous()           # (C or 1, k, k): broadcasts over channels
            have = torch.isfinite(d).to(d_norm.dtype)
            all_accum += d_norm * have
            all_sum += have
            m_sum += m
            sat_cnt += sat[0].any(0).to(torch.float32)
            cov_cnt += have[0, 1 if channels > 1 else 0]
            x_accum += d_norm * m                      # PIPELINE (change 6): mean over the valid pixels
            del m, d_norm, d
            say(idx + 1, n_imgs, f"PSF {idx + 1}/{n_imgs} ({n_used} stars)")
        if not active:
            raise RuntimeError("no frame passed the selection (transmission, PSF stars)")
        var_floors = [v / (s * s) for v, s in zip(var_floors, scale_t)]
        self.last_run_info['psf_stars'] = [psf_stars.get(i, 0) for i in range(n_imgs)]
        self.last_run_info['frames_used'] = list(active)
        self.last_run_info['frames_dropped'] = {int(k): v for k, v in sorted(dropped.items())}
        self.last_run_info['var_floor_median'] = float(np.median(var_floors)) if var_floors else None
        self.last_run_info['var_floors'] = [float(v) for v in var_floors]
        self.last_run_info['kernels'] = [kernel_list_hr[i].cpu().numpy() if i in kernel_list_hr else None for i in range(n_imgs)]
        self.last_run_info['flux_scale_p5_p50_p95'] = [round(float(v), 4) for v in np.percentile(scale_t, [5, 50, 95])]

        # 3. Initialize Latent Image x (HR)
        # (pixels without a valid sample in any frame - the core of a star saturated all night - start from the frames'
        # clipped values, the script's initial guess: a multiplicative update never grows a pixel that starts at 0)
        x_lr = torch.where(m_sum > 0, x_accum / torch.clamp(m_sum, min=1.0), all_accum / torch.clamp(all_sum, min=1.0))
        mean_norm = torch.where(all_sum > 0, all_accum / torch.clamp(all_sum, min=1.0), torch.full_like(all_sum, float('nan')))[0].cpu().numpy() if saturated_photometry else None
        del all_accum, all_sum
        del x_accum
        x = F.interpolate(x_lr, scale_factor=r, mode='nearest') if r > 1 else x_lr
        if point_init and r == 1:
            core_i = ((sat_cnt >= 0.5 * torch.clamp(cov_cnt, min=1.0)) & (cov_cnt > 0)).cpu().numpy()
            if core_i.any():
                hp_ = x_lr - F.avg_pool2d(F.pad(x_lr, (1, 1, 1, 1), mode='replicate'), 3, stride=1)
                nz = [float(1.4826 * torch.median(torch.abs(hp_[0, c] - torch.median(hp_[0, c])))) / math.sqrt(8.0 / 9.0) for c in range(channels)]
                km_i = torch.stack([kernel_list_hr[i] for i in active]).mean(0).cpu().numpy()
                x, pinfo = self._point_initial(x, core_i, latent_sigma, nz, valid=(m_sum[0] > 0).cpu().numpy(), kmean=km_i,
                                               valid_frac=(m_sum[0] / float(max(1, len(active)))).cpu().numpy())
                self.last_run_info['point_initial'] = pinfo
        # PIPELINE (change 11, lower_bound): the Lower Bound Constraint Algorithm (Lanteri, Roche & Aime 2002; Roche
        # et al. 2003, A&A 409, 1077): x = b + u, u >= 0, the multiplicative update applied to u.  Richardson-Lucy type
        # updates dig a moat round every star on a sky background (IC 1805: -1.4 % of the restored peak at 5-10 px,
        # where the defocused PSF's transfer function has a notch) because x >= 0 lies far below the sky.  After
        # seqsubsky each channel's sky is a flat pedestal, so b is one level per channel: the darkest sky (2nd
        # percentile of the initial image smoothed over 9 x 9 px) less 3 x the initial image's pixel noise (robust
        # scatter of its 3 x 3 high-pass), so that noise does not touch it (at the darkest sky itself 10-20 % of the
        # pixels sat on the bound - a clipped floor) and a moat is at most ~3 sigma deep.
        lb = torch.zeros((1, channels, 1, 1), dtype=x.dtype, device=dev)
        if lower_bound:
            hp = x_lr - F.avg_pool2d(F.pad(x_lr, (1, 1, 1, 1), mode='replicate'), 3, stride=1)
            gen = torch.Generator(device="cpu").manual_seed(0)
            sig = []
            for c in range(channels):
                e = hp[0, c][m_sum[0, c] > 0]
                if not e.numel():
                    sig.append(0.0)
                    continue
                e = e[torch.randperm(e.numel(), generator=gen)[:1_000_000].to(dev)]
                # (the 3 x 3 high-pass keeps 8/9 of a white noise's variance: sigma = 1.4826 MAD / sqrt(8/9))
                sig.append(float(1.4826 * torch.median(torch.abs(e - torch.median(e))) / math.sqrt(8.0 / 9.0)))
            if str(lower_bound) == "local":
                # (the latent's noise: pixels whose initial image has little structure - a high-pass of the whole tile
                # reads M 42's nebula detail as noise and dropped the bound far below the nebula)
                g2 = F.avg_pool2d(F.pad(x_lr, (2, 2, 2, 2), mode='replicate'), 5, stride=1)
                gx = torch.abs(g2[..., :, 2:] - g2[..., :, :-2])[..., 1:-1, :]
                gy = torch.abs(g2[..., 2:, :] - g2[..., :-2, :])[..., :, 1:-1]
                grad = F.pad(gx + gy, (1, 1, 1, 1), mode='replicate')
                calm = torch.zeros_like(grad, dtype=torch.bool)
                for c in range(channels):
                    g_c = grad[0, c].flatten()
                    thr = torch.quantile(g_c[torch.randperm(g_c.numel(), device=dev)[:500_000]], 0.3)
                    calm[0, c] = grad[0, c] <= thr
                hp0 = x_lr - F.avg_pool2d(F.pad(x_lr, (1, 1, 1, 1), mode='replicate'), 3, stride=1)
                s0 = []
                for c in range(channels):
                    e = hp0[0, c][calm[0, c]].flatten()
                    e = e[torch.randperm(e.numel(), device=dev)[:500_000]]
                    s0.append(float(1.4826 * torch.median(torch.abs(e - torch.median(e))) / math.sqrt(8.0 / 9.0)))
                # the sources: the reference stars and every core saturated in at least half its frames
                from scipy.ndimage import label as _label, center_of_mass
                core0 = ((sat_cnt >= 0.5 * torch.clamp(cov_cnt, min=1.0)) & (cov_cnt > 0)).cpu().numpy()
                pts = [(float(px), float(py)) for px, py in catalog]
                if core0.any():
                    lab0, n0 = _label(core0)
                    pts += [(float(cx_), float(cy_)) for cy_, cx_ in center_of_mass(core0, lab0, range(1, n0 + 1))]
                gch = 1 if channels > 1 else 0
                src = self._source_mask(x_lr[0, gch].detach().cpu().numpy(), pts, s0[gch]) if pts else None
                self.last_run_info['bound_source_fraction'] = round(float(src.mean()), 4) if src is not None else 0.0
                bg_map = self._local_background(x_lr, m_sum > 0, src)
                if r > 1:
                    bg_map = F.interpolate(bg_map, scale_factor=r, mode='bilinear', align_corners=False)
                sig0 = torch.tensor(s0, device=dev, dtype=x.dtype)[None, :, None, None]
                lb = torch.clamp(bg_map - float(bound_sigma) * sig0, min=0.0)
                if r > 1:
                    calm = F.interpolate(calm.float(), scale_factor=r, mode='nearest') > 0.5
                self.last_run_info['bound_sigma'] = []
                self.last_run_info['lower_bound'] = "local: source-masked 32 px mesh background (5 x 5 minimum of the 1 px Gaussian off the sources) - bound_sigma sigma"
            else:
                sm = F.avg_pool2d(F.pad(x_lr, (4, 4, 4, 4), mode='replicate'), 9, stride=1)
                for c in range(channels):
                    v = sm[0, c][m_sum[0, c] > 0]
                    if v.numel():
                        v = v[torch.randperm(v.numel(), generator=gen)[:1_000_000].to(dev)]
                        lb[0, c] = max(0.0, float(torch.quantile(v, 0.02)) - float(bound_sigma) * sig[c])
                self.last_run_info['lower_bound'] = [float(v) * norm_factor for v in lb.flatten()]

        # Stop Criterion Mask: effective if present in > mu fraction of frames
        m_eff_lr = (m_sum / float(max(len(active), 1)) > float(mu)).to(x.dtype)
        vfrac = (m_sum[0] / float(max(1, len(active)))).cpu().numpy()
        self._vfrac = vfrac
        self.last_run_info['effective_fraction'] = float(m_eff_lr.mean().item())
        del m_sum
        m_eff_hr = self.upsample_adjoint(m_eff_lr, r)
        M_eff_hr_count = torch.clamp(torch.sum(m_eff_hr), min=1.0)

        k_size = int(kernel_list_hr[active[0]].shape[-1])
        conv = FFTConv(height * r, width * r, k_size, 'constant' if self.formal_mode else 'reflect', dev)
        u_prev = None
        total_steps = iterations * len(active)
        chi2_t: dict = {}
        # pixels saturated in at least half of the frames that cover them: the cores of saturated stars
        sat_core = ((sat_cnt >= 0.5 * torch.clamp(cov_cnt, min=1.0)) & (cov_cnt > 0)).cpu().numpy()
        self._sat_core = sat_core
        self.last_run_info['bloom_grow'] = int(bloom_grow)
        del sat_cnt, cov_cnt
        current_step = 0
        EPSILON = 1e-8 if self.formal_mode else 1e-5
        kappa_f = float(max(kappa, 1.000001))

        log(f"Starting Optimization (SR x{r})...")
        for k in range(iterations):
            x_hat = conv.rfft(x)
            num_hat = torch.zeros_like(x_hat[0])
            den_hat = torch.zeros_like(x_hat[0])

            for i, d, sat in frames(active):
                check()
                m_batch, y_batch = self._mask_and_norm(d, norm_factor, sat_norm, sat, i)
                k_hat = conv.kernel_hat(kernel_list_hr[i])

                # --- FORWARD MODEL ---
                prediction_lr = self.downsample(conv.convolve(x_hat, k_hat), r)
                # PIPELINE (change 12): a saturated pixel is censored data (Whyte, Sivic & Zisserman 2014, IJCV
                # 110:205): the sky there is at least the clipped value.  It takes part while the model is below it
                # (pulling the core up) and carries no information above it.  With the script's plain mask a star
                # saturated in every frame has no data in its core and the update drained it to the sky.
                if censor_saturated:
                    cens = sat & torch.isfinite(d) & (prediction_lr < y_batch)
                    m_batch = torch.maximum(m_batch, cens.to(m_batch.dtype))
                v_src = torch.clamp(prediction_lr, min=0.0) if variance_from == "model" else y_batch
                if ptc is not None:
                    # PIPELINE (change 14, noise_model "ptc"): the variance of the measured photon transfer, sky noise
                    # of the frame + gain x the signal above the sky
                    b_ = torch.tensor(ptc["slope"], dtype=v_src.dtype, device=v_src.device)[None, :, None, None]
                    sky_ = torch.tensor(self._norm[2], dtype=v_src.dtype, device=v_src.device)[None, :, None, None]
                    v_batch = torch.clamp(var_floors[i] + b_ * torch.clamp(v_src - sky_, min=0.0) / float(self._norm[1][i]), min=1e-12)
                else:
                    v_batch = self._variance(v_src, var_floors[i], shot_noise_scale, read_noise)

                # --- RESIDUALS ---
                if ptc is not None and huber_sigma:
                    # the paper's Huber on the standardised residual (r / sqrt(v), delta in sigma): the script's absolute
                    # delta 0.05 treated every bright star core as an outlier and left it unfitted (15-25 % of the peak)
                    psi_lr = self.huber_weight((y_batch - prediction_lr) / torch.sqrt(v_batch), delta=float(huber_sigma))
                else:
                    psi_lr = self.huber_weight(y_batch - prediction_lr, delta=huber_delta)
                W_lr = (m_batch / v_batch) * psi_lr
                if k == FRAME_REJECT_ITER - 1:
                    # the frame's fit: median standardised residual^2 over its valid green pixels (chi2(1) median 0.455)
                    zc = ((y_batch - prediction_lr) / torch.sqrt(v_batch))[0, 1 if channels > 1 else 0]
                    zc = zc[m_batch[0, 1 if channels > 1 else 0] > 0].flatten()
                    zc = zc[torch.randperm(zc.numel(), device=zc.device)[:200_000]]
                    chi2_t[i] = float(torch.median(zc * zc)) / 0.455

                # --- BACKWARD (Adjoint) ---
                # Term 1: F^T D^T (W * y);  Term 2: F^T D^T (W * D F x)
                terms = torch.cat([self.upsample_adjoint(W_lr * y_batch, r), self.upsample_adjoint(W_lr * prediction_lr, r)], 0)
                spec = conv.correlate_hat(terms, k_hat)
                num_hat += spec[0]
                den_hat += spec[1]
                del spec, terms, W_lr, psi_lr, prediction_lr, m_batch, y_batch, v_batch, k_hat, d, sat

                current_step += 1
                say(current_step, total_steps, f"Iter {k + 1}/{iterations} - Img {i + 1}/{n_imgs}")

            numerator = conv.back(num_hat)[None]
            denominator = conv.back(den_hat)[None]
            del num_hat, den_hat, x_hat
            if k == FRAME_REJECT_ITER - 1 and chi2_t:
                # PIPELINE (change 22): frames the model cannot fit (patchy cloud, a sky changing across the frame:
                # M 31 2026-10-21 frames at chi2 30-65 among frames at 0.5-1) leave the restoration from here on - the
                # robust outlier rule on the frames' chi2: > median + 6 x 1.4826 MAD and > 2
                cv_ = np.array(list(chi2_t.values()))
                lim = max(2.0, float(np.median(cv_) + 6 * 1.4826 * np.median(np.abs(cv_ - np.median(cv_)))))
                for i_, c_ in chi2_t.items():
                    if c_ > lim and len(active) > 1:
                        active.remove(i_)
                        dropped[i_] = f"fit: chi2 {c_:.1f} (limit {lim:.1f})"
                self.last_run_info['frames_used'] = list(active)
                self.last_run_info['frames_dropped'] = {int(a): b for a, b in sorted(dropped.items())}
                self.last_run_info['frame_chi2_median'] = round(float(np.median(cv_)), 3)

            # --- UPDATE STEP ---
            denominator = torch.clamp(denominator, min=EPSILON)
            ratio = numerator / denominator
            ratio = torch.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)
            ratio = torch.clamp(ratio, min=1.0 / kappa_f, max=kappa_f)
            del numerator, denominator

            # Stop Criterion (Appendix C)
            if u_prev is not None:
                u_ratio = ratio / torch.clamp(u_prev, min=EPSILON)
                u_ratio = torch.nan_to_num(u_ratio, nan=1.0, posinf=1.0, neginf=1.0)
                diff = torch.abs(u_ratio - 1.0) * m_eff_hr
                md_val = float((torch.sum(diff) / M_eff_hr_count).item())
                self.last_run_info['mean_diff_history'].append(md_val)
                self.last_run_info['mean_diff'] = md_val
                del u_ratio, diff
                if md_val < float(stop_epsilon):
                    self.last_run_info.update({'converged': True, 'iters_done': k + 1})
                    break
            u_prev = ratio

            # (the margin of the local bound follows the latent's own pixel noise, which the deconvolution raises above
            # the input's: 3 x the input's sigma clipped the restored sky into a flat floor - IC 1805 crops)
            if str(lower_bound) == "local" and k % 5 == 4:
                hp_x = x - F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), 3, stride=1)
                sig_k = []
                for c in range(channels):
                    e = hp_x[0, c][calm[0, c]].flatten()
                    e = e[torch.randperm(e.numel(), device=dev)[:500_000]]
                    sig_k.append(float(1.4826 * torch.median(torch.abs(e - torch.median(e))) / math.sqrt(8.0 / 9.0)))
                sig_now = torch.maximum(sig0, torch.tensor(sig_k, device=dev, dtype=x.dtype)[None, :, None, None])
                self.last_run_info['bound_sigma'].append([round(float(v) * norm_factor, 8) for v in sig_now.flatten()])
                lb = torch.clamp(bg_map - float(bound_sigma) * sig_now, min=0.0)
            # Apply
            x = lb + torch.clamp(x - lb, min=0.0) * ratio
            x = torch.nan_to_num(x, nan=EPSILON)
            x = torch.maximum(x, lb)
            if k == FRAME_REJECT_ITER - 1 and saturated_points:
                x, sinfo = self._saturated_point_sources(x, sat_core, lb, latent_sigma, r)
                self.last_run_info['saturated_cores'] = sinfo
            if dev.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

        pool.shutdown(wait=False)
        if not self.last_run_info['converged']:
            self.last_run_info['iters_done'] = iterations

        # (the restoration as fitted, before the saturated cores are rewritten: what the wing calibration measures)
        self._x_fitted = (x[0].detach().cpu().numpy() * norm_factor).astype(np.float32)
        if saturated_photometry and sat_core.any() and r == 1:
            kmean = torch.stack([kernel_list_hr[i] for i in active]).mean(0).cpu().numpy()
            psi = self._restored_psf(x[0].detach().cpu().numpy().astype(np.float64), catalog, sat_core) if (catalog and r == 1) else None
            self.last_run_info['restored_psf_stars'] = None if psi is None else True
            x, sinfo = self._saturated_star_photometry(x, sat_core, mean_norm, kmean, grow=int(bloom_grow) + 1 if not censor_saturated else 2, ann=(1, 5) if not censor_saturated else (1 + int(bloom_grow), 5 + int(bloom_grow)), sigma=latent_sigma, valid_frac=vfrac, inner_px=0.0 if not censor_saturated else 2.0, blend_px=2.0 if not censor_saturated else SAT_BLEND, psi=psi, catalog=catalog)
            self.last_run_info['saturated_photometry'] = sinfo
        self._lb_used = torch.broadcast_to(lb, x.shape)[0].cpu().numpy() if lower_bound else None
        res_np = (x[0].cpu().numpy() * norm_factor).astype(np.float32)
        return res_np
