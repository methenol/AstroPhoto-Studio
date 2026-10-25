"""Optional ImageMM restoration stage (``deconv_method = "imagemm"``): VeraLux ImageMM on the session's subs.

ImageMM (Sukurdeep, Budavari, Connolly & Navarro 2025, AJ 170:233) restores the *individual* exposures jointly:
``y_t = D(h_t * x) + noise``, one PSF, variance and mask per exposure.  The engine is VeraLux ImageMM v1.0.0 by
Riccardo Paterniti (``sirilpipe/veralux_imagemm.py``, the script's maths without its Qt window; what changed and why
is listed there).  This module does what the script's two tabs did by hand, headless, on the frames the stack used:

1. Pre-processing (the script's tab 1: ``calibrate -debayer``, ``seqsubsky <seq> 1``, ``register``).  The calibrated
   sequence of the stack (``process/state.json``; already dark / flat / banding corrected, still CFA for the Bayer
   drizzle) is debayered by ``calibrate <seq> -debayer -prefix=imm_`` (the script's no-master branch), background-
   subtracted per frame by ``seqsubsky imm_<seq> 1`` and registered by ``seqapplyreg -interp=lanczos4
   -framing=max`` with the stack's *own* transforms (copied from ``<seq>_.seq``), so the frames land on stack.fits'
   canvas.  The script registers with interpolation too; the stack's Bayer drizzle would leave 50-75 % of the R / B
   pixels of an unshifted frame empty, which ``seqsubsky`` then fills (its samples see the zeros: the reference
   frame of IC 1805 kept its sky while the others were levelled).
2. Saturation (VeraLux's ``saturation_level``, which it reads from BITPIX - None for Siril's float frames): the
   sensor ceiling is the highest raw value found as a plateau (>= 4 pixels at the maximum) over the frames' raw
   files (frames.json ``path``); pixels at >= 0.995 x the ceiling (``SATURATION_FRAC``) that are still >= half the
   ceiling in Siril's calibrated frame are saturated (the others are hot pixels, ~90 % on DWARF 3), grown by the
   reach of the debayer and Lanczos-4 kernels (``SAT_GROW`` px) and registered by Siril with the frames' own
   transforms (``-interp=nearest``).  DWARF 3: a 12-bit sensor (ceiling 4095 in a 16-bit file).
3. Star lists (the script's per-frame ``findstar`` loop): ``seqfindstar`` on the registered frames, default
   detection settings (``setfindstar reset``), positions converted to array pixels (``siril_to_array``).
4. The frames, cropped to the stack's box ``stack_meta['crop']`` in stack.fits' row order, go to a disk store with
   NaN where a frame has no data and a flag on its saturated pixels; ``ImageMMEngine.run_imagemm`` restores them
   with the script's defaults (20 iterations, kappa 2, Huber delta 0.05, mu 0.1, epsilon 1e-5, practical mode) and
   the pipeline's fixes (``veralux_imagemm`` docstring).

Output: ``imagemm.fits`` (float32 (3, H, W): the crop, row order and WCS of stack.fits, rescaled for r > 1) and
``imagemm_meta.json``.  torch is imported inside ``available()`` / ``run()`` only.
"""
from __future__ import annotations

import glob
import logging
import os
import re
import shutil
import time

import cv2
import numpy as np

from . import fitsio
from .util import clean_json, read_json, write_json

log = logging.getLogger("sirilpipe")
STAGE = "imagemm"
OUT_FITS = "imagemm.fits"
OUT_META = "imagemm_meta.json"
WORK = os.path.join("process", "imagemm")          # the frame store (deleted afterwards)
PREP_VERSION = 8                                     # bump when the store's content changes (imagemm_keep_prepared)

SATURATION_FRAC = 0.995      # VeraLux build_mask_and_variance(saturation_frac=0.995)
SAT_PLATEAU = 4              # raw pixels at the frame's maximum that make it a saturation ceiling
# a registered pixel is touched by a saturated raw pixel within the debayer's reach (RCD, Siril's default: 3 px) plus
# the Lanczos-4 kernel's (4 px)
SAT_GROW = 7
SAT_GROW_DRIZZLE = 2         # a square drizzle drop (pixfrac <= 1) reaches the output pixels within 1 px
EDGE_ERODE = 6               # px of each registered frame's footprint edge dropped (Lanczos-4 reach + debayer)
EDGE_ERODE_DRIZZLE = 2       # drizzle: an edge pixel's partial drops are normalised by their weight
INTERP = "lanczos4"          # Siril's default registration interpolation (the script's `register`)

# stack_params keys (webui/server.py STACK_DEFAULTS) -> VeraLux run_imagemm arguments
PARAM_KEYS = {"imagemm_iterations": "iterations", "imagemm_huber": "huber_delta", "imagemm_kappa": "kappa",
              "imagemm_mu": "mu", "imagemm_epsilon": "stop_epsilon", "imagemm_sigma": "psf_sigma",
              "imagemm_shot_noise": "shot_noise_scale", "imagemm_read_noise": "read_noise",
              "imagemm_saturation": "saturation_level", "imagemm_formal": "formal_mode",
              "imagemm_variance": "variance_from", "imagemm_psf_channels": "per_channel_psf",
              "imagemm_latent_sigma": "latent_sigma", "imagemm_lower_bound": "lower_bound",
              "imagemm_reference_psf": "reference_psf", "imagemm_photometry": "photometry",
              "imagemm_noise_model": "noise_model", "imagemm_huber_sigma": "huber_sigma", "imagemm_halo": "halo",
              "imagemm_saturated_points": "saturated_points", "imagemm_bound_sigma": "bound_sigma",
              "imagemm_saturated_photometry": "saturated_photometry", "imagemm_psf_wings": "psf_wings",
              "imagemm_psf_asymmetry": "psf_asymmetry", "imagemm_psf_radial_wings": "psf_radial_wings",
              "imagemm_bloom_grow": "bloom_grow", "imagemm_point_init": "point_init",
              "imagemm_censor_saturated": "censor_saturated"}
LIMITS = {"iterations": (1, 200), "kappa": (1.01, 50.0), "huber_delta": (0.001, 1.0), "mu": (0.0, 1.0),
          "stop_epsilon": (1e-8, 1e-3), "psf_sigma": (0.5, 3.0), "shot_noise_scale": (0.0, 10.0),
          "read_noise": (0.0, 1.0)}                  # the script's spin-box ranges


def available() -> dict:
    """``{"ok", "torch", "cuda", "mps", "reason"}`` - whether the stage can run here.  ``ok`` needs torch; a GPU is
    optional (the CPU is slow but exact)."""
    out = {"ok": False, "torch": False, "cuda": False, "mps": False, "reason": ""}
    try:
        import torch                                   # noqa: WPS433 (optional, large)
    except Exception as e:                             # ImportError, or a broken CUDA install
        out["reason"] = f"torch is not installed ({type(e).__name__}); build the image with IMAGEMM=cuda|cpu"
        return out
    out["torch"] = True
    out["version"] = str(torch.__version__)
    try:
        out["cuda"] = bool(torch.cuda.is_available())
        out["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        pass
    out["ok"] = True
    return out


# --------------------------------------------------------------------------- parameters
def _flag(v) -> bool:
    return str(v).strip().lower() not in ("false", "0", "off", "no", "none", "") and bool(v)


def map_params(stack_params: dict | None) -> dict:
    """VeraLux ``run_imagemm`` keyword arguments (+ ``formal_mode``) from ``stack_params``, the script's GUI defaults
    where unset, clamped to its spin-box ranges.  ``imagemm_r`` is the super-resolution factor (1..3; the script
    allows 4); ``imagemm_sigma`` 0 = the script's 1.0; ``imagemm_saturation`` 0 = measured on the raw frames.
    Development options (not in the UI): ``imagemm_max_frames`` (the N best frames by weight), ``imagemm_region``
    ([x, y, w, h] inside the stack's crop box) and ``imagemm_keep_prepared`` (keep the frame store for the next run)."""
    from .veralux_imagemm import DEFAULTS
    sp = stack_params or {}
    kw = {k: v for k, v in DEFAULTS.items() if k != "upscale_factor"}
    for key, arg in PARAM_KEYS.items():
        if sp.get(key) not in (None, ""):
            kw[arg] = sp[key]
    kw["iterations"] = int(round(float(kw["iterations"])))
    for k in ("kappa", "huber_delta", "mu", "stop_epsilon", "psf_sigma", "shot_noise_scale", "read_noise"):
        kw[k] = float(kw[k])
    if kw["psf_sigma"] <= 0:
        kw["psf_sigma"] = DEFAULTS["psf_sigma"]
    for k, (lo, hi) in LIMITS.items():
        kw[k] = type(kw[k])(min(hi, max(lo, kw[k])))
    sat = float(kw["saturation_level"] or 0.0)
    kw["saturation_level"] = sat if sat > 0 else None
    kw["formal_mode"] = _flag(kw["formal_mode"])
    kw["latent_sigma"] = min(2.0, max(0.3, float(kw["latent_sigma"] or DEFAULTS["latent_sigma"])))
    _lb = str(kw["lower_bound"]).strip().lower()
    kw["lower_bound"] = _lb if _lb in ("local", "global") else ("global" if _flag(kw["lower_bound"]) else False)
    kw["per_channel_psf"] = _flag(kw["per_channel_psf"])
    kw["reference_psf"] = _flag(kw["reference_psf"])
    kw["photometry"] = _flag(kw["photometry"])
    kw["noise_model"] = "ptc" if str(kw["noise_model"]) == "ptc" else "script"
    kw["huber_sigma"] = float(kw["huber_sigma"] or 0.0)
    kw["halo"] = _flag(kw["halo"])
    kw["saturated_points"] = _flag(kw["saturated_points"])
    kw["saturated_photometry"] = _flag(kw["saturated_photometry"])
    kw["psf_wings"] = _flag(kw["psf_wings"])
    kw["psf_asymmetry"] = _flag(kw["psf_asymmetry"])
    kw["psf_radial_wings"] = _flag(kw["psf_radial_wings"])
    kw["bound_sigma"] = max(0.0, float(kw["bound_sigma"]))
    kw["bloom_grow"] = int(min(8, max(0, round(float(kw["bloom_grow"] or 0)))))
    kw["point_init"] = _flag(kw["point_init"])
    kw["censor_saturated"] = _flag(kw["censor_saturated"])
    kw["variance_from"] = "data" if str(kw["variance_from"]) == "data" else "model"
    kw["r"] = int(min(3, max(1, round(float(sp.get("imagemm_r") or 1)))))
    return kw


def pick_device(preference: str | None = "auto"):
    """cuda / mps / cpu as asked ("auto" = the GPU when there is one)."""
    import torch
    pref = str(preference or "auto").lower()
    if pref.startswith("cuda") and torch.cuda.is_available():
        dev = torch.device(pref)
        return dev if dev.index is None or dev.index < torch.cuda.device_count() else torch.device("cuda:0")
    if pref == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- sequences
def included_rows(session_dir: str) -> list[dict]:
    rows = read_json(os.path.join(session_dir, "frames.json"), None)
    if not rows:
        raise RuntimeError("frames.json is missing: analyse and stack the dataset first")
    return rows


def read_seq(path: str) -> dict:
    """A Siril ``.seq`` file: {"lines", "S" (fields), "nums" (file number per image), "R" ({layer: [line per image]})}."""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    out = {"lines": lines, "S": None, "nums": [], "R": {}}
    for ln in lines:
        if ln.startswith("S "):
            out["S"] = ln.split()
        elif ln.startswith("I "):
            out["nums"].append(int(ln.split()[1]))
        elif re.match(r"R\d* ", ln):
            out["R"].setdefault(int(ln.split()[0][1:] or 0), []).append(ln)
    return out


def registration_of(seq: dict) -> tuple[dict[int, dict[int, str]], int | None]:
    """({layer: {file number: R line}}, the reference frame's file number) of a registered sequence."""
    by = {layer: dict(zip(seq["nums"], rl)) for layer, rl in seq["R"].items() if len(rl) == len(seq["nums"])}
    if not by:
        raise RuntimeError("the calibrated sequence has no registration data: stack again")
    ref = None
    try:
        ref = seq["nums"][int(seq["S"][6])]
    except (TypeError, IndexError, ValueError):
        pass
    return by, ref


def write_registration(path: str, nums: list[int], reg: dict[int, dict[int, str]], ref: int | None, layers: int,
                       name: str | None = None, mono_layer: bool = False) -> None:
    """Give the sequence ``path`` (its frames numbered ``nums``) the stack's transforms: Siril's ``calibrate`` /
    ``seqsubsky`` keep the file numbers but not the registration, so ``seqapplyreg`` of the new frames needs the R
    lines of the calibrated sequence, frame by frame.  Writes the file anew when ``name`` is given (a sequence made
    here: the saturation masks).  ``mono_layer``: the registration on layer 0 (a 1-layer sequence)."""
    ref_idx = nums.index(ref) if ref in nums else 0
    if name is None:
        old = read_seq(path)
        head = []
        for ln in old["lines"]:
            if re.match(r"R\d* ", ln):
                continue
            if ln.startswith("S "):
                p = ln.split()
                p[6] = str(ref_idx)
                ln = " ".join(p)
            elif ln.startswith("L "):
                ln = f"L {layers}"
            head.append(ln)
        if old["nums"] != nums:
            raise RuntimeError(f"{os.path.basename(path)}: frames {old['nums'][:3]}... are not the included frames {nums[:3]}...")
    else:
        head = ["#Siril sequence file. Contains list of images, selection, registration data and statistics",
                "#S 'sequence_name' start_index nb_images nb_selected fixed_len reference_image version variable_size fz_flag drizzle",
                f"S '{name}_' {nums[0]} {len(nums)} {len(nums)} 5 {ref_idx} 6 0 0 0", f"L {layers}"] + [f"I {n} 1" for n in nums]
    body = []
    for layer, lines in sorted(reg.items()):
        tag = "R0" if mono_layer else f"R{layer}"
        for n in nums:
            body.append(re.sub(r"^R\d*", tag, lines[n]))
        if mono_layer:
            break
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(head + body) + "\n")


def sequence_placements(seq_file: str, canvas: tuple[int, int]) -> dict[int, tuple[int, int, int, int]]:
    """Where each frame of a Siril registered sequence (``r_<seq>_.seq``) sits on the stack's canvas:
    {file number: (ox, oy, w, h)}, the frame's pixel (row, col) being the canvas pixel (oy + row, ox + col).

    ``seqapplyreg -framing=max`` writes every frame at its own size (a variable-size sequence) and records the
    translation of each in the registration matrix H (``R`` lines: [1 0 tx; 0 1 ty; 0 0 1], Siril's y up); the
    stack's ``-maximize`` then places the frames on the union canvas.  Measured against stack_raw.fit (phase
    correlation, 8 frames of M 42, score 0.76-0.96): ox = tx - min(tx) and oy = canvas_h - h - (ty - min(ty))
    (the array's row 0 is the bottom row)."""
    sizes = []
    layers: dict[int, list[list[float]]] = {}
    with open(seq_file, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("I "):
                t = ln.split()
                w, h = t[3].split(",")
                sizes.append((int(t[1]), int(w), int(h)))
            elif ln.startswith("R") and " H " in ln:
                t = ln.split()
                i = t.index("H")
                layers.setdefault(int(t[0][1:] or 0), []).append([float(v) for v in t[i + 1:i + 10]])
    n = len(sizes)
    hs = next((v for v in layers.values() if len(v) == n), None)
    cw, ch = canvas
    if hs is None:
        if all((w, h) == (cw, ch) for _, w, h in sizes):
            return {num: (0, 0, w, h) for num, w, h in sizes}
        raise RuntimeError(f"{os.path.basename(seq_file)}: no registration data to place the frames of different sizes")
    tx = np.array([m[2] for m in hs])
    ty = np.array([m[5] for m in hs])
    out = {}
    for (num, w, h), x, y in zip(sizes, tx, ty):
        out[num] = (int(round(x - tx.min())), int(round(ch - h - (y - ty.min()))), w, h)
    return out


# --------------------------------------------------------------------------- frames
def read_frame(path: str, box: tuple[int, int, int, int], place: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """The box (x, y, w, h) of the stack's canvas, as seen by a registered frame, float32 (h, w, 3): the frame
    sits on the canvas at ``place`` = (ox, oy, w, h) (``sequence_placements``); canvas pixels it does not cover are
    0 (no data).  Only the rows needed are read."""
    from astropy.io import fits
    x, y, w, h = box
    out = np.zeros((h, w, 3), np.float32)
    ox, oy, fw, fh = place if place is not None else (0, 0, w + x, h + y)
    x0, x1 = max(x, ox), min(x + w, ox + fw)
    y0, y1 = max(y, oy), min(y + h, oy + fh)
    if x1 <= x0 or y1 <= y0:
        return out
    with fits.open(path, memmap=True) as hdul:
        d = hdul[0].data
        if d.ndim == 2:
            d = d[None]
        if d.shape[1] != fh or d.shape[2] != fw:
            raise RuntimeError(f"{os.path.basename(path)}: {d.shape[2]} x {d.shape[1]}, the sequence says {fw} x {fh}")
        a = np.array(d[:, y0 - oy:y1 - oy, x0 - ox:x1 - ox], dtype=np.float32)
        if a.shape[0] == 1:
            a = np.repeat(a, 3, 0)
    out[y0 - y:y1 - y, x0 - x:x1 - x] = np.moveaxis(a, 0, -1)
    return out


def stack_is_flipped(ref: np.ndarray, frame: np.ndarray) -> tuple[bool, dict]:
    """Whether stack.fits has its rows in the opposite order to the canvas the registered frames sit on.

    Siril's plate solve rewrites stack.fits and the rewritten file reads back vertically reversed against the
    cropped ``stack_raw.fit`` (measured on M 42: identical to the crop flipped), with no ROWORDER card to say so.
    The frames are put in the stack's order, whichever it is: the normalised correlation of a frame (g channel,
    smoothed, 4x down-sampled, over its pixels with data) with the reference decides.  ``ref``/``frame``:
    (h, w, 3)."""
    def small(a):
        g = cv2.GaussianBlur(np.ascontiguousarray(a[..., 1], np.float32), (0, 0), 2.0)
        return cv2.resize(g, (max(8, g.shape[1] // 4), max(8, g.shape[0] // 4)), interpolation=cv2.INTER_AREA)
    have = cv2.resize((frame[..., 1] != 0).astype(np.float32), (max(8, frame.shape[1] // 4), max(8, frame.shape[0] // 4)),
                      interpolation=cv2.INTER_AREA) > 0.99

    def corr(a, b, m):
        if m.sum() < 100:
            return -1.0
        a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
        d = float(np.sqrt((a * a).sum() * (b * b).sum()))
        return float((a * b).sum() / d) if d > 0 else -1.0
    f, r = small(frame), small(ref)
    straight = corr(f, r, have)
    flipped = corr(f[::-1], r, have[::-1])
    return flipped > straight + 0.05, {"correlation": round(straight, 3), "correlation_flipped": round(flipped, 3)}


def alignment_shift(ref: np.ndarray, frame: np.ndarray) -> list[float]:
    """(dx, dy) of a frame against stack.fits (phase correlation of the smoothed g channels, 4x down-sampled
    and scaled back): the frames must sit on the stack's pixels."""
    have = frame[..., 1] != 0
    a = np.where(have, frame[..., 1], np.median(frame[..., 1][have]) if have.any() else 0).astype(np.float32)
    b = np.ascontiguousarray(ref[..., 1], np.float32)
    a = cv2.GaussianBlur(a, (0, 0), 2.0)
    b = cv2.GaussianBlur(b, (0, 0), 2.0)
    (dx, dy), _ = cv2.phaseCorrelate(b.astype(np.float64), a.astype(np.float64))
    return [round(float(dx), 3), round(float(dy), 3)]


# --------------------------------------------------------------------------- saturation (raw frames)
def raw_saturation(paths: dict[int, str], progress=None, checkpoint=None) -> tuple[float | None, dict[int, np.ndarray], dict]:
    """(sensor ceiling, {file number: (row, col) of the saturated raw pixels}, info) from the frames' raw FITS files.

    The ceiling is the highest frame maximum that is a plateau (>= ``SAT_PLATEAU`` pixels at it: a saturated star
    has many, an unsaturated peak one); a pixel is saturated at >= ``SATURATION_FRAC`` x the ceiling.  No plateau in
    any frame: no saturation (None)."""
    from astropy.io import fits
    cand: dict[int, tuple[np.ndarray, np.ndarray]] = {}      # pixels >= 0.98 x the frame's maximum: (row, col), value
    maxima: list[tuple[float, int]] = []
    unreadable = 0
    n = len(paths)
    for k, (num, p) in enumerate(sorted(paths.items())):
        if checkpoint:
            checkpoint()
        try:
            with fits.open(p, memmap=False) as hdul:
                hdu = hdul[0] if hdul[0].data is not None else next(h for h in hdul if h.data is not None)
                a = np.asarray(hdu.data, dtype=np.float32)
        except Exception:                                            # noqa: BLE001 (a raw camera format: no mask)
            unreadable += 1
            continue
        if a.ndim != 2:
            unreadable += 1
            continue
        mx = float(a.max())
        maxima.append((mx, int((a >= mx).sum())))
        rc = np.argwhere(a >= 0.98 * mx).astype(np.int32)
        cand[num] = (rc, a[rc[:, 0], rc[:, 1]])
        if progress and (k % 10 == 0 or k == n - 1):
            progress(k + 1, n, f"ImageMM: saturation of the raw frames {k + 1}/{n}")
    plateaus = [m for m, c in maxima if c >= SAT_PLATEAU]
    info = {"raw_frames": len(maxima), "unreadable": unreadable}
    if not plateaus:
        return None, {}, {**info, "ceiling": None}
    ceiling = max(plateaus)
    # (a frame whose maximum is below 0.995 x the ceiling has no pixel there; one at the ceiling has them all in cand)
    out = {num: rc[v >= SATURATION_FRAC * ceiling] for num, (rc, v) in cand.items()}
    counts = [len(v) for v in out.values()]
    return ceiling, out, {**info, "ceiling": ceiling, "frames_with_plateau": len(plateaus),
                          "saturated_pixels_median": float(np.median(counts)) if counts else 0.0}


def confirm_on_calibrated(sat: dict[int, np.ndarray], calibrated: dict[int, str], ceiling: float) -> tuple[dict[int, np.ndarray], bool | None, dict]:
    """The raw-saturated pixels that are saturated star light: still >= half the saturated level of their CFA lattice in
    Siril's calibrated frame.

    A raw pixel at the ceiling is also a hot pixel, which the dark subtraction / cosmetic correction removed from the
    calibrated frame (IC 1805: ~600 of the ~700 pixels at 4095 per frame, some in pairs or only in part of the night).
    A saturated star pixel stays at (ceiling - dark) / flat, and the flat differs by colour: one level for all (half the
    ceiling) dropped most saturated BLUE pixels as hot (M 31 2026-10-23: red and blue flats ~3 x green's; blue cores
    then entered the fit as valid data, flat-topped, their light 36-65 % of the stack's).  The level of each 2 x 2
    lattice is the 90th percentile of its candidates' calibrated values over all frames (the saturated star pixels; the
    corrected hot pixels fall far below).  The same test tells whether Siril read the raw file's rows reversed (a
    TOP-DOWN file): returns (pixels, raw rows = the calibrated rows: True / False / None when no star pixel is found
    either way, info)."""
    from astropy.io import fits
    vals = {True: {}, False: {}}
    for num, rc in sat.items():
        p = calibrated.get(num)
        if p is None or not len(rc):
            continue
        with fits.open(p, memmap=True) as hdul:
            d = hdul[0].data
            H = d.shape[-2]
            for straight in (True, False):
                rows = rc[:, 0] if straight else H - 1 - rc[:, 0]
                v = np.asarray(d[..., rows, rc[:, 1]], dtype=np.float32)
                vals[straight][num] = (v.max(0) if v.ndim == 2 else v, rows % 2, rc[:, 1] % 2)
    kept = {True: {}, False: {}}
    levels = {}
    for straight in (True, False):
        lv = {}
        for i in (0, 1):
            for j in (0, 1):
                pool = [v[(ry == i) & (cx == j)] for v, ry, cx in vals[straight].values()]
                pool = np.concatenate(pool) if pool else np.zeros(0, np.float32)
                lv[(i, j)] = float(np.percentile(pool, 90)) if pool.size else np.inf
        # (floors: a tenth of the ceiling in Siril's [0, 1] units - a flat up to 10 x green's - and 0.15 of the brightest
        # lattice's level: a lattice whose candidates are all hot pixels must not set its own level)
        top = max([v for v in lv.values() if np.isfinite(v)], default=np.inf)
        cn = ceiling / 65535.0 if ceiling > 2.0 else ceiling
        lv = {k: max(v, 0.1 * cn, 0.15 * top) for k, v in lv.items()}
        levels[straight] = lv
        for num, rc in sat.items():
            if num not in vals[straight]:
                kept[straight][num] = rc[:0]
                continue
            v, ry, cx = vals[straight][num]
            thr = np.array([lv[(int(a_), int(b_))] for a_, b_ in zip(ry, cx)], np.float32) * 0.5
            kept[straight][num] = rc[v >= thr]
    n_s = sum(len(v) for v in kept[True].values())
    n_f = sum(len(v) for v in kept[False].values())
    n_all = sum(len(v) for v in sat.values())
    info = {"raw_saturated": n_all, "star_pixels": max(n_s, n_f), "hot_pixels": n_all - max(n_s, n_f),
            "kept_straight": n_s, "kept_flipped": n_f}
    if n_s == n_f == 0:
        return {}, None, info
    same = n_s >= n_f
    info["lattice_levels"] = {f"{k[0]}{k[1]}": round(v, 5) for k, v in levels[same].items()}
    return kept[same], same, info


def cfa_layout(calibrated: str, drizzled: str) -> tuple[dict[tuple[int, int], int] | None, dict]:
    """{(row % 2, col % 2): channel} of a CFA frame, measured: the mean of each 2 x 2 sub-lattice of the calibrated CFA
    frame against the means of the drizzled frame's channels (the two green lattices are the diagonal pair that agree
    best; red and blue the remaining two, matched to the drizzled R and B means).  None when it cannot tell."""
    from astropy.io import fits
    cal = np.asarray(fits.getdata(calibrated), dtype=np.float64)
    drz = np.asarray(fits.getdata(drizzled), dtype=np.float64)
    if cal.ndim != 2 or drz.ndim != 3 or drz.shape[0] != 3:
        return None, {"reason": "not a CFA / RGB pair"}
    lat = {(i, j): float(np.nanmean(cal[i::2, j::2])) for i in (0, 1) for j in (0, 1)}
    ch = [float(np.mean(drz[c][drz[c] != 0])) for c in range(3)]
    pairs = [((0, 0), (1, 1)), ((0, 1), (1, 0))]
    rel = [abs(lat[a] - lat[b]) / max(abs(lat[a]) + abs(lat[b]), 1e-12) for a, b in pairs]
    g = pairs[int(np.argmin(rel))]
    o = [p for p in lat if p not in g]
    # R / B: the assignment with the smaller relative mismatch to the drizzled means (relative to green, so that scale
    # differences of the two images cancel)
    gl, gd = 0.5 * (lat[g[0]] + lat[g[1]]), ch[1]
    cost0 = abs(lat[o[0]] / gl - ch[0] / gd) + abs(lat[o[1]] / gl - ch[2] / gd)
    cost1 = abs(lat[o[0]] / gl - ch[2] / gd) + abs(lat[o[1]] / gl - ch[0] / gd)
    out = {g[0]: 1, g[1]: 1}
    if cost0 <= cost1:
        out[o[0]], out[o[1]] = 0, 2
    else:
        out[o[0]], out[o[1]] = 2, 0
    info = {"layout": {f"{k[0]}{k[1]}": "RGB"[v] for k, v in out.items()}, "green_agreement": round(float(min(rel)), 4),
            "rb_margin": round(float(abs(cost0 - cost1)), 4)}
    return out, info


def write_saturation_masks(process: str, sat: dict[int, np.ndarray], shape: tuple[int, int], flip: bool, grow: int = SAT_GROW,
                           layout: dict | None = None) -> None:
    """``immsat_NNNNN.fit``: 255 where a raw pixel within ``grow`` px is saturated (uint8, the CFA geometry); with a CFA
    ``layout`` three planes, plane c from the saturated raw pixels of colour c only."""
    from astropy.io import fits
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    H, W = shape
    for num, rc in sat.items():
        planes = 3 if layout else 1
        m = np.zeros((planes, H, W), np.uint8)
        if len(rc):
            rows = (H - 1 - rc[:, 0]) if flip else rc[:, 0]
            cols = rc[:, 1]
            if layout:
                colour = np.array([layout[(int(r_) % 2, int(c_) % 2)] for r_, c_ in zip(rows, cols)])
                for c in range(3):
                    sel = colour == c
                    m[c, rows[sel], cols[sel]] = 255
                    m[c] = cv2.dilate(m[c], se)
            else:
                m[0, rows, cols] = 255
                m[0] = cv2.dilate(m[0], se)
        fits.writeto(os.path.join(process, f"immsat_{num:05d}.fit"), m if layout else m[0], overwrite=True)


# --------------------------------------------------------------------------- preparation
def _remove(pattern: str) -> None:
    for p in glob.glob(pattern):
        try:
            os.remove(p)
        except OSError:
            pass


def clean_intermediates(process: str, seq: str) -> None:
    """Remove every file the preparation writes into ``process/`` (up to 100 MB per frame)."""
    for pat in (f"imm_*{seq}_*", f"imc_{seq}_*", f"bkg_im*{seq}_*", f"r_bkg_im*{seq}_*", f"r_im*{seq}_*", "immsat_*",
                "r_immsat_*", os.path.join("cache", f"r_bkg_im*{seq}_*.lst"), os.path.join("cache", f"r_im*{seq}_*.lst")):
        _remove(os.path.join(process, pat))
    shutil.rmtree(os.path.join(process, "drizztmp"), ignore_errors=True)


def siril_preprocess(session_dir: str, meta: dict, inc: list[int], rows: list[dict], say, checkpoint, cosmetic: bool = True,
                     subsky: bool = False, defer_masks: bool = False, local=None) -> dict:
    """Steps 1-3 of the module docstring.  Returns {"seq", "frames" (file number -> registered frame), "masks" (file
    number -> registered saturation mask, may be empty), "stars" (file number -> findstar list), "saturation"}.
    ``local``: library paths -> {library path: file to read} (Session.local_copy), for the raw frames."""
    from .siril import SirilRunner, quote, ranges
    process = os.path.join(session_dir, "process")
    state = read_json(os.path.join(process, "state.json"), {}) or {}
    seq, n = state.get("seq"), int(state.get("n") or 0)
    if not seq or not os.path.exists(os.path.join(process, f"{seq}_.seq")):
        raise RuntimeError("the registration (process/state.json, the calibrated sequence) is missing: stack again")
    # (an HDR stack - several exposure groups - is restored too: every group is registered in the one sequence, on one
    # grid; the engine judges each frame's transmission per second of exposure and weights its noise by its own scale)
    src = read_seq(os.path.join(process, f"{seq}_.seq"))
    reg, ref = registration_of(src)
    nums = [src["nums"][i] for i in inc]                             # file numbers of the included frames
    excluded = [i for i in range(len(src["nums"])) if i not in set(inc)]
    scale = float(meta.get("scale") or 1.0)
    clean_intermediates(process, seq)
    runner = SirilRunner(process, progress=say, checkpoint=checkpoint)

    # 1. debayer + per-frame background (VeraLux tab 1, no-master branch: `calibrate light -debayer`, `seqsubsky 1`)
    lines = [f"cd {quote(process)}", f"select {seq} 1 {n}"]
    lines += [f"unselect {seq} {a + 1} {b + 1}" for a, b in ranges(excluded)]
    # cosmetic correction of each sub (Siril find_cosme, 3 / 3 sigma): the stack rejects warm / cold pixels by sigma
    # clipping across frames, ImageMM has only its Huber weights - on an alt-az mount's rotating field a warm pixel the
    # dark map misses traces a dotted arc through the restoration (IC 1805)
    base = seq
    if cosmetic:
        lines.append(f"seqfind_cosme{'' if state.get('debayer') else '_cfa'} {seq} 3 3 -prefix=imc_")
        base = f"imc_{seq}"
    # the frames the stack integrated: a Bayer-drizzle stack's CFA frames are drizzled again with its own transforms
    # and drizzle parameters (no demosaicing: colours never mix; the pixels no drop reached are missing data).  The
    # script's `calibrate -debayer` + interpolation (RCD) mixed the colours of every star - M 31 2026-10-22: ImageMM
    # R/G = 0.32 + 1.28 x the stack's, the colour calibration's fit then off by an intercept of 0.95
    drizzle = meta.get("mode") == "drizzle" and not state.get("debayer") and not subsky
    if state.get("debayer") or drizzle:
        sub = base                                                   # (RGB already, or CFA for the drizzle)
    else:
        lines.append(f"calibrate {base} -debayer -prefix=imm_")
        sub = f"imm_{base}"
    # the script's per-frame absolute background (seqsubsky 1) is an option, off by default: on a large target the
    # fitted background takes part of the target's light, differently per frame (M 31 2026-10-21: the outer disk lost).
    # The engine normalises each frame against the frames' mean instead (veralux_imagemm change 23).
    if subsky:
        bkg = f"bkg_{sub}" if sub != seq else f"bkg_imm_{seq}"
        lines.append(f"seqsubsky {sub} 1 -prefix={bkg[:-len(sub)]}")
    elif sub == seq:
        bkg = f"imm_{seq}"                                           # (our own name over the stack's frames: links)
        for k in nums:
            dst = os.path.join(process, f"{bkg}_{k:05d}.fit")
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(f"{seq}_{k:05d}.fit", dst)
    else:
        bkg = sub
    if cosmetic or subsky or not (state.get("debayer") or drizzle):  # (any Siril processing to run)
        runner.run("\n".join(lines) + "\n", "imagemm_debayer_subsky", expect=[f"{bkg}_.seq"] if bkg != f"imm_{seq}" or subsky else [],
                   count_glob=f"{bkg}_[0-9]*.fit", total=len(nums), label="ImageMM: cosmetic correction, debayer" + (" + background (seqsubsky)" if subsky else ""))
    if subsky:
        _remove(os.path.join(process, f"imm_*{seq}_*"))
        _remove(os.path.join(process, f"imc_{seq}_*"))
    elif bkg != f"imc_{seq}":
        _remove(os.path.join(process, f"imc_{seq}_*"))
    from astropy.io import fits
    first = fits.getheader(os.path.join(process, f"{bkg}_{nums[0]:05d}.fit"))
    layers = int(first.get("NAXIS3") or 1) if int(first.get("NAXIS") or 2) == 3 else 1
    if os.path.exists(os.path.join(process, f"{bkg}_.seq")):
        write_registration(os.path.join(process, f"{bkg}_.seq"), nums, reg, ref, layers)
    else:
        write_registration(os.path.join(process, f"{bkg}_.seq"), nums, reg, ref, layers, name=bkg)

    # 2. registration with the stack's transforms + star lists (VeraLux: `register`, `findstar` per frame)
    rs = f"-framing=max -scale={scale:g}"
    if drizzle:
        dz = meta.get("drizzle") or {}
        reg_args = f"-drizzle -pixfrac={float(dz.get('pixfrac') or 1.0):g} -kernel={dz.get('kernel') or 'square'}"
    else:
        reg_args = f"-interp={INTERP}"
    lines = [f"cd {quote(process)}", f"seqapplyreg {bkg} {rs} {reg_args}", "setfindstar reset", f"seqfindstar r_{bkg}"]
    runner.run("\n".join(lines) + "\n", "imagemm_register", expect=[f"r_{bkg}_.seq"],
               count_glob=f"r_{bkg}_[0-9]*.fit", total=len(nums), label="ImageMM: registering the frames")
    _remove(os.path.join(process, f"{bkg}_*"))
    _remove(os.path.join(process, f"imm_*{seq}_*"))
    _remove(os.path.join(process, f"imc_{seq}_*"))

    # 3. saturation masks from the raw frames, one per colour: a drizzled pixel's channel c is censored only when a raw
    # pixel of colour c that drops into it was saturated.  One mask for all colours (any colour saturated nearby)
    # censored the unsaturated R / B measurements round a saturated G core - 50-84 % of the masked values were under
    # half the ceiling (M 31) - and the restoration grew a coloured crown on one side of every saturated star.
    say(0, 1, "ImageMM: saturation of the raw frames")
    raw_paths = {src["nums"][i]: rows[i].get("path") for i in inc if rows[i].get("path")}
    if local is not None:
        to_read = local(list(raw_paths.values()))
        raw_paths = {k: to_read.get(p, p) for k, p in raw_paths.items()}
    ceiling, sat, sat_info = raw_saturation(raw_paths, say, checkpoint)
    masks = False
    if ceiling is not None:
        sat, same, o_info = confirm_on_calibrated(sat, {k: os.path.join(process, f"{seq}_{k:05d}.fit") for k in sat}, ceiling)
        sat_info.update(o_info)
        if same is None:
            sat_info["skipped"] = "no raw-saturated pixel is still saturated in the calibrated frames"
        else:
            sat_info["raw_rows_reversed"] = not same
            sat_info["saturated_pixels_median"] = float(np.median([len(v) for v in sat.values()]))
            hdr = fits.getheader(os.path.join(process, f"{seq}_{nums[0]:05d}.fit"))
            shape = (int(hdr["NAXIS2"]), int(hdr["NAXIS1"]))
            layout = None
            if int(hdr.get("NAXIS") or 2) == 2:              # a CFA frame: which colour each 2 x 2 position is
                probe = nums[len(nums) // 2]
                layout, l_info = cfa_layout(os.path.join(process, f"{seq}_{probe:05d}.fit"),
                                            os.path.join(process, f"r_{bkg}_{probe:05d}.fit"))
                sat_info["cfa_layout"] = l_info
            write_saturation_masks(process, sat, shape, flip=not same, grow=SAT_GROW_DRIZZLE if drizzle else SAT_GROW, layout=layout)
            write_registration(os.path.join(process, "immsat_.seq"), sorted(sat), reg, ref, 3 if layout else 1, name="immsat", mono_layer=True)
            n_sat = len(sat)

            def register_masks():
                # (16-bit output: the masks are 0 / 1 and as 32-bit floats they needed 29 GB for M 31's 299 frames; every
                # script starts with set32bits again - siril.HEADER)
                runner.run(f"cd {quote(process)}\nset16bits\nseqapplyreg immsat {rs} -interp=nearest\n", "imagemm_register_masks",
                           expect=["r_immsat_.seq"], count_glob="r_immsat_[0-9]*.fit", total=n_sat, label="ImageMM: registering the saturation masks")
                _remove(os.path.join(process, "immsat_*"))
            if not defer_masks:
                register_masks()
            masks = True
    deferred = {"register_masks": register_masks} if (masks and defer_masks) else {}
    return {**deferred, "seq": seq, "nums": nums, "frames": {k: os.path.join(process, f"r_{bkg}_{k:05d}.fit") for k in nums},
            "masks": {k: os.path.join(process, f"r_immsat_{k:05d}.fit") for k in nums} if masks else {},
            "stars": {k: os.path.join(process, "cache", f"r_{bkg}_{k:05d}.lst") for k in nums},
            "seq_file": os.path.join(process, f"r_{bkg}_.seq"), "resampling": "drizzle" if drizzle else INTERP,
            "mask_seq_file": os.path.join(process, "r_immsat_.seq") if masks else None, "saturation": sat_info}


def build_store(work: str, prep: dict, keys: list[int], names: list[str], box: tuple[int, int, int, int],
                canvas: tuple[int, int], flip: bool, region: tuple[int, int, int, int] | None, say, checkpoint,
                consume: bool = False, with_masks: bool = True):
    """Step 4: the frames ``keys`` (file numbers) cropped to ``box`` (and ``region`` inside it), in stack.fits' row
    order, into a ``FrameStore`` with NaN where there is no data or a saturated pixel; their stars in its pixels.
    ``consume``: each registered frame and mask is deleted once it is in the store (the store and the registered
    frames together need ~2 x the frames' size on disk - M 31's 299 frames: 26 + 36 GB; the registration then failed
    for space, and the run fell back to BlurXTerminator)."""
    from .veralux_imagemm import FrameStore, parse_findstar, siril_to_array
    places = sequence_placements(prep["seq_file"], canvas)
    mplaces = sequence_placements(prep["mask_seq_file"], canvas) if (with_masks and prep.get("mask_seq_file")) else {}
    bx, by, bw, bh = box
    rx, ry, rw, rh = region if region else (0, 0, bw, bh)
    os.makedirs(work, exist_ok=True)
    store = FrameStore(os.path.join(work, "frames.npy"), (len(keys), 3, rh, rw))
    stats = {"stars": [], "invalid_fraction": [], "saturated_fraction": []}
    for k, num in enumerate(keys):
        if checkpoint:
            checkpoint()
        a = read_frame(prep["frames"][num], box, places[num])
        bad = (a == 0) | ~np.isfinite(a)                         # outside the frame's footprint (exactly 0)
        # the footprint's edge: within the Lanczos-4 kernel's reach of the outside a registered pixel is interpolated
        # against zeros (attenuated, not missing) and the debayer's edge rows are extrapolated - they were fitted as
        # data and printed bands along the image borders (M 31, 2026-10-22).  Footprint shrunk by EDGE_ERODE px.
        foot = (~bad.all(-1)).astype(np.uint8)
        e_ = EDGE_ERODE_DRIZZLE if prep.get("resampling") == "drizzle" else EDGE_ERODE
        foot = cv2.erode(foot, np.ones((2 * e_ + 1, 2 * e_ + 1), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=1)
        bad |= (foot == 0)[..., None]
        sat = np.zeros(a.shape, bool)                            # (h, w, 3): per channel (a mono mask: all three)
        if with_masks and num in prep.get("masks", {}):
            sat = read_frame(prep["masks"][num], box, mplaces[num]) > 0.5
        if flip:
            a, bad, sat = a[::-1], bad[::-1], sat[::-1]
        a = np.where(bad, np.float32(np.nan), a)[ry:ry + rh, rx:rx + rw]
        store.data[k] = np.moveaxis(a, -1, 0)
        sw_ = sat[ry:ry + rh, rx:rx + rw]
        store.sat[k] = (sw_[..., 0] * 1 + sw_[..., 1] * 2 + sw_[..., 2] * 4).astype(np.uint8)
        stats["invalid_fraction"].append(float((bad.any(-1) | sat.any(-1))[ry:ry + rh, rx:rx + rw].mean()))
        stats["saturated_fraction"].append(float(sw_.any(-1).mean()))
        ox, oy, fw, fh = places[num]
        stars = []
        for X, Y in parse_findstar(prep["stars"][num]):
            col, row = siril_to_array(X, Y, fh)
            cx, cy = ox + col - bx, oy + row - by                    # canvas -> the box
            if flip:
                cy = bh - 1 - cy
            cx, cy = cx - rx, cy - ry
            if 0 <= cx < rw and 0 <= cy < rh:
                stars.append((float(cx), float(cy)))
        store.stars[k] = stars
        store.names[k] = names[k]
        stats["stars"].append(len(stars))
        if consume:
            for p_ in (prep["frames"].get(num), (prep.get("masks") or {}).get(num) if with_masks else None):
                if p_:
                    try:
                        os.remove(p_)
                    except OSError:
                        pass
        if k % 10 == 0 or k == len(keys) - 1:
            say(k + 1, len(keys), f"ImageMM: frame store {k + 1}/{len(keys)}")
    store.flush()
    return store, stats


def add_masks(store, stats: dict, prep: dict, keys: list[int], box: tuple[int, int, int, int], canvas: tuple[int, int],
              flip: bool, region: tuple[int, int, int, int] | None, say, checkpoint, consume: bool = True) -> None:
    """The registered saturation masks into a store ``build_store`` made without them (``with_masks=False``): the
    frames are stored and deleted first, then the masks are registered into the space they freed (M 31's 299 frames:
    36 GB of registered frames and 29 GB of registered masks did not fit beside each other)."""
    mplaces = sequence_placements(prep["mask_seq_file"], canvas)
    bx, by, bw, bh = box
    rx, ry, rw, rh = region if region else (0, 0, bw, bh)
    for k, num in enumerate(keys):
        if checkpoint:
            checkpoint()
        if num not in prep.get("masks", {}):
            continue
        sat = read_frame(prep["masks"][num], box, mplaces[num]) > 0.5
        if flip:
            sat = sat[::-1]
        sw_ = sat[ry:ry + rh, rx:rx + rw]
        store.sat[k] = (sw_[..., 0] * 1 + sw_[..., 1] * 2 + sw_[..., 2] * 4).astype(np.uint8)
        stats["saturated_fraction"][k] = float(sw_.any(-1).mean())
        stats["invalid_fraction"][k] = float((~np.isfinite(store.data[k]).all(0) | sw_.any(-1)).mean())
        if consume:
            try:
                os.remove(prep["masks"][num])
            except OSError:
                pass
        if k % 20 == 0 or k == len(keys) - 1:
            say(k + 1, len(keys), f"ImageMM: saturation masks {k + 1}/{len(keys)}")
    store.flush()


# --------------------------------------------------------------------------- output
def rescale_wcs(header, r: int, offset: tuple[int, int] = (0, 0), t: int = 0):
    """The header of an image whose pixel u = r * (old pixel - offset) + t (0-based; ``veralux_imagemm.sr_origin``):
    CRPIX' = r (CRPIX - 1 - offset) + t + 1, the CD matrix (or CDELT) / r and the SIP terms (A_p_q ... of u, v
    relative to CRPIX) x r^(1 - p - q)."""
    h = header.copy()
    if "CRPIX1" in h and "CRPIX2" in h:
        h["CRPIX1"] = r * (float(h["CRPIX1"]) - 1 - offset[0]) + t + 1
        h["CRPIX2"] = r * (float(h["CRPIX2"]) - 1 - offset[1]) + t + 1
    for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2", "CDELT1", "CDELT2"):
        if k in h and r != 1:
            h[k] = float(h[k]) / r
    if r != 1:
        for k in list(h.keys()):
            m = re.fullmatch(r"(A|B|AP|BP)_(\d+)_(\d+)", k)
            if m:
                h[k] = float(h[k]) * r ** (1 - int(m.group(2)) - int(m.group(3)))
    return h


def match_to_stack(out: np.ndarray, ref: np.ndarray, r: int = 1, sigma: float = 8.0, ap: float = 8.0,
                   min_stars: int = 10) -> tuple[np.ndarray, list]:
    """``out`` (C, r H, r W) on the photometric scale of ``ref`` (C, H, W, stack.fits' window), per channel ref = a out + b.

    a: the median over the unsaturated stars of their flux ratio ref / out (aperture ``ap`` px, local sky from the
    annulus 1.25-1.75 ap) - photometric calibration on stars; the restoration conserves flux, so a star's light is the
    same in both.  b: the median of ref - a out over the image smoothed over ``sigma`` px.  The former fit of a line
    through the two smoothed images had no lever arm where the field is mostly dark sky: its slope came from a few stars
    and the field's brightest (clipped in the stack, not in the restoration) - green at 0.4 x on M 31's outer field
    (2026-10-23).  With fewer than ``min_stars`` stars that line (fitted after a 3-sigma clip) is kept."""
    from scipy.ndimage import maximum_filter
    res = []
    o = np.empty_like(out)
    C = out.shape[0]
    xs_ = [cv2.resize(out[c], (ref.shape[2], ref.shape[1]), interpolation=cv2.INTER_AREA) if r > 1 else out[c]
           for c in range(C)]
    g = ref[1 if C > 1 else 0].astype(np.float32)
    sm = cv2.GaussianBlur(g, (0, 0), 1.0)
    med = float(np.nanmedian(sm))
    sd = 1.4826 * float(np.nanmedian(np.abs(sm - med)))
    top = float(np.nanmax(ref)) if np.isfinite(ref).any() else 1.0
    R = int(np.ceil(1.75 * ap))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    rr = np.hypot(yy, xx)
    apm, anm = rr <= ap, (rr >= 1.25 * ap) & (rr <= 1.75 * ap)
    H, W = g.shape
    pk = (sm == maximum_filter(sm, 9)) & (sm > med + 30.0 * sd) & (np.nanmax(ref, 0) < 0.5 * top)
    ys, xs = np.nonzero(pk)
    sel = (ys > R) & (ys < H - R - 1) & (xs > R) & (xs < W - R - 1)
    ys, xs = ys[sel], xs[sel]
    for c in range(C):
        x = xs_[c].astype(np.float32)
        rc = ref[c].astype(np.float32)
        rat = []
        for y0, x0 in zip(ys, xs):
            wr, wo = rc[y0 - R:y0 + R + 1, x0 - R:x0 + R + 1], x[y0 - R:y0 + R + 1, x0 - R:x0 + R + 1]
            fr = float((wr[apm] - np.nanmedian(wr[anm])).sum())
            fo = float((wo[apm] - np.nanmedian(wo[anm])).sum())
            if fr > 0 and fo > 0 and np.isfinite(fr) and np.isfinite(fo):
                rat.append(fr / fo)
        lx = cv2.GaussianBlur(x, (0, 0), sigma)[::3, ::3].ravel()
        lr = cv2.GaussianBlur(rc, (0, 0), sigma)[::3, ::3].ravel()
        ok = np.isfinite(lx) & np.isfinite(lr)
        if len(rat) >= min_stars:
            a = float(np.median(rat))
            b = float(np.median(lr[ok] - a * lx[ok]))
        else:
            a, b = np.polyfit(lx[ok], lr[ok], 1)
            d = lr - (a * lx + b)
            keep = ok & (np.abs(d) <= 3 * 1.4826 * np.median(np.abs(d[ok])))
            a, b = np.polyfit(lx[keep], lr[keep], 1)
        o[c] = (out[c] * a + b).astype(np.float32)
        res.append([round(float(a), 6), round(float(b), 7)])
    return o, res


def clip_preserving_colour(out: np.ndarray) -> tuple[np.ndarray, int]:
    """``out`` (C, H, W) with every pixel whose brightest channel exceeds 1 scaled so that it is 1 (all channels by the
    same factor: the hue of a restored star core is kept); (result, pixels scaled)."""
    mx = out.max(0)
    over = mx > 1.0
    f = np.where(over, 1.0 / np.maximum(mx, 1e-12), 1.0).astype(np.float32)
    return np.ascontiguousarray(np.clip(out * f[None], 0.0, 1.0).astype(np.float32)), int(over.sum())


def _norm_peak(out: np.ndarray) -> float:
    """Scale that keeps the restored image in the [0, 1] the pipeline's linear files use: a restored star core can
    exceed the frames' ceiling.  The maximum of the image smoothed over 3 x 3 (a lone hot pixel does not set the scale;
    the few pixels above 1 after it are a star's single brightest pixels)."""
    sm = np.stack([cv2.blur(np.ascontiguousarray(c, np.float32), (3, 3)) for c in out])
    mx = float(np.nanmax(sm))
    if mx <= 1.0:
        return 1.0
    return 1.0 / mx


def _signature(keys, crop, region, scale, flip, cosmetic=True, subsky=False) -> dict:
    return {"version": PREP_VERSION, "cosmetic": cosmetic, "subsky": subsky, "frames": [int(k) for k in keys], "crop": list(crop),
            "region": list(region) if region else None, "scale": scale, "flip": bool(flip)}


def run(session_dir: str, stack_meta: dict, stack_params: dict, progress=None, checkpoint=None, local=None) -> dict:
    """Restore the included frames of a stacked session with VeraLux ImageMM -> ``imagemm.fits`` and
    ``imagemm_meta.json`` (returned).  Raises on failure: the restore job then fails with the reason (no BlurXTerminator in its place).
    ``local``: see ``siril_preprocess``."""
    av = available()
    if not av["ok"]:
        raise RuntimeError(av["reason"])
    import torch
    from . import veralux_imagemm as V
    t0 = time.time()
    sp = dict(stack_params or {})
    kw = map_params(sp)
    r = kw.pop("r")
    formal = kw.pop("formal_mode")
    say = progress or (lambda i, n, m: None)
    say(0, 1, "ImageMM: loading the stack")
    stack, header = fitsio.read_rgb(os.path.join(session_dir, "stack.fits"))
    meta = dict(stack_meta or read_json(os.path.join(session_dir, "stack_meta.json"), {}) or {})
    crop = [int(v) for v in meta.get("crop") or [0, 0, stack.shape[2], stack.shape[1]]]
    if (crop[2], crop[3]) != (stack.shape[2], stack.shape[1]):
        raise RuntimeError(f"stack.fits is {stack.shape[2]} x {stack.shape[1]} but stack_meta.crop is {crop}")
    canvas = tuple(int(v) for v in (meta.get("stack_canvas") or [crop[0] + crop[2], crop[1] + crop[3]]))
    region = sp.get("imagemm_region")
    off = (0, 0)
    if region:                                  # development: restore a window of the stack
        rx, ry, rw, rh = (int(v) for v in region)
        rx, ry = max(0, rx), max(0, ry)
        rw, rh = min(rw, crop[2] - rx), min(rh, crop[3] - ry)
        region, off = (rx, ry, rw, rh), (rx, ry)
    rows = included_rows(session_dir)
    inc = [i for i, row in enumerate(rows) if row.get("included")]
    if not inc:
        raise RuntimeError("no included frames")
    process = os.path.join(session_dir, "process")
    # (development: imagemm_work names another folder, so a production run never deletes a kept store)
    work = os.path.join(session_dir, "process", str(sp["imagemm_work"])) if sp.get("imagemm_work") else os.path.join(session_dir, WORK)
    keep = _flag(sp.get("imagemm_keep_prepared", False))
    state = read_json(os.path.join(process, "state.json"), {}) or {}
    seq = state.get("seq") or "pp_light"
    try:
        src_nums = read_seq(os.path.join(process, f"{seq}_.seq"))["nums"]
    except OSError:
        raise RuntimeError("the calibrated sequence is missing: stack again") from None
    maxf = int(sp.get("imagemm_max_frames") or 0)
    sel = inc
    if maxf and len(sel) > maxf:
        sel = sorted(sorted(sel, key=lambda i: -(rows[i].get("weight") or 0.0))[:maxf])
    keys = [src_nums[i] for i in sel]
    names = [rows[i].get("name") or str(src_nums[i]) for i in sel]
    sig_path = os.path.join(work, "store.json")
    store = prep_info = None
    old = read_json(sig_path, None) if keep else None
    try:
        if old and os.path.exists(os.path.join(work, "frames.npy")):
            sig = _signature(keys, crop, region, float(meta.get("scale") or 1.0), old.get("signature", {}).get("flip"),
                             _flag(sp.get("imagemm_cosmetic", True)), _flag(sp.get("imagemm_subsky", False)))
            if old.get("signature") == sig:
                store = V.FrameStore(os.path.join(work, "frames.npy"))
                store.stars = [[tuple(s) for s in st] for st in old["stars"]]
                store.names = list(old["names"])
                prep_info = old["info"]
                say(1, 1, "ImageMM: frame store reused")
        if store is None:
            shutil.rmtree(work, ignore_errors=True)
            t1 = time.time()
            prep = siril_preprocess(session_dir, meta, inc, rows, say, checkpoint, defer_masks=True, local=local, cosmetic=_flag(sp.get("imagemm_cosmetic", True)),
                                    subsky=_flag(sp.get("imagemm_subsky", False)))
            best = keys[len(keys) // 2]
            places = sequence_placements(prep["seq_file"], canvas)
            ext = (max(p[0] + p[2] for p in places.values()), max(p[1] + p[3] for p in places.values()))
            # (Siril rounds the -framing=max bounding box of interpolated and drizzled frames differently by up to 1 px
            # - IC 1805: 4062 vs 4063 - while each frame is placed by its own translation; the alignment decides)
            if abs(ext[0] - canvas[0]) > 1 or abs(ext[1] - canvas[1]) > 1:
                raise RuntimeError(f"the registered frames span {ext[0]} x {ext[1]}, stack.fits' canvas is {canvas[0]} x {canvas[1]}")
            probe = read_frame(prep["frames"][best], tuple(crop), places[best])
            flip, flip_info = stack_is_flipped(np.moveaxis(stack, 0, -1), probe)
            shift = alignment_shift(np.moveaxis(stack, 0, -1), probe[::-1] if flip else probe)
            if max(abs(v) for v in shift) > 0.5:
                raise RuntimeError(f"the registered frames are {shift} px off stack.fits")
            del probe
            # (the frames into the store first, each registered frame deleted as it goes; then the saturation masks are
            # registered into the space that freed and added - side by side they needed ~65 GB for M 31's 299 frames)
            store, st = build_store(work, prep, keys, names, tuple(crop), canvas, flip, region, say, checkpoint, consume=True,
                                    with_masks=not prep.get("register_masks"))
            if prep.get("register_masks"):
                prep["register_masks"]()
                add_masks(store, st, prep, keys, tuple(crop), canvas, flip, region, say, checkpoint)
            prep_info = {"saturation": prep["saturation"], "frames_flipped": {**flip_info, "flipped": bool(flip)},
                         "alignment_shift_px": shift, "stars_median": float(np.median(st["stars"])),
                         "invalid_fraction_median": float(np.median(st["invalid_fraction"])),
                         "saturated_fraction_median": float(np.median(st["saturated_fraction"])),
                         "seconds": round(time.time() - t1, 1)}
            clean_intermediates(process, prep["seq"])
            if keep:
                write_json(sig_path, clean_json({"signature": _signature(keys, crop, region, float(meta.get("scale") or 1.0), flip,
                                                                         _flag(sp.get("imagemm_cosmetic", True)),
                                                                         _flag(sp.get("imagemm_subsky", False))),
                                                 "stars": store.stars, "names": store.names, "info": prep_info}))

        # each frame's exposure time (an HDR set mixes them; the engine's transmission and noise follow it)
        store.exptimes = [float(rows[i].get("exptime") or 1.0) for i in sel]
        dev = pick_device(sp.get("device"))
        engine = V.ImageMMEngine(str(dev), formal_mode=formal)
        t2 = time.time()

        def cb(i, n, m):
            say(i, n, f"ImageMM: {m}")
        out = engine.run_imagemm(store, upscale_factor=r, callback=cb, cancel=checkpoint,
                                 log=lambda m: log.info("ImageMM: %s", m), **kw)
        t_rest = time.time() - t2
        info = engine.last_run_info
        if out.shape[0] == 1:
            out = np.repeat(out, 3, 0)
        if not np.isfinite(out).all():
            raise RuntimeError("ImageMM produced non-finite pixels")
        # the stack's photometric scale, offset and channel balance (post's AI tools and colour calibration work on
        # absolute levels; seqsubsky levelled each channel's sky and the frames sit ~14 x below the stack's scale)
        rx_, ry_ = off
        ref_win = stack[:, ry_:ry_ + out.shape[1] // r, rx_:rx_ + out.shape[2] // r]
        out, phot = match_to_stack(out, ref_win, r)
        # cores of stars saturated in the data are restored above the stack's ceiling (censored data, point sources): on
        # the stack's scale they exceed 1 (M 31: 22 x).  Scaling the whole image to fit made every other star 22 x
        # fainter after the stretch (they vanished); clipping each channel at 1 coloured the cores.  The pixels above 1
        # are scaled down as a whole, their colour ratio kept.
        k_norm = 1.0
        out, clipped = clip_preserving_colour(out)
        hdr = rescale_wcs(header, r, off, V.sr_origin(r))
        hdr["HISTORY"] = f"VeraLux ImageMM {V.VERSION} (Sukurdeep et al. 2025, AJ 170:233) of the registered frames"
        if r > 1:
            hdr["HISTORY"] = f"VeraLux ImageMM SR x{r:g}"
        fitsio.write_rgb(os.path.join(session_dir, OUT_FITS), out, hdr)
        kern = [k for k in (info.pop("kernels", None) or []) if k is not None]
        meta_out = {
            "ok": True, "method": f"VeraLux ImageMM {V.VERSION} ({'formal' if formal else 'practical'} mode)",
            "source": V.SOURCE, "paper": "Sukurdeep, Budavari, Connolly & Navarro 2025, AJ 170:233",
            "frames_used": len(info.get("frames_used") or range(len(store))), "frames_total": len(inc),
            "frames": [store.names[i] for i in (info.get("frames_used") or range(len(store)))],
            "frames_skipped": [{"frame": store.names[i], "why": why} for i, why in (info.get("frames_dropped") or {}).items()],
            "r": r, "scale": float(meta.get("scale") or 1.0) * r, "crop": crop, "region_offset": list(off),
            "shape": [out.shape[2], out.shape[1]],
            "psf": {"measured_on": "each registered frame's own stars (findstar, median of 33 x 33 cut-outs, green)",
                    "stars_median": float(np.median(info.get("psf_stars") or [0])),
                    "stars_min": int(min(info.get("psf_stars") or [0])),
                    "kernel_size": int(kern[0].shape[-1]) if kern else None,
                    "kernel_fwhm_px_median": [round(float(np.median([_fwhm(k[c]) for k in kern])) / r, 3)
                                              for c in range(kern[0].shape[0])] if kern else None,
                    # the restored stars' width (information: their equivalent Gaussian sigma)
                    "restored_sigma_px": restored_sigma(out, [(x_ * r, y_ * r) for x_, y_ in (store.stars[0] if len(store) else [])])},
            "params": {**kw, "r": r, "formal_mode": formal},
            "iterations": {"done": info.get("iters_done"), "max": info.get("iters_max"), "converged": info.get("converged"),
                           "mean_diff": info.get("mean_diff"), "mean_diff_history": info.get("mean_diff_history")},
            "stopping": f"Appendix C: mean |u_k / u_k-1 - 1| over the effective pixels (mu {kw['mu']:g}) < {kw['stop_epsilon']:g}",
            "norm_factor": info.get("norm_factor"), "var_floor_median": info.get("var_floor_median"),
            "effective_fraction": info.get("effective_fraction"), "preparation": prep_info,
            "peak_scale": k_norm, "photometric_match": phot, "clipped_above_1": clipped, "device": str(dev), "torch": torch.__version__,
            "seconds": round(time.time() - t0, 1), "seconds_restore": round(t_rest, 1),
            "seconds_per_frame_iteration": round(t_rest / max(1, info.get("iters_done") or 1) / max(1, len(store)), 4),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        write_json(os.path.join(session_dir, OUT_META), clean_json(meta_out))
        return clean_json(meta_out)
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)
        clean_intermediates(process, seq)


def restored_sigma(img: np.ndarray, stars: list[tuple[float, float]], half: int = 8) -> float | None:
    """The equivalent Gaussian sigma (px) of the restored stars: the median second moment of the isolated,
    unsaturated stars of ``stars`` in the green channel of ``img`` (C, H, W), local sky removed."""
    g = img[1] if img.shape[0] > 1 else img[0]
    H, W = g.shape
    top = float(np.percentile(g, 99.99))
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    out = []
    for x, y in stars:
        ix, iy = int(round(x)), int(round(y))
        if not (3 * half < ix < W - 3 * half and 3 * half < iy < H - 3 * half):
            continue
        big = g[iy - 3 * half:iy + 3 * half + 1, ix - 3 * half:ix + 3 * half + 1]
        c = g[iy - half:iy + half + 1, ix - half:ix + half + 1] - np.median(big)
        if c.max() >= 0.5 * top or c.max() <= 0:
            continue
        w = np.maximum(c, 0)
        s_ = w.sum()
        my, mx = (w * yy).sum() / s_, (w * xx).sum() / s_
        out.append(np.sqrt(((w * ((yy - my) ** 2 + (xx - mx) ** 2)).sum() / s_) / 2))
    return round(float(np.median(out)), 3) if len(out) >= 5 else None


def _fwhm(k: np.ndarray) -> float:
    """FWHM (px) of a kernel from its second moments (Gaussian equivalent)."""
    k = np.maximum(np.asarray(k, np.float64), 0)
    s = k.sum()
    if s <= 0:
        return 0.0
    yy, xx = np.mgrid[:k.shape[0], :k.shape[1]]
    cy, cx = (k * yy).sum() / s, (k * xx).sum() / s
    var = ((k * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum() / s) / 2
    return float(2.3548 * np.sqrt(max(var, 0.0)))
