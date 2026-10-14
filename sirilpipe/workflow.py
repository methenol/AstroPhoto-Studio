"""The recipe: pure functions that turn a classified dataset + calibration + stack_params into
Siril scripts (``.ssf`` text) and into the frame-selection / drizzle decisions.

Nothing here runs Siril or touches the disk (except ``stage_inputs``, which creates the staging
symlinks), so the whole recipe is unit-testable.  Every number that comes from the community
rather than from us carries its source in a comment.

Pipeline (one ``siril-cli`` per stage; commands verified on Siril 1.4.4, see siril.md)::

    link light            symlinks of the selected subs only (``failed_*`` etc. never reach Siril)
    calibrate light ...   dark/flat when matched, ``-cfa``: keep the Bayer mosaic (Bayer drizzle)
    [merge]               several nights / exposure groups -> one sequence, one registration
    register -2pass       transforms + best reference + per-frame fwhm/wfwhm/roundness/nbstars
    unselect ...          frames our rules (frames.json "included") exclude
    seqapplyreg -drizzle  Bayer drizzle onto the -framing=max canvas of the included frames
    stack rej w ...       Winsorized sigma clipping, wFWHM weights, additive+scale normalisation
"""
from __future__ import annotations

import math
import os
import shutil
import statistics
from dataclasses import dataclass, field

from . import config
from .dataset import Frame
from .siril import quote, ranges

# --------------------------------------------------------------------------- constants
# Registration: official-script defaults + spec-chum's DWARF recipe
# (https://github.com/spec-chum/dwarf-siril-script: register -2pass -transf=homography -minpairs=10 -maxstars=2000)
REGISTER_ARGS = "-2pass -transf=homography -minpairs=10 -maxstars=2000"
# Official Seestar_Preprocessing.ssf v1.3 (siril-scripts): frames more than 2.5 sigma rounder-than-worst are dropped
ROUNDNESS_K = 2.5
# Bayer drizzle: Siril docs / OSC_Preprocessing_BayerDrizzle.ssf: scale 1, pixfrac 1, square kernel for OSC
DRIZZLE_1X = {"scale": 1.0, "pixfrac": 1.0, "kernel": "square"}
# 2x drizzle only with plenty of data and undersampled stars: PixInsight DrizzleIntegration guidance quoted in
# workflows.md section 10 (FWHM < ~2 px) and the community minimum of 100 subs (lancepickens: 50+; DESIGN: 100)
DRIZZLE_2X = {"scale": 2.0, "pixfrac": 0.5, "kernel": "square"}   # pixfrac ~ 1/scale (Siril drizzle docs)
DRIZZLE2_MIN_FRAMES = 100
DRIZZLE2_MAX_FWHM = 2.0
# Frames kept by `sensitivity` (DESIGN section 4): 1.0 -> best 90 % by wFWHM, 0.5 -> 97 %, 2.0 -> 75 %
KEEP_PERCENT = ((0.5, 97.0), (1.0, 90.0), (2.0, 75.0))
MIN_STARS = 5                    # DESIGN: subs with nbstars < 5 are excluded
FASTNORM_FRAMES = config.FASTNORM_FRAMES   # Siril docs/forums: -fastnorm for > 1000 short subs (siril.md 2.4)


# --------------------------------------------------------------------------- data
@dataclass
class Unit:
    """Frames calibrated together: one night folder x one exposure group (own darks/flats)."""
    index: int
    dir: str                       # absolute work dir (the process dir itself when there is one unit)
    frames: list[Frame]
    calib: dict = field(default_factory=dict)   # calibration report (dark/flat/bias choices)
    masters: dict = field(default_factory=dict)  # kind -> absolute master path actually passed to Siril
    group: int = 0                 # exposure group (HDR: one per exposure time; 0 when there is only one)

    @property
    def calibrated(self) -> bool:
        return bool(self.masters.get("dark") or self.masters.get("flat") or self.masters.get("bias"))


# --------------------------------------------------------------------------- decisions
def keep_percent(sensitivity: float) -> float:
    """% of frames kept by weighted FWHM for a ``sensitivity`` (piecewise linear through DESIGN's three points)."""
    s = float(sensitivity)
    pts = KEEP_PERCENT
    if s <= pts[0][0]:
        return min(100.0, pts[0][1] + (pts[0][0] - s) * 14.0)
    for (s0, p0), (s1, p1) in zip(pts, pts[1:]):
        if s <= s1:
            return p0 + (p1 - p0) * (s - s0) / (s1 - s0)
    return max(50.0, pts[-1][1] - (s - pts[-1][0]) * 15.0)


def decide_resampling(params: dict, n_included: int, median_fwhm: float | None, cfa: bool = True) -> dict:
    """How the registered frames are resampled: Bayer drizzle 1x (default), 2x only when the rule allows,
    demosaic + Lanczos when asked (or when the data is not a Bayer mosaic).  ``reason`` is recorded in
    stack_meta so the user sees why a requested scale was not used."""
    mode = str(params.get("mode", "auto") or "auto").lower()
    want = float(params.get("scale", 1.0) or 1.0)
    out: dict = {"mode": "drizzle", "requested_scale": want, "reason": ""}
    if mode == "demosaic" or not cfa:
        out.update(mode="demosaic", scale=1.0, pixfrac=None, kernel=None, interp="lanczos4",
                   reason="demosaic + Lanczos4 registration" if cfa else "not a Bayer mosaic")
        return out
    if want > 1.0:
        ok = n_included >= DRIZZLE2_MIN_FRAMES and median_fwhm is not None and median_fwhm < DRIZZLE2_MAX_FWHM
        if ok:
            out.update(DRIZZLE_2X)
            out["reason"] = f"2x drizzle: {n_included} frames, median FWHM {median_fwhm:.2f} px"
            return out
        why = (f"only {n_included} frames (< {DRIZZLE2_MIN_FRAMES})" if n_included < DRIZZLE2_MIN_FRAMES
               else f"median FWHM {median_fwhm:.2f} px is not < {DRIZZLE2_MAX_FWHM} px"
               if median_fwhm is not None else "no FWHM measurement")
        out.update(DRIZZLE_1X)
        out["reason"] = f"requested {want:g}x drizzle, using 1x: {why}"
        return out
    out.update(DRIZZLE_1X)
    out["reason"] = "Bayer drizzle 1x (scale 1, pixfrac 1)"
    return out


def rejection(params: dict, resampling: dict) -> dict:
    """Rejection type and sigmas: Winsorized (``w``); MAD (``a``) for drizzled data above 1x (the Siril stacking
    doc calls it the most effective for the sparse R/B coverage); never GESDT (bug #1993 fixed only in 1.4.5)."""
    lo = float(params.get("sigma_low", 4.0) or 4.0)
    hi = float(params.get("sigma_high", 3.0) or 3.0)
    kind = "a" if (resampling["mode"] == "drizzle" and resampling.get("scale", 1.0) > 1.0) else "w"
    return {"type": kind, "name": {"w": "winsorized", "a": "mad"}[kind], "low": lo, "high": hi}


def select_frames(stats: list[dict], sensitivity: float = 1.0) -> list[list[str]]:
    """Reasons to exclude each frame ([] = keep).  ``stats``: per frame dicts with registered (bool), nbstars,
    roundness, wfwhm (from the .seq).  Rules (same semantics as Siril's -filter-* options, evaluated here so the
    UI shows exactly what will be stacked and user overrides compose with them):
      1. registration failed (Siril excluded the frame);
      2. fewer than MIN_STARS stars;
      3. roundness more than ROUNDNESS_K sigma below the mean (``-filter-round=2.5k``);
      4. worse than the best ``keep_percent(sensitivity)`` % by weighted FWHM (``-filter-wfwhm=N%``)."""
    n = len(stats)
    reasons: list[list[str]] = [[] for _ in range(n)]
    alive = []
    for i, s in enumerate(stats):
        if not s.get("registered", True):
            reasons[i].append("registration failed (no matching star field: clouds, or too few stars)")
        elif s.get("nbstars") is None or s["nbstars"] < MIN_STARS:
            reasons[i].append(f"fewer than {MIN_STARS} stars")
        else:
            alive.append(i)
    rr = [stats[i]["roundness"] for i in alive if stats[i].get("roundness") is not None]
    if len(rr) >= 10:
        mu, sd = statistics.fmean(rr), statistics.pstdev(rr)
        thr = mu - ROUNDNESS_K * sd
        for i in list(alive):
            r = stats[i].get("roundness")
            if r is not None and r < thr:
                reasons[i].append(f"stars too elongated (roundness {r:.2f} < {thr:.2f})")
                alive.remove(i)
    pct = keep_percent(sensitivity)
    if len(alive) >= 10 and pct < 100.0:
        order = sorted(alive, key=lambda i: (stats[i].get("wfwhm") is None, stats[i].get("wfwhm") or 0.0))
        keep = max(3, int(math.ceil(len(order) * pct / 100.0)))
        cut = stats[order[keep - 1]].get("wfwhm")
        for i in order[keep:]:
            reasons[i].append(f"weighted FWHM {stats[i].get('wfwhm') or 0:.2f} px above the best {pct:.0f}% cut ({cut:.2f})")
    return reasons


# --------------------------------------------------------------------------- staging
def stage_inputs(unit: Unit, paths: list[str], prefix: str = "light") -> str:
    """Create ``<unit.dir>/src/<prefix>_00001.fit ...`` symlinks (in the given order) and remove every
    stale product of an earlier run (sequences, calibrated and registered frames, drizzle temp).  Siril's
    ``link`` then sees exactly the frames we want, whatever else sits in the telescope's folder (``failed_*``,
    jpg/png, stacks).  Why not point Siril at the raw folder: ``link``/``convert`` take every FITS in it."""
    os.makedirs(unit.dir, exist_ok=True)
    clean_process_dir(unit.dir)
    src = os.path.join(unit.dir, "src")
    os.makedirs(src, exist_ok=True)
    for i, p in enumerate(paths, 1):
        os.symlink(os.path.abspath(p), os.path.join(src, f"{prefix}_{i:05d}.fit"))
    return src


def clean_process_dir(d: str, keep_logs: bool = True) -> None:
    """Delete sequences and frames of a previous run in ``d`` (Siril refuses to re-create a sequence over a
    stale .seq: siril.md section 8.6) but keep logs/ and scripts/."""
    if not os.path.isdir(d):
        return
    for name in os.listdir(d):
        p = os.path.join(d, name)
        if name in ("logs", "scripts") and keep_logs:
            continue
        if os.path.isdir(p) and not os.path.islink(p):
            if name.startswith(("src", "drizztmp", "cache", "u")) or name == "src":
                shutil.rmtree(p, ignore_errors=True)
            continue
        if name.endswith((".fit", ".fits", ".seq", ".lst", ".txt", ".csv")) or os.path.islink(p):
            try:
                os.remove(p)
            except OSError:
                pass


def clean_registered(d: str) -> None:
    """Remove the registered (r_*) frames and their sequence: the largest intermediates (up to 100 MB per
    frame for a drizzled Seestar sub)."""
    for name in os.listdir(d) if os.path.isdir(d) else []:
        if name.startswith("r_") and name.endswith((".fit", ".seq")):
            try:
                os.remove(os.path.join(d, name))
            except OSError:
                pass
    shutil.rmtree(os.path.join(d, "drizztmp"), ignore_errors=True)


# --------------------------------------------------------------------------- scripts
def master_script(kind: str, work: str, out_base: str, bias: str | None = None) -> str:
    """Stack raw calibration frames (staged as ``<work>/src/<kind>_NNNNN.fit``) into ``<out_base>.fit``.

    dark / bias: ``stack <seq> rej 3 3 -nonorm`` (official OSC_Preprocessing.ssf).
    flat: calibrate with the bias master, then ``stack pp_flat rej 3 3 -norm=mul`` (same script).
    """
    lines = [f"cd {quote(os.path.join(work, 'src'))}", f"link {kind} -out={quote(work)}", f"cd {quote(work)}"]
    if kind == "flat":
        lines.append(f"calibrate flat {quote('-bias=' + bias) if bias else ''}".rstrip())
        lines.append(f"stack pp_flat rej 3 3 -norm=mul {quote('-out=' + out_base)}")
    else:
        lines.append(f"stack {kind} rej 3 3 -nonorm {quote('-out=' + out_base)}")
    return "\n".join(lines) + "\n"


def flat_bias_script(work: str, flat: str, bias: str, prefix: str = "cal_") -> str:
    """Subtract the bias master from an already-stacked flat master (DWARF CALI_FRAME flats): the flat is then
    normalised by Siril, but its pedestal (~200 ADU of 2400) would flatten the correction by ~8 %."""
    return "\n".join([f"cd {quote(work)}", f"calibrate_single {quote(flat)} {quote('-bias=' + bias)} -prefix={prefix}"]) + "\n"


def calibrate_script(unit: Unit, debayer: bool, force: bool = False) -> str:
    """link + calibrate one unit.  ``-cfa`` keeps the mosaic for Bayer drizzle; ``-debayer`` only for the demosaic
    path.  Bias is never passed together with a dark (it would be subtracted twice: Naztronomy, workflows.md 11.1).
    Without any master and without debayering, calibrate is skipped altogether (it would only copy the frames),
    unless ``force`` (another unit of a merged sequence is calibrated: all units must produce ``pp_light``)."""
    lines = [f"cd {quote(os.path.join(unit.dir, 'src'))}", f"link light -out={quote(unit.dir)}", f"cd {quote(unit.dir)}"]
    m = unit.masters
    if unit.calibrated or debayer or force:
        args = ["calibrate light"]
        if m.get("dark"):
            args.append(quote("-dark=" + m["dark"]))
        elif m.get("bias"):
            args.append(quote("-bias=" + m["bias"]))
        if m.get("flat"):
            args.append(quote("-flat=" + m["flat"]))
        if m.get("dark"):
            # official scripts use bare -cc=dark; spec-chum/DSA use "3 3" (hot and cold pixel sigma)
            args.append("-cc=dark 3 3")
        if debayer:
            args.append("-debayer")
        else:
            args.append("-cfa")
            if m.get("flat"):
                args.append("-equalize_cfa")
        args.append("-prefix=pp_")
        lines.append(" ".join(args))
    return "\n".join(lines) + "\n"


def seq_name(units: list[Unit], debayer: bool) -> str:
    """Name of the (merged) calibrated sequence the registration works on."""
    return "pp_light" if (debayer or any(u.calibrated for u in units)) else "light"


def merge_script(units: list[Unit], process_dir: str, seq: str) -> str:
    """Merge the units' sequences into ``<process_dir>/<seq>`` (one registration => true multi-night rejection;
    DESIGN section 4)."""
    srcs = " ".join(quote(os.path.join(u.dir, seq)) for u in units)
    return f"cd {quote(process_dir)}\nmerge {srcs} {seq}\n"


def register_script(process_dir: str, seq: str, n: int, resampling: dict | None = None) -> str:
    """Select everything, then compute the 2-pass registration (transforms only: no output images)."""
    return "\n".join([f"cd {quote(process_dir)}", f"select {seq} 1 {n}", f"register {seq} {REGISTER_ARGS}"]) + "\n"


def apply_script(process_dir: str, seq: str, n: int, excluded: list[int], resampling: dict) -> str:
    """``unselect`` the excluded frames (0-based indices; Siril's select/unselect are 1-based and inclusive),
    then ``seqapplyreg`` of the included ones.  ``-filter-included`` makes Siril honour the selection."""
    lines = [f"cd {quote(process_dir)}", f"select {seq} 1 {n}"]
    for a, b in ranges(excluded):
        lines.append(f"unselect {seq} {a + 1} {b + 1}")
    args = [f"seqapplyreg {seq}", "-framing=max", f"-scale={resampling.get('scale', 1.0):g}"]
    if resampling["mode"] == "drizzle":
        args += ["-drizzle", f"-pixfrac={resampling['pixfrac']:g}", f"-kernel={resampling['kernel']}"]
    else:
        args.append(f"-interp={resampling.get('interp', 'lanczos4')}")
    args.append("-filter-included")
    lines.append(" ".join(args))
    return "\n".join(lines) + "\n"


def stack_script(process_dir: str, seq: str, rej: dict, n_stacked: int, out: str = "stack_raw",
                 weight: str = "wfwhm", output_norm: bool = True) -> str:
    """The final integration of the registered frames.  Flags: the community consensus of siril.md section 5.6
    (``-norm=addscale -output_norm -rgb_equal -maximize -32b -weight=wfwhm``; ``-fastnorm`` above 1000 frames)."""
    args = [f"stack r_{seq}", f"rej {rej['type']} {rej['low']:g} {rej['high']:g}", "-norm=addscale"]
    if output_norm:
        args.append("-output_norm")
    args += ["-rgb_equal", "-maximize", "-32b"]
    if weight:
        args.append(f"-weight={weight}")
    args.append("-filter-included")
    if n_stacked > FASTNORM_FRAMES:
        args.append("-fastnorm")
    args.append(quote("-out=" + out))
    return f"cd {quote(process_dir)}\n" + " ".join(args) + "\n"


def group_stack_script(process_dir: str, seq: str, n_reg: int, others: list[int], ref_pos: int, rej: dict,
                       out: str, n_stacked: int) -> str:
    """Stack ONE exposure group of the registered sequence (HDR).  ``others`` = 1-based positions (in the r_
    sequence) of the frames of the other groups, ``ref_pos`` = position of this group's best frame.

    * ``setref``: the additive+scale normalisation matches every frame to the sequence's reference; making the
      reference a frame of THIS group keeps the group's own exposure scale, which the HDR blend needs (a long
      exposure normalised to a short reference would lose the exposure ratio);
    * no ``-output_norm`` and no ``-rgb_equal`` for the same reason (the blend works on radiance; colour balance
      is equalised once, on the blend)."""
    lines = [f"cd {quote(process_dir)}", f"select r_{seq} 1 {n_reg}"]
    for a, b in ranges([o - 1 for o in others]):
        lines.append(f"unselect r_{seq} {a + 1} {b + 1}")
    lines.append(f"setref r_{seq} {ref_pos}")
    args = [f"stack r_{seq}", f"rej {rej['type']} {rej['low']:g} {rej['high']:g}", "-norm=addscale", "-maximize", "-32b",
            "-weight=wfwhm", "-filter-included"]
    if n_stacked > FASTNORM_FRAMES:
        args.append("-fastnorm")
    args.append(quote("-out=" + out))
    lines.append(" ".join(args))
    return "\n".join(lines) + "\n"


def platesolve_script(session_dir: str, stack: str, frame: Frame | None, force: bool = True,
                      solved: str = "stack_solved") -> str:
    """Plate-solve the stack with Siril (online catalogues: NOMAD/Gaia via VizieR).

    Centre and scale come from the stack header (it inherits RA/DEC/FOCALLEN/XPIXSZ from the first sub: DWARF has
    RA/DEC in degrees + FOCALLEN + XPIXSZ, Seestar has RA/DEC, ASIAIR too).  Explicit centre coordinates are NOT
    passed: Siril 1.4.4 rejects every format we tried ("83.8,-5.4", "83.8 -5.4", sexagesimal) as an invalid
    argument, while the header route works.  ``-focal``/``-pixelsize`` are given explicitly (the pixel size of the
    stack, which differs from the sub's when drizzled at 2x).  Siril flips an upside-down solution to north-up
    (default behaviour, kept: the exported image is then upright)."""
    lines = [f"cd {quote(session_dir)}", f"load {quote(stack + '.fits')}"]   # explicit: a stale stack.fit would win
    args = ["platesolve" + (" -force" if force else "")]
    if frame is not None and frame.focallen:
        args.append(f"-focal={frame.focallen:g}")
    if frame is not None and frame.pixsize:
        args.append(f"-pixelsize={frame.pixsize:g}")
    lines.append(" ".join(args))
    # `setext fit`: Siril writes <name>.fit; the session moves it over stack.fits afterwards
    lines.append(f"save {quote(solved)}")
    lines.append("close")
    return "\n".join(lines) + "\n"
