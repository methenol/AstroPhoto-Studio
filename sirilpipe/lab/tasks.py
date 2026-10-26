"""Tunable experiments of the Siril backend: what a trial runs and how it is scored.

A task declares its parameters (search space with the pipeline default of each and what it maps to:
``pipeline`` = an integration & compute option (stack_params / STACK_DEFAULTS), ``processing`` = a processing
setting (post.DEFAULTS / EXTRA_DEFAULTS), ``code`` = a constant of the code), its metrics (direction of improvement,
whether they need a ground truth) and two functions: ``prepare`` loads what a study shares between trials, once;
``run`` executes one trial and returns its metrics and a preview image (h, w, 3).

Search spaces.  A parameter may be conditional: ``when`` = {other: [values]} (all must hold) or a list of such dicts
(any may hold); an inactive parameter is not sampled and keeps its fixed value, so a trial is never spent on a setting
that has no effect (the RBF smoothing of a polynomial model, the Cosmic Clarity amounts with Sharpen off).
``hypertune`` marks the parameters the hyper-tuned pipeline (sirilpipe.hypertune) searches: True, or a condition on
the dataset ("dual": dual-band data only); the palette and the colour grade are matters of style, the user's, never searched;
``ht_choices`` narrows a categorical there.

Scores and constraints.  ``score`` is the task's figure of merit (the hyper-tune objective and the lab's default),
``limits(ctx)`` its constraints ({metric: ("max" | "min", bound)}), whose violations every trial records (Optuna's
constrained TPE / NSGA-II, Ozaki et al. 2022): the best trial is the best feasible one, and a trial within
``min_gain`` of the baseline on the score does not replace it.  The measurements are paired (lab.objectives): the
same stars and sky apertures, chosen once on a reference image, in every trial.

Real datasets are the pipeline's own sessions (read only: trials work in the study's own folder on copies of
the cached products; the hyper-tune's stack study re-stacks the session itself, which it stacks next anyway).
Synthetic datasets (``synthetic.py``) are Seestar-format subs of an analytic sky with an exact truth, stacked once
by Siril; truth metrics compare the result with the truth registered onto the stack's grid (astroalign on the
stars, so the crop, flip and rotation of the stack do not matter).

Tasks (pipeline order):
  stack       registration & integration (Session.run_stack): rejection, weighting, frame selection, resampling, drizzle drop / kernel
  satellite   Cosmic Clarity satellite-trail removal on the stack: trails left vs change elsewhere
  background  gradient removal (the restore stage's background step): model, degree, smoothing
  restore     the linear restoration (post.restore): deconvolution / denoise tools and strengths (BXT, GraXpert, RL; NXT, GraXpert, Siril)
  stars       star separation (post.remove_stars): StarXTerminator, StarNet2, Cosmic Clarity DarkStar
  finish      the finishing chain (pro.py): stretch, highlights, noise reduction (NXT / Cosmic Clarity), Cosmic Clarity Sharpen, palette, colour, stars
  superres    super-resolution of the finished image: Cosmic Clarity Super-Resolution against Lanczos
"""
from __future__ import annotations

import json
import math
import os
import pickle
import shutil
import time

import cv2
import numpy as np

from . import metrics as MX
from . import objectives as OB
from . import synthetic as SY


# ----------------------------------------------------------------------------- datasets
class Dataset:
    """A real session (``{"kind": "real", "folder": <subs>}``, several nights of one target joined with
    ``os.pathsep``) or a generated synthetic one (``{"kind": "synthetic", "dir": <output/lab/synthetic/name>}``).
    ``session``: a Session object to use instead of opening one (the hyper-tune passes the job's own)."""

    def __init__(self, spec: dict, workdir: str, session=None):
        from ..util import split_folders
        self.spec = spec
        self.kind = spec["kind"]
        self.given = session
        if self.kind == "synthetic":
            self.dir = spec["dir"]
            self.folder = os.path.join(self.dir, "subs")
            self.workdir = os.path.join(self.dir, "work")
        else:
            self.dir = None
            self.folder = os.pathsep.join(split_folders(spec["folder"]))
            self.workdir = workdir
        if self.spec.get("name"):
            self.name = self.spec["name"]
        elif self.dir:
            self.name = os.path.basename(os.path.normpath(self.dir))
        else:
            parts = split_folders(self.folder)
            self.name = os.path.basename(os.path.normpath(parts[0])) + (f" (+{len(parts) - 1} nights)" if len(parts) > 1 else "")

    @property
    def synthetic(self) -> bool:
        return self.kind == "synthetic"

    def session(self, workdir: str | None = None):
        if self.given is not None and workdir is None:
            return self.given
        from ..session import Session
        return Session(self.folder, workdir or self.workdir)

    def ensure_stacked(self, log, cancel):
        """The dataset's session, analysed and stacked with the pipeline defaults if needed (synthetic datasets
        only: a real session must have been stacked in the pipeline)."""
        s = self.session()
        if self.given is None:
            s.cancel_flag = cancel
        if s.status()["stacked"]:
            return s
        if not self.synthetic:
            raise RuntimeError(f"{self.folder} has not been stacked: run 'Register & integrate' first")
        log("Analysing and stacking the synthetic subs with Siril (once per dataset)")
        sp = dict(self.spec.get("stack", {}))
        s.run_analysis(1.0, _prog(log), stack_params=sp)
        s.run_stack(sp, _prog(log))
        s.plate_solve(_prog(log))
        return s


def _prog(log, every: int = 25):
    def f(i, n, msg):
        if i == 1 or i == n or i % every == 0:
            log(msg)
    return f


def stretch(img: np.ndarray, ref: dict | None = None) -> tuple[np.ndarray, dict]:
    """8-bit BGR preview in an asinh stretch; ``ref`` (from the first trial) keeps it identical across the
    trials of a study.  ``img``: (h, w, 3) linear."""
    if ref is None:
        L = img.mean(-1)
        med = float(np.median(L))
        sig = float(1.4826 * np.median(np.abs(L - med))) or 1e-6
        hi = float(np.percentile(L, 99.8))
        ref = {"bg": med, "k": 2 * sig, "top": float(np.arcsinh(max(hi - med, 1e-6) / (2 * sig)))}
    y = np.arcsinh(np.maximum(img - ref["bg"], 0) / ref["k"]) / ref["top"]
    return (np.clip(y, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8), ref


def _hwc(arr: np.ndarray) -> np.ndarray:
    """(3, H, W) -> (H, W, 3) contiguous float32."""
    return np.ascontiguousarray(np.moveaxis(arr, 0, -1), np.float32)


def _mean(v):
    v = [q for q in (v or []) if q is not None]
    return float(np.mean(v)) if v else float("nan")


def _copy_products(src_dir: str, dst_dir: str, names) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    for n in names:
        p = os.path.join(src_dir, n)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(dst_dir, n))


def _read(path: str) -> np.ndarray:
    from .. import fitsio
    return fitsio.read_rgb(path)[0]


def _checkpoint(cancel):
    def f():
        if cancel.is_set():
            raise RuntimeError("cancelled")
    return f


def _centre_crop(img: np.ndarray, c: int = 1024) -> np.ndarray:
    H, W = img.shape[:2]
    y0, x0 = max(0, H // 2 - c // 2), max(0, W // 2 - c // 2)
    return np.ascontiguousarray(img[y0:y0 + c, x0:x0 + c])


def _store(study_dir: str, name: str, obj=None):
    """Pickle ``obj`` into the study folder (``obj`` None: load it back, None when absent): what a continued study must measure against."""
    path = os.path.join(study_dir, name)
    if obj is None:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    os.replace(tmp, path)
    return obj


# ----------------------------------------------------------------------------- truth on the stack's grid
def _truth_render(ds: Dataset, sc: float, extended: bool, stars: bool) -> np.ndarray:
    """The true sky on the reference sub's grid at scale ``sc`` through the subs' mean PSF, rendered once per dataset (minutes on the CPU):
    only its registration onto a result differs between trials."""
    key = (round(float(sc), 6), bool(extended), bool(stars))
    cache = ds.__dict__.setdefault("_truth_renders", {})
    if key not in cache:
        truth, _ = SY.load_truth(ds.dir)
        spec = truth["spec"]
        n = len(truth["frames"])
        psf = SY.effective_psf(ds.dir, list(range(n)), None, sc)
        H0, W0 = int(spec["height"] * sc), int(spec["width"] * sc)
        cache[key] = SY.truth_image(ds.dir, 0, (H0, W0), sc, psf, extended=extended, stars=stars)
    return cache[key]


def truth_on_stack(ds: Dataset, s, stack: np.ndarray, extended: bool = True, stars: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """The true sky rendered on the reference sub's grid through the subs' mean PSF, registered onto the
    stack's grid (crop, flip, rotation, drizzle scale) by astroalign on the star positions.  Returns
    (truth (H, W, 3) in the truth's units, valid mask, info)."""
    import astroalign as aa
    sc = float(s.meta.get("scale", 1.0))
    full = _truth_render(ds, sc, extended, stars)
    full_stars = _truth_render(ds, sc, False, True)
    # control points: the brightest genuine stars of each (a synthetic stack keeps hot-pixel blocks, which swamped astroalign's own detection);
    # the noiseless render gets a pedestal and a whisper of noise so that the detector has a background to measure
    peak = float(full_stars.max()) or 1.0
    noisy = full_stars + np.float32(1e-3 * peak) + np.random.default_rng(0).normal(0, 1e-4 * peak, full_stars.shape).astype(np.float32)
    dst_img = _hwc(stack)
    tf, (s_pts, d_pts) = aa.find_transform(OB.star_points(noisy), OB.star_points(dst_img), max_control_points=120)
    H, W = dst_img.shape[:2]
    M = tf.params[:2]
    out = np.stack([cv2.warpAffine(full[..., c].astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR, borderValue=np.nan) for c in range(3)], -1)
    valid = np.isfinite(out).all(-1)
    out = np.nan_to_num(out)
    info = {"matched_stars": int(len(s_pts)), "scale": float(tf.scale), "rotation_deg": float(np.degrees(tf.rotation))}
    return out, valid, info


# ----------------------------------------------------------------------------- base
class Task:
    name = ""
    label = ""
    description = ""
    params: list[dict] = []
    metrics: dict[str, dict] = {}
    options: list[dict] = []
    default_objective = ""
    score = ""                      # the figure of merit: the hyper-tune objective
    min_gain = 0.0                  # an improvement of the score below this does not replace the baseline (the measurement's resolution)
    min_gain_rel = 0.0              # ... nor one below this fraction of the baseline's score (scores that are distances, not logarithms)
    objective_note = ""             # how the score and the constraints are defined (shown in the Experiments tab)
    display_ready = False

    def describe(self) -> dict:
        return {"name": self.name, "label": self.label, "description": self.description, "params": self.params,
                "metrics": self.metrics, "options": self.options, "default_objective": self.default_objective or self.score,
                "score": self.score, "min_gain": self.min_gain, "min_gain_rel": self.min_gain_rel, "objective_note": self.objective_note,
                "constraints": {k: list(v) for k, v in self.static_limits().items()}, "sampler": getattr(self, "ht_sampler", "tpe")}

    def defaults(self) -> dict:
        return {p["name"]: p["default"] for p in self.params}

    def static_limits(self) -> dict:
        return {}

    def canonical(self, active: dict, ctx: dict) -> dict:
        """The settings that decide what a trial runs (``active``: the sampled ones): settings that run the same thing map to one key, so the study
        evaluates them once (lab.run's duplicate detection)."""
        return active

    def limits(self, ctx: dict) -> dict:
        """The constraints of a trial ({metric: ("max" | "min", bound)}); may depend on the study's reference (``ctx``)."""
        return self.static_limits()

    def prepare(self, ds: Dataset, opts: dict, device, log, cancel, study_dir: str) -> dict:
        raise NotImplementedError

    def run(self, ctx: dict, p: dict, log, cancel) -> tuple[dict, np.ndarray | None]:
        raise NotImplementedError


TRUTH_METRICS = {
    "truth_nrmse": {"label": "Truth: normalised rms error", "direction": "minimize", "truth": True},
    "truth_faint_nrmse": {"label": "Truth: error on faint / extended emission", "direction": "minimize", "truth": True},
    "truth_ssim": {"label": "Truth: SSIM (stretched)", "direction": "maximize", "truth": True},
    "truth_psnr": {"label": "Truth: PSNR (dB)", "direction": "maximize", "truth": True},
    "truth_star_dmag_mad": {"label": "Truth: star photometry scatter (mag)", "direction": "minimize", "truth": True},
    "truth_star_dmag_abs": {"label": "Truth: |star photometry bias| (mag)", "direction": "minimize", "truth": True},
}

DEPTH_METRICS = {
    "depth": {"label": "Depth: (ln S/N point + ln S/N extended) / 2 of the reference stars, empty-aperture noise", "direction": "maximize"},
    "snr_point": {"label": "Median S/N of the reference stars, 0.68 FWHM aperture", "direction": "maximize"},
    "snr_total": {"label": "Median S/N of the reference stars, 2.5 FWHM aperture", "direction": "maximize"},
    "fwhm": {"label": "Star FWHM (px of the reference grid)", "direction": "minimize"},
}


def _truth_row(m: dict) -> dict:
    out = {f"truth_{k}": v for k, v in m.items() if k in ("nrmse", "faint_nrmse", "ssim", "psnr", "star_dmag_mad")}
    if "star_dmag_median" in m:
        out["truth_star_dmag_abs"] = abs(m["star_dmag_median"])
        out["truth_star_dmag_median"] = m["star_dmag_median"]
    return out


def _truth_scores(ds: Dataset, s, result: np.ndarray, sigma_eval: float, stars_in_result: bool = True, fit_plane: bool = True) -> dict:
    """Truth metrics of a (3, H, W) linear result on the stack's grid."""
    try:
        tr, valid, _ = truth_on_stack(ds, s, result, extended=True, stars=stars_in_result)
        ts, _, _ = truth_on_stack(ds, s, result, extended=False, stars=True)
        est = _hwc(result)
        unsat = cv2.erode((est.max(-1) < 0.98).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        m = MX.truth_metrics(est, tr, valid & unsat, star_truth=ts if stars_in_result else None, sigma_eval=sigma_eval, fit_plane=fit_plane)
        return _truth_row(m)
    except Exception as e:                                   # noqa: BLE001
        return {"truth_error": f"{type(e).__name__}: {e}"}


def _processing_params(keys: list[str], tune: set[str], hypertune: dict | None = None, when: dict | None = None) -> list[dict]:
    """Task parameters taken from the UI's PARAM_SPEC so that ranges and labels follow the pipeline.  ``hypertune``: {key: True | condition},
    ``when``: {key: condition} (see the module docstring)."""
    from ..post import DEFAULTS, EXTRA_DEFAULTS, PARAM_SPEC
    out = []
    spec = {s["key"]: s for s in PARAM_SPEC}
    hypertune, when = hypertune or {}, when or {}
    for k in keys:
        sp = spec.get(k)
        d = DEFAULTS.get(k, EXTRA_DEFAULTS.get(k))
        if sp is None:
            continue
        p = {"name": k, "label": sp["label"], "default": d, "tune": k in tune, "processing": k}
        if sp["type"] == "range":
            p.update(type="int" if isinstance(sp.get("step"), int) and isinstance(d, int) else "float", low=sp["min"], high=sp["max"])
        elif sp["type"] == "bool":
            p.update(type="bool")
        else:
            p.update(type="categorical", choices=list(sp["options"]))
        if k in hypertune:
            p["hypertune"] = hypertune[k]
        if k in when:
            p["when"] = when[k]
        out.append(p)
    return out


GATE_NAMES = ("sky_median", "clipped", "white", "sky_chroma", "sky_chroma_quadrants", "sky_band_structure", "sky_pixel_noise", "stars_count",
              "stars_round", "stars_coloured")


def _gate_violations(g: dict) -> dict:
    """Per gate, how far outside its limit the picture is (relative units; 0 = passes)."""
    out = {}
    for k in GATE_NAMES:
        v = g.get(k)
        out[k] = _gate_distance({k: v}) if isinstance(v, dict) else 0.0
    return out


def _gate_distance(g: dict) -> float:
    """How far outside their limits the failed gates are (relative units; 0 = every gate passes)."""
    dist = 0.0
    for k, v in g.items():
        if not isinstance(v, dict) or v.get("pass") or v.get("value") is None:
            continue
        lim, val = v.get("limit"), v.get("value")
        if isinstance(val, dict):                             # stars_count -> its deviation; sky_chroma -> max(|a|, |b|)
            val = val.get("deviation") if "deviation" in val else (max(x for x in val.values() if isinstance(x, (int, float))) if any(isinstance(x, (int, float)) for x in val.values()) else None)
        if val is None:
            dist += 1.0
            continue
        if isinstance(lim, (list, tuple)):
            lo, hi = lim
            dist += (lo - val) / lo if val < lo else (val - hi) / hi
        elif isinstance(lim, (int, float)) and lim:
            dist += abs(val - lim) / lim
    return float(dist)


# ----------------------------------------------------------------------------- registration & integration
class StackTask(Task):
    name = "stack"
    label = "Registration & integration"
    description = ("Re-stacks the dataset with each trial's settings (Siril; the registration is reused when its settings are unchanged): rejection algorithm and "
                   "sigmas, frame weighting, frame selection sensitivity, resampling (Bayer drizzle / demosaic), drizzle drop size and kernel, banding correction.  "
                   "Scored by the depth of the stack on the reference stars of the first trial (mapped onto each stack by astroalign, so another crop or drizzle "
                   "scale compares directly); synthetic data also against the truth seen through the subs' mean PSF.")
    objective_note = ("Score: stack_score - held-out validation.  The registered frames go alternately (in time) to two halves of the same statistics; "
                      "the held-out half is stacked once with the baseline's settings, each trial stacks the other half with its own (selection, weighting, "
                      "rejection, resampling).  The trial's stack, matched to the held-out stack's PSF by a kernel fitted per tile (difference imaging: Alard "
                      "& Lupton 1998, Bramich 2008), is compared with it in noise-weighted chi2 (noise variance linear in the level, detected stars masked, "
                      "both high-passed at 8 FWHM): the halves' noise is independent, so the chi2 measures the trial stack's error against the true sky at "
                      "the held-out resolution - more frames and a better weighting lower it, artefacts a lax rejection lets through and the blur of poor "
                      "frames raise it.  stack_score = 0.5 ln(chi2 baseline / chi2).  Depth (matched-aperture S/N, Labbe et al. 2003) is reported.  "
                      "Without a frame list the whole stack's depth is the score.  Constraints against the baseline: star ellipticity at most 0.02 worse (registration / frame selection), "
                      "single-pixel outliers (hot pixels, cosmic rays a lax rejection lets through) at most 1.5 x (or +2 per Mpx), trail pixels at most 1.25 x "
                      "(or +20 per million: on M 31 the search's best trials raised the high rejection sigma to ~7, which lets trails through on a night that has "
                      "them), and the auto-crop keeping "
                      "at least 90 % of the baseline's field (a synthetic stack without banding correction came out cropped to a third of it).")
    params = [
        {"name": "rejection", "label": "Rejection algorithm (auto: winsorized, MAD for drizzle > 1x)", "type": "categorical",
         "choices": ["auto", "winsorized", "sigma", "mad", "linear", "median", "none"], "ht_choices": ["auto", "winsorized", "sigma", "mad", "linear"],
         "default": "auto", "tune": False, "hypertune": True, "pipeline": "rejection"},
        {"name": "sigma_low", "label": "Rejection σ low", "type": "float", "low": 1.5, "high": 8.0, "default": 4.0, "tune": True, "hypertune": True, "pipeline": "sigma_low",
         "when": {"rejection": ["auto", "winsorized", "sigma", "mad", "linear"]}},
        {"name": "sigma_high", "label": "Rejection σ high", "type": "float", "low": 1.5, "high": 8.0, "default": 3.0, "tune": True, "hypertune": True, "pipeline": "sigma_high",
         "when": {"rejection": ["auto", "winsorized", "sigma", "mad", "linear"]}},
        {"name": "stack_weighting", "label": "Frame weighting (stack -weight)", "type": "categorical", "choices": ["wfwhm", "nbstars", "noise", "nbstack", "none"],
         "default": "wfwhm", "tune": False, "hypertune": True, "pipeline": "stack_weighting"},
        {"name": "sensitivity", "label": "Frame rejection sensitivity", "type": "float", "low": 0.5, "high": 2.0, "default": 1.0, "tune": False, "hypertune": True, "pipeline": "sensitivity"},
        {"name": "mode", "label": "Resampling", "type": "categorical", "choices": ["auto", "drizzle", "demosaic"], "ht_choices": ["auto", "demosaic"],
         "default": "auto", "tune": False, "hypertune": True, "pipeline": "mode"},
        {"name": "drizzle_pixfrac", "label": "Drizzle drop size (pixfrac; 0 = 1 / scale)", "type": "float", "low": 0.5, "high": 1.0, "default": 1.0, "tune": False,
         "hypertune": True, "pipeline": "drizzle_pixfrac", "when": {"mode": ["auto", "drizzle"]}},
        {"name": "drizzle_kernel", "label": "Drizzle kernel", "type": "categorical", "choices": ["square", "point", "turbo", "gaussian", "lanczos2", "lanczos3"],
         "ht_choices": ["square", "turbo", "gaussian"], "default": "square", "tune": False, "hypertune": True, "pipeline": "drizzle_kernel", "when": {"mode": ["auto", "drizzle"]}},
        {"name": "scale", "label": "Drizzle scale", "type": "categorical", "choices": [1.0, 1.5, 2.0], "default": 1.0, "tune": False, "pipeline": "scale"},
        {"name": "pattern_correction", "label": "Row banding correction (seqfixbanding)", "type": "bool", "default": True, "tune": False, "hypertune": True, "pipeline": "pattern_correction"},
        {"name": "crop_threshold", "label": "Crop: minimum frame coverage", "type": "float", "low": 0.1, "high": 1.0, "default": 0.3, "tune": False, "pipeline": "crop_threshold"},
    ]
    metrics = {
        "stack_score": {"label": "Stack score: fidelity against the held-out frames, 0.5 ln(chi2 baseline / chi2)", "direction": "maximize"},
        "heldout_chi2": {"label": "Held-out mean chi2 (1 = noise only): the held-out half against the trial's stack matched to its PSF", "direction": "minimize"},
        **DEPTH_METRICS,
        "ellipticity": {"label": "Star ellipticity (median 1 - b / a)", "direction": "minimize"},
        "outliers": {"label": "Single-pixel outliers per Mpx (hot pixels / cosmic rays left)", "direction": "minimize"},
        "dm_mad": {"label": "Photometric scatter against the baseline stack (mag)", "direction": "minimize"},
        "bg_noise": {"label": "Background noise (centre crop)", "direction": "minimize"},
        "roundness": {"label": "Star ellipticity (centre crop, gates)", "direction": "minimize"},
        "n_frames": {"label": "Frames stacked", "direction": "maximize"},
        "field_fraction": {"label": "Field kept by the auto-crop (fraction of the baseline's)", "direction": "maximize"},
        "trails": {"label": "Satellite / aircraft trail pixels per million left in the stack", "direction": "minimize"},
        "seconds": {"label": "Run time (s)", "direction": "minimize"},
        **TRUTH_METRICS,
    }
    options = [{"name": "crop", "label": "Preview / crop metrics (px per side)", "type": "int", "default": 1024},
               {"name": "sigma_eval", "label": "Truth comparison resolution σ (px)", "type": "float", "default": 1.0},
               {"name": "in_place", "label": "Stack the session itself (hyper-tune; 0 = the study's own folder)", "type": "int", "default": 0, "internal": True}]
    score = default_objective = "stack_score"
    min_gain = 0.002              # Siril stacking is deterministic (four identical stacks: the same depth to 1e-15), the held-out fit too (fixed pixels);
                                  # both halves' sky noise dilute the held-out score: on M 31 eight stacking settings spread over -0.012 .. +0.001

    def limits(self, ctx):
        b = ctx.get("base") or {}
        out = {"n_frames": ("min", 3), "field_fraction": ("min", 0.9)}
        if b.get("ellipticity") is not None:
            out["ellipticity"] = ("max", b["ellipticity"] + 0.02)
        if b.get("outliers") is not None:
            out["outliers"] = ("max", max(1.5 * b["outliers"], b["outliers"] + 2.0))
        if b.get("trails") is not None:                      # a lax rejection (high sigma near 7 kept depth rising on M 31) lets trails through
            out["trails"] = ("max", max(1.25 * b["trails"], b["trails"] + 20.0))
        return out

    def prepare(self, ds, opts, device, log, cancel, study_dir):
        from ..session import Session
        if int(opts.get("in_place", 0) or 0) and ds.given is not None:
            s = ds.given
            log("Stacking the session itself (hyper-tune): the registration of its analysis is reused")
        else:
            base = ds.ensure_stacked(log, cancel) if ds.synthetic else ds.session()
            wd = os.path.join(study_dir, "work", "stack")
            s = Session(ds.folder, wd)
            s.cancel_flag = cancel
            for n in ("calibration.json", "calib_prefs.json", "overrides.json"):
                if os.path.exists(base._p(n)):
                    os.makedirs(s.dir, exist_ok=True)
                    shutil.copy(base._p(n), s._p(n))
            log("Calibrating and registering once (the trials reuse the registration unless its settings change)")
            s.run_analysis(1.0, _prog(log, 20), stack_params={})
        saved = _store(study_dir, "reference.pkl") or {}
        ctx = {"ds": ds, "session": s, "opts": opts, "device": device, "study_dir": study_dir, "yard": saved.get("yard"), "base": saved.get("base")}
        self._held_out(ctx, log)
        return ctx

    def _held_out(self, ctx, log):
        """Held-out validation of the stack: the registered frames, in time order, alternately to two halves of the same statistics; the held-out
        half is stacked once with the baseline's settings (the user's), the trials stack the other half with theirs (Session.stack_subsets) -
        frame selection, weighting, rejection and resampling all act within it.  Without a frame list (or < 6 frames) the trials stack the whole
        session and are scored on depth (logged)."""
        s = ctx["session"]
        try:
            rows = s._load_rows()
            reg = sorted((i for i, r in enumerate(rows) if r.get("registered")), key=lambda i: (rows[i].get("time") or "", i))
            if len(reg) < 6:
                raise RuntimeError(f"{len(reg)} registered frames")
            held, trial = reg[0::2], reg[1::2]
            base_sp = dict(ctx["opts"].get("stack_params") or {})
            H = s.stack_subsets({"held": held}, stack_params=base_sp, place=False)["held"]
            v = OB.valid_region(OB.lum(_hwc(H)))
            ys, xs = np.nonzero(v)
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
            my, mx = (y1 - y0) // 20, (x1 - x0) // 20                         # the rim, where few frames overlap, off
            ctx["held_box"] = (y0 + my, y1 - my, x0 + mx, x1 - mx)
            ctx["held"] = np.ascontiguousarray(H[:, y0 + my:y1 - my, x0 + mx:x1 - mx])
            ctx["trial_frames"] = trial
            log(f"Held out: {len(held)} of {len(reg)} frames (every second one in time), stacked with the baseline's settings; the trials stack the other "
                f"{len(trial)} with theirs")
        except Exception as e:                         # noqa: BLE001
            ctx["held"] = None
            log(f"No held-out frames ({type(e).__name__}: {e}): the trials stack the session and are scored on depth")

    def run(self, ctx, p, log, cancel):
        from .. import gates
        s, ds = ctx["session"], ctx["ds"]
        sp = {**(ctx["opts"].get("stack_params") or {}),
              **{k: p[k] for k in ("rejection", "sigma_low", "sigma_high", "stack_weighting", "sensitivity", "mode", "drizzle_pixfrac", "drizzle_kernel",
                                   "scale", "pattern_correction", "crop_threshold") if k in p}}
        t = time.time()
        from .. import post
        if ctx.get("held") is not None:
            from ..session import place_on
            T = s.stack_subsets({"trial": ctx["trial_frames"]}, _prog(log, 20), stack_params=sp, place=False)["trial"]
            nf = int((getattr(s, "last_subsets", {}).get("trial") or {}).get("frames") or 0)
            T_on, _ = place_on(ctx["held"], T)
            meta = {"n_frames": nf, "scale": 1.0, "resampling_reason": None, "rejection": None}
            st = T_on
        else:
            meta = s.run_stack(sp, _prog(log, 20))
            st = _read(s._p("stack.fits"))
        dt = time.time() - t
        st, _ = post.normalise_scale(st)
        est = _hwc(st)
        if ctx["yard"] is None:                                 # the first trial (the baseline) is the reference of the paired measurements
            ctx["yard"] = OB.Yardstick(est, keep_image=False)
            log(f"Reference: {len(ctx['yard'].sx)} stars, FWHM {ctx['yard'].fwhm:.2f} px, {len(ctx['yard'].ax)} sky apertures")
        y = ctx["yard"]
        scale = float(meta.get("scale") or 1.0) / float((ctx.get("base") or {}).get("scale") or float(meta.get("scale") or 1.0))
        field = est.shape[0] * est.shape[1] / (y.shape[0] * y.shape[1] * scale * scale)
        if ctx.get("held") is not None:                          # on the held-out grid: the share of it the trial's frames cover, against the baseline's
            vf = float(OB.valid_region(OB.lum(est)).mean())
            ctx.setdefault("vf0", vf)
            field = vf / max(ctx["vf0"], 1e-9)
        try:
            m = y.measure(est)
        except Exception as e:                                   # noqa: BLE001  the reference stars cannot be mapped onto it (a stack cropped to part of the field)
            m = dict(OB.Yardstick(est, keep_image=False).ref)
            m["alignment"] = f"unpaired (its own stars): {type(e).__name__}: {e}"
            log(f"Reference stars not mapped onto this stack ({e}); depth measured on its own stars")
        m.update(OB.Yardstick.shape_and_outliers(est, y.fwhm))
        x = _centre_crop(est, int(ctx["opts"].get("crop", 1024)))
        lum = x.mean(-1)
        noise = float(1.4826 * np.median(np.abs(lum - np.median(lum))))
        ms = gates.measure_stars(np.moveaxis(x, -1, 0), noise * 5.0) if noise > 0 else {"ellipticity": None}
        out = {k: m.get(k) for k in ("depth", "snr_point", "snr_total", "fwhm", "ellipticity", "outliers", "dm_mad", "n_stars", "alignment")}
        out.update(bg_noise=MX.background_noise(x), roundness=ms.get("ellipticity"), n_frames=meta["n_frames"], seconds=dt, field_fraction=float(field),
                   trails=trail_pixels(est),
                   scale=float(meta.get("scale") or 1.0),
                   resampling=meta.get("resampling_reason"), rejection=(meta.get("rejection") or {}).get("type"))
        if ctx.get("held") is not None:
            if ctx.get("heldout") is None:                     # the baseline trial: the noise model and the unprocessed chi2 of the held-out comparison
                ctx["heldout"] = OB.HeldOut(_hwc(ctx["held"]), est, OB.Yardstick(_hwc(ctx["held"]), keep_image=False).fwhm)
                h = ctx["heldout"]
                log(f"Held-out reference: {h.n_stars} stars masked, noise variance {h.nlf[0]:.3g} + {h.nlf[1]:.3g} x level; baseline chi2 {h.chi2_r0:.4f}")
            h = ctx["heldout"]
            out["heldout_chi2"] = h.chi2(est)
            out["heldout_tiles"] = h.last_tiles
            out["stack_score"] = 0.5 * math.log(h.chi2_r0 / max(out["heldout_chi2"], 1e-12))
        elif out.get("depth") is not None:
            out["stack_score"] = out["depth"]
        if ctx["base"] is None:
            ctx["base"] = dict(out)
            _store(ctx["study_dir"], "reference.pkl", {"yard": y, "base": ctx["base"]})
        if ds.synthetic:
            out.update(_truth_scores(ds, s, st, float(ctx["opts"].get("sigma_eval", 1.0))))
        return out, x


# ----------------------------------------------------------------------------- satellite trails (Cosmic Clarity)
def trail_pixels(img: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Pixels on straight trails, per million: the luminance high-passed at 16 px (galaxies, nebulae, gradients out), the star-like detections masked -
    round ones only: a trail is detected as a source too, an elongated one, and stays - smoothed at 1.5 px and thresholded at 3 sigma of the smoothed
    noise; the straight segments longer than a tenth of the frame found by the probabilistic Hough transform (Matas et al. 2000; OpenCV).
    ``mask``: further pixels to leave out."""
    import sep as _sep
    L = OB.lum(img)
    valid = OB.valid_region(L)
    # high-pass: sep's sigma-clipped background on a 16 px mesh follows galaxies and nebulae but not a few-pixel-wide trail crossing a mesh
    bk = _sep.Background(L, mask=~valid, bw=16, bh=16, fw=3, fh=3)
    sub = np.ascontiguousarray(L - bk.back(), np.float32)
    rms = max(bk.globalrms, 1e-12)
    star = np.zeros(L.shape, np.uint8)
    try:
        objs = _sep.extract(sub, 5.0, err=rms, mask=~valid, minarea=5)
        for x_, y_, a_, b_ in zip(objs["x"], objs["y"], objs["a"], objs["b"]):
            if b_ >= 0.3 * a_ and a_ < 0.02 * max(L.shape):                  # round and compact: a star (a trail is long and thin)
                cv2.circle(star, (int(round(x_)), int(round(y_))), int(round(3 * a_ + 3)), 1, -1)
    except Exception:                                                    # noqa: BLE001
        pass
    z = sub / rms
    z[(star > 0) | ~valid] = 0.0
    if mask is not None:
        z[mask] = 0.0
    sm = cv2.GaussianBlur(z.astype(np.float32), (0, 0), 1.5)
    s0 = OB.nmad(sm[valid & (star == 0)][::17]) or 1.0
    binary = ((sm / s0) > 3.0).astype(np.uint8)
    h, w = binary.shape
    f = 2 if max(h, w) > 2000 else 1
    if f > 1:
        binary = cv2.resize(binary, (w // f, h // f), interpolation=cv2.INTER_NEAREST)
    lines = cv2.HoughLinesP(binary * 255, 1, np.pi / 360, threshold=max(60, int(0.05 * min(h, w) / f)),
                            minLineLength=int(0.1 * min(h, w) / f), maxLineGap=8)
    if lines is None:
        return 0.0
    canvas = np.zeros(binary.shape, np.uint8)
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):        # (N, 1, 4) in OpenCV 4, (N, 4) in 5
        cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), 1, 3)
    return float((canvas & binary).sum() * f * f / max(valid.sum(), 1) * 1e6)


class SatelliteTask(Task):
    name = "satellite"
    label = "Satellite trails (Cosmic Clarity)"
    description = ("Cosmic Clarity satellite-trail removal (siril-scripts CosmicClarity_Native.py, the script's defaults) on the dataset's stack, as the restore stage "
                   "runs it, against the stack as it is.  Scored by the straight trails left (Hough transform on the source-masked sky) and the change it makes "
                   "away from them (it must leave everything else alone).")
    objective_note = ("Score: trail pixels per million left in the stack (probabilistic Hough on the 3-sigma sky).  Constraint: the change at scales above the "
                      "PSF, away from the stars and the trails, at most 0.5 x the noise there.  With no trail in the stack, removal changes nothing and the "
                      "baseline (off) stays.")
    params = [{"name": "satellite_removal", "label": "Cosmic Clarity satellite-trail removal", "type": "bool", "default": False, "tune": True, "hypertune": True,
               "pipeline": "satellite_removal"}]
    metrics = {"trails": {"label": "Trail pixels per million", "direction": "minimize"},
               "trails_detected": {"label": "Cosmic Clarity detected trails (1 = yes)", "direction": "minimize"},
               "large_change": {"label": "Change above the PSF scale away from stars and trails (× noise)", "direction": "minimize"},
               "seconds": {"label": "Run time (s)", "direction": "minimize"}}
    score = default_objective = "trails"
    min_gain = 1.0

    def static_limits(self):
        return {"large_change": ("max", 0.5)}

    def prepare(self, ds, opts, device, log, cancel, study_dir):
        from .. import post
        s = ds.ensure_stacked(log, cancel)
        st, _ = post.normalise_scale(_read(s._p("stack.fits")))
        img = _hwc(st)
        y = OB.Yardstick(img)
        t0 = trail_pixels(img)
        log(f"Stack: {t0:.0f} trail pixels per million")
        return {"ds": ds, "session": s, "stack": st, "img": img, "yard": y, "trails0": t0, "device": device, "study_dir": study_dir, "n": 0}

    def run(self, ctx, p, log, cancel):
        from .. import tools
        y = ctx["yard"]
        if not p.get("satellite_removal"):
            return {"trails": ctx["trails0"], "trails_detected": 0, "large_change": 0.0, "seconds": 0.0}, _centre_crop(ctx["img"])
        ctx["n"] += 1
        work = os.path.join(ctx["study_dir"], "work", f"sat_{ctx['n']}")
        t = time.time()
        try:
            out, meta = tools.cosmic_clarity.run("satellite", ctx["stack"], ["--sat-mode", "full", "--sensitivity", "0.1"], work=work, device=str(ctx["device"]),
                                                 checkpoint=_checkpoint(cancel))
        finally:
            shutil.rmtree(work, ignore_errors=True)
        img = _hwc(out)
        tr = trail_pixels(img)
        # the change away from the trails: the trails found on the original are masked too
        m = y.measure(img, same_grid=True, normalise_gain=False)
        return {"trails": tr, "trails_detected": int(bool(meta.get("trails_detected"))), "large_change": m.get("large_change"), "seconds": time.time() - t}, _centre_crop(img)


# ----------------------------------------------------------------------------- gradient removal
def _sky_yardstick(img: np.ndarray, tile: int = 96) -> dict:
    """Tiles of the (h, w, 3) image: star-masked medians, which tiles are sky (robust plane through the faintest) and which hold emission."""
    import sep as _sep
    L = np.ascontiguousarray(img.mean(-1), np.float32)
    bk = _sep.Background(L, bw=64, bh=64)
    objs = _sep.extract(L - bk.back(), 5.0, err=bk.globalrms, minarea=5)
    sm = np.zeros(L.shape, np.uint8)
    for x_, y_, a_ in zip(objs["x"], objs["y"], objs["a"]):
        cv2.circle(sm, (int(round(x_)), int(round(y_))), int(round(3 * a_ + 3)), 1, -1)
    h, w = L.shape
    ys, xs = np.arange(0, h - tile + 1, tile), np.arange(0, w - tile + 1, tile)
    tiles, meds, cx, cy = [], [], [], []
    for y in ys:
        for x in xs:
            win = L[y:y + tile, x:x + tile]
            m = sm[y:y + tile, x:x + tile] == 0
            if m.mean() < 0.4:
                continue
            tiles.append((y, x))
            meds.append(float(np.median(win[m])))
            cy.append(y + tile / 2)
            cx.append(x + tile / 2)
    meds = np.asarray(meds)
    cx, cy = np.asarray(cx) / w, np.asarray(cy) / h
    noise = float(1.4826 * np.median(np.abs((L - cv2.GaussianBlur(L, (0, 0), 1.5))[::3, ::3])))
    # a robust plane through the faintest tiles decides sky vs emission
    sel = meds <= np.percentile(meds, 40)
    A = np.stack([np.ones(sel.sum()), cx[sel], cy[sel]], 1)
    coef, *_ = np.linalg.lstsq(A, meds[sel], rcond=None)
    plane = coef[0] + coef[1] * cx + coef[2] * cy
    sky = meds <= plane + 2.0 * noise / np.sqrt(tile * tile * 0.5)
    emission = meds >= plane + 8.0 * noise / np.sqrt(tile * tile * 0.5)
    return {"tiles": tiles, "cx": cx, "cy": cy, "sky": sky, "emission": emission, "noise": noise, "tile": tile, "mask": sm, "med0": meds}


def _tile_medians(img: np.ndarray, yard: dict) -> np.ndarray:
    L = np.ascontiguousarray(img.mean(-1), np.float32)
    t = yard["tile"]
    out = []
    for y, x in yard["tiles"]:
        win = L[y:y + t, x:x + t]
        m = yard["mask"][y:y + t, x:x + t] == 0
        out.append(float(np.median(win[m])))
    return np.asarray(out)


def model_structure(before: np.ndarray, after: np.ndarray, valid: np.ndarray) -> float:
    """How much of the image's structure at 1 - 10 % of the frame the gradient model took away: the std of the removed component's band-pass
    (difference of Gaussians at 1 % and 10 % of the long side) over that of the image's own.  A gradient (sky glow, vignetting, amp glow) is smooth
    at those scales - the scale separation every background extraction rests on (DBE / ABE, GraXpert) - and nebulae, dust and galaxy halos are not."""
    L0, L1 = OB.lum(before), OB.lum(after)
    h, w = L0.shape
    f = max(1, int(round(max(h, w) / 800)))
    sz = (w // f, h // f)
    a, b = cv2.resize(L0, sz, interpolation=cv2.INTER_AREA), cv2.resize(L1, sz, interpolation=cv2.INTER_AREA)
    v = cv2.resize(valid.astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST) > 0
    s1, s2 = 0.01 * max(sz), 0.10 * max(sz)
    yy, xx = np.mgrid[0:sz[1], 0:sz[0]].astype(np.float32)
    xx, yy = xx / sz[0] - 0.5, yy / sz[1] - 0.5
    A = np.stack([np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy], -1)

    def band(x):
        # a quadratic surface (the least any gradient model holds) off first: the blur's mirrored borders would turn a ramp into band-pass structure
        coef, *_ = np.linalg.lstsq(A[v], x[v], rcond=None)
        r = (x - A @ coef).astype(np.float32)
        return cv2.GaussianBlur(r, (0, 0), s1) - cv2.GaussianBlur(r, (0, 0), s2)
    d = a - b
    return float(np.std(band(d)[v]) / max(np.std(band(a)[v]), 1e-12))


def gradient_context(s, study_dir: str, device, log, cancel, src_dir: str | None = None) -> dict:
    """What the gradient model is judged on, set once on the stack as the restore stage loads it: the scoring tiles (sky / emission), the valid
    region, the stars' photometry (the photometric scale) and whether the sky-survey reference covers the field (then auto = the survey model)."""
    from .. import post
    # the stack as the restore stage loads it (post.Restorer: row order, _sanitize - an HDR stack peaking at 2.13 is divided by it - and
    # normalise_scale); comparing the file's own values with the restored ones scored that rescale as removed signal
    wd = os.path.join(study_dir, "work", "bg_ref")
    _copy_products(src_dir or s.dir, wd, ("stack.fits", "stack_meta.json"))
    try:
        stack = post.Restorer(wd, s.meta, {}, dict(post.DEFAULTS), None, None).arr.copy()
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    img = _hwc(stack)
    yard = _sky_yardstick(img)
    log(f"{len(yard['tiles'])} gradient scoring tiles: {int(yard['sky'].sum())} sky, {int(yard['emission'].sum())} emission")
    # does the sky-survey reference apply to this field (plate-solved, covered)?  Then "auto" is the survey model, which ignores the smoothing and the
    # correction - those trials are one setting (canonical); on M 31 37 of 60 trials had been auto with differing, unused smoothing values
    skyref_ok = False
    wd = os.path.join(study_dir, "work", "bg_probe")
    _copy_products(src_dir or s.dir, wd, ("stack.fits", "stack_meta.json", "skyref.npz"))
    try:
        r = post.Restorer(wd, s.meta, {"device": str(device)}, {**post.DEFAULTS, "bg_method": "reference"}, None, _checkpoint(cancel))
        r.background()
        # the survey's own record: when it fails, background() falls back to GraXpert, whose record comes last (a synthetic field without a plate
        # solution had read as covered, so GraXpert trials of different smoothings counted as one setting)
        skyref_ok = any(x["step"] == "gradient" and str(x.get("tool", "")).startswith("sky survey") and x.get("status") in (None, "done")
                        for x in r.steps.records)
    except Exception as e:                     # noqa: BLE001
        log(f"Sky-survey reference probe: {e}")
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    log("The sky-survey reference covers this field: auto = the survey model" if skyref_ok else "No sky-survey reference for this field: auto = GraXpert")
    try:                                       # the photometric scale: the stars' local-background photometry, which no gradient model changes
        stars = OB.Yardstick(img, keep_image=False)
    except Exception as e:                     # noqa: BLE001
        stars = None
        log(f"No star photometry for the photometric scale ({e}): results are compared as they are")
    return {"stack": stack, "img": img, "yard": yard, "valid": OB.valid_region(OB.lum(img)), "stars": stars, "skyref": skyref_ok}


def gradient_metrics(g: dict, before: np.ndarray, res: np.ndarray) -> dict:
    """The gradient model's effect, ``before`` -> ``res`` ((h, w, 3), the image before and after the model) on the tiles of ``gradient_context``:
    emission_change (|fraction of the emission tiles' signal above the sky the model changed|), sky_flatness (the gradient left in the sky tiles, a
    robust quadratic's range in pixel noise) and model_structure (what it removed at 1 - 10 % of the frame)."""
    yard = g["yard"]
    # a global rescale is no change of the emission: post._sanitize divides the result by its peak when the gradient model's pedestal lifts
    # a saturated core above 1 (M 31: x 0.89, which had scored every model as removing 11 % of the galaxy); the stars' fluxes give the scale
    gain = 1.0
    if g.get("stars") is not None:
        dm = g["stars"].measure(res, same_grid=True, normalise_gain=False).get("dm_median")
        if dm is not None and np.isfinite(dm):
            gain = float(10 ** (-0.4 * dm))
    res_n = res / np.float32(gain)
    T1 = _tile_medians(res_n, yard)
    sky = yard["sky"]
    cx, cy = yard["cx"][sky], yard["cy"][sky]
    A = np.stack([np.ones(sky.sum()), cx, cy, cx * cx, cx * cy, cy * cy], 1)
    coef, *_ = np.linalg.lstsq(A, T1[sky], rcond=None)
    out = {"sky_flatness": float(np.ptp(A @ coef) / max(yard["noise"], 1e-12)), "model_structure": model_structure(before, res_n, g["valid"]),
           "bg_gain": gain, "signal_removed": 0.0}                   # no emission tiles: nothing a model could take away
    em = yard["emission"]
    if em.any() and sky.any():
        b0 = yard["med0"][em] - np.median(yard["med0"][sky])
        b1 = T1[em] - np.median(T1[sky])
        out["signal_removed"] = float(np.clip(1.0 - b1.sum() / max(b0.sum(), 1e-12), -1.0, 1.0))
    out["emission_change"] = abs(out["signal_removed"])
    return out


class BackgroundTask(Task):
    name = "background"
    label = "Gradient removal"
    description = ("The restore stage's gradient model on the stack: the NSNS sky survey reference (when plate-solved and covered), GraXpert's AI background extraction with its "
                   "smoothing, or Siril subsky (polynomial degree / RBF smoothing), subtraction or division.  Real data: the emission the model changed (the signal of "
                   "the emission tiles above the sky), the gradient left in the sky (range of a robust quadratic over the star-free sky tiles, × the pixel noise) and "
                   "the structure the model took away at 1-10 % of the frame (a gradient has none there).  "
                   "Sky and emission tiles are set once against a robust plane through the faintest tiles.  Synthetic data: against the truth (constant offset fitted).")
    objective_note = ("Score: emission_change = |signal_removed|, the fraction of the emission tiles' signal above the sky that the model changed - "
                      "the gradient model that leaves the emission most intact.  Constraints: the gradient left in the sky tiles at most 3 × the pixel noise "
                      "(sky_flatness: the model must remove the gradient) and the model's own structure at 1-10 % of the frame at most 15 % of the "
                      "image's there.  Validated against the truth of a synthetic sky (12 models: Spearman rho +0.99 between emission_change and the "
                      "error against the truth, -0.19 for sky_flatness): the 'sky' tiles of a nebula field hold faint nebula, so the flattest sky is "
                      "the one whose model ate it - on Sh2-131 GraXpert left the flattest sky (0.14 × noise against 0.77 for the sky-survey "
                      "reference) by taking 43 % of the nebula's signal.  Sky flatness ranks no model; it only rejects one that leaves the gradient.")
    params = _processing_params(["bg_method", "bg_correction", "bg_degree", "bg_smoothing"], {"bg_method", "bg_smoothing", "bg_degree"},
                                hypertune={"bg_method": True, "bg_correction": True, "bg_degree": True, "bg_smoothing": True},
                                when={"bg_degree": {"bg_method": ["poly"]}, "bg_smoothing": {"bg_method": ["auto", "graxpert", "rbf"]},
                                      "bg_correction": {"bg_method": ["auto", "graxpert", "poly", "rbf"]}})
    # reference = auto without its fallback: the same model wherever auto uses the survey
    next(q for q in params if q["name"] == "bg_method")["ht_choices"] = ["auto", "graxpert", "poly", "rbf"]
    metrics = {
        "emission_change": {"label": "Emission changed by the model (|fraction of the emission tiles' signal|)", "direction": "minimize"},
        "sky_flatness": {"label": "Gradient left in the sky (× pixel noise)", "direction": "minimize"},
        "model_structure": {"label": "Structure the model removed at 1-10 % of the frame (fraction)", "direction": "minimize"},
        "signal_removed": {"label": "Extended emission removed (fraction, emission tiles)", "direction": "minimize"},
        "bg_gain": {"label": "Global scale of the result (stars' photometry; judged after it)", "direction": "minimize"},
        "seconds": {"label": "Run time (s)", "direction": "minimize"},
        **{k: v for k, v in TRUTH_METRICS.items() if k in ("truth_nrmse", "truth_faint_nrmse", "truth_ssim", "truth_psnr")},
    }
    options = [{"name": "sigma_eval", "label": "Truth comparison resolution σ (px)", "type": "float", "default": 4.0}]
    score = default_objective = "emission_change"
    min_gain = 0.02

    def static_limits(self):
        return {"model_structure": ("max", 0.15), "sky_flatness": ("max", 3.0)}

    def prepare(self, ds, opts, device, log, cancel, study_dir):
        s = ds.ensure_stacked(log, cancel)
        g = gradient_context(s, study_dir, device, log, cancel)
        return {"ds": ds, "session": s, "study_dir": study_dir, "n": 0, "opts": opts, "device": device, **g}

    def canonical(self, active, ctx):
        m = str(active.get("bg_method", "auto"))
        if m == "reference" or (m == "auto" and ctx.get("skyref")):
            return {"bg_method": "sky-survey reference" if ctx.get("skyref") else "reference (unavailable)"}
        if m == "auto":
            return {**{k: v for k, v in active.items()}, "bg_method": "graxpert"}       # auto without the survey = GraXpert
        return active

    def run(self, ctx, p, log, cancel):
        from .. import post
        s, ds, yard = ctx["session"], ctx["ds"], ctx["yard"]
        ctx["n"] += 1
        wd = os.path.join(ctx["study_dir"], "work", f"bg_{ctx['n']}")
        _copy_products(s.dir, wd, ("stack.fits", "stack_meta.json", "skyref.npz"))
        prm = {**post.DEFAULTS, **p}
        t = time.time()
        try:
            r = post.Restorer(wd, s.meta, {"device": str(ctx["device"])}, prm, None, _checkpoint(cancel))     # loads the stack as the restore does
            before = _hwc(r.arr)
            r.background()
            dt = time.time() - t
            res = _hwc(r.arr)
            step = ([x for x in r.steps.records if x["step"] == "gradient"] or [{}])[-1]        # auto: the survey may be skipped, its fallback ran
            if step.get("status") not in (None, "done"):
                raise RuntimeError(f"{step.get('tool')} {step.get('status')}: {step.get('note') or ''}")
            out = {**gradient_metrics(ctx, before, res), "seconds": dt, "method_used": step.get("tool"), "note": step.get("note")}
            if ds.synthetic:
                out.update(_truth_scores(ds, s, r.arr, float(ctx["opts"].get("sigma_eval", 4.0)), stars_in_result=True, fit_plane=False))
            return out, res
        finally:
            shutil.rmtree(wd, ignore_errors=True)


# ----------------------------------------------------------------------------- linear restoration
_BXT = ["auto", "bxt"]
_STRENGTH_TOOLS = ["auto", "rl", "graxpert"]          # the Deconvolution slider acts on RL (blend) and GraXpert (-strength); auto falls back to RL without BXT


class RestoreTask(Task):
    name = "restore"
    label = "Linear restoration (restore stage)"
    description = ("The linear stage (post.restore) on a copy of the stack, all of its variables in one search: Cosmic Clarity's satellite-trail removal, the "
                   "gradient model (sky-survey reference, GraXpert, Siril polynomial / RBF; subtraction or division), the deconvolution tool of the objects and "
                   "of the stars (BlurXTerminator, GraXpert, Siril RL) with their strengths and the noise reduction tool (NoiseXTerminator, GraXpert, Siril) "
                   "with its strength.  The trials restore a stack of the frames other than the sharpest third, which is held out as the independent "
                   "reference of the score; the best trial's settings then restore the full stack.  The photometry constraints are measured against the "
                   "trial's own state before the deconvolution (the same trail removal and gradient model).  The state before the deconvolution (trail "
                   "removal, gradient, plate solve, colour calibration) is computed once per such settings.")
    objective_note = ("Score (restore_score, higher is better) = fidelity_gain + ln(1 - emission_change).  fidelity_gain: held-out validation - the "
                      "sharpest third of the frames is stacked on its own and held out; every trial restores the other frames, and the restoration, matched "
                      "to the held-out stack's PSF by a kernel fitted per tile (difference imaging: Alard & Lupton 1998, Bramich 2008), is compared with it "
                      "in noise-weighted chi2 (noise variance linear in the signal level, fitted on the unrestored stack; detected stars masked; both "
                      "high-passed at 8 FWHM).  The held-out noise is independent of the trial's, so the chi2 measures the error against the true sky at "
                      "the held-out resolution: a deconvolution that recovers real structure and a denoiser that removes noise lower it, one that invents, "
                      "smears or misplaces structure raises it.  fidelity_gain = 0.5 ln(chi2 unrestored / chi2).  ln(1 - emission_change): a gradient model "
                      "that takes a fraction f of the nebula's signal lowers its S/N by ln(1 - f) (emission_change ranked 12 gradient models against a "
                      "synthetic sky's truth at Spearman rho +0.99).  Constraints for what the chi2 does not see: star flux conserved (|median dmag| bright "
                      "<= 0.05, faint <= 0.10), no dark rings round the bright stars (moat), the gradient left in the sky <= 3 x the noise, the model's "
                      "structure at 1 - 10 % of the frame <= 15 %, no more satellite trails than Cosmic Clarity's removal leaves.  Without frames to hold "
                      "out (an HDR stack, < 6 frames) the depth gained replaces the fidelity gain (logged).")
    params = (_processing_params(["bg_method", "bg_smoothing", "bg_degree", "bg_correction", "deconvolution", "denoise", "white_balance",
                                  "bxt_auto", "bxt_sharpen_stars", "bxt_sharpen_nonstellar", "bxt_halos", "linear_denoise_auto", "linear_denoise_strength", "nxt_iterations"],
                                 {"deconvolution", "denoise"},
                                 hypertune={"bg_method": True, "bg_smoothing": True, "bg_degree": True, "bg_correction": True,
                                            "deconvolution": True, "denoise": True, "bxt_auto": True, "bxt_sharpen_stars": True, "bxt_sharpen_nonstellar": True,
                                            "bxt_halos": True, "linear_denoise_auto": True, "linear_denoise_strength": True, "nxt_iterations": True},
                                 when={"bg_degree": {"bg_method": ["poly"]}, "bg_smoothing": {"bg_method": ["auto", "graxpert", "rbf"]},
                                       "bg_correction": {"bg_method": ["auto", "graxpert", "poly", "rbf"]},
                                       "deconvolution": [{"deconv_method": _STRENGTH_TOOLS}, {"deconv_stars_method": _STRENGTH_TOOLS}],
                                       "bxt_auto": [{"deconv_method": _BXT}, {"deconv_stars_method": _BXT}],
                                       "bxt_sharpen_stars": {"bxt_auto": [False], "deconv_stars_method": _BXT},
                                       "bxt_halos": {"bxt_auto": [False], "deconv_stars_method": _BXT},
                                       "bxt_sharpen_nonstellar": {"bxt_auto": [False], "deconv_method": _BXT},
                                       "denoise": {"denoise_method": ["auto", "nxt", "graxpert", "siril"], "linear_denoise_auto": [True]},
                                       "linear_denoise_auto": {"denoise_method": ["auto", "nxt", "graxpert", "siril"]},
                                       "linear_denoise_strength": {"denoise_method": ["auto", "nxt", "graxpert", "siril"], "linear_denoise_auto": [False]},
                                       "nxt_iterations": {"denoise_method": ["auto", "nxt"], "linear_denoise_auto": [False]}})
              + [{"name": "deconv_method", "label": "Deconvolution tool (objects)", "type": "categorical", "choices": ["auto", "bxt", "graxpert", "rl", "none"],
                  "default": "auto", "tune": False, "hypertune": True, "pipeline": "deconv_method"},
                 {"name": "deconv_stars_method", "label": "Deconvolution tool (stars)", "type": "categorical", "choices": ["auto", "bxt", "graxpert", "rl", "none"],
                  "default": "auto", "tune": False, "hypertune": True, "pipeline": "deconv_stars_method"},
                 {"name": "denoise_method", "label": "Noise reduction tool", "type": "categorical", "choices": ["auto", "nxt", "graxpert", "siril", "none"],
                  "default": "auto", "tune": False, "hypertune": True, "pipeline": "denoise_method"},
                 {"name": "graxpert_psfsize", "label": "GraXpert PSF size (px; 0 = measured)", "type": "float", "low": 0.0, "high": 14.0, "default": 0.0, "tune": False,
                  "pipeline": "graxpert_psfsize", "when": [{"deconv_method": ["graxpert"]}, {"deconv_stars_method": ["graxpert"]}]},
                 {"name": "satellite_removal", "label": "Cosmic Clarity satellite-trail removal (before the gradient)", "type": "bool", "default": False, "tune": False,
                  "hypertune": True, "pipeline": "satellite_removal"}])
    # reference = auto without its fallback: the same model wherever auto uses the survey
    next(q for q in params if q["name"] == "bg_method")["ht_choices"] = ["auto", "graxpert", "poly", "rbf"]
    # the hyper-tune searches the tools that run on the GPU: Siril's Richardson-Lucy deconvolution and its denoiser have no GPU implementation
    # (a Siril RL trial took 10-25 min of CPU on M 31 against about a minute for BlurXTerminator / GraXpert); both stay the user's to choose
    for _q in params:
        if _q["name"] in ("deconv_method", "deconv_stars_method"):
            _q["ht_choices"] = ["auto", "bxt", "graxpert", "none"]
        elif _q["name"] == "denoise_method":
            _q["ht_choices"] = ["auto", "nxt", "graxpert", "none"]
    del _q
    metrics = {
        "restore_score": {"label": "Restore score: fidelity gained against the held-out frames + ln(1 - emission changed by the gradient model)", "direction": "maximize"},
        "fidelity_gain": {"label": "Fidelity gained against the held-out sharpest frames: 0.5 ln(chi2 unrestored / chi2)", "direction": "maximize"},
        "heldout_chi2": {"label": "Held-out mean chi2 (1 = noise only): the sharpest frames' stack against the restoration matched to its PSF", "direction": "minimize"},
        "depth_gain": {"label": "Depth gained over the state before the deconvolution (ln S/N)", "direction": "maximize"},
        "depth_before": {"label": "Depth before the deconvolution (after trail removal and gradient model)", "direction": "maximize"},
        **DEPTH_METRICS,
        "dm_bright_abs": {"label": "|dmag| bright reference stars against the state before the deconvolution", "direction": "minimize"},
        "dm_faint_abs": {"label": "|dmag| faint reference stars against the state before the deconvolution", "direction": "minimize"},
        "large_change": {"label": "Change above the PSF scale away from the stars (x noise)", "direction": "minimize"},
        "moat": {"label": "Dark ring depth round bright stars (x pixel noise)", "direction": "minimize"},
        "gain": {"label": "Global scale against the state before the deconvolution (a change of units; the photometry is judged after it)", "direction": "minimize"},
        "emission_change": {"label": "Emission changed by the gradient model (|fraction of the emission tiles' signal|)", "direction": "minimize"},
        "signal_removed": {"label": "Extended emission removed by the gradient model (fraction, emission tiles)", "direction": "minimize"},
        "sky_flatness": {"label": "Gradient left in the sky (x pixel noise)", "direction": "minimize"},
        "model_structure": {"label": "Structure the gradient model removed at 1-10 % of the frame (fraction)", "direction": "minimize"},
        "trails": {"label": "Satellite / aircraft trail pixels per million left", "direction": "minimize"},
        "bg_noise": {"label": "Background noise", "direction": "minimize"},
        "sharpness": {"label": "Sharpness S_F (Krotkov; rewards noise - reported only)", "direction": "maximize"},
        "seconds": {"label": "Run time (s)", "direction": "minimize"},
        **TRUTH_METRICS,
    }
    options = [{"name": "sigma_eval", "label": "Truth comparison resolution sigma (px)", "type": "float", "default": 1.0}]
    score = default_objective = "restore_score"
    min_gain = 0.01               # 3 sigma of the difference of two runs: identical restorations scatter by sd 0.0023 in depth (GPU BXT / NXT, Sh2-131)
    NOOP = {"deconv_method": "none", "deconv_stars_method": "none", "denoise_method": "none", "satellite_removal": False}
    CACHE = 4                     # states before the deconvolution kept (one per trail removal / gradient / colour settings; a full-frame M 31 state is 150 MB)

    def limits(self, ctx):
        # the held-out fidelity judges the image's structure (stars masked); the constraints cover what it does not see: the stars' photometry
        # and dark rings, the gradient model's flatness and structure, the satellite trails
        out = {"dm_bright_abs": ("max", 0.05), "dm_faint_abs": ("max", 0.10), "sky_flatness": ("max", 3.0), "model_structure": ("max", 0.15)}
        out["moat"] = ("max", (ctx.get("ref_moat") or 0.0) + 1.0)
        if ctx.get("trails_min") is not None:
            # no trial may leave more satellite trails than Cosmic Clarity's removal takes out of this stack (the depth's robust noise does not see them)
            t = float(ctx["trails_min"])
            out["trails"] = ("max", max(1.25 * t, t + 20.0))
        return out

    def prepare(self, ds, opts, device, log, cancel, study_dir):
        s = ds.ensure_stacked(log, cancel)
        from ..hypertune import available_tools
        tools = available_tools()
        auto_dc = "bxt" if tools.get("bxt") else "rl"
        auto_dn = next((t for t, k in (("nxt", "nxt"), ("graxpert", "graxpert_denoise")) if tools.get(k)), "siril")
        src, sharp = self._held_out(s, study_dir, log, cancel)
        g = gradient_context(s, study_dir, device, log, cancel, src_dir=src)
        return {"ds": ds, "session": s, "study_dir": study_dir, "n": 0, "device": device, "opts": opts, "refs": {}, "pre": {},
                "auto_deconv": auto_dc, "auto_denoise": auto_dn, "cc_satellite": bool(tools.get("cc_satellite")), "grad": g, "skyref": g["skyref"],
                "src_dir": src, "sharp_dir": sharp}

    HELD_OUT = 1.0 / 3.0          # the sharpest third of the frames is held out

    def _held_out(self, s, study_dir, log, cancel):
        """The held-out frames: the sharpest third of the stacked frames (FWHM) stacked on their own, the rest stacked for the trials to restore -
        both on stack.fits's pixel grid (Session.stack_subsets).  Returns (trials' source dir, held-out dir), or (None, None) when the stack
        cannot be split (an HDR stack, fewer than 6 frames, no frame list): then the trials restore the stack and are scored on depth (logged)."""
        from .. import fitsio
        try:
            rows = s._load_rows()
            inc = sorted((i for i, r in enumerate(rows) if r.get("included")), key=lambda i: rows[i].get("fwhm") or 1e9)
            k = int(round(len(inc) * self.HELD_OUT))
            if k < 3 or len(inc) - k < 3:
                raise RuntimeError(f"{len(inc)} frames: too few to hold out a third")
            st = s.stack_subsets({"sharp": inc[:k], "rest": inc[k:]})
            hdr = fitsio.read_rgb(s._p("stack.fits"))[1]
            dirs = {}
            for name, arr in st.items():
                d = os.path.join(study_dir, "work", f"heldout_{name}")
                _copy_products(s.dir, d, ("stack_meta.json", "skyref.npz"))
                fitsio.write_rgb(os.path.join(d, "stack.fits"), arr, hdr)
                dirs[name] = d
            fw = lambda ix: float(np.median([rows[i]["fwhm"] for i in ix if rows[i].get("fwhm")]))
            log(f"Held out: the {k} sharpest frames (median FWHM {fw(inc[:k]):.2f} px); the trials restore the other {len(inc) - k} ({fw(inc[k:]):.2f} px)")
            return dirs["rest"], dirs["sharp"]
        except Exception as e:                         # noqa: BLE001
            log(f"No held-out frames ({type(e).__name__}: {e}): the trials restore the stack and are scored on the depth gained")
            return None, None

    def canonical(self, active, ctx):
        """What runs: auto = BlurXTerminator when licensed (else Siril RL), the denoiser auto = the first available of NXT / GraXpert / Siril, the
        gradient model auto = the sky-survey reference where it covers the field (its smoothing / correction unused), else GraXpert; the
        Deconvolution slider only switches a BXT-only deconvolution on (> 0) or off; strengths of tools that do not run do not count."""
        a = dict(active)
        if "deconv_method" in a or "deconv_stars_method" in a:
            o = str(a.get("deconv_method", "auto")).replace("auto", ctx.get("auto_deconv", "auto"))
            st = str(a.get("deconv_stars_method", "auto") or o).replace("auto", ctx.get("auto_deconv", "auto"))
            a["deconv_method"], a["deconv_stars_method"] = o, st
            if "deconvolution" in a and not {"rl", "graxpert"} & {o, st}:
                a["deconvolution"] = float(a["deconvolution"]) > 0
            if "bxt" not in (o, st):
                for k in ("bxt_auto", "bxt_sharpen_stars", "bxt_sharpen_nonstellar", "bxt_halos"):
                    a.pop(k, None)
        if "denoise_method" in a:
            a["denoise_method"] = str(a["denoise_method"]).replace("auto", ctx.get("auto_denoise", "auto"))
            if a["denoise_method"] != "nxt":
                a.pop("nxt_iterations", None)
        if "bg_method" in a:
            m = str(a["bg_method"])
            if m == "reference" or (m == "auto" and ctx.get("skyref")):
                for k in ("bg_smoothing", "bg_degree", "bg_correction"):
                    a.pop(k, None)
                a["bg_method"] = "sky-survey reference" if ctx.get("skyref") else "reference (unavailable)"
            elif m == "auto":
                a["bg_method"] = "graxpert"
        return a

    def _settings(self, ctx, p):
        from .. import post
        tool_keys = ("deconv_method", "deconv_stars_method", "denoise_method", "graxpert_psfsize", "satellite_removal")
        prm = {**post.DEFAULTS, **{k: v for k, v in p.items() if k not in tool_keys}}
        s = ctx["session"]
        sp = {**(s.meta.get("params") or {}), **(ctx["opts"].get("stack_params") or {}),
              **{k: p[k] for k in tool_keys if k in p}, "device": str(ctx["device"])}
        if not p.get("deconv_stars_method"):
            sp["deconv_stars_method"] = ""
        return prm, sp

    def _restore(self, ctx, prm, sp, tag, cancel, src: str | None = None):
        """post.restore's steps on a copy of the stack: (restored image, its meta, the state before the deconvolution, the gradient model's metrics).
        Everything before the deconvolution (trail removal, gradient, plate solve, BlurXTerminator's correct-only PSF pass, SPCC) depends only on
        the gradient / colour settings, the trail removal and whether BXT is among the deconvolution tools: that state is computed once per such
        key and the trials run from it (a full restore was 2 min a trial on M 31, most of it SPCC)."""
        import copy
        from .. import post
        s = ctx["session"]
        wd = os.path.join(ctx["study_dir"], "work", tag)
        _copy_products(src or ctx.get("src_dir") or s.dir, wd, ("stack.fits", "stack_meta.json", "skyref.npz"))
        try:
            r = post.Restorer(wd, s.meta, sp, {**post.DEFAULTS, **prm}, None, _checkpoint(cancel))
            psf = (r.source != "imagemm.fits" and float(r.p["deconvolution"]) > 0 and r.deconv != ("none", "none") and bool({"auto", "bxt"} & set(r.deconv)))
            key = json.dumps([{k: r.p.get(k) for k in ("background", "bg_method", "bg_correction", "bg_degree", "bg_smoothing", "white_balance", "spcc_sensor",
                                                        "spcc_filter", "spcc_white_ref")}, bool(sp.get("satellite_removal")), psf], sort_keys=True, default=str)
            pre = ctx.setdefault("pre", {}) if src is None else {}
            t0 = time.time()
            if key in pre:
                c = pre.pop(key)
                pre[key] = c                                            # most recently used last
                r.arr, r.header, r.info = c["arr"].copy(), c["header"].copy(), copy.deepcopy(c["info"])
                r.psf_corrected, r.steps.records = c["psf_corrected"], copy.deepcopy(c["records"])
            else:
                r.satellite()
                before = _hwc(r.arr)
                r.background()
                step = ([x for x in r.steps.records if x["step"] == "gradient"] or [{}])[-1]
                if r.p.get("background") and step.get("status") not in (None, "done", "skipped"):
                    raise RuntimeError(f"gradient model {step.get('tool')} {step.get('status')}: {step.get('note') or ''}")
                grad = gradient_metrics(ctx["grad"], before, _hwc(r.arr)) if r.p.get("background") else {"emission_change": 0.0}
                grad["gradient_tool"] = step.get("tool")
                del before
                solved = r.plate_solve()
                r.psf_correction()
                r.colour(solved)
                while len(pre) >= self.CACHE:
                    pre.pop(next(iter(pre)))
                pre[key] = {"arr": r.arr.copy(), "header": r.header.copy(), "info": copy.deepcopy(r.info), "psf_corrected": r.psf_corrected,
                            "records": copy.deepcopy(r.steps.records), "grad": grad}
            r.deconvolve()
            r.denoise()
            meta = r.save()
            meta["seconds"] = round(time.time() - t0, 1)
            return r.arr.copy(), meta, key
        finally:
            shutil.rmtree(wd, ignore_errors=True)

    def _paired(self, ctx, key, log):
        """The yardstick of the paired constraints: the trial's own state before the deconvolution (same trail removal, gradient and colour)."""
        refs = ctx["refs"]
        if key not in refs:
            while len(refs) >= self.CACHE:
                refs.pop(next(iter(refs)))
            refs[key] = OB.Yardstick(_hwc(ctx["pre"][key]["arr"]))
        return refs[key]

    def _trails_bound(self, ctx, prm, sp, lin0, log, cancel):
        """The trails left in this stack (``lin0``: its restore without trail removal), and what Cosmic Clarity's removal leaves (once): the bound
        of the trails constraint."""
        t0, t1 = trail_pixels(_hwc(lin0)), None
        if ctx.get("cc_satellite"):
            try:
                lin1, _, _ = self._restore(ctx, prm, {**sp, **self.NOOP, "satellite_removal": True}, "restore_trails1", cancel)
                t1 = trail_pixels(_hwc(lin1))
            except Exception as e:                       # noqa: BLE001
                log(f"Cosmic Clarity trail removal on the reference: {e}")
        ctx["trails_min"] = min(t0, t1) if t1 is not None else t0
        log(f"Satellite trails: {t0:.0f} pixels per million in the stack" + (f", {t1:.0f} after Cosmic Clarity's removal" if t1 is not None else ""))

    def run(self, ctx, p, log, cancel):
        s, ds = ctx["session"], ctx["ds"]
        ctx["n"] += 1
        prm, sp = self._settings(ctx, p)
        if "depth_yard" not in ctx:
            # the stars and empty-sky apertures of the depth, once for the whole study: the baseline's (trial 0) settings without deconvolution / noise
            # reduction - every trial's depth on the same positions (depth is a ratio of fluxes to noise: neither a gradient's pedestal nor a change
            # of units moves it)
            lin, _, _ = self._restore(ctx, prm, {**sp, **self.NOOP}, "restore_ref", cancel)
            ctx["depth_yard"] = OB.Yardstick(_hwc(lin), keep_image=False)
            y = ctx["depth_yard"]
            log(f"Depth reference: {len(y.sx)} stars, FWHM {y.fwhm:.2f} px, {len(y.ax)} sky apertures, depth {y.ref.get('depth', float('nan')):.3f}")
            self._trails_bound(ctx, prm, sp, lin, log, cancel)
            if ctx.get("sharp_dir"):
                # the held-out reference: the sharpest frames' stack through the same linear stage (gradient, colour) without deconvolution / noise
                # reduction, and the noise model from the unrestored trial source
                sh, _, _ = self._restore(ctx, prm, {**sp, **self.NOOP}, "restore_heldout", cancel, src=ctx["sharp_dir"])
                fw = OB.Yardstick(_hwc(sh), keep_image=False).fwhm
                ctx["heldout"] = OB.HeldOut(_hwc(sh), _hwc(lin), fw)
                h = ctx["heldout"]
                log(f"Held-out reference: FWHM {fw:.2f} px, {h.n_stars} stars masked, noise variance {h.nlf[0]:.3g} + {h.nlf[1]:.3g} x level; "
                    f"unrestored chi2 {h.chi2_r0:.4f}")
        t = time.time()
        lin, meta, key = self._restore(ctx, prm, sp, f"restore_{ctx['n']}", cancel)
        dt = time.time() - t
        est = _hwc(lin)
        yp = self._paired(ctx, key, log)
        ctx["ref_moat"] = yp.ref.get("moat")
        m = yp.measure(est, same_grid=True)
        d = ctx["depth_yard"].measure(est, same_grid=True, normalise_gain=False)
        out = {k: m.get(k) for k in ("dm_bright_abs", "dm_faint_abs", "dm_bright", "dm_faint", "large_change", "moat", "gain")}
        out.update({k: d.get(k) for k in ("depth", "snr_point", "snr_total", "fwhm")})
        pre = ctx["pre"][key]
        if "depth0" not in pre:
            # the depth of this trial's own state before the deconvolution (the same trail removal and gradient model), on the same stars and apertures:
            # the depth gained is the deconvolution's and the noise reduction's.  The depth itself is not invariant to the gradient model - the nebula's
            # structure in the empty-sky apertures and annuli is noise to it: on the synthetic sky GraXpert, which took 85 % of the nebula, read 0.40
            # deeper than a quadratic with the same (no) restoration
            pre["depth0"] = ctx["depth_yard"].measure(_hwc(pre["arr"]), same_grid=True, normalise_gain=False).get("depth")
        out["depth_before"] = pre["depth0"]
        out["depth_gain"] = (out["depth"] - pre["depth0"]) if out.get("depth") is not None and pre["depth0"] is not None else None
        g = pre["grad"]
        out.update({k: g.get(k) for k in ("emission_change", "signal_removed", "sky_flatness", "model_structure", "gradient_tool")})
        keep = math.log(max(1.0 - float(g.get("emission_change") or 0.0), 1e-3))
        if ctx.get("heldout") is not None:
            h = ctx["heldout"]
            out["heldout_chi2"] = h.chi2(est)
            out["heldout_tiles"] = h.last_tiles
            out["fidelity_gain"] = 0.5 * math.log(h.chi2_r0 / max(out["heldout_chi2"], 1e-12))
            out["restore_score"] = float(out["fidelity_gain"] + keep)
        elif out.get("depth_gain") is not None:
            out["restore_score"] = float(out["depth_gain"] + keep)
        out.update(bg_noise=MX.background_noise(est), sharpness=float(np.mean([MX.sharpness(est[..., c]) for c in range(3)])), seconds=dt,
                   trails=trail_pixels(est), tools=" + ".join(meta.get("tools") or []) or "none",
                   failed=[f"{r['tool']}: {r['status']}" for r in meta.get("steps", []) if r.get("status") in ("failed", "unavailable")])
        if ds.synthetic:
            out.update(_truth_scores(ds, s, lin, float(ctx["opts"].get("sigma_eval", 1.0))))
        return out, est


# ----------------------------------------------------------------------------- star separation
class StarsTask(Task):
    name = "stars"
    label = "Star separation"
    description = ("Star removal (post.remove_stars: StarXTerminator, StarNet2, Cosmic Clarity DarkStar) on the dataset's linear image, scored on the reference stars "
                   "of the linear image: the light left at their positions in the starless image (local-background apertures; light left and holes dug both count), "
                   "and the change away from every star.  Synthetic data: the starless image against the true sky without stars.")
    objective_note = ("Score: star_residual = Σ |flux at the reference stars in the starless image| / Σ their flux in the linear image (2.5 FWHM apertures, local "
                      "annulus background): left-over cores and inpainting holes are both errors.  Constraints: the starless image unchanged above the PSF scale "
                      "away from the stars within 1 × the noise there (the nebula and the sky are not the remover's to touch), and holes (a star's aperture "
                      "more than 3 σ below its surroundings) at most 5 % of the stars.")
    params = [
        # the hyper-tune searches the GPU tools: StarNet2's build is CPU-only (tools.py) - it stays the user's to choose
        {"name": "star_removal", "label": "Tool", "type": "categorical", "choices": ["auto", "sxt", "starnet", "darkstar"], "ht_choices": ["sxt", "darkstar"],
         "default": "auto", "tune": True, "hypertune": True, "processing": "star_removal"},
        {"name": "starnet_stride", "label": "StarNet2 stride", "type": "categorical", "choices": [64, 128, 256], "default": 256, "tune": False, "hypertune": True,
         "processing": "starnet_stride", "when": {"star_removal": ["starnet"]}},
    ]
    metrics = {
        "star_residual": {"label": "Star light left + holes at the reference stars (fraction of their flux)", "direction": "minimize"},
        "holes": {"label": "Stars with a hole dug in the starless image (fraction)", "direction": "minimize"},
        "offstar_large_change": {"label": "Change above the PSF scale away from the stars (× noise)", "direction": "minimize"},
        "residual_flux": {"label": "Star flux left in the starless image (fraction, positive only)", "direction": "minimize"},
        "residual_stars": {"label": "Stars still detected in the starless image (fraction)", "direction": "minimize"},
        "offstar_change": {"label": "Mean change away from the stars (× noise)", "direction": "minimize"},
        "seconds": {"label": "Run time (s)", "direction": "minimize"},
        **{k: v for k, v in TRUTH_METRICS.items() if k in ("truth_nrmse", "truth_faint_nrmse", "truth_ssim", "truth_psnr")},
    }
    options = [{"name": "sigma_eval", "label": "Truth comparison resolution σ (px)", "type": "float", "default": 1.0}]
    score = default_objective = "star_residual"
    min_gain = 0.002

    def static_limits(self):
        return {"offstar_large_change": ("max", 1.0), "holes": ("max", 0.05)}

    def prepare(self, ds, opts, device, log, cancel, study_dir):
        import sep as _sep
        s = ds.ensure_stacked(log, cancel)
        src = s._p("linear.fits") if os.path.exists(s._p("linear.fits")) else s._p("stack.fits")
        lin = _read(src)
        L = OB.lum(_hwc(lin))
        y = OB.Yardstick(_hwc(lin))
        bk = _sep.Background(L, bw=64, bh=64)
        sub = L - bk.back()
        objs = _sep.extract(sub, 5.0, err=bk.globalrms, minarea=5)
        if len(objs) < 20:
            raise RuntimeError(f"only {len(objs)} stars found: too few to score a star removal")
        fw = float(np.median(2.0 * np.sqrt(objs["a"] * objs["b"]))) or 3.0
        r_ap = max(1.5 * fw, 2.0)
        f0, _, _ = _sep.sum_circle(L, objs["x"], objs["y"], r_ap, bkgann=(2.5 * r_ap, 4 * r_ap), subpix=5)
        keep = f0 > 0
        mask = np.zeros(L.shape, np.uint8)
        for x_, y_ in zip(objs["x"], objs["y"]):
            cv2.circle(mask, (int(round(x_)), int(round(y_))), int(round(2 * r_ap)), 1, -1)
        off = mask == 0
        off[:8], off[-8:], off[:, :8], off[:, -8:] = False, False, False, False
        log(f"{len(y.sx)} reference stars (FWHM {y.fwhm:.2f} px); {int(keep.sum())} detected stars; {int(off.mean() * 100)} % of the image away from every star")
        from ..hypertune import available_tools
        tools = available_tools()
        return {"ds": ds, "session": s, "lin": lin, "L0": L, "src": os.path.basename(src), "ref_x": objs["x"][keep], "ref_y": objs["y"][keep], "ref_f": f0[keep], "r_ap": r_ap,
                "fw": fw, "off": off, "noise": float(bk.globalrms), "study_dir": study_dir, "n": 0, "opts": opts, "device": device, "yard": y,
                "auto_stars": "sxt" if tools.get("sxt") else "starnet"}

    def canonical(self, active, ctx):
        a = dict(active)
        if "star_removal" in a:
            a["star_removal"] = str(a["star_removal"]).replace("auto", ctx.get("auto_stars", "auto"))
            if a["star_removal"] != "starnet":
                a.pop("starnet_stride", None)
        return a

    def run(self, ctx, p, log, cancel):
        import sep as _sep
        from scipy.spatial import cKDTree
        from .. import post
        s, ds, y = ctx["session"], ctx["ds"], ctx["yard"]
        ctx["n"] += 1
        wd = os.path.join(ctx["study_dir"], "work", f"stars_{ctx['n']}")
        _copy_products(s.dir, wd, ("stack.fits", "stack_meta.json", "linear.fits", "restore_meta.json"))
        prm = {**post.DEFAULTS, **p}
        sp = {**(s.meta.get("params") or {}), "device": str(ctx["device"])}
        t = time.time()
        try:
            meta = post.remove_stars(wd, s.meta, prm, sp, None, _checkpoint(cancel))
            dt = time.time() - t
            if not os.path.exists(os.path.join(wd, "starless.fits")):
                raise RuntimeError(f"no starless image: {meta.get('tool')} {meta.get('reason') or ''} {[a.get('note') for a in meta.get('attempts', [])]}")
            starless = _read(os.path.join(wd, "starless.fits"))
            L1 = OB.lum(_hwc(starless))
            # the reference stars of the linear image: what is left there (light and holes both)
            f1 = y._phot(L1, y.sx, y.sy, y.r_total)
            ok = np.isfinite(f1) & (y.f_total > 0)
            m1 = y.measure(_hwc(starless), same_grid=True, normalise_gain=False)
            sig = m1.get("sigma_total") or float("nan")
            out = {"star_residual": float(np.abs(f1[ok]).sum() / y.f_total[ok].sum()),
                   "holes": float(np.mean(f1[ok] < -3.0 * sig)) if np.isfinite(sig) else None,
                   "offstar_large_change": m1.get("large_change"), "seconds": dt, "tool": meta.get("tool")}
            r = ctx["r_ap"]
            fl, _, _ = _sep.sum_circle(L1, ctx["ref_x"], ctx["ref_y"], r, bkgann=(2.5 * r, 4 * r), subpix=5)
            out["residual_flux"] = float(np.clip(fl, 0, None).sum() / ctx["ref_f"].sum())
            bk = _sep.Background(L1, bw=64, bh=64)
            objs = _sep.extract(L1 - bk.back(), 5.0, err=max(bk.globalrms, ctx["noise"]), minarea=5)
            if len(objs):
                d, _ = cKDTree(np.stack([objs["x"], objs["y"]], 1)).query(np.stack([ctx["ref_x"], ctx["ref_y"]], 1))
                out["residual_stars"] = float(np.mean(d < max(1.5 * ctx["fw"], 2.0)))
            else:
                out["residual_stars"] = 0.0
            out["offstar_change"] = float(np.mean(np.abs(L1 - ctx["L0"])[ctx["off"]]) / max(ctx["noise"], 1e-9))
            if ds.synthetic:
                out.update(_truth_scores(ds, s, starless, float(ctx["opts"].get("sigma_eval", 1.0)), stars_in_result=False))
            return out, _hwc(starless)
        finally:
            shutil.rmtree(wd, ignore_errors=True)


# ----------------------------------------------------------------------------- finishing
_HUBBLE = ["sho", "hso", "osh", "ohs", "hos"]
GATE_WEIGHT = 1.0             # finish_score: one failed gate outweighs any difference of the structural fidelity (0 - 1)
MIN_GAIN_H = 0.005            # half a percent of the significant structure made visible


def _finish_source(layers) -> np.ndarray:
    """The linear luminance the finish renders (the starless layer plus the star layer, or the linear image): the HDR side of the fidelity."""
    lin = layers.base if layers.stars is None else layers.base + layers.stars
    return OB.luminance_y(np.nan_to_num(lin, nan=0.0))


def _finish_ldr(img: np.ndarray, shape) -> np.ndarray:
    """The finished image's luminance on the linear grid (the finishing chain keeps the grid; a crop or a resize would show as a shape mismatch)."""
    y = OB.luminance_y(np.clip(img, 0.0, 1.0))
    if y.shape != tuple(shape):
        y = cv2.resize(y, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return y
_GHS = ["ghs2", "autoghs"]


class FinishTask(Task):
    name = "finish"
    label = "Stretch & finish (finishing chain)"
    display_ready = True
    description = ("The finishing chain (pro.py) on the dataset's cached starless and star layers (or the linear / stacked image): the stretch tool and its "
                   "settings, highlights, the noise reduction of the stretched image (NoiseXTerminator or Cosmic Clarity Denoise), Cosmic Clarity Sharpen, "
                   "dual-band unmixing, local contrast, saturation, the star layer and the sky level.  Scored by the structural fidelity of the finished image to "
                   "its own linear data (how much of the structure the data holds above its noise the finish makes visible, TMQI's multi-scale measure), "
                   "subject to every finishing gate (sky level, clipping, sky chroma, sky noise, star count / shape / colour).  No reference image: the palette "
                   "and the colour grade are the user's.  Trial images are the finished renders (one chain, same data).  Option 'external': run "
                   "NoiseXTerminator / GraXpert / Cosmic Clarity on the stretched image in every trial (slower, the export's result; without it those settings "
                   "have no effect).")
    objective_note = ("Score (finish_score, lower is better): the number of failed finishing gates, then 1 - H.  H is the structural fidelity of the finished "
                      "luminance to the linear luminance (Yeganeh & Wang, IEEE TIP 2013, the authors' TMQI.m: 11x11 Gaussian windows, 5 scales, the Mannos "
                      "CSF thresholds of visibility on the finished side), adapted to astronomical data: a structure counts where the linear data's local "
                      "deviation is significant above its own noise (Immerkaer's estimator, 3 sigma), and H is the fraction of that structure the finish makes "
                      "visible in its shape (hit rate x local correlation, scale-weighted).  Constraints: every finishing gate (FINISH-SPEC) no further outside "
                      "its limit than in the baseline - gate by gate, so that no gate is traded for another.  The gates bound what H alone would reward: "
                      "the sky's noise stretched into view, clipping, a sky off its level.")
    params = _processing_params(
        ["stretch_method", "stretch", "stretch_boost", "auto_stretch", "contrast", "hdr", "hdr_method", "color_preservation", "luminance_denoise", "chroma_denoise",
         "cc_denoise", "cc_denoise_model", "cc_sharpen", "cc_stellar_amount", "cc_nonstellar_amount", "palette", "hubble_formula", "oiii_boost", "oiii_unmix",
         "synthetic_luminance", "saturation", "local_contrast", "sharpen", "star_reduction", "star_intensity", "star_saturation", "star_color_preservation",
         "halo_suppress", "scnr", "black_point", "brightness", "sky_levelling", "sky_level",
         "grade_amount", "grade_temperature", "grade_tint", "grade_warm_hue", "grade_warm_sat", "grade_cool_hue", "grade_cool_sat", "grade_contrast"],
        {"stretch", "contrast", "hdr", "saturation", "local_contrast", "star_reduction"},
        hypertune={**{k: True for k in ("stretch_method", "stretch", "stretch_boost", "contrast", "hdr", "hdr_method", "color_preservation", "luminance_denoise",
                                       "chroma_denoise", "cc_denoise", "cc_denoise_model", "cc_sharpen", "cc_stellar_amount", "cc_nonstellar_amount", "saturation",
                                       "local_contrast", "sharpen", "star_reduction", "star_intensity", "star_saturation", "star_color_preservation", "halo_suppress",
                                       "scnr", "black_point", "brightness", "sky_levelling", "sky_level")},
                   # the palette, its formula / OIII boost and the colour grade are matters of style: the user's, never searched
                   "oiii_unmix": "dual", "synthetic_luminance": "dual"},
        when={"stretch_boost": {"stretch_method": ["statstretch"]}, "contrast": {"stretch_method": _GHS}, "color_preservation": {"stretch_method": _GHS},
              "cc_denoise_model": {"cc_denoise": [True]}, "cc_stellar_amount": {"cc_sharpen": ["both", "stellar"]},
              "cc_nonstellar_amount": {"cc_sharpen": ["both", "nonstellar"]}, "hubble_formula": {"palette": _HUBBLE}, "oiii_boost": {"palette": ["nbn"]},
              "brightness": {"sky_levelling": [False]}, "sky_level": {"sky_levelling": [True]}})
    # every setting is searched over its whole range: what is feasible is the objective's to decide (the finishing gates are its constraints),
    # not a narrowed search space
    params.append({"name": "ghs_d2", "label": "GHS pass 2 D (ghs2)", "type": "float", "low": 0.3, "high": 6.0, "default": 2.5, "tune": True, "hypertune": True,
                   "processing": "ghs_d2", "when": {"stretch_method": ["ghs2"]}})
    params.append({"name": "star_stretch", "label": "Star stretch amount (Seti Astro)", "type": "float", "low": 2.0, "high": 8.0, "default": 5.0, "tune": False,
                   "hypertune": True, "processing": "star_stretch"})
    metrics = {
        "finish_score": {"label": "Finish score: gates failed + (1 - structural fidelity H)", "direction": "minimize"},
        "fidelity": {"label": "Structural fidelity H: significant linear structure made visible (0-1)", "direction": "maximize"},
        "truth_fidelity": {"label": "Truth: structural fidelity to the true sky (synthetic data)", "direction": "maximize"},
        **{f"fidelity_s{i + 1}": {"label": f"Fidelity at scale {i + 1} ({2 ** i} px)", "direction": "maximize"} for i in range(5)},
        "gate_distance": {"label": "Gates: distance outside the limits (0 = all pass)", "direction": "minimize"},
        **{f"gate_{k}": {"label": f"Gate {k}: distance outside its limit", "direction": "minimize"} for k in GATE_NAMES},
        "gates_failed": {"label": "Gates failed (count)", "direction": "minimize"},
        "cnr": {"label": "Contrast-to-noise: (nebula p90 - sky) / sky pixel noise", "direction": "maximize"},
        "nebula_contrast": {"label": "Nebula contrast (p90 - sky, lightness)", "direction": "maximize"},
        "sky_median": {"label": "Sky level", "direction": "minimize"},
        "sky_noise": {"label": "Sky pixel noise", "direction": "minimize"},
        "sky_chroma": {"label": "Sky chroma (max |a|, |b|)", "direction": "minimize"},
        "clipped": {"label": "Clipped fraction", "direction": "minimize"},
        "stars_coloured": {"label": "Star colour saturation", "direction": "maximize"},
        "seconds": {"label": "Run time (s)", "direction": "minimize"},
    }
    options = [{"name": "external", "label": "AI noise reduction / Cosmic Clarity per trial (1 = yes)", "type": "int", "default": 0},
               {"name": "max_side", "label": "Render size (px, 0 = full)", "type": "int", "default": 0}]
    score = "finish_score"
    default_objective = "finish_score"
    # the search: the (1+1)-CMA-ES for constrained optimisation from the baseline (lab.es) - constrained TPE over the full ranges left 50 of 60 trials
    # outside the gates on M 31, its only feasible trials the warm-start perturbations of the baseline
    ht_sampler = "es"
    min_gain = MIN_GAIN_H        # the finishing chain is deterministic (three identical renders, identical scores): a margin of significance, not of noise

    def limits(self, ctx):
        """No gate may get worse than in the baseline: each gate's violation at most the baseline's.  (A bound on the summed distance let trials buy
        one gate with another - on M 31, where Auto-finish's result fails five gates, crushing the sky to black fixed the colour gates and 8 such
        renders passed as feasible.)"""
        b = (ctx.get("base") or {}).get("gate_violations") or {}
        return {f"gate_{k}": ("max", float(b.get(k, 0.0)) + 1e-6) for k in GATE_NAMES}

    def prepare(self, ds, opts, device, log, cancel, study_dir):
        from .. import post
        s = ds.ensure_stacked(log, cancel)
        p0 = dict(post.DEFAULTS)
        max_side = int(opts.get("max_side", 0) or 0) or None
        layers = post._Layers(s.dir, p0, max_side)
        dual = post._dual_band(s.meta, layers.header)
        log(f"Layers: {os.path.basename(layers.src)}{' + stars' if layers.stars is not None else ''}")
        hdr = _finish_source(layers)
        truth = None
        if ds.synthetic:
            # validation: the fidelity to the true sky (registered onto the layers, scaled to their units by a robust fit), significance at the data's noise
            try:
                lin = layers.base if layers.stars is None else layers.base + layers.stars
                tr, valid, _ = truth_on_stack(ds, s, lin, extended=True, stars=True)
                tl, ok = OB.luminance_y(tr), valid & np.isfinite(hdr)
                A = np.stack([tl[ok][::7], np.ones(int(ok[::1].sum()))[::7]], 1)
                coef, *_ = np.linalg.lstsq(A, hdr[ok][::7], rcond=None)
                truth = np.where(valid, coef[0] * tl + coef[1], hdr)
                log(f"Truth registered onto the layers (scale {coef[0]:.4g})")
            except Exception as e:                       # noqa: BLE001
                log(f"No truth fidelity: {type(e).__name__}: {e}")
        work = os.path.join(study_dir, "work", "finish")
        os.makedirs(work, exist_ok=True)
        saved = _store(study_dir, "reference.pkl") or {}
        return {"ds": ds, "session": s, "layers": layers, "dual": dual, "hdr": hdr, "truth": truth, "work": work, "device": device, "opts": opts, "header": layers.header,
                "study_dir": study_dir, "base": saved.get("base")}

    def run(self, ctx, p, log, cancel):
        from .. import finish, gates, post, pro
        layers = ctx["layers"]
        prm = {**post.DEFAULTS, **post.EXTRA_DEFAULTS, **(ctx["opts"].get("params") or {}), **p}
        if cancel.is_set():
            raise RuntimeError("cancelled")
        t = time.time()
        fin = pro.Finisher(ctx["work"], ctx["header"], prm, ctx["dual"], external=bool(int(ctx["opts"].get("external", 0))), device=str(ctx["device"]),
                           checkpoint=_checkpoint(cancel), keep_pictures=False)
        img, info = fin.run(layers.base, layers.stars)
        dt = time.time() - t
        sm = layers.starnet.get("stars") or {}
        level = float(pro.mtf_balance(pro.star_stretch_m(float(prm.get("star_stretch", pro.STAR_STRETCH))), 5.0 * float(sm["sky_noise"]))) if sm.get("sky_noise") else None
        g = gates.compute(img, starless=info.get("starless") if layers.stars is not None and prm["star_reduction"] < 1.0 else None, linear_star_count=sm.get("count"), star_level=level)
        lum = finish.luminance(img)
        hs = pro._hist_stats(lum)
        mask, _ = pro.nebula_mask(lum, hs["mode"])
        neb = float(np.percentile(lum[mask], 90) - g["sky_median"]["value"]) if mask.any() else 0.0
        noise = g.get("sky_pixel_noise", {}).get("value")
        cnr = float(neb / noise) if noise else None
        gv = _gate_violations(g)
        out = {"gate_distance": _gate_distance(g), "gates_failed": len(g.get("failed", [])), "nebula_contrast": neb, "sky_median": g["sky_median"]["value"],
               **{f"gate_{k}": v for k, v in gv.items()},
               "sky_noise": noise, "cnr": cnr,
               "sky_chroma": max(g["sky_chroma"]["value"].values()) if isinstance(g.get("sky_chroma", {}).get("value"), dict) else None,
               "clipped": g["clipped"]["value"], "stars_coloured": g.get("stars_coloured", {}).get("value"), "seconds": dt, "failed": g.get("failed")}
        # lexicographic: the finishing gates are the specification - fewer failed gates first (H is 0 - 1, a gate counts 1), then the structural
        # fidelity.  On M 31 Auto-finish's result failed the star-colour gate after BXT's star sharpening
        fails = len(g.get("failed", []))
        H, hl = OB.structural_fidelity(ctx["hdr"], _finish_ldr(img, ctx["hdr"].shape), mode="hits")
        out.update({"fidelity": H, **{f"fidelity_s{i + 1}": v for i, v in enumerate(hl)}})
        if ctx.get("truth") is not None:
            out["truth_fidelity"] = OB.structural_fidelity(ctx["truth"], _finish_ldr(img, ctx["hdr"].shape), mode="hits", noise_from=ctx["hdr"])[0]
        out["finish_score"] = GATE_WEIGHT * fails + (1.0 - H)
        if ctx.get("base") is None:
            ctx["base"] = {"gate_distance": out["gate_distance"], "gate_violations": gv}
            _store(ctx["study_dir"], "reference.pkl", {"base": ctx["base"]})
        from .. import export as export_mod
        return out, export_mod.display_image(img, export_mod.bottom_up(ctx["header"]))        # (H, W, 3) north-up like the exports


# ----------------------------------------------------------------------------- super-resolution (Cosmic Clarity)
class SuperresTask(Task):
    name = "superres"
    label = "Super-resolution (Cosmic Clarity)"
    display_ready = True
    description = ("Enlarges the dataset's finished image (the last export's finish.fits) 2x, 3x or 4x with Cosmic Clarity Super-Resolution (SetiAstro, its models for "
                   "those scales) or Lanczos, as the export would.  Scored by the stars' FWHM in the original pixels (resolution gained) under the constraint that "
                   "the enlargement, binned back, reproduces the original (cycle consistency): an enlargement may add detail, not change the image.")
    objective_note = ("Score: star FWHM in original pixels (the enlarged FWHM / the factor).  Constraints: the enlargement area-binned back to the original grid within "
                      "PSNR >= 32 dB of it, and its total flux within 2 % of the original's.")
    params = [{"name": "superres_tool", "label": "Enlargement", "type": "categorical", "choices": ["lanczos", "cosmic_clarity"], "default": "lanczos", "tune": True},
              {"name": "superres_factor", "label": "Factor", "type": "categorical", "choices": [2, 3, 4], "default": 2, "tune": False}]
    metrics = {"fwhm_native": {"label": "Star FWHM (original px)", "direction": "minimize"},
               "cycle_psnr": {"label": "Binned back against the original: PSNR (dB)", "direction": "maximize"},
               "flux_ratio_err": {"label": "|total flux ratio - 1|", "direction": "minimize"},
               "noise_gain": {"label": "Sky pixel noise binned back / original", "direction": "minimize"},
               "seconds": {"label": "Run time (s)", "direction": "minimize"}}
    options = []
    score = default_objective = "fwhm_native"
    min_gain = 0.05

    def static_limits(self):
        return {"cycle_psnr": ("min", 32.0), "flux_ratio_err": ("max", 0.02)}

    def prepare(self, ds, opts, device, log, cancel, study_dir):
        from .. import fitsio
        s = ds.ensure_stacked(log, cancel)
        path = s._p("process", "finish", "finish.fits")
        if not os.path.exists(path):
            raise RuntimeError("no finished image yet: export the dataset once (its finish.fits is the input)")
        img, hdr = fitsio.read_rgb(path)
        log(f"Finished image {img.shape[2]}x{img.shape[1]} from the last export")
        return {"ds": ds, "session": s, "img": np.clip(img, 0, 1).astype(np.float32), "header": hdr, "device": device, "study_dir": study_dir, "n": 0,
                "yard": OB.Yardstick(_hwc(img), keep_image=False)}

    def run(self, ctx, p, log, cancel):
        from .. import export as export_mod
        from .. import tools
        img, y = ctx["img"], ctx["yard"]
        f = int(p.get("superres_factor", 2))
        C, H, W = img.shape
        t = time.time()
        if p.get("superres_tool") == "cosmic_clarity":
            ctx["n"] += 1
            work = os.path.join(ctx["study_dir"], "work", f"sr_{ctx['n']}")
            try:
                out, _ = tools.cosmic_clarity.run("superres", img, ["--scale", str(f)], work=work, device=str(ctx["device"]), checkpoint=_checkpoint(cancel))
            finally:
                shutil.rmtree(work, ignore_errors=True)
            out = np.clip(out, 0, 1).astype(np.float32)
        else:
            out = np.stack([cv2.resize(img[c], (W * f, H * f), interpolation=cv2.INTER_LANCZOS4) for c in range(C)]).clip(0, 1).astype(np.float32)
        dt = time.time() - t
        back = np.stack([cv2.resize(out[c], (W, H), interpolation=cv2.INTER_AREA) for c in range(C)])
        mse = float(np.mean((back - img) ** 2))
        m = y.measure(_hwc(out), transform=lambda pts: np.asarray(pts, np.float64) * f + (f - 1) / 2.0)
        nb, n0 = OB.nmad((back - cv2.GaussianBlur(back, (0, 0), 2.0)).mean(0)[::5, ::5]), OB.nmad((img - cv2.GaussianBlur(img, (0, 0), 2.0)).mean(0)[::5, ::5])
        res = {"fwhm_native": m.get("fwhm"), "cycle_psnr": float(10 * np.log10(1.0 / max(mse, 1e-12))),
               "flux_ratio_err": abs(float(out.mean() / max(img.mean(), 1e-12)) - 1.0), "noise_gain": float(nb / n0) if n0 else None, "seconds": dt}
        crop = _centre_crop(_hwc(out), 900)
        return res, export_mod.display_image(np.moveaxis(crop, -1, 0), export_mod.bottom_up(ctx["header"]))


TASKS = {t.name: t for t in (StackTask(), SatelliteTask(), BackgroundTask(), RestoreTask(), StarsTask(), FinishTask(), SuperresTask())}
