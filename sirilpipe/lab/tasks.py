"""Tunable experiments of the Siril backend: what a trial runs and how it is scored.

A task declares its parameters (search space with the pipeline default of each and what it maps to:
``pipeline`` = an integration & compute option (stack_params / STACK_DEFAULTS), ``processing`` = a processing
setting (post.DEFAULTS), ``code`` = a constant of the code), its metrics (direction of improvement, whether
they need a ground truth) and two functions: ``prepare`` loads what a study shares between trials, once;
``run`` executes one trial and returns its metrics and a preview image (h, w, 3).

Real datasets are the pipeline's own sessions (read only: trials work in the study's own folder on copies of
the cached products).  Synthetic datasets (``synthetic.py``) are Seestar-format subs of an analytic sky with
an exact truth, stacked once by Siril; truth metrics compare the result with the truth registered onto the
stack's grid (astroalign on the stars, so the crop, flip and rotation of the stack do not matter).

Tasks:
  finish      the finishing chain (pro.py): stretch tool and its settings, highlights, noise reduction, palette,
              saturation, stars - scored by the finishing gates and the distance to the reference looks
  restore     the linear restoration (post.restore): gradient model, deconvolution and denoise strengths -
              scored by star FWHM, background noise, sharpness, photometry against the stack, truth
  stack       registration & integration (Session.run_stack): rejection sigmas, drizzle, frame selection -
              scored by background noise, star FWHM / roundness, frames kept, truth
  stars       star separation (post.remove_stars): tool and its settings - residual star flux / stars left in
              the starless image, change away from the stars, truth
  background  gradient removal (the restore stage's background step): model, degree, smoothing - gradient
              left in the sky, emission removed, truth
"""
from __future__ import annotations

import json
import os
import shutil
import time

import cv2
import numpy as np

from . import metrics as MX
from . import synthetic as SY


# ----------------------------------------------------------------------------- datasets
class Dataset:
    """A real session (``{"kind": "real", "folder": <subs>}``, several nights of one target joined with
    ``os.pathsep``) or a generated synthetic one (``{"kind": "synthetic", "dir": <output/lab/synthetic/name>}``)."""

    def __init__(self, spec: dict, workdir: str):
        from ..util import split_folders
        self.spec = spec
        self.kind = spec["kind"]
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
        from ..session import Session
        return Session(self.folder, workdir or self.workdir)

    def ensure_stacked(self, log, cancel):
        """The dataset's session, analysed and stacked with the pipeline defaults if needed (synthetic datasets
        only: a real session must have been stacked in the pipeline)."""
        s = self.session()
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


# ----------------------------------------------------------------------------- truth on the stack's grid
def truth_on_stack(ds: Dataset, s, stack: np.ndarray, extended: bool = True, stars: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """The true sky rendered on the reference sub's grid through the subs' mean PSF, registered onto the
    stack's grid (crop, flip, rotation, drizzle scale) by astroalign on the star positions.  Returns
    (truth (H, W, 3) in the truth's units, valid mask, info)."""
    import astroalign as aa
    truth, _ = SY.load_truth(ds.dir)
    spec = truth["spec"]
    n = len(truth["frames"])
    sc = float(s.meta.get("scale", 1.0))
    psf = SY.effective_psf(ds.dir, list(range(n)), None, sc)
    H0, W0 = int(spec["height"] * sc), int(spec["width"] * sc)
    full = SY.truth_image(ds.dir, 0, (H0, W0), sc, psf, extended=extended, stars=stars)
    full_stars = SY.truth_image(ds.dir, 0, (H0, W0), sc, psf, extended=False, stars=True)
    src = np.ascontiguousarray(full_stars.mean(-1), np.float32)
    dst = np.ascontiguousarray(_hwc(stack).mean(-1), np.float32)
    tf, (s_pts, d_pts) = aa.find_transform(src, dst, max_control_points=120, detection_sigma=5, min_area=5)
    H, W = dst.shape
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
    display_ready = False

    def describe(self) -> dict:
        return {"name": self.name, "label": self.label, "description": self.description, "params": self.params,
                "metrics": self.metrics, "options": self.options, "default_objective": self.default_objective}

    def defaults(self) -> dict:
        return {p["name"]: p["default"] for p in self.params}

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


def _processing_params(keys: list[str], tune: set[str]) -> list[dict]:
    """Task parameters taken from the UI's PARAM_SPEC so that ranges and labels follow the pipeline."""
    from ..post import DEFAULTS, EXTRA_DEFAULTS, PARAM_SPEC
    out = []
    spec = {s["key"]: s for s in PARAM_SPEC}
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
        out.append(p)
    return out


# ----------------------------------------------------------------------------- finishing
class FinishTask(Task):
    name = "finish"
    label = "Stretch & finish (finishing chain)"
    display_ready = True
    description = ("The finishing chain (pro.py) on the dataset's cached starless and star layers (or the linear / stacked image): the stretch tool and its "
                   "settings, highlights, the noise reduction of the stretched image, palette / OIII boost, local contrast, saturation and the star layer.  Scored by "
                   "the finishing gates (sky level, clipping, sky chroma, sky noise, star count / shape / colour), the nebula contrast and the distance to the "
                   "reference looks of the target.  Trial images are the finished renders (one chain, same data).  Option 'external': run NoiseXTerminator / GraXpert "
                   "on the stretched image in every trial (slower, the export's result).")
    params = _processing_params(["stretch_method", "stretch", "contrast", "hdr", "color_preservation", "luminance_denoise", "chroma_denoise", "cc_denoise", "cc_denoise_model", "cc_sharpen", "cc_stellar_amount", "cc_nonstellar_amount", "palette", "hubble_formula", "oiii_boost",
                                 "synthetic_luminance", "saturation", "local_contrast", "sharpen", "star_reduction", "star_intensity", "star_saturation", "scnr",
                                 "black_point", "brightness"],
                                {"stretch", "contrast", "hdr", "saturation", "local_contrast", "star_reduction"})
    params.append({"name": "ghs_d2", "label": "GHS pass 2 D (ghs2)", "type": "float", "low": 0.3, "high": 6.0, "default": 2.5, "tune": True, "processing": "ghs_d2"})
    params.append({"name": "star_stretch", "label": "Star stretch amount (Seti Astro)", "type": "float", "low": 2.0, "high": 8.0, "default": 5.0, "tune": False, "processing": "star_stretch"})
    metrics = {
        "gate_distance": {"label": "Gates: distance outside the limits (0 = all pass)", "direction": "minimize"},
        "gates_failed": {"label": "Gates failed (count)", "direction": "minimize"},
        "nebula_contrast": {"label": "Nebula contrast (p90 - sky, lightness)", "direction": "maximize"},
        "look_distance": {"label": "Distance to the reference looks", "direction": "minimize"},
        "sky_median": {"label": "Sky level", "direction": "minimize"},
        "sky_noise": {"label": "Sky pixel noise", "direction": "minimize"},
        "sky_chroma": {"label": "Sky chroma (max |a|, |b|)", "direction": "minimize"},
        "clipped": {"label": "Clipped fraction", "direction": "minimize"},
        "stars_coloured": {"label": "Star colour saturation", "direction": "maximize"},
        "seconds": {"label": "Run time (s)", "direction": "minimize"},
    }
    options = [{"name": "external", "label": "AI noise reduction per trial (1 = yes)", "type": "int", "default": 0},
               {"name": "max_side", "label": "Render size (px, 0 = full)", "type": "int", "default": 0}]
    default_objective = "gate_distance"

    def prepare(self, ds, opts, device, log, cancel, study_dir):
        from .. import looks, post
        s = ds.ensure_stacked(log, cancel)
        p0 = dict(post.DEFAULTS)
        max_side = int(opts.get("max_side", 0) or 0) or None
        layers = post._Layers(s.dir, p0, max_side)
        dual = post._dual_band(s.meta, layers.header)
        ref_looks, ref_agg, desc, _ = looks.reference_looks(s.meta.get("object"), dual)
        log(f"Layers: {os.path.basename(layers.src)}{' + stars' if layers.stars is not None else ''}; {desc}")
        work = os.path.join(study_dir, "work", "finish")
        os.makedirs(work, exist_ok=True)
        return {"ds": ds, "session": s, "layers": layers, "dual": dual, "looks": ref_looks, "work": work, "device": device, "opts": opts, "header": layers.header}

    def run(self, ctx, p, log, cancel):
        from .. import finish, gates, looks, post, pro
        layers = ctx["layers"]
        prm = {**post.DEFAULTS, **p}
        if cancel.is_set():
            raise RuntimeError("cancelled")
        t = time.time()
        fin = pro.Finisher(ctx["work"], ctx["header"], prm, ctx["dual"], external=bool(int(ctx["opts"].get("external", 0))), device=str(ctx["device"]),
                           checkpoint=(lambda: (_ for _ in ()).throw(RuntimeError("cancelled")) if cancel.is_set() else None), keep_pictures=False)
        img, info = fin.run(layers.base, layers.stars)
        dt = time.time() - t
        sm = layers.starnet.get("stars") or {}
        level = float(pro.mtf_balance(pro.star_stretch_m(float(prm.get("star_stretch", pro.STAR_STRETCH))), 5.0 * float(sm["sky_noise"]))) if sm.get("sky_noise") else None
        g = gates.compute(img, starless=info.get("starless") if layers.stars is not None and prm["star_reduction"] < 1.0 else None, linear_star_count=sm.get("count"), star_level=level)
        dist = 0.0
        for k, v in g.items():
            if not isinstance(v, dict) or v.get("pass") or v.get("value") is None:
                continue
            lim, val = v.get("limit"), v.get("value")
            if isinstance(val, dict):                             # stars_count -> its deviation; sky_chroma -> max(|a|, |b|)
                val = val.get("deviation") if "deviation" in val else (max(x for x in val.values() if isinstance(x, (int, float))) if any(isinstance(x, (int, float)) for x in val.values()) else None)
            if val is None:
                continue
            if isinstance(lim, (list, tuple)):
                lo, hi = lim
                dist += (lo - val) / lo if val < lo else (val - hi) / hi
            elif isinstance(lim, (int, float)) and val is not None and lim:
                dist += abs(val - lim) / lim
        lum = finish.luminance(img)
        hs = pro._hist_stats(lum)
        mask, _ = pro.nebula_mask(lum, hs["mode"])
        neb = float(np.percentile(lum[mask], 90) - g["sky_median"]["value"]) if mask.any() else 0.0
        out = {"gate_distance": float(dist), "gates_failed": len(g.get("failed", [])), "nebula_contrast": neb, "sky_median": g["sky_median"]["value"],
               "sky_noise": g.get("sky_pixel_noise", {}).get("value"), "sky_chroma": max(g["sky_chroma"]["value"].values()) if isinstance(g.get("sky_chroma", {}).get("value"), dict) else None,
               "clipped": g["clipped"]["value"], "stars_coloured": g.get("stars_coloured", {}).get("value"), "seconds": dt, "failed": g.get("failed")}
        if ctx["looks"]:
            st = looks.look_stats(_hwc(img))
            out["look_distance"] = looks.score(st, ctx["looks"], looks.object_weight(st))[0]
        from .. import export as export_mod
        return out, export_mod.display_image(img, export_mod.bottom_up(ctx["header"]))        # (H, W, 3) north-up like the exports


# ----------------------------------------------------------------------------- linear restoration
class RestoreTask(Task):
    name = "restore"
    label = "Linear restoration (restore stage)"
    description = ("The linear stage (post.restore) on a copy of the stack: gradient model and smoothing, deconvolution strength (BlurXTerminator / Siril RL), "
                   "denoise strength (NoiseXTerminator / GraXpert), colour calibration.  Scored on the result: star FWHM, background noise, Krotkov sharpness, aperture "
                   "photometry against the stack (bright / faint bias) and, on synthetic data, the truth.  Each trial runs the whole linear stage (1-2 min).")
    params = _processing_params(["bg_method", "bg_smoothing", "bg_degree", "bg_correction", "deconvolution", "denoise", "white_balance"], {"deconvolution", "denoise", "bg_smoothing"})
    params.append({"name": "deconv_method", "label": "Deconvolution tool (objects)", "type": "categorical", "choices": ["auto", "bxt", "graxpert", "rl", "none"], "default": "auto", "tune": False, "pipeline": "deconv_method"})
    params.append({"name": "deconv_stars_method", "label": "Deconvolution tool (stars)", "type": "categorical", "choices": ["auto", "bxt", "graxpert", "rl", "none"], "default": "auto", "tune": False, "pipeline": "deconv_stars_method"})
    params.append({"name": "denoise_method", "label": "Noise reduction tool", "type": "categorical", "choices": ["auto", "nxt", "graxpert", "siril", "none"], "default": "auto", "tune": False, "pipeline": "denoise_method"})
    metrics = {
        "fwhm": {"label": "Star FWHM (px)", "direction": "minimize"},
        "bg_noise": {"label": "Background noise", "direction": "minimize"},
        "sharpness": {"label": "Sharpness S_F (Krotkov)", "direction": "maximize"},
        "dm_bright_abs": {"label": "|photometry bias| bright stars (mag)", "direction": "minimize"},
        "dm_faint_abs": {"label": "|photometry bias| faint stars (mag)", "direction": "minimize"},
        "seconds": {"label": "Run time (s)", "direction": "minimize"},
        **TRUTH_METRICS,
    }
    options = [{"name": "sigma_eval", "label": "Truth comparison resolution σ (px)", "type": "float", "default": 1.0}]
    default_objective = "bg_noise"

    def prepare(self, ds, opts, device, log, cancel, study_dir):
        s = ds.ensure_stacked(log, cancel)
        stack = _read(s._p("stack.fits"))
        return {"ds": ds, "session": s, "stack": stack, "study_dir": study_dir, "n": 0, "device": device, "opts": opts}

    def run(self, ctx, p, log, cancel):
        from .. import post
        s, ds = ctx["session"], ctx["ds"]
        ctx["n"] += 1
        wd = os.path.join(ctx["study_dir"], "work", f"restore_{ctx['n']}")
        _copy_products(s.dir, wd, ("stack.fits", "stack_meta.json", "skyref.npz"))
        prm = {**post.DEFAULTS, **{k: v for k, v in p.items() if k not in ("deconv_method", "deconv_stars_method", "denoise_method")}}
        sp = {**(s.meta.get("params") or {}), "deconv_method": p.get("deconv_method", "auto"), "deconv_stars_method": p.get("deconv_stars_method", ""),
              "denoise_method": p.get("denoise_method", "auto"), "device": str(ctx["device"])}
        t = time.time()
        try:
            def checkpoint():
                if cancel.is_set():
                    raise RuntimeError("cancelled")
            post.restore(wd, s.meta, sp, prm, None, checkpoint)
            dt = time.time() - t
            lin = _read(os.path.join(wd, "linear.fits"))
            est = _hwc(lin)
            sat = 0.98
            out = {"fwhm": MX.star_fwhm(est, sat), "bg_noise": MX.background_noise(est), "sharpness": float(np.mean([MX.sharpness(est[..., c]) for c in range(3)])), "seconds": dt}
            try:
                coadd = _hwc(ctx["stack"])
                mc, dm = MX.photometry(coadd.mean(-1), est.mean(-1) * (np.median(coadd) / max(np.median(est), 1e-9)), thresh=5 * MX.sky_sigma(coadd.mean(-1)))
                if len(mc) >= 4:
                    bright = mc <= np.percentile(mc, 50)
                    out["dm_bright_abs"], out["dm_faint_abs"] = abs(float(np.median(dm[bright]))), abs(float(np.median(dm[~bright])))
            except Exception:                                 # noqa: BLE001
                pass
            if ds.synthetic:
                out.update(_truth_scores(ds, s, lin, float(ctx["opts"].get("sigma_eval", 1.0))))
            return out, est
        finally:
            shutil.rmtree(wd, ignore_errors=True)


# ----------------------------------------------------------------------------- registration & integration
class StackTask(Task):
    name = "stack"
    label = "Registration & integration"
    description = ("Re-stacks the dataset with each trial's settings in the study's own session folder (Siril; the registration is reused when its settings are unchanged): "
                   "rejection sigmas, frame selection sensitivity, resampling and drizzle scale, banding correction.  Real data: background noise and star FWHM / roundness on "
                   "the stack; synthetic data also against the truth seen through the subs' mean PSF.")
    params = [
        {"name": "sigma_low", "label": "Rejection σ low", "type": "float", "low": 1.5, "high": 8.0, "default": 4.0, "tune": True, "pipeline": "sigma_low"},
        {"name": "sigma_high", "label": "Rejection σ high", "type": "float", "low": 1.5, "high": 8.0, "default": 3.0, "tune": True, "pipeline": "sigma_high"},
        {"name": "sensitivity", "label": "Frame rejection sensitivity", "type": "float", "low": 0.5, "high": 2.0, "default": 1.0, "tune": False, "pipeline": "sensitivity"},
        {"name": "mode", "label": "Resampling", "type": "categorical", "choices": ["auto", "drizzle", "demosaic"], "default": "auto", "tune": False, "pipeline": "mode"},
        {"name": "scale", "label": "Drizzle scale", "type": "categorical", "choices": [1.0, 1.5, 2.0], "default": 1.0, "tune": False, "pipeline": "scale"},
        {"name": "pattern_correction", "label": "Row banding correction (seqfixbanding)", "type": "bool", "default": True, "tune": False, "pipeline": "pattern_correction"},
        {"name": "crop_threshold", "label": "Crop: minimum frame coverage", "type": "float", "low": 0.1, "high": 1.0, "default": 0.3, "tune": False, "pipeline": "crop_threshold"},
    ]
    metrics = {
        "bg_noise": {"label": "Background noise", "direction": "minimize"},
        "fwhm": {"label": "Star FWHM (px)", "direction": "minimize"},
        "roundness": {"label": "Star ellipticity", "direction": "minimize"},
        "n_frames": {"label": "Frames stacked", "direction": "maximize"},
        "seconds": {"label": "Run time (s)", "direction": "minimize"},
        **TRUTH_METRICS,
    }
    options = [{"name": "crop", "label": "Scored crop (px per side)", "type": "int", "default": 1024},
               {"name": "sigma_eval", "label": "Truth comparison resolution σ (px)", "type": "float", "default": 1.0}]
    default_objective = "bg_noise"

    def prepare(self, ds, opts, device, log, cancel, study_dir):
        from ..session import Session
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
        return {"ds": ds, "session": s, "opts": opts, "device": device}

    def run(self, ctx, p, log, cancel):
        from .. import gates
        s, ds = ctx["session"], ctx["ds"]
        sp = {k: p[k] for k in ("sigma_low", "sigma_high", "sensitivity", "mode", "scale", "pattern_correction", "crop_threshold")}
        t = time.time()
        meta = s.run_stack(sp, _prog(log, 20))
        dt = time.time() - t
        st = _read(s._p("stack.fits"))
        est = _hwc(st)
        c = int(ctx["opts"].get("crop", 1024))
        H, W = est.shape[:2]
        y0, x0 = max(0, H // 2 - c // 2), max(0, W // 2 - c // 2)
        x = np.ascontiguousarray(est[y0:y0 + c, x0:x0 + c])
        lum = x.mean(-1)
        noise = float(1.4826 * np.median(np.abs(lum - np.median(lum))))
        ms = gates.measure_stars(np.moveaxis(x, -1, 0), noise * 5.0) if noise > 0 else {"ellipticity": None}
        out = {"bg_noise": MX.background_noise(x), "fwhm": MX.star_fwhm(x, 0.98), "roundness": ms.get("ellipticity"), "n_frames": meta["n_frames"], "seconds": dt}
        if ds.synthetic:
            out.update(_truth_scores(ds, s, st, float(ctx["opts"].get("sigma_eval", 1.0))))
        return out, x


# ----------------------------------------------------------------------------- star separation
class StarsTask(Task):
    name = "stars"
    label = "Star separation"
    description = ("Star removal (post.remove_stars: StarXTerminator, else StarNet2; Cosmic Clarity DarkStar) on the dataset's linear image, scored against the stars the detector finds in it: "
                   "their flux left in the starless image (local-background apertures), the fraction still detected, and the change away from every star (damage to "
                   "nebula and sky, × the image's noise).  Synthetic data: the starless image against the true sky without stars.")
    params = [
        {"name": "star_removal", "label": "Tool", "type": "categorical", "choices": ["auto", "sxt", "starnet", "darkstar"], "default": "auto", "tune": True, "processing": "star_removal"},
        {"name": "starnet_stride", "label": "StarNet2 stride", "type": "categorical", "choices": [64, 128, 256], "default": 256, "tune": False, "processing": "starnet_stride"},
    ]
    metrics = {
        "residual_flux": {"label": "Star flux left in the starless image (fraction)", "direction": "minimize"},
        "residual_stars": {"label": "Stars still detected in the starless image (fraction)", "direction": "minimize"},
        "offstar_change": {"label": "Change away from the stars (× noise)", "direction": "minimize"},
        "seconds": {"label": "Run time (s)", "direction": "minimize"},
        **{k: v for k, v in TRUTH_METRICS.items() if k in ("truth_nrmse", "truth_faint_nrmse", "truth_ssim", "truth_psnr")},
    }
    options = [{"name": "sigma_eval", "label": "Truth comparison resolution σ (px)", "type": "float", "default": 1.0}]
    default_objective = "residual_flux"

    def prepare(self, ds, opts, device, log, cancel, study_dir):
        from scipy.spatial import cKDTree  # noqa: F401
        s = ds.ensure_stacked(log, cancel)
        src = s._p("linear.fits") if os.path.exists(s._p("linear.fits")) else s._p("stack.fits")
        lin = _read(src)
        L = np.ascontiguousarray(_hwc(lin).mean(-1), np.float32)
        import sep as _sep
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
        log(f"{int(keep.sum())} reference stars (FWHM {fw:.2f} px), {int(off.mean() * 100)} % of the image away from every star")
        return {"ds": ds, "session": s, "lin": lin, "L0": L, "src": os.path.basename(src), "ref_x": objs["x"][keep], "ref_y": objs["y"][keep], "ref_f": f0[keep], "r_ap": r_ap,
                "fw": fw, "off": off, "noise": float(bk.globalrms), "study_dir": study_dir, "n": 0, "opts": opts, "device": device}

    def run(self, ctx, p, log, cancel):
        import sep as _sep
        from scipy.spatial import cKDTree
        from .. import post
        s, ds = ctx["session"], ctx["ds"]
        ctx["n"] += 1
        wd = os.path.join(ctx["study_dir"], "work", f"stars_{ctx['n']}")
        _copy_products(s.dir, wd, ("stack.fits", "stack_meta.json", "linear.fits", "restore_meta.json"))
        prm = {**post.DEFAULTS, **p}
        sp = {**(s.meta.get("params") or {}), "device": str(ctx["device"])}
        t = time.time()
        try:
            def checkpoint():
                if cancel.is_set():
                    raise RuntimeError("cancelled")
            meta = post.remove_stars(wd, s.meta, prm, sp, None, checkpoint)
            dt = time.time() - t
            if not os.path.exists(os.path.join(wd, "starless.fits")):
                raise RuntimeError(f"no starless image: {meta.get('tool')} {meta.get('reason') or ''}")
            starless = _read(os.path.join(wd, "starless.fits"))
            L1 = np.ascontiguousarray(_hwc(starless).mean(-1), np.float32)
            r = ctx["r_ap"]
            f1, _, _ = _sep.sum_circle(L1, ctx["ref_x"], ctx["ref_y"], r, bkgann=(2.5 * r, 4 * r), subpix=5)
            out = {"residual_flux": float(np.clip(f1, 0, None).sum() / ctx["ref_f"].sum()), "seconds": dt, "tool": meta.get("tool")}
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


class BackgroundTask(Task):
    name = "background"
    label = "Gradient removal"
    description = ("The restore stage's gradient model on the stack: the NSNS sky survey reference (when plate-solved and covered), GraXpert's AI background extraction with its "
                   "smoothing, or Siril subsky (polynomial degree / RBF smoothing), subtraction or division.  Real data: the gradient left in the sky (range of a robust "
                   "quadratic over the star-free sky tiles, × the pixel noise) and the extended emission removed (the fraction of the emission tiles' signal the model took "
                   "away).  Sky and emission tiles are set once against a robust plane through the faintest tiles.  Synthetic data: against the truth (constant offset fitted).")
    params = _processing_params(["bg_method", "bg_correction", "bg_degree", "bg_smoothing"], {"bg_method", "bg_smoothing", "bg_degree"})
    metrics = {
        "sky_flatness": {"label": "Gradient left in the sky (× pixel noise)", "direction": "minimize"},
        "signal_removed": {"label": "Extended emission removed (fraction)", "direction": "minimize"},
        "seconds": {"label": "Run time (s)", "direction": "minimize"},
        **{k: v for k, v in TRUTH_METRICS.items() if k in ("truth_nrmse", "truth_faint_nrmse", "truth_ssim", "truth_psnr")},
    }
    options = [{"name": "sigma_eval", "label": "Truth comparison resolution σ (px)", "type": "float", "default": 4.0}]
    default_objective = "sky_flatness"

    def prepare(self, ds, opts, device, log, cancel, study_dir):
        s = ds.ensure_stacked(log, cancel)
        stack = _read(s._p("stack.fits"))
        img = _hwc(stack)
        yard = _sky_yardstick(img)
        log(f"{len(yard['tiles'])} scoring tiles: {int(yard['sky'].sum())} sky, {int(yard['emission'].sum())} emission")
        return {"ds": ds, "session": s, "stack": stack, "img": img, "yard": yard, "study_dir": study_dir, "n": 0, "opts": opts, "device": device}

    def run(self, ctx, p, log, cancel):
        from .. import fitsio, post
        s, ds, yard = ctx["session"], ctx["ds"], ctx["yard"]
        ctx["n"] += 1
        wd = os.path.join(ctx["study_dir"], "work", f"bg_{ctx['n']}")
        _copy_products(s.dir, wd, ("stack.fits", "stack_meta.json", "skyref.npz"))
        prm = {**post.DEFAULTS, **p}
        t = time.time()
        try:
            r = post.Restorer(wd, s.meta, {"device": str(ctx["device"])}, prm, None, (lambda: (_ for _ in ()).throw(RuntimeError("cancelled")) if cancel.is_set() else None))
            r.arr, r.header = fitsio.read_rgb(os.path.join(wd, "stack.fits"))
            r.arr, _ = post.normalise_scale(r.arr)
            r.background()
            dt = time.time() - t
            res = _hwc(r.arr)
            step = next((x for x in r.steps.records if x["step"] == "gradient"), {})
            T1 = _tile_medians(res, yard)
            sky = yard["sky"]
            # the gradient left: range of a robust quadratic over the sky tiles, in units of the pixel noise
            cx, cy = yard["cx"][sky], yard["cy"][sky]
            A = np.stack([np.ones(sky.sum()), cx, cy, cx * cx, cx * cy, cy * cy], 1)
            coef, *_ = np.linalg.lstsq(A, T1[sky], rcond=None)
            fitv = A @ coef
            out = {"sky_flatness": float(np.ptp(fitv) / max(yard["noise"], 1e-12)), "seconds": dt, "method_used": step.get("tool"), "note": step.get("note")}
            em = yard["emission"]
            if em.any() and sky.any():
                before = yard["med0"][em] - np.median(yard["med0"][sky])
                after = T1[em] - np.median(T1[sky])
                out["signal_removed"] = float(np.clip(1.0 - after.sum() / max(before.sum(), 1e-12), -1.0, 1.0))
            if ds.synthetic:
                out.update(_truth_scores(ds, s, r.arr, float(ctx["opts"].get("sigma_eval", 4.0)), stars_in_result=True, fit_plane=False))
            return out, res
        finally:
            shutil.rmtree(wd, ignore_errors=True)


TASKS = {t.name: t for t in (FinishTask(), RestoreTask(), StackTask(), StarsTask(), BackgroundTask())}
