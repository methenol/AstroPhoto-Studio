"""Tunable experiments: what a trial runs and how it is scored.

A task declares its parameters (search space, with the pipeline default of each and what it maps
to: ``pipeline``, an integration & compute option (STACK_DEFAULTS); ``processing``, a processing
setting (postprocess.DEFAULTS); ``code``, a constant of the code, changed there), its metrics (with the direction of improvement and whether
they need a ground truth) and two functions: ``prepare`` loads everything a study shares
between trials, once; ``run`` executes one trial and returns its metrics and a preview.

Real datasets are the pipeline's own sessions (read only, except the caches the pipeline
itself would write, such as the prepared ImageMM exposures); stacking trials work in the
study's own folder.  Synthetic datasets (``synthetic.py``) are scored against their exact
truth as well.
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
import shutil
import time

import cv2
import numpy as np
import torch

from . import metrics as MX
from . import synthetic as SY


# ----------------------------------------------------------------------------- datasets
class Dataset:
    """A real session (``{"kind": "real", "folder": <subs>}``) or a generated synthetic one
    (``{"kind": "synthetic", "dir": <output/lab/synthetic/name>}``)."""

    def __init__(self, spec: dict, workdir: str):
        self.spec = spec
        self.kind = spec["kind"]
        if self.kind == "synthetic":
            self.dir = spec["dir"]
            self.folder = os.path.join(self.dir, "subs")
            self.workdir = os.path.join(self.dir, "work")
        else:
            self.dir = None
            self.folder = spec["folder"]
            self.workdir = workdir
        self.name = spec.get("name") or os.path.basename(os.path.normpath(self.dir or self.folder))

    @property
    def synthetic(self) -> bool:
        return self.kind == "synthetic"

    def session(self, workdir: str | None = None):
        from ..pipeline import Session
        return Session(self.folder, workdir or self.workdir)

    def ensure_stacked(self, log, cancel) -> "object":
        """The dataset's session, analysed and stacked with the pipeline defaults if needed
        (synthetic datasets only: a real session must have been stacked in the pipeline)."""
        s = self.session()
        s.cancel_flag = cancel
        if s.status()["stacked"]:
            return s
        if not self.synthetic:
            raise RuntimeError(f"{self.folder} has not been stacked: run 'Register & integrate' first")
        from ..pipeline import STACK_DEFAULTS
        log("Analysing and stacking the synthetic subs (once per dataset)")
        s.run_analysis(1.0, _prog(log))
        s.run_stack({**STACK_DEFAULTS, **self.spec.get("stack", {})}, _prog(log))
        return s


def _prog(log, every: int = 25):
    def f(i, n, msg):
        if i == 1 or i == n or i % every == 0:
            log(msg)
    return f


def _device(name: str):
    from ..denoise import pick_device
    return pick_device(name or "auto")


def stretch(img: np.ndarray, ref: dict | None = None) -> tuple[np.ndarray, dict]:
    """8-bit preview in an asinh stretch; ``ref`` (from the first trial) keeps it identical
    across the trials of a study."""
    if ref is None:
        L = img.mean(-1)
        med = float(np.median(L))
        sig = float(1.4826 * np.median(np.abs(L - med))) or 1e-6
        hi = float(np.percentile(L, 99.8))
        ref = {"bg": med, "k": 2 * sig, "top": float(np.arcsinh(max(hi - med, 1e-6) / (2 * sig)))}
    y = np.arcsinh(np.maximum(img - ref["bg"], 0) / ref["k"]) / ref["top"]
    return (np.clip(y, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8), ref


def _auto_window(ref: np.ndarray, size: int, where: str = "auto") -> tuple[int, int, int, int]:
    """A size x size window of the reference grid: the brightest extended structure (stars
    removed by a morphological opening) or the centre."""
    H, W = ref.shape[:2]
    size = min(size, H - 140, W - 140)
    if where == "center":
        cy, cx = H // 2, W // 2
    else:
        L = ref.mean(-1)
        Lo = cv2.dilate(cv2.erode(L, np.ones((15, 15))), np.ones((15, 15)))
        sm = cv2.blur(Lo, (size, size))
        h2 = size // 2 + 64
        sm[:h2], sm[-h2:], sm[:, :h2], sm[:, -h2:] = -np.inf, -np.inf, -np.inf, -np.inf
        cy, cx = np.unravel_index(np.argmax(sm), sm.shape)
    y0, x0 = int(cy - size // 2), int(cx - size // 2)
    return y0, y0 + size, x0, x0 + size


@contextlib.contextmanager
def _overriding(table: dict, values: dict):
    """A module's table of constants (``postprocess.STAR_DETECT``, ...) with the keys it shares with
    ``values`` set for the duration (studies run in their own process: nothing else sees it)."""
    old = dict(table)
    table.update({k: v for k, v in values.items() if k in table})
    try:
        yield
    finally:
        table.clear()
        table.update(old)


def _mean(v):
    v = [q for q in (v or []) if q is not None]
    return float(np.mean(v)) if v else float("nan")


# ----------------------------------------------------------------------------- base
class Task:
    name = ""
    label = ""
    description = ""
    params: list[dict] = []
    metrics: dict[str, dict] = {}
    options: list[dict] = []
    default_objective = ""

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


# ----------------------------------------------------------------------------- ImageMM
class ImageMMTask(Task):
    name = "imagemm"
    label = "ImageMM restoration"
    description = ("ImageMM (arXiv:2501.03002) on a window of the reference grid, restored from the even subs; "
                   "scored on the odd subs through each one's own PSF (held-out χ² excess, 0 = perfect), "
                   "with the paper's metrics, and against the exact truth on synthetic data. The held-out score "
                   "predicts data through the measured PSFs, so it rewards fitting the data under the pipeline's "
                   "own model; on synthetic data the truth metrics measure closeness to the real sky.")
    params = [
        {"name": "robust", "label": "Robust (Huber, Algorithm 3)", "type": "bool", "default": True, "tune": True,
         "pipeline": "imagemm_robust"},
        {"name": "delta", "label": "Huber δ", "type": "float", "low": 0.5, "high": 6.0, "default": 2.0, "tune": True,
         "pipeline": "imagemm_delta"},
        {"name": "kappa", "label": "Update clip κ", "type": "float", "low": 1.2, "high": 6.0, "default": 2.0,
         "tune": False, "pipeline": "imagemm_kappa"},
        {"name": "psf_model", "label": "PSF model", "type": "categorical", "choices": ["empirical", "moffat"],
         "default": "empirical", "tune": True, "pipeline": "imagemm_psf"},
        {"name": "n_groups", "label": "Seeing groups (0 = every sub)", "type": "int", "low": 0, "high": 32,
         "default": 0, "tune": True, "pipeline": "imagemm_groups"},
        {"name": "accelerate", "label": "Biggs–Andrews acceleration", "type": "bool", "default": True, "tune": True,
         "pipeline": "imagemm_accelerate"},
        {"name": "stop", "label": "Stopping rule", "type": "categorical", "choices": ["flux", "c15", "elementwise"],
         "default": "flux", "tune": False, "pipeline": "imagemm_stop"},
        {"name": "epsilon", "label": "Tolerance ε", "type": "float", "low": 1e-8, "high": 1e-3, "log": True,
         "default": 1e-4, "tune": False, "pipeline": "imagemm_epsilon"},
        {"name": "max_iters", "label": "Max iterations", "type": "int", "low": 50, "high": 5000, "log": True,
         "default": 2000, "tune": False, "pipeline": "imagemm_max_iters"},
        {"name": "r", "label": "Super-resolution r", "type": "categorical", "choices": [1, 2], "default": 1,
         "tune": False, "pipeline": "imagemm_r"},
        {"name": "sigma", "label": "g_σ of Eq. 11 (0 = the paper's: 1 at r = 1, 1.1 at r = 2)", "type": "float", "low": 0.0,
         "high": 1.6, "default": 0.0, "tune": False, "pipeline": "imagemm_sigma"},
    ]
    metrics = {
        "heldout_src": {"label": "Held-out χ² excess, sources", "direction": "minimize"},
        "heldout_sky": {"label": "Held-out χ² excess, sky", "direction": "minimize"},
        "seconds": {"label": "Run time (s)", "direction": "minimize"},
        "iterations": {"label": "Iterations", "direction": "minimize"},
        "S_F": {"label": "Sharpness S_F", "direction": "maximize"},
        "sigma_sky": {"label": "σ_sky", "direction": "minimize"},
        "ssim_vs_coadd": {"label": "SSIM vs coadd", "direction": "maximize"},
        **TRUTH_METRICS,
    }
    options = [
        {"name": "window", "label": "Window (px, reference grid)", "type": "int", "default": 256},
        {"name": "where", "label": "Window position", "type": "categorical", "choices": ["auto", "center"],
         "default": "auto"},
        {"name": "sigma_eval", "label": "Truth comparison resolution σ (px)", "type": "float", "default": 1.0},
    ]
    default_objective = "heldout_src"

    def prepare(self, ds, opts, device, log, cancel, study_dir):
        s = ds.ensure_stacked(log, cancel)
        log("Loading the prepared exposures (the first time prepares every sub)")
        es = s.exposure_set(progress=_prog(log, 20))
        ref = es.ref - es.sky_ref
        window = _auto_window(ref, int(opts.get("window", 256)), opts.get("where", "auto"))
        idx = es.usable()
        y0, y1, x0, x1 = window
        ctx = {"es": es, "window": window, "idx_a": idx[0::2], "idx_b": idx[1::2],
               "smask": es.smask[y0:y1, x0:x1], "device": device, "ds": ds, "opts": opts, "truth": {}}
        Ya, Va, Ma = es.windows(ctx["idx_a"], *window)
        w = np.where(Ma > 0, 1 / Va, 0)
        ctx["coadd"] = np.moveaxis((w * Ya).sum(0) / np.maximum(w.sum(0), 1e-30), 0, -1)
        if ds.synthetic:
            ctx["ref_idx"] = s.analysis["ref_idx"]
        log(f"Window y {y0}:{y1} x {x0}:{x1}; {len(ctx['idx_a'])} subs restored, {len(ctx['idx_b'])} held out")
        return ctx

    def _truth(self, ctx, r, sigma):
        key = (r, sigma)
        if key not in ctx["truth"]:
            ds, es = ctx["ds"], ctx["es"]
            psf = SY.Gaussian(sigma) if sigma else None
            H, W = es.H0, es.W0
            full = SY.truth_image(ds.dir, ctx["ref_idx"], (H * r, W * r), float(r), psf, device=ctx["device"]) * r * r
            stars = SY.truth_image(ds.dir, ctx["ref_idx"], (H * r, W * r), float(r), psf, device=ctx["device"],
                                   extended=False) * r * r
            y0, y1, x0, x1 = ctx["window"]
            sl = (slice(y0 * r, y1 * r), slice(x0 * r, x1 * r))
            pos = SY.star_positions(ds.dir, ctx["ref_idx"], float(r))
            pos = {"x": pos["x"] - x0 * r, "y": pos["y"] - y0 * r, "flux": pos["flux"] * r * r}
            ctx["truth"][key] = (full[sl], stars[sl], pos)
        return ctx["truth"][key]

    def run(self, ctx, p, log, cancel):
        from .. import imagemm as M
        es, window, dev = ctx["es"], ctx["window"], ctx["device"]
        r = int(p["r"])
        sigma = float(p["sigma"]) if float(p["sigma"]) > 0 else (1.1 if r > 1 else 1.0)
        # the held-out subs are always predicted through their measured (empirical) PSFs, the
        # same yardstick for every trial whatever PSF model the trial restores with
        kern = None
        y0, y1, x0, x1 = window
        at = ((y0 + y1 - 1) / 2, (x0 + x1 - 1) / 2)             # the field-dependent PSFs at the window
        kb = es.kernels(ctx["idx_b"], "empirical", at=at)
        if r > 1 or sigma:
            log(f"Eq. 11 kernels (r = {r}, σ = {sigma})")
            kern, _ = M.superresolved_kernels(es.kernels(ctx["idx_a"], p["psf_model"], at=at), r, sigma, device=dev)
            kb, _ = M.superresolved_kernels(kb, r, sigma, device=dev)
        n_groups = min(int(p["n_groups"]), len(ctx["idx_a"]))
        def it_log(k, c):
            if cancel.is_set():
                raise RuntimeError("cancelled")
            if k % 100 == 0:
                log(f"iteration {k}, criterion {c:.2e}")
        t = time.time()
        x, info = M.restore_cutout(es, *window, idx=ctx["idx_a"], r=r, kernels=kern, robust=bool(p["robust"]),
                                   psf_model=p["psf_model"], n_groups=n_groups, device=dev,
                                   delta=float(p["delta"]), kappa=float(p["kappa"]), epsilon=float(p["epsilon"]),
                                   stop=p["stop"], max_iters=int(p["max_iters"]), accelerate=bool(p["accelerate"]),
                                   log=it_log)
        dt = time.time() - t
        ch = MX.heldout_chi2(es, ctx["idx_b"], x, r, kb, window, ctx["smask"], dev)   # on the exposure grid
        out = {"heldout_src": _mean(ch["src"]), "heldout_sky": _mean(ch["sky"]), "heldout_src_rgb": ch["src"],
               "heldout_sky_rgb": ch["sky"], "seconds": dt, "iterations": int(info["iterations"]),
               "converged": bool(info["converged"]), "n_groups_used": n_groups}
        x1 = x if r == 1 else x.reshape(x.shape[0] // r, r, x.shape[1] // r, r, 3).mean((1, 3))
        out.update(MX.paper_metrics(x1, ctx["coadd"]))
        if ctx["ds"].synthetic:
            full, stars, pos = self._truth(ctx, r, sigma)
            valid = np.ones(full.shape[:2], bool)
            e = int(4 * r)
            valid[:e], valid[-e:], valid[:, :e], valid[:, -e:] = False, False, False, False
            unsat = (ctx["coadd"].max(-1) < 0.5 * es.sat)
            unsat = cv2.erode(unsat.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            if r > 1:
                unsat = np.repeat(np.repeat(unsat, r, 0), r, 1)
            m = MX.truth_metrics(x, full, valid & unsat, star_truth=stars, stars=pos,
                                 sigma_eval=float(ctx["opts"].get("sigma_eval", 1.0)) * r)
            out.update(_truth_row(m))
        return out, x1


# ----------------------------------------------------------------------------- N2N denoiser
def _deep_crop(cov: np.ndarray, size: int):
    s = min(size, cov.shape[0] - 16, cov.shape[1] - 16)
    s -= s % 16
    c = cv2.blur(cov, (max(s // 4, 3), max(s // 4, 3)))
    c[: s // 2], c[-s // 2:], c[:, : s // 2], c[:, -s // 2:] = 0, 0, 0, 0
    cy, cx = np.unravel_index(np.argmax(c), c.shape)
    y0, x0 = max(0, cy - s // 2), max(0, cx - s // 2)
    return slice(y0, y0 + s), slice(x0, x0 + s)


class DenoiseTask(Task):
    name = "denoise"
    label = "Noise2Noise denoiser"
    description = ("The pipeline's Noise2Noise U-Net trained on the two half-stacks (two of every three 256 px "
                   "bands), applied to half A and scored against half B on the held-out bands "
                   "(error / half-stack noise variance: 1 = no better than raw, lower is better), "
                   "and against the truth on synthetic data.")
    params = [
        {"name": "iters", "label": "Training steps", "type": "int", "low": 250, "high": 8000, "log": True,
         "default": 2000, "tune": True, "pipeline": "denoise_iters"},
        {"name": "max_lr", "label": "Peak learning rate (one-cycle)", "type": "float", "low": 1e-4, "high": 5e-3,
         "log": True, "default": 1e-3, "tune": True},
        {"name": "patch", "label": "Patch size", "type": "categorical", "choices": [64, 96, 128, 192, 256],
         "default": 128, "tune": True},
        {"name": "batch", "label": "Batch size", "type": "int", "low": 4, "high": 32, "default": 16, "tune": False},
        {"name": "base", "label": "U-Net width (first level)", "type": "categorical", "choices": [16, 24, 32, 48],
         "default": 32, "tune": False},
        {"name": "tta", "label": "Self-ensemble (rotations / flips)", "type": "categorical", "choices": [1, 8],
         "default": 8, "tune": False},
    ]
    metrics = {
        "heldout_lin": {"label": "Held-out error, linear (× noise var.)", "direction": "minimize"},
        "heldout_str": {"label": "Held-out error, stretched (× noise var.)", "direction": "minimize"},
        "seconds": {"label": "Run time (s)", "direction": "minimize"},
        **TRUTH_METRICS,
    }
    options = [{"name": "crop", "label": "Crop (px per side, stack grid)", "type": "int", "default": 1024},
               {"name": "sigma_eval", "label": "Truth comparison resolution σ (px)", "type": "float", "default": 1.0}]
    default_objective = "heldout_str"

    def prepare(self, ds, opts, device, log, cancel, study_dir):
        from ..denoise import Stabiliser
        from ..pipeline import _load_fits
        s = ds.ensure_stacked(log, cancel)
        cov = _load_fits(s._p("coverage.fits"))
        sl = _deep_crop(cov, int(opts.get("crop", 1024)))
        a, b = _load_fits(s._p("half_a.fits"))[sl], _load_fits(s._p("half_b.fits"))[sl]
        full = _load_fits(s._p("stack.fits"))[sl]
        cov = cov[sl]
        train, test = MX.split_masks(a.shape)
        good = cov >= 0.6 * np.percentile(cov, 90)
        sat = s.meta.get("saturation", 63471.0)
        unsat = cv2.erode((full.max(-1) < 0.5 * sat).astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
        stab = Stabiliser(a, b)
        ctx = {"a": a, "b": b, "full": full, "train": train & good, "test": test, "valid": good & unsat,
               "stab": stab, "var": MX.NoiseModel(a, b).v, "device": device, "ds": ds, "opts": opts,
               "ga": stab.fwd(a), "gb": stab.fwd(b), "sl": sl, "scale": float(s.meta.get("scale", 1.0))}
        if ds.synthetic:
            fr = [i for i, f in enumerate(s.analysis["frames"]) if f["accepted"]]
            sc = ctx["scale"]
            psf = SY.effective_psf(ds.dir, fr, [s.analysis["frames"][i]["weight"] for i in fr], sc)
            H, W = s.meta["shape"][:2]
            ref = s.analysis["ref_idx"]
            ctx["truth"] = SY.truth_image(ds.dir, ref, (H, W), sc, psf, device=device)[sl]
            ctx["truth_stars"] = SY.truth_image(ds.dir, ref, (H, W), sc, psf, device=device, extended=False)[sl]
            pos = SY.star_positions(ds.dir, ref, sc)
            ctx["truth_pos"] = {"x": pos["x"] - sl[1].start, "y": pos["y"] - sl[0].start, "flux": pos["flux"]}
        log(f"Crop {a.shape[0]}×{a.shape[1]} px of the stack; {int(ctx['train'].mean() * 100)} % of it for training")
        return ctx

    def run(self, ctx, p, log, cancel):
        from ..denoise import _batch_and_tile, infer, train_n2n
        dev = ctx["device"]
        _, tile = _batch_and_tile(dev)
        t = time.time()
        net = train_n2n(ctx["ga"], ctx["gb"], iters=int(p["iters"]), patch=int(p["patch"]), batch=int(p["batch"]),
                        device=dev, progress=_prog(log, 250), cancel=cancel.is_set, sample_mask=ctx["train"],
                        max_lr=float(p["max_lr"]), base=int(p["base"]))
        xa = ctx["stab"].inv(infer(net, ctx["ga"], tile=tile, tta=int(p["tta"])))
        dt = time.time() - t
        out = MX.halfstack_score(xa, ctx["a"], ctx["b"], ctx["test"], ctx["valid"], ctx["stab"], ctx["var"])
        out["seconds"] = dt
        if ctx["ds"].synthetic:
            m = MX.truth_metrics(xa, ctx["truth"], ctx["valid"], star_truth=ctx["truth_stars"], stars=ctx["truth_pos"],
                                 sigma_eval=float(ctx["opts"].get("sigma_eval", 1.0)) * ctx["scale"])
            out.update(_truth_row(m))
        return out, xa


# ----------------------------------------------------------------------------- N2N restoration network
class NetworkTask(Task):
    name = "network"
    label = "Noise2Noise restoration network"
    description = ("The pipeline's N2N denoiser plus deconvolution network, trained with the benchmark window "
                   "(and a margin) excluded, predicting the window from half A only; scored like ImageMM on the "
                   "held-out subs through their own PSFs, and against the truth on synthetic data. "
                   "Needs the prepared ImageMM exposures for scoring.")
    params = [
        {"name": "iters", "label": "Denoiser training steps", "type": "int", "low": 250, "high": 8000, "log": True,
         "default": 2000, "tune": True, "pipeline": "denoise_iters"},
        {"name": "deconv_iters", "label": "Deconvolution training steps", "type": "int", "low": 250, "high": 8000,
         "log": True, "default": 2000, "tune": True},
        {"name": "groups", "label": "Multi-frame loss groups (0 = half-stack loss)", "type": "int", "low": 0,
         "high": 16, "default": 0, "tune": True, "pipeline": "network_groups"},
        {"name": "max_lr", "label": "Denoiser peak learning rate", "type": "float", "low": 1e-4, "high": 5e-3,
         "log": True, "default": 1e-3, "tune": False},
    ]
    metrics = {k: v for k, v in ImageMMTask.metrics.items() if k != "iterations"}
    options = [
        {"name": "window", "label": "Window (px, reference grid)", "type": "int", "default": 256},
        {"name": "where", "label": "Window position", "type": "categorical", "choices": ["auto", "center"],
         "default": "auto"},
        {"name": "train_crop", "label": "Training region (px per side, reference grid; 0 = whole stack)",
         "type": "int", "default": 1536},
        {"name": "sigma_eval", "label": "Truth comparison resolution σ (px)", "type": "float", "default": 1.0},
    ]
    default_objective = "heldout_src"

    def prepare(self, ds, opts, device, log, cancel, study_dir):
        ctx = ImageMMTask().prepare(ds, opts, device, log, cancel, study_dir)
        from ..denoise import Stabiliser
        from ..pipeline import _load_fits
        s = ds.session()
        ctx["session"] = s
        sc = int(round(float(s.meta.get("scale", 1.0))))
        a, b = _load_fits(s._p("half_a.fits")), _load_fits(s._p("half_b.fits"))
        full, cov = _load_fits(s._p("stack.fits")), _load_fits(s._p("coverage.fits"))
        y0, y1, x0, x1 = ctx["window"]
        tc = int(opts.get("train_crop", 1536))
        if tc:
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            H0, W0 = a.shape[0] // sc, a.shape[1] // sc
            Y0, X0 = int(np.clip(cy - tc // 2, 0, max(H0 - tc, 0))), int(np.clip(cx - tc // 2, 0, max(W0 - tc, 0)))
            Y1, X1 = min(H0, Y0 + tc), min(W0, X0 + tc)
        else:
            Y0, X0, Y1, X1 = 0, 0, a.shape[0] // sc, a.shape[1] // sc
        sl = (slice(Y0 * sc, Y1 * sc), slice(X0 * sc, X1 * sc))
        ctx.update(a=a[sl], b=b[sl], full=full[sl], cov=cov[sl], off=(Y0, X0), sc=sc,
                   sat=s.meta.get("saturation", 63471.0))
        ctx["stab"] = Stabiliser(ctx["a"], ctx["b"])
        ctx["ga"], ctx["gb"] = ctx["stab"].fwd(ctx["a"]), ctx["stab"].fwd(ctx["b"])
        return ctx

    def run(self, ctx, p, log, cancel):
        from ..denoise import _batch_and_tile, _sky_map, channel_psf_field, infer, train_n2n, train_n2n_deconv
        dev, es, s = ctx["device"], ctx["es"], ctx["sc"]
        y0, y1, x0, x1 = ctx["window"]
        Y0, X0 = ctx["off"]
        a, b, full, cov = ctx["a"], ctx["b"], ctx["full"], ctx["cov"]
        marg = 64
        mask = cov >= 0.5 * np.percentile(cov[cov > 0], 90)
        wy0, wy1, wx0, wx1 = (y0 - Y0) * s, (y1 - Y0) * s, (x0 - X0) * s, (x1 - X0) * s
        mask[max(0, wy0 - marg * s):wy1 + marg * s, max(0, wx0 - marg * s):wx1 + marg * s] = False
        batch, tile = _batch_and_tile(dev)
        t = time.time()
        net = train_n2n(ctx["ga"], ctx["gb"], iters=int(p["iters"]), batch=batch, device=dev, sample_mask=mask,
                        progress=_prog(log, 250), cancel=cancel.is_set, max_lr=float(p["max_lr"]))
        da, db = infer(net, ctx["ga"], tile=tile, tta=8), infer(net, ctx["gb"], tile=tile, tta=8)
        stab = ctx["stab"]
        den = stab.inv(0.5 * (da + db))
        psfs = channel_psf_field(den, ctx["sat"], spacing=900.0 * s)
        if psfs is None:
            raise RuntimeError("not enough isolated stars to measure the PSF")
        var = cv2.GaussianBlur(0.5 * (a - b) ** 2, (0, 0), 10 * s)
        var = np.maximum(var, np.percentile(var[::4, ::4], 1, axis=(0, 1)) * 0.5).astype(np.float32)
        unsat = (full.max(-1) < 0.5 * ctx["sat"]).astype(np.uint8)
        weight = cv2.erode(unsat, np.ones((9, 9), np.uint8)).astype(np.float32)
        sky = _sky_map(full, int(64 * s))
        mf = None
        if int(p["groups"]):
            mf = ctx["session"].multiframe_targets(int(p["groups"]), 1.1, "empirical", device=dev)
            # the targets cover the whole reference (1x) grid: crop them to the training region,
            # whose origin (Y0, X0) is on that grid (the stack crop starts at s Y0, s X0)
            H0, W0 = es.H0, es.W0
            h1, w1 = a.shape[0] // s, a.shape[1] // s
            sets = []
            for T_ in mf["sets"]:
                T2 = dict(T_)
                for k in ("y", "v", "m"):
                    T2[k] = T_[k][:, Y0:Y0 + h1, X0:X0 + w1]
                    assert T_[k].shape[1:3] == (H0, W0)
                T2["kernels"] = torch.as_tensor(T_["kernels"], dtype=torch.float32, device=dev)
                sets.append(T2)
            mf = {**mf, "sets": sets}
        dnet = train_n2n_deconv(copy.deepcopy(net), da, db, a, b, stab, psfs, var, weight, sky,
                                iters=int(p["deconv_iters"]), batch=max(4, batch * 3 // 4), device=dev,
                                sample_mask=mask, mf=mf, progress=_prog(log, 250), cancel=cancel.is_set)
        ctxp = 64 * s
        Ya, Yb, Xa, Xb = wy0 - ctxp, wy1 + ctxp, wx0 - ctxp, wx1 + ctxp
        if Ya < 0 or Xa < 0 or Yb > a.shape[0] or Xb > a.shape[1]:
            raise RuntimeError("the window needs 64 px of context inside the training region")
        x = stab.inv(infer(dnet, da[Ya:Yb, Xa:Xb], tile=tile, tta=8))[ctxp:-ctxp, ctxp:-ctxp]
        if s > 1:
            x = x.reshape((y1 - y0), s, (x1 - x0), s, 3).mean((1, 3))
        x = x - es.sky_ref[y0:y1, x0:x1]
        dt = time.time() - t
        kb = es.kernels(ctx["idx_b"], "empirical", at=((y0 + y1 - 1) / 2, (x0 + x1 - 1) / 2))
        ch = MX.heldout_chi2(es, ctx["idx_b"], x, 1, kb, ctx["window"], ctx["smask"], dev)
        out = {"heldout_src": _mean(ch["src"]), "heldout_sky": _mean(ch["sky"]), "heldout_src_rgb": ch["src"],
               "heldout_sky_rgb": ch["sky"], "seconds": dt}
        out.update(MX.paper_metrics(x, ctx["coadd"]))
        if ctx["ds"].synthetic:
            full_t, stars_t, pos = ImageMMTask()._truth(ctx, 1, None)
            valid = np.ones(full_t.shape[:2], bool)
            valid[:4], valid[-4:], valid[:, :4], valid[:, -4:] = False, False, False, False
            unsat = cv2.erode((ctx["coadd"].max(-1) < 0.5 * es.sat).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            m = MX.truth_metrics(x, full_t, valid & unsat, star_truth=stars_t, stars=pos,
                                 sigma_eval=float(ctx["opts"].get("sigma_eval", 1.0)))
            out.update(_truth_row(m))
        return out, x


# ----------------------------------------------------------------------------- stacking
class StackTask(Task):
    name = "stack"
    label = "Registration & integration"
    description = ("Re-stacks the accepted subs with each trial's settings (in the study's own folder: the "
                   "dataset's session is not touched).  Real data: background noise and star FWHM (both on the "
                   "native pixel scale); synthetic data also against the truth seen through the subs' mean PSF.")
    params = [
        {"name": "sigma_low", "label": "Rejection σ low", "type": "float", "low": 1.5, "high": 8.0, "default": 4.0,
         "tune": True, "pipeline": "sigma_low"},
        {"name": "sigma_high", "label": "Rejection σ high", "type": "float", "low": 1.5, "high": 8.0, "default": 3.0,
         "tune": True, "pipeline": "sigma_high"},
        {"name": "local_norm", "label": "Local normalisation", "type": "bool", "default": True, "tune": True,
         "pipeline": "local_norm"},
        {"name": "sensitivity", "label": "Frame rejection sensitivity", "type": "float", "low": 0.5, "high": 2.0,
         "default": 1.0, "tune": False, "pipeline": "sensitivity"},
        {"name": "mode", "label": "Resampling", "type": "categorical", "choices": ["auto", "drizzle", "demosaic"],
         "default": "auto", "tune": False, "pipeline": "mode"},
        {"name": "scale", "label": "Output scale", "type": "categorical", "choices": [1.0, 1.5, 2.0], "default": 1.0,
         "tune": False, "pipeline": "scale"},
        {"name": "pattern_correction", "label": "Sensor pattern correction", "type": "bool", "default": True,
         "tune": False, "pipeline": "pattern_correction"},
    ]
    metrics = {
        "bg_noise": {"label": "Background noise (native pixels)", "direction": "minimize"},
        "fwhm": {"label": "Star FWHM (native pixels)", "direction": "minimize"},
        "seconds": {"label": "Run time (s)", "direction": "minimize"},
        **TRUTH_METRICS,
    }
    options = [{"name": "crop", "label": "Scored crop (px per side, native grid)", "type": "int", "default": 1024},
               {"name": "sigma_eval", "label": "Truth comparison resolution σ (native px)", "type": "float",
                "default": 1.0}]
    default_objective = "bg_noise"

    def prepare(self, ds, opts, device, log, cancel, study_dir):
        base = ds.session()
        if base.analysis is None:
            if ds.synthetic:
                ds.ensure_stacked(log, cancel)
                base = ds.session()
            else:
                raise RuntimeError(f"{ds.folder} has not been analysed: run 'Analyse frames' first")
        return {"ds": ds, "base": base, "device": device, "opts": opts, "study_dir": study_dir, "n": 0}

    def run(self, ctx, p, log, cancel):
        from ..pipeline import STACK_DEFAULTS, Session, _load_fits
        base, ds = ctx["base"], ctx["ds"]
        ctx["n"] += 1
        wd = os.path.join(ctx["study_dir"], "work", f"trial_{ctx['n']}")
        s = Session(ds.folder, wd)
        for f in ("analysis.pkl", "defects.npy"):
            if os.path.exists(base._p(f)):
                shutil.copy(base._p(f), s._p(f))
        s._load_state()
        s.cancel_flag = cancel
        t = time.time()
        try:
            meta = s.run_stack({**STACK_DEFAULTS, **{k: p[k] for k in ("sigma_low", "sigma_high", "local_norm",
                                                                         "sensitivity", "mode", "scale",
                                                                         "pattern_correction")}},
                               _prog(log, 20))
            dt = time.time() - t
            st, cov = _load_fits(s._p("stack.fits")), _load_fits(s._p("coverage.fits"))
            sc = float(meta["scale"])
            c = int(ctx["opts"].get("crop", 1024) * sc)
            sl = _deep_crop(cov, c)
            x = st[sl]
            x1 = cv2.resize(x, (int(x.shape[1] / sc), int(x.shape[0] / sc)), interpolation=cv2.INTER_AREA) if sc != 1 else x
            out = {"bg_noise": MX.background_noise(x1), "fwhm": MX.star_fwhm(x, meta["saturation"]) / sc,
                   "seconds": dt, "n_frames": meta["n_frames"]}
            if ds.synthetic:
                fr = [i for i, f in enumerate(s.analysis["frames"]) if f["accepted"]]
                psf = SY.effective_psf(ds.dir, fr, [s.analysis["frames"][i]["weight"] for i in fr], sc)
                H, W = st.shape[:2]
                ref = s.analysis["ref_idx"]
                tr = SY.truth_image(ds.dir, ref, (H, W), sc, psf, device=ctx["device"])[sl]
                ts = SY.truth_image(ds.dir, ref, (H, W), sc, psf, device=ctx["device"], extended=False)[sl]
                pos = SY.star_positions(ds.dir, ref, sc)
                pos = {"x": pos["x"] - sl[1].start, "y": pos["y"] - sl[0].start, "flux": pos["flux"]}
                good = cov[sl] >= 0.6 * np.percentile(cov[sl], 90)
                unsat = cv2.erode((x.max(-1) < 0.5 * meta["saturation"]).astype(np.uint8),
                                  np.ones((int(9 * sc) | 1,) * 2, np.uint8)) > 0
                m = MX.truth_metrics(x, tr, good & unsat, star_truth=ts, stars=pos,
                                     sigma_eval=float(ctx["opts"].get("sigma_eval", 1.0)) * sc)
                out.update(_truth_row(m))
            return out, x1
        finally:
            shutil.rmtree(wd, ignore_errors=True)


# ----------------------------------------------------------------------------- star separation
_STAR_TRUTH = {k: v for k, v in TRUTH_METRICS.items() if k in ("truth_nrmse", "truth_faint_nrmse", "truth_ssim",
                                                                  "truth_psnr")}
STAR_METRICS = {
    "residual_flux": {"label": "Star flux left in the starless image (fraction)", "direction": "minimize"},
    "residual_stars": {"label": "Stars still detected in the starless image (fraction)", "direction": "minimize"},
    "offstar_change": {"label": "Change away from the stars (× noise)", "direction": "minimize"},
    "seconds": {"label": "Run time (s)", "direction": "minimize"},
    **_STAR_TRUTH,
}
_STAR_OPTIONS = [
    {"name": "window", "label": "Scored window (px)", "type": "int", "default": 1024},
    {"name": "where", "label": "Window position", "type": "categorical", "choices": ["auto", "center"],
     "default": "auto"},
    {"name": "sigma_eval", "label": "Truth comparison resolution σ (px)", "type": "float", "default": 1.0},
]
_STAR_SCORING = ("Scored on a window of the linear image (the brightest extended structure, where stars on "
                 "nebulosity are hardest) against the stars the default detector finds in it: their flux left in "
                 "the starless image (local-background apertures), the fraction still detected, and the change "
                 "away from every star (damage to the nebula and sky, × the stack's noise). Synthetic data: the "
                 "starless image against the true sky without stars.")


def _star_prepare(ds, opts, device, log, cancel) -> dict:
    """The linear image the pipeline separates stars on, a window of it, and the yardstick: the
    stars the default detector finds there and the default mask's star-free pixels.  A synthetic
    dataset goes through the linear stage without a restoration, deconvolution or white balance,
    so that its stars are the subs' mean PSF and the truth is known on that grid."""
    import sep
    from ..pipeline import DEFAULTS, _load_fits
    from ..postprocess import classic_star_separation, detect_stars_for_mask, linear_stage, luminance
    s = ds.ensure_stacked(log, cancel)
    ctx = {"ds": ds, "opts": opts, "device": device}
    if ds.synthetic:
        st, cov = _load_fits(s._p("stack.fits")), _load_fits(s._p("coverage.fits"))
        sat = float(s.meta.get("saturation", 63471.0))
        lin, info = linear_stage(st, cov, None, {**DEFAULTS, "denoise": 0.0, "deconvolution": 0.0,
                                                 "white_balance": "none"}, sat)
        detect, rsig = None, None
    else:
        log("Linear image with the pipeline's default processing (the first time: background, colour, restoration)")
        lin, info = s.linear(dict(DEFAULTS))
        restored = info.get("restoration") == "ImageMM"
        det = s._lin_cache[3] if restored else None
        detect = luminance(det) if det is not None else None
        rsig = info.get("restored_sigma") if restored else None
    noise = float(info.get("noise_ref") or 0.0)
    window = _auto_window(lin, int(opts.get("window", 1024)), opts.get("where", "auto"))
    y0, y1, x0, x1 = window
    win = np.ascontiguousarray(lin[y0:y1, x0:x1])
    win_det = np.ascontiguousarray(detect[y0:y1, x0:x1]) if detect is not None else None
    ctx.update(lin=lin, detect=detect, rsig=rsig, noise=noise, window=window, win=win, win_det=win_det)
    # the yardstick (default constants: nothing is overridden here)
    L0 = np.ascontiguousarray(luminance(win), np.float32)
    if rsig is not None:
        objs, _, fw = detect_stars_for_mask(win_det, 1.0, noise)
    else:
        objs, _, fw = detect_stars_for_mask(L0, 1.0, noise, detect_floor=0.0)
    if len(objs) < 20:
        raise RuntimeError(f"only {len(objs)} stars in the window: too few to score a star removal")
    r_ap = max(1.5 * fw, 2.0)
    f0, _, _ = sep.sum_circle(L0, objs["x"], objs["y"], r_ap, bkgann=(2.5 * r_ap, 4 * r_ap), subpix=5)
    keep = f0 > 0
    smask0, _ = classic_star_separation(win, 1.0, noise_ref=noise, detect_L=win_det, restored_sigma=rsig)
    k = int(2 * round(2 * fw) + 1)
    off = cv2.dilate((smask0 > 0.05).astype(np.uint8), np.ones((k, k), np.uint8)) == 0
    off[:8], off[-8:], off[:, :8], off[:, -8:] = False, False, False, False
    ctx.update(L0=L0, ref_x=objs["x"][keep], ref_y=objs["y"][keep], ref_f=f0[keep], r_ap=r_ap, fw=fw, off=off)
    if ds.synthetic:
        fr = [i for i, f in enumerate(s.analysis["frames"]) if f["accepted"]]
        sc = float(s.meta.get("scale", 1.0))
        psf = SY.effective_psf(ds.dir, fr, [s.analysis["frames"][i]["weight"] for i in fr], sc)
        H, W = s.meta["shape"][:2]
        ref = s.analysis["ref_idx"]
        cy0, _, cx0, _ = info.get("crop") or [0, 0, 0, 0]
        sl = (slice(cy0 + y0, cy0 + y1), slice(cx0 + x0, cx0 + x1))
        ctx["truth"] = SY.truth_image(ds.dir, ref, (H, W), sc, psf, device=device, stars=False)[sl] / sat
        ctx["truth_stars"] = SY.truth_image(ds.dir, ref, (H, W), sc, psf, device=device, extended=False)[sl] / sat
        valid = cv2.erode((win.max(-1) < 0.5).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        valid[:8], valid[-8:], valid[:, :8], valid[:, -8:] = False, False, False, False
        ctx["valid"] = valid
    log(f"Window y {y0}:{y1} x {x0}:{x1}: {int(keep.sum())} reference stars (FWHM {fw:.2f} px), "
        f"{int(off.mean() * 100)} % of it away from every star")
    return ctx


def _star_score(ctx: dict, starless: np.ndarray) -> dict:
    import sep
    from scipy.spatial import cKDTree
    from ..postprocess import detect_stars_for_mask, luminance
    L1 = np.ascontiguousarray(luminance(starless), np.float32)
    r = ctx["r_ap"]
    f1, _, _ = sep.sum_circle(L1, ctx["ref_x"], ctx["ref_y"], r, bkgann=(2.5 * r, 4 * r), subpix=5)
    out = {"residual_flux": float(np.clip(f1, 0, None).sum() / ctx["ref_f"].sum())}
    # stars still there: the default detector on the starless image (a restoration's sky speckle
    # is kept out by detecting at the stack's noise)
    objs, _, _ = detect_stars_for_mask(L1, 1.0, ctx["noise"], detect_floor=None if ctx["rsig"] is not None else 0.0)
    if len(objs):
        d, _ = cKDTree(np.stack([objs["x"], objs["y"]], 1)).query(np.stack([ctx["ref_x"], ctx["ref_y"]], 1))
        out["residual_stars"] = float(np.mean(d < max(1.5 * ctx["fw"], 2.0)))
    else:
        out["residual_stars"] = 0.0
    out["n_reference_stars"] = int(len(ctx["ref_x"]))
    out["offstar_change"] = float(np.mean(np.abs(L1 - ctx["L0"])[ctx["off"]]) / max(ctx["noise"], 1e-9))
    if ctx["ds"].synthetic:
        m = MX.truth_metrics(starless, ctx["truth"], ctx["valid"], star_truth=ctx["truth_stars"],
                             sigma_eval=float(ctx["opts"].get("sigma_eval", 1.0)))
        out.update(_truth_row(m))
    return out


class StarSeparationTask(Task):
    name = "stars"
    label = "Star detection & separation (classic)"
    description = ("The classic star separation the pipeline uses until the AI star remover is trained, and "
                   "whose starless image the remover is trained on: star detection (sep, a top-hat pass for "
                   "stars on bright extended light, shape and concentration tests), photometric masks with "
                   "measured halos, and push-pull inpainting. " + _STAR_SCORING + " Its parameters are "
                   "constants of postprocess.py (STAR_DETECT, STAR_MASK).")
    params = [
        {"name": "detect_sigma", "label": "Detection threshold (× rms)", "type": "float", "low": 2.0, "high": 8.0,
         "default": 4.0, "tune": True, "code": "postprocess.STAR_DETECT"},
        {"name": "tophat_sigma", "label": "Top-hat pass threshold (× rms)", "type": "float", "low": 4.0,
         "high": 16.0, "default": 8.0, "tune": True, "code": "postprocess.STAR_DETECT"},
        {"name": "compact_fwhm", "label": "Max. size of a bright star (× FWHM)", "type": "float", "low": 1.5,
         "high": 6.0, "default": 3.0, "tune": False, "code": "postprocess.STAR_DETECT"},
        {"name": "max_elongation", "label": "Max. elongation a / b", "type": "float", "low": 1.2, "high": 4.0,
         "default": 2.0, "tune": False, "code": "postprocess.STAR_DETECT"},
        {"name": "concentration", "label": "Max. concentration index", "type": "float", "low": 1.5, "high": 6.0,
         "default": 3.0, "tune": False, "code": "postprocess.STAR_DETECT"},
        {"name": "deblend_cont", "label": "Deblending contrast", "type": "float", "low": 1e-4, "high": 0.05,
         "log": True, "default": 0.002, "tune": False, "code": "postprocess.STAR_DETECT"},
        {"name": "grow", "label": "Mask growth (× every radius)", "type": "float", "low": 0.6, "high": 2.0,
         "default": 1.0, "tune": True, "code": "postprocess.STAR_MASK"},
        {"name": "radius_scale", "label": "Mask radius (× core radius at the noise)", "type": "float", "low": 0.8,
         "high": 2.5, "default": 1.35, "tune": True, "code": "postprocess.STAR_MASK"},
        {"name": "min_radius_fwhm", "label": "Min. mask radius (× FWHM)", "type": "float", "low": 0.5, "high": 2.5,
         "default": 1.0, "tune": True, "code": "postprocess.STAR_MASK"},
        {"name": "halo_floor", "label": "Halo end (fraction of the peak)", "type": "float", "low": 2e-4,
         "high": 0.02, "log": True, "default": 0.002, "tune": True, "code": "postprocess.STAR_MASK"},
        {"name": "halo_margin", "label": "Halo mask margin (× halo radius)", "type": "float", "low": 1.0,
         "high": 1.6, "default": 1.15, "tune": False, "code": "postprocess.STAR_MASK"},
        {"name": "grain", "label": "Inpainting grain (× texture noise)", "type": "float", "low": 0.0, "high": 3.0,
         "default": 1.8, "tune": False, "code": "postprocess.STAR_MASK"},
    ]
    metrics = {**{k: v for k, v in STAR_METRICS.items() if not k.startswith("truth")},
               "masked_fraction": {"label": "Area masked and inpainted (fraction)", "direction": "minimize"},
               **_STAR_TRUTH}
    options = _STAR_OPTIONS
    default_objective = "residual_flux"

    def prepare(self, ds, opts, device, log, cancel, study_dir):
        return _star_prepare(ds, opts, device, log, cancel)

    def run(self, ctx, p, log, cancel):
        from .. import postprocess as PP
        t = time.time()
        with _overriding(PP.STAR_DETECT, p), _overriding(PP.STAR_MASK, p):
            smask, starless = PP.classic_star_separation(ctx["win"], 1.0, noise_ref=ctx["noise"],
                                                         detect_L=ctx["win_det"], restored_sigma=ctx["rsig"])
        dt = time.time() - t
        out = _star_score(ctx, starless)
        out.update(seconds=dt, masked_fraction=float((smask > 0.3).mean()))
        return out, starless


class StarRemoverTask(Task):
    name = "star_remover"
    label = "AI star remover"
    description = ("The pipeline's AI star remover (starnet.py): a U-Net trained on the dataset's own classic "
                   "starless image (default constants) with the image's own stars rendered and pasted onto it, "
                   "then applied to the window. " + _STAR_SCORING)
    params = [
        {"name": "iters", "label": "Training steps", "type": "int", "low": 250, "high": 8000, "log": True,
         "default": 3000, "tune": True, "pipeline": "star_remover_iters"},
        {"name": "max_lr", "label": "Peak learning rate (one-cycle)", "type": "float", "low": 1e-4, "high": 5e-3,
         "log": True, "default": 1e-3, "tune": True, "code": "starnet.train"},
        {"name": "patch", "label": "Patch size", "type": "categorical", "choices": [64, 96, 128, 192],
         "default": 128, "tune": True, "code": "starnet.train"},
        {"name": "star_weight", "label": "Extra loss weight on star pixels", "type": "float", "low": 0.0,
         "high": 10.0, "default": 4.0, "tune": True, "code": "starnet.train"},
        {"name": "bright_fraction", "label": "Batch share from bright backgrounds", "type": "float", "low": 0.0,
         "high": 0.9, "default": 0.5, "tune": True, "code": "starnet.train"},
        {"name": "paste_fraction", "label": "Own bright stars pasted (probability)", "type": "float", "low": 0.0,
         "high": 1.0, "default": 0.5, "tune": False, "code": "starnet.train"},
        {"name": "base", "label": "U-Net width (first level)", "type": "categorical", "choices": [16, 24, 32, 48],
         "default": 32, "tune": False, "code": "starnet.train"},
    ]
    metrics = STAR_METRICS
    options = _STAR_OPTIONS + [{"name": "train_crop", "label": "Training region (px per side; 0 = whole image)",
                                "type": "int", "default": 2048}]
    default_objective = "residual_flux"

    def prepare(self, ds, opts, device, log, cancel, study_dir):
        from ..postprocess import classic_star_separation
        ctx = _star_prepare(ds, opts, device, log, cancel)
        lin, (y0, y1, x0, x1) = ctx["lin"], ctx["window"]
        H, W = lin.shape[:2]
        tc = int(opts.get("train_crop", 2048))
        if tc and (tc < H or tc < W):
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            Y0, X0 = int(np.clip(cy - tc // 2, 0, max(H - tc, 0))), int(np.clip(cx - tc // 2, 0, max(W - tc, 0)))
            reg = (slice(Y0, min(H, Y0 + tc)), slice(X0, min(W, X0 + tc)))
        else:
            reg = (slice(0, H), slice(0, W))
        ctx["train_lin"] = np.ascontiguousarray(lin[reg])
        log("Classic starless image of the training region (the remover's training backgrounds)")
        det = ctx["detect"][reg] if ctx["detect"] is not None else None
        _, ctx["train_starless"] = classic_star_separation(ctx["train_lin"], 1.0, noise_ref=ctx["noise"],
                                                           detect_L=det, restored_sigma=ctx["rsig"])
        m = 64                                   # context round the window for the network
        Y0, X0 = max(0, y0 - m), max(0, x0 - m)
        ctx["win_ctx"] = np.ascontiguousarray(lin[Y0:min(H, y1 + m), X0:min(W, x1 + m)])
        ctx["win_off"] = (y0 - Y0, x0 - X0)
        return ctx

    def run(self, ctx, p, log, cancel):
        from .. import starnet
        t = time.time()
        net, meta = starnet.train(ctx["train_lin"], ctx["train_starless"], noise=ctx["noise"], iters=int(p["iters"]),
                                  patch=int(p["patch"]), device=ctx["device"], progress=_prog(log, 250),
                                  cancel=cancel.is_set, base=int(p["base"]), max_lr=float(p["max_lr"]),
                                  star_weight=float(p["star_weight"]), bright_fraction=float(p["bright_fraction"]),
                                  paste_fraction=float(p["paste_fraction"]))
        full = starnet.remove_stars(net, meta, ctx["win_ctx"])
        del net
        oy, ox = ctx["win_off"]
        h, w = ctx["win"].shape[:2]
        starless = np.ascontiguousarray(full[oy:oy + h, ox:ox + w])
        dt = time.time() - t
        out = _star_score(ctx, starless)
        out["seconds"] = dt
        return out, starless


# ----------------------------------------------------------------------------- gradient removal
class BackgroundTask(Task):
    name = "background"
    label = "Gradient removal"
    description = ("The linear stage's gradient model (light pollution, airglow, vignetting residual) on the stack, "
                   "auto-cropped like the pipeline: a sky survey reference (NSNS DR0.2, when the stack is plate-solved "
                   "and covered), or tile samples with a polynomial or RBF. Real data: the gradient left in the sky "
                   "(the range of a robust quadratic over the star-free sky tiles, plus the border's offset from the "
                   "interior, × the per-pixel noise) and the extended emission removed (the fraction of the emission "
                   "tiles' signal the model took away). Sky and emission tiles are set once, against a robust plane "
                   "through the faintest tiles, so no trial's model decides what counts as sky. Where faint emission "
                   "fills the frame, no tile is sky and the two pull against each other: tune them together (two "
                   "objectives, Pareto front), or on synthetic data against the truth (only a constant offset is "
                   "fitted; synthetic data have no plate solution, so no survey reference). Sampling, clipping and "
                   "the auto choice are constants of postprocess.py (BACKGROUND).")
    params = [
        {"name": "bg_method", "label": "Gradient model", "type": "categorical",
         "choices": ["auto", "reference", "poly", "rbf"], "default": "auto", "tune": True, "processing": "bg_method"},
        {"name": "bg_correction", "label": "Correction", "type": "categorical", "choices": ["subtract", "divide"],
         "default": "subtract", "tune": False, "processing": "bg_correction"},
        {"name": "bg_degree", "label": "Polynomial degree", "type": "int", "low": 0, "high": 4, "default": 2,
         "tune": True, "processing": "bg_degree"},
        {"name": "bg_smoothing", "label": "RBF smoothing", "type": "float", "low": 0.0, "high": 1.0, "default": 0.5,
         "tune": False, "processing": "bg_smoothing"},
        {"name": "grid", "label": "Sample grid (tiles across)", "type": "int", "low": 8, "high": 48, "default": 24,
         "tune": False, "code": "postprocess.BACKGROUND"},
        {"name": "clip_high", "label": "Reject samples above the model (× rms)", "type": "float", "low": 0.5,
         "high": 4.0, "default": 1.5, "tune": False, "code": "postprocess.BACKGROUND"},
        {"name": "clip_low", "label": "Reject samples below the model (× rms)", "type": "float", "low": 1.0,
         "high": 8.0, "default": 4.0, "tune": False, "code": "postprocess.BACKGROUND"},
        {"name": "start_percentile", "label": "First fit: samples up to this percentile", "type": "float",
         "low": 30.0, "high": 95.0, "default": 75.0, "tune": False, "code": "postprocess.BACKGROUND"},
        {"name": "min_sky_fraction", "label": "Star-free area a sample tile needs", "type": "float", "low": 0.1,
         "high": 0.9, "default": 0.4, "tune": False, "code": "postprocess.BACKGROUND"},
        {"name": "nebula_fraction", "label": "Polynomial → plane below this sky fraction", "type": "float",
         "low": 0.0, "high": 0.9, "default": 0.35, "tune": False, "code": "postprocess.BACKGROUND"},
        {"name": "auto_margin", "label": "Auto: gradient gain a flexible model needs (× noise)", "type": "float",
         "low": 0.0, "high": 2.0, "default": 0.25, "tune": False, "code": "postprocess.BACKGROUND"},
        {"name": "auto_margin_rel", "label": "Auto: relative gain a flexible model needs", "type": "float",
         "low": 0.0, "high": 0.9, "default": 0.3, "tune": False, "code": "postprocess.BACKGROUND"},
        {"name": "ref_degree", "label": "Reference: gradient degree", "type": "int", "low": 1, "high": 5,
         "default": 3, "tune": False, "code": "postprocess.BACKGROUND"},
        {"name": "ref_smooth", "label": "Reference: comparison resolution (survey px)", "type": "float",
         "low": 0.5, "high": 4.0, "default": 1.5, "tune": False, "code": "postprocess.BACKGROUND"},
        {"name": "ref_clip_high", "label": "Reference: ignore light above survey + gradient (× rms)",
         "type": "float", "low": 1.0, "high": 10.0, "default": 2.5, "tune": False, "code": "postprocess.BACKGROUND"},
    ]
    metrics = {
        "sky_flatness": {"label": "Gradient left in the sky (× pixel noise)", "direction": "minimize"},
        "signal_removed": {"label": "Extended emission removed (fraction)", "direction": "minimize"},
        "seconds": {"label": "Run time (s)", "direction": "minimize"},
        **{k: v for k, v in TRUTH_METRICS.items() if k in ("truth_nrmse", "truth_faint_nrmse", "truth_ssim",
                                                             "truth_psnr")},
    }
    options = [{"name": "sigma_eval", "label": "Truth comparison resolution σ (px)", "type": "float",
                "default": 4.0}]
    default_objective = "sky_flatness"

    def prepare(self, ds, opts, device, log, cancel, study_dir):
        from ..pipeline import DEFAULTS, _load_fits
        from ..postprocess import auto_crop_box, sky_yardstick
        s = ds.ensure_stacked(log, cancel)
        st, cov = _load_fits(s._p("stack.fits")), _load_fits(s._p("coverage.fits"))
        y0, y1, x0, x1 = auto_crop_box(cov, DEFAULTS["crop_threshold"])
        img = np.ascontiguousarray(st[y0:y1, x0:x1], np.float32)
        yard = sky_yardstick(img)
        ctx = {"ds": ds, "opts": opts, "img": img, "yard": yard, "reference": None}
        if not ds.synthetic:
            log("Sky survey reference (plate solution and NSNS maps; fetched once per stack)")
            sref = s.sky_reference()
            if sref is not None:
                ctx["reference"] = {**sref, "origin": (y0, x0)}
            else:
                log("No sky survey reference (not plate-solved, offline, or outside the survey): 'reference' "
                    "falls back to 'auto'")
        if not yard["emission"].any():
            log("No extended emission found: 'Extended emission removed' is not measured on this dataset")
        if ds.synthetic:
            fr = [i for i, f in enumerate(s.analysis["frames"]) if f["accepted"]]
            sc = float(s.meta.get("scale", 1.0))
            psf = SY.effective_psf(ds.dir, fr, [s.analysis["frames"][i]["weight"] for i in fr], sc)
            H, W = st.shape[:2]
            ref = s.analysis["ref_idx"]
            sl = (slice(y0, y1), slice(x0, x1))
            ctx["truth"] = SY.truth_image(ds.dir, ref, (H, W), sc, psf, device=device)[sl]
            ctx["truth_stars"] = SY.truth_image(ds.dir, ref, (H, W), sc, psf, device=device, extended=False)[sl]
            sat = float(s.meta.get("saturation", 63471.0))
            valid = (cov[sl] >= 0.6 * np.percentile(cov[sl], 90))
            valid &= cv2.erode((img.max(-1) < 0.5 * sat).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
            ctx["valid"] = valid
        log(f"{len(yard['tiles'])} scoring tiles: {int(yard['sky'].sum())} sky, {int(yard['emission'].sum())} emission")
        return ctx

    def run(self, ctx, p, log, cancel):
        from .. import postprocess as PP
        img, yard = ctx["img"], ctx["yard"]
        t = time.time()
        with _overriding(PP.BACKGROUND, p):
            bg, info = PP.background_model(img, p["bg_method"], int(p["bg_degree"]), reference=ctx["reference"],
                                           smoothing=float(p["bg_smoothing"]))
        dt = time.time() - t
        if p["bg_correction"] == "divide":
            lvl = np.maximum(bg.reshape(-1, 3).mean(0), 1e-12)
            res = img / np.maximum(bg / lvl, 0.05) - lvl
        else:
            res = img - bg
        T1 = PP.tile_medians(res, yard)
        out = {"sky_flatness": PP.gradient_left(res, yard, T1), "seconds": dt,
               "method_used": info.get("auto") or info.get("method"), "note": info.get("note")}
        sr = PP.emission_removed(res, yard, T1)
        if sr is not None:
            out["signal_removed"] = sr
        if ctx["ds"].synthetic:
            m = MX.truth_metrics(res, ctx["truth"], ctx["valid"], star_truth=ctx["truth_stars"],
                                 sigma_eval=float(ctx["opts"].get("sigma_eval", 4.0)), fit_plane=False,
                                 fit_offset=True)
            out.update(_truth_row(m))
        return out, res


TASKS = {t.name: t for t in (ImageMMTask(), DenoiseTask(), NetworkTask(), StackTask(), StarSeparationTask(),
                             StarRemoverTask(), BackgroundTask())}
