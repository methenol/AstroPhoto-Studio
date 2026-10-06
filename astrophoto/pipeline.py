"""Orchestration with on-disk caching (used by both the CLI and the web UI).

Heavy stages (analysis, integration, ML denoise) are cached per dataset in
``<workdir>/<dataset-slug>/``; processing parameters can then be iterated on
interactively without touching the raw frames again.
"""
from __future__ import annotations

import contextlib
import glob
import hashlib
import math
import json
import os
import pickle
import re
import shutil
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from astropy.io import fits
from PIL import Image

from . import __version__
from .analysis import analyse, finalize_selection
from .frames import FrameInfo, build_defect_map, discover, read_frame, superpixel
from .postprocess import DEFAULTS, is_narrowband, linear_stage, luminance, nonlinear_stage
from .stacking import Integrator, sensor_pattern

STACK_DEFAULTS = {
    "mode": "auto",          # auto | drizzle | demosaic
    "scale": 1.0,            # output up-sampling (1, 1.5, 2)
    "sigma_low": 4.0,
    "sigma_high": 3.0,
    "local_norm": True,
    "sensitivity": 1.0,      # frame-rejection aggressiveness
    "pattern_correction": True,  # measure the sensor pattern calibration left in the subs and remove it
                                 # (stacking.sensor_pattern: one extra stack pass the first time)
    "denoise_iters": 2000,
    "ai_deconvolution": True,  # train the self-supervised deconvolution network after the denoiser
    "deconv_method": "imagemm",  # imagemm | network | none  (best on held-out subs: experiments/README.md)
    # ImageMM (arXiv:2501.03002) on the individual subs, see astrophoto/imagemm.py
    "imagemm_r": 1,          # super-resolution factor r (Algorithm 2 for r > 1)
    "imagemm_sigma": 0.0,    # g_sigma of Eq. 11 in latent pixels; 0 = the paper's value (1 at r = 1 - its
                             # Fig. 5 -, 1.1 at r = 2)
    "imagemm_robust": True,  # Algorithm 3 (Huber, delta = 2) instead of the L2 loss
    "imagemm_delta": 2.0,    # Huber threshold delta of Algorithm 3 (the paper: 2)
    "imagemm_kappa": 2.0,    # clipping of the multiplicative update, kappa (the paper: 2)
    "imagemm_epsilon": 1e-4,  # stopping tolerance: flux rule ~1e-4; Eq. C15 (the paper) 1e-4 ... 1e-6
    "imagemm_stop": "flux",  # flux (sum |x_k - x_k-1| / sum x_k) | c15 (Eq. C15, the paper) | elementwise
                             # (mean |u'_k/u'_k-1 - 1|).  C15 stops far from the fixed point on sky-dominated
                             # cutouts (pixels clamped at kappa in two successive iterations have a ratio of
                             # exactly 1, and ratios above and below 1 cancel): 2 - 23 iterations on IC 405
    "imagemm_max_iters": 2000,
    "imagemm_psf": "empirical",  # empirical | moffat
    "imagemm_groups": 0,     # 0 = every exposure (the paper); N = N seeing-group coadds
    "imagemm_accelerate": True,  # Biggs & Andrews extrapolation (not in the paper): the converged
                                 # result in half the time of the plain run
    "imagemm_n2n": True,     # ImageMM on the two halves of the subs + Noise2Noise pass
    "imagemm_background": True,  # restore on a sky pedestal (imagemm.mm_restore): the exposures are
                                 # sky-subtracted and non-negativity would clip half the latent's sky at 0
    "n2n_split": "alternate",  # how the subs are split into the two Noise2Noise halves (half stacks,
                               # ImageMM's N2N pass, the network's multi-frame targets): alternate
                               # (frame by frame) | dither (whole dither blocks, analysis.half_split)
    "star_remover": True,    # train the AI star remover (starnet.py) after the restoration
    "star_remover_iters": 3000,
    "autofinish": True,      # before the export: tune the processing settings and fit a colour grade to
                             # reference images of the target (autofinish.py)
    "network_groups": 0,    # > 0: the network's data term is ImageMM's multi-frame likelihood over
                             # this many seeing-group coadds of the other half's subs
    "device": "auto",        # auto | cuda | cuda:N | mps | cpu
}

# Named restoration recipes for clients that choose one by name instead of setting every option
# (the /api/v1 job API).  Each holds only what differs from
# STACK_DEFAULTS, so "default" always follows STACK_DEFAULTS as it changes.
PROFILES = {
    "default": {"label": "Pipeline defaults (STACK_DEFAULTS)", "stack_params": {}},
    "imagemm": {"label": "ImageMM: multi-frame restoration of every sub (no Noise2Noise pass)",
                "stack_params": {"deconv_method": "imagemm", "imagemm_n2n": False}},
    "n2n-imagemm": {"label": "ImageMM on the two halves of the subs + Noise2Noise pass",
                    "stack_params": {"deconv_method": "imagemm", "imagemm_n2n": True}},
    "n2n-network": {"label": "Noise2Noise denoiser + self-supervised deconvolution network (conv2d U-Net); "
                             "much faster than ImageMM",
                    "stack_params": {"deconv_method": "network", "ai_deconvolution": True}},
    "n2n-rl": {"label": "Noise2Noise denoiser, Richardson-Lucy (TV) deconvolution when rendering",
               "stack_params": {"deconv_method": "none", "ai_deconvolution": False}},
}

LINEAR_KEYS = ["crop", "crop_threshold", "background", "bg_method", "bg_degree", "bg_smoothing", "bg_correction",
               "white_balance", "spcc_sensor", "spcc_filter", "spcc_white_ref", "denoise", "deconvolution",
               "restored_resolution"]


def local_copy_enabled() -> bool:
    """``ASTROPHOTO_LOCAL_COPY``: jobs read the subs from a local copy (Session.local_copy)."""
    return os.environ.get("ASTROPHOTO_LOCAL_COPY", "").strip().lower() in ("1", "true", "yes", "on")


def local_copy_dir(workdir: str) -> str:
    """``ASTROPHOTO_LOCAL_DIR``, else ``<workdir>/.local_copy``."""
    return os.path.abspath(os.environ.get("ASTROPHOTO_LOCAL_DIR") or os.path.join(workdir, ".local_copy"))


def restoration_done(session_dir: str) -> bool:
    """The Restore step has finished: the network output (denoised.fits), or an ImageMM
    restoration (imagemm.fits, recorded as the last restoration in restore_meta.json)."""
    if os.path.exists(os.path.join(session_dir, "denoised.fits")):
        return True
    meta = os.path.join(session_dir, "restore_meta.json")
    if os.path.exists(meta) and os.path.exists(os.path.join(session_dir, "imagemm.fits")):
        try:
            return json.load(open(meta)).get("deconv_method") == "imagemm"
        except Exception:
            return False
    return False


def restored_view(latent: np.ndarray, factor: float, info: dict) -> np.ndarray:
    """The restoration seen through a Gaussian g_sigma, sigma = ``factor`` x the Eq. 11 sigma_0 (the
    latent already is the sky through g_sigma_0, so it is convolved with the remaining
    sqrt(sigma^2 - sigma_0^2)).  Relative to sigma_0 because what the extra blur is for - the
    latent's pixel-scale speckle - lives on the latent grid: at r = 2 an input-pixel sigma of
    1.25 blurred 2.2 latent px and gave the super-resolution back."""
    return restored_view_sigma(latent, restored_sigma(factor, info), info)


def restored_sigma(factor: float, info: dict) -> float:
    """sigma of ``restored_view`` in latent pixels."""
    s0 = float((info.get("eq11") or {}).get("sigma") or 0.0)
    return float(factor) * (s0 if s0 > 0 else 1.0)


def restored_view_sigma(latent: np.ndarray, sigma: float, info: dict) -> np.ndarray:
    s0 = float((info.get("eq11") or {}).get("sigma") or 0.0)            # latent pixels
    s = math.sqrt(max(sigma ** 2 - s0 ** 2, 0.0))
    if s < 0.05:
        return latent
    return np.stack([cv2.GaussianBlur(latent[..., c], (0, 0), s) for c in range(latent.shape[-1])], -1).astype(np.float32)


def slugify(path: str) -> str:
    base = os.path.basename(os.path.normpath(path)) or "dataset"
    h = hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:6]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("_") + "-" + h


def split_folders(folder) -> list[str]:
    """The folders of a dataset: one path, a list, or several joined with ``os.pathsep`` (subs of
    the same target from several nights, stacked together).  Sorted, so a selection names one session."""
    parts = folder if isinstance(folder, (list, tuple)) else str(folder).split(os.pathsep)
    return sorted({os.path.abspath(os.path.expanduser(p.strip())) for p in parts if p and p.strip()})


def dataset_slug(folder) -> str:
    """The cache folder name of a dataset (one folder: as before; several: the first one's name,
    how many more, and a hash of them all)."""
    folders = split_folders(folder)
    if len(folders) == 1:
        return slugify(folders[0])
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.basename(folders[0]) or "dataset").strip("_")
    h = hashlib.sha1("\n".join(folders).encode()).hexdigest()[:6]
    return f"{base}+{len(folders) - 1}-{h}"


def _save_fits(path, arr, header=None):
    data = np.moveaxis(arr, -1, 0) if arr.ndim == 3 else arr
    fits.PrimaryHDU(data.astype(np.float32), header=header).writeto(path, overwrite=True)


def _load_fits(path):
    d = fits.getdata(path).astype(np.float32)
    return np.moveaxis(d, 0, -1) if d.ndim == 3 else d


def clean_json(o):
    """Recursively replace NaN/inf with None and numpy scalars with Python types (strict JSON)."""
    if isinstance(o, dict):
        return {k: clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean_json(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class _CompatUnpickler(pickle.Unpickler):
    """Load caches written before the package was renamed astropipe -> astrophoto."""

    def find_class(self, module, name):
        if module == "astropipe" or module.startswith("astropipe."):
            module = "astrophoto" + module[len("astropipe"):]
        return super().find_class(module, name)


class Cancelled(Exception):
    pass


class Session:
    """All state for one dataset: a folder of subs, or several folders of the same target (nights)
    stacked together (a list, or paths joined with ``os.pathsep``)."""

    def __init__(self, folder: str | list[str], workdir: str = "output"):
        self.folders = split_folders(folder)
        if not self.folders:
            raise ValueError("no dataset folder given")
        self.folder = os.pathsep.join(self.folders)
        self.dir = os.path.join(os.path.abspath(workdir), dataset_slug(self.folders))
        os.makedirs(self.dir, exist_ok=True)
        self.infos: list[FrameInfo] = []
        self.analysis: dict | None = None
        self.defects: np.ndarray | None = None
        self.overrides: dict[str, bool] = {}
        self._stack_cache: dict | None = None
        self._den_cache: np.ndarray | None = None
        self._sharp_cache: np.ndarray | None = None
        self._restored_cache: dict | None = None
        self._lin_cache: tuple[str, np.ndarray, dict] | None = None
        self._starless_cache: tuple[str, np.ndarray] | None = None
        self.lock = threading.RLock()
        self.cancel_flag = threading.Event()
        self.pause_flag = threading.Event()      # set: long stages wait at their next checkpoint
        self._load_state()

    # ------------------------------------------------------------- persistence
    def _p(self, name):
        return os.path.join(self.dir, name)

    def _load_state(self):
        if os.path.exists(self._p("analysis.pkl")):
            try:
                with open(self._p("analysis.pkl"), "rb") as f:
                    st = _CompatUnpickler(f).load()
                self.infos, self.analysis, self.overrides = st["infos"], st["analysis"], st.get("overrides", {})
                if os.path.exists(self._p("defects.npy")):
                    self.defects = np.load(self._p("defects.npy"))
            except Exception:
                self.analysis = None

    def _save_analysis(self):
        with open(self._p("analysis.pkl"), "wb") as f:
            pickle.dump({"infos": self.infos, "analysis": self.analysis, "overrides": self.overrides}, f)
        with open(self._p("frames.json"), "w") as f:
            json.dump(self.frames_table(), f, indent=1, default=_json_default)

    @property
    def meta(self) -> dict:
        p = self._p("stack_meta.json")
        return json.load(open(p)) if os.path.exists(p) else {}

    def status(self) -> dict:
        return {
            "folder": self.folder,
            "folders": self.folders,
            "workdir": self.dir,
            "n_files": len(self.infos) if self.infos else None,
            "analysed": self.analysis is not None,
            "stacked": os.path.exists(self._p("stack.fits")),
            "denoised": restoration_done(self.dir),
            "deconvolved": os.path.exists(self._p("sharp.fits")) or os.path.exists(self._p("imagemm.fits")),
            "star_remover": self.star_remover_ready(),
            "autofinished": os.path.exists(self._p("autofinish.json")),
            "stack_meta": clean_json(self.meta),
            "filter": self.infos[0].filter if self.infos else None,
            "object": self.infos[0].object if self.infos else None,
            "narrowband": is_narrowband(self.infos[0].filter) if self.infos else None,
            "telescope": getattr(self.infos[0], "telescope", "") if self.infos else None,
            "calibration_checked": os.path.exists(self._p("calibration.json")),
            "calibration": self._calibration_status(),
        }

    def _calibration_status(self) -> dict | None:
        cal = getattr(self.infos[0], "calib", None) if self.infos else None
        if not cal:
            return None
        return clean_json({"summary": cal.get("summary"), **cal.get("report", {})})

    def checkpoint(self) -> bool:
        """Called by every long stage between units of work (a frame, a sub, an iteration):
        waits here while the job is paused, and returns True when it has been cancelled."""
        while self.pause_flag.is_set() and not self.cancel_flag.is_set():
            time.sleep(0.25)
        return self.cancel_flag.is_set()

    def _check_cancel(self):
        if self.cancel_flag.is_set():
            raise Cancelled()

    # ------------------------------------------------------------- stage 1
    def scan(self, progress=None):
        self.infos = discover(self.folders)
        if not self.infos:
            raise RuntimeError(f"No light frames (FITS) found in {', '.join(self.folders)}")
        from .calibration import attach
        self.calib_error = None
        try:
            attach(self.infos, self.folders, self.dir, self.calib_prefs, progress)
        except Exception as e:          # a broken master must not stop the session: calibrate from the headers
            self.calib_error = f"{type(e).__name__}: {e}"
            print(f"calibration masters not used: {self.calib_error}")
        self._attach_pattern()
        return self.infos

    # ------------------------------------------------------------- sensor pattern
    def _pattern_key(self):
        return json.loads(json.dumps(self._calib_signature(getattr(self.infos[0], "calib", None)) if self.infos else None))

    def _attach_pattern(self) -> bool:
        """Point every sub at the measured sensor pattern when it was made with the current calibration."""
        path, meta = self._p("pattern.npy"), self._p("pattern.json")
        ok = False
        if os.path.exists(path) and os.path.exists(meta):
            try:
                ok = json.load(open(meta)).get("calib") == self._pattern_key()
            except Exception:
                ok = False
        for info in self.infos:
            info.pattern = path if ok else None
        return ok

    # ------------------------------------------------------------- calibration (stage 0)
    @property
    def calib_prefs(self) -> dict:
        from .calibration import DEFAULT_PREFS
        p = self._p("calib_prefs.json")
        try:
            return {**DEFAULT_PREFS, **(json.load(open(p)) if os.path.exists(p) else {})}
        except Exception:
            return dict(DEFAULT_PREFS)

    def set_calib_prefs(self, prefs: dict) -> dict:
        from .calibration import DEFAULT_PREFS
        new = {**self.calib_prefs, **{k: v for k, v in (prefs or {}).items() if k in DEFAULT_PREFS}}
        json.dump(new, open(self._p("calib_prefs.json"), "w"), indent=1)
        return new

    @staticmethod
    def _calib_signature(calib: dict | None):
        if not calib:
            return None
        return tuple(calib.get(k) for k in ("bias", "dark", "flat", "flat_bias", "adu_scale", "relevel", "optimize"))

    def run_calibration(self, progress=None) -> dict:
        """Find and choose the calibration masters (and fit the dark scale, check the flat) for this
        session's subs.  When the choice differs from the one the frame analysis was made with, the
        analysis is stale: it is dropped, so the next stage measures the frames again."""
        with self.lock:
            before = self._calib_signature(getattr(self.infos[0], "calib", None)) if (self.infos and self.analysis) else "-"
            self.scan(progress)
            after = self._calib_signature(getattr(self.infos[0], "calib", None))
            changed = self.analysis is not None and before != after
            if changed:
                self.analysis = None
                for f in ("analysis.pkl", "defects.npy", "frames.json", "pattern.npy", "pattern.json"):
                    if os.path.exists(self._p(f)):
                        os.remove(self._p(f))
            info = self.calibration_info(with_candidates=False)
            json.dump(clean_json(info), open(self._p("calibration.json"), "w"), indent=1, default=_json_default)
            return {"changed_analysis": changed, "summary": (info.get("report") or {}).get("summary")}

    def calibration_info(self, with_candidates: bool = True) -> dict:
        """What ``python -m astrophoto calibration`` prints: the telescope profile, the lights'
        settings, the masters chosen (and the ones that would fit), the fitted scale, the flat check."""
        from .calibration import candidates
        if not self.infos:
            self.scan()
        i0 = self.infos[0]
        profile = {"telescope": i0.telescope or i0.instrument or "", "instrument": i0.instrument,
                   "sensor": i0.sensor, "bayer": i0.bayer, "focallen": i0.focallen, "pixsize": i0.pixsize,
                   "width": i0.width, "height": i0.height, "camera_slot": i0.camera_slot,
                   "bit_depth": int(round(16 - np.log2(i0.adu_scale))) if getattr(i0, "adu_scale", 1) > 1 else 16,
                   "device_rejected": int(sum(getattr(i, "device_rejected", False) for i in self.infos))}
        out = {"profile": profile, "n_lights": len(self.infos), "prefs": self.calib_prefs,
               "env": os.environ.get("ASTROPHOTO_CALIB", ""), "error": getattr(self, "calib_error", None),
               "report": self._calibration_status()}
        if with_candidates:
            try:
                out["candidates"] = candidates(self.infos, self.folders, self.dir, self.calib_prefs)
            except Exception as e:
                out["candidates"] = {"bias": [], "dark": [], "flat": []}
                out["error"] = out["error"] or f"{type(e).__name__}: {e}"
        return clean_json(out)

    def run_analysis(self, sensitivity: float = 1.0, progress=None):
        with self.lock:
            self.scan(progress)
            if progress:
                progress(0, 1, f"Building hot-pixel map from {len(self.infos)} frames")
            self.defects = build_defect_map(self.infos)
            np.save(self._p("defects.npy"), self.defects)
            self._check_cancel()
            self.analysis = analyse(self.infos, self.defects, progress=progress, sensitivity=sensitivity)
            self.analysis["sensitivity"] = sensitivity
            self._apply_overrides()
            self._save_analysis()
            self._make_reference_preview()
            return self.frames_table()

    def reselect(self, sensitivity: float):
        """Re-run the rejection logic with a different aggressiveness (no re-measurement)."""
        with self.lock:
            res = finalize_selection(self.analysis["frames"], self.analysis["ref_idx"],
                                     self.analysis["grid"], sensitivity)
            res["sensitivity"] = sensitivity
            self.analysis = res
            self._apply_overrides()
            self._save_analysis()
            return self.frames_table()

    def set_override(self, name: str, accepted: bool | None):
        with self.lock:
            if accepted is None:
                self.overrides.pop(name, None)
            else:
                self.overrides[name] = bool(accepted)
            self.reselect(self.analysis.get("sensitivity", 1.0))

    def _apply_overrides(self):
        fr = self.analysis["frames"]
        wmax = max((f["weight"] for f in fr), default=1) or 1
        for f in fr:
            f["auto_accepted"] = not f["reject_reasons"]
            if f["name"] in self.overrides:
                f["accepted"] = self.overrides[f["name"]] and f["transform"] is not None
                f["overridden"] = True
                if f["accepted"] and f["weight"] == 0:
                    f["weight"] = 0.5 * wmax
                if not f["accepted"]:
                    f["weight"] = 0.0
            else:
                f["overridden"] = False

    def frames_table(self) -> list[dict]:
        if not self.analysis:
            return []
        out = []
        for info, f in zip(self.infos, self.analysis["frames"]):
            out.append({
                "name": f["name"], "time": info.date_obs, "exptime": info.exptime,
                "n_stars": f["n_stars"], "fwhm": f["fwhm"], "elongation": f["elongation"],
                "background": float(np.mean(f["background"])), "noise": f["noise"],
                "transparency": f["transparency"], "obstructed": f["obstructed_frac"],
                "reg_rms": f["reg_rms"], "anomaly": f.get("anomaly", 0.0), "weight": f["weight"],
                "accepted": f["accepted"], "auto_accepted": f.get("auto_accepted", f["accepted"]),
                "overridden": f.get("overridden", False), "reasons": f["reject_reasons"],
                "tile_mask": f["tile_mask"].tolist() if f.get("tile_mask") is not None else None,
                "is_reference": f["name"] == self.analysis["frames"][self.analysis["ref_idx"]]["name"],
                "device_rejected": getattr(info, "device_rejected", False),
            })
        return clean_json(out)

    def _make_reference_preview(self):
        ref = self.infos[self.analysis["ref_idx"]]
        self.frame_thumbnail(ref.name, 900, self._p("reference.jpg"))

    def frame_thumbnail(self, name: str, size: int = 360, out_path: str | None = None) -> bytes:
        info = next(i for i in self.infos if i.name == name)
        cache = out_path or self._p(f"thumbs/{name}_{size}.jpg")
        if os.path.exists(cache):
            return open(cache, "rb").read()
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        raw = read_frame(info)
        sp = superpixel(raw, info.bayer)
        img = autostretch(sp)
        h, w = img.shape[:2]
        s = size / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", (img[..., ::-1] * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 85])
        open(cache, "wb").write(buf.tobytes())
        return buf.tobytes()

    # ------------------------------------------------------------- stage 2
    def run_stack(self, params: dict | None = None, progress=None):
        p = {**STACK_DEFAULTS, **(params or {})}
        with self.lock:
            if self.analysis is None:
                self.run_analysis(p["sensitivity"], progress)
            elif abs(self.analysis.get("sensitivity", 1.0) - p["sensitivity"]) > 1e-6:
                self.reselect(p["sensitivity"])
            # the prepared ImageMM exposures use this coadd as their reference: stale after a restack
            for d in ("groups", "imagemm"):
                if os.path.isdir(self._p(d)):
                    shutil.rmtree(self._p(d))
            def integrate():
                return Integrator(self.infos, self.analysis, self.defects, mode=p["mode"], scale=float(p["scale"]),
                                  sigma_low=float(p["sigma_low"]), sigma_high=float(p["sigma_high"]),
                                  local_norm=bool(p["local_norm"]), progress=progress,
                                  cancel=self.checkpoint, split=p.get("n2n_split", "alternate"))
            pattern_info = None
            if not p.get("pattern_correction", True):
                for info in self.infos:
                    info.pattern = None
                out = integrate().run()
            elif self._attach_pattern():
                out = integrate().run()
                pattern_info = json.load(open(self._p("pattern.json")))
            else:
                # first pass without it, the pattern measured against that stack, then the stack again
                for info in self.infos:
                    info.pattern = None
                integ = integrate()
                out = integ.run()
                if progress:
                    progress(0, 1, "Sensor pattern: measuring what calibration left in the subs")
                try:
                    P, pattern_info = sensor_pattern(integ.items, out["stack"], self.defects, out["scale"],
                                                     progress=progress, cancel=self.checkpoint)
                except ValueError as e:      # too few subs: stack without it
                    pattern_info = {"skipped": str(e)}
                else:
                    del out
                    np.save(self._p("pattern.npy"), P)
                    pattern_info = {**pattern_info, "calib": self._pattern_key(),
                                    "created": datetime.now().isoformat(timespec="seconds")}
                    json.dump(pattern_info, open(self._p("pattern.json"), "w"), indent=1)
                    self._attach_pattern()
                    out = integrate().run()
            hdr = fits.Header()
            info0 = self.infos[0]
            for k, v in {"OBJECT": info0.object, "FILTER": info0.filter, "TELESCOP": getattr(info0, "telescope", ""),
                         "CALIBRAT": (getattr(info0, "calib", None) or {}).get("summary", "header BIAS"),
                         "NFRAMES": out["n_frames"],
                         "TOTEXP": out["total_exposure"], "STKMODE": out["mode"], "STKSCALE": out["scale"],
                         "CREATOR": f"AstroPhoto Studio {__version__}", "BAYERPAT": info0.bayer}.items():
                hdr[k] = v
            _save_fits(self._p("stack.fits"), out["stack"], hdr)
            _save_fits(self._p("half_a.fits"), out["half_a"])
            _save_fits(self._p("half_b.fits"), out["half_b"])
            _save_fits(self._p("coverage.fits"), out["coverage"])
            _save_fits(self._p("rejection_map.fits"), out["rejected_frac"])
            meta = {"mode": out["mode"], "scale": out["scale"], "n_frames": out["n_frames"],
                    "total_exposure": out["total_exposure"], "params": p,
                    "saturation": 65535.0 - info0.bias, "filter": info0.filter, "object": info0.object,
                    "calibration": self._calibration_status(),
                    "sensor_pattern": pattern_info,
                    "local_norm": out.get("local_norm"),
                    "split": out.get("split"),
                    "created": datetime.now().isoformat(timespec="seconds"),
                    "shape": list(out["stack"].shape)}
            json.dump(meta, open(self._p("stack_meta.json"), "w"), indent=1, default=_json_default)
            for f in ("denoised.fits", "sharp.fits", "restore_meta.json", "restore_nets.pt", "imagemm.fits",
                      "imagemm_coverage.fits", "starnet.pt", "starnet.json", "starless_ml.npz"):
                if os.path.exists(self._p(f)):
                    os.remove(self._p(f))
            self._stack_cache = {"stack": out["stack"], "coverage": out["coverage"]}
            self._den_cache = None
            self._sharp_cache = None
            self._lin_cache = None
            return meta

    def exposure_set(self, progress=None):
        """The prepared ImageMM exposures (exposures.ExposureSet), cached in imagemm/."""
        from .exposures import ExposureSet
        ref = _load_fits(self._p("stack.fits"))
        s = float(self.meta.get("scale", 1.0))
        W0, H0 = self.infos[0].width, self.infos[0].height
        if abs(s - 1) > 1e-6:
            # a drizzled stack at integer scale s: output pixel u is centred on reference
            # (u + 0.5)/s - 0.5, so s x s area averaging is exactly the reference grid
            ref = cv2.resize(ref, (W0, H0), interpolation=cv2.INTER_AREA)
        es = ExposureSet(self.infos, self.analysis, self.defects, ref, self.meta.get("saturation", 63471.0))
        path = self._p("imagemm/exposures.pkl")
        if os.path.exists(path):
            try:
                es.load(path)
            except RuntimeError as e:                 # another frame selection, or an older preparation
                print(f"ImageMM: preparing the exposures again ({e})", file=sys.stderr)
            else:
                es.normalise_seeing()
                return self._apply_sky_model(es)
        # everything cached from earlier prepared exposures (the network's multi-frame targets)
        # is stale once they are prepared again
        for f in glob.glob(self._p("imagemm/mf_targets_*.pkl")):
            os.remove(f)
        es.prepare(progress=progress, cancel=self.checkpoint)
        os.makedirs(self._p("imagemm"), exist_ok=True)
        es.save(path)
        es.normalise_seeing()
        return self._apply_sky_model(es)

    def multiframe_targets(self, n_groups: int, sigma: float, psf_model: str, progress=None, device="auto") -> dict:
        """Targets of the deconvolution network's multi-frame data term: seeing-group coadds of
        the odd subs (for the half-A input, which holds the even subs) and of the even subs
        (for half B), with their PSFs on the stack grid - the group PSF for a 1x stack, the
        Eq. 11 kernels (r = scale, g_sigma) for an integer drizzle scale."""
        import pickle
        from .denoise import pick_device
        from .imagemm import superresolved_kernels
        cache = self._p(f"imagemm/mf_targets_g{n_groups}_s{sigma:g}_{psf_model}_sky.pkl")
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                return pickle.load(f)
        s = float(self.meta.get("scale", 1.0))
        if abs(s - round(s)) > 1e-6:
            raise RuntimeError("the multi-frame loss needs an integer stack scale (1x or 2x)")
        s = int(round(s))
        es = self.exposure_set(progress)
        use = es.usable()
        # the split the stack's halves were made with (its targets are the other half's subs)
        halves = self.halves((self.meta.get("split") or {}).get("mode", "alternate"))
        sets = []
        for other in (1, 0):                        # half A's targets: the subs of half B, and back
            idx = [k for k in use if halves[k] == other]
            T_ = es.group_coadds(idx, n_groups, psf_model, progress=progress, cancel=self.checkpoint)
            # the network works in the stack's units, sky pedestal included: put the reference
            # sky model the exposures were background-subtracted by back into the targets
            T_["y"] = T_["y"] + es.sky_ref[None]
            T_["sky_included"] = True
            if s > 1:
                T_["kernels"], _ = superresolved_kernels(T_["kernels"], s, sigma, device=pick_device(device))
            sets.append(T_)
        mf = {"sets": sets, "s": s, "delta": 2.0}
        with open(cache, "wb") as f:            # imagemm/ is cleared when the session is restacked
            pickle.dump(mf, f, protocol=4)
        return mf

    def halves(self, split: str) -> np.ndarray:
        """Half (0 = A, 1 = B) of every stacked frame, in the order of the stacker's and the
        prepared exposures' frames (accepted, weight > 0)."""
        from .analysis import half_split
        frames = self.analysis["frames"]
        labels, _ = half_split(self.infos, frames, split)
        return labels[[i for i, f in enumerate(frames) if f["accepted"] and f["weight"] > 0]]

    def run_denoise(self, params: dict | None = None, progress=None):
        """Restoration: Noise2Noise denoiser and (optionally) the N2N deconvolution network,
        or ImageMM on the individual subs."""
        from .denoise import n2n_restore
        p = {**STACK_DEFAULTS, **(params or {})}
        method = p.get("deconv_method") or "network"
        if method == "network" and not p.get("ai_deconvolution", True):
            method = "none"
        with self.lock:
            if method == "imagemm":
                from . import imagemm
                es = self.exposure_set(progress)
                r_ = int(p["imagemm_r"])
                # the paper's g_sigma (Eq. 11): 1 at r = 1, 1.1 at r = 2.  At r = 1 the latent is then the
                # sky seen through a 1 px Gaussian - band-limited, so a star between pixel centres is
                # represented exactly; with the measured PSFs directly (no Eq. 11) the latent can only
                # split such a star over whole pixels, the model comes out broader than the star, and the
                # fit pulls light out of a ring 3-4 px around every star (IC 405: -1 % of the peak)
                sigma = float(p.get("imagemm_sigma") or 0) or (1.1 if r_ > 1 else 1.0)
                lat, info = imagemm.restore(
                    es, r=int(p["imagemm_r"]), sigma=sigma, psf_model=p["imagemm_psf"],
                    n_groups=int(p["imagemm_groups"]), robust=bool(p["imagemm_robust"]),
                    delta=float(p.get("imagemm_delta", 2.0)), kappa=float(p.get("imagemm_kappa", 2.0)),
                    epsilon=float(p["imagemm_epsilon"]), stop=p.get("imagemm_stop", "flux"),
                    max_iters=int(p["imagemm_max_iters"]),
                    accelerate=bool(p["imagemm_accelerate"]), n2n=bool(p.get("imagemm_n2n")),
                    n2n_iters=int(p["denoise_iters"]), device=p["device"], progress=progress,
                    halves=self.halves(p.get("n2n_split", "alternate")) if p.get("imagemm_n2n") else None,
                    background=bool(p.get("imagemm_background", True)),
                    kernel_cache=self._p(f"imagemm/kernels_r{r_}_s{sigma}_{p['imagemm_psf']}.pkl"),
                    cancel=self.checkpoint)
                cov = info.pop("coverage").mean(-1)
                resid = info.pop("residual", None)
                _save_fits(self._p("imagemm.fits"), lat)
                if resid is not None:
                    _save_fits(self._p("imagemm_residual.fits"), resid)
                elif os.path.exists(self._p("imagemm_residual.fits")):
                    os.remove(self._p("imagemm_residual.fits"))
                _save_fits(self._p("imagemm_coverage.fits"), (cov / max(float(cov.max()), 1e-12)).astype(np.float32))
                info["ptc"] = {k: np.asarray(v).tolist() for k, v in es.ptc.items()}
                info["sky_model"] = es.sky_info.get("auto") or es.sky_info.get("method")
                self._restored_cache = None
            else:
                st = self._load_stack()
                mf = None
                if method == "network" and int(p.get("network_groups") or 0) > 0:
                    mf = self.multiframe_targets(int(p["network_groups"]), float(p.get("imagemm_sigma") or 0) or 1.1,
                                                 p["imagemm_psf"], progress, device=p["device"])
                a, b = _load_fits(self._p("half_a.fits")), _load_fits(self._p("half_b.fits"))
                den, sharp, info = n2n_restore(
                    a, b, st["stack"], iters=int(p["denoise_iters"]), device=p["device"], coverage=st["coverage"],
                    progress=progress, cancel=self.checkpoint, deconvolve=method == "network",
                    sat=self.meta.get("saturation", 63471.0), px_scale=float(self.meta.get("scale", 1.0)),
                    save_path=self._p("restore_nets.pt"), mf=mf)
                del a, b
                _save_fits(self._p("denoised.fits"), den)
                if sharp is not None:
                    _save_fits(self._p("sharp.fits"), sharp)
                elif os.path.exists(self._p("sharp.fits")):
                    os.remove(self._p("sharp.fits"))
                self._den_cache = den
                self._sharp_cache = sharp
            info["deconv_method"] = method
            info["created"] = datetime.now().isoformat(timespec="seconds")
            json.dump(info, open(self._p("restore_meta.json"), "w"), indent=1, default=_json_default)
            self._lin_cache = None
            return True

    def _restoration_method(self) -> str | None:
        p = self._p("restore_meta.json")
        return json.load(open(p)).get("deconv_method") if os.path.exists(p) else None

    def _load_restored(self):
        """The ImageMM latent image and its coverage, if that was the last restoration."""
        if self._restoration_method() != "imagemm" or not os.path.exists(self._p("imagemm.fits")):
            return None
        if self._restored_cache is None:
            img = _load_fits(self._p("imagemm.fits"))
            bad = int((~np.isfinite(img)).sum())
            if bad:
                raise RuntimeError(f"The ImageMM restoration (imagemm.fits) has {bad} non-finite pixels "
                                   f"({bad / img.size:.3%}); run Restore again")
            res = None
            if os.path.exists(self._p("imagemm_residual.fits")):
                res = _load_fits(self._p("imagemm_residual.fits"))
                if res.shape != img.shape or not np.isfinite(res).all():
                    res = None
            self._restored_cache = {"image": img, "coverage": _load_fits(self._p("imagemm_coverage.fits")),
                                    "residual": res}
        return self._restored_cache

    def _restore_info(self) -> dict:
        p = self._p("restore_meta.json")
        return json.load(open(p)) if os.path.exists(p) else {}

    def _load_stack(self):
        if self._stack_cache is None:
            if not os.path.exists(self._p("stack.fits")):
                raise RuntimeError("Dataset has not been stacked yet")
            self._stack_cache = {"stack": _load_fits(self._p("stack.fits")),
                                 "coverage": _load_fits(self._p("coverage.fits"))}
        return self._stack_cache

    def _load_denoised(self):
        if self._den_cache is None and os.path.exists(self._p("denoised.fits")):
            self._den_cache = _load_fits(self._p("denoised.fits"))
        return self._den_cache

    def _load_sharp(self):
        if self._sharp_cache is None and os.path.exists(self._p("sharp.fits")):
            self._sharp_cache = _load_fits(self._p("sharp.fits"))
        return self._sharp_cache

    # ------------------------------------------------------------- stage 3
    def linear(self, params: dict, progress=None):
        p = {**DEFAULTS, **(params or {})}
        sol = self._p("explore/solution.json")
        survey = p["background"] and p["bg_method"] in ("auto", "reference")
        key = (json.dumps({k: p[k] for k in LINEAR_KEYS}, sort_keys=True) + self.meta.get("created", "") +
               (str(os.path.getmtime(sol)) if (p["white_balance"] == "auto" or survey) and os.path.exists(sol) else "") +
               (str(os.path.getmtime(self._p("skyref.npz"))) if survey and os.path.exists(self._p("skyref.npz")) else ""))
        with self.lock:
            if self._lin_cache and self._lin_cache[0] == key:
                return self._lin_cache[1], self._lin_cache[2]
            st = self._load_stack()
            rest = self._load_restored()
            if rest is not None:
                # ImageMM's latent image is already restored (deconvolved, sky noise suppressed):
                # no denoise blend and no second deconvolution
                img = rest["image"]
                # crop on the frames' footprint (the stack's coverage): the restoration's own map is
                # the fraction of subs with valid data, which is ~0 on every saturated star (masked
                # in all subs), and the largest rectangle avoiding those holes is a thin band of the
                # field (IC 405: rows 1248-3168 of 7680)
                cov = _resize_to(st["coverage"], img.shape[:2])
                # the restoration is shown as the sky seen through a Gaussian g_sigma (the paper's Eq. 11,
                # sigma = 1 at r = 1); one already made with Eq. 11 at sigma_0 only gets the remaining
                # sqrt(sigma^2 - sigma_0^2) (none at the default resolution)
                img = restored_view(img, float(p.get("restored_resolution", 1.0)), self._restore_info())
                resid = rest.get("residual")
                if resid is not None:        # the output's noise realisation (N2N pass), seen the same way
                    resid = restored_view(resid, float(p.get("restored_resolution", 1.0)), self._restore_info())
                ref = st["stack"]
                clip_ref = cv2.resize(ref, (img.shape[1], img.shape[0]),
                                      interpolation=cv2.INTER_AREA if ref.shape[1] > img.shape[1] else cv2.INTER_LINEAR)
                lin, info = linear_stage(img, cov, None, p, self.meta.get("saturation", 63471.0),
                                         progress=progress, restored=True, clip_ref=clip_ref,
                                         ref_stars=self._ref_stars(p, img.shape[1] / ref.shape[1]))
                info["restoration"] = "ImageMM"
                sky_model = self._restore_info().get("sky_model")
                info["background"] = {"method": "subtracted before the restoration"
                                                + (f": {sky_model}" if sky_model else " (degree-2 polynomial: restored "
                                                   "before the sky-survey reference; run Restore again to use it)")}
                info["upscaled"] = img.shape[1] / st["stack"].shape[1]
                # the coadd on the same grid, cropped and scaled like the output: stars are found on
                # it (the restoration's sky speckle would pass for faint stars)
                det = clip_ref
                if info.get("crop"):
                    y0, y1, x0, x1 = info["crop"]
                    det = det[y0:y1, x0:x1]
                # (white-balanced like the output: the palette measures the Ha leakage on it)
                detect = np.ascontiguousarray(det * np.asarray(info.get("wb_gains", [1, 1, 1]), np.float32)
                                              / float(info.get("white_level", self.meta.get("saturation", 63471.0))),
                                              np.float32)
                info["restored_sigma"] = restored_sigma(float(p.get("restored_resolution", 1.0)), self._restore_info())
                if resid is not None:        # cropped, white-balanced and normalised like the output
                    if info.get("crop"):
                        y0, y1, x0, x1 = info["crop"]
                        resid = resid[y0:y1, x0:x1]
                    resid = np.ascontiguousarray(resid * np.asarray(info.get("wb_gains", [1, 1, 1]), np.float32)
                                                 / float(info["white_level"]), np.float32)
            else:
                detect = resid = None
                den = self._load_denoised()
                sharp = self._load_sharp() if den is not None else None
                if sharp is not None:
                    # the deconvolution network deconvolves towards points: a bright star became a
                    # single-pixel spike 10x its stacked peak (C 33: 175 000 ADU against 17 700, 2.8x
                    # the white level) with its surroundings drained - a dark hole round a dot whose
                    # colour is whichever channel spiked highest (red after white balance).  It is
                    # shown, like an ImageMM restoration, through a Gaussian of "restored_resolution"
                    # px: flux-conserving, Nyquist-sampled cores, the same profile in every channel
                    sharp = restored_view_sigma(sharp, float(p.get("restored_resolution", 1.0)), {})
                lin, info = linear_stage(st["stack"], st["coverage"], den, p, self.meta.get("saturation", 63471.0),
                                         progress=progress, sharp=sharp, ref_stars=self._ref_stars(p),
                                         sky_reference=self.sky_reference() if survey else None)
            self._lin_cache = (key, lin, info, detect, resid)
            return lin, info

    def _ref_stars(self, p: dict, up: float = 1.0) -> dict | None:
        """Catalogue stars for the spectrophotometric white balance (spcc.py), in pixels of an image
        ``up`` x the stack's resolution, or {"error": why not}; None unless it is "auto"."""
        if p.get("white_balance") != "auto":
            return None
        from .spcc import reference_stars
        try:
            r = reference_stars(self, p)
        except Exception as e:
            return {"error": f"photometric calibration unavailable: {e}"}
        if r is None:
            return {"error": "the stack is not plate-solved yet (Explore tab; done automatically after stacking "
                             "when online)"}
        if abs(up - 1) > 1e-6:
            r = {**r, "x": (r["x"] + 0.5) * up - 0.5, "y": (r["y"] + 0.5) * up - 0.5}
        return r

    def plate_solve(self, progress=None) -> dict | None:
        """Solve the stack against Gaia (astrometry.py) unless it already is; best effort (needs a
        connection and RA/Dec in the headers): None when it cannot."""
        from .astrometry import load_solution, solve_session
        try:
            sol = load_solution(self) or solve_session(self, progress=progress)
        except Cancelled:
            raise
        except Exception as e:
            if progress:
                progress(1, 1, f"Plate solving skipped ({type(e).__name__}: {str(e)[:80]})")
            return None
        if progress:
            progress(1, 1, "Sky survey reference for the gradient model")
        self.sky_reference()                     # fetched now, not at the first preview
        return sol

    # ------------------------------------------------------------- sky survey reference
    def sky_reference(self, fetch: bool = True) -> dict | None:
        """A calibrated sky survey's maps of the field on the stack's grid (skyref.py), for the gradient
        model: {"refs", "px_per_ref"}, cached in skyref.npz.  None without a plate solution, offline
        (a failed fetch is not retried for 10 minutes) or outside the survey (the model then falls back)."""
        from . import skyref
        from .astrometry import load_solution
        if not os.path.exists(self._p("stack.fits")):
            return None
        sol = load_solution(self)
        if sol is None:
            return None
        shape = tuple(sol.get("shape") or self.meta.get("shape", [0, 0])[:2])
        key = json.dumps([sol["wcs_header"][:4000], list(shape), skyref.SURVEYS, skyref.FACTOR])
        path = self._p("skyref.npz")
        cached = getattr(self, "_skyref", None)
        if cached and cached[0] == key:
            return cached[1]
        if os.path.exists(path):
            try:
                z = np.load(path)
                if str(z["key"]) == key:
                    ref = {"refs": z["refs"].astype(np.float32), "px_per_ref": float(skyref.FACTOR)}
                    self._skyref = (key, ref)
                    return ref
            except Exception:
                pass
        fail = getattr(self, "_skyref_fail", None)
        if not fetch or (fail and fail[0] == key and time.time() - fail[1] < 600):
            return None
        try:
            refs = skyref.fetch(sol["wcs"], shape)
        except Exception as e:
            self._skyref_fail = (key, time.time(), f"{type(e).__name__}: {e}")
            print(f"sky survey reference unavailable: {self._skyref_fail[2]}")
            return None
        np.savez_compressed(path, key=key, refs=refs.astype(np.float32))
        ref = {"refs": refs, "px_per_ref": float(skyref.FACTOR)}
        self._skyref = (key, ref)
        return ref

    def _apply_sky_model(self, es):
        """The sky model every exposure is background-subtracted by before ImageMM (``es.sky_ref``, on
        the reference coadd's 1x grid): against the sky survey when available, else the gradient model's
        "auto" choice.  It is subtracted when a window is restored, so a new one needs no new preparation."""
        from .postprocess import background_model
        sref = self.sky_reference()
        ref = None
        if sref is not None:
            ref = {**sref, "px_per_ref": sref["px_per_ref"] / float(self.meta.get("scale", 1.0))}
        es.sky_ref, info = background_model(es.ref, "auto", 2, reference=ref)
        es.sky_info = {k: v for k, v in info.items() if not k.startswith("_")}
        es.sky_ref = es.sky_ref.astype(np.float32)
        return es

    # ------------------------------------------------------------- AI star remover
    def _star_source(self) -> str:
        """What a star remover is trained for: the current restoration (or stack)."""
        return str(self._restore_info().get("created") or self.meta.get("created", ""))

    def star_remover_ready(self) -> bool:
        """A star remover has been trained on the current restoration (a restack or a new
        restoration makes the old one stale)."""
        p = self._p("starnet.json")
        if not (os.path.exists(p) and os.path.exists(self._p("starnet.pt"))):
            return False
        try:
            return json.load(open(p)).get("source") == self._star_source()
        except Exception:
            return False

    def train_star_remover(self, params: dict | None = None, stack_params: dict | None = None, progress=None):
        """Train the AI star remover (starnet.py) on this dataset's full-resolution linear image."""
        from . import starnet
        from .postprocess import classic_star_separation
        sp = {**STACK_DEFAULTS, **(stack_params or {})}
        with self.lock:
            lin, info = self.linear(params or {}, progress)
            detect = self._lin_cache[3]
            if progress:
                progress(0, 1, "Classic star separation (training backgrounds)")
            _, starless = classic_star_separation(
                lin, 1.0, noise_ref=info.get("noise_ref"),
                detect_L=luminance(detect) if detect is not None else None,
                restored_sigma=info.get("restored_sigma") if info.get("restoration") == "ImageMM" else None)
            net, meta = starnet.train(lin, starless, noise=float(info.get("noise_ref") or 0.0),
                                      iters=int(sp["star_remover_iters"]), device=sp["device"],
                                      progress=progress, cancel=self.checkpoint)
            meta["source"] = self._star_source()
            meta["created"] = datetime.now().isoformat(timespec="seconds")
            starnet.save(self._p("starnet.pt"), net, meta)
            self._starless_cache = None
            if os.path.exists(self._p("starless_ml.npz")):
                os.remove(self._p("starless_ml.npz"))
            return meta

    def ml_starless(self, lin: np.ndarray, progress=None) -> np.ndarray:
        """The AI star remover's starless version of the current full-resolution linear image
        (cached in memory and on disk, keyed by the linear parameters, the model and a sample of the
        linear image itself: keyed by the parameters alone, a cache made by an earlier version of the
        linear stage (another pedestal) was reused, and the difference became a grey "star layer"
        over the whole sky)."""
        from . import starnet
        key = (self._lin_cache[0] + str(os.path.getmtime(self._p("starnet.pt")))
               + f"|{float(lin[::97, ::89].astype(np.float64).sum()):.9e}")
        if self._starless_cache and self._starless_cache[0] == key:
            return self._starless_cache[1]
        path = self._p("starless_ml.npz")
        out = None
        if os.path.exists(path):
            try:
                z = np.load(path)
                if str(z["key"]) == key and z["img"].shape == lin.shape:
                    out = z["img"].astype(np.float32)
            except Exception:
                out = None
        if out is None:
            net, meta = starnet.load(self._p("starnet.pt"))
            out = starnet.remove_stars(net, meta, lin, progress=progress)
            del net
            np.savez(path, key=key, img=out.astype(np.float16))
        self._starless_cache = (key, out)
        return out

    def render(self, params: dict, max_size: int | None = 1400, progress=None) -> tuple[np.ndarray, dict]:
        lin, params, f, info = self.render_inputs(params, max_size, progress)
        out = nonlinear_stage(lin, params, self.meta.get("filter", ""), px_scale=f, progress=progress)
        return out, info

    def render_inputs(self, params: dict, max_size: int | None = 1400, progress=None):
        """What ``render`` gives the non-linear stage: the linear image at the render size, the
        parameters with the linear stage's references added (noise, starless image, ...), the scale
        and the linear stage's info.  Auto-finish renders many settings from one of these."""
        lin, info = self.linear(params, progress)
        detect = self._lin_cache[3] if self._lin_cache and len(self._lin_cache) > 3 else None
        resid = self._lin_cache[4] if self._lin_cache and len(self._lin_cache) > 4 else None
        starless = None
        if params.get("star_removal", DEFAULTS["star_removal"]) != "classic" and params.get("star_separation", True):
            if self.star_remover_ready():
                starless = self.ml_starless(lin, progress)
                info = {**info, "star_removal": "AI star remover"}
            else:
                info = {**info, "star_removal": "classic (train the AI star remover to use it)"}
        f = 1.0
        if max_size and max(lin.shape[:2]) > max_size:
            f = max_size / max(lin.shape[:2])
            size = (int(lin.shape[1] * f), int(lin.shape[0] * f))
            lin = cv2.resize(lin, size, interpolation=cv2.INTER_AREA)
            if starless is not None:
                starless = cv2.resize(starless, size, interpolation=cv2.INTER_AREA)
            if detect is not None:
                detect = cv2.resize(detect, size, interpolation=cv2.INTER_AREA)
            if resid is not None:            # averaged down with the image, its noise with it
                resid = cv2.resize(resid, size, interpolation=cv2.INTER_AREA)
        params = {**params, "_noise_ref": info.get("noise_ref"), "_noise_proxy": info.get("noise_proxy"),
                  "_starless": starless}
        if info.get("restoration") == "ImageMM":
            params["_restored_sigma"] = info.get("restored_sigma", 1.0)
            params["_detect_ref"] = detect
            params["_noise_resid"] = resid
        return lin, params, f, info

    def render_before(self, params: dict, max_size: int = 1400) -> np.ndarray:
        """Plain auto-stretched stack (same crop) for before/after comparison."""
        st = self._load_stack()
        _, info = self.linear(params)
        img = st["stack"]
        if "crop" in info:
            y0, y1, x0, x1 = (int(round(v / info.get("upscaled", 1.0))) for v in info["crop"])
            img = img[y0:y1, x0:x1]
        f = min(1.0, max_size / max(img.shape[:2]))
        img = cv2.resize(img, (int(img.shape[1] * f), int(img.shape[0] * f)), interpolation=cv2.INTER_AREA)
        return autostretch(img)

    def export(self, params: dict, quality: int = 95, upscale: float = 1.0, tiff: bool = True,
               progress=None) -> dict:
        img, info = self.render(params, max_size=None, progress=progress)
        if upscale and upscale > 1:
            img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)
            img = np.clip(img, 0, 1)
        os.makedirs(self._p("exports"), exist_ok=True)
        obj = (self.meta.get("object") or "image").replace(" ", "")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = self._p(f"exports/{obj}_{stamp}")
        im8 = Image.fromarray((img * 255 + 0.5).astype(np.uint8))
        desc = (f"{self.meta.get('object', '')} | {self.meta.get('n_frames')} x subs, "
                f"{self.meta.get('total_exposure', 0) / 60:.1f} min | AstroPhoto Studio {__version__}")
        exif = Image.Exif()
        exif[0x010E] = desc  # ImageDescription
        exif[0x0131] = f"AstroPhoto Studio {__version__}"  # Software
        im8.save(base + ".jpg", quality=int(quality), subsampling=0, optimize=True, exif=exif)
        files = {"jpg": base + ".jpg"}
        if tiff:
            import tifffile
            tifffile.imwrite(base + ".tif", (img * 65535 + 0.5).astype(np.uint16), photometric="rgb",
                             compression="zlib", description=desc)
            files["tif"] = base + ".tif"
        json.dump({"params": {**DEFAULTS, **params}, "linear_info": info, "meta": self.meta},
                  open(base + ".json", "w"), indent=1, default=_json_default)
        files["json"] = base + ".json"
        files["size"] = [int(img.shape[1]), int(img.shape[0])]
        return files

    def export_linear_fits(self, params: dict) -> str:
        lin, _ = self.linear(params)
        path = self._p("exports/linear_processed.fits")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _save_fits(path, lin)
        return path

    # ------------------------------------------------------------- local copy of the subs
    @contextlib.contextmanager
    def local_copy(self, progress=None):
        """Read the subs from a local copy for the duration of a job (``ASTROPHOTO_LOCAL_COPY``):
        the folder (``ASTROPHOTO_LOCAL_DIR``, default ``<workdir>/.local_copy``) is cleared, the
        dataset's light frames are copied into it, every read of a sub's pixels or header goes to the
        copy (frames.local_path; worker processes included), and the copy is removed when the job
        ends, however it ends.  For a library on a NAS: each stage reads every sub again (analysis,
        integration, the ImageMM preparation), from the local disk instead of the network.  Off, or
        with nothing to copy, this does nothing."""
        from .frames import LOCAL_ENV, local_copy_path
        if not local_copy_enabled():
            yield None
            return
        root = local_copy_dir(os.path.dirname(self.dir))
        shutil.rmtree(root, ignore_errors=True)
        os.makedirs(root, exist_ok=True)
        prev = os.environ.get(LOCAL_ENV)
        try:
            if not self.infos:
                self.scan(progress)
            paths = sorted({i.path for i in self.infos})
            total = sum(os.path.getsize(p) for p in paths)
            done = 0
            for n, p in enumerate(paths):
                q = local_copy_path(p, root)
                os.makedirs(os.path.dirname(q), exist_ok=True)
                shutil.copyfile(p, q)
                done += os.path.getsize(p)
                if progress:
                    progress(n + 1, len(paths), f"Copying the subs to the local disk ({n + 1}/{len(paths)}, "
                                                f"{done / 2**30:.1f} of {total / 2**30:.1f} GB)")
                self._check_cancel()
            os.environ[LOCAL_ENV] = root
            yield root
        finally:
            if prev is None:
                os.environ.pop(LOCAL_ENV, None)
            else:
                os.environ[LOCAL_ENV] = prev
            shutil.rmtree(root, ignore_errors=True)

    # ------------------------------------------------------------- one-shot
    def run_all(self, stack_params=None, proc_params=None, progress=None, **export_kw):
        t0 = time.time()
        sp = {**STACK_DEFAULTS, **(stack_params or {})}
        with self.local_copy(progress):                # the stages that read the subs
            self.run_analysis(sp["sensitivity"], progress)
            self.run_stack(sp, progress)
            self.plate_solve(progress)                 # catalogue stars for the colour calibration
            if float((proc_params or {}).get("denoise", DEFAULTS["denoise"])) > 0:
                self.run_denoise(sp, progress)
        if sp["star_remover"]:
            self.train_star_remover(proc_params, sp, progress)
        fin = None
        if sp["autofinish"]:
            fin = self.autofinish(proc_params, progress)
            proc_params = fin["params"]
        files = self.export(proc_params or {}, progress=progress, **export_kw)
        if fin:
            files["autofinish"] = fin
        files["seconds"] = round(time.time() - t0, 1)
        return files

    def autofinish(self, params: dict | None = None, progress=None) -> dict:
        """Tune the processing settings and fit a colour grade to reference images of the target
        (autofinish.py): {params, report}, also saved as autofinish.json."""
        from .autofinish import autofinish
        with self.lock:
            return autofinish(self, params, progress)

    def autofinish_result(self) -> dict | None:
        p = self._p("autofinish.json")
        try:
            return json.load(open(p)) if os.path.exists(p) else None
        except Exception:
            return None


def _resize_to(img: np.ndarray, shape) -> np.ndarray:
    """Resample an image (or a coverage map) onto another grid of the same field."""
    interp = cv2.INTER_LANCZOS4 if img.ndim == 3 else cv2.INTER_LINEAR
    out = cv2.resize(img, (shape[1], shape[0]), interpolation=interp)
    return np.maximum(out, 0) if img.ndim == 2 else out


def autostretch(img: np.ndarray, target: float = 0.2) -> np.ndarray:
    """Linked screen-transfer-function autostretch (PixInsight STF style)."""
    x = img.astype(np.float32)
    L = luminance(x) if x.ndim == 3 else x
    med = np.median(L[::4, ::4])
    mad = 1.4826 * np.median(np.abs(L[::4, ::4] - med))
    lo = med - 2.8 * mad
    hi = np.percentile(L[::4, ::4], 99.95)
    x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    m0 = (med - lo) / max(hi - lo, 1e-6)
    # midtone transfer so that the median lands on target
    m = m0 * (target - 1) / (2 * m0 * target - m0 - target)
    x = ((m - 1) * x) / ((2 * m - 1) * x - m)
    if x.ndim == 3:
        # neutralise the background for display
        bgc = np.median(x[::4, ::4].reshape(-1, 3), axis=0)
        x = np.clip(x - (bgc - bgc.mean()), 0, 1)
    return np.clip(x, 0, 1).astype(np.float32)
