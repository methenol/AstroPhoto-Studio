"""AstroPhoto Studio web UI (FastAPI) on the Siril backend (``sirilpipe``).

    python -m webui.server            # http://127.0.0.1:8000
    python -m webui.server --host 0.0.0.0 --port 8080 --images /path/to/telescope/exports

The module imports without any of the external tools installed: ``tools_status()`` then reports
them missing and the Tools panel says so.
"""
from __future__ import annotations

import argparse
import glob
import inspect
import io
import json
import os
import re
import sys
import threading
import time
import traceback
import uuid
import warnings

import cv2
import numpy as np
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from sirilpipe import __version__  # noqa: E402
from sirilpipe import finish as finish_mod  # noqa: E402
from sirilpipe import tools as tools_mod  # noqa: E402
from sirilpipe.post import DEFAULTS, EXTRA_DEFAULTS, PARAM_SPEC  # noqa: E402
from sirilpipe.session import Cancelled, PreviewSuperseded, Session  # noqa: E402
from sirilpipe.util import clean_json, dataset_slug, restoration_done, slugify, split_folders  # noqa: E402

CONFIG = {"images": os.path.join(ROOT, "images"), "workdir": os.path.join(ROOT, "output")}

app = FastAPI(title="AstroPhoto Studio", version=__version__)
app.add_middleware(GZipMiddleware, minimum_size=4096)
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
class _NoCacheStatic(StaticFiles):
    """Static files answer with ``Cache-Control: no-cache`` so browsers revalidate them on every load."""
    async def get_response(self, path, scope):
        r = await super().get_response(path, scope)
        r.headers["Cache-Control"] = "no-cache"
        return r


app.mount("/static", _NoCacheStatic(directory=STATIC), name="static")

SESSIONS: dict[str, Session] = {}
JOBS: dict[str, dict] = {}   # every job: queued, running, paused and finished (saved in <workdir>/jobs.json)
JOB_LOCK = threading.Lock()   # held while a job runs (one heavy job at a time: memory, and RC-Astro's licence)
PREVIEW_CACHE: dict[str, bytes] = {}
CONTROL: dict[str, threading.Event] = {}   # job id -> "pause requested" (the server's own pause point)

# --------------------------------------------------------------------------- the v1 contract tables
# (api-contract.md: same keys, labels and profile names as the pre-Siril server; keys the Siril
# backend has no use for - ImageMM / Noise2Noise / network options - stay accepted and are ignored)

STACK_DEFAULTS = {
    "mode": "auto",          # auto | drizzle | demosaic
    "scale": 1.0,            # output up-sampling (1, 1.5, 2; 3, 4 with Cosmic Clarity)
    "superres_method": "drizzle",   # drizzle = Siril Bayer drizzle at `scale`; cosmic_clarity = stack at 1x, Cosmic Clarity Super-Resolution 2/3/4x on the finished image
    "sigma_low": 4.0,
    "sigma_high": 3.0,
    "local_norm": True,
    "sensitivity": 1.0,      # frame-rejection aggressiveness
    "pattern_correction": True,
    "denoise_iters": 2000,
    "ai_deconvolution": True,
    "deconv_method": "auto",   # deconvolution of the objects: auto = BlurXTerminator when licensed, else Siril RL; bxt | graxpert | rl (that tool only);
                               # imagemm = VeraLux ImageMM (needs torch); none.  Cosmic Clarity Sharpen is the finishing setting cc_sharpen (stretched image)
    "deconv_stars_method": "auto",  # deconvolution of the stars, the same choices (post.deconv_choice; a request naming only deconv_method: the stars follow it)
    "denoise_method": "auto",  # linear noise reduction: auto = NXT when licensed, else GraXpert / Siril; nxt | graxpert | siril (that tool only); none.  CC Denoise: cc_denoise
    "rejection": "auto",       # stack rejection: auto = winsorized (MAD for drizzle > 1x); winsorized | sigma | mad | linear | median | none (workflow.REJECTIONS)
    "stack_weighting": "wfwhm",  # stack -weight=: wfwhm | nbstars | noise | nbstack | none
    "drizzle_pixfrac": 0.0,    # drizzle drop size; 0 = the rule's (1 / scale)
    "drizzle_kernel": "square",  # drizzle kernel: square | point | turbo | gaussian | lanczos2 | lanczos3
    "hypertune": False,        # Run everything: before each stage, an experiment-lab study of that stage on this dataset; the stage then runs with its best trial
    "hypertune_trials": 60,    # trials per stage study (the baseline included); 30-120 search the stages' spaces
    "satellite_removal": False,  # Cosmic Clarity satellite-trail removal on the stack before the restoration
    "graxpert_psfsize": 0.0,   # GraXpert deconvolution -psfsize (Image FWHM, px); 0 = the star FWHM measured on the linear image
    # VeraLux ImageMM (sirilpipe/veralux_imagemm.py; the script's GUI defaults).  The 1.x ImageMM keys (imagemm_delta,
    # imagemm_stop, imagemm_max_iters, imagemm_n2n, ...) stay accepted and are ignored.
    "imagemm_r": 1,                # super-resolution factor (1..3); restored_resolution sets it too
    "imagemm_sigma": 0.0,          # SR PSF sigma (latent px); 0 = the script's 1.0
    "imagemm_iterations": 150,     # cap; the paper's Appendix-C rule stops it (M 31: 92); the script's GUI default is 20
    "imagemm_huber": 0.05,         # Huber delta (normalised units)
    "imagemm_kappa": 2.0,          # update clipping [1/kappa, kappa] (Eq. 9)
    "imagemm_mu": 0.1,             # effective-pixel fraction of the stopping rule
    "imagemm_epsilon": 1e-5,       # stopping epsilon
    "imagemm_formal": False,       # formal mode (paper assumptions): zero padding, no PSF TV, v = shot y + read^2
    "imagemm_shot_noise": 1.0,     # formal mode: shot-noise scale
    "imagemm_read_noise": 0.0,     # formal mode: read noise (normalised units)
    "imagemm_saturation": 0.0,     # saturation level of the frames; 0 = measured on the raw frames
    # the pipeline's fixes to the script (sirilpipe/veralux_imagemm.py docstring), measured on IC 1805 / M 42 crops
    "imagemm_variance": "model",   # variance at the model prediction ("data": the script's, noisy on faint skies)
    "imagemm_psf_channels": True,  # one PSF per colour channel (the script: green for all)
    "imagemm_lower_bound": "local",  # latent >= the local background - 3 sigma (LBCA): no dark moat round stars; "global", False
    "imagemm_cosmetic": True,      # Siril find_cosme on every sub: warm / cold pixels (ImageMM has no rejection)
    "imagemm_latent_sigma": 1.0,   # g_sigma of the latent (Eq. 11; the paper's 1 px at r = 1, the script's 0.45)
    "imagemm_bound_sigma": 1.0,    # lower bound's margin below the local background (latent sigma): 3 left rings, 0 a halo
    "imagemm_saturated_points": False,  # saturated cores as point sources at iteration 10 (superseded by the photometry)
    "imagemm_saturated_photometry": True,  # saturated cores' colour from their unsaturated wings (PSF photometry), shape g_sigma
    "imagemm_psf_asymmetry": True,  # the optics' fixed one-sided wing (m = 1, 2) from the frames' mean, in every kernel
    "imagemm_psf_radial_wings": True,  # the PSF's extended wing joined from unsaturated + saturated stars (Infante-Sainz 2020)
    "imagemm_bloom_grow": 4,       # px round each frame's saturated pixels treated as missing (charge spill / non-linearity)
    "imagemm_point_init": True,    # saturated stars start as points (the censored cores left the starting halo in place)
    "imagemm_censor_saturated": False,  # saturated pixels as censored data (Whyte et al.) instead of missing (the paper's mask)
    "imagemm_reference_psf": True, "imagemm_photometry": True, "imagemm_noise_model": "ptc", "imagemm_huber_sigma": 2.0,
    "imagemm_subsky": False,       # the script's per-frame seqsubsky (removes part of large targets)
    "imagemm_n2n": True,           # v1 contract field (reported by /api/v1/profiles "effective"); no effect
    "n2n_split": "alternate",
    "n2n_loss": "asinh_mse",
    "star_remover": True,
    "star_remover_iters": 3000,
    "autofinish": True,
    "deconv_target_fwhm": 0.0,
    "deconv_iters": 6000,
    "deconv_sources": 1.0,
    "network_groups": 0,
    "device": "auto",
}


def default_scale(deconv_method: str) -> float:
    """The stack scale a restoration runs at unless set (contract: 1.0 for imagemm, else 2.0).  With the
    Siril backend a scale of 2 asks for 2x drizzle, which run_stack grants only when the data supports it
    (enough dithered subs, FWHM under 2 px) and otherwise records why it stayed at 1x."""
    return 1.0 if deconv_method == "imagemm" else 2.0


# The 1.x profile names are part of the API contract (the telescope controller submits them); their
# stack_params map onto the Siril backend's restoration choices:
#   deconv_method  auto    = BlurXTerminator when licensed, else Siril makepsf + Richardson-Lucy
#                  rl      = Siril Richardson-Lucy even when BlurXTerminator is licensed
#                  imagemm = VeraLux ImageMM multi-frame restoration of the subs (slow; needs IMAGEMM=cuda)
#                  none    = no deconvolution
#                  bxt / graxpert = that tool only (not used by the 1.x profiles); the former graxpert_obj / graxpert_stellar /
#                  graxpert_both stay accepted.  deconv_method is the objects' tool; the profiles leave deconv_stars_method out, so the
#                  stars follow it (_run_session_job)
# ai_deconvolution and imagemm_n2n (the 1.x ImageMM's Noise2Noise pass) are accepted for compatibility and have no effect.
PROFILES = {
    "default": {"label": "Pipeline defaults (STACK_DEFAULTS)", "stack_params": {}},
    "imagemm": {"label": "ImageMM: multi-frame restoration of every registered sub (slow, GPU)",
                "stack_params": {"deconv_method": "imagemm", "imagemm_n2n": True}},
    "n2n-imagemm": {"label": "ImageMM multi-frame restoration (same as imagemm with this backend)",
                    "stack_params": {"deconv_method": "imagemm", "imagemm_n2n": True}},
    "n2n-network": {"label": "Fast restoration: BlurXTerminator + NoiseXTerminator when licensed, else Siril "
                             "deconvolution + GraXpert denoise; 2x drizzle when the data supports it "
                             "(was: Noise2Noise + deconvolution network)",
                    "stack_params": {"deconv_method": "auto", "ai_deconvolution": True,
                                     "scale": default_scale("auto")}},
    "n2n-rl": {"label": "Siril Richardson-Lucy (TV) deconvolution, no AI restoration "
                        "(was: Noise2Noise denoiser + Richardson-Lucy)",
               "stack_params": {"deconv_method": "rl", "ai_deconvolution": False, "scale": default_scale("rl")}},
}

PRESETS = {
    "Balanced": {},
    "Vivid nebula": {"saturation": 1.8, "local_contrast": 1.4, "stretch": 0.09, "contrast": 4.0, "oiii_boost": 1.0,
                     "star_reduction": 0.5, "star_intensity": 0.8},
    "Starless-ish": {"star_reduction": 0.8, "star_intensity": 0.45, "local_contrast": 1.0},
    "Galaxy / broadband": {"palette": "natural", "stretch": 0.09, "contrast": 4.0, "saturation": 1.4,
                           "local_contrast": 0.8, "star_reduction": 0.2, "bg_degree": 2},
    "Natural colour": {"palette": "natural", "saturation": 1.3, "scnr": 0.6},
    "Deep & dark": {"stretch": 0.06, "black_point": 0.0, "contrast": 6.0, "local_contrast": 1.0},
}




def get_session(folder: str) -> Session:
    """The session of a dataset: one folder, or several (same target, several nights) joined
    with os.pathsep, stacked together."""
    folders = split_folders(folder)
    if not folders:
        raise HTTPException(400, "No folder given")
    missing = [f for f in folders if not os.path.isdir(f)]
    if missing:
        raise HTTPException(404, f"Folder not found: {', '.join(missing)}")
    key = dataset_slug(folders)
    if key not in SESSIONS:
        SESSIONS[key] = Session(folders, CONFIG["workdir"])
    return SESSIONS[key]


def _call(fn, *args, **kw):
    """Call ``fn`` with the keyword arguments its signature accepts (the tool wrappers take
    different optional ones: progress, host, ...)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args, **kw)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(*args, **kw)
    return fn(*args, **{k: v for k, v in kw.items() if k in sig.parameters})


# ------------------------------------------------------------------ headers & devices


def read_info(path: str) -> dict:
    """object / filter / exptime / instrument / date of one sub, from its FITS header (no pixels
    read).  Uses ``sirilpipe.dataset`` (DWARF names, shotsInfo.json, ...); for files it does not
    count as lights (a ``stacked-*.fits`` it skips by name) the plain header is used."""
    from sirilpipe import dataset
    fr = dataset.read_frame(path)
    if fr is not None:
        return {"object": fr.object, "filter": fr.filter, "exptime": float(fr.exptime or 0.0),
                "instrument": fr.telescope or fr.instrument, "date": (fr.date_obs or "")[:10] or None}
    from astropy.io import fits
    h = fits.getheader(path)
    exp = h.get("EXPTIME", h.get("EXPOSURE", 0.0))
    return {"object": str(h.get("OBJECT") or "").strip(), "filter": str(h.get("FILTER") or "").strip(),
            "exptime": float(exp or 0.0), "instrument": str(h.get("TELESCOP") or h.get("INSTRUME") or "").strip(),
            "date": str(h.get("DATE-OBS") or "")[:10] or None}


_DEVICES: dict = {}


def device_info() -> dict:
    """The GPUs, from nvidia-smi (no torch): ``{"default": "cuda"|"cpu", "gpus": [{id, name, vram_gb}]}``.
    The Siril backend decides per tool what runs on the GPU; this is only what the UI badge shows."""
    if _DEVICES:
        return _DEVICES
    gpus: list[dict] = []
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=4)
        for line in out.stdout.strip().splitlines():
            i, name, mem = [x.strip() for x in line.split(",")]
            gpus.append({"id": int(i), "name": name, "vram_gb": round(float(mem) / 1024, 1)})
    except Exception:
        pass
    _DEVICES.update({"default": "cuda" if gpus else "cpu", "gpus": gpus})
    return _DEVICES


def tools_summary() -> dict:
    """``{name: bool}`` for /api/v1 (additive): which external tools work."""
    try:
        st = tools_status()
        return {k: bool(v.get("ok")) for k, v in st.items() if isinstance(v, dict)}
    except Exception:
        return {}


def tools_status() -> dict:
    """tools.tools_status(), never raising: a missing tool is ``ok: false``, a broken module is an error entry."""
    try:
        return clean_json(tools_mod.tools_status())
    except Exception as e:  # the tools package must never take the server down
        return {n: {"ok": False, "error": f"{type(e).__name__}: {e}"} for n in ("siril", "graxpert", "starnet", "rcastro", "cosmic_clarity", "gaia")}


# ------------------------------------------------------------------ pages

@app.get("/", response_class=HTMLResponse)
def index():
    """The page, with every static asset URL stamped with the server version: a browser that cached the
    scripts of an older build would otherwise run them against the new page (the removed device selector
    broke the whole UI that way after the 2.0 upgrade)."""
    html = open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
    html = re.sub(r'(/static/[\w./-]+\.(?:js|css))', rf"\1?v={__version__}", html)
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


@app.get("/api/system")
def system():
    return clean_json({"version": __version__, "backend": "siril", "devices": device_info(), "defaults": {**DEFAULTS, **EXTRA_DEFAULTS},
                       "stack_defaults": STACK_DEFAULTS, "param_spec": list(PARAM_SPEC), "presets": PRESETS,
                       "images_root": CONFIG["images"], "tools": tools_status(), "job_kinds": JOB_KINDS})


CALIB_DIRS = frozenset({"cali_frame", "dwarf_dark", "dark", "darks", "flat", "flats", "bias", "biases", "offsets",
                        "calib", "restacked", "solving_failed"})   # calibration libraries etc. are not datasets
FITS_SUFFIXES = (".fit", ".fits", ".fts")
SCAN_WORKERS = 16            # directory listings in flight: a network share is latency-bound, not bandwidth-bound
SCAN_TTL = 15.0              # seconds a library scan is reused (the page asks for it on every refresh)
_SCAN_MEMO: dict = {}
_SCAN_LOCK = threading.Lock()


def _list_dir(d: str) -> tuple[list[str], list[str]]:
    """(sub-folders to descend into, FITS file names) of one folder, from a single directory listing.
    ``DirEntry.is_dir()`` comes from the listing itself (d_type) on local disks, NFS and SMB, so no file
    is stat'ed; hidden folders and calibration folders are not descended into."""
    subs, fits_names = [], []
    try:
        with os.scandir(d) as it:
            for e in it:
                name = e.name
                if name.startswith("."):
                    continue
                try:
                    if e.is_dir():
                        if name.lower() not in CALIB_DIRS:
                            subs.append(e.path)
                        continue
                except OSError:
                    continue
                if name.lower().endswith(FITS_SUFFIXES):
                    fits_names.append(name)
    except OSError:
        pass
    return subs, sorted(fits_names)


def scan_library(root: str) -> dict[str, list[str]]:
    """{folder: its FITS file names} for every folder under ``root`` (``root`` included) that holds FITS
    files.  Breadth-first, every level's listings in parallel; symlinked folders are followed once."""
    from concurrent.futures import ThreadPoolExecutor
    found: dict[str, list[str]] = {}
    seen = {os.path.realpath(root)}
    level = [root]
    with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as ex:
        while level:
            nxt = []
            for d, (subs, names) in zip(level, ex.map(_list_dir, level)):
                if names:
                    found[d] = names
                for sd in subs:
                    rp = os.path.realpath(sd)
                    if rp not in seen:
                        seen.add(rp)
                        nxt.append(sd)
            level = nxt
    return found


def _library_cache_path() -> str:
    return os.path.join(CONFIG["workdir"], ".library_cache.json")


def _dataset_infos(items: list[tuple[str, list[str]]]) -> dict[str, dict]:
    """Header summary (object, filter, exptime, instrument, date) of each dataset's first sub.  Kept in
    ``<workdir>/.library_cache.json`` keyed by folder, first file and number of files, so a header is read
    over the network only for a new or changed dataset; the reads that are needed run in parallel."""
    from concurrent.futures import ThreadPoolExecutor
    cache = _read_json(_library_cache_path()) or {}
    out, todo = {}, []
    for d, names in items:
        c = cache.get(d)
        if c and c.get("first") == names[0] and c.get("n") == len(names):
            out[d] = c.get("info") or {}
        else:
            todo.append((d, names))

    def one(item):
        d, names = item
        try:
            fi = read_info(os.path.join(d, names[0]))
            return d, {k: fi[k] for k in ("object", "filter", "exptime", "instrument", "date")}
        except Exception:
            return d, {}
    if todo:
        with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as ex:
            for d, info in ex.map(one, todo):
                out[d] = info
        keep = {d: {"first": names[0], "n": len(names), "info": out[d]} for d, names in items}
        try:
            os.makedirs(CONFIG["workdir"], exist_ok=True)
            tmp = _library_cache_path() + ".tmp"
            with open(tmp, "w") as f:
                json.dump(clean_json({**{k: v for k, v in cache.items() if k not in keep}, **keep}), f)
            os.replace(tmp, _library_cache_path())
        except OSError:
            pass
    return out


@app.get("/api/datasets")
def datasets(root: str | None = None, refresh: bool = False):
    """The FITS folders under the images root.  One parallel ``scandir`` walk (``scan_library``) and cached
    headers (``_dataset_infos``): on a network share the old recursive glob stat'ed every file of the
    library and globbed every folder once per extension, and the first sub's header was read again for
    every dataset on every call.  The scan is reused for ``SCAN_TTL`` seconds (``refresh`` forces a new one)."""
    root = os.path.abspath(os.path.expanduser(root or CONFIG["images"]))
    if not os.path.isdir(root):
        return {"root": root, "datasets": []}
    with _SCAN_LOCK:                      # concurrent requests share one scan
        memo = _SCAN_MEMO.get(root)
        if refresh or not memo or time.time() - memo[0] > SCAN_TTL:
            found = scan_library(root)
            order = ([root] if root in found else []) + sorted(d for d in found if d != root)
            items = [(d, found[d]) for d in order]
            memo = (time.time(), items, _dataset_infos(items))
            _SCAN_MEMO[root] = memo
    _, items, infos = memo
    out = []
    for d, names in items:
        info = {"path": d, "name": os.path.relpath(d, root) if d != root else os.path.basename(d), "n_fits": len(names)}
        fi = infos.get(d) or {}
        if fi:
            info.update(fi)
            try:
                info["total_min"] = round(float(fi["exptime"]) * len(names) / 60, 1)
            except (KeyError, TypeError, ValueError):
                pass
        cache_dir = os.path.join(CONFIG["workdir"], slugify(d))       # local disk: cheap
        info["cached"] = {"analysed": os.path.exists(os.path.join(cache_dir, "frames.json")),
                          "stacked": os.path.exists(os.path.join(cache_dir, "stack.fits")),
                          "denoised": bool(restoration_done(cache_dir))}
        out.append(info)
    return {"root": root, "datasets": out, "sep": os.pathsep}


def _read_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _scanned(s: Session) -> Session:
    """Read the subs' headers once per session (the dataset card and the thumbnails need them)."""
    if hasattr(s, "scan") and not getattr(s, "infos", True):
        try:
            s.scan()
        except Exception as e:
            raise HTTPException(400, f"{type(e).__name__}: {e}")
    return s


def _ui_status(s: Session) -> dict:
    """Session.status() plus the keys the UI's step ticks read (the pre-Siril names)."""
    st = dict(s.status() or {})
    cal = st.get("calibration")
    st.setdefault("folder", os.pathsep.join(s.folders))
    st.setdefault("folders", list(s.folders))
    st.setdefault("workdir", s.dir)
    st.setdefault("stacked", bool(st.get("has_stack")))
    st.setdefault("denoised", bool(st.get("has_linear")))
    st.setdefault("star_remover", bool(st.get("has_starless")))
    st.setdefault("autofinished", bool(st.get("has_autofinish")))
    st.setdefault("analysed", bool(st.get("analysed") or st.get("has_analysis")
                                   or os.path.exists(os.path.join(s.dir, "frames.json"))))
    st.setdefault("narrowband", bool(st.get("is_dualband")))
    st.setdefault("telescope", st.get("instrument"))
    st.setdefault("calibration_checked", bool(cal))
    if "stack_meta" not in st:
        st["stack_meta"] = _read_json(os.path.join(s.dir, "stack_meta.json")) or {}
    if isinstance(cal, str):                      # a bare summary string
        st["calibration"] = {"summary": cal}
    return st


def _frames_payload(s: Session) -> dict:
    r = s.frames()
    if isinstance(r, list):
        r = {"frames": r}
    return {"frames": r.get("frames") or [], "medians": r.get("medians"), "sensitivity": r.get("sensitivity", 1.0)}


@app.post("/api/open")
def open_dataset(body: dict = Body(...)):
    s = _scanned(get_session(body["folder"]))
    fr = _frames_payload(s)
    return clean_json({"status": _ui_status(s), "frames": fr["frames"], "medians": fr["medians"]})


@app.get("/api/frames")
def frames(folder: str):
    return clean_json(_frames_payload(get_session(folder)))


@app.post("/api/frames/override")
def override(body: dict = Body(...)):
    s = get_session(body["folder"])
    s.set_override(body["name"], body.get("accepted"))
    return clean_json({"frames": _frames_payload(s)["frames"]})


@app.post("/api/frames/sensitivity")
def sensitivity(body: dict = Body(...)):
    """Re-select the frames at another rejection sensitivity (the stored measurements are reused)."""
    s = get_session(body["folder"])
    value = float(body["sensitivity"])
    fn = getattr(s, "reselect", None) or getattr(s, "set_sensitivity", None)
    if fn is None:
        raise HTTPException(501, "this backend applies the rejection sensitivity when it analyses the frames")
    r = fn(value)
    return clean_json({"frames": r if isinstance(r, list) else _frames_payload(s)["frames"]})


@app.get("/api/calibration")
def calibration_get(folder: str):
    """The calibration panel: telescope profile, masters chosen and available, notes."""
    s = get_session(folder)
    return clean_json({**s.calibration_info(), "status": _ui_status(s)})


@app.post("/api/calibration")
def calibration_set(body: dict = Body(...)):
    """Save this dataset's calibration settings (enabled, library folder, dark / flat / bias:
    auto | none | a master's path).  Run the Calibrate step (or Analyse) to apply them."""
    s = get_session(body["folder"])
    prefs = body.get("prefs") or {}
    r = s.set_calibration(prefs)
    return clean_json({"prefs": r if isinstance(r, dict) else prefs})


def _autostretch_jpeg(arr: np.ndarray, side: int, q: int = 85) -> bytes:
    """Quick display stretch (percentile clip, then an MTF taking the median to 0.25) of a 2-D or 3-plane image."""
    a = np.asarray(arr, dtype=np.float32)
    a = np.moveaxis(a, 0, -1) if a.ndim == 3 and a.shape[0] in (1, 3) else a
    if a.ndim == 2:
        a = a[..., None]
    scale = side / max(a.shape[:2])
    if scale < 1:
        a = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        a = a[..., None] if a.ndim == 2 else a
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.8)
    a = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
    med = float(np.median(a))
    t = 0.25                                             # the median goes to 0.25 (midtones transfer function)
    if 0 < med < 1:
        m = med * (1 - t) / (med * (1 - 2 * t) + t)
        a = ((m - 1) * a) / ((2 * m - 1) * a - m)
    img = np.clip(a, 0, 1)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    ok, buf = cv2.imencode(".jpg", (img[..., ::-1] * 255 + 0.5).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()


def _thumbnail(s: Session, name: str, size: int) -> bytes:
    """A sub's thumbnail: the session's own, else a stretched, binned view of the mosaic."""
    fn = getattr(s, "frame_thumbnail", None)
    if fn:
        return fn(name, size)
    from astropy.io import fits
    path = next((p for p in (name, *[os.path.join(d, name) for d in s.folders]) if os.path.isfile(p)), None)
    if path is None:
        raise HTTPException(404, f"no frame {name}")
    data = fits.getdata(path).astype(np.float32)
    if data.ndim == 2:                            # CFA mosaic: 2x2 binning is a cheap RGB-less luminance
        h, w = data.shape[0] // 2 * 2, data.shape[1] // 2 * 2
        data = data[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    return _autostretch_jpeg(data, size)


@app.get("/api/thumb")
def thumb(folder: str, name: str, size: int = 360):
    try:
        return Response(_thumbnail(_scanned(get_session(folder)), name, size), media_type="image/jpeg")
    except KeyError:
        raise HTTPException(404, f"no frame {name}")


# ------------------------------------------------------------------ jobs

JOB_KINDS = {"calibrate": "Calibrate", "analyse": "Analyse frames", "stack": "Register & integrate", "denoise": "Restore",
             "starnet": "Train star remover", "autofinish": "Auto-finish", "all": "Run everything & export",
             "export": "Export", "tools": "Tools"}

ACTIVE = ("running", "paused", "queued")
QUEUE: list[str] = []                     # queued job ids, first to run first
QUEUE_LOCK = threading.Lock()
QUEUE_EVENT = threading.Event()
_WORKER: dict = {}
CANCELLED: set[str] = set()                # job ids with a cancel request (the server's own checkpoint)
_SAVED = {"t": 0.0}


def _jobs_file() -> str:
    return os.path.join(CONFIG["workdir"], "jobs.json")


def _save_jobs(force: bool = False):
    """Persist every job (state, stage history, settings) so a page refresh or a server restart
    loses nothing; progress updates are throttled to one write every few seconds."""
    import json as _json
    now = time.time()
    if not force and now - _SAVED["t"] < 5:
        return
    _SAVED["t"] = now
    try:
        os.makedirs(CONFIG["workdir"], exist_ok=True)
        tmp = _jobs_file() + ".tmp"
        with open(tmp, "w") as f:
            _json.dump(clean_json({"jobs": [dict(j) for j in list(JOBS.values())], "queue": list(QUEUE)}), f)
        os.replace(tmp, _jobs_file())
    except Exception:
        pass


def _load_jobs():
    """Jobs of earlier server runs: finished ones as history; anything that was queued,
    running or paused when the server stopped is marked interrupted (it can be run again)."""
    import json as _json
    if JOBS or not os.path.exists(_jobs_file()):
        return
    try:
        st = _json.load(open(_jobs_file()))
    except Exception:
        return
    for j in st.get("jobs", []):
        if j.get("state") in ACTIVE:
            j["state"] = "interrupted"
            j["message"] = "Interrupted: the server stopped while this job was " + ("waiting" if j.get("started") is None else "running")
            j["ended"] = j.get("ended") or j.get("updated") or time.time()
        JOBS[j["id"]] = j


def _ensure_worker():
    if _WORKER.get("t") is None or not _WORKER["t"].is_alive():
        _WORKER["t"] = threading.Thread(target=_worker_loop, daemon=True)
        _WORKER["t"].start()


def _worker_loop():
    """Runs queued jobs one at a time, in order (one heavy job at a time: memory)."""
    while True:
        QUEUE_EVENT.wait(1.0)
        with QUEUE_LOCK:
            job_id = QUEUE.pop(0) if QUEUE else None
            if not QUEUE:
                QUEUE_EVENT.clear()
        if job_id and job_id in JOBS and JOBS[job_id]["state"] == "queued":
            _run_job(job_id)


def _stage_key(msg: str) -> str:
    """A stage name without its counters: 'ImageMM: cutout 5/84 (...)' -> 'ImageMM: cutout #/#'."""
    import re
    return re.sub(r"\d+(\.\d+)?", "#", msg.split(" (")[0]).strip()


def _active_seconds(j: dict, now: float | None = None) -> float:
    if not j.get("started"):
        return 0.0
    end = j.get("ended") or now or time.time()
    paused = j.get("paused_total", 0.0) + ((end - j["pause_started"]) if j.get("pause_started") else 0.0)
    return max(0.0, end - j["started"] - paused)


def _typical_seconds(j: dict) -> tuple[float | None, str | None]:
    """How long this kind of job took before: on the same dataset, or on another one scaled by
    the number of subs (the work of every stage grows with it)."""
    same = [h for h in list(JOBS.values()) if h is not j and h["state"] == "done" and h["kind"] == j["kind"] and h.get("variant") == j.get("variant")
            and h.get("folder") == j.get("folder") and h.get("active_seconds")]
    if same:
        h = max(same, key=lambda q: q.get("ended", 0))
        return h["active_seconds"], "the last run on this dataset"
    other = [h for h in list(JOBS.values()) if h is not j and h["state"] == "done" and h["kind"] == j["kind"] and h.get("variant") == j.get("variant")
             and h.get("active_seconds") and h.get("n_subs") and j.get("n_subs")]
    if other:
        h = max(other, key=lambda q: q.get("ended", 0))
        return h["active_seconds"] * j["n_subs"] / h["n_subs"], f"a run on another dataset, scaled to {j['n_subs']} subs"
    return None, None


def _job_view(j: dict) -> dict:
    """A job with its live timings: elapsed and active time, the current stage's ETA from its own
    rate of progress, and an estimate of the whole job from earlier runs."""
    now = time.time()
    v = {k: val for k, val in j.items() if k not in ("params", "stack_params", "export_opts")}
    v["label"] = JOB_KINDS.get(j["kind"], j["kind"]) + (" (hyper-tuned)" if j.get("variant") == "hypertune" else "")
    v["elapsed"] = round(((j.get("ended") or now) - j["started"]) if j.get("started") else 0.0, 1)
    v["active"] = round(_active_seconds(j, now), 1)
    if j["state"] in ("running", "paused") and j.get("stage_started"):
        t_stage = (j.get("pause_started") or now) - j["stage_started"] - j.get("stage_paused", 0.0)
        p = j.get("progress", 0.0)
        v["stage_eta"] = round(t_stage * (1 - p) / p, 0) if p > 0.02 and t_stage > 5 else None
    typ, basis = _typical_seconds(j)
    if typ is not None and j["state"] in ACTIVE:
        v["typical_total"] = round(typ, 0)
        v["typical_basis"] = basis
        v["total_eta"] = round(max(typ - v["active"], 0), 0)
    if j["state"] == "queued":
        with QUEUE_LOCK:
            pos = QUEUE.index(j["id"]) if j["id"] in QUEUE else None
        v["position"] = (pos + 1) if pos is not None else None
        ahead = [JOBS[q] for q in QUEUE[:pos] if q in JOBS] if pos is not None else []
        running = [r for r in list(JOBS.values()) if r["state"] in ("running", "paused")]
        v["waiting_on"] = [{"id": r["id"], "label": JOB_KINDS.get(r["kind"], r["kind"]), "dataset": r.get("dataset"),
                            "state": r["state"]} for r in running + ahead]
    return v




# ------------------------------------------------------------------ running a job

def _set_flag(s, name: str, on: bool):
    """Raise or clear a session's cancel / pause request, whichever form the Session uses
    (``<name>_flag`` Event, or the ``<name>_requested`` bool)."""
    ev = getattr(s, f"{name}_flag", None)
    if isinstance(ev, threading.Event):
        ev.set() if on else ev.clear()
    else:
        setattr(s, f"{name}_requested", on)


def _norm_autofinish(res) -> dict:
    """Session.autofinish returns ``{"autofinish": {params, report}}``; the job result wants that
    wrapper, while the "all" kind needs the inner {params, report}."""
    return res.get("autofinish", res) if isinstance(res, dict) else {}


def _run_session_job(kind: str, s: Session, progress, params: dict, stack_params: dict, export_opts: dict):
    """Map a job kind onto the Session methods (api-contract.md section 2)."""
    sp = {**STACK_DEFAULTS, **stack_params}
    if "deconv_method" in stack_params and "deconv_stars_method" not in stack_params:
        sp["deconv_stars_method"] = ""           # one deconvolution named (an API client, a 1.x profile): the stars follow it (post.deconv_choice)
    # the auto-crop is decided at stack time, so the two processing settings that control it travel
    # with the stack parameters (Session.run_stack only receives those)
    for k in ("crop", "crop_threshold"):
        if k in params and k not in stack_params:
            sp[k] = params[k]
    if kind == "calibrate":
        return s.run_calibration(progress)
    if kind == "analyse":
        r = s.run_analysis(float(sp["sensitivity"]), progress, sp)
        return r if isinstance(r, dict) else {"n": r}
    if kind == "stack":
        r = s.run_stack(sp, progress)
        s.plate_solve(progress)                  # catalogue stars for the colour calibration
        return r
    if kind == "denoise":
        _call(s.run_denoise, sp, progress, params=params)
        return {"ok": True}
    if kind == "starnet":
        return s.train_star_remover(params, sp, progress)
    if kind == "autofinish":
        return {"autofinish": _norm_autofinish(s.autofinish(params, progress))}
    if kind == "export":
        return s.export(params, progress=progress, **export_opts)
    if kind == "all" and sp.get("hypertune"):
        # Hyper-tune each stage: an experiment-lab study of every stage before it runs (sirilpipe.hypertune); the studies appear in the Experiments tab
        from sirilpipe import hypertune
        return hypertune.run_all(s, sp, params, export_opts, progress, int(sp.get("hypertune_trials") or 60), workdir=CONFIG["workdir"])
    if kind == "all":
        s.run_analysis(float(sp["sensitivity"]), progress, sp)
        s.run_stack(sp, progress)
        s.plate_solve(progress)
        _call(s.run_denoise, sp, progress, params=params)
        if sp["star_remover"]:
            s.train_star_remover(params, sp, progress)
        fin = None
        if sp["autofinish"]:                     # the tuned settings are the ones exported
            fin = _norm_autofinish(s.autofinish(params, progress))
            params = fin["params"]
        result = s.export(params, progress=progress, **export_opts)
        if fin:
            result["autofinish"] = fin
        return result
    raise ValueError(f"unknown job kind {kind!r}")


def _tool_member(tool: str, attr: str):
    """``sirilpipe.tools.<tool>.<attr>``: the module's singleton (``graxpert`` / ``starnet`` /
    ``rcastro``, which share their status caches), else the wrapper class instantiated."""
    names = {"graxpert": "GraXpert", "rcastro": "RCAstro", "starnet": "StarNet", "cosmic_clarity": "CosmicClarity"}
    if tool not in names:
        raise ValueError(f"unknown tool {tool!r} (graxpert | rcastro | starnet | cosmic_clarity)")
    obj = getattr(tools_mod, tool, None) or getattr(tools_mod, names[tool])     # the module's singleton, else the class
    if inspect.isclass(obj):
        obj = obj()
    return getattr(obj, attr)


def _run_tools_job(params: dict, progress) -> dict:
    """The ``tools`` kind: model / tool downloads, run under the same lock as processing."""
    tool = params.get("tool")
    if tool == "graxpert":
        fn = _tool_member("graxpert", "download_models")
    elif tool == "rcastro":
        fn = _tool_member("rcastro", "download_models")
    elif tool == "starnet":
        fn = _tool_member("starnet", "download")
    elif tool == "cosmic_clarity":
        fn = _tool_member("cosmic_clarity", "download_models")
    else:
        raise ValueError(f"unknown tool {tool!r} (graxpert | rcastro | starnet | cosmic_clarity)")
    progress(0, 1, f"Downloading {tool} models" if tool != "starnet" else "Downloading StarNet")
    ret = _call(fn, progress=progress, accept_licence=bool(params.get("accept_licence")),
                kinds=params.get("models"))
    progress(1, 1, "Done")
    return {"tool": tool, "ok": True, **(ret if isinstance(ret, dict) else {})}


def _run_job(job_id: str):
    job = JOBS[job_id]
    kind, folder = job["kind"], job["folder"]
    params, stack_params, export_opts = job.get("params") or {}, job.get("stack_params") or {}, job.get("export_opts") or {}
    s = get_session(folder) if kind != "tools" else None
    if s is not None:
        _set_flag(s, "cancel", False)
        _set_flag(s, "pause", False)
    CANCELLED.discard(job_id)
    CONTROL.setdefault(job_id, threading.Event())

    def progress(i, n, msg):
        now = time.time()
        key = _stage_key(msg)
        if key != job.get("stage_key"):
            if job.get("stage_key"):
                job["stages"].append([job["stage_key"], round(now - job["stage_started"] - job.get("stage_paused", 0.0), 1)])
                job["stages"] = job["stages"][-100:]
            job["stage_key"], job["stage_started"], job["stage_paused"] = key, now, 0.0
        job["progress"] = i / max(n, 1)
        job["message"] = msg
        job["stage"] = msg
        job["updated"] = now
        short = msg.split(" (")[0].rsplit(" ", 1)[0]
        if not job["log"] or job["log"][-1][1] != short:
            job["log"].append([round(now - job["started"], 1), short])
            job["log"] = job["log"][-200:]
        _save_jobs()
        while CONTROL[job_id].is_set() and job_id not in CANCELLED:     # paused: wait here
            time.sleep(0.25)
        if job_id in CANCELLED or (s is not None and s.checkpoint()):   # checkpoint: pause / cancel in the Session
            raise Cancelled()

    with JOB_LOCK:
        job["state"] = "running"
        job["started"] = time.time()
        job["message"] = "Starting"
        _save_jobs(force=True)
        try:
            if kind == "tools":
                job["result"] = _run_tools_job(params, progress)
            else:
                # ASTROPHOTO_LOCAL_COPY: the copy folder is cleared, and the subs are read from a local copy
                # (made when a stage first reads them, deleted when the job ends)
                with s.local_copy(progress):
                    job["result"] = _run_session_job(kind, s, progress, params, stack_params, export_opts)
            job["state"] = "done"
            job["progress"] = 1.0
            job["message"] = "Finished"
        except Cancelled:
            job["state"] = "cancelled"
            job["message"] = "Cancelled"
        except Exception as e:
            if "cancelled" in str(e):
                job["state"] = "cancelled"
                job["message"] = "Cancelled"
            else:
                job["message"] = f"{type(e).__name__}: {e}"
                job["traceback"] = traceback.format_exc()
                job["device_state"] = ""
                _log_job_error(s, job, kind, stack_params)
                job["state"] = "error"          # last: a poller that sees "error" finds the traceback too
        finally:
            job["ended"] = time.time()
            if job.get("pause_started"):
                job["paused_total"] = job.get("paused_total", 0.0) + job["ended"] - job.pop("pause_started")
            if job.get("stage_key"):
                job["stages"].append([job["stage_key"], round(job["ended"] - job["stage_started"] - job.get("stage_paused", 0.0), 1)])
            job["active_seconds"] = round(_active_seconds(job), 1)
            if s is not None:
                _set_flag(s, "pause", False)
            CONTROL.pop(job_id, None)
            CANCELLED.discard(job_id)
            PREVIEW_CACHE.clear()
            _release_device_memory()
            _save_jobs(force=True)


def _log_job_error(s, job, kind, stack_params):
    """Append the failure (stage, settings, traceback) to the session's job_errors.log (the
    workdir's for a tools job), so it survives the browser and the server."""
    try:
        d = s.dir if s is not None else CONFIG["workdir"]
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "job_errors.log"), "a") as f:
            f.write(f"===== {time.strftime('%Y-%m-%d %H:%M:%S')}  job {job['id']} ({kind}) failed after "
                    f"{time.time() - job['started']:.1f}s\n")
            f.write(f"last stage: {job.get('stage')}\n")
            f.write(f"settings: {json.dumps(stack_params or job.get('params'), default=str)}\n")
            f.write(job["traceback"] + "\n")
    except Exception:
        pass


def _release_device_memory():
    """After a job: let Python hand its arrays back (the external tools run in their own processes)."""
    import gc
    gc.collect()


@app.post("/api/jobs")
def start_job(body: dict = Body(...)):
    """Queue a pipeline job; it runs when every job ahead of it has finished."""
    kind = body["kind"]
    if kind not in JOB_KINDS:
        raise HTTPException(400, "unknown job kind")
    _load_jobs()
    job_id = uuid.uuid4().hex[:10]
    if kind == "tools":
        tool = (body.get("params") or {}).get("tool")
        folder, dataset, n_subs = "", f"Tools: {tool}", None
    else:
        s = get_session(body["folder"])
        st = s.status()
        folder = body["folder"]
        dataset = (st.get("object") or os.path.basename(s.folders[0])) + (f" ({len(s.folders)} sessions)" if len(s.folders) > 1 else "")
        n_subs = st.get("n_files")
    JOBS[job_id] = {"id": job_id, "kind": kind, "folder": folder, "dataset": dataset,
                    "n_subs": n_subs, "state": "queued", "progress": 0.0, "message": "Waiting in the queue",
                    "log": [], "stages": [], "created": time.time(), "started": None, "result": None,
                    "params": body.get("params") or {}, "stack_params": body.get("stack_params") or {},
                    "export_opts": body.get("export") or {}}
    if kind == "all" and (body.get("stack_params") or {}).get("hypertune"):
        JOBS[job_id]["variant"] = "hypertune"          # its own label, and its own history for the time estimate (hours, not minutes)
    # set by /api/v1 (webui/api.py): who submitted it, which profile, the client's own reference
    for k in ("origin", "profile", "client_ref", "label"):
        if body.get(k):
            JOBS[job_id][k] = body[k]
    with QUEUE_LOCK:
        QUEUE.append(job_id)
        QUEUE_EVENT.set()
    _ensure_worker()
    _save_jobs(force=True)
    return JSONResponse(clean_json(_job_view(JOBS[job_id])))


def _external_activity() -> list[dict]:
    """Work on this machine outside the queue (the experiments of the old backend): none now."""
    return []


@app.get("/api/jobs")
def list_jobs(limit: int = 50):
    _load_jobs()
    order = {"running": 0, "paused": 0, "queued": 1}
    snap = list(JOBS.values())
    active = sorted([j for j in snap if j["state"] in ACTIVE],
                    key=lambda j: (order[j["state"]], QUEUE.index(j["id"]) if j["id"] in QUEUE else -1))
    done = sorted([j for j in snap if j["state"] not in ACTIVE],
                  key=lambda j: j.get("ended") or j.get("created") or 0, reverse=True)[:limit]
    return JSONResponse(clean_json({"active": [_job_view(j) for j in active], "history": [_job_view(j) for j in done],
                                    "external": _external_activity()}))


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    _load_jobs()
    if job_id not in JOBS:
        raise HTTPException(404)
    return JSONResponse(clean_json(_job_view(JOBS[job_id])))


def _job_session(j: dict):
    return get_session(j["folder"]) if j.get("folder") else None


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404)
    if j["state"] == "queued":
        with QUEUE_LOCK:
            if job_id in QUEUE:
                QUEUE.remove(job_id)
        j.update(state="cancelled", message="Removed from the queue", ended=time.time())
    elif j["state"] in ("running", "paused"):
        CANCELLED.add(job_id)
        s = _job_session(j)
        if s is not None:
            _set_flag(s, "cancel", True)
            _set_flag(s, "pause", False)
        j["message"] = "Cancelling at the next checkpoint"
    _save_jobs(force=True)
    return {"ok": True}


@app.post("/api/jobs/{job_id}/pause")
def pause_job(job_id: str):
    """Pause a running job: it stops at its next progress report and keeps its memory until it is
    resumed or cancelled."""
    j = JOBS.get(job_id)
    if not j or j["state"] != "running":
        raise HTTPException(409, "Only a running job can be paused")
    CONTROL.setdefault(job_id, threading.Event()).set()
    s = _job_session(j)
    if s is not None:
        _set_flag(s, "pause", True)
    j["state"] = "paused"
    j["pause_started"] = time.time()
    _save_jobs(force=True)
    return {"ok": True}


@app.post("/api/jobs/{job_id}/resume")
def resume_job(job_id: str):
    j = JOBS.get(job_id)
    if not j or j["state"] != "paused":
        raise HTTPException(409, "The job is not paused")
    now = time.time()
    dt = now - j.pop("pause_started", now)
    j["paused_total"] = j.get("paused_total", 0.0) + dt
    j["stage_paused"] = j.get("stage_paused", 0.0) + dt
    j["state"] = "running"
    if job_id in CONTROL:
        CONTROL[job_id].clear()
    s = _job_session(j)
    if s is not None:
        _set_flag(s, "pause", False)
    _save_jobs(force=True)
    return {"ok": True}


@app.post("/api/jobs/{job_id}/rerun")
def rerun_job(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404)
    return start_job({"kind": j["kind"], "folder": j["folder"], "params": j.get("params"),
                      "stack_params": j.get("stack_params"), "export": j.get("export_opts"),
                      "origin": j.get("origin"), "profile": j.get("profile"), "label": j.get("label")})


@app.delete("/api/jobs")
def clear_history():
    for k in [k for k, j in list(JOBS.items()) if j["state"] not in ACTIVE]:
        del JOBS[k]
    _save_jobs(force=True)
    return {"ok": True}


# ------------------------------------------------------------------ tools & licensing

def _exclusive(fn):
    """Run ``fn`` while no job runs (RC-Astro's licence calls must never overlap processing)."""
    if not JOB_LOCK.acquire(blocking=False):
        raise HTTPException(409, "A job is running - licence changes are possible when it finishes")
    try:
        return fn()
    finally:
        JOB_LOCK.release()


def _tool_error(e: Exception):
    if isinstance(e, HTTPException):
        raise e
    raise HTTPException(502, f"{type(e).__name__}: {e}")


@app.get("/api/tools")
def tools_get():
    return JSONResponse(tools_status())


@app.post("/api/tools/rcastro/license")
def rcastro_license(body: dict = Body(...)):
    """Register this computer with an 8-letter activation code (rc-astro license CODE --json)."""
    code = "".join(ch for ch in str(body.get("code") or "") if ch.isalnum()).upper()
    if len(code) != 8:
        raise HTTPException(400, "give the 8-letter activation code (XXXX-XXXX) from your RC-Astro account page")
    try:
        report = _exclusive(lambda: _call(_tool_member("rcastro", "license_code"), code, host=f"AstroPhotoStudio-{__version__}"))
    except Exception as e:
        _tool_error(e)
    return JSONResponse(clean_json({"report": report, "tools": tools_status().get("rcastro")}))


def _products(body: dict) -> str | list:
    p = body.get("product", body.get("products", "all"))
    if isinstance(p, str):
        p = p.strip().lower() or "all"
        if p != "all" and not all(x in ("bxt", "nxt", "sxt") for x in p.split(",")):
            raise HTTPException(400, "product: bxt | nxt | sxt | all")
        return p
    if isinstance(p, list) and p and all(x in ("bxt", "nxt", "sxt") for x in p):
        return p
    raise HTTPException(400, "product: bxt | nxt | sxt | all")


@app.post("/api/tools/rcastro/activate")
def rcastro_activate(body: dict = Body(default={})):
    """Activate a product (or all) of the account on this computer (rc-astro license --activate)."""
    product = _products(body)
    try:
        report = _exclusive(lambda: _call(_tool_member("rcastro", "activate"), product, host=f"AstroPhotoStudio-{__version__}"))
    except Exception as e:
        _tool_error(e)
    return JSONResponse(clean_json({"report": report, "tools": tools_status().get("rcastro")}))


@app.post("/api/tools/rcastro/deactivate")
def rcastro_deactivate(body: dict = Body(...)):
    """Free this computer's activation of a product (rc-astro license --deactivate)."""
    if not (body.get("product") or body.get("products")):
        raise HTTPException(400, "product: bxt | nxt | sxt | all")
    product = _products(body)
    try:
        report = _exclusive(lambda: _call(_tool_member("rcastro", "deactivate"), product, host=f"AstroPhotoStudio-{__version__}"))
    except Exception as e:
        _tool_error(e)
    return JSONResponse(clean_json({"report": report, "tools": tools_status().get("rcastro")}))


@app.post("/api/tools/gaia/clear")
def tools_gaia_clear():
    """Remove the cached chunks of the Gaia DR3 SPCC catalogue (they are fetched again when a field in that part of the sky is calibrated)."""
    if any(j.get("state") in ("running",) for j in JOBS.values()):
        raise HTTPException(409, "a job is running: clear the cache when the queue is idle")
    return {**tools_mod.gaia.clear(), "status": clean_json(tools_mod.gaia.status())}


@app.post("/api/tools/models/download")
def tools_download(body: dict = Body(...)):
    """Queue a model / tool download (job kind ``tools``): GraXpert's models, RC-Astro's
    (download-models), Cosmic Clarity's (the siril-scripts script's own download, ~2.4 GB), or StarNet2
    (needs ``accept_licence: true``: the download comes from starnetastro.com under their licence)."""
    tool = body.get("tool")
    if tool not in ("graxpert", "rcastro", "starnet", "cosmic_clarity"):
        raise HTTPException(400, "tool: graxpert | rcastro | starnet | cosmic_clarity")
    if tool == "starnet" and not body.get("accept_licence"):
        raise HTTPException(400, "StarNet is downloaded from starnetastro.com under their licence: send accept_licence: true")
    opts = {k: body[k] for k in ("accept_licence", "models", "product") if k in body}
    return start_job({"kind": "tools", "folder": "", "params": {"tool": tool, **opts}, "origin": "ui",
                      "label": f"Download {tool} models" if tool != "starnet" else "Download StarNet"})


# ------------------------------------------------------------------ previews

def _state_stamp(s: Session) -> list:
    """mtimes of the files a preview renders from: the cache key changes when a stage rewrites them."""
    return [os.path.getmtime(p) if os.path.exists(p) else 0
            for p in (os.path.join(s.dir, n) for n in ("stack.fits", "linear.fits", "starless.fits", "stars.fits"))]


def _raw_stack_jpeg(s: Session, side: int) -> bytes:
    """The "before" view: the stack as it came out of Siril, only display-stretched."""
    from astropy.io import fits
    f = os.path.join(s.dir, "stack.fits")
    if not os.path.exists(f):
        raise RuntimeError("Register & integrate the dataset first")
    return _autostretch_jpeg(fits.getdata(f), side)


PREVIEW_SEQ: dict[str, int] = {}             # session dir -> number of the newest preview request
PREVIEW_SEQ_LOCK = threading.Lock()


@app.post("/api/preview")
def preview(body: dict = Body(...)):
    s = get_session(body["folder"])
    params = body.get("params") or {}
    which = body.get("which", "after")
    size = body.get("size", 1400)
    size = 100000 if size in (0, None, "full") else int(size)
    key = json.dumps([s.dir, params, which, size, _state_stamp(s)], sort_keys=True)
    if key in PREVIEW_CACHE:
        return _preview_response(PREVIEW_CACHE[key], 0.0)
    if JOB_LOCK.locked():
        raise HTTPException(409, "A pipeline job is running - preview available when it finishes")
    if which != "before":
        with PREVIEW_SEQ_LOCK:                   # the newest request per dataset wins: an older one still waiting for the render lock is skipped
            n = PREVIEW_SEQ[s.dir] = PREVIEW_SEQ.get(s.dir, 0) + 1
    try:
        t0 = time.time()
        if which == "before":
            data = _raw_stack_jpeg(s, size)
        else:
            data = s.preview_jpeg(params, max_side=size, superseded=lambda: PREVIEW_SEQ.get(s.dir) != n)
    except PreviewSuperseded:
        return Response(status_code=204, headers={"X-Superseded": "1"})
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        raise HTTPException(400, str(e))
    if len(PREVIEW_CACHE) > 24:
        PREVIEW_CACHE.clear()
    PREVIEW_CACHE[key] = data
    return _preview_response(data, time.time() - t0)


def _preview_response(data: bytes, seconds: float) -> Response:
    from PIL import Image
    w, h = Image.open(io.BytesIO(data)).size
    return Response(data, media_type="image/jpeg",
                    headers={"X-Render-Time": f"{seconds:.2f}", "X-Width": str(w), "X-Height": str(h)})


@app.get("/api/autofinish")
def autofinish_result(folder: str):
    """The last Auto-finish of a dataset: its settings and report (null before the first)."""
    s = get_session(folder)
    fn = getattr(s, "autofinish_result", None)
    res = fn() if fn else _read_json(os.path.join(s.dir, "autofinish.json"))
    return JSONResponse(clean_json({"result": res}))


@app.get("/api/linear_info")
def linear_info(folder: str):
    """What the linear stage did (restore_meta.json: each tool, its parameters and seconds)."""
    s = get_session(folder)
    return clean_json({"restore": _read_json(os.path.join(s.dir, "restore_meta.json")),
                       "starnet": _read_json(os.path.join(s.dir, "starnet_meta.json"))})


def _stage_logs(s: Session) -> list[dict]:
    d = os.path.join(s.dir, "process", "logs")
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.log")), key=os.path.getmtime):
        out.append({"name": os.path.basename(f), "bytes": os.path.getsize(f), "modified": os.path.getmtime(f)})
    return out


def _map_png(path: str, kind: str, max_side: int = 1400) -> bytes:
    """A map FITS (coverage: frames per pixel; rejection: rejected fraction) as a PNG with a colour ramp and north up."""
    from astropy.io import fits as _fits
    a = np.nan_to_num(_fits.getdata(path).astype(np.float32))
    if a.ndim == 3:
        a = a.max(0)
    a = a[::-1]                                            # Siril bottom-up rows -> display
    s = max(1.0, max(a.shape) / max_side)
    if s > 1:
        a = cv2.resize(a, (int(a.shape[1] / s), int(a.shape[0] / s)), interpolation=cv2.INTER_AREA)
    hi = float(np.percentile(a, 99.5)) or 1.0
    u8 = (np.clip(a / hi, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(u8, cv2.COLORMAP_VIRIDIS if kind == "coverage" else cv2.COLORMAP_INFERNO)
    return cv2.imencode(".png", img)[1].tobytes()


@app.get("/api/diagnostic")
def diagnostic(folder: str, kind: str = "logs", name: str | None = None, tail: int = 400):
    """``kind=logs``: the Siril / tool stage logs of the session (list); ``kind=log&name=``: one
    log's last lines; ``kind=reference``: the reference thumbnail when the session wrote one."""
    s = get_session(folder)
    if kind == "logs":
        return clean_json({"logs": _stage_logs(s)})
    if kind == "log":
        f = os.path.join(s.dir, "process", "logs", os.path.basename(name or ""))
        if not name or not os.path.isfile(f):
            raise HTTPException(404)
        with open(f, errors="replace") as fh:
            return PlainTextResponse("".join(fh.readlines()[-max(tail, 1):]))
    if kind in ("coverage", "rejection"):
        f = os.path.join(s.dir, "coverage.fits" if kind == "coverage" else "rejection_map.fits")
        if not os.path.exists(f):
            raise HTTPException(404)
        return Response(_map_png(f, kind), media_type="image/png")
    if kind == "reference":
        f = os.path.join(s.dir, "reference.jpg")
        if not os.path.exists(f):
            raise HTTPException(404)
        return FileResponse(f, media_type="image/jpeg")
    raise HTTPException(404)


@app.get("/api/exports")
def exports(folder: str):
    s = get_session(folder)
    d = os.path.join(s.dir, "exports")
    items = []
    for f in sorted(glob.glob(os.path.join(d, "*.jpg")), reverse=True):
        base = f[:-4]
        items.append({"jpg": f, "tif": base + ".tif" if os.path.exists(base + ".tif") else None,
                      "name": os.path.basename(f), "size_mb": round(os.path.getsize(f) / 2**20, 1),
                      "created": time.ctime(os.path.getmtime(f))})
    return {"exports": items}


@app.get("/api/download")
def download(path: str, inline: bool = False):
    path = os.path.abspath(path)
    if not path.startswith(os.path.abspath(CONFIG["workdir"])) or not os.path.isfile(path):
        raise HTTPException(403)
    return FileResponse(path, filename=None if inline else os.path.basename(path))


# ------------------------------------------------------------------ removed features

SERVER_STARTED = time.time()


def _lab(*parts) -> str:
    d = os.path.join(CONFIG["workdir"], "lab", *parts)
    os.makedirs(d, exist_ok=True)
    return d


def _alive(pid) -> bool:
    """Whether process ``pid`` is running.  On Windows os.kill(pid, 0) would send it
    CTRL_C_EVENT (signal 0), so the process's exit code is queried instead."""
    try:
        pid = int(pid)
    except Exception:
        return False
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)            # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259   # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _read_status(d: str) -> dict:
    import json as _json
    p = os.path.join(d, "status.json")
    try:
        st = _json.load(open(p))
    except Exception:
        return {"state": "new"}
    if st.get("state") in ("starting", "preparing", "running"):
        if st.get("in_process"):
            # a hyper-tune study runs inside the server: one started before this server came up was interrupted with it
            if float(st.get("started") or 0) < SERVER_STARTED:
                st["state"] = "died"
                st["message"] = (st.get("message") or "") + " (the server restarted)"
        elif not _alive(st.get("pid")):
            st["state"] = "died"
            st["message"] = (st.get("message") or "") + " (the process is gone)"
    return st


def _spawn(module: str, d: str):
    import subprocess
    out = open(os.path.join(d, "stdout.txt"), "a")
    # detached from the server: its own session (POSIX) or process group (Windows, so that
    # "Stop" can send it Ctrl-Break without touching the server)
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    return subprocess.Popen([sys.executable, "-W", "ignore", "-m", module, d], cwd=ROOT, stdout=out,
                            stderr=subprocess.STDOUT, **kw)


def _safe_name(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("_")[:60] or "untitled"


@app.get("/api/lab/tasks")
def lab_tasks():
    from sirilpipe.lab.synthetic import DEFAULT_SPEC
    from sirilpipe.lab.tasks import TASKS
    return clean_json({"tasks": [t.describe() for t in TASKS.values()], "synthetic_defaults": DEFAULT_SPEC,
                       "samplers": [{"name": "tpe", "label": "TPE (Bayesian, multivariate)"},
                                    {"name": "es", "label": "(1+1)-CMA-ES, constrained (local search from your settings; one objective)"},
                                    {"name": "nsga2", "label": "NSGA-II (multi-objective)"},
                                    {"name": "random", "label": "Random search"},
                                    {"name": "grid", "label": "Grid (needs steps)"}]})


@app.get("/api/lab/datasets")
def lab_datasets():
    import json as _json
    real = []
    for d in datasets()["datasets"]:
        # every session of the target, analysed or not: the drop-down keeps the ones a study can
        # read (analysed/stacked), the "nights of one target" box lists them all - what counts for
        # a combination is the cache of the combination itself, which the form checks separately
        real.append({"kind": "real", "folder": d["path"], "name": d.get("object") or d["name"],
                     "object": d.get("object"), "date": d.get("date"), "filter": d.get("filter"),
                     "exptime": d.get("exptime"), "total_min": d.get("total_min"),
                     "stacked": d["cached"]["stacked"], "analysed": d["cached"].get("analysed", d["cached"]["stacked"]),
                     "n_fits": d["n_fits"]})
    syn = []
    root = _lab("synthetic")
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        spec = {}
        try:
            spec = _json.load(open(os.path.join(d, "spec.json")))
        except Exception:
            pass
        syn.append({"kind": "synthetic", "dir": d, "name": name, "spec": spec, "status": _read_status(d)})
    return clean_json({"real": real, "synthetic": syn, "sep": os.pathsep})


@app.get("/api/lab/dataset_state")
def lab_dataset_state(folder: str):
    """What is cached for a dataset selection (several nights joined with os.pathsep included):
    the Experiments tab shows whether the combined session still needs Analyse / Register."""
    d = os.path.join(CONFIG["workdir"], dataset_slug(split_folders(folder)))
    return {"analysed": os.path.exists(os.path.join(d, "frames.json")),
            "stacked": os.path.exists(os.path.join(d, "stack.fits")),
            "denoised": restoration_done(d)}


@app.post("/api/lab/synthetic")
def lab_synthetic(body: dict = Body(...)):
    import json as _json
    from sirilpipe.lab.synthetic import DEFAULT_SPEC
    name = _safe_name(body.get("name") or "synthetic")
    d = os.path.join(_lab("synthetic"), name)
    if os.path.exists(d):
        raise HTTPException(409, f"A synthetic dataset called {name} already exists")
    os.makedirs(d)
    spec = {k: body.get("spec", {}).get(k, v) for k, v in DEFAULT_SPEC.items()}
    spec["name"] = name
    spec["device"] = body.get("device", "auto")
    _json.dump(spec, open(os.path.join(d, "spec.json"), "w"), indent=1)
    _spawn("sirilpipe.lab.generate", d)
    return {"name": name, "dir": d}


@app.delete("/api/lab/synthetic/{name}")
def lab_synthetic_delete(name: str):
    d = os.path.join(_lab("synthetic"), _safe_name(name))
    if not os.path.isdir(d):
        raise HTTPException(404)
    st = _read_status(d)
    if st.get("state") == "running":
        raise HTTPException(409, "Still generating")
    shutil.rmtree(d)
    return {"ok": True}


@app.post("/api/lab/studies")
def lab_create(body: dict = Body(...)):
    import json as _json
    from sirilpipe.lab.tasks import TASKS
    cfg = body["config"]
    if cfg.get("task") not in TASKS:
        raise HTTPException(400, "unknown task")
    if not cfg.get("objectives"):
        raise HTTPException(400, "choose an objective")
    ds = cfg.get("dataset") or {}
    if ds.get("kind") == "real":
        # a real dataset is one session or several nights of one target joined with os.pathsep
        # (the same form the Process tab's "stack together with" selection sends)
        given = split_folders(ds.get("folder") or "")
        if not given:
            raise HTTPException(400, "No dataset folder given")
        missing = [f for f in given if not os.path.isdir(f)]
        if missing:
            raise HTTPException(400, f"Folder not found: {', '.join(missing)}")
        ds["folder"] = os.pathsep.join(given)
        ds["name"] = ds.get("name") or os.path.basename(os.path.normpath(given[0]))
    sid = time.strftime("%Y%m%d-%H%M%S") + "_" + _safe_name(cfg.get("name") or cfg["task"])
    d = os.path.join(_lab("studies"), sid)
    os.makedirs(d)
    cfg["name"] = cfg.get("name") or sid
    cfg["workdir"] = CONFIG["workdir"]
    cfg["created"] = time.time()
    _json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=1)
    _spawn("sirilpipe.lab.run", d)
    return {"id": sid}


def _study_dir(sid: str) -> str:
    d = os.path.join(_lab("studies"), _safe_name(sid))
    if not os.path.isdir(d):
        raise HTTPException(404)
    return d


def _load_trials(d: str, cfg: dict):
    try:
        import optuna
    except ImportError:                                   # an image built without optuna: the studies are listed without their trials
        return None, []
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    db = os.path.join(d, "study.db")
    if not os.path.exists(db):
        return None, []
    try:
        study = optuna.load_study(study_name=cfg["name"], storage=f"sqlite:///{db}")
    except Exception:
        return None, []
    rows = []
    for t in study.trials:
        cons = t.user_attrs.get("constraints")
        rows.append({"number": t.number, "state": t.state.name, "params": t.params,
                     "values": t.values, "metrics": t.user_attrs.get("metrics"), "error": t.user_attrs.get("error"),
                     "feasible": None if cons is None else all(v <= 0 for v in cons), "duplicate_of": t.user_attrs.get("duplicate_of"),
                     "baseline": bool(t.user_attrs.get("baseline") or t.system_attrs.get("fixed_params")),
                     "seconds": (t.datetime_complete - t.datetime_start).total_seconds()
                     if t.datetime_complete and t.datetime_start else None,
                     "image": os.path.exists(os.path.join(d, "trials", f"{t.number}.jpg"))})
    return study, rows


@app.get("/api/lab/studies")
def lab_list():
    import json as _json
    out = []
    root = _lab("studies")
    for sid in sorted(os.listdir(root), reverse=True):
        d = os.path.join(root, sid)
        try:
            cfg = _json.load(open(os.path.join(d, "config.json")))
        except Exception:
            continue
        st = _read_status(d)
        study, rows = _load_trials(d, cfg)
        done = [r for r in rows if r["state"] == "COMPLETE"]
        best = None
        if done and len(cfg["objectives"]) == 1 and study is not None:
            try:
                best = _best_trial(study, cfg)[0].values[0]
            except Exception:
                best = None
        out.append({"id": sid, "name": cfg["name"], "task": cfg["task"], "dataset": cfg["dataset"].get("name"),
                    "state": st.get("state"), "message": st.get("message"), "n_trials": cfg.get("n_trials"),
                    "n_complete": len(done), "n_total": len(rows), "best": best, "hypertune": bool(cfg.get("hypertune")),
                    "objectives": cfg["objectives"], "created": cfg.get("created")})
    return clean_json({"studies": out})


def _best_trial(study, cfg):
    """The best trial as the study's objectives define it (sirilpipe.lab.run.best_trial): among the trials that meet the task's
    constraints, the optimum of a single objective (the baseline kept when the gain is below the task's resolution); with
    several objectives, the Pareto-optimal trial that is best on the first (primary)."""
    from sirilpipe.lab.run import best_trial
    from sirilpipe.lab.tasks import TASKS
    return best_trial(study, cfg, TASKS.get(cfg.get("task")))


@app.get("/api/lab/best")
def lab_best():
    """Every study with a completed trial and the pipeline settings of its best trial."""
    import json as _json
    from sirilpipe.lab.tasks import TASKS
    out = []
    root = _lab("studies")
    for sid in sorted(os.listdir(root), reverse=True):
        d = os.path.join(root, sid)
        try:
            cfg = _json.load(open(os.path.join(d, "config.json")))
        except Exception:
            continue
        study, rows = _load_trials(d, cfg)
        if study is None or not any(r["state"] == "COMPLETE" for r in rows):
            continue
        try:
            t, why = _best_trial(study, cfg)
        except Exception:
            continue
        task = TASKS.get(cfg["task"])
        if task is None:                                         # a study of a task the Siril backend does not have (imagemm, network, ...): its settings cannot be applied
            continue
        allp = t.user_attrs.get("params_all") or {**task.defaults(), **t.params}
        settings, processing, other = {}, {}, {}
        for p in task.params:
            if p["name"] in allp:
                if p.get("pipeline"):
                    settings[p["pipeline"]] = allp[p["name"]]
                elif p.get("processing"):
                    processing[p["processing"]] = allp[p["name"]]
                else:
                    other[p["name"]] = allp[p["name"]]
        out.append({"id": sid, "name": cfg["name"], "task": cfg["task"], "task_label": task.label,
                    "dataset": cfg["dataset"].get("name"), "state": _read_status(d).get("state"),
                    "trial": t.number, "values": t.values, "objectives": cfg["objectives"], "criterion": why,
                    "settings": settings, "processing": processing, "not_pipeline": other})
    return clean_json({"studies": out})


@app.get("/api/lab/studies/{sid}")
def lab_detail(sid: str, log_lines: int = 200):
    import json as _json
    from sirilpipe.lab.tasks import TASKS
    d = _study_dir(sid)
    cfg = _json.load(open(os.path.join(d, "config.json")))
    st = _read_status(d)
    study, rows = _load_trials(d, cfg)
    best, importance = [], {}
    if study is not None and any(r["state"] == "COMPLETE" for r in rows):
        try:
            if len(cfg["objectives"]) == 1:
                best = [_best_trial(study, cfg)[0].number]
            else:
                best = [t.number for t in study.best_trials]
        except Exception:
            best = []
        tuned = [k for k, v in cfg.get("space", {}).items() if v.get("tune")]
        n_done = sum(r["state"] == "COMPLETE" for r in rows)
        if tuned and n_done >= 4:
            import optuna
            for i, o in enumerate(cfg["objectives"]):
                try:
                    importance[o["metric"]] = optuna.importance.get_param_importances(
                        study, target=(lambda t, i=i: t.values[i]))
                except Exception as e:
                    importance[o["metric"]] = {"_error": str(e)}
    log = []
    lp = os.path.join(d, "log.txt")
    if os.path.exists(lp):
        with open(lp, errors="replace") as f:
            log = f.readlines()[-log_lines:]
    task = TASKS.get(cfg["task"])
    params = task.params if task else []
    metrics = task.metrics if task else {o["metric"]: {"label": o["metric"], "direction": o.get("direction", "minimize")} for o in cfg.get("objectives", [])}
    pipeline_map = {p["name"]: p.get("pipeline") for p in params}
    processing_map = {p["name"]: p.get("processing") for p in params}
    code_map = {p["name"]: p.get("code") for p in params}
    best_info = None
    if best:
        try:
            best_info = _best_trial(study, cfg)[1]
        except Exception:
            best_info = None
    return clean_json({"id": sid, "config": cfg, "status": st, "trials": rows, "best": best, "best_criterion": best_info,
                       "objective_note": getattr(task, "objective_note", "") if task else "",
                       "importance": importance, "log": "".join(log), "pipeline_map": pipeline_map,
                       "processing_map": processing_map, "code_map": code_map, "metrics": metrics,
                       "legacy_task": task is None})


@app.get("/api/lab/studies/{sid}/trial/{n}.jpg")
def lab_trial_image(sid: str, n: int):
    f = os.path.join(_study_dir(sid), "trials", f"{int(n)}.jpg")
    if not os.path.exists(f):
        raise HTTPException(404)
    return FileResponse(f, media_type="image/jpeg")


@app.post("/api/lab/studies/{sid}/stop")
def lab_stop(sid: str):
    import signal
    st = _read_status(_study_dir(sid))
    if st.get("state") not in ("starting", "preparing", "running"):
        raise HTTPException(409, "The study is not running")
    if st.get("in_process"):
        raise HTTPException(409, "This study is part of a hyper-tuned Run everything job: cancel the job (Jobs tab or the Progress card) to stop it")
    # graceful: the runner cancels the running trial and marks the study stopped
    os.kill(int(st["pid"]), signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
    return {"ok": True}


@app.post("/api/lab/studies/{sid}/continue")
def lab_continue(sid: str, body: dict = Body(...)):
    import json as _json
    d = _study_dir(sid)
    st = _read_status(d)
    if st.get("state") in ("starting", "preparing", "running"):
        raise HTTPException(409, "The study is already running")
    cfg = _json.load(open(os.path.join(d, "config.json")))
    _, rows = _load_trials(d, cfg)
    done = sum(1 for r in rows if r["state"] in ("COMPLETE", "FAIL", "PRUNED"))
    cfg["n_trials"] = done + int(body.get("extra", 10))
    if body.get("device"):
        cfg["device"] = body["device"]
    _json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=1)
    _spawn("sirilpipe.lab.run", d)
    return {"ok": True, "n_trials": cfg["n_trials"]}


@app.delete("/api/lab/studies/{sid}")
def lab_delete(sid: str):
    d = _study_dir(sid)
    if _read_status(d).get("state") in ("starting", "preparing", "running"):
        raise HTTPException(409, "Stop the study first")
    shutil.rmtree(d)
    return {"ok": True}




# ------------------------------------------------------------------ explore (plate solve + Gaia / SIMBAD annotation)

EXPLORE: dict[str, dict] = {}       # session dir -> {"state": running|done|error, "message", "progress"}


def _solve_thread(s: Session, gmax: float):
    from astrophoto import astrometry
    st = EXPLORE[s.dir]

    def progress(i, n, msg):
        st["message"] = msg
        st["progress"] = i / max(n, 1)
    try:
        astrometry.solve_session(s, gmax=gmax, progress=progress)
        st.update(state="done", message="Solved", progress=1.0)
    except Exception as e:
        st.update(state="error", message=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())


@app.post("/api/explore/solve")
def explore_solve(body: dict = Body(...)):
    s = get_session(body["folder"])
    if not _ui_status(s)["stacked"]:
        raise HTTPException(400, "Stack the dataset first")
    cur = EXPLORE.get(s.dir)
    if cur and cur["state"] == "running":
        return cur
    EXPLORE[s.dir] = {"state": "running", "message": "Starting", "progress": 0.0}
    threading.Thread(target=_solve_thread, args=(s, float(body.get("gmax", 16.0))), daemon=True).start()
    return EXPLORE[s.dir]


@app.get("/api/explore/status")
def explore_status(folder: str):
    from astrophoto import astrometry
    s = get_session(folder)
    st = dict(EXPLORE.get(s.dir) or {"state": "idle"})
    sol = astrometry.load_solution(s) if _ui_status(s)["stacked"] else None
    st["solved"] = sol is not None
    if sol is not None:
        st["solution"] = {k: sol.get(k) for k in ("n_matched", "rms_arcsec", "scale_arcsec", "depth_g", "solved", "center")}
    return clean_json(st)


@app.post("/api/explore/data")
def explore_data(body: dict = Body(...)):
    from astrophoto import astrometry
    s = get_session(body["folder"])
    sol = astrometry.load_solution(s) if _ui_status(s)["stacked"] else None
    if sol is None:
        raise HTTPException(400, "not solved")
    try:
        h, w = (int(v) for v in sol["shape"])
        e = finish_mod.EDGE_CROP     # the render trims this rim from the stack (post._Layers): the view is that crop
        info = {"crop": [e, h - e, e, w - e]} if min(h, w) > 8 * e else {}
        return JSONResponse(clean_json(astrometry.annotate(s, info, (h, w))))
    except RuntimeError as e:
        raise HTTPException(400, str(e))


# the programmatic API for other programs: /api/v1.  It is given this module
# itself: run as `python -m webui.server`, an `import webui.server` would load a second copy
# with its own (empty) job table.
from webui import api as _api  # noqa: E402

_api.mount(app, sys.modules[__name__])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--images", default=CONFIG["images"],
                    help="root folder that contains the session folders of subs (Seestar *_sub, DWARF DWARF_RAW_*, ...)")
    ap.add_argument("--workdir", default=CONFIG["workdir"])
    ap.add_argument("--calib", help="extra calibration library folder(s), e.g. a copy of a DWARF's CALI_FRAME")
    ap.add_argument("--no-calibration", action="store_true", help="ignore bias / dark / flat masters")
    a = ap.parse_args()
    if a.no_calibration:
        os.environ["ASTROPHOTO_CALIB"] = "off"
    elif a.calib:
        os.environ["ASTROPHOTO_CALIB"] = os.path.abspath(a.calib)
    CONFIG["images"] = os.path.abspath(a.images)
    CONFIG["workdir"] = os.path.abspath(a.workdir)
    _load_jobs()
    import uvicorn
    print(f"AstroPhoto Studio {__version__} web UI -> http://{a.host}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
