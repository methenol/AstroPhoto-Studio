"""AstroPhoto Studio web UI (FastAPI).

    python -m webui.server            # http://127.0.0.1:8000
    python -m webui.server --host 0.0.0.0 --port 8080 --images /path/to/telescope/exports
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import os
import shutil
import sys
import threading
import time
import traceback
import uuid
import warnings

import cv2
import numpy as np
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from astrophoto import __version__  # noqa: E402
from astrophoto.pipeline import (DEFAULTS, STACK_DEFAULTS, Cancelled, Session, clean_json,  # noqa: E402
                                 dataset_slug, restoration_done, slugify, split_folders)

CONFIG = {"images": os.path.join(ROOT, "images"), "workdir": os.path.join(ROOT, "output")}

app = FastAPI(title="AstroPhoto Studio", version=__version__)
app.add_middleware(GZipMiddleware, minimum_size=4096)
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

SESSIONS: dict[str, Session] = {}
JOBS: dict[str, dict] = {}   # every job: queued, running, paused and finished (saved in <workdir>/jobs.json)
JOB_LOCK = threading.Lock()   # held while a job runs (one heavy job at a time: memory)
PREVIEW_CACHE: dict[str, bytes] = {}

# UI metadata for every processing parameter
PARAM_SPEC = [
    {"group": "Linear", "key": "crop", "label": "Auto-crop stacking edges", "type": "bool"},
    {"group": "Linear", "key": "crop_threshold", "label": "Crop: min. frame coverage", "type": "range", "min": 0.1, "max": 1.0, "step": 0.05},
    {"group": "Linear", "key": "background", "label": "Gradient removal", "type": "bool"},
    {"group": "Linear", "key": "bg_method", "label": "Gradient model (auto: sky survey reference when plate-solved, NSNS DR0.2)", "type": "select", "options": ["auto", "reference", "poly", "rbf"]},
    {"group": "Linear", "key": "bg_correction", "label": "Gradient correction", "type": "select", "options": ["subtract", "divide"]},
    {"group": "Linear", "key": "bg_degree", "label": "Polynomial degree", "type": "range", "min": 0, "max": 4, "step": 1},
    {"group": "Linear", "key": "bg_smoothing", "label": "RBF smoothing", "type": "range", "min": 0, "max": 1, "step": 0.05},
    {"group": "Linear", "key": "white_balance", "label": "Colour calibration (auto: spectrophotometric, Gaia)", "type": "select", "options": ["auto", "stars", "background", "none"]},
    {"group": "Linear", "key": "spcc_sensor", "label": "Colour calibration: camera sensor", "type": "select", "options": ["auto"]},
    {"group": "Linear", "key": "spcc_filter", "label": "Colour calibration: filter", "type": "select", "options": ["auto"]},
    {"group": "Linear", "key": "spcc_white_ref", "label": "Colour calibration: white reference", "type": "select", "options": ["average_spiral_galaxy", "g2v"]},
    {"group": "Linear", "key": "denoise", "label": "AI denoise (Noise2Noise)", "type": "range", "min": 0, "max": 1, "step": 0.05},
    {"group": "Linear", "key": "deconvolution", "label": "Deconvolution (AI network / ImageMM, or Richardson-Lucy)", "type": "range", "min": 0, "max": 1, "step": 0.05},
    {"group": "Linear", "key": "restored_resolution", "label": "ImageMM: restored image resolution (× Eq. 11 σ)", "type": "range", "min": 1, "max": 3, "step": 0.05},
    {"group": "Stretch", "key": "stretch_method", "label": "Stretch algorithm", "type": "select", "options": ["ghs", "asinh", "mtf", "log"]},
    {"group": "Stretch", "key": "stretch", "label": "Stretch (background level)", "type": "range", "min": 0.04, "max": 0.35, "step": 0.01},
    {"group": "Stretch", "key": "auto_stretch", "label": "Adapt stretch to target size", "type": "bool"},
    {"group": "Stretch", "key": "hdr", "label": "HDR (protect bright cores)", "type": "range", "min": 0, "max": 1.5, "step": 0.05},
    {"group": "Stretch", "key": "contrast", "label": "GHS focus (contrast)", "type": "range", "min": -1, "max": 10, "step": 0.25},
    {"group": "Stretch", "key": "color_preservation", "label": "Colour-preserving stretch", "type": "range", "min": 0, "max": 1, "step": 0.05},
    {"group": "Colour", "key": "palette", "label": "Palette", "type": "select", "options": ["auto", "natural", "hoo", "foraxx", "hoo_warm"]},
    {"group": "Colour", "key": "oiii_boost", "label": "OIII boost (dual-band)", "type": "range", "min": 0.5, "max": 3, "step": 0.05},
    {"group": "Colour", "key": "synthetic_luminance", "label": "Synthetic luminance (dual-band)", "type": "range", "min": 0, "max": 1, "step": 0.05},
    {"group": "Colour", "key": "oiii_unmix", "label": "Remove Ha leakage from OIII", "type": "bool"},
    {"group": "Colour", "key": "saturation", "label": "Saturation", "type": "range", "min": 0.5, "max": 3, "step": 0.05},
    {"group": "Colour", "key": "chroma_denoise", "label": "Colour noise reduction", "type": "range", "min": 0, "max": 1, "step": 0.05},
    {"group": "Colour", "key": "scnr", "label": "SCNR (remove green cast)", "type": "range", "min": 0, "max": 1, "step": 0.05},
    {"group": "Stars", "key": "star_separation", "label": "Process stars separately", "type": "bool"},
    {"group": "Stars", "key": "star_removal", "label": "Star removal (auto: AI once trained)", "type": "select", "options": ["auto", "ai", "classic"]},
    {"group": "Stars", "key": "star_reduction", "label": "Star reduction (1 = starless)", "type": "range", "min": 0, "max": 1, "step": 0.05},
    {"group": "Stars", "key": "star_intensity", "label": "Star brightness", "type": "range", "min": 0, "max": 1.5, "step": 0.05},
    {"group": "Stars", "key": "star_saturation", "label": "Star colour", "type": "range", "min": 0, "max": 2.5, "step": 0.05},
    {"group": "Stars", "key": "star_color_preservation", "label": "Star colour intensity (stretch)", "type": "range", "min": 0, "max": 1, "step": 0.05},
    {"group": "Stars", "key": "halo_suppress", "label": "Halo suppression", "type": "range", "min": 0, "max": 1, "step": 0.05},
    {"group": "Detail", "key": "luminance_denoise", "label": "Fine-grain noise reduction", "type": "range", "min": 0, "max": 1.5, "step": 0.05},
    {"group": "Detail", "key": "local_contrast", "label": "Local contrast (structures)", "type": "range", "min": 0, "max": 2, "step": 0.05},
    {"group": "Detail", "key": "sharpen", "label": "Final sharpening", "type": "range", "min": 0, "max": 1.5, "step": 0.05},
    {"group": "Finish", "key": "black_point", "label": "Black point", "type": "range", "min": 0, "max": 0.2, "step": 0.005},
    {"group": "Finish", "key": "brightness", "label": "Midtones", "type": "range", "min": -1, "max": 1, "step": 0.05},
    {"group": "Grade", "key": "grade_amount", "label": "Grade strength", "type": "range", "min": 0, "max": 1, "step": 0.05},
    {"group": "Grade", "key": "grade_temperature", "label": "Temperature (cool – warm)", "type": "range", "min": -1, "max": 1, "step": 0.05},
    {"group": "Grade", "key": "grade_tint", "label": "Tint (green – magenta)", "type": "range", "min": -1, "max": 1, "step": 0.05},
    {"group": "Grade", "key": "grade_warm_hue", "label": "Warm hues: shift (magenta – yellow)", "type": "range", "min": -1, "max": 1, "step": 0.05},
    {"group": "Grade", "key": "grade_warm_sat", "label": "Warm hues: saturation", "type": "range", "min": 0, "max": 2, "step": 0.05},
    {"group": "Grade", "key": "grade_cool_hue", "label": "Cool hues: shift (green – blue)", "type": "range", "min": -1, "max": 1, "step": 0.05},
    {"group": "Grade", "key": "grade_cool_sat", "label": "Cool hues: saturation", "type": "range", "min": 0, "max": 2, "step": 0.05},
    {"group": "Grade", "key": "grade_contrast", "label": "Object contrast (S-curve)", "type": "range", "min": -1, "max": 1, "step": 0.05},
]

PRESETS = {
    "Balanced": {},
    "Vivid nebula": {"saturation": 2.0, "local_contrast": 0.8, "stretch": 0.18, "contrast": 3.0, "oiii_boost": 1.3,
                     "star_reduction": 0.5, "star_intensity": 0.8},
    "Starless-ish": {"star_reduction": 0.8, "star_intensity": 0.45, "local_contrast": 0.7},
    "Galaxy / broadband": {"palette": "natural", "stretch": 0.12, "contrast": 4.0, "saturation": 1.4,
                           "local_contrast": 0.4, "star_reduction": 0.2, "bg_degree": 2},
    "Natural colour": {"palette": "natural", "saturation": 1.3, "scnr": 0.6},
    "Deep & dark": {"stretch": 0.1, "black_point": 0.04, "contrast": 4.0, "local_contrast": 0.6},
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


# ------------------------------------------------------------------ pages

@app.get("/", response_class=HTMLResponse)
def index():
    return open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()


@app.get("/api/system")
def system():
    try:
        from astrophoto.denoise import device_info
        dev = device_info()
    except Exception as e:  # torch missing
        dev = {"default": "cpu", "gpus": [], "error": str(e)}
    return {"version": __version__, "devices": dev, "defaults": DEFAULTS, "stack_defaults": STACK_DEFAULTS,
            "param_spec": _param_spec(), "presets": PRESETS, "images_root": CONFIG["images"]}


def _param_spec() -> list[dict]:
    """PARAM_SPEC with the sensors and filters of the SPCC database (cached index; offline: auto only)."""
    from astrophoto import spcc
    cache = os.path.join(CONFIG["workdir"], "spcc_db")
    lists = {"spcc_sensor": spcc.list_curves("osc_sensors", cache), "spcc_filter": spcc.list_curves("osc_filters", cache)}
    return [{**p, "options": ["auto"] + [n for n in lists[p["key"]] if n != "README"]} if p["key"] in lists else p
            for p in PARAM_SPEC]


@app.get("/api/datasets")
def datasets(root: str | None = None):
    root = os.path.abspath(os.path.expanduser(root or CONFIG["images"]))
    out = []
    if not os.path.isdir(root):
        return {"root": root, "datasets": []}
    candidates = [root] + sorted(d for d in glob.glob(os.path.join(root, "**"), recursive=True)
                                 if os.path.isdir(d) and os.path.abspath(d) != root)
    for d in candidates:
        # calibration libraries are not datasets (DWARF CALI_FRAME / DWARF_DARK, darks/ flats/ biases/)
        parts = {p.lower() for p in os.path.relpath(d, root).replace("\\", "/").split("/")}
        if parts & {"cali_frame", "dwarf_dark", "dark", "darks", "flat", "flats", "bias", "biases", "offsets",
                    "calib", "restacked", "solving_failed"}:
            continue
        fits_files = [f for ext in ("*.fit", "*.fits", "*.fts") for f in glob.glob(os.path.join(d, ext))]
        if not fits_files:
            continue
        info = {"path": d, "name": os.path.relpath(d, root) if d != root else os.path.basename(d),
                "n_fits": len(fits_files)}
        try:
            from astrophoto.frames import read_info
            fi = read_info(sorted(fits_files)[0])
            info.update({"object": fi.object, "filter": fi.filter, "exptime": fi.exptime,
                         "instrument": fi.telescope or fi.instrument, "date": (fi.date_obs or "")[:10] or None})
            info["total_min"] = round(info["exptime"] * len(fits_files) / 60, 1)
        except Exception:
            pass
        cache_dir = os.path.join(CONFIG["workdir"], slugify(d))
        info["cached"] = {"analysed": os.path.exists(os.path.join(cache_dir, "analysis.pkl")),
                          "stacked": os.path.exists(os.path.join(cache_dir, "stack.fits")),
                          "denoised": restoration_done(cache_dir)}
        out.append(info)
    return {"root": root, "datasets": out, "sep": os.pathsep}


@app.post("/api/open")
def open_dataset(body: dict = Body(...)):
    s = get_session(body["folder"])
    if not s.infos:
        s.scan()
    return clean_json({"status": s.status(), "frames": s.frames_table(), "medians": (s.analysis or {}).get("medians")})


@app.get("/api/frames")
def frames(folder: str):
    s = get_session(folder)
    return clean_json({"frames": s.frames_table(), "medians": (s.analysis or {}).get("medians"),
                       "sensitivity": (s.analysis or {}).get("sensitivity", 1.0)})


@app.post("/api/frames/override")
def override(body: dict = Body(...)):
    s = get_session(body["folder"])
    s.set_override(body["name"], body.get("accepted"))
    return {"frames": s.frames_table()}


@app.post("/api/frames/sensitivity")
def sensitivity(body: dict = Body(...)):
    s = get_session(body["folder"])
    return {"frames": s.reselect(float(body["sensitivity"]))}


@app.get("/api/calibration")
def calibration_get(folder: str):
    """The calibration panel: telescope profile, masters chosen and available, fitted dark scale,
    flat check, notes (what ``python -m astrophoto calibration`` prints)."""
    s = get_session(folder)
    return clean_json({**s.calibration_info(), "status": s.status()})


@app.post("/api/calibration")
def calibration_set(body: dict = Body(...)):
    """Save this dataset's calibration settings (enabled, library folder, dark / flat / bias:
    auto | none | a master's path).  Run the Calibrate step (or Analyse) to apply them."""
    s = get_session(body["folder"])
    prefs = s.set_calib_prefs(body.get("prefs") or {})
    return {"prefs": prefs}


@app.get("/api/thumb")
def thumb(folder: str, name: str, size: int = 360):
    s = get_session(folder)
    if not s.infos:
        s.scan()
    return Response(s.frame_thumbnail(name, size), media_type="image/jpeg")


# ------------------------------------------------------------------ jobs

JOB_KINDS = {"calibrate": "Calibrate", "analyse": "Analyse frames", "stack": "Register & integrate", "denoise": "Restore", "starnet": "Train star remover",
             "autofinish": "Auto-finish", "all": "Run everything & export", "export": "Export"}
ACTIVE = ("running", "paused", "queued")
QUEUE: list[str] = []                     # queued job ids, first to run first
QUEUE_LOCK = threading.Lock()
QUEUE_EVENT = threading.Event()
_WORKER: dict = {}
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
    same = [h for h in list(JOBS.values()) if h is not j and h["state"] == "done" and h["kind"] == j["kind"]
            and h.get("folder") == j.get("folder") and h.get("active_seconds")]
    if same:
        h = max(same, key=lambda q: q.get("ended", 0))
        return h["active_seconds"], "the last run on this dataset"
    other = [h for h in list(JOBS.values()) if h is not j and h["state"] == "done" and h["kind"] == j["kind"]
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
    v["label"] = JOB_KINDS.get(j["kind"], j["kind"])
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


def _run_job(job_id: str):
    job = JOBS[job_id]
    kind, folder = job["kind"], job["folder"]
    params, stack_params, export_opts = job.get("params") or {}, job.get("stack_params") or {}, job.get("export_opts") or {}
    s = get_session(folder)
    s.cancel_flag.clear()
    s.pause_flag.clear()

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
        if s.checkpoint():                       # waits here while paused
            raise Cancelled()

    with JOB_LOCK:
        job["state"] = "running"
        job["started"] = time.time()
        job["message"] = "Starting"
        _save_jobs(force=True)
        try:
            # the stages that read the subs do it from a local copy when ASTROPHOTO_LOCAL_COPY is on
            sp_ = {**STACK_DEFAULTS, **stack_params}
            reads_subs = kind in ("calibrate", "analyse", "stack", "all") or (
                kind == "denoise" and sp_.get("deconv_method") == "imagemm")
            with (s.local_copy(progress) if reads_subs else contextlib.nullcontext()):
                if kind == "calibrate":
                    job["result"] = s.run_calibration(progress)
                elif kind == "analyse":
                    s.run_analysis(float(stack_params.get("sensitivity", 1.0)), progress)
                    job["result"] = {"n": len(s.infos)}
                elif kind == "stack":
                    job["result"] = s.run_stack(stack_params, progress)
                    s.plate_solve(progress)              # catalogue stars for the colour calibration
                elif kind == "denoise":
                    s.run_denoise(stack_params, progress)
                    job["result"] = {"ok": True}
                elif kind == "starnet":
                    job["result"] = s.train_star_remover(params, stack_params, progress)
                elif kind == "autofinish":
                    job["result"] = {"autofinish": s.autofinish(params, progress)}
                elif kind == "all":
                    sp = {**STACK_DEFAULTS, **stack_params}
                    s.run_analysis(sp["sensitivity"], progress)
                    s.run_stack(sp, progress)
                    s.plate_solve(progress)
                    s.run_denoise(sp, progress)
                    if sp["star_remover"]:
                        s.train_star_remover(params, sp, progress)
                    fin = None
                    if sp["autofinish"]:                 # the tuned settings are the ones exported
                        fin = s.autofinish(params, progress)
                        params = fin["params"]
                    job["result"] = s.export(params, progress=progress, **export_opts)
                    if fin:
                        job["result"]["autofinish"] = fin
                elif kind == "export":
                    job["result"] = s.export(params, progress=progress, **export_opts)
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
                job["state"] = "error"
                job["message"] = f"{type(e).__name__}: {e}"
                job["traceback"] = traceback.format_exc()
                job["device_state"] = _device_state()
                _log_job_error(s, job, kind, stack_params)
        finally:
            job["ended"] = time.time()
            if job.get("pause_started"):
                job["paused_total"] = job.get("paused_total", 0.0) + job["ended"] - job.pop("pause_started")
            if job.get("stage_key"):
                job["stages"].append([job["stage_key"], round(job["ended"] - job["stage_started"] - job.get("stage_paused", 0.0), 1)])
            job["active_seconds"] = round(_active_seconds(job), 1)
            s.pause_flag.clear()
            PREVIEW_CACHE.clear()
            _release_device_memory()
            _save_jobs(force=True)


def _device_state() -> str:
    """Memory of the CUDA devices at the moment of a failure (empty without CUDA)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return ""
        out = []
        for d in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(d)
            out.append(f"cuda:{d} {torch.cuda.get_device_name(d)}: {free / 2**30:.2f} of {total / 2**30:.2f} GiB free; "
                       f"this process: {torch.cuda.memory_allocated(d) / 2**30:.2f} GiB allocated, "
                       f"{torch.cuda.memory_reserved(d) / 2**30:.2f} GiB reserved, peak "
                       f"{torch.cuda.max_memory_allocated(d) / 2**30:.2f} GiB")
        return "\n".join(out)
    except Exception as e:
        return f"(device state unavailable: {e})"


def _log_job_error(s, job, kind, stack_params):
    """Append the failure (stage, settings, traceback, device memory) to the session's
    job_errors.log, so it survives the browser and the server."""
    import json as _json
    try:
        with open(s._p("job_errors.log"), "a") as f:
            f.write(f"===== {time.strftime('%Y-%m-%d %H:%M:%S')}  job {job['id']} ({kind}) failed after "
                    f"{time.time() - job['started']:.1f}s\n")
            f.write(f"last stage: {job.get('stage')}\n")
            f.write(f"settings: {_json.dumps(stack_params, default=str)}\n")
            if job.get("device_state"):
                f.write(f"devices:\n{job['device_state']}\n")
            f.write(job["traceback"] + "\n")
    except Exception:
        pass


def _release_device_memory():
    """Return the memory PyTorch's caching allocator still holds to the driver after a job
    (it keeps freed blocks reserved otherwise, including after a failure)."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


@app.post("/api/jobs")
def start_job(body: dict = Body(...)):
    """Queue a pipeline job; it runs when every job ahead of it has finished."""
    kind = body["kind"]
    if kind not in JOB_KINDS:
        raise HTTPException(400, "unknown job kind")
    _load_jobs()
    s = get_session(body["folder"])
    job_id = uuid.uuid4().hex[:10]
    st = s.status()
    JOBS[job_id] = {"id": job_id, "kind": kind, "folder": body["folder"], "dataset": (st.get("object") or os.path.basename(s.folders[0]))
                    + (f" ({len(s.folders)} sessions)" if len(s.folders) > 1 else ""),
                    "n_subs": st.get("n_files"), "state": "queued", "progress": 0.0, "message": "Waiting in the queue",
                    "log": [], "stages": [], "created": time.time(), "started": None, "result": None,
                    "params": body.get("params") or {}, "stack_params": body.get("stack_params") or {},
                    "export_opts": body.get("export") or {}}
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
    """Experiments studies and synthetic-data generation run as their own processes (sharing the
    GPU with the queue): the ones running now."""
    import json as _json
    out = []
    root = os.path.join(CONFIG["workdir"], "lab")
    for sub, kind in (("studies", "Experiment study"), ("synthetic", "Synthetic dataset")):
        d0 = os.path.join(root, sub)
        if not os.path.isdir(d0):
            continue
        for name in sorted(os.listdir(d0)):
            d = os.path.join(d0, name)
            st = _read_status(d)
            if st.get("state") not in ("starting", "preparing", "running"):
                continue
            item = {"kind": kind, "id": name, "state": st.get("state"), "message": st.get("message"),
                    "started": st.get("started")}
            try:
                cfg = _json.load(open(os.path.join(d, "config.json")))
                item["name"] = cfg.get("name")
                item["device"] = cfg.get("device")
                item["n_trials"] = cfg.get("n_trials")
                item["trial"] = st.get("trial")
            except Exception:
                item["name"] = name
            out.append(item)
    return out


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
        s = get_session(j["folder"])
        s.cancel_flag.set()
        s.pause_flag.clear()
        j["message"] = "Cancelling at the next checkpoint"
    _save_jobs(force=True)
    return {"ok": True}


@app.post("/api/jobs/{job_id}/pause")
def pause_job(job_id: str):
    """Pause a running job: it stops at its next checkpoint (a frame, a sub, an ImageMM iteration)
    and keeps its memory (and GPU memory) until it is resumed or cancelled."""
    j = JOBS.get(job_id)
    if not j or j["state"] != "running":
        raise HTTPException(409, "Only a running job can be paused")
    get_session(j["folder"]).pause_flag.set()
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
    get_session(j["folder"]).pause_flag.clear()
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


# ------------------------------------------------------------------ previews

def _jpeg(img: np.ndarray, q: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", (np.clip(img, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8),
                           [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()


@app.post("/api/preview")
def preview(body: dict = Body(...)):
    s = get_session(body["folder"])
    params = body.get("params") or {}
    which = body.get("which", "after")
    size = body.get("size", 1400)
    size = None if size in (0, None, "full") else int(size)
    import json as _json
    key = _json.dumps([s.dir, params, which, size, s.meta.get("created")], sort_keys=True)
    if key in PREVIEW_CACHE:
        return Response(PREVIEW_CACHE[key], media_type="image/jpeg")
    if JOB_LOCK.locked():
        raise HTTPException(409, "A pipeline job is running – preview available when it finishes")
    try:
        t0 = time.time()
        if which == "before":
            img = s.render_before(params, size or 100000)
            info = {}
        else:
            img, info = s.render(params, max_size=size)
        data = _jpeg(img, 92)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    if len(PREVIEW_CACHE) > 24:
        PREVIEW_CACHE.clear()
    PREVIEW_CACHE[key] = data
    return Response(data, media_type="image/jpeg",
                    headers={"X-Render-Time": f"{time.time() - t0:.2f}", "X-Width": str(img.shape[1]),
                             "X-Height": str(img.shape[0])})


@app.get("/api/autofinish")
def autofinish_result(folder: str):
    """The last Auto-finish of a dataset: its settings and report (null before the first)."""
    return JSONResponse(clean_json({"result": get_session(folder).autofinish_result()}))


@app.get("/api/linear_info")
def linear_info(folder: str):
    s = get_session(folder)
    if s._lin_cache:
        return clean_json(s._lin_cache[2])
    return {}


@app.get("/api/diagnostic")
def diagnostic(folder: str, kind: str):
    s = get_session(folder)
    p = {"coverage": "coverage.fits", "rejection": "rejection_map.fits"}.get(kind)
    if kind == "reference":
        f = os.path.join(s.dir, "reference.jpg")
        if not os.path.exists(f):
            raise HTTPException(404)
        return FileResponse(f, media_type="image/jpeg")
    if not p or not os.path.exists(os.path.join(s.dir, p)):
        raise HTTPException(404)
    from astrophoto.pipeline import _load_fits
    m = _load_fits(os.path.join(s.dir, p))
    m = m / max(np.percentile(m, 99.9), 1e-6)
    m = cv2.resize(m, None, fx=min(1, 900 / max(m.shape)), fy=min(1, 900 / max(m.shape)), interpolation=cv2.INTER_AREA)
    col = cv2.applyColorMap((np.clip(m, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    ok, buf = cv2.imencode(".jpg", col)
    return Response(buf.tobytes(), media_type="image/jpeg")


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


# ------------------------------------------------------------------ explore (plate solving + catalogues)

EXPLORE: dict[str, dict] = {}    # dataset dir -> solve state


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
    if not s.status()["stacked"]:
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
    sol = astrometry.load_solution(s) if s.status()["stacked"] else None
    st["solved"] = sol is not None
    if sol is not None:
        st["solution"] = {k: sol[k] for k in ("n_matched", "rms_arcsec", "scale_arcsec", "depth_g", "solved", "center")}
    return clean_json(st)


@app.post("/api/explore/data")
def explore_data(body: dict = Body(...)):
    from astrophoto import astrometry
    s = get_session(body["folder"])
    if JOB_LOCK.locked() and s._lin_cache is None:
        raise HTTPException(409, "A pipeline job is running – the sky map is available when it finishes")
    lin, info = s.linear(body.get("params") or {})
    try:
        return JSONResponse(clean_json(astrometry.annotate(s, info, lin.shape)))
    except RuntimeError as e:
        raise HTTPException(400, str(e))


# ------------------------------------------------------------------ experiment lab (Optuna studies)

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
    if st.get("state") in ("starting", "preparing", "running") and not _alive(st.get("pid")):
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
    from astrophoto.lab.synthetic import DEFAULT_SPEC
    from astrophoto.lab.tasks import TASKS
    return clean_json({"tasks": [t.describe() for t in TASKS.values()], "synthetic_defaults": DEFAULT_SPEC,
                       "samplers": [{"name": "tpe", "label": "TPE (Bayesian, multivariate)"},
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
                     "stacked": d["cached"]["stacked"], "analysed": d["cached"]["analysed"],
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
    return {"analysed": os.path.exists(os.path.join(d, "analysis.pkl")),
            "stacked": os.path.exists(os.path.join(d, "stack.fits")),
            "denoised": restoration_done(d)}


@app.post("/api/lab/synthetic")
def lab_synthetic(body: dict = Body(...)):
    import json as _json
    from astrophoto.lab.synthetic import DEFAULT_SPEC
    name = _safe_name(body.get("name") or "synthetic")
    d = os.path.join(_lab("synthetic"), name)
    if os.path.exists(d):
        raise HTTPException(409, f"A synthetic dataset called {name} already exists")
    os.makedirs(d)
    spec = {k: body.get("spec", {}).get(k, v) for k, v in DEFAULT_SPEC.items()}
    spec["name"] = name
    spec["device"] = body.get("device", "auto")
    _json.dump(spec, open(os.path.join(d, "spec.json"), "w"), indent=1)
    _spawn("astrophoto.lab.generate", d)
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
    from astrophoto.lab.tasks import TASKS
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
    _spawn("astrophoto.lab.run", d)
    return {"id": sid}


def _study_dir(sid: str) -> str:
    d = os.path.join(_lab("studies"), _safe_name(sid))
    if not os.path.isdir(d):
        raise HTTPException(404)
    return d


def _load_trials(d: str, cfg: dict):
    import optuna
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
        rows.append({"number": t.number, "state": t.state.name, "params": t.params,
                     "values": t.values, "metrics": t.user_attrs.get("metrics"), "error": t.user_attrs.get("error"),
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
        if done and len(cfg["objectives"]) == 1:
            sign = 1 if (cfg["objectives"][0].get("direction") or "minimize") == "minimize" else -1
            best = min((r["values"][0] for r in done), key=lambda v: sign * v)
        out.append({"id": sid, "name": cfg["name"], "task": cfg["task"], "dataset": cfg["dataset"].get("name"),
                    "state": st.get("state"), "message": st.get("message"), "n_trials": cfg.get("n_trials"),
                    "n_complete": len(done), "n_total": len(rows), "best": best,
                    "objectives": cfg["objectives"], "created": cfg.get("created")})
    return clean_json({"studies": out})


def _best_trial(study, cfg):
    """The best trial as the study's objectives define it: the optimum of a single objective;
    with several objectives, the Pareto-optimal trial that is best on the first (primary)."""
    objs = cfg["objectives"]
    if len(objs) == 1:
        return study.best_trial, f"best {objs[0]['metric']} ({objs[0].get('direction', 'minimize')})"
    front = study.best_trials
    sign = 1 if objs[0].get("direction", "minimize") == "minimize" else -1
    t = min(front, key=lambda q: sign * q.values[0])
    return t, (f"Pareto-optimal on {' × '.join(o['metric'] for o in objs)}, best {objs[0]['metric']} "
               f"of the {len(front)} on the front")


@app.get("/api/lab/best")
def lab_best():
    """Every study with a completed trial and the pipeline settings of its best trial."""
    import json as _json
    from astrophoto.lab.tasks import TASKS
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
        task = TASKS[cfg["task"]]
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
        if cfg["task"] in ("imagemm", "network"):
            settings["deconv_method"] = cfg["task"]
        out.append({"id": sid, "name": cfg["name"], "task": cfg["task"], "task_label": task.label,
                    "dataset": cfg["dataset"].get("name"), "state": _read_status(d).get("state"),
                    "trial": t.number, "values": t.values, "objectives": cfg["objectives"], "criterion": why,
                    "settings": settings, "processing": processing, "not_pipeline": other})
    return clean_json({"studies": out})


@app.get("/api/lab/studies/{sid}")
def lab_detail(sid: str, log_lines: int = 200):
    import json as _json
    from astrophoto.lab.tasks import TASKS
    d = _study_dir(sid)
    cfg = _json.load(open(os.path.join(d, "config.json")))
    st = _read_status(d)
    study, rows = _load_trials(d, cfg)
    best, importance = [], {}
    if study is not None and any(r["state"] == "COMPLETE" for r in rows):
        try:
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
    task = TASKS[cfg["task"]]
    pipeline_map = {p["name"]: p.get("pipeline") for p in task.params}
    processing_map = {p["name"]: p.get("processing") for p in task.params}
    code_map = {p["name"]: p.get("code") for p in task.params}
    return clean_json({"id": sid, "config": cfg, "status": st, "trials": rows, "best": best,
                       "importance": importance, "log": "".join(log), "pipeline_map": pipeline_map,
                       "processing_map": processing_map, "code_map": code_map, "metrics": task.metrics})


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
    _spawn("astrophoto.lab.run", d)
    return {"ok": True, "n_trials": cfg["n_trials"]}


@app.delete("/api/lab/studies/{sid}")
def lab_delete(sid: str):
    d = _study_dir(sid)
    if _read_status(d).get("state") in ("starting", "preparing", "running"):
        raise HTTPException(409, "Stop the study first")
    shutil.rmtree(d)
    return {"ok": True}


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
