"""Programmatic API for other programs (automation, scripts, remote clients), mounted at /api/v1
by webui.server.

Jobs submitted here go into the same queue as the web UI's (one heavy job at a time) and show
up in its Jobs panel.  No authentication: it is meant for a private network (Tailscale).

    GET    /api/v1/health                   cheap reachability check (no PyTorch import)
    GET    /api/v1                          version, devices, profiles, presets, job kinds, roots
    GET    /api/v1/profiles                 named restoration recipes (astrophoto.pipeline.PROFILES)
    POST   /api/v1/datasets/resolve         does the server see these session folders? (subs, target, filter)
    GET    /api/v1/uploads?prefix=          uploaded files and sizes (to resume an upload)
    PUT    /api/v1/uploads/{path}           upload one file (raw body), for clients whose data the
                                            server cannot see (a field session on a laptop or phone)
    POST   /api/v1/jobs                     queue a job (idempotent per client_ref)
    GET    /api/v1/jobs?state=&client_ref=  jobs, newest first
    GET    /api/v1/jobs/{id}                one job: state, stage, progress, ETAs, outputs
    POST   /api/v1/jobs/{id}/cancel         cancel (queued: removed; running: at the next checkpoint)
    GET    /api/v1/jobs/{id}/files/{name}   download an output (jpg / tif / json)
    GET    /api/v1/queue                    running + queued jobs in order, with positions and ETAs
    POST   /api/v1/queue/move               move a queued job to another position

Paths are relative to the server's images folder (``--images``) or, with ``"source": "uploads"``,
to its uploads folder (``<workdir>/uploads``, or ``ASTROPHOTO_UPLOADS``).  Absolute paths are
taken as they are.
"""
from __future__ import annotations

import glob
import os
import time

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

API_VERSION = 1
FITS_EXT = ("*.fit", "*.fits", "*.fts", "*.FIT", "*.FITS", "*.FTS")

router = APIRouter(prefix="/api/v1", tags=["api"])
srv = None        # the webui.server module (set by mount)


def mount(app, server_module):
    global srv
    srv = server_module
    app.include_router(router)


# ------------------------------------------------------------------ helpers

def _uploads_root() -> str:
    return os.path.abspath(os.environ.get("ASTROPHOTO_UPLOADS") or os.path.join(srv.CONFIG["workdir"], "uploads"))


def _root(source: str | None) -> str:
    if source in (None, "", "images"):
        return srv.CONFIG["images"]
    if source == "uploads":
        return _uploads_root()
    raise HTTPException(400, f"unknown source {source!r} (images | uploads)")


def _resolve(path: str, source: str | None) -> str:
    if not path or not str(path).strip():
        raise HTTPException(400, "empty path")
    p = os.path.expanduser(str(path).strip())
    if os.path.isabs(p):
        return os.path.abspath(p)
    root = _root(source)
    full = os.path.abspath(os.path.join(root, p))
    if os.path.commonpath([full, root]) != root:
        raise HTTPException(400, f"path escapes the {source or 'images'} folder: {path}")
    return full


def _fits_in(d: str) -> list[str]:
    return sorted({f for ext in FITS_EXT for f in glob.glob(os.path.join(d, ext))})


def _light_folders(d: str) -> list[str]:
    """The folders holding a session's subs: the folder itself (a DWARF ``DWARF_RAW_…`` folder, a
    Seestar ``*_sub`` folder), or its ``lights/`` folder and any ``lights/rejected/`` beside them."""
    if _fits_in(d):
        return [d]
    from astrophoto.calibration import _child
    lights = _child(d, "lights")
    if lights and _fits_in(lights):
        out = [lights]
        rej = os.path.join(lights, "rejected")
        if os.path.isdir(rej) and _fits_in(rej):
            out.append(rej)
        return out
    return [d]


def _describe(folders: list[str]) -> dict:
    """What the server sees in a dataset's folders (the light frames are counted, not read)."""
    out = {"folders": folders, "exists": all(os.path.isdir(f) for f in folders), "n_fits": 0}
    files = [f for d in folders if os.path.isdir(d) for f in _fits_in(d)]
    subs = [f for f in files if not os.path.basename(f).lower().startswith(("stacked", "master"))]
    out["n_fits"] = len(subs)
    out["bytes"] = sum(os.path.getsize(f) for f in subs)
    if subs:
        try:
            from astrophoto.frames import read_info
            fi = read_info(subs[len(subs) // 2])
            out.update({"object": fi.object, "filter": fi.filter, "exptime": fi.exptime,
                        "instrument": fi.telescope or fi.instrument, "date": (fi.date_obs or "")[:10] or None,
                        "total_min": round(fi.exptime * len(subs) / 60, 1)})
        except Exception as e:
            out["read_error"] = f"{type(e).__name__}: {e}"
    return out


def _dataset(body: dict) -> list[str]:
    """The light folders of a request: ``path`` (one session) or ``paths`` (several nights of one target)."""
    paths = body.get("paths") or ([body["path"]] if body.get("path") else [])
    if not paths:
        raise HTTPException(400, "give path (one session folder) or paths (several)")
    folders = []
    for p in paths:
        folders += _light_folders(_resolve(p, body.get("source")))
    return sorted(set(folders))


def _outputs(j: dict) -> list[dict]:
    res = j.get("result") if isinstance(j.get("result"), dict) else {}
    out = []
    for kind in ("jpg", "tif", "json"):
        f = res.get(kind)
        if isinstance(f, str) and os.path.isfile(f):
            out.append({"kind": kind, "name": os.path.basename(f), "bytes": os.path.getsize(f),
                        "url": f"/api/v1/jobs/{j['id']}/files/{os.path.basename(f)}"})
    return out


def _view(j: dict, full: bool = False) -> dict:
    v = srv._job_view(j)
    v["kind_label"] = v["label"]                 # _job_view's label is the kind's name
    v["label"] = j.get("label") or v.get("dataset")
    v["folders"] = srv.split_folders(j["folder"])
    v["outputs"] = _outputs(j)
    if not full:
        for k in ("log", "stages", "traceback", "device_state"):
            v.pop(k, None)
    return srv.clean_json(v)


def _job(job_id: str) -> dict:
    srv._load_jobs()
    j = srv.JOBS.get(job_id)
    if not j:
        raise HTTPException(404, f"no job {job_id}")
    return j


def _profiles() -> dict:
    from astrophoto.pipeline import PROFILES, STACK_DEFAULTS
    return {name: {"label": p.get("label", name), "stack_params": p.get("stack_params", {}),
                   "params": p.get("params", {}),
                   "effective": {k: {**STACK_DEFAULTS, **p.get("stack_params", {})}[k]
                                 for k in ("deconv_method", "ai_deconvolution", "imagemm_n2n", "star_remover", "scale")}}
            for name, p in PROFILES.items()}


# ------------------------------------------------------------------ info

@router.get("/health")
def health():
    running = [j for j in list(srv.JOBS.values()) if j["state"] in ("running", "paused")]
    return {"ok": True, "api": API_VERSION, "version": srv.__version__, "busy": bool(running),
            "queued": len(srv.QUEUE), "time": time.time()}


@router.get("")
def info():
    from astrophoto.pipeline import STACK_DEFAULTS
    try:
        from astrophoto.denoise import device_info
        dev = device_info()
    except Exception as e:
        dev = {"default": "cpu", "gpus": [], "error": str(e)}
    srv._load_jobs()
    return srv.clean_json({
        "name": "AstroPhoto Studio", "api": API_VERSION, "version": srv.__version__, "devices": dev,
        "images_root": srv.CONFIG["images"], "uploads_root": _uploads_root(), "workdir": srv.CONFIG["workdir"],
        "job_kinds": srv.JOB_KINDS, "default_kind": "all", "default_profile": "default",
        "profiles": _profiles(), "presets": srv.PRESETS, "stack_defaults": STACK_DEFAULTS,
        "queue": {"running": sum(j["state"] in ("running", "paused") for j in srv.JOBS.values()),
                  "queued": len(srv.QUEUE)}})


@router.get("/profiles")
def profiles():
    return {"default": "default", "profiles": _profiles()}


# ------------------------------------------------------------------ datasets & uploads

@router.post("/datasets/resolve")
def resolve(body: dict = Body(...)):
    return srv.clean_json({"images_root": srv.CONFIG["images"], "uploads_root": _uploads_root(),
                           **_describe(_dataset(body))})


@router.get("/uploads")
def uploads(prefix: str = ""):
    root = _uploads_root()
    base = _resolve(prefix, "uploads") if prefix else root
    files = []
    if os.path.isdir(base):
        for d, _, names in os.walk(base):
            for n in names:
                if n.endswith(".part"):
                    continue
                f = os.path.join(d, n)
                files.append({"path": os.path.relpath(f, root).replace(os.sep, "/"), "bytes": os.path.getsize(f)})
    return {"root": root, "files": sorted(files, key=lambda x: x["path"])}


@router.put("/uploads/{path:path}")
async def upload(path: str, request: Request):
    """One file, streamed to ``<uploads>/<path>`` (written to .part, renamed when complete, so an
    interrupted upload never leaves a truncated sub behind)."""
    dest = _resolve(path, "uploads")
    if dest == _uploads_root():
        raise HTTPException(400, "give a file path")
    expected = request.headers.get("content-length")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    n = 0
    try:
        with open(tmp, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
                n += len(chunk)
        if expected is not None and int(expected) != n:
            raise HTTPException(400, f"incomplete upload: {n} of {expected} bytes")
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return {"path": os.path.relpath(dest, _uploads_root()).replace(os.sep, "/"), "bytes": n}


# ------------------------------------------------------------------ jobs

@router.post("/jobs")
def submit(body: dict = Body(...)):
    """Queue a job on a session (or several sessions of one target).

    body: path | paths, source (images | uploads), kind (default all = analyse, stack, restore,
    star remover, auto-finish, export; stack_params.autofinish = false exports the given settings
    as they are), profile (a name from /profiles), stack_params / params (override the
    profile and the defaults), preset (a processing preset of the web UI), export (quality,
    upscale, tiff), client_ref (the client's id for this job: submitting it again returns the job
    already queued or done instead of a second one), label (shown in the Jobs panel)."""
    from astrophoto.pipeline import PROFILES
    srv._load_jobs()
    ref = body.get("client_ref")
    if ref:
        prior = [j for j in srv.JOBS.values() if j.get("client_ref") == ref
                 and j["state"] not in ("cancelled", "error", "interrupted")]
        if prior:
            return JSONResponse(_view(max(prior, key=lambda j: j.get("created") or 0)) | {"duplicate": True})
    kind = body.get("kind") or "all"
    if kind not in srv.JOB_KINDS:
        raise HTTPException(400, f"unknown kind {kind!r}; one of {list(srv.JOB_KINDS)}")
    profile = body.get("profile") or "default"
    if profile not in PROFILES:
        raise HTTPException(400, f"unknown profile {profile!r}; one of {list(PROFILES)}")
    preset = body.get("preset")
    if preset and preset not in srv.PRESETS:
        raise HTTPException(400, f"unknown preset {preset!r}; one of {list(srv.PRESETS)}")
    folders = _dataset(body)
    seen = _describe(folders)
    if not seen["exists"]:
        raise HTTPException(404, f"folder not found on the server: {', '.join(f for f in folders if not os.path.isdir(f))}")
    if not seen["n_fits"]:
        raise HTTPException(422, f"no FITS subs in {', '.join(folders)}")
    prof = PROFILES[profile]
    job = srv.start_job({
        "kind": kind, "folder": os.pathsep.join(folders),
        "stack_params": {**prof.get("stack_params", {}), **(body.get("stack_params") or {})},
        "params": {**srv.PRESETS.get(preset or "", {}), **prof.get("params", {}), **(body.get("params") or {})},
        "export": body.get("export") or {},
        "origin": body.get("origin") or "api", "profile": profile, "client_ref": ref,
        "label": body.get("label")})
    import json as _json
    j = srv.JOBS[_json.loads(job.body)["id"]]
    # the session is not scanned before the job runs, so name it from its subs as the UI does
    if seen.get("object"):
        j["dataset"] = seen["object"] + (f" ({len(folders)} sessions)" if len(folders) > 1 else "")
        srv._save_jobs(force=True)
    return JSONResponse(_view(j) | {"duplicate": False, "dataset_info": srv.clean_json(seen)})


@router.get("/jobs")
def jobs(state: str | None = None, client_ref: str | None = None, origin: str | None = None, limit: int = 50):
    srv._load_jobs()
    snap = sorted(srv.JOBS.values(), key=lambda j: j.get("created") or 0, reverse=True)
    states = set(state.split(",")) if state else None
    refs = set(client_ref.split(",")) if client_ref else None
    out = [j for j in snap if (not states or j["state"] in states) and (not refs or j.get("client_ref") in refs)
           and (not origin or j.get("origin") == origin)]
    return {"jobs": [_view(j) for j in out[:limit]]}


@router.get("/jobs/{job_id}")
def job(job_id: str):
    return _view(_job(job_id), full=True)


@router.post("/jobs/{job_id}/cancel")
def cancel(job_id: str):
    _job(job_id)
    srv.cancel_job(job_id)
    return _view(srv.JOBS[job_id])


@router.get("/jobs/{job_id}/files/{name}")
def job_file(job_id: str, name: str, inline: bool = False):
    j = _job(job_id)
    for o in _outputs(j):
        if o["name"] == name:
            f = j["result"][o["kind"]]
            return FileResponse(f, filename=None if inline else name)
    raise HTTPException(404, f"job {job_id} has no output {name}")


@router.get("/queue")
def queue():
    srv._load_jobs()
    with srv.QUEUE_LOCK:
        order = list(srv.QUEUE)
    running = [j for j in srv.JOBS.values() if j["state"] in ("running", "paused")]
    queued = [srv.JOBS[i] for i in order if i in srv.JOBS and srv.JOBS[i]["state"] == "queued"]
    views = [_view(j) for j in running + queued]
    # the wait before each queued job starts, from the running job's and the earlier ones' estimates
    wait, known = 0.0, True
    for v in views:
        if v["state"] == "queued":
            v["starts_in"] = round(wait) if known else None
        est = v.get("total_eta") if v["state"] in ("running", "paused") else v.get("typical_total")
        if est is None:
            known = False
        else:
            wait += est
    return srv.clean_json({"running": views[:len(running)], "queued": views[len(running):],
                           "external": srv._external_activity(),
                           "drains_in": round(wait) if known and views else (0 if not views else None)})


@router.post("/queue/move")
def move(body: dict = Body(...)):
    """Move a queued job: position 1 = next to run."""
    job_id = body.get("id")
    pos = int(body.get("position", 1))
    with srv.QUEUE_LOCK:
        if job_id not in srv.QUEUE:
            raise HTTPException(409, f"job {job_id} is not waiting in the queue")
        srv.QUEUE.remove(job_id)
        srv.QUEUE.insert(max(0, min(pos - 1, len(srv.QUEUE))), job_id)
        order = list(srv.QUEUE)
    srv._save_jobs(force=True)
    return {"queue": order}
