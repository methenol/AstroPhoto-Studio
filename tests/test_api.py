"""Web server + /api/v1 contract tests (FastAPI TestClient, a fake Session and fake external tools).

The server is the compatibility surface for the telescope controller (api-contract.md): routes, job
view fields, states, output names and the idempotent ``client_ref`` behaviour are checked here
without Siril, GraXpert, StarNet or RC-Astro.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# --------------------------------------------------------------------------- fakes for modules not landed yet

def _install_stand_ins():
    """When a sirilpipe module the server imports does not exist (yet), register a minimal one so
    the server module can be tested on its own; the real ones are used as soon as they import."""
    import sirilpipe
    for name in ("session", "post", "tools"):
        try:
            __import__(f"sirilpipe.{name}")
        except Exception:
            mod = types.ModuleType(f"sirilpipe.{name}")
            mod.__stand_in__ = True
            sys.modules[f"sirilpipe.{name}"] = mod
            setattr(sirilpipe, name, mod)
    ses, post, tl = sys.modules["sirilpipe.session"], sys.modules["sirilpipe.post"], sys.modules["sirilpipe.tools"]
    if not hasattr(ses, "Cancelled"):
        ses.Cancelled = type("Cancelled", (Exception,), {})
    if not hasattr(ses, "Session"):
        ses.Session = object
    if not hasattr(post, "DEFAULTS"):
        post.DEFAULTS = {"stretch": 0.16, "saturation": 1.5}
    if not hasattr(post, "PARAM_SPEC"):
        post.PARAM_SPEC = [{"group": "Stretch", "key": "stretch", "label": "Stretch", "type": "range", "min": 0.04,
                            "max": 0.35, "step": 0.01},
                           {"group": "Colour", "key": "saturation", "label": "Saturation", "type": "range", "min": 0.5,
                            "max": 3, "step": 0.05}]
    if not hasattr(tl, "tools_status"):
        tl.tools_status = lambda: {}


_install_stand_ins()
from webui import server  # noqa: E402  (after the stand-ins)

Cancelled = server.Cancelled

# a licenseReport as `rc-astro license --json` prints it (tools.md C.5): BXT on trial, SXT / NXT not activated
LICENSE_REPORT = {
    "operation": "check", "computer": "user@astrophoto-studio", "account": {"email": "someone@example.com"},
    "activationUrl": "https://www.rc-astro.com/account",
    "licenseStatus": [
        {"product": "bxt", "name": "BlurXTerminator", "status": "trial", "message": "Trial: 27 days left", "daysLeft": 27, "mlVersion": 5},
        {"product": "nxt", "name": "NoiseXTerminator", "status": "available", "message": "Owned by your account; not activated on this computer",
         "command": "rc-astro license --activate nxt"},
        {"product": "sxt", "name": "StarXTerminator", "status": "notOwned", "message": "Not owned by this account"}],
}
RC_STATUS = {
    "ok": True, "version": "2.6.10", "schema": 6, "computer": "user@astrophoto-studio", "account": "someone@example.com",
    "activationUrl": "https://www.rc-astro.com/account",
    "products": [
        {"key": "bxt", "name": "BlurXTerminator", "status": "trial", "valid": True, "message": "Trial: 27 days left", "mlVersions": [5, 4, 2], "modelPresent": True},
        {"key": "nxt", "name": "NoiseXTerminator", "status": "available", "valid": False, "message": "Owned by your account; not activated on this computer", "mlVersions": [4], "modelPresent": False},
        {"key": "sxt", "name": "StarXTerminator", "status": "notOwned", "valid": False, "message": "Not owned by this account", "mlVersions": [], "modelPresent": False}],
    "attention": None,
}


class FakeRCAstro:
    calls: list = []

    def license_code(self, code):
        FakeRCAstro.calls.append(("license", code))
        return {**LICENSE_REPORT, "operation": "code"}

    def activate(self, product="all"):
        FakeRCAstro.calls.append(("activate", product))
        return {**LICENSE_REPORT, "operation": "activate"}

    def deactivate(self, product="all"):
        FakeRCAstro.calls.append(("deactivate", product))
        return {**LICENSE_REPORT, "operation": "deactivate"}

    def download_models(self, progress=None):
        FakeRCAstro.calls.append(("models",))
        if progress:
            progress(1, 2, "Downloading RC-Astro models")
        return {"downloaded": ["bxt"]}


class FakeGraXpert:
    def download_models(self, kinds=None, progress=None):
        if progress:
            progress(1, 4, "GraXpert model 1/4")
            progress(4, 4, "GraXpert model 4/4")
        return {"models": {"bge": True}}


class FakeStarNet:
    def download(self, accept_licence=False, progress=None):
        if not accept_licence:
            raise RuntimeError("licence not accepted")
        return {"path": "/tmp/starnet2"}


def fake_tools_module():
    m = types.SimpleNamespace(RCAstro=FakeRCAstro, GraXpert=FakeGraXpert, StarNet=FakeStarNet)
    m.tools_status = lambda: {
        "siril": {"ok": True, "version": "1.4.4", "path": "/usr/local/bin/siril-cli"},
        "graxpert": {"ok": True, "version": "3.2.0a3", "models": {"bge": True, "denoise": False, "deconv_obj": False, "deconv_stellar": False}, "gpu": False},
        "starnet": {"ok": False, "version": None, "path": None, "licence_accepted": False},
        "rcastro": RC_STATUS}
    return m


class FakeSession:
    """The Session interface of DESIGN section 2, doing no real work but writing the files the server looks at."""
    instances: list = []

    def __init__(self, folders, workdir):
        from sirilpipe.util import dataset_slug
        self.folders = sorted(folders)
        self.dir = os.path.join(workdir, dataset_slug(self.folders))
        os.makedirs(self.dir, exist_ok=True)
        self.cancel_requested = False
        self.pause_requested = False
        self.calls: list = []
        self.prefs = {"enabled": True}
        FakeSession.instances.append(self)

    def status(self):
        return {"object": "M 42", "n_files": None, "has_stack": os.path.exists(os.path.join(self.dir, "stack.fits")),
                "has_linear": False, "has_starless": False, "has_autofinish": False, "is_dualband": False,
                "instrument": "DWARF 3", "calibration": None}

    def checkpoint(self):
        if self.cancel_requested:
            raise Cancelled()

    def _steps(self, progress, name, n=3, delay=0.05):
        for i in range(n):
            progress(i, n, f"{name} {i + 1}/{n}")
            time.sleep(delay)

    def run_calibration(self, progress):
        self.calls.append("calibrate"); self._steps(progress, "Calibrating")
        return {"changed_analysis": False, "summary": "no masters"}

    def run_analysis(self, sensitivity, progress):
        self.calls.append(("analyse", sensitivity)); self._steps(progress, "Registering (pass 1/2)")
        with open(os.path.join(self.dir, "frames.json"), "w") as f:
            json.dump([], f)
        return {"n": 3}

    def run_stack(self, stack_params, progress):
        self.calls.append(("stack", dict(stack_params))); self._steps(progress, "Stacking")
        open(os.path.join(self.dir, "stack.fits"), "wb").write(b"x")
        return {"object": "M 42", "n_frames": 3}

    def plate_solve(self, progress):
        self.calls.append("solve"); progress(1, 1, "Plate solving")
        return {}

    def run_denoise(self, stack_params, progress, params=None):
        self.calls.append("restore"); self._steps(progress, "Deconvolution")
        return {}

    def train_star_remover(self, params, stack_params, progress):
        self.calls.append("stars"); self._steps(progress, "Star removal")
        return {"tool": "starnet"}

    def autofinish(self, params, progress):
        self.calls.append("autofinish"); progress(1, 1, "Auto-finish")
        return {"autofinish": {"params": {**params, "saturation": 1.7}, "report": {"decisions": []}}}

    def export(self, params, progress=None, quality=95, upscale=1.0, tiff=True):
        self.calls.append(("export", dict(params), quality, upscale, tiff))
        if progress:
            progress(1, 1, "Exporting")
        d = os.path.join(self.dir, "exports")
        os.makedirs(d, exist_ok=True)
        base = os.path.join(d, "M42_20261014-101500")
        out = {"size": [8, 8]}
        for ext in ("jpg", "tif", "json"):
            if ext == "tif" and not tiff:
                continue
            open(f"{base}.{ext}", "wb").write(b"data-" + ext.encode())
            out[ext] = f"{base}.{ext}"
        return out

    def frames(self):
        return {"frames": [{"name": "sub_0001.fits", "accepted": True, "overridden": False, "reasons": []}],
                "medians": None, "sensitivity": 1.0}

    def set_override(self, name, accepted):
        self.calls.append(("override", name, accepted))

    def calibration_info(self):
        return {"profile": {}, "n_lights": 3, "prefs": self.prefs, "report": None, "candidates": {"bias": [], "dark": [], "flat": []}}

    def set_calibration(self, prefs):
        self.prefs = prefs
        return prefs

    def preview_jpeg(self, params, max_side=1600):
        import cv2
        ok, buf = cv2.imencode(".jpg", np.full((6, 9, 3), 128, np.uint8))
        return buf.tobytes()


# --------------------------------------------------------------------------- fixtures

@pytest.fixture()
def env(tmp_path, monkeypatch):
    from astropy.io import fits
    images = tmp_path / "images"
    work = tmp_path / "work"
    for night in ("night1", "night2"):
        d = images / "M 42" / night
        d.mkdir(parents=True)
        for i in range(3):
            h = fits.Header()
            h["OBJECT"], h["FILTER"], h["EXPTIME"] = "M 42", "Astro", 10.0
            h["TELESCOP"], h["DATE-OBS"], h["BAYERPAT"] = "DWARF 3", f"2026-10-1{night[-1]}T22:00:0{i}", "RGGB"
            fits.PrimaryHDU(np.zeros((8, 8), np.float32), header=h).writeto(d / f"sub_{i:04d}.fits")
    (images / "empty").mkdir()
    # a calibration folder is not a dataset
    (images / "darks").mkdir()
    fits.PrimaryHDU(np.zeros((8, 8), np.float32)).writeto(images / "darks" / "dark_0001.fits")
    monkeypatch.setitem(server.CONFIG, "images", str(images))
    monkeypatch.setitem(server.CONFIG, "workdir", str(work))
    monkeypatch.setattr(server, "Session", FakeSession)
    monkeypatch.setattr(server, "tools_mod", fake_tools_module())
    for d in (server.JOBS, server.SESSIONS, server.PREVIEW_CACHE):
        d.clear()
    server.QUEUE.clear()
    FakeSession.instances.clear()
    FakeRCAstro.calls.clear()
    from fastapi.testclient import TestClient
    yield types.SimpleNamespace(client=TestClient(server.app), images=str(images), work=str(work))
    _drain()


def _drain(timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not server.JOB_LOCK.locked() and not server.QUEUE:
            return
        time.sleep(0.05)


def wait_state(client, job_id, states=("done", "error", "cancelled"), timeout=15, base="/api/v1"):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"{base}/jobs/{job_id}").json()
        if j["state"] in states:
            return j
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} stuck: {j}")


def submit(env, **body):
    body.setdefault("path", "M 42/night1")
    r = env.client.post("/api/v1/jobs", json=body)
    assert r.status_code == 200, r.text
    return r.json()


# --------------------------------------------------------------------------- v1: info

def test_health(env):
    h = env.client.get("/api/v1/health").json()
    assert h["ok"] is True and h["api"] == 1 and h["version"] and h["busy"] is False and h["queued"] == 0
    assert isinstance(h["time"], float)


def test_info_fields(env):
    r = env.client.get("/api/v1").json()
    for k in ("name", "api", "version", "devices", "images_root", "uploads_root", "workdir", "job_kinds", "default_kind",
              "default_profile", "profiles", "presets", "stack_defaults", "queue"):
        assert k in r, k
    assert r["name"] == "AstroPhoto Studio" and r["api"] == 1
    assert r["default_kind"] == "all" and r["default_profile"] == "default"
    assert r["images_root"] == env.images and r["workdir"] == env.work
    assert {"default", "gpus"} <= set(r["devices"])
    for kind, label in {"calibrate": "Calibrate", "analyse": "Analyse frames", "stack": "Register & integrate",
                        "denoise": "Restore", "starnet": "Train star remover", "autofinish": "Auto-finish",
                        "all": "Run everything & export", "export": "Export"}.items():
        assert r["job_kinds"][kind] == label
    # additive fields
    assert r["backend"] == "siril"
    assert r["tools"]["siril"] is True and r["tools"]["rcastro"] is True and r["tools"]["starnet"] is False
    assert r["queue"] == {"running": 0, "queued": 0}
    assert r["presets"]["Balanced"] == {} and "Vivid nebula" in r["presets"]


def test_profiles(env):
    r = env.client.get("/api/v1/profiles").json()
    assert r["default"] == "default"
    assert list(r["profiles"]) == ["default", "imagemm", "n2n-imagemm", "n2n-network", "n2n-rl"]
    p = r["profiles"]["n2n-network"]
    assert p["label"].startswith("Fast restoration: BlurXTerminator")
    assert p["stack_params"] == {"deconv_method": "auto", "ai_deconvolution": True, "scale": 2.0}
    assert p["effective"] == {"deconv_method": "auto", "ai_deconvolution": True, "imagemm_n2n": True,
                              "star_remover": True, "scale": 2.0}
    d = r["profiles"]["default"]["effective"]
    assert d == {"deconv_method": "auto", "ai_deconvolution": True, "imagemm_n2n": True, "star_remover": True, "scale": 1.0}
    assert r["profiles"]["n2n-rl"]["effective"]["ai_deconvolution"] is False
    assert r["profiles"]["n2n-rl"]["effective"]["deconv_method"] == "rl"


def test_datasets_resolve(env):
    r = env.client.post("/api/v1/datasets/resolve", json={"path": "M 42/night1"}).json()
    assert r["images_root"] == env.images and "uploads_root" in r
    assert r["exists"] is True and r["n_fits"] == 3 and r["bytes"] > 0
    assert r["object"] == "M 42" and r["filter"] == "Astro" and r["exptime"] == 10.0
    assert r["instrument"] == "DWARF 3" and r["date"] == "2026-10-11" and r["total_min"] == 0.5
    assert r["folders"] == [os.path.join(env.images, "M 42", "night1")]
    two = env.client.post("/api/v1/datasets/resolve", json={"paths": ["M 42/night2", "M 42/night1"]}).json()
    assert two["n_fits"] == 6 and two["folders"] == sorted(two["folders"])
    missing = env.client.post("/api/v1/datasets/resolve", json={"path": "nope"}).json()
    assert missing["exists"] is False and missing["n_fits"] == 0


def test_resolve_errors(env):
    c = env.client
    assert c.post("/api/v1/datasets/resolve", json={}).status_code == 400
    r = c.post("/api/v1/datasets/resolve", json={"path": "../x"})
    assert r.status_code == 400 and "escapes" in r.json()["detail"]
    r = c.post("/api/v1/datasets/resolve", json={"path": "a", "source": "weird"})
    assert r.status_code == 400 and "unknown source" in r.json()["detail"]


def test_light_folders_child(env):
    d = os.path.join(env.images, "target", "Lights")
    os.makedirs(os.path.join(d, "rejected"))
    from astropy.io import fits
    for sub in (d, os.path.join(d, "rejected")):
        fits.PrimaryHDU(np.zeros((4, 4), np.float32)).writeto(os.path.join(sub, "a.fits"))
    r = env.client.post("/api/v1/datasets/resolve", json={"path": "target"}).json()
    assert r["folders"] == sorted([d, os.path.join(d, "rejected")]) and r["n_fits"] == 2


def test_uploads_roundtrip(env):
    c = env.client
    assert c.get("/api/v1/uploads").json()["files"] == []
    r = c.put("/api/v1/uploads/s1/sub_1.fits", content=b"abc")
    assert r.json() == {"path": "s1/sub_1.fits", "bytes": 3}
    assert c.get("/api/v1/uploads", params={"prefix": "s1"}).json()["files"] == [{"path": "s1/sub_1.fits", "bytes": 3}]
    assert c.put("/api/v1/uploads/", content=b"x").status_code in (400, 404, 405)


# --------------------------------------------------------------------------- v1: jobs

def test_submit_all_job_runs_and_outputs(env):
    j = submit(env, client_ref="ref-1", label="M42 night 1", profile="n2n-network",
               stack_params={"autofinish": True}, params={"saturation": 1.8}, export={"quality": 92, "upscale": 1, "tiff": True})
    assert j["duplicate"] is False and j["state"] in ("queued", "running", "done")
    assert j["kind"] == "all" and j["kind_label"] == "Run everything & export" and j["label"] == "M42 night 1"
    assert j["dataset"] == "M 42" and j["profile"] == "n2n-network" and j["origin"] == "api" and j["client_ref"] == "ref-1"
    assert j["folders"] == [os.path.join(env.images, "M 42", "night1")] and j["folder"] == j["folders"][0]
    assert j["dataset_info"]["n_fits"] == 3 and j["dataset_info"]["object"] == "M 42"
    for k in ("log", "stages", "traceback", "device_state", "params", "stack_params", "export_opts"):
        assert k not in j
    assert j["outputs"] == []
    done = wait_state(env.client, j["id"])
    assert done["state"] == "done" and done["progress"] == 1.0 and done["message"] == "Finished"
    assert done["log"] and done["stages"]
    names = [o["name"] for o in done["outputs"]]
    assert names == ["M42_20261014-101500.jpg", "M42_20261014-101500.tif", "M42_20261014-101500.json"]
    assert [o["kind"] for o in done["outputs"]] == ["jpg", "tif", "json"]
    assert done["outputs"][0]["url"] == f"/api/v1/jobs/{j['id']}/files/M42_20261014-101500.jpg"
    assert done["result"]["size"] == [8, 8] and done["result"]["autofinish"]["params"]["saturation"] == 1.7
    for k in ("elapsed", "active", "ended", "active_seconds", "stage", "stage_key", "created", "started"):
        assert k in done
    # the server path of "all": analyse, stack, solve, restore, star removal, autofinish, export (tuned params exported)
    s = FakeSession.instances[0]
    assert [c if isinstance(c, str) else c[0] for c in s.calls] == ["analyse", "stack", "solve", "restore", "stars", "autofinish", "export"]
    stack_call = s.calls[1][1]
    assert stack_call["deconv_method"] == "auto" and stack_call["scale"] == 2.0 and stack_call["sigma_low"] == 4.0
    export_call = s.calls[-1]
    assert export_call[1]["saturation"] == 1.7 and export_call[2:] == (92, 1.0, True)
    f = env.client.get(done["outputs"][0]["url"])
    assert f.status_code == 200 and f.content == b"data-jpg"
    assert "attachment" in f.headers["content-disposition"]
    assert "attachment" not in env.client.get(done["outputs"][0]["url"], params={"inline": "true"}).headers.get("content-disposition", "")
    assert env.client.get(f"/api/v1/jobs/{j['id']}/files/nope.jpg").status_code == 404


def test_autofinish_off_exports_given_params(env):
    j = submit(env, stack_params={"autofinish": False, "star_remover": False}, params={"saturation": 1.1})
    done = wait_state(env.client, j["id"])
    s = FakeSession.instances[0]
    assert [c if isinstance(c, str) else c[0] for c in s.calls] == ["analyse", "stack", "solve", "restore", "export"]
    assert s.calls[-1][1] == {"saturation": 1.1} and "autofinish" not in done["result"]


def test_idempotent_client_ref(env):
    a = submit(env, client_ref="same")
    b = submit(env, client_ref="same", path="M 42/night2", kind="export")
    assert b["duplicate"] is True and b["id"] == a["id"] and "dataset_info" not in b and b["kind"] == "all"
    wait_state(env.client, a["id"])
    c = submit(env, client_ref="same")            # done jobs are still returned
    assert c["duplicate"] is True and c["id"] == a["id"]
    assert len(env.client.get("/api/v1/jobs", params={"client_ref": "same"}).json()["jobs"]) == 1


def test_client_ref_reusable_after_cancel(env):
    gate = threading.Event()
    orig = FakeSession.run_analysis

    def slow(self, sens, progress):
        gate.wait(5)
        return orig(self, sens, progress)
    FakeSession.run_analysis = slow
    try:
        first = submit(env, client_ref="r2", kind="analyse")
        second = submit(env, client_ref="r2b", kind="analyse")
        assert env.client.post(f"/api/v1/jobs/{second['id']}/cancel").json()["state"] == "cancelled"
        again = submit(env, client_ref="r2b", kind="analyse")
        assert again["duplicate"] is False and again["id"] != second["id"]
        gate.set()
        wait_state(env.client, first["id"]); wait_state(env.client, again["id"])
    finally:
        FakeSession.run_analysis = orig
        gate.set()


def test_submit_validation(env):
    c = env.client
    r = c.post("/api/v1/jobs", json={"path": "M 42/night1", "kind": "bogus"})
    assert r.status_code == 400 and r.json()["detail"].startswith("unknown kind 'bogus'; one of [")
    r = c.post("/api/v1/jobs", json={"path": "M 42/night1", "profile": "zzz"})
    assert r.status_code == 400 and r.json()["detail"].startswith("unknown profile 'zzz'; one of")
    r = c.post("/api/v1/jobs", json={"path": "M 42/night1", "preset": "zzz"})
    assert r.status_code == 400 and r.json()["detail"].startswith("unknown preset 'zzz'; one of")
    r = c.post("/api/v1/jobs", json={})
    assert r.status_code == 400 and "give path" in r.json()["detail"]
    r = c.post("/api/v1/jobs", json={"path": "missing-folder"})
    assert r.status_code == 404 and r.json()["detail"].startswith("folder not found on the server:")
    r = c.post("/api/v1/jobs", json={"path": "empty"})
    assert r.status_code == 422 and r.json()["detail"].startswith("no FITS subs in")
    r = c.post("/api/v1/jobs", json={"path": "M 42/night1", "kind": "tools"})
    assert r.status_code == 400


def test_preset_and_param_merge_order(env):
    j = submit(env, preset="Vivid nebula", params={"saturation": 1.2}, stack_params={"autofinish": False, "star_remover": False})
    wait_state(env.client, j["id"])
    exported = FakeSession.instances[0].calls[-1][1]
    assert exported["saturation"] == 1.2 and exported["local_contrast"] == 1.4 and exported["stretch"] == 0.09


def test_job_list_filters_and_get(env):
    a = submit(env, kind="calibrate", origin="telescope")
    b = submit(env, kind="analyse")
    wait_state(env.client, a["id"]); wait_state(env.client, b["id"])
    lst = env.client.get("/api/v1/jobs").json()["jobs"]
    assert [x["id"] for x in lst] == [b["id"], a["id"]]            # newest first
    assert all("log" not in x for x in lst)
    assert [x["id"] for x in env.client.get("/api/v1/jobs", params={"origin": "telescope"}).json()["jobs"]] == [a["id"]]
    assert len(env.client.get("/api/v1/jobs", params={"state": "done", "limit": 1}).json()["jobs"]) == 1
    assert env.client.get("/api/v1/jobs", params={"state": "error"}).json()["jobs"] == []
    one = env.client.get(f"/api/v1/jobs/{a['id']}").json()
    assert "log" in one and "stages" in one and one["result"] == {"changed_analysis": False, "summary": "no masters"}
    r = env.client.get("/api/v1/jobs/unknown")
    assert r.status_code == 404 and r.json()["detail"] == "no job unknown"


def test_error_job(env):
    def boom(self, sens, progress):
        raise ValueError("kaput")
    orig = FakeSession.run_analysis
    FakeSession.run_analysis = boom
    try:
        j = submit(env, kind="analyse")
        done = wait_state(env.client, j["id"])
    finally:
        FakeSession.run_analysis = orig
    assert done["state"] == "error" and done["message"] == "ValueError: kaput"
    assert "Traceback" in done["traceback"] and done["device_state"] == ""
    assert os.path.exists(os.path.join(FakeSession.instances[0].dir, "job_errors.log"))


def test_cancel_queued_and_running(env):
    gate = threading.Event()
    orig = FakeSession.run_analysis

    def slow(self, sens, progress):
        for i in range(200):
            progress(i, 200, f"Measuring {i}/200")
            if gate.wait(0.02):
                break
        return {"n": 3}
    FakeSession.run_analysis = slow
    try:
        running = submit(env, kind="analyse")
        queued = submit(env, kind="analyse", path="M 42/night2")
        t0 = time.time()
        while env.client.get(f"/api/v1/jobs/{running['id']}").json()["state"] != "running" and time.time() - t0 < 5:
            time.sleep(0.02)
        q = env.client.get("/api/v1/queue").json()
        assert [x["id"] for x in q["running"]] == [running["id"]] and [x["id"] for x in q["queued"]] == [queued["id"]]
        assert q["queued"][0]["position"] == 1 and "starts_in" in q["queued"][0] and q["queued"][0]["waiting_on"][0]["id"] == running["id"]
        assert q["external"] == [] and "drains_in" in q
        c = env.client.post(f"/api/v1/jobs/{queued['id']}/cancel").json()
        assert c["state"] == "cancelled" and c["message"] == "Removed from the queue" and "ended" in c
        c = env.client.post(f"/api/v1/jobs/{running['id']}/cancel").json()
        assert c["message"] in ("Cancelling at the next checkpoint", "Cancelled")
        done = wait_state(env.client, running["id"])
        assert done["state"] == "cancelled" and done["message"] == "Cancelled"
        assert env.client.post(f"/api/v1/jobs/{running['id']}/cancel").json()["state"] == "cancelled"     # finished: no-op
        assert env.client.post("/api/v1/jobs/nope/cancel").status_code == 404
    finally:
        FakeSession.run_analysis = orig
        gate.set()


def test_queue_move(env):
    gate = threading.Event()
    orig = FakeSession.run_calibration

    def slow(self, progress):
        gate.wait(5)
        return {"changed_analysis": False, "summary": None}
    FakeSession.run_calibration = slow
    try:
        first = submit(env, kind="calibrate")
        t0 = time.time()
        while env.client.get(f"/api/v1/jobs/{first['id']}").json()["state"] != "running" and time.time() - t0 < 5:
            time.sleep(0.02)
        a = submit(env, kind="calibrate", path="M 42/night2")
        b = submit(env, kind="analyse")
        assert env.client.get("/api/v1/queue").json()["queued"][0]["id"] == a["id"]
        r = env.client.post("/api/v1/queue/move", json={"id": b["id"], "position": 1})
        assert r.json() == {"queue": [b["id"], a["id"]]}
        r = env.client.post("/api/v1/queue/move", json={"id": first["id"]})
        assert r.status_code == 409 and r.json()["detail"] == f"job {first['id']} is not waiting in the queue"
        gate.set()
        for j in (first, a, b):
            wait_state(env.client, j["id"])
    finally:
        FakeSession.run_calibration = orig
        gate.set()


def test_jobs_persist_and_interrupt(env, monkeypatch):
    j = submit(env, kind="calibrate")
    wait_state(env.client, j["id"])
    server._save_jobs(force=True)
    saved = json.load(open(os.path.join(env.work, "jobs.json")))
    assert [x["id"] for x in saved["jobs"]] == [j["id"]] and saved["jobs"][0]["params"] == {} and "queue" in saved
    saved["jobs"][0]["state"], saved["jobs"][0]["started"] = "running", 1.0
    json.dump(saved, open(os.path.join(env.work, "jobs.json"), "w"))
    server.JOBS.clear()
    got = env.client.get(f"/api/v1/jobs/{j['id']}").json()
    assert got["state"] == "interrupted" and got["message"] == "Interrupted: the server stopped while this job was running"


# --------------------------------------------------------------------------- UI routes

def test_ui_pages(env):
    c = env.client
    r = c.get("/")
    assert r.status_code == 200 and "AstroPhoto" in r.text and "tools.js" in r.text
    assert "explore.js" not in r.text and "lab.js" not in r.text and 'data-tab="lab"' not in r.text
    assert c.get("/static/app.js").status_code == 200 and c.get("/static/tools.js").status_code == 200
    assert c.get("/static/lab.js").status_code == 404


def test_system(env):
    s = env.client.get("/api/system").json()
    assert s["backend"] == "siril" and s["version"] and "defaults" in s and s["param_spec"]
    assert s["stack_defaults"]["deconv_method"] == "auto" and s["stack_defaults"]["sigma_low"] == 4.0
    assert s["tools"]["rcastro"]["ok"] is True and "Balanced" in s["presets"] and s["images_root"] == env.images
    assert {"group", "key", "label", "type"} <= set(s["param_spec"][0])


def test_datasets_discovery(env):
    r = env.client.get("/api/datasets").json()
    assert r["root"] == env.images and r["sep"] == os.pathsep
    names = sorted(d["name"] for d in r["datasets"])
    assert names == [os.path.join("M 42", "night1"), os.path.join("M 42", "night2")]      # darks/ is not a dataset
    d = r["datasets"][0]
    assert d["n_fits"] == 3 and d["object"] == "M 42" and d["filter"] == "Astro" and d["total_min"] == 0.5
    assert d["cached"] == {"analysed": False, "stacked": False, "denoised": False}
    assert env.client.get("/api/datasets", params={"root": "/nonexistent-xyz"}).json()["datasets"] == []


def test_open_frames_calibration_thumb(env):
    c = env.client
    folder = os.path.join(env.images, "M 42", "night1")
    r = c.post("/api/open", json={"folder": folder}).json()
    st = r["status"]
    for k in ("folder", "folders", "workdir", "stacked", "denoised", "star_remover", "autofinished", "analysed", "narrowband", "telescope"):
        assert k in st, k
    assert st["object"] == "M 42" and st["stacked"] is False and r["frames"][0]["name"] == "sub_0001.fits"
    assert c.get("/api/frames", params={"folder": folder}).json()["sensitivity"] == 1.0
    o = c.post("/api/frames/override", json={"folder": folder, "name": "sub_0001.fits", "accepted": False}).json()
    assert "frames" in o and ("override", "sub_0001.fits", False) in FakeSession.instances[0].calls
    cal = c.get("/api/calibration", params={"folder": folder}).json()
    assert cal["n_lights"] == 3 and "status" in cal
    assert c.post("/api/calibration", json={"folder": folder, "prefs": {"enabled": False}}).json() == {"prefs": {"enabled": False}}
    t = c.get("/api/thumb", params={"folder": folder, "name": "sub_0001.fits", "size": 64})
    assert t.status_code == 200 and t.headers["content-type"] == "image/jpeg" and t.content[:2] == b"\xff\xd8"
    assert c.get("/api/frames", params={"folder": "/does/not/exist"}).status_code == 404
    assert c.get("/api/frames", params={"folder": ""}).status_code in (400, 422)


def test_ui_jobs_flow(env):
    c = env.client
    folder = os.path.join(env.images, "M 42", "night1")
    j = c.post("/api/jobs", json={"kind": "export", "folder": folder, "params": {"saturation": 1.3},
                                  "stack_params": {}, "export": {"quality": 90, "upscale": 1, "tiff": False}}).json()
    assert j["label"] == "Export" and j["state"] in ("queued", "running", "done")     # UI label is the kind's name
    done = wait_state(c, j["id"], base="/api")
    assert done["state"] == "done" and "tif" not in done["result"]
    lst = c.get("/api/jobs").json()
    assert set(lst) == {"active", "history", "external"} and lst["external"] == [] and lst["history"][0]["id"] == j["id"]
    r = c.post(f"/api/jobs/{j['id']}/rerun").json()
    assert r["kind"] == "export" and "client_ref" not in r
    wait_state(c, r["id"], base="/api")
    assert c.post("/api/jobs/nope/cancel").status_code == 404
    assert c.post(f"/api/jobs/{j['id']}/pause").status_code == 409
    assert c.post(f"/api/jobs/{j['id']}/resume").status_code == 409
    assert c.post("/api/jobs", json={"kind": "nope", "folder": folder}).json()["detail"] == "unknown job kind"
    assert c.delete("/api/jobs").json() == {"ok": True}
    assert c.get("/api/jobs").json()["history"] == []
    ex = c.get("/api/exports", params={"folder": folder}).json()["exports"]
    assert ex and ex[0]["name"] == "M42_20261014-101500.jpg" and ex[0]["tif"] is None
    assert c.get("/api/download", params={"path": ex[0]["jpg"]}).content == b"data-jpg"
    assert c.get("/api/download", params={"path": "/etc/passwd"}).status_code == 403


def test_pause_resume(env):
    c = env.client
    folder = os.path.join(env.images, "M 42", "night1")
    gate = threading.Event()
    orig = FakeSession.run_calibration

    def stepped(self, progress):
        for i in range(400):
            progress(i, 400, f"Calibrating {i}/400")
            if gate.wait(0.01):
                break
        return {"changed_analysis": False, "summary": None}
    FakeSession.run_calibration = stepped
    try:
        j = c.post("/api/jobs", json={"kind": "calibrate", "folder": folder}).json()
        t0 = time.time()
        while c.get(f"/api/jobs/{j['id']}").json()["state"] != "running" and time.time() - t0 < 5:
            time.sleep(0.02)
        assert c.post(f"/api/jobs/{j['id']}/pause").json() == {"ok": True}
        time.sleep(0.4)
        p1 = c.get(f"/api/jobs/{j['id']}").json()
        assert p1["state"] == "paused"
        time.sleep(0.4)
        assert c.get(f"/api/jobs/{j['id']}").json()["progress"] == p1["progress"]          # blocked at the pause point
        assert c.post(f"/api/jobs/{j['id']}/resume").json() == {"ok": True}
        time.sleep(0.3)
        assert c.get(f"/api/jobs/{j['id']}").json()["progress"] > p1["progress"]
        gate.set()
        assert wait_state(c, j["id"], base="/api")["state"] == "done"
    finally:
        FakeSession.run_calibration = orig
        gate.set()


def test_preview(env):
    c = env.client
    folder = os.path.join(env.images, "M 42", "night1")
    r = c.post("/api/preview", json={"folder": folder, "params": {}, "which": "after", "size": 1400})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.headers["x-width"] == "9" and r.headers["x-height"] == "6" and "x-render-time" in r.headers
    b = c.post("/api/preview", json={"folder": folder, "which": "before"})
    assert b.status_code == 400                                         # not stacked yet
    with server.JOB_LOCK:                                               # a job running: 409 unless cached
        r2 = c.post("/api/preview", json={"folder": folder, "params": {"stretch": 0.2}, "which": "after"})
        assert r2.status_code == 409
        assert c.post("/api/preview", json={"folder": folder, "params": {}, "which": "after", "size": 1400}).status_code == 200


def test_diagnostic_logs_and_autofinish(env):
    c = env.client
    folder = os.path.join(env.images, "M 42", "night1")
    sess = server.get_session(folder)
    assert c.get("/api/diagnostic", params={"folder": folder, "kind": "logs"}).json() == {"logs": []}
    logs = os.path.join(sess.dir, "process", "logs")
    os.makedirs(logs)
    open(os.path.join(logs, "stack.log"), "w").write("\n".join(f"line {i}" for i in range(10)))
    got = c.get("/api/diagnostic", params={"folder": folder, "kind": "logs"}).json()["logs"]
    assert got[0]["name"] == "stack.log" and got[0]["bytes"] > 0
    assert c.get("/api/diagnostic", params={"folder": folder, "kind": "log", "name": "stack.log", "tail": 2}).text == "line 8\nline 9"
    assert c.get("/api/diagnostic", params={"folder": folder, "kind": "log", "name": "../x"}).status_code == 404
    assert c.get("/api/diagnostic", params={"folder": folder, "kind": "reference"}).status_code == 404
    assert c.get("/api/autofinish", params={"folder": folder}).json() == {"result": None}
    json.dump({"params": {"stretch": 0.2}, "report": {}}, open(os.path.join(sess.dir, "autofinish.json"), "w"))
    assert c.get("/api/autofinish", params={"folder": folder}).json()["result"]["params"] == {"stretch": 0.2}
    assert set(c.get("/api/linear_info", params={"folder": folder}).json()) == {"restore", "starnet"}


def test_lab_and_explore_removed(env):
    c = env.client
    assert c.get("/api/lab/best").json() == {"studies": []}
    for method, path in (("get", "/api/lab/tasks"), ("get", "/api/lab/studies"), ("post", "/api/lab/studies"),
                         ("delete", "/api/lab/studies/x"), ("get", "/api/explore/status"), ("post", "/api/explore/solve"),
                         ("post", "/api/explore/data")):
        r = getattr(c, method)(path)
        assert r.status_code == 410 and "removed" in r.json()["detail"], path


# --------------------------------------------------------------------------- tools

def test_tools_status(env):
    t = env.client.get("/api/tools").json()
    assert set(t) == {"siril", "graxpert", "starnet", "rcastro"}
    assert t["rcastro"]["products"][0]["message"] == "Trial: 27 days left"
    assert t["rcastro"]["activationUrl"].startswith("https://www.rc-astro.com")


def test_tools_status_never_crashes(env, monkeypatch):
    def broken():
        raise RuntimeError("tool probe exploded")
    monkeypatch.setattr(server.tools_mod, "tools_status", broken)
    t = env.client.get("/api/tools")
    assert t.status_code == 200 and t.json()["rcastro"]["ok"] is False and "exploded" in t.json()["rcastro"]["error"]
    assert env.client.get("/api/system").status_code == 200 and env.client.get("/api/v1").json()["tools"]["siril"] is False


def test_rcastro_license(env):
    c = env.client
    r = c.post("/api/tools/rcastro/license", json={"code": "abcd-efgh"})
    assert r.status_code == 200
    body = r.json()
    assert body["report"]["operation"] == "code" and body["report"]["licenseStatus"][0]["message"] == "Trial: 27 days left"
    assert body["tools"]["account"] == "someone@example.com"
    assert FakeRCAstro.calls[-1] == ("license", "ABCDEFGH")
    for bad in ({}, {"code": ""}, {"code": "abc"}, {"code": "abcd-efgh-i"}):
        assert c.post("/api/tools/rcastro/license", json=bad).status_code == 400


def test_rcastro_activate_deactivate(env):
    c = env.client
    assert c.post("/api/tools/rcastro/activate", json={"product": "nxt"}).json()["report"]["operation"] == "activate"
    assert FakeRCAstro.calls[-1][:2] == ("activate", "nxt")
    assert c.post("/api/tools/rcastro/activate", json={}).status_code == 200
    assert FakeRCAstro.calls[-1][:2] == ("activate", "all")
    assert c.post("/api/tools/rcastro/activate", json={"product": "xyz"}).status_code == 400
    assert c.post("/api/tools/rcastro/deactivate", json={"product": "bxt"}).json()["report"]["operation"] == "deactivate"
    assert FakeRCAstro.calls[-1][:2] == ("deactivate", "bxt")
    assert c.post("/api/tools/rcastro/deactivate", json={}).status_code == 400


def test_licence_calls_refused_while_a_job_runs(env):
    with server.JOB_LOCK:
        r = env.client.post("/api/tools/rcastro/license", json={"code": "abcdefgh"})
    assert r.status_code == 409 and "job is running" in r.json()["detail"]
    assert not any(c[0] == "license" for c in FakeRCAstro.calls)


def test_rcastro_failure_is_a_502(env, monkeypatch):
    def refuse(self, code):
        raise RuntimeError("code rejected")
    monkeypatch.setattr(FakeRCAstro, "license_code", refuse)
    r = env.client.post("/api/tools/rcastro/license", json={"code": "abcdefgh"})
    assert r.status_code == 502 and "code rejected" in r.json()["detail"]
    assert not server.JOB_LOCK.locked()


def test_tools_job_kind(env):
    c = env.client
    j = c.post("/api/tools/models/download", json={"tool": "graxpert"}).json()
    assert j["kind"] == "tools" and j["label"] == "Tools" and j["folder"] == ""
    done = wait_state(c, j["id"], base="/api")
    assert done["state"] == "done" and done["result"]["tool"] == "graxpert" and done["progress"] == 1.0
    v1 = c.get(f"/api/v1/jobs/{j['id']}").json()
    assert v1["kind"] == "tools" and v1["folders"] == [] and v1["kind_label"] == "Tools" and v1["state"] == "done"
    j = c.post("/api/tools/models/download", json={"tool": "rcastro"}).json()
    assert wait_state(c, j["id"], base="/api")["result"]["downloaded"] == ["bxt"]
    j = c.post("/api/tools/models/download", json={"tool": "starnet", "accept_licence": True}).json()
    assert wait_state(c, j["id"], base="/api")["result"]["path"] == "/tmp/starnet2"
    assert c.get("/api/v1").json()["job_kinds"]["tools"] == "Tools"


def test_tools_job_errors(env):
    c = env.client
    assert c.post("/api/tools/models/download", json={"tool": "nope"}).status_code == 400
    r = c.post("/api/tools/models/download", json={"tool": "starnet"})
    assert r.status_code == 400 and "starnetastro.com" in r.json()["detail"]


def test_tools_job_shares_the_job_lock(env):
    """A download queued behind a processing job waits for it: they never overlap."""
    gate = threading.Event()
    orig = FakeSession.run_calibration

    def slow(self, progress):
        gate.wait(5)
        return {"changed_analysis": False, "summary": None}
    FakeSession.run_calibration = slow
    try:
        a = submit(env, kind="calibrate")
        t0 = time.time()
        while env.client.get(f"/api/v1/jobs/{a['id']}").json()["state"] != "running" and time.time() - t0 < 5:
            time.sleep(0.02)
        d = env.client.post("/api/tools/models/download", json={"tool": "graxpert"}).json()
        time.sleep(1.3)
        assert env.client.get(f"/api/jobs/{d['id']}").json()["state"] == "queued"
        gate.set()
        wait_state(env.client, a["id"]); assert wait_state(env.client, d["id"], base="/api")["state"] == "done"
    finally:
        FakeSession.run_calibration = orig
        gate.set()


def test_tools_job_cancel_does_not_need_a_session(env):
    j = env.client.post("/api/tools/models/download", json={"tool": "graxpert"}).json()
    r = env.client.post(f"/api/jobs/{j['id']}/cancel")
    assert r.status_code == 200
    wait_state(env.client, j["id"], states=("done", "cancelled", "error"))


def _real_backend_present():
    try:
        import sirilpipe.session, sirilpipe.post, sirilpipe.tools  # noqa: F401
        return not hasattr(sys.modules["sirilpipe.tools"], "__stand_in__")
    except Exception:
        return False


@pytest.mark.skipif(not _real_backend_present(), reason="sirilpipe.session / post / tools have not landed")
def test_server_imports_without_tools():
    """`python -m webui.server` must import on a host with none of the tools installed."""
    import subprocess
    code = ("import os; os.environ['PATH']='/nonexistent'; import webui.server as s; "
            "t = s.tools_status(); assert isinstance(t, dict); print(sorted(t))")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
