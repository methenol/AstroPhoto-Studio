"""session.py orchestration with a fake SirilRunner (no Siril, no Docker): scripts issued, files and metadata
written, overrides reaching the unselect commands, cancel, plate-solve failure being non-fatal."""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest

from sirilpipe import fitsio, session as sess
from sirilpipe.siril import SirilError, SirilResult
from sirilpipe.util import Cancelled, clean_json, dataset_slug, restoration_done, slugify, split_folders

from .conftest import FIX, load, touch


class FakeRunner:
    """Records scripts; creates what Siril would create for the stages the session runs."""
    log: list[tuple[str, str]] = []
    fail_stages: set[str] = set()

    def __init__(self, workdir, progress=None, checkpoint=None, **kw):
        self.workdir, self.progress, self.checkpoint = workdir, progress, checkpoint

    def run(self, script, stage, expect=(), **kw):
        FakeRunner.log.append((stage, script))
        if self.checkpoint:
            self.checkpoint()
        if stage in FakeRunner.fail_stages:
            raise SirilError(f"Siril stage '{stage}' failed")
        w = self.workdir
        os.makedirs(w, exist_ok=True)
        if stage.startswith("calibrate"):
            for e in expect:
                touch(e)
        elif stage.startswith("banding_merge"):                     # one merged mosaic per sub (the real seqmerge_cfa output names)
            import re
            from astropy.io import fits
            last = expect[0]
            m = re.match(r"(.*_)(\d{5})\.fit$", last)
            for i in range(1, int(m.group(2)) + 1):
                fits.PrimaryHDU(np.zeros((2, 2), np.float32)).writeto(f"{m.group(1)}{i:05d}.fit", overwrite=True)
        elif stage.startswith("banding"):
            for e in expect:
                touch(e)
        elif stage == "register":
            shutil.copy(os.path.join(FIX, "siril_registered_all.seq"), os.path.join(w, "light_.seq"))
        elif stage == "apply_registration":
            shutil.copy(os.path.join(FIX, "siril_applied.seq"), os.path.join(w, "r_light_.seq"))
        elif stage == "stack":
            data = np.zeros((3, 60, 80), np.float32)
            data[:, 5:55, 5:75] = np.random.default_rng(0).random((3, 50, 70)).astype(np.float32) * 0.1 + 0.01
            fitsio.write_rgb(os.path.join(w, "stack_raw.fit"), data, {"EXPTIME": 15.0, "OBJECT": "M 42"})
        elif stage == "platesolve":
            for e in expect:
                fitsio.write_rgb(e, np.ones((3, 40, 60), np.float32), {"CTYPE1": "RA---TAN", "CRVAL1": 83.8, "CRVAL2": -5.4,
                                                                      "CD1_1": -0.0007, "CD2_2": 0.0007})
        else:
            for e in expect:
                touch(e)
        return SirilResult(log_path="", seconds=0.1)


@pytest.fixture(autouse=True)
def _all_registered_seq(tmp_path_factory):
    """A 12-frame registered .seq whose frames match the 12 subs the tests create."""
    src = os.path.join(FIX, "siril_registered.seq")
    dst = os.path.join(FIX, "siril_registered_all.seq")
    if not os.path.exists(dst):
        text = open(src).read().replace("I 3 0", "I 3 1").replace("I 4 0", "I 4 1")
        open(dst, "w").write(text)
    yield


@pytest.fixture()
def session(tmp_path, fake_headers, monkeypatch):
    heads = load("dwarf_headers.json")
    base = next(iter(heads.values()))
    d = tmp_path / "DWARF_RAW_TELE_M 42_EXP_15_GAIN_60_x"
    for i in range(12):
        n = f"M 42_15s60_Duo-Band_2026101004{12 + i:02d}00000_41C.fits"
        touch(str(d / n))
        fake_headers[n] = {**base, "DATE-OBS": f"2026-10-10T04:{12 + i:02d}:00"}
    monkeypatch.setattr(sess, "SirilRunner", FakeRunner)
    FakeRunner.log, FakeRunner.fail_stages = [], set()
    return sess.Session([str(d)], str(tmp_path / "out"))


def test_slug_helpers_keep_old_semantics(tmp_path):
    a, b = str(tmp_path / "n1"), str(tmp_path / "n 2")
    assert slugify(a).startswith("n1-") and len(slugify(a).split("-")[-1]) == 6
    assert split_folders(f"{b}{os.pathsep}{a}") == sorted([a, b]) == split_folders([a, b])
    assert dataset_slug([a]) == slugify(a)
    assert "+1-" in dataset_slug([a, b])
    c = clean_json({"a": float("nan"), "b": np.float32(1.5), "c": [np.int64(2), float("inf")]})
    assert c == {"a": None, "b": 1.5, "c": [2, None]}
    assert not restoration_done(str(tmp_path))


def test_status_before_and_after_scan(session):
    st = session.status()
    assert st["n_files"] is None and st["has_stack"] is False and st["exports"] == []
    session.scan()
    st = session.status()
    assert st["n_files"] == 12 and st["object"] == "M 42" and st["kind"] == "dwarf" and st["is_dualband"] is True
    assert st["exptime"] == 15.0 and st["total_min"] == 3.0 and st["instrument"] == "DWARF 3"
    for k in ("has_linear", "has_starless", "has_autofinish", "calibration", "tools"):
        assert k in st


def test_analysis_writes_frames_json_with_included_flags(session):
    r = session.run_analysis(1.0)
    assert r == {"n": 12}
    rows = json.load(open(os.path.join(session.dir, "frames.json")))
    assert len(rows) == 12
    r0, r6 = rows[0], rows[6]
    assert r0["fwhm"] == pytest.approx(3.56601) and r0["nbstars"] == 102 and r0["included"] is True
    assert r0["is_reference"] is True and r6["wfwhm"] == pytest.approx(10.8049)
    assert {"name", "path", "exptime", "gain", "temp", "date", "wfwhm", "roundness", "bkg", "nbstars", "included",
            "reason"} <= set(r0)
    assert session.analysis["medians"]["fwhm"] and session.frames()["sensitivity"] == 1.0
    stages = [s for s, _ in FakeRunner.log]
    assert stages[:3] == ["calibrate_1", "register", ] or "register" in stages
    # a second analysis with the same inputs reuses the registration
    FakeRunner.log.clear()
    session.run_analysis(1.0)
    assert FakeRunner.log == [] and len(session.frames_table()) == 12 and session.analysis["sensitivity"] == 1.0
    s2 = sess.Session(session.folders, session.workdir)            # fresh object (server restart): same reuse
    s2.run_analysis(1.5)
    assert FakeRunner.log == [] and len(s2.frames_table()) == 12 and s2.frames()["sensitivity"] == 1.5


def test_override_reaches_unselect_and_survives_reload(session):
    session.run_analysis(1.0)
    name = session.frames_table()[4]["name"]
    session.set_override(name, False)
    assert session.frames_table()[4]["included"] is False and session.frames_table()[4]["overridden"] is True
    session.run_stack({})
    apply = dict(FakeRunner.log)["apply_registration"]
    assert "select light 1 12" in apply and "unselect light 5 5" in apply and apply.strip().endswith("-filter-included")
    # a new Session object (server restart) sees overrides and frames
    s2 = sess.Session(session.folders, session.workdir)
    assert s2.overrides == {name: False} and s2.frames_table()[4]["included"] is False
    s2.set_override(name, None)
    assert s2.frames_table()[4]["overridden"] is False


def test_run_stack_writes_stack_and_meta(session):
    session.run_analysis(1.0)
    meta = session.run_stack({"sigma_low": 3, "sigma_high": 3})
    assert os.path.exists(os.path.join(session.dir, "stack.fits"))
    on_disk = json.load(open(os.path.join(session.dir, "stack_meta.json")))
    for k in ("mode", "scale", "n_frames", "n_total", "total_exposure", "params", "saturation", "filter", "object",
              "calibration", "created", "shape", "drizzle", "rejection", "weights", "framing", "crop", "pixel_scale",
              "fwhm_px", "siril_version", "nights", "is_dualband", "plate_solve"):
        assert k in on_disk, k
    assert on_disk["mode"] == "drizzle" and on_disk["scale"] == 1.0 and on_disk["rejection"] == \
        {"type": "winsorized", "low": 3.0, "high": 3.0}
    assert on_disk["n_total"] == 12 and on_disk["n_frames"] == meta["n_frames"] and on_disk["pixel_scale"] == pytest.approx(2.75, abs=0.01)
    assert on_disk["crop"][2] <= 80 and on_disk["shape"] == on_disk["crop"][2:]
    stack_script = dict(FakeRunner.log)["stack"]
    assert "rej w 3 3" in stack_script and "-weight=wfwhm" in stack_script
    # the inherited WCS is not kept, plate solving then fails softly
    FakeRunner.fail_stages = {"platesolve"}
    r = session.plate_solve()
    assert r["ok"] is False and session.meta["plate_solve"]["ok"] is False
    FakeRunner.fail_stages = set()
    r = session.plate_solve()
    assert r["ok"] is True and session.meta["wcs"] is True and r["pixel_scale"] == pytest.approx(2.52, abs=0.01)


def test_two_x_request_falls_back_with_reason(session):
    session.run_analysis(1.0)
    meta = session.run_stack({"scale": 2})
    assert meta["scale"] == 1.0 and "requested 2x drizzle, using 1x" in meta["resampling_reason"]


def test_new_stack_invalidates_later_products(session):
    session.run_analysis(1.0)
    for f in ("linear.fits", "starless.fits", "autofinish.json"):
        touch(os.path.join(session.dir, f))
    session.run_stack({})
    assert not os.path.exists(os.path.join(session.dir, "linear.fits"))


def test_cancel_during_siril_stage(session):
    session.cancel_flag.set()
    with pytest.raises(Cancelled):
        session.run_analysis(1.0)
    session.cancel_flag.clear()
    assert session.checkpoint() is False


def test_calibration_info_and_prefs(session):
    session.scan()
    info = session.calibration_info()
    assert info["profile"]["telescope"] == "DWARF 3" and info["n_lights"] == 12
    assert set(info["candidates"]) == {"bias", "dark", "flat"}
    assert session.set_calib_prefs({"flat": "none", "bogus": 1})["flat"] == "none"
    assert "bogus" not in session.calib_prefs


def test_run_calibration_reports_summary(session):
    r = session.run_calibration()
    assert set(r) == {"changed_analysis", "summary"} and r["changed_analysis"] is False
    assert os.path.exists(os.path.join(session.dir, "calibration.json"))


def test_post_stages_are_delegated_lazily(session, monkeypatch):
    import sys
    import types
    calls = []
    fake = types.ModuleType("sirilpipe.post")
    fake.restore = lambda *a: calls.append("restore") or {"ok": True}
    fake.remove_stars = lambda *a: calls.append("stars") or {"tool": "x"}
    fake.autofinish = lambda *a: calls.append("auto") or {"params": {"stretch": 0.2}, "report": {}}
    fake.export = lambda d, meta, params, opts, p, c: calls.append(("export", opts)) or {"jpg": "a.jpg", "size": [1, 1]}
    fake.preview_jpeg = lambda d, meta, params, max_side=1600: b"jpeg"
    monkeypatch.setitem(sys.modules, "sirilpipe.post", fake)
    monkeypatch.setattr("sirilpipe.post", fake, raising=False)
    session.run_denoise({}, None, {})
    session.train_star_remover({}, {})
    a = session.autofinish({})
    assert a["params"] == {"stretch": 0.2} and a["autofinish"]["params"] == {"stretch": 0.2}
    assert session.export({}, quality=90, upscale=2.0, tiff=False)["jpg"] == "a.jpg"
    assert ("export", {"quality": 90, "upscale": 2.0, "tiff": False}) in calls
    assert session.preview_jpeg({}) == b"jpeg" and calls[:2] == ["restore", "stars"]


def test_local_copy_copies_and_cleans(session, monkeypatch, tmp_path):
    monkeypatch.setenv("ASTROPHOTO_LOCAL_COPY", "1")
    monkeypatch.setenv("ASTROPHOTO_LOCAL_DIR", str(tmp_path / "local"))
    msgs = []
    with session.local_copy(lambda i, n, m: msgs.append(m)) as s:
        assert s._local and all(os.path.exists(p) for p in s._local.values())
        assert msgs[-1].startswith("Copying the subs to the local disk (12/12")
    assert session._local == {} and not any(os.path.isdir(p) for p in (tmp_path / "local").glob("*") if any(p.iterdir()))


def test_banding_stage_runs_only_when_measured(session, monkeypatch):
    """Calibrated CFA subs are measured (workflow.row_banding_ratio); the Siril banding stages run only above the threshold and are recorded."""
    from sirilpipe import workflow
    monkeypatch.setattr(workflow, "row_banding_ratio", lambda paths, pattern="RGGB": {"ratio": 1.36, "needed": 1.1, "banded": True, "n_subs": 8})
    session.run_analysis(1.0)
    stages = [s for s, _ in FakeRunner.log]
    assert stages.index("calibrate_1") < stages.index("banding_split_1") < stages.index("banding_fix_1") < stages.index("banding_merge_1") < stages.index("register")
    assert "seqfixbanding CFA3_light 1 1" in dict(FakeRunner.log)["banding_fix_1"]
    st = json.load(open(os.path.join(session.process, "state.json")))
    assert st["banding"]["applied"] is True and st["banding"]["amount"] == 1.0 and st["banding"]["measured"][0]["ratio"] == 1.36
    assert st["signature"]["banding"][:2] == [1.0, 1.0]
    meta = session.run_stack({})
    assert meta["banding"]["applied"] is True
    # not banded: no Siril banding stage
    FakeRunner.log.clear()
    monkeypatch.setattr(workflow, "row_banding_ratio", lambda paths, pattern="RGGB": {"ratio": 0.98, "needed": 1.1, "banded": False, "n_subs": 8})
    session.run_stack({"pattern_correction": False})
    assert not [s for s, _ in FakeRunner.log if s.startswith("banding")]
