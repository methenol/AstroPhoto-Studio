"""session.py orchestration with a fake SirilRunner (no Siril, no Docker): scripts issued, files and metadata
written, overrides reaching the unselect commands, cancel, plate-solve failure being non-fatal."""
from __future__ import annotations

import glob
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
        elif stage.startswith("banding_merge") or stage.startswith("frame_bkg"):   # one merged / background-subtracted file per sub
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


def test_a_lone_sub_in_its_night_folder_is_left_out(tmp_path, fake_headers, monkeypatch):
    """Siril makes no sequence of one image ("No sequence `light' found", Sh2-132 2026-10-18): a night folder
    holding a single sub of the dataset is not calibrated as a unit of its own."""
    heads = load("dwarf_headers.json")
    base = next(iter(heads.values()))
    d1, d2 = tmp_path / "night1", tmp_path / "night2"
    for i in range(5):
        n = f"M 42_15s60_Duo-Band_2026101004{12 + i:02d}00000_41C.fits"
        touch(str(d1 / n))
        fake_headers[n] = {**base, "DATE-OBS": f"2026-10-10T04:{12 + i:02d}:00"}
    lone = "M 42_15s60_Duo-Band_20261011041200000_41C.fits"
    touch(str(d2 / lone))
    fake_headers[lone] = {**base, "DATE-OBS": "2026-10-11T04:12:00"}
    monkeypatch.setattr(sess, "SirilRunner", FakeRunner)
    s = sess.Session([str(d1), str(d2)], str(tmp_path / "out"))
    s.scan()
    units = s._make_units(debayer=False)
    assert len(units) == 1 and len(units[0].frames) == 5 and s.dropped_subs == [lone]


def test_a_dataset_of_one_sub_fails_before_siril(tmp_path, fake_headers, monkeypatch):
    heads = load("dwarf_headers.json")
    base = next(iter(heads.values()))
    d = tmp_path / "DWARF_RAW_TELE_SNR_x"
    n = "SNR_60s60_Duo-Band_20261018025859583_37C.fits"
    touch(str(d / n))
    fake_headers[n] = {**base, "OBJECT": "SNR", "DATE-OBS": "2026-10-18T02:58:59"}
    monkeypatch.setattr(sess, "SirilRunner", FakeRunner)
    FakeRunner.log, FakeRunner.fail_stages = [], set()
    s = sess.Session([str(d)], str(tmp_path / "out"))
    s.scan()
    with pytest.raises(RuntimeError, match="only 1 sub"):
        s._ensure_registered(None, 1.0)
    assert not any(st.startswith("calibrate") for st, _ in FakeRunner.log)


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

def test_status_carries_the_verdict_of_the_newest_export(session):
    """``post.export`` measures the gates, prints the table and files the report, and consumed none of it: a render
    that failed five of the twelve was delivered and listed exactly like one that failed none (C 20, 2026-10-31).
    The card now states what the pipeline itself measured about the picture it handed out."""
    d = os.path.join(session.dir, "exports")
    os.makedirs(d, exist_ok=True)
    assert session.status()["export_gates"] is None                     # nothing exported: nothing to report
    json.dump({"linear_info": {"gates": {"passed": False, "failed": ["sky_pixel_noise", "stars_count"],
                                         "sky_pixel_noise": {"value": 0.0172, "limit": 0.008, "pass": False}}}},
              open(os.path.join(d, "M42_20261031-101010.json"), "w"))
    g = session.status()["export_gates"]
    assert g["export"] == "M42_20261031-101010" and g["passed"] is False and g["error"] is None
    assert g["failed"] == ["sky_pixel_noise", "stars_count"]
    json.dump({"linear_info": {"gates": {"passed": True, "failed": []}}},
              open(os.path.join(d, "M42_20261031-111111.json"), "w"))
    assert session.status()["export_gates"]["passed"] is True           # the newest export is what the card says
    json.dump({"params": {}}, open(os.path.join(d, "M42_20261031-121212.json"), "w"))
    assert session.status()["export_gates"] is None                     # one that measured nothing says so


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


def test_two_x_request_is_stacked_at_two_x(session):
    session.run_analysis(1.0)
    meta = session.run_stack({"scale": 2})
    assert meta["scale"] == 2.0 and "2x drizzle as requested" in meta["resampling_reason"]


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
    from sirilpipe import post as real_post
    fake = types.ModuleType("sirilpipe.post")
    fake.deconv_choice = real_post.deconv_choice          # Session.run_denoise asks it whether ImageMM runs
    fake.restore = lambda *a: calls.append("restore") or {"ok": True}
    fake.remove_stars = lambda *a: calls.append("stars") or {"tool": "x"}
    fake.autofinish = lambda *a: calls.append("auto") or {"params": {"stretch": 0.2}, "report": {}}
    fake.export = lambda d, meta, params, opts, p, c: calls.append(("export", opts)) or {"jpg": "a.jpg", "size": [1, 1]}
    fake.preview_jpeg = lambda d, meta, params, max_side=1600, checkpoint=None: b"jpeg"
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
    """The folder is cleared at the start (a dead job's copy), the subs are copied when calibration stages them, the
    staging symlinks point at the copy during the job and back at the library afterwards, and the copy is deleted."""
    monkeypatch.setenv("ASTROPHOTO_LOCAL_COPY", "1")
    root = tmp_path / "local"
    monkeypatch.setenv("ASTROPHOTO_LOCAL_DIR", str(root))
    (root / "stale").mkdir(parents=True)
    (root / "stale" / "old.fit").write_bytes(b"x")
    msgs = []
    with session.local_copy(lambda i, n, m: msgs.append(m)) as s:
        assert not (root / "stale").exists() and s._local == {}
        s.run_analysis(1.0)
        assert len(s._local) == 12 and all(os.path.exists(p) and p.startswith(str(root)) for p in s._local.values())
        assert msgs[-1 - [m.startswith("Copying") for m in reversed(msgs)].index(True)].startswith(
            "Copying the subs to the local disk (12/12")
        links = sorted(glob.glob(os.path.join(s.process, "src", "*.fit")))
        assert links and all(os.readlink(p).startswith(str(root)) for p in links)
        assert s._local_paths([f.path for f in s.fs.frames[:2]]) == {f.path: s._local[f.path] for f in s.fs.frames[:2]}
    assert session._local == {} and not root.exists()
    lib = {f.path for f in session.fs.frames}
    assert all(os.readlink(p) in lib for p in links)
    msgs.clear()
    with session.local_copy(lambda i, n, m: msgs.append(m)):
        session.run_stack({})                       # the registration is reused: no sub is read, nothing copied
    assert not any(m.startswith("Copying") for m in msgs)

def test_a_second_job_leaves_a_running_jobs_local_copy_alone(session, monkeypatch, tmp_path):
    """The copy root is shared by every job on the output volume and used to be cleared at each job's entry, so a
    second job that started while a first was still copying deleted the first's copies under it and the first died
    renaming a ``.part`` file (border_census, 2026-10-31, exit 1, while an export was using the same root).  Each
    job now copies into a folder of its own, which the other neither reads nor removes."""
    monkeypatch.setenv("ASTROPHOTO_LOCAL_COPY", "1")
    root = tmp_path / "local"
    monkeypatch.setenv("ASTROPHOTO_LOCAL_DIR", str(root))
    other = sess.Session(session.folders, session.workdir)
    with session.local_copy() as a, other.local_copy() as b:      # b enters while a has copied nothing yet
        a.scan()
        paths = [f.path for f in a.fs.frames]
        a._local_paths(paths)                                     # this is where the first used to die
        b._local_paths(paths)
        assert len(a._local) == len(b._local) == len(paths)
        assert set(a._local.values()).isdisjoint(b._local.values())
        assert all(os.path.exists(p) for p in list(a._local.values()) + list(b._local.values()))
        assert os.path.basename(a._local_root).startswith(f"{os.path.basename(a.dir)}-{os.getpid()}-")
    assert not root.exists()                                      # both folders gone, then the empty root


def test_a_dead_jobs_copy_is_reaped_and_a_live_jobs_is_not(session, monkeypatch, tmp_path):
    """What the shared folder used to reclaim from every job at entry is now reclaimed only from jobs that are
    gone: a folder whose pid no longer answers, one too old to still be copying even if it does (a container
    restart recycles pids), and debris from when the root itself was the working folder.  A live job's is untouchable."""
    import subprocess
    import sys
    monkeypatch.setenv("ASTROPHOTO_LOCAL_COPY", "1")
    root = tmp_path / "local"
    monkeypatch.setenv("ASTROPHOTO_LOCAL_DIR", str(root))
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()                                                 # reaped: this pid answers to nobody now
    dead_pid = gone.pid
    for name in (f"M 42-{dead_pid}-deadbeef",                    # its job is gone
                 f"M 42-{os.getpid()}-cafebab0",                 # a job running now
                 f"M 42-{os.getpid()}-f00df00d",                 # answers, but from before a restart
                 "5f4dcc3b5aa7"):                                 # the old layout's hash folder
        d = root / name
        d.mkdir(parents=True)
        (d / "sub_00001.fit").write_bytes(b"x")
    os.utime(root / f"M 42-{os.getpid()}-f00df00d", (0, 0))
    with session.local_copy() as s:
        assert not (root / f"M 42-{dead_pid}-deadbeef").exists()
        assert not (root / "5f4dcc3b5aa7").exists()
        assert not (root / f"M 42-{os.getpid()}-f00df00d").exists()
        assert (root / f"M 42-{os.getpid()}-cafebab0" / "sub_00001.fit").exists()
        assert s._local_root.startswith(str(root)) and os.path.isdir(s._local_root)
    assert (root / f"M 42-{os.getpid()}-cafebab0").exists()           # the neighbour outlives our exit
    shutil.rmtree(root / f"M 42-{os.getpid()}-cafebab0")
    root.rmdir()
    assert not root.exists()


def test_a_reference_frame_left_out_by_the_selection_is_replaced_for_the_apply(session):
    """Ced 19l (2026-11-10): the registration's reference frame (identity homography) was rejected by the frame selection;
    seqapplyreg of the other frames aborted ("The reference image has a null matrix and was not previously aligned").
    The apply then sets the best included frame as its reference."""
    session.run_analysis(1.0)
    name = session.frames_table()[0]["name"]                    # frame 1: the fixture registration's reference
    session.set_override(name, False)
    session.run_stack({})
    apply = dict(FakeRunner.log)["apply_registration"]
    rows = session.frames_table()
    ref = next(int(l.split()[-1]) for l in apply.splitlines() if l.startswith("setref light "))
    assert "unselect light 1 1" in apply and ref != 1 and rows[ref - 1]["included"]
    assert apply.index("setref") < apply.index("unselect")
    session.set_override(name, None)                           # reference included again: no setref
    FakeRunner.log = []
    session.run_stack({})
    assert "setref" not in dict(FakeRunner.log)["apply_registration"]


def test_local_copy_off_and_unsafe_dir(session, monkeypatch, tmp_path):
    monkeypatch.delenv("ASTROPHOTO_LOCAL_COPY", raising=False)
    with session.local_copy() as s:
        assert s._local_root is None
    monkeypatch.setenv("ASTROPHOTO_LOCAL_COPY", "true")
    monkeypatch.setenv("ASTROPHOTO_LOCAL_DIR", session.workdir)
    with pytest.raises(RuntimeError):
        with session.local_copy():
            pass
    assert os.path.isdir(session.dir)


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


def test_per_frame_background_runs_before_registration_and_is_a_choice(session):
    """local_norm (default on): Siril seqsubsky 1 on each Bayer plane of the calibrated subs, before the registration; off, the stage does not
    run, and the change re-registers (the signature carries it)."""
    session.run_analysis(1.0)
    stages = [s for s, _ in FakeRunner.log]
    assert stages.index("calibrate_1") < stages.index("frame_bkg_1") < stages.index("register")
    assert "seqsubsky CFA0_light 1" in dict(FakeRunner.log)["frame_bkg_1"]
    st = json.load(open(os.path.join(session.process, "state.json")))
    assert st["frame_bkg_requested"] is True and st["signature"]["frame_bkg"] == 1
    FakeRunner.log.clear()
    meta = session.run_stack({"local_norm": False})
    stages = [s for s, _ in FakeRunner.log]
    assert "register" in stages and not [s for s in stages if s.startswith("frame_bkg")]
    assert meta["local_norm"] is False


def test_previews_of_a_dataset_never_overlap_and_a_superseded_one_is_skipped(session, monkeypatch):
    """Two renders at once shared process/preview and overwrote each other's Siril inputs (an error at a random stage for every quick slider move)."""
    import threading
    import time as _time
    from sirilpipe import post
    active, overlap = [0], [False]

    def fake_preview(d, meta, params, max_side=1600, checkpoint=None):
        active[0] += 1
        overlap[0] |= active[0] > 1
        _time.sleep(0.05)
        active[0] -= 1
        return b"jpeg"
    monkeypatch.setattr(post, "preview_jpeg", fake_preview)
    ts = [threading.Thread(target=session.preview_jpeg, args=({"saturation": 1 + i / 10},)) for i in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not overlap[0]
    with pytest.raises(sess.PreviewSuperseded):
        session.preview_jpeg({}, superseded=lambda: True)
    assert session.preview_jpeg({}, superseded=lambda: False) == b"jpeg"


def test_a_preview_that_is_rendering_stops_when_a_newer_one_is_asked_for(session, monkeypatch):
    """The render polls ``superseded`` through the chain's checkpoints: a slider moved while a 40 s render runs starts the new render at once."""
    from sirilpipe import post
    polls = []

    def fake_preview(d, meta, params, max_side=1600, checkpoint=None):
        for _ in range(5):
            checkpoint()
            polls.append(1)
        return b"jpeg"
    monkeypatch.setattr(post, "preview_jpeg", fake_preview)
    newer = iter([False, False, False, True])
    with pytest.raises(sess.PreviewSuperseded):
        session.preview_jpeg({}, superseded=lambda: next(newer))
    assert len(polls) == 2
    assert issubclass(sess.PreviewSuperseded, sess.Cancelled)
