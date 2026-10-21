"""Session: the object the web server / v1 API / CLI drive (DESIGN section 2).

One Session = one dataset (one folder, or several night folders of the same target) with its cache
directory ``<workdir>/<slug>/`` (DESIGN section 3).  Siril does the pre-processing; the post-stack
stages (restore, star removal, auto-finish, export, preview) live in ``sirilpipe.post`` and are imported
lazily so this module works (and is testable) without it.

Progress: every long step calls ``progress(i, n, msg)``; the stage names are DESIGN section 2's list.
Cancel / pause: cooperative through ``checkpoint()`` (called by every progress callback and by the Siril
runner while it streams output; it kills the child process through ``Cancelled``).
"""
from __future__ import annotations

import contextlib
import glob
import hashlib
import logging
import os
import shutil
import statistics
import threading
import time
from datetime import datetime

import numpy as np

from . import __version__, calibration, dataset, fitsio, stackpost, workflow
from .dataset import Frame, FrameSet
from .siril import SirilError, SirilRunner, parse_seq
from .util import (Cancelled, clean_json, dataset_slug, read_json, restoration_done, slugify, split_folders,  # noqa: F401
                   write_json)

log = logging.getLogger("sirilpipe")

SESSION_FORMAT = 1
SWAP_NAME = "stack_original.fits"        # stack.fits while post.restore reads imagemm.fits in its place
_IMAGEMM_WARNED = False                  # log "ImageMM unavailable" once per process
STAGES = ["Scanning frames", "Calibrating", "Registering (pass 1/2)", "Registering (pass 2/2)",
          "Applying registration", "Stacking", "Cropping", "Plate solving"]


def local_copy_enabled() -> bool:
    """``ASTROPHOTO_LOCAL_COPY``: jobs read the subs from a local copy (Session.local_copy)."""
    return os.environ.get("ASTROPHOTO_LOCAL_COPY", "").strip().lower() in ("1", "true", "yes", "on")


def local_copy_dir(workdir: str) -> str:
    """``ASTROPHOTO_LOCAL_DIR``, else ``<workdir>/.local_copy``."""
    return os.path.abspath(os.environ.get("ASTROPHOTO_LOCAL_DIR") or os.path.join(workdir, ".local_copy"))


class Session:
    """All state for one dataset.  Thread-safe for the server's pattern (one job at a time per dataset;
    status/frames reads from request threads)."""

    def __init__(self, folders: str | list[str], workdir: str = "output", limit: int | None = None):
        self.folders = split_folders(folders)
        if not self.folders:
            raise ValueError("no dataset folder given")
        self.folder = os.pathsep.join(self.folders)
        self.workdir = os.path.abspath(workdir)
        self.dir = os.path.join(self.workdir, dataset_slug(self.folders))
        os.makedirs(self.dir, exist_ok=True)
        self.process = os.path.join(self.dir, "process")
        self.calib_dir = os.path.join(self.dir, "calib")
        self.limit = limit                       # development option: only the first N subs
        self.fs: FrameSet | None = None
        self.infos: list[Frame] = []
        self.analysis: dict | None = read_json(self._p("analysis.json"))
        self.overrides: dict[str, bool] = read_json(self._p("overrides.json"), {}) or {}
        self.calib_report: dict | None = None
        self.calib_candidates: list = []
        self.calib_error: str | None = None
        self.lock = threading.RLock()
        self.cancel_flag = threading.Event()
        self.pause_flag = threading.Event()
        self._local: dict[str, str] = {}         # original path -> local copy while local_copy() is active
        self.timings: dict[str, float] = read_json(os.path.join(self.process, "timings.json"), {}) or {}
        self._rows: list[dict] | None = None
        self._units: list[workflow.Unit] = []

    # ------------------------------------------------------------------ helpers
    def _p(self, *names: str) -> str:
        return os.path.join(self.dir, *names)

    @property
    def cancel_requested(self) -> bool:
        return self.cancel_flag.is_set()

    @property
    def meta(self) -> dict:
        return read_json(self._p("stack_meta.json"), {}) or {}

    def checkpoint(self) -> bool:
        """Called between units of work: waits while paused, raises ``Cancelled`` once cancelled."""
        while self.pause_flag.is_set() and not self.cancel_flag.is_set():
            time.sleep(0.25)
        if self.cancel_flag.is_set():
            raise Cancelled()
        return False

    def _cb(self, progress):
        """A progress callback that also polls cancel/pause (the server relies on that)."""
        def f(i, n, msg):
            if progress:
                progress(i, n, msg)
            self.checkpoint()
        return f

    def _runner(self, progress) -> SirilRunner:
        os.makedirs(self.process, exist_ok=True)
        return SirilRunner(self.process, progress=self._cb(progress), checkpoint=self.checkpoint)

    def _time(self, stage: str, seconds: float) -> None:
        self.timings[stage] = round(seconds, 1)
        try:
            write_json(os.path.join(self.process, "timings.json"), self.timings)
        except OSError:
            pass

    def _path(self, f: Frame) -> str:
        return self._local.get(f.path, f.path)

    # ------------------------------------------------------------------ scanning / calibration
    @property
    def calib_prefs(self) -> dict:
        return {**calibration.DEFAULT_PREFS, **(read_json(self._p("calib_prefs.json"), {}) or {})}

    def set_calib_prefs(self, prefs: dict) -> dict:
        new = {**self.calib_prefs, **{k: v for k, v in (prefs or {}).items() if k in calibration.DEFAULT_PREFS}}
        write_json(self._p("calib_prefs.json"), new)
        return new

    def scan(self, progress=None) -> list[Frame]:
        """Read the headers, classify, and choose calibration masters."""
        with self.lock:
            p = self._cb(progress)
            p(0, 1, "Scanning frames")
            fs = dataset.scan(self.folders, limit=self.limit, progress=p)
            if not fs:
                raise RuntimeError(f"No light frames (FITS) found in {', '.join(self.folders)}")
            self.fs = fs
            self.infos = fs.frames
            self.calib_error = None
            self.calib_report, self.calib_candidates = calibration.find_and_choose(fs, self.calib_prefs)
            note = next((n for n in self.calib_report.get("notes", []) if "unreadable" in n), None)
            self.calib_error = note
            p(1, 1, f"Scanning frames ({len(fs.frames)} subs)")
            return self.infos

    def _calib_signature(self, report: dict | None):
        if not report:
            return None
        return [(report.get(k) or {}).get("path") for k in ("dark", "flat", "bias")]

    def run_calibration(self, progress=None) -> dict:
        """Find the masters, build the ones that come as raw frames, record the choice.  ``changed_analysis``
        is True when the chosen masters differ from the ones the registration was made with (the registration
        is then stale and dropped)."""
        with self.lock:
            before = read_json(self._p("calibration.json"), {}) or {}
            self.scan(progress)
            units = self._make_units(debayer=False)
            self._ensure_masters(units, progress)
            after = self._calib_signature(self.calib_report)
            changed = bool(self._rows is not None or os.path.exists(self._p("frames.json"))) and \
                self._calib_signature(before.get("report")) != after
            if changed:
                self._drop_analysis()
            info = self.calibration_info(with_candidates=False)
            write_json(self._p("calibration.json"), info)
            return {"changed_analysis": changed, "summary": (self.calib_report or {}).get("summary")}

    def _drop_analysis(self) -> None:
        self.analysis = None
        self._rows = None
        for f in ("frames.json", "analysis.json"):
            if os.path.exists(self._p(f)):
                os.remove(self._p(f))
        st = os.path.join(self.process, "state.json")
        if os.path.exists(st):
            os.remove(st)

    def calibration_info(self, with_candidates: bool = True) -> dict:
        """What the calibration panel shows: the rig, the lights' settings, the masters chosen and the ones that
        would fit (``candidates``)."""
        if self.fs is None:
            self.scan()
        fs, f0 = self.fs, self.fs.first
        profile = {"telescope": f0.telescope or f0.instrument or "", "instrument": f0.instrument,
                   "sensor": fs.sensor, "bayer": f0.bayer, "focallen": f0.focallen, "pixsize": f0.pixsize,
                   "width": f0.width, "height": f0.height, "camera_slot": f0.camera_slot, "bit_depth": 16,
                   "device_rejected": sum(1 for f in fs.frames + [f for g in fs.exposure_groups for f in g.frames] if f.device_rejected),
                   "kind": fs.kind}
        out = {"profile": profile, "n_lights": len(fs.frames), "prefs": self.calib_prefs,
               "env": os.environ.get("ASTROPHOTO_CALIB", ""), "error": self.calib_error,
               "report": self.calib_report}
        if with_candidates:
            out["candidates"] = calibration.candidate_listing(self.calib_candidates)
        return clean_json(out)

    def set_calibration(self, prefs: dict) -> dict:
        """Alias of ``set_calib_prefs`` (DESIGN section 2 name)."""
        return self.set_calib_prefs(prefs)

    # ------------------------------------------------------------------ units & masters
    def _hdr_wanted(self, stack_params: dict | None = None) -> bool:
        """HDR (several exposure times of the same target) unless switched off with ``hdr: false``."""
        sp = stack_params or {}
        return bool(self.fs and self.fs.exposure_groups and sp.get("hdr", True) not in (False, 0, "false", "off"))

    def _make_units(self, debayer: bool, hdr: bool = False) -> list[workflow.Unit]:
        """One unit per (exposure group x night folder), each with its own calibration masters (a 60 s dark
        for the 60 s subs).  A single unit works in the process dir itself."""
        fs = self.fs
        groups = fs.hdr_groups() if hdr else [None]
        spec: list[tuple[int, str, list[Frame]]] = []
        for gi, g in enumerate(groups):
            frames = g.frames if g is not None else fs.frames
            for folder in fs.folders:
                frs = [f for f in frames if f.folder == folder]
                if frs:
                    spec.append((gi, folder, frs))
        # Siril makes no sequence of a single image ("No sequence `light' found" in calibrate): a night folder that
        # holds one sub of a group (the DWARF rejected the rest as failed_) cannot be calibrated as a unit of its
        # own.  It is left out when other units carry the stack; a dataset of one sub fails in _ensure_registered.
        self.dropped_subs = []
        if len(spec) > 1:
            keep = [s for s in spec if len(s[2]) >= 2]
            if keep:
                self.dropped_subs = [f.name for s in spec if len(s[2]) < 2 for f in s[2]]
                if self.dropped_subs:
                    log.warning("leaving out %d sub(s) alone in their night folder (Siril cannot calibrate a "
                                "one-image sequence): %s", len(self.dropped_subs), ", ".join(self.dropped_subs))
                spec = keep
        units: list[workflow.Unit] = []
        for k, (gi, folder, frs) in enumerate(spec):
            single = len(spec) == 1
            d = self.process if single else os.path.join(self.process, f"u{k + 1}")
            if single:
                rep = self.calib_report or {}
            else:
                sub = FrameSet(folders=[folder], frames=frs, kind=fs.kind, sensor=fs.sensor,
                               calibration_files=fs.calibration_files)
                rep = calibration.choose(sub, self.calib_candidates, self.calib_prefs)
            units.append(workflow.Unit(index=k, dir=d, frames=frs, calib=rep, group=gi))
        return units

    def _ensure_masters(self, units: list[workflow.Unit], progress=None) -> None:
        """Build raw-frame masters through Siril (cached by name = hash of the frames) and calibrate DWARF-style
        flat masters with the bias.  Fills ``unit.masters`` with the file names to pass to Siril."""
        runner = None
        for u in units:
            rep = u.calib or {}
            if not rep.get("enabled", True):
                continue
            paths: dict[str, str] = {}
            for kind in ("bias", "dark", "flat"):
                c = rep.get(kind)
                if not c:
                    continue
                if c.get("raw_frames"):
                    target = os.path.join(self.calib_dir, c["master_name"])
                    if not os.path.exists(target):
                        runner = runner or self._runner(progress)
                        work = os.path.join(self.calib_dir, "work_" + c["master_name"].rsplit(".", 1)[0])
                        shutil.rmtree(work, ignore_errors=True)
                        os.makedirs(os.path.join(work, "src"))
                        for i, src in enumerate(c["raw_frames"], 1):
                            os.symlink(src, os.path.join(work, "src", f"{kind}_{i:05d}.fit"))
                        bias = paths.get("bias") if kind == "flat" else None
                        t0 = time.time()
                        runner.run(workflow.master_script(kind, work, target[:-4], bias),
                                   f"master_{kind}", expect=[target],
                                   label=f"Calibrating: master {kind} ({len(c['raw_frames'])} frames)")
                        self._time(f"master_{kind}", time.time() - t0)
                        shutil.rmtree(work, ignore_errors=True)
                    paths[kind] = target
                else:
                    paths[kind] = c["path"]
            fb = rep.get("flat_bias")
            if paths.get("flat") and paths.get("bias") and fb and not (rep["flat"] or {}).get("raw_frames"):
                # an already-stacked flat master still carries the bias pedestal
                stem = os.path.splitext(os.path.basename(paths["flat"]))[0]
                bstem = os.path.splitext(os.path.basename(paths["bias"]))[0]
                prefix = f"cal_{hashlib.sha1(bstem.encode()).hexdigest()[:6]}_"
                out = os.path.join(self.calib_dir, f"{prefix}{stem}.fit")
                if not os.path.exists(out):
                    os.makedirs(self.calib_dir, exist_ok=True)
                    runner = runner or self._runner(progress)
                    runner.run(workflow.flat_bias_script(self.calib_dir, paths["flat"], paths["bias"], prefix),
                               "flat_bias", expect=[out], label="Calibrating: flat master")
                paths["flat"] = out
            if paths.get("dark"):
                paths.pop("bias", None)               # never bias + dark on lights (it would be subtracted twice)
            u.masters = paths

    # ------------------------------------------------------------------ registration / analysis
    @staticmethod
    def _banding_param(sp: dict | None) -> bool | None:
        """The stack parameter (``pattern_correction``, the API contract's key; ``banding`` as an alias): None = not given."""
        sp = sp or {}
        v = sp.get("pattern_correction", sp.get("banding"))
        return None if v is None else v not in (False, 0, "false", "off")

    def _banding_on(self, units: list[workflow.Unit], debayer: bool, banding: bool) -> str | None:
        """Bayer pattern to de-band with (None: switched off, demosaic path, or not a Bayer mosaic: mono data are not touched)."""
        pat = (units[0].frames[0].bayer or "").upper() if units and units[0].frames else ""
        return pat if banding and not debayer and pat in workflow.BANDING_PATTERNS else None

    def _band_unit(self, u: workflow.Unit, seq: str, pattern: str, progress) -> None:
        """Row-noise removal on the calibrated CFA subs of one unit (workflow.BANDING_*): split into Bayer planes, seqfixbanding each, merge back,
        install over ``<seq>_NNNNN.fit``.  Intermediates are deleted stage by stage (peak: 3 x the calibrated subs)."""
        n = len(u.frames)
        r = SirilRunner(self.process, progress=self._cb(progress), checkpoint=self.checkpoint)
        r.run(workflow.banding_split_script(u, seq), f"banding_split_{u.index + 1}", expect=[os.path.join(u.dir, f"CFA3_{seq}_{n:05d}.fit")],
              count_glob=os.path.relpath(os.path.join(u.dir, f"CFA3_{seq}_[0-9]*.fit"), self.process), total=n, label="Banding (splitting the Bayer planes)")
        r.run(workflow.banding_fix_script(u, seq), f"banding_fix_{u.index + 1}", expect=[os.path.join(u.dir, f"unband_CFA3_{seq}_{n:05d}.fit")],
              label="Banding (Siril seqfixbanding)")
        workflow.banding_cleanup(u.dir, seq, ("CFA",))
        r.run(workflow.banding_merge_script(u, seq, pattern), f"banding_merge_{u.index + 1}", expect=[os.path.join(u.dir, f"mCFA_unband_CFA0_{seq}_{n:05d}.fit")],
              label="Banding (merging the Bayer planes)")
        workflow.banding_cleanup(u.dir, seq, ("unband_CFA",))
        if workflow.banding_install(u.dir, seq) != n:
            raise SirilError(f"banding: {n} subs expected in {u.dir}")
        workflow.banding_cleanup(u.dir, seq, ("mCFA_unband_CFA",))

    def _signature(self, units: list[workflow.Unit], debayer: bool, banding: str | None = None) -> dict:
        h = hashlib.sha1()
        for u in units:
            for f in u.frames:
                try:
                    st = os.stat(f.path)
                    h.update(f"{f.path}:{st.st_size}:{int(st.st_mtime)}".encode())
                except OSError:
                    h.update(f.path.encode())
            for k in sorted(u.masters):
                p = u.masters[k]
                h.update(f"{k}:{p}:{os.path.getmtime(p) if os.path.exists(p) else 0}".encode())
        h.update(("|".join(f"g{u.group}" for u in units)).encode())
        return {"format": SESSION_FORMAT, "frames": h.hexdigest(), "debayer": debayer,
                "register": workflow.REGISTER_ARGS, "n": sum(len(u.frames) for u in units),
                "banding": [workflow.BANDING_AMOUNT, workflow.BANDING_SIGMA, banding] if banding else None}

    def _ordered_frames(self, units: list[workflow.Unit]) -> list[Frame]:
        return [f for u in units for f in u.frames]

    def _debayer(self, stack_params: dict | None) -> bool:
        return str((stack_params or {}).get("mode", "auto")).lower() == "demosaic"

    def _ensure_registered(self, progress, sensitivity: float, debayer: bool = False, force: bool = False,
                           hdr: bool | None = None, banding: bool | None = None) -> dict:
        """Calibrate + 2-pass registration unless the process dir already holds a registration of exactly
        these inputs.  Returns the state dict."""
        if self.fs is None:
            self.scan(progress)
        hdr = self._hdr_wanted() if hdr is None else hdr
        units = self._make_units(debayer, hdr)
        n_subs = sum(len(u.frames) for u in units)
        if n_subs < dataset.MIN_STACK_SUBS:
            # (Siril would fail on it with "No sequence `light' found": it makes no sequence of a single image)
            raise RuntimeError(f"only {n_subs} sub(s) in this dataset; at least {dataset.MIN_STACK_SUBS} are needed to "
                               f"register and stack")
        self._ensure_masters(units, progress)
        st_path = os.path.join(self.process, "state.json")
        state = read_json(st_path, {}) or {}
        # banding=None (analysis): keep what the registered state was made with (default on), so analysing never re-registers
        band = self._banding_on(units, debayer, state.get("banding_requested", True) if banding is None else banding)
        sig = self._signature(units, debayer, band)
        seq = workflow.seq_name(units, debayer)
        seqfile = os.path.join(self.process, f"{seq}_.seq")
        if not force and state.get("signature") == sig and os.path.exists(seqfile) and \
                parse_seq(seqfile).registered_layer is not None:
            self._units = units
            rows = self._load_rows()
            if rows is None or [r["name"] for r in rows] != [f.name for f in self._ordered_frames(units)]:
                self._build_rows(parse_seq(seqfile), units, sensitivity)      # frames.json lost or stale
            return state
        runner = self._runner(progress)
        n = sum(len(u.frames) for u in units)
        workflow.clean_process_dir(self.process)
        for u in units:
            if u.dir != self.process:
                shutil.rmtree(u.dir, ignore_errors=True)
        t0 = time.time()
        done = 0
        for u in units:
            workflow.stage_inputs(u, [self._path(f) for f in u.frames])
            pre = seq
            off = done

            def prog(i, n_, msg, off=off, nu=len(u.frames)):        # aggregate the counter over several units
                if progress and len(units) > 1 and n_ == nu:
                    progress(off + i, n, f"Calibrating ({off + i}/{n})")
                elif progress:
                    progress(i, n_, msg)
            r = SirilRunner(self.process, progress=self._cb(prog), checkpoint=self.checkpoint)
            r.run(workflow.calibrate_script(u, debayer, force=(seq == "pp_light")), f"calibrate_{u.index + 1}", expect=[os.path.join(u.dir, f"{pre}_00001.fit")],
                  count_glob=os.path.relpath(os.path.join(u.dir, f"{pre}_[0-9]*.fit"), self.process),
                  total=len(u.frames), label="Calibrating")
            done += len(u.frames)
        self._time("calibrate", time.time() - t0)
        band_info = None
        if band:
            t0 = time.time()
            # measured first (calibrated subs, the same numbers as the report): Seestar M 27 (IMX585) measures 0.98 and keeps its 16-bit subs untouched
            meas = [workflow.row_banding_ratio(sorted(glob.glob(os.path.join(u.dir, f"{seq}_[0-9]*.fit"))), band) for u in units]
            band_info = {"measured": meas, "applied": any(m["banded"] for m in meas)}
            for u, m in zip(units, meas):
                if m["banded"]:
                    self._band_unit(u, seq, band, progress)
            self._time("banding", time.time() - t0)
        if len(units) > 1:
            runner.run(workflow.merge_script(units, self.process, seq), "merge", expect=[f"{seq}_.seq"],
                       label="Calibrating (merging nights)")
        t0 = time.time()
        runner.run(workflow.register_script(self.process, seq, n), "register", expect=[f"{seq}_.seq"],
                   label="Registering (pass 1/2)", markers=[(r"Trial #|choosing image|Matching stars", "Registering (pass 2/2)")])
        self._time("register", time.time() - t0)
        state = {"signature": sig, "seq": seq, "n": n, "debayer": debayer, "hdr": hdr, "created": time.time(), "banding_requested": band is not None,
                 "left_out_single_subs": list(getattr(self, "dropped_subs", []) or []),
                 "banding": {"command": f"seqfixbanding <CFA plane> {workflow.BANDING_AMOUNT:g} {workflow.BANDING_SIGMA:g}", "amount": workflow.BANDING_AMOUNT,
                             "sigma": workflow.BANDING_SIGMA, "pattern": band, "sequence": seq, **band_info} if band else None}
        seqd = parse_seq(seqfile)
        if len(seqd.frames) != n:
            raise SirilError(f"registration produced {len(seqd.frames)} frames for {n} subs (see {self.process}/logs)")
        write_json(st_path, state)
        self._units = units
        self._build_rows(seqd, units, sensitivity)
        return state

    def _build_rows(self, seqd, units: list[workflow.Unit], sensitivity: float) -> None:
        """frames.json from the registration data + selection rules + user overrides."""
        frames = self._ordered_frames(units)
        groups = [u.group for u in units for _ in u.frames]
        stats = []
        for f, sf in zip(frames, seqd.frames):
            stats.append({"registered": bool(sf.included) and sf.fwhm is not None and sf.H is not None,
                          "nbstars": sf.nbstars, "roundness": sf.roundness, "wfwhm": sf.wfwhm, "fwhm": sf.fwhm,
                          "bkg": sf.background, "quality": sf.quality, "H": sf.H})
        rows = []
        for i, (f, s) in enumerate(zip(frames, stats)):
            rows.append({"idx": i, "group": groups[i], "name": f.name, "path": f.path, "folder": f.folder, "time": f.date_obs, "date": f.date_obs,
                         "exptime": f.exptime, "gain": f.gain, "temp": f.temp,
                         "registered": s["registered"], "fwhm": s["fwhm"], "wfwhm": s["wfwhm"],
                         "roundness": s["roundness"], "elongation": (1.0 / s["roundness"]) if s["roundness"] else None,
                         "bkg": s["bkg"], "background": s["bkg"], "nbstars": s["nbstars"], "n_stars": s["nbstars"],
                         "quality": s["quality"], "H": s["H"], "is_reference": i == max(seqd.reference, 0),
                         "device_rejected": bool(f.device_rejected), "transparency": None, "obstructed": None, "reg_rms": None,
                         "anomaly": None, "tile_mask": None, "noise": None})
        self._rows = rows
        self._apply_selection(sensitivity)
        ok = [r for r in rows if r["registered"] and r["fwhm"]]
        self.analysis = {"sensitivity": sensitivity, "n": len(rows), "created": time.time(),
                         "medians": {"fwhm": statistics.median([r["fwhm"] for r in ok]) if ok else None,
                                     "wfwhm": statistics.median([r["wfwhm"] for r in ok]) if ok else None,
                                     "roundness": statistics.median([r["roundness"] for r in ok if r["roundness"]]) if ok else None,
                                     "nbstars": statistics.median([r["nbstars"] for r in ok if r["nbstars"] is not None]) if ok else None,
                                     "background": statistics.median([r["bkg"] for r in ok if r["bkg"] is not None]) if ok else None}}
        write_json(self._p("analysis.json"), self.analysis)

    def _apply_selection(self, sensitivity: float) -> None:
        rows = self._rows or []
        reasons: list[list[str]] = [[] for _ in rows]
        for g in sorted({r.get("group", 0) for r in rows}):         # wFWHM ranks are only comparable within an exposure
            idx = [i for i, r in enumerate(rows) if r.get("group", 0) == g]
            for i, why in zip(idx, workflow.select_frames([rows[i] for i in idx], sensitivity)):
                reasons[i] = why
        wmin = min([r["wfwhm"] for r in rows if r.get("wfwhm")] or [1.0])
        for r, why in zip(rows, reasons):
            r["auto_accepted"] = not why
            r["overridden"] = r["name"] in self.overrides
            acc = (not why)
            if r["overridden"]:
                acc = bool(self.overrides[r["name"]]) and r["registered"]
            r["accepted"] = r["included"] = acc
            r["reasons"] = why
            r["reason"] = "; ".join(why) if why else ("accepted by hand" if r["overridden"] and acc else None)
            r["weight"] = (wmin / r["wfwhm"]) if (acc and r.get("wfwhm")) else 0.0
        write_json(self._p("frames.json"), [{k: v for k, v in r.items() if k != "H"} for r in rows])

    def _load_rows(self) -> list[dict] | None:
        """Rows from memory, else frames.json (+ the homographies from the .seq when present)."""
        if self._rows is None:
            rows = read_json(self._p("frames.json"))
            if rows is None:
                return None
            st = read_json(os.path.join(self.process, "state.json"), {}) or {}
            seqfile = os.path.join(self.process, f"{st.get('seq', 'pp_light')}_.seq")
            if os.path.exists(seqfile):
                sq = parse_seq(seqfile)
                if len(sq.frames) == len(rows):
                    for r, sf in zip(rows, sq.frames):
                        r["H"] = sf.H
            self._rows = rows
        return self._rows

    def run_analysis(self, sensitivity: float = 1.0, progress=None, stack_params: dict | None = None) -> dict:
        """Scan, calibrate, register (2-pass) and measure every sub: ``frames.json`` with fwhm / wfwhm /
        roundness / background / nbstars and the ``included`` flag; stops before any resampling."""
        with self.lock:
            t0 = time.time()
            self.scan(progress)
            self._rows = None
            self._ensure_registered(progress, float(sensitivity), banding=self._banding_param(stack_params))
            self._load_rows()
            self._apply_selection(float(sensitivity))
            self.analysis = {**(self.analysis or read_json(self._p("analysis.json"), {}) or {}),
                             "sensitivity": float(sensitivity)}
            write_json(self._p("analysis.json"), self.analysis)
            self._time("analysis", time.time() - t0)
            return {"n": len(self.infos)}

    def reselect(self, sensitivity: float) -> list[dict]:
        """Re-run the selection rules with another ``sensitivity`` (no re-measurement)."""
        with self.lock:
            if self._load_rows() is None:
                return []
            self._apply_selection(float(sensitivity))
            self.analysis = {**(self.analysis or {}), "sensitivity": float(sensitivity)}
            write_json(self._p("analysis.json"), self.analysis)
            return self.frames_table()

    def set_override(self, name: str, accepted: bool | None) -> list[dict]:
        """Force a sub in (True) or out (False); None clears the override.  Reaches Siril as select/unselect."""
        with self.lock:
            if accepted is None:
                self.overrides.pop(name, None)
            else:
                self.overrides[name] = bool(accepted)
            write_json(self._p("overrides.json"), self.overrides)
            if self._load_rows() is not None:
                self._apply_selection((self.analysis or {}).get("sensitivity", 1.0))
            return self.frames_table()

    def frames_table(self) -> list[dict]:
        rows = self._load_rows()
        return clean_json([{k: v for k, v in r.items() if k != "H"} for r in rows]) if rows else []

    def frames(self) -> dict:
        """DESIGN section 2: the frames payload (rows, medians, sensitivity)."""
        a = self.analysis or {}
        return clean_json({"frames": self.frames_table(), "medians": a.get("medians"),
                           "sensitivity": a.get("sensitivity", 1.0)})

    # ------------------------------------------------------------------ thumbnails
    def frame_thumbnail(self, name: str, size: int = 360, out_path: str | None = None) -> bytes:
        """JPEG of one raw sub: 2x2 super-pixel debayer + autostretch (cached)."""
        import cv2
        from astropy.io import fits
        f = next((x for x in self.infos if x.name == name), None)
        if f is None:
            raise KeyError(name)
        cache = out_path or self._p("thumbs", f"{name}_{size}.jpg")
        if os.path.exists(cache):
            return open(cache, "rb").read()
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        raw = np.asarray(fits.getdata(self._path(f)), np.float32)
        img = _superpixel(raw, f.bayer)
        img = _autostretch(img)
        h, w = img.shape[:2]
        s = size / max(h, w)
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", (img[..., ::-1] * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 85])
        with open(cache, "wb") as fh:
            fh.write(buf.tobytes())
        return buf.tobytes()

    # ------------------------------------------------------------------ stack
    def run_stack(self, stack_params: dict | None = None, progress=None) -> dict:
        """Apply the registration (Bayer drizzle), stack, crop by coverage -> ``stack.fits`` + ``stack_meta.json``."""
        with self.lock:
            sp = dict(stack_params or {})
            sens = float(sp.get("sensitivity", 1.0) or 1.0)
            debayer = self._debayer(sp)
            t_all = time.time()
            if self.fs is None:
                self.scan(progress)
            state = self._ensure_registered(progress, sens, debayer, banding=self._banding_param(sp))
            rows = self._load_rows()
            if (self.analysis or {}).get("sensitivity") != sens:
                self._apply_selection(sens)
                self.analysis = {**(self.analysis or {}), "sensitivity": sens}
                write_json(self._p("analysis.json"), self.analysis)
            seq, n = state["seq"], state["n"]
            inc = [i for i, r in enumerate(rows) if r["included"]]
            if len(inc) < 3:
                why: dict[str, int] = {}
                for r in rows:
                    if not r["included"]:
                        k = (r.get("reasons") or ["excluded by override"])[0].split(" (")[0]
                        why[k] = why.get(k, 0) + 1
                raise RuntimeError(f"only {len(inc)} usable frames of {len(rows)} after selection; need at least 3 ("
                                   + "; ".join(f"{v} x {k}" for k, v in sorted(why.items(), key=lambda kv: -kv[1])) + ")")
            hdr_dropped = None
            if state.get("hdr"):
                # an exposure group is stacked on its own and needs 3+ usable frames; when fewer than two groups have
                # them there is nothing to blend: the stack is the viable group's alone (Sh2-132: of the 60 s night's
                # six subs, most of them live-stack rejects, too few registered), the other frames are left out
                n_of: dict[int, int] = {}
                for i in inc:
                    n_of[rows[i].get("group", 0)] = n_of.get(rows[i].get("group", 0), 0) + 1
                viable = [g for g, c in n_of.items() if c >= 3]
                if len(viable) < 2 and viable:
                    keep_g = max(viable, key=lambda g: n_of[g])
                    hdr_dropped = [{"group": g, "exptime": rows[next(i for i in inc if rows[i].get("group", 0) == g)]["exptime"],
                                    "n": c, "why": "fewer than 3 usable frames: no HDR blend"} for g, c in n_of.items() if g != keep_g]
                    inc = [i for i in inc if rows[i].get("group", 0) == keep_g]
                    log.warning("HDR: only one exposure group has 3+ usable frames; stacking it alone (%s left out)", hdr_dropped)
                if len(inc) < 3:
                    raise RuntimeError(f"no exposure group has 3 usable frames after selection")
            inc_set = set(inc)
            excluded = [i for i in range(len(rows)) if i not in inc_set]
            medf = statistics.median([rows[i]["fwhm"] for i in inc if rows[i].get("fwhm")] or [0.0]) or None
            rs = workflow.decide_resampling(sp, len(inc), medf, cfa=True)
            rej = workflow.rejection(sp, rs)
            runner = self._runner(progress)
            workflow.clean_registered(self.process)
            t0 = time.time()
            runner.run(workflow.apply_script(self.process, seq, n, excluded, rs), "apply_registration",
                       expect=[f"r_{seq}_.seq"], count_glob=f"r_{seq}_[0-9]*.fit", total=len(inc),
                       label="Applying registration")
            if workflow.unstackable_max_framing(os.path.join(self.process, f"r_{seq}_.seq")):
                # every frame at offset 0 of the max canvas but one a row short: Siril cannot stack that (see
                # workflow.unstackable_max_framing); the reference frame's framing is the same grid at one size
                log.info("registered frames all at offset 0 with differing sizes: applying again with -framing=current")
                rs = {**rs, "framing": "current"}
                workflow.clean_registered(self.process)
                runner.run(workflow.apply_script(self.process, seq, n, excluded, rs), "apply_registration",
                           expect=[f"r_{seq}_.seq"], count_glob=f"r_{seq}_[0-9]*.fit", total=len(inc),
                           label="Applying registration")
            self._time("apply_registration", time.time() - t0)
            t0 = time.time()
            groups = sorted({rows[i].get("group", 0) for i in inc})
            hdr_info = None
            if state.get("hdr") and len(groups) > 1:
                arr, hdr, hdr_info = self._stack_hdr(runner, seq, rows, inc, groups, rs, rej, progress)
            else:
                runner.run(workflow.stack_script(self.process, seq, rej, len(inc)), "stack", expect=["stack_raw.fit"],
                           label="Stacking")
                arr, hdr = fitsio.read_rgb(os.path.join(self.process, "stack_raw.fit"))
            self._time("stack", time.time() - t0)
            meta = self._finish_stack(arr, hdr, sp, rs, rej, inc, rows, medf, progress, hdr_info)
            if hdr_dropped:
                meta["hdr_left_out"] = hdr_dropped
            workflow.clean_registered(self.process)
            self._time("run_stack_total", time.time() - t_all)
            meta["timings"] = dict(self.timings)
            write_json(self._p("stack_meta.json"), meta)
            self._invalidate_after_stack()
            return clean_json(meta)

    def _stack_hdr(self, runner: SirilRunner, seq: str, rows: list[dict], inc: list[int], groups: list[int], rs: dict,
                   rej: dict, progress):
        """Stack each exposure group of the one registered sequence, then blend them (workflows.md 9.2).

        All groups were registered together (one reference, one pixel grid).  Each group is stacked on its own with
        a reference frame of that group (see ``workflow.group_stack_script``); the group stacks have the canvases of
        their own frames, so they are placed on the union canvas (offset from the projected frame outlines, refined
        by phase correlation to a whole pixel) and blended with inverse-variance weights."""
        rseq = parse_seq(os.path.join(self.process, f"r_{seq}_.seq"))
        pos_of = {sf.filenum - 1: k + 1 for k, sf in enumerate(rseq.frames)}      # frame index -> 1-based position
        n_reg = len(rseq.frames)
        stacks: list[np.ndarray] = []
        headers = []
        exps: list[float] = []
        gidx: list[int] = []
        info: dict = {"groups": []}
        for g in groups:
            members = [i for i in inc if rows[i].get("group", 0) == g]
            if len(members) < 3:
                info.setdefault("dropped", []).append({"group": g, "n": len(members), "why": "fewer than 3 frames"})
                continue
            others = [pos_of[i] for i in inc if rows[i].get("group", 0) != g]
            best = min(members, key=lambda i: rows[i].get("wfwhm") or 1e9)
            out = f"stack_raw_g{g}"
            runner.run(workflow.group_stack_script(self.process, seq, n_reg, others, pos_of[best], rej, out, len(members)),
                       f"stack_group{g}", expect=[f"{out}.fit"],
                       label=f"Stacking ({rows[members[0]]['exptime']:g} s group)")
            a, h = fitsio.read_rgb(os.path.join(self.process, f"{out}.fit"))
            stacks.append(a)
            headers.append(h)
            exps.append(float(rows[members[0]]["exptime"]))
            gidx.append(g)
            info["groups"].append({"group": g, "exptime": exps[-1], "n": len(members), "reference": rows[best]["name"]})
        return self._blend_groups(stacks, headers, exps, gidx, rows, inc, rs, info)

    def _aperture(self, rs: dict) -> float:
        """Photometry aperture radius (stack px) for the group gain: ~3 FWHM of the registered subs."""
        try:
            fw = [r["fwhm"] for r in self._load_rows() if r.get("included") and r.get("fwhm")]
            return float(np.clip(3.0 * statistics.median(fw) * float(rs.get("scale", 1.0)), 6.0, 16.0))
        except Exception:
            return 8.0

    def _blend_groups(self, stacks: list[np.ndarray], headers: list, exps: list[float], gidx: list[int],
                      rows: list[dict], inc: list[int], rs: dict, info: dict):
        """Place the group stacks on the union canvas (geometry + whole-pixel refinement) and blend them."""
        if len(stacks) < 2:
            raise RuntimeError("HDR needs at least two exposure groups with 3+ usable frames each")
        f0 = self.fs.first
        scale = float(rs.get("scale", 1.0))
        hm_all = [rows[i]["H"] for i in inc]
        offs = []
        conv = None
        for g, a in zip(gidx, stacks):
            hm_g = [rows[i]["H"] for i in inc if rows[i].get("group", 0) == g]
            dx, dy, conv = stackpost.group_offset(a, hm_g, hm_all, (f0.width, f0.height), scale, conv)
            offs.append([dx, dy])
        canvas = (max(o[1] + a.shape[1] for o, a in zip(offs, stacks)), max(o[0] + a.shape[2] for o, a in zip(offs, stacks)))
        placed = [stackpost.paste(a, canvas, o[0], o[1]) for a, o in zip(stacks, offs)]
        ref = int(np.argmax(exps))
        info["groups"][ref]["offset"] = offs[ref]
        for k in range(len(placed)):
            if k == ref:
                continue
            rx, ry = stackpost.refine_shift(placed[ref], placed[k])
            if rx or ry:
                offs[k][0] += rx
                offs[k][1] += ry
                placed[k] = stackpost.paste(stacks[k], canvas, offs[k][0], offs[k][1])
            info["groups"][k]["offset"] = offs[k]
            info["groups"][k]["residual_shift"] = [rx, ry]
        valids = [stackpost.valid_mask(p) for p in placed]
        sats = [[stackpost.saturation_level(p[c], v) for c in range(p.shape[0])] for p, v in zip(placed, valids)]   # before any resampling
        for k in range(len(placed)):                    # sub-pixel (per channel) registration + PSF matching onto the longest
            if k == ref:
                continue
            placed[k], valids[k], info["groups"][k]["subpixel"] = stackpost.align_to_reference(
                placed[ref], placed[k], valids[ref], valids[k], sats[ref], sats[k])
            placed[k], valids[k], info["groups"][k]["psf_sigma"] = stackpost.match_psf(
                placed[ref], placed[k], valids[ref], valids[k], sats[ref], sats[k])
        blend, rep = stackpost.hdr_blend(placed, exps, valids, radius=self._aperture(rs), saturation=sats)
        bg_valid = np.any(np.stack(valids), axis=0)
        self._hdr_footprint = np.logical_and.reduce(valids)       # where every group has data (the crop stays inside)
        info["rgb_equal"] = stackpost.equalize_backgrounds(blend, bg_valid & stackpost.valid_mask(blend))
        info["blend"] = rep
        hdr = headers[ref].copy()
        hdr["STACKCNT"] = sum(g["n"] for g in info["groups"])
        hdr["LIVETIME"] = float(sum(g["n"] * g["exptime"] for g in info["groups"]))
        return blend, hdr, info

    def _finish_stack(self, arr: np.ndarray, hdr, sp: dict, rs: dict, rej: dict, inc: list[int], rows: list[dict],
                      medf, progress, hdr_info: dict | None = None) -> dict:
        """Crop the raw stack by coverage, write ``stack.fits`` and build the metadata."""
        cb = self._cb(progress)
        cb(0, 1, "Cropping")
        t0 = time.time()
        H, W = arr.shape[1:]
        do_crop = bool(sp.get("crop", True))
        thr = float(sp.get("crop_threshold", 0.3) or 0.3)
        hgroups = None
        if hdr_info and hdr_info.get("groups"):          # HDR: crop to where every group has data, else a seam (M 45: 15 s group alone) shows
            hgroups = [[rows[i]["H"] for i in inc if rows[i].get("group", 0) == g["group"] and rows[i].get("H")]
                       for g in hdr_info["groups"]]
        f0 = self.fs.first
        info: dict = {"method": "off"}
        box = (0, 0, W, H)
        if do_crop:
            hm = [rows[i]["H"] for i in inc if rows[i].get("H")]
            box, info = stackpost.auto_crop(arr, hm if len(hm) == len(inc) else None, (f0.width, f0.height),
                                            float(rs.get("scale", 1.0)), thr, hgroups,
                                            getattr(self, "_hdr_footprint", None) if hdr_info else None)
        x, y, w, h = box
        out = np.ascontiguousarray(arr[:, y:y + h, x:x + w])
        scale = float(rs.get("scale", 1.0))
        band = (read_json(os.path.join(self.process, "state.json"), {}) or {}).get("banding")
        hdr = hdr.copy() if hasattr(hdr, "copy") else hdr
        if band and band.get("applied") and hasattr(hdr, "add_history"):
            hdr.add_history(f"AstroPhoto Studio: {band['command']} on {band['pattern']} subs")
        fitsio.write_rgb(self._p("stack.fits"), out, fitsio.strip_wcs(hdr))   # the inherited WCS no longer fits
        self._write_maps(box, inc, rows, (f0.width, f0.height), scale, (W, H), info)
        self._time("crop", time.time() - t0)
        cb(1, 1, "Cropping")
        valid = out[:, ::4, ::4]
        sat = float((out >= 0.999).mean())
        ps = self.fs.pixel_scale
        wf = [rows[i]["wfwhm"] for i in inc if rows[i].get("wfwhm")]
        total_exp = sum(rows[i]["exptime"] or 0 for i in inc)
        masters = {}
        for u in getattr(self, "_units", []):
            for k, v in u.masters.items():
                masters.setdefault(k, os.path.basename(v))
        rep = self.calib_report or {}
        meta = {
            "mode": rs["mode"], "scale": scale, "object": self.fs.object, "filter": self.fs.filter,
            "is_dualband": self.fs.is_dualband, "kind": self.fs.kind, "bayer": self.fs.bayer, "sensor": self.fs.sensor,
            "n_frames": len(inc), "n_total": len(rows), "total_exposure": total_exp,
            "exptime": f0.exptime, "gain": f0.gain,
            "drizzle": {"scale": scale, "pixfrac": rs.get("pixfrac"), "kernel": rs.get("kernel")} if rs["mode"] == "drizzle" else None,
            "resampling_reason": rs["reason"],
            "rejection": {"type": rej["name"], "low": rej["low"], "high": rej["high"]},
            "weights": "wfwhm", "normalisation": "addscale", "framing": rs.get("framing", "max"),
            "crop": [x, y, w, h], "crop_info": info, "shape": [w, h], "stack_canvas": [W, H],
            "pixel_scale": (ps / scale) if ps else None,
            "fwhm_px": (medf * scale) if medf else None,
            "saturation": sat,
            "created": datetime.now().isoformat(timespec="seconds"),
            "calibration": {"dark": masters.get("dark"), "flat": masters.get("flat"), "bias": masters.get("bias"),
                            "cc": "dark 3 3" if masters.get("dark") else None, "summary": rep.get("summary")},
            "banding": band,
            "siril_version": "1.4.4", "sirilpipe_version": __version__,
            "params": sp, "local_norm": False, "sensor_pattern": None, "split": None,
            "nights": [{"folder": d, "n": len([i for i in inc if rows[i]["folder"] == d])} for d, _ in self.fs.nights],
            "plate_solve": None,
        }
        if hdr_info:
            meta["hdr"] = hdr_info
            meta["exptime"] = max(g["exptime"] for g in hdr_info["groups"])
            meta["weights"] = "wfwhm (per group); groups blended by inverse variance"
        return meta

    def _write_maps(self, box, inc: list[int], rows: list[dict], sub_size, scale: float, canvas, info: dict) -> None:
        """The Diagnostics tab's maps, cropped like the stack: ``coverage.fits`` (frames covering each pixel, from the registration homographies) and
        ``rejection_map.fits`` (Siril's ``-rejmap`` of the integration: the share of frames rejected per pixel - satellites, planes, cosmic rays)."""
        from astropy.io import fits as _fits
        x, y, w, h = box
        try:
            hm = [rows[i]["H"] for i in inc if rows[i].get("H")]
            if len(hm) == len(inc) and hm:
                conv = (info or {}).get("convention") or "FHF"
                cov, _ = stackpost.frame_coverage(hm, sub_size[0], sub_size[1], scale, canvas, conv, down=1)
                cov = cov[y:y + h, x:x + w].astype(np.float32)
                _fits.PrimaryHDU(cov).writeto(self._p("coverage.fits"), overwrite=True)
        except Exception as e:                                        # noqa: BLE001  a missing map never fails the stack
            log.warning("coverage map not written: %s", e)
        try:
            # Siril 1.4.4 names the merged map "<out>_low+high_rejmap.fit" (-rejmap); older builds "rejmap_<out>.fit"
            cands = sorted(set(glob.glob(os.path.join(self.process, "stack_raw*rejmap*.fit*")) + glob.glob(os.path.join(self.process, "*rejmap*.fit*"))))
            if cands:
                rm = _fits.getdata(cands[-1]).astype(np.float32)
                if rm.ndim == 3:
                    rm = rm.max(0)
                rm = rm[y:y + h, x:x + w] if rm.shape[0] >= y + h and rm.shape[1] >= x + w else rm
                _fits.PrimaryHDU(rm).writeto(self._p("rejection_map.fits"), overwrite=True)
                for f in cands:
                    os.remove(f)
        except Exception as e:                                        # noqa: BLE001
            log.warning("rejection map not written: %s", e)

    def _invalidate_after_stack(self) -> None:
        """A new stack makes every later product stale (restore, star removal, auto-finish)."""
        for f in ("linear.fits", "starless.fits", "stars.fits", "restore_meta.json", "starnet_meta.json", "autofinish.json",
                  "imagemm.fits", "imagemm_meta.json"):
            if os.path.exists(self._p(f)):
                os.remove(self._p(f))

    # ------------------------------------------------------------------ plate solving
    def plate_solve(self, progress=None) -> dict | None:
        """``platesolve -force`` on stack.fits; a failure is recorded in the meta and does not raise (SPCC then
        falls back to background neutralisation)."""
        with self.lock:
            stack = self._p("stack.fits")
            if not os.path.exists(stack):
                raise RuntimeError("no stack to plate-solve")
            cb = self._cb(progress)
            cb(0, 1, "Plate solving")
            meta = self.meta
            f0 = self.fs.first if self.fs else None
            scale = float((meta.get("drizzle") or {}).get("scale", 1.0) or 1.0) if meta.get("drizzle") else 1.0
            fr = f0
            if f0 is not None and scale != 1.0:
                import dataclasses
                fr = dataclasses.replace(f0, pixsize=(f0.pixsize / scale) if f0.pixsize else None)
            t0 = time.time()
            res: dict
            try:
                runner = SirilRunner(self.dir, progress=self._cb(progress), checkpoint=self.checkpoint)
                tmp = self._p("stack_solved.fit")
                for stale in (tmp, self._p("stack.fit")):
                    if os.path.exists(stale):
                        os.remove(stale)
                runner.run(workflow.platesolve_script(self.dir, "stack", fr), "platesolve", expect=[tmp],
                           label="Plate solving")
                _, h = fitsio.read_fits(tmp)
                res = _wcs_summary(h)
                if res.get("ok"):
                    os.replace(tmp, stack)
                else:
                    os.remove(tmp)
                res["seconds"] = round(time.time() - t0, 1)
            except Cancelled:
                raise
            except Exception as e:
                log.warning("plate solving failed: %s", str(e).splitlines()[0])
                res = {"ok": False, "error": str(e).splitlines()[0][:300], "seconds": round(time.time() - t0, 1)}
            self._time("platesolve", time.time() - t0)
            meta["plate_solve"] = res
            if res.get("ok"):
                meta["wcs"] = True
            if meta:
                write_json(self._p("stack_meta.json"), meta)
            cb(1, 1, "Plate solving")
            return clean_json(res)

    # ------------------------------------------------------------------ post-stack stages (sirilpipe.post)
    def run_denoise(self, stack_params: dict | None = None, progress=None, params: dict | None = None) -> dict:
        """"Restore": linear post-processing -> linear.fits (gradient, colour calibration, deconvolution, denoise).

        ``deconv_method`` "imagemm" (explicit choice; "auto" = BlurXTerminator when licensed, else Siril RL):
        the registered frames are restored jointly by ``imagemm_stage`` first (``imagemm.fits``, same grid as
        stack.fits) and ``post.restore`` then works on that instead of stack.fits - without its own
        deconvolution, ImageMM being one.  Without torch, or if ImageMM fails, the normal chain runs and
        restore_meta.json says so (``imagemm``: "unavailable" / "failed")."""
        from sirilpipe import post
        with self.lock:
            sp = dict(stack_params or {})
            # the processing setting ``restored_resolution`` is ImageMM's super-resolution factor r (api-contract: 1..3) unless imagemm_r was set explicitly
            try:
                rr = int(round(float((params or {}).get("restored_resolution", 1.0))))
                if "imagemm_r" not in sp and rr > 1:
                    sp["imagemm_r"] = rr
            except (TypeError, ValueError):
                pass
            self._recover_stack_swap()
            used, info = False, None
            if sp.get("deconv_method") == "imagemm":
                used, info = self._imagemm_stage(sp, progress)
            if used:
                out = self._restore_from(post, "imagemm.fits", info, sp, params, progress)
            else:
                out = post.restore(self.dir, self.meta, sp, params or {}, self._cb(progress), self.checkpoint)
            if info is not None:
                out = self._record_imagemm(out, used, info)
            return out

    def _imagemm_stage(self, sp: dict, progress) -> tuple[bool, dict]:
        """Run the ImageMM stage.  Returns (imagemm.fits is usable, record for restore_meta.json)."""
        global _IMAGEMM_WARNED
        from sirilpipe import imagemm_stage
        av = imagemm_stage.available()
        if not av["ok"]:
            if not _IMAGEMM_WARNED:
                _IMAGEMM_WARNED = True
                log.warning("ImageMM is not available (%s): restoring with BlurXTerminator / Siril RL instead", av["reason"])
            return False, {"state": "unavailable", "reason": av["reason"]}
        try:
            return True, {"state": "ran", **imagemm_stage.run(self.dir, self.meta, sp, self._cb(progress), self.checkpoint)}
        except Cancelled:
            raise
        except Exception as e:
            log.warning("ImageMM failed (%s): restoring with BlurXTerminator / Siril RL instead", str(e).splitlines()[0])
            return False, {"state": "failed", "reason": str(e).splitlines()[0][:400]}

    def _restore_from(self, post, name: str, info: dict, sp: dict, params, progress) -> dict:
        """``post.restore`` on ``name`` (a file of the session with stack.fits' header) instead of stack.fits.

        post.py honours ``stack_meta['restore_source']`` (or a ``restore_input`` argument) when it knows it;
        otherwise stack.fits is swapped for ``name`` while it runs and put back whatever happens.  The
        deconvolution is off for it: ImageMM is the deconvolution."""
        import inspect
        meta = dict(self.meta)
        r = int(info.get("r") or 1)
        if r != 1:
            meta.update(scale=float(meta.get("scale") or 1.0) * r, shape=[int(v) for v in info["shape"]],
                        pixel_scale=(meta["pixel_scale"] / r) if meta.get("pixel_scale") else None)
        meta["restore_source"] = name
        sp2 = {**sp, "deconv_method": "none"}
        cb = self._cb(progress)
        kw: dict = {}
        takes = "restore_input" in inspect.signature(post.restore).parameters
        if takes:
            kw["restore_input"] = name
        try:
            honours = "restore_source" in inspect.getsource(post)
        except (TypeError, OSError):
            honours = False
        if takes or honours:
            return post.restore(self.dir, meta, sp2, params or {}, cb, self.checkpoint, **kw)
        with self._stack_as(name):
            return post.restore(self.dir, meta, sp2, params or {}, cb, self.checkpoint)

    @contextlib.contextmanager
    def _stack_as(self, name: str):
        stack, keep = self._p("stack.fits"), self._p("process", SWAP_NAME)
        os.replace(stack, keep)
        try:
            shutil.copyfile(self._p(name), stack)
            yield
        finally:
            os.replace(keep, stack)

    def _recover_stack_swap(self) -> None:
        """A crash inside ``_stack_as`` leaves the real stack under ``process/``: put it back."""
        keep = self._p("process", SWAP_NAME)
        if os.path.exists(keep):
            os.replace(keep, self._p("stack.fits"))

    def _record_imagemm(self, out: dict, used: bool, info: dict) -> dict:
        """What ImageMM did, in restore_meta.json (and the returned meta): ``imagemm`` = ran | unavailable | failed."""
        path = self._p("restore_meta.json")
        rm = read_json(path, None)
        if rm is None:
            rm = dict(out or {})
        rec = {"imagemm": info["state"]}
        if used:
            rec["imagemm_info"] = {k: v for k, v in info.items() if k not in ("state", "frames", "frames_skipped")}
            rec["deconvolution_by"] = "imagemm"
        else:
            rec["imagemm_reason"] = info.get("reason")
        rm.update(rec)
        write_json(path, rm)
        out = dict(out or {})
        out.update(rec)
        return out

    def train_star_remover(self, params: dict | None = None, stack_params: dict | None = None, progress=None) -> dict:
        """Star removal -> starless.fits + stars.fits."""
        from sirilpipe import post
        with self.lock:
            return post.remove_stars(self.dir, self.meta, params or {}, stack_params or {}, self._cb(progress), self.checkpoint)

    def autofinish(self, params: dict | None = None, progress=None) -> dict:
        """Tune the processing params from image statistics.  Returns ``{"params", "report"}`` and also nested
        under ``"autofinish"`` (DESIGN section 2 wording) so both server conventions work."""
        from sirilpipe import post
        with self.lock:
            res = post.autofinish(self.dir, self.meta, params or {}, self._cb(progress), self.checkpoint)
            out = dict(res)
            out["autofinish"] = {"params": res.get("params"), "report": res.get("report")}
            return out

    def export(self, params: dict | None = None, progress=None, quality: int = 95, upscale: float = 1.0,
               tiff: bool = True) -> dict:
        from sirilpipe import post
        with self.lock:
            opts = {"quality": quality, "upscale": upscale, "tiff": tiff}
            return post.export(self.dir, self.meta, params or {}, opts, self._cb(progress), self.checkpoint)

    def preview_jpeg(self, params: dict | None = None, max_side: int = 1600) -> bytes:
        from sirilpipe import post
        return post.preview_jpeg(self.dir, self.meta, params or {}, max_side=max_side)

    def run_all(self, stack_params: dict | None = None, params: dict | None = None, export_opts: dict | None = None,
                progress=None) -> dict:
        """analysis -> stack -> plate solve -> restore -> (star removal) -> (auto-finish) -> export."""
        sp, pr, eo = dict(stack_params or {}), dict(params or {}), dict(export_opts or {})
        t0 = time.time()
        self.run_analysis(float(sp.get("sensitivity", 1.0) or 1.0), progress, sp)
        self.run_stack(sp, progress)
        self.plate_solve(progress)
        self.run_denoise(sp, progress, pr)
        if sp.get("star_remover", True):
            self.train_star_remover(pr, sp, progress)
        fin = None
        if sp.get("autofinish", True):
            fin = self.autofinish(pr, progress)
            pr = fin.get("params") or pr
        res = dict(self.export(pr, progress=progress, **eo))
        if fin:
            res["autofinish"] = {"params": fin.get("params"), "report": fin.get("report")}
        res["seconds"] = round(time.time() - t0, 1)
        return res

    # ------------------------------------------------------------------ local copy
    @contextlib.contextmanager
    def local_copy(self, progress=None):
        """With ASTROPHOTO_LOCAL_COPY on: copy the subs to a local disk first (slow NAS), read them from there
        and delete the copy afterwards.  Caches keep the library paths."""
        if not local_copy_enabled():
            yield self
            return
        if self.fs is None:
            self.scan(progress)
        root = local_copy_dir(self.workdir)
        files = [f for f in self.fs.frames]
        total = sum(os.path.getsize(f.path) for f in files if os.path.exists(f.path))
        done = 0
        os.makedirs(root, exist_ok=True)
        try:
            for i, f in enumerate(files, 1):
                d = os.path.join(root, hashlib.sha1(os.path.dirname(f.path).encode()).hexdigest()[:12])
                os.makedirs(d, exist_ok=True)
                dst = os.path.join(d, os.path.basename(f.path))
                if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(f.path):
                    shutil.copyfile(f.path, dst + ".part")
                    os.replace(dst + ".part", dst)
                self._local[f.path] = dst
                done += os.path.getsize(dst)
                if progress:
                    progress(i, len(files), f"Copying the subs to the local disk ({i}/{len(files)}, "
                             f"{done / 1e9:.1f} of {total / 1e9:.1f} GB)")
                self.checkpoint()
            yield self
        finally:
            self._local = {}
            for d in {os.path.dirname(p) for p in glob.glob(os.path.join(root, "*", "*"))}:
                shutil.rmtree(d, ignore_errors=True)

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        """Everything the UI needs to draw the dataset card and its step ticks (derived from files)."""
        f0 = self.infos[0] if self.infos else None
        fs = self.fs
        meta = self.meta
        exports = sorted(os.path.basename(p) for p in glob.glob(self._p("exports", "*")))
        total_min = round(fs.total_exposure / 60.0, 1) if fs else None
        st = {
            "folder": self.folder, "folders": self.folders, "workdir": self.dir,
            "object": f0.object if f0 else meta.get("object"), "filter": f0.filter if f0 else meta.get("filter"),
            "n_files": len(self.infos) if self.infos else None, "exptime": f0.exptime if f0 else None,
            "total_min": total_min, "instrument": (f0.telescope or f0.instrument) if f0 else None,
            "telescope": (f0.telescope or f0.instrument) if f0 else None,
            "kind": fs.kind if fs else meta.get("kind"), "is_dualband": fs.is_dualband if fs else meta.get("is_dualband"),
            "narrowband": fs.is_dualband if fs else meta.get("is_dualband"),
            "analysed": self.analysis is not None, "stacked": os.path.exists(self._p("stack.fits")),
            "has_stack": os.path.exists(self._p("stack.fits")), "has_linear": os.path.exists(self._p("linear.fits")),
            "has_starless": os.path.exists(self._p("starless.fits")),
            "has_autofinish": os.path.exists(self._p("autofinish.json")),
            "denoised": restoration_done(self.dir), "deconvolved": os.path.exists(self._p("restore_meta.json")),
            "star_remover": os.path.exists(self._p("starless.fits")), "autofinished": os.path.exists(self._p("autofinish.json")),
            "exports": exports, "stack_meta": clean_json(meta),
            "calibration_checked": os.path.exists(self._p("calibration.json")),
            "calibration": clean_json({"summary": (self.calib_report or {}).get("summary"), **(self.calib_report or {})})
            if self.calib_report else (read_json(self._p("calibration.json"), {}) or {}).get("report"),
            "tools": self._tool_names(),
            "version": __version__,
        }
        return st

    @staticmethod
    def _tool_names() -> list[str]:
        """Names of the external tools the pipeline can use (status() is polled often: no probing here;
        ``tools.tools_status()`` gives the real state for the Tools panel)."""
        return ["siril", "graxpert", "starnet", "rcastro"]


# ======================================================================= helpers
def _superpixel(raw: np.ndarray, pattern: str) -> np.ndarray:
    pattern = (pattern or "RGGB").upper()
    h, w = raw.shape
    h2, w2 = h // 2 * 2, w // 2 * 2
    out = np.zeros((h2 // 2, w2 // 2, 3), np.float32)
    cnt = [0, 0, 0]
    idx = {"R": 0, "G": 1, "B": 2}
    for k, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        c = idx[pattern[k]]
        out[..., c] += raw[dy:h2:2, dx:w2:2]
        cnt[c] += 1
    for c in range(3):
        out[..., c] /= max(cnt[c], 1)
    return out


def _autostretch(img: np.ndarray, target: float = 0.2) -> np.ndarray:
    """Per-channel-linked MTF autostretch for thumbnails."""
    x = img - np.percentile(img, 0.5)
    x = np.clip(x, 0, None)
    hi = np.percentile(x, 99.7) + 1e-9
    x = np.clip(x / hi, 0, 1)
    m = float(np.median(x)) or 1e-3
    mid = np.clip((m * (target - 1)) / (target * (2 * m - 1) - m), 1e-3, 0.999)
    return np.clip(((mid - 1) * x) / (((2 * mid - 1) * x) - mid), 0, 1).astype(np.float32)


def _wcs_summary(h) -> dict:
    """Plate-solution facts from a FITS header written by Siril's platesolve."""
    if "CTYPE1" not in h or "CRVAL1" not in h:
        return {"ok": False, "error": "platesolve finished but the stack has no WCS"}
    cd = np.array([[h.get("CD1_1", 0.0), h.get("CD1_2", 0.0)], [h.get("CD2_1", 0.0), h.get("CD2_2", 0.0)]], float)
    if not cd.any() and "CDELT1" in h:
        pc = np.array([[h.get("PC1_1", 1.0), h.get("PC1_2", 0.0)], [h.get("PC2_1", 0.0), h.get("PC2_2", 1.0)]], float)
        cd = np.diag([h["CDELT1"], h.get("CDELT2", h["CDELT1"])]) @ pc
    scale = float(np.sqrt(abs(np.linalg.det(cd))) * 3600.0)
    rot = float(np.degrees(np.arctan2(cd[0, 1], cd[1, 1]))) if cd.any() else None
    return {"ok": True, "ra": float(h["CRVAL1"]), "dec": float(h["CRVAL2"]), "pixel_scale": scale, "rotation": rot,
            "ctype": h.get("CTYPE1")}
