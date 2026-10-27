"""Hyper-tuned "Run everything & export": before each stage, an experiment-lab study of that stage on this dataset; the stage then runs with the
study's best trial, and the next stage's study starts from that result.

Stages (the pipeline's order; each study is a normal Experiments-tab study, ``lab/studies/<time>_ht-<stage>-<target>``, so its trials, charts and
previews can be inspected and continued there):

  stack       Registration & integration (Session.run_stack, re-stacking the session itself): rejection algorithm / sigmas, frame weighting, frame
              selection, resampling, drizzle drop size / kernel, banding correction.  Score: depth (point- and extended-source S/N, empty apertures).
  restore     one study of the whole restoration: satellite-trail removal (Cosmic Clarity, a variable), the gradient model, deconvolution / noise
              reduction tools and strengths (super-resolution is the user's choice of output and runs as chosen).  Score: depth, with flux conservation, fidelity above the PSF scale, no dark
              rings, the gradient model's emission loss / structure bounded and no more trails than the removal leaves.
  stars       the star remover.  Score: star light left + holes, nebula untouched.
  finish      after Auto-finish (when on: its settings are the study's baseline), the finishing chain, the export's own (NXT / Cosmic Clarity per trial).
              Score: structural fidelity to the linear data (the significant structure made visible, TMQI's measure adapted to the data's
              noise), every finishing gate no worse than the baseline.  No reference image is involved anywhere.

The objective of the whole run is the chain of these: each stage is optimised for what that stage can be judged on without a ground truth, on
the output of the best earlier stages (coordinate-wise, the stage order the pipeline itself has); the export then renders the best finishing
trial at full resolution and records its gates.  What is searched: every parameter a task marks ``hypertune`` (lab.tasks), its choices limited to
the tools that can run here (an unlicensed BlurXTerminator would otherwise be "chosen" by a trial that ran no deconvolution); the baseline of each
study is the user's own settings, and a trial replaces it only when it is feasible and better by more than the measurement's resolution
(``Task.min_gain``).  Choices of output and of style the search has no measure for stay the user's: the drizzle / super-resolution scale, the crop,
the palette and colour grade, and an ImageMM restoration (hours per run: its study is not run; the
stage runs as chosen).
"""
from __future__ import annotations

import json
import logging
import math
import os
import time

log = logging.getLogger(__name__)

STAGES = [("stack", "Register & integrate"), ("restore", "Restore"), ("stars", "Remove stars"), ("finish", "Finish")]
MIN_TRIALS, MAX_TRIALS, DEFAULT_TRIALS = 10, 500, 25


def available_tools() -> dict:
    """Which tools can run on this machine now (local checks only)."""
    from . import tools
    out = {}
    try:
        rc = tools.rcastro.available()
        for k in ("bxt", "nxt", "sxt"):
            out[k] = bool(rc and tools.rcastro.licensed(k))
    except Exception:                                           # noqa: BLE001
        out.update(bxt=False, nxt=False, sxt=False)
    try:
        gst = tools.graxpert.status()
        gm = gst.get("models") or {}
        out["graxpert_bge"] = bool(gst.get("ok") and gm.get("bge", True))
        out["graxpert_denoise"] = bool(gst.get("ok") and gm.get("denoise", True))
        out["graxpert_deconv"] = bool(gst.get("ok") and gm.get("deconv_obj", False) and gm.get("deconv_stellar", False))
    except Exception:                                           # noqa: BLE001
        out.update(graxpert_bge=False, graxpert_denoise=False, graxpert_deconv=False)
    try:
        out["starnet"] = bool(tools.starnet.status().get("ok"))
    except Exception:                                           # noqa: BLE001
        out["starnet"] = False
    try:
        for m in ("sharpen", "denoise", "satellite", "darkstar", "superres"):
            out[f"cc_{m}"] = bool(tools.cosmic_clarity.ready(m))
    except Exception:                                           # noqa: BLE001
        for m in ("sharpen", "denoise", "satellite", "darkstar", "superres"):
            out[f"cc_{m}"] = False
    return out


# the tool a choice needs (param, value) -> available_tools key
NEEDS = {("deconv_method", "bxt"): "bxt", ("deconv_stars_method", "bxt"): "bxt", ("deconv_method", "graxpert"): "graxpert_deconv",
         ("deconv_stars_method", "graxpert"): "graxpert_deconv", ("denoise_method", "nxt"): "nxt", ("denoise_method", "graxpert"): "graxpert_denoise",
         ("star_removal", "sxt"): "sxt", ("star_removal", "starnet"): "starnet", ("star_removal", "darkstar"): "cc_darkstar",
         ("bg_method", "graxpert"): "graxpert_bge", ("cc_sharpen", "both"): "cc_sharpen", ("cc_sharpen", "stellar"): "cc_sharpen",
         ("cc_sharpen", "nonstellar"): "cc_sharpen", ("cc_denoise", True): "cc_denoise", ("satellite_removal", True): "cc_satellite",
         # "auto" deconvolution is BlurXTerminator, or Siril RL (CPU only) without its licence: searched only where it is BXT
         ("deconv_method", "auto"): "bxt", ("deconv_stars_method", "auto"): "bxt"}


def task_param(task, name: str) -> dict:
    return next(p for p in task.params if p["name"] == name)


def rule_pixfrac(sp: dict) -> float:
    """The drizzle drop size the pipeline's rule uses when ``drizzle_pixfrac`` is 0: 1 / scale (1 at 1x and with Cosmic Clarity's super-resolution)."""
    sc = float(sp.get("scale", 1.0) or 1.0)
    return round(1.0 / sc, 3) if sc > 1 and str(sp.get("superres_method") or "drizzle") == "drizzle" else 1.0


def _current(p: dict, sp: dict, params: dict):
    """The user's current value of a task parameter (the study's baseline)."""
    from . import post
    if p.get("pipeline"):
        k = p["pipeline"]
        v = sp.get(k, p["default"])
        if k == "drizzle_pixfrac" and not float(v or 0):      # 0 = the rule's drop size
            v = rule_pixfrac(sp)
        return v
    if p.get("processing"):
        k = p["processing"]
        return params.get(k, post.DEFAULTS.get(k, post.EXTRA_DEFAULTS.get(k, p["default"])))
    return p["default"]


def build_space(task, sp: dict, params: dict, *, dual: bool, tools: dict, extra_fixed: dict | None = None,
                only: tuple[str, ...] | None = None, exclude: tuple[str, ...] = ()) -> tuple[dict, list[str]]:
    """The hyper-tune search space of ``task``: its ``hypertune`` parameters tuned (choices without the tools that cannot run), the others fixed at
    the user's values (``only``: tune just these; ``exclude``: never these).  Returns (space, notes)."""
    space, notes = {}, []
    for p in task.params:
        cur = _current(p, sp, params)
        if extra_fixed and p["name"] in extra_fixed:
            space[p["name"]] = {"tune": False, "value": extra_fixed[p["name"]]}
            continue
        if (only is not None and p["name"] not in only) or p["name"] in exclude:
            space[p["name"]] = {"tune": False, "value": cur}
            continue
        cond = p.get("hypertune")
        on = cond is True or (cond == "dual" and dual)
        if not on:
            space[p["name"]] = {"tune": False, "value": cur}
            continue
        e = {"tune": True, "value": cur}
        if p["type"] in ("float", "int"):
            lo, hi = p.get("ht_range") or (p["low"], p["high"])           # a setting that targets a gated quantity: the gate's band
            tr = getattr(task, "trust_region", None)
            if tr and isinstance(cur, (int, float)) and not isinstance(cur, bool):
                # a trust region round the baseline (the incumbent: Auto-finish's measured settings): the trials refine it instead of re-discovering
                # how to stretch - 28 finishing settings searched over their full ranges left most trials outside the gates (Sh2-131, 2026-10-25)
                span = (hi - lo) * tr
                lo, hi = max(lo, float(cur) - span), min(hi, float(cur) + span)
                if p["type"] == "int":
                    lo, hi = int(math.floor(lo)), int(math.ceil(hi))
                if hi <= lo:
                    space[p["name"]] = {"tune": False, "value": cur}
                    continue
            e.update(low=lo, high=hi, log=bool(p.get("log")))
        elif p["type"] == "categorical":
            ch = list(p.get("ht_choices") or p["choices"])
            drop = [c for c in ch if (p["name"], c) in NEEDS and not tools.get(NEEDS[(p["name"], c)], True)]
            ch = [c for c in ch if c not in drop]
            if drop:
                notes.append(f"{p['name']}: {', '.join(map(str, drop))} cannot run here (not installed / licensed) - not searched")
            if str(cur) in [str(d) for d in drop]:
                cur = ch[0] if ch else cur
                e["value"] = cur
            e["choices"] = ch
            if len(ch) < 2:
                e = {"tune": False, "value": ch[0] if ch else cur}
        elif p["type"] == "bool":
            if (p["name"], True) in NEEDS and not tools.get(NEEDS[(p["name"], True)], True):
                notes.append(f"{p['name']}: the tool cannot run here - kept off")
                e = {"tune": False, "value": False}
        space[p["name"]] = e
    return space, notes


class HyperTune:
    """One hyper-tuned run of a Session (``run``)."""

    def __init__(self, s, sp: dict, params: dict, export_opts: dict | None, progress, n_trials: int, workdir: str | None = None, device: str = "auto"):
        from .lab.tasks import TASKS
        self.s, self.sp, self.params, self.eo = s, dict(sp), dict(params or {}), dict(export_opts or {})
        self.progress = progress or (lambda i, n, m: None)
        self.n = int(min(MAX_TRIALS, max(MIN_TRIALS, int(n_trials or DEFAULT_TRIALS))))
        self.workdir = workdir or os.path.dirname(os.path.abspath(s.dir))
        self.device = str(device or sp.get("device") or "auto")
        self.tasks = TASKS
        self.report: dict = {"trials_per_stage": self.n, "stages": [], "started": time.time()}
        self.tools = available_tools()
        self.report["tools"] = self.tools

    # ------------------------------------------------------------------ helpers
    def _cancelled(self) -> bool:
        ev = getattr(self.s, "cancel_flag", None)
        return bool(ev is not None and ev.is_set())

    def _check(self):
        from .util import Cancelled
        if self._cancelled():
            raise Cancelled()

    def _dataset_spec(self) -> dict:
        from .util import split_folders
        folders = list(getattr(self.s, "folders", None) or split_folders(self.s.folder))
        name = (self.s.meta.get("object") if self.s.meta else None) or os.path.basename(os.path.normpath(folders[0]))
        return {"kind": "real", "folder": os.pathsep.join(folders), "name": f"{name}" + (f" · {len(folders)} nights" if len(folders) > 1 else "")}

    def _dual(self) -> bool:
        from . import post
        try:
            return bool(post._dual_band(self.s.meta or {}))
        except Exception:                                       # noqa: BLE001
            return False

    def study(self, stage: str, label: str, options: dict | None = None, extra_fixed: dict | None = None, only: tuple[str, ...] | None = None,
              exclude: tuple[str, ...] = (), sampler: str | None = None, n_trials: int | None = None, key: str | None = None, warm: bool = False) -> dict:
        """Create and run the study of one stage in-process; returns its best record (lab.run.best_record) with the study id."""
        from .lab import run as labrun
        task = self.tasks[stage]
        sampler = sampler or getattr(task, "ht_sampler", "tpe")
        space, notes = build_space(task, self.sp, self.params, dual=self._dual(), tools=self.tools, extra_fixed=extra_fixed,
                                   only=only, exclude=exclude)
        tuned = [k for k, v in space.items() if v.get("tune")]
        if sampler == "grid":                                   # every combination once (the user's own value may add a choice)
            n_trials = 1
            for k in tuned:
                n_trials *= 2 if task_param(task, k)["type"] == "bool" else len(space[k].get("choices") or task_param(task, k)["choices"])
        rec = {"stage": key or stage, "label": label, "score": task.score, "tuned": tuned, "notes": notes, "started": time.time()}
        if not tuned:
            rec.update(state="skipped", reason="nothing to search (every choice fixed or unavailable)")
            self.report["stages"].append(rec)
            return rec
        obj = (self.s.meta or {}).get("object") or "dataset"
        import re
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"ht-{key or stage}-{obj}").strip("_")[:60]
        sid = time.strftime("%Y%m%d-%H%M%S") + "_" + safe
        d = os.path.join(self.workdir, "lab", "studies", sid)
        os.makedirs(d, exist_ok=True)
        cfg = {"name": f"Hyper-tune {obj}: {label}", "task": stage, "dataset": self._dataset_spec(), "options": {**(options or {})},
               "space": space, "objectives": [{"metric": task.score, "direction": task.metrics[task.score]["direction"]}],
               "sampler": sampler, "seed": 0, "n_trials": int(n_trials or self.n), "timeout_min": 0, "warm_start": (max(3, int(n_trials or self.n) // 8) if warm else 0), "device": self.device, "baseline": True, "constraints": True,
               "workdir": self.workdir, "created": time.time(), "hypertune": {"stage": stage, "session": self.s.dir}}
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(cfg, f, indent=1, default=str)
        rec["study"] = sid
        state = {"trial": 0}
        n = int(n_trials or self.n)

        def on_log(msg: str):
            if msg.startswith("Trial "):
                try:
                    state["trial"] = int(msg.split()[1].rstrip(":"))
                except (ValueError, IndexError):
                    pass
            k = state["trial"]
            self.progress(min(k, n - 1), n, f"Hyper-tune · {label} · trial {k + 1}/{n}: {msg[:160]}")

        self.progress(0, n, f"Hyper-tune · {label}: {len(tuned)} settings, {n} trials")
        st = labrun.run_study(d, cancel=self.s.cancel_flag, on_log=on_log, session=self.s)
        self._check()
        if st.get("state") == "error":
            rec.update(state="error", reason=st.get("error"))
            self.report["stages"].append(rec)
            log.warning("hyper-tune %s study failed: %s", stage, st.get("error"))
            return rec
        try:
            best = json.load(open(os.path.join(d, "best.json")))
        except (OSError, ValueError):
            rec.update(state="error", reason="no completed trial")
            self.report["stages"].append(rec)
            return rec
        rec.update(state="done", best_trial=best["trial"], criterion=best["criterion"], best=best["values"][0] if best.get("values") else None,
                   baseline=(best.get("baseline_values") or [None])[0], feasible=best.get("feasible"), is_baseline=best.get("is_baseline"),
                   settings=best.get("settings"), processing=best.get("processing"), n_complete=best.get("n_complete"),
                   metrics={k: v for k, v in (best.get("metrics") or {}).items() if isinstance(v, (int, float, bool)) or k == "violations"})
        changed = {}
        for p in task.params:
            key = p.get("pipeline") or p.get("processing")
            new = (best.get("settings") or {}).get(key, (best.get("processing") or {}).get(key)) if key else None
            if key is None or new is None:
                continue
            old = _current(p, self.sp, self.params)
            if str(old) != str(new):
                changed[key] = [old, new]
        rec["changed"] = changed
        rec["seconds"] = round(time.time() - rec["started"], 1)
        self.report["stages"].append(rec)
        return rec

    def adopt(self, rec: dict, keys: tuple[str, ...] | None = None):
        """Take the best trial's settings into the run's stack / processing settings."""
        if rec.get("state") != "done":
            return
        for k, v in (rec.get("settings") or {}).items():
            if keys is None or k in keys:
                self.sp[k] = v
        for k, v in (rec.get("processing") or {}).items():
            if keys is None or k in keys:
                self.params[k] = v

    # ------------------------------------------------------------------ the run
    def run(self) -> dict:
        from . import post
        s, sp = self.s, self.sp
        t0 = time.time()
        # the calibration and registration are the stack stage's work: they run inside its experiment (StackTask.prepare), not before it
        # one study per pipeline stage, the user's number of trials each: 1. Register & integrate (the trials re-stack this session)
        user_sp = dict(sp)
        rec = self.study("stack", "Register & integrate", options={"in_place": 1, "crop": 1024, "stack_params": user_sp})
        self.adopt(rec)
        pf = (rec.get("settings") or {}).get("drizzle_pixfrac")
        if pf is not None and not float(user_sp.get("drizzle_pixfrac") or 0) and abs(float(pf) - rule_pixfrac(user_sp)) < 1e-9:
            self.sp["drizzle_pixfrac"] = 0.0                    # the best trial kept the rule's drop size: keep the rule
        self.progress(0, 1, "Hyper-tune · stacking with the best registration & integration settings")
        s.run_stack(self.sp, self.progress)
        s.plate_solve(self.progress)
        self._check()
        # 2. Restore: satellite-trail removal, gradient model, deconvolution and noise reduction in one study
        if "imagemm" in post.deconv_choice(self.sp):
            self.report["stages"].append({"stage": "restore", "label": "Restore", "state": "skipped",
                                          "reason": "ImageMM chosen: a multi-frame restoration takes hours per run, its settings are not searched; the stage runs as chosen"})
        else:
            rec = self.study("restore", "Restore", options={"stack_params": dict(self.sp)})
            self.adopt(rec)
        self.progress(0, 1, "Hyper-tune · restoring with the best settings")
        s.run_denoise(self.sp, self.progress, self.params)
        self._check()
        # 3. Remove stars
        if self.sp.get("star_remover", True):
            rec = self.study("stars", "Remove stars")
            self.adopt(rec)
            self.progress(0, 1, "Hyper-tune · separating the stars with the best tool")
            s.train_star_remover(self.params, self.sp, self.progress)
            self._check()
        # 4. Finish: the experiment first, from your settings; with Auto-finish on, its proposal is one trial of the study (the stage itself is not
        # run before its experiment), and the best trial finishes the export
        fin = None
        rec = self.study("finish", "Finish", options={"external": 1, "max_side": 0, "params": dict(self.params),
                                                      "autofinish_candidate": int(bool(self.sp.get("autofinish", True)))})
        self.adopt(rec)
        # 8. export with the best of every stage
        self.progress(0, 1, "Hyper-tune · exporting with the best settings of every stage")
        res = dict(s.export(self.params, progress=self.progress, **self.eo))
        self.report["seconds"] = round(time.time() - t0, 1)
        self.report["stack_params"] = {k: self.sp.get(k) for k in sorted(self.sp) if not k.startswith("imagemm_") and k not in ("n2n_split", "n2n_loss")}
        self.report["params"] = dict(self.params)
        g = None
        try:
            side = json.load(open(res["json"])) if res.get("json") else {}
            g = (side.get("linear") or side.get("linear_info") or {}).get("gates")
        except Exception:                                       # noqa: BLE001
            pass
        if isinstance(g, dict):
            self.report["export_gates"] = {"passed": g.get("passed"), "failed": g.get("failed")}
        try:
            with open(os.path.join(s.dir, "hypertune.json"), "w") as f:
                json.dump(self.report, f, indent=1, default=str)
        except OSError:
            pass
        res["hypertune"] = self.report
        res["autofinish"] = {"params": dict(self.params), "report": (fin or {}).get("report") or {"hypertune": True}}
        res["seconds"] = round(time.time() - t0, 1)
        return res


def run_all(s, sp: dict, params: dict, export_opts: dict | None, progress, n_trials: int | None = None, workdir: str | None = None) -> dict:
    """The hyper-tuned Run everything & export of Session ``s`` (``sp['hypertune_trials']`` trials per stage)."""
    n = int(n_trials or sp.get("hypertune_trials") or DEFAULT_TRIALS)
    return HyperTune(s, sp, params, export_opts, progress, n, workdir=workdir, device=str(sp.get("device") or "auto")).run()
