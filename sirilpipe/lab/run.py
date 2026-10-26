"""Run one Optuna study (Akiba et al. 2019, KDD) of an experiment-lab task.

    python -m sirilpipe.lab.run <study_dir>

``<study_dir>/config.json``::

    {"name": ..., "task": "stack" | "satellite" | "background" | "restore" | "stars" | "finish" | "superres",
     "dataset": {"kind": "real", "folder": ...} | {"kind": "synthetic", "dir": ...},
     "options": {...task options...},
     "space": {param: {"tune": bool, "low", "high", "log", "step", "choices", "value"}},
     "objectives": [{"metric": ..., "direction": "minimize" | "maximize"}, ...],
     "sampler": "tpe" | "es" | "random" | "nsga2" | "grid", "seed": 0,
     "n_trials": 30, "timeout_min": 0, "device": "auto", "baseline": true, "constraints": true}

Trials are stored in ``<study_dir>/study.db`` (SQLite, Optuna RDB storage); every metric is
a user attribute of its trial; the progress is in ``status.json`` and ``log.txt``, and each
trial's result is previewed in ``trials/<number>.jpg`` (one stretch for the whole study).
The first trial is the baseline: a tuned parameter's ``value`` in the space (the settings
the user runs; the pipeline default when not given), so every other trial is compared with
what the pipeline does today.  ``constraints``: the task's constraints (``Task.limits``) are
recorded with every trial (``violations``; Optuna's constrained TPE / NSGA-II samplers steer
away from infeasible regions, Ozaki et al. 2022) and the best trial is chosen among the
feasible ones (``best_trial``), written to ``best.json`` at the end.  A parameter whose
``when`` condition fails is not sampled; a trial whose active settings repeat an earlier
completed trial's reuses its result instead of running again.  SIGTERM stops the study
after cancelling the running trial; ``run_study`` runs a study in-process (the hyper-tuned
pipeline: sirilpipe.hypertune).
"""
from __future__ import annotations

import json
import math
import os
import platform
import shutil
import signal
import sys
import threading
import time
import traceback
from datetime import datetime

import numpy as np


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


def _jsonable(m):
    return json.loads(json.dumps(m, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def condition_holds(pdef: dict, values: dict) -> bool:
    """A parameter's ``when``: {param: [values]} (all hold) or a list of such dicts (any holds); compared as strings (bool / number / text alike)."""
    w = pdef.get("when")
    if not w:
        return True
    alts = w if isinstance(w, list) else [w]
    for alt in alts:
        if all(str(values.get(k)) in {str(v) for v in allowed} for k, allowed in alt.items()):
            return True
    return False


def ordered(params: list[dict]) -> list[dict]:
    """The parameters with every ``when`` parent before its children."""
    done, out, rest = set(), [], list(params)
    while rest:
        nxt = []
        for p in rest:
            w = p.get("when")
            parents = {k for alt in (w if isinstance(w, list) else [w]) for k in (alt or {})}
            if parents <= done or not parents & {q["name"] for q in rest}:
                out.append(p)
                done.add(p["name"])
            else:
                nxt.append(p)
        if len(nxt) == len(rest):                      # a cycle: keep the declared order
            out += nxt
            break
        rest = nxt
    return out


def suggest(trial, name: str, spec: dict, pdef: dict):
    """Sample one parameter, or return its fixed value."""
    if not spec.get("tune"):
        return spec.get("value", pdef["default"])
    t = pdef["type"]
    if t == "bool":
        return trial.suggest_categorical(name, [True, False])
    if t == "categorical":
        return trial.suggest_categorical(name, list(spec.get("choices") or pdef["choices"]))
    lo, hi = float(spec.get("low", pdef.get("low"))), float(spec.get("high", pdef.get("high")))
    log = bool(spec.get("log", pdef.get("log", False)))
    step = spec.get("step")
    if t == "int":
        return trial.suggest_int(name, int(lo), int(hi), log=log, step=int(step) if step and not log else 1)
    return trial.suggest_float(name, lo, hi, log=log, step=float(step) if step and not log else None)


def baseline_space(space: dict, task) -> dict:
    """The space with each tuned parameter's range / choices widened to hold its baseline value (the user's own setting may lie outside the
    task's default range, and a fixed value outside a distribution would derail the sampler)."""
    out = {}
    for p in task.params:
        sp = dict(space.get(p["name"], {}))
        if sp.get("tune") and "value" in sp:
            v = sp["value"]
            if p["type"] in ("float", "int") and isinstance(v, (int, float)) and not isinstance(v, bool):
                sp["low"] = min(float(sp.get("low", p.get("low"))), float(v))
                sp["high"] = max(float(sp.get("high", p.get("high"))), float(v))
                if p["type"] == "int":
                    sp["low"], sp["high"] = int(math.floor(sp["low"])), int(math.ceil(sp["high"]))
            elif p["type"] == "categorical":
                ch = list(sp.get("choices") or p["choices"])
                if v not in ch and str(v) not in [str(c) for c in ch]:
                    ch.append(v)
                sp["choices"] = ch
        out[p["name"]] = sp
    return out


def grid_space(space: dict, task) -> dict:
    """The search space of a grid study: categorical / bool choices, and stepped numeric ranges."""
    out = {}
    for p in task.params:
        sp = space.get(p["name"], {})
        if not sp.get("tune"):
            continue
        if p["type"] == "bool":
            out[p["name"]] = [True, False]
        elif p["type"] == "categorical":
            out[p["name"]] = list(sp.get("choices") or p["choices"])
        else:
            lo, hi = float(sp.get("low", p["low"])), float(sp.get("high", p["high"]))
            step = sp.get("step")
            if not step:
                raise ValueError(f"grid search: give '{p['name']}' a step")
            vals = list(np.arange(lo, hi + 1e-9 * abs(step), float(step)))
            out[p["name"]] = [int(round(v)) for v in vals] if p["type"] == "int" else [float(v) for v in vals]
    return out


def _violation(t) -> float:
    return float(sum(max(0.0, v) for v in (t.user_attrs.get("constraints") or [])))


def best_trial(study, cfg: dict, task=None):
    """The best trial as the study's objectives define it, among the feasible trials (every constraint met; when none is, the least
    infeasible): the optimum of a single objective - the baseline kept when the optimum improves the task's score by less than
    ``task.min_gain``; with several objectives, the Pareto-optimal trial that is best on the first (primary).  Returns (trial, why)."""
    import optuna
    objs = cfg["objectives"]
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.values]
    if not done:
        raise ValueError("no completed trial")
    feas = [t for t in done if _violation(t) <= 0]
    note = ""
    if feas:
        pool = feas
        if len(feas) < len(done):
            note = f"; {len(done) - len(feas)} of {len(done)} trials broke a constraint"
    else:
        least = min(_violation(t) for t in done)
        pool = [t for t in done if _violation(t) <= least + 1e-12]
        note = f"; no trial met every constraint: the least infeasible (violation {least:.3g})"
    signs = [1 if (o.get("direction") or "minimize") == "minimize" else -1 for o in objs]
    if len(objs) == 1:
        sgn = signs[0]
        best = min(pool, key=lambda q: sgn * q.values[0])
        why = f"best {objs[0]['metric']} ({objs[0].get('direction', 'minimize')})"
        base = next((t for t in pool if t.user_attrs.get("baseline")), None)
        gain = 0.0
        if task is not None and objs[0]["metric"] == getattr(task, "score", None) and base is not None:
            # the measurement's resolution: absolute (log-scale scores), or relative to the baseline's score (a distance)
            gain = max(float(getattr(task, "min_gain", 0.0) or 0.0), float(getattr(task, "min_gain_rel", 0.0) or 0.0) * abs(base.values[0]))
        if base is not None and best.number != base.number and gain > 0 and sgn * (base.values[0] - best.values[0]) < gain:
            why = (f"baseline kept: the best {objs[0]['metric']} ({best.values[0]:.4g}, trial {best.number}) improves it by less than {gain:g} "
                   f"(the measurement's resolution)")
            best = base
        return best, why + note

    def dominates(a, b):
        return all(s * x <= s * y for s, x, y in zip(signs, a.values, b.values)) and any(s * x < s * y for s, x, y in zip(signs, a.values, b.values))
    front = [t for t in pool if not any(dominates(o, t) for o in pool if o is not t)]
    t = min(front, key=lambda q: signs[0] * q.values[0])
    return t, (f"Pareto-optimal on {' × '.join(o['metric'] for o in objs)}, best {objs[0]['metric']} of the {len(front)} on the front{note}")


def best_record(study, cfg: dict, task) -> dict:
    """``best_trial`` with the settings it ran, split as the pipeline takes them (pipeline / processing / not settings)."""
    t, why = best_trial(study, cfg, task)
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
    base = next((q for q in study.trials if q.user_attrs.get("baseline") and q.values), None)
    return {"trial": t.number, "values": t.values, "criterion": why, "settings": settings, "processing": processing, "not_pipeline": other,
            "metrics": t.user_attrs.get("metrics"), "feasible": _violation(t) <= 0, "baseline_trial": base.number if base else None,
            "baseline_values": base.values if base else None, "baseline_metrics": base.user_attrs.get("metrics") if base else None,
            "is_baseline": bool(t.user_attrs.get("baseline")), "n_complete": sum(1 for q in study.trials if q.values)}


def run_study(study_dir: str, cancel: threading.Event | None = None, on_log=None, session=None, install_signals: bool = False) -> dict:
    """Run the study of ``study_dir`` to its trial budget.  ``cancel``: an Event that stops it (set by SIGTERM with ``install_signals``);
    ``on_log(msg)``: called with every log line (an exception from it - the hyper-tune job's cancel - stops the study);
    ``session``: the Session the task's dataset uses (the hyper-tune's own).  Returns the final status, with the best trial."""
    import optuna
    from .tasks import TASKS, Dataset

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_dir = os.path.abspath(study_dir)
    cfg = json.load(open(os.path.join(study_dir, "config.json")))
    task = TASKS[cfg["task"]]
    status_path = os.path.join(study_dir, "status.json")
    log_path = os.path.join(study_dir, "log.txt")
    os.makedirs(os.path.join(study_dir, "trials"), exist_ok=True)
    status = {"state": "starting", "pid": os.getpid(), "started": time.time(), "message": "Starting",
              "trial": None, "host": platform.node()}
    if not install_signals:
        status["in_process"] = True          # inside another process (the hyper-tune job): stopped through it, never by a signal to this pid
    _write_json(status_path, status)
    cancel = cancel or threading.Event()
    lock = threading.Lock()
    outer = {"error": None}

    def log(msg):
        line = f"{datetime.now().strftime('%H:%M:%S')}  {msg}"
        with lock:
            with open(log_path, "a") as f:
                f.write(line + "\n")
            status["message"] = msg
            _write_json(status_path, status)
        if on_log is not None and outer["error"] is None:
            try:
                on_log(msg)
            except BaseException as e:                        # the caller's cancel: stop at the next checkpoint
                outer["error"] = e
                cancel.set()

    if install_signals:
        def on_term(signum, frame):
            cancel.set()
            log("Stop requested: cancelling the running trial")
        signal.signal(signal.SIGTERM, on_term)
        signal.signal(signal.SIGINT, on_term)
        if hasattr(signal, "SIGBREAK"):          # Windows: the server stops a study with Ctrl-Break
            signal.signal(signal.SIGBREAK, on_term)

    try:
        device = str(cfg.get("device") or "auto")
        log(f"Task {task.label}; dataset {cfg['dataset'].get('name') or cfg['dataset']}; device {device}")
        ds = Dataset(cfg["dataset"], cfg.get("workdir") or os.path.dirname(os.path.dirname(study_dir)), session=session)
        status["state"] = "preparing"
        ctx = task.prepare(ds, cfg.get("options", {}), device, log, cancel, study_dir)
        objectives = cfg["objectives"]
        directions = [o.get("direction") or task.metrics[o["metric"]]["direction"] for o in objectives]
        space = baseline_space(cfg.get("space", {}), task)
        pdefs = {p["name"]: p for p in task.params}
        order = ordered(task.params)
        use_constraints = bool(cfg.get("constraints", True))
        seed = int(cfg.get("seed", 0))
        name = cfg.get("sampler", "tpe")
        # Optuna >= 5: each trial reports its constraints (Trial.set_constraint), which the samplers read; before that, a constraints_func
        per_trial = hasattr(optuna.trial.Trial, "set_constraint")
        cfun = (lambda ft: ft.user_attrs.get("constraints") or [0.0]) if use_constraints and not per_trial else None
        if name == "random":
            sampler = optuna.samplers.RandomSampler(seed=seed)
        elif name == "nsga2":
            sampler = optuna.samplers.NSGAIISampler(seed=seed, constraints_func=cfun)
        elif name == "grid":
            sampler = optuna.samplers.GridSampler(grid_space(space, task), seed=seed)
        elif name == "es":
            # (1+1)-CMA-ES for constrained optimisation (Arnold & Hansen 2012): local search from the baseline under the constraints (lab.es)
            from .es import ConstrainedOnePlusOneES
            sampler = ConstrainedOnePlusOneES(space, pdefs, directions[0], seed=seed, sigma0=float(cfg.get("es_sigma0", 0.02)))
        else:
            # multivariate TPE (Falkner et al. 2018) models interactions between parameters; group=True for the conditional spaces
            # the random start comes on top of the enqueued trials (baseline, warm start): counted among them, it never ran - on M 31 the finishing
            # sampler then proposed the final sky level "off", the baseline's, 30 times out of 30
            enq = (1 if cfg.get("baseline", True) else 0) + int(cfg.get("warm_start", 0) or 0)
            sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True, constraints_func=cfun,
                                                 n_startup_trials=min(10, max(3, int(cfg.get("n_trials", 30)) // 4)) + enq)
        storage = f"sqlite:///{os.path.join(study_dir, 'study.db')}"
        study = optuna.create_study(study_name=cfg["name"], storage=storage, directions=directions, sampler=sampler,
                                    load_if_exists=True)
        study.set_metric_names([o["metric"] for o in objectives])
        if cfg.get("baseline", True) and not study.trials and name != "grid":
            base = {k: sp.get("value", pdefs[k]["default"]) for k, sp in space.items() if sp.get("tune") and k in pdefs}
            if base:
                study.enqueue_trial(base, user_attrs={"baseline": True})
                # warm start (``warm_start`` trials): perturbations of the baseline - each numeric setting moved by a normal draw of 5 % of its range, the
                # choices kept - so that the sampler meets the feasible region before its random start wanders the full ranges (finishing on M 31: of 29
                # trials over the full ranges only the baseline was usable)
                rng = np.random.default_rng(seed + 1)
                for _ in range(int(cfg.get("warm_start", 0) or 0)):
                    q = {}
                    for k, v in base.items():
                        sp, pd = space[k], pdefs[k]
                        if pd["type"] in ("float", "int") and isinstance(v, (int, float)) and not isinstance(v, bool):
                            lo, hi = float(sp.get("low", pd["low"])), float(sp.get("high", pd["high"]))
                            x = float(np.clip(v + rng.normal(0, 0.05 * (hi - lo)), lo, hi))
                            q[k] = int(round(x)) if pd["type"] == "int" else x
                        else:
                            q[k] = v
                    study.enqueue_trial(q)
        stretch_path = os.path.join(study_dir, "stretch.json")
        n_trials = int(cfg.get("n_trials", 30))
        status["state"] = "running"
        status["n_trials"] = n_trials
        seen: dict[str, int] = {}
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE and t.user_attrs.get("active_key"):
                seen.setdefault(t.user_attrs["active_key"], t.number)

        def objective(trial):
            if cancel.is_set():
                study.stop()
                raise optuna.TrialPruned("stopped")
            p, active = {}, {}
            for pd in order:
                k = pd["name"]
                sp = space.get(k, {})
                if sp.get("tune") and condition_holds(pd, p):
                    p[k] = suggest(trial, k, sp, pd)
                    active[k] = p[k]
                else:
                    p[k] = sp.get("value", pd["default"])
            status["trial"] = trial.number
            log(f"Trial {trial.number}: " + (", ".join(f"{k}={v}" for k, v in active.items()) or "(no parameter active)"))
            trial.set_user_attr("params_all", _jsonable(p))
            if name == "grid" and cfg.get("baseline", True) and all(str(v) == str(space.get(k, {}).get("value", pdefs[k]["default"])) for k, v in active.items()):
                trial.set_user_attr("baseline", True)          # the grid point at the user's own settings is the baseline to beat
            key = json.dumps(_jsonable(task.canonical(active, ctx) if hasattr(task, "canonical") else active), sort_keys=True)
            trial.set_user_attr("active_key", key)
            t0 = time.time()
            dup = seen.get(key)
            if dup is not None:
                prev = study.trials[dup]
                m = dict(prev.user_attrs.get("metrics") or {})
                trial.set_user_attr("duplicate_of", dup)
                trial.set_user_attr("constraints", prev.user_attrs.get("constraints") or [0.0])
                if use_constraints and per_trial:
                    for k, v in sorted(((m.get("violations") or {}) or {"none": 0.0}).items()):
                        trial.set_constraint(k, float(v))
                src = os.path.join(study_dir, "trials", f"{dup}.jpg")
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(study_dir, "trials", f"{trial.number}.jpg"))
                log(f"Trial {trial.number}: the same active settings as trial {dup} - its result is reused")
            else:
                try:
                    m, img = task.run(ctx, p, log, cancel)
                except Exception as e:
                    if cancel.is_set():
                        study.stop()
                        raise optuna.TrialPruned("stopped")
                    trial.set_user_attr("error", f"{type(e).__name__}: {e}")
                    log(f"Trial {trial.number} failed: {type(e).__name__}: {e}")
                    with open(log_path, "a") as f:
                        f.write(traceback.format_exc())
                    raise
                m = _jsonable(m)
                if use_constraints:
                    from .objectives import violations
                    lim = task.limits(ctx)
                    viol = violations(m, lim)
                    m["violations"] = {k: round(v, 6) for k, v in viol.items()}
                    m["feasible"] = all(v <= 0 for v in viol.values())
                    trial.set_user_attr("constraints", [viol[k] for k in sorted(viol)] or [0.0])
                    if per_trial:
                        for k, v in sorted((viol or {"none": 0.0}).items()):
                            trial.set_constraint(k, float(v))
                    trial.set_user_attr("limits", {k: list(v) for k, v in lim.items()})
                if img is not None:
                    from .tasks import stretch
                    import cv2
                    if getattr(task, "display_ready", False):          # already a finished (sRGB) image
                        im8 = (np.clip(img, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8)
                    else:
                        ref = json.load(open(stretch_path)) if os.path.exists(stretch_path) else None
                        im8, ref = stretch(img, ref)
                        if not os.path.exists(stretch_path):
                            _write_json(stretch_path, ref)
                    s = min(1.0, 900 / max(im8.shape[:2]))
                    if s < 1:
                        im8 = cv2.resize(im8, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(study_dir, "trials", f"{trial.number}.jpg"), im8, [cv2.IMWRITE_JPEG_QUALITY, 90])
            trial.set_user_attr("metrics", m)
            trial.set_user_attr("wall_seconds", time.time() - t0)
            vals = []
            for o in objectives:
                v = m.get(o["metric"])
                if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                    raise RuntimeError(f"metric {o['metric']} is not available for this trial")
                vals.append(float(v))
            if dup is None:
                seen.setdefault(key, trial.number)
            feas = "" if m.get("feasible", True) else "  (breaks: " + ", ".join(k for k, v in (m.get("violations") or {}).items() if v > 0) + ")"
            log(f"Trial {trial.number} done: " + ", ".join(f"{o['metric']} = {v:.5g}" for o, v in zip(objectives, vals)) + feas)
            return vals if len(vals) > 1 else vals[0]

        # exactly n_trials trials (the user's number per stage); a trial whose settings repeat an evaluated one returns that result at no cost
        timeout = float(cfg.get("timeout_min") or 0) * 60 or None
        done = sum(1 for t in study.trials if t.state.is_finished())
        study.optimize(objective, n_trials=max(0, n_trials - done), timeout=timeout, catch=(Exception,), gc_after_trial=True)
        try:
            best = best_record(study, cfg, task)
            _write_json(os.path.join(study_dir, "best.json"), best)
            status["best"] = {"trial": best["trial"], "criterion": best["criterion"], "values": best["values"]}
            log(f"Best: trial {best['trial']} ({best['criterion']})")
        except ValueError:
            status["best"] = None
        status["state"] = "stopped" if cancel.is_set() else "done"
        log("Study stopped" if cancel.is_set() else "Study finished")
    except Exception as e:
        status["state"] = "error"
        status["error"] = f"{type(e).__name__}: {e}"
        with open(log_path, "a") as f:
            f.write(traceback.format_exc())
        log(f"Error: {type(e).__name__}: {e}")
    finally:
        status["ended"] = time.time()
        _write_json(status_path, status)
    if outer["error"] is not None:
        raise outer["error"]
    return status


def main():
    run_study(sys.argv[1], install_signals=True)


if __name__ == "__main__":
    main()
