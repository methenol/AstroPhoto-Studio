"""The hyper-tuned pipeline's pieces: the paired objectives (lab.objectives), the study runner's conditional spaces, constraints and best-trial
rule (lab.run), the hyper-tune search space (sirilpipe.hypertune) and the new stacking options (workflow)."""
from __future__ import annotations

import numpy as np
import pytest

from sirilpipe import hypertune, workflow
from sirilpipe.lab import objectives as OB
from sirilpipe.lab import run as labrun
from sirilpipe.lab.tasks import TASKS, model_structure, trail_pixels


# ----------------------------------------------------------------------------- a synthetic star field
def star_field(seed=0, n=500, size=600, fwhm=3.0, noise=0.002, sky=0.05, hot=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    img = np.full((size, size), sky, np.float32)
    s = fwhm / 2.355
    xs, ys = rng.uniform(20, size - 20, n), rng.uniform(20, size - 20, n)
    fl = 10 ** rng.uniform(-1.5, 0.3, n)
    for x, y, f in zip(xs, ys, fl):
        x0, y0 = int(x), int(y)
        sl = (slice(max(0, y0 - 12), y0 + 13), slice(max(0, x0 - 12), x0 + 13))
        img[sl] += f / (2 * np.pi * s * s) * np.exp(-((xx[sl] - x) ** 2 + (yy[sl] - y) ** 2) / (2 * s * s))
    for _ in range(hot):
        img[rng.integers(20, size - 20), rng.integers(20, size - 20)] += 0.3
    img += rng.normal(0, noise, img.shape).astype(np.float32)
    return np.repeat(img[..., None], 3, -1)


@pytest.fixture(scope="module")
def field():
    return star_field()


@pytest.fixture(scope="module")
def yard(field):
    return OB.Yardstick(field)


def test_yardstick_reference(yard):
    assert len(yard.sx) >= 50 and len(yard.ax) >= 500
    assert 2.5 < yard.fwhm < 4.5                      # 2 x half-light radius of a 3 px Gaussian = 3 px
    r = yard.ref
    assert r["dm_median"] == pytest.approx(0, abs=1e-9) and r["large_change"] == pytest.approx(0, abs=1e-9)
    assert r["snr_point"] > 5 and r["depth"] > 1


def test_depth_rewards_noise_reduction_and_flux_constraint_catches_light_loss(field, yard):
    clean = star_field(noise=0.001)                    # the same sky, half the noise
    m = yard.measure(clean, same_grid=True)
    assert m["depth"] > yard.ref["depth"] + 0.3
    assert abs(m["dm_median"]) < 0.02
    dim = field.copy()
    dim -= 0.05
    dim *= 0.7                                        # 30 % of every star's light gone
    dim += 0.05
    m = yard.measure(dim, same_grid=True)
    assert m["dm_bright_abs"] > 0.3                   # -2.5 log10(0.7) = 0.39 mag


def test_large_change_flags_smearing_but_not_noise_removal(yard):
    import cv2
    clean = star_field(noise=0.0)
    m = yard.measure(clean, same_grid=True)           # all the noise removed: about 1 x the noise at the PSF scale
    assert m["large_change"] < 1.25
    smeared = np.stack([cv2.GaussianBlur(star_field()[..., c], (0, 0), 6.0) for c in range(3)], -1)
    assert yard.measure(smeared, same_grid=True)["large_change"] > yard.measure(clean, same_grid=True)["large_change"]


def test_measure_on_another_grid(field, yard):
    import cv2
    up = np.stack([cv2.resize(field[..., c], None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LANCZOS4) for c in range(3)], -1)[12:, 7:]
    m = yard.measure(up)
    assert m["fwhm"] == pytest.approx(yard.ref["fwhm"], rel=0.15)        # reported in the reference's pixels
    assert m["dm_mad"] < 0.05                                             # the same stars were found


def test_defects_are_not_stars():
    sub = np.zeros((40, 40), np.float32)
    sub[10, 10] = 1.0
    sub[20:22, 20:22] = 1.0                                               # a Bayer-drizzled hot pixel
    yy, xx = np.mgrid[0:40, 0:40]
    sub += np.exp(-((xx - 30) ** 2 + (yy - 30) ** 2) / (2 * (2.3 / 2.355) ** 2)).astype(np.float32)
    w = OB.wing_fraction(sub, [10, 20.5, 30], [10, 20.5, 30])
    assert w[0] < OB.WING_MIN and w[1] < OB.WING_MIN and w[2] > 0.3
    out = OB.Yardstick.shape_and_outliers(star_field(hot=40), 3.0)
    assert out["outliers"] > 50                                           # per Mpx: 40 in 0.36 Mpx (less the border)


def test_violations():
    v = OB.violations({"a": 0.2, "b": 5}, {"a": ("max", 0.1), "b": ("min", 3), "c": ("max", 1)})
    assert v["a"] == pytest.approx(0.1) and v["b"] == -2 and v["c"] == 1.0


def test_trails_and_model_structure(field):
    import cv2
    t0 = trail_pixels(field)
    line = np.zeros(field.shape[:2], np.float32)
    cv2.line(line, (5, 100), (590, 400), 0.006, 2)                       # a faint trail: 3 x the pixel noise
    tr = field + line[..., None]
    assert trail_pixels(tr) > t0 + 300
    valid = np.ones(field.shape[:2], bool)
    yy, xx = np.mgrid[0:600, 0:600] / 600.0
    grad = (0.01 * xx + 0.005 * yy)[..., None].astype(np.float32)
    blob = (0.02 * np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / (2 * 0.05 ** 2)))[..., None].astype(np.float32)
    img = field + grad + blob
    assert model_structure(img, img - grad, valid) < 0.1                  # a smooth gradient model
    assert model_structure(img, img - grad - blob, valid) > 0.5           # a model that followed the nebula


# ----------------------------------------------------------------------------- the study runner
def test_conditions_and_order():
    pd = {"name": "x", "when": [{"a": ["p"]}, {"b": [True]}]}
    assert labrun.condition_holds(pd, {"a": "p"}) and labrun.condition_holds(pd, {"a": "q", "b": True})
    assert not labrun.condition_holds(pd, {"a": "q", "b": False})
    ps = [{"name": "c", "when": {"b": [1]}}, {"name": "b", "when": {"a": [1]}}, {"name": "a"}]
    assert [p["name"] for p in labrun.ordered(ps)] == ["a", "b", "c"]
    # every task's conditions name its own parameters
    for t in TASKS.values():
        names = {p["name"] for p in t.params}
        for p in t.params:
            w = p.get("when")
            for alt in (w if isinstance(w, list) else [w] if w else []):
                assert set(alt) <= names, (t.name, p["name"], alt)


def test_baseline_space_widens_to_the_users_value():
    t = TASKS["stack"]
    sp = labrun.baseline_space({"sigma_low": {"tune": True, "low": 2, "high": 6, "value": 9.0}, "rejection": {"tune": True, "choices": ["auto"], "value": "linear"}}, t)
    assert sp["sigma_low"]["high"] == 9.0 and "linear" in sp["rejection"]["choices"]


def _study(values, cons, baseline=0, direction="maximize"):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(direction=direction)
    for i, (v, c) in enumerate(zip(values, cons)):
        t = optuna.trial.create_trial(params={}, distributions={}, value=v,
                                      user_attrs={"constraints": c, "baseline": i == baseline})
        st.add_trial(t)
    return st


def test_best_trial_feasible_and_min_gain():
    task = TASKS["restore"]                                               # score depth (maximize), min_gain 0.005
    cfg = {"objectives": [{"metric": "depth", "direction": "maximize"}], "task": "restore"}
    st = _study([3.0, 3.5, 3.2], [[0.0], [0.4], [-0.1]])                  # trial 1 is best but breaks a constraint
    t, _ = labrun.best_trial(st, cfg, task)
    assert t.number == 2
    st = _study([3.0, 3.003], [[0.0], [0.0]])                             # better by less than the resolution: the baseline stays
    t, why = labrun.best_trial(st, cfg, task)
    assert t.number == 0 and "baseline kept" in why
    st = _study([19.30, 19.23], [[0.0], [0.0]], direction="minimize")   # a distance: the gain must exceed 2 % of the baseline's
    t, why = labrun.best_trial(st, {"objectives": [{"metric": "finish_score", "direction": "minimize"}]}, TASKS["finish"])
    assert t.number == 0 and "baseline kept" in why
    st = _study([3.0, 3.1], [[0.5], [0.2]])                               # none feasible: the least infeasible
    t, why = labrun.best_trial(st, cfg, task)
    assert t.number == 1 and "least infeasible" in why


# ----------------------------------------------------------------------------- the hyper-tune's space
def test_build_space_prunes_tools_and_keeps_style_choices():
    tools = {k: True for k in ("bxt", "nxt", "sxt", "graxpert_bge", "graxpert_denoise", "graxpert_deconv", "starnet", "cc_sharpen", "cc_denoise", "cc_darkstar", "cc_satellite")}
    tools["bxt"] = tools["nxt"] = False
    sp, notes = hypertune.build_space(TASKS["restore"], {"deconv_method": "bxt"}, {}, dual=False, refs=False, tools=tools)
    assert "bxt" not in sp["deconv_method"]["choices"] and sp["deconv_method"]["value"] != "bxt"
    assert "nxt" not in sp["denoise_method"]["choices"] and notes
    fin, _ = hypertune.build_space(TASKS["finish"], {}, {"palette": "hoo"}, dual=True, refs=False, tools=tools)
    assert not fin["palette"]["tune"] and fin["palette"]["value"] == "hoo"        # style: only against references
    assert fin["synthetic_luminance"]["tune"] and fin["cc_sharpen"]["tune"]
    fin2, _ = hypertune.build_space(TASKS["finish"], {}, {}, dual=True, refs=True, tools=tools)
    assert fin2["palette"]["tune"] and fin2["grade_amount"]["tune"]
    st, _ = hypertune.build_space(TASKS["stack"], {"scale": 2.0, "drizzle_pixfrac": 0.0}, {}, dual=False, refs=False, tools=tools)
    assert st["drizzle_pixfrac"]["value"] == 0.5 and not st["scale"]["tune"]
    assert hypertune.rule_pixfrac({"scale": 2.0, "superres_method": "cosmic_clarity"}) == 1.0


# ----------------------------------------------------------------------------- stacking options
def test_rejection_weighting_and_drizzle_options():
    rs = {"mode": "drizzle", "scale": 1.0}
    assert workflow.rejection({}, rs)["type"] == "w"
    assert workflow.rejection({}, {"mode": "drizzle", "scale": 2.0})["type"] == "a"
    assert workflow.rejection({"rejection": "linear"}, rs) == {"type": "l", "name": "linear", "low": 4.0, "high": 3.0}
    assert workflow.weighting({}) == "wfwhm" and workflow.weighting({"stack_weighting": "none"}) == "" and workflow.weighting({"stack_weighting": "x"}) == "wfwhm"
    out = workflow.decide_resampling({"drizzle_pixfrac": 0.7, "drizzle_kernel": "gaussian"}, 50, 3.0)
    assert out["pixfrac"] == 0.7 and out["kernel"] == "gaussian"
    assert workflow.decide_resampling({}, 50, 3.0)["pixfrac"] == 1.0
    assert workflow.decide_resampling({"mode": "demosaic", "drizzle_pixfrac": 0.7}, 50, 3.0).get("pixfrac") is None
    script = workflow.stack_script("/p", "seq", workflow.rejection({"rejection": "sigma"}, rs), 10, weight=workflow.weighting({"stack_weighting": "noise"}))
    assert "rej s 4 3" in script and "-weight=noise" in script


class _ToyTask:
    """A task whose 'image' is analytic: score = -(x - 0.3)^2 (max at x = 0.3), a constraint x <= 0.25 when mode = 'a'."""
    name, label, score, min_gain, display_ready = "toy", "Toy", "score", 0.0, False
    params = [{"name": "mode", "type": "categorical", "choices": ["a", "b"], "default": "a"},
              {"name": "x", "type": "float", "low": 0.0, "high": 1.0, "default": 0.5, "when": {"mode": ["a"]}},
              {"name": "y", "type": "categorical", "choices": [1, 2], "default": 1, "when": {"mode": ["b"]}}]
    metrics = {"score": {"label": "s", "direction": "maximize"}, "x": {"label": "x", "direction": "minimize"}}
    runs = 0

    def defaults(self):
        return {p["name"]: p["default"] for p in self.params}

    def limits(self, ctx):
        return {"x": ("max", 0.25)}

    def prepare(self, ds, opts, device, log, cancel, study_dir):
        return {}

    def run(self, ctx, p, log, cancel):
        type(self).runs += 1
        x = p["x"] if p["mode"] == "a" else 0.0
        return {"score": -(x - 0.3) ** 2 - (0.2 if p["mode"] == "b" else 0.0) * p["y"], "x": x}, None


def test_run_study_in_process(tmp_path, monkeypatch):
    import json
    import threading
    from sirilpipe.lab import tasks as T
    monkeypatch.setitem(T.TASKS, "toy", _ToyTask())
    _ToyTask.runs = 0
    d = tmp_path / "study"
    d.mkdir()
    cfg = {"name": "toy", "task": "toy", "dataset": {"kind": "real", "folder": str(tmp_path)}, "options": {},
           "space": {"mode": {"tune": True, "value": "a"}, "x": {"tune": True, "low": 0.0, "high": 1.0, "value": 0.5}, "y": {"tune": True, "value": 1}},
           "objectives": [{"metric": "score", "direction": "maximize"}], "sampler": "tpe", "n_trials": 25, "baseline": True, "constraints": True}
    (d / "config.json").write_text(json.dumps(cfg))
    lines = []
    st = labrun.run_study(str(d), on_log=lines.append)
    assert st["state"] == "done"
    best = json.loads((d / "best.json").read_text())
    assert best["feasible"] and best["processing"] == {} and best["not_pipeline"]["mode"] == "a"
    assert best["not_pipeline"]["x"] <= 0.25 and best["not_pipeline"]["x"] > 0.1     # the constrained optimum is at the bound
    assert _ToyTask.runs == 25 and any("reused" in m for m in lines)                 # 25 distinct evaluations; repeats (mode b: 2 settings) reused on top
    # cancelling through the log callback stops the study and re-raises in the caller
    d2 = tmp_path / "s2"
    d2.mkdir()
    (d2 / "config.json").write_text(json.dumps({**cfg, "name": "toy2"}))

    class Stop(Exception):
        pass

    def on_log(m):
        if m.startswith("Trial 2"):
            raise Stop()
    ev = threading.Event()
    with pytest.raises(Stop):
        labrun.run_study(str(d2), cancel=ev, on_log=on_log)
    assert ev.is_set() and json.loads((d2 / "status.json").read_text())["state"] == "stopped"


def test_shift_fallback(field, yard):
    shifted = np.ascontiguousarray(field[7:, 11:])                        # another crop of the same grid
    tf = yard._shift_transform(*(lambda L: (L - np.median(L), OB.nmad(L)))(OB.lum(shifted)))
    p = tf(np.array([[100.0, 100.0]]))
    assert p[0] == pytest.approx([89.0, 93.0], abs=0.3)


def test_depth_rewards_resolution_at_equal_flux_and_noise(yard):
    sharp = star_field(fwhm=2.0)                                          # the same stars and noise, a tighter PSF
    m = yard.measure(sharp, same_grid=True)
    assert m["fwhm"] < yard.ref["fwhm"] * 0.8
    assert m["snr_point"] > yard.ref["snr_point"] * 1.2 and m["depth"] > yard.ref["depth"]
    assert m["dm_bright_abs"] < 0.05                                      # flux kept: feasible


def test_finish_searches_full_ranges():
    sp, _ = hypertune.build_space(TASKS["finish"], {}, {"saturation": 1.45, "black_point": 0.0}, dual=False, refs=False, tools={})
    assert sp["saturation"]["low"] == 1 and sp["saturation"]["high"] == 4 and sp["black_point"]["tune"] and sp["stretch"]["tune"]


def test_registration_grid_space():
    sp, _ = hypertune.build_space(TASKS["stack"], {"mode": "auto", "pattern_correction": True}, {}, dual=False, refs=False, tools={},
                                  only=("mode", "pattern_correction"))
    assert [k for k, v in sp.items() if v.get("tune")] == ["mode", "pattern_correction"]
    grid = labrun.grid_space(labrun.baseline_space(sp, TASKS["stack"]), TASKS["stack"])
    assert grid == {"mode": ["auto", "demosaic"], "pattern_correction": [True, False]}
    st, _ = hypertune.build_space(TASKS["stack"], {}, {}, dual=False, refs=False, tools={}, exclude=("mode", "pattern_correction"))
    assert not st["mode"]["tune"] and st["sigma_low"]["tune"] and st["rejection"]["tune"]


def test_exhausted_space_stops(tmp_path, monkeypatch):
    import json
    from sirilpipe.lab import tasks as T

    class Tiny(_ToyTask):
        params = [{"name": "mode", "type": "categorical", "choices": ["a", "b"], "default": "a"}]
        n = 0

        def run(self, ctx, p, log, cancel):
            type(self).n += 1
            return {"x": 0.0 if p["mode"] == "a" else 1.0, "score": 0.0}, None
    monkeypatch.setitem(T.TASKS, "tiny", Tiny())
    d = tmp_path / "s"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"name": "tiny", "task": "tiny", "dataset": {"kind": "real", "folder": str(tmp_path)}, "space": {"mode": {"tune": True}},
                                               "objectives": [{"metric": "x", "direction": "minimize"}], "n_trials": 60, "constraints": False}))
    st = labrun.run_study(str(d))
    assert st["state"] == "done" and Tiny.n == 2                         # two settings exist: two evaluations, then the space is exhausted


def test_background_canonical_settings():
    t = TASKS["background"]
    on, off = {"skyref": True}, {"skyref": False}
    a = t.canonical({"bg_method": "auto", "bg_smoothing": 0.2, "bg_correction": "divide"}, on)
    b = t.canonical({"bg_method": "reference", "bg_correction": "subtract"}, on)
    assert a == b                                                                     # the survey model, whatever the unused settings
    assert t.canonical({"bg_method": "auto", "bg_smoothing": 0.2, "bg_correction": "divide"}, off) == \
        t.canonical({"bg_method": "graxpert", "bg_smoothing": 0.2, "bg_correction": "divide"}, off)
    assert t.canonical({"bg_method": "rbf", "bg_smoothing": 0.2}, on) == {"bg_method": "rbf", "bg_smoothing": 0.2}


def test_restore_and_stars_canonical():
    t = TASKS["restore"]
    ctx = {"auto_deconv": "bxt", "auto_denoise": "nxt"}
    a = t.canonical({"deconv_method": "auto", "deconv_stars_method": "auto", "deconvolution": 0.7, "denoise_method": "auto", "denoise": 0.9}, ctx)
    b = t.canonical({"deconv_method": "bxt", "deconv_stars_method": "bxt", "deconvolution": 0.2, "denoise_method": "nxt", "denoise": 0.9}, ctx)
    assert a == b
    assert t.canonical({"deconv_method": "rl", "deconv_stars_method": "rl", "deconvolution": 0.2}, ctx)["deconvolution"] == 0.2
    s = TASKS["stars"]
    assert s.canonical({"star_removal": "auto"}, {"auto_stars": "sxt"}) == s.canonical({"star_removal": "sxt"}, {"auto_stars": "sxt"})


def test_warm_start_perturbs_the_baseline(tmp_path, monkeypatch):
    import json
    import optuna
    from sirilpipe.lab import tasks as T
    monkeypatch.setitem(T.TASKS, "toy", _ToyTask())
    d = tmp_path / "w"
    d.mkdir()
    cfg = {"name": "w", "task": "toy", "dataset": {"kind": "real", "folder": str(tmp_path)}, "warm_start": 4,
           "space": {"mode": {"tune": True, "value": "a"}, "x": {"tune": True, "low": 0.0, "high": 1.0, "value": 0.5}, "y": {"tune": True, "value": 1}},
           "objectives": [{"metric": "score", "direction": "maximize"}], "n_trials": 8}
    (d / "config.json").write_text(json.dumps(cfg))
    labrun.run_study(str(d))
    st = optuna.load_study(study_name="w", storage=f"sqlite:///{d / 'study.db'}")
    xs = [t.params["x"] for t in sorted(st.trials, key=lambda t: t.number)[:5]]
    assert xs[0] == 0.5 and all(abs(x - 0.5) < 0.25 for x in xs[1:]) and len(set(xs)) == 5
    assert all(t.params["mode"] == "a" for t in sorted(st.trials, key=lambda t: t.number)[:5])
