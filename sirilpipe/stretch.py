"""The alternative community stretches of the default chain (``pro.Finisher``), run by the tools themselves through Siril or the pyscript virtualenv.

``stretch_method`` values other than ``ghs2`` (the GHS two-pass recipe, which ``pro.py`` runs as Siril ``ght`` + ``linstretch`` commands): ``statstretch`` (``pyscript
Statistical_Stretch.py``, called by ``pro.py`` itself), and the four handled here, each with ONE free parameter raised until the background (the sky mode) of the stretched image sits
at 0.10 (Brecher), trials on a reduced copy:

``autoghs``      Siril ``autoghs -linked <k> <D> -b= -hp= -lp=`` (SP = median + k sigma), D free.
``modasinh``     Siril ``modasinh -D= -LP= -SP= -HP= -independent``, D free.
``autostretch``  Siril ``autostretch -linked <shadowsclip> <targetbg>``, targetbg free.
``hms``          VeraLux HyperMetric Stretch ``process_veralux_v6`` (siril-scripts; no command-line mode: its GUI-free core is called by ``community_worker.py``), Log D free.

The exact command lines are recorded in the chain's sidecar.
"""
from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np

from . import fitsio
from .siril import SirilError, SirilRunner

SIGNAL_SIGMA = 4.0                  # "signal" = more than this many sigmas above the sky mode (trial levels)
METHODS = ("ghs2", "autoghs", "modasinh", "autostretch", "statstretch", "hms")
TRIAL_SIDE = 900                    # px: the trial stretches of the non-analytic methods run on a copy reduced to this long side
TRIAL_POINTS = 9                    # values of the method's parameter tried per Siril run
AUTOGHS_K = -1.2                    # SP target in sky-noise sigmas: the left half-maximum of the sky peak (median - 1.18 noise sigma), the GHS tutorial's "just left of the
                                    # peak" that the two-pass recipe uses too; at k 0 the sky sits on SP, where GHS keeps it near 0 whatever D (IC 1805 2026-10-16: dark at D 15).
                                    # Siril's autoghs computes SP = median + shadowsclip x the image's STANDARD DEVIATION (measured 2026-10-25: SP(k) linear in k with slope
                                    # = np.std of the image), which a nebula inflates far above the sky noise (M 42: 160 x): -1.2 put SP at -0.0077, the sky mapped below 0
                                    # and the fit chose D 0.055 (a black render).  autoghs_shadowsclip converts the sky-noise target into the shadowsclip Siril needs.
AUTOGHS_B, AUTOGHS_HP = 13.0, 0.7   # Siril's own defaults (siril command reference)
AUTOGHS_MAX_PASSES = 4              # repeated autoghs passes at most (D 10 x 2 passes put the Heart's sky at 0.116, 4 passes of D 3 at 0.114)
AUTOSTRETCH_CLIP = -2.8             # autostretch's own default shadowsclip
STAT_SIGMA = 5.0                    # Statistical_Stretch.py's default black point sigma
STAT_HDR_KNEE = 0.75                # Statistical_Stretch.py's default HDR knee
HMS_SCRIPT = "VeraLux/VeraLux_HyperMetric_Stretch.py"      # siril-scripts, Riccardo Paterniti, GPL-3.0-or-later (v1.5.2 read)
HMS_ENTRY = "process_veralux_v6"                          # the function the script's PROCESS button calls (GUI-free numpy)
HMS_B = 6.0                                               # "Protect highlights b": the GUI's default
HMS_CONVERGENCE = 3.5                                     # "Color convergence": the GUI's default (only matters for colour images)
HMS_PROFILE = "Rec.709 (Recommended)"
MODASINH_SP_SIGMAS = -1.2           # modasinh: SP this many noise sigmas from the sky mode: the left half-maximum of the sky peak (as autoghs / the GHS recipe); above the sky the
                                    # sky maps below SP and no D within Siril's 0-10 reaches the 0.10 background (IC 1805 2026-10-16)


def autoghs_shadowsclip(lum: np.ndarray, sky: float, noise: float) -> float:
    """The ``shadowsclip`` that puts Siril's autoghs symmetry point (median + shadowsclip x std of the image) on the sky's left half-maximum
    (``sky`` + AUTOGHS_K x ``noise``, never below 0)."""
    x = np.asarray(lum, dtype=np.float32)[::2, ::2]
    med, sd = float(np.median(x)), float(np.std(x))
    sp = max(float(sky) + AUTOGHS_K * float(noise), 0.0)
    return float((sp - med) / sd) if sd > 0 else 0.0


def community_python() -> str | None:
    """The Python of the pyscript virtualenv (sirilpy + PyQt6 installed by Siril on its first ``pyscript``), where the siril-scripts modules can be imported."""
    from . import config
    p = os.path.join(config.home_dir(), ".local", "share", "siril", "venv", "bin", "python")
    return p if os.path.exists(p) else None


def community_script(rel: str) -> str | None:
    from . import config
    p = os.path.join(config.home_dir(), ".local", "share", "siril-scripts", rel)
    return p if os.path.exists(p) else None


def siril_available() -> bool:
    from . import config
    return os.path.exists(config.siril_bin())


def _to_channels(planes: list[np.ndarray]) -> np.ndarray:
    """1 plane -> mono image (2-D), 2 planes -> [a, b, b], 3 planes -> as is."""
    if len(planes) == 1:
        return np.ascontiguousarray(planes[0], dtype=np.float32)
    if len(planes) == 2:
        return np.ascontiguousarray(np.stack([planes[0], planes[1], planes[1]]), dtype=np.float32)
    return np.ascontiguousarray(np.stack(planes), dtype=np.float32)


def _from_channels(arr: np.ndarray, n: int) -> list[np.ndarray]:
    if n == 1:
        return [np.ascontiguousarray(arr[0])]
    return [np.ascontiguousarray(arr[i]) for i in range(n)]


def _reduce(plane: np.ndarray, side: int) -> np.ndarray:
    h, w = plane.shape
    if max(h, w) <= side:
        return plane
    s = side / max(h, w)
    return cv2.resize(np.ascontiguousarray(plane), (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)


class SirilStretcher:
    """Runs a stretch of linear-in-[0,1] planes through ``siril-cli``.

    ``work``      the folder Siril works in (inputs / outputs / scripts / logs go there)
    ``steps``     the post.py step recorder (``sub`` for progress, ``checkpoint`` for cancel), optional
    ``allow``     False = never start Siril (the preview of a machine without it, tests)"""

    def __init__(self, work: str, steps=None, allow: bool = True):
        self.work = work
        self.steps = steps
        self.allow = allow and siril_available()
        self.calls: list[dict] = []
        os.makedirs(work, exist_ok=True)

    # ------------------------------------------------------------------ one Siril run
    def _run(self, planes: list[np.ndarray], blocks: list[list[str]], tag: str, must_log: str | None = None) -> list[list[np.ndarray]]:
        """Load the planes once and, for every block of command lines, ``load`` / commands / ``save``; returns the outputs, one list of planes per block."""
        n = len(planes)
        fin = os.path.join(self.work, f"{tag}_in.fit")
        fitsio.write_rgb(fin, _to_channels(planes), None)
        script, outs = [], []
        for i, lines in enumerate(blocks):
            out = f"{tag}_o{i}"
            outs.append(os.path.join(self.work, out + ".fit"))
            if os.path.exists(outs[-1]):
                os.remove(outs[-1])
            script += [f"load {tag}_in"] + lines + [f"save {out}"]
        runner = SirilRunner(self.work, progress=getattr(self.steps, "sub", None), checkpoint=getattr(self.steps, "checkpoint", None))
        res = runner.run("\n".join(script) + "\n", f"stretch-{tag}", expect=[os.path.basename(o) for o in outs])
        if must_log and not any(must_log in ln for ln in res.tail) and len(blocks) == 1:
            raise SirilError(f"stretch-{tag}: the script did not report '{must_log}' (a failing pyscript does not fail Siril)", res.log_path, "\n".join(res.tail))
        out_planes = [_from_channels(fitsio.read_rgb(o)[0] if n > 1 else fitsio.read_fits(o)[0][None], n) for o in outs]
        for f in [fin] + outs:
            try:
                os.remove(f)
            except OSError:
                pass
        self.calls.append({"tag": tag, "blocks": blocks, "seconds": round(res.seconds, 1), "script": res.script_path})
        return out_planes

    def _run_trials(self, planes: list[np.ndarray], blocks: list[list[str]], tag: str) -> list[list[np.ndarray] | None]:
        """``_run`` for the trial values of a fit, one output per block, None for a trial Siril cannot complete.  A trial past the method's
        usable range can leave an image Siril has no statistics for (autoghs D 2.66 x 3 passes clipped M 42's trial copy flat: "stats failed",
        and the one script of 9 trials - and with it the preview - failed); the batch is then run one trial per script and the failing values
        are left out of the fit.  When every trial fails the error stands."""
        try:
            return self._run(planes, blocks, tag)
        except SirilError as e:
            if len(blocks) == 1:
                raise
            first = e
        outs: list[list[np.ndarray] | None] = []
        for i, b in enumerate(blocks):
            try:
                outs.append(self._run(planes, [b], f"{tag}_t{i}")[0])
            except SirilError:
                outs.append(None)
        if all(o is None for o in outs):
            raise first
        return outs

    def _run_worker(self, plane: np.ndarray, kwargs_list: list[dict], tag: str) -> list[np.ndarray]:
        """Call VeraLux HyperMetric Stretch's ``process_veralux_v6`` (siril-scripts) on a mono plane, once per kwargs dict, in the pyscript virtualenv (no Siril process)."""
        import subprocess
        py, script = community_python(), community_script(HMS_SCRIPT)
        if not py or not script:
            raise SirilError("the pyscript virtualenv or the siril-scripts clone is missing: HyperMetric Stretch cannot run")
        fin = os.path.join(self.work, f"{tag}_in.npy")
        np.save(fin, np.ascontiguousarray(plane, dtype=np.float32))
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "community_worker.py")
        prefix = os.path.join(self.work, f"{tag}_out")
        cmd = [py, worker, "--script", script, "--entry", HMS_ENTRY, "--input", fin, "--output-prefix", prefix, "--kwargs", json.dumps(kwargs_list)]
        t0 = time.time()
        from . import config
        r = subprocess.run(cmd, capture_output=True, text=True, env=config.siril_env({"QT_QPA_PLATFORM": "offscreen"}), timeout=900)
        outs = []
        try:
            if r.returncode != 0:
                raise SirilError(f"HyperMetric Stretch worker failed (exit {r.returncode}): " + (r.stderr or r.stdout).strip().splitlines()[-1][:300])
            for i in range(len(kwargs_list)):
                outs.append(np.load(f"{prefix}_{i}.npy"))
        finally:
            for f in [fin] + [f"{prefix}_{i}.npy" for i in range(len(kwargs_list))]:
                try:
                    os.remove(f)
                except OSError:
                    pass
        self.calls.append({"tag": tag, "worker": cmd, "seconds": round(time.time() - t0, 1)})
        return outs

    @staticmethod
    def hms_kwargs(log_d: float) -> dict:
        return {"log_D": float(log_d), "protect_b": HMS_B, "convergence_power": HMS_CONVERGENCE, "working_space": HMS_PROFILE, "processing_mode": "scientific",
                "color_grip": 1.0, "shadow_convergence": 0.0, "linear_expansion": 0.0, "use_adaptive_anchor": True}

    # ------------------------------------------------------------------ levels of a stretched copy
    @staticmethod
    def _levels(out_lum: np.ndarray, in_lum: np.ndarray, ctx: dict) -> tuple[float, float]:
        """(sky, nebula p90) of a stretched lightness plane, with the pixel sets of the input: the sky = the pixels within the input's noise of its mode, the nebula = the
        pixels the fit calls signal (the sky and the subject of the trial image)."""
        sky_in, noise = ctx["sky"], max(ctx["noise"], 1e-4)
        m_sky = np.abs(in_lum - sky_in) <= noise
        m_sig = in_lum > sky_in + SIGNAL_SIGMA * noise
        sky = float(np.median(out_lum[m_sky])) if m_sky.sum() > 50 else float(np.median(out_lum))
        neb = float(np.percentile(out_lum[m_sig], 90)) if m_sig.sum() > 200 else float(np.percentile(out_lum, 99.0))
        return sky, neb

    # ------------------------------------------------------------------ method -> command lines
    def command(self, method: str, value: float | None, ctx: dict) -> list[str]:
        """The Siril script lines of ``method`` at the trial / fitted ``value`` of its free parameter."""
        if method == "hms":
            return [f"# VeraLux HyperMetric Stretch ({HMS_SCRIPT}, v1.5.2), GUI settings: processing mode Scientific, sensor profile {HMS_PROFILE}, Log D {value:.3f}, Protect highlights b {HMS_B:g}, "
                    f"Color convergence {HMS_CONVERGENCE:g}, Color grip 1, Shadow convergence 0, Linear expansion 0, Adaptive anchor on; called headless as "
                    f"community_worker.py --entry {HMS_ENTRY} --kwargs '{json.dumps(self.hms_kwargs(value))}'"]
        if method == "autoghs":
            # autoghs's stretchamount is a linear D of at most 10: one pass lifts a black-pointed deep-sky stack by ~11x at best (IC 1805 2026-10-16: sky 0.005 at D 10),
            # so the command is repeated ``passes`` times with the same D (GHS practice: several moderate stretches), the pass count chosen by the fit
            d = float(np.clip(np.nan_to_num(float(value), nan=1.0), 1e-3, 9.999999))      # Siril: 0 < D <= 10 (see modasinh)
            k = float(ctx.get("autoghs_k", AUTOGHS_K))
            return [f"autoghs -linked {k:.6f} {d:.6f} -b={AUTOGHS_B:g} -hp={AUTOGHS_HP:g} -lp=0 -clipmode=clip"] * int(ctx.get("passes", 1))
        if method == "modasinh":
            sp = float(np.clip(ctx["sky"] + MODASINH_SP_SIGMAS * ctx.get("band", ctx["noise"]), 0.0, 0.9))
            # Siril accepts 0 < D <= 10 only: a value printed as 10.000001 (the grid's end in floating point) or a NaN
            # from a trial fit that saw no sky stopped the whole script ("D must be > 0.0 and <= 10.0", IC 1805)
            d = float(np.clip(np.nan_to_num(float(value), nan=1.0), 1e-3, 10.0))
            return [f"modasinh -D={min(d, 9.999999):.6f} -LP=0 -SP={sp:.6f} -HP=1 -clipmode=clip -independent"]
        if method == "autostretch":
            return [f"autostretch -linked {AUTOSTRETCH_CLIP:g} {value:.6f}"]
        if method == "statstretch":
            line = f"pyscript Statistical_Stretch.py -median {value:.6f} -sigma {STAT_SIGMA:g} -linked"
            if float(ctx.get("boost", 0.0)) > 0:
                line += f" -boost {float(ctx['boost']):.4f}"
            if float(ctx.get("hdr", 0.0)) > 0:
                line += f" -hdr -hdramount {float(np.clip(ctx['hdr'] / 1.5, 0.0, 1.0)):.4f} -hdrknee {STAT_HDR_KNEE:g}"
            return [line]
        raise ValueError(method)

    # ------------------------------------------------------------------ the fit of the non-analytic methods (trials on a reduced copy)
    def _fit_value(self, method: str, ctx: dict, aims: dict) -> tuple[float | None, dict]:
        """The free parameter of ``method`` that lands the sky on ``aims['sky']`` (autostretch / statstretch: the median-based ones) or the nebula's 90th percentile
        above the sky on ``aims['neb'] - aims['sky']`` (autoghs / modasinh): two Siril runs of ``TRIAL_POINTS`` trials each (coarse log grid, then a fine one) on a copy
        of the lightness reduced to ``TRIAL_SIDE``."""
        lo_min, hi_max = {"autoghs": (0.05, 10.0), "modasinh": (0.05, 10.0), "autostretch": (0.01, 0.9), "statstretch": (0.02, 0.9), "hms": (0.3, 6.0)}[method]
        in_lum = _reduce(ctx["lum"], TRIAL_SIDE)                      # lo_min / hi_max: the tool's own limits (modasinh / autoghs: Siril rejects D > 10)
        if method == "autoghs":                                        # SP from the statistics of the image Siril is given (trial copy / full planes)
            ctx["autoghs_k"] = autoghs_shadowsclip(ctx["lum"], ctx["sky"], ctx["noise"])
        ctx_t = {**ctx, "lum": in_lum}
        if method == "autoghs":
            ctx_t["autoghs_k"] = autoghs_shadowsclip(np.clip(in_lum, 0.0, 1.0), ctx["sky"], ctx["noise"])
        mono = [np.clip(in_lum, 0.0, 1.0).astype(np.float32)]
        by_sky = method in ("autostretch", "statstretch", "hms") or aims.get("neb") is None
        target = aims["sky"] if by_sky else aims["neb"] - aims["sky"]
        trials: list[tuple[float, float, float]] = []
        for passes in (range(1, AUTOGHS_MAX_PASSES + 1) if method == "autoghs" else (1,)):
            ctx_t["passes"] = ctx["passes"] = passes                  # the apply repeats the command as often as the fit did
            lo, hi = lo_min, hi_max
            reachable = True
            for rnd in range(2):
                vals = list(np.exp(np.linspace(np.log(lo), np.log(hi), TRIAL_POINTS)))
                if method == "hms":
                    outs = [[o] for o in self._run_worker(mono[0], [self.hms_kwargs(v) for v in vals], f"fit{rnd}")]
                else:
                    outs = self._run_trials(mono, [self.command(method, v, ctx_t) for v in vals], f"fit{rnd}_p{passes}")
                for v, o in zip(vals, outs):
                    valid = o is not None and float(o[0].min()) >= -1e-4 and float(o[0].max()) <= 1.0 + 1e-4     # a stretch maps [0, 1] into [0, 1]
                    sky, neb = self._levels(o[0], in_lum, ctx_t) if valid else (float("nan"), float("nan"))
                    trials.append((v, sky, neb))
                ys = np.array([(t[1] if by_sky else t[2] - t[1]) for t in trials[-TRIAL_POINTS:]])
                fin = np.isfinite(ys)                                 # a trial Siril could not complete has no level
                if not fin.any():
                    raise SirilError(f"{method}: no trial value between {vals[0]:.4g} and {vals[-1]:.4g} gave a valid stretch ({passes} pass(es))")
                failed_above = np.array(vals)[~fin & (np.array(vals) > max(np.array(vals)[fin]))] if fin.any() else np.array([])
                xs, ys = np.array(vals)[fin], ys[fin]
                order = np.argsort(ys)
                if ys.max() < target and failed_above.size:
                    # the values past the strongest usable one failed by overshooting (an image clipped flat): the target lies between them
                    lo, hi = float(xs.max()), float(failed_above.min())
                    continue
                if ys.max() < target or ys.min() > target:
                    reachable = reachable and not (rnd == 0 and ys.max() < target)    # the strongest value of the tool stays short: one more pass (autoghs)
                    best = float(xs[int(np.argmin(np.abs(ys - target)))])
                    lo, hi = max(lo_min, best / 1.5), min(hi_max, best * 1.5)
                    continue
                v = float(np.exp(np.interp(target, ys[order], np.log(xs[order]))))
                lo, hi = max(lo_min, v / 1.12), min(hi_max, v * 1.12)
            if reachable or method != "autoghs":
                break
        fine = trials[-TRIAL_POINTS:]
        ys = np.array([(t[1] if by_sky else t[2] - t[1]) for t in fine])
        xs = np.array([t[0] for t in fine])
        order = np.argsort(ys)
        fin = np.isfinite(ys)
        if fin.sum() >= 2:
            o2 = np.argsort(ys[fin])
            value = float(np.exp(np.interp(target, ys[fin][o2], np.log(xs[fin][o2]))))
        else:                                                     # no usable trial: the middle of the tool's range
            value = float(np.sqrt(lo_min * hi_max))
        value = float(np.clip(value, lo_min, hi_max))
        return value, {"trials": len(trials), "target": round(float(target), 4), "range": [round(float(xs.min()), 5), round(float(xs.max()), 5)], "passes": int(ctx.get("passes", 1))}

    def stretch_to_sky(self, us: list[np.ndarray], ctx: dict, method: str) -> tuple[list[np.ndarray], dict]:
        """The alternative community stretches of the default chain (``autoghs``, ``modasinh``, ``autostretch``, ``hms``): ONE free parameter of the tool, raised until the
        sky mode of the stretched image lands on ``ctx['sky_aim']`` (the Brecher background 0.10; trials on a reduced copy, then the full planes).  ``ctx``: ``lum`` (linear
        lightness), ``sky`` (its mode), ``noise`` (its pixel sigma).  No GHS fit, no nebula aim."""
        t0 = time.time()
        ctx = {"band": ctx.get("noise"), **ctx}
        value, fi = self._fit_value(method, ctx, {"sky": ctx["sky_aim"], "neb": None})
        lines = self.command(method, value, ctx)
        if method == "hms":
            res = self._run_worker(np.stack(us), [self.hms_kwargs(value)], "apply")[0]
            res = np.moveaxis(res, -1, 0) if res.ndim == 3 and res.shape[-1] == 3 and res.shape[0] != 3 else res
            out = [res[i] for i in range(3)]
        else:
            out = self._run(us, [lines], "apply")[0]
        rec = {"method": method, "commands": lines, "value": round(value, 6), "trial_fit": fi, "passes": int(ctx.get("passes", 1)), "seconds": round(time.time() - t0, 1),
               "tool": {"autoghs": "siril autoghs", "modasinh": "siril modasinh", "autostretch": "siril autostretch",
                        "hms": "VeraLux HyperMetric Stretch (siril-scripts, process_veralux_v6 headless)"}[method]}
        return [np.clip(o, 0, 1) for o in out], rec
