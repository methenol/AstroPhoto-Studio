"""The render's stretch, run by the community tools themselves.

The user's rule: the stretch is not a numpy re-implementation but Siril's own commands and the siril-scripts Python script that has a headless mode.  Methods
(``params["stretch_method"]``; the sidecar records the exact command lines, so a stretch can be repeated by hand in Siril):

=============  ===========================================================================================================================================
``ghs``        Siril ``ght -D= -B= -SP= -LP= -HP= -clipmode=clip -independent``: Generalised Hyperbolic Stretch (Payne & Cranfield, https://ghsastro.co.uk,
               Siril tutorial https://siril.org/tutorials/ghs/).  D, B, SP, LP are fitted from the image (``finish.ghs_fit_noise_floor``): SP above the sky's
               noise, the sky on its aim, the nebula's 90th percentile on its aim, the lowest slope at the sky among the curves that meet both (noise floor).
``autoghs``    Siril ``autoghs -linked <k> <D> -b= -hp= -lp=`` (SP = median + k sigma), D fitted to the nebula aim.
``modasinh``   Siril ``modasinh -D= -LP= -SP= -HP= -independent``, SP above the sky's noise, D fitted to the nebula aim.
``autostretch``  Siril ``autostretch -linked <shadowsclip> <targetbg>`` (the GUI's auto-stretch), targetbg fitted so the sky lands on its aim.
``statstretch``  ``pyscript Statistical_Stretch.py -median <T> -sigma 5 -linked [-boost b] [-hdr ...]``: Seti Astro's Statistical Stretch as ported to Siril by
               C. Richard (siril-scripts/processing/Statistical_Stretch.py, the one stretch script of the repository with an argparse mode), T fitted so the sky lands on its aim.
=============  ===========================================================================================================================================

The scripts of VeraLux (HyperMetric Stretch, Curves, ...) have no command-line mode: they open a PyQt6 window and block in ``app.exec()`` (tested in this image: no
output, the process has to be killed), so they cannot be part of a headless pipeline and are not used (research/pro-finishing.md section 6).

When Siril is not available (a developer machine) ``ghs`` falls back to ``finish.ghs`` (a numpy mirror of ``ght``, identical to 1e-6, checked in the tests), every other
method falls back to ``ghs`` fitted on the same aims; the record says so.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import cv2
import numpy as np

from . import finish, fitsio
from .siril import SirilError, SirilRunner

METHODS = ("ghs", "autoghs", "modasinh", "autostretch", "statstretch", "hms")
TRIAL_SIDE = 900                    # px: the trial stretches of the non-analytic methods run on a copy reduced to this long side
TRIAL_POINTS = 9                    # values of the method's parameter tried per Siril run
AUTOGHS_K = 0.0                     # autoghs shadowsclip (SP = median + k sigma of each channel)
AUTOGHS_B, AUTOGHS_HP = 13.0, 0.7   # Siril's own defaults (siril command reference)
AUTOSTRETCH_CLIP = -2.8             # autostretch's own default shadowsclip
STAT_SIGMA = 5.0                    # Statistical_Stretch.py's default black point sigma
STAT_HDR_KNEE = 0.75                # Statistical_Stretch.py's default HDR knee
HMS_SCRIPT = "VeraLux/VeraLux_HyperMetric_Stretch.py"      # siril-scripts, Riccardo Paterniti, GPL-3.0-or-later (v1.5.2 read)
HMS_ENTRY = "process_veralux_v6"                          # the function the script's PROCESS button calls (GUI-free numpy)
HMS_B = 6.0                                               # "Protect highlights b": the GUI's default
HMS_CONVERGENCE = 3.5                                     # "Color convergence": the GUI's default (only matters for colour images)
HMS_PROFILE = "Rec.709 (Recommended)"
MODASINH_SP_SIGMAS = 3.0            # modasinh: SP this many (band) noise sigmas above the sky (GHS tutorial: SP above the noise you want to hide)


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
        pixels the fit calls signal (same masks as ``finish.ghs_stretch`` measured the aims with)."""
        sky_in, noise = ctx["sky"], max(ctx["noise"], 1e-4)
        m_sky = np.abs(in_lum - sky_in) <= noise
        m_sig = in_lum > sky_in + finish.SIGNAL_SIGMA * noise
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
            return [f"autoghs -linked {AUTOGHS_K:g} {value:.6f} -b={AUTOGHS_B:g} -hp={AUTOGHS_HP:g} -lp=0 -clipmode=clip"]
        if method == "modasinh":
            sp = float(np.clip(ctx["sky"] + MODASINH_SP_SIGMAS * ctx.get("band", ctx["noise"]), 0.001, 0.9))
            return [f"modasinh -D={value:.6f} -LP=0 -SP={sp:.6f} -HP=1 -clipmode=clip -independent"]
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

    @staticmethod
    def ght_line(fit: dict) -> str:
        """``ght`` for a ``finish.ghs_fit`` result (Siril's -D is ln(D + 1))."""
        return (f"ght -D={np.log1p(fit['D']):.6f} -B={fit['b']:g} -SP={fit['sp']:.6f} -LP={fit.get('lp', 0.0):.6f} -HP={fit.get('hp', 1.0):.6f} "
                f"-clipmode=clip -independent")

    # ------------------------------------------------------------------ the fit of the non-analytic methods (trials on a reduced copy)
    def _fit_value(self, method: str, ctx: dict, aims: dict) -> tuple[float | None, dict]:
        """The free parameter of ``method`` that lands the sky on ``aims['sky']`` (autostretch / statstretch: the median-based ones) or the nebula's 90th percentile
        above the sky on ``aims['neb'] - aims['sky']`` (autoghs / modasinh): two Siril runs of ``TRIAL_POINTS`` trials each (coarse log grid, then a fine one) on a copy
        of the lightness reduced to ``TRIAL_SIDE``."""
        lo, hi = {"autoghs": (0.05, 10.0), "modasinh": (0.05, 10.0), "autostretch": (0.01, 0.9), "statstretch": (0.02, 0.9), "hms": (0.3, 6.0)}[method]
        in_lum = _reduce(ctx["lum"], TRIAL_SIDE)
        ctx_t = {**ctx, "lum": in_lum}
        mono = [np.clip(in_lum, 0.0, 1.0).astype(np.float32)]
        by_sky = method in ("autostretch", "statstretch", "hms")
        trials: list[tuple[float, float, float]] = []
        for rnd in range(2):
            vals = list(np.exp(np.linspace(np.log(lo), np.log(hi), TRIAL_POINTS)))
            if method == "hms":
                outs = [[o] for o in self._run_worker(mono[0], [self.hms_kwargs(v) for v in vals], f"fit{rnd}")]
            else:
                outs = self._run(mono, [self.command(method, v, ctx_t) for v in vals], f"fit{rnd}", must_log=None)
            for v, o in zip(vals, outs):
                sky, neb = self._levels(o[0], in_lum, ctx_t)
                trials.append((v, sky, neb))
            ys = np.array([(t[1] if by_sky else t[2] - t[1]) for t in trials[-TRIAL_POINTS:]])
            target = aims["sky"] if by_sky else aims["neb"] - aims["sky"]
            xs = np.array(vals)
            order = np.argsort(ys)
            if ys.max() < target or ys.min() > target:
                best = float(xs[int(np.argmin(np.abs(ys - target)))])
                lo, hi = best / 1.5, best * 1.5
                continue
            v = float(np.exp(np.interp(target, ys[order], np.log(xs[order]))))
            lo, hi = v / 1.12, v * 1.12
        fine = trials[-TRIAL_POINTS:]
        ys = np.array([(t[1] if by_sky else t[2] - t[1]) for t in fine])
        xs = np.array([t[0] for t in fine])
        order = np.argsort(ys)
        value = float(np.exp(np.interp(target, ys[order], np.log(xs[order]))))
        return value, {"trials": len(trials), "target": round(float(target), 4), "range": [round(float(xs.min()), 5), round(float(xs.max()), 5)]}

    # ------------------------------------------------------------------ the public entry point
    def stretch(self, us: list[np.ndarray], ctx: dict) -> tuple[list[np.ndarray], dict]:
        """Stretch the planes ``us`` (anchored, scaled to [0, 1]) with ``ctx['method']``.  ``ctx`` carries the image statistics the fit needs (``finish.ghs_stretch``):
        ``sky``, ``noise``, ``band`` (the sky's 32-128 px noise), ``neb``, ``bright``, ``lum`` (the lightness in the same units), ``sky_aim``, ``neb_aims``, ``b_max``,
        ``hdr``, ``boost``.  Returns the stretched planes and the record (method, tool, exact commands, fitted parameters, notes)."""
        method = ctx.get("method", "ghs")
        t0 = time.time()
        rec: dict = {"requested": method}
        aims = {"sky": ctx["sky_aim"], "neb": None}
        # 1. parameters
        fit = finish.ghs_fit_noise_floor(ctx["sky"], ctx["neb"], ctx.get("bright"), ctx.get("band", ctx["noise"]), ctx["sky_aim"], tuple(ctx["neb_aims"]), ctx["b_max"],
                                         finish.SKY_NOISE_LIMIT, focus=ctx.get("focus"))
        aims["neb"] = fit["neb_aim"]
        rec["ghs_fit"] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in fit.items()}
        lines: list[str] | None = None
        used = "ghs"
        if not self.allow:
            rec["note"] = "Siril not available: numpy mirror of ght (finish.ghs)" + ("" if method == "ghs" else f"; {method} not run, ght used instead")
            outs = [finish.ghs(np.clip(u, 0, 1), fit["b"], fit["D"], fit["sp"], fit["lp"], fit["hp"]) for u in us]
            rec.update(method="ghs", tool="numpy mirror of siril ght", commands=[self.ght_line(fit)], seconds=round(time.time() - t0, 1))
            return outs, rec
        cache = self._cache_key(method, ctx, aims)
        if method != "ghs":
            try:
                cached = self._cache_get(cache)
                if cached is None:
                    value, fi = self._fit_value(method, ctx, aims)
                    self._cache_put(cache, {"value": value, "fit": fi})
                else:
                    value, fi = cached["value"], cached["fit"]
                lines = self.command(method, value, ctx)
                used = method
                rec["trial_fit"] = fi
                rec["value"] = round(value, 6)
            except Exception as e:                       # noqa: BLE001  a failing community tool must not stop the render: ght takes over, the record says why
                if e.__class__.__name__ == "Cancelled":
                    raise
                rec["fallback"] = f"{method} failed ({str(e).splitlines()[0][:200]}): ght used"
                lines, used = None, "ghs"
        if lines is None:
            lines = [self.ght_line(fit)]
        # 2. the stretch of the full planes
        try:
            if used == "hms":
                out = [self._run_worker(us[0], [self.hms_kwargs(rec["value"])], "apply")[0]]
            else:
                out = self._run(us, [lines], "apply", must_log="Statistical stretch applied" if used == "statstretch" else None)[0]
            if used == "statstretch" and all(float(np.abs(o - np.clip(u, 0, 1)).max()) < 1e-6 for o, u in zip(out, us)):
                raise SirilError("Statistical_Stretch.py left the image unchanged")
        except Exception as e:                           # noqa: BLE001
            if e.__class__.__name__ == "Cancelled":
                raise
            rec["fallback"] = f"{used} failed ({str(e).splitlines()[0][:200]}): numpy mirror of ght used"
            out = [finish.ghs(np.clip(u, 0, 1), fit["b"], fit["D"], fit["sp"], fit["lp"], fit["hp"]) for u in us]
            lines, used = [self.ght_line(fit)], "ghs"
            rec.update(method="ghs", tool="numpy mirror of siril ght", commands=lines, seconds=round(time.time() - t0, 1))
            return [np.clip(o, 0, 1) for o in out], rec
        rec.update(method=used, commands=lines, seconds=round(time.time() - t0, 1),
                   tool={"ghs": "siril ght", "autoghs": "siril autoghs", "modasinh": "siril modasinh", "autostretch": "siril autostretch",
                         "statstretch": "siril pyscript Statistical_Stretch.py (Seti Astro, C. Richard)",
                         "hms": "VeraLux HyperMetric Stretch (siril-scripts, process_veralux_v6 headless)"}[used])
        return [np.clip(o, 0, 1) for o in out], rec

    # ------------------------------------------------------------------ fit cache (the preview and the export of one image fit once)
    def _cache_key(self, method: str, ctx: dict, aims: dict) -> str:
        lum = ctx["lum"]
        sample = np.ascontiguousarray(lum[::37, ::37]).tobytes()
        h = hashlib.sha1(sample + json.dumps({k: (round(float(ctx[k]), 6) if isinstance(ctx.get(k), (int, float)) else ctx.get(k)) for k in
                                              ("sky", "noise", "band", "neb", "boost", "hdr", "sky_aim")}, sort_keys=True).encode() + method.encode()
                         + json.dumps(aims, sort_keys=True).encode())
        return h.hexdigest()[:20]

    def _cache_get(self, key: str):
        try:
            with open(os.path.join(self.work, "stretch_fit_cache.json"), encoding="utf-8") as f:
                return json.load(f).get(key)
        except (OSError, ValueError):
            return None

    def _cache_put(self, key: str, doc: dict) -> None:
        path = os.path.join(self.work, "stretch_fit_cache.json")
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            d = {}
        d[key] = doc
        if len(d) > 40:
            d.pop(next(iter(d)))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f)
