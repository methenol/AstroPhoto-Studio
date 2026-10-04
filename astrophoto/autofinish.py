"""Auto-finish: the last pipeline step.  It tunes the processing settings (the sliders of the
Process tab) and fits a colour grade so that the finished image looks like professional images of
the same target.

What "looks like" means is measured, not guessed: ``look_stats`` describes a finished (display-
referred, sRGB) image by statistics that hardly depend on the framing - the sky's level, colour
and grain, the object's tonal range, structure contrast, colour distribution (OKLab chroma and a
hue histogram) and how much of the frame the stars take.  The same statistics were measured on
reference astrophotographs of each target (freely licensed images on Wikimedia Commons, see
``data/autofinish_refs.json`` for every source and ``experiments/autofinish_refs.py`` to rebuild
it).  The objective is the distance to the *nearest* reference (a soft minimum over them): the
references of one target differ in style (pink or crimson Ha, blue or teal OIII, a dark or a
lifted sky), and the average of several styles is none of them - a muddy compromise.

1. The settings are searched on a small render (``PROXY`` px) by a coordinate pattern search; for
   dual-band data the palette is chosen first.  Penalties keep it from clipping highlights,
   crushing the sky to black or bringing up the noise.
2. The colour grade (postprocess.color_grade: white balance of the object, hue and chroma of the
   warm and the cool hues, S-curve) is fitted on that render with the settings found.

A target without references of its own uses those of its class (emission nebula, supernova
remnant, planetary nebula, galaxy, dark nebula), and without a known class all the dual-band or all
the broadband references.
"""
from __future__ import annotations

import json
import math
import os
import re
import time

import cv2
import numpy as np

from .postprocess import DEFAULTS, GRADE_KEYS, color_grade, is_narrowband, nonlinear_stage, rgb_to_oklab

REF_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "autofinish_refs.json")
SIZE = 800                 # long side the statistics are measured at (references and renders alike)
PROXY = 1000               # render size of the search
N_HUE = 18                 # hue histogram bins (20 deg)

# settings the search tunes: (key, low, high, first step).  Noise reduction and sharpening are not:
# their effect is below the resolution of the proxy render
SEARCH = [
    ("stretch", 0.06, 0.30, 0.03),
    ("contrast", 0.0, 8.0, 1.0),
    ("hdr", 0.0, 1.5, 0.25),
    ("saturation", 0.8, 2.6, 0.25),
    ("local_contrast", 0.0, 1.6, 0.25),
    ("black_point", 0.0, 0.10, 0.015),
    ("brightness", -0.6, 0.6, 0.15),
    ("star_intensity", 0.5, 1.2, 0.15),
    ("star_reduction", 0.0, 0.6, 0.15),
    ("oiii_boost", 0.7, 2.5, 0.25),        # dual-band palettes only
]
GRADE_SEARCH = [
    # (a gentle white balance only: wider, the fit tinted a desaturated image into the references' hues -
    # M 42 went to saturation 0.8 with temperature and tint at -1, the hue histogram matched, the colour gone)
    ("grade_temperature", -0.4, 0.4),
    ("grade_tint", -0.4, 0.4),
    ("grade_warm_hue", -1.0, 1.0),
    ("grade_warm_sat", 0.5, 1.8),
    ("grade_cool_hue", -1.0, 1.0),
    ("grade_cool_sat", 0.5, 2.0),
    ("grade_contrast", -0.6, 1.0),
]
NB_PALETTES = ("foraxx", "hoo", "hoo_warm", "natural")

# scalar statistics compared with the references: (key, weight, tolerance, log scale)
TERMS = [
    ("sky_L", 3.0, 0.025, False),        # background brightness
    ("sky_C", 1.0, 0.008, False),        # background colour cast
    ("sig_L50", 1.5, 0.05, False),       # the object's midtones
    ("peak_L", 1.5, 0.06, False),        # its brightest structures (the same whatever the framing)
    ("detail", 1.0, 0.25, True),         # structure (band-pass) contrast on the object
    ("C50", 1.5, 0.015, False),          # the object's colourfulness
    ("C90", 1.0, 0.025, False),
    ("star_C", 0.3, 0.015, False),       # star colour
]
# one-sided: penalised above the reference only (key, weight, tolerance, smallest reference)
CEILINGS = [
    ("star_frac", 1.0, 0.012, 0.01),     # stars cover more of the frame (framing dependent: never matched)
    ("sky_blotch", 4.0, 0.003, 0.004),   # colour mottle of the sky
    ("sky_noise", 3.0, 0.004, 0.006),    # grain of the sky
    ("obj_blotch", 2.0, 0.003, 0.004),   # colour mottle on the object (amplified OIII noise: blue patches in Ha)
]
REGULARISE = 1.5                         # cost of moving a setting across its whole range


# ------------------------------------------------------------------ statistics

def _hue_hist(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    h = (np.degrees(np.arctan2(b, a)) % 360.0) / (360.0 / N_HUE)
    hist = np.bincount(np.minimum(h.astype(int), N_HUE - 1), weights=w, minlength=N_HUE)
    return hist / max(float(hist.sum()), 1e-12)


def hue_distance(h1, h2) -> float:
    """Earth mover's distance between two hue histograms on the circle, in fractions of a half
    turn (0 = the same colours, 1 = every colour moved to its opposite)."""
    c = np.cumsum(np.asarray(h1, float) - np.asarray(h2, float))
    return float(np.abs(c - np.median(c)).sum()) / (N_HUE / 2)


def look_stats(rgb: np.ndarray) -> dict:
    """Framing-independent statistics of a finished sRGB image (0..1), measured at SIZE px."""
    rgb = np.clip(np.asarray(rgb, np.float32), 0, 1)
    f = SIZE / max(rgb.shape[:2])
    if f < 1:
        rgb = cv2.resize(rgb, (round(rgb.shape[1] * f), round(rgb.shape[0] * f)), interpolation=cv2.INTER_AREA)
    lab = rgb_to_oklab(rgb)
    L = np.ascontiguousarray(lab[..., 0])
    # the stars: what a 7 px opening removes; the rest is the object and the sky
    opened = cv2.morphologyEx(L, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    stars = (L - opened) > 0.08
    Ls = cv2.GaussianBlur(opened, (0, 0), 3)
    sky_sel = Ls <= np.percentile(Ls, 20)
    sky_L = float(np.median(Ls[sky_sel]))
    labs = cv2.GaussianBlur(lab, (0, 0), 2)
    Cs = np.hypot(labs[..., 1], labs[..., 2])
    hp = L - cv2.GaussianBlur(L, (0, 0), 1.2)
    quiet = sky_sel & ~cv2.dilate(stars.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    sky_noise = float(1.4826 * np.median(np.abs(hp[quiet]))) if quiet.sum() > 500 else 0.0
    # colour mottle of the sky: the spread of its (smoothed) colour
    ab = labs[..., 1:][quiet] if quiet.sum() > 500 else labs[..., 1:][sky_sel]
    sky_blotch = float(np.hypot(*(1.4826 * np.median(np.abs(ab - np.median(ab, 0)), 0))))
    sig = (Ls > sky_L + 0.06) & ~stars
    sig_frac = float(sig.mean())             # (before the fallback: object_weight reads it)
    if sig.mean() < 0.01:                     # a faint object: its brightest tenth
        sig = (Ls >= np.percentile(Ls, 90)) & ~stars
    dog = cv2.GaussianBlur(opened, (0, 0), 1.5) - cv2.GaussianBlur(opened, (0, 0), 6)
    # hue weights: chroma above a floor, so near-grey pixels (whose hue is noise, or a tint) do not count
    w = np.clip(Cs[sig] - 0.008, 0, None) * np.clip(Ls[sig] - sky_L, 0.005, None)
    # colour mottle on the object: its colour's departure from the colour 6 px around
    mott = labs[..., 1:] - cv2.GaussianBlur(np.ascontiguousarray(labs[..., 1:]), (0, 0), 6)
    obj_blotch = float(np.hypot(*(1.4826 * np.median(np.abs(mott[sig]), 0))))
    return {
        "sky_L": sky_L,
        "sky_C": float(np.median(Cs[sky_sel])),
        "sky_noise": sky_noise,
        "sig_frac": sig_frac,
        "sig_L50": float(np.percentile(Ls[sig], 50)),
        "sig_L90": float(np.percentile(Ls[sig], 90)),
        "peak_L": float(np.percentile(Ls[~stars], 99.5)),
        "sky_blotch": sky_blotch,
        "obj_blotch": obj_blotch,
        "detail": float(np.mean(np.abs(dog[sig]))),
        "C50": float(np.percentile(Cs[sig], 50)),
        "C90": float(np.percentile(Cs[sig], 90)),
        "hue": _hue_hist(labs[..., 1][sig], labs[..., 2][sig], w).tolist(),
        "star_frac": float(stars.mean()),
        "star_C": float(np.median(Cs[stars])) if stars.sum() > 50 else 0.0,
        "clip_hi": float((opened > 0.97).mean()),
        "crush": float((L < 0.012).mean()),
    }


def aggregate(stats: list[dict]) -> dict:
    """References of one target (or class) combined: the median of every statistic, the mean hue
    histogram, and the spread (tolerance) of each statistic among them."""
    out = {"n": len(stats), "spread": {}}
    for k in stats[0]:
        if k == "hue":
            out["hue"] = np.mean([s["hue"] for s in stats], 0).tolist()
            out["spread"]["hue"] = float(np.median([hue_distance(s["hue"], out["hue"]) for s in stats])) if len(stats) > 1 else 0.0
            continue
        v = np.array([s[k] for s in stats], float)
        log = any(t[0] == k and t[3] for t in TERMS)
        if log:
            v = np.log(np.maximum(v, 1e-6))
        med = float(np.median(v))
        out[k] = math.exp(med) if log else med
        out["spread"][k] = float(1.4826 * np.median(np.abs(v - med))) if len(v) > 2 else 0.0
    return out


# ------------------------------------------------------------------ references

_REFS: dict | None = None


def references() -> dict:
    global _REFS
    if _REFS is None:
        _REFS = json.load(open(REF_FILE)) if os.path.exists(REF_FILE) else {"targets": {}, "classes": {}}
    return _REFS


def _norm(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (name or "").upper())


def reference_looks(object_name: str | None, narrowband: bool) -> tuple[list[dict], dict | None, str, list[dict]]:
    """The reference looks for a target: its own references, else those of its class, else the
    dual-band / broadband ones.  Returns (looks, their aggregate, description, sources)."""
    refs = references()
    n = _norm(object_name)
    for t in refs.get("targets", {}).values():
        if n and n in {_norm(a) for a in t["names"]}:
            return t["looks"], t["look"], f"{len(t['looks'])} reference images of {t['label']}", t["sources"]
    cls = refs.get("aliases", {}).get(n)
    if not (cls and cls in refs.get("classes", {})):
        cls = "narrowband" if narrowband else "broadband"
    if cls in refs.get("classes", {}):
        c = refs["classes"][cls]
        return c["looks"], c["look"], f"{len(c['looks'])} reference images of {c['label']}", c.get("sources", [])
    return [], None, "no references", []


# ------------------------------------------------------------------ objective

OBJECT_TERMS = {"sig_L50", "peak_L", "detail", "C50", "C90", "hue", "obj_blotch"}


def object_weight(st: dict) -> float:
    """How much the object statistics count: fully when the object covers 8 % of the frame or more,
    down to a quarter for a small object in a wide field (M 27 in a Seestar frame).  The references
    frame their object closely, and matching its brightness in a wide field only lifts the sky and
    its noise.  Taken from the starting render, so the search cannot lower it."""
    return float(np.clip(st.get("sig_frac", 1.0) / 0.08, 0.25, 1.0))


def score_one(st: dict, ref: dict, obj_w: float = 1.0) -> tuple[float, dict]:
    """Distance of a look to one reference look (lower is better) and its terms."""
    terms = {}
    for k, wgt, tol, log in TERMS:
        if k not in ref:
            continue
        d = (math.log(max(st[k], 1e-6)) - math.log(max(ref[k], 1e-6))) if log else st[k] - ref[k]
        terms[k] = wgt * (d / tol) ** 2 * (obj_w if k in OBJECT_TERMS else 1.0)
    if "hue" in ref:
        # robust: dual-band data cannot always reach an RGB reference's hues, and a quadratic term then
        # outweighed everything else - the cheapest way to lower it was to drain the colour
        terms["hue"] = obj_w * 3.0 * math.log1p((hue_distance(st["hue"], ref["hue"]) / 0.12) ** 2)
    for k, wgt, tol, floor in CEILINGS:
        if k in ref:
            terms[k] = wgt * (max(st[k] - max(ref[k], floor), 0) / tol) ** 2 * (obj_w if k in OBJECT_TERMS else 1.0)
    # quality penalties, whatever the references do
    terms["clip_hi"] = 4.0 * (max(st["clip_hi"] - max(ref.get("clip_hi", 0.0), 0.002), 0) / 0.003) ** 2
    terms["crush"] = 2.0 * (max(st["crush"] - max(ref.get("crush", 0.0), 0.02) - 0.02, 0) / 0.03) ** 2
    return float(sum(terms.values())), terms


def score(st: dict, looks: list[dict], obj_w: float = 1.0, temperature: float = 2.0) -> tuple[float, dict]:
    """Soft minimum of the distances to the reference looks, and the terms of the nearest."""
    res = [score_one(st, r, obj_w) for r in looks]
    v = np.array([r[0] for r in res])
    soft = float(v.min() - temperature * math.log(np.mean(np.exp(-(v - v.min()) / temperature))))
    near = int(v.argmin())
    return soft, {**res[near][1], "_nearest": near}


# ------------------------------------------------------------------ the search

def _bounded(key: str, v: float) -> float:
    for k, lo, hi, _ in SEARCH:
        if k == key:
            return float(min(max(v, lo), hi))
    for k, lo, hi in GRADE_SEARCH:
        if k == key:
            return float(min(max(v, lo), hi))
    return v


def _reg(q: dict, start: dict, space) -> float:
    """Cost of moving the settings away from where the search started."""
    return REGULARISE * sum(((float(q[k]) - float(start[k])) / (s[2] - s[1])) ** 2 for s in space for k in [s[0]])


def fit_grade(img: np.ndarray, looks: list[dict], base: dict, budget: int = 160, obj_w: float = 1.0) -> tuple[dict, float]:
    """The colour grade that brings an ungraded render closest to the references (coordinate
    pattern search on the grade parameters; each step costs one grade + statistics of a small image)."""
    f = 600 / max(img.shape[:2])
    small = cv2.resize(img, (round(img.shape[1] * f), round(img.shape[0] * f)), interpolation=cv2.INTER_AREA) if f < 1 else img
    g = {k: float(base.get(k, DEFAULTS[k])) for k, _, _ in GRADE_SEARCH}
    g["grade_amount"] = 1.0
    g0 = dict(g)

    def cost(gp):
        return score(look_stats(color_grade(small, gp)), looks, obj_w)[0] + _reg(gp, g0, GRADE_SEARCH)
    best = cost(g)
    steps = {k: (hi - lo) / 8 for k, lo, hi in GRADE_SEARCH}
    n = 0
    while n < budget and max(steps.values()) > 0.02:
        improved = False
        for k, _, _ in GRADE_SEARCH:
            for sgn in (1, -1):
                cand = {**g, k: _bounded(k, g[k] + sgn * steps[k])}
                if cand[k] == g[k]:
                    continue
                c = cost(cand)
                n += 1
                if c < best - 1e-4:
                    g, best, improved = cand, c, True
                    break
        if not improved:
            steps = {k: s / 2 for k, s in steps.items()}
    return g, best


def autofinish(session, params: dict | None = None, progress=None, budget: int = 140) -> dict:
    """Tune the processing settings and fit the colour grade of a stacked dataset.  Returns the new
    settings (``params``: the given ones with the tuned keys replaced) and a report.  Also saved as
    ``autofinish.json`` in the session folder."""
    t0 = time.time()
    meta = session.meta
    obj, filt = meta.get("object") or "", meta.get("filter") or ""
    nb = is_narrowband(filt)
    looks, _, ref_desc, sources = reference_looks(obj, nb)
    if not looks:
        raise RuntimeError("Auto-finish has no reference statistics (astrophoto/data/autofinish_refs.json is missing)")
    base = {**DEFAULTS, **(params or {})}
    # the search starts from the defaults of what it tunes (a previous run's grade would bias it)
    p = {**base, **{k: DEFAULTS[k] for k in GRADE_KEYS}, "grade_amount": 1.0}

    def tick(i, n, msg):
        if progress:
            progress(i, n, msg)
    tick(0, budget, "Auto-finish: preparing the preview")
    lin, extra, f, _ = session.render_inputs(p, max_size=PROXY, progress=None)
    extra_keys = {k: v for k, v in extra.items() if k.startswith("_")}
    cache: dict = {}
    count = [0]

    def evaluate(q):
        """The statistics and the image of a render with settings ``q`` (cached)."""
        key = json.dumps({k: q[k] for k in sorted(q) if not k.startswith("_")}, sort_keys=True, default=str)
        if key not in cache:
            img = nonlinear_stage(lin, {**q, **extra_keys}, filt, px_scale=f)
            cache[key] = (look_stats(img), img)
            count[0] += 1
            tick(min(count[0], budget), budget, f"Auto-finish: tuning the settings ({count[0]} renders)")
        return cache[key]

    start_stats = evaluate(base)[0]
    obj_w = object_weight(start_stats)
    start_score = score(start_stats, looks, obj_w)[0]
    p0 = dict(p)

    def cost(q, reg=True):
        return score(evaluate(q)[0], looks, obj_w)[0] + (_reg(q, p0, search) if reg else 0.0)
    # dual-band data: the palette first
    search = [s for s in SEARCH if s[0] != "oiii_boost"]
    if nb:
        scores = {pal: cost({**p, "palette": pal}, reg=False) for pal in NB_PALETTES}
        p["palette"] = min(scores, key=scores.get)
        if p["palette"] != "natural":
            search = SEARCH
    best = cost(p)
    steps = {k: s for k, _, _, s in search}
    while count[0] < budget and any(steps[k] > s0 / 6 for k, _, _, s0 in search):
        improved = False
        for k, lo, hi, s0 in search:
            if count[0] >= budget or steps[k] <= s0 / 6:
                continue
            for sgn in (1, -1):
                cand = {**p, k: round(_bounded(k, float(p[k]) + sgn * steps[k]), 4)}
                if cand[k] == p[k]:
                    continue
                c = cost(cand)
                if c < best - 1e-3:
                    p, best, improved = cand, c, True
                    break
        if not improved:
            steps = {k: v / 2 for k, v in steps.items()}
    tick(budget, budget, "Auto-finish: fitting the colour grade")
    g, _ = fit_grade(evaluate(p)[1], looks, p, obj_w=obj_w)
    p.update({k: round(v, 3) for k, v in g.items()})
    final = look_stats(nonlinear_stage(lin, {**p, **extra_keys}, filt, px_scale=f))
    final_score, final_terms = score(final, looks, obj_w)
    near = final_terms.pop("_nearest")
    tuned = [k for k, *_ in search] + list(GRADE_KEYS) + ["grade_amount"] + (["palette"] if nb else [])
    out_params = {**base, **{k: p[k] for k in tuned}}
    report = {
        "object": obj, "filter": filt, "reference": ref_desc,
        "nearest": sources[near] if near < len(sources) else None,
        "score_before": round(start_score, 3), "score_after": round(final_score, 3),
        "terms_after": {k: round(v, 3) for k, v in final_terms.items()},
        "changed": {k: [base.get(k), out_params[k]] for k in tuned if base.get(k) != out_params[k]},
        "stats_before": _brief(start_stats), "stats_after": _brief(final), "stats_reference": _brief(looks[near]),
        "object_weight": round(obj_w, 2), "renders": count[0], "seconds": round(time.time() - t0, 1),
    }
    res = {"params": {k: out_params[k] for k in DEFAULTS if k in out_params}, "report": report,
           "created": time.time()}
    try:
        with open(os.path.join(session.dir, "autofinish.json"), "w") as fh:
            json.dump(res, fh, indent=1, default=float)
    except OSError:
        pass
    return res


def _brief(st: dict) -> dict:
    return {k: round(float(st[k]), 4) for k, *_ in TERMS + CEILINGS if k in st}
