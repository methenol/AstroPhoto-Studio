"""Auto-finish: the last pipeline step.  It works through the Process sliders the way a finisher
does - one control at a time, in the professional order, watching one measurement per control
and stopping where the image says stop - and ends with a natural colour grade for the kind of
target, checked against reference photographs of it.

Every decision is a one-dimensional fit of a slider to a measurement of the render itself:

    tonal foundation   stretch        -> the object's midtones (where the references put them)
                       black point    -> the sky's brightness, never crushing its noise to black
                       HDR            -> the brightest extended structure kept off the ceiling
                       midtones       -> the object's midtones re-checked after the black point
    contrast           GHS focus      -> the structure-to-grain ratio on the object, grain bounded
    noise              fine-grain NR  -> the least that brings the sky grain to the references' grain
                       colour NR      -> the least that removes the colour mottle of sky and object
    detail             local contrast -> raised until the grain rises faster than the structure
                       sharpening     -> raised until the fine noise on faint structure rises
    stars              reduction      -> star coverage no more than the references' (a ceiling)
                       brightness     -> the brightest stars just below clipping
                       halo removal   -> until the blue excess round bright stars is gone
    colour             saturation     -> the references' colourfulness, mottle and gamut bounded
                       OIII boost     -> the warm / cool balance of the references (dual-band)
                       SCNR           -> only if the green hues exceed the references'
    grade              hue shifts     -> towards the nearest reference's red and teal / blue hues
                       white balance  -> the star field neutral (broadband)
                       S-curve        -> the references' tonal spread on the object

The targets ("where a professional puts the sky, the midtones, the colour") are not constants
in the code: they are the medians of the same measurements over reference astrophotographs of
the target - freely licensed images on Wikimedia Commons, measured by ``look_stats`` and kept as
statistics in ``data/autofinish_refs.json`` (``experiments/autofinish_refs.py`` rebuilds it).
The limits ("where to stop") come from the image: its own grain, clipping and star coverage.  A
target without references of its own uses those of its class (emission nebula, supernova
remnant, planetary nebula, galaxy, dark nebula); without a known class, all dual-band or all
broadband references.  The result is compared with the nearest reference at the end; that
distance is reported, never optimised.
"""
from __future__ import annotations

import json
import math
import os
import re
import time

import cv2
import numpy as np

from .postprocess import DEFAULTS, GRADE_KEYS, is_narrowband, nonlinear_stage, rgb_to_oklab

REF_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "autofinish_refs.json")
SIZE = 800                 # long side the statistics are measured at (references and renders alike)
PROXY = 1000               # render size of the fits
N_HUE = 18                 # hue histogram bins (20 deg)
NB_PALETTES = ("foraxx", "hoo", "hoo_warm", "natural")

# the sliders Auto-finish sets, with the range it may use and the slider's step (webui PARAM_SPEC)
RANGES = {
    "stretch": (0.05, 0.32, 0.01),
    "black_point": (0.0, 0.15, 0.005),
    "hdr": (0.0, 1.5, 0.05),
    "brightness": (-0.5, 0.5, 0.05),
    "contrast": (0.0, 8.0, 0.25),
    "luminance_denoise": (0.0, 1.5, 0.05),
    "chroma_denoise": (0.0, 1.0, 0.05),
    "local_contrast": (0.0, 1.6, 0.05),
    "sharpen": (0.0, 1.2, 0.05),
    "star_reduction": (0.0, 0.6, 0.05),
    "star_intensity": (0.4, 1.2, 0.05),
    "halo_suppress": (0.0, 1.0, 0.05),
    "saturation": (0.8, 2.6, 0.05),
    "oiii_boost": (0.7, 2.5, 0.05),
    "scnr": (0.0, 1.0, 0.05),
    "grade_temperature": (-0.3, 0.3, 0.05),
    "grade_tint": (-0.3, 0.3, 0.05),
    "grade_warm_hue": (-0.4, 0.4, 0.05),
    "grade_warm_sat": (0.7, 1.5, 0.05),
    "grade_cool_hue": (-0.4, 0.4, 0.05),
    "grade_cool_sat": (0.7, 1.5, 0.05),
    "grade_contrast": (-0.3, 0.3, 0.05),
}
TUNED = list(RANGES) + ["grade_amount", "palette"]

# the rules of thumb a finisher works by: how much grain, clipping and star coverage is acceptable,
# and how far the references pull.  The Experiments tab's "Auto-finish" task tunes them (lab/tasks.py)
RULES = {
    "grain": 1.0,             # acceptable sky grain, x the references' grain (1 = as grainy as they are)
    "grain_floor": 0.003,     # ... but never asked below this (OKLab L at 800 px: a smooth sky) ...
    "grain_max": 0.008,       # ... nor above this: the typical reference's grain (median over all of them)
    "noise_rise": 0.08,       # local contrast / sharpening stop when the grain has risen by this fraction
    "crush": 0.001,           # share of the frame the black point may clip to black
    "clip": 0.001,            # share of the frame that extended structure may clip at white
    "star_clip": 0.0003,      # share of the frame the brightest stars may clip
    "star_cover": 1.0,        # star coverage ceiling, x the references' coverage
    "object_frac": 0.08,      # object targets count fully from this share of the frame ...
    "object_min_weight": 0.3,  # ... and at least this much for a small object in a wide field
    "field_chroma": 0.008,    # the sky stays neutral: the 90th-percentile OKLab chroma of the sky away from
                              # stars may not exceed this (below one just-noticeable difference of tint)
    "mottle_rise": 0.3,       # colour steps may raise the object's colour mottle by this fraction at most
    "colour_clip": 0.005,     # share of the frame where a coloured pixel may sit at a channel's ceiling (flat,
                              # detail-less colour: the red of a bright Ha region at too much saturation)
    "grade": 1.0,             # strength of the colour grade (0 = none)
}

# scalar statistics compared with the references in the final report: (key, weight, tolerance, log scale)
TERMS = [
    ("sky_L", 3.0, 0.025, False), ("sky_C", 1.0, 0.008, False), ("sig_L50", 1.5, 0.05, False),
    ("peak_L", 1.5, 0.06, False), ("detail", 1.0, 0.25, True), ("C50", 1.5, 0.015, False),
    ("C90", 1.0, 0.025, False), ("star_C", 0.3, 0.015, False),
]
CEILINGS = [("star_frac", 1.0, 0.012, 0.01), ("sky_blotch", 4.0, 0.003, 0.004), ("sky_noise", 3.0, 0.004, 0.006),
            ("obj_blotch", 2.0, 0.003, 0.004)]
OBJECT_TERMS = {"sig_L50", "peak_L", "detail", "C50", "C90", "hue", "obj_blotch"}
_WARM, _COOL = 40.0, 220.0           # OKLab hue centres of the grade's warm and cool sectors (degrees)


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


def _bin_centres() -> np.ndarray:
    return (np.arange(N_HUE) + 0.5) * (360.0 / N_HUE)


def sector_stats(hist) -> dict:
    """Weight and circular mean hue of a hue histogram in the grade's warm (red-orange-yellow) and
    cool (green-cyan-blue) sectors, and the weight of the green hues (120-180 deg)."""
    h = np.asarray(hist, float)
    deg = _bin_centres()
    out = {}
    for name, centre in (("warm", _WARM), ("cool", _COOL)):
        w = h * np.clip(np.cos(np.deg2rad(deg - centre)), 0, None) ** 1.5
        tot = float(w.sum())
        ang = math.degrees(math.atan2(float((w * np.sin(np.deg2rad(deg))).sum()),
                                      float((w * np.cos(np.deg2rad(deg))).sum()))) % 360.0 if tot > 1e-9 else centre
        out[name] = tot
        out[name + "_hue"] = ang
    out["green"] = float(h[(deg >= 120) & (deg < 180)].sum())
    return out


def look_stats(rgb: np.ndarray) -> dict:
    """Framing-independent statistics of a finished sRGB image (0..1), measured at SIZE px.  The
    same function measures the reference photographs (experiments/autofinish_refs.py)."""
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
    stars_d = cv2.dilate(stars.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    quiet = sky_sel & ~stars_d
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
    st = {
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
    # --- what the finisher watches while turning the knobs (not in the references)
    # structure against grain: the band-pass amplitude on the object over the same band-pass on the sky
    dog_sky = float(1.4826 * np.median(np.abs(dog[quiet]))) if quiet.sum() > 500 else 0.0
    st["detail_snr"] = st["detail"] / max(dog_sky, 1e-5)
    # fine noise on the faint part of the object: what sharpening raises first
    faint = sig & (Ls < sky_L + 0.2) & ~stars_d
    st["fine_noise"] = float(1.4826 * np.median(np.abs(hp[faint]))) if faint.sum() > 500 else st["sky_noise"]
    st["mid_noise"] = float(1.4826 * np.median(np.abs(dog[faint]))) if faint.sum() > 500 else dog_sky
    # colour in the sky away from stars: colour noise shows as tinted patches (the 90th percentile of
    # the smoothed chroma) long before the sky's median colour moves
    st["field_C"] = float(np.percentile(Cs[quiet], 90)) if quiet.sum() > 500 else st["sky_C"]
    st["star_clip"] = float((stars & (L > 0.985)).mean())
    # blue excess round bright stars: the colour of a 2-5 px ring against the colour of its surroundings
    bright = (L - opened) > 0.3
    if bright.sum() > 30:
        k3, k9 = (np.ones((5, 5), np.uint8), np.ones((11, 11), np.uint8))
        ring = cv2.dilate(bright.astype(np.uint8), k9).astype(bool) & ~cv2.dilate(bright.astype(np.uint8), k3).astype(bool)
        around = cv2.dilate(bright.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool) & ~cv2.dilate(bright.astype(np.uint8), k9).astype(bool) & ~stars
        if ring.sum() > 50 and around.sum() > 50:
            st["halo_blue"] = float(max(np.median(labs[..., 2][around]) - np.median(labs[..., 2][ring]), 0.0))
        else:
            st["halo_blue"] = 0.0
    else:
        st["halo_blue"] = 0.0
    # star colour cast: the median OKLab a / b of unsaturated star cores (the star field is neutral on average)
    cores = stars & (L > 0.3) & (L < 0.9)
    st["star_a"] = float(np.median(lab[..., 1][cores])) if cores.sum() > 50 else 0.0
    st["star_b"] = float(np.median(lab[..., 2][cores])) if cores.sum() > 50 else 0.0
    # colour clipping: coloured pixels with a channel at the ceiling
    st["gamut_clip"] = float(((rgb.max(-1) > 0.995) & (Cs > 0.04) & ~stars).mean())
    st["tonal_spread"] = (st["sig_L90"] - st["sig_L50"]) / max(st["sig_L50"] - sky_L, 0.02)
    st.update({"hue_" + k: v for k, v in sector_stats(st["hue"]).items()})
    return st


def fine_stats(rgb: np.ndarray) -> dict:
    """What a finisher checks at 100 % zoom on a full-resolution crop: the sky's grain, the fine
    noise on the faint part of the object, the fine detail on its bright part, dark overshoot
    (sharpening halos) and clipping.  No resizing: this is the pixel scale of the export."""
    rgb = np.clip(np.asarray(rgb, np.float32), 0, 1)
    lab = rgb_to_oklab(rgb)
    L = np.ascontiguousarray(lab[..., 0])
    opened = cv2.morphologyEx(L, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    stars = (L - opened) > 0.08
    stars_d = cv2.dilate(stars.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    Ls = cv2.GaussianBlur(opened, (0, 0), 4)
    sky_sel = Ls <= np.percentile(Ls, 20)
    sky_L = float(np.median(Ls[sky_sel]))
    hp = L - cv2.GaussianBlur(L, (0, 0), 1.0)
    quiet = sky_sel & ~stars_d
    grain = float(1.4826 * np.median(np.abs(hp[quiet]))) if quiet.sum() > 500 else 0.0
    sig = (Ls > sky_L + 0.06) & ~stars_d
    faint = sig & (Ls < sky_L + 0.2)
    bright = sig & (Ls >= sky_L + 0.2)
    obj_noise = float(1.4826 * np.median(np.abs(hp[faint]))) if faint.sum() > 500 else grain
    fine_detail = float(np.mean(np.abs(hp[bright]))) if bright.sum() > 500 else 0.0
    return {"grain": grain, "obj_noise": obj_noise, "fine_detail": fine_detail,
            "overshoot": float((hp < -0.04).mean()),
            "clip_hi": float((opened > 0.97).mean()), "sky_L": sky_L}


def typical() -> dict:
    """The median of every statistic over ALL reference images, whatever the target: what a
    professional finish usually measures like (star coverage, grain, colourfulness)."""
    refs = references()
    looks = [l for t in refs.get("targets", {}).values() for l in t.get("looks", [])]
    if not looks:
        return {}
    keys = [k for k in looks[0] if k != "hue"]
    return {k: float(np.median([l[k] for l in looks if k in l])) for k in keys}


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


def target_class(object_name: str | None) -> str | None:
    """The kind of target (emission, snr, planetary, galaxy, dark) from the references' tables."""
    refs = references()
    n = _norm(object_name)
    for t in refs.get("targets", {}).values():
        if n and n in {_norm(a) for a in t["names"]}:
            return t.get("class")
    return refs.get("aliases", {}).get(n)


# ------------------------------------------------------------------ the distance report

def object_weight(st: dict) -> float:
    """How much the object's targets count: fully when the object covers 8 % of the frame or more,
    down to ``object_min_weight`` for a small object in a wide field (M 27 in a Seestar frame).
    The references frame their object closely; matching its brightness in a wide field would only
    lift the sky and its noise."""
    return float(np.clip(st.get("sig_frac", 1.0) / RULES["object_frac"], RULES["object_min_weight"], 1.0))


def score_one(st: dict, ref: dict, obj_w: float = 1.0) -> tuple[float, dict]:
    """Distance of a look to one reference look (lower is better) and its terms.  Reported, not
    optimised: it says how far the finished image is from the photographs."""
    terms = {}
    for k, wgt, tol, log in TERMS:
        if k not in ref:
            continue
        d = (math.log(max(st[k], 1e-6)) - math.log(max(ref[k], 1e-6))) if log else st[k] - ref[k]
        terms[k] = wgt * (d / tol) ** 2 * (obj_w if k in OBJECT_TERMS else 1.0)
    if "hue" in ref:
        terms["hue"] = obj_w * 3.0 * math.log1p((hue_distance(st["hue"], ref["hue"]) / 0.12) ** 2)
    for k, wgt, tol, floor in CEILINGS:
        if k in ref:
            terms[k] = wgt * (max(st[k] - max(ref[k], floor), 0) / tol) ** 2 * (obj_w if k in OBJECT_TERMS else 1.0)
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


# ------------------------------------------------------------------ one-dimensional fits

def _snap(key: str, v: float) -> float:
    lo, hi, step = RANGES[key]
    v = min(max(float(v), lo), hi)
    return float(round(round(v / step) * step, 4))


class Finisher:
    """Renders settings on the proxy image (cached) and fits one slider at a time."""

    def __init__(self, lin, extra, filt, px_scale, progress=None, budget=160, stats_fn=None, label="Auto-finish"):
        self.lin, self.extra, self.filt, self.f = lin, extra, filt, px_scale
        self.progress, self.budget, self.stats_fn, self.label = progress, budget, stats_fn or look_stats, label
        self.cache: dict[str, tuple[dict, np.ndarray]] = {}
        self.renders = 0
        self.stage = ""
        self.steps: list[dict] = []

    def evaluate(self, q: dict) -> tuple[dict, np.ndarray]:
        key = json.dumps({k: q[k] for k in sorted(q) if not k.startswith("_")}, sort_keys=True, default=str)
        if key not in self.cache:
            img = nonlinear_stage(self.lin, {**q, **self.extra}, self.filt, px_scale=self.f)
            self.cache[key] = (self.stats_fn(img), img)
            self.renders += 1
            if self.progress:
                self.progress(min(self.renders, self.budget), self.budget,
                              f"{self.label}: {self.stage} ({self.renders} renders)")
        return self.cache[key]

    def stats(self, q: dict) -> dict:
        return self.evaluate(q)[0]

    # --- the fits.  Each assumes the measurement moves one way with the slider, which holds for
    #     every pair below over the range used; the bisection tolerates a little noise

    def match(self, p: dict, key: str, measure, target: float, lo: float | None = None, hi: float | None = None,
              tol: float = 0.0, rounds: int = 6, min_effect: float = 0.0) -> float:
        """The slider value at which ``measure(stats)`` meets ``target`` (bisection between lo and
        hi; the bound itself if the target lies outside).  A slider that moves the measurement by
        less than ``min_effect`` over its whole range is left where it is: there is nothing to fit."""
        r_lo, r_hi, _ = RANGES[key]
        lo, hi = r_lo if lo is None else lo, r_hi if hi is None else hi
        m_lo, m_hi = measure(self.stats({**p, key: _snap(key, lo)})), measure(self.stats({**p, key: _snap(key, hi)}))
        if abs(m_hi - m_lo) < min_effect:
            return _snap(key, float(p[key]))
        if abs(m_lo - target) <= tol:
            return _snap(key, lo)
        if abs(m_hi - target) <= tol:
            return _snap(key, hi)
        up = m_hi > m_lo
        if (target >= max(m_lo, m_hi)):
            return _snap(key, hi if up else lo)
        if (target <= min(m_lo, m_hi)):
            return _snap(key, lo if up else hi)
        for _ in range(rounds):
            mid = _snap(key, 0.5 * (lo + hi))
            if mid in (_snap(key, lo), _snap(key, hi)):
                break
            m = measure(self.stats({**p, key: mid}))
            if abs(m - target) <= tol:
                return mid
            if (m < target) == up:
                lo = mid
            else:
                hi = mid
        # the closer bound
        c_lo, c_hi = self.stats({**p, key: _snap(key, lo)}), self.stats({**p, key: _snap(key, hi)})
        return _snap(key, lo if abs(measure(c_lo) - target) <= abs(measure(c_hi) - target) else hi)

    def largest(self, p: dict, key: str, ok, lo: float | None = None, hi: float | None = None, rounds: int = 6) -> float:
        """The largest slider value for which ``ok(stats)`` still holds (ok assumed true at lo)."""
        r_lo, r_hi, _ = RANGES[key]
        lo, hi = r_lo if lo is None else lo, r_hi if hi is None else hi
        if ok(self.stats({**p, key: _snap(key, hi)})):
            return _snap(key, hi)
        if not ok(self.stats({**p, key: _snap(key, lo)})):
            return _snap(key, lo)
        for _ in range(rounds):
            mid = _snap(key, 0.5 * (lo + hi))
            if mid in (_snap(key, lo), _snap(key, hi)):
                break
            if ok(self.stats({**p, key: mid})):
                lo = mid
            else:
                hi = mid
        return _snap(key, lo)

    def smallest(self, p: dict, key: str, ok, lo: float | None = None, hi: float | None = None, rounds: int = 6) -> float:
        """The smallest slider value for which ``ok(stats)`` holds (ok assumed true at hi)."""
        r_lo, r_hi, _ = RANGES[key]
        lo, hi = r_lo if lo is None else lo, r_hi if hi is None else hi
        if ok(self.stats({**p, key: _snap(key, lo)})):
            return _snap(key, lo)
        if not ok(self.stats({**p, key: _snap(key, hi)})):
            return _snap(key, hi)
        for _ in range(rounds):
            mid = _snap(key, 0.5 * (lo + hi))
            if mid in (_snap(key, lo), _snap(key, hi)):
                break
            if ok(self.stats({**p, key: mid})):
                hi = mid
            else:
                lo = mid
        return _snap(key, hi)

    def climb(self, p: dict, key: str, gain, cost, cost_limit: float, lo: float, hi: float, step: float,
              min_gain: float = 0.02) -> float:
        """Raise a slider in steps while the measurement ``gain`` keeps growing (by ``min_gain``
        relative per step) and ``cost`` stays below ``cost_limit``: the finisher's "a bit more
        ... a bit more ... that's where it starts to look worse"."""
        v = _snap(key, lo)
        g0 = gain(self.stats({**p, key: v}))
        while v + step <= hi + 1e-9:
            nxt = _snap(key, v + step)
            st = self.stats({**p, key: nxt})
            if cost(st) > cost_limit:
                break
            g = gain(st)
            if g < g0 * (1 + min_gain):
                break
            v, g0 = nxt, g
        return v

    def note(self, step: str, key: str, before: float, after: float, measured: str, was: float, now: float,
             target, why: str):
        self.steps.append({"step": step, "setting": key, "from": before, "to": after, "measured": measured,
                           "before": round(float(was), 4), "after": round(float(now), 4),
                           "target": None if target is None else round(float(target), 4), "why": why})


CROP = 1024                # side of the full-resolution crop the 1:1 decisions are made on


def _crop_centre(img: np.ndarray, win: int) -> tuple[int, int]:
    """Centre (y, x) of the ``win`` px window of a render holding the most structured part of the
    object, with some sky around it (the window scores structure energy, discounted where the
    object fills the whole window: the stretch solver needs sky in the crop)."""
    lab = rgb_to_oklab(np.clip(img, 0, 1))
    L = np.ascontiguousarray(lab[..., 0])
    opened = cv2.morphologyEx(L, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    Ls = cv2.GaussianBlur(opened, (0, 0), 3)
    sky_L = float(np.median(Ls[Ls <= np.percentile(Ls, 20)]))
    sig = (Ls > sky_L + 0.06).astype(np.float32)
    dog = np.abs(cv2.GaussianBlur(opened, (0, 0), 1.5) - cv2.GaussianBlur(opened, (0, 0), 6)) * sig
    win = max(32, min(win, min(img.shape[:2])))
    energy = cv2.blur(dog, (win, win))
    fill = cv2.blur(sig, (win, win))
    score_ = energy * (1.0 - np.clip((fill - 0.6) / 0.4, 0, 1) * 0.7)
    m = win // 2
    inner = np.zeros_like(score_, bool)
    inner[m:-m or None, m:-m or None] = True
    score_[~inner] = -1
    cy, cx = np.unravel_index(int(np.argmax(score_)), score_.shape)
    return int(cy), int(cx)


def _targets(looks: list[dict], agg: dict | None, start: dict, obj_w: float) -> dict:
    """What the references ask for, per quantity: the medians over the reference looks (the
    aggregate), the object's brightness targets tempered for a small object in a wide field."""
    agg = agg or aggregate(looks)
    t = {k: float(agg[k]) for k in ("sky_L", "sig_L50", "sig_L90", "peak_L", "C50", "C90", "sky_noise", "sky_blotch",
                                   "obj_blotch", "star_frac", "sky_C") if k in agg}
    # a small object: between what the image shows now and what the close-up references show
    for k in ("sig_L50", "sig_L90", "peak_L"):
        t[k] = start[k] + obj_w * (t[k] - start[k])
    t["grain"] = float(np.clip(RULES["grain"] * t.get("sky_noise", 0.006), RULES["grain_floor"], RULES["grain_max"]))
    # star coverage: the target's references, unless they are denser star fields than references
    # usually are (the Veil's are Milky Way fields): then the typical coverage
    typ = typical()
    t["star_cover"] = RULES["star_cover"] * max(min(t.get("star_frac", 0.05), typ.get("star_frac", 0.05)), 0.01)
    t["hue"] = agg.get("hue")
    return t


def autofinish(session, params: dict | None = None, progress=None, budget: int | None = None,
               save: bool = True) -> dict:
    """Finish a stacked dataset: set the Process sliders one by one from measurements of the render,
    with the references of the target as the targets, then grade the colour.  Returns the new
    settings (``params``: the given ones with the tuned keys replaced) and a report of every step.
    Also saved as ``autofinish.json`` in the session folder (unless ``save`` is False)."""
    t0 = time.time()
    meta = session.meta
    obj, filt = meta.get("object") or "", meta.get("filter") or ""
    nb = is_narrowband(filt)
    looks, agg, ref_desc, sources = reference_looks(obj, nb)
    if not looks:
        raise RuntimeError("Auto-finish has no reference statistics (astrophoto/data/autofinish_refs.json is missing)")
    cls = target_class(obj)
    base = {**DEFAULTS, **(params or {})}
    # start from neutral finishing settings: what the search sets is measured, not inherited
    p = {**base, **{k: DEFAULTS[k] for k in RANGES if k in DEFAULTS}, **{k: DEFAULTS[k] for k in GRADE_KEYS},
         "grade_amount": 1.0, "brightness": 0.0}
    if progress:
        progress(0, 1, "Auto-finish: preparing the preview")
    lin, extra, f, _ = session.render_inputs(p, max_size=PROXY, progress=None)
    extra_keys = {k: v for k, v in extra.items() if k.startswith("_")}
    F = Finisher(lin, extra_keys, filt, f, progress, budget or 160)
    start = F.stats(base)
    obj_w = object_weight(start)
    T = _targets(looks, agg, start, obj_w)
    score_before = score(start, looks, obj_w)[0]

    def st_of(key, val):
        return F.stats({**p, key: val})

    def set_(step, key, val, measured, target, why, G=None):
        G = G or F
        was = G.stats(p)[measured]
        before = p[key]
        p[key] = val
        now = G.stats(p)[measured]
        F.note(step, key, before, val, measured, was, now, target, why)

    # ---- palette (dual-band): the hue family of the references
    if nb:
        F.stage = "choosing the palette"
        d = {pal: min(hue_distance(F.stats({**p, "palette": pal})["hue"], r["hue"]) for r in looks) for pal in NB_PALETTES}
        pal = min(d, key=d.get)
        if d[pal] > d["foraxx"] - 0.01:          # no clear winner (a faint object's hues are noise): the usual palette
            pal = "foraxx"
        F.steps.append({"step": "palette", "setting": "palette", "from": base.get("palette"), "to": pal,
                        "measured": "hue distance to the nearest reference", "before": round(d.get(base.get("palette"), d[pal]), 3),
                        "after": round(d[pal], 3), "target": None,
                        "why": "the palette whose hues are closest to the references'"})
        p["palette"] = pal

    # ---- tonal foundation
    F.stage = "setting the stretch"
    v = F.match(p, "stretch", lambda s: s["sig_L50"], T["sig_L50"], tol=0.01)
    set_("stretch", "stretch", v, "sig_L50", T["sig_L50"], "the object's midtones where the references put them")
    F.stage = "setting the black point"
    # as dark a sky as the references', as long as the sky's noise is not clipped to black
    crush_ok = lambda s: s["crush"] <= RULES["crush"]
    v_sky = F.match(p, "black_point", lambda s: s["sky_L"], T["sky_L"], tol=0.005)
    v_max = F.largest(p, "black_point", crush_ok)
    v = min(v_sky, v_max)
    set_("black point", "black_point", v, "sky_L", T["sky_L"],
         "the sky at the references' brightness" if v == v_sky else "any darker clips the sky's noise to black")
    F.stage = "protecting the highlights"
    clip_ok = lambda s: s["clip_hi"] <= RULES["clip"] and s["peak_L"] <= max(T["peak_L"], 0.6)
    v = F.smallest(p, "hdr", clip_ok)
    set_("HDR", "hdr", v, "peak_L", T["peak_L"], "the least compression that keeps the brightest structure off the ceiling")
    F.stage = "setting the midtones"
    v = F.match(p, "brightness", lambda s: s["sig_L50"], T["sig_L50"], lo=-0.4, hi=0.4, tol=0.01)
    set_("midtones", "brightness", v, "sig_L50", T["sig_L50"], "the midtones re-checked after the black point and HDR")

    # ---- contrast: the GHS focus that gives the most structure for the grain
    F.stage = "setting the contrast"
    best_v = float(p["contrast"])
    best = st_of("contrast", best_v)["detail_snr"]
    for cand in (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
        if cand == best_v:
            continue
        s = st_of("contrast", cand)
        if s["sky_noise"] > 1.25 * T["grain"] and cand > best_v:
            break
        if s["detail_snr"] > best * 1.05:
            best, best_v = s["detail_snr"], cand
    set_("contrast", "contrast", best_v, "detail_snr", None,
         "the focus with clearly more structure against grain" if best_v != float(base.get("contrast", DEFAULTS["contrast"]))
         else "no other focus gave clearly more structure against grain")

    # ---- noise and detail at 100 %: a full-resolution crop of the most structured part of the
    #      object, rendered at the export's pixel scale.  Fine-grain reduction and sharpening act
    #      below the proxy's resolution; a finisher judges them at 1:1
    F.stage = "cutting a full-resolution crop"
    lin_full, extra_full, _, _ = session.render_inputs(p, max_size=None, progress=None)
    cy, cx = _crop_centre(F.evaluate(p)[1], int(CROP * f))
    h, w = lin_full.shape[:2]
    cs = min(CROP, h, w)
    y0 = int(np.clip(round(cy / f) - cs // 2, 0, h - cs))
    x0 = int(np.clip(round(cx / f) - cs // 2, 0, w - cs))
    sl = (slice(y0, y0 + cs), slice(x0, x0 + cs))
    crop_extra = {k: (v[sl] if isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[:2] == (h, w) else v)
                  for k, v in extra_full.items() if k.startswith("_")}
    G = Finisher(np.ascontiguousarray(lin_full[sl]), crop_extra, filt, 1.0, progress, F.budget, fine_stats, "Auto-finish (1:1 crop)")
    del lin_full, extra_full
    F.stage = G.stage = "setting the noise reduction at 1:1"
    q0 = {**p, "sharpen": 0.0}
    v = G.smallest(q0, "luminance_denoise", lambda s_: s_["grain"] <= T["grain"])
    reached = G.stats({**q0, "luminance_denoise": v})["grain"] <= T["grain"] * 1.02
    set_("fine-grain NR", "luminance_denoise", v, "grain", T["grain"],
         "the least that brings the sky's grain at 1:1 to the references'" if reached
         else "the most it can do; the sky at 1:1 stays grainier than the references", G)
    F.stage = "setting the colour noise reduction"
    v = F.smallest(p, "chroma_denoise",
                   lambda s_: s_["sky_blotch"] <= max(T["sky_blotch"], 0.002) and s_["obj_blotch"] <= max(T["obj_blotch"], 0.002))
    set_("colour NR", "chroma_denoise", v, "obj_blotch", T["obj_blotch"], "the least that removes the colour mottle of sky and object")

    # ---- detail: as much local contrast as the faint parts bear without visible noise or mottle
    F.stage = "setting the local contrast"
    s0 = F.stats({**p, "local_contrast": 0.0})
    # local contrast lifts faint structure and its mid-scale noise together, by design: it is allowed
    # three times the rise sharpening is, and is stopped by visible mottle or tint first
    mid_lim = max(T["grain"], s0["mid_noise"] * (1 + 3 * RULES["noise_rise"]))
    blotch_lim = max(T["obj_blotch"], 0.002) * 1.5
    field_lim = RULES["field_chroma"]
    v = F.largest(p, "local_contrast", lambda s_: s_["mid_noise"] <= mid_lim and s_["obj_blotch"] <= blotch_lim
                  and s_["clip_hi"] <= RULES["clip"] and s_["field_C"] <= field_lim)
    set_("local contrast", "local_contrast", v, "mid_noise", mid_lim,
         "raised until the noise or colour mottle on the faint parts would show")
    F.stage = G.stage = "setting the sharpening at 1:1"
    s0 = G.stats({**p, "sharpen": 0.0})
    noise_lim = max(T["grain"], s0["obj_noise"] * (1 + RULES["noise_rise"]))
    over0 = s0["overshoot"]
    over_lim = max(2.5 * over0, over0 + 0.002)      # dark rings round stars and edges: sharpening halos
    v = G.largest(p, "sharpen", lambda s_: s_["obj_noise"] <= noise_lim and s_["overshoot"] <= over_lim
                  and s_["clip_hi"] <= RULES["clip"])
    set_("sharpening", "sharpen", v, "obj_noise", noise_lim,
         "raised until the fine noise on faint structure, or dark halos, began to show at 1:1", G)

    # ---- stars
    if p.get("star_separation", True):
        F.stage = "setting the stars"
        ceiling = T["star_cover"]
        v = F.smallest(p, "star_reduction", lambda s_: s_["star_frac"] <= ceiling)
        set_("star reduction", "star_reduction", v, "star_frac", ceiling,
             "stars cover no more of the frame than in a typical reference" if F.stats({**p, "star_reduction": v})["star_frac"] <= ceiling * 1.05
             else "as much reduction as allowed; the field is denser than the references'")
        v = F.largest(p, "star_intensity", lambda s_: s_["star_clip"] <= RULES["star_clip"], lo=0.4, hi=1.2)
        set_("star brightness", "star_intensity", v, "star_clip", RULES["star_clip"], "the brightest stars just below clipping")
        v = F.smallest(p, "halo_suppress", lambda s_: s_["halo_blue"] <= 0.004)
        set_("halo removal", "halo_suppress", v, "halo_blue", 0.004, "until the blue excess round bright stars is gone")

    # ---- colour
    F.stage = "setting the colour"
    s0 = F.stats(p)
    blotch_lim = max(blotch_lim, s0["obj_blotch"] * (1 + RULES["mottle_rise"]))
    # the sky's tint: neutral by the rule, or at least not made worse by the colour steps (a short
    # stack's sky keeps some colour mottle that no colour NR removes; desaturating the object would not help)
    field_lim = max(field_lim, s0["field_C"] * 1.15)
    sky_c_lim = max(T["sky_C"], 0.006, s0["sky_C"] * 1.15)
    colour_ok = lambda s: (s["obj_blotch"] <= blotch_lim and s["gamut_clip"] <= RULES["colour_clip"]
                           and s["sky_C"] <= sky_c_lim and s["field_C"] <= field_lim)
    v_t = F.match(p, "saturation", lambda s: s["C50"], T["C50"], tol=0.002, min_effect=0.004)
    if v_t == _snap("saturation", float(p["saturation"])) and abs(F.stats(p)["C50"] - T["C50"]) > 0.004:
        F.note("saturation", "saturation", p["saturation"], p["saturation"], "C50", s0["C50"], s0["C50"], T["C50"],
               "left alone: the slider barely moves this object's colour (too faint for its colour to be measured)")
    else:
        v_m = F.largest(p, "saturation", colour_ok)
        v = min(v_t, v_m)
        set_("saturation", "saturation", v, "C50", T["C50"],
             "the references' colourfulness" if v == v_t else "any more and the colour mottles, clips or tints the sky")
    if nb and p["palette"] != "natural":
        ref_cool = float(np.median([sector_stats(r["hue"])["cool"] for r in looks]))
        v_t = F.match(p, "oiii_boost", lambda s: s["hue_cool"], ref_cool, tol=0.003, min_effect=0.005)
        v_m = F.largest(p, "oiii_boost", colour_ok)
        v = min(v_t, v_m)
        set_("OIII boost", "oiii_boost", v, "hue_cool", ref_cool,
             "the warm / cool balance of the references" if v == v_t else "any more OIII and its noise tints the sky")
    ref_green = float(np.median([sector_stats(r["hue"])["green"] for r in looks]))
    if F.stats(p)["hue_green"] > ref_green + 0.03:
        v = F.smallest(p, "scnr", lambda s: s["hue_green"] <= ref_green + 0.03, lo=p["scnr"])
        set_("SCNR", "scnr", v, "hue_green", ref_green, "green hues brought down to the references' share")

    # colour noise re-checked after the colour steps: the least colour NR that keeps the field neutral
    F.stage = "re-checking the colour noise"
    s = F.stats(p)
    if s["field_C"] > field_lim or s["obj_blotch"] > blotch_lim or s["sky_blotch"] > max(T["sky_blotch"], 0.002):
        v = F.smallest(p, "chroma_denoise", lambda s_: s_["field_C"] <= field_lim and s_["obj_blotch"] <= blotch_lim
                       and s_["sky_blotch"] <= max(T["sky_blotch"], 0.002), lo=p["chroma_denoise"])
        set_("colour NR (re-check)", "chroma_denoise", v, "field_C", field_lim,
             "the colour steps had tinted the field; the colour reduction follows")

    # ---- grade: natural colour for the kind of target, towards the nearest reference's hues
    F.stage = "grading the colour"
    g = RULES["grade"]
    s = F.stats(p)
    ref = sector_stats((agg or aggregate(looks))["hue"])
    mine = sector_stats(s["hue"])
    for sector, key in (("warm", "grade_warm_hue"), ("cool", "grade_cool_hue")):
        if mine[sector] > 0.05 and ref[sector] > 0.05:
            d = (ref[sector + "_hue"] - mine[sector + "_hue"] + 180.0) % 360.0 - 180.0   # degrees, signed
            # the grade rotates 25 deg per unit; a sector with little weight gets a proportionally smaller pull
            p[key] = _snap(key, g * d / 25.0 * min(1.0, mine[sector] / 0.3))
            F.note("grade: " + sector + " hues", key, 0.0, p[key], "hue_" + sector + "_hue", mine[sector + "_hue"],
                   F.stats(p)["hue_" + sector + "_hue"], ref[sector + "_hue"],
                   f"towards the references' {sector} hue")
    if not nb:
        # the balance of warm and cool colour of a broadband target: galaxies keep their blue arms,
        # reflection nebulae their blue; a sector that the references show more of is lifted
        for sector, key in (("warm", "grade_warm_sat"), ("cool", "grade_cool_sat")):
            if mine[sector] > 0.02 and ref[sector] > 0.02:
                p[key] = _snap(key, 1 + g * (math.sqrt(ref[sector] / mine[sector]) - 1))
                F.note("grade: " + sector + " chroma", key, 1.0, p[key], "hue_" + sector, mine[sector],
                       F.stats(p)["hue_" + sector], ref[sector], f"the references' share of {sector} colour")
        # white balance on the star field: the average star is neutral
        s = F.stats(p)
        p["grade_tint"] = _snap("grade_tint", -g * s["star_a"] / 0.025)
        p["grade_temperature"] = _snap("grade_temperature", -g * s["star_b"] / 0.025)
        s2 = F.stats(p)
        F.note("grade: white balance", "grade_temperature/grade_tint", 0.0, p["grade_temperature"], "star_b", s["star_b"],
               s2["star_b"], 0.0, "the star field made neutral")
    # tonal spread of the object: the references' separation of highlights from midtones
    s = F.stats(p)
    ref_spread = float(np.median([(r["sig_L90"] - r["sig_L50"]) / max(r["sig_L50"] - r["sky_L"], 0.02) for r in looks]))
    if s["tonal_spread"] > 0.05:
        p["grade_contrast"] = _snap("grade_contrast", g * 0.5 * (ref_spread / s["tonal_spread"] - 1))
        F.note("grade: S-curve", "grade_contrast", 0.0, p["grade_contrast"], "tonal_spread", s["tonal_spread"],
               F.stats(p)["tonal_spread"], ref_spread, "the references' tonal spread on the object")
    if g <= 0:
        p.update({k: DEFAULTS[k] for k in GRADE_KEYS})

    # ---- the last look: grain re-checked after local contrast, sharpening and the grade
    F.stage = G.stage = "checking the result"
    sg = G.stats(p)
    if sg["grain"] > T["grain"] * 1.15 and p["luminance_denoise"] < RANGES["luminance_denoise"][1]:
        v = G.smallest(p, "luminance_denoise", lambda s_: s_["grain"] <= T["grain"], lo=p["luminance_denoise"])
        set_("fine-grain NR (re-check)", "luminance_denoise", v, "grain", T["grain"],
             "the detail steps had raised the grain at 1:1; the reduction follows", G)
    s = F.stats(p)
    if s["clip_hi"] > RULES["clip"]:
        # the detail steps and the grade have pushed the brightest structure onto the ceiling
        v = F.smallest(p, "hdr", lambda s_: s_["clip_hi"] <= RULES["clip"], lo=p["hdr"])
        set_("HDR (re-check)", "hdr", v, "clip_hi", RULES["clip"],
             "the later steps had pushed the brightest structure onto the ceiling; HDR follows")
    final = F.stats(p)
    score_after, terms = score(final, looks, obj_w)
    near = terms.pop("_nearest")
    out_params = {**base, **{k: p[k] for k in TUNED if k in p}}
    report = {
        "object": obj, "filter": filt, "class": cls, "reference": ref_desc,
        "nearest": sources[near] if near < len(sources) else None,
        "steps": F.steps,
        "targets": {k: round(float(v), 4) for k, v in T.items() if k != "hue" and v is not None},
        "score_before": round(score_before, 3), "score_after": round(score_after, 3),
        "terms_after": {k: round(v, 3) for k, v in terms.items()},
        "changed": {k: [base.get(k), out_params[k]] for k in TUNED if k in out_params and base.get(k) != out_params[k]},
        "stats_before": _brief(start), "stats_after": _brief(final), "stats_reference": _brief(looks[near]),
        "object_weight": round(obj_w, 2), "renders": F.renders + G.renders, "crop": [y0, x0, cs],
        "seconds": round(time.time() - t0, 1),
    }
    res = {"params": {k: out_params[k] for k in DEFAULTS if k in out_params}, "report": report, "created": time.time()}
    if save:
        try:
            with open(os.path.join(session.dir, "autofinish.json"), "w") as fh:
                json.dump(res, fh, indent=1, default=float)
        except OSError:
            pass
    return res


def _brief(st: dict) -> dict:
    keys = [k for k, *_ in TERMS + CEILINGS] + ["sig_L90", "detail_snr", "fine_noise", "star_clip", "halo_blue", "clip_hi", "crush"]
    return {k: round(float(st[k]), 4) for k in keys if k in st}
