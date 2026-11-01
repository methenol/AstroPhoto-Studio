"""Reference looks for Auto-finish: the medians of framing-independent statistics over 58 freely licensed reference astrophotographs of 16 targets
(Wikimedia Commons; only the statistics and the attributions are kept in astrophoto/data/autofinish_refs.json, built by experiments/autofinish_refs.py on
the main branch).  ``look_stats`` measures a finished sRGB image the same way; ``reference_looks`` finds a target's own references, else its class's, else
the dual-band / broadband ones; ``score`` is the distance to the nearest reference, reported before and after Auto-finish as a check (not an objective).
Ported unchanged from the application's original autofinish.py."""
from __future__ import annotations

import json
import math
import os
import re

import cv2
import numpy as np

from .finish import rgb_to_oklab

REF_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "astrophoto", "data", "autofinish_refs.json")
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


