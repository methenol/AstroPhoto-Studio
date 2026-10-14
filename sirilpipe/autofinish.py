"""Auto-finish: choose the processing settings from measurements of the image, with fixed rules.

No network, no machine learning, no reference library: every setting is a deterministic function of
numbers measured on the restored image, and the report lists each decision with the measurement behind
it (the web UI renders ``report.steps`` as one row per setting: ``step, from, to, measured, before,
after, target, why``).  Rules (DESIGN section 4 "Autofinish"):

* **stretch**: target sky median by how much of the frame holds signal (object coverage);
* **star_reduction / star_intensity**: from the star density (stars per megapixel, measured when the
  stars were removed);
* **hdr**: the smallest highlight protection (0, 0.3, 0.6, 0.9, 1.2) that keeps the object's cores below 0.3 %
  clipped pixels in a trial render;
* **saturation**: raised until the object's mean colour saturation reaches the aim (never lowered: the
  palette's own colour is kept);
* **oiii_boost** (dual-band): so the OIII signal over the sky is at least 0.7 x the Ha signal (never reduced);
* **gradient check**: the residual large-scale gradient after the restoration, in display units, reported
  (changing it needs the linear stage to run again, so it is not a setting here).

``grade_*`` stay neutral.  The result is written to ``autofinish.json``.
"""
from __future__ import annotations

import os
import time
from datetime import datetime

import cv2
import numpy as np

from . import finish

# Aims of the rules (documented constants, not tuned per image).
STRETCH_RULE = ((0.15, 0.10), (0.30, 0.13), (0.60, 0.16), (1.00, 0.20))      # (coverage, target median) breakpoints
STAR_DENSITY = ((150.0, 0.25, 1.0), (600.0, 0.30, 1.0))                       # per MP: sparse, normal; dense above
CLIP_LIMIT = 0.0005                # fraction of pixels allowed at >= 0.985 in the starless render (a white core of a few hundred pixels is allowed)
HDR_LADDER = (0.0, 0.3, 0.6, 0.9, 1.2)       # 0 = the plain statistical (midtones) stretch: the most contrast in the highlights
SATURATION_AIM = {"broadband": 0.32, "dual-band": 0.30}                        # mean HSV saturation of the object
OIII_TO_HA = 0.7                   # OIII lifted to at least 0.7 x the Ha signal (never lowered: a strong OIII is the data, e.g. the M 42 core)
GRADIENT_LIMIT = 0.03              # display units: a visible sky gradient


def target_for_coverage(c: float) -> float:
    """Piecewise-linear target sky median over the breakpoints of ``STRETCH_RULE`` (0.10 for a small object on empty
    sky up to 0.20 for a nebula that fills the frame; 0.16 = the default at 60 % coverage)."""
    pts = ((0.0, STRETCH_RULE[0][1]),) + STRETCH_RULE
    c = float(np.clip(c, 0, 1))
    for (c0, t0), (c1, t1) in zip(pts, pts[1:]):
        if c <= c1:
            return t0 + (t1 - t0) * (c - c0) / max(c1 - c0, 1e-9)
    return pts[-1][1]


def star_settings(per_mp: float) -> tuple[float, float, str]:
    """(star_reduction, star_intensity, class) from stars per megapixel."""
    if per_mp < STAR_DENSITY[0][0]:
        return STAR_DENSITY[0][1], STAR_DENSITY[0][2], "sparse"
    if per_mp <= STAR_DENSITY[1][0]:
        return STAR_DENSITY[1][1], STAR_DENSITY[1][2], "normal"
    # a dense field keeps its faint stars (the Seestar's own picture of M 27 shows 4000 per MP): reduction only trims the bright
    # stars' footprints (finish.reduce_stars), never the count; 0.65 / 0.8 removed 90 % of them
    return float(min(0.40, 0.30 + (per_mp - STAR_DENSITY[1][0]) / 3000.0 * 0.10)), 0.95, "dense"


def gradient_residual(lum_lin: np.ndarray, target: float, noise_sigma: float) -> dict:
    """Large-scale sky gradient left in the linear luminance, in display units.  The luminance is reduced to
    64 x 64 block means; the sky blocks are the darkest 60 %; the spread (P95 - P5) of that sky map is pushed through
    the stretch's own slope at the sky level: the statistical stretch puts its black point 5 sigma (the pixel noise
    ``noise_sigma``) below the sky median and sends the median to ``target``."""
    small = cv2.resize(np.ascontiguousarray(lum_lin), (64, 64), interpolation=cv2.INTER_AREA)
    sky = small[small <= np.percentile(small, 60)]
    if sky.size < 20 or noise_sigma <= 0:
        return {"spread_linear": 0.0, "spread_display": 0.0}
    spread = float(np.percentile(sky, 95) - np.percentile(sky, 5))
    m = finish.STAT_SIGMA * noise_sigma
    cur = finish.Curve(m, target, 0.0)
    lo = float(cur(np.array([max(m - spread / 2, 0.0)], np.float32))[0])
    hi = float(cur(np.array([m + spread / 2], np.float32))[0])
    return {"spread_linear": round(spread, 6), "spread_display": round(hi - lo, 4)}


def measure(img: np.ndarray) -> dict:
    """Statistics of a rendered display image (3, H, W): sky level, sky colour cast, clipping, object saturation."""
    lum = finish.luminance(img)
    med = float(np.median(lum))
    sm = cv2.GaussianBlur(lum, (0, 0), 2.0)
    sky = sm <= med + 0.02
    obj = sm > med + 0.08
    casts = [float(np.median(img[c][sky])) for c in range(3)] if sky.sum() > 100 else [med] * 3
    return {"sky_L": round(med, 4), "sky_C": round(max(casts) - min(casts), 4),
            "clip_frac": round(float((img.max(0) >= 0.985).mean()), 5),
            "C50": round(finish.mean_saturation(img, obj if obj.sum() > 200 else None), 4),
            "object_frac": round(float(obj.mean()), 4)}


def score(stats: dict, target: float) -> float:
    """0..1: sky at the target level, neutral sky, no clipped cores (mean of the three)."""
    a = 1.0 - min(1.0, abs(stats["sky_L"] - target) / max(target, 1e-6))
    b = 1.0 - min(1.0, stats["sky_C"] / 0.02)
    c = 1.0 - min(1.0, stats["clip_frac"] / 0.01)
    return round((a + b + c) / 3.0, 3)


def run(session: str, meta: dict, params: dict, progress=None, checkpoint=None) -> dict:
    """Tune ``params`` for the image in ``session``; writes and returns ``{"params", "report"}``."""
    from . import post
    from .post import _save_json

    t0 = time.time()
    steps_total = 8
    state = {"i": 0}

    def tick(msg):
        state["i"] += 1
        if checkpoint:
            checkpoint()
        if progress:
            progress(state["i"], steps_total, f"Auto-finish: {msg}")

    p0 = {**post.DEFAULTS, **(params or {})}
    tick("loading")
    layers = post._Layers(session, p0, 1000)
    dual = post._dual_band(meta, layers.header)
    full_mp = _full_megapixels(layers, session)
    tuned: dict = {}
    steps: list[dict] = []

    def decide(key, new, measured, before, target=None, why=""):
        old = p0.get(key)
        tuned[key] = new
        steps.append({"step": key, "from": old, "to": new, "measured": measured, "before": before, "after": None,
                      "target": target, "why": why})

    base = layers.base
    lum_lin = finish.luminance(base)

    # 1. stretch target from the object coverage
    tick("object coverage")
    cov = finish.object_coverage(lum_lin)
    target = round(target_for_coverage(cov), 3)
    decide("stretch", target, "object_frac", round(cov, 3), None,
           f"the object covers {cov * 100:.0f} % of the frame: sky median {target} (0.10 small object ... 0.20 fills the frame)")
    decide("auto_stretch", False, "object_frac", round(cov, 3), None, "the coverage rule above replaces the automatic adaptation")

    # 2. stars
    tick("stars")
    star_info = _star_density(layers, session, full_mp)
    cls_stars = "none"
    if star_info is not None:
        red, inten, cls_stars = star_settings(star_info["per_megapixel"])
        decide("star_reduction", round(red, 2), "star_frac", star_info["per_megapixel"], None,
               f"{star_info['per_megapixel']:.0f} stars per megapixel ({cls_stars} field)")
        decide("star_intensity", round(inten, 2), "star_frac", star_info["per_megapixel"], None, f"{cls_stars} star field")

    # trial renders (numpy, 1000 px): stats before = the given params, after = the tuned ones
    tick("trial render")
    p_before = {**p0}
    img0, _ = post._render(layers, p_before, dual, fast=True)
    stats_before = measure(img0)
    p1 = {**p0, **tuned}

    # 3. highlight protection: the lowest hdr whose starless render keeps the cores below the clip limit
    tick("highlights")
    chosen = HDR_LADDER[0]
    clip_seen = None
    for h in HDR_LADDER:
        pt = {**p1, "hdr": h, "star_reduction": 1.0}
        st = measure(post._render(layers, pt, dual, fast=True)[0])
        clip_seen, chosen = st["clip_frac"], h
        if st["clip_frac"] <= CLIP_LIMIT:
            break
    decide("hdr", chosen, "peak_L", clip_seen, CLIP_LIMIT,
           f"{(clip_seen or 0) * 100:.2f} % of the starless render is clipped at hdr {chosen} (limit {CLIP_LIMIT * 100:.2f} %)")
    p1["hdr"] = chosen

    # 4. dual-band: OIII boost
    tick("colour")
    if dual:
        boost, ratio = _oiii_boost(base, p1, target)
        if boost is not None:
            decide("oiii_boost", boost, "C50", ratio, OIII_TO_HA,
                   f"OIII signal is {ratio:.2f} x the Ha signal above the sky: boost to reach {OIII_TO_HA}")
            p1["oiii_boost"] = boost

    # 5. saturation from the measured object colour (never below 1.0: the palette's own colour is kept)
    p_sat = {**p1, "saturation": 1.0}
    s0 = measure(post._render(layers, p_sat, dual, fast=True)[0])["C50"]
    aim = SATURATION_AIM["dual-band" if dual else "broadband"]
    sat = float(np.clip(round(aim / max(s0, 1e-3) / 0.05) * 0.05, 1.0, 2.2))
    decide("saturation", round(sat, 2), "C50", s0, aim, f"object colourfulness {s0:.2f} at saturation 1.0, aim {aim}")
    p1["saturation"] = round(sat, 2)

    # 6. gradient residual (report only)
    tick("gradient check")
    grad = gradient_residual(lum_lin, target, finish.lower_half_sigma(finish.sample(lum_lin)))
    grad_note = ("a visible sky gradient remains: run Restore again with a higher gradient smoothing"
                 if grad["spread_display"] > GRADIENT_LIMIT else "no visible gradient")

    tick("final measurements")
    img1, _ = post._render(layers, p1, dual, fast=True)
    stats_after = measure(img1)
    for s in steps:
        s["after"] = {"stretch": stats_after["sky_L"], "hdr": stats_after["clip_frac"], "saturation": stats_after["C50"]}.get(s["step"])
    tuned_full = {**p0, **tuned, "saturation": p1["saturation"], "hdr": p1["hdr"]}
    cls = ("dual-band " if dual else "broadband ") + ("nebula" if cov > 0.0 and cov < 0.8 else "field")
    report = {
        "reference": "measured statistics (fixed rules, no reference image)", "class": cls,
        "score_before": score(stats_before, p0["stretch"]), "score_after": score(stats_after, target),
        "nearest": None, "renders": 5, "seconds": round(time.time() - t0, 1), "steps": steps,
        "stats_before": stats_before, "stats_after": stats_after,
        "gradient": {**grad, "limit": GRADIENT_LIMIT, "note": grad_note},
        "coverage": round(cov, 4), "star_field": cls_stars, "created": datetime.now().isoformat(timespec="seconds"),
        "summary": f"{cls}: stretch {target}, saturation {p1['saturation']}, hdr {p1['hdr']}",
    }
    result = {"params": tuned_full, "report": report}
    _save_json(os.path.join(session, "autofinish.json"), result)
    return result


def _full_megapixels(layers, session: str) -> float:
    from . import fitsio
    try:
        from astropy.io import fits
        h = fits.getheader(layers.src)
        return float(h["NAXIS1"]) * float(h["NAXIS2"]) / 1e6
    except Exception:
        return layers.base.shape[1] * layers.base.shape[2] / 1e6


def _star_density(layers, session: str, full_mp: float) -> dict | None:
    """Stars per megapixel of the full frame: from the removal's own measurement, else from the star layer."""
    from .post import _load_json
    if layers.stars is None:
        return None
    sm = _load_json(os.path.join(session, "starnet_meta.json")).get("stars")
    if sm and sm.get("per_megapixel") is not None:
        return {"count": sm["count"], "per_megapixel": float(sm["per_megapixel"])}
    noise = finish.lower_half_sigma(finish.sample(finish.luminance(layers.base)))
    n, _ = finish.count_stars(layers.stars, noise)
    return {"count": n, "per_megapixel": n / max(full_mp, 1e-6)}


def _oiii_boost(base: np.ndarray, p: dict, target: float) -> tuple[float | None, float]:
    """OIII boost so the stretched OIII signal over the sky equals ``OIII_TO_HA`` x the Ha signal (both
    channels are stretched to the same sky median, so the signal is the excess above it, in the object)."""
    pos = finish.curve_position(p["stretch_method"], float(p["hdr"]))
    ha, oiii, _ = finish.extract_ha_oiii(base, bool(p["oiii_unmix"]))
    h_st, _ = finish.stretch_channel(ha, target, position=pos)
    o_st, _ = finish.stretch_channel(oiii, target, position=pos)
    obj = cv2.GaussianBlur(h_st, (0, 0), 2.0) > target + 0.08
    if obj.sum() < 200:
        return None, 0.0
    s_h = float(np.median(h_st[obj] - target))
    s_o = float(np.median(o_st[obj] - target))
    if s_h <= 1e-3:
        return None, 0.0
    ratio = max(s_o, 1e-3) / s_h
    return float(np.clip(round(OIII_TO_HA / ratio / 0.05) * 0.05, 1.0, 3.0)), round(ratio, 3)
