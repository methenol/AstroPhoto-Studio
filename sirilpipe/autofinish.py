"""Auto-finish: choose the processing settings from measurements of the image, with fixed rules.

No network, no machine learning, no reference library: every setting is a deterministic function of
numbers measured on the restored image, and the report lists each decision with the measurement behind
it (the web UI renders ``report.steps`` as one row per setting: ``step, from, to, measured, before,
after, target, why``).  Rules (DESIGN section 4 "Autofinish"):

* **stretch** (= the sky's display level) and **contrast** (= the GHS local intensity b): the stretch is the Generalised Hyperbolic Stretch of
  ``finish.ghs_stretch`` with the sky sent to ``finish.SKY_AIM`` and the 90th percentile of the signal to ``finish.NEBULA_AIM``; b is the highest
  of ``finish.GHS_B_LADDER`` for which a D / SP meeting both aims exists (measured in a trial render);
* **star_reduction / star_intensity**: from the star density (stars per megapixel, measured when the
  stars were removed);
* **star_stretch**: the Seti Astro Star Stretch amount that sends the median detected star of the full-resolution star layer to
  display level 0.3 (finish.star_stretch_amount);
* **oiii_boost** (dual-band): the Blanshan / Cranfield narrowband-normalisation boost (applied to the colour-resolution OIII, never to the pixel noise) that
  puts the OIII signal over the sky at ``OIII_TO_HA`` x the Ha signal in the object;
* **saturation**: raised until the object's mean colour saturation reaches the aim (never lowered: the
  palette's own colour is kept);
* **black_point**: 0 (the stretch anchors the sky itself), **hdr**: the lowest of ``HDR_LADDER`` (Statistical_Stretch's HDR highlight compress) for which at most
  ``CLIP_LIMIT`` of the starless render's pixels are white (lightness >= ``WHITE_LEVEL``);
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
SKY_AIM = finish.SKY_AIM           # display level of the sky (the user's 0.07-0.10: dark grey, not black)
STAR_DENSITY = ((150.0, 0.25, 1.0), (600.0, 0.30, 1.0))                       # per MP: sparse, normal; dense above
CLIP_LIMIT = 0.0005                # reported: fraction of pixels allowed at >= 0.985 in the starless render
SATURATION_AIM = {"broadband": 0.32, "dual-band": 0.40}                        # mean HSV saturation of the object
OIII_TO_HA = 0.3                   # OIII signal over the sky at 0.3 x the Ha signal in the object (0.5 / 1.2 on IC 1805 washed the whole sky teal: the faint OIII plateau and its
                                   # residual noise at colour resolution are boosted with it; 0.3 keeps the OIII arcs teal-grey where they are and the sky neutral)
OIII_BOOST_RANGE = (0.5, 1.2)
LOCAL_CONTRAST = 1.0               # SNR-gated wavelet gain 1 + amount on the 2-16 px scales (finish.starlet_denoise ``contrast``): only coefficients standing 5 sigma above the
                                   # measured noise of their scale are amplified, so on noisy data it does nothing and on clean data it brings out the filaments
SYNTH_LUM = 0.9                    # share of the lightness taken from the Ha plane (finish.ha_luminance): the OIII noise never reaches the lightness
HDR_LADDER = (0.0, 0.3, 0.6, 0.9, 1.2, 1.5)
CROWD_LEVEL, CROWD_LIMIT = 0.9, 0.02    # of the pixels above 0.5, at most 2 % may sit above 0.9 (a core that is not blown to white can still be crowded into the last 10 % of the range: M 42's Trapezium)
WHITE_LEVEL = 0.97                 # lightness at which a pixel counts as white (a blown core)
GRADIENT_LIMIT = 0.03              # display units: a visible sky gradient


def sky_level(img: np.ndarray) -> float:
    """Display level of the sky: the median of the darkest 45 % of the smoothed luminance (the nebula and the stars sit above it;
    the whole-frame median of a nebula that fills a third of the frame is lifted by the nebula)."""
    lum = cv2.GaussianBlur(np.ascontiguousarray(finish.luminance(img)), (0, 0), 2.0)
    s = finish.sample(lum, valid_only=False)
    return float(np.median(s[s <= np.percentile(s, 45)]))


def star_settings(per_mp: float) -> tuple[float, float, str]:
    """(star_reduction, star_intensity, class) from stars per megapixel."""
    if per_mp < STAR_DENSITY[0][0]:
        return STAR_DENSITY[0][1], STAR_DENSITY[0][2], "sparse"
    if per_mp <= STAR_DENSITY[1][0]:
        return STAR_DENSITY[1][1], STAR_DENSITY[1][2], "normal"
    # a dense field keeps its faint stars (the Seestar's own picture of M 27 shows 4000 per MP): reduction only trims the bright
    # stars' footprints (finish.reduce_stars), never the count; 0.65 / 0.8 removed 90 % of them
    return float(min(0.40, 0.30 + (per_mp - STAR_DENSITY[1][0]) / 3000.0 * 0.10)), 0.95, "dense"


def gradient_residual(lum_lin: np.ndarray, slope: float) -> dict:
    """Large-scale sky gradient left in the linear luminance, in display units.  The luminance is reduced to
    64 x 64 block means; the sky blocks are the darkest 60 %; the spread (P95 - P5) of that sky map is multiplied by the
    stretch's slope at the sky level (``slope``: display units per linear unit, from the GHS fit)."""
    small = cv2.resize(np.ascontiguousarray(lum_lin), (64, 64), interpolation=cv2.INTER_AREA)
    sky = small[small <= np.percentile(small, 60)]
    if sky.size < 20 or slope <= 0:
        return {"spread_linear": 0.0, "spread_display": 0.0}
    spread = float(np.percentile(sky, 95) - np.percentile(sky, 5))
    return {"spread_linear": round(spread, 6), "spread_display": round(spread * slope, 4)}


def stretch_slope(ghs_info: dict) -> float:
    """Display units per linear unit at the sky of the GHS recorded by ``finish.ghs_stretch``."""
    g = ghs_info.get("ghs") or {}
    white, sky = float(ghs_info.get("white") or 0), float(ghs_info.get("sky_in") or 0)
    if white <= 0 or not g:
        return 0.0
    h = 0.01
    t = finish.ghs(np.array([sky, sky + h], np.float64), g["b"], g["D"], g["sp"])
    return float((t[1] - t[0]) / h / white)


def measure(img: np.ndarray) -> dict:
    """Statistics of a rendered display image (3, H, W): sky level, sky colour cast, clipping, object saturation."""
    lum = finish.luminance(img)
    med = float(np.median(lum))
    sm = cv2.GaussianBlur(lum, (0, 0), 2.0)
    sky = sm <= med + 0.02
    obj = sm > med + 0.08
    casts = [float(np.median(img[c][sky])) for c in range(3)] if sky.sum() > 100 else [med] * 3
    return {"sky_L": round(med, 4), "sky_C": round(max(casts) - min(casts), 4),
            "clip_frac": round(float((img.max(0) >= 0.985).mean()), 5), "white_frac": round(float((lum >= WHITE_LEVEL).mean()), 5),
            "crowd": round(float((lum >= CROWD_LEVEL).sum() / max(int((lum >= 0.5).sum()), 1)), 4),
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

    # 1. the sky's display level (the stretch setting)
    tick("object coverage")
    cov = finish.object_coverage(lum_lin)
    decide("stretch", SKY_AIM, "object_frac", round(cov, 3), SKY_AIM,
           f"the object covers {cov * 100:.0f} % of the frame; the GHS sends the sky to {SKY_AIM} (dark grey) whatever the coverage: "
           f"the black point is the sky itself, not a multiple of the pixel noise")
    decide("auto_stretch", False, "object_frac", round(cov, 3), None, "the sky aim above replaces the automatic adaptation")
    decide("black_point", 0.0, "sky_L", SKY_AIM, SKY_AIM, "the stretch's anchor already sits at the sky: no extra black point")

    # 2. stars
    tick("stars")
    star_info = _star_density(layers, session, full_mp)
    cls_stars = "none"
    if star_info is not None:
        red, inten, cls_stars = star_settings(star_info["per_megapixel"])
        decide("star_reduction", round(red, 2), "star_frac", star_info["per_megapixel"], None,
               f"{star_info['per_megapixel']:.0f} stars per megapixel ({cls_stars} field)")
        decide("star_intensity", round(inten, 2), "star_frac", star_info["per_megapixel"], None, f"{cls_stars} star field")
        amount, sa = _star_stretch(session, p0)
        if sa.get("median_peak") is not None:
            decide("star_stretch", amount, "median_star_peak", sa["median_peak"], finish.STAR_MEDIAN_AIM,
                   f"the median of {sa['stars']} detected stars peaks at {sa['median_peak']:.4g} (linear): amount {amount} puts it at "
                   f"{sa['median_after']}, the brightest unsaturated stars ({sa['bright_peak']:.3g}) at {sa['bright_after']}")

    # trial renders (numpy, 1000 px): stats before = the given params, after = the tuned ones
    tick("trial render")
    p_before = {**p0}
    img0, _ = post._render(layers, p_before, dual, fast=True)
    stats_before = measure(img0)
    p1 = {**p0, **tuned, "stretch_method": "ghs", "contrast": 13.0, "oiii_boost": 1.0, "saturation": 1.0, "hdr": 0.0,
          "synthetic_luminance": SYNTH_LUM, "local_contrast": LOCAL_CONTRAST}

    # 3. GHS local intensity: the highest b whose D / SP meet both aims
    tick("stretch")
    flayers = post._Layers(session, p1, None)               # full resolution: the wavelet denoise and the statistics of the fit depend on the pixel scale
    im_t, info_t = post._render(flayers, {**p1, "star_reduction": 1.0}, dual, fast=True)
    g = info_t.get("ghs") or {}
    gg = g.get("ghs") or {}
    b = float(gg.get("b", p0["contrast"]))
    decide("contrast", b, "ghs_error", gg.get("error"), finish.GHS_FIT_TOLERANCE,
           f"GHS b {b:g} (D {gg.get('D')}, SP {gg.get('sp')}): sky {g.get('sky_in')} -> {SKY_AIM}, the signal's 90th percentile {g.get('nebula_in')} -> {finish.NEBULA_AIM} "
           f"(fit error {gg.get('error')}); the highest b of {finish.GHS_B_LADDER} that meets both")
    p1["contrast"] = b
    tried = []
    for h in HDR_LADDER:
        m = measure(post._render(layers, {**p1, "hdr": h, "star_reduction": 1.0}, dual, fast=True)[0])
        tried.append((h, m["white_frac"], m["crowd"]))
    within = [t for t in tried if t[1] <= CLIP_LIMIT and t[2] <= CROWD_LIMIT]
    cost = [t[1] / CLIP_LIMIT + t[2] / CROWD_LIMIT for t in tried]
    # the lowest setting that keeps the whites and the crowding of the top within their limits; when none does (a core the compress cannot save), the lowest that gets
    # within 10 % of the best achievable (a stronger compress that gains nothing only darkens the highlights)
    best = min(cost)
    hdr, white, crowd = within[0] if within else next(t for t, c in zip(tried, cost) if c <= best * 1.1 + 1e-9)
    p1["hdr"] = hdr
    decide("hdr", hdr, "white_frac", white, CLIP_LIMIT,
           f"{white * 100:.3f} % of the starless render is white (lightness >= {WHITE_LEVEL}) and {crowd * 100:.1f} % of its bright pixels sit above {CROWD_LEVEL} at HDR {hdr}: "
           f"the lowest of {HDR_LADDER} that keeps them within {CLIP_LIMIT * 100:.2f} % / {CROWD_LIMIT * 100:.0f} % "
           f"(Statistical_Stretch's HDR highlight compress, knee {finish.HDR_KNEE_GHS})")

    decide("local_contrast", LOCAL_CONTRAST, "ghs_error", p0["local_contrast"], None,
           "wavelet gain on the 2-16 px scales, gated by each coefficient's measured signal-to-noise (>= 5 sigma of its scale): the GHS stretches tone, this separates the filaments from the glow where the data support it")
    p1["local_contrast"] = LOCAL_CONTRAST

    # 4. dual-band: OIII boost (Blanshan / Cranfield normalisation strength)
    tick("colour")
    if dual:
        boost, ratio = _oiii_boost(flayers.base, {**p1, "_denoised_linear": flayers.denoised})
        if boost is not None:
            decide("oiii_boost", boost, "C50", ratio, OIII_TO_HA,
                   f"after the narrowband normalisation (boost 1) the OIII signal is {ratio:.2f} x the Ha signal over the sky in the object: boost {boost} gives {OIII_TO_HA}")
            p1["oiii_boost"] = boost
        decide("synthetic_luminance", SYNTH_LUM, "C50", p0["synthetic_luminance"], None,
               f"lightness {SYNTH_LUM:.0%} from the Ha plane (the high-SNR channel), the colour from the Ha : OIII ratio at colour resolution")
        p1["synthetic_luminance"] = SYNTH_LUM

    # 4b. final sharpening from the measured signal-to-noise (the unsharp mask sharpens whatever is in the lightness: nothing below SNR 10, the setting's 0.25 at 50)
    snr = _snr_before_denoise(session, lum_lin)
    sharp = round(finish.snr_ladder(snr, 0.0, 0.25), 2)
    decide("sharpen", sharp, "object_snr", round(snr, 2), None,
           f"per-pixel signal-to-noise of the object {snr:.1f}: sharpening {sharp} (0 up to {finish.SNR_LOW:g}, 0.25 from {finish.SNR_HIGH:g})")
    p1["sharpen"] = sharp

    # 5. saturation from the measured object colour (never below 1.0: the palette's own colour is kept)
    p_sat = {**p1, "saturation": 1.0}
    s0 = measure(post._render(layers, p_sat, dual, fast=True)[0])["C50"]
    aim = SATURATION_AIM["dual-band" if dual else "broadband"]
    sat = float(np.clip(round(aim / max(s0, 1e-3) / 0.05) * 0.05, 1.0, 2.2))
    decide("saturation", round(sat, 2), "C50", s0, aim, f"object colourfulness {s0:.2f} at saturation 1.0, aim {aim}")
    p1["saturation"] = round(sat, 2)

    # 6. gradient residual (report only)
    tick("gradient check")
    grad = gradient_residual(lum_lin, stretch_slope(g))
    grad_note = ("a visible sky gradient remains: run Restore again with a higher gradient smoothing"
                 if grad["spread_display"] > GRADIENT_LIMIT else "no visible gradient")

    tick("final measurements")
    img1, _ = post._render(layers, p1, dual, fast=True)
    stats_after = measure(img1)
    for s in steps:
        s["after"] = {"stretch": stats_after["sky_L"], "hdr": stats_after["clip_frac"], "saturation": stats_after["C50"]}.get(s["step"])
    tuned_full = {**p0, **tuned, "saturation": p1["saturation"], "hdr": p1["hdr"], "contrast": p1["contrast"], "oiii_boost": p1["oiii_boost"],
                  "black_point": 0.0, "synthetic_luminance": p1["synthetic_luminance"], "local_contrast": p1["local_contrast"], "sharpen": p1["sharpen"]}
    cls = ("dual-band " if dual else "broadband ") + ("nebula" if cov > 0.0 and cov < 0.8 else "field")
    report = {
        "reference": "measured statistics (fixed rules, no reference image)", "class": cls,
        "score_before": score(stats_before, SKY_AIM), "score_after": score(stats_after, SKY_AIM),
        "nearest": None, "renders": 5, "seconds": round(time.time() - t0, 1), "steps": steps,
        "stats_before": stats_before, "stats_after": stats_after,
        "gradient": {**grad, "limit": GRADIENT_LIMIT, "note": grad_note},
        "coverage": round(cov, 4), "star_field": cls_stars, "created": datetime.now().isoformat(timespec="seconds"),
        "summary": f"{cls}: sky {SKY_AIM}, GHS b {b:g}, saturation {p1['saturation']}" + (f", OIII boost {p1['oiii_boost']}" if dual else ""),
    }
    result = {"params": tuned_full, "report": report}
    _save_json(os.path.join(session, "autofinish.json"), result)
    return result


def _snr_before_denoise(session: str, lum_lin: np.ndarray) -> float:
    """The object's per-pixel signal-to-noise as the restore stage measured it on the stack (before any denoiser: the denoised starless layer has a pixel noise that
    says nothing about the data); measured here on the given luminance when the restore did not record it."""
    from .post import _load_json
    v = _load_json(os.path.join(session, "restore_meta.json")).get("snr")
    if v:
        return float(v)
    e = finish.EDGE_CROP
    return finish.object_snr(lum_lin[e:-e, e:-e] if min(lum_lin.shape) > 8 * e else lum_lin)


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


def _star_stretch(session: str, p: dict) -> tuple[float, dict]:
    """Star Stretch amount from the FULL-resolution star layer (a reduced layer averages the peaks down) and the sky noise of the
    starless layer."""
    from . import post
    full = post._Layers(session, p, None)
    if full.stars is None:
        return finish.STAR_STRETCH, {"stars": 0, "median_peak": None}
    noise = finish.lower_half_sigma(finish.sample(finish.luminance(full.base)))
    return finish.star_stretch_amount(full.stars, noise)


def _oiii_boost(base: np.ndarray, p: dict) -> tuple[float | None, float]:
    """The ``oiii_boost`` (Blanshan / Cranfield normalisation strength, ``finish.nbn_oiii``) at which the OIII's signal over the sky, at colour resolution, is
    ``OIII_TO_HA`` x the Ha signal, both medians over the object (pixels of the smoothed Ha 0.08 above its sky).  Bisection on the log of the boost; returns
    ``(boost, ratio at boost 1)``."""
    h_st, o_st, _, _ = finish.dual_band_planes(base, p)
    h_c, o_c, _ = finish.colour_planes(h_st, o_st)
    sky_h = float(np.median(h_c[h_c <= np.percentile(h_c, 40)]))
    obj = h_c > sky_h + 0.08
    if obj.sum() < 200:
        return None, 0.0

    def ratio(boost: float) -> float:
        o_n, _ = finish.nbn_oiii(h_c, o_c, boost)
        sky_o = float(np.median(o_n[o_n <= np.percentile(o_n, 40)]))
        return max(float(np.median(o_n[obj]) - sky_o), 1e-4) / max(float(np.median(h_c[obj]) - sky_h), 1e-4)

    r1 = ratio(1.0)
    lo, hi = OIII_BOOST_RANGE
    if ratio(hi) < OIII_TO_HA:
        return float(hi), round(r1, 3)
    if ratio(lo) > OIII_TO_HA:
        return float(lo), round(r1, 3)
    for _ in range(12):
        mid = float(np.sqrt(lo * hi))
        if ratio(mid) < OIII_TO_HA:
            lo = mid
        else:
            hi = mid
    return float(round(np.sqrt(lo * hi) / 0.05) * 0.05), round(r1, 3)
