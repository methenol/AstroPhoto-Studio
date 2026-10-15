"""Auto-finish: choose the settings of the professional chain (``pro.py``) from measurements of the image, among PUBLISHED values only (research/FINISH-SPEC.md,
research/pro-workflow.md).  No fitting, no reference library: every setting is one of the published values, picked by a stated rule on a measured number, and the report lists
each decision with the measurement, the rule and its source (the web UI renders ``report.steps`` as one row per setting: ``step, from, to, measured, before, after, target, why``).

=====================  ======================================================================================================================================
setting                rule (source)
=====================  ======================================================================================================================================
stretch_method         ``ghs2`` (the GHS two-pass recipe, Siril GHS tutorial + ghsastro) unless the settings name another published tool
stretch                the background after the stretch: 0.10 (Brecher's five finished images 0.09 / 0.10 / 0.10 / 0.11 / 0.11); for Statistical Stretch the median target
                       follows the size of the target: 0.10 when it covers less than half of the frame, 0.25 (the script's default) when it fills it (pro-workflow.md A8)
hdr                    on (1 = the published values: GHS HP < 1, Statistical Stretch -hdr 0.15 / 0.75, Wavescale HDR 5 / 1.5 / 5.0) when, after the stretch, more than 0.05 % of
                       the starless pixels have any channel >= 0.995 or Lab lightness >= 0.97 (FINISH-SPEC gates; Siril / fas37: < 0.1 % clipped)
luminance_denoise      NoiseXTerminator ``--dn`` on the stretched starless: 0.85 on the noisiest data (per-pixel object SNR <= 3) down to 0.6 on clean data (SNR >= 15)
                       (RC-Astro / Brecher 0.6-0.85-0.9: the noisier the data the stronger)
chroma_denoise         NXT colour denoise 0.95 (NXT manual: colour 90-100 %)
palette / oiii_boost   dual-band: HOO Classic (Ha = R, OIII = (G + B) / 2), NBN only when the user sets an OIII boost; broadband: natural colour
scnr                   broadband: 0.6 (SW Skies 0.60-0.75, low end) when the subject has a green cast (median G - (R + B) / 2 > 0.01 after the stretch), else 0; dual-band: 0
                       (green is the OIII; NBN never applies SCNR to HOO)
local_contrast         the blend of the Siril ``clahe`` passes: 0.2 (0.5, the lowest published strength: Sh2-140 / IC 417) for noisy data (SNR <= 3), 0.35 (0.875, M51's second
                       pass) for clean data (SNR >= 15), between by log SNR (Brecher: after the noise reduction, under a background mask)
saturation             Siril ``satu`` amount from the published 0.25-0.5: the subject's mean saturation after the palette >= 0.35 -> 0.25, >= 0.20 -> 0.3, else 0.5
star_reduction         by star density: sparse (< 150 per MP) 0.4 (Brecher StarReduction 0.4), normal 0.5, dense (> 600) 0.6 (DSA value 0.2 = strength 0.6)
star_stretch           Seti Astro Star Stretch amount 5 (default; documented range 0-8)
star_saturation        Seti colour boost 1.0 (default) unless the stretched stars' flux colour is below 0.10 saturation: then 1.5, and 2.0 (the documented maximum) if still short
=====================  ======================================================================================================================================

``grade_*`` stay neutral.  The result is written to ``autofinish.json``.
"""
from __future__ import annotations

import os
import time
from datetime import datetime

import cv2
import numpy as np

from . import finish, gates, pro

CLIP_LIMIT = 0.0005                    # 0.05 % (FINISH-SPEC)
STAR_DENSITY = ((150.0, 0.4), (600.0, 0.5))      # per MP: sparse (< 150), normal (<= 600); dense above -> 0.6
STAR_REDUCTION_DENSE = 0.6
NXT_DN = (0.85, 0.60)                  # noisy -> clean
LC_BLEND = (0.5, 0.875)                # local_contrast: the lowest published strength (x 0.4 = 0.2) -> M51's 0.35
SATU_AMOUNTS = ((0.35, 0.25), (0.20, 0.30))      # (saturation of the subject at least ..., satu amount); below the last: 0.5
SCNR_CAST = 0.01
SCNR_AMOUNT = 0.6
STAR_COLOUR_MIN = 0.10
STAR_BOOSTS = (1.0, 1.5, 2.0)
STATS_SIDE = 1200


GHS_D2_SMALL, GHS_D2_FILLING = 2.5, 1.5      # second GHS pass D by subject coverage (measured 2026-10-15; see autofinish.run)
SATU_BROADBAND_MAX = 0.3                     # satu amount cap for broadband (Siril tutorial 0.25-0.5; the middle value keeps SPCC's calibration)
SATU_CANDIDATES = (0.25, 0.3, 0.5)           # Siril tutorial satu values
SATU_DUAL_MIN = 0.45                         # HOO renders judged right at satu 0.45 (the Siril tutorial range ends at 0.5)
HDR_DUAL_DEFAULT = 0.5                       # GHS HP protection at half strength on dual-band nebulae (see run)
STAR_BOOST_BROADBAND_MAX = 1.2               # colour boost 1.5 made M 31's faint stars green / magenta (chroma noise); 1.2 at most on broadband


def star_settings(per_mp: float) -> tuple[float, str]:
    """(star_reduction, density class)."""
    if per_mp < STAR_DENSITY[0][0]:
        return STAR_DENSITY[0][1], "sparse"
    if per_mp <= STAR_DENSITY[1][0]:
        return STAR_DENSITY[1][1], "normal"
    return STAR_REDUCTION_DENSE, "dense"


def hdr_rule(clipped: float, white: float) -> bool:
    return clipped > CLIP_LIMIT or white > CLIP_LIMIT


def nxt_strength(snr: float) -> float:
    return round(finish.snr_ladder(snr, *NXT_DN), 2)


def local_contrast_amount(snr: float) -> float:
    return round(finish.snr_ladder(snr, *LC_BLEND), 2)


def satu_amount(subject_saturation: float) -> float:
    for floor, amount in SATU_AMOUNTS:
        if subject_saturation >= floor:
            return amount
    return 0.5


def green_cast(img: np.ndarray, mask: np.ndarray | None) -> float:
    """Median of G - (R + B) / 2 over the subject pixels (display units)."""
    g = img[1] - 0.5 * (img[0] + img[2])
    return float(np.median(g[mask])) if mask is not None and mask.sum() > 100 else float(np.median(g))


def subject_saturation(img: np.ndarray, mask: np.ndarray | None) -> float:
    return finish.mean_saturation(img, mask)


def star_boost(sat_after_stretch: float) -> float:
    """Smallest of the Seti colour boosts 1.0 / 1.5 / 2.0 whose (linear) scaling of the flux colour reaches ``STAR_COLOUR_MIN``."""
    for b in STAR_BOOSTS:
        if sat_after_stretch * b >= STAR_COLOUR_MIN:
            return b
    return STAR_BOOSTS[-1]


def run(session: str, meta: dict, params: dict, progress=None, checkpoint=None) -> dict:
    """Tune ``params`` for the image in ``session``; writes and returns ``{"params", "report"}``."""
    from . import post
    from .post import _save_json

    t0 = time.time()
    steps_total = 6
    state = {"i": 0}

    def tick(msg):
        state["i"] += 1
        if checkpoint:
            checkpoint()
        if progress:
            progress(state["i"], steps_total, f"Auto-finish: {msg}")

    p0 = {**post.DEFAULTS, **(params or {})}
    tick("loading")
    layers = post._Layers(session, p0, None)
    dual = post._dual_band(meta, layers.header)
    work = os.path.join(session, "process", "autofinish")
    os.makedirs(work, exist_ok=True)
    tuned: dict = {}
    steps: list[dict] = []

    def decide(key, new, measured, before, target=None, why="", after=None):
        tuned[key] = new
        steps.append({"step": key, "from": p0.get(key), "to": new, "measured": measured, "before": before, "after": after, "target": target, "why": why})

    base = layers.base
    lum_lin = finish.luminance(base)
    snr = _snr(session, lum_lin)
    cov = finish.object_coverage(lum_lin)
    method = "ghs2" if str(p0.get("stretch_method")) in ("ghs", "") else str(p0.get("stretch_method"))
    decide("stretch_method", method, "object_snr", round(snr, 2), None, "the GHS two-pass recipe by Siril ght (Siril GHS tutorial, ghsastro): the default; the settings may name another published tool")
    small = cov < pro.STAT_FILLING_COVERAGE
    decide("stretch", pro.SKY_TARGET, "object_frac", round(cov, 3), pro.SKY_TARGET,
           f"the background after the stretch is {pro.SKY_TARGET} (Brecher: 0.10 / 0.11 / 0.09 / 0.11 / 0.10 on five finished images)"
           + (f"; the target covers {cov * 100:.0f} % of the frame: a small target, so Statistical Stretch would send its median to {pro.STAT_SMALL}" if small else
              f"; the target fills {cov * 100:.0f} % of the frame: Statistical Stretch would use its default median {pro.STAT_FILLING}"))
    decide("black_point", 0.0, "sky_L", pro.SKY_TARGET, None, "the black point is part of the stretch (linstretch -BP at the histogram toe) and of the last step")
    # the second GHS pass's D (contrast of the subject): measured on the 2026-10-15 renders - 2.5 on a small emission nebula (IC 1805, M 42) gives the structure its
    # range, the same on a frame-filling bright object (M 31) blows the core; 1.5 there.  The GHS docs leave D to taste ("until the object has the contrast you want").
    d2 = GHS_D2_SMALL if small else GHS_D2_FILLING
    decide("ghs_d2", d2, "object_frac", round(cov, 3), None,
           f"second GHS pass D {d2} ({'small subject' if small else 'frame-filling subject'}: 2.5 measured right on IC 1805 / M 42, 1.5 on M 31 whose core blew at 2.5)")

    # stretch without highlight protection: clipping measure
    tick("stretch")
    p1 = {**p0, **tuned, "hdr": 0.0, "stretch_method": method}
    fin = pro.Finisher(work, layers.header, p1, dual, external=False, checkpoint=checkpoint, keep_pictures=False)
    s0 = fin.stretch(base)
    st0 = pro.level_stats(s0)
    hdr_on = hdr_rule(st0["clipped"], st0["white"])
    hdr_value, st1 = (HDR_DUAL_DEFAULT if dual else 0.0), st0
    if hdr_on:
        hdr_value = 1.0
        tick("highlights")
        fin2 = pro.Finisher(work, layers.header, {**p1, "hdr": hdr_value}, dual, external=False, checkpoint=checkpoint, keep_pictures=False)
        s0 = fin2.stretch(base)
        st1 = pro.level_stats(s0)
        if method == "ghs2" and hdr_rule(st1["clipped"], st1["white"]) and st1["white"] >= st0["white"]:
            hdr_value = 1.0                                                      # the published HP protection did not help: Wavescale HDR as the second published tool
            tuned["hdr_method"] = "wavescale"
    if not hdr_on and hdr_value > 0:
        steps.append({"step": "hdr (default)", "from": 0.0, "to": hdr_value, "measured": None, "before": None, "target": None,
                      "why": "dual-band nebula: GHS HP < 1 in pass 2 at half strength (Siril GHS tutorial: HP protects bright nebula and stars); judged on IC 1805 / M 42 2026-10-15"})
    decide("hdr", hdr_value, "white_frac", {"clipped": st0["clipped"], "white": st0["white"]}, CLIP_LIMIT,
           (f"after the stretch {st0['clipped'] * 100:.3f} % of the starless pixels have a channel >= 0.995 and {st0['white'] * 100:.3f} % a lightness >= 0.97 (limit {CLIP_LIMIT * 100:.2f} %): "
            f"highlight protection on ({'GHS HP < 1 in pass 2' if method == 'ghs2' else 'the tool-native HDR or Wavescale HDR'}), then {st1['clipped'] * 100:.3f} % / {st1['white'] * 100:.3f} %")
           if hdr_on else f"after the stretch {st0['clipped'] * 100:.3f} % clipped, {st0['white'] * 100:.3f} % white: within the {CLIP_LIMIT * 100:.2f} % limit, no highlight protection",
           after={"clipped": st1["clipped"], "white": st1["white"]})

    # noise reduction, local contrast
    nxt = nxt_strength(snr)
    decide("luminance_denoise", nxt, "object_snr", round(snr, 2), None,
           f"per-pixel signal-to-noise of the object {snr:.1f}: NoiseXTerminator --dn {nxt} on the stretched starless (RC-Astro / Brecher: 0.6-0.85, the noisier the data the stronger)")
    decide("chroma_denoise", pro.NXT_DC, "object_snr", round(snr, 2), None, "NoiseXTerminator colour denoise 0.95 (--dc; NXT manual: colour 90-100 %)")
    lc = local_contrast_amount(snr)
    decide("local_contrast", lc, "object_snr", round(snr, 2), None,
           f"clahe blend {pro.LC_STRENGTH * lc:.2f} (published LHE strengths 0.2-0.35, Brecher): {'the lowest' if lc <= 0.55 else 'between the lowest and M51 second pass'} for SNR {snr:.1f}")

    # colour
    tick("colour")
    pal = fin.palette_name()
    mask, _ = pro.nebula_mask(finish.luminance(s0), pro._hist_stats(finish.luminance(s0))["mode"])
    if dual:
        user_pal = str(p0.get("palette", "auto"))
        if user_pal in ("", "auto"):
            decide("palette", "hoo", "dual-band filter", None, None, "dual-band: HOO Classic, Ha = R and OIII = (G + B) / 2 (Hubble_Palette_from_Dual-Band_OSC.py)")
        else:
            decide("palette", user_pal, "settings", None, None, f"the settings name the palette {user_pal}")
        decide("scnr", 0.0, "dual-band filter", None, None, "no SCNR on HOO: the green is the OIII (NarrowbandNormalization.py applies SCNR to SHO-type palettes only)")
        decide("oiii_boost", float(p0.get("oiii_boost", 1.0)), "dual-band filter", None, None, "NBN (OIII boost != 1) is only run on request")
        col = pro.Finisher(work, layers.header, {**p1, "palette": tuned["palette"], "oiii_boost": float(p0.get("oiii_boost", 1.0)), "hdr": hdr_value}, True, external=False,
                           checkpoint=checkpoint, keep_pictures=False).colour(s0)
    else:
        cast = green_cast(s0, mask)
        scnr = SCNR_AMOUNT if cast > SCNR_CAST else 0.0
        decide("scnr", scnr, "green cast", round(cast, 4), SCNR_CAST,
               f"median G - (R + B) / 2 over the subject {cast:+.3f}: " + (f"a green cast, SCNR {SCNR_AMOUNT} (SW Skies 0.60-0.75)" if scnr else "no green cast, no SCNR (PixInsight SPCC doc: calibration is kept)"))
        decide("palette", "natural", "broadband filter", None, None, "broadband: natural colour")
        col = s0
    sat0 = subject_saturation(col, mask)
    amount = satu_amount(sat0)
    if not dual:
        amount = min(amount, SATU_BROADBAND_MAX)                   # a galaxy / reflection nebula: the stronger satu tinted M 31's disc magenta; 0.3 is the published middle value
    else:
        amount = max(amount, SATU_DUAL_MIN)                        # HOO: at least the published upper value (judged on IC 1805 / M 42 2026-10-15; 0.25 left the nebula muted)
    # stars
    tick("stars")
    star_info = _star_density(layers, session)
    cls_stars = "none"
    if star_info is not None:
        red, cls_stars = star_settings(star_info["per_megapixel"])
        decide("star_reduction", red, "star_frac", star_info["per_megapixel"], None,
               f"{star_info['per_megapixel']:.0f} stars per megapixel ({cls_stars} field): reduction strength {red} (DSA star reduction value {0.5 * (1 - red):.2f}; Brecher 0.4, DSA default 0.2); "
               "lowered through the published 0.5 / 0.4 / 0.3 / 0.2 if the picture's star count would fall more than 18 % below the linear layer's (FINISH-SPEC gate: within 20 %)")
        amount_st, clip_st = pro.STAR_STRETCH, 0.0
        for amount_st in (5.0, 4.0, 3.0):
            sf = pro.Finisher(work, layers.header, {**p1, "star_saturation": 1.0, "star_intensity": 1.0, "star_stretch": amount_st}, dual, external=False, checkpoint=checkpoint,
                              keep_pictures=False)
            st = sf.stars(layers.stars)
            clip_st = float((st.max(0) >= 0.995).mean())
            if clip_st <= 0.8 * CLIP_LIMIT:                                # a margin: the screen blend and the colour boost clip a little more
                break
        decide("star_stretch", amount_st, "star_frac", star_info["per_megapixel"], pro.STAR_STRETCH,
               f"Seti Astro Star Stretch amount 5 (default; documented range 0-8): mtf 0 {pro.star_stretch_m(5.0):.6f} 1; "
               + (f"{clip_st * 100:.3f} % of the star layer's pixels are clipped (limit {CLIP_LIMIT * 100:.2f} %)" if amount_st == 5.0 and clip_st <= 0.8 * CLIP_LIMIT else
                  f"lowered to {amount_st:g} (published 5 / 4 / 3) so that the clipped fraction of the stretched stars is {clip_st * 100:.3f} % (limit {CLIP_LIMIT * 100:.2f} %)"))
        p1["star_stretch"] = amount_st
        sm = layers.starnet.get("stars") or {}
        level = float(pro.mtf_balance(pro.star_stretch_m(amount_st), 5.0 * float(sm.get("sky_noise") or 2e-5)))
        boost, ms = STAR_BOOSTS[0], {"saturation": None}
        for boost in [b for b in STAR_BOOSTS if dual or b <= STAR_BOOST_BROADBAND_MAX]:   # the smallest published boost whose stretched stars reach the colour limit (measured, not predicted)
            sb = pro.Finisher(work, layers.header, {**p1, "star_saturation": boost, "star_intensity": 1.0}, dual, external=False, checkpoint=checkpoint, keep_pictures=False)
            st_b = sb.stars(layers.stars)
            ms = gates.measure_stars(st_b, level / gates.STAR_SIGMA)
            if (ms["saturation"] or 0.0) >= STAR_COLOUR_MIN:
                break
        st = st_b
        decide("star_saturation", boost, "star_C", ms["saturation"], STAR_COLOUR_MIN,
               f"Seti colour boost {boost} (default 1.0, maximum 2.0): the smallest whose stretched stars' flux colour reaches saturation {STAR_COLOUR_MIN} (measured {ms['saturation']})")
        lin_n = (layers.starnet.get("stars") or {}).get("count")
        if lin_n:
            for cand in [c for c in (0.6, 0.5, 0.4, 0.3, 0.2) if c <= red]:
                rc = pro.Finisher(work, layers.header, {**p1, "star_reduction": cand}, dual, external=False, checkpoint=checkpoint, keep_pictures=False)
                full = rc.recombine(col, st)
                n_now = gates.measure_stars(gates.star_layer(full, col), level / gates.STAR_SIGMA)["count"]
                dev = abs(n_now - lin_n) / lin_n
                tuned["star_reduction"] = cand
                steps[[x["step"] for x in steps].index("star_reduction")].update({"to": cand, "after": {"star_count": n_now, "linear": lin_n, "deviation": round(dev, 3)}})
                if dev <= 0.18:
                    break
        decide("star_intensity", 1.0, "star_frac", star_info["per_megapixel"], None, "stars as stretched (the screen blend and the reduction value set their weight)")
    else:
        decide("star_reduction", p0["star_reduction"], "star_frac", None, None, "no separate star layer: the stars stay in the image")

    # gradient report (information)
    tick("final measurements")
    tuned_full = {**p0, **tuned, "stretch_method": method, "grade_temperature": 0.0, "grade_contrast": 0.0}
    tuned_full["saturation"] = round(1.0 + amount, 2)
    cls = ("dual-band " if dual else "broadband ") + ("nebula" if 0.0 < cov < 0.8 else "field")
    report = {
        "reference": "published values by fixed rules (research/pro-workflow.md); no reference images (those grade the Experiments tab's trials only)", "class": cls, "nearest": None,
        "seconds": round(time.time() - t0, 1), "steps": steps,
        "stats_after_stretch": st1, "coverage": round(cov, 4), "snr": round(snr, 2), "star_field": cls_stars, "created": datetime.now().isoformat(timespec="seconds"),
        "summary": f"{cls}: {method}, background {pro.SKY_TARGET}, " + ("HDR on, " if hdr_value else "") + f"NXT {nxt}, saturation {tuned_full['saturation']}",
    }
    result = {"params": tuned_full, "report": report}
    _save_json(os.path.join(session, "autofinish.json"), result)
    return result


def _snr(session: str, lum_lin: np.ndarray) -> float:
    """The object's per-pixel signal-to-noise as the restore stage measured it on the stack (before any denoiser); measured on the given luminance when it did not record it."""
    from .post import _load_json
    v = _load_json(os.path.join(session, "restore_meta.json")).get("snr")
    if v:
        return float(v)
    e = finish.EDGE_CROP
    return finish.object_snr(lum_lin[e:-e, e:-e] if min(lum_lin.shape) > 8 * e else lum_lin)


def _star_density(layers, session: str) -> dict | None:
    """Stars per megapixel of the full frame: from the removal's own measurement, else from the star layer."""
    from .post import _load_json
    if layers.stars is None:
        return None
    sm = _load_json(os.path.join(session, "starnet_meta.json")).get("stars")
    if sm and sm.get("per_megapixel") is not None:
        return {"count": sm["count"], "per_megapixel": float(sm["per_megapixel"])}
    noise = finish.lower_half_sigma(finish.sample(finish.luminance(layers.base)))
    n, per = finish.count_stars(layers.stars, noise)
    return {"count": n, "per_megapixel": per}
