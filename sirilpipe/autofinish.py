"""Auto-finish: choose the settings of the professional chain (``pro.py``) from measurements of the image, among PUBLISHED values only (research/FINISH-SPEC.md,
research/pro-workflow.md).  No fitting, no reference library: every setting is one of the published values, picked by a stated rule on a measured number, and the report lists
each decision with the measurement, the rule and its source (the web UI renders ``report.steps`` as one row per setting: ``step, from, to, measured, before, after, target, why``).

=====================  ======================================================================================================================================
setting                rule (source)
=====================  ======================================================================================================================================
stretch_method         the settings' stretch tool (``statstretch`` = Seti Astro Statistical Stretch unless another published tool is named)
stretch_boost          Statistical Stretch's curves boost: the largest value in the script's range 0 .. 1 whose stretch passes the clipping gates (no more than
                       0.05 % of the starless pixels with a channel >= 0.995 or Lab lightness >= 0.97); bisection on the image
ghs_d2                 GHS pass 2 D (when the GHS two-pass is the stretch tool): the largest value in Siril's ``ght -D`` range 0 .. 10 that passes the same gates
stretch                the background after the stretch: 0.10 (Brecher's five finished images 0.09 / 0.10 / 0.10 / 0.11 / 0.11); for Statistical Stretch the median target
                       follows the size of the target: 0.10 when it covers less than half of the frame, 0.25 (the script's default) when it fills it (pro-workflow.md A8)
black_point            measured on the stretched image: the sky median - 2.8 x its MADN (PixInsight ScreenTransferFunction / Siril autostretch shadows clipping), applied
                       in the last step with ``sky_levelling`` taking the sky back to the background level above
hdr                    on (1 = the published values: GHS HP < 1, Statistical Stretch -hdr 0.15 / 0.75, Wavescale HDR 5 / 1.5 / 5.0) when the stretch at the lowest boost / D
                       still fails the clipping gates (FINISH-SPEC; Siril / fas37: < 0.1 % clipped), else off
luminance_denoise      NoiseXTerminator ``--dn`` on the stretched starless: the lowest published strength 0.6 (RC-Astro / Brecher 0.6-0.85); the finishing chain
                       re-runs the tool harder - at most DENOISE_ESCALATION extra passes, to its own maximum - while the delivered sky is above the sky_pixel_noise
                       gate, and refuses a harder pass that fails another gate
chroma_denoise         NXT colour denoise 0.95 (NXT manual: colour 90-100 %)
palette / oiii_boost   dual-band: HOO Classic (Ha = R, OIII = (G + B) / 2), NBN only when the user picks the palette nbn (its OIII boost is the user's); broadband: natural colour
scnr                   broadband: 0.6 (SW Skies 0.60-0.75, low end) when the subject has a green cast (median G - (R + B) / 2 > 0.01 after the stretch), else 0; dual-band: 0
                       (green is the OIII; NBN never applies SCNR to HOO)
local_contrast         the largest blend of the Siril ``clahe`` passes within Brecher's published LHE strengths 0.2-0.35 whose result passes the clipping gates
saturation             the largest factor in the setting's range 1 .. 4 for the GHS saturation stretch (Siril ``ght -sat``) that pushes no more than 0.05 % of the
                       subject's pixels to the gamut boundary (HSL saturation >= 0.995: a channel pinned at 0 or 1, the clipping gate applied to colour) beyond what
                       the palette left there; bisection
star_reduction         the strongest published value first (0.6 = DSA value 0.2), lowered through 0.5 / 0.4 (Brecher) / 0.3 / 0.2 until the picture keeps the linear stars
                       (stars_count gate)
star_stretch           Seti Astro Star Stretch amount 5 (default; documented range 0-8)
star_saturation        Seti colour boost 1.0 (default) unless the stretched stars' flux colour is below 0.10 saturation: then 1.5, and 2.0 (the documented maximum) if still short
=====================  ======================================================================================================================================

``grade_*`` stay neutral.  The result is written to ``autofinish.json``.
"""
from __future__ import annotations

import os
import time
from datetime import datetime

import numpy as np

from . import finish, gates, pro

CLIP_LIMIT = 0.0005                    # 0.05 % (FINISH-SPEC)
STAR_REDUCTIONS = (0.6, 0.5, 0.4, 0.3, 0.2)      # published star reduction strengths, strongest first (DSA value 0.2 = 0.6; Brecher 0.4)
NXT_DN = (0.85, 0.60)                  # NXT --dn published range (RC-Astro / Brecher)
LC_BLEND = (0.5, 0.875)                # local_contrast: x LC_STRENGTH 0.4 = Brecher's published LHE strengths 0.2 - 0.35
SCNR_CAST = 0.01
SCNR_AMOUNT = 0.6
STAR_COLOUR_MIN = 0.10
STAR_BOOSTS = (1.0, 1.5, 2.0)
STAR_STRETCHES = (5.0, 4.0, 3.0, 2.0, 1.0)  # Seti Astro Star Stretch amounts, its default 5 down through its documented range 0 - 8
STATS_SIDE = 1200


STAT_BOOST_RANGE = (0.0, 1.0)                # Statistical_Stretch.py curves boost: the script's slider 0 .. 1.00
GHS_D2_RANGE = (0.05, 10.0)                  # Siril ght -D
SEARCH_STEPS = 5                             # bisection steps of a range search: 1/32 of the range
STF_SHADOWS = 2.8                            # MADN below the median: the shadows clipping of PixInsight's ScreenTransferFunction auto stretch and Siril's autostretch (-2.8)
GAMUT_S = 0.995                              # HSL saturation at the gamut boundary (a channel at 0 or 1)


def hdr_rule(clipped: float, white: float) -> bool:
    return clipped > CLIP_LIMIT or white > CLIP_LIMIT


def largest_passing(lo: float, hi: float, passes, steps: int = SEARCH_STEPS) -> tuple[float, bool]:
    """(value, passed): the largest value in [lo, hi] for which ``passes`` holds, assuming it holds below some value and fails above (more stretch, more
    clipping); ``hi`` when it passes there, ``lo`` with passed False when even ``lo`` fails."""
    if passes(hi):
        return hi, True
    if not passes(lo):
        return lo, False
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if passes(mid):
            lo = mid
        else:
            hi = mid
    return lo, True


def black_point(img: np.ndarray) -> tuple[float, float, float]:
    """(black point, sky median, sky MADN) of a stretched image's lightness over the gates' sky mask: the median - STF_SHADOWS x MADN, at least 0."""
    lum = finish.luminance(img)
    sm = gates.sky_mask(lum)
    v = lum[sm] if sm.sum() > 100 else finish.sample(lum)
    med, sig = float(np.median(v)), finish.mad_sigma(v)
    return max(0.0, med - STF_SHADOWS * sig), med, sig


def delivered(img: np.ndarray, bp: float) -> np.ndarray:
    """``img`` through the last step's levels as ``pro.Finisher.finalise`` applies them with ``sky_levelling``: the black point (linstretch -BP),
    then the midtones that take the sky (the gates' sky mask) to SKY_TARGET - so a clipping or star gate is measured on the picture delivered."""
    x = np.clip((np.asarray(img, np.float32) - bp) / max(1.0 - bp, 1e-9), 0.0, 1.0)
    lum = finish.luminance(x)
    sm = gates.sky_mask(lum)
    sky = float(np.median(lum[sm])) if sm.sum() > 100 else float(np.median(lum))
    y = pro.SKY_TARGET
    if not 1e-5 < sky < 0.999:
        return x
    m = float(np.clip(sky * (y - 1.0) / (2.0 * sky * y - sky - y), 0.02, 0.98))
    return pro.mtf_balance(m, x).astype(np.float32)


def gamut_fraction(img: np.ndarray, mask: np.ndarray | None) -> float:
    """The fraction of the subject's pixels at the gamut boundary (HSL saturation >= GAMUT_S)."""
    sat = pro.hsl_saturation(img)
    sel = sat[mask] if mask is not None and mask.sum() > 50 else sat
    return float((sel >= GAMUT_S).mean())


def green_cast(img: np.ndarray, mask: np.ndarray | None) -> float:
    """Median of G - (R + B) / 2 over the subject pixels (display units)."""
    g = img[1] - 0.5 * (img[0] + img[2])
    return float(np.median(g[mask])) if mask is not None and mask.sum() > 100 else float(np.median(g))


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
    steps_total = 7
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
    method = str(p0.get("stretch_method") or "statstretch")
    method = {"ghs": "ghs2", "mtf": "statstretch", "asinh": "statstretch", "log": "statstretch"}.get(method, method)
    decide("stretch_method", method, "object_snr", round(snr, 2), None,
           f"the settings name {method}")
    small = cov < pro.STAT_FILLING_COVERAGE
    decide("stretch", pro.SKY_TARGET, "object_frac", round(cov, 3), pro.SKY_TARGET,
           f"the background after the stretch is {pro.SKY_TARGET} (Brecher: 0.10 / 0.11 / 0.09 / 0.11 / 0.10 on five finished images)"
           + (f"; the target covers {cov * 100:.0f} % of the frame: a small target, so Statistical Stretch would send its median to {pro.STAT_SMALL}" if small else
              f"; the target fills {cov * 100:.0f} % of the frame: Statistical Stretch would use its default median {pro.STAT_FILLING}"))
    # stretch strength: the largest value of the tool's own range whose stretch passes the clipping gates (highlight protection off while searching)
    tick("stretch")
    p1 = {**p0, **tuned, "hdr": 0.0, "stretch_method": method}
    runs: dict = {}

    def stretched(extra: dict) -> np.ndarray:
        f = pro.Finisher(work, layers.header, {**p1, **extra}, dual, external=False, checkpoint=checkpoint, keep_pictures=False, coverage=getattr(layers, "coverage", None))
        return f.stretch(base)

    def try_value(key: str, v: float) -> bool:
        out = stretched({key: v})
        st = pro.level_stats(delivered(out, black_point(out)[0]))           # clipping as delivered: after the black point and the sky's midtones
        runs[round(v, 4)] = (out, st)
        return not hdr_rule(st["clipped"], st["white"])

    key, rng = ("stretch_boost", STAT_BOOST_RANGE) if method == "statstretch" else ("ghs_d2", GHS_D2_RANGE) if method == "ghs2" else (None, None)
    if key:
        v, passed = largest_passing(rng[0], rng[1], lambda x: try_value(key, x))
        v = round(v, 4)
        s0, st0 = runs[v]
        p1[key] = v
        tried = {k_: {"clipped": r[1]["clipped"], "white": r[1]["white"]} for k_, r in sorted(runs.items())}
        decide(key, v, "clipped / white", tried, CLIP_LIMIT,
               f"the largest {'Statistical Stretch curves boost' if key == 'stretch_boost' else 'GHS pass 2 D'} in {rng[0]:g} .. {rng[1]:g} whose stretch keeps clipped "
               f"and white pixels within {CLIP_LIMIT * 100:.2f} % (bisection, {len(runs)} stretches)" + ("" if passed else ": none did, the lowest stands and highlight protection is on"),
               after={"clipped": st0["clipped"], "white": st0["white"]})
    else:
        s0 = stretched({})
        st0 = pro.level_stats(s0)
        passed = not hdr_rule(st0["clipped"], st0["white"])
    hdr_on = not passed
    hdr_value, st1 = (1.0 if hdr_on else 0.0), st0
    if hdr_on:
        tick("highlights")
        s0 = stretched({"hdr": hdr_value})
        st1 = pro.level_stats(s0)
        if method == "ghs2" and hdr_rule(st1["clipped"], st1["white"]) and st1["white"] >= st0["white"]:
            tuned["hdr_method"] = "wavescale"                                    # the published HP protection did not help: Wavescale HDR as the second published tool
    decide("hdr", hdr_value, "white_frac", {"clipped": st0["clipped"], "white": st0["white"]}, CLIP_LIMIT,
           (f"the stretch at its lowest strength leaves {st0['clipped'] * 100:.3f} % clipped and {st0['white'] * 100:.3f} % white (limit {CLIP_LIMIT * 100:.2f} %): highlight protection on "
            f"({'GHS HP < 1 in pass 2' if method == 'ghs2' else 'the tool-native HDR or Wavescale HDR'}), then {st1['clipped'] * 100:.3f} % / {st1['white'] * 100:.3f} %")
           if hdr_on else "the stretch passes the clipping gates: no highlight protection",
           after={"clipped": st1["clipped"], "white": st1["white"]})
    bp, sky_med, sky_sig = black_point(s0)
    decide("black_point", round(bp, 5), "sky_L", {"median": round(sky_med, 5), "madn": float(f"{sky_sig:.3g}")}, None,
           f"sky median {sky_med:.4f} - {STF_SHADOWS} x MADN {sky_sig:.4g} (PixInsight STF / Siril autostretch shadows clipping), measured on the stretched starless")
    decide("sky_levelling", True, "sky_L", round(sky_med, 4), pro.SKY_TARGET, f"after the black point the last step's midtones take the sky back to {pro.SKY_TARGET}")
    decide("sky_level", pro.SKY_TARGET, "sky_L", round(sky_med, 4), pro.SKY_TARGET, "Brecher's background level")

    # noise reduction: the lowest published strength; the finishing chain raises it until the delivered sky passes the sky_pixel_noise gate
    nxt = NXT_DN[1]
    decide("luminance_denoise", nxt, "sky_pixel_noise", None, gates.LIMITS["sky_pixel_noise"],
           f"NoiseXTerminator --dn {nxt} on the stretched starless, the lowest of the published 0.6-0.85 (RC-Astro / Brecher); the chain re-runs the tool harder, "
           f"up to {pro.DENOISE_ESCALATION} passes, while the delivered sky's pixel noise is above its {gates.LIMITS['sky_pixel_noise']} gate")
    decide("chroma_denoise", pro.NXT_DC, None, None, None, "NoiseXTerminator colour denoise 0.95 (--dc; NXT manual: colour 90-100 %)")

    # colour
    tick("colour")
    mask, _ = pro.nebula_mask(finish.luminance(s0), pro._hist_stats(finish.luminance(s0))["mode"])
    if dual:
        user_pal = str(p0.get("palette", "auto"))
        if user_pal in ("", "auto"):
            decide("palette", "hoo", "dual-band filter", None, None, "dual-band: HOO Classic, Ha = R and OIII = (G + B) / 2 (Hubble_Palette_from_Dual-Band_OSC.py)")
        else:
            decide("palette", user_pal, "settings", None, None, f"the settings name the palette {user_pal}")
        decide("oiii_boost", float(p0.get("oiii_boost", 1.0)), "dual-band filter", None, None, "the OIII boost of the NBN palette is the user's setting (NBN runs only when picked as the palette)")
        pal_p = {**p1, "palette": tuned["palette"], "oiii_boost": float(p0.get("oiii_boost", 1.0)), "hdr": hdr_value, "scnr": 0.0}
        col = pro.Finisher(work, layers.header, pal_p, True, external=False, checkpoint=checkpoint, keep_pictures=False).colour(s0)
        if tuned["palette"] in pro.HA_GREEN_PALETTES:
            # Ha in the green channel (SHO / OHS of the Hubble palette script): the green cast rule of the broadband branch, measured after the palette
            cast = green_cast(col, mask)
            scnr = SCNR_AMOUNT if cast > SCNR_CAST else 0.0
            decide("scnr", scnr, "green cast", round(cast, 4), SCNR_CAST,
                   f"{tuned['palette'].upper()} puts Ha in green; median G - (R + B) / 2 over the subject {cast:+.3f}: "
                   + (f"SCNR {SCNR_AMOUNT} (SW Skies 0.60-0.75; NarrowbandNormalization.py applies SCNR to SHO-type palettes)" if scnr else "no green cast, no SCNR"))
            if scnr:
                col = pro.Finisher(work, layers.header, {**pal_p, "scnr": scnr}, True, external=False, checkpoint=checkpoint, keep_pictures=False).colour(s0)
        else:
            decide("scnr", 0.0, "dual-band filter", None, None, "no SCNR on HOO: the green is the OIII (NarrowbandNormalization.py applies SCNR to SHO-type palettes only)")
    else:
        cast = green_cast(s0, mask)
        scnr = SCNR_AMOUNT if cast > SCNR_CAST else 0.0
        decide("scnr", scnr, "green cast", round(cast, 4), SCNR_CAST,
               f"median G - (R + B) / 2 over the subject {cast:+.3f}: " + (f"a green cast, SCNR {SCNR_AMOUNT} (SW Skies 0.60-0.75)" if scnr else "no green cast, no SCNR (PixInsight SPCC doc: calibration is kept)"))
        decide("palette", "natural", "broadband filter", None, None, "broadband: natural colour")
        col = s0
    # local contrast: the largest published CLAHE strength whose result passes the clipping gates
    def lc_ok(v: float) -> bool:
        f = pro.Finisher(work, layers.header, {**p1, "local_contrast": v}, dual, external=False, checkpoint=checkpoint, keep_pictures=False)
        st = pro.level_stats(delivered(f.local_contrast(col), bp))
        lc_runs[round(v, 4)] = {"clipped": st["clipped"], "white": st["white"]}
        return not hdr_rule(st["clipped"], st["white"])

    lc_runs: dict = {}
    lc, _ok = largest_passing(LC_BLEND[0], LC_BLEND[1], lc_ok, steps=3)
    lc = round(lc, 3)
    decide("local_contrast", lc, "clipped / white", lc_runs, CLIP_LIMIT,
           f"clahe blend {pro.LC_STRENGTH * lc:.2f}: the largest of the published LHE strengths {pro.LC_STRENGTH * LC_BLEND[0]:.2f}-{pro.LC_STRENGTH * LC_BLEND[1]:.2f} "
           f"(Brecher) whose result keeps clipped and white pixels within {CLIP_LIMIT * 100:.2f} %")
    col = pro.Finisher(work, layers.header, {**p1, "local_contrast": lc}, dual, external=False, checkpoint=checkpoint, keep_pictures=False).local_contrast(col)
    g0 = gamut_fraction(col, mask)                                     # what the palette already put at the boundary is not the step's

    def sat_ok(k: float) -> bool:
        f = pro.Finisher(work, layers.header, {**p1, "saturation": k}, dual, external=False, checkpoint=checkpoint, keep_pictures=False)
        out = f.saturation(col)
        st = pro.level_stats(delivered(out, bp))
        g = gamut_fraction(out, mask) - g0
        sat_runs[round(k, 4)] = {"gamut": round(g, 6), "clipped": st["clipped"], "white": st["white"]}
        return g <= CLIP_LIMIT and not hdr_rule(st["clipped"], st["white"])   # nothing pushed out of gamut, and the clipping gates still pass

    sat_runs: dict = {}
    k_sat, _ok = largest_passing(1.0, pro.SAT_FACTOR_MAX, sat_ok)
    k_sat = round(k_sat, 3)
    decide("saturation", k_sat, "subject gamut / clipped", {"gamut_before": round(g0, 6), **dict(sorted(sat_runs.items()))}, CLIP_LIMIT,
           f"the largest factor in 1 .. {pro.SAT_FACTOR_MAX:g} for the GHS saturation stretch (Siril ght -sat) that pushes no more than {CLIP_LIMIT * 100:.2f} % of the subject "
           f"to the gamut boundary (HSL saturation >= {GAMUT_S}: a channel pinned at 0 or 1) beyond the {g0 * 100:.2f} % the palette left there, with the "
           f"clipping gates still passing; bisection, {len(sat_runs)} trials")
    col = pro.Finisher(work, layers.header, {**p1, "saturation": k_sat}, dual, external=False, checkpoint=checkpoint, keep_pictures=False).saturation(col)
    # stars
    tick("stars")
    star_info = _star_density(layers, session)
    cls_stars = "none"
    if star_info is not None:
        red, cls_stars = STAR_REDUCTIONS[0], f"{star_info['per_megapixel']:.0f} per MP"
        decide("star_reduction", red, "star_frac", star_info["per_megapixel"], None,
               f"the strongest published reduction {red} (DSA star reduction value {0.5 * (1 - red):.2f}), lowered through {' / '.join(f'{c:g}' for c in STAR_REDUCTIONS[1:])} "
               "until the picture's star count falls no more than 18 % below the linear layer's (FINISH-SPEC gate: within 20 %)")
        amount_st, clip_st = pro.STAR_STRETCH, 0.0
        for amount_st in STAR_STRETCHES:
            sf = pro.Finisher(work, layers.header, {**p1, "star_saturation": 1.0, "star_intensity": 1.0, "star_stretch": amount_st}, dual, external=False, checkpoint=checkpoint,
                              keep_pictures=False)
            st = sf.stars(layers.stars)
            clip_st = float((delivered(sf.recombine(col, st), bp).max(0) >= 0.995).mean())   # where the gate measures it: screened over the picture, as delivered
            if clip_st <= CLIP_LIMIT:
                break
        decide("star_stretch", amount_st, "star_frac", star_info["per_megapixel"], pro.STAR_STRETCH,
               f"Seti Astro Star Stretch amount 5 (default; documented range 0-8): mtf 0 {pro.star_stretch_m(5.0):.6f} 1; "
               + (f"{clip_st * 100:.3f} % of the recombined picture's pixels are clipped (limit {CLIP_LIMIT * 100:.2f} %)" if amount_st == 5.0 and clip_st <= CLIP_LIMIT else
                  f"lowered to {amount_st:g} (5 / 4 / 3 / 2 / 1, the documented range): the stars screened over the picture leave {clip_st * 100:.3f} % of its pixels clipped "
                  f"(limit {CLIP_LIMIT * 100:.2f} %)"))
        p1["star_stretch"] = amount_st
        sm = layers.starnet.get("stars") or {}
        sky_n = float(sm.get("sky_noise") or 2e-5)

        def star_colour(prm):
            # measured as the gates measure it, on the recombined picture (screen blend, star reduction, halo suppression): colour (stars_coloured),
            # colours on the blackbody locus (stars_physical) and the linear layer's stars still visible (stars_count) - on the star layer alone
            # M 31's stars read 0.10 at boost 1.0 and the finished picture 0.034
            fb = pro.Finisher(work, layers.header, prm, dual, external=False, checkpoint=checkpoint, keep_pictures=False)
            st_x = fb.stars(layers.stars)
            full = delivered(fb.halos(fb.recombine(col, st_x)), bp)
            lvl = float(pro.mtf_balance(pro.star_stretch_m(float(prm.get("star_stretch", amount_st))), 5.0 * sky_n))
            g_ = gates.compute(full, starless=delivered(col, bp), linear_stars=layers.stars, linear_noise=sky_n, star_level=lvl, linear_base=layers.base)
            unclipped = g_["clipped"]["pass"] and g_["white"]["pass"]           # a colour boost must not push the cores' brightest channel to 1
            return st_x, {"saturation": g_["stars_coloured"]["value"], "off_locus": g_["stars_physical"]["value"], "clipped": g_["clipped"]["value"],
                          "ok": unclipped and all(g_[k]["pass"] for k in ("stars_coloured", "stars_physical", "stars_count")),
                          "physical": unclipped and g_["stars_physical"]["pass"] and g_["stars_count"]["pass"]}

        # the smallest intervention whose stars pass the three star gates: the colour boost first (Seti's published 1 / 1.5 / 2), then a gentler
        # star stretch (the published 5 / 4 / 3, never above the clipping-safe amount) - on M 31 the stretch of 5 left the stars white at every
        # boost (0.05 at 2, with 76 % of them green or magenta), a stretch of 3 with the boost 2 gave 0.115, all on the locus, all kept - and only
        # then colour preservation (the Lupton et al. 2004 colour-preserving star stretch, pro.Finisher.stars)
        amounts = [a_ for a_ in STAR_STRETCHES if a_ <= amount_st]
        cands = [(am, bo, 0.0) for am in amounts for bo in STAR_BOOSTS] + [(am, bo, cp) for cp in (0.25, 0.5, 0.75, 1.0) for am in amounts for bo in STAR_BOOSTS]
        chosen, best_phys = None, None
        for am, bo, cp in cands:
            st_x, ms_x = star_colour({**p1, "star_stretch": am, "star_saturation": bo, "star_intensity": 1.0, "star_color_preservation": cp})
            if ms_x["physical"] and (best_phys is None or (ms_x["saturation"] or 0.0) > (best_phys[4]["saturation"] or 0.0)):
                best_phys = (am, bo, cp, st_x, ms_x)
            if ms_x["ok"]:
                chosen = (am, bo, cp, st_x, ms_x)
                break
        am, boost, cp, st, ms = chosen or best_phys or (amount_st, STAR_BOOSTS[0], 0.0, *star_colour({**p1, "star_saturation": STAR_BOOSTS[0], "star_intensity": 1.0}))
        if am != amount_st:
            decide("star_stretch", am, "star_C", ms["saturation"], STAR_COLOUR_MIN,
                   f"Seti Astro Star Stretch amount {am:g} (5 / 4 / 3 / 2 / 1, the documented range): the stars of the clipping-safe amount {amount_st:g} did not pass the star gates "
                   "at any colour boost")
            p1["star_stretch"] = am
        level = float(pro.mtf_balance(pro.star_stretch_m(p1["star_stretch"]), 5.0 * sky_n))
        decide("star_saturation", boost, "star_C", ms["saturation"], STAR_COLOUR_MIN,
               f"Seti colour boost {boost} with colour preservation {cp}: the smallest intervention whose stars, in the recombined picture, pass the star gates "
               f"(colour {ms['saturation']} >= {STAR_COLOUR_MIN}, off the blackbody locus {ms.get('off_locus')}, the linear stars kept)"
               + ("" if chosen else " - none passed all three: the most coloured of the physically coloured"))
        decide("star_color_preservation", cp, "star_C", ms["saturation"], STAR_COLOUR_MIN, "with the colour boost above")
        lin_n = (layers.starnet.get("stars") or {}).get("count")
        if lin_n:
            for cand in STAR_REDUCTIONS:
                rc = pro.Finisher(work, layers.header, {**p1, "star_reduction": cand}, dual, external=False, checkpoint=checkpoint, keep_pictures=False)
                full = delivered(rc.halos(rc.recombine(col, st)), bp)
                sc = gates.compute(full, starless=delivered(col, bp), linear_stars=layers.stars, linear_noise=sky_n, star_level=level)["stars_count"]["value"]
                dev = float(sc.get("deviation") or 0.0) if isinstance(sc, dict) else 0.0     # the share of the linear stars no longer visible
                tuned["star_reduction"] = cand
                steps[[x["step"] for x in steps].index("star_reduction")].update({"to": cand, "after": {"stars_count": sc, "deviation": round(dev, 3)}})
                if dev <= 0.18:
                    break
        decide("star_intensity", 1.0, "star_frac", star_info["per_megapixel"], None, "stars as stretched (the screen blend and the reduction value set their weight)")
        # Cosmic Clarity Sharpen's stellar amount: the sharpening runs on the composed picture and tightens the star cores, so the largest amount in
        # the script's 0 .. 1 whose result still passes the clipping gates, measured as delivered (the non-stellar amount stays the settings')
        if str(p0.get("cc_sharpen") or "off").lower() in ("both", "stellar"):
            tick("sharpening")
            pc = {**p1, **tuned}
            comp_f = pro.Finisher(work, layers.header, pc, dual, external=False, checkpoint=checkpoint, keep_pictures=False)
            composed = comp_f.halos(comp_f.recombine(col, st))
            cc_runs: dict = {}

            def cc_ok(v: float) -> bool:
                f = pro.Finisher(work, layers.header, {**pc, "cc_stellar_amount": v}, dual, external=True, checkpoint=checkpoint, keep_pictures=False)
                stx = pro.level_stats(delivered(f.cc_sharpen(composed), bp))
                cc_runs[round(v, 4)] = {"clipped": stx["clipped"], "white": stx["white"]}
                return not hdr_rule(stx["clipped"], stx["white"])

            amt, _ok = largest_passing(0.0, 1.0, cc_ok, steps=4)
            decide("cc_stellar_amount", round(amt, 3), "clipped / white", dict(sorted(cc_runs.items())), CLIP_LIMIT,
                   f"the largest Cosmic Clarity Sharpen stellar amount in 0 .. 1 whose result keeps clipped and white pixels within {CLIP_LIMIT * 100:.2f} % "
                   f"of the delivered picture (the sharpening tightens the star cores); bisection, {len(cc_runs)} runs")
    else:
        decide("star_reduction", p0["star_reduction"], "star_frac", None, None, "no separate star layer: the stars stay in the image")

    # gradient report (information)
    tick("final measurements")
    tuned_full = {**p0, **tuned, "stretch_method": method, "grade_temperature": 0.0, "grade_contrast": 0.0}
    cls = ("dual-band " if dual else "broadband ") + ("nebula" if 0.0 < cov < 0.8 else "field")
    report = {
        "reference": "published values by fixed rules (research/pro-workflow.md); no reference images (those grade the Experiments tab's trials only)", "class": cls, "nearest": None,
        "seconds": round(time.time() - t0, 1), "steps": steps,
        "stats_after_stretch": st1, "coverage": round(cov, 4), "snr": round(snr, 2), "star_field": cls_stars, "created": datetime.now().isoformat(timespec="seconds"),
        "summary": f"{cls}: {method}, background {pro.SKY_TARGET}, black point {tuned_full['black_point']}, " + ("HDR on, " if hdr_value else "") + f"NXT {nxt}, saturation {tuned_full['saturation']}",
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
