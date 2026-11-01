"""Narrowband normalisation (HOO) of dual-band data: a port of ``process_image`` of siril-scripts ``processing/NarrowbandNormalization.py`` (Yannick Dutertre "Cuiv", v1.0.5,
GPL-3.0-or-later; clean-room port of the maths of Bill Blanshan and Mike Cranfield's NarrowbandNormalization PixelMath script, https://cosmicphotons.com/scripts/).

The script has no command-line mode (a PyQt6 window), so Siril cannot run it headless.  Its core function is GUI-free numpy; this module is the same arithmetic for the HOO
palette (Ha in R, OIII in B, synthetic green) with the lightness stage left off, and ``tests/test_pro.py`` compares it with the script's own function, run through
``community_worker.py --entry process_image``, to 1e-4 (the tolerance of the specification; the two agree to float32 rounding).

Use it on STRETCHED data (the script: "unlike Veralux Alchemy, this script should be used on stretched data").

Defaults are the script's reset values (``_on_reset``): shadow point 1.0, OIII boost 1.0, blend Mode 1 with amount 0.6, highlight reduction 1.0, brightness 1.0, no SCNR on HOO
(green is the OIII: the script applies SCNR to the SHO-type palettes only).
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-6                    # the script's float32 divide-by-zero guard

DEFAULTS = {"palette": "HOO", "lightness": "Off", "blend_mode": "Mode 1", "blend_amount": 0.6, "scnr": 0, "oiii_boost": 1.0, "sii_boost": 1.0,
            "shadow_point": 1.0, "highlight_reduction": 1.0, "brightness": 1.0}


def _mtf(m: float, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if abs(m - 0.5) < 1e-9:
        return x.copy()
    denom = (2.0 * m - 1.0) * x - m
    denom = np.where(np.abs(denom) < _EPS, np.copysign(_EPS, denom), denom)
    return ((m - 1.0) * x) / denom


def _rescale(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if abs(hi - lo) < _EPS:
        return np.clip(x - lo, 0.0, 1.0)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _channel_stats(ch: np.ndarray, blackpoint: float) -> tuple[float, float]:
    """(M, E0) as the PixelMath M / E0 expressions: M = min + shadow x (median - min); E0 = adev / 1.2533 + mean - M."""
    mn, med = float(np.min(ch)), float(np.median(ch))
    M = mn + blackpoint * (med - mn)
    mean = float(np.mean(ch, dtype=np.float64))
    adev = float(np.mean(np.abs(ch - mean), dtype=np.float64))
    return M, adev / 1.2533 + mean - M


def _boost_factor(a_target: float, a_ref: float, boost: float) -> float:
    denom = a_target - 2.0 * a_target * a_ref + a_ref
    if abs(denom) < 1e-9:
        denom = 1e-9
    return (a_target * (1.0 - a_ref) / denom) / boost


def _synthetic_green(ha: np.ndarray, oiii: np.ndarray, mode: str, amount: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    if mode == "Mode 1":
        g = amount * ha + (1.0 - amount) * oiii
    elif mode == "Mode 2":
        g = (np.clip(ha, 0, 1) ** amount) * (np.clip(oiii, 0, 1) ** (1.0 - amount))
    else:
        g = 1.0 - (1.0 - amount * ha) * (1.0 - (1.0 - amount) * oiii)
    return np.clip(g, 0.0, 1.0)


def normalise(data: np.ndarray, **params) -> np.ndarray:
    """HOO narrowband normalisation of the stretched ``data`` (H, W, 3) in [0, 1] with Ha in R and OIII in B (G is replaced by the synthetic green).  ``params`` override
    ``DEFAULTS`` (``oiii_boost``, ``blend_mode``, ``blend_amount``, ``shadow_point``, ``highlight_reduction``, ``brightness``)."""
    q = {**DEFAULTS, **params}
    data = np.asarray(data, dtype=np.float32)
    ha, oiii = data[:, :, 0], data[:, :, 2]
    M_ha, E0_ha = _channel_stats(ha, q["shadow_point"])
    M_o, E0_o = _channel_stats(oiii, q["shadow_point"])
    ref = 1.0 - M_o
    if abs(ref) < 1e-9:
        ref = 1e-9
    A0_ha, A0_o = E0_ha / ref, E0_o / ref
    E1 = _boost_factor(A0_o, A0_ha, q["oiii_boost"])
    stretched = _mtf(E1, _rescale(oiii, M_o, 1.0))
    oiii_norm = np.clip(1.0 - (1.0 - stretched) * (1.0 - np.minimum(oiii, M_o)), 0.0, 1.0)
    out = np.empty_like(data)
    out[:, :, 0] = ha
    out[:, :, 2] = oiii_norm
    out[:, :, 1] = _synthetic_green(ha, oiii_norm, q["blend_mode"], q["blend_amount"])
    hl = max(float(q["highlight_reduction"]), 1e-3)                  # E11: highlight reduction
    out = _mtf(1.0 - 0.5 / hl, out) * out + out * (1.0 - out)
    out = _mtf(1.0 / max(float(q["brightness"]), 1e-3) * 0.5, out)  # E12: brightness
    return _rescale(out, 0.0, 1.0).astype(np.float32)


def info(data: np.ndarray, **params) -> dict:
    """The numbers the normalisation derived (for the report): M and E0 of Ha and OIII, the boost factor E1."""
    q = {**DEFAULTS, **params}
    ha, oiii = data[:, :, 0], data[:, :, 2]
    M_ha, E0_ha = _channel_stats(ha, q["shadow_point"])
    M_o, E0_o = _channel_stats(oiii, q["shadow_point"])
    ref = max(1.0 - M_o, 1e-9)
    return {"M_ha": round(M_ha, 6), "M_oiii": round(M_o, 6), "A0_ha": round(E0_ha / ref, 6), "A0_oiii": round(E0_o / ref, 6),
            "E1": round(_boost_factor(E0_o / ref, E0_ha / ref, q["oiii_boost"]), 6)}
