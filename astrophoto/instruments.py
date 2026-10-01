"""Which telescope took a sub, and what its FITS headers leave out.

Every value the pipeline needs normally comes from the FITS headers (``frames.read_info``).
Smart telescopes write fewer header keys than an astronomy camera driver does, so a
profile of the instrument fills the gaps: focal length, pixel size, Bayer pattern and
the sensor's spectral curve (for colour calibration).  A header value always wins over
the profile.

DWARFLAB DWARF telescopes also encode the capture settings in the file and folder names
and in a ``shotsInfo.json`` next to the subs; those are read as a fallback too:

    DWARF_RAW_TELE_M 31_EXP_15_GAIN_60_2025-10-01-21-30-00-100/
        shotsInfo.json                               target, exp, gain, ir, binning, temps, RA/Dec
        M 31_15s60_Astro_20251001-213012345_37C.fits          a light sub
        failed_M 31_15s60_Astro_20251001-213530123_37C.fits   a sub the DWARF's live stack rejected
        stacked-...fits                                       the DWARF's own stack (ignored)
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Profile:
    key: str
    name: str
    focallen: float | None = None     # mm
    pixsize: float | None = None      # um, unbinned
    bayer: str | None = None
    sensor: str | None = None         # Siril SPCC database sensor name
    aperture: float | None = None     # mm
    dwarf: bool = False               # DWARFLAB file layout (CALI_FRAME, shotsInfo.json, file names)
    bit_depth: int | None = None      # ADC bits when the FITS data is not scaled to 16 bits


PROFILES = {
    # telephoto: Sony IMX678 (3840x2160, 2.0 um, RGGB), 35 mm f/4.3; 2x2 binning gives 1920x1080 at 4 um.
    # Its subs and CALI_FRAME masters hold the 12-bit ADC values unscaled (0..4095; black level ~200)
    "dwarf3": Profile("dwarf3", "DWARF 3", 150.0, 2.0, "RGGB", "Sony_IMX678", 35.0, dwarf=True, bit_depth=12),
    # the wide-angle camera (cam_1) has other optics: only its calibration layout is known here
    "dwarf3_wide": Profile("dwarf3_wide", "DWARF 3 wide-angle", dwarf=True, bit_depth=12),
    "dwarf_mini": Profile("dwarf_mini", "DWARF mini", 150.0, 2.9, None, "Sony_IMX662", 30.0, dwarf=True),
    "dwarf2": Profile("dwarf2", "DWARF II", 100.0, 1.45, None, "Sony_IMX415", 24.0, dwarf=True),
    "seestar_s50": Profile("seestar_s50", "ZWO Seestar S50", 250.0, 2.9, "GRBG", None, 50.0),
    "seestar_s30": Profile("seestar_s30", "ZWO Seestar S30", 150.0, 2.9, "GRBG", None, 30.0),
}
# a camera none of the profiles recognises: the defaults the pipeline has always used
GENERIC = Profile("generic", "", 250.0, 2.9, "GRBG")

# [failed_]<target>_<exp>s<gain>_<filter>_<YYYYMMDD>-<hhmmss[fff]>_<temp>C.fits
DWARF_SUB_RE = re.compile(
    r"^(?P<failed>failed[_\-])?(?P<target>.+?)_(?P<exp>[0-9.]+)s(?P<gain>[0-9.]+)_(?P<filter>[^_]+)_"
    r"(?P<date>\d{8})-(?P<time>\d{6,9})_(?P<temp>[-+]?[0-9.]+)C\.(?:fits?|fts)$", re.IGNORECASE)

# the DWARF flat's "ir" code in CALI_FRAME file names: 0 = VIS (UV/IR cut), 1 = Astro, 2 = Duo-Band
DWARF_FILTERS = {0: "VIS", 1: "Astro", 2: "Duo-Band"}


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def filter_code(name) -> int | None:
    """DWARF filter code (the ``ir_N`` of its flats) of a filter name, None if unknown."""
    if isinstance(name, (int, float)) and int(name) in DWARF_FILTERS:
        return int(name)
    n = _norm(name)
    if n in ("0", "1", "2"):
        return int(n)
    if any(k in n for k in ("duo", "dual", "band", "narrow", "hoo", "lp")):
        return 2
    if "astro" in n:
        return 1
    if any(k in n for k in ("vis", "ircut", "uvir", "none", "clear", "off")):
        return 0
    return None


def identify(header, path: str) -> Profile:
    """The telescope that wrote ``header`` (headers first, then the file and folder names)."""
    text = " ".join(str(header.get(k, "") or "") for k in ("TELESCOP", "INSTRUME", "ORIGIN", "CREATOR", "SWCREATE"))
    n = _norm(text)
    folder = os.path.basename(os.path.dirname(os.path.abspath(path)))
    wide = re.search(r"(?:^|[_\W])WIDE(?:[_\W]|$)", folder.upper()) or _norm(header.get("CAMNAME")) == "wide"
    if "dwarfmini" in n:
        return PROFILES["dwarf_mini"]
    if "dwarfiii" in n or "dwarf3" in n:
        return PROFILES["dwarf3_wide" if wide else "dwarf3"]
    if "dwarfii" in n or "dwarf2" in n:
        return PROFILES["dwarf2"]
    if "seestar" in n:
        return PROFILES["seestar_s30" if "s30" in n else "seestar_s50"]
    looks_dwarf = ("dwarf" in n or DWARF_SUB_RE.match(os.path.basename(path)) or folder.upper().startswith("DWARF_")
                   or shots_info(os.path.dirname(os.path.abspath(path))))
    if looks_dwarf:
        # a DWARF that does not name its model: its sensor's pixel size tells the DWARF II apart
        px = float(header.get("XPIXSZ", 0) or 0)
        if 1.3 < px < 1.6:
            return PROFILES["dwarf2"]
        return PROFILES["dwarf3_wide" if wide else "dwarf3"]
    return GENERIC


def parse_dwarf_name(path: str) -> dict:
    """Capture settings encoded in a DWARF sub's file name ({} when it is not one)."""
    m = DWARF_SUB_RE.match(os.path.basename(path))
    if not m:
        return {}
    d, t = m.group("date"), m.group("time")
    frac = t[6:]
    iso = f"{d[:4]}-{d[4:6]}-{d[6:]}T{t[:2]}:{t[2:4]}:{t[4:6]}" + (f".{frac.ljust(6, '0')[:6]}" if frac else "")
    return {"object": m.group("target").strip(), "exptime": float(m.group("exp")), "gain": float(m.group("gain")),
            "filter": m.group("filter"), "date_obs": iso, "temp": float(m.group("temp")),
            "failed": bool(m.group("failed"))}


def coord(v, hours: bool) -> float | None:
    """A coordinate in degrees from a number or a sexagesimal string, either in hours (``hours``)
    or in degrees."""
    if v is None or v == "":
        return None
    try:
        return float(v) * (15.0 if hours else 1.0)
    except (TypeError, ValueError):
        pass
    try:
        from astropy.coordinates import Angle
        return float(Angle(str(v), unit="hourangle" if hours else "deg").degree)
    except Exception:
        return None


@lru_cache(maxsize=64)
def shots_info(folder: str) -> dict:
    """The DWARF session's shotsInfo.json, normalised ({} when there is none)."""
    p = os.path.join(folder, "shotsInfo.json")
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}
    low = {str(k).lower(): v for k, v in raw.items()} if isinstance(raw, dict) else {}

    def get(*keys):
        for k in keys:
            if k in low and low[k] not in (None, ""):
                return low[k]
        return None
    out = {"raw": raw}
    for key, names in (("target", ("target", "object")), ("exptime", ("exp", "exposure", "exptime")),
                       ("gain", ("gain",)), ("filter", ("ir", "filter"))):
        v = get(*names)
        if v is not None:
            out[key] = v
    b = get("binning", "bin")
    if b is not None:
        try:
            out["binning"] = int(str(b).replace("x", "*").split("*")[0])
        except ValueError:
            pass
    temps = [float(v) for v in (get("mintemp"), get("maxtemp")) if v is not None]
    if temps:
        out["temp"] = sum(temps) / len(temps)
        out["temp_range"] = [min(temps), max(temps)]
    ra, dec = get("ra", "ra_deg", "radeg", "rightascension"), get("dec", "dec_deg", "decdeg", "declination")
    out["ra"] = coord(ra, hours=True)          # DWARF 3: "RA": 20.98 for 314.7 deg - hours, as numbers too
    out["dec"] = coord(dec, hours=False)
    for key in ("shotstaken", "shotsstacked"):
        if key in low:
            out[key] = low[key]
    return out


def camera_slot(profile: Profile) -> str | None:
    """The DWARF calibration folder of the camera (cam_0 telephoto, cam_1 wide-angle)."""
    if not profile.dwarf:
        return None
    return "cam_1" if profile.key.endswith("_wide") else "cam_0"

