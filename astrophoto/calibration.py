"""Calibration masters (bias, dark, flat) applied to every light before anything else.

Where they come from
    * a DWARFLAB DWARF's ``CALI_FRAME`` library, found next to the session folder (the
      DWARF's ``Astronomy`` folder, or any parent up to three levels).  It holds the factory
      and user-captured masters, stacked on the telescope, one folder per camera::

          CALI_FRAME/bias/cam_0/bias_gain_2_bin_1.fits
          CALI_FRAME/dark/cam_0/dark_exp_15.000000_gain_60_bin_1_38C_stack_10.fits
          CALI_FRAME/flat/cam_0/flat_gain_2_bin_1_ir_1.fits        (ir: 0 VIS, 1 Astro, 2 Duo-Band)

    * any camera: ``darks/``, ``flats/``, ``biases/`` (or ``dark``, ``flat``, ``bias``, ``offsets``)
      folders in the session folder or beside it (the Siril layout), and calibration frames
      (``IMAGETYP`` Dark / Flat / Bias) among the lights.  Individual frames are median-combined
      into a master once, cached in the session folder.
    * ``ASTROPHOTO_CALIB``: extra library folders (``os.pathsep``-separated), or ``off``.

Which master is used
    The frame size must match (binning is implied) and so must the camera slot (DWARF cam_0 /
    cam_1).  Dark: same gain and exposure, then the nearest sensor temperature, then the larger
    stack.  A dark of another exposure is only used with a bias (its thermal part is scaled).
    Flat: the same filter (DWARF ``ir`` code, or the FILTER name).  Bias: the nearest gain.

How it is applied (ADU)
    light - bias - k * (dark - bias), divided by the flat normalised to 1 in each CFA site
    (the two greens separately, as Siril's -equalize_cfa: the flat never changes colour balance).
    k is fitted per sub on the dark's hot pixels (dark optimisation, as Siril's ``-opt``):
    an uncooled sensor's dark current doubles every ~6 C, and a DWARF's sensor drifts by 10 C
    and more in a night, so a dark taken at another temperature still subtracts the right amount
    of thermal signal and amp glow.  Without a bias the dark is subtracted as it is.
    Pixels saturated in the raw sub stay saturated after the flat, and pixels the masters show
    to be unusable (hot beyond linear range, dead in the flat) join the defect map.
"""
from __future__ import annotations

import glob
import hashlib
import os
import re
from dataclasses import dataclass, asdict

import numpy as np
from astropy.io import fits

from .instruments import DWARF_FILTERS, filter_code

WHITE = 65535.0
SAT_RAW = 0.98 * WHITE          # a raw 16-bit pixel at or above this is saturated (12-bit ADCs top out at 65520)
MAX_FILES = 3000                # calibration files looked at per library
MAX_COMBINE = 64                # individual frames median-combined into a master
FIT_SAMPLE = 6                  # subs whose dark scale is fitted when the calibration is attached (for the report)

_DIR_KINDS = {"bias": "bias", "biases": "bias", "offset": "bias", "offsets": "bias",
              "dark": "dark", "darks": "dark", "flat": "flat", "flats": "flat"}


@dataclass
class Master:
    path: str
    kind: str                       # bias | dark | flat
    shape: tuple
    exptime: float | None = None
    gain: float | None = None
    temp: float | None = None
    binning: int | None = None
    filter: str = ""
    fcode: int | None = None        # DWARF ir code
    cam: str | None = None          # DWARF camera slot
    stack: int = 1
    is_master: bool = False

    @property
    def name(self):
        return os.path.basename(self.path)


# ================================================================ parsing
def _num(pattern: str, s: str, cast=float):
    m = re.search(pattern, s, re.IGNORECASE)
    return cast(m.group(1)) if m else None


def _kind(text: str) -> str | None:
    t = text.lower()
    for k in ("dark", "flat", "bias", "offset"):
        if k in t:
            return "bias" if k == "offset" else k
    return None


def parse_master(path: str, kind_hint: str | None = None) -> Master | None:
    """A calibration frame's metadata: FITS headers first, then the (DWARF) file name tokens."""
    try:
        h = fits.getheader(path)
    except Exception:
        return None
    if int(h.get("NAXIS", 0)) != 2:
        return None                 # a debayered (colour) master cannot calibrate CFA data
    name = os.path.basename(path)
    typ = str(h.get("IMAGETYP", "") or h.get("IMAGETYPE", "") or h.get("FRAME", "") or "")
    kind = _kind(typ) or _kind(name.split("_")[0]) or kind_hint
    if kind not in ("bias", "dark", "flat"):
        return None
    rel = path.replace("\\", "/")

    def hv(*keys):
        for k in keys:
            v = h.get(k)
            if v not in (None, ""):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return None
    m = Master(path=path, kind=kind, shape=(int(h["NAXIS2"]), int(h["NAXIS1"])))
    m.exptime = hv("EXPTIME", "EXPOSURE")
    if m.exptime is None:
        m.exptime = _num(r"exp[_\-]([0-9.]+)", name)
    m.gain = hv("GAIN")
    if m.gain is None:
        m.gain = _num(r"gain[_\-]([0-9.]+)", name)
    m.temp = hv("CCD-TEMP", "SET-TEMP", "TEMPERAT")
    if m.temp is None:
        m.temp = _num(r"(?:^|_)([-+]?[0-9]+(?:\.[0-9]+)?)C(?:_|\.|\(|$)", name)
    b = hv("XBINNING")
    m.binning = int(b) if b else _num(r"bin[_\-]([0-9]+)", name, int)
    stack = hv("STACKCNT", "NCOMBINE")
    stack = int(stack) if stack else _num(r"stack[_\-]([0-9]+)", name, int)
    m.stack = stack or 1
    m.is_master = m.stack > 1 or "master" in name.lower() or "/CALI_FRAME/" in rel.upper() + "/"
    m.filter = str(h.get("FILTER", "") or "").strip()
    ir = _num(r"(?:^|_)ir[_\-]?([0-9])(?:_|\.|$)", name, int)
    m.fcode = ir if ir is not None else (filter_code(m.filter) if m.filter else None)
    if not m.filter and m.fcode is not None and ir is not None:
        m.filter = DWARF_FILTERS.get(m.fcode, "")
    cam = re.search(r"(?:^|/)(cam_[0-9])(?:/|$)", rel, re.IGNORECASE)
    m.cam = cam.group(1).lower() if cam else ({"tele": "cam_0", "wide": "cam_1"}.get(
        str(h.get("CAMNAME", "")).strip().lower()))
    return m


# ================================================================ finding the library
def _fits_under(root: str, depth: int) -> list[str]:
    out = []
    for d in range(depth + 1):
        for ext in ("fit", "fits", "fts", "FIT", "FITS", "FTS"):
            out.extend(glob.glob(os.path.join(root, *(["*"] * d), f"*.{ext}")))
        if len(out) > MAX_FILES:
            break
    return sorted(set(out))[:MAX_FILES]


def _child(parent: str, name: str) -> str | None:
    """A child folder of ``parent`` named ``name``, any case."""
    try:
        for c in os.listdir(parent):
            if c.lower() == name.lower() and os.path.isdir(os.path.join(parent, c)):
                return os.path.join(parent, c)
    except OSError:
        pass
    return None


def find_dwarf_library(folder: str) -> str | None:
    """The DWARF CALI_FRAME folder for a session folder: in it, beside it, or in an ancestor
    (``Astronomy/CALI_FRAME`` for ``Astronomy/DWARF_RAW_...``) up to three levels up."""
    d = os.path.abspath(folder)
    for _ in range(4):
        hit = _child(d, "CALI_FRAME")
        if hit:
            return hit
        try:
            for c in os.listdir(d):
                sub = os.path.join(d, c)
                if os.path.isdir(sub) and not c.upper().startswith("DWARF_RAW"):
                    hit = _child(sub, "CALI_FRAME")
                    if hit:
                        return hit
        except OSError:
            pass
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def _candidates(folder: str, extra: list[str]) -> list[tuple[str, str | None]]:
    """(file, kind from its folder name) of every calibration candidate for ``folder``."""
    seen, out = set(), []

    def add(path, hint):
        if path not in seen:
            seen.add(path)
            out.append((path, hint))

    def walk(root, depth):
        for p in _fits_under(root, depth):
            parts = os.path.relpath(p, root).replace("\\", "/").split("/")[:-1]
            hint = next((_DIR_KINDS[x.lower()] for x in reversed(parts) if x.lower() in _DIR_KINDS),
                        _DIR_KINDS.get(os.path.basename(root).lower()))
            add(p, hint)
    for root in extra:
        if os.path.isdir(root):
            walk(root, 3)
    lib = find_dwarf_library(folder)
    if lib:
        walk(lib, 3)
    for base in (folder, os.path.dirname(os.path.abspath(folder))):
        for name in _DIR_KINDS:
            d = _child(base, name)
            if d:
                walk(d, 2)
    # calibration frames saved among the lights
    for p in _fits_under(folder, 0):
        try:
            if _kind(str(fits.getheader(p).get("IMAGETYP", ""))):
                add(p, None)
        except Exception:
            pass
    return out


# ================================================================ masters from individual frames
def _combine(members: list[Master], cache_dir: str) -> Master:
    """Median of individual calibration frames, written once to ``cache_dir/calib``."""
    members = sorted(members, key=lambda m: m.path)[:MAX_COMBINE]
    key = hashlib.sha1("|".join(f"{m.path}:{os.path.getmtime(m.path)}" for m in members).encode()).hexdigest()[:12]
    out_dir = os.path.join(cache_dir, "calib")
    path = os.path.join(out_dir, f"master_{members[0].kind}_{key}.fits")
    m0 = members[0]
    temps = [m.temp for m in members if m.temp is not None]
    res = Master(**{**asdict(m0), "path": path, "stack": len(members), "is_master": True,
                    "temp": float(np.median(temps)) if temps else None})
    if os.path.exists(path):
        return res
    os.makedirs(out_dir, exist_ok=True)
    h, w = m0.shape
    med = np.empty((h, w), np.float32)
    # sections are read strip by strip (scaled uint16 data, BZERO 32768, cannot be memory-mapped)
    hduls = [fits.open(m.path, memmap=False) for m in members]
    try:
        for y0 in range(0, h, 128):
            y1 = min(h, y0 + 128)
            strip = np.stack([np.asarray(hd[0].section[y0:y1, :], np.float32) for hd in hduls])
            med[y0:y1] = np.median(strip, axis=0)
    finally:
        for hd in hduls:
            hd.close()
    hdr = fits.Header()
    for k, v in {"IMAGETYP": m0.kind.capitalize(), "EXPTIME": m0.exptime, "GAIN": m0.gain, "CCD-TEMP": res.temp,
                 "XBINNING": m0.binning, "FILTER": m0.filter or None, "STACKCNT": len(members)}.items():
        if v is not None:
            hdr[k] = v
    fits.writeto(path + ".tmp", med, hdr, overwrite=True, output_verify="silentfix")
    os.replace(path + ".tmp", path)
    return res


def load_library(folder: str, cache_dir: str, extra: list[str] | None = None) -> list[Master]:
    """Every usable calibration master for ``folder`` (individual frames combined)."""
    masters, singles = [], {}
    for path, hint in _candidates(folder, extra or []):
        m = parse_master(path, hint)
        if m is None:
            continue
        if m.is_master:
            masters.append(m)
        else:
            key = (m.kind, m.shape, round(m.exptime or 0, 3), m.gain, m.binning, m.fcode, _nf(m.filter), m.cam)
            singles.setdefault(key, []).append(m)
    for group in singles.values():
        if len(group) == 1:
            masters.append(group[0])
        else:
            try:
                masters.append(_combine(group, cache_dir))
            except Exception as e:
                print(f"calibration: could not combine {len(group)} {group[0].kind} frames: {e}")
    return masters


def _nf(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


# ================================================================ choosing masters
def _compatible(m: Master, shape, cam) -> bool:
    return tuple(m.shape) == tuple(shape) and (m.cam is None or cam is None or m.cam == cam)


def _near(a, b, tol):
    return a is None or b is None or abs(a - b) <= tol


def _nearest_gain(biases: list[Master], gain):
    if not biases:
        return None
    return min(biases, key=lambda m: (abs((m.gain if m.gain is not None else 0) - (gain or 0)), -m.stack))


def select(masters: list[Master], shape, exptime, gain, temp, filt: str, cam: str | None) -> dict:
    """The bias, dark and flat for lights of this size / exposure / gain / temperature / filter."""
    comp = [m for m in masters if _compatible(m, shape, cam)]
    biases = [m for m in comp if m.kind == "bias"]
    darks = [m for m in comp if m.kind == "dark" and _near(m.gain, gain, 0.5)]
    flats = [m for m in comp if m.kind == "flat"]
    same_gain_bias = [b for b in biases if _near(b.gain, gain, 0.5)]
    bias = _nearest_gain(same_gain_bias or biases, gain)

    def tkey(m):
        dt = abs(m.temp - temp) if (m.temp is not None and temp is not None) else 50.0
        return (dt, -m.stack)
    dark, k0 = None, 1.0
    tol = max(0.05, 0.02 * (exptime or 0))
    exact = [d for d in darks if d.exptime is not None and exptime and abs(d.exptime - exptime) <= tol]
    if exact:
        dark = min(exact, key=tkey)
    elif bias is not None and exptime:
        scalable = [d for d in darks if d.exptime]
        if scalable:
            # the nearest exposure (a longer one measures the thermal signal with less noise)
            dark = min(scalable, key=lambda d: (abs(np.log(d.exptime / exptime)), tkey(d)))
            k0 = exptime / dark.exptime

    fcode = filter_code(filt) if filt else None

    def fmatch(f: Master) -> int:
        if f.fcode is not None and fcode is not None:
            return 2 if f.fcode == fcode else -1
        if f.filter and filt:
            return 2 if _nf(f.filter) == _nf(filt) else -1
        return 1                          # the flat or the lights do not name the filter
    flats = [f for f in flats if fmatch(f) >= 0]
    flat = max(flats, key=lambda f: (fmatch(f), f.stack)) if flats else None
    flat_bias = _nearest_gain(biases, flat.gain) if flat is not None else None
    return {"bias": bias, "dark": dark, "flat": flat, "flat_bias": flat_bias, "k0": float(k0)}


# ================================================================ applying
_PREP: dict = {}


def _load(path: str, scale: float = 1.0) -> np.ndarray:
    """A master in 16-bit ADU: ``scale`` as the lights' (FrameInfo.adu_scale; a DWARF 3's masters
    are 12-bit like its subs), a 32-bit master normalised to 0..1 (Siril's default) to 0..65535."""
    data = fits.getdata(path)
    was_float = data.dtype.kind == "f"
    data = np.asarray(data, np.float32)
    if was_float and np.nanmax(data) <= 2.0:
        data = data * WHITE
    elif scale != 1.0:
        data = data * scale
    return np.nan_to_num(data, nan=0.0)


def _same_colour_median(a: np.ndarray) -> np.ndarray:
    """Median of the four nearest same-colour CFA neighbours (+-2 px) of every pixel."""
    p = np.pad(a, 2, mode="reflect")
    h, w = a.shape
    nb = np.stack([p[0:h, 2:w + 2], p[4:h + 4, 2:w + 2], p[2:h + 2, 0:w], p[2:h + 2, 4:w + 4]])
    return np.median(nb, axis=0)


def _neighbours_at(a: np.ndarray, ys, xs) -> np.ndarray:
    h, w = a.shape
    vals = [a[np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)] for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2))]
    return np.median(np.stack(vals), axis=0)


def _prepared(calib: dict) -> dict:
    """Master arrays of one calibration (per process, loaded once)."""
    key = (calib.get("bias"), calib.get("dark"), calib.get("flat"), calib.get("flat_bias"),
           calib.get("adu_scale", 1.0), calib.get("relevel", False))
    hit = _PREP.get(key)
    if hit is not None:
        return hit
    if len(_PREP) > 2:
        _PREP.clear()
    P = {"B": None, "D": None, "T": None, "F": None, "hot": None, "defects": None}
    sc = float(calib.get("adu_scale", 1.0))
    B = _load(calib["bias"], sc) if calib.get("bias") else None
    D = _load(calib["dark"], sc) if calib.get("dark") else None
    shape = (B if B is not None else D).shape if (B is not None or D is not None) else None
    defects = None
    if D is not None and B is not None:
        if calib.get("relevel"):
            # a bias of another gain has another pedestal (a DWARF 3's gain-2 bias sits 2 ADU above
            # its gain-60 dark): level it to the dark's, so the thermal map scaled by k is thermal only
            B = B + float(np.median((D - B)[::3, ::3]))
        T = D - B                       # thermal signal (and amp glow) of the dark
        resid = T - _same_colour_median(T)
        sub = resid[::3, ::3]
        mad = 1.4826 * float(np.median(np.abs(sub - np.median(sub)))) + 1e-3
        thr = max(8 * mad, 20.0)
        hot = np.flatnonzero((resid > thr) & (D < SAT_RAW))
        if len(hot) > 40000:            # a random subset is enough for the scale fit
            hot = np.random.default_rng(0).choice(hot, 40000, replace=False)
        ys, xs = np.unravel_index(hot, T.shape)
        P["hot"] = (ys, xs, resid.ravel()[hot].astype(np.float32))
        P["B"], P["T"] = B, T
        ped = float(np.median(B))
        defects = (D >= SAT_RAW) | (resid > 0.25 * (WHITE - ped))
    elif D is not None:
        P["D"] = D
        ped = float(np.median(D))
        defects = D >= SAT_RAW
    elif B is not None:
        P["B"] = B
        ped = float(np.median(B))
    else:
        ped = float(calib.get("header_bias", 0.0))
    P["pedestal"] = ped
    if calib.get("flat"):
        F = _load(calib["flat"], sc)
        FB = _load(calib["flat_bias"], sc) if calib.get("flat_bias") else B
        note = "as stored"
        if FB is not None and FB.shape == F.shape:
            # a master flat stored without its bias removed still has the pedestal in its darkest pixels
            if np.percentile(F[::4, ::4], 0.1) > 0.5 * float(np.median(FB)):
                F = F - FB
                note = "bias removed"
        F = F.astype(np.float32)
        h, w = F.shape
        cy, cx = h // 4, w // 4
        for dy in (0, 1):
            for dx in (0, 1):
                site = F[dy::2, dx::2]
                centre = site[cy // 2:(h - cy) // 2, cx // 2:(w - cx) // 2]
                site /= max(float(np.median(centre)), 1e-6)
        dead = ~np.isfinite(F) | (F < 0.3)
        F[dead] = 1.0
        np.clip(F, 0.05, 20.0, out=F)
        P["F"] = F
        P["flat_note"] = note
        defects = dead if defects is None else (defects | dead)
        shape = shape or F.shape
    P["defects"] = defects
    P["shape"] = shape
    _PREP[key] = P
    return P


def dark_scale(raw: np.ndarray, calib: dict) -> float:
    """k in light - bias - k (dark - bias): the ratio of the sub's hot-pixel excess over its
    same-colour neighbours to the dark's.  Sky, nebulosity and stars are smooth on the scale
    of a pixel and cancel in the excess; the hot pixels' dark current does not."""
    P = _prepared(calib)
    k0 = float(calib.get("k0", 1.0))
    if P["hot"] is None or len(P["hot"][0]) < 50:
        return k0
    ys, xs, t = P["hot"]
    v = raw[ys, xs]
    ok = v < SAT_RAW
    e = v - _neighbours_at(raw, ys, xs)
    ok &= t > 0
    if ok.sum() < 50:
        return k0
    # least absolute deviations through the origin: the t-weighted median of e / t.  Least squares
    # is ruled by the few hundred hottest pixels, whose dark current grows faster with temperature
    # than the bulk's (DWARF 3, C 20 at 42 C against a 27 C dark: 1.77 by least squares, 1.30 by L1,
    # and only the L1 scale leaves less residual than plain subtraction in every strength band)
    r, w = e[ok] / t[ok], t[ok]
    o = np.argsort(r)
    cw = np.cumsum(w[o])
    k = float(r[o][np.searchsorted(cw, 0.5 * cw[-1])])
    return float(np.clip(k, 0.0, 10.0 * max(k0, 0.1)))


def apply(raw: np.ndarray, calib: dict) -> np.ndarray:
    """Calibrate a raw CFA frame (float32 ADU, modified in place and returned)."""
    P = _prepared(calib)
    if P["shape"] is None or raw.shape != P["shape"]:
        raw -= P["pedestal"]
        return raw
    sat = raw >= SAT_RAW
    if P["T"] is not None:
        k = dark_scale(raw, calib) if calib.get("optimize", True) else float(calib.get("k0", 1.0))
        raw -= P["B"]
        raw -= k * P["T"]
    elif P["D"] is not None:
        raw -= P["D"]
    elif P["B"] is not None:
        raw -= P["B"]
    else:
        raw -= P["pedestal"]
    if P["F"] is not None:
        raw /= P["F"]
    if sat.any():
        raw[sat] = np.maximum(raw[sat], WHITE - P["pedestal"])
    return raw


def defects(calib: dict | None, shape) -> np.ndarray | None:
    """Pixels the masters show to be unusable (dark beyond linear range, dead in the flat)."""
    if not calib:
        return None
    P = _prepared(calib)
    d = P.get("defects")
    return d if d is not None and d.shape == tuple(shape) else None


# ================================================================ attaching to a session
def _describe(m: Master | None) -> dict | None:
    if m is None:
        return None
    return {"file": m.name, "path": m.path, "exptime": m.exptime, "gain": m.gain, "temp": m.temp,
            "filter": m.filter or (DWARF_FILTERS.get(m.fcode) if m.fcode is not None else ""), "stack": m.stack}


DEFAULT_PREFS = {"enabled": True, "library": "", "dark": "auto", "flat": "auto", "bias": "auto"}


def _light_settings(infos: list) -> dict:
    temps = [i.temp for i in infos if i.temp is not None and np.isfinite(i.temp)]
    return {"shape": (infos[0].height, infos[0].width),
            "exptime": float(np.median([i.exptime for i in infos])),
            "gain": float(np.median([i.gain for i in infos])),
            "temp": float(np.median(temps)) if temps else None,
            "temp_range": [float(min(temps)), float(max(temps))] if temps else None,
            "filter": infos[0].filter, "cam": getattr(infos[0], "camera_slot", None)}


def _library(infos: list, folder: str, cache_dir: str, prefs: dict) -> list[Master]:
    setting = os.environ.get("ASTROPHOTO_CALIB", "").strip()
    extra = [p for p in setting.split(os.pathsep) if p] if setting.lower() not in ("", "off", "none", "0", "false") else []
    extra += [p for p in str(prefs.get("library") or "").split(os.pathsep) if p.strip()]
    return load_library(folder, cache_dir, [os.path.expanduser(p.strip()) for p in extra])


def candidates(infos: list, folder: str, cache_dir: str, prefs: dict | None = None) -> dict:
    """The masters that fit these lights (frame size and camera), per kind, for choosing by hand."""
    prefs = {**DEFAULT_PREFS, **(prefs or {})}
    if not infos:
        return {"bias": [], "dark": [], "flat": []}
    ls = _light_settings(infos)
    out = {"bias": [], "dark": [], "flat": []}
    for m in _library(infos, folder, cache_dir, prefs):
        if _compatible(m, ls["shape"], ls["cam"]):
            out[m.kind].append(_describe(m))
    for k in out:
        out[k].sort(key=lambda d: (d["file"]))
    return out


def attach(infos: list, folder: str, cache_dir: str, prefs: dict | None = None, progress=None) -> dict | None:
    """Find and choose the masters for a session's lights and store them on every FrameInfo
    (``info.calib``; ``info.bias`` becomes the masters' pedestal, which the saturation level uses).

    ``prefs`` (the web UI's per-dataset settings, DEFAULT_PREFS): ``enabled``; ``library``, extra
    folders; ``dark`` / ``flat`` / ``bias``: "auto", "none" or the path of a master to use."""
    prefs = {**DEFAULT_PREFS, **(prefs or {})}
    setting = os.environ.get("ASTROPHOTO_CALIB", "").strip()
    for info in infos:
        info.calib = None
    if not infos or not prefs.get("enabled", True) or setting.lower() in ("off", "none", "0", "false"):
        return None
    if progress:
        progress(0, 3, "Calibration: finding bias / dark / flat masters")
    masters = _library(infos, folder, cache_dir, prefs)
    if not masters:
        return None
    i0 = infos[0]
    ls = _light_settings(infos)
    temp, exptime, gain = ls["temp"], ls["exptime"], ls["gain"]
    sel = select(masters, ls["shape"], exptime, gain, temp, i0.filter, ls["cam"])
    chosen_by_hand = []
    for kind in ("bias", "dark", "flat"):
        want = prefs.get(kind, "auto") or "auto"
        if want == "none":
            sel[kind] = None
            chosen_by_hand.append(f"{kind}: none")
        elif want != "auto":
            hit = next((m for m in masters if m.path == want and m.kind == kind
                        and _compatible(m, ls["shape"], ls["cam"])), None)
            if hit is not None:
                sel[kind] = hit
                chosen_by_hand.append(f"{kind}: {hit.name}")
                if kind == "dark":
                    sel["k0"] = exptime / hit.exptime if (hit.exptime and exptime) else 1.0
    if sel["flat"] is not None and (prefs.get("bias") not in (None, "auto")):
        sel["flat_bias"] = sel["bias"] or sel["flat_bias"]
    elif sel["flat"] is None:
        sel["flat_bias"] = None
    if not any(sel[k] for k in ("bias", "dark", "flat")):
        return None
    calib = {k: (sel[k].path if sel[k] else None) for k in ("bias", "dark", "flat", "flat_bias")}
    b, d = sel["bias"], sel["dark"]
    calib.update({"k0": sel["k0"], "optimize": b is not None and d is not None,
                  "header_bias": float(i0.bias), "adu_scale": float(getattr(i0, "adu_scale", 1.0)),
                  "relevel": bool(b is not None and d is not None and b.gain is not None and d.gain is not None
                                  and abs(b.gain - d.gain) > 0.5)})
    if progress:
        progress(1, 3, "Calibration: fitting the dark's thermal scale and checking the flat")
    P = _prepared(calib)
    # a sample of the subs (the telescope's accepted ones first: the others are often cloudy)
    pool = [i for i in infos if not getattr(i, "device_rejected", False)]
    pool = pool if len(pool) >= FIT_SAMPLE else infos
    sample = [pool[i] for i in np.linspace(0, len(pool) - 1, min(FIT_SAMPLE, len(pool))).round().astype(int)]
    from .frames import load_raw_adu
    ks, cal = [], []
    for info in sample:
        raw = load_raw_adu(info.path, calib["adu_scale"])
        if calib["optimize"]:
            ks.append(dark_scale(raw, calib))
        if P["F"] is not None:
            cal.append(apply(raw, calib))
    flat_check = None
    if cal:
        after = np.median(np.stack(cal), axis=0)
        del cal
        before = after * P["F"]
        flat_check = {"before": round(sky_unevenness(before), 2), "after": round(sky_unevenness(after), 2)}
    report = {"bias": _describe(sel["bias"]), "dark": _describe(sel["dark"]), "flat": _describe(sel["flat"]),
              "flat_bias": _describe(sel["flat_bias"]), "pedestal": round(P["pedestal"], 1),
              "light_temp": temp, "dark_scale": [round(min(ks), 3), round(max(ks), 3)] if ks else None,
              "flat_note": P.get("flat_note"), "flat_check": flat_check, "notes": [],
              "lights": {**{k: v for k, v in ls.items() if k != "shape"}, "n": len(infos),
                         "width": i0.width, "height": i0.height},
              "chosen_by_hand": chosen_by_hand}
    if d is None:
        report["notes"].append(f"no dark for {exptime:g} s at gain {gain:g}: hot pixels come from the "
                               "temporal median only")
    elif d.temp is not None and temp is not None and abs(d.temp - temp) > 5 and not calib["optimize"]:
        report["notes"].append(f"dark taken at {d.temp:g} C, lights at {temp:.0f} C and no bias to "
                               "scale its thermal signal")
    if calib["relevel"]:
        report["notes"].append(f"bias taken at gain {b.gain:g}, dark at {d.gain:g}: bias levelled to the dark's "
                               "pedestal")
    if flat_check and prefs.get("flat", "auto") == "auto" and flat_check["after"] > 1.1 * flat_check["before"] + 0.5:
        # the flat does not describe this optical train (another filter position, dust that moved, ...)
        report["notes"].append(f"the flat made the sky less even ({flat_check['before']:g} -> "
                               f"{flat_check['after']:g} ADU): not used")
        calib["flat"] = calib["flat_bias"] = None
        report["flat"] = report["flat_bias"] = None
        P = _prepared(calib)
    if sel["flat"] is None:
        report["notes"].append(f"no flat for filter {i0.filter or '(unnamed)'}: vignetting is left to "
                               "gradient removal")
    calib["report"] = report
    calib["summary"] = summary(report)
    if progress:
        progress(3, 3, "Calibration: " + calib["summary"])
    for info in infos:
        info.calib = calib
        info.bias = P["pedestal"]
    return calib


def sky_unevenness(img: np.ndarray, block: int = 24) -> float:
    """Large-scale unevenness of a calibrated sky (ADU): robust scatter of star-free block medians in
    each CFA site after removing a plane (clouds and light pollution make gradients; vignetting and
    dust rings are what a flat corrects, and they are not planar)."""
    vals = []
    for dy in (0, 1):
        for dx in (0, 1):
            s = img[dy::2, dx::2]
            h, w = (s.shape[0] // block) * block, (s.shape[1] // block) * block
            g = np.median(s[:h, :w].reshape(h // block, block, w // block, block).transpose(0, 2, 1, 3)
                          .reshape(h // block, w // block, -1), axis=2)
            yy, xx = np.mgrid[0:g.shape[0], 0:g.shape[1]]
            A = np.c_[np.ones(g.size), yy.ravel(), xx.ravel()]
            r = g.ravel() - A @ np.linalg.lstsq(A, g.ravel(), rcond=None)[0]
            vals.append(1.4826 * float(np.median(np.abs(r - np.median(r)))))
    return float(np.median(vals))


def summary(rep: dict) -> str:
    parts = []
    if rep.get("dark"):
        d = rep["dark"]
        s = f"dark {d['exptime']:g} s" if d.get("exptime") else "dark"
        if d.get("temp") is not None:
            s += f" @ {d['temp']:g} C"
        if rep.get("dark_scale"):
            lo, hi = rep["dark_scale"]
            s += f" (thermal x{lo:.2f}" + (f"-{hi:.2f})" if hi - lo > 0.005 else ")")
        parts.append(s)
    if rep.get("bias"):
        parts.append("bias")
    if rep.get("flat"):
        f = rep["flat"]
        parts.append(f"flat{' ' + f['filter'] if f.get('filter') else ''}")
    return " + ".join(parts) if parts else "none"
