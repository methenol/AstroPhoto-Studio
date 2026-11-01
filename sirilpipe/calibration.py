"""Find and choose calibration masters (dark, flat, bias) for a dataset.

This module only *decides*; Siril applies them (``workflow.py`` writes the commands).

Where masters come from, in this order
    * a DWARF ``CALI_FRAME`` library (in / beside the session folder, or an ancestor up to three
      levels): ``bias/cam_0/bias_gain_2_bin_1.fits``,
      ``dark/cam_0/dark_exp_15.000000_gain_60_bin_1_38C_stack_20.fits``,
      ``flat/cam_0/flat_gain_2_bin_1_ir_2.fits``; these are already-stacked masters;
    * ``darks/ flats/ bias/`` (or biases, offsets) folders in or beside the light folders, and
      ``master_*.fit`` files there (the Siril layout); several raw frames are stacked by Siril
      into ``<session>/calib/master_<kind>_<hash>.fit``;
    * ``ASTROPHOTO_CALIB`` library folders (and the UI's per-dataset ``library`` preference).

How one is chosen
    dark   same size, gain (+-0.5) and exposure (+-max(0.05 s, 2 %)); nearest sensor temperature
           (the delta is logged; > 5 C is flagged); then the larger stack.  A dark of another
           exposure is never scaled: Siril only gets a dark that really matches.
    flat   same size and, when both sides name a filter, the same filter.  DWARF flats carry only
           the vendor's ``ir_N`` code: DWARFLAB documents 0 = VIS, 1 = Astro, 2 = Duo-Band, and a
           flat is used only when the lights' FILTER header is exactly one of those three names
           (nothing is inferred from "LP" or free text).  The mapping itself is vendor
           documentation, not something we measured, so it is recorded as the reason in
           ``calibration.json`` and the user can switch the flat off (``flat: "none"``) or pick a
           file by hand.
    bias   used only to calibrate the flat (never on lights together with a dark: it would be
           subtracted twice, Naztronomy issue, workflows.md 11.1).  Same size, nearest gain.
    Seestar: no calibration frames exist; lights-only.
"""
from __future__ import annotations

import glob
import hashlib
import os
import re
from dataclasses import asdict, dataclass

from astropy.io import fits

from . import config
from .dataset import FITS_EXT, FrameSet, _norm

MAX_FILES = 3000
MAX_RAW = 200                  # raw calibration frames stacked into one master
DEFAULT_PREFS = {"enabled": True, "library": "", "dark": "auto", "flat": "auto", "bias": "auto"}
_DIR_KINDS = {"bias": "bias", "biases": "bias", "offset": "bias", "offsets": "bias",
              "dark": "dark", "darks": "dark", "flat": "flat", "flats": "flat"}
# DWARFLAB's flat codes (help.dwarflab.com "How to View and Obtain the Files on DWARF 3")
DWARF_IR_NAMES = {0: "VIS", 1: "Astro", 2: "Duo-Band"}


@dataclass
class Master:
    """A calibration file (a master, or one raw frame of a set)."""
    path: str
    kind: str                       # bias | dark | flat
    shape: tuple                    # (height, width)
    exptime: float | None = None
    gain: float | None = None
    temp: float | None = None
    binning: int | None = None
    filter: str = ""
    ir: int | None = None           # DWARF flat code
    cam: str | None = None          # DWARF camera slot
    stack: int = 1
    is_master: bool = False
    instrument: str = ""            # INSTRUME of the calibration frame (camera model), when written

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


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
    """A calibration frame's facts: FITS header first, then the (DWARF) file-name tokens."""
    try:
        h = fits.getheader(path)
    except Exception:
        return None
    if int(h.get("NAXIS", 0) or 0) != 2:
        return None                          # a debayered colour master cannot calibrate CFA data
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
    m = Master(path=os.path.abspath(path), kind=kind, shape=(int(h["NAXIS2"]), int(h["NAXIS1"])))
    m.exptime = hv("EXPTIME", "EXPOSURE")
    if m.exptime is None:
        m.exptime = _num(r"exp[_\-]([0-9.]+)", name)
    m.gain = hv("GAIN")
    if m.gain is None:
        m.gain = _num(r"gain[_\-]([0-9.]+)", name)
    m.temp = hv("CCD-TEMP", "SET-TEMP", "TEMPERAT", "DET-TEMP")
    if m.temp is None:
        m.temp = _num(r"(?:^|_)([-+]?[0-9]+(?:\.[0-9]+)?)C(?:_|\.|\(|$)", name)
    b = hv("XBINNING")
    m.binning = int(b) if b else _num(r"bin[_\-]([0-9]+)", name, int)
    st = hv("STACKCNT", "NCOMBINE")
    st = int(st) if st else _num(r"stack[_\-]([0-9]+)", name, int)
    m.stack = st or 1
    m.is_master = m.stack > 1 or "master" in name.lower() or "/CALI_FRAME/" in (rel.upper() + "/")
    m.filter = str(h.get("FILTER", "") or "").strip()
    m.instrument = str(h.get("INSTRUME", "") or "").strip()
    m.ir = _num(r"(?:^|_)ir[_\-]?([0-9])(?:_|\.|$)", name, int)
    cam = re.search(r"(?:^|/)(cam_[0-9])(?:/|$)", rel, re.IGNORECASE)
    m.cam = cam.group(1).lower() if cam else None
    return m


# ------------------------------------------------------------------ discovery
def _fits_under(root: str, depth: int) -> list[str]:
    out: list[str] = []
    for d in range(depth + 1):
        for ext in FITS_EXT + tuple(e.upper() for e in FITS_EXT):
            out.extend(glob.glob(os.path.join(glob.escape(root), *(["*"] * d), f"*{ext}")))
        if len(out) > MAX_FILES:
            break
    return sorted(set(out))[:MAX_FILES]


def _child(parent: str, name: str) -> str | None:
    """A child folder of ``parent`` named ``name`` in any letter case."""
    try:
        for c in os.listdir(parent):
            if c.lower() == name.lower() and os.path.isdir(os.path.join(parent, c)):
                return os.path.join(parent, c)
    except OSError:
        pass
    return None


def find_dwarf_library(folder: str) -> str | None:
    """The DWARF CALI_FRAME for a session folder: in it, beside it, or in an ancestor (up to 3 up)."""
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


def candidates(fs: FrameSet, prefs: dict | None = None) -> list[Master]:
    """Every calibration file that might apply to this dataset, parsed."""
    prefs = {**DEFAULT_PREFS, **(prefs or {})}
    seen: set[str] = set()
    out: list[Master] = []

    def add(path: str, hint: str | None):
        path = os.path.abspath(path)
        if path in seen:
            return
        seen.add(path)
        m = parse_master(path, hint)
        if m:
            out.append(m)

    def walk(root: str, depth: int):
        for p in _fits_under(root, depth):
            parts = os.path.relpath(p, root).replace("\\", "/").split("/")[:-1]
            hint = next((_DIR_KINDS[x.lower()] for x in reversed(parts) if x.lower() in _DIR_KINDS),
                        _DIR_KINDS.get(os.path.basename(root).lower()))
            add(p, hint)

    if not config.library_disabled():
        libs = config.library_folders() + ([prefs["library"]] if prefs.get("library") else [])
        for root in libs:
            if os.path.isdir(root):
                walk(root, 3)
    walked: set[str] = set()
    for f in fs.folders:
        # CALI_FRAME is a DWARF library: a different rig's lights must not pick it up just because
        # a sensor of the same size sits beside it (a 3840x2160 ASI585 next to a DWARF's folder).
        lib = find_dwarf_library(f) if fs.kind == "dwarf" else None
        if lib and lib not in walked:
            walked.add(lib)
            walk(lib, 3)
        for base in (f, os.path.dirname(os.path.abspath(f))):
            for name in _DIR_KINDS:
                d = _child(base, name)
                if d and d not in walked:
                    walked.add(d)
                    walk(d, 2)
            # Siril layout: master_*.fit beside lights/ (masters/ folder too)
            md = _child(base, "masters")
            if md and md not in walked:
                walked.add(md)
                walk(md, 1)
            for p in _fits_under(base, 0):
                if re.match(r"master", os.path.basename(p), re.I):
                    add(p, None)
    for p in fs.calibration_files:
        add(p, None)
    return out


# ------------------------------------------------------------------ choosing
def _group_key(m: Master) -> tuple:
    return (m.kind, m.shape, round(m.exptime or 0.0, 3), m.gain, m.binning, m.ir, _norm(m.filter), m.cam)


def _near(a, b, tol) -> bool:
    return a is None or b is None or abs(a - b) <= tol


def _as_choice(m: Master | list[Master], how: str, temp_delta: float | None = None) -> dict:
    """One selection: a ready master, or the raw frames Siril must stack."""
    members = m if isinstance(m, list) else [m]
    m0 = members[0]
    raw = not m0.is_master and len(members) > 1
    d = {"file": m0.name if not raw else f"{len(members)} x {m0.name}", "path": m0.path,
         "kind": m0.kind, "exptime": m0.exptime, "gain": m0.gain, "temp": m0.temp, "filter": m0.filter or None,
         "stack": sum(x.stack for x in members), "how": how, "temp_delta": temp_delta,
         "raw_frames": [x.path for x in sorted(members, key=lambda x: x.path)[:MAX_RAW]] if raw else None}
    if raw:
        key = hashlib.sha1("|".join(f"{x.path}:{os.path.getmtime(x.path)}" for x in members[:MAX_RAW]).encode()).hexdigest()[:10]
        d["master_name"] = f"master_{m0.kind}_{key}.fit"
    return d


def _pick_by_group(cands: list[Master], keyfn):
    """Group raw frames of one setting together; masters stay single."""
    groups: dict[tuple, list[Master]] = {}
    for m in cands:
        if m.is_master:
            groups.setdefault(("master", m.path), []).append(m)
        else:
            groups.setdefault(_group_key(m), []).append(m)
    return list(groups.values())


def choose(fs: FrameSet, masters: list[Master], prefs: dict | None = None) -> dict:
    """Pick dark / flat / bias for the dataset.  Returns the report dict (also ``calibration.json``)."""
    prefs = {**DEFAULT_PREFS, **(prefs or {})}
    f0 = fs.first
    shape = (f0.height, f0.width)
    notes: list[str] = []
    report: dict = {"enabled": bool(prefs["enabled"]), "dark": None, "flat": None, "bias": None,
                    "flat_bias": None, "notes": notes, "chosen_by_hand": [],
                    "lights": {"exptime": f0.exptime, "gain": f0.gain, "filter": f0.filter,
                               "temp_range": _temp_range(fs)}, "summary": None}
    if fs.kind == "seestar":
        notes.append("Seestar: the telescope provides no calibration frames; lights-only processing.")
        report["summary"] = "No calibration frames (Seestar lights-only recipe)."
        if not prefs["enabled"] or not masters:
            return report
    if not prefs["enabled"]:
        report["summary"] = "Calibration switched off."
        return report
    cam = f0.camera_slot
    comp = [m for m in masters if tuple(m.shape) == shape and (m.cam is None or cam is None or m.cam == cam)
            and (not m.instrument or not f0.instrument or _norm(m.instrument) == _norm(f0.instrument))
            and (fs.kind == "dwarf" or "/CALI_FRAME/" not in m.path.upper().replace("\\", "/") + "/")]
    temps = [f.temp for f in fs.frames if f.temp is not None]
    temp = sum(temps) / len(temps) if temps else None

    def by_hand(kind: str) -> Master | None:
        v = prefs.get(kind, "auto")
        if v in ("auto", "", None):
            return None
        if v == "none":
            report["chosen_by_hand"].append(kind)
            return False                                     # type: ignore[return-value]
        m = parse_master(v, kind)
        if m is None or tuple(m.shape) != shape:
            notes.append(f"{kind}: the chosen file {os.path.basename(str(v))} does not match the lights' size; ignored.")
            return None
        m.is_master = True
        report["chosen_by_hand"].append(kind)
        return m

    # ---- bias
    bias = by_hand("bias")
    if bias is None:
        pool = [m for m in comp if m.kind == "bias"]
        if pool:
            gains = [m for m in pool if _near(m.gain, f0.gain, 0.5)] or pool
            grp = _pick_by_group(gains, None)
            best = min(grp, key=lambda g: (abs((g[0].gain or 0) - (f0.gain or 0)), -sum(x.stack for x in g)))
            bias = best
            report["bias"] = _as_choice(bias, "bias nearest to the lights' gain")
    elif bias is not False:
        report["bias"] = _as_choice(bias, "chosen by hand")

    # ---- dark
    dark = by_hand("dark")
    if dark is None:
        tol = max(0.05, 0.02 * f0.exptime)
        pool = [m for m in comp if m.kind == "dark" and _near(m.gain, f0.gain, 0.5)
                and m.exptime is not None and abs(m.exptime - f0.exptime) <= tol
                and (m.binning is None or f0.binning is None or m.binning == f0.binning)]
        if pool:
            def tkey(g):
                t = g[0].temp
                dt = abs(t - temp) if (t is not None and temp is not None) else 99.0
                return (dt, -sum(x.stack for x in g))
            best = min(_pick_by_group(pool, None), key=tkey)
            dt = (abs(best[0].temp - temp) if (best[0].temp is not None and temp is not None) else None)
            how = f"exposure {best[0].exptime:g} s and gain {best[0].gain:g} match; nearest temperature"
            if dt is not None:
                how += f" ({best[0].temp:g} C vs lights {temp:.1f} C, delta {dt:.1f} C)"
                if dt > 5.0:
                    notes.append(f"dark temperature differs by {dt:.1f} C from the lights (> 5 C); "
                                 "hot-pixel correction (-cc=dark) copes, thermal signal is not scaled.")
            report["dark"] = _as_choice(best, how, dt)
            dark = best
        elif any(m.kind == "dark" for m in comp):
            notes.append(f"darks exist but none at exposure {f0.exptime:g} s / gain {f0.gain:g}; "
                         "no dark used (darks are never scaled).")
    elif dark is not False:
        report["dark"] = _as_choice(dark, "chosen by hand")

    # ---- flat
    flat = by_hand("flat")
    if flat is None:
        pool = [m for m in comp if m.kind == "flat"]
        chosen, how = None, ""
        if pool and fs.kind == "dwarf":
            ir_map = {_norm(v): k for k, v in DWARF_IR_NAMES.items()}
            code = ir_map.get(_norm(f0.filter))
            if code is None:
                notes.append(f"flat not used: the lights' filter '{f0.filter}' is not one of DWARFLAB's "
                             "VIS / Astro / Duo-Band, so its flat cannot be told unambiguously.")
            else:
                hits = [m for m in pool if m.ir == code]
                if hits:
                    chosen = _pick_by_group(hits, None)[0]
                    how = (f"DWARFLAB flat code ir_{code} = {DWARF_IR_NAMES[code]} equals the lights' FILTER "
                           "(vendor documentation; not verified on sky)")
        elif pool:
            fl = [m for m in pool if not m.filter or not f0.filter or _norm(m.filter) == _norm(f0.filter)]
            if fl:
                grp = _pick_by_group(fl, None)
                chosen = max(grp, key=lambda g: sum(x.stack for x in g))
                how = "flat matches the lights' size" + (" and filter" if f0.filter and chosen[0].filter else "")
        if chosen:
            report["flat"] = _as_choice(chosen, how)
            flat = chosen
    elif flat is not False:
        report["flat"] = _as_choice(flat, "chosen by hand")
    report["flat_bias"] = report["bias"] if (report["flat"] and report["bias"]) else None
    if report["flat"] and not report["bias"]:
        notes.append("flat used without a bias master: its pedestal stays in (flat is normalised by its mean).")
    report["summary"] = _summary(report, fs)
    return report


def _temp_range(fs: FrameSet):
    t = [f.temp for f in fs.frames if f.temp is not None]
    return [min(t), max(t)] if t else None


def _summary(rep: dict, fs: FrameSet) -> str:
    bits = []
    for k in ("dark", "flat", "bias"):
        c = rep.get(k)
        if c:
            bits.append(f"{k}: {c['file']}" + (f" ({c['stack']} frames)" if c.get("raw_frames") else ""))
    if not bits:
        return "No calibration masters matched; lights-only processing."
    return "Calibration: " + "; ".join(bits)


def find_and_choose(fs: FrameSet, prefs: dict | None = None) -> tuple[dict, list[Master]]:
    """Discover, then choose.  Returns (report, all candidates)."""
    prefs = {**DEFAULT_PREFS, **(prefs or {})}
    try:
        masters = candidates(fs, prefs) if prefs["enabled"] else []
    except Exception as e:                      # a broken library must not stop the pipeline
        masters = []
        rep = choose(fs, [], prefs)
        rep["notes"].append(f"calibration library unreadable: {type(e).__name__}: {e}")
        return rep, masters
    return choose(fs, masters, prefs), masters


def candidate_listing(masters: list[Master]) -> dict:
    """What the calibration panel lists under 'candidates' (all, by kind)."""
    out: dict[str, list] = {"bias": [], "dark": [], "flat": []}
    seen = set()
    for m in masters:
        k = _group_key(m) if not m.is_master else ("m", m.path)
        if k in seen:
            continue
        seen.add(k)
        d = asdict(m)
        d["file"] = m.name
        out[m.kind].append(d)
    return out
