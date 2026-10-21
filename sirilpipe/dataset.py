"""Find, read and classify the light frames of a dataset (one or several night folders).

Only FITS headers are read here (fast, no pixels).  Rules, all from the research notes:

* ``failed_*`` (the DWARF live stack's rejects, usually a passing cloud) are read like every other sub and
  marked ``device_rejected``: the frame analysis grades them (README), they are not dropped unseen;
* skipped by name: thumbnails (``*_thn.*``,
  ``thumbnail``), ``stacked*`` / ``Stacked_*`` (the telescope's live stack), ``master*``,
  DWARF ``img_*`` previews, our own ``pp_``/``r_`` products; jpg/png never match the FITS globs;
* skipped by header: ``RESTACK`` != 0 (DWARF restacked products), non-light ``IMAGETYP``,
  debayered cubes (NAXIS != 2) and subs whose size differs from the dataset;
* grouped by (OBJECT, FILTER, EXPTIME, GAIN, size, BAYERPAT); the group with the most total
  exposure is *the dataset*, the other exposure groups of the same object/filter/size are kept
  for the HDR blend (DWARF 15 s + 60 s of M 42) when they hold at least ``MIN_STACK_SUBS`` subs;
* a folder named ``rejected`` (``lights/rejected``, the device-rejected set) is ignored when there
  are other folders.
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime

from astropy.io import fits

FITS_EXT = (".fit", ".fits", ".fts")
MIN_STACK_SUBS = 3          # the fewest subs a stack (or an HDR exposure group) is made of (session.stack_and_finish)

# Names that are never lights.  ``stack`` is matched as a prefix only: a sub of "Stacking test"
# would otherwise vanish (the old pipeline matched it anywhere).
_SKIP_NAME = re.compile(r"^(failed[_\-]|stack|master|img_|pp_|r_pp_|bkg_)|_thn\b|thumbnail", re.IGNORECASE)
_CALIB_NAME = re.compile(r"^(dark|flat|bias|offset)", re.IGNORECASE)
_DWARF_SUB_RE = re.compile(
    r"^(?P<failed>failed[_\-])?(?P<target>.+?)_(?P<exp>[0-9.]+)s(?P<gain>[0-9.]+)_(?P<filter>[^_]+)_"
    r"(?P<date>\d{8})-(?P<time>\d{6,9})_(?P<temp>[-+]?[0-9.]+)C\.(?:fits?|fts)$", re.IGNORECASE)

# Dual-band (Ha + OIII) filters: after upper-casing and dropping non-alphanumerics.  "LP" is the
# Seestar's own dual-band-ish filter (DESIGN section 4).
DUALBAND_FILTERS = {"LP", "DUO", "DUAL", "DUOBAND", "DUALBAND", "LENHANCE", "LEXTREME", "LULTIMATE", "HOO",
                    "ALPT", "NBZ", "NBZLITE", "LENHANCEMONO", "DNB", "DUONARROWBAND", "OPTOLONGLEXTREME",
                    "OPTOLONGLENHANCE", "ASKARCOLOURMAGIC", "STC", "STCDUO", "STCASTROMULTISPECTRA"}

# SPCC "oscsensor" names for the sensors the rigs use (Siril's siril-spcc-database; siril.md 2.5).
_ASI_SENSOR = {"585": "Sony IMX585", "678": "Sony IMX678", "662": "Sony IMX662", "715": "Sony IMX715",
               "533": "Sony IMX533", "294": "Sony IMX294", "183": "Sony IMX183", "178": "Sony IMX178",
               "224": "Sony IMX224", "385": "Sony IMX385", "482": "Sony IMX482", "462": "Sony IMX462",
               "2600": "Sony IMX571", "2400": "Sony IMX410", "6200": "Sony IMX455", "676": "Sony IMX676"}


def is_dualband(filter_name: str) -> bool:
    """A dual-band (Ha + OIII) filter: Seestar LP, DWARF Duo-Band, L-eNhance / L-eXtreme, ..."""
    return re.sub(r"[^A-Z0-9]", "", (filter_name or "").upper()) in DUALBAND_FILTERS


@dataclass
class Frame:
    """One light sub (header facts only)."""
    path: str
    name: str                      # identity used by overrides / thumbnails (unique inside a dataset)
    folder: str
    object: str
    filter: str
    exptime: float
    gain: float
    temp: float | None
    date_obs: str
    timestamp: float
    bayer: str
    width: int
    height: int
    binning: int = 1
    ra: float | None = None        # degrees
    dec: float | None = None
    focallen: float | None = None  # mm
    pixsize: float | None = None   # um (effective, binning included)
    instrument: str = ""
    telescope: str = ""
    kind: str = "generic"          # dwarf | seestar | generic
    camera_slot: str | None = None  # DWARF calibration camera: cam_0 tele / cam_1 wide
    device_rejected: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def group_key(self) -> tuple:
        return (_norm(self.object), _norm(self.filter), round(self.exptime, 2), round(self.gain, 1),
                self.width, self.height, self.bayer.upper())


@dataclass
class ExposureGroup:
    """Frames sharing (object, filter, exposure, gain, size, bayer)."""
    exptime: float
    gain: float
    frames: list[Frame]
    key: tuple = ()

    @property
    def total_exposure(self) -> float:
        return sum(f.exptime for f in self.frames)

    def summary(self) -> dict:
        return {"exptime": self.exptime, "gain": self.gain, "n": len(self.frames),
                "total_min": round(self.total_exposure / 60.0, 1)}


@dataclass
class FrameSet:
    """The classified dataset."""
    folders: list[str]
    frames: list[Frame]                         # the dominant group, sorted by time
    exposure_groups: list[ExposureGroup] = field(default_factory=list)   # HDR candidates (other exposures)
    skipped: dict = field(default_factory=dict)  # reason -> count
    other_groups: list[dict] = field(default_factory=list)  # groups left out (other object/filter/size)
    kind: str = "generic"
    sensor: str = ""                            # SPCC oscsensor name ("" unknown)
    shots_info: dict = field(default_factory=dict)
    calibration_files: list[str] = field(default_factory=list)   # dark/flat/bias frames found among the lights

    def __bool__(self) -> bool:
        return bool(self.frames)

    # ---- dataset-level facts, taken from the first (earliest) sub
    @property
    def first(self) -> Frame:
        return self.frames[0]

    @property
    def object(self) -> str:
        return self.first.object

    @property
    def filter(self) -> str:
        return self.first.filter

    @property
    def exptime(self) -> float:
        return self.first.exptime

    @property
    def gain(self) -> float:
        return self.first.gain

    @property
    def bayer(self) -> str:
        return self.first.bayer

    @property
    def size(self) -> tuple[int, int]:
        return (self.first.width, self.first.height)

    @property
    def is_dualband(self) -> bool:
        return is_dualband(self.filter)

    @property
    def total_exposure(self) -> float:
        return sum(f.exptime for f in self.frames)

    @property
    def pixel_scale(self) -> float | None:
        """Arcsec per pixel from XPIXSZ / FOCALLEN (None when either is unknown)."""
        f = self.first
        if f.pixsize and f.focallen:
            return 206.265 * f.pixsize / f.focallen
        return None

    @property
    def nights(self) -> list[tuple[str, list[Frame]]]:
        """(folder, frames) per folder with frames, folder order."""
        out = []
        for d in self.folders:
            fr = [f for f in self.frames if f.folder == d]
            if fr:
                out.append((d, fr))
        return out

    @property
    def date(self) -> str | None:
        d = self.first.date_obs
        return d[:10] if d else None

    def hdr_groups(self) -> list[ExposureGroup]:
        """All exposure groups to stack for an HDR blend, shortest exposure first (dataset included);
        only when another group has a clearly different exposure (ratio >= 2)."""
        if not self.exposure_groups:
            return []
        main = ExposureGroup(self.exptime, self.gain, self.frames)
        groups = [main] + list(self.exposure_groups)
        return sorted(groups, key=lambda g: g.exptime)


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def _parse_time(s: str) -> float:
    try:
        return datetime.fromisoformat(s.replace("Z", "")).timestamp()
    except Exception:
        return 0.0


def _num(h, *keys, default=None):
    for k in keys:
        v = h.get(k)
        if v not in (None, ""):
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return default


def _coord(v, hours: bool) -> float | None:
    """Degrees from a number or a sexagesimal string (hours or degrees)."""
    if v in (None, ""):
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


def shots_info(folder: str) -> dict:
    """The DWARF session's ``shotsInfo.json`` (RA in hours there!), {} when absent."""
    p = os.path.join(folder, "shotsInfo.json")
    try:
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except (OSError, ValueError):
        return {}


def classify_kind(h, name: str, folder: str) -> str:
    """dwarf | seestar | generic from the header text, then (for a DWARF) the file/folder names."""
    text = _norm(" ".join(str(h.get(k, "") or "") for k in ("TELESCOP", "INSTRUME", "ORIGIN", "CREATOR", "SWCREATE")))
    if "seestar" in text or re.search(r"^s(30|50)(pro)?_\d+", _norm(h.get("TELESCOP", ""))):
        return "seestar"
    if "dwarf" in text:
        return "dwarf"
    if (_DWARF_SUB_RE.match(name) or os.path.basename(folder).upper().startswith("DWARF_")
            or os.path.exists(os.path.join(folder, "shotsInfo.json"))):
        return "dwarf"
    return "generic"


# Fallbacks when a smart telescope's header omits optics (header values always win).
_PROFILE = {("dwarf", "tele"): (150.0, 2.0, "RGGB"), ("seestar", "s30"): (150.0, 2.9, "GRBG"),
            ("seestar", "s50"): (250.0, 2.9, "GRBG")}


def read_frame(path: str, folder: str | None = None) -> Frame | None:
    """Header of one file -> Frame, or None when it is not a usable light (reason via ``skip_reason``)."""
    r = _read_frame(path, folder)
    return r if isinstance(r, Frame) else None


def _read_frame(path: str, folder: str | None = None):
    name = os.path.basename(path)
    folder = folder or os.path.dirname(os.path.abspath(path))
    if _CALIB_NAME.search(name):
        return "calibration frame"
    # ``failed_``: a sub the DWARF's live stack rejected (usually a passing cloud).  It is read and graded like
    # every other sub (README; the frames table notes it), not dropped unseen: on a cloudy night the live stack
    # fails most subs, and dropping them left datasets of one sub (SNR G180.0-01.7, Sh2-132's 60 s night)
    rej = re.match(r"failed[_\-]", name, re.I)
    if _SKIP_NAME.search(name[rej.end():] if rej else name):
        return "telescope/derived product"
    try:
        h = fits.getheader(path)
    except Exception:
        return "unreadable"
    if int(h.get("NAXIS", 0) or 0) != 2:
        return "not a 2-D mosaic"
    typ = str(h.get("IMAGETYP", "") or h.get("FRAME", "") or "").strip().lower()
    if typ and typ not in ("light", "light frame", "lightframe"):
        return "calibration frame"
    if int(_num(h, "RESTACK", default=0) or 0) != 0:
        return "restacked"
    kind = classify_kind(h, name, folder)
    dn = _DWARF_SUB_RE.match(name) if kind == "dwarf" else None
    si = shots_info(folder) if kind == "dwarf" else {}
    width, height = int(h["NAXIS1"]), int(h["NAXIS2"])
    date_obs = str(h.get("DATE-OBS", "") or "").strip()
    if not date_obs and dn:
        d, t = dn.group("date"), dn.group("time")
        date_obs = f"{d[:4]}-{d[4:6]}-{d[6:]}T{t[:2]}:{t[2:4]}:{t[4:6]}"
    filt = str(h.get("FILTER", "") or "").strip() or (dn.group("filter") if dn else "") or str(si.get("ir", "") or "")
    telescop = str(h.get("TELESCOP", "") or "").strip()
    prof = None
    if kind == "dwarf":
        prof = _PROFILE[("dwarf", "tele")]
    elif kind == "seestar":
        prof = _PROFILE[("seestar", "s30" if "s30" in _norm(h.get("CREATOR", "") or telescop) else "s50")]
    binning = int(_num(h, "XBINNING", "CCDXBIN", default=0) or 0) or 1
    pixsize = _num(h, "XPIXSZ", "PIXSIZE1")
    if pixsize is None and prof:
        pixsize = prof[1] * binning
    ra = _coord(h.get("RA"), hours=isinstance(h.get("RA"), str))
    if ra is None:
        ra = _coord(h.get("OBJCTRA"), hours=True)
    if ra is None and si.get("RA") is not None:
        ra = _coord(si.get("RA"), hours=True)            # shotsInfo: hours
    dec = _coord(h.get("DEC"), hours=False)
    if dec is None:
        dec = _coord(h.get("OBJCTDEC"), hours=False)
    if dec is None and si.get("DEC") is not None:
        dec = _coord(si.get("DEC"), hours=False)
    temp = _num(h, "CCD-TEMP", "DET-TEMP", "SENSTEMP", "TEMPERAT")
    if temp is None and dn:
        temp = float(dn.group("temp"))
    return Frame(
        path=os.path.abspath(path), name=name, folder=folder,
        object=str(h.get("OBJECT", "") or "").strip() or (dn.group("target").strip() if dn else "") or str(si.get("target", "") or ""),
        filter=filt,
        exptime=float(_num(h, "EXPTIME", "EXPOSURE", default=(float(dn.group("exp")) if dn else 0.0)) or 0.0),
        gain=float(_num(h, "GAIN", default=(float(dn.group("gain")) if dn else 0.0)) or 0.0),
        temp=temp, date_obs=date_obs, timestamp=_parse_time(date_obs),
        bayer=(str(h.get("BAYERPAT", "") or "").strip() or (prof[2] if prof else "")).upper(),
        width=width, height=height, binning=binning, ra=ra, dec=dec,
        focallen=_num(h, "FOCALLEN") or (prof[0] if prof else None), pixsize=pixsize,
        instrument=str(h.get("INSTRUME", "") or "").strip(), device_rejected=bool(rej),
        telescope=telescop or str(h.get("CREATOR", "") or "").strip(),
        kind=kind,
        camera_slot=("cam_1" if (kind == "dwarf" and (os.path.basename(folder).upper().find("WIDE") >= 0
                                                    or str(h.get("CAMERA", "")).strip().lower() == "wide"))
                     else ("cam_0" if kind == "dwarf" else None)),
    )


def sensor_name(f: Frame) -> str:
    """SPCC ``-oscsensor`` name for a frame's rig ("" when unknown): DESIGN section 4 mapping."""
    t = _norm(f"{f.instrument} {f.telescope}")
    if f.kind == "dwarf":
        return "Sony IMX662" if "mini" in t else ("Sony IMX415" if re.search(r"dwarf(ii|2)\b", t) else "Sony IMX678")
    if f.kind == "seestar":
        if "pro" in t:
            return "Sony IMX585"                    # S50 Pro / S30 Pro carry the IMX585 (Naztronomy mapping)
        return "ZWO Seestar S30" if "s30" in t else "ZWO Seestar S50"
    m = re.search(r"asi(\d{3,4})", t)
    if m and m.group(1) in _ASI_SENSOR:
        return _ASI_SENSOR[m.group(1)]
    m = re.search(r"imx(\d{3})", t)
    return f"Sony IMX{m.group(1)}" if m else ""


def list_fits(folder: str) -> list[str]:
    """FITS files directly in ``folder`` (any case of extension), sorted by name."""
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    return sorted(os.path.join(folder, n) for n in names
                  if n.lower().endswith(FITS_EXT) and os.path.isfile(os.path.join(folder, n)))


def scan(folders: list[str], limit: int | None = None, progress=None, workers: int = 8) -> FrameSet:
    """Read and classify the lights in ``folders`` (each used as given: the caller resolves
    ``lights/`` subfolders, see ``resolve_light_folders``).  ``limit`` keeps the first N subs by time
    (development option)."""
    folders = [os.path.abspath(f) for f in folders]
    use = [f for f in folders if os.path.basename(f).lower() != "rejected"] or folders
    files = [(p, d) for d in use for p in list_fits(d)]
    skipped: dict[str, int] = {}
    frames: list[Frame] = []
    calib_files: list[str] = []

    def one(item):
        return _read_frame(*item)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for item, r in zip(files, ex.map(one, files)):
            done += 1
            if progress and done % 200 == 0:
                progress(done, len(files), f"Scanning frames ({done}/{len(files)})")
            if isinstance(r, Frame):
                frames.append(r)
            else:
                skipped[r] = skipped.get(r, 0) + 1
                if r == "calibration frame":
                    calib_files.append(item[0])
    if not frames:
        return FrameSet(folders=use, frames=[], skipped=skipped, calibration_files=calib_files)
    groups: dict[tuple, list[Frame]] = {}
    for f in frames:
        groups.setdefault(f.group_key, []).append(f)
    best_key = max(groups, key=lambda k: (sum(f.exptime for f in groups[k]), len(groups[k])))
    best = sorted(groups[best_key], key=lambda f: (f.timestamp, f.name))
    ref = best[0]
    hdr, other = [], []
    for k, g in groups.items():
        if k == best_key:
            continue
        same_target = (k[0], k[1], k[4], k[5], k[6]) == (best_key[0], best_key[1], best_key[4], best_key[5], best_key[6])
        eg = ExposureGroup(g[0].exptime, g[0].gain, sorted(g, key=lambda f: (f.timestamp, f.name)), key=k)
        ratio = max(eg.exptime, ref.exptime) / max(min(eg.exptime, ref.exptime), 1e-6)
        # an HDR group is stacked on its own: it needs the stack's minimum of subs (a single 60 s sub beside
        # the 30 s set - the DWARF rejected the others as failed_ - made a one-frame unit Siril cannot calibrate)
        if same_target and ratio >= 2.0 and len(g) >= MIN_STACK_SUBS:
            hdr.append(eg)
        else:
            other.append({"object": g[0].object, "filter": g[0].filter, "exptime": g[0].exptime, "n": len(g),
                          "size": [g[0].width, g[0].height],
                          **({"reason": f"fewer than {MIN_STACK_SUBS} subs for an HDR group"}
                             if same_target and ratio >= 2.0 else {})})
    if limit:
        best = _limit_per_folder(best, limit)
    _unique_names(best + [f for g in hdr for f in g.frames])
    fs = FrameSet(folders=use, frames=best, exposure_groups=sorted(hdr, key=lambda g: g.exptime),
                  skipped=skipped, other_groups=other, calibration_files=calib_files, kind=ref.kind, sensor=sensor_name(ref),
                  shots_info=shots_info(ref.folder) if ref.kind == "dwarf" else {})
    return fs


def _limit_per_folder(frames: list[Frame], limit: int) -> list[Frame]:
    """Development option: ``limit`` subs in total, an equal share of the earliest subs of every night folder
    (so a multi-night dataset stays multi-night)."""
    folders = sorted({f.folder for f in frames})
    quota = max(1, limit // max(len(folders), 1))
    out: list[Frame] = []
    for d in folders:
        out.extend([f for f in frames if f.folder == d][:quota])
    return sorted(out, key=lambda f: (f.timestamp, f.name))


def _unique_names(frames: list[Frame]) -> None:
    """Subs from several folders can share a file name (Light_0001.fits every night): those are
    prefixed by their folder, since ``name`` identifies a sub (overrides, thumbnails)."""
    counts: dict[str, int] = {}
    for f in frames:
        counts[f.name] = counts.get(f.name, 0) + 1
    for f in frames:
        if counts[f.name] > 1:
            tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.basename(f.folder)).strip("_")
            f.name = f"{tag}__{f.name}"


def resolve_light_folders(path: str) -> list[str]:
    """Light folders of a user-given path (api-contract section 1 ``_light_folders``): the folder
    itself when it holds FITS, else its ``lights`` child (plus ``lights/rejected``), else the path."""
    if list_fits(path):
        return [path]
    try:
        for c in os.listdir(path):
            d = os.path.join(path, c)
            if c.lower() == "lights" and os.path.isdir(d) and list_fits(d):
                out = [d]
                for c2 in os.listdir(d):
                    if c2.lower() == "rejected" and list_fits(os.path.join(d, c2)):
                        out.append(os.path.join(d, c2))
                return out
    except OSError:
        pass
    return [path]
