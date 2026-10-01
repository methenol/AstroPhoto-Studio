"""Frame discovery, raw FITS reading, CFA helpers and cosmetic correction."""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime

import cv2
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter

CHANNEL_INDEX = {"R": 0, "G": 1, "B": 2}


@dataclass
class FrameInfo:
    path: str
    name: str
    object: str
    filter: str
    exptime: float
    gain: float
    temp: float
    date_obs: str
    timestamp: float
    bayer: str
    bias: float
    width: int
    height: int
    ra: float | None
    dec: float | None
    focallen: float
    pixsize: float
    instrument: str = ""   # INSTRUME: camera model (the sensor's spectral response for colour calibration)
    telescope: str = ""    # the instrument profile that filled missing headers (instruments.py)
    sensor: str = ""       # SPCC sensor curve known from the profile ("" = detect from the headers)
    camera_slot: str | None = None   # DWARF calibration camera (cam_0 telephoto / cam_1 wide-angle)
    device_rejected: bool = False    # the telescope's own live stack rejected this sub (DWARF "failed_")
    calib: dict | None = field(default=None, repr=False)   # calibration masters (calibration.py)
    adu_scale: float = 1.0           # raw values x this = 16-bit ADU (DWARF 3: 12-bit data, x16)

    def to_dict(self):
        return asdict(self)


def _parse_time(s: str) -> float:
    try:
        return datetime.fromisoformat(s.replace("Z", "")).timestamp()
    except Exception:
        return 0.0


def read_info(path: str) -> FrameInfo:
    """A sub's metadata from its FITS header.  What the header leaves out comes from the
    telescope's profile, and for a DWARF from its file name and shotsInfo.json (instruments.py)."""
    from . import instruments
    h = fits.getheader(path)
    prof = instruments.identify(h, path)
    fn = instruments.parse_dwarf_name(path) if prof.dwarf else {}
    si = instruments.shots_info(os.path.dirname(os.path.abspath(path))) if prof.dwarf else {}

    def num(key, *alts, fallback=None):
        for k in (key, *alts):
            v = h.get(k)
            if v not in (None, ""):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return fallback

    def pick(*vals, default=None):
        return next((v for v in vals if v not in (None, "")), default)
    width, height = int(h["NAXIS1"]), int(h["NAXIS2"])
    # data written at the ADC's bit depth (DWARF 3: 0..4095) is scaled to 16 bits, so the white
    # level, the saturation tests and every ADU threshold mean the same as for a 16-bit camera
    scale = float(2 ** (16 - prof.bit_depth)) if prof.bit_depth and int(h.get("BITPIX", 16)) == 16 else 1.0
    binning = int(num("XBINNING", fallback=0) or si.get("binning") or 0)
    if not binning:
        # a DWARF 3 telephoto sub at 1920x1080 is binned 2x2 from its 3840x2160 sensor
        binning = 2 if prof.key == "dwarf3" and max(width, height) <= 1920 else 1
    date_obs = pick(str(h.get("DATE-OBS", "") or "").strip(), fn.get("date_obs"), default="")
    filt = pick(str(h.get("FILTER", "") or "").strip(), fn.get("filter"), si.get("filter"), default="")
    if isinstance(filt, (int, float)):
        filt = instruments.DWARF_FILTERS.get(int(filt), str(filt))
    ra = instruments.coord(h.get("RA"), hours=isinstance(h.get("RA"), str))
    if ra is None:
        ra = instruments.coord(h.get("OBJCTRA"), hours=True)      # "00 42 44" in hours
    dec = instruments.coord(h.get("DEC"), hours=False)
    if dec is None:
        dec = instruments.coord(h.get("OBJCTDEC"), hours=False)
    return FrameInfo(
        path=path,
        name=os.path.basename(path),
        object=str(pick(str(h.get("OBJECT", "") or "").strip(), fn.get("object"), si.get("target"), default="")),
        filter=str(filt),
        exptime=float(pick(num("EXPTIME", "EXPOSURE"), fn.get("exptime"), si.get("exptime"), default=0.0)),
        gain=float(pick(num("GAIN"), fn.get("gain"), si.get("gain"), default=0.0)),
        temp=float(pick(num("CCD-TEMP", "DET-TEMP", "SENSTEMP"), fn.get("temp"), si.get("temp"),
                        default=float("nan"))),
        date_obs=date_obs,
        timestamp=_parse_time(date_obs),
        bayer=str(h.get("BAYERPAT", "") or "").strip() or prof.bayer or instruments.GENERIC.bayer,
        bias=float(h.get("BIAS", 0) or 0) * scale,
        width=width,
        height=height,
        ra=ra if ra is not None else si.get("ra"),
        dec=dec if dec is not None else si.get("dec"),
        focallen=float(num("FOCALLEN", fallback=None) or prof.focallen or instruments.GENERIC.focallen),
        pixsize=float(num("XPIXSZ", fallback=None) or (prof.pixsize * binning if prof.pixsize else None)
                      or instruments.GENERIC.pixsize),
        instrument=str(h.get("INSTRUME", "") or "").strip(),
        telescope=prof.name or str(h.get("TELESCOP", "") or "").strip(),
        sensor=prof.sensor or "",
        camera_slot=instruments.camera_slot(prof),
        device_rejected=bool(fn.get("failed")),
        adu_scale=scale,
    )


def discover(folder: str) -> list[FrameInfo]:
    """Find light frames in ``folder``, keep the dominant (filter, exposure, size) group."""
    paths = sorted(
        p for ext in ("*.fit", "*.fits", "*.fts", "*.FIT", "*.FITS")
        for p in glob.glob(os.path.join(folder, ext))
    )
    # the telescope's own products next to the subs: stacks, thumbnails
    paths = sorted(p for p in set(paths)
                   if not re.search(r"stack|thumbnail|_thn|^master", os.path.basename(p), re.IGNORECASE))
    infos = []
    for p in paths:
        try:
            h = fits.getheader(p)
            if str(h.get("IMAGETYP", "Light") or "").strip().lower() not in ("light", "light frame", ""):
                continue
            if int(h.get("NAXIS", 0)) != 2:
                continue
            infos.append(read_info(p))
        except Exception:
            continue
    if not infos:
        return []
    groups: dict[tuple, list[FrameInfo]] = {}
    for fi in infos:
        groups.setdefault((fi.filter, fi.width, fi.height, fi.bayer), []).append(fi)
    best = max(groups.values(), key=lambda g: sum(f.exptime for f in g))
    _check_adu_scale(best)
    return sorted(best, key=lambda f: f.timestamp or 0)


def _check_adu_scale(infos: list[FrameInfo]):
    """A profile's bit depth is only trusted while the data agrees: a sub with values beyond it
    (a firmware that scales to 16 bits) turns the scaling off for the whole session."""
    scale = infos[0].adu_scale
    if scale == 1.0:
        return
    for info in infos[:: max(1, len(infos) // 3)][:3]:
        if float(fits.getdata(info.path).max()) > 65535.0 / scale:
            for i in infos:
                i.bias /= i.adu_scale
                i.adu_scale = 1.0
            return


def load_raw_adu(path: str, scale: float = 1.0) -> np.ndarray:
    """A raw CFA frame as float32 16-bit ADU (``scale``: FrameInfo.adu_scale), nothing removed."""
    data = fits.getdata(path).astype(np.float32)
    if scale != 1.0:
        data *= scale
    return data


def read_raw(path: str, bias: float, calib: dict | None = None, scale: float = 1.0) -> np.ndarray:
    """Read a raw CFA frame as float32 with the black level removed: by the calibration
    masters (bias, dark, flat; calibration.py) when the session has them, else by the
    header's BIAS (clipped at 0, as always).

    A calibrated frame is not clipped: a DWARF 3's Duo-Band sky is a few ADU above black with
    ~10x that in read noise, and clipping the negative half of the noise would add a bias of
    the same size as the faint signal.  Everything downstream is linear in the data."""
    data = load_raw_adu(path, scale)
    if calib:
        from .calibration import apply
        return apply(data, calib)
    data -= bias
    np.maximum(data, 0, out=data)
    return data


def read_frame(info: FrameInfo) -> np.ndarray:
    """``read_raw`` of a sub with its session's calibration."""
    return read_raw(info.path, info.bias, getattr(info, "calib", None), getattr(info, "adu_scale", 1.0))


# --------------------------------------------------------------------------- CFA

def cfa_channel_map(pattern: str, shape: tuple[int, int]) -> np.ndarray:
    """Return an int8 array with the colour index (0=R,1=G,2=B) of every CFA site."""
    pattern = pattern.upper()
    cmap = np.empty(shape, np.int8)
    for k, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        cmap[dy::2, dx::2] = CHANNEL_INDEX[pattern[k]]
    return cmap


def cfa_masks(pattern: str, shape: tuple[int, int]) -> np.ndarray:
    cmap = cfa_channel_map(pattern, shape)
    return np.stack([(cmap == c).astype(np.float32) for c in range(3)], axis=-1)


def superpixel(raw: np.ndarray, pattern: str) -> np.ndarray:
    """2x2 super-pixel debayer -> half resolution RGB (no interpolation)."""
    pattern = pattern.upper()
    h, w = raw.shape
    h2, w2 = h // 2 * 2, w // 2 * 2
    sites = {(0, 0): pattern[0], (0, 1): pattern[1], (1, 0): pattern[2], (1, 1): pattern[3]}
    out = np.zeros((h2 // 2, w2 // 2, 3), np.float32)
    cnt = [0, 0, 0]
    for (dy, dx), c in sites.items():
        ci = CHANNEL_INDEX[c]
        out[..., ci] += raw[dy:h2:2, dx:w2:2]
        cnt[ci] += 1
    for ci in range(3):
        out[..., ci] /= max(cnt[ci], 1)
    return out


_CV_CODE_CACHE: dict[str, int] = {}


def _opencv_bayer_code(pattern: str) -> int:
    """Empirically find the OpenCV edge-aware demosaic code for ``pattern``.

    OpenCV's Bayer naming is notoriously offset, so we test each code on a
    synthetic mosaic rather than trusting names.
    """
    if pattern in _CV_CODE_CACHE:
        return _CV_CODE_CACHE[pattern]
    cmap = cfa_channel_map(pattern, (16, 16))
    truth = np.array([1000, 5000, 20000], np.float32)
    mosaic = truth[cmap].astype(np.uint16)
    for code in (cv2.COLOR_BayerBG2RGB_EA, cv2.COLOR_BayerGB2RGB_EA,
                 cv2.COLOR_BayerRG2RGB_EA, cv2.COLOR_BayerGR2RGB_EA):
        rgb = cv2.cvtColor(mosaic, code)[4:12, 4:12].reshape(-1, 3).mean(0)
        if np.allclose(rgb, truth, rtol=0.02):
            _CV_CODE_CACHE[pattern] = code
            return code
    raise ValueError(f"Unsupported Bayer pattern {pattern}")


def demosaic(raw: np.ndarray, pattern: str) -> np.ndarray:
    """Edge-aware demosaic of a (bias-subtracted, float) CFA frame -> RGB float32."""
    code = _opencv_bayer_code(pattern.upper())
    # OpenCV's edge-aware demosaic takes integers: offset so calibrated noise below black survives
    # (the interpolation is affine, so the offset comes back out exactly)
    off = float(max(0.0, -float(raw.min()))) if raw.size else 0.0
    off = np.ceil(off)
    u16 = np.clip(np.round(raw + off), 0, 65535).astype(np.uint16)
    rgb = cv2.cvtColor(u16, code).astype(np.float32)
    if off:
        rgb -= off
    return rgb


# ------------------------------------------------------------------- cosmetic

def build_defect_map(infos: list[FrameInfo], n_sample: int = 24, k_sigma: float = 7.0) -> np.ndarray:
    """Detect hot/cold pixels from the median of a sample of *unregistered* frames.

    Real sky features wander between frames (tracking drift and alt-az field
    rotation) while sensor defects stay put, so the temporal median keeps the
    defects.  A pixel is flagged when it deviates strongly from same-colour
    neighbours **and** is isolated (its immediate neighbours are not elevated),
    which protects the cores of stars that barely moved.
    """
    idx = np.linspace(0, len(infos) - 1, min(n_sample, len(infos))).round().astype(int)
    stack = np.stack([read_frame(infos[i]) for i in np.unique(idx)])
    med = np.median(stack, axis=0)
    del stack
    h, w = med.shape
    defects = np.zeros((h, w), bool)
    for dy in (0, 1):
        for dx in (0, 1):
            plane = med[dy::2, dx::2]
            resid = plane - median_filter(plane, size=5, mode="reflect")
            sigma = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-6
            defects[dy::2, dx::2] = np.abs(resid) > k_sigma * sigma
    # isolation test on the full-resolution mosaic (8-neighbourhood, any colour)
    neigh = median_filter(med, footprint=np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], bool), mode="reflect")
    local_bg = median_filter(med, size=9, mode="reflect")
    elevated_neigh = (neigh - local_bg) > 0.25 * np.abs(med - local_bg)
    defects &= ~elevated_neigh
    # pixels the calibration masters show to be unusable (hot beyond linear range, dead in the flat)
    from .calibration import defects as calib_defects
    extra = calib_defects(getattr(infos[0], "calib", None), defects.shape)
    if extra is not None:
        defects |= extra
    return defects


def fix_defects(raw: np.ndarray, defects: np.ndarray | None) -> np.ndarray:
    """Replace defective CFA pixels with the median of same-colour neighbours."""
    if defects is None or not defects.any():
        return raw
    ys, xs = np.nonzero(defects)
    h, w = raw.shape
    vals = []
    for oy, ox in ((-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (-2, 2), (2, -2), (2, 2)):
        yy = np.clip(ys + oy, 0, h - 1)
        xx = np.clip(xs + ox, 0, w - 1)
        v = raw[yy, xx].copy()
        v[defects[yy, xx]] = np.nan
        vals.append(v)
    rep = np.nanmedian(np.stack(vals), axis=0)
    rep = np.where(np.isfinite(rep), rep, np.median(raw))
    raw[ys, xs] = rep
    return raw
