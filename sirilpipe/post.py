"""The post-stack stages of the Siril backend: Restore, Star removal, Auto-finish, Export, preview.

``session.py`` calls the five functions below with the session folder (the layout of DESIGN section 3):

    restore(dir, stack_meta, stack_params, params, progress, checkpoint)       -> restore_meta   (linear.fits)
    remove_stars(dir, stack_meta, params, stack_params, progress, checkpoint)  -> starnet_meta   (starless.fits, stars.fits)
    autofinish(dir, stack_meta, params, progress, checkpoint)                  -> {"params", "report"}
    export(dir, stack_meta, params, export_opts, progress, checkpoint)         -> {"jpg", "tif", "json", "size"}
    preview_jpeg(dir, stack_meta, params, max_side=1600)                       -> bytes

Every step has a chain of alternatives and records which one ran (``restore_meta.json``): a tool that is
missing, unlicensed (RC-Astro exit 77) or failing raises :class:`tools.ToolUnavailable` / ``ToolError`` and the
next one takes over; nothing is ever inpainted or synthesised (project rule): photons resolve to star or sky.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from datetime import datetime

import cv2
import numpy as np

from . import export as export_mod
from . import finish, fitsio, tools
from .siril import SirilError, SirilRunner
from .tools import ToolError, ToolUnavailable
from .util import Cancelled

RL_TIMEOUT = float(os.environ.get("RL_TIMEOUT", 1800))      # Siril's Richardson-Lucy: ~40 s / iteration on a loaded 8 MP frame

# Siril failing = a failed script, or the binary not being startable at all (no Siril on a dev machine)
SIRIL_FAILS = (SirilError, OSError)

try:
    from . import __version__
except Exception:                                     # pragma: no cover
    __version__ = "2.0.0"


# =========================================================================== parameters

# Processing parameters (api-contract section 3).  Same keys and defaults as the old backend so older
# clients keep working; keys that have no effect any more stay accepted and are listed in the sidecar's
# ``ignored_params`` (see IGNORED_KEYS).
DEFAULTS: dict = {
    # linear stage
    "crop": True, "crop_threshold": 0.3,
    "background": True, "bg_method": "auto", "bg_correction": "subtract", "bg_degree": 2, "bg_smoothing": 0.5,
    "white_balance": "auto", "spcc_sensor": "auto", "spcc_filter": "auto", "spcc_white_ref": "average_spiral_galaxy",
    "denoise": 0.95, "deconvolution": 0.7, "restored_resolution": 1.0,
    # non-linear stage
    "palette": "auto", "oiii_boost": 1.0, "synthetic_luminance": 0.5,
    "stretch_method": "ghs", "stretch": 0.09, "auto_stretch": True, "hdr": 0.6, "contrast": 2.0,
    "color_preservation": 0.7,
    "star_separation": True, "star_removal": "auto", "star_reduction": 0.35, "star_intensity": 0.9,
    "star_saturation": 1.0, "halo_suppress": 0.6, "star_color_preservation": 0.5,
    "saturation": 1.3, "chroma_denoise": 0.8, "luminance_denoise": 0.6, "oiii_unmix": True,
    "local_contrast": 1.0, "scnr": 0.4, "black_point": 0.0, "brightness": 0.0, "sharpen": 0.25,
    # colour grade: neutral; not applied by this backend
    "grade_amount": 1.0, "grade_temperature": 0.0, "grade_tint": 0.0, "grade_warm_hue": 0.0, "grade_warm_sat": 1.0,
    "grade_cool_hue": 0.0, "grade_cool_sat": 1.0, "grade_contrast": 0.0,
}

# Accepted but without effect in this backend (the stack is cropped by the stack job, the ImageMM
# restoration and the colour-grade module do not exist any more).
IGNORED_KEYS = ("crop", "crop_threshold", "restored_resolution", "grade_amount", "grade_temperature", "grade_tint",
                "grade_warm_hue", "grade_warm_sat", "grade_cool_hue", "grade_cool_sat", "grade_contrast")


# Honoured but not part of the 47-key API contract (DEFAULTS): ``star_stretch`` is the Seti Astro Star Stretch amount, written by autofinish
# (finish.STAR_STRETCH when absent).
EXTRA_KEYS = ("star_stretch",)


def _R(group, key, label, lo, hi, step):
    return {"group": group, "key": key, "label": label, "type": "range", "min": lo, "max": hi, "step": step}


def _B(group, key, label):
    return {"group": group, "key": key, "label": label, "type": "bool"}


def _S(group, key, label, options):
    return {"group": group, "key": key, "label": label, "type": "select", "options": options}


# UI metadata: only the keys that have an effect now (labels and ranges as in webui/server.py where the
# meaning is unchanged).  spcc_sensor / spcc_filter options are completed by the server from the SPCC database.
PARAM_SPEC: list[dict] = [
    _B("Linear", "background", "Gradient removal (GraXpert AI background extraction)"),
    _S("Linear", "bg_method", "Gradient model (auto: GraXpert AI, Siril RBF when the model is missing)", ["auto", "poly", "rbf"]),
    _S("Linear", "bg_correction", "Gradient correction", ["subtract", "divide"]),
    _R("Linear", "bg_degree", "Polynomial degree (poly model)", 1, 4, 1),
    _R("Linear", "bg_smoothing", "Gradient smoothing", 0, 1, 0.05),
    _S("Linear", "white_balance", "Colour calibration (auto: Siril SPCC with Gaia)", ["auto", "stars", "background", "none"]),
    _S("Linear", "spcc_sensor", "Colour calibration: camera sensor", ["auto"]),
    _S("Linear", "spcc_filter", "Colour calibration: filter", ["auto"]),
    _S("Linear", "spcc_white_ref", "Colour calibration: white reference", ["average_spiral_galaxy", "g2v"]),
    _R("Linear", "denoise", "Light denoise before stretch (NoiseXTerminator, GraXpert or Siril)", 0, 1, 0.05),
    _R("Linear", "deconvolution", "Deconvolution (BlurXTerminator, else Richardson-Lucy)", 0, 1, 0.05),
    _S("Stretch", "stretch_method", "Stretch (ghs: denoise + generalised hyperbolic stretch; mtf, asinh, log: statistical)", ["ghs", "mtf", "asinh", "log"]),
    _R("Stretch", "stretch", "Sky level (ghs) / median target (statistical)", 0.04, 0.35, 0.01),
    _B("Stretch", "auto_stretch", "Adapt statistical stretch to target size (not used by ghs)"),
    _R("Stretch", "hdr", "HDR (protect bright cores; statistical stretch only)", 0, 1.5, 0.05),
    _R("Stretch", "contrast", "GHS local intensity b (highest tried; the fit picks the highest that meets the sky / nebula levels)", 0.25, 13, 0.25),
    _R("Stretch", "color_preservation", "Colour-preserving stretch", 0, 1, 0.05),
    _S("Colour", "palette", "Palette", ["auto", "natural", "hoo", "foraxx", "hoo_warm"]),
    _R("Colour", "oiii_boost", "OIII normalisation strength (1 = Blanshan / Cranfield equal level; applied after the stretch)", 0.5, 3, 0.05),
    _R("Colour", "synthetic_luminance", "Synthetic luminance (dual-band)", 0, 1, 0.05),
    _B("Colour", "oiii_unmix", "Remove Ha leakage from OIII"),
    _R("Colour", "saturation", "Saturation", 0.5, 3, 0.05),
    _R("Colour", "chroma_denoise", "Colour noise reduction", 0, 1, 0.05),
    _R("Colour", "scnr", "SCNR (remove green cast, broadband only)", 0, 1, 0.05),
    _B("Stars", "star_separation", "Process stars separately"),
    _S("Stars", "star_removal", "Star removal (auto: StarXTerminator, else StarNet2)", ["auto", "sxt", "starnet", "none"]),
    _R("Stars", "star_reduction", "Star reduction (1 = starless)", 0, 1, 0.05),
    _R("Stars", "star_intensity", "Star brightness", 0, 1.5, 0.05),
    _R("Stars", "star_saturation", "Star colour", 0, 2.5, 0.05),
    _R("Stars", "star_color_preservation", "Star colour intensity (stretch)", 0, 1, 0.05),
    _R("Stars", "halo_suppress", "Halo suppression", 0, 1, 0.05),
    _R("Detail", "luminance_denoise", "Noise reduction (wavelet shrinkage before the stretch + NoiseXTerminator / GraXpert after; 0.6 = standard)", 0, 1.5, 0.05),
    _R("Detail", "local_contrast", "Local contrast (Revela structure gain 1 + x on the signal)", 0, 2, 0.05),
    _R("Detail", "sharpen", "Final sharpening", 0, 1.5, 0.05),
    _R("Finish", "black_point", "Black point", 0, 0.2, 0.005),
    _R("Finish", "brightness", "Midtones", -1, 1, 0.05),
]

# Same presets as before ("Balanced" = DEFAULTS); only keys that are still honoured matter.
PRESETS: dict = {
    "Balanced": {},
    "Vivid nebula": {"saturation": 1.8, "local_contrast": 1.4, "stretch": 0.09, "contrast": 4.0, "oiii_boost": 1.0,
                     "star_reduction": 0.5, "star_intensity": 0.8},
    "Starless-ish": {"star_reduction": 0.8, "star_intensity": 0.45, "local_contrast": 1.0},
    "Galaxy / broadband": {"palette": "natural", "stretch": 0.09, "contrast": 4.0, "saturation": 1.4,
                           "local_contrast": 0.8, "star_reduction": 0.2, "bg_degree": 2},
    "Natural colour": {"palette": "natural", "saturation": 1.3, "scnr": 0.6},
    "Deep & dark": {"stretch": 0.06, "black_point": 0.0, "contrast": 6.0, "local_contrast": 1.0},
}

# Sensor / filter mapping for Siril's SPCC database (workflows.md 7.3, Naztronomy FILTER_COMMANDS_MAP).
# Narrowband wavelengths for the DWARF Duo-Band filter: Naztronomy 2.0.7 (Ha 656.28 nm / 18 nm, OIII 500.70 nm / 30 nm
# for both G and B; an approximation, as siril.md 11 notes).
SPCC_NARROWBAND = ["-narrowband", "-rwl=656.28", "-rbw=18", "-gwl=500.70", "-gbw=30", "-bwl=500.70", "-bbw=30"]
SPCC_WHITE_REF = {"average_spiral_galaxy": "Average Spiral Galaxy", "g2v": "Star, type G2(v)"}


# =========================================================================== helpers

def _p(session: str, *parts: str) -> str:
    return os.path.join(session, *parts)


def _load_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_json(path: str, doc: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, default=export_mod._json_default)
    os.replace(tmp, path)


def _sanitize(arr: np.ndarray) -> np.ndarray:
    """Finite and within [0, 1] (the tools rescale anything else and the inverse would be lost).  A tiny
    negative excess (background subtraction) is clipped; values above 1 (SPCC gains) are scaled in by the maximum."""
    arr = np.nan_to_num(arr.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    mx = float(arr.max())
    if 1.0 < mx <= 4.0:
        arr = arr / np.float32(mx)
    return np.clip(arr, 0.0, 1.0)


def normalise_scale(arr: np.ndarray) -> tuple[np.ndarray, float]:
    """Bring a stack to a known scale: peak = 1.0.  DWARF 3 data is 12-bit (the stack peaks near 0.0625), Seestar and
    ASI585 are 16-bit.  Done explicitly, once, in the restore (and for a stack-only export) so that no tool's implicit
    rescale (GraXpert / RC-Astro rescale out-of-range input) can change the linear relation between files.
    The factor is recorded in restore_meta.json (``scale``)."""
    peak = float(np.percentile(arr[:, ::7, ::7], 99.999)) if arr.size else 1.0
    peak = max(peak, float(arr.max()) * 0.5, 1e-6)
    if peak < 0.5:
        k = float(min(1.0 / float(arr.max()), 64.0)) if arr.max() > 0 else 1.0
        return (arr * np.float32(k)).astype(np.float32), k
    return arr, 1.0


def _has_wcs(header) -> bool:
    return all(k in header for k in ("CTYPE1", "CRVAL1", "CRPIX1")) and any(k in header for k in ("CD1_1", "PC1_1", "CDELT1"))


def _dual_band(meta: dict, header=None) -> bool:
    if "is_dualband" in meta and meta["is_dualband"] is not None:
        return bool(meta["is_dualband"])
    return finish.is_narrowband(meta.get("filter") or (header.get("FILTER") if header is not None else None))


def _write_like(path: str, arr: np.ndarray, header) -> None:
    fitsio.write_rgb(path, arr, header)


def _read_like(path: str, ref_header) -> np.ndarray:
    """Read a Siril output; rows are put back in the reference image's row order (Siril writes its own
    bottom-up order; a TOP-DOWN input would otherwise come back upside down)."""
    arr, hdr = fitsio.read_rgb(path)
    if export_mod.bottom_up(hdr) != export_mod.bottom_up(ref_header):
        arr = np.ascontiguousarray(arr[:, ::-1])
    return arr


class _Steps:
    """Step recorder + progress reporter shared by the stages."""

    def __init__(self, n: int, progress, checkpoint):
        self.n, self.i = n, 0
        self.progress, self.checkpoint = progress, checkpoint
        self.records: list[dict] = []

    def begin(self, msg: str) -> None:
        self.i += 1
        self.check()
        if self.progress:
            self.progress(min(self.i, self.n), self.n, msg)

    def sub(self, _i=None, _n=None, msg: str = "") -> None:
        """Sub-progress from a tool: keep our step counters, show the tool's message."""
        if self.progress and msg:
            self.progress(min(self.i, self.n), self.n, msg)

    def check(self) -> None:
        if self.checkpoint:
            self.checkpoint()

    def add(self, step: str, tool: str, status: str = "done", seconds: float = 0.0, note: str = "", **params) -> dict:
        rec = {"step": step, "tool": tool, "status": status, "seconds": round(seconds, 1), "params": params}
        if note:
            rec["note"] = note
        self.records.append(rec)
        return rec


def _siril_run(work: str, steps: _Steps, script: str, stage: str, expect=(), timeout: float | None = None) -> object:
    runner = SirilRunner(work, progress=steps.sub, checkpoint=steps.checkpoint, timeout=timeout)
    return runner.run(script, stage, expect=list(expect))


def _siril_image(work: str, steps: _Steps, name: str, arr: np.ndarray, header, body: str, stage: str,
                 with_log: bool = False, timeout: float | None = None):
    """Write ``arr``, run ``load name_in; <body>; save name_out`` in Siril, read the result back.

    Returns ``(array, header_out)``, plus the interesting lines of Siril's log when ``with_log`` (SPCC prints
    its fitted factors there).  The caller's arrays are bottom-up (Siril's own order), so the rows come back
    in the same order."""
    fin, fout = os.path.join(work, f"{name}_in.fit"), os.path.join(work, f"{name}_out.fit")
    if os.path.exists(fout):
        os.remove(fout)
    _write_like(fin, arr, header)
    try:
        res = _siril_run(work, steps, f"load {name}_in\n{body}\nsave {name}_out\n", stage, expect=[f"{name}_out.fit"],
                         timeout=timeout)
        out, hdr = fitsio.read_rgb(fout)
        if export_mod.bottom_up(hdr) != export_mod.bottom_up(header):
            out = np.ascontiguousarray(out[:, ::-1])
        if with_log:
            keep = [ln for ln in res.tail if re.search(r"spcc|gaia|white|factor|K[012]|catalog|stars", ln, re.I)][-8:]
            return out, hdr, keep
        return out, hdr
    finally:
        for f in (fin, fout):
            if os.path.exists(f):
                os.remove(f)


def _device_gpu(stack_params: dict | None) -> str:
    return str((stack_params or {}).get("device") or "auto")


# =========================================================================== restore (linear)

def spcc_arguments(header, meta: dict, p: dict) -> tuple[list[str], str]:
    """Siril ``spcc`` arguments for this camera and filter (workflows.md 7.3) and a description.

    Explicit ``spcc_sensor`` / ``spcc_filter`` win.  Raises ``ValueError`` when the camera cannot be told
    (the caller then falls back to background neutralisation)."""
    ident = " ".join(str(x) for x in (header.get("TELESCOP"), header.get("INSTRUME"), header.get("CREATOR"),
                                      header.get("CAMERA"), meta.get("instrument"), meta.get("telescope"),
                                      meta.get("kind"), meta.get("sensor")) if x).lower()
    filt = str(header.get("FILTER") or meta.get("filter") or "")
    fnorm = re.sub(r"[^a-z0-9]", "", filt.lower())
    sensor = None
    if "seestar" in ident or re.search(r"\bs(30|50)\b", ident):
        pro = "pro" in ident
        sensor = "Sony IMX585" if pro else ("ZWO Seestar S30" if "s30" in ident else "ZWO Seestar S50")
    elif "dwarf" in ident:
        sensor = "Sony IMX662" if "mini" in ident else ("Sony IMX415" if re.search(r"dwarf ?(2|ii)\b", ident) else "Sony IMX678")
    else:
        m = re.search(r"imx ?(\d{3})", ident)
        if m:
            sensor = f"Sony IMX{m.group(1)}"
        elif "585" in ident:
            sensor = "Sony IMX585"
        elif "294" in ident:
            sensor = "Sony IMX294"
        elif "533" in ident:
            sensor = "Sony IMX533"
        elif "2600" in ident or "571" in ident:
            sensor = "Sony IMX571"
    if meta.get("sensor") and not sensor:
        sensor = str(meta["sensor"])                    # stack_meta.sensor is already an SPCC oscsensor name
    if p.get("spcc_sensor") not in (None, "", "auto"):
        sensor = str(p["spcc_sensor"])
    if not sensor:
        raise ValueError("camera sensor unknown (no SPCC database entry guessed from the headers)")
    args = [f"-oscsensor={sensor}", "-catalog=gaia", f"-whiteref={SPCC_WHITE_REF.get(p.get('spcc_white_ref'), 'Average Spiral Galaxy')}"]
    desc = sensor
    if p.get("spcc_filter") not in (None, "", "auto"):
        args.append(f"-oscfilter={p['spcc_filter']}")
        desc += f" + {p['spcc_filter']}"
    elif fnorm in ("duoband", "dualband", "duo"):
        args += SPCC_NARROWBAND
        desc += " + dual-band (narrowband mode)"
    elif fnorm == "lp" and "seestar" in ident:
        args.append("-oscfilter=ZWO Seestar LP")
        desc += " + ZWO Seestar LP"
    else:
        args.append("-oscfilter=UV/IR Block")
        desc += " + UV/IR Block"
    return args, desc


def _spcc_database_present() -> bool:
    home = os.environ.get("HOME") or os.path.expanduser("~")
    return os.path.isdir(os.path.join(home, ".local", "share", "siril-spcc-database"))


class Restorer:
    """The linear post-processing chain on ``stack.fits`` -> ``linear.fits`` (DESIGN section 4 "Restore")."""

    def __init__(self, session: str, meta: dict, stack_params: dict, p: dict, progress, checkpoint):
        self.session, self.meta, self.sp, self.p = session, meta, stack_params or {}, p
        self.steps = _Steps(7, progress, checkpoint)
        self.work = _p(session, "process", "restore")
        os.makedirs(self.work, exist_ok=True)
        # session.py may point the restore at another file with stack.fits' grid (ImageMM's imagemm.fits)
        self.source = str((meta or {}).get("restore_source") or "stack.fits")
        if not os.path.exists(_p(session, self.source)):
            self.source = "stack.fits"
        self.arr, self.header = fitsio.read_rgb(_p(session, self.source))
        if not export_mod.bottom_up(self.header):                    # one row order for everything downstream (Siril's)
            self.arr = np.ascontiguousarray(self.arr[:, ::-1])
            self.header["ROWORDER"] = "BOTTOM-UP"
        self.arr, self.scale = normalise_scale(_sanitize(self.arr))
        self.gpu = _device_gpu(self.sp)
        self.dual = _dual_band(meta, self.header)
        self.psf_corrected = False
        self.info: dict = {"spcc": None}

    # ---- plumbing
    def _file(self, arr: np.ndarray, name: str) -> str:
        path = os.path.join(self.work, name)
        _write_like(path, arr, self.header)
        return path

    def _out(self, name: str) -> str:
        return os.path.join(self.work, name)

    def _rc(self, product: str, arr: np.ndarray, tag: str, **params) -> tuple[np.ndarray, dict]:
        """One RC-Astro run on ``arr``; raises ToolUnavailable / ToolError for the caller's fallback."""
        s = self.steps
        src, dst = self._file(arr, f"{tag}_in.fits"), self._out(f"{tag}_out.fits")
        meta = tools.rcastro.run(product, src, dst, device=_device_gpu(self.sp), progress=s.sub,
                                 checkpoint=s.checkpoint, **params)
        return _sanitize(fitsio.read_rgb(dst)[0]), meta

    # ---- steps
    def background(self) -> None:
        s, p = self.steps, self.p
        s.begin("Removing gradient")
        if not p["background"]:
            s.add("gradient", "none", "skipped", note="background=false")
            return
        method = p["bg_method"] if p["bg_method"] in ("poly", "rbf") else "auto"
        smoothing = float(p["bg_smoothing"])
        t0 = time.time()
        if method == "auto":
            try:
                src, dst = self._file(self.arr, "bge_in.fits"), self._out("bge_out.fits")
                meta = tools.graxpert.background_extraction(src, dst, smoothing=smoothing, correction=p["bg_correction"],
                                                            gpu=self.gpu, progress=s.sub, checkpoint=s.checkpoint)
                self.arr = _sanitize(fitsio.read_rgb(dst)[0])
                s.add("gradient", "graxpert", seconds=time.time() - t0, smoothing=smoothing,
                      correction=meta["correction"], model=meta.get("model"), providers=meta.get("providers"))
                return
            except Cancelled:
                raise
            except (ToolUnavailable, ToolError) as e:
                s.add("gradient", "graxpert", "failed", time.time() - t0, note=str(e))
        # Siril: polynomial or RBF subtraction (also the fallback when the AI model is missing)
        if method == "poly":
            cmd = f"subsky {int(np.clip(p['bg_degree'], 1, 4))} -samples=20 -tolerance=1.0"
        else:
            cmd = f"subsky -rbf -samples=20 -tolerance=1.0 -smooth={smoothing:.2f}"
        t0 = time.time()
        try:
            out, _ = _siril_image(self.work, s, "bg", self.arr, self.header, cmd, "Removing gradient")
            self.arr = _sanitize(out)
            s.add("gradient", "siril subsky", seconds=time.time() - t0, command=cmd)
        except SIRIL_FAILS as e:
            s.add("gradient", "siril subsky", "failed", time.time() - t0, note=str(e).splitlines()[0])

    def plate_solve(self) -> bool:
        """WCS for SPCC.  The stack job normally solved it already; solve again only when it is missing."""
        s = self.steps
        s.begin("Plate solving")
        if _has_wcs(self.header):
            s.add("platesolve", "none", "skipped", note="the stack is already plate-solved")
            return True
        hints = ""
        if self.header.get("FOCALLEN") and self.header.get("XPIXSZ"):
            scale = float(self.meta.get("scale") or 1.0) or 1.0
            hints = f" -focal={float(self.header['FOCALLEN']):g} -pixelsize={float(self.header['XPIXSZ']) / scale:g}"
        t0 = time.time()
        try:
            out, hdr = _siril_image(self.work, s, "solve", self.arr, self.header, f"platesolve -force{hints}", "Plate solving")
            if not _has_wcs(hdr):
                raise SirilError("Siril reported success but wrote no WCS")
            self.arr = _sanitize(out)
            for card in hdr.cards:                                   # take over the astrometric solution (WCS + SIP)
                if re.match(r"(CTYPE|CRVAL|CRPIX|CD\d|PC\d|CDELT|CUNIT|CROTA|LONPOLE|LATPOLE|EQUINOX|RADESYS|[AB]P?_|PLTSOLVD|"
                            r"OBJCTRA|OBJCTDEC|RA$|DEC$)", card.keyword):
                    self.header[card.keyword] = (card.value, card.comment)
            s.add("platesolve", "siril", seconds=time.time() - t0)
            return True
        except SIRIL_FAILS as e:
            s.add("platesolve", "siril", "failed", time.time() - t0, note=str(e).splitlines()[0])
            return False

    def psf_correction(self) -> None:
        """BlurXTerminator "Correct Only" before colour calibration (RC-Astro's manual: it makes SPCC more
        consistent).  Only when BXT is licensed and deconvolution is wanted; otherwise nothing runs here."""
        s, p = self.steps, self.p
        s.begin("PSF correction")
        if self.source == "imagemm.fits":
            s.add("psf-correction", "none", "skipped", note="deconvolution by ImageMM")
            return
        if float(p["deconvolution"]) <= 0 or (self.sp.get("deconv_method") or "auto") == "none":
            s.add("psf-correction", "none", "skipped", note="deconvolution off")
            return
        t0 = time.time()
        try:
            tools.rcastro.require("bxt")
            self.arr, meta = self._rc("bxt", self.arr, "bxtc", correct_only=True)
            self.psf_corrected = True
            s.add("psf-correction", "rc-astro bxt", seconds=time.time() - t0, correct_only=True, flags=meta.get("flags"))
        except Cancelled:
            raise
        except (ToolUnavailable, ToolError) as e:
            s.add("psf-correction", "rc-astro bxt", "unavailable", time.time() - t0, note=str(e))

    def colour(self, solved: bool) -> None:
        s, p = self.steps, self.p
        s.begin("Colour calibration (SPCC)")
        mode = p["white_balance"]
        if mode == "none":
            s.add("colour", "none", "skipped", note="white_balance=none")
            return
        if mode == "auto":
            t0 = time.time()
            try:
                if not solved:
                    raise SirilError("the image is not plate-solved")
                if not _spcc_database_present():
                    raise SirilError("the SPCC sensor/filter database is not installed")
                args, desc = spcc_arguments(self.header, self.meta, p)
                body = "spcc " + " ".join(f'"{a}"' for a in args)
                out, _, log = _siril_image(self.work, s, "spcc", self.arr, self.header, body,
                                           "Colour calibration (SPCC)", with_log=True)
                self.arr = _sanitize(out)
                self.info["spcc"] = {"args": args, "description": desc, "log": log}
                s.add("colour", "siril spcc", seconds=time.time() - t0, sensor=desc, catalog="gaia")
                return
            except Cancelled:
                raise
            except (*SIRIL_FAILS, ValueError) as e:
                note = str(e).splitlines()[0]
                s.add("colour", "siril spcc", "failed", time.time() - t0, note=note)
                self.info["spcc"] = {"failed": note}
                mode = "fallback"
        t0 = time.time()
        if mode == "stars":
            self.arr, det = finish.star_white_balance(self.arr)
            self.arr, det2 = finish.neutralize_background(self.arr)
            s.add("colour", "stars + background neutralisation", seconds=time.time() - t0, **det, **det2)
        else:                                                        # "background", or the fallback after a failed SPCC
            self.arr, det = finish.neutralize_background(self.arr)
            note = ""
            if mode == "fallback":
                note = "SPCC failed: background neutralisation" + ("" if self.dual else " + rmgreen")
                if not self.dual:
                    self.arr = finish.rmgreen_np(self.arr, 0.5)      # average-neutral SCNR (= Siril rmgreen type 0)
            s.add("colour", "background neutralisation", seconds=time.time() - t0, note=note, **det)
        self.arr = _sanitize(self.arr)

    def deconvolve(self) -> None:
        s, p = self.steps, self.p
        s.begin("Deconvolution")
        if self.source == "imagemm.fits":
            s.add("deconvolution", "imagemm", "skipped", note="deconvolution by ImageMM")
            return
        d = float(np.clip(p["deconvolution"], 0, 1))
        method = (self.sp.get("deconv_method") or "auto")
        if d <= 0 or method == "none":
            s.add("deconvolution", "none", "skipped", note="deconvolution=0" if d <= 0 else "deconv_method=none")
            return
        t0 = time.time()
        try:                                                         # BlurXTerminator sharpening (auto PSF)
            if method == "rl":                                       # the n2n-rl profile: Siril's own deconvolution
                raise ToolUnavailable("deconv_method=rl: Siril Richardson-Lucy requested")
            tools.rcastro.require("bxt")
            ss, sn = 0.25 + 0.25 * d, 0.5 + 0.4 * d                  # DESIGN section 4; AstroPipe: ss 0.25, sn 0.60, halos 0
            extra: dict = {}
            fwhm = float(self.meta.get("fwhm_px") or 0)
            if p.get("bxt_manual_psf") and 1.0 < fwhm:                 # RC-Astro manual: PSF diameter = the measured star FWHM in px (max 8)
                extra = {"ansp": False, "nsd": min(8.0, fwhm)}
            self.arr, meta = self._rc("bxt", self.arr, "bxt", ss=ss, sn=sn, ash=0.0, **extra)
            s.add("deconvolution", "rc-astro bxt", seconds=time.time() - t0, ss=round(ss, 3), sn=round(sn, 3),
                  flags=meta.get("flags"), **extra)
            return
        except Cancelled:
            raise
        except (ToolUnavailable, ToolError) as e:
            s.add("deconvolution", "rc-astro bxt", "unavailable", time.time() - t0, note=str(e))
        t0 = time.time()
        # DESIGN: rl -iters=10..30 -alpha=3000 -tv.  fftw_timelimit=0 = FFTW "estimate" planning: with Siril's default
        # (measure, 60 s per plan) the first run on every new image size spent 10+ minutes planning FFTs (measured:
        # 1200 px crop 101 s -> 6 s once the wisdom existed; with the limit at 0 a 1800 px crop takes 13 s from scratch).
        iters = int(round(10 + 20 * d))
        try:
            out, _ = _siril_image(self.work, s, "rl", self.arr, self.header,
                                  f"set core.fftw_timelimit=0\nmakepsf stars\nrl -iters={iters} -alpha=3000 -tv\n"
                                  "set core.fftw_timelimit=60", "Deconvolution",
                                  timeout=RL_TIMEOUT)
            res = _sanitize(out)
            self.arr = (self.arr + np.float32(d) * (res - self.arr)).astype(np.float32)     # strength = linear blend
            s.add("deconvolution", "siril rl", seconds=time.time() - t0, iterations=iters, alpha=3000, tv=True,
                  blend=round(d, 2))
        except SIRIL_FAILS as e:
            s.add("deconvolution", "siril rl", "failed", time.time() - t0, note=str(e).splitlines()[0])

    def denoise(self) -> None:
        s, p = self.steps, self.p
        s.begin("Noise reduction")
        d = float(np.clip(p["denoise"], 0, 1))
        if d <= 0:
            s.add("denoise", "none", "skipped", note="denoise=0")
            return
        t0 = time.time()
        try:
            tools.rcastro.require("nxt")
            dn = round(0.3 * d, 3)         # light linear pass (AstroPipe: NXT 0.25 after BXT); the main pass is on the starless
            self.arr, meta = self._rc("nxt", self.arr, "nxt", dn=dn)
            s.add("denoise", "rc-astro nxt", seconds=time.time() - t0, dn=dn, flags=meta.get("flags"))
            return
        except Cancelled:
            raise
        except (ToolUnavailable, ToolError) as e:
            s.add("denoise", "rc-astro nxt", "unavailable", time.time() - t0, note=str(e))
        t0 = time.time()
        try:
            src, dst = self._file(self.arr, "gxd_in.fits"), self._out("gxd_out.fits")
            strength = round(0.5 * d, 3)                              # DESIGN: GraXpert strength = 0.5 x denoise
            meta = tools.graxpert.denoise(src, dst, strength=strength, gpu=self.gpu, progress=s.sub, checkpoint=s.checkpoint)
            self.arr = _sanitize(fitsio.read_rgb(dst)[0])
            s.add("denoise", "graxpert", seconds=time.time() - t0, strength=strength, model=meta.get("model"),
                  providers=meta.get("providers"))
            return
        except Cancelled:
            raise
        except (ToolUnavailable, ToolError) as e:
            s.add("denoise", "graxpert", "unavailable", time.time() - t0, note=str(e))
        t0 = time.time()
        mod = round(min(1.0, 0.85 * d), 2)
        try:
            out, _ = _siril_image(self.work, s, "dn", self.arr, self.header, f"denoise -mod={mod}", "Noise reduction")
            self.arr = _sanitize(out)
            s.add("denoise", "siril denoise", seconds=time.time() - t0, mod=mod)
        except SIRIL_FAILS as e:
            s.add("denoise", "siril denoise", "failed", time.time() - t0, note=str(e).splitlines()[0])

    def save(self) -> dict:
        s = self.steps
        hdr = self.header.copy()
        for r in s.records:
            if r["status"] == "done" and r["tool"] != "none":
                hdr.add_history(f"AstroPhoto Studio {__version__}: {r['step']} with {r['tool']}")
        fitsio.write_rgb(_p(self.session, "linear.fits"), self.arr, hdr)
        lum = finish.luminance(self.arr)
        ran = [r["tool"] for r in s.records if r["status"] == "done" and r["tool"] != "none"]
        return {"created": datetime.now().isoformat(timespec="seconds"), "steps": s.records, "tools": ran,
                "scale": self.scale, "wcs": _has_wcs(self.header), "dual_band": self.dual, "spcc": self.info["spcc"],
                "stats": {"median": round(float(np.median(finish.sample(lum))), 6),
                          "noise_sigma": round(finish.lower_half_sigma(finish.sample(lum)), 6),
                          "max": round(float(self.arr.max()), 4)},
                "name": " + ".join(ran) or "none",
                "params": {k: self.p[k] for k in ("background", "bg_method", "bg_correction", "bg_smoothing",
                                                  "white_balance", "denoise", "deconvolution")}}


def restore(session_dir: str, stack_meta: dict, stack_params: dict, params: dict, progress=None, checkpoint=None) -> dict:
    """Linear restoration: gradient -> plate solve -> [BXT correct only] -> colour calibration -> deconvolution
    -> light denoise, ``stack.fits`` -> ``linear.fits`` (+ ``restore_meta.json``).

    Each step tries its best tool and falls back (GraXpert -> Siril RBF; SPCC -> background neutralisation +
    rmgreen; BXT -> Siril Richardson-Lucy; NXT -> GraXpert -> Siril denoise) and records what ran.  RC-Astro's
    BlurXTerminator manual puts "Correct Only" before colour calibration (it makes SPCC more consistent), so
    that half of BXT runs before SPCC and the sharpening half after it."""
    p = {**DEFAULTS, **(params or {})}
    r = Restorer(session_dir, stack_meta or {}, stack_params or {}, p, progress, checkpoint)
    t0 = time.time()
    r.background()
    solved = r.plate_solve()
    r.psf_correction()
    r.colour(solved)
    r.deconvolve()
    r.denoise()
    r.steps.begin("Saving linear image")
    meta = r.save()
    meta["seconds"] = round(time.time() - t0, 1)
    _save_json(_p(session_dir, "restore_meta.json"), meta)
    return meta


# =========================================================================== star removal

STAR_FILES = ("starless.fits", "stars.fits")


def _linear_source(session: str) -> str:
    for name in ("linear.fits", "stack.fits"):
        if os.path.exists(_p(session, name)):
            return _p(session, name)
    raise RuntimeError("Nothing to process: stack the dataset first")


def remove_stars(session_dir: str, stack_meta: dict, params: dict, stack_params: dict, progress=None, checkpoint=None) -> dict:
    """Separate the stars: ``linear.fits`` -> ``starless.fits`` + ``stars.fits`` (+ ``starnet_meta.json``).

    Order: StarXTerminator (licensed; RC-Astro: best on linear data) -> StarNet2 ``--linear`` -> none.  The
    star layer is ``linear - starless`` clipped at 0, exact for linear data (RC-Astro: unscreen is for stretched
    images only).  When StarNet2 cannot run on linear data the meta says ``stage: stretched`` and the export
    removes the stars after the stretch instead."""
    p = {**DEFAULTS, **(params or {})}
    sp = stack_params or {}
    steps = _Steps(3, progress, checkpoint)
    t_all = time.time()
    for f in STAR_FILES:
        if os.path.exists(_p(session_dir, f)):
            os.remove(_p(session_dir, f))
    meta: dict = {"tool": "none", "stage": "none", "attempts": [], "created": datetime.now().isoformat(timespec="seconds")}
    wanted = str(p["star_removal"]).lower()
    if not p["star_separation"] or wanted in ("none", "classic"):
        meta["reason"] = "star_separation=false" if not p["star_separation"] else f"star_removal={wanted}"
        _save_json(_p(session_dir, "starnet_meta.json"), meta)
        return meta
    order = {"sxt": ["sxt"], "starnet": ["starnet"]}.get(wanted, ["sxt", "starnet"])
    work = _p(session_dir, "process", "stars")
    os.makedirs(work, exist_ok=True)
    src = _linear_source(session_dir)
    lin, header = fitsio.read_rgb(src)
    starless_path = _p(session_dir, "starless.fits")
    ok = False
    for tool in order:
        steps.begin("Star removal")
        t0 = time.time()
        try:
            if tool == "sxt":
                tools.rcastro.require("sxt")
                m = tools.rcastro.run("sxt", src, starless_path, device=_device_gpu(sp), progress=steps.sub,
                                      checkpoint=steps.checkpoint, stars=True)
                side = m.get("stars_path")
                if side and os.path.exists(side):
                    os.remove(side)                                   # recomputed below, the same way for every tool
                meta.update(tool="sxt", stage="linear", flags=m.get("flags"))
            else:
                try:
                    m = tools.starnet.run(src, starless_path, linear=True, progress=steps.sub, checkpoint=steps.checkpoint)
                    meta.update(tool="starnet2", stage="linear", version=m.get("version"), gpu=m.get("gpu"))
                except ToolError as e:
                    meta["attempts"].append({"tool": "starnet2 --linear", "status": "failed", "note": str(e)})
                    if "linear" not in (e.log or str(e)).lower():     # a real failure, not an old CLI without --linear
                        raise
                    # a StarNet2 without --linear (< 2.5): remove the stars after the stretch, in the export
                    meta.update(tool="starnet2", stage="stretched")
                    _save_json(_p(session_dir, "starnet_meta.json"), meta)
                    meta["seconds"] = round(time.time() - t_all, 1)
                    return meta
            ok = True
            meta["attempts"].append({"tool": meta["tool"], "status": "done", "seconds": round(time.time() - t0, 1)})
            break
        except Cancelled:
            raise
        except (ToolUnavailable, ToolError) as e:
            meta["attempts"].append({"tool": tool, "status": "unavailable" if isinstance(e, ToolUnavailable) else "failed",
                                     "note": str(e)})
    if ok:
        steps.begin("Star removal")
        starless, _ = fitsio.read_rgb(starless_path)
        starless = np.clip(np.nan_to_num(starless), 0, 1).astype(np.float32)
        stars = np.maximum(lin - starless, 0).astype(np.float32)
        if meta["tool"] == "starnet2":              # StarNet leaves the scattered-light wings of bright stars in the starless layer
            t_w = time.time()
            starless, winfo, stars = finish.subtract_star_wings(lin, starless, stars)
            meta["wing_subtraction"] = {**winfo, "seconds": round(time.time() - t_w, 1)}
        fitsio.write_rgb(starless_path, starless, header)
        fitsio.write_rgb(_p(session_dir, "stars.fits"), stars, header)
        sky_noise = finish.lower_half_sigma(finish.sample(finish.luminance(starless)))
        n, per_mp = finish.count_stars(stars, sky_noise)
        meta["stars"] = {"count": n, "per_megapixel": round(per_mp, 1), "min_sigma": 5.0, "sky_noise": round(sky_noise, 6),
                         "flux_fraction": round(float(stars.sum() / max(float(lin.sum()), 1e-9)), 4)}
    meta["seconds"] = round(time.time() - t_all, 1)
    _save_json(_p(session_dir, "starnet_meta.json"), meta)
    return meta


# =========================================================================== export / preview

class _Layers:
    """The images the render needs, loaded once (and reduced for a preview)."""

    def __init__(self, session: str, p: dict, max_side: int | None = None):
        self.session = session
        self.src = _linear_source(session)
        self.base, self.header = fitsio.read_rgb(self.src)
        if os.path.basename(self.src) == "stack.fits":
            self.base, _ = normalise_scale(self.base)
        self.starnet = _load_json(_p(session, "starnet_meta.json"))
        self.stars = None
        self.separate = False
        self.stage = "linear"
        sl, st = _p(session, "starless.fits"), _p(session, "stars.fits")
        if p["star_separation"] and self.starnet.get("tool") not in (None, "none"):
            if self.starnet.get("stage") == "stretched":
                self.stage = "stretched"
            elif os.path.exists(sl) and os.path.exists(st) and os.path.getmtime(sl) >= os.path.getmtime(self.src):
                self.base = fitsio.read_rgb(sl)[0]
                self.stars = fitsio.read_rgb(st)[0]
                self.separate = True
        e = finish.EDGE_CROP
        if min(self.base.shape[1:]) > 8 * e:                  # the stack's outer rim: fewer dithered frames contribute, so it is noisier (and a wavelet / GraXpert artefact lives there)
            self.base = np.ascontiguousarray(self.base[:, e:-e, e:-e])
            if self.stars is not None:
                self.stars = np.ascontiguousarray(self.stars[:, e:-e, e:-e])
        self.fwhm_px = float(_load_json(_p(session, "stack_meta.json")).get("fwhm_px") or 3.0)
        if max_side:
            self.fwhm_px *= min(1.0, max_side / max(self.base.shape[1:]))
            self.base = _reduce(self.base, max_side)
            if self.stars is not None:
                self.stars = _reduce(self.stars, max_side)


def _reduce(arr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = arr.shape[1:]
    if max(h, w) <= max_side:
        return arr
    s = max_side / max(h, w)
    size = (max(1, int(round(w * s))), max(1, int(round(h * s))))
    return np.stack([cv2.resize(arr[c], size, interpolation=cv2.INTER_AREA) for c in range(3)])


_PREVIEW_CACHE: dict[tuple, _Layers] = {}


def preview_jpeg(session_dir: str, stack_meta: dict, params: dict, max_side: int = 1600) -> bytes:
    """A quick render of the best state on disk (starless + stars, else linear, else the stack) as JPEG bytes.

    Same stretch, palette and star math as the export, in numpy at reduced size; the Siril / RC-Astro / GraXpert
    steps (GHS contrast pass, second denoise) are left out, so the picture differs from the export in noise and
    contrast only slightly."""
    p = {**DEFAULTS, **(params or {})}
    src = _linear_source(session_dir)
    key = (session_dir, max_side, p["star_separation"]) + tuple(
        os.path.getmtime(_p(session_dir, f)) if os.path.exists(_p(session_dir, f)) else 0
        for f in ("stack.fits", "linear.fits", "starless.fits", "stars.fits", "starnet_meta.json"))
    layers = _PREVIEW_CACHE.get(key)
    if layers is None:
        layers = _Layers(session_dir, p, max_side)
        if len(_PREVIEW_CACHE) >= 3:
            _PREVIEW_CACHE.pop(next(iter(_PREVIEW_CACHE)))
        _PREVIEW_CACHE[key] = layers
    img, _ = _render(layers, p, _dual_band(stack_meta or {}, layers.header), fast=True)
    return export_mod.jpeg_bytes(export_mod.display_image(img, export_mod.bottom_up(layers.header)), 88, max_side)


def _render(layers: _Layers, p: dict, dual: bool, fast: bool, hooks: dict | None = None, tick=None,
            work: str | None = None, sp: dict | None = None, steps: _Steps | None = None) -> tuple[np.ndarray, dict]:
    """Run ``finish.render_layers`` on the loaded layers, doing the stretched-stage star removal when needed."""
    base, stars, pre, st_stretched = layers.base, layers.stars, False, False
    if layers.stage == "stretched" and not fast and work is not None and steps is not None:
        base, stars = _stretched_star_removal(layers, p, dual, work, sp or {}, steps)
        pre = st_stretched = stars is not None
        if stars is None:
            base = layers.base
    p = {**p, "_fwhm_px": layers.fwhm_px}
    img, info = finish.render_layers(base, stars, p, dual_band=dual, pre_stretched=pre, stars_stretched=st_stretched,
                                     hooks=hooks, tick=tick)
    info["star_layer"] = "separate" if stars is not None else "none"
    return img, info


def _stretched_star_removal(layers: _Layers, p: dict, dual: bool, work: str, sp: dict, steps: _Steps):
    """StarNet2 on the stretched image (its network expects non-linear data): stretch -> remove stars -> the star
    layer is the unscreened difference.  Returns (starless stretched, stars stretched) or (None, None)."""
    pos = finish.curve_position(p["stretch_method"], float(p["hdr"]))
    shd = finish.curve_shoulder(p["stretch_method"], float(p["hdr"]))
    cov = finish.object_coverage(finish.luminance(layers.base))
    target = finish.adapted_target(float(p["stretch"]), cov, bool(p["auto_stretch"]))
    full, _ = finish.stretch_rgb(layers.base, target, position=pos, shoulder=shd, color_preservation=float(p["color_preservation"]))
    src, dst = os.path.join(work, "sn_in.fits"), os.path.join(work, "sn_out.fits")
    fitsio.write_rgb(src, full, layers.header)
    try:
        tools.starnet.run(src, dst, linear=False, progress=steps.sub, checkpoint=steps.checkpoint)
    except Cancelled:
        raise
    except (ToolUnavailable, ToolError) as e:
        steps.add("star removal (stretched)", "starnet2", "failed", note=str(e))
        return None, None
    starless = np.clip(fitsio.read_rgb(dst)[0], 0, 1)
    return starless, finish.unscreen(full, starless)


class _ExportHooks:
    """The Siril / RC-Astro / GraXpert versions of the render's steps (export only)."""

    def __init__(self, work: str, header, steps: _Steps, sp: dict, tag_tools: dict):
        self.work, self.header, self.steps, self.sp, self.used = work, header, steps, sp, tag_tools
        self.gpu = _device_gpu(sp)

    def denoise(self, img: np.ndarray, p: dict):
        """Second noise pass on the stretched starless layer (the RC-Astro recommended place): NXT 0.8 when licensed,
        else GraXpert mild, else nothing (Siril's NL-Bayes would smear nebula detail at this point)."""
        s = self.steps
        ld = float(np.clip(p["luminance_denoise"], 0, 1.5))
        t0 = time.time()
        src, dst = os.path.join(self.work, "dn2_in.fits"), os.path.join(self.work, "dn2_out.fits")
        fitsio.write_rgb(src, np.clip(img, 0, 1), self.header)
        try:
            tools.rcastro.require("nxt")
            dn = round(min(0.95, 0.8 * ld / 0.6), 3)
            m = tools.rcastro.run("nxt", src, dst, device=self.gpu, progress=s.sub, checkpoint=s.checkpoint, dn=dn)
            s.add("denoise (stretched)", "rc-astro nxt", seconds=time.time() - t0, dn=dn, flags=m.get("flags"))
            return np.clip(fitsio.read_rgb(dst)[0], 0, 1)
        except Cancelled:
            raise
        except (ToolUnavailable, ToolError) as e:
            s.add("denoise (stretched)", "rc-astro nxt", "unavailable", time.time() - t0, note=str(e))
        t0 = time.time()
        try:
            strength = round(min(1.0, 0.5 * ld / 0.6), 3)
            m = tools.graxpert.denoise(src, dst, strength=strength, gpu=self.gpu, progress=s.sub, checkpoint=s.checkpoint)
            s.add("denoise (stretched)", "graxpert", seconds=time.time() - t0, strength=strength, model=m.get("model"),
                  providers=m.get("providers"))
            return np.clip(fitsio.read_rgb(dst)[0], 0, 1)
        except Cancelled:
            raise
        except (ToolUnavailable, ToolError) as e:
            s.add("denoise (stretched)", "graxpert", "unavailable", time.time() - t0, note=str(e))
        return None

    def final_color(self, img: np.ndarray, p: dict, dual: bool):
        """satu / rmgreen on the starless layer through Siril (numpy twin on failure); black point + midtones are applied after the star blend, in numpy."""
        t0 = time.time()
        lines = []
        amount = float(p["saturation"]) - 1.0
        if abs(amount) > 1e-3:
            lines.append(f"satu {amount:.3f}")
        if not dual and float(p["scnr"]) > 0:                     # no SCNR on dual-band: the green is the OIII
            lines.append(f"rmgreen 0 {float(np.clip(p['scnr'], 0, 1)):.3f}")
        if not lines:
            return None
        try:
            out, _ = _siril_image(self.work, self.steps, "fc", np.clip(img, 0, 1), self.header, "\n".join(lines), "Colour")
            self.steps.add("colour finish", "siril " + " + ".join(ln.split()[0] for ln in lines),
                           seconds=time.time() - t0, commands=lines)
            return np.clip(out, 0, 1)
        except SIRIL_FAILS as e:
            self.steps.add("colour finish", "siril", "failed", time.time() - t0, note=str(e).splitlines()[0] + " (numpy used)")
            return None


def _record_numpy_steps(steps: _Steps, info: dict, p: dict) -> None:
    """The render's numpy steps (denoise, background pass, GHS, narrowband normalisation) as records of the sidecar, with the numbers that drove them."""
    g = info.get("ghs") or {}
    if not g:
        return
    steps.add("denoise (linear)", "numpy starlet", amount=g.get("denoise"), noise_sigma=g.get("noise_sigma"), thresholds_k=list(finish.STARLET_K), cold_sigma=finish.COLD_SIGMA)
    steps.add("background (second pass)", "numpy poly", degree=finish.FLATTEN_DEGREE, span=g.get("flatten_span"))
    steps.add("stretch", "numpy ghs", sky_aim=float(p["stretch"]), nebula_aim=finish.NEBULA_AIM, anchors=g.get("anchors"), white=g.get("white"), sky_in=g.get("sky_in"),
              nebula_in=g.get("nebula_in"), bright_in=g.get("bright_in"), signal_frac=g.get("signal_frac"), **{k: v for k, v in (g.get("ghs") or {}).items()})
    pi = info.get("palette_info") or {}
    if "nbn_strength" in pi:
        steps.add("narrowband normalisation", "numpy nbn", oiii_boost=pi.get("oiii_boost"), midtones_strength=pi.get("nbn_strength"), ha_leak_k=pi.get("ha_leak_k"),
                  noise_gain=pi.get("gain"))


def export(session_dir: str, stack_meta: dict, params: dict, export_opts: dict | None = None, progress=None,
           checkpoint=None) -> dict:
    """The final render and the files: ``{"jpg", "tif", "json", "size"}`` (api-contract "Outputs")."""
    opts = {"quality": 95, "upscale": 1.0, "tiff": True}
    opts.update({k: v for k, v in (export_opts or {}).items() if v is not None})
    given = params or {}
    p = {**DEFAULTS, **given}
    meta = stack_meta or _load_json(_p(session_dir, "stack_meta.json"))
    steps = _Steps(2, progress, checkpoint)
    t_all = time.time()
    steps.begin("Stretching")
    layers = _Layers(session_dir, p)
    dual = _dual_band(meta, layers.header)
    work = _p(session_dir, "process", "finish")
    os.makedirs(work, exist_ok=True)
    sp = (meta.get("params") or {}) if isinstance(meta, dict) else {}
    hooks_obj = _ExportHooks(work, layers.header, steps, sp, {})
    hooks = {"denoise": hooks_obj.denoise, "final_color": hooks_obj.final_color}

    def tick(name):
        steps.sub(msg=f"Stretching ({name.lower()})")

    t0 = time.time()
    img, info = _render(layers, p, dual, fast=False, hooks=hooks, tick=tick, work=work, sp=sp, steps=steps)
    info["render_seconds"] = round(time.time() - t0, 1)
    _record_numpy_steps(steps, info, p)
    steps.begin("Exporting")
    disp = export_mod.display_image(img, export_mod.bottom_up(layers.header))
    lum = finish.sample(finish.luminance(layers.base))
    restore_meta = _load_json(_p(session_dir, "restore_meta.json"))
    linear_info = {
        "source": os.path.basename(layers.src), "star_layer": info.get("star_layer"), "star_removal": layers.starnet.get("tool", "none"),
        "star_removal_stage": layers.starnet.get("stage", "none"), "restoration": restore_meta.get("name", "none (stack only)"),
        "palette": info.get("palette"), "dual_band": dual, "target_median": info.get("target_median"),
        "coverage": info.get("coverage"), "stretch": info.get("stretch") or info.get("palette_info"), "stars": info.get("stars"),
        "noise_ref": round(finish.lower_half_sigma(lum), 6), "crop": meta.get("crop"),
        "steps": steps.records, "render_seconds": info["render_seconds"], "seconds": round(time.time() - t_all, 1),
        "siril_tools": sorted({r["tool"] for r in steps.records if r["status"] == "done"}),
    }
    ignored = sorted(k for k in given if k in IGNORED_KEYS or (k not in DEFAULTS and k not in EXTRA_KEYS))
    files = export_mod.write_outputs(session_dir, disp, meta=meta, params_full=p, linear_info=linear_info,
                                     quality=int(opts["quality"]), upscale=float(opts["upscale"] or 1.0),
                                     tiff=bool(opts["tiff"]), ignored=ignored)
    return files


# =========================================================================== auto-finish

def autofinish(session_dir: str, stack_meta: dict, params: dict, progress=None, checkpoint=None) -> dict:
    """Tune the processing settings from measurements of the image (autofinish.py): ``{"params", "report"}``,
    also written to ``autofinish.json``."""
    from . import autofinish as af
    return af.run(session_dir, stack_meta or {}, params or {}, progress, checkpoint)


# =========================================================================== dev entry point

def main(argv: list[str] | None = None) -> int:
    """``python -m sirilpipe.post --session DIR [--steps restore,stars,autofinish,export] [--device cpu]``"""
    import argparse
    ap = argparse.ArgumentParser(description="Run the post-stack stages on a session folder")
    ap.add_argument("--session", required=True)
    ap.add_argument("--steps", default="restore,stars,autofinish,export")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--params", default="{}", help="JSON object of processing params")
    args = ap.parse_args(argv)
    d = os.path.abspath(args.session)
    meta = _load_json(_p(d, "stack_meta.json"))
    sp = {**(meta.get("params") or {}), "device": args.device}
    params = json.loads(args.params)
    t_start = time.time()

    def progress(i, n, msg):
        print(f"[{time.time() - t_start:7.1f}s] {i}/{n} {msg}", flush=True)

    steps = args.steps.split(",")
    out: dict = {}
    if "restore" in steps:
        out["restore"] = restore(d, meta, sp, params, progress)
    if "stars" in steps:
        out["stars"] = remove_stars(d, meta, params, sp, progress)
    if "autofinish" in steps:
        af = autofinish(d, meta, params, progress)
        params = af["params"]
        out["autofinish"] = af["report"].get("summary", "ok")
    if "export" in steps:
        out["export"] = export(d, meta, params, {}, progress)
    print(json.dumps(out, indent=1, default=export_mod._json_default)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
