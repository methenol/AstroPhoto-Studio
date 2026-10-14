"""Optional ImageMM restoration stage on Siril's registered frames (``deconv_method = "imagemm"``).

ImageMM (Sukurdeep, Budavari, Connolly & Navarro 2025, arXiv:2501.03002) restores the *individual* exposures
jointly: ``y(t) = D(h(t) * x) + noise``, a PSF, a variance and a mask per exposure.  This stage feeds it with
the frames Siril registers for the stack (``seqapplyreg``: homography registration + Bayer drizzle, already
debayered RGB on the -framing=max canvas) instead of the repo's own raw-sub preparation, and reuses the
repo's implementation (``astrophoto/imagemm.py``) and its PSF machinery (``astrophoto/exposures.py``)
unchanged:

* frames: ``process/r_<seq>_NNNNN.fit`` of the frames ``frames.json`` marks ``included`` (Siril deletes them
  after the stack; they are registered again when missing, with the stack's own resampling), cropped to the
  stack's box ``stack_meta['crop']``, same row order as ``stack.fits``.  A frame is never resampled again.
* masks: a drizzled frame has no data where no drop landed (exactly 0 in Siril's output; the red and blue
  planes of a Bayer drizzle at scale 1 mostly) and outside the frame's footprint.  Those pixels are masked
  per channel.  Pixels whose drops used a saturated raw sample are the "censored" pixels of Whyte, Sivic &
  Zisserman (2014), exactly as in ``ExposureSet.window`` (``mm_restore(whyte=...)``): the saturation level
  is measured on the frames (``saturation_level``).
* photometric scale T, smooth sky deviation, reference sky: as ``ExposureSet.prepare_one`` (aperture photometry
  of isolated stars against the reference = ``stack.fits``), so the latent sky is 0.
* variances: the photon-transfer curve of consecutive frames (``ExposureSet._ptc_fit``): read + sky noise and
  shot noise per channel, measured on the data (no gain or read noise is invented).
* PSF: measured on the frames' *own* coadd (never on ``stack.fits``: the two resampling paths differ by ~3 % of
  a star's flux and that alone made dark rings round stars, see ``ExposureSet.prepare``), seeing-mixture
  kernels G(sigma_t) * Q per frame, unit-sum kernels (``imagemm.unit_sum``).
* one prepare at a time, serial (no process pools), then ``imagemm.restore`` with the paper's settings
  (Algorithm 3, kappa 2, delta 2, Eq. C15 stop), Whyte et al. for saturated stars.

Output: ``imagemm.fits`` (float32 (3, H, W), the row order, scale and crop of ``stack.fits``, WCS copied and
rescaled for r > 1) and ``imagemm_meta.json``.  Nothing here is imported unless the stage runs; torch is
imported inside ``available()`` / ``run()`` only.
"""
from __future__ import annotations

import glob
import logging
import math
import os
import shutil
import time

import cv2
import numpy as np

from . import fitsio
from .util import clean_json, read_json, write_json

log = logging.getLogger("sirilpipe")
STAGE = "imagemm"
OUT_FITS = "imagemm.fits"
OUT_META = "imagemm_meta.json"
WORK = os.path.join("process", "imagemm")          # prepared frames (deleted afterwards)

# stack_params keys (webui/server.py STACK_DEFAULTS) -> imagemm.restore arguments
PARAM_KEYS = {"imagemm_robust": "robust", "imagemm_delta": "delta", "imagemm_kappa": "kappa",
              "imagemm_epsilon": "epsilon", "imagemm_stop": "stop", "imagemm_max_iters": "max_iters",
              "imagemm_psf": "psf_model", "imagemm_groups": "n_groups", "imagemm_accelerate": "accelerate",
              "imagemm_background": "background"}
STOPS = ("c15", "flux")


def available() -> dict:
    """``{"ok", "torch", "cuda", "mps", "reason"}`` - whether the stage can run here.  ``ok`` needs torch (and the
    repo's ImageMM modules); a GPU is optional (the CPU is slow but exact)."""
    out = {"ok": False, "torch": False, "cuda": False, "mps": False, "reason": ""}
    try:
        import torch                                   # noqa: WPS433 (optional, large)
    except Exception as e:                             # ImportError, or a broken CUDA install
        out["reason"] = f"torch is not installed ({type(e).__name__}); build the image with IMAGEMM=cuda|cpu"
        return out
    out["torch"] = True
    out["version"] = str(torch.__version__)
    try:
        out["cuda"] = bool(torch.cuda.is_available())
        out["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        pass
    try:
        import astrophoto.imagemm  # noqa: F401
        import astrophoto.exposures  # noqa: F401
    except Exception as e:
        out["reason"] = f"the ImageMM modules do not import: {type(e).__name__}: {e}"
        return out
    out["ok"] = True
    return out


# --------------------------------------------------------------------------- parameters
def map_params(stack_params: dict | None) -> dict:
    """``imagemm.restore`` keyword arguments from ``stack_params`` (paper defaults where unset).

    imagemm_r is the latent super-resolution factor (output = r x the frames' grid); imagemm_sigma 0 = the
    paper's g_sigma (1 at r = 1, 1.1 above).  Development options (not in the UI): ``imagemm_max_frames`` (the
    N best frames by weight) and ``imagemm_region`` ([x, y, w, h] inside the stack's crop box)."""
    sp = stack_params or {}
    kw: dict = {"r": 1, "sigma": None, "psf_model": "empirical", "n_groups": 0, "robust": True, "delta": 2.0,
                "kappa": 2.0, "epsilon": 1e-6, "stop": "c15", "max_iters": 1000, "accelerate": False,
                "background": False}
    for key, arg in PARAM_KEYS.items():
        if sp.get(key) is not None:
            kw[arg] = sp[key]
    kw["r"] = int(min(3, max(1, round(float(sp.get("imagemm_r") or 1)))))
    # g_sigma of Eq. 11 (the paper: 1 at r = 1, 1.1 at r = 2).  At r = 1 the latent is the sky seen through a
    # 1 px Gaussian, band-limited, so a star between pixel centres is represented exactly; with the measured
    # PSFs directly the fit pulls light out of a ring round every star (IC 405)
    sig = float(sp.get("imagemm_sigma") or 0.0)
    kw["sigma"] = sig if sig > 0 else (1.1 if kw["r"] > 1 else 1.0)
    kw["robust"] = bool(kw["robust"])
    kw["accelerate"] = bool(kw["accelerate"])
    kw["background"] = bool(kw["background"])
    kw["delta"], kw["kappa"], kw["epsilon"] = float(kw["delta"]), float(kw["kappa"]), float(kw["epsilon"])
    kw["max_iters"] = int(kw["max_iters"])
    kw["n_groups"] = int(kw["n_groups"] or 0)
    kw["stop"] = str(kw["stop"]) if str(kw["stop"]) in STOPS else "c15"
    kw["psf_model"] = "moffat" if str(kw["psf_model"]) == "moffat" else "empirical"
    return kw


def _device_arg(stack_params: dict | None) -> str:
    return str((stack_params or {}).get("device") or "auto")


# --------------------------------------------------------------------------- registered frames
def included_rows(session_dir: str) -> list[dict]:
    rows = read_json(os.path.join(session_dir, "frames.json"), None)
    if not rows:
        raise RuntimeError("frames.json is missing: analyse and stack the dataset first")
    return rows


def resampling_of(stack_meta: dict) -> dict:
    """The resampling the stack was made with (``workflow.decide_resampling`` form), from stack_meta."""
    if stack_meta.get("mode") == "drizzle":
        d = stack_meta.get("drizzle") or {}
        return {"mode": "drizzle", "scale": float(d.get("scale") or stack_meta.get("scale") or 1.0),
                "pixfrac": float(d.get("pixfrac") or 1.0), "kernel": d.get("kernel") or "square"}
    return {"mode": "demosaic", "scale": float(stack_meta.get("scale") or 1.0), "interp": "lanczos4"}


def registered_files(process_dir: str, seq: str) -> list[str]:
    return sorted(glob.glob(os.path.join(process_dir, f"r_{seq}_[0-9]*.fit")))


def sequence_placements(seq_file: str, canvas: tuple[int, int]) -> dict[int, tuple[int, int, int, int]]:
    """Where each frame of a Siril registered sequence (``r_<seq>_.seq``) sits on the stack's canvas:
    {file number: (ox, oy, w, h)}, the frame's pixel (row, col) being the canvas pixel (oy + row, ox + col).

    ``seqapplyreg -framing=max`` writes every frame at its own size (a variable-size sequence) and records the
    translation of each in the registration matrix H (``R`` lines: [1 0 tx; 0 1 ty; 0 0 1], Siril's y up); the
    stack's ``-maximize`` then places the frames on the union canvas.  Measured against stack_raw.fit (phase
    correlation, 8 frames of M 42, score 0.76-0.96): ox = tx - min(tx) and oy = canvas_h - h - (ty - min(ty))
    (the array's row 0 is the bottom row)."""
    sizes = []
    layers: dict[int, list[list[float]]] = {}
    with open(seq_file, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("I "):
                t = ln.split()
                w, h = t[3].split(",")
                sizes.append((int(t[1]), int(w), int(h)))
            elif ln.startswith("R") and " H " in ln:
                t = ln.split()
                i = t.index("H")
                layers.setdefault(int(t[0][1:] or 0), []).append([float(v) for v in t[i + 1:i + 10]])
    n = len(sizes)
    hs = next((v for v in layers.values() if len(v) == n), None)
    cw, ch = canvas
    if hs is None:
        if all((w, h) == (cw, ch) for _, w, h in sizes):
            return {num: (0, 0, w, h) for num, w, h in sizes}
        raise RuntimeError(f"{os.path.basename(seq_file)}: no registration data to place the frames of different sizes")
    tx = np.array([m[2] for m in hs])
    ty = np.array([m[5] for m in hs])
    out = {}
    for (num, w, h), x, y in zip(sizes, tx, ty):
        out[num] = (int(round(x - tx.min())), int(round(ch - h - (y - ty.min()))), w, h)
    return out


def ensure_registered(session_dir: str, stack_meta: dict, progress=None, checkpoint=None) -> tuple[dict, dict]:
    """The registered frames of the included subs: ({row index: file}, info).

    Siril removes ``r_*`` after the stack (up to 100 MB each), so they are made again here with the same
    ``seqapplyreg`` the stack used (``workflow.apply_script``; all included frames, so that the -framing=max
    canvas is the stack's) and removed by ``run`` when the stage ends."""
    from . import workflow
    from .siril import SirilRunner
    process = os.path.join(session_dir, "process")
    state = read_json(os.path.join(process, "state.json"), {}) or {}
    seq, n = state.get("seq"), int(state.get("n") or 0)
    if not seq or not os.path.exists(os.path.join(process, f"{seq}_.seq")):
        raise RuntimeError("the registration (process/state.json, the calibrated sequence) is missing: stack again")
    if state.get("hdr"):
        raise RuntimeError("ImageMM does not support HDR (several exposure groups) stacks")
    rows = included_rows(session_dir)
    inc = [i for i, r in enumerate(rows) if r.get("included")]
    excluded = [i for i in range(len(rows)) if i not in set(inc)]
    rs = resampling_of(stack_meta)
    info = {"resampling": rs, "regenerated": False, "seq": seq}
    files = registered_files(process, seq)
    if len(files) != len(inc):
        workflow.clean_registered(process)
        runner = SirilRunner(process, progress=progress, checkpoint=checkpoint)
        runner.run(workflow.apply_script(process, seq, n, excluded, rs), "apply_registration_imagemm",
                   expect=[f"r_{seq}_.seq"], count_glob=f"r_{seq}_[0-9]*.fit", total=len(inc),
                   label="ImageMM: registering the frames")
        files = registered_files(process, seq)
        info["regenerated"] = True
    if len(files) != len(inc):
        raise RuntimeError(f"expected {len(inc)} registered frames, found {len(files)} in {process}")
    # Siril names the r_ frames like the sequence (file number = sequence index + 1) and skips the unselected ones
    by_row = {}
    for f in files:
        num = int(os.path.basename(f).rsplit("_", 1)[1].split(".")[0])
        by_row[num - 1] = f
    if set(by_row) != set(inc):
        by_row = dict(zip(inc, files))          # (a different numbering: they are in sequence order)
    cw, ch = stack_meta.get("stack_canvas") or [0, 0]
    places = sequence_placements(os.path.join(process, f"r_{seq}_.seq"), (int(cw), int(ch)))
    info["placements"] = {i: places[int(os.path.basename(f).rsplit("_", 1)[1].split(".")[0])] for i, f in by_row.items()}
    return by_row, info


# --------------------------------------------------------------------------- frames
def read_frame(path: str, box: tuple[int, int, int, int], place: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """The box (x, y, w, h) of the stack's canvas, as seen by a registered frame, float32 (h, w, 3): the frame
    sits on the canvas at ``place`` = (ox, oy, w, h) (``sequence_placements``); canvas pixels it does not cover are
    0 (no data).  Only the rows needed are read."""
    from astropy.io import fits
    x, y, w, h = box
    out = np.zeros((h, w, 3), np.float32)
    ox, oy, fw, fh = place if place is not None else (0, 0, w + x, h + y)
    x0, x1 = max(x, ox), min(x + w, ox + fw)
    y0, y1 = max(y, oy), min(y + h, oy + fh)
    if x1 <= x0 or y1 <= y0:
        return out
    with fits.open(path, memmap=True) as hdul:
        d = hdul[0].data
        if d.ndim == 2:
            d = np.repeat(d[None], 3, 0)
        if d.shape[1] != fh or d.shape[2] != fw:
            raise RuntimeError(f"{os.path.basename(path)}: {d.shape[2]} x {d.shape[1]}, the sequence says {fw} x {fh}")
        a = np.array(d[:, y0 - oy:y1 - oy, x0 - ox:x1 - ox], dtype=np.float32)
    out[y0 - y:y1 - y, x0 - x:x1 - x] = np.moveaxis(a, 0, -1)
    return out


def stack_is_flipped(ref: np.ndarray, frame: np.ndarray) -> tuple[bool, dict]:
    """Whether stack.fits has its rows in the opposite order to the canvas the registered frames sit on.

    Siril's plate solve rewrites stack.fits and the rewritten file reads back vertically reversed against the
    cropped ``stack_raw.fit`` (measured on M 42: identical to the crop flipped), with no ROWORDER card to say so.
    The frames are put in the stack's order, whichever it is: the normalised correlation of a frame (g channel,
    smoothed, 4x down-sampled, over its pixels with data) with the reference decides.  ``ref``/``frame``:
    (h, w, 3)."""
    def small(a):
        g = cv2.GaussianBlur(np.ascontiguousarray(a[..., 1], np.float32), (0, 0), 2.0)
        return cv2.resize(g, (max(8, g.shape[1] // 4), max(8, g.shape[0] // 4)), interpolation=cv2.INTER_AREA)
    have = cv2.resize((frame[..., 1] != 0).astype(np.float32), (max(8, frame.shape[1] // 4), max(8, frame.shape[0] // 4)),
                      interpolation=cv2.INTER_AREA) > 0.99

    def corr(a, b, m):
        if m.sum() < 100:
            return -1.0
        a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
        d = float(np.sqrt((a * a).sum() * (b * b).sum()))
        return float((a * b).sum() / d) if d > 0 else -1.0
    f, r = small(frame), small(ref)
    straight = corr(f, r, have)
    flipped = corr(f[::-1], r, have[::-1])
    return flipped > straight + 0.05, {"correlation": round(straight, 3), "correlation_flipped": round(flipped, 3)}


def saturation_level(frames: list[np.ndarray], plateau: int = 6, tol: float = 0.03) -> float | None:
    """Saturation level of the registered frames, in their units, measured on them: a saturated star is a
    plateau of (nearly) equal values at the sensor's ceiling.  Per frame and channel the maximum is the
    ceiling when at least ``plateau`` pixels lie within ``tol`` of it (a sharp unsaturated peak never has);
    the level is the median ceiling over the frames and channels that show a plateau, and None when fewer
    than a third of them do (no saturated star: nothing to censor)."""
    tops = []
    n = 0
    for y in frames:
        for c in range(3):
            v = y[..., c]
            m = float(v.max())
            n += 1
            if m > 0 and int((v >= (1 - tol) * m).sum()) >= plateau:
                tops.append(m)
    if not n or len(tops) < max(3, n // 3):
        return None
    return float(np.median(tops))


def measurement_masks(y: np.ndarray, sat: float | None, thr: float = 0.94) -> dict:
    """The masks of a drizzled frame y (h, w, 3), per channel unless noted:

    ``data``     pixels where a drop landed (Siril writes exactly 0 where none did) and finite;
    ``hole``     missing pixels inside the footprint (at least 5 of the 8 neighbours have data): the gaps of
                 the drop lattice, ~3 % of the red and blue planes of a Bayer drizzle at scale 1;
    ``region``   data or hole;  ``interior`` the region eroded by 1 px (its border pixels are the ones
                 with a partial drop weight at the edge of the footprint);
    ``cens``     pixels whose drops used a saturated raw sample (value >= thr x the saturation level, grown
                 by the drop footprint: 1 px for green, 2 px for the 2x2 drops of red and blue) that hold data
                 - Whyte et al.'s saturated pixels, recorded values kept;
    ``valid``    the ImageMM mask m: data, interior and not censored."""
    h, w, _ = y.shape
    data = np.isfinite(y) & (y != 0)
    k8 = np.ones((3, 3), np.float32)
    k8[1, 1] = 0
    hole = np.zeros_like(data)
    region = np.zeros_like(data)
    interior = np.zeros_like(data)
    cens = np.zeros_like(data)
    for c in range(3):
        d = data[..., c]
        nb = cv2.filter2D(d.astype(np.float32), -1, k8, borderType=cv2.BORDER_CONSTANT)
        hole[..., c] = ~d & (nb >= 5)
        region[..., c] = d | hole[..., c]
        interior[..., c] = cv2.erode(region[..., c].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        if sat is not None:
            s = (np.where(d, y[..., c], 0) >= thr * sat).astype(np.uint8)
            if s.any():
                g = 1 if c == 1 else 2
                s = cv2.dilate(s, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
                cens[..., c] = (s > 0) & d
    valid = data & interior & ~cens
    return {"data": data, "hole": hole, "region": region, "interior": interior, "cens": cens, "valid": valid}


def fill_holes(y: np.ndarray, masks: dict) -> np.ndarray:
    """Copy of y with the lattice holes replaced by the mean of their neighbours that hold data.  Used for
    measuring (aperture photometry, the sky surface, PSF-star cut-outs), where a single missing pixel among
    ~150 of a star's core would bias a flux; the restoration's mask keeps them masked (``valid``)."""
    out = y.copy()
    k8 = np.ones((3, 3), np.float32)
    k8[1, 1] = 0
    for c in range(3):
        hl = masks["hole"][..., c]
        if hl.any():
            d = masks["data"][..., c].astype(np.float32)
            num = cv2.filter2D(np.where(d > 0, y[..., c], 0).astype(np.float32), -1, k8, borderType=cv2.BORDER_CONSTANT)
            den = cv2.filter2D(d, -1, k8, borderType=cv2.BORDER_CONSTANT)
            out[..., c] = np.where(hl, num / np.maximum(den, 1), y[..., c])
    return out


def surface_window(coefs: np.ndarray, H: int, W: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """The smooth sky surface of a frame (``exposures.fit_smooth_surface`` coefficients (3, terms), normalised
    [-1, 1] coordinates of the (H, W) field) on the window, (3, y1-y0, x1-x0), evaluated exactly."""
    from astrophoto.exposures import _poly_terms
    x = (np.arange(x0, x1) + 0.5) / W * 2 - 1
    y = (np.arange(y0, y1) + 0.5) / H * 2 - 1
    xx, yy = np.meshgrid(x, y)
    terms = np.stack(_poly_terms(xx, yy, 2), 0)
    return np.tensordot(np.asarray(coefs, np.float64), terms, axes=([1], [0])).astype(np.float32)





def make_set_class():
    """``RegisteredSet``, an ``exposures.ExposureSet`` whose exposures are Siril's registered frames.  A function
    so that importing this module imports neither astrophoto nor torch."""
    from astrophoto import exposures as E

    class RegisteredSet(E.ExposureSet):
        """Prepared registered frames in the form ``imagemm.restore`` consumes (the ExposureSet interface:
        ``usable``, ``node_kernels``/``kernels``, ``windows`` with censored pixels, ``sat``, ``params[k]['T']``,
        ``psf_ref``).  Everything about PSFs, photometric scales, photon-transfer variances and the seeing
        mixture is ExposureSet's own code; only the frame source (no raw subs, no demosaic, no own
        registration: the frames are Siril's registered, drizzled RGB) and the window extraction differ."""

        PREP_VERSION = E.ExposureSet.PREP_VERSION

        def __init__(self, files: dict, rows: list, box: tuple[int, int, int, int], ref: np.ndarray,
                     work: str, scale: float, fwhm_guess: float, places: dict | None = None, progress=None,
                     checkpoint=None, sat_box: tuple[int, int, int, int] | None = None, flip: bool = False):
            import types
            self.keys = sorted(files)
            self.files = [files[k] for k in self.keys]
            self.places = [(places or {}).get(k) for k in self.keys]
            self.box = tuple(int(v) for v in box)
            self.flip = bool(flip)             # the frames are read upside down (see stack_is_flipped)
            self.sat_box = tuple(int(v) for v in (sat_box or box))       # where the saturation is looked for
            self.work = work
            self.progress = progress
            self.checkpoint = checkpoint
            self.H0, self.W0 = ref.shape[:2]
            self.ref = ref.astype(np.float32)
            self.pattern = None
            self.defects = np.zeros((self.H0, self.W0), bool)
            self.info0 = types.SimpleNamespace(width=self.W0, height=self.H0, bayer=None)
            fw = np.array([float(rows[k].get("fwhm") or np.nan) for k in self.keys], float) * float(scale)
            med = float(np.nanmedian(fw)) if np.isfinite(fw).any() else float(fwhm_guess)
            self.fwhm_rel = np.where(np.isfinite(fw), fw / med, 1.0)
            self.fwhm_guess = med if np.isfinite(med) and med > 0 else float(fwhm_guess)
            self.items = [(types.SimpleNamespace(name=os.path.basename(f), path=f), {"fwhm": float(self.fwhm_guess * q)})
                          for f, q in zip(self.files, self.fwhm_rel)]
            self.sat = 1e30
            self.sat_frames = None             # saturation in the frames' units (None: no saturated stars)
            self.coadd = None
            self.coverage = None
            self._acc = None
            self.workers = 4
            self.params = []
            self.ptc = None
            self.nodes = E.psf_nodes(self.H0, self.W0)
            self.fwhm_max = self.fwhm_med = self.fwhm_ref = self.fwhm_guess

        # -------------------------------------------------------- bookkeeping
        def say(self, i, n, msg):
            if self.progress:
                self.progress(i, n, msg)
            elif self.checkpoint:
                self.checkpoint()

        def _frame_paths(self, k):
            return (os.path.join(self.work, f"y_{k:05d}.npy"), os.path.join(self.work, f"f_{k:05d}.npy"))

        def _read(self, k: int, box=None) -> np.ndarray:
            y = read_frame(self.files[k], box or self.box, self.places[k])
            return np.ascontiguousarray(y[::-1]) if self.flip else y

        # -------------------------------------------------------- the pass
        def prepare(self, progress=None, cancel=None):
            """One pass over the frames (serial; one prepare at a time).  Per frame: masks, photometric scale
            T against the reference (aperture photometry of isolated stars), the sky surface, the PSF-star
            cut-outs and field PSF; the prepared frame goes to ``work`` for ``windows``.  Then the photon
            transfer curve, the PSF of the frames' own coadd, and the seeing-mixture kernels."""
            from astrophoto.exposures import poly_background
            n = len(self.files)
            os.makedirs(self.work, exist_ok=True)
            first = [self._read(k, self.sat_box)
                     for k in np.linspace(0, n - 1, min(n, 12)).round().astype(int)]
            self.sat_frames = saturation_level(first)
            del first
            self.say(0, n, "ImageMM: reference sky model and star catalogue")
            self.sky_ref, self.sky_info = poly_background(self.ref, 2)
            ref_bs = (self.ref - self.sky_ref).astype(np.float32)
            # saturation of the reference (a stack of the saturated plateaus): from the first frame's censored pixels
            sat_ref = 1e30
            if self.sat_frames is not None:
                m0 = measurement_masks(self._read(0), self.sat_frames)   # (the first frame, on this box)
                c0 = m0["cens"].all(-1)
                if c0.sum() >= 4:
                    sat_ref = float(np.median(self.ref[c0]))
                else:
                    sat_ref = float(np.percentile(self.ref, 99.99))
                del m0
            self.sat_ref = sat_ref
            self.fwhm_ref = self.fwhm_guess
            self.cat = E.star_catalog(ref_bs, sat_ref, self.fwhm_ref)
            fw = self.fwhm_ref * self.fwhm_rel
            self.fwhm_med = self.fwhm_ref
            self.fwhm_max = float(np.nanmax(fw))
            self.smask = E.star_mask(self.ref.shape[:2], self.cat, self.fwhm_ref)
            self.items = [(i, {"fwhm": float(q)}) for (i, _), q in zip(self.items, fw)]
            self.params = [None] * n
            S = np.zeros((self.H0, self.W0, 3), np.float32)
            Wt = np.zeros((self.H0, self.W0, 3), np.float32)
            acc = {"X": [], "v": [], "n": []}
            prev = None
            for k in range(n):
                if self.checkpoint:
                    self.checkpoint()
                self.say(k, n, f"ImageMM: preparing frames {k}/{n}")
                p, cur = self.prepare_frame(k)
                self.params[k] = p
                if cur is None:
                    prev = None
                    continue
                if prev is not None:
                    self._ptc_pair(prev, cur, acc)
                prev = cur
                v = cur["_valid"]
                S += cur["_yb"] * v
                Wt += v
            self.say(n, n, f"ImageMM: preparing frames {n}/{n}")
            seen = Wt > 0
            self.coadd = np.where(seen, S / np.maximum(Wt, 1), self.ref).astype(np.float32)
            self.coverage = Wt.mean(-1).astype(np.float32)
            del S, prev
            self.ptc = self._ptc_fit(acc)
            usable = [k for k in range(n) if self.params[k] and np.all(np.isfinite(self.params[k]["T"]))]
            if len(usable) < 3:
                raise RuntimeError(f"ImageMM: only {len(usable)} of {n} frames have a photometric scale in every "
                                   "channel (too few isolated stars against the stack)")
            if self.sat_frames is not None:
                # saturation in the latent units (frame / T, sky model off): what mm_restore's Whyte update takes
                Tm = float(np.median([self.params[k]["T"] for k in usable]))
                self.sat = float(self.sat_frames - Tm * float(np.median(self.sky_ref)))
            self.say(0, 1, "ImageMM: PSF of the frames' coadd")
            cbs = self.coadd - self.sky_ref
            self.psf_ref = [E.empirical_psf_field(cbs[..., c], seen[..., c], self.cat, int(math.ceil(3.5 * self.fwhm_max)),
                                                  self.fwhm_ref, self.nodes) for c in range(3)]
            for r in self.psf_ref:
                if r is not None:
                    r["source"] = "frames"
            self._fit_exposure_kernels(1, progress=self.say, cancel=self.checkpoint)
            return self

        def prepare_frame(self, k: int):
            from astrophoto.exposures import fit_smooth_surface
            E_ = E
            fwhm = float(self.items[k][1]["fwhm"])
            y = self._read(k)
            ms = measurement_masks(y, self.sat_frames)
            yf = fill_holes(y, ms)
            hard = (ms["region"] & ms["interior"]).all(-1)
            if ms["cens"].any():
                grown = cv2.dilate(ms["cens"].any(-1).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                hard &= ~grown
            T, T_err = E_.aperture_ratio(yf, self.ref, hard, self.cat, radius=3.0 * max(fwhm, self.fwhm_ref))
            base = {"refine": None, "residual": None, "T": T, "T_err": T_err, "fwhm": fwhm,
                    "valid_frac": float(ms["valid"].mean())}
            log.debug("frame %d: T %s +- %s, valid %.3f, holes %s, censored %.5f", k, T, T_err, ms["valid"].mean(),
                      ms["hole"].mean(axis=(0, 1)), ms["cens"].mean())
            if not np.all(np.isfinite(T)):
                return {**base, "surf": None, "psf": [None] * 3, "psf_field": [None] * 3, "psf_stars": [0, 0, 0]}, None
            e = yf / T[None, None, :] - self.ref
            coefs = fit_smooth_surface(e, hard & ~self.smask, deg=2)
            surf = np.moveaxis(surface_window(coefs, self.H0, self.W0, 0, self.H0, 0, self.W0), 0, -1)
            ybs = (yf / T[None, None, :] - surf - self.sky_ref).astype(np.float32)
            half = int(math.ceil(3.5 * max(fwhm, self.fwhm_ref)))
            fe = max(fwhm, self.fwhm_ref)
            fe_ = math.ceil(fe * 10 - 1e-9) / 10
            cuts = [E_._psf_cutouts(ybs[..., c], hard, self.cat, half, fe_, 1000) for c in range(3)]
            fields = [E_.empirical_psf_field(ybs[..., c], hard, self.cat, half, fe, self.nodes, cutouts=cu)
                      for c, cu in enumerate(cuts)]
            flags = (ms["valid"].astype(np.uint8) | (ms["cens"].astype(np.uint8) << 1))
            ypath, fpath = self._frame_paths(k)
            np.save(ypath, np.ascontiguousarray(np.moveaxis(ybs, -1, 0)))
            np.save(fpath, np.ascontiguousarray(np.moveaxis(flags, -1, 0)))
            p = {**base, "surf": coefs, "psf": [f["psf"] if f else None for f in fields], "psf_field": fields,
                 "psf_stars": [f["n_stars"] if f else 0 for f in fields],
                 "cuts": [E_._compact_cutouts(cu, 60) for cu in cuts]}
            cur = {"_y": np.where(ms["data"], y, 0).astype(np.float32), "_valid": ms["valid"], "_surf": surf, "T": T,
                   "_yb": (y / T[None, None, :] - surf).astype(np.float32)}
            return p, cur

        # -------------------------------------------------------- data for a window
        def window(self, k, y0, y1, x0, x1, censored=False):
            res = self.windows([k], y0, y1, x0, x1, censored)
            z = [np.moveaxis(r[0], 0, -1) for r in res]
            return tuple(z)

        def windows(self, idx, y0, y1, x0, x1, censored=False):
            """Stacked y, v, m (and c) of the chosen frames on a window: (n, 3, h, w) float32 each."""
            from concurrent.futures import ThreadPoolExecutor
            n, h, w = len(idx), y1 - y0, x1 - x0
            Y = np.empty((n, 3, h, w), np.float32)
            V = np.empty_like(Y)
            Mk = np.empty_like(Y)
            Cn = np.empty_like(Y) if censored else None
            refw = np.moveaxis(self.ref[y0:y1, x0:x1], -1, 0)
            c0, c1 = self.ptc["c0"][:, None, None], self.ptc["c1"][:, None, None]

            def one(a):
                k = idx[a]
                p = self.params[k]
                ypath, fpath = self._frame_paths(k)
                Y[a] = np.load(ypath, mmap_mode="r")[:, y0:y1, x0:x1]
                fl = np.load(fpath, mmap_mode="r")[:, y0:y1, x0:x1]
                Mk[a] = (fl & 1) > 0
                if censored:
                    Cn[a] = (fl & 2) > 0
                T = np.asarray(p["T"], np.float32)[:, None, None]
                surf = surface_window(p["surf"], self.H0, self.W0, y0, y1, x0, x1)
                level = np.maximum(T * (refw + surf), 0)               # the expected level in the frame's units
                V[a] = (c0 + c1 * level) / T ** 2
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                list(ex.map(one, range(n)))
            return (Y, V, Mk, Cn) if censored else (Y, V, Mk)

        def cleanup(self):
            shutil.rmtree(self.work, ignore_errors=True)

    return RegisteredSet


# --------------------------------------------------------------------------- output
def rescale_wcs(header, r: int, offset: tuple[int, int] = (0, 0)):
    """The header of an image whose pixel (x, y) = r * (old pixel - offset): latent sample j sits at the old
    pixel coordinate j / r (``imagemm.restore``), so CRPIX' = r (CRPIX - 1 - offset) + 1, the CD matrix (or
    CDELT) / r and the SIP terms (A_p_q ... of u, v relative to CRPIX) x r^(1 - p - q)."""
    import re
    h = header.copy()
    if "CRPIX1" in h and "CRPIX2" in h:
        h["CRPIX1"] = r * (float(h["CRPIX1"]) - 1 - offset[0]) + 1
        h["CRPIX2"] = r * (float(h["CRPIX2"]) - 1 - offset[1]) + 1
    for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2", "CDELT1", "CDELT2"):
        if k in h and r != 1:
            h[k] = float(h[k]) / r
    if r != 1:
        for k in list(h.keys()):
            m = re.fullmatch(r"(A|B|AP|BP)_(\d+)_(\d+)", k)
            if m:
                h[k] = float(h[k]) * r ** (1 - int(m.group(2)) - int(m.group(3)))
    return h


def _norm_peak(out: np.ndarray) -> float:
    """Scale that keeps the restored image in the [0, 1] the pipeline's linear files use: a restored star core can
    exceed the stack's ceiling (that is the point of Whyte et al.'s update)."""
    mx = float(out.max())
    if mx <= 1.0:
        return 1.0
    if mx > 16.0:
        mx = float(np.percentile(out[:, ::2, ::2], 99.9999))
    return 1.0 / max(mx, 1.0)


def run(session_dir: str, stack_meta: dict, stack_params: dict, progress=None, checkpoint=None) -> dict:
    """Restore the included registered frames of a stacked session with ImageMM -> ``imagemm.fits`` and
    ``imagemm_meta.json`` (returned).  Raises on failure; the caller falls back to the normal path."""
    av = available()
    if not av["ok"]:
        raise RuntimeError(av["reason"])
    import torch
    from astrophoto import imagemm
    from astrophoto.denoise import pick_device
    from . import workflow
    t0 = time.time()
    sp = dict(stack_params or {})
    kw = map_params(sp)
    r = kw.pop("r")
    say = progress or (lambda i, n, m: None)
    say(0, 1, "ImageMM: loading the stack and the registered frames")
    stack_path = os.path.join(session_dir, "stack.fits")
    stack, header = fitsio.read_rgb(stack_path)
    meta = dict(stack_meta or read_json(os.path.join(session_dir, "stack_meta.json"), {}) or {})
    crop = [int(v) for v in meta.get("crop") or [0, 0, stack.shape[2], stack.shape[1]]]
    if (crop[2], crop[3]) != (stack.shape[2], stack.shape[1]):
        raise RuntimeError(f"stack.fits is {stack.shape[2]} x {stack.shape[1]} but stack_meta.crop is {crop}")
    region = sp.get("imagemm_region")
    off, reg_win = (0, 0), None
    if region:                                  # development: restore a window of the stack (all of it is prepared)
        rx, ry, rw, rh = (int(v) for v in region)
        rx, ry = max(0, rx), max(0, ry)
        rw, rh = min(rw, crop[2] - rx), min(rh, crop[3] - ry)
        off, reg_win = (rx, ry), (ry, ry + rh, rx, rx + rw)
    scale = float(meta.get("scale") or 1.0)
    files, reg = ensure_registered(session_dir, meta, say, checkpoint)
    reg_info = {k: v for k, v in reg.items() if k != "placements"}
    rows = included_rows(session_dir)
    keys = sorted(files)
    maxf = int(sp.get("imagemm_max_frames") or 0)
    if maxf and len(keys) > maxf:
        keys = sorted(sorted(keys, key=lambda i: -(rows[i].get("weight") or 0.0))[:maxf])
    # orientation of stack.fits against the canvas: tested on the frame that covers most of the crop
    best = max(keys, key=lambda k: min(crop[0] + crop[2], reg["placements"][k][0] + reg["placements"][k][2])
               - max(crop[0], reg["placements"][k][0]))
    probe = read_frame(files[best], tuple(crop), reg["placements"][best])
    flip, flip_info = stack_is_flipped(np.moveaxis(stack, 0, -1), probe)
    del probe
    Reg = make_set_class()
    work = os.path.join(session_dir, WORK)
    shutil.rmtree(work, ignore_errors=True)
    try:
        es = Reg({k: files[k] for k in keys}, rows, tuple(crop), np.moveaxis(stack, 0, -1),
                 work, scale, float(meta.get("fwhm_px") or 4.0), places=reg["placements"], progress=say,
                 checkpoint=checkpoint, flip=flip)
        t1 = time.time()
        es.prepare()
        t_prep = time.time() - t1
        dev = pick_device(_device_arg(sp))
        psf_mix = [(None if p is None else p.get("sigma_median")) for p in es.psf_ref]
        t2 = time.time()
        lat, info = imagemm.restore(es, r=r, device=dev, progress=say, cancel=checkpoint, region=reg_win,
                                    kernel_cache=os.path.join(work, "kernels.pkl"), **kw)
        t_rest = time.time() - t2
        cov = info.pop("coverage")
        info.pop("residual", None)
        sky = np.moveaxis(es.sky_ref, -1, 0)
        if reg_win:
            sky = sky[:, reg_win[0]:reg_win[1], reg_win[2]:reg_win[3]]
        out = np.moveaxis(lat, -1, 0).astype(np.float32)
        if r == 1:
            out = out + sky.astype(np.float32)
        else:
            out = out + np.stack([cv2.resize(sky[c], (out.shape[2], out.shape[1]), interpolation=cv2.INTER_LINEAR)
                                  for c in range(3)]).astype(np.float32)
        if not np.isfinite(out).all():
            raise RuntimeError("ImageMM produced non-finite pixels")
        k_norm = _norm_peak(out)
        out = np.ascontiguousarray(out * np.float32(k_norm))
        usable = es.usable()
        tiles = info.get("tiles") or []
        iters = [t["iterations"] for t in tiles]
        hdr = rescale_wcs(header, r, off)
        hdr["HISTORY"] = "ImageMM restoration (arXiv:2501.03002) of the registered frames"
        fitsio.write_rgb(os.path.join(session_dir, OUT_FITS), out, hdr)
        meta_out = {
            "ok": True, "method": info.get("method"), "paper": "Sukurdeep, Budavari, Connolly & Navarro 2025, arXiv:2501.03002",
            "frames_used": len(usable), "frames_total": len(keys), "frames": [rows[keys[k]]["name"] for k in usable],
            "frames_skipped": [rows[keys[k]]["name"] for k in range(len(keys)) if k not in set(usable)],
            "r": r, "scale": scale * r, "crop": crop, "region_offset": list(off), "shape": [out.shape[2], out.shape[1]],
            "psf": {"measured_on": "the registered frames' own coadd (not stack.fits)", "model": kw["psf_model"],
                    "fwhm_px": round(float(es.fwhm_ref), 3), "fwhm_latent_px": round(float(info.get("fwhm") or 0), 3),
                    "size": info.get("psf_size"), "stars": [r_.get("n_stars") if r_ else None for r_ in es.psf_ref],
                    "seeing_sigma_median": psf_mix, "unit_sum": True, "eq11": info.get("eq11")},
            "noise": {"model": "photon transfer curve of consecutive frames, var = c0 + c1 * level per channel",
                      "c0": [float(v) for v in es.ptc["c0"]], "c1": [float(v) for v in es.ptc["c1"]],
                      "chi2_red": [float(v) for v in es.ptc["chi2_red"]]},
            "saturation": {"level_frames": es.sat_frames, "level_latent": float(es.sat) if es.sat_frames else None,
                           "whyte": bool(es.sat_frames), "saturated_pixels": info.get("saturated_pixels")},
            "params": {**{k: v for k, v in kw.items()}, "r": r},
            "iterations": {"min": min(iters) if iters else None, "median": float(np.median(iters)) if iters else None,
                           "max": max(iters) if iters else None, "cutouts": len(tiles),
                           "converged": int(sum(bool(t["converged"]) for t in tiles))},
            "stopping": ("Eq. C15: |mean(m~ xi_k) - 1| < epsilon" if kw["stop"] == "c15" else kw["stop"])
                        + f", epsilon {kw['epsilon']:g}, at most {kw['max_iters']} iterations",
            "sky": {"model": "degree-2 polynomial of the stack, subtracted before and added back after the restoration",
                    "info": {k: v for k, v in (es.sky_info or {}).items() if isinstance(v, (str, int, float, bool))}},
            "peak_scale": k_norm, "resampling": reg_info["resampling"], "frames_flipped": flip_info | {"flipped": bool(flip)}, "registered_again": reg_info["regenerated"],
            "device": str(dev), "torch": torch.__version__, "regridded": False,
            "seconds": round(time.time() - t0, 1), "seconds_prepare": round(t_prep, 1), "seconds_restore": round(t_rest, 1),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        write_json(os.path.join(session_dir, OUT_META), clean_json(meta_out))
        return clean_json(meta_out)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if not sp.get("imagemm_keep_registered"):                    # (development: keep them for the next run)
            workflow.clean_registered(os.path.join(session_dir, "process"))
