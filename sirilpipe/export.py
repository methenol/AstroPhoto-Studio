"""Writers of the exported picture: 8-bit sRGB JPEG, 16-bit TIFF and the JSON sidecar.

Names and shapes follow ``api-contract.md`` section 2 "Outputs": ``<session>/exports/<Object>_<YYYYmmdd-HHMMSS>.{jpg,tif,json}``
with ``<Object>`` = the stack meta's ``object`` without spaces (default ``image``); JPEG quality
``quality`` at 4:4:4 with EXIF ImageDescription ``"<obj> | <n> x subs, <min> min | AstroPhoto Studio <ver>"``
and Software; TIFF 16-bit RGB with zlib; the sidecar holds ``params`` (``{**DEFAULTS, **params}``),
``linear_info`` and ``meta`` (the stack meta) plus ``ignored_params``.
"""
from __future__ import annotations

import io
import json
import os
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

try:
    from . import __version__
except Exception:                                    # pragma: no cover
    __version__ = "2.0.0"


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def display_image(rgb: np.ndarray, bottom_up: bool = True) -> np.ndarray:
    """A pipeline image (3, H, W) -> display image (H, W, 3) float32 in [0, 1].  FITS files written by
    Siril keep the bottom row first (``ROWORDER = BOTTOM-UP``, also assumed when the keyword is missing),
    so the rows are flipped for viewing; ``TOP-DOWN`` files are left alone."""
    img = np.moveaxis(rgb, 0, -1)
    if bottom_up:
        img = img[::-1]
    return np.ascontiguousarray(np.clip(img, 0.0, 1.0), dtype=np.float32)


def bottom_up(header) -> bool:
    """True unless the header says ROWORDER = TOP-DOWN."""
    try:
        return str(header.get("ROWORDER", "BOTTOM-UP")).strip().upper() != "TOP-DOWN"
    except Exception:
        return True


def srgb_profile() -> bytes | None:
    """An sRGB ICC profile for the JPEG (the picture is display-referred sRGB), if lcms is available."""
    try:
        from PIL import ImageCms
        return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    except Exception:
        return None


def description(meta: dict) -> str:
    n = meta.get("n_frames") or meta.get("n_included") or meta.get("n_total")
    minutes = float(meta.get("total_exposure") or 0) / 60
    return f"{meta.get('object', '')} | {n} x subs, {minutes:.1f} min | AstroPhoto Studio {__version__}"


def jpeg_bytes(img: np.ndarray, quality: int = 90, max_side: int | None = None) -> bytes:
    """Encode an (H, W, 3) float image as JPEG bytes (the preview path), optionally downscaled."""
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, (max(1, int(round(img.shape[1] * s))), max(1, int(round(img.shape[0] * s)))),
                         interpolation=cv2.INTER_AREA)
    buf = io.BytesIO()
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(buf, "JPEG", quality=int(quality))
    return buf.getvalue()


def write_outputs(session_dir: str, img: np.ndarray, *, meta: dict, params_full: dict, linear_info: dict,
                  quality: int = 95, upscale: float = 1.0, tiff: bool = True, ignored: list[str] | None = None) -> dict:
    """Write jpg (+tif) and the sidecar for the display image ``img`` (H, W, 3 float in [0, 1]).

    ``upscale`` > 1 resizes the final image with Lanczos first.  Returns
    ``{"jpg", "tif"?, "json", "size": [w, h]}`` (absolute paths)."""
    if upscale and float(upscale) > 1:
        img = np.clip(cv2.resize(img, None, fx=float(upscale), fy=float(upscale), interpolation=cv2.INTER_LANCZOS4), 0, 1)
        linear_info = {**linear_info, "upscaled": float(upscale)}
    out_dir = os.path.join(session_dir, "exports")
    os.makedirs(out_dir, exist_ok=True)
    obj = (meta.get("object") or "image").replace(" ", "")
    base = os.path.join(out_dir, f"{obj}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    desc = description(meta)
    im8 = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
    exif = Image.Exif()
    exif[0x010E] = desc                                   # ImageDescription
    exif[0x0131] = f"AstroPhoto Studio {__version__}"     # Software
    kw = {}
    icc = srgb_profile()
    if icc:
        kw["icc_profile"] = icc
    im8.save(base + ".jpg", quality=int(quality), subsampling=0, optimize=True, exif=exif, **kw)
    files: dict = {"jpg": base + ".jpg"}
    if tiff:
        import tifffile
        tifffile.imwrite(base + ".tif", (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16), photometric="rgb",
                         compression="zlib", description=desc)
        files["tif"] = base + ".tif"
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({"params": params_full, "linear_info": linear_info, "meta": meta, "ignored_params": ignored or []},
                  f, indent=1, default=_json_default)
    files["json"] = base + ".json"
    files["size"] = [int(img.shape[1]), int(img.shape[0])]
    return files
