"""Tiny FITS helpers shared by the stack/post modules.

Convention for every image the pipeline keeps (stack.fits, linear.fits, starless.fits, ...):
float32, shape ``(3, H, W)`` (planes R, G, B), rows in Siril's order (the order in which Siril
itself wrote them: row 0 of the array is the first row stored in the file, Siril's bottom-up
convention; ROWORDER is kept in the header), values in [0, 1] for linear data.  Keeping the
file's own row order means Siril can load what we write (and the WCS written by ``platesolve``
stays valid) without any flipping on our side.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
from astropy.io import fits

# Keys that describe the storage of the array, not the image; astropy regenerates them.
_STRUCTURAL = ("SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "EXTEND", "BZERO", "BSCALE",
               "BLANK", "PCOUNT", "GCOUNT")


def read_fits(path: str) -> tuple[np.ndarray, fits.Header]:
    """Read the primary HDU as float32 (any dimensionality) with its header."""
    with fits.open(path, memmap=False) as hdul:
        hdu = hdul[0] if hdul[0].data is not None else next(h for h in hdul if h.data is not None)
        return np.asarray(hdu.data, dtype=np.float32), hdu.header.copy()


def read_rgb(path: str) -> tuple[np.ndarray, fits.Header]:
    """Read an image as ``(3, H, W)`` float32 plus header.  A mono image is replicated to 3 planes
    (so callers can treat every pipeline image alike); use ``read_fits`` to keep it mono."""
    data, header = read_fits(path)
    if data.ndim == 2:
        data = np.repeat(data[None], 3, axis=0)
    elif data.ndim == 3 and data.shape[0] != 3 and data.shape[-1] == 3:
        data = np.moveaxis(data, -1, 0)           # (H, W, 3) written by other tools
    if data.ndim != 3 or data.shape[0] != 3:
        raise ValueError(f"{path}: expected a (3, H, W) or (H, W) image, got {data.shape}")
    return np.ascontiguousarray(data), header


def write_rgb(path: str, arr: np.ndarray, header: fits.Header | dict | None = None) -> None:
    """Write ``arr`` (``(3, H, W)`` or ``(H, W)``) as float32 FITS, atomically.

    Why atomic: the web UI polls for these files (step ticks come from file existence), so a
    half-written stack must never be visible under its final name."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 3 and arr.shape[0] != 3:
        arr = np.moveaxis(arr, -1, 0)
    hdr = fits.Header()
    if header is not None:
        for card in (header.cards if isinstance(header, fits.Header) else []):
            if card.keyword in _STRUCTURAL or card.keyword in ("COMMENT", "HISTORY", ""):
                if card.keyword in ("COMMENT", "HISTORY"):
                    hdr.add_history(str(card.value)) if card.keyword == "HISTORY" else hdr.add_comment(str(card.value))
                continue
            try:
                hdr[card.keyword] = (card.value, card.comment)
            except Exception:
                continue
        if isinstance(header, dict):
            for k, v in header.items():
                hdr[k] = v
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".fits", dir=os.path.dirname(os.path.abspath(path)) or ".")
    os.close(fd)
    try:
        fits.PrimaryHDU(arr, header=hdr).writeto(tmp, overwrite=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def update_header(path: str, **cards) -> None:
    """Set header cards in place (no data rewrite)."""
    with fits.open(path, mode="update") as hdul:
        for k, v in cards.items():
            hdul[0].header[k] = v


def header_dict(header: fits.Header, keys: tuple[str, ...] | None = None) -> dict:
    """The scalar cards of a header as a plain dict (JSON-safe)."""
    out = {}
    for k in header:
        if k in ("COMMENT", "HISTORY", "") or (keys and k not in keys):
            continue
        v = header[k]
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
    return out


_WCS_KEYS = ("CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CDELT1", "CDELT2",
             "CD1_1", "CD1_2", "CD2_1", "CD2_2", "PC1_1", "PC1_2", "PC2_1", "PC2_2", "LONPOLE", "LATPOLE", "EQUINOX",
             "PLTSOLVD", "OBJCTRA", "OBJCTDEC")
_WCS_PREFIX = ("A_", "B_", "AP_", "BP_")


def strip_wcs(header: fits.Header) -> fits.Header:
    """A copy of ``header`` without any plate solution (RA/DEC pointing cards stay).  Registration and cropping
    change the geometry, so a WCS inherited from a sub (ASIAIR writes one) would be wrong."""
    h = header.copy()
    for k in list(h.keys()):
        if k in _WCS_KEYS or k.startswith(_WCS_PREFIX):
            del h[k]
    return h
