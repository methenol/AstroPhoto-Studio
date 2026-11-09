"""One mosaic project, one image.

The manifest (``stack_params["mosaic"]``, written by the telescope controller or any client of ``POST /api/v1/mosaics``) says what
was shot: the sky footprint, and for each panel where it was aimed and which session folders hold its subs.

    panels   individually aimed panels: each is calibrated, graded, stacked and plate-solved on its own in
             ``<session>/panels/<id>/`` (kept, and rebuilt only when its subs change), then the linear stacks are
             projected on one canvas by Siril (``seqplatesolve``, ``seqapplyreg -framing=max``, ``stack -maximize``).
    native   the scope's own mosaic: one set of subs whose field changes; they are plate-solved one by one and integrated on one canvas.

The result is the session's ``stack.fits`` + ``stack_meta.json``, so the finishing chain is the same as for any other image, on the linear
mosaic: nothing is stretched before the seams are gone.  ``mosaic_report.json`` is the feedback to the scheduler: per panel the integration
the stack accepted, where it really landed, and which panels must be shot again because the footprint has a hole there.
"""
from __future__ import annotations

import glob
import hashlib
import json
import logging
import math
import os
import shutil
import time

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from . import fitsio, workflow
from .siril import SirilError, SirilRunner
from .tools import gaia_astro
from .util import Cancelled, read_json, write_json

log = logging.getLogger(__name__)

MAX_SIDE = 32768                  # Siril 1.4 cannot handle an image wider or taller than this
GRID = 81                         # samples per side of the footprint when measuring its coverage
COVERED_ENOUGH = 99.0             # % of the footprint: below this the panels next to the holes are shot again
PANEL_SP_KEYS = ("mode", "sensitivity", "sigma_low", "sigma_high", "rejection", "stack_weighting", "local_norm", "pattern_correction",
                 "drizzle_pixfrac", "drizzle_kernel", "crop", "crop_threshold")


# ---------------------------------------------------------------------- manifest

def normalise_manifest(m: dict) -> dict:
    """The manifest with every field checked; raises ValueError with what is wrong."""
    if not isinstance(m, dict):
        raise ValueError("the mosaic manifest must be an object")
    pid = str(m.get("project_id") or "").strip()
    if not pid:
        raise ValueError("the manifest has no project_id")
    execution = m.get("execution") or "panels"
    if execution not in ("panels", "native"):
        raise ValueError(f"execution must be panels or native, not {execution!r}")
    panels = m.get("panels")
    if not isinstance(panels, list) or not panels:
        raise ValueError("the manifest has no panels")
    if execution == "native" and len(panels) != 1:
        raise ValueError("a native mosaic is one set of subs: exactly one panel")
    fp = m.get("footprint") or {}
    try:
        footprint = {"ra_deg": float(fp["ra_deg"]) % 360.0, "dec_deg": float(fp["dec_deg"]), "width_arcmin": float(fp["width_arcmin"]),
                     "height_arcmin": float(fp["height_arcmin"]), "angle_deg": float(fp.get("angle_deg") or 0.0)}
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"bad footprint: {e!r}") from None
    if footprint["width_arcmin"] <= 0 or footprint["height_arcmin"] <= 0 or abs(footprint["dec_deg"]) > 90:
        raise ValueError("the footprint needs a positive size and a valid declination")
    out, seen = [], set()
    for p in panels:
        i = str((p or {}).get("id") or "").strip()
        if not i or i in seen or not i.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"bad or repeated panel id {i!r}")
        seen.add(i)
        out.append({"id": i, "ra_deg": float(p["ra_deg"]) % 360.0, "dec_deg": float(p["dec_deg"]), "folders": list(p.get("folders") or [])})
    return {"project_id": pid, "target": str(m.get("target") or pid), "execution": execution,
            "overlap": float(m.get("overlap") or 0.2), "footprint": footprint, "panels": out}


def finishing_params(sp: dict) -> dict:
    """The stack parameters for the chain after the assembly.  ImageMM restores from the registered subs of one stack, which a mosaic does not have."""
    out = {k: v for k, v in sp.items() if k not in ("mosaic", "roi_crop_percent", "hypertune")}
    for k in ("deconv_method", "deconv_stars_method"):
        if out.get(k) == "imagemm":
            out[k] = "auto"
    return out


# ---------------------------------------------------------------------- sky geometry

def tangent_to_sky(ra0: float, dec0: float, east_deg, north_deg):
    """Gnomonic projection back to the sky: offsets (degrees) on the tangent plane at (ra0, dec0) -> (ra, dec) degrees."""
    xi, eta, d0 = np.radians(east_deg), np.radians(north_deg), math.radians(dec0)
    den = math.cos(d0) - eta * math.sin(d0)
    ra = ra0 + np.degrees(np.arctan2(xi, den))
    dec = np.degrees(np.arctan2(math.sin(d0) + eta * math.cos(d0), np.hypot(xi, den)))
    return ra % 360.0, dec


def separation_arcmin(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    a1, d1, a2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    h = math.sin((d2 - d1) / 2) ** 2 + math.cos(d1) * math.cos(d2) * math.sin((a2 - a1) / 2) ** 2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(h)))) * 60.0


def footprint_samples(fp: dict, n: int = GRID):
    """A grid of sky positions over the footprint's rectangle: (ra, dec), degrees."""
    u = np.linspace(-0.5, 0.5, n) * fp["width_arcmin"] / 60.0
    v = np.linspace(-0.5, 0.5, n) * fp["height_arcmin"] / 60.0
    uu, vv = np.meshgrid(u, v)
    a = math.radians(fp["angle_deg"])
    return tangent_to_sky(fp["ra_deg"], fp["dec_deg"], uu * math.cos(a) - vv * math.sin(a), uu * math.sin(a) + vv * math.cos(a))


def solved_wcs(header) -> WCS | None:
    try:
        w = WCS(header, naxis=2)
        return w if w.has_celestial and "RA" in (w.wcs.ctype[0] or "") else None
    except Exception:                                                   # noqa: BLE001  a header without a usable solution
        return None


def _inside(w: WCS, shape: tuple[int, int], ra, dec) -> np.ndarray:
    x, y = w.all_world2pix(ra, dec, 0, quiet=True) if w.sip is not None else w.wcs_world2pix(ra, dec, 0)
    with np.errstate(invalid="ignore"):
        return (x >= -0.5) & (x < shape[1] - 0.5) & (y >= -0.5) & (y < shape[0] - 0.5)     # a pixel spans its centre +/- 0.5


def measure_coverage(manifest: dict, solved: dict[str, tuple[WCS, tuple[int, int]]]) -> dict:
    """How much of the footprint the solved panels really cover, and which planned panels are nearest to what is missing."""
    ra, dec = footprint_samples(manifest["footprint"])
    covered = np.zeros(ra.shape, bool)
    for w, shape in solved.values():
        covered |= _inside(w, shape, ra, dec)
    pct = float(covered.mean() * 100.0)
    revisit: list[str] = []
    if pct < COVERED_ENOUGH:
        cd = np.cos(np.radians(dec[~covered]))
        dist = np.stack([np.hypot(((ra[~covered] - p["ra_deg"] + 180) % 360 - 180) * cd, dec[~covered] - p["dec_deg"]) for p in manifest["panels"]])
        revisit = sorted({manifest["panels"][i]["id"] for i in np.unique(dist.argmin(0))})
    return {"percent": round(pct, 1), "revisit": revisit}


def canvas_estimate(manifest: dict, solved: dict[str, tuple[WCS, tuple[int, int]]]) -> dict:
    """Output size of the mosaic from the solved fields, before anything heavy is started: (w, h) px and the memory of one float32 RGB canvas."""
    fp = manifest["footprint"]
    ref = WCS(naxis=2)
    ref.wcs.crval, ref.wcs.crpix, ref.wcs.ctype = [fp["ra_deg"], fp["dec_deg"]], [0.0, 0.0], ["RA---TAN", "DEC--TAN"]
    scale = min(float(np.sqrt(np.abs(np.linalg.det(w.pixel_scale_matrix)))) for w, _ in solved.values())
    ref.wcs.cdelt = [scale, scale]
    xs, ys = [], []
    for w, (h, wd) in solved.values():
        ra, dec = w.wcs_pix2world([0, wd - 1, wd - 1, 0], [0, 0, h - 1, h - 1], 0)
        x, y = ref.wcs_world2pix(ra, dec, 0)
        xs += list(x)
        ys += list(y)
    w_px, h_px = int(math.ceil(round(max(xs) - min(xs), 3))) + 1, int(math.ceil(round(max(ys) - min(ys), 3))) + 1
    return {"width": w_px, "height": h_px, "gigabytes": round(w_px * h_px * 3 * 4 / 1e9, 2), "pixel_scale": round(scale * 3600.0, 3)}


def feather_px(manifest: dict, sample_width: int) -> int:
    """Seams blend over about half the overlap of two panels: wide enough to hide them, inside the part both panels have data for."""
    return max(16, int(round(sample_width * max(manifest["overlap"], 0.05) / 2.0)))


# ---------------------------------------------------------------------- the panels

def _fits_files(folders: list[str]) -> list[str]:
    out = {f for d in folders for pat in ("*.fit", "*.fits", "*.fts", "*.FIT", "*.FITS", "*.FTS") for f in glob.glob(os.path.join(d, pat))}
    return sorted(out)


def panel_signature(folders: list[str], sp: dict) -> str:
    """What makes a panel's stack stale: its subs and the settings of the stacking.  A night added to one panel leaves the others alone."""
    files = [(os.path.basename(f), os.path.getsize(f)) for f in _fits_files(folders)]
    return hashlib.sha256(json.dumps([files, {k: sp.get(k) for k in PANEL_SP_KEYS}], sort_keys=True, default=str).encode()).hexdigest()


def panel_stack_params(sp: dict) -> dict:
    """Panels are integrated at the sensor's own resolution (a drizzled mosaic would not fit); the finishing happens on the assembled image."""
    return {**finishing_params(sp), "scale": 1.0, "superres_method": "drizzle", "autofinish": False, "star_remover": False}


def _panel_report(panel: dict, meta: dict, stack: str, reused: bool) -> tuple[dict, tuple | None]:
    hdr = fits.getheader(stack, 0)
    wcs = solved_wcs(hdr) if (meta.get("plate_solve") or {}).get("ok") else None
    exp = float(meta.get("exptime") or 0.0)
    rep = {"id": panel["id"], "status": "stacked" if wcs else "unsolved", "reused": reused,
           "accepted_sec": round(float(meta.get("total_exposure") or 0.0), 1), "input_sec": round(float(meta.get("n_total") or 0) * exp, 1),
           "frames_accepted": meta.get("n_frames"), "frames_total": meta.get("n_total"), "solved": bool(wcs)}
    if not wcs:
        return rep, None
    shape = (int(hdr["NAXIS2"]), int(hdr["NAXIS1"]))
    ra, dec = wcs.wcs_pix2world([(shape[1] - 1) / 2.0], [(shape[0] - 1) / 2.0], 0)
    rep.update(ra_deg=round(float(ra[0]), 5), dec_deg=round(float(dec[0]), 5),
               offset_arcmin=round(separation_arcmin(float(ra[0]), float(dec[0]), panel["ra_deg"], panel["dec_deg"]), 2))
    return rep, (wcs, shape)


def stack_panel(s, panel: dict, sp: dict, progress, label: str, session_cls) -> tuple[dict, tuple | None, str | None, dict | None]:
    """A panel's linear, plate-solved stack (kept from the last run when nothing changed)."""
    def prog(i, n, msg):
        if progress:
            progress(i, n, f"{label}: {msg}")
    work = os.path.join(s.dir, "panels", panel["id"])
    os.makedirs(work, exist_ok=True)
    ps = session_cls(panel["folders"], work)
    sig = panel_signature(panel["folders"], sp)
    sig_file = ps._p("panel_signature.json")
    stack = ps._p("stack.fits")
    if read_json(sig_file, {}).get("signature") == sig and os.path.exists(stack) and (ps.meta.get("plate_solve") or {}).get("ok"):
        rep, sol = _panel_report(panel, ps.meta, stack, True)
        return rep, sol, stack, ps.meta
    try:
        ps.run_analysis(float(sp.get("sensitivity", 1.0) or 1.0), prog, sp)
        ps.run_stack(sp, prog)
        ps.plate_solve(prog)
    except Cancelled:
        raise
    except Exception as e:                                              # noqa: BLE001  one bad panel is a hole to shoot again, not a lost mosaic
        msg = (str(e).splitlines() or [repr(e)])[0][:300]
        log.warning("mosaic panel %s failed: %s", panel["id"], msg)
        if os.path.exists(sig_file):
            os.remove(sig_file)
        return {"id": panel["id"], "status": "failed", "error": msg, "reused": False, "accepted_sec": 0.0, "input_sec": 0.0, "solved": False}, None, None, None
    write_json(sig_file, {"signature": sig, "created": time.time()})
    meta = ps.meta
    rep, sol = _panel_report(panel, meta, stack, False)
    return rep, sol, stack, meta


# ---------------------------------------------------------------------- the assembly

def _runner(s, progress):
    return SirilRunner(s.process, progress=s._cb(progress), checkpoint=s.checkpoint)


def _clean(process: str) -> None:
    shutil.rmtree(process, ignore_errors=True)
    os.makedirs(process, exist_ok=True)


def assemble_panels(s, manifest: dict, stacks: list[str], feather: int, progress) -> tuple[np.ndarray, fits.Header]:
    _clean(s.process)
    link = os.path.join(s.process, "_link")
    os.makedirs(link)
    for i, f in enumerate(stacks, 1):
        os.symlink(f, os.path.join(link, f"panel_{i:05d}.fit"))
    local = gaia_astro.ensure(s._cb(progress), s.checkpoint)            # the local catalogue first, the online one when that fails (Session.plate_solve)
    for lines in ([gaia_astro.siril_lines()] if local else []) + [None]:
        try:
            _runner(s, progress).run(workflow.mosaic_panels_script(s.process, "panel", feather, len(stacks), lines), "mosaic", expect=["mosaic_raw.fit"],
                                     label="Assembling the mosaic")
            break
        except SirilError:
            if lines is None:
                raise
            for stale in glob.glob(os.path.join(s.process, "r_panel*")) + glob.glob(os.path.join(s.process, "panel_*")):
                os.remove(stale)
    return fitsio.read_rgb(os.path.join(s.process, "mosaic_raw.fit"))


def stack_native(s, manifest: dict, sp: dict, progress) -> tuple[np.ndarray, fits.Header, dict]:
    """Calibrate (debayered: the frames are projected with their solutions, not drizzled), solve every sub, integrate on one canvas."""
    if s.fs is None:
        s.scan(progress)
    units = s._make_units(True, False)
    s._ensure_masters(units, progress)
    workflow.clean_process_dir(s.process)                               # (keeps what _ensure_masters just made)
    n = sum(len(u.frames) for u in units)
    runner = _runner(s, progress)
    seq = workflow.seq_name(units, True)
    for u in units:
        workflow.stage_inputs(u, [s._path(f) for f in u.frames])
        runner.run(workflow.calibrate_script(u, True, force=True), f"calibrate_{u.index + 1}", expect=[os.path.join(u.dir, f"{seq}_00001.fit")], label="Calibrating")
    if len(units) > 1:
        runner.run(workflow.merge_script(units, s.process, seq), "merge", expect=[f"{seq}_.seq"], label="Calibrating (merging nights)")
    local = gaia_astro.ensure(s._cb(progress), s.checkpoint)
    catalog = gaia_astro.siril_lines() if local else None
    feather = feather_px(manifest, s.fs.first.width)
    runner.run(workflow.mosaic_subs_script(s.process, seq, feather, s.fs.first, catalog), "mosaic", expect=["mosaic_raw.fit"], label="Solving and integrating the mosaic's subs")
    arr, hdr = fitsio.read_rgb(os.path.join(s.process, "mosaic_raw.fit"))
    return arr, hdr, {"n": n, "exptime": float(s.fs.first.exptime or 0.0), "feather": feather}


def _write_stack(s, arr: np.ndarray, hdr, manifest: dict, base: dict, panels: list[dict], progress) -> dict:
    """The mosaic as the session's stack: the finishing chain reads it like any other."""
    H, W = arr.shape[1:]
    fitsio.write_rgb(s._p("stack.fits"), arr, hdr)
    meta = {**base, "mode": "mosaic", "scale": 1.0, "drizzle": None, "crop": [0, 0, W, H], "crop_info": {"method": "mosaic"}, "shape": [W, H], "stack_canvas": [W, H],
            "n_frames": sum(int(p.get("frames_accepted") or 0) for p in panels), "n_total": sum(int(p.get("frames_total") or 0) for p in panels),
            "total_exposure": sum(float(p.get("accepted_sec") or 0.0) for p in panels), "object": manifest["target"], "saturation": float((arr[:, ::4, ::4] >= 0.999).mean()),
            "plate_solve": None, "params": {"mosaic": manifest["project_id"]}, "mosaic": {"project_id": manifest["project_id"], "execution": manifest["execution"]}}
    meta.pop("roi", None)
    write_json(s._p("stack_meta.json"), meta)
    s._invalidate_after_stack()
    wcs = solved_wcs(hdr)
    if wcs is not None:                                                 # Siril carried the projection's solution through the stack
        meta.update(plate_solve={"ok": True, "source": "panels"}, wcs=True)
        write_json(s._p("stack_meta.json"), meta)
    else:
        s.plate_solve(progress)
    return s.meta


def assemble(s, sp: dict, progress) -> dict:
    """Build the session's ``stack.fits`` from the mosaic manifest; returns the report (also ``mosaic_report.json``)."""
    from .session import Session
    manifest = normalise_manifest(sp.get("mosaic"))
    t0 = time.time()
    if manifest["execution"] == "native":
        arr, hdr, native = stack_native(s, manifest, sp, progress)
        accepted = native["n"] * native["exptime"]
        panel = manifest["panels"][0]
        reports = [{"id": panel["id"], "status": "stacked", "reused": False, "accepted_sec": round(accepted, 1), "input_sec": round(accepted, 1),
                    "frames_accepted": native["n"], "frames_total": native["n"], "solved": True}]
        w = solved_wcs(hdr)
        solved = {panel["id"]: (w, arr.shape[1:])} if w is not None else {}
        base = {"exptime": native["exptime"], "pixel_scale": s.fs.pixel_scale, "fwhm_px": None, "filter": s.fs.filter, "is_dualband": s.fs.is_dualband,
                "kind": s.fs.kind, "bayer": s.fs.bayer, "sensor": s.fs.sensor, "gain": s.fs.first.gain}
    else:
        psp = panel_stack_params(sp)
        reports, solved, stacks, metas = [], {}, [], []
        total = len(manifest["panels"])
        for i, p in enumerate(manifest["panels"], 1):
            if not p["folders"]:
                reports.append({"id": p["id"], "status": "missing", "reused": False, "accepted_sec": 0.0, "input_sec": 0.0, "solved": False})
                continue
            rep, sol, stack, meta = stack_panel(s, p, psp, progress, f"Panel {p['id']} ({i}/{total})", Session)
            reports.append(rep)
            if sol:
                solved[p["id"]] = sol
                stacks.append(stack)
                metas.append(meta)
        if len(stacks) < 2:
            raise RuntimeError(f"only {len(stacks)} of {total} panels could be stacked and plate-solved; a mosaic needs two ("
                               + "; ".join(f"{r['id']}: {r.get('error') or r['status']}" for r in reports if r["status"] != "stacked") + ")")
        est = canvas_estimate(manifest, solved)
        if max(est["width"], est["height"]) > MAX_SIDE:
            raise RuntimeError(f"the mosaic would be {est['width']} x {est['height']} px, over Siril's limit of {MAX_SIDE} px per side")
        log.info("mosaic %s: canvas %d x %d px, %.1f GB per RGB canvas", manifest["project_id"], est["width"], est["height"], est["gigabytes"])
        feather = feather_px(manifest, solved[next(iter(solved))][1][1])
        arr, hdr = assemble_panels(s, manifest, stacks, feather, progress)
        m0 = metas[0]
        fwhm = [m["fwhm_px"] for m in metas if m.get("fwhm_px")]
        base = {k: m0.get(k) for k in ("exptime", "gain", "filter", "is_dualband", "kind", "bayer", "sensor", "pixel_scale", "calibration", "siril_version", "sirilpipe_version", "normalisation")}
        base["fwhm_px"] = float(np.median(fwhm)) if fwhm else None
    _write_stack(s, arr, hdr, manifest, base, reports, progress)
    coverage = measure_coverage(manifest, solved) if solved else {"percent": 0.0, "revisit": [p["id"] for p in manifest["panels"]]}
    for r in reports:
        if r["status"] != "stacked" and r["id"] not in coverage["revisit"]:
            coverage["revisit"].append(r["id"])
    H, W = arr.shape[1:]
    report = {"project_id": manifest["project_id"], "target": manifest["target"], "execution": manifest["execution"], "status": "assembled",
              "panels": reports, "coverage": coverage, "canvas": {"width": W, "height": H}, "seconds": round(time.time() - t0, 1)}
    write_json(s._p("mosaic_report.json"), report)
    return report
