"""Spectrophotometric colour calibration (SPCC), after Siril's
(https://siril.readthedocs.io/en/stable/processing/color-calibration/spcc.html).

The colour of every matched star is predicted from physics instead of assuming the average
star is white:

1. The stack's plate solution (Explore: Gaia DR3 via VizieR, ``astrometry.py``) gives every
   Gaia star in the frame with its GSP-Phot effective temperature and extinction A_G.
2. Each star's spectrum is a Pickles (1998, PASP 110, 863) library spectrum of its temperature
   (dwarf or giant from its absolute magnitude), reddened by its own extinction with the
   Cardelli, Clayton & Mathis (1989) law (R_V = 3.1).
3. The spectrum is integrated (photon counts) through the camera's R/G/B quantum efficiency
   and the filter's transmission, which predicts the star's colour in *this* camera.
4. Aperture photometry of the same stars in the image gives their measured colours; the
   per-channel gains are the robust (sigma-clipped median) ratio of predicted to measured,
   relative to a white reference (the average spiral galaxy, as Siril's default, or a G2V
   star), so that the white reference renders neutral.

Siril uses each star's Gaia XP spectrum in step 2; XP spectra are only served per-star by
the ESA archive's datalink (minutes per few hundred stars), while temperature and extinction
come with the catalogue query Explore already makes.

Sensor, filter and reference curves come from Siril's SPCC database
(https://gitlab.com/free-astro/siril-spcc-database, GPL-3.0), downloaded on first use and
cached, never bundled.  The camera and filter are detected from the FITS headers (INSTRUME,
sensor geometry, FILTER) or chosen explicitly.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request

import numpy as np

DB_PROJECT = "free-astro/siril-spcc-database"
DB_REF = "main"
DB_RAW = "https://gitlab.com/" + DB_PROJECT + "/-/raw/" + DB_REF + "/{path}"
DB_TREE = ("https://gitlab.com/api/v4/projects/" + urllib.parse.quote(DB_PROJECT, safe="") +
           "/repository/tree?path={path}&ref=" + DB_REF + "&per_page=100&page={page}")
INDEX_MAX_AGE = 30 * 86400
WL = np.arange(300.0, 1101.0, 1.0)          # integration grid, nm
AV_GRID = np.arange(0.0, 8.01, 0.25)         # extinction grid, mag
AG_PER_AV = 0.789                            # A_G / A_V for a typical star (Gaia DR3 GSP-Phot, Danielski+2018)

WHITE_REFS = {"average_spiral_galaxy": "Average_spiral_galaxy", "g2v": "Star_g2v"}

# Pickles templates and their effective temperatures (Pecaut & Mamajek 2013 for dwarfs; giants from
# the same table's giant sequence).  File names are wb_refs/Star_<code>.json.
PICKLES = {
    "dwarf": [("o5v", 41400), ("o9v", 31500), ("b0v", 29000), ("b1v", 26000), ("b3v", 17000), ("b57v", 14500),
              ("b8v", 12500), ("b9v", 10700), ("a0v", 9700), ("a2v", 9300), ("a3v", 8800), ("a5v", 8100),
              ("a7v", 7650), ("f0v", 7220), ("f2v", 6810), ("f5v", 6510), ("f6v", 6340), ("f8v", 6170),
              ("g0v", 5920), ("g2v", 5770), ("g5v", 5660), ("g8v", 5490), ("k0v", 5280), ("k2v", 5040),
              ("k3v", 4830), ("k4v", 4600), ("k5v", 4410), ("k7v", 4070), ("m0v", 3850), ("m1v", 3660),
              ("m2v", 3560), ("m3v", 3430), ("m4v", 3210), ("m5v", 3060), ("m6v", 2810)],
    "giant": [("g5iii", 5010), ("g8iii", 4940), ("k0iii", 4810), ("k1iii", 4610), ("k2iii", 4500), ("k3iii", 4320),
              ("k4iii", 4080), ("k5iii", 3980), ("m0iii", 3850), ("m1iii", 3720), ("m2iii", 3620), ("m3iii", 3530),
              ("m4iii", 3420), ("m5iii", 3330), ("m6iii", 3200)],
}

# Sensor geometry (width, height, pixel size um) -> sensor, for headers that do not name the camera.
# Several sensors share a geometry (IMX462 / IMX662); those are resolved by the camera name only.
SENSOR_GEOMETRY = [
    (3840, 2160, 2.9, "Sony_IMX585"), (1920, 1080, 2.9, "Sony_IMX462"), (1920, 1080, 2.9, "Sony_IMX662"),
    (6248, 4176, 3.76, "Sony_IMX571"), (9576, 6388, 3.76, "Sony_IMX455"), (3008, 3008, 3.76, "Sony_IMX533"),
    (4144, 2822, 4.63, "Sony_IMX294"), (5496, 3672, 2.4, "Sony_IMX183"), (3096, 2080, 2.4, "Sony_IMX178"),
    (1304, 976, 3.75, "Sony_IMX224"), (1936, 1096, 3.75, "Sony_IMX385"), (4944, 3284, 4.78, "Sony_IMX071"),
    (6024, 4024, 5.94, "Sony_IMX410"), (1920, 1080, 5.8, "Sony_IMX482"), (3840, 2160, 2.0, "Sony_IMX678"),
    (3552, 3552, 2.0, "Sony_IMX676"), (3840, 2160, 1.45, "Sony_IMX715"), (4056, 3040, 1.55, "Sony_IMX477"),
    (3864, 2192, 1.45, "Sony_IMX415"),
]
# camera model numbers that differ from the sensor's
CAMERA_ALIASES = {"2600": "571", "6200": "455", "2400": "410", "268": "571", "600": "455", "410": "410",
                  "224": "224", "662": "662", "678": "678", "715": "715"}


# ============================================================ database
def _get(url: str, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "AstroPhoto-Studio"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def list_curves(kind: str, cache_dir: str) -> list[str]:
    """Names (file stems) in one folder of the database: osc_sensors, osc_filters or wb_refs.
    Cached for a month; an offline machine keeps using the cached index, or gets []."""
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, f"index_{kind}.json")
    if os.path.exists(p) and time.time() - os.path.getmtime(p) < INDEX_MAX_AGE:
        return json.load(open(p))
    try:
        names, page = [], 1
        while True:
            items = json.loads(_get(DB_TREE.format(path=kind, page=page)))
            names += [it["name"][:-5] for it in items if it["type"] == "blob" and it["name"].endswith(".json")]
            if len(items) < 100:
                break
            page += 1
        json.dump(sorted(names), open(p, "w"))
        return sorted(names)
    except Exception:
        return json.load(open(p)) if os.path.exists(p) else []


def load_curve(kind: str, name: str, cache_dir: str) -> dict:
    """One database file -> {channel: (wavelength nm, value 0..1)}; channel is RED / GREEN / BLUE,
    or LUM for a curve that applies to every channel (filters, reference spectra)."""
    d = os.path.join(cache_dir, kind)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name + ".json")
    if not os.path.exists(p):
        raw = _get(DB_RAW.format(path=urllib.parse.quote(f"{kind}/{name}.json")))
        open(p, "wb").write(raw)
    entries = json.load(open(p))
    entries = entries if isinstance(entries, list) else [entries]
    out = {}
    for e in entries:
        wl = np.asarray(e["wavelength"]["value"], float)
        if str(e["wavelength"].get("units", "nm")).lower().startswith("ang"):
            wl = wl / 10.0
        v = np.asarray(e["values"]["value"], float) / float(e["values"].get("range") or 1)
        o = np.argsort(wl)
        out[str(e.get("channel") or "LUM").upper()] = (wl[o], v[o])
    return out


def _on_grid(curve: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    wl, v = curve
    return np.interp(WL, wl, v, left=0.0, right=0.0)


# ============================================================ camera & filter
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def detect_sensor(instrument: str, width: int, height: int, pixsize: float, available: list[str]) -> tuple[str | None, str]:
    """(database sensor name, how it was found) from the FITS header, or (None, reason)."""
    inst = instrument or ""
    geo = sorted((int(width), int(height)))

    def geometry_matches():
        return [n for (w, h, px, n) in SENSOR_GEOMETRY
                if sorted((w, h)) == geo and abs(px - pixsize) < 0.06 and n in available]
    m = re.search(r"seestar\s*(s\d+)", inst, re.I)
    if m:
        cand = f"ZWO_Seestar_{m.group(1).upper()}"
        # a Seestar curve is its 1080p sensor's; a Seestar with another sensor is found by geometry
        if cand in available and geo == [1080, 1920]:
            return cand, f"camera {inst}"
    # a sensor or camera model number in the instrument name (ASI585MC, IMX571, QHY268C, Uranus-C 585 ...)
    for num in re.findall(r"(\d{3,4})", inst):
        num = CAMERA_ALIASES.get(num, num)
        hits = [n for n in available if re.search(rf"(?<!\d){num}(?!\d)", n)]
        if len(hits) == 1:
            return hits[0], f"camera {inst}"
    hits = [n for n in available if _norm(n) and inst and _norm(n) in _norm(inst)]
    if len(hits) == 1:
        return hits[0], f"camera {inst}"
    g = geometry_matches()
    if len(g) == 1:
        return g[0], f"sensor geometry {width}x{height}, {pixsize:g} um"
    if len(g) > 1:
        return None, f"sensor geometry {width}x{height}, {pixsize:g} um fits {', '.join(g)}: choose the sensor"
    return None, f"unknown camera{' ' + inst if inst else ''} ({width}x{height}, {pixsize:g} um): choose the sensor"


def detect_filter(filter_name: str, instrument: str, available: list[str]) -> tuple[str | None, str]:
    """(database filter name, how) from the FILTER header.  An unnamed filter is taken to be a plain
    UV/IR-cut window, which most one-shot-colour cameras have (choose "No_filter" if not)."""
    f = (filter_name or "").strip()
    fn = _norm(f)
    seestar = "seestar" in (instrument or "").lower()
    if fn in ("lp", "duoband", "dualband") and (seestar or not instrument):
        hits = [n for n in available if "seestar_lp" in n.lower()]
        if hits:
            return hits[0], f"filter {f} (Seestar)"
    if fn in ("ircut", "uvir", "uvircut", "l", "lum", "luminance") and "UV-IR-Block" in available:
        return "UV-IR-Block", f"filter {f}"
    if fn in ("", "none", "clear", "open"):
        return ("UV-IR-Block" if "UV-IR-Block" in available else None), "no filter in the header: UV/IR-cut assumed"
    if len(fn) >= 3:
        hits = [n for n in available if fn in _norm(n)]
        if len(hits) == 1:
            return hits[0], f"filter {f}"
    return None, f"filter {f} is not in the database: choose it"


def resolve(infos0, params: dict, cache_dir: str) -> dict:
    """The sensor, filter and white reference to use, with how each was chosen."""
    sensors = list_curves("osc_sensors", cache_dir)
    filters = list_curves("osc_filters", cache_dir)
    s_pref, f_pref = params.get("spcc_sensor", "auto"), params.get("spcc_filter", "auto")
    if s_pref and s_pref != "auto":
        sensor, s_how = s_pref, "chosen"
    else:
        sensor, s_how = detect_sensor(getattr(infos0, "instrument", "") or "", infos0.width, infos0.height,
                                      infos0.pixsize, sensors)
    if f_pref and f_pref != "auto":
        filt, f_how = f_pref, "chosen"
    else:
        filt, f_how = detect_filter(infos0.filter, getattr(infos0, "instrument", "") or "", filters)
    ref = WHITE_REFS.get(params.get("spcc_white_ref", "average_spiral_galaxy"), "Average_spiral_galaxy")
    return {"sensor": sensor, "sensor_how": s_how, "filter": filt, "filter_how": f_how, "white_ref": ref,
            "db_offline": not sensors}


def system_response(sensor: str, filt: str | None, cache_dir: str) -> np.ndarray:
    """(3, len(WL)): quantum efficiency x filter transmission of the R, G and B pixels."""
    s = load_curve("osc_sensors", sensor, cache_dir)
    resp = np.stack([_on_grid(s[c]) for c in ("RED", "GREEN", "BLUE")])
    if filt and filt != "No_filter":
        fc = load_curve("osc_filters", filt, cache_dir)
        for i, c in enumerate(("RED", "GREEN", "BLUE")):
            resp[i] *= _on_grid(fc.get(c) or fc.get("LUM") or next(iter(fc.values())))
    return resp


# ============================================================ spectra
def ccm89(wl_nm: np.ndarray, rv: float = 3.1) -> np.ndarray:
    """A_lambda / A_V of Cardelli, Clayton & Mathis (1989), infrared and optical/NIR parts."""
    x = 1000.0 / np.asarray(wl_nm, float)         # inverse microns
    a, b = np.zeros_like(x), np.zeros_like(x)
    ir = x < 1.1
    a[ir], b[ir] = 0.574 * x[ir] ** 1.61, -0.527 * x[ir] ** 1.61
    op = ~ir
    y = np.minimum(x[op], 3.3) - 1.82
    a[op] = np.polyval([0.32999, -0.77530, 0.01979, 0.72085, -0.02427, -0.50447, 0.17699, 1.0], y)
    b[op] = np.polyval([-2.09002, 5.30260, -0.62251, -5.38434, 1.07233, 2.28305, 1.41338, 0.0], y)
    return a + b / rv


def _counts(spec: np.ndarray, resp: np.ndarray) -> np.ndarray:
    """Photon counts per channel of an F_lambda spectrum on WL (energy -> photons: x lambda)."""
    return (resp * (spec * WL)[None]).sum(1)


def template_table(resp: np.ndarray, cache_dir: str) -> dict:
    """log counts of every Pickles template through ``resp`` on the extinction grid:
    {class: (log Teff (T,), log counts (T, A_V, 3))}."""
    ext = 10 ** (-0.4 * np.outer(AV_GRID, ccm89(WL)))        # (A_V, WL)
    out = {}
    for cls, rows in PICKLES.items():
        lt, lc = [], []
        for code, teff in rows:
            spec = _on_grid(load_curve("wb_refs", "Star_" + code, cache_dir)["LUM"])
            c = np.stack([_counts(spec * e, resp) for e in ext])
            lt.append(np.log10(teff))
            lc.append(np.log10(np.maximum(c, 1e-300)))
        o = np.argsort(lt)
        out[cls] = (np.asarray(lt)[o], np.asarray(lc)[o])
    return out


def expected_colours(teff: np.ndarray, av: np.ndarray, giant: np.ndarray, table: dict) -> np.ndarray:
    """(N, 3) log10 counts of each star (up to its brightness), interpolated in log Teff and A_V."""
    out = np.full((len(teff), 3), np.nan)
    ai = np.clip(av / AV_GRID[1], 0, len(AV_GRID) - 1.001)
    a0 = ai.astype(int)
    fa = (ai - a0)[:, None]
    for cls, sel in (("giant", giant), ("dwarf", ~giant)):
        lt, lc = table[cls]
        idx = np.nonzero(sel)[0]
        if not len(idx):
            continue
        x = np.clip(np.log10(teff[idx]), lt[0], lt[-1])
        j = np.clip(np.searchsorted(lt, x) - 1, 0, len(lt) - 2)
        ft = ((x - lt[j]) / (lt[j + 1] - lt[j]))[:, None]
        k0, f = a0[idx], fa[idx]
        v = lambda jj, kk: lc[jj, kk]                     # (n, 3)
        lo = v(j, k0) * (1 - f) + v(j, k0 + 1) * f
        hi = v(j + 1, k0) * (1 - f) + v(j + 1, k0 + 1) * f
        out[idx] = lo * (1 - ft) + hi * ft
    return out


# ============================================================ session level
def reference_stars(sess, params: dict) -> dict | None:
    """Gaia stars of the solved field with their predicted colours in this camera, relative to the
    white reference: {"x", "y" (stack pixels), "g", "expected" (N, 3; G = 1), "meta"}; None when the
    stack is not plate-solved.  Raises RuntimeError when the curves cannot be had."""
    from .astrometry import load_solution
    sol = load_solution(sess)
    if sol is None:
        return None
    cache_dir = os.path.join(os.path.dirname(sess.dir), "spcc_db")
    meta = resolve(sess.infos[0], params, cache_dir)
    if meta["db_offline"]:
        raise RuntimeError("the SPCC database could not be reached (offline?) and is not cached yet")
    if not meta["sensor"]:
        raise RuntimeError(meta["sensor_how"])
    key = json.dumps([meta["sensor"], meta["filter"], meta["white_ref"], sol.get("solved")])
    cache = os.path.join(sess._p("explore"), "spcc_stars.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        if str(z["key"]) == key:
            return {"x": z["x"], "y": z["y"], "g": z["g"], "expected": z["expected"], "meta": meta}
    gaia = dict(np.load(os.path.join(sess._p("explore"), "gaia.npz")))
    resp = system_response(meta["sensor"], meta["filter"], cache_dir)
    table = template_table(resp, cache_dir)
    ref = _counts(_on_grid(load_curve("wb_refs", meta["white_ref"], cache_dir)["LUM"]), resp)
    teff, ag = gaia["teff"], np.nan_to_num(gaia.get("ag", np.zeros_like(gaia["teff"])), nan=0.0)
    plx, eplx = gaia["plx"], gaia["plx_err"]
    good_plx = np.isfinite(plx) & (plx > 0) & (plx / np.maximum(eplx, 1e-9) >= 5)
    abs_g = np.where(good_plx, gaia["g"] - ag + 5 * np.log10(np.where(good_plx, plx, 1.0) / 100.0), np.nan)
    giant = (teff < 5200) & np.isfinite(abs_g) & (abs_g < 2.5)
    use = np.isfinite(teff) & (teff > 2500)
    lc = expected_colours(teff[use], np.clip(ag[use] / AG_PER_AV, 0, AV_GRID[-1]), giant[use], table)
    rel = lc - np.log10(ref)[None]                     # colour relative to the white reference
    expected = 10 ** (rel - rel[:, 1:2])               # G = 1
    x, y = sol["wcs"].all_world2pix(gaia["ra"][use], gaia["dec"][use], 0)
    np.savez(cache, key=key, x=x, y=y, g=gaia["g"][use], expected=expected)
    return {"x": x, "y": y, "g": gaia["g"][use], "expected": expected, "meta": meta}


def photometric_gains(img: np.ndarray, stars: dict, sat_map: np.ndarray, noise: float = 0.0,
                      min_stars: int = 15) -> tuple[np.ndarray | None, dict]:
    """Per-channel gains (G = 1) that make the measured star colours of ``img`` (linear,
    background-neutralised) match their predicted colours.  ``stars``: positions in ``img``
    pixels (x, y), Gaia G and predicted colours; ``sat_map``: True where the data were clipped."""
    import sep
    from scipy.spatial import cKDTree
    h, w = img.shape[:2]
    x, y, g, e = stars["x"], stars["y"], stars["g"], stars["expected"]
    subs = []
    for c in range(3):
        ch = np.ascontiguousarray(img[..., c], np.float32)
        subs.append(ch - sep.Background(ch, bw=64, bh=64).back())
    # aperture: holds the whole star in the widest colour channel (refractor halos, see
    # postprocess.measure_star_colors), from the half-light radii of the brighter stars
    inner = (x > 30) & (y > 30) & (x < w - 31) & (y < h - 31)
    bright = np.nonzero(inner)[0][np.argsort(g[inner])][:200]
    r50 = []
    for sub in subs:
        rr, fl = sep.flux_radius(sub, x[bright], y[bright], np.full(len(bright), 15.0), 0.5, subpix=5)
        ok = np.isfinite(rr) & (rr > 0) & (fl == 0)
        r50.append(np.median(rr[ok]) if ok.any() else np.nan)
    r = float(np.clip(5.0 * np.nanmax(r50) if np.isfinite(r50).any() else 6.0, 3.0, 40.0))
    m = r + 12
    ok = (x > m) & (y > m) & (x < w - m - 1) & (y < h - m - 1) & np.isfinite(e).all(1)
    # isolated from every catalogued neighbour bright enough to matter (within 3 mag)
    tree = cKDTree(np.stack([x, y], 1))
    pairs = tree.query_pairs(r + 10, output_type="ndarray")
    crowded = np.zeros(len(x), bool)
    for i, j in pairs:
        if g[j] < g[i] + 3:
            crowded[i] = True
        if g[i] < g[j] + 3:
            crowded[j] = True
    ok &= ~crowded
    xi, yi = np.round(x).astype(int), np.round(y).astype(int)
    idx = np.nonzero(ok)[0]
    clipped = np.array([sat_map[max(0, yy - 3):yy + 4, max(0, xx - 3):xx + 4].any() for xx, yy in zip(xi[idx], yi[idx])],
                       bool)
    idx = idx[~clipped]
    if len(idx) < min_stars:
        return None, {"n_stars": int(len(idx)), "note": f"only {len(idx)} isolated, unsaturated catalogue stars"}
    fl, fe = [], []
    for sub in subs:
        f, ferr, _ = sep.sum_circle(sub, x[idx], y[idx], r, bkgann=(r + 4, r + 10), err=max(noise, 1e-12))
        fl.append(f)
        fe.append(ferr)
    fl, fe = np.stack(fl, 1), np.stack(fe, 1)
    good = (fl > 20 * fe).all(1)
    fl, ex = fl[good], e[idx][good]
    if len(fl) < min_stars:
        return None, {"n_stars": int(len(fl)), "note": f"only {len(fl)} catalogue stars with a high-SNR flux"}
    # log of (predicted / measured) colour, per star, R and B relative to G; sigma-clipped median
    d = np.log10(ex) - np.log10(fl / fl[:, 1:2])
    keep = np.ones(len(d), bool)
    for _ in range(5):
        med = np.median(d[keep], 0)
        sig = 1.4826 * np.median(np.abs(d[keep] - med), 0) + 1e-6
        new = (np.abs(d - med) < 3 * sig).all(1)
        if new.sum() < min_stars or np.array_equal(new, keep):
            break
        keep = new
    med = np.median(d[keep], 0)
    gains = np.clip(10 ** med, 0.1, 10).astype(np.float32)
    gains[1] = 1.0
    scatter = 2.5 * 1.4826 * np.median(np.abs(d[keep] - med), 0)          # mag
    return gains, {"n_stars": int(keep.sum()), "n_candidates": int(len(d)), "aperture_px": round(r, 1),
                   "scatter_mag": [round(float(scatter[0]), 3), round(float(scatter[2]), 3)]}
