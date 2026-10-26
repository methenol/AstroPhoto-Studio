"""No-reference objectives of the experiment lab and the hyper-tuned pipeline: paired measurements of a trial against one reference image.

Real data has no ground truth, and the obvious single proxies are gameable (measured on Sh2-131, 2026-10-25): Krotkov sharpness rose as the denoise
strength fell (it scores noise as detail), and the star FWHM "improved" with GraXpert's stars model, which keeps about a fifth of the star flux.  The
objectives here therefore pair one figure of merit with constraints that close those exploits, and every trial is measured on the SAME stars and the
SAME empty-sky apertures, chosen once on a reference image (the study's no-op reference or its baseline trial), so trial-to-trial differences are not
lost in the scatter of a re-detection.

Depth (stack and restore stages): the signal-to-noise ratio of point sources and of extended emission.

* Fluxes: circular apertures with a local background from an annulus (``sep.sum_circle``, Barbary 2016; SExtractor's method, Bertin & Arnouts 1996).
  The point-source aperture has radius 0.68 FWHM of the reference: the radius that maximises the S/N of a Gaussian PSF on a background-limited image
  (radius 1.6 sigma), so a sharper image puts more of a star's flux into it.  The extended-source aperture (2.5 FWHM) holds nearly all of the flux.
* Noise: the "empty aperture" method of deep-field photometry (Labbe et al. 2003, Quadri et al. 2007, Skelton et al. 2014): the scatter (normalised
  MAD) of the sums of the same apertures dropped on source-free sky, after the 64 px mesh background is subtracted.  Unlike sigma_pixel * sqrt(area) it
  measures the noise that is really there, including the pixel-to-pixel correlation drizzle, interpolation, deconvolution and denoising introduce.
* ``depth`` = (ln median S/N_point + ln median S/N_extended) / 2: higher is better.  S/N is dimensionless, so stacks normalised differently, or on
  another grid (drizzle scale, crop), compare directly; their star positions are mapped onto the trial by an astroalign transform (Beroiz et al. 2020).

Constraints (each a measured quantity with its published or physical rationale):

* ``dm_*``: aperture photometry of the reference stars in the large aperture against the reference image (magnitudes).  A restoration must conserve
  flux: deconvolution redistributes light within the PSF, and denoising removes zero-mean noise (Richardson 1972 / Lucy 1974 conserve flux by
  construction; BlurXTerminator's documentation states it does).
* ``large_change``: rms of the difference to the reference at scales above the PSF (both smoothed by a Gaussian of sigma = 1 FWHM), away from the stars,
  in units of the reference's own noise at that scale.  The PSF band-limits the sky, so what a restoration removes below the PSF scale is noise; above
  it, the restoration must agree with the data within the noise - the discrepancy principle (Morozov 1966) at the scale the data constrain.  A
  denoiser that only removes noise changes the image there by about 1 x the noise; one that smears faint structure, by more.
* ``moat``: the deepest dip of the bright stars' radial profiles below their local background (units of the pixel noise): the dark rings of
  over-deconvolution (Gibbs ringing of RL, the "black-ringed ball-bearing stars" of the 2026-10-21 ImageMM run).
* ``ellipticity`` / ``outliers``: star shape and single-pixel outliers (hot pixels, cosmic rays a rejection let through), for the stack.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import sep

# a frame-filling galaxy or nebula puts millions of pixels above a detection threshold: sep's default buffer (300 000 pixels) raised "internal pixel
# buffer full" on M 31 (the star-removal study failed there, 2026-10-25)
sep.set_extract_pixstack(20_000_000)
sep.set_sub_object_limit(4096)

AP_POINT = 0.68            # point-source aperture radius / FWHM (the S/N optimum of a Gaussian PSF: 1.6 sigma)
AP_TOTAL = 2.5             # total-flux aperture radius / FWHM
ANNULUS = (3.5, 5.5)       # local-background annulus / FWHM
EDGE = 16                  # px kept away from the image's border and from missing data
N_APERTURES = 4000


def lum(img: np.ndarray) -> np.ndarray:
    """(h, w, 3) or (h, w) -> (h, w) float32 C-contiguous (channel mean), NaN -> 0."""
    L = img.mean(-1) if img.ndim == 3 else img
    return np.ascontiguousarray(np.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0), np.float32)


def nmad(v) -> float:
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return float("nan")
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def valid_region(L: np.ndarray, edge: int = EDGE) -> np.ndarray:
    """Pixels with data (not 0 / NaN: the stack's uncovered rim, drizzle holes), eroded by ``edge``."""
    ok = (L != 0) & np.isfinite(L)
    ok[:edge], ok[-edge:], ok[:, :edge], ok[:, -edge:] = False, False, False, False
    if edge > 0:
        ok = cv2.erode(ok.astype(np.uint8), np.ones((2 * edge + 1, 2 * edge + 1), np.uint8)) > 0
    return ok


def _background(L: np.ndarray, valid: np.ndarray):
    bk = sep.Background(L, mask=~valid, bw=64, bh=64, fw=3, fh=3)
    return bk.back(), float(bk.globalrms)


def wing_fraction(sub: np.ndarray, x, y) -> np.ndarray:
    """Per source: the light in the 5 x 5 box round its peak outside its 4 brightest pixels, over those 4.  A star sampled at FWHM >= 1.5 px has
    0.5 or more there; a pixel defect - a hot pixel or cosmic ray, a single pixel or, after a Bayer drizzle, a 2 x 2 block - nearly none."""
    h, w = sub.shape
    out = np.zeros(len(x))
    for i, (xi, yi) in enumerate(zip(np.asarray(x), np.asarray(y))):
        cx, cy = int(round(xi)), int(round(yi))
        if cx < 2 or cy < 2 or cx >= w - 2 or cy >= h - 2:
            continue
        box = sub[cy - 2:cy + 3, cx - 2:cx + 3].ravel()
        top = np.sort(box)[-4:]
        t = float(top.sum())
        out[i] = float((box.sum() - t) / t) if t > 0 else 0.0
    return out


WING_MIN = 0.15


def star_points(img: np.ndarray, n: int = 150, thresh: float = 5.0) -> np.ndarray:
    """(N, 2) x, y of the ``n`` brightest point sources (pixel defects left out): the control points to align two images of one field on."""
    L = lum(img)
    valid = valid_region(L, edge=4)
    back, rms = _background(L, valid)
    sub = np.ascontiguousarray(L - back, np.float32)
    objs = sep.extract(sub, thresh, err=rms, mask=~valid, minarea=5, deblend_cont=0.005)
    real = (objs["npix"] >= 9) & (wing_fraction(sub, objs["x"], objs["y"]) >= WING_MIN)
    top = np.nonzero(real)[0][np.argsort(-objs["flux"][real])][:n]
    return np.stack([objs["x"][top], objs["y"][top]], 1).astype(np.float64)


def _stretch_for_match(sub: np.ndarray, rms: float) -> np.ndarray:
    return np.ascontiguousarray(np.arcsinh(np.clip(sub / max(rms, 1e-12), -3, None) / 3.0), np.float32)


def _gauss(img: np.ndarray, s: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), float(s)) if s > 0 else img


class Yardstick:
    """Stars, empty-sky apertures and noise of a reference image, applied unchanged to every trial (``measure``)."""

    def __init__(self, ref: np.ndarray, *, keep_image: bool = True, n_apertures: int = N_APERTURES, seed: int = 0, max_stars: int = 400):
        L = lum(ref)
        h, w = L.shape
        self.shape = (h, w)
        valid = valid_region(L)
        if valid.mean() < 0.2:
            raise RuntimeError("the reference image has too little valid area to measure")
        back, rms = _background(L, valid)
        sub = np.ascontiguousarray(L - back, np.float32)
        objs, seg = sep.extract(sub, 2.0, err=rms, mask=~valid, minarea=5, segmentation_map=True, deblend_cont=0.005)
        if len(objs) < 10:
            raise RuntimeError(f"only {len(objs)} sources in the reference image: too few to measure")
        sat = float(np.percentile(L[valid], 99.99))
        # point sources, not pixel defects: a hot pixel / cosmic ray holds most of its flux in one pixel, a star sampled at FWHM >= 1.5 px under a third
        real = (objs["npix"] >= 9) & (wing_fraction(sub, objs["x"], objs["y"]) >= WING_MIN)
        # the PSF: isolated, unsaturated, high-S/N stars' half-light radius
        from scipy.spatial import cKDTree
        xy = np.stack([objs["x"], objs["y"]], 1)
        d, _ = cKDTree(xy).query(xy, k=2)
        size = np.sqrt(objs["a"] * objs["b"])
        clean = real & (objs["flag"] == 0) & (objs["peak"] < 0.5 * sat) & (objs["peak"] > 20 * rms) & (d[:, 1] > 10 * size + 6)
        if clean.sum() < 5:
            clean = real & (objs["peak"] < 0.7 * sat) & (objs["peak"] > 10 * rms)
        # half-light radius of the flux within 8 a (sep's isophotal flux stops at the detection threshold and would understate it)
        r50, _ = sep.flux_radius(sub, objs["x"][clean], objs["y"][clean], np.maximum(8 * objs["a"][clean], 5.0), 0.5, subpix=5)
        r50 = r50[np.isfinite(r50) & (r50 > 0)]
        self.fwhm = float(np.clip(np.median(2 * r50), 1.2, 30.0)) if len(r50) else 3.0
        f = self.fwhm
        self.r_point, self.r_total = max(1.5, AP_POINT * f), max(3.0, AP_TOTAL * f)
        self.annulus = (max(self.r_total + 1.5, ANNULUS[0] * f), max(self.r_total + 4.0, ANNULUS[1] * f))
        # reference stars: unsaturated, well inside the valid area, nothing else within the outer annulus
        ann = self.annulus[1]
        inside = valid[np.clip(objs["y"].astype(int), 0, h - 1), np.clip(objs["x"].astype(int), 0, w - 1)]
        inside &= (objs["x"] > ann + 2) & (objs["x"] < w - ann - 3) & (objs["y"] > ann + 2) & (objs["y"] < h - ann - 3)
        iso = d[:, 1] > ann
        ok = real & inside & iso & (objs["peak"] < 0.5 * sat) & (objs["peak"] > 8 * rms) & (objs["flag"] & 0x04 == 0)
        idx = np.nonzero(ok)[0]
        idx = idx[np.argsort(-objs["flux"][idx])][:max_stars]
        if len(idx) < 8:
            raise RuntimeError(f"only {len(idx)} isolated unsaturated stars in the reference image: too few to measure")
        self.sx, self.sy = objs["x"][idx].astype(np.float64), objs["y"][idx].astype(np.float64)
        self.bright = np.argsort(-objs["peak"][idx])[:40]                    # the brightest unsaturated ones: the moat measurement
        top = np.nonzero(real)[0][np.argsort(-objs["flux"][real])][:200]        # alignment points: the brightest detections, chosen as on a trial (_fit_transform)
        self.align_xy = np.stack([objs["x"][top], objs["y"][top]], 1).astype(np.float64)
        # the source mask (every detection, grown by one FWHM) and the empty-aperture centres: a disc of r_total free of it
        m = (seg > 0).astype(np.uint8)
        k = int(max(3, round(f))) * 2 + 1
        m = cv2.dilate(m, np.ones((k, k), np.uint8)) > 0
        free = valid & ~m
        dist = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5)
        cand = np.argwhere(dist > self.r_total + 1)
        rng = np.random.default_rng(seed)
        if len(cand) < 200:
            cand = np.argwhere(dist > self.r_point + 1)
        if len(cand) < 50:
            raise RuntimeError("no source-free sky to measure the noise on")
        pick = cand[rng.choice(len(cand), size=min(n_apertures, len(cand)), replace=False)]
        self.ay, self.ax = pick[:, 0].astype(np.float64), pick[:, 1].astype(np.float64)
        # away from the stars: every detection's footprint grown by 2 total-flux radii (the large-scale fidelity is judged there)
        k2 = int(round(2 * self.r_total)) * 2 + 1
        self.offstar = valid & ~(cv2.dilate((seg > 0).astype(np.uint8), np.ones((k2, k2), np.uint8)) > 0)
        self.valid = valid
        self.n_sources = int(len(objs))
        # a stretched central crop of the reference: the fallback alignment (template matching) when the stars cannot be matched
        c = int(min(1024, 0.6 * min(h, w)))                                  # small enough to lie inside another crop of the same field
        self.tpl_y0, self.tpl_x0 = (h - c) // 2, (w - c) // 2
        self.tpl = _stretch_for_match(sub[self.tpl_y0:self.tpl_y0 + c, self.tpl_x0:self.tpl_x0 + c], rms)
        # reference fluxes (large aperture) and the noise at the PSF scale, for the paired constraints
        self.f_total = self._phot(L, self.sx, self.sy, self.r_total)
        self.f_point = self._phot(L, self.sx, self.sy, self.r_point)
        # the fidelity scale: the PSF core's full width at half maximum (azimuthal profiles of the reference stars).  2 x the half-light radius,
        # which sizes the apertures, is 1.35 - 1.45 x larger on real stars' Moffat wings (Sh2-131 stack: 8.1 vs 6.0 px, M 31: 4.3 vs 3.0)
        self.fwhm_core = self._half_max_width(sub, rms) or f
        fs = self.fwhm_core
        self.sigma_scale = nmad(_gauss(sub, fs)[self.ay.astype(int), self.ax.astype(int)])
        self.ref_smooth = _gauss(L, fs) if keep_image else None
        # flux conservation on the same grid: a wide aperture (3 x the half-light diameter, the stars' extended wings inside) with the median of a far
        # annulus as the background - a local annulus at 3.5-5.5 FWHM still holds the wings, which a deconvolution pulls into the core: BlurXTerminator's
        # star sharpening alone "brightened" M 31's stars by 0.04-0.06 mag that way while conserving their flux; a global background map instead let a
        # denoiser's small shift of the local sky (summed over the wide aperture) "brighten" faint stars by 0.6 mag
        self.r_cons = max(6.0, 3.0 * f)
        self.f_cons = self._phot_cons(L) if keep_image else None
        self.ref = self.measure(ref, same_grid=True)

    def _half_max_width(self, sub, rms) -> float | None:
        """Median FWHM (half-maximum diameter of the azimuthal profile, 0.25 px steps) of the reference stars peaking above 20 sigma."""
        R = int(min(40, max(8, 3 * self.fwhm)))
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        h, w = sub.shape
        out = []
        for x, y in zip(self.sx, self.sy):
            x0, y0 = int(round(x)), int(round(y))
            if x0 < R or y0 < R or x0 >= w - R or y0 >= h - R:
                continue
            c = sub[y0 - R:y0 + R + 1, x0 - R:x0 + R + 1]
            r = np.hypot(xx - (x - x0), yy - (y - y0))
            core = c[r < 1.0]
            pk = float(core.max()) if core.size else float(c.max())
            if pk < 20 * rms:
                continue
            edges = np.arange(0.0, R, 0.25)
            prof = np.array([c[(r >= a) & (r < a + 0.5)].mean() if ((r >= a) & (r < a + 0.5)).any() else np.nan for a in edges])
            below = np.nonzero(prof < pk / 2)[0]
            if len(below) and below[0] > 0:
                k = below[0]
                a0, a1, p0, p1 = edges[k - 1], edges[k], prof[k - 1], prof[k]
                out.append(2 * (a0 + (p0 - pk / 2) / max(p0 - p1, 1e-12) * (a1 - a0) + 0.25))
            if len(out) >= 150:
                break
        return float(np.median(out)) if len(out) >= 5 else None

    # ------------------------------------------------------------------ pieces
    def _phot_cons(self, L, x=None, y=None) -> np.ndarray:
        """Flux in the wide aperture (r_cons) less the median of a far annulus (1.4 - 2 r_cons): the wings inside, the local sky level - which a denoiser
        may shift - cancelled, neighbours in the annulus ignored by the median."""
        x = self.sx if x is None else x
        y = self.sy if y is None else y
        h, w = L.shape
        r = self.r_cons
        R = int(np.ceil(2.0 * r)) + 1
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
        out = np.full(len(x), np.nan)
        for i, (xi, yi) in enumerate(zip(np.asarray(x), np.asarray(y))):
            x0, y0 = int(round(xi)), int(round(yi))
            if x0 - R < 0 or y0 - R < 0 or x0 + R >= w or y0 + R >= h:
                continue
            cut = L[y0 - R:y0 + R + 1, x0 - R:x0 + R + 1]
            rr = np.hypot(xx - (xi - x0), yy - (yi - y0))
            inner, ann = rr <= r, (rr >= 1.4 * r) & (rr <= 2.0 * r)
            out[i] = float(cut[inner].sum() - np.median(cut[ann]) * inner.sum())
        return out

    def _phot(self, L, x, y, r):
        fl, _, _ = sep.sum_circle(L, x, y, r, bkgann=self.annulus, subpix=5)
        return np.asarray(fl, np.float64)

    def _positions(self, transform):
        if transform is None:
            return self.sx, self.sy, self.ax, self.ay
        s = transform(np.stack([self.sx, self.sy], 1))
        a = transform(np.stack([self.ax, self.ay], 1))
        return s[:, 0], s[:, 1], a[:, 0], a[:, 1]

    def transform_to(self, img: np.ndarray):
        """The reference -> ``img`` mapping (a callable on (N, 2) x, y) from the stars of both (astroalign); None when the grids coincide."""
        import astroalign as aa
        L = lum(img)
        if L.shape == self.shape:
            return None if self._same_grid(L) else self._fit_transform(L, aa)
        return self._fit_transform(L, aa)

    def _same_grid(self, L) -> bool:
        """The brightest reference stars still peak where they were (within 0.5 px): the trial is on the reference's grid."""
        sel = self.bright[:15]
        xs, ys = self.sx[sel], self.sy[sel]
        valid = valid_region(L)
        back, rms = _background(L, valid)
        sub = np.ascontiguousarray(L - back, np.float32)
        try:
            objs = sep.extract(sub, 8.0, err=rms, mask=~valid, minarea=5)
        except Exception:                                   # noqa: BLE001
            return False
        if len(objs) < 5:
            return False
        from scipy.spatial import cKDTree
        dd, _ = cKDTree(np.stack([objs["x"], objs["y"]], 1)).query(np.stack([xs, ys], 1))
        return bool(np.median(dd) < 0.5)

    def _fit_transform(self, L, aa):
        valid = valid_region(L)
        back, rms = _background(L, valid)
        sub = np.ascontiguousarray(L - back, np.float32)
        objs = sep.extract(sub, 2.0, err=rms, mask=~valid, minarea=5, deblend_cont=0.005)
        if len(objs) < 10:
            raise RuntimeError("too few stars in the trial image to map the reference stars onto it")
        real = (objs["npix"] >= 9) & (wing_fraction(sub, objs["x"], objs["y"]) >= WING_MIN)
        top = np.nonzero(real)[0][np.argsort(-objs["flux"][real])][:200]
        dst = np.stack([objs["x"][top], objs["y"][top]], 1)
        try:
            tf, _ = aa.find_transform(self.align_xy, dst, max_control_points=60)
            return lambda pts: tf(np.asarray(pts, np.float64))
        except Exception:                                   # noqa: BLE001  too few / ambiguous matches: a pure shift (same registration, another crop)
            return self._shift_transform(sub, rms)

    def _shift_transform(self, sub, rms):
        """The reference -> image mapping as a pure translation, from template matching of the reference's stretched central crop
        (normalised cross-correlation, sub-pixel by a parabola through the peak).  Stack trials of one study share the registration's
        reference frame and the output scale: they differ by their crop."""
        img = _stretch_for_match(sub, rms)
        if img.shape[0] < self.tpl.shape[0] or img.shape[1] < self.tpl.shape[1]:
            raise RuntimeError("the trial image is smaller than the reference's alignment template")
        r = cv2.matchTemplate(img, self.tpl, cv2.TM_CCOEFF_NORMED)
        _, peak, _, (px, py) = cv2.minMaxLoc(r)
        if peak < 0.5:
            raise RuntimeError(f"the trial image does not match the reference (correlation {peak:.2f})")

        def sub_px(a, b, c):
            d = a - 2 * b + c
            return 0.0 if d == 0 else 0.5 * (a - c) / d
        fx = sub_px(r[py, px - 1], r[py, px], r[py, px + 1]) if 0 < px < r.shape[1] - 1 else 0.0
        fy = sub_px(r[py - 1, px], r[py, px], r[py + 1, px]) if 0 < py < r.shape[0] - 1 else 0.0
        dx, dy = px + fx - self.tpl_x0, py + fy - self.tpl_y0
        return lambda pts: np.asarray(pts, np.float64) + np.array([dx, dy])

    # ------------------------------------------------------------------ measurement
    def measure(self, img: np.ndarray, *, same_grid: bool = False, transform=None, normalise_gain: bool = True) -> dict:
        """Depth, photometry and shape of ``img`` ((h, w, 3) or (h, w) linear) on the reference stars and apertures.  ``same_grid``: the
        trial is on the reference's pixel grid (restore / stars / background trials; the large-scale fidelity is measured then);
        otherwise ``transform`` maps the reference positions onto it (``transform_to``).  ``normalise_gain``: judge the photometry and the
        fidelity after a global rescale (a restoration's change of units); off where any change of level is the damage (star / trail removal)."""
        L = lum(img)
        if not same_grid and transform is None and L.shape != self.shape:
            transform = self.transform_to(img)
        gain = 1.0
        if normalise_gain and same_grid and self.ref_smooth is not None and L.shape == self.shape:
            # a global rescale is a change of units, not of the image (post._sanitize divides a restored image by its peak when deconvolved star cores
            # exceed 1): the photometry and the fidelity are judged after it, measured as the ratio of the off-star levels at the PSF scale
            m0 = self.offstar & self.valid
            a, b = float(np.median(_gauss(L, self.fwhm_core)[m0])), float(np.median(self.ref_smooth[m0]))
            if a > 0 and b > 0:
                gain = a / b
                L = (L / np.float32(gain)).astype(np.float32)
        sx, sy, ax, ay = self._positions(None if same_grid else transform)
        h, w = L.shape
        valid = valid_region(L)
        back, rms = _background(L, valid)
        sub = np.ascontiguousarray(L - back, np.float32)
        inb = (ax > self.r_total + 1) & (ax < w - self.r_total - 2) & (ay > self.r_total + 1) & (ay < h - self.r_total - 2)
        inb &= valid[np.clip(ay.astype(int), 0, h - 1), np.clip(ax.astype(int), 0, w - 1)]
        scale = 1.0
        if transform is not None:                               # apertures follow the grid's scale (drizzle 2x: twice the radii)
            p = transform(np.array([[0.0, 0.0], [100.0, 0.0]]))
            scale = float(np.hypot(*(p[1] - p[0])) / 100.0)
        rt = self.r_total * scale
        ann = (self.annulus[0] * scale, self.annulus[1] * scale)
        sb = np.isfinite(sx) & np.isfinite(sy) & (sx > ann[1] + 1) & (sx < w - ann[1] - 2) & (sy > ann[1] + 1) & (sy < h - ann[1] - 2)

        def phot(r):                                                     # stars whose annulus leaves the frame (another crop) get NaN
            f = np.full(len(sx), np.nan)
            if sb.any():
                f[sb] = sep.sum_circle(L, sx[sb], sy[sb], r, bkgann=ann, subpix=5)[0]
            return f
        ft = phot(rt)
        et, _, _ = sep.sum_circle(sub, ax[inb], ay[inb], rt, subpix=5)
        sig_t = nmad(et)
        out: dict = {"n_stars": int(sb.sum()), "n_apertures": int(inb.sum()), "sigma_total": sig_t, "gain": gain,
                     "pixel_noise": nmad(sub[ay[inb].astype(int), ax[inb].astype(int)])}
        okt = sb & (ft > 0) & np.isfinite(ft)
        # FWHM of the reference stars in this image (twice the half-light radius within the total aperture)
        fw_img = None
        okr = okt & (ft > 20 * sig_t)
        if okr.sum() >= 5:
            r50, _ = sep.flux_radius(sub, sx[okr], sy[okr], np.full(int(okr.sum()), rt * 1.2), 0.5, normflux=ft[okr], subpix=5)
            r50 = r50[np.isfinite(r50) & (r50 > 0)]
            if len(r50) >= 5:
                fw_img = float(np.median(2 * r50))
                out["fwhm"] = fw_img / scale                                     # in the reference's pixels
        # point sources in the S/N-optimal aperture of THIS image's PSF (radius 0.68 FWHM): a sharper image puts the same light into a smaller
        # aperture, which holds less noise - the detection limit a restoration or a registration improves (flux kept: the photometry constraint)
        rp = max(1.5, AP_POINT * fw_img) if fw_img else self.r_point * scale
        fp = phot(rp)
        ep, _, _ = sep.sum_circle(sub, ax[inb], ay[inb], rp, subpix=5)
        sig_p = nmad(ep)
        out.update(sigma_point=sig_p, r_point=rp / scale)
        okp = sb & (fp > 0) & np.isfinite(fp)
        if okp.sum() >= 5 and okt.sum() >= 5 and sig_p > 0 and sig_t > 0:
            snr_p = float(np.median(fp[okp] / sig_p))
            snr_t = float(np.median(ft[okt] / sig_t))
            out.update(snr_point=snr_p, snr_total=snr_t, depth=0.5 * (math.log(snr_p) + math.log(snr_t)))
        # photometry against the reference (the same stars): flux conservation - on the same grid in the wide apertures over the reference's background,
        # on another grid (a stack trial: only the scatter matters there) in the local-background apertures
        if same_grid and self.f_cons is not None and L.shape == self.shape:
            ft_c, ref_t = self._phot_cons(L), self.f_cons
            okd = (ft_c > 0) & (ref_t > 0) & np.isfinite(ft_c) & np.isfinite(ref_t)
            ft_use = ft_c
        else:
            ref_t, ft_use = self.f_total, ft
            okd = sb & (ft > 0) & (ref_t > 0) & np.isfinite(ft)
        if okd.sum() >= 5:
            dm = -2.5 * np.log10(ft_use[okd] / ref_t[okd])
            bright = ref_t[okd] >= np.median(ref_t[okd])
            out.update(dm_median=float(np.median(dm)), dm_mad=nmad(dm), dm_bright=float(np.median(dm[bright])), dm_faint=float(np.median(dm[~bright])))
            out["dm_bright_abs"], out["dm_faint_abs"] = abs(out["dm_bright"]), abs(out["dm_faint"])
        out["moat"] = self._moat(sub, sx, sy, scale, out.get("pixel_noise") or rms)
        if same_grid and self.ref_smooth is not None and L.shape == self.shape:
            d = _gauss(L, self.fwhm_core) - self.ref_smooth
            m = self.offstar & valid
            if m.sum() > 1000 and self.sigma_scale > 0:
                dd = d[m]
                out["large_change"] = float(np.sqrt(np.mean((dd - np.median(dd)) ** 2)) / self.sigma_scale)
        return out

    def _moat(self, sub, sx, sy, scale, noise) -> float | None:
        """Median over the brightest reference stars of the deepest dip of the azimuthal profile (0.7 - 4 FWHM) below the annulus background, in pixel noise."""
        if not noise or not np.isfinite(noise) or noise <= 0:
            return None
        f = self.fwhm * scale
        R = int(math.ceil(self.annulus[1] * scale)) + 1
        h, w = sub.shape
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        vals = []
        for i in self.bright:
            x0, y0 = int(round(sx[i])), int(round(sy[i]))
            if x0 - R < 0 or y0 - R < 0 or x0 + R >= w or y0 + R >= h:
                continue
            cut = sub[y0 - R:y0 + R + 1, x0 - R:x0 + R + 1]
            r = np.hypot(xx - (sx[i] - x0), yy - (sy[i] - y0))
            bg = float(np.median(cut[(r >= self.annulus[0] * scale) & (r <= self.annulus[1] * scale)]))
            prof = []
            for r0 in np.arange(0.7 * f, 4.0 * f, max(1.0, 0.35 * f)):
                ring = (r >= r0) & (r < r0 + max(1.0, 0.35 * f))
                if ring.sum() >= 6:
                    prof.append(float(np.mean(cut[ring])))
            if prof:
                vals.append(max(0.0, bg - min(prof)) / noise)
        return float(np.median(vals)) if len(vals) >= 3 else None

    # ------------------------------------------------------------------ stack-only measurements
    @staticmethod
    def shape_and_outliers(img: np.ndarray, fwhm: float) -> dict:
        """Median ellipticity of isolated unsaturated stars and single-pixel outliers per megapixel (hot pixels / cosmic rays left by the rejection)."""
        L = lum(img)
        valid = valid_region(L)
        back, rms = _background(L, valid)
        sub = np.ascontiguousarray(L - back, np.float32)
        out: dict = {}
        try:
            objs = sep.extract(sub, 10.0, err=rms, mask=~valid, minarea=5)
            if len(objs) >= 10:
                from scipy.spatial import cKDTree
                sat = float(np.percentile(L[valid], 99.99))
                xy = np.stack([objs["x"], objs["y"]], 1)
                d, _ = cKDTree(xy).query(xy, k=2)
                ok = (objs["flag"] == 0) & (objs["peak"] < 0.5 * sat) & (d[:, 1] > 8 * fwhm)
                if ok.sum() >= 5:
                    out["ellipticity"] = float(np.median(1.0 - objs["b"][ok] / np.maximum(objs["a"][ok], 1e-6)))
        except Exception:                                    # noqa: BLE001
            pass
        # outliers: sources 8 sigma up whose light sits in at most a 2 x 2 block (wing_fraction) - hot pixels and cosmic rays the rejection let through
        try:
            o = sep.extract(sub, 8.0, err=rms, mask=~valid, minarea=1, filter_kernel=None)
            n = int((wing_fraction(sub, o["x"], o["y"]) < WING_MIN).sum()) if len(o) else 0
        except Exception:                                    # noqa: BLE001
            n = 0
        out["outliers"] = float(n / max(valid.sum(), 1) * 1e6)
        return out


def violations(m: dict, limits: dict) -> dict:
    """Constraint violations of a trial: ``limits`` = {metric: ("max" | "min", bound)}; a missing metric counts as violated by 1.  Value > 0 = violated."""
    out = {}
    for k, (kind, bound) in limits.items():
        v = m.get(k)
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            out[k] = 1.0
            continue
        out[k] = float(v - bound) if kind == "max" else float(bound - v)
    return out


# ----------------------------------------------------------------------------- finishing: structural fidelity of the finished image to its linear data
# TMQI structural fidelity (H. Yeganeh and Z. Wang, "Objective Quality Assessment of Tone-Mapped Images", IEEE TIP 22(2), 657-667, 2013; the authors'
# TMQI.m v1.0): local SSIM-like fidelity between the HDR source and the tone-mapped (finished) image, with the local signal strengths passed through a
# psychometric (Galton's ogive) visibility mapping, over 5 scales (2x2 box down-sampling), Gaussian 11x11 sigma 1.5 window, C1 = 0.01, C2 = 10, the
# finished side's threshold from the Mannos CSF at 16, 8, 4, 2, 1 cycles/degree, the scales weighted {0.0448, 0.2856, 0.3001, 0.2363, 0.1333}.
# ONE adaptation for astronomical data, on the source side only: TMQI.m rescales the HDR luminance to 2^32 - 1 and applies the visual threshold, so any
# variation of the source - its noise included - counts as significant, which suits daylight HDR photographs (nearly noise-free) and would reward a
# finish that stretches the sky's noise into view.  Here the source's local signal is significant when its local standard deviation exceeds the noise at
# that scale (Immerkaer 1996, "Fast noise variance estimation", CVIU 64(2)) by 3 sigma of the window's standard-deviation estimate: u = sigma_n
# (1 + 3 / sqrt(2 N_eff)), the ogive's spread u / 3 as in the paper (Crozier's ratio k = 3).  The paper's statistical naturalness term (fitted to the
# brightness / contrast of ~3000 everyday photographs) is not used: the finishing gates (FINISH-SPEC) are the astronomical image's bounds.
TMQI_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _tmqi_window() -> np.ndarray:
    g = cv2.getGaussianKernel(11, 1.5)
    w = g @ g.T
    return (w / w.sum()).astype(np.float64)


def immerkaer_sigma(img: np.ndarray) -> float:
    """Noise standard deviation by Immerkaer's Laplacian-difference estimator: sqrt(pi/2) / (6 (W-2)(H-2)) sum |I * N|."""
    n = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    r = cv2.filter2D(img.astype(np.float64), -1, n, borderType=cv2.BORDER_REFLECT)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2.0) * np.abs(r).sum() / (6.0 * r.size))


def _normcdf(x, mu, sd):
    from scipy.special import ndtr
    return ndtr((x - mu) / max(sd, 1e-12))


def structural_fidelity(hdr_lum: np.ndarray, ldr_lum: np.ndarray, mode: str = "hits", noise_from: np.ndarray | None = None) -> tuple[float, list[float]]:
    """S of the finished luminance ``ldr_lum`` (display values 0-1) against the linear luminance ``hdr_lum`` (same grid): (S, the five scales' s_l).
    ``noise_from``: the image whose noise sets the significance (same grid and units as ``hdr_lum``; default ``hdr_lum`` itself) - a noiseless
    truth is judged at the noise of the data it stands for."""
    w = _tmqi_window()
    neff = 1.0 / float((w ** 2).sum())
    lo, hi = float(np.min(hdr_lum)), float(np.max(hdr_lum))
    k_norm = (2.0 ** 32 - 1) / max(hi - lo, 1e-30)
    a = (np.asarray(hdr_lum, np.float64) - lo) * k_norm
    nz = None if noise_from is None else (np.asarray(noise_from, np.float64) - lo) * k_norm
    b = np.asarray(ldr_lum, np.float64) * 255.0
    s_local, f = [], 32.0
    for _ in range(5):
        f /= 2.0
        mu1, mu2 = cv2.filter2D(a, -1, w, borderType=cv2.BORDER_REFLECT), cv2.filter2D(b, -1, w, borderType=cv2.BORDER_REFLECT)
        s1 = cv2.filter2D(a * a, -1, w, borderType=cv2.BORDER_REFLECT) - mu1 * mu1
        s2 = cv2.filter2D(b * b, -1, w, borderType=cv2.BORDER_REFLECT) - mu2 * mu2
        s12 = cv2.filter2D(a * b, -1, w, borderType=cv2.BORDER_REFLECT) - mu1 * mu2
        e = 5                                                         # filter2 'valid': the window's half width off every side
        sg1, sg2, s12 = np.sqrt(np.maximum(s1, 0))[e:-e, e:-e], np.sqrt(np.maximum(s2, 0))[e:-e, e:-e], s12[e:-e, e:-e]
        u1 = immerkaer_sigma(a if nz is None else nz) * (1.0 + 3.0 / np.sqrt(2.0 * neff))     # the source: significant above its own noise at this scale
        p1 = _normcdf(sg1, u1, u1 / 3.0)
        csf = 100.0 * 2.6 * (0.0192 + 0.114 * f) * np.exp(-(0.114 * f) ** 1.1)
        u2 = 128.0 / (1.4 * csf)                                       # the finished image: TMQI.m's visibility threshold
        p2 = _normcdf(sg2, u2, u2 / 3.0)
        corr = (s12 + 10.0) / (sg1 * sg2 + 10.0)
        if mode == "tmqi":
            smap = ((2 * p1 * p2 + 0.01) / (p1 * p1 + p2 * p2 + 0.01)) * corr
            s_local.append(float(np.mean(smap)))
        else:
            # detection: hits = significant structure made visible (and kept in its shape: the correlation), false alarms = insignificant variation (noise)
            # made visible; Youden's J = hit rate - false-alarm rate (W. J. Youden, Cancer 3, 32-35, 1950)
            sig = float(p1.sum()) or 1e-9
            hit = float((p1 * p2 * np.clip(corr, 0, 1)).sum()) / sig
            fa = float(((1 - p1) * p2).sum()) / (float((1 - p1).sum()) or 1e-9)
            s_local.append(hit - fa if mode == "youden" else hit)
        k = np.ones((2, 2)) / 4.0
        a = cv2.filter2D(a, -1, k, borderType=cv2.BORDER_REFLECT)[::2, ::2]
        b = cv2.filter2D(b, -1, k, borderType=cv2.BORDER_REFLECT)[::2, ::2]
        if nz is not None:
            nz = cv2.filter2D(nz, -1, k, borderType=cv2.BORDER_REFLECT)[::2, ::2]
    S = float(np.prod([max(s, 1e-9) ** wt for s, wt in zip(s_local, TMQI_WEIGHTS)])) if mode == "tmqi" else float(np.dot(s_local, TMQI_WEIGHTS))
    return S, s_local


def luminance_y(rgb: np.ndarray) -> np.ndarray:
    """Y of linear sRGB (TMQI.m's RGBtoYxy row), (3, H, W) or (H, W, 3) -> (H, W)."""
    x = rgb if rgb.shape[0] == 3 and rgb.ndim == 3 else np.moveaxis(rgb, -1, 0)
    return (0.2126 * x[0] + 0.7152 * x[1] + 0.0722 * x[2]).astype(np.float64)


# ----------------------------------------------------------------------------- restoration: held-out validation against independent frames
# The restoration is run on a stack of part of the frames (R); the sharpest frames, stacked on their own on the same pixel grid (S), are held
# out.  Their noise is independent of R's, so E||S - k * f(R)||^2 = ||x_S - k * f(R)||^2 + E||n_S||^2 for any restoration f and any kernel k
# fixed beforehand: the error against S estimates, up to a constant, the error against the true sky x_S at S's resolution (the identity
# Noise2Noise rests on, Lehtinen et al. 2018, used here to validate rather than to train).  k is the kernel that matches the restoration to
# S's point-spread function, fitted by least squares per tile as in difference imaging (Alard & Lupton 1998, ApJ 503, 325; the pixel-kernel
# form of Bramich 2008, MNRAS 386, L77, with a differential background): a deconvolution that recovers the true sky is blurred back to S
# exactly, one that invents or misplaces structure is not, and a denoiser lowers the error by removing R's noise unless it removes signal
# with it.  The residual is weighted by the inverse of its noise variance, a linear function of the signal level (photon and sky noise;
# the noise-level function of Foi et al. 2008, IEEE TIP 17(10)), fitted once on the unrestored R: the score is the mean chi^2 (1 = noise).
# Both sides are high-passed (8 FWHM) first - the gradient model is judged on its own (emission_change) - and detected stars are masked
# (a kernel per tile cannot match every star's centroid and colour; stars are judged by the photometry constraints).
HO_KERNEL = 6                    # kernel half width (13 x 13: S's FWHM ~4 px against a deconvolved ~2 px)
HO_TILES = (4, 8)                # the kernel varies over the field (aberrations, BlurXTerminator's correction)
HO_FIT_PX = 25000                # pixels per tile in the kernel fit (169 + 1 unknowns)


def _ho_highpass(L: np.ndarray, sigma: float) -> np.ndarray:
    L = np.ascontiguousarray(L, np.float32)
    return (L - cv2.GaussianBlur(L, (0, 0), sigma)).astype(np.float32)


class HeldOut:
    """The held-out reference: S (the sharpest frames' stack, restored without deconvolution / noise reduction) and the noise model."""

    def __init__(self, S: np.ndarray, R0: np.ndarray, fwhm: float, seed: int = 0):
        S, R0 = lum(S), lum(R0)
        self.sigma_hp = 8.0 * float(fwhm)
        self.B = _ho_highpass(S, self.sigma_hp)
        valid = valid_region(S) & valid_region(R0)
        sky = float(np.median(S[valid]))
        noise = immerkaer_sigma(self.B.astype(np.float64))
        bk = sep.Background(np.ascontiguousarray(self.B), bw=64, bh=64)
        objs = sep.extract(np.ascontiguousarray(self.B - bk.back()), 3.0, err=bk.globalrms, minarea=4)
        stars = np.zeros(S.shape, np.uint8)
        for x_, y_, a_ in zip(objs["x"], objs["y"], objs["a"]):
            cv2.circle(stars, (int(round(x_)), int(round(y_))), int(round(3 * a_ + 4)), 1, -1)
        self.mask = valid & (stars == 0) & (S < 0.9 * float(S.max()))
        self.n_stars = int(len(objs))
        self.level = np.maximum(cv2.GaussianBlur(S, (0, 0), 3.0) - sky, 0).astype(np.float32)
        # the kernel-fit pixels of each tile, drawn once: every trial is fitted on the same pixels (a fresh draw per trial scattered the score)
        rng = np.random.default_rng(seed)
        H, W = S.shape
        k = HO_KERNEL
        self.tiles = []
        for ty in range(HO_TILES[0]):
            for tx in range(HO_TILES[1]):
                y0, y1, x0, x1 = ty * H // HO_TILES[0], (ty + 1) * H // HO_TILES[0], tx * W // HO_TILES[1], (tx + 1) * W // HO_TILES[1]
                yy, xx = np.nonzero(self.mask[y0 + k:y1 - k, x0 + k:x1 - k])
                if len(yy) < 2000:
                    continue
                sel = rng.choice(len(yy), min(HO_FIT_PX, len(yy)), replace=False)
                self.tiles.append(((y0, y1, x0, x1), yy[sel] + y0 + k, xx[sel] + x0 + k))
        # the noise-level function, from the unrestored R matched to S (no weights yet)
        r0 = self.B - self._predict(_ho_highpass(R0, self.sigma_hp), np.ones_like(self.B))
        m = self.mask
        q = np.quantile(self.level[m], np.linspace(0, 0.999, 25))
        lx, ly = [], []
        for lo, hi in zip(q[:-1], q[1:]):
            sel = m & (self.level >= lo) & (self.level < hi)
            if sel.sum() > 500:
                d = r0[sel]
                lx.append(float(np.median(self.level[sel])))
                ly.append(float((1.4826 * np.median(np.abs(d - np.median(d)))) ** 2))
        b, a = np.polyfit(lx, ly, 1) if len(lx) >= 3 else (0.0, float(noise ** 2))
        a = max(float(a), 1e-3 * max(ly or [noise ** 2]))
        self.nlf = (a, max(float(b), 0.0))
        self.w = (1.0 / (self.nlf[0] + self.nlf[1] * self.level)).astype(np.float32)
        self.chi2_r0 = self.chi2(R0)

    def _predict(self, A: np.ndarray, w: np.ndarray) -> np.ndarray:
        """S's high-pass predicted from ``A`` (the trial's high-pass) by the least-squares kernel of each tile (weights ``w``).  On the GPU
        (PyTorch CUDA) when there is one: the design matrices (25 000 x 170 a tile) and the convolutions are its work."""
        k = HO_KERNEL
        try:
            import torch
            if torch.cuda.is_available():
                return self._predict_torch(A, w, torch)
        except ImportError:
            pass
        out = np.zeros_like(A)
        offs = [(dy, dx) for dy in range(-k, k + 1) for dx in range(-k, k + 1)]
        for (y0, y1, x0, x1), ys, xs in self.tiles:
            sw = np.sqrt(w[ys, xs]).astype(np.float64)[:, None]
            X = np.hstack([np.stack([A[ys - dy, xs - dx] for dy, dx in offs], 1), np.ones((len(ys), 1))]).astype(np.float64)
            coef, *_ = np.linalg.lstsq(X * sw, self.B[ys, xs].astype(np.float64) * sw[:, 0], rcond=None)
            kern = coef[:-1].reshape(2 * k + 1, 2 * k + 1).astype(np.float32)
            out[y0:y1, x0:x1] = cv2.filter2D(A[y0:y1, x0:x1], -1, kern, borderType=cv2.BORDER_REFLECT) + np.float32(coef[-1])
        return out

    def _predict_torch(self, A: np.ndarray, w: np.ndarray, torch) -> np.ndarray:
        import torch.nn.functional as F
        k = HO_KERNEL
        dev = torch.device("cuda")
        At = torch.from_numpy(np.ascontiguousarray(A, np.float32)).to(dev)
        Bt = torch.from_numpy(np.ascontiguousarray(self.B, np.float32)).to(dev)
        Wt = torch.from_numpy(np.ascontiguousarray(w, np.float32)).to(dev)
        d = torch.arange(-k, k + 1, device=dev)
        dy, dx = torch.meshgrid(d, d, indexing="ij")
        dy, dx = dy.reshape(-1), dx.reshape(-1)
        out = torch.zeros_like(At)
        for (y0, y1, x0, x1), ys, xs in self.tiles:
            yt = torch.from_numpy(ys).to(dev)[:, None]
            xt = torch.from_numpy(xs).to(dev)[:, None]
            X = At[yt - dy[None, :], xt - dx[None, :]].double()
            X = torch.cat([X, torch.ones((X.shape[0], 1), dtype=X.dtype, device=dev)], 1)
            sw = Wt[yt[:, 0], xt[:, 0]].double().sqrt()[:, None]
            coef = torch.linalg.lstsq(X * sw, (Bt[yt[:, 0], xt[:, 0]].double() * sw[:, 0])[:, None]).solution[:, 0]
            kern = coef[:-1].reshape(1, 1, 2 * k + 1, 2 * k + 1).float()
            tile = F.pad(At[None, None, y0:y1, x0:x1], (k, k, k, k), mode="reflect")
            out[y0:y1, x0:x1] = F.conv2d(tile, kern)[0, 0] + coef[-1].float()        # conv2d is a correlation, as cv2.filter2D
        return out.cpu().numpy()

    def chi2(self, img: np.ndarray) -> float:
        """Mean chi^2 of S against the restoration ``img`` (on S's grid) matched to S's PSF: 1 = only noise left."""
        f = lum(img)
        m = self.mask & valid_region(f)
        r = self.B - self._predict(_ho_highpass(f, self.sigma_hp), self.w)
        q = (r.astype(np.float64) ** 2) * self.w
        # per tile, for the paired comparison of two trials on the same held-out pixels (lab.run.best_trial)
        self.last_tiles = [float(np.mean(q[y0:y1, x0:x1][m[y0:y1, x0:x1]])) for (y0, y1, x0, x1), _, _ in self.tiles if m[y0:y1, x0:x1].any()]
        return float(np.mean(q[m]))
