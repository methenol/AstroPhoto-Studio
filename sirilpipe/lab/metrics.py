"""Quality metrics for the experiment lab (Siril backend).

Real data has no ground truth: the scores are proxies measured on the result (star FWHM, background noise, Krotkov sharpness, aperture photometry
against the stack) and the finishing gates (gates.py) for the finished picture.  Synthetic data (synthetic.py) has an exact truth: ``truth_metrics``
compares a result with it on a common grid after a per-channel scale + plane fit (the stack's photometric scale and sky are not part of the truth).
"""
from __future__ import annotations

import numpy as np
import sep


def sharpness(img):
    """S_F (Krotkov 1988): log magnitude of the 2-D Fourier transform, averaged over frequencies."""
    return float(np.mean(np.log(np.abs(np.fft.fft2(img)) + 1e-12)))


def sky_sigma(img):
    return float(sep.Background(np.ascontiguousarray(img, np.float32)).globalrms)


def photometry(coadd, img, thresh):
    """Sources of ``coadd`` (sep, absolute threshold); fluxes in the same elliptical apertures
    (2.5 Kron radii) on the coadd and on ``img``.  Returns (coadd magnitudes, img - coadd)."""
    L = np.ascontiguousarray(coadd, np.float32)
    objs = sep.extract(L, thresh)
    kr, _ = sep.kron_radius(L, objs["x"], objs["y"], objs["a"], objs["b"], objs["theta"], 6.0)
    r = 2.5 * np.maximum(kr, 1.0)
    fc, _, _ = sep.sum_ellipse(L, objs["x"], objs["y"], objs["a"], objs["b"], objs["theta"], r)
    fi, _, _ = sep.sum_ellipse(np.ascontiguousarray(img, np.float32), objs["x"], objs["y"], objs["a"], objs["b"],
                               objs["theta"], r)
    ok = (fc > 0) & (fi > 0)
    mc = -2.5 * np.log10(fc[ok])
    dm = -2.5 * np.log10(fi[ok]) - mc
    return mc, dm


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    """Structural similarity (Wang et al. 2004) with the standard 11 px Gaussian window (sigma 1.5), channel mean for (h, w, 3)."""
    import cv2
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if a.ndim == 2:
        a, b = a[..., None], b[..., None]
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    vals = []
    for c in range(a.shape[-1]):
        x, y = a[..., c], b[..., c]
        mx, my = cv2.GaussianBlur(x, (11, 11), 1.5), cv2.GaussianBlur(y, (11, 11), 1.5)
        sxx = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mx * mx
        syy = cv2.GaussianBlur(y * y, (11, 11), 1.5) - my * my
        sxy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mx * my
        s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
        vals.append(float(s.mean()))
    return float(np.mean(vals))


def paper_metrics(x, coadd):
    """S_F (Krotkov), sigma_sky, PSNR / SSIM against the reference image, and photometry."""
    rng = float(coadd.max() - coadd.min()) or 1.0
    rms = float(np.sqrt(np.mean((coadd - x) ** 2)))
    out = {"S_F": float(np.mean([sharpness(x[..., c]) for c in range(3)])),
           "sigma_sky": float(np.mean([sky_sigma(x[..., c]) for c in range(3)])),
           "psnr_vs_coadd": float(20 * np.log10(rng / max(rms, 1e-12))),
           "ssim_vs_coadd": ssim(coadd, x, rng)}
    try:
        mc, dm = photometry(coadd.mean(-1), x.mean(-1), thresh=5 * sky_sigma(coadd.mean(-1)))
        if len(mc) >= 4:
            bright = mc <= np.percentile(mc, 50)
            out["dm_bright"] = float(np.median(dm[bright]))
            out["dm_faint"] = float(np.median(dm[~bright]))
    except Exception:
        pass
    return out


# ----------------------------------------------------------------------------- stack proxies
def star_fwhm(img, sat, thresh=15.0):
    """Median FWHM (2 x half-light radius) of isolated, unsaturated stars (luminance)."""
    from scipy.spatial import cKDTree
    L = np.ascontiguousarray(img.mean(-1) if img.ndim == 3 else img, np.float32)
    bk = sep.Background(L, bw=64, bh=64)
    sub = L - bk.back()
    o = sep.extract(sub, thresh, err=bk.globalrms, minarea=5)
    if len(o) < 5:
        return float("nan")
    d, _ = cKDTree(np.stack([o["x"], o["y"]], 1)).query(np.stack([o["x"], o["y"]], 1), k=2)
    ok = (o["peak"] < 0.5 * sat) & (o["flag"] == 0) & (d[:, 1] > 12 * np.sqrt(o["a"] * o["b"]))
    if ok.sum() < 5:
        return float("nan")
    r, _ = sep.flux_radius(sub, o["x"][ok], o["y"][ok], 6 * o["a"][ok], 0.5, normflux=o["flux"][ok], subpix=5)
    return float(np.median(2 * r))


def background_noise(img):
    """Robust per-channel noise of the background (sep global rms), channel mean."""
    return float(np.mean([sky_sigma(img[..., c]) for c in range(img.shape[-1])]))


# ----------------------------------------------------------------------------- ground truth
def _plane_fit(diff, mask):
    """Least-squares plane (offset + x, y gradient) of ``diff`` (h, w) over ``mask``."""
    h, w = diff.shape
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    A = np.stack([np.ones(mask.sum()), xx[mask], yy[mask]], 1)
    coef, *_ = np.linalg.lstsq(A, diff[mask], rcond=None)
    return coef[0] + coef[1] * xx + coef[2] * yy


def gaussian_blur(img, sigma):
    import cv2
    if not sigma:
        return img
    return np.stack([cv2.GaussianBlur(img[..., c], (0, 0), sigma) for c in range(img.shape[-1])], -1)


def truth_metrics(est, truth, valid, star_truth=None, stars=None, sigma_eval: float = 1.0,
                  fit_plane: bool = True, ap_radius: float = 3.0, fit_offset: bool = False, fit_scale: bool = True):
    """Comparison of ``est`` with the exact ``truth`` (both (h, w, 3), same grid and units) on
    ``valid`` pixels, after smoothing both with a Gaussian of ``sigma_eval`` pixels (the
    resolution at which they are compared) and, if ``fit_plane``, removing a per-channel
    plane from the difference (the sky background and its gradient are not part of the
    truth).  ``fit_offset`` (with ``fit_plane`` off) removes only a per-channel constant: for
    scoring a gradient removal, whose errors a plane would absorb.

    * nrmse: rms error / rms of the truth's structure (per channel, averaged); lower is better
    * psnr: 20 log10(truth peak / rms error)
    * ssim: SSIM (Wang et al. 2004) of the asinh-stretched images
    * faint_nrmse: nrmse where the truth's stars are negligible (extended emission and sky)
    * star_dmag_median / star_dmag_mad: aperture photometry of isolated true stars
      (``stars``: dict with x, y, flux (per channel) on this grid) vs their true fluxes
    """
    e, t = gaussian_blur(est, sigma_eval), gaussian_blur(truth, sigma_eval)
    if fit_scale:                                     # the Siril stack is in its own linear units: a per-channel gain against the truth (least squares on the valid pixels)
        e = e.copy()
        for c in range(3):
            tv, ev = t[..., c][valid], e[..., c][valid]
            g = float(np.dot(tv - tv.mean(), ev - ev.mean()) / max(np.dot(ev - ev.mean(), ev - ev.mean()), 1e-12))
            e[..., c] = e[..., c] * g if g > 0 else e[..., c]
    out = {}
    nr, ps, fr = [], [], []
    starless = None
    if star_truth is not None:
        st = gaussian_blur(star_truth, sigma_eval).max(-1)
        starless = valid & (st < 0.02 * np.maximum(t.max(-1), 1e-6) + 1e-3 * np.percentile(t, 99.9))
    for c in range(3):
        d = e[..., c] - t[..., c]
        if fit_plane:
            d = d - _plane_fit(d, valid)
        elif fit_offset:
            d = d - float(np.median(d[valid]))
        rms = float(np.sqrt(np.mean(d[valid] ** 2)))
        spread = float(np.std(t[..., c][valid])) or 1.0
        nr.append(rms / spread)
        ps.append(20 * np.log10(max(float(t[..., c][valid].max()), 1e-9) / max(rms, 1e-12)))
        if starless is not None and starless.sum() > 100:
            fr.append(float(np.sqrt(np.mean(d[starless] ** 2))) / (float(np.std(t[..., c][starless])) or 1.0))
        if c == 0:
            dplane = [d]
        else:
            dplane.append(d)
    out["nrmse"] = float(np.mean(nr))
    out["psnr"] = float(np.mean(ps))
    if fr:
        out["faint_nrmse"] = float(np.mean(fr))
    # SSIM in a common asinh stretch set by the truth
    k = max(float(np.percentile(t[valid], 50)) * 0.2, 1e-3 * float(np.percentile(t[valid], 99.9)), 1e-6)
    top = float(np.arcsinh(np.percentile(t[valid], 99.9) / k))
    ec = np.stack([t[..., c] + dplane[c] for c in range(3)], -1)        # plane-corrected estimate
    sa, sb = np.arcsinh(np.maximum(ec, 0) / k) / top, np.arcsinh(np.maximum(t, 0) / k) / top
    ys, xs = np.nonzero(valid)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    out["ssim"] = ssim(np.clip(sb[sl], 0, 1.5), np.clip(sa[sl], 0, 1.5), 1.5)
    if stars is not None and star_truth is not None and len(stars["x"]):
        # photometry of the stars alone: the true extended emission is removed from the
        # estimate, and the estimate is compared with the true stars in the same aperture
        sb_ = gaussian_blur(star_truth, sigma_eval)
        ecl = np.ascontiguousarray((ec - (t - sb_)).mean(-1), np.float32)
        stl = np.ascontiguousarray(sb_.mean(-1), np.float32)
        x, y, f = stars["x"], stars["y"], stars["flux"].mean(-1)
        h, w = valid.shape
        ok = (x > ap_radius + 2) & (x < w - ap_radius - 3) & (y > ap_radius + 2) & (y < h - ap_radius - 3)
        ok &= valid[np.clip(y.astype(int), 0, h - 1), np.clip(x.astype(int), 0, w - 1)]
        if ok.sum() >= 5:
            from scipy.spatial import cKDTree
            # isolated: no other star with >= 10 % of its flux within 4 aperture radii
            tree = cKDTree(np.stack([x, y], 1))
            iso = np.array([all(f[j] < 0.1 * f[i] for j in tree.query_ball_point([x[i], y[i]], 4 * ap_radius) if j != i)
                            if ok[i] else False for i in range(len(x))])
            if iso.sum() >= 5:
                fe, _, _ = sep.sum_circle(ecl, x[iso], y[iso], ap_radius + 2 * sigma_eval, subpix=5)
                ft, _, _ = sep.sum_circle(stl, x[iso], y[iso], ap_radius + 2 * sigma_eval, subpix=5)
                pos = (fe > 0) & (ft > 0)
                if pos.sum() >= 5:
                    dm = -2.5 * np.log10(fe[pos] / ft[pos])
                    out["star_dmag_median"] = float(np.median(dm))
                    out["star_dmag_mad"] = float(1.4826 * np.median(np.abs(dm - np.median(dm))))
                    out["n_stars_phot"] = int(pos.sum())
    return out
