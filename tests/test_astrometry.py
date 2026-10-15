"""Pure-function tests of astrophoto.astrometry (no network)."""
import math

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from astrophoto import astrometry as A


def test_gnomonic_centre_and_axes():
    xi, eta = A.gnomonic([10.0, 10.0, 10.5], [20.0, 20.5, 20.0], 10.0, 20.0)
    assert abs(xi[0]) < 1e-12 and abs(eta[0]) < 1e-12
    assert abs(xi[1]) < 1e-9 and abs(eta[1] - 0.5) < 1e-3           # north is +eta
    assert xi[2] > 0 and abs(xi[2] - 0.5 * math.cos(math.radians(20))) < 1e-3   # east is +xi


def test_pixel_scale_and_decimal_year():
    assert abs(A.pixel_scale_guess(250.0, 2.9) - 2.393) < 0.01
    assert abs(A.pixel_scale_guess(250.0, 2.9, 2.0) - 1.1965) < 0.01
    assert abs(A.decimal_year("2026-07-02T12:00:00") - 2026.5) < 0.01


def _wcs(parity: int) -> WCS:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [83.8, -5.4]
    w.wcs.crpix = [50.5, 40.5]
    w.wcs.cd = [[-1e-3 * parity, 0], [0, 1e-3]]
    return w


def test_header_wcs_and_parity():
    for parity in (1, -1):
        hdr = _wcs(parity).to_header()
        w = A.header_wcs(hdr, (80, 100))
        assert w is not None and w.pixel_shape == (100, 80)
        assert A.header_wcs(fits.Header(), (80, 100)) is None


def test_bottom_up():
    assert A.bottom_up(fits.Header()) is True
    assert A.bottom_up(fits.Header({"ROWORDER": "TOP-DOWN"})) is False


def test_solve_from_wcs_recovers_matches():
    w = _wcs(1)
    rng = np.random.default_rng(1)
    x, y = rng.uniform(5, 95, 60), rng.uniform(5, 75, 60)
    ra, dec = w.all_pix2world(x, y, 0)
    det = {"x": x + rng.normal(0, 0.1, 60), "y": y + rng.normal(0, 0.1, 60), "fwhm": 3.0}
    sol = A.solve_from_wcs(det, {"ra": ra, "dec": dec, "g": np.full(60, 12.0)}, w, (80, 100))
    assert sol["n_matched"] >= 55 and sol["rms_arcsec"] < 0.8 and abs(sol["scale_arcsec"] - 3.6) < 0.01
