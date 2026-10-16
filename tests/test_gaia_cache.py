"""The per-field cache of the Gaia DR3 SPCC catalogue: chunk lookup, status, and SPCC switching to the cached chunk."""
import os

import numpy as np
import pytest
from astropy.io import fits

from sirilpipe import post, tools

pytest.importorskip("astropy_healpix")


def test_field_chunks_match_sirils_region_lists():
    # Siril_Catalog_Installer.py: "Summer Triangle" = [9, 12, 13, 14, 15, 29, 31], "Orion to Taurus" = [0, 1, 6, 20, 21, 22, 23]
    assert set(tools.healpix1_pixels(312.8, 31.2, 2.5)) <= {9, 12, 13, 14, 15, 29, 31}          # C 33, Cygnus
    assert set(tools.healpix1_pixels(83.8, -5.4, 2.5)) <= {0, 1, 6, 20, 21, 22, 23}             # M 42
    assert tools.healpix1_pixels(312.8, 31.2, 0.1) == [12] and tools.healpix1_pixels(38.2, 61.5, 0.1) == [3]


def test_status_clear_and_covers(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    g = tools.GaiaCatalog()
    assert g.status()["installed"] == 0 and not g.covers([3])
    os.makedirs(g.dir())
    for n in (3, 7):
        with open(g.chunk_path(n), "wb") as f:
            f.write(b"x" * 10)
    st = g.status()
    assert st["chunks"] == [3, 7] and st["bytes"] == 20 and g.covers([3, 7]) and not g.covers([3, 8])
    assert g.clear() == {"removed": 2} and g.status()["installed"] == 0


def test_field_circle_from_wcs():
    h = fits.Header()
    h["CRVAL1"], h["CRVAL2"] = 312.8, 31.2
    h["CD1_1"], h["CD1_2"], h["CD2_1"], h["CD2_2"] = -0.001, 0.0, 0.0, 0.001
    ra, dec, r = post.field_circle(h, (2000, 4000))
    assert (ra, dec) == (312.8, 31.2) and r == pytest.approx(0.5 * 0.001 * np.hypot(2000, 4000))
    with pytest.raises(KeyError):
        post.field_circle(fits.Header(), (10, 10))
