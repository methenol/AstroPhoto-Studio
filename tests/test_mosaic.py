"""The mosaic assembly's bookkeeping and the region-of-interest crop (no Siril: the Siril script itself is covered by the docker tests)."""
from __future__ import annotations

import json
import os
import threading

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from sirilpipe import fitsio, mosaic, stackpost, workflow
from sirilpipe.session import Session


def header(ra, dec, w=1000, h=600, scale=2 / 3600):
    wcs = WCS(naxis=2)
    wcs.wcs.crval, wcs.wcs.crpix, wcs.wcs.ctype = [ra, dec], [w / 2 + 0.5, h / 2 + 0.5], ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cd = [[-scale, 0], [0, scale]]
    hdr = wcs.to_header(relax=True)
    hdr["NAXIS1"], hdr["NAXIS2"] = w, h
    return hdr


MANIFEST = {"project_id": "ngc7000-mosaic", "target": "NGC 7000", "execution": "panels", "overlap": 0.25,
            "footprint": {"ra_deg": 150.0, "dec_deg": 0.0, "width_arcmin": 60.0, "height_arcmin": 18.0},
            "panels": [{"id": "P1", "ra_deg": 149.77, "dec_deg": 0.0, "folders": ["/a"]}, {"id": "P2", "ra_deg": 150.23, "dec_deg": 0.0, "folders": ["/b"]}]}


def test_roi_box_is_centred_even_and_keeps_the_percentage():
    assert stackpost.roi_box(4000, 2000, 25) == (1500, 750, 1000, 500)
    x, y, w, h = stackpost.roi_box(3841, 2161, 33)
    assert w % 2 == 0 and h % 2 == 0 and abs(x - (3841 - w) / 2) <= 1 and abs(y - (2161 - h) / 2) <= 1


def test_roi_box_never_goes_below_the_minimum_side():
    assert stackpost.roi_box(1000, 600, 1)[2:] == (256, 256)
    assert stackpost.roi_box(200, 100, 50)[2:] == (200, 100)
    with pytest.raises(ValueError):
        stackpost.roi_box(100, 100, 0)


def session_with_linear(tmp_path, meta=None):
    s = Session.__new__(Session)
    s.dir = str(tmp_path)
    s.process = str(tmp_path / "process")
    s.cancel_flag, s.pause_flag = threading.Event(), threading.Event()
    arr = np.arange(3 * 400 * 1000, dtype=np.float32).reshape(3, 400, 1000) / 1.2e6
    fitsio.write_rgb(s._p("linear.fits"), arr, header(150.0, 10.0, 1000, 400))
    (tmp_path / "stack_meta.json").write_text(json.dumps(meta or {"crop": [0, 0, 1000, 400]}))
    return s, arr


def test_roi_crops_the_restored_image_around_the_centre_and_moves_the_solution(tmp_path):
    s, arr = session_with_linear(tmp_path)
    (tmp_path / "starless.fits").write_text("stale")
    info = s._apply_roi({"roi_crop_percent": 50})
    out, hdr = fitsio.read_rgb(s._p("linear.fits"))
    x, y, w, h = info["box"]
    assert (w, h) == (500, 256) and (x, y) == (250, 72)
    assert np.array_equal(out, arr[:, y:y + h, x:x + w])
    # the sky position at the centre of the frame is the same before and after the crop
    assert WCS(hdr, naxis=2).wcs_pix2world([[500 - x, 200 - y]], 0)[0] == pytest.approx(WCS(header(150.0, 10.0, 1000, 400), naxis=2).wcs_pix2world([[500, 200]], 0)[0], abs=1e-9)
    assert not (tmp_path / "starless.fits").exists()
    assert s.meta["roi"]["percent"] == 50 and s.meta["crop"] == [0, 0, 1000, 400]      # the stack's own box is unchanged


def test_roi_off_leaves_the_image_alone_and_forgets_an_earlier_crop(tmp_path):
    s, arr = session_with_linear(tmp_path, {"roi": {"percent": 50}})
    for off in ({}, {"roi_crop_percent": 0}, {"roi_crop_percent": 100}, {"roi_crop_percent": "x"}):
        assert s._apply_roi(off) is None
    assert np.array_equal(fitsio.read_rgb(s._p("linear.fits"))[0], arr) and "roi" not in s.meta


def test_manifest_is_checked():
    assert mosaic.normalise_manifest(MANIFEST)["execution"] == "panels"
    for bad in ({**MANIFEST, "project_id": ""}, {**MANIFEST, "panels": []}, {**MANIFEST, "execution": "x"}, {**MANIFEST, "footprint": {"ra_deg": 1}},
                {**MANIFEST, "panels": [MANIFEST["panels"][0], MANIFEST["panels"][0]]}, {**MANIFEST, "execution": "native"}):
        with pytest.raises(ValueError):
            mosaic.normalise_manifest(bad)


def test_finishing_params_drop_what_a_mosaic_cannot_use():
    sp = mosaic.finishing_params({"deconv_method": "imagemm", "deconv_stars_method": "bxt", "roi_crop_percent": 20, "mosaic": {}, "hypertune": True})
    assert sp == {"deconv_method": "auto", "deconv_stars_method": "bxt"}
    assert mosaic.panel_stack_params({"scale": 2.0, "autofinish": True})["scale"] == 1.0


def solved(*centres):
    return {k: (WCS(header(ra, dec), naxis=2), (600, 1000)) for k, (ra, dec) in centres}


def test_coverage_of_a_complete_footprint():
    m = mosaic.normalise_manifest(MANIFEST)
    cov = mosaic.measure_coverage(m, solved(("P1", (149.77, 0.0)), ("P2", (150.23, 0.0))))
    assert cov == {"percent": 100.0, "revisit": []}


def test_a_hole_names_the_panel_to_shoot_again():
    m = mosaic.normalise_manifest(MANIFEST)
    cov = mosaic.measure_coverage(m, solved(("P1", (149.77, 0.0))))
    assert cov["percent"] < 70 and cov["revisit"] == ["P2"]


def test_canvas_estimate_matches_the_projection():
    m = mosaic.normalise_manifest(MANIFEST)
    est = mosaic.canvas_estimate(m, solved(("P1", (149.77, 0.0)), ("P2", (150.23, 0.0))))
    assert abs(est["height"] - 600) <= 2 and abs(est["width"] - (1000 + 0.46 / (2 / 3600))) <= 2 and est["gigabytes"] > 0


def test_panels_script_has_the_recipe_and_uses_the_local_catalogue():
    text = workflow.mosaic_panels_script("/w", "panel", 60, 4, ['set "core.catalogue_gaia_astro=/c"'])
    assert "seqplatesolve panel -catalog=localgaia\n" in text and "seqapplyreg panel -framing=max" in text and "-force" not in text
    assert all(f in text for f in ("rej n 3 3", "-maximize", "-overlap_norm", "-feather=60", "-norm=addscale"))


class FakePanel:
    """A panel's Session: counts the stacks it is asked for."""
    runs = 0
    fail = False

    def __init__(self, folders, work):
        self.dir = work

    def _p(self, *n):
        return os.path.join(self.dir, *n)

    @property
    def meta(self):
        return json.load(open(self._p("stack_meta.json"))) if os.path.exists(self._p("stack_meta.json")) else {}

    def run_analysis(self, *a):
        pass

    def run_stack(self, sp, prog):
        if FakePanel.fail:
            raise RuntimeError("only 2 usable frames\nmore")
        FakePanel.runs += 1
        hdr = header(149.77, 0.0)
        del hdr["NAXIS1"], hdr["NAXIS2"]
        fits.PrimaryHDU(np.zeros((3, 600, 1000), np.float32), hdr).writeto(self._p("stack.fits"), overwrite=True)
        json.dump({"total_exposure": 1500.0, "exptime": 30.0, "n_frames": 50, "n_total": 60, "plate_solve": {"ok": True}}, open(self._p("stack_meta.json"), "w"))

    def plate_solve(self, prog):
        pass


@pytest.fixture
def panel_dir(tmp_path):
    FakePanel.runs, FakePanel.fail = 0, False
    subs = tmp_path / "subs"
    subs.mkdir()
    (subs / "a_0001.fits").write_bytes(b"x" * 10)
    host = Session.__new__(Session)
    host.dir = str(tmp_path / "s")
    return host, {"id": "P1", "ra_deg": 149.77, "dec_deg": 0.0, "folders": [str(subs)]}, subs


def test_panel_is_reused_until_its_subs_change(panel_dir):
    host, panel, subs = panel_dir
    rep, sol, stack, _ = mosaic.stack_panel(host, panel, {}, None, "P1", FakePanel)
    assert rep["accepted_sec"] == 1500.0 and rep["input_sec"] == 1800.0 and rep["solved"] and not rep["reused"] and sol
    assert rep["offset_arcmin"] == pytest.approx(0.0, abs=0.01) and os.path.exists(stack)
    rep2, *_ = mosaic.stack_panel(host, panel, {}, None, "P1", FakePanel)
    assert rep2["reused"] and FakePanel.runs == 1
    (subs / "a_0002.fits").write_bytes(b"y" * 10)                       # a new night's subs
    rep3, *_ = mosaic.stack_panel(host, panel, {}, None, "P1", FakePanel)
    assert not rep3["reused"] and FakePanel.runs == 2


def test_a_failed_panel_is_a_hole_not_a_failed_mosaic(panel_dir):
    host, panel, _ = panel_dir
    FakePanel.fail = True
    rep, sol, stack, meta = mosaic.stack_panel(host, panel, {}, None, "P1", FakePanel)
    assert rep["status"] == "failed" and rep["error"] == "only 2 usable frames" and sol is None and stack is None and meta is None
