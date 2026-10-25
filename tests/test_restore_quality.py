"""The restore stage's noise-aware strengths (BXT sn, NXT dn, GraXpert strength follow the measured signal-to-noise) and the plate-solve retries
(Siril's online catalogue answers 503 now and then: IC 1805 / NGC 1499, 2026-10-15)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from sirilpipe import finish, fitsio, post                       # noqa: E402
from sirilpipe.siril import SirilError                           # noqa: E402
from sirilpipe.tools import ToolUnavailable                      # noqa: E402


def _stack(amp, h=360, w=480, noise=3e-4, seed=3):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    neb = amp * np.exp(-(((xx - w / 2) / 80) ** 2 + ((yy - h / 2) / 60) ** 2))
    return np.stack([0.02 + neb * k + rng.normal(0, noise, (h, w)) for k in (1.0, 0.5, 0.5)]).astype(np.float32)


class FakeRC:
    """rcastro stand-in: records the parameters, returns the image unchanged."""

    def __init__(self, licensed=("bxt", "nxt")):
        self.licensed, self.calls = licensed, []

    def require(self, key):
        if key not in self.licensed:
            raise ToolUnavailable(f"rc-astro {key}", "not licensed", 77)

    def run(self, product, src, dst, **params):
        self.calls.append((product, {k: v for k, v in params.items() if k not in ("device", "progress", "checkpoint")}))
        arr, hdr = fitsio.read_rgb(src)
        fitsio.write_rgb(dst, arr, hdr)
        return {"tool": f"rc-astro {product}", "flags": [f"--{k}" for k in params]}


def _restorer(tmp_path, arr, params=None, header=None):
    d = tmp_path / "s"
    d.mkdir()
    hdr = {"ROWORDER": "BOTTOM-UP", "FILTER": "Duo-Band", **(header or {})}
    fitsio.write_rgb(str(d / "stack.fits"), arr, hdr)
    p = {**post.DEFAULTS, **(params or {})}
    return post.Restorer(str(d), {"filter": "Duo-Band"}, {}, p, None, None)


@pytest.fixture()
def fake_rc(monkeypatch):
    rc = FakeRC()
    monkeypatch.setattr(post.tools, "rcastro", rc)
    return rc


def test_bxt_and_nxt_strengths_follow_the_signal_to_noise(tmp_path, fake_rc):
    for name, amp in (("noisy", 4e-4), ("clean", 8e-3)):
        d = tmp_path / name
        d.mkdir()
        r = _restorer(d, _stack(amp))
        fake_rc.calls.clear()
        r.deconvolve()
        r.denoise()
        bxt = next(c for c in fake_rc.calls if c[0] == "bxt")[1]
        nxt = next(c for c in fake_rc.calls if c[0] == "nxt")[1]
        if name == "noisy":
            assert bxt["sn"] == pytest.approx(0.5, abs=0.03) and nxt["dn"] >= 0.75 and 0.25 <= bxt["ss"] <= 0.35        # research/pro-workflow.md A5 / A6
            snr_noisy = r.snr()
        else:
            assert 0.55 <= bxt["sn"] <= 0.6 and nxt["dn"] <= 0.6
            assert r.snr() > snr_noisy
        assert r.info["snr"] == pytest.approx(r.snr(), abs=0.01)


def test_the_denoise_setting_scales_the_ai_strength_and_zero_skips_it(tmp_path, fake_rc):
    r = _restorer(tmp_path, _stack(4e-4), {"denoise": 0.475})
    r.denoise()
    assert fake_rc.calls[-1][1]["dn"] == pytest.approx(0.4, abs=0.01)                      # half the setting's default: half the SNR-derived strength
    (tmp_path / "b").mkdir()
    r2 = _restorer(tmp_path / "b", _stack(4e-4), {"denoise": 0.0})
    n = len(fake_rc.calls)
    r2.denoise()
    assert len(fake_rc.calls) == n and r2.steps.records[-1]["tool"] == "none"


def test_graxpert_stands_in_for_nxt_with_the_snr_strength(tmp_path, monkeypatch):
    monkeypatch.setattr(post.tools, "rcastro", FakeRC(licensed=()))
    seen = {}

    class FakeGX:
        def denoise(self, src, dst, strength=0.5, **kw):
            seen["strength"] = strength
            arr, hdr = fitsio.read_rgb(src)
            fitsio.write_rgb(dst, arr, hdr)
            return {"model": "3.0.2", "providers": "cpu"}

    monkeypatch.setattr(post.tools, "graxpert", FakeGX())
    r = _restorer(tmp_path, _stack(4e-4))
    r.denoise()
    assert seen["strength"] >= 0.75 and r.steps.records[-1]["tool"] == "graxpert" and r.steps.records[-1]["params"]["snr"] > 0


def test_plate_solve_retries_the_catalogue_then_gaia(tmp_path, monkeypatch):
    monkeypatch.setattr(post, "PLATESOLVE_PAUSE", 0.0)
    r = _restorer(tmp_path, _stack(4e-4), header={"FOCALLEN": 150.0, "XPIXSZ": 2.0})
    bodies = []

    def fake(work, steps, name, arr, header, body, stage, **kw):
        bodies.append(body)
        if len(bodies) < 3:
            raise SirilError("Siril stage 'Plate solving' failed (script execution failed)")
        hdr = header.copy()
        for k, v in (("CTYPE1", "RA---TAN"), ("CTYPE2", "DEC--TAN"), ("CRVAL1", 38.1), ("CRVAL2", 61.4), ("CRPIX1", 10.0), ("CRPIX2", 10.0), ("CD1_1", 1e-4), ("CD2_2", 1e-4)):
            hdr[k] = v
        return arr, hdr

    monkeypatch.setattr(post, "_siril_image", fake)
    assert r.plate_solve() is True and post._has_wcs(r.header)
    assert [b.split("-catalog=")[1] for b in bodies] == ["nomad", "gaia", "nomad"]            # NOMAD, Gaia DR3, NOMAD again after the pause
    assert all("-focal=150" in b and "-pixelsize=2" in b for b in bodies)
    recs = [x for x in r.steps.records if x["step"] == "platesolve"]
    assert [x["status"] for x in recs] == ["failed", "failed", "done"] and recs[-1]["params"]["attempt"] == 3


def test_plate_solve_reports_failure_only_when_every_attempt_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(post, "PLATESOLVE_PAUSE", 0.0)
    r = _restorer(tmp_path, _stack(4e-4), header={"FOCALLEN": 150.0, "XPIXSZ": 2.0})
    monkeypatch.setattr(post, "_siril_image", lambda *a, **k: (_ for _ in ()).throw(SirilError("Server unreachable")))
    assert r.plate_solve() is False
    assert len([x for x in r.steps.records if x["step"] == "platesolve" and x["status"] == "failed"]) == 4


def test_deconv_method_rl_uses_siril_richardson_lucy_not_bxt(tmp_path, fake_rc, monkeypatch):
    """The n2n-rl profile / an experiment with deconvolution "rl": BXT is skipped (even when licensed) and Siril's RL
    runs.  The skip raised ToolUnavailable with one argument - a TypeError that failed every such restoration."""
    bodies = []

    def fake(work, steps, name, arr, header, body, stage, **kw):
        bodies.append(body)
        return arr, header

    monkeypatch.setattr(post, "_siril_image", fake)
    d = tmp_path / "s"
    d.mkdir()
    fitsio.write_rgb(str(d / "stack.fits"), _stack(4e-4), {"ROWORDER": "BOTTOM-UP", "FILTER": "Duo-Band"})
    r = post.Restorer(str(d), {"filter": "Duo-Band"}, {"deconv_method": "rl"}, dict(post.DEFAULTS), None, None)
    r.deconvolve()
    assert not any(c[0] == "bxt" for c in fake_rc.calls)
    assert any(body.startswith("set core.fftw_timelimit=0") and "rl -iters=" in body for body in bodies)
    tools_used = [x["tool"] for x in r.steps.records if x["step"] == "deconvolution"]
    assert tools_used == ["rc-astro bxt", "siril rl"] and r.steps.records[-1]["status"] == "done"


def test_survey_gradient_keeps_the_sky_pedestal_and_clips_nothing(tmp_path, monkeypatch):
    """skyref's model carries the whole sky pedestal: subtracting all of it put the sky on 0 and _sanitize zeroed the lower half of its noise
    (M 42 2026-10-24: 44 % of the values, pixel noise 1.72e-4 -> 1.01e-4), which NXT turned into blocks.  The shape goes, the pedestal stays."""
    from sirilpipe import skyref
    r = _restorer(tmp_path, _stack(0.004))
    r.header["CTYPE1"], r.header["CTYPE2"] = "RA---TAN", "DEC--TAN"
    h, w = r.arr.shape[1:]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ped = float(np.median(finish.luminance(r.arr)))                  # the sky at the Restorer's scale (normalise_scale)
    g = 0.1 * ped * (xx / w)
    grad = ped + g                                                    # pedestal + a linear gradient, the same in the 3 channels
    monkeypatch.setattr(post, "_has_wcs", lambda hdr: True)
    monkeypatch.setattr(post.Restorer, "_survey_gradient", lambda self: (np.repeat(grad[..., None], 3, -1),
                        {"method": "reference", "degree": 3, "coverage": 1.0, "gradient_range": 1.0, "survey_coefficients": {}, "credit": "",
                         "_emission": np.zeros((h // skyref.FACTOR, w // skyref.FACTOR), np.float32), "_emission_block": skyref.FACTOR}))
    noise_in = finish.lower_half_sigma(finish.sample(finish.luminance(r.arr)))
    r.arr = (r.arr + g[None]).astype(np.float32)
    r.background()
    rec = [x for x in r.steps.records if x["step"] == "gradient"][-1]
    assert rec["status"] == "done" and rec["params"]["clipped_fraction"] == 0.0
    assert int((r.arr == 0).sum()) == 0
    sky = finish.luminance(r.arr)
    assert abs(float(np.median(sky)) / ped - 1) < 0.05               # the pedestal kept
    assert abs(finish.lower_half_sigma(finish.sample(sky)) / noise_in - 1) < 0.1     # the noise untouched
    assert abs(float(np.median(sky[:, : w // 8])) - float(np.median(sky[:, -w // 8:]))) < 0.1 * float(g.max())      # the gradient gone


def test_reduced_wcs_keeps_the_pixel_scale_of_a_pc_cdelt_header():
    """A PC + CDELT plate solution (Siril 1.4): get_pc() alone asked hips2fits for ~6 degrees per reduced pixel."""
    from astropy.wcs import WCS
    from sirilpipe import skyref
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [83.82, -5.38]
    w.wcs.crpix = [1850.5, 904.5]
    w.wcs.cdelt = [-0.000766, 0.000766]
    w.wcs.pc = [[0.8744, 0.4851], [-0.4850, 0.8745]]
    red = skyref.reduced_wcs(w, 8)
    assert np.allclose(red.pixel_scale_matrix, w.pixel_scale_matrix * 8, rtol=1e-9)
    # the same sky position at the same (reduced) place
    ra, dec = w.wcs_pix2world([[400.0, 300.0]], 0)[0]
    x, y = red.wcs_world2pix([[ra, dec]], 0)[0]
    assert abs(x - (400.0 + 0.5) / 8 + 0.5) < 1e-6 and abs(y - (300.0 + 0.5) / 8 + 0.5) < 1e-6
