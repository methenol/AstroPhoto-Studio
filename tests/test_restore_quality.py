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
