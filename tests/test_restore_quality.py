"""The restore stage's noise-aware strengths (BXT sn, NXT dn, GraXpert strength follow the measured signal-to-noise) and the plate-solve retries
(Siril's online catalogue answers 503 now and then: IC 1805 / NGC 1499, 2026-10-15)."""
from __future__ import annotations

import os
import sys

import cv2
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
    assert tools_used == ["siril rl"] and r.steps.records[-1]["status"] == "done"


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
                        {"method": "reference", "gradient_scale_px": 1024.0, "coverage": 1.0, "gradient_range": 1.0, "credit": "",
                         "survey_coefficients": {n: [0.0, 0.0, 0.0] for n, _ in skyref.SURVEYS},
                         "_emission": np.zeros((h // skyref.FACTOR, w // skyref.FACTOR), np.float32), "_emission_block": skyref.FACTOR},
                        np.zeros((h // skyref.FACTOR, w // skyref.FACTOR, 3), np.float32)))
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


def _star_field(seed=7):
    """A sky with twelve isolated stars (Gaussian sigma 1.4 px) of a range of brightnesses, 120 px apart so every star's sky annulus is empty."""
    rng = np.random.default_rng(seed)
    H, W = 560, 460
    a = np.full((3, H, W), 0.02, np.float32) + rng.normal(0, 3e-4, (3, H, W)).astype(np.float32)
    pos = []
    for r in range(4):
        for c in range(3):
            x, y = 100 + c * 120, 100 + r * 120
            gy, gx = np.mgrid[-12:13, -12:13]
            g = np.exp(-((gx ** 2 + gy ** 2) / (2 * 1.4 ** 2)))
            f = 1.0 / (1.0 + 0.4 * len(pos))                       # brightest first, the faintest a fifth of it
            for k, amp in enumerate((0.28, 0.30, 0.29)):
                a[k, y - 12:y + 13, x - 12:x + 13] += f * amp * g
            pos.append((x, y))
    return a.astype(np.float32), pos


def _light(a, pos, r=14):
    """Each star's light above the frame's sky -- an aperture sum, so the test's statistic is not the gate's."""
    sky = np.median(a, axis=(1, 2))[:, None, None]
    return [float((a[:, y - r:y + r + 1, x - r:x + r + 1] - sky).sum()) for x, y in pos]


def test_the_gate_reports_every_star_whose_light_the_restoration_deleted():
    """A deconvolution may move a star's light inward, which is its job; it may not delete it.  The gate measures and records
    it - it never pastes the reference's pixels back."""
    ref, pos = _star_field()
    lost = ref.copy()
    for x, y in pos:                                             # every star at 70 % of the light the frames took
        lost[:, y - 16:y + 17, x - 16:x + 17] = np.median(ref, axis=(1, 2))[:, None, None] + 0.7 * (
            ref[:, y - 16:y + 17, x - 16:x + 17] - np.median(ref, axis=(1, 2))[:, None, None])
    before, want = _light(lost, pos), _light(ref, pos)
    assert all(b < 0.8 * w for b, w in zip(before, want)), (before, want)      # the defect is really there
    out, info = post.star_flux_guard(lost, ref)
    assert info["mode"] == "population" and info["refused"] == len(pos), info   # a tool doing it to every star
    assert out is lost


def test_the_gate_leaves_a_star_alone_when_the_restoration_only_moved_its_light():
    """A true deconvolution is what the step is for, and it is not this gate's business: the same light in a narrower
    profile is the whole point.  Refusing those stars would leave the pipeline delivering the unsharpened stack and
    calling it restoration."""
    ref, pos = _star_field()
    sharpened = ref.copy()
    sky = np.median(ref, axis=(1, 2))[:, None, None]
    gy, gx = np.mgrid[-20:21, -20:21]
    wide = np.exp(-((gx ** 2 + gy ** 2) / (2 * 1.4 ** 2)))
    narrow = np.exp(-((gx ** 2 + gy ** 2) / (2 * 1.0 ** 2)))
    for x, y in pos:
        for k in range(3):
            patch = ref[k, y - 20:y + 21, x - 20:x + 21] - sky[k, 0, 0]
            scale = float(patch.sum()) / float(narrow.sum())     # the star's own light, in a tighter profile
            sharpened[k, y - 20:y + 21, x - 20:x + 21] = sky[k, 0, 0] + narrow * scale
    assert sharpened[0].max() > 1.2 * ref[0].max()                              # the core really was sharpened...
    assert abs(sum(_light(sharpened, pos)) - sum(_light(ref, pos))) < 0.02 * sum(_light(ref, pos))  # ...and the light kept
    out, info = post.star_flux_guard(sharpened, ref)
    assert info["refused"] == 0, info                              # reshaped, not robbed: nothing to hand back
    assert out is sharpened


def test_the_gate_refuses_the_star_whose_one_channel_was_erased():
    """The coloured crown: the saturated-core rewrite anchors every channel on green, so where green alone collapses
    the star is written with green's collapse (C 49 star (179,726): 0.080 of the stack's green against red and blue
    intact).  Summed over the channels that is a 30 % loss the population would shrug off, so a single channel moved
    by twice the tolerance refuses the star on its own."""
    ref, pos = _star_field()
    x, y = pos[5]
    hurt = ref.copy()
    hurt[1, y - 16:y + 17, x - 16:x + 17] = np.median(ref[1]) + 0.2 * (
        ref[1, y - 16:y + 17, x - 16:x + 17] - np.median(ref[1]))
    out, info = post.star_flux_guard(hurt, ref)
    assert info["mode"] == "individual" and info["refused"] == 1, info
    assert [w[:2] for w in info["worst"]] == [[x, y]] and info["worst"][0][2] < 0.85, info
    assert out is hurt


def test_the_photometry_verdict_reaches_the_restore_record_without_joining_the_tools(tmp_path, fake_rc, monkeypatch):
    """``restore_meta.json`` is how an export is audited after the fact, so the gate's numbers have to be there; and the
    step's records are the list of tools that ran -- ``save`` writes one FITS HISTORY line per entry and the lab prints
    ``tools`` with " + " -- so a measurement must not appear in it."""
    monkeypatch.setattr(post.Restorer, "image_fwhm", lambda self: 4.5)
    r = _restorer(tmp_path, _star_field()[0])
    r.deconvolve()
    meta = r.save()
    gate = meta["star_photometry"]["deconvolution"]
    assert gate and gate["stars"] >= 8 and gate["refused"] == 0 and gate["align"] is True, gate
    assert not [x for x in meta["steps"] if x["tool"] == "photometry gate"], meta["steps"]
    assert "photometry gate" not in meta["tools"], meta["tools"]


def test_the_photometry_gate_can_be_turned_off_and_says_so(tmp_path, fake_rc, monkeypatch):
    """A paid tool's output the owner wants delivered untouched gets an opt-out, in the codebase's idiom, and the record
    says the check was declined rather than staying silent about ever having run."""
    monkeypatch.setattr(post.Restorer, "image_fwhm", lambda self: 4.5)
    r = _restorer(tmp_path, _star_field()[0])
    r.sp["star_photometry_guard"] = False
    r.deconvolve()
    assert r.save()["star_photometry"] == {"deconvolution": {"skipped": "star_photometry_guard=false"}}


def test_a_pedestal_is_never_read_as_star_light(tmp_path):
    """The scale a gate infers from two sky medians cannot tell a per-channel pedestal (background subtraction,
    ``finish.neutralize_background``) from a per-channel gain (white balance, SPCC): a star's light is an excess over
    its own local sky, so the first cancels in it and the second does not.  Measured on C 20 2026-11-01, the colour
    fallback's offsets (0.034/0/0.033 over gains 1.777/1.0/1.733) made a sky-aligned gate report every one of 752
    stars carrying 1.39 of the light the frames took and refuse the whole frame.  The restoration stage therefore
    compares arrays that already share units, and this is what that buys."""
    ref, pos = _star_field()
    pedestal = (ref - np.array([0.012, 0.0, 0.009])[:, None, None]).astype(np.float32)
    out, strict = post.star_flux_guard(pedestal, ref, align=False)
    assert strict["mode"] == "individual" and strict["refused"] == 0 and abs(strict["median"] - 1.0) < 0.02, strict
    assert out is pedestal
    _o, aligned = post.star_flux_guard(pedestal, ref, align=True)
    assert aligned["median"] > 1.2, aligned                          # the same pixels, judged the wrong way


def test_the_restoration_is_judged_when_it_is_adopted(tmp_path, monkeypatch):
    """ImageMM's output is fitted to the stack window per channel (``imagemm_stage.match_to_stack``), so that is the
    one moment the frames' light can be read off both arrays without a scale to guess.  Judged there, a restoration
    that took 30 % of every star is recorded before the white balance and the star pass can bury the evidence."""
    d = tmp_path / "s"
    d.mkdir()
    ref, pos = _star_field()
    lost = ref.copy()
    for x, y in pos:
        lost[:, y - 16:y + 17, x - 16:x + 17] = np.median(ref, axis=(1, 2))[:, None, None] + 0.7 * (
            ref[:, y - 16:y + 17, x - 16:x + 17] - np.median(ref, axis=(1, 2))[:, None, None])
    fitsio.write_rgb(d / "stack.fits", ref.astype(np.float32))
    fitsio.write_rgb(d / "imagemm.fits", lost.astype(np.float32))
    mask = np.zeros(ref.shape[1:], np.float32)
    for x, y in pos:
        mask[y - 6:y + 7, x - 6:x + 7] = 1.0                          # the stars' footprint, as the join wants it
    monkeypatch.setattr(post, "star_footprint_mask", lambda work, steps, arr, header: (mask, {"stars": len(pos)}))
    r = post.Restorer(str(d), {"filter": "Duo-Band", "restore_source": "imagemm.fits",
                               "imagemm_r": 1, "imagemm_offset": [0, 0]},
                      {"deconv_method": "none", "deconv_stars_method": "imagemm", "stack_weighting": "wfwhm"},
                      dict(post.DEFAULTS), None, None)
    gate = r.info["star_photometry"]["restore"]
    assert gate["mode"] == "population" and gate["refused"] >= 8 and gate["align"] is False, gate


def test_a_denoiser_that_eats_the_stars_is_asked_again_weaker(tmp_path, monkeypatch):
    """A denoiser has no reason to touch a star's light -- the stack already averaged that noise away -- and on the
    fleet it did: C 49's linear went from 1.03 to 0.87 of the frames' light through this step alone.  The answer is the
    finisher's ladder, run the other way: the same tool asked again, weaker, on the same input, and the first candidate
    inside tolerance ships -- not a patch of unsmoothed stack dropped into a denoised sky."""
    monkeypatch.setattr(post.tools, "rcastro", FakeRC(licensed=()))     # no NXT: GraXpert stands in, as it does unlicensed
    calls = []

    class Eater:
        def denoise(self, src, dst, strength=0.5, **kw):
            calls.append(strength)
            arr, hdr = fitsio.read_rgb(src)
            sky = np.median(arr, axis=(1, 2))[:, None, None]
            fitsio.write_rgb(dst, sky + (arr - sky) * (1.0 - 0.5 * strength ** 2), hdr)   # dimmer the harder it denoises
            return {"model": "3.0.2", "providers": "cpu"}

    monkeypatch.setattr(post.tools, "graxpert", Eater())
    ref, pos = _star_field()
    r = _restorer(tmp_path, ref)
    want = _light(r.arr, pos)
    r.denoise()
    gate = r.save()["star_photometry"]["denoise"]
    assert len(calls) >= 2 and gate["kept"] < gate["strength"] and gate["kept"] > 0.0, (calls, gate)
    assert calls[-1] == gate["kept"], (calls, gate)                     # the candidate that shipped was the last one run
    rec = [x for x in r.steps.records if x["tool"] == "graxpert"][-1]
    assert rec["params"]["strength"] == gate["kept"], rec               # the log describes the picture that shipped
    after = _light(r.arr, pos)
    assert all(abs(a - w) <= 0.05 * w for a, w in zip(after, want)), (gate, after, want)




def test_join_stars_takes_the_whole_star_and_keeps_the_base_images_sky():
    """NGC 1499 2026-11-03: ImageMM's objects joined to the stack's stars at the stars' footprints left a blurred disc in a dark
    annulus round every star (the deconvolution had moved the wing into a core that was discarded) and, taken whole to the wing's
    reach, the stack's noisier sky came with it.  The join: the other image's star whole inside the footprint, its smooth wing out
    to where the images agree, on the base image's sky (its background offset removed)."""
    rng = np.random.default_rng(7)
    C, N = 3, 241
    yy, xx = np.mgrid[:N, :N].astype(np.float64)
    r = np.hypot(xx - 120, yy - 120)
    wide = 0.5 * (1 + (r / 3.0) ** 2) ** -1.6                          # the stack's star: a Moffat with a long wing
    narrow = np.exp(-r ** 2 / (2 * 0.8 ** 2)) * float(wide.sum()) / (2 * np.pi * 0.8 ** 2)
    base = (0.10 + 0.0004 + narrow + rng.normal(0, 0.0005, (C, N, N))).astype(np.float32)   # deconvolved, quieter, sky 4e-4 higher
    other = (0.10 + wide + rng.normal(0, 0.0010, (C, N, N))).astype(np.float32)
    disc = r <= 12.0
    footprint = np.clip(cv2.GaussianBlur(disc.astype(np.float32), (0, 0), 1.0), 0, 1)
    joined, mask, info = post.join_stars(base, other, footprint, [(120.0, 120.0, 12.0)], disc, 1.0)
    sky_b = float(np.median(base[1][r > 100]))
    for r0 in (14, 20, 30, 45):
        ring = (r >= r0) & (r < r0 + 2)
        got = float(np.median(joined[1][ring])) - sky_b
        want = float(np.median(wide[ring]))
        assert abs(got - want) < 0.15 * want + 0.0002, (r0, got, want)       # the stack's wing, on the base image's sky
    far = r > 100
    assert abs(float(np.median(joined[1][far])) - sky_b) < 1e-4                # the base image's sky level
    ann = (r > 20) & (r < 40)
    hp = joined[1] - cv2.GaussianBlur(joined[1], (0, 0), 3)
    assert 1.4826 * float(np.median(np.abs(hp[ann]))) < 1.4 * 1.4826 * float(np.median(np.abs((base[1] - cv2.GaussianBlur(base[1], (0, 0), 3))[ann])))
    assert info["extension_max"] > 1.5                                          # followed past the footprint
