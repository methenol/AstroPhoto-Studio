"""The restore stage's noise-aware strengths (BXT sn, NXT dn, GraXpert strength follow the measured signal-to-noise) and the plate-solve retries
(Siril's online catalogue answers 503 now and then: IC 1805 / NGC 1499, 2026-10-15)."""
from __future__ import annotations

import math
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


def _debt_case(bias=1.0, step=0.3, ramp=False, corner=False, deep=False, chans=(1.0, 1.0, 1.0), single=None, h=600, w=900, sky=0.05, noise=6e-5, seed=5):
    """A flat sky with real structure at the top border, and a gradient step's answer to it.

    ``step`` is that structure, as a percentage of the sky, broad across the outer lattice cell: the step is entitled to take it out and the tool must let it.  ``bias`` is what
    the step invents in the same band instead - the outer cell is extrapolated, the grid having stopped half a cell inside the frame - dying out over the outer 40 % of the cell,
    which is how the misfit measures on real frames (NGC 1499's right border, +2.25 % of the sky, was gone by 71 px of a 251 px cell).  ``ramp`` ends the invented part partway
    along the border, which is how structure the frames themselves carry measures - where the nebula sits the step had nothing to change, and a deviation that dies there must
    be refused (M 31's left border: +0.15 % at one end against its 0.31 % 3 SE).  ``corner`` instead grows it fivefold from one end of the line to the other, one sign the
    whole length, as a misfit does where two extrapolated borders meet (NGC 1499's right border: +1.01 %, +2.46 %, +5.55 %), and must still be taken.  ``deep`` lets the
    invented part stand the whole depth of the cell, which is what the sky does and an extrapolated cell does not; ``chans`` scales it per channel, and ``single`` confines one
    channel's copy to the right half of the line.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    band = math.ceil(w / post.tools.GX_MESH_POINTS)
    shoulder = (step / 100.0) * sky * np.maximum(0.0, 1.0 - yy / band)
    depth = band if deep else 0.4 * band
    src = np.stack([sky + shoulder + rng.normal(0, noise, (h, w)) for _ in range(3)])
    slip = []
    for c in range(3):
        s = (bias * chans[c] / 100.0) * sky * np.maximum(0.0, 1.0 - yy / depth)
        if ramp:
            s = s * np.maximum(0.0, 1.0 - xx / (0.62 * w))                             # nothing of it at all along the far end of the line
        if corner:
            s = s * (1.0 + 4.0 * xx / w)                                                # one sign, fivefold, all the way to the corner
        if single == c:
            s = s * (xx >= 0.5 * w)
        slip.append(s)
    return src.astype(np.float32), (src - shoulder + np.stack(slip)).astype(np.float32), band


def _top_deviation(a, band, w=900):
    """The top border's level over the frame's own interior, as a percentage of it."""
    border = float(np.median(a[:, 0:4, w // 4:3 * w // 4]))
    interior = float(np.median(a[:, 2 * band:4 * band, w // 4:3 * w // 4]))
    return (border - interior) / interior * 100.0


def test_a_gradient_step_may_not_leave_the_border_more_deviant_than_its_input_made_it():
    """The invariant border_debt exists to hold (LBN 429, 2026-10-30): a background model's border cell is extrapolated, and what it leaves there is owed back - while
    everything the frames themselves carried at the border stays exactly where they put it."""
    src, dst, band = _debt_case(bias=1.0, step=0.3)
    assert _top_deviation(src, band) > 0.2 and _top_deviation(dst, band) > 0.7             # the frames carry a shoulder; the step left a bigger one behind
    field, info = post.border_debt(src, dst)
    assert field is not None, info
    after = _top_deviation(dst - field, band)
    assert after <= _top_deviation(src, band) + 0.05, (info, after)                         # no more deviant than the frame that went in, and no less: the shoulder is sky
    assert after < _top_deviation(dst, band) - 0.3, (info, after)                           # the invented part of it is what came away
    assert float(np.abs(field[(slice(0, 3), slice(4 * band, None), slice(0, None))]).max()) == 0.0     # the interior is not this tool's to touch


def test_a_step_that_improved_its_border_is_left_completely_alone():
    """Removing the structure the frames really do carry at the border is the step's job; undoing that would be the tool's own kind of damage."""
    src, dst, band = _debt_case(bias=0.0, step=1.0)
    assert _top_deviation(src, band) > 0.5 and abs(_top_deviation(dst, band)) < 0.1         # the step took a real 1 % border level out
    assert post.border_debt(src, dst)[0] is None, post.border_debt(src, dst)[1]
    assert post.border_debt(src, src)[0] is None, post.border_debt(src, src)[1]             # and a step that did nothing at all leaves nothing behind


def test_the_debt_refuses_a_deviation_that_is_not_along_the_whole_border():
    """Uniformity along the border is the discriminator: an extrapolated lattice cell misfits along the whole side it covers, nebulosity that happens to reach the edge sits
    at one end of it - and taking that would be this pipeline's unrecoverable error."""
    src, dst, band = _debt_case(bias=3.0, step=0.3, ramp=True)
    assert _top_deviation(dst, band) > _top_deviation(src, band) + 0.15                     # an excess really is there, for a tool that looked only at the middle
    field, info = post.border_debt(src, dst)
    assert field is None, info
    assert any(s[-1] == "the deviation is not along the whole border" for s in info["sides"]), info


def test_the_debt_takes_a_border_bigger_than_the_old_cap_when_the_deviation_dies_inside_the_cell():
    """NGC 1499 (2026-10-30): the right border's debt measured +2.25 % of the sky, the 2 % cap refused it, and the export showed that border as -59 % of the display sky.  The
    profile says what the cap could not: the deviation was gone by 71 px of a 251 px cell, so it is the model's own cell and nothing else."""
    src, dst, band = _debt_case(bias=3.0, step=0.3)
    field, info = post.border_debt(src, dst)
    assert field is not None, info
    assert _top_deviation(dst - field, band) < _top_deviation(dst, band) - 2.0, (info, _top_deviation(dst - field, band))
    big, info = post.border_debt(*_debt_case(bias=5.0, step=0.3)[:2])                       # the ceiling is still the last resort
    assert big is None and any(s[-1] == "more sky than a model can have added" for s in info["sides"]), info


def test_the_debt_refuses_a_deviation_that_stands_past_the_models_cell():
    """A deviation still standing in the inner quarter of the band did not come from the extrapolated cell - it came in with the sky, and the sky stays."""
    src, dst, band = _debt_case(bias=1.0, step=0.3, deep=True)
    assert _top_deviation(dst, band) > _top_deviation(src, band) + 0.5                       # a real excess, for a tool that looked only at the depth
    field, info = post.border_debt(src, dst)
    assert field is None, info
    assert any(s[-1] == "the deviation runs past the model's cell" for s in info["sides"]), info


def test_one_refused_channel_refuses_the_side_so_the_border_cannot_turn_colour():
    """The cell is one object measured in three channels; handing back the channels that passed while leaving the one that did not paints the border in the colour that was left
    out, which is worse than the band the correction came to remove."""
    src, dst, band = _debt_case(bias=1.0, step=0.3, single=1)
    assert _top_deviation(dst, band) > _top_deviation(src, band) + 0.5                       # the excess is there in two channels that would pass on their own
    field, info = post.border_debt(src, dst)
    assert field is None, info
    assert any(s[-1] == "the deviation is not along the whole border" for s in info["sides"]), info


def test_a_border_whose_debt_grows_toward_a_corner_is_still_one_object_and_is_taken():
    """The ends are held to agree in kind, not in size: NGC 1499's right border ran +1.01 %, +2.46 % and +5.55 % of the sky from its near corner to the one it shares with the
    bottom border, where two extrapolated borders meet, and refusing it for that spread - as the factor-of-two rule this replaces did - left the export's -59 % band exactly
    where it was."""
    src, dst, band = _debt_case(bias=0.8, step=0.3, corner=True)
    assert _top_deviation(dst, band) > _top_deviation(src, band) + 0.5                     # a real excess, for a tool that looked only at the middle
    field, info = post.border_debt(src, dst)
    assert field is not None, info
    assert not [s for s in info["sides"] if s[0] == 0], info                                # the side was taken, corners and all
    after = _top_deviation(dst - field, band)
    assert after <= _top_deviation(src, band) + 0.05, (info, after)                         # level with what the frames carried, and not a fraction below it


def _star_field(seed=7):
    """A sky with twelve isolated stars of the PSF these frames actually carry (C 49, 2026-11-01: Gaussian sigma
    1.37 px, 99 % of the light inside r=6), 120 px apart so every star's 40-70 px sky annulus is empty."""
    rng = np.random.default_rng(seed)
    H, W = 560, 460
    a = np.full((3, H, W), 0.02, np.float32) + rng.normal(0, 3e-4, (3, H, W)).astype(np.float32)
    pos = []
    for r in range(4):
        for c in range(3):
            x, y = 100 + c * 120, 100 + r * 120
            gy, gx = np.mgrid[-12:13, -12:13]
            g = np.exp(-((gx ** 2 + gy ** 2) / (2 * 1.4 ** 2)))
            for k, amp in enumerate((0.28, 0.30, 0.29)):
                a[k, y - 12:y + 13, x - 12:x + 13] += amp * g
            pos.append((x, y))
    return a.astype(np.float32), pos


def _light(a, pos, r=14):
    """Each star's light above the frame's sky -- an aperture sum, so the test's statistic is not the gate's."""
    sky = np.median(a, axis=(1, 2))[:, None, None]
    return [float((a[:, y - r:y + r + 1, x - r:x + r + 1] - sky).sum()) for x, y in pos]


def test_the_gate_hands_back_a_star_whose_light_the_restoration_deleted():
    """The case the shipped renders show.  Measured against each session's own stack.fits over 60 isolated stars: C 49
    with ImageMM on the stars delivered 0.72 of the star's light, M 31 0.72, C 20 1.10 with the core at 1.35, NGC 1499
    1.13 with the core at 1.36 -- while the same sessions rendered with no star pass kept 0.91-0.95.  A deconvolution
    may move a star's light inward, which is its job; it may not delete it, and the frames are the witness, so the
    star comes back from the reference rather than being tuned away."""
    ref, pos = _star_field()
    lost = ref.copy()
    for x, y in pos:                                             # every star at 70 % of the light the frames took
        lost[:, y - 16:y + 17, x - 16:x + 17] = np.median(ref, axis=(1, 2))[:, None, None] + 0.7 * (
            ref[:, y - 16:y + 17, x - 16:x + 17] - np.median(ref, axis=(1, 2))[:, None, None])
    before, want = _light(lost, pos), _light(ref, pos)
    assert all(b < 0.8 * w for b, w in zip(before, want)), (before, want)      # the defect is really there
    out, info = post.star_flux_guard(lost, ref)
    assert info["mode"] == "population" and info["refused"] == len(pos), info   # a tool doing it to every star
    after = _light(out, pos)
    assert all(abs(a - w) < 0.05 * w for a, w in zip(after, want)), (info, after, want)


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
    want = _light(ref, [(x, y)])[0]
    assert abs(_light(out, [(x, y)])[0] - want) < 0.05 * want       # and it came back
    others = [p for p in pos if p != (x, y)]
    assert all(abs(a - b) < 1e-3 for a, b in zip(_light(out, others), _light(ref, others))), "the other stars moved"


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
    that took 30 % of every star hands them back before the white balance and the star pass can bury the evidence."""
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
    want = [w * r._phot_scale for w in _light(ref, pos)]   # the chain scales the frame to peak 1; both sides on that scale
    after = _light(r.arr, pos)
    assert all(abs(a - w) < 0.05 * w for a, w in zip(after, want)), (gate, after, want)


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


def test_a_refused_star_comes_back_as_the_frames_carried_it():
    """A star the restoration dimmed is delivered exactly as the frames carried it, so the correction does not depend
    on how any measurement chooses to subtract the sky.  On C 49 (ImageMM on both regions) the restoration had dimmed
    the stars *and* darkened the sky around them, which is what makes the pixels the honest thing to check: re-running
    the guard afterwards does not read 1.00, because a disc handed back from ``stack.fits`` sitting in the
    restoration's own darker yardstick cannot look like the frames again -- so this asserts what the frames are
    delivering rather than a number that a change of convention would silently move."""
    ref, pos = _star_field()
    hurt = ref.copy()
    gy, gx = np.mgrid[-80:81, -80:81]
    rr = np.hypot(gx, gy)
    for x, y in pos:
        sl = (slice(y - 80, y + 81), slice(x - 80, x + 81))
        for c in range(3):
            patch = hurt[c][sl]
            patch[rr <= 16] = np.median(ref[c]) + 0.7 * (ref[c][sl][rr <= 16] - np.median(ref[c]))   # the light it took
            patch[(rr > 30) & (rr <= 70)] -= 0.004                                                    # the sky it darkened
            hurt[c][sl] = patch
    out, before = post.star_flux_guard(hurt, ref, align=False)
    assert before["mode"] == "population" and before["refused"] == len(pos), before
    disc = np.hypot(*np.mgrid[-post.GUARD_R_APER:post.GUARD_R_APER + 1,
                              -post.GUARD_R_APER:post.GUARD_R_APER + 1]) <= post.GUARD_R_APER
    for x, y in pos:
        s = post.GUARD_R_APER
        got, want = out[:, y - s:y + s + 1, x - s:x + s + 1], ref[:, y - s:y + s + 1, x - s:x + s + 1]
        assert np.array_equal(got[:, disc], want[:, disc]), (x, y)
    _o, again = post.star_flux_guard(out, ref, align=False)
    assert again["median"] > 1.1, (before, again)          # the yardstick moved with the sky, not with the pixels


def _painted_pair(h=200, w=240, noise=3.5e-4, seed=11, depth=0.0025):
    """A sky carrying its own noise, and the stripe a trail remover paints over it: matte, and below the sky it covers."""
    rng = np.random.default_rng(seed)
    src = np.stack([0.030 + rng.normal(0, noise, (h, w)) for _ in range(3)]).astype(np.float32)
    dst = src.copy()
    dst[:, 60:150, 120:128] = np.float32(0.030 - depth)
    return src, dst


def test_a_matte_patch_from_the_trail_remover_goes_back_to_the_frames():
    """Cosmic Clarity's satellite remover painted a trail out of IC 1848 (2026-11-01) and left the stripe matte: 94 %
    of it at one value, 8.3 % of the sky below its surroundings -- reproduced by running the step alone on that
    session's own ``stack.fits``.  No border tool can see it, because the patch never touches the frame, and the
    stretch's slope at the sky (~120) delivers the pedestal as a black bar: the rectangle the owner reported.  The
    fill is paint rather than a measurement, so the pixels come back to the frames."""
    src, dst = _painted_pair()
    weight, info = post.painted_patch(src, dst)
    assert weight is not None and info["cells"] >= 8, info
    assert info["flat_frac"] >= 0.9 and info["offset_sky_pct"] > 5.0, info
    assert (weight[64:144, 120:128] == 1.0).all(), "the painted pixels themselves go back, not just their neighbourhood"
    assert (weight[:100, :100] == 0.0).all(), "the guard's reach is the patch and its edge, not the frame"


def test_a_trail_removed_at_the_right_level_is_left_alone():
    """The remover is worth having, so the guard has to miss the fills that did no harm: one that leaves the sky where
    it was is invisible after the stretch however flat it is, and the trail it took out was worth more than the texture
    it cost.  And an area that was already matte in the input -- a saturated core, a masked corner, a frame the
    telescope itself left blank -- is the frames' own, not the tool's doing."""
    src, _ = _painted_pair()
    same_level = src.copy()
    same_level[:, 60:150, 120:128] = np.float32(0.030)                      # matte, but the sky it covers untouched
    assert post.painted_patch(src, same_level)[0] is None, "a level fill is the tool doing its job"
    already = src.copy()
    already[:, 60:150, 120:128] = np.float32(0.0275)                        # matte before the tool ever saw it
    shifted = already.copy()
    shifted[:, 60:150, 120:128] = np.float32(0.0250)
    assert post.painted_patch(already, shifted)[0] is None, "the input was already matte there"


def test_a_matte_patch_too_shallow_to_be_proved_is_left_alone():
    """The level test has two floors and this one is the narrower: a cell median carries 1.25 sigma over root(cell),
    so a shift smaller than that is not the tool's, it is the noise's, whatever the fill's own value says.  A patch
    this shallow is invisible after the stretch as well -- which is the point of measuring before correcting."""
    rng = np.random.default_rng(5)
    src = np.stack([0.030 + rng.normal(0, 3.5e-4, (200, 240)) for _ in range(3)]).astype(np.float32)
    dst = src.copy()
    dst[:, 60:150, 120:128] = np.float32(0.030 - 0.00002)                          # 0.07 % of the sky: under the cell's SE
    assert post.painted_patch(src, dst)[0] is None


def test_the_trail_removers_patch_is_judged_on_the_step_itself(tmp_path, monkeypatch):
    """The guard has to run where the paint is applied -- the step's own input against its own output -- or the bar
    ships, and its verdict belongs in the record so an export can be audited after the fact."""
    class Painter:
        def run(self, mode, arr, cli, **kw):
            out = np.array(arr, np.float32)
            out[:, 60:150, 120:128] = np.float32(np.median(arr) * (1.0 - 0.03))   # 3 % of the sky under it, as measured
            return out, {"command": "CosmicClarity_Native.py --mode satellite", "trails_detected": True,
                         "backend": "fake", "device": "cpu"}
    monkeypatch.setattr(post.tools, "cosmic_clarity", Painter())
    r = _restorer(tmp_path, _stack(4e-4))
    r.sp["satellite_removal"] = True                                   # a stack parameter, where the app puts it
    before = r.arr.copy()
    r.satellite()
    rec = [x for x in r.steps.records if x["step"] == "satellite"][-1]
    assert rec["status"] == "done" and rec["params"]["painted_patch"]["cells"] >= 8, rec
    assert np.array_equal(r.arr[:, 60:150, 120:128], before[:, 60:150, 120:128]), "the stripe must be the frames' again"


def test_the_verdict_grows_along_a_fill_that_came_in_thin():
    """The middle of a fill reads as matte; where it came in thin it left the sky's noise behind while still sitting at
    the fill's level, and a cell-by-cell verdict leaves that half painted in the middle of the patch it just gave
    back.  IC 1848's bar had 80 of its 150 rows in exactly that state -- the guard found the top and shipped the
    bottom black."""
    src, dst = _painted_pair()
    rng = np.random.default_rng(4)
    dst[:, 105:150, 120:128] = np.float32(0.030 - 0.0025) + rng.normal(0, 4 * 3.5e-4, (45, 8)).astype(np.float32)
    weight, info = post.painted_patch(src, dst)
    assert weight is not None, info
    assert info["cells_found"] < info["cells"], f"the verdict must reach past the matte core: {info}"
    assert (weight[64:144, 120:128] == 1.0).all(), "the whole stripe, thick end and thin"


def test_a_step_that_moved_the_sky_over_the_frame_is_not_a_patch():
    """The growth walks wherever the step keeps the same direction, so it needs a ceiling: a step that moved the sky
    over a quarter of the frame is a gradient, a background model or the sky, and handing that back would undo the
    step wholesale -- an argument the owner settles in the settings, not one a weight map decides."""
    src, _ = _painted_pair(h=200, w=200)
    dst = src.copy()
    dst[:] = np.float32(0.030 - 0.0025)                                  # the whole frame repainted at one level
    assert post.painted_patch(src, dst)[0] is None
