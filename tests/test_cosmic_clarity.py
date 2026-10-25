"""Cosmic Clarity (siril-scripts CosmicClarity_Native.py) as the optional path next to the RC-Astro tools, and the Hubble palette script on dual-band data:
the stack / restore / star-removal / finishing decisions, the plate solution of a super-resolved image, the model inventory and the palette commands."""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from sirilpipe import fitsio, post, pro, tools, workflow
from sirilpipe.tools import ToolUnavailable

RNG = np.random.default_rng(11)


# ----------------------------------------------------------------------------- stacking: who makes the super-resolution

def test_cosmic_clarity_superres_stacks_at_1x():
    rs = workflow.decide_resampling({"scale": 2.0, "mode": "drizzle", "superres_method": "cosmic_clarity"}, 300, 1.5)
    assert rs["scale"] == 1.0 and rs["superres"] == "cosmic_clarity" and rs["requested_scale"] == 2.0
    assert "Cosmic Clarity" in rs["reason"]
    # drizzle (the default) keeps its behaviour
    assert workflow.decide_resampling({"scale": 2.0, "mode": "drizzle"}, 300, 1.5)["scale"] == 2.0


def test_superres_factor_is_one_of_the_models():
    f = post.cc_superres_factor
    assert f({"superres_method": "cosmic_clarity", "scale": 1.5}) == 1.5        # no rounding: the export records that 1.5x cannot run
    assert f({"superres_method": "cosmic_clarity", "scale": 2}) == 2
    assert f({"superres_method": "cosmic_clarity", "scale": 3}) == 3
    assert f({"superres_method": "cosmic_clarity", "scale": 4}) == 4
    assert f({"superres_method": "cosmic_clarity", "scale": 1}) == 0
    assert f({"superres_method": "drizzle", "scale": 2}) == 0
    assert f({}) == 0


# ----------------------------------------------------------------------------- restore: the steps that run

class FakeCC:
    """tools.cosmic_clarity stand-in: records the calls; superres enlarges by pixel repetition."""

    def __init__(self, ready=("sharpen", "denoise", "superres", "darkstar", "satellite")):
        self.ready, self.calls = set(ready), []

    def run(self, mode, arr, cli, **kw):
        if mode not in self.ready:
            raise ToolUnavailable(f"cosmic clarity {mode}", "models not installed")
        self.calls.append((mode, list(cli)))
        out = np.asarray(arr, np.float32)
        if mode == "superres":
            f = int(cli[cli.index("--scale") + 1])
            out = np.repeat(np.repeat(out, f, axis=1), f, axis=2)
        elif mode == "darkstar":
            out = out * 0.9
        meta = {"command": f"CosmicClarity_Native.py --mode {mode} " + " ".join(cli), "backend": "auto", "device": "cpu"}
        if mode == "satellite":
            meta["trails_detected"] = False
        return out, meta


class FakeRC:
    def __init__(self, licensed=("bxt", "nxt")):
        self.licensed, self.calls, self.params = licensed, [], []

    def require(self, key):
        if key not in self.licensed:
            raise ToolUnavailable(f"rc-astro {key}", "not licensed", 77)

    def run(self, product, src, dst, **params):
        self.calls.append(product)
        self.params.append(params)
        arr, hdr = fitsio.read_rgb(src)
        fitsio.write_rgb(dst, arr, hdr)
        return {"tool": f"rc-astro {product}", "flags": []}


def _stack(h=96, w=128):
    a = 0.02 + 4e-3 * RNG.standard_normal((3, h, w))
    a[:, 40:43, 60:63] += 0.6
    return np.clip(a, 0, 1).astype(np.float32)


def _restorer(tmp_path, sp, params=None):
    d = tmp_path / "s"
    d.mkdir(exist_ok=True)
    hdr = {"ROWORDER": "BOTTOM-UP", "FILTER": "Duo-Band"}
    fitsio.write_rgb(str(d / "stack.fits"), _stack(), hdr)
    return post.Restorer(str(d), {"filter": "Duo-Band"}, sp, {**post.DEFAULTS, **(params or {})}, None, None)


@pytest.fixture()
def fakes(monkeypatch):
    cc, rc = FakeCC(), FakeRC()
    monkeypatch.setattr(post.tools, "cosmic_clarity", cc)
    monkeypatch.setattr(post.tools, "rcastro", rc)
    return cc, rc


class FakeGX:
    """tools.graxpert stand-in: records the deconvolution / denoise calls, returns the image unchanged."""

    def __init__(self):
        self.calls = []

    def deconvolve(self, src, dst, kind="obj", strength=0.5, psfsize=3.0, **kw):
        self.calls.append(("deconv", kind, strength, psfsize))
        arr, hdr = fitsio.read_rgb(src)
        fitsio.write_rgb(dst, arr, hdr)
        return {"model": "test"}

    def denoise(self, src, dst, strength=0.5, **kw):
        self.calls.append(("denoise", strength))
        arr, hdr = fitsio.read_rgb(src)
        fitsio.write_rgb(dst, arr, hdr)
        return {"model": "test"}


@pytest.fixture()
def fake_gx(monkeypatch):
    gx = FakeGX()
    monkeypatch.setattr(post.tools, "graxpert", gx)
    return gx


@pytest.mark.parametrize("method,kinds", [("graxpert_obj", ["obj"]), ("graxpert_stellar", ["stellar"]), ("graxpert_both", ["obj", "stellar"])])
def test_graxpert_deconvolution_as_chosen(tmp_path, monkeypatch, fakes, fake_gx, method, kinds):
    cc, rc = fakes
    monkeypatch.setattr(post.Restorer, "image_fwhm", lambda self: 4.5)
    r = _restorer(tmp_path, {"deconv_method": method}, {"deconvolution": 0.6})
    r.psf_correction()
    r.deconvolve()
    assert rc.calls == []                                            # no BlurXTerminator, not even correct-only
    assert [c[1] for c in fake_gx.calls] == kinds
    assert all(c[2] == pytest.approx(0.6) and c[3] == 4.5 for c in fake_gx.calls)    # -strength = the slider, -psfsize = the image FWHM


def test_graxpert_deconvolution_needs_the_image_fwhm(tmp_path, monkeypatch, fakes, fake_gx):
    monkeypatch.setattr(post.Restorer, "image_fwhm", lambda self: None)
    r = _restorer(tmp_path, {"deconv_method": "graxpert_obj"})
    r.deconvolve()
    assert fake_gx.calls == [] and [x["status"] for x in r.steps.records if x["tool"] == "graxpert"] == ["failed"]


# ----------------------------------------------------------------------------- deconvolution per region (objects / stars)

@pytest.mark.parametrize("sp,want", [
    ({}, ("auto", "auto")),
    ({"deconv_method": "rl"}, ("rl", "rl")),                                   # one choice named: the stars follow it
    ({"deconv_method": "rl", "deconv_stars_method": ""}, ("rl", "rl")),
    ({"deconv_method": "imagemm", "deconv_stars_method": "bxt"}, ("imagemm", "bxt")),
    ({"deconv_method": "graxpert_obj"}, ("graxpert", "none")),                 # the former single GraXpert choices
    ({"deconv_method": "graxpert_stellar"}, ("none", "graxpert")),
    ({"deconv_method": "graxpert_both"}, ("graxpert", "graxpert")),
    ({"deconv_method": "graxpert_obj", "deconv_stars_method": "bxt"}, ("graxpert", "bxt")),
])
def test_deconv_choice(sp, want):
    assert post.deconv_choice(sp) == want


@pytest.mark.parametrize("obj,stars,kinds", [("graxpert", "graxpert", ["obj", "stellar"]), ("graxpert", "none", ["obj"]), ("none", "graxpert", ["stellar"])])
def test_graxpert_per_region(tmp_path, monkeypatch, fakes, fake_gx, obj, stars, kinds):
    monkeypatch.setattr(post.Restorer, "image_fwhm", lambda self: 4.5)
    _restorer(tmp_path, {"deconv_method": obj, "deconv_stars_method": stars}).deconvolve()
    assert [c[1] for c in fake_gx.calls] == kinds


def test_bxt_objects_graxpert_stars(tmp_path, monkeypatch, fakes, fake_gx):
    _, rc = fakes
    monkeypatch.setattr(post.Restorer, "image_fwhm", lambda self: 4.5)
    r = _restorer(tmp_path, {"deconv_method": "bxt", "deconv_stars_method": "graxpert"})
    r.psf_correction()
    r.deconvolve()
    assert rc.calls == ["bxt", "bxt"] and rc.params[0].get("correct_only")      # correct-only, then the objects' sharpening
    assert rc.params[1]["ss"] == 0.0 and rc.params[1]["sn"] > 0                  # Sharpen Stars off: the stars are GraXpert's
    assert [c[1] for c in fake_gx.calls] == ["stellar"]
    assert [x.get("params", {}).get("region") for x in r.steps.records if x["step"] == "deconvolution"] == ["objects", "stars"]


def test_bxt_same_tool_runs_once_on_both(tmp_path, fakes):
    _, rc = fakes
    _restorer(tmp_path, {"deconv_method": "bxt", "deconv_stars_method": "bxt"}).deconvolve()
    assert rc.calls == ["bxt"] and rc.params[0]["ss"] > 0 and rc.params[0]["sn"] > 0


def test_rl_objects_through_the_star_mask_then_bxt_stars(tmp_path, monkeypatch, fakes):
    _, rc = fakes
    mask = np.zeros((96, 128), np.float32)
    mask[30:50, 50:70] = 1.0
    monkeypatch.setattr(post, "star_footprint_mask", lambda work, steps, arr, header: (mask, {"stars": 1}))
    monkeypatch.setattr(post, "_siril_image", lambda work, steps, name, arr, header, *a, **k: (arr * 0.5, header))
    r = _restorer(tmp_path, {"deconv_method": "rl", "deconv_stars_method": "bxt"}, {"deconvolution": 0.5})
    before = r.arr.copy()
    r.deconvolve()
    diff = r.arr - before
    want = -0.25 * before                                            # 0.5 x (the fake RL's arr / 2 - arr)
    assert np.allclose(diff[:, mask == 0], want[:, mask == 0], atol=1e-6) and np.allclose(diff[:, mask == 1], 0.0, atol=1e-6)   # RL only off the stars, at the slider's blend
    assert rc.calls == ["bxt"] and rc.params[0]["sn"] == 0.0 and rc.params[0]["ss"] > 0


def test_auto_stars_unlicensed_falls_back_to_masked_rl(tmp_path, monkeypatch, fakes):
    _, rc = fakes
    rc.licensed = ()
    mask = np.zeros((96, 128), np.float32)
    mask[30:50, 50:70] = 1.0
    monkeypatch.setattr(post, "star_footprint_mask", lambda work, steps, arr, header: (mask, {"stars": 1}))
    monkeypatch.setattr(post, "_siril_image", lambda work, steps, name, arr, header, *a, **k: (arr * 0.5, header))
    r = _restorer(tmp_path, {"deconv_method": "none", "deconv_stars_method": "auto"}, {"deconvolution": 1.0})
    before = r.arr.copy()
    r.deconvolve()
    diff = r.arr - before
    assert np.allclose(diff[:, mask == 1], -0.5 * before[:, mask == 1], atol=1e-6) and np.allclose(diff[:, mask == 0], 0.0, atol=1e-6)


def test_imagemm_without_the_session_is_recorded_not_replaced(tmp_path, fakes):
    _, rc = fakes
    r = _restorer(tmp_path, {"deconv_method": "imagemm", "deconv_stars_method": "none"})
    r.deconvolve()
    assert rc.calls == [] and [(x["tool"], x["status"]) for x in r.steps.records if x["tool"] == "imagemm"] == [("imagemm", "unavailable")]


def test_imagemm_objects_takes_the_stars_from_the_stack(tmp_path, monkeypatch, fakes):
    _, rc = fakes
    d = tmp_path / "s"
    d.mkdir()
    hdr = {"ROWORDER": "BOTTOM-UP", "FILTER": "Duo-Band"}
    stack = np.full((3, 48, 64), 0.2, np.float32)
    fitsio.write_rgb(str(d / "stack.fits"), stack, hdr)
    fitsio.write_rgb(str(d / "imagemm.fits"), np.full((3, 96, 128), 0.6, np.float32), hdr)
    mask = np.zeros((96, 128), np.float32)
    mask[10:20, 10:20] = 1.0
    seen = []
    monkeypatch.setattr(post, "star_footprint_mask", lambda work, steps, arr, header: (seen.append(arr.shape), (mask, {"stars": 1}))[1])
    r = post.Restorer(str(d), {"filter": "Duo-Band", "restore_source": "imagemm.fits", "imagemm_r": 2, "imagemm_offset": [0, 0]},
                      {"deconv_method": "imagemm", "deconv_stars_method": "bxt"}, dict(post.DEFAULTS), None, None)
    assert seen == [(3, 96, 128)]                                    # the mask measured on the stack, resampled to ImageMM's grid
    k = r.scale
    assert np.allclose(r.arr[:, mask == 1], 0.2 * k, atol=1e-4) and np.allclose(r.arr[:, mask == 0], 0.6 * k, atol=1e-4)
    r.psf_correction()
    r.deconvolve()
    assert rc.calls == ["bxt"] and rc.params[0]["sn"] == 0.0           # no correct-only over ImageMM's objects; BXT sharpens the stars only


def test_imagemm_region_keeps_one_scale_with_an_hdr_stack_above_1(tmp_path, monkeypatch, fakes):
    """An HDR stack peaks above 1 (the blend; Sh2-131 2.13) and ImageMM's output at 1: the two parts must be joined on their
    own values and scaled together - each sanitised on its own put the stack's stars 2.13 x darker than ImageMM's sky."""
    d = tmp_path / "s"
    d.mkdir()
    hdr = {"ROWORDER": "BOTTOM-UP", "FILTER": "Duo-Band"}
    stack = np.full((3, 64, 64), 0.02, np.float32)
    stack[:, 30, 30] = 2.0                                           # an HDR-blended core
    imm = np.full((3, 64, 64), 0.02, np.float32)
    imm[:, 30, 30] = 1.0
    fitsio.write_rgb(str(d / "stack.fits"), stack, hdr)
    fitsio.write_rgb(str(d / "imagemm.fits"), imm, hdr)
    mask = np.zeros((64, 64), np.float32)
    mask[20:40, 20:40] = 1.0
    monkeypatch.setattr(post, "star_footprint_mask", lambda work, steps, arr, header: (mask, {"stars": 1}))
    r = post.Restorer(str(d), {"filter": "Duo-Band", "restore_source": "imagemm.fits", "imagemm_r": 1, "imagemm_offset": [0, 0]},
                      {"deconv_method": "imagemm", "deconv_stars_method": "graxpert"}, dict(post.DEFAULTS), None, None)
    sky_in, sky_out = float(np.median(r.arr[1][mask == 1])), float(np.median(r.arr[1][mask == 0]))
    assert sky_in == pytest.approx(sky_out, rel=1e-5)                 # the stack's sky round the star = ImageMM's sky
    assert float(r.arr[1, 30, 30]) / sky_out == pytest.approx(100.0, rel=1e-4)   # the core keeps its ratio to the sky


def test_explicit_bxt_does_not_fall_back(tmp_path, monkeypatch, fakes):
    _, rc = fakes
    rc.licensed = ()
    ran = []
    monkeypatch.setattr(post, "_siril_image", lambda *a, **k: (ran.append(1), (a[3], a[4]))[1])
    r = _restorer(tmp_path, {"deconv_method": "bxt"})
    r.deconvolve()
    assert ran == [] and [(x["tool"], x["status"]) for x in r.steps.records if x["step"] == "deconvolution"] == [("rc-astro bxt", "unavailable")]


@pytest.mark.parametrize("method,want", [("nxt", "rc-astro nxt"), ("graxpert", "graxpert")])
def test_explicit_linear_denoise_runs_that_tool(tmp_path, fakes, fake_gx, method, want):
    _, rc = fakes
    r = _restorer(tmp_path, {"denoise_method": method})
    r.denoise()
    done = [x["tool"] for x in r.steps.records if x["step"] == "denoise" and x["status"] == "done"]
    assert done == [want]
    assert ("nxt" in rc.calls) == (method == "nxt") and bool(fake_gx.calls) == (method == "graxpert")


def test_explicit_nxt_unlicensed_runs_nothing_else(tmp_path, fakes, fake_gx):
    _, rc = fakes
    rc.licensed = ()
    r = _restorer(tmp_path, {"denoise_method": "nxt"})
    r.denoise()
    assert fake_gx.calls == [] and [x["status"] for x in r.steps.records if x["step"] == "denoise"] == ["unavailable"]


def test_superres_scale_without_a_model_is_not_rounded(tmp_path, fakes):
    cc, _ = fakes
    info: dict = {}
    assert post._superres_final(_stack(), {"superres_method": "cosmic_clarity", "scale": 1.5}, str(tmp_path), post._Steps(1, None, None), info) is None
    assert cc.calls == [] and info["superres"]["status"] == "unavailable"


def test_graxpert_psfsize_setting_overrides_the_measurement(tmp_path, monkeypatch, fakes, fake_gx):
    monkeypatch.setattr(post.Restorer, "image_fwhm", lambda self: 4.5)
    _restorer(tmp_path, {"deconv_method": "graxpert_obj", "graxpert_psfsize": 6.0}).deconvolve()
    assert fake_gx.calls[-1][3] == 6.0


def test_rl_runs_no_bxt_correct_only(tmp_path, fakes):
    _, rc = fakes
    _restorer(tmp_path, {"deconv_method": "rl"}).psf_correction()
    assert rc.calls == []


def test_darkstar_unavailable_runs_no_other_remover(tmp_path, monkeypatch, fakes):
    cc, rc = fakes
    cc.ready = set()
    d = tmp_path / "s"
    d.mkdir()
    fitsio.write_rgb(str(d / "linear.fits"), _stack(), {"ROWORDER": "BOTTOM-UP"})
    meta = post.remove_stars(str(d), {}, {**post.DEFAULTS, "star_removal": "darkstar"}, {})
    assert meta["tool"] == "none" and [a["tool"] for a in meta["attempts"]] == ["darkstar"] and rc.calls == []


def test_cc_settings_are_shown_in_the_options_panel():
    opt = {p["key"] for p in post.PARAM_SPEC if p.get("host") == "options"}
    assert opt == {"cc_sharpen", "cc_stellar_amount", "cc_nonstellar_amount", "cc_denoise", "cc_denoise_model"}
    assert all(k in post.EXTRA_KEYS and k in post.EXTRA_DEFAULTS for k in opt)


def test_superres_on_the_finished_image(tmp_path, fakes):
    cc, _ = fakes
    steps = post._Steps(1, None, None)
    info: dict = {}
    img = _stack()
    out = post._superres_final(img, {"superres_method": "cosmic_clarity", "scale": 2}, str(tmp_path), steps, info)
    assert out.shape[1:] == (2 * img.shape[1], 2 * img.shape[2]) and info["superres"]["factor"] == 2
    assert post._superres_final(img, {"superres_method": "drizzle", "scale": 2}, str(tmp_path), steps, {}) is None
    cc.ready = set()
    info = {}
    assert post._superres_final(img, {"superres_method": "cosmic_clarity", "scale": 3}, str(tmp_path), steps, info) is None
    assert info["superres"]["status"] == "failed"


def test_satellite_removal_only_when_asked(tmp_path, fakes):
    cc, _ = fakes
    _restorer(tmp_path, {}).satellite()
    assert cc.calls == []
    _restorer(tmp_path, {"satellite_removal": True}).satellite()
    assert cc.calls[-1][0] == "satellite"


def test_darkstar_star_layer_is_linear_minus_starless(tmp_path, fakes):
    cc, _ = fakes
    d = tmp_path / "s"
    d.mkdir()
    lin = _stack()
    fitsio.write_rgb(str(d / "linear.fits"), lin, {"ROWORDER": "BOTTOM-UP"})
    meta = post.remove_stars(str(d), {}, {**post.DEFAULTS, "star_removal": "darkstar"}, {})
    assert meta["tool"] == "darkstar" and meta["stage"] == "linear"
    starless, _ = fitsio.read_rgb(str(d / "starless.fits"))
    stars, _ = fitsio.read_rgb(str(d / "stars.fits"))
    assert np.abs(starless + stars - lin).max() < 1e-5


@pytest.mark.parametrize("model,flag", [("full", None), ("lite", "--lite"), ("walking", "--walking")])
def test_finisher_cc_denoise_as_chosen(tmp_path, monkeypatch, model, flag):
    cc = FakeCC()
    monkeypatch.setattr(pro.tools, "cosmic_clarity", cc)
    fin = pro.Finisher(str(tmp_path), {"ROWORDER": "BOTTOM-UP"}, {"luminance_denoise": 0.7, "chroma_denoise": 0.95, "cc_denoise": True, "cc_denoise_model": model},
                       True, keep_pictures=False)
    fin.denoise(_stack())
    mode, cli = cc.calls[-1]
    assert mode == "denoise" and cli[cli.index("--denoise-strength") + 1] == "0.7" and cli[cli.index("--denoise-color") + 1] == "0.95"
    assert (flag in cli) if flag else not ({"--lite", "--walking"} & set(cli))
    assert fin.records[-1]["tool"] == "cosmic clarity denoise"


def test_finisher_cc_denoise_unavailable_runs_nothing_else(tmp_path, monkeypatch):
    cc = FakeCC(ready=())
    monkeypatch.setattr(pro.tools, "cosmic_clarity", cc)
    rc = FakeRC()
    monkeypatch.setattr(pro.tools, "rcastro", rc)
    fin = pro.Finisher(str(tmp_path), {}, {"cc_denoise": True}, True, keep_pictures=False)
    fin.denoise(_stack())
    assert rc.calls == [] and fin.records[-1]["status"] == "unavailable"


def test_finisher_cc_sharpen_as_chosen(tmp_path, monkeypatch):
    for choice, mode in (("both", "Both"), ("stellar", "Stellar Only"), ("nonstellar", "Non-Stellar Only"), ("off", None)):
        cc = FakeCC()
        monkeypatch.setattr(pro.tools, "cosmic_clarity", cc)
        fin = pro.Finisher(str(tmp_path), {}, {"cc_sharpen": choice, "cc_stellar_amount": 0.3, "cc_nonstellar_amount": 0.6}, True, keep_pictures=False)
        fin.cc_sharpen(_stack())
        if mode is None:
            assert cc.calls == []
        else:
            m, cli = cc.calls[-1]
            assert m == "sharpen" and cli[cli.index("--sharpening-mode") + 1] == mode
            assert cli[cli.index("--stellar-amount") + 1] == "0.3" and cli[cli.index("--nonstellar-amount") + 1] == "0.6"
    cc = FakeCC()
    monkeypatch.setattr(pro.tools, "cosmic_clarity", cc)
    pro.Finisher(str(tmp_path), {}, {"cc_sharpen": "both"}, True, external=False, keep_pictures=False).cc_sharpen(_stack())
    assert cc.calls == []                                            # the preview runs no AI step


# ----------------------------------------------------------------------------- the Hubble palette script

def test_hubble_preset_choice():
    fin = lambda p: pro.Finisher("/tmp", {}, p, True, keep_pictures=False)      # noqa: E731
    assert fin({}).hubble_preset("sho") == "Classic"
    assert fin({}).hubble_preset("hoo_improved") == "Improved"
    assert fin({"oiii_unmix": True}).hubble_preset("hoo") == "Improved"
    assert fin({"hubble_formula": "nonlinear_s2"}).hubble_preset("sho") == "NonLinear S2"
    assert fin({"hubble_formula": "advanced"}).hubble_preset("hoo_improved") == "Advanced"


def test_hubble_palette_runs_the_script(tmp_path, monkeypatch):
    lines_run = []
    monkeypatch.setattr(pro, "_community_script", lambda rel: "/scripts/" + os.path.basename(rel))
    monkeypatch.setattr(pro.Finisher, "siril", lambda self, tag, arr, lines, stage, **kw: (lines_run.append((tag, lines, kw.get("out"))), arr)[1])
    fin = pro.Finisher(str(tmp_path), {}, {"palette": "sho", "synthetic_luminance": 0.0}, True, keep_pictures=False)
    fin.colour(_stack())
    tag, lines, out = lines_run[0]
    assert lines[0].startswith("pyscript ") and lines[0].endswith("hubble_palette_worker.py --script /scripts/Hubble_Palette_from_Dual-Band_OSC.py --palette SHO --preset Classic")
    assert out == "hp_in_result_SHO_Classic"                         # the script's own result file name
    assert fin.records[-1]["tool"] == "siril pyscript Hubble_Palette_from_Dual-Band_OSC.py"


def test_hubble_palette_script_missing_is_recorded_not_replaced(tmp_path, monkeypatch):
    lines_run = []
    monkeypatch.setattr(pro, "_community_script", lambda rel: None)
    monkeypatch.setattr(pro.Finisher, "siril", lambda self, tag, arr, lines, stage, **kw: (lines_run.append(lines), arr)[1])
    fin = pro.Finisher(str(tmp_path), {}, {"palette": "sho", "hubble_formula": "improved", "synthetic_luminance": 0.0}, True, keep_pictures=False)
    x = _stack()
    out = fin.colour(x)
    assert lines_run == [] and fin.records[-1]["status"] == "failed" and np.array_equal(out, x)


def test_scnr_after_ha_in_green_palettes_only(tmp_path, monkeypatch):
    lines_run = []
    monkeypatch.setattr(pro, "_community_script", lambda rel: "/s.py")
    monkeypatch.setattr(pro.Finisher, "siril", lambda self, tag, arr, lines, stage, **kw: (lines_run.append(lines), arr)[1])
    for pal, want in (("sho", True), ("hso", False), ("hoo", False)):
        lines_run.clear()
        pro.Finisher(str(tmp_path), {}, {"palette": pal, "scnr": 0.6, "synthetic_luminance": 0.0}, True, keep_pictures=False).colour(_stack())
        assert any(ls == ["rmgreen 3 0.600"] for ls in lines_run) == want


# ----------------------------------------------------------------------------- the model inventory (no torch import)

FAKE_SCRIPT = '''
VERSION        = "1.2.0"
MODEL_FILES: dict[str, str] = {
    "sharpen_stellar": "s.pth", "sharpen_stellar_onnx": "s.onnx",
    "darkstar_color": "d.pt", "darkstar_color_onnx": "d.onnx",
}
MODE_MODELS: dict[str, list[str]] = {"sharpen": ["sharpen_stellar"], "darkstar": ["darkstar_color"]}
'''


def test_models_inventory_torch_onnx_missing(tmp_path, monkeypatch):
    data = tmp_path / "share"
    (data / "siril-scripts" / "processing").mkdir(parents=True)
    (data / "siril-scripts" / tools.CC_SCRIPT).write_text(FAKE_SCRIPT)
    md = data / "siril" / "cosmic_clarity"
    md.mkdir(parents=True)
    monkeypatch.setenv("XDG_DATA_HOME", str(data))
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object() if name == "onnxruntime" else real(name, *a))
    cc = tools.CosmicClarity()
    assert cc.models()["sharpen"] is False and cc.models()["darkstar"] is False
    (md / "s.pth").write_bytes(b"x")
    (md / "d.onnx").write_bytes(b"x")
    m = cc.models()
    assert m["sharpen"] == "torch" and m["darkstar"] == "onnx" and m["denoise"] is False
    st = cc.status()
    assert st["version"] == "1.2.0" and st["models"]["darkstar"] == "onnx"



def test_imagemm_region_measures_the_psf_on_the_stack(tmp_path, monkeypatch, fakes):
    """With ImageMM for the objects, GraXpert deconvolves the stack's stars: its -psfsize is the stack's FWHM, not the joined image's
    (whose isolated faint stars are ImageMM's restored ones)."""
    d = tmp_path / "s"
    d.mkdir()
    hdr = {"ROWORDER": "BOTTOM-UP", "FILTER": "Duo-Band"}
    fitsio.write_rgb(str(d / "stack.fits"), np.full((3, 64, 64), 0.2, np.float32), hdr)
    fitsio.write_rgb(str(d / "imagemm.fits"), np.full((3, 64, 64), 0.6, np.float32), hdr)
    monkeypatch.setattr(post, "star_footprint_mask", lambda work, steps, arr, header: (np.zeros((64, 64), np.float32), {"stars": 0}))
    from sirilpipe.lab import metrics as MX
    seen = []
    monkeypatch.setattr(MX, "star_fwhm", lambda img, sat: (seen.append(float(np.median(img))), 4.3)[1])
    r = post.Restorer(str(d), {"filter": "Duo-Band", "restore_source": "imagemm.fits", "imagemm_r": 1, "imagemm_offset": [0, 0]},
                      {"deconv_method": "imagemm", "deconv_stars_method": "graxpert"}, dict(post.DEFAULTS), None, None)
    assert r.image_fwhm() == 4.3 and seen == [pytest.approx(0.2)]   # measured on the stack's values
