"""The ImageMM stage on Siril's registered frames: frame loading / placement / masks, parameter mapping,
availability without torch, the Session wiring and a small end-to-end restoration on synthetic frames
(that one needs torch and is skipped without it)."""
from __future__ import annotations

import json
import os
import sys
import types

import numpy as np
import pytest
from astropy.io import fits

from sirilpipe import fitsio, imagemm_stage as S


# --------------------------------------------------------------------------- availability / parameters
def test_available_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)             # `import torch` raises ImportError
    av = S.available()
    assert av["ok"] is False and av["torch"] is False and av["cuda"] is False
    assert "IMAGEMM" in av["reason"]


def test_available_does_not_import_torch_at_module_import():
    src = open(S.__file__, encoding="utf-8").read()
    head = src.split("def available")[0]
    assert "import torch" not in head and "import astrophoto" not in head


def test_map_params_defaults_and_keys():
    kw = S.map_params({})
    assert kw == {"r": 1, "sigma": 1.0, "psf_model": "empirical", "n_groups": 0, "robust": True, "delta": 2.0,
                  "kappa": 2.0, "epsilon": 1e-6, "stop": "c15", "max_iters": 1000, "accelerate": False,
                  "background": False}
    kw = S.map_params({"imagemm_r": 2, "imagemm_kappa": 3, "imagemm_delta": 1.5, "imagemm_epsilon": 1e-5,
                       "imagemm_stop": "flux", "imagemm_max_iters": 200, "imagemm_psf": "moffat",
                       "imagemm_robust": False})
    assert kw["r"] == 2 and kw["sigma"] == 1.1                     # the paper's g_sigma at r = 2
    assert (kw["kappa"], kw["delta"], kw["epsilon"]) == (3.0, 1.5, 1e-5)
    assert kw["stop"] == "flux" and kw["max_iters"] == 200 and kw["psf_model"] == "moffat" and kw["robust"] is False
    assert S.map_params({"imagemm_sigma": 1.3, "imagemm_r": 1})["sigma"] == 1.3
    assert S.map_params({"imagemm_stop": "nonsense"})["stop"] == "c15"
    assert S.map_params({"imagemm_r": 9})["r"] == 3 and S.map_params({"imagemm_r": 0})["r"] == 1
    assert S.map_params(None)["r"] == 1


def test_resampling_of_stack_meta():
    rs = S.resampling_of({"mode": "drizzle", "scale": 1.0, "drizzle": {"scale": 1.0, "pixfrac": 1.0, "kernel": "square"}})
    assert rs == {"mode": "drizzle", "scale": 1.0, "pixfrac": 1.0, "kernel": "square"}
    assert S.resampling_of({"mode": "demosaic", "scale": 1.0})["mode"] == "demosaic"


# --------------------------------------------------------------------------- frames
SEQ = """#Siril sequence file.
S 'r_pp_light_' 1 3 3 5 1 6 1 0 1
L 3
I 1 1 100,80
I 3 1 98,76
I 4 1 96,72
R1 3.5 4 0.8 0 0.0004 100 H 1 0 -1 0 1 0 0 0 1
R1 3.5 4 0.8 0 0.0004 100 H 1 0 3 0 1 2 0 0 1
R1 3.5 4 0.8 0 0.0004 100 H 1 0 5 0 1 6 0 0 1
"""


def test_sequence_placements(tmp_path):
    f = tmp_path / "r_pp_light_.seq"
    f.write_text(SEQ)
    pl = S.sequence_placements(str(f), (104, 87))
    # ox = tx - min(tx); oy = canvas_h - h - (ty - min(ty)) (row 0 is the bottom row)
    assert pl == {1: (0, 87 - 80 - 0, 100, 80), 3: (4, 87 - 76 - 2, 98, 76), 4: (6, 87 - 72 - 6, 96, 72)}


def test_sequence_placements_without_registration_needs_one_size(tmp_path):
    f = tmp_path / "r_.seq"
    f.write_text("S 'r_' 1 1 1 5 1 6 1 0 1\nL 3\nI 1 1 10,10\n")
    assert S.sequence_placements(str(f), (10, 10)) == {1: (0, 0, 10, 10)}
    with pytest.raises(RuntimeError):
        S.sequence_placements(str(f), (12, 10))


def _frame_file(tmp_path, shape=(3, 40, 50), name="r_1.fit"):
    a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1
    path = str(tmp_path / name)
    fits.PrimaryHDU(a).writeto(path, overwrite=True)
    return path, a


def test_read_frame_crop_inside(tmp_path):
    path, a = _frame_file(tmp_path)
    y = S.read_frame(path, (5, 7, 20, 10), (0, 0, 50, 40))
    assert y.shape == (10, 20, 3) and y.dtype == np.float32
    np.testing.assert_array_equal(y, np.moveaxis(a[:, 7:17, 5:25], 0, -1))


def test_read_frame_placed_on_canvas_with_gaps(tmp_path):
    path, a = _frame_file(tmp_path)                       # 50 x 40 frame
    # the frame sits at (ox, oy) = (10, 6) on the canvas; the box reaches 8 px left of and 4 above it
    y = S.read_frame(path, (2, 2, 30, 20), (10, 6, 50, 40))
    assert y.shape == (20, 30, 3)
    assert (y[:4] == 0).all() and (y[:, :8] == 0).all()               # canvas the frame does not cover: no data
    np.testing.assert_array_equal(y[4:, 8:], np.moveaxis(a[:, 0:16, 0:22], 0, -1))
    assert (S.read_frame(path, (100, 100, 5, 5), (10, 6, 50, 40)) == 0).all()      # entirely outside


def test_read_frame_size_mismatch_is_an_error(tmp_path):
    path, _ = _frame_file(tmp_path)
    with pytest.raises(RuntimeError):
        S.read_frame(path, (0, 0, 5, 5), (0, 0, 60, 40))


# --------------------------------------------------------------------------- masks
def _drizzled(h=60, w=70, sat=0.06):
    rng = np.random.default_rng(1)
    y = (rng.normal(0.0003, 0.0001, (h, w, 3)) + 0.001).astype(np.float32)
    y[:6] = 0                                  # outside the footprint
    y[:, :4] = 0
    y[30, 30, 0] = 0                           # a lattice hole inside the footprint (red)
    y[31, 40, 2] = 0
    y[40:43, 50:53] = np.float32(0.0605)       # a saturated plateau (all channels)
    return y


def test_measurement_masks():
    y = _drizzled()
    m = S.measurement_masks(y, 0.0605)
    assert not m["data"][:6].any() and m["data"][20:, 10:].mean() > 0.99
    assert m["hole"][30, 30, 0] and m["hole"][31, 40, 2] and not m["hole"][2, 10, 0]   # holes are inside, border zeros are not
    assert m["region"][30, 30, 0] and not m["valid"][30, 30, 0]                        # a hole stays masked for ImageMM
    assert not m["interior"][6, 20, 1] and m["interior"][8, 20, 1]                     # the footprint's border pixel is masked
    # the censored pixels: the plateau grown by the drop footprint (1 px green, 2 px red / blue)
    assert m["cens"][41, 51].all()
    assert m["cens"][39, 51, 1] and not m["cens"][38, 51, 1]
    assert m["cens"][38, 51, 0] and not m["cens"][37, 51, 0]
    assert not (m["valid"] & m["cens"]).any()
    assert m["valid"][20, 20].all()


def test_measurement_masks_without_saturation():
    m = S.measurement_masks(_drizzled(), None)
    assert not m["cens"].any()


def test_fill_holes_only_for_measuring():
    y = _drizzled()
    m = S.measurement_masks(y, 0.0605)
    f = S.fill_holes(y, m)
    nb = y[29:32, 29:32, 0].copy()
    nb[1, 1] = 0
    assert f[30, 30, 0] == pytest.approx(nb.sum() / 8, rel=1e-5)
    assert (f[:6] == 0).all()                           # no fill outside the footprint
    assert y[30, 30, 0] == 0                            # the input is untouched


def test_saturation_level_needs_a_plateau():
    sat = [_drizzled() for _ in range(4)]
    assert S.saturation_level(sat) == pytest.approx(0.0605, rel=1e-4)
    rng = np.random.default_rng(0)
    plain = []
    for _ in range(4):
        a = rng.normal(0.001, 0.0001, (60, 70, 3)).astype(np.float32)
        a[20, 20] = 0.03                                # a bright unsaturated peak, no plateau
        plain.append(a)
    assert S.saturation_level(plain) is None


def test_stack_is_flipped():
    rng = np.random.default_rng(3)
    base = np.zeros((200, 240, 3), np.float32)
    yy, xx = np.mgrid[:200, :240]
    for _ in range(40):
        cy, cx, amp = rng.uniform(10, 190), rng.uniform(10, 230), rng.uniform(0.2, 1.0)
        base += (amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 18.0))[..., None].astype(np.float32)
    frame = base * 0.1
    f, info = S.stack_is_flipped(base, frame)
    assert f is False and info["correlation"] > 0.9
    f, info = S.stack_is_flipped(np.ascontiguousarray(base[::-1]), frame)
    assert f is True and info["correlation_flipped"] > 0.9


def test_surface_window_equals_full():
    coefs = np.array([[0.1, 0.02, -0.03, 0.01, 0.005, -0.002]] * 3)
    full = S.surface_window(coefs, 50, 60, 0, 50, 0, 60)
    win = S.surface_window(coefs, 50, 60, 10, 30, 15, 45)
    np.testing.assert_allclose(win, full[:, 10:30, 15:45], atol=1e-6)


def test_norm_peak():
    assert S._norm_peak(np.full((3, 4, 4), 0.5, np.float32)) == 1.0
    a = np.full((3, 4, 4), 0.5, np.float32)
    a[0, 0, 0] = 2.5
    assert S._norm_peak(a) == pytest.approx(0.4)


# --------------------------------------------------------------------------- WCS
def test_rescale_wcs():
    h = fits.Header()
    for k, v in {"CRPIX1": 101.0, "CRPIX2": 51.0, "CD1_1": -1e-3, "CD1_2": 2e-4, "CD2_1": 2e-4, "CD2_2": 1e-3,
                 "A_2_0": 1e-6, "B_0_2": 2e-6, "A_ORDER": 2}.items():
        h[k] = v
    g = S.rescale_wcs(h, 2, (10, 20))
    assert g["CRPIX1"] == 2 * (101 - 1 - 10) + 1 and g["CRPIX2"] == 2 * (51 - 1 - 20) + 1
    assert g["CD1_1"] == pytest.approx(-5e-4) and g["CD2_2"] == pytest.approx(5e-4)
    assert g["A_2_0"] == pytest.approx(1e-6 * 2 ** (1 - 2)) and g["B_0_2"] == pytest.approx(2e-6 * 2 ** -1)
    assert g["A_ORDER"] == 2
    same = S.rescale_wcs(h, 1)
    assert same["CRPIX1"] == 101.0 and same["CD1_1"] == -1e-3


# --------------------------------------------------------------------------- the Session wiring
def _session(tmp_path):
    from sirilpipe.session import Session
    s = Session(str(tmp_path / "images"), str(tmp_path / "out"))
    fitsio.write_rgb(s._p("stack.fits"), np.full((3, 4, 4), 0.25, np.float32))
    fitsio.write_rgb(s._p("imagemm.fits"), np.full((3, 4, 4), 0.75, np.float32))
    with open(s._p("stack_meta.json"), "w") as f:
        json.dump({"scale": 1.0, "shape": [4, 4], "pixel_scale": 2.0}, f)
    os.makedirs(s._p("process"), exist_ok=True)
    return s


def _fake_post(seen):
    def restore(session_dir, meta, sp, params, progress=None, checkpoint=None, **kw):
        a, _ = fitsio.read_rgb(os.path.join(session_dir, (meta or {}).get("restore_source") or "stack.fits"))   # like post.Restorer
        seen.append({"stack_value": float(a[0, 0, 0]), "sp": dict(sp), "meta": dict(meta), "kw": kw})
        doc = {"name": "fake", "seconds": 0.1}
        with open(os.path.join(session_dir, "restore_meta.json"), "w") as f:
            json.dump(doc, f)
        return doc
    return types.SimpleNamespace(restore=restore)


def test_run_denoise_falls_back_when_torch_is_missing(tmp_path, monkeypatch, caplog):
    from sirilpipe import post, session
    s = _session(tmp_path)
    seen = []
    monkeypatch.setattr(post, "restore", _fake_post(seen).restore)
    monkeypatch.setattr(S, "available", lambda: {"ok": False, "torch": False, "cuda": False, "mps": False,
                                                 "reason": "torch is not installed"})
    monkeypatch.setattr(session, "_IMAGEMM_WARNED", False)
    out = s.run_denoise({"deconv_method": "imagemm"})
    out2 = s.run_denoise({"deconv_method": "imagemm"})
    assert len(seen) == 2 and seen[0]["sp"]["deconv_method"] == "imagemm" and seen[0]["stack_value"] == 0.25
    assert out["imagemm"] == "unavailable" and "torch" in out["imagemm_reason"]
    rm = json.load(open(s._p("restore_meta.json")))
    assert rm["imagemm"] == "unavailable" and out2["imagemm"] == "unavailable"
    assert sum("ImageMM is not available" in r.message for r in caplog.records) == 1      # logged once


def test_run_denoise_auto_does_not_touch_imagemm(tmp_path, monkeypatch):
    from sirilpipe import post
    s = _session(tmp_path)
    seen = []
    monkeypatch.setattr(post, "restore", _fake_post(seen).restore)

    def boom(*a, **k):
        raise AssertionError("the stage must not run for deconv_method=auto")
    monkeypatch.setattr(S, "run", boom)
    out = s.run_denoise({"deconv_method": "auto"})
    assert "imagemm" not in out and len(seen) == 1


def test_run_denoise_with_imagemm_reads_imagemm_fits_and_restores_the_stack(tmp_path, monkeypatch):
    from sirilpipe import post
    s = _session(tmp_path)
    seen = []
    monkeypatch.setattr(post, "restore", _fake_post(seen).restore)
    monkeypatch.setattr(S, "available", lambda: {"ok": True, "torch": True, "cuda": False, "mps": False, "reason": ""})
    monkeypatch.setattr(S, "run", lambda *a, **k: {"ok": True, "r": 1, "shape": [4, 4], "frames": ["a", "b"], "seconds": 1.0})
    out = s.run_denoise({"deconv_method": "imagemm"})
    c = seen[0]
    assert c["stack_value"] == 0.75                          # post.restore read imagemm.fits in stack.fits' place
    assert c["sp"]["deconv_method"] == "none"                # ImageMM is the deconvolution
    assert c["meta"]["restore_source"] == "imagemm.fits"
    a, _ = fitsio.read_rgb(s._p("stack.fits"))
    assert float(a[0, 0, 0]) == 0.25 and not os.path.exists(s._p("process", "stack_original.fits"))   # put back
    assert out["imagemm"] == "ran" and out["deconvolution_by"] == "imagemm" and "frames" not in out["imagemm_info"]


def test_run_denoise_imagemm_failure_falls_back(tmp_path, monkeypatch):
    from sirilpipe import post
    s = _session(tmp_path)
    seen = []
    monkeypatch.setattr(post, "restore", _fake_post(seen).restore)
    monkeypatch.setattr(S, "available", lambda: {"ok": True, "torch": True, "cuda": False, "mps": False, "reason": ""})

    def fail(*a, **k):
        raise RuntimeError("not enough GPU memory\nmore")
    monkeypatch.setattr(S, "run", fail)
    out = s.run_denoise({"deconv_method": "imagemm"})
    assert seen[0]["stack_value"] == 0.25 and seen[0]["sp"]["deconv_method"] == "imagemm"
    assert out["imagemm"] == "failed" and out["imagemm_reason"] == "not enough GPU memory"


def test_stack_swap_is_undone_on_error_and_recovered_after_a_crash(tmp_path):
    s = _session(tmp_path)
    with pytest.raises(ValueError):
        with s._stack_as("imagemm.fits"):
            assert float(fitsio.read_rgb(s._p("stack.fits"))[0][0, 0, 0]) == 0.75
            raise ValueError
    assert float(fitsio.read_rgb(s._p("stack.fits"))[0][0, 0, 0]) == 0.25
    os.replace(s._p("stack.fits"), s._p("process", "stack_original.fits"))          # a crash inside the swap
    s._recover_stack_swap()
    assert float(fitsio.read_rgb(s._p("stack.fits"))[0][0, 0, 0]) == 0.25


# --------------------------------------------------------------------------- end to end (synthetic frames, needs torch)
def _star_field(h, w, stars, sigma, shift, rng, gain=300.0, sky=0.02, sat=None):
    yy, xx = np.mgrid[:h, :w].astype(np.float64)
    img = np.full((h, w), sky)
    for (cy, cx, amp) in stars:
        img += amp * np.exp(-((yy - cy - shift[0]) ** 2 + (xx - cx - shift[1]) ** 2) / (2 * sigma ** 2))
    noisy = rng.poisson(img * gain) / gain
    return noisy


def test_registered_set_prepare_and_restore_end_to_end(tmp_path):
    torch = pytest.importorskip("torch")
    from astrophoto import imagemm
    rng = np.random.default_rng(7)
    h, w, n = 260, 300, 8
    # a jittered grid: isolated stars, as PSF stars must be
    stars = [(cy + rng.uniform(-5, 5), cx + rng.uniform(-5, 5), rng.uniform(0.5, 1.2))
             for cy in range(30, h - 20, 40) for cx in range(30, w - 20, 40)]
    Reg = S.make_set_class()
    files, rows, places, imgs = {}, [], {}, []
    for k in range(n):
        shift = rng.uniform(-0.4, 0.4, 2)
        img = np.stack([_star_field(h, w, stars, 1.5, shift, rng, gain=400.0) * (1.0 + 0.02 * c) for c in range(3)])
        img = img * np.float32(0.1)
        imgs.append(img)
        path = str(tmp_path / f"r_{k + 1:05d}.fit")
        fits.PrimaryHDU(img.astype(np.float32)).writeto(path)
        files[k] = path
        rows.append({"name": f"f{k}", "fwhm": 3.5, "weight": 1.0})
        places[k] = (0, 0, w, h)
    ref = np.moveaxis(np.mean(imgs, 0), 0, -1) / 0.1                  # the "stack": the mean of the frames, 10x their scale
    es = Reg(files, rows, (0, 0, w, h), ref.astype(np.float32), str(tmp_path / "work"), 1.0, 3.5, places=places)
    es.prepare()
    assert len(es.usable()) >= 6 and es.ptc["c1"].min() > 0
    assert es.sat_frames is None and es.sat > 1e20                      # no saturated star in this field
    Y, V, Mk = es.windows(es.usable()[:3], 100, 180, 120, 200)
    assert Y.shape == (3, 3, 80, 80) and (V > 0).all() and Mk.min() >= 0
    k = es.kernels(es.usable()[:2], "empirical")
    np.testing.assert_allclose(imagemm.unit_sum(k).sum((-2, -1)), 1.0, rtol=1e-5)
    lat, info = imagemm.restore(es, r=1, sigma=None, max_iters=40, device="cpu", region=(100, 180, 120, 200))
    assert lat.shape == (80, 80, 3) and np.isfinite(lat).all()
    assert info["n_exposures"] == len(es.usable()) and info["tiles"]
    es.cleanup()
    assert not os.path.exists(str(tmp_path / "work"))
