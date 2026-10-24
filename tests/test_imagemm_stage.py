"""The VeraLux ImageMM stage: parameters, availability without torch, Siril sequence files (the stack's transforms
copied to the new frames), raw-frame saturation, frame placement / store, the Session wiring, and run() end to end on
synthetic registered frames (the Siril steps faked; needs torch)."""
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


def test_map_params_defaults_are_the_scripts_gui_defaults_plus_the_fixes():
    kw = S.map_params({})
    assert kw == {"iterations": 150, "kappa": 2.0, "huber_delta": 0.05, "mu": 0.1, "stop_epsilon": 1e-5,
                  "saturation_level": None, "shot_noise_scale": 1.0, "read_noise": 0.0, "psf_sigma": 1.0,
                  "formal_mode": False, "r": 1,
                  # the pipeline's fixes (veralux_imagemm module docstring), on by default
                  "variance_from": "model", "per_channel_psf": True, "latent_sigma": 1.0, "lower_bound": "local",
                  "reference_psf": True, "photometry": True, "noise_model": "ptc", "huber_sigma": 2.0, "halo": False,
                  "saturated_points": False, "saturated_photometry": True, "psf_wings": False, "psf_asymmetry": True, "psf_radial_wings": True,
                  "bound_sigma": 1.0, "bloom_grow": 4, "point_init": True, "censor_saturated": False}
    # the 1.x keys are ignored
    assert S.map_params({"imagemm_delta": 2.0, "imagemm_max_iters": 2000, "imagemm_n2n": True}) == kw


def test_map_params_keys_ranges_and_types():
    kw = S.map_params({"imagemm_r": 2, "imagemm_iterations": "35", "imagemm_huber": 0.1, "imagemm_kappa": 3,
                       "imagemm_mu": 0.2, "imagemm_epsilon": 1e-6, "imagemm_sigma": 0, "imagemm_formal": "false",
                       "imagemm_saturation": 0.06})
    assert kw["r"] == 2 and kw["iterations"] == 35 and kw["huber_delta"] == 0.1 and kw["kappa"] == 3.0
    assert kw["mu"] == 0.2 and kw["stop_epsilon"] == 1e-6 and kw["psf_sigma"] == 1.0 and kw["formal_mode"] is False
    assert kw["saturation_level"] == 0.06
    kw = S.map_params({"imagemm_r": 9, "imagemm_iterations": 5000, "imagemm_kappa": 0.5, "imagemm_formal": True})
    assert kw["r"] == 3 and kw["iterations"] == 200 and kw["kappa"] == 1.01 and kw["formal_mode"] is True


# --------------------------------------------------------------------------- sequences
SRC_SEQ = """#Siril sequence file. Contains list of images, selection, registration data and statistics
#S 'sequence_name' start_index nb_images nb_selected fixed_len reference_image version variable_size fz_flag drizzle
S 'pp_light_' 1 4 4 5 2 6 0 0 0
L 1
I 1 1
I 2 1
I 3 1
I 4 1
R0 7.7 11.1 0.72 0 0.0005 246 H 1 0 -6 0 1 1.5 0 0 1
R0 7.5 10.5 0.75 0 0.0005 255 H 1 0 -9 0 1 6.9 0 0 1
R0 7.4 10.1 0.75 0 0.0005 259 H 1 0 0 0 1 0 0 0 1
R0 7.4 10.1 0.75 0 0.0005 259 H 1 0 4 0 1 -2 0 0 1
"""


def test_read_seq_and_registration_of(tmp_path):
    f = tmp_path / "pp_light_.seq"
    f.write_text(SRC_SEQ)
    seq = S.read_seq(str(f))
    assert seq["nums"] == [1, 2, 3, 4] and list(seq["R"]) == [0] and len(seq["R"][0]) == 4
    reg, ref = S.registration_of(seq)
    assert ref == 3 and reg[0][2].endswith("H 1 0 -9 0 1 6.9 0 0 1")


def test_write_registration_into_a_siril_output_sequence(tmp_path):
    src = tmp_path / "pp_light_.seq"
    src.write_text(SRC_SEQ)
    reg, ref = S.registration_of(S.read_seq(str(src)))
    dst = tmp_path / "bkg_imm_pp_light_.seq"       # as seqsubsky writes it: frames 2..4 kept their numbers
    dst.write_text("#Siril sequence file.\nS 'bkg_imm_pp_light_' 2 3 3 5 -1 6 0 0 0\nL -1\nI 2 1\nI 3 1\nI 4 1\n")
    S.write_registration(str(dst), [2, 3, 4], reg, ref, 3)
    out = S.read_seq(str(dst))
    assert out["S"][6] == "1" and "L 3" in out["lines"]                 # frame 3 (the reference) is image 1 here
    assert out["R"][0] == [reg[0][2], reg[0][3], reg[0][4]]
    with pytest.raises(RuntimeError):
        S.write_registration(str(dst), [1, 3, 4], reg, ref, 3)        # not the frames of that sequence


def test_write_registration_of_a_new_mono_sequence(tmp_path):
    src = tmp_path / "pp_light_.seq"
    src.write_text(SRC_SEQ.replace("L 1", "L 3").replace("R0 ", "R1 "))
    reg, ref = S.registration_of(S.read_seq(str(src)))
    S.write_registration(str(tmp_path / "immsat_.seq"), [1, 4], reg, ref, 1, name="immsat", mono_layer=True)
    out = S.read_seq(str(tmp_path / "immsat_.seq"))
    assert out["S"][1] == "'immsat_'" and out["S"][2] == "1" and out["S"][3] == "2" and out["S"][6] == "0"
    assert out["nums"] == [1, 4] and list(out["R"]) == [0] and out["R"][0][1].startswith("R0 ")


# --------------------------------------------------------------------------- saturation of the raw frames
def _raw(tmp_path, name, plateau, ceiling=4095, extra_peak=None):
    a = np.full((40, 50), 300, np.uint16)
    a[35, 45] = ceiling                              # a hot pixel: every frame, alone
    if plateau:
        a[10:13, 20:23] = ceiling
        a[10, 23] = ceiling - 10                     # 0.9976 x ceiling: saturated too (>= 0.995)
        a[30, 30] = int(ceiling * 0.99)              # bright but not saturated
    if extra_peak:
        a[5, 5] = extra_peak
    p = str(tmp_path / name)
    fits.PrimaryHDU(a).writeto(p, overwrite=True)
    return p


def test_raw_saturation_needs_a_plateau_and_uses_the_sensor_ceiling(tmp_path):
    paths = {1: _raw(tmp_path, "a.fits", True), 2: _raw(tmp_path, "b.fits", False, extra_peak=4000),
             3: str(tmp_path / "missing.fits")}
    ceiling, sat, info = S.raw_saturation(paths)
    assert ceiling == 4095 and info["unreadable"] == 1 and info["frames_with_plateau"] == 1
    assert len(sat[1]) == 11 and [10, 23] in sat[1].tolist() and [30, 30] not in sat[1].tolist()   # (+ the hot pixel)
    assert sat[2].tolist() == [[35, 45]]              # only the hot pixel: 4000 < 0.995 x 4095 is a peak, not saturation
    assert S.raw_saturation({2: paths[2]})[0] is None  # no plateau anywhere: no saturation


def test_confirm_on_calibrated_drops_hot_pixels_and_finds_the_row_order(tmp_path):
    cal = np.full((40, 50), 0.001, np.float32)
    cal[10:13, 20:23] = 0.06                                   # the star: still saturated after calibration
    cal[35, 45] = 0.0012                                       # the hot pixel: removed by the dark / cosmetic correction
    p = str(tmp_path / "pp_light_00001.fit")
    fits.PrimaryHDU(cal).writeto(p)
    rc = np.array([[r, c] for r in range(10, 13) for c in range(20, 23)] + [[35, 45]], np.int32)
    kept, same, info = S.confirm_on_calibrated({1: rc}, {1: p}, 4095.0)
    assert same is True and len(kept[1]) == 9 and [35, 45] not in kept[1].tolist() and info["hot_pixels"] == 1
    flipped = np.c_[39 - rc[:, 0], rc[:, 1]].astype(np.int32)            # a TOP-DOWN raw file
    kept, same, _ = S.confirm_on_calibrated({1: flipped}, {1: p}, 4095.0)
    assert same is False and len(kept[1]) == 9
    assert S.confirm_on_calibrated({1: rc[-1:]}, {1: p}, 4095.0)[1] is None


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


def test_norm_peak():
    assert S._norm_peak(np.full((3, 8, 8), 0.5, np.float32)) == 1.0
    a = np.full((3, 8, 8), 0.5, np.float32)
    a[0, 2:5, 2:5] = 2.5                                   # a restored star core above the ceiling: the image is scaled
    assert S._norm_peak(a) == pytest.approx(0.4)
    b = np.full((3, 8, 8), 0.5, np.float32)
    b[1, 4, 4] = 2.5                                       # a lone hot pixel does not set the scale
    assert S._norm_peak(b) == 1.0


# --------------------------------------------------------------------------- WCS
def test_rescale_wcs():
    h = fits.Header()
    for k, v in {"CRPIX1": 101.0, "CRPIX2": 51.0, "CD1_1": -1e-3, "CD1_2": 2e-4, "CD2_1": 2e-4, "CD2_2": 1e-3,
                 "A_2_0": 1e-6, "B_0_2": 2e-6, "A_ORDER": 2}.items():
        h[k] = v
    g = S.rescale_wcs(h, 2, (10, 20), 1)
    assert g["CRPIX1"] == 2 * (101 - 1 - 10) + 1 + 1 and g["CRPIX2"] == 2 * (51 - 1 - 20) + 1 + 1
    assert g["CD1_1"] == pytest.approx(-5e-4) and g["CD2_2"] == pytest.approx(5e-4)
    assert g["A_2_0"] == pytest.approx(1e-6 * 2 ** (1 - 2)) and g["B_0_2"] == pytest.approx(2e-6 * 2 ** -1)
    assert g["A_ORDER"] == 2
    same = S.rescale_wcs(h, 1)
    assert same["CRPIX1"] == 101.0 and same["CD1_1"] == -1e-3
    off = S.rescale_wcs(h, 1, (10, 20))
    assert off["CRPIX1"] == 91.0 and off["CRPIX2"] == 31.0


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


def test_run_denoise_fails_when_torch_is_missing(tmp_path, monkeypatch, caplog):
    from sirilpipe import post, session
    s = _session(tmp_path)
    seen = []
    monkeypatch.setattr(post, "restore", _fake_post(seen).restore)
    monkeypatch.setattr(S, "available", lambda: {"ok": False, "torch": False, "cuda": False, "mps": False,
                                                 "reason": "torch is not installed"})
    monkeypatch.setattr(session, "_IMAGEMM_WARNED", False)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="ImageMM unavailable: torch is not installed"):
            s.run_denoise({"deconv_method": "imagemm"})
    assert seen == []                                        # (no BlurXTerminator / RL restore in its place)
    rm = json.load(open(s._p("restore_meta.json")))
    assert rm["imagemm"] == "unavailable" and "torch" in rm["imagemm_reason"]
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


def test_run_denoise_imagemm_failure_fails_the_job_without_bxt(tmp_path, monkeypatch):
    from sirilpipe import post
    s = _session(tmp_path)
    seen = []
    monkeypatch.setattr(post, "restore", _fake_post(seen).restore)
    monkeypatch.setattr(S, "available", lambda: {"ok": True, "torch": True, "cuda": False, "mps": False, "reason": ""})

    def fail(*a, **k):
        raise RuntimeError("not enough GPU memory\nmore")
    monkeypatch.setattr(S, "run", fail)
    with pytest.raises(RuntimeError, match="ImageMM failed: not enough GPU memory"):
        s.run_denoise({"deconv_method": "imagemm"})
    assert seen == []                                        # (no BlurXTerminator / RL restore in its place)
    rm = json.load(open(s._p("restore_meta.json")))
    assert rm["imagemm"] == "failed" and rm["imagemm_reason"] == "not enough GPU memory"


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


# --------------------------------------------------------------------------- run() end to end (Siril faked, needs torch)
# (the PSF cut-outs are 81 x 81: the stars lie >= 45 px inside the stack's box)
STARS = [(60.3, 65.7, 0.5), (120.6, 70.2, 0.4), (90.1, 115.4, 0.6), (130.4, 120.9, 0.3), (65.2, 110.5, 0.45), (100.5, 55.5, 0.35)]


def _sky(w, h, sigma, rng=None, noise=0.0):
    yy, xx = np.mgrid[:h, :w].astype(np.float64)
    img = np.full((h, w), 0.02)
    for cx, cy, a in STARS:
        img += a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    if rng is not None:
        img = img + rng.normal(0, noise, img.shape)
    return img


def _fake_session(tmp_path, n=6, cw=200, ch=190, crop=(5, 5, 190, 180)):
    d = tmp_path / "sess"
    proc = d / "process"
    (proc / "cache").mkdir(parents=True)
    clean = _sky(cw, ch, 2.0)
    x, y, w, h = crop
    hdr = fits.Header()
    hdr["CRPIX1"], hdr["CRPIX2"], hdr["CD1_1"], hdr["CD2_2"] = 40.0, 35.0, -1e-3, 1e-3
    fitsio.write_rgb(str(d / "stack.fits"), np.repeat(clean[None, y:y + h, x:x + w], 3, 0).astype(np.float32), hdr)
    meta = {"crop": list(crop), "stack_canvas": [cw, ch], "scale": 1.0}
    json.dump(meta, open(d / "stack_meta.json", "w"))
    rows = [{"name": f"f{i}.fits", "included": i != 1, "weight": 1.0, "path": str(tmp_path / f"raw{i}.fits")} for i in range(n)]
    json.dump(rows, open(d / "frames.json", "w"))
    json.dump({"seq": "pp_light", "n": n, "debayer": False, "hdr": False}, open(proc / "state.json", "w"))
    lines = ["S 'pp_light_' 1 %d %d 5 0 6 0 0 0" % (n, n), "L 1"] + [f"I {i + 1} 1" for i in range(n)]
    lines += ["R0 2 3 0.9 0 0.0004 50 H 1 0 0 0 1 0 0 0 1"] * n
    (proc / "pp_light_.seq").write_text("\n".join(lines) + "\n")
    return str(d), meta, clean


def _fake_preprocess(cw, ch):
    def prep(session_dir, meta, inc, rows, say, checkpoint, cosmetic=True, subsky=False, defer_masks=False):
        proc = os.path.join(session_dir, "process")
        rng = np.random.default_rng(7)
        nums = [i + 1 for i in inc]
        lines = ["S 'r_bkg_imm_pp_light_' 1 %d %d 5 0 6 1 0 0" % (len(nums), len(nums)), "L 3"]
        lines += [f"I {k} 1 {cw},{ch}" for k in nums] + ["R1 2 3 0.9 0 0.0004 50 H 1 0 0 0 1 0 0 0 1"] * len(nums)
        open(os.path.join(proc, "r_bkg_imm_pp_light_.seq"), "w").write("\n".join(lines) + "\n")
        for k in nums:
            img = _sky(cw, ch, 2.0, rng, 0.003).astype(np.float32)
            img[:, :3] = 0                                     # outside this frame's footprint
            fits.PrimaryHDU(np.repeat(img[None], 3, 0)).writeto(os.path.join(proc, f"r_bkg_imm_pp_light_{k:05d}.fit"))
            with open(os.path.join(proc, "cache", f"r_bkg_imm_pp_light_{k:05d}.lst"), "w") as f:
                f.write("# star#\tlayer\tB\tA\tbeta\tX\tY\n")
                for j, (cx, cy, a) in enumerate(STARS):    # Siril's convention: X = col + 0.5, Y = H - row - 0.5
                    f.write(f"{j + 1}\t1\t0.02\t{a}\t-1\t{cx + 0.5:.3f}\t{ch - cy - 0.5:.3f}\n")
        return {"seq": "pp_light", "nums": nums, "frames": {k: os.path.join(proc, f"r_bkg_imm_pp_light_{k:05d}.fit") for k in nums},
                "masks": {}, "stars": {k: os.path.join(proc, "cache", f"r_bkg_imm_pp_light_{k:05d}.lst") for k in nums},
                "seq_file": os.path.join(proc, "r_bkg_imm_pp_light_.seq"), "mask_seq_file": None,
                "saturation": {"ceiling": None}}
    return prep


def _fwhm_at(img, cx, cy, r=7):
    ix, iy = int(round(cx)), int(round(cy))
    c = np.maximum(img[iy - r:iy + r + 1, ix - r:ix + r + 1] - np.median(img), 0)
    yy, xx = np.mgrid[:c.shape[0], :c.shape[1]]
    s = c.sum()
    my, mx = (c * yy).sum() / s, (c * xx).sum() / s
    return 2.3548 * np.sqrt(((c * ((yy - my) ** 2 + (xx - mx) ** 2)).sum() / s) / 2), (mx - r + ix, my - r + iy)


def test_run_end_to_end_on_synthetic_frames(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    d, meta, clean = _fake_session(tmp_path)
    monkeypatch.setattr(S, "siril_preprocess", _fake_preprocess(200, 190))
    out = S.run(d, meta, {"device": "cpu", "imagemm_iterations": 15})
    img, hdr = fitsio.read_rgb(os.path.join(d, S.OUT_FITS))
    assert img.shape == (3, 180, 190) and np.isfinite(img).all()
    assert out["frames_used"] == 5 and out["frames_total"] == 5 and out["r"] == 1 and out["shape"] == [190, 180]
    prep = out["preparation"]
    assert prep["frames_flipped"]["flipped"] is False and max(abs(v) for v in prep["alignment_shift_px"]) < 0.3
    assert prep["stars_median"] == len(STARS) and out["psf"]["stars_median"] == len(STARS)
    for cx, cy, _ in STARS:                                        # sharper, and where the stack has them
        w_in, _ = _fwhm_at(clean[5:185, 5:195], cx - 5, cy - 5)
        w_out, (mx, my) = _fwhm_at(img[1], cx - 5, cy - 5)
        assert w_out < 0.85 * w_in and abs(mx - (cx - 5)) < 0.3 and abs(my - (cy - 5)) < 0.3
    assert hdr["CRPIX1"] == 40.0
    assert not os.path.exists(os.path.join(d, S.WORK))             # store removed
    assert not os.path.exists(os.path.join(d, "process", "r_bkg_imm_pp_light_00001.fit"))
    assert json.load(open(os.path.join(d, S.OUT_META)))["method"].startswith("VeraLux ImageMM")


def test_run_region_and_kept_store(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    d, meta, _ = _fake_session(tmp_path)
    calls = []
    prep = _fake_preprocess(200, 190)
    monkeypatch.setattr(S, "siril_preprocess", lambda *a, **k: calls.append(1) or prep(*a, **k))
    # (a region where all six stars lie >= 42 px inside, for the 81 px PSF cut-outs)
    sp = {"device": "cpu", "imagemm_iterations": 3, "imagemm_region": [5, 5, 180, 170], "imagemm_keep_prepared": True}
    out = S.run(d, meta, sp)
    img, hdr = fitsio.read_rgb(os.path.join(d, S.OUT_FITS))
    assert img.shape == (3, 170, 180) and out["region_offset"] == [5, 5] and hdr["CRPIX1"] == 35.0
    out2 = S.run(d, meta, sp)                                          # the store is reused: no Siril step
    assert len(calls) == 1 and out2["preparation"] == out["preparation"]
    assert os.path.exists(os.path.join(d, S.WORK, "frames.npy"))


def test_match_to_stack_on_a_dark_star_field_keeps_each_channels_scale():
    """M 31's outer field 2026-10-23: a line through the two smoothed images had no lever arm on dark sky and the
    field's brightest star (clipped in the stack only) set green's slope to 0.4.  The stars' flux ratio sets it now."""
    rng = np.random.default_rng(3)
    N = 400
    yy, xx = np.mgrid[:N, :N].astype(np.float64)
    out = np.full((3, N, N), 0.02) + rng.normal(0, 0.001, (3, N, N))
    for k in range(40):
        cx, cy = rng.uniform(20, N - 20, 2)
        out += 0.05 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 2.0)[None]
    out += 40.0 * np.exp(-((xx - 200) ** 2 + (yy - 200) ** 2) / 8.0)[None]       # the restored bright star
    gains = np.array([0.8, 1.3, 0.6])
    ref = np.minimum(out * gains[:, None, None] + 0.01, 1.0)                     # the stack: clipped at 1
    o, res = S.match_to_stack(out.astype(np.float32), ref.astype(np.float32))
    for c in range(3):
        assert abs(res[c][0] / gains[c] - 1.0) < 0.03, res


def test_confirm_on_calibrated_keeps_saturated_pixels_under_a_strong_flat(tmp_path):
    """M 31 2026-10-23: red / blue flats ~3 x green's put saturated blue pixels below half the ceiling after calibration,
    and the single level dropped them as hot pixels; each CFA lattice has its own saturated level."""
    cal = np.full((40, 50), 0.001, np.float32)
    cal[10:16, 20:26] = 0.06                                   # the star: saturated
    for r in range(10, 16):
        for c in range(20, 26):
            if r % 2 == 1 and c % 2 == 1:
                cal[r, c] = 0.02                                # its blue lattice: saturated, under a flat 3 x green's
    cal[35, 45] = 0.0012                                       # a hot pixel, corrected
    p = str(tmp_path / "pp_light_00001.fit")
    fits.PrimaryHDU(cal).writeto(p)
    rc = np.array([[r, c] for r in range(10, 16) for c in range(20, 26)] + [[35, 45]], np.int32)
    kept, same, info = S.confirm_on_calibrated({1: rc}, {1: p}, 4095.0)
    k = kept[1].tolist()
    assert len(k) == 36 and [11, 21] in k and [35, 45] not in k


def test_saturation_masks_added_after_the_frames_match_masks_read_with_them(tmp_path):
    """The frames go into the store first (each registered frame deleted as it goes), the saturation masks are added in
    a second pass: the store's saturated pixels and statistics are those of the one-pass build."""
    proc = tmp_path / "process"
    (proc / "cache").mkdir(parents=True)
    cw, ch = 60, 40
    nums = [1, 2, 3]
    rng = np.random.default_rng(3)

    def seq(name, layers):
        lines = [f"S '{name}_' 1 3 3 5 0 6 1 0 0", f"L {layers}"] + [f"I {k} 1 {cw},{ch}" for k in nums]
        lines += ["R1 2 3 0.9 0 0.0004 50 H 1 0 0 0 1 0 0 0 1"] * len(nums)
        (proc / f"{name}_.seq").write_text("\n".join(lines) + "\n")

    seq("r_imm_pp_light", 3)
    seq("r_immsat", 3)
    for k in nums:
        img = (0.1 + rng.normal(0, 0.01, (3, ch, cw))).astype(np.float32)
        fits.PrimaryHDU(img).writeto(proc / f"r_imm_pp_light_{k:05d}.fit")
        m = np.zeros((3, ch, cw), np.uint8)
        m[k % 3, 10 + k:14 + k, 20:24] = 255
        fits.PrimaryHDU(m).writeto(proc / f"r_immsat_{k:05d}.fit")
        (proc / "cache" / f"r_imm_pp_light_{k:05d}.lst").write_text("# star#\tlayer\tB\tA\tbeta\tX\tY\n")

    def prep():
        return {"frames": {k: str(proc / f"r_imm_pp_light_{k:05d}.fit") for k in nums},
                "masks": {k: str(proc / f"r_immsat_{k:05d}.fit") for k in nums},
                "stars": {k: str(proc / "cache" / f"r_imm_pp_light_{k:05d}.lst") for k in nums},
                "seq_file": str(proc / "r_imm_pp_light_.seq"), "mask_seq_file": str(proc / "r_immsat_.seq"), "resampling": "lanczos4"}

    box, canvas = (0, 0, cw, ch), (cw, ch)
    say = lambda *a: None
    one, st1 = S.build_store(str(tmp_path / "w1"), prep(), nums, ["a", "b", "c"], box, canvas, False, None, say, None)
    two, st2 = S.build_store(str(tmp_path / "w2"), prep(), nums, ["a", "b", "c"], box, canvas, False, None, say, None,
                             consume=True, with_masks=False)
    assert not any(os.path.exists(proc / f"r_imm_pp_light_{k:05d}.fit") for k in nums)          # (consumed)
    assert all(os.path.exists(proc / f"r_immsat_{k:05d}.fit") for k in nums)                   # (not yet)
    S.add_masks(two, st2, prep(), nums, box, canvas, False, None, say, None)
    assert np.array_equal(one.sat, two.sat) and one.sat.any()
    assert st1["saturated_fraction"] == st2["saturated_fraction"]
    assert np.allclose(st1["invalid_fraction"], st2["invalid_fraction"])
    assert not any(os.path.exists(proc / f"r_immsat_{k:05d}.fit") for k in nums)
