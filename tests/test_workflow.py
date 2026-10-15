"""workflow.py: decisions and generated Siril scripts."""
from __future__ import annotations

import os

import pytest

from sirilpipe import workflow as w
from sirilpipe.dataset import Frame


def _stats(n=40, bad=()):
    out = []
    for i in range(n):
        out.append({"registered": True, "nbstars": 100, "roundness": 0.85, "wfwhm": 3.0 + 0.01 * i, "fwhm": 3.0})
    for i, kw in dict(bad).items():
        out[i].update(kw)
    return out


def test_keep_percent_matches_design_points():
    assert w.keep_percent(1.0) == 90.0 and w.keep_percent(0.5) == 97.0 and w.keep_percent(2.0) == 75.0
    assert 90 > w.keep_percent(1.5) > 75 and w.keep_percent(0.1) == 100.0 and w.keep_percent(10) >= 50


def test_select_frames_rules():
    st = _stats(40, {3: {"registered": False}, 4: {"nbstars": 2}, 5: {"roundness": 0.2}})
    why = w.select_frames(st, 1.0)
    assert "registration failed" in why[3][0] and "fewer than 5 stars" in why[4][0] and "elongated" in why[5][0]
    # worst 10 % by wFWHM among the remaining frames are dropped, the best are kept
    cut = [i for i, r in enumerate(why) if r and i not in (3, 4, 5)]
    assert cut and min(cut) > 30 and all("weighted FWHM" in why[i][0] for i in cut)
    assert not any(why[i] for i in range(0, 30) if i not in (3, 4, 5))
    # sensitivity 0.5 keeps more, 2.0 keeps fewer
    n = lambda s: sum(1 for r in w.select_frames(_stats(100), s) if r)
    assert n(0.5) < n(1.0) < n(2.0)
    # few frames: no percentage cut
    assert not any(w.select_frames(_stats(8), 2.0))


def test_decide_resampling_default_is_bayer_drizzle_1x():
    r = w.decide_resampling({}, 148, 3.6)
    assert (r["mode"], r["scale"], r["pixfrac"], r["kernel"]) == ("drizzle", 1.0, 1.0, "square")


def test_two_x_drizzle_only_when_rule_allows():
    ok = w.decide_resampling({"scale": 2}, 300, 1.7)
    assert ok["scale"] == 2.0 and ok["pixfrac"] == 0.5
    few = w.decide_resampling({"scale": 2}, 99, 1.7)
    assert few["scale"] == 1.0 and "only 99 frames" in few["reason"]
    soft = w.decide_resampling({"scale": 2.0}, 300, 3.1)
    assert soft["scale"] == 1.0 and "3.10 px" in soft["reason"]
    assert w.decide_resampling({"scale": 2}, 300, None)["scale"] == 1.0


def test_demosaic_mode():
    r = w.decide_resampling({"mode": "demosaic", "scale": 2}, 300, 1.5)
    assert r["mode"] == "demosaic" and r["scale"] == 1.0 and r["interp"] == "lanczos4"


def test_rejection_winsorized_never_gesdt():
    rs = w.decide_resampling({}, 100, 3.0)
    assert w.rejection({}, rs) == {"type": "w", "name": "winsorized", "low": 4.0, "high": 3.0}
    assert w.rejection({"sigma_low": 3, "sigma_high": 2.5}, rs)["low"] == 3.0
    two = w.decide_resampling({"scale": 2}, 300, 1.5)
    assert w.rejection({}, two)["type"] == "a"
    for p in ({}, {"scale": 2}, {"mode": "demosaic"}):
        assert w.rejection(p, w.decide_resampling(p, 300, 1.5))["type"] != "g"


def test_apply_script_unselect_ranges_are_one_based_and_filter_included():
    rs = w.decide_resampling({}, 20, 3.0)
    s = w.apply_script("/a b/process", "pp_light", 20, [1, 2, 3, 9], rs)
    lines = s.splitlines()
    assert lines[0] == 'cd "/a b/process"' and lines[1] == "select pp_light 1 20"
    assert "unselect pp_light 2 4" in lines and "unselect pp_light 10 10" in lines
    last = lines[-1]
    assert last.startswith("seqapplyreg pp_light -framing=max -scale=1 -drizzle -pixfrac=1 -kernel=square")
    assert last.endswith("-filter-included")
    d = w.apply_script("/p", "pp_light", 5, [], w.decide_resampling({"mode": "demosaic"}, 5, 3.0))
    assert "-drizzle" not in d and "-interp=lanczos4" in d


def test_stack_script_flags():
    rs = w.decide_resampling({}, 100, 3.0)
    s = w.stack_script("/p", "pp_light", w.rejection({}, rs), 100)
    assert "stack r_pp_light rej w 4 3 -norm=addscale -output_norm -rgb_equal -maximize -32b -weight=wfwhm -filter-included -out=stack_raw" in s
    assert "-fastnorm" not in s
    assert "-fastnorm" in w.stack_script("/p", "pp_light", w.rejection({}, rs), 1001)
    assert "-output_norm" not in w.stack_script("/p", "light", w.rejection({}, rs), 10, output_norm=False)
    assert "GESDT" not in s and " rej g " not in s


def _unit(tmp_path, masters):
    return w.Unit(index=0, dir=str(tmp_path / "process"), frames=[], masters=masters)


def test_calibrate_script_dwarf_with_dark_and_flat(tmp_path):
    u = _unit(tmp_path, {"dark": "/lib/CALI FRAME/dark.fits", "flat": "/c/flat.fit", "bias": "/lib/b.fits"})
    s = w.calibrate_script(u, debayer=False)
    cal = [ln for ln in s.splitlines() if ln.startswith("calibrate")][0]
    assert '"-dark=/lib/CALI FRAME/dark.fits"' in cal and "-flat=/c/flat.fit" in cal
    assert "-bias" not in cal                       # never bias together with a dark
    assert "-cc=dark 3 3 -cfa -equalize_cfa -prefix=pp_" in cal and "-debayer" not in cal
    assert s.startswith(f"cd {u.dir}/src\nlink light -out={u.dir}")


def test_calibrate_script_no_masters_skips_calibrate_unless_demosaic(tmp_path):
    u = _unit(tmp_path, {})
    assert "calibrate light" not in w.calibrate_script(u, False)
    assert w.seq_name([u], False) == "light" and w.seq_name([u], True) == "pp_light"
    assert "calibrate light -debayer -prefix=pp_" in w.calibrate_script(u, True)


def test_calibrate_script_bias_only_when_no_dark(tmp_path):
    u = _unit(tmp_path, {"bias": "/b.fit"})
    assert "-bias=/b.fit" in w.calibrate_script(u, False) and "-cc=dark" not in w.calibrate_script(u, False)


def test_master_scripts():
    d = w.master_script("dark", "/c/work_d", "/c/master_dark_x")
    assert "stack dark rej 3 3 -nonorm -out=/c/master_dark_x" in d and "link dark -out=/c/work_d" in d
    f = w.master_script("flat", "/c/work_f", "/c/master_flat_x", bias="/c/bias.fit")
    assert "calibrate flat -bias=/c/bias.fit" in f and "stack pp_flat rej 3 3 -norm=mul -out=/c/master_flat_x" in f
    assert "calibrate_single" in w.flat_bias_script("/c", "/f.fits", "/b.fits")


def test_register_and_merge_scripts(tmp_path):
    r = w.register_script("/p", "pp_light", 30)
    assert "select pp_light 1 30" in r and "register pp_light -2pass -transf=homography -minpairs=10 -maxstars=2000" in r
    us = [w.Unit(0, "/p/u1", []), w.Unit(1, "/p/u 2", [])]
    m = w.merge_script(us, "/p", "pp_light")
    assert 'merge /p/u1/pp_light "/p/u 2/pp_light" pp_light' in m


def test_platesolve_script_has_no_coordinates():
    f = Frame(path="/x", name="x", folder="/", object="M 42", filter="", exptime=15, gain=60, temp=None, date_obs="",
              timestamp=0, bayer="RGGB", width=10, height=10, ra=83.8, dec=-5.4, focallen=150.0, pixsize=2.0)
    s = w.platesolve_script("/s", "stack", f)
    assert "platesolve -force -focal=150 -pixelsize=2" in s and "83.8" not in s and "save stack_solved" in s


def test_stage_inputs_links_only_selected_and_cleans(tmp_path):
    subs = []
    for i in range(3):
        p = tmp_path / "raw" / f"s {i}.fits"
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b"x")
        subs.append(str(p))
    u = _unit(tmp_path, {})
    os.makedirs(u.dir)
    for stale in ("pp_light_00001.fit", "light_.seq", "r_pp_light_00001.fit"):
        open(os.path.join(u.dir, stale), "w").close()
    os.makedirs(os.path.join(u.dir, "logs"))
    open(os.path.join(u.dir, "logs", "keep.log"), "w").close()
    src = w.stage_inputs(u, subs[:2])
    assert sorted(os.listdir(src)) == ["light_00001.fit", "light_00002.fit"]
    assert os.path.realpath(os.path.join(src, "light_00002.fit")) == subs[1]
    assert not os.path.exists(os.path.join(u.dir, "pp_light_00001.fit")) and os.path.exists(os.path.join(u.dir, "logs", "keep.log"))
    assert os.listdir(u.dir).count("light_.seq") == 0


def test_every_community_constant_has_a_source_comment():
    src = open(w.__file__, encoding="utf-8").read()
    for token in ("REGISTER_ARGS", "ROUNDNESS_K", "DRIZZLE_1X", "DRIZZLE_2X", "KEEP_PERCENT", "FASTNORM_FRAMES"):
        i = src.index(token + " =") if token + " =" in src else src.index(token)
        assert "#" in src[max(0, i - 400):i + 200], token


def test_banding_scripts_split_fix_merge(tmp_path):
    u = _unit(tmp_path, {"dark": "/d.fit"})
    assert w.banding_split_script(u, "pp_light") == f"cd {u.dir}\nseqsplit_cfa pp_light\n"
    fix = w.banding_fix_script(u, "pp_light").splitlines()
    # Siril docs: "Values of 1 and 1 are often good enough"; one seqfixbanding per Bayer plane (1.4.4 has no -cfa)
    assert fix[1:] == [f"seqfixbanding CFA{k}_pp_light 1 1" for k in range(4)]
    mrg = w.banding_merge_script(u, "pp_light", "rggb").splitlines()[1]
    assert mrg == "seqmerge_cfa unband_CFA0_pp_light unband_CFA1_pp_light unband_CFA2_pp_light unband_CFA3_pp_light RGGB"
    assert w.BANDING_AMOUNT == 1.0 and w.BANDING_SIGMA == 1.0


def test_banding_install_restores_filter_and_replaces_links(tmp_path):
    import numpy as np
    from astropy.io import fits
    d = str(tmp_path)
    raw = os.path.join(d, "raw.fit")
    fits.PrimaryHDU(np.zeros((4, 4), np.float32), header=fits.Header({"FILTER": "Duo-Band"})).writeto(raw)
    os.symlink(raw, os.path.join(d, "light_00001.fit"))                  # lights-only sets: the sequence links to the raw sub
    fits.PrimaryHDU(np.ones((4, 4), np.float32), header=fits.Header({"FILTER": "Duo-Band_CFA0"})).writeto(os.path.join(d, "mCFA_unband_CFA0_light_00001.fit"))
    open(os.path.join(d, "mCFA_unband_CFA0_light_.seq"), "w").close()
    assert w.banding_install(d, "light") == 1
    assert not os.path.islink(os.path.join(d, "light_00001.fit"))
    assert fits.getheader(os.path.join(d, "light_00001.fit"))["FILTER"] == "Duo-Band"
    assert fits.getdata(raw).max() == 0                                   # the raw sub is never written
    w.banding_cleanup(d, "light", ("mCFA_unband_CFA",))
    assert not os.path.exists(os.path.join(d, "mCFA_unband_CFA0_light_.seq"))


def test_image_banding_ratio_reference_noise_and_gate():
    import numpy as np
    rng = np.random.default_rng(1)
    rows = rng.normal(0, 0.02, (400, 1)).astype(np.float32)               # row stripes
    noisy = rng.normal(0, 1.0, (3, 400, 600)).astype(np.float32) + rows
    smooth = rows + rng.normal(0, 0.02, (3, 400, 600)).astype(np.float32)  # a "denoised" image: white noise gone, stripes kept
    own = w.image_banding_ratio(smooth)
    ref = w.image_banding_ratio(smooth, noise_from=noisy)
    assert own["banded"] and own["ratio"] > ref["ratio"]                   # against the denoised noise the ratio is inflated
    assert ref["needed"] == pytest.approx(1 + 5 / (800 ** 0.5), abs=1e-3)
    assert not w.image_banding_ratio(rng.normal(0, 1, (3, 400, 600)))["banded"]
