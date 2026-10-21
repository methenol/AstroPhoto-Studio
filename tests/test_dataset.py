"""dataset.py: skipping rules, grouping, kind / dual-band / pixel scale on real headers of the three rigs."""
from __future__ import annotations

import os

from sirilpipe import dataset

from .conftest import load, touch


def _install(tmp_path, fake, fixture: str, sub: str, extra_names=(), limit=None):
    d = tmp_path / sub
    heads = load(fixture)
    for n, cards in list(heads.items())[:limit]:
        touch(str(d / n))
        fake[n] = cards
    for n in extra_names:
        touch(str(d / n))
    return str(d), heads


def test_dwarf_session(tmp_path, fake_headers):
    d, heads = _install(tmp_path, fake_headers, "dwarf_headers.json", "DWARF_RAW_TELE_M 42_EXP_15_GAIN_60_x",
                        extra_names=["failed_M 42_15s60_Duo-Band_20261010-041927508_41C.fits",
                                     "stacked-16_M 42_15s60_Duo-Band_20261010-041213041.fits",
                                     "img_reference.png", "stacked_thumbnail.jpg"])
    fake_headers["failed_M 42_15s60_Duo-Band_20261010-041927508_41C.fits"] = load("dwarf_failed_headers.json")[
        next(iter(load("dwarf_failed_headers.json")))]
    fs = dataset.scan([d])
    # the live stack's reject is read and graded like every other sub (README), marked as such
    assert len(fs.frames) == len(heads) + 1 and "failed" not in fs.skipped
    assert [f.name for f in fs.frames if f.device_rejected] == ["failed_M 42_15s60_Duo-Band_20261010-041927508_41C.fits"]
    assert fs.kind == "dwarf" and fs.is_dualband
    assert (fs.object, fs.filter, fs.exptime, fs.gain, fs.bayer) == ("M 42", "Duo-Band", 15.0, 60.0, "RGGB")
    assert fs.size == (3840, 2160)
    assert abs(fs.pixel_scale - 2.75) < 0.01            # 2.0 um / 150 mm
    assert fs.sensor == "Sony IMX678"
    assert fs.first.camera_slot == "cam_0"
    assert fs.first.temp == 41.0                        # DET-TEMP
    assert [f.timestamp for f in fs.frames] == sorted(f.timestamp for f in fs.frames)


def test_seestar_portrait_ignores_jpg_and_thumbnails(tmp_path, fake_headers):
    d, heads = _install(tmp_path, fake_headers, "seestar_headers.json", "M 27_sub")
    for n in list(heads):
        touch(os.path.join(d, n.replace(".fit", ".jpg")))
        touch(os.path.join(d, n.replace(".fit", "_thn.jpg")))
    touch(os.path.join(d, "Stacked_6_M 27_10.0s_LP_20260921-212646.fit"))
    fake_headers["Stacked_6_M 27_10.0s_LP_20260921-212646.fit"] = next(iter(heads.values()))
    fs = dataset.scan([d])
    assert len(fs.frames) == len(heads)
    assert fs.kind == "seestar" and fs.size == (2160, 3840) and fs.bayer == "GRBG"
    assert fs.filter == "LP" and fs.is_dualband
    assert fs.sensor == "Sony IMX585"                    # S50 Pro
    assert abs(fs.pixel_scale - 2.30) < 0.01
    assert fs.skipped.get("telescope/derived product") == 1


def test_asi585_generic_rig(tmp_path, fake_headers):
    d, heads = _install(tmp_path, fake_headers, "asi585_headers.json", "NGC 7000")
    for n in list(heads):
        touch(os.path.join(d, n.replace(".fit", "_thn.jpg")))
    fs = dataset.scan([d])
    assert fs.kind == "generic" and fs.bayer == "RGGB" and fs.filter == ""
    assert not fs.is_dualband
    assert fs.sensor == "Sony IMX585"
    assert abs(fs.pixel_scale - 1.625) < 0.01
    assert fs.first.temp == -16.5


def _sixty(tmp_path, fake_headers, heads, n):
    d2 = tmp_path / "b"
    for i, (_, cards) in enumerate(list(heads.items())[:n]):
        name = f"M 42_60s60_Duo-Band_2026101004130{i}000_41C.fits"
        touch(str(d2 / name))
        fake_headers[name] = {**cards, "EXPTIME": 60.0, "DATE-OBS": f"2026-10-10T05:00:0{i}"}
    return str(d2)


def test_exposure_groups_for_hdr(tmp_path, fake_headers):
    d, heads = _install(tmp_path, fake_headers, "dwarf_headers.json", "a")
    d2 = _sixty(tmp_path, fake_headers, heads, 3)
    fs = dataset.scan([d, d2])
    assert fs.exptime == 60.0 and len(fs.frames) == 3    # 3 x 60 s = 180 s > 6 x 15 s: 60 s is the dataset
    assert [g.exptime for g in fs.exposure_groups] == [15.0]
    assert [g.exptime for g in fs.hdr_groups()] == [15.0, 60.0]


def test_a_lone_sub_of_another_exposure_is_no_hdr_group(tmp_path, fake_headers):
    """Sh2-132 (2026-10-18): one 60 s sub beside the 30 s set (the DWARF had rejected the other 60 s subs as
    failed_) became an HDR group of one frame, a calibration unit Siril cannot make a sequence of."""
    d, heads = _install(tmp_path, fake_headers, "dwarf_headers.json", "a")
    d2 = _sixty(tmp_path, fake_headers, heads, 1)
    fs = dataset.scan([d, d2])
    assert len(fs.frames) == len(heads) and fs.exptime == 15.0
    assert fs.exposure_groups == [] and fs.hdr_groups() == []
    assert any(o["exptime"] == 60.0 and o["n"] == 1 and "reason" in o for o in fs.other_groups)


def test_size_mismatch_and_other_object_left_out(tmp_path, fake_headers):
    d, heads = _install(tmp_path, fake_headers, "dwarf_headers.json", "a")
    cards = {**next(iter(heads.values())), "NAXIS1": 1920, "NAXIS2": 1080}
    fake_headers["small.fits"] = cards
    touch(os.path.join(d, "small.fits"))
    fake_headers["other.fits"] = {**next(iter(heads.values())), "OBJECT": "M 31"}
    touch(os.path.join(d, "other.fits"))
    fs = dataset.scan([d])
    assert len(fs.frames) == len(heads)
    assert len(fs.other_groups) == 2


def test_restack_and_non_light_skipped(tmp_path, fake_headers):
    d, heads = _install(tmp_path, fake_headers, "dwarf_headers.json", "a", limit=3)
    fake_headers["r.fits"] = {**next(iter(heads.values())), "RESTACK": 1}
    fake_headers["dk.fits"] = {**next(iter(heads.values())), "IMAGETYP": "Dark Frame"}
    for n in ("r.fits", "dk.fits", "dark_1.fits"):
        touch(os.path.join(d, n))
    fs = dataset.scan([d])
    assert len(fs.frames) == 3
    assert fs.skipped["restacked"] == 1 and fs.skipped["calibration frame"] == 2
    assert len(fs.calibration_files) == 2


def test_multi_night_names_and_rejected_folder(tmp_path, fake_headers):
    heads = load("dwarf_headers.json")
    n0, cards = next(iter(heads.items()))
    paths = []
    for night in ("n1", "n2", "lights/rejected"):
        p = tmp_path / night
        touch(str(p / "light_0001.fits"))
        paths.append(str(p))
    fake_headers["light_0001.fits"] = cards
    fs = dataset.scan(sorted(paths))
    names = sorted(f.name for f in fs.frames)
    assert len(names) == 2 and names[0].startswith("n1__") and names[1].startswith("n2__")   # rejected ignored
    assert [os.path.basename(d) for d, _ in fs.nights] == ["n1", "n2"]
    # only the rejected folder given: used anyway
    assert len(dataset.scan([paths[2]]).frames) == 1


def test_resolve_light_folders(tmp_path, fake_headers):
    touch(str(tmp_path / "t" / "Lights" / "a.fits"))
    touch(str(tmp_path / "t" / "Lights" / "rejected" / "b.fits"))
    r = dataset.resolve_light_folders(str(tmp_path / "t"))
    assert [os.path.basename(x) for x in r] == ["Lights", "rejected"]
    touch(str(tmp_path / "u" / "a.fit"))
    assert dataset.resolve_light_folders(str(tmp_path / "u")) == [str(tmp_path / "u")]
    assert dataset.resolve_light_folders(str(tmp_path / "empty")) == [str(tmp_path / "empty")]


def test_limit_keeps_first_subs(tmp_path, fake_headers):
    d, heads = _install(tmp_path, fake_headers, "dwarf_headers.json", "a")
    assert len(dataset.scan([d], limit=2).frames) == 2


def test_dualband_names():
    for f in ("Duo-Band", "LP", "L-eXtreme", "L-eNhance", "Dual Band", "ALP-T"):
        assert dataset.is_dualband(f), f
    for f in ("", "Astro", "UV/IR Block", "L", "Optolong L-Pro"):
        assert not dataset.is_dualband(f), f
