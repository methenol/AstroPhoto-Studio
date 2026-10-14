"""calibration.py: DWARF CALI_FRAME matching on the real library layout, generic folders, raw-frame sets."""
from __future__ import annotations

import os

from sirilpipe import calibration, dataset

from .conftest import load, touch


def _dwarf_set(tmp_path, fake, filter_name="Duo-Band", temp=42):
    heads = load("dwarf_headers.json")
    d = tmp_path / "Astronomy" / "DWARF_RAW_TELE_M 42_EXP_15_GAIN_60_x"
    for n, cards in heads.items():
        touch(str(d / n))
        fake[n] = {**cards, "DET-TEMP": temp, "FILTER": filter_name}
    for n, cards in load("dwarf_cali_headers.json").items():
        sub = "bias" if n.startswith("bias") else "dark" if n.startswith("dark") else "flat"
        touch(str(tmp_path / "Astronomy" / "CALI_FRAME" / sub / "cam_0" / n))
        fake[n] = cards
    return dataset.scan([str(d)])


def test_dwarf_dark_exact_exposure_gain_nearest_temperature(tmp_path, fake_headers):
    fs = _dwarf_set(tmp_path, fake_headers, temp=42)
    rep, masters = calibration.find_and_choose(fs)
    assert rep["dark"]["file"] == "dark_exp_15.000000_gain_60_bin_1_38C_stack_20.fits"      # 38 C beats 27 C
    assert abs(rep["dark"]["temp_delta"] - 4.0) < 0.5   # lights average 41-43 C
    assert "delta" in rep["dark"]["how"]
    assert not any("5 C" in n for n in rep["notes"])


def test_dwarf_dark_temperature_delta_above_5c_is_flagged(tmp_path, fake_headers):
    fs = _dwarf_set(tmp_path, fake_headers, temp=47)
    rep, _ = calibration.find_and_choose(fs)
    assert rep["dark"]["file"].endswith("38C_stack_20.fits")
    assert any("> 5 C" in n for n in rep["notes"])


def test_dwarf_flat_only_by_unambiguous_ir_rule(tmp_path, fake_headers):
    fs = _dwarf_set(tmp_path, fake_headers, "Duo-Band")
    rep, _ = calibration.find_and_choose(fs)
    assert rep["flat"]["file"] == "flat_gain_2_bin_1_ir_2.fits" and "vendor documentation" in rep["flat"]["how"]
    assert rep["bias"]["file"] == "bias_gain_2_bin_1.fits" and rep["flat_bias"] is not None
    fs2 = _dwarf_set(tmp_path, fake_headers, "Astro")
    assert calibration.find_and_choose(fs2)[0]["flat"]["file"].endswith("ir_1.fits")
    fs3 = _dwarf_set(tmp_path, fake_headers, "LP")                       # not one of the three names: no flat
    rep3, _ = calibration.find_and_choose(fs3)
    assert rep3["flat"] is None and any("flat not used" in n for n in rep3["notes"])


def test_prefs_switch_off_or_choose_by_hand(tmp_path, fake_headers):
    fs = _dwarf_set(tmp_path, fake_headers)
    rep, masters = calibration.find_and_choose(fs, {"flat": "none"})
    assert rep["flat"] is None and "flat" in rep["chosen_by_hand"]
    assert rep["dark"] is not None
    rep, _ = calibration.find_and_choose(fs, {"enabled": False})
    assert rep["dark"] is None and rep["summary"] == "Calibration switched off."
    hand = next(m.path for m in masters if m.kind == "dark" and "27C" in m.name)
    rep, _ = calibration.find_and_choose(fs, {"dark": hand})
    assert rep["dark"]["file"].endswith("27C_stack_3.fits") and rep["dark"]["how"] == "chosen by hand"


def test_other_rigs_never_use_dwarf_library(tmp_path, fake_headers):
    """An ASI585 3840x2160 next to a DWARF's CALI_FRAME must not pick it up."""
    for n, cards in load("dwarf_cali_headers.json").items():
        sub = "bias" if n.startswith("bias") else "dark" if n.startswith("dark") else "flat"
        touch(str(tmp_path / "CALI_FRAME" / sub / "cam_0" / n))
        fake_headers[n] = cards
    d = tmp_path / "NGC 7000"
    for n, cards in load("asi585_headers.json").items():
        touch(str(d / n))
        fake_headers[n] = cards
    fs = dataset.scan([str(d)])
    rep, masters = calibration.find_and_choose(fs)
    assert masters == [] and rep["dark"] is None and rep["flat"] is None
    assert "lights-only" in rep["summary"]


def test_seestar_is_lights_only(tmp_path, fake_headers):
    d = tmp_path / "M 27_sub"
    for n, cards in load("seestar_headers.json").items():
        touch(str(d / n))
        fake_headers[n] = cards
    fs = dataset.scan([str(d)])
    rep, _ = calibration.find_and_choose(fs)
    assert rep["dark"] is None and rep["flat"] is None and "Seestar" in rep["notes"][0]


def test_generic_folders_raw_frames_become_one_master_set(tmp_path, fake_headers):
    """darks/ flats/ bias/ beside lights/: matched by exposure/gain, raw frames grouped for Siril to stack."""
    base = tmp_path / "proj"
    heads = load("asi585_headers.json")
    for n, cards in heads.items():
        touch(str(base / "lights" / n))
        fake_headers[n] = cards
    base_cards = next(iter(heads.values()))
    for i in range(4):
        n = f"dark_{i}.fit"
        touch(str(base / "darks" / n))
        fake_headers[n] = {**base_cards, "IMAGETYP": "Dark", "EXPTIME": 300.0, "CCD-TEMP": -15.0}
        n = f"darkwrong_{i}.fit"
        touch(str(base / "darks" / n))
        fake_headers[n] = {**base_cards, "IMAGETYP": "Dark", "EXPTIME": 60.0}
        n = f"flat_{i}.fit"
        touch(str(base / "flats" / n))
        fake_headers[n] = {**base_cards, "IMAGETYP": "Flat", "EXPTIME": 0.01, "GAIN": 100}
    touch(str(base / "bias" / "master_bias.fit"))
    fake_headers["master_bias.fit"] = {**base_cards, "IMAGETYP": "Bias", "EXPTIME": 0.0, "STACKCNT": 50}
    fs = dataset.scan(dataset.resolve_light_folders(str(base)))
    rep, masters = calibration.find_and_choose(fs)
    assert rep["dark"]["raw_frames"] and len(rep["dark"]["raw_frames"]) == 4
    assert rep["dark"]["master_name"].startswith("master_dark_") and rep["dark"]["exptime"] == 300.0
    assert rep["flat"]["raw_frames"] and len(rep["flat"]["raw_frames"]) == 4
    assert rep["bias"]["file"] == "master_bias.fit" and rep["bias"]["raw_frames"] is None
    listing = calibration.candidate_listing(masters)
    assert len(listing["dark"]) == 2 and len(listing["flat"]) == 1 and len(listing["bias"]) == 1


def test_darks_of_another_exposure_are_not_used(tmp_path, fake_headers):
    base = tmp_path / "p"
    heads = load("asi585_headers.json")
    for n, cards in heads.items():
        touch(str(base / "lights" / n))
        fake_headers[n] = cards
    touch(str(base / "darks" / "dark_60.fit"))
    fake_headers["dark_60.fit"] = {**next(iter(heads.values())), "IMAGETYP": "Dark", "EXPTIME": 60.0}
    fs = dataset.scan(dataset.resolve_light_folders(str(base)))
    rep, _ = calibration.find_and_choose(fs)
    assert rep["dark"] is None and any("never scaled" in n for n in rep["notes"])


def test_parse_master_dwarf_name_tokens(tmp_path, fake_headers):
    n = "dark_exp_15.000000_gain_60_bin_1_38C_stack_20.fits"
    p = touch(str(tmp_path / "CALI_FRAME" / "dark" / "cam_0" / n))
    fake_headers[n] = load("dwarf_cali_headers.json")[n]
    m = calibration.parse_master(p)
    assert (m.kind, m.exptime, m.gain, m.temp, m.binning, m.stack, m.cam) == ("dark", 15.0, 60.0, 38.0, 1, 20, "cam_0")
    assert m.is_master
    n = "flat_gain_2_bin_1_ir_2.fits"
    p = touch(str(tmp_path / "CALI_FRAME" / "flat" / "cam_0" / n))
    fake_headers[n] = load("dwarf_cali_headers.json")[n]
    assert calibration.parse_master(p).ir == 2
