"""Checks of astrophoto/calibration.py and instruments.py on a synthetic DWARF 3 drive.

    python experiments/test_calibration.py

A DWARF 3 writes its subs with few header keys, and its factory / user masters to
Astronomy/CALI_FRAME.  This builds that layout with a known sky, pedestal, dark current
(at a different sensor temperature than the lights), hot pixels and vignetting, and checks
that the right masters are found and chosen, the thermal scale is recovered per sub, and
the calibrated subs are flat and free of hot pixels.  A generic camera's darks/ folder of
individual frames is checked as well.
"""
import json
import os
import sys
import tempfile

import numpy as np
from astropy.io import fits

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from astrophoto import calibration, instruments  # noqa: E402
from astrophoto.frames import build_defect_map, discover, read_frame  # noqa: E402

H, W = 192, 256
PED = 800.0                       # black level (ADU)
EXP, GAIN = 15.0, 60
T_DARK, T_LIGHT = 38.0, 31.0
K_TRUE = 2 ** ((T_LIGHT - T_DARK) / 6.0)   # dark current doubles every 6 C
rng = np.random.default_rng(1)


def write_u16(path, data, hdr=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fits.writeto(path, np.clip(np.round(data), 0, 65535).astype(np.uint16), hdr, overwrite=True)


def model():
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = ((yy - H / 2) ** 2 + (xx - W / 2) ** 2) / (0.5 * np.hypot(H, W)) ** 2
    vign = (1 - 0.45 * r2).astype(np.float32)
    qe = np.ones((H, W), np.float32)                       # RGGB site sensitivities of the flat lamp
    qe[0::2, 0::2], qe[1::2, 1::2] = 0.6, 0.8
    bias = PED + rng.normal(0, 3, (H, W)).astype(np.float32)        # fixed-pattern offset
    thermal = rng.gamma(2.0, 4.0, (H, W)).astype(np.float32)        # dark current at T_DARK, EXP
    hot = rng.choice(H * W, 400, replace=False)
    thermal.ravel()[hot] += rng.uniform(300, 4000, 400)
    return vign, qe, bias, thermal, hot


def build_dwarf_drive(base):
    vign, qe, bias, thermal, hot = model()
    cal = os.path.join(base, "Astronomy", "CALI_FRAME")
    write_u16(os.path.join(cal, "bias", "cam_0", "bias_gain_2_bin_1.fits"), bias)
    write_u16(os.path.join(cal, "dark", "cam_0", f"dark_exp_{EXP:.6f}_gain_{GAIN}_bin_1_{T_DARK:.0f}C_stack_10.fits"),
              bias + thermal + rng.normal(0, 1, (H, W)))
    write_u16(os.path.join(cal, "dark", "cam_0", f"dark_exp_{EXP:.6f}_gain_{GAIN}_bin_1_50C_stack_10.fits"),
              bias + 2 * thermal)
    write_u16(os.path.join(cal, "dark", "cam_0", "dark_exp_30.000000_gain_60_bin_1_38C_stack_10.fits"),
              bias + 2 * thermal)
    write_u16(os.path.join(cal, "dark", "cam_1", f"dark_exp_{EXP:.6f}_gain_{GAIN}_bin_1_31C_stack_10.fits"),
              bias * 0 + 5000)                              # the wide-angle camera's: must never be picked
    # master flats stored with the pedestal still in them, one per filter
    write_u16(os.path.join(cal, "flat", "cam_0", "flat_gain_2_bin_1_ir_1.fits"), bias + 30000 * vign * qe)
    write_u16(os.path.join(cal, "flat", "cam_0", "flat_gain_2_bin_1_ir_2.fits"), bias + 20000 * (1 - 0.1 * vign) * qe)

    sess = os.path.join(base, "Astronomy", f"DWARF_RAW_TELE_M 31_EXP_{EXP:g}_GAIN_{GAIN}_2025-10-01-21-30-00-100")
    os.makedirs(sess, exist_ok=True)
    json.dump({"target": "M 31", "exp": EXP, "gain": GAIN, "ir": "Astro", "binning": "1*1",
               "minTemp": 30, "maxTemp": 32, "shotsTaken": 8, "shotsStacked": 7, "RA": 10.68, "DEC": 41.27},
              open(os.path.join(sess, "shotsInfo.json"), "w"))
    sky = 400.0
    names = []
    for t in range(8):
        stars = np.zeros((H, W), np.float32)
        for _ in range(25):
            y, x = rng.integers(4, H - 4), rng.integers(4, W - 4)
            stars[y - 1:y + 2, x - 1:x + 2] += rng.uniform(200, 3000) * np.array([[.3, .6, .3], [.6, 1, .6], [.3, .6, .3]])
        signal = (sky + stars) * vign
        light = bias + K_TRUE * thermal + rng.poisson(np.maximum(signal + K_TRUE * thermal, 0)) - K_TRUE * thermal
        light += rng.normal(0, 2.5, (H, W))
        pre = "failed_" if t == 7 else ""
        name = f"{pre}M 31_{EXP:g}s{GAIN}_Astro_20251001-2130{t:02d}123_{T_LIGHT:.0f}C.fits"
        hdr = fits.Header()
        hdr["DATE-OBS"] = f"2025-10-01T21:30:{t:02d}"
        write_u16(os.path.join(sess, name), light, hdr)       # no BAYERPAT, FILTER, EXPTIME, TELESCOP ...
        names.append(name)
    fits.writeto(os.path.join(sess, "stacked-16_M 31_15s60_Astro.fits"), np.zeros((3, H, W), np.float32))
    return sess, vign, hot


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    return bool(ok)


def test_dwarf(base):
    print("DWARF 3 drive")
    sess, vign, hot = build_dwarf_drive(base)
    good = True
    infos = discover(sess)
    i0 = infos[0]
    good &= check(len(infos) == 8, f"8 light subs discovered, the DWARF's stack skipped ({len(infos)})")
    good &= check(sum(i.device_rejected for i in infos) == 1, "the failed_ sub is flagged, not dropped")
    good &= check(i0.telescope == "DWARF 3" and i0.bayer == "RGGB" and i0.sensor == "Sony_IMX678",
                  f"profile: {i0.telescope}, {i0.bayer}, {i0.sensor}")
    good &= check(i0.exptime == EXP and i0.gain == GAIN and i0.filter == "Astro" and i0.temp == T_LIGHT,
                  f"file name: {i0.exptime:g} s, gain {i0.gain:g}, {i0.filter}, {i0.temp:g} C")
    good &= check(i0.object == "M 31" and abs(i0.ra - 10.68) < 1e-6, "object from the name, RA/Dec from shotsInfo.json")
    good &= check(i0.focallen == 150 and i0.camera_slot == "cam_0", f"focal length {i0.focallen:g} mm, {i0.camera_slot}")

    cal = calibration.attach(infos, sess, os.path.join(base, "work"))
    rep = cal["report"]
    good &= check(rep["dark"]["file"].endswith("_38C_stack_10.fits") and "exp_15" in rep["dark"]["file"],
                  f"dark: nearest temperature, same exposure, telephoto camera ({rep['dark']['file']})")
    good &= check(rep["flat"]["file"].endswith("ir_1.fits"), f"flat of the Astro filter ({rep['flat']['file']})")
    good &= check(rep["bias"] is not None and rep["flat_note"] == "bias removed", "bias found; removed from the flat")
    lo, hi = rep["dark_scale"]
    good &= check(abs(lo - K_TRUE) < 0.06 and abs(hi - K_TRUE) < 0.06,
                  f"thermal scale {lo:.3f}-{hi:.3f} per sub (true {K_TRUE:.3f})")
    good &= check(abs(i0.bias - PED) < 5, f"pedestal {i0.bias:.1f} ADU (true {PED:g})")
    print("   ", cal["summary"])

    raw = read_frame(i0)
    plain = fits.getdata(i0.path).astype(np.float32) - PED
    sky_c, sky_r = raw[H // 2 - 20:H // 2 + 20, W // 2 - 20:W // 2 + 20], raw[4:30, 4:30]
    ratio = np.median(sky_r) / np.median(sky_c)
    ratio0 = np.median(plain[4:30, 4:30]) / np.median(plain[H // 2 - 20:H // 2 + 20, W // 2 - 20:W // 2 + 20])
    good &= check(abs(ratio - 1) < 0.05, f"vignetting removed: corner / centre {ratio0:.2f} -> {ratio:.2f}")
    mask = np.zeros(H * W, bool)
    mask[hot] = True
    res_hot = np.abs(raw.ravel()[mask] - np.median(raw)).mean()
    res_hot0 = np.abs(plain.ravel()[mask] - np.median(plain)).mean()
    good &= check(res_hot < 0.1 * res_hot0, f"hot pixels subtracted: mean excess {res_hot0:.0f} -> {res_hot:.0f} ADU")
    dm = build_defect_map(infos)
    good &= check(dm.shape == (H, W), f"defect map with the masters' defects ({int(dm.sum())} px)")

    os.environ["ASTROPHOTO_CALIB"] = "off"
    good &= check(calibration.attach(discover(sess), sess, os.path.join(base, "work")) is None,
                  "ASTROPHOTO_CALIB=off disables it")
    os.environ.pop("ASTROPHOTO_CALIB")
    return good


def test_generic(base):
    print("generic camera: darks/ of individual frames beside lights/")
    good = True
    top = os.path.join(base, "generic")
    bias = 256.0 + rng.normal(0, 2, (H, W))
    thermal = rng.gamma(2.0, 3.0, (H, W))
    for t in range(5):
        h = fits.Header({"IMAGETYP": "Dark Frame", "EXPTIME": 60.0, "GAIN": 100, "CCD-TEMP": -10.0,
                         "INSTRUME": "ZWO ASI2600MC Pro", "BAYERPAT": "RGGB"})
        write_u16(os.path.join(top, "darks", f"dark_{t}.fits"), bias + thermal + rng.normal(0, 3, (H, W)), h)
    for t in range(4):
        h = fits.Header({"IMAGETYP": "Light Frame", "EXPTIME": 60.0, "GAIN": 100, "CCD-TEMP": -10.0,
                         "INSTRUME": "ZWO ASI2600MC Pro", "BAYERPAT": "RGGB", "FILTER": "L-eXtreme"})
        write_u16(os.path.join(top, "lights", f"light_{t}.fits"), bias + thermal + 300 + rng.normal(0, 3, (H, W)), h)
    infos = discover(os.path.join(top, "lights"))
    cal = calibration.attach(infos, os.path.join(top, "lights"), os.path.join(base, "work2"))
    good &= check(cal is not None and cal["report"]["dark"]["stack"] == 5, "5 darks median-combined into a master")
    good &= check(infos[0].telescope == "" and infos[0].focallen == 250, "no profile: header values and old defaults")
    raw = read_frame(infos[0])
    good &= check(abs(np.median(raw) - 300) < 3 and raw.std() < 6, f"dark removed: median {np.median(raw):.1f} (300)")
    from astrophoto.postprocess import is_narrowband
    good &= check(is_narrowband("L-eXtreme") and is_narrowband("Duo-Band") and not is_narrowband("Astro"),
                  "dual-band filters recognised (L-eXtreme, Duo-Band; not Astro)")
    return good


def test_names():
    print("DWARF file names")
    good = True
    d = instruments.parse_dwarf_name("failed_C 20_15s60_Duo-Band_20260814-041951434_37C.fits")
    good &= check(d.get("failed") and d["object"] == "C 20" and d["filter"] == "Duo-Band" and d["temp"] == 37
                  and d["date_obs"] == "2026-08-14T04:19:51.434000", f"parsed {d}")
    m = calibration.parse_master.__doc__ is not None
    good &= check(instruments.filter_code("Duo-Band") == 2 and instruments.filter_code("Astro") == 1
                  and instruments.filter_code("VIS") == 0 and m, "filter codes VIS 0, Astro 1, Duo-Band 2")
    return good


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        ok = test_names() & test_dwarf(tmp) & test_generic(tmp)
    print("all passed" if ok else "FAILURES")
    sys.exit(0 if ok else 1)
