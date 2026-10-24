"""Unit tests of sirilpipe.tools: RC-Astro flag building from the parameter documents, NDJSON parsing, licence
report parsing, exit-code mapping (with a fake ``rc-astro``), GraXpert model detection, StarNet2 rules.

No real tool is ever started: fake executables in a temp dir stand in for rc-astro / starnet2."""
from __future__ import annotations

import json
import os
import stat
import textwrap

import numpy as np
import pytest

from sirilpipe import fitsio, tools
from sirilpipe.tools import ToolError, ToolUnavailable


# ----------------------------------------------------------------------------- flag building (parameter documents)

@pytest.fixture(scope="module")
def schemas():
    return {k: tools.load_schema(k) for k in ("bxt", "nxt", "sxt")}


def test_bxt_correct_only_pins_other_flags(schemas):
    flags = tools.build_flags(schemas["bxt"], {"correct_only": True, "ss": 0.4, "sn": 0.7})
    assert flags == ["--correct-only"]                       # ss / sn are pinned to 0 by the mode: never emitted


def test_bxt_values_and_defaults(schemas):
    flags = tools.build_flags(schemas["bxt"], {"ss": 0.35, "sn": 0.5, "ash": 0.0, "overlap": 0.2})
    assert flags == ["--ss", "0.35"]                         # sn / ash / overlap are at their defaults: omitted


def test_bxt_ranges_are_clamped(schemas):
    flags = tools.build_flags(schemas["bxt"], {"ss": 9.0, "sn": -3.0, "ash": 0.9})
    assert flags == ["--ss", "0.70", "--ash", "0.50", "--sn", "0.00"]      # schema order


def test_bxt_manual_psf_and_bool_off(schemas):
    flags = tools.build_flags(schemas["bxt"], {"ansp": False, "nsd": 3.5})
    assert flags == ["--nsd", "3.5", "--no-ansp"]
    # with the automatic PSF on a manual diameter is disabled (ansp && !lp): left out instead of a CLI conflict
    assert tools.build_flags(schemas["bxt"], {"nsd": 3.5}) == []


def test_bxt_lunar_planetary_disables_stellar_flags(schemas):
    assert tools.build_flags(schemas["bxt"], {"lp": True, "ss": 0.5, "ash": 0.2}) == ["--lp"]


def test_unknown_parameter_is_refused(schemas):
    with pytest.raises(ValueError):
        tools.build_flags(schemas["bxt"], {"sharpen_all_the_things": 1})


def test_nxt_flags_and_gui_only(schemas):
    assert tools.build_flags(schemas["nxt"], {"dn": 0.25, "it": 3}) == ["--dn", "0.25", "--it", "3"]
    assert tools.build_flags(schemas["nxt"], {"csep": True, "di": 0.8, "dc": 0.5}) == ["--di", "0.80", "--dc", "0.50"]
    assert tools.build_flags(schemas["nxt"], {"dn": 0.9}) == []          # default: omitted
    assert tools.build_flags(schemas["nxt"], {"it": 99}) == ["--it", "5"]


def test_sxt_stars_and_unscreen(schemas):
    assert tools.build_flags(schemas["sxt"], {"stars": True}) == ["--stars"]
    assert tools.build_flags(schemas["sxt"], {"stars": True, "unscreen": True}) == ["--stars", "--unscreen"]
    assert tools.build_flags(schemas["sxt"], {"unscreen": True}) == []   # disabled unless stars
    assert tools.build_flags(schemas["sxt"], {"overlap": 0.3}) == ["--overlap", "0.30"]


def test_every_flag_comes_from_the_documents(schemas):
    known = {p["flag"] for s in schemas.values() for p in s["parameters"] if "flag" in p}
    known |= {m["flag"] for s in schemas.values() for m in s.get("modes", [])}
    flags = tools.build_flags(schemas["bxt"], {"ss": 0.3, "ash": 0.1, "sn": 0.8, "ansp": False, "nsd": 2.0, "overlap": 0.3})
    for f in flags:
        if f.startswith("--") and not f.startswith("--no-"):
            assert f in known


def test_device_argument():
    assert [tools.device_arg(d) for d in ("auto", "cpu", "cuda", "cuda:1", "mps", None, "gpu0")] == \
           ["auto", "cpu", "gpu", "gpu1", "cpu", "auto", "gpu0"]


# ----------------------------------------------------------------------------- NDJSON events, licence report

SAMPLE = [
    '{"event":"info","topic":"version","cliVersion":"2.6.10","schemaVersion":6}',
    '{"event":"status","phase":"initializing","message":"Initializing"}',
    '{"event":"info","topic":"device","device":"gpu","name":"NVIDIA GeForce RTX 4080 Laptop GPU","provider":"CUDA","runtime":"onnxruntime 1.23.2"}',
    '{"event":"progress","done":0.9,"mpPerSec":0.3,"eta":52.4}',
    '{"event":"progress","done":100.0,"mpPerSec":9.3,"eta":0}',
    '{"event":"warning","message":"Float samples outside [0,1] were rescaled"}',
    '{"event":"status","phase":"saving","message":"Saving","output":"out.fits"}',
    '{"event":"status","phase":"complete","message":"Done","output":"out.fits"}',
    '{"event":"info","topic":"license","message":"Trial license expires in 12 days.","daysLeft":12}',
]


def test_parse_event_ignores_plain_text_and_garbage():
    assert tools.parse_event("Error: unknown option --x") is None
    assert tools.parse_event("{not json") is None
    assert tools.parse_event('{"no_event_key": 1}') is None
    assert tools.parse_event(SAMPLE[0])["cliVersion"] == "2.6.10"


def test_event_stream_summary():
    res = tools.RcResult()
    for ln in SAMPLE:
        ev = tools.parse_event(ln)
        if ev:
            res.feed(ev)
    assert (res.cli_version, res.schema_version) == ("2.6.10", 6)
    assert res.device["provider"] == "CUDA"
    assert res.last_progress == 100.0 and res.outputs == ["out.fits", "out.fits"]
    assert res.warnings == ["Float samples outside [0,1] were rescaled"] and not res.errors


def test_error_event_is_collected():
    res = tools.RcResult()
    res.feed(tools.parse_event('{"event":"error","message":"A license is required to use this software"}'))
    assert res.errors == ["A license is required to use this software"]


def test_license_report_parsing():
    report = {"event": "info", "topic": "licenseReport", "operation": "code", "computer": "user@astrophoto-studio",
              "account": {"email": "a@b.c"}, "activationUrl": "https://www.rc-astro.com/account",
              "licenseStatus": [{"product": "bxt", "name": "BlurXTerminator", "status": "activated", "action": "activated",
                                 "mlVersion": 5},
                                {"product": "nxt", "status": "notOwned", "message": "not owned"}],
              "attention": {"reason": "codeRejected", "message": "The code was not accepted", "command": "rc-astro license"}}
    res = tools.RcResult()
    res.feed(tools.parse_event(json.dumps(report)))
    rep = tools.normalize_report(res.report)
    assert rep["operation"] == "code" and rep["account"]["email"] == "a@b.c"
    assert [s["product"] for s in rep["licenseStatus"]] == ["bxt", "nxt"]
    assert rep["attention"]["reason"] == "codeRejected"
    assert tools.normalize_report(None) is None


def test_activation_code_normalisation():
    assert tools.normalize_code("abcd efgh") == "ABCD-EFGH"
    assert tools.normalize_code(" ab-cdefgh ") == "ABCD-EFGH"
    with pytest.raises(ValueError):
        tools.normalize_code("short")


# ----------------------------------------------------------------------------- fake rc-astro

FAKE_RC = textwrap.dedent('''\
    #!/usr/bin/env python3
    import json, sys, os, shutil
    args = sys.argv[1:]
    log = os.environ.get("FAKE_RC_LOG")
    if log:
        open(log, "a").write(json.dumps(args) + "\\n")
    products = {"schemaVersion": int(os.environ.get("FAKE_SCHEMA", "6")), "cliVersion": "2.6.10",
                "activationUrl": "https://www.rc-astro.com/account",
                "products": [
                    {"key": "bxt", "name": "RC-Astro BlurXTerminator", "displayName": "BlurXTerminator", "mlVersions": [5, 4, 2],
                     "license": {"status": "trial", "valid": True, "message": "Trial license expires in 27 days", "daysLeft": 27}},
                    {"key": "sxt", "name": "RC-Astro StarXTerminator", "displayName": "StarXTerminator", "mlVersions": [11],
                     "license": {"status": "required", "valid": False, "message": "A license is required to use this software"}},
                    {"key": "nxt", "name": "RC-Astro NoiseXTerminator", "displayName": "NoiseXTerminator", "mlVersions": [3.1, 2],
                     "license": {"status": os.environ.get("FAKE_NXT", "required"), "valid": os.environ.get("FAKE_NXT") == "activated",
                                 "message": "x"}}]}
    prod = next((a for a in args if a in ("bxt", "nxt", "sxt")), None)
    if "--json" in args and prod is None and "license" not in args and "download-models" not in args:
        print(json.dumps(products)); sys.exit(0)
    if prod and args[-1] == "--json" or (prod and "--json" in args and "-o" not in args):
        doc = json.load(open(os.environ["FAKE_SCHEMA_DIR"] + "/rc-astro-%s.json" % prod))
        print(json.dumps(doc)); sys.exit(0)
    if "license" in args:
        print(json.dumps({"event": "info", "topic": "licenseReport", "operation": "activate", "activationUrl": "u",
                          "licenseStatus": [{"product": "bxt", "status": "activated"}]}))
        sys.exit(0)
    code = int(os.environ.get("FAKE_EXIT", "0"))
    if code == 77:
        print(json.dumps({"event": "error", "message": "A license is required to use this software"})); sys.exit(77)
    if code:
        print("Unknown option: --zzz", file=sys.stderr); sys.exit(code)
    out = args[args.index("-o") + 1]
    src = args[-1]
    shutil.copy(src, out)
    print(json.dumps({"event": "info", "topic": "version", "cliVersion": "2.6.10", "schemaVersion": 6}))
    print(json.dumps({"event": "progress", "done": 50.0}))
    print(json.dumps({"event": "status", "phase": "saving", "output": out}))
    if "--stars" in args:
        side = out.replace(".fits", "-stars.fits")
        shutil.copy(src, side)
        print(json.dumps({"event": "status", "phase": "saving", "output": side}))
    print(json.dumps({"event": "status", "phase": "complete", "output": out}))
''')


@pytest.fixture()
def fake_rc(tmp_path, monkeypatch):
    exe = tmp_path / "rc-astro"
    exe.write_text(FAKE_RC)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("RCASTRO_BIN", str(exe))
    monkeypatch.setenv("FAKE_RC_LOG", str(tmp_path / "calls.log"))
    monkeypatch.setenv("FAKE_SCHEMA_DIR", os.path.join(os.path.dirname(tools.__file__), "data"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    rc = tools.RCAstro()
    rc.calls = lambda: ([json.loads(ln) for ln in (tmp_path / "calls.log").read_text().splitlines()]
                        if (tmp_path / "calls.log").exists() else [])
    return rc


def _fits(path, v=0.2, size=(32, 40)):
    fitsio.write_rgb(str(path), np.full((3,) + size, v, np.float32), {"OBJECT": "t"})
    return str(path)


def test_info_and_licensing(fake_rc):
    info = fake_rc.info(refresh=True)
    assert info["ok"] and info["schemaVersion"] == 6
    assert [(p["key"], p["valid"]) for p in info["products"]] == [("bxt", True), ("sxt", False), ("nxt", False)]
    assert fake_rc.licensed("bxt") and not fake_rc.licensed("sxt")
    st = fake_rc.status()
    assert st["attention"]["reason"] == "unregistered" and st["activationUrl"].startswith("https://")


def test_run_builds_the_command_line(fake_rc, tmp_path):
    src, dst = _fits(tmp_path / "in.fits"), str(tmp_path / "out.fits")
    meta = fake_rc.run("bxt", src, dst, device="cuda", correct_only=True)
    assert os.path.exists(dst) and meta["flags"] == ["--correct-only"]
    argv = fake_rc.calls()[-1]
    # global flags first, then the product; --host because schemaVersion >= 6
    assert argv[:argv.index("bxt")] == ["--host", f"AstroPhotoStudio-{tools.APP_VERSION}", "--device", "gpu", "--json", "--no-banner"]
    assert argv[argv.index("bxt"):] == ["bxt", "-o", dst, "--overwrite", "--depth", "32F", "--correct-only", src]


def test_host_flag_only_for_schema_6(fake_rc, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SCHEMA", "5")
    fake_rc.info(refresh=True)
    src, dst = _fits(tmp_path / "in.fits"), str(tmp_path / "out.fits")
    fake_rc.run("bxt", src, dst, ss=0.3)
    assert "--host" not in fake_rc.calls()[-1]


def test_unlicensed_product_is_unavailable_without_running(fake_rc, tmp_path):
    src = _fits(tmp_path / "in.fits")
    n = len(fake_rc.calls())
    with pytest.raises(ToolUnavailable) as e:
        fake_rc.run("sxt", src, str(tmp_path / "o.fits"), stars=True)
    assert e.value.code == 77 and "license" in e.value.reason.lower()
    assert all("-o" not in c for c in fake_rc.calls()[n:])            # nothing was processed


def test_exit_77_maps_to_unavailable(fake_rc, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_EXIT", "77")
    with pytest.raises(ToolUnavailable) as e:
        fake_rc.run("bxt", _fits(tmp_path / "in.fits"), str(tmp_path / "o.fits"), ss=0.3)
    assert e.value.code == 77


def test_usage_errors_map_to_toolerror_with_message(fake_rc, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_EXIT", "105")
    with pytest.raises(ToolError) as e:
        fake_rc.run("bxt", _fits(tmp_path / "in.fits"), str(tmp_path / "o.fits"), ss=0.3)
    assert e.value.code == 105 and "Unknown option" in str(e.value)


def test_sxt_sidecar_from_saving_events(fake_rc, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_NXT", "activated")
    # make sxt valid: patch the cached info
    info = fake_rc.info(refresh=True)
    for p in info["products"]:
        p["valid"] = True
    src, dst = _fits(tmp_path / "in.fits"), str(tmp_path / "starless.fits")
    meta = fake_rc.run("sxt", src, dst, stars=True)
    assert meta["stars_path"] == str(tmp_path / "starless-stars.fits") and os.path.exists(meta["stars_path"])


def test_progress_events_reach_the_callback(fake_rc, tmp_path):
    seen = []
    fake_rc.run("bxt", _fits(tmp_path / "in.fits"), str(tmp_path / "o.fits"), progress=lambda i, n, m: seen.append((i, n, m)), ss=0.3)
    assert any(i == 50 and n == 100 for i, n, _ in seen)


def test_license_commands(fake_rc):
    rep = fake_rc.activate("all")
    assert rep["ok"] and rep["licenseStatus"][0]["product"] == "bxt"
    assert "--activate" in fake_rc.calls()[-1] and "all" in fake_rc.calls()[-1]
    with pytest.raises(ValueError):
        fake_rc.activate("zzz")
    fake_rc.license_code("abcd-efgh")
    assert "ABCD-EFGH" in fake_rc.calls()[-1]


def test_missing_binary(monkeypatch, tmp_path):
    monkeypatch.setenv("RCASTRO_BIN", str(tmp_path / "nope"))
    rc = tools.RCAstro()
    assert not rc.available()
    st = rc.status()
    assert st["ok"] is False and st["products"] == []
    with pytest.raises(ToolUnavailable):
        rc.run("bxt", "a", "b")


# ----------------------------------------------------------------------------- GraXpert / StarNet2

def test_graxpert_models_are_detected_in_xdg_data_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    gx = tools.GraXpert()
    assert gx.models() == {"bge": False, "denoise": False, "deconv_obj": False, "deconv_stellar": False}
    for d, v in (("bge-ai-models", "1.0.1"), ("bge-ai-models", "1.0.0"), ("denoise-ai-models", "3.0.2")):
        (tmp_path / "GraXpert" / d / v).mkdir(parents=True)
        (tmp_path / "GraXpert" / d / v / "model.onnx").write_bytes(b"x")
    assert gx.models()["bge"] and gx.models()["denoise"] and not gx.models()["deconv_obj"]
    assert gx.local_versions("bge") == ["1.0.0", "1.0.1"]


def test_graxpert_missing_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAXPERT_BIN", str(tmp_path / "none"))
    monkeypatch.setenv("GRAXPERT_RUNNER", str(tmp_path / "none.py"))
    with pytest.raises(ToolUnavailable):
        tools.GraXpert().background_extraction(str(tmp_path / "a.fits"), str(tmp_path / "b.fits"))


def test_unit_range_roundtrip():
    a = np.array([[-0.2, 0.5], [1.5, 0.0]], np.float32)
    u, rng = tools._unit_range(a)
    assert u.min() == 0 and u.max() == 1
    assert np.allclose(tools._undo_range(u, rng), a, atol=1e-6)
    b = np.array([[0.1, 0.9]], np.float32)
    assert tools._unit_range(b)[1] is None


@pytest.fixture()
def fake_starnet(tmp_path, monkeypatch):
    d = tmp_path / "starnet2"
    d.mkdir()
    exe = d / "starnet2"
    exe.write_text(textwrap.dedent('''\
        #!/usr/bin/env python3
        import sys, shutil, json, os
        a = sys.argv[1:]
        if "--version" in a:
            print("starnet2  version: 2.6.2"); sys.exit(0)
        open(os.environ["FAKE_SN_LOG"], "w").write(json.dumps(a))
        shutil.copy(a[a.index("-i") + 1], a[a.index("-o") + 1])
        print('{"progress": 0.5}')
        '''))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("STARNET_DIR", str(d))
    monkeypatch.setenv("FAKE_SN_LOG", str(tmp_path / "sn.json"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ASTROPHOTO_TOOLS_HOME", str(tmp_path / "tools"))
    return tools.StarNet()


def test_starnet_command_line(fake_starnet, tmp_path):
    src, dst = _fits(tmp_path / "in.fits", size=(600, 700)), str(tmp_path / "out.fits")
    meta = fake_starnet.run(src, dst, linear=True, stride=255)
    argv = json.loads((tmp_path / "sn.json").read_text())
    assert argv[:2] == ["-i", src] and "--linear" in argv and argv[argv.index("-s") + 1] == "254"
    assert meta["linear"] and fake_starnet.version() == "2.6.2"
    fake_starnet.run(src, dst, linear=False)
    assert "--linear" not in json.loads((tmp_path / "sn.json").read_text())


def test_starnet_needs_512_px(fake_starnet, tmp_path):
    with pytest.raises(ToolUnavailable):
        fake_starnet.run(_fits(tmp_path / "s.fits", size=(300, 900)), str(tmp_path / "o.fits"))


def test_starnet_nonlinear_input_must_be_unit_range(fake_starnet, tmp_path):
    big = np.full((3, 600, 600), 2.0, np.float32)
    fitsio.write_rgb(str(tmp_path / "b.fits"), big, {})
    with pytest.raises(ToolError):
        fake_starnet.run(str(tmp_path / "b.fits"), str(tmp_path / "o.fits"), linear=False)


def test_starnet_download_needs_licence_acceptance(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTROPHOTO_TOOLS_HOME", str(tmp_path / "tools"))
    monkeypatch.setenv("STARNET_DIR", str(tmp_path / "absent"))
    with pytest.raises(ToolUnavailable):
        tools.StarNet().download(accept_licence=False)
    st = tools.StarNet().status()
    assert st["ok"] is False and st["licence_accepted"] is False


def test_tools_status_shape(monkeypatch, tmp_path):
    for k in ("SIRIL_BIN", "GRAXPERT_BIN", "RCASTRO_BIN"):
        monkeypatch.setenv(k, str(tmp_path / "missing"))
    monkeypatch.setenv("STARNET_DIR", str(tmp_path / "missing"))
    monkeypatch.setenv("ASTROPHOTO_TOOLS_HOME", str(tmp_path / "t"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "x"))
    tools.rcastro._info_cache = None                      # the singleton caches info() for a few seconds
    st = tools.tools_status()
    assert set(st) == {"siril", "graxpert", "starnet", "rcastro", "cosmic_clarity", "gaia"}
    assert set(st["cosmic_clarity"]) >= {"ok", "version", "models", "torch", "models_dir"}
    assert set(st["siril"]) >= {"ok", "version", "path"}
    assert set(st["graxpert"]) == {"ok", "version", "models", "gpu"}
    assert set(st["graxpert"]["models"]) == {"bge", "denoise", "deconv_obj", "deconv_stellar"}
    assert set(st["starnet"]) == {"ok", "version", "path", "licence_accepted"}
    assert set(st["rcastro"]) >= {"ok", "version", "schema", "account", "activationUrl", "products", "attention"}
    assert not any(st[k]["ok"] for k in st)


def test_run_streaming_kills_the_child_when_the_checkpoint_raises(tmp_path):
    class Stop(Exception):
        pass

    n = {"c": 0}

    def cp():
        n["c"] += 1
        if n["c"] > 3:
            raise Stop()

    with pytest.raises(Stop):
        tools.run_streaming(["sleep", "30"], checkpoint=cp, env=dict(os.environ))
