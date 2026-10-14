"""siril.py: .seq parser on real Siril 1.4.4 files and SirilRunner against a fake ``siril-cli``."""
from __future__ import annotations

import os
import stat
import time

import pytest

from sirilpipe import siril
from sirilpipe.util import Cancelled

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def test_parse_registered_seq_from_real_siril():
    seq = siril.parse_seq(os.path.join(FIX, "siril_registered.seq"))
    assert seq.name == "pp_light" and seq.number == 12 and seq.reference == 0 and len(seq.frames) == 12
    f0, f6 = seq.frames[0], seq.frames[6]
    assert f0.fwhm == pytest.approx(3.56601) and f0.nbstars == 102 and f0.roundness == pytest.approx(0.835888)
    assert f0.H == [1, 0, 0, 0, 1, 0, 0, 0, 1]
    assert f6.fwhm == pytest.approx(5.86222) and f6.wfwhm == pytest.approx(10.8049) and f6.nbstars == 59
    assert f6.background == pytest.approx(0.000451986)
    assert f6.H[2] == pytest.approx(27.3824) and f6.H[5] == pytest.approx(-12.3103)
    # the fixture was saved after `unselect pp_light 3 4`
    assert [f.included for f in seq.frames] == [True, True, False, False] + [True] * 8 and seq.registered_layer == 0
    assert seq.selected == 10


def test_parse_selection_flags_in_applied_seq():
    seq = siril.parse_seq(os.path.join(FIX, "siril_applied.seq"))
    assert seq.name == "r_pp_light" and len(seq.frames) == 10 and seq.layers == 3
    assert [f.filenum for f in seq.frames][:4] == [1, 2, 5, 6]          # frames 3 and 4 were unselected


def test_seq_stats_and_ranges():
    st = siril.seq_stats(os.path.join(FIX, "siril_registered.seq"))
    assert len(st) == 12 and set(st[0]) >= {"fwhm", "wfwhm", "roundness", "bkg", "nbstars", "included"}
    assert siril.ranges([5, 1, 2, 3, 9, 10]) == [(1, 3), (5, 5), (9, 10)]
    assert siril.ranges([]) == []


def test_parse_seq_tolerates_garbage(tmp_path):
    p = tmp_path / "x_.seq"
    p.write_text("# c\nS 'x_' 1 2 2 5 0 6 0 0 0\nL 1\nI 1 1\nI 2 0\nR0 nan 1 2\nweird\n")
    seq = siril.parse_seq(str(p))
    assert [f.included for f in seq.frames] == [True, False]


def test_quote_and_header():
    assert siril.quote("-out=/a b/c") == '"-out=/a b/c"' and siril.quote("-out=/a/c") == "-out=/a/c"
    t = siril.with_header("cd x\nlink light", 0.5)
    assert t.splitlines()[:4] == ["requires 1.4.0", "setext fit", "set32bits", "setmem 0.5"]
    assert siril.with_header("requires 1.4.0\nset32bits\ncd x").count("set32bits") == 1


FAKE = """#!/bin/sh
# fake siril-cli: args: -d DIR -s SCRIPT ; behaviour from the script text
DIR="$2"; SCRIPT="$4"
cd "$DIR"
if grep -q "touch" "$SCRIPT"; then
  for i in 1 2 3 4; do touch "out_0000$i.fit"; echo "progress: Processing image $i (x), $((i*25)).00%"; sleep 0.2; done
fi
if grep -q "fail" "$SCRIPT"; then
  echo "log: Error in line 3 ('stack'): bad"; echo "log: Script execution failed."; exit 1
fi
if grep -q "sleep" "$SCRIPT"; then sleep 30; fi
echo "log: Script execution finished successfully."
exit 0
"""


@pytest.fixture()
def fake_siril(tmp_path):
    p = tmp_path / "siril-cli"
    p.write_text(FAKE)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def test_runner_success_counts_files_and_writes_log_and_script(tmp_path, fake_siril):
    seen = []
    r = siril.SirilRunner(str(tmp_path / "w name"), progress=lambda i, n, m: seen.append((i, n, m)), siril_bin=fake_siril)
    os.makedirs(r.workdir)
    res = r.run("cd x\ntouch", "calibrate", expect=["out_00004.fit"], count_glob="out_0*.fit", total=4, label="Calibrating")
    assert os.path.exists(res.log_path) and "/logs/calibrate-" in res.log_path
    assert open(res.script_path).read().startswith("requires 1.4.0\nsetext fit\nset32bits")
    assert any(m == "Calibrating (4/4)" for _, _, m in seen) and seen[0][0] == 0
    assert [i for i, n, m in seen if n == 4][-1] == 4


def test_runner_parses_percent_lines_without_file_counter(tmp_path, fake_siril):
    seen = []
    r = siril.SirilRunner(str(tmp_path), progress=lambda i, n, m: seen.append((i, n, m)), siril_bin=fake_siril)
    r.run("touch", "register", label="Registering")
    assert (50, 100, "Registering (50%)") in seen


def test_runner_failure_raises_with_last_log_lines(tmp_path, fake_siril):
    r = siril.SirilRunner(str(tmp_path), siril_bin=fake_siril)
    with pytest.raises(siril.SirilError) as e:
        r.run("fail", "stack")
    assert "Script execution failed" in str(e.value) and "Error in line 3" in e.value.tail and e.value.log_path


def test_runner_missing_expected_output(tmp_path, fake_siril):
    r = siril.SirilRunner(str(tmp_path), siril_bin=fake_siril)
    with pytest.raises(siril.SirilError, match="did not create"):
        r.run("noop", "x", expect=["nothere.fit"])


def test_runner_cancel_kills_child(tmp_path, fake_siril):
    flag = {"n": 0}

    def checkpoint():
        flag["n"] += 1
        if flag["n"] > 3:
            raise Cancelled()
    r = siril.SirilRunner(str(tmp_path), checkpoint=checkpoint, siril_bin=fake_siril)
    t0 = time.time()
    with pytest.raises(Cancelled):
        r.run("sleep", "slow")
    assert time.time() - t0 < 10                      # the 30 s sleep did not run to the end


def test_runner_stdin_variant(tmp_path):
    p = tmp_path / "siril-cli"
    p.write_text("#!/bin/sh\ncat > /dev/null\necho 'log: Script execution finished successfully.'\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    siril.SirilRunner(str(tmp_path), siril_bin=str(p)).run("cd x", "stdin_stage", via_stdin=True)


def test_runner_exit_code_without_marker_is_failure(tmp_path):
    p = tmp_path / "siril-cli"
    p.write_text("#!/bin/sh\necho hi\nexit 3\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    with pytest.raises(siril.SirilError, match="exit code 3"):
        siril.SirilRunner(str(tmp_path), siril_bin=str(p)).run("cd x", "s")
