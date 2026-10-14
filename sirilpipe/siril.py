"""Run ``siril-cli`` scripts and read Siril's sequence statistics.

One ``siril-cli`` process per stage, so progress, cancel and failure are per stage.  The script
is written to ``<workdir>/scripts/<stage>.ssf`` (the user can re-run it by hand) and the output is
streamed to ``<workdir>/logs/<stage>-<timestamp>.log`` and, line by line, to a progress callback.

Progress: Siril's script mode prints per-frame lines; two sources are combined: a counter of
output files appearing in the working directory (``count_glob`` / ``total``: exact and independent
of Siril's wording) and the percentages / "n/m" counters in the log lines.
"""
from __future__ import annotations

import glob
import os
import queue
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field

from . import config
from .util import Cancelled  # noqa: F401  (re-exported: post.py and the session catch it from here)

HEADER = ("requires 1.4.0", "setext fit", "set32bits")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_FAIL = re.compile(r"Script execution failed|Error in line|Exiting batch processing|Segmentation fault|"
                   r"core dumped", re.IGNORECASE)
_OK = re.compile(r"Script execution finished successfully", re.IGNORECASE)
_PCT = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")
_COUNT = re.compile(r"(?:image|frame)s?\s+(\d+)\s*(?:/|of|out of)\s*(\d+)", re.IGNORECASE)


class SirilError(RuntimeError):
    """A Siril stage failed; the message ends with the last 40 log lines."""

    def __init__(self, message: str, log_path: str | None = None, tail: str = ""):
        super().__init__(message)
        self.log_path = log_path
        self.tail = tail


@dataclass
class SirilResult:
    log_path: str
    seconds: float
    script_path: str = ""
    returncode: int = 0
    lines: int = 0
    tail: list[str] = field(default_factory=list)


def quote(arg: str) -> str:
    """Quote one Siril command argument that contains spaces as a whole (``"-out=/a b/c"``)."""
    if any(c in arg for c in ' \t"'):
        return '"' + arg.replace('"', "'") + '"'
    return arg


def with_header(script_text: str, mem_ratio: float | None = None) -> str:
    """The script with the mandatory first lines: ``requires 1.4.0``, ``setext fit``, ``set32bits`` (and
    ``setmem`` when asked), each only if the script does not already carry it."""
    body = [ln for ln in script_text.strip().splitlines()]
    have = {ln.split()[0] for ln in body if ln.strip() and not ln.lstrip().startswith("#")}
    head = [h for h in HEADER if h.split()[0] not in have]
    if mem_ratio and "setmem" not in have:
        head.append(f"setmem {mem_ratio:g}")
    if body and body[0].startswith("requires"):      # `requires` must stay the first command
        return "\n".join([body[0]] + head + body[1:]) + "\n"
    return "\n".join(head + body) + "\n"


class SirilRunner:
    """``SirilRunner(workdir, progress=None, checkpoint=None).run(script_text, stage, expect=()) -> SirilResult``

    ``workdir`` is Siril's starting directory (``-d``), normally ``<session>/process``.
    ``progress(i, n, msg)`` and ``checkpoint()`` (raises ``Cancelled``) are the session's callbacks.
    """

    def __init__(self, workdir: str, progress=None, checkpoint=None, siril_bin: str | None = None,
                 timeout: float | None = None, mem_ratio: float | None = 0.6):
        self.workdir = os.path.abspath(workdir)
        self.progress = progress
        self.checkpoint = checkpoint
        self.siril_bin = siril_bin or config.siril_bin()
        self.timeout = config.SIRIL_TIMEOUT if timeout is None else timeout
        self.mem_ratio = mem_ratio
        self._proc: subprocess.Popen | None = None

    # ------------------------------------------------------------------ public
    def run(self, script_text: str, stage: str, expect=(), *, count_glob: str | None = None,
            total: int | None = None, label: str | None = None, via_stdin: bool = False,
            env: dict | None = None, markers=()) -> SirilResult:
        """Run one script; raise ``SirilError`` on failure, ``Cancelled`` on cancel.

        expect       paths (relative to ``workdir``; globs allowed) that must exist afterwards.
        count_glob   glob (relative to ``workdir``) of files this stage creates one per frame;
                     with ``total`` it drives the progress counter exactly.
        label        text shown in progress messages (default: the stage name).
        via_stdin    feed the script on stdin (``-s -``) instead of a file.
        markers      (regex, new label) pairs: when a log line matches, later progress uses the new label
                     (used to tell the two registration passes apart).
        """
        os.makedirs(os.path.join(self.workdir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.workdir, "scripts"), exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.+-]+", "_", stage).strip("_") or "stage"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        script_path = os.path.join(self.workdir, "scripts", f"{safe}.ssf")
        log_path = os.path.join(self.workdir, "logs", f"{safe}-{stamp}.log")
        text = with_header(script_text, self.mem_ratio)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(text)
        label = label or stage
        cmd = [self.siril_bin, "-d", self.workdir, "-s", "-" if via_stdin else script_path]
        env_full = config.siril_env(env)
        pre = self._count(count_glob) if count_glob else 0
        t0 = time.time()
        self._report(0, total or 1, label)
        self._proc = proc = subprocess.Popen(
            cmd, cwd=self.workdir, env=env_full, stdin=subprocess.PIPE if via_stdin else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
            start_new_session=True)
        if via_stdin:
            try:
                proc.stdin.write(text)
                proc.stdin.close()
            except OSError:
                pass
        lines: "queue.Queue[str | None]" = queue.Queue()

        def pump():
            for ln in proc.stdout:
                lines.put(ln)
            lines.put(None)
        threading.Thread(target=pump, daemon=True).start()
        tail: list[str] = []
        n_lines, failed, ok, last_i = 0, False, False, -1
        try:
            with open(log_path, "w", encoding="utf-8") as log:
                eof = False
                while not eof:
                    try:
                        ln = lines.get(timeout=0.5)
                    except queue.Empty:
                        ln = ""
                    if ln is None:
                        eof = True
                    elif ln:
                        ln = _ANSI.sub("", ln).rstrip("\r\n")
                        n_lines += 1
                        log.write(ln + "\n")
                        log.flush()
                        tail.append(ln)
                        del tail[:-40]
                        for rx, new in markers:
                            if label != new and re.search(rx, ln):
                                label = new
                                last_i = -1
                        failed = failed or bool(_FAIL.search(ln))
                        ok = ok or bool(_OK.search(ln))
                        if not count_glob:
                            last_i = self._parse_line(ln, last_i, label, total)
                    if count_glob and total:
                        i = max(0, min(total, self._count(count_glob) - pre))
                        if i != last_i:
                            last_i = i
                            self._report(i, total, f"{label} ({i}/{total})")
                    if self.checkpoint:
                        self.checkpoint()
                    if self.timeout and time.time() - t0 > self.timeout:
                        raise SirilError(f"{stage}: Siril timed out after {self.timeout:.0f} s", log_path, "\n".join(tail))
            proc.wait()
        except BaseException:
            self._kill(proc)
            raise
        finally:
            self._proc = None
        rc = proc.returncode
        if failed or rc != 0 or not ok:
            why = ("script execution failed" if failed else f"exit code {rc}" if rc != 0
                   else "the script did not report success")
            raise SirilError(f"Siril stage '{stage}' failed ({why}); see {log_path}\n--- last log lines ---\n"
                             + "\n".join(tail[-40:]), log_path, "\n".join(tail[-40:]))
        missing = [e for e in expect if not glob.glob(e if os.path.isabs(e) else os.path.join(self.workdir, e))]
        if missing:
            raise SirilError(f"Siril stage '{stage}' finished but did not create: {', '.join(missing)}; see {log_path}\n"
                             "--- last log lines ---\n" + "\n".join(tail[-40:]), log_path, "\n".join(tail[-40:]))
        self._report(total or 1, total or 1, f"{label} ({total}/{total})" if total else label)
        return SirilResult(log_path=log_path, seconds=time.time() - t0, script_path=script_path, returncode=rc,
                           lines=n_lines, tail=tail[-40:])

    def cancel(self) -> None:
        """Kill the running child (also done automatically when ``checkpoint`` raises)."""
        if self._proc is not None:
            self._kill(self._proc)

    # ------------------------------------------------------------------ internals
    def _count(self, pattern: str) -> int:
        return len(glob.glob(os.path.join(self.workdir, pattern)))

    def _report(self, i: int, n: int, msg: str) -> None:
        if self.progress:
            self.progress(i, n, msg)

    def _parse_line(self, ln: str, last_i: int, label: str, total: int | None) -> int:
        """Progress from a log line when no file counter is available."""
        m = _COUNT.search(ln)
        if m:
            i, n = int(m.group(1)), int(m.group(2))
            if i != last_i:
                self._report(i, n, f"{label} ({i}/{n})")
            return i
        m = _PCT.search(ln)
        if m and ("progress" in ln.lower() or ln.strip().endswith("%")):
            pct = min(100.0, float(m.group(1)))
            i = int(pct)
            if i != last_i:
                self._report(i, 100, f"{label} ({i}%)")
            return i
        return last_i

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        """Terminate Siril and the processes AppRun started (own session / process group)."""
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
        except (ProcessLookupError, PermissionError):
            pass


# ======================================================================= .seq parsing
@dataclass
class SeqFrame:
    """One image of a sequence: selection flag and registration statistics (layer 1 = green)."""
    index: int                       # 0-based position in the sequence
    filenum: int = 0
    included: bool = True
    fwhm: float | None = None        # px
    wfwhm: float | None = None       # px (weighted by star count)
    roundness: float | None = None   # 0..1, 1 = round
    quality: float | None = None
    background: float | None = None  # [0,1] image units
    nbstars: int | None = None
    H: list[float] | None = None     # 3x3 homography row-major (image -> reference)


@dataclass
class Sequence:
    name: str
    beg: int = 1
    number: int = 0
    selected: int = 0
    reference: int = -1
    layers: int = 1
    frames: list[SeqFrame] = field(default_factory=list)
    registered_layer: int | None = None


_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|inf|-nan|-inf"


def parse_seq(path: str) -> Sequence:
    """Parse a Siril ``.seq`` file: selection (``I`` lines) and registration data (``R<layer>`` lines).

    Format (Siril 1.4): ``S 'name' beg number selnum fixed ref version``; ``L n`` layers;
    ``I filenum included``; ``R<layer> fwhm wfwhm roundness quality background nbstars H h00..h22``
    (several layers => several R lines per image, in image order).  Only the layer with the most
    star data is kept (the registered layer; the green one for CFA drizzle).  Unknown lines are
    ignored, so a format drift degrades to missing statistics instead of an exception.
    """
    seq = Sequence(name=os.path.basename(path).rsplit(".", 1)[0].rstrip("_"))
    reg: dict[int, dict[int, list[float]]] = {}      # layer -> image index -> values
    cur_i = -1
    sel_idx = -1
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            ln = raw.strip()
            if not ln or ln.startswith("#"):
                continue
            tag = ln[0]
            if tag == "S":
                m = re.match(r"S\s+'([^']*)'\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+)\s+(-?\d+)", ln)
                if m:
                    seq.name = m.group(1).rstrip("_")
                    seq.beg, seq.number, seq.selected = int(m.group(2)), int(m.group(3)), int(m.group(4))
                    seq.reference = int(m.group(6))
            elif tag == "L":
                try:
                    seq.layers = max(1, int(ln.split()[1]))
                except (IndexError, ValueError):
                    pass
            elif tag == "I":
                parts = ln.split()
                try:
                    cur_i += 1
                    fr = SeqFrame(index=cur_i, filenum=int(parts[1]), included=bool(int(parts[2])) if len(parts) > 2 else True)
                    seq.frames.append(fr)
                except (IndexError, ValueError):
                    pass
            elif tag == "R" and len(ln) > 1 and ln[1].isdigit():
                m = re.match(r"R(\d+)\s+(.*)$", ln)
                if not m:
                    continue
                layer = int(m.group(1))
                body = m.group(2)
                h_pos = body.find("H")
                head = body[:h_pos] if h_pos >= 0 else body
                hvals = body[h_pos + 1:].split() if h_pos >= 0 else []
                vals = [_tofloat(x) for x in head.split()]
                vals.append([_tofloat(x) for x in hvals[:9]] if len(hvals) >= 9 else None)
                idx = len(reg.setdefault(layer, {}))
                reg[layer][idx] = vals
    if reg:
        # the layer with the most valid fwhm values is the registered one
        best = max(reg, key=lambda k: sum(1 for v in reg[k].values() if v and v[0] not in (None, 0.0)))
        seq.registered_layer = best
        for i, v in reg[best].items():
            if i >= len(seq.frames):
                break
            fr = seq.frames[i]
            fr.fwhm, fr.wfwhm, fr.roundness, fr.quality, fr.background = (v[0], v[1], v[2], v[3], v[4]) if len(v) >= 6 else (None,) * 5
            fr.nbstars = int(v[5]) if len(v) >= 7 and v[5] is not None else None
            fr.H = v[-1] if isinstance(v[-1], list) else None
    return seq


def _tofloat(x: str):
    try:
        v = float(x)
        return v if v == v and abs(v) != float("inf") else None
    except ValueError:
        return None


def seq_stats(seqfile: str) -> list[dict]:
    """Per-frame registration stats of a ``.seq`` as dicts (fwhm, wfwhm, roundness, bkg, nbstars,
    included), in sequence order."""
    seq = parse_seq(seqfile)
    return [{"index": f.index, "included": f.included, "fwhm": f.fwhm, "wfwhm": f.wfwhm, "roundness": f.roundness,
             "quality": f.quality, "bkg": f.background, "nbstars": f.nbstars} for f in seq.frames]


def ranges(indices: list[int]) -> list[tuple[int, int]]:
    """Contiguous runs of sorted indices -> (first, last) pairs, for ``select``/``unselect`` commands."""
    out: list[tuple[int, int]] = []
    for i in sorted(set(indices)):
        if out and i == out[-1][1] + 1:
            out[-1] = (out[-1][0], i)
        else:
            out.append((i, i))
    return out
