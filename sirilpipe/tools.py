"""Wrappers for the external tools: GraXpert, StarNet2 and the RC-Astro CLI (BXT / NXT / SXT).

Every wrapper talks to its tool as a separate process (that is what the tools' licences allow for
StarNet2 and RC-Astro) and raises :class:`ToolUnavailable` when the tool cannot be used right now
(missing, not licensed, no models, image too small, ...) so that ``post.py`` can fall back to the
next alternative.  :class:`ToolError` is a real failure (the tool ran and failed).

All paths come from the environment (``sirilpipe.config``: SIRIL_BIN, GRAXPERT_BIN, GRAXPERT_RUNNER,
STARNET_DIR, RCASTRO_BIN, TOOL_LD_LIBRARY_PATH, HOME); nothing refers to a host installation.  Facts
about the tools' command lines, output naming and exit codes are from ``research/tools.md`` (sections
cited in the comments).
"""
from __future__ import annotations

import bz2
import hashlib
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import config, fitsio

try:                                                  # the version string goes into RC-Astro's --host
    from . import __version__ as APP_VERSION
except Exception:                                     # pragma: no cover
    APP_VERSION = "2.0.0"

Progress = Callable[[int, int, str], None]


# --------------------------------------------------------------------------- exceptions

class ToolUnavailable(RuntimeError):
    """The tool cannot be used now (missing, unlicensed, no models, input unsupported).  The caller
    falls back to the next alternative; ``reason`` is shown to the user and recorded."""

    def __init__(self, tool: str, reason: str, code: int | None = None):
        super().__init__(f"{tool}: {reason}")
        self.tool, self.reason, self.code = tool, reason, code


class ToolError(RuntimeError):
    """The tool ran and failed (non-zero exit that is not "unavailable")."""

    def __init__(self, tool: str, message: str, code: int | None = None, log: str = ""):
        super().__init__(f"{tool}: {message}")
        self.tool, self.message, self.code, self.log = tool, message, code, log


# --------------------------------------------------------------------------- process helpers

def _cuda_lib_dirs() -> list[str]:
    """Every ``<site-packages>/nvidia/<pkg>/lib`` directory beside the ones TOOL_LD_LIBRARY_PATH names.

    onnxruntime-gpu 1.22's CUDA provider also needs libcurand (and nvrtc / nvjitlink) from the same pip wheels;
    with only cudart, cublas, cufft and cudnn on the path it silently falls back to the CPU (measured: GraXpert
    denoise of an 8 MP stack ~13 min instead of seconds).  Adding the sibling wheel directories fixes that."""
    out: list[str] = []
    for d in config.get("TOOL_LD_LIBRARY_PATH").split(os.pathsep):
        m = re.match(r"(.*/nvidia)/[^/]+/lib/?$", d)
        if m and os.path.isdir(m.group(1)):
            for pkg in sorted(os.listdir(m.group(1))):
                lib = os.path.join(m.group(1), pkg, "lib")
                if os.path.isdir(lib) and lib not in out:
                    out.append(lib)
    return out


def tool_env(extra: dict | None = None, kind: str = "graxpert") -> dict:
    """Environment of a tool subprocess: HOME set, and the CUDA libraries on LD_LIBRARY_PATH *per tool*:

    * ``rcastro``: everything TOOL_LD_LIBRARY_PATH names (the pip nvidia wheels and the rc-astro folder, whose
      CUDA provider and onnxruntime sit beside the binary);
    * ``graxpert``: the nvidia wheel folders only;
    * ``starnet``: nothing.  StarNet2 is CPU-only and carries its own onnxruntime 1.26 (RUNPATH ``$ORIGIN/lib``);
      LD_LIBRARY_PATH outranks RUNPATH, so the rc-astro folder on the path made it load rc-astro's older
      libonnxruntime ("version VERS_1.26.0 not found")."""
    env = config.tool_env(extra)
    env.setdefault("HOME", config.home_dir())
    base = [d for d in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if d]
    tool_dirs = [d for d in config.get("TOOL_LD_LIBRARY_PATH").split(os.pathsep) if d]
    if kind == "starnet":
        dirs = []
    elif kind == "rcastro":
        dirs = tool_dirs + _cuda_lib_dirs()
    else:
        dirs = [d for d in tool_dirs if "rc-astro" not in d] + _cuda_lib_dirs()
    seen: list[str] = []
    for d in dirs + base:
        if d not in seen:
            seen.append(d)
    if seen:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(seen)
    else:
        env.pop("LD_LIBRARY_PATH", None)
    os.makedirs(env["HOME"], exist_ok=True)
    return env


def _kill(proc: subprocess.Popen) -> None:
    """Stop a child and its own children (GraXpert forks workers)."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()


def run_streaming(argv: list[str], *, env: dict | None = None, kind: str = "graxpert", cwd: str | None = None,
                  on_line: Callable[[str], None] | None = None, checkpoint: Callable[[], None] | None = None,
                  timeout: float | None = None, keep: int = 400) -> tuple[int, list[str]]:
    """Run a process, hand its merged stdout/stderr lines to ``on_line`` as they arrive and call
    ``checkpoint()`` (which raises on cancel) about four times a second; the child is killed when
    the checkpoint raises or the timeout passes.  Returns (exit code, the last ``keep`` lines)."""
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env or tool_env(kind=kind),
                                cwd=cwd, text=True, errors="replace", bufsize=1, start_new_session=True)
    except OSError as e:
        raise ToolUnavailable(os.path.basename(argv[0]), f"cannot start: {e}") from e
    lines: "queue.Queue[str | None]" = queue.Queue()

    def reader():
        for ln in proc.stdout:               # type: ignore[union-attr]
            lines.put(ln.rstrip("\r\n"))
        lines.put(None)

    threading.Thread(target=reader, daemon=True).start()
    tail: list[str] = []
    t0 = time.time()
    limit = timeout if timeout is not None else (config.TOOL_TIMEOUT or None)
    try:
        while True:
            try:
                ln = lines.get(timeout=0.25)
            except queue.Empty:
                ln = ""
            if ln is None:
                break
            if ln:
                tail.append(ln)
                if len(tail) > keep:
                    del tail[: len(tail) - keep]
                if on_line:
                    on_line(ln)
            if checkpoint:
                checkpoint()
            if limit and time.time() - t0 > limit:
                raise ToolError(os.path.basename(argv[0]), f"timed out after {limit:.0f} s")
        return proc.wait(), tail
    except BaseException:
        _kill(proc)
        raise
    finally:
        if proc.poll() is None:
            _kill(proc)


def run_quiet(argv: list[str], timeout: float = 60, env: dict | None = None, cwd: str | None = None,
              kind: str = "graxpert") -> tuple[int, str]:
    """Run a short query; (exit code, stdout+stderr).  Never raises for a missing binary: (127, message)."""
    try:
        p = subprocess.run(argv, capture_output=True, text=True, errors="replace", timeout=timeout,
                           env=env or tool_env(kind=kind), cwd=cwd)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError:
        return 127, f"{argv[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, f"{argv[0]}: timed out"
    except OSError as e:
        return 126, str(e)


def _is_exe(path: str | None) -> bool:
    return bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)


def _data_home() -> str:
    return os.environ.get("XDG_DATA_HOME") or os.path.join(config.home_dir(), ".local", "share")


def _config_home() -> str:
    return os.environ.get("XDG_CONFIG_HOME") or os.path.join(config.home_dir(), ".config")


def _version_key(v: str) -> tuple:
    return tuple(int(x) if x.isdigit() else 0 for x in re.split(r"[.\-]", v))


def _unit_range(a: np.ndarray) -> tuple[np.ndarray, tuple[float, float] | None]:
    """GraXpert / RC-Astro rescale anything outside [0, 1] themselves (tools.md A.4, C.4); rescale it
    here instead, where the inverse is known.  Returns (array in [0,1], (lo, span) or None)."""
    lo, hi = float(np.min(a)), float(np.max(a))
    if lo >= 0.0 and hi <= 1.0:
        return a, None
    span = max(hi - lo, 1e-12)
    return ((a - lo) / span).astype(np.float32), (lo, span)


def _undo_range(a: np.ndarray, rng: tuple[float, float] | None) -> np.ndarray:
    return a if rng is None else (a * rng[1] + rng[0]).astype(np.float32)


# --------------------------------------------------------------------------- Siril

def siril_status() -> dict:
    """{ok, version, path} for the status panel."""
    path = config.siril_bin()
    if not (os.path.isfile(path) or shutil.which(path)):
        return {"ok": False, "version": None, "path": path, "reason": "siril-cli not found"}
    rc, out = run_quiet([path, "-v"], timeout=40, env=config.siril_env())
    m = re.search(r"(\d+\.\d+\.\d+\S*)", out)
    return {"ok": rc == 0 and bool(m), "version": m.group(1) if m else None, "path": path}


# --------------------------------------------------------------------------- GraXpert

# <models>/<dir>/<version>/model.onnx (tools.md A.5)
GRAXPERT_MODEL_DIRS = {"bge": "bge-ai-models", "denoise": "denoise-ai-models",
                       "deconv_obj": "deconvolution-object-ai-models",
                       "deconv_stellar": "deconvolution-stars-ai-models"}
GRAXPERT_COMMANDS = {"bge": "background-extraction", "denoise": "denoising",
                     "deconv_obj": "deconv-obj", "deconv_stellar": "deconv-stellar"}

# Pre-fetch of the newest model of every kind with the wheel's own client (tools.md A.5).
_GRAXPERT_DOWNLOAD_PY = r'''
import sys; sys.modules["tkinter"] = None
from graxpert.ai_model_handling import (download_version, list_remote_versions, bge_ai_models_dir,
    denoise_ai_models_dir, deconvolution_object_ai_models_dir, deconvolution_stars_ai_models_dir)
from graxpert.s3_secrets import (bge_bucket_name, denoise_bucket_name,
    deconvolution_object_bucket_name, deconvolution_stars_bucket_name)
import os
want = sys.argv[1:] or ["bge", "denoise", "deconv_obj", "deconv_stellar"]
kinds = {"bge": (bge_ai_models_dir, bge_bucket_name), "denoise": (denoise_ai_models_dir, denoise_bucket_name),
         "deconv_obj": (deconvolution_object_ai_models_dir, deconvolution_object_bucket_name),
         "deconv_stellar": (deconvolution_stars_ai_models_dir, deconvolution_stars_bucket_name)}
for k in want:
    d, b = kinds[k]
    vers = sorted(v["version"] for v in list_remote_versions(b))
    if not vers:
        print("MODEL", k, "none-remote", flush=True); continue
    v = vers[-1]
    if os.path.isfile(os.path.join(d, v, "model.onnx")):
        print("MODEL", k, v, "present", flush=True); continue
    print("MODEL", k, v, "downloading", flush=True)
    download_version(d, b, v)
    print("MODEL", k, v, "done", flush=True)
'''


@dataclass
class GraXpert:
    """GraXpert CLI (3.2.0a3 in its own venv, with the in-process GPU runner of tools.md A.6)."""

    # ---- locate
    def command(self) -> list[str]:
        py, runner = config.get("GRAXPERT_BIN"), config.get("GRAXPERT_RUNNER")
        if runner and os.path.isfile(runner) and _is_exe(py):
            return [py, runner]
        if _is_exe(py) and os.path.basename(py).startswith("graxpert"):
            return [py]
        found = shutil.which("graxpert") if not os.environ.get("GRAXPERT_BIN") else None
        if found:
            return [found]
        raise ToolUnavailable("graxpert", "GraXpert is not installed (GRAXPERT_BIN)")

    def available(self) -> bool:
        try:
            self.command()
            return True
        except ToolUnavailable:
            return False

    def version(self) -> str | None:
        try:
            py = self.command()[0]
        except ToolUnavailable:
            return None
        rc, out = run_quiet([py, "-c", "import importlib.metadata as m; print(m.version('graxpert'))"], 30)
        return out.strip().splitlines()[-1] if rc == 0 and out.strip() else None

    # ---- models
    def models_root(self) -> str:
        return os.path.join(_data_home(), "GraXpert")

    def local_versions(self, kind: str) -> list[str]:
        d = os.path.join(self.models_root(), GRAXPERT_MODEL_DIRS[kind])
        try:
            vs = [v for v in os.listdir(d) if os.path.isfile(os.path.join(d, v, "model.onnx"))]
        except OSError:
            return []
        return sorted(vs, key=_version_key)

    def models(self) -> dict[str, bool]:
        return {k: bool(self.local_versions(k)) for k in GRAXPERT_MODEL_DIRS}

    def download_models(self, kinds: list[str] | None = None, progress: Progress | None = None,
                        checkpoint: Callable[[], None] | None = None) -> dict:
        """Fetch the newest model of each kind (about 1.2 GB in total) into $XDG_DATA_HOME/GraXpert."""
        cmd = self.command()
        py = cmd[0]
        kinds = kinds or list(GRAXPERT_MODEL_DIRS)
        done: list[str] = []
        total = len(kinds)

        def on_line(ln: str):
            if ln.startswith("MODEL ") and progress:
                parts = ln.split()
                if parts[-1] in ("done", "present"):
                    done.append(parts[1])
                progress(len(done), total, f"GraXpert models: {' '.join(parts[1:])}")

        rc, tail = run_streaming([py, "-c", _GRAXPERT_DOWNLOAD_PY, *kinds], on_line=on_line, checkpoint=checkpoint,
                                 env=tool_env({"PYTHONWARNINGS": "ignore"}))
        if rc != 0:
            raise ToolError("graxpert", "model download failed: " + " | ".join(tail[-3:]), rc, "\n".join(tail))
        return {"models": self.models()}

    # ---- run
    @staticmethod
    def _gpu_arg(gpu: bool | str | None) -> bool:
        if isinstance(gpu, str):
            return gpu.lower() not in ("cpu", "false", "0", "no", "mps")
        return True if gpu is None else bool(gpu)

    def run(self, kind: str, src: str, dst: str, *, flags: list[str] | None = None, gpu: bool | str | None = True,
            progress: Progress | None = None, checkpoint: Callable[[], None] | None = None,
            stage: str = "GraXpert") -> dict:
        """Run one GraXpert command on FITS ``src`` and write the result to ``dst`` with ``src``'s header.

        The input is copied to a scratch folder (GraXpert writes beside its input, tools.md A.4) and
        scaled into [0,1] when needed; the output keeps the input's units.  When the GPU run fails the
        command is repeated on the CPU (the log's "Used inference providers" line tells which ran)."""
        cmd = self.command()
        if kind not in GRAXPERT_COMMANDS:
            raise ValueError(f"unknown GraXpert command {kind!r}")
        if not self.local_versions(kind):
            # the first run downloads the model itself; only an offline machine cannot
            pass
        arr, header = fitsio.read_rgb(src)
        arr, rng = _unit_range(arr)
        scratch = tempfile.mkdtemp(prefix="gx_", dir=os.path.dirname(os.path.abspath(dst)) or None)
        try:
            inp = os.path.join(scratch, "in.fits")
            fitsio.write_rgb(inp, arr, header)
            ver = self.local_versions(kind)
            use_gpu = self._gpu_arg(gpu)
            meta: dict = {"tool": "graxpert", "command": GRAXPERT_COMMANDS[kind], "model": ver[-1] if ver else None}
            for attempt in (use_gpu, False) if use_gpu else (False,):
                argv = cmd + [inp, "-cli", "-cmd", GRAXPERT_COMMANDS[kind], "-output", "out",
                              "-gpu", "true" if attempt else "false"]
                if ver:
                    argv += ["-ai_version", ver[-1]]           # reproducible, and no model lookup surprises
                argv += flags or []
                providers: list[str] = []

                def on_line(ln: str):
                    m = re.search(r"Used inference providers.*?(\[.*\])", ln)
                    if m:
                        providers.append(m.group(1))
                    pm = re.search(r"(\d{1,3})\s*%", ln)
                    if pm and progress:
                        progress(min(100, int(pm.group(1))), 100, f"{stage}: {pm.group(1)}%")

                rc, tail = run_streaming(argv, env=tool_env({"PYTHONWARNINGS": "ignore"}), on_line=on_line,
                                         checkpoint=checkpoint, cwd=scratch)
                out = next((p for p in (os.path.join(scratch, "out.fits"), os.path.join(scratch, "in_GraXpert.fits"))
                            if os.path.isfile(p)), None)
                meta["providers"] = providers[-1] if providers else None
                meta["gpu"] = bool(attempt and providers and "CUDA" in providers[-1])
                if rc == 0 and out:
                    break
                if attempt:
                    meta["gpu_fallback"] = " | ".join(tail[-2:])
                    continue
                raise ToolError("graxpert", f"{GRAXPERT_COMMANDS[kind]} failed (exit {rc}): " + " | ".join(tail[-4:]),
                                rc, "\n".join(tail))
            res, _ = fitsio.read_rgb(out)
            if res.shape != arr.shape:
                raise ToolError("graxpert", f"output shape {res.shape} differs from the input {arr.shape}")
            fitsio.write_rgb(dst, _undo_range(res, rng), header)      # re-attach the input header (WCS, keywords)
            return meta
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def background_extraction(self, src: str, dst: str, smoothing: float = 0.5, correction: str = "subtraction",
                              **kw) -> dict:
        corr = "Division" if str(correction).lower().startswith("div") else "Subtraction"
        meta = self.run("bge", src, dst, flags=["-correction", corr, "-smoothing", f"{float(np.clip(smoothing, 0, 1)):.3f}"],
                        stage="GraXpert background", **kw)
        meta.update(smoothing=float(smoothing), correction=corr.lower())
        return meta

    def denoise(self, src: str, dst: str, strength: float = 0.5, batch_size: int = 4, **kw) -> dict:
        meta = self.run("denoise", src, dst, flags=["-strength", f"{float(np.clip(strength, 0, 1)):.3f}",
                                                    "-batch_size", str(int(batch_size))], stage="GraXpert denoise", **kw)
        meta.update(strength=float(strength))
        return meta

    def deconvolve(self, src: str, dst: str, kind: str = "obj", strength: float = 0.5, psfsize: float = 3.0,
                   batch_size: int = 4, **kw) -> dict:
        key = "deconv_stellar" if kind.startswith("stel") else "deconv_obj"
        meta = self.run(key, src, dst, flags=["-strength", f"{float(np.clip(strength, 0, 1)):.3f}",
                                              "-psfsize", f"{float(psfsize):.2f}", "-batch_size", str(int(batch_size))],
                        stage="GraXpert deconvolution", **kw)
        meta.update(strength=float(strength), psfsize=float(psfsize))
        return meta

    def status(self) -> dict:
        ok = self.available()
        libs = [p for p in config.get("TOOL_LD_LIBRARY_PATH").split(os.pathsep) if p]
        gpu = bool(libs) and all(os.path.isdir(p) for p in libs) and os.path.exists("/dev/nvidiactl")
        return {"ok": ok, "version": self.version() if ok else None, "models": self.models(), "gpu": gpu}


# --------------------------------------------------------------------------- StarNet2

# Official download (tools.md B.2).  The licence forbids mirroring: the file always comes from the
# author's server and only after the user accepted the licence.
STARNET_URL = "https://download.starnetastro.com/starnet2_linux_2.6.2-0242_ORT_x64_cli.zip"
STARNET_SHA256 = "066a7683f0321ae8400d3bc1023a48747ac228e1d14e76b5d39c4e48cbce4920"
STARNET_MIN_SIDE = 512                   # tools.md B.4: the network needs at least 512 x 512 px


@dataclass
class StarNet:
    """StarNet2 CLI (``starnet2``), run as a separate process."""

    def install_dirs(self) -> list[str]:
        dirs = []
        d = config.get("STARNET_DIR")
        if d:
            dirs.append(d)
        dirs.append(os.path.join(config.get("TOOLS_HOME"), "starnet2"))
        return dirs

    def exe(self) -> str | None:
        for d in self.install_dirs():
            p = os.path.join(d, "starnet2")
            if _is_exe(p):
                return p
        return None

    def available(self) -> bool:
        return self.exe() is not None

    def version(self) -> str | None:
        exe = self.exe()
        if not exe:
            return None
        rc, out = run_quiet([exe, "--version"], 30, cwd=os.path.dirname(exe), kind="starnet")
        m = re.search(r"version:?\s*(\d+\.\d+\.\d+\S*)", out) or re.search(r"(\d+\.\d+\.\d+)", out)
        return m.group(1) if m else (out.strip().splitlines() or [None])[0]

    def licence_accepted(self) -> bool:
        """The user's acceptance: STARNET_ACCEPT_LICENSE=yes at build/run time, or the marker the
        download wrote, or a user-supplied install (they obtained it from the author themselves)."""
        if os.environ.get("STARNET_ACCEPT_LICENSE", "").lower() in ("yes", "1", "true"):
            return True
        return any(os.path.exists(os.path.join(d, ".licence_accepted")) for d in self.install_dirs()) or self.available()

    def download(self, accept_licence: bool, progress: Progress | None = None,
                 checkpoint: Callable[[], None] | None = None) -> dict:
        """Download StarNet2 from the official URL into $HOME/tools/starnet2 (needs ``accept_licence``)."""
        if not accept_licence:
            raise ToolUnavailable("starnet2", "the StarNet2 licence must be accepted before it is downloaded")
        dest = os.path.join(config.get("TOOLS_HOME"), "starnet2")
        tmp = tempfile.mkdtemp(prefix="starnet_dl_", dir=config.get("TOOLS_HOME") if os.path.isdir(config.get("TOOLS_HOME")) else None)
        try:
            zpath = os.path.join(tmp, "starnet2.zip")
            sha = hashlib.sha256()
            with urllib.request.urlopen(STARNET_URL, timeout=60) as r, open(zpath, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                got = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    sha.update(chunk)
                    got += len(chunk)
                    if progress:
                        progress(got, max(total, got), f"Downloading StarNet2 ({got / 1e6:.0f} of {total / 1e6:.0f} MB)")
                    if checkpoint:
                        checkpoint()
            if sha.hexdigest() != STARNET_SHA256:
                raise ToolError("starnet2", f"download checksum mismatch ({sha.hexdigest()[:12]}...)")
            ex = os.path.join(tmp, "x")
            with zipfile.ZipFile(zpath) as z:
                z.extractall(ex)
            root = next((os.path.join(ex, d) for d in os.listdir(ex) if os.path.isdir(os.path.join(ex, d))), ex)
            if os.path.isdir(dest):
                shutil.rmtree(dest)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.move(root, dest)
            for name in ("starnet2",):                           # zipfile drops the exec bit (tools.md B.2)
                p = os.path.join(dest, name)
                if os.path.exists(p):
                    os.chmod(p, 0o755)
            with open(os.path.join(dest, ".licence_accepted"), "w") as f:
                f.write(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
            return {"path": dest, "version": self.version()}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def run(self, src: str, dst: str, *, linear: bool = True, stride: int = 256, mask: str | None = None,
            upsample: bool = False, progress: Progress | None = None,
            checkpoint: Callable[[], None] | None = None) -> dict:
        """Remove the stars from FITS ``src`` -> ``dst``.

        ``--linear`` makes StarNet stretch linear data internally and reverse it (tools.md B.4);
        without it the network expects stretched data and floats must be within [0,1]."""
        exe = self.exe()
        if not exe:
            raise ToolUnavailable("starnet2", "StarNet2 is not installed (download it in the Tools panel)")
        arr, header = fitsio.read_rgb(src)
        h, w = arr.shape[1:]
        if min(h, w) < (256 if upsample else STARNET_MIN_SIDE):
            raise ToolUnavailable("starnet2", f"image {w}x{h} is smaller than the {STARNET_MIN_SIDE} px StarNet2 needs")
        if not np.isfinite(arr).all():
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
            fitsio.write_rgb(src, arr, header)
        if not linear and (arr.min() < 0 or arr.max() > 1):
            raise ToolError("starnet2", "non-linear input must lie within [0, 1]")
        stride = int(stride) - int(stride) % 2
        argv = [exe, "-i", os.path.abspath(src), "-o", os.path.abspath(dst), "-s", str(max(2, min(512, stride))),
                "--machine-progress"]
        if linear:
            argv.append("--linear")
        if upsample:
            argv.append("-u")
        if mask:
            argv += ["-m", os.path.abspath(mask)]

        def on_line(ln: str):
            if not progress:
                return
            pct = None
            try:
                ev = json.loads(ln)
                for k in ("progress", "percent", "done", "percentage"):
                    if isinstance(ev, dict) and isinstance(ev.get(k), (int, float)):
                        pct = float(ev[k]) * (100 if ev[k] <= 1.0 and k == "progress" else 1)
                        break
            except ValueError:
                m = re.search(r"(\d+(?:\.\d+)?)\s*%", ln)
                pct = float(m.group(1)) if m else None
            if pct is not None:
                progress(int(min(100, pct)), 100, f"StarNet2: {pct:.0f}%")

        t0 = time.time()
        rc, tail = run_streaming(argv, kind="starnet", cwd=os.path.dirname(exe), on_line=on_line, checkpoint=checkpoint)
        if rc != 0 or not os.path.isfile(dst):
            raise ToolError("starnet2", f"failed (exit {rc}): " + " | ".join(tail[-4:]), rc, "\n".join(tail))
        res, _ = fitsio.read_rgb(dst)
        if res.shape != arr.shape:
            raise ToolError("starnet2", f"output shape {res.shape} differs from the input {arr.shape}")
        fitsio.write_rgb(dst, res, header)                      # keep our header (StarNet replays it, add nothing)
        gpu = any("CUDA execution provider)" in ln for ln in tail)
        return {"tool": "starnet2", "version": self.version(), "linear": linear, "stride": stride,
                "gpu": gpu, "seconds": round(time.time() - t0, 1)}

    def status(self) -> dict:
        exe = self.exe()
        return {"ok": exe is not None, "version": self.version() if exe else None,
                "path": os.path.dirname(exe) if exe else self.install_dirs()[-1],
                "licence_accepted": self.licence_accepted()}


# --------------------------------------------------------------------------- RC-Astro

PRODUCT_MODEL_PREFIX = {"bxt": "BlurXTerminator", "nxt": "NoiseXTerminator", "sxt": "StarXTerminator"}
PRODUCTS = ("bxt", "nxt", "sxt")
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_schema(product: str) -> dict:
    """Parameter document of a product (``rc-astro <product> --json``), the packaged copy of 2.6.10."""
    with open(os.path.join(_DATA_DIR, f"rc-astro-{product}.json"), encoding="utf-8") as f:
        return json.load(f)


def _eval_condition(expr: str, ctx: dict) -> bool:
    """Evaluate the schema's tiny boolean language (``a && !b || c``) over ``ctx``."""
    if not expr or not re.fullmatch(r"[\w\s!&|()]+", expr):
        return False
    py = expr.replace("&&", " and ").replace("||", " or ")
    py = re.sub(r"!(?!=)", " not ", py)
    names = {n: bool(ctx.get(n)) for n in re.findall(r"[A-Za-z_]\w*", py) if n not in ("and", "or", "not")}
    return bool(eval(py, {"__builtins__": {}}, names))        # noqa: S307 - validated charset, bool names only


def build_flags(schema: dict, params: dict) -> list[str]:
    """argv flags for ``params`` from the product's parameter document.

    Only names the document declares are accepted (unknown names raise ``ValueError``); values are
    clamped to the declared range; a value equal to the default is omitted (RC-Astro: "at-default
    parameters can be omitted"); bool-with-default-true is switched off with ``--no-<flag>``; GUI-only
    switches are never emitted; a parameter pinned by a mode (``correct_only``) or disabled by a
    ``disabledIf`` rule is left out (the CLI rejects the conflict, tools.md C.1)."""
    specs = {p["name"]: p for p in schema.get("parameters", [])}
    modes = {m["name"]: m for m in schema.get("modes", [])}
    mode_alias = {"correct_only": "correctOnlyMode"}
    given = dict(params)
    flags: list[str] = []
    active_modes: list[dict] = []
    for k in list(given):
        name = mode_alias.get(k, k)
        if name in modes:
            if given.pop(k):
                active_modes.append(modes[name])
    pinned = {n for m in active_modes for n in m.get("pins", {})}
    for m in active_modes:
        flags.append(m["flag"])
    for k in given:
        if k not in specs:
            raise ValueError(f"rc-astro {schema.get('key')}: unknown parameter {k!r}")
    ctx = {n: given.get(n, s.get("default")) for n, s in specs.items()}
    ctx.update({n: v for m in active_modes for n, v in m.get("pins", {}).items()})
    for name, spec in specs.items():
        if name not in given or spec.get("guiOnly") or name in pinned or "flag" not in spec:
            continue
        val = given[name]
        if val is None:
            continue
        if spec.get("disabledIf") and _eval_condition(spec["disabledIf"], ctx):
            continue
        typ, default = spec.get("type"), spec.get("default")
        if typ == "bool":
            val = bool(val)
            if val != bool(default):
                flags.append(spec["flag"] if val else "--no-" + spec["flag"].lstrip("-"))
            continue
        val = float(val) if typ == "float" else int(round(float(val)))
        if "min" in spec:
            val = max(spec["min"], val)
        if "max" in spec:
            val = min(spec["max"], val)
        if typ == "float":
            prec = int(spec.get("precision", 2))
            val = round(val, prec)
            if abs(val - float(default)) < 10 ** -(prec + 1):
                continue
            flags += [spec["flag"], f"{val:.{prec}f}"]
        else:
            if val != default:
                flags += [spec["flag"], str(val)]
    return flags


def parse_event(line: str) -> dict | None:
    """One NDJSON line of ``--json`` (tools.md C.3) -> dict; plain text (usage errors) -> None."""
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        ev = json.loads(line)
    except ValueError:
        return None
    return ev if isinstance(ev, dict) and "event" in ev else None


@dataclass
class RcResult:
    """What the event stream of one run said."""
    outputs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    device: dict | None = None
    cli_version: str | None = None
    schema_version: int | None = None
    report: dict | None = None
    models: list[dict] = field(default_factory=list)
    last_progress: float = 0.0

    def feed(self, ev: dict) -> None:
        kind = ev.get("event")
        if kind == "info":
            topic = ev.get("topic")
            if topic == "version":
                self.cli_version, self.schema_version = ev.get("cliVersion"), ev.get("schemaVersion")
            elif topic == "device":
                self.device = ev
            elif topic == "modelDownload":
                self.models.append(ev)
            elif topic == "licenseReport" or "licenseStatus" in ev or isinstance(ev.get("report"), dict):
                self.report = ev.get("report") if isinstance(ev.get("report"), dict) else ev
        elif kind == "status":
            if ev.get("output"):
                self.outputs.append(ev["output"])
        elif kind == "progress":
            self.last_progress = float(ev.get("done") or 0.0)
        elif kind == "warning":
            self.warnings.append(str(ev.get("message")))
        elif kind == "error":
            self.errors.append(str(ev.get("message")))


def normalize_report(rep: dict | None) -> dict | None:
    """The licence report of ``rc-astro license ... --json`` in the shape the web UI shows."""
    if not rep:
        return None
    return {"operation": rep.get("operation"), "computer": rep.get("computer"),
            "account": rep.get("account"), "activationUrl": rep.get("activationUrl"),
            "licenseStatus": rep.get("licenseStatus") or [], "attention": rep.get("attention")}


def normalize_code(code: str) -> str:
    """Activation codes are 8 letters, case/hyphen/space insensitive (tools.md C.5)."""
    c = re.sub(r"[\s\-]+", "", code or "").upper()
    if not re.fullmatch(r"[A-Z0-9]{8}", c):
        raise ValueError("an activation code is 8 letters (XXXX-XXXX)")
    return c[:4] + "-" + c[4:]


def device_arg(device: str | None) -> str:
    """stack_params.device (auto|cuda|cuda:N|mps|cpu) -> rc-astro ``--device`` (auto|cpu|gpu|gpuN)."""
    d = (device or "auto").lower()
    if d.startswith("cuda") or d.startswith("gpu"):
        n = re.search(r"(\d+)$", d)
        return f"gpu{n.group(1)}" if n else "gpu"
    return "cpu" if d in ("cpu", "mps") else "auto"


@dataclass
class RCAstro:
    """The RC-Astro CLI (BlurXTerminator / NoiseXTerminator / StarXTerminator)."""
    _info_cache: tuple[float, dict] | None = field(default=None, repr=False)

    # ---- locate / query
    def exe(self) -> str | None:
        p = config.get("RCASTRO_BIN")
        if os.path.sep in p:
            return p if _is_exe(p) else None
        return shutil.which(p)

    def available(self) -> bool:
        return self.exe() is not None

    def _argv(self, *args: str, host: bool = False, device: str | None = None) -> list[str]:
        exe = self.exe()
        if not exe:
            raise ToolUnavailable("rc-astro", "the RC-Astro CLI is not installed (installer not provided at build time)")
        argv = [exe]
        if host and (self.info().get("schemaVersion") or 0) >= 6:      # older CLIs exit non-zero on --host (C.8)
            argv += ["--host", f"AstroPhotoStudio-{APP_VERSION}"]
        if device:
            argv += ["--device", device]
        return argv + list(args)

    def info(self, refresh: bool = False, ttl: float = 15.0) -> dict:
        """Parsed ``rc-astro --json --no-banner`` (local, no network, no licence consumed)."""
        if not refresh and self._info_cache and time.time() - self._info_cache[0] < ttl:
            return self._info_cache[1]
        exe = self.exe()
        if not exe:
            doc = {"ok": False, "reason": "RC-Astro CLI not installed (installer not provided at build time)",
                   "schemaVersion": None, "products": [], "activationUrl": "https://www.rc-astro.com/account"}
        else:
            rc, out = run_quiet([exe, "--json", "--no-banner"], timeout=40, kind="rcastro")
            try:
                raw = json.loads(out[out.index("{"):])
                doc = self._normalize_info(raw)
            except (ValueError, KeyError):
                doc = {"ok": False, "reason": f"unreadable output (exit {rc}): {out.strip()[:200]}", "schemaVersion": None,
                       "products": [], "activationUrl": "https://www.rc-astro.com/account"}
        self._info_cache = (time.time(), doc)
        return doc

    def _normalize_info(self, raw: dict) -> dict:
        products = []
        for p in raw.get("products", []):
            lic = p.get("license") or {}
            products.append({"key": p.get("key"), "name": p.get("displayName") or p.get("name"),
                             "status": lic.get("status", "required"), "valid": bool(lic.get("valid")),
                             "message": lic.get("message", ""), "daysLeft": lic.get("daysLeft"),
                             "mlVersions": p.get("mlVersions", []), "modelPresent": self.model_present(p.get("key"))})
        return {"ok": True, "cliVersion": raw.get("cliVersion"), "schemaVersion": raw.get("schemaVersion"),
                "account": raw.get("account"), "activationUrl": raw.get("activationUrl", "https://www.rc-astro.com/account"),
                "devices": raw.get("devices", []), "products": products}

    def config_dirs(self) -> list[str]:
        dirs = [os.path.join(_config_home(), "RC-Astro")]
        if os.environ.get("RCASTRO_DATA_DIR"):
            dirs.insert(0, os.environ["RCASTRO_DATA_DIR"])
        return dirs

    def model_present(self, key: str | None) -> bool:
        prefix = PRODUCT_MODEL_PREFIX.get(key or "")
        if not prefix:
            return False
        for d in self.config_dirs():
            try:
                if any(f.startswith(prefix + ".") and f.endswith(".onnx") for f in os.listdir(d)):
                    return True
            except OSError:
                pass
        return False

    def product(self, key: str) -> dict | None:
        return next((p for p in self.info().get("products", []) if p["key"] == key), None)

    def licensed(self, key: str) -> bool:
        p = self.product(key)
        return bool(p and p["valid"])

    def require(self, key: str) -> None:
        """Raise ToolUnavailable unless ``key`` is installed and licensed (a local check)."""
        if not self.available():
            raise ToolUnavailable("rc-astro", "the RC-Astro CLI is not installed")
        p = self.product(key)
        if not p or not p["valid"]:
            raise ToolUnavailable(f"rc-astro {key}", (p or {}).get("message") or "not licensed", 77)

    # ---- licence management (never run while a job is processing: the queue lock serialises)
    def _license_cmd(self, args: list[str], timeout: float = 120) -> dict:
        argv = self._argv("license", *args, "--json", "--no-banner", host=True)
        res = RcResult()
        rc, tail = run_streaming(argv, timeout=timeout, kind="rcastro", on_line=lambda ln: (parse_event(ln) and res.feed(parse_event(ln))))
        self._info_cache = None
        report = normalize_report(res.report) or {}
        report.update(ok=(rc == 0), exit_code=rc, error="; ".join(res.errors) or (None if rc == 0 else " | ".join(tail[-2:])))
        return report

    def license_code(self, code: str) -> dict:
        return self._license_cmd([normalize_code(code)])

    def license_status(self) -> dict:
        return self._license_cmd([])

    def activate(self, product: str = "all") -> dict:
        if product != "all" and product not in PRODUCTS:
            raise ValueError("product is bxt, nxt, sxt or all")
        return self._license_cmd(["--activate", product])

    def deactivate(self, product: str = "all") -> dict:
        if product != "all" and product not in PRODUCTS:
            raise ValueError("product is bxt, nxt, sxt or all")
        return self._license_cmd(["--deactivate", product])

    def download_models(self, progress: Progress | None = None, checkpoint: Callable[[], None] | None = None,
                        force: bool = False) -> dict:
        res = RcResult()

        def on_line(ln: str):
            ev = parse_event(ln)
            if not ev:
                return
            res.feed(ev)
            if ev.get("topic") == "modelDownload" and progress:
                tot = ev.get("total") or 0
                got = ev.get("received") or 0
                progress(int(got), int(tot) or int(got) or 1, f"RC-Astro model {ev.get('file')}: {ev.get('status')}")

        args = ["download-models", "--json", "--no-banner"] + (["--force"] if force else [])
        rc, tail = run_streaming(self._argv(*args, host=True), kind="rcastro", on_line=on_line, checkpoint=checkpoint)
        self._info_cache = None
        if rc != 0:
            raise ToolError("rc-astro", "; ".join(res.errors) or " | ".join(tail[-3:]), rc)
        return {"files": sorted({m.get("file") for m in res.models if m.get("file")})}

    # ---- processing
    def run(self, product: str, src: str, dst: str, *, device: str | None = "auto",
            progress: Progress | None = None, checkpoint: Callable[[], None] | None = None,
            ml_version: float | int | None = None, **params) -> dict:
        """Run ``rc-astro <product>`` on FITS ``src`` -> ``dst`` (float32, values in [0, 1]).

        ``params`` are the schema's parameter names (``ss=0.3``, ``correct_only=True``, ``dn=0.25``,
        ``stars=True`` ...).  Unlicensed products raise :class:`ToolUnavailable` before anything runs
        (a local check) and again when the CLI reports exit 77; usage errors (100-114) raise
        :class:`ToolError` with the CLI's message.  For ``sxt`` with ``stars=True`` the star image is
        found from the ``saving`` events (its name changes between CLI versions) and returned as
        ``meta["stars_path"]``."""
        if product not in PRODUCTS:
            raise ValueError(f"unknown RC-Astro product {product!r}")
        self.require(product)
        flags = build_flags(self._schema(product), params)
        dst = os.path.abspath(dst)
        out_dir = os.path.dirname(dst)
        os.makedirs(out_dir, exist_ok=True)
        before = set(os.listdir(out_dir))
        # global flags come before the product name (tools.md C.1); --host only when schemaVersion >= 6 (C.8)
        argv = self._argv("--json", "--no-banner", host=True, device=device_arg(device))
        argv += [product, "-o", dst, "--overwrite", "--depth", "32F"]
        if ml_version:
            argv += ["--ml-version", str(ml_version)]
        argv += flags + [os.path.abspath(src)]
        res = RcResult()

        def on_line(ln: str):
            ev = parse_event(ln)
            if not ev:
                return
            res.feed(ev)
            if ev.get("event") == "progress" and progress:
                progress(int(res.last_progress), 100, f"{product.upper()}: {res.last_progress:.0f}%")
            elif ev.get("topic") == "modelDownload" and progress:
                progress(0, 100, f"{product.upper()}: downloading model {ev.get('file')}")

        t0 = time.time()
        rc, tail = run_streaming(argv, kind="rcastro", on_line=on_line, checkpoint=checkpoint)
        msg = "; ".join(res.errors) or " | ".join(tail[-3:])
        if rc == 77:
            self._info_cache = None
            raise ToolUnavailable(f"rc-astro {product}", msg or "not licensed", 77)
        if 100 <= rc <= 114:
            raise ToolError(f"rc-astro {product}", f"usage error: {msg}", rc, "\n".join(tail))
        if rc != 0 or not os.path.isfile(dst):
            raise ToolError(f"rc-astro {product}", f"failed (exit {rc}): {msg}", rc, "\n".join(tail))
        meta = {"tool": f"rc-astro {product}", "cli_version": res.cli_version, "flags": flags,
                "device": (res.device or {}).get("device"), "provider": (res.device or {}).get("provider"),
                "warnings": res.warnings, "seconds": round(time.time() - t0, 1)}
        if product == "sxt" and params.get("stars"):
            meta["stars_path"] = self._find_sidecar(dst, res.outputs, before, out_dir)
        return meta

    def _schema(self, product: str) -> dict:
        """The live parameter document when the CLI answers (licence-free query), else the packaged copy."""
        try:
            rc, out = run_quiet(self._argv(product, "--json", "--no-banner"), timeout=40, kind="rcastro")
            doc = json.loads(out[out.index("{"):])
            if doc.get("parameters"):
                return doc
        except Exception:
            pass
        return load_schema(product)

    @staticmethod
    def _find_sidecar(dst: str, outputs: list[str], before: set[str], out_dir: str) -> str | None:
        """The star image SXT wrote beside the starless output: named in a ``saving`` event (the name
        differs between CLI versions: ``-stars``, ``-difference``, ...), else a new file in the folder."""
        for o in outputs:
            p = o if os.path.isabs(o) else os.path.join(out_dir, o)
            if os.path.abspath(p) != dst and os.path.isfile(p):
                return p
        stem = os.path.splitext(os.path.basename(dst))[0]
        for f in sorted(set(os.listdir(out_dir)) - before):
            if f.startswith(stem) and os.path.join(out_dir, f) != dst:
                return os.path.join(out_dir, f)
        return None

    def status(self) -> dict:
        info = self.info(refresh=False)
        if not info.get("ok"):
            return {"ok": False, "version": None, "schema": None, "account": None,
                    "activationUrl": info.get("activationUrl"), "products": [], "reason": info.get("reason"),
                    "attention": None}
        attention = None
        if not info.get("account"):
            attention = {"reason": "unregistered",
                         "message": "This computer is not registered with an RC-Astro account. Get an activation code "
                                    f"at {info.get('activationUrl')} and enter it here.",
                         "command": "rc-astro license"}
        return {"ok": True, "version": info.get("cliVersion"), "schema": info.get("schemaVersion"),
                "account": (info.get("account") or {}).get("email"), "activationUrl": info.get("activationUrl"),
                "products": info["products"], "attention": attention}


# --------------------------------------------------------------------------- module singletons

# --------------------------------------------------------------------------- Gaia DR3 SPCC catalogue: a per-field cache
# Siril 1.4's SPCC reads Gaia DR3 xp_sampled spectra from Zenodo by HTTP range requests on every run (minutes per field; a dropped connection fails or crashes the
# command) or from a local copy of the catalogue.  The local copy is split into 48 level-1 HEALPix chunks (nested order; Siril_Catalog_Installer.py's layout
# "<siril user data>/siril_cat1_healpix8_xpsamp/siril_cat1_healpix8_xpsamp_<n>.dat", registered as core.catalogue_gaia_photo).  A partial set is legitimate:
# Siril's own installer offers chunks by latitude or by area.  Here the chunks covering a field are fetched when that field is first colour calibrated and kept,
# so the cache grows only with the sky the user images (27-626 MB compressed, about twice that unpacked, per chunk) instead of the whole 21 GB.
GAIA_SPCC_RECORD = 14738271
GAIA_SPCC_CHUNKS = 48
GAIA_SPCC_NAME = "siril_cat1_healpix8_xpsamp"
GAIA_SPCC_ATTEMPTS = 3
GAIA_SPCC_MARGIN_DEG = 0.5               # around the field's radius: the stars SPCC samples near the edge
GAIA_SPCC_TIMEOUT = 45                   # s per connection / read: Zenodo resets TLS connections when it is overloaded; SPCC itself runs online meanwhile
GAIA_SPCC_BACKOFF = 1800                 # s: after a failed fetch no further fetch is tried (every queued job would otherwise wait the attempts out)


def healpix1_pixels(ra_deg: float, dec_deg: float, radius_deg: float) -> list[int]:
    """The level-1 (nside 2, nested) HEALPix pixels within ``radius_deg`` of (ra, dec): the catalogue chunks a field needs."""
    from astropy import units as u
    from astropy_healpix import HEALPix
    hp = HEALPix(nside=2, order="nested")
    px = hp.cone_search_lonlat(ra_deg * u.deg, dec_deg * u.deg, radius_deg * u.deg)
    return sorted(int(i) for i in np.atleast_1d(px))


@dataclass
class GaiaCatalog:
    """The cached chunks of the offline Gaia DR3 SPCC catalogue in Siril's user data folder."""

    def dir(self) -> str:
        home = os.environ.get("HOME") or os.path.expanduser("~")
        return os.path.join(home, ".local", "share", "siril", GAIA_SPCC_NAME)

    def chunk_path(self, n: int) -> str:
        return os.path.join(self.dir(), f"{GAIA_SPCC_NAME}_{n}.dat")

    def installed(self) -> list[int]:
        return [n for n in range(GAIA_SPCC_CHUNKS) if os.path.isfile(self.chunk_path(n))]

    def covers(self, pixels: list[int]) -> bool:
        return all(os.path.isfile(self.chunk_path(n)) for n in pixels)

    def status(self) -> dict:
        inst = self.installed()
        size = sum(os.path.getsize(self.chunk_path(n)) for n in inst)
        partial = sum(os.path.getsize(os.path.join(self.dir(), f)) for f in os.listdir(self.dir()) if f.endswith((".bz2", ".part"))) if os.path.isdir(self.dir()) else 0
        return {"ok": bool(inst), "chunks": inst, "installed": len(inst), "total": GAIA_SPCC_CHUNKS, "bytes": size, "partial_bytes": partial, "path": self.dir(),
                "record": f"https://zenodo.org/records/{GAIA_SPCC_RECORD}",
                "paused_until": GaiaCatalog._backoff_until if time.time() < GaiaCatalog._backoff_until else None, "paused_why": GaiaCatalog._backoff_why if time.time() < GaiaCatalog._backoff_until else None}

    def clear(self) -> dict:
        n = 0
        if os.path.isdir(self.dir()):
            for f in os.listdir(self.dir()):
                if f.startswith(GAIA_SPCC_NAME):
                    os.remove(os.path.join(self.dir(), f))
                    n += 1
        return {"removed": n}

    _backoff_until: float = 0.0
    _backoff_why: str = ""

    def ensure_field(self, ra_deg: float, dec_deg: float, radius_deg: float, progress: Progress | None = None,
                     checkpoint: Callable[[], None] | None = None) -> dict:
        """Make sure the chunks covering the field are cached (download, sha256, decompress).  Returns the pixel list and what was fetched; raises ToolError when
        a chunk cannot be fetched (and then declines further fetches for ``GAIA_SPCC_BACKOFF`` seconds: the caller runs SPCC online)."""
        pixels = healpix1_pixels(ra_deg, dec_deg, radius_deg + GAIA_SPCC_MARGIN_DEG)
        missing = [n for n in pixels if not os.path.isfile(self.chunk_path(n))]
        if missing and time.time() < self._backoff_until:
            raise ToolError("gaia", f"not retried until {time.strftime('%H:%M', time.localtime(self._backoff_until))}: {self._backoff_why}")
        os.makedirs(self.dir(), exist_ok=True)
        try:
            for i, n in enumerate(missing):
                self._install_chunk(n, i, len(missing), progress, checkpoint)
        except ToolError as e:
            GaiaCatalog._backoff_until, GaiaCatalog._backoff_why = time.time() + GAIA_SPCC_BACKOFF, str(e)[:160]
            raise
        return {"pixels": pixels, "fetched": missing}

    def _install_chunk(self, n: int, i: int, total: int, progress, checkpoint) -> None:
        name = f"{GAIA_SPCC_NAME}_{n}.dat"
        url = f"https://zenodo.org/records/{GAIA_SPCC_RECORD}/files/{name}.bz2"
        bz, part = os.path.join(self.dir(), name + ".bz2"), os.path.join(self.dir(), name + ".part")
        last: Exception | None = None
        for attempt in range(GAIA_SPCC_ATTEMPTS):
            try:
                with urllib.request.urlopen(url + ".sha256sum", timeout=GAIA_SPCC_TIMEOUT) as r:
                    want = r.read().decode("ascii", "replace").split()[0].lower()
                have = os.path.getsize(bz) if os.path.exists(bz) else 0
                try:
                    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
                    with urllib.request.urlopen(req, timeout=GAIA_SPCC_TIMEOUT) as r:
                        resumed = have > 0 and r.status == 206
                        if not resumed:
                            have = 0
                        total_b = int(r.headers.get("Content-Length") or 0) + have
                        got = have
                        with open(bz, "ab" if resumed else "wb") as f:
                            while True:
                                chunk = r.read(1 << 20)
                                if not chunk:
                                    break
                                f.write(chunk)
                                got += len(chunk)
                                if progress:
                                    progress(i + 0.9 * got / max(total_b, got), total, f"Gaia SPCC catalogue: sky chunk {n} ({got / 1e6:.0f} of {total_b / 1e6:.0f} MB), cached for every later field here")
                                if checkpoint:
                                    checkpoint()
                except urllib.error.HTTPError as e:
                    if e.code != 416 or not have:                   # 416: the partial file already is the whole file
                        raise
                sha = hashlib.sha256()
                with open(bz, "rb") as f:
                    for blk in iter(lambda: f.read(1 << 22), b""):
                        sha.update(blk)
                if sha.hexdigest() != want:
                    os.remove(bz)
                    raise ToolError("gaia", f"chunk {n}: checksum mismatch (removed)")
                if progress:
                    progress(i + 0.92, total, f"Gaia SPCC catalogue: unpacking sky chunk {n}")
                with bz2.BZ2File(bz, "rb") as fi, open(part, "wb") as fo:
                    shutil.copyfileobj(fi, fo, 1 << 22)
                os.replace(part, self.chunk_path(n))
                os.remove(bz)
                return
            except (OSError, ToolError) as e:
                last = e
                if os.path.exists(part):
                    os.remove(part)
                if checkpoint:
                    checkpoint()
                time.sleep(5.0 * (attempt + 1))
        raise ToolError("gaia", f"sky chunk {n} could not be cached after {GAIA_SPCC_ATTEMPTS} attempts: {last}")



graxpert = GraXpert()
starnet = StarNet()
rcastro = RCAstro()
gaia = GaiaCatalog()


def tools_status() -> dict:
    """The ``GET /api/tools`` document (DESIGN section 5)."""
    return {"siril": siril_status(), "graxpert": graxpert.status(), "starnet": starnet.status(),
            "rcastro": rcastro.status(), "gaia": gaia.status()}
