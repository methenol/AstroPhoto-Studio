"""Paths and environment of the external tools (Siril, GraXpert, StarNet, RC-Astro).

Everything is read from the environment *on access* (module ``__getattr__``), so tests and the
web UI can change a variable and see it take effect, and the defaults match the Docker image
(DESIGN.md section 6).  ``siril_env()`` / ``tool_env()`` build the environment for subprocesses:
Siril needs a writable ``HOME`` (it writes ``~/.config/siril`` and reads the SPCC database and the
script repository from ``~/.local/share``), headless Qt, and a C.UTF-8 locale.
"""
from __future__ import annotations

import os
import shutil

# name -> (environment variable, default); a callable default is evaluated on access.
_SETTINGS: dict[str, tuple[str, object]] = {
    "SIRIL_BIN": ("SIRIL_BIN", "/usr/local/bin/siril-cli"),
    "SIRIL_DATA": ("SIRIL_DATA", "/opt/siril-data"),
    "GRAXPERT_BIN": ("GRAXPERT_BIN", "/opt/graxpert/bin/python"),
    "GRAXPERT_RUNNER": ("GRAXPERT_RUNNER", "/opt/graxpert/gx_run.py"),
    "STARNET_DIR": ("STARNET_DIR", "/opt/starnet2"),
    "RCASTRO_BIN": ("RCASTRO_BIN", "rc-astro"),
    "TOOL_LD_LIBRARY_PATH": ("TOOL_LD_LIBRARY_PATH", ""),
    "CALIB_LIBRARY": ("ASTROPHOTO_CALIB", ""),
    "TOOLS_HOME": ("ASTROPHOTO_TOOLS_HOME", lambda: os.path.join(home_dir(), "tools")),
}

# Timeouts (seconds).  Generous: a 2400-sub drizzle stack takes hours; the cancel button, not a
# timeout, is the normal way out.  0 disables.
SIRIL_TIMEOUT = float(os.environ.get("SIRIL_TIMEOUT", 0) or 0)
TOOL_TIMEOUT = float(os.environ.get("TOOL_TIMEOUT", 3 * 3600) or 0)

# Frames above this count get Siril's cheaper normalisation estimators (-fastnorm).
FASTNORM_FRAMES = 1000


def home_dir() -> str:
    """The writable HOME used for tool configuration, licences and models."""
    return os.environ.get("HOME") or os.path.expanduser("~")


def __getattr__(name: str):
    if name in _SETTINGS:
        var, default = _SETTINGS[name]
        val = os.environ.get(var)
        if val not in (None, ""):
            return val
        return default() if callable(default) else default
    raise AttributeError(name)


def get(name: str) -> str:
    """Value of one of the settings above (SIRIL_BIN, SIRIL_DATA, GRAXPERT_BIN, ...)."""
    return __getattr__(name)


def siril_bin() -> str:
    """Path of ``siril-cli`` (SIRIL_BIN, else PATH, else the image's default)."""
    configured = os.environ.get("SIRIL_BIN")
    if configured:
        return configured
    return shutil.which("siril-cli") or get("SIRIL_BIN")


def siril_env(extra: dict | None = None) -> dict:
    """Environment for a ``siril-cli`` process: writable HOME, offscreen Qt, UTF-8 locale.

    ``APPDIR`` is deliberately not set here: the image's wrapper (``siril-cli`` -> ``AppRun``)
    sets it.  The Siril data repositories are made visible under ``$HOME/.local/share`` by the
    container entrypoint (HOME is on a volume)."""
    env = dict(os.environ)
    env.setdefault("HOME", home_dir())
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env["LC_ALL"] = env.get("LC_ALL") or "C.UTF-8"
    env.setdefault("XDG_CACHE_HOME", os.path.join(env["HOME"], ".cache"))
    if extra:
        env.update(extra)
    return env


def tool_env(extra: dict | None = None) -> dict:
    """Environment for GraXpert / StarNet / RC-Astro subprocesses: the CUDA user-space libraries
    (``TOOL_LD_LIBRARY_PATH``) are prepended to ``LD_LIBRARY_PATH`` for these tools only, never
    for Siril or the web server (Siril's AppImage ships its own libraries)."""
    env = dict(os.environ)
    env.setdefault("HOME", home_dir())
    ld = get("TOOL_LD_LIBRARY_PATH")
    if ld:
        env["LD_LIBRARY_PATH"] = ld + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    if extra:
        env.update(extra)
    return env


def library_folders() -> list[str]:
    """Extra calibration library folders (``ASTROPHOTO_CALIB``, ``os.pathsep``-separated).
    ``off``/``none``/``0``/``false`` disables the library lookup altogether (returns None-like [])."""
    raw = get("CALIB_LIBRARY").strip()
    if not raw or raw.lower() in ("off", "none", "0", "false"):
        return []
    return [p for p in raw.split(os.pathsep) if p.strip()]


def library_disabled() -> bool:
    return get("CALIB_LIBRARY").strip().lower() in ("off", "none", "0", "false")
