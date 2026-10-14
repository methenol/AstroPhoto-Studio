"""Small helpers shared by the session, the web server and the API (same semantics as the old
``astrophoto.pipeline`` functions the server imported: ``slugify``, ``split_folders``,
``dataset_slug``, ``clean_json``, ``restoration_done``)."""
from __future__ import annotations

import hashlib
import json
import os
import re

import numpy as np


class Cancelled(Exception):
    """Raised by ``Session.checkpoint`` (and propagated through SirilRunner / post) when the user cancels."""


def slugify(path: str) -> str:
    """Cache folder name of one dataset folder: sanitised basename + 6 hex digits of the abspath hash."""
    base = os.path.basename(os.path.normpath(path)) or "dataset"
    h = hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:6]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("_") + "-" + h


def split_folders(folder) -> list[str]:
    """The folders of a dataset: one path, a list, or several joined with ``os.pathsep`` (subs of the
    same target from several nights, stacked together).  Sorted + absolute so a selection names one
    session whatever order the client sent."""
    parts = folder if isinstance(folder, (list, tuple)) else str(folder).split(os.pathsep)
    return sorted({os.path.abspath(os.path.expanduser(p.strip())) for p in parts if p and p.strip()})


def dataset_slug(folder) -> str:
    """Cache folder name of a dataset: one folder -> ``slugify``; several -> ``<first>+<n-1>-<hash>``."""
    folders = split_folders(folder)
    if len(folders) == 1:
        return slugify(folders[0])
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.basename(folders[0]) or "dataset").strip("_")
    h = hashlib.sha1("\n".join(folders).encode()).hexdigest()[:6]
    return f"{base}+{len(folders) - 1}-{h}"


def clean_json(o):
    """Recursively replace NaN/inf with None and numpy scalars/arrays with Python types (strict JSON)."""
    if isinstance(o, dict):
        return {str(k): clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean_json(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(path: str, obj, indent: int = 1) -> None:
    """Atomic strict-JSON write (the UI reads these files while jobs run)."""
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(clean_json(obj), f, indent=indent, default=json_default)
    os.replace(tmp, path)


def read_json(path: str, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def restoration_done(session_dir: str) -> bool:
    """The Restore step has finished: ``linear.fits`` exists (written last, atomically)."""
    return os.path.exists(os.path.join(session_dir, "linear.fits"))
