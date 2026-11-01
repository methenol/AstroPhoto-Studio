#!/usr/bin/env python3
"""Report which external tools are usable in this environment, as JSON on stdout.

Why: the pipeline depends on Siril, GraXpert, StarNet2 and RC-Astro; this tells in one call whether each
binary runs, which version it is and whether the GPU / CUDA libraries are visible to it.  Read-only: it
never activates a licence or downloads anything.  Run in the container:
    docker run --rm --gpus all --entrypoint python astrophoto-studio:cuda /app/tools/check_tools.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SIRIL = os.environ.get("SIRIL_BIN", "/usr/local/bin/siril-cli")
GX_PY = os.environ.get("GRAXPERT_BIN", "/opt/graxpert/bin/python")
GX_RUN = os.environ.get("GRAXPERT_RUNNER", "/opt/graxpert/gx_run.py")
STARNET_DIR = os.environ.get("STARNET_DIR", "/opt/starnet2")
RCASTRO = os.environ.get("RCASTRO_BIN", "/usr/local/bin/rc-astro")
TOOL_LD = os.environ.get("TOOL_LD_LIBRARY_PATH", "")


def _run(argv: list[str], ld: bool = False, timeout: int = 60) -> tuple[int, str, str]:
    env = dict(os.environ)
    if ld and TOOL_LD:
        env["LD_LIBRARY_PATH"] = TOOL_LD + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, p.stdout, p.stderr
    except Exception as e:  # missing binary, timeout
        return -1, "", f"{type(e).__name__}: {e}"


def _last(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def siril() -> dict:
    rc, out, err = _run([SIRIL, "-v"])
    return {"ok": rc == 0, "path": SIRIL, "version": _last(out), "error": "" if rc == 0 else _last(err)}


def graxpert() -> dict:
    rc, out, err = _run([GX_PY, GX_RUN, "-v"], ld=True, timeout=120)
    ver = next((ln for ln in (out + err).splitlines() if "version" in ln.lower()), _last(out))
    providers: list[str] = []
    rc2, out2, _ = _run([GX_PY, "-c", "import onnxruntime as o; print(','.join(o.get_available_providers()))"], ld=True)
    if rc2 == 0:
        providers = out2.strip().split(",")
    return {"ok": rc == 0, "runner": GX_RUN, "version": ver.strip(), "onnx_providers": providers,
            "gpu": "CUDAExecutionProvider" in providers}


def starnet() -> dict:
    exe = shutil.which("starnet2") or str(Path(STARNET_DIR) / "starnet2")
    if not os.access(exe, os.X_OK):
        return {"ok": False, "path": None, "error": "not installed"}
    rc, out, err = _run([exe, "--version"])
    return {"ok": rc == 0, "path": exe, "version": _last(out) or _last(err)}


def rcastro() -> dict:
    if not os.access(RCASTRO, os.X_OK):
        return {"ok": False, "path": None, "error": "installer not provided at build time"}
    rc, out, err = _run([RCASTRO, "--json", "--no-banner"], ld=True)
    res: dict = {"ok": rc == 0, "path": RCASTRO}
    try:
        d = json.loads(out)
        res["schemaVersion"] = d.get("schemaVersion")
        res["account"] = d.get("account")
        res["products"] = [{"key": p.get("key"), "status": (p.get("license") or {}).get("status"),
                            "valid": (p.get("license") or {}).get("valid")} for p in d.get("products", [])]
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e} {_last(err)}"
    rc, out, err = _run([RCASTRO, "--device", "--json"], ld=True)
    try:
        res["devices"] = json.loads(out)
    except Exception:
        res["devices_raw"] = (out or err).strip()[:500]
    prov = Path(RCASTRO).resolve().parent / "librca_onnxruntime_providers_cuda.so"
    if prov.exists():
        rc, out, _ = _run(["ldd", str(prov)], ld=True)
        missing = [ln.split()[0] for ln in out.splitlines() if "not found" in ln]
        res["cuda_libs"] = {"provider": str(prov), "missing": missing, "resolved": rc == 0 and not missing}
    return res


def nvidia() -> dict:
    rc, out, _ = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    return {"visible": rc == 0 and bool(out.strip()), "gpus": [ln.strip() for ln in out.splitlines() if ln.strip()]}


def main() -> int:
    report = {"tool_ld_library_path": TOOL_LD, "nvidia": nvidia(), "siril": siril(), "graxpert": graxpert(),
              "starnet2": starnet(), "rcastro": rcastro()}
    json.dump(report, sys.stdout, indent=2)
    print()
    return 0 if report["siril"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
