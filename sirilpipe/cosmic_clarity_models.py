"""Download and install the Cosmic Clarity AI4 models with siril-scripts' ``processing/CosmicClarity_Native.py`` itself (its *Download Models* button, headless).

Run by the pyscript virtualenv's Python (sirilpy, PyQt6, torch), not by Siril: ``python cosmic_clarity_models.py --script <CosmicClarity_Native.py> --models-dir <dir>
[--zip <models.zip>]``.  The script file is imported as it is (its window is never started) and its own ``download_models`` (the HuggingFace mirror of SetiAstro's models,
``HF_MODELS_URL``) and ``install_models_zip`` put the models where its command-line mode looks for them (``<Siril user data>/cosmic_clarity``).  ``--zip`` installs an
archive already downloaded instead.  Progress lines are printed for the caller (``tools.CosmicClarity.download_models``).
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def main() -> int:
    ap = argparse.ArgumentParser(prog="cosmic_clarity_models.py")
    ap.add_argument("--script", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--zip", default=None, help="install this models.zip instead of downloading it")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("cosmic_clarity_native", a.script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    dst = Path(a.models_dir)
    dst.mkdir(parents=True, exist_ok=True)
    mod.set_models_dir_override(dst)

    def say(msg: str) -> None:
        print(msg, flush=True)

    if a.zip:
        archive = Path(a.zip)
    else:
        archive = dst.parent / "cosmic_clarity_models.zip"
        say(f"Downloading {mod.HF_MODELS_URL}")
        mod.download_models(archive, progress_cb=say)
    try:
        mod.install_models_zip(archive, progress_cb=say)
    finally:
        if not a.zip and archive.exists():
            archive.unlink()
    # a model is there as PyTorch weights or as its ONNX file (the mirror carries DarkStar as ONNX only; cosmic_clarity_worker.py runs it through onnxruntime)
    missing = sorted({k for keys in mod.MODE_MODELS.values() for k in keys
                      if not mod.model_path(k).exists() and not (k + "_onnx" in mod.MODEL_FILES and mod.model_path(k + "_onnx").exists())})
    if missing:
        say(f"ERROR: models still missing after the installation: {missing}")
        return 1
    say(f"Cosmic Clarity models installed in {dst} (script {mod.VERSION})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
