"""Run one Cosmic Clarity AI4 operation with siril-scripts' ``processing/CosmicClarity_Native.py`` (Cosmic Clarity engines and models (c) SetiAstro / Franklin Marek;
Siril adaptation Adrian Knagg-Baugh, GPL-3.0-or-later) on a numpy image, headless.

Executed by the pyscript virtualenv's Python (sirilpy, PyQt6, torch; onnxruntime for the ONNX-only models), not by Siril:

    python cosmic_clarity_worker.py --script <CosmicClarity_Native.py> --models-dir <dir> --input in.npy --output out.npy [--stars-output so.npy]
                                    -- --mode sharpen --sharpening-mode Both --stellar-amount 0.5 ...

Everything after ``--`` is the script's own command line, parsed by its own ``_build_arg_parser``; the parameters are then put together exactly as its ``run_cli`` does
(tile size and overlap snapped to multiples of 32, ``target_median`` 0.25, the colour strength defaulting to the luminance strength) and the script's ``process_sharpen`` /
``process_denoise`` / ``process_superres`` / ``process_darkstar`` / ``process_satellite`` - the functions its CLI and its GUI call - run on the image (H, W, 3, float32
in [0, 1]).  Why not ``pyscript CosmicClarity_Native.py --mode ...`` inside Siril:

* its CLI accepts a mode only when the PyTorch weights of that mode are installed, while the HuggingFace mirror its *Download Models* button fetches carries DarkStar
  (``darkstar_*_AI4``) as ONNX only; the script's model loaders run those ONNX models (``set_backend_pref("onnx")``, its GUI's backend setting) - here the backend is
  switched to ONNX for a mode only when its PyTorch weights are missing;
* its CLI's ``--stars-only`` branch refers to ``self`` outside a class (NameError, script 1.2.0): the star layer is wanted here anyway as ``image - starless``
  (``--darkstar-mode additive``, the pipeline's rule for every star remover), so the starless image is what is kept.

Progress lines ``progress: <done>/<total> <label>`` are printed for the caller (``tools.CosmicClarity.run``).
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _load(script: str):
    spec = importlib.util.spec_from_file_location("cosmic_clarity_native", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cli_params(mod, args) -> dict:
    """``run_cli``'s parameter dictionary (CosmicClarity_Native.py 1.2.0, verbatim apart from the indentation)."""
    tile_size = max(128, min(1024, (args.tile_size // 32) * 32))
    overlap = max(32, min(96, (args.overlap // 32) * 32))
    overlap = min(overlap, (tile_size // 2) & ~31)
    overlap = max(32, overlap)
    params: dict = {"use_gpu": not args.cpu, "chunk_size": tile_size, "overlap": overlap}
    mode = args.mode
    if mode == "sharpen":
        params.update({"sharpening_mode": args.sharpening_mode, "stellar_amount": args.stellar_amount, "nonstellar_amount": args.nonstellar_amount,
                       "nonstellar_psf": args.nonstellar_psf, "auto_psf": not args.no_auto_psf, "sharpen_channels_separately": args.sharpen_channels,
                       "temp_stretch": args.sharpen_stretch, "target_median": 0.25})
    elif mode == "denoise":
        params.update({"denoise_mode": args.denoise_mode, "denoise_strength": args.denoise_strength,
                       "denoise_color": args.denoise_color if args.denoise_color is not None else args.denoise_strength,
                       "lite": args.lite, "walking": args.walking, "temp_stretch": args.denoise_stretch, "target_median": 0.25})
    elif mode == "superres":
        params.update({"scale": args.scale})
    elif mode == "darkstar":
        params.update({"darkstar_mode": args.darkstar_mode, "output_stars_only": args.stars_only})
    elif mode == "satellite":
        params.update({"sat_mode": args.sat_mode, "clip_trail": not args.no_clip_trail, "sensitivity": args.sensitivity,
                       "temp_stretch": args.sat_stretch, "target_median": 0.25})
    return params


def model_keys(mod, args) -> list[str]:
    """The script's model keys the mode loads (``MODE_MODELS``; superres: only the chosen scale's)."""
    if args.mode == "denoise":
        return mod.MODE_MODELS["denoise_1w" if args.walking else "denoise_lite" if args.lite else "denoise"]
    if args.mode == "superres":
        return [f"superres_{args.scale}x"]
    return mod.MODE_MODELS[args.mode]


def choose_backend(mod, args) -> str:
    """"auto" (the script's default: PyTorch on a GPU) when the mode's PyTorch weights are installed, else "onnx" when its ONNX files are (and onnxruntime is)."""
    keys = model_keys(mod, args)
    if all(mod.model_path(k).exists() for k in keys):
        return "auto"
    onnx = [k + "_onnx" for k in keys]
    if all(k in mod.MODEL_FILES and mod.model_path(k).exists() for k in onnx):
        if mod._get_ort() is None:
            raise RuntimeError(f"the {args.mode} models are installed as ONNX only and onnxruntime is not available")
        return "onnx"
    missing = [mod.MODEL_FILES[k] for k in keys if not mod.model_path(k).exists()]
    raise RuntimeError(f"Cosmic Clarity models missing for {args.mode}: {missing} (in {mod.models_dir()})")


def main() -> int:
    argv = sys.argv[1:]
    rest = []
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(prog="cosmic_clarity_worker.py")
    ap.add_argument("--script", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--info", default=None, help="write what ran (backend, device, parameters) to this JSON file")
    a = ap.parse_args(argv)

    import numpy as np
    mod = _load(a.script)
    mod.set_models_dir_override(Path(a.models_dir))
    args = mod._build_arg_parser().parse_args(rest)
    params = cli_params(mod, args)
    backend = choose_backend(mod, args)
    mod.set_backend_pref(backend)

    last = [0.0]

    def progress(done: int, total: int, label: str) -> None:
        now = time.time()
        if done >= total or now - last[0] > 1.0:
            last[0] = now
            print(f"progress: {done}/{max(total, 1)} {label}", flush=True)

    img = np.load(a.input).astype(np.float32, copy=False)
    t0 = time.time()
    info: dict = {"mode": args.mode, "backend": backend, "params": params, "script_version": mod.VERSION,
                  "device": str(mod.get_device(params["use_gpu"]))}
    if args.mode == "sharpen":
        out = mod.process_sharpen(img, params, progress_cb=progress)
    elif args.mode == "denoise":
        out = mod.process_denoise(img, params, progress_cb=progress)
    elif args.mode == "superres":
        out = mod.process_superres(img, params, progress_cb=progress)
    elif args.mode == "darkstar":
        out, _ = mod.process_darkstar(img, {**params, "output_stars_only": False}, progress_cb=progress)
    elif args.mode == "satellite":
        out, detected = mod.process_satellite(img, params, progress_cb=progress)
        info["trails_detected"] = bool(detected)
    else:
        raise ValueError(args.mode)
    info["seconds"] = round(time.time() - t0, 1)
    if backend == "onnx":                                             # which execution provider onnxruntime really used (it falls back to the CPU silently)
        info["onnx_providers"] = sorted({p for m in mod._MODEL_CACHE.values() for v in vars(m).values()
                                         if hasattr(v, "get_providers") for p in v.get_providers()})
    np.save(a.output, np.ascontiguousarray(out, dtype=np.float32))
    if a.info:
        with open(a.info, "w") as f:
            json.dump(info, f, default=str)
    print(f"Cosmic Clarity [{args.mode}] complete ({backend}, {info['device']}, {info['seconds']} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
