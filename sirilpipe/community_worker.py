"""Run a function of a siril-scripts Python script headless, on numpy arrays (executed by the pyscript virtualenv's Python, which has sirilpy and PyQt6; stdlib + numpy only here).

``python community_worker.py --script <siril-scripts file> --entry <function> --input in.npy --output-prefix out --kwargs '[{...}, {...}]'``

The script file is imported as it is (its Qt window code is never started: every siril-scripts script guards ``main()``), the function ``entry`` is called once per kwargs
dict with the array as first argument, and the results are saved as ``<prefix>_<i>.npy``.  This is the community code itself, not a port: e.g. VeraLux HyperMetric
Stretch's ``process_veralux_v6`` (the function its PROCESS button calls), HDR_multiscale's ``compute_wavescale_hdr`` or NarrowbandNormalization's ``process_image``.
Inputs are saved as given (``HWC`` for the colour functions); a tuple result is reduced to its first element.
"""
import argparse
import importlib.util
import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--entry", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-prefix", required=True)
    ap.add_argument("--kwargs", default="[{}]")
    ap.add_argument("--args", default="[]", help="JSON list of positional arguments after the array")
    a = ap.parse_args()
    import numpy as np
    spec = importlib.util.spec_from_file_location("community_script", a.script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, a.entry)
    img = np.load(a.input)
    pos = json.loads(a.args)
    for i, kw in enumerate(json.loads(a.kwargs)):
        out = fn(img.copy(), *pos, **kw)
        if isinstance(out, tuple):                       # e.g. compute_wavescale_hdr -> (image, mask): the image is the result
            out = out[0]
        np.save(f"{a.output_prefix}_{i}.npy", np.asarray(out, dtype=np.float32))
    return 0


if __name__ == "__main__":
    sys.exit(main())
