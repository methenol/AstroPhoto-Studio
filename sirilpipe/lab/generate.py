"""Generate a synthetic dataset in the background.

    python -m sirilpipe.lab.generate <dataset_dir>

reads ``<dataset_dir>/spec.json`` (``synthetic.DEFAULT_SPEC`` keys), writes the subs and the truth
(``synthetic.generate``), then analyses and stacks them with the pipeline defaults (Siril) so that
every task can use the dataset at once.  Progress in ``<dataset_dir>/status.json``.
"""
import json
import os
import sys
import time
import traceback


def main():
    d = os.path.abspath(sys.argv[1])
    status = {"state": "running", "pid": os.getpid(), "started": time.time(), "message": "Starting"}

    def write():
        tmp = os.path.join(d, "status.json.tmp")
        json.dump(status, open(tmp, "w"))
        os.replace(tmp, os.path.join(d, "status.json"))

    def progress(i, n, msg):
        status["message"], status["progress"] = msg, i / max(n, 1)
        write()
    write()
    try:
        from . import synthetic
        from .tasks import Dataset
        spec = json.load(open(os.path.join(d, "spec.json")))
        synthetic.generate(spec, d, progress=progress)
        ds = Dataset({"kind": "synthetic", "dir": d}, d)
        import threading
        ds.ensure_stacked(lambda m: progress(0, 1, m), threading.Event())
        status["state"], status["message"] = "done", "Ready"
    except Exception as e:
        status["state"], status["message"] = "error", f"{type(e).__name__}: {e}"
        status["traceback"] = traceback.format_exc()
    status["ended"] = time.time()
    write()


if __name__ == "__main__":
    main()
