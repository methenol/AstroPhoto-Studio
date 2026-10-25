"""Command line: ``python -m sirilpipe run|analyse|stack|calibration DIR [DIR ...] [--workdir W] [--limit N]``."""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import __version__
from .session import Session
from .util import clean_json


def _progress():
    last = {"msg": None, "t": 0.0}

    def p(i, n, msg):
        now = time.time()
        key = msg.split(" (")[0]
        if key != last["msg"] or now - last["t"] > 5 or i >= n:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
            last["msg"], last["t"] = key, now
    return p


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m sirilpipe", description="AstroPhoto Studio core (Siril pipeline)")
    ap.add_argument("--version", action="version", version=f"sirilpipe {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("calibration", "find / build calibration masters and print the choice"),
                        ("analyse", "calibrate, register and measure every sub (frames.json)"),
                        ("stack", "analyse + stack + crop + plate solve (stack.fits, stack_meta.json)"),
                        ("run", "everything: stack, restore, star removal, auto-finish, export")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("folders", nargs="+", help="session folder(s); several = several nights of one target")
        p.add_argument("--workdir", default="output")
        p.add_argument("--limit", type=int, default=None, help="development: use only the first N subs")
        p.add_argument("--stack-param", "-s", action="append", default=[], metavar="KEY=VALUE",
                       help="stack_params, e.g. -s scale=2 -s mode=demosaic -s sensitivity=1.5")
        p.add_argument("--param", "-p", action="append", default=[], metavar="KEY=VALUE", help="processing params")
        p.add_argument("--no-solve", action="store_true", help="skip plate solving")
        if name == "run":
            p.add_argument("--hypertune", type=int, default=0, metavar="TRIALS",
                           help="hyper-tune each stage first: an experiment study of TRIALS trials per stage (much longer; the studies land in WORKDIR/lab/studies)")
    sub.add_parser("devices", help="the GPUs the AI tools can use (nvidia-smi) and which tools are available")
    sub.add_parser("tools", help="status of Siril, GraXpert, StarNet2 and the RC-Astro tools (licences, models)")
    return ap


def _kv(items: list[str]) -> dict:
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd in ("devices", "tools"):
        from . import tools as tools_mod
        st = tools_mod.tools_status()
        if args.cmd == "devices":
            try:
                import subprocess
                out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4).stdout.strip()
                print("GPUs:\n" + (out or "  none (CPU only)"))
            except Exception:                                         # noqa: BLE001
                print("GPUs: none (nvidia-smi not available)")
            print("GraXpert GPU:", st.get("graxpert", {}).get("gpu"), "| RC-Astro:", [(p["key"], p["status"]) for p in st.get("rcastro", {}).get("products", [])])
        else:
            print(json.dumps(st, indent=1, default=str))
        return 0
    s = Session(args.folders, args.workdir, limit=args.limit)
    sp, pr, prog = _kv(args.stack_param), _kv(args.param), _progress()
    t0 = time.time()
    with s.local_copy(prog):                       # ASTROPHOTO_LOCAL_COPY
        if args.cmd == "calibration":
            s.run_calibration(prog)
            print(json.dumps(s.calibration_info(), indent=1))
        elif args.cmd == "analyse":
            print(s.run_analysis(float(sp.get("sensitivity", 1.0)), prog))
        elif args.cmd == "stack":
            s.run_analysis(float(sp.get("sensitivity", 1.0)), prog, sp)
            meta = s.run_stack(sp, prog)
            if not args.no_solve:
                s.plate_solve(prog)
            print(json.dumps(clean_json(s.meta), indent=1))
        elif args.cmd == "run":
            if args.hypertune:
                sp.update(hypertune=True, hypertune_trials=int(args.hypertune))
            print(json.dumps(clean_json(s.run_all(sp, pr, {}, prog)), indent=1))
    print(f"done in {time.time() - t0:.0f} s; session folder {s.dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
