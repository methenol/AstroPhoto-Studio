"""Denoiser-loss ablation: does the production asinh-domain MSE buy its noise reduction with a
systematic bias in faint signal?  Same U-Net, patches, seed and budget for every objective
(astrophoto.denoise.make_n2n_loss):

  asinh_mse       production: MSE in the variance-stabilised domain (Jensen-gap bias possible)
  asinh_unbiased  nonlinear Noise2Noise with the expected transformed target (cf. arXiv:2512.24794)
  lin_mse         MSE in linear units (what ImageMM's N2N pass uses)
  lin_chi2        inverse-variance chi² in linear units (arXiv:2609.21350)
  lin_huber       robust chi² (Huber, 3 sigma)

Scores: the held-out lin / str errors (bench.score); on the synthetic twin of each dataset
(bench.synthetic: exact truth, measured noise law) the rms error, faint-region error and the
signed bias by signal level (bench.bias_metrics); for all: AstroSURE-style detection rate /
false-alarm rate at a common absolute threshold and STAR-style aperture flux error.

usage: python exp_denoise_loss.py [datasets...] [--losses l1,l2] [--iters N] [--crop PX]
                                  [--no-real | --no-synthetic] [--seed S] [--out results.json]
"""
import argparse
import json
import os

import numpy as np
import torch

import bench
import models
from astrophoto.denoise import N2N_LOSSES, make_n2n_loss, pick_device


def run(B, losses, iters, dev, seed, tag):
    ga, gb = B["stab"].fwd(B["a"]), B["stab"].fwd(B["b"])
    var = B["nm"].var()
    ref = bench.reference_catalog(B)
    res = {}

    def evaluate(est, name, seconds=None):
        s = bench.score(est, B)
        if B.get("synthetic"):
            s.update(bench.bias_metrics(est, B))
        s.update(bench.detection_metrics(est, B, ref))
        s.update(bench.flux_metrics(est, B, ref))
        if seconds is not None:
            s["seconds"] = round(seconds, 1)
        res[name] = s
        extra = f"  rmse {s['rmse_sigma']:.3f}σ faint bias {s['faint_bias_sigma']:+.4f}σ" if B.get("synthetic") else ""
        print(f"{tag:12s} {name:15s} lin {s['lin']:.3f} ({s['lin_db']:+.2f} dB)  str {s['str']:.3f} ({s['str_db']:+.2f} dB)"
              f"{extra}  DR {s.get('dr', 0):.3f} (faint {s.get('dr_faint') or 0:.3f}, n {s['n_ref']}/{s['n_ref_faint']}) "
              f"FAR {s.get('far', 0):.3f} (n_det {s['n_det']})"
              f"  flux err {s.get('flux_err', float('nan')):.4f} bias {s.get('flux_bias', float('nan')):+.4f}"
              + (f"  {s['seconds']}s" if seconds is not None else ""), flush=True)

    evaluate(B["a"], "raw A")
    for kind in losses:
        lf = make_n2n_loss(kind, B["stab"], dev)

        def loss_fn(pred, tgt, inp, v, lf=lf):
            return lf(pred, tgt, v)
        with bench.Timer() as t:
            net = models.train(models.make("unet"), ga, gb, B["train"], iters=iters, device=dev, seed=seed,
                               loss_fn=loss_fn, aux=var, tag=f"{tag}/{kind}", log_every=max(iters // 2, 1))
            est = B["stab"].inv(models.infer(net, ga, device=dev, tta=1))
        evaluate(est, kind, t.dt)
        del net
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=["M42", "M31"])
    ap.add_argument("--losses", default=",".join(N2N_LOSSES))
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--crop", type=int, default=bench.CROP)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-real", action="store_true")
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument("--out", default="results_denoise_loss.json")
    args = ap.parse_args()
    dev = pick_device()
    losses = args.losses.split(",")
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for name in args.datasets:
        B = bench.prepare(name, args.crop)
        print(f"{name}: crop {B['a'].shape[1]} px, half sigma {np.round(B['stab'].sigma, 2)}, "
              f"noise law (c0, c1, sky) {[tuple(round(float(v), 3) for v in t) for t in zip(*bench.variance_law(B['a'], B['b'], B['full']))]}",
              flush=True)
        if not args.no_real:
            res[name] = run(B, losses, args.iters, dev, args.seed, name)
            res[name]["_setup"] = {"crop": args.crop, "iters": args.iters, "seed": args.seed}
            json.dump(res, open(args.out, "w"), indent=1)
        if not args.no_synthetic:
            S = bench.synthetic(B, seed=args.seed)
            res[S["name"]] = run(S, losses, args.iters, dev, args.seed, S["name"])
            res[S["name"]]["_setup"] = {"crop": args.crop, "iters": args.iters, "seed": args.seed,
                                        "law": [[float(v) for v in c] for c in S["law"]]}
            json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
