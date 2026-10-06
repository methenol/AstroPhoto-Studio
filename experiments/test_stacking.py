"""Stacker checks that need no data on disk.

    python experiments/test_stacking.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from astrophoto.stacking import Integrator, _block_median  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def drizzle_like(h=256, w=256, sky=1000.0, seed=0):
    """A sub as a Bayer drizzle leaves it: green everywhere, red empty (value and weight 0) on every
    other row inside a band - where a rotated red lattice misses the output pixels."""
    rng = np.random.default_rng(seed)
    vals = (sky + rng.normal(0, 10, (h, w, 3))).astype(np.float32)
    wts = np.ones((h, w, 3), np.float32)
    band = (slice(64, 192, 2), slice(0, w))
    vals[band + (0,)] = 0.0
    wts[band + (0,)] = 0.0
    return vals, wts


def test_block_median_per_channel():
    vals, wts = drizzle_like()
    good = _block_median(vals, wts > 0, 64)[..., 0]          # red's own validity
    old = _block_median(vals, wts[..., 1] > 0, 64)[..., 0]   # green's, as before 2026-10-06
    # where red has too few samples a block is undefined (NaN; the local normalisation fills it from its
    # neighbours), elsewhere it is the sky - never a level dragged down by empty pixels
    fin = np.isfinite(good)
    check("block median of red with red's validity: the sky, or undefined where red has no samples",
          np.allclose(good[fin], 1000, atol=5) and not fin[1:3].any() and fin[[0, 3]].all(),
          f"band blocks {good[1:3, 0]}, outside {good[[0, 3], 0].round(1)}")
    check("with green's validity the empty red pixels drag the band to 0 (the old failure)", float(old[1:3].mean()) < 100,
          f"band blocks {old[1:3, 0].round(1)}")


def test_normalise_offsets_per_channel():
    vals, wts = drizzle_like()
    it = Integrator.__new__(Integrator)                      # only the state _normalise uses
    it.offsets, it.gradients, it.ref_level = {}, {}, np.zeros(3, np.float32)
    _, valid = it._normalise(0, vals.copy(), wts, False)
    check("validity is per channel", valid.shape == wts.shape and not valid[64, 0, 0] and valid[64, 0, 1])
    check("per-sub offsets ignore a channel's empty pixels", np.allclose(it.offsets[0], 1000, atol=5),
          f"offsets {it.offsets[0].round(1)}")


if __name__ == "__main__":
    t0 = time.time()
    for f in (test_block_median_per_channel, test_normalise_offsets_per_channel):
        f()
    print(f"\n{len(FAILS)} failed: {FAILS}  ({time.time() - t0:.0f}s)")
