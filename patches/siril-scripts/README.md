# Patches to siril-scripts

Applied by the Dockerfile to the image's clone of https://gitlab.com/free-astro/siril-scripts with `git apply`
(a patch that no longer applies fails the build); the entrypoint replaces the volume's copy whenever it is not
the image's.

- `statistical-stretch-black-point.patch` - `processing/Statistical_Stretch.py` (3.1.3, "exact port from SAS Pro")
  rescales `(img - black_point) / (1 - black_point)` and feeds it to its rational curve without clamping at 0.
  Below the black point the curve passes its pole at `r = -m(1 - t) / (t - m)` (m the rescaled median, t the
  target) and returns values >= 1: pixels darker than the black point come out white.  After a denoiser the
  sky noise is tiny, the black point sits a hair under the median and the pole a hair under the black point
  (M 42 Seestar 2026-11-04: black point 0.00365, median 0.00384, every pixel under ~0.0019 white), so a dark
  lane beside the Trapezium became a flat magenta - red after HOO - blob.  SAS Pro has the same formula (its
  numba kernel and numpy fallback, `imageops/stretch.py`).  The script's own clip statistics count those
  pixels as black-clipped; the patch clamps the rescaled value at 0 in its three stretch paths.
