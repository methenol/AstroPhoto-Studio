"""ImageMM's operators and solver against their definitions (Sukurdeep et al. 2025, arXiv:2501.03002): the forward
model D H(t) x and its adjoint, the latent / exposure grid alignment, Eq. 11's solver, the multiplicative update on
noiseless data, and the saturated-pixel update (Whyte et al. 2014) for a flat-fielded ceiling.  Small sizes; they
need torch and are skipped without it."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from astrophoto import imagemm as M  # noqa: E402


def _gauss_kernel(k, s, dy=0.0, dx=0.0):
    yy, xx = np.mgrid[:k, :k] - (k - 1) / 2
    g = np.exp(-0.5 * ((yy - dy) ** 2 + (xx - dx) ** 2) / s ** 2)
    return (g / g.sum()).astype(np.float32)


@pytest.mark.parametrize("r", [1, 2])
def test_forward_is_valid_convolution_then_average_pooling(r):
    rng = np.random.default_rng(0)
    n, C, d = 3, 2, 12
    ks = 7 if r == 1 else 8
    K = torch.from_numpy(rng.random((n, C, ks, ks)).astype(np.float32))
    X, Y = M.latent_shape(d, d, ks, r)
    x = torch.from_numpy(rng.random((1, C, X, Y)).astype(np.float32))
    ops = M.Operators(K, r, chunk=2)
    got = torch.cat([ops.forward(x, a, b) for a, b in ops.ranges()])
    ref = torch.stack([torch.stack([F.conv2d(x[:, c:c + 1], K[t, c].flip(-2, -1)[None, None])[0, 0]
                                    for c in range(C)]) for t in range(n)])
    ref = F.avg_pool2d(ref, r) if r > 1 else ref
    assert got.shape == (n, C, d, d)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("r", [1, 2])
def test_adjoint_of_the_forward_model(r):
    """<D H x, z> = <x, H^T D^T z> / r^2: D averages r x r, D^T (the paper's) replicates without the 1 / r^2."""
    rng = np.random.default_rng(1)
    n, C, d = 4, 3, 10
    ks = 5 if r == 1 else 6
    K = torch.from_numpy(rng.random((n, C, ks, ks)).astype(np.float64))
    X, Y = M.latent_shape(d, d, ks, r)
    x = torch.from_numpy(rng.random((1, C, X, Y)))
    z = torch.from_numpy(rng.random((n, C, d, d)))
    ops = M.Operators(K, r)
    lhs = float((ops.forward(x, 0, n) * z).sum())
    rhs = float((x * ops.adjoint_sum(z, 0, n)).sum()) / r ** 2
    assert lhs == pytest.approx(rhs, rel=1e-10)


@pytest.mark.parametrize("r,ks", [(1, 9), (2, 10), (3, 9)])
def test_exposure_origin_aligns_latent_and_exposure_grids(r, ks):
    """A latent point at latent r i + exposure_origin is seen centred on exposure pixel i through a symmetric
    kernel (each exposure pixel pools the r x r outputs centred there)."""
    d = 9
    K = torch.from_numpy(_gauss_kernel(ks, 1.2 * r)[None, None])
    X, Y = M.latent_shape(d, d, ks, r)
    o = M.exposure_origin(ks, r)
    assert abs(o - round(o)) < 1e-9
    x = torch.zeros(1, 1, X, Y, dtype=torch.float32)
    i = 4
    x[0, 0, r * i + int(round(o)), r * i + int(round(o))] = 1.0
    y = M.Operators(K, r).forward(x, 0, 1)[0, 0].numpy()
    if r % 2 == 1:                                    # odd r: the point is at the centre of the pooled block
        assert np.unravel_index(np.argmax(y), y.shape) == (i, i)
        np.testing.assert_allclose(y[i - 1, i], y[i + 1, i], rtol=1e-4)


def test_refine_psfs_per_target_blur_matches_the_shared_one():
    f = np.stack([_gauss_kernel(15, 2.0 + 0.2 * q) for q in range(3)])
    g = M.gaussian_psf_mc(1.0, 2 * 15 - 1, n_samples=2_000_000)
    h1, m1 = M.refine_psfs(f, 1, 1.0, g=g, device="cpu")
    h2, m2 = M.refine_psfs(f, 1, 1.0, g=np.repeat(g[None], 3, 0), device="cpu")
    np.testing.assert_allclose(h1, h2, rtol=1e-4, atol=1e-7)


def test_mm_restore_reaches_the_noiseless_truth():
    """Algorithm 1 (r = 1, Gaussian loss) on noiseless exposures of a non-negative latent: the forward model of the
    result matches the data."""
    rng = np.random.default_rng(3)
    n, d, ks = 3, 24, 9
    K = torch.from_numpy(np.stack([_gauss_kernel(ks, s)[None] for s in (1.2, 1.5, 1.8)]))
    X, Y = M.latent_shape(d, d, ks, 1)
    truth = np.full((1, 1, X, Y), 0.01, np.float32)
    for _ in range(6):
        truth[0, 0, rng.integers(4, X - 4), rng.integers(4, Y - 4)] += rng.uniform(1, 5)
    ops = M.Operators(K, 1)
    y = ops.forward(torch.from_numpy(truth), 0, n)
    var = torch.full_like(y, 1e-4)
    mask = torch.ones_like(y)
    x0 = M.initial_guess(y, mask, ks, 1)
    x, info = M.mm_restore(y, var, mask, K, x0, r=1, robust=False, max_iters=3000, epsilon=1e-9)
    fit = ops.forward(x, 0, n)
    assert float((fit - y).abs().max()) < 2e-2 * float(y.max())


def test_whyte_holds_a_clipped_star_at_its_own_flat_fielded_ceiling():
    """A star clipped at a ceiling that differs per exposure and pixel (ceiling / flat): with each measurement's
    own level the restored model exceeds the ceiling where the data clip, and stays at the data elsewhere."""
    rng = np.random.default_rng(4)
    n, d, ks = 3, 32, 11
    K = torch.from_numpy(np.stack([_gauss_kernel(ks, s)[None] for s in (1.4, 1.6, 1.9)]))
    X, Y = M.latent_shape(d, d, ks, 1)
    lat = np.full((1, 1, X, Y), 1e-3, np.float32)
    lat[0, 0, X // 2, Y // 2] = 4.0 / float(_gauss_kernel(ks, 1.4).max())
    ops = M.Operators(K, 1)
    truth = ops.forward(torch.from_numpy(lat), 0, n)
    flat = torch.from_numpy((1.0 + 0.1 * rng.uniform(-1, 1, (n, 1, 1, 1))).astype(np.float32))
    L = (1.0 / flat).expand_as(truth).contiguous()
    y = torch.minimum(truth, L)
    sat = (truth >= L).float()
    var = torch.full_like(y, 1e-6)
    mask = 1.0 - sat
    x0 = M.initial_guess(y, torch.ones_like(y), ks, 1)
    wh = {"level": L, "saturated": sat, "phi": 0.9, "a": 50.0, "erode": 3, "soft": 3.0}
    x, _ = M.mm_restore(y, var, mask, K, x0, r=1, robust=True, max_iters=300, epsilon=0, whyte=wh)
    fit = ops.forward(x, 0, n)
    assert torch.isfinite(x).all()
    assert float((fit * sat).amax()) > float((L * sat).amax())          # above the ceiling where it clipped
    # beyond the measurements the saturated latent reaches (footprint + erosion + the clipped core), the plain fit
    c_e = X // 2 - (ks - 1) // 2
    yy, xx = np.mgrid[:d, :d]
    far = torch.from_numpy((np.hypot(yy - c_e, xx - c_e) > 15).astype(np.float32))
    assert float(far.sum()) > 0
    assert float(((fit - y).abs() * far).max()) < 0.01
