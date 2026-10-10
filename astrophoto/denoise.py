"""Self-supervised Noise2Noise denoising trained per dataset.

The stacker produces two half-stacks (even/odd frames) that contain the same
signal with *independent* noise.  A CNN trained to map half A -> half B (and
vice versa) with an L2 loss learns E[signal | noisy input] (Lehtinen et al.
2018, "Noise2Noise"), i.e. a denoiser tuned to this exact camera, sky and
integration – no pretrained model, no hallucinated detail from other images.

Training happens in a variance-stabilised domain g(x) = asinh((x-b)/(k*sigma)),
which is invertible, so the denoiser returns linear data and slots into the
linear stage of the pipeline.

The same half-stack pairs also train a *deconvolution* network (two-stage
design of ZS-DeconvNet, Qiao et al. 2024): its input is the denoised half A,
and its output x, blurred by the PSF measured from the stars, must predict
the other, raw half B (chi-squared loss with the measured per-pixel noise).
Because B's noise is independent of everything the network sees, the only
way to lower the loss is to recover the true, sharper sky.  Hessian
regularisation (as in ZS-DeconvNet) and a physical sky-floor prior (real flux
never dips below the local sky, which is what removes the dark "moats"
deconvolution normally leaves around stars) constrain what the PSF cannot see.
Benchmarks and ablations: experiments/README.md.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pick_device(preference: str = "auto") -> torch.device:
    """Choose a compute device: NVIDIA CUDA, Apple Metal (MPS) or CPU.

    ``preference`` may be "auto", "cuda", "cuda:1", "mps" or "cpu".  The
    ``ASTROPHOTO_DEVICE`` environment variable overrides "auto".
    """
    import os
    pref = (preference or "auto").lower()
    if pref == "auto":
        pref = os.environ.get("ASTROPHOTO_DEVICE", "auto").lower()
    if pref.startswith("cuda") and torch.cuda.is_available():
        dev = torch.device(pref)
        if dev.index is not None and dev.index >= torch.cuda.device_count():
            # e.g. ASTROPHOTO_DEVICE=cuda:1 on a single-GPU machine: the GPUs are numbered from 0
            print(f"[astrophoto] {pref} not present ({torch.cuda.device_count()} CUDA device(s)); using cuda:0")
            return torch.device("cuda:0")
        return dev
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info() -> dict:
    """Describe available accelerators (shown in the web UI)."""
    info = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(),
            "default": str(pick_device()), "gpus": []}
    if info["cuda"]:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            info["gpus"].append({"id": f"cuda:{i}", "name": p.name, "vram_gb": round(p.total_memory / 2**30, 1)})
    if info["mps"]:
        info["gpus"].append({"id": "mps", "name": "Apple Silicon GPU (Metal)", "vram_gb": None})
    return info


def _batch_and_tile(device: torch.device) -> tuple[int, int]:
    """Size training batch / inference tile to the available VRAM."""
    if device.type == "cuda":
        vram = torch.cuda.get_device_properties(device).total_memory / 2**30
        if vram >= 16:
            return 32, 1024
        if vram >= 8:
            return 16, 768
        return 8, 512
    if device.type == "mps":
        return 16, 512
    return 8, 384


def _autocast(device: torch.device):
    """Mixed precision on CUDA (tensor cores on RTX cards); full precision elsewhere."""
    import contextlib
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


class _Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Compact 3-level U-Net with a residual output (predicts the correction)."""

    def __init__(self, ch=3, base=32, cin=None):
        super().__init__()
        self.ch = ch
        self.e1 = _Block(cin or ch, base)
        self.e2 = _Block(base, base * 2)
        self.e3 = _Block(base * 2, base * 4)
        self.bott = _Block(base * 4, base * 4)
        self.d3 = _Block(base * 8, base * 2)
        self.d2 = _Block(base * 4, base)
        self.d1 = _Block(base * 2, base)
        self.out = nn.Conv2d(base, ch, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        b = self.bott(F.avg_pool2d(e3, 2))
        d3 = self.d3(torch.cat([F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False), e1], 1))
        return x[:, :self.ch] + self.out(d1)


def widen_input(net: UNet, extra: int) -> UNet:
    """A copy of ``net`` taking ``extra`` more input channels, which start with zero weights (so it
    begins as exactly ``net``)."""
    w = UNet(ch=net.ch, base=net.out.in_channels, cin=net.e1.net[0].in_channels + extra)
    sd = {k: v.clone() for k, v in net.state_dict().items()}
    old = sd["e1.net.0.weight"]
    sd["e1.net.0.weight"] = torch.cat([old, torch.zeros(old.shape[0], extra, *old.shape[2:], dtype=old.dtype,
                                                        device=old.device)], 1)
    w.load_state_dict(sd)
    return w.to(next(net.parameters()).device)


class Stabiliser:
    """Invertible asinh variance-stabilising transform, per channel."""

    def __init__(self, a: np.ndarray, b: np.ndarray, k: float = 3.0):
        diff = (a - b)[::4, ::4]
        self.sigma = (1.4826 * np.median(np.abs(diff - np.median(diff, axis=(0, 1))), axis=(0, 1)) / np.sqrt(2)).astype(np.float32)
        self.sigma = np.maximum(self.sigma, 1e-6)
        self.bg = np.median(((a + b) / 2)[::4, ::4], axis=(0, 1)).astype(np.float32)
        self.k = k

    def fwd(self, x):
        return np.arcsinh((x - self.bg) / (self.k * self.sigma)).astype(np.float32)

    def inv(self, g):
        return (np.sinh(g) * self.k * self.sigma + self.bg).astype(np.float32)


def _to_t(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None]


N2N_LOSSES = ("asinh_mse", "asinh_unbiased", "lin_mse", "lin_chi2", "lin_huber")


def make_n2n_loss(kind: str, stab: "Stabiliser", device, delta: float = 3.0):
    """Noise2Noise training objectives for a network that predicts in the stabilised domain.

    Returns ``loss(pred, tgt, var)`` for stabilised-domain tensors ``pred``, ``tgt`` (B, C, H, W)
    and ``var``, the noise variance of ONE half-stack in linear units² (B, C, H, W; a local
    estimate independent of the pixel's own noise, e.g. the smoothed (A-B)²/2).  Writing
    s = k·sigma and x̂ = sinh(pred)·s + bg for the linear estimate:

    * ``asinh_mse``: MSE in the stabilised domain (production).  Its minimiser is
      E[asinh((B-bg)/s) | A], which differs from asinh((x-bg)/s) by the Jensen gap of the
      transform (second order: ½ g''(u) var/s²).
    * ``asinh_unbiased``: nonlinear Noise2Noise with the gap removed (cf. Tinits & Mann,
      arXiv:2512.24794): ``pred`` is read as the clean image and its *expected* stabilised
      noisy value E_n[asinh((x̂ + n - bg)/s)] under Gaussian noise of variance ``var``
      (Gauss-Hermite, 7 nodes) is matched to the target by MSE.  The fixed point is pred =
      asinh((x-bg)/s) exactly for Gaussian noise.
    * ``lin_mse``: MSE in linear units (the loss of ImageMM's N2N pass), scaled by mean(var).
    * ``lin_chi2``: inverse-variance weighted MSE in linear units (Ye et al., arXiv:2609.21350).
    * ``lin_huber``: Huber loss of the normalised linear residual with threshold ``delta``
      (quadratic within ±delta sigma, linear beyond: robust to cosmic rays and defects left in
      one half only).
    """
    if kind not in N2N_LOSSES:
        raise ValueError(f"unknown N2N loss {kind!r}, expected one of {N2N_LOSSES}")
    s = torch.tensor(stab.k * stab.sigma, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    bg = torch.tensor(stab.bg, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    cap = 60.0                                    # sinh(60) ~ 5.7e25: keeps float32 finite

    def lin(g):
        return torch.sinh(g.clamp(-cap, cap)) * s + bg

    if kind == "asinh_mse":
        def loss(pred, tgt, var=None):
            return F.mse_loss(pred, tgt)
        return loss

    if kind == "asinh_unbiased":
        t, w = np.polynomial.hermite.hermgauss(7)
        nodes = torch.tensor(t * np.sqrt(2.0), dtype=torch.float32, device=device)
        wts = torch.tensor(w / np.sqrt(np.pi), dtype=torch.float32, device=device)

        def loss(pred, tgt, var):
            u = torch.sinh(pred.clamp(-cap, cap))                  # clean estimate in units of s
            sd = torch.sqrt(var.clamp_min(1e-12)) / s              # noise sigma in the same units
            exp = 0
            for ti, wi in zip(nodes, wts):
                exp = exp + wi * torch.asinh(u + ti * sd)
            return F.mse_loss(exp, tgt)
        return loss

    if kind == "lin_mse":
        def loss(pred, tgt, var):
            return ((lin(pred) - lin(tgt)) ** 2).mean() / var.mean().clamp_min(1e-12)
        return loss

    if kind == "lin_chi2":
        def loss(pred, tgt, var):
            return ((lin(pred) - lin(tgt)) ** 2 / var.clamp_min(1e-12)).mean()
        return loss

    def loss(pred, tgt, var):                      # lin_huber
        r = (lin(pred) - lin(tgt)) / torch.sqrt(var.clamp_min(1e-12))
        a = r.abs()
        return torch.where(a <= delta, r * r, 2 * delta * a - delta * delta).mean()
    return loss


def train_n2n(ga: np.ndarray, gb: np.ndarray, iters: int = 2000, patch: int = 128, batch: int = 16,
              device=None, progress=None, cancel=None, seed: int = 0, sample_mask: np.ndarray | None = None,
              max_lr: float = 1e-3, base: int = 32, loss: str = "asinh_mse", stab: "Stabiliser | None" = None,
              var: np.ndarray | None = None, delta: float = 3.0) -> UNet:
    """Noise2Noise (Lehtinen et al. 2018) U-Net on the stabilised half-stacks; one-cycle
    schedule (Smith & Topin 2019) peaking at ``max_lr``; ``base`` = first-level channels.

    ``loss``: training objective (``make_n2n_loss``); every choice but the default needs the
    ``stab`` the halves were transformed with and ``var``, the per-pixel noise variance of
    one half in linear units² (h, w, 3), sampled with the same patches.  Ablation:
    experiments/exp_denoise_loss.py."""
    device = device or pick_device()
    if loss != "asinh_mse" and (stab is None or var is None):
        raise ValueError("train_n2n: this loss needs stab and var")
    loss_fn = make_n2n_loss(loss, stab, device, delta) if loss != "asinh_mse" else None
    tv = torch.from_numpy(np.ascontiguousarray(var.transpose(2, 0, 1))) if loss_fn else None
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = UNet(base=base).to(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    use_scaler = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=iters, pct_start=0.15)
    h, w, _ = ga.shape
    ta, tb = torch.from_numpy(ga.transpose(2, 0, 1)).contiguous(), torch.from_numpy(gb.transpose(2, 0, 1)).contiguous()
    # candidate patch corners: only where the patch is inside well-covered data
    if sample_mask is not None:
        m = sample_mask[: h - patch, : w - patch]
        from scipy.ndimage import minimum_filter
        ok = minimum_filter(sample_mask.astype(np.uint8), size=patch, origin=-(patch // 2))[: h - patch, : w - patch] > 0
        cand = np.argwhere(ok)
        if len(cand) < 100:
            cand = np.argwhere(np.ones_like(m, bool))
    else:
        cand = None
    t0 = time.time()
    for it in range(iters):
        if cancel and cancel():
            raise RuntimeError("cancelled")
        if cand is not None:
            pick = cand[rng.integers(0, len(cand), batch)]
            ys, xs = pick[:, 0], pick[:, 1]
        else:
            ys = rng.integers(0, h - patch, batch)
            xs = rng.integers(0, w - patch, batch)
        swap = rng.random(batch) < 0.5
        inp, tgt, aux = [], [], []
        for y, x, s in zip(ys, xs, swap):
            pa = ta[:, y:y + patch, x:x + patch]
            pb = tb[:, y:y + patch, x:x + patch]
            if s:
                pa, pb = pb, pa
            k = int(rng.integers(0, 4))
            pa, pb = torch.rot90(pa, k, (1, 2)), torch.rot90(pb, k, (1, 2))
            fl = rng.random() < 0.5
            if fl:
                pa, pb = pa.flip(2), pb.flip(2)
            inp.append(pa)
            tgt.append(pb)
            if tv is not None:
                pv = torch.rot90(tv[:, y:y + patch, x:x + patch], k, (1, 2))
                aux.append(pv.flip(2) if fl else pv)
        inp = torch.stack(inp).to(device)
        tgt = torch.stack(tgt).to(device)
        with _autocast(device):
            pred = net(inp)
        if loss_fn is None:
            loss = F.mse_loss(pred.float(), tgt)
        else:
            loss = loss_fn(pred.float(), tgt, torch.stack(aux).to(device))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if progress and (it % 25 == 0 or it == iters - 1):
            progress(it + 1, iters, f"Training Noise2Noise denoiser {it + 1}/{iters} (loss {loss.item():.4f}, {time.time() - t0:.0f}s)")
    return net.eval()


_D4 = [(0, False), (1, False), (2, False), (3, False), (0, True), (1, True), (2, True), (3, True)]


def _edge_weights(n: int, overlap: int, start_inner: bool, end_inner: bool) -> np.ndarray:
    """1-D blending weights for one tile side-by-side with its neighbours.

    Predictions within ~overlap/3 of an interior tile edge are unreliable (the
    network sees reflected padding instead of real sky), so they get weight 0;
    the rest of the overlap is a smooth ramp.  Sides on the true image border
    keep full weight."""
    w = np.ones(n, np.float32)
    dead = overlap // 3
    t = np.linspace(0, 1, overlap - dead, dtype=np.float32)
    ramp = np.concatenate([np.zeros(dead, np.float32), t * t * (3 - 2 * t)])
    if start_inner:
        w[:overlap] = ramp
    if end_inner:
        w[-overlap:] = np.minimum(w[-overlap:], ramp[::-1])
    return w


@torch.no_grad()
def infer(net: nn.Module, g: np.ndarray, tile: int = 512, overlap: int = 96, device=None, tta: int = 1) -> np.ndarray:
    """Tiled inference with feathered overlaps.

    ``tta`` averages the prediction over that many rotations/flips of the input
    (geometric self-ensemble, Timofte et al. 2016, arXiv:1511.02228): free
    extra noise reduction that also removes any orientation bias of the network.
    """
    device = device or next(net.parameters()).device
    h, w, c = g.shape
    tile = min(tile, max(h, w))
    overlap = min(overlap, tile // 3)
    out = None
    acc = np.zeros((h, w, 1), np.float32)
    step = tile - overlap
    tfs = _D4[:: 8 // tta] if tta in (1, 2, 4) else _D4
    ys = sorted({min(y, max(h - tile, 0)) for y in range(0, max(h - overlap, 1), step)})
    xs = sorted({min(x, max(w - tile, 0)) for x in range(0, max(w - overlap, 1), step)})
    for ya in ys:
        for xa in xs:
            y1, x1 = min(ya + tile, h), min(xa + tile, w)
            patch = g[ya:y1, xa:x1]
            ph, pw = patch.shape[:2]
            padh, padw = (-ph) % 8, (-pw) % 8
            t = _to_t(patch).to(device)
            if padh or padw:
                t = F.pad(t, (0, padw, 0, padh), mode="reflect")
            r = 0
            for k, fl in tfs:
                tt = torch.rot90(t, k, (2, 3))
                if fl:
                    tt = tt.flip(3)
                with _autocast(device):
                    y = net(tt).float()
                if fl:
                    y = y.flip(3)
                r = r + torch.rot90(y, -k, (2, 3))
            r = (r / len(tfs))[0, :, :ph, :pw].cpu().numpy().transpose(1, 2, 0)
            wgt = np.outer(_edge_weights(ph, overlap, ya > 0, y1 < h), _edge_weights(pw, overlap, xa > 0, x1 < w))[..., None]
            if out is None:
                out = np.zeros((h, w, r.shape[-1]), np.float32)
            out[ya:y1, xa:x1] += r * wgt
            acc[ya:y1, xa:x1] += wgt
    return out / np.maximum(acc, 1e-6)


def channel_psfs(img: np.ndarray, sat: float, max_half: int = 20) -> np.ndarray | None:
    """Per-channel empirical PSFs (refractors focus colours differently), padded to one size."""
    from .postprocess import estimate_psf
    psfs = []
    for c in range(img.shape[2]):
        p, _ = estimate_psf(img[..., c], sat, max_half=max_half)
        if p is None:
            return None
        psfs.append(p)
    n = max(p.shape[0] for p in psfs)
    return np.stack([np.pad(p, (n - p.shape[0]) // 2) for p in psfs]).astype(np.float32)


def channel_psf_field(img: np.ndarray, sat: float, max_half: int = 20, spacing: float = 900.0) -> dict | None:
    """Per-channel empirical PSFs over the field: ``channel_psfs`` in overlapping windows (twice
    the node spacing) around the nodes of a grid at most ``spacing`` px apart (exposures.psf_nodes),
    the whole-image PSF where a window has too few stars.  The Seestar's star FWHM changes by ~40 %
    from the centre to the edge of the field; with one PSF for the whole image the deconvolution
    network was trained to over-deconvolve the (sharper) centre - tiny stars with dark outlines on
    M 76 - and under-deconvolve the edges.
    Returns {"kernels": (C, ny, nx, k, k) - symmetrised, unit sum -, "nodes": (rows, cols)} or None."""
    from .exposures import psf_nodes
    glob = channel_psfs(img, sat, max_half)
    if glob is None:
        return None
    H, W = img.shape[:2]
    ys, xs = psf_nodes(H, W, spacing)
    sy, sx = (ys[1] - ys[0]) if len(ys) > 1 else H, (xs[1] - xs[0]) if len(xs) > 1 else W
    per = [[None] * len(xs) for _ in ys]
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            y0, y1 = int(max(0, y - sy)), int(min(H, y + sy + 1))
            x0, x1 = int(max(0, x - sx)), int(min(W, x + sx + 1))
            per[i][j] = channel_psfs(img[y0:y1, x0:x1], sat, max_half)
    k = max([glob.shape[1]] + [q.shape[1] for row in per for q in row if q is not None])
    C = img.shape[2]
    K = np.zeros((C, len(ys), len(xs), k, k), np.float32)
    for i in range(len(ys)):
        for j in range(len(xs)):
            q = per[i][j] if per[i][j] is not None else glob
            o = (k - q.shape[1]) // 2
            K[:, i, j, o:o + q.shape[1], o:o + q.shape[1]] = q
    return {"kernels": K, "nodes": (ys, xs), "n_fallback": int(sum(q is None for row in per for q in row))}


def stack_psf_field(img: np.ndarray, sat: float, valid: np.ndarray | None = None, px_scale: float = 1.0) -> dict | None:
    """Per-channel field PSF of a stack: ImageMM's PSFEx-style model (exposures.empirical_psf_field:
    crowded-field cut-outs, flux^2-weighted polynomial over the field, harmonic wings) on nodes
    ``PSF_NODE_SPACING`` (x the stack scale) apart.  Every channel is centred on the luminance
    centroid of the catalogue stars, so the kernels keep the lateral colour offsets and the
    asymmetry (coma, elongation) of the real stars.

    ``channel_psf_field`` symmetrised every kernel over the 8 rotations / flips and kept only
    round stars: deconvolved with it, the elongated, comatic stars of C 33 (DWARF 3, 2x drizzle,
    ellipticity 0.21) came out with ellipticity 0.47 and comet tails - the network removed the
    round part of the blur and left the asymmetric part, relatively amplified.
    Returns {"kernels": (C, ny, nx, k, k) unit sum, "nodes": (rows, cols), ...} or None."""
    from .exposures import PSF_NODE_SPACING, empirical_psf_field, psf_nodes, star_catalog
    H, W, C = img.shape
    bs = img - _sky_map(img, int(64 * px_scale))
    L = np.ascontiguousarray(bs.mean(-1), np.float32)
    import sep
    rms = sep.Background(L, bw=64, bh=64).globalrms
    objs = sep.extract(L, 10.0, err=rms, minarea=5)
    ok = (objs["flag"] == 0) & (objs["peak"] < 0.3 * sat)
    if ok.sum() < 10:
        return None
    o = objs[ok]
    fwhm = float(np.median(2 * sep.flux_radius(L, o["x"], o["y"], 6 * o["a"], 0.5, subpix=5)[0]))
    if not np.isfinite(fwhm) or fwhm <= 0:
        return None
    cat = star_catalog(bs, sat, fwhm)
    half = int(np.ceil(3.5 * fwhm))
    nodes = psf_nodes(H, W, PSF_NODE_SPACING * px_scale)
    valid = np.ones((H, W), bool) if valid is None else valid
    valid = valid & (img.max(-1) < 0.5 * sat)
    fields = [empirical_psf_field(bs[..., c], valid, cat, half, fwhm, nodes) for c in range(C)]
    if any(f is None for f in fields):
        return None
    # crop to the largest support radius (the kernels are zero beyond it)
    r = int(np.ceil(max(f["support"] for f in fields)))
    K = np.stack([f["nodes"] for f in fields])[..., half - r:half + r + 1, half - r:half + r + 1]
    K = K / K.sum((-2, -1), keepdims=True)
    return {"kernels": np.ascontiguousarray(K, np.float32), "nodes": nodes, "n_fallback": 0, "fwhm": fwhm,
            "n_stars": [int(f["n_stars"]) for f in fields], "deg": [int(f["deg"]) for f in fields]}


def gauss_kernel(fwhm: float, n: int, dy: float = 0.0, dx: float = 0.0) -> np.ndarray:
    """Round Gaussian of ``fwhm`` px sampled on an n x n grid, centred (dy, dx) px off the centre
    pixel, unit sum: the target PSF of the deconvolution network (``split_target``)."""
    s = fwhm / 2.3548
    o = np.arange(n) - n // 2
    g = np.exp(-0.5 * ((o[:, None] - dy) ** 2 + (o[None, :] - dx) ** 2) / s ** 2)
    return (g / g.sum()).astype(np.float32)


def moffat_target(fwhm: float, beta: float = 4.765) -> dict:
    """The deconvolution network's target PSF r: a round Moffat profile (Moffat 1969) of ``fwhm`` px
    with beta = 4.765, the profile of a star seen through Kolmogorov turbulence (Trujillo et al.
    2001, MNRAS 328, 977) - a natural star shape: a compact core and a soft halo.
    A Gaussian target (beta -> infinity) kept the core and dropped the halo: the stars became
    hard-edged discs falling straight to the sky, a black circle round each in a stretch.  The
    stack's own profile (beta 1.85 on C 33 and M 42) kept so much halo that the half-light diameter
    doubled.  Returns {"r", "p": profile at radii r (px), "scale": 1, "fwhm", "beta"}."""
    alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    r_ = np.linspace(0.0, 30.0 * fwhm, 6001)
    return {"r": r_, "p": (1.0 + (r_ / alpha) ** 2) ** (-beta), "scale": 1.0, "fwhm": float(fwhm), "beta": float(beta)}


def target_kernel(t, n: int, dy: float = 0.0, dx: float = 0.0, sub: int = 5) -> np.ndarray:
    """The target PSF ``t`` (``moffat_target`` dict, or a Gaussian FWHM) on an n x n grid, centred
    (dy, dx) px off the centre pixel, integrated over each pixel (sub x sub samples), unit sum."""
    if not isinstance(t, dict):
        return gauss_kernel(float(t), n, dy, dx)
    o = (np.arange(n * sub) + 0.5) / sub - 0.5 - n // 2
    Y, X = np.meshgrid(o - dy, o - dx, indexing="ij")
    v = np.interp(np.hypot(Y, X) * t["scale"], t["r"], t["p"], right=0.0)
    v = v.reshape(n, sub, n, sub).mean((1, 3))
    return (v / v.sum()).astype(np.float32)


def has_target(t) -> bool:
    """A target resolution is set: a ``moffat_target`` dict or a Gaussian FWHM > 0."""
    return isinstance(t, dict) or (t is not None and float(t) > 0)


def scale_target(t, factor: float):
    """The target ``t`` on a grid ``factor`` x coarser (a 2x stack's target on the subs' 1x grid: 0.5)."""
    if not isinstance(t, dict):
        return float(t) * factor
    return {**t, "scale": t["scale"] / factor, "fwhm": t["fwhm"] * factor}


def split_target(K: np.ndarray, fwhm, eps: float = 1e-4) -> tuple[np.ndarray, float]:
    """Kernels s with s * r = K for the round target r (``fwhm``: a ``moffat_target``, or the FWHM of a
    Gaussian, px) (Magain, Courbin & Sohy 1998, ApJ 494, 472: deconvolve to a finite, well-sampled
    resolution, not to points).

    The network then outputs r * sky and is fitted through s, so a point source comes out as r -
    round, with its sub-pixel position - instead of whatever the data constrain of a point.  The
    PSF is wider along its long axis, its transfer function falls faster there, and a deconvolution
    towards points (the network, or Richardson-Lucy with the same kernel) stops short along that
    axis: C 33's stars (ellipticity 0.21) came out with ellipticity 0.33 - 0.42, oriented as the PSF.
    s is the Tikhonov-regularised least-squares solution (Fourier division, kernel centres at index 0
    of a zero-padded grid).  It has negative lobes (it is an operator, not a PSF); a non-negative s
    (Richardson-Lucy) cannot reproduce the PSF's core once r shares the PSF's halo (1.5 % of the peak
    against 0.27 % here, ``eps`` = 1e-4).  K (..., n, n) with unit sums.  Returns (s (..., n, n) unit
    sum, relative rms of s * r - K)."""
    n = K.shape[-1]
    c = n // 2
    N = 1 << int(np.ceil(np.log2(2 * n)))
    dev = pick_device()
    dev = dev if dev.type == "cuda" else torch.device("cpu")      # (MPS has no float64)

    def place(z):                       # (..., n, n) centred -> (..., N, N) with the centre at index 0
        z = torch.as_tensor(np.asarray(z, np.float64), device=dev)
        big = torch.zeros(z.shape[:-2] + (N, N), dtype=torch.float64, device=dev)
        big[..., :n, :n] = z
        return torch.roll(big, (-c, -c), (-2, -1))
    take = lambda big: torch.roll(big, (c, c), (-2, -1))[..., :n, :n]
    flat = np.asarray(K, np.float64).reshape(-1, n, n)
    Rh = torch.fft.fft2(place(target_kernel(fwhm, n)))
    Kh = torch.fft.fft2(place(flat))
    s = take(torch.fft.ifft2(Kh * Rh.conj() / (Rh.abs() ** 2 + eps)).real)
    s = s / s.sum((-2, -1), keepdim=True)
    back = take(torch.fft.ifft2(torch.fft.fft2(place(s.cpu().numpy())) * Rh).real)
    kt = torch.as_tensor(flat, device=dev)
    err = float(torch.sqrt(((back - kt) ** 2).sum() / (kt ** 2).sum()))
    return s.cpu().numpy().reshape(K.shape).astype(np.float32), err


def photon_slope(a: np.ndarray, b: np.ndarray, full: np.ndarray, tile: int = 768, q: float = 25.0) -> np.ndarray:
    """The photon term c1 of ``variance_law`` (variance of one half per ADU above the sky), as the
    ``q``-th percentile of its fit over ``tile`` px tiles.  Over a whole field the half-stack
    difference also holds the halves' residual registration and seeing differences round stars, which
    grow towards the edges and pass for photon noise: on C 33 (2x drizzle) the slope was 0.2 - 0.6 in
    the middle of the field and 4 - 22 in its corners, and the whole-field fit (4.8 / 1.9 / 6.0) gave
    the simulated sources of the deconvolution network's source term ten times their photon noise -
    the network then shrank every star's flux by 15 - 30 %.  Photon noise is the same everywhere, so
    the cleanest tiles measure it."""
    H, W = full.shape[:2]
    vals = []
    for y in range(0, max(H - tile // 2, 1), tile):
        for x in range(0, max(W - tile // 2, 1), tile):
            sl = (slice(y, min(y + tile, H)), slice(x, min(x + tile, W)))
            vals.append(variance_law(a[sl], b[sl], full[sl])[1])
    return np.percentile(np.stack(vals), q, axis=0).astype(np.float32)


def variance_law(a: np.ndarray, b: np.ndarray, full: np.ndarray, nbins: int = 24):
    """Per-channel noise variance of one half-stack as a function of the signal above the sky,
    var = c0 + c1 * max(level - sky, 0), from (A-B)^2/2 against the full stack's level.

    c0 is the robust variance at the sky level; c1 the slope fitted to the binned robust
    variances between the sky and sky + 4 sigma (beyond that the half-stack difference is
    dominated by structure, registration and seeing differences between the halves, not by pixel
    noise).  Returns (c0, c1, sky), arrays of shape (C,)."""
    C = full.shape[-1]
    c0, c1, skys = np.zeros(C, np.float32), np.zeros(C, np.float32), np.zeros(C, np.float32)
    for c in range(C):
        lv, d2 = full[..., c].ravel()[::7], (0.5 * (a[..., c] - b[..., c]) ** 2).ravel()[::7]
        sky = float(np.median(lv))
        v0 = float(np.median(d2[np.abs(lv - sky) < 0.5 * np.sqrt(np.median(d2) / 0.4549)]) / 0.4549)
        sg = np.sqrt(v0)
        edges = np.linspace(sky, sky + 4 * sg, nbins + 1)
        xs, ys = [], []
        for i in range(nbins):
            m = (lv >= edges[i]) & (lv < edges[i + 1])
            if m.sum() > 200:
                xs.append(float(np.median(lv[m])) - sky)
                ys.append(float(np.median(d2[m])) / 0.4549 - v0)   # median of a chi²_1 variate = 0.4549 x variance
        slope = float(np.dot(xs, ys) / max(np.dot(xs, xs), 1e-9)) if xs else 0.0
        c0[c], c1[c], skys[c] = v0, max(slope, 0.0), sky
    return c0, c1, skys


def star_population(img: np.ndarray, sat: float, fwhm: float, px_scale: float = 1.0) -> dict | None:
    """Fluxes and colours of the stack's stars, for the simulated point sources of the
    deconvolution network's star term (``train_n2n_deconv``).  "fluxes": the luminance (channel
    mean, as the PSF catalogue) fluxes of every 3-sigma detection up to twice the brightest
    unsaturated star - the field's own star counts, which the simulated stars are drawn from.  Drawn
    log-uniformly instead, most simulated stars were bright (real counts rise steeply towards faint
    fluxes), and the wider the range the less the network learnt of faint stars: on C 33 their peaks
    fell to a third (peak / flux 0.047 -> 0.013 at 3 - 6 sigma).  "colours": per-channel aperture
    fluxes over the channel mean."""
    import sep
    from .exposures import star_catalog
    bs = img - _sky_map(img, int(64 * px_scale))
    cat = star_catalog(bs, sat, fwhm)
    if len(cat["x"]) < 10:
        return None
    rad = 1.5 * fwhm
    fl = np.stack([sep.sum_circle(np.ascontiguousarray(bs[..., c]), cat["x"], cat["y"], rad, subpix=5)[0]
                   for c in range(img.shape[2])], 1)
    ok = (fl > 0).all(1)
    col = fl[ok] / fl[ok].mean(1, keepdims=True)
    hi = 2.0 * float(np.max(cat["flux"]))
    fluxes = np.asarray(cat["all_flux"], np.float64)
    fluxes = fluxes[(fluxes > 0) & (fluxes <= hi)]
    if len(fluxes) < 20:
        fluxes = np.asarray(cat["flux"], np.float64)
    return {"flux": (float(fluxes.min()), hi), "fluxes": fluxes.astype(np.float32), "colours": col.astype(np.float32)}


def _sky_map(img: np.ndarray, box: int) -> np.ndarray:
    import sep
    return np.stack([sep.Background(np.ascontiguousarray(img[..., c], np.float32), bw=box, bh=box).back()
                     for c in range(img.shape[2])], -1).astype(np.float32)


def deconv_floor(sharp: np.ndarray, den: np.ndarray, size: int) -> np.ndarray:
    """Deconvolution may move a star's halo light into its core, but it must never make
    a pixel darker than the local sky unless the data itself is darker there (a dark
    lane).  The sky is a robust local median of the denoised image over ~3 PSF widths
    (stars barely affect a median).  This removes the faint dark disks the stretch
    would otherwise reveal around bright stars and tight groups, where the true PSF is
    wider than the median one used for training."""
    from scipy.ndimage import median_filter
    f = 4
    k = max(5, (3 * size) // f | 1)
    small = cv2.resize(den, (den.shape[1] // f, den.shape[0] // f), interpolation=cv2.INTER_AREA)
    sky = median_filter(small, size=(k, k, 1), mode="reflect")
    sky = cv2.resize(sky, (den.shape[1], den.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.maximum(sharp, np.minimum(den, sky)).astype(np.float32)


def finish_sharp(sharp: np.ndarray, den: np.ndarray, full: np.ndarray, sat: float, psf_size: int) -> np.ndarray:
    """Common last step of every deconvolution: the local-sky floor (``deconv_floor``),
    then saturated stars, which carry no shape information, get the denoised profile back."""
    sharp = deconv_floor(sharp, den, psf_size)
    unsat = (full.max(-1) < 0.5 * sat).astype(np.uint8)
    satm = cv2.dilate(1 - unsat, np.ones((2 * psf_size + 1,) * 2, np.uint8)).astype(np.float32)
    satm = cv2.GaussianBlur(satm, (0, 0), psf_size / 3)[..., None]
    return (sharp * (1 - satm) + den * satm).astype(np.float32)


def mf_data_term(x: torch.Tensor, tgts: list, mf: dict, tw: torch.Tensor, patch: int, device):
    """Multi-frame data term of ``train_n2n_deconv`` for a batch of linear stack-grid patches x
    (B, C, P, P): tgts[q] = (target set index, patch origin y, x) on the stack grid (multiples
    of s); tw (C, H, W): > 0 where the stack is not saturated.  Returns (sum of 2 rho_Huber
    over valid exposure pixels of all groups, number of those pixels)."""
    from .imagemm import stack_forward
    s_ = int(mf["s"])
    delta = float(mf.get("delta", 2.0))
    data = torch.zeros((), device=device)
    count = torch.zeros((), device=device)
    for q, (which, y, xo) in enumerate(tgts):
        T_ = mf["sets"][which]
        pred, e0 = stack_forward(x[q:q + 1], T_["kernels"], s_)                # (1, G, C, n, n)
        n = pred.shape[-1]
        lo = int(round(e0))
        ey, ex = y // s_ + lo, xo // s_ + lo
        sly = (slice(None), slice(ey, ey + n), slice(ex, ex + n))
        yt, vt, mt = (torch.from_numpy(np.ascontiguousarray(np.moveaxis(T_[k][sly], -1, 1))).to(device)
                      for k in ("y", "v", "m"))
        # an exposure pixel counts only if every stack pixel of its s x s block is unsaturated
        ok = tw[:, y:y + patch, xo:xo + patch][None].to(device).gt(0).float()
        ok = (F.avg_pool2d(ok, s_) > 0.999).float() if s_ > 1 else ok
        mt = mt * ok[0, :, lo:lo + n, lo:lo + n]
        zres = (yt - pred[0]) / vt.clamp_min(1e-30).sqrt()
        a = zres.abs()
        rho = torch.where(a <= delta, 0.5 * zres ** 2, delta * (a - 0.5 * delta))       # Huber, Eq. 15
        data = data + (2 * rho * mt).sum()
        count = count + mt.sum()
    return data, count


class _StarTerm:
    """Simulated-source supervision of the deconvolution network (``train_n2n_deconv``, ``stars``).

    Sources are points (stars) or, with probability ``stars["extended"]`` (default 0), extended: elliptical
    Gaussian blobs and thin curved filaments (Gaussian cross-section) from a fraction of the PSF
    width to several PSF widths, at surface brightnesses of 1 - 30 x the sky noise - nebular
    structure.  Trained on points alone the network learnt that every compact faint feature is a
    star: on C 33 it pulled nebular knots into points and the held-out stretched-domain error rose
    from 0.073 to 0.083 - 0.108.  (That rise was mostly the sky drift the low-frequency data term now
    pins; with it, extended sources did not change the nebulosity visibly but kept C 33's stars oval,
    ellipticity 0.08 -> 0.16, so they are off by default.)  Targets: the source convolved with r."""

    def __init__(self, stars: dict, stab: Stabiliser, Kpsf: np.ndarray, at, fwhm: float, nc: int, rng, device,
                 raw_input: bool = False):
        self.raw_input = raw_input
        if not has_target(fwhm):
            raise ValueError("the star term needs a target resolution (target_fwhm > 0)")
        self.target = fwhm
        fwhm = fwhm["fwhm"] if isinstance(fwhm, dict) else float(fwhm)
        self.den = stars["net"].to(device).eval()
        for q in self.den.parameters():
            q.requires_grad_(False)
        self.lo, self.hi = (float(v) for v in stars["flux"])
        self.fluxes = np.asarray(stars["fluxes"], np.float64) if stars.get("fluxes") is not None else None
        self.col = np.asarray(stars["colours"], np.float32)
        self.c1 = np.asarray(stars["c1"], np.float32)[:, None, None]
        self.c1t = torch.tensor(self.c1, device=device)[None]
        self.per, self.frac = int(stars.get("per_patch", 6)), float(stars.get("frac", 0.5))
        self.Kpsf, self.at, self.fwhm, self.nc, self.rng, self.dev = Kpsf, at, fwhm, nc, rng, device
        self.R = Kpsf.shape[-1] // 2
        # the target kernel out to the PSF radius: a profile target keeps the star's halo
        self.Rt = self.R if isinstance(self.target, dict) else int(np.ceil(3 * fwhm))
        self.p_ext = float(stars.get("extended", 0.0))
        self.model_err = float(stars.get("model_err", 0.0))
        self.sky_sd = float(np.sqrt(np.mean(stars.get("c0", [1.0]))))
        self.psf_sd = float(stars.get("psf_fwhm", 2.0 * fwhm)) / 2.3548
        # faintest simulated sources (peak / extended surface brightness, sky sigmas)
        self.min_snr_pt = float(stars.get("min_snr", 0.0))
        self.min_snr_ext = float(stars.get("min_snr_ext", 1.0))
        self.f_min = lambda K: self.min_snr_pt * self.sky_sd / max(float(K.mean(0).max()), 1e-12)
        self.sk = torch.tensor(stab.k * stab.sigma, device=device).view(1, -1, 1, 1)
        self.bg = torch.tensor(stab.bg, device=device).view(1, -1, 1, 1)

    @staticmethod
    def _paste(dst, src, cy, cx):
        """Add src (C, m, m), centred on integer (cy, cx), into dst (C, P, P), clipped to dst."""
        m, P = src.shape[-1] // 2, dst.shape[-1]
        y0, y1, x0, x1 = max(cy - m, 0), min(cy + m + 1, P), max(cx - m, 0), min(cx + m + 1, P)
        if y0 < y1 and x0 < x1:
            dst[:, y0:y1, x0:x1] += src[:, y0 - cy + m:y1 - cy + m, x0 - cx + m:x1 - cx + m]

    def _extended(self, K: np.ndarray, fy: float, fx: float):
        """One extended source centred (fy, fx) px off a pixel centre: (input contribution e * psf,
        target e * r), both (C, m, m)."""
        from scipy.signal import fftconvolve
        rng, sd = self.rng, self.psf_sd
        gs = []                                           # Gaussian components: (y, x, covariance 2x2)
        if rng.random() < 0.5:                            # elliptical blob
            # resolved along both axes: a blob narrower than the PSF along one axis looks, once blurred, like
            # an elongated star, and its "stay elongated" target made the network keep C 33's stars oval
            # (ellipticity 0.08 -> 0.15)
            s2 = sd * np.exp(rng.uniform(np.log(1.0), np.log(3.0)))
            s1 = s2 / rng.uniform(0.4, 1.0)
            th = rng.uniform(0, np.pi)
            Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            gs.append((0.0, 0.0, Rm @ np.diag([s1 ** 2, s2 ** 2]) @ Rm.T))
            ext = 3.5 * s1
        else:                                             # filament: quadratic Bezier, Gaussian cross-section
            L = sd * rng.uniform(6.0, 16.0)                  # many PSF widths long: unmistakably not a star
            w = sd * np.exp(rng.uniform(np.log(0.2), np.log(1.2)))
            th = rng.uniform(0, np.pi)
            p0 = np.array([np.sin(th), np.cos(th)]) * L / 2
            p1 = rng.normal(0, L / 4, 2)
            for t in np.linspace(0, 1, max(8, int(2 * L))):
                q = (1 - t) ** 2 * (-p0) + 2 * (1 - t) * t * p1 + t * t * p0
                gs.append((q[0], q[1], np.eye(2) * w ** 2))
            ext = L / 2 + np.abs(p1).max() / 2 + 3.5 * w
        m = int(np.ceil(ext + self.R + 3 * self.fwhm))
        o = np.arange(-m, m + 1, dtype=np.float64)
        Y, X = np.meshgrid(o - fy, o - fx, indexing="ij")

        def render(extra):
            img = np.zeros_like(Y)
            for (y0, x0, C) in gs:
                Ci = np.linalg.inv(C + extra * np.eye(2))
                dy, dx = Y - y0, X - x0
                img += np.exp(-0.5 * (Ci[0, 0] * dy * dy + 2 * Ci[0, 1] * dy * dx + Ci[1, 1] * dx * dx)) \
                    / (2 * np.pi * np.sqrt(np.linalg.det(C + extra * np.eye(2))))
            return img / len(gs)
        # truth sampled with a 0.3 px Gaussian (finer than any pixel structure), its own blur put into the target
        e = render(0.09)
        tgt = fftconvolve(e, target_kernel(self.target, 2 * self.Rt + 1).astype(np.float64), mode="same")
        col = np.exp(rng.normal(0, 0.5, self.nc))
        col = col / col.mean()
        a_ = np.stack([fftconvolve(e, K[c].astype(np.float64), mode="same") for c in range(self.nc)])
        # surface brightness: the blurred source's peak at 0.5 - 30 x the sky noise
        sb = self.sky_sd * np.exp(rng.uniform(np.log(self.min_snr_ext), np.log(30.0))) / max(float(a_.mean(0).max()), 1e-12)
        return ((a_ * col[:, None, None]) * sb).astype(np.float32), \
            (tgt[None] * col[:, None, None] * sb).astype(np.float32)

    def __call__(self, net, pick, ta, tb, tsig, P: int, augment: bool):
        from .exposures import fourier_shift
        rng = self.rng
        n = max(1, int(round(len(pick) * self.frac)))
        raw, inj, tgt, msk, sig = [], [], [], [], []
        yy, xx = np.mgrid[:P, :P]
        for y, x in pick[:n]:
            src = ta if rng.random() < 0.5 else tb
            K = self.at(self.Kpsf, y + P / 2, x + P / 2)
            S_ = np.zeros((self.nc, P, P), np.float32)
            T_ = np.zeros((self.nc, P, P), np.float32)
            M_ = np.zeros((P, P), bool)
            edge = max(int(np.ceil(3 * self.fwhm)), 6)
            for _ in range(self.per):
                cy, cx = rng.uniform(edge, P - 1 - edge, 2)
                iy, ix = int(round(cy)), int(round(cx))
                fy, fx = cy - iy, cx - ix
                if rng.random() < self.p_ext:
                    a_, t_ = self._extended(K, fy, fx)
                    self._paste(S_, a_, iy, ix)
                    self._paste(T_, t_, iy, ix)
                    m = a_.shape[-1] // 2
                    sup = np.zeros((P, P), bool)
                    lum = a_.mean(0) > 0.01 * a_.mean(0).max()
                    tmp = np.zeros((1, P, P), np.float32)
                    self._paste(tmp, lum[None].astype(np.float32), iy, ix)
                    M_ |= tmp[0] > 0
                    continue
                # stratified: half from the field's star counts (smeared down to half: below detection), so
                # faint stars are learnt as they occur; half log-uniform over the whole range, so the bright
                # stars - rare in the counts - are learnt too (from the counts alone C 33's stars came out
                # oval again: ellipticity 0.10 -> 0.16)
                if self.fluxes is not None and rng.random() < 0.5:
                    f0 = float(self.fluxes[rng.integers(len(self.fluxes))]) * 2.0 ** -rng.random()
                else:
                    f0 = np.exp(rng.uniform(np.log(max(self.lo, self.f_min(K))), np.log(self.hi)))
                f = f0 * self.col[rng.integers(len(self.col))]
                ks = np.stack([fourier_shift(K[c].astype(np.float64), fy, fx) for c in range(self.nc)])
                self._paste(S_, (ks * f[:, None, None]).astype(np.float32), iy, ix)
                g = target_kernel(self.target, 2 * self.Rt + 1, fy, fx)
                self._paste(T_, g[None] * f[:, None, None], iy, ix)
                M_ |= np.hypot(yy - cy, xx - cx) <= self.R
            S_ = S_ + rng.standard_normal(S_.shape).astype(np.float32) * np.sqrt(self.c1 * np.clip(S_, 0, None))
            raw.append(src[:, y:y + P, x:x + P])
            inj.append(torch.from_numpy(S_))
            tgt.append(torch.from_numpy(T_))
            msk.append(torch.from_numpy(M_))
            sig.append(tsig[:, y:y + P, x:x + P])
        raw, inj, tgt, sig = (torch.stack(z).to(self.dev) for z in (raw, inj, tgt, sig))
        msk = torch.stack(msk).to(self.dev)[:, None].float()
        fwd = lambda v: torch.asinh((v - self.bg) / self.sk)
        with torch.no_grad(), _autocast(self.dev):
            gin = torch.cat([fwd(raw), fwd(raw + inj)])
            d = self.den(gin).float()
        if self.raw_input:
            d = torch.cat([d, gin], 1)
        tf = [(int(rng.integers(0, 4)), bool(rng.random() < 0.5)) if augment else (0, False) for _ in range(n)]
        aug = lambda z, t: (torch.rot90(z, t[0], (1, 2)).flip(2) if t[1] else torch.rot90(z, t[0], (1, 2)))
        back = lambda z, t: torch.rot90(z.flip(2) if t[1] else z, -t[0], (1, 2))
        inp = torch.stack([aug(d[q], tf[q % n]) for q in range(2 * n)])
        with _autocast(self.dev):
            g = net(inp)
        g = torch.stack([back(g[q].float(), tf[q % n]) for q in range(2 * n)])
        x = torch.sinh(g) * self.sk + self.bg
        # chi^2 per pixel: the stack's noise plus the simulated star's own photon noise (a full stack
        # has half the variance of one half: c1 / 2 per ADU).  (In the stabilised domain, as the denoiser's
        # loss, the stars stayed oval: C 33 ellipticity 0.09 -> 0.29.)
        # optionally plus a systematic floor of ``model_err`` x the source (a chi^2 fit with model error):
        # 0.05 weighted faint stars up but lost 16 % of the stars' flux on C 33, so it is off by default
        var = sig ** 2 + 0.5 * self.c1t * tgt.clamp_min(0) + (self.model_err * tgt) ** 2
        # Huber (delta = 2, as ImageMM's Algorithm 3) on the normalised residual: as a square, a bright
        # simulated star's residual (hundreds of sigmas) outweighed every faint one, and faint stars stayed
        # broad (C 33: half-light FWHM 5.8 px at 3 - 6 sigma)
        z_ = (x[n:] - x[:n] - tgt) / var.sqrt()
        diff = torch.where(z_.abs() <= 2.0, z_ * z_, 4.0 * z_.abs() - 4.0)
        return (diff * msk).sum() / (msk.sum() * self.nc).clamp_min(1.0)


def train_n2n_deconv(net: nn.Module, da: np.ndarray, db: np.ndarray, a: np.ndarray, b: np.ndarray,
                     stab: Stabiliser, psfs: np.ndarray, var: np.ndarray, weight: np.ndarray, sky: np.ndarray,
                     iters: int = 2000, patch: int = 128, batch: int = 12, hessian: float = 0.3,
                     floor: float = 3.0, margin: float = 0.5, device=None, progress=None, cancel=None,
                     sample_mask: np.ndarray | None = None, seed: int = 0, mf: dict | None = None,
                     augment: bool = True, target_fwhm: float = 0.0, stars: dict | None = None,
                     lowfreq: float = 1.0, raw: tuple | None = None) -> nn.Module:
    """Self-supervised deconvolution network (see module docstring).

    da/db: denoised halves (stabilised domain) = network inputs; a/b: raw linear halves
    = targets; var: per-pixel noise variance of one half; weight: 0 where the data is
    saturated; sky: smooth per-channel sky level (linear).
    loss = chi2(psf * x, other half) + hessian * |Hessian(g(x))|^2 + floor * |undershoot below sky|^2

    ``mf``: ImageMM's multi-frame likelihood (arXiv:2501.03002, Eq. 14 with the Huber loss)
    as the data term instead of chi2 against the other half-stack.  mf = {"sets": [T_A, T_B],
    "s": stack scale, "delta": 2.0}: T_A is the target set for the half-A input (built from
    the odd subs, independent of half A), T_B for the half-B input (even subs); each is
    {"y", "v", "m": (G, H0, W0, C) seeing-group coadds on the exposure grid (exposures.
    ExposureSet.group_coadds), "kernels": (G, C, k, k) torch tensor on the stack grid}.
    The data term is 2 mean(rho_Huber((y_g - D H_g x) / sigma_g)) over valid pixels of all
    groups (= chi2 for small residuals), with D H_g from imagemm.stack_forward.
    Augmentation (rotations / flips) is applied to the network input only; its output is
    turned back before the forward model, so asymmetric PSFs are handled exactly.

    ``raw``: (stabilised half A, half B): extra input channels next to the denoised half (the network
    is then a ``widen_input`` UNet).  The denoiser's MMSE estimate smears faint stars (M 42: FWHM 6.5
    px against the stack's 4.7) and smooths faint nebulosity, detail no later stage can recover from
    the denoised half alone; with the raw half the data term decides where its detail is real.
    ``target_fwhm`` > 0: the output is the sky seen through a round Gaussian of that FWHM (px),
    fitted through the kernels s with s * r = psf (``split_target``, MCS 1998).
    ``stars``: simulated point-source supervision (with this stack's own PSF and noise): {"net": the
    trained denoiser, "flux": (lo, hi), "fluxes": the field's star fluxes (``star_population``; half
    the sources are drawn from them, half log-uniform over (lo, hi)), "colours": (M, C) colour
    vectors, "c0", "c1": (C,) sky variance and photon variance per ADU of one half
    (``variance_law``, ``photon_slope``), "psf_fwhm", "weight", "per_patch" (6)}.  Each step, point
    sources (the local PSF at a random sub-pixel position, with their photon noise) are added to
    the RAW halves of some patches, which go through the denoiser and the network with and
    without them; the difference of the two outputs must be flux x r at the source's position
    (within the PSF radius; Huber on the residual in units of the stack's noise plus the source's
    photon noise).  Light adds, so this is exactly what a star contributes to the output; the term
    teaches the network the round target profile along the axis where the data alone constrain it
    weakly, and that faint stars are points.  No
    simulated light reaches the output: the network is applied to the data only.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = net.to(device).train()
    k = torch.tensor(stab.k * stab.sigma, device=device).view(1, -1, 1, 1)
    bg = torch.tensor(stab.bg, device=device).view(1, -1, 1, 1)
    inv = lambda g: torch.sinh(g) * k + bg
    T = lambda z: torch.from_numpy(np.ascontiguousarray(z.transpose(2, 0, 1), dtype=np.float32))
    tda, tdb, ta, tb = T(da), T(db), T(a), T(b)
    if raw is not None:                                 # the stabilised raw halves as extra input channels
        tda, tdb = torch.cat([tda, T(raw[0])]), torch.cat([tdb, T(raw[1])])
    tw = T(weight[..., None] / var)
    tsky, tsig = T(sky), T(np.sqrt(var / 2))
    # psfs: (C, k, k), or a field {"kernels": (C, ny, nx, k, k), "nodes"} (stack_psf_field): each
    # patch is then blurred with the PSF where it lies (bilinear between the nodes, at its centre).
    # conv2d is a cross-correlation: the kernels are flipped so that it evaluates the true convolution
    # k * x (as imagemm.stack_forward).  Unflipped, an asymmetric PSF is applied rotated by 180 degrees
    # and the network puts each star's core off-centre with its halo light beside it as a plateau
    field = psfs if isinstance(psfs, dict) else None
    from .exposures import interp_nodes
    if field is None:                                   # one PSF for the whole image: a 1 x 1 node grid
        field = {"kernels": np.asarray(psfs)[:, None, None], "nodes": (np.zeros(1), np.zeros(1))}
    Kpsf = field["kernels"]
    Kf = split_target(Kpsf, target_fwhm)[0] if has_target(target_fwhm) else Kpsf
    nc, r = Kf.shape[0], Kf.shape[-1] // 2
    single = len(field["nodes"][0]) == 1 and len(field["nodes"][1]) == 1
    at = lambda K, y, x: K[:, 0, 0] if single else interp_nodes(K, field["nodes"], y, x)
    kernel_at = lambda y, x: torch.from_numpy(np.ascontiguousarray(at(Kf, y, x)[..., ::-1, ::-1], np.float32))
    if stars is not None:
        star_term = _StarTerm(stars, stab, Kpsf, at, target_fwhm, nc, rng, device, raw_input=raw is not None)
    if mf is not None:
        s_ = int(mf["s"])
        patch -= patch % s_
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    h, w, _ = da.shape
    cand = None
    if sample_mask is not None:
        from scipy.ndimage import minimum_filter
        ok = minimum_filter(sample_mask.astype(np.uint8), size=patch, origin=-(patch // 2))[: h - patch, : w - patch] > 0
        cand = np.argwhere(ok)
        if len(cand) < 100:
            cand = None
    t0 = time.time()
    for it in range(iters):
        if cancel and cancel():
            raise RuntimeError("cancelled")
        if cand is not None:
            pick = cand[rng.integers(0, len(cand), batch)]
        else:
            pick = np.stack([rng.integers(0, h - patch, batch), rng.integers(0, w - patch, batch)], 1)
        if mf is not None:
            pick = pick - pick % s_                    # patch origins on the exposure grid's blocks
            inp, xs_, tgts = [], [], []
            for (y, x), sw in zip(pick, rng.random(batch) < 0.5):
                sl = (slice(None), slice(y, y + patch), slice(x, x + patch))
                src = tdb if sw else tda
                rot, fl = int(rng.integers(0, 4)), bool(rng.random() < 0.5)
                z = torch.rot90(src[sl], rot, (1, 2))
                inp.append(z.flip(2) if fl else z)
                xs_.append((rot, fl))
                tgts.append((1 if sw else 0, y, x))
            inp = torch.stack(inp).to(device)
            with _autocast(device):
                g = net(inp)
            g = g.float()
            # back to the sky's orientation before the forward model (PSFs are not symmetric)
            gb = []
            for q, (rot, fl) in enumerate(xs_):
                z = g[q]
                z = z.flip(2) if fl else z
                gb.append(torch.rot90(z, -rot, (1, 2)))
            g = torch.stack(gb)
            x = inv(g)
            data, count = mf_data_term(x, tgts, mf, tw, patch, device)
            chi2 = data / count.clamp_min(1.0)
        else:
            parts = [[] for _ in range(5)]
            kers, xs_ = [], []
            for (y, x), s in zip(pick, rng.random(batch) < 0.5):
                sl = (slice(None), slice(y, y + patch), slice(x, x + patch))
                src = (tdb, ta) if s else (tda, tb)
                # rotations / flips of the network input only: its output is turned back before the
                # forward model, so asymmetric PSFs (stack_psf_field) are handled exactly
                rot, fl = (int(rng.integers(0, 4)), bool(rng.random() < 0.5)) if augment else (0, False)
                z = torch.rot90(src[0][sl], rot, (1, 2))
                parts[0].append(z.flip(2) if fl else z)
                for lst, z in zip(parts[1:], (src[1][sl], tw[sl], tsky[sl], tsig[sl])):
                    lst.append(z)
                xs_.append((rot, fl))
                kers.append(kernel_at(y + patch / 2, x + patch / 2))
            inp, tgt, wt, skyp, sigp = (torch.stack(z).to(device) for z in parts)
            with _autocast(device):
                g = net(inp)
            g = g.float()
            gb = []
            for q, (rot, fl) in enumerate(xs_):
                z = g[q]
                z = z.flip(2) if fl else z
                gb.append(torch.rot90(z, -rot, (1, 2)))
            g = torch.stack(gb)
            x = inv(g)
            xp = F.pad(x, (r, r, r, r), mode="reflect")
            B_ = xp.shape[0]                             # one kernel per sample and channel: grouped conv
            kw = torch.stack(kers).to(device).reshape(B_ * nc, 1, 2 * r + 1, 2 * r + 1)
            kx = F.conv2d(xp.reshape(1, B_ * nc, *xp.shape[-2:]), kw, groups=B_ * nc).reshape(B_, nc, *x.shape[-2:])
            sl = (slice(None), slice(None), slice(r, -r), slice(r, -r))
            chi2 = ((kx - tgt) ** 2 * wt)[sl].mean() / (wt[sl] > 0).float().mean().clamp_min(1e-3)
            if lowfreq > 0:
                # the same data term on 16 px block means, against the noise of a block mean: the per-pixel
                # chi^2 hardly sees a uniform offset (0.05 sigma adds 0.0025 per pixel), and the network's
                # sky drifted by that much (C 33 held-out: +0.03 - 0.1 sigma); per block it adds 0.6
                ps = 16
                m = (wt[sl] > 0).float()
                cnt = F.avg_pool2d(m, ps)
                mean_d = F.avg_pool2d((kx - tgt)[sl] * m, ps) / cnt.clamp_min(1e-3)
                var_b = F.avg_pool2d(m / wt[sl].clamp_min(1e-30), ps) / cnt.clamp_min(1e-3) / (ps * ps * cnt.clamp_min(1e-3))
                # blocks with a >= 5 sigma source are left out: there the residual is the PSF model's
                # error, and fitting it built dark / bright rings round the stars (6 - 10 px, C 33)
                src_ = F.max_pool2d((((tgt - skyp) / sigp)[sl]).amax(1, keepdim=True), ps) > 5.0
                okb = ((cnt > 0.5) & ~src_).float()
                # Huber (delta = 2, as ImageMM's Algorithm 3): next to bright stars a block mean is dominated
                # by the PSF model's error, not by noise, and as a square it outweighed everything else
                # (held-out linear error 0.11 -> 0.23 without the star term)
                zb = mean_d / var_b.clamp_min(1e-30).sqrt()
                rho = torch.where(zb.abs() <= 2.0, zb * zb, 4.0 * zb.abs() - 4.0)
                chi2_lf = (rho * okb).sum() / okb.sum().clamp_min(1.0)
        if mf is not None:
            skyp = torch.stack([tsky[:, y:y + patch, xo:xo + patch] for (_, y, xo) in tgts]).to(device)
            sigp = torch.stack([tsig[:, y:y + patch, xo:xo + patch] for (_, y, xo) in tgts]).to(device)
        dxx = g[..., :, 2:] - 2 * g[..., :, 1:-1] + g[..., :, :-2]
        dyy = g[..., 2:, :] - 2 * g[..., 1:-1, :] + g[..., :-2, :]
        dxy = g[..., 1:, 1:] - g[..., 1:, :-1] - g[..., :-1, 1:] + g[..., :-1, :-1]
        hess = dxx.pow(2).mean() + dyy.pow(2).mean() + 2 * dxy.pow(2).mean()
        under = F.relu(skyp - margin * sigp - x) / sigp
        loss = chi2 + hessian * hess + floor * under.pow(2).mean()
        if lowfreq > 0 and mf is None:
            loss = loss + lowfreq * chi2_lf
        sterm = None
        if stars is not None and mf is None:
            sterm = star_term(net, pick, ta, tb, tsig, patch, augment)
            loss = loss + float(stars.get("weight", 1.0)) * sterm
        if not torch.isfinite(loss):
            raise RuntimeError(f"deconvolution network: non-finite loss at step {it} (data {float(chi2):.4g}, "
                               f"hessian {float(hess):.4g}, network output g in [{float(g.min()):.3g}, "
                               f"{float(g.max()):.3g}], x in [{float(x.min()):.3g}, {float(x.max()):.3g}])")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if progress and (it % 25 == 0 or it == iters - 1):
            progress(it + 1, iters, f"Training deconvolution network {it + 1}/{iters} "
                                    f"(chi2 {chi2.item():.3f}" + (f", stars {sterm.item():.3f}" if sterm is not None else "")
                                    + f", {time.time() - t0:.0f}s)")
    return net.eval()


def auto_target_fwhm(px_scale: float = 1.0) -> float:
    """Default target resolution of the deconvolution network (FWHM of the Moffat target, stack px):
    1.25 px of the subs' own grid, at least 2 stack px (the Nyquist sampling MCS 1998 ask of the
    target).  On the 2x drizzled stacks (C 33, M 42) 2.5 px gave compact stars without ringing; 3 px
    left them a third larger.  (A
    2.5 px Gaussian target rang - its truncated tails ask for frequencies above the subs' Nyquist
    limit: -0.4 sigma at 6 px, +0.2 at 10 px.)"""
    return max(2.0, 1.25 * float(px_scale))


def deconv_setup(full: np.ndarray, a: np.ndarray, b: np.ndarray, sat: float, px_scale: float = 1.0,
                 valid: np.ndarray | None = None, target_fwhm: float = 0.0, sources: float = 1.0,
                 den_net: nn.Module | None = None) -> tuple[dict | None, float, dict | None]:
    """What the deconvolution network needs from a stack: its field PSF (``stack_psf_field``), the
    target resolution (``moffat_target`` of ``target_fwhm`` px; 0 = ``auto_target_fwhm``, < 0 = none,
    towards points; returned as 0.0 then)
    and, with ``sources`` > 0 and the trained denoiser ``den_net``, the simulated-source term's
    population (``star_population``, ``variance_law``; ``train_n2n_deconv``'s ``stars``)."""
    import copy
    psfs = stack_psf_field(full, sat, valid, px_scale)
    if psfs is None:
        return None, 0.0, None
    tf = auto_target_fwhm(px_scale) if not target_fwhm else max(float(target_fwhm), 0.0)
    target = moffat_target(tf) if tf > 0 else 0.0
    stars = None
    if sources > 0 and tf > 0 and den_net is not None:
        pop = star_population(full, sat, psfs["fwhm"], px_scale)
        if pop is not None:
            c0 = variance_law(a, b, full)[0]
            c1 = photon_slope(a, b, full, tile=int(384 * px_scale))
            stars = {"net": copy.deepcopy(den_net), **pop, "c0": c0, "c1": c1, "psf_fwhm": psfs["fwhm"],
                     "weight": float(sources)}
    return psfs, target, stars


def n2n_restore(half_a: np.ndarray, half_b: np.ndarray, full: np.ndarray | None = None,
                iters: int = 2000, device: str = "auto", progress=None, cancel=None,
                coverage: np.ndarray | None = None, deconvolve: bool = True,
                deconv_iters: int | None = None, sat: float = 63471.0,
                px_scale: float = 1.0, save_path: str | None = None,
                mf: dict | None = None, loss: str = "asinh_mse", target_fwhm: float = 0.0,
                sources: float = 1.0) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Train on the half-stacks; return (denoised, deconvolved-or-None, info), all linear.

    ``target_fwhm``, ``sources``: the deconvolution network's target resolution and simulated-source
    term (``deconv_setup``); the multi-frame data term (``mf``) deconvolves towards points, without them.

    ``mf``: the ImageMM multi-frame data term for the deconvolution network (see
    ``train_n2n_deconv``); its kernels may be numpy arrays (moved to the device here).
    ``loss``: the denoiser's training objective (``make_n2n_loss``; ablation in
    experiments/README.md).

    High-SNR pixels (bright star cores, > ~80 sigma) are rare in training data
    and noise there is invisible, so the denoiser smoothly hands them back to the
    original stack – photometry of stars is preserved exactly.  Saturated stars
    carry no shape information, so the deconvolved image hands those back to the
    denoised one.  ``save_path``: keep the trained networks (a few MB) for re-use.
    """
    import copy
    info = {}
    mask = None
    if coverage is not None:
        mask = coverage >= 0.5 * np.percentile(coverage[coverage > 0], 90)
    if mask is not None and mask.mean() > 0.05:
        ys, xs = np.nonzero(mask[::4, ::4])
        y0, y1, x0, x1 = ys.min() * 4, ys.max() * 4, xs.min() * 4, xs.max() * 4
        stab = Stabiliser(half_a[y0:y1, x0:x1], half_b[y0:y1, x0:x1])
    else:
        mask = None
        stab = Stabiliser(half_a, half_b)
    ga, gb = stab.fwd(half_a), stab.fwd(half_b)
    dev = pick_device(device)
    batch, tile = _batch_and_tile(dev)
    tta = 8 if dev.type != "cpu" else 2
    if dev.type == "cpu":
        iters = min(iters, 600)
    # per-pixel noise variance of one half (also the deconvolution network's chi² weights)
    var = cv2.GaussianBlur(0.5 * (half_a - half_b) ** 2, (0, 0), 10 * px_scale)
    var = np.maximum(var, np.percentile(var[::4, ::4], 1, axis=(0, 1)) * 0.5).astype(np.float32)
    info["n2n_loss"] = loss
    net = train_n2n(ga, gb, iters=iters, batch=batch, device=dev, progress=progress, cancel=cancel,
                    sample_mask=mask, loss=loss, stab=stab, var=var)
    if progress:
        progress(0, 2, "Denoising half-stack A")
    da = infer(net, ga, tile=tile, tta=tta)
    if progress:
        progress(1, 2, "Denoising half-stack B")
    db = infer(net, gb, tile=tile, tta=tta)
    # ensemble of both independent reconstructions (removes residual noise further)
    g_den = 0.5 * (da + db)
    den = stab.inv(g_den)
    if full is None:
        full = 0.5 * (half_a + half_b)
    from scipy.ndimage import maximum_filter
    g_full = np.abs(stab.fwd(full)).max(axis=2, keepdims=True)
    g_full = maximum_filter(g_full, size=(5, 5, 1))
    keep = np.clip((g_full - 3.5) / 1.5, 0, 1)
    del g_full
    den = (den * (1 - keep) + full * keep).astype(np.float32)
    del keep
    sharp = psfs = dnet = None
    if deconvolve:
        if mf is not None:
            target_fwhm, sources = -1.0, 0.0
        psfs, tf, stars = deconv_setup(full, half_a, half_b, sat, px_scale, mask, target_fwhm, sources, net)
        if psfs is None:
            info["deconvolution"] = "skipped: not enough isolated stars to measure the PSF"
        else:
            ksize = int(psfs["kernels"].shape[-1])
            info["psf_size"] = ksize
            info["psf_field"] = {"nodes": [len(psfs["nodes"][0]), len(psfs["nodes"][1])], "fwhm": psfs["fwhm"],
                                 "stars": psfs["n_stars"], "deg": psfs["deg"]}
            info["target_fwhm"] = tf["fwhm"] if isinstance(tf, dict) else 0.0
            info["target"] = f"Moffat beta {tf['beta']}" if isinstance(tf, dict) else "none"
            info["sources"] = float(sources) if stars is not None else 0.0
            unsat = (full.max(-1) < 0.5 * sat).astype(np.uint8)
            weight = cv2.erode(unsat, np.ones((9, 9), np.uint8)).astype(np.float32)
            sky = _sky_map(full, int(64 * px_scale))
            if mf is not None:
                for T_ in mf["sets"]:
                    T_["kernels"] = torch.as_tensor(T_["kernels"], dtype=torch.float32, device=dev)
                info["multiframe"] = {"groups": len(mf["sets"][0]["kernels"]), "s": mf["s"]}
            dnet = train_n2n_deconv(widen_input(copy.deepcopy(net), ga.shape[-1]), da, db, half_a, half_b, stab, psfs,
                                    var, weight, sky, raw=(ga, gb),
                                    iters=deconv_iters or (iters if dev.type != "cpu" else min(iters, 400)),
                                    batch=max(4, batch * 3 // 4), device=dev, progress=progress,
                                    cancel=cancel, sample_mask=mask, mf=mf, target_fwhm=tf, stars=stars,
                                    lowfreq=1.0 if stars is not None else 0.0)
            del sky, stars
            if progress:
                progress(0, 1, "Deconvolving")
            ov = int(min(tile // 3, max(128, 4 * ksize)))
            # each half through the network with its own raw data, the two estimates averaged (as the denoiser)
            sharp = stab.inv(0.5 * (infer(dnet, np.concatenate([da, ga], -1), tile=max(tile, 3 * ov), overlap=ov, tta=tta)
                                    + infer(dnet, np.concatenate([db, gb], -1), tile=max(tile, 3 * ov), overlap=ov, tta=tta)))
            del var, da, db
            sharp = finish_sharp(sharp, den, full, sat, ksize)
    if save_path:
        torch.save({"denoiser": net.state_dict(), "deconv": dnet.state_dict() if dnet is not None else None,
                    "stab": {"sigma": stab.sigma, "bg": stab.bg, "k": stab.k},
                    "psfs": psfs}, save_path)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    elif dev.type == "mps":
        torch.mps.empty_cache()
    return den, sharp, info


def n2n_denoise(half_a: np.ndarray, half_b: np.ndarray, full: np.ndarray | None = None,
                iters: int = 2000, device: str = "auto", progress=None, cancel=None,
                coverage: np.ndarray | None = None) -> np.ndarray:
    """Denoise only (backwards-compatible wrapper around :func:`n2n_restore`)."""
    return n2n_restore(half_a, half_b, full, iters, device, progress, cancel, coverage, deconvolve=False)[0]
