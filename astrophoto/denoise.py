"""The U-Net and tiled inference ImageMM's Noise2Noise pass uses (Lehtinen et al. 2018)."""
from __future__ import annotations

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



def _to_t(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None]




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

