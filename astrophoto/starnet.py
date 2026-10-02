"""AI star removal, trained per dataset (after StarNet, which Siril uses for its starless
workflow: https://siril.readthedocs.io/en/stable/processing/stars/starnet.html).

StarNet is trained once on other people's images.  Like the Noise2Noise denoiser, this network
needs no pretrained weights: its training pairs are made from the image itself.

* Background: the dataset's own classic starless image (photometric star mask + push-pull
  inpainting, ``postprocess.classic_star_separation``) - its nebulae, galaxies, residual
  gradient, noise and grain.
* Stars: rendered onto random crops of it with the image's own star profile per colour channel,
  measured by stacking its stars out to several FWHM - so a refractor's wider blue halo, or the
  dark ring a restoration leaves round a star, is part of what the network learns to remove -
  with the flux and colour distribution of the stars detected in the image, from below the
  detection limit to saturated (clipped, neutral) cores at the level the image's own sit at;
  and the image's own brightest stars, cut from its classic star layer and pasted elsewhere
  (rotated, flipped), so that whatever shape its bright and saturated stars have is learnt too.
* Input = background + stars, target = background, in a variance-stabilised (asinh) domain;
  the U-Net of denoise.py predicts the correction.

The network so learns what a star looks like through this camera and optics, and removes the
real stars where masking does badly: stars on nebulosity (where inpainting smears the nebula),
halos wider than any mask, and faint stars below the detection limit.  At display time the star
layer is the image minus the network's starless image; the "Star reduction" slider fades it out
completely at 1 (``postprocess.nonlinear_stage``).
"""
from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np
import sep
import torch

from .denoise import UNet, _autocast, _batch_and_tile, infer, pick_device
from .postprocess import _extract, luminance, mad_sigma

DOMAIN_K = 3.0          # asinh softening in units of the per-pixel noise
DOMAIN_SCALE = 4.0      # stabilised values divided by this (stars at the white level ~2)


class Domain:
    """Invertible asinh stabilisation around the sky level: noise ~ unit slope, stars compressed."""

    def __init__(self, bg: np.ndarray, sigma: np.ndarray):
        self.bg = np.asarray(bg, np.float32)
        self.sigma = np.maximum(np.asarray(sigma, np.float32), 1e-7)

    def fwd(self, x):
        return (np.arcsinh((x - self.bg) / (DOMAIN_K * self.sigma)) / DOMAIN_SCALE).astype(np.float32)

    def inv(self, g):
        return (np.sinh(g * DOMAIN_SCALE) * DOMAIN_K * self.sigma + self.bg).astype(np.float32)

    def fwd_t(self, x: torch.Tensor) -> torch.Tensor:
        bg = torch.as_tensor(self.bg, device=x.device).view(1, -1, 1, 1)
        s = torch.as_tensor(self.sigma, device=x.device).view(1, -1, 1, 1)
        return torch.asinh((x - bg) / (DOMAIN_K * s)) / DOMAIN_SCALE

    def to_dict(self):
        return {"bg": self.bg.tolist(), "sigma": self.sigma.tolist()}


def domain_for(lin: np.ndarray, noise: float = 0.0) -> Domain:
    """Sky level and per-pixel noise per channel (never below ``noise``, the data's real noise: a
    restoration's or a denoised image's own sky can be nearly flat)."""
    bg = np.median(lin[::4, ::4].reshape(-1, 3), 0)
    sig = [max(mad_sigma((lin[..., c] - cv2.GaussianBlur(np.ascontiguousarray(lin[..., c]), (0, 0), 1.5))[::3, ::3]),
               float(noise)) for c in range(3)]
    return Domain(bg, sig)


# ============================================================ what the stars look like
def measured_kernel(ch: np.ndarray, objs, fw: float, R: int, sat_level: float) -> np.ndarray | None:
    """The star profile of one colour channel as it is in the image, per unit core flux, out to
    radius R: the median of flux-normalised, sub-pixel re-centred cut-outs of unsaturated, isolated
    stars, background taken from the outermost ring.  Unlike a model it keeps whatever the optics
    and processing put round a star: a refractor's wide halo, a restoration's dark ring.  Beyond
    2 FWHM it is replaced by its azimuthal average (the halo is faint; averaging beats the noise)."""
    h, w = ch.shape
    d_nn = objs["nn"]
    ok = ((objs["peak"] < 0.5 * sat_level) & (objs["x"] > R + 2) & (objs["y"] > R + 2) & (objs["x"] < w - R - 3) &
          (objs["y"] < h - R - 3))
    for iso in (1.0, 0.6, 0.4):
        sel = np.nonzero(ok & (d_nn > iso * R))[0]
        if len(sel) >= 25:
            break
    if len(sel) < 8:
        return None
    sel = sel[np.argsort(-objs["flux"][sel])][:300]
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    rad = np.hypot(xx, yy)
    ring = rad >= R - 3
    core = rad <= max(2.0 * fw, 2.0)
    cuts = []
    for i in sel:
        x, y = float(objs["x"][i]), float(objs["y"][i])
        xi, yi = int(round(x)), int(round(y))
        c = ch[yi - R - 2:yi + R + 3, xi - R - 2:xi + R + 3].astype(np.float32)
        M = np.float32([[1, 0, xi - x], [0, 1, yi - y]])
        c = cv2.warpAffine(c, M, (c.shape[1], c.shape[0]), flags=cv2.INTER_CUBIC)[2:-2, 2:-2]
        c = c - np.median(c[ring])
        f = c[core].sum()
        if f > 0:
            cuts.append(c / f)
    if len(cuts) < 8:
        return None
    k = np.median(np.stack(cuts), 0)
    ri = np.round(rad).astype(int)
    prof = np.bincount(ri.ravel(), k.ravel()) / np.maximum(np.bincount(ri.ravel()), 1)
    outer = rad > 2.0 * fw
    k[outer] = prof[ri[outer]]
    k[rad > R] = 0.0
    return k.astype(np.float32)


def star_model(lin: np.ndarray, noise: float = 0.0) -> dict:
    """The image's stars, measured: the star profile of each channel, the flux and colour of every
    detected star, their surface density, and the level saturated cores sit at."""
    from scipy.spatial import cKDTree
    L = np.ascontiguousarray(luminance(lin))
    bkg = sep.Background(L, bw=64, bh=64)
    sub = L - bkg.back()
    objs = _extract(sub, 5.0, max(bkg.globalrms, noise), minarea=3)
    fwhm = 2 * sep.flux_radius(sub, objs["x"], objs["y"], 6 * objs["a"], 0.5, subpix=5)[0]
    round_ = (objs["flag"] == 0) & (objs["a"] / np.maximum(objs["b"], 1e-3) < 1.5)
    fw = float(np.nanmedian(fwhm[round_])) if round_.any() else 3.0
    xy = np.stack([objs["x"], objs["y"]], 1)
    nn = cKDTree(xy).query(xy, k=2)[0][:, 1] if len(xy) > 1 else np.full(len(xy), 1e9)
    # the level clipped cores sit at: the brightest stars' peaks (the linear stage renders anything
    # clipped neutral at its max channel; a restoration's white level is its own maximum)
    peaks = lin.max(-1)[np.clip(objs["y"].astype(int), 0, lin.shape[0] - 1), np.clip(objs["x"].astype(int), 0, lin.shape[1] - 1)]
    sat_level = float(np.median(np.sort(peaks)[-10:])) if len(peaks) >= 10 else 1.0
    R = int(np.clip(round(6 * fw), 12, 48))
    o = {"x": objs["x"], "y": objs["y"], "peak": objs["peak"] + float(np.median(bkg.back())), "flux": objs["flux"],
         "nn": nn}
    kernels = []
    for c in range(3):
        ch = np.ascontiguousarray(lin[..., c])
        kernels.append(measured_kernel(ch - sep.Background(ch, bw=64, bh=64).back(), o, fw, R, sat_level))
    if all(k_ is None for k_ in kernels):
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        g = np.exp(-(xx ** 2 + yy ** 2) / (2 * (fw / 2.3548) ** 2)).astype(np.float32)
        kernels = [g / g.sum()] * 3
    kernels = [k_ if k_ is not None else next(q for q in kernels if q is not None) for k_ in kernels]
    fl = []
    for c in range(3):
        ch = np.ascontiguousarray(lin[..., c])
        s_ = ch - sep.Background(ch, bw=64, bh=64).back()
        fl.append(sep.sum_circle(s_, objs["x"], objs["y"], max(2.0 * fw, 2.0))[0])
    fl = np.stack(fl, 1)
    good = round_ & (fl > 0).all(1)
    lum = fl[good] @ np.array([0.2126, 0.7152, 0.0722])
    col = fl[good] / np.maximum(lum, 1e-12)[:, None]
    h, w = L.shape
    return {"kernels": np.stack(kernels), "fwhm": fw, "flux": lum.astype(np.float32), "colour": col.astype(np.float32),
            "density": float(len(objs) / (h * w)), "n_detected": int(good.sum()), "sat_level": sat_level,
            "flux_floor": float(3 * max(bkg.globalrms, noise) * np.pi * fw ** 2)}


def real_star_stamps(lin: np.ndarray, starless: np.ndarray, model_fw: float, n: int = 150, R: int = 40):
    """The image's own brightest stars as they are - star, halo, whatever the optics and the
    processing did to them (clipped cores rendered neutral, a restoration's coadd-filled saturated
    stars) - as the classic star layer ``lin - starless`` in a disc of radius R round each, tapered
    to 0 at the edge.  (n, 3, 2R+1, 2R+1)."""
    L = np.ascontiguousarray(luminance(lin))
    bkg = sep.Background(L, bw=64, bh=64)
    objs = _extract(L - bkg.back(), 20.0, bkg.globalrms, minarea=5)
    h, w = L.shape
    ok = (objs["x"] > R) & (objs["y"] > R) & (objs["x"] < w - R - 1) & (objs["y"] < h - R - 1)
    objs = objs[ok]
    objs = objs[np.argsort(-objs["flux"])][:n]
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    taper = np.clip((R - np.hypot(xx, yy)) / (0.25 * R), 0, 1).astype(np.float32)
    layer = lin - starless
    out = []
    for x, y in zip(objs["x"], objs["y"]):
        xi, yi = int(round(x)), int(round(y))
        st = layer[yi - R:yi + R + 1, xi - R:xi + R + 1] * taper[..., None]
        out.append(np.maximum(st, 0).transpose(2, 0, 1))
    return np.stack(out).astype(np.float32) if out else np.zeros((0, 3, 2 * R + 1, 2 * R + 1), np.float32)


def paste_stamps(stars: torch.Tensor, stamps: torch.Tensor, rng: np.random.Generator, p: float = 0.5):
    """Add one or two real star stamps (random rotation, flip and a little flux scaling; chosen with
    probability ~ sqrt(flux)) to about half of the star fields, in place."""
    n, _, P, _ = stars.shape
    k = stamps.shape[-1]
    if not len(stamps) or k >= P:
        return stars
    # the brightest (rarest, and hardest: clipped, neutral, widest) are drawn most often
    prob = np.sqrt(np.maximum(stamps.sum((1, 2, 3)).cpu().numpy(), 1e-12))
    prob = prob / prob.sum()
    for b in range(n):
        for _ in range(int(rng.random() < p) + int(rng.random() < p * 0.4)):
            st = stamps[int(rng.choice(len(stamps), p=prob))]
            st = torch.rot90(st, int(rng.integers(0, 4)), (1, 2))
            if rng.random() < 0.5:
                st = st.flip(2)
            y, x = int(rng.integers(-k // 3, P - 2 * k // 3)), int(rng.integers(-k // 3, P - 2 * k // 3))
            ys, xs = max(0, -y), max(0, -x)
            ye, xe = min(k, P - y), min(k, P - x)
            stars[b, :, y + ys:y + ye, x + xs:x + xe] += st[:, ys:ye, xs:xe] * float(rng.uniform(0.8, 1.2))
    return stars


def render_stars(model: dict, n: int, patch: int, rng: np.random.Generator, device) -> torch.Tensor:
    """(n, 3, patch, patch) star fields (linear, per unit of the image): Poisson numbers of stars at
    the image's density (x 0.5-2.5), fluxes and colours drawn from its detected stars (with jitter,
    extended below the detection limit and up past saturation), each seen through its channel's
    measured profile - point sources splatted bilinearly and convolved with the kernels, which are
    widened or narrowed a little per batch (the PSF changes across the field)."""
    K = model["kernels"]
    s = rng.uniform(0.85, 1.2)
    if abs(s - 1) > 0.02:
        R0 = K.shape[-1] // 2
        R1 = int(round(R0 * s))
        K = np.stack([cv2.resize(k_, (2 * R1 + 1, 2 * R1 + 1), interpolation=cv2.INTER_LINEAR) for k_ in K])
        K = K * (model["kernels"].sum((1, 2)) / K.sum((1, 2)))[:, None, None]
    R = K.shape[-1] // 2
    lam = np.clip(model["density"] * patch * patch * rng.uniform(0.5, 2.5, n), 2, 80)
    counts = rng.poisson(lam)
    lf = np.log10(np.maximum(model["flux"], 1e-12))
    lo = np.log10(max(model["flux_floor"], 1e-12)) - 0.7
    idx_b, fx, xs, ys, cols = [], [], [], [], []
    for b, m in enumerate(counts):
        kind = rng.random(m)
        f = np.where(kind < 0.7, rng.choice(lf, m) + rng.normal(0, 0.1, m),
                     np.where(kind < 0.9, rng.uniform(lo, np.percentile(lf, 30), m),
                              rng.uniform(np.percentile(lf, 90), lf.max() + 1.2, m)))
        idx_b.append(np.full(m, b))
        fx.append(10 ** f)
        xs.append(rng.uniform(-R, patch + R, m) + R)
        ys.append(rng.uniform(-R, patch + R, m) + R)
        cols.append(model["colour"][rng.integers(0, len(model["colour"]), m)] * rng.normal(1, 0.05, (m, 3)))
    idx_b, fx, xs, ys, cols = (np.concatenate(v) for v in (idx_b, fx, xs, ys, cols))
    size = patch + 4 * R + 2
    canvas = torch.zeros(n, 3, size, size, device=device)
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    fxr, fyr = xs - x0, ys - y0
    amp = torch.as_tensor((fx[:, None] * cols).astype(np.float32), device=device)             # (N, 3)
    ib = torch.as_tensor(idx_b, device=device)
    for dy, dx, wgt in ((0, 0, (1 - fyr) * (1 - fxr)), (0, 1, (1 - fyr) * fxr), (1, 0, fyr * (1 - fxr)), (1, 1, fyr * fxr)):
        wt = torch.as_tensor(wgt.astype(np.float32), device=device)[:, None]
        yy = torch.as_tensor(y0 + dy + R, device=device)
        xx = torch.as_tensor(x0 + dx + R, device=device)
        for c in range(3):
            canvas[:, c].index_put_((ib, yy, xx), amp[:, c] * wt[:, 0], accumulate=True)
    ker = torch.as_tensor(K[:, None], device=device)                                          # (3, 1, k, k)
    out = torch.nn.functional.conv2d(canvas, ker.flip(-1, -2), groups=3)
    o = R                                                  # canvas index p + 2R -> patch pixel p
    return out[:, :, o:o + patch, o:o + patch]


# ============================================================ training & inference
def train(lin: np.ndarray, starless: np.ndarray, noise: float = 0.0, iters: int = 3000, patch: int = 128,
          device="auto", progress=None, cancel=None, seed: int = 0, base: int = 32):
    """Train the star remover on (starless crop + rendered stars -> starless crop) pairs.
    ``lin``: the linear image (0..1, white level 1); ``starless``: its classic starless image."""
    device = pick_device(device) if isinstance(device, str) else device
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dom = domain_for(lin, noise)
    model = star_model(lin, noise)
    stamps = torch.from_numpy(real_star_stamps(lin, starless, model["fwhm"])).to(device)
    if model["n_detected"] < 20:
        raise RuntimeError(f"only {model['n_detected']} stars detected: too few to learn what a star looks like")
    batch, _ = _batch_and_tile(device)
    net = UNet(base=base).to(device)
    use_scaler = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=iters, pct_start=0.15)
    bg_all = torch.from_numpy(np.ascontiguousarray(starless.transpose(2, 0, 1))).to(device)
    h, w = starless.shape[:2]
    sig_t = torch.as_tensor(dom.sigma, device=device).view(1, 3, 1, 1)
    # half of every batch comes from the brightest backgrounds: drawn uniformly, crops are nearly all
    # dark sky (a nebula fills a few % of a wide field), the network never saw a star on bright
    # nebulosity - where the stabilised domain compresses it differently - and left them all in
    # place (C 33: every star on the Veil's filaments)
    Ls = cv2.GaussianBlur(luminance(starless), (0, 0), patch / 4)[: h - patch, : w - patch]
    bright = np.argwhere(Ls[::8, ::8] >= np.percentile(Ls[::8, ::8], 90)) * 8
    t0 = time.time()
    for it in range(iters):
        if cancel and cancel():
            raise RuntimeError("cancelled")
        ys = rng.integers(0, h - patch, batch)
        xs = rng.integers(0, w - patch, batch)
        if len(bright):
            pick = bright[rng.integers(0, len(bright), batch)]
            on = rng.random(batch) < 0.5
            ys = np.where(on, np.clip(pick[:, 0] - patch // 2, 0, h - patch - 1), ys)
            xs = np.where(on, np.clip(pick[:, 1] - patch // 2, 0, w - patch - 1), xs)
        bg = torch.stack([bg_all[:, y:y + patch, x:x + patch] for y, x in zip(ys, xs)])
        k = int(rng.integers(0, 4))
        bg = torch.rot90(bg, k, (2, 3))
        if rng.random() < 0.5:
            bg = bg.flip(3)
        stars = paste_stamps(render_stars(model, batch, patch, rng, device), stamps, rng)
        x = bg + stars
        # clipped cores: the linear stage renders anything clipped in one channel neutral (its max
        # channel); they sit at the level the image's brightest stars peak at
        lvl = model["sat_level"] * float(rng.uniform(0.9, 1.1))
        clip = (x >= lvl).any(1, keepdim=True)
        x = torch.where(clip, torch.full_like(x, lvl), x)
        inp, tgt = dom.fwd_t(x), dom.fwd_t(bg)
        # star pixels weigh more: the sky (identity) is most of every crop
        wgt = 1 + 4 * (stars.mean(1, keepdim=True) > sig_t.mean()).float()
        with _autocast(device):
            pred = net(inp)
        loss = (wgt * (pred.float() - tgt).abs()).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if progress and (it % 25 == 0 or it == iters - 1):
            progress(it + 1, iters, f"Training star remover {it + 1}/{iters} (loss {loss.item():.4f}, {time.time() - t0:.0f}s)")
    meta = {"domain": dom.to_dict(), "base": base, "iters": iters, "fwhm": model["fwhm"],
            "kernel_radius": int(model["kernels"].shape[-1] // 2), "sat_level": model["sat_level"],
            "n_stars": model["n_detected"], "shape": list(lin.shape)}
    return net.eval(), meta


def save(path: str, net, meta: dict):
    torch.save({"state": net.state_dict(), "meta": meta}, path)
    json.dump(meta, open(os.path.splitext(path)[0] + ".json", "w"), indent=1)


def load(path: str, device="auto"):
    device = pick_device(device) if isinstance(device, str) else device
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = UNet(base=int(ck["meta"].get("base", 32)))
    net.load_state_dict(ck["state"])
    return net.to(device).eval(), ck["meta"]


def remove_stars(net, meta: dict, lin: np.ndarray, device=None, progress=None) -> np.ndarray:
    """The starless image of ``lin`` (same grid, linear).  It is not clamped below ``lin``: the
    network also fills whatever the image had round its stars in their place (a restoration's dark
    rings), which a starless image must not keep."""
    device = device or next(net.parameters()).device
    d = meta["domain"]
    dom = Domain(d["bg"], d["sigma"])
    _, tile = _batch_and_tile(device)
    if progress:
        progress(0, 1, "Removing stars (AI)")
    g = infer(net, dom.fwd(lin), tile=tile, overlap=128, device=device, tta=2)
    return dom.inv(g)
