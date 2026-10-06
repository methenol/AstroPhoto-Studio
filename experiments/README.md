# ML restoration experiments

This folder has the benchmark harness and the results behind the AI denoise and
deconvolution stage in `astrophoto/denoise.py`. Everything here is reproducible
from the cached half-stacks in `output/`.

## How to score a method without ground truth

The stacker writes two **half-stacks** (even and odd frames): the same sky with
independent noise. Each method sees only half A, and is trained only on two of every
three 256 px vertical bands. Its estimate x̂ is then compared with half B on the
held-out bands. Because B's noise is independent of everything the method saw,

    E|x̂ − B|² = E|x̂ − x|² + σ_B²

so subtracting B's measured noise variance gives the **true error** of x̂, with no
clean image needed. For deconvolution the same identity applies after re-blurring
by the measured PSF k: E|k∗x̂ − B|² − σ_B² (the fidelity score).

Every score is quoted in units of one half-stack's noise variance:
**1.0 = no better than the raw half, lower is better**. Each is computed in two ways:

* **lin**: linear, inverse-variance weighted. Dominated by stars and bright structure.
* **str**: in an asinh-stretched domain. This is what faint nebulosity looks like
  after stretching.

Deconvolution is also scored on:
* median **FWHM** of isolated stars
* **ringing**: azimuthal profile minimum / peak
* **moat**: undershoot below the local sky, in sky-noise σ
* **background noise** in faint star-free regions, relative to the denoised input

`viz.py` renders the same stretch of every method side by side for a visual check.

Test crops: the deepest 1536² region of M 27 (LP filter, planetary nebula) and
IC 5070 (LP filter, emission nebula), and the deepest 2048² region of M 31 (IRCUT
broadband, galaxy, 2× drizzle).

## Denoising (`exp_denoise.py`)

Noise2Noise training, A↔B (Lehtinen et al. 2018, arXiv:1803.04189), in the asinh
variance-stabilised domain, 2000 steps. Scores are dB of error reduction relative
to raw half A, M 27:

| Method | lin | str | Notes |
|---|---|---|---|
| Non-local means (classic) | +3.9 dB | +5.9 dB | tuned to the measured noise |
| **U-Net (production)** | **+6.9 dB** | **+7.3 dB** | 0.47 M params |
| U-Net + 8× self-ensemble (Timofte et al. 2016, arXiv:1511.02228) | **+7.0 dB** | **+7.4 dB** | +5 s of inference, **adopted** |
| U-Net, 2× training steps | +6.8 dB | +7.2 dB | overfits slightly |
| Wider 4-level residual U-Net | +6.2 dB | +7.2 dB | more capacity doesn't help |
| NAFNet (Chen et al. 2022, arXiv:2204.04676) + 8× self-ensemble | +5.5 dB | +7.4 dB | ties on faint signal, worse on stars, 4× slower to train |
| ZS-N2N 2-layer net (Mansour & Heckel 2023, arXiv:2303.11253) | +6.0 dB | +7.0 dB | only 2 min: most of the gain comes from training on the data itself |

The noise in one night's stack is simple enough that a compact U-Net is already
near the limit of what these data allow. Averaging the network over the 8
rotations and flips is the one free improvement, and it is on in production.
The recent astro papers point the same way: AstroSURE (arXiv:2604.16793) uses a
~1 M-parameter U-Net and reports that Noise2Noise nearly matches supervised
training when paired exposures exist, as they do here.

### Splitting the halves by dither block (`n2n_split`, 2026-10-03)

**Question.** Smart telescopes dither in blocks: a DWARF 3 moves every 6 subs below 60 s and drops
the frame exposed during the move. Alternating frames (A B A B …) puts both halves at every
pointing. Whatever calibration leaves fixed on the sensor (a few-frame dark at another temperature
leaves ~90 ADU rms on a DWARF 3 before `sensor_pattern` removes it) then sits on the same sky pixels
in A and B. Noise2Noise would keep it as signal. The alternative, `n2n_split: dither`, deals whole
blocks to the halves. Blocks come from `analysis.dither_blocks`: gaps in the cadence and jumps of the
field on the sensor.

**A fair score.** The usual held-out score (against half B) cannot judge this, because it rewards
whatever the two halves share. The lab's denoiser task therefore holds out one third of the dither
blocks as a separate coadd C. A and B are built from the other subs under each split, the network
is trained on A/B, and its output from A is scored against C (`blocks_lin`, `blocks_str`). C never
shares a pointing with A, under either split. On synthetic data the truth decides.

**Results** (2000 steps, 1024 px crop, paired: same subs, same C, same training seed):

| Dataset | Metric | alternate | dither |
|---|---|---|---|
| Synthetic, 72 subs in blocks of 6, 12 e⁻ fixed pattern | truth NRMSE | **0.121** | 0.134 |
| | truth faint NRMSE | 0.061 | **0.058** |
| | `blocks_str` | **0.0525** | 0.0528 |
| Same, no fixed pattern (control) | truth NRMSE | **0.119** | 0.133 |
| | truth faint NRMSE | **0.042** | 0.045 |
| NGC 281, DWARF 3, 120 subs, 20 blocks | `blocks_lin` | **1.311** | 1.314 |
| | `blocks_str` | **0.1182** | 0.1185 |
| | (`heldout_str`, against B: biased) | 0.025 | 0.054 |

**Verdict: no improvement, kept as an option; `alternate` stays the default.**
- On synthetic data the block split costs ~10 % NRMSE, with and without a fixed pattern. Halves made of whole blocks are less alike in seeing and transparency than interleaved frames.
- The fixed pattern only moves faint-emission error in its favour by ~10 %, from 5 % worse to 5 % better.
- On real DWARF 3 data the two are within 0.3 % on the fair score: `sensor_pattern` already removes what the block split was meant to decorrelate.
- The biased score against half B would have shown the frame split twice as good on real data. It shows nothing about this question.

Reproduce: Experiments tab → *Noise2Noise denoiser*, tune `split`, grid sampler. Synthetic datasets:
`dither_every: 6`, `fixed_pattern_e: 12`.

### Training objective: is the asinh-domain MSE biased? (`exp_denoise_loss.py`, 2026-10-06)

**Question.** The production denoiser is trained with MSE in the variance-stabilised domain
g(x) = asinh((x − bg)/3σ). Its minimiser is E[g(B) | A], not g(E[B]): a nonlinear transform
leaves a Jensen-gap bias (Tinits & Mann 2025, arXiv:2512.24794), and heteroscedastic-noise work
favours inverse-variance χ² losses over MSE (Ye et al. 2026, arXiv:2609.21350). Same U-Net, patches,
seed and 1000 steps for five objectives (`astrophoto.denoise.make_n2n_loss`):
`asinh_mse` (production), `asinh_unbiased` (the network's output is read as the clean image and its
*expected* transformed noisy value under the measured noise, by Gauss–Hermite quadrature, is matched
to the target: the Jensen gap removed exactly for Gaussian noise), `lin_mse` (ImageMM's N2N pass),
`lin_chi2`, `lin_huber` (robust χ², δ = 3σ).

**Scores.** Held-out lin / str as above; on a synthetic twin of each crop (`bench.synthetic`: truth
= the lightly smoothed full stack, two halves with Gaussian noise following the variance-vs-level law
measured from A − B) the rms error and the *signed bias by true signal level* (`bench.bias_metrics`),
plus AstroSURE-style detection rate / false-alarm rate at one common absolute threshold of 2σ
(`bench.detection_metrics`, arXiv:2604.16793) and STAR-style aperture flux error against the truth or
half B (`bench.flux_metrics`, arXiv:2507.16385). Deepest 1536² crops of M 42 (duo-band, 204 subs) and
M 31 (IRCUT, 565 subs), single seed.

| M 31 | lin dB | str dB | synthetic rms (σ) | bias 1–8σ above sky (σ) | bias ≥ 64σ (star cores) | faint DR | flux err |
|---|---|---|---|---|---|---|---|
| raw half A | +0.2 | 0 | 1.00 | 0.00 | +0.03 | 0.74 | 0.018 |
| **asinh_mse (production)** | +10.5 | **+12.1** | 0.345 | −0.06 … −0.09 | −0.5 … −0.9 | 0.69 | **0.021** |
| asinh_unbiased | +10.7 | +11.9 | 0.326 | **−0.01 … −0.02** | −0.7 … −1.1 | **0.71** | **0.019** |
| lin_mse | +10.1 | +10.7 | 0.303 | −0.01 … −0.03 | **0.00** | 0.62 | 0.030 |
| lin_chi2 | +11.1 | +11.6 | 0.301 | −0.01 … −0.03 | **0.00** | 0.62 | 0.029 |
| lin_huber | **+11.2** | +11.7 | **0.300** | −0.01 … −0.02 | **0.00** | 0.62 | 0.028 |

M 42 agrees on every column where it has enough pixels (lin: linear losses best by 0.3–0.5 dB; str:
asinh losses best; synthetic rms 0.29 σ linear vs 0.33 σ asinh); its bins above 32 σ hold under 200
pixels and the measured variance law there is dominated by registration differences between the
halves, so its bright-end biases are not quoted.

**Verdict: the bias exists and is small; no objective beats the production one.** (`results_denoise_loss.json`)
- The production loss does bias faint signal, by **−0.05 … −0.09 σ ≈ 1 ADU** at 1–8 σ above the sky.
  `asinh_unbiased` removes it (−0.01 … −0.02 σ, the same floor every learned method shows at the sky
  level), but over three seeds it costs 0.25 dB of stretched-domain error on M 31 (below). It is the
  option if a photometric use ever needs the faint end unbiased.
- The large bias is elsewhere: both asinh objectives undershoot **star cores by 0.5–1 σ** (≈ 1 % of a
  100 σ peak) because the transform compresses them out of the loss. Linear losses remove it and win
  the linear score and the exact rms, but they **over-smooth the faint end**: 7 % fewer faint sources
  recovered, flux error 50 % worse, and 0.4–1.4 dB worse in the stretched domain, which is what the
  final image shows. χ² weighting or Huber makes no difference to this (the noise is nearly
  homoscedastic after stacking: measured slope 0 on M 31).
- The stretched-domain score, the one that matters for a picture, still favours the production
  objective. The pipeline's stars are restored by ImageMM / the deconvolution network from the raw
  halves, not by the denoiser, so the star-core undershoot does not reach the output.

**Three seeds for the two asinh objectives** (`results_denoise_loss_seed{1,2}.json`), to decide the
default. M 31 stretched-domain score: production 12.1 / 12.2 / 12.1 dB against 11.9 / 12.0 / 11.8 dB for
`asinh_unbiased`, a consistent −0.25 dB; on the M 31 twin the unbiased variant also had the larger exact
rms in two seeds of three (0.33 / 0.40 / 0.41 σ against 0.35 / 0.34 / 0.37 σ), 4–9 % fewer faint sources
and twice the flux error. On M 42 the two are within seed noise either way. Removing a 1 ADU bias
is not worth 0.25 dB of faint-structure error, so **`asinh_mse` stays the default**.

The objective is a pipeline setting: `n2n_loss` (`--n2n-loss`, *Denoiser objective* in the web UI, the
lab denoiser task's *Training objective*), used by the Noise2Noise + deconvolution-network restoration
(`n2n-network`); ImageMM's own Noise2Noise pass is a different network and keeps its linear MSE.

## Deconvolution (`exp_deconv.py`)

Every method gets the N2N-denoised half A and per-channel PSFs measured from the
stars. The PSFs differ a lot between colours: on IC 5070 the red (Hα) PSF covers
about twice the area of the green one, because refractors focus colours
differently.

| M 27 (FWHM 4.22 px) | fidelity lin / str | FWHM | ringing | moat | bg noise |
|---|---|---|---|---|---|
| Denoised only (not deconvolved) | 0.20 / 0.19 | 4.22 | 0 | – | ×1.0 |
| Old production RL + TV, masked | 1.70 / 0.38 | 4.08 | 0 | – | ×1.0 |
| Plain Richardson–Lucy, 30 it | 0.45 / 0.21 | 2.35 | −2.5% | – | ×2.9 |
| DPIR plug-and-play (arXiv:2008.13751), N2N-trained conditional prior | 2.9 / 0.35 | 3.22 | 0 | – | ×0.29 |
| N2N deconv net, 1-stage | 0.37 / 0.22 | 2.05 | −1.7% | – | ×4.3 |
| + Hessian 0.3 | 0.40 / 0.23 | 2.27 | −1.2% | – | ×1.8 |
| 2-stage (input = denoised half), Hessian 0.3 | 0.41 / 0.23 | 2.13 | −1.2% | – | ×1.8 |
| **+ sky-floor prior (3.0, margin 0.5σ)**, adopted | 0.51 / 0.24 | **2.29** | **0** | **+0.2σ** | **×0.97** |

| Adopted method vs alternatives | IC 5070 (FWHM 6.95 px) | M 31 (FWHM 5.98 px) |
|---|---|---|
| Old production RL | 6.38 px, fid 3.41 | 5.47 px, fid 0.82 |
| Plain RL 30 it | 3.76 px, **moat −56σ**, bg ×2.2 | 3.46 px, **moat −53σ**, bg ×1.7 |
| **N2N deconv (adopted)** | **2.83 px**, moat −0.6σ, bg ×0.54 | **2.35 px**, moat −0.9σ, bg ×0.72 |

![Sky-floor ablation on M 27](figures/m27_sky_floor_ablation.jpg)
*M 27, same stretch: denoised only, then N2N deconvolution without the sky floor
(dark moats), then with floor weight 1 and 10.*

![Methods on IC 5070](figures/ic5070_methods.jpg)
![Methods on M 31](figures/m31_methods.jpg)
*Raw half, denoised, old production RL, plain RL (ringing), adopted N2N deconvolution.*

**What worked.** The **Noise2Noise deconvolution network** is our adaptation of
ZS-DeconvNet (Qiao et al., Nat. Commun. 2024). ZS-DeconvNet trains on re-corrupted
copies of a single image; we train on the genuinely independent half-stacks. The
network receives the denoised half A and outputs x. Its loss is

    χ²(k ∗ x, raw half B)  +  0.3·|Hessian(asinh x)|²  +  3·|max(0, sky − 0.5σ − x)/σ|²

The χ² can only fall if x is closer to the true, sharper sky. The Hessian term
(from ZS-DeconvNet) and the **sky-floor prior** constrain what the PSF cannot see.
The sky-floor prior is our addition: real flux is never below the local sky.

Three observations led to the final design:

1. **χ² normalisation matters.** With an un-normalised weighted MSE the Hessian
   term was 10⁴× too small to do anything.
2. **Two-stage beats one-stage.** Feeding the denoised half, as ZS-DeconvNet
   does, gave sharper stars and less noise at the same regularisation.
3. **The sky floor removes dark moats.** Without it the network, like RL, dug
   dark moats around stars. Those are invisible in linear numbers but obvious
   after stretching.

**What didn't work.**

* **Plain Richardson–Lucy:** too noisy, with deep ringing.
* **Old masked RL:** safe but almost no effect.
* **SNR-gating any method:** cleaner background, but it gives back most of the sharpening.
* **DPIR:** a noise-level-conditional denoiser trained on one night's data isn't
  well enough calibrated across noise levels to act as a plug-and-play prior. It
  either over-smoothed or left stars blurred, with fidelity 2–6× worse.

**Found in full-pipeline testing.** Around bright stars and tight groups, where the
real PSF is wider than the median PSF, the network left a smooth undershoot of
about 0.6σ over a disk about 15 px wide. The benchmark missed it (it measures
isolated stars), but a hard stretch shows it as a dark disk. Production now clamps
the output at min(denoised, local median sky over 3 PSF widths) (`deconv_floor`),
which removes it and leaves dark lanes alone.

## ImageMM (arXiv:2501.03002) on the individual subs

`astrophoto/imagemm.py` implements ImageMM (Sukurdeep, Budavári, Connolly & Navarro 2025)
as published. `astrophoto/exposures.py` produces the data products the paper assumes
(Sec. 2) from the raw subs.

**The paper, equation by equation.**
* **Latent image:** background-subtracted (sky = 0), non-negative, padded by d′ − 1 so every
  exposure pixel is fully modelled.
* **Operators:** H(t) is the valid convolution with each sub's PSF; D is average pooling and
  Dᵀ subdivides into replicas.
* **Updates:** the multiplicative MM updates of Algorithms 1 and 2, with W = m/v and κ = 2
  clipping.
* **Robust variant:** Algorithm 3, with Huber weights ψ recomputed every iteration (δ = 2).
* **Initial guess:** the median of the exposures.
* **Stopping:** Eq. C15, with μ = 0.1 and ε in the paper's range.
* **Super-resolution:** the Eq. 11 kernels are solved by Adam against a Monte-Carlo g_σ.
* **Convolution:** computed directly (im2col × kernel matrix, bit-identical to `conv2d`).

**Exposures (Sec. 2 inputs).**
* **Demosaic:** linear (bilinear), because the model is a linear convolution.
* **Registration:** the analysis transform plus a WINPOS-based residual refinement. Its RMS
  equals the centroid noise (0.08–0.10 px on M 27).
* **Photometric scale:** per channel, from aperture photometry against the coadd.
* **Background:** each sub's smooth deviation from the coadd, plus the coadd's sky model.
* **Variances:** a photon-transfer fit over all consecutive pairs (M 27: c₁ ≈ 21–29 ADU per
  electron, read noise ≈ 1.2–1.4 e⁻). Normalised pair differences have σ = 0.97–0.99 at every level.
* **Masks:** saturation and footprint, and repaired hot pixels only where they dominate
  the interpolation.
* **PSFs:** each sub's own empirical PSF per channel, from 70–85 stars, measured at the
  reference positions.

**Verified against the paper's own model** (`test_imagemm.py`, synthetic Eq. 1/10 data with known truth):

| Check | Result |
|---|---|
| ⟨DHx, z⟩ = ⟨x, HᵀDᵀz⟩/r²; operators = `conv2d`/`conv_transpose2d` | exact / 6·10⁻⁷ |
| Monte-Carlo g_σ vs exact pixel integral | 7·10⁻⁵ |
| Eq. 11: D(h∗g_σ) = f (r = 1, 2, 4; paper reports 3.9·10⁻⁸) | 5·10⁻¹³ (stops at 10⁻⁸·mean f²) |
| Algorithm 1 loss non-increasing | yes |
| Algorithm 3: reduced χ² at convergence | 1.012 |
| Sky noise vs coadd (paper: "virtually none") | σ 19.0 → 0.58 |
| Star flux vs truth | ×1.003 |
| Satellite trail, L2 vs Huber | 690 → 0.000 |
| Algorithm 2 (r = 2): reduced χ² | 0.989 |
| Tiled vs whole-field restoration | 2.5·10⁻⁵ of peak |

**Stopping rule on real data** (`diag_convergence.py`: M 27, 256² cutout, 271 subs, 1500
iterations, distances inside the field):
* **Data fit:** the χ² settles at 1.0835 after about 100 iterations.
* **Slow tail:** only the brightest star cores keep sharpening after that.
* **Eq. C15 checks the *mean* of u′ₖ/u′ₖ₋₁.** The paper calls this a necessary condition, and ratios above and below 1 cancel:
  * at ε = 10⁻⁴ it stops at 14 iterations, 68% of the peak away from the 1500-iteration solution;
  * at ε = 10⁻⁶ it stops at 185 iterations, max 11%, RMS 0.5%.
* **The elementwise mean |u′ₖ/u′ₖ₋₁ − 1| < 10⁻⁶** stops at 784 iterations, max 3.5%, RMS 0.2%.
* **Defaults:** (superseded, see the audit below) the flux rule Σ|xₖ − xₖ₋₁|/Σxₖ < 10⁻⁴; Eq. C15
  and the elementwise rule are options.

**The padding is weakly constrained.** Latent pixels in the corners of the padding are seen
only through the faintest PSF wings of a few edge pixels, and they can grow without bound
(3·10⁸ against a field peak of 3·10⁴ in the test). The whole field is therefore restored in
cutouts that overlap by at least two kernel widths, and each cutout's edge band is discarded
when blending.

**Additions (not in the paper), each checked against its own reference:**
* **Biggs & Andrews (1997) acceleration.** Extrapolated steps bring the mean in Eq. C15 to 1
  about 100× early, so accelerated runs stop at ε/100. Measured against a 5000-iteration
  solution: 168 instead of 557 iterations, and closer to it (field max 18% vs 39%).
* **Seeing groups.** Each group is replaced by its inverse-variance coadd with the
  weight-averaged PSF. When a group shares one PSF, the iterates are identical to using every
  sub (1·10⁻⁶).
* **Moffat PSF.** A pixel-integrated elliptical Moffat, fitted by weighted least squares.
  It recovers the true parameters within 2%.
* **Noise2Noise pass.** ImageMM is run on the even and on the odd subs separately, and the
  two are combined by an L2 loss on the linear values (an asinh-domain target would bias
  faint signal).
* **Multi-frame loss for the deconvolution network.** ImageMM's Huber likelihood is taken
  over seeing-group coadds of the other half's subs:
  * **Kernels:** on the stack grid, with an exact half-pixel alignment for 2× drizzle
    (centroids within 10⁻⁶ px).
  * **Check:** with the true sky, the data term equals the noise level (0.97–1.01).

**Held-out benchmark** (`bench_imagemm.py`). Each method restores from the even subs of a
512² M 27 window; every odd sub is then predicted through its own PSF. The table reports
the excess of (y − DHx̂)²/v over 1, which is 0 for a perfect restoration. The other columns
are the paper's Sec. 5.2 and 5.3 metrics (S_F, σ_sky). The PSFs are the corrected empirical
PSFs (see below).

| Method | held-out χ² excess, sources (R, G, B) | sky | S_F | σ_sky | time |
|---|---|---|---|---|---|
| Coadd of the even subs (not deconvolved) | 0.207, 0.161, 0.131 | 0.124, 0.081, 0.068 | 8.5–9.7 | 10.2–12.9 | – |
| ImageMM, Algorithm 3, Eq. C15 (ε = 10⁻⁶) | 0.166, 0.119, 0.107 | 0.124, 0.082, 0.069 | 11.6–12.1 | 0.002–0.005 | 328 s |
| … elementwise stopping rule | 0.166, 0.119, 0.107 | same | 11.8–12.2 | 0.002–0.003 | 1009 s |
| **… + Biggs–Andrews acceleration** | **0.166, 0.119, 0.107** | same | 11.8–12.2 | **0.001–0.002** | **168 s** |
| ImageMM, Algorithm 1 (L2, no outlier protection) | 0.165, 0.116, 0.105 | 0.124, 0.082, 0.069 | 12.4–12.9 | 0.002–0.008 | 243 s |
| … Moffat PSF models | 0.173, 0.128, 0.112 | 0.124, 0.081, 0.068 | 12.0–12.8 | 0.006–0.007 | 558 s |
| … 8 seeing groups | 0.166, 0.121, 0.107 | 0.124, 0.082, 0.069 | 12.3–13.1 | 0.002–0.008 | 202 s |
| … Noise2Noise pass | 0.215, 0.475, 0.700 | 0.124, 0.082, 0.319 | 11.1–12.3 | 1.3–1.9 | 834 s |
| N2N deconvolution network (window held out of training) | 0.184, 0.127, 0.111 | 0.127, 0.084, 0.070 | 11.7–11.9 | 2.5–4.0 | 1018 s |
| … with ImageMM's multi-frame loss (8 seeing groups) | 0.174, 0.128, 0.117 | 0.123, 0.081, 0.068 | 12.1–12.2 | 3.4–5.1 | 2817 s |
| ImageMM 2× super-resolution (Algorithm 2, σ = 1.1, 200 accelerated iterations) | 0.165, 0.116, 0.106 on its 2× grid (0.170, 0.123, 0.109 averaged to 1×) | 0.124, 0.082, 0.069 | 10.4–10.9 | 0.004–0.006 | 3661 s |
<!-- ROWS -->

**Defaults, chosen from this table.**
* **Restoration: ImageMM.** It is the best in every channel on held-out subs, ahead of the
  coadd and of both networks, and its sky noise is about 1000× lower.
* **Loss: Algorithm 3 (Huber).** L2 is 1–2% better on this metric but has no protection
  against outliers such as satellites.
* **PSFs:** measured empirical PSFs, every sub.
* **Acceleration: on.** It reaches the fully converged result in half the time.
* **Resolution: 1×.** 2× super-resolution predicts the held-out subs 1–2% better on its own
  grid, but costs 22× the time (about 18 s per iteration on a 512² window). Eq. C15 also
  plateaus at about 9·10⁻⁵ at r = 2, so it never declares convergence. On an M4 the full
  field at 2× would take days, so it is an option.
* **Moffat PSFs, seeing groups, Noise2Noise pass and the network's multi-frame loss: off.**
  None of them is better. Groups remain a speed option.
* **Target resolution g_σ: the paper's (σ = 1 at 1×, its Fig. 5; 1.1 at 2×), with h ≥ 0 in
  Eq. 11.** Solved unconstrained, as the paper states, the r = 1 kernels had 30–75% negative
  flux, and ImageMM ran to its iteration cap with them: the multiplicative update (Eqs. 7–9)
  assumes non-negative kernels. The cause was isolated on four synthetic subs and real M 27
  subs by solving Eq. 11 (Adam, stopped at the paper's accuracy of 2 × 10⁻⁴ of the mean
  square) from different PSFs. Negative flux:

  | PSF solved from | Adam | exact minimiser, at 2 × 10⁻⁴ |
  |---|---|---|
  | the true PSF (generator) | 32–53% | 1–16% |
  | the Moffat fit | 24–59% | 0.2–44% |
  | measured, synthetic or real, cut or not | 48–76% | 66–95% |

  Two sources, neither the synthetic data:
  * Measured PSFs carry pixel noise that no h * g₁ can represent. The r = 1 operator has
    conditioning 1.8 × 10⁻⁴, so even the exact solution oscillates. The paper used smooth
    PSFEx models, where this does not arise.
  * Adam's per-parameter steps also drive the directions g₁ barely constrains.

  With the constraint h ≥ 0 (projected Adam) and a stall stop (less than 1% gain over 1000
  iterations), a 128 px synthetic window with 20 subs converges (77 iterations; at σ = 0,
  46). Against the true sky seen through g₁:

  | | nrmse | faint-emission nrmse | SSIM | PSNR |
  |---|---|---|---|---|
  | σ = 1, measured PSFs | 0.147 | 0.849 | 0.860 | 50.8 dB |
  | σ = 0, then viewed through g₁ | 0.172 | 0.705 | 0.908 | 49.4 dB |
  | σ = 1, Moffat fits (753 s for Eq. 11) | 0.238 | 0.774 | 0.886 | 46.8 dB |

  σ = 1 with measured PSFs has the lower overall error, but σ = 0 is better on faint
  extended emission and SSIM. This is one window; the Experiments tab can settle it on
  larger synthetic sets.
* **Rings around stars.** The dark discs around bright stars in processed images came from
  the HDR step, not the restoration. HDR's large-scale brightness map included the stars,
  and a deconvolved star (light packed into a few pixels) darkened a disc about two blur
  widths across around itself. Stars are now removed from that map by a morphological
  opening. What remained came from the restoration itself - a field-averaged PSF with a wing
  pedestal, and at σ = 0 the pixel latent - and is fixed (audit above).

The networks were trained with the benchmark window (plus 64 px) excluded from every
patch, including the multi-frame targets. They predict the window from half A only, and are
scored after 2×2 averaging onto the 1× grid.

**Reading the table.**
* **Held-out prediction.** ImageMM predicts every held-out sub about 20% better on sources
  than the coadd does.
* **Sky.** The sky term is the same for every method. It is a data-level floor (residual
  per-sub background, variance model), not something the restoration causes.
* **Robustness.** L2 is 1–2% better on this metric, but it has no protection against outliers
  in individual subs. In the tests the Huber loss removed a satellite trail completely, which
  L2 cannot do.
* **Acceleration.** It reaches the fully converged result, the same as the elementwise
  stopping rule, in half the time of the paper's stop.
* **Moffat models.** They fit the subs worse than their own empirical PSFs.
* **Seeing groups.** They cost nothing on this metric and save time.
* **Noise2Noise pass.** These numbers predate the fix of its noise scale (see the audit): the
  scale was ~10⁻⁴ of the real noise, so the loss ignored every source pixel. Fixed, it turns
  the speckle of a converged restoration's faint nebulosity into smooth emission and keeps the
  stars sharp (IC 405 region comparison).
* **Photometry.** Measured without annulus subtraction, ImageMM/coadd flux is about 1.3 in
  8 px apertures and 1.04 at 32 px. Most of this is the coadd's seeing halo, which small
  apertures miss while ImageMM gathers it back into the core. The residual +4% at 32 px
  matches the paper's note that ImageMM concentrates sky-background flux into sources.

**A bug found by this benchmark: biased empirical PSFs.** The first version of the per-sub
PSF estimator had three flaws:
* it set the noisy mean's negative pixels to zero *before* normalising;
* it weighted stars by flux instead of flux²;
* it kept the full cut-out.

Together these put 22–39% of the PSF flux beyond 2 FWHM, where about 5% is real. The heavy
wings made ImageMM over-concentrate flux (−0.54 mag in small apertures). The corrected
estimator:
* weights stars by flux²;
* cuts each kernel where its azimuthal profile stops being significant;
* clips negative pixels only after that cut.

On a synthetic field it matches the true PSF to 0.2% of the peak. On M 27 it leaves 4–11%
beyond 8 px, and it improved the held-out score (0.174/0.122/0.115 before).

## Experiment lab (Optuna)

The scripts above are one-off comparisons. The **Experiments** tab of the web UI
(`astrophoto/lab/`) turns every tunable part into an Optuna study (Akiba et al. 2019). You
can configure, run, monitor and review a study there, or from the command line:

```bash
python -m astrophoto.lab.generate output/lab/synthetic/<name>     # needs <name>/spec.json
python -m astrophoto.lab.run output/lab/studies/<study>           # needs <study>/config.json
```

### Tasks (`astrophoto/lab/tasks.py`)

Each task declares:

- its parameters: the range, the pipeline default and the pipeline setting each maps to
- its metrics, each with a direction
- `prepare` (shared loading, once per study) and `run` (one trial)

| Task | One trial | Real-data score | Synthetic data also |
|---|---|---|---|
| `imagemm` | ImageMM on a window, from the even subs | held-out χ² excess on the odd subs through their own PSFs (source / sky), S_F, σ_sky, SSIM vs coadd | truth at r = 1: the sky integrated over pixels; at r > 1, or with g_σ: the sky through g_σ |
| `denoise` | N2N U-Net on the half-stacks (steps, peak learning rate, patch, batch, width, self-ensemble) | half-B error on held-out bands, linear and stretched (× noise variance; 1 = raw) | truth seen through the subs' weighted mean PSF |
| `network` | N2N denoiser + deconvolution network, window held out of training | as `imagemm` | as `imagemm` |
| `stack` | re-stack in the study folder (rejection σ, local normalisation, resampling, scale, sensitivity) | background noise and star FWHM, both in native pixels | truth through the subs' mean PSF |
| `autofinish` | Auto-finish with the objective's constants (`autofinish.OBJECTIVE`: weights, soft minimum, slider cost, small-object rule, budget); the trial image is the finished render | distance to the reference looks (always with the default objective), sky grain and colour mottle, mottle on the object, how far the sliders moved | real data only |

The `denoise` task exposes the peak learning rate and patch size. This is the place to
test whether 2× drizzled stacks want a different schedule; their N2N loss converges to a
larger share of irreducible noise, so their loss curve is flatter even when training works.

### Truth metrics (`metrics.truth_metrics`)

Both the result and the truth are smoothed to a common resolution σ_eval. A per-channel
plane (the sky background and its gradient, which are not part of the truth) is removed
from the difference. Then:

- nrmse: rms error / rms of the true structure
- PSNR
- SSIM: of the asinh-stretched images
- faint_nrmse: nrmse where the true stars are negligible
- star photometry: of isolated, unsaturated true stars, with the true extended emission
  removed and the true stars in the same aperture

### Synthetic data (`astrophoto/lab/synthetic.py`)

The sky is analytic, and its components are evaluated exactly:

- stars from a power-law luminosity function with blackbody colours
- Gaussian clouds and filaments, a planetary-nebula shell, Sérsic galaxies

Rendering:

- Each sub has its own geometry (dither, field rotation) and an elliptical Moffat PSF.
- Extended emission is integrated over pixels by midpoint quadrature on a 3 × 3 sub-grid
  and convolved by FFT.
- Stars are exact stamps at their true positions.
- The image is then mosaicked through the Bayer pattern, with Poisson and read noise, a
  bias and 16-bit clipping.

Checked:

- a single star keeps its flux and lands within 0.0002 px of its true position, at 1× and 2×
- extended flux is conserved through the PSF and at 2×
- the pipeline's stack of a synthetic set matches the rendered truth to 0.05 px
- the nebula gain equals the pipeline's photometric convention (subs divided by their
  transparency relative to the median sub), which is the convention `truth_image` uses

### Storage and review

A study lives in `output/lab/studies/<id>/`:

- `config.json`
- `study.db`: Optuna's SQLite storage, with every metric of every trial kept as a user
  attribute
- `status.json`, `log.txt`
- `trials/<n>.jpg`: one stretch for the whole study

Trial 0 is the pipeline's current setting. The "best" trial is the optimum of a single
objective. With two objectives, it is the Pareto-optimal trial that is best on the first.
That trial is what *From experiment* applies to the pipeline settings.

## Auto-finish: a finisher's procedure, checked against reference astrophotographs (2026-10-06)

Auto-finish (`astrophoto/autofinish.py`) sets the Process sliders one at a time, in the order a
finisher works, each from one measurement of the render, and stops each where the image says stop.
The reference photographs (`autofinish_refs.py`: 58 freely licensed Wikimedia Commons images of 16
targets, measured by `look_stats`, statistics and attributions only in
`astrophoto/data/autofinish_refs.json`) supply the **targets** - the medians of the sky's
brightness, the object's midtones and highlights, its colourfulness, the star coverage, the grain -
and the hue direction of the grade. The **limits** come from the image: its own grain, clipping,
colour mottle, field tint and star coverage. The procedure, its measurements and its stop rules are
in the module docstring and the main README; the rules of thumb are `autofinish.RULES`.

Two kinds of render are used: a 1000 px proxy for the tonal, star and colour steps (1.3 s each on
the CPU), and a 1024 px **full-resolution crop** of the most structured part of the object for
fine-grain noise reduction and sharpening, which act below the proxy's resolution: the finisher
judges them at 1:1, and so does the procedure (`fine_stats`: the sky's grain, the fine noise on the
faint part of the object, dark overshoot round stars and edges, clipping).

**Why the redesign.** The first Auto-finish (2026-10-04) searched the sliders and a colour grade to
minimise a weighted distance between the render's statistics and the references' (a pattern search,
140 renders, then a grade fit). It matched numbers, not the finisher's intent, and found the
cheapest way to match them: it desaturated M 42 and tinted the grey pixels into the references' hue
histogram; it lifted a wide field's sky to match a close-up's brightness; it doubled every star to
match dense Milky Way references; it turned the grade's chroma gain on the sky into blue blotches.
Each was patched with another penalty, and the objective grew to twelve weighted terms, a soft
minimum and a regulariser - and still could not tune noise reduction or sharpening, whose effect
the proxy render does not show. The new procedure has no objective to game: every slider answers
one question, and the distance to the references is only reported afterwards.

**Validation** (the same datasets as before; distance to the references reported, not optimised;
renders and time on the CPU).

| Dataset | Telescope / data | Distance to the references | Renders, time | What it set (besides the grade) |
|---|---|---|---|---|
| C 33 Eastern Veil (2 nights) | DWARF 3, Duo-Band, ImageMM | 18.2 → 12.4 | 76, 137 s | palette foraxx; stretch 0.16→0.18; HDR 0.6→0; focus 2→0; fine NR 0.6→0; colour NR 0.8→0; local contrast 0.5→0.95; sharpen 0.25→0.35; star reduction 0.35→0.1; star brightness 0.9→1.0; halo 0.6→0; saturation 1.5→2.55; OIII 1.0→2.5 |
| C 20 North America (20 min) | DWARF 3, Duo-Band, ImageMM | 74.8 → 34.5 | 63, 109 s | palette hoo_warm; black point 0.02→0.07 (sky at the references', then clipping stopped it); midtones +0.2; focus 2→0; NR 0; local contrast →1.6; sharpen →0.85; saturation →2.6 (the Ha is faint: C50 0.006) |
| NGC 281 (23 min) | DWARF 3, Duo-Band, ImageMM | 67.9 → 44.4 | 52, 43 s | palette natural; stretch →0.23, black point →0.065, midtones +0.3; focus 2→0; NR 0; local contrast →1.6; sharpen →1.2; star brightness →1.2; saturation left alone (the slider barely moves so faint an object's colour) |
| NGC 7000 (315 min) | Seestar S50, network restoration | 43.5 → 20.7 | 61, 62 s | stretch →0.14, black point →0.055, midtones +0.2; focus 2→0; fine NR 0; colour NR →1.0; local contrast →0; sharpen →1.1; star brightness →0.65 (Deneb-class stars clipped); saturation →1.25 (red channel clipping) |
| M 31 | Seestar S50, IRCUT, plain stack | 27.3 → 23.1 | 66, 109 s | stretch →0.09; focus 2→0; fine NR 0; colour NR →0.85; local contrast →0; sharpen →0; saturation →1.7; HDR re-check 0→0.4 (the grade had pushed the core onto the ceiling) |

IC 1396 (Seestar S50, LP, 171 subs, network restoration) was added after the fix below: 41.8 → 27.1
in 63 renders; the class references (emission nebulae) since it has none of its own.

**A defect found on the way, in the processing itself (2026-10-06).** The dual-band palette gave the
exported IC 1396 red patches with hard edges on an otherwise grey nebula, with any restoration preset,
and later gave NGC 6960's background sky a warm pattern where it had been neutral.

One mechanism caused both. `postprocess._palette_chroma` did not use the palette's colour: it rebuilt
the chroma from the line maps smoothed at three scales, chose a scale per pixel by how *significant* a
line was there, and zeroed the chroma where neither line was significant, with ramps between 2.5 σ and
7 σ measured against a *local* sky (a 96 px median). On a nebula that fills the frame that median is
the nebula, so only its brighter clumps counted as emission: 60 % of IC 1396's pixels came out with no
colour at all, in patches with hard edges. Measuring significance against the frame's sky and weighting
the chroma smoothly only turned the patches into a smooth modulation of the saturation, and dropping
the weight entirely left the scale selection still varying the colour from pixel to pixel while the
sky's own faint structure took on the palette's hue.

**The whole mechanism is gone.** The palette's colour is now used as it is composed, at full strength,
identically everywhere. Colour noise is handled by the chroma noise reduction that follows it, which is
uniform and on a slider. Checked on NGC 6960 (dark sky, 2× drizzle), IC 1396 (frame-filling faint Hα)
and C 33 (dark sky, bright filaments), at preview size and at 1:1: no patches, no speckle, the sky's
median chroma at 1:1 is 0.0015 in OKLab on all three.

What was left in NGC 6960's sky after that was not colour at all: a network of dark bands at many
angles, already in the stack. It came from the integration of a Bayer drizzle (2×). Each sub's red and
blue lattices, rotated against the output grid, leave lines of output pixels with no red (blue) where
green has data, about 3 % of them, in a different grid for every sub. The stacker took every channel's
validity from green, so those empty pixels went into red's and blue's per-sub offsets and the block
medians of the local normalisation. The normalisation then subtracted each sub's grid of lines, smoothed
to a block or two, and the stack summed them. A 1× demosaic has every colour at every pixel, which is
why the earlier 1× stack of the same subs was clean. Validity is now per channel
(`stacking.Integrator._normalise`). Re-stacking 30 of the subs at 2×, the background's large-scale
spread (5th to 95th percentile) fell from 186 to 48 ADU in red and from 302 to 46 ADU in blue, the same
as green (36 ADU, unchanged). The survey-reference gradient fit's large red and blue scatter on that
stack (33 and 55 ADU against 3.8 in green) was a symptom of the bands, not their cause.
Regression test: `experiments/test_stacking.py`.

**What the measurements decided, and why that is right:**

- *GHS focus went to 0 on every dataset*: the softest focus gave 8–39 % more structure against
  grain than the default 2 (the rule asks for 5 %). A higher focus concentrates contrast near the
  sky level, where these stacks have noise, not structure.
- *Fine-grain noise reduction went to 0 on the restorations*: ImageMM's and the network's output
  has a sky grain far below the references' at 1:1 (C 33: 0.0015 against 0.008 in OKLab L), so a
  finisher would not blur it further. On M 31 (a plain stack, no restoration) it stayed at a low
  value for the same reason.
- *HDR went to 0* where nothing extended reaches 60 % lightness (none of the three has a burning
  core); it will engage on M 42.
- *Sharpening is limited by halos, not noise*, on a restoration: the dark overshoot round stars and
  edges (`fine_stats.overshoot`) doubles between 0 and 0.3 and the rule stops it at about 2.5x the
  unsharpened value.
- *The OIII boost is limited by the field's tint*: more OIII put cyan speckle into C 33's sky long
  before the darkest sky's median colour moved; the 90th-percentile chroma of the field (neither
  object nor star) catches it, and may not exceed 0.01.
- *Saturation is limited by colour mottle* only relative to the image's own mottle (a rise of 30 %):
  a reference's absolute mottle is a JPEG's, and a 63-frame stack cannot reach it without draining
  its colour, which is what the first run on NGC 7000 did. The same holds for the sky's colour: the
  limit is the rule or 15 % above what the image already has, whichever is larger. What did bind on
  NGC 7000 is **colour clipping**: at the default saturation 0.85 % of the frame has its red channel
  on the ceiling (flat, detail-less red); the rule allows 0.5 %.
- *A slider that does nothing measurable is left alone*: on NGC 281 (23 min) saturation moves the
  object's colourfulness by 0.0004 over its whole range, so chasing the references' value would only
  have run it to a bound.
- *Halo suppression went to 0 everywhere*: none of these restorations shows a blue excess round
  bright stars (the restoration packs the halo light back into the cores). It would engage on a
  plain refractor stack with chromatic halos.

## Literature consulted

* Lehtinen et al. 2018, *Noise2Noise*, arXiv:1803.04189
* Qiao et al. 2024, *ZS-DeconvNet* (Nat. Commun.), zero-shot deconvolution with Hessian regularisation
* Zhang et al. 2021, *DPIR*, arXiv:2008.13751, plug-and-play restoration with a deep denoiser prior
* Chen et al. 2022, *NAFNet*, arXiv:2204.04676
* Mansour & Heckel 2023, *Zero-Shot Noise2Noise*, arXiv:2303.11253
* Timofte et al. 2016, *Seven ways to improve example-based SR* (self-ensemble), arXiv:1511.02228
* AstroSURE 2026, arXiv:2604.16793, and ASTERIS 2026, arXiv:2602.17205 (self-supervised astronomical denoising)
* Self-supervised single-image deconvolution with Siamese networks, arXiv:2308.09426
* Sukurdeep, Budavári, Connolly & Navarro 2025, *ImageMM*, arXiv:2501.03002
* Biggs & Andrews 1997, *Acceleration of iterative image restoration algorithms*, Appl. Opt. 36, 1766
* Moffat 1969, A&A 3, 455; Trujillo et al. 2001, MNRAS 328, 977 (Moffat PSF)
* Bertin & Arnouts 1996 (SExtractor; WINPOS centroids, `sep`); Janesick 2007 (photon transfer)
* Krotkov 1988 (Fourier sharpness S_F)
* Akiba et al. 2019, *Optuna* (KDD); Bergstra et al. 2011 / Falkner et al. 2018 (TPE); Deb et al. 2002 (NSGA-II);
  Hutter et al. 2014 (fANOVA importances)
* Wang et al. 2004 (SSIM); Ciotti & Bertin 1999 (Sérsic b_n)
* Gaia Collaboration 2023, A&A 674, A1 (Gaia DR3); Wenger et al. 2000 (SIMBAD); Beroiz et al. 2020 (astroalign);
  Pecaut & Mamajek 2013 (spectral classes); Planck Collaboration 2020 (cosmology)

## Reproduce

```bash
cd experiments
python exp_denoise.py M27 IC5070 M31
python exp_deconv.py M27 --methods prod_rl,rl30,n2n_deconv2s_h30_f300_m50
python viz.py M27 rawA den prod_rl n2n_deconv2s_h30_f300_m50
python test_imagemm.py                       # ImageMM verification
python test_exposures.py M27 16              # exposure preparation checks
python diag_convergence.py M27 256 1500      # stopping rules on real data
python bench_imagemm.py M27 --size 512 --methods imagemm,imagemm_l2,imagemm_accel,imagemm_moffat,imagemm_g8
```
Method names encode their settings: `_h30` = Hessian 0.30, `_f300` = floor 3.0,
`_m50` = margin 0.5σ, `2s` = two-stage.
