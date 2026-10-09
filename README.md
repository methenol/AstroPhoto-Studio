# AstroPhoto Studio — raw FITS subs → finished astrophoto

An end-to-end pipeline that turns the raw one-shot-colour FITS subs from a smart
telescope or an astronomy camera into a finished image. It
handles the whole chain: calibration, frame grading, cloud/tree/obstruction rejection,
registration with alt-az field rotation, local normalisation, sigma-clipped
Bayer-drizzle integration, AI denoising, gradient removal, deconvolution,
star separation, narrowband palettes, stretching and export to a
full-resolution JPEG + 16-bit TIFF. It includes a web UI and a CLI.

Nothing is tuned for a particular object or telescope. Every decision comes from the
FITS headers (`BAYERPAT`, `BIAS`, `FILTER`, `EXPTIME`, …), from calibration frames
when there are any, and from statistics of the data. Where a telescope leaves header
keys out, a profile of that telescope fills them in (see
[Supported telescopes and cameras](#supported-telescopes-and-cameras)). A dual-band filter
(Seestar `LP`, DWARF `Duo-Band`, L-eXtreme, …) automatically gets an Ha/OIII (HOO)
workflow. A broadband filter (`IRCUT`, `Astro`, `VIS`, none) gets a natural-colour RGB workflow.

## Quick start

```bash
source .venv/bin/activate
pip install -r requirements.txt          # or: uv pip install -r requirements.txt

# Web UI  →  http://127.0.0.1:8000
python -m webui.server                   # --images /path/to/sessions  --port 8080  --host 0.0.0.0

# or headless, one command:
python -m astrophoto run "images/IC 5070_sub"
python -m astrophoto run DIR --palette hoo --saturation 1.8 --scale 1.5 --upscale 2 --device cuda
python -m astrophoto run NIGHT1 NIGHT2     # several sessions of one target, stacked together
python -m astrophoto analyse DIR          # just the frame-quality report
python -m astrophoto calibration DIR      # telescope profile + the bias / dark / flat masters it will use
python -m astrophoto devices              # show GPUs PyTorch can use
```

Point the UI or CLI at any folder of light subs: a Seestar `<Object>_sub` folder, a DWARF
`DWARF_RAW_…` session folder, or a `lights/` folder from any camera. Only same-size
light frames with the dominant filter are used. JPG/PNG previews, the telescope's own stacks,
thumbnails and calibration frames are skipped. Several sessions of the same target (one
folder per night) can be stacked as one dataset: in the UI, open one of them and tick the others
under *Stack together with*; on the CLI, give all the folders. A combination has its own cache
folder (`output/<first folder>+<n more>-<hash>/`), and the calibration masters are searched beside
every folder. Results are cached in `output/<folder>-<hash>/`:
`stack.fits`, the two half stacks, `denoised.fits`, the coverage and
rejection maps, and `exports/`.

### NVIDIA GPUs (Windows / Linux)

The AI denoiser runs on NVIDIA CUDA, Apple Metal (MPS) or CPU. The default
PyTorch wheel on Windows/Linux is often CPU-only, so install the CUDA build:

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows  (Linux: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -m astrophoto devices          # should list your RTX card
```

On CUDA the denoiser uses mixed precision (bf16 on RTX 30/40/50, fp16 on
older cards). It sizes the training batch and the inference tiles to the
card's VRAM: 8 GB cards work fine, and 4–6 GB cards use smaller tiles. Pick a
specific GPU with `--device cuda:1`, or with `ASTROPHOTO_DEVICE=cuda:1`. The
UI also has a device selector. Everything outside the denoiser (NumPy,
OpenCV, SEP) runs on the CPU and is platform-independent.

### Docker (NVIDIA or CPU)

`Dockerfile` and `docker-compose.yml` run the web UI in a container, with one profile per
compute backend. NVIDIA is the default (`COMPOSE_PROFILES=nvidia` in `.env`); `--profile cpu`
replaces it:

```bash
cp .env.example .env                              # then set IMAGES_DIR, OUTPUT_DIR, PORT
docker compose up -d --build                      # NVIDIA GPU
docker compose --profile cpu up -d --build        # CPU only
# → http://localhost:8000  (or your PORT)
docker compose logs -f                            # follow the server log (add --profile cpu for CPU)
docker compose down                               # stop
```

- **Images folder, read-only:** `IMAGES_DIR` (default `./images`) is mounted at `/data/images`.
  Nothing is ever written there: masters combined from individual calibration frames, previews
  and plate solutions all go to the output folder. Quote paths with spaces in `.env`, e.g.
  `IMAGES_DIR="/Volumes/Home/Seestar Images/images"`.
- **Output folder, read-write:** `OUTPUT_DIR` (default `./output`) is mounted at `/data/output`.
  It holds the per-dataset caches, stacks, exports, the job history and the SPCC database
  download. It is created if missing. Point it at the same `output/` as a native install to
  share results, but use one or the other at a time. Cache folders are named after the dataset's
  path, so a dataset opened as `/data/images/…` in the container gets its own cache, separate
  from the one a native run made.
- **Port:** `PORT` on the host maps to 8000 in the container.
- **User:** the server runs as `PUID:PGID`. By default that's whoever owns `OUTPUT_DIR` on the
  host, or `1000:1000` when Docker had to create the folder (then it belongs to root). The
  entrypoint starts as root only to give that user the output folder (it changes only files
  that belong to someone else), then drops root. On a NAS share that squashes root, that
  ownership change cannot work. Set `PUID` / `PGID` in `.env` to the share's owner; the
  container says so and exits if the folder stays unwritable.
- **NVIDIA:** needs the NVIDIA driver and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
  on the host (Linux, or Windows with WSL2). The image uses PyTorch's CUDA 12.8 wheels, which
  cover RTX 20 to 50 series cards with a driver of version 570 or newer. For older drivers, set
  `TORCH_INDEX=https://download.pytorch.org/whl/cu126` and rebuild. `GPU_COUNT` and
  `ASTROPHOTO_DEVICE=cuda:1` select GPUs.
- **macOS:** Docker on a Mac cannot pass the Apple GPU (Metal) into a container, so only the
  `cpu` profile works there, and the AI stages are much slower than a native run on MPS.
  Docker Desktop must be allowed to share the images folder (*Settings → Resources → File
  sharing*; `/Volumes` covers mounted NAS shares).
- **Calibration library elsewhere:** set `ASTROPHOTO_CALIB` to its path *inside* the container
  (under `/data/images`), or set it per dataset in the UI's *Calibration* panel.
- **Images on a NAS:** `LOCAL_COPY=true` in `.env` makes every job that reads the subs (calibrate,
  analyse, stack, ImageMM restore, run everything) copy the dataset's light frames to a local folder
  first and read them from there, so each stage does not read every sub over the network again.
  The folder (`LOCAL_DIR`, default `/data/output/.local_copy` on the output volume) is cleared
  before the copy and deleted when the job ends, also when it fails or is cancelled. It needs free
  space for one dataset's subs. Natively, set `ASTROPHOTO_LOCAL_COPY=true` (and optionally
  `ASTROPHOTO_LOCAL_DIR`).
- **Memory:** worker pools are sized from the container's memory and CPU limits (`mem_limit`,
  `cpus`), not the host's. Stacking and PyTorch share data through `/dev/shm`
  (`SHM_SIZE`, default 8 GB).

### Programmatic API

The web server also has a JSON API at `/api/v1` for scripts and other programs, such as a
capture computer that queues its sessions when a night ends. Jobs submitted there go into the
same queue as the UI's and show in its Jobs panel. The API has no authentication, so keep the
server on a private network.

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/health` | Quick reachability check |
| `GET /api/v1` | Version, devices, profiles, presets, job kinds, images and uploads folders |
| `GET /api/v1/profiles` | Named restoration recipes (`astrophoto/pipeline.py` `PROFILES`) |
| `POST /api/v1/datasets/resolve` | Whether the server sees a session folder: sub count, target, filter |
| `PUT /api/v1/uploads/{path}`, `GET /api/v1/uploads?prefix=` | Upload subs (and calibration) the server cannot see; resumable |
| `POST /api/v1/jobs` | Queue a job |
| `GET /api/v1/jobs`, `GET /api/v1/jobs/{id}` | State, stage, progress, ETAs and output files |
| `POST /api/v1/jobs/{id}/cancel` | Cancel a job |
| `GET /api/v1/jobs/{id}/files/{name}` | Download an output (JPEG, TIFF, JSON sidecar) |
| `GET /api/v1/queue`, `POST /api/v1/queue/move` | Running and queued jobs in order, with start estimates; reorder |

```bash
curl -X POST http://server:8000/api/v1/jobs -H 'content-type: application/json' -d '{
  "paths": ["DWARF_RAW_TELE_C 33_EXP_15_GAIN_60_2026-09-30-22-11-12-896",
            "DWARF_RAW_TELE_C 33_EXP_15_GAIN_60_2026-10-01-21-02-01-264"],
  "profile": "n2n-network", "client_ref": "c33-both-nights"}'
```

- **Paths** are relative to the images folder (`--images`), or to the uploads folder with
  `"source": "uploads"` (`<workdir>/uploads`, or `ASTROPHOTO_UPLOADS`). Several `paths` are stacked
  together as one dataset, like *Stack together with* in the UI. A folder whose subs are in a
  `lights/` subfolder is also accepted.
- **kind** defaults to `all` (analyse, stack, restore, star remover, auto-finish, export). The other
  kinds are the UI's pipeline steps. `"stack_params": {"autofinish": false}` exports with the given
  processing settings instead of tuning them.
- **profile** picks a restoration recipe. The default, `default`, is `STACK_DEFAULTS` (ImageMM with
  its Noise2Noise pass). `imagemm` is ImageMM on all subs at once, without the pass. `n2n-network`
  (Noise2Noise + the deconvolution network) is the fast one, and `n2n-rl` uses Noise2Noise with
  Richardson–Lucy (the UI's *Noise2Noise + Richardson-Lucy*); both stack at 2× drizzle (`"stack_params": {"scale": 1}` for native). `stack_params`, `params` and `preset` override single settings on top of the
  profile. `GET /api/v1/profiles` lists them with their settings.
- **client_ref** makes a submission idempotent: sending it again returns the job already queued
  or finished, so a client can safely retry after a lost reply.
- Uploaded sessions keep their layout, so a DWARF `CALI_FRAME` uploaded beside the session folders
  is found as usual.

## What happens to your data

| Stage | Technique |
|---|---|
| **Calibration** | **Bias, dark and flat masters** when there are any: a DWARF's factory and user masters (`CALI_FRAME`), or `darks/` `flats/` `biases/` folders for any camera (see [Calibration frames](#calibration-frames)). The dark's thermal signal is **scaled per sub from its hot pixels**, so a dark taken at another sensor temperature still fits. The flat is normalised in each CFA site separately, so it corrects vignetting and dust without shifting colour. Without masters, the black level comes from the FITS `BIAS` header. In both cases, residual hot and warm pixels are found from the temporal median of unregistered subs: sky drifts between frames but sensor defects don't. A pixel is flagged when it is an isolated same-colour outlier, which protects star cores. Pixels the masters show are unusable (hot beyond the linear range, dead in the flat) are added. |
| **Frame grading** | SEP star extraction on every sub measures star count, FWHM, elongation (wind or tracking trails), sky level and noise. |
| **Registration** | Asterism (triangle) matching, then RANSAC similarity refinement on every matched star. This handles the field rotation of an alt-az mount, which can reach about 100° over a long session. A robust **third-order polynomial distortion model** is then fitted per frame to hundreds of matched stars, which keeps edge stars round as the field rotates across the optics. |
| **Cloud / obstruction detection** | Reference-star photometry: each bright reference star that should appear in a frame is looked up, and the flux ratio is aggregated on a tile grid. Tiles whose stars dim or vanish (a tree, a roof, a passing cloud) become per-frame masks, so a partly blocked frame still contributes its clean area. |
| **Rejection & weighting** | Robust median/MAD tests on each metric, plus an unsupervised **Isolation Forest** over the multivariate metrics. Weights are signal²/noise² × sharpness. Sensitivity is adjustable, and each frame can be overridden in the UI. |
| **Integration** | Streaming three-pass integration with bounded memory (hundreds of subs fit in 16 GB of RAM). **Local normalisation** removes each frame's rotating gradient against the running mean. Weighted **sigma clipping** removes satellites, planes and cosmic rays. **Bayer drizzle** resamples each colour's samples directly, with no demosaic interpolation. Optional 1.5× or 2× output uses the dithering and rotation between frames. Frames alternate between two independent **half stacks**. |
| **AI denoise** | **Noise2Noise**: a U-Net is trained *on your own data* to map half-stack A to half-stack B. Because the noise in the two is independent, the network learns the expected clean signal for this exact sensor, sky and integration. It uses no pretrained weights, so it can't invent detail from other people's images. Training runs in a variance-stabilised (asinh) domain, inference averages 8 rotations/flips (self-ensemble), and bright star cores are handed back unchanged. The halves alternate frame by frame; *Noise2Noise halves: dither blocks* deals whole dither blocks instead, which tested no better with sensor pattern correction on (`experiments/README.md`). *Denoiser objective* picks what the network minimises: asinh-domain MSE (default), the same with the transform's Jensen-gap bias removed (about 1 ADU at faint levels on M 31), or linear MSE / χ² / Huber, which keep star cores exact but recover fewer faint sources; the ablation is in `experiments/README.md`. |
| **Gradient removal** | **Against a sky survey** once the stack is plate-solved (done automatically after stacking when online), the idea of PixInsight's MARS. The calibrated, gradient-free maps of the NSNS survey (Hα, [OIII], continuum) are resampled onto your field. Each colour channel is fitted as a mix of those maps plus a smooth polynomial, and only the polynomial is removed: what the survey cannot explain is the gradient. Emission that fills the frame is kept. Sample-based models remove part of a frame-filling nebula's diffuse body (about 20 % on the North America Nebula). Without a plate solution, offline, or outside the survey: tile samples with stars masked and an iterative *lower-envelope* fit (polynomial of degree 0–4, or thin-plate RBF with adjustable smoothing). *Auto* keeps a plane unless a more flexible model leaves clearly less gradient in the sky. The gradient is subtracted (light pollution, airglow) or divided out (vignetting). The ImageMM restoration uses the same sky model. |
| **Crop** | Largest fully covered rectangle, found by an aspect-ratio search on the coverage map and centred on the deepest part of the stack. The minimum coverage is adjustable. |
| **Colour** | Background neutralisation, then **spectrophotometric colour calibration** (SPCC, as in Siril). Once the stack is plate-solved (done automatically after stacking when online), every Gaia DR3 star in the field gets a predicted colour *in your camera*: a Pickles library spectrum for its Gaia temperature, reddened by its own Gaia extinction (CCM89), integrated through your sensor's R/G/B response and your filter's transmission. The per-channel gains are the robust fit of the measured star colours to those predictions, relative to a white reference (average spiral galaxy, or G2V). Sensor and filter come from the FITS headers (camera name, model number, sensor geometry, `FILTER`), or you pick them from Siril's database. Without a plate solution it falls back to star-based white balance (the average star is white). Aperture photometry uses isolated, unsaturated stars, with the aperture sized to the *widest* colour channel. Pixels clipped in any channel are rendered neutral. |
| **AI deconvolution** (option) | A second network is trained on the same half-stack pairs to *undo the blur*. This is a Noise2Noise adaptation of ZS-DeconvNet. The network takes the denoised half A. Its output, blurred by **PSFs measured from your own stars (one per colour channel)**, must predict the raw half B (χ² with the measured per-pixel noise). The only way to lower that loss is to recover the true, sharper sky. A Hessian penalty and a physical **sky-floor prior** (no flux below the local sky) prevent the dark rings and noise that classic deconvolution produces. In held-out tests on M 27, IC 5070 and M 31, star FWHM fell by 2–2.5× with no ringing, against 1.1× for the old masked Richardson–Lucy (see `experiments/README.md`). Saturated stars are handed back to the denoised image. Richardson–Lucy with TV regularisation remains as the fallback when too few stars are available. |
| **Restoration: ImageMM** (default) | **ImageMM** (Sukurdeep et al. 2025, arXiv:2501.03002), as published. One non-negative sky image is fitted to **every individual sub at once**, each sub with its own PSF, photon-transfer variances and masks, by majorization-minimization with Huber-robust weights (removes satellite trails), the paper's update clipping, starting image (the median of the exposures) and stopping rule (Eq. C15; it converges in 24–157 iterations per cutout on M 42, the paper's "under 100"). Biggs–Andrews acceleration and a flux-based stopping rule are options, off by default: run 2000 iterations past the paper's convergence they deepened the dark crescents above saturated stars. The PSFs are measured on the subs themselves: the preparation sums the registered subs into their own coadd and measures the field PSF on it (the pipeline's stack is resampled and combined differently and its stars differ from the subs' by a few per cent in the core, which showed up as a dark ring round every bright star); the brightest unsaturated stars are included, and the wings are smoothed by a low-order angular expansion that keeps their real asymmetry. Each sub's kernel is then G(σ<sub>t</sub>) ∗ Q: a seeing-free PSF Q (the coadd's PSF deconvolved by the mean of the subs' Gaussians) convolved with the sub's own Gaussian seeing σ<sub>t</sub>, fitted on its bright stars — a coadd PSF is a mixture over seeing, and no radial scaling of a mixture fits any single sub. The subs' kernels average to the coadd's PSF by construction. The latent is non-negative, as in the paper (a sky pedestal, 50× the subs' pixel noise, is an option, off by default: it let the latent dig 50σ below the sky beside saturated stars). **Saturated stars** follow Whyte, Sivic & Zisserman, *Deblurring shaken and partially saturated images* (IJCV 2014): ImageMM's masks drop every pixel whose resampling uses a saturated raw sample, and the bright latent pixels behind them, estimated from incomplete data, spread their errors through the blur — on M 42 a flat-topped core and a dark crescent above every saturated star (to −130 ADU). Each iteration the latent is split at 0.9× saturation: the bright pixels are updated from all the data through a smooth saturation response, the rest only from the measurements the bright pixels cannot reach (outside them grown by the PSF's support); pixels with no such data keep the measured image, so a saturated star keeps its recorded glow round a deconvolved core. Without saturated pixels the update is the paper's exactly. Every restoration kernel is scaled to unit sum first: the non-negative Eq. 11 solution of an aberrated PSF carries up to 1–4 % too much flux in the field's corners, which on the pedestal became a different sky level in every cutout (M 42, 2026-10-06: steps of up to 55 ADU). The latent resolution (Gaussian σ, 1.0 at 1×) is checked against the PSFs before the Eq. 11 solve and raised in steps of 0.1 if the PSF core is flatter than that Gaussian allows (the only non-negative Eq. 11 kernel is then ring-shaped and rings every star; the chosen σ is reported in the restoration info). The subs are prepared by linear demosaicing and star-refined registration (RMS at the centroid-noise level), with per-colour photometric scales and background models. Options: 2× super-resolution (the paper's Algorithm 2), Moffat PSFs, seeing groups, and the Noise2Noise pass (on by default: the two halves of the subs are restored separately and combined by a Noise2Noise network trained on the pair; star cores far above its training range are the mean of the two halves, where the network's output was ~35 % too green on M 42). The first run prepares the subs once (about 15 min for 200 subs, 55 min for 565); the full-field restoration time is measured below. |
| **Star separation** | An **AI star remover trained on your own image** (like StarNet, with no pretrained weights). Its training pairs are made from the dataset: the classic starless image as the background, plus stars rendered with the image's own measured star profile per colour channel (halo and any dark ring included), star colours and brightnesses, up to saturated cores. The network then removes the real stars, including faint ones below the detection limit and stars on nebulosity, where inpainting smears. **Star reduction** at 1 gives the fully starless image. Until the remover is trained, the classic method is used: stars detected on a background mesh scaled to the PSF, with a concentration index that keeps galaxy nuclei and nebula knots out of the star layer, and push-pull inpainting with matched grain. |
| **Star colour & halos** | Refractors bring blue/violet (and the OIII band) to a slightly different focus, so bright stars get coloured rings. Halo light above the local background is desaturated in linear data. The star layer uses a luminance-only stretch, true linear star colour and an "unscreen" recombination, which gives white cores with no coloured blooming, dark donuts or tints over bright backgrounds. |
| **Stretch** | Four algorithms, as in Siril: **Generalized Hyperbolic Stretch** (default), **arcsinh**, **histogram transformation** (midtones transfer function, as in an autostretch) and **logarithmic**. For each one the strength is solved automatically so that the starless background lands on a target level. The stretch is colour-preserving, with luminance-preserving gamut mapping so that saturated highlights never darken. |
| **Narrowband (dual-band filter)** | Ha comes from the red pixels and OIII from the green and blue pixels. Ha **leakage into OIII is estimated from the data** (lower envelope of OIII/Ha over high-SNR Ha pixels) and removed. OIII is then linearly fitted to Ha, both are stretched with one curve, and they are combined as **Foraxx** (dynamic), HOO or warm HOO. **Synthetic luminance** (LRGB-style) takes lightness from the best-SNR all-channel stretch, so red-dominant Ha regions keep their full brightness. The palette's colour is used as composed, at full strength everywhere: **nothing is desaturated and no region is treated differently from another**. Colour noise is the job of the *Colour noise reduction* slider, which treats every pixel alike. |
| **Finishing** | Post-stretch starlet shrinkage on luminance, OKLab chroma noise reduction, wavelet local contrast, perceptual (OKLab) vibrance with background protection, SCNR, curves and masked sharpening. |
| **Colour grade** | An OKLab grade of the finished image: white balance of the object (temperature, tint), hue and chroma of the warm and the cool hues separately (Ha towards crimson or orange, OIII towards cyan or blue), and an S-curve of the object's lightness. The sky is protected. Neutral until Auto-finish (or you) sets it. |
| **Auto-finish** (default before every export of *Run everything*) | Sets the sliders one at a time from measurements of the image, the way a finisher does, with reference photographs of the target as the targets and the image's own noise and clipping as the limits; then a natural colour grade for the kind of target (see [Auto-finish](#auto-finish)). |

## Supported telescopes and cameras

Any one-shot-colour (Bayer) camera that writes 2-D FITS light frames works. The headers used
are `BAYERPAT`, `EXPTIME`, `GAIN`, `FILTER`, `DATE-OBS`, `BIAS`, `CCD-TEMP` (or `DET-TEMP`), `FOCALLEN`, `XPIXSZ`,
`XBINNING`, `INSTRUME`, `TELESCOP`, `RA`/`DEC` (or `OBJCTRA`/`OBJCTDEC`) and `IMAGETYP`.
Smart telescopes leave some of these out, so a profile of the telescope (`astrophoto/instruments.py`)
fills the gaps. A header value always wins over the profile.

| Telescope | Recognised from | Profile fills in |
|---|---|---|
| **DWARFLAB DWARF 3** (telephoto) | `TELESCOP`/`INSTRUME` `DWARF 3` (or the DWARF file layout) | 12-bit data → 16-bit ADU, Sony IMX678 sensor curve. Where a header is missing: RGGB, 150 mm, 2.0 µm × binning, and exposure / gain / filter / temperature / time from the file name, target and RA/Dec from `shotsInfo.json` |
| DWARF mini, DWARF II | headers or file layout | sensor (IMX662 / IMX415), focal length, pixel size |
| **ZWO Seestar** S50 / S30 | `INSTRUME`/`TELESCOP` | GRBG, focal length, pixel size (black level from its `BIAS` header) |
| Astronomy cameras (ZWO, QHY, Player One, Touptek, …) | `INSTRUME` | nothing: their capture software writes full headers |

Without a profile, missing values fall back to GRBG, 250 mm and 2.9 µm. The web UI shows
the telescope that was recognised, and `python -m astrophoto calibration DIR` prints the values it used.

### DWARF 3

Copy the DWARF's `Astronomy` folder from the telescope (USB or the app's file access), or
at least the session folders together with `CALI_FRAME`. The DWARF writes `CALI_FRAME`
beside the session folders, and it is found there (or in any parent up to three levels up):

```
Dwarf/
├── CALI_FRAME/                        factory + your own masters, found automatically
│   ├── bias/cam_0/bias_gain_2_bin_1.fits
│   ├── dark/cam_0/dark_exp_15.000000_gain_60_bin_1_27C_stack_3.fits
│   ├── flat/cam_0/flat_gain_2_bin_1_ir_2.fits          ir: 0 VIS, 1 Astro, 2 Duo-Band
│   └── …/cam_1/…                                      the wide-angle camera's
└── DWARF_RAW_TELE_C 20_EXP_15_GAIN_60_2026-09-30-21-03-44-039/   ← open this folder
    ├── shotsInfo.json
    ├── C 20_15s60_Duo-Band_20260930-210453262_35C.fits
    ├── failed_C 20_15s60_Duo-Band_20260930-210438244_35C.fits
    ├── Thumbnail/, img_*.png/.tif, stacked*.jpg/.png    previews: ignored
    └── stacked-16_C 20_….fits                          the DWARF's own RGB stack: ignored
```

What a DWARF 3 sub is (checked on real C 20 data):

- 3840×2160 uint16, RGGB, and full headers: `TELESCOP`/`INSTRUME` `DWARF 3`, `FILTER`, `EXPTIME`,
  `GAIN`, `XPIXSZ` 2.0, `FOCALLEN` 150, `DET-TEMP` (sensor temperature), `RA`/`DEC` in degrees,
  `DATE-OBS`. The profile and the file name only fill in what a header leaves out.
- **12-bit values, unscaled** (0…4095, black level ≈ 200), in the subs and in the masters alike.
  Both are read ×16 as 16-bit ADU, so saturation and every threshold mean what they do for a
  16-bit camera. If a sub ever holds values above 4095 (a firmware that scales to 16 bits), the
  scaling turns itself off.
- `shotsInfo.json` gives the RA in hours (`"RA": 20.98` for 314.7°); it is only a fallback.
- The Duo-Band sky of a 15 s sub is only a few ADU above black, with about twice that in read
  noise. Calibrated subs are therefore **not clipped at 0**: clipping the noise below black would
  add a bias as large as the faint signal.

What the pipeline takes from the DWARF:

- **Factory and user masters** (`CALI_FRAME`). It picks the telephoto camera's (`cam_0`) dark
  with the same exposure, gain and binning, at the nearest sensor temperature. The flat is the one
  for the subs' filter (`ir_0` VIS, `ir_1` Astro, `ir_2` Duo-Band). The DWARF stores its flats
  with the pedestal still in them, and the bias removes it. The bias is taken at gain 2 while
  lights and darks are at gain 60, and their pedestals differ by about 2 ADU. So the bias is
  levelled to the dark's pedestal before the dark's thermal signal is scaled. The wide-angle
  camera's masters (`cam_1`) are only used for wide-angle sessions.
- **Darks fitted to each sub's temperature.** The sensor is uncooled: C 20 ran at 35–42 °C
  against darks at 27 °C. The dark's thermal signal (dark − bias) is scaled for each sub, by how
  much its hot pixels stand above their neighbours (the idea of Siril's dark optimisation). The
  fit is a robust L1 fit. A least-squares fit is ruled by the few hundred hottest pixels, which
  grow faster with temperature than the rest. On C 20 the fitted scale rises steadily from
  ×1.29 at 35 °C to ×1.50 at 41 °C. It leaves 5–18 % less hot-pixel residual than plain dark
  subtraction, and more the warmer the sub.
- **Flats checked against the sky.** The factory flats are nearly flat (1–2 % centre to corner).
  Each session compares the large-scale unevenness of its sky with and without the flat (on
  C 20: 11.4 → 10.9 ADU). A flat that makes the sky less even is not used, and the reason is
  reported.
- **`failed_` subs**, which the DWARF's live stack rejected, usually because of cloud, are not
  dropped unseen. They are graded like every other sub, so a frame with a passing cloud can
  still give its clean tiles. The frames table notes them, and you can override them in the
  Frames tab.
- **Colour calibration** uses the IMX678 sensor curve. The Siril SPCC database has no DWARF 3
  filter curves: `Astro` and `VIS` are taken as UV/IR cuts, and `Duo-Band` uses DWARFLAB's
  DWARF mini dual-band curve, the closest one available. You can pick others in
  *Linear → Colour calibration*.
- `Duo-Band` subs get the Ha/OIII workflow; `Astro` and `VIS` subs get natural-colour RGB.

If `CALI_FRAME` lives elsewhere, set it in the web UI's *Calibration* panel (*Library folder*),
or pass `--calib /path/to/CALI_FRAME` (CLI and web server) or `ASTROPHOTO_CALIB=/path/to/CALI_FRAME`.
To record darks for your own exposure, gain and temperature, use the DWARF app's dark library.
The masters it stacks land in `CALI_FRAME` and are picked up automatically. A dark close to the
lights' temperature still gives the best result.

## Calibration frames

Calibration is optional: without it, sessions calibrate from the `BIAS` header as before.
Masters are looked for in these places:

1. a DWARF `CALI_FRAME` folder, in the session folder, beside it, or in a parent up to three levels up
2. `darks/`, `flats/`, `biases/` (also `dark`, `flat`, `bias`, `offsets`) folders in the session folder
   or beside it (the Siril `lights/ darks/ flats/ biases/` layout)
3. calibration frames (`IMAGETYP` Dark / Flat / Bias) among the lights
4. the dataset's *Library folder* in the web UI, `--calib DIR` or `ASTROPHOTO_CALIB` (several folders
   separated by `:`; `ASTROPHOTO_CALIB=off` disables calibration everywhere)

Individual frames are median-combined into a master once and cached in the session's
`calib/` folder. Frames that already are masters (`STACKCNT` > 1, `stack_N` or `master` in
the name, anything under `CALI_FRAME`) are used as they are. Masters stored as 32-bit
floats normalised to 0–1 (Siril's default) are rescaled to ADU.

A master must match the lights' frame size (and so their binning) and camera. The choice
is then:

- **Dark**: same gain and exposure, nearest temperature, then the largest stack. A dark of
  another exposure is used only with a bias, scaled by the exposure ratio and then refined per sub.
- **Flat**: the lights' filter. A flat stored with its pedestal still in it is detected and
  the bias removed. The flat is dropped if it makes the sky less even.
- **Bias**: the nearest gain, levelled to the dark's pedestal when their gains differ.

Each sub is then calibrated as `(light − bias − k·(dark − bias)) / flat`, where `k` is fitted
per sub when a bias is available. Without a bias, it is `(light − dark) / flat`. The flat is
normalised in each CFA site, and saturated pixels stay saturated after the flat division.

### In the web UI

**Calibrate** is the first step of the *Pipeline* card. *Analyse frames* and *Run everything*
calibrate first as well, as their first progress stage. The *Calibration* panel under the
steps shows what `python -m astrophoto calibration DIR` prints:

- the telescope profile (sensor, Bayer pattern, bit depth, optics) and the lights' exposure,
  gain, filter and temperature range
- the dark, bias and flat in use, the fitted thermal-scale range, the flat check, the pedestal, and notes

Its settings are kept per dataset:

- **Use calibration masters** on or off
- an extra **Library folder**
- **Dark / Flat / Bias**: *Auto*, *None*, or any master that fits the lights

*Save & calibrate* applies the settings. If the masters in use change, the frame analysis is
dropped, so *Analyse frames* measures the subs again with the new calibration. The dataset panel
shows a one-line summary (hover over it for the files), and the stack's FITS header records it
too (`CALIBRAT`). Checks on a synthetic DWARF drive: `python experiments/test_calibration.py`.

## Super-resolution

Every sub from an alt-az smart telescope lands on the sky slightly shifted and rotated (tracking drift and alt-az
field rotation), so a stack samples the sky on a finer grid than any single frame. With
**Super-resolution 1.5× / 2×** (web UI: *Integration & compute options*; CLI: `--scale 2`),
the Bayer-drizzle integrator resamples every colour sample straight onto the finer grid.
There is no demosaic step and no invented detail: the extra resolution comes from the data.

Measured on 100 Seestar S50 subs of M 27: star FWHM went from **7.7″ at 1× to 6.5″ at 2×**, about 16% sharper,
with better colour resolution because each colour channel is sampled directly. At 2× the pixel scale
(1.15″/px) already samples the seeing-limited stars properly, so going beyond 2× gains nothing.
The deconvolution stage then works on the finer grid.

Costs: about 4× the stacking time and 4× larger stack and export files. It needs roughly 50+ subs to
fill the finer grid evenly. Stacking parallelism is limited automatically to fit about 3 GB of RAM;
raise it with `ASTROPHOTO_STACK_RAM_GB=6` on bigger machines. The export **Upscale** option is only
interpolation. Use Super-resolution for real detail.

## Colour calibration from the stars' physics

With **Colour calibration: auto** (the default), the pipeline does not assume that the average
star is white. It predicts each catalogue star's colour through your own camera and filter, then
solves the gains that make the image agree with those predictions:

- **Stars**: every Gaia DR3 star in the plate-solved field that has a GSP-Phot temperature. Each
  star gets a Pickles (1998) spectrum, dwarf or giant from its absolute magnitude, reddened by its
  own extinction A_G with the Cardelli, Clayton & Mathis (1989) law.
- **Camera and filter**: quantum-efficiency and transmission curves from
  [Siril's SPCC database](https://gitlab.com/free-astro/siril-spcc-database) (GPL-3.0). They are
  downloaded on first use to `output/spcc_db/` and are not bundled with this project.
  - The sensor is detected from the `INSTRUME` header (camera name or model number, e.g. ASI2600
    → IMX571) or from the sensor geometry (width × height × pixel size).
  - The filter is detected from `FILTER`.
  - When detection is ambiguous or unknown, pick the sensor or filter in *Linear → Colour
    calibration*. The pipeline then falls back to star-based white balance and says why.
- **Fit**: aperture photometry of the isolated, unsaturated stars, and a sigma-clipped median of
  predicted ÷ measured colour.
  - The processing info reports the number of stars used, the scatter, and the sensor and filter
    chosen.
  - On M 31 (Seestar, IMX585, IRCUT): 792 stars, 0.09 mag scatter.

Siril predicts star colours from each star's Gaia XP spectrum. The Gaia archive serves those
spectra one star at a time (minutes for a few hundred stars), while temperature and extinction
come with the catalogue query that plate solving already makes.

## AI star remover

**Train AI star remover** (Pipeline panel, and part of *Run everything*) takes about 2–3 minutes
on an Apple-silicon GPU. It trains a U-Net to remove stars from this dataset (details in the
*Star separation* row above). Once it is trained, *Star removal: auto* uses it for star
separation. The **Star reduction** slider then works in two ranges:

- up to 0.5 it shrinks the stars;
- from 0.5 to 1 it also fades them out, and 1 is the starless image.

A new restack or restoration makes the trained remover stale, and it has to be trained again.

## Auto-finish

**Auto-finish** (the last Pipeline step, the ✨ button under the Process sliders, and part of *Run
everything* unless *Auto-finish before the export* is unticked) works through the sliders on the
right the way a finisher does: one control at a time, in the professional order, watching one
measurement per control and stopping where the image says stop. Every step and its reason are
listed above the sliders afterwards, and the sliders land where it left them, so you can carry on
by hand.

| In this order | What it watches | Where it stops |
|---|---|---|
| Stretch | the object's midtones | where the reference images put them |
| Black point | the sky's brightness | the references' sky, but never clipping the sky's noise to black |
| HDR | the brightest extended structure | the least compression that keeps it off the ceiling |
| Midtones | the object's midtones again | re-checked after the black point and HDR |
| GHS focus (contrast) | structure against grain on the object | the focus with the most structure, grain bounded |
| Fine-grain noise reduction | the sky's grain on a **full-resolution crop** (1:1) | the least that brings it to the references' grain |
| Colour noise reduction | colour mottle of sky and object | the least that removes it |
| Local contrast | mid-scale noise and mottle on the faint parts | raised until they would show |
| Sharpening | fine noise and dark halos on the 1:1 crop | raised until they begin to show |
| Star reduction | the share of the frame the stars cover | no more than in a typical reference |
| Star brightness | clipped star cores | the brightest stars just below clipping |
| Halo suppression | blue excess round bright stars | until it is gone |
| Saturation | the object's colourfulness | the references' colourfulness, unless the colour mottles, clips or tints the field |
| OIII boost (dual-band) | the warm / cool balance | the references' balance, unless OIII noise tints the sky |
| SCNR | the share of green hues | only if it exceeds the references' |
| Colour grade | hue of the red and of the teal / blue sectors, star colour cast, tonal spread | towards the nearest reference's hues; the star field neutral (broadband); the references' tonal spread |

The **targets** are not constants in the code. They are the medians of the same measurements over
reference astrophotographs of the target: 58 freely licensed images on Wikimedia Commons of 16
targets (M 42, M 31, M 27, M 20, M 76, NGC 281, NGC 6960, NGC 6992, NGC 7000, NGC 7635, NGC 7662,
IC 405, IC 1318, IC 5070, Sh2-142, LDN 1235), chosen for natural-colour or HOO looks that a one-shot-
colour camera can reach. Only the statistics and the attributions are kept, in
`astrophoto/data/autofinish_refs.json`; `python experiments/autofinish_refs.py` rebuilds it. The
**limits** (where to stop) come from the image itself: its own grain, clipping, colour mottle and
star coverage, measured on a 1000 px render for the tonal and colour steps and on a 1024 px
full-resolution crop of the most structured part of the object for noise reduction and sharpening,
which act below the resolution of a small render. A target without references of its own uses those
of its class (emission nebulae, supernova remnants, planetary nebulae, galaxies, dark and reflection
nebulae), found from the `OBJECT` header and a list of common names; without a known class, all the
dual-band or all the broadband references. For a small object in a wide field (M 27 in a Seestar
frame) the object's brightness targets count in proportion to the share of the frame it covers.

The finished image is then compared with the nearest reference and that distance is reported,
before and after. It is a check, not the objective: the earlier Auto-finish searched the sliders to
minimise this distance and could pay for a match with the wrong trade-offs (desaturating and
tinting to match a hue histogram, lifting the sky to match a close-up's brightness). The rules of
thumb it works by (how much grain, clipping and star coverage is acceptable, how strong the grade
is) are `autofinish.RULES`, tunable in the Experiments tab (task *Auto-finish (finishing look)*).
`autofinish.json` in the session folder keeps the last result; on the CLI `--no-autofinish` turns the
step off. About 60 renders, one to two minutes on the CPU.

## Gradient removal against a sky survey

A gradient model fitted to "free sky" samples cannot tell a gradient from faint emission that fills
the field. On a wide nebula field it takes part of the nebula with it. Once the stack is
plate-solved, the gradient model therefore compares the image with a survey of the same field:
the [Northern Sky Narrowband Survey](http://www.simg.de/nebulae3/dr0_2) (NSNS DR0.2). Its Hα map
is calibrated in Rayleighs against WHAM; it also has [OIII] and star-subtracted continuum maps, at
about 10″, for declinations −16° to +76°. The maps come from the CDS hips2fits service, resampled
onto your plate solution, and are cached with the session (`skyref.npz`).

- Each colour channel is fitted, robustly and with stars masked, as a mix of the survey maps plus a
  degree-3 polynomial. Only the polynomial is removed. Light far above survey + gradient (a bright
  star's halo: the survey is star-subtracted) does not pull the fit.
- The survey also shows where the field has the least emission. The sky's zero point is set there,
  not at the image's median (in a field full of nebula, the median is nebula).
- Limits: NSNS's [OIII] and continuum maps are background-filtered above about 3°. On fields wider
  than that, structure larger than 3° is treated as gradient. Outside the survey, offline, or
  without a plate solution, the sample-based *auto* model is used; the processing info says which.
- The ImageMM restoration subtracts the same sky model from every sub. A restoration made before
  this was added used a degree-2 polynomial: run *Restore* again to use the survey.

NSNS DR0.2 is © its authors, CC BY-NC-SA 4.0 (non-commercial use; cite
[doi:10.3847/2515-5172/adfec7](https://doi.org/10.3847/2515-5172/adfec7)). Thanks to CDS
(Strasbourg) for the HiPS and hips2fits services.

## Explore: what else is in your image

The **Explore** tab identifies everything catalogued in the field. It plate-solves the stack
against **Gaia DR3**:

- Star triangles are matched to the catalogue (astroalign), trying both mirror orientations.
- A TAN-SIP world-coordinate solution is then fitted to thousands of matched stars
  (`astropy.wcs.utils.fit_wcs_from_points`), at about 0.3″ rms on M 27.
- Positions are propagated from Gaia's 2016.0 epoch to the night of observation.

It then looks up every **SIMBAD** object in the field. The results show in three views:
**Overlay** (markers on the processed image), **Side by side** (image next to a star map, with
pan and zoom linked) or **Star map**.

Hover over any star or object for a pop-out. It shows:

- the object's type, with a short explanation
- brightness compared with the naked-eye limit
- colour and surface temperature, noting when interstellar dust has reddened it
- distance from the Gaia parallax, which is also how many years ago its light left it
- luminosity relative to the Sun, corrected for dust
- motion across the sky
- for galaxies: redshift, lookback time and physical size (Planck 2018 cosmology)

Links go to SIMBAD, Wikipedia and Gaia. The side panel gives the field's constellation,
size, orientation and depth, and a searchable list of named objects, galaxies, nebulae and
variable or double stars. Click a list entry to fly to it.

Plate solving needs an internet connection once per stack; the catalogues are cached in
the session folder (`explore/`).

## Experiments (Optuna)

The **Experiments** tab tunes the pipeline with [Optuna](https://optuna.org) studies. For
each study you choose:

- an experiment: ImageMM restoration, the Noise2Noise denoiser, the N2N restoration
  network, registration & integration, star detection & separation (classic), the AI star
  remover, or gradient removal
- a dataset: a stacked real session, or a **synthetic** one generated in the tab
- which parameters to tune, and over what ranges
- one or two objectives
- a sampler: multivariate TPE, NSGA-II, random or grid

Each study runs as its own background process, so several can run at once on different
GPUs. While it runs you see the optimisation history, Pareto front, fANOVA parameter
importances, slice plots, a trials table with a preview of every result in one shared
stretch, and the log. Trial 0 is the pipeline's current settings, so every result is shown
as a change from the baseline. Stop a study, continue it with more trials, or apply its
best trial to the pipeline: from the study itself, or with **From experiment** in
*Integration & compute options*. That sets integration & compute options and processing
settings. Some tuned values are constants of the code instead, such as the star detection
thresholds and mask radii (`postprocess.STAR_DETECT`, `STAR_MASK`), the gradient model's
sampling (`postprocess.BACKGROUND`) and the star remover's training (`starnet.train`). The
study names these, and a better value becomes the default for every dataset when it is
changed there.

How results are scored:

- **Real data** is scored on held-out data only, never on data the method saw:
  - restorations: the odd subs, predicted through each one's own PSF
  - the denoiser: the independent half-stack, on held-out bands
  - star separation and the star remover: the stars the default detector finds in a window
    (flux left behind and the fraction still detected), and the change away from every star
  - gradient removal: the flatness of the star-free sky tiles, and how much of the brightest
    tiles' emission the model took away
- **Synthetic datasets** are raw subs of an analytic sky in the Seestar S50's format (stars, nebulae,
  galaxies) with the same held-out scores, plus comparison against the exact truth: error,
  SSIM, faint-emission error and star photometry. Each sub has its own dither, field
  rotation, Moffat seeing, transparency, sky gradient, shot and read noise, hot pixels,
  satellite trails and cosmic rays.

Details: [experiments/README.md](experiments/README.md#experiment-lab-optuna).

## Research & benchmarks

`experiments/` holds the harness used to choose the ML models. It scores every
denoiser and deconvolver on held-out data against the independent half-stack, so
no ground truth is needed. It covers U-Net variants, NAFNet, ZS-N2N,
non-local means, Richardson–Lucy, DPIR plug-and-play and the N2N deconvolution
network. The results and the arXiv papers behind each choice are in
[experiments/README.md](experiments/README.md).

## Tips

- **Presets** in the Process tab are starting points: *Balanced*, *Vivid nebula*,
  *Galaxy / broadband*, *Natural colour* and so on.
- Processing sliders re-render a fast preview in about a second. The heavy
  stages (analysis, stacking, denoise) only re-run when you press their buttons.
- If you change frame selection or sensitivity in the Frames tab, run
  **Register & integrate** again, then **AI denoise & deconvolve**.
- For very short sessions (fewer than about 15 subs) the stacker switches from
  drizzle to demosaic automatically. The denoiser still works, but has less to learn from.
- Export writes a JSON sidecar with every parameter, so a result can be reproduced.
