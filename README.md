# AstroPhoto Studio — raw FITS subs → finished astrophoto

An end-to-end pipeline that turns the raw one-shot-colour FITS subs from a smart telescope or an
astronomy camera into a finished image, built on **Siril 1.4** (headless) with **GraXpert**,
**StarNet2** and the **RC-Astro** tools (BlurXTerminator, NoiseXTerminator, StarXTerminator; licensed
from the web UI), with SetiAstro's **Cosmic Clarity** models (Sharpen, Denoise, Super-Resolution, DarkStar,
satellite-trail removal) as an optional path instead of or next to them.  It handles the whole chain: calibration, frame grading, registration with alt-az
field rotation, Bayer-drizzle integration with rejection, plate solving, gradient removal (GraXpert or
the NSNS sky survey), spectrophotometric colour calibration (Siril SPCC), deconvolution, noise reduction,
star separation, the community's finishing workflow (Statistical Stretch or the GHS two-pass stretch, highlight protection,
HOO and Hubble-like SHO palettes by the Hubble_Palette_from_Dual-Band_OSC.py script on dual-band data, Foraxx, Seti Astro star stretch, screen recombination) and export to a full-resolution
JPEG + 16-bit TIFF with a sidecar that lists every Siril command and tool call.  Every step is a named
community tool with its published parameters; the sources are cited in the code.  It includes a web UI,
a programmatic API (`/api/v1`), an Explore tab (Gaia / SIMBAD annotation), an Experiments tab (Optuna
studies of every stage) and a CLI.  Everything runs inside the Docker image; nothing is installed on
the host.

Nothing is tuned for a particular object or telescope. Every decision comes from the FITS headers
(`BAYERPAT`, `FILTER`, `EXPTIME`, `GAIN`, …), from calibration frames when there are any, and from
statistics of the data. A dual-band filter (Seestar `LP`, DWARF `Duo-Band`, L-eXtreme, …) gets an
Ha/OIII (HOO) workflow with SPCC in narrowband mode; a broadband filter gets a natural-colour workflow.

## Quick start

```bash
cp .env.example .env                      # IMAGES_DIR, OUTPUT_DIR, PORT, STARNET_ACCEPT_LICENSE, IMAGEMM
mkdir -p vendor && cp ~/Downloads/rc-astro-cli-*-linux-x64.sh vendor/   # optional: the RC-Astro CLI installer
docker compose up -d --build              # NVIDIA GPU (default profile); --profile cpu for CPU only
# → http://localhost:8000  (or your PORT)
```

Then open the web UI, choose a session folder and press **Run everything & export**.  The *Tools* card
shows Siril, GraXpert, StarNet2 and RC-Astro; paste your RC-Astro activation code there once (the
licence is kept in `OUTPUT_DIR/.home`, bound to the container's fixed hostname).  For development
without rebuilding: `docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d` mounts the
repository's code into the container.

Headless, inside the container:

```bash
docker compose exec astrophoto python -m sirilpipe run "/data/images/DWARF_RAW_TELE_M 42_…"   # everything
docker compose exec astrophoto python -m sirilpipe stack DIR [DIR ...]   # several nights of one target
docker compose exec astrophoto python -m sirilpipe analyse DIR           # frame-quality report
docker compose exec astrophoto python -m sirilpipe calibration DIR       # masters it will use
docker compose exec astrophoto python -m sirilpipe devices               # GPUs and tool status
```

### NVIDIA GPUs

The RC-Astro tools and GraXpert run on CUDA inside the container (the image carries the CUDA 12 user-space
libraries; the host needs only the NVIDIA driver and the NVIDIA Container Toolkit).  The optional ImageMM
restoration needs PyTorch: build with `IMAGEMM=cuda` (about 4 GB more).  Pick the device per job in
*Integration & compute options* (auto / a GPU / CPU).

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
- **Tools:** `STARNET_ACCEPT_LICENSE=yes` downloads the StarNet2 CLI from starnetastro.com at build time
  (you accept its licence); the RC-Astro CLI installer goes to `vendor/` before the build; GraXpert's
  AI models are downloaded on first use into `OUTPUT_DIR/.home` (or from the Tools card); Cosmic Clarity's models (2.4 GB) are
  downloaded from the Tools card and need an image built with `IMAGEMM=cuda` (or `cpu`) for PyTorch. Colour calibration (Siril SPCC) uses Gaia DR3 online;
  the sky chunk covering a field (1 of 48, 27-626 MB) is fetched on first use and cached in `OUTPUT_DIR/.home` so later fields in that region need
  no download (the Tools card shows and clears the cache).
- **ImageMM:** `IMAGEMM=cuda|cpu|off` adds PyTorch for the multi-frame restoration (`deconv_method = imagemm`).
- **Calibration library elsewhere:** set `ASTROPHOTO_CALIB` to its path *inside* the container
  (under `/data/images`), or set it per dataset in the UI's *Calibration* panel.
- **Images on a NAS:** `LOCAL_COPY=true` in `.env` makes jobs copy the dataset's light frames to a
  local folder and read them from there instead of over the network: the copy is made when a stage
  first reads the subs (calibration before registering, the ImageMM restore's raw-saturation pass);
  a job that reuses an existing registration reads no sub and copies nothing.
  The folder (`LOCAL_DIR`, default `/data/output/.local_copy` on the output volume) is cleared
  at the start of every job (a job that died leaves its copy behind) and deleted when the job ends,
  also when it fails or is cancelled. It needs free
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
- **profile** picks a restoration recipe. The default, `default`, is `STACK_DEFAULTS`. `imagemm`
  (and `n2n-imagemm`) is VeraLux ImageMM on all subs at once. `n2n-network`
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
| **Restoration: ImageMM** | **VeraLux ImageMM** v1.0.0 by Riccardo Paterniti ([vera-lux-siril-scripts](https://gitlab.com/RikyPate/vera-lux-siril-scripts/-/blob/IMM_update/VeraLux/VeraLux_ImageMM.py), GPL-3.0-or-later), an implementation of ImageMM (Sukurdeep et al. 2025, AJ 170:233), run headless (`sirilpipe/veralux_imagemm.py`). One sky image is fitted to **every sub at once** by the paper's majorize-minimize updates, each sub with its own PSF (measured on its own stars, per colour channel, at the stars' common positions), its transmission, its sky level and gradient (both against the mean of the subs: the target's own light is never modelled away), a measured photon-transfer noise model and Huber-robust weights on the standardised residuals. Subs the model cannot use are dropped and listed (transmission under half the median - cloud, dawn - , no measurable PSF, or a fit far outside the noise; M 31: 33 of 299). The preparation runs as Siril commands on the stack's calibrated subs: cosmetic correction, `calibrate -debayer`, registration with the stack's own transforms (Lanczos-4); saturated pixels are found on the raw subs and left out with a margin that grows with the saturated patch (the light next to a full well exceeds the PSF's); a saturated star starts as a point at the light PSF photometry of its unsaturated wings measures, and where the subs' data are sparse it is drawn as the restored image's own PSF at its wings' colour. The PSF reaches 80 px: an extended wing joined from unsaturated and saturated stars (Infante-Sainz et al. 2020), with its colour and its lobe measured on the saturated stars. Changed from the script, each a defect measured on real data and listed in the module: the forward model is a convolution (the script drew stars rotated by 180°), star positions follow Siril's convention (the script read them mirrored), the PSF fit keeps the wings, frames stream from disk, and the script's per-sub `seqsubsky` is an option, off (it removed part of large targets). Every run is checked against its own subs (every sub fits within its noise). Options: `restored_resolution` (super-resolution r = 2–3), formal mode, and the `imagemm_*` processing settings. |
| **Star separation** | An **AI star remover trained on your own image** (like StarNet, with no pretrained weights). Its training pairs are made from the dataset: the classic starless image as the background, plus stars rendered with the image's own measured star profile per colour channel (halo and any dark ring included), star colours and brightnesses, up to saturated cores. The network then removes the real stars, including faint ones below the detection limit and stars on nebulosity, where inpainting smears. **Star reduction** at 1 gives the fully starless image. Until the remover is trained, the classic method is used: stars detected on a background mesh scaled to the PSF, with a concentration index that keeps galaxy nuclei and nebula knots out of the star layer, and push-pull inpainting with matched grain. |
| **Star colour & halos** | Refractors bring blue/violet (and the OIII band) to a slightly different focus, so bright stars get coloured rings. Halo light above the local background is desaturated in linear data. The star layer uses a luminance-only stretch, true linear star colour and an "unscreen" recombination, which gives white cores with no coloured blooming, dark donuts or tints over bright backgrounds. *Star colour preservation* blends in the colour-preserving stretch of Lupton et al. (2004): every pixel's RGB scaled by one factor of its intensity, so the stars keep their linear colour ratios (an L\*a\*b\* composition had pushed 45 % of M 31's well-measured stars off the blackbody locus, green or magenta). |
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

## Star separation

**Remove stars** (Pipeline panel, and part of *Run everything*) separates the stars from the linear
restored image with **StarXTerminator** when licensed (on linear data, no unscreen, as RC-Astro
recommends), else **StarNet2**.  The star layer is stretched on its own (Seti Astro Star Stretch, which is
Siril's `mtf` with m = 1/(3^a+1)), colour-balanced for dual-band data (invert, SCNR, invert), reduced with the
Deep Space Astro star-reduction formula and screened back onto the starless image (`~(~starless*~stars)`).
The **Star reduction** slider runs from 0 (as stretched) to 1 (starless).

## Auto-finish

**Auto-finish** (the last Pipeline step, the ✨ button under the Process sliders, and part of *Run
everything* unless *Auto-finish before the export* is unticked) sets the processing settings from
measurements of the image by published rules, and lists every decision with its measurement and its
source above the sliders:

| Setting | Rule |
|---|---|
| Stretch | Seti Astro Statistical Stretch (median 0.10 small subject / 0.25 frame-filling) by default; alternatives: Siril GHS two-pass recipe (background 0.10, second pass D 2.5 small subject / 1.5 above 30 % coverage), `autoghs`, `modasinh`, `autostretch`, VeraLux HMS, each after a black point at the histogram toe |
| Highlights | GHS HP < 1 at half strength on dual-band nebulae; full when more than 0.05 % of the stretched pixels clip |
| Noise reduction | NoiseXTerminator 0.6–0.85 on the stretched starless image by the measured SNR (RC-Astro, Brecher), colour denoise 0.95 |
| Local contrast | Siril `clahe` under a background mask at the lowest published LHE strengths |
| Palette / SCNR | HOO for dual-band (no SCNR: the green is OIII); natural colour and SCNR only against a measured green cast for broadband |
| Saturation | Siril `ght -sat` (GHS saturation stretch) under the subject mask, D fitted so the subject's mean HSL saturation reaches 0.45 (HOO) / 0.25 (broadband); the `saturation` setting is the factor 1-3 |
| Stars | Seti Star Stretch amount 5 (lowered to 4 / 3 against clipping); then the least intervention - colour boost 1 / 1.5 / 2, a gentler star stretch 4 / 3, colour preservation - whose stars, measured in the recombined picture as the finishing gates measure them, are coloured (saturation ≥ 0.10), physically coloured (at most 10 % of the well-measured stars off the blackbody locus: green or magenta stars are chroma noise) and kept (at most 20 % of the linear layer's 5σ stars below visibility); reduction by star density, checked the same way |

Auto-finish uses no reference images: every value comes from the image's own measurements, and no Experiments
or hyper-tune objective uses reference images either.  `autofinish.json` in the session folder keeps the last result.

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

The **Experiments** tab tunes the pipeline with [Optuna](https://optuna.org) studies. For each study you
choose an experiment, a dataset (a stacked real session, several nights of one target, or a **synthetic**
one generated in the tab), which parameters to tune and over what ranges, one or two objectives, and a
sampler (multivariate TPE, NSGA-II, random or grid).  Each study runs as its own background process.
While it runs you see the optimisation history, Pareto front, fANOVA importances, slice plots, a trials
table with a preview of every result, and the log.  Trial 0 is the pipeline's current settings.  Stop a
study, continue it, or apply its best trial to the pipeline (from the study, or with **From experiment** in
*Integration & compute options*).

Experiments, in pipeline order, each with a score and constraints (shown under the experiment in the tab;
a trial that breaks a constraint is marked ✗, the samplers steer away from such regions, and the best trial
is the best one that meets them - the baseline stays unless a trial beats it by more than the measurement's
resolution).  The measurements are paired: the same stars and the same empty-sky apertures, chosen once on a
reference image, are measured in every trial, so the differences between trials are not lost in the
scatter of a new detection.

| Experiment | Searches | Score | Constraints |
|---|---|---|---|
| **Registration & integration** | rejection algorithm and sigmas, frame weighting, frame selection, Bayer drizzle / demosaic, drizzle drop size and kernel, banding correction | fidelity against held-out frames: every second frame (in time) stacked with your settings and held out, the trial stacks the others; matched to the held-out stack's PSF per tile, noise-weighted χ² (depth, the matched-aperture S/N, reported) | star ellipticity, single-pixel outliers (hot pixels a lax rejection lets through) and satellite / aircraft trails no worse than the baseline; the auto-crop keeps 90 % of the field |
| **Satellite trails (Cosmic Clarity)** | Cosmic Clarity trail removal on the stack | trail pixels left (Hough transform on the sky) | nothing else changed above the PSF scale |
| **Gradient removal** | sky-survey reference, GraXpert, Siril polynomial / RBF, subtract / divide, degree, smoothing | emission changed by the model (validated on a synthetic sky: ρ +0.99 with the error against the truth; the flatness of the "sky" was ρ −0.19, since a model that eats faint nebula flattens it best) | gradient left in the sky ≤ 3 × the pixel noise, the model's own structure at 1-10 % of the frame ≤ 15 % of the image's |
| **Linear restoration** | satellite-trail removal (Cosmic Clarity), gradient model (sky-survey reference, GraXpert, polynomial / RBF, subtract / divide), deconvolution tool of the objects and of the stars (BXT, GraXpert, RL) and strengths (BXT Sharpen Stars / Nonstellar / Halos), noise-reduction tool and strength (NXT with iterations, GraXpert, Siril) | fidelity against held-out frames (the sharpest third stacked on its own; restoration matched to its PSF per tile, noise-weighted χ², stars masked) + ln(1 − emission changed by the gradient model) | star flux conserved (\|Δmag\| ≤ 0.05 bright / 0.10 faint), no dark rings round bright stars, gradient left ≤ 3 × noise, model structure ≤ 15 %, trails no more than Cosmic Clarity's removal leaves |
| **Star separation** | StarXTerminator, StarNet2 (stride), Cosmic Clarity DarkStar | star light left + holes dug at the reference stars | nebula / sky untouched above the PSF scale, holes ≤ 5 % of the stars |
| **Stretch & finish** | stretch tool and settings, highlights, NXT / Cosmic Clarity Denoise, Cosmic Clarity Sharpen, saturation, local contrast, stars, black point, final sky level (or midtones); palette and colour grade in the lab only (the hyper-tune keeps yours) | failed finishing gates first, then 1 − *H*: the structural fidelity of the finished image to its linear data (TMQI, Yeganeh & Wang 2013, with the significance set by the data's noise) - how much of the structure the data holds above its noise is made visible | every finishing gate no worse than in the baseline, gate by gate |
| **Super-resolution (Cosmic Clarity)** | Cosmic Clarity Super-Resolution vs Lanczos, 2-4× | star FWHM in original pixels | binned back, the enlargement reproduces the original (PSNR ≥ 32 dB, flux within 2 %) |

A parameter that has no effect under the other settings (the RBF smoothing of a polynomial model, the Cosmic
Clarity amounts with Sharpen off) is not sampled, and a trial that repeats an earlier trial's settings reuses its
result.  Synthetic datasets (Seestar-format subs of an analytic sky with stars, nebulae, galaxies, dither, field
rotation, Moffat seeing, noise, hot pixels, satellite trails) are also scored against their exact truth,
registered onto the stack's grid.

### Hyper-tune each stage

The **Hyper-tune each stage** checkbox under *Run everything & export* (off by default; *Trials per stage*
appears when it is ticked) makes Run everything search the best configuration this framework offers for the
target.  Before each of the four stages it runs that stage's experiment on this dataset with exactly the number
of trials you set (a normal study in the Experiments tab, named *Hyper-tune <target>: <stage>*; the first trial is
your current settings), runs the stage with the best trial and moves on; the export renders the best finishing
trial.  Registration, restoration and star removal are searched by multivariate TPE (Falkner et al. 2018) with the
constraints; the finish - some 28 settings against a dozen gates whose feasible region lies round your settings - by
the (1+1)-CMA-ES for constrained optimisation (Arnold & Hansen, GECCO 2012), a local search from your (Auto-finish's)
settings that adapts its step size by the success rule and learns from every trial that breaks a gate to step away
from that direction (constrained TPE over the full ranges left 50 of 60 trials outside the gates on M 31).  A trial
whose settings repeat an earlier one's (TPE re-proposes; on a plate-solved field the gradient's
"auto" is the sky survey whatever its smoothing) returns that result at no cost and still counts as a trial.

| Stage | Searched | Objective (no ground truth, no reference image) |
|---|---|---|
| Register & integrate | rejection algorithm and sigmas, frame weighting, frame selection, resampling, drizzle drop size / kernel, banding correction | **held-out fidelity**: the frames go alternately (in time) to two halves; one is held out and stacked with your settings, each trial stacks the other half with its own; the trial's stack, matched to the held-out stack's PSF by a kernel fitted per tile on luminance (difference imaging: Alard & Lupton 1998, Bramich 2008) and applied to R, G and B, is compared with it in noise-weighted χ² per channel - the halves' noise is independent, so this measures the trial's error against the true sky in every channel (on luminance alone a Bayer-drizzled stack won and its R / B band networks failed the finish's colour gates); a trial must beat your settings by two standard errors of the per-tile gains; trails the trial lets through are looked for in the residual; constraints: at least 3 frames, ≥ 90 % of the field kept, star ellipticity at most 0.02 worse, single-pixel outliers and trails no more than in your stack (+50 % / +25 %); depth (matched-aperture S/N) is reported |
| Restore | satellite-trail removal (Cosmic Clarity), gradient model (sky survey, GraXpert, polynomial, RBF; subtract / divide), deconvolution tool of objects and of stars with strengths and BlurXTerminator's settings, noise reduction tool, strength and NXT iterations | **held-out fidelity + ln(1 − emission changed by the gradient model)**: the sharpest third of the frames is stacked on its own and held out, the trials restore the other frames, and the restoration matched to the held-out stack's PSF is compared with it in noise-weighted χ² (stars masked) - a deconvolution that recovers real structure and a denoiser that removes noise lower it, invented, smeared or misplaced structure raises it; a gradient model that takes a fraction *f* of the nebula's signal costs ln(1 − *f*); constraints: star flux conserved (bright ≤ 0.05 mag, faint ≤ 0.10), no dark rings, the gradient left in the sky ≤ 3 × the noise, the model's own structure at 1-10 % of the frame ≤ 15 %, no more trails than Cosmic Clarity's removal leaves |
| Remove stars | the remover (StarXTerminator, StarNet2 and its stride, DarkStar) | **star light left** at the reference stars (left-over cores and dug holes both count); constraints: nebula and sky untouched within the noise, holes at ≤ 5 % of the stars |
| Finish | stretch tool and settings, highlights, noise reduction of the stretched image (NXT / Cosmic Clarity Denoise), Cosmic Clarity Sharpen, dual-band unmixing, local contrast, saturation, star layer, black point / final sky level | **structural fidelity H**: the share of the structure your linear data holds above its noise that the finished image makes visible, in its shape - TMQI's multi-scale structural fidelity (Yeganeh & Wang, IEEE TIP 2013, the authors' code: 5 scales, CSF visibility thresholds) with the significance set by the data's own noise (Immerkaer 1996); the score counts failed finishing gates and their distance outside the limits first (a direction towards passing), then 1 − H; constraints: every gate your settings pass still passes, gate by gate, so that none is traded for another - the gates bound what H alone would reward (the sky's noise stretched into view, clipping, a sky off its level); the sky gates measure the sky with the stars excluded |

How the objectives were checked: on a synthetic sky with exact truth, the gradient objective ranked 12 models at
Spearman ρ +0.99 against the error to the truth (the flattest sky ρ −0.19: it is the model that ate the faint
nebula).  Depth (matched-aperture S/N) was the stack and restore score first and was replaced: on M 31 it rose 5.5×
in point-source S/N when BlurXTerminator shrank the stars from 4.5 to 2.0 px and as much again for a heavy
GraXpert denoise - star reshaping and smoothing add no information, and depth cannot see structure; the held-out
χ² on the same restorations is 1.02 unrestored (noise alone), 1.006 with NXT, 1.08-1.10 with BlurXTerminator's
default processing; the finishing fidelity *H* measured on the data tracked *H* measured against the true sky (at the data's
noise) at ρ +0.85 over 60 random finishes of the restored synthetic set, +0.90 over the 29 whose sky lay in the gate's
band; on M 31 and on Sh2-131 it fell under every damage done to a finish (M 31 / Sh2-131: blur 0.81 → 0.76 / 0.65
and 0.72 → 0.69 / 0.57, contrast flattened 0.76 / 0.66, over-sharpening 0.78 / 0.70 and 0.69 / 0.61, a brighter
midtone 0.79 / 0.69, added noise 0.805 / 0.712).  Darkening the image raises it slightly (more contrast in the
bright structure while the faint large-scale light, its coarsest scale, falls), and clipping the highlights leaves it
level: the sky and clipping gates bound those directions, so a finish chosen by *H* sits towards the dark end of the
sky band.  A trial replaces your settings only by more than the measurement's measured
resolution: identical restorations scatter by 0.0023 in depth (BlurXTerminator and NoiseXTerminator on the GPU are
not bit-reproducible), so restoration needs a gain of 0.01; Siril's stacking is deterministic (0.005); the
finishing chain is deterministic too and needs 0.005 in H.  Restoration trials share the state before the
deconvolution (trail removal, gradient, plate solve, colour calibration) per such settings.  Every stage's
experiment runs before the stage (the calibration and registration inside the stack experiment); with Auto-finish
on, its proposal is one trial of the finishing experiment, adopted only if it scores best.  The tools offered are those that can run here (a tool that is not
licensed or installed is not searched).  Super-resolution, the output scale, the crop, the palette and the colour
grade are yours and run as chosen; an ImageMM restoration (hours per run) is not searched.  Expect hours: a stack
trial re-stacks the subs.  The report (every stage's study, its score yours → best and what changed) appears above
the sliders and in `hypertune.json` in the session folder.
CLI: `python -m sirilpipe run DIR --hypertune 25`; API: `"stack_params": {"hypertune": true, "hypertune_trials": 25}`.  25 trials a stage is the default; 20-30 is the target.

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
