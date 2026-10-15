# The Siril backend (branch `backend`)

AstroPhoto Studio 2.x processes with Siril 1.4 (headless), GraXpert, StarNet2 and the RC-Astro tools
(BlurXTerminator, NoiseXTerminator, StarXTerminator). Everything is installed inside the Docker image;
nothing is needed on the host. The web UI, the `/api/v1` job API and the job queue are unchanged, so a
client written against 1.x keeps working.

## Build and run

```bash
cp .env.example .env                 # IMAGES_DIR, OUTPUT_DIR, PORT, STARNET_ACCEPT_LICENSE, IMAGEMM
mkdir -p vendor && cp ~/Downloads/rc-astro-cli-*-linux-x64.sh vendor/     # optional: RC-Astro CLI installer
docker compose up -d --build         # NVIDIA profile (default); --profile cpu for the CPU image
```

| Variable | Meaning |
|---|---|
| `IMAGES_DIR` | session folders, read-only at `/data/images` (API paths resolve here, as in 1.x) |
| `OUTPUT_DIR` | caches, exports, **and `.home/`: tool licences, models, Siril config** (keep it across rebuilds) |
| `STARNET_ACCEPT_LICENSE=yes` | download the StarNet2 CLI from starnetastro.com at build time (you accept its licence) |
| `IMAGEMM=cuda\|cpu\|off` | add PyTorch for the optional ImageMM multi-frame restoration (cuda adds ~4 GB) |
| `LOCAL_COPY=true` | copy a dataset's subs to the output volume before a job (NAS shares) |

The container's hostname is fixed (`astrophoto-studio`) because the RC-Astro licence is bound to the
machine identity. Activation is done once in the web UI (*Tools* card: paste the 8-letter code from
rc-astro.com/account); the licence and the models then live in `OUTPUT_DIR/.home/.config/RC-Astro`.
Do not copy that folder to another container: each container counts as a computer.

`docker-compose.dev.yml` mounts the repo's `sirilpipe/` and `webui/` into the container for development
(`docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d`, then `restart` after edits).

## What a job does

Every stage is one `siril-cli -s script.ssf` run (scripts and logs are kept under
`output/<dataset>/process/`) or one external-tool call. The recipe follows the official Siril scripts
(OSC preprocessing with Bayer drizzle, Seestar preprocessing), Naztronomy's smart-telescope script and
RC-Astro's published guidance; the sources are cited next to each constant in `sirilpipe/workflow.py`,
`post.py` and `finish.py`.

1. **Scan** FITS subs (DWARF `DWARF_RAW_*`, Seestar `*_sub`, generic `lights/`), skipping failed and
   stacked frames; group by object / filter / exposure / gain; several nights stack together; several
   exposure lengths of one object become an HDR pair.
2. **Calibrate** — DWARF darks from `CALI_FRAME` (exact exposure and gain, nearest temperature), flats
   when the filter code is unambiguous, bias only for flats; generic `darks/flats/bias` folders or
   master files; raw calibration frames are stacked by Siril. `-cfa -equalize_cfa -cc=dark`.
3. **Register** — 2-pass homography registration; per-frame FWHM, roundness, background and star
   counts go to `frames.json` and the Frames tab; weak frames (clouds, trailing) are excluded.
4. **Integrate** — `seqapplyreg` with Bayer drizzle (1×, pixfrac 1; 2× only for well-sampled, large
   sets), `-framing=max`, winsorized sigma rejection, additive-scale normalisation, wFWHM weights,
   32-bit output; auto-crop to the covered area; plate solve.
5. **Restore** (linear) — GraXpert AI background extraction; plate solve (NOMAD, then Gaia DR3, retried: the
   online catalogue answers 503 now and then); SPCC colour calibration (sensor and filter from the headers;
   narrowband mode for dual-band filters); BlurXTerminator correct-only then sharpen (or Siril `makepsf` +
   Richardson-Lucy) with the non-stellar sharpening following the measured per-pixel signal-to-noise
   (`finish.object_snr`: 0.3 on noisy data up to 0.75 on clean, RC-Astro: sharpen less on noisy data);
   then THE noise reduction: NoiseXTerminator `--dn` 0.9 .. 0.6 by the same signal-to-noise (or GraXpert's AI
   denoise 1.0 .. 0.6, or Siril) on the linear data.  `deconv_method = imagemm` runs ImageMM on the
   registered subs instead (needs `IMAGEMM=cuda`).
6. **Stars** — StarXTerminator (or StarNet2) splits stars and starless images.
7. **Auto-finish** — sky level, GHS local intensity, HDR compress, star intensity, OIII normalisation strength,
   saturation, sharpening and local contrast chosen from measured statistics; the report lists each decision.
8. **Export** — only the grain the AI denoiser leaves (starlet scales 1-3, BayesShrink with bivariate parent
   support, Chang-Yu-Vetterli / Sendur-Selesnick) is cleaned in the linear domain, a second polynomial
   background pass flattens the sky, and a Generalised Hyperbolic Stretch (Payne & Cranfield) with D / SP
   solved so the sky sits at 0.085 and the nebula's 90th percentile at 0.6 ("stretch where the signal is").
   Dual-band: Ha and OIII are stretched with one GHS fitted on Ha; the colour (Ha : OIII ratio, OIII
   normalised to Ha with Blanshan / Cranfield) is taken at colour resolution (0.3 % of the long side) from
   background-neutralised planes, the lightness from the full-resolution Ha plus the OIII in proportion to its
   signal-to-noise ("HOO with Ha luminance"), so OIII noise never reaches the picture; display-level sky
   neutralisation (offsets only).  HDR highlight compress (Statistical_Stretch), output black level to the sky
   aim, asinh star stretch, screen recombination, SCNR for broadband, sRGB JPEG + 16-bit TIFF + a JSON sidecar
   listing every step and parameter.  A second NoiseXTerminator / GraXpert pass on the stretched image runs
   only for the legacy stretches (one noise reduction, not two).  The outer 24 px of the stack are trimmed
   (noisier rim).  The legacy statistical stretch stays available as `stretch_method` mtf / asinh / log.

Which tool runs is decided at run time from the licences (`GET /api/tools`); the Linear panel in the
UI says what will be used.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q tests -m "not docker"      # unit tests: fixtures are real FITS headers and Siril .seq files
```
The `docker`-marked test runs the pipeline inside the image on a NAS dataset.
