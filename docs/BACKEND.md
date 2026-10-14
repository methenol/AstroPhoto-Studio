# The Siril backend (branch `backend`)

AstroPhoto Studio 2.x processes with Siril 1.4 (headless), GraXpert, StarNet2 and the RC-Astro tools
(BlurXTerminator, NoiseXTerminator, StarXTerminator). Everything is installed inside the Docker image;
nothing is needed on the host. The web UI, the `/api/v1` job API and the job queue are unchanged, so a
client written against 1.x keeps working.

## Build and run

```bash
cp .env.example .env                 # IMAGES_DIR, LIBRARY_DIR, OUTPUT_DIR, PORT, STARNET_ACCEPT_LICENSE, IMAGEMM
mkdir -p vendor && cp ~/Downloads/rc-astro-cli-*-linux-x64.sh vendor/     # optional: RC-Astro CLI installer
docker compose up -d --build         # NVIDIA profile (default); --profile cpu for the CPU image
```

| Variable | Meaning |
|---|---|
| `IMAGES_DIR` | session folders, read-only at `/data/images` (relative API paths resolve here) |
| `LIBRARY_DIR` | optional second folder, read-only at `/data/library` (use absolute paths in the API) |
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
5. **Restore** (linear) — GraXpert AI background extraction; SPCC colour calibration (sensor and filter
   from the headers; narrowband mode for dual-band filters); BlurXTerminator correct-only then
   sharpen (or Siril `makepsf` + Richardson-Lucy); NoiseXTerminator (or GraXpert denoise).
   `deconv_method = imagemm` runs ImageMM on the registered subs instead (needs `IMAGEMM=cuda`).
6. **Stars** — StarXTerminator (or StarNet2) splits stars and starless images.
7. **Auto-finish** — stretch target, star intensity, OIII boost, saturation and highlight protection
   chosen from measured statistics; the report lists each decision.
8. **Export** — statistical stretch of the starless image, GHS contrast, a second NoiseXTerminator
   pass, asinh star stretch, screen recombination, HOO / Foraxx palettes for dual-band data, SCNR for
   broadband, sRGB JPEG + 16-bit TIFF + a JSON sidecar listing every step and parameter.

Which tool runs is decided at run time from the licences (`GET /api/tools`); the Linear panel in the
UI says what will be used.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q tests -m "not docker"      # unit tests: fixtures are real FITS headers and Siril .seq files
```
The `docker`-marked test runs the pipeline inside the image on a NAS dataset.
