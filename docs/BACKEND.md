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
5. **Restore** (linear) — GraXpert AI background extraction; plate solve (NOMAD, then Gaia DR3, retried); BlurXTerminator correct-only;
   SPCC colour calibration (sensor and filter from the headers; narrowband mode for dual-band filters); BlurXTerminator sharpen with
   the 1-2 h values of research/pro-workflow.md (Sharpen Stars 0.25-0.35, Nonstellar 0.5-0.6, halos 0; the lower end on noisy data); NoiseXTerminator `--dn` 0.8 .. 0.5 on the linear
   data (GraXpert denoise 0.8 .. 0.6, or Siril, as fallbacks).  `deconv_method = imagemm` runs ImageMM on the registered subs instead.
6. **Stars** — StarXTerminator on the linear data (or StarNet2 `--linear`), no unscreen: `stars = linear - starless`.
7. **Auto-finish** — `autofinish.py` picks among published values by stated rules (small vs frame-filling target, clipping -> HDR, dual-band vs broadband -> palette / SCNR,
   star density -> reduction, SNR -> NXT strength); every decision is in the report with its rule.
8. **Export** — `pro.py`, the professional chain, every step a Siril command, a pyscript or an rc-astro call, a picture saved after each step
   (`process/finish/steps/NN_<step>.png`) and the verbatim commands in the sidecar (`linear_info.chain`):
   colour-gradient removal (`subsky 2`, mean model added back) and background neutralisation (per-channel offsets) on the linear starless;
   **GHS two-pass stretch** by `ght` (pass 1 SP at the histogram's left half-maximum, b 10, D raised until the background reaches 0.10; `linstretch -BP` at the histogram toe;
   pass 2 SP on the nebula mask's median, b 4, D 1, LP = background, HP < 1 when `hdr` is on) — alternatives `statstretch` (`pyscript Statistical_Stretch.py -median 0.10|0.25 -sigma 5 -linked`),
   `autoghs`, `modasinh`, `autostretch`, `hms`; highlights (GHS HP / Statistical Stretch `-hdr` / Wavescale HDR); NoiseXTerminator on the stretched starless
   (`--di --dn`, `--dc 0.95`; GraXpert fallback); palette (HOO classic / improved / foraxx / hoo_warm by `pm`; NBN port when `oiii_boost` != 1) or `rmgreen 3`; local contrast by `clahe`
   under a mask (150 and 40 px); `satu 0.25-0.5`; stars by the Seti Star Stretch (`mtf 0 0.004098 1`), magenta fix, DSA star reduction and a screen recombination
   (`pm "~(~$starless$*~$stars$)"`); black point at the toe and `icc_convert_to sRGB relative`; sRGB JPEG + 16-bit TIFF + the JSON sidecar.  The exporter computes and prints the
   **gates** of research/FINISH-SPEC.md (`gates.py`) and stores them in the sidecar.  The preview runs the same chain without the AI steps.  The outer 24 px of the stack are trimmed.

Which tool runs is decided at run time from the licences (`GET /api/tools`); the Linear panel in the
UI says what will be used.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q tests -m "not docker"      # unit tests: fixtures are real FITS headers and Siril .seq files
```
The `docker`-marked test runs the pipeline inside the image on a NAS dataset.

## Everything the 1.x application offered, on the Siril backend

| Feature | Where it lives now |
|---|---|
| Process tab, presets, live preview | the preview runs the export's finishing chain (`sirilpipe/pro.py`) without the AI noise reduction |
| Every processing control (stretch, highlights, colour, stars, detail, finish, colour grade) | steps of the finishing chain; nothing is ignored (`sirilpipe/post.py` PARAM_SPEC) |
| Super-resolution 1.5x / 2x | Siril Bayer drizzle at the chosen scale when picked explicitly; `auto` keeps 1x unless the data supports more |
| Device choice | *Integration & compute options* > Device (RC-Astro tools, GraXpert, ImageMM) |
| Calibration panel | masters found beside the session, DWARF CALI_FRAME matching, overrides |
| Frames tab | per-frame FWHM, roundness, background, star count from Siril's registration; overrides |
| Colour calibration from the stars' physics | Siril SPCC with the Siril SPCC database (narrowband mode for dual-band) |
| Gradient removal against a sky survey | NSNS DR0.2 through CDS hips2fits (`sirilpipe/skyref.py`), `bg_method` auto / reference |
| Auto-finish | published rules on the image's own measurements (no reference images) |
| Reference looks | the 58 reference photographs' statistics (`sirilpipe/looks.py`) grade the Experiments tab's finishing trials (`look_distance`) |
| AI star remover | StarXTerminator, else StarNet2 (`star_removal`) |
| Diagnostics | stage logs, reference frame, integration coverage map, Siril rejection map |
| Explore | plate solution from Siril's WCS (own solver as fallback), Gaia DR3 + SIMBAD annotation |
| Experiments (Optuna) | `sirilpipe/lab/`: finishing chain, linear restoration, registration & integration, star separation, gradient removal; synthetic datasets with truth; *From experiment* in the Process tab |
| ImageMM | `deconv_method = imagemm` (needs `IMAGEMM=cuda`), `restored_resolution` = r |
| CLI | `python -m sirilpipe run|stack|analyse|calibration|devices|tools` |
| `/api/v1` | unchanged |
