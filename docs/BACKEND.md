# The Siril backend (branch `backend`)

AstroPhoto Studio 2.x processes with Siril 1.4 (headless), GraXpert, StarNet2 and the RC-Astro tools
(BlurXTerminator, NoiseXTerminator, StarXTerminator), with SetiAstro's Cosmic Clarity models (Sharpen, Denoise,
Super-Resolution, DarkStar, satellite-trail removal) as an optional path next to them. Everything is installed inside the Docker image;
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
| `IMAGEMM=cuda\|cpu\|off` | add PyTorch for the optional ImageMM multi-frame restoration and for Cosmic Clarity (+ torchvision, onnxruntime; cuda adds ~4 GB) |
| `LOCAL_COPY=true` | jobs read the subs from a copy on the output volume (NAS shares): copied when a stage first reads them, cleared at the start of every job and deleted at its end |

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
   data (GraXpert denoise 0.8 .. 0.6, or Siril, as fallbacks).  `deconv_method = imagemm` runs VeraLux ImageMM (`sirilpipe/veralux_imagemm.py`) on the subs instead: the stack's own calibrated, cosmetic-corrected Bayer-drizzle frames with its registration, per-colour saturation masks (pixels within `imagemm_bloom_grow` px of them left out, and within 0.75 x the equivalent radius of a large saturated patch), then the multi-frame restoration with per-frame kernels (plus the optics' static asymmetry and an extended wing to 80 px joined from unsaturated and saturated stars, with its per-channel colour and lobe measured on the saturated stars); a saturated star starts as a point at the light PSF photometry of its unsaturated wings measures, centred on its missing data, and where the frames' data are sparse it becomes the restored image's PSF at its wings' colour, joined to the restoration with the restoration's own structure kept and the PSF's tail fading on; saturated galaxy nuclei get their colour from their wings; the result is put on the stack's photometric scale per channel from the unsaturated stars' flux ratios (sky offset from the median); no BlurXTerminator.
6. **Stars** — StarXTerminator on the linear data (or StarNet2 `--linear`; Cosmic Clarity DarkStar with `star_removal = darkstar`), no unscreen: `stars = linear - starless`.
   **Linear tools are the user's choice** (*Integration & options*): deconvolution per region, `deconv_method` for the objects
   and `deconv_stars_method` for the stars (a request naming only `deconv_method` deconvolves the stars with it too; the former
   `graxpert_obj / graxpert_stellar / graxpert_both` values stay accepted).  Each is auto (BlurXTerminator when licensed, else
   Siril RL) / bxt / rl / imagemm / none / **graxpert** (GraXpert's AI deconvolution models, `-strength`
   = the Deconvolution slider, `-psfsize` = `graxpert_psfsize`, or when 0 the star FWHM measured on the linear image, GraXpert's
   "Image FWHM (in pixels)"; BlurXTerminator's correct-only pass runs only with auto / bxt; measured on Sh2-131 at strength 0.5,
   PSF 4.48 px: the objects model keeps the flux and leaves the star FWHM almost unchanged, the stars model takes the FWHM from
   4.5 to 2.0 px and keeps 19-27 % of the stars' flux, independent of the data's scale).  One tool for both regions runs once on
   the whole image; two tools run objects first, then stars: BXT with the other region's sharpening at 0, GraXpert with its
   objects or stars model, Siril RL blended in through a star mask (each star Siril's `findstar` fits with a Moffat profile,
   out to where the profile falls to the sky noise, softened by the median star's sigma).  ImageMM for one region only: the
   other region is taken from stack.fits (Lanczos to ImageMM's grid) through that mask, measured on the stack, then
   deconvolved by its own tool. `denoise_method` auto (NXT, else GraXpert, else Siril) / nxt / graxpert / siril / none. A tool chosen
   explicitly that cannot run is recorded and nothing else runs in its place; the same holds for DarkStar, Cosmic Clarity's steps,
   the Hubble palette script and a super-resolution scale Cosmic Clarity has no model for (2x / 3x / 4x only).

   **Cosmic Clarity** (optional, instead of or in addition to the RC-Astro tools): siril-scripts' `CosmicClarity_Native.py`
   ("Cosmic Clarity AI4 for Siril", engines and models © SetiAstro) — its processing functions run headless by
   `sirilpipe/cosmic_clarity_worker.py` in Siril's pyscript environment, which is given the image's PyTorch / torchvision /
   onnxruntime (a `.pth` file). `satellite_removal` runs on the stack before the gradient, DarkStar (`star_removal =
   darkstar`) on the linear image; **Sharpen** (`cc_sharpen` off / both / stellar / nonstellar, `cc_stellar_amount`,
   `cc_nonstellar_amount`, the script's defaults 0.5) and **Denoise** (`cc_denoise`, model `cc_denoise_model` full / lite /
   walking, strengths = the two noise sliders) are settings in *Integration & options* applied to the stretched image, and
   **Super-Resolution** (`superres_method = cosmic_clarity`, 2/3/4x from `scale`; the stack integrates at 1x) enlarges the
   finished image in the export. Why not on the linear image: the script stretches linear input itself and inverts the stretch
   afterwards, and on this pipeline's background-subtracted linear data (sky ~1e-4 above the minimum, stars up to 1) that
   inversion is so steep round the stars that Sharpen came back with 6.2x the flux, Denoise 2.2x, Super-Resolution 0.75x
   (Sh2-131 two nights, 2026-10-24); on the stretched image they keep it (0.96 / 1.00 / 1.00), and DarkStar's linear starless
   matched StarXTerminator's (nebula p99 above the sky 8.8e-5 vs 8.3e-5 after BXT + NXT). DarkStar is sensitive to the linear
   image's noise: after GraXpert's denoise (sky noise 2.4e-5) its starless kept 35 % of the nebula StarXTerminator's did, after
   NXT (1.0e-5) all of it. The models (2.4 GB, the script's own *Download Models*: SetiAstro's files mirrored on HuggingFace)
   are installed from the Tools card into `OUTPUT_DIR/.home/.local/share/siril/cosmic_clarity`; DarkStar is published as ONNX
   only and runs through onnxruntime-gpu (CUDA 12 build).
7. **Auto-finish** — `autofinish.py` picks among published values by stated rules (small vs frame-filling target, clipping -> HDR, dual-band vs broadband -> palette / SCNR,
   star density -> reduction, SNR -> NXT strength); every decision is in the report with its rule.
8. **Export** — `pro.py`, the professional chain, every step a Siril command, a pyscript or an rc-astro call, a picture saved after each step
   (`process/finish/steps/NN_<step>.png`) and the verbatim commands in the sidecar (`linear_info.chain`):
   colour-gradient removal (`subsky 2`, mean model added back) and background neutralisation (per-channel offsets) on the linear starless;
   **Statistical Stretch** (Seti Astro, `pyscript Statistical_Stretch.py`, median 0.10 for a small subject / 0.25 frame-filling) as the default; or the **GHS two-pass stretch** by `ght` (pass 1 SP at the histogram's left half-maximum, b 10, D raised until the background reaches 0.10; `linstretch -BP` at the histogram toe;
   pass 2 SP on the nebula mask's median, b 4, D 1, LP = background, HP < 1 when `hdr` is on) — alternatives `statstretch` (`pyscript Statistical_Stretch.py -median 0.10|0.25 -sigma 5 -linked`),
   `autoghs`, `modasinh`, `autostretch`, `hms` (each after `linstretch -BP` at the histogram toe, its one parameter raised until the background is 0.10); highlights (GHS HP / Statistical Stretch `-hdr` / Wavescale HDR); NoiseXTerminator on the stretched starless
   (`--di --dn`, `--dc 0.95`; GraXpert fallback; Cosmic Clarity Denoise instead with `cc_denoise`); Cosmic Clarity Sharpen on the recombined image with `cc_sharpen`; dual-band palette by
   **Hubble_Palette_from_Dual-Band_OSC.py** itself (siril-scripts, Carlo Mollicone – AstroBOH; GUI-only, run headless inside Siril by
   `sirilpipe/hubble_palette_worker.py`, which selects the palette and formula preset and presses its *Apply*): HOO (default, Classic or
   Improved) and the Hubble-like SHO / HSO / OSH / OHS / HOS with its synthetic SII, presets Classic / Improved / Advanced / NonLinear S2
   (`hubble_formula`); SCNR after the Ha-in-green palettes (SHO, OHS); foraxx / hoo_warm by `pm`; NBN port as the palette `nbn` (its OIII boost `oiii_boost`); broadband `rmgreen 3`; local contrast by `clahe`
   under a mask (150 and 40 px); the GHS saturation stretch (`ght -sat`, D fitted so the subject's mean saturation reaches its target, under the range mask); stars by the Seti Star Stretch (`mtf 0 0.004098 1`), magenta fix, DSA star reduction and a screen recombination
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
| Super-resolution 1.5x / 2x | Siril Bayer drizzle at the chosen scale when picked explicitly; `auto` keeps 1x unless the data supports more; or Cosmic Clarity Super-Resolution 2x / 3x / 4x of the finished image (`superres_method`) |
| Device choice | *Integration & compute options* > Device (RC-Astro tools, GraXpert, ImageMM) |
| Calibration panel | masters found beside the session, DWARF CALI_FRAME matching, overrides |
| Frames tab | per-frame FWHM, roundness, background, star count from Siril's registration; overrides |
| Colour calibration from the stars' physics | Siril SPCC with the Siril SPCC database (narrowband mode for dual-band), online Gaia DR3; the chunk of the Gaia catalogue covering the field (1 of 48) is fetched once and cached under `.home/.local/share/siril/siril_cat1_healpix8_xpsamp` so later fields in that region calibrate offline (`-catalog=localgaia`); a failed SPCC is retried once, then the stars white balance + neutralisation |
| Gradient removal against a sky survey | NSNS DR0.2 through CDS hips2fits (`sirilpipe/skyref.py`), `bg_method` auto / reference |
| Auto-finish | published rules on the image's own measurements (no reference images) |
| Reference looks | the 58 reference photographs' statistics (`sirilpipe/looks.py`) grade the Experiments tab's finishing trials (`look_distance`) |
| AI star remover | StarXTerminator, else StarNet2 (`star_removal`); Cosmic Clarity DarkStar on request |
| Cosmic Clarity | Sharpen / Denoise / Super-Resolution / DarkStar / satellite trails, chosen in *Integration & options* and *Stars*; models from the Tools card |
| Diagnostics | stage logs, reference frame, integration coverage map, Siril rejection map |
| Explore | plate solution from Siril's WCS (own solver as fallback), Gaia DR3 + SIMBAD annotation |
| Experiments (Optuna) | `sirilpipe/lab/`: registration & integration, satellite trails, gradient removal, linear restoration, star separation, finishing chain, super-resolution; paired no-reference objectives with constraints (`lab/objectives.py`), conditional search spaces; synthetic datasets with truth; *From experiment* in the Process tab |
| Hyper-tune each stage | `sirilpipe/hypertune.py`: Run everything with a study of every stage before it runs (`stack_params.hypertune`, `hypertune_trials`), in the job's own process; studies in the Experiments tab, report in `hypertune.json` |
| ImageMM | VeraLux ImageMM v1.0.0 headless: `deconv_method = imagemm` (needs `IMAGEMM=cuda`), `restored_resolution` = r, `imagemm_*` = the script's parameters |
| CLI | `python -m sirilpipe run|stack|analyse|calibration|devices|tools` |
| `/api/v1` | unchanged |
