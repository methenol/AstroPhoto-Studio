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
   Before restoring, the frames' whole-pixel origin on stack.fits' canvas is measured (Siril rounds its `-framing=max` box
   differently by up to 1 px, and an HDR stack's canvas is the union of the exposure groups' own boxes): the correlation
   decides when it is confident, and where a real sky leaves it flat - best 0.49, the whole search within 0.003 - the
   origins are tried from the frames' own box outward and the sub-pixel residual decides, the least move inside
   `imagemm_stage.ALIGN_TOLERANCE` winning.  Frames that already sit on the stack's pixels are therefore never shifted
   (C 20 2026-11-01 was refused for 1 px the correlation imagined, and delivered un-restored), and a session is refused
   only when no origin within the sizes' allowance puts them there; every origin measured is recorded with the
   preparation (`frames_flipped.origin_residuals`).
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
   The step then guards the stars' photometry, at the two points where a reference genuinely shares the image's units.
   A restoration that came from ImageMM is judged **as it is adopted**, against `stack.fits` on the same grid (the two
   were fitted to each other by `imagemm_stage.match_to_stack` and both then carry the same scale factor, so no scale
   has to be guessed); the star pass is judged at the end of the step, against the image as it found it.  A star's
   light is its aperture out to r=12 above its own neighbourhood sky -- a sum, so a deconvolution that merely moves
   light inward reads 1.00 -- and the stars whose light moved by more than a tenth, in total or in one channel alone
   (the coloured crown), are handed back from the reference pixels on a taper, because a deconvolution may move a
   star's light inward but may not delete it.  Nothing is judged against the frames after the white balance: a
   per-channel pedestal (background subtraction) cancels in a star's excess and a per-channel gain (white balance,
   SPCC) does not, and a scale read off two sky medians cannot tell them apart -- judging that way reported every star
   of C 20 as carrying 1.39 of the light the frames took, and would have handed the frame back un-white-balanced.
   `restore_meta.json` records the population's ratios, how many stars were refused and which mode decided
   (`star_photometry`); `star_photometry_guard = false` turns the check off.

   The same yardstick guards the **denoiser**, which has no reason to touch a star: the stack has already averaged
   that noise away, so whatever a denoiser removes from a star is light it should not have taken.  After the tool
   runs, the same isolated population is measured again against the image the step was handed.  If the median star
   has lost more than the tenth the restoration is allowed, the same tool is re-run weaker — the ladder from the
   requested strength to zero in `post.DENOISE_DESCALATION` steps — and the strongest pass whose stars are back
   inside tolerance is the one that ships, with its real strength written back into the step's recorded `params` and
   every attempt kept in `restore_meta.json`.  Measured on C 49's chain: GraXpert at the requested strength (0.80)
   left the stars at 0.75 of the light the frames took; the ladder tried 0.53 (0.82) and shipped 0.27, stars at 0.91.


   The **satellite remover** repaints rather than models, so it is judged on its pixels.  A fill gives itself away
   twice: nothing inside it varies, and it need not sit at the level of the sky it covers.  `post.painted_patch`
   compares the step's own input and output on a `post.PAINT_CELL` grid and hands the frames back the cells the tool
   made matte (or quiet) while their input was not *and* whose sky moved — past `post.PAINT_MIN` of the sky and past
   the cell median's own `post.PAINT_SE` standard errors, so a fill that left the sky alone is left as the tool made
   it.  The verdict then grows along the fill (`post.PAINT_GROW`), because a fill thins at its edges and leaves the
   sky's noise there, and refuses itself above `post.PAINT_MAX` of the frame, which is a gradient and not paint.
   Measured on IC 1848 (2026-11-01): `--sat-mode full --sensitivity 0.1` run alone on the session's own stack laid a
   7 × 150 px stripe 8.3 % of the sky below its surroundings, 94 % of it at one value — and the stretch, whose slope
   at the sky is 75-120, delivered that matte pedestal as a black bar.  The verdict is recorded as
   `params.painted_patch` on the step.

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
   (`process/finish/steps/NN_<step>.png` — shown through a linked autostretch, so they only carry the truth from the stretch onwards) and the
   verbatim commands in the sidecar (`linear_info.chain`, whose per-step `levels` also carry `sky_channels`, the median of each channel over
   `gates.sky_mask` — the numbers a colour cast is traced with, and which a layer holding no sky at all, the star layer, records nothing for):
   colour-gradient removal (`subsky 2`, mean model added back) and background neutralisation (per-channel offsets) on the linear starless;
   **Statistical Stretch** (Seti Astro, `pyscript Statistical_Stretch.py`, median 0.10 for a small subject / 0.25 frame-filling) as the default; or the **GHS two-pass stretch** by `ght` (pass 1 SP at the histogram's left half-maximum, b 10, D raised until the background reaches 0.10; `linstretch -BP` at the histogram toe;
   pass 2 SP on the nebula mask's median, b 4, D 1, LP = background, HP < 1 when `hdr` is on) — alternatives `statstretch` (`pyscript Statistical_Stretch.py -median 0.10|0.25 -sigma 5 -linked`),
   `autoghs`, `modasinh`, `autostretch`, `hms` (each after `linstretch -BP` at the histogram toe, its one parameter raised until the background is 0.10); highlights (GHS HP / Statistical Stretch `-hdr` / Wavescale HDR); NoiseXTerminator on the stretched starless
   (`--di --dn`, `--dc 0.95`; GraXpert fallback; Cosmic Clarity Denoise instead with `cc_denoise`) — the amount is a floor, not a ceiling: the sky
   the tool leaves is measured with the export's own statistic (`gates.pixel_noise`) and, while it sits above the `sky_pixel_noise` limit, the same
   tool is re-run harder up to its own maximum (at most `pro.DENOISE_ESCALATION` extra passes, the ladder ending on the ceiling), a harder pass
   that newly fails any other gate being refused and every attempt recorded in the chain; Cosmic Clarity Sharpen on the recombined image with `cc_sharpen`; dual-band palette by
   **Hubble_Palette_from_Dual-Band_OSC.py** itself (siril-scripts, Carlo Mollicone – AstroBOH; GUI-only, run headless inside Siril by
   `sirilpipe/hubble_palette_worker.py`, which selects the palette and formula preset and presses its *Apply*): HOO (default, Classic or
   Improved) and the Hubble-like SHO / HSO / OSH / OHS / HOS with its synthetic SII, presets Classic / Improved / Advanced / NonLinear S2
   (`hubble_formula`); SCNR after the Ha-in-green palettes (SHO, OHS); foraxx / hoo_warm by `pm`; NBN port as the palette `nbn` (its OIII boost `oiii_boost`); broadband `rmgreen 3`; local contrast by `clahe`
   under a mask (150 and 40 px) whose sky floor is the frame's own for one tile in from each edge, where that band is sky rather than the subject (`edge_sky_floor`); the GHS saturation stretch (`ght -sat`, D fitted so the subject's mean saturation reaches its target, under the range mask); stars by the Seti Star Stretch (`mtf 0 0.004098 1`), magenta fix, DSA star reduction and a screen recombination
   (`pm "~(~$starless$*~$stars$)"`); black point at the toe and `icc_convert_to sRGB relative`; sRGB JPEG + 16-bit TIFF + the JSON sidecar.  The exporter computes and prints the
   **gates** of research/FINISH-SPEC.md (`gates.py`) and stores them in the sidecar; the dataset card (`status.export_gates`) and the exports list
   (`GET /api/exports`, `gates.verdict`) show that verdict, so a render that failed its own gates is never handed out
   silently.  Where the chain holds a tool that can reach a failing number it is tried first - the delivered sky's
   noise, the sky's colour through SCNR, the sky's level through the final midtones, the star light under the
   deconvolution and the denoiser - and a gate no tool can answer is still delivered as a measurement, because the
   person who set the parameters decides what to do with the picture.
   `stars_coloured` measures a star's flux colour **above its own local sky** - the aperture's sum minus its area times the
   median of the ring around it, per channel (`gates._SAT_APER`/`_SAT_RING`), with the peak detection reading above the star
   layer's own median floor.  Summed raw, a brighter pedestal under the star is added equally to all three channels and
   (max - min) / max collapses with the star unchanged - the same red star reads 0.80 over a dark sky and 0.003 over a
   pedestal of 0.3, and sky pixels pass an absolute detection level once the pedestal rises - so the gate was reading the
   sky's brightness and calling it the stars' colour (M 31, 2026-11-01: identical stars, 0.101 with the sky a shade lower and
   0.097 against the 0.10 floor once the stretch's black clip lifted it).  A colour gate may not move with the light a star
   sits on, which is also why the clip is no longer judged against this floor.
   The preview runs the same chain without the AI steps.  The outer 24 px of the stack are trimmed.
   SCNR (`rmgreen 3`) is judged on the sky it leaves, by the gate's own number.  Its amount is decided on the
   *subject* (`autofinish`: median G - (R + B) / 2 over the subject mask) and Siril reduces green over the whole
   frame while preserving lightness - the green it lifts out comes back through red and blue - so a subject's cast
   gets subtracted from a neutral sky and leaves it magenta (M 31, 2026-11-01: the sky's median |a*| went 0.4 -> 13.4,
   6.7 x the gate's limit, and the render was delivered).  `pro.Finisher._scnr` measures the sky's median |a*| and
   |b*| over `gates.sky_mask` before and after the step, retries at `pro.SCNR_SKY_TRIES` fractions of the amount
   (the strongest pass whose sky is still inside `sky_chroma` ships, so the subject keeps its correction), and
   refuses the step - `status: refused`, both measurements in `params` - when every pass leaves the sky past the
   gate *and* `pro.SCNR_SKY_WORSE` times as coloured as it arrived.
   The stretched-image denoiser is judged on the sky it leaves too, and had no such rule until it was measured: over the
   fleet's exports (`output/.ab12/audit_exports.py --cast`, which reads the per-step `sky_channels` the chain records) it is
   the step that moves the sky's colour most wherever no palette runs - GraXpert's noise reduction on IC 434 walked the sky
   +31.5e-3 in (R-G)/(R+G) - and it was the one colour-changing step with neither a refusal rule nor a recorded before/after.
   `pro.Finisher._keep_denoise` wraps every backend (Cosmic Clarity, NXT, GraXpert): the sky's median |a*|, |b*| **and the
   spread of its four quadrants' signed tints** (`gates.quadrant_spread`, the number `sky_chroma_quadrants` gates) before
   and after, the step retried at the weakest strength that tool itself documents, and refused - pixels handed back,
   `status: refused`, both measurements in `params` with `refused_for` naming which of the two ruled - when every pass
   leaves either number past its gate *and* `pro.DENOISE_SKY_WORSE` times as coloured as it arrived.  Both are asked
   because |a*| over the sky is dominated by chroma *noise*: a denoiser that smooths the noise while tinting part of the
   sky *improves* it, and C 20 (2026-11-01) was delivered at |a*| 1.73 of 2.0 with its quadrants 1.97 apart of 1.5.  A
   refused tool is dropped from `self.denoiser` as well, so the
   noise ladder above cannot escalate the very step that coloured the sky.
   The last midtones pass is the sky's last chance at its own number: `brightness` / the Midtones slider goes through
   `mtf 0 m 1`, and the value the chain was handed can leave the sky above `sky_median` (M 31, Seestar, 2026-11-01:
   the sky left `finalise` at 0.134 against the window's 0.12 top, delivered with the gate failing and no record of
   which number had been missed).  `sky_median_guard` (on) solves `m` to the edge of the window with the same
   closed form `sky_levelling` uses - recorded as `params.sky_median_clamped = {from, to, brightness}` - and touches
   nothing on a render whose sky is already inside it.  Off, the slider is delivered exactly as asked.

   Statistical Stretch's **black clip** is judged the same way - on the sky, not on the picture's average.  The clip is
   a threshold at the sky minus `sigma` times the sky's own noise, so wherever the sky is locally noisier more pixels
   fall under it, and the toe of the curve (slope 75-120 at the sky) turns that into a lightness step: C 20 (EXP_15,
   2026-09-30) hands that step a sky flat to 0.1 % and takes back sky-mask cells spread +25 % / -29.8 %, which is the
   fleet's "wide rim offset" family (C 20 +36 %, M 31 -17 %, NGC 1499 +13/-12) - none of it in the linear frame, and
   the script has no background option to blame because it takes no background argument at all.  `pro.Finisher._stretch_stat`
   therefore runs the command twice, with and without `-noblackclip`, and ships the clip-free one only when its sky is
   `pro.STRETCH_CLIP_GAIN` times more even - `pro.sky_lump`, the worst `pro.SKY_LUMP_CELL` cell's deviation from the sky
   over `gates.sky_mask` - so unevenness that *is* the subject costs nothing to keep: a render loses its black point only
   when the black point was what cost it the sky.  Measured on the delivered files: C 20 31.8 with the clip and 6.4
   without (its right rim offset +43 % of sky with it), and both M 31 sessions also measured the clip as the cost.  Both
   commands and both measurements are in the step's `params` (`black_clip`, `sky_lump`, `tried`).

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
| Reference looks | the 58 reference photographs' statistics (`sirilpipe/looks.py`, data kept): no objective uses them - Auto-finish measures the image itself, the finishing task scores structural fidelity |
| AI star remover | StarXTerminator, else StarNet2 (`star_removal`); Cosmic Clarity DarkStar on request |
| Cosmic Clarity | Sharpen / Denoise / Super-Resolution / DarkStar / satellite trails, chosen in *Integration & options* and *Stars*; models from the Tools card |
| Diagnostics | stage logs, reference frame, integration coverage map, Siril rejection map |
| Explore | plate solution from Siril's WCS (own solver as fallback), Gaia DR3 + SIMBAD annotation |
| Experiments (Optuna) | `sirilpipe/lab/`: registration & integration, satellite trails, gradient removal, linear restoration, star separation, finishing chain, super-resolution; paired no-reference objectives with constraints (`lab/objectives.py`), conditional search spaces; synthetic datasets with truth; *From experiment* in the Process tab |
| Hyper-tune each stage | `sirilpipe/hypertune.py`: Run everything with a study of every stage before it runs (`stack_params.hypertune`, `hypertune_trials`), in the job's own process; studies in the Experiments tab, report in `hypertune.json` |
| ImageMM | VeraLux ImageMM v1.0.0 headless: `deconv_method = imagemm` (needs `IMAGEMM=cuda`), `restored_resolution` = r, `imagemm_*` = the script's parameters |
| CLI | `python -m sirilpipe run|stack|analyse|calibration|devices|tools` |
| `/api/v1` | unchanged |
