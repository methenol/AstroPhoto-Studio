# Patches to the Siril binaries

Applied by the Dockerfile to the extracted Siril AppImage (`/opt/siril/usr/bin/siril-cli` and `siril`). Each patch
checks the exact bytes it replaces, so a different Siril build fails the build instead of being patched blindly.

- `multi_save_list_walk.py` - **seqsplit_cfa (and every multi-output sequence command: seqextract_*, seqsplit)
  can hang forever.** In Siril 1.4.0 - 1.4.5, `multi_save()` (`src/core/processing.c`) searches the list of
  finished images for its own entry with `list = g_list_next(multi_args->processed_images)`. That re-reads the
  head's successor on every pass, so the search finds only entries 0 and 1. If 24 worker threads have pushed
  more results than that before a thread saves, the thread spins at 100 % CPU **holding the sequence lock**, and
  the other 23 workers block on it. The pattern is intermittent: the same `seqsplit_cfa pp_light` ran in
  2.5 s for the banding stage and then hung a minute later for the per-frame background stage (C 31,
  2026-11-12: 72 of 156 frames split, one thread looping at `multi_save+159` for 1.5 h). Sh2-205
  (2026-11-11, 7 h) hung the same way. Upstream master fixed it as `g_list_next(list)`; no release has the
  fix yet (checked 1.4.5). The patch is the same fix in machine code: one ModRM byte, `mov 0x8(%rdi),%rax`
  -> `mov 0x8(%rax),%rax`. Drop it once `SIRIL_VER` is a release whose `multi_save` advances `list` (the
  script then fails the build, since the buggy loop is no longer found).
