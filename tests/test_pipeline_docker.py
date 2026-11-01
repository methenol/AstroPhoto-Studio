"""Integration: the real pipeline on 30 subs of a DWARF M 42 night.  Runs only where ``siril-cli`` exists and the
data is mounted (inside the image):

    docker run --rm -v $PWD:/app -v /mnt/Astro_Photographs:/data/images:ro -w /app --entrypoint python \
        astrophoto-studio:cuda -m pytest tests/test_pipeline_docker.py -m docker -q
"""
from __future__ import annotations

import json
import os
import shutil

import pytest

from sirilpipe import config, fitsio
from sirilpipe.session import Session

NIGHT = ("/data/images/Dwarf/DWARF_RAW_TELE_M 42_EXP_15_GAIN_60_2026-10-10-04-11-32-312",
         "/mnt/Astro_Photographs/Dwarf/DWARF_RAW_TELE_M 42_EXP_15_GAIN_60_2026-10-10-04-11-32-312")
FOLDER = next((p for p in NIGHT if os.path.isdir(p)), None)
pytestmark = [pytest.mark.docker,
              pytest.mark.skipif(not (shutil.which("siril-cli") or os.path.exists(config.siril_bin())) or FOLDER is None,
                                 reason="needs siril-cli and the DWARF M 42 data")]


def test_stack_30_subs(tmp_path):
    s = Session([FOLDER], str(tmp_path), limit=30)
    seen = []
    s.run_analysis(1.0, lambda i, n, m: seen.append(m.split(" (")[0]))
    rows = json.load(open(os.path.join(s.dir, "frames.json")))
    assert len(rows) == 30 and sum(r["included"] for r in rows) >= 15
    meta = s.run_stack({}, lambda i, n, m: seen.append(m.split(" (")[0]))
    s.plate_solve()
    arr, hdr = fitsio.read_rgb(os.path.join(s.dir, "stack.fits"))
    assert arr.shape[0] == 3 and arr.shape[2] == meta["shape"][0] and arr.min() >= 0 and arr.max() <= 1.0
    assert hdr.get("PLTSOLVD") and s.meta["plate_solve"]["ok"]
    assert meta["scale"] == 1.0 and meta["mode"] == "drizzle" and meta["crop_info"]["method"] == "frame coverage"
    for stage in ("Scanning frames", "Calibrating", "Registering (pass 1/2)", "Applying registration", "Stacking",
                  "Cropping", "Plate solving"):
        assert stage in seen, stage
    # border rows are real data, not the black of the -maximize canvas
    assert (arr[:, 0, :] > 0).all() and (arr[:, -1, :] > 0).all() and (arr[:, :, 0] > 0).all() and (arr[:, :, -1] > 0).all()
