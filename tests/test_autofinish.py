"""Unit tests of sirilpipe.autofinish: the rules as functions, and a full run on a synthetic session."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_finish import synthetic_linear                      # noqa: E402

import shutil

from sirilpipe import autofinish as af, finish, fitsio, post, stretch   # noqa: E402


@pytest.fixture()
def session(tmp_path, monkeypatch):
    d = tmp_path / "s"
    d.mkdir()
    lin = synthetic_linear(h=500, w=700, nstars=120, blob=0.05)
    hdr = {"OBJECT": "NGC 1", "ROWORDER": "BOTTOM-UP"}
    fitsio.write_rgb(str(d / "stack.fits"), lin, hdr)
    starless = np.minimum(lin, np.float32(0.04))
    fitsio.write_rgb(str(d / "linear.fits"), lin, hdr)
    fitsio.write_rgb(str(d / "starless.fits"), starless, hdr)
    fitsio.write_rgb(str(d / "stars.fits"), np.maximum(lin - starless, 0), hdr)
    json.dump({"tool": "starnet2", "stage": "linear", "stars": {"count": 900, "per_megapixel": 450.0}},
              open(d / "starnet_meta.json", "w"))
    post._PREVIEW_CACHE.clear()
    return str(d)


@pytest.mark.docker
def test_run_writes_report_with_a_measurement_per_decision(session):
    """Autofinish on a synthetic session through the real Siril chain: published values only, a reason and a measurement per decision."""
    if not stretch.siril_available():
        pytest.skip("no siril-cli")
    out = af.run(session, {"object": "NGC 1", "filter": "IRCUT"}, {}, None, None)
    assert os.path.exists(os.path.join(session, "autofinish.json"))
    p, rep = out["params"], out["report"]
    assert set(p) >= set(post.DEFAULTS)
    assert p["stretch_method"] == "ghs2" and p["stretch"] == 0.10 and p["hdr"] in (0.0, 1.0) and p["black_point"] == 0.0
    assert p["saturation"] in (1.25, 1.3, 1.5) and p["star_reduction"] in (0.2, 0.3, 0.4, 0.5, 0.6) and p["scnr"] in (0.0, 0.6)
    assert 0.6 <= p["luminance_denoise"] <= 0.85 and p["chroma_denoise"] == 0.95
    assert p["grade_temperature"] == 0 and p["grade_contrast"] == 0
    for s in rep["steps"]:
        assert s["why"] and "to" in s and "from" in s
    assert {"stretch", "hdr", "luminance_denoise", "saturation", "star_reduction"} <= {s["step"] for s in rep["steps"]}
    assert rep["class"].startswith("broadband")


def test_run_references_only_defined_names():
    """Every global name used in autofinish.run must exist at import time (a NameError here reached production once)."""
    import ast
    import sirilpipe.autofinish as af
    src = ast.parse(open(af.__file__).read())
    fn = next(n for n in src.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    local = {a.arg for a in fn.args.args} | {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    local |= {n.name for n in ast.walk(fn) if isinstance(n, ast.FunctionDef)}
    for n in ast.walk(fn):
        if isinstance(n, (ast.FunctionDef, ast.Lambda)):
            local |= {a.arg for a in n.args.args + n.args.kwonlyargs} | {a.arg for a in (n.args.vararg, n.args.kwarg) if a}
    for n in ast.walk(fn):
        if isinstance(n, ast.ImportFrom):
            local |= {a.asname or a.name for a in n.names}
        if isinstance(n, ast.Import):
            local |= {(a.asname or a.name).split(".")[0] for a in n.names}
        if isinstance(n, ast.comprehension):
            local |= {x.id for x in ast.walk(n.target) if isinstance(x, ast.Name)}
        if isinstance(n, ast.ExceptHandler) and n.name:
            local.add(n.name)
    import builtins
    used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    missing = sorted(u for u in used if u not in local and not hasattr(af, u) and not hasattr(builtins, u))
    assert not missing, missing
