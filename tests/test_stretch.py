"""The alternative community stretches (stretch.py) and the numpy mirror of Siril's ght; the tests marked ``docker`` run the real tools (siril-cli, the siril-scripts Python
scripts in the pyscript virtualenv) and are part of the image's integration run."""
from __future__ import annotations

import numpy as np
import pytest

from sirilpipe import finish, stretch


def _scene(seed=3, h=240, w=320):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    neb = 0.35 * np.exp(-(((xx - 160) / 40.0) ** 2 + ((yy - 120) / 30.0) ** 2))
    return np.clip(0.02 + 0.5 * neb * 0.1 + rng.normal(0, 0.0005, (h, w)), 0, 1).astype(np.float32)


def test_ghs_slope_family_is_the_ghs_harmonic_family():
    # slope (1 + b D t)^(-1 - 1/b) from SP: monotone decreasing both sides, normalised 0 -> 0 and 1 -> 1, LP / HP linear continuations
    x = np.linspace(0, 1, 2001)
    y = finish.ghs(x, 6.0, 20.0, 0.1, lp=0.02, hp=0.8)
    assert y[0] == pytest.approx(0.0, abs=1e-6) and y[-1] == pytest.approx(1.0, abs=1e-6) and (np.diff(y) > 0).all()
    s = np.diff(y) / np.diff(x)
    assert s[int(0.1 * 2000)] > s[int(0.5 * 2000)] > 0 and s[int(0.02 * 2000)] > s[int(0.01 * 2000)] - 1e-3    # slope peaks at SP, falls on both sides, constant below LP
    assert np.allclose(s[: int(0.019 * 2000)], s[0], rtol=1e-3) and np.allclose(s[int(0.81 * 2000):], s[-1], rtol=1e-3)


@pytest.mark.docker
def test_numpy_ghs_mirror_equals_siril_ght(tmp_path):
    if not stretch.siril_available():
        pytest.skip("no siril-cli")
    x = np.concatenate([np.linspace(0, 0.1, 500), np.linspace(0.1, 1, 500)]).astype(np.float32)
    plane = np.tile(x[None, :], (4, 1))
    st = stretch.SirilStretcher(str(tmp_path))
    for D, b, sp, lp, hp in ((2.0, 6.0, 0.1, 0.0, 1.0), (3.0, 10.0, 0.05, 0.02, 1.0), (4.0, 3.0, 0.3, 0.1, 0.8), (1.5, 0.0, 0.2, 0.0, 1.0)):
        line = f"ght -D={D} -B={b:g} -SP={sp} -LP={lp} -HP={hp} -clipmode=clip -independent"
        out = st._run([plane], [[line]], "mirror")[0][0]
        assert np.abs(out - finish.ghs(plane, b, float(np.expm1(D)), sp, lp, hp)).max() < 2e-6


@pytest.mark.docker
@pytest.mark.parametrize("method", ["autoghs", "modasinh", "autostretch", "hms"])
def test_every_alternative_tool_runs_headless_and_lands_the_sky(tmp_path, method):
    if not stretch.siril_available():
        pytest.skip("no siril-cli")
    if method == "hms" and not (stretch.community_python() and stretch.community_script(stretch.HMS_SCRIPT)):
        pytest.skip("pyscript virtualenv / siril-scripts clone missing")
    img = _scene()
    u = np.clip((img - 0.0195) / 0.01 + 0.02, 0, 1).astype(np.float32)
    ctx = {"sky": float(np.median(u)), "noise": 0.01, "lum": u, "sky_aim": 0.10}
    st = stretch.SirilStretcher(str(tmp_path))
    out, rec = st.stretch_to_sky([u, u, u], ctx, method)
    assert rec["method"] == method and rec["commands"]
    assert 0.0 <= out[0].min() and out[0].max() <= 1.0
    assert float(np.median(out[0])) == pytest.approx(0.10, abs=0.02)
