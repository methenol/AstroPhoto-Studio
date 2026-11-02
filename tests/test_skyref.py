

def test_fetch_falls_back_to_the_surveys_own_hips_server_when_cds_does_not_answer(monkeypatch):
    """CDS hips2fits timed out on every host (2026-11-01, alasky and both alaskybis mirrors): the NSNS HiPS is read from the survey's
    own server instead, and the error names both sources only when neither answered."""
    import numpy as np
    from sirilpipe import skyref
    calls = []

    def cds(*a, **k):
        calls.append("cds")
        raise TimeoutError("timed out")

    monkeypatch.setattr(skyref, "fetch_hips2fits", cds)
    monkeypatch.setattr(skyref, "fetch_direct", lambda *a, **k: (calls.append("direct"), np.zeros((2, 2, 3), np.float32))[1])
    maps, source = skyref.fetch(None, (16, 16))
    assert maps.shape == (2, 2, 3) and calls == ["cds", "direct"] and "simg.de" in source

    def down(*a, **k):
        raise OSError("unreachable")

    monkeypatch.setattr(skyref, "fetch_direct", down)
    import pytest
    with pytest.raises(RuntimeError, match="CDS hips2fits.*own HiPS server"):
        skyref.fetch(None, (16, 16))
    assert set(skyref.HIPS_SERVICE) == {name for name, _ in skyref.SURVEYS}


def test_local_linear_regression_has_no_edge_bias():
    """A plane comes back exactly, corners included, where a weighted mean flattens it toward the interior; NaN gaps are filled from around."""
    import numpy as np
    from sirilpipe import skyref
    yy, xx = np.mgrid[0:60, 0:90].astype(np.float32)
    plane = (0.02 + 0.001 * xx - 0.0005 * yy).astype(np.float32)
    holes = plane.copy()
    holes[20:30, 40:55] = np.nan
    out = skyref._local_linear(holes, 15.0)
    assert np.allclose(out, plane, atol=1e-5)
    mean = skyref._nan_blur(plane, 15.0)
    assert abs(float(mean[0, 0] - plane[0, 0])) > 100 * abs(float(out[0, 0] - plane[0, 0]))


def test_star_halos_leave_the_survey_and_emission_around_a_star_stays():
    """The survey's scattered light around a bright star (a core and an r^-2 aureole: NSNS [OIII] around zeta Ori) is taken out of its map;
    a plateau of emission around another star (an HII region around its ionising star) is not an aureole and stays."""
    import numpy as np
    from sirilpipe import skyref
    rng = np.random.default_rng(1)
    h, w = 240, 160
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (1.0 + 0.05 * rng.standard_normal((h, w))).astype(np.float32)
    r1 = np.hypot(yy - 60, xx - 80)
    halo = np.where(r1 < 4, 300.0, 300.0 * (4.0 / np.maximum(r1, 4.0)) ** 2).astype(np.float32)
    r2 = np.hypot(yy - 180, xx - 80)
    plateau = np.where(r2 < 50, 12.0, 0.0).astype(np.float32)
    m = base + halo + plateau
    refs = np.stack([m, base.copy(), base.copy()], -1)
    out, done = skyref.remove_star_halos(refs, [("bright", 1.9, 80.0, 60.0), ("ionising", 3.8, 80.0, 180.0)])
    assert [(d["star"], d["map"]) for d in done] == [("bright", 0)]
    assert 1.6 < done[0]["alpha"] < 2.4
    ring = (r1 > 8) & (r1 < 40)
    assert float(np.median(out[..., 0][ring])) < 1.0 + 0.5 and float(np.median(m[ring])) > 3.0
    truth = base + plateau                                                # the plateau is emission: it stays, only the aureole reaching it goes
    assert float(np.median(np.abs(out[..., 0] - truth)[r2 < 45])) < 0.1
    assert float(np.median(out[..., 0][r2 < 45])) > 12.0
    assert float(out[..., 0].min()) > float(base.min()) - 1e-6            # nothing below the background


def test_line_profiles_find_row_bands_and_leave_noise_alone():
    """A wide row band and an edge ramp in the residual come back in the row profile; pure noise gives a profile below one line's
    standard error (the smoothness is chosen by the unbiased risk estimate); a nebula patch covering less than half the lines does not move them."""
    import numpy as np
    from sirilpipe import skyref
    rng = np.random.default_rng(2)
    H, W = 800, 600
    yy = np.arange(H, dtype=np.float32)[:, None]
    band = 4e-4 * np.exp(-0.5 * ((yy - 400) / 40) ** 2) - 3e-4 * np.clip(1 - yy / 120, 0, None)   # ~6 and ~4 line standard errors
    noise = (3e-3 * rng.standard_normal((H, W, 3))).astype(np.float32)
    resid = noise + band[..., None].astype(np.float32)
    resid[100:300, 50:250] += 5e-3                                          # a patch the survey did not explain
    mask = np.zeros((H, W), bool)
    rows, cols, info = skyref.line_profiles(resid, mask)
    truth = (band - np.median(band)).ravel()
    se = 1.2533 * 1.4826 * 3e-3 * 1.2533 / 8 / np.sqrt(W // 8)               # one line's standard error (block medians of 8 x 8 px)
    assert np.abs(rows[:, 1] - truth)[8:-8].max() < 1.2e-4 and float(rows[400, 1]) > 0.6 * float(truth[400])
    assert np.abs(cols).max() < se
    r0, c0, _ = skyref.line_profiles(noise, mask)
    assert np.abs(r0).max() < se and np.abs(c0).max() < se
