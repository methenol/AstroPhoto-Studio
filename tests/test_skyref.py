

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
