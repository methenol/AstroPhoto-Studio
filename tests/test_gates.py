"""``sky_uniformity``: the border's sky floor over the smooth surface fitted through the mesh of sky floors - a gross one-sided border offset."""
import numpy as np

from sirilpipe import gates


def _picture(band=0.0, blob=0.0, size=600):
    """A finished picture: sky at 0.10 with its noise, an optional band across the top of the frame, an optional extended subject in the middle.  The band is two cell
    rows of the gate's mesh deep - the scale of the canvas border bands (300 - 1000 px) the gate is for."""
    rng = np.random.default_rng(4)
    lum = 0.10 + rng.normal(0, 0.002, (size, size)).astype(np.float32)
    if blob:
        y, x = np.mgrid[0:size, 0:size]
        lum += blob * np.exp(-((y - size / 2) ** 2 + (x - size / 2) ** 2) / (2 * (size / 8) ** 2)).astype(np.float32)
    img = np.stack([lum] * 3)
    if band:
        d = 2 * size // gates.SKY_BORDER_MESH
        img[:, :d, :] = np.clip(img[:, :d, :] + band, 0.0, 1.0)
    return np.clip(img, 0.0, 1.0)


def test_a_border_band_fails_and_the_same_picture_without_it_passes():
    clean = gates.compute(_picture())["sky_uniformity"]
    banded = gates.compute(_picture(band=0.10))["sky_uniformity"]
    assert clean["pass"], clean
    assert not banded["pass"], banded
    assert banded["value"] > 4 * max(clean["value"], 1e-4)


def test_an_extended_subject_in_the_middle_does_not_fail_it():
    """The 12 x 12 mesh this gate replaced failed nebula frames: a spiral arm is not a seam."""
    g = gates.compute(_picture(blob=0.25))["sky_uniformity"]
    assert g["pass"], g


def _star_field(colour=(1.0, 0.62, 0.2), pedestal=0.0, size=160, step=16, peak=0.6):
    """A star layer - the stars of a screen blend, as ``gates.measure_stars`` is handed them: identical Gaussian stars of one
    colour on a neutral ``pedestal``.  Whether the sky under the star sits at 0 or at 0.3, the star is the same object, so a
    colour measurement of the star must not move with it."""
    rng = np.random.default_rng(7)
    a = (pedestal + rng.normal(0, 0.002, (3, size, size))).astype(np.float32)
    sig = 3.0 / 2.355                                                # the pipeline's own FWHM (stack_meta's 3 px)
    r = int(np.ceil(4 * sig))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sig ** 2)).astype(np.float32)
    for y in range(step, size - step, step):
        for x in range(step, size - step, step):
            for c in range(3):
                a[c, y - r:y + r + 1, x - r:x + r + 1] += peak * colour[c] * g
    return np.clip(a, 0.0, 1.0)


def test_the_stars_colour_is_measured_above_its_own_local_sky():
    """M 31 (2026-11-01): the same stars, untouched, read 0.101 against the 0.10 ``stars_coloured`` floor with the sky at its
    usual level and 0.097 - a failed gate - once dropping the stretch's black clip raised the pedestal under them.  Summing
    the aperture raw charges the star with the sky it sits on, and a neutral pedestal adds the same light to all three
    channels, so (max - min) / max shrinks as the sky brightens: the gate was reading the sky's brightness and calling it the
    stars' colour.  Aperture photometry cancels exactly that with a local sky, and so must the gate."""
    flat = gates.measure_stars(_star_field(), 5e-4)["saturation"]
    raised = gates.measure_stars(_star_field(pedestal=0.3), 5e-4)["saturation"]
    assert flat is not None and raised is not None
    assert flat > 0.6 and raised > 0.6, (flat, raised)                # a red star, still called red at either sky level
    assert abs(flat - raised) <= 0.02, (flat, raised)


def test_a_neutral_star_reads_less_coloured_than_a_red_one_at_either_sky_level():
    """The invariance above is only worth anything if the measurement still sees colour: a grey star of the same brightness
    has to stay well below the red one, with the sky under it at either level."""
    for ped in (0.0, 0.3):
        red = gates.measure_stars(_star_field(pedestal=ped), 5e-4)["saturation"]
        grey = gates.measure_stars(_star_field(colour=(1.0, 1.0, 1.0), pedestal=ped), 5e-4)["saturation"]
        assert grey < red - 0.2, (ped, grey, red)
