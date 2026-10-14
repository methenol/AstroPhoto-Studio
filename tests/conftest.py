"""Shared fixtures: real FITS headers of the three rigs (tests/fixtures/*.json) served through a patched
``astropy.io.fits.getheader`` so the dataset/calibration code runs on empty files with the right names."""
from __future__ import annotations

import json
import os

import pytest
from astropy.io import fits

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def load(name: str) -> dict:
    with open(os.path.join(FIX, name), encoding="utf-8") as f:
        return json.load(f)


def make_header(cards: dict) -> fits.Header:
    h = fits.Header()
    for k, v in cards.items():
        h[k] = v
    return h


@pytest.fixture()
def fake_headers(monkeypatch):
    """Register {basename: cards} and patch ``fits.getheader`` (used by dataset.py and calibration.py)."""
    table: dict[str, dict] = {}
    real = fits.getheader

    def getheader(path, *a, **kw):
        base = os.path.basename(str(path))
        if base in table:
            return make_header(table[base])
        return real(path, *a, **kw)
    monkeypatch.setattr(fits, "getheader", getheader)
    return table


def touch(path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").close()
    return path


def pytest_configure(config):
    config.addinivalue_line("markers", "docker: integration test that needs siril-cli and the NAS data (run inside the image)")
