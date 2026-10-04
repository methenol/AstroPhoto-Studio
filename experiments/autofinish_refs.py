"""Build astrophoto/data/autofinish_refs.json: the reference looks Auto-finish tunes towards.

    python experiments/autofinish_refs.py [--cache DIR]

Every reference is a freely licensed astrophotograph on Wikimedia Commons, chosen by eye for each
target: well-processed, natural-colour or HOO-style images (the looks a one-shot-colour camera with
or without a dual-band filter can reach).  SHO / Hubble-palette, infrared, wide-field-with-landscape
and light-polluted or green-cast images were left out, and so were sets that framed the target
unlike a wide-field telescope does (IC 1396: close-ups of the globule only; it uses the emission
nebulae's references instead).  The images are downloaded at 1200 px (the Commons thumbnail
service), measured with ``autofinish.look_stats`` and only the statistics and the attributions are
kept: no image is stored in the repository.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from astrophoto.autofinish import REF_FILE, aggregate, look_stats  # noqa: E402

UA = {"User-Agent": "AstroPhotoStudio-reference-fetch/1.0 (research script)"}

# target -> label, class, names it goes by (the FITS OBJECT header), Commons file titles
TARGETS = {
    "m42": {"label": "M 42, the Orion Nebula", "class": "emission", "names": ["M42", "M 42", "NGC 1976", "Orion Nebula", "M43"],
        "titles": ["Orion nebula and Running Man nebula.jpg",
                   "M42 - Orinonebel-RunningMan - reworked - Flickr - cfaobam.jpg",
                   "Orion Nebula, NGC1977 - Astrophotography.jpg"]},
    "ngc281": {"label": "NGC 281, the Pacman Nebula", "class": "emission", "names": ["NGC 281", "Sh2-184", "Pacman Nebula"],
        "titles": ["PacMan Nebula, NGC 281.jpg",
                   "NGC 281 - The Pacman Nebula (50862556267).jpg",
                   "Pacman Nebula NGC 281 (Giovanni Barbarino).jpg",
                   "NGC 281 - The Pacman Nebula.jpg"]},
    "ngc6992": {"label": "NGC 6992, the Eastern Veil", "class": "snr", "names": ["NGC 6992", "NGC 6995", "C 33", "Caldwell 33", "IC 1340", "Eastern Veil"],
        "titles": ["NGC 6992 Eastern Veil Nebula.jpg",
                   "The Eastern Veil Nebula (NGC6992).jpg",
                   "The Eastern Veil Nebula (48662530613).png",
                   "The Eastern Veil Nebula - Flickr - astrophotography andy.jpg",
                   "NGC6992 Veil Nebula Supernova Remnant from the Mount Lemmon SkyCenter Schulman Telescope courtesy Adam Block.jpg"]},
    "ngc7000": {"label": "NGC 7000, the North America Nebula", "class": "emission", "names": ["NGC 7000", "C 20", "Caldwell 20", "North America Nebula", "North American Nebula"],
        "titles": ["NGC 7000 Nebula.jpg",
                   "North America Nebula (NGC 7000 or Caldwell 20).jpg",
                   "NGC7000 North American Nebula by Adam Block.jpg"]},
    "m31": {"label": "M 31, the Andromeda Galaxy", "class": "galaxy", "names": ["M31", "M 31", "NGC 224", "Andromeda Galaxy"],
        "titles": ["M31-Andromede-16-09-2023-Hamois.jpg",
                   "Galaxie d'Andromède M31.jpg",
                   "Andromeda Galaxy (M31).jpg",
                   "The Andromeda galaxy - Flickr - astrophotography andy.jpg",
                   "M31 - Andromeda - Flickr - Christian Gloor.jpg"]},
    "m27": {"label": "M 27, the Dumbbell Nebula", "class": "planetary", "names": ["M27", "M 27", "NGC 6853", "Dumbbell Nebula"],
        "titles": ["M27-Dumbbell Nebula-29-07-2026.jpg",
                   "M27, NGC 6853, Dumbbell Nebula (noao-02674).jpg",
                   "The Dumbbell Nebula - Messier 27.jpg",
                   "M27 Dumbbell.png"]},
    "ic5070": {"label": "IC 5070, the Pelican Nebula", "class": "emission", "names": ["IC 5070", "IC 5067", "Pelican Nebula"],
        "titles": ["IC 5070 - Pelican Nebula.jpg",
                   "NGC 7000 + IC 5070.jpg",
                   "Ic5070s.jpg"]},
    "ngc7635": {"label": "NGC 7635, the Bubble Nebula", "class": "emission", "names": ["NGC 7635", "C 11", "Caldwell 11", "Bubble Nebula", "Sh2-162"],
        "titles": ["2023-07-22 NGC 7635 Bubble nebula in Cassiopeia.png",
                   "The Bubble Nebula (NGC 7635) (noao0702a).jpg",
                   "NGC 7635 - Bubble Nebula (50806848087).jpg",
                   "Bubble nebula NGC 7635 (52107071289).jpg"]},
    "ngc6960": {"label": "NGC 6960, the Western Veil", "class": "snr", "names": ["NGC 6960", "C 34", "Caldwell 34", "Western Veil", "Witch's Broom"],
        "titles": ["Veil Nebula - NGC6960.jpg",
                   "Western veil nebula.jpg",
                   "The Western Veil Nebula.png",
                   "Witch broom nebula tõrva.jpg",
                   "2025 NGC6960 NGC6979 The Western Veil nebula and Pickering's Triangle with Askar 103APO+0.6X + ASI533MC-P (Rocco Parisi).jpg"]},
    "ic405": {"label": "IC 405, the Flaming Star Nebula", "class": "emission", "names": ["IC 405", "C 31", "Caldwell 31", "Flaming Star Nebula", "Sh2-229"],
        "titles": ["Flaming star nebula tõrva.jpg",
                   "IC 405 and IC 410- The Flaming Star Nebula (noao-ic405block).jpg",
                   "The Flaming Star Nebula, Tadpole Nebula IC405 IC410 IC417 & NGC1931.jpg",
                   "HaRGB-cropped.png",
                   "Flamingstarnebula.jpg",
                   "IC405-19x4min-20210401.jpg"]},
    "m20": {"label": "M 20, the Trifid Nebula", "class": "emission", "names": ["M20", "M 20", "NGC 6514", "Trifid Nebula"],
        "titles": ["Messier 20 Nebulosa Trifida en LRGB.jpg",
                   "Close up of the Trifid Nebula M20.jpg",
                   "M20- The Trifid Nebula (noao-m20castano).jpg",
                   "M20- The Trifid Nebula Wide (noao-m20johnson).jpg",
                   "Trifid Nebula aka M20.jpg"]},
    "m76": {"label": "M 76, the Little Dumbbell", "class": "planetary", "names": ["M76", "M 76", "NGC 650", "NGC 651", "Little Dumbbell"],
        "titles": ["M76 (Little Dumbbell) (noao-m76block).jpg",
                   "Little Dumbbell Nebula M76 by Goran Nilsson, Wim van Berlo & Liverpool Telescope.jpg",
                   "M76 - Little Dumbell Nebula.jpg"]},
    "ngc7662": {"label": "NGC 7662, the Blue Snowball", "class": "planetary", "names": ["NGC 7662", "C 22", "C22", "Caldwell 22", "Blue Snowball", "C22 Blue Snowball"],
        "titles": ["Caldwell 22.jpg"]},
    "ic1318": {"label": "IC 1318, the Butterfly Nebula", "class": "emission", "names": ["IC 1318", "IC 1318-Butterfly Nebula", "Butterfly Nebula", "Sadr Region", "Gamma Cygni Nebula"],
        "titles": ["IC 1318 - Butterfly Nebula - Gamma Cygni Nebula - by farajiibrahim.jpg",
                   "LDN889 - the Butterfly nebula (gianni).jpg"]},
    "sh2142": {"label": "Sh2-142 and NGC 7380, the Wizard Nebula", "class": "emission", "names": ["Sh2-142", "SH2-142", "NGC 7380", "Wizard Nebula"],
        "titles": ["The Wizard Nebula (NGC 7380) (Giovanni Barbarino).png",
                   "2025 NGC7380 Wizard Nebula 5s with Askar 103APO+0.6X + TL805+0.8X +L-eN + ASI533MC-P (Rocco Parisi).jpg",
                   "Ngc7380 20160815.jpg"]},
    "ldn1235": {"label": "LDN 1235, the Dark Shark", "class": "dark", "names": ["LDN 1235", "LDN1235", "Dark Shark", "LDN 1235 Dark Shark"],
        "titles": ["Dark Shark.png",
                   "LDN1235, The shark nebula (Explored^) - Flickr - Tupolev und seine Kamera.jpg"]},
}
CLASSES = {
    "emission": "emission nebulae", "snr": "supernova remnants", "planetary": "planetary nebulae",
    "galaxy": "galaxies", "dark": "dark and reflection nebulae", "narrowband": "dual-band (Ha + OIII) targets", "broadband": "broadband targets",
}
# other common targets by class (no references of their own: they use their class's).  Dark and
# reflection nebulae, galaxies and clusters are broadband targets; the rest dual-band ones
ALIASES = {
    "emission": ["IC 5070", "Pelican Nebula", "NGC 7635", "Bubble Nebula", "IC 1396", "IC 1396A", "Elephant's Trunk Nebula", "Sh2-131", "IC 1805", "Heart Nebula",
                 "IC 1848", "Soul Nebula", "M8", "Lagoon Nebula", "M16", "Eagle Nebula", "M17", "M20", "NGC 2237",
                 "Rosette Nebula", "C 49", "IC 405", "IC 410", "NGC 6888", "Crescent Nebula", "C 27", "NGC 2264",
                 "IC 434", "Horsehead Nebula", "NGC 2024", "Sh2-101", "NGC 6820", "IC 1318", "NGC 1499", "California Nebula",
                 "NGC 2174", "IC 443", "NGC 3372", "C 11", "C 19", "IC 5146", "Cocoon Nebula"],
    "snr": ["NGC 6960", "C 34", "Western Veil", "NGC 6979", "Pickering's Triangle", "Veil Nebula", "Cygnus Loop",
            "M1", "Crab Nebula", "Simeis 147", "Sh2-240"],
    "planetary": ["M27", "M 27", "Dumbbell Nebula", "M76", "M 76", "M57", "Ring Nebula", "NGC 7293", "Helix Nebula",
                  "NGC 6543", "C 6", "NGC 7662", "C 22", "NGC 2392", "C 39", "M97", "NGC 7008", "Abell 21"],
    "galaxy": ["M31", "M 31", "Andromeda Galaxy", "M33", "M51", "M81", "M82", "M101", "M63", "M64", "M104",
               "NGC 891", "NGC 7331", "C 30", "NGC 4565", "NGC 2403", "C 7", "M106", "M74", "M77"],
}


def _get(url: str) -> bytes:
    for a in range(8):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
            time.sleep(1.5)
            return r
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(5 * 2 ** a)
    raise RuntimeError("rate limited")


def fetch(title: str, cache: str) -> tuple[np.ndarray, dict]:
    os.makedirs(cache, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", title)[:80]
    jpg, meta_f = os.path.join(cache, stem + ".jpg"), os.path.join(cache, stem + ".json")
    if not (os.path.exists(jpg) and os.path.exists(meta_f)):
        q = urllib.parse.urlencode({"action": "query", "titles": "File:" + title, "prop": "imageinfo",
                                    "iiprop": "url|extmetadata", "iiurlwidth": 1200, "format": "json"})
        page = next(iter(json.loads(_get("https://commons.wikimedia.org/w/api.php?" + q))["query"]["pages"].values()))
        ii = page["imageinfo"][0]
        md = ii.get("extmetadata", {})
        Image.open(io.BytesIO(_get(ii["thumburl"]))).convert("RGB").save(jpg, quality=95)
        json.dump({"title": title, "url": ii["descriptionurl"],
                   "license": md.get("LicenseShortName", {}).get("value"),
                   "author": re.sub(r"<[^>]+>", "", md.get("Artist", {}).get("value", "")).strip()},
                  open(meta_f, "w"), indent=1)
    return np.asarray(Image.open(jpg).convert("RGB"), np.float32) / 255.0, json.load(open(meta_f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join("output", "autofinish_refs"))
    ap.add_argument("--titles", help="JSON {target: [Commons titles]} replacing the titles above")
    a = ap.parse_args()
    if a.titles:
        for k, v in json.load(open(a.titles)).items():
            TARGETS[k]["titles"] = v
    out = {"note": "Reference looks for Auto-finish (astrophoto/autofinish.py), built by experiments/autofinish_refs.py",
           "targets": {}, "classes": {}, "aliases": {}}
    per_class: dict[str, list] = {}
    def rnd(v):
        return ({k: rnd(x) for k, x in v.items()} if isinstance(v, dict) else [rnd(x) for x in v] if isinstance(v, list)
                else round(v, 5) if isinstance(v, float) else v)
    for key, t in TARGETS.items():
        stats, sources = [], []
        for title in t["titles"]:
            img, meta = fetch(title, a.cache)
            st = look_stats(img)
            stats.append(st)
            sources.append(meta)
            print(f"{key:8s} {title[:60]:60s} " + " ".join(f"{k}={st[k]:.3f}" for k in ("sky_L", "sig_L50", "C50", "detail")))
        if not stats:
            continue
        out["targets"][key] = {"label": t["label"], "class": t["class"], "names": t["names"],
                               "look": rnd(aggregate(stats)), "looks": [rnd(x) for x in stats], "sources": sources}
        for cls in (t["class"], "narrowband" if t["class"] in ("emission", "snr", "planetary") else "broadband"):
            per_class.setdefault(cls, []).extend(zip(stats, sources))
    if "broadband" not in per_class and per_class:          # no broadband references: all of them
        per_class["broadband"] = [x for c in ("emission", "snr", "planetary", "galaxy") for x in per_class.get(c, [])]
    for cls, items in per_class.items():
        out["classes"][cls] = {"label": CLASSES[cls], "look": rnd(aggregate([x[0] for x in items])),
                               "looks": [rnd(x[0]) for x in items], "sources": [x[1] for x in items]}
    for cls, names in ALIASES.items():
        for n in names:
            out["aliases"][re.sub(r"[^A-Z0-9]", "", n.upper())] = cls
    os.makedirs(os.path.dirname(REF_FILE), exist_ok=True)
    json.dump(out, open(REF_FILE, "w"), indent=1)
    print("written", REF_FILE)


if __name__ == "__main__":
    main()
