"""Run siril-scripts' ``processing/Hubble_Palette_from_Dual-Band_OSC.py`` (Carlo Mollicone - AstroBOH, GPL-3.0-or-later) headless, inside Siril.

The script has no command-line mode: it is a PyQt6 window whose *Apply* button calls ``HubblePaletteApp.run_hubble_palette_logic``, which splits the loaded RGB image,
derives Ha / OIII / synthetic SII with Siril ``pm`` (the chosen formula preset) and composes the chosen palette with ``rgbcomp``, then loads the result.  This worker,
started by Siril with ``pyscript hubble_palette_worker.py --script <file> --palette SHO --preset Classic`` while the starless image is loaded, imports the script file as
it is, builds its window offscreen, selects the palette's radio button and the formula preset exactly as a user would, and presses *Apply* (``on_apply``).  Everything that
touches the image is the script's own code; the Siril script that started the worker saves the image the script loaded.

The script reports its errors in message boxes (modal, they would wait forever offscreen): they are turned into exceptions, so a failure ends the pyscript with the message
in Siril's log.  Success is the script's own log line "Successfully generated and loaded the <palette> palette!".
"""
import argparse
import importlib.util
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PALETTES = ("HSO", "SHO", "OSH", "OHS", "HOS", "HOO")
PRESETS = ("Classic", "Improved", "Advanced", "NonLinear S2")


def main() -> int:
    ap = argparse.ArgumentParser(prog="hubble_palette_worker.py")
    ap.add_argument("--script", required=True, help="path of Hubble_Palette_from_Dual-Band_OSC.py")
    ap.add_argument("--palette", required=True, choices=PALETTES)
    ap.add_argument("--preset", required=True, help="formula preset: " + " | ".join(PRESETS) + " (underscores for spaces)")
    a = ap.parse_args([x for x in sys.argv[1:] if not x.startswith("--siril")])
    preset = a.preset.replace("_", " ")

    spec = importlib.util.spec_from_file_location("hubble_palette_from_dual_band_osc", a.script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from PyQt6.QtWidgets import QApplication, QMessageBox

    def _raise(_parent, title, text, *args, **kwargs):
        raise RuntimeError(f"{title}: {text}")

    QMessageBox.critical = staticmethod(_raise)
    QMessageBox.warning = staticmethod(_raise)

    app = QApplication.instance() or QApplication([sys.argv[0]])     # noqa: F841  (the window needs one)
    win = mod.HubblePaletteApp()
    if not hasattr(win, "radio_buttons"):                             # the constructor returned early: no connection, no image, or not RGB
        print("ERROR: the Hubble palette script did not start (no RGB image loaded in Siril?)", file=sys.stderr)
        return 1
    by_name = {name: pid for pid, name in win.display_names.items()}
    win.radio_buttons[by_name[a.palette]].setChecked(True)
    win.preset_combobox.setCurrentText(preset)
    if win.preset_combobox.currentText() != preset:
        print(f"ERROR: the script has no formula preset {preset!r} ({', '.join(win.preset_formulas)})", file=sys.stderr)
        return 1
    print(f"Hubble palette {a.palette}, preset {preset}: HA = {win.ha_formula_entry.text()}, OIII = {win.oiii_formula_entry.text()}, "
          f"S2 = {win.s2_formula_entry.text()}")
    win.on_apply()
    try:
        win.siril.disconnect()
    except Exception:                                                 # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
