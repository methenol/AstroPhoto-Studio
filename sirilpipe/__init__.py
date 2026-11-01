"""AstroPhoto Studio core: Siril-driven stacking pipeline (the "backend" branch).

Siril (siril-cli) does the pre-processing (calibration, registration, drizzle, stacking); this
package finds and classifies the subs, writes the Siril scripts, runs them, reads the registration
statistics back and keeps the session artefacts the web UI and the /api/v1 job API depend on.
"""
__version__ = "2.0.0"
