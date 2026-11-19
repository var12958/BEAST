"""Console entry point with a friendly dependency check.

The ``asap`` script points here rather than straight at ``beast.cli`` so that a
broken or incomplete install produces an actionable message instead of a raw
``ModuleNotFoundError`` traceback.
"""

from __future__ import annotations

import sys


def report_missing_dependency(error: ModuleNotFoundError) -> None:
    """Explain how to repair an install that is missing a package."""
    package = error.name or "a required package"
    if package.startswith("beast"):
        message = (
            "BEAST is not installed correctly: its own package could not be "
            "imported."
        )
    else:
        message = f"BEAST can't start because the Python package '{package}' is missing."
    print(
        f"{message}\n"
        "Reinstall it and its dependencies from the BEAST directory:\n"
        "  python -m pip install -e .",
        file=sys.stderr,
    )


def main() -> None:
    """Run the ``asap`` command."""
    try:
        # Imported lazily so a missing dependency is caught and explained.
        from beast.cli import app
    except ModuleNotFoundError as error:
        report_missing_dependency(error)
        raise SystemExit(1) from None

    app()
