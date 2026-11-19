"""Measure Phase 3 cold/warm indexing and context-building costs.

Run from the repository root::

    python scripts/phase3_benchmark.py

Uses the same fixtures as the test suite, written to a temporary directory with
an isolated cache, so nothing touches the developer's project.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from phase3_definition_of_done import (  # noqa: E402
    AUTH_QUESTION,
    OVERVIEW_QUESTION,
    load_fixtures,
    write_project,
)

from beast.intelligence import ProjectIntelligence  # noqa: E402


def measure(root: Path) -> None:
    cache = Path(tempfile.mkdtemp(prefix="beast-bench-cache-"))
    os.environ["BEAST_CACHE_DIR"] = str(cache)
    intelligence = ProjectIntelligence(root)
    try:
        started = time.perf_counter()
        cold = intelligence.refresh(force=True)
        cold_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        warm = intelligence.refresh(force=True)
        warm_ms = (time.perf_counter() - started) * 1000

        print(
            f"  index: {cold.total} files cold in {cold_ms:.0f} ms, "
            f"warm in {warm_ms:.0f} ms "
            f"(added={warm.added}, changed={warm.changed}, removed={warm.removed}, "
            f"analysed={warm.analyzed})"
        )

        for label, question in (
            ("overview", OVERVIEW_QUESTION),
            ("auth", AUTH_QUESTION),
        ):
            started = time.perf_counter()
            bundle = intelligence.context_for(question)
            elapsed_ms = (time.perf_counter() - started) * 1000
            print(
                f"  context[{label}]: {len(bundle.files)} files, "
                f"{bundle.symbol_count} symbols, {bundle.characters:,} chars, "
                f"{elapsed_ms:.0f} ms, redactions={bundle.redactions}"
            )
    finally:
        intelligence.close()


def main() -> int:
    fixtures = load_fixtures()
    for name, files in fixtures.items():
        print(name)
        measure(write_project(files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
