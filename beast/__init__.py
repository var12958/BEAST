"""BEAST — a local-first AI coding agent for the terminal.

The package is deliberately split into layers so the interactive agent can
grow without rewrites:

* :mod:`beast.ollama`  — talking to a local Ollama instance (no cloud, no API keys)
* :mod:`beast.core`    — conversation state and the agent orchestration seam
* :mod:`beast.workspace` — knowing which project the user is working in
* :mod:`beast.ui`      — the full-screen Textual interface
* :mod:`beast.cli`     — the ``asap launch beast`` entry point
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
