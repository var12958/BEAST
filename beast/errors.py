"""User-facing error types.

Every failure that a normal user can hit is represented by a
:class:`BeastError` subclass.  Each error carries a human readable message and
an optional actionable hint, which the CLI renders as a panel and the TUI
renders inline.  Raw tracebacks are only ever shown when debug logging is on.
"""

from __future__ import annotations

from collections.abc import Sequence


class BeastError(Exception):
    """Base class for all errors that BEAST handles gracefully."""

    exit_code: int = 1
    title: str = "BEAST hit a problem"

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        # ``detail`` is extra technical context kept for the log file only.
        self.detail = detail

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


class WorkspaceError(BeastError):
    """The active workspace directory cannot be used."""

    title = "Workspace problem"


class OllamaConfigError(BeastError):
    """The configured Ollama endpoint is not a usable URL."""

    title = "Invalid Ollama endpoint"
    exit_code = 9


class OllamaNotInstalledError(BeastError):
    """The ``ollama`` executable is not on PATH."""

    title = "Ollama is not installed"
    exit_code = 2


class OllamaNotRunningError(BeastError):
    """The Ollama HTTP service is not reachable."""

    title = "Ollama is not running"
    exit_code = 3


class OllamaTimeoutError(BeastError):
    """Ollama accepted the connection but did not answer in time."""

    title = "Ollama timed out"
    exit_code = 4


class OllamaProtocolError(BeastError):
    """Ollama answered with something BEAST does not understand."""

    title = "Unexpected Ollama response"
    exit_code = 5


class NoModelsError(BeastError):
    """No chat-capable local model is installed."""

    title = "No local models available"
    exit_code = 6


class ModelNotFoundError(BeastError):
    """The requested model is not installed locally."""

    title = "Model not available"
    exit_code = 7

    def __init__(self, model: str, available: Sequence[str] = ()) -> None:
        self.model = model
        self.available = tuple(available)
        message = f"The model '{model}' is not available in your local Ollama."
        if self.available:
            listing = ", ".join(self.available)
            hint = f"Installed models: {listing}\nPick one with --model, or run without --model to choose interactively."
        else:
            hint = "Install a model with `ollama pull <model>` and try again."
        super().__init__(message, hint=hint)


class UnsupportedModelError(BeastError):
    """The requested model cannot be used by BEAST (e.g. a cloud model)."""

    title = "Model not supported"
    exit_code = 8


class CloudAuthError(BeastError):
    """Ollama Cloud rejected the request because the user is not signed in."""

    title = "Ollama Cloud sign-in required"
    exit_code = 12


class CloudUnavailableError(BeastError):
    """Ollama Cloud could not be reached for a cloud-hosted model."""

    title = "Ollama Cloud is unreachable"
    exit_code = 13


class ToolCallingUnsupportedError(BeastError):
    """The selected model rejected the tool definitions."""

    title = "Model cannot use tools"
    exit_code = 10


class AgentIterationLimitError(BeastError):
    """The model kept calling tools past the safety limit."""

    title = "Agent stopped"
    exit_code = 11

    def __init__(self, limit: int) -> None:
        super().__init__(
            f"BEAST stopped after {limit} tool rounds to avoid an endless loop.",
            hint=(
                "The model kept requesting tools without finishing. Give it a "
                "narrower instruction, or check the transcript for a tool that "
                "kept failing."
            ),
        )
        self.limit = limit
