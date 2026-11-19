"""Detect and describe the Ollama installation BEAST talks to.

By default BEAST talks to the Ollama service on this machine
(``http://127.0.0.1:11434``).  There are no cloud endpoints, no API keys and no
fallbacks to hosted providers.  The endpoint can be pointed at another host
(for example Ollama in WSL or Docker) with ``BEAST_OLLAMA_URL`` or
``--ollama-url``; see the README.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from beast.errors import (
    OllamaConfigError,
    OllamaNotInstalledError,
    OllamaNotRunningError,
    OllamaTimeoutError,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434
DEFAULT_BASE_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"

#: Override the endpoint (e.g. Ollama in WSL/Docker) without touching code.
BASE_URL_ENV = "BEAST_OLLAMA_URL"


def default_base_url() -> str:
    """Return the Ollama endpoint, honouring ``BEAST_OLLAMA_URL``.

    An empty or blank environment variable is treated as "not set" so a stray
    ``BEAST_OLLAMA_URL=`` cannot break the launch.
    """
    override = os.environ.get(BASE_URL_ENV, "").strip()
    return (override or DEFAULT_BASE_URL).rstrip("/")


def validate_base_url(url: str) -> str:
    """Return ``url`` normalised, or raise :class:`OllamaConfigError`.

    Only ``http`` and ``https`` URLs with a host are accepted.  Validating here
    matters because ``httpx.InvalidURL`` is not an ``httpx.HTTPError``: without
    this check a typo such as ``--ollama-url http://127.0.0.1:abc`` would escape
    the friendly error handling and surface as an unexpected crash.
    """
    candidate = url.strip().rstrip("/")
    try:
        parts = urlsplit(candidate)
        # ``port`` raises ValueError for non-numeric or out-of-range ports.
        parts.port
    except ValueError as exc:
        raise OllamaConfigError(
            f"The Ollama endpoint '{url}' is not a valid URL.",
            hint=(
                "Check --ollama-url (or BEAST_OLLAMA_URL). The default is "
                f"{DEFAULT_BASE_URL}."
            ),
            detail=repr(exc),
        ) from exc

    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise OllamaConfigError(
            f"The Ollama endpoint '{url}' is not a valid http:// or https:// URL.",
            hint=(
                "Check --ollama-url (or BEAST_OLLAMA_URL). The default is "
                f"{DEFAULT_BASE_URL}."
            ),
        )
    return candidate


def find_executable() -> str | None:
    """Return the path to the ``ollama`` CLI, or ``None`` when missing."""
    return shutil.which("ollama")


@dataclass(frozen=True, slots=True)
class OllamaStatus:
    """Outcome of probing the local Ollama service."""

    base_url: str
    is_installed: bool
    is_running: bool
    executable: str | None = None
    version: str | None = None
    error: str | None = None
    hint: str | None = None
    #: One of ``None``, ``"connect"``, ``"timeout"`` or ``"http"``.
    failure: str | None = None
    detail: str | None = None

    @property
    def is_ready(self) -> bool:
        return self.is_running

    @property
    def connection_label(self) -> str:
        return "LOCAL / OLLAMA" if self.is_running else "OFFLINE"

    @property
    def version_label(self) -> str:
        return f"v{self.version}" if self.version else "unknown version"


async def _probe(http: httpx.AsyncClient) -> tuple[bool, str | None]:
    """Return ``(is_running, version)`` for a client pointed at Ollama."""
    response = await http.get("/api/version")
    if response.status_code == 200:
        try:
            payload = response.json()
        except ValueError:
            return True, None
        version = payload.get("version") if isinstance(payload, dict) else None
        return True, version if isinstance(version, str) else None

    # Very old builds do not expose /api/version; the tags endpoint is the
    # authoritative "is it serving?" check, so fall back to it.
    response = await http.get("/api/tags")
    return response.status_code == 200, None


async def probe(
    base_url: str | None = None,
    *,
    timeout: float = 3.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> OllamaStatus:
    """Check whether Ollama is installed and serving requests.

    Never raises for connection problems — the returned status describes the
    failure so the caller can decide how to present it.  The one exception is
    a malformed endpoint, which raises :class:`OllamaConfigError`.
    """
    url = validate_base_url(base_url or default_base_url())
    executable = find_executable()

    def unavailable(**kwargs: object) -> OllamaStatus:
        return OllamaStatus(
            base_url=url,
            is_installed=executable is not None,
            is_running=False,
            executable=executable,
            **kwargs,  # type: ignore[arg-type]
        )

    try:
        async with httpx.AsyncClient(
            base_url=url,
            timeout=httpx.Timeout(timeout, connect=min(timeout, 3.0)),
            transport=transport,
        ) as http:
            is_running, version = await _probe(http)
    except httpx.TimeoutException as exc:
        return unavailable(
            failure="timeout",
            error=f"Ollama did not respond within {timeout:.0f} seconds.",
            hint="Ollama may still be starting. Wait a moment and try again.",
            detail=repr(exc),
        )
    except httpx.ConnectError as exc:
        return unavailable(
            failure="connect",
            error=f"Could not reach the Ollama service at {url}.",
            hint="Start it with `ollama serve`, or launch the Ollama app, then try again.",
            detail=repr(exc),
        )
    except httpx.HTTPError as exc:
        return unavailable(
            failure="http",
            error=f"The Ollama service at {url} returned an unexpected error.",
            hint="Check the Ollama log output and make sure the service is healthy.",
            detail=repr(exc),
        )

    if not is_running:
        return unavailable(
            failure="http",
            error=f"The Ollama service at {url} is reachable but not serving models.",
            hint="Restart Ollama (`ollama serve`) and try again.",
        )

    return OllamaStatus(
        base_url=url,
        is_installed=executable is not None,
        is_running=True,
        executable=executable,
        version=version,
    )


def ensure_ready(status: OllamaStatus) -> None:
    """Raise the most specific :class:`BeastError` for a failed probe."""
    if status.is_running:
        return

    if not status.is_installed:
        raise OllamaNotInstalledError(
            "Ollama doesn't appear to be installed on this machine.",
            hint=(
                "Install it from https://ollama.com/download, then run "
                "`ollama serve` (or start the Ollama app) so BEAST can reach it."
            ),
            detail=status.detail,
        )

    if status.failure == "timeout":
        raise OllamaTimeoutError(
            status.error or "Ollama did not respond in time.",
            hint=status.hint,
            detail=status.detail,
        )

    raise OllamaNotRunningError(
        status.error or f"Could not reach the Ollama service at {status.base_url}.",
        hint=status.hint
        or "Start Ollama with `ollama serve`, or launch the Ollama app, then try again.",
        detail=status.detail,
    )
