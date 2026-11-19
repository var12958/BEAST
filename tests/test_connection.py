"""Ollama installation and service detection."""

from __future__ import annotations

import httpx
import pytest

from beast.errors import (
    OllamaConfigError,
    OllamaNotInstalledError,
    OllamaNotRunningError,
    OllamaTimeoutError,
)
from beast.ollama import connection
from beast.ollama.connection import (
    DEFAULT_BASE_URL,
    OllamaStatus,
    default_base_url,
    ensure_ready,
    find_executable,
    probe,
    validate_base_url,
)

BASE = "http://ollama.test"


def test_default_base_url_is_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(connection.BASE_URL_ENV, raising=False)

    assert DEFAULT_BASE_URL == "http://127.0.0.1:11434"
    assert default_base_url() == DEFAULT_BASE_URL


def test_base_url_can_be_overridden_by_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(connection.BASE_URL_ENV, "http://192.168.1.5:11434/")

    assert default_base_url() == "http://192.168.1.5:11434"


def test_a_blank_environment_override_falls_back_to_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(connection.BASE_URL_ENV, "   ")

    assert default_base_url() == DEFAULT_BASE_URL


def test_base_urls_are_normalised() -> None:
    assert validate_base_url("http://127.0.0.1:11434/") == DEFAULT_BASE_URL
    assert validate_base_url(" https://ollama.internal:443 ") == (
        "https://ollama.internal:443"
    )


@pytest.mark.parametrize(
    "value",
    [
        "",
        "not-a-url",
        "ftp://127.0.0.1:11434",
        "http://",
        "http://127.0.0.1:abc",
        "http://127.0.0.1:99999",
    ],
)
def test_invalid_base_urls_are_rejected(value: str) -> None:
    with pytest.raises(OllamaConfigError):
        validate_base_url(value)


async def test_probe_rejects_a_malformed_endpoint_before_connecting() -> None:
    with pytest.raises(OllamaConfigError):
        await probe("http://127.0.0.1:abc")


def test_find_executable_uses_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(connection.shutil, "which", lambda name: "/x/ollama")

    assert find_executable() == "/x/ollama"


def test_find_executable_reports_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(connection.shutil, "which", lambda name: None)

    assert find_executable() is None


async def test_probe_detects_a_running_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(connection.shutil, "which", lambda name: "/x/ollama")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/version"
        return httpx.Response(200, json={"version": "0.33.2"})

    status = await probe(BASE, transport=httpx.MockTransport(handler))

    assert status.is_running is True
    assert status.is_installed is True
    assert status.is_ready is True
    assert status.version == "0.33.2"
    assert status.version_label == "v0.33.2"
    assert status.connection_label == "LOCAL / OLLAMA"
    assert status.failure is None


async def test_probe_falls_back_to_tags_for_old_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(connection.shutil, "which", lambda name: "/x/ollama")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(404, text="Not Found")
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": []})

    status = await probe(BASE, transport=httpx.MockTransport(handler))

    assert status.is_running is True
    assert status.version is None
    assert status.version_label == "unknown version"


async def test_probe_reports_connection_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(connection.shutil, "which", lambda name: None)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    status = await probe(BASE, transport=httpx.MockTransport(handler))

    assert status.is_running is False
    assert status.is_installed is False
    assert status.failure == "connect"
    assert BASE in (status.error or "")
    assert status.connection_label == "OFFLINE"
    assert "ollama serve" in (status.hint or "")


async def test_probe_reports_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(connection.shutil, "which", lambda name: "/x/ollama")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow", request=request)

    status = await probe(BASE, timeout=0.25, transport=httpx.MockTransport(handler))

    assert status.is_running is False
    assert status.is_installed is True
    assert status.failure == "timeout"
    assert "0 seconds" in (status.error or "") or "did not respond" in (status.error or "")


async def test_probe_reports_a_service_that_is_up_but_not_serving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(connection.shutil, "which", lambda name: "/x/ollama")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="starting")

    status = await probe(BASE, transport=httpx.MockTransport(handler))

    assert status.is_running is False
    assert status.failure == "http"
    assert "not serving models" in (status.error or "")


async def test_probe_swallows_unexpected_http_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(connection.shutil, "which", lambda name: "/x/ollama")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("socket gone", request=request)

    status = await probe(BASE, transport=httpx.MockTransport(handler))

    assert status.is_running is False
    assert status.failure == "http"


def test_ensure_ready_accepts_a_running_service() -> None:
    ensure_ready(OllamaStatus(base_url=BASE, is_installed=True, is_running=True))


def test_ensure_ready_explains_a_missing_installation() -> None:
    status = OllamaStatus(
        base_url=BASE, is_installed=False, is_running=False, failure="connect"
    )

    with pytest.raises(OllamaNotInstalledError) as excinfo:
        ensure_ready(status)

    assert "doesn't appear to be installed" in excinfo.value.message
    assert "ollama.com/download" in (excinfo.value.hint or "")


def test_ensure_ready_explains_a_stopped_service() -> None:
    status = OllamaStatus(
        base_url=BASE,
        is_installed=True,
        is_running=False,
        failure="connect",
        error="Could not reach Ollama.",
        hint="Start it with `ollama serve`.",
    )

    with pytest.raises(OllamaNotRunningError) as excinfo:
        ensure_ready(status)

    assert "Could not reach Ollama." in excinfo.value.message
    assert "ollama serve" in (excinfo.value.hint or "")


def test_ensure_ready_explains_a_timeout() -> None:
    status = OllamaStatus(
        base_url=BASE,
        is_installed=True,
        is_running=False,
        failure="timeout",
        error="Ollama did not respond within 3 seconds.",
    )

    with pytest.raises(OllamaTimeoutError):
        ensure_ready(status)


def test_ensure_ready_falls_back_to_a_generic_hint() -> None:
    status = OllamaStatus(
        base_url=BASE, is_installed=True, is_running=False, failure="connect"
    )

    with pytest.raises(OllamaNotRunningError) as excinfo:
        ensure_ready(status)

    assert BASE in excinfo.value.message
    assert "ollama serve" in (excinfo.value.hint or "")
