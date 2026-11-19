"""Async HTTP client for the local Ollama API.

The client is deliberately small: it knows how to list installed models and
how to stream a chat completion.  It contains no provider-specific logic and
never contacts anything outside the configured endpoint, which is loopback by
default.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import httpx
from pydantic import ValidationError

from beast.errors import (
    BeastError,
    CloudAuthError,
    CloudUnavailableError,
    ModelNotFoundError,
    OllamaNotRunningError,
    OllamaProtocolError,
    OllamaTimeoutError,
    ToolCallingUnsupportedError,
)

from .connection import default_base_url, validate_base_url
from .models import ChatChunk, OllamaModel, OllamaModelList

logger = logging.getLogger(__name__)

#: Generation can legitimately take minutes on CPU, so allow a long read
#: timeout while still failing fast on a dead socket at connect time.
DEFAULT_READ_TIMEOUT = 300.0
DEFAULT_CONNECT_TIMEOUT = 5.0


def _decode_body(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace").strip()


def _error_message_from_body(raw: bytes) -> str | None:
    """Pull Ollama's ``{"error": "..."}`` message out of a response body."""
    text = _decode_body(raw)
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text[:500]
    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        return payload["error"]
    return text[:500]


#: Substrings Ollama uses when its model runner dies rather than answering.
_RUNNER_FAILURE_SIGNALS = (
    "cuda",
    "out of memory",
    "terminated",
    "exit status",
    "failed to load",
    "unable to load",
    "llama-server",
)

#: Ollama Cloud models are identified by this tag suffix.  The local Ollama
#: service proxies them to ollama.com using the account stored by `ollama signin`.
CLOUD_TAG_SUFFIX = ":cloud"

#: Messages the Ollama service relays when the account is not signed in.
_CLOUD_AUTH_SIGNALS = (
    "unauthorized",
    "not signed in",
    "sign in",
    "signin",
    "authentication",
    "forbidden",
    "401",
    "403",
    "invalid token",
    "token expired",
    "api key",
)

#: Messages that mean ollama.com could not be reached from the local service.
_CLOUD_NETWORK_SIGNALS = (
    "dial tcp",
    "no such host",
    "i/o timeout",
    "connection refused",
    "connection reset",
    "network is unreachable",
    "tls handshake",
    "temporary failure in name resolution",
)


def is_cloud_model(model: str) -> bool:
    """Return True for an Ollama Cloud model identifier (``name:cloud``)."""
    return model.strip().lower().endswith(CLOUD_TAG_SUFFIX)


def _cloud_failure(model: str, status: int, detail: str | None) -> BeastError | None:
    """Classify a cloud-model failure, or return ``None`` for other errors.

    The message never repeats credentials: Ollama Cloud auth lives in the
    Ollama CLI's own store and is never read or logged by BEAST.
    """
    if not is_cloud_model(model):
        return None
    lowered = (detail or "").lower()
    if status in (401, 403) or any(
        signal in lowered for signal in _CLOUD_AUTH_SIGNALS
    ):
        return CloudAuthError(
            f"Ollama Cloud refused the request for '{model}': not signed in.",
            hint=(
                "Run `ollama signin` in another terminal and complete the "
                "browser sign-in, then try again. BEAST reuses that account; "
                "it never asks for or stores an API key."
            ),
            detail=detail,
        )
    if any(signal in lowered for signal in _CLOUD_NETWORK_SIGNALS):
        return CloudUnavailableError(
            f"The Ollama service could not reach Ollama Cloud for '{model}'.",
            hint=(
                "Check your internet connection, then try again. Local models "
                "keep working while the cloud is unavailable."
            ),
            detail=detail,
        )
    return None


def _chat_failure(model: str, detail: str | None) -> tuple[str, str]:
    """Turn a raw Ollama error into (message, hint) for a human reader.

    The raw text is kept separately as error detail so it reaches the log file
    without being thrown at the user.
    """
    lowered = (detail or "").lower()
    if any(signal in lowered for signal in _RUNNER_FAILURE_SIGNALS):
        return (
            f"Ollama could not start the model '{model}'.",
            "The model runner crashed — that usually points at the GPU driver "
            "or not enough memory. Check that `ollama run <model>` works in "
            "another terminal, then try a smaller model.",
        )
    return (
        f"Ollama could not complete the chat request for '{model}'.",
        "Check that the model runs with `ollama run <model>`, then try again.",
    )


class OllamaClient:
    """Thin async wrapper over ``/api/tags`` and ``/api/chat``.

    ``transport`` exists so tests can inject :class:`httpx.MockTransport`; in
    normal use it is ``None`` and a real connection is used.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = DEFAULT_READ_TIMEOUT,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = validate_base_url(base_url or default_base_url())
        self._timeout = httpx.Timeout(timeout, connect=connect_timeout)
        self._transport = transport
        self._http: httpx.AsyncClient | None = None

    @property
    def http(self) -> httpx.AsyncClient:
        """Lazily create the underlying client inside the running event loop."""
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self._timeout,
                transport=self._transport,
                headers={"User-Agent": "beast-agent/0.1"},
            )
        return self._http

    async def close(self) -> None:
        if self._http is not None and not self._http.is_closed:
            await self._http.aclose()
        self._http = None

    async def __aenter__(self) -> OllamaClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # ------------------------------------------------------------------ models

    async def list_models(self) -> list[OllamaModel]:
        """Return every model installed in the local Ollama store."""
        try:
            response = await self.http.get("/api/tags")
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                "Timed out while asking Ollama for its model list.",
                hint="Ollama may be busy loading a model. Wait a moment and try again.",
                detail=repr(exc),
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaNotRunningError(
                f"Could not reach Ollama at {self.base_url}.",
                hint="Start it with `ollama serve`, or launch the Ollama app.",
                detail=repr(exc),
            ) from exc

        if response.status_code != 200:
            raise OllamaProtocolError(
                f"Ollama returned HTTP {response.status_code} for the model list.",
                hint="Restart Ollama and try again.",
                detail=_decode_body(response.content),
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise OllamaProtocolError(
                "Ollama returned a model list that is not valid JSON.",
                detail=_decode_body(response.content) or repr(exc),
            ) from exc

        try:
            return OllamaModelList.model_validate(payload).models
        except ValidationError as exc:
            raise OllamaProtocolError(
                "Ollama returned a model list in an unexpected format.",
                hint="Your Ollama version may be unsupported. Update Ollama and try again.",
                detail=str(exc),
            ) from exc

    # ------------------------------------------------------------------- cloud

    async def check_cloud_auth(self) -> bool | None:
        """Check whether the local Ollama service is signed in to Ollama Cloud.

        ``POST /api/me`` returns the account the Ollama CLI is signed in with.
        BEAST only inspects the status code — the response can contain the
        user's email, so it is never parsed, stored or logged.

        Returns ``True`` when signed in and ``None`` when this Ollama build does
        not expose the endpoint (older releases); raises :class:`CloudAuthError`
        when the request is explicitly rejected.
        """
        try:
            response = await self.http.post("/api/me")
        except httpx.HTTPError:
            # The chat request itself reports connectivity problems clearly.
            return None

        if response.status_code == 200:
            return True
        if response.status_code in (401, 403):
            raise CloudAuthError(
                "Ollama Cloud is not signed in on this machine.",
                hint=(
                    "Run `ollama signin` in another terminal and complete the "
                    "browser sign-in, then try again. BEAST reuses that account; "
                    "it never asks for or stores an API key."
                ),
            )
        return None

    # -------------------------------------------------------------------- chat

    async def stream_chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        options: Mapping[str, Any] | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> AsyncIterator[ChatChunk]:
        """Stream a chat completion for ``model`` as :class:`ChatChunk` objects.

        ``tools`` carries Ollama function definitions; the model may answer with
        ``message.tool_calls`` instead of text.  Yields one chunk per
        newline-delimited JSON object Ollama sends.  Raises :class:`BeastError`
        subclasses for every failure mode the UI must show.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": [dict(message) for message in messages],
            "stream": True,
        }
        if options:
            payload["options"] = dict(options)
        if tools:
            payload["tools"] = [dict(tool) for tool in tools]

        logger.debug(
            "chat request model=%s messages=%d tools=%d",
            model,
            len(messages),
            len(tools or ()),
        )

        try:
            async with self.http.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    detail = _error_message_from_body(body) or f"HTTP {response.status_code}"
                    if response.status_code == 404:
                        raise ModelNotFoundError(model)
                    if response.status_code == 400 and "tool" in detail.lower():
                        raise ToolCallingUnsupportedError(
                            f"The model '{model}' rejected the tool definitions.",
                            hint=(
                                "This model does not support tool calling. Pick a "
                                "model the model list marks as tool-capable."
                            ),
                            detail=detail,
                        )
                    cloud_error = _cloud_failure(
                        model, response.status_code, detail
                    )
                    if cloud_error is not None:
                        raise cloud_error
                    message, hint = _chat_failure(model, detail)
                    raise OllamaProtocolError(message, hint=hint, detail=detail)

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise OllamaProtocolError(
                            "Ollama streamed a response BEAST could not parse.",
                            hint="Update Ollama — the streaming format may have changed.",
                            detail=f"{exc}: {line[:300]}",
                        ) from exc

                    if not isinstance(data, dict):
                        raise OllamaProtocolError(
                            "Ollama streamed an unexpected payload.",
                            detail=line[:300],
                        )

                    error = data.get("error")
                    if isinstance(error, str) and error:
                        cloud_error = _cloud_failure(model, 500, error)
                        if cloud_error is not None:
                            raise cloud_error
                        message, hint = _chat_failure(model, error)
                        raise OllamaProtocolError(
                            message,
                            hint=hint,
                            detail=line[:500],
                        )

                    try:
                        chunk = ChatChunk.model_validate(data)
                    except ValidationError as exc:
                        raise OllamaProtocolError(
                            "Ollama streamed a chunk in an unexpected format.",
                            detail=f"{exc}: {line[:300]}",
                        ) from exc

                    yield chunk

        except httpx.TimeoutException as exc:
            cloud = is_cloud_model(model)
            raise OllamaTimeoutError(
                (
                    f"The cloud reply for '{model}' timed out."
                    if cloud
                    else "Ollama stopped responding while generating a reply."
                ),
                hint=(
                    "Check your internet connection and try again."
                    if cloud
                    else (
                        "The model may not fit in available memory. Try a "
                        "smaller model, or wait for it to finish loading."
                    )
                ),
                detail=repr(exc),
            ) from exc
        except httpx.ConnectError as exc:
            raise OllamaNotRunningError(
                "Lost the connection to Ollama.",
                hint="Make sure Ollama is still running, then try again.",
                detail=repr(exc),
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaProtocolError(
                "The connection to Ollama failed mid-response.",
                hint="Try again, and check the Ollama service if it keeps happening.",
                detail=repr(exc),
            ) from exc
