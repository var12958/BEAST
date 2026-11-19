"""HTTP communication with Ollama: model listing and streamed chat."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

import httpx
import pytest

from beast.errors import (
    ModelNotFoundError,
    OllamaNotRunningError,
    OllamaProtocolError,
    OllamaTimeoutError,
    ToolCallingUnsupportedError,
)
from beast.ollama.client import OllamaClient
from beast.ollama.models import ChatChunk

BASE = "http://ollama.test"
MODEL = "qwen3:8b"

Handler = Callable[[httpx.Request], httpx.Response]


def make_client(handler: Handler) -> OllamaClient:
    return OllamaClient(base_url=BASE, transport=httpx.MockTransport(handler))


def ndjson(*payloads: Mapping[str, Any]) -> bytes:
    return b"".join(
        json.dumps(payload).encode("utf-8") + b"\n" for payload in payloads
    )


def chat_handler(
    *payloads: Mapping[str, Any], status: int = 200, sink: dict | None = None
) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        if sink is not None:
            sink["body"] = json.loads(request.content)
        return httpx.Response(status, content=ndjson(*payloads))

    return handler


def chunk(content: str, *, done: bool = False) -> dict[str, Any]:
    return {
        "model": MODEL,
        "message": {"role": "assistant", "content": content},
        "done": done,
    }


# --------------------------------------------------------------------- models


async def test_list_models_reads_a_real_payload(tags_payload: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json=tags_payload)

    async with make_client(handler) as client:
        models = await client.list_models()

    assert [model.name for model in models][:1] == ["deepseek-v4.1-flash:cloud"]
    assert "qwen3:8b" in {model.name for model in models}


async def test_client_uses_the_configured_base_url() -> None:
    client = OllamaClient(base_url="http://192.168.1.5:11434/")

    assert client.base_url == "http://192.168.1.5:11434"


async def test_client_reports_a_failed_model_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    async with make_client(handler) as client:
        with pytest.raises(OllamaProtocolError) as excinfo:
            await client.list_models()

    assert "HTTP 500" in excinfo.value.message
    assert excinfo.value.detail == "boom"


async def test_client_rejects_a_non_json_model_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    async with make_client(handler) as client:
        with pytest.raises(OllamaProtocolError) as excinfo:
            await client.list_models()

    assert "not valid JSON" in excinfo.value.message


async def test_client_rejects_an_unexpected_model_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"unexpected": True}]})

    async with make_client(handler) as client:
        with pytest.raises(OllamaProtocolError) as excinfo:
            await client.list_models()

    assert "unexpected format" in excinfo.value.message
    assert "Update Ollama" in (excinfo.value.hint or "")


async def test_list_models_wraps_connection_failures() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    async with make_client(handler) as client:
        with pytest.raises(OllamaNotRunningError):
            await client.list_models()


async def test_list_models_wraps_timeouts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    async with make_client(handler) as client:
        with pytest.raises(OllamaTimeoutError):
            await client.list_models()


# ----------------------------------------------------------------------- chat


async def test_stream_chat_yields_each_chunk() -> None:
    sink: dict = {}
    handler = chat_handler(
        chunk("Hello"), chunk(" world"), chunk("", done=True), sink=sink
    )

    async with make_client(handler) as client:
        replies = [
            incoming.content
            async for incoming in client.stream_chat(
                MODEL, [{"role": "user", "content": "hi"}]
            )
        ]

    assert replies == ["Hello", " world", ""]
    assert sink["body"] == {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }


async def test_stream_chat_passes_options_through() -> None:
    sink: dict = {}
    handler = chat_handler(chunk("ok", done=True), sink=sink)

    async with make_client(handler) as client:
        async for _ in client.stream_chat(
            MODEL,
            [{"role": "user", "content": "hi"}],
            options={"temperature": 0.2, "num_ctx": 8192},
        ):
            pass

    assert sink["body"]["options"] == {"temperature": 0.2, "num_ctx": 8192}


async def test_stream_chat_skips_blank_keepalive_lines() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=b"\n" + ndjson(chunk("hi", done=True)) + b"\n"
        )

    async with make_client(handler) as client:
        replies = [
            incoming.content async for incoming in client.stream_chat(MODEL, [])
        ]

    assert replies == ["hi"]


async def test_stream_chat_reports_a_missing_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404, json={"error": f"model '{MODEL}' not found"}
        )

    async with make_client(handler) as client:
        with pytest.raises(ModelNotFoundError) as excinfo:
            async for _ in client.stream_chat(MODEL, []):
                pass

    assert excinfo.value.model == MODEL
    assert MODEL in excinfo.value.message


async def test_stream_chat_translates_runner_crashes() -> None:
    crash = (
        "llama-server process has terminated: exit status 0xc0000409 "
        "(CUDA error: shared object initialization failed)"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": crash})

    async with make_client(handler) as client:
        with pytest.raises(OllamaProtocolError) as excinfo:
            async for _ in client.stream_chat(MODEL, []):
                pass

    error = excinfo.value
    assert error.message == f"Ollama could not start the model '{MODEL}'."
    assert "memory" in (error.hint or "")
    assert crash in (error.detail or "")


async def test_stream_chat_reports_an_error_object_mid_stream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=ndjson(chunk("partial"), {"error": "failed to load model"}),
        )

    async with make_client(handler) as client:
        received: list[str] = []
        with pytest.raises(OllamaProtocolError) as excinfo:
            async for incoming in client.stream_chat(MODEL, []):
                received.append(incoming.content)

    assert received == ["partial"]
    assert "could not start" in excinfo.value.message


async def test_stream_chat_rejects_unparsable_streams() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{not json}\n")

    async with make_client(handler) as client:
        with pytest.raises(OllamaProtocolError) as excinfo:
            async for _ in client.stream_chat(MODEL, []):
                pass

    assert "could not parse" in excinfo.value.message


async def test_stream_chat_rejects_non_object_payloads() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'["unexpected"]\n')

    async with make_client(handler) as client:
        with pytest.raises(OllamaProtocolError) as excinfo:
            async for _ in client.stream_chat(MODEL, []):
                pass

    assert "unexpected payload" in excinfo.value.message


async def test_stream_chat_rejects_malformed_chunks() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'{"message": "not an object"}\n')

    async with make_client(handler) as client:
        with pytest.raises(OllamaProtocolError) as excinfo:
            async for _ in client.stream_chat(MODEL, []):
                pass

    assert "unexpected format" in excinfo.value.message


async def test_stream_chat_reports_a_dropped_connection() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("reset", request=request)

    async with make_client(handler) as client:
        with pytest.raises(OllamaNotRunningError) as excinfo:
            async for _ in client.stream_chat(MODEL, []):
                pass

    assert "Lost the connection" in excinfo.value.message


async def test_stream_chat_reports_a_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("no data", request=request)

    async with make_client(handler) as client:
        with pytest.raises(OllamaTimeoutError) as excinfo:
            async for _ in client.stream_chat(MODEL, []):
                pass

    assert "stopped responding" in excinfo.value.message


async def test_stream_chat_wraps_other_transport_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("broken pipe", request=request)

    async with make_client(handler) as client:
        with pytest.raises(OllamaProtocolError) as excinfo:
            async for _ in client.stream_chat(MODEL, []):
                pass

    assert "mid-response" in excinfo.value.message


async def test_chunk_properties_expose_content_and_completion() -> None:
    stream = iter([chunk("a"), chunk("", done=True)])

    first = next(stream)
    last = next(stream)

    assert ChatChunk.model_validate(first).content == "a"
    assert ChatChunk.model_validate(last).done is True


async def test_close_is_idempotent() -> None:
    async with make_client(chat_handler(chunk("ok", done=True))) as client:
        await client.close()
        await client.close()


# ---------------------------------------------------------------------- tools


async def test_stream_chat_sends_tool_definitions() -> None:
    sink: dict = {}
    handler = chat_handler(chunk("ok", done=True), sink=sink)
    tools = [
        {
            "type": "function",
            "function": {"name": "read_file", "parameters": {"type": "object"}},
        }
    ]

    async with make_client(handler) as client:
        async for _ in client.stream_chat(MODEL, [], tools=tools):
            pass

    assert sink["body"]["tools"] == tools


async def test_stream_chat_omits_tools_when_none_are_given() -> None:
    sink: dict = {}
    handler = chat_handler(chunk("ok", done=True), sink=sink)

    async with make_client(handler) as client:
        async for _ in client.stream_chat(MODEL, []):
            pass

    assert "tools" not in sink["body"]


async def test_stream_chat_explains_a_model_without_tool_support() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": f"registry.ollama.ai/library/{MODEL} does not support tools"},
        )

    async with make_client(handler) as client:
        with pytest.raises(ToolCallingUnsupportedError) as excinfo:
            async for _ in client.stream_chat(MODEL, [], tools=[{"type": "function"}]):
                pass

    error = excinfo.value
    assert "does not support tools" in (error.detail or "")
    assert "tool calling" in (error.hint or "")


async def test_stream_chat_parses_tool_calls_from_the_stream() -> None:
    call_chunk = {
        "model": MODEL,
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "list_files", "arguments": {"path": "."}}}
            ],
        },
    }
    handler = chat_handler(call_chunk, chunk("", done=True))

    async with make_client(handler) as client:
        calls = [
            call
            async for incoming in client.stream_chat(MODEL, [])
            for call in incoming.tool_calls
        ]

    assert [call.name for call in calls] == ["list_files"]
