"""The agent seam: conversation history, streaming, model pinning."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import pytest

from beast.core.agent import SYSTEM_PROMPT, Agent
from beast.errors import OllamaProtocolError
from beast.ollama.models import ChatChunk


class FakeStreamer:
    """Stands in for ``OllamaClient`` and records every request."""

    def __init__(self, *parts: str) -> None:
        self.parts = parts
        self.calls: list[tuple[str, list[dict[str, str]], Mapping[str, Any] | None]] = []

    async def stream_chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        options: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[ChatChunk]:
        self.calls.append((model, [dict(message) for message in messages], options))
        for part in self.parts:
            yield ChatChunk(model=model, message={"role": "assistant", "content": part})
        yield ChatChunk(model=model, message={"role": "assistant", "content": ""}, done=True)


def roles(agent: Agent) -> list[str]:
    return [message.role.value for message in agent.conversation]


def contents(agent: Agent) -> list[str]:
    return [message.content for message in agent.conversation]


async def test_stream_yields_each_delta() -> None:
    agent = Agent(FakeStreamer("Hello", " there"), "qwen3:8b")

    deltas = [delta async for delta in agent.stream("hi")]

    assert deltas == ["Hello", " there"]


async def test_stream_sends_the_selected_model() -> None:
    fake = FakeStreamer("ok")
    agent = Agent(fake, "llama3.2:3b")

    [delta async for delta in agent.stream("hi")]

    assert fake.calls[0][0] == "llama3.2:3b"


async def test_stream_sends_system_prompt_then_history() -> None:
    fake = FakeStreamer("one", " two")
    agent = Agent(fake, "m")

    [delta async for delta in agent.stream("first")]
    [delta async for delta in agent.stream("second")]

    first_request = fake.calls[0][1]
    second_request = fake.calls[1][1]
    assert first_request[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert first_request[1] == {"role": "user", "content": "first"}
    assert [message["role"] for message in second_request] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert second_request[2] == {"role": "assistant", "content": "one two"}
    assert second_request[3] == {"role": "user", "content": "second"}


async def test_history_records_the_full_reply() -> None:
    agent = Agent(FakeStreamer("Hi", " there"), "m")

    [delta async for delta in agent.stream("hello")]

    assert roles(agent) == ["system", "user", "assistant"]
    assert contents(agent)[-1] == "Hi there"


async def test_empty_prompts_are_rejected_before_any_request() -> None:
    fake = FakeStreamer("unused")
    agent = Agent(fake, "m")

    with pytest.raises(ValueError):
        [delta async for delta in agent.stream("   ")]

    assert fake.calls == []
    assert roles(agent) == ["system"]


async def test_a_failed_stream_keeps_the_partial_reply() -> None:
    class FailingStreamer:
        async def stream_chat(
            self,
            model: str,
            messages: Sequence[Mapping[str, Any]],
            *,
            options: Mapping[str, Any] | None = None,
        ) -> AsyncIterator[ChatChunk]:
            yield ChatChunk(message={"role": "assistant", "content": "partial"})
            raise OllamaProtocolError("Ollama stopped.")

    agent = Agent(FailingStreamer(), "m")
    received: list[str] = []

    with pytest.raises(OllamaProtocolError):
        async for delta in agent.stream("go"):
            received.append(delta)

    assert received == ["partial"]
    assert contents(agent)[-1] == "partial"


async def test_an_interrupted_stream_keeps_what_was_shown() -> None:
    class SlowStreamer:
        async def stream_chat(
            self,
            model: str,
            messages: Sequence[Mapping[str, Any]],
            *,
            options: Mapping[str, Any] | None = None,
        ) -> AsyncIterator[ChatChunk]:
            yield ChatChunk(message={"role": "assistant", "content": "first"})
            await asyncio.sleep(30)
            yield ChatChunk(message={"role": "assistant", "content": "never"})

    agent = Agent(SlowStreamer(), "m")
    stream = agent.stream("go")

    assert await anext(stream) == "first"
    await stream.aclose()

    assert contents(agent)[-1] == "first"


async def test_reset_keeps_the_model_but_forgets_history() -> None:
    fake = FakeStreamer("ok")
    agent = Agent(fake, "m")
    [delta async for delta in agent.stream("hi")]

    agent.reset()

    assert agent.model == "m"
    assert roles(agent) == ["system"]


async def test_options_reach_the_client() -> None:
    fake = FakeStreamer("ok")
    agent = Agent(fake, "m", options={"temperature": 0.1})

    [delta async for delta in agent.stream("hi")]

    assert fake.calls[0][2] == {"temperature": 0.1}
