"""Agent integration: project context is evidence for one turn, not history."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from beast.core.agent import SYSTEM_PROMPT, Agent
from beast.core.events import AnalysisEvent, ContextReady, TextDelta
from beast.intelligence import ProjectIntelligence
from beast.intelligence.models import ContextBundle
from beast.ollama.models import ChatChunk
from beast.tools import ApprovalRequest, ToolContext, ToolRegistry
from beast.workspace import Workspace


class ScriptedClient:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def stream_chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        options: Mapping[str, Any] | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> AsyncIterator[ChatChunk]:
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "tools": tools,
            }
        )
        content = self.responses.pop(0) if self.responses else "done"
        yield ChatChunk(model=model, message={"role": "assistant", "content": content})
        yield ChatChunk(model=model, message={"role": "assistant", "content": ""}, done=True)


class FakeProvider:
    def __init__(self, bundle: ContextBundle | Exception) -> None:
        self.bundle = bundle
        self.queries: list[str] = []

    def context_for(self, query: str) -> ContextBundle:
        self.queries.append(query)
        if isinstance(self.bundle, Exception):
            raise self.bundle
        return self.bundle


class Approver:
    async def approve(self, request: ApprovalRequest) -> bool:
        return True


def make_bundle(
    text: str = "## Project context\n### `auth.py`\n",
    *,
    files: tuple[str, ...] = ("auth.py",),
    symbols: int = 4,
    redactions: int = 0,
) -> ContextBundle:
    return ContextBundle(
        text=text,
        query="q",
        files=files,
        symbol_count=symbols,
        characters=len(text),
        redactions=redactions,
    )


async def collect(agent: Agent, prompt: str) -> list[Any]:
    return [event async for event in agent.run(prompt)]


async def test_context_is_attached_for_one_turn_only() -> None:
    client = ScriptedClient("first answer", "second answer")
    provider = FakeProvider(make_bundle(redactions=2))
    agent = Agent(client, "m", context_provider=provider)

    events = await collect(agent, "Find the authentication implementation.")

    assert [event.stage for event in events if isinstance(event, AnalysisEvent)] == [
        "scan",
        "context",
    ]
    assert any(isinstance(event, ContextReady) for event in events)
    assert [event.text for event in events if isinstance(event, TextDelta)] == [
        "first answer"
    ]
    assert provider.queries == ["Find the authentication implementation."]

    first_request = client.calls[0]["messages"]
    assert first_request[0]["role"] == "system"
    assert "## Project context" in first_request[0]["content"]
    context_messages = [
        event.message
        for event in events
        if isinstance(event, AnalysisEvent) and event.stage == "context"
    ]
    assert context_messages
    assert "2 secrets redacted." in context_messages[0]
    # The stored conversation never contains transient project context.
    assert agent.conversation.messages[0].content == SYSTEM_PROMPT
    assert all(
        "## Project context" not in message.content
        for message in agent.conversation.messages
    )

    await collect(agent, "Now the second question.")

    second_request = client.calls[1]["messages"]
    assert second_request[0]["content"].count("## Project context") == 1
    assert all(
        "## Project context" not in message.content
        for message in agent.conversation.messages
    )


async def test_analysis_failure_does_not_block_the_answer() -> None:
    client = ScriptedClient("answer anyway")
    provider = FakeProvider(RuntimeError("index exploded"))
    agent = Agent(client, "m", context_provider=provider)

    events = await collect(agent, "Explain the router.")

    errors = [
        event
        for event in events
        if isinstance(event, AnalysisEvent) and event.stage == "error"
    ]
    assert errors
    assert "index exploded" in errors[0].detail
    assert not any(isinstance(event, ContextReady) for event in events)
    assert [event.text for event in events if isinstance(event, TextDelta)] == [
        "answer anyway"
    ]
    assert client.calls[0]["messages"][0]["content"] == SYSTEM_PROMPT


async def test_empty_context_is_reported_and_not_attached() -> None:
    client = ScriptedClient("ok")
    provider = FakeProvider(make_bundle(text="   \n", files=(), symbols=0))
    agent = Agent(client, "m", context_provider=provider)

    events = await collect(agent, "Anything?")

    messages = [
        event.message
        for event in events
        if isinstance(event, AnalysisEvent) and event.stage == "context"
    ]
    assert messages == ["No relevant project context was found."]
    assert not any(isinstance(event, ContextReady) for event in events)
    assert client.calls[0]["messages"][0]["content"] == SYSTEM_PROMPT


async def test_agent_without_a_provider_emits_no_analysis_events() -> None:
    client = ScriptedClient("plain")
    agent = Agent(client, "m")

    events = await collect(agent, "hello")

    assert not any(isinstance(event, AnalysisEvent) for event in events)
    assert client.calls[0]["messages"][0]["content"] == SYSTEM_PROMPT


async def test_project_info_tool_is_registered_only_with_intelligence(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path)
    approver = Approver()
    plain = ToolContext(workspace=workspace, approver=approver)
    intelligence = ProjectIntelligence(tmp_path)

    try:
        assert ToolRegistry.default(plain).names == (
            "list_files",
            "read_file",
            "search_files",
            "write_file",
            "edit_file",
            "run_command",
        )
        with_intelligence = ToolContext(
            workspace=workspace,
            approver=approver,
            intelligence=intelligence,
        )
        names = ToolRegistry.default(with_intelligence).names
        assert names[-1] == "project_info"
        assert len(names) == 7
    finally:
        intelligence.close()


def test_run_can_be_driven_with_asyncio_run() -> None:
    # Guards the public entry point used by headless callers.
    client = ScriptedClient("headless")
    agent = Agent(client, "m", context_provider=FakeProvider(make_bundle()))

    events = asyncio.run(collect(agent, "hi"))

    assert [event.text for event in events if isinstance(event, TextDelta)] == [
        "headless"
    ]
