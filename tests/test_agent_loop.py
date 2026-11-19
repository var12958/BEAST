"""The THINK → ACT → OBSERVE loop driven by a scripted model."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import Field

from beast.core.agent import SYSTEM_PROMPT, TOOL_SYSTEM_PROMPT, Agent
from beast.core.events import ApprovalPending, TextDelta, ToolFinished, ToolStarted
from beast.errors import AgentIterationLimitError
from beast.ollama.models import ChatChunk
from beast.tools import ApprovalRequest, Permission, Tool, ToolContext, ToolRegistry
from beast.tools.schemas import ToolArgs
from beast.workspace import Workspace


class Approver:
    def __init__(self, answer: bool = True) -> None:
        self.answer = answer
        self.requests: list[ApprovalRequest] = []

    async def approve(self, request: ApprovalRequest) -> bool:
        self.requests.append(request)
        return self.answer


class ScriptedClient:
    """Replays one scripted response per request."""

    def __init__(self, *turns: Sequence[Mapping[str, Any]]) -> None:
        self.turns = list(turns)
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
                "model": model,
                "messages": [dict(message) for message in messages],
                "tools": tools,
            }
        )
        turn = (
            self.turns.pop(0)
            if self.turns
            else [{"role": "assistant", "content": "done"}]
        )
        if isinstance(turn, Mapping):
            turn = [turn]
        for message in turn:
            yield ChatChunk(model=model, message=dict(message))
        yield ChatChunk(
            model=model, message={"role": "assistant", "content": ""}, done=True
        )


def text(content: str) -> dict[str, Any]:
    return {"role": "assistant", "content": content}


def call(name: str, **arguments: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
    }


def calls(*entries: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": name, "arguments": arguments}}
            for name, arguments in entries
        ],
    }


def workspace(tmp_path: Path) -> tuple[ToolContext, Approver]:
    approver = Approver()
    return (
        ToolContext(workspace=Workspace(tmp_path), approver=approver),
        approver,
    )


async def collect(agent: Agent, prompt: str) -> list[Any]:
    return [event async for event in agent.run(prompt)]


# --------------------------------------------------------------- loop behaviour


async def test_the_model_drives_the_tool_sequence(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    ctx, _ = workspace(tmp_path)
    client = ScriptedClient(
        call("list_files", path="."),
        call("read_file", path="a.txt"),
        [text("The file contains alpha.")],
    )
    agent = Agent(client, "qwen3:8b", tools=ToolRegistry.default(ctx))

    events = await collect(agent, "what is in a.txt?")

    started = [event.name for event in events if isinstance(event, ToolStarted)]
    finished = [event.name for event in events if isinstance(event, ToolFinished)]
    assert started == ["list_files", "read_file"]
    assert finished == ["list_files", "read_file"]
    assert all(event.result.ok for event in events if isinstance(event, ToolFinished))
    assert [event.text for event in events if isinstance(event, TextDelta)] == [
        "The file contains alpha."
    ]
    assert len(client.calls) == 3


async def test_tool_results_go_back_to_the_model(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    ctx, _ = workspace(tmp_path)
    client = ScriptedClient(call("read_file", path="a.txt"), [text("ok")])
    agent = Agent(client, "m", tools=ToolRegistry.default(ctx))

    await collect(agent, "read it")

    second = client.calls[1]["messages"]
    assert second[-2]["role"] == "assistant"
    assert second[-2]["tool_calls"][0]["function"]["name"] == "read_file"
    assert second[-1] == {
        "role": "tool",
        "content": "alpha",
        "tool_name": "read_file",
    }


async def test_multiple_calls_in_one_turn_all_execute(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    ctx, _ = workspace(tmp_path)
    client = ScriptedClient(
        calls(("read_file", {"path": "a.txt"}), ("read_file", {"path": "b.txt"})),
        [text("both")],
    )
    agent = Agent(client, "m", tools=ToolRegistry.default(ctx))
    approver = ctx.approver

    events = await collect(agent, "read both")

    finished = [event for event in events if isinstance(event, ToolFinished)]
    assert [event.result.content for event in finished] == ["alpha", "beta"]
    tool_messages = [
        message for message in client.calls[1]["messages"] if message["role"] == "tool"
    ]
    assert [message["content"] for message in tool_messages] == ["alpha", "beta"]
    assert approver.requests == []  # reads are automatic


async def test_unknown_tools_are_reported_to_the_model(tmp_path: Path) -> None:
    ctx, _ = workspace(tmp_path)
    client = ScriptedClient(
        call("delete_everything", path="."), [text("I will not.")]
    )
    agent = Agent(client, "m", tools=ToolRegistry.default(ctx))

    events = await collect(agent, "delete it")

    finished = [event for event in events if isinstance(event, ToolFinished)]
    assert finished[0].result.ok is False
    assert "Unknown tool" in finished[0].result.summary
    observation = client.calls[1]["messages"][-1]
    assert observation["role"] == "tool"
    assert "There is no tool named 'delete_everything'" in observation["content"]


async def test_malformed_arguments_are_answered_without_running(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    ctx, _ = workspace(tmp_path)
    client = ScriptedClient(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": "not json"}}
                ],
            }
        ],
        [text("Sorry.")],
    )
    agent = Agent(client, "m", tools=ToolRegistry.default(ctx))

    events = await collect(agent, "read a.txt")

    finished = [event for event in events if isinstance(event, ToolFinished)]
    assert finished[0].result.ok is False
    observation = client.calls[1]["messages"][-1]
    assert "not a JSON object" in observation["content"]


async def test_invalid_arguments_are_validated_before_execution(
    tmp_path: Path,
) -> None:
    ctx, _ = workspace(tmp_path)
    client = ScriptedClient(call("read_file", path=123), [text("ok")])
    agent = Agent(client, "m", tools=ToolRegistry.default(ctx))

    events = await collect(agent, "read 123")

    finished = [event for event in events if isinstance(event, ToolFinished)]
    assert finished[0].result.ok is False
    assert "Invalid arguments" in finished[0].result.summary


async def test_the_loop_stops_at_the_iteration_limit(tmp_path: Path) -> None:
    ctx, _ = workspace(tmp_path)
    client = ScriptedClient(*([call("list_files", path=".")] * 5))
    agent = Agent(
        client, "m", tools=ToolRegistry.default(ctx), max_iterations=3
    )

    with pytest.raises(AgentIterationLimitError) as excinfo:
        await collect(agent, "loop forever")

    assert excinfo.value.limit == 3
    assert len(client.calls) == 3


async def test_writes_require_approval_and_a_refusal_is_reported(
    tmp_path: Path,
) -> None:
    ctx, approver = workspace(tmp_path)
    approver.answer = False
    client = ScriptedClient(
        call("write_file", path="new.txt", content="hello"), [text("Not written.")]
    )
    agent = Agent(client, "m", tools=ToolRegistry.default(ctx))

    events = await collect(agent, "create new.txt")

    assert len(approver.requests) == 1
    assert isinstance(events[1], ApprovalPending)
    finished = [event for event in events if isinstance(event, ToolFinished)]
    assert finished[0].result.rejected is True
    assert not (tmp_path / "new.txt").exists()
    observation = client.calls[1]["messages"][-1]
    assert "rejected" in observation["content"].lower()


async def test_approved_writes_apply_and_the_loop_continues(tmp_path: Path) -> None:
    ctx, approver = workspace(tmp_path)
    client = ScriptedClient(
        call("write_file", path="new.txt", content="hello"), [text("Created.")]
    )
    agent = Agent(client, "m", tools=ToolRegistry.default(ctx))

    events = await collect(agent, "create new.txt")

    assert len(approver.requests) == 1
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "hello"
    assert isinstance(events[-1], TextDelta)
    assert events[-1].text == "Created."


# ---------------------------------------------------------------- configuration


async def test_tool_definitions_are_sent_with_each_request(tmp_path: Path) -> None:
    ctx, _ = workspace(tmp_path)
    client = ScriptedClient([text("hi")])
    agent = Agent(client, "m", tools=ToolRegistry.default(ctx))

    await collect(agent, "hello")

    definitions = client.calls[0]["tools"]
    assert definitions is not None
    assert {entry["function"]["name"] for entry in definitions} == set(
        ToolRegistry.default(ctx).names
    )


async def test_chat_only_agents_send_no_tools_and_use_the_chat_prompt() -> None:
    client = ScriptedClient([text("hi")])
    agent = Agent(client, "m")

    await collect(agent, "hello")

    assert client.calls[0]["tools"] is None
    assert client.calls[0]["messages"][0] == {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }
    assert SYSTEM_PROMPT != TOOL_SYSTEM_PROMPT


async def test_agent_tools_are_visible_and_reset_keeps_them(tmp_path: Path) -> None:
    ctx, _ = workspace(tmp_path)
    client = ScriptedClient([text("hi")], [text("again")])
    registry = ToolRegistry.default(ctx)
    agent = Agent(client, "m", tools=registry)

    await collect(agent, "one")
    agent.reset()

    assert agent.tools is registry
    assert len(agent.conversation) == 1
    assert agent.conversation.messages[0].content == TOOL_SYSTEM_PROMPT


# ------------------------------------------------------------------ stopping


class SlowArgs(ToolArgs):
    seconds: float = Field(default=30.0)


class SlowTool(Tool):
    name = "slow"
    description = "Waits, for cancellation tests."
    permission = Permission.READ
    args_model = SlowArgs

    async def execute(self, args: SlowArgs):  # type: ignore[override]
        await asyncio.sleep(args.seconds)
        from beast.tools import ToolResult

        return ToolResult.success("finished")


async def test_cancelling_during_a_tool_keeps_the_conversation_valid(
    tmp_path: Path,
) -> None:
    ctx, _ = workspace(tmp_path)
    client = ScriptedClient(call("slow", seconds=30), [text("never")])
    agent = Agent(client, "m", tools=ToolRegistry([SlowTool(ctx)]))

    async def drive() -> None:
        async for _ in agent.run("wait"):
            pass

    task = asyncio.create_task(drive())
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    roles = [message.role.value for message in agent.conversation]
    assert roles == ["system", "user", "assistant", "tool"]
    assert agent.conversation.messages[-1].tool_name == "slow"
    assert "stopped" in agent.conversation.messages[-1].content.lower()


class StallingClient:
    """Emits a tool call and then stalls before the turn can finish."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    async def stream_chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        options: Mapping[str, Any] | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> AsyncIterator[ChatChunk]:
        self.calls.append([dict(message) for message in messages])
        yield ChatChunk(
            model=model,
            message={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "list_files", "arguments": {"path": "."}}}
                ],
            },
        )
        await asyncio.sleep(30)


async def test_cancelling_while_the_model_streams_answers_its_tool_calls(
    tmp_path: Path,
) -> None:
    ctx, _ = workspace(tmp_path)
    agent = Agent(
        StallingClient(), "m", tools=ToolRegistry.default(ctx)
    )

    async def drive() -> None:
        async for _ in agent.run("look around"):
            pass

    task = asyncio.create_task(drive())
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    roles = [message.role.value for message in agent.conversation]
    assert roles == ["system", "user", "assistant", "tool"]
    assert agent.conversation.messages[-1].tool_name == "list_files"
    assert "interrupted" in agent.conversation.messages[-1].content.lower()


async def test_stopping_at_tool_finished_leaves_the_result_recorded(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    ctx, _ = workspace(tmp_path)
    agent = Agent(
        ScriptedClient(call("read_file", path="a.txt"), [text("never")]),
        "m",
        tools=ToolRegistry.default(ctx),
    )

    async for event in agent.run("read it"):
        if isinstance(event, ToolFinished):
            break

    # The observation is recorded before the UI is told, so stopping the
    # consumer cannot leave an unanswered tool call in the history.
    assert agent.conversation.messages[-1].role.value == "tool"
    assert agent.conversation.messages[-1].content == "alpha"


async def test_stopping_before_a_tool_runs_still_answers_the_call(
    tmp_path: Path,
) -> None:
    ctx, approver = workspace(tmp_path)
    agent = Agent(
        ScriptedClient(
            call("write_file", path="new.txt", content="hello"), [text("never")]
        ),
        "m",
        tools=ToolRegistry.default(ctx),
    )

    stream = agent.run("create new.txt")
    async for event in stream:
        if isinstance(event, ApprovalPending):
            break
    await stream.aclose()

    assert approver.requests == []
    assert not (tmp_path / "new.txt").exists()
    assert [message.role.value for message in agent.conversation] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert "stopped before this tool reported" in (
        agent.conversation.messages[-1].content.lower()
    )


async def test_non_object_tool_arguments_are_answered_without_crashing(
    tmp_path: Path,
) -> None:
    ctx, _ = workspace(tmp_path)
    client = ScriptedClient(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "list_files", "arguments": ["."]}}
                ],
            }
        ],
        [text("ok")],
    )
    agent = Agent(client, "m", tools=ToolRegistry.default(ctx))

    events = await collect(agent, "list it")

    finished = [event for event in events if isinstance(event, ToolFinished)]
    assert finished[0].result.ok is False
    assert "Malformed arguments" in finished[0].result.summary
    assert "not a JSON object" in client.calls[1]["messages"][-1]["content"]
