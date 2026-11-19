"""Headless TUI tests driven by Textual's ``run_test`` harness."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from textual.widgets import OptionList, Static

from beast.core.autonomy import RunSummary
from beast.core.changes import ChangeReport, ChangeTracker, GitStatus
from beast.core.events import (
    ChangeSummary,
    DebugAttempt,
    PlanReady,
    PlanStarted,
    RunFinished,
    TaskCompleted,
    TaskStarted,
    ValidationFinished,
    ValidationStarted,
)
from beast.core.planner import Plan, PlannedTask
from beast.core.task_manager import Task, TaskProgress, TaskStatus
from beast.core.validation import ValidationReport
from beast.errors import OllamaProtocolError
from beast.ollama.connection import OllamaStatus
from beast.ollama.models import ChatChunk, ModelInventory, OllamaModel, split_models
from beast.tools import ApprovalRequest, Permission, ToolResult
from beast.ui.app import BeastApp
from beast.ui.screens import ApprovalScreen, ChatScreen, ModelSelectScreen
from beast.ui.widgets import (
    AgentState,
    AssistantMessage,
    BrandHeader,
    CommandOutput,
    PromptArea,
    RunActivity,
    RunSummaryView,
    Splash,
    StatusBar,
    SystemNotice,
    TaskBoard,
    ToolActivity,
    UserMessage,
)
from beast.workspace import Workspace

BASE = "http://ollama.test"


class FakeClient:
    """Minimal stand-in for ``OllamaClient``."""

    def __init__(
        self,
        parts: Sequence[str] = ("Hello", " from BEAST"),
        *,
        error: Exception | None = None,
        delay: float = 0.0,
        models: Sequence[OllamaModel] = (),
    ) -> None:
        self.parts = parts
        self.error = error
        self.delay = delay
        self.models = list(models)
        self.calls: list[tuple[str, list[dict[str, str]]]] = []
        self.reloads = 0

    async def stream_chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        options: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[ChatChunk]:
        self.calls.append((model, [dict(message) for message in messages]))
        for index, part in enumerate(self.parts):
            if index and self.delay:
                await asyncio.sleep(self.delay)
            yield ChatChunk(
                model=model, message={"role": "assistant", "content": part}
            )
        if self.error is not None:
            raise self.error
        yield ChatChunk(model=model, message={"role": "assistant", "content": ""}, done=True)

    async def list_models(self) -> list[OllamaModel]:
        self.reloads += 1
        return self.models

    async def close(self) -> None:
        return None


def model(name: str, family: str = "qwen3") -> OllamaModel:
    return OllamaModel(
        name=name,
        size=5_000_000_000,
        details={"family": family, "quantization_level": "Q4_K_M"},
        capabilities=["completion"],
    )


QWEN = model("qwen3:8b")
LLAMA = model("llama3.2:3b", family="llama")
TOOL_QWEN = OllamaModel(
    name="qwen3:8b",
    size=5_000_000_000,
    details={"family": "qwen3", "quantization_level": "Q4_K_M"},
    capabilities=["completion", "tools"],
)
UNKNOWN_QWEN = OllamaModel(
    name="qwen3:8b",
    size=5_000_000_000,
    details={"family": "qwen3", "quantization_level": "Q4_K_M"},
)


def status() -> OllamaStatus:
    return OllamaStatus(
        base_url=BASE,
        is_installed=True,
        is_running=True,
        executable="ollama",
        version="0.33.2",
    )


def make_app(
    client: Any,
    *,
    initial: OllamaModel | None = None,
    models: Sequence[OllamaModel] = (QWEN, LLAMA),
    workspace: Path | None = None,
    auto: str | None = None,
) -> BeastApp:
    return BeastApp(
        workspace=Workspace(workspace or Path.cwd()),
        inventory=ModelInventory(selectable=tuple(sorted(models, key=lambda m: m.name))),
        status=status(),
        client=client,  # type: ignore[arg-type]
        initial_model=initial,
        initial_auto=auto,
    )


def text_of(widget: Any) -> str:
    rendered = widget.render()
    return rendered.plain if hasattr(rendered, "plain") else str(rendered)


async def run_turn(pilot: Any, screen: ChatScreen, prompt_text: str) -> None:
    prompt = screen.query_one(PromptArea)
    prompt.load_text(prompt_text)
    await pilot.press("enter")
    await pilot.pause(0.8)


# --------------------------------------------------------------- model picker


async def test_picker_lists_every_selectable_model() -> None:
    app = make_app(FakeClient())

    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ModelSelectScreen)
        options = app.screen.query_one("#models", OptionList)
        # One LOCAL MODELS heading plus the two local models.
        assert options.option_count == 3


async def test_picker_groups_local_and_cloud_models() -> None:
    inventory = split_models(
        [
            QWEN,
            OllamaModel(name="remote:cloud", remote_host="https://ollama.com"),
            OllamaModel(name="embed:latest", capabilities=["embedding"]),
        ]
    )
    app = BeastApp(
        workspace=Workspace(Path.cwd()),
        inventory=inventory,
        status=status(),
        client=FakeClient(),  # type: ignore[arg-type]
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        options = app.screen.query_one("#models", OptionList)
        prompts = [
            str(options.get_option_at_index(index).prompt).upper()
            for index in range(options.option_count)
        ]
        note = text_of(app.screen.query_one("#select-note"))

        # Two headings, one local model, one cloud model.
        assert options.option_count == 4
        assert any("LOCAL MODELS" in prompt for prompt in prompts)
        assert any("CLOUD MODELS" in prompt for prompt in prompts)
        assert any("REMOTE:CLOUD" in prompt for prompt in prompts)
        assert "Hidden: 1 embedding-only model." in note
        # The cloud model is a visible, selectable row, not a hidden one.
        assert not any("HIDDEN" in prompt for prompt in prompts)


async def test_picker_selects_a_cloud_model() -> None:
    cloud = OllamaModel(
        name="kimi-k2.7-code:cloud",
        remote_host="https://ollama.com",
        capabilities=["completion", "tools"],
    )
    inventory = split_models([QWEN, cloud])
    app = BeastApp(
        workspace=Workspace(Path.cwd()),
        inventory=inventory,
        status=status(),
        client=FakeClient(),  # type: ignore[arg-type]
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        options = app.screen.query_one("#models", OptionList)
        index = next(
            index
            for index in range(options.option_count)
            if options.get_option_at_index(index).id == cloud.identifier
        )
        options.highlighted = index
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, ChatScreen)
        assert app.screen.current_model == cloud.identifier


async def test_picker_labels_missing_tool_support() -> None:
    app = make_app(FakeClient(), models=(LLAMA, UNKNOWN_QWEN))

    async with app.run_test() as pilot:
        await pilot.pause()
        options = app.screen.query_one("#models", OptionList)
        prompts = [
            str(options.get_option_at_index(index).prompt)
            for index in range(options.option_count)
        ]

    assert any("no tool calling" in prompt for prompt in prompts)
    assert any("tool support unknown" in prompt for prompt in prompts)


async def test_choosing_a_model_opens_the_chat_screen() -> None:
    app = make_app(FakeClient())

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, ChatScreen)
        assert app.screen.current_model == "llama3.2:3b"


async def test_picker_highlights_the_active_model() -> None:
    client = FakeClient()
    app = make_app(client, initial=QWEN)

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+n")
        await pilot.pause()
        assert isinstance(app.screen, ModelSelectScreen)
        options = app.screen.query_one("#models", OptionList)
        # Index 0 is the LOCAL MODELS heading; qwen3:8b sorts after llama3.2:3b.
        assert options.highlighted == 2


async def test_escape_returns_from_the_picker_to_the_conversation() -> None:
    app = make_app(FakeClient(), initial=QWEN)

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+n")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, ChatScreen)


async def test_refreshing_the_picker_picks_up_new_models() -> None:
    client = FakeClient(models=[QWEN, LLAMA, model("new:1b", family="llama")])
    app = make_app(client)

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+r")
        await pilot.pause(0.3)
        options = app.screen.query_one("#models", OptionList)

    assert client.reloads == 1
    # One heading plus the three local models.
    assert options.option_count == 4


# --------------------------------------------------------------------- chat


async def test_sending_a_prompt_streams_a_reply() -> None:
    client = FakeClient()
    app = make_app(client, initial=QWEN)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        await run_turn(pilot, screen, "say hello")

        assert screen._state is AgentState.IDLE
        assert client.calls[0][0] == "qwen3:8b"
        assert client.calls[0][1][-1] == {"role": "user", "content": "say hello"}

        message = screen.query_one(AssistantMessage)
        assert message._body == "Hello from BEAST"
        assert screen.query_one(PromptArea).text == ""
        assert screen.query_one(StatusBar).render().plain.startswith("● IDLE")
        footer = screen.query_one(".message-footer")
        assert footer.display is False


async def test_the_prompt_is_cleared_and_a_user_message_is_shown() -> None:
    app = make_app(FakeClient(), initial=QWEN)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        await run_turn(pilot, screen, "explain this")

        assert len(list(screen.query(UserMessage))) == 1
        assert screen.query_one(PromptArea).text == ""


async def test_enter_does_nothing_while_a_reply_is_streaming() -> None:
    client = FakeClient(delay=30)
    app = make_app(client, initial=QWEN)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        prompt = screen.query_one(PromptArea)
        prompt.load_text("first question")
        await pilot.press("enter")
        await pilot.pause(0.2)

        assert screen._state is AgentState.THINKING
        assert prompt.can_submit is False

        prompt.load_text("second question")
        await pilot.press("enter")
        await pilot.pause(0.2)
        assert len(list(screen.query(UserMessage))) == 1
        assert len(client.calls) == 1

        await pilot.press("escape")
        await pilot.pause(0.3)


async def test_shift_enter_adds_a_newline_instead_of_sending() -> None:
    app = make_app(FakeClient(), initial=QWEN)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        prompt = screen.query_one(PromptArea)
        prompt.load_text("first line")
        prompt.move_cursor((0, len("first line")))

        await pilot.press("shift+enter")
        await pilot.pause()

        assert "\n" in prompt.text
        assert list(screen.query(UserMessage)) == []


async def test_a_streaming_error_is_reported_inline() -> None:
    error = OllamaProtocolError(
        "Ollama could not start the model 'qwen3:8b'.",
        hint="Check the GPU driver, then try a smaller model.",
    )
    app = make_app(FakeClient(error=error), initial=QWEN)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        await run_turn(pilot, screen, "hello")

        # Text received before the failure is shown, matching the history the
        # agent keeps for the next turn.
        assert screen.query_one(AssistantMessage)._body == "Hello from BEAST"

        footer = screen.query_one(".message-footer")
        assert footer.display is True
        content = text_of(footer)
        assert "Unexpected Ollama response" in content
        assert "could not start the model" in content
        assert "smaller model" in content
        assert screen._state is AgentState.IDLE
        assert screen.query_one(PromptArea).can_submit is True


async def test_an_empty_response_gets_an_explanation() -> None:
    app = make_app(FakeClient(parts=()), initial=QWEN)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        await run_turn(pilot, screen, "hello")

        footer = screen.query_one(".message-footer")
        assert footer.display is True
        assert "empty response" in text_of(footer)


async def test_escape_interrupts_a_running_reply() -> None:
    app = make_app(FakeClient(delay=30), initial=QWEN)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        prompt = screen.query_one(PromptArea)
        prompt.load_text("long answer please")
        await pilot.press("enter")
        await pilot.pause(0.2)
        assert screen._state is AgentState.THINKING

        await pilot.press("escape")
        await pilot.pause(0.4)

        assert screen._state is AgentState.IDLE
        footer = screen.query_one(".message-footer")
        assert footer.display is True
        assert "Stopped" in text_of(footer)
        # The partial turn stays in the transcript and in the agent's history.
        assert len(screen._agent.conversation) == 3


async def test_clear_resets_the_conversation() -> None:
    app = make_app(FakeClient(), initial=QWEN)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        await run_turn(pilot, screen, "hello")
        assert screen.query(AssistantMessage)

        await screen.action_clear()
        await pilot.pause()

        assert len(screen._agent.conversation) == 1
        assert list(screen.query(UserMessage)) == []
        assert list(screen.query(AssistantMessage)) == []
        assert len(list(screen.query(Splash))) == 1


async def test_switching_models_keeps_the_transcript_but_starts_a_new_context() -> None:
    client = FakeClient()
    app = make_app(client, initial=QWEN)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        await run_turn(pilot, screen, "hello")
        assert screen.query(AssistantMessage)

        await pilot.press("ctrl+n")
        await pilot.pause()
        await pilot.press("down")  # from qwen3:8b (index 1) to llama3.2:3b (index 0)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, ChatScreen)
        assert app.screen is screen
        assert screen.current_model == "llama3.2:3b"
        notices = [text_of(note) for note in screen.query(SystemNotice)]
        switched = [note for note in notices if "Switched to llama3.2:3b" in note]
        assert len(switched) == 1
        assert "new conversation" in switched[0]
        # Transcript is preserved; agent context is not.
        assert screen.query(AssistantMessage)
        assert len(screen._agent.conversation) == 1
        assert screen._agent.model == "llama3.2:3b"
        # The header and the welcome panel both show the new model.
        assert "llama3.2:3b" in text_of(screen.query_one("#brand-meta", Static))
        assert "llama3.2:3b" in text_of(screen.query_one(Splash))


async def test_the_header_shows_model_connection_and_workspace() -> None:
    app = make_app(FakeClient(), initial=QWEN)

    async with app.run_test() as pilot:
        await pilot.pause()
        header = app.screen.query_one(BrandHeader)
        meta = text_of(header.query_one("#brand-meta"))
        workspace = text_of(header.query_one("#brand-workspace"))

    assert "qwen3:8b" in meta
    assert "LOCAL / OLLAMA" in meta
    assert "Ollama v0.33.2" in meta
    assert "Workspace" in workspace


async def test_the_header_hides_the_logo_on_narrow_terminals() -> None:
    app = make_app(FakeClient(), initial=QWEN)

    async with app.run_test(size=(50, 24)) as pilot:
        await pilot.pause()
        assert app.screen.query_one("#brand-logo").display is False


async def test_the_header_keeps_the_logo_when_there_is_room() -> None:
    app = make_app(FakeClient(), initial=QWEN)

    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        assert app.screen.query_one("#brand-logo").display is True


@pytest.mark.parametrize(
    ("width", "visible", "hidden"),
    [
        (70, "IDLE", "LOCAL / OLLAMA"),
        (100, "LOCAL / OLLAMA", "ctrl+l clear"),
        (160, "ctrl+l clear", None),
    ],
)
async def test_the_status_bar_adapts_to_the_terminal_width(
    width: int, visible: str, hidden: str | None
) -> None:
    app = make_app(FakeClient(), initial=QWEN)

    async with app.run_test(size=(width, 24)) as pilot:
        await pilot.pause()
        status = app.screen.query_one(StatusBar).render().plain
        workspace = app.screen.query_one(StatusBar)._workspace.display()

    assert visible in status
    if hidden is not None:
        assert hidden not in status
    if width >= 112:
        assert workspace in status
    else:
        assert workspace not in status


# ------------------------------------------------------------------ tool calls


class ScriptedClient:
    """Emits one scripted response per request, then a short final answer."""

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

    async def list_models(self) -> list[OllamaModel]:
        return []

    async def close(self) -> None:
        return None


def tool_call(name: str, **arguments: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
    }


async def wait_for_screen(pilot: Any, kind: type, timeout: float = 3.0) -> Any:
    """Pause until ``kind`` is the top screen (or fail with context)."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if isinstance(pilot.app.screen, kind):
            return pilot.app.screen
        await pilot.pause(0.05)
    raise AssertionError(f"{kind.__name__} never appeared")


async def test_chat_only_models_get_a_visible_notice() -> None:
    app = make_app(FakeClient(), initial=QWEN)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        notices = [text_of(note) for note in screen.query(SystemNotice)]

    assert any("does not advertise tool calling" in note for note in notices)
    assert screen._agent.tools is None


async def test_models_without_capability_data_get_the_unknown_notice() -> None:
    app = make_app(FakeClient(), initial=UNKNOWN_QWEN)

    async with app.run_test() as pilot:
        await pilot.pause()
        notices = [text_of(note) for note in app.screen.query(SystemNotice)]

    assert any("did not report tool support" in note for note in notices)


async def test_tool_capable_models_get_the_full_toolset() -> None:
    app = make_app(ScriptedClient(), initial=TOOL_QWEN)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert screen._agent.tools is not None
        assert "run_command" in screen._agent.tools.names
        assert len(list(screen.query(SystemNotice))) == 0


async def test_a_write_waits_for_approval_and_then_runs(tmp_path: Path) -> None:
    client = ScriptedClient(
        tool_call("write_file", path="index.html", content="<html>hi</html>"),
        [{"role": "assistant", "content": "Created the page."}],
    )
    app = make_app(client, initial=TOOL_QWEN, workspace=tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        prompt = screen.query_one(PromptArea)
        prompt.load_text("create index.html")
        await pilot.press("enter")

        approval = await wait_for_screen(pilot, ApprovalScreen)
        assert "BEAST wants to modify" in text_of(
            approval.query_one("#approval-heading")
        )
        assert "index.html" in text_of(approval.query_one("#approval-target"))
        assert "Create new file" in text_of(approval.query_one("#approval-action"))
        assert screen._state is AgentState.APPROVAL
        assert not (tmp_path / "index.html").exists()

        await pilot.press("y")
        await pilot.pause(0.8)

        assert isinstance(app.screen, ChatScreen)
        assert (tmp_path / "index.html").read_text(encoding="utf-8") == "<html>hi</html>"
        activity = screen.query_one(ToolActivity)
        assert activity._phase == "ok"
        bodies = [message._body for message in screen.query(AssistantMessage)]
        assert any("Created the page." in body for body in bodies)
        assert screen._state is AgentState.IDLE


async def test_a_rejected_write_is_reported_and_never_applied(
    tmp_path: Path,
) -> None:
    client = ScriptedClient(
        tool_call("write_file", path="index.html", content="x"),
        [{"role": "assistant", "content": "Understood."}],
    )
    app = make_app(client, initial=TOOL_QWEN, workspace=tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        prompt = screen.query_one(PromptArea)
        prompt.load_text("create index.html")
        await pilot.press("enter")

        await wait_for_screen(pilot, ApprovalScreen)
        await pilot.press("n")
        await pilot.pause(0.8)

        assert isinstance(app.screen, ChatScreen)
        assert not (tmp_path / "index.html").exists()
        activity = screen.query_one(ToolActivity)
        assert activity._phase == "rejected"
        observation = client.calls[1]["messages"][-1]
        assert observation["role"] == "tool"
        assert "rejected" in observation["content"].lower()


async def test_unknown_tool_calls_are_shown_as_failures(tmp_path: Path) -> None:
    client = ScriptedClient(
        tool_call("format_disk", path="/"), [{"role": "assistant", "content": "No."}]
    )
    app = make_app(client, initial=TOOL_QWEN, workspace=tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        await run_turn(pilot, screen, "do something odd")

        activity = screen.query_one(ToolActivity)
        assert activity._phase == "failed"
        assert "Unknown tool" in activity._summary
        assert "✕ [TOOL] format_disk" in text_of(activity)


async def test_a_command_asks_for_execution_approval(tmp_path: Path) -> None:
    client = ScriptedClient(
        tool_call("run_command", command="echo hello"),
        [{"role": "assistant", "content": "Done."}],
    )
    app = make_app(client, initial=TOOL_QWEN, workspace=tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        prompt = screen.query_one(PromptArea)
        prompt.load_text("say hello")
        await pilot.press("enter")

        approval = await wait_for_screen(pilot, ApprovalScreen)
        assert "run a command" in text_of(approval.query_one("#approval-heading"))
        assert "echo hello" in text_of(approval.query_one("#approval-target"))

        await pilot.click("#reject")
        await pilot.pause(0.8)

        assert isinstance(app.screen, ChatScreen)
        assert screen.query_one(ToolActivity)._phase == "rejected"


# ------------------------------------------------------------ approval widget


async def test_approval_screen_escape_rejects() -> None:
    app = make_app(FakeClient(), initial=QWEN)
    request = ApprovalRequest(
        tool="write_file",
        permission=Permission.WRITE,
        target="index.html",
        action="Create new file",
        detail="<html>",
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        results: list[bool] = []
        app.push_screen(ApprovalScreen(request), results.append)
        await pilot.pause()
        screen = app.screen

        assert "index.html" in text_of(screen.query_one("#approval-target"))
        assert "<html>" in text_of(screen.query_one("#approval-detail"))

        await pilot.press("escape")
        await pilot.pause()

    assert results == [False]


async def test_approval_screen_button_approves() -> None:
    app = make_app(FakeClient(), initial=QWEN)
    request = ApprovalRequest(
        tool="run_command",
        permission=Permission.EXECUTE,
        target="echo hi",
        action="Run this command in the workspace",
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        results: list[bool] = []
        app.push_screen(ApprovalScreen(request), results.append)
        await pilot.pause()
        assert "run a command" in text_of(app.screen.query_one("#approval-heading"))

        await pilot.click("#approve")
        await pilot.pause()

    assert results == [True]


# -------------------------------------------------------------- tool activity


async def test_tool_activity_renders_each_phase() -> None:
    activity = ToolActivity(
        "write_file", {"path": "index.html"}, Permission.WRITE
    )
    assert "▸ [TOOL] write_file" in text_of(activity)
    assert "index.html" in text_of(activity)

    activity.mark_awaiting_approval()
    assert "awaiting approval" in text_of(activity)

    await activity.finish(ToolResult.success("Created index.html (3 lines)"))
    assert "✓ [TOOL] write_file" in text_of(activity)
    assert "Created index.html" in text_of(activity)

    failed = ToolActivity("read_file", {"path": "gone.txt"}, Permission.READ)
    await failed.finish(ToolResult.failure("File not found: gone.txt"))
    assert "✕ [TOOL] read_file" in text_of(failed)

    rejected = ToolActivity("run_command", {"command": "rm -rf ."}, Permission.EXECUTE)
    await rejected.finish(ToolResult.failure("Rejected", rejected=True))
    assert "⊘ [TOOL] run_command" in text_of(rejected)


async def test_tool_activity_marks_calls_that_never_reported() -> None:
    running = ToolActivity("run_command", {"command": "sleep 10"}, Permission.EXECUTE)
    running.mark_awaiting_approval()
    running.mark_interrupted()

    assert running._phase == "rejected"
    assert "Stopped by the user" in text_of(running)

    finished = ToolActivity("read_file", {"path": "a.txt"}, Permission.READ)
    await finished.finish(ToolResult.success("Read a.txt"))
    finished.mark_interrupted()
    assert finished._phase == "ok"


# ------------------------------------------------------------ slash commands


def track_change(
    root: Path, tracker: ChangeTracker, relative: str, before: str, after: str
) -> Path:
    target = root / relative
    target.write_text(before, encoding="utf-8")
    tracker.capture(target)
    target.write_text(after, encoding="utf-8")
    tracker.finalize(target)
    return target


def notices(screen: ChatScreen) -> list[str]:
    return [text_of(widget) for widget in screen.query(SystemNotice)]


async def test_help_lists_the_slash_commands_without_calling_the_model() -> None:
    client = FakeClient()
    app = make_app(client, initial=QWEN)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        await run_turn(pilot, screen, "/help")

        outputs = list(screen.query(CommandOutput))
        assert len(outputs) == 1
        text = text_of(outputs[0])
        assert "Slash commands" in text
        assert "/auto" in text
        assert "never commits" in text
        assert client.calls == []
        assert list(screen.query(UserMessage)) == []


async def test_an_empty_change_report_is_explained() -> None:
    app = make_app(FakeClient(), initial=QWEN)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        await run_turn(pilot, screen, "/changes")

        text = text_of(screen.query_one(CommandOutput))
        assert "No changes were made." in text


async def test_diff_without_changes_produces_a_notice() -> None:
    app = make_app(FakeClient(), initial=QWEN)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        await run_turn(pilot, screen, "/diff")

        assert any(
            "No BEAST-attributed changes to diff." in text
            for text in notices(screen)
        )
        assert len(list(screen.query(CommandOutput))) == 0


async def test_unknown_commands_are_rejected_gently() -> None:
    app = make_app(FakeClient(), initial=QWEN)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        await run_turn(pilot, screen, "/bogus thing")

        assert any(
            "Unknown command /bogus" in text for text in notices(screen)
        )
        assert list(screen.query(UserMessage)) == []


async def test_rollback_with_nothing_tracked_only_notifies() -> None:
    app = make_app(FakeClient(), initial=QWEN)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        await run_turn(pilot, screen, "/rollback")

        assert any("Nothing to roll back." in text for text in notices(screen))
        assert isinstance(app.screen, ChatScreen)


async def test_rollback_asks_for_approval_and_restores_on_yes(
    tmp_path: Path,
) -> None:
    app = make_app(FakeClient(), initial=QWEN, workspace=tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        target = track_change(
            tmp_path, screen._changes, "app.py", "original\n", "beast edit\n"
        )

        prompt = screen.query_one(PromptArea)
        prompt.load_text("/rollback")
        await pilot.press("enter")

        approval = await wait_for_screen(pilot, ApprovalScreen)
        assert "all BEAST changes" in text_of(
            approval.query_one("#approval-target")
        )
        assert "Restore BEAST-attributed files" in text_of(
            approval.query_one("#approval-action")
        )

        await pilot.press("y")
        await pilot.pause(0.8)

        assert target.read_text(encoding="utf-8") == "original\n"
        assert any(
            "restored  app.py" in text_of(output)
            for output in screen.query(CommandOutput)
        )


async def test_a_rejected_rollback_leaves_the_file_untouched(
    tmp_path: Path,
) -> None:
    app = make_app(FakeClient(), initial=QWEN, workspace=tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        target = track_change(
            tmp_path, screen._changes, "app.py", "original\n", "beast edit\n"
        )

        prompt = screen.query_one(PromptArea)
        prompt.load_text("/rollback")
        await pilot.press("enter")

        await wait_for_screen(pilot, ApprovalScreen)
        await pilot.press("n")
        await pilot.pause(0.8)

        assert target.read_text(encoding="utf-8") == "beast edit\n"
        assert any(
            "Rollback rejected; nothing was changed." in text
            for text in notices(screen)
        )


async def test_git_status_is_shown_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGitInspector:
        def __init__(self, root: Path) -> None:
            self.root = root

        async def status(self) -> GitStatus:
            return GitStatus(
                available=True, branch="main", changes=("?? new.txt",)
            )

    monkeypatch.setattr("beast.ui.screens.GitInspector", FakeGitInspector)
    app = make_app(FakeClient(), initial=QWEN)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        await run_turn(pilot, screen, "/git")

        text = text_of(screen.query_one(CommandOutput))
        assert "Git branch: main" in text
        assert "1 changed path(s)" in text
        assert "?? new.txt" in text


# ----------------------------------------------------------- autonomous mode


class FakeAutoRunner:
    """A deterministic stand-in for ``AutonomousRunner``."""

    instances: list["FakeAutoRunner"] = []

    def __init__(self, client: Any, model: str, **kwargs: Any) -> None:
        self.client = client
        self.model = model
        self.options = kwargs
        self.requests: list[str] = []
        self.finished = False
        self.stop_requested = False
        FakeAutoRunner.instances.append(self)

    @staticmethod
    def events(request: str) -> list[Any]:
        pending = Task(id="t1", title="Inspect the app")
        running = pending.transition(TaskStatus.IN_PROGRESS, attempts=1)
        done = running.transition(TaskStatus.COMPLETED, verified=True)
        plan = Plan(
            goal=request,
            tasks=(PlannedTask(id="t1", title="Inspect the app"),),
        )
        initial = TaskProgress.initial(1)
        final = TaskProgress(
            total=1,
            completed=1,
            failed=0,
            blocked=0,
            cancelled=0,
            pending=0,
            running=0,
        )
        summary = RunSummary(
            goal=request,
            tasks=(done,),
            progress=final,
            changes=ChangeReport(),
            duration=2.0,
            validation_runs=1,
            debug_attempts=1,
            model="qwen3:8b",
        )
        return [
            PlanStarted(request=request, model="qwen3:8b"),
            PlanReady(plan=plan, tasks=(pending,)),
            TaskStarted(task=running, progress=initial),
            ValidationStarted(task_id="t1", files=("app.py",)),
            ValidationFinished(task_id="t1", report=ValidationReport()),
            DebugAttempt(
                task_id="t1",
                attempt=1,
                max_attempts=2,
                signature="abc",
                evidence="boom",
            ),
            TaskCompleted(task=done, progress=final),
            ChangeSummary(report=ChangeReport()),
            RunFinished(summary=summary),
        ]

    async def run(self, request: str, context: str = "") -> AsyncIterator[Any]:
        self.requests.append(request)
        for event in self.events(request):
            yield event
        self.finished = True

    def request_stop(self) -> None:
        self.stop_requested = True


async def test_auto_refuses_a_model_without_tool_support() -> None:
    app = make_app(FakeClient(), initial=QWEN)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        await run_turn(pilot, screen, "/auto build a page")

        assert any("cannot use tools" in text for text in notices(screen))
        assert screen._runner is None
        assert screen.query_one(TaskBoard).display is False


async def test_auto_requires_a_request() -> None:
    app = make_app(FakeClient(), initial=TOOL_QWEN)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        await run_turn(pilot, screen, "/auto")

        assert any("Usage: /auto <request>" in text for text in notices(screen))


async def test_auto_runs_the_board_and_reports_the_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeAutoRunner.instances.clear()
    monkeypatch.setattr("beast.ui.screens.AutonomousRunner", FakeAutoRunner)
    app = make_app(FakeClient(), initial=TOOL_QWEN)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        await run_turn(pilot, screen, "/auto build the page")

        assert len(FakeAutoRunner.instances) == 1
        runner = FakeAutoRunner.instances[0]
        assert runner.requests == ["build the page"]
        assert runner.options["supports_tools"] is True
        assert runner.options["tracker"] is screen._changes
        assert runner.options["approver"] is app
        assert runner.model == "qwen3:8b"

        user_messages = [widget._text for widget in screen.query(UserMessage)]
        assert user_messages == ["/auto build the page"]

        board = screen.query_one(TaskBoard)
        assert board.display is True
        board_text = text_of(board)
        assert "TASKS" in board_text
        assert "Inspect the app" in board_text
        assert "1/1 tasks" in board_text
        assert "COMPLETED" in board_text

        kinds = {activity._kind for activity in screen.query(RunActivity)}
        assert {"plan", "validation", "debug"} <= kinds

        summaries = list(screen.query(RunSummaryView))
        assert len(summaries) == 1
        summary_text = text_of(summaries[0])
        assert "Run completed — 1/1 tasks completed" in summary_text
        assert "Validation: 1 verified" in summary_text
        assert "Changes: 0 created, 0 modified, 0 deleted" in summary_text

        assert screen._state is AgentState.IDLE
        assert screen.query_one(PromptArea).can_submit is True


async def test_initial_auto_starts_the_run_without_a_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeAutoRunner.instances.clear()
    monkeypatch.setattr("beast.ui.screens.AutonomousRunner", FakeAutoRunner)
    app = make_app(FakeClient(), initial=TOOL_QWEN, auto="add dark mode")

    async with app.run_test() as pilot:
        await pilot.pause(1.0)
        screen = app.screen

        assert len(FakeAutoRunner.instances) == 1
        assert FakeAutoRunner.instances[0].requests == ["add dark mode"]
        assert [widget._text for widget in screen.query(UserMessage)] == [
            "/auto add dark mode"
        ]
        assert screen.query_one(TaskBoard).display is True
        assert len(list(screen.query(RunSummaryView))) == 1


async def test_initial_auto_still_refuses_a_chat_only_model() -> None:
    app = make_app(FakeClient(), initial=QWEN, auto="add dark mode")

    async with app.run_test() as pilot:
        await pilot.pause(1.0)
        screen = app.screen

        assert any("cannot use tools" in text for text in notices(screen))
        assert screen._runner is None
        assert screen.query_one(TaskBoard).display is False


async def test_ctrl_o_toggles_the_task_board() -> None:
    app = make_app(FakeClient(), initial=QWEN)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        board = screen.query_one(TaskBoard)
        assert board.display is False

        await pilot.press("ctrl+o")
        await pilot.pause()
        assert board.display is True

        await pilot.press("ctrl+o")
        await pilot.pause()
        assert board.display is False


