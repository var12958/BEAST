"""TUI integration: project indicators, the analysis line and tool wiring.

The dashboard stays the same; Phase 3 only fills in the project rows on the
welcome panel, shows a single ``[ANALYSIS]`` line per turn, and offers the
``project_info`` tool to tool-capable models.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from beast.core.agent import SYSTEM_PROMPT, TOOL_SYSTEM_PROMPT
from beast.ollama.connection import OllamaStatus
from beast.ollama.models import ChatChunk, ModelInventory, OllamaModel
from beast.ui.app import BeastApp
from beast.ui.screens import ChatScreen
from beast.ui.widgets import AnalysisActivity, PromptArea, Splash
from beast.workspace import Workspace

BASE = "http://ollama.test"

CHAT_MODEL = OllamaModel(
    name="qwen3:8b",
    size=5_000_000_000,
    details={"family": "qwen3", "quantization_level": "Q4_K_M"},
    capabilities=["completion"],
)
TOOL_MODEL = OllamaModel(
    name="qwen3:8b",
    size=5_000_000_000,
    details={"family": "qwen3", "quantization_level": "Q4_K_M"},
    capabilities=["completion", "tools"],
)

AUTH_QUESTION = (
    "Find the authentication implementation in this project and explain how it works."
)


class ChatClient:
    """Chat-only client: the agent never sends a ``tools`` argument."""

    def __init__(self, content: str = "answer") -> None:
        self.content = content
        self.calls: list[list[dict[str, Any]]] = []

    async def stream_chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        options: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[ChatChunk]:
        self.calls.append([dict(message) for message in messages])
        yield ChatChunk(
            model=model, message={"role": "assistant", "content": self.content}
        )
        yield ChatChunk(
            model=model, message={"role": "assistant", "content": ""}, done=True
        )

    async def list_models(self) -> list[OllamaModel]:
        return []

    async def close(self) -> None:
        return None


class ToolClient:
    """Tool-capable client that records the tools payload as well as messages."""

    def __init__(self, content: str = "answer") -> None:
        self.content = content
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
        yield ChatChunk(
            model=model, message={"role": "assistant", "content": self.content}
        )
        yield ChatChunk(
            model=model, message={"role": "assistant", "content": ""}, done=True
        )

    async def list_models(self) -> list[OllamaModel]:
        return []

    async def close(self) -> None:
        return None


def status() -> OllamaStatus:
    return OllamaStatus(
        base_url=BASE,
        is_installed=True,
        is_running=True,
        executable="ollama",
        version="0.33.2",
    )


def make_app(client: Any, workspace: Path, model: OllamaModel) -> BeastApp:
    return BeastApp(
        workspace=Workspace(workspace),
        inventory=ModelInventory(selectable=(model,)),
        status=status(),
        client=client,
        initial_model=model,
    )


def text_of(widget: Any) -> str:
    rendered = widget.render()
    return rendered.plain if hasattr(rendered, "plain") else str(rendered)


async def wait_until(
    pilot: Any,
    probe: Callable[[], bool],
    description: str,
    timeout: float = 5.0,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if probe():
            return
        await pilot.pause(0.05)
    raise AssertionError(f"timed out waiting for {description}")


async def run_turn(pilot: Any, screen: ChatScreen, prompt_text: str) -> None:
    prompt = screen.query_one(PromptArea)
    prompt.load_text(prompt_text)
    await pilot.press("enter")
    await pilot.pause(1.0)


def detected_project(screen: ChatScreen) -> bool:
    splashes = list(screen.query(Splash))
    return bool(splashes and "Flask" in text_of(splashes[0]))


# ------------------------------------------------------------------- splash


async def test_splash_reports_the_detected_project_indicators(
    flask_project: Path,
) -> None:
    app = make_app(ChatClient(), flask_project, CHAT_MODEL)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ChatScreen)
        await wait_until(pilot, lambda: detected_project(screen), "project detection")

        splash = screen.query_one(Splash)
        rows = {
            label.strip(): value for label, value, _ in splash._project_rows()
        }
        assert rows["Language"] == "Python"
        assert rows["Framework"] == "Flask"
        assert rows["Pkg manager"] == "pip"
        assert rows["Source files"] == "2"
        assert rows["Test files"] == "1"

        rendered = text_of(splash)
        assert "Language" in rendered
        assert "Workspace" in rendered
        assert "scanning" not in rendered


async def test_clear_keeps_the_detected_project_on_the_new_splash(
    flask_project: Path,
) -> None:
    app = make_app(ChatClient(), flask_project, CHAT_MODEL)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        await wait_until(pilot, lambda: detected_project(screen), "project detection")

        await screen.action_clear()
        await pilot.pause()

        splash = screen.query_one(Splash)
        rendered = text_of(splash)
        assert "Flask" in rendered
        assert "scanning" not in rendered


# ------------------------------------------------------------ analysis line


async def test_analysis_line_reports_the_attached_context(
    flask_project: Path,
) -> None:
    client = ToolClient()
    app = make_app(client, flask_project, TOOL_MODEL)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        await run_turn(pilot, screen, AUTH_QUESTION)

        def shown() -> bool:
            activities = list(screen.query(AnalysisActivity))
            return bool(activities and "Project context" in text_of(activities[0]))

        await wait_until(pilot, shown, "the analysis line")

        activity = screen.query_one(AnalysisActivity)
        rendered = text_of(activity)
        assert rendered.startswith("[ANALYSIS] ")
        assert "Project context" in rendered

        request = client.calls[0]
        system = request["messages"][0]
        assert system["role"] == "system"
        assert "## Project context" in system["content"]
        assert "### `auth.py`" in system["content"]
        # The context belongs to the request, not to the stored conversation.
        assert all(
            "## Project context" not in message["content"]
            for message in request["messages"][1:]
        )
        assert screen._agent.conversation.messages[0].content == TOOL_SYSTEM_PROMPT


async def test_chat_only_models_do_not_analyse_the_project(
    flask_project: Path,
) -> None:
    client = ChatClient()
    app = make_app(client, flask_project, CHAT_MODEL)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert screen._agent.context_provider is None

        await run_turn(pilot, screen, "hello")

        assert list(screen.query(AnalysisActivity)) == []
        assert client.calls[0][0]["content"] == SYSTEM_PROMPT


# --------------------------------------------------------------- tool wiring


async def test_tool_capable_models_are_offered_project_info(
    flask_project: Path,
) -> None:
    app = make_app(ToolClient(), flask_project, TOOL_MODEL)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen

        tools = screen._agent.tools
        assert tools is not None
        assert tools.names[-1] == "project_info"
        assert len(tools.names) == 7
        names = {
            definition["function"]["name"] for definition in tools.definitions
        }
        assert names == set(tools.names)

        assert screen._agent.context_provider is app.intelligence
