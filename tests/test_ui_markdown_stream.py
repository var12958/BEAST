"""Mouse-selection regressions for streamed Markdown.

Textual's compositor can briefly report a Markdown block that
``Markdown.update`` has just detached.  If a mouse event lands in that window,
Textual's selection code dereferences ``widget.parent.region`` and crashes
with ``AttributeError: 'NoneType' object has no attribute 'region'``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from textual import events
from textual.app import App
from textual.containers import VerticalScroll
from textual.widgets import Markdown

from beast.ollama.connection import OllamaStatus
from beast.ollama.models import ChatChunk, ModelInventory, OllamaModel
from beast.ui.app import BeastApp
from beast.ui.screens import BeastScreen
from beast.ui.widgets import AssistantMessage
from beast.workspace import Workspace

BASE = "http://ollama.test"

LONG_MARKDOWN = (
    "\n\n".join(
        f"## Section {index}\n\n"
        + " ".join(f"word{index}-{word}" for word in range(30))
        for index in range(12)
    )
    + "\n\n```python\nprint('hello from BEAST')\n```\n\nFinal paragraph."
)

RACE_MARKDOWN = "\n\n".join(
    f"replacement paragraph {index}" for index in range(201)
)


class FakeChatClient:
    """Stream a fixed list of chunks without touching the network."""

    def __init__(self, parts: Sequence[str], *, delay: float = 0.0) -> None:
        self.parts = parts
        self.delay = delay

    async def stream_chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        options: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[ChatChunk]:
        for index, part in enumerate(self.parts):
            if index and self.delay:
                await asyncio.sleep(self.delay)
            yield ChatChunk(model=model, message={"role": "assistant", "content": part})
        yield ChatChunk(
            model=model, message={"role": "assistant", "content": ""}, done=True
        )

    async def list_models(self) -> list[OllamaModel]:
        return []

    async def close(self) -> None:
        return None


def chat_model() -> OllamaModel:
    return OllamaModel(
        name="qwen3:8b",
        size=5_000_000_000,
        details={"family": "qwen3", "quantization_level": "Q4_K_M"},
        capabilities=["completion"],
    )


def beast_app(client: FakeChatClient) -> BeastApp:
    model = chat_model()
    return BeastApp(
        workspace=Workspace(Path.cwd()),
        inventory=ModelInventory(selectable=(model,)),
        status=OllamaStatus(
            base_url=BASE,
            is_installed=True,
            is_running=True,
            executable="ollama",
            version="0.33.2",
        ),
        client=client,  # type: ignore[arg-type]
        initial_model=model,
    )


def chunks(text: str, count: int) -> list[str]:
    """Split text into roughly ``count`` fragments."""
    size = max(1, len(text) // count)
    return [text[index : index + size] for index in range(0, len(text), size)]


def conversation(app: BeastApp) -> VerticalScroll:
    return app.screen.query_one("#conversation", VerticalScroll)


class MarkdownScreen(BeastScreen):
    """Full-screen Markdown, mirroring Textual's upstream reproduction."""

    def compose(self):
        yield Markdown("# Before\n\nClick target paragraph.", id="doc")


async def test_mouse_down_on_a_detached_markdown_block_is_ignored() -> None:
    app: App[None] = App()
    async with app.run_test(size=(100, 30)) as pilot:
        await app.push_screen(MarkdownScreen())
        await pilot.pause(0.1)
        screen = app.screen
        assert isinstance(screen, MarkdownScreen)
        document = screen.query_one("#doc", Markdown)
        old_paragraph = document.query("MarkdownParagraph").first()
        x = old_paragraph.region.x + 1
        y = old_paragraph.region.y

        async def replace() -> None:
            await document.update(RACE_MARKDOWN)

        update = asyncio.create_task(replace())
        saw_race = False
        try:
            for _ in range(1000):
                await asyncio.sleep(0)
                hit, _offset = screen.get_widget_and_offset_at(x, y)
                stale_hit, _stale_offset = (
                    screen._compositor.get_widget_and_offset_at(x, y)
                )
                if old_paragraph.parent is None and stale_hit is old_paragraph:
                    # The screen must treat the compositor's stale hit as a
                    # miss rather than handing it to Textual's selection code.
                    assert hit is None
                    screen._forward_event(
                        events.MouseDown(None, x, y, 0, 0, 1, False, False, False)
                    )
                    saw_race = True
                    break
            assert saw_race, "detached-widget race window was not observed"
        finally:
            await update

        await pilot.pause()
        hit, _offset = screen.get_widget_and_offset_at(x, y)
        assert hit is not None and hit.is_attached
        await pilot.mouse_down(offset=(x, y))
        assert screen._select_state is not None
        await pilot.mouse_up(offset=(x, y))


async def test_streaming_keeps_earlier_markdown_blocks_mounted() -> None:
    app = beast_app(FakeChatClient([]))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        scroll = conversation(app)
        await scroll.mount(AssistantMessage("test-model"))
        await pilot.pause()

        message = app.screen.query_one(AssistantMessage)
        await message.update_body("first paragraph")
        await pilot.pause()

        document = message.query_one(Markdown)
        first = document.query("MarkdownParagraph").first()
        body = "first paragraph"
        for index in range(1, 8):
            body += f"\n\nparagraph {index} " + ("more words " * 20)
            await message.update_body(body)

        await pilot.pause()
        assert first.is_attached
        assert first.parent is document
        assert document.source == body
        assert message._body == body


async def test_mouse_events_during_and_after_a_streamed_reply() -> None:
    expected = LONG_MARKDOWN
    client = FakeChatClient(chunks(expected, 30), delay=0.002)
    app = beast_app(client)

    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.pause()
        screen = app.screen
        prompt = screen.query_one("#prompt")
        prompt.load_text("write a long answer")
        await pilot.press("enter")

        for _ in range(200):
            await pilot.pause(0.01)
            messages = list(screen.query(AssistantMessage))
            if messages:
                await pilot.mouse_down(offset=(10, 8))
                await pilot.mouse_up(offset=(10, 8))
            if screen._state.name == "IDLE" and messages:
                break

        await pilot.pause(0.5)
        message = screen.query_one(AssistantMessage)
        assert message._body == expected

        markdown = message.query_one(Markdown)
        assert markdown.source == expected
        assert len(list(markdown.query("MarkdownFence"))) == 1
        markdown.scroll_visible(animate=False)
        await pilot.pause()
        await pilot.mouse_down(offset=(10, 8))
        await pilot.mouse_up(offset=(10, 8))

        await pilot.resize_terminal(60, 20)
        await pilot.pause()
        markdown.scroll_visible(animate=False)
        await pilot.pause()
        await pilot.mouse_down(offset=(10, 8))
        await pilot.mouse_up(offset=(10, 8))
        assert message._body == expected
