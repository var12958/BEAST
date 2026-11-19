"""Ollama Cloud models: discovery, picker, execution and failure handling.

Everything that would contact Ollama is mocked (``httpx.MockTransport`` or a
fake client).  Model-list parsing uses the real captured ``/api/tags`` payload
in ``tests/data/tags_real.json``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from textual.widgets import OptionList

from beast.errors import (
    CloudAuthError,
    CloudUnavailableError,
    OllamaProtocolError,
    ToolCallingUnsupportedError,
)
from beast.ollama.client import OllamaClient, is_cloud_model
from beast.ollama.connection import OllamaStatus
from beast.ollama.models import (
    ChatChunk,
    ModelInventory,
    OllamaModel,
    OllamaModelList,
    split_models,
)
from beast.ui.app import BeastApp
from beast.ui.screens import ChatScreen, ModelSelectScreen
from beast.workspace import Workspace

BASE = "http://ollama.test"
LOCAL = "qwen3:1.7b"
CLOUD = "kimi-k2.7-code:cloud"

Handler = Callable[[httpx.Request], httpx.Response]

LOCAL_MODEL = OllamaModel(
    name=LOCAL,
    size=1_359_293_444,
    details={"family": "qwen3", "quantization_level": "Q4_K_M"},
    capabilities=["completion", "tools"],
)
CLOUD_MODEL = OllamaModel(
    name=CLOUD,
    remote_host="https://ollama.com",
    remote_model="kimi-k2.7-code",
    capabilities=["completion", "tools", "vision"],
)
CLOUD_PLAIN = OllamaModel(
    name="plain:cloud",
    remote_host="https://ollama.com",
    capabilities=["completion"],
)


def inventory_of(*models: OllamaModel) -> ModelInventory:
    return split_models(models)


def status() -> OllamaStatus:
    return OllamaStatus(
        base_url=BASE,
        is_installed=True,
        is_running=True,
        executable="ollama",
        version="0.33.2",
    )


class FakeClient:
    """Records chat requests; replies with one chunk."""

    def __init__(self, models: Sequence[OllamaModel] = ()) -> None:
        self.models = list(models)
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []
        self.reloads = 0

    async def stream_chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        options: Mapping[str, Any] | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ):
        self.calls.append((model, [dict(message) for message in messages]))
        yield ChatChunk(
            model=model, message={"role": "assistant", "content": "ok"}, done=True
        )

    async def list_models(self) -> list[OllamaModel]:
        self.reloads += 1
        return list(self.models)

    async def close(self) -> None:
        return None


def make_app(
    models: Sequence[OllamaModel],
    *,
    client: FakeClient | None = None,
    initial: OllamaModel | None = None,
) -> BeastApp:
    return BeastApp(
        workspace=Workspace(Path.cwd()),
        inventory=inventory_of(*models),
        status=status(),
        client=client or FakeClient(),  # type: ignore[arg-type]
        initial_model=initial,
    )


def make_client(handler: Handler) -> OllamaClient:
    return OllamaClient(base_url=BASE, transport=httpx.MockTransport(handler))


def ndjson(*payloads: Mapping[str, Any]) -> bytes:
    return b"".join(
        json.dumps(payload).encode("utf-8") + b"\n" for payload in payloads
    )


def chunk(content: str = "", *, done: bool = False) -> dict[str, Any]:
    return {
        "model": CLOUD,
        "message": {"role": "assistant", "content": content},
        "done": done,
    }


def option_index(options: OptionList, identifier: str) -> int:
    return next(
        index
        for index in range(options.option_count)
        if options.get_option_at_index(index).id == identifier
    )


# ------------------------------------------------------------------ discovery


def test_local_models_are_discovered(tags_payload: dict) -> None:
    inventory = split_models(OllamaModelList.model_validate(tags_payload).models)

    assert [model.name for model in inventory.selectable] == ["qwen3:8b"]
    assert all(model.is_cloud is False for model in inventory.selectable)


def test_cloud_models_are_discovered(tags_payload: dict) -> None:
    inventory = split_models(OllamaModelList.model_validate(tags_payload).models)

    assert [model.name for model in inventory.cloud] == [
        "deepseek-v4-pro:cloud",
        "deepseek-v4.1-flash:cloud",
        "kimi-k2.7-code:cloud",
    ]
    assert all(model.is_cloud for model in inventory.cloud)


def test_mixed_store_keeps_local_and_cloud_models(tags_payload: dict) -> None:
    inventory = split_models(OllamaModelList.model_validate(tags_payload).models)

    assert len(inventory.selectable) == 1
    assert len(inventory.cloud) == 3
    assert len(inventory.usable) == 4
    assert len(inventory.non_chat) == 1
    assert inventory.hidden == 1


def test_classification_uses_metadata_not_a_hardcoded_list() -> None:
    assert OllamaModel(name="anything:cloud").is_cloud is True
    assert OllamaModel(name="x", remote_model="x").is_cloud is True
    assert OllamaModel(name="x", remote_host="https://ollama.com").is_cloud is True
    assert OllamaModel(name="cloudy:latest").is_cloud is False
    assert OllamaModel(name=LOCAL, size=1).location_display == "local"
    assert OllamaModel(name=CLOUD).location_display == "cloud"
    assert is_cloud_model(CLOUD) is True
    assert is_cloud_model("qwen3:8b") is False


# --------------------------------------------------------------------- picker


async def test_picker_shows_both_categories() -> None:
    app = make_app([LOCAL_MODEL, CLOUD_MODEL])

    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ModelSelectScreen)
        options = app.screen.query_one("#models", OptionList)
        prompts = [
            str(options.get_option_at_index(index).prompt).upper()
            for index in range(options.option_count)
        ]

    assert any("LOCAL MODELS" in prompt for prompt in prompts)
    assert any("CLOUD MODELS" in prompt for prompt in prompts)
    assert any(LOCAL.upper() in prompt for prompt in prompts)
    assert any(CLOUD.upper() in prompt for prompt in prompts)


async def test_selecting_a_cloud_model_opens_the_chat_screen() -> None:
    client = FakeClient()
    app = make_app([LOCAL_MODEL, CLOUD_MODEL], client=client)

    async with app.run_test() as pilot:
        await pilot.pause()
        options = app.screen.query_one("#models", OptionList)
        options.highlighted = option_index(options, CLOUD)
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, ChatScreen)
        assert app.screen.current_model == CLOUD


async def test_switching_between_local_and_cloud_models() -> None:
    client = FakeClient()
    app = make_app([LOCAL_MODEL, CLOUD_MODEL], client=client, initial=LOCAL_MODEL)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.screen.current_model == LOCAL

        await pilot.press("ctrl+n")
        await pilot.pause()
        options = app.screen.query_one("#models", OptionList)
        options.highlighted = option_index(options, CLOUD)
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, ChatScreen)
        assert app.screen.current_model == CLOUD

        await pilot.press("ctrl+n")
        await pilot.pause()
        options = app.screen.query_one("#models", OptionList)
        options.highlighted = option_index(options, LOCAL)
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, ChatScreen)
        assert app.screen.current_model == LOCAL


async def test_cloud_model_without_tools_gets_no_tool_registry() -> None:
    app = make_app([CLOUD_MODEL, CLOUD_PLAIN], initial=CLOUD_PLAIN)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ChatScreen)
        assert app.screen._agent is not None
        assert app.screen._agent.tools is None


async def test_cloud_model_with_tools_gets_the_full_tool_registry() -> None:
    app = make_app([CLOUD_MODEL], initial=CLOUD_MODEL)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ChatScreen)
        assert app.screen._agent is not None
        assert app.screen._agent.tools is not None
        # The six core tools stay available; project_info is added because
        # project intelligence is active in the app.
        assert {
            "list_files",
            "read_file",
            "write_file",
            "edit_file",
            "search_files",
            "run_command",
        } <= set(app.screen._agent.tools.names)


# ------------------------------------------------------------------ sign-in


async def test_cloud_signin_preflight_accepts_a_valid_account() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/me"
        # Account details may be present; BEAST must not need or keep them.
        return httpx.Response(200, json={"email": "someone@example.com"})

    async with make_client(handler) as client:
        assert await client.check_cloud_auth() is True


async def test_cloud_signin_preflight_reports_a_missing_signin() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    async with make_client(handler) as client:
        with pytest.raises(CloudAuthError) as excinfo:
            await client.check_cloud_auth()

    assert "not signed in" in excinfo.value.message
    assert "ollama signin" in (excinfo.value.hint or "")


async def test_cloud_signin_preflight_tolerates_older_ollama() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    async with make_client(handler) as client:
        assert await client.check_cloud_auth() is None


async def test_cloud_signin_preflight_tolerates_a_transport_problem() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    async with make_client(handler) as client:
        assert await client.check_cloud_auth() is None


async def test_missing_cloud_signin_is_reported_in_the_tui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RejectingClient(FakeClient):
        async def check_cloud_auth(self) -> bool:
            raise CloudAuthError(
                "Ollama Cloud is not signed in on this machine.",
                hint="Run `ollama signin` in another terminal.",
            )

    notices: list[str] = []
    monkeypatch.setattr(
        ChatScreen,
        "notify",
        lambda self, message, **kwargs: notices.append(str(message)),
    )
    app = make_app(
        [CLOUD_MODEL], client=RejectingClient(), initial=CLOUD_MODEL
    )

    async with app.run_test() as pilot:
        await pilot.pause(0.5)

    assert any("ollama signin" in notice for notice in notices)


# ------------------------------------------------------------------- failures


async def test_cloud_auth_failure_is_explained() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    async with make_client(handler) as client:
        with pytest.raises(CloudAuthError) as excinfo:
            async for _ in client.stream_chat(CLOUD, []):
                pass

    error = excinfo.value
    assert "not signed in" in error.message
    assert "ollama signin" in (error.hint or "")
    assert "API key" in (error.hint or "")


async def test_cloud_connectivity_failure_is_explained() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={
                "error": (
                    'Post "https://ollama.com/api/chat": dial tcp: lookup '
                    "ollama.com: no such host"
                )
            },
        )

    async with make_client(handler) as client:
        with pytest.raises(CloudUnavailableError) as excinfo:
            async for _ in client.stream_chat(CLOUD, []):
                pass

    error = excinfo.value
    assert "could not reach Ollama Cloud" in error.message
    assert "internet connection" in (error.hint or "")


async def test_cloud_stream_error_object_maps_to_auth_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=ndjson({"error": "please sign in to use cloud models"}),
        )

    async with make_client(handler) as client:
        with pytest.raises(CloudAuthError):
            async for _ in client.stream_chat(CLOUD, []):
                pass


async def test_cloud_model_rejecting_tools_is_explained() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": f"registry.ollama.ai/library/{CLOUD} does not support tools"
            },
        )

    async with make_client(handler) as client:
        with pytest.raises(ToolCallingUnsupportedError) as excinfo:
            async for _ in client.stream_chat(
                CLOUD, [], tools=[{"type": "function"}]
            ):
                pass

    assert "tool calling" in (excinfo.value.hint or "")


# -------------------------------------------------- local regression coverage


async def test_local_model_requests_are_unchanged() -> None:
    sink: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        sink["body"] = json.loads(request.content)
        return httpx.Response(200, content=ndjson(chunk("hi", done=True)))

    tools = [{"type": "function", "function": {"name": "list_files"}}]

    async with make_client(handler) as client:
        replies = [
            incoming.content
            async for incoming in client.stream_chat(
                LOCAL, [{"role": "user", "content": "hi"}], tools=tools
            )
        ]

    assert replies == ["hi"]
    assert sink["body"]["model"] == LOCAL
    assert sink["body"]["stream"] is True
    assert sink["body"]["tools"] == tools


async def test_local_failures_keep_their_existing_messages() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    async with make_client(handler) as client:
        with pytest.raises(OllamaProtocolError) as excinfo:
            async for _ in client.stream_chat(LOCAL, []):
                pass

    # Not reclassified as a cloud error.
    assert not isinstance(excinfo.value, CloudAuthError)
    assert "could not complete the chat request" in excinfo.value.message
