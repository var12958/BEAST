"""Launch orchestration: discovery, model resolution, error rendering."""

from __future__ import annotations

import io
import types
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

import beast.main as main_module
from beast.errors import (
    BeastError,
    ModelNotFoundError,
    NoModelsError,
    UnsupportedModelError,
)
from beast.main import (
    Discovery,
    _require_interactive_terminal,
    _resolve_model,
    discover,
    launch,
    print_models,
    render_error,
    run,
)
from beast.ollama.connection import OllamaStatus
from beast.ollama.models import ModelInventory, OllamaModel, OllamaModelList, split_models
from beast.workspace import Workspace

BASE = "http://ollama.test"


def running_status() -> OllamaStatus:
    return OllamaStatus(
        base_url=BASE,
        is_installed=True,
        is_running=True,
        executable="ollama",
        version="0.33.2",
    )


def sample_inventory() -> ModelInventory:
    return split_models(
        [
            OllamaModel(
                name="qwen3:8b",
                size=5_232_000_000,
                details={"family": "qwen3", "quantization_level": "Q4_K_M"},
                capabilities=["completion"],
            ),
            OllamaModel(
                name="remote:cloud",
                remote_host="https://ollama.com",
                capabilities=["completion"],
            ),
            OllamaModel(
                name="embed:latest",
                size=274_000_000,
                capabilities=["embedding"],
            ),
        ]
    )


def empty_console() -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    return Console(file=buffer, width=120, color_system=None), buffer


# ------------------------------------------------------------------- discovery


async def test_discover_returns_a_split_inventory(
    monkeypatch: pytest.MonkeyPatch, local_model: OllamaModel
) -> None:
    async def fake_probe(base_url: str | None = None) -> OllamaStatus:
        return running_status()

    class FakeClient:
        def __init__(self, base_url: str) -> None:
            self.base_url = base_url

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *exc_info: object) -> None:
            return None

        async def list_models(self) -> list[OllamaModel]:
            return [
                local_model,
                OllamaModel(name="remote:cloud", remote_host="https://ollama.com"),
            ]

    monkeypatch.setattr(main_module, "probe", fake_probe)
    monkeypatch.setattr(main_module, "OllamaClient", FakeClient)

    result = await discover()

    assert result.status.is_running is True
    assert [model.name for model in result.inventory.selectable] == ["qwen3:8b"]
    assert len(result.inventory.cloud) == 1


async def test_discover_skips_the_client_when_ollama_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_probe(base_url: str | None = None) -> OllamaStatus:
        return OllamaStatus(
            base_url=BASE,
            is_installed=False,
            is_running=False,
            failure="connect",
            error="nope",
        )

    class ExplodingClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("client must not be built when Ollama is down")

    monkeypatch.setattr(main_module, "probe", fake_probe)
    monkeypatch.setattr(main_module, "OllamaClient", ExplodingClient)

    result = await discover()

    assert result.status.is_running is False
    assert result.inventory.total == 0


# --------------------------------------------------------------- model choice


def test_resolve_model_accepts_an_installed_model() -> None:
    inventory = sample_inventory()

    assert _resolve_model(inventory, "qwen3:8b").name == "qwen3:8b"


def test_resolve_model_accepts_cloud_models() -> None:
    resolved = _resolve_model(sample_inventory(), "remote:cloud")

    assert resolved.name == "remote:cloud"
    assert resolved.is_cloud is True


def test_resolve_model_rejects_embedding_models() -> None:
    with pytest.raises(UnsupportedModelError) as excinfo:
        _resolve_model(sample_inventory(), "embed:latest")

    assert "embedding model" in excinfo.value.message


def test_resolve_model_lists_what_is_installed() -> None:
    with pytest.raises(ModelNotFoundError) as excinfo:
        _resolve_model(sample_inventory(), "ghost:70b")

    assert excinfo.value.model == "ghost:70b"
    assert "qwen3:8b" in (excinfo.value.hint or "")


# ------------------------------------------------------------ terminal checks


def test_interactive_terminal_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main_module.sys, "stdin", types.SimpleNamespace(isatty=lambda: False)
    )
    monkeypatch.setattr(
        main_module.sys, "stdout", types.SimpleNamespace(isatty=lambda: True)
    )

    with pytest.raises(BeastError):
        _require_interactive_terminal()


def test_interactive_terminal_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main_module.sys, "stdin", types.SimpleNamespace(isatty=lambda: True)
    )
    monkeypatch.setattr(
        main_module.sys, "stdout", types.SimpleNamespace(isatty=lambda: True)
    )

    _require_interactive_terminal()


# ------------------------------------------------------------------ table view


def test_print_models_lists_ready_and_hidden_models(
    tags_payload: dict, tmp_path: Path
) -> None:
    console, buffer = empty_console()
    discovery = Discovery(
        status=running_status(),
        inventory=split_models(OllamaModelList.model_validate(tags_payload).models),
    )

    print_models(console, discovery, Workspace(tmp_path))

    output = buffer.getvalue()
    assert "qwen3:8b" in output
    assert "5.2 GB" in output
    assert "Q4_K_M" in output
    assert "local · ready" in output
    assert "cloud · ready" in output
    assert "embedding — hidden" in output
    assert "1 local + 3 cloud chat models ready for BEAST." in output
    assert "Hidden: 1 embedding-only model." in output
    assert "TOOLS" in output
    assert "yes" in output


def test_print_models_explains_an_empty_store(tmp_path: Path) -> None:
    console, buffer = empty_console()
    discovery = Discovery(status=running_status(), inventory=ModelInventory())

    print_models(console, discovery, Workspace(tmp_path))

    assert "No models are installed yet" in buffer.getvalue()
    assert "ollama pull" in buffer.getvalue()


def test_print_models_includes_workspace_and_ollama_version(tmp_path: Path) -> None:
    console, buffer = empty_console()
    discovery = Discovery(status=running_status(), inventory=sample_inventory())

    print_models(console, discovery, Workspace(tmp_path))

    assert "v0.33.2" in buffer.getvalue()
    assert "Workspace" in buffer.getvalue()


# ---------------------------------------------------------------------- launch


@pytest.fixture
def fake_app(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the Textual app with a recorder and stub the TTY check."""
    record: dict[str, Any] = {}

    class FakeApp:
        def __init__(self, **kwargs: Any) -> None:
            record["init"] = kwargs

        def run(self) -> None:
            record["ran"] = True

    monkeypatch.setattr(main_module, "BeastApp", FakeApp)
    monkeypatch.setattr(main_module, "_require_interactive_terminal", lambda: None)
    return record


def install_discovery(
    monkeypatch: pytest.MonkeyPatch, inventory: ModelInventory
) -> None:
    async def fake_discover(base_url: str | None = None) -> Discovery:
        return Discovery(status=running_status(), inventory=inventory)

    monkeypatch.setattr(main_module, "discover", fake_discover)


def test_launch_runs_the_app_with_the_discovered_models(
    monkeypatch: pytest.MonkeyPatch, fake_app: dict[str, Any], tmp_path: Path
) -> None:
    install_discovery(monkeypatch, sample_inventory())

    code = launch(workspace_path=tmp_path)

    assert code == 0
    assert fake_app["ran"] is True
    assert fake_app["init"]["workspace"].root == tmp_path.resolve()
    assert fake_app["init"]["initial_model"] is None
    assert [model.name for model in fake_app["init"]["inventory"].selectable] == [
        "qwen3:8b"
    ]


def test_launch_can_skip_the_picker(
    monkeypatch: pytest.MonkeyPatch, fake_app: dict[str, Any], tmp_path: Path
) -> None:
    install_discovery(monkeypatch, sample_inventory())

    launch(workspace_path=tmp_path, model_name="qwen3:8b")

    assert fake_app["init"]["initial_model"].name == "qwen3:8b"


def test_launch_lists_models_without_starting_the_ui(
    monkeypatch: pytest.MonkeyPatch, fake_app: dict[str, Any], tmp_path: Path
) -> None:
    install_discovery(monkeypatch, sample_inventory())

    code = launch(workspace_path=tmp_path, list_models=True)

    assert code == 0
    assert "ran" not in fake_app


def test_launch_asks_for_a_model_when_none_are_installed(
    monkeypatch: pytest.MonkeyPatch, fake_app: dict[str, Any], tmp_path: Path
) -> None:
    install_discovery(monkeypatch, ModelInventory())

    with pytest.raises(NoModelsError) as excinfo:
        launch(workspace_path=tmp_path)

    assert "ollama pull" in (excinfo.value.hint or "")
    assert "ran" not in fake_app


def test_launch_rejects_an_unknown_model(
    monkeypatch: pytest.MonkeyPatch, fake_app: dict[str, Any], tmp_path: Path
) -> None:
    install_discovery(monkeypatch, sample_inventory())

    with pytest.raises(ModelNotFoundError):
        launch(workspace_path=tmp_path, model_name="not-installed:1b")


# ------------------------------------------------------------------- entrypoint


def test_run_returns_the_launch_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "launch", lambda **kwargs: 0)

    assert run() == 0


def test_run_passes_debug_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setenv("BEAST_DEBUG", "1")
    monkeypatch.setattr(
        main_module, "launch", lambda **kwargs: captured.update(kwargs) or 0
    )

    run()

    assert captured["debug"] is True


def test_run_renders_errors_and_returns_their_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def failing_launch(**kwargs: Any) -> int:
        raise NoModelsError("No chat-capable local models are installed.")

    monkeypatch.setattr(main_module, "launch", failing_launch)

    code = run()

    assert code == 6
    assert "No chat-capable local models are installed." in capsys.readouterr().out


def test_run_explains_a_malformed_ollama_url(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = run(base_url="http://127.0.0.1:abc", list_models=True)

    assert code == 9
    output = capsys.readouterr().out
    assert "Invalid Ollama endpoint" in output
    assert "not a valid URL" in output
    assert "Traceback" not in output


def test_run_treats_ctrl_c_as_an_interrupt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupted(**kwargs: Any) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(main_module, "launch", interrupted)

    assert run() == 130
    assert "Interrupted." in capsys.readouterr().out


def test_run_hides_unexpected_tracebacks(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def crashed(**kwargs: Any) -> int:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(main_module, "launch", crashed)

    assert run() == 1
    output = capsys.readouterr().out
    assert "Unexpected error" in output
    assert "--debug" in output
    assert "Traceback" not in output


def test_run_reraises_unexpected_errors_in_debug_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def crashed(**kwargs: Any) -> int:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(main_module, "launch", crashed)

    with pytest.raises(RuntimeError):
        run(debug=True)


def test_render_error_points_at_the_log_file() -> None:
    main_module.setup_logging()
    console, buffer = empty_console()

    render_error(console, BeastError("Something broke.", hint="Try this."))

    output = buffer.getvalue()
    assert "Something broke." in output
    assert "Try this." in output
    assert "Logs:" in output
