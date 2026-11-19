"""CLI registration and argument plumbing.

These tests never start the TUI: ``beast.cli.run`` is replaced with a recorder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import beast.cli as cli
from beast.cli import app

runner = CliRunner()


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> int:
        calls.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "run", fake_run)
    return calls


def test_version_flag_reports_the_package_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "BEAST 0.1.0" in result.stdout


def test_root_help_lists_the_launch_command() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "launch" in result.stdout


def test_launch_group_lists_beast() -> None:
    result = runner.invoke(app, ["launch", "--help"])

    assert result.exit_code == 0
    assert "beast" in result.stdout


def test_beast_help_documents_every_option() -> None:
    result = runner.invoke(app, ["launch", "beast", "--help"])

    assert result.exit_code == 0
    for flag in (
        "--model",
        "--workspace",
        "--ollama-url",
        "--list-models",
        "--debug",
        "--auto",
    ):
        assert flag in result.stdout


def test_launch_beast_runs_from_the_current_directory(recorded: dict[str, Any]) -> None:
    result = runner.invoke(app, ["launch", "beast"])

    assert result.exit_code == 0
    assert recorded == {
        "workspace_path": None,
        "model_name": None,
        "base_url": None,
        "debug": False,
        "list_models": False,
    }


def test_options_are_forwarded(
    tmp_path: Path, recorded: dict[str, Any]
) -> None:
    result = runner.invoke(
        app,
        [
            "launch",
            "beast",
            "--model",
            "qwen3:8b",
            "--workspace",
            str(tmp_path),
            "--ollama-url",
            "http://127.0.0.1:9999",
            "--list-models",
            "--debug",
        ],
    )

    assert result.exit_code == 0
    assert recorded["model_name"] == "qwen3:8b"
    assert recorded["workspace_path"] == tmp_path.resolve()
    assert recorded["base_url"] == "http://127.0.0.1:9999"
    assert recorded["list_models"] is True
    assert recorded["debug"] is True


def test_auto_request_is_forwarded(recorded: dict[str, Any]) -> None:
    result = runner.invoke(
        app, ["launch", "beast", "--auto", "add a dark mode toggle"]
    )

    assert result.exit_code == 0
    assert recorded["auto"] == "add a dark mode toggle"


def test_short_flags_work(tmp_path: Path, recorded: dict[str, Any]) -> None:
    result = runner.invoke(
        app, ["launch", "beast", "-m", "llama3.2:3b", "-w", str(tmp_path)]
    )

    assert result.exit_code == 0
    assert recorded["model_name"] == "llama3.2:3b"


def test_a_missing_workspace_is_a_usage_error(
    tmp_path: Path, recorded: dict[str, Any]
) -> None:
    result = runner.invoke(
        app, ["launch", "beast", "--workspace", str(tmp_path / "nope")]
    )

    assert result.exit_code == 2
    assert recorded == {}


def test_the_launch_exit_code_is_propagated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "run", lambda **kwargs: 6)

    result = runner.invoke(app, ["launch", "beast"])

    assert result.exit_code == 6
