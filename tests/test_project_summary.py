"""Non-interactive project summary: rendering, launch short-circuit, CLI wiring."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console
from typer.testing import CliRunner

import beast.cli as cli
import beast.main as main_module
from beast.main import launch, print_project
from beast.workspace import Workspace

runner = CliRunner()


def render(project: Path) -> str:
    buffer = io.StringIO()
    console = Console(file=buffer, width=120, color_system=None)
    print_project(console, Workspace(project))
    return buffer.getvalue()


# ----------------------------------------------------------------- rendering


def test_project_summary_reports_detected_evidence(flask_project: Path) -> None:
    output = render(flask_project)

    assert "Project analysis" in output
    assert "Workspace" in output
    assert "Project:" in output
    assert "Primary language: Python" in output
    assert "Frameworks: Flask" in output
    assert "Package managers: pip" in output
    assert "Test frameworks: pytest" in output
    assert "Source files: 2" in output
    assert "Test files: 1" in output
    assert "Manifests:" in output
    assert "requirements.txt (requirements)" in output
    assert "Dependencies:" in output
    assert "Flask 3.0.0" in output
    assert "pytest 8.0.0" in output
    assert "Structure:" in output
    assert "app.py" in output
    assert "auth.py" in output


def test_project_summary_handles_a_typescript_project(react_project: Path) -> None:
    output = render(react_project)

    assert "Primary language: TypeScript" in output
    assert "Frameworks: React" in output
    assert "Package managers: npm" in output
    assert "Test frameworks: Vitest" in output
    assert "Source files: 5" in output
    assert "Test files: 1" in output
    assert "react" in output
    assert "vitest" in output


def test_project_summary_reports_every_ecosystem(multilang_project: Path) -> None:
    output = render(multilang_project)

    assert "Primary language: Python" in output
    for language in ("Python", "TypeScript", "Go", "C++", "Vue"):
        assert language in output
    assert "FastAPI" in output
    assert "pip" in output
    assert "npm" in output
    assert "go modules" in output
    assert "CMake" in output
    assert "fastapi" in output
    assert "gin" in output


def test_project_summary_never_reads_credential_files(flask_project: Path) -> None:
    output = render(flask_project)

    assert "not-for-the-model-123456" not in output
    assert ".env" not in output


# -------------------------------------------------------------------- launch


def test_launch_with_a_project_summary_does_not_contact_ollama(
    monkeypatch: pytest.MonkeyPatch,
    flask_project: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Ollama must not be contacted for a project summary")

    monkeypatch.setattr(main_module, "discover", explode)

    code = launch(workspace_path=flask_project, project_summary=True)

    assert code == 0
    output = capsys.readouterr().out
    assert "Project analysis" in output
    assert "Flask 3.0.0" in output


def test_run_forwards_the_project_summary_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_launch(**kwargs: Any) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(main_module, "launch", fake_launch)

    assert main_module.run(project_summary=True) == 0
    assert captured["project_summary"] is True


# ----------------------------------------------------------------------- CLI


def test_cli_help_documents_the_project_summary_flag() -> None:
    result = runner.invoke(cli.app, ["launch", "beast", "--help"])

    assert result.exit_code == 0
    assert "--project-summary" in result.stdout


def test_cli_forwards_the_project_summary_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(**options: Any) -> int:
        captured.update(options)
        return 0

    monkeypatch.setattr(cli, "run", fake_run)

    result = runner.invoke(cli.app, ["launch", "beast", "--project-summary"])

    assert result.exit_code == 0
    assert captured["project_summary"] is True
    assert captured["list_models"] is False


def test_cli_omits_the_flag_when_it_was_not_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(**options: Any) -> int:
        captured.update(options)
        return 0

    monkeypatch.setattr(cli, "run", fake_run)

    result = runner.invoke(cli.app, ["launch", "beast"])

    assert result.exit_code == 0
    assert "project_summary" not in captured
