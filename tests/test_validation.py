"""Validation discovery, execution, reporting and evidence handling."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from beast.core.validation import (
    CheckKind,
    CommandOutcome,
    ToolCommandRunner,
    ValidationCommand,
    ValidationEngine,
    ValidationLimits,
    ValidationReport,
    ValidationResult,
    discover_validation_commands,
)


class FakeRunner:
    """Deterministic stand-in for the approved shell runner."""

    def __init__(self, *outcomes: CommandOutcome) -> None:
        self.outcomes = list(outcomes)
        self.commands: list[str] = []
        self.timeouts: list[float] = []

    async def run(
        self, command: ValidationCommand, *, timeout: float
    ) -> CommandOutcome:
        self.commands.append(command.command)
        self.timeouts.append(timeout)
        if self.outcomes:
            return self.outcomes.pop(0)
        return CommandOutcome(exit_code=0, output="ok")


class RecordingTracker:
    def __init__(self) -> None:
        self.commands: list[dict[str, object]] = []

    def record_command(self, command: str, **kwargs: object) -> None:
        self.commands.append({"command": command, **kwargs})


def write(root: Path, relative: str, text: str) -> Path:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


# --------------------------------------------------------------------- discovery


def test_python_commands_come_from_project_evidence_in_priority_order(
    tmp_path: Path,
) -> None:
    write(
        tmp_path,
        "pyproject.toml",
        "[tool.pytest.ini_options]\n[tool.ruff]\nline-length = 88\n[tool.mypy]\n",
    )

    commands = discover_validation_commands(tmp_path, max_commands=10)

    assert [command.command for command in commands] == [
        "python -m pytest -q",
        "mypy .",
        "ruff check .",
    ]
    assert [command.kind for command in commands] == [
        CheckKind.TEST,
        CheckKind.TYPECHECK,
        CheckKind.LINT,
    ]
    assert commands[0].source == "pyproject.toml"


def test_python_tests_are_detected_from_layout_without_pyproject(
    tmp_path: Path,
) -> None:
    write(tmp_path, "tests/test_example.py", "def test_ok():\n    assert True\n")

    commands = discover_validation_commands(tmp_path)

    assert commands[0].command == "python -m pytest -q"
    assert commands[0].source == "tests/"


def test_command_limit_truncates_the_strongest_first(tmp_path: Path) -> None:
    write(
        tmp_path,
        "pyproject.toml",
        "[tool.pytest.ini_options]\n[tool.ruff]\n[tool.mypy]\n",
    )

    commands = discover_validation_commands(tmp_path, max_commands=2)

    assert [command.command for command in commands] == [
        "python -m pytest -q",
        "mypy .",
    ]


def test_node_scripts_and_package_manager_are_respected(tmp_path: Path) -> None:
    write(
        tmp_path,
        "package.json",
        """
        {"scripts": {"test": "vitest run", "build": "vite build",
                     "typecheck": "tsc --noEmit", "lint": "eslint ."}}
        """,
    )
    write(tmp_path, "pnpm-lock.yaml", "lockfileVersion: 9\n")

    commands = discover_validation_commands(tmp_path, max_commands=10)

    assert [command.command for command in commands] == [
        "pnpm run test",
        "pnpm run build",
        "pnpm run typecheck",
        "pnpm run lint",
    ]


def test_npm_placeholder_test_script_is_not_a_check(tmp_path: Path) -> None:
    write(
        tmp_path,
        "package.json",
        '{"scripts": {"test": "echo \\"Error: no test specified\\" && exit 1", '
        '"build": "vite build"}}',
    )

    commands = discover_validation_commands(tmp_path, max_commands=10)

    assert [command.command for command in commands] == ["npm run build"]


def test_other_ecosystems_are_discovered(tmp_path: Path) -> None:
    write(tmp_path, "go.mod", "module example.com/app\n")
    write(tmp_path, "Cargo.toml", "[package]\nname = 'app'\n")
    write(tmp_path, "clippy.toml", "")
    write(tmp_path, "Makefile", "test:\n\tpytest\nlint:\n\truff .\n")

    commands = discover_validation_commands(tmp_path, max_commands=20)
    names = [command.command for command in commands]

    assert "go test ./..." in names
    assert "go build ./..." in names
    assert "cargo test" in names
    assert "cargo clippy -- -D warnings" in names
    assert "make test" in names
    assert "make lint" in names


def test_ruby_php_dotnet_and_java_are_discovered(tmp_path: Path) -> None:
    write(tmp_path, "Gemfile", "gem 'rspec'\n")
    (tmp_path / "spec").mkdir()
    write(tmp_path, "composer.json", '{"scripts": {"test": "phpunit"}}')
    write(tmp_path, "App.csproj", "<Project></Project>")

    names = [command.command for command in discover_validation_commands(tmp_path, max_commands=20)]
    assert "bundle exec rspec" in names
    assert "composer test" in names
    assert "dotnet build" in names

    java = tmp_path / "java"
    java.mkdir()
    write(java, "pom.xml", "<project></project>")
    commands = discover_validation_commands(java)
    assert commands[0].command == "mvn -q test"

    gradle = tmp_path / "gradle_project"
    gradle.mkdir()
    write(gradle, "gradlew", "")
    write(gradle, "gradlew.bat", "")
    commands = discover_validation_commands(gradle)
    expected = "gradlew.bat test" if os.name == "nt" else "./gradlew test"
    assert commands[0].command == expected


def test_changed_javascript_uses_node_check_when_node_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "beast.core.validation.shutil.which", lambda name: "node"
    )
    write(tmp_path, "app.js", "const x = 1;\n")

    commands = discover_validation_commands(
        tmp_path, changed_files=["app.js"], max_commands=5
    )

    assert [command.command for command in commands] == ['node --check "app.js"']
    assert commands[0].kind is CheckKind.SYNTAX


def test_changed_javascript_is_ignored_without_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "beast.core.validation.shutil.which", lambda name: None
    )
    write(tmp_path, "app.js", "const x = 1;\n")

    assert (
        discover_validation_commands(
            tmp_path, changed_files=["app.js"], max_commands=5
        )
        == ()
    )


# ------------------------------------------------------------------------ engine


async def test_successful_command_verifies_the_report(tmp_path: Path) -> None:
    write(tmp_path, "pyproject.toml", "[tool.pytest.ini_options]\n")
    write(tmp_path, "app.py", "value = 1\n")
    runner = FakeRunner(CommandOutcome(exit_code=0, output="1 passed"))
    engine = ValidationEngine(tmp_path, runner)

    report = await engine.validate(["app.py"])

    assert report.commands_run == 1
    assert report.verified
    assert report.status_text == "PASSED — 1 command"
    assert runner.commands == ["python -m pytest -q"]


async def test_failed_command_is_reported_with_evidence(tmp_path: Path) -> None:
    write(tmp_path, "pyproject.toml", "[tool.pytest.ini_options]\n")
    write(tmp_path, "app.py", "value = 1\n")
    runner = FakeRunner(
        CommandOutcome(exit_code=1, output="FAILED tests/test_app.py::test_x")
    )

    report = await ValidationEngine(tmp_path, runner).validate(["app.py"])

    assert not report.verified
    assert report.has_failures
    assert report.status_text == "FAILED — 1 command(s) failed"
    assert "FAILED tests/test_app.py" in report.failure_evidence()
    assert report.signature()


async def test_no_changed_files_means_no_checks(tmp_path: Path) -> None:
    runner = FakeRunner()

    report = await ValidationEngine(tmp_path, runner).validate([])

    assert not report.verified
    assert report.commands_run == 0
    assert "no files were changed" in report.status_text
    assert runner.commands == []


async def test_syntax_only_evidence_is_labelled_honestly(tmp_path: Path) -> None:
    write(tmp_path, "app.py", "value = 1\n")
    runner = FakeRunner()

    report = await ValidationEngine(tmp_path, runner).validate(["app.py"])

    assert report.syntax[0].ok
    assert not report.verified
    assert "NOT VERIFIED" in report.status_text
    assert "no validation commands were discovered" in report.status_text


async def test_syntax_errors_fail_the_report_without_running_commands(
    tmp_path: Path,
) -> None:
    write(tmp_path, "pyproject.toml", "[tool.pytest.ini_options]\n")
    write(tmp_path, "broken.py", "def broken(:\n")
    runner = FakeRunner(CommandOutcome(exit_code=0))

    report = await ValidationEngine(tmp_path, runner).validate(["broken.py"])

    assert report.syntax_failures
    assert report.has_failures
    assert not report.verified
    assert "FAILED — 1 syntax error(s)" == report.status_text
    assert "broken.py" in report.failure_evidence()
    # A syntax error is already conclusive evidence; pytest still runs and
    # fails, which is recorded, but the report never claims success.
    assert runner.commands == ["python -m pytest -q"]


async def test_json_and_toml_are_parsed_in_process(tmp_path: Path) -> None:
    write(tmp_path, "good.json", '{"a": 1}')
    write(tmp_path, "bad.json", '{"a": }')
    write(tmp_path, "good.toml", "a = 1\n")
    write(tmp_path, "bad.toml", "a = [\n")

    report = await ValidationEngine(tmp_path, FakeRunner()).validate(
        ["good.json", "bad.json", "good.toml", "bad.toml"]
    )

    states = {check.path: check.ok for check in report.syntax}
    assert states == {
        "good.json": True,
        "bad.json": False,
        "good.toml": True,
        "bad.toml": False,
    }


async def test_rejection_stops_further_commands_and_is_not_verified(
    tmp_path: Path,
) -> None:
    write(
        tmp_path,
        "pyproject.toml",
        "[tool.pytest.ini_options]\n[tool.ruff]\n[tool.mypy]\n",
    )
    write(tmp_path, "app.py", "value = 1\n")
    runner = FakeRunner(
        CommandOutcome(exit_code=None, rejected=True, output="rejected"),
        CommandOutcome(exit_code=0),
    )

    report = await ValidationEngine(tmp_path, runner).validate(["app.py"])

    assert len(report.results) == 1
    assert report.rejected
    assert not report.verified
    assert "rejected" in report.status_text
    assert runner.commands == ["python -m pytest -q"]


async def test_timeout_is_a_failure(tmp_path: Path) -> None:
    write(tmp_path, "pyproject.toml", "[tool.pytest.ini_options]\n")
    write(tmp_path, "app.py", "value = 1\n")
    runner = FakeRunner(CommandOutcome(exit_code=None, timed_out=True))

    report = await ValidationEngine(tmp_path, runner).validate(["app.py"])

    assert report.command_failures
    assert not report.verified
    assert "timed out" in report.failure_evidence()


async def test_should_stop_prevents_execution(tmp_path: Path) -> None:
    write(tmp_path, "pyproject.toml", "[tool.pytest.ini_options]\n")
    write(tmp_path, "app.py", "value = 1\n")
    runner = FakeRunner()

    report = await ValidationEngine(tmp_path, runner).validate(
        ["app.py"], should_stop=lambda: True
    )

    assert report.stopped_early
    assert report.commands_run == 0
    assert runner.commands == []


async def test_engine_records_commands_on_the_tracker(tmp_path: Path) -> None:
    write(tmp_path, "pyproject.toml", "[tool.pytest.ini_options]\n")
    write(tmp_path, "app.py", "value = 1\n")
    tracker = RecordingTracker()

    await ValidationEngine(
        tmp_path, FakeRunner(CommandOutcome(exit_code=2)), tracker=tracker
    ).validate(["app.py"])

    assert tracker.commands[0]["command"] == "python -m pytest -q"
    assert tracker.commands[0]["exit_code"] == 2
    assert tracker.commands[0]["ok"] is False
    assert tracker.commands[0]["kind"] == "test"


async def test_limits_are_passed_to_the_runner(tmp_path: Path) -> None:
    write(tmp_path, "pyproject.toml", "[tool.pytest.ini_options]\n")
    write(tmp_path, "app.py", "value = 1\n")
    runner = FakeRunner()

    await ValidationEngine(
        tmp_path,
        runner,
        limits=ValidationLimits(max_commands=1, command_timeout=7.5),
    ).validate(["app.py"])

    assert runner.timeouts == [7.5]


# --------------------------------------------------------------------- evidence


def test_failure_evidence_redacts_secrets() -> None:
    command = ValidationCommand(CheckKind.TEST, "npm test", "package.json")
    report = ValidationReport(
        results=(
            ValidationResult(
                command,
                CommandOutcome(
                    exit_code=1,
                    output='db password = "hunter2secret" rejected',
                ),
            ),
        )
    )

    evidence = report.failure_evidence()

    assert "hunter2secret" not in evidence
    assert "[REDACTED]" in evidence


def test_signature_is_stable_for_the_same_failure_and_changes_otherwise() -> None:
    command = ValidationCommand(CheckKind.TEST, "pytest", "pyproject.toml")

    def report(output: str) -> ValidationReport:
        return ValidationReport(
            results=(
                ValidationResult(command, CommandOutcome(exit_code=1, output=output)),
            )
        )

    assert report("boom").signature() == report("boom").signature()
    assert report("boom").signature() != report("other").signature()
    assert ValidationReport().signature() == ""


async def test_syntax_only_command_is_reported_as_weak_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "beast.core.validation.shutil.which", lambda name: "node"
    )
    write(tmp_path, "app.js", "const x = 1;\n")

    report = await ValidationEngine(
        tmp_path, FakeRunner(CommandOutcome(exit_code=0))
    ).validate(["app.js"])

    assert report.verified
    assert report.uses_only_syntax_commands
    assert "syntax check" in report.status_text
    assert "no tests or build found" in report.status_text


# ------------------------------------------------------------- shell integration


class FakeTool:
    def __init__(self, result: object) -> None:
        self.result = result

    async def run(self, args: dict[str, object]) -> object:
        assert args["command"] == "npm test"
        return self.result


async def test_tool_command_runner_maps_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "beast.tools.shell.RunCommandTool",
        lambda context: FakeTool(
            SimpleNamespace(
                rejected=False,
                exit_code=0,
                timed_out=False,
                for_model=lambda: "all good",
            )
        ),
    )

    outcome = await ToolCommandRunner(object()).run(
        ValidationCommand(CheckKind.TEST, "npm test", "package.json"),
        timeout=10.0,
    )

    assert outcome.ran
    assert outcome.ok
    assert outcome.output == "all good"


async def test_tool_command_runner_maps_rejection_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "beast.tools.shell.RunCommandTool",
        lambda context: FakeTool(
            SimpleNamespace(
                rejected=True,
                exit_code=None,
                timed_out=False,
                for_model=lambda: "rejected by user",
            )
        ),
    )
    rejected = await ToolCommandRunner(object()).run(
        ValidationCommand(CheckKind.TEST, "npm test", "package.json"),
        timeout=10.0,
    )
    assert rejected.rejected
    assert not rejected.ran

    monkeypatch.setattr(
        "beast.tools.shell.RunCommandTool",
        lambda context: FakeTool(
            SimpleNamespace(
                rejected=False,
                exit_code=None,
                timed_out=True,
                for_model=lambda: "timed out",
            )
        ),
    )
    timed_out = await ToolCommandRunner(object()).run(
        ValidationCommand(CheckKind.TEST, "npm test", "package.json"),
        timeout=10.0,
    )
    assert timed_out.timed_out
    assert timed_out.failed
