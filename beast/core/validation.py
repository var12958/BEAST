"""Validation discovery and execution.

After the agent changes code, BEAST validates it with evidence from the
project itself — never with a guessed template.  Discovery reads real files
(``pyproject.toml``, ``package.json``, ``go.mod``, ``Cargo.toml``,
``Makefile``, ``pom.xml``, …) and proposes only commands that those files
declare.  Nothing is installed, and every command still goes through the
normal approval dialog.

Two kinds of checks exist:

* **Syntax checks** run in-process with the language's own parser (Python
  ``ast``, JSON, TOML).  They read files BEAST just wrote; they never execute
  the code and need no approval.
* **Commands** are executed as shell commands, one approval each, through the
  same :class:`~beast.tools.shell.RunCommandTool` used by the agent.

A report can be *failed*, *not verified* (no command ran, or the user rejected
one) or *verified* (at least one command ran and every check passed).  BEAST
never upgrades "no checks" to "passed".
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from beast.workspace.secrets import redact_secrets

#: Config files are small; never read more than this from one.
MAX_CONFIG_BYTES = 262_144
#: Source files larger than this are not syntax-checked.
MAX_SOURCE_BYTES = 2_000_000


class CheckKind(StrEnum):
    """What kind of evidence a validation command provides."""

    SYNTAX = "syntax"
    TEST = "test"
    LINT = "lint"
    TYPECHECK = "typecheck"
    BUILD = "build"


#: Lower sorts first: tests are the strongest evidence, lint is the weakest.
_KIND_PRIORITY = {
    CheckKind.TEST: 0,
    CheckKind.SYNTAX: 1,
    CheckKind.BUILD: 2,
    CheckKind.TYPECHECK: 3,
    CheckKind.LINT: 4,
}


@dataclass(frozen=True, slots=True)
class ValidationCommand:
    """One discovered command, with the file that justified it."""

    kind: CheckKind
    command: str
    source: str
    description: str = ""

    def display(self) -> str:
        return f"[{self.kind.value.upper()}] {self.command}  ({self.source})"


@dataclass(frozen=True, slots=True)
class SyntaxCheck:
    """The result of parsing one changed source file."""

    path: str
    ok: bool
    error: str = ""

    def line(self) -> str:
        return (
            f"[SYNTAX] {self.path} — ok"
            if self.ok
            else f"[SYNTAX] {self.path} — {self.error}"
        )


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    """Structured result of running one validation command."""

    exit_code: int | None
    output: str = ""
    rejected: bool = False
    timed_out: bool = False
    skipped_reason: str = ""

    @property
    def ran(self) -> bool:
        return not self.rejected and not self.skipped_reason

    @property
    def ok(self) -> bool:
        return self.ran and not self.timed_out and self.exit_code == 0

    @property
    def failed(self) -> bool:
        return self.ran and not self.ok


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """One command and what happened when it ran."""

    command: ValidationCommand
    outcome: CommandOutcome


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """The complete outcome of one validation pass."""

    syntax: tuple[SyntaxCheck, ...] = ()
    discovered: tuple[ValidationCommand, ...] = ()
    results: tuple[ValidationResult, ...] = ()
    note: str = ""
    stopped_early: bool = False
    changed_files: tuple[str, ...] = ()

    # ------------------------------------------------------------------ status

    @property
    def syntax_failures(self) -> tuple[SyntaxCheck, ...]:
        return tuple(check for check in self.syntax if not check.ok)

    @property
    def command_failures(self) -> tuple[ValidationResult, ...]:
        return tuple(result for result in self.results if result.outcome.failed)

    @property
    def rejected(self) -> bool:
        return any(result.outcome.rejected for result in self.results)

    @property
    def has_failures(self) -> bool:
        return bool(self.syntax_failures or self.command_failures)

    @property
    def commands_run(self) -> int:
        return sum(1 for result in self.results if result.outcome.ran)

    @property
    def verified(self) -> bool:
        """True only when a command ran and every check passed."""
        return (
            not self.has_failures
            and not self.rejected
            and self.commands_run > 0
        )

    @property
    def uses_only_syntax_commands(self) -> bool:
        ran = [result for result in self.results if result.outcome.ran]
        return bool(ran) and all(
            result.command.kind is CheckKind.SYNTAX for result in ran
        )

    @property
    def status_text(self) -> str:
        if self.syntax_failures:
            return f"FAILED — {len(self.syntax_failures)} syntax error(s)"
        if self.command_failures:
            return f"FAILED — {len(self.command_failures)} command(s) failed"
        if self.rejected:
            return "NOT VERIFIED — a validation command was rejected"
        if self.stopped_early:
            return "STOPPED — validation was interrupted"
        if self.verified:
            count = self.commands_run
            noun = "command" if count == 1 else "commands"
            if self.uses_only_syntax_commands:
                return (
                    f"PASSED — {count} syntax check"
                    f"{'' if count == 1 else 's'} only; no tests or build found"
                )
            return f"PASSED — {count} {noun}"
        if self.note:
            return f"NOT VERIFIED — {self.note}"
        if self.syntax:
            return "NOT VERIFIED — syntax checks passed, no test command found"
        return "NOT VERIFIED — no checks could be run"

    # ---------------------------------------------------------------- evidence

    def failure_evidence(self, *, max_chars: int = 8_000) -> str:
        """Actual error output for the model, bounded and redacted."""
        parts: list[str] = []
        for check in self.syntax_failures:
            parts.append(f"- syntax error in {check.path}: {check.error}")
        for result in self.command_failures:
            state = "timed out" if result.outcome.timed_out else (
                f"exit {result.outcome.exit_code}"
            )
            parts.append(
                f"- [{result.command.kind.value.upper()}] "
                f"{result.command.command} ({state})\n"
                f"{_clip_evidence(result.outcome.output)}"
            )
        if not parts:
            return ""
        text = "Validation failures:\n" + "\n".join(parts)
        text = redact_secrets(text).text
        if len(text) > max_chars:
            text = f"{text[: max_chars - 20].rstrip()}\n… (evidence truncated)"
        return text

    def signature(self) -> str:
        """A stable identifier for this failure set (repeated-failure detection)."""
        if not self.has_failures:
            return ""
        pieces: list[str] = []
        for check in self.syntax_failures:
            pieces.append(f"syntax:{check.path}:{check.error[:160]}")
        for result in self.command_failures:
            pieces.append(
                f"{result.command.command}:{result.outcome.exit_code}:"
                f"{_tail(result.outcome.output, 240)}"
            )
        digest = hashlib.sha256("\n".join(pieces).encode("utf-8")).hexdigest()
        return digest[:16]

    def render(self, *, limit: int = 20) -> str:
        """Compact transcript summary."""
        lines = [f"Validation: {self.status_text}"]
        for check in self.syntax[:limit]:
            lines.append(f"  {check.line()}")
        for result in self.results[:limit]:
            outcome = result.outcome
            if outcome.rejected:
                state = "rejected by user"
            elif outcome.skipped_reason:
                state = outcome.skipped_reason
            elif outcome.timed_out:
                state = "timed out"
            else:
                state = f"exit {outcome.exit_code}"
            lines.append(f"  {result.command.display()} → {state}")
        if self.note and not self.verified:
            lines.append(f"  {self.note}")
        if self.stopped_early:
            lines.append("  stopped before all checks ran")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ValidationLimits:
    """Budgets for one validation pass."""

    max_commands: int = 3
    command_timeout: float = 120.0
    max_evidence_chars: int = 8_000


class CommandRunner(Protocol):
    """Executes one discovered command (injected so tests stay deterministic)."""

    async def run(
        self, command: ValidationCommand, *, timeout: float
    ) -> CommandOutcome:
        """Run ``command`` and return its outcome."""
        ...


class ValidationEngine:
    """Discovers, executes and interprets a project's own checks."""

    def __init__(
        self,
        root: Path | str,
        runner: CommandRunner,
        *,
        limits: ValidationLimits | None = None,
        tracker: object | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.runner = runner
        self.limits = limits or ValidationLimits()
        #: Optional :class:`~beast.core.changes.ChangeTracker` for the report.
        self.tracker = tracker

    async def validate(
        self,
        changed_files: Sequence[str] = (),
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> ValidationReport:
        """Check the files BEAST changed, then run the project's commands."""
        changed = tuple(dict.fromkeys(changed_files))
        if not changed:
            return ValidationReport(
                note="no files were changed by this task",
                changed_files=changed,
            )

        syntax = self._syntax_checks(changed)
        commands = discover_validation_commands(
            self.root,
            changed_files=changed,
            max_commands=self.limits.max_commands,
        )
        if should_stop is not None and should_stop():
            return ValidationReport(
                syntax=syntax,
                discovered=commands,
                note="validation was interrupted",
                stopped_early=True,
                changed_files=changed,
            )

        results: list[ValidationResult] = []
        stopped = False
        for command in commands:
            if should_stop is not None and should_stop():
                stopped = True
                break
            outcome = await self.runner.run(
                command, timeout=self.limits.command_timeout
            )
            results.append(ValidationResult(command=command, outcome=outcome))
            self._record(command, outcome)
            if outcome.rejected:
                # Keep asking would ignore the user's decision.
                break

        note = ""
        if not commands and all(check.ok for check in syntax):
            note = "no validation commands were discovered for this project"

        return ValidationReport(
            syntax=syntax,
            discovered=commands,
            results=tuple(results),
            note=note,
            stopped_early=stopped,
            changed_files=changed,
        )

    def _syntax_checks(self, changed: Sequence[str]) -> tuple[SyntaxCheck, ...]:
        checks: list[SyntaxCheck] = []
        for relative in changed:
            checker = _SYNTAX_CHECKERS.get(Path(relative).suffix.lower())
            if checker is None:
                continue
            target = self.root / relative
            try:
                if not target.is_file() or target.stat().st_size > MAX_SOURCE_BYTES:
                    continue
                text = target.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                checks.append(
                    SyntaxCheck(relative, False, f"could not read file: {exc}")
                )
                continue
            error = checker(text, relative)
            checks.append(SyntaxCheck(relative, error is None, error or ""))
        return tuple(checks)

    def _record(self, command: ValidationCommand, outcome: CommandOutcome) -> None:
        recorder = getattr(self.tracker, "record_command", None)
        if recorder is None:
            return
        recorder(
            command.command,
            exit_code=outcome.exit_code,
            ok=outcome.ok,
            rejected=outcome.rejected,
            timed_out=outcome.timed_out,
            kind=command.kind.value,
        )


class ToolCommandRunner:
    """Runs validation commands through the approved shell tool."""

    def __init__(self, context: object) -> None:
        self.context = context

    async def run(
        self, command: ValidationCommand, *, timeout: float
    ) -> CommandOutcome:
        from beast.tools.shell import RunCommandTool

        tool = RunCommandTool(self.context)  # type: ignore[arg-type]
        result = await tool.run(
            {"command": command.command, "timeout": max(1.0, timeout)}
        )
        if result.rejected:
            return CommandOutcome(
                exit_code=None,
                output=result.for_model(),
                rejected=True,
            )
        return CommandOutcome(
            exit_code=getattr(result, "exit_code", None),
            output=result.for_model(),
            timed_out=getattr(result, "timed_out", False),
        )


# ------------------------------------------------------------------ syntax checks


def _check_python(text: str, path: str) -> str | None:
    try:
        ast.parse(text, filename=path)
    except SyntaxError as exc:
        location = f"line {exc.lineno}" if exc.lineno else "unknown line"
        detail = f"{location}: {exc.msg}"
        if exc.text:
            detail += f" — {exc.text.strip()[:160]}"
        return detail
    except ValueError as exc:  # e.g. embedded null byte
        return str(exc)
    return None


def _check_json(text: str, path: str) -> str | None:
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        return f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
    return None


def _check_toml(text: str, path: str) -> str | None:
    try:
        tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        return str(exc)
    return None


_SYNTAX_CHECKERS = {
    ".py": _check_python,
    ".pyw": _check_python,
    ".json": _check_json,
    ".toml": _check_toml,
}


# --------------------------------------------------------------------- discovery


def discover_validation_commands(
    root: Path | str,
    *,
    changed_files: Sequence[str] = (),
    max_commands: int = 3,
) -> tuple[ValidationCommand, ...]:
    """Find commands the project itself declares, strongest evidence first."""
    base = Path(root).resolve()
    commands: list[ValidationCommand] = []

    commands.extend(_discover_python(base))
    commands.extend(_discover_node(base))
    commands.extend(_discover_go(base))
    commands.extend(_discover_rust(base))
    commands.extend(_discover_java(base))
    commands.extend(_discover_dotnet(base))
    commands.extend(_discover_ruby(base))
    commands.extend(_discover_php(base))
    commands.extend(_discover_make(base))
    commands.extend(_discover_javascript_syntax(base, changed_files))

    seen: set[str] = set()
    unique: list[ValidationCommand] = []
    for command in sorted(
        commands, key=lambda item: _KIND_PRIORITY[item.kind]
    ):
        if command.command in seen:
            continue
        seen.add(command.command)
        unique.append(command)
    return tuple(unique[: max(0, max_commands)])


def _read(base: Path, relative: str) -> str | None:
    target = base / relative
    try:
        if not target.is_file():
            return None
        if target.stat().st_size > MAX_CONFIG_BYTES:
            return None
        return target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _has_python_tests(base: Path, pyproject: str | None) -> str | None:
    if pyproject and (
        "pytest" in pyproject or "[tool.pytest" in pyproject
    ):
        return "pyproject.toml"
    for relative in ("pytest.ini", "conftest.py"):
        if (base / relative).is_file():
            return relative
    for relative in ("tox.ini", "setup.cfg"):
        text = _read(base, relative)
        if text and ("[pytest]" in text or "[tool:pytest]" in text):
            return relative
    tests = base / "tests"
    if tests.is_dir() and any(tests.glob("test_*.py")):
        return "tests/"
    if any(base.glob("test_*.py")):
        return "test_*.py"
    return None


def _discover_python(base: Path) -> list[ValidationCommand]:
    commands: list[ValidationCommand] = []
    pyproject = _read(base, "pyproject.toml")
    evidence = _has_python_tests(base, pyproject)
    if evidence:
        commands.append(
            ValidationCommand(
                CheckKind.TEST,
                "python -m pytest -q",
                evidence,
                "pytest",
            )
        )
    ruff_source = None
    if pyproject and "[tool.ruff" in pyproject:
        ruff_source = "pyproject.toml"
    elif (base / "ruff.toml").is_file():
        ruff_source = "ruff.toml"
    elif (base / ".ruff.toml").is_file():
        ruff_source = ".ruff.toml"
    if ruff_source is not None:
        commands.append(
            ValidationCommand(
                CheckKind.LINT, "ruff check .", ruff_source, "ruff"
            )
        )
    mypy_source = None
    if pyproject and "[tool.mypy" in pyproject:
        mypy_source = "pyproject.toml"
    elif (base / "mypy.ini").is_file():
        mypy_source = "mypy.ini"
    if mypy_source is not None:
        commands.append(
            ValidationCommand(
                CheckKind.TYPECHECK, "mypy .", mypy_source, "mypy"
            )
        )
    return commands


def _package_manager(base: Path) -> tuple[str, str]:
    if (base / "pnpm-lock.yaml").is_file():
        return "pnpm", "pnpm-lock.yaml"
    if (base / "yarn.lock").is_file():
        return "yarn", "yarn.lock"
    if (base / "bun.lockb").is_file() or (base / "bun.lock").is_file():
        return "bun", "bun.lock"
    if (base / "package-lock.json").is_file():
        return "npm", "package-lock.json"
    return "npm", "package.json"


def _discover_node(base: Path) -> list[ValidationCommand]:
    text = _read(base, "package.json")
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    scripts = payload.get("scripts") if isinstance(payload, dict) else None
    if not isinstance(scripts, dict):
        return []
    manager, source = _package_manager(base)
    commands: list[ValidationCommand] = []

    test_script = scripts.get("test")
    if isinstance(test_script, str) and "no test specified" not in test_script:
        commands.append(
            ValidationCommand(CheckKind.TEST, _script_command(manager, "test"), source)
        )
    for name, kind in (
        ("typecheck", CheckKind.TYPECHECK),
        ("type-check", CheckKind.TYPECHECK),
        ("lint", CheckKind.LINT),
        ("build", CheckKind.BUILD),
    ):
        if name in scripts:
            commands.append(
                ValidationCommand(kind, _script_command(manager, name), source)
            )
    return commands


def _script_command(manager: str, script: str) -> str:
    if manager == "npm" and script == "test":
        return "npm test"
    if manager in {"npm", "pnpm"}:
        return f"{manager} run {script}"
    return f"{manager} {script}"


def _discover_go(base: Path) -> list[ValidationCommand]:
    if not (base / "go.mod").is_file():
        return []
    return [
        ValidationCommand(CheckKind.TEST, "go test ./...", "go.mod"),
        ValidationCommand(CheckKind.BUILD, "go build ./...", "go.mod"),
        ValidationCommand(CheckKind.LINT, "go vet ./...", "go.mod"),
    ]


def _discover_rust(base: Path) -> list[ValidationCommand]:
    if not (base / "Cargo.toml").is_file():
        return []
    commands = [
        ValidationCommand(CheckKind.TEST, "cargo test", "Cargo.toml"),
        ValidationCommand(CheckKind.BUILD, "cargo build", "Cargo.toml"),
    ]
    if (base / "clippy.toml").is_file() or (base / ".clippy.toml").is_file():
        commands.append(
            ValidationCommand(
                CheckKind.LINT, "cargo clippy -- -D warnings", "clippy.toml"
            )
        )
    return commands


def _discover_java(base: Path) -> list[ValidationCommand]:
    if (base / "pom.xml").is_file():
        return [ValidationCommand(CheckKind.TEST, "mvn -q test", "pom.xml")]
    if (base / "gradlew").is_file() or (base / "gradlew.bat").is_file():
        wrapper = "gradlew.bat" if os.name == "nt" else "./gradlew"
        return [
            ValidationCommand(CheckKind.TEST, f"{wrapper} test", "Gradle wrapper")
        ]
    return []


def _discover_dotnet(base: Path) -> list[ValidationCommand]:
    projects = list(base.glob("*.csproj")) + list(base.glob("*/*.csproj"))
    if not projects:
        return []
    commands = [
        ValidationCommand(CheckKind.BUILD, "dotnet build", projects[0].name)
    ]
    has_tests = any(
        "test" in project.name.lower() or _project_mentions_tests(project)
        for project in projects
    )
    if has_tests or (base / f"{base.name}.sln").is_file():
        commands.insert(
            0, ValidationCommand(CheckKind.TEST, "dotnet test", projects[0].name)
        )
    return commands


def _project_mentions_tests(project: Path) -> bool:
    text = _read(project.parent, project.name) or ""
    lowered = text.lower()
    return any(
        marker in lowered for marker in ("xunit", "nunit", "mstest", "microsoft.net.test.sdk")
    )


def _discover_ruby(base: Path) -> list[ValidationCommand]:
    gemfile = _read(base, "Gemfile")
    if not gemfile:
        return []
    if "rspec" in gemfile and (base / "spec").is_dir():
        return [
            ValidationCommand(CheckKind.TEST, "bundle exec rspec", "Gemfile")
        ]
    return []


def _discover_php(base: Path) -> list[ValidationCommand]:
    text = _read(base, "composer.json")
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    scripts = payload.get("scripts") if isinstance(payload, dict) else None
    if isinstance(scripts, dict) and "test" in scripts:
        return [ValidationCommand(CheckKind.TEST, "composer test", "composer.json")]
    return []


_MAKE_TARGET = re.compile(r"^([A-Za-z][\w.-]*)\s*:(?!=)", re.MULTILINE)


def _discover_make(base: Path) -> list[ValidationCommand]:
    text = _read(base, "Makefile")
    if not text:
        return []
    targets = set(_MAKE_TARGET.findall(text))
    commands: list[ValidationCommand] = []
    for target, kind in (
        ("test", CheckKind.TEST),
        ("check", CheckKind.TEST),
        ("lint", CheckKind.LINT),
        ("build", CheckKind.BUILD),
    ):
        if target in targets:
            commands.append(
                ValidationCommand(kind, f"make {target}", "Makefile")
            )
    return commands


def _discover_javascript_syntax(
    base: Path, changed_files: Sequence[str]
) -> list[ValidationCommand]:
    """Use ``node --check`` for changed plain JavaScript files, if Node exists."""
    if shutil.which("node") is None:
        return []
    commands: list[ValidationCommand] = []
    for relative in changed_files:
        if Path(relative).suffix.lower() not in {".js", ".mjs", ".cjs"}:
            continue
        target = base / relative
        if not target.is_file() or target.stat().st_size > MAX_SOURCE_BYTES:
            continue
        # The path is quoted for both POSIX shells and cmd.exe.
        commands.append(
            ValidationCommand(
                CheckKind.SYNTAX,
                f'node --check "{relative}"',
                relative,
            )
        )
        if len(commands) >= 2:
            break
    return commands


# ------------------------------------------------------------------- evidence


def _clip_evidence(text: str, *, limit: int = 3_000) -> str:
    """Keep the start and the end of long output; errors usually end it."""
    text = text.strip()
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return f"{text[:head]}\n… (output truncated) …\n{text[-tail:]}"


def _tail(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    return compact[-limit:]
