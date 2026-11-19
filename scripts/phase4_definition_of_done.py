"""Phase 4 definition-of-done harness.

Runs the four required demonstrations through BEAST's real Phase 4 runtime:

* **A** — build a small task-management application with tests.
* **B** — add a dark-mode toggle to an existing web page.
* **C** — find and fix a bug using the project's own failing test.
* **D** — keep failing validation until the correction limit is reached.

By default the *model decisions* are scripted (deterministic fake planner and
agents) while everything else is real: the task manager, tool registry,
approval protocol, change tracker, validation discovery, subprocess execution,
failure parsing and rollback metadata.  With ``--real`` the same harness drives
a locally installed Ollama model and a real :class:`beast.core.agent.Agent`.

The harness is unattended, so it approves file writes and only accepts commands
that look like project validation.  Any other command is rejected and the
rejection is part of the transcript.  This is a demo harness, not a sandbox:
command approval in BEAST does not provide OS-level isolation.

Usage::

    python scripts/phase4_definition_of_done.py
    python scripts/phase4_definition_of_done.py --only c
    python scripts/phase4_definition_of_done.py --real --model qwen3:8b --only a
"""

from __future__ import annotations

import argparse
import asyncio
import re
import tempfile
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from beast.core.autonomy import AutonomousRunner, AutonomyLimits, RunSummary
from beast.core.changes import ChangeTracker
from beast.core.events import (
    ApprovalPending,
    DebugAttempt,
    PlanReady,
    PlanStarted,
    RunFinished,
    TaskBlocked,
    TaskCompleted,
    TaskFailed,
    TaskRetryScheduled,
    TaskStarted,
    TextDelta,
    ToolFinished,
    ToolStarted,
    ValidationFinished,
    ValidationStarted,
)
from beast.core.planner import Plan, PlannedTask
from beast.tools import ApprovalRequest, Permission, ToolContext
from beast.tools.filesystem import WriteFileTool
from beast.workspace import Workspace

SAFE_COMMAND = re.compile(
    r"^(?:"
    r"python(?:3)? -m pytest(?:\s|$)"
    r"|pytest(?:\s|$)"
    r"|node --check(?:\s|$)"
    r"|ruff check(?:\s|$)"
    r"|mypy(?:\s|$)"
    r"|npm (?:test|run (?:test|lint|build|typecheck))(?:\s|$)"
    r"|(?:pnpm|yarn|bun) (?:test|run (?:test|lint|build|typecheck))(?:\s|$)"
    r"|go (?:test|vet)(?:\s|$)"
    r"|cargo (?:test|check)(?:\s|$)"
    r"|(?:mvn|\./mvnw|gradle|\./gradlew|gradlew\.bat) (?:\S+ )?(?:test|check)(?:\s|$)"
    r"|dotnet (?:test|build)(?:\s|$)"
    r"|(?:ruby|rake) (?:\S+ )?test(?:\s|$)"
    r"|phpunit(?:\s|$)"
    r"|make (?:test|check|all)(?:\s|$)"
    r")"
)
SHELL_METACHARACTERS = re.compile(r"[;&|<>`$]")


@dataclass(frozen=True)
class Write:
    path: str
    content: str


class DemoApprover:
    """Approves writes, and only validation-shaped commands."""

    def __init__(self) -> None:
        self.decisions: list[tuple[str, str, bool]] = []

    async def approve(self, request: ApprovalRequest) -> bool:
        target = str(request.target)
        if request.permission is Permission.WRITE:
            approved = True
        elif request.permission is Permission.EXECUTE:
            approved = bool(
                SAFE_COMMAND.match(target)
                and not SHELL_METACHARACTERS.search(target)
            )
        else:
            approved = True
        self.decisions.append((request.permission.value, target, approved))
        return approved


class FixedPlanner:
    """Planner that returns one pre-built plan (model decisions scripted)."""

    def __init__(self, plan: Plan) -> None:
        self.plan_value = plan

    async def plan(self, request: str, context: str = "") -> Plan:
        return self.plan_value

    async def revise(self, *args: Any, **kwargs: Any) -> Plan:
        raise AssertionError("this demo does not expect a plan revision")


class ScriptedAgent:
    """Performs scripted writes through the real tool and approval stack."""

    def __init__(
        self,
        steps: Sequence[Sequence[Write]],
        context: ToolContext,
    ) -> None:
        self._steps = list(steps)
        self._context = context
        self._call = 0
        self._writes = 0

    async def run(self, prompt: str) -> AsyncIterator[Any]:
        index = min(self._call, len(self._steps) - 1)
        self._call += 1
        step = self._steps[index]
        for item in step:
            arguments = {
                "path": item.path,
                "content": item.content,
            }
            yield ToolStarted(
                name="write_file",
                arguments=arguments,
                permission=Permission.WRITE,
            )
            yield ApprovalPending(name="write_file")
            result = await WriteFileTool(self._context).run(arguments)
            yield ToolFinished(name="write_file", result=result)
            if result.ok:
                self._writes += 1
        yield TextDelta(
            text=(
                f"scripted step {index + 1}: wrote {len(step)} file(s); "
                f"BEAST will validate the result."
            )
        )


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    request: str
    fixture: dict[str, str]
    writes: tuple[tuple[Write, ...], ...] = ()
    plan: tuple[tuple[str, str], ...] = ()
    max_debug_attempts: int = 2
    assertions: str = ""
    checks: tuple[str, ...] = field(default_factory=tuple)


def scenario_a() -> Scenario:
    app = '''\
"""Small in-memory task manager used by the Phase 4 demo."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Task:
    title: str
    done: bool = False


@dataclass
class TaskStore:
    tasks: list[Task] = field(default_factory=list)

    def add(self, title: str) -> Task:
        task = Task(title=title)
        self.tasks.append(task)
        return task

    def complete(self, index: int) -> Task:
        task = self.tasks[index]
        task.done = True
        return task

    def pending(self) -> list[Task]:
        return [task for task in self.tasks if not task.done]
'''
    tests = '''\
from task_app import TaskStore


def test_add_and_complete() -> None:
    store = TaskStore()
    store.add("write demo")
    store.add("run tests")
    store.complete(0)

    assert [task.title for task in store.pending()] == ["run tests"]
    assert store.tasks[0].done is True
'''
    pyproject = '[tool.pytest.ini_options]\ntestpaths = ["."]\n'
    return Scenario(
        key="a",
        title="Task-management app",
        request="Create a small task management app with tests.",
        fixture={},
        writes=(
            (
                Write("pyproject.toml", pyproject),
                Write("task_app.py", app),
                Write("test_task_app.py", tests),
            ),
        ),
        plan=(("t1", "Create the task manager and its tests"),),
        checks=("task_app.py", "test_task_app.py", "pyproject.toml"),
    )


def scenario_b() -> Scenario:
    html = '''\
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <link rel="stylesheet" href="styles.css">
    <title>Demo page</title>
  </head>
  <body>
    <main>
      <h1>Demo page</h1>
      <button id="theme-toggle" type="button">Toggle theme</button>
    </main>
    <script src="app.js"></script>
  </body>
</html>
'''
    css = '''\
:root {
  --bg: #f7f7f5;
  --fg: #1c1c1a;
  --panel: #ffffff;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font-family: system-ui, sans-serif;
}

main {
  max-width: 40rem;
  margin: 4rem auto;
  padding: 2rem;
  background: var(--panel);
}
'''
    javascript = '''\
const root = document.documentElement;

function currentTheme() {
  return root.dataset.theme || "light";
}
'''
    dark_css = css + '''
:root[data-theme="dark"] {
  --bg: #151513;
  --fg: #f2f2ee;
  --panel: #232320;
}
'''
    dark_js = '''\
const root = document.documentElement;
const toggle = document.querySelector("#theme-toggle");

function currentTheme() {
  return root.dataset.theme || "light";
}

function applyTheme(theme) {
  root.dataset.theme = theme;
}

if (toggle) {
  toggle.addEventListener("click", () => {
    applyTheme(currentTheme() === "dark" ? "light" : "dark");
  });
}
'''
    return Scenario(
        key="b",
        title="Dark-mode toggle",
        request=(
            "Add a dark mode toggle to the existing page, preserving the "
            "current light styles."
        ),
        fixture={
            "index.html": html,
            "styles.css": css,
            "app.js": javascript,
        },
        writes=(
            (
                Write("styles.css", dark_css),
                Write("app.js", dark_js),
            ),
        ),
        plan=(("t1", "Add dark mode to the existing stylesheet and script"),),
        checks=("index.html", "styles.css", "app.js"),
    )


CALCULATOR = '''\
def add(left: int, right: int) -> int:
    return left - right
'''
CALC_TEST = '''\
from calculator import add


def test_add() -> None:
    assert add(2, 3) == 5
'''
PYPROJECT = '[tool.pytest.ini_options]\ntestpaths = ["."]\n'


def scenario_c() -> Scenario:
    changed_bug = '''\
def add(left: int, right: int) -> int:
    # The existing test says this should add; reproduce the failure first.
    return left - right
'''
    fixed = '''\
def add(left: int, right: int) -> int:
    return left + right
'''
    return Scenario(
        key="c",
        title="Find and fix a bug",
        request="Make the project's failing test pass without weakening it.",
        fixture={
            "pyproject.toml": PYPROJECT,
            "calculator.py": CALCULATOR,
            "test_calculator.py": CALC_TEST,
        },
        writes=(
            (Write("calculator.py", changed_bug),),
            (Write("calculator.py", fixed),),
        ),
        plan=(("t1", "Run the failing test, find the bug and fix it"),),
        checks=("calculator.py", "test_calculator.py"),
    )


def scenario_d() -> Scenario:
    attempt_one = '''\
def add(left: int, right: int) -> int:
    return left - right  # attempt 1: still subtracts
'''
    attempt_two = '''\
def add(left: int, right: int) -> int:
    return left + right + 1  # attempt 2: still off by one
'''
    attempt_three = '''\
def add(left: int, right: int) -> int:
    return left + right + 2  # attempt 3: still wrong
'''
    return Scenario(
        key="d",
        title="Bounded correction",
        request="Make the failing test pass.",
        fixture={
            "pyproject.toml": PYPROJECT,
            "calculator.py": CALCULATOR,
            "test_calculator.py": CALC_TEST,
        },
        writes=(
            (Write("calculator.py", attempt_one),),
            (Write("calculator.py", attempt_two),),
            (Write("calculator.py", attempt_three),),
        ),
        plan=(("t1", "Fix the failing test"),),
        max_debug_attempts=2,
        checks=("calculator.py", "test_calculator.py"),
    )


SCENARIOS = {
    "a": scenario_a,
    "b": scenario_b,
    "c": scenario_c,
    "d": scenario_d,
}


def materialise(root: Path, files: dict[str, str]) -> None:
    for relative, text in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


def event_lines(event: Any) -> list[str]:
    if isinstance(event, PlanStarted):
        return [f"[PLAN] model={event.model} request={event.request!r}"]
    if isinstance(event, PlanReady):
        warnings = f" warnings={list(event.plan.warnings)}" if event.plan.warnings else ""
        return [f"[PLAN] ready: {len(event.tasks)} task(s){warnings}"]
    if isinstance(event, TaskStarted):
        return [f"[TASK] {event.task.id} started: {event.task.title}"]
    if isinstance(event, ToolStarted):
        return [f"[TOOL] {event.name} {event.arguments}"]
    if isinstance(event, ApprovalPending):
        return [f"[APPROVAL] {event.name}"]
    if isinstance(event, ToolFinished):
        state = "rejected" if event.result.rejected else ("ok" if event.result.ok else "failed")
        return [f"[TOOL] {event.name} -> {state}: {event.result.for_model()[:400]}"]
    if isinstance(event, ValidationStarted):
        return [f"[VALIDATION] started for {event.task_id}: {list(event.files)}"]
    if isinstance(event, ValidationFinished):
        return [f"[VALIDATION] {event.report.status_text}"] + [
            f"  {line}" for line in event.report.render(limit=4).splitlines()[1:]
        ]
    if isinstance(event, DebugAttempt):
        return [
            f"[DEBUG] attempt {event.attempt}/{event.max_attempts} "
            f"signature={event.signature[:12]}"
        ]
    if isinstance(event, TaskCompleted):
        verified = {True: "verified", False: "failed", None: "unverified"}[
            event.task.verified
        ]
        return [
            f"[TASK] {event.task.id} completed ({verified}): "
            f"{event.task.detail[:400]}"
        ]
    if isinstance(event, TaskFailed):
        return [f"[TASK] {event.task.id} FAILED: {event.detail[:800]}"]
    if isinstance(event, TaskBlocked):
        return [f"[TASK] {event.task.id} blocked: {event.reason[:400]}"]
    if isinstance(event, TaskRetryScheduled):
        return [
            f"[TASK] {event.task.id} retry {event.attempt}/{event.max_attempts}"
        ]
    if isinstance(event, TextDelta):
        return [f"[MODEL] {event.text[:400]}"]
    if isinstance(event, RunFinished):
        return ["[RUN]", *[f"  {line}" for line in event.summary.render().splitlines()]]
    return []


def write_transcript(
    output: Path,
    scenario: Scenario,
    model: str,
    lines: list[str],
    verdict: str,
    real: bool,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    prefix = "real-" if real else ""
    target = output / f"{prefix}{scenario.key}-{scenario.title.lower().replace(' ', '-')}.md"
    header = [
        f"# Phase 4 demo {scenario.key.upper()} — {scenario.title}",
        "",
        f"- Mode: {'real Ollama model' if real else 'scripted model, real runtime'}",
        f"- Model: `{model}`",
        f"- Verdict: **{verdict}**",
        f"- Request: {scenario.request}",
        "",
        "Everything below the request is captured from BEAST's own event stream.",
        "The unattended harness approves file writes and rejects commands that do",
        "not look like project validation.",
        "",
        "```text",
        *lines,
        "```",
        "",
    ]
    target.write_text("\n".join(header), encoding="utf-8")
    return target


def verify(
    scenario: Scenario,
    root: Path,
    summary: RunSummary,
    validation_statuses: list[str],
    runner: AutonomousRunner,
) -> list[str]:
    """Return human-readable checks (FAIL entries are reported, not raised)."""
    results: list[str] = []

    def check(label: str, condition: bool) -> None:
        results.append(f"{'PASS' if condition else 'FAIL'} — {label}")

    def text(relative: str) -> str:
        target = root / relative
        try:
            return target.read_text(encoding="utf-8")
        except OSError:
            return ""

    for relative in scenario.checks:
        check(f"{relative} exists", (root / relative).is_file())

    if scenario.key == "a":
        check("task manager has tests on disk", (root / "test_task_app.py").is_file())
        check(
            "validation ran and passed",
            any(status.startswith("PASSED") for status in validation_statuses),
        )
        check("all tasks completed", summary.progress.completed == summary.progress.total)
        check("tasks were verified", summary.verified_tasks == summary.progress.total)
        check("three files created", summary.changes.summary_line().startswith("3 created"))
    elif scenario.key == "b":
        css = text("styles.css")
        js = text("app.js")
        check("dark theme tokens added", 'data-theme="dark"' in css)
        check("original light styles preserved", "--bg: #f7f7f5" in css)
        check("toggle handler added", "addEventListener" in js)
        check("original helper preserved", "currentTheme" in js)
        check(
            "validation reported honestly",
            any("PASSED" in status or "NOT VERIFIED" in status for status in validation_statuses),
        )
    elif scenario.key == "c":
        fixed = text("calculator.py")
        check("bug fixed on disk", "left + right" in fixed and "left - right" not in fixed)
        check(
            "validation first failed",
            any(status.startswith("FAILED") for status in validation_statuses),
        )
        check(
            "validation later passed",
            any(status.startswith("PASSED") for status in validation_statuses),
        )
        check("task completed", summary.progress.completed == 1)
        check("task verified", summary.verified_tasks == 1)
        check("one debug attempt used", summary.debug_attempts == 1)
    elif scenario.key == "d":
        check("run did not claim success", summary.verdict == "failed")
        check("task is failed", summary.failed_tasks and summary.failed_tasks[0].id == "t1")
        check("no verified task", summary.verified_tasks == 0)
        check("correction limit respected", summary.debug_attempts == 2)
        check(
            "failure explains the limit",
            "still fails after 2 correction" in (summary.tasks[0].detail or ""),
        )
        check(
            "validation never passed",
            not any(status.startswith("PASSED") for status in validation_statuses),
        )

    report = runner.changes.report()
    check("change tracker recorded at least one file", bool(report.files))
    return results


async def run_scenario(
    scenario: Scenario,
    *,
    real: bool,
    model: str,
    client: Any,
) -> tuple[list[str], list[str], RunSummary, AutonomousRunner]:
    temporary = tempfile.TemporaryDirectory(prefix=f"beast-phase4-{scenario.key}-")
    root = Path(temporary.name)
    materialise(root, scenario.fixture)

    workspace = Workspace(root)
    tracker = ChangeTracker(root)
    approver = DemoApprover()
    tool_context = ToolContext(
        workspace=workspace,
        approver=approver,
        changes=tracker,
        command_timeout=120.0,
    )
    limits = AutonomyLimits(
        max_tasks=6,
        max_debug_attempts=scenario.max_debug_attempts,
        max_task_attempts=1,
        max_plan_revisions=0,
        max_run_seconds=600.0,
        task_timeout_seconds=300.0,
        command_timeout_seconds=120.0,
    )

    plan = Plan(
        goal=scenario.request,
        tasks=tuple(
            PlannedTask(id=task_id, title=title)
            for task_id, title in scenario.plan
        ),
    )
    scripted = None
    if not real:
        scripted = ScriptedAgent(scenario.writes, tool_context)

    runner = AutonomousRunner(
        client,
        model,
        workspace=workspace,
        approver=approver,
        tracker=tracker,
        limits=limits,
        planner=None if real else FixedPlanner(plan),
        agent_factory=None if real else (lambda: scripted),
        supports_tools=True,
    )

    lines: list[str] = [
        f"workspace: {root}",
        f"fixture files: {sorted(scenario.fixture)}",
        "",
    ]
    validation_statuses: list[str] = []
    summary: RunSummary | None = None
    started = time.monotonic()
    async for event in runner.run(scenario.request):
        lines.extend(event_lines(event))
        if isinstance(event, ValidationFinished):
            validation_statuses.append(event.report.status_text)
        if isinstance(event, RunFinished):
            summary = event.summary

    lines.append("")
    lines.append("approval decisions:")
    for permission, target, approved in approver.decisions:
        state = "approved" if approved else "rejected"
        lines.append(f"  {state:8} [{permission}] {target[:300]}")
    lines.append("")
    lines.append(f"elapsed: {time.monotonic() - started:.1f}s")

    if summary is None:
        raise AssertionError("the run produced no RunFinished summary")

    checks = verify(scenario, root, summary, validation_statuses, runner)
    lines.extend(["", "checks:", *[f"  {line}" for line in checks]])
    for relative in sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    ):
        lines.append(f"file on disk: {relative}")
    return lines, checks, summary, runner


async def main_async(args: argparse.Namespace) -> int:
    keys = [args.only] if args.only else list(SCENARIOS)
    model = args.model if args.real else "scripted-model"
    client: Any = None
    if args.real:
        from beast.ollama.client import OllamaClient

        client = OllamaClient()
    failures = 0
    try:
        for key in keys:
            scenario = SCENARIOS[key]()
            print(f"== demo {key.upper()}: {scenario.title} ==")
            try:
                lines, checks, summary, runner = await run_scenario(
                    scenario, real=args.real, model=model, client=client
                )
            except Exception as error:  # noqa: BLE001 - report the demo outcome
                failures += 1
                print(f"   FAILED: {error}")
                if args.real:
                    print("   (real-model demos depend on the local model and may be skipped)")
                continue
            print(summary.render())
            for line in checks:
                print(f"   {line}")
            failed = [line for line in checks if line.startswith("FAIL")]
            if failed:
                failures += 1
                print(f"   DEMO {key.upper()} DID NOT PASS ({len(failed)} check(s))")
            target = write_transcript(
                args.output,
                scenario,
                model,
                lines,
                summary.verdict,
                args.real,
            )
            print(f"   transcript: {target}")
    finally:
        if client is not None:
            await client.close()
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Phase 4 definition-of-done demonstrations."
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="use the selected local Ollama model instead of scripted model decisions",
    )
    parser.add_argument("--model", default="qwen3:8b", help="Ollama model for --real")
    parser.add_argument("--only", choices=list(SCENARIOS), help="run one demo")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/phase4-dod"),
        help="directory for Markdown transcripts",
    )
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
