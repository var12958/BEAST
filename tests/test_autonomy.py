"""Autonomous runner behaviour with deterministic fake models and agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from beast.core.autonomy import (
    AutonomyLimits,
    AutonomousRunner,
    RunSummary,
)
from beast.core.changes import ChangeTracker
from beast.core.events import (
    AgentEvent,
    AnalysisEvent,
    ChangeSummary,
    DebugAttempt,
    PlanReady,
    PlanRevised,
    PlanStarted,
    RunFinished,
    TaskCompleted,
    TaskFailed,
    TaskRetryScheduled,
    TaskStarted,
    ValidationFinished,
    ValidationStarted,
)
from beast.core.planner import Plan, PlanError, PlannedTask
from beast.core.validation import CommandOutcome
from beast.errors import BeastError
from beast.workspace import Workspace


class FakeApprover:
    async def approve(self, request: object) -> bool:
        return True


def write(root: Path, relative: str, text: str) -> Path:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


class FakePlanner:
    def __init__(
        self,
        plan: Plan,
        *,
        revisions: tuple[Plan, ...] = (),
        fail_plan: str = "",
        fail_revise: str = "",
    ) -> None:
        self.plan_value = plan
        self.revisions = list(revisions)
        self.fail_plan = fail_plan
        self.fail_revise = fail_revise
        self.plan_calls: list[dict[str, Any]] = []
        self.revise_calls: list[dict[str, Any]] = []

    async def plan(self, request: str, *, context: str = "") -> Plan:
        self.plan_calls.append({"request": request, "context": context})
        if self.fail_plan:
            raise PlanError(self.fail_plan)
        return self.plan_value

    async def revise(
        self,
        request: str,
        *,
        context: str = "",
        completed: list[tuple[str, str]],
        remaining: list[PlannedTask],
        failure: str,
        revision: int = 1,
        max_tasks: int | None = None,
    ) -> Plan:
        self.revise_calls.append(
            {
                "request": request,
                "completed": completed,
                "remaining": remaining,
                "failure": failure,
                "revision": revision,
                "max_tasks": max_tasks,
            }
        )
        if self.fail_revise:
            raise PlanError(self.fail_revise)
        if not self.revisions:
            raise PlanError("no revision available")
        return self.revisions.pop(0)


class FakeCommands:
    def __init__(self, *outcomes: CommandOutcome) -> None:
        self.outcomes = list(outcomes)
        self.commands: list[str] = []

    async def run(self, command: object, *, timeout: float) -> CommandOutcome:
        self.commands.append(command.command)  # type: ignore[attr-defined]
        if self.outcomes:
            return self.outcomes.pop(0)
        return CommandOutcome(exit_code=0, output="passed")


class WritingAgent:
    """Writes files through the tracker, like the real file tool does."""

    def __init__(
        self,
        tracker: ChangeTracker,
        root: Path,
        files: dict[str, str] | None = None,
        *,
        prompts: list[str] | None = None,
        error: str = "",
    ) -> None:
        self.tracker = tracker
        self.root = root
        self.files = files or {}
        self.prompts = prompts if prompts is not None else []
        self.error = error

    async def run(self, prompt: str) -> Any:
        self.prompts.append(prompt)
        if self.error:
            raise BeastError(self.error)
        for relative, content in self.files.items():
            target = self.root / relative
            self.tracker.capture(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            self.tracker.finalize(target)
        if False:  # pragma: no cover - keeps this an async generator
            yield AgentEvent


class AgentFactory:
    """Builds one agent per call, in script order."""

    def __init__(self, agents: list[WritingAgent]) -> None:
        self.agents = list(agents)
        self.created: list[WritingAgent] = []

    def __call__(self) -> WritingAgent:
        if not self.agents:
            raise AssertionError("runner asked for more agents than scripted")
        agent = self.agents.pop(0)
        self.created.append(agent)
        return agent


def limits(**overrides: Any) -> AutonomyLimits:
    base: dict[str, Any] = {
        "max_tasks": 6,
        "max_agent_iterations": 4,
        "max_debug_attempts": 2,
        "max_task_attempts": 1,
        "max_plan_revisions": 1,
        "max_run_seconds": 60.0,
        "task_timeout_seconds": 30.0,
        "command_timeout_seconds": 10.0,
    }
    base.update(overrides)
    return AutonomyLimits(**base)


def one_task_plan(task_id: str = "t1", title: str = "Do the work") -> Plan:
    return Plan(goal="Ship it", tasks=(PlannedTask(task_id, title),))


async def collect(
    runner: AutonomousRunner, request: str, context: str = ""
) -> list[AgentEvent]:
    return [event async for event in runner.run(request, context=context)]


def make_runner(
    tmp_path: Path,
    planner: FakePlanner,
    agents: AgentFactory,
    commands: FakeCommands | None = None,
    *,
    tracker: ChangeTracker | None = None,
    **limit_overrides: Any,
) -> tuple[AutonomousRunner, ChangeTracker]:
    workspace = Workspace(tmp_path)
    tracker = tracker or ChangeTracker(tmp_path)
    for agent in agents.agents:
        # Agents write through the same tracker the runner validates from.
        agent.tracker = tracker
    runner = AutonomousRunner(
        object(),
        "local-model",
        workspace=workspace,
        approver=FakeApprover(),
        tracker=tracker,
        limits=limits(**limit_overrides),
        planner=planner,  # type: ignore[arg-type]
        command_runner=commands or FakeCommands(),
        agent_factory=agents,
        supports_tools=True,
    )
    return runner, tracker


def of_type(events: list[AgentEvent], kind: type) -> list[Any]:
    return [event for event in events if isinstance(event, kind)]


def final_summary(events: list[AgentEvent]) -> RunSummary:
    finished = of_type(events, RunFinished)
    assert len(finished) == 1
    return finished[0].summary


# ------------------------------------------------------------------- happy path


async def test_successful_two_task_run_is_planned_executed_and_validated(
    tmp_path: Path,
) -> None:
    write(tmp_path, "pyproject.toml", "[tool.pytest.ini_options]\n")
    plan = Plan(
        goal="Ship it",
        tasks=(
            PlannedTask("t1", "Create the page"),
            PlannedTask("t2", "Add behaviour", depends_on=("t1",)),
        ),
    )
    prompts: list[str] = []
    tracker = ChangeTracker(tmp_path)
    agents = AgentFactory(
        [
            WritingAgent(tracker, tmp_path, {"index.html": "<html>"}, prompts=prompts),
            WritingAgent(tracker, tmp_path, {"styles.css": ".x { color: red; }"}, prompts=prompts),
        ]
    )
    commands = FakeCommands(
        CommandOutcome(exit_code=0), CommandOutcome(exit_code=0)
    )
    runner, _ = make_runner(
        tmp_path, FakePlanner(plan), agents, commands
    )

    events = await collect(runner, "build a page")

    assert isinstance(events[0], PlanStarted)
    ready = of_type(events, PlanReady)[0]
    assert [task.id for task in ready.tasks] == ["t1", "t2"]
    assert len(of_type(events, TaskStarted)) == 2
    assert len(of_type(events, TaskCompleted)) == 2
    assert len(of_type(events, ValidationStarted)) == 2
    assert len(of_type(events, ValidationFinished)) == 2
    assert len(of_type(events, ChangeSummary)) == 3
    summary = final_summary(events)
    assert summary.verdict == "completed"
    assert summary.progress.total == 2
    assert summary.progress.completed == 2
    assert summary.verified_tasks == 2
    assert summary.validation_runs == 2
    assert summary.changes.summary_line() == "2 created, 0 modified, 0 deleted"
    assert "2/2 tasks completed" in summary.render()
    assert commands.commands == ["python -m pytest -q", "python -m pytest -q"]
    assert "Original request: build a page" in prompts[0]
    assert "Remaining tasks (do NOT do these now):\n- [t2] Add behaviour" in prompts[0]
    assert "[t1] Create the page" in prompts[1]
    assert "Files BEAST has already changed" in prompts[1]
    assert runner.finished


async def test_a_task_without_changes_completes_but_is_not_verified(
    tmp_path: Path,
) -> None:
    tracker = ChangeTracker(tmp_path)
    agents = AgentFactory([WritingAgent(tracker, tmp_path)])
    runner, _ = make_runner(tmp_path, FakePlanner(one_task_plan()), agents)

    events = await collect(runner, "inspect only")

    summary = final_summary(events)
    assert summary.verdict == "completed"
    assert summary.verified_tasks == 0
    assert summary.unverified_tasks == 1
    assert summary.tasks[0].verified is None
    assert summary.validation_runs == 1
    assert "no files were changed by this task" in summary.tasks[0].detail


# --------------------------------------------------------------- self-correction


async def test_a_failed_validation_is_fed_back_and_the_fix_is_revalidated(
    tmp_path: Path,
) -> None:
    write(tmp_path, "pyproject.toml", "[tool.pytest.ini_options]\n")
    tracker = ChangeTracker(tmp_path)
    agents = AgentFactory(
        [
            WritingAgent(tracker, tmp_path, {"app.txt": "broken"}),
            WritingAgent(tracker, tmp_path, {"fix.txt": "fixed"}),
        ]
    )
    commands = FakeCommands(
        CommandOutcome(exit_code=1, output="FAILED tests/test_app.py - boom"),
        CommandOutcome(exit_code=0, output="passed"),
    )
    runner, _ = make_runner(tmp_path, FakePlanner(one_task_plan()), agents, commands)

    events = await collect(runner, "fix it")

    debug = of_type(events, DebugAttempt)
    assert len(debug) == 1
    assert debug[0].attempt == 1
    assert debug[0].max_attempts == 2
    assert "FAILED tests/test_app.py" in debug[0].evidence
    summary = final_summary(events)
    assert summary.verdict == "completed"
    assert summary.verified_tasks == 1
    assert summary.debug_attempts == 1
    assert summary.validation_runs == 2
    fix_prompt = agents.created[0].prompts[1]
    assert "FAILED tests/test_app.py" in fix_prompt
    assert "correction attempt 1" in fix_prompt
    assert "Do not claim it is fixed" in fix_prompt


async def test_debug_attempts_are_bounded_and_then_the_task_fails(
    tmp_path: Path,
) -> None:
    write(tmp_path, "pyproject.toml", "[tool.pytest.ini_options]\n")
    tracker = ChangeTracker(tmp_path)
    agents = AgentFactory(
        [
            WritingAgent(tracker, tmp_path, {"a.txt": "1"}),
            WritingAgent(tracker, tmp_path, {"b.txt": "2"}),
            WritingAgent(tracker, tmp_path, {"c.txt": "3"}),
        ]
    )
    commands = FakeCommands(
        CommandOutcome(exit_code=1, output="error one"),
        CommandOutcome(exit_code=1, output="error two"),
        CommandOutcome(exit_code=1, output="error three"),
    )
    runner, _ = make_runner(
        tmp_path,
        FakePlanner(one_task_plan()),
        agents,
        commands,
        max_debug_attempts=2,
    )

    events = await collect(runner, "fix it")

    assert len(of_type(events, DebugAttempt)) == 2
    failed = of_type(events, TaskFailed)
    assert len(failed) == 1
    assert "after 2 correction attempt(s)" in failed[0].detail
    summary = final_summary(events)
    assert summary.verdict == "failed"
    assert summary.progress.failed == 1
    assert summary.debug_attempts == 2
    assert summary.validation_runs == 3


async def test_a_repeated_failure_signature_stops_the_loop_early(
    tmp_path: Path,
) -> None:
    write(tmp_path, "pyproject.toml", "[tool.pytest.ini_options]\n")
    tracker = ChangeTracker(tmp_path)
    agents = AgentFactory(
        [
            WritingAgent(tracker, tmp_path, {"a.txt": "1"}),
            WritingAgent(tracker, tmp_path, {"b.txt": "2"}),
            WritingAgent(tracker, tmp_path, {"c.txt": "3"}),
        ]
    )
    commands = FakeCommands(
        CommandOutcome(exit_code=1, output="identical"),
        CommandOutcome(exit_code=1, output="identical"),
        CommandOutcome(exit_code=1, output="identical"),
    )
    runner, _ = make_runner(
        tmp_path,
        FakePlanner(one_task_plan()),
        agents,
        commands,
        max_debug_attempts=4,
    )

    events = await collect(runner, "fix it")

    assert len(of_type(events, DebugAttempt)) == 1
    failed = of_type(events, TaskFailed)[0]
    assert "same validation failure repeated" in failed.detail
    summary = final_summary(events)
    assert summary.verdict == "failed"
    assert summary.debug_attempts == 1
    assert summary.validation_runs == 2
    # The correction ran on the same agent (its context is preserved) and no
    # second correction was requested after the repeated signature.
    assert len(agents.created) == 1
    assert len(agents.created[0].prompts) == 2


async def test_a_rejected_validation_marks_the_work_unverified_and_stops(
    tmp_path: Path,
) -> None:
    write(tmp_path, "pyproject.toml", "[tool.pytest.ini_options]\n")
    tracker = ChangeTracker(tmp_path)
    agents = AgentFactory(
        [WritingAgent(tracker, tmp_path, {"app.txt": "x"})]
    )
    commands = FakeCommands(
        CommandOutcome(exit_code=None, rejected=True, output="rejected")
    )
    runner, _ = make_runner(
        tmp_path, FakePlanner(one_task_plan()), agents, commands
    )

    events = await collect(runner, "change it")

    summary = final_summary(events)
    assert summary.verdict == "completed"
    assert summary.tasks[0].status.value == "COMPLETED"
    assert summary.tasks[0].verified is None
    assert summary.unverified_tasks == 1
    assert summary.debug_attempts == 0
    assert "unverified" in summary.tasks[0].detail


async def test_a_model_error_fails_the_task_without_claiming_success(
    tmp_path: Path,
) -> None:
    tracker = ChangeTracker(tmp_path)
    agents = AgentFactory(
        [WritingAgent(tracker, tmp_path, error="the model stopped responding")]
    )
    runner, _ = make_runner(tmp_path, FakePlanner(one_task_plan()), agents)

    events = await collect(runner, "do work")

    failed = of_type(events, TaskFailed)[0]
    assert "the model stopped responding" in failed.detail
    summary = final_summary(events)
    assert summary.verdict == "failed"
    assert summary.tasks[0].verified is False
    assert of_type(events, ValidationStarted) == []


# ---------------------------------------------------------------- retries and plans


async def test_a_task_is_retried_until_the_attempt_limit(tmp_path: Path) -> None:
    write(tmp_path, "pyproject.toml", "[tool.pytest.ini_options]\n")
    tracker = ChangeTracker(tmp_path)
    agents = AgentFactory(
        [
            WritingAgent(tracker, tmp_path, {"a.txt": "first"}),
            WritingAgent(tracker, tmp_path, {"b.txt": "second"}),
        ]
    )
    commands = FakeCommands(
        CommandOutcome(exit_code=1, output="still broken"),
        CommandOutcome(exit_code=0, output="passed"),
    )
    runner, _ = make_runner(
        tmp_path,
        FakePlanner(one_task_plan()),
        agents,
        commands,
        max_debug_attempts=0,
        max_task_attempts=2,
    )

    events = await collect(runner, "do work")

    assert len(of_type(events, TaskStarted)) == 2
    retry = of_type(events, TaskRetryScheduled)
    assert len(retry) == 1
    assert retry[0].attempt == 1
    assert retry[0].max_attempts == 2
    summary = final_summary(events)
    assert summary.verdict == "completed"
    assert summary.tasks[0].attempts == 2
    assert summary.tasks[0].verified is True


async def test_a_failed_task_can_trigger_a_bounded_plan_revision(
    tmp_path: Path,
) -> None:
    write(tmp_path, "pyproject.toml", "[tool.pytest.ini_options]\n")
    plan = Plan(
        goal="Ship it",
        tasks=(
            PlannedTask("t1", "Broken path"),
            PlannedTask("t2", "Old next step"),
        ),
    )
    revised = Plan(
        goal="Ship it",
        tasks=(PlannedTask("r1", "New next step"),),
        revision=1,
    )
    tracker = ChangeTracker(tmp_path)
    agents = AgentFactory(
        [
            WritingAgent(tracker, tmp_path, error="cannot do this"),
            WritingAgent(tracker, tmp_path, {"fixed.txt": "done"}),
        ]
    )
    planner = FakePlanner(plan, revisions=(revised,))
    runner, _ = make_runner(tmp_path, planner, agents)

    events = await collect(runner, "do work")

    revised_events = of_type(events, PlanRevised)
    assert len(revised_events) == 1
    assert [task.id for task in revised_events[0].tasks] == ["r1"]
    assert revised_events[0].revision == 1
    call = planner.revise_calls[0]
    assert [task.id for task in call["remaining"]] == ["t2"]
    assert call["completed"] == []
    assert "cannot do this" in call["failure"]
    summary = final_summary(events)
    assert summary.verdict == "partial"
    assert summary.revisions == 1
    completed = [task.id for task in summary.completed_tasks]
    assert completed == ["r1"]


async def test_a_failed_revision_is_reported_and_execution_continues(
    tmp_path: Path,
) -> None:
    write(tmp_path, "pyproject.toml", "[tool.pytest.ini_options]\n")
    plan = Plan(
        goal="Ship it",
        tasks=(
            PlannedTask("t1", "Broken path"),
            PlannedTask("t2", "Still doable"),
        ),
    )
    tracker = ChangeTracker(tmp_path)
    agents = AgentFactory(
        [
            WritingAgent(tracker, tmp_path, error="cannot do this"),
            WritingAgent(tracker, tmp_path, {"done.txt": "done"}),
        ]
    )
    planner = FakePlanner(plan, fail_revise="the model produced no JSON")
    runner, _ = make_runner(tmp_path, planner, agents)

    events = await collect(runner, "do work")

    analysis = of_type(events, AnalysisEvent)
    assert analysis and analysis[0].stage == "plan"
    assert "continuing with the current plan" in analysis[0].message
    summary = final_summary(events)
    assert summary.verdict == "partial"
    assert sorted(task.id for task in summary.completed_tasks) == ["t2"]


async def test_planning_failures_stop_the_run_before_any_task(
    tmp_path: Path,
) -> None:
    planner = FakePlanner(one_task_plan(), fail_plan="not valid JSON")
    agents = AgentFactory([])
    runner, _ = make_runner(tmp_path, planner, agents)

    events = await collect(runner, "do work")

    summary = final_summary(events)
    assert summary.verdict == "stopped"
    assert "planning failed" in summary.stopped_reason
    assert summary.tasks == ()
    assert planner.plan_calls


# --------------------------------------------------------------------- controls


async def test_a_model_without_tool_support_is_refused_not_switched(
    tmp_path: Path,
) -> None:
    planner = FakePlanner(one_task_plan())
    runner = AutonomousRunner(
        object(),
        "no-tools-model",
        workspace=Workspace(tmp_path),
        approver=FakeApprover(),
        limits=limits(),
        planner=planner,  # type: ignore[arg-type]
        agent_factory=AgentFactory([]),
        supports_tools=False,
    )

    events = await collect(runner, "do work")

    summary = final_summary(events)
    assert summary.verdict == "stopped"
    assert "does not support tool calling" in summary.stopped_reason
    assert "will not switch models silently" in summary.stopped_reason
    assert planner.plan_calls == []
    assert summary.model == "no-tools-model"


async def test_cancellation_before_execution_cancels_every_task(
    tmp_path: Path,
) -> None:
    planner = FakePlanner(
        Plan(
            goal="Ship it",
            tasks=(PlannedTask("t1", "One"), PlannedTask("t2", "Two")),
        )
    )
    agents = AgentFactory([])
    runner, _ = make_runner(tmp_path, planner, agents)
    runner.request_stop()

    events = await collect(runner, "do work")

    summary = final_summary(events)
    assert summary.cancelled
    assert summary.verdict == "cancelled"
    assert summary.progress.cancelled == 2
    assert "stopped by the user" in summary.stopped_reason
    assert of_type(events, TaskStarted) == []


async def test_the_run_time_limit_stops_the_loop(tmp_path: Path) -> None:
    planner = FakePlanner(one_task_plan())
    runner, _ = make_runner(
        tmp_path,
        planner,
        AgentFactory([]),
        max_run_seconds=0.0,
    )

    events = await collect(runner, "do work")

    summary = final_summary(events)
    assert summary.verdict == "stopped"
    assert "time limit" in summary.stopped_reason
    assert summary.progress.cancelled == 1


async def test_the_task_time_limit_is_reported_honestly(tmp_path: Path) -> None:
    values = [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 5.0]

    def clock() -> float:
        return values.pop(0) if values else 5.0

    tracker = ChangeTracker(tmp_path)
    agents = AgentFactory([WritingAgent(tracker, tmp_path, {"a.txt": "x"})])
    runner = AutonomousRunner(
        object(),
        "local-model",
        workspace=Workspace(tmp_path),
        approver=FakeApprover(),
        tracker=tracker,
        limits=limits(max_run_seconds=60.0, task_timeout_seconds=1.0),
        planner=FakePlanner(one_task_plan()),  # type: ignore[arg-type]
        command_runner=FakeCommands(),
        agent_factory=agents,
        supports_tools=True,
        clock=clock,
    )

    events = await collect(runner, "do work")

    failed = of_type(events, TaskFailed)
    assert failed
    assert "time limit" in failed[0].detail
    summary = final_summary(events)
    assert summary.verdict == "failed"


async def test_an_agent_that_cannot_be_created_fails_the_task(
    tmp_path: Path,
) -> None:
    def broken_factory() -> WritingAgent:
        raise RuntimeError("model unavailable")

    runner = AutonomousRunner(
        object(),
        "local-model",
        workspace=Workspace(tmp_path),
        approver=FakeApprover(),
        limits=limits(),
        planner=FakePlanner(one_task_plan()),  # type: ignore[arg-type]
        command_runner=FakeCommands(),
        agent_factory=broken_factory,
        supports_tools=True,
    )

    events = await collect(runner, "do work")

    failed = of_type(events, TaskFailed)[0]
    assert "could not start the agent" in failed.detail
    assert final_summary(events).verdict == "failed"


# ---------------------------------------------------------------------- limits


def test_limits_are_read_from_the_environment_and_clamped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BEAST_MAX_TASKS", "99")
    monkeypatch.setenv("BEAST_MAX_DEBUG_ATTEMPTS", "not-a-number")
    monkeypatch.setenv("BEAST_COMMAND_TIMEOUT", "1")
    monkeypatch.setenv("BEAST_MAX_RUN_SECONDS", "999999")

    parsed = AutonomyLimits.from_env()

    assert parsed.max_tasks == 50
    assert parsed.max_debug_attempts == 2
    assert parsed.command_timeout_seconds == 5
    assert parsed.max_run_seconds == 7_200
    assert parsed.validation_limits().command_timeout == 5


def test_validation_limits_carry_the_command_timeout() -> None:
    parsed = limits(command_timeout_seconds=42.0)

    assert parsed.validation_limits().command_timeout == 42.0
    assert parsed.validation_limits().max_commands == 3
