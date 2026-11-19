"""Bounded autonomous execution.

:class:`AutonomousRunner` connects the pieces added in Phase 4:

1. the model decomposes the request into a plan (:mod:`beast.core.planner`);
2. a :class:`~beast.core.task_manager.TaskManager` tracks dependencies and
   statuses;
3. each task is executed by a fresh :class:`~beast.core.agent.Agent` that may
   use the normal tools — every write and command still needs approval;
4. the task's files are validated with the project's own commands
   (:mod:`beast.core.validation`);
5. failures are fed back to the model for a bounded number of corrections;
6. if a task still fails, the model may revise the unfinished plan;
7. every outcome is reported as an event and in a final :class:`RunSummary`.

Limits come from :class:`AutonomyLimits` and are configurable through
``BEAST_*`` environment variables.  There is no path through this module that
installs packages, commits, pushes, or edits outside the workspace.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from beast.errors import BeastError
from beast.tools.base import Approver, ToolContext
from beast.tools.registry import ToolRegistry
from beast.workspace import Workspace

from .agent import Agent
from .changes import ChangeReport, ChangeTracker
from .events import (
    AgentEvent,
    AnalysisEvent,
    ChangeSummary,
    DebugAttempt,
    PlanReady,
    PlanRevised,
    PlanStarted,
    RunFinished,
    TaskBlocked,
    TaskCompleted,
    TaskFailed,
    TaskRetryScheduled,
    TaskStarted,
    ValidationFinished,
    ValidationStarted,
)
from .planner import Plan, PlanError, PlannedTask, TaskPlanner
from .task_manager import (
    Task,
    TaskGraphError,
    TaskManager,
    TaskProgress,
    TaskStatus,
)
from .validation import (
    CommandRunner,
    ToolCommandRunner,
    ValidationEngine,
    ValidationLimits,
    ValidationReport,
)

if TYPE_CHECKING:  # pragma: no cover
    from beast.intelligence import ProjectIntelligence
    from beast.intelligence.context_builder import ContextProvider

logger = logging.getLogger(__name__)

#: System prompt for the agents that execute individual plan tasks.
AUTONOMOUS_SYSTEM = (
    "You are BEAST's task executor. You are given ONE task from a larger plan "
    "and you complete only that task inside the user's project.\n"
    "\n"
    "Rules:\n"
    "- Use the available tools to inspect the project before changing it.\n"
    "- Make the smallest coherent change that completes this task.\n"
    "- Read a file before editing it, and preserve unrelated code and behaviour.\n"
    "- Match the project's existing style, structure and conventions.\n"
    "- Do not implement later tasks from the plan.\n"
    "- Do not install dependencies. If something required is missing, stop and "
    "explain what is missing instead of installing it.\n"
    "- The runtime requires approval for writes and commands; never claim an "
    "action was approved or succeeded without a tool result.\n"
    "- BEAST will validate the result. Do not claim the task is fixed or "
    "complete if validation has not run.\n"
    "- When the task is done, reply with a short factual summary of what changed.\n"
)


class AgentLike(Protocol):
    """The part of ``Agent`` the runner needs (injectable in tests)."""

    def run(self, prompt: str) -> AsyncIterator[AgentEvent]:
        """Run one prompt and stream events."""
        ...


#: Builds a fresh agent for one task.
AgentFactory = Callable[[], AgentLike]


class TaskTimeoutError(RuntimeError):
    """A task or the whole run exceeded its time budget."""


def _env_int(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring non-numeric %s=%r", name, raw)
        return default
    return max(minimum, min(maximum, value))


def _env_float(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Ignoring non-numeric %s=%r", name, raw)
        return default
    return max(minimum, min(maximum, value))


@dataclass(frozen=True, slots=True)
class AutonomyLimits:
    """Everything that can stop a run, all bounded and configurable."""

    max_tasks: int = 12
    #: Model turns allowed inside one ``Agent.run`` call.
    max_agent_iterations: int = 12
    #: Fix attempts after the first failed validation, per task.
    max_debug_attempts: int = 2
    #: Whole-task retries (a new agent run), per task.
    max_task_attempts: int = 2
    #: Times the model may rewrite the unfinished plan.
    max_plan_revisions: int = 2
    #: Wall-clock budget for the whole run.
    max_run_seconds: float = 900.0
    #: Wall-clock budget for one task, including debugging.
    task_timeout_seconds: float = 300.0
    #: Ceiling for one validation command.
    command_timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls) -> AutonomyLimits:
        """Read overrides from ``BEAST_*`` variables, clamped to safe ranges."""
        return cls(
            max_tasks=_env_int(
                "BEAST_MAX_TASKS", 12, minimum=1, maximum=50
            ),
            max_agent_iterations=_env_int(
                "BEAST_MAX_AGENT_ITERATIONS", 12, minimum=1, maximum=60
            ),
            max_debug_attempts=_env_int(
                "BEAST_MAX_DEBUG_ATTEMPTS", 2, minimum=0, maximum=5
            ),
            max_task_attempts=_env_int(
                "BEAST_MAX_TASK_ATTEMPTS", 2, minimum=1, maximum=5
            ),
            max_plan_revisions=_env_int(
                "BEAST_MAX_PLAN_REVISIONS", 2, minimum=0, maximum=5
            ),
            max_run_seconds=_env_float(
                "BEAST_MAX_RUN_SECONDS", 900.0, minimum=10.0, maximum=7_200.0
            ),
            task_timeout_seconds=_env_float(
                "BEAST_TASK_TIMEOUT", 300.0, minimum=10.0, maximum=3_600.0
            ),
            command_timeout_seconds=_env_float(
                "BEAST_COMMAND_TIMEOUT", 120.0, minimum=5.0, maximum=600.0
            ),
        )

    def validation_limits(self) -> ValidationLimits:
        return ValidationLimits(
            max_commands=3,
            command_timeout=self.command_timeout_seconds,
        )


@dataclass(frozen=True, slots=True)
class RunSummary:
    """The authoritative outcome of an autonomous run."""

    goal: str
    tasks: tuple[Task, ...]
    progress: TaskProgress
    changes: ChangeReport
    duration: float
    cancelled: bool = False
    stopped_reason: str = ""
    revisions: int = 0
    debug_attempts: int = 0
    validation_runs: int = 0
    model: str = ""

    @property
    def completed_tasks(self) -> tuple[Task, ...]:
        return tuple(
            task for task in self.tasks if task.status is TaskStatus.COMPLETED
        )

    @property
    def failed_tasks(self) -> tuple[Task, ...]:
        return tuple(
            task
            for task in self.tasks
            if task.status in {TaskStatus.FAILED, TaskStatus.BLOCKED}
        )

    @property
    def verified_tasks(self) -> int:
        return sum(
            1 for task in self.completed_tasks if task.verified is True
        )

    @property
    def unverified_tasks(self) -> int:
        return sum(
            1 for task in self.completed_tasks if task.verified is None
        )

    @property
    def verdict(self) -> str:
        """One of ``cancelled``, ``stopped``, ``completed``, ``partial``, ``failed``."""
        if self.cancelled:
            return "cancelled"
        if self.stopped_reason and not self.progress.finished:
            return "stopped"
        if self.failed_tasks:
            return "partial" if self.completed_tasks else "failed"
        if self.progress.completed == self.progress.total:
            return "completed"
        return "stopped"

    def render(self) -> str:
        lines = [
            f"Run {self.verdict} — {self.progress.completed}/"
            f"{self.progress.total} tasks completed"
            + (
                f", {self.progress.failed} failed"
                if self.progress.failed
                else ""
            )
            + (
                f", {self.progress.blocked} blocked"
                if self.progress.blocked
                else ""
            )
            + (
                f", {self.progress.cancelled} cancelled"
                if self.progress.cancelled
                else ""
            ),
            f"Validation: {self.verified_tasks} verified, "
            f"{self.unverified_tasks} completed without verification",
            f"Changes: {self.changes.summary_line()}",
            f"Limits used: {self.debug_attempts} debug attempt(s), "
            f"{self.revisions} plan revision(s), {self.validation_runs} validation pass(es)",
            f"Duration: {self.duration:.0f}s",
        ]
        if self.stopped_reason:
            lines.append(f"Note: {self.stopped_reason}")
        return "\n".join(lines)


class AutonomousRunner:
    """Plans, executes, validates and repairs a request within hard bounds."""

    def __init__(
        self,
        client: object,
        model: str,
        *,
        workspace: Workspace,
        approver: Approver,
        intelligence: "ProjectIntelligence | None" = None,
        tracker: ChangeTracker | None = None,
        limits: AutonomyLimits | None = None,
        planner: TaskPlanner | None = None,
        command_runner: CommandRunner | None = None,
        agent_factory: AgentFactory | None = None,
        context_provider: "ContextProvider | None" = None,
        supports_tools: bool | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client
        self._model = model
        self._workspace = workspace
        self._approver = approver
        self._intelligence = intelligence
        self._limits = limits or AutonomyLimits.from_env()
        self._tracker = tracker or ChangeTracker(workspace.root)
        self._planner = planner or TaskPlanner(
            client,  # type: ignore[arg-type]
            model,
            max_tasks=self._limits.max_tasks,
        )
        self._context_provider = context_provider
        self._agent_factory = agent_factory
        self._supports_tools = supports_tools
        self._clock = clock
        self._tool_context = ToolContext(
            workspace=workspace,
            approver=approver,
            command_timeout=self._limits.command_timeout_seconds,
            intelligence=intelligence,
            changes=self._tracker,
        )
        self._command_runner: CommandRunner = (
            command_runner or ToolCommandRunner(self._tool_context)
        )
        self._validation = ValidationEngine(
            workspace.root,
            self._command_runner,
            limits=self._limits.validation_limits(),
            tracker=self._tracker,
        )
        self._cancel_requested = False
        self._finished = False
        self._debug_attempts = 0
        self._validation_runs = 0
        self._last_failure_evidence = ""
        self._terminal: dict[str, Task] = {}

    # -------------------------------------------------------------------- state

    @property
    def changes(self) -> ChangeTracker:
        return self._tracker

    @property
    def limits(self) -> AutonomyLimits:
        return self._limits

    @property
    def finished(self) -> bool:
        return self._finished

    def request_stop(self) -> None:
        """Ask the run to stop at the next safe point (cooperative)."""
        self._cancel_requested = True

    # ---------------------------------------------------------------------- run

    async def run(
        self,
        request: str,
        *,
        context: str = "",
    ) -> AsyncIterator[AgentEvent]:
        """Execute ``request`` autonomously, yielding progress as it happens."""
        started = self._clock()
        deadline = started + self._limits.max_run_seconds
        manager: TaskManager | None = None
        plan: Plan | None = None
        stopped_reason = ""

        try:
            yield PlanStarted(request=request, model=self._model)
            if self._supports_tools is False:
                stopped_reason = (
                    "the selected model does not support tool calling; BEAST "
                    "will not switch models silently"
                )
                yield self._finish(
                    request, manager, plan, started, stopped_reason=stopped_reason
                )
                return

            try:
                plan = await self._planner.plan(request, context=context)
                manager = TaskManager(
                    [
                        Task(
                            id=task.id,
                            title=task.title,
                            description=task.description,
                            depends_on=task.depends_on,
                            max_attempts=self._limits.max_task_attempts,
                        )
                        for task in plan.tasks
                    ],
                    max_tasks=self._limits.max_tasks,
                    max_revisions=self._limits.max_plan_revisions,
                )
            except (PlanError, TaskGraphError) as exc:
                stopped_reason = f"planning failed: {exc.message}"
                yield self._finish(
                    request, None, None, started, stopped_reason=stopped_reason
                )
                return

            yield PlanReady(plan=plan, tasks=manager.tasks)

            task_runs = 0
            hard_task_limit = (
                self._limits.max_tasks
                * (self._limits.max_task_attempts + 1)
                + self._limits.max_plan_revisions * self._limits.max_tasks
                + 5
            )

            while not manager.is_finished:
                if self._should_stop(deadline):
                    stopped_reason = self._stop_reason(deadline)
                    manager.cancel_all(stopped_reason)
                    self._remember_all(manager)
                    break
                if task_runs >= hard_task_limit:
                    stopped_reason = "the task-run safety limit was reached"
                    manager.cancel_all(stopped_reason)
                    self._remember_all(manager)
                    break

                task = manager.next_task()
                if task is None:
                    blocked = manager.block_unreachable(
                        "a dependency failed or was cancelled"
                    )
                    for item in blocked:
                        self._remember(item)
                        yield TaskBlocked(
                            task=item,
                            reason=item.detail,
                            progress=manager.progress(),
                        )
                    if manager.is_finished:
                        break
                    stopped_reason = (
                        "no task can run because its dependencies cannot complete"
                    )
                    manager.cancel_all(stopped_reason)
                    self._remember_all(manager)
                    break

                task_runs += 1
                running = manager.start(task.id)
                self._remember(running)
                yield TaskStarted(task=running, progress=manager.progress())

                async for event in self._execute_task(
                    manager, running, request, plan, deadline
                ):
                    yield event

                updated = manager.get(task.id)
                if updated.status is TaskStatus.FAILED:
                    blocked = manager.block_unreachable(
                        f"dependency {task.id} failed"
                    )
                    for item in blocked:
                        self._remember(item)
                        yield TaskBlocked(
                            task=item,
                            reason=item.detail,
                            progress=manager.progress(),
                        )
                    async for event in self._maybe_revise(
                        manager,
                        request,
                        context,
                        failure=updated.detail,
                    ):
                        yield event

            yield ChangeSummary(report=self._tracker.report())
            yield self._finish(
                request,
                manager,
                plan,
                started,
                stopped_reason=stopped_reason,
            )
        except asyncio.CancelledError:
            self._cancel_requested = True
            if manager is not None:
                manager.cancel_all("cancelled by the user")
                self._remember_all(manager)
            raise
        finally:
            self._finished = True

    # ----------------------------------------------------------------- one task

    async def _execute_task(
        self,
        manager: TaskManager,
        task: Task,
        request: str,
        plan: Plan,
        run_deadline: float,
    ) -> AsyncIterator[AgentEvent]:
        """Run one task, validate it and repair it within the configured bounds."""
        task_deadline = min(
            self._clock() + self._limits.task_timeout_seconds, run_deadline
        )
        evidence = ""
        try:
            agent = self._make_agent()
        except Exception as exc:  # noqa: BLE001 - report, never crash the UI
            logger.exception("Could not create an agent for %s", task.id)
            async for event in self._fail_task(
                manager, task, f"could not start the agent: {exc}"
            ):
                yield event
            return

        prompt = self._task_prompt(task, request, plan, manager)
        try:
            async for event in self._run_agent(agent, prompt, task_deadline):
                yield event
        except TaskTimeoutError:
            async for event in self._fail_task(
                manager, task, "the task exceeded its time limit"
            ):
                yield event
            return
        except (BeastError, asyncio.CancelledError) as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            async for event in self._fail_task(
                manager, task, f"the model failed: {exc.message}"
            ):
                yield event
            return
        except Exception as exc:  # noqa: BLE001 - a model error is a task failure
            logger.exception("Task %s failed while running the model", task.id)
            async for event in self._fail_task(
                manager, task, f"the model failed: {exc}"
            ):
                yield event
            return

        files = self._tracker.changed_paths()
        yield ValidationStarted(task_id=task.id, files=files)
        report = await self._validate(task, task_deadline)
        yield ValidationFinished(task_id=task.id, report=report)
        yield ChangeSummary(report=self._tracker.report())

        attempt = 0
        previous_signature = ""
        while report.has_failures:
            signature = report.signature()
            if attempt >= self._limits.max_debug_attempts:
                async for event in self._fail_task(
                    manager,
                    task,
                    (
                        f"validation still fails after {attempt} correction "
                        f"attempt(s):\n{report.failure_evidence(max_chars=2_000)}"
                    ),
                ):
                    yield event
                return
            if signature and signature == previous_signature:
                async for event in self._fail_task(
                    manager,
                    task,
                    (
                        "the same validation failure repeated; BEAST stopped "
                        "instead of retrying the identical fix.\n"
                        f"{report.failure_evidence(max_chars=2_000)}"
                    ),
                ):
                    yield event
                return
            if self._should_stop(task_deadline):
                async for event in self._fail_task(
                    manager, task, self._stop_reason(task_deadline)
                ):
                    yield event
                return

            attempt += 1
            self._debug_attempts += 1
            previous_signature = signature
            evidence = report.failure_evidence(
                max_chars=self._limits.validation_limits().max_evidence_chars
            )
            yield DebugAttempt(
                task_id=task.id,
                attempt=attempt,
                max_attempts=self._limits.max_debug_attempts,
                signature=signature,
                evidence=evidence,
            )

            try:
                fix_prompt = self._fix_prompt(task, evidence, attempt)
                async for event in self._run_agent(
                    agent, fix_prompt, task_deadline
                ):
                    yield event
            except TaskTimeoutError:
                async for event in self._fail_task(
                    manager, task, "the correction attempt exceeded the task limit"
                ):
                    yield event
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reported as task evidence
                evidence = f"the correction attempt failed to run: {exc}"
                logger.warning("Correction attempt for %s failed: %s", task.id, exc)
            files = self._tracker.changed_paths()
            yield ValidationStarted(task_id=task.id, files=files)
            report = await self._validate(task, task_deadline)
            yield ValidationFinished(task_id=task.id, report=report)
            yield ChangeSummary(report=self._tracker.report())

        # Validation resolved: pass only with command evidence.
        if report.stopped_early and self._should_stop(task_deadline):
            async for event in self._fail_task(
                manager, task, self._stop_reason(task_deadline)
            ):
                yield event
            return
        if report.verified:
            completed = manager.complete(
                task.id,
                detail=f"validated: {report.status_text}",
                verified=True,
            )
            self._remember(completed)
            yield TaskCompleted(task=completed, progress=manager.progress())
        elif report.rejected:
            completed = manager.complete(
                task.id,
                detail=(
                    "a validation command was rejected, so the work is "
                    "unverified; BEAST did not retry it"
                ),
                verified=None,
            )
            self._remember(completed)
            yield TaskCompleted(task=completed, progress=manager.progress())
        else:
            completed = manager.complete(
                task.id,
                detail=report.status_text,
                verified=None,
            )
            self._remember(completed)
            yield TaskCompleted(task=completed, progress=manager.progress())

    async def _validate(
        self, task: Task, deadline: float
    ) -> ValidationReport:
        files = self._tracker.changed_paths()
        self._validation_runs += 1
        report = await self._validation.validate(
            files,
            should_stop=lambda: self._should_stop(deadline),
        )
        logger.info(
            "validation for %s: %s (%d file(s))",
            task.id,
            report.status_text,
            len(files),
        )
        return report

    async def _run_agent(
        self,
        agent: AgentLike,
        prompt: str,
        deadline: float,
    ) -> AsyncIterator[AgentEvent]:
        """Stream an agent run, stopping cooperatively at the deadline."""
        async for event in agent.run(prompt):
            if self._should_stop(deadline):
                raise TaskTimeoutError(self._stop_reason(deadline))
            yield event

    async def _fail_task(
        self,
        manager: TaskManager,
        task: Task,
        reason: str,
    ) -> AsyncIterator[AgentEvent]:
        """Record a failure and emit the terminal or retry event."""
        self._last_failure_evidence = reason
        updated = manager.fail(task.id, reason)
        self._remember(updated)
        if updated.status is TaskStatus.PENDING:
            yield TaskRetryScheduled(
                task=updated,
                attempt=updated.attempts,
                max_attempts=updated.max_attempts,
            )
        else:
            yield TaskFailed(
                task=updated,
                progress=manager.progress(),
                detail=reason,
            )

    # ----------------------------------------------------------------- revision

    async def _maybe_revise(
        self,
        manager: TaskManager,
        request: str,
        context: str,
        *,
        failure: str,
    ) -> AsyncIterator[AgentEvent]:
        if manager.revisions >= self._limits.max_plan_revisions:
            return
        remaining = [
            PlannedTask(
                id=task.id,
                title=task.title,
                description=task.description,
                depends_on=task.depends_on,
            )
            for task in manager.tasks
            if task.status is TaskStatus.PENDING
        ]
        if not remaining:
            return
        completed = [
            (task.id, task.title)
            for task in manager.tasks
            if task.status is TaskStatus.COMPLETED
        ]
        capacity = max(1, self._limits.max_tasks - len(completed))
        try:
            revised = await self._planner.revise(
                request,
                context=context,
                completed=completed,
                remaining=remaining,
                failure=failure or self._last_failure_evidence,
                revision=manager.revisions + 1,
                max_tasks=capacity,
            )
            new_tasks = [
                Task(
                    id=task.id,
                    title=task.title,
                    description=task.description,
                    depends_on=task.depends_on,
                    max_attempts=self._limits.max_task_attempts,
                )
                for task in revised.tasks
            ]
            manager.revise(
                new_tasks,
                reason=f"after failure: {failure[:300]}",
            )
        except (PlanError, TaskGraphError, BeastError) as exc:
            yield AnalysisEvent(
                stage="plan",
                message=f"Plan revision failed; continuing with the current plan: {exc.message}",
            )
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - revision is best-effort
            logger.exception("Plan revision crashed")
            yield AnalysisEvent(
                stage="plan",
                message=f"Plan revision failed; continuing with the current plan: {exc}",
            )
            return

        yield PlanRevised(
            plan=revised,
            reason=f"after failure: {failure[:300]}",
            revision=manager.revisions,
            tasks=manager.tasks,
        )

    # -------------------------------------------------------------------- agents

    def _make_agent(self) -> AgentLike:
        if self._agent_factory is not None:
            return self._agent_factory()
        return Agent(
            self._client,  # type: ignore[arg-type]
            self._model,
            system_prompt=AUTONOMOUS_SYSTEM,
            tools=ToolRegistry.default(self._tool_context),
            max_iterations=self._limits.max_agent_iterations,
            context_provider=self._context_provider,
        )

    def _task_prompt(
        self,
        task: Task,
        request: str,
        plan: Plan,
        manager: TaskManager,
    ) -> str:
        completed = [
            f"- [{item.id}] {item.title}"
            + (f" — {item.detail}" if item.detail else "")
            for item in manager.tasks
            if item.status is TaskStatus.COMPLETED
        ]
        remaining = [
            f"- [{item.id}] {item.title}"
            for item in manager.tasks
            if item.status is TaskStatus.PENDING
        ]
        lines = [
            f"Original request: {request}",
            "",
            f"Plan goal: {plan.goal}",
            "",
            "Completed tasks:",
            *(completed or ["- none"]),
            "",
            "Remaining tasks (do NOT do these now):",
            *(remaining or ["- none"]),
            "",
            f"Current task [{task.id}]: {task.title}",
        ]
        if task.description:
            lines.extend(["", task.description])
        if task.detail:
            lines.extend(
                [
                    "",
                    "This task was attempted before and did not finish:",
                    task.detail[:4_000],
                ]
            )
        files = sorted(self._tracker.changed_paths())
        if files:
            lines.extend(
                [
                    "",
                    "Files BEAST has already changed in this run:",
                    *(f"- {path}" for path in files[:40]),
                ]
            )
        lines.extend(
            [
                "",
                "Complete only this task now. Inspect the project with tools, make "
                "the necessary changes, and finish with a short summary. BEAST "
                "will validate the result afterwards.",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _fix_prompt(task: Task, evidence: str, attempt: int) -> str:
        return (
            f"Task [{task.id}]: {task.title}\n\n"
            "BEAST ran the project's validation commands and they failed. This "
            "is the actual output:\n\n"
            f"{evidence}\n\n"
            f"Fix only these failures (correction attempt {attempt}). Re-read the "
            "relevant files, make the smallest correct change, and do not modify "
            "unrelated code. Do not claim it is fixed — BEAST will run the "
            "validation again."
        )

    # ------------------------------------------------------------------ summary

    def _finish(
        self,
        request: str,
        manager: TaskManager | None,
        plan: Plan | None,
        started: float,
        *,
        stopped_reason: str = "",
    ) -> RunFinished:
        tasks = self._collected_tasks(manager)
        progress = (
            manager.progress()
            if manager is not None
            else TaskProgress(0, 0, 0, 0, 0, 0, 0)
        )
        summary = RunSummary(
            goal=(plan.goal if plan is not None else request[:160]),
            tasks=tasks,
            progress=progress,
            changes=self._tracker.report(),
            duration=max(0.0, self._clock() - started),
            cancelled=self._cancel_requested,
            stopped_reason=stopped_reason,
            revisions=(manager.revisions if manager is not None else 0),
            debug_attempts=self._debug_attempts,
            validation_runs=self._validation_runs,
            model=self._model,
        )
        self._finished = True
        return RunFinished(summary=summary)

    def _collected_tasks(
        self, manager: TaskManager | None
    ) -> tuple[Task, ...]:
        if manager is None:
            return tuple(self._terminal.values())
        seen = set(self._terminal)
        combined = list(self._terminal.values())
        combined.extend(
            task for task in manager.tasks if task.id not in seen
        )
        return tuple(combined)

    def _remember(self, task: Task) -> None:
        if task.is_terminal:
            self._terminal[task.id] = task

    def _remember_all(self, manager: TaskManager) -> None:
        for task in manager.tasks:
            self._remember(task)

    # ------------------------------------------------------------------- limits

    def _should_stop(self, deadline: float) -> bool:
        return self._cancel_requested or self._clock() >= deadline

    def _stop_reason(self, deadline: float) -> str:
        if self._cancel_requested:
            return "stopped by the user"
        if self._clock() >= deadline:
            return "the time limit was reached"
        return "stopped"
