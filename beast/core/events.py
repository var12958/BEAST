"""Events emitted by the agent loop and the autonomous runner.

The UI consumes these instead of driving execution itself, so the engines stay
headless and testable while the terminal can show text, tool activity, plan
progress, validation and approval waits as they happen.

Phase 1-3 events (``TextDelta`` … ``ToolFinished``) are unchanged; Phase 4
adds the planning, task, validation, debugging and summary events below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from beast.core.changes import ChangeReport
from beast.core.planner import Plan
from beast.core.task_manager import Task, TaskProgress
from beast.core.validation import ValidationReport
from beast.intelligence.models import ContextBundle
from beast.tools.base import Permission, ToolResult

if TYPE_CHECKING:  # pragma: no cover - avoids an import cycle at runtime
    from beast.core.autonomy import RunSummary


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A piece of the assistant's reply."""

    text: str


@dataclass(frozen=True, slots=True)
class AnalysisEvent:
    """Progress from the project-analysis stage, before the model is asked."""

    stage: str
    message: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ContextReady:
    """The project context that will accompany this turn."""

    bundle: ContextBundle


@dataclass(frozen=True, slots=True)
class ToolStarted:
    """The model asked for a tool and the agent is about to run it."""

    name: str
    arguments: Any
    permission: Permission


@dataclass(frozen=True, slots=True)
class ApprovalPending:
    """A write or command is waiting for the user's decision."""

    name: str


@dataclass(frozen=True, slots=True)
class ToolFinished:
    """A tool call completed, failed or was rejected."""

    name: str
    result: ToolResult


# ---------------------------------------------------------------- autonomous mode


@dataclass(frozen=True, slots=True)
class PlanStarted:
    """The runner asked the model to decompose the request."""

    request: str
    model: str


@dataclass(frozen=True, slots=True)
class PlanReady:
    """The model's plan parsed into an executable task graph."""

    plan: Plan
    tasks: tuple[Task, ...]


@dataclass(frozen=True, slots=True)
class PlanRevised:
    """The model replaced the unfinished part of the plan."""

    plan: Plan
    reason: str
    revision: int
    tasks: tuple[Task, ...]


@dataclass(frozen=True, slots=True)
class TaskStarted:
    """One task began running; ``progress`` is the board after the change."""

    task: Task
    progress: TaskProgress


@dataclass(frozen=True, slots=True)
class TaskCompleted:
    """A task finished and its work passed validation (or could not be checked)."""

    task: Task
    progress: TaskProgress


@dataclass(frozen=True, slots=True)
class TaskFailed:
    """A task exhausted its attempts, or validation rejected its work."""

    task: Task
    progress: TaskProgress
    detail: str = ""


@dataclass(frozen=True, slots=True)
class TaskBlocked:
    """A task can never run because a dependency failed."""

    task: Task
    reason: str
    progress: TaskProgress


@dataclass(frozen=True, slots=True)
class TaskRetryScheduled:
    """A failed task was returned to PENDING for another attempt."""

    task: Task
    attempt: int
    max_attempts: int


@dataclass(frozen=True, slots=True)
class ValidationStarted:
    """The runner is about to check a task's changes."""

    task_id: str
    files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ValidationFinished:
    """Validation ran, with a report the UI can summarise."""

    task_id: str
    report: ValidationReport


@dataclass(frozen=True, slots=True)
class DebugAttempt:
    """The runner is asking the model to fix a specific, observed failure."""

    task_id: str
    attempt: int
    max_attempts: int
    signature: str
    evidence: str


@dataclass(frozen=True, slots=True)
class ChangeSummary:
    """What BEAST created, modified and deleted so far."""

    report: ChangeReport


@dataclass(frozen=True, slots=True)
class RunFinished:
    """The autonomous run ended; ``summary`` is the authoritative result."""

    summary: "RunSummary"


AgentEvent = (
    TextDelta
    | AnalysisEvent
    | ContextReady
    | ToolStarted
    | ApprovalPending
    | ToolFinished
    | PlanStarted
    | PlanReady
    | PlanRevised
    | TaskStarted
    | TaskCompleted
    | TaskFailed
    | TaskBlocked
    | TaskRetryScheduled
    | ValidationStarted
    | ValidationFinished
    | DebugAttempt
    | ChangeSummary
    | RunFinished
)
