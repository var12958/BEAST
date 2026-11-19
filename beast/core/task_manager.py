"""Task state for autonomous execution.

A plan is produced by the model (see :mod:`beast.core.planner`); this module
owns what happens to that plan afterwards: statuses, dependencies, retries,
cancellation and revisions.

Design rules:

* A :class:`Task` is an immutable value.  Every transition returns a new task
  and the manager stores the new value, so nothing observes a half-applied
  change.
* Dependencies are explicit ids.  A task becomes ready only when *every* task
  it depends on is ``COMPLETED``.
* A task whose dependency failed (or was cancelled) can never run; it is
  marked ``BLOCKED`` instead of waiting forever.
* Cancellation, retries and revisions are bounded so a run cannot loop.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

#: Safety valve for a model that returns an enormous plan.
DEFAULT_MAX_TASKS = 50


class TaskStatus(StrEnum):
    """The lifecycle of one development task."""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"


#: Statuses a task never leaves on its own.
TERMINAL_STATUSES = frozenset(
    {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.BLOCKED,
        TaskStatus.CANCELLED,
    }
)

#: Statuses that make dependants unreachable.
UNREACHABLE_STATUSES = frozenset(
    {TaskStatus.FAILED, TaskStatus.BLOCKED, TaskStatus.CANCELLED}
)


class TaskGraphError(ValueError):
    """A plan's task graph cannot be executed as written."""


@dataclass(frozen=True, slots=True)
class Task:
    """One unit of development work.

    ``attempts`` counts how many times the task has been started.  A task may
    be retried while ``attempts < max_attempts``; after that a failure is final.
    ``verified`` records whether the validation engine actually proved the
    work: ``True`` passed, ``False`` failed, ``None`` means it could not be
    checked (no commands, rejected approval, or no changes).
    """

    id: str
    title: str
    description: str = ""
    depends_on: tuple[str, ...] = ()
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    max_attempts: int = 1
    detail: str = ""
    verified: bool | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def can_retry(self) -> bool:
        return self.attempts < self.max_attempts

    def transition(self, status: TaskStatus, **changes: object) -> Task:
        """Return a copy with ``status`` and the given changes applied."""
        return replace(self, status=status, **changes)


@dataclass(frozen=True, slots=True)
class TaskProgress:
    """A compact snapshot for the terminal progress panel."""

    total: int
    completed: int
    failed: int
    blocked: int
    cancelled: int
    pending: int
    running: int

    @classmethod
    def initial(cls, total: int) -> TaskProgress:
        """The state of a freshly parsed plan."""
        return cls(
            total=total,
            completed=0,
            failed=0,
            blocked=0,
            cancelled=0,
            pending=total,
            running=0,
        )

    @property
    def finished(self) -> int:
        return self.completed + self.failed + self.blocked + self.cancelled

    @property
    def percent(self) -> int:
        """Progress towards a finished plan (0-100)."""
        if self.total <= 0:
            return 100
        return round(self.finished * 100 / self.total)

    def line(self) -> str:
        return (
            f"{self.finished}/{self.total} tasks ({self.percent}%)"
            f" · {self.completed} completed · {self.failed} failed"
            f" · {self.blocked} blocked · {self.pending + self.running} remaining"
        )


class TaskManager:
    """Owns the mutable state of one plan while keeping tasks immutable."""

    def __init__(
        self,
        tasks: list[Task] | tuple[Task, ...],
        *,
        max_tasks: int = DEFAULT_MAX_TASKS,
        max_revisions: int = 2,
    ) -> None:
        if not tasks:
            raise TaskGraphError("A plan must contain at least one task.")
        if len(tasks) > max_tasks:
            raise TaskGraphError(
                f"The plan has {len(tasks)} tasks; the limit is {max_tasks}."
            )
        self._max_tasks = max_tasks
        self._max_revisions = max_revisions
        self._order: list[str] = []
        self._tasks: dict[str, Task] = {}
        self._history: list[Task] = []
        self._revisions = 0
        self._last_revision_reason = ""
        self._load(tasks)

    # -------------------------------------------------------------- construction

    def _load(self, tasks: list[Task] | tuple[Task, ...]) -> None:
        """Validate and install a complete task set."""
        seen: set[str] = set()
        for task in tasks:
            if not task.id:
                raise TaskGraphError("Every task needs a non-empty id.")
            if task.id in seen:
                raise TaskGraphError(f"Duplicate task id {task.id!r}.")
            seen.add(task.id)

        for task in tasks:
            if task.id in task.depends_on:
                raise TaskGraphError(f"Task {task.id!r} depends on itself.")
            missing = [dep for dep in task.depends_on if dep not in seen]
            if missing:
                raise TaskGraphError(
                    f"Task {task.id!r} depends on unknown task(s): "
                    f"{', '.join(missing)}."
                )

        self._assert_acyclic(tasks)
        self._order = [task.id for task in tasks]
        self._tasks = {task.id: task for task in tasks}

    @staticmethod
    def _assert_acyclic(tasks: list[Task] | tuple[Task, ...]) -> None:
        by_id = {task.id: task for task in tasks}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str, trail: tuple[str, ...]) -> None:
            if task_id in visited:
                return
            if task_id in visiting:
                cycle = " → ".join((*trail, task_id))
                raise TaskGraphError(f"The plan contains a dependency cycle: {cycle}.")
            visiting.add(task_id)
            for dependency in by_id[task_id].depends_on:
                visit(dependency, (*trail, task_id))
            visiting.discard(task_id)
            visited.add(task_id)

        for task in tasks:
            visit(task.id, ())

    # ------------------------------------------------------------------ reading

    @property
    def tasks(self) -> tuple[Task, ...]:
        """Every active task, in plan order."""
        return tuple(self._tasks[task_id] for task_id in self._order)

    @property
    def history(self) -> tuple[Task, ...]:
        """Tasks replaced by a plan revision."""
        return tuple(self._history)

    @property
    def revisions(self) -> int:
        return self._revisions

    @property
    def last_revision_reason(self) -> str:
        return self._last_revision_reason

    @property
    def current(self) -> Task | None:
        return next(
            (
                task
                for task in self.tasks
                if task.status is TaskStatus.IN_PROGRESS
            ),
            None,
        )

    @property
    def is_finished(self) -> bool:
        return all(task.is_terminal for task in self.tasks)

    @property
    def has_failures(self) -> bool:
        return any(
            task.status in {TaskStatus.FAILED, TaskStatus.BLOCKED}
            for task in self.tasks
        )

    def get(self, task_id: str) -> Task:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise KeyError(f"Unknown task {task_id!r}.") from exc

    @property
    def completed_ids(self) -> tuple[str, ...]:
        return tuple(
            task.id
            for task in self.tasks
            if task.status is TaskStatus.COMPLETED
        )

    def progress(self) -> TaskProgress:
        counts = {status: 0 for status in TaskStatus}
        for task in self.tasks:
            counts[task.status] += 1
        return TaskProgress(
            total=len(self._tasks),
            completed=counts[TaskStatus.COMPLETED],
            failed=counts[TaskStatus.FAILED],
            blocked=counts[TaskStatus.BLOCKED],
            cancelled=counts[TaskStatus.CANCELLED],
            pending=counts[TaskStatus.PENDING],
            running=counts[TaskStatus.IN_PROGRESS],
        )

    # ------------------------------------------------------------------ ordering

    def ready(self) -> tuple[Task, ...]:
        """Pending tasks whose dependencies are all completed, in plan order."""
        return tuple(
            task
            for task in self.tasks
            if task.status is TaskStatus.PENDING and self._dependencies_met(task)
        )

    def next_task(self) -> Task | None:
        """The first ready task, or ``None`` when nothing can run."""
        ready = self.ready()
        return ready[0] if ready else None

    def _dependencies_met(self, task: Task) -> bool:
        return all(
            self._tasks[dependency].status is TaskStatus.COMPLETED
            for dependency in task.depends_on
        )

    def block_unreachable(self, reason: str) -> tuple[Task, ...]:
        """Mark pending tasks whose dependencies can never complete as blocked."""
        blocked: list[Task] = []
        changed = True
        while changed:
            changed = False
            for task in self.tasks:
                if task.status is not TaskStatus.PENDING:
                    continue
                if any(
                    self._tasks[dependency].status in UNREACHABLE_STATUSES
                    for dependency in task.depends_on
                ):
                    blocked_task = task.transition(
                        TaskStatus.BLOCKED,
                        detail=reason,
                        verified=False,
                    )
                    self._tasks[task.id] = blocked_task
                    blocked.append(blocked_task)
                    changed = True
        return tuple(blocked)

    # ------------------------------------------------------------------ mutation

    def start(self, task_id: str) -> Task:
        """Move a pending task to ``IN_PROGRESS`` and count the attempt."""
        task = self._require(task_id)
        if task.status is not TaskStatus.PENDING:
            raise TaskGraphError(
                f"Task {task_id!r} cannot start from {task.status.value}."
            )
        if not self._dependencies_met(task):
            raise TaskGraphError(
                f"Task {task_id!r} cannot start before its dependencies complete."
            )
        started = task.transition(
            TaskStatus.IN_PROGRESS,
            attempts=task.attempts + 1,
            detail="",
        )
        self._tasks[task_id] = started
        return started

    def complete(
        self,
        task_id: str,
        *,
        detail: str = "",
        verified: bool | None = None,
    ) -> Task:
        """Finish a running task successfully."""
        task = self._require_running(task_id)
        completed = task.transition(
            TaskStatus.COMPLETED, detail=detail, verified=verified
        )
        self._tasks[task_id] = completed
        return completed

    def fail(self, task_id: str, detail: str) -> Task:
        """Record a failure; retry while attempts remain, then fail finally."""
        task = self._require_running(task_id)
        if task.can_retry:
            pending = task.transition(
                TaskStatus.PENDING, detail=detail, verified=False
            )
            self._tasks[task_id] = pending
            return pending
        failed = task.transition(
            TaskStatus.FAILED, detail=detail, verified=False
        )
        self._tasks[task_id] = failed
        return failed

    def mark_blocked(self, task_id: str, reason: str) -> Task:
        task = self._require(task_id)
        if task.is_terminal:
            raise TaskGraphError(
                f"Task {task_id!r} is already {task.status.value}."
            )
        blocked = task.transition(
            TaskStatus.BLOCKED, detail=reason, verified=False
        )
        self._tasks[task_id] = blocked
        return blocked

    def cancel(self, task_id: str, reason: str = "cancelled") -> Task:
        task = self._require(task_id)
        if task.is_terminal:
            return task
        cancelled = task.transition(TaskStatus.CANCELLED, detail=reason)
        self._tasks[task_id] = cancelled
        return cancelled

    def cancel_all(self, reason: str = "cancelled") -> tuple[Task, ...]:
        """Cancel every task that is not finished yet."""
        return tuple(
            self.cancel(task.id, reason)
            for task in self.tasks
            if not task.is_terminal
        )

    # ------------------------------------------------------------------ revision

    def revise(
        self,
        tasks: list[Task] | tuple[Task, ...],
        *,
        reason: str = "",
    ) -> tuple[Task, ...]:
        """Replace the unfinished part of the plan with ``tasks``.

        Completed work is preserved and its ids may be used as dependencies.
        Unfinished tasks are archived in :attr:`history`.  The new graph must be
        acyclic and within the task limit, otherwise nothing changes.
        """
        if self._revisions >= self._max_revisions:
            raise TaskGraphError(
                f"The plan was already revised {self._revisions} time(s)."
            )
        completed = [
            task for task in self.tasks if task.status is TaskStatus.COMPLETED
        ]
        if len(completed) + len(tasks) > self._max_tasks:
            raise TaskGraphError(
                f"The revised plan would have {len(completed) + len(tasks)} "
                f"tasks; the limit is {self._max_tasks}."
            )
        combined = [*completed, *tasks]
        # Validate against a scratch manager so a bad revision changes nothing.
        TaskManager(
            combined,
            max_tasks=self._max_tasks,
            max_revisions=self._max_revisions,
        )
        self._history.extend(
            task for task in self.tasks if not task.is_terminal
        )
        self._revisions += 1
        self._last_revision_reason = reason
        self._order = [task.id for task in combined]
        self._tasks = {task.id: task for task in combined}
        return tuple(self.tasks)

    def _require(self, task_id: str) -> Task:
        return self.get(task_id)

    def _require_running(self, task_id: str) -> Task:
        task = self._require(task_id)
        if task.status is not TaskStatus.IN_PROGRESS:
            raise TaskGraphError(
                f"Task {task_id!r} is {task.status.value}, not IN_PROGRESS."
            )
        return task
