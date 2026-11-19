"""Task graph behaviour: dependencies, retries, cancellation and revisions."""

from __future__ import annotations

import pytest

from beast.core.task_manager import (
    Task,
    TaskGraphError,
    TaskManager,
    TaskStatus,
)


def task(task_id: str, *depends_on: str, **kwargs: object) -> Task:
    return Task(id=task_id, title=f"Task {task_id}", depends_on=depends_on, **kwargs)


def test_tasks_start_with_no_attempts_and_are_immutable() -> None:
    original = task("t1")
    started = original.transition(TaskStatus.IN_PROGRESS, attempts=1)

    assert original.status is TaskStatus.PENDING
    assert original.attempts == 0
    assert started.status is TaskStatus.IN_PROGRESS
    assert started.attempts == 1


def test_ready_respects_dependencies_and_plan_order() -> None:
    manager = TaskManager([task("a"), task("b", "a"), task("c", "a")])

    assert [item.id for item in manager.ready()] == ["a"]

    manager.start("a")
    manager.complete("a", verified=True)

    assert [item.id for item in manager.ready()] == ["b", "c"]
    assert [item.id for item in manager.ready()][0] == "b"


def test_starting_before_dependencies_complete_is_refused() -> None:
    manager = TaskManager([task("a"), task("b", "a")])

    with pytest.raises(TaskGraphError, match="dependencies"):
        manager.start("b")


def test_duplicate_unknown_self_and_cyclic_dependencies_are_rejected() -> None:
    with pytest.raises(TaskGraphError, match="Duplicate"):
        TaskManager([task("a"), task("a")])
    with pytest.raises(TaskGraphError, match="unknown"):
        TaskManager([task("a", "missing")])
    with pytest.raises(TaskGraphError, match="itself"):
        TaskManager([task("a", "a")])
    with pytest.raises(TaskGraphError, match="cycle"):
        TaskManager([task("a", "b"), task("b", "a")])


def test_a_failed_dependency_blocks_dependants() -> None:
    manager = TaskManager(
        [task("a"), task("b", "a"), task("c", "b"), task("d")]
    )
    manager.start("a")
    manager.fail("a", "it broke")  # max_attempts=1 → FAILED

    blocked = manager.block_unreachable("dependency failed")

    assert {item.id for item in blocked} == {"b", "c"}
    assert manager.get("b").status is TaskStatus.BLOCKED
    assert manager.get("c").status is TaskStatus.BLOCKED
    assert manager.get("d").status is TaskStatus.PENDING
    assert manager.next_task().id == "d"


def test_retries_are_bounded_then_the_failure_is_final() -> None:
    manager = TaskManager([task("a", max_attempts=2)])

    started = manager.start("a")
    assert started.attempts == 1
    retry = manager.fail("a", "first failure")
    assert retry.status is TaskStatus.PENDING
    assert retry.can_retry

    started = manager.start("a")
    assert started.attempts == 2
    failed = manager.fail("a", "second failure")
    assert failed.status is TaskStatus.FAILED
    assert failed.verified is False
    assert manager.is_finished


def test_completed_work_records_verification_state() -> None:
    manager = TaskManager([task("a")])
    manager.start("a")
    completed = manager.complete("a", detail="checked", verified=None)

    assert completed.status is TaskStatus.COMPLETED
    assert completed.verified is None
    assert manager.progress().completed == 1
    assert manager.progress().percent == 100


def test_cancel_all_only_changes_unfinished_tasks() -> None:
    manager = TaskManager([task("a"), task("b")])
    manager.start("a")
    manager.complete("a")

    cancelled = manager.cancel_all("stop")

    assert [item.id for item in cancelled] == ["b"]
    assert manager.get("a").status is TaskStatus.COMPLETED
    assert manager.get("b").status is TaskStatus.CANCELLED


def test_revision_preserves_completed_tasks_and_archives_unfinished() -> None:
    manager = TaskManager([task("a"), task("b", "a")], max_revisions=1)
    manager.start("a")
    manager.complete("a", verified=True)
    manager.start("b")

    revised = manager.revise(
        [Task(id="c", title="New path", depends_on=("a",))],
        reason="b failed",
    )

    assert [item.id for item in revised] == ["a", "c"]
    assert manager.get("a").status is TaskStatus.COMPLETED
    assert [item.id for item in manager.history] == ["b"]
    assert manager.revisions == 1
    assert "b failed" in manager.last_revision_reason
    assert manager.next_task().id == "c"


def test_a_bad_revision_changes_nothing() -> None:
    manager = TaskManager([task("a")], max_revisions=1)
    manager.start("a")
    manager.complete("a")

    with pytest.raises(TaskGraphError, match="unknown"):
        manager.revise([Task(id="b", title="b", depends_on=("missing",))])

    assert [item.id for item in manager.tasks] == ["a"]
    assert manager.revisions == 0


def test_revision_limit_and_task_limit_are_enforced() -> None:
    manager = TaskManager([task("a"), task("b")], max_revisions=1)
    manager.revise([Task(id="c", title="c")])

    with pytest.raises(TaskGraphError, match="already revised"):
        manager.revise([Task(id="d", title="d")])

    small = TaskManager([task("a")], max_tasks=2)
    small.start("a")
    small.complete("a")
    with pytest.raises(TaskGraphError, match="limit"):
        small.revise(
            [
                Task(id="b", title="b"),
                Task(id="c", title="c"),
            ]
        )


def test_progress_line_reports_every_bucket() -> None:
    manager = TaskManager(
        [task("a"), task("b"), task("c"), task("d")]
    )
    manager.start("a")
    manager.complete("a")
    manager.start("b")
    manager.fail("b", "failed")
    manager.cancel("c")

    progress = manager.progress()

    assert (progress.total, progress.completed, progress.failed) == (4, 1, 1)
    assert progress.cancelled == 1
    assert progress.pending == 1
    assert "3/4" in progress.line()
    assert progress.percent == 75
