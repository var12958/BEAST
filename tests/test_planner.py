"""Planner parsing: JSON tolerance, normalisation and bounded plans."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import pytest

from beast.core.planner import (
    MAX_TITLE_CHARS,
    Plan,
    PlanError,
    PlannedTask,
    TaskPlanner,
)


class Chunk:
    def __init__(self, content: str) -> None:
        self.content = content


class ScriptedPlannerModel:
    """A model that streams a fixed answer and records its prompts."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []

    async def stream_chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        options: Mapping[str, Any] | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> AsyncIterator[Chunk]:
        self.calls.append((model, [dict(message) for message in messages]))
        answer = self.answers.pop(0) if self.answers else ""
        for line in answer.splitlines(keepends=True):
            yield Chunk(line)
        if answer and not answer.endswith(("\n", "\r")):
            return


@pytest.fixture
def planner() -> TaskPlanner:
    return TaskPlanner(ScriptedPlannerModel(), "test-model", max_tasks=4)


def parse(planner: TaskPlanner, answer: str, **kwargs: Any) -> Plan:
    return planner.parse(answer, request="build a feature", **kwargs)


def test_parses_a_plain_json_plan(planner: TaskPlanner) -> None:
    plan = parse(
        planner,
        """
        {"goal": "Ship a page", "tasks": [
          {"id": "t1", "title": "Inspect", "description": "Read the app"},
          {"id": "t2", "title": "Build", "depends_on": ["t1"]}
        ]}
        """,
    )

    assert plan.goal == "Ship a page"
    assert [task.id for task in plan.tasks] == ["t1", "t2"]
    assert plan.tasks[1].depends_on == ("t1",)
    assert plan.warnings == ()


def test_accepts_fences_and_surrounding_prose(planner: TaskPlanner) -> None:
    plan = parse(
        planner,
        "Here is the plan:\n```json\n"
        '{"tasks": [{"id": "a", "title": "One"}, {"id": "b", "title": "Two"}]}\n'
        "```\nGood luck.",
    )

    assert [task.id for task in plan.tasks] == ["a", "b"]
    assert plan.goal == "build a feature"


def test_accepts_a_bare_list_and_generates_ids(planner: TaskPlanner) -> None:
    plan = parse(planner, '["Inspect the app", "Add the feature", "Run tests"]')

    assert [task.id for task in plan.tasks] == ["t1", "t2", "t3"]
    assert plan.tasks[0].title == "Inspect the app"


def test_accepts_alternate_field_names(planner: TaskPlanner) -> None:
    plan = parse(
        planner,
        '{"plan": [{"id": "read", "step": "Read", "details": "Look around"}, '
        '{"id": "write", "task": "Write", "depends": "read"}]}',
    )

    assert [task.title for task in plan.tasks] == ["Read", "Write"]
    assert plan.tasks[1].depends_on == ("read",)


def test_duplicate_and_empty_ids_are_normalised(planner: TaskPlanner) -> None:
    plan = parse(
        planner,
        '{"tasks": [{"id": "work", "title": "First"}, '
        '{"id": "work", "title": "Second"}, {"title": "Third"}]}',
    )

    ids = [task.id for task in plan.tasks]
    assert ids[0] == "work"
    assert ids[1] == "work-2"
    assert ids[2] == "t3"
    assert any("Duplicate task id" in warning for warning in plan.warnings)


def test_unknown_dependencies_are_dropped(planner: TaskPlanner) -> None:
    plan = parse(
        planner,
        '{"tasks": [{"id": "a", "title": "A", "depends_on": ["ghost"]}]}',
    )

    assert plan.tasks[0].depends_on == ()
    assert any("unknown" in warning for warning in plan.warnings)


def test_cycles_are_broken_and_reported(planner: TaskPlanner) -> None:
    plan = parse(
        planner,
        '{"tasks": [{"id": "a", "title": "A", "depends_on": ["b"]}, '
        '{"id": "b", "title": "B", "depends_on": ["a"]}]}',
    )

    dependencies = {task.id: task.depends_on for task in plan.tasks}
    assert not (dependencies["a"] and dependencies["b"])
    assert any("cycle" in warning for warning in plan.warnings)


def test_self_dependencies_are_removed(planner: TaskPlanner) -> None:
    plan = parse(
        planner,
        '{"tasks": [{"id": "a", "title": "A", "depends_on": ["a"]}]}',
    )

    assert plan.tasks[0].depends_on == ()


def test_plans_larger_than_the_limit_are_truncated(planner: TaskPlanner) -> None:
    entries = ", ".join(
        f'{{"id": "t{index}", "title": "Task {index}", '
        f'"depends_on": ["t{index - 1}"]}}'
        for index in range(1, 8)
    )
    plan = parse(planner, '{"tasks": [' + entries + "]}")

    assert len(plan.tasks) == 4
    assert plan.tasks[-1].id == "t4"
    assert plan.tasks[0].depends_on == ()
    assert any("only the first 4" in warning for warning in plan.warnings)


def test_titles_are_clipped(planner: TaskPlanner) -> None:
    plan = parse(
        planner,
        '{"tasks": [{"id": "a", "title": "' + "x" * 400 + '"}]}',
    )

    assert len(plan.tasks[0].title) == MAX_TITLE_CHARS


def test_oversized_and_empty_answers_are_refused(planner: TaskPlanner) -> None:
    with pytest.raises(PlanError, match="JSON"):
        parse(planner, "I will inspect the files and then write code.")
    with pytest.raises(PlanError, match="no usable tasks"):
        parse(planner, '{"goal": "nothing", "tasks": []}')


async def test_an_empty_model_answer_is_refused(planner: TaskPlanner) -> None:
    with pytest.raises(PlanError, match="empty"):
        await planner.plan("do something")


def test_entry_shapes_are_reported_but_do_not_hide_good_tasks(
    planner: TaskPlanner,
) -> None:
    plan = parse(
        planner,
        '{"tasks": [42, {"title": ""}, {"id": "ok", "title": "Valid"}]}',
    )

    assert [task.id for task in plan.tasks] == ["ok"]
    assert len(plan.warnings) >= 2


def test_revision_keeps_dependencies_on_completed_work(
    planner: TaskPlanner,
) -> None:
    plan = parse(
        planner,
        '{"tasks": [{"id": "new", "title": "New", "depends_on": ["done"]}]}',
        revision=1,
        known_extra=["done"],
    )

    assert plan.tasks[0].depends_on == ("done",)
    assert plan.revision == 1


def test_revision_caps_the_remaining_tasks(planner: TaskPlanner) -> None:
    plan = parse(
        planner,
        '{"tasks": [{"id": "a", "title": "A"}, {"id": "b", "title": "B"}, '
        '{"id": "c", "title": "C"}]}',
        revision=1,
        max_tasks=2,
    )

    assert [task.id for task in plan.tasks] == ["a", "b"]


async def test_revise_renames_ids_that_collide_with_completed_work() -> None:
    model = ScriptedPlannerModel(
        '{"tasks": [{"id": "t1", "title": "Retry"}, '
        '{"id": "t2", "title": "Next", "depends_on": ["t1"]}]}'
    )
    planner = TaskPlanner(model, "test-model", max_tasks=6)

    plan = await planner.revise(
        "build a feature",
        completed=[("t1", "Inspect")],
        remaining=[PlannedTask(id="t2", title="Old")],
        failure="tests failed",
        revision=1,
        max_tasks=5,
    )

    assert [task.id for task in plan.tasks] == ["t1-r2", "t2"]
    assert plan.tasks[1].depends_on == ("t1-r2",)
    assert any("t1→t1-r2" in warning for warning in plan.warnings)
    prompt = model.calls[0][1][-1]["content"]
    assert "tests failed" in prompt
    assert "t1" in prompt


async def test_plan_and_revise_use_the_selected_model_only() -> None:
    model = ScriptedPlannerModel(
        '{"tasks": [{"id": "a", "title": "A"}]}',
        '{"tasks": [{"id": "b", "title": "B"}]}',
    )
    planner = TaskPlanner(model, "local-model:7b", max_tasks=4)

    plan = await planner.plan("do work")
    revised = await planner.revise(
        "do work",
        completed=[("a", "A")],
        remaining=[],
        failure="broken",
    )

    assert [task.id for task in plan.tasks] == ["a"]
    assert [task.id for task in revised.tasks] == ["b"]
    assert {call[0] for call in model.calls} == {"local-model:7b"}
