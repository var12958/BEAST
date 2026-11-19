"""Change tracking, unified diffs, conservative rollback and read-only Git."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from beast.core.changes import (
    ChangeKind,
    ChangeTracker,
    GitInspector,
    GitStatus,
)
from beast.tools import ApprovalRequest, ToolContext
from beast.tools.filesystem import WriteFileTool
from beast.tools.shell import RunCommandTool
from beast.workspace import Workspace


def write(root: Path, relative: str, text: str) -> Path:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


class Yes:
    def __init__(self, answer: bool = True) -> None:
        self.answer = answer
        self.requests: list[ApprovalRequest] = []

    async def approve(self, request: ApprovalRequest) -> bool:
        self.requests.append(request)
        return self.answer


# ------------------------------------------------------------------- tracking


def test_created_files_are_recorded_without_a_snapshot(tmp_path: Path) -> None:
    tracker = ChangeTracker(tmp_path)
    target = tmp_path / "new.txt"

    tracker.capture(target)
    write(tmp_path, "new.txt", "hello")
    change = tracker.finalize(target)

    assert change is not None
    assert change.kind is ChangeKind.CREATED
    assert change.before_sha is None
    assert change.after_sha
    assert not change.has_snapshot
    assert tracker.changed_paths() == ("new.txt",)


def test_modified_files_keep_the_original_snapshot(tmp_path: Path) -> None:
    write(tmp_path, "app.txt", "old\n")
    tracker = ChangeTracker(tmp_path)
    target = tmp_path / "app.txt"

    tracker.capture(target)
    write(tmp_path, "app.txt", "new\n")
    change = tracker.finalize(target)

    assert change is not None
    assert change.kind is ChangeKind.MODIFIED
    assert change.has_snapshot
    assert Path(change.snapshot).read_text(encoding="utf-8") == "old\n"


def test_repeated_writes_still_roll_back_to_the_first_snapshot(
    tmp_path: Path,
) -> None:
    write(tmp_path, "app.txt", "original\n")
    tracker = ChangeTracker(tmp_path)
    target = tmp_path / "app.txt"

    tracker.capture(target)
    write(tmp_path, "app.txt", "first edit\n")
    tracker.finalize(target)
    tracker.capture(target)
    write(tmp_path, "app.txt", "second edit\n")
    tracker.finalize(target)

    assert Path(tracker.changes[0].snapshot).read_text(encoding="utf-8") == "original\n"
    report = tracker.rollback()
    assert report.restored == ("app.txt",)
    assert target.read_text(encoding="utf-8") == "original\n"


def test_deletion_restores_the_original_after_rollback(tmp_path: Path) -> None:
    write(tmp_path, "gone.txt", "keep me\n")
    tracker = ChangeTracker(tmp_path)
    target = tmp_path / "gone.txt"
    tracker.capture(target)
    target.unlink()
    change = tracker.finalize(target, kind=ChangeKind.DELETED)

    assert change is not None
    assert change.kind is ChangeKind.DELETED
    assert tracker.changed_paths() == ("gone.txt",)
    result = tracker.rollback()
    assert result.restored == ("gone.txt",)
    assert target.read_text(encoding="utf-8") == "keep me\n"


def test_record_delete_tracks_a_deletion_but_cannot_restore_it(
    tmp_path: Path,
) -> None:
    write(tmp_path, "gone.txt", "keep me\n")
    tracker = ChangeTracker(tmp_path)
    target = tmp_path / "gone.txt"
    target.unlink()

    change = tracker.record_delete(target)

    assert change is not None
    assert change.kind is ChangeKind.DELETED
    assert tracker.changed_paths() == ("gone.txt",)
    result = tracker.rollback()
    assert result.restored == ()
    assert result.skipped == (("gone.txt", "no snapshot available"),)
    assert not target.exists()


def test_rollback_deletes_only_files_beast_created(tmp_path: Path) -> None:
    tracker = ChangeTracker(tmp_path)
    target = tmp_path / "made.txt"
    tracker.capture(target)
    write(tmp_path, "made.txt", "beast")
    tracker.finalize(target)

    result = tracker.rollback()

    assert result.deleted == ("made.txt",)
    assert not target.exists()
    assert not tracker.changes[0].active


def test_rollback_skips_files_the_user_edited_afterwards(tmp_path: Path) -> None:
    write(tmp_path, "app.txt", "original\n")
    tracker = ChangeTracker(tmp_path)
    target = tmp_path / "app.txt"
    tracker.capture(target)
    write(tmp_path, "app.txt", "beast edit\n")
    tracker.finalize(target)
    write(tmp_path, "app.txt", "user edit\n")

    result = tracker.rollback()

    assert result.restored == ()
    assert result.skipped == (("app.txt", "changed after BEAST wrote it"),)
    assert target.read_text(encoding="utf-8") == "user edit\n"


def test_rollback_skips_created_files_the_user_changed(tmp_path: Path) -> None:
    tracker = ChangeTracker(tmp_path)
    target = tmp_path / "made.txt"
    tracker.capture(target)
    write(tmp_path, "made.txt", "beast")
    tracker.finalize(target)
    write(tmp_path, "made.txt", "user content")

    result = tracker.rollback()

    assert result.deleted == ()
    assert result.skipped[0][0] == "made.txt"
    assert target.exists()


def test_rollback_reports_unknown_and_already_handled_paths(tmp_path: Path) -> None:
    tracker = ChangeTracker(tmp_path)
    result = tracker.rollback(["missing.txt"])
    assert result.skipped == (("missing.txt", "not tracked"),)

    target = tmp_path / "made.txt"
    tracker.capture(target)
    write(tmp_path, "made.txt", "x")
    tracker.finalize(target)
    tracker.rollback()
    second = tracker.rollback()
    assert second.skipped == (("made.txt", "already rolled back"),)
    assert "Nothing to roll back." in ChangeTracker(tmp_path).rollback().render()


def test_rollback_can_select_specific_paths(tmp_path: Path) -> None:
    write(tmp_path, "a.txt", "a\n")
    write(tmp_path, "b.txt", "b\n")
    tracker = ChangeTracker(tmp_path)
    for name in ("a.txt", "b.txt"):
        target = tmp_path / name
        tracker.capture(target)
        write(tmp_path, name, "changed\n")
        tracker.finalize(target)

    result = tracker.rollback(["a.txt"])

    assert result.restored == ("a.txt",)
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "a\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "changed\n"


def test_paths_outside_the_workspace_are_refused(tmp_path: Path) -> None:
    tracker = ChangeTracker(tmp_path / "workspace")

    with pytest.raises(ValueError, match="outside the workspace"):
        tracker.capture(tmp_path / "elsewhere.txt")


def test_reports_commands_approvals_failures_and_summaries(
    tmp_path: Path,
) -> None:
    tracker = ChangeTracker(tmp_path)
    tracker.record_command(
        "pytest", exit_code=1, ok=False, kind="test"
    )
    tracker.record_command(
        "rm thing", exit_code=None, ok=False, rejected=True
    )
    tracker.record_approval(
        tool="write_file", target="a.txt", action="write", approved=False
    )
    tracker.record_failure("the model stopped responding")
    target = tmp_path / "new.txt"
    tracker.capture(target)
    write(tmp_path, "new.txt", "x")
    tracker.finalize(target)

    report = tracker.report()

    assert report.summary_line() == "1 created, 0 modified, 0 deleted"
    rendered = report.render()
    assert "+ new.txt" in rendered
    assert "$ pytest  [exit 1]" in rendered
    assert "rejected" in rendered
    assert "Approvals rejected (1)" in rendered
    assert "the model stopped responding" in rendered
    assert not report.is_empty
    assert ChangeTracker(tmp_path).report().render().startswith(
        "No changes were made."
    )


def test_unified_diff_shows_both_sides(tmp_path: Path) -> None:
    write(tmp_path, "app.txt", "one\ntwo\n")
    tracker = ChangeTracker(tmp_path)
    target = tmp_path / "app.txt"
    tracker.capture(target)
    write(tmp_path, "app.txt", "one\nthree\n")
    tracker.finalize(target)

    diff = tracker.diff("app.txt")

    assert "-two" in diff
    assert "+three" in diff
    assert "a/app.txt" in diff
    assert "No tracked change" in tracker.diff("other.txt")


def test_diff_for_created_file_uses_dev_null(tmp_path: Path) -> None:
    tracker = ChangeTracker(tmp_path)
    target = tmp_path / "new.txt"
    tracker.capture(target)
    write(tmp_path, "new.txt", "hello\n")
    tracker.finalize(target)

    diff = tracker.diff("new.txt")

    assert "/dev/null" in diff
    assert "+hello" in diff


# ---------------------------------------------------------- tool integration


async def test_write_tool_records_the_change_and_approval(tmp_path: Path) -> None:
    tracker = ChangeTracker(tmp_path)
    approver = Yes()
    context = ToolContext(
        workspace=Workspace(tmp_path), approver=approver, changes=tracker
    )

    result = await WriteFileTool(context).run(
        {"path": "index.html", "content": "<html></html>"}
    )

    assert result.ok
    assert len(approver.requests) == 1
    report = tracker.report()
    assert report.files[0].path == "index.html"
    assert report.files[0].kind is ChangeKind.CREATED
    assert report.approvals[0].approved is True
    assert report.approvals[0].tool == "write_file"


async def test_rejected_write_is_recorded_as_a_rejected_approval(
    tmp_path: Path,
) -> None:
    tracker = ChangeTracker(tmp_path)
    context = ToolContext(
        workspace=Workspace(tmp_path), approver=Yes(answer=False), changes=tracker
    )

    result = await WriteFileTool(context).run(
        {"path": "index.html", "content": "x"}
    )

    assert result.rejected
    assert not (tmp_path / "index.html").exists()
    assert tracker.changes == ()
    assert tracker.report().approvals[0].approved is False


async def test_command_tool_records_exit_code(tmp_path: Path) -> None:
    tracker = ChangeTracker(tmp_path)
    context = ToolContext(
        workspace=Workspace(tmp_path), approver=Yes(), changes=tracker
    )

    result = await RunCommandTool(context).run({"command": "echo hello"})

    assert result.ok
    assert result.exit_code == 0
    records = tracker.report().commands
    assert records[0].command == "echo hello"
    assert records[0].exit_code == 0
    assert records[0].ok is True


# ------------------------------------------------------------------------ git


class FakeProcess:
    def __init__(self, returncode: int = 0, stdout: bytes = b"", stderr: bytes = b""):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr

    def kill(self) -> None:
        pass

    async def wait(self) -> int:
        return self.returncode


async def test_git_inspector_runs_only_read_only_queries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, ...]] = []

    async def fake_exec(*args: Any, **kwargs: Any) -> FakeProcess:
        calls.append(tuple(str(part) for part in args))
        if "status" in args:
            return FakeProcess(stdout=b"?? new.txt\n M app.py\n")
        return FakeProcess(stdout=b"main\n")

    monkeypatch.setattr(
        "beast.core.changes.asyncio.create_subprocess_exec", fake_exec
    )

    status = await GitInspector(tmp_path).status()

    assert status.available
    assert status.branch == "main"
    assert status.changes == ("?? new.txt", " M app.py")
    assert "main" in status.render()
    assert "2 changed path(s)" in status.render()
    executed = [call[:2] for call in calls]
    assert executed == [("git", "--no-pager"), ("git", "--no-pager")]
    assert not any(
        forbidden in args
        for call in calls
        for args in call
        for forbidden in ("commit", "push", "reset", "checkout", "clean")
    )


async def test_git_inspector_reports_unavailable_repositories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_exec(*args: Any, **kwargs: Any) -> FakeProcess:
        return FakeProcess(returncode=128, stderr=b"not a git repository")

    monkeypatch.setattr(
        "beast.core.changes.asyncio.create_subprocess_exec", fake_exec
    )

    status = await GitInspector(tmp_path).status()

    assert not status.available
    assert "not a git repository" in status.error
    assert "not available" in status.render()


async def test_git_inspector_handles_missing_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken_exec(*args: Any, **kwargs: Any) -> FakeProcess:
        raise FileNotFoundError("git is not installed")

    monkeypatch.setattr(
        "beast.core.changes.asyncio.create_subprocess_exec", broken_exec
    )

    status = await GitInspector(tmp_path).status()

    assert not status.available
    assert "could not start" in status.error


async def test_git_diff_is_read_only_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_exec(*args: Any, **kwargs: Any) -> FakeProcess:
        return FakeProcess(stdout=b"--- a/app.py\n+++ b/app.py\n+new\n")

    monkeypatch.setattr(
        "beast.core.changes.asyncio.create_subprocess_exec", fake_exec
    )

    diff = await GitInspector(tmp_path).diff("app.py")

    assert "+new" in diff


def test_git_status_rendering() -> None:
    assert "clean" in GitStatus(available=True, branch="main").render()
    assert "not available" in GitStatus(available=False).render()
