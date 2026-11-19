"""Change tracking, unified diffs and safe rollback.

BEAST records every file it changes, every command it runs and every approval
decision.  The original bytes of a file are copied to a private backup
directory *after* the user approves a change but *before* the write, so the
snapshot is always the user's pre-BEAST content.

Rollback is deliberately conservative:

* it restores a file only when the current content still matches the hash BEAST
  wrote — if you edited the file afterwards, BEAST refuses to touch it;
* it never runs ``git reset``, ``git checkout``, ``git clean`` or any other
  blanket revert;
* it only touches paths BEAST itself tracked.

Git is used read-only for reporting (status, diffs) when a repository exists.
Nothing in this module commits, pushes or discards work.
"""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import logging
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)

#: Files larger than this are not snapshotted (rollback refuses them).
MAX_SNAPSHOT_BYTES = 8_000_000
#: Cap for one rendered diff.
MAX_DIFF_CHARS = 24_000
MAX_DIFF_LINES = 400


class ChangeKind(StrEnum):
    """How BEAST changed a file."""

    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass(frozen=True, slots=True)
class FileChange:
    """One tracked file, from its pre-BEAST snapshot to its last BEAST write."""

    path: str
    kind: ChangeKind
    before_sha: str | None
    after_sha: str | None
    snapshot: str = ""
    timestamp: float = 0.0
    rolled_back: bool = False

    @property
    def has_snapshot(self) -> bool:
        return bool(self.snapshot)

    @property
    def active(self) -> bool:
        return not self.rolled_back


@dataclass(frozen=True, slots=True)
class CommandRecord:
    """One approved command BEAST executed."""

    command: str
    exit_code: int | None
    ok: bool
    rejected: bool = False
    timed_out: bool = False
    kind: str = "command"
    timestamp: float = 0.0


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    """One approval decision made by the user."""

    tool: str
    target: str
    action: str
    approved: bool
    timestamp: float = 0.0


@dataclass(frozen=True, slots=True)
class ChangeReport:
    """Everything BEAST did during a session or autonomous run."""

    files: tuple[FileChange, ...] = ()
    commands: tuple[CommandRecord, ...] = ()
    approvals: tuple[ApprovalRecord, ...] = ()
    failures: tuple[str, ...] = ()
    backup_dir: str = ""

    @property
    def active_files(self) -> tuple[FileChange, ...]:
        return tuple(change for change in self.files if change.active)

    @property
    def created(self) -> tuple[FileChange, ...]:
        return tuple(
            change
            for change in self.active_files
            if change.kind is ChangeKind.CREATED
        )

    @property
    def modified(self) -> tuple[FileChange, ...]:
        return tuple(
            change
            for change in self.active_files
            if change.kind is ChangeKind.MODIFIED
        )

    @property
    def deleted(self) -> tuple[FileChange, ...]:
        return tuple(
            change
            for change in self.active_files
            if change.kind is ChangeKind.DELETED
        )

    @property
    def is_empty(self) -> bool:
        return not (self.files or self.commands or self.approvals)

    def summary_line(self) -> str:
        """One line for progress panels and run summaries."""
        parts = [
            f"{len(self.created)} created",
            f"{len(self.modified)} modified",
            f"{len(self.deleted)} deleted",
        ]
        rolled = sum(1 for change in self.files if change.rolled_back)
        if rolled:
            parts.append(f"{rolled} rolled back")
        return ", ".join(parts)

    def render(self, *, include_commands: bool = True) -> str:
        """Human-readable summary for the transcript and final report."""
        lines: list[str] = []
        if self.created:
            lines.append(f"Created ({len(self.created)}):")
            lines.extend(f"  + {change.path}" for change in self.created)
        if self.modified:
            lines.append(f"Modified ({len(self.modified)}):")
            lines.extend(f"  ~ {change.path}" for change in self.modified)
        if self.deleted:
            lines.append(f"Deleted ({len(self.deleted)}):")
            lines.extend(f"  - {change.path}" for change in self.deleted)
        rolled = [change for change in self.files if change.rolled_back]
        if rolled:
            lines.append(f"Rolled back ({len(rolled)}):")
            lines.extend(f"  ↩ {change.path}" for change in rolled)
        if include_commands and self.commands:
            lines.append(f"Commands ({len(self.commands)}):")
            for record in self.commands:
                state = (
                    "rejected"
                    if record.rejected
                    else "timed out"
                    if record.timed_out
                    else f"exit {record.exit_code}"
                )
                lines.append(f"  $ {record.command}  [{state}]")
        rejected = [item for item in self.approvals if not item.approved]
        if rejected:
            lines.append(f"Approvals rejected ({len(rejected)}):")
            lines.extend(
                f"  ⊘ {item.action}: {item.target}" for item in rejected
            )
        if self.failures:
            lines.append("Failed operations:")
            lines.extend(f"  ! {failure}" for failure in self.failures)
        if not lines:
            lines.append("No changes were made.")
        if self.backup_dir:
            lines.append(f"Snapshots: {self.backup_dir}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class RollbackReport:
    """The outcome of a rollback request."""

    restored: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()
    skipped: tuple[tuple[str, str], ...] = ()

    @property
    def changed(self) -> int:
        return len(self.restored) + len(self.deleted)

    def render(self) -> str:
        lines: list[str] = []
        for path in self.restored:
            lines.append(f"restored  {path}")
        for path in self.deleted:
            lines.append(f"deleted   {path}")
        for path, reason in self.skipped:
            lines.append(f"skipped   {path}: {reason}")
        if not lines:
            lines.append("Nothing to roll back.")
        return "\n".join(lines)


@dataclass(slots=True)
class _Original:
    """The pre-BEAST state of one path (kept across repeated writes)."""

    snapshot: Path | None
    sha: str | None
    missing: bool
    too_large: bool = False


@dataclass
class ChangeTracker:
    """Records BEAST-attributed changes and can reverse them."""

    root: Path
    backup_dir: Path | None = None
    clock: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        if self.backup_dir is None:
            self.backup_dir = Path(
                tempfile.mkdtemp(prefix="beast-changes-")
            )
        else:
            self.backup_dir = Path(self.backup_dir)
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._originals: dict[str, _Original] = {}
        self._pending: dict[str, _Original] = {}
        self._changes: dict[str, FileChange] = {}
        self._order: list[str] = []
        self._commands: list[CommandRecord] = []
        self._approvals: list[ApprovalRecord] = []
        self._failures: list[str] = []
        self._counter = 0

    # -------------------------------------------------------------- recording

    def capture(self, path: Path | str) -> None:
        """Snapshot ``path`` before BEAST writes to it.

        Calling this again for the same path keeps the *first* snapshot, so
        several writes in one session still roll back to the user's original.
        """
        relative = self._relative(path)
        if relative in self._originals:
            self._pending[relative] = self._originals[relative]
            return
        target = self.root / relative
        original = self._read_original(relative, target)
        self._originals[relative] = original
        self._pending[relative] = original

    def finalize(
        self, path: Path | str, *, kind: ChangeKind | None = None
    ) -> FileChange | None:
        """Record the state of ``path`` after BEAST wrote or deleted it."""
        relative = self._relative(path)
        pending = self._pending.pop(relative, None)
        if pending is None:
            # A write that was not captured cannot be attributed safely.
            logger.info("change for %s was not captured; not tracked", relative)
            return None

        target = self.root / relative
        exists = target.exists()
        after_sha = _sha_file(target) if exists else None
        if kind is None:
            kind = (
                ChangeKind.MODIFIED if not pending.missing else ChangeKind.CREATED
            )
        elif kind is ChangeKind.DELETED:
            after_sha = None

        previous = self._changes.get(relative)
        change = FileChange(
            path=relative,
            kind=previous.kind if previous is not None else kind,
            before_sha=pending.sha,
            after_sha=after_sha,
            snapshot=str(pending.snapshot) if pending.snapshot else "",
            timestamp=float(self.clock()),  # type: ignore[operator]
        )
        if previous is None:
            self._order.append(relative)
        self._changes[relative] = change
        return change

    def record_delete(self, path: Path | str) -> FileChange | None:
        """Track a deletion performed outside the file tools."""
        relative = self._relative(path)
        self.capture(relative)
        return self.finalize(relative, kind=ChangeKind.DELETED)

    def record_command(
        self,
        command: str,
        *,
        exit_code: int | None,
        ok: bool,
        rejected: bool = False,
        timed_out: bool = False,
        kind: str = "command",
    ) -> None:
        self._commands.append(
            CommandRecord(
                command=command,
                exit_code=exit_code,
                ok=ok,
                rejected=rejected,
                timed_out=timed_out,
                kind=kind,
                timestamp=float(self.clock()),  # type: ignore[operator]
            )
        )

    def record_approval(
        self,
        *,
        tool: str,
        target: str,
        action: str,
        approved: bool,
    ) -> None:
        self._approvals.append(
            ApprovalRecord(
                tool=tool,
                target=target,
                action=action,
                approved=approved,
                timestamp=float(self.clock()),  # type: ignore[operator]
            )
        )

    def record_failure(self, description: str) -> None:
        self._failures.append(description)

    # ------------------------------------------------------------------ reading

    @property
    def changes(self) -> tuple[FileChange, ...]:
        return tuple(
            self._changes[relative]
            for relative in self._order
            if relative in self._changes
        )

    def changed_paths(self) -> tuple[str, ...]:
        return tuple(change.path for change in self.changes if change.active)

    def report(self) -> ChangeReport:
        return ChangeReport(
            files=self.changes,
            commands=tuple(self._commands),
            approvals=tuple(self._approvals),
            failures=tuple(self._failures),
            backup_dir=str(self.backup_dir),
        )

    def diff(self, relative: str, *, max_lines: int = MAX_DIFF_LINES) -> str:
        """A unified diff of BEAST's change to ``relative``."""
        change = self._changes.get(relative)
        if change is None:
            return f"No tracked change for {relative}."
        before = b""
        if not change.has_snapshot and change.kind is not ChangeKind.CREATED:
            return (
                f"No snapshot is available for {relative} "
                "(the file was too large or was not captured)."
            )
        if change.has_snapshot:
            try:
                before = Path(change.snapshot).read_bytes()
            except OSError:
                before = b""
        target = self.root / relative
        after = target.read_bytes() if target.exists() else b""

        before_text = before.decode("utf-8", errors="replace")
        after_text = after.decode("utf-8", errors="replace")
        fromfile = "/dev/null" if change.before_sha is None else f"a/{relative}"
        tofile = "/dev/null" if not target.exists() else f"b/{relative}"
        lines = list(
            difflib.unified_diff(
                before_text.splitlines(),
                after_text.splitlines(),
                fromfile=fromfile,
                tofile=tofile,
                lineterm="",
            )
        )
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines.append(f"… (diff truncated at {max_lines} lines)")
        text = "\n".join(lines)
        if len(text) > MAX_DIFF_CHARS:
            text = f"{text[:MAX_DIFF_CHARS]}\n… (diff truncated)"
        return text or f"{relative} has no textual difference."

    # ----------------------------------------------------------------- rollback

    def rollback(
        self, paths: list[str] | tuple[str, ...] | None = None
    ) -> RollbackReport:
        """Restore BEAST's changes, skipping anything that changed afterwards."""
        selected = (
            list(paths)
            if paths is not None
            else [relative for relative in reversed(self._order)]
        )
        restored: list[str] = []
        deleted: list[str] = []
        skipped: list[tuple[str, str]] = []

        for relative in selected:
            change = self._changes.get(relative)
            if change is None:
                skipped.append((relative, "not tracked"))
                continue
            if not change.active:
                skipped.append((relative, "already rolled back"))
                continue
            target = self.root / relative
            current_sha = _sha_file(target) if target.exists() else None

            if change.kind is ChangeKind.CREATED:
                if current_sha is None:
                    skipped.append((relative, "already absent"))
                    continue
                if current_sha != change.after_sha:
                    skipped.append(
                        (relative, "changed after BEAST wrote it")
                    )
                    continue
                try:
                    target.unlink()
                except OSError as exc:
                    skipped.append((relative, f"delete failed: {exc}"))
                    continue
                deleted.append(relative)
                self._mark_rolled_back(relative, change)
                continue

            if change.kind is ChangeKind.MODIFIED:
                if current_sha is None:
                    skipped.append((relative, "file deleted after BEAST wrote it"))
                    continue
                if current_sha != change.after_sha:
                    skipped.append(
                        (relative, "changed after BEAST wrote it")
                    )
                    continue
                original = self._originals.get(relative)
                if original is None or original.snapshot is None:
                    skipped.append((relative, "no snapshot available"))
                    continue
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(original.snapshot.read_bytes())
                except OSError as exc:
                    skipped.append((relative, f"restore failed: {exc}"))
                    continue
                restored.append(relative)
                self._mark_rolled_back(relative, change)
                continue

            # DELETED
            if current_sha is not None:
                skipped.append((relative, "a file now exists at this path"))
                continue
            original = self._originals.get(relative)
            if original is None or original.snapshot is None:
                skipped.append((relative, "no snapshot available"))
                continue
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(original.snapshot.read_bytes())
            except OSError as exc:
                skipped.append((relative, f"restore failed: {exc}"))
                continue
            restored.append(relative)
            self._mark_rolled_back(relative, change)

        return RollbackReport(
            restored=tuple(restored),
            deleted=tuple(deleted),
            skipped=tuple(skipped),
        )

    def _mark_rolled_back(self, relative: str, change: FileChange) -> None:
        self._changes[relative] = replace(change, rolled_back=True)

    # ------------------------------------------------------------------- housekeeping

    def discard(self) -> None:
        """Delete the snapshot directory (rollback is no longer possible)."""
        shutil.rmtree(self.backup_dir, ignore_errors=True)

    # ------------------------------------------------------------------- helpers

    def _read_original(self, relative: str, target: Path) -> _Original:
        if not target.exists():
            return _Original(None, None, missing=True)
        try:
            size = target.stat().st_size
        except OSError:
            return _Original(None, None, missing=False, too_large=True)
        if size > MAX_SNAPSHOT_BYTES:
            return _Original(None, None, missing=False, too_large=True)
        try:
            data = target.read_bytes()
        except OSError:  # pragma: no cover - permission race
            return _Original(None, None, missing=False, too_large=True)
        self._counter += 1
        snapshot = self.backup_dir / f"{self._counter:04d}.bin"
        try:
            snapshot.write_bytes(data)
        except OSError:  # pragma: no cover - disk full / read-only cache
            return _Original(None, _sha_bytes(data), missing=False)
        return _Original(snapshot, _sha_bytes(data), missing=False)

    def _relative(self, path: Path | str) -> str:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(
                f"{resolved} is outside the workspace {self.root}."
            ) from exc
        return relative.as_posix()


# ---------------------------------------------------------------------- hashing


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(131_072), b""):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------- git status


@dataclass(frozen=True, slots=True)
class GitStatus:
    """Read-only information about the surrounding Git repository."""

    available: bool
    branch: str = ""
    changes: tuple[str, ...] = ()
    error: str = ""

    def render(self, limit: int = 50) -> str:
        if not self.available:
            return f"Git: not available ({self.error or 'no repository'})."
        lines = [f"Git branch: {self.branch or '(detached)'}"]
        if not self.changes:
            lines.append("Working tree: clean")
        else:
            lines.append(f"Working tree: {len(self.changes)} changed path(s)")
            lines.extend(f"  {entry}" for entry in self.changes[:limit])
            if len(self.changes) > limit:
                lines.append(f"  … {len(self.changes) - limit} more")
        return "\n".join(lines)


class GitInspector:
    """Runs read-only Git queries inside the workspace.

    No command is accepted from the model or the user: only the fixed,
    read-only queries implemented here are ever executed.  This never commits,
    pushes, resets, checks out or cleans.
    """

    def __init__(self, root: Path | str, *, timeout: float = 8.0) -> None:
        self.root = Path(root).resolve()
        self.timeout = timeout

    async def status(self) -> GitStatus:
        code, output, error = await self._run("status", "--porcelain=v1")
        if code != 0:
            return GitStatus(
                available=False, error=(error or output).strip()[:200]
            )
        _, branch, _ = await self._run("branch", "--show-current")
        changes = tuple(
            line for line in output.splitlines() if line.strip()
        )
        return GitStatus(
            available=True,
            branch=branch.strip(),
            changes=changes,
        )

    async def diff(self, relative: str) -> str:
        """A unified diff for one tracked path, from Git itself."""
        code, output, error = await self._run(
            "diff", "--", relative
        )
        if code != 0:
            return f"git diff failed: {(error or output).strip()[:400]}"
        if not output.strip():
            return (
                f"No Git diff for {relative} "
                "(untracked or unchanged in the index)."
            )
        if len(output) > MAX_DIFF_CHARS:
            output = f"{output[:MAX_DIFF_CHARS]}\n… (diff truncated)"
        return output

    async def _run(self, *args: str) -> tuple[int, str, str]:
        try:
            environment = os.environ.copy()
            environment["GIT_TERMINAL_PROMPT"] = "0"
            environment["GIT_OPTIONAL_LOCKS"] = "0"
            process = await asyncio.create_subprocess_exec(
                "git",
                "--no-pager",
                "-c",
                "core.quotepath=false",
                *args,
                cwd=str(self.root),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=environment,
            )
        except (OSError, ValueError) as exc:
            return 127, "", f"git could not start: {exc}"

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return 124, "", "git timed out"
        return (
            process.returncode or 0,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )
