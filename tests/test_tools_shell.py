"""The command tool: approval, output capture, exit codes and timeouts."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from beast.tools import ApprovalRequest, Permission, ToolContext
from beast.tools.shell import RunCommandTool, _read_limited
from beast.workspace import Workspace


class Approver:
    def __init__(self, answer: bool = True) -> None:
        self.answer = answer
        self.requests: list[ApprovalRequest] = []

    async def approve(self, request: ApprovalRequest) -> bool:
        self.requests.append(request)
        return self.answer


def context(
    tmp_path: Path, answer: bool = True
) -> tuple[ToolContext, Approver]:
    approver = Approver(answer)
    return ToolContext(workspace=Workspace(tmp_path), approver=approver), approver


def python(code: str) -> str:
    """A shell command that runs ``code`` with the current interpreter."""
    return f'"{sys.executable}" -c "{code}"'


async def test_command_runs_in_the_workspace_after_approval(
    tmp_path: Path,
) -> None:
    ctx, approver = context(tmp_path)

    result = await RunCommandTool(ctx).run(
        {"command": python("import os; print(os.getcwd())")}
    )

    assert result.ok is True
    assert result.summary == "Exited with code 0"
    directory = result.content.split("--- stdout ---\n", 1)[1].splitlines()[0]
    assert Path(directory).resolve() == tmp_path.resolve()
    assert approver.requests[0].permission is Permission.EXECUTE
    assert "python" in approver.requests[0].target.lower()
    assert "Working directory:" in approver.requests[0].detail
    assert tmp_path.name in approver.requests[0].detail


async def test_rejected_commands_never_run(tmp_path: Path) -> None:
    marker = tmp_path / "ran.txt"
    ctx, approver = context(tmp_path, answer=False)

    result = await RunCommandTool(ctx).run(
        {"command": python(f"open(r'{marker}', 'w').write('x')")}
    )

    assert result.ok is False
    assert result.rejected is True
    assert "Do not retry" in result.content
    assert not marker.exists()
    assert len(approver.requests) == 1


async def test_non_zero_exit_codes_are_reported(tmp_path: Path) -> None:
    ctx, _ = context(tmp_path)

    result = await RunCommandTool(ctx).run(
        {"command": python("import sys; print('out'); sys.exit(3)")},
    )

    assert result.ok is True  # the command ran; its exit code is the result
    assert "Exit code: 3" in result.content
    assert "out" in result.content
    assert result.summary == "Exited with code 3"


async def test_stderr_is_captured(tmp_path: Path) -> None:
    ctx, _ = context(tmp_path)

    result = await RunCommandTool(ctx).run(
        {"command": python("import sys; sys.stderr.write('problem')")}
    )

    assert result.ok is True
    stderr = result.content.split("--- stderr ---", 1)[1]
    assert "problem" in stderr


async def test_timeouts_terminate_the_command(tmp_path: Path) -> None:
    ctx, _ = context(tmp_path)

    result = await asyncio.wait_for(
        RunCommandTool(ctx).run(
            {"command": python("import time; time.sleep(30)"), "timeout": 0.4}
        ),
        timeout=20,
    )

    assert result.ok is False
    assert "timed out" in result.summary


async def test_the_command_timeout_is_capped_by_the_context(
    tmp_path: Path,
) -> None:
    approver = Approver(True)
    ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=approver,
        command_timeout=0.4,
    )

    result = await asyncio.wait_for(
        RunCommandTool(ctx).run(
            {"command": python("import time; time.sleep(30)"), "timeout": 600}
        ),
        timeout=20,
    )

    assert result.ok is False
    assert "timed out after 0s" in result.summary


async def test_cancelling_the_turn_kills_the_command(tmp_path: Path) -> None:
    ctx, _ = context(tmp_path)
    task = asyncio.create_task(
        RunCommandTool(ctx).run({"command": python("import time; time.sleep(30)")})
    )
    await asyncio.sleep(0.5)

    task.cancel()
    result = await asyncio.gather(task, return_exceptions=True)

    assert isinstance(result[0], asyncio.CancelledError)


async def test_output_is_truncated_to_the_context_limit(tmp_path: Path) -> None:
    approver = Approver(True)
    ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=approver,
        max_output_chars=120,
    )

    result = await RunCommandTool(ctx).run(
        {"command": python("print('x' * 500)")}
    )

    assert result.ok is True
    assert "output truncated at 120 characters" in result.content


async def test_unstartable_commands_become_failures(tmp_path: Path) -> None:
    ctx, _ = context(tmp_path)

    result = await RunCommandTool(ctx).run(
        {"command": "definitely-not-a-real-command-xyz"}
    )

    # cmd.exe/sh report a non-zero status rather than failing to start, which
    # is a normal observation for the model rather than a crash.
    assert result.ok is True
    assert "Exit code" in result.content


async def test_read_limited_drains_and_keeps_the_prefix() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(b"a" * 100 + b"b" * 900)
    reader.feed_eof()

    kept = await _read_limited(reader, 100)

    assert kept == b"a" * 100
