"""The command tool.

Commands run through the platform shell, inside the workspace directory, with
stdin closed, captured output and a hard timeout.  Every command requires the
user's explicit approval, and the approval prompt shows the exact command.

This is **not** a sandbox.  A command the user approves can do anything that
user's account can do, including leaving the workspace.  The workspace check
applies to BEAST's file tools, not to arbitrary programs.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess

from pydantic import Field

from beast.tools.base import Permission, Tool, ToolResult
from beast.tools.schemas import ToolArgs

logger = logging.getLogger(__name__)

#: Bytes kept from each output stream; the remainder is drained and discarded.
OUTPUT_BYTES = 64_000


class RunCommandArgs(ToolArgs):
    command: str = Field(min_length=1, description="The shell command to run.")
    timeout: float = Field(
        default=60.0,
        gt=0,
        le=600,
        description="Seconds to wait before terminating the command.",
    )


def _process_kwargs() -> dict[str, object]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


async def _read_limited(stream: asyncio.StreamReader, limit: int) -> bytes:
    """Read at most ``limit`` bytes, then drain the rest so the process ends."""
    kept = bytearray()
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            break
        if len(kept) < limit:
            kept.extend(chunk[: limit - len(kept)])
    return bytes(kept)


async def _terminate(process: asyncio.subprocess.Process) -> None:
    """Kill the shell and, where possible, its children."""
    if process.returncode is not None:
        return
    try:
        if os.name == "nt":
            await asyncio.to_thread(
                subprocess.run,
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                check=False,
            )
            process.kill()
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.kill()
        except ProcessLookupError:  # pragma: no cover - already gone
            pass
    await process.wait()


class RunCommandTool(Tool):
    """Run one shell command in the project directory, after approval."""

    name = "run_command"
    description = (
        "Run a shell command in the project directory and return stdout, "
        "stderr and the exit code. Use it to run builds, tests or formatters. "
        "The user must approve every command. Commands cannot wait for input."
    )
    permission = Permission.EXECUTE
    args_model = RunCommandArgs

    async def execute(self, args: RunCommandArgs) -> ToolResult:  # type: ignore[override]
        timeout = min(args.timeout, self.context.command_timeout)
        workspace = self.context.workspace
        if not await self.request_approval(
            target=args.command,
            action="Run this command in the workspace",
            detail=f"Working directory: {workspace.display()}",
        ):
            self.context.record_command(
                args.command,
                exit_code=None,
                ok=False,
                rejected=True,
            )
            return ToolResult.failure(
                "The user rejected this command.",
                content=(
                    "The user rejected this command and nothing was run. "
                    "Do not retry it without asking the user."
                ),
                rejected=True,
            )

        try:
            # The command went through explicit approval and is meant to be a
            # shell command, so shell semantics are intended here. Passing the
            # string (rather than an argv list) also avoids Windows' cmd.exe
            # quoting pitfalls with paths that contain spaces.
            process = await asyncio.create_subprocess_shell(
                args.command,
                cwd=str(workspace.root),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **_process_kwargs(),  # type: ignore[arg-type]
            )
        except (OSError, ValueError) as exc:
            self.context.record_command(
                args.command, exit_code=None, ok=False
            )
            return ToolResult.failure(
                f"Could not start the command: {exc}",
                content=f"The command could not be started: {exc}",
            )

        stdout_task = asyncio.create_task(
            _read_limited(process.stdout, OUTPUT_BYTES)  # type: ignore[arg-type]
        )
        stderr_task = asyncio.create_task(
            _read_limited(process.stderr, OUTPUT_BYTES)  # type: ignore[arg-type]
        )

        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            timed_out = True
            await _terminate(process)
        except asyncio.CancelledError:
            # Esc must not leave a shell running in the background.
            await _terminate(process)
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            raise

        stdout, stderr = await asyncio.gather(stdout_task, stderr_task)
        if timed_out:
            self.context.record_command(
                args.command, exit_code=None, ok=False, timed_out=True
            )
            return ToolResult.failure(
                f"Command timed out after {timeout:.0f}s and was terminated.",
                content=(
                    f"The command did not finish within {timeout:.0f} seconds and "
                    "was terminated. Try a faster command or a longer timeout."
                ),
                timed_out=True,
            )

        code = process.returncode
        text = self._format(code, stdout, stderr)
        summary = f"Exited with code {code}"
        self.context.record_command(
            args.command, exit_code=code, ok=(code == 0)
        )
        return ToolResult(
            ok=True,
            summary=summary,
            content=text,
            exit_code=code,
        )

    def _format(self, code: int | None, stdout: bytes, stderr: bytes) -> str:
        out = stdout.decode("utf-8", errors="replace").rstrip()
        err = stderr.decode("utf-8", errors="replace").rstrip()
        parts = [f"Exit code: {code}"]
        parts.append("--- stdout ---")
        parts.append(out or "(no output)")
        parts.append("--- stderr ---")
        parts.append(err or "(no output)")
        text = "\n".join(parts)
        limit = self.context.max_output_chars
        if len(text) > limit:
            text = f"{text[:limit]}\n… (output truncated at {limit} characters)"
        return text
