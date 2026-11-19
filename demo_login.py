"""Headless Phase 2 demonstration: the login-page task.

This drives the real agent loop against the local Ollama model.  File writes
are approved automatically so the demo can run unattended; shell commands are
refused, because there is no human at the keyboard to judge them.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from beast.core.agent import Agent
from beast.core.events import ApprovalPending, TextDelta, ToolFinished, ToolStarted
from beast.ollama.client import OllamaClient
from beast.tools import Permission, ToolContext, ToolRegistry
from beast.workspace import Workspace

MODEL = "qwen3:8b"
PROMPT = "Create a simple login page using HTML, CSS and JavaScript."

USAGE = "usage: python demo_login.py <workspace> [model]"


class DemoApprover:
    """Prints every request; approves writes, refuses commands."""

    async def approve(self, request) -> bool:
        print(
            f"\n[APPROVAL] {request.tool} ({request.permission}) "
            f"target={request.target!r}",
            flush=True,
        )
        print(f"  action: {request.action}", flush=True)
        for line in request.detail.splitlines()[:10]:
            print(f"  | {line}", flush=True)
        approved = request.permission is not Permission.EXECUTE
        print(
            f"  -> {'approved' if approved else 'REJECTED (headless: no shell)'}",
            flush=True,
        )
        return approved


async def main() -> int:
    if len(sys.argv) < 2:
        print(USAGE)
        return 2
    workspace_path = Path(sys.argv[1])
    model = sys.argv[2] if len(sys.argv) > 2 else MODEL
    print(f"model: {model}\nprompt: {PROMPT}\n", flush=True)
    workspace_path.mkdir(parents=True, exist_ok=True)
    workspace = Workspace(workspace_path)

    context = ToolContext(workspace=workspace, approver=DemoApprover())
    registry = ToolRegistry.default(context)

    async with OllamaClient() as client:
        agent = Agent(client, model, tools=registry)
        async for event in agent.run(PROMPT):
            if isinstance(event, TextDelta):
                print(event.text, end="", flush=True)
            elif isinstance(event, ToolStarted):
                print(
                    f"\n[TOOL] {event.name} {event.arguments}", flush=True
                )
            elif isinstance(event, ApprovalPending):
                pass
            elif isinstance(event, ToolFinished):
                status = (
                    "OK"
                    if event.result.ok
                    else ("REJECTED" if event.result.rejected else "FAILED")
                )
                print(f"[{status}] {event.result.summary}", flush=True)

    files = sorted(
        path.relative_to(workspace_path).as_posix()
        for path in workspace_path.rglob("*")
        if path.is_file()
    )
    print("\n\nFILES ON DISK:")
    for name in files:
        print(f"  {name}")
    for message in agent.conversation:
        if message.role.value == "tool":
            print(f"\n[OBSERVATION {message.tool_name}] {message.content[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
