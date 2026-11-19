"""Phase 3 definition-of-done harness.

Runs the two acceptance questions against three generated fixtures
(Python/Flask, React/TypeScript, multi-language) using the real local model
through Ollama.  Every answer is produced from the same agent pipeline the TUI
uses: project intelligence, transient context attachment and the tool loop.

Usage::

    python scripts/phase3_definition_of_done.py --model qwen3:8b

The transcripts are written to ``docs/phase3-dod/`` so the evidence can be
inspected without re-running the models.  Nothing in this script hardcodes a
project description: all facts come from the fixture files on disk.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from beast.core.agent import Agent  # noqa: E402
from beast.core.events import (  # noqa: E402
    AnalysisEvent,
    ApprovalPending,
    ContextReady,
    TextDelta,
    ToolFinished,
    ToolStarted,
)
from beast.errors import BeastError, OllamaProtocolError  # noqa: E402
from beast.intelligence import ProjectIntelligence  # noqa: E402
from beast.ollama import OllamaClient, probe  # noqa: E402
from beast.tools import ApprovalRequest, ToolContext, ToolRegistry  # noqa: E402
from beast.workspace import Workspace  # noqa: E402

OVERVIEW_QUESTION = (
    "Understand this project and explain its architecture, technologies, "
    "important files, dependencies and how the main components interact. "
    "Do not modify anything."
)
AUTH_QUESTION = (
    "Find the authentication implementation in this project and explain how it works."
)

PROJECTS = ("flask", "react", "multilang")


def load_fixtures() -> dict[str, dict[str, str]]:
    """Read the fixture dictionaries from the test suite (single source)."""
    spec = importlib.util.spec_from_file_location(
        "beast_test_fixtures", ROOT / "tests" / "conftest.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise SystemExit("Could not load tests/conftest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        "flask": module.FLASK_FILES,
        "react": module.REACT_FILES,
        "multilang": module.MULTILANG_FILES,
    }


def write_project(files: dict[str, str]) -> Path:
    root = Path(tempfile.mkdtemp(prefix="beast-dod-"))
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return root


class DenyMutatingApprover:
    """Read tools need no approval; writes and commands are refused."""

    def __init__(self, log: Callable[[str], None]) -> None:
        self._log = log

    async def approve(self, request: ApprovalRequest) -> bool:
        self._log(
            f"[DENIED] {request.tool}: {request.action} -> {request.target}"
        )
        return False


async def ask(
    client: OllamaClient,
    model: str,
    workspace: Path,
    intelligence: ProjectIntelligence,
    question: str,
    emit: Callable[[str], None],
) -> str:
    registry = ToolRegistry.default(
        ToolContext(
            workspace=Workspace(workspace),
            approver=DenyMutatingApprover(emit),
            intelligence=intelligence,
        )
    )
    agent = Agent(
        client,
        model,
        tools=registry,
        context_provider=intelligence,
    )

    answer: list[str] = []
    tool_calls: list[str] = []
    async for event in agent.run(question):
        if isinstance(event, TextDelta):
            answer.append(event.text)
        elif isinstance(event, AnalysisEvent):
            emit(f"[ANALYSIS] {event.message}"
                 + (f" ({event.detail})" if event.detail else ""))
        elif isinstance(event, ContextReady):
            files = ", ".join(event.bundle.files) or "(none)"
            emit(
                f"[CONTEXT] {len(event.bundle.files)} file(s), "
                f"{event.bundle.symbol_count} symbols, "
                f"{event.bundle.characters:,} chars: {files}"
            )
        elif isinstance(event, ToolStarted):
            emit(f"[TOOL] {event.name} {json.dumps(event.arguments, default=str)}")
            tool_calls.append(event.name)
        elif isinstance(event, ToolFinished):
            emit(f"[TOOL] {event.name}: {event.result.summary}")
        elif isinstance(event, ApprovalPending):
            emit(f"[APPROVAL] waiting for {event.name}")

    return "".join(answer).strip()


async def ask_with_retry(
    client: OllamaClient,
    model: str,
    workspace: Path,
    intelligence: ProjectIntelligence,
    question: str,
    emit: Callable[[str], None],
) -> str:
    """Run one question, retrying once if the model runner dies while loading."""
    try:
        return await ask(
            client, model, workspace, intelligence, question, emit
        )
    except OllamaProtocolError as exc:
        if "could not start the model" not in str(exc):
            raise
        emit("[RETRY] the model runner failed to start; retrying once")
        await asyncio.sleep(2)
        return await ask(
            client, model, workspace, intelligence, question, emit
        )


async def run_project(
    name: str,
    files: dict[str, str],
    client: OllamaClient,
    model: str,
    out_dir: Path,
) -> Path:
    root = write_project(files)
    cache = Path(tempfile.mkdtemp(prefix="beast-cache-"))
    os.environ["BEAST_CACHE_DIR"] = str(cache)

    transcript: list[str] = [
        f"# {name}",
        "",
        f"Workspace: `{root}`",
        "",
        "## Detected overview",
        "",
        "```",
    ]
    intelligence = ProjectIntelligence(root)
    try:
        overview = intelligence.overview()
        transcript.extend(overview.strip().splitlines())
        transcript.extend(["```", ""])

        log: list[str] = []
        emit = log.append

        for index, question in enumerate((OVERVIEW_QUESTION, AUTH_QUESTION), 1):
            log.append(f"### Question {index}")
            log.append("")
            log.append(f"> {question}")
            log.append("")
            started = time.perf_counter()
            try:
                answer = await ask_with_retry(
                    client, model, root, intelligence, question, emit
                )
            except BeastError as exc:
                elapsed = time.perf_counter() - started
                log.append(f"Model error after {elapsed:.1f}s: {exc.message}")
                if exc.hint:
                    log.append(f"Hint: {exc.hint}")
                if exc.detail:
                    log.append(f"Detail: {exc.detail}")
                log.append("")
                intelligence.refresh(force=True)
                continue
            elapsed = time.perf_counter() - started
            log.append(f"Answer ({elapsed:.1f}s):")
            log.append("")
            log.append(answer or "(no answer)")
            log.append("")
            # The index may have been invalidated by a refused write; refresh
            # so the next question sees the actual files on disk.
            intelligence.refresh(force=True)

        transcript.extend(log)
    finally:
        intelligence.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{name}.md"
    target.write_text("\n".join(transcript) + "\n", encoding="utf-8")
    return target


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--only",
        choices=PROJECTS,
        action="append",
        help="Run only the named project; may be repeated.",
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "docs" / "phase3-dod"
    )
    args = parser.parse_args()

    status = await probe(args.base_url)
    if not status.is_ready:
        detail = status.detail or status.error or "unknown reason"
        print(f"Ollama is not ready: {detail}")
        return 2

    fixtures = load_fixtures()
    selected = args.only or list(PROJECTS)

    async with OllamaClient(args.base_url) as client:
        models = {m.name for m in await client.list_models()}
        if args.model not in models:
            print(f"Model {args.model!r} is not installed. Have: {sorted(models)}")
            return 2

        for name in selected:
            print(f"\n=== {name} ===")
            target = await run_project(
                name, fixtures[name], client, args.model, args.out
            )
            print(f"transcript: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
