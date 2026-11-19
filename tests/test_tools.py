"""The tool contract: schemas, validation, permissions and the registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import Field

from beast.errors import BeastError
from beast.tools import (
    BUILTIN_TOOLS,
    ApprovalRequest,
    Permission,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
)
from beast.tools.schemas import ToolArgs, tool_schema
from beast.workspace import Workspace


class Recorder:
    """An approver that records requests and answers with a fixed value."""

    def __init__(self, answer: Any = True) -> None:
        self.answer = answer
        self.requests: list[ApprovalRequest] = []

    async def approve(self, request: ApprovalRequest) -> Any:
        self.requests.append(request)
        return self.answer


def context(tmp_path: Path, approver: Recorder | None = None) -> ToolContext:
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=approver or Recorder(),  # type: ignore[arg-type]
    )


class EchoArgs(ToolArgs):
    text: str = Field(min_length=1, description="Text to echo.")


class EchoTool(Tool):
    name = "echo"
    description = "Echo text back."
    permission = Permission.READ
    args_model = EchoArgs

    def __init__(self, ctx: ToolContext, *, explode: bool = False) -> None:
        super().__init__(ctx)
        self.calls = 0
        self.explode = explode

    async def execute(self, args: EchoArgs) -> ToolResult:  # type: ignore[override]
        self.calls += 1
        if self.explode:
            raise RuntimeError("boom")
        return ToolResult.success("echoed", content=args.text)


class WriteishTool(EchoTool):
    name = "writeish"
    description = "A tool that needs approval."
    permission = Permission.WRITE


# ------------------------------------------------------------------- definitions


def test_definition_matches_the_ollama_tools_payload() -> None:
    definition = EchoTool.definition()

    assert definition["type"] == "function"
    function = definition["function"]
    assert function["name"] == "echo"
    assert function["description"] == "Echo text back."
    assert function["parameters"]["type"] == "object"
    assert function["parameters"]["properties"]["text"]["type"] == "string"
    assert "title" not in function["parameters"]


def test_schema_rejects_unknown_keys() -> None:
    schema = tool_schema(EchoArgs)

    assert schema["additionalProperties"] is False


def test_registry_exposes_tools_in_registration_order(tmp_path: Path) -> None:
    registry = ToolRegistry.default(context(tmp_path))

    assert registry.names == tuple(tool.name for tool in BUILTIN_TOOLS)
    assert set(registry.names) == {
        "list_files",
        "read_file",
        "search_files",
        "write_file",
        "edit_file",
        "run_command",
    }
    definitions = registry.definitions
    assert [entry["function"]["name"] for entry in definitions] == list(registry.names)


def test_registry_rejects_duplicate_names(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    registry = ToolRegistry([EchoTool(ctx)])

    with pytest.raises(ValueError):
        registry.register(EchoTool(ctx))


def test_registry_reports_unknown_tools(tmp_path: Path) -> None:
    registry = ToolRegistry([EchoTool(context(tmp_path))])

    assert registry.get("missing") is None
    assert registry.get("echo") is not None
    assert registry.is_empty is False
    assert ToolRegistry().is_empty is True


# -------------------------------------------------------------------- execution


async def test_run_validates_then_executes(tmp_path: Path) -> None:
    tool = EchoTool(context(tmp_path))

    result = await tool.run({"text": "hello"})

    assert result.ok is True
    assert result.content == "hello"
    assert tool.calls == 1


async def test_invalid_arguments_never_reach_execute(tmp_path: Path) -> None:
    tool = EchoTool(context(tmp_path))

    result = await tool.run({"text": "", "invented": True})

    assert result.ok is False
    assert "Invalid arguments" in result.summary
    assert tool.calls == 0
    assert "match its schema" in result.content


async def test_non_mapping_arguments_are_rejected(tmp_path: Path) -> None:
    tool = EchoTool(context(tmp_path))

    result = await tool.run("not an object")  # type: ignore[arg-type]

    assert result.ok is False
    assert tool.calls == 0


async def test_beast_errors_become_failures(tmp_path: Path) -> None:
    class RefusingTool(EchoTool):
        async def execute(self, args: EchoArgs) -> ToolResult:  # type: ignore[override]
            raise BeastError("Nope.")

    result = await RefusingTool(context(tmp_path)).run({"text": "x"})

    assert result.ok is False
    assert "Nope." in result.summary


async def test_unexpected_errors_never_escape(tmp_path: Path) -> None:
    tool = EchoTool(context(tmp_path), explode=True)

    result = await tool.run({"text": "x"})

    assert result.ok is False
    assert "unexpected error" in result.summary


# ------------------------------------------------------------------- approvals


async def test_read_tools_do_not_ask_for_approval(tmp_path: Path) -> None:
    approver = Recorder(False)
    tool = EchoTool(context(tmp_path, approver))

    await tool.run({"text": "hello"})

    assert approver.requests == []


async def test_request_approval_describes_the_action(tmp_path: Path) -> None:
    approver = Recorder(True)
    tool = WriteishTool(context(tmp_path, approver))

    approved = await tool.request_approval(
        target="notes.txt", action="Create new file", detail="body"
    )

    assert approved is True
    request = approver.requests[0]
    assert request.tool == "writeish"
    assert request.permission is Permission.WRITE
    assert request.target == "notes.txt"
    assert request.action == "Create new file"
    assert request.detail == "body"


@pytest.mark.parametrize("answer", [False, None, 1, "yes", 0])
async def test_only_an_explicit_true_counts_as_approval(
    tmp_path: Path, answer: Any
) -> None:
    tool = WriteishTool(context(tmp_path, Recorder(answer)))

    assert await tool.request_approval(target="x", action="y") is False


async def test_a_broken_approver_is_a_refusal(tmp_path: Path) -> None:
    class Broken:
        async def approve(self, request: ApprovalRequest) -> bool:
            raise RuntimeError("UI gone")

    tool = WriteishTool(context(tmp_path, Broken()))  # type: ignore[arg-type]

    assert await tool.request_approval(target="x", action="y") is False


# --------------------------------------------------------------------- results


def test_result_helpers() -> None:
    assert ToolResult.success("done").ok is True
    failure = ToolResult.failure("no", rejected=True)
    assert failure.ok is False
    assert failure.rejected is True
    assert failure.for_model() == "no"
    assert ToolResult.success("done", content="details").for_model() == "details"
