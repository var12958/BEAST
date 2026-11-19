"""Workspace confinement, credential refusal and index invalidation."""

from __future__ import annotations

import asyncio
from pathlib import Path

from beast.intelligence import ProjectIntelligence
from beast.tools import ApprovalRequest, ToolContext
from beast.tools.filesystem import (
    EditFileTool,
    ListFilesTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)
from beast.tools.project import ProjectInfoTool
from beast.workspace import Workspace


class Approver:
    def __init__(self, answer: bool = True) -> None:
        self.answer = answer
        self.requests: list[ApprovalRequest] = []

    async def approve(self, request: ApprovalRequest) -> bool:
        self.requests.append(request)
        return self.answer


def context_for(root: Path, intelligence=None) -> ToolContext:
    return ToolContext(
        workspace=Workspace(root),
        approver=Approver(),
        intelligence=intelligence,
    )


def run(tool, arguments):
    return asyncio.run(tool.run(arguments))


def test_read_refuses_env_files_but_allows_templates(flask_project: Path) -> None:
    context = context_for(flask_project)
    (flask_project / ".env.example").write_text(
        "FLASK_SECRET_KEY=replace-me\n", encoding="utf-8"
    )

    refused = run(ReadFileTool(context), {"path": ".env"})
    assert refused.ok is False
    assert "credential-bearing" in refused.summary
    assert "not-for-the-model" not in refused.content

    allowed = run(ReadFileTool(context), {"path": ".env.example"})
    assert allowed.ok is True
    assert "FLASK_SECRET_KEY" in allowed.content


def test_read_redacts_likely_secrets(flask_project: Path) -> None:
    (flask_project / "config.py").write_text(
        'AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"\n', encoding="utf-8"
    )

    result = run(ReadFileTool(context_for(flask_project)), {"path": "config.py"})

    assert result.ok is True
    assert "AKIAIOSFODNN7EXAMPLE" not in result.content
    assert "[REDACTED]" in result.content
    assert "secret(s) redacted" in result.summary


def test_search_skips_credential_files_and_redacts_matches(
    flask_project: Path,
) -> None:
    (flask_project / ".env").write_text(
        "SUPER_SECRET_TOKEN=abc12345\n", encoding="utf-8"
    )
    (flask_project / "notes.txt").write_text(
        "SUPER_SECRET_TOKEN appears in documentation\n", encoding="utf-8"
    )
    (flask_project / "aws.txt").write_text(
        "key AKIAIOSFODNN7EXAMPLE here\n", encoding="utf-8"
    )
    tool = SearchFilesTool(context_for(flask_project))

    found = run(tool, {"query": "SUPER_SECRET_TOKEN"})
    assert found.ok is True
    assert "notes.txt" in found.content
    assert ".env" not in found.content

    redacted = run(tool, {"query": "AKIA"})
    assert redacted.ok is True
    assert "aws.txt" in redacted.content
    assert "AKIAIOSFODNN7EXAMPLE" not in redacted.content
    assert "[REDACTED]" in redacted.content
    assert "secret(s) redacted" in redacted.summary


def test_every_file_tool_rejects_path_traversal(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("do not read\n", encoding="utf-8")
    context = context_for(root)

    calls = [
        (ListFilesTool, {"path": "../"}),
        (ReadFileTool, {"path": "../outside.txt"}),
        (WriteFileTool, {"path": "../escape.txt", "content": "x"}),
        (
            EditFileTool,
            {"path": "../outside.txt", "old_text": "do not", "new_text": "did"},
        ),
        (SearchFilesTool, {"query": "do not", "path": "../"}),
    ]

    for tool_type, arguments in calls:
        result = run(tool_type(context), arguments)
        assert result.ok is False, tool_type.name
        assert "outside the workspace" in result.summary, tool_type.name

    assert outside.read_text(encoding="utf-8") == "do not read\n"
    assert not (tmp_path / "escape.txt").exists()


def test_project_info_refuses_sensitive_and_escaping_paths(
    flask_project: Path, tmp_path: Path
) -> None:
    intelligence = ProjectIntelligence(flask_project)
    try:
        tool = ProjectInfoTool(context_for(flask_project, intelligence))

        sensitive = run(
            tool, {"action": "file_symbols", "path": ".env"}
        )
        assert sensitive.ok is False
        assert "credential-bearing" in sensitive.summary

        escaping = run(
            tool, {"action": "file_symbols", "path": "../outside.py"}
        )
        assert escaping.ok is False
        assert "outside the workspace" in escaping.summary
    finally:
        intelligence.close()


def test_project_info_without_intelligence_fails_cleanly(flask_project: Path) -> None:
    result = run(ProjectInfoTool(context_for(flask_project)), {"action": "overview"})

    assert result.ok is False
    assert "not available" in result.summary


def test_write_and_edit_invalidate_the_project_index(write_files) -> None:
    root = write_files({"a.py": "x = 1\n"})
    intelligence = ProjectIntelligence(root)
    try:
        intelligence.index  # build the initial index
        assert "b.py" not in intelligence.index.files

        written = run(
            WriteFileTool(context_for(root, intelligence)),
            {"path": "b.py", "content": "b = 2\n"},
        )
        assert written.ok is True
        update = intelligence.refresh(force=False)
        assert update.added == 1
        assert "b.py" in intelligence.index.files

        edited = run(
            EditFileTool(context_for(root, intelligence)),
            {"path": "a.py", "old_text": "x = 1", "new_text": "x = 222"},
        )
        assert edited.ok is True
        update = intelligence.refresh(force=False)
        assert update.modified == 1
        assert intelligence.file_analysis("a.py") is not None
    finally:
        intelligence.close()
