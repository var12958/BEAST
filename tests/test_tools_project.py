"""The ``project_info`` tool: every read-only action returns indexed evidence."""

from __future__ import annotations

import asyncio
from pathlib import Path

from beast.intelligence import ProjectIntelligence
from beast.tools import ApprovalRequest, ToolContext, ToolResult
from beast.tools.project import ProjectInfoTool
from beast.workspace import Workspace


class Approver:
    async def approve(self, request: ApprovalRequest) -> bool:
        return True


def tool_for(root: Path, *, intelligence=None, max_output_chars: int = 30_000):
    owned = intelligence if intelligence is not None else ProjectIntelligence(root)
    tool = ProjectInfoTool(
        ToolContext(
            workspace=Workspace(root),
            approver=Approver(),
            max_output_chars=max_output_chars,
            intelligence=owned,
        )
    )
    return tool, owned


def project_info(root: Path, action: str, **arguments) -> ToolResult:
    tool, intelligence = tool_for(root)
    try:
        return asyncio.run(tool.run({"action": action, **arguments}))
    finally:
        intelligence.close()


# ----------------------------------------------------------------- overview


def test_overview_uses_the_detected_project_facts(flask_project: Path) -> None:
    result = project_info(flask_project, "overview")

    assert result.ok is True
    assert result.summary == "Project overview returned."
    assert "Primary language: Python" in result.content
    assert "Frameworks: Flask" in result.content
    assert "Flask 3.0.0" in result.content
    assert "Structure:" in result.content
    assert "app.py" in result.content


# ---------------------------------------------------------------- structure


def test_structure_lists_directories_key_files_and_the_tree(
    flask_project: Path,
) -> None:
    result = project_info(flask_project, "structure")

    assert result.ok is True
    assert "Source directories: ." in result.content
    assert "Test directories: tests" in result.content
    assert "Documentation: README.md" in result.content
    assert "Entry points: app.py" in result.content
    assert "tests/" in result.content
    assert "test_auth.py" in result.content
    assert "Key files:" in result.content


# ------------------------------------------------------------- dependencies


def test_dependencies_reports_managers_scopes_and_versions(
    flask_project: Path,
) -> None:
    result = project_info(flask_project, "dependencies")

    assert result.ok is True
    assert "Package managers: pip" in result.content
    assert "requirements.txt (requirements)" in result.content
    assert "Flask 3.0.0" in result.content
    assert "pytest 8.0.0" in result.content


def test_dependencies_respects_the_result_limit(flask_project: Path) -> None:
    result = project_info(flask_project, "dependencies", limit=1)

    assert result.ok is True
    assert "Flask 3.0.0" in result.content
    assert "pytest 8.0.0" not in result.content


# ----------------------------------------------------------------- symbols


def test_symbols_lists_indexed_declarations(flask_project: Path) -> None:
    result = project_info(flask_project, "symbols")

    assert result.ok is True
    assert "Indexed symbols" in result.content
    assert "auth.py:" in result.content
    assert "authenticate" in result.content


def test_symbols_can_be_filtered_by_query(flask_project: Path) -> None:
    result = project_info(flask_project, "symbols", query="authenticate")

    assert result.ok is True
    assert "Symbols matching 'authenticate'" in result.content
    assert "auth.py" in result.content


def test_symbols_explains_an_empty_match(flask_project: Path) -> None:
    result = project_info(flask_project, "symbols", query="kwyjibo")

    assert result.ok is True
    assert "No indexed symbols match 'kwyjibo'." in result.content


def test_symbols_explains_an_unparsed_project(write_files) -> None:
    root = write_files({"README.md": "# notes\n"})

    result = project_info(root, "symbols")

    assert result.ok is True
    assert "No symbols are indexed" in result.content


# ------------------------------------------------------------------ search


def test_search_finds_symbols_by_name(flask_project: Path) -> None:
    result = project_info(flask_project, "search", query="authenticate")

    assert result.ok is True
    assert "auth.py" in result.content
    assert "authenticate" in result.content

    missing = project_info(flask_project, "search", query="kwyjibo")
    assert missing.ok is True
    assert "No indexed symbols match" in missing.content


def test_search_requires_a_query(flask_project: Path) -> None:
    result = project_info(flask_project, "search")

    assert result.ok is False
    assert "requires a query" in result.summary


# ------------------------------------------------------------- file symbols


def test_file_symbols_returns_imports_and_declarations(flask_project: Path) -> None:
    result = project_info(flask_project, "file_symbols", path="auth.py")

    assert result.ok is True
    assert "auth.py — Python" in result.content
    assert "Imports:" in result.content
    assert "from functools import wraps" in result.content
    assert "Symbols:" in result.content
    assert "authenticate" in result.content


def test_file_symbols_requires_a_path(flask_project: Path) -> None:
    result = project_info(flask_project, "file_symbols")

    assert result.ok is False
    assert "requires a path" in result.summary


def test_file_symbols_reports_a_source_file_without_declarations(
    flask_project: Path,
) -> None:
    (flask_project / "constants.py").write_text(
        "# Placeholder for values that were never added.\n", encoding="utf-8"
    )

    result = project_info(flask_project, "file_symbols", path="constants.py")

    assert result.ok is True
    assert "constants.py — Python" in result.content
    assert "No declarations found." in result.content


def test_file_symbols_explains_missing_and_unparsed_files(
    flask_project: Path,
) -> None:
    missing = project_info(flask_project, "file_symbols", path="missing.py")
    assert missing.ok is True
    assert "missing.py is not a file." in missing.content

    not_source = project_info(flask_project, "file_symbols", path="README.md")
    assert not_source.ok is True
    assert "not a recognised source file" in not_source.content

    (flask_project / "script.rb").write_text("puts 'hi'\n", encoding="utf-8")
    unparsed = project_info(flask_project, "file_symbols", path="script.rb")
    assert unparsed.ok is True
    assert "no analyser" in unparsed.content


# ----------------------------------------------------------------- refresh


def test_refresh_reports_incremental_changes(flask_project: Path) -> None:
    tool, intelligence = tool_for(flask_project)
    try:
        first = asyncio.run(tool.run({"action": "overview"}))
        assert first.ok is True

        (flask_project / "new_module.py").write_text(
            "def added():\n    return 1\n", encoding="utf-8"
        )
        result = asyncio.run(tool.run({"action": "refresh"}))

        assert result.ok is True
        assert result.summary == "Project index refreshed."
        assert "7 files indexed" in result.content
        assert "1 added" in result.content
        assert "1 analysed" in result.content
    finally:
        intelligence.close()


# --------------------------------------------------------------- truncation


def test_large_output_is_truncated(flask_project: Path) -> None:
    tool, intelligence = tool_for(flask_project, max_output_chars=120)
    try:
        result = asyncio.run(tool.run({"action": "overview"}))
    finally:
        intelligence.close()

    assert result.ok is True
    assert len(result.content) <= 140
    assert result.content.endswith("… (output truncated)")
