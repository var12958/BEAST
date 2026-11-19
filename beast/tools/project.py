"""The ``project_info`` tool: structured project intelligence for the model.

This tool exposes the same evidence the automatic context engine uses, but on
demand and at the model's chosen granularity.  It is read-only and requires no
approval; every path is resolved inside the workspace and credential-bearing
files are refused.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from beast.tools.base import Permission, Tool, ToolResult
from beast.tools.schemas import ToolArgs
from beast.workspace.secrets import is_sensitive_path, redact_secrets


class ProjectInfoArgs(ToolArgs):
    """Arguments for :class:`ProjectInfoTool`."""

    action: Literal[
        "overview",
        "structure",
        "dependencies",
        "symbols",
        "file_symbols",
        "search",
        "refresh",
    ] = Field(
        default="overview",
        description=(
            "overview: languages, frameworks, entry points and dependencies; "
            "structure: directory tree and key directories; "
            "dependencies: declared dependencies from manifests; "
            "symbols: indexed classes/functions (optionally filtered by query); "
            "file_symbols: symbols and imports for one file (requires path); "
            "search: find indexed symbols by name; refresh: re-scan changed files."
        ),
    )
    path: str | None = Field(
        default=None,
        description="File path for file_symbols, relative to the workspace.",
    )
    query: str | None = Field(
        default=None,
        description="Search terms for symbols/search, e.g. 'authentication login'.",
    )
    limit: int = Field(
        default=40,
        ge=1,
        le=200,
        description="Maximum entries to return.",
    )


class ProjectInfoTool(Tool):
    """Read-only access to the project index and analyses."""

    name = "project_info"
    description = (
        "Inspect the project's indexed structure, symbols and dependencies without "
        "reading whole files. Use it to locate relevant code (search), list symbols "
        "in a file, or answer questions about technologies and layout. Read-only."
    )
    permission = Permission.READ
    args_model = ProjectInfoArgs

    async def execute(self, args: BaseModel) -> ToolResult:
        assert isinstance(args, ProjectInfoArgs)
        intelligence = self.context.intelligence
        if intelligence is None:
            return ToolResult.failure(
                "Project intelligence is not available in this session.",
                content=(
                    "Project intelligence is not available, so project_info cannot "
                    "answer. Use list_files, search_files and read_file instead."
                ),
            )

        if args.action == "refresh":
            update = intelligence.refresh(force=True)
            return ToolResult.success(
                "Project index refreshed.",
                content=(
                    "Project index refreshed: "
                    f"{update.total} files indexed, {update.added} added, "
                    f"{update.modified} modified, {update.removed} removed, "
                    f"{update.analyzed} analysed, {update.unchanged} unchanged."
                    + (" The scan was truncated by configured limits." if update.truncated else "")
                ),
            )

        if args.action == "overview":
            text = intelligence.overview()
        elif args.action == "structure":
            text = self._structure()
        elif args.action == "dependencies":
            text = self._dependencies(args.limit)
        elif args.action == "symbols":
            text = self._symbols(args.query, args.limit)
        elif args.action == "file_symbols":
            if not args.path:
                return ToolResult.failure(
                    "file_symbols requires a path.",
                    content=(
                        "Call project_info with action='file_symbols' and a path "
                        "relative to the workspace."
                    ),
                )
            text = self._file_symbols(args.path)
            if text is None:
                return ToolResult.failure(
                    "That file is credential-bearing and is not analysed.",
                    content=(
                        "BEAST does not expose credential-bearing files such as "
                        ".env, private keys or token stores."
                    ),
                )
        elif args.action == "search":
            if not args.query:
                return ToolResult.failure(
                    "search requires a query.",
                    content="Call project_info with action='search' and a query.",
                )
            text = self._search(args.query, args.limit)
        else:  # pragma: no cover - the Literal validates this
            return ToolResult.failure("Unknown action.")

        text = text.strip()
        if not text:
            text = "No project information is available."
        # Symbol signatures and docstrings come from the index; redact them the
        # same way file reads and automatic context are redacted.
        text = redact_secrets(text).text
        limit = self.context.max_output_chars
        if len(text) > limit:
            text = text[: limit - 40].rstrip() + "\n… (output truncated)"
        return ToolResult.success(
            f"Project {args.action} returned.", content=text
        )

    # ------------------------------------------------------------- renderers

    def _structure(self) -> str:
        intelligence = self.context.intelligence
        assert intelligence is not None
        structure = intelligence.structure()
        detection = intelligence.detection()
        lines = [
            f"Project: {detection.name}",
            f"Source directories: {', '.join(structure.source_dirs) or '(root)'}",
            f"Test directories: {', '.join(structure.test_dirs) or 'none detected'}",
            f"Documentation: {', '.join(structure.docs) or 'none detected'}",
            f"Entry points: {', '.join(detection.entry_points) or 'none detected'}",
            "",
            structure.render_tree(max_lines=90),
        ]
        if structure.key_files:
            lines.extend(["", "Key files: " + ", ".join(structure.key_files)])
        if structure.truncated:
            lines.append("\nThe tree is truncated at the configured depth/file limits.")
        return "\n".join(lines)

    def _dependencies(self, limit: int) -> str:
        intelligence = self.context.intelligence
        assert intelligence is not None
        report = intelligence.dependencies()
        lines = [
            "Package managers: "
            + (", ".join(report.package_managers) or "none detected"),
            "Manifests: "
            + (
                ", ".join(
                    f"{manifest.path} ({manifest.kind})"
                    for manifest in report.manifests
                )
                or "none detected"
            ),
            "",
            report.render(limit=limit),
        ]
        if report.warnings:
            lines.extend(["", "Warnings:", *[f"- {warning}" for warning in report.warnings]])
        return "\n".join(lines)

    def _symbols(self, query: str | None, limit: int) -> str:
        intelligence = self.context.intelligence
        assert intelligence is not None
        if query:
            return self._search(query, limit)

        entries = intelligence.symbols()
        if not entries:
            return "No symbols are indexed (only Python and JavaScript/TypeScript are parsed)."
        lines = [f"Indexed symbols (showing {min(len(entries), limit)} of {len(entries)}):"]
        lines.extend(
            f"- {relative}:{symbol.line} — "
            f"{symbol.kind.value} {symbol.display}"
            for relative, symbol in entries[:limit]
        )
        return "\n".join(lines)

    def _file_symbols(self, path: str) -> str | None:
        assert self.context.intelligence is not None
        target = self.context.workspace.resolve(path)
        relative = self.relative(target)
        if is_sensitive_path(relative):
            return None
        if not target.is_file():
            return f"{relative} is not a file."
        indexed = self.context.intelligence.index.files.get(relative)
        analysis = self.context.intelligence.file_analysis(relative)
        if analysis is None:
            reason = indexed.skipped if indexed else "not indexed"
            if indexed is not None and not indexed.is_source:
                reason = "not a recognised source file"
            return f"No symbol analysis is available for {relative} ({reason})."
        lines = [
            f"{relative} — {analysis.language}, {analysis.lines} lines"
            + ("" if analysis.ok else f" (parse error: {analysis.error})"),
        ]
        if analysis.imports:
            lines.append("Imports:")
            lines.extend(f"- {imported.display()} (line {imported.line})" for imported in analysis.imports[:40])
        if analysis.symbols:
            lines.append("Symbols:")
            lines.extend(
                f"- {symbol.kind.value} {symbol.display} "
                f"(lines {symbol.line}-{symbol.end_line})"
                + (f" — {symbol.doc}" if symbol.doc else "")
                for symbol in analysis.symbols
            )
        else:
            lines.append("No declarations found.")
        return "\n".join(lines)

    def _search(self, query: str, limit: int) -> str:
        assert self.context.intelligence is not None
        hits = self.context.intelligence.search(query, limit=limit)
        if not hits:
            return f"No indexed symbols match {query!r}."
        lines = [f"Symbols matching {query!r}:"]
        lines.extend(
            f"- {hit.path}:{hit.symbol.line} — "
            f"{hit.symbol.kind.value} {hit.symbol.display}"
            + (f" — {hit.symbol.doc}" if hit.symbol.doc else "")
            for hit in hits
        )
        return "\n".join(lines)
