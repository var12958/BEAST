"""Workspace file tools: list, read, write, edit and search.

Every path goes through :meth:`beast.workspace.Workspace.resolve`, so the model
cannot reach outside the project it was pointed at.  Generated and tool-owned
directories (``.git``, ``node_modules``, virtual environments, caches) are both
hidden from exploration and refused as write targets.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import Field

from beast.tools.base import Permission, Tool, ToolResult
from beast.tools.schemas import ToolArgs
from beast.workspace.scanning import (
    IGNORED_DIRS_LOWER,
    ScanLimits,
    is_junction,
    looks_binary,
    scan_project,
)
from beast.workspace.secrets import is_sensitive_path, redact_secrets

#: Safety valve for pathological trees during a search.
MAX_VISITED = 20_000


def _is_ignored_name(name: str) -> bool:
    return name.lower() in IGNORED_DIRS_LOWER


def _protected_component(path: Path, root: Path) -> str | None:
    """Return the first ignored component in ``path``, if any."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    return next(
        (
            part
            for part in relative.parts
            if part.lower() in IGNORED_DIRS_LOWER
        ),
        None,
    )


def _read_prefix(path: Path, limit: int) -> tuple[bytes, bool]:
    """Read at most ``limit`` bytes, never loading the whole file.

    Returns the bytes read and whether the file is larger than that.
    """
    size = path.stat().st_size
    with path.open("rb") as handle:
        data = handle.read(limit)
    return data, size > limit


def _read_text(path: Path, limit: int) -> str | None:
    """Read up to ``limit`` bytes of text, or None when it looks binary."""
    data, _ = _read_prefix(path, limit)
    if looks_binary(data):
        return None
    return data.decode("utf-8", errors="replace")


def _preview(text: str, limit: int = 600) -> str:
    stripped = text.rstrip()
    if len(stripped) <= limit:
        return stripped
    return f"{stripped[:limit]}\n… (truncated)"


def _lines(text: str) -> int:
    return len(text.splitlines())


def _normalise_extensions(values: list[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        cleaned = value.strip().lower().lstrip(".")
        if cleaned:
            result.append(f".{cleaned}")
    return tuple(result)


def _walk_files(
    root: Path,
    *,
    max_depth: int | None = None,
    max_visited: int = MAX_VISITED,
) -> Iterator[Path]:
    """Yield files under ``root`` using the shared, gitignore-aware walker."""
    scan = scan_project(
        root,
        limits=ScanLimits(
            max_files=max_visited,
            max_depth=max_depth or 30,
            respect_gitignore=True,
        ),
    )
    for scanned in scan.files:
        yield scanned.path


# ------------------------------------------------------------------ list_files


class ListFilesArgs(ToolArgs):
    path: str = Field(default=".", description="Directory relative to the project root.")
    recursive: bool = Field(default=False, description="Descend into sub-directories.")
    max_depth: int = Field(
        default=3,
        ge=1,
        le=20,
        description="How many directory levels to descend when recursive.",
    )
    limit: int = Field(
        default=300,
        ge=1,
        le=2000,
        description="Stop after this many entries.",
    )


class ListFilesTool(Tool):
    """List files and directories inside the project."""

    name = "list_files"
    description = (
        "List files and directories in the project. Use it to explore the "
        "repository before reading or changing files. Generated directories "
        "such as .git, node_modules and virtual environments are skipped."
    )
    permission = Permission.READ
    args_model = ListFilesArgs

    async def execute(self, args: ListFilesArgs) -> ToolResult:  # type: ignore[override]
        root = self.context.workspace.resolve(args.path)
        if not root.exists():
            return ToolResult.failure(f"Directory not found: {args.path}")
        if not root.is_dir():
            return ToolResult.failure(f"Not a directory: {args.path}")

        entries: list[str] = []
        truncated = False

        def walk(directory: Path, depth: int) -> None:
            nonlocal truncated
            try:
                children = sorted(
                    directory.iterdir(),
                    key=lambda item: (not item.is_dir(), item.name.lower()),
                )
            except OSError:
                return
            for entry in children:
                if len(entries) >= args.limit:
                    truncated = True
                    return
                relative = entry.relative_to(self.context.workspace.root).as_posix()
                if is_sensitive_path(relative):
                    continue
                if entry.is_symlink():
                    entries.append(f"{relative}@ (symlink, not followed)")
                    continue
                if is_junction(entry):
                    continue
                if entry.is_dir():
                    if _is_ignored_name(entry.name):
                        continue
                    entries.append(f"{relative}/")
                    if args.recursive and depth < args.max_depth:
                        walk(entry, depth + 1)
                else:
                    entries.append(relative)

        walk(root, 0)
        if not entries:
            return ToolResult.success(
                f"No files under {args.path}.",
                content="(empty directory)",
            )

        summary = f"{len(entries)} entries under {args.path}"
        content = "\n".join(entries)
        if truncated:
            summary += f" (stopped at {args.limit})"
            content += f"\n… (stopped at {args.limit} entries)"
        return ToolResult.success(summary, content=content)


# ------------------------------------------------------------------ read_file


class ReadFileArgs(ToolArgs):
    path: str = Field(description="File path relative to the project root.")
    max_bytes: int = Field(
        default=200_000,
        ge=1,
        le=5_000_000,
        description="Maximum bytes to read from the start of the file.",
    )


class ReadFileTool(Tool):
    """Read a text file from the project."""

    name = "read_file"
    description = (
        "Read a text file from the project. Always read a file before editing "
        "it. Binary files are refused, and very large files are truncated."
    )
    permission = Permission.READ
    args_model = ReadFileArgs

    async def execute(self, args: ReadFileArgs) -> ToolResult:  # type: ignore[override]
        path = self.context.workspace.resolve(args.path)
        relative = self.relative(path)
        if is_sensitive_path(relative):
            return ToolResult.failure(
                f"Refusing to read credential-bearing file: {relative}",
                content=(
                    "BEAST does not expose environment files, private keys or "
                    "credential stores. Use configuration documentation or an "
                    "example/template file instead."
                ),
            )
        if not path.exists():
            return ToolResult.failure(f"File not found: {args.path}")
        if path.is_dir():
            return ToolResult.failure(
                f"{args.path} is a directory. Use list_files instead."
            )

        limit = min(args.max_bytes, self.context.max_read_bytes)
        data, truncated = _read_prefix(path, limit)
        if looks_binary(data):
            return ToolResult.failure(
                f"{args.path} looks like a binary file and cannot be read as text."
            )

        redacted = redact_secrets(data.decode("utf-8", errors="replace"))
        text = redacted.text
        summary = f"Read {relative} ({_lines(text)} lines)"
        if truncated:
            text += f"\n\n… (truncated at {limit} bytes)"
            summary += f", truncated at {limit} bytes"
        if redacted.count:
            summary += f", {redacted.count} secret(s) redacted"
            text += (
                f"\n\n… ({redacted.count} likely secret value(s) were redacted "
                "before showing this file)"
            )
        return ToolResult.success(summary, content=text)


# ----------------------------------------------------------------- write_file


class WriteFileArgs(ToolArgs):
    path: str = Field(description="File path relative to the project root.")
    content: str = Field(description="The complete file contents to write.")


class WriteFileTool(Tool):
    """Create a file, or replace an existing file after approval."""

    name = "write_file"
    description = (
        "Create a new file or replace an existing one with complete contents. "
        "The user must approve every write. Provide the whole file: this tool "
        "does not append."
    )
    permission = Permission.WRITE
    args_model = WriteFileArgs

    async def execute(self, args: WriteFileArgs) -> ToolResult:  # type: ignore[override]
        target = self.context.workspace.resolve(args.path)
        if target.exists() and target.is_dir():
            return ToolResult.failure(f"{args.path} is a directory, not a file.")

        blocked = _protected_component(target, self.context.workspace.root)
        if blocked is not None:
            return ToolResult.failure(
                f"Refusing to write inside '{blocked}': that directory is generated or tool-owned."
            )

        relative = self.relative(target)
        exists = target.exists()
        action = "Overwrite existing file" if exists else "Create new file"
        detail = f"{relative}\n\n{_preview(args.content)}"
        if not await self.request_approval(
            target=relative, action=action, detail=detail
        ):
            return ToolResult.failure(
                "The user rejected this file change.",
                content=(
                    "The user rejected this file change and nothing was written. "
                    "Do not retry the same change without asking the user."
                ),
                rejected=True,
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        # Snapshot any previous contents before the first BEAST write so the
        # change can be shown as a diff and rolled back precisely.
        self.context.capture_file(target)
        target.write_bytes(args.content.encode("utf-8"))
        self.context.finalize_file(target)
        verb = "Updated" if exists else "Created"
        summary = f"{verb} {relative} ({_lines(args.content)} lines)"
        return ToolResult.success(
            summary,
            content=f"{summary}. The file is now on disk.",
        )


# ------------------------------------------------------------------ edit_file


class EditFileArgs(ToolArgs):
    path: str = Field(description="File path relative to the project root.")
    old_text: str = Field(
        min_length=1,
        description="Exact text to replace. It must occur exactly once.",
    )
    new_text: str = Field(description="Replacement text.")


class EditFileTool(Tool):
    """Replace one exact passage in an existing file, with approval."""

    name = "edit_file"
    description = (
        "Replace one exact passage in an existing file. old_text must occur "
        "exactly once; include enough surrounding context to make it unique. "
        "The rest of the file is preserved. The user must approve the change."
    )
    permission = Permission.WRITE
    args_model = EditFileArgs

    async def execute(self, args: EditFileArgs) -> ToolResult:  # type: ignore[override]
        target = self.context.workspace.resolve(args.path)

        blocked = _protected_component(target, self.context.workspace.root)
        if blocked is not None:
            return ToolResult.failure(
                f"Refusing to edit inside '{blocked}': that directory is generated or tool-owned."
            )

        if not target.exists():
            return ToolResult.failure(
                f"File not found: {args.path}. Use write_file to create it."
            )
        if target.is_dir():
            return ToolResult.failure(f"{args.path} is a directory, not a file.")

        data = target.read_bytes()
        if b"\x00" in data[:8192]:
            return ToolResult.failure(f"{args.path} is a binary file and cannot be edited.")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            # Rewriting through a lossy decode would replace every invalid byte
            # in the untouched remainder with U+FFFD.
            return ToolResult.failure(
                f"{args.path} is not UTF-8 encoded and cannot be edited safely.",
                content=(
                    f"{args.path} contains bytes that are not valid UTF-8. "
                    "Editing it would corrupt the parts of the file that are "
                    "not being changed. Convert it to UTF-8 first, or edit it "
                    "with a dedicated tool."
                ),
            )

        occurrences = text.count(args.old_text)
        if occurrences == 0:
            return ToolResult.failure(
                f"Could not find that text in {args.path}.",
                content=(
                    f"old_text was not found in {args.path}. Read the file again "
                    "and copy the exact text, including indentation."
                ),
            )
        if occurrences > 1:
            return ToolResult.failure(
                f"That text occurs {occurrences} times in {args.path}.",
                content=(
                    f"old_text occurs {occurrences} times in {args.path}, so the "
                    "edit is ambiguous. Include more surrounding lines to make it unique."
                ),
            )

        relative = self.relative(target)
        detail = (
            f"{relative}\n\n"
            f"- {_preview(args.old_text, 300)}\n"
            f"+ {_preview(args.new_text, 300)}"
        )
        if not await self.request_approval(
            target=relative, action="Apply this edit", detail=detail
        ):
            return ToolResult.failure(
                "The user rejected this edit.",
                content=(
                    "The user rejected this edit and the file is unchanged. "
                    "Do not retry the same edit without asking the user."
                ),
                rejected=True,
            )

        updated = text.replace(args.old_text, args.new_text, 1)
        self.context.capture_file(target)
        target.write_bytes(updated.encode("utf-8"))
        self.context.finalize_file(target)
        delta = _lines(args.new_text) - _lines(args.old_text)
        change = "no line count change" if delta == 0 else f"{delta:+d} lines"
        summary = f"Edited {relative} ({change})"
        return ToolResult.success(
            summary,
            content=f"{summary}. The change is on disk.",
        )


# ---------------------------------------------------------------- search_files


class SearchFilesArgs(ToolArgs):
    query: str = Field(min_length=1, description="Text or filename fragment to find.")
    path: str = Field(default=".", description="Directory to search, relative to the project.")
    by: Literal["content", "filename"] = Field(
        default="content",
        description="Search inside file contents or in file names.",
    )
    extensions: list[str] = Field(
        default_factory=list,
        description="Limit to these extensions, e.g. ['py', 'js']. Empty means all text files.",
    )
    max_results: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Stop after this many matches.",
    )
    case_sensitive: bool = Field(default=False, description="Match case-sensitively.")


class SearchFilesTool(Tool):
    """Find files by name or text, returning paths and line numbers."""

    name = "search_files"
    description = (
        "Search the project for a filename fragment or for text inside files. "
        "Returns paths and line numbers. Use it to locate code before reading it."
    )
    permission = Permission.READ
    args_model = SearchFilesArgs

    async def execute(self, args: SearchFilesArgs) -> ToolResult:  # type: ignore[override]
        root = self.context.workspace.resolve(args.path)
        if not root.exists():
            return ToolResult.failure(f"Path not found: {args.path}")

        needle = args.query if args.case_sensitive else args.query.lower()
        extensions = _normalise_extensions(args.extensions)
        matches: list[str] = []
        truncated = False
        skipped_binary = 0
        skipped_unreadable = 0
        redactions = 0

        files = [root] if root.is_file() else _walk_files(root)
        for file in files:
            relative = self.relative(file)
            if is_sensitive_path(relative):
                continue
            if extensions and file.suffix.lower() not in extensions:
                continue
            if args.by == "filename":
                haystack = file.name if args.case_sensitive else file.name.lower()
                if needle in haystack:
                    matches.append(relative)
            else:
                try:
                    text = _read_text(file, self.context.max_read_bytes)
                except OSError:
                    # A single unreadable file must not abort the whole search.
                    skipped_unreadable += 1
                    continue
                if text is None:
                    skipped_binary += 1
                    continue
                for number, line in enumerate(text.splitlines(), start=1):
                    haystack = line if args.case_sensitive else line.lower()
                    if needle not in haystack:
                        continue
                    # Redact the whole line first: truncating before redaction
                    # could hide the closing quote a pattern needs to match.
                    snippet = redact_secrets(line.strip())
                    redactions += snippet.count
                    matches.append(f"{relative}:{number}: {snippet.text[:240]}")
                    break
            if len(matches) >= args.max_results:
                truncated = True
                break

        noun = "match" if len(matches) == 1 else "matches"
        if not matches:
            content = f"No matches for {args.query!r}."
            if skipped_binary:
                content += " Binary files were skipped."
            if skipped_unreadable:
                content += f" {skipped_unreadable} file(s) could not be read."
            return ToolResult.success(f"No {noun} for {args.query!r}.", content=content)

        summary = f"{len(matches)} {noun} for {args.query!r}"
        content = "\n".join(matches)
        if truncated:
            summary += f" (stopped at {args.max_results})"
            content += f"\n… (stopped at {args.max_results} matches)"
        if redactions:
            summary += f" ({redactions} secret(s) redacted)"
            content += f"\n… ({redactions} likely secret value(s) redacted)"
        return ToolResult.success(summary, content=content)
