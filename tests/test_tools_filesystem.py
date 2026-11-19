"""The file tools, exercised against a real temporary workspace."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from beast.tools import ApprovalRequest, Permission, ToolContext
from beast.tools.filesystem import (
    EditFileTool,
    ListFilesTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)
from beast.workspace import Workspace


class Approver:
    def __init__(self, answer: bool = True) -> None:
        self.answer = answer
        self.requests: list[ApprovalRequest] = []

    async def approve(self, request: ApprovalRequest) -> bool:
        self.requests.append(request)
        return self.answer


def context(tmp_path: Path, answer: bool = True) -> tuple[ToolContext, Approver]:
    approver = Approver(answer)
    return ToolContext(workspace=Workspace(tmp_path), approver=approver), approver


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")
    return path


# ------------------------------------------------------------------ list_files


async def test_list_files_shows_sorted_entries(tmp_path: Path) -> None:
    write(tmp_path / "b.txt", "b")
    write(tmp_path / "a.txt", "a")
    (tmp_path / "src").mkdir()

    result = await ListFilesTool(context(tmp_path)[0]).run({})

    assert result.ok is True
    assert result.content.splitlines() == ["src/", "a.txt", "b.txt"]


async def test_list_files_is_shallow_unless_recursive(tmp_path: Path) -> None:
    write(tmp_path / "src" / "main.py", "print('hi')")

    shallow = await ListFilesTool(context(tmp_path)[0]).run({})
    deep = await ListFilesTool(context(tmp_path)[0]).run(
        {"recursive": True, "max_depth": 2}
    )

    assert "src/" in shallow.content
    assert "src/main.py" not in shallow.content
    assert "src/main.py" in deep.content


async def test_list_files_skips_generated_directories(tmp_path: Path) -> None:
    write(tmp_path / "app.py", "x")
    write(tmp_path / "node_modules" / "pkg" / "index.js", "x")
    write(tmp_path / ".git" / "config", "x")
    write(tmp_path / "__pycache__" / "app.pyc", "x")

    result = await ListFilesTool(context(tmp_path)[0]).run({"recursive": True})

    assert "app.py" in result.content
    assert "node_modules" not in result.content
    assert ".git" not in result.content
    assert "__pycache__" not in result.content


async def test_list_files_reports_a_missing_directory(tmp_path: Path) -> None:
    result = await ListFilesTool(context(tmp_path)[0]).run({"path": "missing"})

    assert result.ok is False
    assert "not found" in result.summary


async def test_list_files_reports_a_file_instead_of_a_directory(tmp_path: Path) -> None:
    write(tmp_path / "a.txt", "a")

    result = await ListFilesTool(context(tmp_path)[0]).run({"path": "a.txt"})

    assert result.ok is False
    assert "Not a directory" in result.summary


async def test_list_files_honours_the_entry_limit(tmp_path: Path) -> None:
    for index in range(5):
        write(tmp_path / f"f{index}.txt", "x")

    result = await ListFilesTool(context(tmp_path)[0]).run({"limit": 2})

    assert "stopped at 2" in result.summary
    assert "(stopped at 2 entries)" in result.content


async def test_list_files_marks_symlinks_without_following_them(
    tmp_path: Path,
) -> None:
    target = write(tmp_path / "target.txt", "x")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")

    result = await ListFilesTool(context(tmp_path)[0]).run({})

    assert "link.txt@ (symlink, not followed)" in result.content


# ------------------------------------------------------------------- read_file


async def test_read_file_returns_content_and_line_count(tmp_path: Path) -> None:
    write(tmp_path / "notes.txt", "one\ntwo\n")

    result = await ReadFileTool(context(tmp_path)[0]).run({"path": "notes.txt"})

    assert result.ok is True
    assert result.content == "one\ntwo\n"
    assert result.summary == "Read notes.txt (2 lines)"


async def test_read_file_reports_missing_and_directory_paths(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()

    missing = await ReadFileTool(context(tmp_path)[0]).run({"path": "nope.txt"})
    directory = await ReadFileTool(context(tmp_path)[0]).run({"path": "src"})

    assert missing.ok is False
    assert "File not found" in missing.summary
    assert directory.ok is False
    assert "list_files" in directory.summary


async def test_read_file_refuses_binary_content(tmp_path: Path) -> None:
    (tmp_path / "image.bin").write_bytes(b"\x00\x01\x02")

    result = await ReadFileTool(context(tmp_path)[0]).run({"path": "image.bin"})

    assert result.ok is False
    assert "binary" in result.summary


async def test_read_file_truncates_to_the_requested_size(tmp_path: Path) -> None:
    write(tmp_path / "big.txt", "x" * 50)

    result = await ReadFileTool(context(tmp_path)[0]).run(
        {"path": "big.txt", "max_bytes": 10}
    )

    assert result.ok is True
    assert result.content.startswith("x" * 10)
    assert "truncated at 10 bytes" in result.content
    assert "truncated" in result.summary


async def test_read_file_reads_only_the_requested_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write(tmp_path / "notes.txt", "hello")

    def _refuse(self: Path) -> bytes:
        raise AssertionError("read_file must not load the whole file")

    monkeypatch.setattr(Path, "read_bytes", _refuse)

    result = await ReadFileTool(context(tmp_path)[0]).run({"path": "notes.txt"})

    assert result.ok is True
    assert result.content == "hello"


# ------------------------------------------------------------------ write_file


async def test_write_file_creates_parents_after_approval(tmp_path: Path) -> None:
    ctx, approver = context(tmp_path)

    result = await WriteFileTool(ctx).run(
        {"path": "src/app.js", "content": "console.log(1)\n"}
    )

    assert result.ok is True
    assert (tmp_path / "src" / "app.js").read_text(encoding="utf-8") == "console.log(1)\n"
    assert approver.requests[0].permission is Permission.WRITE
    assert approver.requests[0].action == "Create new file"
    assert approver.requests[0].target == "src/app.js"
    assert "console.log(1)" in approver.requests[0].detail
    assert result.summary.startswith("Created src/app.js")


async def test_write_file_marks_an_overwrite(tmp_path: Path) -> None:
    write(tmp_path / "app.js", "old")
    ctx, approver = context(tmp_path)

    result = await WriteFileTool(ctx).run({"path": "app.js", "content": "new"})

    assert result.ok is True
    assert (tmp_path / "app.js").read_text(encoding="utf-8") == "new"
    assert approver.requests[0].action == "Overwrite existing file"
    assert result.summary.startswith("Updated app.js")


async def test_rejected_write_leaves_the_file_untouched(tmp_path: Path) -> None:
    write(tmp_path / "app.js", "old")
    ctx, _ = context(tmp_path, answer=False)

    result = await WriteFileTool(ctx).run({"path": "app.js", "content": "new"})

    assert result.ok is False
    assert result.rejected is True
    assert (tmp_path / "app.js").read_text(encoding="utf-8") == "old"
    assert "Do not retry" in result.content


async def test_write_file_refuses_generated_directories(tmp_path: Path) -> None:
    ctx, approver = context(tmp_path)

    result = await WriteFileTool(ctx).run(
        {"path": "node_modules/pkg/index.js", "content": "x"}
    )

    assert result.ok is False
    assert "generated or tool-owned" in result.summary
    assert approver.requests == []
    assert not (tmp_path / "node_modules").exists()


async def test_write_file_refuses_a_directory_target(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    ctx, approver = context(tmp_path)

    result = await WriteFileTool(ctx).run({"path": "src", "content": "x"})

    assert result.ok is False
    assert "directory" in result.summary
    assert approver.requests == []


# ------------------------------------------------------------------- edit_file


async def test_edit_file_replaces_one_exact_passage(tmp_path: Path) -> None:
    write(tmp_path / "app.py", "value = 1\nprint(value)\n")
    ctx, approver = context(tmp_path)

    result = await EditFileTool(ctx).run(
        {"path": "app.py", "old_text": "value = 1", "new_text": "value = 2"}
    )

    assert result.ok is True
    assert "value = 1" not in (tmp_path / "app.py").read_text(encoding="utf-8")
    assert (tmp_path / "app.py").read_text(encoding="utf-8").startswith("value = 2")
    assert approver.requests[0].action == "Apply this edit"
    assert "- value = 1" in approver.requests[0].detail
    assert "+ value = 2" in approver.requests[0].detail


async def test_edit_file_reports_missing_files_and_text(tmp_path: Path) -> None:
    write(tmp_path / "app.py", "hello")

    missing_file = await EditFileTool(context(tmp_path)[0]).run(
        {"path": "nope.py", "old_text": "x", "new_text": "y"}
    )
    missing_text = await EditFileTool(context(tmp_path)[0]).run(
        {"path": "app.py", "old_text": "absent", "new_text": "y"}
    )

    assert missing_file.ok is False
    assert "write_file" in missing_file.summary
    assert missing_text.ok is False
    assert "Could not find" in missing_text.summary


async def test_edit_file_rejects_ambiguous_matches(tmp_path: Path) -> None:
    write(tmp_path / "app.py", "x = 1\nx = 1\n")

    result = await EditFileTool(context(tmp_path)[0]).run(
        {"path": "app.py", "old_text": "x = 1", "new_text": "x = 2"}
    )

    assert result.ok is False
    assert "occurs 2 times" in result.summary
    assert "more surrounding lines" in result.content


async def test_edit_file_requires_approval(tmp_path: Path) -> None:
    write(tmp_path / "app.py", "x = 1\n")
    ctx, _ = context(tmp_path, answer=False)

    result = await EditFileTool(ctx).run(
        {"path": "app.py", "old_text": "x = 1", "new_text": "x = 2"}
    )

    assert result.ok is False
    assert result.rejected is True
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "x = 1\n"


async def test_edit_file_refuses_binary_and_generated_paths(tmp_path: Path) -> None:
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01")
    ctx, approver = context(tmp_path)

    binary = await EditFileTool(ctx).run(
        {"path": "blob.bin", "old_text": "a", "new_text": "b"}
    )
    generated = await EditFileTool(ctx).run(
        {"path": ".git/config", "old_text": "a", "new_text": "b"}
    )

    assert binary.ok is False
    assert "binary" in binary.summary
    assert generated.ok is False
    assert "generated or tool-owned" in generated.summary
    assert approver.requests == []


async def test_edit_file_refuses_non_utf8_files_without_changing_them(
    tmp_path: Path,
) -> None:
    payload = b"# caf\xe9\nvalue = 1\n"
    (tmp_path / "legacy.py").write_bytes(payload)

    result = await EditFileTool(context(tmp_path)[0]).run(
        {"path": "legacy.py", "old_text": "value = 1", "new_text": "value = 2"}
    )

    assert result.ok is False
    assert "UTF-8" in result.summary
    # A lossy decode/encode round trip would have rewritten 0xE9 as U+FFFD.
    assert (tmp_path / "legacy.py").read_bytes() == payload


# ---------------------------------------------------------------- search_files


async def test_search_files_returns_paths_and_line_numbers(tmp_path: Path) -> None:
    write(tmp_path / "app.py", "import os\nprint('needle')\n")
    write(tmp_path / "src" / "lib.js", "// needle\n")

    result = await SearchFilesTool(context(tmp_path)[0]).run({"query": "needle"})

    assert result.ok is True
    assert "app.py:2: print('needle')" in result.content
    assert "src/lib.js:1: // needle" in result.content


async def test_search_files_can_match_file_names(tmp_path: Path) -> None:
    write(tmp_path / "login.html", "<html></html>")
    write(tmp_path / "app.js", "x")

    result = await SearchFilesTool(context(tmp_path)[0]).run(
        {"query": "login", "by": "filename"}
    )

    assert result.ok is True
    assert result.content == "login.html"


async def test_search_files_filters_by_extension(tmp_path: Path) -> None:
    write(tmp_path / "app.py", "needle")
    write(tmp_path / "app.js", "needle")

    result = await SearchFilesTool(context(tmp_path)[0]).run(
        {"query": "needle", "extensions": ["py"]}
    )

    assert "app.py" in result.content
    assert "app.js" not in result.content


async def test_search_files_skips_ignored_directories(tmp_path: Path) -> None:
    write(tmp_path / "app.py", "needle")
    write(tmp_path / "node_modules" / "pkg.js", "needle")

    result = await SearchFilesTool(context(tmp_path)[0]).run({"query": "needle"})

    assert "app.py" in result.content
    assert "node_modules" not in result.content


async def test_search_files_is_case_insensitive_by_default(tmp_path: Path) -> None:
    write(tmp_path / "app.py", "Needle")

    loose = await SearchFilesTool(context(tmp_path)[0]).run({"query": "needle"})
    strict = await SearchFilesTool(context(tmp_path)[0]).run(
        {"query": "needle", "case_sensitive": True}
    )

    assert loose.content == "app.py:1: Needle"
    assert strict.ok is True
    assert "No match" in strict.summary


async def test_search_files_skips_binary_files(tmp_path: Path) -> None:
    (tmp_path / "blob.bin").write_bytes(b"\x00needle")

    result = await SearchFilesTool(context(tmp_path)[0]).run({"query": "needle"})

    assert result.ok is True
    assert "No matches" in result.summary
    assert "Binary files were skipped" in result.content


async def test_search_files_reports_a_missing_path(tmp_path: Path) -> None:
    result = await SearchFilesTool(context(tmp_path)[0]).run(
        {"query": "x", "path": "missing"}
    )

    assert result.ok is False
    assert "not found" in result.summary


async def test_search_files_skips_unreadable_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write(tmp_path / "locked.py", "needle")
    write(tmp_path / "open.py", "needle")

    import beast.tools.filesystem as filesystem

    real_read = filesystem._read_text

    def flaky_read(path: Path, limit: int) -> str | None:
        if path.name == "locked.py":
            raise PermissionError("access denied")
        return real_read(path, limit)

    monkeypatch.setattr(filesystem, "_read_text", flaky_read)

    found = await SearchFilesTool(context(tmp_path)[0]).run({"query": "needle"})
    missing = await SearchFilesTool(context(tmp_path)[0]).run({"query": "absent"})

    assert found.ok is True
    assert "open.py:1: needle" in found.content
    assert "locked.py" not in found.content
    assert missing.ok is True
    assert "1 file(s) could not be read" in missing.content


async def test_search_files_redacts_before_truncating_snippets(
    tmp_path: Path,
) -> None:
    # The closing quote of the assignment lies past the 240-character snippet
    # limit; truncating first would leave the value unmatched and unredacted.
    filler = "x" * 220
    write(tmp_path / "config.py", f'{filler} password = "hunter2secretvalue"\n')

    result = await SearchFilesTool(context(tmp_path)[0]).run({"query": "password"})

    assert result.ok is True
    assert "hunter2secretvalue" not in result.content


# ------------------------------------------------------------------- workspace


async def test_tools_refuse_paths_outside_the_workspace(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    ctx, approver = context(tmp_path)

    read = await ReadFileTool(ctx).run({"path": "../outside.txt"})
    write_result = await WriteFileTool(ctx).run(
        {"path": "../outside.txt", "content": "x"}
    )
    listed = await ListFilesTool(ctx).run({"path": ".."})

    for result in (read, write_result, listed):
        assert result.ok is False
        assert "outside the workspace" in result.summary
    assert approver.requests == []
    assert outside.read_text(encoding="utf-8") == "secret"


async def test_a_symlink_cannot_escape_the_workspace(tmp_path: Path) -> None:
    outside = write(tmp_path.parent / "outside.txt", "secret")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")

    result = await ReadFileTool(context(tmp_path)[0]).run({"path": "link.txt"})

    assert result.ok is False
    assert "outside the workspace" in result.summary
