"""Incremental indexing: metadata, symbols, caching and cache integrity."""

from __future__ import annotations

import json
from pathlib import Path

from beast.intelligence.index_manager import IndexManager
from beast.workspace.scanning import ScanLimits


def manager_for(root: Path, **kwargs) -> IndexManager:
    return IndexManager(root, **kwargs)


def test_first_refresh_indexes_only_real_project_files(flask_project: Path) -> None:
    manager = manager_for(flask_project)
    index, update = manager.refresh()

    assert set(index.files) == {
        ".gitignore",
        "README.md",
        "app.py",
        "auth.py",
        "requirements.txt",
        "tests/test_auth.py",
    }
    assert ".env" not in index.files
    assert update.added == 6
    assert update.modified == 0
    assert update.removed == 0
    assert update.unchanged == 0
    assert update.analyzed == 3
    assert update.skipped == 0
    assert update.total == 6
    assert update.truncated is False

    auth = index.file_analysis("auth.py")
    assert auth is not None
    assert auth.ok is True
    assert any(symbol.name == "authenticate" for symbol in auth.symbols)
    assert index.file_analysis("README.md") is None
    assert index.file_analysis("missing.py") is None
    assert "app.py" in {file.relative for file in index.source_files}


def test_second_refresh_is_incremental_and_keeps_analyses(
    flask_project: Path,
) -> None:
    manager = manager_for(flask_project)
    first, _ = manager.refresh()
    second, update = manager.refresh()

    assert update.added == 0
    assert update.modified == 0
    assert update.removed == 0
    assert update.unchanged == 6
    assert update.analyzed == 0
    assert update.total == 6
    assert set(second.files) == set(first.files)
    assert second.file_analysis("auth.py") is not None
    assert second.symbol_count == first.symbol_count


def test_modified_file_is_reanalysed(flask_project: Path) -> None:
    manager = manager_for(flask_project)
    manager.refresh()

    auth = flask_project / "auth.py"
    auth.write_text(
        auth.read_text(encoding="utf-8") + "\n# touched by the test\n",
        encoding="utf-8",
    )

    index, update = manager.refresh()

    assert update.modified == 1
    assert update.added == 0
    assert update.unchanged == 5
    assert update.analyzed == 1
    assert update.total == 6
    assert index.file_analysis("auth.py") is not None


def test_deleted_file_is_removed_from_the_index(flask_project: Path) -> None:
    manager = manager_for(flask_project)
    manager.refresh()

    (flask_project / "tests" / "test_auth.py").unlink()

    index, update = manager.refresh()

    assert update.removed == 1
    assert update.added == 0
    assert update.unchanged == 5
    assert update.total == 5
    assert "tests/test_auth.py" not in index.files


def test_ignored_and_sensitive_paths_are_never_indexed(flask_project: Path) -> None:
    (flask_project / "node_modules" / "pkg").mkdir(parents=True)
    (flask_project / "node_modules" / "pkg" / "index.js").write_text(
        "module.exports = {};\n", encoding="utf-8"
    )
    (flask_project / "credentials.json").write_text("{}", encoding="utf-8")
    (flask_project / "server.pem").write_text("key material", encoding="utf-8")
    (flask_project / ".env.local").write_text("TOKEN=abc\n", encoding="utf-8")

    index, update = manager_for(flask_project).refresh()

    assert "node_modules/pkg/index.js" not in index.files
    assert "credentials.json" not in index.files
    assert "server.pem" not in index.files
    assert ".env" not in index.files
    assert ".env.local" not in index.files
    assert update.total == 6


def test_unsupported_language_is_skipped_with_a_reason(flask_project: Path) -> None:
    (flask_project / "script.rb").write_text("puts 'hi'\n", encoding="utf-8")

    index, update = manager_for(flask_project).refresh()

    indexed = index.files["script.rb"]
    assert indexed.is_source is True
    assert indexed.analysis is None
    assert indexed.skipped == "no analyser"
    assert update.skipped == 1
    assert update.analyzed == 3


def test_oversized_source_is_skipped_with_a_reason(write_files) -> None:
    root = write_files({"large.py": "# padding\n" * 40})
    manager = manager_for(root, limits=ScanLimits(max_file_bytes=128))

    index, update = manager.refresh()

    indexed = index.files["large.py"]
    assert indexed.analysis is None
    assert indexed.skipped == "exceeds size limit"
    assert update.skipped == 1
    assert update.analyzed == 0


def test_binary_source_files_record_a_skip_reason(write_files) -> None:
    root = write_files({"app.py": "x = 1\n", "bundle.min.js": "console.log(1);\n"})

    index, update = manager_for(root).refresh()

    bundled = index.files["bundle.min.js"]
    assert bundled.is_source is True
    assert bundled.is_binary is True
    assert bundled.skipped == "binary"
    assert update.skipped == 1


def test_scan_limit_truncates_the_index_deterministically(write_files) -> None:
    root = write_files({"a.py": "a = 1\n", "b.py": "b = 2\n", "c.py": "c = 3\n"})

    index, update = manager_for(root, limits=ScanLimits(max_files=2)).refresh()

    assert update.truncated is True
    assert update.total == 2
    assert set(index.files) == {"a.py", "b.py"}


def test_cache_survives_a_new_manager(flask_project: Path) -> None:
    first_manager = manager_for(flask_project)
    first, _ = first_manager.refresh()

    second_manager = manager_for(flask_project)
    loaded = second_manager.load()

    assert loaded is not None
    assert loaded.root == flask_project.resolve()
    assert set(loaded.files) == set(first.files)
    assert loaded.file_analysis("auth.py") is not None
    assert loaded.symbol_count == first.symbol_count


def test_corrupt_cache_is_ignored_and_rebuilt(flask_project: Path) -> None:
    manager = manager_for(flask_project)
    manager.refresh()
    manager.cache_path.write_text("{not json", encoding="utf-8")

    assert manager.load() is None

    index, update = manager.refresh()
    assert update.added == 6
    assert set(index.files) == {
        ".gitignore",
        "README.md",
        "app.py",
        "auth.py",
        "requirements.txt",
        "tests/test_auth.py",
    }


def test_tampered_cache_cannot_inject_unsafe_paths(flask_project: Path) -> None:
    manager = manager_for(flask_project)
    manager.cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "root": str(flask_project.resolve()),
        "created_at": 0,
        "updated_at": 0,
        "truncated": False,
        "fingerprint": "",
        "files": {
            "app.py": {
                "size": 10,
                "mtime_ns": 10,
                "language": "Python",
                "is_source": True,
                "is_binary": False,
            },
            "../evil.py": {
                "size": 1,
                "mtime_ns": 1,
                "language": "Python",
                "is_source": True,
                "is_binary": False,
            },
            ".env": {
                "size": 1,
                "mtime_ns": 1,
                "language": None,
                "is_source": False,
                "is_binary": False,
            },
        },
    }
    manager.cache_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = manager.load()

    assert loaded is not None
    assert set(loaded.files) == {"app.py"}


def test_cache_directory_inside_the_workspace_is_not_indexed(write_files) -> None:
    root = write_files({"app.py": "x = 1\n"})
    manager = manager_for(root, cache_dir=root / "cache")

    _, first = manager.refresh()
    _, second = manager.refresh()

    assert first.total == 1
    assert second.added == 0
    assert second.unchanged == 1
    assert second.total == 1


def test_search_finds_symbols_by_name_and_reports_paths(flask_project: Path) -> None:
    index, _ = manager_for(flask_project).refresh()

    hits = index.search("authenticate")

    assert hits
    assert hits[0].path == "auth.py"
    assert any(hit.symbol.name == "authenticate" for hit in hits)
    assert len(index.search("authenticate", limit=1)) == 1
    assert index.search("") == ()
    assert index.search("   ") == ()
