"""Directory structure analysis, scanning limits and path classification."""

from __future__ import annotations

from pathlib import Path

import pytest

from beast.intelligence.models import ProjectStructure
from beast.intelligence.structure_analyzer import StructureAnalyzer
from beast.workspace.scanning import (
    ScanLimits,
    is_binary_path,
    is_config_file,
    is_documentation,
    is_entry_point,
    is_test_path,
    looks_binary,
    parse_gitignore,
    scan_project,
)


def structure_for(
    root: Path, *, analyzer: StructureAnalyzer | None = None
) -> tuple[ProjectStructure, tuple[str, ...]]:
    scan = scan_project(root)
    analyzer = analyzer or StructureAnalyzer()
    return analyzer.build(scan.files, root=root), tuple(
        file.relative for file in scan.files
    )


def test_flask_structure_names_code_tests_docs_and_config(
    flask_project: Path,
) -> None:
    structure, _ = structure_for(flask_project)

    assert structure.source_dirs == (".",)
    assert structure.test_dirs == ("tests",)
    assert structure.docs == ("README.md",)
    assert {"requirements.txt", ".gitignore"} <= set(structure.config_files)
    assert {"README.md", "app.py", "requirements.txt"} <= set(structure.key_files)
    assert structure.truncated is False


def test_tree_rendering_is_readable_and_hides_credentials(
    flask_project: Path,
) -> None:
    structure, _ = structure_for(flask_project)
    rendered = structure.render_tree()

    assert "app.py" in rendered
    assert "auth.py" in rendered
    assert "tests/" in rendered
    assert "test_auth.py" in rendered
    assert ".env" not in rendered


def test_deep_directories_are_bounded_and_marked(write_files) -> None:
    root = write_files(
        {
            "a/b/c/d/e/deep.py": "x = 1\n",
            "a/top.py": "y = 2\n",
        }
    )
    structure, _ = structure_for(root, analyzer=StructureAnalyzer(max_depth=2))

    assert structure.truncated is True
    rendered = structure.render_tree()
    assert "a/" in rendered
    assert "deep.py" not in rendered


def test_many_files_in_one_directory_are_bounded(write_files) -> None:
    files = {f"module_{index}.py": f"VALUE_{index} = {index}\n" for index in range(12)}
    root = write_files(files)
    structure, _ = structure_for(
        root, analyzer=StructureAnalyzer(max_files_per_dir=3)
    )

    assert structure.truncated is True
    rendered = structure.render_tree()
    shown = sum(1 for name in files if name in rendered)
    assert shown <= 3


def test_gitignore_rules_are_applied_with_negation(write_files) -> None:
    root = write_files(
        {
            ".gitignore": "secret/\n*.log\n!keep.log\nbuild_output/\n",
            "app.py": "x = 1\n",
            "secret/token.py": "TOKEN = 'x'\n",
            "debug.log": "noise\n",
            "keep.log": "kept\n",
            "build_output/artifact.py": "y = 2\n",
        }
    )
    scan = scan_project(root)
    relatives = {file.relative for file in scan.files}

    assert "app.py" in relatives
    assert "keep.log" in relatives
    assert "token.py" not in relatives
    assert "secret/token.py" not in relatives
    assert "debug.log" not in relatives
    assert "build_output/artifact.py" not in relatives


def test_nested_gitignore_applies_to_its_directory(write_files) -> None:
    root = write_files(
        {
            ".gitignore": "*.log\n",
            "sub/.gitignore": "*.tmp\n",
            "sub/drop.tmp": "x\n",
            "sub/keep.py": "y = 1\n",
            "other/drop.tmp": "z\n",
        }
    )
    relatives = {file.relative for file in scan_project(root).files}

    assert "sub/keep.py" in relatives
    assert "sub/drop.tmp" not in relatives
    # A nested .gitignore does not leak into a sibling directory.
    assert "other/drop.tmp" in relatives


def test_nested_gitignore_patterns_are_relative_to_their_own_directory(
    write_files,
) -> None:
    root = write_files(
        {
            "sub/.gitignore": "docs/*.md\n/output/\n",
            "sub/docs/drop.md": "x\n",
            "sub/docs/keep.py": "y = 1\n",
            "sub/output/out.py": "z = 2\n",
            "docs/keep.md": "keep\n",
            "sub/keep.py": "w = 3\n",
        }
    )
    relatives = {file.relative for file in scan_project(root).files}

    assert "sub/docs/drop.md" not in relatives
    assert "sub/docs/keep.py" in relatives
    assert "sub/output/out.py" not in relatives
    # The rule is anchored at ``sub/``; it must not hide the root ``docs/``.
    assert "docs/keep.md" in relatives
    assert "sub/keep.py" in relatives


def test_generated_directories_are_matched_case_insensitively(write_files) -> None:
    root = write_files(
        {
            "NODE_MODULES/pkg/index.js": "module.exports = {};\n",
            ".GIT/config": "[core]\n",
            "app.py": "x = 1\n",
        }
    )

    relatives = {file.relative for file in scan_project(root).files}

    assert relatives == {"app.py"}


def test_generated_directories_are_never_scanned(write_files) -> None:
    root = write_files(
        {
            "app.py": "x = 1\n",
            "node_modules/pkg/index.js": "module.exports = {};\n",
            ".git/config": "[core]\n",
            "venv/lib/site.py": "y = 2\n",
            "__pycache__/app.cpython-312.pyc": b"\x00\x01",
            "dist/bundle.js": "z = 3;\n",
        }
    )
    relatives = {file.relative for file in scan_project(root).files}

    assert relatives == {"app.py"}


def test_scan_limits_truncate_deterministically(write_files) -> None:
    root = write_files({f"file_{index}.py": "x = 1\n" for index in range(6)})

    limited = scan_project(root, limits=ScanLimits(max_files=2))
    assert limited.truncated is True
    assert len(limited.files) == 2

    shallow = write_files({"a/b/deep.py": "x = 1\n"})
    result = scan_project(shallow, limits=ScanLimits(max_depth=0))
    assert result.truncated is True
    assert all("/" not in file.relative for file in result.files)


def test_extra_ignore_patterns_are_supported(write_files) -> None:
    root = write_files({"keep.py": "x = 1\n", "generated.gen.py": "y = 2\n"})
    relatives = {
        file.relative
        for file in scan_project(
            root, limits=ScanLimits(extra_ignore=("*.gen.py",))
        ).files
    }

    assert relatives == {"keep.py"}


def test_binary_detection_uses_extensions_and_content(write_files) -> None:
    assert is_binary_path("image.png") is True
    assert is_binary_path("bundle.min.js") is True
    assert is_binary_path("app.py") is False
    assert looks_binary(b"PK\x03\x04\x00\x00") is True
    assert looks_binary(b"plain text") is False

    root = write_files({"asset.png": b"\x89PNG", "module.py": "x = 1\n"})
    by_name = {file.relative: file for file in scan_project(root).files}
    assert by_name["asset.png"].is_binary is True
    assert by_name["module.py"].is_binary is False


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("tests/test_auth.py", True),
        ("src/__tests__/button.test.tsx", True),
        ("src/components/Button.tsx", False),
        ("conftest.py", True),
        ("pkg/handler_test.go", True),
        ("app.py", False),
    ],
)
def test_test_path_classification(path: str, expected: bool) -> None:
    assert is_test_path(path) is expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("app.py", True),
        ("src/main.tsx", True),
        ("app/page.tsx", True),
        ("cmd/server/main.go", True),
        ("src/bin/tool.rs", True),
        ("lib/helper.py", False),
        ("src/components/Button.tsx", False),
    ],
)
def test_entry_point_classification(path: str, expected: bool) -> None:
    assert is_entry_point(path) is expected


def test_config_and_documentation_classification() -> None:
    assert is_config_file("pyproject.toml") is True
    assert is_config_file("vite.config.ts") is True
    assert is_config_file(".github/workflows/ci.yml") is True
    assert is_config_file("src/main.py") is False
    assert is_documentation("README.md") is True
    assert is_documentation("docs/architecture.md") is True
    assert is_documentation("src/config.txt") is False


def test_gitignore_parser_anchoring_and_directory_only_rules() -> None:
    rules = parse_gitignore("/root_only/\n*.log\n!important.log\ndocs/\n")
    from beast.workspace.scanning import is_ignored

    assert is_ignored(rules, "root_only", is_dir=True) is True
    assert is_ignored(rules, "nested/root_only", is_dir=True) is False
    assert is_ignored(rules, "any/debug.log", is_dir=False) is True
    assert is_ignored(rules, "any/important.log", is_dir=False) is False
    assert is_ignored(rules, "docs/file.md", is_dir=False) is False
    assert is_ignored(rules, "other/docs", is_dir=True) is True
