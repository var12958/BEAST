"""Bounded project-structure analysis.

The structure is deliberately shallow and selective: directories, the files
that explain a directory (manifests, entry points, configuration, docs), and
counts for source/test directories.  It is an overview, not a file listing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from beast.intelligence.models import ProjectStructure, TreeNode
from beast.workspace.scanning import (
    DOC_DIR_NAMES,
    TEST_DIR_NAMES,
    ScannedFile,
    is_config_file,
    is_documentation,
    is_entry_point,
    is_test_path,
)
from beast.workspace.secrets import is_sensitive_path

#: First path segments that usually only group real projects.
_CONTAINER_DIRS = frozenset(
    {"src", "lib", "libs", "packages", "apps", "services", "modules", "projects"}
)

_KEY_ROOT_NAMES = frozenset(
    {
        "readme.md",
        "readme.rst",
        "readme.txt",
        "license",
        "license.md",
        "licence",
        "changelog.md",
        "contributing.md",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "package.json",
        "package-lock.json",
        "tsconfig.json",
        "makefile",
        "cmakelists.txt",
        "dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        "compose.yml",
        "compose.yaml",
        ".gitignore",
        "go.mod",
        "cargo.toml",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "gemfile",
        "composer.json",
        "pubspec.yaml",
        "package.swift",
        "requirements.txt",
        "manage.py",
    }
)


class StructureAnalyzer:
    """Builds a compact tree plus source/test/docs/config summaries."""

    def __init__(
        self,
        *,
        max_depth: int = 4,
        max_children: int = 24,
        max_files_per_dir: int = 8,
        max_nodes: int = 600,
    ) -> None:
        self.max_depth = max_depth
        self.max_children = max_children
        self.max_files_per_dir = max_files_per_dir
        self.max_nodes = max_nodes

    def build(
        self,
        files: Iterable[ScannedFile],
        *,
        root: str | Path,
        truncated: bool = False,
    ) -> ProjectStructure:
        base = Path(root).resolve()
        visible = [
            file
            for file in files
            if not file.is_binary and not is_sensitive_path(file.relative)
        ]

        dirs: set[tuple[str, ...]] = {()}
        children: dict[tuple[str, ...], set[str]] = {}
        per_dir: dict[tuple[str, ...], list[ScannedFile]] = {}
        deep_dirs: set[tuple[str, ...]] = set()

        for file in visible:
            parts = tuple(file.relative.split("/"))
            parent = parts[:-1]
            directory_depth = len(parent)
            if directory_depth <= self.max_depth:
                per_dir.setdefault(parent, []).append(file)
            else:
                deep_dirs.add(parent[: self.max_depth])
            for depth in range(min(directory_depth, self.max_depth) + 1):
                prefix = parent[:depth]
                dirs.add(prefix)
                if depth > 0:
                    children.setdefault(prefix[:-1], set()).add(prefix[-1])

        for directory in dirs:
            if len(directory) > self.max_depth:
                continue
            deeper = any(
                len(other) > len(directory)
                and other[: len(directory)] == directory
                for other in dirs
            )
            if deeper or directory in deep_dirs:
                children.setdefault(directory, set())

        budget = [self.max_nodes]
        any_truncated = [truncated]
        tree = self._build_node(
            base, (), dirs, children, per_dir, deep_dirs, budget, any_truncated
        )

        production = [
            file
            for file in visible
            if file.is_source and not is_test_path(file.relative)
        ]
        source_dirs = _directory_counts(
            (file.relative for file in production), limit=12
        )
        test_dirs = _directory_counts(
            (
                file.relative
                for file in visible
                if file.is_source and is_test_path(file.relative)
            ),
            limit=10,
        )
        docs = _documentation(visible)
        config_files = _config_files(visible)
        key_files = _key_files(visible)

        return ProjectStructure(
            tree=tree,
            source_dirs=tuple(source_dirs),
            test_dirs=tuple(test_dirs),
            docs=tuple(docs),
            config_files=tuple(config_files),
            key_files=tuple(key_files),
            truncated=any_truncated[0] or tree.truncated,
        )

    # ------------------------------------------------------------------ tree

    def _build_node(
        self,
        base: Path,
        directory: tuple[str, ...],
        dirs: set[tuple[str, ...]],
        children: dict[tuple[str, ...], set[str]],
        per_dir: dict[tuple[str, ...], list[ScannedFile]],
        deep_dirs: set[tuple[str, ...]],
        budget: list[int],
        any_truncated: list[bool],
    ) -> TreeNode:
        # Every directory node shows its own segment; using ``base.name`` here
        # would label every nested directory with the workspace name.
        name = directory[-1] if directory else (base.name or str(base))
        relative = "/".join(directory)
        node_path = relative or "."
        is_truncated = directory in deep_dirs and len(directory) >= self.max_depth
        if budget[0] <= 0:
            any_truncated[0] = True
            return TreeNode(
                name=name,
                path=node_path,
                is_dir=True,
                children=(),
                truncated=True,
            )
        budget[0] -= 1

        child_dirs = sorted(
            children.get(directory, set()), key=lambda item: item.lower()
        )
        dir_nodes: list[TreeNode] = []
        for child in child_dirs[: self.max_children]:
            child_path = directory + (child,)
            if len(child_path) >= self.max_depth:
                any_truncated[0] = True
                child_truncated = (
                    child_path in deep_dirs
                    or bool(per_dir.get(child_path))
                    or any(
                        other != child_path and other[: len(child_path)] == child_path
                        for other in dirs
                    )
                )
                if budget[0] <= 0:
                    any_truncated[0] = True
                    break
                budget[0] -= 1
                dir_nodes.append(
                    TreeNode(
                        name=child,
                        path="/".join(child_path),
                        is_dir=True,
                        children=(),
                        truncated=child_truncated,
                    )
                )
                continue
            dir_nodes.append(
                self._build_node(
                    base,
                    child_path,
                    dirs,
                    children,
                    per_dir,
                    deep_dirs,
                    budget,
                    any_truncated,
                )
            )
        if len(child_dirs) > self.max_children:
            is_truncated = True
            any_truncated[0] = True

        selected, files_truncated = _select_files(
            per_dir.get(directory, []), self.max_files_per_dir
        )
        file_nodes = [
            TreeNode(
                name=Path(file.relative).name,
                path=file.relative,
                is_dir=False,
            )
            for file in selected
            if not file.is_binary and not is_sensitive_path(file.relative)
        ]
        if files_truncated:
            is_truncated = True
            any_truncated[0] = True

        return TreeNode(
            name=name,
            path=node_path,
            is_dir=True,
            children=tuple(dir_nodes + file_nodes),
            truncated=is_truncated,
        )


def _select_files(
    files: Iterable[ScannedFile], limit: int
) -> tuple[list[ScannedFile], bool]:
    ranked = [
        file
        for file in files
        if Path(file.relative).name != "__init__.py"
    ]
    ranked.sort(key=_file_priority)
    selected = ranked[:limit]
    return selected, len(ranked) > limit


def _file_priority(file: ScannedFile) -> tuple[int, str]:
    name = Path(file.relative).name
    if is_entry_point(file.relative):
        return (0, name.lower())
    if is_config_file(file.relative):
        return (1, name.lower())
    if is_documentation(file.relative):
        return (2, name.lower())
    if is_test_path(file.relative):
        return (4, name.lower())
    if file.is_source:
        return (3, name.lower())
    return (5, name.lower())


def _directory_counts(relatives: Iterable[str], *, limit: int) -> list[str]:
    counts: Counter[str] = Counter()
    for relative in relatives:
        counts[_top_directory(relative)] += 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [directory for directory, _ in ordered[:limit]]


def _top_directory(relative: str) -> str:
    parts = relative.split("/")[:-1]
    if not parts:
        return "."
    if parts[0].lower() in TEST_DIR_NAMES:
        return parts[0]
    if parts[0].lower() in _CONTAINER_DIRS and len(parts) > 1:
        return f"{parts[0]}/{parts[1]}"
    return parts[0]


def _documentation(files: Iterable[ScannedFile]) -> list[str]:
    found: list[str] = []
    seen_dirs: set[str] = set()
    for file in sorted(files, key=lambda item: item.relative):
        relative = file.relative
        parts = relative.split("/")
        if len(parts) > 1 and parts[0].lower() in DOC_DIR_NAMES:
            if parts[0] not in seen_dirs:
                seen_dirs.add(parts[0])
                found.append(parts[0])
            continue
        if len(parts) == 1 and is_documentation(relative):
            found.append(relative)
    return found[:12]


def _config_files(files: Iterable[ScannedFile]) -> list[str]:
    found = [
        file.relative
        for file in files
        if is_config_file(file.relative) and not is_sensitive_path(file.relative)
    ]

    def priority(relative: str) -> tuple[int, int, str]:
        return (relative.count("/"), len(relative), relative)

    return sorted(dict.fromkeys(found), key=priority)[:40]


def _key_files(files: Iterable[ScannedFile]) -> list[str]:
    found: list[str] = []
    for file in sorted(files, key=lambda item: item.relative):
        relative = file.relative
        name = Path(relative).name.lower()
        if name.startswith(".env"):
            continue
        if "/" not in relative and (name in _KEY_ROOT_NAMES or is_entry_point(relative)):
            found.append(relative)
        elif is_entry_point(relative) and not is_test_path(relative):
            found.append(relative)
        elif relative.startswith((".github/workflows/", ".gitlab/", ".circleci/")):
            found.append(relative)
    return list(dict.fromkeys(found))[:20]
