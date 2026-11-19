"""Immutable value objects shared by the intelligence layer.

Every analyser returns one of these frozen dataclasses, so the index, the
context builder and the terminal interface all speak the same vocabulary and
nothing accidentally mutates a cached result.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SymbolKind(StrEnum):
    """The kind of declaration a :class:`Symbol` describes."""

    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    INTERFACE = "interface"
    TYPE = "type"
    ENUM = "enum"
    CONSTANT = "constant"


@dataclass(frozen=True, slots=True)
class Symbol:
    """A class, function, method or type declaration found in a file."""

    name: str
    kind: SymbolKind
    line: int = 0
    end_line: int = 0
    signature: str = ""
    parent: str | None = None
    doc: str = ""
    exported: bool = False
    decorators: tuple[str, ...] = ()

    @property
    def qualified_name(self) -> str:
        return f"{self.parent}.{self.name}" if self.parent else self.name

    @property
    def display(self) -> str:
        """A readable label with the declaration's detail, without duplication.

        ``signature`` contains the whole declaration (``def login(...)``), so
        naively appending it to the qualified name repeats the identifier.
        This strips the leading modifiers, keyword and name when present.
        """
        signature = " ".join(self.signature.split())
        if not signature:
            return self.qualified_name
        for modifier in ("export default ", "export ", "default ", "declare "):
            if signature.startswith(modifier):
                signature = signature[len(modifier) :]
                break

        remainder: str | None = None
        for keyword in (
            "async def ",
            "def ",
            "class ",
            "async function ",
            "function ",
            "const ",
            "let ",
            "var ",
            "interface ",
            "type ",
            "enum ",
        ):
            if signature.startswith(keyword):
                rest = signature[len(keyword) :].lstrip()
                if rest.startswith(self.name):
                    rest = rest[len(self.name) :]
                remainder = rest
                break
        if remainder is None:
            if signature.startswith(self.name):
                remainder = signature[len(self.name) :]
            else:
                remainder = f" {signature}"

        remainder = remainder.rstrip()
        if remainder and not remainder.strip(" =;:"):
            return self.qualified_name
        return f"{self.qualified_name}{remainder}"


@dataclass(frozen=True, slots=True)
class ImportInfo:
    """One import statement, normalised across languages."""

    module: str
    names: tuple[str, ...] = ()
    line: int = 0
    level: int = 0
    is_from: bool = False

    def display(self) -> str:
        prefix = "." * self.level
        if self.is_from and self.names:
            return f"from {prefix}{self.module} import {', '.join(self.names)}"
        return f"import {prefix}{self.module}"


@dataclass(frozen=True, slots=True)
class FileAnalysis:
    """The result of analysing one source file."""

    language: str
    symbols: tuple[Symbol, ...] = ()
    imports: tuple[ImportInfo, ...] = ()
    lines: int = 0
    ok: bool = True
    error: str = ""


@dataclass(frozen=True, slots=True)
class LanguageStat:
    """How much of the project is written in one language."""

    name: str
    file_count: int
    extensions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ManifestInfo:
    """A detected dependency or build manifest."""

    path: str
    kind: str
    ecosystem: str = ""
    package_manager: str = ""


@dataclass(frozen=True, slots=True)
class Dependency:
    """One declared dependency."""

    name: str
    version: str | None = None
    scope: str = "runtime"
    ecosystem: str = ""
    source: str = ""

    def display(self) -> str:
        if self.version:
            return f"{self.name} {self.version}"
        return self.name


@dataclass(frozen=True, slots=True)
class DependencyReport:
    """Everything discovered from dependency manifests."""

    dependencies: tuple[Dependency, ...] = ()
    manifests: tuple[ManifestInfo, ...] = ()
    package_managers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def names(self) -> frozenset[str]:
        return frozenset(dependency.name.lower() for dependency in self.dependencies)

    def by_scope(self, scope: str) -> tuple[Dependency, ...]:
        return tuple(
            dependency
            for dependency in self.dependencies
            if dependency.scope == scope
        )

    def render(self, limit: int = 40) -> str:
        """A compact, evidence-backed dependency listing."""
        if not self.dependencies:
            return "No dependency manifests detected."
        lines: list[str] = []
        groups: dict[str, list[Dependency]] = {}
        for dependency in self.dependencies:
            groups.setdefault(dependency.scope, []).append(dependency)
        for scope in ("runtime", "dev", "build", "peer", "optional", "unknown"):
            items = groups.get(scope)
            if not items:
                continue
            lines.append(f"{scope}:")
            for dependency in items[:limit]:
                source = f"  ({dependency.source})" if dependency.source else ""
                lines.append(f"  - {dependency.display()}{source}")
            if len(items) > limit:
                lines.append(f"  … {len(items) - limit} more")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ProjectDetection:
    """What kind of project this is, derived from files and manifests."""

    name: str
    languages: tuple[LanguageStat, ...] = ()
    primary_language: str | None = None
    frameworks: tuple[str, ...] = ()
    package_managers: tuple[str, ...] = ()
    build_systems: tuple[str, ...] = ()
    test_frameworks: tuple[str, ...] = ()
    manifests: tuple[ManifestInfo, ...] = ()
    entry_points: tuple[str, ...] = ()
    config_files: tuple[str, ...] = ()
    source_file_count: int = 0
    test_file_count: int = 0

    @property
    def language_names(self) -> tuple[str, ...]:
        return tuple(language.name for language in self.languages)

    def summary_line(self) -> str:
        """One-line summary used by the header and the splash panel."""
        parts = [self.primary_language or "unknown"]
        if self.frameworks:
            parts.append(", ".join(self.frameworks[:3]))
        if self.package_managers:
            parts.append(self.package_managers[0])
        parts.append(f"{self.source_file_count} source files")
        if self.test_file_count:
            parts.append(f"{self.test_file_count} test files")
        return " · ".join(parts)


@dataclass(frozen=True, slots=True)
class TreeNode:
    """One node in a bounded directory tree."""

    name: str
    path: str
    is_dir: bool
    children: tuple["TreeNode", ...] = ()
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class ProjectStructure:
    """The layout of the project, limited to what is useful for context."""

    tree: TreeNode | None = None
    source_dirs: tuple[str, ...] = ()
    test_dirs: tuple[str, ...] = ()
    docs: tuple[str, ...] = ()
    config_files: tuple[str, ...] = ()
    key_files: tuple[str, ...] = ()
    truncated: bool = False

    def render_tree(self, max_lines: int = 80) -> str:
        if self.tree is None:
            return "(empty project)"
        lines: list[str] = []
        self._render_node(self.tree, "", lines, max_lines)
        if len(lines) >= max_lines:
            lines.append("… (tree truncated)")
        return "\n".join(lines)

    def _render_node(
        self,
        node: TreeNode,
        prefix: str,
        lines: list[str],
        max_lines: int,
    ) -> None:
        if len(lines) >= max_lines:
            return
        label = f"{node.name}/" if node.is_dir else node.name
        if node.truncated:
            label += " …"
        lines.append(f"{prefix}{label}")
        if not node.is_dir:
            return
        for index, child in enumerate(node.children):
            last = index == len(node.children) - 1
            branch = "└─ " if last else "├─ "
            self._render_node(child, prefix + ("   " if last else "│  "), lines, max_lines)


@dataclass(frozen=True, slots=True)
class ProjectProfile:
    """Everything BEAST knows about the project without reading every symbol."""

    detection: ProjectDetection
    structure: ProjectStructure
    dependencies: DependencyReport

    def render_overview(self, *, include_dependencies: bool = True) -> str:
        """Human-readable overview built only from inspected evidence."""
        detection = self.detection
        lines = [
            f"Project: {detection.name}",
            f"Languages: {', '.join(detection.language_names) or 'none detected'}",
            f"Primary language: {detection.primary_language or 'unknown'}",
            f"Frameworks: {', '.join(detection.frameworks) or 'none detected'}",
            f"Package managers: {', '.join(detection.package_managers) or 'none detected'}",
            f"Build systems: {', '.join(detection.build_systems) or 'none detected'}",
            f"Test frameworks: {', '.join(detection.test_frameworks) or 'none detected'}",
            f"Source files: {detection.source_file_count}",
            f"Test files: {detection.test_file_count}",
            f"Entry points: {', '.join(detection.entry_points) or 'none detected'}",
            f"Source directories: {', '.join(self.structure.source_dirs) or '(root)'}",
            f"Test directories: {', '.join(self.structure.test_dirs) or 'none detected'}",
            f"Documentation: {', '.join(self.structure.docs) or 'none detected'}",
        ]
        if detection.manifests:
            lines.append("Manifests:")
            lines.extend(
                f"  - {manifest.path} ({manifest.kind})"
                for manifest in detection.manifests
            )
        if include_dependencies and self.dependencies.dependencies:
            lines.append("Dependencies:")
            lines.append(self.dependencies.render(limit=20))
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ContextBundle:
    """The compact project context attached to one agent turn."""

    text: str
    query: str = ""
    files: tuple[str, ...] = ()
    symbol_count: int = 0
    characters: int = 0
    truncated: bool = False
    redactions: int = 0
    profile: ProjectProfile | None = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass(frozen=True, slots=True)
class IndexUpdate:
    """A summary of one incremental index refresh."""

    added: int = 0
    modified: int = 0
    removed: int = 0
    unchanged: int = 0
    analyzed: int = 0
    skipped: int = 0
    total: int = 0
    truncated: bool = False

    @property
    def changed(self) -> int:
        return self.added + self.modified + self.removed


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A symbol matched by the index search."""

    path: str
    symbol: Symbol
    score: float = 0.0
