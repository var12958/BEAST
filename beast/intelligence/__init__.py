"""Project intelligence for BEAST (Phase 3).

The public entry point is :class:`ProjectIntelligence`.  It owns the scan
index, the detection/structure/dependency analysers and the context builder,
and it is the only object the agent and the tool layer need to know about.

Design rules:

* everything is derived from files inside ``root`` (the active workspace);
* every expensive operation is incremental or bounded;
* credential-bearing files are excluded before any content is read;
* the facade is thread-safe because the agent runs analysis in a worker thread.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from beast.intelligence.code_analyzer import (
    analyze_source,
    analyzer_for,
    register_analyzer,
    supported_extensions,
    supported_languages,
)
from beast.intelligence.context_builder import (
    ContextBuilder,
    ContextLimits,
    ContextProvider,
)
from beast.intelligence.dependency_analyzer import DependencyAnalyzer
from beast.intelligence.index_manager import (
    IndexManager,
    IndexedFile,
    ProjectIndex,
)
from beast.intelligence.models import (
    ContextBundle,
    DependencyReport,
    FileAnalysis,
    IndexUpdate,
    ProjectDetection,
    ProjectProfile,
    ProjectStructure,
    SearchHit,
    Symbol,
)
from beast.intelligence.project_detector import ProjectDetector
from beast.intelligence.structure_analyzer import StructureAnalyzer
from beast.workspace.scanning import ScanLimits, ScannedFile

__all__ = [
    "ContextBuilder",
    "ContextLimits",
    "ContextProvider",
    "DependencyAnalyzer",
    "IndexManager",
    "IndexedFile",
    "ProjectIndex",
    "ProjectIntelligence",
    "ProjectDetector",
    "StructureAnalyzer",
    "analyze_source",
    "analyzer_for",
    "register_analyzer",
    "supported_extensions",
    "supported_languages",
]

_DEFAULT_REFRESH_INTERVAL = 5.0


class ProjectIntelligence:
    """Facade over indexing, analysis and context construction."""

    def __init__(
        self,
        root: str | Path,
        *,
        limits: ScanLimits | None = None,
        cache_dir: str | Path | None = None,
        context_limits: ContextLimits | None = None,
        refresh_interval: float = _DEFAULT_REFRESH_INTERVAL,
    ) -> None:
        self.root = Path(root).resolve()
        self.limits = limits or ScanLimits()
        self.refresh_interval = refresh_interval
        self.index_manager = IndexManager(
            self.root, limits=self.limits, cache_dir=cache_dir
        )
        self.detector = ProjectDetector()
        self.structure_analyzer = StructureAnalyzer()
        self.dependency_analyzer = DependencyAnalyzer()
        self.context_builder = ContextBuilder(
            self.root, limits=context_limits or ContextLimits()
        )
        self._lock = threading.RLock()
        self._index: ProjectIndex | None = None
        self._profile: ProjectProfile | None = None
        self._last_refresh = 0.0
        self._last_update: IndexUpdate | None = None

    # ------------------------------------------------------------------ state

    @property
    def index(self) -> ProjectIndex:
        with self._lock:
            self._ensure_locked()
            assert self._index is not None
            return self._index

    @property
    def profile(self) -> ProjectProfile:
        with self._lock:
            self._ensure_locked()
            assert self._profile is not None
            return self._profile

    @property
    def last_update(self) -> IndexUpdate | None:
        with self._lock:
            return self._last_update

    def detection(self) -> ProjectDetection:
        return self.profile.detection

    def invalidate(self) -> None:
        """Mark the cached analysis stale (e.g. after the agent writes a file)."""
        with self._lock:
            self._last_refresh = 0.0

    # ---------------------------------------------------------------- refresh

    def refresh(self, *, force: bool = True) -> IndexUpdate:
        """Re-scan and incrementally rebuild the index.

        ``force=False`` is a no-op while the refresh interval has not elapsed.
        """
        with self._lock:
            now = time.monotonic()
            if (
                not force
                and self._index is not None
                and now - self._last_refresh < self.refresh_interval
            ):
                return self._last_update or IndexUpdate()
            index, update = self.index_manager.refresh()
            profile = self._build_profile(index)
            self._index = index
            self._profile = profile
            self._last_refresh = time.monotonic()
            self._last_update = update
            return update

    def _ensure_locked(self) -> None:
        if self._index is None or self._profile is None:
            self.refresh(force=True)
            return
        if time.monotonic() - self._last_refresh >= self.refresh_interval:
            self.refresh(force=True)

    def _build_profile(self, index: ProjectIndex) -> ProjectProfile:
        files = tuple(_scanned_from_index(self.root, indexed) for indexed in index.files.values())
        dependencies = self.dependency_analyzer.analyze(
            self.root, relative_paths=index.files.keys()
        )
        detection = self.detector.detect(
            files, dependencies, root=self.root
        )
        structure = self.structure_analyzer.build(
            files, root=self.root, truncated=index.truncated
        )
        return ProjectProfile(
            detection=detection,
            structure=structure,
            dependencies=dependencies,
        )

    # ------------------------------------------------------------ intelligence

    def context_for(self, query: str) -> ContextBundle:
        """Retrieve and render the project context relevant to ``query``."""
        with self._lock:
            self._ensure_locked()
            assert self._index is not None and self._profile is not None
            index, profile = self._index, self._profile
        # Build outside the lock: rendering only reads immutable snapshots.
        return self.context_builder.build(query, index=index, profile=profile)

    def overview(self, *, include_dependencies: bool = True) -> str:
        """Render a project overview (used by the project tool and CLI)."""
        profile = self.profile
        text = profile.render_overview(include_dependencies=include_dependencies)
        tree = profile.structure.render_tree(max_lines=70)
        if tree:
            text += f"\n\nStructure:\n{tree}"
        return text

    def summary_line(self) -> str:
        """One-line project summary for the TUI header."""
        return self.profile.detection.summary_line()

    def structure(self) -> ProjectStructure:
        return self.profile.structure

    def dependencies(self) -> DependencyReport:
        return self.profile.dependencies

    def file_analysis(self, relative: str) -> FileAnalysis | None:
        return self.index.file_analysis(relative)

    def symbols(self, relative: str | None = None) -> tuple[tuple[str, Symbol], ...]:
        """Symbols for one file, or every indexed symbol."""
        index = self.index
        if relative is not None:
            analysis = index.file_analysis(relative)
            if analysis is None:
                return ()
            return tuple((relative, symbol) for symbol in analysis.symbols)
        return tuple(index.symbols())

    def search(self, query: str, *, limit: int = 20) -> tuple[SearchHit, ...]:
        return self.index.search(query, limit=limit)

    def close(self) -> None:
        """Drop cached state; the on-disk cache is left in place."""
        with self._lock:
            self._index = None
            self._profile = None
            self._last_refresh = 0.0


def _scanned_from_index(root: Path, indexed: IndexedFile) -> ScannedFile:
    return ScannedFile(
        path=root / indexed.relative,
        relative=indexed.relative,
        size=indexed.size,
        mtime_ns=indexed.mtime_ns,
        language=indexed.language,
        is_source=indexed.is_source,
        is_binary=indexed.is_binary,
    )
