"""Incremental, local-only project index.

The index is a single JSON document in the user's cache directory.  It holds
file metadata and, for Python/JavaScript/TypeScript sources, the symbols and
imports extracted by :mod:`beast.intelligence.code_analyzer`.

Incremental refresh compares ``size`` and ``mtime_ns``: unchanged files keep
their previous analysis, modified/added files are re-parsed, deleted files are
dropped.  Parsing only ever happens for files inside the scanned workspace, and
credential-bearing files are never indexed at all.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from beast.intelligence.code_analyzer import analyze_source, analyzer_for
from beast.intelligence.models import (
    FileAnalysis,
    ImportInfo,
    IndexUpdate,
    SearchHit,
    Symbol,
    SymbolKind,
)
from beast.intelligence.text import score_identifiers, score_text, tokenize
from beast.workspace.scanning import (
    ScanLimits,
    ScannedFile,
    is_junction,
    read_text_limited,
    scan_project,
)
from beast.workspace.secrets import is_sensitive_path

#: Bump when the on-disk layout changes.
INDEX_VERSION = 1
#: Bump when analyser behaviour changes, forcing re-analysis of unchanged files.
ANALYZER_VERSION = 1

_MAX_SEARCH_RESULTS = 200


@dataclass(frozen=True, slots=True)
class IndexedFile:
    """One file in the index: cheap metadata plus an optional analysis."""

    relative: str
    size: int
    mtime_ns: int
    language: str | None
    is_source: bool
    is_binary: bool
    analysis: FileAnalysis | None = None
    skipped: str = ""


@dataclass(frozen=True, slots=True)
class ProjectIndex:
    """An immutable snapshot of the project index."""

    root: Path
    files: Mapping[str, IndexedFile]
    created_at: float = 0.0
    updated_at: float = 0.0
    truncated: bool = False
    fingerprint: str = ""

    @property
    def source_files(self) -> tuple[IndexedFile, ...]:
        return tuple(
            file
            for file in self.files.values()
            if file.is_source and not file.is_binary
        )

    @property
    def analyzed(self) -> tuple[IndexedFile, ...]:
        return tuple(
            file for file in self.files.values() if file.analysis is not None
        )

    @property
    def symbol_count(self) -> int:
        return sum(
            len(file.analysis.symbols)
            for file in self.files.values()
            if file.analysis is not None
        )

    def symbols(self) -> Iterator[tuple[str, Symbol]]:
        for relative, file in sorted(self.files.items()):
            if file.analysis is None:
                continue
            for symbol in file.analysis.symbols:
                yield relative, symbol

    def file_analysis(self, relative: str) -> FileAnalysis | None:
        file = self.files.get(relative) or self.files.get(
            relative.replace("\\", "/")
        )
        return file.analysis if file else None

    def search(self, query: str, *, limit: int = 20) -> tuple[SearchHit, ...]:
        """Rank indexed symbols against ``query`` using lexical evidence."""
        tokens = tokenize(query)
        if not tokens:
            return ()
        limit = max(1, min(limit, _MAX_SEARCH_RESULTS))
        hits: list[tuple[float, SearchHit]] = []
        for relative, file in self.files.items():
            analysis = file.analysis
            if analysis is None:
                continue
            path_score = score_text(tokens, relative) * 0.35
            for symbol in analysis.symbols:
                name_score = score_identifiers(tokens, symbol.name)
                qualified_score = score_text(tokens, symbol.qualified_name)
                doc_score = score_text(tokens, symbol.doc)
                score = (
                    name_score * 3.0
                    + qualified_score * 1.2
                    + doc_score * 0.5
                    + path_score
                )
                if score <= 0:
                    continue
                hits.append(
                    (
                        score,
                        SearchHit(path=relative, symbol=symbol, score=score),
                    )
                )
        hits.sort(
            key=lambda item: (
                -item[0],
                item[1].path,
                item[1].symbol.line,
                item[1].symbol.name,
            )
        )
        return tuple(hit for _, hit in hits[:limit])


class IndexManager:
    """Builds, loads, caches and incrementally refreshes a :class:`ProjectIndex`."""

    def __init__(
        self,
        root: str | Path,
        *,
        limits: ScanLimits | None = None,
        cache_dir: str | Path | None = None,
        analyzer_version: int = ANALYZER_VERSION,
    ) -> None:
        self.root = Path(root).resolve()
        self.limits = limits or ScanLimits()
        self.analyzer_version = analyzer_version
        self._cache_dir = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir is not None
            else _default_cache_dir()
        )
        # A custom cache directory may live inside the workspace.  Never index
        # the index itself: that would make every refresh see a "new" file.
        self._cache_prefix = _relative_prefix(self._cache_dir, self.root)
        self._lock = threading.RLock()
        self._fingerprint = _fingerprint(self.limits, analyzer_version)

    @property
    def cache_path(self) -> Path:
        digest = hashlib.sha1(
            str(self.root).casefold().encode("utf-8", errors="replace")
        ).hexdigest()[:16]
        return self._cache_dir / f"index-v{INDEX_VERSION}-{digest}.json"

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def _is_cache_path(self, relative: str) -> bool:
        """True when ``relative`` belongs to this manager's own cache."""
        if self._cache_prefix:
            return relative == self._cache_prefix or relative.startswith(
                f"{self._cache_prefix}/"
            )
        if self._cache_dir == self.root:
            # Pathological: the workspace *is* the cache directory.  Skip only
            # the index document and its temporary siblings.
            name = relative.rsplit("/", 1)[-1]
            return name == self.cache_path.name or name.startswith(".index-")
        return False

    def load(self) -> ProjectIndex | None:
        """Load a cached index, returning ``None`` when absent or unusable."""
        with self._lock:
            path = self.cache_path
            try:
                if not path.is_file() or path.stat().st_size > 32_000_000:
                    return None
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
            if not isinstance(data, dict):
                return None
            if data.get("version") != INDEX_VERSION:
                return None
            stored_root = data.get("root")
            if not isinstance(stored_root, str) or not _same_root(
                stored_root, self.root
            ):
                return None
            files_data = data.get("files")
            if not isinstance(files_data, dict):
                return None
            files: dict[str, IndexedFile] = {}
            for relative, payload in files_data.items():
                indexed = _indexed_from_dict(relative, payload)
                if indexed is not None:
                    files[relative] = indexed
            return ProjectIndex(
                root=self.root,
                files=files,
                created_at=_float(data.get("created_at")),
                updated_at=_float(data.get("updated_at")),
                truncated=bool(data.get("truncated")),
                fingerprint=str(data.get("fingerprint") or ""),
            )

    # ------------------------------------------------------------------ build

    def refresh(self) -> tuple[ProjectIndex, IndexUpdate]:
        """Scan the workspace and incrementally update the cached index."""
        with self._lock:
            previous = self.load()
            index, update = self._build(previous)
            self._save(index)
            return index, update

    def _build(
        self, previous: ProjectIndex | None
    ) -> tuple[ProjectIndex, IndexUpdate]:
        scan = scan_project(self.root, limits=self.limits)
        previous_files: Mapping[str, IndexedFile] = (
            previous.files if previous and previous.fingerprint == self._fingerprint else {}
        )
        reusable = bool(previous_files)

        files: dict[str, IndexedFile] = {}
        added = modified = unchanged = removed = analyzed = skipped = 0

        for scanned in scan.files:
            if is_sensitive_path(scanned.relative):
                continue
            if self._is_cache_path(scanned.relative):
                continue
            relative = scanned.relative
            old = previous_files.get(relative)
            if old is not None and old.size == scanned.size and old.mtime_ns == scanned.mtime_ns:
                files[relative] = old
                unchanged += 1
                continue

            if old is not None:
                modified += 1
            else:
                added += 1

            indexed, did_analyze, was_skipped = self._index_file(scanned)
            files[relative] = indexed
            analyzed += int(did_analyze)
            skipped += int(was_skipped)

        removed = sum(1 for relative in previous_files if relative not in files)
        now = time.time()
        index = ProjectIndex(
            root=self.root,
            files=files,
            created_at=(previous.created_at if previous else now) or now,
            updated_at=now,
            truncated=scan.truncated,
            fingerprint=self._fingerprint,
        )
        update = IndexUpdate(
            added=added,
            modified=modified,
            removed=removed,
            unchanged=unchanged,
            analyzed=analyzed,
            skipped=skipped,
            total=len(files),
            truncated=scan.truncated,
        )
        return index, update

    def _inside(self, path: Path) -> bool:
        """True when ``path`` still resolves inside the workspace root.

        The scanner never follows links, but a link can be introduced between
        the scan and the read, or a parent directory can be a junction.
        """
        try:
            if path.is_symlink() or is_junction(path):
                return False
            return path.resolve().is_relative_to(self.root)
        except OSError:
            return False

    def _index_file(self, scanned: ScannedFile) -> tuple[IndexedFile, bool, bool]:
        analysis: FileAnalysis | None = None
        skipped_reason = ""
        did_analyze = False
        candidate = (
            scanned.is_source
            and not scanned.is_binary
            and scanned.size <= self.limits.max_file_bytes
            and analyzer_for(scanned.path) is not None
        )
        if candidate:
            if not self._inside(scanned.path):
                skipped_reason = "outside the workspace"
            else:
                text = read_text_limited(scanned.path, self.limits.max_file_bytes)
                if text is None:
                    skipped_reason = "unreadable or binary"
                else:
                    result = analyze_source(text, scanned.path)
                    if result is None:
                        skipped_reason = "no analyser"
                    else:
                        analysis = result
                        did_analyze = True
        elif scanned.is_source:
            if scanned.is_binary:
                skipped_reason = "binary"
            elif scanned.size > self.limits.max_file_bytes:
                skipped_reason = "exceeds size limit"
            else:
                skipped_reason = "no analyser"

        return (
            IndexedFile(
                relative=scanned.relative,
                size=scanned.size,
                mtime_ns=scanned.mtime_ns,
                language=scanned.language,
                is_source=scanned.is_source,
                is_binary=scanned.is_binary,
                analysis=analysis,
                skipped=skipped_reason,
            ),
            did_analyze,
            bool(skipped_reason),
        )

    # ------------------------------------------------------------------- save

    def _save(self, index: ProjectIndex) -> None:
        payload = {
            "version": INDEX_VERSION,
            "root": str(index.root),
            "created_at": index.created_at,
            "updated_at": index.updated_at,
            "truncated": index.truncated,
            "fingerprint": index.fingerprint,
            "files": {
                relative: _indexed_to_dict(indexed)
                for relative, indexed in sorted(index.files.items())
            },
        }
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self._cache_dir,
                prefix=".index-",
                suffix=".tmp",
                delete=False,
            )
            temporary = Path(handle.name)
            try:
                with handle:
                    json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
                os.replace(temporary, self.cache_path)
            except OSError:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        except OSError:
            # A read-only cache directory must not break analysis.
            return


def _default_cache_dir() -> Path:
    override = os.environ.get("BEAST_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".beast" / "cache"


def _relative_prefix(directory: Path, root: Path) -> str:
    """Return ``directory`` relative to ``root`` as posix text, or ``""``.

    ``""`` means the directory is the root or lies outside it; callers treat
    that as "no dedicated subtree to exclude".
    """
    try:
        relative = directory.relative_to(root)
    except ValueError:
        return ""
    posix = relative.as_posix()
    return "" if posix == "." else posix


def _fingerprint(limits: ScanLimits, analyzer_version: int) -> str:
    payload = {
        "analyzer": analyzer_version,
        "max_files": limits.max_files,
        "max_depth": limits.max_depth,
        "max_file_bytes": limits.max_file_bytes,
        "respect_gitignore": limits.respect_gitignore,
        "extra_ignore": sorted(limits.extra_ignore),
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _same_root(left: str, right: Path) -> bool:
    try:
        return Path(left).resolve() == right.resolve()
    except OSError:  # pragma: no cover - defensive
        return False


def _float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------- serialisation


def _indexed_to_dict(indexed: IndexedFile) -> dict:
    payload: dict[str, object] = {
        "size": indexed.size,
        "mtime_ns": indexed.mtime_ns,
        "language": indexed.language,
        "is_source": indexed.is_source,
        "is_binary": indexed.is_binary,
    }
    if indexed.skipped:
        payload["skipped"] = indexed.skipped
    if indexed.analysis is not None:
        payload["analysis"] = {
            "language": indexed.analysis.language,
            "lines": indexed.analysis.lines,
            "ok": indexed.analysis.ok,
            "error": indexed.analysis.error,
            "symbols": [
                {
                    "name": symbol.name,
                    "kind": str(symbol.kind),
                    "line": symbol.line,
                    "end_line": symbol.end_line,
                    "signature": symbol.signature,
                    "parent": symbol.parent,
                    "doc": symbol.doc,
                    "exported": symbol.exported,
                    "decorators": list(symbol.decorators),
                }
                for symbol in indexed.analysis.symbols
            ],
            "imports": [
                {
                    "module": imported.module,
                    "names": list(imported.names),
                    "line": imported.line,
                    "level": imported.level,
                    "is_from": imported.is_from,
                }
                for imported in indexed.analysis.imports
            ],
        }
    return payload


def _indexed_from_dict(relative: str, payload: object) -> IndexedFile | None:
    if (
        not isinstance(payload, dict)
        or not _safe_relative(relative)
        or is_sensitive_path(relative)
    ):
        return None
    try:
        size = int(payload.get("size", 0))
        mtime_ns = int(payload.get("mtime_ns", 0))
    except (TypeError, ValueError):
        return None
    language = payload.get("language")
    analysis = _analysis_from_dict(payload.get("analysis"))
    return IndexedFile(
        relative=relative,
        size=size,
        mtime_ns=mtime_ns,
        language=language if isinstance(language, str) else None,
        is_source=bool(payload.get("is_source")),
        is_binary=bool(payload.get("is_binary")),
        analysis=analysis,
        skipped=str(payload.get("skipped") or ""),
    )


def _analysis_from_dict(payload: object) -> FileAnalysis | None:
    if not isinstance(payload, dict):
        return None
    language = payload.get("language")
    if not isinstance(language, str):
        language = ""
    symbols: list[Symbol] = []
    for raw in payload.get("symbols") or []:
        symbol = _symbol_from_dict(raw)
        if symbol is not None:
            symbols.append(symbol)
    imports: list[ImportInfo] = []
    for raw in payload.get("imports") or []:
        if not isinstance(raw, dict):
            continue
        module = raw.get("module")
        if not isinstance(module, str):
            continue
        names = raw.get("names") or []
        imports.append(
            ImportInfo(
                module=module,
                names=tuple(str(name) for name in names if isinstance(name, str)),
                line=_int(raw.get("line")),
                level=_int(raw.get("level")),
                is_from=bool(raw.get("is_from")),
            )
        )
    return FileAnalysis(
        language=language,
        symbols=tuple(symbols),
        imports=tuple(imports),
        lines=_int(payload.get("lines")),
        ok=bool(payload.get("ok", True)),
        error=str(payload.get("error") or ""),
    )


def _symbol_from_dict(payload: object) -> Symbol | None:
    if not isinstance(payload, dict):
        return None
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        return None
    try:
        kind = SymbolKind(str(payload.get("kind", "")))
    except ValueError:
        return None
    parent = payload.get("parent")
    decorators = payload.get("decorators") or []
    return Symbol(
        name=name,
        kind=kind,
        line=_int(payload.get("line")),
        end_line=_int(payload.get("end_line")),
        signature=str(payload.get("signature") or ""),
        parent=parent if isinstance(parent, str) else None,
        doc=str(payload.get("doc") or ""),
        exported=bool(payload.get("exported")),
        decorators=tuple(
            str(item) for item in decorators if isinstance(item, str)
        ),
    )


def _int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _safe_relative(relative: str) -> bool:
    """Reject absolute or traversal paths that could appear in a tampered cache."""
    if (
        not relative
        or "\\" in relative
        or relative.startswith("/")
        or ":" in relative
    ):
        return False
    parts = relative.split("/")
    return all(part not in {"", ".", ".."} for part in parts)
