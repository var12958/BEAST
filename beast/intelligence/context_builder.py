"""Relevance-based project context construction.

The context engine answers one question: *which small slice of this project
should the model see for this request?*  It scores indexed paths, symbols,
imports and (for a bounded candidate pool) file content against the request,
then renders a compact, line-numbered excerpt within a character budget.

Nothing is included just because it exists.  Credential-bearing files are
excluded before scoring, and the rendered text is redacted as a final guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from beast.intelligence.index_manager import IndexedFile, ProjectIndex
from beast.intelligence.models import (
    ContextBundle,
    ProjectProfile,
    Symbol,
    SymbolKind,
)
from beast.intelligence.text import (
    match_score,
    score_identifiers,
    score_text,
    tokenize,
)
from beast.workspace.scanning import read_text_limited
from beast.workspace.secrets import is_sensitive_path, redact_secrets

#: Intents that ask for a whole-project explanation rather than one topic.
_OVERVIEW_PREFIXES = (
    "architect",
    "component",
    "dependenc",
    "overview",
    "structure",
    "technolog",
)

#: Tokens that ask for dependency information to be rendered in full.
_DEPENDENCY_PREFIXES = ("dependenc", "librar", "package")

#: Names that usually anchor an application's architecture.
_IMPORTANT_NAMES = frozenset(
    {
        "__init__.py",
        "app.py",
        "main.py",
        "routes.py",
        "models.py",
        "views.py",
        "config.py",
        "settings.py",
        "server.py",
        "api.py",
        "auth.py",
        "database.py",
        "index.ts",
        "index.tsx",
        "app.tsx",
        "main.tsx",
        "router.ts",
        "api.ts",
        "server.ts",
    }
)


@dataclass(frozen=True, slots=True)
class ContextLimits:
    """Budgets for one context bundle.  All are character/line based."""

    max_characters: int = 14_000
    max_files: int = 8
    max_file_characters: int = 4_000
    excerpt_lines: int = 60
    candidate_pool: int = 40
    max_read_bytes: int = 128_000
    symbols_per_file: int = 10
    dependency_limit: int = 24


class ContextProvider(Protocol):
    """What :class:`beast.core.agent.Agent` needs from the intelligence layer."""

    def context_for(self, query: str) -> ContextBundle:
        """Return the project context relevant to ``query``."""


class ContextBuilder:
    """Builds a bounded, evidence-linked context bundle for one request."""

    def __init__(
        self, root: str | Path, *, limits: ContextLimits | None = None
    ) -> None:
        self.root = Path(root).resolve()
        self.limits = limits or ContextLimits()

    def build(
        self,
        query: str,
        *,
        index: ProjectIndex,
        profile: ProjectProfile,
    ) -> ContextBundle:
        limits = self.limits
        tokens = tokenize(query)
        overview = not tokens or any(
            token.startswith(_OVERVIEW_PREFIXES) for token in tokens
        )
        wants_dependencies = overview or any(
            token.startswith(_DEPENDENCY_PREFIXES) for token in tokens
        )

        candidates = self._rank(index, tokens, overview=overview)
        selected = self._select_files(
            candidates,
            index=index,
            profile=profile,
            overview=overview,
            tokens=tokens,
        )

        header = self._header(
            profile,
            tokens=tokens,
            overview=overview,
            wants_dependencies=wants_dependencies,
            selected=selected,
        )
        parts: list[str] = [header]
        used = len(header)
        included: list[str] = []
        symbol_count = 0
        budget_hit = False

        if overview:
            tree = profile.structure.render_tree(max_lines=55)
            if tree:
                section = f"\n## Project structure\n```text\n{tree}\n```\n"
                parts.append(section)
                used += len(section)

        for relative in selected:
            remaining = limits.max_characters - used
            if remaining < 400:
                budget_hit = True
                break
            block, symbols = self._render_file(
                relative, index=index, tokens=tokens, overview=overview
            )
            if not block:
                continue
            if len(block) > remaining:
                block = block[: max(0, remaining - 80)].rstrip()
                block += "\n… (context budget reached)\n"
                budget_hit = True
            parts.append(block)
            used += len(block)
            included.append(relative)
            symbol_count += symbols
            if len(included) >= limits.max_files:
                budget_hit = budget_hit or len(selected) > len(included)
                break

        text = "".join(parts).strip() + "\n"
        redacted = redact_secrets(text)
        text = redacted.text
        truncated = budget_hit or index.truncated
        if truncated:
            text += (
                "\n_Context was truncated to stay within the configured budget; "
                "inspect files directly if more detail is needed._\n"
            )
        return ContextBundle(
            text=text,
            query=query,
            files=tuple(included),
            symbol_count=symbol_count,
            characters=len(text),
            truncated=truncated,
            redactions=redacted.count,
            profile=profile,
        )

    # --------------------------------------------------------------- ranking

    def _rank(
        self,
        index: ProjectIndex,
        tokens: tuple[str, ...],
        *,
        overview: bool,
    ) -> list[tuple[float, str]]:
        if not tokens:
            return []
        ranked: list[tuple[float, str]] = []
        for relative, file in index.files.items():
            if file.is_binary or is_sensitive_path(relative):
                continue
            score = score_identifiers(tokens, relative) * 0.8
            analysis = file.analysis
            if analysis is not None:
                symbol_scores: list[float] = []
                for symbol in analysis.symbols:
                    symbol_scores.append(
                        score_identifiers(tokens, symbol.name) * 2.0
                        + score_text(tokens, symbol.qualified_name) * 0.6
                        + score_text(tokens, symbol.doc) * 0.4
                    )
                symbol_scores.sort(reverse=True)
                score += min(sum(symbol_scores[:3]), 300.0)
                for imported in analysis.imports:
                    score += score_text(tokens, imported.module) * 0.3
                    score += score_text(tokens, " ".join(imported.names)) * 0.2
            if score > 0:
                ranked.append((score, relative))
        ranked.sort(key=lambda item: (-item[0], item[1]))

        # Content evidence is expensive, so only a bounded pool is read.  When
        # path/symbol/import evidence is sparse (common in small or oddly named
        # projects) the pool is topped up with source files so that text that
        # only appears in file content can still be found.
        pool = ranked[: self.limits.candidate_pool]
        if len(pool) < self.limits.candidate_pool:
            seen = {relative for _, relative in pool}
            extras = [
                relative
                for relative, file in sorted(index.files.items())
                if relative not in seen
                and file.is_source
                and not file.is_binary
                and not is_sensitive_path(relative)
            ]
            for relative in extras:
                pool.append((0.0, relative))
                if len(pool) >= self.limits.candidate_pool:
                    break

        enriched: list[tuple[float, str]] = []
        for score, relative in pool:
            text = self._read(relative)
            if text:
                score += min(_content_score(tokens, text), 90.0)
            enriched.append((score, relative))
        enriched.sort(key=lambda item: (-item[0], item[1]))
        return enriched

    def _select_files(
        self,
        ranked: list[tuple[float, str]],
        *,
        index: ProjectIndex,
        profile: ProjectProfile,
        overview: bool,
        tokens: tuple[str, ...],
    ) -> list[str]:
        selected: list[str] = []

        if overview:
            for relative in self._overview_files(index, profile):
                if relative not in selected:
                    selected.append(relative)
                if len(selected) >= self.limits.max_files:
                    break
        for _, relative in ranked:
            if len(selected) >= self.limits.max_files:
                break
            if relative not in selected:
                selected.append(relative)

        # If lexical ranking found nothing at all, fall back to the symbol index.
        if not selected and tokens:
            for hit in index.search(" ".join(tokens), limit=self.limits.max_files):
                if hit.path not in selected:
                    selected.append(hit.path)
        return selected[: self.limits.max_files]

    def _overview_files(
        self, index: ProjectIndex, profile: ProjectProfile
    ) -> list[str]:
        candidates: list[str] = []
        candidates.extend(profile.structure.docs)
        candidates.extend(
            manifest.path for manifest in profile.dependencies.manifests
        )
        candidates.extend(profile.detection.entry_points)
        candidates.extend(profile.structure.key_files)

        names = [
            relative
            for relative in sorted(index.files)
            if Path(relative).name.lower() in _IMPORTANT_NAMES
        ][:6]
        candidates.extend(names)

        important_names = [
            relative
            for relative, file in sorted(index.files.items())
            if file.analysis is not None
        ]
        important_names.sort(key=lambda relative: -len(index.files[relative].analysis.symbols))
        candidates.extend(important_names[:6])

        valid: list[str] = []
        for relative in candidates:
            file = index.files.get(relative)
            if file is None or file.is_binary or is_sensitive_path(relative):
                continue
            if not self._read(relative):
                continue
            if relative not in valid:
                valid.append(relative)
        return valid[: self.limits.max_files]

    # ---------------------------------------------------------------- render

    def _header(
        self,
        profile: ProjectProfile,
        *,
        tokens: tuple[str, ...],
        overview: bool,
        wants_dependencies: bool,
        selected: list[str],
    ) -> str:
        detection = profile.detection
        structure = profile.structure
        dependencies = profile.dependencies
        lines = [
            "## Project context",
            "This section was gathered automatically from project files. "
            "Use it as evidence, and inspect files directly when you need more.",
            "",
            f"Project: {detection.name}",
            f"Languages: {', '.join(detection.language_names) or 'none detected'}",
            f"Frameworks: {', '.join(detection.frameworks) or 'none detected'}",
            f"Package managers: {', '.join(detection.package_managers) or 'none detected'}",
            f"Build systems: {', '.join(detection.build_systems) or 'none detected'}",
            f"Test frameworks: {', '.join(detection.test_frameworks) or 'none detected'}",
            f"Source files: {detection.source_file_count}; "
            f"test files: {detection.test_file_count}",
        ]
        if structure.source_dirs:
            lines.append(f"Source directories: {', '.join(structure.source_dirs)}")
        if structure.test_dirs:
            lines.append(f"Test directories: {', '.join(structure.test_dirs)}")
        if detection.entry_points:
            lines.append(f"Entry points: {', '.join(detection.entry_points[:8])}")
        if detection.manifests:
            lines.append(
                "Manifests: "
                + ", ".join(
                    f"{manifest.path} ({manifest.kind})"
                    for manifest in detection.manifests[:8]
                )
            )
        if selected:
            lines.append("Relevant files: " + ", ".join(selected))
        lines.append("")

        if wants_dependencies:
            relevant = _relevant_dependencies(profile, tokens, overview)
            if relevant:
                lines.append("### Dependencies")
                lines.append(relevant)
                lines.append("")
        return "\n".join(lines)

    def _render_file(
        self,
        relative: str,
        *,
        index: ProjectIndex,
        tokens: tuple[str, ...],
        overview: bool,
    ) -> tuple[str, int]:
        text = self._read(relative)
        if text is None:
            return "", 0
        indexed = index.files.get(relative)
        language = indexed.language if indexed else None
        lines = text.count("\n") + 1
        heading = f"### `{relative}`"
        if language:
            heading += f" — {language}, {lines} lines"
        parts = [f"\n{heading}\n"]

        symbol_count = 0
        if indexed is not None and indexed.analysis is not None:
            summary, symbol_count = self._symbol_summary(
                indexed, tokens=tokens, overview=overview
            )
            if summary:
                parts.append(summary + "\n")
            imports = [imported.display() for imported in indexed.analysis.imports]
            if imports:
                parts.append("Imports: " + ", ".join(imports[:10]) + "\n")

        excerpt = _excerpt(
            text,
            tokens,
            line_count=self.limits.excerpt_lines,
            max_characters=self.limits.max_file_characters,
        )
        fence = "````" if "```" in excerpt else "```"
        code_language = (language or "").lower().replace("#", "sharp").replace("+", "p")
        parts.append(f"{fence}{code_language}\n{excerpt}\n{fence}\n")
        return "".join(parts), symbol_count

    def _symbol_summary(
        self,
        indexed: IndexedFile,
        *,
        tokens: tuple[str, ...],
        overview: bool,
    ) -> tuple[str, int]:
        analysis = indexed.analysis
        if analysis is None or not analysis.symbols:
            return "", 0
        scored: list[tuple[float, Symbol]] = []
        for symbol in analysis.symbols:
            score = score_identifiers(tokens, symbol.name) if tokens else 0.0
            score += score_text(tokens, symbol.qualified_name) * 0.3 if tokens else 0.0
            if overview:
                score += 1.0 if symbol.kind in {SymbolKind.CLASS, SymbolKind.FUNCTION} else 0.5
            scored.append((score, symbol))
        scored.sort(
            key=lambda item: (
                -item[0],
                item[1].line,
                item[1].name,
            )
        )
        chosen = scored[: self.limits.symbols_per_file]
        if not chosen:
            return "", 0
        lines = ["Symbols:"]
        for _, symbol in chosen:
            label = symbol.display
            if len(label) > 150:
                label = label[:147] + "..."
            doc = f" — {symbol.doc}" if symbol.doc else ""
            lines.append(
                f"- {symbol.kind.value} `{label}` (line {symbol.line}){doc}"
            )
        return "\n".join(lines), len(chosen)

    # ------------------------------------------------------------------ read

    def _read(self, relative: str) -> str | None:
        if is_sensitive_path(relative):
            return None
        try:
            candidate = (self.root / relative).resolve()
            candidate.relative_to(self.root)
        except (OSError, ValueError):
            return None
        if not candidate.is_file():
            return None
        return read_text_limited(candidate, self.limits.max_read_bytes)


def _content_score(tokens: tuple[str, ...], text: str) -> float:
    folded = text.casefold()
    score = 0.0
    for token in tokens:
        score += min(folded.count(token), 8) * 3.0
        if len(token) >= 6:
            score += min(folded.count(token[:5]), 8) * 1.2
        if len(token) >= 5:
            score += min(folded.count(token[:4]), 8) * 0.5
    return score


def _relevant_dependencies(
    profile: ProjectProfile, tokens: tuple[str, ...], overview: bool
) -> str:
    report = profile.dependencies
    if not report.dependencies:
        return "None declared."
    matched = [
        dependency
        for dependency in report.dependencies
        if not tokens
        or any(
            match_score(token, dependency.name) > 0
            or score_identifiers((token,), dependency.name) > 0
            for token in tokens
        )
    ]
    if not overview:
        if not matched:
            return ""
        return "\n".join(
            f"- {dependency.display()} ({dependency.scope})"
            + (f" — {dependency.source}" if dependency.source else "")
            for dependency in matched[:12]
        )
    return report.render(limit=20)


def _excerpt(
    text: str,
    tokens: tuple[str, ...],
    *,
    line_count: int,
    max_characters: int,
) -> str:
    lines = text.splitlines()
    if not lines:
        return ""
    anchor = 0
    if tokens:
        best = 0.0
        for index, line in enumerate(lines):
            if len(line) > 2_000:
                continue
            folded = line.casefold()
            score = sum(1.0 for token in tokens if token in folded)
            if score > best:
                best = score
                anchor = index
        if best == 0:
            anchor = 0
    start = max(0, anchor - line_count // 3)
    end = min(len(lines), start + line_count)
    if end - start < line_count and start > 0:
        start = max(0, end - line_count)
    numbered = [
        f"{line_number + 1:>5} | {lines[line_number]}"
        for line_number in range(start, end)
    ]
    body = "\n".join(numbered)
    prefix = f"... lines 1-{start} omitted ...\n" if start > 0 else ""
    suffix = (
        f"\n... lines {end + 1}-{len(lines)} omitted ..."
        if end < len(lines)
        else ""
    )
    excerpt = prefix + body + suffix
    if len(excerpt) > max_characters:
        excerpt = excerpt[: max_characters - 24].rstrip() + "\n... excerpt truncated ..."
    return excerpt
