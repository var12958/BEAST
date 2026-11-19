"""Language-aware source analysis.

Python is parsed with the standard-library :mod:`ast` module: classes,
functions, methods, imports, signatures, decorators and module constants come
from the real syntax tree, never from regular expressions.

JavaScript/TypeScript have no parser in BEAST's dependency set, so they use a
dedicated structural analyser: strings, template literals and comments are
masked out (preserving offsets and line numbers), brace depth is computed, and
declarations are recognised in the masked view.  This is deliberately not
regex-only — the masking step prevents commented-out or quoted text from being
mistaken for code — but it is not a full ECMAScript parser either.  It covers
the declarations BEAST needs for retrieval: classes, methods, functions,
exported arrow functions, interfaces, type aliases, enums and imports.

Adding a language means writing one class with ``language``, ``extensions``
and ``analyze()``, then registering it; the index, context builder and tools
need no changes.
"""

from __future__ import annotations

import ast
import bisect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from beast.intelligence.models import (
    FileAnalysis,
    ImportInfo,
    Symbol,
    SymbolKind,
)

_IDENTIFIER = re.compile(r"^[A-Za-z_$][\w$]*$")
_CONSTANT_NAME = re.compile(r"^[A-Z][A-Z0-9_]{1,}$")


class LanguageAnalyzer(Protocol):
    """One analyser per language family."""

    language: str
    extensions: frozenset[str]

    def analyze(self, text: str, path: str) -> FileAnalysis:
        """Extract declarations and imports from ``text``."""
        ...


# ------------------------------------------------------------------------- Python


def _unparse(node: ast.AST | None, limit: int = 160) -> str:
    if node is None:
        return ""
    try:
        value = ast.unparse(node)
    except Exception:  # pragma: no cover - ast.unparse is very reliable
        return "…"
    value = " ".join(value.split())
    return value if len(value) <= limit else f"{value[: limit - 1]}…"


def _doc_first_line(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    try:
        doc = ast.get_docstring(node, clean=True)
    except Exception:  # pragma: no cover - defensive
        return ""
    if not doc:
        return ""
    line = doc.strip().splitlines()[0].strip()
    return line if len(line) <= 200 else f"{line[:199]}…"


def _python_function_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    keyword = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    arguments = _unparse(node.args, limit=300)
    returns = f" -> {_unparse(node.returns, limit=100)}" if node.returns else ""
    signature = f"{keyword} {node.name}({arguments}){returns}"
    if len(signature) > 180:
        signature = f"{signature[:179]}…"
    return signature


def _python_class_signature(node: ast.ClassDef) -> str:
    bases = [_unparse(base, limit=80) for base in node.bases]
    keywords = [
        f"{keyword.arg}={_unparse(keyword.value, limit=60)}"
        for keyword in node.keywords
        if keyword.arg
    ]
    arguments = ", ".join([*bases, *keywords])
    return f"class {node.name}({arguments})" if arguments else f"class {node.name}"


def _decorators(node: ast.AST) -> tuple[str, ...]:
    decorators: list[str] = []
    for decorator in getattr(node, "decorator_list", ()) or ():
        text = _unparse(decorator, limit=80)
        decorators.append(text if text.startswith("@") else f"@{text}")
    return tuple(decorators)


class PythonAnalyzer:
    """AST-based analysis for Python (``.py`` and ``.pyi``)."""

    language = "Python"
    extensions = frozenset({".py", ".pyi"})

    def analyze(self, text: str, path: str) -> FileAnalysis:
        line_count = len(text.splitlines())
        try:
            tree = ast.parse(text, filename=path or "<unknown>")
        except (SyntaxError, ValueError) as exc:
            return FileAnalysis(
                language=self.language,
                lines=line_count,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

        visitor = _PythonVisitor()
        visitor.visit(tree)
        return FileAnalysis(
            language=self.language,
            symbols=tuple(visitor.symbols),
            imports=tuple(visitor.imports),
            lines=line_count,
        )


class _PythonVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.symbols: list[Symbol] = []
        self.imports: list[ImportInfo] = []
        self._stack: list[tuple[str, SymbolKind]] = []

    def _parent(self) -> str | None:
        return ".".join(name for name, _ in self._stack) or None

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(
                ImportInfo(
                    module=alias.name,
                    names=((f"{alias.name} as {alias.asname}",) if alias.asname else ()),
                    line=getattr(node, "lineno", 0),
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.imports.append(
            ImportInfo(
                module=node.module or "",
                names=tuple(
                    alias.name if not alias.asname else f"{alias.name} as {alias.asname}"
                    for alias in node.names
                    if alias.name != "*"
                ),
                line=getattr(node, "lineno", 0),
                level=node.level or 0,
                is_from=True,
            )
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        parent = self._parent()
        self.symbols.append(
            Symbol(
                name=node.name,
                kind=SymbolKind.CLASS,
                line=node.lineno,
                end_line=getattr(node, "end_lineno", node.lineno) or node.lineno,
                signature=_python_class_signature(node),
                parent=parent,
                doc=_doc_first_line(node),
                exported=not node.name.startswith("_"),
                decorators=_decorators(node),
            )
        )
        self._stack.append((node.name, SymbolKind.CLASS))
        for child in node.body:
            self.visit(child)
        self._stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node)

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        parent_kind = self._stack[-1][1] if self._stack else None
        kind = SymbolKind.METHOD if parent_kind is SymbolKind.CLASS else SymbolKind.FUNCTION
        self.symbols.append(
            Symbol(
                name=node.name,
                kind=kind,
                line=node.lineno,
                end_line=getattr(node, "end_lineno", node.lineno) or node.lineno,
                signature=_python_function_signature(node),
                parent=self._parent(),
                doc=_doc_first_line(node),
                exported=not node.name.startswith("_"),
                decorators=_decorators(node),
            )
        )
        self._stack.append((node.name, kind))
        for child in node.body:
            self.visit(child)
        self._stack.pop()

    # Module-level constants are useful anchors when reading a project.
    def visit_Assign(self, node: ast.Assign) -> None:
        if not self._stack:
            for target in node.targets:
                if isinstance(target, ast.Name) and _CONSTANT_NAME.match(target.id):
                    value = _unparse(node.value, limit=80)
                    self.symbols.append(
                        Symbol(
                            name=target.id,
                            kind=SymbolKind.CONSTANT,
                            line=node.lineno,
                            end_line=getattr(node, "end_lineno", node.lineno) or node.lineno,
                            signature=f"{target.id} = {value}",
                            exported=not target.id.startswith("_"),
                        )
                    )
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if (
            not self._stack
            and isinstance(node.target, ast.Name)
            and _CONSTANT_NAME.match(node.target.id)
        ):
            annotation = _unparse(node.annotation, limit=80)
            self.symbols.append(
                Symbol(
                    name=node.target.id,
                    kind=SymbolKind.CONSTANT,
                    line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno) or node.lineno,
                    signature=f"{node.target.id}: {annotation}",
                    exported=not node.target.id.startswith("_"),
                )
            )
        self.generic_visit(node)


# --------------------------------------------------------- JavaScript / TypeScript

#: Words that look like calls at the start of a line but are never methods.
_CONTROL_WORDS = frozenset(
    {
        "if",
        "for",
        "while",
        "switch",
        "catch",
        "with",
        "function",
        "return",
        "new",
        "else",
        "do",
        "try",
        "await",
        "yield",
        "typeof",
        "delete",
        "void",
        "in",
        "of",
        "instanceof",
        "throw",
        "case",
        "break",
        "continue",
    }
)

_CLASS = re.compile(
    r"\bclass\s+([A-Za-z_$][\w$]*)(?:\s+extends\s+([A-Za-z_$][\w$.]*))?"
)
_FUNCTION = re.compile(
    r"(?m)^[ \t]*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
    r"function\s*\*?\s*([A-Za-z_$][\w$]*)\s*\("
)
_METHOD = re.compile(
    r"(?m)^[ \t]*(?:(?:public|private|protected|static|async|readonly|override|"
    r"abstract|declare|get|set|accessor)\s+)*([A-Za-z_$][\w$]*)\s*\??\s*\("
)
_CLASS_ARROW = re.compile(
    r"(?m)^[ \t]*(?:(?:static|readonly|public|private|protected|override)\s+)*"
    r"([A-Za-z_$][\w$]*)\s*(?::\s*[^=\n]+)?=\s*(?:async\s+)?"
    r"(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*(?::\s*[^=\n]+)?=>"
)
_ARROW_ASSIGN = re.compile(
    r"(?m)^[ \t]*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)"
    r"\s*(?::\s*[^=\n]+)?=\s*(?:async\s+)?"
    r"(?:function\s*\*?\s*[A-Za-z_$][\w$]*\s*)?"
    r"(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*(?::\s*[^=\n]+)?=>"
)
_TYPE_DECL = re.compile(
    r"(?m)^[ \t]*(?:export\s+)?(?:default\s+)?(?:declare\s+)?"
    r"(interface|type|enum)\s+([A-Za-z_$][\w$]*)"
)
_CONSTANT = re.compile(
    r"(?m)^[ \t]*(?:export\s+)?(?:const|let|var)\s+([A-Z][A-Z0-9_]*)\s*="
)
_IMPORT_FROM = re.compile(
    r"(?m)^[ \t]*(?:import|export)\s+(?:type\s+)?([^;\n]*?)\s+from\s+['\"]([^'\"]+)['\"]"
)
_IMPORT_SIDE_EFFECT = re.compile(
    r"(?m)^[ \t]*import\s+['\"]([^'\"]+)['\"]"
)
_REQUIRE = re.compile(r"\brequire\s*\(\s*['\"]([^'\"]+)['\"]\s*\)")
_DYNAMIC_IMPORT = re.compile(r"\bimport\s*\(\s*['\"]([^'\"]+)['\"]\s*\)")

_TYPE_KINDS = {
    "interface": SymbolKind.INTERFACE,
    "type": SymbolKind.TYPE,
    "enum": SymbolKind.ENUM,
}


def _mask_javascript(text: str) -> str:
    """Replace comments and string contents with spaces, keeping offsets."""
    characters = list(text)
    length = len(text)
    index = 0
    while index < length:
        char = text[index]
        if char == "/" and index + 1 < length:
            following = text[index + 1]
            if following == "/":
                cursor = index
                while cursor < length and text[cursor] not in "\r\n":
                    characters[cursor] = " "
                    cursor += 1
                index = cursor
                continue
            if following == "*":
                cursor = index
                while cursor < length:
                    if (
                        text[cursor] == "*"
                        and cursor + 1 < length
                        and text[cursor + 1] == "/"
                    ):
                        characters[cursor] = " "
                        characters[cursor + 1] = " "
                        cursor += 2
                        break
                    if text[cursor] not in "\r\n":
                        characters[cursor] = " "
                    cursor += 1
                index = cursor
                continue
        if char in "'\"`":
            # An apostrophe inside JSX text ("Don't") is not a string start.
            if (
                char == "'"
                and index > 0
                and (text[index - 1].isalnum() or text[index - 1] in "_$")
            ):
                index += 1
                continue
            quote = char
            cursor = index
            characters[cursor] = " "
            cursor += 1
            while cursor < length:
                current = text[cursor]
                if current == "\\":
                    characters[cursor] = " "
                    if cursor + 1 < length:
                        characters[cursor + 1] = " "
                    cursor += 2
                    continue
                characters[cursor] = " " if current not in "\r\n" else current
                if current == quote:
                    cursor += 1
                    break
                cursor += 1
            index = cursor
            continue
        index += 1
    return "".join(characters)


def _brace_depths(masked: str) -> list[int]:
    depths = [0] * (len(masked) + 1)
    depth = 0
    for index, char in enumerate(masked):
        depths[index] = depth
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(depth - 1, 0)
    depths[len(masked)] = depth
    return depths


def _matching(masked: str, start: int, opening: str, closing: str) -> int | None:
    if start >= len(masked) or masked[start] != opening:
        return None
    depth = 0
    for index in range(start, len(masked)):
        char = masked[index]
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index
    return None


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for match in re.finditer(r"\n", text):
        starts.append(match.end())
    return starts


def _line_of(starts: Sequence[int], position: int) -> int:
    return bisect.bisect_right(starts, position)


def _compact(value: str, limit: int = 180) -> str:
    value = " ".join(value.split())
    return value if len(value) <= limit else f"{value[: limit - 1]}…"


def _signature_after(
    masked: str, open_paren: int
) -> tuple[str, int | None, int | None]:
    """Return ``(signature, close_paren, body_open)`` for a declaration."""
    close = _matching(masked, open_paren, "(", ")")
    if close is None:
        return "", None, None
    signature = masked[open_paren : close + 1]
    cursor = close + 1
    while cursor < len(masked) and masked[cursor] in " \t\r\n?":
        cursor += 1
    if cursor < len(masked) and masked[cursor] == ":":
        end = cursor
        while end < len(masked) and masked[end] not in "{;":
            end += 1
        signature += masked[close + 1 : end].rstrip()
    body_open = None
    probe = close + 1
    while probe < len(masked) and masked[probe] not in "{;":
        probe += 1
    if probe < len(masked) and masked[probe] == "{":
        body_open = probe
    return _compact(signature), close, body_open


@dataclass(frozen=True, slots=True)
class _ClassRange:
    name: str
    base: str
    open_brace: int
    close_brace: int
    body_depth: int


def _import_names(clause: str) -> tuple[str, ...]:
    names: list[str] = []
    brace = re.search(r"\{([^}]*)\}", clause)
    if brace:
        for part in brace.group(1).split(","):
            cleaned = part.strip()
            if cleaned:
                names.append(cleaned)
    outside = re.sub(r"\{[^}]*\}", "", clause)
    for part in outside.split(","):
        cleaned = part.strip()
        if cleaned and cleaned not in {"type", "*"}:
            names.append(cleaned)
    cleaned_names = tuple(
        name[len("type ") :] if name.startswith("type ") else name for name in names
    )
    return cleaned_names


def _exported_at(masked: str, start: int, end: int) -> bool:
    """True when the declaration starting at ``start`` carries ``export``.

    Some patterns consume the ``export`` keyword, others start at the
    declaration itself, so both the current line's prefix and the matched
    text are checked.
    """
    line_start = masked.rfind("\n", 0, start) + 1
    return "export" in masked[line_start:start] or "export" in masked[start:end]


class JavaScriptAnalyzer:
    """Structural analysis for JavaScript, JSX, TypeScript and TSX."""

    language = "JavaScript"
    extensions = frozenset(
        {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"}
    )

    def analyze(self, text: str, path: str) -> FileAnalysis:
        language = "TypeScript" if Path(path).suffix.lower() in {
            ".ts",
            ".tsx",
            ".mts",
            ".cts",
        } else "JavaScript"
        line_count = len(text.splitlines())
        masked = _mask_javascript(text)
        depths = _brace_depths(masked)
        starts = _line_starts(text)

        symbols: list[Symbol] = []
        imports: list[ImportInfo] = []
        seen: set[tuple[str, int, SymbolKind]] = set()

        def add(symbol: Symbol) -> None:
            key = (symbol.qualified_name, symbol.line, symbol.kind)
            if key in seen:
                return
            seen.add(key)
            symbols.append(symbol)

        # Class bodies, used to attach methods and to bound arrow properties.
        classes: list[_ClassRange] = []
        for match in _CLASS.finditer(masked):
            name = match.group(1)
            brace = masked.find("{", match.end())
            close = _matching(masked, brace, "{", "}") if brace != -1 else None
            if brace == -1 or close is None:
                continue
            line = _line_of(starts, match.start())
            classes.append(
                _ClassRange(
                    name=name,
                    base=match.group(2) or "",
                    open_brace=brace,
                    close_brace=close,
                    body_depth=depths[brace] + 1,
                )
            )
            add(
                Symbol(
                    name=name,
                    kind=SymbolKind.CLASS,
                    line=line,
                    end_line=_line_of(starts, close),
                    signature=f"class {name}" + (f" extends {match.group(2)}" if match.group(2) else ""),
                    exported=_exported_at(masked, match.start(), match.end()),
                )
            )

        def enclosing(position: int) -> _ClassRange | None:
            for candidate in classes:
                if (
                    candidate.open_brace < position < candidate.close_brace
                    and depths[position] == candidate.body_depth
                ):
                    return candidate
            return None

        # Methods declared in a class body.
        for match in _METHOD.finditer(masked):
            name = match.group(1)
            if name in _CONTROL_WORDS:
                continue
            owner = enclosing(match.start())
            if owner is None:
                continue
            open_paren = masked.rfind("(", match.start(), match.end())
            if open_paren == -1:
                continue
            signature, _, body_open = _signature_after(masked, open_paren)
            if body_open is None:
                continue
            end_line = _line_of(starts, body_open)
            close = _matching(masked, body_open, "{", "}")
            if close is not None:
                end_line = _line_of(starts, close)
            add(
                Symbol(
                    name=name,
                    kind=SymbolKind.METHOD,
                    line=_line_of(starts, match.start()),
                    end_line=end_line,
                    signature=f"{name}{signature}",
                    parent=owner.name,
                    exported=not name.startswith("_"),
                )
            )

        # Arrow-function class fields.
        for match in _CLASS_ARROW.finditer(masked):
            owner = enclosing(match.start())
            if owner is None:
                continue
            name = match.group(1)
            add(
                Symbol(
                    name=name,
                    kind=SymbolKind.METHOD,
                    line=_line_of(starts, match.start()),
                    end_line=_line_of(starts, match.end()),
                    signature=_compact(match.group(0).strip()),
                    parent=owner.name,
                    exported=not name.startswith("_"),
                )
            )

        # Top-level function declarations.
        for match in _FUNCTION.finditer(masked):
            if enclosing(match.start()) is not None:
                continue
            name = match.group(1)
            open_paren = masked.rfind("(", match.start(), match.end())
            signature, _, body_open = _signature_after(masked, open_paren)
            if body_open is None:
                tail = masked[match.end() : match.end() + 200]
                if ";" not in tail.split("\n", 1)[0]:
                    continue
            end_line = _line_of(starts, match.start())
            if body_open is not None:
                close = _matching(masked, body_open, "{", "}")
                end_line = _line_of(starts, close) if close is not None else end_line
            add(
                Symbol(
                    name=name,
                    kind=SymbolKind.FUNCTION,
                    line=_line_of(starts, match.start()),
                    end_line=end_line,
                    signature=f"function {name}{signature}",
                    exported=_exported_at(masked, match.start(), match.end()),
                )
            )

        # Module-level arrow functions assigned to a name.
        for match in _ARROW_ASSIGN.finditer(masked):
            if enclosing(match.start()) is not None:
                continue
            name = match.group(1)
            add(
                Symbol(
                    name=name,
                    kind=SymbolKind.FUNCTION,
                    line=_line_of(starts, match.start()),
                    end_line=_line_of(starts, match.end()),
                    signature=_compact(match.group(0).strip()),
                    exported=_exported_at(masked, match.start(), match.end()),
                )
            )

        # TypeScript interfaces, type aliases and enums.
        for match in _TYPE_DECL.finditer(masked):
            kind_word = match.group(1)
            name = match.group(2)
            end_line = _line_of(starts, match.start())
            brace = masked.find("{", match.end())
            if brace != -1:
                close = _matching(masked, brace, "{", "}")
                if close is not None:
                    end_line = _line_of(starts, close)
            add(
                Symbol(
                    name=name,
                    kind=_TYPE_KINDS[kind_word],
                    line=_line_of(starts, match.start()),
                    end_line=end_line,
                    signature=_compact(masked[match.start() : match.end()]),
                    exported=_exported_at(masked, match.start(), match.end()),
                )
            )

        # Module-level SCREAMING_CASE constants.
        for match in _CONSTANT.finditer(masked):
            name = match.group(1)
            add(
                Symbol(
                    name=name,
                    kind=SymbolKind.CONSTANT,
                    line=_line_of(starts, match.start()),
                    end_line=_line_of(starts, match.start()),
                    signature=_compact(masked[match.start() : match.end() + 60].strip()),
                    exported=_exported_at(masked, match.start(), match.end()),
                )
            )

        # Imports run against the original text so the module string survives.
        for match in _IMPORT_FROM.finditer(text):
            if not _declaration_at(masked, match.start(), ("import", "export")):
                continue
            imports.append(
                ImportInfo(
                    module=match.group(2),
                    names=_import_names(match.group(1)),
                    line=_line_of(starts, match.start()),
                    is_from=True,
                )
            )
        for match in _IMPORT_SIDE_EFFECT.finditer(text):
            if not _declaration_at(masked, match.start(), ("import",)):
                continue
            imports.append(
                ImportInfo(
                    module=match.group(1),
                    line=_line_of(starts, match.start()),
                    is_from=True,
                )
            )
        for pattern in (_REQUIRE, _DYNAMIC_IMPORT):
            for match in pattern.finditer(text):
                imports.append(
                    ImportInfo(
                        module=match.group(1),
                        line=_line_of(starts, match.start()),
                    )
                )

        symbols.sort(key=lambda symbol: (symbol.line, symbol.name))
        return FileAnalysis(
            language=language,
            symbols=tuple(symbols),
            imports=tuple(imports),
            lines=line_count,
        )


def _declaration_at(masked: str, position: int, words: Sequence[str]) -> bool:
    start = position
    while start < len(masked) and masked[start] in " \t":
        start += 1
    return any(masked.startswith(word, start) for word in words)


# ------------------------------------------------------------------------ registry

_ANALYZERS: list[LanguageAnalyzer] = [PythonAnalyzer(), JavaScriptAnalyzer()]
_BY_EXTENSION: dict[str, LanguageAnalyzer] = {}


def register_analyzer(analyzer: LanguageAnalyzer) -> None:
    """Make ``analyzer`` available for its extensions (idempotent)."""
    for extension in analyzer.extensions:
        _BY_EXTENSION[extension.lower()] = analyzer
    if analyzer not in _ANALYZERS:
        _ANALYZERS.append(analyzer)


for _analyzer in _ANALYZERS:
    for _extension in _analyzer.extensions:
        _BY_EXTENSION[_extension.lower()] = _analyzer


def analyzer_for(path: str | Path) -> LanguageAnalyzer | None:
    return _BY_EXTENSION.get(Path(path).suffix.lower())


def supported_extensions() -> frozenset[str]:
    return frozenset(_BY_EXTENSION)


def supported_languages() -> tuple[str, ...]:
    return tuple(analyzer.language for analyzer in _ANALYZERS)


def analyze_source(text: str, path: str | Path) -> FileAnalysis | None:
    """Analyse ``text`` when a language analyser handles ``path``."""
    analyzer = analyzer_for(path)
    if analyzer is None:
        return None
    return analyzer.analyze(text, str(path))
