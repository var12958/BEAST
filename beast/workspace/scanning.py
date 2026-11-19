"""Filesystem scanning shared by the file tools and the intelligence layer.

Everything that decides *which* files belong to a project lives here:

* generated directories that are never source (``.git``, ``node_modules`` …),
* ``.gitignore`` rules, including negation,
* binary/generated file extensions,
* a single depth- and count-limited walker.

Keeping this module in :mod:`beast.workspace` (rather than in the tools or the
intelligence package) means both layers can depend on it without an import
cycle.  The walker never reads file contents; callers that need text read a
bounded prefix themselves and check for NUL bytes.
"""

from __future__ import annotations

import fnmatch
import itertools
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

#: Directories that never contain project source worth analysing.
IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".idea",
        ".vscode",
        ".venv",
        "venv",
        "env",
        ".tox",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "dist",
        "build",
        "target",
        ".next",
        "site-packages",
        # BEAST and common tool caches.
        ".beast",
        ".cache",
        "htmlcov",
        ".coverage",
        ".parcel-cache",
        ".turbo",
        ".nuxt",
        ".svelte-kit",
        ".dart_tool",
        ".gradle",
        ".angular",
        ".terraform",
        "bower_components",
        "jspm_packages",
        ".yarn",
        ".pnpm-store",
        ".sass-cache",
        "Pods",
        "DerivedData",
    }
)

#: Case-folded view of :data:`IGNORED_DIRS`.  Windows and macOS file systems
#: are case-insensitive, so ``NODE_MODULES`` must be ignored just like
#: ``node_modules``.
IGNORED_DIRS_LOWER = frozenset(name.lower() for name in IGNORED_DIRS)

#: Files that are generated or binary and are never parsed as source.
IGNORED_FILE_SUFFIXES = frozenset(
    {
        ".pyc",
        ".pyo",
        ".pyd",
        ".class",
        ".jar",
        ".war",
        ".ear",
        ".o",
        ".obj",
        ".a",
        ".lib",
        ".so",
        ".dylib",
        ".dll",
        ".exe",
        ".bin",
        ".wasm",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".ico",
        ".webp",
        ".svgz",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".wav",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
        ".eot",
        ".sqlite",
        ".db",
        ".db3",
        ".lockb",
        ".map",
        ".min.js",
        ".min.css",
    }
)

#: Extension → human language name.  Only files with an entry are "source".
LANGUAGES: dict[str, str] = {
    ".py": "Python",
    ".pyi": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".mjs": "JavaScript",
    ".cjs": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".mts": "TypeScript",
    ".cts": "TypeScript",
    ".java": "Java",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".scala": "Scala",
    ".groovy": "Groovy",
    ".c": "C",
    ".h": "C",
    ".cpp": "C++",
    ".cc": "C++",
    ".cxx": "C++",
    ".hpp": "C++",
    ".hh": "C++",
    ".hxx": "C++",
    ".cs": "C#",
    ".go": "Go",
    ".rs": "Rust",
    ".rb": "Ruby",
    ".php": "PHP",
    ".swift": "Swift",
    ".dart": "Dart",
    ".lua": "Lua",
    ".r": "R",
    ".pl": "Perl",
    ".pm": "Perl",
    ".ex": "Elixir",
    ".exs": "Elixir",
    ".erl": "Erlang",
    ".hrl": "Erlang",
    ".hs": "Haskell",
    ".ml": "OCaml",
    ".mli": "OCaml",
    ".clj": "Clojure",
    ".cljs": "Clojure",
    ".vue": "Vue",
    ".svelte": "Svelte",
    ".astro": "Astro",
    ".sql": "SQL",
    ".sh": "Shell",
    ".bash": "Shell",
    ".zsh": "Shell",
    ".ps1": "PowerShell",
    ".html": "HTML",
    ".htm": "HTML",
    ".css": "CSS",
    ".scss": "SCSS",
    ".sass": "Sass",
    ".less": "Less",
}

#: Extensions treated as source when counting languages and symbols.
SOURCE_EXTENSIONS = frozenset(LANGUAGES)

#: Default limits.  They are intentionally conservative: a monorepo should
#: produce a useful overview quickly rather than a perfect one slowly.
DEFAULT_MAX_FILES = 5_000
DEFAULT_MAX_DEPTH = 15
DEFAULT_MAX_FILE_BYTES = 512_000


def language_for_path(path: str | Path) -> str | None:
    """Return the language for ``path``'s extension, or ``None``."""
    return LANGUAGES.get(Path(path).suffix.lower())


def is_source_path(path: str | Path) -> bool:
    """True when ``path`` has a recognised programming-language extension."""
    return language_for_path(path) is not None


def is_binary_path(path: str | Path) -> bool:
    """True when ``path`` is a known binary or generated artefact."""
    name = Path(path).name.lower()
    suffix = Path(path).suffix.lower()
    if suffix in IGNORED_FILE_SUFFIXES:
        return True
    if name.endswith(".min.js") or name.endswith(".min.css"):
        return True
    return False


def looks_binary(data: bytes, *, sample: int = 8_192) -> bool:
    """A NUL byte in the first ``sample`` bytes is a reliable binary signal."""
    return b"\x00" in data[:sample]


# --------------------------------------------------------------- classification

#: Directory names that conventionally contain tests, at any depth.
TEST_DIR_NAMES = frozenset(
    {
        "test",
        "tests",
        "testing",
        "__tests__",
        "spec",
        "specs",
        "e2e",
        "cypress",
        "playwright",
        "integration_tests",
        "unit_tests",
        "functional_tests",
        "acceptance_tests",
    }
)

_DOC_DIR_NAMES = frozenset(
    {"docs", "doc", "documentation", "wiki", "guides", "guide", "manual"}
)
#: Public alias used by the structure analyzer.
DOC_DIR_NAMES = _DOC_DIR_NAMES
_DOC_STEMS = frozenset(
    {
        "readme",
        "changelog",
        "changes",
        "contributing",
        "architecture",
        "license",
        "licence",
        "authors",
        "code_of_conduct",
        "security",
        "upgrading",
    }
)

_TEST_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^test_.*\.py$", re.IGNORECASE),
    re.compile(r".*_tests?\.py$", re.IGNORECASE),
    re.compile(r"^conftest\.py$", re.IGNORECASE),
    re.compile(r".*\.(?:test|spec)\.[cm]?[jt]sx?$", re.IGNORECASE),
    re.compile(r".*(?:Test|Tests|IT)\.(?:java|kt|cs|fs|swift)$"),
    re.compile(r".*_test\.go$"),
    re.compile(r".*_(?:spec|test)\.rb$", re.IGNORECASE),
    re.compile(r".*Tests?\.php$", re.IGNORECASE),
    re.compile(r"^test_.*\.(?:c|cc|cpp|cs|go|rs|rb|php|swift|dart|java|kt)$"),
    re.compile(r".*_(?:test|tests)\.(?:c|cc|cpp|cs|rs)$", re.IGNORECASE),
)

#: Configuration files that are useful to mention to the model.
CONFIG_FILE_NAMES = frozenset(
    {
        ".editorconfig",
        ".gitattributes",
        ".gitignore",
        ".dockerignore",
        ".pre-commit-config.yaml",
        ".pre-commit-config.yml",
        ".flake8",
        ".pylintrc",
        "alembic.ini",
        "angular.json",
        "build.gradle",
        "build.gradle.kts",
        "build.sbt",
        "cargo.toml",
        "cmakelists.txt",
        "cmakepresets.json",
        "compose.yaml",
        "compose.yml",
        "composer.json",
        "conanfile.py",
        "conanfile.txt",
        "conftest.py",
        "docker-compose.yaml",
        "docker-compose.yml",
        "dockerfile",
        "firebase.json",
        "gemfile",
        "go.mod",
        "go.sum",
        "gradle.properties",
        "jest.config.js",
        "jest.config.ts",
        "justfile",
        "karma.conf.js",
        "lerna.json",
        "makefile",
        "gnumakefile",
        "meson.build",
        "mix.exs",
        "mypy.ini",
        "netlify.toml",
        "noxfile.py",
        "nuxt.config.js",
        "nuxt.config.ts",
        "nx.json",
        "package.json",
        "package.swift",
        "pipfile",
        "pnpm-workspace.yaml",
        "pom.xml",
        "procfile",
        "pubspec.yaml",
        "pyproject.toml",
        "pytest.ini",
        "renovate.json",
        "requirements.txt",
        "runtime.txt",
        "settings.gradle",
        "settings.gradle.kts",
        "setup.cfg",
        "setup.py",
        "tailwind.config.js",
        "tailwind.config.ts",
        "taskfile.yml",
        "tox.ini",
        "tsconfig.json",
        "jsconfig.json",
        "turbo.json",
        "vercel.json",
        "vcpkg.json",
        "workspace",
        "module.bazel",
        "build.bazel",
        "jenkinsfile",
        "environment.yml",
        "constraints.txt",
    }
)

_CONFIG_PREFIXES = (
    ".babelrc",
    ".eslintrc",
    ".prettierrc",
    "astro.config.",
    "babel.config.",
    "cypress.config.",
    "eslint.config.",
    "jest.config.",
    "next.config.",
    "playwright.config.",
    "postcss.config.",
    "prettier.config.",
    "rollup.config.",
    "svelte.config.",
    "tailwind.config.",
    "vite.config.",
    "vitest.config.",
    "webpack.config.",
    "karma.conf.",
)

_CONFIG_DIR_PREFIXES = (
    ".github/workflows/",
    ".github/actions/",
    ".gitlab/",
    ".circleci/",
)

ENTRY_FILE_NAMES = frozenset(
    {
        "main.py",
        "app.py",
        "manage.py",
        "wsgi.py",
        "asgi.py",
        "cli.py",
        "__main__.py",
        "run.py",
        "server.py",
        "bot.py",
        "index.js",
        "index.ts",
        "index.jsx",
        "index.tsx",
        "main.js",
        "main.ts",
        "main.jsx",
        "main.tsx",
        "server.js",
        "server.ts",
        "app.js",
        "app.ts",
        "app.jsx",
        "app.tsx",
        "main.go",
        "main.rs",
        "main.c",
        "main.cpp",
        "main.cc",
        "main.cs",
        "program.cs",
        "main.java",
        "application.java",
        "main.kt",
        "main.swift",
        "application.swift",
        "main.dart",
        "index.php",
        "config.ru",
    }
)


def is_test_path(relative: str) -> bool:
    """True when ``relative`` is test code by convention.

    This is a naming convention, not a semantic judgement: a file in
    ``tests/`` or named ``*.spec.ts`` is counted as test code.
    """
    path = Path(relative)
    if {part.lower() for part in path.parts[:-1]} & TEST_DIR_NAMES:
        return True
    name = path.name
    return any(pattern.match(name) for pattern in _TEST_NAME_PATTERNS)


def is_documentation(relative: str) -> bool:
    """True for README-style documentation and files in a docs directory."""
    path = Path(relative)
    if {part.lower() for part in path.parts[:-1]} & _DOC_DIR_NAMES:
        return True
    suffix = path.suffix.lower()
    if suffix in {".md", ".rst", ".adoc"}:
        return True
    return suffix == ".txt" and path.stem.lower() in _DOC_STEMS


def is_config_file(relative: str) -> bool:
    """True when ``relative`` is a project configuration/build/CI file."""
    path = Path(relative)
    name = path.name.lower()
    if name in CONFIG_FILE_NAMES:
        return True
    if any(name.startswith(prefix) for prefix in _CONFIG_PREFIXES):
        return True
    if name.startswith("requirements") and name.endswith(".txt"):
        return True
    lowered = relative.lower()
    if lowered.startswith(_CONFIG_DIR_PREFIXES) and path.suffix.lower() in {
        ".yml",
        ".yaml",
    }:
        return True
    return False


def is_entry_point(relative: str) -> bool:
    """True for conventional application entry-point files."""
    path = Path(relative)
    name = path.name.lower()
    parts = tuple(part.lower() for part in path.parts[:-1])
    suffix = path.suffix.lower()

    if name in ENTRY_FILE_NAMES:
        return True
    if name in {"page.tsx", "page.jsx"} and parts in {("app",), ("src", "app")}:
        return True
    if name in {"index.tsx", "index.ts", "_app.tsx", "_app.ts"} and parts[:1] == (
        "pages",
    ):
        return True
    if name == "main.go" and "cmd" in parts:
        return True
    if parts[:1] == ("src",) and parts[1:2] == ("bin",) and suffix == ".rs":
        return True
    if parts[:1] == ("sources",) and name == "main.swift":
        return True
    return False



@dataclass(frozen=True, slots=True)
class ScanLimits:
    """Bounds for one project scan."""

    max_files: int = DEFAULT_MAX_FILES
    max_depth: int = DEFAULT_MAX_DEPTH
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    respect_gitignore: bool = True
    #: Extra glob patterns ignored in addition to :data:`IGNORED_DIRS`.
    extra_ignore: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ScannedFile:
    """Metadata for one file discovered under the workspace root."""

    path: Path
    relative: str
    size: int
    mtime_ns: int
    language: str | None
    is_source: bool
    is_binary: bool

    @property
    def suffix(self) -> str:
        return self.path.suffix.lower()


@dataclass(frozen=True, slots=True)
class ScanResult:
    """The outcome of :func:`scan_project`."""

    root: Path
    files: tuple[ScannedFile, ...]
    truncated: bool = False
    visited: int = 0
    errors: int = 0
    gitignore_rules: int = 0

    @property
    def source_files(self) -> tuple[ScannedFile, ...]:
        return tuple(file for file in self.files if file.is_source)


# --------------------------------------------------------------------- gitignore


@dataclass(frozen=True, slots=True)
class GitIgnoreRule:
    """One compiled ``.gitignore`` line, relative to its directory."""

    base: str
    regex: re.Pattern[str]
    negated: bool = False
    dir_only: bool = False

    def matches(self, relative: str, *, is_dir: bool) -> bool:
        if self.dir_only and not is_dir:
            return False
        if self.base:
            prefix = f"{self.base}/"
            if not relative.startswith(prefix):
                return False
            relative = relative[len(prefix) :]
        return self.regex.fullmatch(relative) is not None


def _glob_to_regex(pattern: str) -> str:
    """Translate a gitignore glob into a regular-expression fragment."""
    out: list[str] = []
    index = 0
    length = len(pattern)
    while index < length:
        char = pattern[index]
        if char == "*":
            if index + 1 < length and pattern[index + 1] == "*":
                # ``**/`` matches zero or more directories; ``**`` matches all.
                if index + 2 < length and pattern[index + 2] == "/":
                    out.append(r"(?:.*/)?")
                    index += 3
                    continue
                out.append(r".*")
                index += 2
                continue
            out.append(r"[^/]*")
            index += 1
            continue
        if char == "?":
            out.append(r"[^/]")
            index += 1
            continue
        if char == "[":
            end = pattern.find("]", index + 1)
            if end == -1:
                out.append(r"\[")
                index += 1
                continue
            body = pattern[index + 1 : end].replace("\\", r"\\")
            if body.startswith("!"):
                body = "^" + body[1:]
            elif body.startswith("^"):
                body = "\\" + body
            # Ranges such as ``a-z`` stay intact; a literal ``]`` cannot be
            # expressed by the simple glob syntax and is treated literally.
            out.append(f"[{body}]")
            index = end + 1
            continue
        out.append(re.escape(char))
        index += 1
    return "".join(out)


_FLAGS = re.IGNORECASE if os.name == "nt" else 0


def parse_gitignore(text: str, *, base: str = "") -> tuple[GitIgnoreRule, ...]:
    """Compile the non-comment lines of a ``.gitignore`` file."""
    rules: list[GitIgnoreRule] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        negated = line.startswith("!")
        if negated:
            line = line[1:].strip()
            if not line:
                continue
        if line.startswith("\\#") or line.startswith("\\!"):
            line = line[1:]

        dir_only = line.endswith("/")
        if dir_only:
            line = line[:-1]
        if not line:
            continue

        anchored = line.startswith("/")
        if anchored:
            line = line[1:]
        has_separator = "/" in line

        fragment = _glob_to_regex(line)
        if not (anchored or has_separator):
            fragment = rf"(?:.*/)?{fragment}"
        try:
            compiled = re.compile(fragment, _FLAGS)
        except re.error:  # pragma: no cover - defensive for exotic classes
            continue
        rules.append(
            GitIgnoreRule(
                base=base,
                regex=compiled,
                negated=negated,
                dir_only=dir_only,
            )
        )
    return tuple(rules)


def is_ignored(
    rules: Sequence[GitIgnoreRule], relative: str, *, is_dir: bool
) -> bool:
    """Apply rules in order; the last matching rule wins."""
    ignored = False
    for rule in rules:
        if rule.matches(relative, is_dir=is_dir):
            ignored = not rule.negated
    return ignored


def _read_gitignore(
    directory: Path, root: Path
) -> tuple[GitIgnoreRule, ...]:
    path = directory / ".gitignore"
    try:
        # A symlinked or junctioned .gitignore could point outside the
        # workspace; only a regular file inside the tree is trusted.
        if path.is_symlink() or is_junction(path) or not path.is_file():
            return ()
        size = path.stat().st_size
        if size > 64_000:
            return ()
        relative = (
            directory.relative_to(root).as_posix() if directory != root else ""
        )
        return tuple(
            parse_gitignore(
                path.read_text(encoding="utf-8", errors="replace"),
                base=relative,
            )
        )
    except OSError:
        return ()


# ------------------------------------------------------------------------ walker


def _matches_extra(relative: str, name: str, patterns: Sequence[str]) -> bool:
    lowered = relative.lower()
    lowered_name = name.lower()
    return any(
        fnmatch.fnmatch(lowered, pattern.lower())
        or fnmatch.fnmatch(lowered_name, pattern.lower())
        for pattern in patterns
    )


def scan_project(
    root: str | Path,
    *,
    limits: ScanLimits | None = None,
    extra_ignore: Iterable[str] = (),
) -> ScanResult:
    """Walk ``root`` and return metadata for every file worth analysing.

    The walker is deterministic (entries are sorted), never follows symlinks,
    never descends into generated directories, applies ``.gitignore`` rules
    from the root downwards, and stops at ``limits.max_files`` /
    ``limits.max_depth``.
    """
    base = Path(root).resolve()
    limits = limits or ScanLimits()
    extra = tuple(limits.extra_ignore) + tuple(extra_ignore)

    files: list[ScannedFile] = []
    truncated = False
    visited = 0
    errors = 0
    rule_count = 0

    # Each stack entry carries the accumulated gitignore rules for its branch.
    stack: list[tuple[Path, int, tuple[GitIgnoreRule, ...]]] = [(base, 0, ())]
    while stack:
        directory, depth, inherited = stack.pop()

        rules = inherited
        if limits.respect_gitignore:
            local_texts = _read_gitignore(directory, base)
            if local_texts:
                relative_dir = (
                    directory.relative_to(base).as_posix()
                    if directory != base
                    else ""
                )
                rules = inherited + local_texts
                rule_count += len(local_texts)

        # Sort for deterministic traversal, but never materialise an unbounded
        # directory: one huge folder must not defeat the scan limits.
        entry_cap = limits.max_files + 1
        try:
            listing = list(itertools.islice(directory.iterdir(), entry_cap))
        except OSError:
            errors += 1
            continue
        if len(listing) >= entry_cap:
            truncated = True
            listing = listing[: limits.max_files]
        entries = sorted(
            listing, key=lambda item: (not _safe_is_dir(item), item.name.lower())
        )

        for entry in entries:
            if len(files) >= limits.max_files:
                truncated = True
                break
            try:
                if entry.is_symlink() or is_junction(entry):
                    continue
                is_dir = entry.is_dir()
            except OSError:
                errors += 1
                continue

            relative = entry.relative_to(base).as_posix()
            name = entry.name

            if name.lower() in IGNORED_DIRS_LOWER:
                continue
            if extra and _matches_extra(relative, name, extra):
                continue
            if limits.respect_gitignore and rules and is_ignored(
                rules, relative, is_dir=is_dir
            ):
                continue

            if is_dir:
                if depth >= limits.max_depth:
                    truncated = True
                    continue
                stack.append((entry, depth + 1, rules))
                continue

            visited += 1
            try:
                stat = entry.stat()
            except OSError:
                errors += 1
                continue

            language = language_for_path(entry)
            files.append(
                ScannedFile(
                    path=entry,
                    relative=relative,
                    size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    language=language,
                    is_source=language is not None,
                    is_binary=is_binary_path(entry),
                )
            )

        if len(files) >= limits.max_files:
            truncated = True
            break

    files.sort(key=lambda item: item.relative)
    return ScanResult(
        root=base,
        files=tuple(files),
        truncated=truncated,
        visited=visited,
        errors=errors,
        gitignore_rules=rule_count,
    )


def is_junction(path: Path) -> bool:
    """True when ``path`` is a Windows directory junction.

    Junctions are not symlinks: ``is_symlink()`` is False and ``is_dir()``
    follows them, so without this check the walker would descend into a
    directory that lives outside the workspace.
    """
    checker = getattr(path, "is_junction", None)
    if checker is not None:
        try:
            return bool(checker())
        except OSError:  # pragma: no cover - unreadable entry
            return False
    try:
        # Python 3.11 fallback: IO_REPARSE_TAG_MOUNT_POINT.
        return path.lstat().st_reparse_tag == 0xA0000003
    except (OSError, AttributeError):  # pragma: no cover - non-Windows
        return False


def _safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:  # pragma: no cover - unreadable entry
        return False


def read_text_limited(path: Path, limit: int) -> str | None:
    """Read at most ``limit`` bytes of UTF-8 text, or ``None`` if binary."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            data = handle.read(limit)
    except OSError:
        return None
    if looks_binary(data):
        return None
    text = data.decode("utf-8", errors="replace")
    if size > limit:
        text += f"\n… (truncated at {limit} bytes)"
    return text
