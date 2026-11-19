"""Evidence-based project detection.

Every statement produced here is traceable to a file that was actually seen:
a declared dependency, a manifest, a conventional entry point or a
framework-specific configuration file.  Nothing is inferred from the project
name or from a hardcoded template.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Iterable

from beast.intelligence.models import (
    DependencyReport,
    LanguageStat,
    ProjectDetection,
)
from beast.workspace.scanning import (
    ScannedFile,
    is_config_file,
    is_entry_point,
    is_junction,
    is_test_path,
    read_text_limited,
)
from beast.workspace.secrets import is_sensitive_path

#: Reading a configuration file larger than this is not worth it.
MAX_CONFIG_BYTES = 256_000

# Exact dependency name → framework.  Keys are lowercase.
_FRAMEWORKS: dict[str, str] = {
    # Python
    "flask": "Flask",
    "django": "Django",
    "fastapi": "FastAPI",
    "starlette": "Starlette",
    "tornado": "Tornado",
    "aiohttp": "aiohttp",
    "sanic": "Sanic",
    "quart": "Quart",
    "litestar": "Litestar",
    "pyramid": "Pyramid",
    "bottle": "Bottle",
    "streamlit": "Streamlit",
    "gradio": "Gradio",
    "dash": "Dash",
    "reflex": "Reflex",
    # JavaScript / TypeScript
    "react": "React",
    "react-dom": "React",
    "next": "Next.js",
    "vue": "Vue",
    "nuxt": "Nuxt",
    "@angular/core": "Angular",
    "svelte": "Svelte",
    "@sveltejs/kit": "SvelteKit",
    "astro": "Astro",
    "solid-js": "Solid",
    "preact": "Preact",
    "express": "Express",
    "fastify": "Fastify",
    "koa": "Koa",
    "hono": "Hono",
    "@nestjs/core": "NestJS",
    "@hapi/hapi": "hapi",
    "electron": "Electron",
    "react-native": "React Native",
    "expo": "Expo",
    "@remix-run/react": "Remix",
    "gatsby": "Gatsby",
    # Ruby
    "rails": "Rails",
    "sinatra": "Sinatra",
    "hanami": "Hanami",
    "grape": "Grape",
    # PHP
    "laravel/framework": "Laravel",
    "symfony/framework-bundle": "Symfony",
    "slim/slim": "Slim",
    "cakephp/cakephp": "CakePHP",
    # Rust
    "actix-web": "Actix Web",
    "axum": "Axum",
    "rocket": "Rocket",
    "warp": "Warp",
    "poem": "Poem",
    "tide": "Tide",
    # Go
    "github.com/gin-gonic/gin": "Gin",
    "github.com/labstack/echo": "Echo",
    "github.com/gofiber/fiber": "Fiber",
    "github.com/go-chi/chi": "Chi",
    "github.com/gorilla/mux": "Gorilla Mux",
    "github.com/beego/beego": "Beego",
    "github.com/spf13/cobra": "Cobra",
    # Swift / Elixir / Scala
    "vapor/vapor": "Vapor",
    "phoenix": "Phoenix",
    "plug": "Plug",
    "com.typesafe.play:play": "Play",
}

# Prefix → framework, checked after exact names.
_FRAMEWORK_PREFIXES: tuple[tuple[str, str], ...] = (
    ("org.springframework.boot:", "Spring Boot"),
    ("org.springframework:", "Spring"),
    ("io.quarkus:", "Quarkus"),
    ("io.micronaut:", "Micronaut"),
    ("io.ktor:", "Ktor"),
    ("io.javalin:", "Javalin"),
    ("androidx.", "Android"),
    ("com.android.", "Android"),
    ("org.jetbrains.compose", "Compose Multiplatform"),
    ("microsoft.aspnetcore", "ASP.NET Core"),
    ("microsoft.maui", "MAUI"),
    ("symfony/", "Symfony"),
)

_TEST_FRAMEWORKS: dict[str, str] = {
    # Python
    "pytest": "pytest",
    "nose": "nose",
    "nose2": "nose2",
    "ward": "ward",
    # JavaScript / TypeScript
    "jest": "Jest",
    "vitest": "Vitest",
    "mocha": "Mocha",
    "jasmine": "Jasmine",
    "karma": "Karma",
    "ava": "AVA",
    "tap": "tap",
    "cypress": "Cypress",
    "playwright": "Playwright",
    "@playwright/test": "Playwright",
    # Java / Kotlin
    "junit": "JUnit",
    "org.junit.jupiter:junit-jupiter": "JUnit",
    "org.testng:testng": "TestNG",
    "org.mockito:mockito-core": "Mockito",
    "io.cucumber:cucumber-jvm": "Cucumber",
    "org.spockframework:spock-core": "Spock",
    # Ruby
    "rspec": "RSpec",
    "minitest": "Minitest",
    "cucumber": "Cucumber",
    # PHP
    "phpunit/phpunit": "PHPUnit",
    "pestphp/pest": "Pest",
    # .NET
    "xunit": "xUnit",
    "nunit": "NUnit",
    "mstest.testframework": "MSTest",
    # Go
    "github.com/onsi/ginkgo": "Ginkgo",
    "github.com/stretchr/testify": "testify",
}

_TEST_CONFIGS: tuple[tuple[str, str], ...] = (
    ("pytest.ini", "pytest"),
    ("conftest.py", "pytest"),
    ("jest.config.", "Jest"),
    ("vitest.config.", "Vitest"),
    ("cypress.config.", "Cypress"),
    ("playwright.config.", "Playwright"),
    ("karma.conf.", "Karma"),
    (".mocharc", "Mocha"),
    ("phpunit.xml", "PHPUnit"),
    (".rspec", "RSpec"),
    ("noxfile.py", "Nox"),
    ("tox.ini", "tox"),
)

_BUILD_FILES: tuple[tuple[str, str], ...] = (
    ("CMakeLists.txt", "CMake"),
    ("Makefile", "Make"),
    ("makefile", "Make"),
    ("GNUmakefile", "Make"),
    ("meson.build", "Meson"),
    ("BUILD.bazel", "Bazel"),
    ("MODULE.bazel", "Bazel"),
    ("WORKSPACE", "Bazel"),
    ("pom.xml", "Maven"),
    ("build.gradle", "Gradle"),
    ("build.gradle.kts", "Gradle"),
    ("Cargo.toml", "Cargo"),
    ("go.mod", "Go modules"),
    ("build.sbt", "sbt"),
    ("Package.swift", "SwiftPM"),
    ("configure.ac", "Autotools"),
    ("build.ninja", "Ninja"),
    ("justfile", "Just"),
    ("Taskfile.yml", "Task"),
    ("nx.json", "Nx"),
    ("turbo.json", "Turborepo"),
    ("lerna.json", "Lerna"),
    ("pnpm-workspace.yaml", "pnpm workspace"),
)

_BUNDLERS: tuple[tuple[str, str], ...] = (
    ("vite.config.", "Vite"),
    ("webpack.config.", "webpack"),
    ("rollup.config.", "Rollup"),
    ("esbuild.config.", "esbuild"),
    ("tsup.config.", "tsup"),
)

_PYTHON_BACKENDS = {
    "setuptools.build_meta": "setuptools",
    "poetry.core.masonry.api": "Poetry",
    "hatchling.build": "Hatchling",
    "flit_core.buildapi": "Flit",
    "pdm.backend": "PDM",
    "maturin": "Maturin",
}

_ENTRY_PRIORITY = {
    "main.py": 0,
    "app.py": 1,
    "manage.py": 2,
    "__main__.py": 3,
    "index.ts": 4,
    "index.js": 5,
    "main.ts": 6,
    "main.tsx": 7,
    "main.js": 8,
    "src/main.rs": 9,
    "main.go": 10,
    "Program.cs": 11,
    "main.dart": 12,
    "main.c": 13,
    "main.cpp": 14,
}

#: Root manifests that themselves prove which package manager is used.
_MANIFEST_MANAGERS = (
    ("requirements", "pip"),
    ("pyproject.toml", "pip"),
    ("setup.py", "pip"),
    ("setup.cfg", "pip"),
    ("Pipfile", "pipenv"),
    ("package.json", "npm"),
    ("pom.xml", "Maven"),
    ("build.gradle", "Gradle"),
    ("Cargo.toml", "Cargo"),
    ("go.mod", "Go modules"),
    ("Gemfile", "Bundler"),
    ("composer.json", "Composer"),
    ("pubspec.yaml", "Pub"),
    ("Package.swift", "SwiftPM"),
    ("build.sbt", "sbt"),
    ("mix.exs", "Mix"),
)


class ProjectDetector:
    """Derives a :class:`ProjectDetection` from a scan and dependency report."""

    def __init__(self, *, max_config_bytes: int = MAX_CONFIG_BYTES) -> None:
        self.max_config_bytes = max_config_bytes

    def detect(
        self,
        scan_files: Iterable[ScannedFile],
        dependencies: DependencyReport,
        *,
        root: str | Path,
        name: str | None = None,
    ) -> ProjectDetection:
        base = Path(root).resolve()
        files = tuple(scan_files)
        relatives = [file.relative for file in files]

        source_files = [
            file for file in files if file.is_source and not file.is_binary
        ]
        test_files = [file for file in source_files if is_test_path(file.relative)]
        production = [file for file in source_files if not is_test_path(file.relative)]
        counted = production or source_files

        languages = _language_stats(counted)
        primary = languages[0].name if languages else None

        dependency_names = [dependency.name.lower() for dependency in dependencies.dependencies]
        dependency_set = set(dependency_names)
        frameworks = _frameworks(
            dependency_names,
            dependency_set,
            relatives,
            base,
            max_config_bytes=self.max_config_bytes,
        )

        package_managers = list(dependencies.package_managers)
        if not package_managers:
            package_managers = _package_managers(relatives)
        build_systems = _build_systems(relatives, base, dependency_set)

        entry_points = _entry_points(relatives, base)
        config_files = _config_files(relatives)
        test_frameworks = _test_frameworks(dependency_set, relatives)

        return ProjectDetection(
            name=name or base.name or "project",
            languages=tuple(languages),
            primary_language=primary,
            frameworks=tuple(frameworks),
            package_managers=tuple(package_managers),
            build_systems=tuple(build_systems),
            test_frameworks=tuple(test_frameworks),
            manifests=dependencies.manifests,
            entry_points=tuple(entry_points),
            config_files=tuple(config_files),
            source_file_count=len(production),
            test_file_count=len(test_files),
        )


def _language_stats(files: Iterable[ScannedFile]) -> list[LanguageStat]:
    counts: dict[str, int] = {}
    extensions: dict[str, set[str]] = {}
    for file in files:
        if not file.language:
            continue
        counts[file.language] = counts.get(file.language, 0) + 1
        extensions.setdefault(file.language, set()).add(file.suffix)
    return [
        LanguageStat(
            name=language,
            file_count=count,
            extensions=tuple(sorted(extensions[language])),
        )
        for language, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0].lower())
        )
    ]


def _frameworks(
    ordered_names: Iterable[str],
    names: set[str],
    relatives: list[str],
    base: Path,
    *,
    max_config_bytes: int = MAX_CONFIG_BYTES,
) -> list[str]:
    found: list[str] = []
    for name in ordered_names:
        label = _FRAMEWORKS.get(name)
        if label is None:
            for prefix, candidate in _FRAMEWORK_PREFIXES:
                if name.startswith(prefix):
                    label = candidate
                    break
        if label and label not in found:
            found.append(label)

    def has(predicate) -> bool:
        return any(predicate(relative) for relative in relatives)

    def add(label: str) -> None:
        if label not in found:
            found.append(label)

    # Framework-specific files are evidence even when the manifest is absent.
    if "Django" not in found and has(lambda rel: rel.endswith("/manage.py") or rel == "manage.py"):
        add("Django")
    if has(lambda rel: Path(rel).name.startswith("next.config.")):
        add("Next.js")
    if has(lambda rel: Path(rel).name.startswith("nuxt.config.")):
        add("Nuxt")
    if has(lambda rel: Path(rel).name.startswith("svelte.config.")):
        add("SvelteKit" if "svelte" in names else "Svelte")
    if has(lambda rel: Path(rel).name.startswith("astro.config.")):
        add("Astro")
    if "angular.json" in relatives:
        add("Angular")

    manifest_paths = [
        base / relative
        for relative in relatives
        if Path(relative).name in {"pubspec.yaml", "CMakeLists.txt"}
    ]
    for path in manifest_paths:
        text = _read_config(base, path, max_config_bytes)
        if path.name == "pubspec.yaml" and "sdk: flutter" in text:
            add("Flutter")
        if path.name == "CMakeLists.txt":
            if "find_package(Qt" in text or "find_package(QT" in text:
                add("Qt")
            if "find_package(Boost" in text:
                add("Boost")
    for relative in relatives:
        if Path(relative).suffix.lower() in {".csproj", ".fsproj"}:
            text = _read_config(
                base, base / relative, min(max_config_bytes, 128_000)
            )
            if "Microsoft.NET.Sdk.Web" in text:
                add("ASP.NET Core")
    return _dedupe(found)


def _build_systems(
    relatives: list[str], base: Path, dependency_names: set[str]
) -> list[str]:
    found: list[str] = []
    names = {Path(relative).name for relative in relatives}
    for filename, label in _BUILD_FILES:
        if filename in names:
            _append(found, label)
    for prefix, label in _BUNDLERS:
        if any(Path(relative).name.startswith(prefix) for relative in relatives):
            _append(found, label)
    if "package.json" in names:
        scripts = _package_json(base).get("scripts")
        if isinstance(scripts, dict) and scripts:
            _append(found, "npm scripts")
    backend = _python_build_backend(base)
    if backend:
        _append(found, backend)
    return found


def _package_managers(relatives: list[str]) -> list[str]:
    names = {Path(relative).name for relative in relatives}
    found: list[str] = []
    for filename, manager in _MANIFEST_MANAGERS:
        if filename == "requirements":
            if any(name.startswith("requirements") and name.endswith(".txt") for name in names):
                _append(found, manager)
            continue
        if filename in names:
            _append(found, manager)
    for name, manager in (
        ("poetry.lock", "Poetry"),
        ("uv.lock", "uv"),
        ("Pipfile.lock", "pipenv"),
        ("yarn.lock", "Yarn"),
        ("pnpm-lock.yaml", "pnpm"),
        ("bun.lockb", "bun"),
        ("package-lock.json", "npm"),
        ("Cargo.lock", "Cargo"),
        ("go.sum", "Go modules"),
        ("Gemfile.lock", "Bundler"),
        ("composer.lock", "Composer"),
    ):
        if name in names:
            # Lockfile evidence wins over the default associated with a manifest.
            if manager == "Yarn" and "npm" in found:
                found.remove("npm")
            _append(found, manager)
    return found


def _entry_points(relatives: list[str], base: Path) -> list[str]:
    candidates = [relative for relative in relatives if is_entry_point(relative)]
    package = _package_json(base)
    for key in ("main", "module", "browser"):
        value = package.get(key)
        if isinstance(value, str):
            candidates.extend(_existing_relative(base, value))
    binary = package.get("bin")
    if isinstance(binary, str):
        candidates.extend(_existing_relative(base, binary))
    elif isinstance(binary, dict):
        for value in binary.values():
            if isinstance(value, str):
                candidates.extend(_existing_relative(base, value))
    candidates = [
        relative
        for relative in _dedupe(candidates)
        if not is_sensitive_path(relative)
    ]

    def priority(relative: str) -> tuple[int, int, str]:
        name = Path(relative).name
        return (
            _ENTRY_PRIORITY.get(name, 50),
            relative.count("/"),
            relative,
        )

    return sorted(candidates, key=priority)[:25]


def _existing_relative(base: Path, value: str) -> list[str]:
    cleaned = value.replace("\\", "/").strip()
    # Only a single leading ``./`` is a plain relative prefix.  ``.app/main.js``
    # is a real path, and ``../x`` must stay parent-traversing so it is refused
    # below rather than silently rewritten to ``x``.
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    if not cleaned or cleaned.startswith(("/", "../")) or cleaned == "..":
        return []
    if any(part == ".." for part in cleaned.split("/")):
        return []
    candidate = base / cleaned
    try:
        resolved = candidate.resolve()
        resolved.relative_to(base.resolve())
    except (OSError, ValueError):
        return []
    if resolved.is_file() and not is_sensitive_path(cleaned):
        return [cleaned]
    return []


def _config_files(relatives: list[str]) -> list[str]:
    found = [
        relative
        for relative in relatives
        if is_config_file(relative) and not is_sensitive_path(relative)
    ]

    def priority(relative: str) -> tuple[int, int, str]:
        return (relative.count("/"), len(relative), relative)

    return sorted(dict.fromkeys(found), key=priority)[:60]


def _test_frameworks(names: set[str], relatives: list[str]) -> list[str]:
    found: list[str] = []
    for name in sorted(names):
        label = _TEST_FRAMEWORKS.get(name)
        if label:
            _append(found, label)
    lowered = [relative.lower() for relative in relatives]
    for marker, label in _TEST_CONFIGS:
        if any(
            Path(relative).name.lower().startswith(marker) or marker in relative
            for relative in lowered
        ):
            _append(found, label)
    return found


def _read_config(base: Path, path: Path, limit: int) -> str:
    """Read a config file only when it is a regular file inside ``base``."""
    try:
        if path.is_symlink() or is_junction(path) or not path.is_file():
            return ""
        if not path.resolve().is_relative_to(base.resolve()):
            return ""
    except OSError:
        return ""
    return read_text_limited(path, limit) or ""


def _package_json(base: Path) -> dict:
    path = base / "package.json"
    text = _read_config(base, path, 512_000)
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _python_build_backend(base: Path) -> str | None:
    path = base / "pyproject.toml"
    text = _read_config(base, path, 512_000)
    if not text:
        return None
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    build_system = data.get("build-system")
    if not isinstance(build_system, dict):
        return None
    backend = build_system.get("build-backend")
    if not isinstance(backend, str):
        return None
    return _PYTHON_BACKENDS.get(backend)


def _append(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
