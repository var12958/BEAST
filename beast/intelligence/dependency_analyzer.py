"""Dependency manifest analysis.

BEAST reads the manifests a project actually contains; it never installs,
resolves or executes anything.  ``setup.py`` is parsed with :mod:`ast` and only
literal keyword values are read — the file is never imported.  Gradle, Gemfile
and a few other ecosystems are read with narrow line patterns, which is enough
to report declared dependencies and clearly documented as such.

Every dependency remembers the manifest it came from, so the model can cite
evidence instead of guessing.
"""

from __future__ import annotations

import ast
import configparser
import json
import re
import tomllib
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from pathlib import Path

from beast.intelligence.models import Dependency, DependencyReport, ManifestInfo
from beast.workspace.scanning import (
    ScannedFile,
    ScanLimits,
    is_junction,
    read_text_limited,
    scan_project,
)

#: A manifest may not be larger than this; anything bigger is not hand-written.
MAX_MANIFEST_BYTES = 1_000_000

_DEV_GROUP_HINTS = frozenset(
    {
        "dev",
        "development",
        "develop",
        "test",
        "tests",
        "testing",
        "lint",
        "linting",
        "typecheck",
        "typing",
        "ci",
    }
)

_LOCKFILE_MANAGERS = {
    "package-lock.json": "npm",
    "npm-shrinkwrap.json": "npm",
    "yarn.lock": "yarn",
    "pnpm-lock.yaml": "pnpm",
    "bun.lockb": "bun",
    "bun.lock": "bun",
    "poetry.lock": "poetry",
    "pdm.lock": "pdm",
    "uv.lock": "uv",
    "Pipfile.lock": "pipenv",
    "Cargo.lock": "cargo",
    "go.sum": "go modules",
    "Gemfile.lock": "bundler",
    "composer.lock": "composer",
    "mix.lock": "mix",
    "pubspec.lock": "pub",
    "Package.resolved": "swiftpm",
    "gradle.lockfile": "gradle",
}

_NON_RUNTIME_SCOPES = {
    "dev": "dev",
    "development": "dev",
    "test": "dev",
    "tests": "dev",
    "testing": "dev",
    "docs": "dev",
    "lint": "dev",
    "build": "build",
    "peer": "peer",
    "optional": "optional",
    "provided": "build",
    "compileOnly": "build",
    "classpath": "build",
}

_PEP508 = re.compile(
    r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?\s*(.*)$"
)
_REQUIREMENT_LINE = re.compile(
    r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?\s*([^;]*)(?:;.*)?$"
)
#: Gradle configurations that describe build-time-only tooling.
_GRADLE_BUILD_CONFIGURATIONS = frozenset(
    {
        "compileOnly",
        "compileOnlyApi",
        "classpath",
        "annotationProcessor",
        "kapt",
        "ksp",
    }
)
_GRADLE_DEP = re.compile(
    r"(?m)^\s*(?P<conf>implementation|api|compileOnly|compileOnlyApi|runtimeOnly|"
    r"testImplementation|testRuntimeOnly|testCompileOnly|kapt|ksp|"
    r"annotationProcessor|classpath)\s*[\( ]\s*['\"](?P<coord>[^'\"]+)['\"]"
)
_GO_REQUIRE_LINE = re.compile(
    r"^\s*([A-Za-z0-9._~\-/]+(?:\.[A-Za-z]{2,})?[A-Za-z0-9._~\-/]*)"
    r"\s+(v[0-9][^\s]*)"
)
_GEM = re.compile(
    r"^\s*gem\s+['\"]([^'\"]+)['\"](?:\s*,\s*['\"]([^'\"]+)['\"])?"
)
_PACKAGE_SWIFT = re.compile(
    r"\.package\s*\(\s*(?:name:\s*\"([^\"]+)\"\s*,\s*)?url:\s*\"([^\"]+)\""
)
_SBT = re.compile(r"\"([^\"]+)\"\s*%\s*\"([^\"]+)\"\s*%\s*\"([^\"]+)\"")
_MIX = re.compile(r"\{:([A-Za-z0-9_]+),\s*\"([^\"]*)\"")


def _clean_version(specification: str) -> str | None:
    cleaned = " ".join(specification.split()).strip()
    if not cleaned:
        return None
    if cleaned.startswith("=="):
        cleaned = cleaned[2:].strip()
    return cleaned[:120] or None


def _split_pep508(specification: str) -> tuple[str, str | None] | None:
    specification = specification.split(";", 1)[0].strip()
    if not specification or specification.startswith(("-", "#")):
        return None
    match = _PEP508.match(specification)
    if not match:
        return None
    name = match.group(1)
    rest = match.group(2).strip()
    if rest.startswith("@"):
        return name, rest[:120]
    return name, _clean_version(rest)


def _scope_from_group(group: str) -> str:
    lowered = group.lower()
    if any(hint in lowered for hint in _DEV_GROUP_HINTS):
        return "dev"
    return "optional"


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


class DependencyAnalyzer:
    """Extracts declared dependencies from every supported manifest."""

    def __init__(self, *, max_bytes: int = MAX_MANIFEST_BYTES) -> None:
        self.max_bytes = max_bytes

    # ------------------------------------------------------------------ public

    def analyze(
        self,
        root: str | Path,
        files: Sequence[ScannedFile] | None = None,
        *,
        relative_paths: Iterable[str] | None = None,
    ) -> DependencyReport:
        base = Path(root).resolve()
        if relative_paths is not None:
            paths = sorted(set(relative_paths))
        elif files is not None:
            paths = sorted({file.relative for file in files})
        else:
            scan = scan_project(
                base,
                limits=ScanLimits(max_files=2_000, max_depth=4),
            )
            paths = [file.relative for file in scan.files]

        manifest_paths = [path for path in paths if _recognise(path) is not None]
        # Root-level manifests must not be lost to a truncated scan.
        manifest_paths.extend(_root_manifests(base, manifest_paths))
        manifest_paths = _dedupe(manifest_paths)

        dependencies: list[Dependency] = []
        manifests: list[ManifestInfo] = []
        managers: list[str] = []
        warnings: list[str] = []

        for relative in manifest_paths:
            info = _recognise(relative)
            if info is None:
                continue
            kind, ecosystem, default_manager = info
            path = base / relative
            if is_junction(path) or path.is_symlink():
                continue
            if not path.is_file():
                continue
            try:
                # A link swapped in after the scan must not be followed outside
                # the workspace.
                if not path.resolve().is_relative_to(base):
                    warnings.append(f"{relative}: outside the workspace; skipped")
                    continue
            except OSError:
                warnings.append(f"{relative}: could not be opened; skipped")
                continue
            try:
                found, manager = self._parse(path, relative, kind)
            except (
                OSError,
                ValueError,
                TypeError,
                AttributeError,
                KeyError,
                ET.ParseError,
                tomllib.TOMLDecodeError,
            ) as exc:
                warnings.append(f"{relative}: could not parse ({type(exc).__name__})")
                found, manager = [], default_manager
            dependencies.extend(found)
            manager = manager or default_manager
            managers.append(manager)
            manifests.append(
                ManifestInfo(
                    path=relative,
                    kind=kind,
                    ecosystem=ecosystem,
                    package_manager=manager,
                )
            )

        # Lockfiles are evidence of the package manager actually used, and
        # beat the default associated with a manifest (a yarn.lock means Yarn,
        # not npm, even though package.json is also present).
        for relative in paths:
            name = Path(relative).name
            manager = _LOCKFILE_MANAGERS.get(name)
            if not manager:
                continue
            if manager in {"yarn", "pnpm", "bun"} and "npm" in managers:
                managers.remove("npm")
            if manager not in managers:
                managers.insert(0, manager)

        unique: list[Dependency] = []
        seen: set[tuple[str, str, str]] = set()
        for dependency in dependencies:
            key = (
                dependency.ecosystem.lower(),
                dependency.name.lower(),
                dependency.scope,
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(dependency)
        unique.sort(key=lambda dep: (_SCOPE_ORDER.get(dep.scope, 9), dep.name.lower()))

        return DependencyReport(
            dependencies=tuple(unique),
            manifests=tuple(manifests),
            package_managers=tuple(dict.fromkeys(managers)),
            warnings=tuple(warnings[:10]),
        )

    # ------------------------------------------------------------- dispatchers

    def _parse(
        self, path: Path, relative: str, kind: str
    ) -> tuple[list[Dependency], str | None]:
        parsers = {
            "requirements": self._requirements,
            "pyproject": self._pyproject,
            "setup.py": self._setup_py,
            "setup.cfg": self._setup_cfg,
            "pipfile": self._pipfile,
            "package-json": self._package_json,
            "pom": self._pom,
            "gradle": self._gradle,
            "cargo": self._cargo,
            "go-mod": self._go_mod,
            "gemfile": self._gemfile,
            "composer": self._composer,
            "pubspec": self._pubspec,
            "swiftpm": self._swiftpm,
            "csproj": self._csproj,
            "sbt": self._sbt,
            "mix": self._mix,
            "vcpkg": self._vcpkg,
        }
        parser = parsers.get(kind)
        if parser is None:  # pragma: no cover - _recognise and parsers stay in sync
            return [], None
        return parser(path, relative)

    def _text(self, path: Path) -> str:
        text = read_text_limited(path, self.max_bytes)
        return text or ""

    # -------------------------------------------------------------- Python/pip

    def _requirements(
        self, path: Path, relative: str, seen: set[Path] | None = None
    ) -> tuple[list[Dependency], str | None]:
        seen = seen or set()
        if path in seen:
            return [], "pip"
        seen.add(path)
        text = self._text(path)
        scope = (
            "dev"
            if any(hint in path.name.lower() for hint in ("dev", "test", "lint"))
            else "runtime"
        )
        dependencies: list[Dependency] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith(("-r ", "--requirement ")):
                included = stripped.split(maxsplit=1)[1].strip()
                included_path = (path.parent / included).resolve()
                if (
                    included_path.is_file()
                    and path.parent.resolve() in included_path.parents
                ):
                    nested, _ = self._requirements(included_path, relative, seen)
                    dependencies.extend(nested)
                continue
            if stripped.startswith("-"):
                continue
            match = _REQUIREMENT_LINE.match(stripped)
            if not match:
                continue
            name, version = match.group(1), _clean_version(match.group(2))
            dependencies.append(
                Dependency(
                    name=name,
                    version=version,
                    scope=scope,
                    ecosystem="python",
                    source=relative,
                )
            )
        return dependencies, "pip"

    def _pyproject(
        self, path: Path, relative: str
    ) -> tuple[list[Dependency], str | None]:
        data = tomllib.loads(self._text(path))
        dependencies: list[Dependency] = []

        def add(specification: str, scope: str) -> None:
            parsed = _split_pep508(str(specification))
            if parsed is None:
                return
            name, version = parsed
            dependencies.append(
                Dependency(
                    name=name,
                    version=version,
                    scope=scope,
                    ecosystem="python",
                    source=relative,
                )
            )

        # A malformed manifest must not become fake intelligence: a string where
        # a list belongs would otherwise be iterated character by character.
        project = data.get("project")
        if not isinstance(project, dict):
            project = {}
        declared = project.get("dependencies")
        if isinstance(declared, (list, tuple)):
            for specification in declared:
                if isinstance(specification, str):
                    add(specification, "runtime")
        optional = project.get("optional-dependencies")
        if isinstance(optional, dict):
            for group, specifications in optional.items():
                if not isinstance(specifications, (list, tuple)):
                    continue
                for specification in specifications:
                    if isinstance(specification, str):
                        add(specification, _scope_from_group(str(group)))
        groups = data.get("dependency-groups")
        if isinstance(groups, dict):
            for group, specifications in groups.items():
                if not isinstance(specifications, (list, tuple)):
                    continue
                for specification in specifications:
                    if isinstance(specification, str):
                        add(specification, _scope_from_group(str(group)))

        tool = data.get("tool")
        poetry = tool.get("poetry") if isinstance(tool, dict) else None
        if not isinstance(poetry, dict):
            poetry = {}
        poetry_deps = poetry.get("dependencies")
        for name, value in (
            poetry_deps.items() if isinstance(poetry_deps, dict) else ()
        ):
            if str(name).lower() == "python":
                continue
            dependencies.append(
                Dependency(
                    name=str(name),
                    version=_poetry_version(value),
                    scope="runtime",
                    ecosystem="python",
                    source=relative,
                )
            )
        poetry_groups = poetry.get("group")
        if isinstance(poetry_groups, dict):
            for group, payload in poetry_groups.items():
                group_deps = (
                    payload.get("dependencies") if isinstance(payload, dict) else None
                )
                if not isinstance(group_deps, dict):
                    continue
                for name, value in group_deps.items():
                    dependencies.append(
                        Dependency(
                            name=str(name),
                            version=_poetry_version(value),
                            scope=_scope_from_group(str(group)),
                            ecosystem="python",
                            source=relative,
                        )
                    )

        build_system = data.get("build-system")
        if not isinstance(build_system, dict):
            build_system = {}
        build_requires = build_system.get("requires")
        if isinstance(build_requires, (list, tuple)):
            for specification in build_requires:
                if isinstance(specification, str):
                    add(specification, "build")

        build_backend = str(build_system.get("build-backend") or "")
        manager = "pip"
        if poetry or "poetry" in build_backend:
            manager = "poetry"
        elif "pdm" in build_backend or (isinstance(tool, dict) and "pdm" in tool):
            manager = "pdm"
        elif "hatch" in build_backend:
            manager = "hatch"
        elif "flit" in build_backend:
            manager = "flit"
        return dependencies, manager

    def _setup_py(
        self, path: Path, relative: str
    ) -> tuple[list[Dependency], str | None]:
        try:
            tree = ast.parse(self._text(path), filename=str(path))
        except (SyntaxError, ValueError):
            return [], "pip"
        dependencies: list[Dependency] = []

        def add_literal(node: ast.AST, scope: str) -> None:
            value = _literal(node)
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, (list, tuple)):
                return
            for item in value:
                if not isinstance(item, str):
                    continue
                parsed = _split_pep508(item)
                if parsed is None:
                    continue
                name, version = parsed
                dependencies.append(
                    Dependency(
                        name=name,
                        version=version,
                        scope=scope,
                        ecosystem="python",
                        source=relative,
                    )
                )

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            function_name = (
                function.id
                if isinstance(function, ast.Name)
                else function.attr
                if isinstance(function, ast.Attribute)
                else ""
            )
            if function_name != "setup":
                continue
            for keyword in node.keywords:
                if keyword.arg in {"install_requires", "tests_require", "setup_requires"}:
                    scope = {
                        "install_requires": "runtime",
                        "tests_require": "dev",
                        "setup_requires": "build",
                    }[keyword.arg]
                    add_literal(keyword.value, scope)
                elif keyword.arg == "extras_require":
                    extras = _literal(keyword.value)
                    if isinstance(extras, dict):
                        for group, values in extras.items():
                            if not isinstance(values, (list, tuple)):
                                continue
                            for item in values:
                                if not isinstance(item, str):
                                    continue
                                parsed = _split_pep508(item)
                                if parsed is None:
                                    continue
                                dependencies.append(
                                    Dependency(
                                        name=parsed[0],
                                        version=parsed[1],
                                        scope=_scope_from_group(str(group)),
                                        ecosystem="python",
                                        source=relative,
                                    )
                                )
        return dependencies, "pip"

    def _setup_cfg(
        self, path: Path, relative: str
    ) -> tuple[list[Dependency], str | None]:
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read_string(self._text(path))
        except configparser.Error:
            return [], "pip"
        dependencies: list[Dependency] = []

        def add_block(block: str, scope: str) -> None:
            for line in block.splitlines():
                parsed = _split_pep508(line.strip())
                if parsed is None:
                    continue
                dependencies.append(
                    Dependency(
                        name=parsed[0],
                        version=parsed[1],
                        scope=scope,
                        ecosystem="python",
                        source=relative,
                    )
                )

        if parser.has_option("options", "install_requires"):
            add_block(parser.get("options", "install_requires"), "runtime")
        if parser.has_option("options", "setup_requires"):
            add_block(parser.get("options", "setup_requires"), "build")
        if parser.has_section("options.extras_require"):
            for group, block in parser.items("options.extras_require"):
                add_block(block, _scope_from_group(group))
        return dependencies, "pip"

    def _pipfile(
        self, path: Path, relative: str
    ) -> tuple[list[Dependency], str | None]:
        data = tomllib.loads(self._text(path))
        dependencies: list[Dependency] = []
        for section, scope in (("packages", "runtime"), ("dev-packages", "dev")):
            for name, value in (data.get(section) or {}).items():
                version = value if isinstance(value, str) else None
                dependencies.append(
                    Dependency(
                        name=str(name),
                        version=_clean_version(version or ""),
                        scope=scope,
                        ecosystem="python",
                        source=relative,
                    )
                )
        return dependencies, "pipenv"

    # --------------------------------------------------------- JavaScript / npm

    def _package_json(
        self, path: Path, relative: str
    ) -> tuple[list[Dependency], str | None]:
        data = json.loads(self._text(path) or "{}")
        dependencies: list[Dependency] = []
        fields = (
            ("dependencies", "runtime"),
            ("devDependencies", "dev"),
            ("peerDependencies", "peer"),
            ("optionalDependencies", "optional"),
        )
        for field, scope in fields:
            for name, version in (data.get(field) or {}).items():
                dependencies.append(
                    Dependency(
                        name=str(name),
                        version=_clean_version(str(version)),
                        scope=scope,
                        ecosystem="javascript",
                        source=relative,
                    )
                )
        package_manager = ""
        declared = data.get("packageManager")
        if isinstance(declared, str) and declared:
            package_manager = declared.split("@", 1)[0]
        return dependencies, package_manager or "npm"

    # ----------------------------------------------------------------- Java

    def _pom(self, path: Path, relative: str) -> tuple[list[Dependency], str | None]:
        tree = ET.fromstring(self._text(path))
        dependencies: list[Dependency] = []
        for element in tree.iter():
            if _local_name(element.tag) != "dependency":
                continue
            fields: dict[str, str] = {}
            for child in element:
                fields[_local_name(child.tag)] = (child.text or "").strip()
            artifact = fields.get("artifactId")
            if not artifact:
                continue
            group = fields.get("groupId", "")
            name = f"{group}:{artifact}" if group else artifact
            scope = fields.get("scope", "compile")
            dependencies.append(
                Dependency(
                    name=name,
                    version=fields.get("version") or None,
                    scope="dev" if scope == "test" else (
                        "build" if scope in {"provided", "import"} else "runtime"
                    ),
                    ecosystem="java",
                    source=relative,
                )
            )
        return dependencies, "maven"

    def _gradle(
        self, path: Path, relative: str
    ) -> tuple[list[Dependency], str | None]:
        dependencies: list[Dependency] = []
        for match in _GRADLE_DEP.finditer(self._text(path)):
            coordinate = match.group("coord")
            parts = coordinate.split(":")
            if len(parts) < 2:
                continue
            name = ":".join(parts[:2]) if len(parts) >= 2 else coordinate
            version = parts[2] if len(parts) > 2 else None
            configuration = match.group("conf")
            if configuration.startswith("test"):
                scope = "dev"
            elif configuration in _GRADLE_BUILD_CONFIGURATIONS:
                scope = "build"
            else:
                scope = "runtime"
            dependencies.append(
                Dependency(
                    name=name,
                    version=version,
                    scope=scope,
                    ecosystem="java",
                    source=relative,
                )
            )
        return dependencies, "gradle"

    # ------------------------------------------------------------- Rust / Go

    def _cargo(self, path: Path, relative: str) -> tuple[list[Dependency], str | None]:
        data = tomllib.loads(self._text(path))
        dependencies: list[Dependency] = []

        def add_section(section: dict, scope: str) -> None:
            for name, value in (section or {}).items():
                real_name = str(name)
                version: str | None = None
                if isinstance(value, str):
                    version = _clean_version(value)
                elif isinstance(value, dict):
                    real_name = str(value.get("package") or name)
                    raw = value.get("version")
                    version = _clean_version(str(raw)) if raw else None
                dependencies.append(
                    Dependency(
                        name=real_name,
                        version=version,
                        scope=scope,
                        ecosystem="rust",
                        source=relative,
                    )
                )

        add_section(data.get("dependencies") or {}, "runtime")
        add_section(data.get("dev-dependencies") or {}, "dev")
        add_section(data.get("build-dependencies") or {}, "build")
        workspace = data.get("workspace", {}).get("dependencies") or {}
        add_section(workspace, "runtime")
        return dependencies, "cargo"

    def _go_mod(self, path: Path, relative: str) -> tuple[list[Dependency], str | None]:
        text = self._text(path)
        dependencies: list[Dependency] = []
        in_block = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("require ("):
                in_block = True
                continue
            if in_block and stripped == ")":
                in_block = False
                continue
            if stripped.startswith("require "):
                stripped = stripped[len("require ") :].strip()
            elif not in_block:
                continue
            if not stripped or stripped.startswith("//"):
                continue
            match = _GO_REQUIRE_LINE.match(stripped)
            if not match:
                continue
            dependencies.append(
                Dependency(
                    name=match.group(1),
                    version=match.group(2),
                    scope="runtime",
                    ecosystem="go",
                    source=relative,
                )
            )
        return dependencies, "go modules"

    # -------------------------------------------------------- Ruby / PHP / other

    def _gemfile(
        self, path: Path, relative: str
    ) -> tuple[list[Dependency], str | None]:
        dependencies: list[Dependency] = []
        for line in self._text(path).splitlines():
            match = _GEM.match(line)
            if not match:
                continue
            dependencies.append(
                Dependency(
                    name=match.group(1),
                    version=_clean_version(match.group(2) or ""),
                    scope="runtime",
                    ecosystem="ruby",
                    source=relative,
                )
            )
        return dependencies, "bundler"

    def _composer(
        self, path: Path, relative: str
    ) -> tuple[list[Dependency], str | None]:
        data = json.loads(self._text(path) or "{}")
        dependencies: list[Dependency] = []
        for field, scope in (("require", "runtime"), ("require-dev", "dev")):
            for name, version in (data.get(field) or {}).items():
                if str(name).lower() == "php":
                    continue
                dependencies.append(
                    Dependency(
                        name=str(name),
                        version=_clean_version(str(version)),
                        scope=scope,
                        ecosystem="php",
                        source=relative,
                    )
                )
        return dependencies, "composer"

    def _pubspec(
        self, path: Path, relative: str
    ) -> tuple[list[Dependency], str | None]:
        dependencies: list[Dependency] = []
        section: str | None = None
        entry_indent: int | None = None
        for line in self._text(path).splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if not line.startswith((" ", "\t")):
                key = line.split(":", 1)[0].strip()
                section = {
                    "dependencies": "runtime",
                    "dev_dependencies": "dev",
                }.get(key)
                entry_indent = None
                continue
            if section is None:
                continue
            indent = len(line) - len(line.lstrip(" \t"))
            if entry_indent is None:
                entry_indent = indent
            elif indent != entry_indent:
                # Nested ``hosted:``/``git:``/``version:`` mappings below a
                # dependency are not themselves dependencies.
                continue
            stripped = line.strip()
            if ":" not in stripped:
                continue
            name, value = stripped.split(":", 1)
            name = name.strip()
            value = value.strip().strip("'\"")
            if not name or name in {"flutter", "sdk"}:
                continue
            dependencies.append(
                Dependency(
                    name=name,
                    version=value or None,
                    scope=section,
                    ecosystem="dart",
                    source=relative,
                )
            )
        return dependencies, "pub"

    def _swiftpm(
        self, path: Path, relative: str
    ) -> tuple[list[Dependency], str | None]:
        dependencies: list[Dependency] = []
        for match in _PACKAGE_SWIFT.finditer(self._text(path)):
            url = match.group(2)
            name = match.group(1) or Path(url.rstrip("/")).name
            if name.endswith(".git"):
                name = name[:-4]
            dependencies.append(
                Dependency(
                    name=name,
                    version=None,
                    scope="runtime",
                    ecosystem="swift",
                    source=relative,
                )
            )
        return dependencies, "swiftpm"

    def _csproj(
        self, path: Path, relative: str
    ) -> tuple[list[Dependency], str | None]:
        dependencies: list[Dependency] = []
        try:
            tree = ET.fromstring(self._text(path))
        except ET.ParseError:
            return [], "nuget"
        scope = "dev" if "test" in path.name.lower() else "runtime"
        for element in tree.iter():
            if _local_name(element.tag) != "PackageReference":
                continue
            name = element.attrib.get("Include") or element.attrib.get("Update")
            if not name:
                continue
            version = element.attrib.get("Version")
            if version is None:
                for child in element:
                    if _local_name(child.tag) == "Version":
                        version = (child.text or "").strip()
            dependencies.append(
                Dependency(
                    name=name,
                    version=_clean_version(version or ""),
                    scope=scope,
                    ecosystem="dotnet",
                    source=relative,
                )
            )
        return dependencies, "nuget"

    def _sbt(self, path: Path, relative: str) -> tuple[list[Dependency], str | None]:
        dependencies: list[Dependency] = []
        for line in self._text(path).splitlines():
            if line.lstrip().startswith("//"):
                continue
            for match in _SBT.finditer(line):
                dependencies.append(
                    Dependency(
                        name=f"{match.group(1)}:{match.group(2)}",
                        version=match.group(3),
                        scope="dev" if "Test" in line else "runtime",
                        ecosystem="scala",
                        source=relative,
                    )
                )
        return dependencies, "sbt"

    def _mix(self, path: Path, relative: str) -> tuple[list[Dependency], str | None]:
        dependencies: list[Dependency] = []
        for line in self._text(path).splitlines():
            if line.lstrip().startswith("#"):
                continue
            for match in _MIX.finditer(line):
                dependencies.append(
                    Dependency(
                        name=match.group(1),
                        version=_clean_version(match.group(2) or ""),
                        scope="runtime",
                        ecosystem="elixir",
                        source=relative,
                    )
                )
        return dependencies, "mix"

    def _vcpkg(self, path: Path, relative: str) -> tuple[list[Dependency], str | None]:
        data = json.loads(self._text(path) or "{}")
        dependencies: list[Dependency] = []
        for entry in data.get("dependencies") or []:
            if isinstance(entry, str):
                name, version = entry, None
            elif isinstance(entry, dict):
                name, version = entry.get("name"), entry.get("version>=")
            else:
                continue
            if name:
                dependencies.append(
                    Dependency(
                        name=str(name),
                        version=str(version) if version else None,
                        scope="runtime",
                        ecosystem="cpp",
                        source=relative,
                    )
                )
        return dependencies, "vcpkg"


_SCOPE_ORDER = {
    "runtime": 0,
    "build": 1,
    "dev": 2,
    "peer": 3,
    "optional": 4,
    "unknown": 5,
}


def _poetry_version(value: object) -> str | None:
    if isinstance(value, str):
        return _clean_version(value)
    if isinstance(value, dict):
        raw = value.get("version")
        return _clean_version(str(raw)) if raw else None
    return None


def _literal(node: ast.AST) -> object:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None


def _recognise(relative: str) -> tuple[str, str, str] | None:
    """Return ``(kind, ecosystem, default package manager)`` for a manifest."""
    name = Path(relative).name
    lowered = name.lower()
    suffix = Path(name).suffix.lower()

    if (
        lowered.startswith(("requirements", "constraints"))
        and suffix == ".txt"
        or lowered in {"dev-requirements.txt", "test-requirements.txt"}
    ):
        return "requirements", "python", "pip"
    if lowered == "pyproject.toml":
        return "pyproject", "python", "pip"
    if lowered == "setup.py":
        return "setup.py", "python", "pip"
    if lowered == "setup.cfg":
        return "setup.cfg", "python", "pip"
    if lowered == "pipfile":
        return "pipfile", "python", "pipenv"
    if lowered == "package.json":
        return "package-json", "javascript", "npm"
    if lowered == "pom.xml":
        return "pom", "java", "maven"
    if lowered in {"build.gradle", "build.gradle.kts"}:
        return "gradle", "java", "gradle"
    if lowered == "cargo.toml":
        return "cargo", "rust", "cargo"
    if lowered == "go.mod":
        return "go-mod", "go", "go modules"
    if lowered == "gemfile":
        return "gemfile", "ruby", "bundler"
    if lowered == "composer.json":
        return "composer", "php", "composer"
    if lowered == "pubspec.yaml":
        return "pubspec", "dart", "pub"
    if lowered == "package.swift":
        return "swiftpm", "swift", "swiftpm"
    if suffix in {".csproj", ".fsproj", ".vbproj"}:
        return "csproj", "dotnet", "nuget"
    if lowered == "build.sbt":
        return "sbt", "scala", "sbt"
    if lowered == "mix.exs":
        return "mix", "elixir", "mix"
    if lowered == "vcpkg.json":
        return "vcpkg", "cpp", "vcpkg"
    return None


def _root_manifests(base: Path, already: Sequence[str]) -> list[str]:
    """Include root-level manifests even when the file scan was truncated."""
    found: list[str] = []
    try:
        entries = list(base.iterdir())
    except OSError:
        return found
    known = set(already)
    for entry in entries:
        if entry.is_symlink() or is_junction(entry) or not entry.is_file():
            continue
        relative = entry.name
        if relative in known:
            continue
        if _recognise(relative) is not None:
            found.append(relative)
    return found


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
