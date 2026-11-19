"""Dependency analysis across ecosystems, without installing or executing code."""

from __future__ import annotations

from pathlib import Path

from beast.intelligence.dependency_analyzer import DependencyAnalyzer
from beast.intelligence.models import DependencyReport


def analyze(root: Path, **kwargs) -> DependencyReport:
    return DependencyAnalyzer().analyze(root, **kwargs)


def by_name(report: DependencyReport, name: str) -> list:
    return [
        dependency
        for dependency in report.dependencies
        if dependency.name.lower() == name.lower()
    ]


def test_requirements_files_and_nested_includes(write_files) -> None:
    root = write_files(
        {
            "requirements.txt": (
                "# runtime\n"
                "Flask==3.0.0\n"
                "requests>=2.31\n"
                "-r dev-requirements.txt\n"
                "-c constraints.txt\n"
            ),
            "dev-requirements.txt": "pytest==8.0.0\n",
        }
    )
    report = analyze(root)

    flask = by_name(report, "Flask")[0]
    assert flask.version == "3.0.0"
    assert flask.scope == "runtime"
    assert flask.ecosystem == "python"
    assert flask.source == "requirements.txt"
    assert by_name(report, "requests")[0].version == ">=2.31"
    pytest = by_name(report, "pytest")[0]
    assert pytest.scope == "dev"
    assert pytest.version == "8.0.0"
    assert report.package_managers == ("pip",)


def test_requirements_includes_cannot_escape_the_manifest_directory(
    write_files,
) -> None:
    outside = write_files({"evil-requirements.txt": "evil-package==1.0\n"})
    nested = outside / "nested"
    nested.mkdir()
    (nested / "requirements.txt").write_text(
        "-r ../evil-requirements.txt\n", encoding="utf-8"
    )

    report = analyze(nested)

    assert by_name(report, "evil-package") == []


def test_pyproject_pep621_optional_and_build_dependencies(write_files) -> None:
    root = write_files(
        {
            "pyproject.toml": (
                "[build-system]\n"
                'requires = ["setuptools>=68", "wheel"]\n'
                'build-backend = "setuptools.build_meta"\n'
                "\n"
                "[project]\n"
                'name = "demo"\n'
                'dependencies = ["fastapi>=0.110", "httpx == 0.27 ; python_version>=3.9"]\n'
                "\n"
                "[project.optional-dependencies]\n"
                'dev = ["pytest>=8"]\n'
                'docs = ["mkdocs"]\n'
                "\n"
                "[dependency-groups]\n"
                'test = ["coverage"]\n'
            )
        }
    )
    report = analyze(root)

    assert by_name(report, "fastapi")[0].scope == "runtime"
    assert by_name(report, "fastapi")[0].version == ">=0.110"
    assert by_name(report, "httpx")[0].version == "0.27"
    assert by_name(report, "pytest")[0].scope == "dev"
    assert by_name(report, "coverage")[0].scope == "dev"
    assert by_name(report, "mkdocs")[0].scope == "optional"
    assert {item.name for item in report.dependencies if item.scope == "build"} == {
        "setuptools",
        "wheel",
    }
    assert report.package_managers == ("pip",)


def test_poetry_pyproject_sets_the_package_manager(write_files) -> None:
    root = write_files(
        {
            "pyproject.toml": (
                "[tool.poetry]\n"
                'name = "demo"\n'
                "\n"
                "[tool.poetry.dependencies]\n"
                'python = "^3.11"\n'
                'django = "^4.2"\n'
                "\n"
                "[tool.poetry.group.dev.dependencies]\n"
                'pytest = "^8.0"\n'
                "\n"
                "[build-system]\n"
                'requires = ["poetry-core"]\n'
                'build-backend = "poetry.core.masonry.api"\n'
            )
        }
    )
    report = analyze(root)

    assert report.package_managers == ("poetry",)
    assert by_name(report, "django")[0].scope == "runtime"
    assert by_name(report, "django")[0].version == "^4.2"
    assert by_name(report, "pytest")[0].scope == "dev"
    assert by_name(report, "python") == []


def test_setup_py_is_parsed_but_never_executed(write_files) -> None:
    root = write_files(
        {
            "setup.py": (
                "from pathlib import Path\n"
                'Path("EXECUTED").write_text("this would be bad")\n'
                "from setuptools import setup\n"
                "raise RuntimeError('must not execute')\n"
                'setup(name="demo", install_requires=["Flask==2.0"], '
                'tests_require=["pytest"], setup_requires=["wheel"], '
                'extras_require={"dev": ["black"]})\n'
            )
        }
    )
    report = analyze(root)

    assert not (root / "EXECUTED").exists()
    assert by_name(report, "Flask")[0].scope == "runtime"
    assert by_name(report, "pytest")[0].scope == "dev"
    assert by_name(report, "wheel")[0].scope == "build"
    assert by_name(report, "black")[0].scope == "dev"


def test_setup_cfg_dependencies(write_files) -> None:
    root = write_files(
        {
            "setup.cfg": (
                "[options]\n"
                "install_requires =\n"
                "    flask>=2.0\n"
                "    requests\n"
                "setup_requires =\n"
                "    wheel\n"
                "\n"
                "[options.extras_require]\n"
                "dev =\n"
                "    black\n"
            )
        }
    )
    report = analyze(root)

    assert by_name(report, "flask")[0].version == ">=2.0"
    assert by_name(report, "requests")[0].version is None
    assert by_name(report, "wheel")[0].scope == "build"
    assert by_name(report, "black")[0].scope == "dev"


def test_package_json_scopes_and_lockfile_manager_precedence(write_files) -> None:
    root = write_files(
        {
            "package.json": (
                '{"name":"web",'
                '"dependencies":{"react":"^18.2.0"},'
                '"devDependencies":{"vite":"^5.0.0"},'
                '"peerDependencies":{"react-dom":"^18.2.0"},'
                '"optionalDependencies":{"fsevents":"^2.3.0"}}'
            ),
            "yarn.lock": "# yarn lockfile v1\n",
        }
    )
    report = analyze(root)

    assert by_name(report, "react")[0].scope == "runtime"
    assert by_name(report, "vite")[0].scope == "dev"
    assert by_name(report, "react-dom")[0].scope == "peer"
    assert by_name(report, "fsevents")[0].scope == "optional"
    assert "yarn" in report.package_managers
    assert "npm" not in report.package_managers


def test_package_manager_field_wins_over_default(write_files) -> None:
    root = write_files(
        {
            "package.json": (
                '{"name":"web","packageManager":"pnpm@9.1.0",'
                '"dependencies":{"vue":"^3.4.0"}}'
            )
        }
    )
    report = analyze(root)

    assert report.package_managers[0] == "pnpm"


def test_maven_pom_dependencies_and_scopes(write_files) -> None:
    root = write_files(
        {
            "pom.xml": (
                "<project>"
                "<dependencies>"
                "<dependency>"
                "<groupId>org.springframework.boot</groupId>"
                "<artifactId>spring-boot-starter-web</artifactId>"
                "<version>3.2.0</version>"
                "</dependency>"
                "<dependency>"
                "<groupId>org.junit.jupiter</groupId>"
                "<artifactId>junit-jupiter</artifactId>"
                "<version>5.10.0</version>"
                "<scope>test</scope>"
                "</dependency>"
                "<dependency>"
                "<groupId>org.projectlombok</groupId>"
                "<artifactId>lombok</artifactId>"
                "<scope>provided</scope>"
                "</dependency>"
                "</dependencies>"
                "</project>"
            )
        }
    )
    report = analyze(root)

    assert report.package_managers == ("maven",)
    assert by_name(report, "org.springframework.boot:spring-boot-starter-web")[
        0
    ].version == "3.2.0"
    assert by_name(report, "org.junit.jupiter:junit-jupiter")[0].scope == "dev"
    assert by_name(report, "org.projectlombok:lombok")[0].scope == "build"


def test_gradle_dependencies_and_scopes(write_files) -> None:
    root = write_files(
        {
            "build.gradle": (
                "dependencies {\n"
                "    implementation 'org.springframework.boot:spring-boot-starter-web:3.2.0'\n"
                "    testImplementation 'org.junit.jupiter:junit-jupiter:5.10.0'\n"
                "    compileOnly 'org.projectlombok:lombok:1.18.30'\n"
                '    runtimeOnly "com.h2database:h2:2.2.224"\n'
                "}\n"
            )
        }
    )
    report = analyze(root)

    assert report.package_managers == ("gradle",)
    assert by_name(report, "org.springframework.boot:spring-boot-starter-web")[
        0
    ].scope == "runtime"
    assert by_name(report, "org.junit.jupiter:junit-jupiter")[0].scope == "dev"
    assert by_name(report, "org.projectlombok:lombok")[0].scope == "build"
    assert by_name(report, "com.h2database:h2")[0].version == "2.2.224"


def test_go_mod_block_and_single_require(write_files) -> None:
    root = write_files(
        {
            "go.mod": (
                "module example.com/service\n"
                "\n"
                "go 1.22\n"
                "\n"
                "require github.com/gin-gonic/gin v1.9.0\n"
                "\n"
                "require (\n"
                "\tgithub.com/stretchr/testify v1.9.0\n"
                "\tgolang.org/x/sync v0.7.0\n"
                ")\n"
            )
        }
    )
    report = analyze(root)

    assert report.package_managers == ("go modules",)
    assert by_name(report, "github.com/gin-gonic/gin")[0].version == "v1.9.0"
    assert by_name(report, "github.com/stretchr/testify")[0].scope == "runtime"
    assert by_name(report, "golang.org/x/sync")[0].version == "v0.7.0"


def test_cargo_manifest_scopes(write_files) -> None:
    root = write_files(
        {
            "Cargo.toml": (
                "[package]\n"
                'name = "demo"\n'
                'version = "0.1.0"\n'
                "\n"
                "[dependencies]\n"
                'serde = { version = "1.0", features = ["derive"] }\n'
                'tokio = "1.38"\n'
                "\n"
                "[dev-dependencies]\n"
                'pretty_assertions = "1.4"\n'
                "\n"
                "[build-dependencies]\n"
                'cc = "1.0"\n'
            )
        }
    )
    report = analyze(root)

    assert report.package_managers == ("cargo",)
    assert by_name(report, "serde")[0].version == "1.0"
    assert by_name(report, "serde")[0].scope == "runtime"
    assert by_name(report, "tokio")[0].version == "1.38"
    assert by_name(report, "pretty_assertions")[0].scope == "dev"
    assert by_name(report, "cc")[0].scope == "build"


def test_overlapping_declarations_are_deduplicated_by_scope(write_files) -> None:
    root = write_files(
        {
            "requirements.txt": "Flask==3.0.0\npytest==8.0.0\n",
            "pyproject.toml": (
                "[project]\n"
                'dependencies = ["Flask>=3.0"]\n'
                "\n"
                "[project.optional-dependencies]\n"
                'dev = ["pytest>=8"]\n'
            ),
        }
    )
    report = analyze(root)

    flask_runtime = [
        item
        for item in by_name(report, "Flask")
        if item.scope == "runtime"
    ]
    assert len(flask_runtime) == 1
    pytest_scopes = {item.scope for item in by_name(report, "pytest")}
    assert pytest_scopes == {"runtime", "dev"}


def test_invalid_manifests_produce_a_warning_not_a_crash(write_files) -> None:
    root = write_files({"pom.xml": "<project><dependencies>"})
    report = analyze(root)

    assert report.dependencies == ()
    assert report.warnings
    assert report.warnings[0].startswith("pom.xml: could not parse")


def test_root_manifests_are_found_even_with_no_scanned_paths(
    flask_project: Path,
) -> None:
    report = analyze(flask_project, relative_paths=[])

    assert by_name(report, "Flask")
    assert report.manifests[0].path == "requirements.txt"


def test_ruby_php_and_dart_manifests(write_files) -> None:
    root = write_files(
        {
            "Gemfile": (
                'source "https://rubygems.org"\n'
                'gem "rails", "~> 7.1"\n'
                'gem "puma"\n'
            ),
            "composer.json": (
                '{"require":{"php":">=8.1","laravel/framework":"^10.0"},'
                '"require-dev":{"phpunit/phpunit":"^10.5"}}'
            ),
            "pubspec.yaml": (
                "name: app\n"
                "dependencies:\n"
                "  flutter:\n"
                "    sdk: flutter\n"
                "  http: ^1.2.0\n"
                "dev_dependencies:\n"
                "  test: ^1.24.0\n"
            ),
        }
    )
    report = analyze(root)

    assert {"bundler", "composer", "pub"} <= set(report.package_managers)
    assert by_name(report, "rails")[0].version == "~> 7.1"
    assert by_name(report, "rails")[0].ecosystem == "ruby"
    assert by_name(report, "puma")[0].version is None
    assert by_name(report, "laravel/framework")[0].ecosystem == "php"
    assert by_name(report, "phpunit/phpunit")[0].scope == "dev"
    assert by_name(report, "php") == []
    assert by_name(report, "http")[0].ecosystem == "dart"
    assert by_name(report, "http")[0].scope == "runtime"
    assert by_name(report, "test")[0].scope == "dev"
    assert by_name(report, "flutter") == []
    assert by_name(report, "sdk") == []


def test_swift_dotnet_scala_elixir_and_vcpkg_manifests(write_files) -> None:
    root = write_files(
        {
            "Package.swift": (
                "let package = Package(\n"
                "    dependencies: [\n"
                '        .package(name: "Vapor", '
                'url: "https://github.com/vapor/vapor.git", from: "4.0.0"),\n'
                '        .package(url: "https://github.com/apple/'
                'swift-argument-parser.git", from: "1.3.0")\n'
                "    ]\n"
                ")\n"
            ),
            "App.csproj": (
                '<Project Sdk="Microsoft.NET.Sdk"><ItemGroup>'
                '<PackageReference Include="Newtonsoft.Json" Version="13.0.3" />'
                "<PackageReference Include=\"xunit\">"
                "<Version>2.7.0</Version></PackageReference>"
                "</ItemGroup></Project>"
            ),
            "Tests.csproj": (
                "<Project><ItemGroup>"
                '<PackageReference Include="Microsoft.NET.Test.Sdk" Version="17.9.0" />'
                "</ItemGroup></Project>"
            ),
            "build.sbt": (
                "libraryDependencies ++= Seq(\n"
                '  "org.typelevel" % "cats-core" % "2.10.0",\n'
                '  "org.scalatest" % "scalatest" % "3.2.18" % Test\n'
                ")\n"
            ),
            "mix.exs": (
                "defp deps do\n"
                "  [\n"
                '    {:phoenix, "~> 1.7.0"},\n'
                "  ]\n"
                "end\n"
            ),
            "vcpkg.json": (
                '{"dependencies":["fmt",{"name":"boost","version>=":"1.83.0"}]}'
            ),
        }
    )
    report = analyze(root)

    assert {"swiftpm", "nuget", "sbt", "mix", "vcpkg"} <= set(
        report.package_managers
    )
    assert by_name(report, "Vapor")[0].ecosystem == "swift"
    assert by_name(report, "swift-argument-parser")[0].version is None
    assert by_name(report, "Newtonsoft.Json")[0].version == "13.0.3"
    assert by_name(report, "Newtonsoft.Json")[0].scope == "runtime"
    assert by_name(report, "xunit")[0].version == "2.7.0"
    assert by_name(report, "Microsoft.NET.Test.Sdk")[0].scope == "dev"
    assert by_name(report, "org.typelevel:cats-core")[0].scope == "runtime"
    assert by_name(report, "org.scalatest:scalatest")[0].scope == "dev"
    assert by_name(report, "phoenix")[0].version == "~> 1.7.0"
    assert by_name(report, "fmt")[0].ecosystem == "cpp"
    assert by_name(report, "boost")[0].version == "1.83.0"


def test_pipfile_dependencies_are_read_without_running_pip(write_files) -> None:
    root = write_files(
        {
            "Pipfile": (
                '[[source]]\n'
                'url = "https://pypi.org/simple"\n'
                "\n"
                "[packages]\n"
                'flask = "==3.0.0"\n'
                'requests = "*"\n'
                "\n"
                "[dev-packages]\n"
                'pytest = "==8.0.0"\n'
            )
        }
    )
    report = analyze(root)

    assert report.package_managers == ("pipenv",)
    assert by_name(report, "flask")[0].version == "3.0.0"
    assert by_name(report, "flask")[0].scope == "runtime"
    assert by_name(report, "pytest")[0].scope == "dev"


def test_render_groups_scopes_and_reports_the_remainder(
    flask_project: Path,
) -> None:
    report = analyze(flask_project)

    rendered = report.render(limit=1)

    assert "runtime:" in rendered
    assert "  - Flask 3.0.0  (requirements.txt)" in rendered
    assert "1 more" in rendered
    # A requirements file is runtime evidence; no dev group is invented.
    assert "dev:" not in rendered


def test_requirements_accept_direct_url_references(write_files) -> None:
    root = write_files(
        {
            "requirements.txt": (
                "mypkg @ https://example.com/mypkg-1.0.whl\n"
                "-e ./local-package\n"
                "\n"
            )
        }
    )
    report = analyze(root)

    assert "mypkg-1.0.whl" in (by_name(report, "mypkg")[0].version or "")
    assert by_name(report, "local-package") == []


def test_multi_language_manifests_are_all_read(multilang_project: Path) -> None:
    report = analyze(multilang_project)

    assert {"pip", "npm", "go modules"} <= set(report.package_managers)
    assert by_name(report, "fastapi")[0].ecosystem == "python"
    assert by_name(report, "vue")[0].ecosystem == "javascript"
    assert by_name(report, "github.com/gin-gonic/gin")[0].ecosystem == "go"


def test_malformed_manifests_produce_warnings_not_fake_dependencies(
    write_files,
) -> None:
    root = write_files(
        {
            "package.json": '{"dependencies": "not-a-mapping"}',
            "pom.xml": "<project><dependencies><dependency>",
            "pyproject.toml": (
                "[project]\n"
                'dependencies = "not-a-list"\n'
                "optional-dependencies = 5\n"
                "[tool.poetry]\n"
                'dependencies = "not-a-table"\n'
                "[build-system]\n"
                "requires = 7\n"
            ),
        }
    )

    report = analyze(root)

    # A string iterated as a list used to yield one dependency per character.
    assert "n" not in {dependency.name.lower() for dependency in report.dependencies}
    assert any("package.json" in warning for warning in report.warnings)
    assert any("pom.xml" in warning for warning in report.warnings)
    assert {"npm", "maven"} <= set(report.package_managers)
    # The ``[tool.poetry]`` table is still evidence, even when malformed.
    assert "poetry" in report.package_managers


def test_setup_py_extras_require_tolerates_non_list_values(write_files) -> None:
    root = write_files(
        {
            "setup.py": (
                "from setuptools import setup\n"
                'setup(name="demo", install_requires=["Flask==2.0"], '
                'extras_require={"dev": "pytest", "docs": 5})\n'
            )
        }
    )

    report = analyze(root)

    # A string or number must not be iterated into character-wise fake deps,
    # and the valid install_requires entry must survive.
    assert by_name(report, "Flask")[0].scope == "runtime"
    assert {dependency.name.lower() for dependency in report.dependencies} == {"flask"}


def test_pubspec_nested_mapping_keys_are_not_dependencies(write_files) -> None:
    root = write_files(
        {
            "pubspec.yaml": (
                "name: app\n"
                "dependencies:\n"
                "  http:\n"
                "    hosted:\n"
                "      name: http\n"
                "      url: https://pub.dev\n"
                "    version: ^1.2.0\n"
                "  git_dep:\n"
                "    git:\n"
                "      url: https://example.test/repo.git\n"
                "      ref: main\n"
            )
        }
    )

    report = analyze(root)

    names = {dependency.name for dependency in report.dependencies}
    assert names == {"http", "git_dep"}
    assert by_name(report, "version") == []
    assert by_name(report, "hosted") == []
    assert by_name(report, "git") == []
    assert by_name(report, "ref") == []


def test_gradle_compile_only_api_is_a_build_scope(write_files) -> None:
    root = write_files(
        {
            "build.gradle": (
                "dependencies {\n"
                "    compileOnlyApi 'org.jetbrains:annotations:24.0.0'\n"
                "    implementation 'com.example:core:1.0'\n"
                "}\n"
            )
        }
    )

    report = analyze(root)

    assert by_name(report, "org.jetbrains:annotations")[0].scope == "build"
    assert by_name(report, "com.example:core")[0].scope == "runtime"
