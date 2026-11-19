"""Project detection is evidence-based: every label comes from a real file."""

from __future__ import annotations

from pathlib import Path

from beast.intelligence.dependency_analyzer import DependencyAnalyzer
from beast.intelligence.models import ProjectDetection
from beast.intelligence.project_detector import ProjectDetector
from beast.workspace.scanning import scan_project


def detect(root: Path) -> ProjectDetection:
    scan = scan_project(root)
    dependencies = DependencyAnalyzer().analyze(root, files=scan.files)
    return ProjectDetector().detect(scan.files, dependencies, root=root)


def test_python_flask_project_is_detected_from_manifests_and_files(
    flask_project: Path,
) -> None:
    detection = detect(flask_project)

    assert detection.primary_language == "Python"
    assert "Python" in detection.language_names
    assert detection.frameworks == ("Flask",)
    assert detection.package_managers == ("pip",)
    assert "pytest" in detection.test_frameworks
    assert "app.py" in detection.entry_points
    assert "requirements.txt" in [manifest.path for manifest in detection.manifests]
    assert "requirements.txt" in detection.config_files
    assert detection.source_file_count == 2
    assert detection.test_file_count == 1
    assert "Python" in detection.summary_line()
    assert "Flask" in detection.summary_line()


def test_react_typescript_project_is_detected(react_project: Path) -> None:
    detection = detect(react_project)

    assert detection.primary_language == "TypeScript"
    assert detection.frameworks == ("React",)
    assert detection.package_managers == ("npm",)
    assert "Vitest" in detection.test_frameworks
    assert "Vite" in detection.build_systems
    assert "npm scripts" in detection.build_systems
    assert "src/main.tsx" in detection.entry_points
    assert "src/App.tsx" in detection.entry_points
    assert {"package.json", "tsconfig.json", "vite.config.ts"} <= set(
        detection.config_files
    )
    # Four application modules plus vite.config.ts; the test file is counted
    # separately in test_file_count.
    assert detection.source_file_count == 5
    assert detection.test_file_count == 1


def test_multi_language_project_reports_every_ecosystem(
    multilang_project: Path,
) -> None:
    detection = detect(multilang_project)

    assert detection.primary_language == "Python"
    assert {"Python", "TypeScript", "Go", "C++", "Vue"} <= set(
        detection.language_names
    )
    assert {"FastAPI", "Vue", "Gin"} <= set(detection.frameworks)
    assert {"pip", "npm", "go modules"} <= set(detection.package_managers)
    assert {"CMake", "Go modules"} <= set(detection.build_systems)
    assert {"service/main.py", "web/src/main.ts", "cmd/server/main.go"} <= set(
        detection.entry_points
    )
    assert detection.source_file_count == 6
    assert detection.test_file_count == 0


def test_empty_project_is_described_honestly(write_files) -> None:
    root = write_files({"notes.txt": "nothing to see\n"})

    detection = detect(root)

    assert detection.languages == ()
    assert detection.primary_language is None
    assert detection.frameworks == ()
    assert detection.source_file_count == 0
    assert detection.summary_line() == "unknown · 0 source files"


def test_framework_specific_config_files_are_evidence(write_files) -> None:
    next_root = write_files({"next.config.js": "module.exports = {};\n"})
    assert detect(next_root).frameworks == ("Next.js",)

    (next_root / "next.config.js").unlink()
    (next_root / "angular.json").write_text("{}", encoding="utf-8")
    assert detect(next_root).frameworks == ("Angular",)

    (next_root / "angular.json").unlink()
    (next_root / "manage.py").write_text("print('hello')\n", encoding="utf-8")
    assert detect(next_root).frameworks == ("Django",)


def test_test_framework_can_come_from_a_config_file(write_files) -> None:
    root = write_files(
        {
            "vitest.config.ts": "export default {};\n",
            "src/example.ts": "export const x = 1;\n",
        }
    )

    assert "Vitest" in detect(root).test_frameworks


def test_project_name_does_not_invent_technology(write_files) -> None:
    root = write_files({"docs/notes.md": "A shop, perhaps.\n"})
    detection = detect(root)

    # The directory fixture is named after the test, never after a framework.
    assert detection.name == root.name
    assert detection.frameworks == ()
    assert detection.package_managers == ()
    assert detection.primary_language is None


def test_malformed_config_shapes_do_not_crash_detection(write_files) -> None:
    root = write_files(
        {
            "package.json": '{"dependencies": "nope", "scripts": 7, "bin": 9}',
            "pyproject.toml": (
                "[project]\n"
                'name = "x"\n'
                "dependencies = \"nope\"\n"
                "[build-system]\n"
                'build-backend = "not.a.real.backend"\n'
            ),
            "app.py": "x = 1\n",
        }
    )

    detection = detect(root)

    assert detection.primary_language == "Python"
    # The files themselves are manager evidence; their broken contents are not
    # evidence for scripts or a build backend.
    assert set(detection.package_managers) == {"npm", "pip"}
    assert "npm scripts" not in detection.build_systems
    assert detection.frameworks == ()


def test_entry_point_paths_keep_dot_directories_and_refuse_traversal(
    write_files,
) -> None:
    root = write_files(
        {
            "package.json": (
                '{"main": "./.app/bootstrap.js", "module": "../outside-entry.js"}'
            ),
            ".app/bootstrap.js": "export const start = () => {};\n",
            "outside-entry.js": "module.exports = {};\n",
        }
    )

    detection = detect(root)

    # ``./`` is stripped once; ``.app`` is a real directory.
    assert ".app/bootstrap.js" in detection.entry_points
    # ``../outside-entry.js`` escapes the package root and must not be
    # rewritten to ``outside-entry.js``.
    assert "outside-entry.js" not in detection.entry_points
