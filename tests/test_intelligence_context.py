"""Context relevance, budgets and the secret boundary."""

from __future__ import annotations

from pathlib import Path

from beast.intelligence import ContextBuilder, ContextLimits, ProjectIntelligence


def test_authentication_question_selects_the_authentication_files(
    flask_project: Path,
) -> None:
    intelligence = ProjectIntelligence(flask_project)

    bundle = intelligence.context_for(
        "Find the authentication implementation in this project and explain how it works."
    )

    assert bundle.files
    assert bundle.files[0] == "auth.py"
    assert "tests/test_auth.py" in bundle.files
    assert "## Project context" in bundle.text
    assert "### `auth.py`" in bundle.text
    assert "authenticate" in bundle.text
    assert ".env" not in bundle.files
    assert "not-for-the-model" not in bundle.text
    assert bundle.is_empty is False


def test_content_only_match_is_found(write_files) -> None:
    root = write_files(
        {
            "notes.py": (
                "# The quokka lives in a burrow near the coast.\n"
                "def helper():\n"
                "    return 1\n"
            ),
            "unrelated.py": "def other():\n    return 2\n",
        }
    )

    bundle = ProjectIntelligence(root).context_for("Where is the quokka burrow?")

    assert "notes.py" in bundle.files
    assert "### `notes.py`" in bundle.text
    assert "burrow" in bundle.text


def test_budget_bounds_the_bundle(write_files) -> None:
    files = {}
    for index in range(4):
        body = "\n".join(
            f'WIDGET_{index}_{line} = "widget widget widget {line}"'
            for line in range(40)
        )
        files[f"widget_{index}.py"] = body + "\n"
    root = write_files(files)

    intelligence = ProjectIntelligence(
        root,
        context_limits=ContextLimits(
            max_characters=700,
            max_files=4,
            max_file_characters=300,
            excerpt_lines=15,
        ),
    )
    bundle = intelligence.context_for("widget configuration")

    assert bundle.truncated is True
    assert bundle.characters <= 1_200
    assert "truncated" in bundle.text.lower()
    assert len(bundle.files) <= 4


def test_max_files_is_respected(write_files) -> None:
    files = {
        f"module_{index}.py": (
            f"def widget_{index}():\n"
            f'    return "widget number {index}"\n'
        )
        for index in range(20)
    }
    root = write_files(files)
    intelligence = ProjectIntelligence(
        root, context_limits=ContextLimits(max_files=3)
    )

    bundle = intelligence.context_for("widget")

    assert 0 < len(bundle.files) <= 3


def test_overview_question_includes_structure_and_dependencies(
    flask_project: Path,
) -> None:
    bundle = ProjectIntelligence(flask_project).context_for(
        "Understand this project and explain its architecture, technologies, "
        "dependencies and how the main components interact."
    )

    assert "## Project structure" in bundle.text
    assert "### Dependencies" in bundle.text
    assert "Flask 3.0.0" in bundle.text
    assert "Frameworks: Flask" in bundle.text
    assert "app.py" in bundle.text
    assert "auth.py" in bundle.text


def test_dependency_question_includes_declared_versions(flask_project: Path) -> None:
    bundle = ProjectIntelligence(flask_project).context_for(
        "What dependencies does this project use?"
    )

    assert "### Dependencies" in bundle.text
    assert "Flask 3.0.0" in bundle.text
    assert "pytest 8.0.0" in bundle.text


def test_normal_question_does_not_render_the_dependency_section(
    flask_project: Path,
) -> None:
    bundle = ProjectIntelligence(flask_project).context_for(
        "How is the login route wired to authentication?"
    )

    assert "### Dependencies" not in bundle.text


def test_secrets_are_redacted_before_the_model_sees_them(write_files) -> None:
    root = write_files(
        {
            "config.py": (
                'AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"\n'
                "\n"
                "def load_config():\n"
                '    return {"region": "us-east-1"}\n'
            ),
            ".env": "AWS_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE\n",
        }
    )

    bundle = ProjectIntelligence(root).context_for("AWS access key configuration")

    assert "config.py" in bundle.files
    assert "AKIAIOSFODNN7EXAMPLE" not in bundle.text
    assert "[REDACTED]" in bundle.text
    assert bundle.redactions >= 1
    assert ".env" not in bundle.files


def test_context_reader_cannot_escape_the_workspace(
    tmp_path: Path, write_files
) -> None:
    root = write_files({"app.py": "x = 1\n"})
    outside = tmp_path.parent / "outside.py"
    outside.write_text("outside = True\n", encoding="utf-8")
    (root / ".env").write_text("TOKEN=secret\n", encoding="utf-8")

    builder = ContextBuilder(root)

    assert builder._read("../outside.py") is None
    assert builder._read(str(outside)) is None
    assert builder._read(".env") is None
    # Reading preserves bytes; Windows writes CRLF, so compare logical lines.
    assert (builder._read("app.py") or "").splitlines() == ["x = 1"]
