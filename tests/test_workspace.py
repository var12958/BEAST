"""Workspace detection and path containment."""

from __future__ import annotations

from pathlib import Path

import pytest

from beast.errors import WorkspaceError
from beast.workspace import Workspace


def test_from_cwd_captures_the_launch_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "my-project"
    project.mkdir()
    monkeypatch.chdir(project)

    workspace = Workspace.from_cwd()

    assert workspace.root == project.resolve()
    assert workspace.name == "my-project"


def test_missing_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceError) as excinfo:
        Workspace(tmp_path / "does-not-exist")

    assert "not an existing directory" in excinfo.value.message
    assert excinfo.value.hint


def test_a_file_is_not_a_valid_workspace(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("x", encoding="utf-8")

    with pytest.raises(WorkspaceError):
        Workspace(target)


def test_home_directory_displays_as_tilde() -> None:
    assert Workspace(Path.home()).display() == "~"


def test_paths_under_home_are_shortened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        Path, "home", classmethod(lambda cls: tmp_path.resolve())
    )
    project = tmp_path / "code" / "app"
    project.mkdir(parents=True)

    assert Workspace(project).display() == "~/code/app"


def test_paths_outside_home_stay_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        Path, "home", classmethod(lambda cls: tmp_path / "elsewhere")
    )

    assert Workspace(tmp_path).display() == str(tmp_path.resolve())


def test_resolve_keeps_paths_inside_the_project(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    workspace = Workspace(tmp_path)

    resolved = workspace.resolve("src/app.py")

    assert resolved == (tmp_path / "src" / "app.py").resolve()


def test_resolve_accepts_the_root_itself(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)

    assert workspace.resolve(".") == tmp_path.resolve()


@pytest.mark.parametrize("escape", ["../outside.txt", "..", "src/../../outside"])
def test_resolve_refuses_to_escape_the_project(
    tmp_path: Path, escape: str
) -> None:
    workspace = Workspace(tmp_path)

    with pytest.raises(WorkspaceError) as excinfo:
        workspace.resolve(escape)

    assert "outside the workspace" in excinfo.value.message


def test_resolve_refuses_an_absolute_path_outside(tmp_path: Path) -> None:
    (tmp_path / "project").mkdir()
    workspace = Workspace(tmp_path / "project")
    (tmp_path / "secret.txt").write_text("no", encoding="utf-8")

    with pytest.raises(WorkspaceError):
        workspace.resolve(tmp_path / "secret.txt")
