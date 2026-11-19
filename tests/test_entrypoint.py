"""The ``asap`` console script wrapper."""

from __future__ import annotations

import builtins
import sys
import types

import pytest

from beast import entrypoint


def test_missing_dependency_is_explained(capsys: pytest.CaptureFixture[str]) -> None:
    entrypoint.report_missing_dependency(
        ModuleNotFoundError("No module named 'typer'", name="typer")
    )

    error = capsys.readouterr().err
    assert "'typer' is missing" in error
    assert "python -m pip install -e ." in error


def test_a_broken_beast_install_gets_its_own_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    entrypoint.report_missing_dependency(
        ModuleNotFoundError("No module named 'beast.cli'", name="beast.cli")
    )

    assert "not installed correctly" in capsys.readouterr().err


def test_main_explains_an_import_failure_and_exits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "beast.cli":
            raise ModuleNotFoundError("No module named 'httpx'", name="httpx")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(SystemExit) as excinfo:
        entrypoint.main()

    assert excinfo.value.code == 1
    assert "httpx" in capsys.readouterr().err


def test_main_runs_the_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []
    stub = types.ModuleType("beast.cli")
    stub.app = lambda: calls.append(True)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "beast.cli", stub)

    entrypoint.main()

    assert calls == [True]
