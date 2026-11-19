"""Slash-command parsing: pure behaviour without a terminal."""

from __future__ import annotations

import pytest

from beast.ui.commands import COMMAND_NAMES, HELP_TEXT, parse_command


@pytest.mark.parametrize(
    "text",
    [
        "hello",
        "please run /help for me",
        "",
        "   ",
        "/",
        "/   ",
    ],
)
def test_ordinary_text_is_not_a_command(text: str) -> None:
    assert parse_command(text) is None


def test_a_bare_command_has_an_empty_argument() -> None:
    command = parse_command("/help")

    assert command is not None
    assert command.name == "help"
    assert command.argument == ""
    assert command.raw == "/help"
    assert command.is_known


def test_surrounding_whitespace_is_ignored() -> None:
    command = parse_command("   /diff app.py   ")

    assert command is not None
    assert command.name == "diff"
    assert command.argument == "app.py"
    assert command.raw == "/diff app.py"


def test_names_are_case_insensitive_but_arguments_are_not() -> None:
    command = parse_command("/AUTO Build The Page")

    assert command is not None
    assert command.name == "auto"
    assert command.argument == "Build The Page"


def test_argument_text_keeps_inner_spacing() -> None:
    command = parse_command("/auto  add   a dark mode toggle")

    assert command is not None
    assert command.argument == "add   a dark mode toggle"


def test_unknown_commands_are_reported_as_unknown() -> None:
    command = parse_command("/bogus now")

    assert command is not None
    assert command.name == "bogus"
    assert not command.is_known


def test_every_documented_command_parses() -> None:
    for name in COMMAND_NAMES:
        command = parse_command(f"/{name} argument")
        assert command is not None
        assert command.name == name
        assert command.is_known
        assert f"/{name}" in HELP_TEXT
