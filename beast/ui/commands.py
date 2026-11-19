"""Slash commands for the terminal interface.

Parsing is deliberately small and pure so it can be unit-tested without
Textual.  A prompt that does not start with ``/`` is ordinary chat text.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Commands the chat screen understands.
COMMAND_NAMES = (
    "help",
    "auto",
    "changes",
    "diff",
    "rollback",
    "git",
    "cancel",
)

HELP_TEXT = """\
Slash commands
  /auto <request>      Plan and execute a coding request autonomously
  /changes             Show files BEAST created, modified or deleted
  /diff [path]         Show a unified diff for BEAST's changes
  /rollback [path…]    Restore BEAST's changes (asks for approval)
  /git                 Show read-only git status of the workspace
  /cancel              Stop the running autonomous task
  /help                Show this help

Everything else is sent to the model as a normal prompt.
Writes and commands still require explicit approval.  BEAST never commits,
pushes or installs packages on its own."""


@dataclass(frozen=True, slots=True)
class SlashCommand:
    """A parsed command line."""

    name: str
    argument: str = ""
    raw: str = ""

    @property
    def is_known(self) -> bool:
        return self.name in COMMAND_NAMES


def parse_command(text: str) -> SlashCommand | None:
    """Return a command when ``text`` starts with ``/``, otherwise ``None``.

    A lone ``/`` or a slash inside a sentence is ordinary text.
    """
    stripped = text.strip()
    if not stripped.startswith("/") or len(stripped) < 2:
        return None
    body = stripped[1:]
    name, _, argument = body.partition(" ")
    name = name.strip().lower()
    if not name:
        return None
    return SlashCommand(
        name=name,
        argument=argument.strip(),
        raw=stripped,
    )
