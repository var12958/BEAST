"""Conversation state.

Messages are immutable and a :class:`Conversation` is a value: appending
returns a new conversation instead of mutating an existing one.  That keeps the
history safe to share with the UI and makes the agent's state changes explicit.

The message shapes mirror the Ollama chat API: an assistant message may carry
``tool_calls``, and every tool call is answered by a ``tool`` message that names
the tool and carries its result.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    """Roles understood by Ollama's chat endpoint."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class Message:
    """A single chat message."""

    role: Role
    content: str = ""
    #: Assistant messages only: the tool calls the model requested.
    tool_calls: tuple[dict[str, Any], ...] = ()
    #: Tool messages only: the name of the tool that produced the result.
    tool_name: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """Convert to the shape ``POST /api/chat`` expects."""
        payload: dict[str, Any] = {
            "role": str(self.role),
            "content": self.content,
        }
        if self.tool_calls:
            payload["tool_calls"] = [dict(call) for call in self.tool_calls]
        if self.tool_name:
            payload["tool_name"] = self.tool_name
        return payload


class Conversation:
    """An immutable, ordered list of messages."""

    __slots__ = ("_messages",)

    def __init__(self, messages: Iterable[Message] = ()) -> None:
        self._messages: tuple[Message, ...] = tuple(messages)

    @property
    def messages(self) -> tuple[Message, ...]:
        return self._messages

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self) -> Iterator[Message]:
        return iter(self._messages)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Conversation(messages={len(self._messages)})"

    def append(self, message: Message) -> Conversation:
        """Return a new conversation with ``message`` appended."""
        return Conversation((*self._messages, message))

    def without_last(self) -> Conversation:
        """Return a conversation without its final message."""
        return Conversation(self._messages[:-1])

    def with_system(self, content: str) -> Conversation:
        return self.append(Message(Role.SYSTEM, content))

    def with_user(self, content: str) -> Conversation:
        return self.append(Message(Role.USER, content))

    def with_assistant(self, content: str) -> Conversation:
        return self.append(Message(Role.ASSISTANT, content))

    def with_assistant_tool_calls(
        self,
        content: str,
        tool_calls: Sequence[Mapping[str, Any]],
    ) -> Conversation:
        """Append an assistant turn that requested tools."""
        return self.append(
            Message(
                Role.ASSISTANT,
                content,
                tuple(dict(call) for call in tool_calls),
            )
        )

    def with_tool_result(self, tool_name: str, content: str) -> Conversation:
        """Append the observation returned to the model."""
        return self.append(Message(Role.TOOL, content, tool_name=tool_name))

    def payload(self) -> list[dict[str, Any]]:
        """The message list to send to Ollama."""
        return [message.to_payload() for message in self._messages]
