"""Conversation value object and message payloads."""

from __future__ import annotations

from beast.core.conversation import Conversation, Message, Role


def test_payload_matches_ollama_chat_shape() -> None:
    conversation = Conversation().with_system("be helpful").with_user("hi")

    assert conversation.payload() == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hi"},
    ]


def test_roles_serialise_to_their_api_names() -> None:
    assert str(Role.SYSTEM) == "system"
    assert str(Role.USER) == "user"
    assert str(Role.ASSISTANT) == "assistant"


def test_message_to_payload() -> None:
    assert Message(Role.ASSISTANT, "done").to_payload() == {
        "role": "assistant",
        "content": "done",
    }


def test_appending_returns_a_new_conversation() -> None:
    base = Conversation()

    grown = base.with_user("hello")

    assert len(base) == 0
    assert len(grown) == 1
    assert grown is not base


def test_iteration_follows_insertion_order() -> None:
    conversation = (
        Conversation()
        .with_system("sys")
        .with_user("first")
        .with_assistant("reply")
        .with_user("second")
    )

    assert [message.role for message in conversation] == [
        Role.SYSTEM,
        Role.USER,
        Role.ASSISTANT,
        Role.USER,
    ]
    assert [message.content for message in conversation][-1] == "second"


def test_messages_property_is_a_tuple_snapshot() -> None:
    conversation = Conversation().with_user("hi")

    snapshot = conversation.messages

    assert isinstance(snapshot, tuple)
    assert snapshot == (Message(Role.USER, "hi"),)


def test_tool_messages_serialise_with_their_name() -> None:
    conversation = Conversation().with_tool_result("read_file", "alpha")

    assert conversation.payload() == [
        {"role": "tool", "content": "alpha", "tool_name": "read_file"},
    ]


def test_assistant_tool_calls_are_preserved_in_the_payload() -> None:
    calls = [{"function": {"name": "read_file", "arguments": {"path": "a.txt"}}}]

    conversation = Conversation().with_assistant_tool_calls("thinking", calls)

    message = conversation.payload()[0]
    assert message["role"] == "assistant"
    assert message["content"] == "thinking"
    assert message["tool_calls"] == calls


def test_messages_without_tools_omit_the_tool_keys() -> None:
    conversation = Conversation().with_system("sys").with_assistant("hi")

    for message in conversation.payload():
        assert "tool_calls" not in message
        assert "tool_name" not in message


def test_without_last_drops_the_final_message() -> None:
    conversation = Conversation().with_user("one").with_user("two")

    assert conversation.without_last().payload() == [
        {"role": "user", "content": "one"}
    ]
