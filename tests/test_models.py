"""Model discovery and classification.

The parser is exercised against a real ``/api/tags`` payload because Ollama's
own responses contain quirks (``families: null`` on cloud entries) that a
hand-written fixture would never reproduce.
"""

from __future__ import annotations

import pytest

from beast.ollama.models import (
    ChatChunk,
    OllamaModel,
    OllamaModelList,
    ToolCall,
    split_models,
)


def test_parses_every_model_in_a_real_payload(models: list[OllamaModel]) -> None:
    assert len(models) == 5
    assert {model.name for model in models} == {
        "deepseek-v4.1-flash:cloud",
        "nomic-embed-text:latest",
        "qwen3:8b",
        "deepseek-v4-pro:cloud",
        "kimi-k2.7-code:cloud",
    }


def test_reads_metadata_from_a_local_model(models: list[OllamaModel]) -> None:
    qwen = next(model for model in models if model.name == "qwen3:8b")

    assert qwen.details.family == "qwen3"
    assert qwen.parameter_display == "8.2B"
    assert qwen.quantization_display == "Q4_K_M"
    assert qwen.size_display == "5.2 GB"
    assert qwen.summary() == "qwen3:8b · 5.2 GB · Q4_K_M · qwen3"


def test_cloud_models_are_recognised_by_metadata(models: list[OllamaModel]) -> None:
    cloud = [model for model in models if model.is_cloud]

    assert len(cloud) == 3
    assert all(model.remote_host == "https://ollama.com" for model in cloud)
    assert all(model.name.endswith(":cloud") for model in cloud)


def test_cloud_models_report_no_disk_size(models: list[OllamaModel]) -> None:
    cloud = next(model for model in models if model.is_cloud)

    assert cloud.size == 326  # bytes of metadata, not weights
    assert cloud.size_display == "—"


def test_embedding_model_is_not_chat_capable(models: list[OllamaModel]) -> None:
    embed = next(model for model in models if model.name.startswith("nomic-"))

    assert embed.capabilities == ["embedding"]
    assert embed.is_chat_capable is False
    assert embed.is_cloud is False


def test_chat_models_expose_completion(models: list[OllamaModel]) -> None:
    qwen = next(model for model in models if model.name == "qwen3:8b")

    assert qwen.is_chat_capable is True
    assert "completion" in qwen.capabilities


def test_null_families_list_is_tolerated(models: list[OllamaModel]) -> None:
    cloud = next(model for model in models if model.name == "deepseek-v4-pro:cloud")

    assert cloud.details.families == []
    assert cloud.family_display == "—"


def test_null_details_and_capabilities_are_tolerated() -> None:
    model = OllamaModel.model_validate(
        {"name": "mystery:1b", "details": None, "capabilities": None}
    )

    assert model.details.family == ""
    assert model.capabilities == []
    # No capability list means an older Ollama build: give the model a chance.
    assert model.is_chat_capable is True


def test_missing_fields_fall_back_to_defaults() -> None:
    model = OllamaModel(name="bare:latest")

    assert model.size_display == "—"
    assert model.quantization_display == "—"
    assert model.family_display == "—"
    assert model.parameter_display == "—"


def test_family_display_uses_families_list_when_family_is_blank() -> None:
    model = OllamaModel(name="m", details={"families": ["llama"]})

    assert model.family_display == "llama"


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (5_232_000_000, "5.2 GB"),
        (2_100_000_000, "2.1 GB"),
        (274_302_450, "274 MB"),
        (0, "—"),
        (999, "—"),
    ],
)
def test_size_display_formats_bytes(size: int, expected: str) -> None:
    assert OllamaModel(name="m", size=size).size_display == expected


@pytest.mark.parametrize(
    "model",
    [
        OllamaModel(name="anything:cloud"),
        OllamaModel(name="anything", remote_model="x"),
        OllamaModel(name="anything", remote_host="https://ollama.com"),
    ],
)
def test_cloud_detection_variants(model: OllamaModel) -> None:
    assert model.is_cloud is True


def test_local_tag_named_cloudy_is_not_cloud() -> None:
    assert OllamaModel(name="cloudy:latest").is_cloud is False


def test_split_models_separates_groups(models: list[OllamaModel]) -> None:
    inventory = split_models(models)

    assert [model.name for model in inventory.selectable] == ["qwen3:8b"]
    assert [model.name for model in inventory.cloud] == [
        "deepseek-v4-pro:cloud",
        "deepseek-v4.1-flash:cloud",
        "kimi-k2.7-code:cloud",
    ]
    assert len(inventory.non_chat) == 1
    assert inventory.total == 5
    # Only embedding models are unusable now; cloud models are selectable.
    assert inventory.hidden == 1
    assert inventory.names == (
        "qwen3:8b",
        "deepseek-v4-pro:cloud",
        "deepseek-v4.1-flash:cloud",
        "kimi-k2.7-code:cloud",
    )


def test_usable_inventory_offers_local_then_cloud(models: list[OllamaModel]) -> None:
    inventory = split_models(models)

    assert [model.name for model in inventory.usable] == list(inventory.names)
    assert inventory.usable[0].is_cloud is False
    assert all(model.is_cloud for model in inventory.usable[1:])


def test_hidden_note_explains_filtering(models: list[OllamaModel]) -> None:
    note = split_models(models).hidden_note

    assert note == "Hidden: 1 embedding-only model."


def test_hidden_note_is_absent_when_nothing_is_hidden() -> None:
    inventory = split_models(
        [
            OllamaModel(name="qwen3:8b", capabilities=["completion"]),
            OllamaModel(name="kimi:cloud", capabilities=["completion", "tools"]),
        ]
    )

    assert inventory.hidden_note is None
    assert len(inventory.selectable) == 1
    assert len(inventory.cloud) == 1


def test_split_models_sorts_selectable_models() -> None:
    inventory = split_models(
        [OllamaModel(name="b:latest"), OllamaModel(name="A:latest")]
    )

    assert [model.name for model in inventory.selectable] == ["A:latest", "b:latest"]


def test_split_models_empty_store() -> None:
    inventory = split_models([])

    assert inventory.total == 0
    assert inventory.hidden_note is None
    assert inventory.names == ()


def test_inventory_lists_default_to_empty(tags_payload: dict) -> None:
    parsed = OllamaModelList.model_validate(tags_payload)

    assert len(parsed.models) == 5
    assert OllamaModelList.model_validate({}).models == []


# ---------------------------------------------------------------- tool support


def test_supports_tools_reads_the_capability_list() -> None:
    capable = OllamaModel(name="m", capabilities=["completion", "tools"])
    plain = OllamaModel(name="m", capabilities=["completion"])
    unknown = OllamaModel(name="m")

    assert capable.supports_tools is True
    assert plain.supports_tools is False
    assert unknown.supports_tools is None
    assert capable.tools_display == "yes"
    assert plain.tools_display == "no"
    assert unknown.tools_display == "?"


def test_tool_call_parses_an_object_arguments_payload() -> None:
    call = ToolCall.model_validate(
        {"function": {"name": "read_file", "arguments": {"path": "a.txt"}}}
    )

    assert call.name == "read_file"
    assert call.arguments == {"path": "a.txt"}
    assert call.to_payload() == {
        "function": {"name": "read_file", "arguments": {"path": "a.txt"}}
    }


def test_tool_call_parses_a_json_string_arguments_payload() -> None:
    call = ToolCall.model_validate(
        {"function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}}
    )

    assert call.arguments == {"path": "a.txt"}


def test_tool_call_keeps_unparsable_strings_for_the_agent_to_reject() -> None:
    call = ToolCall.model_validate(
        {"function": {"name": "read_file", "arguments": "{oops"}}
    )

    assert call.arguments == "{oops"
    # The echoed history must remain a valid request for Ollama.
    assert call.to_payload() == {
        "function": {"name": "read_file", "arguments": {}}
    }


def test_tool_call_tolerates_null_arguments() -> None:
    call = ToolCall.model_validate(
        {"function": {"name": "list_files", "arguments": None}}
    )

    assert call.arguments == {}


def test_chat_chunk_exposes_tool_calls() -> None:
    chunk = ChatChunk.model_validate(
        {
            "model": "m",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "list_files", "arguments": {}}}
                ],
            },
        }
    )

    assert [call.name for call in chunk.tool_calls] == ["list_files"]
    assert ChatChunk().tool_calls == ()
