"""Pydantic models for the parts of the Ollama HTTP API that BEAST uses.

Only the fields BEAST actually relies on are declared; everything else that
Ollama sends is ignored so that a future Ollama release adding fields cannot
break the client.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

_BYTES_PER_MB = 1_000_000
_BYTES_PER_GB = 1_000_000_000


class ModelDetails(BaseModel):
    """The ``details`` object returned by ``GET /api/tags``."""

    model_config = ConfigDict(extra="ignore")

    family: str = ""
    families: list[str] = Field(default_factory=list)
    parameter_size: str = ""
    quantization_level: str = ""
    format: str = ""

    @field_validator("families", mode="before")
    @classmethod
    def _null_families(cls, value: object) -> object:
        # Ollama sends ``"families": null`` for some models.
        return [] if value is None else value


class OllamaModel(BaseModel):
    """A single model entry returned by ``GET /api/tags``."""

    model_config = ConfigDict(extra="ignore")

    name: str
    model: str = ""
    size: int = 0
    digest: str = ""
    modified_at: str = ""
    details: ModelDetails = Field(default_factory=ModelDetails)
    capabilities: list[str] = Field(default_factory=list)

    # Ollama's cloud models (``name:cloud``) are proxied through ollama.com and
    # are reported with these fields.  BEAST never routes inference off-machine.
    remote_host: str | None = None
    remote_model: str | None = None

    @field_validator("details", "capabilities", mode="before")
    @classmethod
    def _null_to_default(cls, value: object, info: ValidationInfo) -> object:
        # Some Ollama builds emit ``"details": null`` / ``"capabilities": null``.
        if value is not None:
            return value
        return [] if info.field_name == "capabilities" else {}

    @property
    def identifier(self) -> str:
        """The identifier to pass back to Ollama (``POST /api/chat``)."""
        return self.name

    @property
    def is_cloud(self) -> bool:
        """True when the model is served by Ollama's cloud instead of locally."""
        if self.remote_host or self.remote_model:
            return True
        _, _, tag = self.name.lower().rpartition(":")
        return tag == "cloud"

    @property
    def is_chat_capable(self) -> bool:
        """True when the model can answer chat requests.

        Older Ollama versions omit ``capabilities`` entirely; in that case the
        model is assumed to be usable and the API decides.
        """
        if not self.capabilities:
            return True
        return "completion" in self.capabilities

    @property
    def supports_tools(self) -> bool | None:
        """Whether the model advertises native tool calling.

        ``None`` means this Ollama build does not report capabilities, so BEAST
        cannot know in advance.  A model's name says nothing about support;
        only this metadata does.
        """
        if not self.capabilities:
            return None
        return "tools" in self.capabilities

    @property
    def tools_display(self) -> str:
        """Short label for the CLI table."""
        support = self.supports_tools
        if support is None:
            return "?"
        return "yes" if support else "no"

    @property
    def location_display(self) -> str:
        """Whether inference runs on this machine or through Ollama Cloud."""
        return "cloud" if self.is_cloud else "local"

    @property
    def size_display(self) -> str:
        # Cloud entries report a tiny metadata size; it means nothing on disk.
        if self.is_cloud or not self.size or self.size < _BYTES_PER_MB:
            return "—"
        if self.size >= _BYTES_PER_GB:
            return f"{self.size / _BYTES_PER_GB:.1f} GB"
        return f"{self.size / _BYTES_PER_MB:.0f} MB"

    @property
    def quantization_display(self) -> str:
        return self.details.quantization_level or "—"

    @property
    def family_display(self) -> str:
        if self.details.family:
            return self.details.family
        if self.details.families:
            return self.details.families[0]
        return "—"

    @property
    def parameter_display(self) -> str:
        return self.details.parameter_size or "—"

    def summary(self) -> str:
        """One-line description used in the model picker and CLI table."""
        return (
            f"{self.name} · {self.size_display} · "
            f"{self.quantization_display} · {self.family_display}"
        )


class OllamaModelList(BaseModel):
    """Response of ``GET /api/tags``."""

    model_config = ConfigDict(extra="ignore")

    models: list[OllamaModel] = Field(default_factory=list)


class ToolFunctionCall(BaseModel):
    """The ``function`` object inside an Ollama tool call."""

    model_config = ConfigDict(extra="ignore")

    name: str = ""
    # Ollama normally sends an object here, but models sometimes emit a JSON
    # string or another shape.  Keep whatever arrived so the agent can answer
    # with a corrective observation instead of failing the whole turn.
    arguments: Any = Field(default_factory=dict)

    @field_validator("arguments", mode="before")
    @classmethod
    def _parse_json_string(cls, value: object) -> object:
        if not isinstance(value, str):
            return {} if value is None else value
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return parsed if isinstance(parsed, dict) else value

    def to_payload(self) -> dict[str, Any]:
        # Echo a syntactically valid object back to Ollama even when the model
        # sent something else; the agent reports the malformed call separately.
        arguments = self.arguments if isinstance(self.arguments, dict) else {}
        return {"function": {"name": self.name, "arguments": arguments}}


class ToolCall(BaseModel):
    """A tool call requested by the model."""

    model_config = ConfigDict(extra="ignore")

    # Present in OpenAI-compatible responses, absent from Ollama's native ones.
    id: str | None = None
    type: str | None = None
    function: ToolFunctionCall = Field(default_factory=ToolFunctionCall)

    @property
    def name(self) -> str:
        return self.function.name

    @property
    def arguments(self) -> Any:
        return self.function.arguments

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = self.function.to_payload()
        if self.id is not None:
            payload["id"] = self.id
        if self.type is not None:
            payload["type"] = self.type
        return payload


class ChatMessage(BaseModel):
    """A single message inside a chat request or streaming response."""

    model_config = ConfigDict(extra="ignore")

    role: str = "assistant"
    content: str = ""
    # Reasoning-capable models may stream thoughts separately; BEAST keeps them
    # out of the conversation transcript.
    thinking: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)

    @field_validator("tool_calls", mode="before")
    @classmethod
    def _null_tool_calls(cls, value: object) -> object:
        return [] if value is None else value


class ChatChunk(BaseModel):
    """One newline-delimited JSON object from ``POST /api/chat``."""

    model_config = ConfigDict(extra="ignore")

    model: str = ""
    created_at: str = ""
    message: ChatMessage = Field(default_factory=ChatMessage)
    done: bool = False
    done_reason: str | None = None
    error: str | None = None

    @property
    def content(self) -> str:
        return self.message.content

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(self.message.tool_calls)


@dataclass(frozen=True, slots=True)
class ModelInventory:
    """The installed models, split into what BEAST can and cannot use.

    ``selectable`` holds locally installed chat models, ``cloud`` holds chat
    models served by Ollama Cloud.  Both are usable by BEAST — only
    ``non_chat`` entries (embeddings and similar) are excluded.
    """

    selectable: tuple[OllamaModel, ...] = ()
    cloud: tuple[OllamaModel, ...] = ()
    non_chat: tuple[OllamaModel, ...] = ()

    @property
    def total(self) -> int:
        return len(self.selectable) + len(self.cloud) + len(self.non_chat)

    @property
    def hidden(self) -> int:
        return len(self.non_chat)

    @property
    def usable(self) -> tuple[OllamaModel, ...]:
        """Every model the picker offers: local first, then cloud."""
        return (*self.selectable, *self.cloud)

    @property
    def hidden_note(self) -> str | None:
        """Explain what was filtered out, without naming any specific model."""
        if not self.non_chat:
            return None
        count = len(self.non_chat)
        noun = "model" if count == 1 else "models"
        return f"Hidden: {count} embedding-only {noun}."

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(model.identifier for model in self.usable)


def split_models(models: Iterable[OllamaModel]) -> ModelInventory:
    """Split Ollama models into local, cloud and non-chat groups.

    Selection is capability based, never name based, so any chat model Ollama
    reports — local or cloud — shows up automatically.
    """
    selectable: list[OllamaModel] = []
    cloud: list[OllamaModel] = []
    non_chat: list[OllamaModel] = []
    for model in models:
        if not model.is_chat_capable:
            non_chat.append(model)
        elif model.is_cloud:
            cloud.append(model)
        else:
            selectable.append(model)
    selectable.sort(key=lambda model: model.name.lower())
    cloud.sort(key=lambda model: model.name.lower())
    return ModelInventory(
        selectable=tuple(selectable),
        cloud=tuple(cloud),
        non_chat=tuple(non_chat),
    )
