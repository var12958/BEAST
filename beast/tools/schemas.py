"""Argument schemas shared by the built-in tools.

Each tool declares a pydantic model describing its arguments.  The same model
produces the JSON Schema sent to Ollama and validates the arguments that come
back, so the advertised contract and the enforced contract cannot drift apart.

Pydantic emits a few JSON Schema keywords that are noise for small local
models (titles, ``$defs`` indirection).  :func:`tool_schema` keeps the shape
compact and flat.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ToolArgs(BaseModel):
    """Base for every tool's arguments: reject keys the model invented."""

    model_config = ConfigDict(extra="forbid")


def _strip_titles(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_titles(item)
            for key, item in value.items()
            if key != "title"
        }
    if isinstance(value, list):
        return [_strip_titles(item) for item in value]
    return value


def tool_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a compact JSON Schema for one tool's arguments."""
    schema = _strip_titles(model.model_json_schema())
    schema.setdefault("type", "object")
    return schema
