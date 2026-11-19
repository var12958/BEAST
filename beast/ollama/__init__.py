"""Everything BEAST needs to talk to a locally installed Ollama."""

from __future__ import annotations

from .client import OllamaClient
from .connection import (
    BASE_URL_ENV,
    DEFAULT_BASE_URL,
    OllamaStatus,
    default_base_url,
    ensure_ready,
    find_executable,
    probe,
)
from .models import (
    ChatChunk,
    ChatMessage,
    ModelDetails,
    ModelInventory,
    OllamaModel,
    OllamaModelList,
    split_models,
)

__all__ = [
    "BASE_URL_ENV",
    "DEFAULT_BASE_URL",
    "ChatChunk",
    "ChatMessage",
    "ModelDetails",
    "ModelInventory",
    "OllamaClient",
    "OllamaModel",
    "OllamaModelList",
    "OllamaStatus",
    "default_base_url",
    "ensure_ready",
    "find_executable",
    "probe",
    "split_models",
]
