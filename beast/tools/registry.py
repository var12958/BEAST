"""The tool registry.

The registry is the only place that knows which tools exist.  The agent asks it
for Ollama tool definitions and for the tool that should handle a model's call,
so adding a tool never requires touching the execution loop.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from beast.tools.base import Tool, ToolContext
from beast.tools.filesystem import (
    EditFileTool,
    ListFilesTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)
from beast.tools.project import ProjectInfoTool
from beast.tools.shell import RunCommandTool

#: The tools enabled in this release, in the order shown to the model.
BUILTIN_TOOLS: tuple[type[Tool], ...] = (
    ListFilesTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
    EditFileTool,
    RunCommandTool,
)


class ToolRegistry:
    """A named collection of tools available to one agent session."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Add ``tool``; names must be unique."""
        if tool.name in self._tools:
            raise ValueError(f"A tool named '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Return the tool called ``name``, or ``None`` when unknown."""
        return self._tools.get(name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    @property
    def is_empty(self) -> bool:
        return not self._tools

    @property
    def definitions(self) -> list[dict[str, Any]]:
        """The ``tools`` payload for ``POST /api/chat``."""
        return [tool.definition() for tool in self._tools.values()]

    @classmethod
    def default(cls, context: ToolContext) -> ToolRegistry:
        """Build the standard toolset for one session.

        ``project_info`` is present only when project intelligence was supplied,
        so sessions without it keep exactly the original six-tool contract.
        """
        tools = [tool_type(context) for tool_type in BUILTIN_TOOLS]
        if context.intelligence is not None:
            tools.append(ProjectInfoTool(context))
        return cls(tools)
