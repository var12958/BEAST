"""The contract every BEAST tool implements.

A tool declares a name, a description, a permission level and a pydantic
argument schema.  The registry turns those declarations into Ollama's
``tools`` payload; the agent validates a model's arguments against the same
model before anything touches the disk.

Tools never raise for expected problems.  They return :class:`ToolResult`,
which the agent feeds back to the model as an observation — a failed tool is
information the model can react to, not a crash.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Protocol

from pydantic import BaseModel, ValidationError

from beast.errors import BeastError
from beast.workspace import Workspace

from .schemas import tool_schema

if TYPE_CHECKING:
    from beast.core.changes import ChangeTracker
    from beast.intelligence import ProjectIntelligence

logger = logging.getLogger(__name__)


class Permission(StrEnum):
    """How dangerous a tool is, which decides whether approval is required."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """What the user is asked to approve, shown verbatim in the interface."""

    tool: str
    permission: Permission
    target: str
    action: str
    detail: str = ""


class Approver(Protocol):
    """Anything that can answer yes/no for a tool action."""

    async def approve(self, request: ApprovalRequest) -> bool:
        """Return True only when the user explicitly approved the action."""
        ...


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The structured outcome of a tool call."""

    ok: bool
    summary: str
    #: Text handed back to the model as the observation.
    content: str = ""
    #: True when the user declined the action (as opposed to a failure).
    rejected: bool = False
    #: Process exit status for command tools (``None`` when nothing ran).
    exit_code: int | None = None
    #: True when a command hit its timeout and was terminated.
    timed_out: bool = False

    @classmethod
    def success(
        cls,
        summary: str,
        *,
        content: str = "",
        exit_code: int | None = None,
    ) -> ToolResult:
        return cls(
            ok=True,
            summary=summary,
            content=content,
            exit_code=exit_code,
        )

    @classmethod
    def failure(
        cls,
        summary: str,
        *,
        content: str = "",
        rejected: bool = False,
        exit_code: int | None = None,
        timed_out: bool = False,
    ) -> ToolResult:
        return cls(
            ok=False,
            summary=summary,
            content=content,
            rejected=rejected,
            exit_code=exit_code,
            timed_out=timed_out,
        )

    def for_model(self) -> str:
        """The observation text returned to the model."""
        return self.content or self.summary


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Shared services and limits handed to every tool instance."""

    workspace: Workspace
    approver: Approver
    #: Hard ceiling for a single read; a tool argument may ask for less.
    max_read_bytes: int = 512_000
    #: Ceiling for command output and other large observations.
    max_output_chars: int = 30_000
    #: Hard ceiling for a single command, whatever the model asks for.
    command_timeout: float = 120.0
    #: Project analysis, when this session has it.  ``None`` keeps the original
    #: six-tool set for chat-only or intelligence-disabled sessions.
    intelligence: "ProjectIntelligence | None" = None
    #: Change tracking, when the session wants diffs/rollback.  Optional so
    #: chat-only sessions keep working exactly as before.
    changes: "ChangeTracker | None" = None

    def analysis_invalidated(self) -> None:
        """Tell the project index that a file changed on disk."""
        if self.intelligence is not None:
            self.intelligence.invalidate()

    def capture_file(self, path: Path) -> None:
        """Snapshot a file before BEAST modifies it (for rollback/diffs)."""
        if self.changes is not None:
            self.changes.capture(path)

    def finalize_file(self, path: Path) -> None:
        """Record the result of a BEAST write, then invalidate the index."""
        if self.changes is not None:
            self.changes.finalize(path)
        self.analysis_invalidated()

    def record_delete(self, path: Path) -> None:
        if self.changes is not None:
            self.changes.record_delete(path)
        self.analysis_invalidated()

    def record_approval(
        self,
        *,
        tool: str,
        target: str,
        action: str,
        approved: bool,
    ) -> None:
        if self.changes is not None:
            self.changes.record_approval(
                tool=tool,
                target=target,
                action=action,
                approved=approved,
            )

    def record_command(
        self,
        command: str,
        *,
        exit_code: int | None,
        ok: bool,
        rejected: bool = False,
        timed_out: bool = False,
        kind: str = "agent",
    ) -> None:
        if self.changes is not None:
            self.changes.record_command(
                command,
                exit_code=exit_code,
                ok=ok,
                rejected=rejected,
                timed_out=timed_out,
                kind=kind,
            )


class Tool:
    """Base class for a BEAST tool."""

    name: ClassVar[str]
    description: ClassVar[str]
    permission: ClassVar[Permission] = Permission.READ
    args_model: ClassVar[type[BaseModel]]

    def __init__(self, context: ToolContext) -> None:
        self.context = context

    @classmethod
    def definition(cls) -> dict[str, Any]:
        """The JSON definition sent to Ollama in the ``tools`` field."""
        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": tool_schema(cls.args_model),
            },
        }

    async def run(self, arguments: Mapping[str, Any]) -> ToolResult:
        """Validate ``arguments`` and execute the tool, never raising."""
        try:
            args = self.args_model.model_validate(dict(arguments))
        except ValidationError as exc:
            logger.info("invalid arguments for %s: %s", self.name, exc)
            return ToolResult.failure(
                f"Invalid arguments for {self.name}: {_describe(exc)}",
                content=(
                    f"The arguments for {self.name} were invalid: {_describe(exc)}. "
                    "Call the tool again with arguments that match its schema."
                ),
            )
        except (TypeError, ValueError) as exc:
            return ToolResult.failure(
                f"Invalid arguments for {self.name}: {exc}",
                content=f"The arguments for {self.name} were not an object.",
            )

        try:
            return await self.execute(args)
        except BeastError as exc:
            logger.info("%s refused: %s", self.name, exc.message)
            return ToolResult.failure(
                f"{self.name} failed: {exc.message}",
                content=f"{self.name} failed: {exc.message}",
            )
        except OSError as exc:
            logger.info("%s hit an operating-system error: %s", self.name, exc)
            return ToolResult.failure(
                f"{self.name} could not access the file system: {exc.strerror or exc}",
                content=(
                    f"{self.name} failed with an operating-system error: "
                    f"{exc.strerror or exc}."
                ),
            )
        except Exception:  # noqa: BLE001 - a tool must never kill the agent
            logger.exception("unexpected failure in tool %s", self.name)
            return ToolResult.failure(
                f"{self.name} hit an unexpected error.",
                content=f"{self.name} hit an unexpected error and did not complete.",
            )

    async def execute(self, args: BaseModel) -> ToolResult:
        """Perform the action. Subclasses implement this."""
        raise NotImplementedError

    async def request_approval(
        self,
        *,
        target: str,
        action: str,
        detail: str = "",
    ) -> bool:
        """Ask the user, treating anything but an explicit True as refusal."""
        request = ApprovalRequest(
            tool=self.name,
            permission=self.permission,
            target=target,
            action=action,
            detail=detail,
        )
        try:
            approved = await self.context.approver.approve(request)
        except Exception:  # noqa: BLE001 - a broken approver is a refusal
            logger.exception("the approval UI failed for %s", self.name)
            approved = False
        decision = approved is True
        self.context.record_approval(
            tool=self.name,
            target=target,
            action=action,
            approved=decision,
        )
        return decision

    def relative(self, path: Any) -> str:
        """Format ``path`` relative to the workspace for display."""
        try:
            relative = path.relative_to(self.context.workspace.root)
        except ValueError:  # pragma: no cover - resolve() prevents this
            return str(path)
        return relative.as_posix() or "."


def _describe(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors()[:3]:
        location = ".".join(str(part) for part in error["loc"]) or "arguments"
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts)
