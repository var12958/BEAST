"""The agent layer: conversation state and the tool-calling loop.

BEAST runs a THINK → ACT → OBSERVE loop:

1. Send the conversation and the tool definitions to the selected local model.
2. If the model asks for tools, validate and execute each call.
3. Append every tool result to the conversation and go back to step 1.
4. When the model answers without tool calls, the task is done.

The model decides which tools to call and in what order.  Nothing here encodes
a task sequence; the loop only enforces safety limits, permission checks and
message consistency.

``stream()`` remains the plain-chat path used by chat-only models.  ``run()``
adds the loop and emits :mod:`beast.core.events` for the terminal interface.
Nothing in this module knows about Textual, Rich or the CLI, so the agent can
be driven headlessly and is exercised that way in the tests.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Protocol

from beast.errors import AgentIterationLimitError
from beast.intelligence.context_builder import ContextProvider
from beast.intelligence.models import ContextBundle
from beast.ollama.models import ChatChunk, ToolCall
from beast.tools.base import Permission, ToolResult
from beast.tools.registry import ToolRegistry

from .conversation import Conversation
from .events import (
    AgentEvent,
    AnalysisEvent,
    ApprovalPending,
    ContextReady,
    TextDelta,
    ToolFinished,
    ToolStarted,
)

logger = logging.getLogger(__name__)

#: Hard stop for a model that keeps calling tools instead of finishing.
MAX_ITERATIONS = 12

SYSTEM_PROMPT = (
    "You are BEAST, an AI coding agent running locally in the user's terminal.\n"
    "\n"
    "You help with software engineering: explaining code, designing solutions, "
    "writing and reviewing code, debugging errors and planning changes.\n"
    "\n"
    "How to answer:\n"
    "- Be direct and technically precise. Skip filler and praise.\n"
    "- Use Markdown. Put code in fenced blocks with the correct language tag.\n"
    "- When you propose a change, show the code and name the file path it belongs in.\n"
    "- If the request is ambiguous, ask one short, specific question instead of guessing.\n"
    "- You cannot read the user's files or run commands in this session. Do not claim "
    "otherwise; ask for the relevant file contents when you need them.\n"
)

TOOL_SYSTEM_PROMPT = (
    "You are BEAST, an AI coding agent running locally in the user's terminal.\n"
    "You work inside the user's project directory. Use the tools to inspect and "
    "change it; never guess what a file contains.\n"
    "\n"
    "Tools:\n"
    "- list_files, read_file, search_files: safe, they run immediately.\n"
    "- write_file, edit_file: they change files; the user approves each change.\n"
    "- run_command: runs a shell command; the user approves each command.\n"
    "\n"
    "The tool definitions in each request are authoritative: use the tools that "
    "are available there.\n"
    "\n"
    "A 'Project context' section may be attached automatically for the current "
    "request. It is gathered from files in the workspace: treat it as evidence, "
    "and read the relevant files directly when precision matters.\n"
    "\n"
    "How to work:\n"
    "- Explore before changing: list files, search, then read the relevant files.\n"
    "- Create files with write_file and complete contents.\n"
    "- Use edit_file for a small, exact change; include enough context to make the "
    "target text unique.\n"
    "- After changing files, verify them with read_file and fix any mistakes.\n"
    "- Make sure every file you reference exists, and that any code you show in "
    "your final answer matches the files on disk.\n"
    "- Keep using tools until the task is actually complete, then summarise what "
    "you changed.\n"
    "- If the user rejects an action, do not retry it; ask what they prefer.\n"
    "- Never claim a file change or a command result you did not get from a tool.\n"
    "- Use Markdown. Put code in fenced blocks with the correct language tag.\n"
)


def _context_message(bundle: ContextBundle) -> str:
    files = len(bundle.files)
    label = "file" if files == 1 else "files"
    message = (
        f"Project context: {files} {label}, {bundle.symbol_count} symbols, "
        f"{bundle.characters:,} characters."
    )
    if bundle.redactions:
        secrets = "secret" if bundle.redactions == 1 else "secrets"
        message += f" {bundle.redactions} {secrets} redacted."
    return message


class ChatStreamer(Protocol):
    """Minimal client surface the agent depends on (satisfied by ``OllamaClient``)."""

    def stream_chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        options: Mapping[str, Any] | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> AsyncIterator[ChatChunk]:
        """Stream chat chunks for ``model``."""
        ...


class Agent:
    """Runs one local model, its conversation and its tool loop."""

    def __init__(
        self,
        client: ChatStreamer,
        model: str,
        *,
        system_prompt: str | None = None,
        options: Mapping[str, Any] | None = None,
        tools: ToolRegistry | None = None,
        max_iterations: int = MAX_ITERATIONS,
        context_provider: ContextProvider | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._tools = tools
        self._max_iterations = max_iterations
        self._context_provider = context_provider
        self._turn_context: str | None = None
        if system_prompt is None:
            system_prompt = TOOL_SYSTEM_PROMPT if tools is not None else SYSTEM_PROMPT
        self._system_prompt = system_prompt
        self._options = dict(options) if options else None
        self._conversation = Conversation().with_system(system_prompt)

    @property
    def model(self) -> str:
        return self._model

    @property
    def conversation(self) -> Conversation:
        """The full message history, including tool calls and results."""
        return self._conversation

    @property
    def tools(self) -> ToolRegistry | None:
        return self._tools

    @property
    def context_provider(self) -> ContextProvider | None:
        return self._context_provider

    def reset(self) -> None:
        """Forget the conversation but keep the model and tools."""
        self._conversation = Conversation().with_system(self._system_prompt)
        self._turn_context = None

    # ---------------------------------------------------------------- chat-only

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream the assistant's reply to ``prompt`` as text deltas.

        The user message is recorded before the request; the assistant message
        is recorded when the stream ends — including when the user interrupts —
        so the next turn always sees what was actually said on screen.
        """
        if not prompt.strip():
            raise ValueError("Cannot send an empty prompt.")

        self._conversation = self._conversation.with_user(prompt)
        received: list[str] = []
        try:
            async for chunk in self._client.stream_chat(
                self._model,
                self._conversation.payload(),
                options=self._options,
            ):
                delta = chunk.content
                if not delta:
                    continue
                received.append(delta)
                yield delta
        finally:
            if received:
                self._conversation = self._conversation.with_assistant(
                    "".join(received)
                )

    # ------------------------------------------------------------- agent loop

    async def run(self, prompt: str) -> AsyncIterator[AgentEvent]:
        """Run the model/tool loop for ``prompt`` until it answers without tools.

        Raises :class:`AgentIterationLimitError` if the model never stops asking
        for tools, and propagates model errors.  Cancelling the consuming task
        stops the loop; every message already produced stays in the history.
        """
        if not prompt.strip():
            raise ValueError("Cannot send an empty prompt.")

        self._conversation = self._conversation.with_user(prompt)
        definitions = self._definitions()

        bundle = None
        if self._context_provider is not None:
            yield AnalysisEvent(
                stage="scan", message="Analysing project context…"
            )
            try:
                bundle = await asyncio.to_thread(
                    self._context_provider.context_for, prompt
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # analysis must never block the agent
                logger.warning("Project analysis failed: %s", exc)
                yield AnalysisEvent(
                    stage="error",
                    message=(
                        "Project analysis failed; continuing without automatic "
                        "context."
                    ),
                    detail=str(exc),
                )
            else:
                if bundle is not None and not bundle.is_empty:
                    self._turn_context = bundle.text
                    yield ContextReady(bundle=bundle)
                    yield AnalysisEvent(
                        stage="context",
                        message=_context_message(bundle),
                    )
                else:
                    yield AnalysisEvent(
                        stage="context",
                        message="No relevant project context was found.",
                    )

        try:
            for _ in range(self._max_iterations):
                content: list[str] = []
                calls: list[ToolCall] = []
                try:
                    async for chunk in self._request(definitions):
                        delta = chunk.content
                        if delta:
                            content.append(delta)
                            yield TextDelta(delta)
                        calls.extend(chunk.tool_calls)
                except BaseException:
                    # The turn was interrupted (user stop, consumer closed the
                    # stream, or a model error).  Record what the model produced
                    # and answer any requested calls so the history stays valid
                    # for the next turn.
                    self._record_interrupted(content, calls)
                    raise
                else:
                    if content or calls:
                        self._conversation = self._conversation.with_assistant_tool_calls(
                            "".join(content),
                            [call.to_payload() for call in calls],
                        )

                if not calls:
                    return

                try:
                    for call in calls:
                        async for event in self._handle_tool_call(call):
                            yield event
                finally:
                    # A consumer that stops between events (or closes the stream)
                    # must not leave a requested call without an observation.
                    self._answer_missing_calls(calls)

            raise AgentIterationLimitError(self._max_iterations)
        finally:
            # Context belongs to this request only; it is never persisted.
            self._turn_context = None

    def _answer_missing_calls(self, calls: Sequence[ToolCall]) -> None:
        """Answer any tool call from the current round that has no result yet."""
        answered = 0
        for message in reversed(self._conversation.messages):
            if message.role.value != "tool":
                break
            answered += 1
        for call in calls[answered:]:
            self._conversation = self._conversation.with_tool_result(
                call.name or "(unnamed)",
                "The task stopped before this tool reported a result; "
                "its outcome is unknown.",
            )

    def _record_interrupted(
        self, content: Sequence[str], calls: Sequence[ToolCall]
    ) -> None:
        """Keep the conversation consistent after an interrupted model turn."""
        if not content and not calls:
            return
        self._conversation = self._conversation.with_assistant_tool_calls(
            "".join(content),
            [call.to_payload() for call in calls],
        )
        for call in calls:
            self._conversation = self._conversation.with_tool_result(
                call.name or "(unnamed)",
                "The turn was interrupted before this tool ran; its outcome is unknown.",
            )

    def _definitions(self) -> list[dict[str, Any]] | None:
        if self._tools is None or self._tools.is_empty:
            return None
        return self._tools.definitions

    def _request(
        self, definitions: list[dict[str, Any]] | None
    ) -> AsyncIterator[ChatChunk]:
        messages = self._payload()
        if definitions:
            return self._client.stream_chat(
                self._model,
                messages,
                options=self._options,
                tools=definitions,
            )
        return self._client.stream_chat(
            self._model,
            messages,
            options=self._options,
        )

    def _payload(self) -> list[dict[str, Any]]:
        """Messages for one request, with transient project context attached.

        The context is merged into the leading system message for this request
        only; :meth:`run` clears it afterwards.  The stored conversation never
        contains it, so a long chat does not accumulate stale project snapshots.
        """
        messages = self._conversation.payload()
        if not self._turn_context:
            return messages
        if messages and messages[0].get("role") == "system":
            first = dict(messages[0])
            first["content"] = f"{first.get('content', '')}\n\n{self._turn_context}"
            messages = [first, *messages[1:]]
        else:  # pragma: no cover - every conversation starts with a system message
            messages = [
                {"role": "system", "content": self._turn_context},
                *messages,
            ]
        return messages

    async def _handle_tool_call(self, call: ToolCall) -> AsyncIterator[AgentEvent]:
        name = call.name or "(unnamed)"
        arguments = call.arguments
        tool = self._tools.get(name) if self._tools is not None else None

        def record(content: str) -> None:
            """Append the observation before the UI is notified."""
            self._conversation = self._conversation.with_tool_result(name, content)

        if tool is None:
            available = ", ".join(self._tools.names) if self._tools else "none"
            result = ToolResult.failure(
                f"Unknown tool '{name}'.",
                content=(
                    f"There is no tool named {name!r}. Available tools: "
                    f"{available or 'none'}. Call one of those instead."
                ),
            )
            yield ToolStarted(
                name=name, arguments=arguments, permission=Permission.READ
            )
            record(result.for_model())
            yield ToolFinished(name=name, result=result)
            return

        if not isinstance(arguments, dict):
            result = ToolResult.failure(
                f"Malformed arguments for {name}.",
                content=(
                    f"The arguments for {name} were not a JSON object, so nothing "
                    "ran. Call the tool again with an object that matches its schema."
                ),
            )
            yield ToolStarted(
                name=name, arguments=arguments, permission=tool.permission
            )
            record(result.for_model())
            yield ToolFinished(name=name, result=result)
            return

        yield ToolStarted(
            name=name, arguments=arguments, permission=tool.permission
        )
        if tool.permission is not Permission.READ:
            yield ApprovalPending(name=name)

        try:
            result = await tool.run(arguments)
        except asyncio.CancelledError:
            # Keep the conversation valid: the assistant requested this
            # call, so it must have an answer even when the user stopped.
            record("The user stopped the task before this tool finished.")
            raise

        record(result.for_model())
        yield ToolFinished(name=name, result=result)
