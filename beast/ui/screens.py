"""Full-screen screens: model selection and the chat workspace."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.geometry import Offset
from textual.screen import ModalScreen, Screen
from textual.widget import Widget
from textual.widgets import Button, OptionList, Static
from textual.widgets.option_list import Option
from textual.worker import Worker

from beast.core.agent import Agent
from beast.core.autonomy import AutonomousRunner
from beast.core.changes import ChangeTracker, GitInspector
from beast.core.events import (
    AnalysisEvent,
    ApprovalPending,
    ChangeSummary,
    DebugAttempt,
    PlanReady,
    PlanRevised,
    PlanStarted,
    RunFinished,
    TaskBlocked,
    TaskCompleted,
    TaskFailed,
    TaskRetryScheduled,
    TaskStarted,
    TextDelta,
    ToolFinished,
    ToolStarted,
    ValidationFinished,
    ValidationStarted,
)
from beast.core.task_manager import TaskProgress
from beast.errors import BeastError
from beast.intelligence import ProjectIntelligence
from beast.intelligence.models import ProjectDetection
from beast.ollama.connection import OllamaStatus
from beast.ollama.models import ModelInventory, OllamaModel
from beast.tools import ApprovalRequest, Permission, ToolContext, ToolRegistry
from beast.workspace import Workspace

from .branding import ACCENT, BORDER, ERROR, MUTED, SUCCESS, TEXT, WARNING
from .commands import HELP_TEXT, SlashCommand, parse_command
from .widgets import (
    AgentState,
    AnalysisActivity,
    AssistantMessage,
    BrandHeader,
    CommandOutput,
    Logo,
    PromptArea,
    RunActivity,
    RunSummaryView,
    Splash,
    StatusBar,
    SystemNotice,
    TaskBoard,
    ToolActivity,
    UserMessage,
)

if TYPE_CHECKING:
    from .app import BeastApp

logger = logging.getLogger(__name__)

MODEL_HELP = "↑↓ navigate · enter select · ctrl+r refresh · esc back · ctrl+q quit"
CHAT_ROW_HINT = "Enter sends · Shift+Enter adds a line"


class ApprovalScreen(ModalScreen[bool]):
    """Ask the user to approve a write or a command.

    Dismisses with ``True`` only on an explicit approve action.  Escape and a
    lost screen both mean "no": there is no default-yes path.
    """

    BINDINGS = [
        Binding("y", "approve", "Approve"),
        Binding("a", "approve", "Approve", show=False),
        Binding("n", "reject", "Reject"),
        Binding("escape", "reject", "Reject", show=False),
    ]

    def __init__(self, request: ApprovalRequest) -> None:
        super().__init__()
        self._request = request

    def compose(self) -> ComposeResult:
        request = self._request
        if request.permission is Permission.EXECUTE:
            heading = "BEAST wants to run a command:"
        else:
            heading = "BEAST wants to modify:"

        with Vertical(id="approval-panel"):
            yield Static(
                Text(heading, style=f"bold {ACCENT}"), id="approval-heading"
            )
            yield Static(
                Text(request.target, style=f"bold {TEXT}"), id="approval-target"
            )
            yield Static(
                Text(f"Proposed action: {request.action}", style=TEXT),
                id="approval-action",
            )
            if request.detail:
                yield Static(
                    Text(request.detail, style=MUTED), id="approval-detail"
                )
            with Horizontal(id="approval-buttons"):
                yield Button("Approve", id="approve", variant="success")
                yield Button("Reject", id="reject", variant="error")
            yield Static(
                Text("y / a approve · n / esc reject", style=MUTED),
                id="approval-keys",
            )

    def on_mount(self) -> None:
        self.query_one("#approve", Button).focus()

    @on(Button.Pressed, "#approve")
    def _approve_button(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#reject")
    def _reject_button(self) -> None:
        self.dismiss(False)

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_reject(self) -> None:
        self.dismiss(False)


class BeastScreen(Screen):
    """Base screen with a typed accessor for the owning app."""

    @property
    def beast(self) -> BeastApp:
        return self.app  # type: ignore[return-value]

    def get_widget_and_offset_at(
        self, x: int, y: int
    ) -> tuple[Widget | None, Offset | None]:
        """Return the selectable widget under a screen coordinate.

        Textual's compositor can briefly return a widget that has just been
        detached from the DOM (for example an old ``MarkdownParagraph`` while
        ``Markdown.update`` replaces the document).  Textual's selection code
        assumes the hit widget's parent still exists and dereferences
        ``parent.region``, which raises ``AttributeError`` for a detached
        widget.  A detached widget cannot be selected, so report it as empty
        space instead of handing the screen a hit it cannot use.
        """
        widget, offset = super().get_widget_and_offset_at(x, y)
        if widget is not None and not widget.is_attached:
            return None, None
        return widget, offset


class ModelSelectScreen(BeastScreen):
    """Lists every chat-capable model Ollama reports, grouped by location."""

    BINDINGS = [
        Binding("escape", "dismiss", "Back", show=False),
        Binding("ctrl+r", "refresh", "Refresh"),
    ]

    def __init__(self, *, current_model: str | None = None) -> None:
        super().__init__()
        self._current_model = current_model
        self._models: list[OllamaModel] = []
        self._by_id: dict[str, OllamaModel] = {}
        self._local_count = 0
        self._cloud_count = 0

    def compose(self) -> ComposeResult:
        yield Logo(id="select-logo")
        yield Static(Text("Select a model", style=f"bold {TEXT}"), id="select-title")
        yield Static(id="select-meta")
        yield OptionList(id="models")
        yield Static(id="select-note")
        yield Static(Text(MODEL_HELP, style=MUTED), id="select-keys")

    def on_mount(self) -> None:
        self._populate(self.beast.inventory)
        self.query_one("#models", OptionList).focus()

    # ------------------------------------------------------------- population

    def _populate(self, inventory: ModelInventory) -> None:
        self._models = list(inventory.usable)
        self._by_id = {model.identifier: model for model in self._models}
        self._local_count = len(inventory.selectable)
        self._cloud_count = len(inventory.cloud)
        option_list = self.query_one("#models", OptionList)
        option_list.clear_options()

        if not self._models:
            self._set_note(
                "No chat models are available. Add one with "
                "`ollama pull <model>` (cloud models included) and press Ctrl+R.",
                style=WARNING,
            )
            self._update_meta()
            return

        rows: list[Option] = []
        highlight: int | None = None
        first_model: int | None = None
        for heading, group in (
            ("LOCAL MODELS", inventory.selectable),
            ("CLOUD MODELS", inventory.cloud),
        ):
            if not group:
                continue
            rows.append(Option(Text(heading, style=f"bold {MUTED}"), disabled=True))
            for model in group:
                if first_model is None:
                    first_model = len(rows)
                if model.identifier == self._current_model:
                    highlight = len(rows)
                rows.append(Option(self._option_text(model), id=model.identifier))

        option_list.add_options(rows)
        option_list.highlighted = (
            highlight if highlight is not None else (first_model or 0)
        )
        self._set_note(inventory.hidden_note)
        self._update_meta()

    def _option_text(self, model: OllamaModel) -> Text:
        text = Text()
        text.append(model.name, style=f"bold {TEXT}")
        text.append("    ", style=MUTED)
        text.append(model.size_display, style=MUTED)
        text.append("  ·  ", style=BORDER)
        text.append(model.quantization_display, style=MUTED)
        text.append("  ·  ", style=BORDER)
        text.append(model.family_display, style=MUTED)
        if model.parameter_display != "—":
            text.append("  ·  ", style=BORDER)
            text.append(model.parameter_display, style=MUTED)
        text.append("  ·  ", style=BORDER)
        if model.is_cloud:
            text.append(model.location_display, style=f"bold {ACCENT}")
        else:
            text.append(model.location_display, style=MUTED)
        if model.supports_tools is False:
            text.append("  ·  ", style=BORDER)
            text.append("no tool calling", style=WARNING)
        elif model.supports_tools is None:
            text.append("  ·  ", style=BORDER)
            text.append("tool support unknown", style=WARNING)
        return text

    def _update_meta(self) -> None:
        count = len(self._models)
        noun = "model" if count == 1 else "models"
        capable = sum(1 for model in self._models if model.supports_tools is True)
        meta = Text()
        meta.append(f"{count} chat {noun} available", style=MUTED)
        meta.append(
            f"   ·   {self._local_count} local, {self._cloud_count} cloud",
            style=MUTED,
        )
        if count:
            meta.append(
                f"   ·   {capable} can use tools",
                style=f"bold {SUCCESS}" if capable else MUTED,
            )
        meta.append("   ·   ", style=BORDER)
        meta.append(self.beast.status.connection_label, style=f"bold {SUCCESS}")
        meta.append("   ·   ", style=BORDER)
        meta.append(self.beast.workspace.display(), style=MUTED)
        self.query_one("#select-meta", Static).update(meta)

    def _set_note(self, message: str | None, *, style: str = MUTED) -> None:
        note = self.query_one("#select-note", Static)
        note.update(Text(message, style=style) if message else Text(""))

    # ----------------------------------------------------------------- actions

    @on(OptionList.OptionSelected)
    def _handle_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        model = self._by_id.get(event.option_id or "")
        if model is not None:
            self.beast.choose_model(model)

    def action_dismiss(self) -> None:
        if self.beast.has_chat_screen:
            self.app.pop_screen()
        else:
            self.app.exit()

    def action_refresh(self) -> None:
        self._refresh()

    @work(exclusive=True, group="model-refresh")
    async def _refresh(self) -> None:
        self._set_note("Refreshing model list…")
        try:
            inventory = await self.beast.reload_models()
        except BeastError as error:
            message = f"{error.message} {error.hint or ''}".strip()
            self._set_note(message, style=ERROR)
            logger.info("model refresh failed: %s", error)
            return
        self._populate(inventory)
        self.notify("Model list refreshed.", timeout=2)


class ChatScreen(BeastScreen):
    """The interactive conversation workspace."""

    #: Minimum seconds between Markdown re-renders while streaming. Re-parsing
    #: a growing document on every token would burn CPU for no visible gain.
    STREAM_RENDER_INTERVAL = 0.07

    BINDINGS = [
        Binding("escape", "stop", "Stop", show=False),
        Binding("ctrl+n", "choose_model", "Model"),
        Binding("ctrl+o", "toggle_board", "Tasks"),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        *,
        model: OllamaModel,
        workspace: Workspace,
        status: OllamaStatus,
        intelligence: ProjectIntelligence | None = None,
        initial_auto: str | None = None,
    ) -> None:
        super().__init__()
        self._model = model
        self._workspace = workspace
        self._status = status
        self._intelligence = intelligence
        self._detection: ProjectDetection | None = None
        self._agent: Agent | None = None
        self._turn: Worker[None] | None = None
        self._state = AgentState.IDLE
        self._started_at: float | None = None
        #: One tracker per screen so chat writes, /changes and rollback all see
        #: the same session history.
        self._changes = ChangeTracker(workspace.root)
        self._runner: AutonomousRunner | None = None
        self._auto_turn: Worker[None] | None = None
        self._initial_auto = initial_auto

    # ------------------------------------------------------------------ layout

    def compose(self) -> ComposeResult:
        yield BrandHeader(
            workspace=self._workspace,
            connection=self._status.connection_label,
            id="header",
        )
        yield TaskBoard()
        yield VerticalScroll(id="conversation")
        with Vertical(id="composer"):
            yield PromptArea(id="prompt")
        yield StatusBar(
            workspace=self._workspace,
            connection=self._status.connection_label,
            id="statusbar",
        )

    def on_mount(self) -> None:
        self._agent = self._make_agent(self._model)
        self._verify_cloud_auth()
        self.query_one(BrandHeader).set_model(
            self._model.identifier, ollama_version=self._status.version
        )
        conversation = self.query_one("#conversation", VerticalScroll)
        conversation.mount(self._splash())
        notice = self._tool_notice(self._model)
        if notice:
            conversation.mount(SystemNotice(notice))
        self._refresh_status()
        self.query_one(PromptArea).focus()
        self.set_interval(0.5, self._tick)
        self._analyse_project()
        if self._initial_auto:
            request, self._initial_auto = self._initial_auto, None
            self.call_after_refresh(self._start_initial_auto, request)

    def _start_initial_auto(self, request: str) -> None:
        self.run_worker(
            self._begin_auto(request),
            group="auto-start",
            exclusive=True,
        )

    @work(thread=True, exclusive=True, group="project-analysis")
    def _analyse_project(self) -> None:
        """Detect the project in the background and fill in the splash panel."""
        intelligence = self._intelligence
        if intelligence is None:
            return
        try:
            detection = intelligence.detection()
        except Exception:  # pragma: no cover - analysis must never break the TUI
            logger.exception("project detection failed")
            return
        try:
            self.app.call_from_thread(self._show_detection, detection)
        except Exception:  # pragma: no cover - the screen closed first
            logger.debug("could not show project detection", exc_info=True)

    def _show_detection(self, detection: ProjectDetection) -> None:
        self._detection = detection
        for splash in self.query(Splash):
            splash.set_project(detection)

    @work(exclusive=True, group="cloud-auth")
    async def _verify_cloud_auth(self) -> None:
        """Report a missing Ollama Cloud sign-in as soon as a cloud model is chosen.

        This only runs for cloud models and never blocks the conversation; the
        chat request reports the same problem if the check is unavailable.
        """
        if not self._model.is_cloud:
            return
        checker = getattr(self.beast.client, "check_cloud_auth", None)
        if checker is None:  # pragma: no cover - test doubles
            return
        try:
            await checker()
        except BeastError as error:
            message = f"{error.message} {error.hint or ''}".strip()
            logger.info("cloud authentication check failed: %s", error)
            self.notify(message, severity="error", timeout=8)

    def _make_agent(self, model: OllamaModel) -> Agent:
        """Give the agent tools only when this model advertises support."""
        registry: ToolRegistry | None = None
        if model.supports_tools is True:
            context = ToolContext(
                workspace=self._workspace,
                approver=self.beast,
                intelligence=self._intelligence,
                changes=self._changes,
            )
            registry = ToolRegistry.default(context)
        return Agent(
            self.beast.client,
            model.identifier,
            tools=registry,
            # Project context is attached to every coding turn.  Chat-only
            # sessions keep their original behaviour and system prompt.
            context_provider=self._intelligence if registry is not None else None,
        )

    @staticmethod
    def _tool_notice(model: OllamaModel) -> str | None:
        support = model.supports_tools
        if support is True:
            return None
        if support is False:
            return (
                f"{model.identifier} does not advertise tool calling. "
                "BEAST will chat without tools."
            )
        return (
            f"Ollama did not report tool support for {model.identifier}. "
            "BEAST will chat without tools; a newer Ollama is needed for agents."
        )

    @property
    def current_model(self) -> str:
        if self._agent is not None:
            return self._agent.model
        return self._model.identifier

    def _splash(self) -> Splash:
        return Splash(
            workspace=self._workspace,
            model=self.current_model,
            connection=self._status.connection_label,
            project=self._detection,
        )

    # ------------------------------------------------------------------- turns

    @on(PromptArea.Submitted)
    async def _handle_submit(self, event: PromptArea.Submitted) -> None:
        event.stop()
        command = parse_command(event.value)
        if command is not None:
            if command.name == "cancel":
                self.action_stop()
                return
            if self._state is not AgentState.IDLE:
                self.notify(
                    "BEAST is still working — press Esc to stop it first.",
                    severity="warning",
                    timeout=3,
                )
                return
            await self._handle_command(command)
            return
        if self._state is not AgentState.IDLE:
            self.notify(
                "BEAST is still working — press Esc to stop it first.",
                severity="warning",
                timeout=3,
            )
            return
        await self._begin_turn(event.value)

    # ------------------------------------------------------------- commands

    async def _handle_command(self, command: SlashCommand) -> None:
        self.query_one(PromptArea).clear_text()
        if command.name == "help":
            await self._show_output(HELP_TEXT)
        elif command.name == "auto":
            await self._begin_auto(command.argument)
        elif command.name == "changes":
            await self._show_output(self._changes.report().render())
        elif command.name == "diff":
            await self._show_diff(command.argument)
        elif command.name == "rollback":
            # Approval modals wait on a worker context, so run the command in
            # one instead of blocking the message handler.
            self.run_worker(
                self._rollback(command.argument),
                group="rollback",
                exclusive=True,
                name="rollback",
            )
        elif command.name == "git":
            await self._show_git()
        else:
            await self._show_notice(
                f"Unknown command /{command.name}. Type /help for the list."
            )

    async def _show_output(self, output: str) -> None:
        conversation = self.query_one("#conversation", VerticalScroll)
        await conversation.mount(CommandOutput(output))
        self._scroll_to_end()

    async def _show_notice(self, message: str) -> None:
        conversation = self.query_one("#conversation", VerticalScroll)
        await conversation.mount(SystemNotice(message))
        self._scroll_to_end()

    async def _show_diff(self, argument: str) -> None:
        paths = [argument] if argument else list(self._changes.changed_paths())
        if not paths:
            await self._show_notice("No BEAST-attributed changes to diff.")
            return
        chunks: list[str] = []
        for path in paths:
            diff = self._changes.diff(path)
            if diff.strip():
                chunks.append(diff)
        await self._show_output(
            "\n\n".join(chunks) if chunks else "No textual diff is available."
        )

    async def _rollback(self, argument: str) -> None:
        paths = argument.split() if argument else []
        report = self._changes.report()
        if report.active_files == ():
            await self._show_notice("Nothing to roll back.")
            return
        target = ", ".join(paths) if paths else "all BEAST changes"
        request = ApprovalRequest(
            tool="rollback",
            permission=Permission.WRITE,
            target=target,
            action="Restore BEAST-attributed files to their pre-BEAST content",
            detail=(
                "Only files whose current content still matches BEAST's last "
                "write are touched. Unrelated edits are preserved."
            ),
        )
        if not await self.beast.approve(request):
            await self._show_notice("Rollback rejected; nothing was changed.")
            return
        result = self._changes.rollback(paths or None)
        board = self.query_one(TaskBoard)
        board.set_changes(self._changes.report())
        await self._show_output(result.render())

    async def _show_git(self) -> None:
        git = GitInspector(self._workspace.root)
        status = await git.status()
        await self._show_output(status.render())

    # --------------------------------------------------------- autonomous run

    async def _begin_auto(self, request: str) -> None:
        request = request.strip()
        if not request:
            await self._show_notice("Usage: /auto <request>")
            return
        if self._model.supports_tools is not True:
            await self._show_notice(
                f"{self._model.identifier} cannot use tools, so autonomous mode "
                "is unavailable. Pick a tool-capable model with Ctrl+N."
            )
            return

        prompt = self.query_one(PromptArea)
        prompt.clear_text()
        prompt.can_submit = False

        conversation = self.query_one("#conversation", VerticalScroll)
        await conversation.mount(UserMessage(f"/auto {request}"))
        await conversation.mount(
            SystemNotice(
                "Autonomous mode — the model plans, then executes one task at a time. "
                "Writes and commands still need approval."
            )
        )
        self._scroll_to_end()

        self._state = AgentState.PLANNING
        self._started_at = time.monotonic()
        self._refresh_status(elapsed=0.0)
        self.query_one(TaskBoard).display = True

        self._runner = AutonomousRunner(
            self.beast.client,
            self._model.identifier,
            workspace=self._workspace,
            approver=self.beast,
            intelligence=self._intelligence,
            tracker=self._changes,
            context_provider=self._intelligence,
            supports_tools=True,
        )
        self._auto_turn = self._stream_auto(request)

    @work(exclusive=True, group="auto", name="auto")
    async def _stream_auto(self, request: str) -> None:
        runner = self._runner
        if runner is None:  # pragma: no cover - defensive
            return

        conversation = self.query_one("#conversation", VerticalScroll)
        board = self.query_one(TaskBoard)
        activity: AnalysisActivity | None = None
        tool_widget: ToolActivity | None = None
        message: AssistantMessage | None = None

        try:
            async for event in runner.run(request):
                if isinstance(event, PlanStarted):
                    await conversation.mount(
                        RunActivity("plan", "Planning the work…")
                    )
                    self._scroll_to_end()
                elif isinstance(event, PlanReady):
                    board.start_run(
                        event.tasks, TaskProgress.initial(len(event.tasks))
                    )
                    if event.plan.warnings:
                        await conversation.mount(
                            RunActivity(
                                "plan",
                                f"Plan ready — {len(event.tasks)} tasks "
                                f"({len(event.plan.warnings)} warning(s))",
                            )
                        )
                    else:
                        await conversation.mount(
                            RunActivity(
                                "plan", f"Plan ready — {len(event.tasks)} tasks"
                            )
                        )
                    self._scroll_to_end()
                elif isinstance(event, PlanRevised):
                    board.replace_tasks(event.tasks)
                    await conversation.mount(
                        RunActivity(
                            "plan",
                            f"Plan revised (revision {event.revision}): "
                            f"{event.reason[:160]}",
                        )
                    )
                    self._scroll_to_end()
                elif isinstance(event, TaskStarted):
                    self._state = AgentState.TASK
                    self._refresh_status(elapsed=self._elapsed())
                    board.update_task(event.task, event.progress)
                    board.set_current(event.task)
                elif isinstance(
                    event, (TaskCompleted, TaskFailed, TaskBlocked)
                ):
                    board.update_task(event.task, event.progress)
                    if isinstance(event, TaskFailed):
                        await conversation.mount(
                            RunActivity(
                                "error",
                                f"{event.task.id} failed: {event.detail[:200]}",
                            )
                        )
                    elif isinstance(event, TaskBlocked):
                        await conversation.mount(
                            RunActivity(
                                "error",
                                f"{event.task.id} blocked: {event.reason[:200]}",
                            )
                        )
                    self._scroll_to_end(follow_only=True)
                elif isinstance(event, TaskRetryScheduled):
                    board.update_task(event.task)
                elif isinstance(event, ValidationStarted):
                    self._state = AgentState.VALIDATING
                    self._refresh_status(elapsed=self._elapsed())
                    board.set_validation(
                        f"validating {len(event.files)} file(s)…"
                    )
                elif isinstance(event, ValidationFinished):
                    self._state = AgentState.TASK
                    self._refresh_status(elapsed=self._elapsed())
                    board.set_validation(event.report.status_text)
                    await conversation.mount(
                        RunActivity("validation", event.report.status_text)
                    )
                    self._scroll_to_end(follow_only=True)
                elif isinstance(event, DebugAttempt):
                    self._state = AgentState.DEBUGGING
                    self._refresh_status(elapsed=self._elapsed())
                    board.set_debug(
                        f"debug {event.attempt}/{event.max_attempts}"
                    )
                    await conversation.mount(
                        RunActivity(
                            "debug",
                            f"correcting validation failure "
                            f"({event.attempt}/{event.max_attempts})",
                        )
                    )
                    self._scroll_to_end(follow_only=True)
                elif isinstance(event, ChangeSummary):
                    board.set_changes(event.report)
                elif isinstance(event, TextDelta):
                    # Task agents are not streamed into the transcript: the
                    # board and the final summary are the readable output.
                    continue
                elif isinstance(event, AnalysisEvent):
                    if event.stage == "error":
                        if activity is None:
                            activity = AnalysisActivity(event.message)
                            await conversation.mount(activity)
                        else:
                            activity.set_message(event.stage, event.message)
                elif isinstance(event, ToolStarted):
                    tool_widget = ToolActivity(
                        event.name, event.arguments, event.permission
                    )
                    await conversation.mount(tool_widget)
                    self._state = AgentState.TOOL
                    self._refresh_status(elapsed=self._elapsed())
                    self._scroll_to_end()
                elif isinstance(event, ApprovalPending):
                    if tool_widget is not None:
                        tool_widget.mark_awaiting_approval()
                    self._state = AgentState.APPROVAL
                    self._refresh_status(elapsed=self._elapsed())
                elif isinstance(event, ToolFinished):
                    if tool_widget is not None:
                        await tool_widget.finish(event.result)
                    self._state = AgentState.TASK
                    self._refresh_status(elapsed=self._elapsed())
                    self._scroll_to_end(follow_only=True)
                elif isinstance(event, RunFinished):
                    board.set_summary(event.summary)
                    await conversation.mount(RunSummaryView(event.summary))
                    self._scroll_to_end()

            self._scroll_to_end(follow_only=True)
        except asyncio.CancelledError:
            if tool_widget is not None:
                tool_widget.mark_interrupted()
            board.set_validation("stopped")
            self.app.call_later(
                self._auto_notice, "Autonomous run stopped by the user."
            )
            raise
        except BeastError as error:
            logger.info("autonomous run failed: %s", error)
            if tool_widget is not None:
                tool_widget.mark_interrupted("Run ended before this tool reported.")
            message = AssistantMessage(self.current_model)
            await conversation.mount(message)
            await message.fail(error)
        except Exception:  # noqa: BLE001 - the TUI must never die mid-run
            logger.exception("unexpected failure during an autonomous run")
            if tool_widget is not None:
                tool_widget.mark_interrupted("Run ended before this tool reported.")
            message = AssistantMessage(self.current_model)
            await conversation.mount(message)
            await message.fail(
                BeastError(
                    "BEAST hit an unexpected error during the autonomous run.",
                    hint="Run again with --debug to see the full traceback.",
                )
            )
        finally:
            self._set_idle()

    def _auto_notice(self, text: str) -> None:
        try:
            conversation = self.query_one("#conversation", VerticalScroll)
            conversation.mount(SystemNotice(text))
            self._scroll_to_end()
        except NoMatches:  # pragma: no cover - screen closing
            logger.debug("could not show the autonomous notice")

    async def _begin_turn(self, text: str) -> None:
        prompt = self.query_one(PromptArea)
        prompt.clear_text()
        prompt.can_submit = False

        conversation = self.query_one("#conversation", VerticalScroll)
        await conversation.mount(UserMessage(text))
        self._scroll_to_end()

        self._state = AgentState.THINKING
        self._started_at = time.monotonic()
        self._refresh_status(elapsed=0.0)
        self._turn = self._stream_turn(text)

    @work(exclusive=True, group="chat", name="chat")
    async def _stream_turn(self, text: str) -> None:
        agent = self._agent
        if agent is None:  # pragma: no cover - defensive
            return

        conversation = self.query_one("#conversation", VerticalScroll)
        analysis: AnalysisActivity | None = None
        if agent.context_provider is not None:
            analysis = AnalysisActivity()
            await conversation.mount(analysis)
        message: AssistantMessage | None = AssistantMessage(agent.model)
        await conversation.mount(message)
        self._scroll_to_end()

        buffer = ""
        last_render = 0.0
        tool_widget: ToolActivity | None = None

        try:
            async for event in agent.run(text):
                if isinstance(event, TextDelta):
                    if message is None:
                        message = AssistantMessage(agent.model)
                        await conversation.mount(message)
                        self._scroll_to_end()
                    buffer += event.text
                    now = time.monotonic()
                    if now - last_render >= self.STREAM_RENDER_INTERVAL:
                        last_render = now
                        await message.update_body(buffer)
                        self._scroll_to_end(follow_only=True)
                elif isinstance(event, AnalysisEvent):
                    if analysis is None:
                        analysis = AnalysisActivity(event.message)
                        await conversation.mount(analysis)
                    else:
                        analysis.set_message(event.stage, event.message)
                    self._scroll_to_end()
                elif isinstance(event, ToolStarted):
                    if message is not None:
                        if buffer:
                            await message.update_body(buffer)
                        await message.finish(empty_hint=False)
                        message = None
                        buffer = ""
                    tool_widget = ToolActivity(
                        event.name, event.arguments, event.permission
                    )
                    await conversation.mount(tool_widget)
                    self._state = AgentState.TOOL
                    self._refresh_status(elapsed=self._elapsed())
                    self._scroll_to_end()
                elif isinstance(event, ApprovalPending):
                    if tool_widget is not None:
                        tool_widget.mark_awaiting_approval()
                    self._state = AgentState.APPROVAL
                    self._refresh_status(elapsed=self._elapsed())
                elif isinstance(event, ToolFinished):
                    if tool_widget is not None:
                        await tool_widget.finish(event.result)
                    self._state = AgentState.THINKING
                    self._refresh_status(elapsed=self._elapsed())
                    self._scroll_to_end(follow_only=True)

            if message is not None:
                if buffer:
                    await message.update_body(buffer)
                await message.finish()
            self._scroll_to_end(follow_only=True)
        except asyncio.CancelledError:
            # The screen may still be alive; finish the partial message from a
            # fresh task because this one is being torn down.
            if tool_widget is not None:
                tool_widget.mark_interrupted()
            if message is not None:
                self.app.call_later(self._finish_interrupted, message, buffer)
            raise
        except BeastError as error:
            logger.info("chat request failed: %s", error)
            if tool_widget is not None:
                tool_widget.mark_interrupted("Turn ended before this tool reported.")
            if message is None:
                message = AssistantMessage(agent.model)
                await conversation.mount(message)
            await self._flush_partial(message, buffer)
            await message.fail(error)
        except Exception:  # noqa: BLE001 - the TUI must never die mid-turn
            logger.exception("unexpected failure while streaming a reply")
            if tool_widget is not None:
                tool_widget.mark_interrupted("Turn ended before this tool reported.")
            if message is None:
                message = AssistantMessage(agent.model)
                await conversation.mount(message)
            await self._flush_partial(message, buffer)
            await message.fail(
                BeastError(
                    "BEAST hit an unexpected error while generating a reply.",
                    hint=(
                        "Try again. If it keeps happening, run "
                        "`ASAP launch beast --debug` to see the details."
                    ),
                )
            )
        finally:
            self._set_idle()

    @staticmethod
    async def _flush_partial(message: AssistantMessage, buffer: str) -> None:
        """Show text received before a failure, matching the agent's history."""
        if not buffer:
            return
        try:
            await message.update_body(buffer)
        except Exception:  # pragma: no cover - screen closing during shutdown
            logger.debug("could not show the partial reply", exc_info=True)

    async def _finish_interrupted(
        self, message: AssistantMessage, buffer: str
    ) -> None:
        if not message.is_mounted:  # pragma: no cover - app is closing
            return
        try:
            if buffer:
                await message.update_body(buffer)
            await message.finish(interrupted=True)
        except Exception:  # pragma: no cover - screen closing during shutdown
            logger.debug("could not finalise interrupted message", exc_info=True)

    # ------------------------------------------------------------------ status

    def _tick(self) -> None:
        if self._state is not AgentState.IDLE and self._started_at is not None:
            self._refresh_status(elapsed=time.monotonic() - self._started_at)

    def _elapsed(self) -> float | None:
        if self._started_at is None:
            return None
        return time.monotonic() - self._started_at

    def _refresh_status(self, *, elapsed: float | None = None) -> None:
        try:
            bar = self.query_one(StatusBar)
        except NoMatches:  # pragma: no cover - before compose finishes
            return
        bar.set_context(model=self.current_model, state=self._state, elapsed=elapsed)

    def _set_idle(self) -> None:
        self._state = AgentState.IDLE
        self._started_at = None
        self._refresh_status()
        try:
            prompt = self.query_one(PromptArea)
        except NoMatches:  # pragma: no cover - screen closing
            return
        prompt.can_submit = True
        prompt.focus()

    # ----------------------------------------------------------------- scroll

    def _scroll_to_end(self, *, follow_only: bool = False) -> None:
        try:
            conversation = self.query_one("#conversation", VerticalScroll)
        except NoMatches:  # pragma: no cover - before compose finishes
            return
        if follow_only and not self._is_following(conversation):
            return
        conversation.scroll_end(animate=False)

    @staticmethod
    def _is_following(conversation: VerticalScroll) -> bool:
        offset = conversation.scroll_offset.y
        return offset + conversation.size.height >= conversation.virtual_size.height - 2

    # ---------------------------------------------------------------- actions

    def action_stop(self) -> None:
        if self._state is AgentState.IDLE:
            return
        if self._runner is not None and not self._runner.finished:
            self._runner.request_stop()
        for worker in (self._auto_turn, self._turn):
            if worker is not None and not worker.is_finished:
                worker.cancel()
                break

    def action_toggle_board(self) -> None:
        board = self.query_one(TaskBoard)
        board.display = not board.display

    def action_choose_model(self) -> None:
        self.beast.open_model_picker()

    async def action_clear(self) -> None:
        if self._state is not AgentState.IDLE:
            self.notify(
                "Stop the current reply first (Esc).", severity="warning", timeout=3
            )
            return
        if self._agent is not None:
            self._agent.reset()
        self._runner = None
        conversation = self.query_one("#conversation", VerticalScroll)
        await conversation.remove_children()
        await conversation.mount(self._splash())
        self.query_one(TaskBoard).display = False
        self.notify("Conversation cleared — BEAST starts fresh.", timeout=2)

    def action_quit(self) -> None:
        self.app.exit()

    def switch_model(self, model: OllamaModel) -> bool:
        """Point this screen at ``model``. Returns False when nothing changed."""
        if self._agent is not None and self._agent.model == model.identifier:
            return False

        if self._turn is not None and not self._turn.is_finished:
            self._turn.cancel()
        if self._auto_turn is not None and not self._auto_turn.is_finished:
            self._auto_turn.cancel()
        if self._runner is not None:
            self._runner.request_stop()
        self._runner = None

        self._model = model
        self._agent = self._make_agent(model)
        self._state = AgentState.IDLE
        self._started_at = None
        self._verify_cloud_auth()

        try:
            self.query_one(BrandHeader).set_model(
                model.identifier, ollama_version=self._status.version
            )
            self.query_one(TaskBoard).display = False
            conversation = self.query_one("#conversation", VerticalScroll)
            conversation.mount(
                SystemNotice(f"Switched to {model.identifier} — new conversation")
            )
            notice = self._tool_notice(model)
            if notice:
                conversation.mount(SystemNotice(notice))
            for splash in self.query(Splash):
                splash.set_model(model.identifier)
            self._scroll_to_end()
            self._refresh_status()
            prompt = self.query_one(PromptArea)
            prompt.can_submit = True
            prompt.focus()
        except NoMatches:  # pragma: no cover - screen closing
            logger.debug("could not update widgets for model switch")
        return True
