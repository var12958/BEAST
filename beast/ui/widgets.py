"""Reusable widgets for the BEAST terminal interface."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Markdown, Static, TextArea

from beast import __version__
from beast.core.changes import ChangeReport
from beast.core.task_manager import Task, TaskProgress, TaskStatus
from beast.errors import BeastError
from beast.intelligence.models import ProjectDetection
from beast.tools import Permission, ToolResult
from beast.workspace import Workspace

if TYPE_CHECKING:  # pragma: no cover - avoids importing the runner into the UI
    from beast.core.autonomy import RunSummary

from .branding import (
    ACCENT,
    APP_NAME,
    APP_TITLE,
    BORDER,
    ERROR,
    LOGO_MARK,
    MUTED,
    PROMPT_PLACEHOLDER,
    SPLASH_LOGO,
    SPLASH_WIDTH,
    SUCCESS,
    TAGLINE,
    TEXT,
    USER,
    WARNING,
)


class AgentState(StrEnum):
    """What the agent is doing right now."""

    IDLE = "IDLE"
    THINKING = "THINKING"
    TOOL = "TOOL"
    APPROVAL = "APPROVAL"
    PLANNING = "PLANNING"
    TASK = "TASK"
    VALIDATING = "VALIDATING"
    DEBUGGING = "DEBUGGING"


# --------------------------------------------------------------------- prompt


class PromptArea(TextArea):
    """The prompt box.

    Enter sends, Shift+Enter (or Ctrl+J where the terminal cannot report
    Shift+Enter) inserts a newline.
    """

    BINDINGS = [("ctrl+j", "newline", "New line")]

    class Submitted(Message):
        """Posted when the user sends a non-empty prompt."""

        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def __init__(
        self,
        *,
        placeholder: str = PROMPT_PLACEHOLDER,
        id: str | None = None,  # noqa: A002 - matches Textual's widget API
    ) -> None:
        super().__init__(
            text="",
            placeholder=placeholder,
            soft_wrap=True,
            show_line_numbers=False,
            tab_behavior="focus",
            id=id,
        )
        #: Set to ``False`` while a reply is streaming so Enter cannot queue
        #: a second request.
        self.can_submit = True

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            self.action_submit()
            return
        if event.key in {"shift+enter", "ctrl+enter"}:
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return
        await super()._on_key(event)

    def action_submit(self) -> None:
        value = self.text.strip()
        if not value or not self.can_submit:
            return
        self.post_message(self.Submitted(value))

    def action_newline(self) -> None:
        self.insert("\n")

    def clear_text(self) -> None:
        """Reset the input to empty."""
        self.load_text("")
        self.move_cursor((0, 0))


# ---------------------------------------------------------------------- chrome


class BrandHeader(Vertical):
    """Persistent header: wordmark, model, connection state and workspace."""

    def __init__(
        self,
        *,
        workspace: Workspace,
        connection: str,
        id: str | None = None,  # noqa: A002
    ) -> None:
        super().__init__(id=id)
        self._workspace = workspace
        self._connection = connection
        self._ollama_version: str | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="brand-row"):
            yield Static(Text(LOGO_MARK, style=f"bold {ACCENT}"), id="brand-logo")
            with Vertical(id="brand-info"):
                yield Static(
                    Text.assemble(
                        (APP_TITLE, f"bold {TEXT}"),
                        ("   ", MUTED),
                        (f"v{__version__}", MUTED),
                    ),
                    id="brand-title",
                )
                yield Static(id="brand-meta")
                yield Static(
                    Text.assemble(
                        ("Workspace  ", MUTED),
                        (self._workspace.display(), TEXT),
                    ),
                    id="brand-workspace",
                )

    def on_mount(self) -> None:
        self.set_model(None)
        self._apply_responsive()

    def on_resize(self, event: events.Resize) -> None:
        self._apply_responsive()

    def _apply_responsive(self) -> None:
        try:
            logo = self.query_one("#brand-logo", Static)
        except Exception:  # pragma: no cover - widget not mounted yet
            return
        logo.display = self.size.width >= 62

    def set_model(self, model: str | None, *, ollama_version: str | None = None) -> None:
        if ollama_version is not None:
            self._ollama_version = ollama_version
        meta = Text()
        meta.append("Model  ", style=MUTED)
        meta.append(model or "not selected", style=f"bold {ACCENT}")
        meta.append("   ·   ", style=BORDER)
        meta.append("Connection  ", style=MUTED)
        meta.append(self._connection, style=f"bold {SUCCESS}")
        if self._ollama_version:
            meta.append("   ·   ", style=BORDER)
            meta.append(f"Ollama v{self._ollama_version}", style=MUTED)
        try:
            self.query_one("#brand-meta", Static).update(meta)
        except Exception:  # pragma: no cover - widget not mounted yet
            pass


class StatusBar(Static):
    """One-line status bar, adaptive to terminal width."""

    SHORTCUTS_FULL = (
        "enter send · shift+enter nl · esc stop · "
        "ctrl+n model · ctrl+o tasks · ctrl+l clear · ctrl+q quit"
    )
    SHORTCUTS_SHORT = "esc stop · ctrl+n model · ctrl+o tasks · ctrl+q quit"

    def __init__(
        self,
        *,
        workspace: Workspace,
        connection: str,
        id: str | None = None,  # noqa: A002
    ) -> None:
        super().__init__(id=id)
        self._workspace = workspace
        self._connection = connection
        self._model = "—"
        self._state = AgentState.IDLE
        self._elapsed: float | None = None

    def set_context(
        self,
        *,
        model: str | None = None,
        state: AgentState | None = None,
        elapsed: float | None = None,
    ) -> None:
        if model is not None:
            self._model = model
        if state is not None:
            self._state = state
        if state is AgentState.IDLE:
            self._elapsed = None
        else:
            self._elapsed = elapsed
        self.refresh()

    def render(self) -> Text:
        width = self.size.width or 100
        busy = self._state is not AgentState.IDLE
        colour = WARNING if self._state is AgentState.APPROVAL else (
            ACCENT if busy else SUCCESS
        )

        text = Text()
        text.append("● ", style=f"bold {colour}")
        label = self._state.value
        if busy and self._elapsed is not None:
            label = f"{label} {self._elapsed:.1f}s"
        text.append(label, style=f"bold {colour}")

        text.append("   " + self._model, style=f"bold {TEXT}")
        if width >= 78:
            text.append("   " + self._connection, style=MUTED)
        if width >= 112:
            text.append("   " + self._workspace.display(), style=MUTED)

        shortcuts = ""
        if width >= 128:
            shortcuts = self.SHORTCUTS_FULL
        elif width >= 88:
            shortcuts = self.SHORTCUTS_SHORT

        if shortcuts:
            padding = width - len(text.plain) - len(shortcuts) - 1
            if padding >= 2:
                text.append(" " * padding, style=MUTED)
                text.append(shortcuts, style=MUTED)

        return text


class Logo(Static):
    """The BEAST wordmark, full when there is room, compact otherwise."""

    def render(self) -> Text:
        width = self.size.width or 80
        if width >= SPLASH_WIDTH:
            return Text(SPLASH_LOGO, style=f"bold {ACCENT}")
        return Text(LOGO_MARK, style=f"bold {ACCENT}")


class Splash(Static):
    """Welcome panel shown at the top of a fresh conversation.

    Beyond the model and connection it shows what BEAST detected about the
    project.  Detection runs in the background, so the panel starts in a
    "scanning" state and fills in when the analysis worker reports back.
    """

    def __init__(
        self,
        *,
        workspace: Workspace,
        model: str,
        connection: str,
        project: ProjectDetection | None = None,
    ) -> None:
        super().__init__()
        self._workspace = workspace
        self._model = model
        self._connection = connection
        self._project = project

    def set_model(self, model: str) -> None:
        """Update the model shown in the welcome panel."""
        if model == self._model:
            return
        self._model = model
        self.refresh()

    def set_project(self, project: ProjectDetection) -> None:
        """Fill in the project indicators once analysis has finished."""
        self._project = project
        self.refresh()

    def _project_rows(self) -> tuple[tuple[str, str, str], ...]:
        detection = self._project
        if detection is None:
            return (
                ("Language      ", "scanning…", MUTED),
                ("Framework     ", "—", MUTED),
                ("Pkg manager   ", "—", MUTED),
                ("Source files  ", "—", MUTED),
                ("Test files    ", "—", MUTED),
            )

        languages = detection.language_names
        language_text = ", ".join(languages[:4]) or "none detected"
        if len(languages) > 4:
            language_text += f" +{len(languages) - 4} more"

        framework_text = ", ".join(detection.frameworks[:3]) or "none detected"
        if len(detection.frameworks) > 3:
            framework_text += f" +{len(detection.frameworks) - 3} more"

        manager_text = ", ".join(detection.package_managers[:2]) or "none detected"

        return (
            ("Language      ", language_text, TEXT),
            ("Framework     ", framework_text, TEXT),
            ("Pkg manager   ", manager_text, TEXT),
            ("Source files  ", str(detection.source_file_count), TEXT),
            ("Test files    ", str(detection.test_file_count), TEXT),
        )

    def render(self) -> Text:
        width = self.size.width or 80
        text = Text()

        if width >= SPLASH_WIDTH:
            text.append(SPLASH_LOGO + "\n\n", style=f"bold {ACCENT}")
        else:
            text.append(f"{APP_NAME}\n", style=f"bold {ACCENT}")

        text.append(TAGLINE, style=TEXT)
        text.append("\n\n")

        rows = (
            ("Model       ", self._model, ACCENT),
            ("Workspace   ", self._workspace.display(), TEXT),
            ("Connection  ", self._connection, SUCCESS),
            *self._project_rows(),
        )
        for label, value, style in rows:
            text.append(label, style=MUTED)
            text.append(value, style=f"bold {style}")
            text.append("\n")

        text.append("\n")
        text.append(
            "Ask BEAST to explain, design, write or debug code. "
            "You can paste snippets and stack traces directly.",
            style=TEXT,
        )
        text.append("\n")
        text.append(
            "Enter sends · Shift+Enter adds a line · Esc stops a reply · Ctrl+N changes model",
            style=MUTED,
        )
        return text


# -------------------------------------------------------------------- messages


class UserMessage(Vertical):
    """A prompt the user sent."""

    def __init__(self, text: str) -> None:
        super().__init__(classes="message user-message")
        self._text = text

    def compose(self) -> ComposeResult:
        yield Static(Text("▌ You", style=f"bold {USER}"), classes="message-label")
        yield Static(Text(self._text, style=TEXT), classes="message-body")


class AssistantMessage(Vertical):
    """A streamed model reply, rendered as Markdown."""

    def __init__(self, model: str) -> None:
        super().__init__(classes="message assistant-message")
        self._model = model
        self._body = ""
        #: Text already handed to the Markdown widget. Streaming appends the
        #: new suffix instead of rebuilding every block, which keeps existing
        #: paragraphs mounted while the reply grows.
        self._rendered = ""

    def compose(self) -> ComposeResult:
        yield Static(
            Text.assemble(
                ("◆ BEAST", f"bold {ACCENT}"),
                (f"   {self._model}", MUTED),
            ),
            classes="message-label",
        )
        yield Markdown("", classes="message-body")
        yield Static("", classes="message-footer")

    async def update_body(self, markdown_text: str) -> None:
        """Show the text received so far.

        Streaming sends a growing cumulative string. Appending only the new
        suffix preserves already-mounted Markdown blocks; a full
        ``Markdown.update`` detaches all of them and can leave a stale block
        in the compositor, which crashes Textual's mouse-selection path.
        """
        self._body = markdown_text
        if markdown_text == self._rendered:
            return

        markdown = self.query_one(Markdown)
        if not markdown_text.startswith(self._rendered):
            # A replacement rather than a continuation (defensive: current
            # callers only ever extend the text).
            self._rendered = markdown_text
            await markdown.update(markdown_text)
            return

        fragment = markdown_text[len(self._rendered) :]
        self._rendered = markdown_text
        if fragment:
            await markdown.append(fragment)

    async def finish(
        self,
        *,
        interrupted: bool = False,
        empty_hint: bool = True,
    ) -> None:
        """Mark the reply complete (or stopped by the user).

        ``empty_hint`` is disabled for a text phase that ends because the model
        moved on to a tool call — an empty bubble there is normal, not a failed
        response.
        """
        if interrupted:
            self._set_footer(
                Text("Stopped — the partial reply stays in the conversation.", style=MUTED)
            )
        elif not self._body.strip() and empty_hint:
            self._set_footer(
                Text(
                    "The model returned an empty response. Try rephrasing your prompt.",
                    style=WARNING,
                )
            )
        else:
            self._set_footer(None)

    async def fail(self, error: BeastError) -> None:
        """Show a graceful error instead of a traceback."""
        text = Text()
        text.append(f"✕ {error.title}", style=f"bold {ERROR}")
        text.append(f"\n{error.message}", style=ERROR)
        if error.hint:
            text.append(f"\n{error.hint}", style=MUTED)
        self._set_footer(text)

    def _set_footer(self, content: Text | None) -> None:
        footer = self.query_one(".message-footer", Static)
        if content is None:
            footer.update("")
            footer.display = False
        else:
            footer.update(content)
            footer.display = True


class SystemNotice(Static):
    """A muted in-transcript notice, e.g. when the model changes."""

    def __init__(self, text: str) -> None:
        super().__init__(Text(f"— {text} —", style=f"italic {MUTED}"), classes="system-notice")


class AnalysisActivity(Static):
    """One ``[ANALYSIS]`` line describing project-context work.

    The agent emits several analysis events per turn (scan, then a context
    summary).  They update this single line rather than adding a row each, so
    a busy project never floods the transcript.
    """

    def __init__(self, message: str = "Analysing project context…") -> None:
        super().__init__(classes="analysis-activity")
        self._stage = "scan"
        self._message = message

    def set_message(self, stage: str, message: str) -> None:
        self._stage = stage
        self._message = message
        self.refresh()

    def render(self) -> Text:
        colour = WARNING if self._stage == "error" else ACCENT
        text = Text()
        text.append("[ANALYSIS] ", style=f"bold {colour}")
        text.append(self._message, style=MUTED)
        return text


class ToolActivity(Static):
    """One tool call rendered as a single transcript line.

    The line moves through four visible phases: running, awaiting approval,
    finished successfully, and failed or rejected.
    """

    def __init__(
        self,
        name: str,
        arguments: Any,
        permission: Permission,
    ) -> None:
        super().__init__(classes="tool-activity")
        self._name = name
        self._arguments = arguments
        self._permission = permission
        self._phase = "running"
        self._summary = ""

    def mark_awaiting_approval(self) -> None:
        self._phase = "approval"
        self.refresh()

    def mark_interrupted(self, summary: str = "Stopped by the user.") -> None:
        """Mark a call that never reported a result (cancel or failure).

        Finished calls are left untouched.
        """
        if self._phase not in ("running", "approval"):
            return
        self._phase = "rejected"
        self._summary = summary
        self.refresh()

    async def finish(self, result: ToolResult) -> None:
        if result.rejected:
            self._phase = "rejected"
        else:
            self._phase = "ok" if result.ok else "failed"
        self._summary = result.summary
        self.refresh()

    def render(self) -> Text:
        glyph, colour = {
            "running": ("▸", ACCENT),
            "approval": ("…", WARNING),
            "ok": ("✓", SUCCESS),
            "failed": ("✕", ERROR),
            "rejected": ("⊘", WARNING),
        }[self._phase]

        text = Text()
        text.append(f"{glyph} [TOOL] {self._name}", style=f"bold {colour}")
        hint = _argument_hint(self._arguments)
        if hint:
            text.append(f"  {hint}", style=MUTED)
        if self._phase == "approval":
            text.append("  · awaiting approval", style=WARNING)
        elif self._summary:
            text.append(f"  · {self._summary[:120]}", style=MUTED)
        return text


def _argument_hint(arguments: Any) -> str:
    """A short, human-facing hint about what a tool call targets."""
    if not isinstance(arguments, Mapping):
        return ""
    for key in ("path", "query", "command"):
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return value if len(value) <= 80 else f"{value[:77]}…"
    return ""


# ------------------------------------------------------------ autonomous mode


_TASK_GLYPHS = {
    TaskStatus.PENDING: ("○", MUTED),
    TaskStatus.IN_PROGRESS: ("▶", ACCENT),
    TaskStatus.COMPLETED: ("✓", SUCCESS),
    TaskStatus.FAILED: ("✕", ERROR),
    TaskStatus.BLOCKED: ("⊘", WARNING),
    TaskStatus.CANCELLED: ("–", MUTED),
}


class TaskBoard(Static):
    """Compact, persistent view of the plan and the current run state.

    Hidden until an autonomous run starts.  It shows the overall progress, one
    line per task (bounded so a long plan cannot fill the screen), the current
    validation/debug state and the change count.
    """

    MAX_ROWS = 8

    def __init__(self) -> None:
        super().__init__(classes="task-board")
        self.display = False
        self._tasks: tuple[Task, ...] = ()
        self._progress: TaskProgress | None = None
        self._current: str = ""
        self._validation = ""
        self._debug = ""
        self._changes = ""

    def start_run(self, tasks: tuple[Task, ...], progress: TaskProgress) -> None:
        self._tasks = tasks
        self._progress = progress
        self._current = ""
        self._validation = ""
        self._debug = ""
        self._changes = ""
        self.display = True
        self.refresh()

    def update_tasks(
        self, tasks: tuple[Task, ...], progress: TaskProgress, *, current: str = ""
    ) -> None:
        self._tasks = tasks
        self._progress = progress
        if current:
            self._current = current
        self.display = True
        self.refresh()

    def set_current(self, task: Task) -> None:
        self._current = task.id
        self.refresh()

    def update_task(
        self, task: Task, progress: TaskProgress | None = None
    ) -> None:
        """Replace one task's row while keeping the rest of the plan."""
        self._tasks = tuple(
            task if existing.id == task.id else existing
            for existing in self._tasks
        )
        if progress is not None:
            self._progress = progress
        self.display = True
        self.refresh()

    def replace_tasks(
        self, tasks: tuple[Task, ...], progress: TaskProgress | None = None
    ) -> None:
        """Install a revised plan."""
        self._tasks = tasks
        if progress is not None:
            self._progress = progress
        self._current = ""
        self.display = True
        self.refresh()

    def set_validation(self, message: str) -> None:
        self._validation = message
        self.refresh()

    def set_debug(self, message: str) -> None:
        self._debug = message
        self.refresh()

    def set_changes(self, report: ChangeReport) -> None:
        self._changes = report.summary_line()
        self.refresh()

    def set_summary(self, summary: "RunSummary") -> None:
        self._progress = summary.progress
        self._tasks = summary.tasks
        self._current = ""
        self._validation = summary.verdict.upper()
        self._debug = ""
        self._changes = summary.changes.summary_line()
        self.refresh()

    def render(self) -> Text:
        if not self._tasks and self._progress is None:
            return Text("")
        text = Text()
        text.append("TASKS  ", style=f"bold {ACCENT}")
        if self._progress is not None:
            text.append(self._progress.line(), style=MUTED)
        if self._validation:
            text.append("   ·   ", style=BORDER)
            text.append(self._validation, style=TEXT)
        text.append("\n")

        visible = self._tasks[: self.MAX_ROWS]
        for task in visible:
            glyph, colour = _TASK_GLYPHS[task.status]
            marker = "▸" if task.id == self._current else " "
            text.append(f"{marker} {glyph} ", style=f"bold {colour}")
            text.append(task.title, style=TEXT if task.id == self._current else MUTED)
            if task.status is TaskStatus.IN_PROGRESS and task.attempts > 1:
                text.append(f"  (attempt {task.attempts})", style=MUTED)
            text.append("\n")
        if len(self._tasks) > self.MAX_ROWS:
            text.append(f"  +{len(self._tasks) - self.MAX_ROWS} more task(s)\n", style=MUTED)

        extras: list[str] = []
        if self._debug:
            extras.append(self._debug)
        if self._changes:
            extras.append(self._changes)
        if extras:
            text.append("  " + "   ·   ".join(extras), style=MUTED)
        return text


class RunActivity(Static):
    """One ``[VALIDATION]`` / ``[DEBUG]`` / ``[PLAN]`` transcript line."""

    COLOURS = {
        "validation": SUCCESS,
        "debug": WARNING,
        "plan": ACCENT,
        "error": ERROR,
    }

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(classes="run-activity")
        self._kind = kind
        self._message = message

    def set_message(self, kind: str, message: str) -> None:
        self._kind = kind
        self._message = message
        self.refresh()

    def render(self) -> Text:
        colour = self.COLOURS.get(self._kind, MUTED)
        text = Text()
        text.append(f"[{self._kind.upper()}] ", style=f"bold {colour}")
        text.append(self._message, style=MUTED)
        return text


class RunSummaryView(Static):
    """The final autonomous-run report."""

    def __init__(self, summary: "RunSummary") -> None:
        super().__init__(Text(summary.render(), style=TEXT), classes="run-summary")


class CommandOutput(Static):
    """Multi-line output from a slash command."""

    def __init__(self, output: str) -> None:
        super().__init__(Text(output, style=TEXT), classes="command-output")
