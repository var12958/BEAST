"""Launch orchestration.

Responsibilities, in order:

1. Resolve the workspace (the directory BEAST was launched from).
2. Verify that Ollama is installed and serving.
3. Discover every local model and split out what BEAST can use.
4. Hand off to the Textual application.

Every failure becomes a :class:`BeastError` with a human readable message and
an actionable hint — never a raw traceback.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from beast import __version__
from beast.errors import (
    BeastError,
    ModelNotFoundError,
    NoModelsError,
    UnsupportedModelError,
)
from beast.intelligence import ProjectIntelligence
from beast.ollama.client import OllamaClient
from beast.ollama.connection import OllamaStatus, ensure_ready, probe
from beast.ollama.models import ModelInventory, OllamaModel, split_models
from beast.ui.app import BeastApp
from beast.ui.branding import ACCENT, ERROR, MUTED, SUCCESS, TEXT, WARNING
from beast.utils.logger import get_log_path, setup_logging
from beast.workspace import Workspace

logger = logging.getLogger(__name__)

LIST_MODELS_HINT = (
    "Run `ASAP launch beast` from a real terminal (for example the VS Code "
    "integrated terminal), or use `ASAP launch beast --list-models` to check "
    "your setup without the interface."
)


@dataclass(frozen=True, slots=True)
class Discovery:
    """Result of asking Ollama what it has available."""

    status: OllamaStatus
    inventory: ModelInventory


async def discover(base_url: str | None = None) -> Discovery:
    """Probe Ollama and load the installed model list."""
    status = await probe(base_url)
    if not status.is_running:
        return Discovery(status=status, inventory=ModelInventory())

    async with OllamaClient(status.base_url) as client:
        models = await client.list_models()

    logger.info(
        "discovered %d models at %s", len(models), status.base_url
    )
    return Discovery(status=status, inventory=split_models(models))


def launch(
    *,
    workspace_path: Path | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
    debug: bool = False,
    list_models: bool = False,
    project_summary: bool = False,
    auto: str | None = None,
) -> int:
    """Start BEAST. Raises :class:`BeastError` for expected failures."""
    console = Console()

    workspace = (
        Workspace(workspace_path) if workspace_path is not None else Workspace.from_cwd()
    )
    logger.info(
        "launch workspace=%s requested_model=%s", workspace.root, model_name or "(picker)"
    )

    if project_summary:
        print_project(console, workspace)
        return 0

    discovery = asyncio.run(discover(base_url))
    ensure_ready(discovery.status)

    if list_models:
        print_models(console, discovery, workspace)
        return 0

    inventory = discovery.inventory
    if not inventory.usable:
        raise NoModelsError(
            "No chat-capable models are available.",
            hint=(
                "Add one with Ollama, then launch BEAST again:\n"
                "  ollama pull <model>\n"
                "Browse local and cloud models at https://ollama.com/library."
            ),
            detail=inventory.hidden_note,
        )

    selected = _resolve_model(inventory, model_name) if model_name else None
    _require_interactive_terminal()

    app = BeastApp(
        workspace=workspace,
        inventory=inventory,
        status=discovery.status,
        initial_model=selected,
        initial_auto=auto,
    )
    app.run()
    logger.info("session ended")
    return 0


def _resolve_model(inventory: ModelInventory, requested: str) -> OllamaModel:
    """Map a ``--model`` value to an available model, or explain why not."""
    for model in inventory.usable:
        if requested in {model.identifier, model.name}:
            return model

    for model in inventory.non_chat:
        if requested in {model.identifier, model.name}:
            raise UnsupportedModelError(
                f"'{requested}' is an embedding model, not a chat model.",
                hint="Choose a chat model, or install one with `ollama pull <model>`.",
            )

    raise ModelNotFoundError(requested, inventory.names)


def _require_interactive_terminal() -> None:
    if sys.stdin.isatty() and sys.stdout.isatty():
        return
    raise BeastError(
        "BEAST's terminal interface needs an interactive terminal.",
        hint=LIST_MODELS_HINT,
    )


def print_project(console: Console, workspace: Workspace) -> None:
    """Non-interactive project overview used by ``--project-summary``."""
    intelligence = ProjectIntelligence(workspace.root)
    try:
        overview = intelligence.overview()
    finally:
        intelligence.close()

    console.print(
        Text.assemble(
            ("BEAST ", f"bold {ACCENT}"),
            (f"v{__version__}", MUTED),
            ("   ·   ", MUTED),
            ("Project analysis", f"bold {TEXT}"),
        )
    )
    console.print(Text.assemble(("Workspace  ", MUTED), (workspace.display(), TEXT)))
    console.print()
    console.print(Text(overview, style=TEXT))


def print_models(console: Console, discovery: Discovery, workspace: Workspace) -> None:
    """Non-interactive summary used by ``--list-models`` and tests."""
    status = discovery.status
    console.print(
        Text.assemble(
            ("BEAST ", f"bold {ACCENT}"),
            (f"v{__version__}", MUTED),
            ("   ·   ", MUTED),
            (f"Ollama {status.version_label}", MUTED),
            (" at ", MUTED),
            (status.base_url, MUTED),
        )
    )
    console.print(
        Text.assemble(("Workspace  ", MUTED), (workspace.display(), TEXT))
    )
    console.print()

    inventory = discovery.inventory
    if inventory.total == 0:
        console.print(
            Text(
                "No models are installed yet. Add one with `ollama pull <model>`.",
                style=WARNING,
            )
        )
        return

    table = Table(box=None, pad_edge=False, header_style=f"bold {MUTED}")
    table.add_column("MODEL", style=TEXT)
    table.add_column("SIZE", style=MUTED, justify="right")
    table.add_column("QUANT", style=MUTED)
    table.add_column("FAMILY", style=MUTED)
    table.add_column("TOOLS", style=MUTED)
    table.add_column("STATUS")

    cloud = {model.identifier for model in inventory.cloud}
    for model in (*inventory.selectable, *inventory.cloud, *inventory.non_chat):
        if model.identifier in cloud:
            status_text = Text("cloud · ready", style=f"bold {ACCENT}")
            row_style = None
        elif model.is_chat_capable:
            status_text = Text("local · ready", style=f"bold {SUCCESS}")
            row_style = None
        else:
            status_text = Text("embedding — hidden", style=MUTED)
            row_style = MUTED
        table.add_row(
            model.name,
            model.size_display,
            model.quantization_display,
            model.family_display,
            model.tools_display,
            status_text,
            style=row_style,
        )

    console.print(table)
    local = len(inventory.selectable)
    remote = len(inventory.cloud)
    parts = [f"{local} local"]
    if remote:
        parts.append(f"{remote} cloud")
    noun = "model" if local + remote == 1 else "models"
    console.print()
    console.print(
        Text(f"{' + '.join(parts)} chat {noun} ready for BEAST.", style=MUTED)
    )
    if inventory.hidden_note:
        console.print(Text(inventory.hidden_note, style=MUTED))


def render_error(console: Console, error: BeastError) -> None:
    """Print a friendly error panel instead of a traceback."""
    body = Text()
    body.append(error.message, style="bold")
    if error.hint:
        body.append("\n\n")
        body.append(error.hint, style="dim")

    console.print()
    console.print(
        Panel(
            body,
            title=Text(error.title, style=f"bold {ERROR}"),
            border_style=ERROR,
            padding=(1, 2),
            expand=False,
        )
    )
    path = get_log_path()
    if path is not None:
        console.print(Text(f"Logs: {path}", style="dim"))


def _ensure_utf8_streams() -> None:
    """Switch Windows' legacy console code page to UTF-8.

    BEAST draws box art and status glyphs; on a cp1252 console they turn into
    question marks.  Terminals on Windows 10/11 support UTF-8, so upgrading the
    stream is safe and lossless (errors fall back to replacement characters).
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", "") or "").lower()
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None or encoding in {"utf-8", "utf8"}:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):  # pragma: no cover - exotic streams
            logger.debug("could not switch stream to UTF-8", exc_info=True)


def run(
    *,
    workspace_path: Path | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
    debug: bool = False,
    list_models: bool = False,
    project_summary: bool = False,
    auto: str | None = None,
) -> int:
    """CLI-facing wrapper: never raises for expected problems."""
    debug = debug or os.environ.get("BEAST_DEBUG") == "1"
    setup_logging(debug)
    _ensure_utf8_streams()
    console = Console()

    try:
        return launch(
            workspace_path=workspace_path,
            model_name=model_name,
            base_url=base_url,
            debug=debug,
            list_models=list_models,
            project_summary=project_summary,
            auto=auto,
        )
    except BeastError as error:
        if error.detail:
            logger.debug("failure detail: %s", error.detail)
        logger.error("launch failed: %s", error, exc_info=debug)
        render_error(console, error)
        return error.exit_code
    except KeyboardInterrupt:
        console.print(Text("\nInterrupted.", style=MUTED))
        return 130
    except Exception:  # noqa: BLE001 - last line of defence for the CLI
        logger.exception("unexpected failure during launch")
        if debug:
            raise
        console.print()
        console.print(
            Panel(
                Text(
                    "BEAST hit an unexpected error and had to stop.\n\n"
                    "Run again with --debug to see the full traceback.",
                    style=TEXT,
                ),
                title=Text("Unexpected error", style=f"bold {ERROR}"),
                border_style=ERROR,
                padding=(1, 2),
                expand=False,
            )
        )
        path = get_log_path()
        if path is not None:
            console.print(Text(f"Logs: {path}", style="dim"))
        return 1
