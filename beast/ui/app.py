"""The Textual application object.

The app owns everything that outlives a screen: the Ollama client, the model
inventory, the active workspace and the chat screen instance (kept so switching
models does not throw the transcript away).
"""

from __future__ import annotations

import asyncio
import logging

from textual.app import App
from textual.binding import Binding
from textual.theme import Theme

from beast.intelligence import ProjectIntelligence
from beast.ollama.client import OllamaClient
from beast.ollama.connection import OllamaStatus
from beast.ollama.models import ModelInventory, OllamaModel, split_models
from beast.tools import ApprovalRequest
from beast.workspace import Workspace

from .branding import (
    ACCENT,
    BG,
    ERROR,
    PANEL,
    SUCCESS,
    SURFACE,
    TEXT,
    USER,
    WARNING,
)
from .screens import ApprovalScreen, ChatScreen, ModelSelectScreen

logger = logging.getLogger(__name__)

BEAST_THEME = Theme(
    name="beast",
    primary=ACCENT,
    secondary=USER,
    accent=ACCENT,
    warning=WARNING,
    error=ERROR,
    success=SUCCESS,
    foreground=TEXT,
    background=BG,
    surface=SURFACE,
    panel=PANEL,
    dark=True,
)


class BeastApp(App[None]):
    """BEAST's terminal interface."""

    CSS_PATH = "styles.tcss"
    TITLE = "BEAST"
    SUB_TITLE = "local AI coding agent"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [Binding("ctrl+q", "quit", "Quit", priority=True)]

    def __init__(
        self,
        *,
        workspace: Workspace,
        inventory: ModelInventory,
        status: OllamaStatus,
        client: OllamaClient | None = None,
        initial_model: OllamaModel | None = None,
        intelligence: ProjectIntelligence | None = None,
        initial_auto: str | None = None,
    ) -> None:
        super().__init__()
        self.workspace = workspace
        self.inventory = inventory
        self.status = status
        self.client = client or OllamaClient(status.base_url)
        #: One analysis session per app, shared by the tools and the agent.
        self.intelligence = intelligence or ProjectIntelligence(workspace.root)
        self._initial_model = initial_model
        self._initial_auto = initial_auto
        self._chat: ChatScreen | None = None

    # --------------------------------------------------------------- lifecycle

    def on_mount(self) -> None:
        self.register_theme(BEAST_THEME)
        self.theme = BEAST_THEME.name

        if self._initial_model is not None:
            self._chat = self._make_chat(self._initial_model)
            self.push_screen(self._chat)
        else:
            self.push_screen(ModelSelectScreen())

    async def on_unmount(self) -> None:
        try:
            self.intelligence.close()
        except Exception:  # pragma: no cover - shutdown best effort
            logger.debug("failed to close project intelligence", exc_info=True)
        closer = getattr(self.client, "close", None)
        if closer is None:
            return
        try:
            await closer()
        except Exception:  # pragma: no cover - shutdown best effort
            logger.debug("failed to close the Ollama client", exc_info=True)

    # ------------------------------------------------------------------ screens

    @property
    def has_chat_screen(self) -> bool:
        return self._chat is not None

    def _make_chat(self, model: OllamaModel) -> ChatScreen:
        initial_auto, self._initial_auto = self._initial_auto, None
        return ChatScreen(
            model=model,
            workspace=self.workspace,
            status=self.status,
            intelligence=self.intelligence,
            initial_auto=initial_auto,
        )

    def choose_model(self, model: OllamaModel) -> None:
        """Open (or retarget) the chat screen for ``model``."""
        if self._chat is None:
            self._chat = self._make_chat(model)
            self.push_screen(self._chat)
            return

        switched = self._chat.switch_model(model)
        if self.screen is not self._chat:
            self.pop_screen()
        if not switched:
            self.notify(f"{model.identifier} is already the active model.", timeout=2)

    def open_model_picker(self) -> None:
        """Push the model list on top of the current conversation."""
        current = self._chat.current_model if self._chat is not None else None
        self.push_screen(ModelSelectScreen(current_model=current))

    # -------------------------------------------------------------------- data

    async def reload_models(self) -> ModelInventory:
        """Re-query Ollama and return the refreshed inventory."""
        models = await self.client.list_models()
        self.inventory = split_models(models)
        return self.inventory

    # --------------------------------------------------------------- approvals

    async def approve(self, request: ApprovalRequest) -> bool:
        """Ask the user to approve a write or a command.

        Only an explicit approve action returns ``True``.  Dismissing the modal
        any other way, or the worker being stopped while it waits, counts as a
        refusal.
        """
        screen = ApprovalScreen(request)
        try:
            result = await self.push_screen_wait(screen)
        except asyncio.CancelledError:
            if screen in self.screen_stack:
                self.pop_screen()
            raise
        return result is True
