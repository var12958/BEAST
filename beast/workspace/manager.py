"""The project directory the user launched BEAST from.

The workspace is *not* BEAST's own source tree.  BEAST is installed globally and
can be launched from any project; the directory that was current at launch time
is captured here.

Extension point: future file-inspection tools (tree walking, reading files,
writing patches) should be implemented as modules that receive a
:class:`Workspace` and use :meth:`Workspace.resolve` to keep every path inside
the project root.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from beast.errors import WorkspaceError


@dataclass(frozen=True, slots=True)
class Workspace:
    """An absolute, validated project directory."""

    root: Path

    def __post_init__(self) -> None:
        try:
            resolved = Path(self.root).expanduser().resolve()
        except OSError as exc:
            raise WorkspaceError(
                f"The workspace path '{self.root}' could not be resolved.",
                hint="Check the path and try again.",
                detail=repr(exc),
            ) from exc

        if not resolved.is_dir():
            raise WorkspaceError(
                f"The workspace '{resolved}' is not an existing directory.",
                hint="Launch BEAST from your project folder, or pass --workspace <path>.",
            )

        object.__setattr__(self, "root", resolved)

    @classmethod
    def from_cwd(cls) -> Workspace:
        """Capture the current working directory as the active workspace."""
        try:
            cwd = Path.cwd()
        except OSError as exc:
            raise WorkspaceError(
                "Could not determine the current directory.",
                hint="The directory you launched from may have been deleted or be inaccessible.",
                detail=repr(exc),
            ) from exc
        return cls(cwd)

    @property
    def name(self) -> str:
        """A short display name for the project folder."""
        return self.root.name or self.root.drive or str(self.root)

    def display(self) -> str:
        """Human-friendly path, with the home directory shortened to ``~``."""
        try:
            home = Path.home().resolve()
        except OSError:  # pragma: no cover - exotic environments
            return str(self.root)
        try:
            relative = self.root.relative_to(home)
        except ValueError:
            return str(self.root)
        return f"~/{relative.as_posix()}" if relative.parts else "~"

    def resolve(self, relative: str | Path) -> Path:
        """Resolve ``relative`` inside the workspace, refusing to escape it.

        This is the single primitive future file tools should use to keep paths
        inside the project the user launched BEAST from.  It is a path check,
        not a sandbox: a caller that opens the result should use no-follow
        semantics where the platform supports them.
        """
        candidate = (self.root / relative).expanduser().resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise WorkspaceError(
                f"'{relative}' resolves outside the workspace.",
                hint="BEAST only touches files inside the active project directory.",
            )
        return candidate

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.display()
