"""The ``asap`` command line.

Hierarchy::

    asap
     └── launch
          └── beast

``asap launch beast`` starts BEAST in the directory the command was run from,
so it works from any project opened in the VS Code integrated terminal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from beast import __version__
from beast.main import run
from beast.ollama.connection import BASE_URL_ENV, DEFAULT_BASE_URL

app = typer.Typer(
    name="asap",
    help="Local-first developer tools for the terminal.",
    no_args_is_help=True,
    invoke_without_command=True,
    add_completion=False,
    rich_markup_mode="rich",
)

launch_app = typer.Typer(
    help="Launch a local AI tool.",
    no_args_is_help=True,
    add_completion=False,
)

app.add_typer(launch_app, name="launch")


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            help="Show the BEAST version and exit.",
            is_eager=True,
        ),
    ] = False,
) -> None:
    if version:
        typer.echo(f"BEAST {__version__}")
        raise typer.Exit()


@launch_app.command("beast")
def beast(
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Model to use, skipping the picker. Must be installed locally.",
        ),
    ] = None,
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            "-w",
            help="Project directory to work in. Defaults to the current directory.",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = None,
    ollama_url: Annotated[
        str | None,
        typer.Option(
            "--ollama-url",
            help=(
                f"Ollama endpoint. Defaults to ${BASE_URL_ENV} "
                f"or {DEFAULT_BASE_URL}."
            ),
        ),
    ] = None,
    list_models: Annotated[
        bool,
        typer.Option(
            "--list-models",
            help="List the models Ollama has installed and exit (no interface).",
        ),
    ] = False,
    project_summary: Annotated[
        bool,
        typer.Option(
            "--project-summary",
            help=(
                "Detect languages, frameworks, layout and dependencies in the "
                "workspace, print them, and exit (no interface)."
            ),
        ),
    ] = False,
    auto: Annotated[
        str | None,
        typer.Option(
            "--auto",
            help=(
                "Start in autonomous mode and run this request: the model plans, "
                "executes and validates tasks. Writes and commands still need "
                "approval."
            ),
        ),
    ] = None,
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Show tracebacks and verbose logs."),
    ] = False,
) -> None:
    """Start the BEAST agent in the current project directory."""
    options: dict[str, object] = {
        "workspace_path": workspace,
        "model_name": model,
        "base_url": ollama_url,
        "debug": debug,
        "list_models": list_models,
    }
    if project_summary:
        # Only passed when requested, so the existing call contract is stable.
        options["project_summary"] = True
    if auto:
        options["auto"] = auto
    code = run(**options)  # type: ignore[arg-type]
    raise typer.Exit(code=code)


if __name__ == "__main__":  # pragma: no cover
    app()
