"""Logging setup.

BEAST writes a detailed log to disk on every run; the terminal only shows
errors.  ``--debug`` (or ``BEAST_DEBUG=1``) additionally mirrors the log to
stderr and lets tracebacks surface.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

LOGGER_NAME = "beast"
_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

_log_path: Path | None = None
_configured = False
_stream_enabled = False


def log_directory() -> Path:
    """Where BEAST keeps its log files."""
    override = os.environ.get("BEAST_LOG_DIR")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "BEAST" / "logs"
    state_home = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(state_home) / "beast" / "logs"


def setup_logging(debug: bool = False) -> Path | None:
    """Configure package logging once and return the log file path.

    Returns ``None`` when the log directory cannot be created (for example on
    a read-only filesystem); BEAST keeps working, it just cannot log.
    """
    global _log_path, _configured, _stream_enabled

    logger = logging.getLogger(LOGGER_NAME)
    if _configured:
        if debug and not _stream_enabled:
            _attach_stream(logger)
        return _log_path

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    directory = log_directory()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "beast.log"
        handler: logging.Handler = logging.FileHandler(path, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        _log_path = path
    except OSError:
        # Logging is a convenience, never a reason to fail the launch.
        path = None

    # Third-party HTTP noise is not useful in the log.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True
    if debug:
        _attach_stream(logger)

    logger.debug("logging initialised (file=%s)", _log_path)
    return _log_path


def _attach_stream(logger: logging.Logger) -> None:
    global _stream_enabled
    stream = logging.StreamHandler(stream=sys.stderr)
    stream.setLevel(logging.DEBUG)
    stream.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(stream)
    _stream_enabled = True


def get_log_path() -> Path | None:
    """The active log file path, if logging was configured successfully."""
    return _log_path
