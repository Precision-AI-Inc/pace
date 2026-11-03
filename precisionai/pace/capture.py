# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Route standard ``logging`` records through the Pace logger.

Opt-in: nothing here runs until :func:`capture_logging` is called, so importing Pace
never changes how another library logs.
"""

from __future__ import annotations

import logging
from typing import Any

from precisionai.pace._env import level_name, parse_level
from precisionai.pace.core import Logger
from precisionai.pace.core import logger as default_logger


class PaceHandler(logging.Handler):
    """A ``logging`` handler that writes each record as a Pace log line.

    Parameters
    ----------
    target : Logger
        The Pace logger to write through.
    level : int
        Minimum standard ``logging`` level this handler passes on.
    """

    def __init__(self, target: Logger, level: int = logging.INFO) -> None:
        super().__init__(level)
        self.target = target

    def emit(self, record: logging.LogRecord) -> None:
        """Write ``record`` (with its traceback, if any); never raise."""
        try:
            self.target.log(level_name(record.levelno), self.format(record))
        except Exception:
            self.handleError(record)


def capture_logging(level: Any = "INFO", target: Logger | None = None) -> PaceHandler:
    """Install a handler on the root logger that routes standard ``logging`` through Pace.

    The root logger's own level is lowered to ``level`` when it is higher, because records
    below the root level never reach a handler. Calling this again does not add a second
    handler; it updates the level of the one already installed.

    Parameters
    ----------
    level : str or int
        Lowest level to route, a name such as ``"INFO"`` or a ``logging`` number.
    target : Logger, optional
        Pace logger to write through; the module-level logger by default.

    Returns
    -------
    PaceHandler
        The installed handler. Remove it with ``logging.getLogger().removeHandler(handler)``.

    Raises
    ------
    ValueError
        If ``level`` is not a valid level.
    """
    number = parse_level(level)
    if number is None:
        raise ValueError(f"unknown level {level!r}")
    root = logging.getLogger()
    handler = next((h for h in root.handlers if isinstance(h, PaceHandler)), None)
    if handler is None:
        handler = PaceHandler(target or default_logger, number)
        root.addHandler(handler)
    handler.setLevel(number)
    if root.level > number:
        root.setLevel(number)
    return handler
