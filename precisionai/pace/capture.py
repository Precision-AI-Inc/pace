# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Route standard ``logging`` records through the Pace logger.

Opt-in: nothing here runs until :func:`capture_logging` is called, so importing Pace
never changes how another library logs.
"""

import logging
from collections.abc import Iterable
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


def _is_console_handler(handler: logging.Handler) -> bool:
    """Return whether ``handler`` writes to a console stream, as opposed to a file or a service."""
    return isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)


def _take_over(name: str) -> None:
    """Stop ``name``, its parents and its existing children printing on their own, and let them propagate.

    A record passes through every parent's handlers on its way to the root, and stops at the
    first one that does not propagate, so the parents are part of the job.
    """
    parts = name.split(".")
    family = [logging.getLogger(".".join(parts[:count])) for count in range(1, len(parts) + 1)]
    prefix = f"{name}."
    for key, item in list(logging.root.manager.loggerDict.items()):
        if key.startswith(prefix) and isinstance(item, logging.Logger):
            family.append(item)
    for target in family:
        for handler in [h for h in target.handlers if _is_console_handler(h)]:
            target.removeHandler(handler)
        target.propagate = True


def capture_logging(
    level: Any = "INFO",
    target: Logger | None = None,
    *,
    take_over: str | Iterable[str] = (),
) -> PaceHandler:
    """Install a handler on the root logger that routes standard ``logging`` through Pace.

    The root logger's own level is lowered to ``level`` when it is higher, because records
    below the root level never reach a handler. Calling this again does not add a second
    handler; it updates the level of the one already installed.

    Some libraries print their own log lines: they attach a console handler to their logger
    and often set ``propagate = False``, so their records either appear twice (once in their
    format, once in Pace's) or never reach Pace. Name them in ``take_over`` to switch that off.

    Parameters
    ----------
    level : str or int
        Lowest level to route, a name such as ``"INFO"`` or a ``logging`` number.
    target : Logger, optional
        Pace logger to write through; the module-level logger by default.
    take_over : str or iterable of str
        Names of loggers whose own console output to remove, such as ``"lightning"``.
        For each name, its parents (``lightning`` for ``lightning.pytorch``) and every
        logger beneath it that exists now (``lightning.pytorch``, ``lightning.fabric``), console handlers (a ``StreamHandler`` that is not a
        ``FileHandler``) are removed and ``propagate`` is set to true, so the records reach
        the root logger and so Pace. Handlers that write to files or to services are kept.
        Call this after importing the library, because most create their loggers on import,
        and again after importing a library that creates its loggers later. Removed handlers
        are not restored.

    Returns
    -------
    PaceHandler
        The installed handler. Remove it with ``logging.getLogger().removeHandler(handler)``.

    Raises
    ------
    ValueError
        If ``level`` is not a valid level, or a name in ``take_over`` is empty. Nothing is
        changed in either case.
    """
    number = parse_level(level)
    if number is None:
        raise ValueError(f"unknown level {level!r}")
    names = [take_over] if isinstance(take_over, str) else list(take_over)
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError(f"take_over needs logger names such as 'lightning', got {take_over!r}")
    root = logging.getLogger()
    handler = next((h for h in root.handlers if isinstance(h, PaceHandler)), None)
    if handler is None:
        handler = PaceHandler(target or default_logger, number)
        root.addHandler(handler)
    handler.setLevel(number)
    if root.level > number:
        root.setLevel(number)
    for name in names:
        _take_over(name)
    return handler
