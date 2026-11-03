# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Pace — a logger with progress reporting whose output says when the work will finish.

Each progress record is a readable log line that also carries a short, versioned
trailer, so the same output serves a person watching a terminal and a scheduler
estimating when the job will release its hardware. See ``docs/design.md``.
"""

from precisionai.pace.capture import PaceHandler, capture_logging
from precisionai.pace.core import Logger, logger
from precisionai.pace.forecast import (
    Forecast,
    ForecastConfig,
    ForecastState,
    JobEstimate,
    JobForecast,
    ScopeForecast,
)
from precisionai.pace.protocol import Event, LineSplitter, Record, format_trailer, parse_line
from precisionai.pace.scopes import Phase, Scope, phase, plan, progress, timed, track
from precisionai.pace.version import get_version

__version__ = get_version()

__all__ = [
    "Event",
    "Forecast",
    "ForecastConfig",
    "ForecastState",
    "JobEstimate",
    "JobForecast",
    "LineSplitter",
    "Logger",
    "PaceHandler",
    "Phase",
    "Record",
    "Scope",
    "ScopeForecast",
    "__version__",
    "capture_logging",
    "format_trailer",
    "get_version",
    "logger",
    "parse_line",
    "phase",
    "plan",
    "progress",
    "timed",
    "track",
]
