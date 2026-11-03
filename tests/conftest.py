# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for the Pace test suite."""

from __future__ import annotations

from typing import Iterator

import pytest

from precisionai.pace._env import Clock
from precisionai.pace.core import Logger, logger

CONTROLLED_VARIABLES = (
    "PACE_LEVEL",
    "PACE_STREAM",
    "PACE_COLOR",
    "PACE_BAR",
    "PACE_MIN_INTERVAL",
    "PACE_HEARTBEAT",
    "PACE_ALL_RANKS",
    "NO_COLOR",
    "RANK",
    "LOCAL_RANK",
    "SLURM_PROCID",
)

WALL_START = 1_762_190_202.0


class FakeClock:
    """A clock tests move by hand, so no test ever sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.wall = WALL_START

    def advance(self, seconds: float) -> None:
        self.now += seconds
        self.wall += seconds

    def clock(self) -> Clock:
        return Clock(monotonic=lambda: self.now, wall=lambda: self.wall)


class FakeStream:
    """A text stream that reports an encoding and whether it is a terminal."""

    def __init__(self, tty: bool = False, encoding: str = "utf-8") -> None:
        self.tty = tty
        self.encoding = encoding
        self.flushes = 0
        self._parts: list[str] = []

    def write(self, text: str) -> int:
        self._parts.append(text)
        return len(text)

    def flush(self) -> None:
        self.flushes += 1

    def isatty(self) -> bool:
        return self.tty

    def getvalue(self) -> str:
        return "".join(self._parts)

    def lines(self) -> list[str]:
        return self.getvalue().splitlines()


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in CONTROLLED_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    yield
    logger.reset()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def stream() -> FakeStream:
    return FakeStream()


@pytest.fixture
def log(clock: FakeClock, stream: FakeStream) -> Logger:
    out = Logger(clock=clock.clock())
    out.configure(stream=stream)
    return out
