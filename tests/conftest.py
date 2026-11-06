# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for the Pace test suite."""

import json
import os
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from precisionai.pace.core import Clock, Logger, logger

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
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "SLURM_PROCID",
    "LOCAL_RANK",
)

WALL_START = 1_762_190_202.0
GOLDEN_DIRECTORY = Path(__file__).parent / "fixtures" / "golden"
UPDATE_GOLDEN = "PACE_UPDATE_GOLDEN"


def assert_golden(name: str, text: str) -> None:
    """Compare ``text`` with the stored golden output ``name``, one JSON string per line of output.

    Run with ``PACE_UPDATE_GOLDEN=1`` to write the file instead. Each stored line keeps its own line
    ending, so trailing blanks and carriage returns survive git and the pre-commit hooks.
    """
    path = GOLDEN_DIRECTORY / f"{name}.jsonl"
    chunks = re.findall(r"[^\n]*\n|[^\n]+", text)
    if os.environ.get(UPDATE_GOLDEN):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks), encoding="utf-8")
    assert path.exists(), f"no golden file {path.name}; write it with {UPDATE_GOLDEN}=1"
    with path.open(encoding="utf-8") as handle:
        expected = [json.loads(line) for line in handle]
    assert chunks == expected


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
