# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for configuration resolution, levels and rank detection."""

import sys
import time

import pytest

from precisionai.pace._env import Clock, Settings, current_rank, level_name, parse_level


def test_clock_defaults_to_the_real_clocks() -> None:
    clock = Clock()
    assert clock.monotonic is time.monotonic
    assert clock.wall is time.time


@pytest.mark.parametrize(
    ("value", "expected"),
    [("info", 20), (" WARNING ", 30), ("Success", 25), (7, 7), ("loud", None), (True, None), (None, None), (1.5, None)],
)
def test_parse_level(value: object, expected: object) -> None:
    assert parse_level(value) == expected


@pytest.mark.parametrize(
    ("number", "expected"),
    [(0, "DEBUG"), (10, "DEBUG"), (19, "DEBUG"), (20, "INFO"), (25, "SUCCESS"), (35, "WARNING"), (99, "CRITICAL")],
)
def test_level_name_rounds_down(number: int, expected: str) -> None:
    assert level_name(number) == expected


def test_rank_defaults_to_zero() -> None:
    assert current_rank({}) == 0


def test_rank_prefers_the_global_ranks_over_local_rank() -> None:
    environment = {
        "RANK": "7",
        "OMPI_COMM_WORLD_RANK": "6",
        "PMI_RANK": "5",
        "SLURM_PROCID": "4",
        "LOCAL_RANK": "1",
    }
    for name, expected in [("RANK", 7), ("OMPI_COMM_WORLD_RANK", 6), ("PMI_RANK", 5), ("SLURM_PROCID", 4)]:
        assert current_rank(environment) == expected
        del environment[name]
    assert current_rank(environment) == 1


@pytest.mark.parametrize("variable", ["OMPI_COMM_WORLD_RANK", "PMI_RANK"])
def test_rank_reads_mpi_launchers(variable: str) -> None:
    assert current_rank({variable: "3", "LOCAL_RANK": "0"}) == 3


def test_rank_skips_values_that_are_not_integers() -> None:
    assert current_rank({"RANK": "x", "LOCAL_RANK": " 3 "}) == 3


def test_rank_reads_the_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "6")
    assert current_rank() == 6


def test_defaults_when_nothing_is_set() -> None:
    settings = Settings()
    assert settings.resolve("level") == 20
    assert settings.resolve("min_interval") == 15.0
    assert settings.resolve("heartbeat") == 120.0
    assert settings.resolve("color") == "auto"
    assert settings.resolve("bar") == "auto"
    assert settings.resolve("stream") is sys.stdout
    assert settings.progress_enabled()


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PACE_LEVEL", "debug")
    monkeypatch.setenv("PACE_MIN_INTERVAL", "3.5")
    monkeypatch.setenv("PACE_HEARTBEAT", "9")
    monkeypatch.setenv("PACE_COLOR", "ALWAYS")
    monkeypatch.setenv("PACE_BAR", "never")
    monkeypatch.setenv("PACE_STREAM", "Stderr")
    settings = Settings()
    assert settings.resolve("level") == 10
    assert settings.resolve("min_interval") == 3.5
    assert settings.resolve("heartbeat") == 9.0
    assert settings.resolve("color") == "always"
    assert settings.resolve("bar") == "never"
    assert settings.resolve("stream") is sys.stderr


def test_explicit_settings_override_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PACE_LEVEL", "debug")
    monkeypatch.setenv("PACE_MIN_INTERVAL", "3")
    monkeypatch.setenv("PACE_COLOR", "always")
    settings = Settings(level=40, min_interval=0.0, heartbeat=1.0, color="never", stream="stderr")
    assert settings.resolve("level") == 40
    assert settings.resolve("min_interval") == 0.0
    assert settings.resolve("heartbeat") == 1.0
    assert settings.resolve("color") == "never"
    assert settings.resolve("stream") is sys.stderr


@pytest.mark.parametrize("raw", ["", "  ", "soon", "-4", "nan", "inf"])
def test_bad_environment_numbers_fall_back_to_defaults(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("PACE_MIN_INTERVAL", raw)
    monkeypatch.setenv("PACE_HEARTBEAT", raw)
    settings = Settings()
    assert settings.resolve("min_interval") == 15.0
    assert settings.resolve("heartbeat") == 120.0


def test_unknown_color_mode_is_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PACE_COLOR", "rainbow")
    assert Settings().resolve("color") == "auto"


def test_an_object_stream_is_used_as_is() -> None:
    sink = object()
    assert Settings(stream=sink).resolve("stream") is sink


@pytest.mark.parametrize(("raw", "enabled"), [("1", True), ("true", True), ("YES", True), ("0", False), ("", False)])
def test_all_ranks_words(monkeypatch: pytest.MonkeyPatch, raw: str, enabled: bool) -> None:
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("PACE_ALL_RANKS", raw)
    assert Settings().progress_enabled() is enabled
