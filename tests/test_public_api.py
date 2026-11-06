# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end: what the logger writes, the parser reads and the estimator understands."""

import ast
import importlib
import inspect

import pytest

import precisionai.pace as pace
from precisionai.pace import ForecastState, JobForecast, Logger, parse_line, track

from .conftest import FakeClock, FakeStream

PACKAGE_NAMES = [
    "Clock",
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

# The public names of each documented submodule: its ``__all__``, which `docs/modules.rst` documents and the
# compatibility promise in the README covers. They stay importable from these paths however the code behind them
# is organised. Anything else in these modules is an implementation detail and starts with an underscore.
SUBMODULE_NAMES = {
    "core": "Clock LEVELS Logger logger",
    "scopes": "Phase Scope phase plan progress timed track",
    "capture": "PaceHandler capture_logging",
    "protocol": "Event LineSplitter MARKER Record format_trailer parse_line",
    "forecast": "DEFAULT_CONFIG Forecast ForecastConfig ForecastState JobEstimate JobForecast ScopeForecast",
    "lightning": "PaceCallback",
    "transformers": "PaceCallback",
    "tqdm": "PaceTqdm",
    "version": "get_version",
}

# Logger is public, so each of its attributes without a leading underscore is API too.
LOGGER_ATTRIBUTES = "clock configure critical debug error exception info log reset run_id success warning"


def test_public_names_are_exported() -> None:
    assert sorted(pace.__all__) == pace.__all__
    assert pace.__all__ == sorted(PACKAGE_NAMES)
    for name in pace.__all__:
        assert hasattr(pace, name), name


@pytest.mark.parametrize("module", sorted(SUBMODULE_NAMES))
def test_documented_submodules_offer_their_names(module: str) -> None:
    imported = importlib.import_module(f"precisionai.pace.{module}")
    assert sorted(imported.__all__) == sorted(SUBMODULE_NAMES[module].split())
    assert [name for name in imported.__all__ if not hasattr(imported, name)] == []


def _module_level_names(body: list[ast.stmt]) -> tuple[set[str], set[str]]:
    """Return the names a module body defines and imports, looking inside ``if``, ``try`` and ``with`` blocks."""
    defined: set[str] = set()
    imported: set[str] = set()
    for node in body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            defined.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            defined.update(target.id for target in targets if isinstance(target, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imported.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, (ast.If, ast.Try, ast.With)):
            blocks = [getattr(node, field, []) for field in ("body", "orelse", "finalbody")]
            blocks += [handler.body for handler in getattr(node, "handlers", [])]
            for block in blocks:
                inner_defined, inner_imported = _module_level_names(block)
                defined |= inner_defined
                imported |= inner_imported
    return defined, imported


@pytest.mark.parametrize("module", sorted(SUBMODULE_NAMES))
def test_documented_submodules_define_no_other_public_names(module: str) -> None:
    imported_module = importlib.import_module(f"precisionai.pace.{module}")
    defined, imported = _module_level_names(ast.parse(inspect.getsource(imported_module)).body)
    # A stand-in for a missing optional framework (``Callback = object``) shadows an import; it is not a new name.
    public = {name for name in defined - imported if not name.startswith("_")}
    assert sorted(public - set(imported_module.__all__)) == []


def test_logger_offers_only_its_documented_attributes() -> None:
    public = {name for name in dir(Logger()) if not name.startswith("_")}
    assert sorted(public) == sorted(LOGGER_ATTRIBUTES.split())


@pytest.mark.parametrize("module", sorted(SUBMODULE_NAMES))
def test_documented_submodules_are_attributes_of_the_package(module: str) -> None:
    imported = importlib.import_module(f"precisionai.pace.{module}")
    assert getattr(pace, module) is imported


def test_written_progress_parses_back_and_drives_a_live_forecast(
    log: Logger, clock: FakeClock, stream: FakeStream
) -> None:
    def batches():
        for index in range(100):
            clock.advance(2.0)
            yield index

    for _ in track(batches(), description="Train", unit="batch", total=100, root=True, logger=log):
        pass

    records = [parse_line(line) for line in stream.lines()]
    assert all(record is not None for record in records)
    job = JobForecast()
    for record in records[:-1]:
        assert record is not None
        job.observe(record, record.elapsed)
    middle = job.snapshot(records[-2].elapsed)  # type: ignore[union-attr]
    assert middle.state is ForecastState.LIVE
    assert middle.source == "root"

    final = records[-1]
    assert final is not None
    job.observe(final, final.elapsed)
    assert job.snapshot(final.elapsed).state is ForecastState.COMPLETE
    assert records[0] is not None
    assert records[0].run == log.run_id
