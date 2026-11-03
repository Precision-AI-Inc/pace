"""Run every example script and check its output with the adopt-pace skill's verify.py.

The two framework examples need PyTorch Lightning or Transformers and are covered by the
adapter tests. Everything else runs here as a subprocess, so an example that stops
working, or stops following the practice the docs describe, fails the suite.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterator

import pytest

from precisionai.pace import ForecastState

pytestmark = pytest.mark.examples

REPOSITORY = Path(__file__).resolve().parent.parent
EXAMPLES = REPOSITORY / "examples"
VERIFY_PATH = REPOSITORY / "skills" / "adopt-pace" / "verify.py"
FRAMEWORK_EXAMPLES = {"lightning_example.py", "transformers_example.py"}
TIMEOUT_SECONDS = 60
Results = Dict[str, "subprocess.CompletedProcess[str]"]


@dataclass(frozen=True)
class Expect:
    """What verify.py should find in one example's output."""

    progress: bool = True
    roots: int = 1
    plan: bool = False
    warnings: tuple = ()


EXPECTED: dict[str, Expect] = {
    "basic_logging.py": Expect(progress=False),
    "track_loop.py": Expect(),
    "epochs_and_batches.py": Expect(),
    "manual_progress.py": Expect(),
    "phases_and_plan.py": Expect(roots=0, plan=True),
    "resume.py": Expect(),
    "unknown_total.py": Expect(),
    "timed_steps.py": Expect(),
    "nested_scopes.py": Expect(),
    "stdlib_logging.py": Expect(),
    "distributed.py": Expect(),
    "reading_output.py": Expect(progress=False),
}


def load_verify() -> ModuleType:
    spec = importlib.util.spec_from_file_location("adopt_pace_verify_for_examples", VERIFY_PATH)
    assert spec
    assert isinstance(spec.loader, SourceFileLoader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verify = load_verify()


def example_environment() -> dict[str, str]:
    """Return a clean environment: every Pace setting and rank variable removed, then the ones the run needs."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("PACE_") and key not in {"RANK", "LOCAL_RANK", "SLURM_PROCID", "NO_COLOR"}
    }
    env.update({"PACE_MIN_INTERVAL": "0", "PACE_BAR": "never", "PYTHONIOENCODING": "utf-8"})
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPOSITORY), env.get("PYTHONPATH", "")]))
    return env


def run_example(name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXAMPLES / name)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=example_environment(),
        timeout=TIMEOUT_SECONDS,
        check=False,
    )


@pytest.fixture(scope="module")
def results() -> Iterator[Results]:
    """Run all the examples at once, so the file costs the slowest example rather than their sum."""
    with ThreadPoolExecutor(max_workers=len(EXPECTED)) as pool:
        futures = {name: pool.submit(run_example, name) for name in EXPECTED}
        yield {name: future.result() for name, future in futures.items()}


def test_every_example_is_listed() -> None:
    on_disk = {path.name for path in EXAMPLES.glob("*.py")}
    assert on_disk == set(EXPECTED) | FRAMEWORK_EXAMPLES


def test_every_example_is_in_the_examples_readme() -> None:
    table = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    assert [name for name in sorted(EXPECTED) + sorted(FRAMEWORK_EXAMPLES) if f"`{name}`" not in table] == []


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_example_exits_cleanly(name: str, results: Results) -> None:
    assert results[name].returncode == 0, results[name].stdout + results[name].stderr
    assert results[name].stderr == ""


@pytest.mark.parametrize("name", sorted(n for n, expect in EXPECTED.items() if expect.progress))
def test_progress_output_passes_verify(name: str, results: Results) -> None:
    expect = EXPECTED[name]
    report = verify.analyze(results[name].stdout.splitlines())
    text = verify.format_report(report)

    assert report.records > 0, text
    assert report.errors == [], text
    assert report.malformed == [], text
    assert [finding.message for finding in report.warnings] == list(expect.warnings), text
    assert len(report.roots) == expect.roots, text
    assert bool(report.plans) is expect.plan, text
    assert report.estimate is not None
    assert report.estimate.state is ForecastState.COMPLETE, text
    assert all(scope.total is not None for scope in report.scopes.values()), text


def test_basic_logging_writes_every_level_and_no_progress(results: Results) -> None:
    out = results["basic_logging.py"].stdout

    for level in ("INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"):
        assert f"Z {level}" in out
    assert "Z DEBUG    Visible now" in out
    assert "Hidden: the default level" not in out
    assert "Hidden again" not in out
    assert "Only warnings and above" in out
    assert "Back to the defaults" in out
    assert "KeyError: 'lr'" in out
    assert "[pace:" not in out


def test_resume_starts_above_zero_and_says_so(results: Results) -> None:
    report = verify.analyze(results["resume.py"].stdout.splitlines())
    start = next(line for line in results["resume.py"].stdout.splitlines() if "k=start" in line)

    assert report.scopes["train"].resumed
    assert "c=70" in start
    assert "res=1" in start
    assert "t=100" in start


def test_unknown_total_starts_unknown_and_learns_the_total(results: Results) -> None:
    lines = [line for line in results["unknown_total.py"].stdout.splitlines() if "[pace:" in line]

    assert "t=?" in lines[0]
    assert "t=60" in lines[-1]


def test_nested_scopes_name_the_epochs_as_parent(results: Results) -> None:
    report = verify.analyze(results["nested_scopes.py"].stdout.splitlines())

    children = {name: scope.parent for name, scope in report.scopes.items() if name != "epochs"}
    assert children
    assert set(children.values()) == {"epochs"}
    assert report.scopes["epochs"].parent is None


def test_phases_and_plan_nest_the_loop_in_its_phase(results: Results) -> None:
    report = verify.analyze(results["phases_and_plan.py"].stdout.splitlines())

    assert [name for name, _ in report.plans[-1]] == ["preprocess", "train", "evaluate"]
    assert report.scopes["train-steps"].parent == "train"


def test_timed_steps_are_nested_in_the_root_loop(results: Results) -> None:
    report = verify.analyze(results["timed_steps.py"].stdout.splitlines())

    assert report.scopes["load-model"].parent is None
    saves = [scope for name, scope in report.scopes.items() if name.startswith("save_results")]
    assert len(saves) == 3
    assert {scope.parent for scope in saves} == {"train"}


def test_stdlib_logging_is_routed_through_pace(results: Results) -> None:
    out = results["stdlib_logging.py"].stdout

    assert "Z INFO     Found 6 shards" in out
    assert "Z WARNING  Slow disk: 41.7 MB/s" in out
    assert "Z ERROR    Shard 3 is corrupt, skipping it" in out
    assert "OSError: checksum mismatch" in out
    assert "UserWarning: tf32 is disabled" in out
    assert "Not routed" not in out


def test_distributed_writes_progress_from_rank_zero_only(results: Results) -> None:
    lines = results["distributed.py"].stdout.splitlines()

    progress = [line for line in lines if " PROGRESS " in line]
    assert progress
    assert all(line.startswith("[rank 0] ") for line in progress)
    assert any(line.startswith("[rank 1] ") and "Rank 1 ready" in line for line in lines)
    assert not any(line.startswith("[rank 1] ") and " PROGRESS " in line for line in lines)


def test_reading_output_reports_the_job_as_complete(results: Results) -> None:
    lines = results["reading_output.py"].stdout.splitlines()

    assert "[pace:" not in results["reading_output.py"].stdout
    assert lines[0].startswith("inference start")
    assert any("complete" in line for line in lines)
    assert lines[-1] == "child exited with 0"
