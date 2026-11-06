# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for what the framework adapters share, which need no framework installed."""

import importlib.machinery
import importlib.util
import sys
import types
from typing import Any

import pytest

from precisionai.pace import Event, Record, parse_line
from precisionai.pace import lightning as pace_lightning
from precisionai.pace import transformers as pace_transformers
from precisionai.pace._adapters import AdapterScopes
from precisionai.pace.core import Logger
from tests.conftest import FakeStream


@pytest.fixture
def quiet(log: Logger) -> Logger:
    log.configure(min_interval=0)
    return log


def records(stream: FakeStream) -> list[Record]:
    return [record for record in map(parse_line, stream.lines()) if record is not None]


def load_fresh(module: Any, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, module.__file__)
    assert spec is not None
    assert isinstance(spec.loader, importlib.machinery.SourceFileLoader)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    return fresh


def test_main_scope_is_the_root_and_its_children_name_it_as_parent(quiet: Logger, stream: FakeStream) -> None:
    scopes = AdapterScopes("Train", logger=quiet)
    scopes.begin(10)
    scopes.update(4)
    scopes.begin_child("Validation", 3, "batch")
    scopes.advance_child()
    scopes.end_child()
    scopes.update(10)
    scopes.end()
    found = records(stream)
    assert [(r.scope, r.event) for r in found if r.event in (Event.START, Event.END)] == [
        ("train", Event.START),
        ("validation", Event.START),
        ("validation", Event.END),
        ("train", Event.END),
    ]
    assert {r.parent for r in found if r.scope == "validation"} == {"train"}
    assert {r.root for r in found if r.scope == "train"} == {True}
    assert {r.root for r in found if r.scope == "validation"} == {False}


def test_parent_binding_does_not_leak_into_scopes_opened_afterwards(quiet: Logger, stream: FakeStream) -> None:
    scopes = AdapterScopes("Train", logger=quiet)
    scopes.begin(10)
    scopes.begin_child("Validation", None, "batch")
    scopes.end()
    other = AdapterScopes("Other", logger=quiet, root=False)
    other.begin(1)
    other.end()
    assert {r.parent for r in records(stream) if r.scope == "other"} == {None}


def test_a_child_without_a_main_scope_is_never_the_root(quiet: Logger, stream: FakeStream) -> None:
    scopes = AdapterScopes("Train", logger=quiet)
    scopes.begin_child("Validation", 2, "batch")
    scopes.end_child()
    scopes.begin(5)
    scopes.end()
    found = records(stream)
    assert {r.root for r in found if r.scope == "validation"} == {False}
    assert {r.root for r in found if r.scope == "train"} == {True}
    assert {r.parent for r in found if r.scope == "validation"} == {None}


def test_root_can_be_declined(quiet: Logger, stream: FakeStream) -> None:
    scopes = AdapterScopes("Train", logger=quiet, root=False)
    scopes.begin(5)
    scopes.begin_child("Validation", 2, "batch")
    scopes.end()
    assert not [r for r in records(stream) if r.root]


def test_a_new_child_ends_the_previous_one_and_a_new_main_ends_the_previous_main(
    quiet: Logger, stream: FakeStream
) -> None:
    scopes = AdapterScopes("Train", logger=quiet, scope="one")
    scopes.begin(5)
    scopes.begin_child("Validation", 2, "batch")
    scopes.begin_child("Validation", 2, "batch")
    scopes.begin(5, completed=2, resumed=True)
    scopes.end()
    ends = [r.scope for r in records(stream) if r.event is Event.END]
    assert ends == ["validation", "validation-2", "one", "one-2"]


def test_fail_marks_the_child_and_the_main_scope_and_later_calls_do_nothing(quiet: Logger, stream: FakeStream) -> None:
    scopes = AdapterScopes("Train", logger=quiet)
    scopes.begin(5)
    scopes.begin_child("Validation", 2, "batch")
    scopes.fail()
    scopes.fail()
    scopes.end()
    scopes.update(3)
    scopes.advance_child()
    assert [(r.scope, r.event) for r in records(stream) if r.event in (Event.END, Event.FAIL)] == [
        ("validation", Event.FAIL),
        ("train", Event.FAIL),
    ]
    assert scopes.main is None
    assert scopes.child is None


def test_resume_is_written_on_the_start_record(quiet: Logger, stream: FakeStream) -> None:
    AdapterScopes("Train", logger=quiet).begin(None, completed=7, resumed=True)
    start = records(stream)[0]
    assert start.resumed is True
    assert start.completed == 7
    assert start.total is None


def test_calls_before_a_scope_exists_do_nothing(quiet: Logger, stream: FakeStream) -> None:
    scopes = AdapterScopes("Train", logger=quiet)
    scopes.update(3)
    scopes.advance_child()
    scopes.end_child()
    scopes.end()
    scopes.fail()
    assert stream.getvalue() == ""


@pytest.mark.parametrize("module", [pace_lightning, pace_transformers])
def test_the_adapter_modules_import_without_their_framework(module: Any) -> None:
    assert module.PaceCallback.__name__ == "PaceCallback"


@pytest.mark.parametrize(
    ("module", "flag", "extra"),
    [
        (pace_lightning, "_LIGHTNING_AVAILABLE", "lightning"),
        (pace_transformers, "_TRANSFORMERS_AVAILABLE", "transformers"),
    ],
)
def test_a_missing_framework_is_reported_with_the_extra_to_install(
    monkeypatch: pytest.MonkeyPatch, module: Any, flag: str, extra: str
) -> None:
    monkeypatch.setattr(module, flag, False)
    with pytest.raises(ImportError, match=rf"precisionai-pace\[{extra}\]"):
        module.PaceCallback()


def block(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    for name in names:
        monkeypatch.setitem(sys.modules, name, None)


def test_the_lightning_adapter_falls_back_to_the_older_package_name(monkeypatch: pytest.MonkeyPatch) -> None:
    class Callback:
        pass

    block(monkeypatch, "lightning", "lightning.pytorch", "lightning.pytorch.callbacks")
    older = types.ModuleType("pytorch_lightning")
    callbacks = types.ModuleType("pytorch_lightning.callbacks")
    callbacks.Callback = Callback  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pytorch_lightning", older)
    monkeypatch.setitem(sys.modules, "pytorch_lightning.callbacks", callbacks)
    fresh = load_fresh(pace_lightning, "pace_lightning_older")
    assert fresh._LIGHTNING_AVAILABLE is True
    assert issubclass(fresh.PaceCallback, Callback)


def test_the_lightning_adapter_subclasses_both_packages_when_both_are_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NewCallback:
        pass

    class OldCallback:
        pass

    for package, module, callback in [
        ("lightning", "lightning.pytorch", NewCallback),
        ("pytorch_lightning", None, OldCallback),
    ]:
        monkeypatch.setitem(sys.modules, package, types.ModuleType(package))
        if module is not None:
            monkeypatch.setitem(sys.modules, module, types.ModuleType(module))
        callbacks = types.ModuleType(f"{module or package}.callbacks")
        callbacks.Callback = callback  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, f"{module or package}.callbacks", callbacks)
    sys.modules["pytorch_lightning"].__version__ = "2.6.0"  # type: ignore[attr-defined]
    fresh = load_fresh(pace_lightning, "pace_lightning_both")
    assert fresh._LIGHTNING_AVAILABLE is True
    assert issubclass(fresh.PaceCallback, NewCallback)
    assert issubclass(fresh.PaceCallback, OldCallback)

    # A 1.x pytorch_lightning calls some hooks with other arguments than lightning 2.x's
    # Callback takes, so beside lightning it is not a base.
    sys.modules["pytorch_lightning"].__version__ = "1.9.5"  # type: ignore[attr-defined]
    older = load_fresh(pace_lightning, "pace_lightning_both_older")
    assert issubclass(older.PaceCallback, NewCallback)
    assert not issubclass(older.PaceCallback, OldCallback)


def test_the_adapters_degrade_to_a_plain_class_when_nothing_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    block(
        monkeypatch,
        "lightning",
        "lightning.pytorch",
        "lightning.pytorch.callbacks",
        "pytorch_lightning",
        "pytorch_lightning.callbacks",
        "transformers",
    )
    lightning = load_fresh(pace_lightning, "pace_lightning_absent")
    transformers = load_fresh(pace_transformers, "pace_transformers_absent")
    assert lightning._LIGHTNING_AVAILABLE is False
    assert transformers._TRANSFORMERS_AVAILABLE is False
    with pytest.raises(ImportError, match="lightning is required"):
        lightning.PaceCallback()
    with pytest.raises(ImportError, match="transformers is required"):
        transformers.PaceCallback()
