# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.pace.version."""

from importlib.metadata import version

import precisionai.pace
from precisionai.pace.version import _DISTRIBUTION, _UNKNOWN_VERSION, get_version


def test_get_version_matches_installed_distribution() -> None:
    assert get_version() == version(_DISTRIBUTION)


def test_get_version_unknown_distribution_falls_back() -> None:
    assert get_version("precisionai-pace-does-not-exist") == _UNKNOWN_VERSION


def test_package_exposes_version() -> None:
    assert precisionai.pace.__version__ == get_version()
