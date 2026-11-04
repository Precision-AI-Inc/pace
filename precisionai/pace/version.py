# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Installed package version."""

from importlib.metadata import PackageNotFoundError, version

DISTRIBUTION = "precisionai-pace"
UNKNOWN_VERSION = "0.0.0"


def get_version(distribution: str = DISTRIBUTION) -> str:
    """Return the installed version of ``distribution``.

    Parameters
    ----------
    distribution : str
        Distribution name to look up. Defaults to this package.

    Returns
    -------
    str
        The version string, or ``"0.0.0"`` when the distribution is not installed
        (for example, when the source tree is imported without ``pip install``).
    """
    try:
        return version(distribution)
    except PackageNotFoundError:
        return UNKNOWN_VERSION
