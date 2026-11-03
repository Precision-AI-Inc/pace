# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""The README is also the PyPI project page, where relative links and images do not resolve."""

from __future__ import annotations

import re
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"

_LINK = re.compile(r"\]\(([^)\s]+)\)")
_IMG = re.compile(r'<img[^>]*\bsrc="([^"]+)"')


def _targets(text: str) -> list[str]:
    return _LINK.findall(text) + _IMG.findall(text)


def test_readme_links_and_images_are_absolute() -> None:
    relative = [
        target for target in _targets(README.read_text(encoding="utf-8")) if not re.match(r"https://|#", target)
    ]
    assert relative == []


def test_readme_title_is_the_product_name() -> None:
    assert "\n# Precision AI Pace\n" in README.read_text(encoding="utf-8")
