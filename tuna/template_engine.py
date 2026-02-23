"""Simple template engine — reads a file, replaces {key} placeholders."""

from __future__ import annotations

import re
from pathlib import Path

_SENTINEL_L = "\x00LBRACE\x00"
_SENTINEL_R = "\x00RBRACE\x00"


def _single_pass_replace(content: str, replacements: dict[str, str]) -> str:
    """Replace {key} placeholders in a single pass (prevents injection)."""
    content = content.replace("{{", _SENTINEL_L)
    content = content.replace("}}", _SENTINEL_R)

    def _replacer(m: re.Match) -> str:
        key = m.group(1)
        return str(replacements[key]) if key in replacements else m.group(0)

    content = re.sub(r"\{(\w+)\}", _replacer, content)

    content = content.replace(_SENTINEL_L, "{")
    content = content.replace(_SENTINEL_R, "}")
    return content


def render_template(template_path: str, replacements: dict[str, str]) -> str:
    """Read template file, replace {key} placeholders, return rendered string.

    Uses single-brace {key} syntax. To keep a literal brace in the output
    (e.g. Python dicts in .py.tpl files), templates use ``{{`` / ``}}``
    which are first escaped, then restored after replacement.
    """
    content = Path(template_path).read_text()
    return _single_pass_replace(content, replacements)


def render_string(template: str, replacements: dict[str, str]) -> str:
    """Same as render_template but operates on a string directly."""
    return _single_pass_replace(template, replacements)
