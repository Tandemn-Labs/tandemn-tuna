"""Simple template engine — reads a file, replaces {key} placeholders."""

from __future__ import annotations

from pathlib import Path


def render_template(template_path: str, replacements: dict[str, str]) -> str:
    """Read template file, replace {key} placeholders, return rendered string.

    Uses single-brace {key} syntax. To keep a literal brace in the output
    (e.g. Python dicts in .py.tpl files), templates use ``{{`` / ``}}``
    which are first escaped, then restored after replacement.
    """
    content = Path(template_path).read_text()

    # Protect literal {{ and }} (used in Python templates for dicts)
    content = content.replace("{{", "\x00LBRACE\x00")
    content = content.replace("}}", "\x00RBRACE\x00")

    for key, value in replacements.items():
        content = content.replace(f"{{{key}}}", str(value))

    # Restore literal braces
    content = content.replace("\x00LBRACE\x00", "{")
    content = content.replace("\x00RBRACE\x00", "}")

    return content


def render_string(template: str, replacements: dict[str, str]) -> str:
    """Same as render_template but operates on a string directly."""
    content = template
    content = content.replace("{{", "\x00LBRACE\x00")
    content = content.replace("}}", "\x00RBRACE\x00")

    for key, value in replacements.items():
        content = content.replace(f"{{{key}}}", str(value))

    content = content.replace("\x00LBRACE\x00", "{")
    content = content.replace("\x00RBRACE\x00", "}")

    return content
