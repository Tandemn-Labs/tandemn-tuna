"""Tests for tandemn.template_engine."""

import tempfile
from pathlib import Path

from tandemn.template_engine import render_string, render_template


class TestRenderString:
    def test_basic_replacement(self):
        result = render_string("hello {name}", {"name": "world"})
        assert result == "hello world"

    def test_multiple_replacements(self):
        result = render_string("{a} and {b}", {"a": "X", "b": "Y"})
        assert result == "X and Y"

    def test_preserves_double_braces(self):
        """Double braces become single braces (for Python dict literals in templates)."""
        result = render_string("d = {{{key}: 1}}", {"key": '"x"'})
        assert result == 'd = {"x": 1}'

    def test_no_replacement_needed(self):
        result = render_string("no placeholders here", {})
        assert result == "no placeholders here"

    def test_unreplaced_placeholder_left_as_is(self):
        result = render_string("{a} and {b}", {"a": "X"})
        assert result == "X and {b}"

    def test_numeric_values(self):
        result = render_string("port={port}", {"port": "8000"})
        assert result == "port=8000"

    def test_multiline(self):
        template = "line1={a}\nline2={b}"
        result = render_string(template, {"a": "1", "b": "2"})
        assert result == "line1=1\nline2=2"


class TestRenderTemplate:
    def test_reads_file_and_replaces(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("model={model} gpu={gpu}")
            f.flush()
            path = f.name

        try:
            result = render_template(path, {"model": "llama", "gpu": "H100"})
            assert result == "model=llama gpu=H100"
        finally:
            Path(path).unlink()

    def test_double_braces_in_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py.tpl", delete=False
        ) as f:
            f.write('env({{"KEY": "{value}"}})')
            f.flush()
            path = f.name

        try:
            result = render_template(path, {"value": "hello"})
            assert result == 'env({"KEY": "hello"})'
        finally:
            Path(path).unlink()
