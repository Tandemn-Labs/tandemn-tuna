"""Unit tests for tuna.ui — banner, helpers, SharkSpinner."""

from __future__ import annotations

import io
import time

from rich.console import Console

from tuna.ui import (
    SharkSpinner,
    TunaLogHandler,
    _detect_unicode_support,
    banner,
    error,
    info_panel,
    section,
    status_msg,
    styled_url,
    success,
    warning,
)


def _capture_console() -> tuple[Console, io.StringIO]:
    """Return a (console, buffer) pair that captures Rich output."""
    buf = io.StringIO()
    con = Console(file=buf, force_terminal=True, width=120)
    return con, buf


class TestBanner:
    def test_banner_contains_tuna(self, monkeypatch):
        con, buf = _capture_console()
        monkeypatch.setattr("tuna.ui.console", con)
        banner()
        output = buf.getvalue()
        assert "Tandemn" in output
        assert "▓" in output  # banner block chars present

    def test_banner_contains_tagline(self, monkeypatch):
        con, buf = _capture_console()
        monkeypatch.setattr("tuna.ui.console", con)
        banner()
        assert "Hybrid GPU Inference Orchestrator" in buf.getvalue()


class TestHelpers:
    def test_section(self, monkeypatch):
        con, buf = _capture_console()
        monkeypatch.setattr("tuna.ui.console", con)
        section("DEPLOY")
        assert "DEPLOY" in buf.getvalue()

    def test_info_panel(self, monkeypatch):
        con, buf = _capture_console()
        monkeypatch.setattr("tuna.ui.console", con)
        info_panel("TEST", {"Key1": "val1", "Key2": "val2"})
        output = buf.getvalue()
        assert "TEST" in output
        assert "Key1" in output
        assert "val1" in output

    def test_status_msg(self, monkeypatch):
        con, buf = _capture_console()
        monkeypatch.setattr("tuna.ui.console", con)
        status_msg("doing something")
        assert "doing something" in buf.getvalue()

    def test_success(self, monkeypatch):
        con, buf = _capture_console()
        monkeypatch.setattr("tuna.ui.console", con)
        success("all good")
        output = buf.getvalue()
        assert "all good" in output
        assert "✓" in output

    def test_error(self, monkeypatch):
        con, buf = _capture_console()
        monkeypatch.setattr("tuna.ui.err_console", con)
        error("bad thing")
        output = buf.getvalue()
        assert "bad thing" in output
        assert "✗" in output

    def test_warning(self, monkeypatch):
        con, buf = _capture_console()
        monkeypatch.setattr("tuna.ui.console", con)
        warning("heads up")
        output = buf.getvalue()
        assert "heads up" in output

    def test_styled_url(self):
        result = styled_url("http://example.com")
        assert "orange3" in result
        assert "http://example.com" in result


class TestSharkSpinner:
    def test_context_manager_start_stop(self):
        """Spinner starts and stops without error."""
        spinner = SharkSpinner("Test", file=io.StringIO())
        # Force enabled even if not a TTY
        spinner._enabled = True
        with spinner:
            time.sleep(0.15)
        assert spinner._stop_event.is_set()

    def test_disabled_when_not_tty(self):
        """Spinner should not start a thread when not a TTY."""
        spinner = SharkSpinner("Test", file=io.StringIO())
        spinner._enabled = False
        spinner.start()
        assert spinner._thread is None
        spinner.stop()

    def test_unicode_frames(self):
        spinner = SharkSpinner("Test")
        if _detect_unicode_support():
            assert "𓂁" in spinner.frames[0]
        else:
            assert ">" in spinner.frames[0]

    def test_ascii_fallback(self):
        assert len(SharkSpinner.FRAMES_ASCII) == len(SharkSpinner.FRAMES_UNICODE)

    def test_exception_in_animation_does_not_propagate(self):
        """If animate raises, the spinner stops gracefully."""
        spinner = SharkSpinner("Test", file=io.StringIO())
        spinner._enabled = True
        # Corrupt frames to force an error
        spinner.frames = None  # type: ignore[assignment]
        with spinner:
            time.sleep(0.15)
        # Should not raise

    def test_writes_to_file(self):
        """Spinner output appears in the target file."""
        buf = io.StringIO()
        spinner = SharkSpinner("Swimming", file=buf)
        spinner._enabled = True
        with spinner:
            time.sleep(0.2)
        output = buf.getvalue()
        assert "Swimming" in output


class TestTunaLogHandler:
    def test_info_has_bullet(self, monkeypatch):
        import logging

        con, buf = _capture_console()
        handler = TunaLogHandler(console=con)
        handler._err = con  # capture stderr too
        logger = logging.getLogger("test_ui_info")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.info("hello info")
        output = buf.getvalue()
        assert "hello info" in output
        assert "▸" in output
        logger.removeHandler(handler)

    def test_error_has_cross(self, monkeypatch):
        import logging

        con, buf = _capture_console()
        handler = TunaLogHandler(console=con)
        handler._err = con
        logger = logging.getLogger("test_ui_error")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.error("bad stuff")
        output = buf.getvalue()
        assert "bad stuff" in output
        assert "✗" in output
        logger.removeHandler(handler)

    def test_warning_has_bang(self, monkeypatch):
        import logging

        con, buf = _capture_console()
        handler = TunaLogHandler(console=con)
        handler._err = con
        logger = logging.getLogger("test_ui_warning")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.warning("watch out")
        output = buf.getvalue()
        assert "watch out" in output
        assert "!" in output
        logger.removeHandler(handler)
