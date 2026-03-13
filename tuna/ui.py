"""Shared UI module — Rich console, orange theme, shark fin spinner."""

from __future__ import annotations

import itertools
import logging
import sys
import threading
import time
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


class _ProxyConsole:
    """Proxy that delegates to a fresh Console tied to current sys.stdout/stderr.

    This ensures Rich output is captured by pytest's capsys (which swaps
    sys.stdout at test time) instead of being pinned to the fd at import time.
    """

    def __init__(self, *, stderr: bool = False):
        self._stderr = stderr

    def __getattr__(self, name: str):
        return getattr(Console(stderr=self._stderr), name)


console: Console = _ProxyConsole()  # type: ignore[assignment]
err_console: Console = _ProxyConsole(stderr=True)  # type: ignore[assignment]

BRAND = "bold dark_orange"

# ---------------------------------------------------------------------------
# ASCII art banner
# ---------------------------------------------------------------------------

_BANNER = r"""
 ________ __    __ __    __  ______
|        \  \  |  \  \  |  \/      \
 \▓▓▓▓▓▓▓▓ ▓▓  | ▓▓ ▓▓\ | ▓▓  ▓▓▓▓▓▓\
   | ▓▓  | ▓▓  | ▓▓ ▓▓▓\| ▓▓ ▓▓__| ▓▓
   | ▓▓  | ▓▓  | ▓▓ ▓▓▓▓\ ▓▓ ▓▓    ▓▓
   | ▓▓  | ▓▓  | ▓▓ ▓▓\▓▓ ▓▓ ▓▓▓▓▓▓▓▓
   | ▓▓  | ▓▓__/ ▓▓ ▓▓ \▓▓▓▓ ▓▓  | ▓▓
   | ▓▓   \▓▓    ▓▓ ▓▓  \▓▓▓ ▓▓  | ▓▓
    \▓▓    \▓▓▓▓▓▓ \▓▓   \▓▓\▓▓   \▓▓
"""


def banner() -> None:
    """Print the TUNA ASCII art banner in orange."""
    console.print(f"[{BRAND}]{_BANNER}[/{BRAND}]")
    console.print(f" [{BRAND}]Hybrid GPU Inference Orchestrator by Tandemn[/{BRAND}]")
    console.print()


def section(title: str) -> None:
    """Print a horizontal rule with a bold title."""
    console.rule(f"[bold]{title}[/bold]", style="dark_orange")


def info_panel(
    title: str,
    data: dict[str, str],
    border_style: str = "dark_orange",
) -> None:
    """Print a bordered panel with a key-value table."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold")
    table.add_column("Value")
    for key, value in data.items():
        table.add_row(key, str(value))
    console.print(Panel(table, title=title, border_style=border_style))


def status_msg(msg: str) -> None:
    """Print a dim status bullet."""
    console.print(f"[dim]▸ {msg}[/dim]")


def success(msg: str) -> None:
    """Print a green success message."""
    console.print(f"[green]✓[/green] {msg}")


def error(msg: str) -> None:
    """Print a red error message to stderr."""
    err_console.print(f"[red]✗[/red] {msg}")


def warning(msg: str) -> None:
    """Print a yellow warning message."""
    console.print(f"[yellow]![/yellow] {msg}")


def styled_url(url: str) -> str:
    """Return a Rich-markup styled URL."""
    return f"[orange3]{url}[/orange3]"


# ---------------------------------------------------------------------------
# Shark fin spinner
# ---------------------------------------------------------------------------


def _detect_unicode_support() -> bool:
    """Check if the terminal supports Unicode characters."""
    try:
        encoding = sys.stdout.encoding
        if encoding:
            return encoding.lower() in ("utf-8", "utf8")
    except AttributeError:
        pass
    return False


class SharkSpinner:
    """Animated shark fin spinner: 𓂁 <-> 𓂄

    Usage::

        with SharkSpinner("Deploying model"):
            deploy_model()

    Falls back to ASCII if terminal doesn't support Unicode.
    Disables itself when stdout is not a TTY (CI / pipes).
    """

    FRAMES_UNICODE = [
        "𓂁··············",
        "°𓂁·············",
        "·°𓂁············",
        "··°𓂁···········",
        "···°𓂁··········",
        "····°𓂁·········",
        "·····°𓂁········",
        "······°𓂁·······",
        "·······°𓂁······",
        "········°𓂁·····",
        "·········°𓂁····",
        "··········°𓂁···",
        "···········°𓂁··",
        "············°𓂁·",
        "·············°𓂁",
        "··············𓂄°",
        "·············𓂄°·",
        "············𓂄°··",
        "···········𓂄°···",
        "··········𓂄°····",
        "·········𓂄°·····",
        "········𓂄°······",
        "·······𓂄°·······",
        "······𓂄°········",
        "·····𓂄°·········",
        "····𓂄°··········",
        "···𓂄°···········",
        "··𓂄°············",
        "·𓂄°·············",
        "𓂄°··············",
    ]

    FRAMES_ASCII = [
        ">~~~~~~~~~~~~~~",
        "o>~~~~~~~~~~~~~",
        "~o>~~~~~~~~~~~~",
        "~~o>~~~~~~~~~~~",
        "~~~o>~~~~~~~~~~",
        "~~~~o>~~~~~~~~~",
        "~~~~~o>~~~~~~~~",
        "~~~~~~o>~~~~~~~",
        "~~~~~~~o>~~~~~~",
        "~~~~~~~~o>~~~~~",
        "~~~~~~~~~o>~~~~",
        "~~~~~~~~~~o>~~~",
        "~~~~~~~~~~~o>~~",
        "~~~~~~~~~~~~o>~",
        "~~~~~~~~~~~~~o>",
        "~~~~~~~~~~~~~~<o",
        "~~~~~~~~~~~~~<o~",
        "~~~~~~~~~~~~<o~~",
        "~~~~~~~~~~~<o~~~",
        "~~~~~~~~~~<o~~~~",
        "~~~~~~~~~<o~~~~~",
        "~~~~~~~~<o~~~~~~",
        "~~~~~~~<o~~~~~~~",
        "~~~~~~<o~~~~~~~~",
        "~~~~~<o~~~~~~~~~",
        "~~~~<o~~~~~~~~~~",
        "~~~<o~~~~~~~~~~~",
        "~~<o~~~~~~~~~~~~",
        "~<o~~~~~~~~~~~~~",
        "<o~~~~~~~~~~~~~~",
    ]

    FRAME_DELAY = 0.08  # ~12 fps

    def __init__(
        self,
        message: str = "Loading",
        *,
        file: Optional[object] = None,
    ):
        self.message = message
        self._file = file  # for testing — defaults to sys.stderr at start()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._enabled = sys.stderr.isatty()
        self.frames = (
            self.FRAMES_UNICODE if _detect_unicode_support() else self.FRAMES_ASCII
        )

    def _animate(self) -> None:
        """Animation loop (background thread).

        Writes directly to stderr via raw ANSI — Rich console.print() does
        not handle \\r in-place overwriting correctly.  Stderr avoids
        collisions with log messages on stdout.
        """
        import shutil

        try:
            out = self._file or sys.stderr
            term_width = shutil.get_terminal_size().columns
            frame_cycle = itertools.cycle(self.frames)
            while not self._stop_event.is_set():
                frame = next(frame_cycle)
                line = f"  {self.message} {frame}"
                # Truncate to terminal width to avoid wrapping
                line = line[:term_width - 1]
                out.write(f"\r\033[2K\033[38;5;208m{line}\033[0m")  # orange
                out.flush()
                self._stop_event.wait(self.FRAME_DELAY)
            # Clear the spinner line
            out.write("\r\033[2K")
            out.flush()
        except Exception:
            logging.debug("Spinner animation failed", exc_info=True)
            self._stop_event.set()

    def start(self) -> None:
        """Start the spinner animation."""
        if not self._enabled:
            return
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._animate, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop the spinner animation."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.3)

    def __enter__(self) -> "SharkSpinner":
        self.start()
        return self

    def __exit__(self, *args: object) -> bool:
        self.stop()
        return False


# ---------------------------------------------------------------------------
# Rich logging handler for clean CLI output
# ---------------------------------------------------------------------------


class TunaLogHandler(logging.Handler):
    """Minimal Rich-based log handler.

    Normal mode: just the message with a ``▸`` prefix (INFO) or styled
    prefix for WARNING/ERROR. No timestamps, no logger names.

    Verbose mode: use ``RichHandler`` from Rich instead (configured in main()).
    """

    def __init__(self, console: Optional[Console] = None):
        super().__init__()
        self._console = console or globals()["console"]
        self._err = err_console

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if record.levelno >= logging.ERROR:
                self._err.print(f"[red]✗[/red] {msg}")
            elif record.levelno >= logging.WARNING:
                self._err.print(f"[yellow]![/yellow] {msg}")
            else:
                self._console.print(f"[dim]▸[/dim] {msg}")
        except Exception:
            self.handleError(record)
