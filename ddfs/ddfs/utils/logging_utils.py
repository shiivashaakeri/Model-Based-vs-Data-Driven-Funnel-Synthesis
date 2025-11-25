"""
Logging Utilities for DDFS.

This module provides a centralized logging system with:
- Configurable verbosity levels (DEBUG, INFO, WARNING, ERROR)
- Module-aware logging
- Console output with module names
- Colored output support
"""

from __future__ import annotations

import functools
import logging
import sys
import time
from typing import Callable, Optional

import numpy as np

# =============================================================================
# Color Codes for Terminal Output
# =============================================================================


class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

    @classmethod
    def disable(cls):
        """Disable colors (for non-TTY environments)."""
        cls.RESET = ""
        cls.BOLD = ""
        cls.BLACK = ""
        cls.RED = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.BLUE = ""
        cls.MAGENTA = ""
        cls.CYAN = ""
        cls.WHITE = ""
        cls.BRIGHT_RED = ""
        cls.BRIGHT_GREEN = ""
        cls.BRIGHT_YELLOW = ""
        cls.BRIGHT_BLUE = ""
        cls.BRIGHT_MAGENTA = ""
        cls.BRIGHT_CYAN = ""


# Disable colors if not a TTY
if not sys.stdout.isatty():
    Colors.disable()


# =============================================================================
# Custom Formatter
# =============================================================================


class DDFSFormatter(logging.Formatter):
    """
    Custom formatter for DDFS logging.

    Format: [LEVEL] module_name: message
    With colors for different log levels.
    """

    LEVEL_COLORS = {  # noqa: RUF012
        logging.DEBUG: Colors.CYAN,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BRIGHT_RED + Colors.BOLD,
    }

    def __init__(self, use_colors: bool = True):
        """
        Initialize formatter.

        Parameters
        ----------
        use_colors : bool
            Whether to use colored output.
        """
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record."""
        # Get short module name (last part of dotted path)
        module_parts = record.name.split(".")
        module_name = (
            ".".join(module_parts[1:]) if len(module_parts) > 1 and module_parts[0] == "ddfs" else record.name
        )

        # Format level name
        level_name = record.levelname

        if self.use_colors:
            color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
            formatted = (
                f"{color}[{level_name}]{Colors.RESET} {Colors.BOLD}{module_name}{Colors.RESET}: {record.getMessage()}"
            )
        else:
            formatted = f"[{level_name}] {module_name}: {record.getMessage()}"

        # Handle exceptions
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)

        return formatted


# =============================================================================
# Logger Setup
# =============================================================================


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a module.

    Parameters
    ----------
    name : str
        Module name (typically __name__).

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Example
    -------
    >>> logger = get_logger(__name__)
    >>> logger.info("Starting computation")
    """
    return logging.getLogger(name)


def setup_logging(
    level: int = logging.INFO,
    use_colors: bool = True,
) -> None:
    """
    Setup the DDFS logging system.

    Should be called once at the start of the program.

    Parameters
    ----------
    level : int
        Logging level (e.g., logging.DEBUG, logging.INFO).
    use_colors : bool
        Whether to use colored output.

    Example
    -------
    >>> setup_logging(level=logging.DEBUG)
    """
    # Get root logger for ddfs package
    root_logger = logging.getLogger("ddfs")
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Set formatter
    formatter = DDFSFormatter(use_colors=use_colors)
    console_handler.setFormatter(formatter)

    # Add handler
    root_logger.addHandler(console_handler)

    # Prevent propagation to root logger
    root_logger.propagate = False


def set_log_level(level: int) -> None:
    """
    Set logging level for all DDFS loggers.

    Parameters
    ----------
    level : int
        Logging level.
    """
    logger = logging.getLogger("ddfs")
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def set_debug() -> None:
    """Enable debug logging."""
    set_log_level(logging.DEBUG)


def set_info() -> None:
    """Set logging to info level."""
    set_log_level(logging.INFO)


def set_warning() -> None:
    """Set logging to warning level (suppress info)."""
    set_log_level(logging.WARNING)


def set_error() -> None:
    """Set logging to error level (suppress warnings)."""
    set_log_level(logging.ERROR)


def silence() -> None:
    """Silence all DDFS logging."""
    set_log_level(logging.CRITICAL + 1)


# =============================================================================
# Timing Decorator
# =============================================================================


def timed(func: Optional[Callable] = None, *, log_level: int = logging.DEBUG):
    """
    Decorator to time function execution.

    Can be used with or without arguments.

    Parameters
    ----------
    func : callable, optional
        Function to decorate (when used without parentheses).
    log_level : int
        Logging level for timing messages.

    Returns
    -------
    callable
        Decorated function.

    Example
    -------
    >>> @timed
    ... def my_function():
    ...     pass

    >>> @timed(log_level=logging.INFO)
    ... def my_function():
    ...     pass
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            logger = get_logger(fn.__module__)
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.log(log_level, f"{fn.__name__} completed in {elapsed:.4f}s")
            return result

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


class Timer:
    """
    Context manager for timing code blocks.

    Example
    -------
    >>> with Timer("matrix computation"):
    ...     result = heavy_computation()
    [DEBUG] Timer: matrix computation took 1.2345s
    """

    def __init__(
        self,
        name: str = "block",
        log_level: int = logging.DEBUG,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize timer.

        Parameters
        ----------
        name : str
            Name for the timed block.
        log_level : int
            Logging level for timing message.
        logger : logging.Logger, optional
            Logger to use. If None, uses ddfs.utils logger.
        """
        self.name = name
        self.log_level = log_level
        self.logger = logger or get_logger("ddfs.utils")
        self.start = None
        self.elapsed = None

    def __enter__(self) -> "Timer":
        """Start timing."""
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        """Stop timing and log result."""
        self.elapsed = time.perf_counter() - self.start
        self.logger.log(self.log_level, f"{self.name} took {self.elapsed:.4f}s")


# =============================================================================
# Progress Logging
# =============================================================================


class ProgressLogger:
    """
    Simple progress logger for iterative algorithms.

    Example
    -------
    >>> progress = ProgressLogger(total=100, name="SCvx")
    >>> for i in range(100):
    ...     # do work
    ...     progress.update(i + 1)
    """

    def __init__(
        self,
        total: int,
        name: str = "Progress",
        log_interval: int = 10,
        log_level: int = logging.INFO,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize progress logger.

        Parameters
        ----------
        total : int
            Total number of iterations.
        name : str
            Name for the progress bar.
        log_interval : int
            Log every N iterations.
        log_level : int
            Logging level.
        logger : logging.Logger, optional
            Logger to use.
        """
        self.total = total
        self.name = name
        self.log_interval = log_interval
        self.log_level = log_level
        self.logger = logger or get_logger("ddfs.utils")
        self.current = 0

    def update(self, current: Optional[int] = None, message: str = "") -> None:
        """
        Update progress.

        Parameters
        ----------
        current : int, optional
            Current iteration (if None, increments by 1).
        message : str, optional
            Additional message to log.
        """
        if current is not None:
            self.current = current
        else:
            self.current += 1

        if self.current % self.log_interval == 0 or self.current == self.total:
            percent = 100 * self.current / self.total
            msg = f"{self.name}: {self.current}/{self.total} ({percent:.1f}%)"
            if message:
                msg += f" - {message}"
            self.logger.log(self.log_level, msg)

    def complete(self, message: str = "Done") -> None:
        """
        Mark progress as complete.

        Parameters
        ----------
        message : str
            Completion message.
        """
        self.logger.log(self.log_level, f"{self.name}: {message}")


# =============================================================================
# Logging Utilities
# =============================================================================


def log_array_stats(
    logger: logging.Logger,
    name: str,
    arr: "np.ndarray",
    level: int = logging.DEBUG,
) -> None:
    """
    Log statistics of a numpy array.

    Parameters
    ----------
    logger : logging.Logger
        Logger to use.
    name : str
        Name of the array.
    arr : np.ndarray
        Array to analyze.
    level : int
        Logging level.
    """
    stats = (
        f"{name}: shape={arr.shape}, "
        f"min={np.min(arr):.4e}, max={np.max(arr):.4e}, "
        f"mean={np.mean(arr):.4e}, std={np.std(arr):.4e}"
    )
    logger.log(level, stats)


def log_matrix_properties(
    logger: logging.Logger,
    name: str,
    M: np.ndarray,
    level: int = logging.DEBUG,
) -> None:
    """
    Log properties of a matrix.

    Parameters
    ----------
    logger : logging.Logger
        Logger to use.
    name : str
        Name of the matrix.
    M : np.ndarray
        Matrix to analyze.
    level : int
        Logging level.
    """


    props = [f"{name}: shape={M.shape}"]

    if M.shape[0] == M.shape[1]:
        # Square matrix - compute additional properties
        try:
            eigvals = np.linalg.eigvalsh(M)
            props.append(f"λ_min={np.min(eigvals):.4e}")
            props.append(f"λ_max={np.max(eigvals):.4e}")
            props.append(f"cond={np.max(eigvals) / np.max(np.abs(np.min(eigvals)), 1e-12):.4e}")
        except Exception:
            props.append("(eigenvalue computation failed)")

    logger.log(level, ", ".join(props))


# =============================================================================
# Initialize Default Logging
# =============================================================================

# Setup default logging when module is imported
setup_logging(level=logging.INFO, use_colors=True)
