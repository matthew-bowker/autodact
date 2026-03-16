"""System resource management for long-running work.

Provides two context managers:

``keep_awake()``
    Prevent the OS from sleeping during processing.

``low_priority()``
    Lower the current thread/process priority so the rest of the
    system stays responsive.

On macOS, Windows, and Linux the right OS primitives are used
automatically.  On unsupported platforms both are silent no-ops.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)

_SYSTEM = platform.system()


# ------------------------------------------------------------------
# keep_awake — prevent idle sleep
# ------------------------------------------------------------------

@contextmanager
def keep_awake() -> Generator[None, None, None]:
    """Context manager that inhibits system idle-sleep."""
    if _SYSTEM == "Darwin":
        yield from _keep_awake_macos()
    elif _SYSTEM == "Windows":
        yield from _keep_awake_windows()
    else:
        yield


def _keep_awake_macos() -> Generator[None, None, None]:
    proc: subprocess.Popen | None = None
    try:
        # -i = prevent idle sleep (not display sleep — screen can dim)
        proc = subprocess.Popen(
            ["caffeinate", "-i"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.debug("caffeinate started (pid %d)", proc.pid)
        yield
    except FileNotFoundError:
        logger.debug("caffeinate not found, skipping sleep inhibit")
        yield
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait()
            logger.debug("caffeinate stopped")


def _keep_awake_windows() -> Generator[None, None, None]:
    try:
        import ctypes

        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
        logger.debug("SetThreadExecutionState: sleep inhibited")
        yield
    except Exception:
        logger.debug("SetThreadExecutionState failed, skipping sleep inhibit")
        yield
    finally:
        try:
            import ctypes

            ES_CONTINUOUS = 0x80000000
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            logger.debug("SetThreadExecutionState: sleep restored")
        except Exception:
            pass


# ------------------------------------------------------------------
# low_priority — yield CPU to foreground apps
# ------------------------------------------------------------------

@contextmanager
def low_priority() -> Generator[None, None, None]:
    """Lower the current process priority, restoring it on exit."""
    if _SYSTEM == "Windows":
        yield from _low_priority_windows()
    else:
        # macOS / Linux: os.nice() works on both
        yield from _low_priority_unix()


def _low_priority_unix() -> Generator[None, None, None]:
    try:
        os.nice(10)  # +10 is "background-ish" without being completely starved
        logger.debug("Process nice value raised by 10")
    except OSError:
        logger.debug("os.nice() failed, skipping priority change")
    # nice() is not reversible — the OS only lets you increase it,
    # not decrease it back, so there is no cleanup to do.
    yield


def _low_priority_windows() -> Generator[None, None, None]:
    try:
        import ctypes

        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(
            handle, BELOW_NORMAL_PRIORITY_CLASS,
        )
        logger.debug("Process priority set to BELOW_NORMAL")
        yield
    except Exception:
        logger.debug("SetPriorityClass failed, skipping priority change")
        yield
    finally:
        try:
            import ctypes

            NORMAL_PRIORITY_CLASS = 0x00000020
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(
                handle, NORMAL_PRIORITY_CLASS,
            )
            logger.debug("Process priority restored to NORMAL")
        except Exception:
            pass


# ------------------------------------------------------------------
# Sensible default thread count
# ------------------------------------------------------------------

def default_thread_count() -> int:
    """Return a thread count that leaves headroom for the rest of the system.

    Uses half the available cores (minimum 2, maximum 8).
    """
    cores = os.cpu_count() or 4
    return max(2, min(cores // 2, 8))
