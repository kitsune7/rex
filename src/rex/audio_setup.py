"""
Audio setup utilities.

Suppresses PortAudio diagnostic messages that clutter the console on macOS.
"""

import contextlib
import os


def suppress_portaudio_warnings():
    """
    Suppress PortAudio stderr messages (e.g., AUHAL errors on macOS).

    PortAudio writes diagnostic messages directly to stderr, bypassing Python's
    logging. This function redirects stderr to /dev/null during sounddevice
    import to suppress the initial "PaMacCore (AUHAL)" messages.

    Call this before any sounddevice imports in the application.
    """
    # Save the original stderr file descriptor
    original_stderr_fd = os.dup(2)

    try:
        # Open /dev/null and redirect stderr to it
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)

        # Import sounddevice to trigger PortAudio initialization
        # This is when the AUHAL messages are printed
        import sounddevice  # noqa: F401

    finally:
        # Restore stderr
        os.dup2(original_stderr_fd, 2)
        os.close(original_stderr_fd)


@contextlib.contextmanager
def suppress_stderr():
    """
    Context manager to temporarily suppress stderr.

    Use this when calling sounddevice functions that may print PortAudio warnings.
    """
    original_stderr_fd = os.dup(2)
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
        yield
    finally:
        os.dup2(original_stderr_fd, 2)
        os.close(original_stderr_fd)
