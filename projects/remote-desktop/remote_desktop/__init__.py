"""Stream a desktop screen to a phone and control it from there."""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__", "main"]


def main(argv: list[str] | None = None) -> int:
    """Run the agent; see :mod:`remote_desktop.__main__`."""
    from .__main__ import main as _main

    return _main(argv)
