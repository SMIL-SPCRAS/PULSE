"""
Filter known noisy macOS/Gradio warnings from stderr.
"""

from __future__ import annotations

import sys

BLOCKED_TOKENS = (
    "Class AVFFrameReceiver is implemented in both",
    "Class AVFAudioReceiver is implemented in both",
    "Watermarking for SVG images is currently not supported",
)

BLOCK_NEXT_LINE_TOKENS = ("gradio/helpers.py",)


def should_block_line(line: str) -> bool:
    """Return whether a stderr line should be hidden."""

    return any(token in line for token in BLOCKED_TOKENS)


def should_block_next_line(line: str) -> bool:
    """Return whether the next stderr line should also be hidden."""

    return any(token in line for token in BLOCK_NEXT_LINE_TOKENS)


def main() -> None:
    """Stream stderr while dropping known noisy warnings."""

    skip_next_line = False

    for line in sys.stdin:
        if skip_next_line:
            skip_next_line = False
            continue

        if should_block_line(line):
            continue

        if should_block_next_line(line):
            skip_next_line = True
            continue

        sys.stderr.write(line)
        sys.stderr.flush()


if __name__ == "__main__":
    main()
