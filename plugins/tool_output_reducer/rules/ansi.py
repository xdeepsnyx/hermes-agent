"""ANSI escape sequence stripping.

Covers the three common sequence families produced by terminal tools:

    CSI: \\x1b[ ... final-byte
        Color codes, cursor movement, screen clear. By far the most common.
    OSC: \\x1b] ... (BEL | ESC \\)
        Operating-system commands — terminal title, hyperlinks, etc.
    Two-char escapes: \\x1b @ through \\x1b _
        Less common; covers things like reset (\\x1bc).

Returns the input unchanged if no escapes are present.
"""

from __future__ import annotations

import re

_ANSI_RE = re.compile(
    r"\x1b("
    r"\[[0-?]*[ -/]*[@-~]"           # CSI
    r"|\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC, terminated by BEL or ESC \
    r"|[@-Z\\-_]"                      # two-char escape
    r")"
)


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences. Returns text unchanged if none present."""
    return _ANSI_RE.sub("", text)
