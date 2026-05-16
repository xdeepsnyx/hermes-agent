"""Collapse runs of identical adjacent lines.

For a run of N>1 identical lines, keep one and append a counter line
``  (... K more identical lines ...)`` where K = N-1. Single lines and
non-adjacent duplicates are preserved verbatim.

Safe for terminal/log output where the same line repeats from polling
or retry loops. Caller is responsible for not applying this to formats
where adjacent-line semantics matter (structured data, source code).
"""

from __future__ import annotations


def collapse_adjacent_duplicates(text: str) -> str:
    """If a line appears N>1 times in a row, collapse to one + counter."""
    if not text or "\n" not in text:
        return text

    lines = text.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        j = i
        while j + 1 < n and lines[j + 1] == lines[i]:
            j += 1
        out.append(lines[i])
        if j > i:
            out.append(f"  (... {j - i} more identical lines ...)")
        i = j + 1
    return "\n".join(out)
