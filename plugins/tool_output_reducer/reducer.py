"""Per-tool dispatch + observability log for tool_output_reducer.

Public API:
    reduce_tool_output(tool_name, raw) -> Optional[str]
        Returns reduced string, or None to leave the original untouched.
    REDUCER_DISABLE_ENV: env var name for the disable switch.

Design:
    - Outputs < PASSTHROUGH_THRESHOLD bytes pass through unchanged
      (openhuman's threshold; reduction overhead isn't worth it on tiny
      output and edge cases multiply).
    - JSON error responses (``{"error": ...}``) pass through unchanged so
      stack traces and structured failures reach the LLM intact.
    - A small allowlist of structured-output tools (file reads, supermemory
      ops, etc.) skip the rule chain entirely. Missing a tool from this
      list is safe — the rules themselves are non-destructive on clean
      input.
    - If the rule chain raises, returns None (fail-open).
    - If the reduction didn't actually shrink the output, returns None
      (no point spending a hook return on a no-op).
    - Logs each successful reduction's orig->reduced byte count to
      ~/.hermes/logs/tool_reducer.log for the observation window.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from .rules import ansi, dedupe

logger = logging.getLogger(__name__)

REDUCER_DISABLE_ENV = "HERMES_REDUCER_DISABLE"
PASSTHROUGH_THRESHOLD = 240

_HERMES_HOME = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
LOG_PATH = _HERMES_HOME / "logs" / "tool_reducer.log"

# Tools whose output is already structured / clean — skip the rule chain.
# This is an optimization, not a safety guard: the rules are safe to run
# on these anyway, but skipping avoids spurious "reduced 0 bytes" log lines.
_PASSTHROUGH_TOOLS: frozenset[str] = frozenset(
    {
        "read_file",
        "write_file",
        "edit_file",
        "supermemory_add",
        "supermemory_search",
        "todo_write",
    }
)


def _ist_now() -> str:
    return datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(timespec="seconds")


def _log_reduction(tool_name: str, orig_bytes: int, reduced_bytes: int) -> None:
    """Best-effort observability log. Never raises.

    TODO(2026-05-30): switch to errors-only after observation window.
    """
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = (
            f"[{_ist_now()}] {tool_name} "
            f"{orig_bytes}->{reduced_bytes} "
            f"saved={orig_bytes - reduced_bytes}\n"
        )
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def _is_error_response(raw: str) -> bool:
    """True if raw is a JSON object with an ``error`` key."""
    if not raw.startswith("{"):
        return False
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(obj, dict) and "error" in obj


def reduce_tool_output(tool_name: str, raw: str) -> Optional[str]:
    """Apply universal rule chain (ANSI strip + adjacent-line dedupe) to raw.

    Returns the reduced string, or ``None`` if the original should pass
    through unchanged (too short, allowlisted tool, error response, rule
    error, or no actual reduction).
    """
    if not isinstance(raw, str) or len(raw) < PASSTHROUGH_THRESHOLD:
        return None
    if tool_name in _PASSTHROUGH_TOOLS:
        return None
    if _is_error_response(raw):
        return None

    try:
        reduced = ansi.strip_ansi(raw)
        reduced = dedupe.collapse_adjacent_duplicates(reduced)
    except Exception as e:
        logger.warning("reducer chain error for tool %s: %s", tool_name, e)
        return None

    if len(reduced) >= len(raw):
        return None

    _log_reduction(tool_name, len(raw), len(reduced))
    return reduced
