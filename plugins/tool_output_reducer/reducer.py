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
# on these anyway (ANSI strip + dedupe are no-ops on structured JSON), but
# skipping avoids spurious "reduced 0 bytes" log lines.
#
# Names verified against the actual tool registry (`tools/*.py` +
# `plugins/memory/*` registry.register() calls) — wrong names just fall
# through to the universal chain harmlessly, but correct names are clearer.
_PASSTHROUGH_TOOLS: frozenset[str] = frozenset(
    {
        # File tools
        "read_file",
        "write_file",
        "patch",
        "search_files",
        # Memory / second-brain tools
        "memory",
        "supermemory_store",
        "supermemory_search",
        "supermemory_forget",
        # Task tracker
        "todo",
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


def _apply_chain(text: str) -> str:
    """Run the universal rule chain (ANSI strip + adjacent-line dedupe)."""
    text = ansi.strip_ansi(text)
    text = dedupe.collapse_adjacent_duplicates(text)
    return text


def _try_reduce_json_output(raw: str) -> Optional[str]:
    """If raw is a JSON object with a string ``output`` field, reduce that
    field's content and return the reserialized JSON. Returns ``None`` if
    the shape doesn't match or no actual reduction happened.

    This handles the dominant tool-result shape in hermes-agent: tools like
    ``terminal`` return ``json.dumps({"output": "<bash stdout>", ...})``,
    so the noisy text lives inside a JSON string value rather than at the
    top level. Without this, the dedupe rule sees one giant line (no real
    newlines in the JSON wrapper) and never fires.
    """
    if not raw.startswith("{"):
        return None
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    output = obj.get("output")
    if not isinstance(output, str) or len(output) < PASSTHROUGH_THRESHOLD:
        return None

    try:
        reduced_output = _apply_chain(output)
    except Exception as e:
        logger.warning("reducer chain error on JSON output field: %s", e)
        return None

    if len(reduced_output) >= len(output):
        return None

    obj["output"] = reduced_output
    return json.dumps(obj, ensure_ascii=False)


def reduce_tool_output(tool_name: str, raw: str) -> Optional[str]:
    """Apply universal rule chain (ANSI strip + adjacent-line dedupe) to raw.

    Two reduction paths:
      1. If raw is a JSON object with a string ``output`` field
         (terminal/bash and similar), reduce that field in place.
      2. Otherwise reduce the raw text directly.

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

    # Path 1: JSON-wrapped tool result with `output` field.
    json_reduced = _try_reduce_json_output(raw)
    if json_reduced is not None and len(json_reduced) < len(raw):
        _log_reduction(tool_name, len(raw), len(json_reduced))
        return json_reduced

    # Path 2: raw text (no JSON wrapping, or different schema).
    try:
        reduced = _apply_chain(raw)
    except Exception as e:
        logger.warning("reducer chain error for tool %s: %s", tool_name, e)
        return None

    if len(reduced) >= len(raw):
        return None

    _log_reduction(tool_name, len(raw), len(reduced))
    return reduced
