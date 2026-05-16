"""tool_output_reducer plugin — deterministic pre-LLM compactor for tool results.

Wires the ``transform_tool_result`` hook to strip ANSI escape sequences and
collapse adjacent duplicate lines in tool output before that output is
appended to the LLM conversation. Saves context tokens on noisy tool calls
(bash, web fetch, etc.) without an LLM in the loop.

Conservative v1: two universal rules, with a pass-through allowlist for
tools where structure is the signal and a pass-through guard for error
responses. See ``reducer.py`` for the rule set and ``rules/`` for
individual transforms.

Disable: set ``HERMES_REDUCER_DISABLE=1`` (plugin registers no hooks).

Concept inspired by openhuman's TokenJuice (src/openhuman/tokenjuice/ in
github.com/tinyhumansai/openhuman, GPL-3). This is a clean-room Python
implementation — the rule patterns are common knowledge but no code was
copied from the GPL source.

TODO(2026-05-30): after ~2 weeks of orig->reduced observation in
~/.hermes/logs/tool_reducer.log, prune verbose logging to errors-only
(see ``_log_reduction`` in reducer.py). A one-shot Hermes cron fires
on that date to remind.
"""

from __future__ import annotations

import logging
import os

from .reducer import REDUCER_DISABLE_ENV, reduce_tool_output

logger = logging.getLogger(__name__)


def _on_transform_tool_result(**kwargs):
    """Hook entry. Returns reduced string or None (pass-through)."""
    tool_name = kwargs.get("tool_name", "")
    result = kwargs.get("result", "")
    if not isinstance(result, str):
        return None
    return reduce_tool_output(tool_name, result)


def register(ctx) -> None:
    if os.environ.get(REDUCER_DISABLE_ENV) == "1":
        logger.info(
            "tool_output_reducer disabled via %s=1; not registering hook",
            REDUCER_DISABLE_ENV,
        )
        return
    ctx.register_hook("transform_tool_result", _on_transform_tool_result)
