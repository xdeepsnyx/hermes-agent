"""Tests for the Daily-Checkin state-aware system-prompt addendum.

Custom Deep/Nyx fork patch — when the evening checkin state file shows
state=open AND this session is bound to the state's thread_id (the Discord
#checkin channel post-migration), `_build_daily_checkin_addendum` returns
a mandate string that gets injected into the system prompt so the agent
knows to load the deep-second-brain skill and run the touch/close protocol.

Without this, gateway-spawned reply sessions don't know they're inside a
checkin (observed silent-failure pattern 2026-05-11).
"""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from run_agent import AIAgent


def _make_state(*, open: bool, thread_id: str = "1503366449278746804",
                opener_date: str = "2026-05-12") -> dict:
    return {
        "schema_version": 1,
        "opener_fired_at": f"{opener_date}T21:30:00+05:30",
        "opener_date": opener_date,
        "thread_id": thread_id,
        "state": "open" if open else "closed",
        "closed_at": None if open else f"{opener_date}T23:00:00+05:30",
        "closed_reason": None if open else "completed",
        "turn_count": 0 if open else 7,
        "last_activity_at": f"{opener_date}T21:30:00+05:30",
        "carry_over_from_previous": None,
    }


def _fake_agent(gateway_session_key: str = "") -> AIAgent:
    """Build a minimal AIAgent stub just for testing the addendum helper.

    The helper only reads ``self._gateway_session_key`` and the state file;
    everything else can be bypassed by setting attributes directly on a
    SimpleNamespace that has the bound method dispatched onto it.

    NOTE: the helper matches against the gateway *session key* (the stable
    per-chat routing key, e.g. ``agent:main:discord:group:<channel>:<user>``),
    NOT ``self.session_id`` (the timestamp-based DB id, which never contains
    the channel id — the bug fixed 2026-05-14).
    """
    stub = SimpleNamespace(_gateway_session_key=gateway_session_key)
    # Phase 2 (2026-05-15): the addendum builder calls
    # ``self._read_prep_file_context(opener_date)`` to inject NyxBrain's
    # pre-checkin synthesis. Bind it on the stub so the dispatch works
    # without instantiating a full AIAgent.
    stub._read_prep_file_context = (
        lambda opener_date: AIAgent._read_prep_file_context(stub, opener_date)
    )
    return stub


def _call(stub, state_file: Path):
    """Invoke the helper with a mocked state-file path."""
    with patch.dict("os.environ", {"DAILY_CHECKIN_STATE_FILE": str(state_file)}):
        return AIAgent._build_daily_checkin_addendum(stub)


class TestDailyCheckinAddendum:
    def test_returns_none_when_state_file_missing(self, tmp_path):
        stub = _fake_agent(gateway_session_key="agent:main:discord:group:1503366449278746804:1234")
        result = _call(stub, tmp_path / "nonexistent.json")
        assert result is None

    def test_returns_none_when_state_closed(self, tmp_path):
        sf = tmp_path / "state.json"
        sf.write_text(json.dumps(_make_state(open=False)))
        stub = _fake_agent(gateway_session_key="agent:main:discord:group:1503366449278746804:1234")
        result = _call(stub, sf)
        assert result is None

    def test_returns_none_when_session_not_bound_to_thread(self, tmp_path):
        """Different channel — must NOT inject the mandate into unrelated chats."""
        sf = tmp_path / "state.json"
        sf.write_text(json.dumps(_make_state(open=True, thread_id="1503366449278746804")))
        stub = _fake_agent(gateway_session_key="agent:main:discord:group:9999999999999999999:1234")
        result = _call(stub, sf)
        assert result is None

    def test_returns_addendum_when_open_and_session_bound(self, tmp_path):
        sf = tmp_path / "state.json"
        sf.write_text(json.dumps(_make_state(open=True, thread_id="1503366449278746804",
                                            opener_date="2026-05-12")))
        stub = _fake_agent(
            gateway_session_key="agent:main:discord:group:1503366449278746804:1502392788233224203"
        )
        result = _call(stub, sf)
        assert result is not None
        # Content sanity — addendum must reference the skill, the touch script,
        # the close call, and the opener_date placeholder for the Reflection path.
        assert "deep-second-brain" in result
        assert "## Daily Checkin" in result
        assert "evening_checkin_opener.py touch" in result
        assert "evening_checkin_opener.py close" in result
        assert "2026-05-12.md" in result
        assert "1503366449278746804" in result

    def test_addendum_includes_turn_1_marker_instructions(self, tmp_path):
        """Mandate must instruct the agent to prepend the 🟢 marker on turn 1
        only, with the opener_date interpolated."""
        sf = tmp_path / "state.json"
        sf.write_text(json.dumps(_make_state(open=True, opener_date="2026-05-12")))
        stub = _fake_agent(
            gateway_session_key="agent:main:discord:group:1503366449278746804:1234"
        )
        result = _call(stub, sf)
        assert result is not None
        assert "🟢" in result, "addendum must include the turn-1 green marker"
        assert "turn 1" in result.lower()
        assert "Checkin mode · 2026-05-12 · turn 1" in result
        # Must explicitly mention the N==1 condition + N>1 suppression
        assert "N == 1" in result or "N==1" in result
        assert "N > 1" in result or "N>1" in result

    def test_addendum_includes_close_protocol_with_full_reflection_post(self, tmp_path):
        """Close step must mandate (a) compose, (b) write to file, (c) call close,
        (d) post the FULL Reflection content in chat with 🔻 marker."""
        sf = tmp_path / "state.json"
        sf.write_text(json.dumps(_make_state(open=True, opener_date="2026-05-12")))
        stub = _fake_agent(
            gateway_session_key="agent:main:discord:group:1503366449278746804:1234"
        )
        result = _call(stub, sf)
        assert result is not None
        assert "🔻" in result
        assert "Checkin closed for 2026-05-12" in result
        # All four steps of close must be present
        assert "(a)" in result
        assert "(b)" in result
        assert "(c)" in result
        assert "(d)" in result
        # The post must include the full Reflection content, not just a pointer
        assert "verbatim" in result.lower() or "full Reflection content" in result
        # Want edits prompt must be there
        assert "Want any edits" in result

    def test_addendum_includes_post_close_clause(self, tmp_path):
        """If a future turn sees state == 'closed' (because the prompt is
        cached from when state was open), the agent must NOT re-run the
        protocol — just chat normally."""
        sf = tmp_path / "state.json"
        sf.write_text(json.dumps(_make_state(open=True, opener_date="2026-05-12")))
        stub = _fake_agent(
            gateway_session_key="agent:main:discord:group:1503366449278746804:1234"
        )
        result = _call(stub, sf)
        assert result is not None
        assert "POST-CLOSE MODE" in result or "post-close" in result.lower()
        # Must explicitly say DO NOT re-run
        assert "DO NOT re-run" in result or "do not re-run" in result.lower()
        # Edits should go to the Reflection file directly
        assert "edit the file" in result.lower()

    def test_returns_none_when_thread_id_field_empty(self, tmp_path):
        sf = tmp_path / "state.json"
        s = _make_state(open=True)
        s["thread_id"] = ""
        sf.write_text(json.dumps(s))
        stub = _fake_agent(gateway_session_key="agent:main:discord:group:1503366449278746804:1234")
        result = _call(stub, sf)
        assert result is None

    def test_returns_none_on_malformed_state_file(self, tmp_path):
        sf = tmp_path / "state.json"
        sf.write_text("{this is not valid json")
        stub = _fake_agent(gateway_session_key="agent:main:discord:group:1503366449278746804:1234")
        result = _call(stub, sf)
        assert result is None

    def test_telegram_session_id_format_also_matches(self, tmp_path):
        """If anyone re-points the state file at a Telegram thread, the
        substring match still works since the gateway session key contains
        the thread_id as a path segment."""
        sf = tmp_path / "state.json"
        sf.write_text(json.dumps(_make_state(open=True, thread_id="38717")))
        stub = _fake_agent(gateway_session_key="agent:main:telegram:dm:962467459:38717")
        result = _call(stub, sf)
        assert result is not None
        assert "38717" in result

    def test_addendum_includes_listen_dont_operate_rule(self, tmp_path):
        """The checkin is a listening space — the mandate must forbid
        using tools to investigate/fix things Deep mentions mid-checkin."""
        sf = tmp_path / "state.json"
        sf.write_text(json.dumps(_make_state(open=True, opener_date="2026-05-12")))
        stub = _fake_agent(
            gateway_session_key="agent:main:discord:group:1503366449278746804:1234"
        )
        result = _call(stub, sf)
        assert result is not None
        assert "LISTEN, DON'T OPERATE" in result
        # Must explicitly forbid tool use to diagnose/fix
        assert "diagnose" in result.lower() or "investigate" in result.lower()
        # Must say action items are routed at close, not mid-conversation
        assert "at close" in result.lower() or "AT CLOSE" in result

    def test_addendum_includes_skip_branch(self, tmp_path):
        """Close step 4 must have an explicit SKIPPED branch that still
        writes a _Skipped_ marker and posts a 🔻 close notification."""
        sf = tmp_path / "state.json"
        sf.write_text(json.dumps(_make_state(open=True, opener_date="2026-05-12")))
        stub = _fake_agent(
            gateway_session_key="agent:main:discord:group:1503366449278746804:1234"
        )
        result = _call(stub, sf)
        assert result is not None
        # Both close branches present
        assert "4A" in result and "COMPLETED" in result
        assert "4B" in result and "SKIPPED" in result
        # Skip branch writes a _Skipped_ marker and posts a close notification
        assert "_Skipped" in result
        assert "Checkin skipped for 2026-05-12" in result
        # Valid --reason values are constrained (the agent tried an invalid
        # 'user_declined' in the 2026-05-14 test)
        assert "`completed` or `skipped`" in result or "completed` or `skipped" in result


# =============================================================================
# Phase 2 (2026-05-15): NyxBrain prep file context wiring
# =============================================================================
#
# When the addendum fires for a checkin, it now also reads NyxBrain's pre-
# checkin prep file at ~/.hermes/state/checkin_prep_<opener_date>.md and
# injects its non-opener sections (Context for Hermes / Trend signals /
# Topics / Tone) so the reply-turn agent's responses are informed by
# NyxBrain's earlier synthesis. Missing/stale/malformed prep file = no
# injection (graceful degradation; agent runs as pre-Phase-2).


_SAMPLE_PREP = """---
prep_for: 2026-05-12
generated_at: 2026-05-12T21:15:00+05:30
generator: NyxBrain
---

# Daily Checkin Prep — 2026-05-12

## Opener
Three blank Reflections in a row — was it the week, or did you set the journal down on purpose?

## Context for Hermes (not for posting)
- Three nights of empty Reflections preceded today.
- Friday, no calendar events; commitment load is in Reminders.
- Backlog is heavy: ~15 open reminders, several STALE.

## Trend signals (from the week)
- Recovery arc 82 → 54 → 33 Mon→Wed; body tracked with the week.
- Capture density collapsed Thu/Fri.

## Topics to weave in if Deep goes deep
- The carryover from Monday's anchor.
- Body state across the week.

## Tone
Warm, low-key, unhurried. Don't extract status.
"""


def _call_with_prep(stub, state_file: Path, prep_dir: Path):
    """Invoke the helper with mocked state + prep dir paths."""
    with patch.dict("os.environ", {
        "DAILY_CHECKIN_STATE_FILE": str(state_file),
        "DAILY_CHECKIN_PREP_DIR": str(prep_dir),
    }):
        return AIAgent._build_daily_checkin_addendum(stub)


class TestDailyCheckinAddendumPrepContext:
    def test_prep_context_injected_when_fresh(self, tmp_path):
        """A fresh prep file (matching prep_for) should land its non-opener
        sections inside the addendum under the NyxBrain prep header."""
        sf = tmp_path / "state.json"
        sf.write_text(json.dumps(_make_state(open=True, opener_date="2026-05-12")))
        prep_dir = tmp_path / "prep"
        prep_dir.mkdir()
        (prep_dir / "checkin_prep_2026-05-12.md").write_text(_SAMPLE_PREP)
        stub = _fake_agent(
            gateway_session_key="agent:main:discord:group:1503366449278746804:1234"
        )
        result = _call_with_prep(stub, sf, prep_dir)
        assert result is not None
        # The prep block header must appear
        assert "## NyxBrain prep for tonight" in result
        # The framing must say it's situational awareness, NOT to repeat
        assert "situational awareness" in result.lower()
        assert "NOT" in result  # the "NOT conversation prompts to repeat" guard
        # Each non-opener section title must appear
        assert "## Context for Hermes" in result
        assert "## Trend signals" in result
        assert "## Topics to weave in if Deep goes deep" in result
        assert "## Tone" in result
        # Sample content from each section
        assert "empty Reflections" in result
        assert "82 → 54 → 33" in result
        assert "Warm, low-key, unhurried" in result
        # The H1 title `# Daily Checkin Prep — 2026-05-12` is metadata, not
        # content — must be stripped to avoid clashing with the addendum's
        # own `# Daily Checkin in progress` H1.
        assert "# Daily Checkin Prep" not in result
        # Critically: the `## Opener` line should NOT be in the addendum —
        # it's already been posted to Discord by the 21:30 cron.
        assert "## Opener" not in result
        assert "set the journal down on purpose" not in result
        # Frontmatter shouldn't leak through either
        assert "prep_for: 2026-05-12" not in result
        assert "generator: NyxBrain" not in result

    def test_no_prep_context_when_file_missing(self, tmp_path):
        """No prep file = degraded path: addendum still fires with full
        protocol, just no NyxBrain prep block. Pre-Phase-2 behavior."""
        sf = tmp_path / "state.json"
        sf.write_text(json.dumps(_make_state(open=True, opener_date="2026-05-12")))
        prep_dir = tmp_path / "prep"
        prep_dir.mkdir()
        # No prep file written
        stub = _fake_agent(
            gateway_session_key="agent:main:discord:group:1503366449278746804:1234"
        )
        result = _call_with_prep(stub, sf, prep_dir)
        assert result is not None
        # Addendum still fires with all protocol mandates
        assert "Daily Checkin in progress" in result
        assert "LISTEN, DON'T OPERATE" in result
        assert "evening_checkin_opener.py touch" in result
        # But no prep block
        assert "## NyxBrain prep for tonight" not in result
        assert "situational awareness" not in result.lower()

    def test_no_prep_context_when_prep_for_stale(self, tmp_path):
        """Prep file exists but `prep_for` doesn't match today's opener_date
        — must NOT inject (would mislead with yesterday's context)."""
        sf = tmp_path / "state.json"
        sf.write_text(json.dumps(_make_state(open=True, opener_date="2026-05-12")))
        prep_dir = tmp_path / "prep"
        prep_dir.mkdir()
        # Prep file from a DIFFERENT day
        stale = _SAMPLE_PREP.replace("prep_for: 2026-05-12", "prep_for: 2026-05-11")
        (prep_dir / "checkin_prep_2026-05-12.md").write_text(stale)
        stub = _fake_agent(
            gateway_session_key="agent:main:discord:group:1503366449278746804:1234"
        )
        result = _call_with_prep(stub, sf, prep_dir)
        assert result is not None
        # Addendum still fires
        assert "Daily Checkin in progress" in result
        # But no prep block — the stale freshness check guarded it
        assert "## NyxBrain prep for tonight" not in result

    def test_no_prep_context_when_prep_file_malformed(self, tmp_path):
        """Prep file exists but is garbage / no `prep_for` field — must NOT
        inject. Same graceful-degrade as missing."""
        sf = tmp_path / "state.json"
        sf.write_text(json.dumps(_make_state(open=True, opener_date="2026-05-12")))
        prep_dir = tmp_path / "prep"
        prep_dir.mkdir()
        (prep_dir / "checkin_prep_2026-05-12.md").write_text("garbage content with no prep_for line")
        stub = _fake_agent(
            gateway_session_key="agent:main:discord:group:1503366449278746804:1234"
        )
        result = _call_with_prep(stub, sf, prep_dir)
        assert result is not None
        assert "Daily Checkin in progress" in result
        assert "## NyxBrain prep for tonight" not in result

    def test_no_prep_context_when_only_opener_section(self, tmp_path):
        """Prep file with only the `## Opener` section (no Context/Trend/etc.)
        — after stripping the opener, no content remains, so no prep block."""
        sf = tmp_path / "state.json"
        sf.write_text(json.dumps(_make_state(open=True, opener_date="2026-05-12")))
        prep_dir = tmp_path / "prep"
        prep_dir.mkdir()
        opener_only = (
            "---\nprep_for: 2026-05-12\ngenerator: NyxBrain\n---\n\n"
            "# Daily Checkin Prep — 2026-05-12\n\n"
            "## Opener\n"
            "What's worth holding onto from this week?\n"
        )
        (prep_dir / "checkin_prep_2026-05-12.md").write_text(opener_only)
        stub = _fake_agent(
            gateway_session_key="agent:main:discord:group:1503366449278746804:1234"
        )
        result = _call_with_prep(stub, sf, prep_dir)
        assert result is not None
        assert "Daily Checkin in progress" in result
        # No prep block (opener-only file → empty after stripping)
        assert "## NyxBrain prep for tonight" not in result
        # Critically the opener question must not have leaked into the addendum
        assert "What's worth holding onto" not in result

    def test_prep_context_only_fires_when_addendum_fires(self, tmp_path):
        """Prep file fresh, but state is closed → no addendum at all (no
        prep block either). Defense: prep injection must inherit the same
        gating as the addendum itself; never inject prep for a non-checkin
        session."""
        sf = tmp_path / "state.json"
        sf.write_text(json.dumps(_make_state(open=False, opener_date="2026-05-12")))
        prep_dir = tmp_path / "prep"
        prep_dir.mkdir()
        (prep_dir / "checkin_prep_2026-05-12.md").write_text(_SAMPLE_PREP)
        stub = _fake_agent(
            gateway_session_key="agent:main:discord:group:1503366449278746804:1234"
        )
        result = _call_with_prep(stub, sf, prep_dir)
        # State closed → addendum returns None → no prep block, no anything
        assert result is None
