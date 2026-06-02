"""Tests for ``cron.scheduler._seed_cron_session_title``.

Without this helper, cron-spawned sessions render in the dashboard with the
literal ``[IMPORTANT: The user has invoked the "<skill>" skill, ...]``
preamble that ``_build_job_prompt`` prepends, because:
  1. ``cron/scheduler.py`` does not call ``maybe_auto_title``, so cron
     sessions never get an LLM-summarized title.
  2. ``SessionsPage.tsx`` falls back to ``preview.slice(0, 60)`` whenever
     ``title`` is empty.

These tests pin the helper's contract: pre-create the session row,
seed a deterministic ``"<job_name> · <timestamp>"`` title, truncate when
the assembled title would exceed ``MAX_TITLE_LENGTH``, and stay quiet on
failure (it's a UI nicety — never block job execution).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest


@pytest.fixture
def session_db(tmp_path, monkeypatch):
    """Real on-disk SessionDB so we exercise the actual SQL paths."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import importlib
    import hermes_constants
    importlib.reload(hermes_constants)
    import hermes_state
    importlib.reload(hermes_state)
    db = hermes_state.SessionDB()
    yield db
    db.close()


def _seed():
    from cron.scheduler import _seed_cron_session_title
    return _seed_cron_session_title


def test_seeds_title_with_job_name_and_timestamp(session_db):
    when = dt.datetime(2026, 5, 10, 21, 30, 42)
    title = _seed()(session_db, "cron_jobX_20260510_213042", "Apple Notes Inbox Watcher", when)

    assert title == "Apple Notes Inbox Watcher · 2026-05-10 21:30:42"
    assert (
        session_db.get_session_title("cron_jobX_20260510_213042")
        == "Apple Notes Inbox Watcher · 2026-05-10 21:30:42"
    )


def test_seconds_in_suffix_avoid_collisions_within_same_minute(session_db):
    """Two runs of the same job in the same minute must produce distinct titles."""
    early = dt.datetime(2026, 5, 10, 21, 30, 1)
    late = dt.datetime(2026, 5, 10, 21, 30, 59)

    t1 = _seed()(session_db, "cron_jobX_20260510_213001", "Watcher", early)
    t2 = _seed()(session_db, "cron_jobX_20260510_213059", "Watcher", late)

    assert t1 != t2
    assert t1 is not None and t2 is not None
    # Both rows persisted with their respective titles.
    assert session_db.get_session_title("cron_jobX_20260510_213001") == t1
    assert session_db.get_session_title("cron_jobX_20260510_213059") == t2


def test_creates_session_row_eagerly(session_db):
    """``set_session_title`` is an UPDATE — the helper must INSERT first."""
    when = dt.datetime(2026, 5, 10, 21, 30, 42)
    sid = "cron_jobX_20260510_213042"

    # Row absent before seeding.
    assert session_db.get_session_title(sid) is None

    title = _seed()(session_db, sid, "Watcher", when)

    assert title is not None
    # And the row now exists with the title set.
    assert session_db.get_session_title(sid) == title


def test_truncates_when_assembled_title_exceeds_max_length(session_db):
    """Long job names must be trimmed so the timestamp suffix survives."""
    from hermes_state import SessionDB

    when = dt.datetime(2026, 5, 10, 21, 30, 42)
    long_name = "X" * (SessionDB.MAX_TITLE_LENGTH + 50)
    sid = "cron_jobLong_20260510_213042"

    title = _seed()(session_db, sid, long_name, when)

    assert title is not None
    assert len(title) <= SessionDB.MAX_TITLE_LENGTH
    # Timestamp suffix is preserved — that's what makes titles unique per run.
    assert title.endswith(" · 2026-05-10 21:30:42")


def test_returns_none_when_session_db_missing():
    """No DB → no work, no exception."""
    when = dt.datetime(2026, 5, 10, 21, 30, 42)
    assert _seed()(None, "cron_x", "Watcher", when) is None


def test_swallows_db_errors(session_db, monkeypatch):
    """Title seeding is a UI nicety — must never propagate failures."""
    when = dt.datetime(2026, 5, 10, 21, 30, 42)

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated DB failure")

    monkeypatch.setattr(session_db, "create_session", boom)

    # No exception, just a None return.
    assert _seed()(session_db, "cron_x", "Watcher", when) is None


def test_seed_runs_before_prompt_assembly_failure_is_unaffected(session_db):
    """Document the wiring contract: seeding sits between session_id
    computation and any LLM/runtime-provider work. If the agent path later
    fails (auth, runtime provider, etc.), the title is already in the DB so
    the dashboard still gets a useful label rather than the
    ``[IMPORTANT: ...]`` preamble of the first user message.
    """
    when = dt.datetime(2026, 5, 10, 21, 30, 42)
    sid = "cron_jobX_20260510_213042"

    title = _seed()(session_db, sid, "Apple Notes Inbox Watcher", when)

    # Seeding succeeded independently of any downstream agent work.
    assert title == "Apple Notes Inbox Watcher · 2026-05-10 21:30:42"
    assert session_db.get_session_title(sid) == title
