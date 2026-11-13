"""Local patch 2026-11-13: a failing no_agent script that wrote its own ⚠ sentence is delivered alone."""
from unittest.mock import patch

from cron import scheduler
from cron.scheduler import _script_own_alert


def test_own_sentence_passes_through():
    out = "Script exited with code 3\nstderr:\nrsync: noise\nstdout:\n⚠️ Last night's backup only half finished."
    assert _script_own_alert(out) == "⚠️ Last night's backup only half finished."


def test_other_failures_keep_the_wrapper():
    assert _script_own_alert("Script exited with code 1\nstderr:\nTraceback (most recent call last):") is None
    assert _script_own_alert("Script exited with code 1\nstdout:\nFAILED: thing") is None
    assert _script_own_alert("Script timed out after 960s: x.sh") is None


def test_run_no_agent_job_alert_and_failed_status(tmp_path):
    job = {"script": "x.sh"}
    fail = "Script exited with code 3\nstdout:\n⚠️ Backup stopped partway."
    with patch.object(scheduler, "_run_job_script_with_claim_heartbeat", return_value=(False, fail)):
        ok, doc, alert, err = scheduler._run_no_agent_job(job, "j1", "Nyx360 Daily Backup", None)
    assert ok is False and alert == "⚠️ Backup stopped partway." and "script failed" in doc
    raw = "Script exited with code 1\nstderr:\nboom"
    with patch.object(scheduler, "_run_job_script_with_claim_heartbeat", return_value=(False, raw)):
        ok, _, alert, _ = scheduler._run_no_agent_job(job, "j1", "Job", None)
    assert ok is False and alert.startswith("⚠ Cron watchdog 'Job' script failed")
