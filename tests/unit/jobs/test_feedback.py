"""Feedback metrics job: closes 15-minute windows, then rolls closed windows
up into hourly bot_feedback_metrics rows."""

from unittest.mock import AsyncMock, patch

from src.jobs.feedback import feedback_metrics_job

CLOSE_WINDOWS = "src.jobs.feedback.message_feedback.close_expired_windows"
ROLLUP = "src.jobs.feedback.feedback_metrics.recompute_recent_buckets"


class TestFeedbackMetricsJob:
    async def test_closes_windows_before_rolling_up(self):
        calls = []

        async def record_close():
            calls.append("close")
            return 3

        async def record_rollup(**kwargs):
            calls.append("rollup")
            return 2

        with patch(CLOSE_WINDOWS, side_effect=record_close), \
             patch(ROLLUP, side_effect=record_rollup):
            await feedback_metrics_job(context=None)
        assert calls == ["close", "rollup"]

    async def test_rollup_failure_does_not_raise(self):
        with patch(CLOSE_WINDOWS, AsyncMock(return_value=0)), \
             patch(ROLLUP, AsyncMock(side_effect=RuntimeError("db down"))):
            await feedback_metrics_job(context=None)  # must not raise
