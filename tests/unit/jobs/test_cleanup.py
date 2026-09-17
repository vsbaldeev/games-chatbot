"""Daily cleanup job tests: every store's cleanup_old is called, in an order
that never deletes a still-referenced row (bot_llm_log before
llm_system_prompts)."""

from unittest.mock import AsyncMock, patch

from src.jobs.cleanup import cleanup_messages_job

PATCH_TARGETS = {
    "messages": "src.jobs.cleanup.unified_messages.cleanup_old",
    "history": "src.jobs.cleanup.thread_history.cleanup_old",
    "facts": "src.jobs.cleanup.user_memories.cleanup_stale",
    "llm_log": "src.jobs.cleanup.llm_log.cleanup_old",
    "prompts": "src.jobs.cleanup.llm_system_prompts.cleanup_unreferenced",
    "feedback": "src.jobs.cleanup.message_feedback.cleanup_old",
    "metrics": "src.jobs.cleanup.feedback_metrics.cleanup_old",
}


class TestCleanupMessagesJob:
    async def test_calls_every_stores_cleanup(self):
        with patch(PATCH_TARGETS["messages"], AsyncMock(return_value=1)), \
             patch(PATCH_TARGETS["history"], AsyncMock(return_value=1)), \
             patch(PATCH_TARGETS["facts"], AsyncMock(return_value=1)), \
             patch(PATCH_TARGETS["llm_log"], AsyncMock(return_value=1)) as llm_log_cleanup, \
             patch(PATCH_TARGETS["prompts"], AsyncMock(return_value=1)) as prompts_cleanup, \
             patch(PATCH_TARGETS["feedback"], AsyncMock(return_value=1)) as feedback_cleanup, \
             patch(PATCH_TARGETS["metrics"], AsyncMock(return_value=1)) as metrics_cleanup:
            await cleanup_messages_job(context=None)
        llm_log_cleanup.assert_awaited_once()
        prompts_cleanup.assert_awaited_once()
        feedback_cleanup.assert_awaited_once()
        metrics_cleanup.assert_awaited_once()

    async def test_llm_log_pruned_before_system_prompts(self):
        """A prompt must not be dropped while a bot_llm_log row still cites it."""
        call_order = []

        async def record_llm_log():
            call_order.append("llm_log")
            return 1

        async def record_prompts():
            call_order.append("prompts")
            return 1

        with patch(PATCH_TARGETS["messages"], AsyncMock(return_value=1)), \
             patch(PATCH_TARGETS["history"], AsyncMock(return_value=1)), \
             patch(PATCH_TARGETS["facts"], AsyncMock(return_value=1)), \
             patch(PATCH_TARGETS["llm_log"], side_effect=record_llm_log), \
             patch(PATCH_TARGETS["prompts"], side_effect=record_prompts), \
             patch(PATCH_TARGETS["feedback"], AsyncMock(return_value=1)), \
             patch(PATCH_TARGETS["metrics"], AsyncMock(return_value=1)):
            await cleanup_messages_job(context=None)
        assert call_order == ["llm_log", "prompts"]
