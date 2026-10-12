"""Reply labelling job: drains the pending queue, cheapest stage first.

Two invariants this pins. Rows that can never reach the model (any non-text
reply, and text with no content) must leave the queue by rule rather than sit
pending forever — attempts only increments on a model failure, so without a
rule label the queue would grow without bound. And a model failure must leave a
row pending with attempts bumped, never write a guessed label; once attempts is
exhausted the row is written UNCLEAR so it is counted honestly instead of
disappearing.
"""

from unittest.mock import AsyncMock, patch

from src.jobs.reply_labels import reply_labels_job, split_rows

FETCH = "src.jobs.reply_labels.reply_labels.fetch_pending"
WRITE = "src.jobs.reply_labels.reply_labels.write_label"
BUMP = "src.jobs.reply_labels.reply_labels.bump_attempts"
LABEL = "src.jobs.reply_labels.label_batch"


def make_row(
    *,
    reply_message_id: int,
    media_type: str = "text",
    reply_content: str = "бред",
    attempts: int = 0,
    trigger: str = "explicit",
) -> dict:
    return {
        "chat_id": 1000,
        "reply_message_id": reply_message_id,
        "bot_message_id": 500,
        "media_type": media_type,
        "reply_content": reply_content,
        "bot_content": "я вчера косил траву",
        "source": "pipeline",
        "trigger": trigger,
        "attempts": attempts,
    }


class TestSplitRows:
    def test_gif_goes_to_the_free_stage(self):
        free, text = split_rows([make_row(reply_message_id=1, media_type="animation")])
        assert [row["reply_message_id"] for row in free] == [1]
        assert text == []

    def test_sticker_goes_to_the_free_stage(self):
        free, text = split_rows([make_row(reply_message_id=1, media_type="sticker")])
        assert len(free) == 1
        assert text == []

    def test_voice_goes_to_the_free_stage(self):
        free, text = split_rows([make_row(reply_message_id=1, media_type="voice")])
        assert len(free) == 1
        assert text == []

    def test_photo_goes_to_the_free_stage(self):
        free, text = split_rows([make_row(reply_message_id=1, media_type="photo")])
        assert len(free) == 1
        assert text == []

    def test_text_goes_to_the_model_stage(self):
        free, text = split_rows([make_row(reply_message_id=1)])
        assert free == []
        assert [row["reply_message_id"] for row in text] == [1]

    def test_excluded_trigger_text_never_reaches_the_model_stage(self):
        free, text = split_rows([make_row(reply_message_id=1, trigger="social_link")])
        assert len(free) == 1
        assert text == []

    def test_youtube_short_is_not_excluded(self):
        """Only social_link is excluded; Shorts summaries are still measured."""
        free, text = split_rows([make_row(reply_message_id=1, trigger="youtube_short")])
        assert free == []
        assert [row["reply_message_id"] for row in text] == [1]

    def test_whitespace_only_text_goes_to_the_free_stage(self):
        free, text = split_rows([make_row(reply_message_id=1, reply_content="   ")])
        assert len(free) == 1
        assert text == []

    def test_empty_text_goes_to_the_free_stage(self):
        free, text = split_rows([make_row(reply_message_id=1, reply_content="")])
        assert len(free) == 1
        assert text == []


class TestReplyLabelsJob:
    async def test_gif_labelled_media_without_a_model_call(self):
        rows = [make_row(reply_message_id=1, media_type="animation")]
        with patch(FETCH, AsyncMock(return_value=rows)), \
             patch(WRITE, AsyncMock()) as write, \
             patch(LABEL, AsyncMock()) as label:
            await reply_labels_job(context=None)
        label.assert_not_awaited()
        write.assert_awaited_once_with(
            chat_id=1000, reply_message_id=1, reaction="MEDIA",
            cause=None, labelled_by="media_rule",
        )

    async def test_voice_labelled_media_like_any_other_wordless_reply(self):
        rows = [make_row(reply_message_id=1, media_type="voice")]
        with patch(FETCH, AsyncMock(return_value=rows)), \
             patch(WRITE, AsyncMock()) as write, \
             patch(LABEL, AsyncMock()):
            await reply_labels_job(context=None)
        write.assert_awaited_once_with(
            chat_id=1000, reply_message_id=1, reaction="MEDIA",
            cause=None, labelled_by="media_rule",
        )

    async def test_empty_text_labelled_unclear(self):
        rows = [make_row(reply_message_id=1, reply_content="   ")]
        with patch(FETCH, AsyncMock(return_value=rows)), \
             patch(WRITE, AsyncMock()) as write, \
             patch(LABEL, AsyncMock()) as label:
            await reply_labels_job(context=None)
        label.assert_not_awaited()
        write.assert_awaited_once_with(
            chat_id=1000, reply_message_id=1, reaction="UNCLEAR",
            cause=None, labelled_by="empty_text",
        )

    async def test_text_row_labelled_from_the_model(self):
        rows = [make_row(reply_message_id=1)]
        with patch(FETCH, AsyncMock(return_value=rows)), \
             patch(WRITE, AsyncMock()) as write, \
             patch(LABEL, AsyncMock(return_value={1: ("NEGATIVE", "INVENTED_LIFE")})):
            await reply_labels_job(context=None)
        write.assert_awaited_once_with(
            chat_id=1000, reply_message_id=1, reaction="NEGATIVE",
            cause="INVENTED_LIFE", labelled_by="model",
        )

    async def test_unlabelled_row_gets_its_attempts_bumped(self):
        rows = [make_row(reply_message_id=1)]
        with patch(FETCH, AsyncMock(return_value=rows)), \
             patch(WRITE, AsyncMock()) as write, \
             patch(LABEL, AsyncMock(return_value={})), \
             patch(BUMP, AsyncMock()) as bump:
            await reply_labels_job(context=None)
        write.assert_not_awaited()
        bump.assert_awaited_once_with(chat_id=1000, reply_message_ids=[1])

    async def test_last_attempt_is_written_unclear_not_bumped(self):
        """At the cap the row must leave the queue, counted as unlabelled."""
        rows = [make_row(reply_message_id=1, attempts=2)]
        with patch(FETCH, AsyncMock(return_value=rows)), \
             patch(WRITE, AsyncMock()) as write, \
             patch(LABEL, AsyncMock(return_value={})), \
             patch(BUMP, AsyncMock()) as bump:
            await reply_labels_job(context=None)
        bump.assert_not_awaited()
        write.assert_awaited_once_with(
            chat_id=1000, reply_message_id=1, reaction="UNCLEAR",
            cause=None, labelled_by="unlabelled",
        )

    async def test_excluded_trigger_labelled_skipped_without_a_model_call(self):
        rows = [make_row(reply_message_id=1, trigger="social_link")]
        with patch(FETCH, AsyncMock(return_value=rows)), \
             patch(WRITE, AsyncMock()) as write, \
             patch(LABEL, AsyncMock()) as label:
            await reply_labels_job(context=None)
        label.assert_not_awaited()
        write.assert_awaited_once_with(
            chat_id=1000, reply_message_id=1, reaction="SKIPPED",
            cause=None, labelled_by="excluded_trigger",
        )

    async def test_gif_on_an_excluded_trigger_is_skipped_not_media(self):
        """Excluding a trigger excludes it from every rate, media included."""
        rows = [make_row(reply_message_id=1, media_type="animation", trigger="social_link")]
        with patch(FETCH, AsyncMock(return_value=rows)), \
             patch(WRITE, AsyncMock()) as write, \
             patch(LABEL, AsyncMock()):
            await reply_labels_job(context=None)
        write.assert_awaited_once_with(
            chat_id=1000, reply_message_id=1, reaction="SKIPPED",
            cause=None, labelled_by="excluded_trigger",
        )

    async def test_fetch_failure_does_not_raise(self):
        with patch(FETCH, AsyncMock(side_effect=RuntimeError("db down"))):
            await reply_labels_job(context=None)  # must not raise

    async def test_no_pending_rows_makes_no_model_call(self):
        with patch(FETCH, AsyncMock(return_value=[])), \
             patch(LABEL, AsyncMock()) as label:
            await reply_labels_job(context=None)
        label.assert_not_awaited()
