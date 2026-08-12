"""Caption composition and the 1024-character fitting ladder."""

from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from src.events.link_repost import (
    CAPTION_LIMIT,
    TEXT_LIMIT,
    build_caption,
    deliver_link_message,
    fit_caption,
    resolve_link_delivery,
    truncate_at_sentence,
)
from tests.builders import make_incoming, make_state

COMPRESS_PATCH_TARGET = "src.events.link_repost.compress_to_budget"


class TestBuildCaption:
    def test_credit_and_link_lead_the_caption(self):
        caption = build_caption("Про котиков.", "vasya", "https://youtu.be/abc")
        assert caption == "Скинул @vasya\nhttps://youtu.be/abc\n\nПро котиков."

    def test_summary_alone_when_the_original_survives(self):
        assert build_caption("Про котиков.", None, None) == "Про котиков."

    def test_summary_alone_when_only_one_of_the_pair_is_given(self):
        assert build_caption("Про котиков.", "vasya", None) == "Про котиков."
        assert build_caption("Про котиков.", None, "https://youtu.be/abc") == "Про котиков."


class TestTruncateAtSentence:
    def test_short_text_is_untouched(self):
        assert truncate_at_sentence("Коротко.", 100) == "Коротко."

    def test_cuts_at_the_last_sentence_boundary(self):
        text = "Первое предложение. Второе предложение. Третье."
        assert truncate_at_sentence(text, 30) == "Первое предложение."

    def test_falls_back_to_ellipsis_without_a_boundary(self):
        result = truncate_at_sentence("а" * 100, 10)
        assert len(result) <= 10
        assert result.endswith("…")

    def test_non_positive_budget_yields_empty_text(self):
        assert truncate_at_sentence("что угодно", 0) == ""


class TestFitCaption:
    async def test_within_budget_skips_compression(self):
        compress = AsyncMock()
        with patch(COMPRESS_PATCH_TARGET, new=compress):
            caption = await fit_caption(
                "Коротко.", "vasya", "https://youtu.be/abc", has_video=True,
            )
        compress.assert_not_awaited()
        assert caption.endswith("Коротко.")

    async def test_overflow_is_compressed(self):
        long_summary = "а" * 1200
        with patch(COMPRESS_PATCH_TARGET, new=AsyncMock(return_value="Сжато.")):
            caption = await fit_caption(
                long_summary, "vasya", "https://youtu.be/abc", has_video=True,
            )
        assert caption.endswith("Сжато.")
        assert len(caption) <= CAPTION_LIMIT

    async def test_compressor_gets_the_budget_minus_overhead(self):
        compress = AsyncMock(return_value="Сжато.")
        with patch(COMPRESS_PATCH_TARGET, new=compress):
            await fit_caption("а" * 1200, "vasya", "https://youtu.be/abc", has_video=True)
        budget = compress.await_args.args[1]
        assert 0 < budget < CAPTION_LIMIT

    async def test_still_over_budget_falls_back_to_truncation(self):
        with patch(COMPRESS_PATCH_TARGET, new=AsyncMock(return_value="б" * 1200)):
            caption = await fit_caption(
                "а" * 1200, "vasya", "https://youtu.be/abc", has_video=True,
            )
        assert len(caption) <= CAPTION_LIMIT

    async def test_overflow_without_credit_still_fits(self):
        with patch(COMPRESS_PATCH_TARGET, new=AsyncMock(return_value="б" * 1200)):
            caption = await fit_caption("а" * 1200, None, None, has_video=True)
        assert len(caption) <= CAPTION_LIMIT

    async def test_absurd_url_overhead_still_fits(self):
        with patch(COMPRESS_PATCH_TARGET, new=AsyncMock(return_value="Сжато.")):
            caption = await fit_caption(
                "Про котиков.", "vasya", "https://example.com/" + "a" * 1200,
                has_video=True,
            )
        assert len(caption) <= CAPTION_LIMIT

    async def test_text_only_send_uses_the_larger_text_limit(self):
        long_summary = "а" * 1200
        compress = AsyncMock()
        with patch(COMPRESS_PATCH_TARGET, new=compress):
            caption = await fit_caption(long_summary, None, None, has_video=False)
        compress.assert_not_awaited()
        assert len(caption) <= TEXT_LIMIT

    async def test_text_only_overflow_past_4096_is_still_compressed(self):
        long_summary = "а" * 4200
        with patch(COMPRESS_PATCH_TARGET, new=AsyncMock(return_value="Сжато.")):
            caption = await fit_caption(long_summary, None, None, has_video=False)
        assert len(caption) <= TEXT_LIMIT


def make_msg() -> MagicMock:
    msg = MagicMock()
    msg.chat_id = 1000
    msg.message_id = 55
    msg.is_topic_message = False
    msg.delete = AsyncMock()
    msg.reply_video = AsyncMock(return_value=MagicMock(message_id=901))
    msg.reply_text = AsyncMock(return_value=MagicMock(message_id=902))
    msg.chat.send_video = AsyncMock(return_value=MagicMock(message_id=903))
    msg.chat.send_message = AsyncMock(return_value=MagicMock(message_id=904))
    return msg


class TestDeliverLinkMessageBare:
    async def test_sends_unanchored_video_and_deletes_the_original(self):
        msg = make_msg()
        sent_id, anchored_to, media_type = await deliver_link_message(
            msg, summary="Про котиков.", video=b"bytes", username="vasya",
            url="https://youtu.be/abc", is_bare=True,
        )
        msg.chat.send_video.assert_awaited_once()
        msg.reply_video.assert_not_awaited()
        msg.delete.assert_awaited_once()
        assert (sent_id, anchored_to, media_type) == (903, None, "video")

    async def test_caption_carries_credit_and_link(self):
        msg = make_msg()
        await deliver_link_message(
            msg, summary="Про котиков.", video=b"bytes", username="vasya",
            url="https://youtu.be/abc", is_bare=True,
        )
        caption = msg.chat.send_video.await_args.kwargs["caption"]
        assert "@vasya" in caption and "https://youtu.be/abc" in caption

    async def test_no_video_sends_an_unanchored_text_message(self):
        msg = make_msg()
        sent_id, anchored_to, media_type = await deliver_link_message(
            msg, summary="Про котиков.", video=None, username="vasya",
            url="https://youtu.be/abc", is_bare=True,
        )
        msg.chat.send_message.assert_awaited_once()
        msg.delete.assert_awaited_once()
        assert (sent_id, anchored_to, media_type) == (904, None, "text")

    async def test_send_failure_keeps_the_original_and_replies_with_text(self):
        msg = make_msg()
        msg.chat.send_video = AsyncMock(side_effect=RuntimeError("too big"))
        sent_id, anchored_to, media_type = await deliver_link_message(
            msg, summary="Про котиков.", video=b"bytes", username="vasya",
            url="https://youtu.be/abc", is_bare=True,
        )
        msg.delete.assert_not_awaited()
        msg.reply_text.assert_awaited_once_with("Про котиков.")
        assert (sent_id, anchored_to, media_type) == (902, 55, "text")

    async def test_delete_failure_is_swallowed(self):
        msg = make_msg()
        msg.delete = AsyncMock(side_effect=RuntimeError("not an admin"))
        sent_id, anchored_to, media_type = await deliver_link_message(
            msg, summary="Про котиков.", video=b"bytes", username="vasya",
            url="https://youtu.be/abc", is_bare=True,
        )
        assert (sent_id, anchored_to, media_type) == (903, None, "video")

    async def test_unanchored_send_carries_the_forum_topic_thread(self):
        msg = make_msg()
        msg.is_topic_message = True
        msg.message_thread_id = 77
        await deliver_link_message(
            msg, summary="Про котиков.", video=b"bytes", username="vasya",
            url="https://youtu.be/abc", is_bare=True,
        )
        msg.chat.send_video.assert_awaited_once_with(
            video=ANY, caption=ANY, message_thread_id=77,
        )

    async def test_missing_url_keeps_the_original_despite_is_bare_true(self):
        msg = make_msg()
        sent_id, anchored_to, media_type = await deliver_link_message(
            msg, summary="Про котиков.", video=b"bytes", username="vasya",
            url=None, is_bare=True,
        )
        msg.reply_video.assert_awaited_once()
        msg.chat.send_video.assert_not_awaited()
        msg.delete.assert_not_awaited()
        assert (sent_id, anchored_to, media_type) == (901, 55, "video")


class TestDeliverLinkMessageNotBare:
    async def test_replies_with_video_and_keeps_the_original(self):
        msg = make_msg()
        sent_id, anchored_to, media_type = await deliver_link_message(
            msg, summary="Про котиков.", video=b"bytes", username="vasya",
            url="https://youtu.be/abc", is_bare=False,
        )
        msg.reply_video.assert_awaited_once()
        msg.chat.send_video.assert_not_awaited()
        msg.delete.assert_not_awaited()
        assert (sent_id, anchored_to, media_type) == (901, 55, "video")

    async def test_caption_omits_credit_and_link(self):
        msg = make_msg()
        await deliver_link_message(
            msg, summary="Про котиков.", video=b"bytes", username="vasya",
            url="https://youtu.be/abc", is_bare=False,
        )
        assert msg.reply_video.await_args.kwargs["caption"] == "Про котиков."

    async def test_send_failure_keeps_the_original_and_replies_with_text(self):
        msg = make_msg()
        msg.reply_video = AsyncMock(side_effect=RuntimeError("upload failed"))
        sent_id, anchored_to, media_type = await deliver_link_message(
            msg, summary="Про котиков.", video=b"bytes", username="vasya",
            url="https://youtu.be/abc", is_bare=False,
        )
        msg.delete.assert_not_awaited()
        msg.reply_text.assert_awaited_once_with("Про котиков.")
        assert (sent_id, anchored_to, media_type) == (902, 55, "text")

    async def test_text_only_send_failure_propagates_instead_of_retrying(self):
        msg = make_msg()
        msg.reply_text = AsyncMock(side_effect=RuntimeError("flood control"))
        with pytest.raises(RuntimeError, match="flood control"):
            await deliver_link_message(
                msg, summary="Про котиков.", video=None, username="vasya",
                url="https://youtu.be/abc", is_bare=False,
            )
        msg.delete.assert_not_awaited()
        msg.reply_text.assert_awaited_once_with("Про котиков.")


class TestResolveLinkDelivery:
    def test_shorts_trigger_with_content_resolves(self):
        state = make_state(
            make_incoming(), response_trigger="youtube_short",
            youtube_short_content="блок", youtube_short_video=b"bytes",
            youtube_short_url="https://www.youtube.com/shorts/abc",
        )
        assert resolve_link_delivery(state) == (
            b"bytes", "https://www.youtube.com/shorts/abc",
        )

    def test_social_link_trigger_with_content_resolves(self):
        state = make_state(
            make_incoming(), response_trigger="social_link",
            social_link_content="блок", social_link_video=None,
            social_link_url="https://youtu.be/abc",
        )
        assert resolve_link_delivery(state) == (None, "https://youtu.be/abc")

    def test_failed_fetch_does_not_resolve(self):
        state = make_state(make_incoming(), response_trigger="social_link")
        assert resolve_link_delivery(state) is None

    def test_ordinary_trigger_does_not_resolve(self):
        state = make_state(make_incoming(), response_trigger="explicit")
        assert resolve_link_delivery(state) is None
