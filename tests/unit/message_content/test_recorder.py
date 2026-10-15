"""Recorder: registers the content row and decides when a response gets labelled.

The database and the labelling model are patched out. Labelling runs in a
background task, so the tests await the tasks the recorder tracked.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.feedback.reply_labeller import LabellingCallFailed
from src.message_content import recorder
from src.pipeline.state import ContentKind, WorkerStatus

REGISTER_TARGET = "src.message_content.recorder.message_content.register"
SET_LABEL_TARGET = "src.message_content.recorder.message_content.set_response_label"
LABEL_RESPONSE_TARGET = "src.message_content.recorder.label_response"
LABEL_AND_STORE_TARGET = "src.message_content.recorder.label_and_store"


async def drain_background_tasks() -> None:
    await asyncio.gather(*list(recorder.BACKGROUND_TASKS))


class TestShouldLabel:
    def test_explicit_response_with_a_question_is_labelled(self):
        assert recorder.should_label(content_kind=ContentKind.RESPONSE, trigger="explicit", question="что?")

    @pytest.mark.parametrize("kind", [kind for kind in ContentKind if kind != ContentKind.RESPONSE])
    def test_other_kinds_are_not_labelled(self, kind):
        assert not recorder.should_label(content_kind=kind, trigger="explicit", question="что?")

    @pytest.mark.parametrize("trigger", ["random", "insult_check", "youtube_short", "-"])
    def test_non_explicit_triggers_are_not_labelled(self, trigger):
        assert not recorder.should_label(content_kind=ContentKind.RESPONSE, trigger=trigger, question="что?")

    @pytest.mark.parametrize("question", ["", "   "])
    def test_blank_question_is_not_labelled(self, question):
        assert not recorder.should_label(content_kind=ContentKind.RESPONSE, trigger="explicit", question=question)


class TestRecord:
    async def test_record_registers_the_row(self):
        with patch(REGISTER_TARGET, AsyncMock()) as register:
            await recorder.record(
                chat_id=1, message_id=2, content_kind=ContentKind.LIMIT_NOTICE, worker_status=None,
            )
        register.assert_awaited_once_with(
            chat_id=1, message_id=2, content_kind=ContentKind.LIMIT_NOTICE, worker_status=None,
        )

    async def test_explicit_response_spawns_labelling(self):
        with patch(REGISTER_TARGET, AsyncMock()), patch(LABEL_AND_STORE_TARGET, AsyncMock()) as label_and_store:
            await recorder.record(
                chat_id=1, message_id=2, content_kind=ContentKind.RESPONSE, worker_status=WorkerStatus.OK,
                trigger="explicit", question="когда выйдет GTA 6?", response="19 ноября",
            )
            await drain_background_tasks()
        label_and_store.assert_awaited_once_with(
            chat_id=1, message_id=2, question="когда выйдет GTA 6?", response="19 ноября",
        )

    async def test_other_kinds_spawn_no_labelling(self):
        with patch(REGISTER_TARGET, AsyncMock()), patch(LABEL_AND_STORE_TARGET, AsyncMock()) as label_and_store:
            await recorder.record(
                chat_id=1, message_id=2, content_kind=ContentKind.INSULT_REPLY, worker_status=WorkerStatus.SKIPPED,
                trigger="explicit", question="сам ты бот", response="сам такой",
            )
            await drain_background_tasks()
        label_and_store.assert_not_awaited()

    async def test_blank_question_is_not_labelled(self):
        """A voice note with no usable transcript has no question to judge the reply against."""
        with patch(REGISTER_TARGET, AsyncMock()), patch(LABEL_AND_STORE_TARGET, AsyncMock()) as label_and_store:
            await recorder.record(
                chat_id=1, message_id=2, content_kind=ContentKind.RESPONSE, worker_status=WorkerStatus.OK,
                trigger="explicit", question="  ", response="что-то",
            )
            await drain_background_tasks()
        label_and_store.assert_not_awaited()

    async def test_record_never_raises(self):
        with patch(REGISTER_TARGET, AsyncMock(side_effect=RuntimeError("db down"))):
            await recorder.record(
                chat_id=1, message_id=2, content_kind=ContentKind.ERROR, worker_status=None,
            )

    async def test_failed_register_still_skips_labelling(self):
        """No row, nothing to attach a label to."""
        with patch(REGISTER_TARGET, AsyncMock(side_effect=RuntimeError("db down"))), \
             patch(LABEL_AND_STORE_TARGET, AsyncMock()) as label_and_store:
            await recorder.record(
                chat_id=1, message_id=2, content_kind=ContentKind.RESPONSE, worker_status=WorkerStatus.OK,
                trigger="explicit", question="что?", response="это",
            )
            await drain_background_tasks()
        label_and_store.assert_not_awaited()


class TestLabelAndStore:
    async def test_stores_the_models_label(self):
        with patch(LABEL_RESPONSE_TARGET, AsyncMock(return_value="REFUSED")), \
             patch(SET_LABEL_TARGET, AsyncMock()) as set_label:
            await recorder.label_and_store(chat_id=1, message_id=2, question="как?", response="не скажу")
        set_label.assert_awaited_once_with(chat_id=1, message_id=2, label="REFUSED")

    async def test_out_of_contract_answer_stores_unclear(self):
        with patch(LABEL_RESPONSE_TARGET, AsyncMock(return_value=None)), \
             patch(SET_LABEL_TARGET, AsyncMock()) as set_label:
            await recorder.label_and_store(chat_id=1, message_id=2, question="как?", response="так")
        set_label.assert_awaited_once_with(chat_id=1, message_id=2, label="UNCLEAR")

    async def test_failed_call_stores_unclear(self):
        with patch(LABEL_RESPONSE_TARGET, AsyncMock(side_effect=LabellingCallFailed("no key"))), \
             patch(SET_LABEL_TARGET, AsyncMock()) as set_label:
            await recorder.label_and_store(chat_id=1, message_id=2, question="как?", response="так")
        set_label.assert_awaited_once_with(chat_id=1, message_id=2, label="UNCLEAR")

    async def test_label_and_store_never_raises(self):
        with patch(LABEL_RESPONSE_TARGET, AsyncMock(return_value="ANSWERED")), \
             patch(SET_LABEL_TARGET, AsyncMock(side_effect=RuntimeError("db down"))):
            await recorder.label_and_store(chat_id=1, message_id=2, question="как?", response="так")

    async def test_unexpected_labeller_bug_never_raises(self):
        with patch(LABEL_RESPONSE_TARGET, AsyncMock(side_effect=ValueError("bug"))), \
             patch(SET_LABEL_TARGET, AsyncMock()) as set_label:
            await recorder.label_and_store(chat_id=1, message_id=2, question="как?", response="так")
        set_label.assert_not_awaited()
