"""Writing bot_message_content rows and scheduling response labelling.

record() runs on the delivery path, so it never raises and never waits for a
model: the row is written inline, the labelling call runs in a background
task. Only free-form replies to an explicit question are labelled — canned
kinds are already exact, and directive-driven replies (insult, wind-down) or
link summaries have no question to judge the reply against.
"""

import asyncio

from src import log
from src.feedback.reply_labeller import LabellingCallFailed
from src.message_content.response_labeller import label_response
from src.pipeline.state import ContentKind, WorkerStatus
from src.store import message_content

logger = log.get_logger(__name__)

# Strong references to in-flight labelling tasks: the event loop only keeps a
# weak one, so an unreferenced task can be garbage-collected mid-call.
BACKGROUND_TASKS: set[asyncio.Task] = set()

UNCLEAR = "UNCLEAR"


def should_label(*, content_kind: ContentKind, trigger: str, question: str) -> bool:
    """True only for a free-form response to an explicit, non-blank question."""
    return content_kind == ContentKind.RESPONSE and trigger == "explicit" and bool(question.strip())


async def label_and_store(*, chat_id: int, message_id: int, question: str, response: str) -> None:
    """Label one delivered response and store the verdict. Never raises.

    A failed call or an answer outside the contract is stored as UNCLEAR, so
    it stays distinguishable from a message that is simply not labelled yet.
    """
    try:
        try:
            label = await label_response(question=question, response=response)
        except LabellingCallFailed as error:
            logger.warning("Response labelling call failed for message %s: %s", message_id, error)
            label = None
        await message_content.set_response_label(
            chat_id=chat_id, message_id=message_id, label=label or UNCLEAR,
        )
    except Exception as error:
        logger.warning("Failed to label response %s: %s", message_id, error)


async def record(
    *,
    chat_id: int,
    message_id: int,
    content_kind: ContentKind,
    worker_status: WorkerStatus | None,
    trigger: str = "-",
    question: str = "",
    response: str = "",
) -> None:
    """Register the content row for one bot message and schedule labelling if it applies.

    Never raises — a metrics write must not affect message delivery.

    Args:
        chat_id: Chat the message was sent in.
        message_id: The sent (or edited) message's id.
        content_kind: Why the message exists.
        worker_status: How the worker stage ended, or None if it never finished.
        trigger: Pipeline response_trigger; only "explicit" replies are labelled.
        question: The member's question, for the labeller.
        response: The delivered text, for the labeller.
    """
    try:
        await message_content.register(
            chat_id=chat_id, message_id=message_id, content_kind=content_kind, worker_status=worker_status,
        )
    except Exception as error:
        logger.warning("Failed to register content row for message %s: %s", message_id, error)
        return
    if not should_label(content_kind=content_kind, trigger=trigger, question=question):
        return
    task = asyncio.create_task(
        label_and_store(chat_id=chat_id, message_id=message_id, question=question, response=response)
    )
    BACKGROUND_TASKS.add(task)
    task.add_done_callback(BACKGROUND_TASKS.discard)
