"""Scheduled job: drain the pending reply-label queue.

Runs every 15 minutes (see src/app/jobs.py). The interval must stay well
inside feedback_metrics.recompute_recent_buckets' 3-hour lookback, or a label
arriving later than that lands in an hourly bucket nothing recomputes.

Two stages, cheapest first. The free stage resolves by rule: a reply to a bot
message on an excluded trigger is SKIPPED, every other non-text reply is MEDIA
(someone answered without words — the metric asserts nothing about what they
meant), and text with no content is UNCLEAR. The model stage batches the
remaining text rows.

The excluded-trigger check comes first, so a GIF reply to a Reels summary is
SKIPPED rather than counting toward media_response_rate. Excluded means
excluded from every rate.

Nothing is ever left permanently pending. A model failure bumps attempts and
retries next run; at the cap the row is written UNCLEAR so it leaves the queue
and is counted in unlabelled rather than quietly accumulating.

"A model failure" means the model answered and got this row wrong. A call that
never reached a model — OPENROUTER_API_KEY unset, OpenRouter unreachable, both
legs down — is a condition of the run, not of any row, and spends no attempts
at all. The missing key is checked once per run before any batch is sent, and
a failing call (LabellingCallFailed) ends the model stage with every remaining
row untouched. Without that, a 45-minute outage — three runs at this interval
— would retire the whole queue as UNCLEAR, and write_label's `reaction IS NULL`
guard makes that unrecoverable without hand-written SQL.
"""

from src import config, log
from src.feedback.reply_labeller import LabellingCallFailed, label_batch
from src.store import reply_labels

logger = log.get_logger(__name__)

ROWS_PER_RUN = 200

# Triggers whose replies are not measured at all. Reels and long-form YouTube
# summaries are not a signal the operator reads, so labelling replies to them
# would spend model calls for nothing. Filtered here rather than in
# track_reply, which does not know the trigger and would need a query back on
# the serving path to find out.
EXCLUDED_TRIGGERS: frozenset[str] = frozenset({"social_link"})

# (reaction, labelled_by) for rows the model never sees. cause is always null
# for these: a wordless reply states no reason, and neither does empty text.
MEDIA_LABEL = ("MEDIA", "media_rule")
EMPTY_TEXT_LABEL = ("UNCLEAR", "empty_text")
EXHAUSTED_LABEL = ("UNCLEAR", "unlabelled")
SKIPPED_LABEL = ("SKIPPED", "excluded_trigger")


def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split pending rows into the free stage and the model stage.

    Returns:
        (free, text). A text row with no content goes to the free stage, so
        an empty reply never costs a model call and never cycles forever.
    """
    free: list[dict] = []
    text: list[dict] = []
    for row in rows:
        excluded = row["trigger"] in EXCLUDED_TRIGGERS
        if not excluded and row["media_type"] == "text" and row["reply_content"].strip():
            text.append(row)
        else:
            free.append(row)
    return free, text


def free_label(row: dict) -> tuple[str, str]:
    """Pick the rule label for a row that never reaches the model.

    The excluded-trigger check is first on purpose: a wordless reply to an
    excluded bot message is SKIPPED, not MEDIA, because excluding a trigger
    excludes it from every rate rather than only the text ones.
    """
    if row["trigger"] in EXCLUDED_TRIGGERS:
        return SKIPPED_LABEL
    if row["media_type"] == "text":
        return EMPTY_TEXT_LABEL
    return MEDIA_LABEL


async def store_label(row: dict, reaction: str, cause: str | None, labelled_by: str) -> None:
    """Write one label for a pending row."""
    await reply_labels.write_label(
        chat_id=row["chat_id"], reply_message_id=row["reply_message_id"],
        reaction=reaction, cause=cause, labelled_by=labelled_by,
    )


async def run_free_stage(rows: list[dict]) -> int:
    """Label every rule-resolvable row. Returns the number labelled."""
    for row in rows:
        reaction, labelled_by = free_label(row)
        await store_label(row, reaction, None, labelled_by)
    return len(rows)


async def handle_unlabelled(rows: list[dict]) -> int:
    """Retry rows that still have attempts left; retire the rest as UNCLEAR.

    Returns:
        How many rows were retired rather than left for another attempt.
    """
    retry: dict[int, list[int]] = {}
    retired = 0
    for row in rows:
        if row["attempts"] + 1 < reply_labels.MAX_ATTEMPTS:
            retry.setdefault(row["chat_id"], []).append(row["reply_message_id"])
            continue
        await store_label(row, EXHAUSTED_LABEL[0], None, EXHAUSTED_LABEL[1])
        retired += 1
    for chat_id, ids in retry.items():
        await reply_labels.bump_attempts(chat_id=chat_id, reply_message_ids=ids)
    return retired


async def run_model_stage(rows: list[dict]) -> tuple[int, int]:
    """Label text rows in batches. Returns (labelled, retired unlabelled).

    A LabellingCallFailed ends the stage rather than moving on to the next
    batch: the call never reached a model, so no row in this batch learned
    anything and its attempts stay as they were, and the remaining batches
    would almost certainly fail the same way within the same run. They are
    left pending with attempts untouched for the next run to pick up.
    """
    labelled = 0
    retired = 0
    for start in range(0, len(rows), config.REPLY_LABEL_BATCH_SIZE):
        batch = rows[start:start + config.REPLY_LABEL_BATCH_SIZE]
        try:
            labels = await label_batch(batch)
        except LabellingCallFailed as error:
            logger.warning(
                "Reply labelling: call failed, %d rows stay pending with attempts "
                "untouched: %s", len(rows) - start, error,
            )
            return labelled, retired
        missed: list[dict] = []
        for row in batch:
            label = labels.get(row["reply_message_id"])
            if label is None:
                missed.append(row)
                continue
            await store_label(row, label[0], label[1], "model")
            labelled += 1
        retired += await handle_unlabelled(missed)
    return labelled, retired


async def reply_labels_job(context) -> None:
    """Label as many pending replies as the per-run cap allows.

    Args:
        context: Telegram JobQueue callback context (unused).
    """
    try:
        rows = await reply_labels.fetch_pending(limit=ROWS_PER_RUN)
    except Exception as error:
        logger.warning("Failed to fetch pending reply labels: %s", error)
        return
    if not rows:
        return
    free, text = split_rows(rows)
    if text and not config.OPENROUTER_API_KEY:
        # Checked once here rather than per batch: a missing key is a property
        # of the deployment, and letting each batch discover it separately
        # would charge every text row an attempt for it.
        logger.warning(
            "Reply labelling: OPENROUTER_API_KEY unset — skipping the model stage, "
            "%d text rows stay pending with attempts untouched", len(text),
        )
        text = []
    try:
        by_rule = await run_free_stage(free)
        by_model, retired = await run_model_stage(text)
    except Exception as error:
        logger.warning("Reply labelling run failed partway: %s", error)
        return
    logger.info(
        "Reply labels: %d by rule, %d by model, %d retired unlabelled",
        by_rule, by_model, retired,
    )
