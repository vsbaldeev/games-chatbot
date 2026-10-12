"""Label the same replies with each REPLY_LABEL_MODELS leg and report
disagreement.

High agreement does not prove correctness — two models can be wrong
identically — but disagreement reliably locates the ambiguous cases, which is
what a prompt revision needs to see. Established as LLM-as-a-jury.

Read-only: labels nothing into bot_reply_labels, writes nothing. Safe to run
against production whenever REPLY_LABEL_SYSTEM or the model list changes.

Usage:
    python -m scripts.compare_label_models --rows 40
"""

import argparse
import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src import config, log
from src.agent import ainvoke_with_backoff
from src.config.prompts import REPLY_LABEL_SYSTEM
from src.feedback.reply_labeller import build_batch_payload, parse_label_batch
from src.jobs.reply_labels import EXCLUDED_TRIGGERS
from src.store import db as database
from src.store import reply_labels

logger = log.get_logger(__name__)


async def label_with(model: str, rows: list[dict]) -> dict[int, tuple[str, str | None]]:
    """Label rows with one specific model, bypassing the fallback chain.

    Raises:
        Exception: Whatever the call raises. try_label_with is the caller that
            turns that into a reported failure instead of a crash.
    """
    llm = ChatOpenAI(
        model=model,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        temperature=0.0,
        max_tokens=len(rows) * config.REPLY_LABEL_MAX_TOKENS_PER_ITEM,
        max_retries=0,
    )
    response = await ainvoke_with_backoff(llm, [
        SystemMessage(content=REPLY_LABEL_SYSTEM),
        HumanMessage(content=build_batch_payload(rows)),
    ])
    expected_ids = {row["reply_message_id"] for row in rows}
    return parse_label_batch(response.content or "", expected_ids)


async def try_label_with(model: str, rows: list[dict]) -> dict[int, tuple[str, str | None]] | None:
    """Label rows with one leg, returning None instead of raising on failure.

    This script exists to tell the operator whether a model is usable at all,
    so a leg that 404s, 403s or saturates must be reported rather than take
    the whole comparison down with it — a crash says nothing about the other
    leg, which may well be answering fine.

    Args:
        model: The OpenRouter slug for this leg.
        rows: Pending rows to label.

    Returns:
        The leg's labels, or None when the call never produced any.
    """
    try:
        return await label_with(model, rows)
    except Exception as error:
        logger.warning("Leg %s failed, no labels from it: %s", model, error)
        return None


def report_disagreements(
    rows: list[dict],
    primary: dict[int, tuple[str, str | None]],
    secondary: dict[int, tuple[str, str | None]],
) -> None:
    """Print each reply where the two models assigned different reactions."""
    compared = 0
    disagreed = 0
    for row in rows:
        reply_id = row["reply_message_id"]
        first = primary.get(reply_id)
        second = secondary.get(reply_id)
        if first is None or second is None:
            continue
        compared += 1
        if first[0] == second[0]:
            continue
        disagreed += 1
        logger.info(
            "Disagreement on %s: %s vs %s — %r",
            reply_id, first[0], second[0], row["reply_content"][:80],
        )
    if compared == 0:
        logger.warning("No rows labelled by both models — nothing to compare")
        return
    logger.info(
        "Compared %d rows, %d disagreements, agreement %.2f",
        compared, disagreed, (compared - disagreed) / compared,
    )


async def compare_legs(rows: list[dict]) -> None:
    """Label rows with both legs and report as much as the legs allow.

    Degrades in three steps rather than crashing: both legs down says so, one
    leg down reports how much the surviving leg managed and that no comparison
    is possible, and two live legs get the full agreement figure.
    """
    primary_model, secondary_model = config.REPLY_LABEL_MODELS[:2]
    primary = await try_label_with(primary_model, rows)
    secondary = await try_label_with(secondary_model, rows)
    if primary is None and secondary is None:
        logger.warning("Both legs failed, no comparison possible")
        return
    if primary is None or secondary is None:
        failed = primary_model if primary is None else secondary_model
        survivor = secondary_model if primary is None else primary_model
        labelled = secondary if primary is None else primary
        logger.warning(
            "Leg %s failed, no comparison possible — leg %s labelled %d of %d rows",
            failed, survivor, len(labelled), len(rows),
        )
        return
    report_disagreements(rows, primary, secondary)


async def main() -> None:
    """Fetch pending rows, label them with each leg, report disagreement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=40, help="How many replies to compare")
    arguments = parser.parse_args()
    if len(config.REPLY_LABEL_MODELS) < 2:
        logger.warning("REPLY_LABEL_MODELS has fewer than two legs — nothing to compare")
        return
    await database.init()
    try:
        rows = [
            row for row in await reply_labels.fetch_pending(limit=arguments.rows * 3)
            if row["media_type"] == "text" and row["reply_content"].strip()
            and row["trigger"] not in EXCLUDED_TRIGGERS
        ][:arguments.rows]
        if not rows:
            logger.warning("No pending text replies to compare")
            return
        await compare_legs(rows)
    finally:
        await database.close()


if __name__ == "__main__":
    asyncio.run(main())
