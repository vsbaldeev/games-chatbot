"""post_life_episode — writes, sends and records one scheduled life-post episode.

Sending fans out to every known chat (mirrors ``src/jobs/meme.py``); the post
is only written to ``bot_memories`` canon once at least one chat actually
received it, so a fully failed send leaves the watermark untouched and
catch-up retries the slot later.
"""

import asyncio
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src import achievements, config, log
from src.agent.middleware import ainvoke_with_backoff, strip_thinking
from src.config.prompts import BOT_FACT_DISTILL_SYSTEM
from src.life.writer import Episode, episode_writer_agent
from src.store import bot_memories, embedder, unified_messages

logger = log.get_logger(__name__)

MAX_DISTILLED_FACTS = 3


async def post_life_episode(bot) -> None:
    """Write, send and record the next scheduled life-post episode.

    Args:
        bot: Telegram Bot instance used to send messages.
    """
    episode = await episode_writer_agent.write_episode()
    if episode is None:
        logger.warning("Life post skipped: episode writer produced nothing usable")
        return
    sent_count = await send_episode(bot, episode)
    if sent_count == 0:
        logger.warning("Life post skipped: failed to send to any chat")
        return
    await record_episode(episode)


async def send_episode(bot, episode: Episode) -> int:
    """Send the episode to every known chat.

    Args:
        bot: Telegram Bot instance used to send messages.
        episode: The episode to post.

    Returns:
        Number of chats the post was successfully sent to.
    """
    chat_ids = await achievements.get_all_chat_ids()
    results = await asyncio.gather(
        *[send_to_chat(bot, chat_id, episode) for chat_id in chat_ids],
        return_exceptions=True,
    )
    return sum(1 for result in results if result is True)


async def send_to_chat(bot, chat_id: int, episode: Episode) -> bool:
    """Send one episode to one chat and record it in unified_messages.

    Args:
        bot: Telegram Bot instance used to send the message.
        chat_id: Target chat.
        episode: The episode to post.

    Returns:
        True on success, False on any failure — never raises, so one
        chat's failure cannot abort the fan-out to the others.
    """
    try:
        sent = await bot.send_message(chat_id=chat_id, text=episode.episode_text)
        await unified_messages.insert(
            chat_id=chat_id,
            message_id=sent.message_id,
            user_id=bot.id,
            username=config.BOT_USERNAME,
            content=episode.episode_text,
            media_type="text",
            reply_to_msg_id=None,
        )
        return True
    except Exception as error:
        logger.warning("Life post failed for chat %s: %s", chat_id, error)
        return False


async def record_episode(episode: Episode) -> None:
    """Persist the posted episode and its distilled facts to bot_memories.

    Args:
        episode: The episode that was just successfully posted.
    """
    try:
        embedding = await embedder.embed(episode.episode_text)
        await bot_memories.insert_episode(
            content=episode.episode_text,
            post_format=episode.format,
            current_activity=episode.current_activity,
            embedding=embedding,
        )
        facts = await distill_facts(episode.episode_text)
        await bot_memories.upsert_facts(facts)
    except Exception as error:
        logger.warning("Failed to record posted episode: %s", error)


async def distill_facts(episode_text: str) -> list[str]:
    """Extract durable canon facts from a posted episode.

    MEMORY_MODEL is a reasoning model, so reasoning is disabled — otherwise
    the whole token budget burns inside a ``<think>`` block and no JSON
    answer is produced (same reasoning as ``memory_writer.make_extraction_llm``).

    Args:
        episode_text: The episode text to distill.

    Returns:
        Up to ``MAX_DISTILLED_FACTS`` fact strings, possibly empty.
    """
    llm = ChatGroq(
        model=config.MEMORY_MODEL, api_key=config.GROQ_API_KEY,
        temperature=0.2, max_tokens=256, max_retries=0,
        reasoning_effort="none",
    )
    result = await ainvoke_with_backoff(
        llm, [SystemMessage(content=BOT_FACT_DISTILL_SYSTEM), HumanMessage(content=episode_text)],
    )
    return parse_fact_array(result.content or "")


def parse_fact_array(raw: str) -> list[str]:
    """Parse a JSON array of fact strings, failing soft to an empty list.

    Args:
        raw: Raw model output, possibly wrapped in a think block.

    Returns:
        Up to ``MAX_DISTILLED_FACTS`` trimmed non-empty strings.
    """
    cleaned = strip_thinking(raw)
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        logger.warning("Fact distillation returned unparsable output: %.200s", cleaned)
        return []
    if not isinstance(data, list):
        return []
    facts = [str(item).strip() for item in data if str(item).strip()]
    return facts[:MAX_DISTILLED_FACTS]
