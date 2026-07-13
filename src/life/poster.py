"""post_life_episode — writes, sends and records one scheduled life-post episode.

Sending fans out to every known chat (mirrors ``src/jobs/meme.py``); the post
is only written to ``bot_memories`` canon once at least one chat actually
received it, so a fully failed send leaves the watermark untouched and
catch-up retries the slot later.
"""

import asyncio
import dataclasses
import io
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src import achievements, config, log
from src.agent.middleware import ainvoke_with_backoff, strip_thinking
from src.config.prompts import BOT_FACT_DISTILL_SYSTEM, CHARACTER_VISUAL_PROMPT
from src.imagegen import generate_image
from src.life.writer import (
    ALL_FORMATS,
    PHOTO_FORMAT,
    STORY_FORMAT,
    VOICE_FORMAT,
    Episode,
    episode_writer_agent,
)
from src.store import bot_memories, embedder, unified_messages
from src.tts import SynthesizedVoice, prepare_tts_text, speech_service

logger = log.get_logger(__name__)

MAX_DISTILLED_FACTS = 3


async def post_life_episode(bot) -> None:
    """Write, send and record the next scheduled life-post episode.

    Args:
        bot: Telegram Bot instance used to send messages.
    """
    episode = await episode_writer_agent.write_episode(live_formats())
    if episode is None:
        logger.warning("Life post skipped: episode writer produced nothing usable")
        return
    episode, media = await resolve_media(episode)
    sent_count = await send_episode(bot, episode, media)
    if sent_count == 0:
        logger.warning("Life post skipped: failed to send to any chat")
        return
    await record_episode(episode)


def live_formats() -> tuple[str, ...]:
    """Return the formats currently offered to the episode writer.

    The photo format needs the imagegen service; without ``IMAGEGEN_URL``
    it is not offered at all rather than degrading on every draw.

    Returns:
        ``ALL_FORMATS``, minus ``photo`` when imagegen is not configured.
    """
    if config.IMAGEGEN_URL:
        return ALL_FORMATS
    return tuple(post_format for post_format in ALL_FORMATS if post_format != PHOTO_FORMAT)


@dataclasses.dataclass(frozen=True)
class EpisodeMedia:
    """Media payload built once per episode and reused across the chat fan-out.

    Attributes:
        voice: Synthesized voice payload for voice posts, or None.
        photo_png: Generated PNG bytes for photo posts, or None.
    """

    voice: SynthesizedVoice | None = None
    photo_png: bytes | None = None


async def resolve_media(episode: Episode) -> tuple[Episode, EpisodeMedia]:
    """Build the episode's media payload, degrading to story on failure.

    The payload is built once here and reused across the whole chat
    fan-out. A demoted episode keeps the degraded format, so the recorded
    canon (and the never-repeat-format rule) reflects what was actually
    posted. A media failure demotes the post, never kills it.

    Args:
        episode: The freshly written episode.

    Returns:
        The episode paired with its media, or the episode demoted to the
        ``story`` format paired with empty media when the media build
        failed (text posts pass through with empty media).
    """
    if episode.format == VOICE_FORMAT:
        voice = await build_voice_payload(episode.voice_script)
        if voice is not None:
            return episode, EpisodeMedia(voice=voice)
        logger.warning("Voice media build failed — degrading life post to a text story")
    if episode.format == PHOTO_FORMAT:
        photo_png = await generate_image(f"{CHARACTER_VISUAL_PROMPT}, {episode.image_prompt}")
        if photo_png is not None:
            return episode, EpisodeMedia(photo_png=photo_png)
        logger.warning("Image generation failed — degrading life post to a text story")
    if episode.format == STORY_FORMAT:
        return episode, EpisodeMedia()
    return dataclasses.replace(episode, format=STORY_FORMAT), EpisodeMedia()


async def build_voice_payload(voice_script: str) -> SynthesizedVoice | None:
    """Synthesize the spoken story for a voice life post.

    Args:
        voice_script: The episode's spoken story text.

    Returns:
        The synthesized payload, or None when the TTS service is not ready,
        the script is unspeakable (the ``prepare_tts_text`` contract), or
        synthesis failed — never raises.
    """
    if not speech_service.is_ready:
        return None
    prepared_text = prepare_tts_text(voice_script)
    if prepared_text is None:
        return None
    return await speech_service.synthesize(prepared_text)


async def send_episode(bot, episode: Episode, media: EpisodeMedia) -> int:
    """Send the episode to every known chat.

    Args:
        bot: Telegram Bot instance used to send messages.
        episode: The episode to post.
        media: Prebuilt media payload shared across the fan-out.

    Returns:
        Number of chats the post was successfully sent to.
    """
    chat_ids = await achievements.get_all_chat_ids()
    results = await asyncio.gather(
        *[send_to_chat(bot, chat_id, episode, media) for chat_id in chat_ids],
        return_exceptions=True,
    )
    return sum(1 for result in results if result is True)


async def send_to_chat(bot, chat_id: int, episode: Episode, media: EpisodeMedia) -> bool:
    """Send one episode to one chat and record it in unified_messages.

    A voice post shows only the teaser caption, but ``unified_messages``
    records the full ``episode_text`` — the bot's own posts never need
    transcription when a member replies to them.

    Args:
        bot: Telegram Bot instance used to send the message.
        chat_id: Target chat.
        episode: The episode to post.
        media: Prebuilt media payload shared across the fan-out.

    Returns:
        True on success, False on any failure — never raises, so one
        chat's failure cannot abort the fan-out to the others.
    """
    try:
        sent, media_type = await send_media(bot, chat_id, episode, media)
        await record_sent_message(bot, chat_id, sent, media_type, episode)
        return True
    except Exception as error:
        logger.warning("Life post failed for chat %s: %s", chat_id, error)
        return False


async def send_media(bot, chat_id: int, episode: Episode, media: EpisodeMedia) -> tuple:
    """Send the episode's Telegram message in its format to one chat.

    Args:
        bot: Telegram Bot instance used to send the message.
        chat_id: Target chat.
        episode: The episode to post.
        media: Prebuilt media payload shared across the fan-out.

    Returns:
        ``(sent_message, media_type)`` for the ``unified_messages`` record.
    """
    if media.voice is not None:
        sent = await bot.send_voice(
            chat_id=chat_id,
            voice=io.BytesIO(media.voice.ogg_bytes),
            duration=media.voice.duration_seconds,
            caption=episode.voice_teaser,
        )
        return sent, "voice"
    if media.photo_png is not None:
        sent = await bot.send_photo(
            chat_id=chat_id, photo=media.photo_png, caption=episode.episode_text
        )
        return sent, "photo"
    sent = await bot.send_message(chat_id=chat_id, text=episode.episode_text)
    return sent, "text"


async def record_sent_message(bot, chat_id: int, sent, media_type: str, episode: Episode) -> None:
    """Record one sent life post in unified_messages.

    Photo posts store placeholder content plus the Telegram ``file_id``,
    which plugs the bot's selfies into the existing lazy vision-description
    path — a member replying to the selfie gets a real description of the
    generated frame, same as for member photos.

    Args:
        bot: Telegram Bot instance (source of the bot's id).
        chat_id: Chat the message was sent to.
        sent: The sent ``telegram.Message``.
        media_type: ``unified_messages`` media type of the sent message.
        episode: The posted episode.
    """
    content = episode.episode_text
    file_id = None
    if media_type == "photo":
        content = unified_messages.format_photo_content(episode.episode_text)
        file_id = sent.photo[-1].file_id if sent.photo else None
    await unified_messages.insert(
        chat_id=chat_id,
        message_id=sent.message_id,
        user_id=bot.id,
        username=config.BOT_USERNAME,
        content=content,
        media_type=media_type,
        reply_to_msg_id=None,
        file_id=file_id,
    )


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
