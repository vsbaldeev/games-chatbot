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

from src import achievements, config, log
from src.agent.middleware import ainvoke_with_backoff, strip_thinking
from src.config.prompts import BOT_FACT_DISTILL_SYSTEM, CHARACTER_VISUAL_PROMPT, PHOTO_FRAMING_HINT
from src.imagegen import generate_image
from src.life.photo_judge import score_photo
from src.life.writer import (
    PHOTO_FORMAT,
    STORY_FORMAT,
    VOICE_FORMAT,
    Episode,
    episode_writer_agent,
)
from src.pipeline.memory_writer import make_extraction_llm
from src.store import bot_memories, embedder, unified_messages
from src.tts import SynthesizedVoice, prepare_tts_text, speech_service

logger = log.get_logger(__name__)

MAX_DISTILLED_FACTS = 3

# Rank of a candidate the vision judge could not score: below every real 0-10
# score, so it loses to any scored candidate but still ships when it is the
# only one that rendered. "Unknown", never "bad" — a judge outage must not
# look like a mismatch.
UNSCORED_RANK = -1

# One scheduled post at a time, module-global like selfie.py's generation slot.
# A photo post spends ~16 minutes rendering candidates, and the watermark that
# tells catch-up "this slot is done" (bot_memories.get_latest_posted_at) is only
# written after a successful send — so for those minutes the slot still looks
# unposted. Without this guard the daily 17:00 job and the startup catch-up
# (which runs 60 s after boot) both claim the same slot and post two
# near-identical episodes, each with its own image. Deploying in the minute
# before a slot is exactly what triggers it.
post_in_flight = False


def is_post_in_flight() -> bool:
    """Peek whether a scheduled life post is currently being produced.

    Synchronous read used by the filter node, so a photo request arriving
    mid-post is acked «уже фоткаю» instead of starting a second generation.

    Returns:
        True while :func:`post_life_episode` is producing an episode.
    """
    return post_in_flight


async def post_life_episode(bot, post_format: str) -> None:
    """Write, send and record one scheduled life-post episode.

    Drops the call when a post is already in flight: both triggers (the daily
    job and the startup catch-up) target the same slot, so a concurrent
    second one would duplicate it. Queueing it instead would not help — it
    would still be the same slot's episode, posted twice.

    Args:
        bot: Telegram Bot instance used to send messages.
        post_format: Format this slot posts in, assigned by the weekly
            schedule (``src/jobs/life_post.py``) — the writer is told which
            format to write for, it does not choose one.
    """
    global post_in_flight
    if post_in_flight:
        logger.warning("Life post already in flight — dropping duplicate %s post", post_format)
        return
    post_in_flight = True
    try:
        await publish_episode(bot, post_format)
    finally:
        post_in_flight = False


async def publish_episode(bot, post_format: str) -> None:
    """Write, send and record the episode; the guarded body of a life post.

    Args:
        bot: Telegram Bot instance used to send messages.
        post_format: Format this slot posts in.
    """
    episode = await episode_writer_agent.write_episode(supported_format(post_format))
    if episode is None:
        logger.warning("Life post skipped: episode writer produced nothing usable")
        return
    episode, media = await resolve_media(episode)
    sent_count = await send_episode(bot, episode, media)
    if sent_count == 0:
        logger.warning("Life post skipped: failed to send to any chat")
        return
    logger.info("Life post sent as %s to %d chat(s)", episode.format, sent_count)
    await record_episode(episode)


def supported_format(post_format: str) -> str:
    """Return the scheduled format, downgraded when its backend is unavailable.

    The photo format needs the imagegen service; without ``IMAGEGEN_URL``
    the writer is asked for a text story up front rather than writing for a
    photo that :func:`resolve_media` would then have to demote.

    Args:
        post_format: Format the weekly schedule assigned to this slot.

    Returns:
        ``post_format``, or ``story`` when it is ``photo`` and imagegen is
        not configured.
    """
    if post_format == PHOTO_FORMAT and not config.IMAGEGEN_URL:
        logger.warning("IMAGEGEN_URL is not set — writing the scheduled photo post as a story")
        return STORY_FORMAT
    return post_format


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
    fan-out. A demoted episode keeps the degraded format, so recorded canon
    reflects what the chat actually saw rather than what was scheduled. A
    media failure demotes the post, never kills it.

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
        photo_png = await generate_best_photo(episode.image_prompt)
        if photo_png is not None:
            return episode, EpisodeMedia(photo_png=photo_png)
        logger.warning("Image generation failed — degrading life post to a text story")
    if episode.format == STORY_FORMAT:
        return episode, EpisodeMedia()
    return dataclasses.replace(episode, format=STORY_FORMAT), EpisodeMedia()


async def generate_best_photo(image_prompt: str) -> bytes | None:
    """Generate up to ``IMAGEGEN_CANDIDATES`` selfies and pick the most faithful.

    SD1.5 renders the described interaction stochastically, so each candidate
    (random seed) is scored by the vision judge against ``image_prompt``. The
    first candidate reaching ``PHOTO_JUDGE_PASS_SCORE`` ships immediately;
    otherwise the best-scoring one does — the judge ranks, it never gates. A
    candidate the judge could not score ranks below any scored one but still
    ships if it is all there is.

    Scene before character descriptor in the assembled prompt — see
    ``PHOTO_FRAMING_HINT`` in ``src/config/prompts.py`` for why.

    Args:
        image_prompt: The episode's English scene description.

    Returns:
        PNG bytes of the chosen candidate, or None when every generation
        call failed (service down) — the caller then degrades the post.
    """
    photo_prompt = f"{PHOTO_FRAMING_HINT}{image_prompt}, {CHARACTER_VISUAL_PROMPT}"
    logger.debug("Photo generation prompt: %s", log.snippet(photo_prompt, log.OUTGOING_TEXT_LIMIT))
    best_png = None
    best_rank = UNSCORED_RANK - 1
    for attempt in range(config.IMAGEGEN_CANDIDATES):
        candidate = await generate_image(photo_prompt)
        if candidate is None:
            continue
        rank = await rank_candidate(candidate, image_prompt, attempt)
        if rank >= config.PHOTO_JUDGE_PASS_SCORE:
            return candidate
        if rank > best_rank:
            best_png = candidate
            best_rank = rank
    if best_png is not None:
        logger.info(
            "No photo candidate passed the judge — posting the best one (score %s)",
            best_rank if best_rank >= 0 else "unknown",
        )
    return best_png


async def rank_candidate(candidate: bytes, image_prompt: str, attempt: int) -> int:
    """Score one candidate for ranking against its siblings.

    Args:
        candidate: PNG bytes of the generated candidate.
        image_prompt: Scene the candidate is judged against.
        attempt: Zero-based candidate index, for the log line.

    Returns:
        The judge's 0-10 score, or :data:`UNSCORED_RANK` when the judge could
        not score it — below any real score, so an unscorable candidate loses
        to a scored one but still ships when it is all there is.
    """
    score = await score_photo(candidate, image_prompt)
    if score is None:
        return UNSCORED_RANK
    if score >= config.PHOTO_JUDGE_PASS_SCORE:
        logger.info("Photo candidate %d passed the judge (score %d)", attempt + 1, score)
    return score


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
        log.log_outgoing_text(f"life:{episode.format}", chat_id, chat_visible_text(episode))
        await record_sent_message(bot, chat_id, sent, media_type, episode)
        return True
    except Exception as error:
        logger.warning("Life post failed for chat %s: %s", chat_id, error)
        return False


def chat_visible_text(episode: Episode) -> str:
    """Return the episode text the chat actually reads or hears.

    A voice post shows only its teaser but speaks the full script, so both
    are returned: the caption alone would not tell you what Жора said.

    Args:
        episode: The episode being sent.

    Returns:
        The text worth reading back in the logs for this episode's format.
    """
    if episode.format == VOICE_FORMAT:
        return f"[caption] {episode.voice_teaser} [spoken] {episode.voice_script}"
    return episode.episode_text


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

    Uses the same reasoning-disabled, fallback-chained LLM as
    ``memory_writer.make_extraction_llm`` — see that function's docstring for
    why reasoning is disabled and how the fallback is wired.

    Args:
        episode_text: The episode text to distill.

    Returns:
        Up to ``MAX_DISTILLED_FACTS`` fact strings, possibly empty.
    """
    llm = make_extraction_llm()
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
