"""EpisodeWriterAgent — writes the next installment of Жора's life for scheduled posts."""

import datetime
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from src import config, log
from src.agent.middleware import GroqContextGuard, ThinkingStripper, guarded_ainvoke, should_retry
from src.config.prompts import EPISODE_TEXT_MAX_CHARS, EPISODE_WRITER_SYSTEM
from src.life import calendar_ru
from src.life.engagement import MEMBER, choose_mode
from src.store import bot_memories
from src.utils.llm_json import load_json_object

logger = log.get_logger(__name__)

EPISODE_CONTEXT_EPISODES = 10
EPISODE_CONTEXT_ACTIVITIES = 7
CURRENT_ACTIVITY_MAX_CHARS = 80
# Formats grow as later steps ship: voice, then photo, then video_note.
ALL_FORMATS: tuple[str, ...] = ("story",)
WRITE_ATTEMPTS = 2


@dataclass(frozen=True)
class Episode:
    """One generated life-post episode, ready to hand to the poster.

    Attributes:
        episode_text: The post text (2-3 sentences); used as message text or
            future media caption.
        image_prompt: English scene description for image generation —
            character appearance is prepended separately at generation time.
        voice_script: Conversational retelling of the episode for TTS.
        current_activity: Present-tense activity phrase answering "what are
            you doing right now", or None when missing or over-length.
        format: The chosen post format.
    """

    episode_text: str
    image_prompt: str
    voice_script: str
    current_activity: str | None
    format: str


def build_engagement_lines(mode: str, mention: tuple[str, str] | None) -> list[str]:
    """Return the engagement instruction for this episode: a chat question or a member mention.

    Args:
        mode: ``engagement.SOLO`` or ``engagement.MEMBER``.
        mention: ``(username, fact)`` when mode is ``MEMBER``; otherwise None.

    Returns:
        Prompt lines instructing the writer how to engage the chat this post.
    """
    if mode == MEMBER and mention is not None:
        username, fact = mention
        return [
            f"Задание: упомяни в посте живого участника чата @{username}.",
            f"Известный факт о нём: «{fact}».",
            "Вплети его в историю тепло и по-доброму — как соседа или друга, который "
            "появился в твоей жизни, а не как факт для доклада. Опирайся только на этот "
            "факт, не выдумывай про него ничего сверх. Если факт слишком личный или может "
            "смутить человека на публике — упомяни его нейтрально, без этой детали, просто "
            "по-дружески кольни.",
            "",
        ]
    return [
        "Задание: закончи пост вопросом или подколкой в адрес чата — что-то, на что "
        "хочется ответить, а не просто прочитать.",
        "",
    ]


def build_history_lines(recent_episodes: list[dict], facts: list[str]) -> list[str]:
    """Return prompt lines for past episodes and canon facts.

    Args:
        recent_episodes: Recent episode rows, newest-first (as returned by
            ``bot_memories.get_recent_episodes``).
        facts: Canon facts to ground continuity (newest plus sampled older).

    Returns:
        Prompt lines: past episodes oldest-to-newest (or a first-post note
        when there are none), followed by canon facts when present.
    """
    parts: list[str] = []
    if recent_episodes:
        parts.append("Твои прошлые эпизоды (от старых к новым):")
        parts.extend(f"- {episode['content']}" for episode in reversed(recent_episodes))
        parts.append("")
    else:
        parts.append("У тебя ещё нет прошлых эпизодов — это твой самый первый пост чату.")
        parts.append("")
    if facts:
        parts.append("Факты твоего канона:")
        parts.extend(f"- {fact}" for fact in facts)
        parts.append("")
    return parts


def build_activity_lines(recent_activities: list[tuple[str, float]], now: datetime.datetime) -> list[str]:
    """Return the current date/season line and recent-activity continuity block.

    Args:
        recent_activities: Recent ``(phrase, posted_at)`` pairs, newest
            first, as returned by ``bot_memories.get_recent_activities``.
        now: Current moment, already in Moscow Time.

    Returns:
        Prompt lines: a «Сегодня …» date/season line, followed by a dated
        recent-activities block when any exist.
    """
    parts = [f"Сегодня {calendar_ru.describe_moscow_date(now)} (по Москве).", ""]
    if recent_activities:
        parts.append("Чем ты занимался в последние дни (для непротиворечивости, от новых к старым):")
        parts.extend(
            f"- {calendar_ru.describe_relative_day(posted_at, now)} — {phrase}"
            for phrase, posted_at in recent_activities
        )
        parts.append("")
    return parts


def build_episode_prompt(
    recent_episodes: list[dict],
    facts: list[str],
    recent_activities: list[tuple[str, float]],
    previous_format: str | None,
    allowed_formats: tuple[str, ...],
    mode: str,
    mention: tuple[str, str] | None,
) -> str:
    """Assemble the human turn for the episode writer.

    Args:
        recent_episodes: Recent episode rows, newest-first (as returned by
            ``bot_memories.get_recent_episodes``).
        facts: Canon facts to ground continuity (newest plus sampled older).
        recent_activities: Recent ``(phrase, posted_at)`` pairs, newest
            first, for season-consistent continuity.
        previous_format: Format of the most recent post, or None when there
            is no history yet.
        allowed_formats: Formats the writer may currently choose from.
        mode: ``engagement.SOLO`` or ``engagement.MEMBER`` — how this post
            should engage the chat (see :func:`build_engagement_lines`).
        mention: ``(username, fact)`` when mode is ``MEMBER``; otherwise None.

    Returns:
        The prompt string to send as the human turn.
    """
    now = datetime.datetime.now(calendar_ru.MOSCOW_TZ)
    parts = build_activity_lines(recent_activities, now)
    parts += build_history_lines(recent_episodes, facts)
    parts.append(f"Доступные форматы: {', '.join(allowed_formats)}.")
    if previous_format:
        parts.append(f"Предыдущий пост был в формате «{previous_format}» — выбери другой, если можно.")
    parts.append("")
    parts.extend(build_engagement_lines(mode, mention))
    parts.append("Напиши следующий эпизод. Ответь строго одним JSON-объектом.")
    return "\n".join(parts)


def coerce_current_activity(value: object) -> str | None:
    """Coerce the raw ``current_activity`` value, dropping it when unusable.

    Args:
        value: Raw value from the parsed episode JSON.

    Returns:
        A trimmed string within the character limit, or None when missing,
        empty or over-length.
    """
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    if not trimmed or len(trimmed) > CURRENT_ACTIVITY_MAX_CHARS:
        return None
    return trimmed


def parse_episode(data: dict, allowed_formats: tuple[str, ...]) -> Episode | None:
    """Validate and coerce a parsed episode JSON object into an Episode.

    Args:
        data: Parsed JSON dict from the model.
        allowed_formats: Formats currently offered to the writer.

    Returns:
        The validated Episode, or None when a required field is missing or
        ``episode_text`` exceeds the length limit — a validation failure
        that should trigger a retry, never a silent truncation.
    """
    episode_text = str(data.get("episode_text") or "").strip()
    image_prompt = str(data.get("image_prompt") or "").strip()
    voice_script = str(data.get("voice_script") or "").strip()
    if not episode_text or not image_prompt or not voice_script:
        return None
    if len(episode_text) > EPISODE_TEXT_MAX_CHARS:
        return None
    post_format = str(data.get("format") or "").strip()
    if post_format not in allowed_formats:
        post_format = allowed_formats[0]
    return Episode(
        episode_text=episode_text,
        image_prompt=image_prompt,
        voice_script=voice_script,
        current_activity=coerce_current_activity(data.get("current_activity")),
        format=post_format,
    )


class EpisodeWriterAgent:
    """LLM agent that writes the next life-post episode.

    Mirrors ComedianAgent: owns a LangChain executor with retry/fallback
    middleware. Accepts an injectable ``writer_executor`` for testing so
    production ``init()`` is never required in unit tests.
    """

    def __init__(self, *, writer_executor=None) -> None:
        """Initialize with an optional pre-built executor.

        Args:
            writer_executor: Pre-built agent executor (for testing).
        """
        self.__executor = writer_executor

    async def init(self) -> None:
        """Build the episode-writer executor from configuration."""
        self.__executor = EpisodeWriterAgent.__build_executor()
        logger.info(
            "EpisodeWriterAgent initialized with model: %s", config.EPISODE_MODEL_FALLBACKS[0]
        )

    async def write_episode(self, allowed_formats: tuple[str, ...] = ALL_FORMATS) -> Episode | None:
        """Write the next life episode, retrying once on a malformed response.

        Args:
            allowed_formats: Formats currently live; the writer must pick one.

        Returns:
            The generated Episode, or None when both the model call and the
            retry failed to produce a usable episode — the caller should
            skip this post slot; catch-up will retry it later.

        Raises:
            RuntimeError: If called before ``init()``.
        """
        if self.__executor is None:
            raise RuntimeError("EpisodeWriterAgent.init() must be called before writing")
        recent_episodes = await bot_memories.get_recent_episodes(EPISODE_CONTEXT_EPISODES)
        facts = await bot_memories.get_writer_facts()
        recent_activities = await bot_memories.get_recent_activities(EPISODE_CONTEXT_ACTIVITIES)
        previous_format = recent_episodes[0]["post_format"] if recent_episodes else None
        mode, mention = await choose_mode()
        prompt = build_episode_prompt(
            recent_episodes, facts, recent_activities, previous_format, allowed_formats, mode, mention
        )
        for attempt in range(WRITE_ATTEMPTS):
            episode = await self.__attempt(prompt, allowed_formats)
            if episode is not None:
                return episode
            logger.warning("Episode writer produced an unusable response (attempt %d)", attempt + 1)
        return None

    async def __attempt(self, prompt: str, allowed_formats: tuple[str, ...]) -> Episode | None:
        """Run one model call and parse its output into an Episode.

        Args:
            prompt: Assembled human-turn prompt.
            allowed_formats: Formats currently offered to the writer.

        Returns:
            The parsed Episode, or None on any parse/validation failure.
        """
        result = await guarded_ainvoke(self.__executor, {"messages": [HumanMessage(content=prompt)]})
        raw = result["messages"][-1].content or ""
        data = load_json_object(raw, context="Episode generation")
        if data is None:
            return None
        return parse_episode(data, allowed_formats)

    async def reset_model_index(self) -> None:
        """Rebuild the executor, resetting middleware state to the primary model."""
        await self.init()

    @staticmethod
    def __build_executor():
        """Build the episode-writer executor with retry/fallback middleware.

        Returns:
            Configured LangChain agent executor.
        """
        fallback_llms = [
            ChatGroq(
                model=model, api_key=config.GROQ_API_KEY, temperature=0.8, top_p=0.95,
                max_tokens=config.EPISODE_MAX_TOKENS, max_retries=0,
            )
            for model in config.EPISODE_MODEL_FALLBACKS[1:]
        ]
        primary_llm = ChatGroq(
            model=config.EPISODE_MODEL_FALLBACKS[0],
            api_key=config.GROQ_API_KEY,
            temperature=0.8,
            top_p=0.95,
            max_tokens=config.EPISODE_MAX_TOKENS,
            max_retries=0,
        )
        return create_agent(
            primary_llm,
            [],
            system_prompt=EPISODE_WRITER_SYSTEM,
            middleware=[
                ModelFallbackMiddleware(*fallback_llms),
                ModelRetryMiddleware(retry_on=should_retry, on_failure="error", max_retries=3),
                GroqContextGuard(),
                ThinkingStripper(),
            ],
        )


episode_writer_agent = EpisodeWriterAgent()
