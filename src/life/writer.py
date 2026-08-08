"""EpisodeWriterAgent — writes the next installment of Жора's life for scheduled posts."""

import datetime
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from src import config, log
from src.agent.middleware import GroqContextGuard, ThinkingStripper, guarded_ainvoke, should_retry
from src.config.prompts import (
    EPISODE_TEASER_MAX_CHARS,
    EPISODE_TEXT_MAX_CHARS,
    EPISODE_VOICE_SCRIPT_MAX_CHARS,
    EPISODE_WRITER_SYSTEM,
)
from src.life import calendar_ru
from src.store import bot_memories
from src.utils.llm_json import load_json_object

logger = log.get_logger(__name__)

EPISODE_CONTEXT_EPISODES = 10
EPISODE_CONTEXT_ACTIVITIES = 7
CURRENT_ACTIVITY_MAX_CHARS = 80
# Formats grow as later steps ship: video_note is next. Which format a post
# uses is a scheduling decision — see WEEKLY_SCHEDULE in src/jobs/life_post.py.
STORY_FORMAT = "story"
VOICE_FORMAT = "voice"
PHOTO_FORMAT = "photo"
WRITE_ATTEMPTS = 2


@dataclass(frozen=True)
class Episode:
    """One generated life-post episode, ready to hand to the poster.

    Attributes:
        episode_text: The post text (2-3 sentences); used as message text or
            future media caption.
        image_prompt: English scene description for image generation —
            character appearance is prepended separately at generation time.
        voice_script: Self-contained spoken telling of the episode for TTS —
            for voice posts this IS the story members hear, including the
            engagement question/mention.
        voice_teaser: One dry hook line used as the voice note's caption; a
            lure, never a summary of the episode.
        current_activity: Present-tense activity phrase answering "what are
            you doing right now", or None when missing or over-length.
        format: The post format assigned by the weekly schedule, or the
            format it was demoted to when its media build failed.
    """

    episode_text: str
    image_prompt: str
    voice_script: str
    voice_teaser: str
    current_activity: str | None
    format: str


def build_engagement_lines() -> list[str]:
    """Return this episode's engagement instruction: always a chat question.

    Life posts never mention chat members by name (2026-08-07 decision) —
    every post closes with a question or subtle jab aimed at the chat instead
    of pulling a real member into the story.

    Returns:
        Prompt lines instructing the writer how to engage the chat this post.
    """
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
    post_format: str,
) -> str:
    """Assemble the human turn for the episode writer.

    Args:
        recent_episodes: Recent episode rows, newest-first (as returned by
            ``bot_memories.get_recent_episodes``).
        facts: Canon facts to ground continuity (newest plus sampled older).
        recent_activities: Recent ``(phrase, posted_at)`` pairs, newest
            first, for season-consistent continuity.
        post_format: Format this post ships in, assigned by the weekly
            schedule — the writer writes for it rather than picking one.

    Returns:
        The prompt string to send as the human turn.
    """
    now = datetime.datetime.now(calendar_ru.MOSCOW_TZ)
    parts = build_activity_lines(recent_activities, now)
    parts += build_history_lines(recent_episodes, facts)
    parts.append(f"Формат этого поста: {post_format}.")
    parts.append("")
    parts.extend(build_engagement_lines())
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


def parse_episode(data: dict, post_format: str) -> Episode | None:
    """Validate and coerce a parsed episode JSON object into an Episode.

    Every episode carries all three bodies (text, image prompt, voice
    script) whatever its format: that is what lets ``resolve_media`` demote
    a failed photo or voice post to a text story without rewriting it.

    Args:
        data: Parsed JSON dict from the model.
        post_format: Format assigned to this post; stamped onto the Episode
            rather than read from the model's output.

    Returns:
        The validated Episode, or None when a required field is missing or
        any field exceeds its length limit — a validation failure that
        should trigger a retry, never a silent truncation.
    """
    episode_text = str(data.get("episode_text") or "").strip()
    image_prompt = str(data.get("image_prompt") or "").strip()
    voice_script = str(data.get("voice_script") or "").strip()
    voice_teaser = str(data.get("voice_teaser") or "").strip()
    if not episode_text or not image_prompt or not voice_script or not voice_teaser:
        return None
    if len(episode_text) > EPISODE_TEXT_MAX_CHARS:
        return None
    if len(voice_script) > EPISODE_VOICE_SCRIPT_MAX_CHARS:
        return None
    if len(voice_teaser) > EPISODE_TEASER_MAX_CHARS:
        return None
    return Episode(
        episode_text=episode_text,
        image_prompt=image_prompt,
        voice_script=voice_script,
        voice_teaser=voice_teaser,
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

    async def write_episode(self, post_format: str) -> Episode | None:
        """Write the next life episode, retrying once on a malformed response.

        Args:
            post_format: Format this post ships in, assigned by the weekly
                schedule (``src/jobs/life_post.py``).

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
        prompt = build_episode_prompt(recent_episodes, facts, recent_activities, post_format)
        for attempt in range(WRITE_ATTEMPTS):
            episode = await self.__attempt(prompt, post_format)
            if episode is not None:
                return episode
            logger.warning("Episode writer produced an unusable response (attempt %d)", attempt + 1)
        return None

    async def __attempt(self, prompt: str, post_format: str) -> Episode | None:
        """Run one model call and parse its output into an Episode.

        Args:
            prompt: Assembled human-turn prompt.
            post_format: Format assigned to this post.

        Returns:
            The parsed Episode, or None on any parse/validation failure.
        """
        result = await guarded_ainvoke(self.__executor, {"messages": [HumanMessage(content=prompt)]})
        raw = result["messages"][-1].content or ""
        data = load_json_object(raw, context="Episode generation")
        if data is None:
            return None
        return parse_episode(data, post_format)

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
