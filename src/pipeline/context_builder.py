"""
ContextBuilder — third node in the LangGraph pipeline.

Assembles everything the Agent node needs for an enriched prompt:
  1. Recent history  — last RECENT_HISTORY_LIMIT messages from unified_messages,
                       excluding the current incoming message to avoid duplication.
  2. Replied-to      — the specific message being replied to (for annotation),
                       looked up in the recent window or fetched directly if older;
                       when the store has no row (other bots' posts, command
                       outputs, expired messages) the fallback synthesized from
                       the Telegram update is used instead.
  3. User facts      — per-user memories for every participant visible in recent history
                       plus the initiating user.
  4. Reply chain     — full reply chain with photo and sticker rows lazily
                       enriched via the shared ingester.enrich_media_row helper
                       (also used by the filter node before classification) so
                       WorkerNode and ResponseNode see real descriptions, not
                       placeholders.
"""

import random
import re
import time

from src import achievements, log
from src.pipeline.ingester import enrich_media_row
from src.pipeline.state import AssembledContext, BotState
from src.store import bot_memories, embedder, unified_messages, user_memories, user_tags

logger = log.get_logger(__name__)

RECENT_HISTORY_LIMIT = 20
CHAIN_MSG_CHAR_LIMIT = 400
MENTION_RE = re.compile(r"@(\w+)", re.UNICODE)

# Bot canon retrieval, mirroring user-facts sizing: a handful of relevant
# facts plus at most one or two full episodes when the topic is a specific
# past story.
BOT_FACTS_SIMILAR_LIMIT = 5
BOT_FACTS_NEWEST_LIMIT = 3
BOT_FACTS_CAP = 8
BOT_EPISODES_LIMIT = 2
BOT_ACTIVITY_HISTORY_LIMIT = 7

# Activity gate: the daily refresh means a fresh current activity always
# exists, so injecting it unconditionally made the bot narrate his routine in
# nearly every reply. It now enters the prompt only when the message asks
# what he is doing/did, or on a rare roll so he occasionally volunteers it.
ACTIVITY_VOLUNTEER_PROBABILITY = 0.1
ACTIVITY_QUESTION_RE = re.compile(
    r"ч(?:то|е|ё)\s+(?:\w+\s+){0,2}?(?:по)?дел(?:а|ыва)\w*"  # что делаешь / чё поделываешь
    r"|чем\s+(?:\w+\s+){0,2}?занима\w*"  # чем занимаешься / чем занимался
    r"|чем\s+(?:\w+\s+){0,2}?занят\w*"  # чем занят / чем ты занята
    r"|ч(?:то|е|ё)\s+(?:\w+\s+){0,2}?твори\w*"  # что творишь
    r"|как\s+(?:\w+\s+){0,1}?дела\b"  # как дела / как твои дела
    r"|как\s+прош(?:ел|ёл|ла|ло|ли)\b"  # как прошёл день / как прошли выходные
    r"|что\s+нового\b"  # что нового
    r"|как\s+(?:сам|сама|жизнь|оно)\b",  # как сам / как жизнь / как оно
    re.IGNORECASE,
)


def keep_conversational_facts(facts: list[str]) -> list[str]:
    """Drop counter-tally facts (insult/hack-attempt stats) from a fact list.

    The tallies exist for weekly roles and roasts; in an ordinary reply the
    bot bringing up «Оскорблял бота N раз» reads as holding a grudge, so
    they never enter the reply prompt.

    Args:
        facts: Stored ``user_memories`` fact strings for one user.

    Returns:
        The facts safe to show to the response model; may be empty.
    """
    return [fact for fact in facts if not user_memories.is_counter_fact(fact)]


def is_activity_question(text: str) -> bool:
    """Return True when the message asks what the bot is doing or did.

    Args:
        text: Incoming message text (processed transcript or raw text).

    Returns:
        True when the text matches a Russian "what are you doing / what did
        you do / how are things" question aimed at the bot's activity.
    """
    return bool(ACTIVITY_QUESTION_RE.search(text))


class ContextBuilder:
    """Loads recent history, replied-to message, and user memories into AssembledContext."""

    async def __call__(self, state: BotState) -> dict:
        msg = state["incoming"]
        chat_id = msg["chat_id"]

        bot = state["context_types"].bot
        fallback = msg.get("replied_to_fallback")
        recent = await self.__get_recent(chat_id, msg["message_id"])
        replied_to = await self.__find_replied_to(
            chat_id, msg["reply_to_msg_id"], recent, fallback
        )
        reply_chain = await self.__get_reply_chain(
            chat_id, msg["reply_to_msg_id"], bot, fallback
        )
        user_facts = await self.__collect_user_facts(
            chat_id, msg["user_id"], msg["username"], recent
        )
        asking_user_tag = await user_tags.get_tag(chat_id=chat_id, user_id=msg["user_id"])
        mentioned_tags = await self.__collect_mentioned_tags(
            chat_id, msg, replied_to, asker_username=msg["username"]
        )
        bot_self_facts, bot_self_episodes = await self.__collect_bot_canon(msg)
        bot_current_activity, bot_recent_activities = await self.__collect_activity_context(msg)

        assembled: AssembledContext = {
            "user_facts": user_facts,
            "recent_history": recent,
            "replied_to": replied_to,
            "reply_chain": reply_chain,
            "asking_user_tag": asking_user_tag,
            "mentioned_tags": mentioned_tags,
            "bot_self_facts": bot_self_facts,
            "bot_self_episodes": bot_self_episodes,
            "bot_current_activity": bot_current_activity,
            "bot_recent_activities": bot_recent_activities,
        }
        return {"context": assembled}

    async def __collect_bot_canon(self, msg: dict) -> tuple[list[str], list[str]]:
        """Retrieve canon facts and past episodes relevant to the incoming message.

        Embeds the message once and reuses it for both the fact and episode
        similarity queries. Degrades to empty lists on any failure (embedding
        or database error) — a missing canon block must never fail the
        pipeline.

        Args:
            msg: IncomingMessage dict of the message being processed.

        Returns:
            ``(bot_self_facts, bot_self_episodes)``, each possibly empty.
        """
        text = msg.get("processed_text") or msg.get("raw_text") or ""
        if not text.strip():
            return [], []
        try:
            query_embedding = await embedder.embed(text)
            similar_facts = await bot_memories.find_similar_facts(
                query_embedding, BOT_FACTS_SIMILAR_LIMIT
            )
            newest_facts = await bot_memories.get_facts(BOT_FACTS_NEWEST_LIMIT)
            facts = list(dict.fromkeys(similar_facts + newest_facts))[:BOT_FACTS_CAP]
            episodes = await bot_memories.find_similar_episodes(
                query_embedding, top_k=BOT_EPISODES_LIMIT
            )
            return facts, episodes
        except Exception as err:
            logger.warning("Failed to load bot canon context: %s", err)
            return [], []

    async def __collect_activity_context(
        self, msg: dict
    ) -> tuple[tuple[str, str] | None, list[tuple[str, float]]]:
        """Gate Жора's activity out of the prompt unless asked or a rare roll fires.

        The daily refresh means a fresh activity always exists, so injecting
        it unconditionally made the bot mention it in nearly every reply.
        Include the current activity only when the message asks what he is
        doing/did, or on a small random chance so he occasionally volunteers
        it; the dated history is included only when actually asked — it
        answers dated "what did you do" questions and is never volunteered.

        Args:
            msg: IncomingMessage dict of the message being processed.

        Returns:
            ``(bot_current_activity, bot_recent_activities)`` — ``(None, [])``
            when the gate stays closed.
        """
        text = msg.get("processed_text") or msg.get("raw_text") or ""
        asked = is_activity_question(text)
        volunteered = not asked and random.random() < ACTIVITY_VOLUNTEER_PROBABILITY
        if not asked and not volunteered:
            return None, []
        current = await self.__get_bot_current_activity()
        recent = await self.__get_bot_recent_activities() if asked else []
        logger.debug("Activity gate open (asked=%s, volunteered=%s)", asked, volunteered)
        return current, recent

    async def __get_bot_current_activity(self) -> tuple[str, str] | None:
        """Bucket the newest episode's current-activity phrase by freshness.

        Returns:
            ``(phrase, "fresh")`` within :data:`bot_memories.ACTIVITY_FRESH_HOURS`,
            ``(phrase, "recent")`` within :data:`bot_memories.ACTIVITY_RECENT_HOURS`,
            otherwise None — either the activity is stale (the bot improvises)
            or the lookup failed.
        """
        try:
            activity = await bot_memories.get_current_activity()
        except Exception as err:
            logger.warning("Failed to load bot current activity: %s", err)
            return None
        if activity is None:
            return None
        phrase, posted_at = activity
        age_hours = (time.time() - posted_at) / 3600
        if age_hours < bot_memories.ACTIVITY_FRESH_HOURS:
            return phrase, "fresh"
        if age_hours < bot_memories.ACTIVITY_RECENT_HOURS:
            return phrase, "recent"
        return None

    async def __get_bot_recent_activities(self) -> list[tuple[str, float]]:
        """Load Жора's recent activity history for dated "what did you do" answers.

        Returns:
            Up to :data:`BOT_ACTIVITY_HISTORY_LIMIT` ``(phrase, posted_at)``
            pairs, newest first; empty list on any lookup failure — a
            missing history must never fail the pipeline.
        """
        try:
            return await bot_memories.get_recent_activities(BOT_ACTIVITY_HISTORY_LIMIT)
        except Exception as err:
            logger.warning("Failed to load bot recent activities: %s", err)
            return []

    async def __collect_mentioned_tags(
        self, chat_id: int, msg: dict, replied_to: dict | None, asker_username: str
    ) -> dict[str, dict]:
        """Load weekly roles for members @mentioned in the question or replied to.

        Lets the bot explain another member's role (e.g. "why does @x have this
        tag") by resolving the mentioned usernames to their stored tag + reason.
        The asker's own role is excluded — it is carried separately.

        Args:
            chat_id: Group chat the message belongs to.
            msg: The incoming message dict.
            replied_to: The message being replied to, if any.
            asker_username: Sender's username, excluded from the result.

        Returns:
            Mapping of username to ``{"tag", "reason"}`` for resolvable members.
        """
        text = " ".join(filter(None, [
            msg.get("processed_text"), msg.get("raw_text"),
            (replied_to or {}).get("content"),
        ]))
        mentioned = {mention.lower() for mention in MENTION_RE.findall(text)}
        mentioned.discard(asker_username.lower())
        if not mentioned:
            return {}
        members = await achievements.get_chat_members(chat_id)
        username_by_id = {
            uid: uname for uid, uname in members if uname.lower() in mentioned
        }
        if not username_by_id:
            return {}
        tags_by_id = await user_tags.get_tags_for_users(
            chat_id=chat_id, user_ids=list(username_by_id)
        )
        return {
            username_by_id[uid]: tag for uid, tag in tags_by_id.items()
        }

    async def __get_recent(self, chat_id: int, current_message_id: int) -> list[dict]:
        all_recent = await unified_messages.get_recent(
            chat_id=chat_id, limit=RECENT_HISTORY_LIMIT
        )
        return [row for row in all_recent if row["message_id"] != current_message_id]

    async def __get_reply_chain(
        self, chat_id: int, reply_to_msg_id: int | None, bot, fallback: dict | None
    ) -> list[dict]:
        """Load the reply chain, degrading to a one-element fallback chain.

        Args:
            chat_id: Chat the reply belongs to.
            reply_to_msg_id: Message id being replied to, or None.
            bot: Telegram bot instance for lazy media enrichment.
            fallback: Row-shaped copy of the replied-to message from the
                update, used when the store has no chain for it.

        Returns:
            Chain rows oldest-first (photo and sticker rows enriched, content
            truncated); a one-element chain from the fallback when the store
            has nothing.
        """
        if reply_to_msg_id is None:
            return []
        chain = await unified_messages.get_chain(chat_id=chat_id, message_id=reply_to_msg_id)
        if not chain and fallback:
            chain = [fallback]
        enriched = [await enrich_media_row(row, chat_id, bot) for row in chain]
        return [self.__truncate_chain_row(row) for row in enriched]

    @staticmethod
    def __truncate_chain_row(row: dict) -> dict:
        content = row["content"]
        if len(content) <= CHAIN_MSG_CHAR_LIMIT:
            return row
        return {**row, "content": content[:CHAIN_MSG_CHAR_LIMIT] + "…"}

    async def __find_replied_to(
        self,
        chat_id: int,
        reply_to_msg_id: int | None,
        recent: list[dict],
        fallback: dict | None,
    ) -> dict | None:
        """Resolve the replied-to message: recent window, store, then fallback.

        Args:
            chat_id: Chat the reply belongs to.
            reply_to_msg_id: Message id being replied to, or None.
            recent: Recent-history rows already loaded for this chat.
            fallback: Row-shaped copy of the replied-to message from the
                update, used when the store has no row.

        Returns:
            The best available row for the replied-to message, or None.
        """
        if reply_to_msg_id is None:
            return None
        for row in recent:
            if row["message_id"] == reply_to_msg_id:
                return row
        stored = await unified_messages.get_by_id(chat_id=chat_id, message_id=reply_to_msg_id)
        return stored or fallback

    async def __collect_user_facts(
        self,
        chat_id: int,
        initiating_user_id: int,
        initiating_username: str,
        recent: list[dict],
    ) -> dict[str, list[str]]:
        """Gather reply-context facts for recent participants and the initiator.

        Counter-tally facts are dropped via ``keep_conversational_facts`` —
        users left with no facts are omitted entirely.

        Args:
            chat_id: Chat the conversation happens in.
            initiating_user_id: Id of the user who triggered the pipeline.
            initiating_username: Username of the initiating user.
            recent: Recent-history rows already loaded for this chat.

        Returns:
            Mapping of username to conversational fact strings.
        """
        participant_ids = list({row["user_id"] for row in recent})
        facts_by_id = await user_memories.get_facts_for_users(
            chat_id=chat_id, user_ids=participant_ids
        )
        user_facts: dict[str, list[str]] = {}
        for row in recent:
            uid = row["user_id"]
            uname = row["username"]
            if uid in facts_by_id and uname not in user_facts:
                facts = keep_conversational_facts(facts_by_id[uid])
                if facts:
                    user_facts[uname] = facts

        if initiating_user_id not in facts_by_id:
            initiator_facts = keep_conversational_facts(
                await user_memories.get_facts(chat_id=chat_id, user_id=initiating_user_id)
            )
            if initiator_facts:
                user_facts[initiating_username] = initiator_facts

        return user_facts
