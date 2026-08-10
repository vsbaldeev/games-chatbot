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
  3. User facts      — per-user memories for every participant visible in recent
                       history plus the initiating user, similarity-gated against
                       the incoming message (see USER_FACTS_SIMILAR_LIMIT).
  4. Reply chain     — full reply chain with photo and sticker rows lazily
                       enriched via the shared ingester.enrich_media_row helper
                       (also used by the filter node before classification) so
                       WorkerNode and ResponseNode see real descriptions, not
                       placeholders.
"""

import re

from src import achievements, log
from src.pipeline.ingester import enrich_media_row
from src.pipeline.state import AssembledContext, BotState
from src.store import embedder, unified_messages, user_memories, user_tags

logger = log.get_logger(__name__)

RECENT_HISTORY_LIMIT = 20
CHAIN_MSG_CHAR_LIMIT = 400
MENTION_RE = re.compile(r"@(\w+)", re.UNICODE)

# User-fact retrieval: these gate what the bot recalls about *other people*,
# where a mis-recall reads as the bot being absurd rather than merely
# off-topic.
#
# Injecting every stored fact for every recent participant and asking the prompt
# header to ignore the irrelevant ones did not work — the model surfaced
# unrelated memories and produced absurd replies. Relevance is now decided by
# retrieval, before the prompt is built, so an unrelated fact is never in
# context to be misused. Facts are ranked by cosine similarity to the incoming
# message; anything below the threshold is simply not recalled, even if it was
# learned minutes ago.
USER_FACTS_SIMILAR_LIMIT = 5
USER_FACTS_SIMILARITY_THRESHOLD = 0.85

def keep_conversational_facts(facts: list[str]) -> list[str]:
    """Drop counter-tally facts (hack-attempt stats) from a fact list.

    The tallies exist for weekly roles and roasts; in an ordinary reply the
    bot bringing them up reads as holding a grudge, so they never enter the
    reply prompt.

    Args:
        facts: Stored ``user_memories`` fact strings for one user.

    Returns:
        The facts safe to show to the response model; may be empty.
    """
    return [fact for fact in facts if not user_memories.is_counter_fact(fact)]


def key_facts_by_username(
    facts_by_id: dict[int, list[str]], username_by_id: dict[int, str]
) -> dict[str, list[str]]:
    """Re-key retrieved facts by username, dropping counter tallies.

    Args:
        facts_by_id: Retrieved facts per user id, closest match first.
        username_by_id: Username to render each user id as.

    Returns:
        Mapping of username to conversational facts; users left with nothing
        after filtering are omitted entirely.
    """
    user_facts: dict[str, list[str]] = {}
    for user_id, facts in facts_by_id.items():
        conversational = keep_conversational_facts(facts)
        if conversational:
            user_facts[username_by_id[user_id]] = conversational
    return user_facts


class ContextBuilder:
    """Loads recent history, replied-to message, and user memories into AssembledContext."""

    async def __call__(self, state: BotState) -> dict:
        msg = state["incoming"]
        chat_id = msg["chat_id"]

        bot = state["context_types"].bot
        fallback = msg.get("replied_to_fallback")
        query_embedding = await self.__embed_message(msg)
        recent = await self.__get_recent(chat_id, msg["message_id"])
        replied_to = await self.__find_replied_to(
            chat_id, msg["reply_to_msg_id"], recent, fallback
        )
        reply_chain = await self.__get_reply_chain(
            chat_id, msg["reply_to_msg_id"], bot, fallback
        )
        user_facts = await self.__collect_user_facts(
            chat_id, msg["user_id"], msg["username"], recent, query_embedding
        )
        asking_user_tag = await user_tags.get_tag(chat_id=chat_id, user_id=msg["user_id"])
        mentioned_tags = await self.__collect_mentioned_tags(
            chat_id, msg, replied_to, asker_username=msg["username"]
        )

        assembled: AssembledContext = {
            "user_facts": user_facts,
            "recent_history": recent,
            "replied_to": replied_to,
            "reply_chain": reply_chain,
            "asking_user_tag": asking_user_tag,
            "mentioned_tags": mentioned_tags,
        }
        return {"context": assembled}

    @staticmethod
    async def __embed_message(msg: dict) -> list[float] | None:
        """Embed the incoming message once for every similarity lookup.

        Bot-canon retrieval and per-user fact retrieval rank against the same
        query vector, so it is computed once here instead of once per collector.

        Args:
            msg: IncomingMessage dict of the message being processed.

        Returns:
            The message embedding, or None when the message carries no text or
            embedding failed — callers then skip similarity retrieval rather
            than failing the pipeline.
        """
        text = msg.get("processed_text") or msg.get("raw_text") or ""
        if not text.strip():
            return None
        try:
            return await embedder.embed(text)
        except Exception as err:
            logger.warning("Failed to embed incoming message: %s", err)
            return None

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
        query_embedding: list[float] | None,
    ) -> dict[str, list[str]]:
        """Gather facts relevant to the incoming message for recent participants.

        Retrieval is similarity-gated, not exhaustive — see the rationale on
        USER_FACTS_SIMILAR_LIMIT. Counter-tally facts are dropped via
        ``keep_conversational_facts``; users left with nothing are omitted.

        Args:
            chat_id: Chat the conversation happens in.
            initiating_user_id: Id of the user who triggered the pipeline.
            initiating_username: Username of the initiating user.
            recent: Recent-history rows already loaded for this chat.
            query_embedding: Embedding of the incoming message, or None when it
                could not be computed — no relevance can be judged, so no facts
                are recalled.

        Returns:
            Mapping of username to relevant fact strings, closest match first.
        """
        if query_embedding is None:
            return {}
        username_by_id = {row["user_id"]: row["username"] for row in recent}
        # The current message carries the freshest username, so it wins over
        # whatever the same user was called in older history rows.
        username_by_id[initiating_user_id] = initiating_username
        facts_by_id = await user_memories.find_relevant_facts_for_users(
            chat_id=chat_id,
            user_ids=list(username_by_id),
            embedding=query_embedding,
            top_k=USER_FACTS_SIMILAR_LIMIT,
            threshold=USER_FACTS_SIMILARITY_THRESHOLD,
        )
        return key_facts_by_username(facts_by_id, username_by_id)
