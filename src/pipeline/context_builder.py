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

import re

from src import achievements, log
from src.pipeline.ingester import enrich_media_row
from src.pipeline.state import AssembledContext, BotState
from src.store import unified_messages, user_memories, user_tags

logger = log.get_logger(__name__)

RECENT_HISTORY_LIMIT = 20
CHAIN_MSG_CHAR_LIMIT = 400
MENTION_RE = re.compile(r"@(\w+)", re.UNICODE)


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

        assembled: AssembledContext = {
            "user_facts": user_facts,
            "recent_history": recent,
            "replied_to": replied_to,
            "reply_chain": reply_chain,
            "asking_user_tag": asking_user_tag,
            "mentioned_tags": mentioned_tags,
        }
        return {"context": assembled}

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
        participant_ids = list({row["user_id"] for row in recent})
        facts_by_id = await user_memories.get_facts_for_users(
            chat_id=chat_id, user_ids=participant_ids
        )
        user_facts: dict[str, list[str]] = {}
        for row in recent:
            uid = row["user_id"]
            uname = row["username"]
            if uid in facts_by_id and uname not in user_facts:
                user_facts[uname] = facts_by_id[uid]

        if initiating_user_id not in facts_by_id:
            initiator_facts = await user_memories.get_facts(
                chat_id=chat_id, user_id=initiating_user_id
            )
            if initiator_facts:
                user_facts[initiating_username] = initiator_facts

        return user_facts
