"""
ContextBuilder — third node in the LangGraph pipeline.

Assembles everything the Agent node needs for an enriched prompt:
  1. Recent history  — last RECENT_HISTORY_LIMIT messages from unified_messages,
                       excluding the current incoming message to avoid duplication.
  2. Replied-to      — the specific message being replied to (for annotation),
                       looked up in the recent window or fetched directly if older.
  3. User facts      — per-user memories for every participant visible in recent history
                       plus the initiating user.
  4. Reply chain     — full reply chain with photo rows lazily enriched via vision LLM
                       so WorkerNode and ResponseNode see real descriptions, not placeholders.
"""

from src import log
from src.pipeline.ingester import describe_photo
from src.pipeline.state import AssembledContext, BotState
from src.store import unified_messages, user_memories

logger = log.get_logger(__name__)

RECENT_HISTORY_LIMIT = 20
CHAIN_MSG_CHAR_LIMIT = 400


class ContextBuilder:
    """Loads recent history, replied-to message, and user memories into AssembledContext."""

    async def __call__(self, state: BotState) -> dict:
        msg = state["incoming"]
        chat_id = msg["chat_id"]

        bot = state["context_types"].bot
        recent = await self.__get_recent(chat_id, msg["message_id"])
        replied_to = await self.__find_replied_to(chat_id, msg["reply_to_msg_id"], recent)
        reply_chain = await self.__get_reply_chain(chat_id, msg["reply_to_msg_id"], bot)
        user_facts = await self.__collect_user_facts(
            chat_id, msg["user_id"], msg["username"], recent
        )

        assembled: AssembledContext = {
            "user_facts": user_facts,
            "recent_history": recent,
            "replied_to": replied_to,
            "reply_chain": reply_chain,
        }
        return {"context": assembled}

    async def __get_recent(self, chat_id: int, current_message_id: int) -> list[dict]:
        all_recent = await unified_messages.get_recent(
            chat_id=chat_id, limit=RECENT_HISTORY_LIMIT
        )
        return [row for row in all_recent if row["message_id"] != current_message_id]

    async def __get_reply_chain(self, chat_id: int, reply_to_msg_id: int | None, bot) -> list[dict]:
        if reply_to_msg_id is None:
            return []
        chain = await unified_messages.get_chain(chat_id=chat_id, message_id=reply_to_msg_id)
        enriched = [await self.__maybe_enrich_photo(row, chat_id, bot) for row in chain]
        return [self.__truncate_chain_row(row) for row in enriched]

    @staticmethod
    def __truncate_chain_row(row: dict) -> dict:
        content = row["content"]
        if len(content) <= CHAIN_MSG_CHAR_LIMIT:
            return row
        return {**row, "content": content[:CHAIN_MSG_CHAR_LIMIT] + "…"}

    async def __maybe_enrich_photo(self, row: dict, chat_id: int, bot) -> dict:
        if row["media_type"] != "photo" or not unified_messages.needs_photo_description(row["content"]):
            return row
        file_id = row.get("file_id")
        if not file_id:
            return row
        caption = unified_messages.extract_photo_caption(row["content"])
        description = await describe_photo(file_id, bot)
        if not description:
            return row
        combined = unified_messages.combine_description_and_caption(description, caption)
        try:
            await unified_messages.update_content(
                chat_id=chat_id,
                message_id=row["message_id"],
                content=combined,
            )
        except Exception as err:
            logger.warning("Failed to cache photo description for msg %s: %s", row["message_id"], err)
        return {**row, "content": combined}

    async def __find_replied_to(
        self, chat_id: int, reply_to_msg_id: int | None, recent: list[dict]
    ) -> dict | None:
        if reply_to_msg_id is None:
            return None
        for row in recent:
            if row["message_id"] == reply_to_msg_id:
                return row
        return await unified_messages.get_by_id(chat_id=chat_id, message_id=reply_to_msg_id)

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
