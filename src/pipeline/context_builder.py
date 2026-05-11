"""
ContextBuilder — third node in the LangGraph pipeline.

Assembles everything the Agent node needs for an enriched prompt:
  1. Reply chain   — walk reply_to_msg_id links up to CHAIN_DEPTH_LIMIT hops.
  2. User facts    — per-user memories for every participant in the chain.
  3. Recent history — flat window to fill remaining context slots when the chain is short.
"""

from src import log

from src.pipeline.ingester import describe_photo, transcribe_voice, transcribe_video
from src.pipeline.state import AssembledContext, BotState
from src.store import unified_messages, user_memories
from src.store.unified_messages import (
    VOICE_PLACEHOLDER,
    VIDEO_NOTE_PLACEHOLDER,
    VIDEO_PLACEHOLDER,
)

logger = log.get_logger(__name__)

RECENT_HISTORY_LIMIT = 20


class ContextBuilder:
    """Loads reply chain, user memories, and recent history into AssembledContext."""

    async def __load_chain_facts(
        self,
        chat_id: int,
        reply_to: int | None,
        initiating_user_id: int,
        initiating_username: str,
    ) -> tuple[list, dict]:
        if reply_to is not None:
            chain = await unified_messages.get_chain(chat_id=chat_id, message_id=reply_to)
        else:
            chain = []

        chain_user_ids = list({row["user_id"] for row in chain})
        facts_by_user_id = await user_memories.get_facts_for_users(
            chat_id=chat_id,
            user_ids=chain_user_ids,
        )

        user_facts: dict[str, list[str]] = {}
        for row in chain:
            uid = row["user_id"]
            uname = row["username"]
            if uid in facts_by_user_id and uname not in user_facts:
                user_facts[uname] = facts_by_user_id[uid]

        if initiating_user_id not in facts_by_user_id:
            initiator_facts = await user_memories.get_facts(
                chat_id=chat_id,
                user_id=initiating_user_id,
            )
            if initiator_facts:
                user_facts[initiating_username] = initiator_facts

        return chain, user_facts

    async def __enrich_row(self, row: dict, bot) -> str:
        content = row["content"]
        file_id = row.get("file_id")
        media_type = row["media_type"]
        if not file_id:
            return content
        try:
            if media_type == "photo" and unified_messages.needs_photo_description(content):
                description = await describe_photo(file_id, bot)
                if not description:
                    return content
                caption = unified_messages.extract_photo_caption(content)
                return unified_messages.combine_description_and_caption(description, caption)
            if content == VOICE_PLACEHOLDER:
                return await transcribe_voice(file_id, "voice", bot) or content
            if content == VIDEO_NOTE_PLACEHOLDER:
                return await transcribe_video(file_id, "video_note", bot) or content
            if content == VIDEO_PLACEHOLDER:
                return await transcribe_video(file_id, "video", bot) or content
        except Exception as err:
            logger.warning("Media enrichment failed for message %s: %s", row["message_id"], err)
        return content

    async def __expand_media_groups(self, chain: list[dict], chat_id: int) -> list[dict]:
        """Expand any album message in the chain with its sibling messages."""
        group_ids = {row["media_group_id"] for row in chain if row.get("media_group_id")}
        if not group_ids:
            return chain
        seen_ids = {row["message_id"] for row in chain}
        extra: list[dict] = []
        for group_id in group_ids:
            for row in await unified_messages.get_media_group(chat_id=chat_id, media_group_id=group_id):
                if row["message_id"] not in seen_ids:
                    extra.append(row)
                    seen_ids.add(row["message_id"])
        if not extra:
            return chain
        merged = chain + extra
        merged.sort(key=lambda row: row["message_id"])
        return merged

    async def __enrich_chain_media(
        self, chain: list[dict], chat_id: int, bot
    ) -> list[dict]:
        enriched = []
        for row in chain:
            new_content = await self.__enrich_row(row, bot)
            if new_content != row["content"]:
                await unified_messages.update_content(
                    chat_id=chat_id,
                    message_id=row["message_id"],
                    content=new_content,
                )
                row = {**row, "content": new_content}
            enriched.append(row)
        return enriched

    async def __call__(self, state: BotState) -> dict:
        msg = state["incoming"]
        chat_id = msg["chat_id"]
        bot = state["context_types"].bot

        chain, user_facts = await self.__load_chain_facts(
            chat_id,
            msg["reply_to_msg_id"],
            msg["user_id"],
            msg["username"],
        )
        chain = await self.__expand_media_groups(chain, chat_id)
        chain = await self.__enrich_chain_media(chain, chat_id, bot)

        recent = await unified_messages.get_recent(
            chat_id=chat_id,
            limit=RECENT_HISTORY_LIMIT,
        )

        assembled: AssembledContext = {
            "reply_chain": chain,
            "user_facts": user_facts,
            "recent_history": recent,
        }
        return {"context": assembled}
