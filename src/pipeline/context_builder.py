"""
ContextBuilder — third node in the LangGraph pipeline.

Assembles everything the Agent node needs for an enriched prompt:
  1. Reply chain   — walk reply_to_msg_id links up to CHAIN_DEPTH_LIMIT hops.
  2. User facts    — per-user memories for every participant in the chain.
  3. Recent history — flat window to fill remaining context slots when the chain is short.
"""

from src import log

from src.pipeline.state import AssembledContext, BotState
from src.store import unified_messages, user_memories

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

    async def __call__(self, state: BotState) -> dict:
        msg = state["incoming"]
        chat_id = msg["chat_id"]

        chain, user_facts = await self.__load_chain_facts(
            chat_id,
            msg["reply_to_msg_id"],
            msg["user_id"],
            msg["username"],
        )

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
