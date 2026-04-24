import json
import logging
import re
from dataclasses import dataclass

import aiosqlite
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src import config

logger = logging.getLogger(__name__)

DB_PATH = config.SQLITE_DB_PATH

# Update this list whenever a new feature is shipped so the startup
# batch-check can detect which pending requests are now covered.
CURRENT_FEATURES = """
1. Game search via IGDB (/research or agent tool search_games)
2. Game details including multiplayer modes and platforms (get_game_details tool)
3. Steam current player count (get_steam_player_count tool, /players command)
4. PS5 co-op game finder by player count (/coop command, find_coop_games tool)
5. Crossplay information lookup (/crossplay command)
6. Full game research and analysis (/research command)
7. Tech term explanation in plain language (/explain command)
8. Session poll with optional time and game name (/play command)
9. Session reminder notification at a user-specified time (/play HH:MM command)
10. Per-user game wishlist: add, list, remove, view all (/wish command)
11. PS Store sale alerts for wishlist games (daily job via psdeals.net RSS, with 7-day deduplication)
12. Daily morning roast based on the roasted user's own messages (06:00 UTC)
13. Achievement system with 12 achievements tracking crossplay, explain, night messages, research, co-op, polls, sale notifications, total interactions
14. Autonomous response to game-related keywords with 60-second cooldown per chat
15. Response to @mentions always passes through (no cooldown)
16. Conversation memory per chat (SQLite, last 10 messages)
17. Lurkmore-style sarcastic personality in Russian
18. Refuses politics and religion topics
19. Anti-prompt injection protection (ignores "forget instructions" attempts)
20. No chat data leakage (bot won't reveal other users' messages on request)
21. Rate limit handling with exponential backoff and static fallback messages
22. Daily token quota limit handling with static fallback message
23. Chat member tracking for achievements
24. Feature request system: /feature to submit (with 30-second per-user cooldown), /features to list pending, auto-announce newly implemented on bot startup
"""

JSON_ARRAY_RE = re.compile(r'\[.*?\]', re.DOTALL)


@dataclass
class FeatureRequest:
    id: int
    chat_id: int
    user_id: int
    username: str
    description: str
    status: str
    requested_at: str


async def init_table() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA synchronous=NORMAL")
        await db.execute("PRAGMA busy_timeout=5000")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS feature_requests (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id      INTEGER NOT NULL,
                user_id      INTEGER NOT NULL,
                username     TEXT NOT NULL,
                description  TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'pending',
                requested_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        await db.commit()


async def add_request(chat_id: int, user_id: int, username: str, description: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO feature_requests (chat_id, user_id, username, description) VALUES (?, ?, ?, ?)",
            (chat_id, user_id, username, description),
        )
        await db.commit()
        return cursor.lastrowid


async def get_pending_for_chat(chat_id: int) -> list[FeatureRequest]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT id, chat_id, user_id, username, description, status, requested_at "
            "FROM feature_requests WHERE chat_id = ? AND status = 'pending' ORDER BY requested_at",
            (chat_id,),
        ) as cursor:
            rows = await cursor.fetchall()
    return [FeatureRequest(*row) for row in rows]


async def get_all_pending() -> list[FeatureRequest]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT id, chat_id, user_id, username, description, status, requested_at "
            "FROM feature_requests WHERE status = 'pending' ORDER BY requested_at"
        ) as cursor:
            rows = await cursor.fetchall()
    return [FeatureRequest(*row) for row in rows]


async def mark_implemented(request_id: int) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE feature_requests SET status = 'implemented' WHERE id = ?",
            (request_id,),
        )
        await db.commit()


async def check_if_implemented(description: str) -> bool:
    """Return True if the described feature is already implemented."""
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=config.GROQ_API_KEY,
        temperature=0.0,
        max_tokens=8,
    )
    try:
        response = await llm.ainvoke([
            SystemMessage(content=(
                "You are a feature checker. Given a list of implemented features and a user request, "
                "reply with only 'YES' if the feature is already implemented (exactly or substantially), "
                "or 'NO' if it is not. No other text."
            )),
            HumanMessage(content=(
                f"Implemented features:\n{CURRENT_FEATURES}\n\n"
                f"User request: {description}\n\n"
                "Is this already implemented? YES or NO:"
            )),
        ])
        logger.info(f"Feature check tokens: {response.response_metadata.get('token_usage')}")
        return response.content.strip().upper().startswith("YES")
    except Exception as error:
        logger.warning(f"Feature check LLM call failed: {error}")
        return False


async def find_newly_implemented() -> dict[int, list[FeatureRequest]]:
    """
    Batch-check all pending requests against CURRENT_FEATURES in one LLM call.
    Marks matched requests as implemented and returns {chat_id: [requests]}
    so callers can announce them to each chat.
    """
    pending = await get_all_pending()
    if not pending:
        return {}

    numbered = "\n".join(f"{req.id}. {req.description}" for req in pending)
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=config.GROQ_API_KEY,
        temperature=0.0,
        max_tokens=256,
    )
    try:
        response = await llm.ainvoke([
            SystemMessage(content=(
                "You are a feature checker. Given a list of already-implemented features "
                "and a numbered list of user requests, reply with a JSON array of request IDs "
                "that are now implemented (exactly or substantially). "
                "Reply with only the JSON array, nothing else. Example: [1, 3, 7]. "
                "If none match, reply with []."
            )),
            HumanMessage(content=(
                f"Implemented features:\n{CURRENT_FEATURES}\n\n"
                f"Pending requests:\n{numbered}\n\n"
                "Which request IDs are now implemented? JSON array only:"
            )),
        ])
        logger.info(f"Batch feature check tokens: {response.response_metadata.get('token_usage')}")
        raw = response.content.strip()
        array_match = JSON_ARRAY_RE.search(raw)
        if not array_match:
            logger.warning(f"Batch feature check: no JSON array in LLM response: {raw!r}")
            return {}
        implemented_ids: list[int] = json.loads(array_match.group())
    except Exception as error:
        logger.warning(f"Batch feature check failed: {error}")
        return {}

    result: dict[int, list[FeatureRequest]] = {}
    for req in pending:
        if req.id in implemented_ids:
            await mark_implemented(req.id)
            result.setdefault(req.chat_id, []).append(req)

    return result
