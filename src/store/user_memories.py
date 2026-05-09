"""
Persistent store for per-user facts extracted by the MemoryWriter after each
bot response.

Facts are plain-language sentences produced by a lightweight LLM, for example:
  "Prefers co-op games over singleplayer"
  "Asked about Elden Ring twice in October"
  "Has won 5 duels this month"

The table is keyed by (chat_id, user_id) so the same user is tracked separately
in different group chats.
"""

import re
import time

from src.store import db as database

MAX_FACTS_PER_USER = 30

STAT_FACT_TEMPLATES: dict[str, str] = {
    "photo_messages":     "Отправил {count} фото в чате",
    "video_messages":     "Отправил {count} видео в чате",
    "voice_messages":     "Отправил {count} войсовых сообщений",
    "forwarded_messages": "Сделал {count} репостов в чате",
}

STAT_FACT_LIKE_PATTERNS: list[str] = [
    t.replace("{count}", "%") for t in STAT_FACT_TEMPLATES.values()
]

STAT_FACT_RE = re.compile(
    "^(" + "|".join(
        re.escape(t).replace(r"\{count\}", r"\d+")
        for t in STAT_FACT_TEMPLATES.values()
    ) + ")$"
)


async def init_table() -> None:
    """Create the user_memories table and its lookup index if they do not exist."""
    async with database.acquire() as conn:
        async with conn.transaction():
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_memories (
                    id         BIGSERIAL        PRIMARY KEY,
                    chat_id    BIGINT           NOT NULL,
                    user_id    BIGINT           NOT NULL,
                    username   TEXT             NOT NULL,
                    fact       TEXT             NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_memories_lookup
                ON user_memories (chat_id, user_id)
            """)


async def get_facts(*, chat_id: int, user_id: int) -> list[str]:
    """Return facts for the user, newest first. LLM-extracted only; falls back to stat facts if none exist."""
    async with database.acquire() as conn:
        llm_rows = await conn.fetch(
            """SELECT fact FROM user_memories
               WHERE chat_id = $1 AND user_id = $2
                 AND NOT (fact LIKE ANY($3::text[]))
               ORDER BY updated_at DESC""",
            chat_id, user_id, STAT_FACT_LIKE_PATTERNS,
        )
        if llm_rows:
            return [row["fact"] for row in llm_rows]
        stat_rows = await conn.fetch(
            """SELECT fact FROM user_memories
               WHERE chat_id = $1 AND user_id = $2
               ORDER BY updated_at DESC""",
            chat_id, user_id,
        )
    return [row["fact"] for row in stat_rows]


async def get_facts_for_users(
    *, chat_id: int, user_ids: list[int]
) -> dict[int, list[str]]:
    """Batch-fetch facts for several users. LLM-extracted only per user; falls back to stat facts if none exist."""
    if not user_ids:
        return {}

    async with database.acquire() as conn:
        rows = await conn.fetch(
            """SELECT user_id, fact FROM user_memories
               WHERE chat_id = $1 AND user_id = ANY($2)
               ORDER BY updated_at DESC""",
            chat_id, user_ids,
        )

    llm_facts: dict[int, list[str]] = {}
    stat_facts: dict[int, list[str]] = {}
    for row in rows:
        uid, fact = row["user_id"], row["fact"]
        if STAT_FACT_RE.match(fact):
            stat_facts.setdefault(uid, []).append(fact)
        else:
            llm_facts.setdefault(uid, []).append(fact)

    all_user_ids = set(llm_facts) | set(stat_facts)
    return {uid: llm_facts.get(uid) or stat_facts.get(uid, []) for uid in all_user_ids}


def format_hack_fact(count: int) -> str:
    last_two = count % 100
    last_one = count % 10
    if 11 <= last_two <= 19:
        suffix = "раз"
    elif last_one == 1:
        suffix = "раз"
    elif 2 <= last_one <= 4:
        suffix = "раза"
    else:
        suffix = "раз"
    return f"Пытался взломать бота {count} {suffix}"


async def upsert_hack_attempt(*, chat_id: int, user_id: int, username: str) -> None:
    """Increment (or create) the hack-attempt counter fact for this user."""
    now = time.time()
    async with database.acquire() as conn:
        async with conn.transaction():
            rows = await conn.fetch(
                """
                SELECT id, fact FROM user_memories
                WHERE chat_id = $1 AND user_id = $2 AND fact LIKE 'Пытался взломать бота%'
                """,
                chat_id, user_id,
            )
            if rows:
                match = re.search(r"(\d+)", rows[0]["fact"])
                count = int(match.group(1)) + 1 if match else 2
                await conn.execute(
                    "UPDATE user_memories SET fact = $1, updated_at = $2 WHERE id = $3",
                    format_hack_fact(count), now, rows[0]["id"],
                )
            else:
                await conn.execute(
                    """
                    INSERT INTO user_memories (chat_id, user_id, username, fact, updated_at)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    chat_id, user_id, username, format_hack_fact(1), now,
                )


async def upsert_stat_fact(
    *, chat_id: int, user_id: int, username: str, stat: str, count: int
) -> None:
    """Upsert a single living fact derived from a stat counter (e.g. '47 войсовых сообщений')."""
    template = STAT_FACT_TEMPLATES.get(stat)
    if not template:
        return
    prefix = template.split("{")[0]
    new_fact = template.format(count=count)
    now = time.time()
    async with database.acquire() as conn:
        async with conn.transaction():
            rows = await conn.fetch(
                "SELECT id FROM user_memories WHERE chat_id = $1 AND user_id = $2 AND fact LIKE $3",
                chat_id, user_id, f"{prefix}%",
            )
            if rows:
                await conn.execute(
                    "UPDATE user_memories SET fact = $1, updated_at = $2 WHERE id = $3",
                    new_fact, now, rows[0]["id"],
                )
            else:
                await conn.execute(
                    """INSERT INTO user_memories (chat_id, user_id, username, fact, updated_at)
                       VALUES ($1, $2, $3, $4, $5)""",
                    chat_id, user_id, username, new_fact, now,
                )


async def upsert_facts(
    *, chat_id: int, user_id: int, username: str, facts: list[str]
) -> None:
    """
    Insert new facts for a user and prune the oldest rows so the total stays
    within MAX_FACTS_PER_USER.
    """
    if not facts:
        return

    now = time.time()
    async with database.acquire() as conn:
        async with conn.transaction():
            await conn.executemany(
                """
                INSERT INTO user_memories (chat_id, user_id, username, fact, updated_at)
                VALUES ($1, $2, $3, $4, $5)
                """,
                [(chat_id, user_id, username, fact, now) for fact in facts],
            )
            await conn.execute(
                """
                DELETE FROM user_memories
                WHERE chat_id = $1 AND user_id = $2
                  AND id NOT IN (
                      SELECT id FROM user_memories
                      WHERE chat_id = $1 AND user_id = $2
                      ORDER BY updated_at DESC
                      LIMIT $3
                  )
                """,
                chat_id, user_id, MAX_FACTS_PER_USER,
            )
