"""Deduplicated storage for LLM system prompts, keyed by content hash.

The response system prompt (RESPONSE_PROMPT) is a multi-KB constant; storing
it on every bot_llm_log row would dominate that table's size, so each
distinct text is stored once here and log rows keep only its hash.
"""

import hashlib

from src.store import db as database


def hash_prompt(content: str) -> str:
    """Return the sha256 hex digest identifying a system prompt's text."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


async def ensure(content: str) -> str:
    """Guarantee `content` is stored under its hash; return that hash.

    Cheap to call on every reply: ON CONFLICT DO NOTHING skips the write
    once a prompt text has already been seen.

    Args:
        content: Full system prompt text.

    Returns:
        The sha256 hex digest to store on the bot_llm_log row.
    """
    sha256 = hash_prompt(content)
    async with database.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO llm_system_prompts (sha256, content)
            VALUES ($1, $2)
            ON CONFLICT (sha256) DO NOTHING
            """,
            sha256, content,
        )
    return sha256


async def cleanup_unreferenced() -> int:
    """Delete prompt texts no bot_llm_log row references any more.

    Must run after bot_llm_log's own retention sweep (see llm_log.cleanup_old),
    so a prompt is only dropped once every log row citing it is already gone.

    Returns:
        Number of deleted rows.
    """
    async with database.acquire() as conn:
        result = await conn.execute("""
            DELETE FROM llm_system_prompts
            WHERE NOT EXISTS (
                SELECT 1 FROM bot_llm_log
                WHERE bot_llm_log.system_prompt_sha = llm_system_prompts.sha256
            )
        """)
    return int(result.split()[-1])
