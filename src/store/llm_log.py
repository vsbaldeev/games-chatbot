"""Per-reply LLM I/O log: the exact prompt and response for each delivered
pipeline reply.

Only the response call is logged — not the worker/tool-calling call, see
docs/superpowers/specs/2026-09-17-bot-feedback-metrics-design.md. JSONB
columns are passed as json.dumps(...) text with an explicit ::jsonb cast;
this codebase has no JSONB codec registered on the connection pool
(src/store/db.py only registers the pgvector codec).
"""

import json

from src.store import db as database

RETENTION_DAYS = 60


async def insert_call(
    *,
    chat_id: int,
    bot_message_id: int,
    user_message_id: int,
    user_id: int,
    trigger: str,
    filter_verdict: str,
    system_prompt_sha: str,
    history_messages: list[dict],
    user_prompt: str,
    raw_response: str,
    language_corrected: bool,
    response: str,
    model: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    latency_ms: int,
) -> None:
    """Persist one response-LLM call, keyed by the bot message it produced.

    Args:
        chat_id: Chat the reply was sent in.
        bot_message_id: The sent bot message's id (half the row's primary key).
        user_message_id: The triggering user message's id.
        user_id: The triggering user's id.
        trigger: Pipeline response_trigger ("explicit"/"random"/"youtube_short"/"social_link"/"insult_check").
        filter_verdict: Filter node verdict, or "-" when the filter did not run.
        system_prompt_sha: Hash returned by llm_system_prompts.ensure().
        history_messages: Thread-history turns preceding the final human turn,
            as [{"role": "human"|"ai", "content": str}, ...], oldest-first.
        user_prompt: The final assembled human turn sent to the model.
        raw_response: The model's first output, before language correction.
        language_corrected: True when LanguageCorrectionNode replaced the reply.
        response: The text actually delivered to the chat.
        model: Model name that answered, or None when the provider reported none.
        input_tokens: Input token count, or None when the model reported none.
        output_tokens: Output token count, or None when the model reported none.
        latency_ms: Wall-clock duration of the response call.
    """
    async with database.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO bot_llm_log
                (chat_id, bot_message_id, user_message_id, user_id, trigger,
                 filter_verdict, system_prompt_sha, history_messages, user_prompt,
                 raw_response, language_corrected, response, model,
                 input_tokens, output_tokens, latency_ms)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, $11, $12, $13, $14, $15, $16)
            ON CONFLICT (chat_id, bot_message_id) DO NOTHING
            """,
            chat_id, bot_message_id, user_message_id, user_id, trigger,
            filter_verdict, system_prompt_sha, json.dumps(history_messages), user_prompt,
            raw_response, language_corrected, response, model,
            input_tokens, output_tokens, latency_ms,
        )


async def cleanup_old() -> int:
    """Delete rows older than RETENTION_DAYS. Returns the number of deleted rows."""
    async with database.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM bot_llm_log WHERE created_at < now() - make_interval(days => $1)",
            RETENTION_DAYS,
        )
    return int(result.split()[-1])
