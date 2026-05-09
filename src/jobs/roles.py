"""Scheduled job: assign personality-based member tags every Sunday afternoon."""

import asyncio
import datetime
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from telegram.ext import ContextTypes

from src import achievements, config, log
from src.store.user_memories import get_facts_for_users

logger = log.get_logger(__name__)

TAG_MODEL = "llama-3.3-70b-versatile"
TAG_MAX_CHARS = 16

SYSTEM_PROMPT = (
    "Ты назначаешь короткие роли участникам чата на основе фактов об их поведении. "
    "Для каждого участника найди самую характерную черту или привычку из фактов — "
    "ту, которая лучше всего его определяет, — и придумай короткую остроумную роль "
    f"на русском языке (строго не длиннее {TAG_MAX_CHARS} символов включая пробелы). "
    "Роль должна быть конкретной и меткой, а не общей. "
    "Ответь строго в формате JSON: {\"user_0\": \"роль\", \"user_1\": \"роль\", ...} "
    "используя те же ключи, что и во входных данных. Без какого-либо другого текста."
)


async def _generate_tags(username_facts: dict[str, list[str]]) -> dict[str, str]:
    # Anonymise before sending to the LLM — use opaque keys, remap after.
    anon_to_username = {f"user_{idx}": username for idx, username in enumerate(username_facts)}
    username_to_anon = {username: anon for anon, username in anon_to_username.items()}

    user_lines = [
        f"{username_to_anon[username]}: {'; '.join(facts)}"
        for username, facts in username_facts.items()
    ]
    llm = ChatGroq(
        model=TAG_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0.5,
        top_p=0.9,
        max_tokens=max(512, len(username_facts) * 30),
    )
    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="\n".join(user_lines)),
    ])
    try:
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.content.strip(), flags=re.DOTALL)
        anon_tags = json.loads(raw)
        if not isinstance(anon_tags, dict):
            raise ValueError(f"Expected dict, got {type(anon_tags).__name__}")
        return {anon_to_username[anon]: tag for anon, tag in anon_tags.items() if anon in anon_to_username}
    except Exception as error:
        logger.warning("Tag generation returned non-JSON: %s — %s", error, response.content[:200])
        return {}


async def _assign_tags_for_chat(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int
) -> dict[str, str]:
    members = await achievements.get_chat_members(chat_id)
    if not members:
        return {}

    user_ids = [user_id for user_id, _ in members]
    facts_by_user = await get_facts_for_users(chat_id=chat_id, user_ids=user_ids)

    username_facts = {
        username: facts_by_user[user_id]
        for user_id, username in members
        if user_id in facts_by_user
    }
    if not username_facts:
        return {}

    tags = await _generate_tags(username_facts)

    assigned: dict[str, str] = {}
    for user_id, username in members:
        tag = tags.get(username, "").strip()[:TAG_MAX_CHARS]
        if not tag:
            continue
        try:
            await context.bot.set_chat_member_tag(
                chat_id=chat_id, user_id=user_id, tag=tag
            )
            assigned[username] = tag
        except Exception as error:
            logger.warning(
                "Failed to set tag for %s in chat %s: %s", username, chat_id, error
            )

    return assigned


async def _announce_tags(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int, assigned: dict[str, str]
) -> None:
    if not assigned:
        return
    lines = "\n".join(f"@{username} — {tag}" for username, tag in assigned.items())
    try:
        await context.bot.send_message(
            chat_id=chat_id, text=f"🏷 Роли недели:\n\n{lines}"
        )
    except Exception as error:
        logger.warning("Failed to announce tags for chat %s: %s", chat_id, error)


async def _run_for_chat(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    assigned = await _assign_tags_for_chat(context, chat_id)
    await _announce_tags(context, chat_id, assigned)


async def weekly_roles_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    if datetime.datetime.now(datetime.timezone.utc).weekday() != 6:  # 6 = Sunday
        return
    chat_ids = await achievements.get_all_chat_ids()
    await asyncio.gather(
        *[_run_for_chat(context, chat_id) for chat_id in chat_ids],
        return_exceptions=True,
    )
