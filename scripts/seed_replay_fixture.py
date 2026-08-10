"""Seed a local database with synthetic chat messages for the replay harness.

Local development only. The messages are invented, not copied from any real
chat, so this file is safe to commit; run the harness against the production
database when you need a baseline over real traffic.

The fixture is chosen to exercise the context paths that a prompt change is
most likely to disturb: activity questions (which unlock the bot's current
activity), a bot-directed insult, a reply chain, and enough ordinary filler
for the probabilistic activity-volunteer gate to fire at least once.

Usage::

    python -m scripts.seed_replay_fixture --chat-id -100777
"""

import argparse
import asyncio
import sys

from src.store import bot_memories, db as database, embedder, unified_messages

BOT_USER_ID = 1
BOT_USERNAME = "neurozhora"

# (message_id, user_id, username, content, reply_to_msg_id)
FIXTURE: list[tuple[int, int, str, str, int | None]] = [
    (1001, 501, "member_a", "народ, кто-нибудь брал новую соньку?", None),
    (1002, 502, "member_b", "я жду скидок, сейчас ценник конский", 1001),
    (1003, 503, "member_c", "@neurozhora чем занят?", None),
    (1004, BOT_USER_ID, BOT_USERNAME, "Да так, по хозяйству вожусь.", 1003),
    (1005, 501, "member_a", "@neurozhora как дела вообще?", None),
    (1006, BOT_USER_ID, BOT_USERNAME, "Нормально, не жалуюсь.", 1005),
    (1007, 502, "member_b", "вчера дошёл наконец до финала, концовка слабая", None),
    (1008, 503, "member_c", "какая игра?", 1007),
    (1009, 502, "member_b", "та самая, про которую я месяц ныл", 1008),
    (1010, 501, "member_a", "@neurozhora а ты в это играл?", None),
    (1011, BOT_USER_ID, BOT_USERNAME, "Играл. Финал и правда так себе.", 1010),
    (1012, 503, "member_c", "@neurozhora ты бесполезный, если честно", 1011),
    (1013, BOT_USER_ID, BOT_USERNAME, "Спасибо, записал в блокнот.", 1012),
    (1014, 501, "member_a", "кто в выходные свободен на кооператив?", None),
    (1015, 502, "member_b", "я за, если не слишком поздно", 1014),
    (1016, 503, "member_c", "мне бы отоспаться сначала", 1014),
    (1017, 501, "member_a", "@neurozhora посоветуй что-нибудь на вечер", None),
    (1018, BOT_USER_ID, BOT_USERNAME, "Возьми что покороче, у вас терпения на эпик нет.", 1017),
    (1019, 502, "member_b", "справедливо", 1018),
    (1020, 503, "member_c", "@neurozhora что нового?", None),
]


# Canon the reply path injects: without it the bot-life block of the prompt is
# empty and a biography change produces no visible diff.
CANON_FACTS = [
    "Жора держит пасеку за домом.",
    "Жора перекрасил полдома в прошлом месяце.",
    "У Жоры в избе стоят комп и PS5.",
]
CANON_EPISODES = [
    ("Соседский пёс утащил мою рукавицу под крыльцо и не отдаёт. "
     "Сижу жду, кто первый сдастся.", "чинит крыльцо"),
    ("Ставил брагу по дедовскому рецепту, забыл про неё на три дня. "
     "Теперь в сенях пахнет как в клубе.", "проветривает сени"),
]
CANON_ACTIVITIES = ["чинит улья после медведя", "перебирает дрова в сарае"]


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list without the program name.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chat-id", type=int, default=-100777, help="chat id to seed under")
    return parser.parse_args(argv)


async def seed(chat_id: int) -> int:
    """Insert the fixture messages into ``unified_messages``.

    Args:
        chat_id: Chat id to file the messages under.

    Returns:
        Number of messages inserted.
    """
    inserted = 0
    for message_id, user_id, username, content, reply_to_msg_id in FIXTURE:
        await unified_messages.insert(
            chat_id=chat_id,
            message_id=message_id,
            user_id=user_id,
            username=username,
            content=content,
            media_type="text",
            reply_to_msg_id=reply_to_msg_id,
            file_id=None,
        )
        inserted += 1
    return inserted


async def seed_bot_canon() -> int:
    """Insert synthetic bot canon so the reply prompt's life block is populated.

    Returns:
        Number of canon rows written (facts plus episodes plus activities).
    """
    await bot_memories.upsert_facts(CANON_FACTS)
    for content, activity in CANON_EPISODES:
        await bot_memories.insert_episode(
            content=content,
            post_format="story",
            current_activity=activity,
            embedding=await embedder.embed(content),
        )
    for phrase in CANON_ACTIVITIES:
        await bot_memories.insert_activity(phrase)
    return len(CANON_FACTS) + len(CANON_EPISODES) + len(CANON_ACTIVITIES)


async def run(args: argparse.Namespace) -> int:
    """Seed the database and report what was written.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Process exit code.
    """
    await database.init()
    try:
        messages = await seed(args.chat_id)
        canon = await seed_bot_canon()
    finally:
        await database.close()
    print(f"seeded {messages} message(s) into chat {args.chat_id} and {canon} canon row(s)")
    return 0


def main() -> int:
    """Entry point.

    Returns:
        Process exit code.
    """
    return asyncio.run(run(parse_args(sys.argv[1:])))


if __name__ == "__main__":
    raise SystemExit(main())
