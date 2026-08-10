"""Replay stored chat messages through the reply-prompt assembly path.

For each stored message this rebuilds exactly what the response model would
receive — the shared system prompt plus that message's human turn — without
calling any chat model and without writing anything back to the database.

Run it before a prompt or context change, run it again after, and diff the two
outputs: the diff is precisely what the model will stop or start seeing. That
is a cheaper and far more exact instrument than reading generated replies,
whose wording changes run to run even when nothing was altered.

Deliberately does not use the pipeline graph. The router inserts every message
it handles into ``unified_messages``; replaying through it would pollute the
very history being replayed. This calls ``ContextBuilder`` and the response
node's prompt assembly directly, both of which are read-only.

Usage::

    python -m scripts.replay --limit 40 > before.txt
    # ...make the change...
    python -m scripts.replay --limit 40 > after.txt
    diff -u before.txt after.txt
"""

import argparse
import asyncio
import dataclasses
import random
import sys

from src.config.prompts import RESPONSE_PROMPT
from src.pipeline.context_builder import ContextBuilder
from src.pipeline.response_node import build_response_input
from src.store import db as database

SEPARATOR = "=" * 78
DEFAULT_LIMIT = 40
# Fixed so the 10% activity-volunteer roll in ContextBuilder lands identically
# on both sides of a diff. Without it every run reshuffles which messages carry
# the activity line and the diff is unreadable.
DEFAULT_SEED = 20260810


@dataclasses.dataclass(frozen=True)
class StubUser:
    """Stand-in for ``telegram.User`` carrying only what the pipeline reads."""

    id: int
    username: str
    first_name: str = ""


@dataclasses.dataclass(frozen=True)
class StubChat:
    """Stand-in for ``telegram.Chat`` carrying only the id."""

    id: int


@dataclasses.dataclass(frozen=True)
class StubMessage:
    """Stand-in for ``telegram.Message`` as the router and builder read it."""

    message_id: int
    text: str | None
    caption: str | None
    from_user: StubUser
    chat: StubChat
    reply_to_message: "StubMessage | None" = None


@dataclasses.dataclass(frozen=True)
class StubBot:
    """Stand-in for the Telegram Bot handle used for lazy media enrichment.

    Media enrichment needs real network access; the harness reports the
    failure per message rather than pretending it succeeded.
    """

    id: int
    username: str


@dataclasses.dataclass(frozen=True)
class StubContextTypes:
    """Stand-in for the Telegram context object, which is read only for ``bot``."""

    bot: StubBot


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list without the program name.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chat-id", type=int, default=None,
                        help="chat to replay; defaults to the busiest chat in the store")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"how many recent messages to replay (default {DEFAULT_LIMIT})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="RNG seed pinning probabilistic context gates")
    parser.add_argument("--bot-id", type=int, default=0, help="bot user id for reply-chain checks")
    return parser.parse_args(argv)


async def busiest_chat_id() -> int | None:
    """Return the chat id with the most stored messages.

    Returns:
        The chat id, or None when the store is empty.
    """
    async with database.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT chat_id FROM unified_messages"
            " GROUP BY chat_id ORDER BY count(*) DESC LIMIT 1"
        )
    return row["chat_id"] if row else None


async def load_rows(chat_id: int, limit: int) -> list[dict]:
    """Load the most recent messages for a chat, oldest-first.

    ``unified_messages.get_recent`` omits the reply and media columns the
    context builder needs, so the harness reads the full row itself.

    Args:
        chat_id: Chat to read.
        limit: How many of the most recent messages to take.

    Returns:
        Row dicts in chronological order.
    """
    async with database.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT message_id, user_id, username, content, media_type,
                   reply_to_msg_id, file_id, media_group_id, is_forwarded
            FROM unified_messages
            WHERE chat_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            chat_id, limit,
        )
    return [dict(row) for row in reversed(rows)]


def build_state(row: dict, chat_id: int, bot_id: int) -> dict:
    """Build the pipeline state ``ContextBuilder`` needs for one stored row.

    Args:
        row: A ``unified_messages`` row.
        chat_id: Chat the row belongs to.
        bot_id: Bot user id, used for reply-chain ownership checks.

    Returns:
        A state dict with the ``incoming`` and ``context_types`` keys the
        context builder reads.
    """
    user = StubUser(id=row["user_id"], username=row["username"] or "")
    message = StubMessage(
        message_id=row["message_id"],
        text=row["content"],
        caption=None,
        from_user=user,
        chat=StubChat(id=chat_id),
    )
    incoming = {
        "update": message,
        "chat_id": chat_id,
        "user_id": row["user_id"],
        "username": row["username"] or f"user_{row['user_id']}",
        "raw_text": row["content"],
        "processed_text": row["content"],
        "media_type": row["media_type"],
        "message_id": row["message_id"],
        "reply_to_msg_id": row["reply_to_msg_id"],
        "file_id": row.get("file_id"),
        "is_forwarded": bool(row.get("is_forwarded")),
        "media_group_id": row.get("media_group_id"),
        "replied_to_fallback": None,
    }
    return {
        "incoming": incoming,
        "context_types": StubContextTypes(bot=StubBot(id=bot_id, username="bot")),
    }


async def render_row(builder: ContextBuilder, row: dict, chat_id: int, bot_id: int) -> str:
    """Assemble the human turn the response model would receive for one row.

    Args:
        builder: The context builder instance to reuse across rows.
        row: A ``unified_messages`` row.
        chat_id: Chat the row belongs to.
        bot_id: Bot user id.

    Returns:
        The rendered prompt block, or an error block when assembly failed —
        a single unreplayable message must not abort the whole run.
    """
    state = build_state(row, chat_id, bot_id)
    try:
        context = (await builder(state))["context"]
        return build_response_input(
            state["incoming"]["username"],
            row["content"] or "",
            "",
            context,
            media_type=row["media_type"],
        )
    except Exception as error:
        return f"<<could not assemble: {type(error).__name__}: {error}>>"


def format_block(row: dict, rendered: str) -> str:
    """Format one replayed message as a stable, diffable text block.

    Args:
        row: The ``unified_messages`` row that was replayed.
        rendered: The assembled human turn.

    Returns:
        The text block, separator included.
    """
    header = (
        f"{SEPARATOR}\n"
        f"message_id={row['message_id']} user={row['username']} "
        f"media={row['media_type']} reply_to={row['reply_to_msg_id']}\n"
        f"{'-' * 78}\n"
    )
    return f"{header}{rendered}\n"


async def run(args: argparse.Namespace) -> int:
    """Replay the requested messages and print every assembled prompt.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Process exit code.
    """
    random.seed(args.seed)
    await database.init()
    try:
        chat_id = args.chat_id if args.chat_id is not None else await busiest_chat_id()
        if chat_id is None:
            print("no messages in the store — nothing to replay", file=sys.stderr)
            return 1
        rows = await load_rows(chat_id, args.limit)
        print(f"{SEPARATOR}\nSYSTEM PROMPT\n{'-' * 78}\n{RESPONSE_PROMPT}")
        print(f"{SEPARATOR}\nREPLAYING {len(rows)} message(s) "
              f"from chat {chat_id}, seed {args.seed}")
        builder = ContextBuilder()
        for row in rows:
            rendered = await render_row(builder, row, chat_id, args.bot_id)
            print(format_block(row, rendered))
    finally:
        await database.close()
    return 0


def main() -> int:
    """Entry point.

    Returns:
        Process exit code.
    """
    return asyncio.run(run(parse_args(sys.argv[1:])))


if __name__ == "__main__":
    raise SystemExit(main())
