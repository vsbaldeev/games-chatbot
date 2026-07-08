"""HumorNode — autonomous-humor pipeline node.

Reached only when the opportunity gate fires. Gathers the live conversation
(rendered with ``[#id]`` markers so the comedian can cite its target) and
participant material, asks the comedian whether to joke, and — when it acts —
sets ``state["response"]`` plus a validated ``humor_reply_to_msg_id`` so
``run_pipeline`` anchors the joke to the message it is actually about (or sends
it un-anchored when no valid target was cited). On an abstain or any error it
stays silent (fail-safe to silence), then the graph continues to
``memory_writer`` so facts are still extracted.
"""

from src import config, log
from src.agent.comedian import ComedianDecision
from src.agent.roast_material import format_member_material, gather_member_material
from src.pipeline import humor_gate
from src.pipeline.response_node import render_row
from src.pipeline.state import BotState
from src.store import unified_messages

logger = log.get_logger(__name__)

RECENT_LIMIT = 15
MAX_PARTICIPANTS = 4


def render_conversation(recent: list[dict]) -> str:
    """Render recent messages oldest-first as ``[#id] @user [media]: text`` lines.

    The ``[#id]`` marker lets the comedian cite which message its joke targets,
    so the reply can be anchored to that message instead of the pipeline trigger.

    Args:
        recent: Messages newest-first, as returned by ``get_recent``.

    Returns:
        The rendered conversation, oldest-first.
    """
    return "\n".join(f"[#{row['message_id']}] {render_row(row)}" for row in reversed(recent))


def validate_reply_target(target: int | None, recent: list[dict], bot_id: int) -> int | None:
    """Return the target id if it cites a real participant message, else None.

    Fail-safe: a hallucinated id, a citation of the bot's own message, or a
    missing citation all degrade to None, which sends the joke un-anchored
    rather than attached to the wrong message.

    Args:
        target: The comedian's ``reply_to_message_id`` claim.
        recent: Messages newest-first, as fetched for the comedian prompt.
        bot_id: The bot's user id; its own messages are not valid targets.

    Returns:
        A validated message id, or None.
    """
    if target is None:
        return None
    valid_ids = {row["message_id"] for row in recent if row["user_id"] != bot_id}
    return target if target in valid_ids else None


def distinct_participants(recent: list[dict], bot_id: int) -> list[tuple[int, str]]:
    """Return distinct ``(user_id, username)`` from recent, newest-first.

    Excludes the bot and caps the count at ``MAX_PARTICIPANTS`` to bound cost.

    Args:
        recent: Messages newest-first.
        bot_id: The bot's user id, excluded from the result.

    Returns:
        Up to ``MAX_PARTICIPANTS`` distinct participants.
    """
    seen: set[int] = set()
    participants: list[tuple[int, str]] = []
    for row in recent:
        user_id = row["user_id"]
        if user_id == bot_id or user_id in seen:
            continue
        seen.add(user_id)
        participants.append((user_id, row["username"]))
        if len(participants) >= MAX_PARTICIPANTS:
            break
    return participants


async def gather_participants_material(chat_id: int, recent: list[dict], bot_id: int) -> str:
    """Collect and format material for the recent participants.

    Args:
        chat_id: Chat the participants belong to.
        recent: Messages newest-first.
        bot_id: The bot's user id, excluded.

    Returns:
        Blank-line-separated material blocks (empty when nobody has material).
    """
    blocks: list[str] = []
    for user_id, username in distinct_participants(recent, bot_id):
        material = await gather_member_material(chat_id, user_id, username)
        if material.is_empty:
            continue
        blocks.append(f"@{username}:\n{format_member_material(material)}")
    return "\n\n".join(blocks)


class HumorNode:
    """Generates an autonomous joke from the live conversation, or stays silent."""

    def __init__(self, agent) -> None:
        """Initialize HumorNode.

        Args:
            agent: Comedian agent used to decide and generate the joke.
        """
        self.__agent = agent

    async def __call__(self, state: BotState) -> dict:
        """Decide and, when acting, set the joke into ``state["response"]``.

        Args:
            state: Current pipeline state.

        Returns:
            ``{"response", "response_trigger", "humor_reply_to_msg_id"}`` when a
            joke is produced (the target id is None for an un-anchored joke),
            otherwise ``{}``.
        """
        chat_id = state["incoming"]["chat_id"]
        try:
            recent = await unified_messages.get_recent(chat_id=chat_id, limit=RECENT_LIMIT)
            decision = await self.__decide(chat_id, recent)
        except Exception as error:
            logger.warning("Humor decision failed for chat %s: %s", chat_id, error)
            humor_gate.mark_considered(chat_id)
            return {}
        if decision.act and decision.text.strip():
            target = validate_reply_target(decision.reply_to_message_id, recent, config.BOT_ID)
            humor_gate.mark_joke_sent(chat_id)
            logger.info(
                "Autonomous %s joke in chat %s (reply_to=%s)", decision.register, chat_id, target
            )
            return {
                "response": decision.text,
                "response_trigger": "humor",
                "humor_reply_to_msg_id": target,
            }
        humor_gate.mark_considered(chat_id)
        return {}

    async def __decide(self, chat_id: int, recent: list[dict]) -> ComedianDecision:
        """Build context and ask the comedian for a decision.

        Args:
            chat_id: Chat to consider.
            recent: Messages newest-first, as returned by ``get_recent``.

        Returns:
            The comedian's decision.
        """
        conversation = render_conversation(recent)
        material = await gather_participants_material(chat_id, recent, config.BOT_ID)
        return await self.__agent.decide(conversation, material)
