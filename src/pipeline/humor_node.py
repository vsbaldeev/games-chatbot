"""HumorNode — autonomous-humor pipeline node.

Reached only when the opportunity gate fires. Gathers the live conversation and
participant material, asks the comedian whether to joke, and — when it acts —
sets ``state["response"]`` so ``run_pipeline`` delivers it like any other reply.
On an abstain or any error it stays silent (fail-safe to silence), then the
graph continues to ``memory_writer`` so facts are still extracted.
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
    """Render recent messages oldest-first as ``@user [media]: text`` lines.

    Args:
        recent: Messages newest-first, as returned by ``get_recent``.

    Returns:
        The rendered conversation, oldest-first.
    """
    return "\n".join(render_row(row) for row in reversed(recent))


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
            ``{"response": joke}`` when a joke is produced, otherwise ``{}``.
        """
        chat_id = state["incoming"]["chat_id"]
        try:
            decision = await self.__decide(chat_id)
        except Exception as error:
            logger.warning("Humor decision failed for chat %s: %s", chat_id, error)
            humor_gate.mark_considered(chat_id)
            return {}
        if decision.act and decision.text.strip():
            humor_gate.mark_joke_sent(chat_id)
            logger.info("Autonomous %s joke in chat %s", decision.register, chat_id)
            return {"response": decision.text}
        humor_gate.mark_considered(chat_id)
        return {}

    async def __decide(self, chat_id: int) -> ComedianDecision:
        """Build context and ask the comedian for a decision.

        Args:
            chat_id: Chat to consider.

        Returns:
            The comedian's decision.
        """
        recent = await unified_messages.get_recent(chat_id=chat_id, limit=RECENT_LIMIT)
        conversation = render_conversation(recent)
        material = await gather_participants_material(chat_id, recent, config.BOT_ID)
        return await self.__agent.decide(conversation, material)
