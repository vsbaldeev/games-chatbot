"""
Pure message-building functions and Telegram edit helper for D&D games.

Nothing here touches state or the LLM — callers pass in the data they need.
"""

from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest, TelegramError

from src import log
from src.dnd.state import (
    LobbyState,
    ActiveGame,
    DND_JOIN_CALLBACK,
    DND_ACTION_CALLBACK_PREFIX,
    DND_MIN_PLAYERS,
    DND_ACTION_TIMEOUT,
)

logger = log.get_logger(__name__)


def build_lobby_text(lobby: LobbyState, max_rounds: int, mode: str) -> str:
    """Return the lobby message body shown while players are joining."""
    player_lines = "\n".join(f"• @{username}" for _, username in lobby.players)
    count = len(lobby.players)
    remaining = DND_MIN_PLAYERS - count
    status = "готово к старту!" if count >= DND_MIN_PLAYERS else f"нужно ещё {remaining}"

    headers = {
        "pvp":   "⚔️ D&D — Все против всех (1 раунд)",
        "coop":  "⚔️ D&D — Кооп против Босса (2 раунда)",
        "heist": "🎩 D&D — Великое Ограбление (3 раунда)",
    }
    header = headers.get(mode, "⚔️ D&D Приключение")

    return (
        f"{header}\n\n"
        f"Набирается отряд!\n"
        f"Нужно минимум {DND_MIN_PLAYERS} игрока.\n\n"
        f"Отряд ({count}) — {status}:\n{player_lines}\n\n"
        f"Лобби закроется через 5 минут."
    )


def build_lobby_keyboard() -> InlineKeyboardMarkup:
    """Return the single-button keyboard shown in the lobby message."""
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("⚔️ Присоединиться", callback_data=DND_JOIN_CALLBACK),
    ]])


def build_game_text(game: ActiveGame) -> str:
    """Return the in-round message body with player statuses and scenario."""
    chosen_ids = set(game.choices.keys())
    player_lines = "\n".join(
        f"✅ @{username}" if player_id in chosen_ids else f"⏳ @{username}"
        for player_id, username in game.players
    )

    if game.mode == "coop":
        hp_line = f"👹 {game.boss_name} — ❤️ {game.boss_hp}/{game.boss_max_hp} HP\n\n"
        round_header = f"⚔️ D&D Кооп — Раунд {game.round_number}/{game.max_rounds}"
    elif game.mode == "pvp":
        hp_line = ""
        round_header = "⚔️ D&D — Все против всех"
    else:
        hp_line = ""
        round_header = f"🎩 Великое Ограбление — {heist_phase_name(game.round_number)}"

    return (
        f"{round_header}\n\n"
        f"{hp_line}"
        f"{game.scenario}\n\n"
        f"У каждого игрока {DND_ACTION_TIMEOUT} секунд на выбор:\n"
        f"{player_lines}"
    )


def build_game_keyboard(game: ActiveGame) -> InlineKeyboardMarkup:
    """Return the action-selection keyboard for the current round."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(action, callback_data=f"{DND_ACTION_CALLBACK_PREFIX}{index}")]
        for index, action in enumerate(game.actions)
    ])


def format_roll_lines(player_results: list[dict]) -> list[str]:
    """Format per-player roll results into display strings."""
    lines = []
    for result in player_results:
        roll = result["roll"]
        mark = " 💀" if roll == 1 else " ✨" if roll == 20 else ""
        lines.append(f'• @{result["username"]}: {result["action"]} → 🎲{roll}{mark}')
    return lines


def heist_phase_name(round_number: int) -> str:
    """Map a round number to its heist phase label."""
    return {1: "Проникновение", 2: "Дело", 3: "Побег"}.get(round_number, f"Фаза {round_number}")


async def edit_safe(
    bot: Any,
    chat_id: int,
    message_id: int,
    text: str,
    keyboard: InlineKeyboardMarkup | None = None,
) -> None:
    """Edit a Telegram message, silently ignoring 'not modified' and other transient errors."""
    kwargs: dict[str, Any] = {"chat_id": chat_id, "message_id": message_id}
    if keyboard is not None:
        kwargs["reply_markup"] = keyboard
    try:
        await bot.edit_message_text(text=text, **kwargs)
    except BadRequest as error:
        if "not modified" in str(error).lower():
            return
        logger.warning("DnD edit failed for chat %s msg %s: %s", chat_id, message_id, error)
    except TelegramError as error:
        logger.warning("DnD edit failed for chat %s msg %s: %s", chat_id, message_id, error)
