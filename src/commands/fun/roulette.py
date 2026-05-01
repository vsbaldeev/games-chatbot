"""
Daily Russian roulette game.

Roulette encapsulates all game logic and Telegram messaging.
Module-level wrappers preserve the public API that bot.py imports.
"""

import asyncio
import random

from telegram import Update
from telegram.ext import ContextTypes

from src import achievements, log
from src.achievements import notify_unlocks

logger = log.get_logger(__name__)

# ---------------------------------------------------------------------------
# Message pools
# ---------------------------------------------------------------------------

ROULETTE_ANNOUNCE = [
    "🎰 Внимание, чат. Сегодня — русская рулетка. Барабан заряжен. Кто-то сегодня не жилец.",
    "🔫 Время пришло. Еженедельная рулетка объявляется открытой. Один патрон, один участник.",
    "💀 Русская рулетка запущена. Удача — она такая: сегодня есть, завтра нет.",
    "🎲 Ритуал начинается. Барабан крутится. Кто-то из вас сегодня — жертва случая.",
    "🔫 Господа геймеры. Рулетка. Один выстрел. Один из вас. Начинаем.",
]

ROULETTE_PICK = [
    "🎯 Барабан остановился... жертва выбрана: *@{username}*",
    "🔍 Рулетка определилась. Сегодняшний избранник — *@{username}*",
    "👆 Палец судьбы указывает на *@{username}*. Ну что ж.",
    "🎰 Случайность сделала выбор. Это *@{username}*. Соболезнуем заранее.",
    "🎯 Имя вытащено из барабана: *@{username}*. Удача покинула чат.",
]

ROULETTE_HIT = [
    "🔫 *БАХ!* 💀 @{username} сегодня не повезло. Барабан не врёт.",
    "🔫 Три... два... один... *ВЫСТРЕЛ!* @{username} поймал пулю. Ничего личного — просто статистика.",
    "🎰 Пуля нашла хозяина. 💀 @{username} — до следующей недели.",
    "🔫 Щёлк. *БАМ.* @{username} сегодня в минусе. Рандом не обсуждается.",
    "🎯 Прямо в @{username}. 🔫 Меткость — 100%, удача — 0%.",
    "💀 @{username} — всё. Барабан не соврёт.",
]

ROULETTE_MISS = [
    "🔫 Барабан крутится... @{username}... *клик.* Осечка. Повезло — живи пока.",
    "😮‍💨 @{username} — *клик.* Пусто. Сегодня фартануло.",
    "🔫 Три... два... один... *клик.* @{username} выдохнул. Патрона не было.",
    "😅 @{username} смотрит в ствол... *клик.* Обошлось. На этот раз.",
    "🍀 @{username} — *клик.* Мимо. Видимо, не судьба сегодня.",
    "🔫 @{username}. Пустой патронник. Рулетка решила пощадить — в этот раз.",
]


class Roulette:
    """Runs the Russian roulette game for one or all chats."""

    async def run_all_chats(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Run the daily roulette for every registered chat concurrently."""
        chat_ids = await achievements.get_all_chat_ids()
        await asyncio.gather(
            *[self.run_for_chat(context, chat_id) for chat_id in chat_ids],
            return_exceptions=True,
        )

    async def __send_announce(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        reply_to_message_id: int | None,
    ) -> int | None:
        """Send the opening announcement. Returns the message_id or None on failure."""
        announce = random.choice(ROULETTE_ANNOUNCE)
        try:
            msg = await context.bot.send_message(
                chat_id=chat_id,
                text=announce,
                reply_to_message_id=reply_to_message_id,
            )
            return msg.message_id
        except Exception as error:
            logger.warning("Roulette announce failed for chat %s: %s", chat_id, error)
            return None

    async def __send_pick(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        members: list[tuple[int, str]],
        prev_message_id: int,
    ) -> tuple[int, str, int] | None:
        """Pick a victim and send the pick message. Returns (victim_id, username, msg_id) or None."""
        victim_id, victim_username = random.choice(members)
        pick_msg = random.choice(ROULETTE_PICK).format(username=victim_username)
        try:
            msg = await context.bot.send_message(
                chat_id=chat_id,
                text=pick_msg,
                parse_mode="Markdown",
                reply_to_message_id=prev_message_id,
            )
            return victim_id, victim_username, msg.message_id
        except Exception as error:
            logger.warning("Roulette pick failed for chat %s: %s", chat_id, error)
            return None

    async def __send_shot_result(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        victim_id: int,
        victim_username: str,
        prev_message_id: int,
    ) -> None:
        """Fire the shot, send the result, and credit the achievement if survived."""
        shot = random.random() < 0.5
        result_pool = ROULETTE_HIT if shot else ROULETTE_MISS
        result_msg = random.choice(result_pool).format(username=victim_username)
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=result_msg,
                parse_mode="Markdown",
                reply_to_message_id=prev_message_id,
            )
        except Exception as error:
            logger.warning("Roulette result failed for chat %s: %s", chat_id, error)
            return
        if not shot:
            await achievements.increment_stat(
                victim_id, chat_id, victim_username, "roulette_win_count"
            )
            await notify_unlocks(context, chat_id, victim_id, victim_username)

    async def run_for_chat(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        reply_to_message_id: int | None = None,
    ) -> None:
        """Run one roulette round for the given chat."""
        members = await achievements.get_chat_members(chat_id)
        if len(members) < 2:
            return

        announce_id = await self.__send_announce(context, chat_id, reply_to_message_id)
        if announce_id is None:
            return

        await asyncio.sleep(5)

        pick_result = await self.__send_pick(context, chat_id, members, announce_id)
        if pick_result is None:
            return

        victim_id, victim_username, pick_msg_id = pick_result
        await asyncio.sleep(5)
        await self.__send_shot_result(context, chat_id, victim_id, victim_username, pick_msg_id)

    async def cmd_roulette(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /roulette — run a roulette round on demand."""
        chat_id = update.effective_chat.id
        members = await achievements.get_chat_members(chat_id)
        if len(members) < 2:
            await update.message.reply_text(
                "Не с кем играть. Нужно хотя бы 2 участника в чате."
            )
            return
        await self.run_for_chat(
            context, chat_id, reply_to_message_id=update.message.message_id
        )


# ---------------------------------------------------------------------------
# Module-level singleton + backward-compatible wrappers
# ---------------------------------------------------------------------------

roulette_game = Roulette()


async def russian_roulette(context: ContextTypes.DEFAULT_TYPE) -> None:
    await roulette_game.run_all_chats(context)


async def cmd_roulette(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await roulette_game.cmd_roulette(update, context)
