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
from src.helpers import notify_unlocks

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

        announce = random.choice(ROULETTE_ANNOUNCE)
        try:
            msg1 = await context.bot.send_message(
                chat_id=chat_id,
                text=announce,
                reply_to_message_id=reply_to_message_id,
            )
        except Exception as error:
            logger.warning("Roulette announce failed for chat %s: %s", chat_id, error)
            return

        await asyncio.sleep(5)

        victim_id, victim_username = random.choice(members)
        pick_msg = random.choice(ROULETTE_PICK).format(username=victim_username)
        try:
            msg2 = await context.bot.send_message(
                chat_id=chat_id,
                text=pick_msg,
                parse_mode="Markdown",
                reply_to_message_id=msg1.message_id,
            )
        except Exception as error:
            logger.warning("Roulette pick failed for chat %s: %s", chat_id, error)
            return

        await asyncio.sleep(5)

        shot = random.random() < 0.5
        result_pool = ROULETTE_HIT if shot else ROULETTE_MISS
        result_msg = random.choice(result_pool).format(username=victim_username)
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=result_msg,
                parse_mode="Markdown",
                reply_to_message_id=msg2.message_id,
            )
        except Exception as error:
            logger.warning("Roulette result failed for chat %s: %s", chat_id, error)
            return

        if not shot:
            await achievements.increment_stat(
                victim_id, chat_id, victim_username, "roulette_win_count"
            )
            await notify_unlocks(context, chat_id, victim_id, victim_username)

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
