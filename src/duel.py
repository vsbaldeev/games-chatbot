"""
Emoji duel game between two chat members.

DuelManager encapsulates all state (pending picks, acceptance, active duels)
and drives the full duel lifecycle. Module-level wrappers preserve the public
API that bot.py imports.
"""

import asyncio
import random
import time
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import RetryAfter, TelegramError
from telegram.ext import ContextTypes

from src import achievements, log
from src.helpers import notify_unlocks

logger = log.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DUEL_ACCEPT_CALLBACK = "duel_accept"
DUEL_REJECT_CALLBACK = "duel_reject"
DUEL_FIRE_CALLBACK = "duel_fire"
DUEL_PICK_CALLBACK = "duel_pick"
DUEL_CALLBACK_PATTERN = r"^duel_"

DUEL_PICK_TIMEOUT = 60
DUEL_ACCEPTANCE_TIMEOUT = 30
DUEL_COUNTDOWN_SECONDS = 10
DUEL_FIRE_TIMEOUT = 300

# ---------------------------------------------------------------------------
# Message pools
# ---------------------------------------------------------------------------

DUEL_CHALLENGE = [
    "⚔️ @{p1} вызывает @{p2} на дуэль!\n\n@{p2}, принять вызов?",
    "🔫 @{p1} бросает вызов @{p2}!\n\n@{p2}, ответишь?",
    "💀 @{p1} не боится @{p2}. Дуэль?\n\n@{p2}, твой ход!",
]

DUEL_REJECTED = [
    "🏳️ @{p2} отклонил вызов @{p1}. Трус.",
    "❌ @{p2} испугался и отказался от дуэли с @{p1}.",
    "😂 @{p2} убежал. @{p1} остался стоять в одиночестве.",
]

DUEL_NO_ANSWER = [
    "⏳ @{p2} не ответил на вызов @{p1}. Время вышло.",
    "💨 Тишина. @{p2} проигнорировал вызов @{p1}.",
]

DUEL_MISS = [
    "💨 @{shooter} промахнулся!",
    "😅 Мимо! @{shooter} не попал.",
    "🌀 У @{shooter} дрогнула рука — промах!",
    "🔒 @{shooter} не снял предохранитель. Выстрела не было.",
    "🔒 Щелчок. @{shooter} забыл про предохранитель — тишина.",
]

DUEL_SELF = [
    "🦶 @{shooter} прострелил себе ногу. Дарвиновская премия почти в кармане. @{other} победил!",
    "🧠 @{shooter} тщательно прицелился... в себя. Годы тренировок не прошли даром. @{other} побеждает!",
    "🔄 Пуля @{shooter} обогнула весь чат и вернулась к отправителю. @{other} одержал победу.",
    "🔄 @{shooter} выстрелил так криво, что пуля решила вернуться. Физика сказала «нет». @{other} выиграл.",
]

DUEL_HIT = [
    "💥 @{shooter} попал в @{target}! Победитель: @{shooter}!",
    "🎯 Точный выстрел @{shooter} в @{target}! @{shooter} победил!",
    "🔫 @{shooter} поразил @{target}! Дуэль окончена!",
]

DUEL_EXPIRED = [
    "⏳ Время вышло. Трусы.",
    "💨 Оба испугались курка.",
    "🏳️ Никто не решился нажать. Позор.",
]


class DuelManager:
    """Manages the full lifecycle of duels across multiple Telegram chats."""

    def __init__(self) -> None:
        # message_id → (chat_id, caller_id, caller_username, candidates)
        self.__pending_picks: dict[int, tuple[int, int, str, list[tuple[int, str]]]] = {}
        self.__pick_jobs: dict[int, Any] = {}

        # message_id → (chat_id, p1_id, p1_username, p2_id, p2_username)
        self.__pending_acceptance: dict[int, tuple[int, int, str, int, str]] = {}
        self.__acceptance_jobs: dict[int, Any] = {}

        # message_id → (chat_id, p1_id, p1_username, p2_id, p2_username, shot_log, shown_at)
        self.__active_duels: dict[int, tuple] = {}
        self.__duel_timeout_jobs: dict[int, Any] = {}

        # At most one active duel per chat.
        self.__active_duel_chats: set[int] = set()

    # ------------------------------------------------------------------
    # Public: command and callback entrypoints
    # ------------------------------------------------------------------

    async def cmd_duel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /duel — show a target-picker keyboard."""
        chat_id = update.effective_chat.id
        members = await achievements.get_chat_members(chat_id)

        caller_id = update.effective_user.id
        caller_username = (
            update.effective_user.username
            or update.effective_user.first_name
            or f"user_{caller_id}"
        )

        candidates = [member for member in members if member[0] != caller_id]
        if not candidates:
            await update.message.reply_text("Нет доступных соперников для дуэли.")
            return

        rows = []
        row: list[InlineKeyboardButton] = []
        for index, (_, uname) in enumerate(candidates):
            row.append(InlineKeyboardButton(
                f"@{uname}", callback_data=f"{DUEL_PICK_CALLBACK}:{index}"
            ))
            if len(row) == 2:
                rows.append(row)
                row = []
        if row:
            rows.append(row)

        keyboard = InlineKeyboardMarkup(rows)
        msg = await update.message.reply_text(
            f"@{caller_username}, выбери соперника для дуэли:",
            reply_markup=keyboard,
        )

        self.__pending_picks[msg.message_id] = (chat_id, caller_id, caller_username, candidates)
        job = context.job_queue.run_once(
            self.__expire_pick, DUEL_PICK_TIMEOUT, data=msg.message_id
        )
        self.__pick_jobs[msg.message_id] = job

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Route duel_ callback queries to the appropriate handler."""
        query = update.callback_query
        if query.data == DUEL_ACCEPT_CALLBACK:
            await self.__handle_accept(query, context)
        elif query.data == DUEL_REJECT_CALLBACK:
            await self.__handle_reject(query, context)
        elif query.data == DUEL_FIRE_CALLBACK:
            await self.__handle_fire(query, context)
        elif query.data.startswith(DUEL_PICK_CALLBACK + ":"):
            await self.__handle_pick(query, context)

    # ------------------------------------------------------------------
    # Private: callback handlers
    # ------------------------------------------------------------------

    async def __handle_pick(self, query: Any, context: ContextTypes.DEFAULT_TYPE) -> None:
        message_id = query.message.message_id
        clicker_id = query.from_user.id

        pick_data = self.__pending_picks.get(message_id)
        if not pick_data:
            await query.answer("Выбор уже недействителен.", show_alert=True)
            return

        chat_id, caller_id, caller_username, candidates = pick_data

        if clicker_id != caller_id:
            await query.answer("Это не твоя дуэль, зритель! 👀", show_alert=True)
            return

        index = int(query.data.split(":")[1])
        if index >= len(candidates):
            await query.answer("Ошибка выбора.", show_alert=True)
            return

        target_id, target_username = candidates[index]

        self.__pending_picks.pop(message_id, None)
        job = self.__pick_jobs.pop(message_id, None)
        if job:
            job.schedule_removal()

        await query.answer()
        try:
            await query.edit_message_text(
                f"⚔️ @{caller_username} вызывает @{target_username} на дуэль!",
                reply_markup=None,
            )
        except TelegramError:
            pass

        started = await self.__send_challenge(
            context, chat_id, caller_id, caller_username, target_id, target_username,
        )
        if not started:
            await context.bot.send_message(
                chat_id, "В чате уже идёт дуэль. Дождитесь её завершения."
            )

    async def __handle_accept(self, query: Any, context: ContextTypes.DEFAULT_TYPE) -> None:
        message_id = query.message.message_id
        clicker_id = query.from_user.id

        claimed = self.__pending_acceptance.pop(message_id, None)
        if not claimed:
            await query.answer("Вызов уже недействителен.", show_alert=True)
            return

        chat_id, p1_id, p1_username, p2_id, p2_username = claimed

        if clicker_id != p2_id:
            self.__pending_acceptance[message_id] = claimed
            if clicker_id == p1_id:
                await query.answer("Ты уже бросил вызов, жди ответа!", show_alert=True)
            else:
                await query.answer("Это не твоя дуэль, зритель! 👀", show_alert=True)
            return

        job = self.__acceptance_jobs.pop(message_id, None)
        if job:
            job.schedule_removal()

        await query.answer("Вызов принят! Готовься к бою!")

        context.job_queue.run_once(
            self.__countdown_and_activate,
            0,
            data=(message_id, chat_id, p1_id, p1_username, p2_id, p2_username),
        )

    async def __handle_reject(self, query: Any, context: ContextTypes.DEFAULT_TYPE) -> None:
        message_id = query.message.message_id
        clicker_id = query.from_user.id

        claimed = self.__pending_acceptance.pop(message_id, None)
        if not claimed:
            await query.answer("Вызов уже недействителен.", show_alert=True)
            return

        chat_id, p1_id, p1_username, p2_id, p2_username = claimed

        if clicker_id != p2_id:
            self.__pending_acceptance[message_id] = claimed
            if clicker_id == p1_id:
                await query.answer("Нельзя отклонить собственный вызов!", show_alert=True)
            else:
                await query.answer("Это не твоя дуэль, зритель! 👀", show_alert=True)
            return

        job = self.__acceptance_jobs.pop(message_id, None)
        if job:
            job.schedule_removal()

        self.__active_duel_chats.discard(chat_id)
        await query.answer("Вызов отклонён.")

        reject_text = self.__fmt(random.choice(DUEL_REJECTED), p1=p1_username, p2=p2_username)
        try:
            await query.edit_message_text(reject_text)
        except TelegramError:
            pass

    async def __handle_fire(self, query: Any, context: ContextTypes.DEFAULT_TYPE) -> None:
        message_id = query.message.message_id
        clicker_id = query.from_user.id

        claimed = self.__active_duels.pop(message_id, None)
        if not claimed:
            await query.answer("Дуэль уже завершена.", show_alert=True)
            return

        chat_id, p1_id, p1_username, p2_id, p2_username, shot_log, button_shown_at = claimed
        elapsed = time.monotonic() - button_shown_at

        if clicker_id not in (p1_id, p2_id):
            self.__active_duels[message_id] = claimed
            await query.answer("Это не твоя дуэль, зритель! 👀", show_alert=True)
            return

        shooter_username = (
            query.from_user.username or query.from_user.first_name or f"user_{clicker_id}"
        )
        other_id = p2_id if clicker_id == p1_id else p1_id
        other_username = p2_username if clicker_id == p1_id else p1_username

        outcome = random.choices(["hit", "self", "miss"], weights=[60, 20, 20])[0]
        elapsed_str = f"⚡ {elapsed:.2f} сек"

        if outcome == "miss":
            miss_line = (
                f"{self.__fmt(random.choice(DUEL_MISS), shooter=shooter_username)} ({elapsed_str})"
            )
            updated_log = shot_log + [miss_line]
            # Re-insert before any await so the duel stays claimable by the next shooter.
            self.__active_duels[message_id] = (
                chat_id, p1_id, p1_username, p2_id, p2_username, updated_log, time.monotonic()
            )
            await query.answer("💨 Промах! Теперь ход соперника!")
            keyboard = InlineKeyboardMarkup(
                [[InlineKeyboardButton("🔫", callback_data=DUEL_FIRE_CALLBACK)]]
            )
            try:
                await query.edit_message_text(
                    self.__build_duel_text(p1_username, p2_username, updated_log),
                    reply_markup=keyboard,
                )
            except TelegramError:
                pass

        elif outcome == "self":
            self_line = (
                f"{self.__fmt(random.choice(DUEL_SELF), shooter=shooter_username, other=other_username)}"
                f" ({elapsed_str})"
            )
            updated_log = shot_log + [self_line]
            await query.answer("😵 Ты выстрелил себе! Дуэль проиграна!")
            await self.__finish_duel(
                query, chat_id, message_id,
                self.__build_duel_text(p1_username, p2_username, updated_log),
            )
            await achievements.increment_stat(other_id, chat_id, other_username, "duel_wins")
            await notify_unlocks(context, chat_id, other_id, other_username)

        else:  # hit
            hit_line = (
                f"{self.__fmt(random.choice(DUEL_HIT), shooter=shooter_username, target=other_username)}"
                f" ({elapsed_str})"
            )
            updated_log = shot_log + [hit_line]
            await query.answer("💥 БАХ! Ты попал!")
            await self.__finish_duel(
                query, chat_id, message_id,
                self.__build_duel_text(p1_username, p2_username, updated_log),
            )
            await achievements.increment_stat(clicker_id, chat_id, shooter_username, "duel_wins")
            await notify_unlocks(context, chat_id, clicker_id, shooter_username)

    # ------------------------------------------------------------------
    # Private: job callbacks
    # ------------------------------------------------------------------

    async def __countdown_and_activate(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        message_id, chat_id, p1_id, p1_username, p2_id, p2_username = context.job.data
        header = self.__duel_header(p1_username, p2_username)

        for seconds_left in range(DUEL_COUNTDOWN_SECONDS, 0, -2):
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=f"{header}\n\n⏱ Дуэль начнётся через {seconds_left}...",
                )
            except RetryAfter as error:
                await asyncio.sleep(error.retry_after)
            except TelegramError:
                pass
            await asyncio.sleep(2)

        keyboard = InlineKeyboardMarkup(
            [[InlineKeyboardButton("🔫", callback_data=DUEL_FIRE_CALLBACK)]]
        )
        activated = False
        for _ in range(5):
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=self.__build_duel_text(p1_username, p2_username, []),
                    reply_markup=keyboard,
                )
                activated = True
                break
            except RetryAfter as error:
                await asyncio.sleep(error.retry_after)
            except TelegramError as error:
                logger.warning("Failed to activate duel %s: %s", message_id, error)
                break

        if not activated:
            self.__active_duel_chats.discard(chat_id)
            return

        self.__active_duels[message_id] = (
            chat_id, p1_id, p1_username, p2_id, p2_username, [], time.monotonic()
        )
        job = context.job_queue.run_once(
            self.__expire_duel, DUEL_FIRE_TIMEOUT, data=message_id
        )
        self.__duel_timeout_jobs[message_id] = job

    async def __expire_pick(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        message_id = context.job.data
        pick_data = self.__pending_picks.pop(message_id, None)
        self.__pick_jobs.pop(message_id, None)
        if not pick_data:
            return

        chat_id, _, caller_username, _ = pick_data
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=f"⏳ @{caller_username} не выбрал соперника. Время вышло.",
                reply_markup=None,
            )
        except TelegramError as error:
            logger.warning("Pick expiry failed for msg %s: %s", message_id, error)

    async def __expire_acceptance(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        message_id = context.job.data
        duel_data = self.__pending_acceptance.pop(message_id, None)
        self.__acceptance_jobs.pop(message_id, None)
        if not duel_data:
            return

        chat_id, _, p1_username, _, p2_username = duel_data
        self.__active_duel_chats.discard(chat_id)
        text = self.__fmt(random.choice(DUEL_NO_ANSWER), p1=p1_username, p2=p2_username)
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                reply_markup=None,
            )
        except TelegramError as error:
            logger.warning("Acceptance expiry failed for msg %s: %s", message_id, error)

    async def __expire_duel(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        message_id = context.job.data
        duel_data = self.__active_duels.pop(message_id, None)
        self.__duel_timeout_jobs.pop(message_id, None)
        if not duel_data:
            return

        chat_id, _, p1_username, _, p2_username, shot_log = duel_data[:6]
        self.__active_duel_chats.discard(chat_id)
        full_text = self.__build_duel_text(
            p1_username, p2_username, shot_log + [random.choice(DUEL_EXPIRED)]
        )
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=full_text,
                reply_markup=None,
            )
        except TelegramError as error:
            logger.warning("Duel expiry failed for msg %s: %s", message_id, error)

    # ------------------------------------------------------------------
    # Private: helpers
    # ------------------------------------------------------------------

    async def __send_challenge(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        p1_id: int,
        p1_username: str,
        p2_id: int,
        p2_username: str,
        reply_to_message_id: int | None = None,
    ) -> bool:
        """Send the challenge message. Returns False if a duel is already running in this chat."""
        if chat_id in self.__active_duel_chats:
            return False
        self.__active_duel_chats.add(chat_id)

        challenge_text = self.__fmt(
            random.choice(DUEL_CHALLENGE), p1=p1_username, p2=p2_username
        )
        challenge_text += f"\n\n⏱ У @{p2_username} есть {DUEL_ACCEPTANCE_TIMEOUT} сек."

        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Принять", callback_data=DUEL_ACCEPT_CALLBACK),
            InlineKeyboardButton("❌ Отклонить", callback_data=DUEL_REJECT_CALLBACK),
        ]])

        msg = await context.bot.send_message(
            chat_id=chat_id,
            text=challenge_text,
            reply_markup=keyboard,
            reply_to_message_id=reply_to_message_id,
        )

        self.__pending_acceptance[msg.message_id] = (
            chat_id, p1_id, p1_username, p2_id, p2_username
        )
        job = context.job_queue.run_once(
            self.__expire_acceptance, DUEL_ACCEPTANCE_TIMEOUT, data=msg.message_id
        )
        self.__acceptance_jobs[msg.message_id] = job
        return True

    async def __finish_duel(
        self, query: Any, chat_id: int, message_id: int, text: str
    ) -> None:
        job = self.__duel_timeout_jobs.pop(message_id, None)
        if job:
            job.schedule_removal()
        self.__active_duel_chats.discard(chat_id)
        try:
            await query.edit_message_text(text)
        except TelegramError:
            pass

    @staticmethod
    def __fmt(template: str, **kwargs: str) -> str:
        result = template
        for key, value in kwargs.items():
            result = result.replace("{" + key + "}", value)
        return result

    @staticmethod
    def __duel_header(p1_username: str, p2_username: str) -> str:
        return f"⚔️ @{p1_username} vs @{p2_username}"

    @staticmethod
    def __build_duel_text(p1_username: str, p2_username: str, shot_log: list[str]) -> str:
        header = f"⚔️ @{p1_username} vs @{p2_username}"
        if not shot_log:
            return header
        return header + "\n\n" + "\n".join(shot_log)


# ---------------------------------------------------------------------------
# Module-level singleton + backward-compatible wrappers
# ---------------------------------------------------------------------------

duel_manager = DuelManager()


async def cmd_duel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await duel_manager.cmd_duel(update, context)


async def handle_duel_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await duel_manager.handle_callback(update, context)
