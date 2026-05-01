"""
DndManager: stateful singleton that owns all lobby and game lifecycle.

All Telegram command handlers and callback handlers are methods on this class.
Module-level wrappers in __init__.py expose the same public names that bot.py
imports, so no callers need to change.
"""

import random
from typing import Any

from telegram import Update
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from src import achievements, log
from src.dnd.llm import ScenarioGenerator
from src.dnd.state import (
    LobbyState,
    ActiveGame,
    DND_JOIN_CALLBACK,
    DND_ACTION_CALLBACK_PREFIX,
    DND_MIN_PLAYERS,
    DND_ACTION_TIMEOUT,
    DND_LOBBY_TIMEOUT,
    DND_BOT_PLAYER_ID,
    DND_BOT_PLAYER_NAME,
)
from src.dnd.views import (
    build_lobby_text,
    build_lobby_keyboard,
    build_game_text,
    build_game_keyboard,
    edit_safe,
    format_roll_lines,
    heist_phase_name,
)

logger = log.get_logger(__name__)


class DndManager:
    """Manages all state and logic for D&D lobby/game sessions across chats."""

    def __init__(self) -> None:
        self.__lobbies: dict[int, tuple[LobbyState, int, str]] = {}
        self.__active_games: dict[int, ActiveGame] = {}
        self.__active_chats: set[int] = set()
        self.__lobby_timeout_jobs: dict[int, Any] = {}
        self.__action_timeout_jobs: dict[int, Any] = {}
        self.__llm = ScenarioGenerator()

    # ------------------------------------------------------------------
    # Public: command entrypoints
    # ------------------------------------------------------------------

    async def cmd_pvp(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /dnd_pvp — start a 1-round PvP lobby."""
        await self.__start_lobby(update, context, max_rounds=1, mode="pvp")

    async def cmd_coop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /dnd_coop — start a 2-round coop lobby."""
        await self.__start_lobby(update, context, max_rounds=2, mode="coop")

    async def cmd_heist(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /dnd_heist — start a 3-round heist lobby."""
        await self.__start_lobby(update, context, max_rounds=3, mode="heist")

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Route dnd_ callback queries to the appropriate handler."""
        query = update.callback_query
        data = query.data

        if data == DND_JOIN_CALLBACK:
            await self.__handle_join(query, context)
        elif data.startswith(DND_ACTION_CALLBACK_PREFIX):
            try:
                action_index = int(data[len(DND_ACTION_CALLBACK_PREFIX):])
            except ValueError:
                await query.answer()
                return
            await self.__handle_action(query, context, action_index)
        else:
            await query.answer()

    # ------------------------------------------------------------------
    # Private: lobby management
    # ------------------------------------------------------------------

    async def __start_lobby(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        max_rounds: int,
        mode: str,
    ) -> None:
        chat_id = update.effective_chat.id

        if chat_id in self.__active_chats:
            await update.message.reply_text(
                "⚔️ В чате уже идёт D&D-приключение. Дождитесь его завершения."
            )
            return

        initiator = update.effective_user
        initiator_id = initiator.id
        initiator_username = initiator.username or initiator.first_name or f"user_{initiator_id}"

        # Fill the roster with a bot NPC when there aren't enough real players.
        members = await achievements.get_chat_members(chat_id)
        initial_players: list[tuple[int, str]] = [(initiator_id, initiator_username)]
        if len(members) < DND_MIN_PLAYERS:
            initial_players.append((DND_BOT_PLAYER_ID, DND_BOT_PLAYER_NAME))

        self.__active_chats.add(chat_id)
        lobby = LobbyState(
            chat_id=chat_id,
            message_id=0,
            initiator_id=initiator_id,
            players=initial_players,
        )

        msg = await update.message.reply_text(
            build_lobby_text(lobby, max_rounds, mode),
            reply_markup=build_lobby_keyboard(),
        )
        lobby.message_id = msg.message_id
        self.__lobbies[chat_id] = (lobby, max_rounds, mode)

        job = context.job_queue.run_once(self.__expire_lobby, DND_LOBBY_TIMEOUT, data=chat_id)
        self.__lobby_timeout_jobs[chat_id] = job

    async def __handle_join(self, query: Any, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = query.message.chat_id
        clicker_id = query.from_user.id
        clicker_username = (
            query.from_user.username
            or query.from_user.first_name
            or f"user_{clicker_id}"
        )

        entry = self.__lobbies.get(chat_id)
        if not entry:
            await query.answer("Лобби уже закрыто.", show_alert=True)
            return
        lobby, max_rounds, mode = entry

        if any(player_id == clicker_id for player_id, _ in lobby.players):
            await query.answer("Ты уже в отряде!", show_alert=True)
            return

        lobby.players.append((clicker_id, clicker_username))

        if len(lobby.players) < DND_MIN_PLAYERS:
            await query.answer()
            await edit_safe(
                context.bot,
                query.message.chat_id,
                query.message.message_id,
                build_lobby_text(lobby, max_rounds, mode),
                build_lobby_keyboard(),
            )
            return

        # Minimum reached — atomically claim the lobby before any await to prevent double-start.
        popped = self.__lobbies.pop(chat_id, None)
        if popped is None:
            await query.answer()
            return

        job = self.__lobby_timeout_jobs.pop(chat_id, None)
        if job:
            job.schedule_removal()

        players = list(lobby.players)
        message_id = query.message.message_id

        await query.answer("Отряд собран! Начинаем приключение!")

        try:
            await query.edit_message_text("⚔️ D&D Приключение\n\n🎲 Генерация сценария...")
        except TelegramError:
            pass

        context.job_queue.run_once(
            self.__start_game_job,
            0,
            data=(chat_id, message_id, players, max_rounds, mode),
        )

    async def __handle_action(
        self, query: Any, context: ContextTypes.DEFAULT_TYPE, action_index: int
    ) -> None:
        chat_id = query.message.chat_id
        clicker_id = query.from_user.id

        game = self.__active_games.get(chat_id)
        if not game:
            await query.answer("Игра уже завершена.", show_alert=True)
            return

        player_ids = {player_id for player_id, _ in game.players}
        if clicker_id not in player_ids:
            await query.answer("Ты не в этой партии, зритель! 👀", show_alert=True)
            return

        if clicker_id in game.choices:
            chosen = game.actions[game.choices[clicker_id]]
            await query.answer(f"Ты уже выбрал: {chosen}", show_alert=True)
            return

        if action_index >= len(game.actions):
            await query.answer("Неверное действие.", show_alert=True)
            return

        game.choices[clicker_id] = action_index
        await query.answer(f"Выбрано: {game.actions[action_index]}")

        all_chosen = len(game.choices) == len(game.players)

        await edit_safe(
            context.bot,
            chat_id,
            game.message_id,
            build_game_text(game),
            keyboard=build_game_keyboard(game),
        )

        if all_chosen:
            resolved_game = self.__active_games.pop(chat_id, None)
            if not resolved_game:
                return
            job = self.__action_timeout_jobs.pop(chat_id, None)
            if job:
                job.schedule_removal()
            context.job_queue.run_once(self.__resolve_game_job, 0, data=(chat_id, resolved_game))

    # ------------------------------------------------------------------
    # Private: job callbacks
    # ------------------------------------------------------------------

    async def __start_game_job(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id, message_id, players, max_rounds, mode = context.job.data

        boss_name = ""
        boss_max_hp = 0

        try:
            if mode == "coop":
                scenario, boss_name, actions = await self.__llm.generate_coop_round(
                    len(players), round_number=1,
                    boss_name="", boss_hp=0, boss_max_hp=0, history=[],
                )
                boss_max_hp = random.randint(len(players) * 15, len(players) * 20)
            else:
                scenario, actions = await self.__llm.generate_round(
                    len(players), round_number=1, max_rounds=max_rounds,
                    mode=mode, history=[], players=players,
                )
        except Exception as error:
            logger.warning("DnD scenario generation failed for chat %s: %s", chat_id, error)
            self.__active_chats.discard(chat_id)
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text="⚔️ Волшебник сценариев ушёл на перекур. Попробуйте снова.",
                )
            except TelegramError:
                pass
            return

        game = ActiveGame(
            chat_id=chat_id,
            message_id=message_id,
            scenario=scenario,
            actions=actions,
            players=players,
            max_rounds=max_rounds,
            round_number=1,
            mode=mode,
            boss_name=boss_name,
            boss_hp=boss_max_hp,
            boss_max_hp=boss_max_hp,
        )
        self.__assign_bot_choice(game)
        self.__active_games[chat_id] = game

        await edit_safe(
            context.bot, chat_id, message_id,
            build_game_text(game),
            keyboard=build_game_keyboard(game),
        )

        job = context.job_queue.run_once(self.__expire_actions, DND_ACTION_TIMEOUT, data=chat_id)
        self.__action_timeout_jobs[chat_id] = job

    async def __expire_actions(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = context.job.data
        game = self.__active_games.pop(chat_id, None)
        self.__action_timeout_jobs.pop(chat_id, None)
        if not game:
            return
        context.job_queue.run_once(self.__resolve_game_job, 0, data=(chat_id, game))

    async def __resolve_game_job(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id, game = context.job.data

        player_results = []
        for player_id, username in game.players:
            roll = random.randint(1, 20)
            action_index = game.choices.get(player_id)
            action = game.actions[action_index] if action_index is not None else "бездействовал"
            player_results.append({"username": username, "roll": roll, "action": action})

        is_final = game.round_number >= game.max_rounds

        if game.mode == "coop":
            await self.__resolve_coop_round(context, chat_id, game, player_results, is_final)
        else:
            await self.__resolve_standard_round(context, chat_id, game, player_results, is_final)

    async def __resolve_coop_round(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        game: ActiveGame,
        player_results: list[dict],
        is_final: bool,
    ) -> None:
        total_damage = sum(result["roll"] for result in player_results)
        game.boss_hp = max(0, game.boss_hp - total_damage)
        players_won = game.boss_hp <= 0

        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=game.message_id,
                text=f"⚔️ *D&D Кооп — Раунд {game.round_number}/{game.max_rounds}*\n\n🎲 Подводим итоги...",
                parse_mode="Markdown",
            )
        except TelegramError:
            pass

        try:
            narrative = await self.__llm.generate_coop_narrative(
                scenario=game.scenario,
                player_results=player_results,
                boss_name=game.boss_name,
                damage_this_round=total_damage,
                boss_hp_after=game.boss_hp,
                boss_max_hp=game.boss_max_hp,
                history=game.history,
                players_won=players_won,
                is_final=is_final,
            )
        except Exception as error:
            logger.warning("DnD coop narrative failed for chat %s: %s", chat_id, error)
            narrative = "Летописец выронил перо в разгар битвы. Но отряд устоял. Кажется."

        roll_lines = format_roll_lines(player_results)

        if is_final:
            if players_won:
                round_title = "Победа! 🏆"
                boss_line = f"👹 {game.boss_name} повержен!"
            else:
                round_title = "Поражение 💀"
                boss_line = f"👹 {game.boss_name} выжил... ({game.boss_hp} HP осталось)"
            damage_line = (
                f"💥 Финальный урон: {total_damage} — "
                f"итого {game.boss_max_hp - game.boss_hp}/{game.boss_max_hp}"
            )
        else:
            round_title = f"Раунд {game.round_number}/{game.max_rounds} — Итог"
            boss_line = f"👹 {game.boss_name} — ❤️ {game.boss_hp}/{game.boss_max_hp} HP осталось"
            damage_line = f"💥 Суммарный урон: {total_damage}"

        result_text = (
            f"⚔️ D&D Кооп — {round_title}\n\n"
            f"{narrative}\n\n"
            f"{damage_line}\n"
            f"{boss_line}\n\n"
            f"🎲 Броски:\n" + "\n".join(roll_lines)
        )

        await edit_safe(context.bot, chat_id, game.message_id, result_text)

        if is_final:
            self.__active_chats.discard(chat_id)
            return

        self.__advance_round(game, narrative, player_results)
        context.job_queue.run_once(self.__next_round_job, 0, data=(chat_id, game))

    async def __resolve_standard_round(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        game: ActiveGame,
        player_results: list[dict],
        is_final: bool,
    ) -> None:
        if game.mode == "heist":
            loading_header = f"🎩 Великое Ограбление — {heist_phase_name(game.round_number)}"
        else:
            loading_header = f"⚔️ D&D — Раунд {game.round_number}/{game.max_rounds}"

        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=game.message_id,
                text=f"{loading_header}\n\n🎲 Подводим итоги...",
            )
        except TelegramError:
            pass

        try:
            narrative = await self.__llm.generate_narrative(
                game.scenario, player_results, game.history, is_final,
                is_pvp=(game.mode == "pvp"),
                is_heist=(game.mode == "heist"),
            )
        except Exception as error:
            logger.warning("DnD narrative failed for chat %s: %s", chat_id, error)
            narrative = "Летописец выронил перо и всё размазалось. Но все выжили. Кажется."

        roll_lines = format_roll_lines(player_results)

        if game.mode == "heist":
            heist_result_titles = {
                1: "Проникновение — Итог",
                2: "Дело — Итог",
                3: "Побег — Финал",
            }
            round_title = heist_result_titles.get(game.round_number, f"Фаза {game.round_number} — Итог")
            prefix = "🎩 Великое Ограбление"
        else:
            round_title = "Финал" if is_final else f"Раунд {game.round_number}/{game.max_rounds} — Итог"
            prefix = "⚔️ D&D"

        result_text = (
            f"{prefix} — {round_title}\n\n"
            f"{narrative}\n\n"
            f"🎲 Броски:\n" + "\n".join(roll_lines)
        )

        await edit_safe(context.bot, chat_id, game.message_id, result_text)

        if is_final:
            self.__active_chats.discard(chat_id)
            return

        self.__advance_round(game, narrative, player_results)
        context.job_queue.run_once(self.__next_round_job, 0, data=(chat_id, game))

    async def __next_round_job(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id, game = context.job.data

        if game.mode == "coop":
            loading_text = (
                f"⚔️ D&D Кооп — Раунд {game.round_number}/{game.max_rounds}"
                "\n\n🎲 Генерация продолжения..."
            )
        elif game.mode == "heist":
            loading_text = (
                f"🎩 Великое Ограбление — {heist_phase_name(game.round_number)}"
                "\n\n🎲 Генерация следующей фазы..."
            )
        else:
            loading_text = (
                f"⚔️ D&D — Раунд {game.round_number}/{game.max_rounds}"
                "\n\n🎲 Генерация продолжения..."
            )

        try:
            msg = await context.bot.send_message(chat_id=chat_id, text=loading_text)
            game.message_id = msg.message_id
        except TelegramError as error:
            logger.warning(
                "DnD failed to send round %s message for chat %s: %s",
                game.round_number, chat_id, error,
            )
            self.__active_chats.discard(chat_id)
            return

        try:
            if game.mode == "coop":
                scenario, _, actions = await self.__llm.generate_coop_round(
                    len(game.players), game.round_number,
                    game.boss_name, game.boss_hp, game.boss_max_hp, game.history,
                )
            else:
                scenario, actions = await self.__llm.generate_round(
                    len(game.players), game.round_number, game.max_rounds,
                    game.mode, game.history, game.players,
                )
        except Exception as error:
            logger.warning(
                "DnD round %s generation failed for chat %s: %s",
                game.round_number, chat_id, error,
            )
            self.__active_chats.discard(chat_id)
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=game.message_id,
                    text="⚔️ Волшебник споткнулся на лестнице. Приключение прервано.",
                )
            except TelegramError:
                pass
            return

        game.scenario = scenario
        game.actions = actions
        self.__assign_bot_choice(game)
        self.__active_games[chat_id] = game

        await edit_safe(
            context.bot, chat_id, game.message_id,
            build_game_text(game),
            keyboard=build_game_keyboard(game),
        )

        job = context.job_queue.run_once(self.__expire_actions, DND_ACTION_TIMEOUT, data=chat_id)
        self.__action_timeout_jobs[chat_id] = job

    async def __expire_lobby(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = context.job.data
        entry = self.__lobbies.pop(chat_id, None)
        self.__lobby_timeout_jobs.pop(chat_id, None)
        if not entry:
            return
        lobby, _, _ = entry

        self.__active_chats.discard(chat_id)
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=lobby.message_id,
                text="⚔️ Лобби закрыто — никто не собрал отряд. В следующий раз.",
                reply_markup=None,
            )
        except TelegramError as error:
            logger.warning("DnD lobby expiry failed for chat %s: %s", chat_id, error)

    # ------------------------------------------------------------------
    # Private: game helpers
    # ------------------------------------------------------------------

    def __assign_bot_choice(self, game: ActiveGame) -> None:
        """Pre-populate the bot NPC's action so it always shows ✅ from round start."""
        if any(player_id == DND_BOT_PLAYER_ID for player_id, _ in game.players):
            game.choices[DND_BOT_PLAYER_ID] = random.randrange(len(game.actions))

    def __advance_round(
        self, game: ActiveGame, narrative: str, player_results: list[dict]
    ) -> None:
        game.history.append({
            "scenario": game.scenario,
            "narrative": narrative,
            "results": player_results,
        })
        game.round_number += 1
        game.choices = {}


dnd_manager = DndManager()
