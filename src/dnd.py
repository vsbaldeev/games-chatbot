"""
D&D-style party game. Three modes:

  /dnd_pvp   — 1 round, players compete against each other (PvP)
  /dnd_coop  — 2 rounds, all players vs. a boss NPC (coop)
  /dnd_heist      — 3 rounds, stealth/trickery heist (The Great Heist)

If the chat has fewer than DND_MIN_PLAYERS registered members, the bot joins
as an NPC character to fill the roster so two real users can still play.

Flow (all modes):
  lobby (join button, auto-start at DND_MIN_PLAYERS players)
  →  LLM generates scenario + 4 action buttons
  →  45-second voting phase per round
  →  bot rolls d20 per player  →  LLM narrates the outcome
  →  repeat until final round, each in its own message
"""

import asyncio
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest, TelegramError
from telegram.ext import ContextTypes

from src import achievements, config

logger = logging.getLogger(__name__)

DND_JOIN_CALLBACK = "dnd_join"
DND_ACTION_CALLBACK_PREFIX = "dnd_a"
DND_CALLBACK_PATTERN = r"^dnd_"

DND_MODEL = "llama-3.3-70b-versatile"
DND_MIN_PLAYERS = 3
DND_ACTION_TIMEOUT = 45
DND_LOBBY_TIMEOUT = 300
DND_LLM_TIMEOUT = 30
DND_BOT_PLAYER_ID = 0           # sentinel user_id — no real Telegram user has ID 0
DND_BOT_PLAYER_NAME = "ДнД-Бот"

__ACTION_RE = re.compile(r"^Д([1-3]):\s*(.+)$")


@dataclass
class LobbyState:
    chat_id: int
    message_id: int
    initiator_id: int
    players: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class ActiveGame:
    chat_id: int
    message_id: int
    scenario: str
    actions: list[str]
    players: list[tuple[int, str]]
    max_rounds: int = 3
    round_number: int = 1
    mode: str = "heist"     # "pvp" | "coop" | "heist"
    boss_name: str = ""
    boss_hp: int = 0
    boss_max_hp: int = 0
    choices: dict[int, int] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)


__lobbies: dict[int, tuple[LobbyState, int, str]] = {}
__active_games: dict[int, ActiveGame] = {}
__active_chats: set[int] = set()
__lobby_timeout_jobs: dict[int, Any] = {}
__action_timeout_jobs: dict[int, Any] = {}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def __llm(max_tokens: int = 300) -> ChatGroq:
    return ChatGroq(
        model=DND_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0.95,
        max_tokens=max_tokens,
    )


def __heist_phase_name(round_number: int) -> str:
    return {1: "Проникновение", 2: "Дело", 3: "Побег"}.get(round_number, f"Фаза {round_number}")


def __format_roll_lines(player_results: list[dict]) -> list[str]:
    lines = []
    for result in player_results:
        roll = result["roll"]
        mark = " 💀" if roll == 1 else " ✨" if roll == 20 else ""
        lines.append(f'• @{result["username"]}: {result["action"]} → 🎲{roll}{mark}')
    return lines


def __format_history_lines(history: list[dict]) -> str:
    return "\n\n".join(
        f"Раунд {idx + 1}:\nСитуация: {entry['scenario']}\nИтог: {entry['narrative']}"
        for idx, entry in enumerate(history)
    )


def __has_bot_npc(game: ActiveGame) -> bool:
    return any(player_id == DND_BOT_PLAYER_ID for player_id, _ in game.players)


def __assign_bot_choice(game: ActiveGame) -> None:
    """Pre-populate the bot NPC's action so it always shows ✅ from round start."""
    if __has_bot_npc(game):
        game.choices[DND_BOT_PLAYER_ID] = random.randrange(len(game.actions))


def __advance_round(game: ActiveGame, narrative: str, player_results: list[dict]) -> None:
    game.history.append({"scenario": game.scenario, "narrative": narrative, "results": player_results})
    game.round_number += 1
    game.choices = {}


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------

def __build_lobby_text(lobby: LobbyState, max_rounds: int, mode: str) -> str:
    player_lines = "\n".join(f"• @{username}" for _, username in lobby.players)
    count = len(lobby.players)
    remaining = DND_MIN_PLAYERS - count
    status = "готово к старту!" if count >= DND_MIN_PLAYERS else f"нужно ещё {remaining}"

    if mode == "pvp":
        header = "⚔️ D&D — Все против всех (1 раунд)"
    elif mode == "coop":
        header = "⚔️ D&D — Кооп против Босса (2 раунда)"
    else:  # heist
        header = "🎩 D&D — Великое Ограбление (3 раунда)"

    return (
        f"{header}\n\n"
        f"Набирается отряд!\n"
        f"Нужно минимум {DND_MIN_PLAYERS} игрока.\n\n"
        f"Отряд ({count}) — {status}:\n{player_lines}\n\n"
        f"Лобби закроется через 5 минут."
    )


def __build_lobby_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("⚔️ Присоединиться", callback_data=DND_JOIN_CALLBACK),
    ]])


def __build_game_text(game: ActiveGame) -> str:
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
    else:  # heist
        hp_line = ""
        round_header = f"🎩 Великое Ограбление — {__heist_phase_name(game.round_number)}"

    return (
        f"{round_header}\n\n"
        f"{hp_line}"
        f"{game.scenario}\n\n"
        f"У каждого игрока {DND_ACTION_TIMEOUT} секунд на выбор:\n"
        f"{player_lines}"
    )


def __build_game_keyboard(game: ActiveGame) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(action, callback_data=f"{DND_ACTION_CALLBACK_PREFIX}{index}")]
        for index, action in enumerate(game.actions)
    ])


async def __edit_safe(
    bot,
    chat_id: int,
    message_id: int,
    text: str,
    keyboard: InlineKeyboardMarkup | None = None,
) -> None:
    kwargs: dict[str, Any] = {"chat_id": chat_id, "message_id": message_id}
    if keyboard is not None:
        kwargs["reply_markup"] = keyboard
    try:
        await bot.edit_message_text(text=text, **kwargs)
    except BadRequest as error:
        if "not modified" in str(error).lower():
            return
        logger.warning(f"DnD edit failed for chat {chat_id} msg {message_id}: {error}")
    except TelegramError as error:
        logger.warning(f"DnD edit failed for chat {chat_id} msg {message_id}: {error}")


# ---------------------------------------------------------------------------
# LLM: scenario generation
# ---------------------------------------------------------------------------

async def __generate_round(
    player_count: int,
    round_number: int,
    max_rounds: int,
    mode: str,
    history: list[dict],
    players: list[tuple[int, str]] | None = None,
) -> tuple[str, list[str]]:
    system_prompt = (
        "Ты генератор сценариев для D&D-игры в Telegram-чате геймеров.\n"
        "Твой ответ должен быть строго в формате (без лишних слов):\n"
        "СЦЕНАРИЙ: <одно короткое предложение — смешной абсурдный сценарий>\n"
        "Д1: <понятное действие 3-6 слов>\n"
        "Д2: <понятное действие 3-6 слов>\n"
        "Д3: <понятное действие 3-6 слов>\n\n"
        "Только русский язык. Никакого другого текста."
    )

    if not history:
        if mode == "pvp":
            names = ", ".join(f"@{name}" for _, name in (players or [])) or f"{player_count} игроков"
            user_prompt = (
                f"Придумай смешной абсурдный фэнтезийный сценарий прямой драки между {names}. "
                "Они дерутся друг с другом — один на один, все против всех, без NPC-противников. "
                "Примеры: делят последний кусок пирога у дракона, выясняют кто лучший маг в таверне, "
                "спор перерос в магическую дуэль, гонка за артефактом где мешать друг другу — главное. "
                "Действия — боевые приёмы и трюки против соперников, конкретные для этой ситуации."
            )
        elif mode == "heist":
            user_prompt = (
                f"Придумай смешной абсурдный сценарий ПЕРВОЙ фазы ограбления — проникновение — "
                f"для банды из {player_count} воров. "
                "Отряд только что подобрался к цели (примеры: банк гоблинов, сокровищница дракона-бухгалтера, "
                "дворец Короля Бюрократии, волшебный ломбард). Описание должно задавать место и первое препятствие. "
                "Действия — варианты скрытного/хитрого проникновения: отвлечь охрану, взломать магический замок, "
                "притвориться официантами, пролезть через вентиляцию и т.п. Только стелс и хитрость, никаких лобовых атак."
            )
        else:
            user_prompt = (
                f"Придумай смешной абсурдный фэнтезийный сценарий для группы из {player_count} игроков. "
                "Примеры тематик: пьяный гоблин просит денег, говорящий дракон хочет занять в долг, "
                "подозрительный сундук с лицом, таверна полная говорящих грибов, "
                "крылатый торговец впаривает зелья. Действия должны быть смешными и "
                "контекстно-специфичными именно для этого сценария."
            )
    else:
        history_text = __format_history_lines(history)
        is_final = round_number >= max_rounds
        if mode == "heist":
            heist_phases = {
                2: "само ограбление (добраться до цели и взять трофей)",
                3: "побег (уйти от погони и охраны)",
            }
            phase_desc = heist_phases.get(round_number, "следующая фаза ограбления")
            user_prompt = (
                f"История ограбления банды из {player_count} воров:\n\n{history_text}\n\n"
                f"Придумай следующую ситуацию, которая логично и смешно вытекает из предыдущих событий. "
                f"Это фаза «{phase_desc}» — опиши именно этот этап ограбления. "
                "Действия — только стелс, обман и хитрость (без прямых драк)."
            )
        else:
            final_note = (
                " Это ФИНАЛЬНЫЙ раунд — придумай кульминацию всей истории, эффектную развязку."
                if is_final else ""
            )
            user_prompt = (
                f"История приключения отряда из {player_count} игроков:\n\n{history_text}\n\n"
                f"Придумай следующую ситуацию, которая логично и смешно вытекает из предыдущих событий.{final_note} "
                "Действия должны быть контекстно-специфичными для новой ситуации."
            )

    response = await asyncio.wait_for(
        __llm(120).ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]),
        timeout=DND_LLM_TIMEOUT,
    )
    return __parse_scenario(response.content)


async def __generate_coop_round(
    player_count: int,
    round_number: int,
    boss_name: str,
    boss_hp: int,
    boss_max_hp: int,
    history: list[dict],
) -> tuple[str, str, list[str]]:
    """Returns (scenario, boss_name, actions). boss_name parsed from LLM only in round 1."""
    if round_number == 1:
        system_prompt = (
            "Ты генератор сценариев для D&D-кооп-игры в Telegram-чате геймеров.\n"
            "Твой ответ должен быть строго в формате (без лишних слов):\n"
            "СЦЕНАРИЙ: <одно короткое предложение — встреча с боссом>\n"
            "БОСС: <смешное имя босса (3-6 слов)>\n"
            "Д1: <понятное кооперативное действие 3-6 слов>\n"
            "Д2: <понятное кооперативное действие 3-6 слов>\n"
            "Д3: <понятное кооперативное действие 3-6 слов>\n\n"
            "Только русский язык. Никакого другого текста."
        )
        user_prompt = (
            f"Придумай смешной абсурдный сценарий для {player_count} игроков, "
            "которые вместе сражаются против одного огромного абсурдного босса-NPC. "
            "Действия должны быть кооперативными атаками/стратегиями против этого конкретного босса."
        )
        response = await asyncio.wait_for(
            __llm(120).ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]),
            timeout=DND_LLM_TIMEOUT,
        )
        scenario, parsed_boss_name, actions = __parse_coop_scenario(response.content)
        return scenario, parsed_boss_name, actions
    else:
        system_prompt = (
            "Ты генератор сценариев для D&D-кооп-игры в Telegram-чате геймеров.\n"
            "Твой ответ должен быть строго в формате (без лишних слов):\n"
            "СЦЕНАРИЙ: <одно короткое предложение — продолжение битвы>\n"
            "Д1: <понятное кооперативное действие 3-6 слов>\n"
            "Д2: <понятное кооперативное действие 3-6 слов>\n"
            "Д3: <понятное кооперативное действие 3-6 слов>\n\n"
            "Только русский язык. Никакого другого текста."
        )
        history_text = "\n\n".join(
            f"Раунд {idx + 1}: {entry['narrative']}"
            for idx, entry in enumerate(history)
        )
        user_prompt = (
            f"Финальный раунд битвы с боссом «{boss_name}».\n"
            f"У босса осталось {boss_hp} из {boss_max_hp} HP.\n\n"
            f"Что было:\n{history_text}\n\n"
            "Придумай финальную ситуацию этой битвы. "
            "Действия — финальные кооперативные атаки/решающие манёвры."
        )
        response = await asyncio.wait_for(
            __llm(120).ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]),
            timeout=DND_LLM_TIMEOUT,
        )
        scenario, actions = __parse_scenario(response.content)
        return scenario, boss_name, actions


def __parse_scenario(text: str) -> tuple[str, list[str]]:
    scenario = ""
    parsed: dict[int, str] = {}
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("СЦЕНАРИЙ:"):
            scenario = stripped[len("СЦЕНАРИЙ:"):].strip()
        else:
            match = __ACTION_RE.match(stripped)
            if match:
                parsed[int(match.group(1))] = match.group(2).strip()[:50]

    if not scenario:
        scenario = "Отряд оказался в таверне, где все посетители — говорящие грибы с мнением."
    fallback_actions = ["Бежать со всех ног", "Атаковать в лоб", "Попробовать договориться"]
    actions = [parsed.get(idx) or fallback_actions[idx - 1] for idx in range(1, 4)]
    return scenario, actions


def __parse_coop_scenario(text: str) -> tuple[str, str, list[str]]:
    scenario = ""
    boss_name = ""
    parsed: dict[int, str] = {}
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("СЦЕНАРИЙ:"):
            scenario = stripped[len("СЦЕНАРИЙ:"):].strip()
        elif stripped.startswith("БОСС:"):
            boss_name = stripped[len("БОСС:"):].strip()
        else:
            match = __ACTION_RE.match(stripped)
            if match:
                parsed[int(match.group(1))] = match.group(2).strip()[:50]

    if not scenario:
        scenario = "Перед отрядом возник монструозный противник, исполненный бюрократической мощи."
    if not boss_name:
        boss_name = "Великий Неизвестный Босс"
    fallback_actions = ["Атаковать все вместе", "Найти слабое место", "Отвлечь и ударить сзади"]
    actions = [parsed.get(idx) or fallback_actions[idx - 1] for idx in range(1, 4)]
    return scenario, boss_name, actions


# ---------------------------------------------------------------------------
# LLM: narrative generation
# ---------------------------------------------------------------------------

async def __generate_narrative(
    scenario: str,
    player_results: list[dict],
    history: list[dict],
    is_final: bool,
    is_pvp: bool = False,
    is_heist: bool = False,
) -> str:
    player_lines = []
    for result in player_results:
        roll = result["roll"]
        roll_note = " (КРИТИЧЕСКИЙ ПРОВАЛ)" if roll == 1 else " (КРИТИЧЕСКИЙ УСПЕХ)" if roll == 20 else ""
        player_lines.append(f'• @{result["username"]} выбрал "{result["action"]}" → 🎲{roll}{roll_note}')

    context_block = ""
    if history:
        history_lines = "\n\n".join(
            f"Раунд {idx + 1}: {entry['narrative']}" for idx, entry in enumerate(history)
        )
        context_block = f"Что было раньше:\n{history_lines}\n\n"

    if is_pvp:
        ending_instruction = (
            "Это прямая драка игроков друг с другом. "
            "Опиши конкретные столкновения между ними — кто кого ударил, подставил, обхитрил. "
            "Победитель (самый высокий бросок) должен быть назван явно и смешно прославлен. "
            "Проигравшие (низкие броски) — смешно унижены конкретными соперниками, не абстрактно."
        )
    elif is_heist and is_final:
        ending_instruction = (
            "Это ФИНАЛЬНАЯ фаза ограбления — побег. Заверши историю эффектно: "
            "удалось ли уйти с добычей? Назови победителей или опозорившихся по броскам. "
            "Финал должен быть смешным и окончательным."
        )
    elif is_heist:
        ending_instruction = (
            "Это фаза ограбления — оцени успех каждого по броскам и действиям. "
            "Намекни одной фразой, что следующая фаза ещё впереди."
        )
    elif is_final:
        ending_instruction = (
            "Это ФИНАЛЬНЫЙ раунд — заверши всю историю эффектно, смешно и окончательно. "
            "Дай каждому герою достойный финал."
        )
    else:
        ending_instruction = "Заверши этот раунд и намекни одной фразой, что приключение продолжается."

    system_prompt = (
        "Ты нарратор D&D для группы друзей-геймеров. "
        "Пишешь короткие смешные нарративы на русском языке. "
        "Разговорный стиль, юмор, абсурд. Можно крепкие выражения. "
        "Ответ — только нарратив, без заголовков."
    )
    user_prompt = (
        f"{context_block}"
        f"Текущая ситуация: {scenario}\n\n"
        "Игроки и их действия:\n"
        + "\n".join(player_lines)
        + f"\n\nНапиши смешной нарратив (1-2 предложения). "
        "Обязательно упомяни каждого игрока. "
        "Бросок 1 = катастрофически смешной провал. Бросок 20 = невероятный триумф. "
        f"{ending_instruction}"
    )
    response = await asyncio.wait_for(
        __llm(120).ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]),
        timeout=DND_LLM_TIMEOUT,
    )
    return response.content.strip()


async def __generate_coop_narrative(
    scenario: str,
    player_results: list[dict],
    boss_name: str,
    damage_this_round: int,
    boss_hp_after: int,
    boss_max_hp: int,
    history: list[dict],
    players_won: bool,
    is_final: bool,
) -> str:
    player_lines = []
    for result in player_results:
        roll = result["roll"]
        roll_note = " (КРИТИЧЕСКИЙ ПРОВАЛ)" if roll == 1 else " (КРИТИЧЕСКИЙ УСПЕХ)" if roll == 20 else ""
        player_lines.append(f'• @{result["username"]} выбрал "{result["action"]}" → 🎲{roll}{roll_note}')

    context_block = ""
    if history:
        history_lines = "\n\n".join(
            f"Раунд {idx + 1}: {entry['narrative']}" for idx, entry in enumerate(history)
        )
        context_block = f"Что было раньше:\n{history_lines}\n\n"

    if is_final:
        if players_won:
            outcome_instruction = (
                f"Отряд нанёс финальный удар! «{boss_name}» повержен! "
                "Опиши смешную эпическую победу отряда и позорное поражение босса."
            )
        else:
            outcome_instruction = (
                f"Отряд не успел добить «{boss_name}» — у босса осталось {boss_hp_after} HP. "
                "Опиши смешное горькое поражение отряда и торжество босса."
            )
    else:
        outcome_instruction = (
            f"Отряд нанёс {damage_this_round} урона «{boss_name}». "
            f"У босса осталось {boss_hp_after} из {boss_max_hp} HP. "
            "Опиши атаку отряда — босс ранен, но ещё стоит. Намекни, что битва продолжится."
        )

    system_prompt = (
        "Ты нарратор D&D-кооп для группы друзей-геймеров. "
        "Пишешь короткие смешные нарративы о битве с боссом на русском языке. "
        "Разговорный стиль, юмор, абсурд. Можно крепкие выражения. "
        "Ответ — только нарратив, без заголовков."
    )
    user_prompt = (
        f"{context_block}"
        f"Ситуация: {scenario}\n\n"
        "Игроки и их действия:\n"
        + "\n".join(player_lines)
        + f"\n\n{outcome_instruction}\n"
        "Обязательно упомяни каждого игрока. "
        "Бросок 1 = катастрофически смешной промах. Бросок 20 = невероятно мощный удар. "
        "Нарратив — 1-2 предложения."
    )
    response = await asyncio.wait_for(
        __llm(120).ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]),
        timeout=DND_LLM_TIMEOUT,
    )
    return response.content.strip()


# ---------------------------------------------------------------------------
# Lobby
# ---------------------------------------------------------------------------

async def __start_lobby(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    max_rounds: int,
    mode: str,
) -> None:
    chat_id = update.effective_chat.id

    if chat_id in __active_chats:
        await update.message.reply_text("В чате уже идёт D&D-приключение. Дождитесь его завершения.")
        return

    initiator = update.effective_user
    initiator_id = initiator.id
    initiator_username = initiator.username or initiator.first_name or f"user_{initiator_id}"

    # Fill the roster with a bot NPC when there aren't enough real players in this chat.
    members = await achievements.get_chat_members(chat_id)
    initial_players: list[tuple[int, str]] = [(initiator_id, initiator_username)]
    if len(members) < DND_MIN_PLAYERS:
        initial_players.append((DND_BOT_PLAYER_ID, DND_BOT_PLAYER_NAME))

    __active_chats.add(chat_id)
    lobby = LobbyState(
        chat_id=chat_id,
        message_id=0,
        initiator_id=initiator_id,
        players=initial_players,
    )

    msg = await update.message.reply_text(
        __build_lobby_text(lobby, max_rounds, mode),
        reply_markup=__build_lobby_keyboard(),
    )
    lobby.message_id = msg.message_id
    __lobbies[chat_id] = (lobby, max_rounds, mode)

    job = context.job_queue.run_once(__expire_lobby, DND_LOBBY_TIMEOUT, data=chat_id)
    __lobby_timeout_jobs[chat_id] = job


async def cmd_dnd_pvp(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await __start_lobby(update, context, max_rounds=1, mode="pvp")


async def cmd_dnd_coop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await __start_lobby(update, context, max_rounds=2, mode="coop")


async def cmd_dnd_heist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await __start_lobby(update, context, max_rounds=3, mode="heist")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

async def handle_dnd_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    data = query.data

    if data == DND_JOIN_CALLBACK:
        await __handle_join(query, context)
    elif data.startswith(DND_ACTION_CALLBACK_PREFIX):
        try:
            action_index = int(data[len(DND_ACTION_CALLBACK_PREFIX):])
        except ValueError:
            await query.answer()
            return
        await __handle_action(query, context, action_index)
    else:
        await query.answer()


async def __handle_join(query, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = query.message.chat_id
    clicker_id = query.from_user.id
    clicker_username = (
        query.from_user.username
        or query.from_user.first_name
        or f"user_{clicker_id}"
    )

    entry = __lobbies.get(chat_id)
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
        await __edit_safe(
            context.bot,
            query.message.chat_id,
            query.message.message_id,
            __build_lobby_text(lobby, max_rounds, mode),
            __build_lobby_keyboard(),
        )
        return

    # Minimum reached — atomically claim the lobby before any await to prevent double-start.
    popped = __lobbies.pop(chat_id, None)
    if popped is None:
        await query.answer()
        return

    job = __lobby_timeout_jobs.pop(chat_id, None)
    if job:
        job.schedule_removal()

    players = list(lobby.players)
    message_id = query.message.message_id

    await query.answer("Отряд собран! Начинаем приключение!")

    try:
        await query.edit_message_text(
            "⚔️ D&D Приключение\n\n🎲 Генерация сценария...",
        )
    except TelegramError:
        pass

    context.job_queue.run_once(
        __start_game_job,
        0,
        data=(chat_id, message_id, players, max_rounds, mode),
    )


async def __handle_action(query, context: ContextTypes.DEFAULT_TYPE, action_index: int) -> None:
    chat_id = query.message.chat_id
    clicker_id = query.from_user.id

    game = __active_games.get(chat_id)
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

    await __edit_safe(
        context.bot,
        chat_id,
        game.message_id,
        __build_game_text(game),
        keyboard=__build_game_keyboard(game),
    )

    if all_chosen:
        resolved_game = __active_games.pop(chat_id, None)
        if not resolved_game:
            return
        job = __action_timeout_jobs.pop(chat_id, None)
        if job:
            job.schedule_removal()
        context.job_queue.run_once(__resolve_game_job, 0, data=(chat_id, resolved_game))


# ---------------------------------------------------------------------------
# Jobs: start, resolve, next round, expire
# ---------------------------------------------------------------------------

async def __start_game_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id, message_id, players, max_rounds, mode = context.job.data

    boss_name = ""
    boss_max_hp = 0

    try:
        if mode == "coop":
            scenario, boss_name, actions = await __generate_coop_round(
                len(players), round_number=1,
                boss_name="", boss_hp=0, boss_max_hp=0, history=[],
            )
            boss_max_hp = random.randint(len(players) * 15, len(players) * 20)
        else:
            scenario, actions = await __generate_round(
                len(players), round_number=1, max_rounds=max_rounds, mode=mode, history=[], players=players,
            )
    except Exception as error:
        logger.warning(f"DnD scenario generation failed for chat {chat_id}: {error}")
        __active_chats.discard(chat_id)
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
    __assign_bot_choice(game)
    __active_games[chat_id] = game

    await __edit_safe(
        context.bot, chat_id, message_id,
        __build_game_text(game),
        keyboard=__build_game_keyboard(game),
    )

    job = context.job_queue.run_once(__expire_actions, DND_ACTION_TIMEOUT, data=chat_id)
    __action_timeout_jobs[chat_id] = job


async def __expire_actions(context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = context.job.data
    game = __active_games.pop(chat_id, None)
    __action_timeout_jobs.pop(chat_id, None)
    if not game:
        return
    context.job_queue.run_once(__resolve_game_job, 0, data=(chat_id, game))


async def __resolve_game_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id, game = context.job.data

    player_results = []
    for player_id, username in game.players:
        roll = random.randint(1, 20)
        action_index = game.choices.get(player_id)
        action = game.actions[action_index] if action_index is not None else "бездействовал"
        player_results.append({"username": username, "roll": roll, "action": action})

    is_final = game.round_number >= game.max_rounds

    if game.mode == "coop":
        await __resolve_coop_round(context, chat_id, game, player_results, is_final)
    else:
        await __resolve_standard_round(context, chat_id, game, player_results, is_final)


async def __resolve_coop_round(
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
        narrative = await __generate_coop_narrative(
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
        logger.warning(f"DnD coop narrative failed for chat {chat_id}: {error}")
        narrative = "Летописец выронил перо в разгар битвы. Но отряд устоял. Кажется."

    roll_lines = __format_roll_lines(player_results)

    if is_final:
        if players_won:
            round_title = "Победа! 🏆"
            boss_line = f"👹 {game.boss_name} повержен!"
        else:
            round_title = "Поражение 💀"
            boss_line = f"👹 {game.boss_name} выжил... ({game.boss_hp} HP осталось)"
        damage_line = f"💥 Финальный урон: {total_damage} — итого {game.boss_max_hp - game.boss_hp}/{game.boss_max_hp}"
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

    await __edit_safe(context.bot, chat_id, game.message_id, result_text)

    if is_final:
        __active_chats.discard(chat_id)
        return

    __advance_round(game, narrative, player_results)
    context.job_queue.run_once(__next_round_job, 0, data=(chat_id, game))


async def __resolve_standard_round(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    game: ActiveGame,
    player_results: list[dict],
    is_final: bool,
) -> None:
    if game.mode == "heist":
        loading_header = f"🎩 Великое Ограбление — {__heist_phase_name(game.round_number)}"
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
        narrative = await __generate_narrative(
            game.scenario, player_results, game.history, is_final,
            is_pvp=(game.mode == "pvp"),
            is_heist=(game.mode == "heist"),
        )
    except Exception as error:
        logger.warning(f"DnD narrative failed for chat {chat_id}: {error}")
        narrative = "Летописец выронил перо и всё размазалось. Но все выжили. Кажется."

    roll_lines = __format_roll_lines(player_results)

    if game.mode == "heist":
        heist_result_titles = {1: "Проникновение — Итог", 2: "Дело — Итог", 3: "Побег — Финал"}
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

    await __edit_safe(context.bot, chat_id, game.message_id, result_text)

    if is_final:
        __active_chats.discard(chat_id)
        return

    __advance_round(game, narrative, player_results)
    context.job_queue.run_once(__next_round_job, 0, data=(chat_id, game))


async def __next_round_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id, game = context.job.data

    if game.mode == "coop":
        loading_text = f"⚔️ D&D Кооп — Раунд {game.round_number}/{game.max_rounds}\n\n🎲 Генерация продолжения..."
    elif game.mode == "heist":
        loading_text = f"🎩 Великое Ограбление — {__heist_phase_name(game.round_number)}\n\n🎲 Генерация следующей фазы..."
    else:
        loading_text = f"⚔️ D&D — Раунд {game.round_number}/{game.max_rounds}\n\n🎲 Генерация продолжения..."

    try:
        msg = await context.bot.send_message(
            chat_id=chat_id,
            text=loading_text,
        )
        game.message_id = msg.message_id
    except TelegramError as error:
        logger.warning(f"DnD failed to send round {game.round_number} message for chat {chat_id}: {error}")
        __active_chats.discard(chat_id)
        return

    try:
        if game.mode == "coop":
            scenario, _, actions = await __generate_coop_round(
                len(game.players), game.round_number,
                game.boss_name, game.boss_hp, game.boss_max_hp, game.history,
            )
        else:
            scenario, actions = await __generate_round(
                len(game.players), game.round_number, game.max_rounds, game.mode, game.history, game.players,
            )
    except Exception as error:
        logger.warning(f"DnD round {game.round_number} generation failed for chat {chat_id}: {error}")
        __active_chats.discard(chat_id)
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
    __assign_bot_choice(game)
    __active_games[chat_id] = game

    await __edit_safe(
        context.bot, chat_id, game.message_id,
        __build_game_text(game),
        keyboard=__build_game_keyboard(game),
    )

    job = context.job_queue.run_once(__expire_actions, DND_ACTION_TIMEOUT, data=chat_id)
    __action_timeout_jobs[chat_id] = job


async def __expire_lobby(context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = context.job.data
    entry = __lobbies.pop(chat_id, None)
    __lobby_timeout_jobs.pop(chat_id, None)
    if not entry:
        return
    lobby, _, _ = entry

    __active_chats.discard(chat_id)
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=lobby.message_id,
            text="⚔️ Лобби закрыто — никто не собрал отряд. В следующий раз.",
            reply_markup=None,
        )
    except TelegramError as error:
        logger.warning(f"DnD lobby expiry failed for chat {chat_id}: {error}")
