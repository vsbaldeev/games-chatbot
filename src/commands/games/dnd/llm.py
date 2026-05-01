"""
LLM scenario and narrative generation for D&D games.

ScenarioGenerator wraps all Groq ChatGroq calls behind clean async methods.
Parsing helpers are private; callers only see well-typed return values.
"""

import asyncio
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src import config
from src.commands.games.dnd.state import DND_MODEL, DND_LLM_TIMEOUT

ACTION_RE = re.compile(r"^Д([1-3]):\s*(.+)$")


class ScenarioGenerator:
    """Generates D&D scenarios and narratives via a Groq LLM."""

    def __create_llm(self, max_tokens: int = 300) -> ChatGroq:
        return ChatGroq(
            model=DND_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.95,
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_round(
        self,
        player_count: int,
        round_number: int,
        max_rounds: int,
        mode: str,
        history: list[dict],
        players: list[tuple[int, str]] | None = None,
    ) -> tuple[str, list[str]]:
        """Return (scenario, actions) for one standard round (pvp/heist/default)."""
        system_prompt = self.__round_system_prompt(mode)
        user_prompt = self.__round_user_prompt(
            player_count, round_number, max_rounds, mode, history, players
        )
        response = await asyncio.wait_for(
            self.__create_llm(300).ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            ),
            timeout=DND_LLM_TIMEOUT,
        )
        return self.__parse_scenario(response.content)

    async def generate_coop_round(
        self,
        player_count: int,
        round_number: int,
        boss_name: str,
        boss_hp: int,
        boss_max_hp: int,
        history: list[dict],
    ) -> tuple[str, str, list[str]]:
        """Return (scenario, boss_name, actions). boss_name is parsed from LLM in round 1 only."""
        if round_number == 1:
            return await self.__generate_coop_round_first(player_count)
        return await self.__generate_coop_round_continuation(
            player_count, boss_name, boss_hp, boss_max_hp, history
        )

    async def generate_narrative(
        self,
        scenario: str,
        player_results: list[dict],
        history: list[dict],
        is_final: bool,
        is_pvp: bool = False,
        is_heist: bool = False,
    ) -> str:
        """Return a short narrative string for a completed standard round."""
        player_lines = self.__format_player_lines(player_results)
        context_block = self.__format_context_block(history)
        ending_instruction = self.__standard_ending_instruction(is_pvp, is_heist, is_final)

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
            + f"\n\nНапиши смешной нарратив — СТРОГО 2 коротких предложения, не длиннее. "
            "Обязательно упомяни каждого игрока. "
            "Бросок 1 = катастрофически смешной провал. Бросок 20 = невероятный триумф. "
            f"{ending_instruction}"
        )
        response = await asyncio.wait_for(
            self.__create_llm(600).ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            ),
            timeout=DND_LLM_TIMEOUT,
        )
        return response.content.strip()

    async def generate_coop_narrative(
        self,
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
        """Return a short narrative string for a completed coop round."""
        system_prompt, user_prompt = self.__build_coop_narrative_prompt(
            scenario, player_results, boss_name,
            damage_this_round, boss_hp_after, boss_max_hp,
            history, players_won, is_final,
        )
        response = await asyncio.wait_for(
            self.__create_llm(600).ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            ),
            timeout=DND_LLM_TIMEOUT,
        )
        return response.content.strip()

    # ------------------------------------------------------------------
    # Private: prompt builders
    # ------------------------------------------------------------------

    def __build_coop_narrative_prompt(
        self,
        scenario: str,
        player_results: list[dict],
        boss_name: str,
        damage_this_round: int,
        boss_hp_after: int,
        boss_max_hp: int,
        history: list[dict],
        players_won: bool,
        is_final: bool,
    ) -> tuple[str, str]:
        player_lines = self.__format_player_lines(player_results)
        context_block = self.__format_context_block(history)
        outcome_instruction = self.__coop_outcome_instruction(
            boss_name, damage_this_round, boss_hp_after, boss_max_hp, players_won, is_final
        )
        system_prompt = (
            "Ты нарратор D&D-кооп для группы друзей-геймеров. "
            "Пишешь короткие смешные нарративы о битве с боссом на русском языке. "
            "Разговорный стиль, юмор, абсурд. Можно крепкие выражения. "
            "Ответ — только нарратив, без заголовков."
        )
        user_prompt = (
            f"{context_block}Ситуация: {scenario}\n\nИгроки и их действия:\n"
            + "\n".join(player_lines)
            + f"\n\n{outcome_instruction}\n"
            "Обязательно упомяни каждого игрока. "
            "Бросок 1 = катастрофически смешной промах. Бросок 20 = невероятно мощный удар. "
            "Нарратив — СТРОГО 2 коротких предложения, не длиннее."
        )
        return system_prompt, user_prompt

    def __round_system_prompt(self, mode: str) -> str:
        base = (
            "Ты генератор сценариев для D&D {kind} в Telegram-чате геймеров.\n"
            "Твой ответ должен быть строго в формате (без лишних слов):\n"
            "СЦЕНАРИЙ: <{scenario_hint}>\n"
            "Д1: <{action_hint}>\n"
            "Д2: <{action_hint}>\n"
            "Д3: <{action_hint}>\n\n"
            "Только русский язык. Никакого другого текста."
        )
        if mode == "pvp":
            return base.format(
                kind="PvP-игры",
                scenario_hint="2-3 предложения — смешная абсурдная завязка драки, кто с кем и за что",
                action_hint="боевой приём или трюк против соперников, 3-6 слов",
            )
        if mode == "heist":
            return base.format(
                kind="D&D-ограбления",
                scenario_hint="2-3 предложения — место ограбления, что охраняется, первое препятствие",
                action_hint="хитрый/стелс вариант действия, 3-6 слов",
            )
        return base.format(
            kind="игры",
            scenario_hint="одно короткое предложение — смешной абсурдный сценарий",
            action_hint="понятное действие 3-6 слов",
        )

    def __round_user_prompt(
        self,
        player_count: int,
        round_number: int,
        max_rounds: int,
        mode: str,
        history: list[dict],
        players: list[tuple[int, str]] | None,
    ) -> str:
        if not history:
            return self.__first_round_user_prompt(player_count, mode, players)
        return self.__continuation_user_prompt(
            player_count, round_number, max_rounds, mode, history
        )

    def __first_round_user_prompt(
        self,
        player_count: int,
        mode: str,
        players: list[tuple[int, str]] | None,
    ) -> str:
        if mode == "pvp":
            names = ", ".join(f"@{name}" for _, name in (players or [])) or f"{player_count} игроков"
            return (
                f"Придумай смешной абсурдный фэнтезийный сценарий прямой драки между {names}. "
                "Они дерутся друг с другом — один на один, все против всех, без NPC-противников. "
                "Примеры: делят последний кусок пирога у дракона, выясняют кто лучший маг в таверне, "
                "спор перерос в магическую дуэль, гонка за артефактом где мешать друг другу — главное. "
                "Действия — боевые приёмы и трюки против соперников, конкретные для этой ситуации."
            )
        if mode == "heist":
            return (
                f"Придумай смешной абсурдный сценарий ПЕРВОЙ фазы ограбления — проникновение — "
                f"для банды из {player_count} воров. "
                "Отряд только что подобрался к цели (примеры: банк гоблинов, сокровищница дракона-бухгалтера, "
                "дворец Короля Бюрократии, волшебный ломбард). Описание должно задавать место и первое препятствие. "
                "Действия — варианты скрытного/хитрого проникновения: отвлечь охрану, взломать магический замок, "
                "притвориться официантами, пролезть через вентиляцию и т.п. Только стелс и хитрость, никаких лобовых атак."
            )
        return (
            f"Придумай смешной абсурдный фэнтезийный сценарий для группы из {player_count} игроков. "
            "Примеры тематик: пьяный гоблин просит денег, говорящий дракон хочет занять в долг, "
            "подозрительный сундук с лицом, таверна полная говорящих грибов, "
            "крылатый торговец впаривает зелья. Действия должны быть смешными и "
            "контекстно-специфичными именно для этого сценария."
        )

    def __continuation_user_prompt(
        self,
        player_count: int,
        round_number: int,
        max_rounds: int,
        mode: str,
        history: list[dict],
    ) -> str:
        history_text = self.__format_history_lines(history)
        if mode == "heist":
            heist_phases = {
                2: "само ограбление (добраться до цели и взять трофей)",
                3: "побег (уйти от погони и охраны)",
            }
            phase_desc = heist_phases.get(round_number, "следующая фаза ограбления")
            return (
                f"История ограбления банды из {player_count} воров:\n\n{history_text}\n\n"
                f"Придумай следующую ситуацию, которая логично и смешно вытекает из предыдущих событий. "
                f"Это фаза «{phase_desc}» — опиши именно этот этап ограбления. "
                "Действия — только стелс, обман и хитрость (без прямых драк)."
            )
        is_final = round_number >= max_rounds
        final_note = (
            " Это ФИНАЛЬНЫЙ раунд — придумай кульминацию всей истории, эффектную развязку."
            if is_final else ""
        )
        return (
            f"История приключения отряда из {player_count} игроков:\n\n{history_text}\n\n"
            f"Придумай следующую ситуацию, которая логично и смешно вытекает из предыдущих событий.{final_note} "
            "Действия должны быть контекстно-специфичными для новой ситуации."
        )

    async def __generate_coop_round_first(self, player_count: int) -> tuple[str, str, list[str]]:
        system_prompt = (
            "Ты генератор сценариев для D&D-кооп-игры в Telegram-чате геймеров.\n"
            "Твой ответ должен быть строго в формате (без лишних слов):\n"
            "СЦЕНАРИЙ: <2-3 предложения — смешная встреча с боссом: кто он, что делает, чем опасен>\n"
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
            self.__create_llm(300).ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            ),
            timeout=DND_LLM_TIMEOUT,
        )
        return self.__parse_coop_scenario(response.content)

    async def __generate_coop_round_continuation(
        self,
        player_count: int,
        boss_name: str,
        boss_hp: int,
        boss_max_hp: int,
        history: list[dict],
    ) -> tuple[str, str, list[str]]:
        system_prompt = (
            "Ты генератор сценариев для D&D-кооп-игры в Telegram-чате геймеров.\n"
            "Твой ответ должен быть строго в формате (без лишних слов):\n"
            "СЦЕНАРИЙ: <2-3 предложения — продолжение битвы: что изменилось, как реагирует босс>\n"
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
            self.__create_llm(300).ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            ),
            timeout=DND_LLM_TIMEOUT,
        )
        scenario, actions = self.__parse_scenario(response.content)
        return scenario, boss_name, actions

    # ------------------------------------------------------------------
    # Private: parsing helpers
    # ------------------------------------------------------------------

    def __parse_scenario(self, text: str) -> tuple[str, list[str]]:
        scenario = ""
        parsed: dict[int, str] = {}
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("СЦЕНАРИЙ:"):
                scenario = stripped[len("СЦЕНАРИЙ:"):].strip()
            else:
                match = ACTION_RE.match(stripped)
                if match:
                    parsed[int(match.group(1))] = match.group(2).strip()[:50]

        if not scenario:
            scenario = "Отряд оказался в таверне, где все посетители — говорящие грибы с мнением."
        fallback_actions = ["Бежать со всех ног", "Атаковать в лоб", "Попробовать договориться"]
        actions = [parsed.get(idx) or fallback_actions[idx - 1] for idx in range(1, 4)]
        return scenario, actions

    def __parse_coop_scenario(self, text: str) -> tuple[str, str, list[str]]:
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
                match = ACTION_RE.match(stripped)
                if match:
                    parsed[int(match.group(1))] = match.group(2).strip()[:50]

        if not scenario:
            scenario = "Перед отрядом возник монструозный противник, исполненный бюрократической мощи."
        if not boss_name:
            boss_name = "Великий Неизвестный Босс"
        fallback_actions = ["Атаковать все вместе", "Найти слабое место", "Отвлечь и ударить сзади"]
        actions = [parsed.get(idx) or fallback_actions[idx - 1] for idx in range(1, 4)]
        return scenario, boss_name, actions

    # ------------------------------------------------------------------
    # Private: text formatting helpers
    # ------------------------------------------------------------------

    def __format_player_lines(self, player_results: list[dict]) -> list[str]:
        lines = []
        for result in player_results:
            roll = result["roll"]
            roll_note = (
                " (КРИТИЧЕСКИЙ ПРОВАЛ)" if roll == 1
                else " (КРИТИЧЕСКИЙ УСПЕХ)" if roll == 20
                else ""
            )
            lines.append(f'• @{result["username"]} выбрал "{result["action"]}" → 🎲{roll}{roll_note}')
        return lines

    def __format_context_block(self, history: list[dict]) -> str:
        if not history:
            return ""
        history_lines = "\n\n".join(
            f"Раунд {idx + 1}: {entry['narrative']}" for idx, entry in enumerate(history)
        )
        return f"Что было раньше:\n{history_lines}\n\n"

    def __format_history_lines(self, history: list[dict]) -> str:
        return "\n\n".join(
            f"Раунд {idx + 1}:\nСитуация: {entry['scenario']}\nИтог: {entry['narrative']}"
            for idx, entry in enumerate(history)
        )

    def __standard_ending_instruction(self, is_pvp: bool, is_heist: bool, is_final: bool) -> str:
        if is_pvp:
            return (
                "Это прямая драка игроков друг с другом. "
                "Опиши конкретные столкновения между ними — кто кого ударил, подставил, обхитрил. "
                "Победитель (самый высокий бросок) должен быть назван явно и смешно прославлен. "
                "Проигравшие (низкие броски) — смешно унижены конкретными соперниками, не абстрактно."
            )
        if is_heist and is_final:
            return (
                "Это ФИНАЛЬНАЯ фаза ограбления — побег. Заверши историю эффектно: "
                "удалось ли уйти с добычей? Назови победителей или опозорившихся по броскам. "
                "Финал должен быть смешным и окончательным."
            )
        if is_heist:
            return (
                "Это фаза ограбления — оцени успех каждого по броскам и действиям. "
                "Намекни одной фразой, что следующая фаза ещё впереди."
            )
        if is_final:
            return (
                "Это ФИНАЛЬНЫЙ раунд — заверши всю историю эффектно, смешно и окончательно. "
                "Дай каждому герою достойный финал."
            )
        return "Заверши этот раунд и намекни одной фразой, что приключение продолжается."

    def __coop_outcome_instruction(
        self,
        boss_name: str,
        damage_this_round: int,
        boss_hp_after: int,
        boss_max_hp: int,
        players_won: bool,
        is_final: bool,
    ) -> str:
        if is_final:
            if players_won:
                return (
                    f"Отряд нанёс финальный удар! «{boss_name}» повержен! "
                    "Опиши смешную эпическую победу отряда и позорное поражение босса."
                )
            return (
                f"Отряд не успел добить «{boss_name}» — у босса осталось {boss_hp_after} HP. "
                "Опиши смешное горькое поражение отряда и торжество босса."
            )
        return (
            f"Отряд нанёс {damage_this_round} урона «{boss_name}». "
            f"У босса осталось {boss_hp_after} из {boss_max_hp} HP. "
            "Опиши атаку отряда — босс ранен, но ещё стоит. Намекни, что битва продолжится."
        )
