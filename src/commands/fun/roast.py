"""
"Прожарка" (roast) feature.

Roaster encapsulates LLM-based roast generation and the /roast command handler.
Module-level wrappers preserve the public API that bot.py and handlers.py import.
"""

import random
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from telegram import Update
from telegram.ext import ContextTypes

from src import achievements, config, log
from src.achievements import notify_unlocks
from src.agent import apply_language_correction
from src.store import unified_messages

logger = log.get_logger(__name__)

ROAST_MODEL = "llama-3.3-70b-versatile"

MECHANISMS = ("contradiction", "hyperbole", "forecast", "compliment", "bathos", "understatement", "definition", "observation")
FORMATS = ("protocol", "anecdote", "news", "ad", "instruction")

MECHANISM_EMOJIS = {
    "contradiction": "🔄",
    "hyperbole": "🚀",
    "forecast": "🔮",
    "compliment": "🏆",
    "bathos": "🎈",
    "understatement": "🤷",
    "definition": "📖",
    "observation": "👁",
}

FORMAT_EMOJIS = {
    "protocol": "⚖️",
    "anecdote": "🎤",
    "news": "📰",
    "ad": "📌",
    "instruction": "📝",
}

KNOCKOUT_EXAMPLES = (
    "Примеры:\n"
    "— @vasya три дня выбирал игру и взял Fortnite. Бесплатную.\n"
    "— @petya сказал что игра — говно, брать не советует. Наиграл 400 часов.\n"
    "— @masha написала что уходит из чата навсегда. Написала это пять раз.\n"
    "— @vasya объяснял всем как правильно питаться. Спросил где заказать пиццу в 2 ночи.\n"
    "— @petya написал «иду спать» и вернулся через две минуты.\n"
    "— @masha весь день жаловалась что нет времени. Посмотрела три сериала за ночь.\n"
    "— @vasya сказал что ему всё равно. Уточнял ещё четыре раза.\n\n"
)

FORMAT_SYSTEM_PROMPTS = {
    "protocol": (
        "Ты говоришь сухо — как судебный протокол. "
        "Структура всегда одна: первое предложение — факт или позиция человека, "
        "второе — короткое разоблачение или переворот. "
        "Второе предложение должно быть максимально коротким — чем короче, тем сильнее удар. "
        "Не объясняй шутку, не добавляй ничего после. "
        "Мат разрешён. Только русский язык."
    ),
    "anecdote": (
        "Ты пишешь в стиле КВН. "
        "Структура: нарративный разгон → «И тут выясняется:» → короткий пуант. "
        "Разгон создаёт ожидание, пуант его обрушивает одним коротким предложением. "
        "Не объясняй шутку. Мат разрешён. Только русский язык."
    ),
    "news": (
        "Ты пишешь новостной заголовок в стиле Лентача или ТАСС. "
        "Одно-два предложения. Официальный сухой тон — как будто это реальная новость. "
        "Не объясняй шутку. Мат разрешён. Только русский язык."
    ),
    "ad": (
        "Ты пишешь объявление в стиле Авито. "
        "Формат: «Отдам / Куплю / Ищу / Продам: [предмет]. [короткая причина или деталь].» "
        "Не объясняй шутку. Мат разрешён. Только русский язык."
    ),
    "instruction": (
        "Ты пишешь краткую инструкцию. "
        "Формат: «Как [действие] по @username: шаг → шаг → шаг.» "
        "Максимум три шага. Шаги короткие. "
        "Не объясняй шутку. Мат разрешён. Только русский язык."
    ),
}

MECHANISM_INSTRUCTIONS = {
    "contradiction": "Основывай шутку на противоречии между тем что человек говорит и тем что делает.",
    "hyperbole": "Преувеличь какую-то черту или привычку человека до абсурда.",
    "forecast": "Экстраполируй поведение человека в будущее — логично, но мрачно.",
    "compliment": "Похвали человека так, чтобы похвала на самом деле уничижала.",
    "bathos": "Начни с чего-то грандиозного или важного, закончи чем-то совершенно ничтожным.",
    "understatement": "Преподнеси что-то очевидно плохое как совершенно нормальное и привычное.",
    "definition": "Дай определение человеку — как в словаре или медицинском справочнике.",
    "observation": "Опиши повторяющийся паттерн поведения человека как нейтральный факт. Никакого пуанта — сам факт и есть шутка.",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F9FF\U00002600-\U000027FF\U0001FA00-\U0001FAFF]",
    re.UNICODE,
)


class Roaster:
    """Generates LLM-powered roasts and handles the /roast Telegram command."""

    def __build_supportive_prompts(
        self, target_username: str, history_text: str
    ) -> tuple[str, str]:
        if history_text:
            system = (
                "Ты дружелюбный бот в группе друзей. "
                "Пишешь тёплые, искренние сообщения — строго одно предложение. "
                "Только русский язык."
            )
            user = (
                f"Последние сообщения @{target_username} в чате:\n{history_text}\n\n"
                f"Напиши искреннее тёплое поддерживающее сообщение для @{target_username} — "
                f"как лучший друг, который реально верит в него. Без сарказма, с душой. "
                f"Строго одно предложение."
            )
        else:
            system = (
                "Ты дружелюбный бот в группе друзей. "
                "Пишешь тёплые, искренние сообщения — строго одно предложение. "
                "Только русский язык."
            )
            user = (
                f"@{target_username} ещё не написал в чате ни слова. "
                f"Напиши ему тёплое, дружелюбное сообщение от лица чата — "
                f"позови поучаствовать в общении, скажи что рады его видеть. "
                f"Обязательно упомяни @{target_username}. Строго одно предложение."
            )
        return system, user

    def __build_user_prompt(
        self,
        target_username: str,
        context_text: str,
        mechanism: str,
        include_examples: bool,
    ) -> str:
        instruction = MECHANISM_INSTRUCTIONS[mechanism]
        examples = KNOCKOUT_EXAMPLES if include_examples else ""
        if context_text:
            return (
                f"{examples}"
                f"Сообщения @{target_username}:\n{context_text}\n\n"
                f"Напиши роаст на @{target_username}. {instruction}"
            )
        return (
            f"@{target_username} не написал ни слова в чате. "
            f"Напиши роаст про его молчание. {instruction}"
        )

    async def __extract_roastable_moment(
        self, llm: ChatGroq, target_username: str, history_text: str
    ) -> str:
        messages = [
            SystemMessage(content=(
                "Ты находишь смешные и нелепые детали в переписке. "
                "Отвечай одной строкой — только конкретный факт или цитата, без комментариев."
            )),
            HumanMessage(content=(
                f"Из сообщений @{target_username} выдели ОДНУ самую смешную, нелепую "
                f"или противоречивую деталь или фразу:\n\n{history_text}"
            )),
        ]
        response = await llm.ainvoke(messages)
        return response.content.strip()

    async def generate(self, chat_id: int, user_id: int, target_username: str) -> tuple[str, str]:
        llm = ChatGroq(model=ROAST_MODEL, api_key=config.GROQ_API_KEY, temperature=0.95, max_tokens=180)
        history_text = await self.__get_user_history_text(chat_id, target_username)
        if random.random() < 0.1:
            system_prompt, user_prompt = self.__build_supportive_prompts(target_username, history_text)
            return "🫂", (await self.__invoke(llm, system_prompt, user_prompt)).content
        mechanism = random.choice(MECHANISMS)
        fmt = random.choice(FORMATS)
        is_detective = bool(history_text) and random.random() < 0.5
        context_text = await self.__extract_roastable_moment(llm, target_username, history_text) if is_detective else history_text
        system_prompt = FORMAT_SYSTEM_PROMPTS[fmt]
        user_prompt = self.__build_user_prompt(target_username, context_text, mechanism, fmt == "protocol" and not is_detective)
        header = MECHANISM_EMOJIS[mechanism] + FORMAT_EMOJIS[fmt]
        return header, (await self.__invoke(llm, system_prompt, user_prompt)).content

    async def __invoke(self, llm: ChatGroq, system_prompt: str, user_prompt: str):
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        response = await llm.ainvoke(messages)
        return await apply_language_correction(llm, response, messages)

    async def cmd_roast(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        members = await achievements.get_chat_members(chat_id)
        if not members:
            await update.message.reply_text(
                "В базе нет участников. Пусть сначала кто-нибудь напишет в чат."
            )
            return
        target_id, target_username = random.choice(members)
        await update.message.chat.send_action("typing")
        try:
            header, roast_text = await self.generate(chat_id, target_id, target_username)
            await update.message.reply_text(f"{header} #прожарка @{target_username}\n\n{roast_text}")
            await achievements.increment_stat(target_id, chat_id, target_username, "roasted_count")
            await notify_unlocks(context, chat_id, target_id, target_username)
        except Exception as error:
            logger.error("Roast failed for %s in chat %s: %s", target_username, chat_id, error)
            await update.message.reply_text(
                "Прожарка не задалась. Groq на перекуре — попробуй позже."
            )

    async def __get_user_history_text(self, chat_id: int, username: str) -> str:
        messages = await unified_messages.get_user_messages(chat_id=chat_id, username=username, limit=40)
        meaningful = [msg for msg in messages if self.__is_meaningful(msg)]
        return "\n".join(meaningful)

    @staticmethod
    def __is_meaningful(text: str) -> bool:
        stripped = URL_RE.sub("", text)
        stripped = EMOJI_RE.sub("", stripped).strip()
        return bool(stripped)


# ---------------------------------------------------------------------------
# Module-level singleton + backward-compatible wrappers
# ---------------------------------------------------------------------------

roaster = Roaster()


async def generate_roast_text(chat_id: int, user_id: int, target_username: str) -> tuple[str, str]:
    return await roaster.generate(chat_id, user_id, target_username)


async def cmd_roast(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await roaster.cmd_roast(update, context)
