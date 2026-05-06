"""
MeaninglessFilterNode — third node in the LangGraph pipeline.

Uses an LLM to decide if the message (text or transcribed media) is a "meaningless"
reaction that does not deserve a response (e.g. "ахаха", "lol", "бляяя", "ок").

If classified as MEANINGLESS, sets should_respond=False and fires an emoji reaction
as a lightweight acknowledgement (asyncio.create_task — does not block the pipeline).
"""

import asyncio
import random

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from telegram import ReactionTypeEmoji

from src import config, log
from src.pipeline.state import BotState

logger = log.get_logger(__name__)

REACTION_POOL = ["👍", "❤️", "🔥", "😂", "👀", "🎮", "😎", "💯", "🤣", "⚡", "🫡", "🤙"]

FILTER_SYSTEM = (
    "You are a telegram bot's message filter. "
    "Determine if the user's message is a 'meaningless' reaction that does NOT require a response.\n\n"
    "MEANINGLESS examples:\n"
    "- Laughter: 'ахаха', 'hahaha', 'lol', 'rofl', 'ыыы', '😂😂😂'\n"
    "- Short swearing/interjections: 'бля', 'пиздец', 'wtf', 'офигеть', 'жесть'\n"
    "- Acknowledgments: 'ок', 'окей', 'понял', 'ясно', 'ладно', 'хорошо' (when used as a reaction)\n"
    "- Emojis only: '👍', '🤔', '❤️'\n"
    "- Meaningless filler: 'ну', 'мда', 'хз'\n\n"
    "MEANINGFUL examples:\n"
    "- Questions: 'Как дела?', 'Что нового?'\n"
    "- Commands/Requests: 'Расскажи анекдот', '/duel @user'\n"
    "- Opinions/Descriptions: 'Эта игра просто супер, мне нравится графика'\n"
    "- Greetings: 'Привет', 'Добрый вечер' (bot should greet back)\n"
    "- Starting or continuing a discussion.\n\n"
    "Instructions:\n"
    "1. Analyze the text (can be in Russian or English).\n"
    "2. Reply with ONLY ONE word: 'MEANINGLESS' or 'MEANINGFUL'.\n"
    "3. If unsure, err on the side of 'MEANINGFUL'."
)


class MeaninglessFilterNode:
    def __init__(self) -> None:
        self.__llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=config.GROQ_API_KEY,
            temperature=0.0,
            max_tokens=5,
        )

    async def __call__(self, state: BotState) -> dict:
        if not state.get("should_respond"):
            return {}

        media_type = state["incoming"]["media_type"]
        if media_type != "text":
            text = state["incoming"]["processed_text"] or ""
            if not text.strip():
                logger.warning(
                    "Filter: no transcription for %s message %s, skipping",
                    media_type,
                    state["incoming"]["message_id"],
                )
                asyncio.create_task(self.__send_reaction(state))
                return {"should_respond": False}
            return {}

        text = state["incoming"]["raw_text"] or ""
        if not text.strip():
            return {"should_respond": False}

        decision = await self.__classify(text)

        if decision == "MEANINGLESS":
            logger.info("Filter: Dropping meaningless message %s", state["incoming"]["message_id"])
            asyncio.create_task(self.__send_reaction(state))
            return {"should_respond": False}

        return {"should_respond": True}

    async def __classify(self, text: str) -> str:
        try:
            response = await self.__llm.ainvoke([
                SystemMessage(content=FILTER_SYSTEM),
                HumanMessage(content=text),
            ])
            result = response.content.strip().upper()
            return "MEANINGLESS" if "MEANINGLESS" in result else "MEANINGFUL"
        except Exception as err:
            logger.warning("Meaningless filter failed, failing open (MEANINGFUL): %s", err)
            return "MEANINGFUL"

    async def __send_reaction(self, state: BotState) -> None:
        try:
            bot = state["context_types"].bot
            msg = state["incoming"]
            emoji = random.choice(REACTION_POOL)
            await bot.set_message_reaction(
                chat_id=msg["chat_id"],
                message_id=msg["message_id"],
                reaction=[ReactionTypeEmoji(emoji=emoji)],
            )
        except Exception as err:
            logger.debug("Reaction failed for message %s: %s", state["incoming"]["message_id"], err)
