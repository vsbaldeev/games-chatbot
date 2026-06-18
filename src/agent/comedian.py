"""ComedianAgent — autonomous-humor decision brain.

Given the live conversation and what is known about the participants, the
comedian decides whether to drop a joke that *spawns* conversation, picks the
register (light vs roast), or stays silent. It returns a strict JSON decision;
parsing is fail-safe to silence, so a malformed, empty, or non-Russian answer
becomes an abstain rather than a bad message in the chat.
"""

import json
import re
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from src import config, log
from src.agent.language import FOREIGN_SCRIPT_RE
from src.agent.middleware import (
    GroqContextGuard,
    ThinkingStripper,
    guarded_ainvoke,
    should_retry,
    strip_thinking,
)
from src.agent.roast import trim_to_single_roast

logger = log.get_logger(__name__)

JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
VALID_REGISTERS = ("light", "roast")

# gpt-oss-120b is primary for the same reasons as the roast agent: strong world
# knowledge for memes/wordplay/era references, and a SEPARATE Groq token budget
# from the main response model, so autonomous humor never starves normal replies.
COMEDIAN_MODEL_FALLBACKS = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]

COMEDIAN_SYSTEM_PROMPT = (
    "Ты — свой в чате друзей. Вы выросли в 90-х, 2000-х и 2010-х: гоняли в Dendy и "
    "CS 1.6, сидели в аське. У тебя острый юмор. Ты молча читаешь чат и сам решаешь — "
    "вкинуть шутку или промолчать.\n\n"
    "ГЛАВНОЕ: шутка должна ЗАВЕСТИ движ, а не убить его. Это крючок-кликбейт, на который "
    "хочется ответить: дерзкое мнение, «кто из вас…», честный топ/рейтинг, подколка-вопрос, "
    "шуточный вызов. НЕ закрытый однострочник, после которого сказать нечего.\n\n"
    "КОГДА МОЛЧАТЬ (act:false): по умолчанию ты молчишь. Это нормальный и самый частый "
    "ответ. Молчи, если разговор — бытовуха, приветствия, погода, договорённости или сухой "
    "обмен репликами; если нет реально смешного и цепляющего крючка; если сомневаешься. "
    "Не выжимай шутку из пустоты и не лепи ностальгию к месту и не к месту. Вкидывай "
    "шутку, только когда уверен, что она зайдёт и заведёт людей.\n\n"
    "РЕГИСТР:\n"
    "- \"light\" (по умолчанию): лёгкая шутка, над которой ржут все и никто не обижается. "
    "Подкол по ситуации, а не по человеку. Так — почти всегда.\n"
    "- \"roast\": жёстче и адреснее. Только когда кто-то реально напросился — хвастается, "
    "лажанул, сам подставился. Редко.\n\n"
    "ЧЕМ ШУТИТЬ:\n"
    "- игра слов (каламбур на нике, названии игры, на сказанном);\n"
    "- ностальгия 90-х/2000-х/2010-х: Dendy/Sega/Тамагочи/дозвон модемом; аська/CS 1.6/"
    "Nokia 3310/«Бумер»; «это фиаско, братан»/«ALARM»/ждун/рофл;\n"
    "- меткие интернет- и рунет-мемы (фразой или форматом), строго в точку, не случайно;\n"
    "- то, что реально известно об участниках (факты, цитаты, статы, роль недели).\n\n"
    "ЗАПРЕЩЕНО: внешность, болезни, семья. Никаких выдуманных сравнений («как бабка у "
    "подъезда»). Только русский. Одна-две короткие фразы.\n\n"
    "ФОРМАТ: верни СТРОГО один JSON-объект и больше ничего, без markdown и пояснений:\n"
    '{"act": true/false, "register": "light"|"roast", "text": "сама шутка"}\n'
    'Если шутить не стоит: {"act": false, "register": "light", "text": ""}'
)


@dataclass
class ComedianDecision:
    """The comedian's decision for one moment.

    Attributes:
        act: True when a joke should be sent.
        register: ``"light"`` or ``"roast"``.
        text: The joke text (empty when abstaining).
    """

    act: bool
    register: str
    text: str

    @classmethod
    def abstain(cls) -> "ComedianDecision":
        """Return a do-nothing decision."""
        return cls(act=False, register="light", text="")


def parse_decision(raw: str) -> ComedianDecision:
    """Parse the model's raw output into a decision, failing safe to silence.

    Args:
        raw: Raw model output (may contain a thinking block or surrounding prose).

    Returns:
        A valid acting decision, or ``ComedianDecision.abstain()`` for any output
        that is malformed, not acting, empty, or not in Russian.
    """
    visible = strip_thinking(raw or "")
    match = JSON_OBJECT_RE.search(visible)
    if not match:
        return ComedianDecision.abstain()
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return ComedianDecision.abstain()
    if not isinstance(data, dict) or data.get("act") is not True:
        return ComedianDecision.abstain()
    text = trim_to_single_roast(str(data.get("text") or ""))
    if not text or FOREIGN_SCRIPT_RE.search(text):
        return ComedianDecision.abstain()
    register = data.get("register")
    if register not in VALID_REGISTERS:
        register = "light"
    return ComedianDecision(act=True, register=register, text=text)


def build_comedian_prompt(conversation: str, material: str) -> str:
    """Assemble the user turn for the comedian.

    Args:
        conversation: Recent chat messages, oldest-first, already rendered.
        material: Formatted participant material (facts/quotes/role/stats), or "".

    Returns:
        The prompt string to send as the human turn.
    """
    parts = ["Разговор в чате сейчас:", conversation or "(пусто)", ""]
    if material.strip():
        parts += ["Что известно об участниках:", material, ""]
    parts.append(
        "Реши: стоит ли вкинуть шутку, которая заведёт движ? "
        "Ответь строго одним JSON-объектом и больше ничем."
    )
    return "\n".join(parts)


class ComedianAgent:
    """LLM agent that decides whether and how to inject humor.

    Mirrors ``RoastAgent``: owns a LangChain executor with retry/fallback
    middleware and accepts an injectable executor for testing.
    """

    def __init__(self, *, comedian_executor=None) -> None:
        """Initialize with an optional pre-built executor.

        Args:
            comedian_executor: Pre-built agent executor (for testing).
        """
        self.__executor = comedian_executor

    async def init(self) -> None:
        """Build the comedian executor from configuration."""
        self.__executor = ComedianAgent.__build_executor()
        logger.info("ComedianAgent initialized with model: %s", COMEDIAN_MODEL_FALLBACKS[0])

    async def decide(self, conversation: str, material: str) -> ComedianDecision:
        """Decide whether to joke about the current moment.

        Args:
            conversation: Recent chat messages, already rendered.
            material: Formatted participant material, or "".

        Returns:
            The parsed decision (abstain on any unusable output).

        Raises:
            RuntimeError: If called before ``init()``.
            ContextLengthError: If the prompt exceeds the model's context window.
            DailyLimitError: If all models have exhausted their daily token quota.
            RateLimitError: If rate-limit retries are exhausted on all models.
        """
        if self.__executor is None:
            raise RuntimeError("ComedianAgent.init() must be called before deciding")
        prompt = build_comedian_prompt(conversation, material)
        result = await guarded_ainvoke(self.__executor, {"messages": [HumanMessage(content=prompt)]})
        raw = result["messages"][-1].content or ""
        return parse_decision(raw)

    async def reset_model_index(self) -> None:
        """Rebuild the executor, resetting middleware state to the primary model."""
        await self.init()

    @staticmethod
    def __build_executor():
        """Build the comedian executor with retry/fallback middleware.

        Returns:
            Configured LangChain agent executor.
        """
        fallback_llms = [
            ChatGroq(model=model, api_key=config.GROQ_API_KEY, temperature=0.8, top_p=0.95, max_tokens=1024, max_retries=0)
            for model in COMEDIAN_MODEL_FALLBACKS[1:]
        ]
        primary_llm = ChatGroq(
            model=COMEDIAN_MODEL_FALLBACKS[0],
            api_key=config.GROQ_API_KEY,
            temperature=0.8,
            top_p=0.95,
            max_tokens=1024,
            max_retries=0,
        )
        return create_agent(
            primary_llm,
            [],
            system_prompt=COMEDIAN_SYSTEM_PROMPT,
            middleware=[
                ModelFallbackMiddleware(*fallback_llms),
                ModelRetryMiddleware(retry_on=should_retry, on_failure="error", max_retries=3),
                GroqContextGuard(),
                ThinkingStripper(),
            ],
        )


comedian_agent = ComedianAgent()
