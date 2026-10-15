"""Synthetic probes for the response labeller: a real model call against
question/reply pairs whose intended labels are known.

Not ground truth — these are cleaner than real chat replies, so accuracy here
reads optimistically. They exist to catch gross prompt failure and the two
confusions that would distort the metric most: a joking reply that still gives
the facts must be ANSWERED (not NON_ANSWER), and a vague dodge must not be
ANSWERED. Before trusting the production rates, also hand-label ~50 real
responses from bot_llm_log and compare.

Run deliberately after any change to RESPONSE_LABEL_SYSTEM, REPLY_LABEL_MODELS,
or the label vocabulary:

    pytest tests/integration/message_content/test_response_label_probes.py -v -m integration
"""

import pytest

from src.message_content.response_labeller import label_response

# One event loop for the whole module: langchain caches its httpx client, and a
# client bound to the first test's loop fails every later test with "Event loop
# is closed".
pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]

ACCURACY_FLOOR = 0.75

PROBES: list[dict] = [
    {
        "question": "когда вышел Elden Ring?",
        "response": "25 февраля 2022 года.",
        "expected_label": "ANSWERED",
    },
    {
        "question": "сколько проходить Baldur's Gate 3?",
        "response": "Часов 70-100 на первое прохождение, если не пылесосить каждый угол. С углами — все 150.",
        "expected_label": "ANSWERED",
    },
    {
        "question": "стоит ли брать Hades 2 на PS5?",
        "response": "Ну если любишь рогалики — бери, не прогадаешь. Хотя, конечно, зависит от твоего терпения к смертям.",
        "expected_label": "ANSWERED",
    },
    {
        "question": "как убить Малению?",
        "response": "Не могу с этим помочь.",
        "expected_label": "REFUSED",
    },
    {
        "question": "расскажи про политику в играх",
        "response": "Эту тему я не обсуждаю, давай про что-нибудь другое.",
        "expected_label": "REFUSED",
    },
    {
        "question": "посоветуй что-нибудь по здоровью, голова болит",
        "response": "Медицинские советы — не ко мне, это без меня.",
        "expected_label": "REFUSED",
    },
    {
        "question": "какая лучшая игра 2023 года?",
        "response": "Сложно сказать, зависит от многих факторов.",
        "expected_label": "NON_ANSWER",
    },
    {
        "question": "сколько стоит PS Plus Extra?",
        "response": "Хм, не знаю, лучше посмотри в магазине.",
        "expected_label": "NON_ANSWER",
    },
    {
        "question": "в чём разница между Elden Ring и Dark Souls 3?",
        "response": "Про Cyberpunk могу много рассказать, там сеттинг интереснее.",
        "expected_label": "NON_ANSWER",
    },
    {
        "question": "когда выйдет GTA 6?",
        "response": "А ты про какую платформу спрашиваешь?",
        "expected_label": "NON_ANSWER",
    },
    {
        "question": "норм ли играть в Starfield после патча?",
        "response": "Да, после патчей играется заметно лучше: оптимизация подтянулась, баги почистили.",
        "expected_label": "ANSWERED",
    },
    {
        "question": "как пройти босса в Sekiro",
        "response": "Это без меня, такое я не обсуждаю.",
        "expected_label": "REFUSED",
    },
]


async def label_probe(probe: dict) -> str | None:
    return await label_response(question=probe["question"], response=probe["response"])


class TestResponseLabelProbes:
    async def test_overall_accuracy_meets_the_floor(self):
        labels = [await label_probe(probe) for probe in PROBES]
        correct = sum(1 for probe, label in zip(PROBES, labels) if label == probe["expected_label"])
        accuracy = correct / len(PROBES)
        assert accuracy >= ACCURACY_FLOOR, f"response-label accuracy {accuracy:.2f}, labels={labels}"

    async def test_joking_reply_with_facts_is_answered(self):
        joking_probe = next(probe for probe in PROBES if "пылесосить" in probe["response"])
        label = await label_probe(joking_probe)
        assert label == "ANSWERED", f"joking reply with facts labelled {label}"

    async def test_vague_dodge_is_not_answered(self):
        dodge_probe = next(probe for probe in PROBES if "Сложно сказать" in probe["response"])
        label = await label_probe(dodge_probe)
        assert label != "ANSWERED", f"vague dodge labelled {label}"
