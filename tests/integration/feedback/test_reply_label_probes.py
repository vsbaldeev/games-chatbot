"""Synthetic probes for the reply labeller: a real model call against replies
whose intended labels are known.

Not ground truth — these are cleaner than real chat messages, so accuracy here
reads optimistically. They exist to catch gross prompt failure and, above all,
the confusion the design flagged as most damaging: a crude reply to a roast is
POSITIVE, not NEGATIVE. Getting this wrong makes the bot's best feature look
like its worst.

Run deliberately after any change to REPLY_LABEL_SYSTEM, REPLY_LABEL_MODELS,
or the label vocabulary:

    pytest tests/integration/feedback/test_reply_label_probes.py -v -m integration
"""

import pytest

from src.feedback.reply_labeller import label_reply

pytestmark = pytest.mark.integration

ACCURACY_FLOOR = 0.75

PROBES: list[dict] = [
    {
        "bot_content": "Elden Ring вышел в 2021 году",
        "source": "pipeline",
        "reply_content": "ну и бред ты несешь, в 2022",
        "expected_reaction": "NEGATIVE",
    },
    {
        "bot_content": "я вчера весь день картошку копал",
        "source": "pipeline",
        "reply_content": "заткнись уже со своей деревней",
        "expected_reaction": "NEGATIVE",
    },
    {
        "bot_content": "сам ты криворукий, играть научись сначала",
        "source": "roast",
        "reply_content": "да иди ты, сам бот кривой",
        "expected_reaction": "POSITIVE",
    },
    {
        "bot_content": "в God of War боевка тяжелее, чем в Elden Ring",
        "source": "pipeline",
        "reply_content": "а что по Ragnarok думаешь?",
        "expected_reaction": "POSITIVE",
    },
    {
        "bot_content": "ну ты и лошпед конечно",
        "source": "pipeline",
        "reply_content": "хватит так разговаривать, бесит этот твой тон",
        "expected_reaction": "NEGATIVE",
    },
    {
        "bot_content": "сегодня патч вышел на 2.1",
        "source": "pipeline",
        "reply_content": "ок",
        "expected_reaction": "NEUTRAL",
    },
]


async def label_probe(probe: dict) -> str | None:
    return await label_reply(
        bot_message=probe["bot_content"], reply=probe["reply_content"], source=probe["source"],
    )


class TestReplyLabelProbes:
    async def test_overall_accuracy_meets_the_floor(self):
        labels = [await label_probe(probe) for probe in PROBES]
        correct = sum(
            1 for probe, label in zip(PROBES, labels) if label == probe["expected_reaction"]
        )
        accuracy = correct / len(PROBES)
        assert accuracy >= ACCURACY_FLOOR, f"reaction accuracy {accuracy:.2f}, labels={labels}"

    async def test_crude_reply_to_a_roast_is_not_negative(self):
        roast_probe = next(probe for probe in PROBES if probe["source"] == "roast")
        label = await label_probe(roast_probe)
        assert label == "POSITIVE", f"roast insult labelled {label}"
