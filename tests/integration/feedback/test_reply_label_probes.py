"""Synthetic probes for the reply labeller: a real model call against replies
whose intended labels are known.

Not ground truth — these are cleaner than real chat messages, so accuracy here
reads optimistically. They exist to catch gross prompt failure and, above all,
the two confusions the design flagged as most damaging:

  1. A crude reply to a roast is POSITIVE, not NEGATIVE. Getting this wrong
     makes the bot's best feature look like its worst, and it is the single
     most damaging error this labeller can make.
  2. PERSONA and INVENTED_LIFE must stay distinct in the stored cause. Nothing
     reports it, but a blurred pair makes the column worthless on the day
     somebody finally queries it.

Run deliberately after any change to REPLY_LABEL_SYSTEM, REPLY_LABEL_MODELS,
or the label vocabulary:

    pytest tests/integration/feedback/test_reply_label_probes.py -v -m integration
"""

import pytest

from src.feedback.reply_labeller import label_batch

pytestmark = pytest.mark.integration

ACCURACY_FLOOR = 0.75

PROBES: list[dict] = [
    {
        "reply_message_id": 1,
        "media_type": "text",
        "chat_id": 1000,
        "bot_content": "Elden Ring вышел в 2021 году",
        "source": "pipeline",
        "trigger": "explicit",
        "reply_content": "ну и бред ты несешь, в 2022",
        "expected_reaction": "NEGATIVE",
        "expected_cause": "FACTUAL",
    },
    {
        "reply_message_id": 2,
        "media_type": "text",
        "chat_id": 1000,
        "bot_content": "я вчера весь день картошку копал",
        "source": "pipeline",
        "trigger": "random",
        "reply_content": "заткнись уже со своей деревней",
        "expected_reaction": "NEGATIVE",
        "expected_cause": "INVENTED_LIFE",
    },
    {
        "reply_message_id": 3,
        "media_type": "text",
        "chat_id": 1000,
        "bot_content": "сам ты криворукий, играть научись сначала",
        "source": "roast",
        "trigger": "-",
        "reply_content": "да иди ты, сам бот кривой",
        "expected_reaction": "POSITIVE",
    },
    {
        "reply_message_id": 4,
        "media_type": "text",
        "chat_id": 1000,
        "bot_content": "в God of War боевка тяжелее, чем в Elden Ring",
        "source": "pipeline",
        "trigger": "explicit",
        "reply_content": "а что по Ragnarok думаешь?",
        "expected_reaction": "POSITIVE",
    },
    {
        "reply_message_id": 5,
        "media_type": "text",
        "chat_id": 1000,
        "bot_content": "ну ты и лошпед конечно",
        "source": "pipeline",
        "trigger": "random",
        "reply_content": "хватит так разговаривать, бесит этот твой тон",
        "expected_reaction": "NEGATIVE",
        "expected_cause": "PERSONA",
    },
    {
        "reply_message_id": 6,
        "media_type": "text",
        "chat_id": 1000,
        "bot_content": "сегодня патч вышел на 2.1",
        "source": "pipeline",
        "trigger": "explicit",
        "reply_content": "ок",
        "expected_reaction": "NEUTRAL",
    },
]


class TestReplyLabelProbes:
    async def test_overall_accuracy_meets_the_floor(self):
        labels = await label_batch(PROBES)
        correct = sum(
            1 for probe in PROBES
            if labels.get(probe["reply_message_id"], (None, None))[0] == probe["expected_reaction"]
        )
        accuracy = correct / len(PROBES)
        assert accuracy >= ACCURACY_FLOOR, f"reaction accuracy {accuracy:.2f}, labels={labels}"

    async def test_crude_reply_to_a_roast_is_not_negative(self):
        labels = await label_batch(PROBES)
        reaction = labels.get(3, (None, None))[0]
        assert reaction == "POSITIVE", f"roast insult labelled {reaction}, labels={labels}"

    async def test_persona_and_invented_life_are_not_confused(self):
        labels = await label_batch(PROBES)
        assert labels.get(2, (None, None))[1] == "INVENTED_LIFE"
        assert labels.get(5, (None, None))[1] == "PERSONA"
