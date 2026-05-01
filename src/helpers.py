import datetime
from src import log
import re

from telegram import Update

from src import config

logger = log.get_logger(__name__)

OFFENSE_RE = re.compile(
    r"(тупой|тупая|тупит|идиот|дебил|мудак|г[ао]вн[оа]|хуйн[яе]|нахуй|пиздец|"
    r"отстой|бесполезн|сломан|не работает|глупый|глупая|дерьм[оа]|придур|долбо|"
    r"ёбан|еба[нл]|заткн|иди нах|иди в|stupid|useless|broken|dumb|trash|"
    r"garbage|sucks|piece of shit|fuck)",
    re.IGNORECASE | re.UNICODE,
)

MOSCOW_TZ = datetime.timezone(datetime.timedelta(hours=3))

__INTERMEDIATE_LINE_RE = re.compile(r"^(#{1,3} |\d+\. | {2,}- )")


def get_username(update: Update) -> str:
    user = update.effective_user
    return user.username or user.first_name or f"user_{user.id}"


def is_reply_to_bot(update: Update, bot_id: int) -> bool:
    reply = update.message.reply_to_message
    if not reply or not reply.from_user:
        return False
    return reply.from_user.id == bot_id



def is_night_message(update: Update) -> bool:
    """True if the message was sent between 00:00 and 05:00 Moscow time."""
    if not update.message or not update.message.date:
        return False
    moscow_time = update.message.date.astimezone(MOSCOW_TZ)
    return 0 <= moscow_time.hour < 5


