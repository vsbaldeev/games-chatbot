"""Russian calendar rendering shared by the life-post writer, the daily
activity refresh and the response prompt — all three need the same
season/weekday/relative-day phrasing so Жора's timeline reads consistently.
"""

import datetime
from zoneinfo import ZoneInfo

MOSCOW_TZ = ZoneInfo("Europe/Moscow")

MONTH_NAMES_RU = (
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)

WEEKDAY_NAMES_RU = (
    "понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье",
)

SEASON_NAMES_RU = ("зима", "весна", "лето", "осень")


def season_name(month: int) -> str:
    """Return the Russian name of the meteorological season for a month.

    Args:
        month: Calendar month, 1-12.

    Returns:
        One of «зима», «весна», «лето», «осень» (December-February is
        winter, and so on in three-month blocks).
    """
    return SEASON_NAMES_RU[(month % 12) // 3]


def describe_moscow_date(now: datetime.datetime) -> str:
    """Render a full Russian date/weekday/season line for a Moscow-aware moment.

    Args:
        now: A timezone-aware moment already in Moscow Time.

    Returns:
        A string like «11 июля 2026, пятница, лето».
    """
    weekday = WEEKDAY_NAMES_RU[now.weekday()]
    month = MONTH_NAMES_RU[now.month - 1]
    season = season_name(now.month)
    return f"{now.day} {month} {now.year}, {weekday}, {season}"


def describe_relative_day(posted_at: float, now: datetime.datetime) -> str:
    """Render a relative-day label for a past timestamp against a Moscow "now".

    Args:
        posted_at: Unix timestamp of the past event.
        now: A timezone-aware moment already in Moscow Time.

    Returns:
        «сегодня», «вчера», «позавчера», or a bare «8 июля» date once the
        gap exceeds two days.
    """
    posted = datetime.datetime.fromtimestamp(posted_at, tz=now.tzinfo)
    day_diff = (now.date() - posted.date()).days
    if day_diff == 0:
        return "сегодня"
    if day_diff == 1:
        return "вчера"
    if day_diff == 2:
        return "позавчера"
    return f"{posted.day} {MONTH_NAMES_RU[posted.month - 1]}"
