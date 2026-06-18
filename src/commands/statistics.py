"""Statistics command handlers — achievements and leaderboard."""

from telegram import Update
from telegram.ext import ContextTypes
from telegram.helpers import escape_markdown

from src import achievements
from src.events.members import get_username


async def cmd_achievements(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    username = get_username(update)
    earned = await achievements.get_user_achievements(update.effective_user.id, chat_id)
    if not earned:
        await update.message.reply_text(
            f"{username}, достижений нет. Либо ты новенький, либо слишком нормальный. Оба варианта подозрительны."
        )
        return
    safe_username = escape_markdown(username, version=1)
    recent = earned[-3:]
    lines = [
        f"{achievement.emoji} *{achievement.title}*\n_{achievement.description}_"
        for achievement in recent
    ]
    total = len(earned)
    header = f"Последние достижения {safe_username} ({total} всего):"
    await update.message.reply_text(
        f"{header}\n\n" + "\n\n".join(lines),
        parse_mode="Markdown",
    )


async def cmd_top(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    summary = await achievements.get_chat_achievements_summary(chat_id)

    if not summary:
        await update.message.reply_text("Ни у кого нет достижений. Пишите, реагируйте — зарабатывайте.")
        return

    ranked = sorted(summary.items(), key=lambda item: len(item[1]), reverse=True)[:3]

    medals = ["🥇", "🥈", "🥉"]
    lines = []
    for index, (username, earned) in enumerate(ranked):
        count = len(earned)
        if 11 <= (count % 100) <= 14:
            label = f"{count} достижений"
        elif (count % 10) == 1:
            label = f"{count} достижение"
        elif 2 <= (count % 10) <= 4:
            label = f"{count} достижения"
        else:
            label = f"{count} достижений"
        lines.append(f"{medals[index]} {username} — {label}")

    await update.message.reply_text("🏆 Топ чата:\n\n" + "\n".join(lines))
