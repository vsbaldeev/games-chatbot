import os
from dotenv import load_dotenv

load_dotenv()


def __require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"Required environment variable '{name}' is not set. "
            f"Copy .env.example to .env and fill in the values."
        )
    return value


TELEGRAM_TOKEN: str = __require("TELEGRAM_TOKEN")
BOT_ID: int = int(TELEGRAM_TOKEN.split(":")[0])
GROQ_API_KEY: str = __require("GROQ_API_KEY")
TWITCH_CLIENT_ID: str = __require("TWITCH_CLIENT_ID")
TWITCH_CLIENT_SECRET: str = __require("TWITCH_CLIENT_SECRET")
BOT_USERNAME: str = __require("BOT_USERNAME")

SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "data/chat_history.db")
SQLITE_DB_URL: str = f"sqlite:///{SQLITE_DB_PATH}"
MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))

# Optional — leave empty to disable the respective service
TMDB_API_KEY: str = os.getenv("TMDB_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

