"""Runtime credentials loaded from environment variables."""

import os

from dotenv import load_dotenv

from src.db_url import get_database_url, to_sqlalchemy_url

load_dotenv()


def require_env(name: str) -> str:
    """Return the value of a required environment variable.

    Args:
        name: Environment variable name.

    Returns:
        The variable's value.

    Raises:
        ValueError: If the variable is not set or empty.
    """
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"Required environment variable '{name}' is not set. "
            f"Copy .env.example to .env and fill in the values."
        )
    return value


TELEGRAM_TOKEN: str = require_env("TELEGRAM_TOKEN")
BOT_ID: int = int(TELEGRAM_TOKEN.split(":")[0])
GROQ_API_KEY: str = require_env("GROQ_API_KEY")
TWITCH_CLIENT_ID: str = require_env("TWITCH_CLIENT_ID")
TWITCH_CLIENT_SECRET: str = require_env("TWITCH_CLIENT_SECRET")
BOT_USERNAME: str = require_env("BOT_USERNAME")

DATABASE_URL: str = get_database_url()
# SQLAlchemy sync URL for LangChain (psycopg2 driver)
SQLALCHEMY_DB_URL: str = to_sqlalchemy_url(DATABASE_URL)
MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))

# Optional — leave empty to disable the respective service
TMDB_API_KEY: str = os.getenv("TMDB_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

# Local path of the Silero TTS model file. Downloaded on first start when
# missing; the Docker image pre-bakes it (see Dockerfile).
TTS_MODEL_PATH: str = os.getenv("TTS_MODEL_PATH", ".cache/silero/v5_ru.pt")
