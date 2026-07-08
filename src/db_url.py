"""Database URL resolution, decoupled from the rest of the configuration.

Deliberately kept out of the ``src.config`` package: importing anything from
``src.config`` triggers ``src/config/__init__.py``, which eagerly loads bot
credentials (Telegram token, LLM keys). Database tooling — the Alembic
migration environment — needs only the connection URL, so it imports this
standalone module and can run without those credentials present.
"""

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_DATABASE_URL = "postgresql://chatbot:changeme@localhost:5432/chatbot"


def get_database_url() -> str:
    """Return the asyncpg-style database URL from the environment.

    Returns:
        The ``DATABASE_URL`` value, or a local-development default when unset.
    """
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def to_sqlalchemy_url(database_url: str) -> str:
    """Convert an asyncpg/plain URL to a SQLAlchemy psycopg2 sync URL.

    Args:
        database_url: A ``postgresql://``, ``postgres://`` or
            ``postgresql+asyncpg://`` URL.

    Returns:
        The equivalent ``postgresql+psycopg2://`` URL used by SQLAlchemy and
        Alembic.
    """
    return (
        database_url
        .replace("postgresql+asyncpg://", "postgresql+psycopg2://")
        .replace("postgresql://", "postgresql+psycopg2://")
        .replace("postgres://", "postgresql+psycopg2://")
    )
