"""Alembic migration environment.

Runs migrations against the same PostgreSQL database the bot uses. The
connection URL is derived from the ``DATABASE_URL`` environment variable via
``src.config.db_url`` — a lightweight module that pulls in no bot credentials —
so migrations can run without a Telegram token or LLM keys present.

Schema is defined as raw SQL inside the version scripts (the application uses
asyncpg directly with no SQLAlchemy models), so ``target_metadata`` is ``None``
and autogenerate is not used.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from src.db_url import get_database_url, to_sqlalchemy_url

SQLALCHEMY_DB_URL = to_sqlalchemy_url(get_database_url())

config = context.config
config.set_main_option("sqlalchemy.url", SQLALCHEMY_DB_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode, emitting SQL to stdout."""
    context.configure(
        url=SQLALCHEMY_DB_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a live database connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
