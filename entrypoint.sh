#!/bin/sh
# Container entrypoint: bring the database schema up to date, then start the bot.
#
# Alembic owns the schema (the bot no longer creates tables). Running
# `alembic upgrade head` on every start is idempotent — it is a no-op when the
# database is already current.
set -e

echo "Applying database migrations..."
alembic upgrade head

echo "Starting bot..."
exec python -m src.app
