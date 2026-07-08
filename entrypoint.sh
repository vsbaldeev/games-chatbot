#!/bin/sh
# Container entrypoint: bring the database schema up to date, then start the bot.
#
# Alembic owns the schema (the bot no longer creates tables). Running
# `alembic upgrade head` on every start is idempotent — it is a no-op when the
# database is already current.
set -e

# YouTube breaks yt-dlp's extractor regularly; refreshing it on every start
# keeps the Shorts-summary feature alive without rebuilding the image. The
# --target overlay shadows the baked copy via PYTHONPATH (see Dockerfile).
# Failure-tolerant: a network hiccup must never block bot start.
echo "Updating yt-dlp..."
pip install --no-cache-dir --upgrade --target /app/runtime-deps yt-dlp \
    || echo "yt-dlp update failed — starting with the baked version"

echo "Applying database migrations..."
alembic upgrade head

echo "Starting bot..."
exec python -m src.bot
