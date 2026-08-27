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
# Extras must match requirements.txt's yt-dlp line: curl-cffi (Instagram
# extraction) and default (bundles yt-dlp-ejs, yt-dlp's own JS
# challenge-solver scripts, exact-pinned by yt-dlp to match its own
# version). A bare `pip install yt-dlp` here would upgrade yt-dlp without
# its matching yt-dlp-ejs, silently breaking JS-challenge-dependent
# formats the moment the two drift apart.
# Failure-tolerant: a network hiccup must never block bot start.
echo "Updating yt-dlp..."
pip install --no-cache-dir --upgrade --target /app/runtime-deps "yt-dlp[curl-cffi,default]" \
    || echo "yt-dlp update failed — starting with the baked version"

echo "Applying database migrations..."
alembic upgrade head

echo "Starting bot..."
exec python -m src.bot
