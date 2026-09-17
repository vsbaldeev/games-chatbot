#!/bin/sh
# Container entrypoint: refresh yt-dlp, then start the service.
set -e

# YouTube/Instagram break yt-dlp's extractors regularly; refreshing on every
# start keeps downloads alive without rebuilding the image. The --target
# overlay shadows the baked copy via PYTHONPATH (see Dockerfile). Extras
# must match requirements.txt's yt-dlp line: curl-cffi (Instagram
# extraction) and default (bundles yt-dlp-ejs, yt-dlp's own JS
# challenge-solver scripts, exact-pinned to yt-dlp's own version). A bare
# `pip install yt-dlp` here would upgrade yt-dlp without its matching
# yt-dlp-ejs, silently breaking JS-challenge-dependent formats the moment
# the two drift apart.
# Failure-tolerant: a network hiccup must never block the service starting.
echo "Updating yt-dlp..."
pip install --no-cache-dir --upgrade --target /service/runtime-deps "yt-dlp[curl-cffi,default]" \
    || echo "yt-dlp update failed — starting with the baked version"

echo "Starting download service..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
