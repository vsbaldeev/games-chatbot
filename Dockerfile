FROM python:3.13-slim

WORKDIR /app

# git is required to install igdb-mcp-server from GitHub (not on PyPI);
# curl/unzip fetch the deno binary below
RUN apt-get update && apt-get install -y --no-install-recommends git curl unzip \
    && rm -rf /var/lib/apt/lists/*

# yt-dlp needs a JS runtime to solve YouTube's signature/n-parameter
# challenges; without one it silently falls back to non-JS player clients
# that are missing many formats — including format 18, the one Shorts
# downloads pin to (src/pipeline/shorts.py's SHORT_FORMAT) — so extraction
# intermittently (in practice: every time) fails with "Requested format is
# not available" and no indication why (see shorts.py's YtdlpLogger for how
# that surfaced). Deno is yt-dlp's default/recommended runtime — just being
# on PATH is enough, no yt-dlp-side config needed. Checksum-verified since
# this runs as root during build.
ENV DENO_VERSION=2.9.5
RUN cd /tmp \
    && curl -fsSLO "https://github.com/denoland/deno/releases/download/v${DENO_VERSION}/deno-x86_64-unknown-linux-gnu.zip" \
    && curl -fsSLO "https://github.com/denoland/deno/releases/download/v${DENO_VERSION}/deno-x86_64-unknown-linux-gnu.zip.sha256sum" \
    && sha256sum -c deno-x86_64-unknown-linux-gnu.zip.sha256sum \
    && unzip -q deno-x86_64-unknown-linux-gnu.zip -d /usr/local/bin \
    && chmod +x /usr/local/bin/deno \
    && rm deno-x86_64-unknown-linux-gnu.zip deno-x86_64-unknown-linux-gnu.zip.sha256sum

# Install dependencies in a separate layer so rebuilds on code changes are fast
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY alembic/ ./alembic/
COPY alembic.ini entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Pre-bake the embedding model so the container starts without a download
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed
RUN mkdir -p /app/.cache/fastembed \
    && python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_dir='/app/.cache/fastembed')"

# Pre-bake the Silero TTS model so the container starts without a download
ENV TTS_MODEL_PATH=/app/.cache/silero/v5_ru.pt
RUN mkdir -p /app/.cache/silero \
    && python -c "import torch; torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v5_ru.pt', '/app/.cache/silero/v5_ru.pt')"

# Writable overlay for runtime yt-dlp self-updates: entrypoint.sh (and the
# daily update job) pip-install into it as the non-root user, and PYTHONPATH
# makes it shadow the baked copy from requirements.txt.
ENV PYTHONPATH=/app/runtime-deps

# Non-root user for the process; /data will be the volume mount for SQLite
RUN useradd --no-create-home --shell /bin/false botuser \
    && mkdir /data \
    && mkdir /app/runtime-deps \
    && chown botuser:botuser /data /app/runtime-deps \
    && chown -R botuser:botuser /app/.cache/fastembed

USER botuser

CMD ["./entrypoint.sh"]
