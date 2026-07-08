FROM python:3.13-slim

WORKDIR /app

# git is required to install igdb-mcp-server from GitHub (not on PyPI)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

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
