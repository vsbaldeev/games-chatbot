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

# Pre-bake the embedding model so the container starts without a download
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed
RUN mkdir -p /app/.cache/fastembed \
    && python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_dir='/app/.cache/fastembed')"

# Non-root user for the process; /data will be the volume mount for SQLite
RUN useradd --no-create-home --shell /bin/false botuser \
    && mkdir /data \
    && chown botuser:botuser /data \
    && chown -R botuser:botuser /app/.cache/fastembed

USER botuser

CMD ["python", "-m", "src.bot"]
