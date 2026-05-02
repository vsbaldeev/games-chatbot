FROM python:3.13-slim

WORKDIR /app

# git is required to install igdb-mcp-server from GitHub (not on PyPI)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies in a separate layer so rebuilds on code changes are fast
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Non-root user for the process; /data will be the volume mount for SQLite
RUN useradd --no-create-home --shell /bin/false botuser \
    && mkdir /data \
    && chown botuser:botuser /data

USER botuser

CMD ["python", "-m", "src.bot"]
