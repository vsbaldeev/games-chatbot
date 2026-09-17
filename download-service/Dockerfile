FROM python:3.13-slim

WORKDIR /service

# curl/unzip fetch the deno binary below.
RUN apt-get update && apt-get install -y --no-install-recommends curl unzip \
    && rm -rf /var/lib/apt/lists/*

# yt-dlp needs a JS runtime to solve YouTube's signature/n-parameter
# challenges for Shorts downloads (shorts.py's SHORT_FORMAT pins format 18,
# which needs the "n" parameter descrambled) — without one it silently
# degrades to player clients missing that format. Checksum-verified since
# this runs as root during build.
ENV DENO_VERSION=2.9.5
RUN curl -fsSLO "https://github.com/denoland/deno/releases/download/v${DENO_VERSION}/deno-x86_64-unknown-linux-gnu.zip" \
    && curl -fsSLO "https://github.com/denoland/deno/releases/download/v${DENO_VERSION}/deno-x86_64-unknown-linux-gnu.zip.sha256sum" \
    && sha256sum -c deno-x86_64-unknown-linux-gnu.zip.sha256sum \
    && unzip -q deno-x86_64-unknown-linux-gnu.zip -d /usr/local/bin \
    && chmod +x /usr/local/bin/deno \
    && rm deno-x86_64-unknown-linux-gnu.zip deno-x86_64-unknown-linux-gnu.zip.sha256sum

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x entrypoint.sh

# Writable overlay for runtime yt-dlp self-updates: entrypoint.sh (and the
# daily update job) pip-install into it as the non-root user, and PYTHONPATH
# makes it shadow the baked copy from requirements.txt.
ENV PYTHONPATH=/service/runtime-deps

RUN useradd --no-create-home --shell /bin/false serviceuser \
    && mkdir /service/runtime-deps \
    && chown serviceuser:serviceuser /service/runtime-deps

USER serviceuser

EXPOSE 8000
CMD ["./entrypoint.sh"]
