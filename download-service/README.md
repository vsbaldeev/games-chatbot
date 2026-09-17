Self-hosted yt-dlp download service — owns every yt-dlp download for the bot
(YouTube Shorts, Instagram Reels, long-form YouTube metadata), so a yt-dlp
version bump restarts only this container, never the bot
(`src/downloads/README.md` has the client side).

## Contract

`POST /download {"url": str, "kind": "youtube_short" | "instagram_reel" | "youtube_video"}`
→ `{"video_bytes_base64": str | null, "info": dict}`, or an error status on
failure (502 when the downloader raised, 422 on a malformed request).
`video_bytes_base64` is null for `youtube_video` (metadata-only).

`GET /healthz` → `{"status": "ok"}`.

Three separate modules, not one generic downloader — `shorts.py`,
`instagram_reel.py`, `youtube_video.py` each own their own yt-dlp format
string, player-client pinning, retry signals, and guards. `main.py` only
dispatches by `kind`.

## Self-update

`entrypoint.sh` upgrades yt-dlp into `/service/runtime-deps` (a writable
overlay that shadows the baked package via `PYTHONPATH`) on every container
start. `ytdlp_update.py` checks PyPI daily at 03:30 UTC; when a newer
version exists it installs it and sends itself SIGTERM — `restart:
unless-stopped` (docker-compose.yml) brings the container back on the fresh
version. This restarts only this container: the bot process, Telegram
connection, and every other feature are untouched. Installs use the
`yt-dlp[curl-cffi,default]` extra — `curl-cffi` for Instagram's browser
impersonation, `default` for yt-dlp-ejs (the JS challenge-solver, exact-pinned
by yt-dlp to its own version) — a bare `yt-dlp` install would drift the two
apart and silently break JS-challenge-dependent formats.

## YouTube Shorts mechanics

PO tokens for YouTube bot-detection come automatically from the
`pot-provider` docker-compose sidecar via the bgutil yt-dlp plugin. A JS
runtime (deno, installed in the Dockerfile) solves YouTube's
signature/n-parameter challenges — format 18 (`shorts.py`'s `SHORT_FORMAT`
pin) needs the "n" parameter descrambled. `SHORTS_PLAYER_CLIENTS` pins an
explicit player-client set rather than trusting yt-dlp's shifting default,
which has been observed picking a client with no format 18 at all; see the
comment on that constant in `shorts.py` for the full history of why the set
itself already needed revising once. `YtdlpLogger` routes yt-dlp's internal
warnings through this service's logger — `quiet`/`no_warnings` would
otherwise discard the PO-token/player-client failures that are the actual
cause when a format goes missing.

## Instagram Reel mechanics

Anonymous, no authentication — Instagram aggressively blocks anonymous
scraping, so most failures are expected and simply propagate as a 502.
Instagram's own access-check API withholds its CSRF token inconsistently
(not per-post), so that specific signature gets a bounded retry
(`INSTAGRAM_ACCESS_GATE_SIGNAL`); every other failure fails fast.

## Testing

`pip install -r requirements.txt -r requirements-dev.txt && pytest` from
this directory — a separate venv from the bot's, since this service keeps
`yt-dlp` after the bot drops it.
