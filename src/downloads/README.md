HTTP client for the self-hosted download service (`download-service/`), used
by `src/pipeline/ingest/links/shorts.py`, `src/pipeline/ingest/links/social_links/instagram_reel.py`,
`src/pipeline/ingest/links/social_links/tiktok_video.py`, and
`src/pipeline/ingest/links/social_links/youtube_video.py`.

## Contract

- `download_short(url) -> (bytes, dict) | None`
- `download_reel(url) -> (bytes, dict) | None`
- `download_tiktok(url) -> (bytes, dict) | None`
- `fetch_youtube_video(url) -> dict | None`

Never raises: a download failure degrades the caller to skipping the summary,
never an unhandled error. Disabled entirely when `DOWNLOAD_SERVICE_URL` is
empty.

Downloads finish in seconds, so this is a single `POST /download` per call
— no job/poll machinery. Per-kind request
timeouts (`SHORTS_TIMEOUT_SECONDS=90`, `REEL_TIMEOUT_SECONDS=60`,
`TIKTOK_TIMEOUT_SECONDS=60`, `YOUTUBE_VIDEO_TIMEOUT_SECONDS=30`) mirror the
timeouts each caller used to enforce itself before this client existed
(TikTok's copies the Reel timeout).

All yt-dlp mechanics (PO-token wiring, JS challenge solving, player-client
pinning, retry signals) live in `download-service/`, not here or in the
bot — see `download-service/README.md`.
