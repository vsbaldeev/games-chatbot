HTTP client for the self-hosted download service (`download-service/`), used
by `src/pipeline/shorts.py`, `src/pipeline/social_links/instagram_reel.py`,
and `src/pipeline/social_links/youtube_video.py`.

## Contract

- `download_short(url) -> (bytes, dict) | None`
- `download_reel(url) -> (bytes, dict) | None`
- `fetch_youtube_video(url) -> dict | None`

Never raises (the imagegen `generate_image` / TTS `speech_service`
pattern): a download failure degrades the caller to skipping the summary,
never an unhandled error. Disabled entirely when `DOWNLOAD_SERVICE_URL` is
empty (the `IMAGEGEN_URL` optional-service pattern).

Downloads finish in seconds, so this is a single `POST /download` per call
— no job/poll machinery, unlike `src/imagegen/client.py`. Per-kind request
timeouts (`SHORTS_TIMEOUT_SECONDS=90`, `REEL_TIMEOUT_SECONDS=60`,
`YOUTUBE_VIDEO_TIMEOUT_SECONDS=30`) mirror the timeouts each caller used to
enforce itself before this client existed.

All yt-dlp mechanics (PO-token wiring, JS challenge solving, player-client
pinning, retry signals) live in `download-service/`, not here or in the
bot — see `download-service/README.md`.
