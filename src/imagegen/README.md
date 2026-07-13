HTTP client for the self-hosted image-generation service (`imagegen-service/`),
used by Жора's photo life posts (`src/life/poster.py`).

## Contract

`generate_image(prompt) -> bytes | None` — PNG bytes on success, `None` on any
failure. Never raises (the TTS `speech_service` pattern): a media failure can
only demote a photo post to a text story, never kill it.

- Disabled entirely when `IMAGEGEN_URL` is empty (the `TMDB_API_KEY`
  optional-service pattern) — callers should then not even offer the photo
  format.
- Generation takes ~1.5–3 min on the 4 vCPU CPU host, so the service exposes
  an async job API: `POST /generations` (one retry on connect error, 10 s
  request timeout) then `GET /generations/{id}` every `IMAGEGEN_POLL_SECONDS`
  (10) until `done`/`failed`/404 or `IMAGEGEN_DEADLINE_SECONDS` (900).
- A transport error during polling aborts the job client-side (returns
  `None`): the service sits on the same Docker network, so a flaky poll means
  the service itself is in trouble — degrading beats hanging.

Generation parameters (`IMAGEGEN_STEPS = 6`, `IMAGEGEN_SIZE = 512`,
`IMAGEGEN_GUIDANCE = 1.5`) live in `src/config/models.py`; the URL comes from
the `IMAGEGEN_URL` env (`src/config/credentials.py`).

See `imagegen-service/README.md` for the service side: engine choice
rationale, RAM budget, and API details.
