HTTP client for the self-hosted image-generation service (`imagegen-service/`),
used by Жора's Monday photo life posts (`src/life/poster.py`) and chat-requested
selfies (`src/life/selfie.py`).

## Contract

`generate_image(prompt) -> bytes | None` — PNG bytes on success, `None` on any
failure. Never raises (the TTS `speech_service` pattern): a media failure can
only demote a photo post to a text story, never kill it.

- Disabled entirely when `IMAGEGEN_URL` is empty (the `TMDB_API_KEY`
  optional-service pattern) — the scheduled photo slot is then written as a
  text story up front (`poster.supported_format`).
- Generation takes ~3 min on the 4 vCPU CPU host, so the service exposes an
  async job API: `POST /generations` (10 s timeout) then
  `GET /generations/{id}` every `IMAGEGEN_POLL_SECONDS` (10, with a 30 s
  `POLL_TIMEOUT_SECONDS` per read) until `done`/`failed`/404 or
  `IMAGEGEN_DEADLINE_SECONDS` (1200).

## Patient polling, impatient submitting

The two halves of the job API retry on opposite principles, and the
asymmetry is deliberate:

- **Polling retries almost everything.** A status read is free and
  repeatable, while the job behind it represents ~3 min of CPU. Read
  timeouts, dropped connections and 5xx are logged and retried until the
  deadline (`TransientPollError` / `httpx.TransportError`); only a terminal
  verdict — 404 expired-or-unknown id, or a `failed` job (`GenerationEnded`)
  — stops the wait early. This matters because `TORCH_THREADS` equals the
  host's vCPU count, so a running generation can starve uvicorn's event loop
  and make a perfectly healthy job answer slowly.
- **Submitting retries only connect errors.** A submit that timed out may
  already have created the job server-side, so retrying it would burn a
  second three-minute generation. Only `httpx.ConnectError` — where no
  request can have landed — is retried, once.

An earlier version aborted the whole generation on any transport error
during polling, on the theory that a flaky poll means the service is in
trouble. With 3-minute generations and ~18 polls each that traded a
recoverable blip for a discarded post; the deadline, not the first hiccup,
is what bounds the wait now.

Generation parameters (`IMAGEGEN_STEPS = 20`, `IMAGEGEN_SIZE = 512`,
`IMAGEGEN_GUIDANCE = 6.0`, `IMAGEGEN_CANDIDATES = 3`) live in
`src/config/models.py`; the URL comes from the `IMAGEGEN_URL` env
(`src/config/credentials.py`). Every generation task — scheduled Monday
photo post and chat-requested selfie alike — renders
`IMAGEGEN_CANDIDATES` candidates and ships the best (`poster.generate_best_photo`).

See `imagegen-service/README.md` for the service side: engine choice
rationale, RAM budget, and API details.
