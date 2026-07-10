Text-to-speech — local Silero v5 Russian voice synthesis for voice-in-kind replies.

When the pipeline's reply was triggered by a voice message or video note,
`deliver_response` (src/events/messages.py) asks this package to speak the reply
instead of typing it. Everything runs locally on CPU: no API key, no quota.

## When the bot speaks

```
voice / video_note message
    │
    ├─ forwarded                       → no reply (stored for history only)
    ├─ reply to a bot message          → should_respond=True, trigger="explicit"
    │       (@mention needs text — replying is the only way to address with voice)
    ├─ otherwise                       → 10% roll (MEDIA_RESPONSE_CHANCE)
    │       lost roll                  → no reply (transcript still mined for memories)
    ▼
Whisper transcription (Groq whisper-large-v3, ru)
    ├─ empty/garbage + explicit        → canned «не расслышал» reply — short Russian,
    │                                    itself sent as a voice note when TTS is healthy
    ├─ empty/garbage + random          → emoji reaction, no reply
    ▼
filter / guard
    ├─ MEANINGLESS verdict             → emoji reaction (questions/requests never dropped)
    ├─ prompt injection                → blocked, no reply
    ├─ Groq quota / rate limit         → throttled text notice or 😴 reaction
    ▼
deliver_response (src/events/messages.py)
    ├─ «ищу…» search notice was posted → reply edits that message → stays text
    ▼
TTS gate (try_send_voice_reply → src/tts)
    ├─ model not loaded at startup     → text reply
    ├─ unspeakable — no Cyrillic left after cleaning (URLs→«ссылка», emojis stripped)
    │                                  → text reply
    ├─ longer than TTS_MAX_CHARS=800   → text reply
    ├─ synthesis error / 30s timeout / send failure
    │                                  → text reply
    ▼
🎤 voice note — Silero v5 «aidar», 48 kHz OGG/Opus
    stored in unified_messages with media_type="voice", content = reply text,
    so reply chains, thread context and memory keep working on text
```

A reply is never lost to a TTS problem: every gate below `deliver_response`
degrades to the plain text reply, not to silence.

## Modules

```
service.py
    SpeechService            — owns the Silero model lifecycle
        init()               — validates the configured speaker (fail fast), then
                               loads v5_ru.pt off the event loop; a load failure is
                               logged and leaves the service not-ready (bot starts,
                               all replies stay text)
        synthesize(text)     — serialized, thread-offloaded apply_tts + Opus encode
                               with a timeout; any failure returns None
    SynthesizedVoice         — frozen dataclass: ogg_bytes + duration_seconds

text_prep.py
    prepare_tts_text(text)   — URL → «ссылка», strip emojis, collapse whitespace;
                               returns None (stay text) when empty, no Cyrillic, or
                               longer than TTS_MAX_CHARS

encoder.py
    encode_pcm_to_ogg_opus(samples, rate) — float32 PCM → int16 → OGG/Opus bytes
                               via PyAV (libopus), chunked ~1s frames, fully in-memory
```

## Model

Silero TTS v5 (`v5_ru.pt`, ~140 MB) — chosen for automatic stress placement
(ударения) and homograph resolution, the biggest factor in natural-sounding
Russian. Speaker, sample rate, char cap and thread budget live in
`src/config/models.py` (`TTS_*`); the model file path is `TTS_MODEL_PATH`
(downloaded on first start locally, pre-baked in the Docker image). License:
CC-BY-NC — non-commercial use only.
