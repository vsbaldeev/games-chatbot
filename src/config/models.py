"""LLM model names and fallback chains for all pipeline components.

All model identifiers live here so a Groq deprecation notice means
editing exactly one file.
"""

# Safety — prompt injection guard
GUARD_MODEL = "meta-llama/llama-prompt-guard-2-86m"

# Speech-to-text
WHISPER_MODEL = "whisper-large-v3"

# Transcription language hint (ISO-639-1). The chat is Russian-only; pinning
# the language prevents short notes from flipping into a random language.
# Trade-off: genuinely English voice notes transcribe degraded.
WHISPER_LANGUAGE = "ru"

# Vision: image and video-frame description.
# qwen/qwen3.6-27b is the only non-deprecated vision-capable model on the free tier.
# Reasoning model: callers must pass reasoning_effort="none" or the whole
# max_tokens budget is burned inside a <think> block.
VISION_MODEL = "qwen/qwen3.6-27b"

# Meaningless-message filter (binary yes/no, max_tokens=5).
# llama-3.1-8b-instant has 14.4K RPD vs 1K RPD for larger models.
FILTER_MODEL = "llama-3.1-8b-instant"

# Second opinion before acting on an overheard bot-word insult. The comeback
# payload is aggressive, so the cheap filter's positives are confirmed by a
# stronger model before the bot claps back.
INSULT_CONFIRM_MODEL = "llama-3.3-70b-versatile"

# Memory fact extraction.
# Reasoning model: callers must pass reasoning_effort="none" or the whole
# max_tokens budget is burned inside a <think> block and no JSON is produced.
MEMORY_MODEL = "qwen/qwen3.6-27b"

# Weekly member-role assignment
TAG_MODEL = "llama-3.3-70b-versatile"

# Tool-calling worker fallback chain. No 8B floor: at that size the worker
# skips tools and fabricates facts from parametric memory — for a
# fact-gatherer, no data beats fake data; exhaustion raises an honest
# quota error instead.
WORKER_MODEL_FALLBACKS: list[str] = [
    "openai/gpt-oss-120b",   # primary:    120B, best tool-call quality
    "qwen/qwen3.6-27b",      # fallback-1: 27B,  parallel tools
    "openai/gpt-oss-20b",    # fallback-2: 20B,  structured tool caller
]

# Personality / response fallback chain.
# Meta/llama only — qwen and gpt-oss drift from the Russian casual style.
RESPONSE_MODEL_FALLBACKS: list[str] = [
    "llama-3.3-70b-versatile",  # primary
    "llama-3.1-8b-instant",     # fallback-1: no Meta/llama intermediate on free tier
]

# Autonomous comedian fallback chain
COMEDIAN_MODEL_FALLBACKS: list[str] = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "qwen/qwen3.6-27b",
]

# Roast generation fallback chain
ROAST_MODEL_FALLBACKS: list[str] = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
]

# Life-post episode writer fallback chain. llama first: only Meta/llama holds
# the casual Russian style (see RESPONSE_MODEL_FALLBACKS above).
EPISODE_MODEL_FALLBACKS: list[str] = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
]

# Generous headroom over the ~1100-char JSON contract payload so a post is
# never truncated mid-joke by the model's completion limit.
EPISODE_MAX_TOKENS = 2000

# Text-to-speech — Silero v5 Russian, runs locally on CPU (no API quota).
# Chosen for automatic stress placement and homograph resolution: wrongly
# stressed words are the loudest tell of synthetic Russian speech.
TTS_MODEL_URL = "https://models.silero.ai/models/tts/ru/v5_ru.pt"
TTS_SPEAKER = "aidar"  # male; alternative male voice: "eugene"
TTS_SAMPLE_RATE = 48000  # Silero supports 8000/24000/48000; Opus is native at 48k
TTS_MAX_CHARS = 800  # Silero rejects ~1000+ chars per call; longer replies stay text
TTS_TORCH_THREADS = 4  # Silero plateaus past 4 threads; leaves CPU for the event loop
TTS_TIMEOUT_SECONDS = 30
