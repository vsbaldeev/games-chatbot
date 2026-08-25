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

# Cross-provider fallback for the vision model, used when Groq's daily quota
# for VISION_MODEL is exhausted or the API is unreachable (see
# src.agent.vision.make_vision_llm). Closest size match on OpenRouter, and an
# "instruct" (non-reasoning) variant, so no reasoning_effort workaround is
# needed on this leg. Requires OPENROUTER_API_KEY; without it vision calls
# stay Groq-only. Same OPENROUTER_BASE_URL as the text filter's fallback.
VISION_FALLBACK_MODEL = "qwen/qwen3-vl-32b-instruct"

# Meaningless-message filter. Was llama-3.1-8b-instant for its 14.4K RPD, but
# measured against 30 days of this chat's real addressed messages the 8B model
# answered 3/8 of the drops it should not have made — it labelled plain
# questions MEANINGLESS despite the prompt's rule 4 forbidding exactly that,
# and each mislabel costs a member an emoji instead of an answer. The 70B model
# scored 8/8 on the same set. Volume makes the RPD argument moot: this chat
# sees ~11 addressed messages a day, nowhere near the smaller 1K RPD budget.
#
# llama-3.3-70b-versatile was decommissioned by Groq on 2026-08-16 with no
# same-family replacement on any free tier (Groq's own migration notice
# points at GPT-OSS/Qwen, not another Llama size). qwen/qwen3.6-27b is the
# replacement — the same reasoning-model family already proven in this
# codebase under a tight token budget via reasoning_effort="none" (see
# VISION_MODEL, MEME_JUDGE_MAX_TOKENS=50, and make_filter_llm below), which is
# why it was picked over openai/gpt-oss-120b: gpt-oss has no proven
# "no thinking" mode here, and FILTER_MAX_TOKENS=10 leaves no room to find out.
# Its classification accuracy on this chat's real messages is UNVALIDATED —
# the 8/8-vs-3/8 comparison above no longer reflects the model in use, and it
# now shares a Groq daily quota bucket with VISION_MODEL,
# MEMORY_MODEL_FALLBACKS[0], and TAG_MODEL (see below), undoing the original
# point of picking a different model family for this call site. Re-split if
# quota exhaustion starts correlating across them.
FILTER_MODEL = "qwen/qwen3.6-27b"

# Cross-provider fallback for the filter, used when Groq is out of quota or
# unreachable (see filter_node.make_filter_llm). Same weights, different
# vendor, so a Groq outage degrades to a paid call instead of to silence.
# Requires OPENROUTER_API_KEY; without it the filter is Groq-only. Unaffected
# by the Groq decommission — this is an OpenRouter-hosted model, not Groq's.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
FILTER_FALLBACK_MODEL = "meta-llama/llama-3.3-70b-instruct"

# Second opinion before acting on an overheard bot-word insult. The comeback
# payload is aggressive, so the cheap filter's positives are confirmed by a
# stronger model before the bot claps back.
# llama-3.3-70b-versatile decommissioned 2026-08-16 — see FILTER_MODEL.
# NOTE: still identical to FILTER_MODEL, which makes the confirmation a
# same-model re-ask at temperature 0 — it will nearly always agree, so the
# overheard gate is effectively open. Needs a decision: point this at a
# genuinely different model (openai/gpt-oss-120b) or drop the second call.
INSULT_CONFIRM_MODEL = "qwen/qwen3.6-27b"

# Memory fact extraction fallback chain (chat facts and posted-episode
# canon-fact distillation both use this — see memory_writer.make_extraction_llm).
# Primary is a reasoning model: callers must pass reasoning_effort="none" or
# the whole max_tokens budget is burned inside a <think> block and no JSON is
# produced. openai/gpt-oss-20b is the fallback for Groq daily-quota (TPD)
# exhaustion on the primary — a different model family, so its own quota
# bucket, and it's Groq's own recommended replacement for the model this slot
# used before. (Was llama-3.1-8b-instant, decommissioned by Groq 2026-08-16.)
MEMORY_MODEL_FALLBACKS: list[str] = [
    "qwen/qwen3.6-27b",     # primary
    "openai/gpt-oss-20b",   # fallback: separate, larger daily quota
]

# Weekly member-role assignment and on-request group profiling (roles.py,
# group_profile/generate.py). Was llama-3.3-70b-versatile, decommissioned by
# Groq on 2026-08-16 with no free-tier same-family replacement; the
# immediate stand-in was openai/gpt-oss-120b (a reasoning model), which then
# needed its own truncation fix (reasoning tokens eating into max_tokens,
# see git history) before it could even produce complete JSON reliably.
# qwen/qwen3.6-27b replaced it after a side-by-side on this call's actual
# shape (8 members x 4 rubrics): zero reasoning tokens on every call versus
# gpt-oss's "low" spiking to 289-339 on some calls (same truncation risk,
# just less often), no JSON errors (gpt-oss produced one: a "veredict" key
# typo that silently drops a member), and better rubric adherence and
# Russian-localized output in that same comparison. Called with
# reasoning_effort="none", which Groq accepts for this model (unlike
# gpt-oss-120b, which 400s on "none" — only low/medium/high). Shares this
# Groq quota bucket with VISION_MODEL, FILTER_MODEL and
# MEMORY_MODEL_FALLBACKS[0]; accepted given this call site's low volume
# (weekly roles, cooldown-gated group-profile requests) but unmeasured — see
# FILTER_MODEL's note above on the same bucket getting crowded.
TAG_MODEL = "qwen/qwen3.6-27b"

# Tool-calling worker fallback chain. No 8B floor: at that size the worker
# skips tools and fabricates facts from parametric memory — for a
# fact-gatherer, no data beats fake data; exhaustion raises an honest
# quota error instead. All three legs are Groq — unaffected by the Llama
# decommission (none of them are Llama) but still a single-provider chain,
# same gap as RESPONSE/ROAST had. Left Groq-only deliberately for now: an
# OpenRouter free-tier fallback here needs its tool-calling reliability
# checked first, or a weak fallback reintroduces the exact fabrication risk
# this comment warns against.
WORKER_MODEL_FALLBACKS: list[str] = [
    "openai/gpt-oss-120b",   # primary:    120B, best tool-call quality
    "qwen/qwen3.6-27b",      # fallback-1: 27B,  parallel tools
    "openai/gpt-oss-20b",    # fallback-2: 20B,  structured tool caller
]

# Personality / response fallback chain.
# Meta/llama was the only family that held the Russian casual style — qwen
# and gpt-oss both drifted from it in earlier testing. Groq decommissioned
# every Llama chat model on 2026-08-16, and no provider offers Llama for free
# anymore (checked OpenRouter's live free-tier catalog 2026-08-17: no Llama,
# Qwen or DeepSeek), so that constraint can no longer be met without paying
# per call. openai/gpt-oss-120b is the new Groq-side primary — UNVALIDATED
# for Russian casual style, re-check output quality against the old Llama
# voice. This chain also gained a cross-provider leg it never had before
# (RESPONSE_OPENROUTER_FALLBACKS, below) — until now a Groq-only outage took chat
# replies down entirely (see ResponseAgent.__build_executor).
RESPONSE_MODEL_FALLBACKS: list[str] = [
    "openai/gpt-oss-120b",
]

# Cross-provider (OpenRouter) fallback chain for the response chain, tried in
# order after the Groq legs. Gemma was picked for Russian/Cyrillic quality
# among currently-free OpenRouter models now that Llama/Qwen/DeepSeek are gone
# from that tier too, but it's a single shared free pool (Google AI Studio)
# that saturates under load — this was hitting 429s often enough in practice
# to need a second, differently-sourced leg. GLM added 2026-08-25 as that
# second leg (Zhipu/Z.ai backend, so an independent quota pool from Gemma's);
# its Russian casual-voice quality is UNVALIDATED, re-check same as the Groq
# gpt-oss-120b primary. Requires OPENROUTER_API_KEY; without it ResponseAgent
# stays Groq-only, same fail-open contract as make_filter_llm.
RESPONSE_OPENROUTER_FALLBACKS: list[str] = [
    "google/gemma-4-31b-it:free",
    "z-ai/glm-5.2:free",
]

# Roast generation fallback chain. Middle leg was llama-3.3-70b-versatile,
# decommissioned by Groq 2026-08-16 with no free-tier same-family
# replacement — dropped rather than replaced, since the chain already had
# two working Groq models either side of it.
ROAST_MODEL_FALLBACKS: list[str] = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

# Cross-provider (OpenRouter) fallback chain for the roast chain — see
# RESPONSE_OPENROUTER_FALLBACKS, same models and same reasoning.
ROAST_OPENROUTER_FALLBACKS: list[str] = [
    "google/gemma-4-31b-it:free",
    "z-ai/glm-5.2:free",
]

# Self-hosted image generation (imagegen-service/, SD1.5 on CPU, DPM++ 2M
# Karras). Standard multi-step sampling, not an LCM speed hack: low-step/
# low-guidance sampling reliably hallucinated compositions. One 512px image
# is estimated at ~5-10 min on the 4 vCPU host (confirm at deploy), so the
# client polls the async job API instead of holding a request open.
IMAGEGEN_STEPS = 20
IMAGEGEN_SIZE = 512
IMAGEGEN_GUIDANCE = 6.0
IMAGEGEN_POLL_SECONDS = 10
IMAGEGEN_DEADLINE_SECONDS = 1200

# Best-of-N photo selection: SD1.5 renders subject interactions
# stochastically, so up to N candidates are generated per photo post and a
# vision-LLM judge (VISION_MODEL) scores each against the episode's
# image_prompt (0-10, interaction-weighted). The first candidate scoring
# >= PHOTO_JUDGE_PASS_SCORE ships immediately; otherwise the best one does —
# the judge is a ranker, not a gate.
IMAGEGEN_CANDIDATES = 3
PHOTO_JUDGE_PASS_SCORE = 7
PHOTO_JUDGE_MAX_TOKENS = 150

# Meme vetting gate (src/memes/judge.py). Scraped Telegram channels serve the
# channel author's own posts — donation appeals, ads, "I'm back" announcements —
# in markup identical to a meme post, so the source parser cannot tell them
# apart and a vision judge scores the image itself before it is sent.
# The same score also covers standalone-ness: captions are never reposted, so a
# photo whose joke lives in the source channel's caption must not ship either.
# ATTEMPTS bounds the worst case at 3 downloads + 3 vision calls per send; a
# judge outage aborts the loop early rather than spending the remaining budget
# on a judge that is known to be down (see fetcher.get_meme).
MEME_JUDGE_MAX_TOKENS = 50  # the verdict is just {"score": N}
MEME_JUDGE_PASS_SCORE = 7
MEME_JUDGE_ATTEMPTS = 3

# Chat-requested selfie scene writer (src/life/selfie.py). One small call
# turning a member's Russian photo request into an English scene line. No
# fallback chain: a failure degrades to a canned in-character excuse.
# Was llama-3.3-70b-versatile (decommissioned by Groq 2026-08-16, no
# free-tier Llama replacement anywhere); openai/gpt-oss-120b is the
# replacement. It's a reasoning model with no proven "no thinking" mode in
# this codebase, so SELFIE_SCENE_MAX_TOKENS is raised to the same headroom
# ROAST_MODEL_FALLBACKS' gpt-oss-120b primary needs, rather than risk the
# whole budget disappearing into a hidden <think> block before strip_thinking
# ever sees an answer.
SELFIE_SCENE_MODEL = "openai/gpt-oss-120b"
SELFIE_SCENE_MAX_TOKENS = 1024

# Caption compressor (src/agent/compress.py). One small call that rewrites an
# over-budget video caption shorter without dropping meaning — cutting mid-
# sentence is the thing this exists to avoid. Was llama-3.3-70b-versatile
# (decommissioned by Groq 2026-08-16); openai/gpt-oss-120b is the
# replacement, same reasoning-headroom caveat as SELFIE_SCENE_MODEL above.
# No fallback chain: a failure degrades to the deterministic sentence-
# boundary truncation in src/events/link_repost.py.
CAPTION_COMPRESS_MODEL = "openai/gpt-oss-120b"
CAPTION_COMPRESS_MAX_TOKENS = 1024

# Text-to-speech — Silero v5 Russian, runs locally on CPU (no API quota).
# Chosen for automatic stress placement and homograph resolution: wrongly
# stressed words are the loudest tell of synthetic Russian speech.
TTS_MODEL_URL = "https://models.silero.ai/models/tts/ru/v5_ru.pt"
TTS_SPEAKER = "aidar"  # male; alternative male voice: "eugene"
TTS_SAMPLE_RATE = 48000  # Silero supports 8000/24000/48000; Opus is native at 48k
TTS_MAX_CHARS = 800  # Silero rejects ~1000+ chars per call; longer replies stay text
TTS_TORCH_THREADS = 4  # Silero plateaus past 4 threads; leaves CPU for the event loop
TTS_TIMEOUT_SECONDS = 30
