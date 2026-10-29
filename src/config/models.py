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

# Vision: image and video-frame description (see src.llm.vision.make_vision_llm).
#
# Primary is OpenRouter's qwen3.7-plus (paid, so no daily quota to run out of).
# It replaced qwen/qwen3-vl-32b-instruct, which OpenRouter deprecated on
# 2026-10-09 (404 on every call). Reasoning model: make_vision_llm sends
# OpenRouter's reasoning-off field, or the whole max_tokens budget is burned
# thinking and the description comes back empty (measured: 200 reasoning
# tokens, 0 content). Requires OPENROUTER_API_KEY. Same OPENROUTER_BASE_URL
# as the text filter's fallback.
VISION_PRIMARY_MODEL = "qwen/qwen3.7-plus"

# Reserve, used when OpenRouter fails or is unreachable, and the only vision
# model when OPENROUTER_API_KEY is unset. The only non-deprecated
# vision-capable model on Groq's free tier. Made the reserve after one probe
# on a single red square returned "three red squares" (OpenRouter's answer
# was right) — one sample, not a measured error rate. Reasoning model:
# callers must pass reasoning_effort="none" or the whole max_tokens budget is
# burned inside a <think> block.
VISION_RESERVE_MODEL = "qwen/qwen3.8-27b"

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
# points at GPT-OSS/Qwen, not another Llama size). qwen/qwen3.8-27b is the
# replacement — the same reasoning-model family already proven in this
# codebase under a tight token budget via reasoning_effort="none" (see
# VISION_RESERVE_MODEL, MEME_JUDGE_MAX_TOKENS=50, and make_filter_llm below), which is
# why it was picked over openai/gpt-oss-120b: gpt-oss has no proven
# "no thinking" mode here, and FILTER_MAX_TOKENS=10 leaves no room to find out.
# Its classification accuracy on this chat's real messages is UNVALIDATED —
# the 8/8-vs-3/8 comparison above no longer reflects the model in use, and it
# now shares a Groq daily quota bucket with VISION_RESERVE_MODEL,
# MEMORY_MODEL_FALLBACKS[0], and TAG_MODEL (see below), undoing the original
# point of picking a different model family for this call site. Re-split if
# quota exhaustion starts correlating across them.
FILTER_MODEL = "qwen/qwen3.8-27b"

# Cross-provider fallback for the filter, used when Groq is out of quota or
# unreachable (see filter.classify.make_filter_llm). Same weights, different
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
INSULT_CONFIRM_MODEL = "qwen/qwen3.8-27b"

# Memory fact extraction fallback chain (chat facts and posted-episode
# canon-fact distillation both use this — see memory_writer.make_extraction_llm).
# Primary is a reasoning model: callers must pass reasoning_effort="none" or
# the whole max_tokens budget is burned inside a <think> block and no JSON is
# produced. openai/gpt-oss-20b is the fallback for Groq daily-quota (TPD)
# exhaustion on the primary — a different model family, so its own quota
# bucket, and it's Groq's own recommended replacement for the model this slot
# used before. (Was llama-3.1-8b-instant, decommissioned by Groq 2026-08-16.)
MEMORY_MODEL_FALLBACKS: list[str] = [
    "qwen/qwen3.8-27b",     # primary
    "openai/gpt-oss-20b",   # fallback: separate, larger daily quota
]

# Weekly member-role assignment and on-request group profiling (roles.py,
# group_profile/generate.py). Was llama-3.3-70b-versatile, decommissioned by
# Groq on 2026-08-16 with no free-tier same-family replacement; the
# immediate stand-in was openai/gpt-oss-120b (a reasoning model), which then
# needed its own truncation fix (reasoning tokens eating into max_tokens,
# see git history) before it could even produce complete JSON reliably.
# qwen/qwen3.8-27b replaced it after a side-by-side on this call's actual
# shape (8 members x 4 rubrics): zero reasoning tokens on every call versus
# gpt-oss's "low" spiking to 289-339 on some calls (same truncation risk,
# just less often), no JSON errors (gpt-oss produced one: a "veredict" key
# typo that silently drops a member), and better rubric adherence and
# Russian-localized output in that same comparison. Called with
# reasoning_effort="none", which Groq accepts for this model (unlike
# gpt-oss-120b, which 400s on "none" — only low/medium/high). Shares this
# Groq quota bucket with VISION_RESERVE_MODEL, FILTER_MODEL and
# MEMORY_MODEL_FALLBACKS[0]; accepted given this call site's low volume
# (weekly roles, cooldown-gated group-profile requests) but unmeasured — see
# FILTER_MODEL's note above on the same bucket getting crowded.
TAG_MODEL = "qwen/qwen3.8-27b"

# Reply reaction labelling (src/feedback/reply_labeller.py). OpenRouter-only,
# deliberately: it must never share TAG_MODEL's already-crowded Groq bucket
# (see TAG_MODEL above) — measurement must never degrade chat replies. The call
# is made once per reply, in a background task, with no retry: a 429 or an
# outage permanently costs that reply its label (it is counted in
# unlabelled_replies), so a saturated free pool shows up there rather than being
# retried later. Two legs on independent backends (NVIDIA and Google AI Studio)
# so saturation in one pool does not take out the other — checked live on 2026-10-12, when the
# whole Google AI Studio free pool was returning 429 while NVIDIA answered
# normally, which is exactly the failure this split exists for.
#
# The primary was "z-ai/glm-5.2:free" until 2026-10-12. That slug does not
# exist: OpenRouter's /api/v1/models carries only the paid "z-ai/glm-5.2", and
# a call to the :free variant 404s with "This model is unavailable for free",
# so the chain silently ran on the Gemma leg alone.
#
# nvidia/nemotron-3-super-120b-a12b:free replaced it after checking the live
# free catalog: 120B hybrid MoE with 12B active, text-only, no expiry date, and
# the only non-Google free leg that is both general-purpose (not a coding or
# guardrail model) and reachable from a plain API key — the Thinking Machines
# and Poolside free legs are gated to "agentic harnesses" (403) or expire
# within weeks. Verified on this call's real shape (two Russian replies, JSON
# array out): correct labels, valid JSON, zero reasoning tokens.
#
# It is a reasoning model with reasoning ON by default, so callers must pass
# reasoning_effort="none" — the same idiom FILTER_MODEL, VISION_RESERVE_MODEL and
# TAG_MODEL already carry here. Measured on the same probe: "none" yields 0
# reasoning tokens, while "low" burned 293 and truncated the JSON mid-array.
REPLY_LABEL_MODELS: list[str] = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
]

# Budget for the single {"reaction": "NEGATIVE"} object, ~10 tokens even
# pretty-printed. 64 absorbs a ```json fence some free-tier models add.
# Reasoning tokens are not budgeted for: make_labeller_llm passes
# reasoning_effort="none" instead of paying for a <think> block.
REPLY_LABEL_MAX_TOKENS = 64

# Response labelling (src/message_content/response_labeller.py) reuses
# REPLY_LABEL_MODELS — same OpenRouter-only reasoning as above: measurement must
# never take Groq quota from chat replies. Same budget too: one {"label": "..."}
# object, with headroom for a code fence.
RESPONSE_LABEL_MAX_TOKENS = 64

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
    "qwen/qwen3.8-27b",      # fallback-1: 27B,  parallel tools
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
# gpt-oss-120b primary. Both free legs share the failure mode that started
# this chain (a shared free pool saturating under load), so the chain ends
# in a paid floor: qwen3-235b-a22b-2507, added 2026-08-31 after the Groq
# gpt-oss-120b primary hit its 8000 TPM cap and the free legs alone weren't
# a guaranteed out. Its Russian casual-voice quality is also UNVALIDATED —
# picked over re-adding llama-3.3-70b-instruct (this repo's one proven RU
# pick, used as FILTER_FALLBACK_MODEL) to actually test the newer, larger
# model instead of falling back to the known-safe choice again; revisit if
# output quality disappoints. Requires OPENROUTER_API_KEY; without it
# ResponseAgent stays Groq-only, same fail-open contract as make_filter_llm.
RESPONSE_OPENROUTER_FALLBACKS: list[str] = [
    "google/gemma-4-31b-it:free",
    "z-ai/glm-5.2:free",
    "qwen/qwen3-235b-a22b-2507",
]

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

# Caption compressor (src/llm/compress.py). One small call that rewrites an
# over-budget video caption shorter without dropping meaning — cutting mid-
# sentence is the thing this exists to avoid. Was llama-3.3-70b-versatile
# (decommissioned by Groq 2026-08-16); openai/gpt-oss-120b is the
# replacement, same reasoning-model caveat: the token budget needs headroom for a hidden <think> block.
# No fallback chain: a failure degrades to the deterministic sentence-
# boundary truncation in src/events/link_repost.py.
CAPTION_COMPRESS_MODEL = "openai/gpt-oss-120b"
CAPTION_COMPRESS_MAX_TOKENS = 1024
