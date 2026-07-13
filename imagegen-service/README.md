Self-hosted CPU image generation for Жора's photo life posts. A separate
deploy unit (own container, own 3 GB memory limit) so the ML workload can
never touch the bot's memory envelope; the bot talks to it through
`src/imagegen/client.py` and degrades photo posts to text stories whenever
this service is down.

## Engine: diffusers + SD1.5 (DreamShaper 8) + LCM-LoRA, CPU

Chosen over stable-diffusion.cpp: every layer is inspectable Python
(schedulers, pipelines, LoRA fusion — the canonical stack, and the stated
learning goal), and the Step 7 character LoRA loads natively via
`pipe.load_lora_weights()`. The speed problem is solved by **step
reduction**, not per-step speed: the LCM-LoRA
(`latent-consistency/lcm-lora-sdv1-5`, loaded as the named adapter `lcm` so
Step 7 can add the character LoRA next to it) collapses 25 diffusion steps
to 4–6 at `guidance_scale` ~1.0–1.5, giving ~1.5–3 min per 512² image on
4 vCPU. Base checkpoint `Lykon/dreamshaper-8`: an SD1.5-class stylized-
illustration model — the storybook style deliberately hides CPU-model
artifacts and makes pre-LoRA face drift between selfies tolerable.

Fallbacks if the budget ever tightens: stable-diffusion.cpp (q4-q8 GGUF,
~1.3–2.5 GB peak, but a black-box C++ binary with awkward LoRA handling);
OpenVINO via optimum-intel as an optional 2–3× CPU speedup later (its model
conversion step complicates LoRA iteration, so it is not the baseline).

## Long prompts: Compel instead of the raw 77-token CLIP cap

`CHARACTER_VISUAL_PROMPT + episode.image_prompt` routinely exceeds CLIP's
77-token limit. Passing that combined string as `prompt=` would silently
**truncate the tail** — exactly the episode's scene detail, breaking the
text-photo coherence guarantee (a selfie could render only the character
descriptor and drop the actual scene). `engine.py` builds
`prompt_embeds`/`negative_prompt_embeds` with
[Compel](https://github.com/damian0815/compel) (`truncate_long_prompts=False`)
instead: it chunks the prompt into 77-token windows and concatenates their
embeddings, so nothing is dropped regardless of length. Since `image_prompt`
is LLM-generated text, not something a person hand-typed for Compel's
prompt-weighting syntax, `escape_compel_syntax` backslash-escapes any stray
`(`, `)` or `"` first so they're treated as literal characters instead of
grouping/weighting syntax.

Verified locally (MPS, `Lykon/dreamshaper-8`): before the fix, a combined
85-token prompt truncated at 77 and dropped the requested scene (a fence
repair with a cow) entirely — the render showed a generic portrait. After
the fix, the full prompt conditions the model and the fence appears in
frame. Not a complete fix for adherence, though: LCM at 4–6 steps and
`guidance_scale` ~1.5 limits how many discrete objects a single frame
reliably renders (the cow and scattered tools still didn't appear) — a
known trade-off of the step-reduction speed choice, unrelated to and not
fixed by the token-cap change.

## Memory lifecycle

- **Lazy load**: the pipeline loads on the first job; a cold container
  idles at ~300 MB.
- **Idle unload**: a watchdog frees the pipeline after 15 min without work,
  so the ~3 GB peak exists only around the two weekly generations.
- **float32, not bfloat16**: bf16 halves the weight footprint but silently
  produces black frames on CPUs without native bf16 support; float32 is
  universally correct and fits the 3 GB limit.
- One generation at a time (`asyncio.Semaphore(1)`); the diffusion run
  executes on a worker thread so `/healthz` stays responsive.
- Weights download to `/models` (`HF_HOME`, a compose volume) on first
  start — not baked into the image.

## API

Async job pattern — holding one HTTP request open for minutes is fragile,
and polling is the more instructive design:

```
GET  /healthz            → 200 {"status": "ok", "model_loaded": bool}
POST /generations        → 202 {"generation_id": "..."}
     body: {"prompt": str, "negative_prompt": str?, "width": 512,
            "height": 512, "steps": 6, "guidance_scale": 1.5, "seed": int?}
GET  /generations/{id}   → 200 {"status": "queued"|"running"|"done"|"failed",
                                "image_png_base64": str?, "error": str?}
                           404 unknown/expired id (jobs expire after 1 h)
```

Note: the job module is `registry.py`, not the originally planned
`queue.py` — a top-level `queue.py` would shadow the stdlib module torch
imports internally.

## The avatar

Жора's canonical face is a hero portrait generated on this service from
`CHARACTER_VISUAL_PROMPT` (`src/config/prompts.py`) with a fixed seed. Bots
cannot set their own profile picture through the API — set it manually via
BotFather. Record the winning seed here once picked:

- **Avatar seed**: _not picked yet_

## Local smoke test

```bash
docker compose up --build imagegen
curl localhost:8000/healthz
curl -X POST localhost:8000/generations -H 'Content-Type: application/json' \
  -d '{"prompt": "zhora, rugged slavic village man, storybook illustration", "seed": 42}'
curl localhost:8000/generations/<id>   # poll until "done"
```

Expect 1.5–3 min wall time per image and peak RSS under 3 GB
(`docker stats`).
