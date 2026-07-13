Self-hosted CPU image generation for Жора's photo life posts. A separate
deploy unit (own container, own 3 GB memory limit) so the ML workload can
never touch the bot's memory envelope; the bot talks to it through
`src/imagegen/client.py` and degrades photo posts to text stories whenever
this service is down.

## Engine: diffusers + SD1.5 (DreamShaper 8), DPM++ 2M Karras, CPU

Chosen over stable-diffusion.cpp: every layer is inspectable Python
(schedulers, pipelines, LoRA loading — the canonical stack, and the stated
learning goal), and the Step 7 character LoRA loads natively via
`pipe.load_lora_weights()`. Base checkpoint `Lykon/dreamshaper-8`: an
SD1.5-class stylized-illustration model — the storybook style deliberately
hides CPU-model artifacts and makes pre-LoRA face drift between selfies
tolerable.

**Revision — the LCM-LoRA speed hack was dropped after live testing.** The
original design collapsed 25 diffusion steps to 4–6 via
`latent-consistency/lcm-lora-sdv1-5` at `guidance_scale` ~1.5 (~1.5–3 min
per image on 4 vCPU). In practice that config reliably hallucinated
compositions: subjects merged, went missing, or stood around ignoring the
action the prompt described. Side-by-side at the same seed (MPS, local),
20-step DPM++ 2M Karras @ CFG 6 rendered the multi-subject test scene
coherently, and 28 steps @ CFG 7 added nothing visible over 20. Current
config: `DPMSolverMultistepScheduler` (`use_karras_sigmas=True`,
`algorithm_type="dpmsolver++"`), 20 steps, `guidance_scale` 6 — estimated
~5–10 min per 512² image on the 4 vCPU host (confirm at deploy; posts are
scheduled twice a week, so minutes-per-image is an accepted cost, and the
client deadline is 1200 s per generation).

Adherence techniques researched and rejected for this host: ELLA (T5-XL
adapter, biggest SD1.5 prompt-following gain, but +2.6 GB weights / ~2-3 GB
RAM — blows the 3 GB budget and changes the conditioning the Step 7 LoRA
trains against), SDXL/SD3 (RAM), regional prompting / GLIGEN (layout
planners, research-grade complexity), Attend-and-Excite (improves subject
*presence*, which prompt ordering already fixed — not interactions).
What shipped instead, bot-side: best-of-N candidate generation ranked by a
vision-LLM judge (`src/life/photo_judge.py`).

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
frame, but the cow still didn't — see the next section for why.

## Prompt order: scene before character, not after

Even with the full prompt reaching the model, `CHARACTER_VISUAL_PROMPT,
episode.image_prompt` (character descriptor first) still reliably dropped
secondary scene objects — a test render asking for "a curious cow peeking
from behind the fence" produced only a close-up portrait, no cow, no fence
repair action. Leading tokens dominate composition (observed under the
original LCM config and retained after the switch to full-step sampling): a
leading character descriptor biases the model toward a close-up portrait,
crowding out anything the scene prompt asks for afterward.

Fix verified locally, three prompt variants at the same seed:

| Order | Result |
|---|---|
| character first (original) | close-up portrait; no cow, no fence, no tools |
| `"wide shot, " + scene + character` | wide farm scene; man **and** cow both present |
| `"wide shot, full scene, " + scene + character` + negative prompt against close-ups, higher steps/guidance | cow prominent, but **the man vanished from frame entirely** |

The middle option is what shipped (`PHOTO_FRAMING_HINT = "wide shot, "` in
`src/config/prompts.py`, prepended to `episode.image_prompt` which is then
followed by `CHARACTER_VISUAL_PROMPT` — see `src/life/poster.py`): it is
the only variant that kept the character in frame — the actual point of a
selfie — while also letting the scene's secondary objects render. The more
aggressive tuning (negative prompt + higher guidance/steps) traded away
the character entirely, which is a worse failure than a merely imperfect
scene. Multi-object scenes still won't always render every named object or
interaction — prompt ordering doesn't fully solve that; the full-step
sampling switch (see the engine section) and the bot-side best-of-N judge
loop are the layers that attack it.

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
            "height": 512, "steps": 20, "guidance_scale": 6.0, "seed": int?}
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

Expect roughly 5–10 min wall time per image on the 4 vCPU host (record the
actual number here after the first deploy) and peak RSS under 3 GB
(`docker stats`).
