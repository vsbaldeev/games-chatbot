Жора's self-image — chat-requested selfies and the image pipeline behind them.

## What this package is (and what it used to be)

Until 2026-08-10 this package also drove **scheduled life posts**: three
unprompted posts a week narrating episodes from Жора's invented village life,
backed by an episode writer, a canon store (`bot_memories`), and a silent daily
"current activity" refresh.

That whole feature was retired. Chat feedback was consistent and came from
several members independently: the posts read as noise («с огородом это треш»,
«когда он какую-то смородину собирает и рандомно об этом рассказывает»), a
member imitated the format to mock it, and the same biography leaked into
ordinary replies — 1 reply in 10 volunteered his current activity unprompted.
A chat poll put only ~39% actively in favour, 28% against and ~33% neutral.

The decision was **keep the voice, drop the biography**: Жора is still a
village handyman in manner — dry, unimpressed, sarcastic — but he has no diary,
tells no stories about his day, and never volunteers what he is doing. The
`CHARACTER_SHEET` in `src/config/prompts.py` still defines who he is; nothing
generates episodes about him any more.

Retired with it: `src/life/writer.py`, `src/life/poster.py`,
`src/life/activity.py`, `src/jobs/life_post.py`, `src/jobs/daily_activity.py`,
`src/store/bot_memories.py`, and the bot-canon/activity blocks of the reply
prompt. The `bot_memories` **table** is deliberately left in place — no
migration drops it, so the data survives if the feature is ever revived.

What remains here is the part members actually liked: they can ask Жора for a
photo, and he sends one.

## Chat-requested selfies — `src/life/selfie.py`

Members ask Жора for a photo of himself in chat («сфоткай себя», «покажи свой
огород»). The pipeline's filter classifies the request (`PHOTO_REQUEST`
verdict, the heaviest engagement-budget signal at 4.5) and the reply is an
immediate in-character «ща сфоткаю» ack; `deliver_selfie` then runs as a
fire-and-forget background task launched by the events layer:

1. `write_selfie_scene` (`SELFIE_SCENE_MODEL`, one call, no fallback chain)
   turns the Russian request into one English scene line. A named scene is
   rendered as asked; a bare «сфоткай себя» falls back to a village-yard shot.
2. `generate_best_photo` (`src/life/photo.py`) renders and judges the
   candidates; the photo goes out as a reply to the requesting message with a
   short canned caption and is recorded with `format_photo_content` +
   `file_id`, plugging it into the lazy vision-description path like every
   other photo.
3. Any failure degrades to a canned in-character excuse
   (`SELFIE_FAILED_REPLIES`) — after promising a photo, silence is not an
   option.

One module-global generation slot serializes selfies: the imagegen service has
a single worker, so a request arriving mid-render gets an «уже фоткаю» ack and
no second job. The filter's peek is `selfie.image_generation_in_flight()`. The
peek and the acquire are separate moments, so two overlapping pipelines can
both ack while only one generates — the loser logs and exits; rare and
low-stakes by design.

(Before the life posts were retired this peek also had to cover a scheduled
photo post, which held the worker for ~16 minutes. Selfies are now the only
flow that renders images, so the slot is this module's own.)

## Best-of-N photo selection — `src/life/photo.py`, `src/life/photo_judge.py`

SD1.5 renders subject *interactions* stochastically — all subjects present,
nobody doing what was asked. `generate_best_photo` generates up to
`IMAGEGEN_CANDIDATES` (3) candidates with random seeds and
`src/life/photo_judge.py` scores each against the requested scene via the Groq
vision model (0–10, interaction weighted heaviest, the same multimodal pattern
as `src/pipeline/ingester.py`). The first candidate scoring ≥
`PHOTO_JUDGE_PASS_SCORE` (7) ships immediately (early exit saves CPU-minutes);
otherwise the best-scoring one ships — **the judge ranks, it never gates**. A
judge outage scores as "unknown" (`UNSCORED_RANK`, below any scored candidate
but still shippable): a broken judge must never block a photo that rendered
fine.

Prompt shape matters as much as the judge. Across seed sweeps,
two-subject-plus-interaction prompts failed on 3 of 4 seeds (the second
creature simply doesn't render), while a single-subject action frame scored
9/10 within three candidates. Hence the scene contract: one subject-verb-object
action, at most one other creature, no prop lists.

The character's appearance is deliberately absent from the scene line: the
fixed `CHARACTER_VISUAL_PROMPT` descriptor is prepended at generation time
(after `PHOTO_FRAMING_HINT`), so every selfie shares wardrobe/beard/style while
the scene tracks the request.

A further layer lives on the service side: `CHARACTER_VISUAL_PROMPT + scene`
routinely exceeds CLIP's 77-token limit, and a plain `prompt=` string would
silently truncate — dropping exactly the requested scene detail.
`imagegen-service/engine.py` builds embeddings via Compel instead, so the full
prompt always reaches the model (see that service's README for the bug this
replaced).
