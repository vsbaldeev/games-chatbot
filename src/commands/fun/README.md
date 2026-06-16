Roast ("прожарка") text generation and the /meme command.

The `/roast` command and the weekly scheduled roast were retired in favour of
autonomous humor (see `src/pipeline/` HumorNode). Roast generation
(`Roaster.generate` / `generate_roast_text`) is kept solely for the **offense
auto-roast**: when a user insults the bot twice in a row, the bot claps back.

## Trigger mode

```
auto-roast        — two consecutive offensive replies to bot → immediate roast of the offender
```

## Generation pipeline

```
1. Target — the offending user (chosen by the caller in src/events/messages.py)

2. Pick mode (random, avoiding the angle used last time on this target)
       The target's last RECENT_MODE_WINDOW = 1 anchor_key(s) are read from roast_log and excluded
       from the random draw, so a back-to-back roast lands a different angle. If exclusion would
       leave nothing, the full set is restored.
       Modes are pure angle hints (ROAST_MODE_INSTRUCTIONS): shame / contradiction. (quirk and boast
       were dropped — they produced absurd output and, lacking a matching fact, made boast fabricate
       a "thinks he's smart" premise.) The chosen mode is returned and logged as the roast's anchor_key.

3. Fact selection — none
       The whole stored fact list is passed to the model intact (get_facts, newest first). There is
       no embedding retrieval and no pre-filtering: the pool is already capped (MAX_FACTS_PER_USER =
       30) and each fact is a short sentence, so the entire list fits the context comfortably. The
       model is the better judge of which fact is funniest, and pre-selecting risks hiding the very
       fact that makes the joke land. The mode instruction tells the model which angle to take.

4. LLM call
       model:       openai/gpt-oss-120b  → llama-3.3-70b-versatile → llama-4-scout (fallback chain)
                    gpt-oss-120b is primary: better world-knowledge/fact-comprehension, and it draws
                    on a SEPARATE Groq token budget from llama-3.3 (the main agent model), so heavy
                    roasting does not starve the bot's regular replies.
       temperature: 0.5
       top_p:       0.9
       max_tokens:  1024  (gpt-oss is a reasoning model — hidden reasoning eats output tokens before
                    the answer, so a tight cap leaves visible content empty; trim_to_single_roast
                    clamps the final text to 2 sentences regardless)
       style:       blunt and crude — state the real fact/contradiction plainly, then a short
                    dismissive jab. NO invented imagery, metaphors or similes (these read as absurd
                    Russian). Few-shot examples in ROAST_SYSTEM_PROMPT teach the register. Profanity
                    allowed; never appearance, illness, or family.
       post-trim:   trim_to_single_roast() keeps the first substantial paragraph, max 2 sentences —
                    a deterministic guard, because the model does not reliably self-limit length.
       constraint:  Russian only, no joke explanation

5. Fallback (no facts at all)
       → mock them for never writing anything ("@username вообще ничего не пишет в чате")

6. The offense handler (src/events/messages.py) sends the roast and stores it to
   unified_messages so users can reply to it and the bot has context.
```

## What information the roast reads

```
user_memories
    Source:  LLM-extracted facts accumulated over time from all chat messages
    Content: short plain-language sentences about the user, in Russian
    Window:  all facts for the target (max 30 stored per user per chat)
    Usage:   the full fact list is handed to the LLM, which picks the funniest fact (or contradicting
             pair) for the chosen mode's angle and crafts one short, blunt jab
```

## Auto-roast detection

```
GuardNode classifies message as MALICIOUS + trigger="explicit" (@mention or reply to bot)
    → random refusal sent to user
    → hack attempt recorded in user_memories as "Пытался взломать бота N раз"
    → roasted_count incremented in user_stats

handle_message (src/events/messages.py)
    → OFFENSE_RE matches text + message is a reply to the bot
    → offense_reply_counts[chat_id][user_id] += 1
    → if count >= 2: reset counter, generate roast, increment roasted_count
```

---

# /meme

Sends a random Russian-language meme image with its post title as caption.

## Source

Reddit public JSON API — no credentials required. Four subreddits are queried on every call:

```
r/ru_memes
r/expectedrussians
r/ruAsska
r/Pikabu
```

Each subreddit is fetched independently (`GET /hot.json?limit=100`). A subreddit failure is logged as a warning and skipped; the rest still proceed.

## Post filtering

```
is_video  = true  → skip
is_gallery = true → skip
post_hint = "image"
  OR url ends with .jpg / .jpeg / .png / .gif → keep
```

## Deduplication

Sent meme URLs are recorded in the `sent_memes` table keyed by `(chat_id, url)`. Each chat has its own independent pool. Once all fetched posts for a chat have been sent, the command replies with a text message instead.

## Flow

```
/meme
  → fetch up to 400 posts across all subreddits
  → filter: single image only (no video, no gallery)
  → exclude URLs already in sent_memes for this chat
  → pick random candidate
  → INSERT into sent_memes
  → reply_photo(url, caption=title)
```
