Fun commands: roast ("прожарка") and meme.

## Trigger modes

```
/roast command    — on-demand; caller picks no target — random member from chat_members
weekly job        — one random day per week (deterministic per ISO week); fires at 12:00 UTC
auto-roast        — two consecutive offensive replies to bot → immediate roast of the offender
```

## Generation pipeline

```
1. Pick target
       /roast and weekly job → random.choice(chat_members), bot excluded from the pool
       auto-roast            → the offending user

2. Pick mode (random, 1-in-4 each)
       shame / quirk / boast → embedding-anchored fact selection (step 3a)
       contradiction         → full-fact-list hypocrisy hunt (step 3b)
       The chosen mode is returned and logged as the roast's anchor_key.

3a. Anchored fact selection (shame / quirk / boast — skipped if no embeddings stored)
       i.  Anchor retrieval: all facts ranked by cosine similarity to a fixed "embarrassment anchor"
           embedding; top 8 kept.
       ii. Diverse pick (maximal marginal relevance): start with the most embarrassing fact, then
           greedily add facts scored as EMBARRASSMENT_WEIGHT * anchor_sim − (1−EMBARRASSMENT_WEIGHT)
           * redundancy (max similarity to an already-chosen fact). EMBARRASSMENT_WEIGHT = 0.3, so a
           single dense cluster (e.g. one fandom) can't fill all 3 slots — the roast spans varied,
           universally-understandable angles instead of niche insider facts.
       Falls back to plain get_facts() when no embeddings are available.

3b. Contradiction selection (contradiction mode)
       The whole recent fact list (newest CONTRADICTION_FACT_LIMIT = 12) is passed to the model
       intact — no embedding pre-filter. Embeddings can't tell a funny stance clash ("cares about
       animals" + "eats beef") from a consistent pair ("loves cats" + "has kittens"), so detection
       is left to the LLM. The prompt asks it to roast the funniest contradiction/hypocrisy, or
       fall back to the most absurd single fact when none exists.

4. LLM call
       model:       llama-3.3-70b-versatile
       temperature: 0.5   (controlled — specificity beats randomness for humor)
       top_p:       0.9
       max_tokens:  100
       constraint:  ≤ 2 sentences, Russian only, no joke explanation

5. Fallback (no facts at all)
       → mock them for never writing anything ("@username вообще ничего не пишет в чате")

6. Store roast to unified_messages (/roast only, not weekly job)
       → sent message inserted so users can reply to the roast and the bot has context
```

## What information the roast reads

```
user_memories
    Source:  LLM-extracted facts accumulated over time from all chat messages
    Content: short plain-language sentences about the user, in Russian
    Window:  all facts for the target (max 30 stored per user per chat)
    Usage:   anchor modes pick varied embarrassing facts (anchor retrieval + MMR); contradiction
             mode passes the full list so the LLM can spot a hypocritical pair; either way the LLM
             crafts one targeted, universally-understandable joke
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

## Weekly day selection

```python
# Stateless — survives restarts, varies per ISO week, same day all day
year, week, _ = datetime.date.today().isocalendar()
roast_day = random.Random(year * 1000 + week).randint(0, 6)
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
