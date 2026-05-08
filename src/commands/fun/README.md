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
       /roast and weekly job → random.choice(chat_members)
       auto-roast            → the offending user

2. Load context
       SQLChatMessageHistory  — last 40 messages from message_store for this chat
       Filter to target only  — keep lines starting with "<username>:"
       Strip noise            — drop lines that are only URLs or emojis (no real words)
       Result: up to 40 meaningful text messages from the target, oldest first

3. Roll supportive chance
       10% → warm supportive message ("как лучший друг")
       90% → sarcastic roast ("стендап-комик")

4. LLM call
       model:       llama-3.3-70b-versatile
       temperature: 0.95  (high — different roast every time)
       max_tokens:  180
       constraint:  ≤ 2 sentences, Russian only, must mention @username

5. Fallback (no history)
       If the target has never written a meaningful message:
       → friendly invite message instead of a roast
```

## What information the roast reads

```
unified_messages
    Source:  pipeline message store, keyed by (chat_id, username)
    Content: raw message text; voice/video_note/video entries contain the Whisper
             transcript once ingested (placeholder rows are excluded)
    Window:  last 40 rows for the target user specifically (SQL-level filter)
    Filter:  lines containing only URLs or emojis are stripped
    Usage:   injected verbatim as "Последние сообщения @username в чате:"
             — the LLM reads actual things the target said to write a personalised roast
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
