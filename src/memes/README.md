Meme fetching, vetting and deduplication for the /meme command and the daily job.

## Modules

```
fetcher.py   — gathers candidates, downloads and vets one, returns image bytes
judge.py     — vision-LLM gate: is this image a meme that works with no caption?
sources/     — one module per meme source; each yields MemeCandidate(key, image_url)
store.py     — sent_memes table: tracks which dedup keys have been sent per chat
```

Captions are never carried or reposted. They belong to the channel that posted
them — a scraped caption once made the bot look like it was asking readers for
donations — and a meme that needs one is rejected rather than repaired.

## Sources

Memes come from public sources that need **no API key, no login, no approval**.
Each source module exposes an async `fetch(client)` returning `MemeCandidate`
objects and is registered in `sources/__init__.py` (`SOURCES`). Adding a source
means adding a module and listing its `fetch` — nothing else changes.

```
sources/base.py      — MemeCandidate(key, image_url) and SourceFetcher type
sources/ninegag.py   — 9gag group-posts JSON feed (English/global memes)
sources/telegram.py  — public t.me/s channel web previews (e.g. Russian meme channels)
```

`MemeCandidate.key` is a **stable** identifier (`9gag:<id>`, `tg:<channel>/<id>`),
not the CDN image URL — CDN URLs can rotate, which would break deduplication.

### 9gag (`sources/ninegag.py`)

```
NINEGAG_FEED — https://9gag.com/v1/group-posts/group/default/type/hot

parse_ninegag(payload)
    Keeps posts where type == "Photo" and not nsfw.
    image_url = images.image700.url; key = "9gag:<id>".

fetch(client)
    GETs the feed with a browser User-Agent. On error logs a warning and
    returns []. Returns parse_ninegag(response.json()).
```

### Telegram channels (`sources/telegram.py`)

```
TELEGRAM_CHANNELS — tuple of public channel handles to scrape (editable)
CHANNEL_URL       — https://t.me/s/{channel}

parse_channel(html)
    Parses the public web-preview HTML with BeautifulSoup.
    For each div.tgme_widget_message[data-post], takes the
    a.tgme_widget_message_photo_wrap background-image URL (skips video
    thumbnails and link previews), key = "tg:<data-post>".

    This markup cannot distinguish a meme from the channel author's own
    photo post — a donation appeal, an ad, an announcement. That is what
    judge.py exists to catch.

fetch(client)
    GETs each channel with a browser User-Agent. Per-channel failures are
    logged as warnings and skipped. Returns combined candidates.
```

## fetcher.py

```
gather_candidates() -> list[MemeCandidate]
    Opens one shared httpx.AsyncClient and calls every source in SOURCES,
    combining their candidates. Sources log and swallow their own errors.

vet_candidate(chat_id, candidate) -> bytes | None
    Downloads one candidate and scores it with judge.score_meme().
    Returns the bytes on a pass, None when it should be skipped. Raises
    JudgeUnavailable when the judge gave no verdict — the absence of a
    decision, deliberately not the same value as a rejection.

get_meme(chat_id) -> bytes | None
    Gathers candidates and seen keys once, then vets up to
    MEME_JUDGE_ATTEMPTS of them, re-picking from that one batch rather
    than re-scraping. Returns the bytes of the first candidate to pass,
    else None.

    Downloading lives here, not in the caller, because the judge needs the
    bytes anyway.
```

| Attempt outcome | mark_seen? | Next |
|---|---|---|
| Download failed | no — a CDN hiccup is not the candidate's fault | next attempt |
| Judge returned None (outage) | no | **abort the loop**, return None |
| score < MEME_JUDGE_PASS_SCORE | yes — burned permanently | next attempt |
| score >= MEME_JUDGE_PASS_SCORE | yes | return the bytes |

The gate is **fail-closed**: no verdict means no post. A judge outage aborts
rather than retrying, because attempts 2 and 3 would fail identically and only
burn more downloads; and it marks nothing seen, because an outage is not a
verdict and would otherwise consume three good candidates per chat, forever.

## judge.py

```
detect_image_mime(bytes) — sniffs JPEG/PNG/WebP/GIF magic bytes. Needed
    because meme CDNs serve mixed formats and the data URL must declare
    the right one (life/photo_judge.py can hardcode PNG; this cannot).

make_judge_llm()         — VISION_MODEL with reasoning_effort="none".
    Without it the whole token budget burns inside a <think> block.

parse_verdict(data)      — 0-10 score, or None when missing/out of range.

score_meme(bytes)        — one vision call with MEME_JUDGE_SYSTEM and the
    image alone (no caption: none will be sent, so none is judged).
    Logs the score at INFO so the threshold stays tunable. Returns None on
    any failure — "unknown", never zero.
```

**Known risk — Cyrillic OCR.** Most configured channels are Russian, so the
gate depends on VISION_MODEL reading text rendered inside images. If that is
weak it rejects good memes. `scripts/calibrate_meme_judge.py` scores a live
batch and prints the verdicts without sending anything, which is how
MEME_JUDGE_PASS_SCORE should be re-tuned.

## Database

```sql
CREATE TABLE sent_memes (
    chat_id BIGINT NOT NULL,
    url     TEXT   NOT NULL,
    PRIMARY KEY (chat_id, url)
)
```

The `url` column stores the opaque dedup **key** (`9gag:…`, `tg:…`), not
necessarily an image URL. The `sent_memes` table is provisioned by Alembic
migrations (`alembic upgrade head`), not by the bot at startup.

## store.py

```
get_seen_urls(chat_id)    — SELECT url WHERE chat_id = ? → set[str]
mark_seen(chat_id, key)   — INSERT ON CONFLICT DO NOTHING (chat_id, key)
```
