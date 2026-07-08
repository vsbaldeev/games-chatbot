Meme fetching and deduplication for the /meme command.

## Modules

```
fetcher.py   — gathers candidates from all sources, returns one unseen (image_url, caption)
sources/     — one module per meme source; each yields MemeCandidate(key, image_url, caption)
store.py     — sent_memes table: tracks which dedup keys have been sent per chat
```

## Sources

Memes come from public sources that need **no API key, no login, no approval**.
Each source module exposes an async `fetch(client)` returning `MemeCandidate`
objects and is registered in `sources/__init__.py` (`SOURCES`). Adding a source
means adding a module and listing its `fetch` — nothing else changes.

```
sources/base.py      — MemeCandidate(key, image_url, caption) and SourceFetcher type
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
    image_url = images.image700.url; caption = title; key = "9gag:<id>".

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
    thumbnails and link previews), caption from .tgme_widget_message_text,
    key = "tg:<data-post>".

fetch(client)
    GETs each channel with a browser User-Agent. Per-channel failures are
    logged as warnings and skipped. Returns combined candidates.
```

## fetcher.py

```
gather_candidates() -> list[MemeCandidate]
    Opens one shared httpx.AsyncClient and calls every source in SOURCES,
    combining their candidates. Sources log and swallow their own errors.

get_meme(chat_id) -> (image_url, caption) | None
    Calls gather_candidates(), loads seen keys from store, picks a random
    unseen candidate, records its key via mark_seen(), returns
    (image_url, caption). Returns None if nothing was fetched or all
    candidates were already sent in this chat.
```

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
