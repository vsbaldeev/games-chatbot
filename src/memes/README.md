Meme fetching and deduplication for the /meme command.

## Modules

```
fetcher.py   — fetches posts from Reddit, filters to single images, returns one unseen (url, title)
store.py     — sent_memes table: tracks which URLs have been sent per chat
```

## Database

```sql
CREATE TABLE sent_memes (
    chat_id INTEGER NOT NULL,
    url     TEXT    NOT NULL,
    PRIMARY KEY (chat_id, url)
)
```

Initialized by `store.init_table()`, called at bot startup in `src/bot/__init__.py`.

## fetcher.py

```
SUBREDDITS       — tuple of subreddit names to query
IMAGE_EXTENSIONS — file extensions accepted as single images

_extract_posts(posts)
    Iterates raw Reddit JSON children.
    Skips is_video and is_gallery posts.
    Keeps posts where post_hint == "image" or url ends with an image extension.
    Returns list of (url, title).

fetch_posts()
    Opens a single httpx.AsyncClient session.
    GETs /hot.json?limit=100 for each subreddit.
    Per-subreddit failures are logged as warnings and skipped.
    Returns combined list of (url, title).

get_meme(chat_id) -> (url, title) | None
    Calls fetch_posts(), loads seen URLs from store, picks a random unseen candidate,
    records it via mark_seen(), returns (url, title).
    Returns None if all fetched posts have already been sent in this chat.
```

## store.py

```
init_table()              — CREATE TABLE IF NOT EXISTS sent_memes
get_seen_urls(chat_id)    — SELECT url WHERE chat_id = ? → set[str]
mark_seen(chat_id, url)   — INSERT OR IGNORE (chat_id, url)
```
