MCP tool server implementations — exposed via stdio transport (src/mcp_server.py).

Consumed by the specialist worker agents in src/pipeline/worker_node.py.
Each module maps to one intent domain.

## Tools by module

```
igdb.py   (intent: games)
    search_games(query)                 — full-text game search via IGDB
    get_game_details(game_id)           — name, rating, platforms, genres, release date
    find_coop_games(genre?)             — multiplayer PS5 games with crossplay + TRY price
    find_new_ps5_online_games()         — recent online PS5 releases
    find_singleplayer_ps_games()        — singleplayer PS games, IGDB rating ≥ 75

store.py  (intent: games)
    get_ps_store_sales()                — current PS Store discounts (psdeals.net RSS)
    get_ps_store_price_tr(game_name)    — TRY price lookup for a specific game
    get_steam_player_count(app_id)      — current concurrent players (Steam API)
    get_steam_app_details(app_id)       — name, description, price, metacritic
    get_steam_reviews_summary(app_id)   — review score and count

media.py  (intent: media)
    search_movie_or_tv(query)           — TMDB lookup: year, overview, rating, genres
    search_anime(query)                 — AniList GraphQL: episodes, score, studios
    get_game_reviews(game_name)         — OpenCritic: critic score and review count
    explain_term(term)                  — Wikipedia summary (one paragraph)

web.py    (intent: general)
    web_search(query)                   — Tavily search (falls back to DuckDuckGo)
    fetch_article(url)                  — trafilatura text extraction + summarisation
```

## Auth

```
IGDB          Twitch OAuth client credentials (TWITCH_CLIENT_ID, TWITCH_CLIENT_SECRET)
TMDB          TMDB_API_KEY (optional; disables search_movie_or_tv if absent)
Tavily        TAVILY_API_KEY (optional; falls back to DuckDuckGo)
Steam/AniList no auth required
OpenCritic    no auth required
```
