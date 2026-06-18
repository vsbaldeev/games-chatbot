MCP tool server implementations — exposed via stdio transport (src/mcp_server.py).

All 13 tools are bundled into a single `ALL_TOOLS` list and loaded into the unified
worker agent. The worker selects which tools to call based on the user's question.

## Tools by module

```
igdb.py
    search_games(query)                 — full-text game search via IGDB
    get_game_details(game_id)           — name, rating, platforms, genres, release date
    get_ps5_recommendations()           — top-rated PS5 games with crossplay + TRY price

store.py
    get_ps_store_sales()                — current PS Store discounts (psdeals.net RSS)
    get_ps_store_price_tr(game_name)    — TRY price lookup for a specific game
    get_steam_player_count(app_id)      — current concurrent players (Steam API)
    get_steam_app_details(app_id)       — name, description, price, metacritic
    get_steam_reviews_summary(app_id)   — review score and count

media.py
    search_movie_or_tv(query, type)     — TMDB lookup: year, overview, rating, genres
    search_anime(query)                 — AniList GraphQL: episodes, score, studios
    get_game_reviews(game_name)         — OpenCritic: critic score and review count

web.py
    web_search(query)                   — Tavily search (falls back to DuckDuckGo)
```

## Auth

```
IGDB          Twitch OAuth client credentials (TWITCH_CLIENT_ID, TWITCH_CLIENT_SECRET)
TMDB          TMDB_API_KEY (optional; disables search_movie_or_tv if absent)
Tavily        TAVILY_API_KEY (optional; falls back to DuckDuckGo)
Steam/AniList no auth required
OpenCritic    no auth required
```
