"""LangChain tool modules for the games bot agent."""

from src.tools.igdb import get_game_details, get_ps5_recommendations, search_games
from src.tools.media import search_anime, search_movie_or_tv
from src.tools.store import (
    get_ps_store_price_tr,
    get_ps_store_sales,
    get_steam_app_details,
    get_steam_player_count,
    get_steam_reviews_summary,
)
from src.tools.web import web_search

ALL_TOOLS = [
    search_games,
    get_game_details,
    get_ps5_recommendations,
    get_ps_store_price_tr,
    get_ps_store_sales,
    get_steam_player_count,
    get_steam_app_details,
    get_steam_reviews_summary,
    search_movie_or_tv,
    search_anime,
    web_search,
]
