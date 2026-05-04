"""LangChain tool modules for the games bot agent."""

from src.tools.igdb import get_game_details, get_ps5_recommendations, search_games
from src.tools.media import explain_term, get_game_reviews, search_anime, search_movie_or_tv
from src.tools.store import (
    get_ps_store_price_tr,
    get_ps_store_sales,
    get_steam_app_details,
    get_steam_player_count,
    get_steam_reviews_summary,
)
from src.tools.web import fetch_article, get_current_datetime, web_search

GAMES_TOOLS = [
    search_games,
    get_game_details,
    get_ps5_recommendations,
    get_ps_store_price_tr,
    get_ps_store_sales,
    get_steam_player_count,
    get_steam_app_details,
    get_steam_reviews_summary,
    get_game_reviews,
    web_search,
]

MEDIA_DOMAIN_TOOLS = [
    search_movie_or_tv,
    search_anime,
    web_search,
    fetch_article,
]

GENERAL_TOOLS = [
    web_search,
    fetch_article,
    get_current_datetime,
]

PYTHON_TOOLS = list({tool.name: tool for tool in GAMES_TOOLS + MEDIA_DOMAIN_TOOLS + GENERAL_TOOLS}.values())
