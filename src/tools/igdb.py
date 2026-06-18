"""IGDB game database tools via Twitch OAuth."""

import json
import time
from datetime import datetime, timezone

import httpx
from langchain_core.tools import tool

from src import config

IGDB_API = "https://api.igdb.com/v4"
TWITCH_TOKEN_URL = "https://id.twitch.tv/oauth2/token"

TOKEN_CACHE: dict = {}

PS5_PLATFORM_ID = 167

GAME_MODE_IDS = {
    "singleplayer": 1,
    "multiplayer": 2,
    "coop": 3,
}


async def get_igdb_token() -> str:
    now = time.time()
    if TOKEN_CACHE.get("token") and TOKEN_CACHE.get("expires_at", 0) > now + 60:
        return TOKEN_CACHE["token"]
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(
            TWITCH_TOKEN_URL,
            params={
                "client_id": config.TWITCH_CLIENT_ID,
                "client_secret": config.TWITCH_CLIENT_SECRET,
                "grant_type": "client_credentials",
            },
        )
        response.raise_for_status()
        data = response.json()
    TOKEN_CACHE["token"] = data["access_token"]
    TOKEN_CACHE["expires_at"] = now + data.get("expires_in", 3600)
    return TOKEN_CACHE["token"]


async def igdb_post(endpoint: str, body: str) -> list:
    token = await get_igdb_token()
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            f"{IGDB_API}/{endpoint}",
            content=body,
            headers={
                "Client-ID": config.TWITCH_CLIENT_ID,
                "Authorization": f"Bearer {token}",
                "Content-Type": "text/plain",
            },
        )
        response.raise_for_status()
        return response.json()


def parse_release_year(timestamp: int | None) -> int | None:
    if not timestamp:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).year


def format_game_summary(game: dict) -> dict:
    return {
        "id": game.get("id"),
        "name": game.get("name"),
        "rating": round(game["total_rating"]) if game.get("total_rating") else None,
        "platforms": [platform["name"] for platform in game.get("platforms", [])],
        "year": parse_release_year(game.get("first_release_date")),
    }


@tool
async def search_games(query: str) -> str:
    """
    Search IGDB for games by name.
    Returns up to 5 matches with id, name, rating, platforms, and release year.
    Use the returned id with get_game_details to fetch full information.
    """
    try:
        body = (
            f'search "{query}"; '
            "fields id,name,total_rating,platforms.name,first_release_date; limit 5;"
        )
        results = await igdb_post("games", body)
        return json.dumps([format_game_summary(game) for game in results], ensure_ascii=False)
    except Exception as error:
        return json.dumps({"error": str(error)})


@tool
async def get_game_details(game_id: str) -> str:
    """
    Get full details for a game from IGDB by its numeric ID.
    Returns platforms, genres, game modes, rating, developer, and summary.
    Obtain the ID first using search_games.
    """
    try:
        body = (
            f"where id = {game_id}; "
            "fields name,platforms.name,genres.name,game_modes.name,"
            "total_rating,total_rating_count,"
            "involved_companies.company.name,involved_companies.developer,"
            "summary,first_release_date; limit 1;"
        )
        results = await igdb_post("games", body)
        if not results:
            return json.dumps({"error": f"No game found with id {game_id}"})
        game = results[0]
        developers = [
            company["company"]["name"]
            for company in game.get("involved_companies", [])
            if company.get("developer") and company.get("company")
        ]
        return json.dumps({
            "name": game.get("name"),
            "year": parse_release_year(game.get("first_release_date")),
            "platforms": [platform["name"] for platform in game.get("platforms", [])],
            "genres": [genre["name"] for genre in game.get("genres", [])],
            "game_modes": [mode["name"] for mode in game.get("game_modes", [])],
            "rating": round(game["total_rating"]) if game.get("total_rating") else None,
            "rating_count": game.get("total_rating_count"),
            "developers": developers,
            "summary": (game.get("summary") or "")[:500],
        }, ensure_ascii=False)
    except Exception as error:
        return json.dumps({"error": str(error)})


@tool
async def get_ps5_recommendations(mode: str = "multiplayer") -> str:
    """
    Get top-rated PS5 game recommendations from IGDB, filtered by game mode.
    mode options: "multiplayer", "coop", "singleplayer"
    Returns up to 5 highly-rated PS5 games sorted by rating descending.
    """
    try:
        game_mode_id = GAME_MODE_IDS.get(mode.lower(), GAME_MODE_IDS["multiplayer"])
        body = (
            f"where platforms = ({PS5_PLATFORM_ID}) & game_modes = ({game_mode_id}) "
            "& total_rating > 75 & total_rating_count > 20 & category = 0; "
            "fields name,total_rating,total_rating_count,genres.name,summary; "
            "sort total_rating desc; limit 5;"
        )
        results = await igdb_post("games", body)
        games = [
            {
                "name": game.get("name"),
                "rating": round(game["total_rating"]) if game.get("total_rating") else None,
                "genres": [genre["name"] for genre in game.get("genres", [])],
                "summary": (game.get("summary") or "")[:200],
            }
            for game in results
        ]
        return json.dumps(games, ensure_ascii=False)
    except Exception as error:
        return json.dumps({"error": str(error)})


ALL_TOOLS = [search_games, get_game_details, get_ps5_recommendations]
