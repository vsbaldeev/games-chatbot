"""
Movie, TV, anime, and game-review tools.

- search_movie_or_tv — TMDB API (requires TMDB_API_KEY)
- search_anime       — AniList GraphQL (no key required)
- get_game_reviews   — OpenCritic (no key required)
- explain_term       — Wikipedia intro section

Register with: media_tools.register(mcp, config)
"""

import json

import httpx
from mcp.server.fastmcp import FastMCP

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
OPENCRITIC_SEARCH_API = "https://api.opencritic.com/api/game/search"
OPENCRITIC_GAME_API = "https://api.opencritic.com/api/game"
ANILIST_API = "https://graphql.anilist.co"

__ANILIST_QUERY = """
query ($search: String) {
  Media(search: $search, type: ANIME) {
    title { romaji english native }
    episodes
    status
    averageScore
    studios(isMain: true) { nodes { name } }
    description(asHtml: false)
  }
}
"""


def register(mcp: FastMCP, cfg) -> None:
    """Register all media tools with the FastMCP server."""

    @mcp.tool()
    async def search_movie_or_tv(query: str, media_type: str = "movie") -> str:
        """
        Search TMDB for a movie or TV show.
        media_type must be "movie" or "tv".
        Returns title, year, overview, rating, and genres.
        Requires TMDB_API_KEY to be configured.
        """
        if not cfg.TMDB_API_KEY:
            return json.dumps({"error": "TMDB_API_KEY is not configured"})
        if media_type not in ("movie", "tv"):
            media_type = "movie"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                search_response = await client.get(
                    f"https://api.themoviedb.org/3/search/{media_type}",
                    params={
                        "api_key": cfg.TMDB_API_KEY,
                        "query": query,
                        "language": "en-US",
                        "page": 1,
                    },
                )
                search_response.raise_for_status()
                results = search_response.json().get("results", [])
                if not results:
                    return json.dumps({"error": f"No results for '{query}'"})

                item = results[0]
                item_id = item["id"]

                details_response = await client.get(
                    f"https://api.themoviedb.org/3/{media_type}/{item_id}",
                    params={"api_key": cfg.TMDB_API_KEY, "language": "en-US"},
                )
                details_response.raise_for_status()
                details = details_response.json()

            title = details.get("title") or details.get("name")
            year = (details.get("release_date") or details.get("first_air_date") or "")[:4]
            genres = [g["name"] for g in details.get("genres", [])]
            return json.dumps({
                "title": title,
                "year": year,
                "overview": details.get("overview"),
                "vote_average": details.get("vote_average"),
                "vote_count": details.get("vote_count"),
                "genres": genres,
                "tmdb_url": f"https://www.themoviedb.org/{media_type}/{item_id}",
            }, ensure_ascii=False)
        except Exception as error:
            return json.dumps({"error": str(error)})

    @mcp.tool()
    async def search_anime(query: str) -> str:
        """
        Search AniList for an anime series.
        Returns title, episode count, status, score, studios, and synopsis.
        No API key required.
        """
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
                    ANILIST_API,
                    json={"query": __ANILIST_QUERY, "variables": {"search": query}},
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                media = response.json().get("data", {}).get("Media")

            if not media:
                return json.dumps({"error": f"No anime found for '{query}'"})

            titles = media.get("title", {})
            studios = [node["name"] for node in media.get("studios", {}).get("nodes", [])]
            description = (media.get("description") or "")[:500]
            return json.dumps({
                "title_romaji": titles.get("romaji"),
                "title_english": titles.get("english"),
                "episodes": media.get("episodes"),
                "status": media.get("status"),
                "average_score": media.get("averageScore"),
                "studios": studios,
                "description": description,
            }, ensure_ascii=False)
        except Exception as error:
            return json.dumps({"error": str(error)})

    @mcp.tool()
    async def get_game_reviews(game_name: str) -> str:
        """
        Get critic review summary from OpenCritic for a game.
        Returns overall score, recommendation percentage, and top critic excerpts.
        No API key required.
        """
        try:
            async with httpx.AsyncClient(timeout=15, headers={"User-Agent": "Mozilla/5.0"}) as client:
                search_response = await client.get(
                    OPENCRITIC_SEARCH_API,
                    params={"criteria": game_name},
                )
                search_response.raise_for_status()
                items = search_response.json()
                if not items:
                    return json.dumps({"error": f"No OpenCritic results for '{game_name}'"})

                game_id = items[0]["id"]
                game_response = await client.get(f"{OPENCRITIC_GAME_API}/{game_id}")
                game_response.raise_for_status()
                data = game_response.json()

            top_critics = [
                {"outlet": review.get("Outlet", {}).get("name"), "snippet": review.get("snippet")}
                for review in data.get("Reviews", [])[:3]
                if review.get("snippet")
            ]
            return json.dumps({
                "name": data.get("name"),
                "opencritic_score": data.get("averageScore"),
                "top_critic_score": data.get("topCriticScore"),
                "percent_recommended": data.get("percentRecommended"),
                "num_reviews": data.get("numReviews"),
                "top_reviews": top_critics,
            }, ensure_ascii=False)
        except Exception as error:
            return json.dumps({"error": str(error)})

    @mcp.tool()
    async def explain_term(term: str) -> str:
        """
        Look up a term, technology, or concept on Wikipedia.
        Returns the first paragraph of the article (up to 600 characters).
        Useful for explaining gaming jargon, technical terms, or historical references.
        """
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    WIKIPEDIA_API,
                    params={
                        "action": "query",
                        "prop": "extracts",
                        "exintro": True,
                        "explaintext": True,
                        "redirects": 1,
                        "titles": term,
                        "format": "json",
                        "utf8": 1,
                    },
                )
                response.raise_for_status()
                data = response.json()

            pages = data.get("query", {}).get("pages", {})
            page = next(iter(pages.values()))
            if page.get("pageid") == -1 or not page.get("extract"):
                return json.dumps({"error": f"Wikipedia article not found for '{term}'"})

            extract = page["extract"].strip()
            short_extract = extract[:600]
            if len(extract) > 600:
                short_extract = short_extract.rsplit(" ", 1)[0] + "…"

            return json.dumps({
                "term": page.get("title"),
                "summary": short_extract,
                "url": f"https://en.wikipedia.org/wiki/{page['title'].replace(' ', '_')}",
            }, ensure_ascii=False)
        except Exception as error:
            return json.dumps({"error": str(error)})
