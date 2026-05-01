"""
Web search and article extraction tools.

- web_search   — Tavily (if TAVILY_API_KEY set) or DuckDuckGo fallback; top 5 results
- fetch_article — httpx + trafilatura; extracted article text (max 3000 chars)

Register with: web_tools.register(mcp, config)
"""

import json

import httpx
import trafilatura
from mcp.server.fastmcp import FastMCP


def register(mcp: FastMCP, cfg) -> None:
    """Register web tools with the FastMCP server."""

    @mcp.tool()
    async def web_search(query: str) -> str:
        """
        Search the web for recent information about any topic.
        Uses Tavily when configured, falls back to DuckDuckGo.
        Returns up to 5 results with title, URL, and snippet.
        """
        if cfg.TAVILY_API_KEY:
            return await __search_tavily(query, cfg.TAVILY_API_KEY)
        return await __search_duckduckgo(query)

    @mcp.tool()
    async def fetch_article(url: str) -> str:
        """
        Fetch a web page and extract its main article text.
        Useful for summarising blog posts, reviews, or news articles linked in chat.
        Returns up to 3000 characters of the main content.
        """
        try:
            async with httpx.AsyncClient(
                timeout=20,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0"},
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                html = response.text

            text = trafilatura.extract(html, include_comments=False, include_tables=False)
            if not text:
                return json.dumps({"error": "Could not extract article text from this page"})
            return json.dumps({"url": url, "content": text[:3000]}, ensure_ascii=False)
        except httpx.HTTPStatusError as error:
            return json.dumps({"error": f"HTTP {error.response.status_code} fetching {url}"})
        except Exception as error:
            return json.dumps({"error": str(error)})


async def __search_tavily(query: str, api_key: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": 5,
                    "search_depth": "basic",
                },
            )
            response.raise_for_status()
            data = response.json()

        results = [
            {"title": item.get("title"), "url": item.get("url"), "snippet": item.get("content")}
            for item in data.get("results", [])
        ]
        return json.dumps(results, ensure_ascii=False)
    except Exception as error:
        return json.dumps({"error": f"Tavily search failed: {error}"})


async def __search_duckduckgo(query: str) -> str:
    try:
        from duckduckgo_search import AsyncDDGS

        async with AsyncDDGS() as ddgs:
            raw_results = await ddgs.atext(query, max_results=5)

        results = [
            {"title": item.get("title"), "url": item.get("href"), "snippet": item.get("body")}
            for item in raw_results
        ]
        return json.dumps(results, ensure_ascii=False)
    except Exception as error:
        return json.dumps({"error": f"DuckDuckGo search failed: {error}"})
