"""Web search and article extraction tools."""

import datetime
import json

import httpx
import trafilatura
from langchain_core.tools import tool

from src import config

@tool
def get_current_datetime(part: str = "full", utc_offset: str = "0") -> str:
    """Return the current date/time or a specific part of it.

    Args:
        part: What to return. One of:
              "full"    — full datetime in the given timezone, e.g. "2026-05-03 15:42 (UTC+3)"
              "year"    — current year, e.g. "2026"
              "month"   — current month name and number, e.g. "May (5)"
              "weekday" — day of the week, e.g. "Saturday"
              "time"    — current time HH:MM in the given timezone, e.g. "15:42 (UTC+3)"
              "utc"     — full datetime in UTC, e.g. "2026-05-03 12:42 UTC"
        utc_offset: Timezone as hours offset from UTC, e.g. "3" for Moscow, "-5" for New York EST.
                    Only used for "full" and "time" parts. Defaults to "0" (UTC).
    """
    utc_now = datetime.datetime.now(datetime.timezone.utc)

    if part == "year":
        return str(utc_now.year)

    if part == "month":
        return f"{utc_now.strftime('%B')} ({utc_now.month})"

    if part == "weekday":
        return utc_now.strftime("%A")

    if part == "utc":
        return utc_now.strftime("%Y-%m-%d %H:%M UTC")

    offset = int(utc_offset)
    tz = datetime.timezone(datetime.timedelta(hours=offset))
    local_now = utc_now.astimezone(tz)
    sign = "+" if offset >= 0 else ""
    label = f"UTC{sign}{offset}"

    if part == "time":
        return local_now.strftime(f"%H:%M ({label})")

    return local_now.strftime(f"%Y-%m-%d %H:%M ({label})")


@tool
async def web_search(query: str) -> str:
    """
    Search the web for recent information about any topic.
    Uses Tavily when configured, falls back to DuckDuckGo.
    Returns up to 5 results with title, URL, and snippet.
    """
    if config.TAVILY_API_KEY:
        return await __search_tavily(query)
    return await __search_duckduckgo(query)


@tool
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


async def __search_tavily(query: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": config.TAVILY_API_KEY,
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


ALL_TOOLS = [get_current_datetime, web_search, fetch_article]
