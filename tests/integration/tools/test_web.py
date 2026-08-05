"""Integration tests for the web search tool.

web_search uses DuckDuckGo by default when TAVILY_API_KEY is not configured.
No API keys are required for these tests.
"""

import json

import pytest

from src.tools.web import web_search


@pytest.mark.integration
class TestWebSearch:
    """Tests for web_search against the real DuckDuckGo (or Tavily) backend."""

    async def test_returns_a_list_not_an_error_dict(self):
        """web_search must return a JSON list on success, never an error dict.

        DuckDuckGo may return an empty list on rate-limited or no-result queries,
        so the assertion only checks the response type, not the result count.
        """
        raw = await web_search.ainvoke({"query": "Elden Ring release date"})
        results = json.loads(raw)
        assert isinstance(results, list)

    async def test_each_result_has_required_fields(self):
        """Every result must include title, url, and snippet fields."""
        raw = await web_search.ainvoke({"query": "Elden Ring release date"})
        results = json.loads(raw)
        for result in results:
            assert "title" in result
            assert "url" in result
            assert "snippet" in result
            assert result["url"].startswith("http")
            assert len(result["snippet"].strip()) > 0

    async def test_returns_at_most_five_results(self):
        """The tool is capped at 5 results; the list must never exceed that."""
        raw = await web_search.ainvoke({"query": "Counter-Strike 2 news"})
        results = json.loads(raw)
        assert isinstance(results, list)
        assert len(results) <= 5
