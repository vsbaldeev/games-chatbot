"""HANDLERS registry order test — this order is the multi-link precedence
tie-break the router relies on (Instagram > Reddit > YouTube video)."""

from src.pipeline.social_links import HANDLERS


def test_handlers_registry_priority_order():
    assert [handler.name for handler in HANDLERS] == [
        "instagram_reel", "reddit_post", "youtube_video",
    ]
