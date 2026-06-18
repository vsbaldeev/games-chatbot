"""Unit tests for PS Store HTML scraping helpers.

Covers is_ps_store_addon_page, extract_offer_name, and scrape_ps_store_editions
using minimal HTML snippets that mimic the PS Store Next.js SSR output.
No network calls are made.
"""

import pytest

from src.tools.store import (
    extract_offer_name,
    is_ps_store_addon_page,
    scrape_ps_store_editions,
)


class TestIsPsStoreAddonPage:
    """is_ps_store_addon_page detects Add-On pages from PS Store HTML."""

    @pytest.mark.parametrize("html,description", [
        ('"topCategory":"Add-On"', "topCategory Add-On (mixed case)"),
        ('"topCategory":"ADD_ON"', "topCategory ADD_ON (upper case)"),
        ('"topCategory":"GAME_ADD_ON"', "topCategory GAME_ADD_ON"),
        ('"contentType":"GAME_ADD_ON"', "contentType GAME_ADD_ON"),
        (
            'data-qa="pdp-game-overview-topCategory" class="x">Add-On</span>',
            "rendered topCategory data-qa label",
        ),
    ], ids=[
        "topCategory-mixed-case",
        "topCategory-upper",
        "topCategory-game-addon",
        "contentType-game-addon",
        "data-qa-rendered",
    ])
    def test_returns_true_for_addon_signal(self, html: str, description: str):
        """Any known add-on marker in the HTML must return True."""
        assert is_ps_store_addon_page(html) is True, description

    def test_returns_false_for_regular_game_html(self):
        """A page with a game price but no add-on markers must return False."""
        html = '<span data-qa="mfeCtaMain#offer0#finalPrice">1.399,00 TL</span>'
        assert is_ps_store_addon_page(html) is False

    def test_returns_false_for_empty_html(self):
        """Empty string must return False without raising."""
        assert is_ps_store_addon_page("") is False

    def test_detection_is_case_insensitive(self):
        """Matching must be case-insensitive for the Add-On label."""
        assert is_ps_store_addon_page('"topCategory":"add-on"') is True


class TestExtractOfferName:
    """extract_offer_name pulls the edition label for a given offer index."""

    def test_returns_name_from_label_pattern(self):
        """Finds the name via the primary #label data-qa pattern."""
        html = '<span data-qa="mfeCtaMain#offer0#label">Standard Edition</span>'
        assert extract_offer_name(html, "0") == "Standard Edition"

    def test_returns_name_from_name_pattern(self):
        """Falls back to the secondary #name data-qa pattern."""
        html = '<span data-qa="mfeCtaMain#offer0#name">Deluxe Edition</span>'
        assert extract_offer_name(html, "0") == "Deluxe Edition"

    def test_resolves_correct_index(self):
        """Returns the name for the requested index, not for a different offer."""
        html = (
            '<span data-qa="mfeCtaMain#offer0#label">Standard Edition</span>'
            '<span data-qa="mfeCtaMain#offer1#label">Digital Deluxe</span>'
        )
        assert extract_offer_name(html, "0") == "Standard Edition"
        assert extract_offer_name(html, "1") == "Digital Deluxe"

    def test_returns_none_when_no_name_element(self):
        """Returns None when there is no matching name element."""
        html = '<span data-qa="mfeCtaMain#offer0#finalPrice">999 TL</span>'
        assert extract_offer_name(html, "0") is None

    def test_returns_none_for_empty_name(self):
        """Returns None when the matched element contains only whitespace."""
        html = '<span data-qa="mfeCtaMain#offer0#label">   </span>'
        assert extract_offer_name(html, "0") is None

    def test_strips_inner_html_tags(self):
        """Strips nested HTML tags and returns plain text."""
        html = '<span data-qa="mfeCtaMain#offer0#label"><b>Gold Edition</b></span>'
        assert extract_offer_name(html, "0") == "Gold Edition"


class TestScrapePsStoreEditions:
    """scrape_ps_store_editions extracts all offer editions from PS Store HTML."""

    def test_returns_empty_list_when_no_offers(self):
        """Returns an empty list when no offer price spans are present."""
        assert scrape_ps_store_editions("<html><body>Nothing here</body></html>") == []

    def test_single_offer_with_no_name_gets_standard_edition_label(self):
        """A lone offer with no name label defaults to 'Standard Edition'."""
        html = '<span data-qa="mfeCtaMain#offer0#finalPrice">1.399,00 TL</span>'
        editions = scrape_ps_store_editions(html)
        assert len(editions) == 1
        assert editions[0]["name"] == "Standard Edition"
        assert "1.399" in editions[0]["price_try"] or "1399" in editions[0]["price_try"].replace(".", "")

    def test_single_offer_with_name_uses_scraped_name(self):
        """Uses the scraped label when one is present for the offer."""
        html = (
            '<span data-qa="mfeCtaMain#offer0#label">Deluxe Edition</span>'
            '<span data-qa="mfeCtaMain#offer0#finalPrice">1.899,00 TL</span>'
        )
        editions = scrape_ps_store_editions(html)
        assert len(editions) == 1
        assert editions[0]["name"] == "Deluxe Edition"

    def test_multiple_offers_with_names_returned_in_order(self):
        """All named offers are returned in index order."""
        html = (
            '<span data-qa="mfeCtaMain#offer0#label">Standard</span>'
            '<span data-qa="mfeCtaMain#offer0#finalPrice">1.399,00 TL</span>'
            '<span data-qa="mfeCtaMain#offer1#label">Ultimate</span>'
            '<span data-qa="mfeCtaMain#offer1#finalPrice">2.199,00 TL</span>'
        )
        editions = scrape_ps_store_editions(html)
        assert len(editions) == 2
        assert editions[0]["name"] == "Standard"
        assert editions[1]["name"] == "Ultimate"

    def test_multiple_unnamed_offers_get_edition_n_fallback(self):
        """Multiple offers without labels get 'Edition 1', 'Edition 2' fallbacks."""
        html = (
            '<span data-qa="mfeCtaMain#offer0#finalPrice">1.399,00 TL</span>'
            '<span data-qa="mfeCtaMain#offer1#finalPrice">2.199,00 TL</span>'
        )
        editions = scrape_ps_store_editions(html)
        assert len(editions) == 2
        assert editions[0]["name"] == "Edition 1"
        assert editions[1]["name"] == "Edition 2"

    def test_offer_with_inner_html_tags_in_price_is_stripped(self):
        """HTML tags inside the price span are removed before returning."""
        html = '<span data-qa="mfeCtaMain#offer0#finalPrice"><span>999,00 TL</span></span>'
        editions = scrape_ps_store_editions(html)
        assert len(editions) == 1
        assert "<" not in editions[0]["price_try"]
        assert "999" in editions[0]["price_try"]

    def test_offer_with_no_digits_in_price_is_skipped(self):
        """Offers whose price contains no digits (e.g. placeholder) are skipped."""
        html = (
            '<span data-qa="mfeCtaMain#offer0#finalPrice">Free</span>'
            '<span data-qa="mfeCtaMain#offer1#finalPrice">1.399,00 TL</span>'
        )
        editions = scrape_ps_store_editions(html)
        assert len(editions) == 1
        assert "1.399" in editions[0]["price_try"] or "1399" in editions[0]["price_try"].replace(".", "")

    def test_price_try_key_present_in_each_edition(self):
        """Every returned edition dict has both 'name' and 'price_try' keys."""
        html = '<span data-qa="mfeCtaMain#offer0#finalPrice">599,00 TL</span>'
        editions = scrape_ps_store_editions(html)
        assert "name" in editions[0]
        assert "price_try" in editions[0]
