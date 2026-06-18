"""Unit tests for the Telegram channel meme source.

Covers two units in isolation:
  - parse_channel: pure HTML-to-candidate parsing of t.me/s markup
  - fetch: async multi-channel HTTP path with the httpx client mocked
"""

from unittest.mock import AsyncMock, MagicMock

import httpx

from src.memes.sources import telegram


# ---------------------------------------------------------------------------
# HTML fixtures — minimal but faithful to real t.me/s markup
# ---------------------------------------------------------------------------

PHOTO_MESSAGE = """
<div class="tgme_widget_message_wrap js-widget_message_wrap">
  <div class="tgme_widget_message js-widget_message" data-post="ru2ch/171337">
    <a class="tgme_widget_message_photo_wrap" href="https://t.me/ru2ch/171337"
       style="width:599px;background-image:url('https://cdn4.telesco.pe/file/PHOTO1')"></a>
    <div class="tgme_widget_message_text js-message_text" dir="auto">Подпись один</div>
  </div>
</div>
"""

PHOTO_WITHOUT_TEXT = """
<div class="tgme_widget_message_wrap">
  <div class="tgme_widget_message" data-post="ru2ch/171340">
    <a class="tgme_widget_message_photo_wrap"
       style="background-image:url('https://cdn4.telesco.pe/file/PHOTO2')"></a>
  </div>
</div>
"""

VIDEO_MESSAGE = """
<div class="tgme_widget_message_wrap">
  <div class="tgme_widget_message" data-post="ru2ch/171338">
    <a class="tgme_widget_message_video_player">
      <i class="tgme_widget_message_video_thumb"
         style="background-image:url('https://cdn4.telesco.pe/file/VIDEOTHUMB')"></i>
    </a>
    <div class="tgme_widget_message_text">Видео подпись</div>
  </div>
</div>
"""

LINK_PREVIEW_MESSAGE = """
<div class="tgme_widget_message_wrap">
  <div class="tgme_widget_message" data-post="ru2ch/171339">
    <a class="tgme_widget_message_link_preview" href="https://example.com">
      <i class="link_preview_image"
         style="background-image:url('https://example.com/LINKIMG.jpg')"></i>
    </a>
    <div class="tgme_widget_message_text">Ссылка</div>
  </div>
</div>
"""

PHOTO_WITHOUT_DATA_POST = """
<div class="tgme_widget_message_wrap">
  <div class="tgme_widget_message">
    <a class="tgme_widget_message_photo_wrap"
       style="background-image:url('https://cdn4.telesco.pe/file/ORPHAN')"></a>
  </div>
</div>
"""


def page(*messages: str) -> str:
    """Wrap message fragments in a minimal channel-page body."""
    return "<html><body>" + "".join(messages) + "</body></html>"


# ---------------------------------------------------------------------------
# parse_channel
# ---------------------------------------------------------------------------

class TestParseChannel:
    def test_extracts_key_url_and_caption(self):
        result = telegram.parse_channel(page(PHOTO_MESSAGE))
        assert len(result) == 1
        candidate = result[0]
        assert candidate.key == "tg:ru2ch/171337"
        assert candidate.image_url == "https://cdn4.telesco.pe/file/PHOTO1"
        assert candidate.caption == "Подпись один"

    def test_photo_without_text_has_empty_caption(self):
        result = telegram.parse_channel(page(PHOTO_WITHOUT_TEXT))
        assert len(result) == 1
        assert result[0].caption == ""
        assert result[0].image_url == "https://cdn4.telesco.pe/file/PHOTO2"

    def test_skips_video_messages(self):
        assert telegram.parse_channel(page(VIDEO_MESSAGE)) == []

    def test_skips_link_preview_messages(self):
        assert telegram.parse_channel(page(LINK_PREVIEW_MESSAGE)) == []

    def test_skips_photo_without_data_post(self):
        assert telegram.parse_channel(page(PHOTO_WITHOUT_DATA_POST)) == []

    def test_mixed_page_returns_only_photos(self):
        html = page(PHOTO_MESSAGE, VIDEO_MESSAGE, LINK_PREVIEW_MESSAGE, PHOTO_WITHOUT_TEXT)
        keys = [candidate.key for candidate in telegram.parse_channel(html)]
        assert keys == ["tg:ru2ch/171337", "tg:ru2ch/171340"]

    def test_empty_html_returns_empty_list(self):
        assert telegram.parse_channel("<html></html>") == []


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

def make_response(text: str, *, raise_error: Exception | None = None) -> MagicMock:
    """Build a mock httpx.Response exposing ``.text`` and ``.raise_for_status()``."""
    response = MagicMock()
    response.text = text
    response.raise_for_status = MagicMock(side_effect=raise_error)
    return response


class TestTelegramFetch:
    async def test_combines_candidates_across_channels(self, monkeypatch):
        monkeypatch.setattr(telegram, "TELEGRAM_CHANNELS", ("chan1", "chan2"))
        responses = [make_response(page(PHOTO_MESSAGE)), make_response(page(PHOTO_WITHOUT_TEXT))]
        client = AsyncMock()
        client.get = AsyncMock(side_effect=responses)
        result = await telegram.fetch(client)
        assert {candidate.key for candidate in result} == {"tg:ru2ch/171337", "tg:ru2ch/171340"}

    async def test_one_failing_channel_does_not_abort_others(self, monkeypatch):
        monkeypatch.setattr(telegram, "TELEGRAM_CHANNELS", ("good", "bad"))
        client = AsyncMock()
        client.get = AsyncMock(side_effect=[
            make_response(page(PHOTO_MESSAGE)),
            httpx.ConnectError("down"),
        ])
        result = await telegram.fetch(client)
        assert [candidate.key for candidate in result] == ["tg:ru2ch/171337"]

    async def test_requests_each_channel_url_with_browser_agent(self, monkeypatch):
        monkeypatch.setattr(telegram, "TELEGRAM_CHANNELS", ("alpha", "beta"))
        client = AsyncMock()
        client.get = AsyncMock(return_value=make_response(page()))
        await telegram.fetch(client)
        requested_urls = [call.args[0] for call in client.get.call_args_list]
        assert requested_urls == ["https://t.me/s/alpha", "https://t.me/s/beta"]
        for call in client.get.call_args_list:
            assert call.kwargs["headers"]["User-Agent"].startswith("Mozilla/")
