"""main.py tests — the POST /download dispatcher."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


class TestHealthz:
    def test_returns_ok(self):
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestDownloadDispatch:
    def test_youtube_short_kind_calls_shorts_download(self):
        with patch("main.shorts.download_short", return_value=(b"video", {"id": "abc"})) as mock_download:
            response = client.post("/download", json={"url": "https://x", "kind": "youtube_short"})
        mock_download.assert_called_once_with("https://x")
        assert response.status_code == 200
        body = response.json()
        assert body["info"] == {"id": "abc"}
        assert body["video_bytes_base64"] is not None

    def test_download_failure_returns_502(self):
        with patch("main.shorts.download_short", side_effect=FileNotFoundError("nope")):
            response = client.post("/download", json={"url": "https://x", "kind": "youtube_short"})
        assert response.status_code == 502

    def test_unknown_kind_returns_422(self):
        response = client.post("/download", json={"url": "https://x", "kind": "bogus"})
        assert response.status_code == 422
