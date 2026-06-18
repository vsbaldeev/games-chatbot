"""
Pytest configuration.

IMPORTANT: env vars must be set before any src.* import because src.config
reads them at module load time and raises ValueError when missing.
"""

import os

os.environ.setdefault("TELEGRAM_TOKEN", "123456789:AAtest-token-for-tests")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("TWITCH_CLIENT_ID", "test-twitch-id")
os.environ.setdefault("TWITCH_CLIENT_SECRET", "test-twitch-secret")
os.environ.setdefault("BOT_USERNAME", "testbot")
