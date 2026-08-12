"""Tests for the shared cross-provider vision LLM factory."""

import logging
from unittest.mock import patch

from langchain_core.runnables.fallbacks import RunnableWithFallbacks
from langchain_groq import ChatGroq

from src import config
from src.agent.vision import make_vision_llm

OPENROUTER_KEY_TARGET = "src.agent.vision.config.OPENROUTER_API_KEY"


class TestMakeVisionLlm:
    def test_without_openrouter_key_returns_bare_groq_client(self):
        with patch(OPENROUTER_KEY_TARGET, ""):
            llm = make_vision_llm(max_tokens=200)
        assert isinstance(llm, ChatGroq)
        assert llm.model_name == config.VISION_MODEL
        assert llm.max_tokens == 200
        assert llm.temperature == 0.1

    def test_without_openrouter_key_logs_warning(self, caplog):
        with (
            patch(OPENROUTER_KEY_TARGET, ""),
            caplog.at_level(logging.WARNING, logger="src.agent.vision"),
        ):
            make_vision_llm(max_tokens=200)
        assert "OPENROUTER_API_KEY unset" in caplog.text

    def test_with_openrouter_key_attaches_fallback(self):
        with patch(OPENROUTER_KEY_TARGET, "fake-key"):
            llm = make_vision_llm(max_tokens=150)
        assert isinstance(llm, RunnableWithFallbacks)
        assert llm.runnable.model_name == config.VISION_MODEL
        assert llm.runnable.max_tokens == 150
        assert len(llm.fallbacks) == 1
        assert llm.fallbacks[0].model_name == config.VISION_FALLBACK_MODEL
        assert llm.fallbacks[0].max_tokens == 150
