"""Text-to-speech package — Silero v5 Russian voice synthesis."""

from src.tts.service import SpeechService, SynthesizedVoice
from src.tts.text_prep import prepare_tts_text

speech_service = SpeechService()

__all__ = ["SpeechService", "SynthesizedVoice", "prepare_tts_text", "speech_service"]
