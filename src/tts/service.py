"""Silero TTS v5 speech synthesis service.

Owns the locally-loaded Silero model: one-time startup loading and
non-blocking synthesis of Telegram-ready OGG/Opus voice payloads. Any
runtime failure degrades to ``None`` so callers fall back to text replies.
"""

import asyncio
import dataclasses
import os

import torch

from src import config, log
from src.tts.encoder import encode_pcm_to_ogg_opus

logger = log.get_logger(__name__)

KNOWN_SPEAKERS = frozenset({"aidar", "baya", "kseniya", "xenia", "eugene"})

WARMUP_PHRASE = "Проверка связи."


@dataclasses.dataclass(frozen=True)
class SynthesizedVoice:
    """In-memory OGG/Opus payload ready for Telegram ``send_voice``.

    Attributes:
        ogg_bytes: Complete OGG/Opus file contents.
        duration_seconds: Rounded audio duration for the Telegram client.
    """

    ogg_bytes: bytes
    duration_seconds: int


class SpeechService:
    """Loads the Silero v5 Russian model and synthesizes voice replies."""

    def __init__(self) -> None:
        """Create an uninitialized service; call :meth:`init` before use."""
        self.__model = None
        self.__synthesis_lock = asyncio.Lock()

    @property
    def is_ready(self) -> bool:
        """Whether the model loaded successfully and synthesis is available."""
        return self.__model is not None

    async def init(self) -> None:
        """Validate configuration and load the model off the event loop.

        Fails fast on hard misconfiguration (unknown speaker name). A model
        download/load failure is logged and leaves the service not-ready —
        the bot starts anyway and every reply degrades to text.

        Raises:
            ValueError: If ``config.TTS_SPEAKER`` is not a known Silero voice.
        """
        if config.TTS_SPEAKER not in KNOWN_SPEAKERS:
            raise ValueError(
                f"Unknown TTS speaker '{config.TTS_SPEAKER}'; expected one of {sorted(KNOWN_SPEAKERS)}"
            )
        try:
            await asyncio.to_thread(self.__load_model_sync)
            logger.info("Silero TTS model loaded, speaker=%s", config.TTS_SPEAKER)
        except Exception as error:
            self.__model = None
            logger.error("Silero TTS model failed to load, voice replies disabled: %s", error)

    async def synthesize(self, text: str) -> SynthesizedVoice | None:
        """Synthesize speech for the given text without blocking the event loop.

        Calls are serialized: Silero's thread safety is unverified and
        parallel synthesis would saturate the CPU.

        Args:
            text: Synthesis-ready text (see ``prepare_tts_text``).

        Returns:
            The synthesized voice payload, or None when the service is not
            ready or synthesis/encoding failed or timed out.
        """
        if not self.is_ready:
            return None
        try:
            async with self.__synthesis_lock:
                return await asyncio.wait_for(
                    asyncio.to_thread(self.__synthesize_sync, text),
                    timeout=config.TTS_TIMEOUT_SECONDS,
                )
        except Exception as error:
            logger.warning("TTS synthesis failed, falling back to text: %s", error)
            return None

    def __load_model_sync(self) -> None:
        """Download the model file if missing, then load and warm it up."""
        torch.set_num_threads(config.TTS_TORCH_THREADS)
        model_path = config.TTS_MODEL_PATH
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
            logger.info("Downloading Silero TTS model to %s", model_path)
            torch.hub.download_url_to_file(config.TTS_MODEL_URL, model_path)
        importer = torch.package.PackageImporter(model_path)
        model = importer.load_pickle("tts_models", "model")
        model.to(torch.device("cpu"))
        self.__model = model
        # First apply_tts call pays one-off lazy-init costs; keep it out of
        # the first user-facing reply.
        self.__synthesize_sync(WARMUP_PHRASE)

    def __synthesize_sync(self, text: str) -> SynthesizedVoice:
        """Run Silero synthesis and Opus encoding on the current thread."""
        audio_tensor = self.__model.apply_tts(
            text=text,
            speaker=config.TTS_SPEAKER,
            sample_rate=config.TTS_SAMPLE_RATE,
            put_accent=True,
            put_yo=True,
        )
        pcm_samples = audio_tensor.numpy()
        ogg_bytes = encode_pcm_to_ogg_opus(pcm_samples, config.TTS_SAMPLE_RATE)
        duration_seconds = max(1, round(len(pcm_samples) / config.TTS_SAMPLE_RATE))
        return SynthesizedVoice(ogg_bytes=ogg_bytes, duration_seconds=duration_seconds)
