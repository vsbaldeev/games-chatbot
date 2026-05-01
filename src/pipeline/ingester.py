"""
MessageIngester — second node in the LangGraph pipeline.

Processes raw media into readable text:
  - text      → passed through unchanged
  - voice     → transcribed via Groq Whisper
  - video_note → transcribed via Groq Whisper
  - photo     → described via vision LLM (one-sentence description)

Updates the unified_messages row written by the Router with the real content
so that reply-chain queries later in the pipeline return useful text.
"""

import base64
import io
from src import log

from groq import AsyncGroq
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from src import config
from src.pipeline.state import BotState
from src.store import unified_messages

logger = log.get_logger(__name__)

WHISPER_MODEL = "whisper-large-v3"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

__VISION_PROMPT = (
    "Describe this image in one concise sentence in Russian. "
    "Focus on what's visible: people, objects, text on screen, game UI, etc."
)


class MessageIngester:
    """Converts media messages to text and updates the unified_messages store."""

    async def __call__(self, state: BotState) -> dict:
        msg = state["incoming"]
        media_type = msg["media_type"]
        context_types = state["context_types"]
        bot = context_types.bot

        if media_type == "text":
            processed = msg["raw_text"] or ""
        elif media_type in ("voice", "video_note"):
            processed = await self.__transcribe(msg["file_id"], media_type, bot)
        elif media_type == "photo":
            processed = await self.__describe_photo(msg["file_id"], bot)
        else:
            processed = msg["raw_text"] or ""

        if media_type != "text" and processed:
            try:
                await unified_messages.update_content(
                    chat_id=msg["chat_id"],
                    message_id=msg["message_id"],
                    content=processed,
                )
            except Exception as err:
                logger.warning("Failed to update message content for %s: %s", msg["message_id"], err)

        incoming_update = dict(state["incoming"])
        incoming_update["processed_text"] = processed
        return {"incoming": incoming_update}

    async def __transcribe(self, file_id: str, media_type: str, bot) -> str:
        try:
            filename = "voice.ogg" if media_type == "voice" else "video_note.mp4"
            tg_file = await bot.get_file(file_id)
            buffer = io.BytesIO()
            await tg_file.download_to_memory(buffer)
            buffer.seek(0)
            audio_bytes = buffer.read()

            client = AsyncGroq(api_key=config.GROQ_API_KEY)
            transcription = await client.audio.transcriptions.create(
                file=(filename, audio_bytes),
                model=WHISPER_MODEL,
            )
            return transcription.text.strip()
        except Exception as err:
            logger.error("Transcription failed for file %s: %s", file_id, err)
            return ""

    async def __describe_photo(self, file_id: str, bot) -> str:
        try:
            tg_file = await bot.get_file(file_id)
            buffer = io.BytesIO()
            await tg_file.download_to_memory(buffer)
            raw_bytes = buffer.getvalue()

            if raw_bytes[:4] == b'\x89PNG':
                mime = "image/png"
            elif raw_bytes[:4] == b'RIFF' and raw_bytes[8:12] == b'WEBP':
                mime = "image/webp"
            else:
                mime = "image/jpeg"

            b64_image = base64.b64encode(raw_bytes).decode()
            image_url = f"data:{mime};base64,{b64_image}"

            llm = ChatGroq(
                model=VISION_MODEL,
                api_key=config.GROQ_API_KEY,
                temperature=0.1,
                max_tokens=100,
            )
            response = await llm.ainvoke([
                HumanMessage(content=[
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": __VISION_PROMPT},
                ]),
            ])
            return response.content.strip()
        except Exception as err:
            logger.error("Photo description failed for file %s: %s", file_id, err)
            return ""
