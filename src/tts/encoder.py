"""PCM-to-OGG/Opus encoding for Telegram voice notes.

Pure audio encoding on top of PyAV — no torch or Telegram knowledge, so the
synthesis engine and the delivery layer can evolve independently.
"""

import io

import av
import numpy

# Encode in ~1-second frames: single oversized frames can exceed what the
# resampler inside libopus accepts, chunked frames always work.
SAMPLES_PER_FRAME_SECONDS = 1


def encode_pcm_to_ogg_opus(pcm_samples: numpy.ndarray, sample_rate: int) -> bytes:
    """Encode mono float32 PCM samples into an in-memory OGG/Opus file.

    Args:
        pcm_samples: 1-D float32 array of samples in the [-1, 1] range.
        sample_rate: Sample rate of the PCM data in Hz.

    Returns:
        Bytes of a complete OGG container with a single Opus audio stream,
        ready to be sent via Telegram ``send_voice``.
    """
    int16_samples = (numpy.clip(pcm_samples, -1.0, 1.0) * 32767).astype(numpy.int16)
    output_buffer = io.BytesIO()
    with av.open(output_buffer, mode="w", format="ogg") as container:
        stream = container.add_stream("libopus", rate=sample_rate)
        stream.layout = "mono"
        frame_size = sample_rate * SAMPLES_PER_FRAME_SECONDS
        for start in range(0, len(int16_samples), frame_size):
            chunk = int16_samples[start : start + frame_size]
            frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = sample_rate
            frame.pts = start
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return output_buffer.getvalue()
