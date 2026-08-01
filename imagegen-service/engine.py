"""Stable Diffusion pipeline lifecycle: lazy load, generation, idle unload.

The pipeline loads on the first job (a cold container idles at ~300 MB) and
unloads after ``IDLE_UNLOAD_SECONDS`` without work, so the 3 GB memory
budget is only occupied during the two weekly generations. All pipeline
access is serialized by one lock: generation and unload must never race.
"""

import dataclasses
import gc
import io
import re
import threading
import time

import torch

MODEL_ID = "Lykon/dreamshaper-8"
TORCH_THREADS = 4
IDLE_UNLOAD_SECONDS = 15 * 60

# Compel treats unescaped ( ) " as prompt-weighting/grouping syntax. Our
# prompts are CHARACTER_VISUAL_PROMPT + an LLM-generated image_prompt, not
# something a person hand-wrote for that DSL, so any of these characters
# the writer happens to produce must be treated as literal text.
COMPEL_SYNTAX_RE = re.compile(r'([()"])')


def escape_compel_syntax(text: str) -> str:
    """Escape Compel's structural syntax characters in generated text.

    Args:
        text: Raw prompt text, possibly containing parentheses or quotes.

    Returns:
        Text with ``(``, ``)`` and ``"`` backslash-escaped so Compel's
        parser treats them as literal characters.
    """
    return COMPEL_SYNTAX_RE.sub(r"\\\1", text)


@dataclasses.dataclass(frozen=True)
class GenerationParams:
    """One generation request's parameters.

    Attributes:
        prompt: Full positive prompt.
        negative_prompt: Optional negative prompt.
        width: Output width in pixels.
        height: Output height in pixels.
        steps: Diffusion steps (~20 with DPM++ 2M Karras).
        guidance_scale: CFG scale (~6 for standard sampling).
        seed: Optional seed for reproducible output.
    """

    prompt: str
    negative_prompt: str | None
    width: int
    height: int
    steps: int
    guidance_scale: float
    seed: int | None


class Engine:
    """Owns the Stable Diffusion pipeline and its load/unload lifecycle."""

    def __init__(self, device: str = "cpu") -> None:
        """Create an engine with no pipeline loaded.

        Args:
            device: Torch device to load the pipeline on. Always "cpu" in
                production (the VPS has no GPU); overridable for local
                smoke tests on faster hardware (e.g. "mps" on Apple Silicon).
        """
        self.__device = device
        self.__pipeline = None
        self.__compel = None
        self.__last_used = 0.0
        self.__lock = threading.Lock()

    @property
    def model_loaded(self) -> bool:
        """Whether the pipeline is currently resident in memory."""
        return self.__pipeline is not None

    def generate(self, params: GenerationParams) -> bytes:
        """Generate one image, loading the pipeline first when needed.

        Runs on a worker thread (the caller uses ``asyncio.to_thread``);
        the lock serializes it against ``unload_if_idle``.

        Args:
            params: Generation parameters.

        Returns:
            The generated image as PNG bytes.
        """
        with self.__lock:
            pipeline = self.__ensure_loaded()
            conditioning, negative_conditioning = self.__build_conditioning(params)
            generator = None
            if params.seed is not None:
                generator = torch.Generator(self.__device).manual_seed(params.seed)
            result = pipeline(
                prompt_embeds=conditioning,
                negative_prompt_embeds=negative_conditioning,
                width=params.width,
                height=params.height,
                num_inference_steps=params.steps,
                guidance_scale=params.guidance_scale,
                generator=generator,
            )
            self.__last_used = time.monotonic()
        buffer = io.BytesIO()
        result.images[0].save(buffer, format="PNG")
        return buffer.getvalue()

    def __build_conditioning(self, params: GenerationParams) -> tuple:
        """Build prompt/negative embeddings that bypass CLIP's 77-token cap.

        Passing a plain ``prompt=`` string caps at CLIP's 77-token limit
        and silently drops everything past it — for
        ``CHARACTER_VISUAL_PROMPT + image_prompt`` that tail is exactly the
        episode's scene detail, breaking the text-photo coherence
        guarantee. Compel chunks text into 77-token windows and
        concatenates their embeddings instead, so no content is dropped.
        Caller must hold ``self.__lock`` (``self.__compel`` is set by
        ``__ensure_loaded``).

        Args:
            params: Generation parameters.

        Returns:
            ``(conditioning, negative_conditioning)`` tensors, padded to
            equal length, ready for ``prompt_embeds``/
            ``negative_prompt_embeds``.
        """
        prompt = escape_compel_syntax(params.prompt)
        negative_prompt = escape_compel_syntax(params.negative_prompt or "")
        conditioning = self.__compel.build_conditioning_tensor(prompt)
        negative_conditioning = self.__compel.build_conditioning_tensor(negative_prompt)
        return self.__compel.pad_conditioning_tensors_to_same_length(
            [conditioning, negative_conditioning]
        )

    def unload_if_idle(self) -> bool:
        """Free the pipeline when it has sat unused past the idle window.

        Returns:
            True when the pipeline was unloaded by this call.
        """
        with self.__lock:
            if self.__pipeline is None:
                return False
            if time.monotonic() - self.__last_used < IDLE_UNLOAD_SECONDS:
                return False
            self.__pipeline = None
            self.__compel = None
            gc.collect()
            return True

    def __ensure_loaded(self):
        """Load the pipeline if it is not resident. Caller must hold the lock.

        float32 is deliberate: universally correct on any CPU. bfloat16
        roughly halves the weight footprint but silently produces black
        frames on CPUs without native bf16 — not worth it within a 3 GB
        budget that float32 already fits.

        Returns:
            The ready ``StableDiffusionPipeline``.
        """
        if self.__pipeline is not None:
            return self.__pipeline
        from compel import Compel
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

        torch.set_num_threads(TORCH_THREADS)
        pipeline = StableDiffusionPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32, safety_checker=None
        )
        # Standard multi-step sampling, not the LCM-LoRA speed hack: LCM's
        # 4-6 steps at guidance ~1.5 reliably hallucinated compositions
        # (portraits crowding out the episode's scene, subjects present but
        # not interacting). Verified locally: 20 steps of DPM++ 2M Karras at
        # CFG ~6 renders coherent multi-subject scenes. Posts are scheduled,
        # so minutes-per-image on the CPU host is an accepted cost.
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
        )
        pipeline.to(self.__device)
        self.__pipeline = pipeline
        # truncate_long_prompts=False: chunk-and-concatenate instead of the
        # tokenizer's default 77-token truncation (see __build_conditioning).
        self.__compel = Compel(
            tokenizer=pipeline.tokenizer,
            text_encoder=pipeline.text_encoder,
            truncate_long_prompts=False,
            device=self.__device,
        )
        self.__last_used = time.monotonic()
        return pipeline
