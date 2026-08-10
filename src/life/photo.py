"""Render selfie candidates and pick the most faithful one.

Extracted from the retired scheduled-life-post module: chat-requested selfies
(``src/life/selfie.py``) are the only remaining caller, but the render-score-
rank logic is substantial enough to keep in its own module rather than fold
into the request-handling flow.
"""

from src import config, log
from src.config.prompts import CHARACTER_VISUAL_PROMPT, PHOTO_FRAMING_HINT
from src.imagegen import generate_image
from src.life.photo_judge import score_photo

logger = log.get_logger(__name__)

# Rank of a candidate the vision judge could not score: below every real 0-10
# score, so it loses to any scored candidate but still ships when it is the
# only one that rendered. "Unknown", never "bad" — a judge outage must not
# look like a mismatch.
UNSCORED_RANK = -1


async def generate_best_photo(image_prompt: str) -> bytes | None:
    """Generate up to ``IMAGEGEN_CANDIDATES`` selfies and pick the most faithful.

    SD1.5 renders the described interaction stochastically, so each candidate
    (random seed) is scored by the vision judge against ``image_prompt``. The
    first candidate reaching ``PHOTO_JUDGE_PASS_SCORE`` ships immediately;
    otherwise the best-scoring one does — the judge ranks, it never gates. A
    candidate the judge could not score ranks below any scored one but still
    ships if it is all there is.

    Scene before character descriptor in the assembled prompt — see
    ``PHOTO_FRAMING_HINT`` in ``src/config/prompts.py`` for why.

    Args:
        image_prompt: The requested scene description, in English.

    Returns:
        PNG bytes of the chosen candidate, or None when every generation
        call failed (service down).
    """
    photo_prompt = f"{PHOTO_FRAMING_HINT}{image_prompt}, {CHARACTER_VISUAL_PROMPT}"
    logger.debug("Photo generation prompt: %s", log.snippet(photo_prompt, log.OUTGOING_TEXT_LIMIT))
    best_png = None
    best_rank = UNSCORED_RANK - 1
    for attempt in range(config.IMAGEGEN_CANDIDATES):
        candidate = await generate_image(photo_prompt)
        if candidate is None:
            continue
        rank = await rank_candidate(candidate, image_prompt, attempt)
        if rank >= config.PHOTO_JUDGE_PASS_SCORE:
            return candidate
        if rank > best_rank:
            best_png = candidate
            best_rank = rank
    if best_png is not None:
        logger.info(
            "No photo candidate passed the judge — using the best one (score %s)",
            best_rank if best_rank >= 0 else "unknown",
        )
    return best_png


async def rank_candidate(candidate: bytes, image_prompt: str, attempt: int) -> int:
    """Score one candidate for ranking against its siblings.

    Args:
        candidate: PNG bytes of the generated candidate.
        image_prompt: Scene the candidate is judged against.
        attempt: Zero-based candidate index, for the log line.

    Returns:
        The judge's 0-10 score, or :data:`UNSCORED_RANK` when the judge could
        not score it — below any real score, so an unscorable candidate loses
        to a scored one but still ships when it is all there is.
    """
    score = await score_photo(candidate, image_prompt)
    if score is None:
        return UNSCORED_RANK
    if score >= config.PHOTO_JUDGE_PASS_SCORE:
        logger.info("Photo candidate %d passed the judge (score %d)", attempt + 1, score)
    return score
