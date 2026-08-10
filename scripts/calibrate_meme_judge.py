"""Score meme candidates through the vision gate without sending anything.

Two modes, because a threshold needs evidence from both directions:

``--limit N``
    Gather a live batch from the real sources and score N of them. Shows
    whether genuine memes clear the bar — in particular whether the model
    reads Cyrillic rendered inside an image, which is the one thing that
    cannot be settled by reading code.

``--file PATH [PATH ...]``
    Score local image files. This is the half a random live batch cannot
    supply: known non-memes — a donation appeal, an ad, a channel
    announcement — to confirm the gate actually rejects them. A gate that
    never says no is not a gate, and a batch of passes alone cannot tell
    the difference.

Read-only in both modes: touches no database, sends to no chat, marks nothing
seen. Only the vision model is called, once per image, at roughly 2,700 tokens
a call against a shared daily budget — keep the counts small.

Usage::

    python -m scripts.calibrate_meme_judge --limit 25
    python -m scripts.calibrate_meme_judge --file donation.jpg ad.png
"""

import argparse
import asyncio
import random
from pathlib import Path

from src import config
from src.memes.fetcher import download_image, gather_candidates
from src.memes.judge import detect_image_mime, score_meme

DEFAULT_LIMIT = 25


def format_verdict(score: int | None) -> str:
    """Render a judge score as a fixed-width verdict column.

    Args:
        score: The 0-10 score, or None when the judge gave no verdict.

    Returns:
        A padded ``"<score>  PASS|drop"`` string, or ``"judge-down"``.
    """
    if score is None:
        return "judge-down"
    marker = "PASS" if score >= config.MEME_JUDGE_PASS_SCORE else "drop"
    return f"{score:>2}  {marker}"


async def run_batch(limit: int) -> None:
    """Score a random sample of live candidates from the real sources.

    Args:
        limit: How many candidates to score.
    """
    candidates = await gather_candidates()
    if not candidates:
        print("No candidates fetched — check the sources.")
        return
    sample = random.sample(candidates, min(limit, len(candidates)))
    print(f"Scoring {len(sample)} of {len(candidates)} live candidates "
          f"(pass >= {config.MEME_JUDGE_PASS_SCORE})\n")
    for candidate in sample:
        image = await download_image(candidate.image_url)
        verdict = "dl-fail   " if image is None else format_verdict(await score_meme(image))
        print(f"{verdict}  {candidate.key}\n            {candidate.image_url}")


async def run_files(paths: list[str]) -> None:
    """Score local image files — the way to test known non-memes.

    Args:
        paths: Image file paths to score.
    """
    print(f"Scoring {len(paths)} local file(s) "
          f"(pass >= {config.MEME_JUDGE_PASS_SCORE})\n")
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            print(f"missing     {path}")
            continue
        image = path.read_bytes()
        verdict = format_verdict(await score_meme(image))
        print(f"{verdict}  {path}  [{detect_image_mime(image)}]")


def main() -> None:
    """Parse arguments and run the requested calibration mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"live candidates to score (default {DEFAULT_LIMIT})")
    parser.add_argument("--file", nargs="+", metavar="PATH",
                        help="score these local images instead of a live batch")
    args = parser.parse_args()
    asyncio.run(run_files(args.file) if args.file else run_batch(args.limit))


if __name__ == "__main__":
    main()
