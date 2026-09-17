"""Short, in-character summary of a Short/Reel/YouTube video's top comments.

Replaces quoting comments verbatim: one cheap TAG_MODEL round trip produces
a 1-2 sentence audience-reaction line in the bot's own voice instead. Shared
by src.pipeline.shorts, src.pipeline.social_links.instagram_reel and
src.pipeline.social_links.youtube_video — previously each had its own
verbatim-quote implementation (two near-duplicates, one per label).

Best effort by contract, same as src.agent.compress.compress_to_budget: any
failure or empty output degrades to "", so a comment-summary failure can
only drop that section, never break the caller's content block.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src import config, log
from src.agent import ainvoke_with_backoff
from src.agent.language import normalize_homoglyphs
from src.config.prompts import COMMENT_SUMMARY_SYSTEM

logger = log.get_logger(__name__)

MAX_COMMENTS = 10             # top-level comments considered
COMMENT_CHAR_LIMIT = 200      # truncate each comment before prompting
MAX_TOKENS = 150


def build_user_content(comments: list[dict]) -> str:
    """Render the top comments (text + like count) as the model's human turn.

    Args:
        comments: Comment dicts with ``text``/``like_count`` keys, already
            top-sorted by the download service.

    Returns:
        One line per usable comment, or ``""`` when none have usable text.
    """
    lines = []
    for comment in comments[:MAX_COMMENTS]:
        text = (comment.get("text") or "").strip()
        if not text:
            continue
        if len(text) > COMMENT_CHAR_LIMIT:
            text = text[:COMMENT_CHAR_LIMIT] + "…"
        like_count = comment.get("like_count") or 0
        lines.append(f"- ({like_count} лайков) {text}")
    return "\n".join(lines)


async def summarize_comments(comments: list[dict] | None) -> str:
    """Summarize top comments into one short in-character reaction line.

    Args:
        comments: Comment dicts from the download service's info dict
            (``text``/``like_count`` keys, already top-sorted), or None.

    Returns:
        A ``[Реакция комментаторов]`` block with a 1-2 sentence summary, or
        ``""`` when there are no usable comments or the model call failed —
        never raises.
    """
    user_content = build_user_content(comments or [])
    if not user_content:
        return ""
    llm = ChatGroq(
        model=config.TAG_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0.7,
        top_p=0.9,
        max_tokens=MAX_TOKENS,
        max_retries=0,
        reasoning_effort="none",
    )
    try:
        response = await ainvoke_with_backoff(llm, [
            SystemMessage(content=COMMENT_SUMMARY_SYSTEM),
            HumanMessage(content=user_content),
        ])
    except Exception as error:
        logger.warning("Comment summary failed: %s", error)
        return ""
    if response.response_metadata.get("finish_reason") == "length":
        logger.warning("Comment summary hit max_tokens=%d before finishing", MAX_TOKENS)
    summary = normalize_homoglyphs(response.content or "").strip()
    if not summary:
        logger.warning("Comment summary returned empty output")
        return ""
    return f"[Реакция комментаторов]:\n{summary}"
