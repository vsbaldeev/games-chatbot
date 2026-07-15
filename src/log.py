"""Central logging setup: compact console format, correlation ids, muting.

Format goals:
- one short aligned line per record: ``DD.MM HH:MM:SS L corr logger message``;
- a per-message correlation id (bound in the pipeline entry point) on every
  record, so DEBUG traces group per Telegram update;
- ANSI colors on the level/corr/logger fields only, never the message body,
  so ``grep`` over ``docker compose logs`` keeps working.
"""

import contextvars
import logging
import os
import sys

CORRELATION_ID = contextvars.ContextVar("correlation_id", default="-")

RESET = "\x1b[0m"
DIM = "\x1b[2m"
LEVEL_COLORS = {
    logging.DEBUG: "\x1b[2m",
    logging.INFO: "\x1b[36m",
    logging.WARNING: "\x1b[33m",
    logging.ERROR: "\x1b[31m",
    logging.CRITICAL: "\x1b[1;31m",
}
NAME_WIDTH = 16
CORR_WIDTH = 6
SNIPPET_LIMIT = 120
MUTED_LOGGERS = ("httpx", "httpcore", "telegram.ext.ExtBot", "apscheduler")


def bind_correlation_id(value: str) -> None:
    """Bind a correlation id to the current asyncio/contextvars context.

    Every log record emitted from this context (including tasks spawned via
    ``asyncio.create_task``, which copy the context) carries the id.

    Args:
        value: Short identifier, typically derived from the Telegram update id.
    """
    CORRELATION_ID.set(value)


def snippet(text: str | None, limit: int = SNIPPET_LIMIT) -> str:
    """Collapse whitespace and truncate text for safe DEBUG-level logging.

    Args:
        text: Arbitrary user/LLM content, possibly multi-line or None.
        limit: Maximum length of the returned string.

    Returns:
        A single-line excerpt at most ``limit`` characters long, or
        ``"<empty>"`` when the text is None or blank.
    """
    if text is None or not text.strip():
        return "<empty>"
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "…"


def shorten_logger_name(name: str) -> str:
    """Reduce a dotted logger name to its last two components, left-padded.

    Args:
        name: Full logger name, e.g. ``src.pipeline.filter_node``.

    Returns:
        A compact name like ``pipeline.filter_node`` padded to at least
        ``NAME_WIDTH`` characters for column alignment.
    """
    trimmed = name.removeprefix("src.")
    parts = trimmed.split(".")
    return ".".join(parts[-2:]).ljust(NAME_WIDTH)


class CorrelationFilter(logging.Filter):
    """Handler-level filter stamping the correlation id onto every record.

    Attached to the handler (not a logger) so third-party records passing
    through the root handler also receive the ``corr`` attribute.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach the current correlation id to the record.

        Args:
            record: The log record being emitted.

        Returns:
            Always True — the filter never drops records.
        """
        record.corr = CORRELATION_ID.get()
        return True


class ConsoleFormatter(logging.Formatter):
    """Compact aligned console formatter with optional ANSI colors.

    Line layout: ``DD.MM HH:MM:SS L corr logger message``, where ``L`` is the
    one-character level. WARNING and above append a dim ``(file:line)``
    suffix; exceptions render their traceback on the following lines.
    """

    def __init__(self, use_color: bool) -> None:
        """Initialize the formatter.

        Args:
            use_color: Whether to wrap level/corr/logger fields in ANSI codes.
        """
        super().__init__(datefmt="%d.%m %H:%M:%S")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Render a record as a single aligned console line.

        Args:
            record: The log record to render.

        Returns:
            The formatted line, with traceback appended when present.
        """
        timestamp = self.formatTime(record, self.datefmt)
        level_char = record.levelname[0]
        corr = str(getattr(record, "corr", "-")).ljust(CORR_WIDTH)
        name = shorten_logger_name(record.name)
        message = record.getMessage()
        location = ""
        if record.levelno >= logging.WARNING:
            location = f" ({record.filename}:{record.lineno})"
        if self.use_color:
            color = LEVEL_COLORS.get(record.levelno, "")
            level_char = f"{color}{level_char}{RESET}"
            corr = f"{DIM}{corr}{RESET}"
            name = f"{color}{name}{RESET}"
            location = f"{DIM}{location}{RESET}" if location else ""
        line = f"{timestamp} {level_char} {corr} {name} {message}{location}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        if record.stack_info:
            line = f"{line}\n{self.formatStack(record.stack_info)}"
        return line


def resolve_color_mode() -> bool:
    """Decide whether log output should use ANSI colors.

    Controlled by the ``LOG_COLOR`` env var: ``always`` forces colors on,
    ``never`` forces them off, anything else falls back to a tty check on
    stderr (where the stream handler writes).

    Returns:
        True when colors should be emitted.
    """
    mode = os.getenv("LOG_COLOR", "auto").lower()
    if mode == "always":
        return True
    if mode == "never":
        return False
    return sys.stderr.isatty()


def setup() -> None:
    """Configure root logging: format, correlation ids, third-party muting.

    Idempotent (``force=True``), so re-imports of the entry module cannot
    stack duplicate handlers. Levels come from env vars: ``LOG_LEVEL`` for
    the root (default INFO), plus per-library overrides.
    """
    level = logging.getLevelName(os.getenv("LOG_LEVEL", "INFO").upper())
    handler = logging.StreamHandler()
    handler.addFilter(CorrelationFilter())
    handler.setFormatter(ConsoleFormatter(use_color=resolve_color_mode()))
    logging.basicConfig(level=level, handlers=[handler], force=True)
    for noisy in MUTED_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    asyncpg_level = logging.getLevelName(os.getenv("ASYNCPG_LOG_LEVEL", "WARNING").upper())
    logging.getLogger("asyncpg").setLevel(asyncpg_level)
    tg_app_level = logging.getLevelName(os.getenv("TELEGRAM_APP_LOG_LEVEL", "WARNING").upper())
    logging.getLogger("telegram.ext.Application").setLevel(tg_app_level)
    tg_updater_level = logging.getLevelName(os.getenv("TELEGRAM_UPDATER_LOG_LEVEL", "INFO").upper())
    logging.getLogger("telegram.ext.Updater").setLevel(tg_updater_level)


def get_logger(name: str) -> logging.Logger:
    """Return a stdlib logger by name.

    Args:
        name: Logger name, usually the module's ``__name__``.

    Returns:
        The corresponding ``logging.Logger`` instance.
    """
    return logging.getLogger(name)
