"""Domain exceptions raised by the agent layer and caught by pipeline nodes."""


class RateLimitError(Exception):
    """Raised when Groq rate-limit retries are exhausted."""


class DailyLimitError(Exception):
    """Raised when Groq daily token quota is exhausted."""


class ContextLengthError(Exception):
    """Raised when input exceeds the model's context window."""