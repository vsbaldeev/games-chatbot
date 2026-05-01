"""Handler registries for the Telegram Application."""

from src.handlers.registry import (
    HandlerRegistry,
    EventHandlerRegistry,
    CommandHandlerRegistry,
    MessageHandlerRegistry,
)

__all__ = [
    "HandlerRegistry",
    "EventHandlerRegistry",
    "CommandHandlerRegistry",
    "MessageHandlerRegistry",
]
