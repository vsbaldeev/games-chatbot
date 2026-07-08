"""In-memory sliding-window gate shared by rate-limit-style features.

One small class covers two usage patterns:

* ``seen(key)`` — cooldown gate: the first call within the window returns
  ``False`` and records the key; subsequent calls within the window return
  ``True`` without extending it. Used by the quota-notice cooldown and the
  photo-album gate.
* ``hit(key)`` — escalation counter: every call records a hit and returns
  the number of hits within the window (including the current one). Used by
  the guard flag counter.

State is intentionally in-memory (mirrors ``humor_gate``/``insult_gate``):
losing it on restart is harmless for every consumer. Entries are pruned
lazily on access; a full sweep runs only when the key map grows large.
"""

import time
from typing import Callable, Hashable

FULL_PRUNE_THRESHOLD = 512


class TtlGate:
    """Sliding-window presence check and hit counter keyed by any hashable.

    Uses a monotonic clock by default so wall-clock adjustments cannot
    reopen or extend a window.
    """

    def __init__(
        self,
        window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialise an empty gate.

        Args:
            window_seconds: How long a recorded key or hit stays visible.
            clock: Zero-argument callable returning seconds; injectable
                for tests. Defaults to ``time.monotonic``.
        """
        self.__window_seconds = window_seconds
        self.__clock = clock
        self.__hits: dict[Hashable, list[float]] = {}

    def seen(self, key: Hashable) -> bool:
        """Check whether ``key`` was recorded within the window.

        Records the key on a miss, so the first caller in a window gets
        ``False`` and everyone else gets ``True`` until the window expires.
        Repeated calls do not extend the window.

        Args:
            key: Identifier for the gated event (e.g. a chat id).

        Returns:
            ``True`` if the key is already inside its window, ``False``
            if it was not (in which case it is now recorded).
        """
        now = self.__clock()
        recent = self.__prune_key(key, now)
        if recent:
            return True
        self.__hits[key] = [now]
        self.__maybe_full_prune(now)
        return False

    def hit(self, key: Hashable) -> int:
        """Record one hit for ``key`` and count hits within the window.

        Args:
            key: Identifier for the counted event (e.g. ``(chat_id, user_id)``).

        Returns:
            The number of hits within the window, including this one.
        """
        now = self.__clock()
        recent = self.__prune_key(key, now)
        recent.append(now)
        self.__hits[key] = recent
        self.__maybe_full_prune(now)
        return len(recent)

    def __prune_key(self, key: Hashable, now: float) -> list[float]:
        """Drop expired timestamps for one key and return the survivors.

        Args:
            key: The key being accessed.
            now: Current clock reading.

        Returns:
            The still-valid timestamps for ``key`` (possibly empty).
        """
        recent = [
            moment
            for moment in self.__hits.get(key, [])
            if now - moment < self.__window_seconds
        ]
        if recent:
            self.__hits[key] = recent
        else:
            self.__hits.pop(key, None)
        return recent

    def __maybe_full_prune(self, now: float) -> None:
        """Sweep all keys once the map grows past ``FULL_PRUNE_THRESHOLD``.

        Args:
            now: Current clock reading.
        """
        if len(self.__hits) <= FULL_PRUNE_THRESHOLD:
            return
        expired_keys = [
            key
            for key, moments in self.__hits.items()
            if all(now - moment >= self.__window_seconds for moment in moments)
        ]
        for key in expired_keys:
            del self.__hits[key]
