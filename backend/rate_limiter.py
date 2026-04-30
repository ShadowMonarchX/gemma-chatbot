from __future__ import annotations

import time
from collections import defaultdict, deque

from .errors import RateLimitError


class RateLimiter:
    """In-memory sliding-window rate limiter keyed by client IP."""

    def __init__(self, max_requests: int = 30, window_seconds: int = 60) -> None:
        """Initialize rate limiter settings.

        Args:
            max_requests: Maximum requests allowed in window.
            window_seconds: Sliding window duration in seconds.

        Returns:
            None.
        """
        self._max_requests: int = max_requests
        self._window_seconds: int = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._retry_after: dict[str, int] = defaultdict(int)
        self._hits: int = 0

    def check(self, client_id: str) -> None:
        """Check whether a client can make another request.

        Args:
            client_id: Client key, usually remote IP.

        Returns:
            None.
        """
        now = time.time()
        queue = self._events[client_id]
        threshold = now - self._window_seconds
        while queue and queue[0] <= threshold:
            queue.popleft()

        if len(queue) >= self._max_requests:
            self._hits += 1
            retry_after = int(max((queue[0] + self._window_seconds) - now, 1))
            self._retry_after[client_id] = retry_after
            raise RateLimitError(
                message="Rate limit exceeded",
                status_code=429,
                log_detail=f"retry_after={retry_after}",
            )

        queue.append(now)
        self._retry_after[client_id] = 0

    def get_retry_after(self, client_id: str) -> int:
        """Return retry-after seconds for a throttled client.

        Args:
            client_id: Client key.

        Returns:
            int: Retry duration in seconds.
        """
        return int(self._retry_after.get(client_id, 0))

    def get_hits(self) -> int:
        """Return total number of rate-limit denials.

        Returns:
            int: Total denied request count.
        """
        return self._hits

    def reset(self) -> None:
        """Reset all in-memory limiter state.

        Returns:
            None.
        """
        self._events.clear()
        self._retry_after.clear()
        self._hits = 0


rate_limiter = RateLimiter()
