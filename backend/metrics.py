from __future__ import annotations

import time
from collections import defaultdict, deque


class MetricsCollector:
    """Collects in-memory request, error, and performance metrics."""

    def __init__(self) -> None:
        """Initialize counters and rolling windows."""
        self._started_at: float = time.time()
        self._total_requests: int = 0
        self._errors: int = 0
        self._total_response_ms: int = 0
        self._last_request_ms: int = 0
        self._skill_usage: dict[str, int] = defaultdict(int)
        self._hallucination_guards_triggered: int = 0
        self._rate_limit_hits: int = 0
        self._request_times: deque[float] = deque()
        self._avg_tokens_per_sec: float = 0.0

    def record_request(self, skill_id: str, ms: int, error: bool) -> None:
        """Record one request outcome.

        Args:
            skill_id: Skill used for the request.
            ms: Duration of request processing in milliseconds.
            error: Whether the request finished in error.

        Returns:
            None.
        """
        self._total_requests += 1
        self._last_request_ms = ms
        self._total_response_ms += ms
        self._skill_usage[skill_id] += 1
        self._request_times.append(time.time())
        self._trim_old_requests()
        if error:
            self._errors += 1

    def get_summary(self) -> dict:
        """Return a summarized metrics snapshot.

        Returns:
            dict: Snapshot values for API health and admin endpoints.
        """
        self._trim_old_requests()
        avg_response_ms = 0.0
        if self._total_requests > 0:
            avg_response_ms = round(self._total_response_ms / self._total_requests, 2)

        return {
            "uptime_seconds": int(time.time() - self._started_at),
            "total_requests": self._total_requests,
            "errors": self._errors,
            "avg_response_ms": avg_response_ms,
            "last_request_ms": self._last_request_ms,
            "requests_per_minute": round(float(len(self._request_times)), 2),
            "skill_usage": dict(self._skill_usage),
            "hallucination_guards_triggered": self._hallucination_guards_triggered,
            "rate_limit_hits": self._rate_limit_hits,
            "avg_tokens_per_sec": self._avg_tokens_per_sec,
        }

    def reset(self) -> None:
        """Reset all metrics counters for test isolation or admin use.

        Returns:
            None.
        """
        self._started_at = time.time()
        self._total_requests = 0
        self._errors = 0
        self._total_response_ms = 0
        self._last_request_ms = 0
        self._skill_usage = defaultdict(int)
        self._hallucination_guards_triggered = 0
        self._rate_limit_hits = 0
        self._request_times.clear()
        self._avg_tokens_per_sec = 0.0

    def record_hallucination_guard(self) -> None:
        """Increment injection/guard trigger counter.

        Returns:
            None.
        """
        self._hallucination_guards_triggered += 1

    def record_rate_limit_hit(self) -> None:
        """Increment rate-limit hit counter.

        Returns:
            None.
        """
        self._rate_limit_hits += 1

    def set_avg_tokens_per_sec(self, tokens_per_sec: float) -> None:
        """Update runtime average tokens/sec observed from model manager.

        Args:
            tokens_per_sec: Latest average throughput.

        Returns:
            None.
        """
        self._avg_tokens_per_sec = round(tokens_per_sec, 2)

    def _trim_old_requests(self) -> None:
        """Drop request timestamps older than 60 seconds.

        Returns:
            None.
        """
        cutoff = time.time() - 60
        while self._request_times and self._request_times[0] < cutoff:
            self._request_times.popleft()


metrics_collector = MetricsCollector()
