from __future__ import annotations

import time
from collections import defaultdict, deque


class MetricsCollector:
    """Collects in-memory operational metrics for runtime observability."""

    def __init__(self) -> None:
        """Initialize counters and rolling windows."""
        self._started_at: float = time.time()
        self._total_requests: int = 0
        self._errors: int = 0
        self._total_response_ms: int = 0
        self._total_tokens: int = 0
        self._total_first_token_ms: int = 0
        self._last_request_ms: int = 0
        self._skill_usage: dict[str, int] = defaultdict(int)
        self._model_usage: dict[str, int] = defaultdict(int)
        self._injection_blocks: int = 0
        self._rate_limit_hits: int = 0
        self._request_timestamps: deque[float] = deque()

    def record_request(
        self,
        skill_id: str,
        model_id: str,
        ms: int,
        error: bool,
        tokens_generated: int,
        first_token_ms: int,
    ) -> None:
        """Record a completed request.

        Args:
            skill_id: Skill identifier used by request.
            model_id: Active model identifier used by request.
            ms: End-to-end request duration in milliseconds.
            error: Whether request ended in an error state.
            tokens_generated: Number of generated tokens.
            first_token_ms: First-token latency in milliseconds.

        Returns:
            None.
        """
        self._total_requests += 1
        self._last_request_ms = ms
        self._total_response_ms += ms
        self._total_tokens += max(tokens_generated, 0)
        self._total_first_token_ms += max(first_token_ms, 0)
        self._skill_usage[skill_id] += 1
        self._model_usage[model_id] += 1

        self._request_timestamps.append(time.time())
        self._trim_request_window()

        if error:
            self._errors += 1

    def record_injection_block(self) -> None:
        """Increment injection block counter.

        Returns:
            None.
        """
        self._injection_blocks += 1

    def record_rate_limit_hit(self) -> None:
        """Increment rate limit hit counter.

        Returns:
            None.
        """
        self._rate_limit_hits += 1

    def get_summary(self) -> dict:
        """Return current aggregated metrics.

        Returns:
            dict: Runtime metrics snapshot.
        """
        self._trim_request_window()

        avg_response_ms = 0.0
        avg_tokens_per_sec = 0.0
        avg_first_token_ms = 0.0

        if self._total_requests > 0:
            avg_response_ms = round(self._total_response_ms / self._total_requests, 2)
            avg_first_token_ms = round(self._total_first_token_ms / self._total_requests, 2)

        if self._total_response_ms > 0:
            total_seconds = self._total_response_ms / 1000.0
            avg_tokens_per_sec = round(self._total_tokens / total_seconds, 2)

        return {
            "uptime_seconds": int(time.time() - self._started_at),
            "total_requests": self._total_requests,
            "errors": self._errors,
            "avg_response_ms": avg_response_ms,
            "avg_tokens_per_sec": avg_tokens_per_sec,
            "avg_first_token_ms": avg_first_token_ms,
            "last_request_ms": self._last_request_ms,
            "requests_per_minute": float(len(self._request_timestamps)),
            "skill_usage": dict(self._skill_usage),
            "model_usage": dict(self._model_usage),
            "injection_blocks": self._injection_blocks,
            "rate_limit_hits": self._rate_limit_hits,
        }

    def reset(self) -> None:
        """Reset all metrics state.

        Returns:
            None.
        """
        self._started_at = time.time()
        self._total_requests = 0
        self._errors = 0
        self._total_response_ms = 0
        self._total_tokens = 0
        self._total_first_token_ms = 0
        self._last_request_ms = 0
        self._skill_usage = defaultdict(int)
        self._model_usage = defaultdict(int)
        self._injection_blocks = 0
        self._rate_limit_hits = 0
        self._request_timestamps.clear()

    def _trim_request_window(self) -> None:
        """Trim request timestamps older than 60 seconds.

        Returns:
            None.
        """
        cutoff = time.time() - 60.0
        while self._request_timestamps and self._request_timestamps[0] < cutoff:
            self._request_timestamps.popleft()


metrics_collector = MetricsCollector()
