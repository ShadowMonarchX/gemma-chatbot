from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("SKIP_MODEL_LOAD", "1")
os.environ.setdefault("MODEL_PATH", "backend/models")
os.environ.setdefault("DEFAULT_MODEL", "gemma-2b")
os.environ.setdefault("MAX_TOKENS", "512")

from backend.main import app, chatbot_app
from backend.metrics import metrics_collector
from backend.quantization import QuantizationStrategy
from backend.rate_limiter import rate_limiter


class MockQuantizationStrategy(QuantizationStrategy):
    """Mock quantization strategy with deterministic token streaming."""

    def __init__(self) -> None:
        """Initialize mock state."""
        self.backend_name: str = "mock"
        self.quantization: str = "MOCK"
        self.loaded_model_id: str = ""

    def load_model(self, model_id: str) -> None:
        """Record loaded model ID.

        Args:
            model_id: Model identifier.

        Returns:
            None.
        """
        self.loaded_model_id = model_id

    def generate(self, messages: list[dict], system: str) -> Generator[str, None, None]:
        """Yield deterministic stream chunks.

        Args:
            messages: Sanitized messages.
            system: System prompt.

        Returns:
            Generator[str, None, None]: Mock token stream.
        """
        _ = messages
        _ = system
        for token in ["Hello", " world", "!"]:
            yield token


@pytest.fixture(autouse=True)
def reset_runtime_state() -> Generator[None, None, None]:
    """Reset shared state before and after each test."""
    metrics_collector.reset()
    rate_limiter.reset()
    yield
    metrics_collector.reset()
    rate_limiter.reset()


@pytest.fixture()
def mock_quantization_strategy() -> MockQuantizationStrategy:
    """Create one deterministic quantization strategy fixture."""
    return MockQuantizationStrategy()


@pytest_asyncio.fixture()
async def async_client(
    mock_quantization_strategy: MockQuantizationStrategy,
) -> AsyncGenerator[AsyncClient, None]:
    """Create async client with model manager forced to deterministic mode.

    Args:
        mock_quantization_strategy: Deterministic strategy fixture.

    Returns:
        AsyncGenerator[AsyncClient, None]: Async client fixture.
    """
    manager = chatbot_app.model_manager
    mock_quantization_strategy.load_model("mock://gemma")

    manager._model_cache = {
        "gemma-2b": mock_quantization_strategy,
        "gemma-e2b": mock_quantization_strategy,
        "gemma-e4b": mock_quantization_strategy,
    }
    manager._model_load_times_ms = {
        "gemma-2b": 10,
        "gemma-e2b": 10,
        "gemma-e4b": 10,
    }
    manager._active_strategy = mock_quantization_strategy
    manager._active_model_id = "gemma-2b"
    manager._last_tokens_per_sec = 50.0
    manager._avg_tokens_per_sec = 45.5
    manager._generation_runs = 1

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
