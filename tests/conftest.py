from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("GEMMA_SKIP_MODEL_LOAD", "1")

from backend.main import app, chatbot_app
from backend.metrics import metrics_collector
from backend.quantization import QuantizationStrategy
from backend.rate_limiter import rate_limiter


class MockQuantizationStrategy(QuantizationStrategy):
    """Mock strategy used by tests to stream deterministic tokens."""

    def __init__(self) -> None:
        """Initialize mock state."""
        self.loaded: bool = False

    def load_model(self, model_id: str) -> None:
        """Mark model as loaded.

        Args:
            model_id: Model identifier.

        Returns:
            None.
        """
        _ = model_id
        self.loaded = True

    def generate(self, messages: list[dict], system: str) -> Generator[str, None, None]:
        """Yield deterministic test tokens.

        Args:
            messages: Input messages.
            system: System prompt.

        Returns:
            Generator[str, None, None]: Fixed output stream.
        """
        _ = messages
        _ = system
        for token in ["Hello", " world", "!"]:
            yield token


@pytest.fixture(autouse=True)
def reset_metrics() -> Generator[None, None, None]:
    """Reset shared in-memory runtime state before each test."""
    metrics_collector.reset()
    rate_limiter.reset()
    yield
    metrics_collector.reset()
    rate_limiter.reset()


@pytest.fixture()
def mock_quantization_strategy() -> MockQuantizationStrategy:
    """Provide deterministic quantization strategy instance."""
    return MockQuantizationStrategy()


@pytest_asyncio.fixture()
async def async_client(
    mock_quantization_strategy: MockQuantizationStrategy,
) -> AsyncGenerator[AsyncClient, None]:
    """Create an async API client with mocked model manager.

    Args:
        mock_quantization_strategy: Strategy fixture.

    Returns:
        AsyncGenerator[AsyncClient, None]: Configured async client.
    """
    manager = chatbot_app.model_manager
    manager._strategy = mock_quantization_strategy
    manager._loaded = True
    manager._model_name = "google/gemma-4-2b-it"
    manager._quantization = "INT4"
    manager._model_load_ms = 10
    manager._last_tokens_per_sec = 50.0
    manager._avg_tokens_per_sec = 45.5

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
