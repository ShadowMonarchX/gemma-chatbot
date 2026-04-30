from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
import sys

import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.main as backend_main
from backend.hardware import HardwareProfile


class MockModelBackend:
    model_name = "gemma-4-2b-it"
    quantization = "INT4-mlx"

    def stream_chat(self, messages: list[dict[str, str]], *, max_tokens: int = 512):
        system = messages[0]["content"].lower() if messages else ""
        user_text = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_text = message.get("content", "")
                break

        if "senior software engineer" in system:
            chunks = ["```python\n", "print('mock code response')\n", "```"]
        else:
            base = f"Mock reply: {user_text}".strip()
            chunks = [base[:8], base[8:20], base[20:]]

        for chunk in chunks:
            if chunk:
                yield chunk


@pytest_asyncio.fixture
async def app(monkeypatch) -> AsyncIterator:
    fake_hardware = HardwareProfile(
        chip="Apple M2",
        ram_total_gb=16,
        ram_available_gb=9.2,
        cpu_cores=8,
        metal_gpu=True,
        is_apple_silicon=True,
        quantization="INT4-mlx",
    )

    monkeypatch.setattr(backend_main, "detect_hardware", lambda: fake_hardware)
    monkeypatch.setattr(
        backend_main,
        "load_model_backend",
        lambda hardware: (MockModelBackend(), 35),
    )

    test_app = backend_main.create_app()
    async with LifespanManager(test_app) as manager:
        yield manager.app


@pytest_asyncio.fixture
async def client(app) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as async_client:
        yield async_client
