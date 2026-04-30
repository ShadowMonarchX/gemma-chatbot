from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_health_endpoint_shape(client):
    response = await client.get("/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model"] == "gemma-4-2b-it"
    assert payload["quantization"] in {"INT4-mlx", "INT8-mlx", "Q4_K_M-gguf"}

    assert "hardware" in payload
    assert "chip" in payload["hardware"]
    assert "ram_total_gb" in payload["hardware"]
    assert "ram_available_gb" in payload["hardware"]
    assert "cpu_cores" in payload["hardware"]
    assert "metal_gpu" in payload["hardware"]

    assert "model_load_ms" in payload
    assert "avg_tokens_per_sec" in payload
    assert "uptime_seconds" in payload
    assert "last_request_ms" in payload


@pytest.mark.asyncio
async def test_admin_endpoint_shape(client):
    response = await client.get("/admin")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "ok"
    assert "total_requests" in payload
    assert "errors" in payload
    assert "avg_response_ms" in payload
    assert "skill_usage" in payload
    assert "hallucination_guards_triggered" in payload


@pytest.mark.asyncio
async def test_chat_streams_tokens_and_has_response_ms_header(client):
    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "skill": "chat",
        "stream": True,
    }

    async with client.stream("POST", "/chat", json=payload) as response:
        assert response.status_code == 200
        assert "x-response-ms" in response.headers

        body = ""
        async for chunk in response.aiter_text():
            body += chunk

    assert "event: token" in body
    assert "event: done" in body


@pytest.mark.asyncio
async def test_chat_code_skill_contains_code_block_markers(client):
    payload = {
        "messages": [{"role": "user", "content": "Write hello world in python"}],
        "skill": "code",
        "stream": True,
    }

    async with client.stream("POST", "/chat", json=payload) as response:
        assert response.status_code == 200

        body = ""
        async for chunk in response.aiter_text():
            body += chunk

    assert "```" in body


@pytest.mark.asyncio
async def test_chat_empty_messages_returns_422(client):
    payload = {
        "messages": [],
        "skill": "chat",
        "stream": True,
    }

    response = await client.post("/chat", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_response_time_header_under_5000_ms(client):
    payload = {
        "messages": [{"role": "user", "content": "quick check"}],
        "skill": "chat",
        "stream": False,
    }

    response = await client.post("/chat", json=payload)
    assert response.status_code == 200
    assert "x-response-ms" in response.headers
    assert int(response.headers["x-response-ms"]) < 5000
