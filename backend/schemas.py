from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .hardware import HardwareInfo


class ChatMessage(BaseModel):
    """Single chat turn included in a chat request."""

    model_config = ConfigDict(strict=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4096)


class ChatRequest(BaseModel):
    """Validated request payload for the chat streaming endpoint."""

    model_config = ConfigDict(strict=True)

    messages: list[ChatMessage] = Field(min_length=1, max_length=20)
    skill_id: str = Field(pattern=r"^[a-z_]+$", max_length=32)
    stream: bool = True


class ChatStreamEnvelope(BaseModel):
    """OpenAPI envelope model for SSE stream payload lines."""

    model_config = ConfigDict(strict=True)

    data: str


class ErrorResponse(BaseModel):
    """Sanitized API error response."""

    model_config = ConfigDict(strict=True)

    error: str
    request_id: str


class HealthResponse(BaseModel):
    """Operational health snapshot of the chatbot backend."""

    model_config = ConfigDict(strict=True)

    status: str
    model: str
    quantization: str
    hardware: HardwareInfo
    model_load_ms: int
    avg_tokens_per_sec: float
    uptime_seconds: int
    last_request_ms: int


class AdminResponse(BaseModel):
    """Extended operational metrics for local administration."""

    model_config = ConfigDict(strict=True)

    status: str
    model: str
    quantization: str
    hardware: HardwareInfo
    model_load_ms: int
    avg_tokens_per_sec: float
    uptime_seconds: int
    last_request_ms: int
    total_requests: int
    errors: int
    avg_response_ms: float
    requests_per_minute: float
    skill_usage: dict[str, int]
    hallucination_guards_triggered: int
    rate_limit_hits: int
