from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .hardware import HardwareInfo


class ChatMessage(BaseModel):
    """Single chat message in a request payload."""

    model_config = ConfigDict(strict=True, protected_namespaces=())

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4096)


class ChatRequest(BaseModel):
    """Strict request body for the chat streaming endpoint."""

    model_config = ConfigDict(strict=True, protected_namespaces=())

    messages: list[ChatMessage] = Field(min_length=1, max_length=20)
    skill_id: str = Field(pattern=r"^[a-z_]+$", max_length=32)
    model_id: str = Field(pattern=r"^[a-z0-9_-]+$", max_length=32)
    stream: bool = True


class ChatStreamEnvelope(BaseModel):
    """OpenAPI envelope model representing one SSE `data:` payload."""

    model_config = ConfigDict(strict=True, protected_namespaces=())

    data: str


class ErrorResponse(BaseModel):
    """Sanitized API error envelope returned by exception handlers."""

    model_config = ConfigDict(strict=True, protected_namespaces=())

    error: str
    request_id: str


class SkillResponse(BaseModel):
    """Public skill shape exposed to frontend callers."""

    model_config = ConfigDict(strict=True, protected_namespaces=())

    id: str
    label: str
    system_prompt: str


class ModelInfoResponse(BaseModel):
    """Model metadata returned by the models endpoint."""

    model_config = ConfigDict(strict=True, protected_namespaces=())

    id: str
    label: str
    backend: str
    source: str
    quantization: str
    available: bool
    default: bool
    description: str
    alias_of: str | None = None


class ModelsResponse(BaseModel):
    """Response body for available model catalog and active selection."""

    model_config = ConfigDict(strict=True, protected_namespaces=())

    active_model_id: str
    models: list[ModelInfoResponse]


class HealthResponse(BaseModel):
    """Operational health summary returned by `/api/health`."""

    model_config = ConfigDict(strict=True, protected_namespaces=())

    status: str
    model_id: str
    model_label: str
    backend: str
    quantization: str
    hardware: HardwareInfo
    model_load_ms: int
    avg_tokens_per_sec: float
    last_tokens_per_sec: float
    uptime_seconds: int
    last_request_ms: int


class AdminResponse(BaseModel):
    """Extended operational metrics returned by `/api/admin`."""

    model_config = ConfigDict(strict=True, protected_namespaces=())

    status: str
    model_id: str
    model_label: str
    backend: str
    quantization: str
    hardware: HardwareInfo
    model_load_ms: int
    avg_tokens_per_sec: float
    last_tokens_per_sec: float
    uptime_seconds: int
    last_request_ms: int
    total_requests: int
    errors: int
    avg_response_ms: float
    avg_first_token_ms: float
    requests_per_minute: float
    skill_usage: dict[str, int]
    model_usage: dict[str, int]
    injection_blocks: int
    rate_limit_hits: int
