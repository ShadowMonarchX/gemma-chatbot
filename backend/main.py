from __future__ import annotations

import asyncio
import html
import logging
import time
import traceback
import uuid
from collections.abc import Callable
from typing import Any, Generator

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError as PydanticValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from .config import settings
from .errors import AppError, InjectionError, ModelError, RateLimitError, ValidationError
from .hardware import HardwareDetector, HardwareInfo
from .metrics import MetricsCollector, metrics_collector
from .model_manager import ModelManager
from .quantization import LlamaCppQuantization, QuantizationSelector, QuantizationStrategy
from .rate_limiter import RateLimiter, rate_limiter
from .schemas import (
    AdminResponse,
    ChatRequest,
    ChatStreamEnvelope,
    HealthResponse,
    ModelInfoResponse,
    ModelsResponse,
    SkillResponse,
)
from .skills import SkillRegistry, skill_registry
from .validators import MessageValidator, message_validator


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a UUID request ID to request state and response headers."""

    async def dispatch(self, request: Request, call_next: Callable[..., Any]) -> Any:
        """Create request ID context and invoke the next ASGI layer.

        Args:
            request: Incoming HTTP request.
            call_next: Starlette middleware continuation callback.

        Returns:
            Any: Outgoing response.
        """
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Apply secure baseline headers on all responses."""

    async def dispatch(self, request: Request, call_next: Callable[..., Any]) -> Any:
        """Append security headers to response.

        Args:
            request: Incoming request.
            call_next: Starlette middleware continuation callback.

        Returns:
            Any: Outgoing response with security headers.
        """
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject request bodies larger than configured byte limit."""

    def __init__(self, app: FastAPI, max_body_bytes: int) -> None:
        """Initialize middleware request-size limit.

        Args:
            app: Wrapped ASGI app.
            max_body_bytes: Allowed maximum bytes per request body.

        Returns:
            None.
        """
        super().__init__(app)
        self._max_body_bytes: int = max_body_bytes

    async def dispatch(self, request: Request, call_next: Callable[..., Any]) -> Any:
        """Enforce `Content-Length` guard before route execution.

        Args:
            request: Incoming request.
            call_next: Starlette middleware continuation callback.

        Returns:
            Any: 413 response when too large, otherwise next response.
        """
        content_length_header = request.headers.get("content-length", "")
        if content_length_header.isdigit() and int(content_length_header) > self._max_body_bytes:
            request_id = str(getattr(request.state, "request_id", uuid.uuid4()))
            return JSONResponse(
                status_code=413,
                content={"error": "Request body too large", "request_id": request_id},
                headers={"X-Request-Id": request_id},
            )
        return await call_next(request)


class SSETokenStream:
    """Convert generated tokens to escaped SSE events and record metrics."""

    def __init__(
        self,
        token_stream: Generator[str, None, None],
        metrics: MetricsCollector,
        skill_id: str,
        model_id: str,
        started_at: float,
    ) -> None:
        """Initialize streaming iterator state.

        Args:
            token_stream: Model token generator.
            metrics: Metrics collector instance.
            skill_id: Active skill identifier.
            model_id: Active model identifier.
            started_at: Request start timestamp (`perf_counter`).

        Returns:
            None.
        """
        self._token_stream = token_stream
        self._metrics = metrics
        self._skill_id = skill_id
        self._model_id = model_id
        self._started_at = started_at
        self._is_finished = False
        self._token_count = 0
        self._first_token_ms: int = 0
        self._logger = logging.getLogger("gemma-chatbot.sse")

    def __aiter__(self) -> "SSETokenStream":
        """Return async iterator instance.

        Returns:
            SSETokenStream: Self.
        """
        return self

    async def __anext__(self) -> bytes:
        """Generate next SSE frame on the event loop thread.

        Returns:
            bytes: Encoded `data: ...` SSE message.
        """
        frame = self._next_frame()
        await asyncio.sleep(0)
        return frame

    def _next_frame(self) -> bytes:
        """Generate the next SSE frame from the model token stream."""
        if self._is_finished:
            raise StopAsyncIteration

        try:
            token = next(self._token_stream)
            if self._first_token_ms == 0:
                self._first_token_ms = max(int((time.perf_counter() - self._started_at) * 1000), 1)

            text = str(token)
            self._token_count += max(len(text.split()), 1)
            escaped = html.escape(text, quote=False)
            return f"data: {escaped}\n\n".encode("utf-8")
        except StopIteration:
            self._finalize(error=False)
            self._is_finished = True
            return b"data: [DONE]\n\n"
        except Exception as exc:
            self._finalize(error=True)
            self._is_finished = True
            self._logger.exception("stream_generation_failed model_id=%s", self._model_id)
            message = "Token generation failed"
            if isinstance(exc, ModelError):
                message = exc.message
            escaped = html.escape(message, quote=False)
            return f"event: error\ndata: {escaped}\n\n".encode("utf-8")

    def _finalize(self, error: bool) -> None:
        """Finalize one request metrics record.

        Args:
            error: Whether stream ended in an error state.

        Returns:
            None.
        """
        elapsed_ms = int((time.perf_counter() - self._started_at) * 1000)
        first_token_ms = self._first_token_ms if self._first_token_ms > 0 else elapsed_ms
        self._metrics.record_request(
            skill_id=self._skill_id,
            model_id=self._model_id,
            ms=max(elapsed_ms, 0),
            error=error,
            tokens_generated=self._token_count,
            first_token_ms=max(first_token_ms, 0),
        )


class ChatbotApp:
    """Application container that builds and configures FastAPI chatbot API."""

    def __init__(self) -> None:
        """Initialize app dependencies, hardware detection, and model runtime."""
        self.app = FastAPI(title="Gemma Local Chatbot", version="2.0.0")
        self.logger = logging.getLogger("gemma-chatbot")
        self.logger.setLevel(logging.INFO)

        self._started_at: float = time.time()
        self._startup_error: str | None = None
        self._no_model_error_message = "No models found. Please install at least one model."
        self._download_in_progress_message = "Model not available yet. Download in progress."
        self._download_failed_message = "Model download failed. Check server logs for details."

        self.hardware_detector: HardwareDetector = HardwareDetector()
        self.selector: QuantizationSelector = QuantizationSelector()
        self.model_manager: ModelManager = ModelManager.get_instance()
        self.skills: SkillRegistry = skill_registry
        self.validator: MessageValidator = message_validator
        self.metrics: MetricsCollector = metrics_collector
        self.rate_limiter: RateLimiter = rate_limiter

        self.hardware: HardwareInfo = self.hardware_detector.detect()

        try:
            strategy: QuantizationStrategy = self.selector.select(self.hardware)
            self.model_manager.configure(hardware=self.hardware)
            self.model_manager.load(strategy=strategy)
            model_stats = self.model_manager.get_stats()
            self.logger.info(
                "startup chip=%s backend=%s quantization=%s model=%s load_ms=%s warmup_tps=%s",
                self.hardware.chip,
                model_stats.get("backend", "unknown"),
                model_stats.get("quantization", "unknown"),
                model_stats.get("model_id", "unknown"),
                model_stats.get("model_load_ms", 0),
                model_stats.get("last_tokens_per_sec", 0.0),
            )
        except ModelError as model_exc:
            if self._is_model_unready_error(model_exc.message):
                self._startup_error = model_exc.message
                self.logger.warning("startup_model_unready message=%s", model_exc.message)
            else:
                self.logger.error("startup_failure model_error=%s", model_exc.message)
                self.logger.error(traceback.format_exc())
                try:
                    fallback = LlamaCppQuantization(quant="Q4_K_M", n_gpu_layers=0)
                    self.model_manager.configure(hardware=self.hardware)
                    self.model_manager.load(strategy=fallback)
                    self.logger.warning("startup_fallback backend=llama.cpp reason=%s", model_exc.message)
                except Exception as fallback_exc:
                    self._startup_error = f"{model_exc}; fallback={fallback_exc}"
                    self.logger.error("startup_fallback_failure error=%s", str(fallback_exc))
                    self.logger.error(traceback.format_exc())
        except Exception as primary_exc:
            self.logger.error("startup_failure primary_error=%s", str(primary_exc))
            self.logger.error(traceback.format_exc())
            try:
                fallback = LlamaCppQuantization(quant="Q4_K_M", n_gpu_layers=0)
                self.model_manager.configure(hardware=self.hardware)
                self.model_manager.load(strategy=fallback)
                self.logger.warning("startup_fallback backend=llama.cpp reason=%s", str(primary_exc))
            except Exception as fallback_exc:
                self._startup_error = f"{primary_exc}; fallback={fallback_exc}"
                self.logger.error("startup_fallback_failure error=%s", str(fallback_exc))
                self.logger.error(traceback.format_exc())

    def build(self) -> FastAPI:
        """Register middleware, handlers, routes, and return FastAPI app.

        Returns:
            FastAPI: Configured application.
        """
        self._register_middleware()
        self._register_exception_handlers()
        self._register_routes()
        return self.app

    def _register_routes(self) -> None:
        """Register HTTP endpoints for chat, health, admin, skills, and models."""
        self.app.add_api_route(
            "/api/chat",
            self.chat,
            methods=["POST"],
            response_model=ChatStreamEnvelope,
            response_class=StreamingResponse,
            summary="Stream chat response tokens over SSE",
        )
        self.app.add_api_route(
            "/api/health",
            self.health,
            methods=["GET"],
            response_model=HealthResponse,
            summary="Read service health",
        )
        self.app.add_api_route(
            "/api/admin",
            self.admin,
            methods=["GET"],
            response_model=AdminResponse,
            summary="Read service admin metrics",
        )
        self.app.add_api_route(
            "/api/skills",
            self.list_skills,
            methods=["GET"],
            response_model=list[SkillResponse],
            summary="List available skills",
        )
        self.app.add_api_route(
            "/api/models",
            self.list_models,
            methods=["GET"],
            response_model=ModelsResponse,
            summary="List available model profiles",
        )

    def _register_middleware(self) -> None:
        """Attach request ID, body limit, security headers, and strict CORS."""
        self.app.add_middleware(RequestContextMiddleware)
        self.app.add_middleware(
            BodySizeLimitMiddleware,
            max_body_bytes=settings.request_body_limit_bytes,
        )
        self.app.add_middleware(SecurityHeadersMiddleware)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )

    def _register_exception_handlers(self) -> None:
        """Install consistent exception translation and safe error envelopes."""

        @self.app.exception_handler(AppError)
        async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
            request_id = self._request_id(request)
            self.logger.error("app_error request_id=%s detail=%s", request_id, exc.log_detail)
            if isinstance(exc, ModelError) and self._is_model_unready_error(exc.message):
                payload = self._model_unready_payload(message=exc.message)
                payload["request_id"] = request_id
                return JSONResponse(
                    status_code=503,
                    content=payload,
                    headers={"X-Request-Id": request_id},
                )
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": exc.message, "request_id": request_id},
                headers={"X-Request-Id": request_id},
            )

        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
            request_id = self._request_id(request)
            if isinstance(exc.detail, dict):
                content = dict(exc.detail)
                content.setdefault("error", "Request failed")
            else:
                detail = exc.detail if isinstance(exc.detail, str) else "Request failed"
                content = {"error": detail}
            content["request_id"] = request_id
            headers = {"X-Request-Id": request_id}
            if exc.headers:
                headers.update(exc.headers)
            return JSONResponse(
                status_code=exc.status_code,
                content=content,
                headers=headers,
            )

        @self.app.exception_handler(RequestValidationError)
        async def request_validation_handler(
            request: Request, exc: RequestValidationError
        ) -> JSONResponse:
            request_id = self._request_id(request)
            return JSONResponse(
                status_code=422,
                content={
                    "error": "Validation error",
                    "details": exc.errors(),
                    "request_id": request_id,
                },
                headers={"X-Request-Id": request_id},
            )

        @self.app.exception_handler(PydanticValidationError)
        async def pydantic_validation_handler(
            request: Request, exc: PydanticValidationError
        ) -> JSONResponse:
            request_id = self._request_id(request)
            return JSONResponse(
                status_code=422,
                content={
                    "error": "Validation error",
                    "details": exc.errors(),
                    "request_id": request_id,
                },
                headers={"X-Request-Id": request_id},
            )

        @self.app.exception_handler(Exception)
        async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
            request_id = self._request_id(request)
            self.logger.error("unhandled_exception request_id=%s error=%s", request_id, str(exc))
            self.logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "request_id": request_id},
                headers={"X-Request-Id": request_id},
            )

    async def chat(self, request: Request, payload: ChatRequest) -> StreamingResponse | JSONResponse:
        """Validate request and stream model output tokens as SSE.

        Args:
            request: Incoming HTTP request.
            payload: Strict chat request body.

        Returns:
            StreamingResponse: `text/event-stream` response.
        """
        request_id = self._request_id(request)
        client_id = self._client_ip(request)

        try:
            self.rate_limiter.check(client_id)
        except RateLimitError:
            self.metrics.record_rate_limit_hit()
            retry_after = self.rate_limiter.get_retry_after(client_id)
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(retry_after), "X-Request-Id": request_id},
            )

        if payload.stream is not True:
            raise HTTPException(status_code=422, detail="Only streaming mode is supported")

        try:
            skill = self.skills.get(payload.skill_id)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.message) from exc

        if not self.model_manager.is_model_known(payload.model_id):
            raise HTTPException(status_code=422, detail="Unknown model ID")

        raw_messages = [message.model_dump() for message in payload.messages]
        try:
            sanitized_messages = self.validator.validate_messages(raw_messages)
        except InjectionError as exc:
            self.metrics.record_injection_block()
            raise HTTPException(status_code=400, detail=exc.message) from exc
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.message) from exc

        started = time.perf_counter()
        try:
            token_stream = self.model_manager.generate_stream(
                messages=sanitized_messages,
                system=skill.system_prompt,
                skill=payload.skill_id,
                model_id=payload.model_id,
            )
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.message) from exc
        except ModelError as exc:
            self.metrics.record_request(
                skill_id=payload.skill_id,
                model_id=payload.model_id,
                ms=int((time.perf_counter() - started) * 1000),
                error=True,
                tokens_generated=0,
                first_token_ms=0,
            )
            if self._is_model_unready_error(exc.message):
                payload_body = self._model_unready_payload(message=exc.message)
                payload_body["request_id"] = request_id
                return JSONResponse(
                    status_code=503,
                    content=payload_body,
                    headers={"X-Request-Id": request_id},
                )
            raise HTTPException(status_code=500, detail=exc.message) from exc

        iterator = SSETokenStream(
            token_stream=token_stream,
            metrics=self.metrics,
            skill_id=payload.skill_id,
            model_id=payload.model_id,
            started_at=started,
        )

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Request-Id": request_id,
            "X-Response-Ms": "0",
        }
        return StreamingResponse(iterator, media_type="text/event-stream", headers=headers)

    async def health(self, request: Request) -> HealthResponse:
        """Return high-level health and model runtime snapshot.

        Args:
            request: Incoming request.

        Returns:
            HealthResponse: Health payload.
        """
        _ = request
        stats = self.model_manager.get_stats()
        metrics = self.metrics.get_summary()
        status = "ok" if self._startup_error is None else "degraded"

        return HealthResponse(
            status=status,
            model_id=str(stats.get("model_id", "")),
            model_label=str(stats.get("model_label", "")),
            backend=str(stats.get("backend", "")),
            quantization=str(stats.get("quantization", "")),
            hardware=self.hardware,
            model_load_ms=int(stats.get("model_load_ms", 0)),
            avg_tokens_per_sec=float(stats.get("avg_tokens_per_sec", 0.0)),
            last_tokens_per_sec=float(stats.get("last_tokens_per_sec", 0.0)),
            uptime_seconds=int(metrics.get("uptime_seconds", int(time.time() - self._started_at))),
            last_request_ms=int(metrics.get("last_request_ms", 0)),
        )

    async def admin(self, request: Request) -> AdminResponse:
        """Return full operational and security metrics snapshot.

        Args:
            request: Incoming request.

        Returns:
            AdminResponse: Admin payload.
        """
        health_payload = await self.health(request)
        summary = self.metrics.get_summary()

        return AdminResponse(
            status=health_payload.status,
            model_id=health_payload.model_id,
            model_label=health_payload.model_label,
            backend=health_payload.backend,
            quantization=health_payload.quantization,
            hardware=health_payload.hardware,
            model_load_ms=health_payload.model_load_ms,
            avg_tokens_per_sec=health_payload.avg_tokens_per_sec,
            last_tokens_per_sec=health_payload.last_tokens_per_sec,
            uptime_seconds=health_payload.uptime_seconds,
            last_request_ms=health_payload.last_request_ms,
            total_requests=int(summary.get("total_requests", 0)),
            errors=int(summary.get("errors", 0)),
            avg_response_ms=float(summary.get("avg_response_ms", 0.0)),
            avg_first_token_ms=float(summary.get("avg_first_token_ms", 0.0)),
            requests_per_minute=float(summary.get("requests_per_minute", 0.0)),
            skill_usage=dict(summary.get("skill_usage", {})),
            model_usage=dict(summary.get("model_usage", {})),
            injection_blocks=int(summary.get("injection_blocks", 0)),
            rate_limit_hits=int(summary.get("rate_limit_hits", 0)),
        )

    async def list_skills(self, request: Request) -> list[SkillResponse]:
        """Return available skill definitions.

        Args:
            request: Incoming request.

        Returns:
            list[SkillResponse]: Registered skills.
        """
        _ = request
        return [SkillResponse(**skill.model_dump()) for skill in self.skills.all()]

    async def list_models(self, request: Request) -> ModelsResponse:
        """Return model catalog for runtime model selector.

        Args:
            request: Incoming request.

        Returns:
            ModelsResponse: Model metadata plus active model.
        """
        _ = request
        model_entries = [ModelInfoResponse(**entry) for entry in self.model_manager.list_models()]
        return ModelsResponse(
            active_model_id=self.model_manager.get_active_model_id(),
            models=model_entries,
        )

    def _request_id(self, request: Request) -> str:
        """Resolve request UUID from request state.

        Args:
            request: Incoming request.

        Returns:
            str: Request identifier.
        """
        request_id = getattr(request.state, "request_id", "")
        if request_id:
            return str(request_id)
        return str(uuid.uuid4())

    def _client_ip(self, request: Request) -> str:
        """Resolve best-effort client identifier for rate limiting.

        Args:
            request: Incoming request.

        Returns:
            str: Client IP string.
        """
        if request.client and request.client.host:
            return request.client.host
        return "127.0.0.1"

    def _is_model_unready_error(self, message: str) -> bool:
        """Return whether model startup/download is still pending or failed."""
        normalized = message.strip()
        if normalized in {
            self._no_model_error_message,
            self._download_in_progress_message,
            self._download_failed_message,
        }:
            return True
        return normalized.lower().startswith("model download failed")

    def _model_unready_payload(self, message: str) -> dict[str, Any]:
        """Return user-friendly payload for model download/startup states."""
        normalized = message.strip()
        if normalized == self._no_model_error_message:
            return {"error": self._download_in_progress_message}
        return {"error": normalized}


chatbot_app = ChatbotApp()
app = chatbot_app.build()
