from __future__ import annotations

import html
import logging
import time
import traceback
import uuid
from typing import Any, Generator

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError as PydanticValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from .errors import AppError, InjectionError, ModelError, RateLimitError, ValidationError
from .hardware import HardwareDetector, HardwareInfo
from .metrics import MetricsCollector, metrics_collector
from .model_manager import ModelManager
from .quantization import QuantizationSelector, QuantizationStrategy
from .rate_limiter import RateLimiter, rate_limiter
from .schemas import (
    AdminResponse,
    ChatRequest,
    ChatStreamEnvelope,
    HealthResponse,
)
from .skills import Skill, SkillRegistry, skill_registry
from .validators import MessageValidator, message_validator


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Injects a request ID into request state and response headers."""

    def __init__(self, app: FastAPI) -> None:
        """Initialize middleware with FastAPI app.

        Args:
            app: ASGI application.

        Returns:
            None.
        """
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        """Attach request ID to every response.

        Args:
            request: Incoming request.
            call_next: Next ASGI handler.

        Returns:
            Any: Outgoing response.
        """
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Applies baseline security headers on every HTTP response."""

    def __init__(self, app: FastAPI) -> None:
        """Initialize middleware with FastAPI app.

        Args:
            app: ASGI application.

        Returns:
            None.
        """
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        """Append security headers after request processing.

        Args:
            request: Incoming request.
            call_next: Next ASGI handler.

        Returns:
            Any: Response with hardened headers.
        """
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Rejects HTTP requests whose body exceeds a configured byte limit."""

    def __init__(self, app: FastAPI, max_body_bytes: int = 64 * 1024) -> None:
        """Initialize request size middleware.

        Args:
            app: ASGI application.
            max_body_bytes: Maximum accepted payload size in bytes.

        Returns:
            None.
        """
        super().__init__(app)
        self._max_body_bytes = max_body_bytes

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        """Validate content length before route execution.

        Args:
            request: Incoming request.
            call_next: Next ASGI handler.

        Returns:
            Any: Accepted response or 413 JSON response.
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
    """Iterator that converts generated tokens into escaped SSE event lines."""

    def __init__(
        self,
        token_stream: Generator[str, None, None],
        metrics: MetricsCollector,
        skill_id: str,
        started_at: float,
    ) -> None:
        """Initialize streaming iterator state.

        Args:
            token_stream: Model token generator.
            metrics: Shared metrics collector.
            skill_id: Active skill identifier.
            started_at: Request start timestamp from perf_counter.

        Returns:
            None.
        """
        self._token_stream = token_stream
        self._metrics = metrics
        self._skill_id = skill_id
        self._started_at = started_at
        self._finished = False

    def __iter__(self) -> "SSETokenStream":
        """Return self as an iterator.

        Returns:
            SSETokenStream: Iterator instance.
        """
        return self

    def __next__(self) -> bytes:
        """Return one SSE-formatted event chunk.

        Returns:
            bytes: SSE `data:` line and delimiter.
        """
        if self._finished:
            raise StopIteration

        try:
            token = next(self._token_stream)
            escaped = html.escape(str(token), quote=False)
            return f"data: {escaped}\n\n".encode("utf-8")
        except StopIteration:
            self._finalize(error=False)
            self._finished = True
            return b"data: [DONE]\n\n"
        except Exception:
            self._finalize(error=True)
            self._finished = True
            return b"data: [DONE]\n\n"

    def _finalize(self, error: bool) -> None:
        """Finalize request metrics exactly once.

        Args:
            error: Whether stream ended with an error.

        Returns:
            None.
        """
        elapsed_ms = int((time.perf_counter() - self._started_at) * 1000)
        self._metrics.record_request(skill_id=self._skill_id, ms=elapsed_ms, error=error)


class ChatbotApp:
    """Builds and configures the local-first Gemma chatbot API."""

    def __init__(self) -> None:
        """Initialize app dependencies, hardware detection, and model runtime."""
        self.logger = logging.getLogger("gemma-chatbot")
        self.logger.setLevel(logging.INFO)
        self.app = FastAPI(title="Gemma Local Chatbot", version="1.0.0")
        self._started_at = time.time()

        self.hardware_detector: HardwareDetector = HardwareDetector()
        self.quantization_selector: QuantizationSelector = QuantizationSelector()
        self.skills: SkillRegistry = skill_registry
        self.validator: MessageValidator = message_validator
        self.rate_limiter: RateLimiter = rate_limiter
        self.metrics: MetricsCollector = metrics_collector
        self.model_manager: ModelManager = ModelManager.get_instance()

        self.hardware: HardwareInfo = self.hardware_detector.detect()
        self.strategy: QuantizationStrategy = self.quantization_selector.select(self.hardware)
        self.model_manager.load(self.strategy)

        model_stats = self.model_manager.get_stats()
        self.logger.info(
            "startup chip=%s quantization=%s model_load_ms=%s warmup_tokens_per_sec=%s",
            self.hardware.chip,
            model_stats.get("quantization", "unknown"),
            model_stats.get("model_load_ms", 0),
            model_stats.get("last_tokens_per_sec", 0.0),
        )

    def build(self) -> FastAPI:
        """Build and return the configured FastAPI application.

        Returns:
            FastAPI: Ready-to-run API application.
        """
        self._register_middleware()
        self._register_exception_handlers()
        self._register_routes()
        return self.app

    def _register_routes(self) -> None:
        """Register all HTTP routes on the FastAPI app.

        Returns:
            None.
        """
        self.app.add_api_route(
            "/api/chat",
            self.chat,
            methods=["POST"],
            response_model=ChatStreamEnvelope,
            response_class=StreamingResponse,
        )
        self.app.add_api_route(
            "/api/health",
            self.health,
            methods=["GET"],
            response_model=HealthResponse,
        )
        self.app.add_api_route(
            "/api/admin",
            self.admin,
            methods=["GET"],
            response_model=AdminResponse,
        )
        self.app.add_api_route(
            "/api/skills",
            self.list_skills,
            methods=["GET"],
            response_model=list[Skill],
        )

    def _register_middleware(self) -> None:
        """Register CORS, size guard, and security middleware.

        Returns:
            None.
        """
        self.app.add_middleware(RequestContextMiddleware)
        self.app.add_middleware(BodySizeLimitMiddleware, max_body_bytes=64 * 1024)
        self.app.add_middleware(SecurityHeadersMiddleware)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )

    def _register_exception_handlers(self) -> None:
        """Register consistent exception handlers for API safety.

        Returns:
            None.
        """

        @self.app.exception_handler(AppError)
        async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
            request_id = self._request_id(request)
            self.logger.error("app_error request_id=%s detail=%s", request_id, exc.log_detail)
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": exc.message, "request_id": request_id},
                headers={"X-Request-Id": request_id},
            )

        @self.app.exception_handler(RequestValidationError)
        async def request_validation_handler(
            request: Request, exc: RequestValidationError
        ) -> JSONResponse:
            request_id = self._request_id(request)
            return JSONResponse(
                status_code=422,
                content={"error": "Validation error", "details": exc.errors(), "request_id": request_id},
                headers={"X-Request-Id": request_id},
            )

        @self.app.exception_handler(PydanticValidationError)
        async def pydantic_validation_handler(
            request: Request, exc: PydanticValidationError
        ) -> JSONResponse:
            request_id = self._request_id(request)
            return JSONResponse(
                status_code=422,
                content={"error": "Validation error", "details": exc.errors(), "request_id": request_id},
                headers={"X-Request-Id": request_id},
            )

        @self.app.exception_handler(Exception)
        async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
            request_id = self._request_id(request)
            self.logger.error("request_id=%s unhandled_exception=%s", request_id, str(exc))
            self.logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "request_id": request_id},
                headers={"X-Request-Id": request_id},
            )

    async def chat(self, request: Request, payload: ChatRequest) -> StreamingResponse:
        """Handle SSE chat generation requests.

        Args:
            request: Incoming HTTP request.
            payload: Strict chat payload.

        Returns:
            StreamingResponse: SSE token stream.
        """
        request_id = self._request_id(request)
        client_id = self._client_ip(request)

        try:
            self.rate_limiter.check(client_id)
        except RateLimitError as exc:
            self.metrics.record_rate_limit_hit()
            retry_after = self.rate_limiter.get_retry_after(client_id)
            raise HTTPException(
                status_code=429,
                detail=exc.message,
                headers={"Retry-After": str(retry_after), "X-Request-Id": request_id},
            ) from exc

        try:
            skill = self.skills.get(payload.skill_id)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.message) from exc

        raw_messages = [message.model_dump() for message in payload.messages]
        try:
            sanitized_messages = self.validator.validate_messages(raw_messages)
        except InjectionError as exc:
            self.metrics.record_hallucination_guard()
            raise HTTPException(status_code=400, detail=exc.message) from exc
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.message) from exc

        started = time.perf_counter()
        try:
            token_stream = self.model_manager.generate_stream(
                messages=sanitized_messages,
                system=skill.system_prompt,
                skill=payload.skill_id,
            )
            iterator = SSETokenStream(
                token_stream=token_stream,
                metrics=self.metrics,
                skill_id=payload.skill_id,
                started_at=started,
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            headers = {
                "X-Response-Ms": str(elapsed_ms),
                "X-Request-Id": request_id,
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
            return StreamingResponse(iterator, media_type="text/event-stream", headers=headers)
        except ModelError as exc:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            self.metrics.record_request(skill_id=payload.skill_id, ms=elapsed_ms, error=True)
            raise HTTPException(status_code=500, detail="Model generation failed") from exc

    async def health(self, request: Request) -> HealthResponse:
        """Return health status and model runtime information.

        Args:
            request: Incoming request.

        Returns:
            HealthResponse: Current health payload.
        """
        _ = request
        model_stats = self.model_manager.get_stats()
        metrics_summary = self.metrics.get_summary()
        self.metrics.set_avg_tokens_per_sec(float(model_stats.get("avg_tokens_per_sec", 0.0)))

        return HealthResponse(
            status="ok",
            model=str(model_stats.get("model", "google/gemma-4-2b-it")),
            quantization=str(model_stats.get("quantization", "unknown")),
            hardware=self.hardware,
            model_load_ms=int(model_stats.get("model_load_ms", 0)),
            avg_tokens_per_sec=float(model_stats.get("avg_tokens_per_sec", 0.0)),
            uptime_seconds=int(metrics_summary.get("uptime_seconds", int(time.time() - self._started_at))),
            last_request_ms=int(metrics_summary.get("last_request_ms", 0)),
        )

    async def admin(self, request: Request) -> AdminResponse:
        """Return extended admin metrics for local monitoring.

        Args:
            request: Incoming request.

        Returns:
            AdminResponse: Extended health + metrics payload.
        """
        _ = request
        health_payload = await self.health(request)
        metrics_summary = self.metrics.get_summary()

        return AdminResponse(
            status=health_payload.status,
            model=health_payload.model,
            quantization=health_payload.quantization,
            hardware=health_payload.hardware,
            model_load_ms=health_payload.model_load_ms,
            avg_tokens_per_sec=health_payload.avg_tokens_per_sec,
            uptime_seconds=health_payload.uptime_seconds,
            last_request_ms=health_payload.last_request_ms,
            total_requests=int(metrics_summary.get("total_requests", 0)),
            errors=int(metrics_summary.get("errors", 0)),
            avg_response_ms=float(metrics_summary.get("avg_response_ms", 0.0)),
            requests_per_minute=float(metrics_summary.get("requests_per_minute", 0.0)),
            skill_usage=dict(metrics_summary.get("skill_usage", {})),
            hallucination_guards_triggered=int(
                metrics_summary.get("hallucination_guards_triggered", 0)
            ),
            rate_limit_hits=int(metrics_summary.get("rate_limit_hits", 0)),
        )

    async def list_skills(self, request: Request) -> list[Skill]:
        """Return supported skill definitions.

        Args:
            request: Incoming request.

        Returns:
            list[Skill]: List of available skills.
        """
        _ = request
        return self.skills.all()

    def _request_id(self, request: Request) -> str:
        """Read request ID from state or generate a fallback UUID.

        Args:
            request: Incoming request.

        Returns:
            str: Request identifier.
        """
        request_id = getattr(request.state, "request_id", "")
        if not request_id:
            return str(uuid.uuid4())
        return str(request_id)

    def _client_ip(self, request: Request) -> str:
        """Resolve best-effort client identifier from request metadata.

        Args:
            request: Incoming request.

        Returns:
            str: Client IP or localhost placeholder.
        """
        if request.client and request.client.host:
            return request.client.host
        return "127.0.0.1"


chatbot_app = ChatbotApp()
app = chatbot_app.build()
