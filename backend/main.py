from __future__ import annotations

import asyncio
import json
import os
import time
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from .hardware import HardwareProfile, detect_hardware
from .skills import SKILLS, get_skill_prompt, valid_skill_ids

APP_MODEL_NAME = "gemma-4-2b-it"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    skill: str = "chat"
    stream: bool = True

    @field_validator("skill")
    @classmethod
    def validate_skill(cls, value: str) -> str:
        if value not in valid_skill_ids():
            allowed = ", ".join(sorted(valid_skill_ids()))
            raise ValueError(f"Unknown skill '{value}'. Allowed skills: {allowed}")
        return value


class ModelBackend(Protocol):
    model_name: str
    quantization: str

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
    ) -> Iterable[str]:
        ...


class MLXModelBackend:
    def __init__(self, model_candidates: list[str], quantization: str) -> None:
        from mlx_lm import load, stream_generate  # type: ignore

        self.quantization = quantization
        self.model_name = APP_MODEL_NAME
        self._stream_generate = stream_generate
        self._source_model = ""

        last_error: Exception | None = None
        for model_id in model_candidates:
            if not model_id:
                continue
            try:
                model, tokenizer = load(
                    model_id,
                    tokenizer_config={"trust_remote_code": True},
                )
                self._model = model
                self._tokenizer = tokenizer
                self._source_model = model_id
                break
            except Exception as exc:  # pragma: no cover - depends on local model/env
                last_error = exc

        if not hasattr(self, "_model"):
            message = "Failed to load MLX model from candidates: " + ", ".join(model_candidates)
            if last_error is not None:
                message += f". Last error: {last_error}"
            raise RuntimeError(message)

    def _build_prompt(self, messages: list[dict[str, str]]) -> Any:
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
    ) -> Iterable[str]:
        prompt = self._build_prompt(messages)
        for response in self._stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
        ):
            chunk = getattr(response, "text", "")
            if chunk:
                yield chunk


class LlamaCppBackend:
    def __init__(self, gguf_path: str, quantization: str) -> None:
        from llama_cpp import Llama  # type: ignore

        self.quantization = quantization
        self.model_name = APP_MODEL_NAME
        self._source_model = gguf_path
        self._llm = Llama(
            model_path=gguf_path,
            n_ctx=4096,
            n_gpu_layers=-1,
            n_threads=max((os.cpu_count() or 4) - 1, 1),
            verbose=False,
        )

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
    ) -> Iterable[str]:
        stream = self._llm.create_chat_completion(
            messages=messages,
            stream=True,
            temperature=0.2,
            top_p=0.9,
            max_tokens=max_tokens,
        )
        for chunk in stream:
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            text = delta.get("content")
            if text:
                yield text


@dataclass
class RuntimeState:
    hardware: HardwareProfile | None = None
    model_backend: ModelBackend | None = None
    model_load_ms: int = 0
    started_at: float = field(default_factory=time.time)
    total_requests: int = 0
    errors: int = 0
    total_response_ms: int = 0
    last_request_ms: int = 0
    total_tokens: int = 0
    total_generation_seconds: float = 0.0
    skill_usage: Counter[str] = field(default_factory=Counter)
    hallucination_guards_triggered: int = 0
    generation_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _mlx_model_candidates(quantization: str) -> list[str]:
    if quantization == "INT4-mlx":
        return [
            os.getenv("MLX_INT4_MODEL_ID", "mlx-community/gemma-4-2b-it-4bit"),
            "google/gemma-4-2b-it",
        ]
    return [
        os.getenv("MLX_INT8_MODEL_ID", "mlx-community/gemma-4-2b-it-8bit"),
        "google/gemma-4-2b-it",
    ]


def _gguf_path_candidates() -> list[str]:
    return [
        os.getenv("GGUF_MODEL_PATH", "").strip(),
        "gemma-4-2b-it.Q4_K_M.gguf",
        "models/gemma-4-2b-it.Q4_K_M.gguf",
        "backend/models/gemma-4-2b-it.Q4_K_M.gguf",
    ]


def _resolve_gguf_path() -> str:
    candidates = [Path(path) for path in _gguf_path_candidates() if path]
    for path in candidates:
        if path.exists() and path.is_file():
            return str(path)
    raise RuntimeError(
        "No GGUF model file found. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def load_model_backend(hardware: HardwareProfile) -> tuple[ModelBackend, int]:
    started = time.perf_counter()

    if hardware.quantization in {"INT4-mlx", "INT8-mlx"}:
        backend: ModelBackend = MLXModelBackend(
            _mlx_model_candidates(hardware.quantization),
            hardware.quantization,
        )
    else:
        backend = LlamaCppBackend(_resolve_gguf_path(), hardware.quantization)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return backend, elapsed_ms


def _build_generation_messages(
    system_prompt: str,
    history: list[ChatMessage],
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend({"role": item.role, "content": item.content} for item in history)
    return messages


def _to_sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\\ndata: {json.dumps(payload)}\\n\\n"


def _health_payload(runtime: RuntimeState) -> dict[str, Any]:
    hardware = runtime.hardware
    if hardware is None:
        raise RuntimeError("Runtime hardware profile is unavailable")

    avg_tps = 0.0
    if runtime.total_generation_seconds > 0:
        avg_tps = runtime.total_tokens / runtime.total_generation_seconds

    return {
        "status": "ok",
        "model": APP_MODEL_NAME,
        "quantization": hardware.quantization,
        "hardware": {
            "chip": hardware.chip,
            "ram_total_gb": hardware.ram_total_gb,
            "ram_available_gb": round(hardware.ram_available_gb, 2),
            "cpu_cores": hardware.cpu_cores,
            "metal_gpu": hardware.metal_gpu,
        },
        "model_load_ms": runtime.model_load_ms,
        "avg_tokens_per_sec": round(avg_tps, 1),
        "uptime_seconds": int(time.time() - runtime.started_at),
        "last_request_ms": runtime.last_request_ms,
    }


def _admin_payload(runtime: RuntimeState) -> dict[str, Any]:
    payload = _health_payload(runtime)
    avg_response_ms = 0
    if runtime.total_requests > 0:
        avg_response_ms = int(runtime.total_response_ms / runtime.total_requests)

    payload.update(
        {
            "total_requests": runtime.total_requests,
            "errors": runtime.errors,
            "avg_response_ms": avg_response_ms,
            "skill_usage": {skill["id"]: runtime.skill_usage.get(skill["id"], 0) for skill in SKILLS},
            "hallucination_guards_triggered": runtime.hallucination_guards_triggered,
        }
    )
    return payload


def _ensure_code_block(text: str) -> str:
    if "```" in text:
        return text
    stripped = text.strip()
    if not stripped:
        return "```python\n# No code generated\n```"
    return f"```python\\n{stripped}\\n```"


async def _collect_chunks(
    backend: ModelBackend,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
) -> list[str]:
    loop = asyncio.get_running_loop()

    def _run_sync() -> list[str]:
        return [chunk for chunk in backend.stream_chat(messages, max_tokens=max_tokens)]

    return await loop.run_in_executor(None, _run_sync)


def create_app() -> FastAPI:
    runtime = RuntimeState()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        runtime.hardware = detect_hardware()
        model_backend, model_load_ms = load_model_backend(runtime.hardware)
        runtime.model_backend = model_backend
        runtime.model_load_ms = model_load_ms

        print("[boot] chip=", runtime.hardware.chip, sep="")
        print("[boot] metal_gpu=", runtime.hardware.metal_gpu, sep="")
        print("[boot] quantization=", runtime.hardware.quantization, sep="")
        print("[boot] model_load_ms=", runtime.model_load_ms, sep="")
        yield

    app = FastAPI(title="Gemma 4 Local Chatbot", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["x-response-ms"],
    )

    app.state.runtime = runtime

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return _health_payload(runtime)

    @app.get("/admin")
    async def admin() -> dict[str, Any]:
        return _admin_payload(runtime)

    @app.post("/chat")
    async def chat(body: ChatRequest):
        if runtime.model_backend is None:
            raise HTTPException(status_code=503, detail="Model is not ready")

        system_prompt = get_skill_prompt(body.skill)
        conversation = _build_generation_messages(system_prompt, body.messages)

        async with runtime.generation_lock:
            started = time.perf_counter()
            runtime.total_requests += 1
            runtime.skill_usage[body.skill] += 1

            try:
                chunks = await _collect_chunks(
                    runtime.model_backend,
                    conversation,
                    max_tokens=512,
                )
                text = "".join(chunks)
                if body.skill == "code":
                    text = _ensure_code_block(text)
                    chunks = [text]
            except Exception as exc:
                runtime.errors += 1
                raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

            response_ms = int((time.perf_counter() - started) * 1000)
            runtime.last_request_ms = response_ms
            runtime.total_response_ms += response_ms
            runtime.total_tokens += max(len(chunks), 1)
            runtime.total_generation_seconds += max(response_ms / 1000.0, 0.001)

        headers = {"x-response-ms": str(response_ms)}

        if not body.stream:
            return JSONResponse(content={"text": text, "skill": body.skill}, headers=headers)

        async def sse_stream() -> Iterable[str]:
            for chunk in chunks:
                yield _to_sse("token", {"token": chunk})
                await asyncio.sleep(0)
            yield _to_sse("done", {"response_ms": response_ms})

        return StreamingResponse(
            sse_stream(),
            media_type="text/event-stream",
            headers=headers,
        )

    return app


app = create_app()
