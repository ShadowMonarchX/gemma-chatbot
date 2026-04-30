from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator

from .errors import ModelError
from .hardware import HardwareInfo


class QuantizationStrategy(ABC):
    """Abstract inference strategy interface for model loading and streaming."""

    backend_name: str
    quantization: str

    @abstractmethod
    def load_model(self, model_id: str) -> None:
        """Load model artifacts into memory.

        Args:
            model_id: Model ID (remote) or file path (local), depending on backend.

        Returns:
            None.
        """

    @abstractmethod
    def generate(self, messages: list[dict], system: str) -> Generator[str, None, None]:
        """Stream response tokens for a chat payload.

        Args:
            messages: Sanitized conversation history.
            system: Skill system prompt.

        Returns:
            Generator[str, None, None]: Token stream.
        """


class MLXQuantization(QuantizationStrategy):
    """MLX-backed strategy for Apple Silicon acceleration via Metal."""

    def __init__(self, precision: str = "int4", max_tokens: int = 512) -> None:
        """Initialize MLX strategy settings.

        Args:
            precision: Preferred precision (`int4` or `int8`).
            max_tokens: Maximum generation token count.

        Returns:
            None.
        """
        self.precision: str = precision
        self.max_tokens: int = max_tokens
        self.backend_name: str = "mlx"
        self.quantization: str = precision.upper()
        self.model_name: str = ""
        self._model: object | None = None
        self._tokenizer: object | None = None
        self._stream_generate: object | None = None
        self._mock_mode: bool = (
            os.getenv("GEMMA_SKIP_MODEL_LOAD", "0") == "1"
            or os.getenv("SKIP_MODEL_LOAD", "0") == "1"
        )

    def load_model(self, model_id: str) -> None:
        """Load an MLX model and tokenizer.

        Args:
            model_id: Hugging Face model ID for MLX.

        Returns:
            None.
        """
        self.model_name = model_id
        if self._mock_mode:
            self._model = object()
            self._tokenizer = object()
            self._stream_generate = object()
            return

        try:
            from mlx_lm import load, stream_generate  # type: ignore

            model, tokenizer = load(model_id)
            self._model = model
            self._tokenizer = tokenizer
            self._stream_generate = stream_generate
        except Exception as exc:
            raise ModelError(
                message="Failed to load MLX model",
                status_code=500,
                log_detail=str(exc),
            ) from exc

    def generate(self, messages: list[dict], system: str) -> Generator[str, None, None]:
        """Generate response tokens using MLX streaming APIs.

        Args:
            messages: Sanitized conversation history.
            system: Skill system prompt.

        Returns:
            Generator[str, None, None]: Token stream.
        """
        if self._model is None or self._tokenizer is None:
            raise ModelError(
                message="Model is not loaded",
                status_code=500,
                log_detail="MLX strategy used before load_model",
            )

        if self._mock_mode:
            for token in ["Hello", " world", "!"]:
                yield token
            return

        stream_generate_callable = self._stream_generate
        if stream_generate_callable is None:
            raise ModelError(
                message="MLX generator unavailable",
                status_code=500,
                log_detail="stream_generate not initialized",
            )

        payload = self._build_messages(messages=messages, system=system)
        tokenizer = self._tokenizer
        model = self._model

        try:
            prompt = getattr(tokenizer, "apply_chat_template")(
                payload,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            prompt = getattr(tokenizer, "apply_chat_template")(
                payload,
                add_generation_prompt=True,
            )

        try:
            stream = stream_generate_callable(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
            )
            for chunk in stream:
                text = str(getattr(chunk, "text", ""))
                if text:
                    yield text
        except Exception as exc:
            raise ModelError(
                message="Token generation failed",
                status_code=500,
                log_detail=str(exc),
            ) from exc

    # AFTER (gemma-compatible)
    def _build_messages(self, messages, system):
        payload: list[dict[str, str]] = []
        system_injected = False
        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            if not system_injected and role == "user":
                content = f"{system}\n\n{content}" if system else content
                system_injected = True
            payload.append({"role": role, "content": content})

        # Edge case: no user message yet — inject system as first user turn
        if not system_injected and system:
            payload.insert(0, {"role": "user", "content": system})

        return payload


class LlamaCppQuantization(QuantizationStrategy):
    """Llama.cpp strategy for GGUF execution on CUDA or CPU fallback."""

    def __init__(
        self,
        quant: str = "Q4_K_M",
        max_tokens: int = 512,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
    ) -> None:
        """Initialize llama.cpp strategy options.

        Args:
            quant: GGUF quantization label.
            max_tokens: Maximum generation token count.
            n_ctx: Context window length.
            n_gpu_layers: Number of layers offloaded to GPU.

        Returns:
            None.
        """
        self.quant: str = quant
        self.max_tokens: int = max_tokens
        self.n_ctx: int = n_ctx
        self.n_gpu_layers: int = n_gpu_layers
        self.backend_name: str = "llama.cpp"
        self.quantization: str = quant
        self.model_name: str = ""
        self._llm: object | None = None
        self._mock_mode: bool = (
            os.getenv("GEMMA_SKIP_MODEL_LOAD", "0") == "1"
            or os.getenv("SKIP_MODEL_LOAD", "0") == "1"
        )

    def load_model(self, model_id: str) -> None:
        """Load a GGUF model through llama-cpp-python.

        Args:
            model_id: GGUF file path.

        Returns:
            None.
        """
        self.model_name = model_id
        if self._mock_mode:
            self._llm = object()
            return

        model_path = Path(model_id)
        if not model_path.exists():
            raise ModelError(
                message="GGUF model file not found",
                status_code=500,
                log_detail=f"missing_file={model_path}",
            )

        try:
            from llama_cpp import Llama  # type: ignore

            threads = max((os.cpu_count() or 4) - 1, 1)
            self._llm = Llama(
                model_path=str(model_path),
                n_ctx=self.n_ctx,
                n_threads=threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
            )
        except Exception as exc:
            raise ModelError(
                message="Failed to load llama.cpp model",
                status_code=500,
                log_detail=str(exc),
            ) from exc

    def generate(self, messages: list[dict], system: str) -> Generator[str, None, None]:
        """Generate response tokens from a loaded GGUF model.

        Args:
            messages: Sanitized conversation history.
            system: Skill system prompt.

        Returns:
            Generator[str, None, None]: Token stream.
        """
        if self._llm is None:
            raise ModelError(
                message="Model is not loaded",
                status_code=500,
                log_detail="llama.cpp strategy used before load_model",
            )

        if self._mock_mode:
            for token in ["Hello", " world", "!"]:
                yield token
            return

        payload = self._build_messages(messages=messages, system=system)
        llm = self._llm

        try:
            stream = getattr(llm, "create_chat_completion")(
                messages=payload,
                stream=True,
                max_tokens=self.max_tokens,
                temperature=0.2,
                top_p=0.9,
            )
            for chunk in stream:
                choices = chunk.get("choices") or []
                if not choices:
                    continue

                delta = choices[0].get("delta") or {}
                if "content" in delta and delta["content"]:
                    yield str(delta["content"])
                    continue

                text = choices[0].get("text")
                if text:
                    yield str(text)
        except Exception as exc:
            raise ModelError(
                message="Token generation failed",
                status_code=500,
                log_detail=str(exc),
            ) from exc

    # def _build_messages(
    #     self, messages: list[dict], system: str
    # ) -> list[dict[str, str]]:
    #     """Build chat payload expected by llama.cpp chat completion API.

    #     Args:
    #         messages: Sanitized chat messages.
    #         system: Skill system prompt.

    #     Returns:
    #         list[dict[str, str]]: Ordered chat payload.
    #     """
    #     payload: list[dict[str, str]] = [{"role": "system", "content": system}]
    #     for message in messages:
    #         payload.append(
    #             {
    #                 "role": str(message.get("role", "user")),
    #                 "content": str(message.get("content", "")),
    #             }
    #         )
    #     return payload


class QuantizationSelector:
    """Select the most suitable inference backend for current hardware."""

    def __init__(self) -> None:
        """Initialize threshold values for backend selection."""
        self._minimum_int4_ram_gb: float = 16.0
        self._minimum_int8_ram_gb: float = 8.0
        self._cuda_gpu_layers: int = 35

    def select(self, hw: HardwareInfo) -> QuantizationStrategy:
        """Choose MLX, CUDA-offloaded llama.cpp, or CPU llama.cpp.

        Args:
            hw: Detected hardware profile.

        Returns:
            QuantizationStrategy: Strategy optimized for this host.
        """
        if hw.is_apple_silicon and hw.metal_gpu:
            if hw.ram_total_gb >= self._minimum_int4_ram_gb:
                return MLXQuantization(precision="int4")
            if self._minimum_int8_ram_gb <= hw.ram_total_gb < self._minimum_int4_ram_gb:
                return MLXQuantization(precision="int8")
            return MLXQuantization(precision="int8")

        if hw.cuda_gpu:
            return LlamaCppQuantization(
                quant="Q4_K_M", n_gpu_layers=self._cuda_gpu_layers
            )

        return LlamaCppQuantization(quant="Q4_K_M", n_gpu_layers=0)
