from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator

from .errors import ModelError
from .hardware import HardwareInfo


class QuantizationStrategy(ABC):
    """Abstract interface for model loading and token streaming."""

    @abstractmethod
    def load_model(self, model_id: str) -> None:
        """Load model weights and tokenizer resources.

        Args:
            model_id: Hugging Face model ID or local file path.

        Returns:
            None.
        """

    @abstractmethod
    def generate(self, messages: list[dict], system: str) -> Generator[str, None, None]:
        """Stream generated tokens for a message history.

        Args:
            messages: Sanitized chat history.
            system: System prompt for the active skill.

        Returns:
            Generator[str, None, None]: Streamed output tokens.
        """


class MLXQuantization(QuantizationStrategy):
    """Apple Silicon MLX strategy for INT4/INT8 quantized inference."""

    def __init__(self, precision: str = "int4") -> None:
        """Initialize MLX strategy metadata.

        Args:
            precision: Preferred quantization precision (`int4` or `int8`).

        Returns:
            None.
        """
        self.precision: str = precision
        self.backend_name: str = "mlx"
        self.quantization: str = precision.upper()
        self.model_name: str = ""
        self._model: object | None = None
        self._tokenizer: object | None = None
        self._stream_generate: object | None = None
        self._mock_mode: bool = os.getenv("GEMMA_SKIP_MODEL_LOAD", "0") == "1"

    def load_model(self, model_id: str) -> None:
        """Load MLX model and tokenizer lazily.

        Args:
            model_id: MLX-compatible model identifier.

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

            loaded_model, loaded_tokenizer = load(model_id)
            self._model = loaded_model
            self._tokenizer = loaded_tokenizer
            self._stream_generate = stream_generate
        except Exception as exc:
            raise ModelError(
                message="Failed to load MLX model",
                status_code=500,
                log_detail=str(exc),
            ) from exc

    def generate(self, messages: list[dict], system: str) -> Generator[str, None, None]:
        """Generate tokens with MLX chat templating.

        Args:
            messages: Sanitized conversation history.
            system: Active skill system prompt.

        Returns:
            Generator[str, None, None]: Streamed response tokens.
        """
        if self._model is None or self._tokenizer is None:
            raise ModelError(
                message="Model is not loaded",
                status_code=500,
                log_detail="MLX model requested before load_model",
            )

        if self._mock_mode:
            for token in ["Hello", " world", "!"]:
                yield token
            return

        tokenizer_object = self._tokenizer
        model_object = self._model
        stream_generate_callable = self._stream_generate
        if stream_generate_callable is None:
            raise ModelError(
                message="MLX generator unavailable",
                status_code=500,
                log_detail="stream_generate was not initialized",
            )

        payload = self._build_messages(messages, system)
        try:
            prompt = getattr(tokenizer_object, "apply_chat_template")(
                payload,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            prompt = getattr(tokenizer_object, "apply_chat_template")(
                payload,
                add_generation_prompt=True,
            )

        try:
            for output in stream_generate_callable(
                model_object,
                tokenizer_object,
                prompt=prompt,
                max_tokens=1024,
            ):
                text = getattr(output, "text", "")
                if text:
                    yield text
        except Exception as exc:
            raise ModelError(
                message="Token generation failed",
                status_code=500,
                log_detail=str(exc),
            ) from exc

    def _build_messages(
        self, messages: list[dict], system: str
    ) -> list[dict[str, str]]:
        """Compose system and user/assistant turns for the model template.

        Args:
            messages: Conversation messages.
            system: System prompt text.

        Returns:
            list[dict[str, str]]: Template-compatible message payload.
        """
        payload: list[dict[str, str]] = [{"role": "system", "content": system}]
        for message in messages:
            payload.append(
                {
                    "role": str(message.get("role", "user")),
                    "content": str(message.get("content", "")),
                }
            )
        return payload


class LlamaCppQuantization(QuantizationStrategy):
    """Llama.cpp strategy for Intel and non-Metal Q4_K_M inference."""

    def __init__(self, quant: str = "Q4_K_M") -> None:
        """Initialize llama.cpp strategy metadata.

        Args:
            quant: GGUF quantization label.

        Returns:
            None.
        """
        self.quant: str = quant
        self.backend_name: str = "llama.cpp"
        self.quantization: str = quant
        self.model_name: str = ""
        self._llm: object | None = None
        self._mock_mode: bool = os.getenv("GEMMA_SKIP_MODEL_LOAD", "0") == "1"

    def load_model(self, model_id: str) -> None:
        """Load GGUF model through llama-cpp-python.

        Args:
            model_id: Path to GGUF model.

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
                log_detail=f"Missing file at {model_path}",
            )

        try:
            from llama_cpp import Llama  # type: ignore

            threads = max((os.cpu_count() or 4) - 1, 1)
            self._llm = Llama(
                model_path=str(model_path),
                n_ctx=4096,
                n_threads=threads,
                n_gpu_layers=0,
                verbose=False,
            )
        except Exception as exc:
            raise ModelError(
                message="Failed to load llama.cpp model",
                status_code=500,
                log_detail=str(exc),
            ) from exc

    def generate(self, messages: list[dict], system: str) -> Generator[str, None, None]:
        """Generate streamed tokens using llama.cpp chat completion API.

        Args:
            messages: Sanitized conversation history.
            system: Active skill system prompt.

        Returns:
            Generator[str, None, None]: Streamed response tokens.
        """
        if self._llm is None:
            raise ModelError(
                message="Model is not loaded",
                status_code=500,
                log_detail="llama.cpp model requested before load_model",
            )

        if self._mock_mode:
            for token in ["Hello", " world", "!"]:
                yield token
            return

        llm = self._llm
        payload = self._build_messages(messages, system)

        try:
            stream = getattr(llm, "create_chat_completion")(
                messages=payload,
                stream=True,
                temperature=0.2,
                top_p=0.9,
                max_tokens=1024,
            )
            for chunk in stream:
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                token = delta.get("content")
                if token:
                    yield str(token)
        except Exception as exc:
            raise ModelError(
                message="Token generation failed",
                status_code=500,
                log_detail=str(exc),
            ) from exc

    def _build_messages(
        self, messages: list[dict], system: str
    ) -> list[dict[str, str]]:
        """Compose system and conversation messages for llama.cpp.

        Args:
            messages: Conversation messages.
            system: System prompt text.

        Returns:
            list[dict[str, str]]: Chat payload for llama.cpp.
        """
        payload: list[dict[str, str]] = [{"role": "system", "content": system}]
        for message in messages:
            payload.append(
                {
                    "role": str(message.get("role", "user")),
                    "content": str(message.get("content", "")),
                }
            )
        return payload


class QuantizationSelector:
    """Selects the optimal quantization strategy from detected hardware."""

    def __init__(self) -> None:
        """Initialize selector constants."""
        self._minimum_int8_ram_gb: float = 8.0
        self._minimum_int4_ram_gb: float = 16.0

    def select(self, hw: HardwareInfo) -> QuantizationStrategy:
        """Select a quantization strategy based on machine capabilities.

        Args:
            hw: Detected hardware profile.

        Returns:
            QuantizationStrategy: Selected inference strategy.
        """
        is_apple_silicon = self._is_apple_silicon(hw)
        if is_apple_silicon and hw.ram_total_gb >= self._minimum_int4_ram_gb:
            return MLXQuantization(precision="int4")
        if (
            is_apple_silicon
            and self._minimum_int8_ram_gb <= hw.ram_total_gb < self._minimum_int4_ram_gb
        ):
            return MLXQuantization(precision="int8")
        return LlamaCppQuantization(quant="Q4_K_M")

    def _is_apple_silicon(self, hw: HardwareInfo) -> bool:
        """Infer Apple Silicon status from chip string and Metal support.

        Args:
            hw: Hardware profile.

        Returns:
            bool: True when hardware appears to be Apple Silicon.
        """
        chip_lower = hw.chip.lower()
        apple_tokens = ["apple", "m1", "m2", "m3", "m4"]
        return hw.metal_gpu and any(token in chip_lower for token in apple_tokens)
