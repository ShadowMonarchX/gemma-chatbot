from __future__ import annotations

import time
from typing import ClassVar, Generator

from .errors import ModelError
from .quantization import LlamaCppQuantization, MLXQuantization, QuantizationStrategy


class ModelManager:
    """Singleton manager that loads one model strategy and streams generations."""

    _instance: ClassVar["ModelManager | None"] = None

    def __init__(self) -> None:
        """Initialize model manager runtime state."""
        self._strategy: QuantizationStrategy | None = None
        self._loaded: bool = False
        self._model_name: str = ""
        self._quantization: str = ""
        self._model_load_ms: int = 0
        self._last_tokens_per_sec: float = 0.0
        self._avg_tokens_per_sec: float = 0.0
        self._generation_runs: int = 0

    @classmethod
    def get_instance(cls) -> "ModelManager":
        """Return the process-wide singleton model manager.

        Returns:
            ModelManager: Singleton instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self, strategy: QuantizationStrategy) -> None:
        """Load model weights for a selected strategy once.

        Args:
            strategy: Selected quantization strategy.

        Returns:
            None.
        """
        if self._loaded:
            return

        self._strategy = strategy
        model_id = self._resolve_model_id(strategy)
        started = time.perf_counter()
        strategy.load_model(model_id)
        self._model_load_ms = int((time.perf_counter() - started) * 1000)
        self._model_name = model_id
        self._quantization = self._resolve_quantization(strategy)
        self._loaded = True

        warmup_started = time.perf_counter()
        warmup_tokens = 0
        for token in strategy.generate(
            [{"role": "user", "content": "Say hello in one short sentence."}],
            "You are a concise assistant.",
        ):
            warmup_tokens += max(len(str(token).split()), 1)
            if warmup_tokens >= 4:
                break
        elapsed = max(time.perf_counter() - warmup_started, 0.001)
        self._last_tokens_per_sec = round(warmup_tokens / elapsed, 2)
        self._avg_tokens_per_sec = self._last_tokens_per_sec

    def generate_stream(
        self,
        messages: list[dict],
        system: str,
        skill: str,
    ) -> Generator[str, None, None]:
        """Generate a token stream for the given chat payload.

        Args:
            messages: Sanitized history.
            system: System prompt for active skill.
            skill: Skill identifier.

        Returns:
            Generator[str, None, None]: Stream of output text tokens.
        """
        if not self._loaded or self._strategy is None:
            raise ModelError(
                message="Model not loaded",
                status_code=500,
                log_detail="generate_stream called before load",
            )

        started = time.perf_counter()
        token_count = 0
        for token in self._strategy.generate(messages=messages, system=system):
            text_token = str(token)
            token_count += max(len(text_token.split()), 1)
            yield text_token

        elapsed = max(time.perf_counter() - started, 0.001)
        run_tps = round(token_count / elapsed, 2)
        self._last_tokens_per_sec = run_tps
        self._generation_runs += 1
        weighted_sum = self._avg_tokens_per_sec * (self._generation_runs - 1)
        self._avg_tokens_per_sec = round((weighted_sum + run_tps) / self._generation_runs, 2)

    def get_stats(self) -> dict:
        """Return model runtime metrics.

        Returns:
            dict: Model metadata and throughput metrics.
        """
        return {
            "model": self._model_name,
            "quantization": self._quantization,
            "model_load_ms": self._model_load_ms,
            "last_tokens_per_sec": self._last_tokens_per_sec,
            "avg_tokens_per_sec": self._avg_tokens_per_sec,
        }

    def _resolve_model_id(self, strategy: QuantizationStrategy) -> str:
        """Choose the model identifier/path for a strategy.

        Args:
            strategy: Active quantization strategy.

        Returns:
            str: Model ID or GGUF path.
        """
        if isinstance(strategy, MLXQuantization):
            return "google/gemma-4-2b-it"
        if isinstance(strategy, LlamaCppQuantization):
            return "models/gemma-4-2b-it.Q4_K_M.gguf"
        return "google/gemma-4-2b-it"

    def _resolve_quantization(self, strategy: QuantizationStrategy) -> str:
        """Extract quantization label from a strategy.

        Args:
            strategy: Active quantization strategy.

        Returns:
            str: Quantization label.
        """
        if isinstance(strategy, MLXQuantization):
            return strategy.quantization
        if isinstance(strategy, LlamaCppQuantization):
            return strategy.quantization
        return "UNKNOWN"


model_manager = ModelManager.get_instance()
