from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar, Generator

from .config import settings
from .errors import ConfigurationError, ModelError, ValidationError
from .hardware import HardwareInfo
from .quantization import LlamaCppQuantization, MLXQuantization, QuantizationStrategy


class ModelSpec:
    """Configuration object describing one selectable model profile."""

    def __init__(
        self,
        model_id: str,
        label: str,
        backend: str,
        source: str,
        quantization: str,
        description: str,
        is_default: bool,
        available: bool,
        alias_of: str | None = None,
    ) -> None:
        """Initialize one model specification.

        Args:
            model_id: Runtime model selector key.
            label: Human-readable model label.
            backend: Runtime backend name (mlx or llama.cpp).
            source: Backend model source path or model ID.
            quantization: Quantization label.
            description: Short description of model behavior.
            is_default: Whether this model is the startup default.
            available: Whether model is available on this host.
            alias_of: Canonical model ID when this is an alias profile.

        Returns:
            None.
        """
        self.model_id: str = model_id
        self.label: str = label
        self.backend: str = backend
        self.source: str = source
        self.quantization: str = quantization
        self.description: str = description
        self.is_default: bool = is_default
        self.available: bool = available
        self.alias_of: str | None = alias_of

    def to_dict(self) -> dict:
        """Serialize model specification for API output.

        Returns:
            dict: Model metadata.
        """
        return {
            "id": self.model_id,
            "label": self.label,
            "backend": self.backend,
            "source": self.source,
            "quantization": self.quantization,
            "available": self.available,
            "default": self.is_default,
            "description": self.description,
            "alias_of": self.alias_of,
        }


class ModelManager:
    """Singleton manager that lazily loads, caches, and switches local models."""

    _instance: ClassVar["ModelManager | None"] = None

    def __init__(self) -> None:
        """Initialize runtime state and caches for model execution."""
        self._hardware: HardwareInfo | None = None
        self._base_strategy: QuantizationStrategy | None = None
        self._model_catalog: dict[str, ModelSpec] = {}
        self._model_cache: dict[str, QuantizationStrategy] = {}
        self._model_load_times_ms: dict[str, int] = {}
        self._active_model_id: str = ""
        self._active_strategy: QuantizationStrategy | None = None
        self._last_tokens_per_sec: float = 0.0
        self._avg_tokens_per_sec: float = 0.0
        self._generation_runs: int = 0

    @classmethod
    def get_instance(cls) -> "ModelManager":
        """Return process-wide singleton model manager.

        Returns:
            ModelManager: Singleton instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def configure(self, hardware: HardwareInfo) -> None:
        """Store detected hardware and rebuild model catalog.

        Args:
            hardware: Detected runtime hardware profile.

        Returns:
            None.
        """
        self._hardware = hardware
        self._build_model_catalog()

    def load(self, strategy: QuantizationStrategy) -> None:
        """Set base strategy and eagerly load default model.

        Args:
            strategy: Strategy chosen from hardware selector.

        Returns:
            None.
        """
        if self._hardware is None:
            raise ConfigurationError(
                message="Hardware must be configured before model loading",
                status_code=500,
                log_detail="configure() not called before load()",
            )

        self._base_strategy = strategy
        self._build_model_catalog()
        default_model_id = settings.default_model

        if default_model_id not in self._model_catalog:
            fallback = self._first_available_model_id()
            default_model_id = fallback

        self.switch_model(default_model_id)

    def switch_model(self, model_id: str) -> None:
        """Switch active model at runtime, loading and caching on first use.

        Args:
            model_id: Target model selector key.

        Returns:
            None.
        """
        spec = self._get_model_spec(model_id=model_id)
        if not spec.available:
            raise ValidationError(
                message="Requested model is not available on this machine",
                status_code=422,
                log_detail=f"model_id={model_id}",
            )

        if self._active_model_id == model_id and self._active_strategy is not None:
            return

        if model_id in self._model_cache:
            self._active_strategy = self._model_cache[model_id]
            self._active_model_id = model_id
            return

        strategy = self._create_strategy(spec=spec)
        started = time.perf_counter()
        strategy.load_model(spec.source)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        self._model_cache[model_id] = strategy
        self._model_load_times_ms[model_id] = elapsed_ms
        self._active_strategy = strategy
        self._active_model_id = model_id

        warmup_started = time.perf_counter()
        warmup_tokens = 0
        for token in strategy.generate(
            [{"role": "user", "content": "Reply with one short greeting."}],
            "You are a concise assistant.",
        ):
            warmup_tokens += max(len(str(token).split()), 1)
            if warmup_tokens >= 4:
                break

        elapsed = max(time.perf_counter() - warmup_started, 0.001)
        self._last_tokens_per_sec = round(warmup_tokens / elapsed, 2)
        self._avg_tokens_per_sec = self._last_tokens_per_sec
        self._generation_runs = 1

    def generate_stream(
        self,
        messages: list[dict],
        system: str,
        skill: str,
        model_id: str,
    ) -> Generator[str, None, None]:
        """Generate a token stream using the requested model.

        Args:
            messages: Sanitized chat history.
            system: Active skill system prompt.
            skill: Skill identifier.
            model_id: Model selector key.

        Returns:
            Generator[str, None, None]: Token stream.
        """
        _ = skill
        self.switch_model(model_id=model_id)
        strategy = self._active_strategy

        if strategy is None:
            raise ModelError(
                message="Model not loaded",
                status_code=500,
                log_detail="active_strategy missing after switch_model",
            )

        started = time.perf_counter()
        token_count = 0
        for token in strategy.generate(messages=messages, system=system):
            token_text = str(token)
            token_count += max(len(token_text.split()), 1)
            yield token_text

        elapsed = max(time.perf_counter() - started, 0.001)
        run_tps = round(token_count / elapsed, 2)
        self._last_tokens_per_sec = run_tps

        self._generation_runs += 1
        weighted_sum = self._avg_tokens_per_sec * (self._generation_runs - 1)
        self._avg_tokens_per_sec = round(
            (weighted_sum + run_tps) / self._generation_runs, 2
        )

    def get_stats(self) -> dict:
        """Return active model and generation performance statistics.

        Returns:
            dict: Runtime model metadata and performance stats.
        """
        active_id = self._active_model_id or self._first_available_model_id()
        spec = self._model_catalog.get(active_id)

        if spec is None:
            return {
                "model_id": "",
                "model_label": "",
                "model": "",
                "backend": "",
                "quantization": "",
                "model_load_ms": 0,
                "last_tokens_per_sec": self._last_tokens_per_sec,
                "avg_tokens_per_sec": self._avg_tokens_per_sec,
            }

        return {
            "model_id": spec.model_id,
            "model_label": spec.label,
            "model": spec.source,
            "backend": spec.backend,
            "quantization": spec.quantization,
            "model_load_ms": int(self._model_load_times_ms.get(spec.model_id, 0)),
            "last_tokens_per_sec": self._last_tokens_per_sec,
            "avg_tokens_per_sec": self._avg_tokens_per_sec,
        }

    def list_models(self) -> list[dict]:
        """Return model catalog for frontend model selector UI.

        Returns:
            list[dict]: List of model metadata dictionaries.
        """
        ordered_ids = ["gemma-2b", "gemma-e2b", "gemma-e4b"]
        output: list[dict] = []
        for model_id in ordered_ids:
            spec = self._model_catalog.get(model_id)
            if spec is not None:
                output.append(spec.to_dict())
        return output

    def get_active_model_id(self) -> str:
        """Return active model ID.

        Returns:
            str: Active model key.
        """
        if self._active_model_id:
            return self._active_model_id
        return self._first_available_model_id()

    def is_model_known(self, model_id: str) -> bool:
        """Return whether model exists in the catalog.

        Args:
            model_id: Model selector key.

        Returns:
            bool: True when catalog contains this model key.
        """
        return model_id in self._model_catalog

    def _get_model_spec(self, model_id: str) -> ModelSpec:
        """Resolve and validate a model spec.

        Args:
            model_id: Model selector key.

        Returns:
            ModelSpec: Resolved model spec.
        """
        spec = self._model_catalog.get(model_id)
        if spec is None:
            raise ValidationError(
                message="Unknown model ID",
                status_code=422,
                log_detail=f"model_id={model_id}",
            )
        return spec

    def _first_available_model_id(self) -> str:
        """Return first available model ID from the catalog.

        Returns:
            str: Available model key.
        """
        for model_id in ["gemma-2b", "gemma-e2b", "gemma-e4b"]:
            spec = self._model_catalog.get(model_id)
            if spec and spec.available:
                return model_id

        raise ConfigurationError(
            message="No usable models found for current runtime",
            status_code=500,
            log_detail="model catalog has no available entries",
        )

    def _build_model_catalog(self) -> None:
        """Build model catalog based on detected hardware and settings.

        Returns:
            None.
        """
        if self._hardware is None:
            return

        default_model = settings.default_model
        model_root = Path(settings.model_path)
        use_mlx_catalog = (
            self._hardware.is_apple_silicon
            and self._hardware.metal_gpu
            and not isinstance(self._base_strategy, LlamaCppQuantization)
        )

        if use_mlx_catalog:
            self._model_catalog = {
                "gemma-2b": ModelSpec(
                    model_id="gemma-2b",
                    label="Gemma 2B",
                    backend="mlx",
                    source="google/gemma-2b-it",
                    quantization=self._mlx_quantization(),
                    description="Fast baseline model for local chat.",
                    is_default=default_model == "gemma-2b",
                    available=True,
                ),
                "gemma-e2b": ModelSpec(
                    model_id="gemma-e2b",
                    label="Gemma E2B",
                    backend="mlx",
                    source="google/gemma-2b-it",
                    quantization=self._mlx_quantization(),
                    description="Efficient profile mapped to MLX Gemma 2B.",
                    is_default=default_model == "gemma-e2b",
                    available=True,
                    alias_of="gemma-2b",
                ),
                "gemma-e4b": ModelSpec(
                    model_id="gemma-e4b",
                    label="Gemma E4B",
                    backend="mlx",
                    source="google/gemma-2b-it",
                    quantization=self._mlx_quantization(),
                    description="Quality profile mapped to MLX Gemma 2B.",
                    is_default=default_model == "gemma-e4b",
                    available=True,
                    alias_of="gemma-2b",
                ),
            }
            return

        backend_name = "llama.cpp"
        quantization = "Q4_K_M"
        self._model_catalog = {
            "gemma-2b": ModelSpec(
                model_id="gemma-2b",
                label="Gemma 2B",
                backend=backend_name,
                source=str(model_root / "gemma-2b-it.Q4_K_M.gguf"),
                quantization=quantization,
                description="Fast baseline GGUF model.",
                is_default=default_model == "gemma-2b",
                available=(model_root / "gemma-2b-it.Q4_K_M.gguf").exists()
                or settings.skip_model_load,
            ),
            "gemma-e2b": ModelSpec(
                model_id="gemma-e2b",
                label="Gemma E2B",
                backend=backend_name,
                source=str(model_root / "gemma-e2b.Q4_K_M.gguf"),
                quantization=quantization,
                description="Efficiency-focused GGUF model.",
                is_default=default_model == "gemma-e2b",
                available=(model_root / "gemma-e2b.Q4_K_M.gguf").exists()
                or settings.skip_model_load,
            ),
            "gemma-e4b": ModelSpec(
                model_id="gemma-e4b",
                label="Gemma E4B",
                backend=backend_name,
                source=str(model_root / "gemma-e4b.Q4_K_M.gguf"),
                quantization=quantization,
                description="Higher-quality GGUF model.",
                is_default=default_model == "gemma-e4b",
                available=(model_root / "gemma-e4b.Q4_K_M.gguf").exists()
                or settings.skip_model_load,
            ),
        }

    def _create_strategy(self, spec: ModelSpec) -> QuantizationStrategy:
        """Create backend strategy instance for a model specification.

        Args:
            spec: Model specification object.

        Returns:
            QuantizationStrategy: Runtime strategy ready for loading.
        """
        if spec.backend == "mlx":
            precision = spec.quantization.lower()
            return MLXQuantization(precision=precision, max_tokens=settings.max_tokens)

        gpu_layers = 0
        if self._hardware and self._hardware.cuda_gpu:
            gpu_layers = 35

        return LlamaCppQuantization(
            quant=spec.quantization,
            max_tokens=settings.max_tokens,
            n_gpu_layers=gpu_layers,
        )

    def _mlx_quantization(self) -> str:
        """Infer MLX quantization tier from hardware and selected base strategy.

        Returns:
            str: MLX quantization label.
        """
        if isinstance(self._base_strategy, MLXQuantization):
            return self._base_strategy.quantization

        if self._hardware and self._hardware.ram_total_gb >= 16.0:
            return "INT4"
        return "INT8"


model_manager = ModelManager.get_instance()
