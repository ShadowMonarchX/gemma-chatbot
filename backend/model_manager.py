from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, ClassVar, Generator

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
    _ordered_model_ids: ClassVar[tuple[str, str, str]] = (
        "gemma-2b",
        "gemma-e2b",
        "gemma-e4b",
    )
    _no_model_message: ClassVar[str] = "No models found. Please install at least one model."

    def __init__(self) -> None:
        """Initialize runtime state and caches for model execution."""
        self._logger = logging.getLogger("gemma-chatbot.model-manager")
        self._hardware: HardwareInfo | None = None
        self._base_strategy: QuantizationStrategy | None = None
        self._model_catalog: dict[str, ModelSpec] = {}
        self.available_models: dict[str, dict[str, Any]] = {}
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
        self._logger.info(
            "detected_hardware chip=%s ram_total_gb=%s ram_available_gb=%s metal=%s cuda=%s apple_silicon=%s",
            hardware.chip,
            hardware.ram_total_gb,
            hardware.ram_available_gb,
            hardware.metal_gpu,
            hardware.cuda_gpu,
            hardware.is_apple_silicon,
        )
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
            self._logger.warning(
                "default_model_unknown model_id=%s; falling back to first available",
                default_model_id,
            )
            fallback = self._first_available_model_id()
            if fallback is None:
                self._raise_no_models_error()
            default_model_id = fallback

        try:
            self.switch_model(default_model_id)
        except ModelError as exc:
            fallback = self._first_available_model_id(exclude={default_model_id})
            if fallback is None:
                raise
            self._logger.warning(
                "default_model_load_failed model_id=%s; fallback_model=%s; error=%s",
                default_model_id,
                fallback,
                exc.message,
            )
            self.switch_model(fallback)

    def switch_model(self, model_id: str) -> None:
        """Switch active model at runtime, loading and caching on first use.

        Args:
            model_id: Target model selector key.

        Returns:
            None.
        """
        requested_spec = self._get_model_spec(model_id=model_id)

        candidate_ids: list[str] = []
        if requested_spec.available:
            candidate_ids.append(model_id)
        else:
            fallback = self._first_available_model_id(exclude={model_id})
            if fallback is None:
                self._raise_no_models_error()
            self._logger.warning(
                "requested_model_unavailable requested=%s fallback=%s",
                model_id,
                fallback,
            )
            candidate_ids.append(fallback)

        for fallback_id in self._ordered_available_model_ids(exclude=set(candidate_ids)):
            candidate_ids.append(fallback_id)

        for candidate_id in candidate_ids:
            if self._activate_cached_model(candidate_id):
                return

            spec = self._get_model_spec(candidate_id)
            if not spec.available:
                continue

            try:
                self._load_and_activate_model(spec)
                return
            except Exception as exc:
                detail = str(exc)
                self._logger.warning(
                    "model_load_failed model_id=%s backend=%s backend_type=%s error=%s",
                    candidate_id,
                    spec.backend,
                    self._backend_type(spec.backend),
                    detail,
                )
                self._mark_model_unavailable(candidate_id)

        self._raise_no_models_error()

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

        def stream_tokens() -> Generator[str, None, None]:
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

        return stream_tokens()

    def get_stats(self) -> dict:
        """Return active model and generation performance statistics.

        Returns:
            dict: Runtime model metadata and performance stats.
        """
        active_id = self._active_model_id or self._first_available_model_id()
        spec = self._model_catalog.get(active_id) if active_id else None

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
        output: list[dict] = []
        for model_id in self._ordered_model_ids:
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

        fallback = self._first_available_model_id()
        return fallback or ""

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

    def _first_available_model_id(self, exclude: set[str] | None = None) -> str | None:
        """Return first available model ID from the catalog.

        Args:
            exclude: Model IDs that should not be considered.

        Returns:
            str | None: Available model key or None.
        """
        excluded = exclude or set()
        for model_id in self._ordered_model_ids:
            if model_id in excluded:
                continue
            spec = self._model_catalog.get(model_id)
            if spec and spec.available:
                return model_id
        return None

    def _ordered_available_model_ids(self, exclude: set[str] | None = None) -> list[str]:
        """Return all currently available model IDs in deterministic order.

        Args:
            exclude: Model IDs to skip.

        Returns:
            list[str]: Ordered available IDs.
        """
        excluded = exclude or set()
        ids: list[str] = []
        for model_id in self._ordered_model_ids:
            if model_id in excluded:
                continue
            spec = self._model_catalog.get(model_id)
            if spec and spec.available:
                ids.append(model_id)
        return ids

    def _build_model_catalog(self) -> None:
        """Build model catalog based on detected hardware and settings.

        Returns:
            None.
        """
        if self._hardware is None:
            return

        default_model = settings.default_model
        detected_available = self._detect_available_models()
        self.available_models = detected_available

        self._model_catalog = {}
        for model_id in self._ordered_model_ids:
            template = self._model_template(model_id=model_id)
            detected = detected_available.get(model_id)

            if detected is None:
                self._model_catalog[model_id] = ModelSpec(
                    model_id=model_id,
                    label=str(template["label"]),
                    backend=str(template["backend"]),
                    source=str(template["source"]),
                    quantization=str(template["quantization"]),
                    description=str(template["description"]),
                    is_default=(default_model == model_id),
                    available=False,
                    alias_of=template["alias_of"],
                )
                continue

            self._model_catalog[model_id] = ModelSpec(
                model_id=model_id,
                label=str(detected["label"]),
                backend=str(detected["backend"]),
                source=str(detected["source"]),
                quantization=str(detected["quantization"]),
                description=str(detected["description"]),
                is_default=(default_model == model_id),
                available=True,
                alias_of=detected["alias_of"],
            )

        if self.available_models:
            backends = {entry["backend"] for entry in self.available_models.values()}
            self._logger.info(
                "available_models ids=%s backends=%s",
                list(self.available_models.keys()),
                sorted(backends),
            )
        else:
            self._logger.warning("available_models ids=[]")

    def _detect_available_models(self) -> dict[str, dict[str, Any]]:
        """Detect locally usable models for current hardware/runtime.

        Returns:
            dict[str, dict[str, Any]]: Available model specs keyed by model ID.
        """
        model_root = Path(settings.model_path)
        prefer_mlx = self._should_prefer_mlx_backend()
        mlx_available = self._is_mlx_available()

        available: dict[str, dict[str, Any]] = {}
        for model_id in self._ordered_model_ids:
            template = self._model_template(model_id=model_id)

            if prefer_mlx and mlx_available:
                available[model_id] = {
                    "label": template["label"],
                    "backend": "mlx",
                    "source": template["mlx_source"],
                    "quantization": self._mlx_quantization(),
                    "description": template["mlx_description"],
                    "alias_of": template["mlx_alias_of"],
                }
                continue

            gguf_path = model_root / str(template["gguf_file"])
            if settings.skip_model_load or gguf_path.exists():
                available[model_id] = {
                    "label": template["label"],
                    "backend": "llama.cpp",
                    "source": str(gguf_path),
                    "quantization": "Q4_K_M",
                    "description": template["gguf_description"],
                    "alias_of": None,
                }

        return available

    def _model_template(self, model_id: str) -> dict[str, Any]:
        """Return static template metadata for each known model ID.

        Args:
            model_id: Model selector key.

        Returns:
            dict[str, Any]: Static metadata template.
        """
        model_root = Path(settings.model_path)
        templates: dict[str, dict[str, Any]] = {
            "gemma-2b": {
                "label": "Gemma 2B",
                "mlx_source": "google/gemma-2b-it",
                "mlx_alias_of": None,
                "mlx_description": "Fast baseline model for local chat.",
                "gguf_file": "gemma-2b-it.Q4_K_M.gguf",
                "gguf_description": "Fast baseline GGUF model.",
            },
            "gemma-e2b": {
                "label": "Gemma E2B",
                "mlx_source": "google/gemma-2b-it",
                "mlx_alias_of": "gemma-2b",
                "mlx_description": "Efficient profile mapped to MLX Gemma 2B.",
                "gguf_file": "gemma-e2b.Q4_K_M.gguf",
                "gguf_description": "Efficiency-focused GGUF model.",
            },
            "gemma-e4b": {
                "label": "Gemma E4B",
                "mlx_source": "google/gemma-2b-it",
                "mlx_alias_of": "gemma-2b",
                "mlx_description": "Quality profile mapped to MLX Gemma 2B.",
                "gguf_file": "gemma-e4b.Q4_K_M.gguf",
                "gguf_description": "Higher-quality GGUF model.",
            },
        }

        base = templates[model_id]
        prefer_mlx = self._should_prefer_mlx_backend()
        if prefer_mlx:
            return {
                "label": base["label"],
                "backend": "mlx",
                "source": base["mlx_source"],
                "quantization": self._mlx_quantization(),
                "description": base["mlx_description"],
                "alias_of": base["mlx_alias_of"],
                **base,
            }

        gguf_path = model_root / str(base["gguf_file"])
        return {
            "label": base["label"],
            "backend": "llama.cpp",
            "source": str(gguf_path),
            "quantization": "Q4_K_M",
            "description": base["gguf_description"],
            "alias_of": None,
            **base,
        }

    def _should_prefer_mlx_backend(self) -> bool:
        """Return whether MLX should be preferred for this host.

        Returns:
            bool: True when MLX is preferred backend.
        """
        if self._hardware is None:
            return False

        return (
            self._hardware.is_apple_silicon
            and self._hardware.metal_gpu
            and not isinstance(self._base_strategy, LlamaCppQuantization)
        )

    def _is_mlx_available(self) -> bool:
        """Return whether MLX backend is usable.

        Returns:
            bool: True when MLX runtime can be used.
        """
        if not self._should_prefer_mlx_backend():
            return False

        if settings.skip_model_load:
            return True

        try:
            from mlx_lm import load  # type: ignore

            _ = load
            return True
        except Exception as exc:
            self._logger.warning("mlx_unavailable reason=%s", str(exc))
            return False

    def _activate_cached_model(self, model_id: str) -> bool:
        """Activate model from cache if already loaded.

        Args:
            model_id: Model selector key.

        Returns:
            bool: True when cache activation succeeded.
        """
        if self._active_model_id == model_id and self._active_strategy is not None:
            return True

        strategy = self._model_cache.get(model_id)
        if strategy is None:
            return False

        self._active_strategy = strategy
        self._active_model_id = model_id
        spec = self._model_catalog.get(model_id)
        if spec is not None:
            self._logger.info(
                "selected_model model_id=%s backend=%s backend_type=%s source=%s cached=true",
                spec.model_id,
                spec.backend,
                self._backend_type(spec.backend),
                spec.source,
            )
        return True

    def _load_and_activate_model(self, spec: ModelSpec) -> None:
        """Instantiate strategy, load model, warm up, and activate it.

        Args:
            spec: Model specification object.

        Returns:
            None.
        """
        strategy = self._create_strategy(spec=spec)
        started = time.perf_counter()
        strategy.load_model(spec.source)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        self._model_cache[spec.model_id] = strategy
        self._model_load_times_ms[spec.model_id] = elapsed_ms
        self._active_strategy = strategy
        self._active_model_id = spec.model_id

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
        self._logger.info(
            "selected_model model_id=%s backend=%s backend_type=%s source=%s load_ms=%s warmup_tps=%s",
            spec.model_id,
            spec.backend,
            self._backend_type(spec.backend),
            spec.source,
            elapsed_ms,
            self._last_tokens_per_sec,
        )

    def _mark_model_unavailable(self, model_id: str) -> None:
        """Mark one model unavailable after failed load and clear stale cache.

        Args:
            model_id: Model selector key.

        Returns:
            None.
        """
        spec = self._model_catalog.get(model_id)
        if spec is not None:
            spec.available = False
        self.available_models.pop(model_id, None)
        self._model_cache.pop(model_id, None)

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

    def _backend_type(self, backend: str) -> str:
        """Map backend name to user-friendly backend type label.

        Args:
            backend: Runtime backend name.

        Returns:
            str: Backend type label.
        """
        if backend == "mlx":
            return "MLX"
        return "GGUF"

    def _raise_no_models_error(self) -> None:
        """Raise the canonical no-models-available error."""
        raise ModelError(
            message=self._no_model_message,
            status_code=503,
            log_detail=(
                "no_available_models install_mlx_lm='pip install mlx-lm' "
                "or add_gguf_to='backend/models/'"
            ),
        )


model_manager = ModelManager.get_instance()
