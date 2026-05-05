from __future__ import annotations

import logging
import os
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
        self.model_id: str = model_id
        self.label: str = label
        self.backend: str = backend
        self.source: str = source
        self.quantization: str = quantization
        self.description: str = description
        self.is_default: bool = is_default
        self.available: bool = available
        self.alias_of: str | None = alias_of

    def to_dict(self) -> dict[str, Any]:
        """Serialize model specification for API output."""
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
    _primary_model_id: ClassVar[str] = "gemma-2b"
    _gguf_profile_id: ClassVar[str] = "gemma-2b-gguf"
    _mlx_model_source: ClassVar[str] = "google/gemma-2b-it"
    _gguf_filename: ClassVar[str] = "gemma-2b-it.Q4_K_M.gguf"

    _no_model_message: ClassVar[str] = (
        "No models found. Please install at least one model."
    )

    def __init__(self) -> None:
        self._logger = logging.getLogger("gemma-chatbot.model-manager")
        self._hardware: HardwareInfo | None = None
        self._base_strategy: QuantizationStrategy | None = None

        self._model_catalog: dict[str, ModelSpec] = {}
        self.available_models: dict[str, dict[str, Any]] = {}

        # This cache holds one loaded strategy per logical model profile.
        # The cache prevents reloading model weights on every chat request.
        self._model_cache: dict[str, QuantizationStrategy] = {}
        self._model_load_times_ms: dict[str, int] = {}
        self._active_model_id: str = ""
        self._active_strategy: QuantizationStrategy | None = None
        self._last_tokens_per_sec: float = 0.0
        self._avg_tokens_per_sec: float = 0.0
        self._generation_runs: int = 0

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def configure(self, hardware: HardwareInfo) -> None:
        # We keep hardware info so backend selection can be reproduced consistently.
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
        print(
            "[ModelManager] detected_hardware "
            f"chip={hardware.chip} "
            f"ram_total_gb={hardware.ram_total_gb} "
            f"metal_gpu={hardware.metal_gpu} "
            f"cuda_gpu={hardware.cuda_gpu} "
            f"apple_silicon={hardware.is_apple_silicon}"
        )
        self._prepare_cache_dirs()
        self._build_model_catalog()

    def load(self, strategy: QuantizationStrategy) -> None:
        # The manager must be configured before loading to avoid ambiguous runtime state.
        if self._hardware is None:
            raise ConfigurationError(
                message="Hardware must be configured before model loading",
                status_code=500,
                log_detail="configure() not called before load()",
            )

        # We keep the originally selected strategy for compatibility with existing flow,
        # but model loading is still resolved from the model catalog below.
        self._base_strategy = strategy
        self._prepare_cache_dirs()

        # This detection call is the only place where model availability is refreshed.
        # Keeping it centralized avoids retry loops and inconsistent states.
        print("[ModelManager] Detecting models...")
        self._build_model_catalog()

        default_model_id = self._normalize_requested_model_id(settings.default_model)
        selected_model_id: str | None = None

        # If default model is available we keep current behavior and load it first.
        if self._is_model_available(default_model_id):
            selected_model_id = default_model_id
        else:
            # If default is unavailable we safely fall back to the first available profile.
            selected_model_id = self._first_available_model_id()
            if selected_model_id:
                self._logger.warning(
                    "requested_default_unavailable fallback_model=%s requested=%s",
                    selected_model_id,
                    default_model_id,
                )
                print(
                    "[ModelManager] requested default model is unavailable; "
                    f"falling back to {selected_model_id}"
                )

        if selected_model_id is None:
            # We raise a clean controlled error that the app can convert into a user-friendly response.
            self._raise_no_models_error()

        # This call activates cached model when possible and only loads once when needed.
        self.switch_model(selected_model_id)

    def switch_model(self, model_id: str) -> None:
        # We refresh catalog before switching to pick up any environment/model-path changes.
        self._build_model_catalog()
        resolved_model_id = self._normalize_requested_model_id(model_id)

        candidate_ids: list[str] = []

        # Requested model is used only when it is known and marked available.
        requested_spec = self._model_catalog.get(resolved_model_id)
        if requested_spec is not None and requested_spec.available:
            candidate_ids.append(resolved_model_id)
        else:
            # Unknown or unavailable model IDs do not crash the request path.
            # We always fall back to first available model when possible.
            fallback = self._first_available_model_id()
            if fallback is not None:
                candidate_ids.append(fallback)
                self._logger.warning(
                    "requested_model_unavailable fallback_model=%s requested=%s",
                    fallback,
                    resolved_model_id,
                )
                print(
                    "[ModelManager] requested model is unavailable; "
                    f"falling back to {fallback}"
                )

        if not candidate_ids:
            self._raise_no_models_error()

        # We append other available models as backup candidates so one load failure
        # does not block service startup when another valid profile exists.
        for fallback_id in self._ordered_available_model_ids(
            exclude=set(candidate_ids)
        ):
            candidate_ids.append(fallback_id)

        for candidate_id in candidate_ids:
            # Cache-first activation avoids unnecessary model reload and keeps latency low.
            if self._activate_cached_model(candidate_id):
                return

            spec = self._get_model_spec(model_id=candidate_id)
            if not spec.available:
                continue

            try:
                self._load_and_activate_model(spec)
                return
            except ModelError as exc:
                # We mark failed candidates unavailable to prevent repeated failures
                # during the same process lifetime.
                self._logger.warning(
                    "model_load_failed model_id=%s backend=%s backend_type=%s error=%s",
                    candidate_id,
                    spec.backend,
                    self._backend_type(spec.backend),
                    exc.log_detail or exc.message,
                )
                print(
                    "[ModelManager] model load failed "
                    f"model_id={candidate_id} error={exc.log_detail or exc.message}"
                )
                self._mark_model_unavailable(candidate_id)

        # If every candidate failed we raise one clean error with deterministic wording.
        raise ModelError(
            message="Failed to load any available model",
            status_code=503,
            log_detail="all_model_candidates_failed",
        )

    def generate_stream(
        self,
        messages: list[dict],
        system: str,
        skill: str,
        model_id: str,
    ) -> Generator[str, None, None]:
        _ = skill
        self.switch_model(model_id=model_id)
        strategy = self._active_strategy

        if strategy is None:
            raise ModelError(
                message="Model not loaded",
                status_code=500,
                log_detail="active_strategy missing after switch_model",
            )
        # We wrap the strategy generator to track tokens per second for performance monitoring.
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
                (weighted_sum + run_tps) / self._generation_runs,
                2,
            )

        return stream_tokens()

    def get_stats(self) -> dict[str, Any]:
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

    def list_models(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for model_id in self._ordered_model_ids:
            spec = self._model_catalog.get(model_id)
            if spec is not None:
                output.append(spec.to_dict())
        return output

    def get_active_model_id(self) -> str:
        if self._active_model_id:
            return self._active_model_id

        fallback = self._first_available_model_id()
        return fallback or ""

    def is_model_known(self, model_id: str) -> bool:
        resolved_model_id = self._normalize_requested_model_id(model_id)
        return resolved_model_id in self._model_catalog

    def _get_model_spec(self, model_id: str) -> ModelSpec:
        spec = self._model_catalog.get(model_id)
        if spec is None:
            raise ValidationError(
                message="Unknown model ID",
                status_code=422,
                log_detail=f"model_id={model_id}",
            )
        return spec

    def _is_model_available(self, model_id: str) -> bool:
        spec = self._model_catalog.get(model_id)
        return bool(spec and spec.available)

    def _first_available_model_id(self, exclude: set[str] | None = None) -> str | None:
        excluded = exclude or set()
        for model_id in self._ordered_model_ids:
            if model_id in excluded:
                continue
            spec = self._model_catalog.get(model_id)
            if spec and spec.available:
                return model_id
        return None

    def _ordered_available_model_ids(
        self, exclude: set[str] | None = None
    ) -> list[str]:
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
        if self._hardware is None:
            return

        default_model = self._normalize_requested_model_id(settings.default_model)
        available = self._detect_available_models()
        self.available_models = available

        has_mlx = self._primary_model_id in available
        has_gguf = self._gguf_profile_id in available

        selected_backend = ""
        selected_source = ""
        selected_quantization = ""

        # MLX is intentionally preferred whenever import is available.
        # This ensures Apple Silicon uses mlx_lm.load("google/gemma-2b-it") path.
        if has_mlx:
            selected_backend = "mlx"
            selected_source = self._mlx_model_source
            selected_quantization = self._mlx_quantization()
        # GGUF is kept only as a local-file fallback and never auto-downloaded.
        elif has_gguf:
            selected_backend = "llama.cpp"
            selected_source = self._gguf_model_path().as_posix()
            selected_quantization = "Q4_K_M"

        templates = self._model_templates()
        self._model_catalog = {}
        for model_id in self._ordered_model_ids:
            template = templates[model_id]
            alias_of = (
                self._primary_model_id if model_id != self._primary_model_id else None
            )
            is_available = bool(selected_backend)

            if selected_backend == "mlx":
                description = template["mlx_description"]
            elif selected_backend == "llama.cpp":
                description = template["gguf_description"]
            else:
                description = template["mlx_description"]

            self._model_catalog[model_id] = ModelSpec(
                model_id=model_id,
                label=template["label"],
                backend=(selected_backend or "mlx"),
                source=(selected_source or self._mlx_model_source),
                quantization=(selected_quantization or self._mlx_quantization()),
                description=description,
                is_default=(default_model == model_id),
                available=is_available,
                alias_of=alias_of,
            )

        available_ids = list(self.available_models.keys())
        if available_ids:
            print(f"[ModelManager] available_models={available_ids}")
        else:
            print("[ModelManager] available_models=[]")
        if selected_backend:
            print(
                "[ModelManager] selected_backend "
                f"backend={selected_backend} source={selected_source or self._mlx_model_source}"
            )

    def _detect_available_models(self) -> dict[str, dict[str, Any]]:
        # This function detects available models on the system.
        # It prioritizes MLX because it is optimized for Apple Silicon and
        # because mlx_lm.load handles model download/cache internally.
        print("[ModelManager] Detecting models...")

        available: dict[str, dict[str, Any]] = {}

        # Test mode must stay deterministic even when mlx_lm is not installed.
        # In this mode MLXQuantization runs with mock behavior and never imports mlx.
        if settings.skip_model_load:
            available[self._primary_model_id] = {
                "type": "mlx",
                "id": self._mlx_model_source,
            }
            print("[ModelManager] skip_model_load enabled; exposing MLX profile")
            return available

        # MLX availability is determined purely by import check.
        # We intentionally do not require local files for MLX because huggingface cache
        # may be empty on first boot and mlx_lm.load will populate it on demand.
        if self._is_mlx_dependency_available():
            available[self._primary_model_id] = {
                "type": "mlx",
                "id": self._mlx_model_source,
            }

        # GGUF path is intentionally disabled from any auto-download flow.
        # We only expose GGUF when a valid local file already exists.
        gguf_path = self._gguf_model_path()
        if gguf_path.exists():
            available[self._gguf_profile_id] = {
                "type": "gguf",
                "file": gguf_path.as_posix(),
            }

        return available

    def _model_templates(self) -> dict[str, dict[str, str]]:
        return {
            "gemma-2b": {
                "label": "Gemma 2B",
                "mlx_description": "Fast baseline model for local chat.",
                "gguf_description": "Fast GGUF fallback model.",
            },
            "gemma-e2b": {
                "label": "Gemma E2B",
                "mlx_description": "Efficient profile mapped to MLX Gemma 2B.",
                "gguf_description": "Efficient profile mapped to GGUF Gemma 2B.",
            },
            "gemma-e4b": {
                "label": "Gemma E4B",
                "mlx_description": "Quality profile mapped to MLX Gemma 2B.",
                "gguf_description": "Quality profile mapped to GGUF Gemma 2B.",
            },
        }

    def _normalize_requested_model_id(self, model_id: str) -> str:
        if model_id in self._ordered_model_ids:
            return model_id
        if model_id == self._gguf_profile_id:
            return self._primary_model_id
        return model_id

    def _is_mlx_dependency_available(self) -> bool:
        try:
            from mlx_lm import load  # type: ignore

            _ = load
            return True
        except Exception as exc:
            self._logger.warning("mlx_dependency_unavailable error=%s", str(exc))
            return False

    def _activate_cached_model(self, model_id: str) -> bool:
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
            print(
                "[ModelManager] selected_model "
                f"model_id={spec.model_id} backend={spec.backend} "
                f"backend_type={self._backend_type(spec.backend)} cached=true"
            )

        return True

    def _load_and_activate_model(self, spec: ModelSpec) -> None:
        strategy = self._create_strategy(spec=spec)

        started = time.perf_counter()
        try:
            strategy.load_model(spec.source)
        except ModelError as exc:
            if spec.backend == "mlx":
                raise ModelError(
                    message="Failed to load MLX model",
                    status_code=503,
                    log_detail=exc.log_detail or exc.message,
                ) from exc
            raise
        except Exception as exc:
            if spec.backend == "mlx":
                raise ModelError(
                    message="Failed to load MLX model",
                    status_code=503,
                    log_detail=str(exc),
                ) from exc
            raise ModelError(
                message="Failed to load model",
                status_code=503,
                log_detail=str(exc),
            ) from exc

        elapsed_ms = int((time.perf_counter() - started) * 1000)

        # We cache by logical profile so profile switches avoid repeated load cost.
        self._model_cache[spec.model_id] = strategy
        self._model_load_times_ms[spec.model_id] = elapsed_ms
        self._active_strategy = strategy
        self._active_model_id = spec.model_id

        # Warm-up keeps first real user request latency predictable.
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
        print(
            "[ModelManager] selected_model "
            f"model_id={spec.model_id} backend={spec.backend} "
            f"backend_type={self._backend_type(spec.backend)} load_ms={elapsed_ms}"
        )

    def _mark_model_unavailable(self, model_id: str) -> None:
        spec = self._model_catalog.get(model_id)
        if spec is not None:
            spec.available = False

        self._model_cache.pop(model_id, None)

    def _create_strategy(self, spec: ModelSpec) -> QuantizationStrategy:
        if spec.backend == "mlx":
            return MLXQuantization(
                precision=spec.quantization.lower(),
                max_tokens=settings.max_tokens,
            )

        gpu_layers = 35 if self._hardware and self._hardware.cuda_gpu else 0
        return LlamaCppQuantization(
            quant=spec.quantization,
            max_tokens=settings.max_tokens,
            n_gpu_layers=gpu_layers,
        )

    def _mlx_quantization(self) -> str:
        if isinstance(self._base_strategy, MLXQuantization):
            return self._base_strategy.quantization

        if self._hardware and self._hardware.ram_total_gb >= 16.0:
            return "INT4"
        return "INT8"

    def _backend_type(self, backend: str) -> str:
        if backend == "mlx":
            return "MLX"
        return "GGUF"

    def _prepare_cache_dirs(self) -> None:
        cache_dir = self._model_cache_dir()
        model_dir = self._model_root_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        os.environ.setdefault("HF_HOME", cache_dir.as_posix())
        os.environ.setdefault("HF_HUB_CACHE", (cache_dir / "hub").as_posix())

    def _model_cache_dir(self) -> Path:
        return Path(settings.model_cache_dir).expanduser().resolve()

    def _model_root_dir(self) -> Path:
        return Path(settings.model_path).expanduser().resolve()

    def _gguf_model_path(self) -> Path:
        return self._model_root_dir() / self._gguf_filename

    def _raise_no_models_error(self) -> None:
        raise ModelError(
            message=self._no_model_message,
            status_code=503,
            log_detail="no_available_models",
        )


model_manager = ModelManager.get_instance()
