from __future__ import annotations

import logging
import os
import threading
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
    _gguf_repo_id: ClassVar[str] = "TheBloke/Gemma-2B-IT-GGUF"
    _gguf_filename: ClassVar[str] = "gemma-2b-it.Q4_K_M.gguf"

    _no_model_message: ClassVar[str] = (
        "No models found. Please install at least one model."
    )
    _download_in_progress_message: ClassVar[str] = (
        "Model not available yet. Download in progress."
    )
    _download_failed_message: ClassVar[str] = (
        "Model download failed. Check server logs for details."
    )

    def __init__(self) -> None:
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

        self._download_lock = threading.Lock()
        self._download_in_progress: bool = False
        self._download_error: str | None = None
        self._download_thread: threading.Thread | None = None

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def configure(self, hardware: HardwareInfo) -> None:
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
        self._prepare_cache_dirs()
        self._build_model_catalog()

    def load(self, strategy: QuantizationStrategy) -> None:
        if self._hardware is None:
            raise ConfigurationError(
                message="Hardware must be configured before model loading",
                status_code=500,
                log_detail="configure() not called before load()",
            )

        self._base_strategy = strategy
        self._prepare_cache_dirs()
        self._logger.info("🔍 Detecting models...")
        self._build_model_catalog()

        default_model_id = self._normalize_requested_model_id(settings.default_model)
        first_available = self._first_available_model_id()

        if first_available is None:
            self._trigger_model_acquire(default_model_id)
            self._build_model_catalog()
            first_available = self._first_available_model_id()

            if first_available is None:
                if self._download_in_progress:
                    self._raise_download_in_progress_error()
                if self._download_error:
                    self._raise_download_failed_error()
                self._raise_no_models_error()

        selected_model_id = (
            default_model_id
            if self._is_model_available(default_model_id)
            else (first_available or self._primary_model_id)
        )

        selected_spec = self._model_catalog.get(selected_model_id)
        if selected_spec and selected_spec.backend == "mlx" and not self._is_mlx_cached():
            self._trigger_model_acquire(selected_model_id)
            self._raise_download_in_progress_error()

        self.switch_model(selected_model_id)

    def switch_model(self, model_id: str) -> None:
        self._build_model_catalog()
        resolved_model_id = self._normalize_requested_model_id(model_id)
        requested_spec = self._get_model_spec(model_id=resolved_model_id)

        if self._download_in_progress and not self._is_cached_or_active(resolved_model_id):
            self._raise_download_in_progress_error()
        if self._download_error and not self._is_cached_or_active(resolved_model_id):
            self._raise_download_failed_error()

        candidate_ids: list[str] = []
        if requested_spec.available:
            candidate_ids.append(resolved_model_id)
        else:
            fallback = self._first_available_model_id(exclude={resolved_model_id})
            if fallback:
                self._logger.warning(
                    "⚠️ Falling back to %s because requested model %s is unavailable",
                    fallback,
                    resolved_model_id,
                )
                candidate_ids.append(fallback)
            else:
                self._trigger_model_acquire(resolved_model_id)
                if self._download_in_progress:
                    self._raise_download_in_progress_error()
                if self._download_error:
                    self._raise_download_failed_error()
                self._raise_no_models_error()

        for fallback_id in self._ordered_available_model_ids(
            exclude=set(candidate_ids)
        ):
            candidate_ids.append(fallback_id)

        for candidate_id in candidate_ids:
            if self._activate_cached_model(candidate_id):
                return

            spec = self._get_model_spec(candidate_id)
            if not spec.available:
                continue

            if spec.backend == "mlx" and not self._is_mlx_cached():
                self._trigger_model_acquire(candidate_id)
                self._raise_download_in_progress_error()

            try:
                self._load_and_activate_model(spec)
                return
            except Exception as exc:
                self._logger.warning(
                    "model_load_failed model_id=%s backend=%s backend_type=%s error=%s",
                    candidate_id,
                    spec.backend,
                    self._backend_type(spec.backend),
                    str(exc),
                )
                self._mark_model_unavailable(candidate_id)

        self._trigger_model_acquire(resolved_model_id)
        if self._download_in_progress:
            self._raise_download_in_progress_error()
        if self._download_error:
            self._raise_download_failed_error()
        self._raise_no_models_error()

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

        prefers_mlx = self._should_prefer_mlx_backend()
        has_mlx = self._primary_model_id in available
        has_gguf = self._gguf_profile_id in available

        selected_backend = ""
        selected_source = ""
        selected_quantization = ""

        if prefers_mlx and has_mlx:
            selected_backend = "mlx"
            selected_source = self._mlx_model_source
            selected_quantization = self._mlx_quantization()
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
                backend=(selected_backend or ("mlx" if prefers_mlx else "llama.cpp")),
                source=(
                    selected_source
                    or (
                        self._mlx_model_source
                        if prefers_mlx
                        else self._gguf_model_path().as_posix()
                    )
                ),
                quantization=(
                    selected_quantization
                    or (self._mlx_quantization() if prefers_mlx else "Q4_K_M")
                ),
                description=description,
                is_default=(default_model == model_id),
                available=is_available,
                alias_of=alias_of,
            )

        if self.available_models:
            self._logger.info(
                "available_models ids=%s", list(self.available_models.keys())
            )
        else:
            self._logger.warning("available_models ids=[]")

    def _detect_available_models(self) -> dict[str, dict[str, Any]]:
        self._logger.info("🔍 Detecting models...")

        available: dict[str, dict[str, Any]] = {}

        if self._is_mlx_dependency_available():
            available[self._primary_model_id] = {
                "type": "mlx",
                "id": self._mlx_model_source,
            }

        gguf_path = self._gguf_model_path()
        if settings.skip_model_load or gguf_path.exists():
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

    def _should_prefer_mlx_backend(self) -> bool:
        if self._hardware is None:
            return False

        return (
            self._hardware.is_apple_silicon
            and self._hardware.metal_gpu
            and not isinstance(self._base_strategy, LlamaCppQuantization)
        )

    def _is_mlx_dependency_available(self) -> bool:
        try:
            from mlx_lm import load  # type: ignore

            _ = load
            return True
        except Exception:
            return False

    def _is_mlx_cached(self) -> bool:
        if settings.skip_model_load:
            return True

        try:
            from huggingface_hub import snapshot_download  # type: ignore

            snapshot_download(
                repo_id=self._mlx_model_source,
                local_files_only=True,
                cache_dir=self._model_cache_dir().as_posix(),
            )
            return True
        except Exception:
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

        return True

    def _is_cached_or_active(self, model_id: str) -> bool:
        if self._active_model_id == model_id and self._active_strategy is not None:
            return True
        return model_id in self._model_cache

    def _load_and_activate_model(self, spec: ModelSpec) -> None:
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

    def _hf_token(self) -> str:
        configured = str(settings.hf_token).strip()
        if configured:
            return configured
        return os.getenv("HF_TOKEN", "").strip()

    def _trigger_model_acquire(self, requested_model_id: str) -> None:
        if settings.skip_model_load:
            return

        if self._download_in_progress:
            return

        target = self._primary_model_id
        if requested_model_id == self._gguf_profile_id:
            target = self._gguf_profile_id
        elif not (
            self._should_prefer_mlx_backend() and self._is_mlx_dependency_available()
        ):
            target = self._gguf_profile_id

        self._start_background_download(target)

    def _start_background_download(self, target_model_id: str) -> None:
        with self._download_lock:
            if self._download_in_progress:
                return

            self._download_in_progress = True
            self._download_error = None
            self._download_thread = threading.Thread(
                target=self._download_worker,
                args=(target_model_id,),
                daemon=True,
                name="gemma-model-download",
            )
            self._download_thread.start()

    def _download_worker(self, target_model_id: str) -> None:
        self._logger.info(" Downloading model... target=%s", target_model_id)
        try:
            self._ensure_model_available(target_model_id)
            self._logger.info(" Model ready")
        except ModelError as exc:
            with self._download_lock:
                self._download_error = exc.log_detail or exc.message
            self._logger.error(" Download failed: %s", exc.log_detail or exc.message)
        except Exception as exc:
            with self._download_lock:
                self._download_error = str(exc)
            self._logger.error(" Download failed: %s", str(exc))
        finally:
            with self._download_lock:
                self._download_in_progress = False
            self._build_model_catalog()

    def _ensure_model_available(self, model_id: str) -> None:
        self._prepare_cache_dirs()
        token = self._hf_token()

        if model_id == self._gguf_profile_id:
            model_path = self._gguf_model_path()
            if model_path.exists():
                return

            try:
                from huggingface_hub import hf_hub_download  # type: ignore

                hf_hub_download(
                    repo_id=self._gguf_repo_id,
                    filename=self._gguf_filename,
                    local_dir=model_path.parent.as_posix(),
                    cache_dir=self._model_cache_dir().as_posix(),
                    token=token or None,
                )
            except Exception as exc:
                raise ModelError(
                    message="Failed to download/load GGUF model",
                    status_code=503,
                    log_detail=str(exc),
                ) from exc

            if not model_path.exists():
                raise ModelError(
                    message="Failed to download/load GGUF model",
                    status_code=503,
                    log_detail=f"gguf_missing_after_download path={model_path}",
                )
            return

        try:
            from huggingface_hub import login  # type: ignore
            from mlx_lm import load  # type: ignore

            if token:
                login(token=token, add_to_git_credential=False)

            model, tokenizer = load(self._mlx_model_source)
            del model
            del tokenizer
        except Exception as exc:
            raise ModelError(
                message="Failed to download/load MLX model",
                status_code=503,
                log_detail=str(exc),
            ) from exc

    def _raise_no_models_error(self) -> None:
        raise ModelError(
            message=self._no_model_message,
            status_code=503,
            log_detail="no_available_models",
        )

    def _raise_download_in_progress_error(self) -> None:
        raise ModelError(
            message=self._download_in_progress_message,
            status_code=503,
            log_detail="download_in_progress=true",
        )

    def _raise_download_failed_error(self) -> None:
        detail = self._download_error or "unknown download error"
        raise ModelError(
            message=self._download_failed_message,
            status_code=503,
            log_detail=detail,
        )


model_manager = ModelManager.get_instance()
