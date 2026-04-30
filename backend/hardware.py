from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

import psutil


@dataclass
class HardwareProfile:
    chip: str
    ram_total_gb: int
    ram_available_gb: float
    cpu_cores: int
    metal_gpu: bool
    is_apple_silicon: bool
    quantization: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ram_available_gb"] = round(self.ram_available_gb, 2)
        return data


def _sysctl_value(key: str) -> str:
    try:
        value = subprocess.check_output(["sysctl", "-n", key], text=True).strip()
        return value
    except Exception:
        return ""


def _detect_chip_name() -> str:
    brand = _sysctl_value("machdep.cpu.brand_string")
    if brand:
        return brand

    model = _sysctl_value("hw.model")
    if model:
        return model

    proc = platform.processor().strip()
    if proc:
        return proc

    return platform.machine() or "Unknown CPU"


def _is_apple_silicon() -> bool:
    if platform.system() != "Darwin":
        return False

    arm64_flag = _sysctl_value("hw.optional.arm64")
    if arm64_flag == "1":
        return True

    machine = platform.machine().lower()
    proc = platform.processor().lower()
    return machine == "arm64" or "apple" in proc


def _metal_is_available() -> bool:
    try:
        import mlx.core as mx  # type: ignore

        return bool(mx.metal.is_available())
    except Exception:
        return False


def choose_quantization(
    *, is_apple_silicon: bool, metal_gpu: bool, ram_total_gb: int
) -> str:
    if is_apple_silicon and metal_gpu:
        if ram_total_gb >= 16:
            return "INT4-mlx"
        if ram_total_gb >= 8:
            return "INT8-mlx"

    return "Q4_K_M-gguf"


def detect_hardware() -> HardwareProfile:
    memory = psutil.virtual_memory()
    ram_total_gb = round(memory.total / (1024**3))
    ram_available_gb = memory.available / (1024**3)

    is_apple = _is_apple_silicon()
    metal_gpu = _metal_is_available()
    quantization = choose_quantization(
        is_apple_silicon=is_apple,
        metal_gpu=metal_gpu,
        ram_total_gb=ram_total_gb,
    )

    return HardwareProfile(
        chip=_detect_chip_name(),
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
        cpu_cores=psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1,
        metal_gpu=metal_gpu,
        is_apple_silicon=is_apple,
        quantization=quantization,
    )
