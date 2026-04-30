from __future__ import annotations

import platform
import subprocess

import psutil
from pydantic import BaseModel, ConfigDict


class HardwareInfo(BaseModel):
    """Strict representation of host hardware capabilities."""

    model_config = ConfigDict(strict=True)

    chip: str
    ram_total_gb: float
    ram_available_gb: float
    cpu_cores: int
    metal_gpu: bool
    cuda_gpu: bool
    is_apple_silicon: bool
    platform_system: str


class HardwareDetector:
    """Detects CPU, memory, and accelerator capabilities at startup."""

    def __init__(self) -> None:
        """Initialize constants used during hardware inspection."""
        self._bytes_per_gb: float = float(1024**3)

    def detect(self) -> HardwareInfo:
        """Read machine attributes and return a strict hardware profile.

        Returns:
            HardwareInfo: Full hardware snapshot used by runtime selection.
        """
        memory = psutil.virtual_memory()
        total_ram = self._read_ram()
        available_ram = round(memory.available / self._bytes_per_gb, 2)
        cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1

        chip_name = self._read_apple_silicon()
        processor_name = platform.processor().strip()
        if processor_name and processor_name.lower() not in chip_name.lower():
            chip_name = f"{chip_name} ({processor_name})"

        is_apple = self._is_apple_silicon(chip_name)

        return HardwareInfo(
            chip=chip_name,
            ram_total_gb=total_ram,
            ram_available_gb=available_ram,
            cpu_cores=int(cpu_cores),
            metal_gpu=self._check_metal(),
            cuda_gpu=self._check_cuda(),
            is_apple_silicon=is_apple,
            platform_system=platform.system(),
        )

    def _read_apple_silicon(self) -> str:
        """Read CPU branding via sysctl and platform fallbacks.

        Returns:
            str: Human-readable chip name.
        """
        processor_name = platform.processor().strip()
        command = ["sysctl", "-n", "machdep.cpu.brand_string"]
        try:
            chip = subprocess.check_output(command, text=True).strip()
            if chip:
                return chip
        except Exception:
            if processor_name:
                return processor_name

        if processor_name:
            return processor_name

        machine_name = platform.machine().strip()
        if machine_name:
            return machine_name

        return "Unknown CPU"

    def _read_ram(self) -> float:
        """Read total system RAM in GB.

        Returns:
            float: Rounded total RAM in gigabytes.
        """
        return round(psutil.virtual_memory().total / self._bytes_per_gb, 2)

    def _check_metal(self) -> bool:
        """Detect whether MLX Metal acceleration is available.

        Returns:
            bool: True when Metal backend is available.
        """
        try:
            import mlx.core as mx  # type: ignore

            return bool(mx.metal.is_available())
        except Exception:
            return False

    def _check_cuda(self) -> bool:
        """Detect whether CUDA is available through PyTorch.

        Returns:
            bool: True when CUDA runtime is available.
        """
        try:
            import torch  # type: ignore

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _is_apple_silicon(self, chip_name: str) -> bool:
        """Determine whether hardware appears to be Apple Silicon.

        Args:
            chip_name: Chip branding string.

        Returns:
            bool: True when running on Apple Silicon.
        """
        if platform.system() != "Darwin":
            return False

        normalized = chip_name.lower()
        apple_tokens = ("apple", "m1", "m2", "m3", "m4", "m5")
        if any(token in normalized for token in apple_tokens):
            return True

        return platform.machine().lower() == "arm64"


hardware_detector = HardwareDetector()
