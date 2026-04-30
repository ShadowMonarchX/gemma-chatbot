from __future__ import annotations

import platform
import subprocess

import psutil
from pydantic import BaseModel, ConfigDict


class HardwareInfo(BaseModel):
    """Normalized hardware capabilities consumed by model selection logic."""

    model_config = ConfigDict(strict=True)

    chip: str
    ram_total_gb: float
    ram_available_gb: float
    cpu_cores: int
    metal_gpu: bool


class HardwareDetector:
    """Detects CPU, memory, and Metal capabilities for quantization selection."""

    def __init__(self) -> None:
        """Initialize the hardware detector."""
        self._ram_bytes_per_gb: float = float(1024**3)

    def detect(self) -> HardwareInfo:
        """Read machine capabilities and return a strict hardware descriptor.

        Returns:
            HardwareInfo: Hardware details required by model selection.
        """
        virtual_memory = psutil.virtual_memory()
        chip_name = self._read_apple_silicon()
        detected_processor = platform.processor().strip()
        if detected_processor and detected_processor.lower() not in chip_name.lower():
            chip_name = f"{chip_name} ({detected_processor})"

        total_gb = self._read_ram()
        available_gb = round(virtual_memory.available / self._ram_bytes_per_gb, 2)
        cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1

        return HardwareInfo(
            chip=chip_name,
            ram_total_gb=total_gb,
            ram_available_gb=available_gb,
            cpu_cores=int(cores),
            metal_gpu=self._check_metal(),
        )

    def _read_apple_silicon(self) -> str:
        """Determine chip branding from sysctl and processor introspection.

        Returns:
            str: Human-readable chip identifier.
        """
        processor_name = platform.processor().strip()
        command = ["sysctl", "-n", "machdep.cpu.brand_string"]
        try:
            sysctl_chip = subprocess.check_output(command, text=True).strip()
            if sysctl_chip:
                return sysctl_chip
        except Exception:
            processor_name = platform.processor().strip()
            if processor_name:
                return processor_name

        if processor_name:
            return processor_name

        machine_name = platform.machine().strip()
        if machine_name:
            return machine_name

        return "Unknown CPU"

    def _read_ram(self) -> float:
        """Read total RAM in gigabytes.

        Returns:
            float: Rounded total system RAM in GB.
        """
        total_bytes = psutil.virtual_memory().total
        return round(total_bytes / self._ram_bytes_per_gb, 2)

    def _check_metal(self) -> bool:
        """Check whether Metal GPU acceleration is available.

        Returns:
            bool: True when mlx Metal backend is available.
        """
        try:
            import mlx.core as mx  # type: ignore

            return bool(mx.metal.is_available())
        except Exception:
            return False


hardware_detector = HardwareDetector()
