from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """默认实验配置。

    参数集中在这里，避免入口脚本和算法实现互相硬编码实验条件。
    """

    array_elements: int = 64
    array_arms: int = 8
    array_aperture_m: float = 0.15
    scan_distances_m: tuple[float, ...] = (1.2, 1.6, 2.0)
    scan_extent_m: tuple[float, float] = (-0.6, 0.6)
    scan_step_m: float = 0.01
    tick_step_m: float = 0.1
    source_xy_positions_m: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.5, 0.0), (-0.5, 0.0))
    frequency_hz: float = 25_000.0
    sampling_rate_hz: float = 192_000.0
    duration_s: float = 0.01
    source_waveform_duration_s: float = 0.002
    microphone_noise_std: float = 0.0
    source_waveform_noise_std: float = 0.0
    random_seed: int = 42
    fb_nu_values: tuple[int, ...] = (2, 4, 8, 16)
    output_dir: Path = Path("outputs")

    def __post_init__(self) -> None:
        if self.array_elements <= 0:
            raise ValueError("array_elements must be positive")
        if self.array_arms <= 0:
            raise ValueError("array_arms must be positive")
        if self.array_aperture_m <= 0:
            raise ValueError("array_aperture_m must be positive")
        if self.scan_step_m <= 0:
            raise ValueError("scan_step_m must be positive")
        if self.tick_step_m <= 0:
            raise ValueError("tick_step_m must be positive")
        if self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if self.microphone_noise_std < 0:
            raise ValueError("microphone_noise_std must be non-negative")
        if self.source_waveform_noise_std < 0:
            raise ValueError("source_waveform_noise_std must be non-negative")
