from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ArrayParams:
    """麦克风阵列几何参数。"""

    elements: int = 64
    arms: int = 8
    aperture_m: float = 0.15


@dataclass(frozen=True)
class ScanParams:
    """扫描平面参数。"""

    distances_m: tuple[float, ...] = (1.2, 1.6, 2.0)
    extent_m: tuple[float, float] = (-0.6, 0.6)
    step_m: float = 0.01
    tick_step_m: float = 0.1


@dataclass(frozen=True)
class SourceParams:
    """声源参数。"""

    xy_positions_m: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.5, 0.0), (-0.5, 0.0))
    frequency_hz: float = 25_000.0


@dataclass(frozen=True)
class SignalParams:
    """信号采集与处理参数。"""

    sampling_rate_hz: float = 192_000.0
    duration_s: float = 0.01
    source_waveform_duration_s: float = 0.002
    microphone_noise_std: float = 0.0
    source_waveform_noise_std: float = 0.0
    random_seed: int = 42


@dataclass(frozen=True)
class ExperimentConfig:
    """实验配置，按职责分组。"""

    array: ArrayParams = field(default_factory=ArrayParams)
    scan: ScanParams = field(default_factory=ScanParams)
    source: SourceParams = field(default_factory=SourceParams)
    signal: SignalParams = field(default_factory=SignalParams)
    fb_nu_values: tuple[int, ...] = (2, 4, 8, 16)
    output_dir: Path = Path("outputs")
