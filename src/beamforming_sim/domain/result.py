from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from beamforming_sim.scene import ScanPlane


@dataclass(frozen=True)
class BeamformingResult:
    """波束形成算法的数值结果。

    算法层只保存原始功率，不在这里做显示归一化；归一化和 dB 截断由调用方按用途决定。
    """

    algorithm: str
    plane: ScanPlane
    raw_power: np.ndarray
    frequency_hz: float
    sound_speed_m_s: float = 343.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        power = np.asarray(self.raw_power, dtype=float)

        object.__setattr__(self, "raw_power", power.copy())
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def normalized_power(self) -> np.ndarray:
        """返回相对峰值归一化功率，专供绘图或单图比较使用。"""

        max_power = float(np.max(self.raw_power))
        if max_power <= 0.0:
            return self.raw_power.copy()
        return self.raw_power / max_power

    def power_db(self, floor_db: float = -50.0) -> np.ndarray:
        """返回以当前结果峰值为 0 dB 的相对声图。"""

        normalized = self.normalized_power()
        if float(np.max(normalized)) <= 0.0:
            return np.full_like(normalized, floor_db - 1.0, dtype=float)
        safe_power = np.maximum(normalized, 10.0 ** ((floor_db - 1.0) / 10.0))
        return 10.0 * np.log10(safe_power)
