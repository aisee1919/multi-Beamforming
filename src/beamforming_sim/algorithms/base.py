from __future__ import annotations

from typing import Protocol

import numpy as np

from beamforming_sim.array_geometry import MicrophoneArray
from beamforming_sim.domain import BeamformingResult
from beamforming_sim.scene import ScanPlane
from beamforming_sim.spectral import compute_cross_spectral_matrix, validate_signal_inputs


class Beamformer(Protocol):
    """所有波束形成算法的统一接口。"""

    def run(
        self,
        array: MicrophoneArray,
        plane: ScanPlane,
        signals: np.ndarray,
        sampling_rate_hz: float,
        frequency_hz: float,
        sound_speed_m_s: float = 343.0,
    ) -> BeamformingResult:
        """基于阵列信号返回某个扫描平面的原始功率结果。"""


def validate_beamforming_inputs(
    array: MicrophoneArray,
    signals: np.ndarray,
    sampling_rate_hz: float,
    frequency_hz: float,
    sound_speed_m_s: float,
) -> None:
    """校验波束形成入口参数，避免算法内部出现隐式形状错误。"""

    validate_signal_inputs(signals, sampling_rate_hz, frequency_hz)
    if signals.shape[0] != len(array.positions_m):
        raise ValueError("signals channel count must match microphone count")
    if sound_speed_m_s <= 0:
        raise ValueError("sound_speed_m_s must be positive")


def validate_csm(array: MicrophoneArray, csm: np.ndarray) -> None:
    """校验交叉谱矩阵和阵元数量是否一致。"""

    expected_shape = (len(array.positions_m), len(array.positions_m))
    if csm.shape != expected_shape:
        raise ValueError("csm shape must match microphone count")


def normalize_power(power: np.ndarray) -> np.ndarray:
    """保留给兼容 API 使用的峰值归一化。"""

    power = np.asarray(power, dtype=float)
    max_power = float(np.max(power))
    if max_power <= 0.0:
        return power
    return power / max_power


class CsmBasedBeamformer:
    """基于 CSM 的波束形成器共用流程：signals → CSM → run_from_csm。

    子类只需实现 run_from_csm()，即可通过 run() 自动获得完整信号→结果流程。
    需要 run_from_signals() 语义时也可直接调用此方法。
    """

    def run(
        self,
        array: MicrophoneArray,
        plane: ScanPlane,
        signals: np.ndarray,
        sampling_rate_hz: float,
        frequency_hz: float,
        sound_speed_m_s: float = 343.0,
        block_size: int | None = None,
        overlap: float = 0.5,
    ) -> BeamformingResult:
        validate_beamforming_inputs(array, signals, sampling_rate_hz, frequency_hz, sound_speed_m_s)
        csm = compute_cross_spectral_matrix(signals, sampling_rate_hz, frequency_hz, block_size, overlap)
        return self.run_from_csm(array, plane, csm, frequency_hz, sound_speed_m_s)

    def run_from_csm(
        self,
        array: MicrophoneArray,
        plane: ScanPlane,
        csm: np.ndarray,
        frequency_hz: float,
        sound_speed_m_s: float = 343.0,
    ) -> BeamformingResult:
        raise NotImplementedError
