from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beamforming_sim.algorithms.base import CsmBasedBeamformer, validate_beamforming_inputs, validate_csm
from beamforming_sim.array_geometry import MicrophoneArray
from beamforming_sim.domain import BeamformingResult
from beamforming_sim.scene import ScanPlane
from beamforming_sim.spectral import compute_cross_spectral_matrix


@dataclass(frozen=True)
class ConventionalBeamformer(CsmBasedBeamformer):
    """传统波束形成算法。

    类接口返回原始功率；旧函数接口需要兼容历史行为时再做归一化。
    """

    scan_chunk_size: int | None = None

    @property
    def name(self) -> str:
        return "CBF"

    def run_from_csm(
        self,
        array: MicrophoneArray,
        plane: ScanPlane,
        csm: np.ndarray,
        frequency_hz: float,
        sound_speed_m_s: float = 343.0,
    ) -> BeamformingResult:
        validate_csm(array, csm)
        power = cbf_power_from_csm(
            array,
            plane,
            csm,
            frequency_hz,
            sound_speed_m_s,
            scan_chunk_size=self.scan_chunk_size,
        )
        return BeamformingResult(
            algorithm=self.name,
            plane=plane,
            raw_power=power,
            frequency_hz=frequency_hz,
            sound_speed_m_s=sound_speed_m_s,
        )


def cbf_power_from_csm(
    array: MicrophoneArray,
    plane: ScanPlane,
    csm: np.ndarray,
    frequency_hz: float,
    sound_speed_m_s: float,
    scan_chunk_size: int | None = None,
) -> np.ndarray:
    """使用 P = w^H R w 计算扫描平面的 CBF 原始功率。"""

    validate_csm(array, csm)
    power = np.empty(len(plane.points_m), dtype=float)
    for start, stop in point_chunks(len(plane.points_m), scan_chunk_size):
        steering = steering_matrix(array.positions_m, plane.points_m[start:stop], frequency_hz, sound_speed_m_s)
        power[start:stop] = quadratic_power_from_steering(csm, steering)
    return power


def steering_matrix(
    microphone_positions_m: np.ndarray,
    scan_points_m: np.ndarray,
    frequency_hz: float,
    sound_speed_m_s: float,
) -> np.ndarray:
    """生成近场导向矩阵，形状为 (扫描点数, 阵元数)。"""

    wave_number = 2.0 * np.pi * frequency_hz / sound_speed_m_s
    distances_m = np.linalg.norm(scan_points_m[:, None, :] - microphone_positions_m[None, :, :], axis=2)
    return np.exp(-1j * wave_number * distances_m) / distances_m


def quadratic_power_from_steering(csm: np.ndarray, steering: np.ndarray) -> np.ndarray:
    """对已生成的导向矩阵计算二次型功率。"""

    normalization = np.sum(np.abs(steering) ** 2, axis=1, keepdims=True)
    weights = steering / normalization
    power = np.einsum("pm,mn,pn->p", weights.conj(), csm, weights, optimize=True).real
    return np.maximum(power, 0.0)


def point_chunks(point_count: int, scan_chunk_size: int | None) -> list[tuple[int, int]]:
    """按扫描点切块，避免密集网格下一次性构造过大的导向矩阵。"""

    if scan_chunk_size is None:
        return [(0, point_count)]
    if scan_chunk_size <= 0:
        raise ValueError("scan_chunk_size must be positive")
    return [(start, min(start + scan_chunk_size, point_count)) for start in range(0, point_count, scan_chunk_size)]


def conventional_beamforming(
    array: MicrophoneArray,
    plane: ScanPlane,
    signals: np.ndarray,
    sampling_rate_hz: float,
    frequency_hz: float,
    sound_speed_m_s: float = 343.0,
) -> np.ndarray:
    """兼容入口：返回归一化后的 CBF 能量图。

    新代码应优先使用 ConventionalBeamformer().run(...) 获取原始功率结果。
    """

    result = ConventionalBeamformer().run(array, plane, signals, sampling_rate_hz, frequency_hz, sound_speed_m_s)
    return result.normalized_power()


def run_cbf_for_planes(
    array: MicrophoneArray,
    planes: list[ScanPlane],
    signals: np.ndarray,
    sampling_rate_hz: float,
    frequency_hz: float,
    sound_speed_m_s: float = 343.0,
) -> dict[float, np.ndarray]:
    """兼容入口：复用 CSM 后对多个扫描平面返回归一化 CBF 能量图。"""

    validate_beamforming_inputs(array, signals, sampling_rate_hz, frequency_hz, sound_speed_m_s)
    csm = compute_cross_spectral_matrix(signals, sampling_rate_hz, frequency_hz)
    beamformer = ConventionalBeamformer()
    return {
        plane.distance_m: beamformer.run_from_csm(array, plane, csm, frequency_hz, sound_speed_m_s).normalized_power()
        for plane in planes
    }
