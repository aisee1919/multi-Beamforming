from __future__ import annotations

import numpy as np

from beamforming_sim.algorithms import (
    ConventionalBeamformer,
    FunctionalBeamformer,
    cbf_power_from_csm,
    csm_power_eig,
    fb_power_from_transformed_csm,
    normalize_power,
    steering_matrix,
)
from beamforming_sim.algorithms.base import validate_beamforming_inputs
from beamforming_sim.array_geometry import MicrophoneArray
from beamforming_sim.scene import ScanPlane
from beamforming_sim.spectral import compute_cross_spectral_matrix, validate_signal_inputs


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


def functional_beamforming(
    array: MicrophoneArray,
    plane: ScanPlane,
    signals: np.ndarray,
    sampling_rate_hz: float,
    frequency_hz: float,
    nu: int = 2,
    sound_speed_m_s: float = 343.0,
) -> np.ndarray:
    """兼容入口：返回归一化后的 FB 能量图。"""

    result = FunctionalBeamformer(nu=nu).run(array, plane, signals, sampling_rate_hz, frequency_hz, sound_speed_m_s)
    return result.normalized_power()


def run_fb_for_planes(
    array: MicrophoneArray,
    planes: list[ScanPlane],
    signals: np.ndarray,
    sampling_rate_hz: float,
    frequency_hz: float,
    nu: int = 2,
    sound_speed_m_s: float = 343.0,
) -> dict[float, np.ndarray]:
    """兼容入口：复用 CSM^(1/nu) 后对多个扫描平面返回归一化 FB 能量图。"""

    validate_beamforming_inputs(array, signals, sampling_rate_hz, frequency_hz, sound_speed_m_s)
    csm = compute_cross_spectral_matrix(signals, sampling_rate_hz, frequency_hz)
    csm_pow = csm_power_eig(csm, 1.0 / nu)
    return {
        plane.distance_m: normalize_power(
            fb_power_from_transformed_csm(array, plane, csm_pow, nu, frequency_hz, sound_speed_m_s)
        )
        for plane in planes
    }


def _cbf_from_csm(
    array: MicrophoneArray,
    plane: ScanPlane,
    csm: np.ndarray,
    frequency_hz: float,
    sound_speed_m_s: float,
) -> np.ndarray:
    """兼容旧测试和调试代码：从 CSM 返回归一化 CBF 能量。"""

    return normalize_power(cbf_power_from_csm(array, plane, csm, frequency_hz, sound_speed_m_s))


def _fb_from_csm(
    array: MicrophoneArray,
    plane: ScanPlane,
    csm_pow: np.ndarray,
    nu: int,
    frequency_hz: float,
    sound_speed_m_s: float,
) -> np.ndarray:
    """兼容旧测试和调试代码：从 CSM^(1/nu) 返回归一化 FB 能量。"""

    return normalize_power(fb_power_from_transformed_csm(array, plane, csm_pow, nu, frequency_hz, sound_speed_m_s))


_steering_matrix = steering_matrix
_normalize_energy = normalize_power
_csm_power_eig = csm_power_eig
_validate_signal_inputs = validate_signal_inputs
