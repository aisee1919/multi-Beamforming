from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beamforming_sim.algorithms.base import CsmBasedBeamformer, validate_csm
from beamforming_sim.algorithms.cbf import point_chunks, quadratic_power_from_steering, steering_matrix
from beamforming_sim.array_geometry import MicrophoneArray
from beamforming_sim.domain import BeamformingResult
from beamforming_sim.scene import ScanPlane


@dataclass(frozen=True)
class FunctionalBeamformer(CsmBasedBeamformer):
    """函数波束形成算法。"""

    nu: int = 2
    scan_chunk_size: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.nu, int) or self.nu < 1:
            raise ValueError("nu must be a positive integer")

    @property
    def name(self) -> str:
        return "FB"

    def run_from_csm(
        self,
        array: MicrophoneArray,
        plane: ScanPlane,
        csm: np.ndarray,
        frequency_hz: float,
        sound_speed_m_s: float = 343.0,
    ) -> BeamformingResult:
        validate_csm(array, csm)
        csm_pow = csm_power_eig(csm, 1.0 / self.nu)
        power = fb_power_from_transformed_csm(
            array,
            plane,
            csm_pow,
            self.nu,
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
            metadata={"nu": self.nu},
        )


def fb_power_from_transformed_csm(
    array: MicrophoneArray,
    plane: ScanPlane,
    transformed_csm: np.ndarray,
    nu: int,
    frequency_hz: float,
    sound_speed_m_s: float,
    scan_chunk_size: int | None = None,
) -> np.ndarray:
    """使用 CSM^(1/nu) 计算 FB 原始功率。"""

    validate_csm(array, transformed_csm)
    power = np.empty(len(plane.points_m), dtype=float)
    for start, stop in point_chunks(len(plane.points_m), scan_chunk_size):
        steering = steering_matrix(array.positions_m, plane.points_m[start:stop], frequency_hz, sound_speed_m_s)
        raw = quadratic_power_from_steering(transformed_csm, steering)
        power[start:stop] = raw**nu
    return power


def csm_power_eig(csm: np.ndarray, exponent: float) -> np.ndarray:
    """通过特征分解计算 Hermitian CSM 的分数次幂。"""

    eigenvalues, eigenvectors = np.linalg.eigh(csm)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    powered_eigenvalues = eigenvalues**exponent
    return (eigenvectors * powered_eigenvalues[None, :]) @ eigenvectors.conj().T
