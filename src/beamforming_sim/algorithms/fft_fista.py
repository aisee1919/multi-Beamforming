from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beamforming_sim.algorithms.base import CsmBasedBeamformer, validate_csm
from beamforming_sim.algorithms.cbf import cbf_power_from_csm, steering_matrix
from beamforming_sim.array_geometry import MicrophoneArray
from beamforming_sim.domain import BeamformingResult
from beamforming_sim.scene import ScanPlane


@dataclass(frozen=True)
class FFTFISTABeamformer(CsmBasedBeamformer):
    """FFT-FISTA 反卷积声学成像算法。

    在 CBF dirty map 上通过 FFT 加速 FISTA 求解带 L1 正则和非负约束的反卷积问题。
    PSF 使用中心扫描点的点扩散响应作为平移不变近似。
    实现了 Beamformer Protocol (run / run_from_csm)，可与其他波束形成器多态使用。
    """

    lambda_reg: float = 0.02
    max_iterations: int = 200
    tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if self.lambda_reg < 0:
            raise ValueError("lambda_reg must be non-negative")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")

    @property
    def name(self) -> str:
        return "FFT-FISTA"

    def run_from_csm(
        self,
        array: MicrophoneArray,
        plane: ScanPlane,
        csm: np.ndarray,
        frequency_hz: float,
        sound_speed_m_s: float = 343.0,
    ) -> BeamformingResult:
        """从 CSM 出发：计算 CBF dirty map → FISTA 反卷积。"""
        validate_csm(array, csm)
        dirty_map = cbf_power_from_csm(array, plane, csm, frequency_hz, sound_speed_m_s)
        return self._deconvolve(array, plane, dirty_map, frequency_hz, sound_speed_m_s)

    def run_from_cbf_map(
        self,
        cbf_result: BeamformingResult,
        array: MicrophoneArray,
        frequency_hz: float,
        sound_speed_m_s: float = 343.0,
    ) -> BeamformingResult:
        """从 CBF 原始功率图出发，运行 FFT-FISTA 反卷积。"""
        return self._deconvolve(array, cbf_result.plane, cbf_result.raw_power, frequency_hz, sound_speed_m_s)

    def _deconvolve(
        self,
        array: MicrophoneArray,
        plane: ScanPlane,
        dirty_map: np.ndarray,
        frequency_hz: float,
        sound_speed_m_s: float,
    ) -> BeamformingResult:
        n_y = len(plane.y_coordinates_m)
        n_x = len(plane.x_coordinates_m)

        dirty_map_2d = dirty_map.reshape(n_y, n_x)
        psf_2d = _build_center_psf(array, plane, frequency_hz, sound_speed_m_s, n_y, n_x)
        x = _fista_deconvolution(
            dirty_map=dirty_map_2d,
            psf_2d=psf_2d,
            lambda_reg=self.lambda_reg,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
        )

        return BeamformingResult(
            algorithm=self.name,
            plane=plane,
            raw_power=x.ravel(),
            frequency_hz=frequency_hz,
            sound_speed_m_s=sound_speed_m_s,
            metadata={
                "lambda_reg": self.lambda_reg,
                "max_iterations": self.max_iterations,
            },
        )


def _build_center_psf(
    array: MicrophoneArray,
    plane: ScanPlane,
    frequency_hz: float,
    sound_speed_m_s: float,
    n_y: int,
    n_x: int,
) -> np.ndarray:
    """用中心扫描点的单位点声源响应构造平移不变近似 PSF。"""

    center = np.array([0.0, 0.0, plane.distance_m])
    dists = np.linalg.norm(plane.points_m - center, axis=1)
    center_idx = int(np.argmin(dists))

    a_center = steering_matrix(
        array.positions_m,
        plane.points_m[center_idx : center_idx + 1],
        frequency_hz,
        sound_speed_m_s,
    )[0]

    csm_unit = np.outer(a_center, a_center.conj())
    psf_1d = cbf_power_from_csm(array, plane, csm_unit, frequency_hz, sound_speed_m_s)
    psf_2d = psf_1d.reshape(n_y, n_x)

    psf_max = float(np.max(psf_2d))
    if psf_max > 0:
        psf_2d = psf_2d / psf_max

    return psf_2d


def _fista_deconvolution(
    dirty_map: np.ndarray,
    psf_2d: np.ndarray,
    lambda_reg: float,
    max_iterations: int,
    tolerance: float,
) -> np.ndarray:
    """FFT-FISTA 反卷积求解器。

    min  0.5 * ||H * x - b||_2^2  +  lambda * ||x||_1
    s.t. x >= 0
    """

    H = np.fft.fft2(np.fft.ifftshift(psf_2d))
    L = float(np.max(np.abs(H) ** 2))
    if L <= 0:
        raise ValueError("Lipschitz constant must be positive")

    x = np.zeros_like(dirty_map)
    y = x.copy()
    t = 1.0

    threshold = lambda_reg / L
    inv_L = 1.0 / L

    for _ in range(max_iterations):
        Ay = np.real(np.fft.ifft2(H * np.fft.fft2(y)))
        residual = Ay - dirty_map
        grad = np.real(np.fft.ifft2(np.conj(H) * np.fft.fft2(residual)))

        x_next = np.maximum(0.0, y - inv_L * grad - threshold)

        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        momentum = (t - 1.0) / t_next
        y = x_next + momentum * (x_next - x)

        diff = float(np.linalg.norm(x_next - x))
        denom = float(np.linalg.norm(x)) + 1e-15
        if diff / denom < tolerance:
            break

        x = x_next
        t = t_next

    return x
