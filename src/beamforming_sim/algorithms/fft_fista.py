from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beamforming_sim.algorithms.cbf import cbf_power_from_csm, steering_matrix
from beamforming_sim.array_geometry import MicrophoneArray
from beamforming_sim.domain import BeamformingResult
from beamforming_sim.scene import ScanPlane


@dataclass(frozen=True)
class FFTFISTABeamformer:
    """FFT-FISTA 反卷积声学成像算法。

    在 CBF dirty map 上通过 FFT 加速 FISTA 求解带 L1 正则和非负约束的反卷积问题。
    PSF 使用中心扫描点的点扩散响应作为平移不变近似。
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

    def run_from_cbf_map(
        self,
        cbf_result: BeamformingResult,
        array: MicrophoneArray,
        frequency_hz: float,
        sound_speed_m_s: float = 343.0,
    ) -> BeamformingResult:
        """从 CBF 原始功率图出发，运行 FFT-FISTA 反卷积。

        cbf_result.raw_power 用作 dirty map，必须是未归一化的线性能量。
        """

        plane = cbf_result.plane
        n_y = len(plane.y_coordinates_m)
        n_x = len(plane.x_coordinates_m)

        dirty_map = cbf_result.raw_power.reshape(n_y, n_x)

        # —— 构造 PSF：中心扫描点对单位点声源的 CBF 响应 ——
        psf_2d = _build_center_psf(array, plane, frequency_hz, sound_speed_m_s, n_y, n_x)

        # —— FISTA 迭代求解 ——
        x = _fista_deconvolution(
            dirty_map=dirty_map,
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
    """用中心扫描点的单位点声源响应构造平移不变近似 PSF。

    注意：这是平移不变近似，实际声场中的 PSF 是平移变化的。
    """

    # 选取距离扫描平面几何中心最近的网格点
    center = np.array([0.0, 0.0, plane.distance_m])
    dists = np.linalg.norm(plane.points_m - center, axis=1)
    center_idx = int(np.argmin(dists))

    # 中心点 → 阵列的导向矢量
    a_center = steering_matrix(
        array.positions_m,
        plane.points_m[center_idx : center_idx + 1],
        frequency_hz,
        sound_speed_m_s,
    )[0]

    # 单位声源 CSM (rank-1)
    csm_unit = np.outer(a_center, a_center.conj())

    # PSF = CBF 对该单位声源的响应
    psf_1d = cbf_power_from_csm(array, plane, csm_unit, frequency_hz, sound_speed_m_s)
    psf_2d = psf_1d.reshape(n_y, n_x)

    # 归一化到最大值 1
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

    其中 H 是 PSF 卷积算子，b 是 dirty map。
    """

    # 频域 PSF 及 Lipschitz 常数
    # ifftshift 将 PSF 峰值从图像中心移至 (0,0)，保证 FFT 卷积时延迟对准正确位置
    H = np.fft.fft2(np.fft.ifftshift(psf_2d))
    L = float(np.max(np.abs(H) ** 2))
    if L <= 0:
        raise ValueError("Lipschitz constant must be positive")

    # 迭代初值
    x = np.zeros_like(dirty_map)
    y = x.copy()
    t = 1.0

    threshold = lambda_reg / L
    inv_L = 1.0 / L

    for _ in range(max_iterations):
        # 前向卷积: A(y) = PSF ∗ y
        Ay = np.real(np.fft.ifft2(H * np.fft.fft2(y)))

        # 残差梯度: A^T(Ay - b)
        residual = Ay - dirty_map
        grad = np.real(np.fft.ifft2(np.conj(H) * np.fft.fft2(residual)))

        # 梯度下降 + 软阈值 + 非负投影
        x_next = np.maximum(0.0, y - inv_L * grad - threshold)

        # Nesterov 加速
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        momentum = (t - 1.0) / t_next
        y = x_next + momentum * (x_next - x)

        # 收敛判据
        diff = float(np.linalg.norm(x_next - x))
        denom = float(np.linalg.norm(x)) + 1e-15
        if diff / denom < tolerance:
            break

        x = x_next
        t = t_next

    return x
