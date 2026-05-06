from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from beamforming_sim.algorithms.cbf import point_chunks
from beamforming_sim.array_geometry import MicrophoneArray
from beamforming_sim.domain import BeamformingResult
from beamforming_sim.scene import ScanPlane


@dataclass(frozen=True)
class DAMASFISTABeamformer:
    """DAMAS-FISTA 反卷积声学成像。

    使用 FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) 求解 NNLS 问题：
        min  0.5 ||A x - b||^2   s.t.  x >= 0

    参考: Liang et al., "Learning an Interpretable End-to-End Network for
    Real-Time Acoustic Beamforming", JSV 2024, Algorithm 1.
    """

    max_iterations: int = 1000
    tolerance: float = 1e-4
    dense_point_limit: int = 5000
    scan_chunk_size: int = 256
    max_point_count: int | None = 5000

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if self.dense_point_limit <= 0:
            raise ValueError("dense_point_limit must be positive")
        if self.scan_chunk_size <= 0:
            raise ValueError("scan_chunk_size must be positive")

    @property
    def name(self) -> str:
        return "DAMAS-FISTA"

    def run_from_cbf_map(
        self,
        cbf_result: BeamformingResult,
        array: MicrophoneArray,
        sound_speed_m_s: float | None = None,
    ) -> BeamformingResult:
        """从 CBF dirty map 出发运行 DAMAS-FISTA 反卷积。

        frequency_hz / sound_speed_m_s 取自 cbf_result，避免手动传错。
        """
        frequency_hz = cbf_result.frequency_hz
        sound_speed_m_s = sound_speed_m_s if sound_speed_m_s is not None else cbf_result.sound_speed_m_s

        plane = cbf_result.plane
        n_points = len(plane.points_m)

        if self.max_point_count is not None and n_points > self.max_point_count:
            raise ValueError(
                f"DAMAS-FISTA point count ({n_points}) exceeds max_point_count "
                f"({self.max_point_count}). Use a coarser grid or explicitly "
                f"override max_point_count."
            )

        t0 = time.perf_counter()

        # Build PSF operators matching the CBF dirty map formulation:
        # b_i = w_i^H C w_i  with w_i = a_i / ||a_i||^2
        # A_ij = |w_i^H a_j|^2
        operators = _build_operators(
            array.positions_m,
            plane.points_m,
            frequency_hz,
            sound_speed_m_s,
            self.dense_point_limit,
            self.scan_chunk_size,
        )

        result = _fista_nnls_solve(
            b=cbf_result.raw_power,
            matvec=operators["matvec"],
            rmatvec=operators["rmatvec"],
            lipschitz=operators["lipschitz"],
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
        )

        runtime_s = time.perf_counter() - t0

        return BeamformingResult(
            algorithm=self.name,
            plane=plane,
            raw_power=result["x"],
            frequency_hz=frequency_hz,
            sound_speed_m_s=sound_speed_m_s,
            metadata={
                "iterations": result["iterations"],
                "relative_change": result["relative_change"],
                "residual_norm": result["residual_norm"],
                "lipschitz": operators["lipschitz"],
                "matrix_mode": operators["matrix_mode"],
                "point_count": n_points,
                "converged": result["converged"],
                "runtime_s": runtime_s,
            },
        )


# ---------------------------------------------------------------------------
# Steering / PSF construction
# ---------------------------------------------------------------------------

def _steering_matrix(
    mic_positions: np.ndarray,
    scan_points: np.ndarray,
    frequency_hz: float,
    sound_speed_m_s: float,
) -> np.ndarray:
    """生成导向矩阵 G: (N_points, M_mics)。

    g_n[m] = (r0 / r_{m,n}) * exp(-j * k * (r_{m,n} - r0))

    其中 r0 是阵列中心到扫描点的距离，与论文公式 (1) 一致。
    """
    k = 2.0 * np.pi * frequency_hz / sound_speed_m_s
    distances_m = np.linalg.norm(scan_points[:, None, :] - mic_positions[None, :, :], axis=2)
    r0 = np.linalg.norm(scan_points, axis=1)  # (N,) 距离阵列中心
    G = (r0[:, None] / distances_m) * np.exp(-1j * k * (distances_m - r0[:, None]))
    return G


def _build_operators(
    mic_positions: np.ndarray,
    scan_points: np.ndarray,
    frequency_hz: float,
    sound_speed_m_s: float,
    dense_point_limit: int,
    scan_chunk_size: int,
) -> dict:
    """构造 PSF 矩阵 A 及其算子。

    A_ij = |w_i^H a_j|^2  (与 CBF dirty map 的权重 w_i 保持一致)

    w_i = a_i / ||a_i||^2  — 即 CBF 使用的单位增益导向矢量
    a_j — 原始导向矢量 (steering vector)
    """
    from beamforming_sim.algorithms.cbf import steering_matrix as _steer

    n_points = len(scan_points)
    A_full = _steer(mic_positions, scan_points, frequency_hz, sound_speed_m_s)
    norms = np.sum(np.abs(A_full) ** 2, axis=1)
    W = A_full / norms[:, None]  # w_i = a_i / ||a_i||^2

    if n_points <= dense_point_limit:
        P = np.abs(W @ A_full.conj().T) ** 2
        lip = _lipschitz_from_matrix(P)

        def matvec(x):
            return P @ x

        def rmatvec(y):
            return P.T @ y

        matrix_mode = "dense"
    else:
        AH = A_full.conj().T

        def matvec(x):
            result = np.empty(n_points)
            for start, stop in point_chunks(n_points, scan_chunk_size):
                P_chunk = np.abs(W[start:stop] @ AH) ** 2
                result[start:stop] = P_chunk @ x
            return result

        def rmatvec(y):
            result = np.zeros(n_points)
            for start, stop in point_chunks(n_points, scan_chunk_size):
                P_chunk = np.abs(W[start:stop] @ AH) ** 2
                result += P_chunk.T @ y[start:stop]
            return result

        lip = _power_iteration_lipschitz(matvec, rmatvec, n_points)
        matrix_mode = "chunked"

    return {"matvec": matvec, "rmatvec": rmatvec, "lipschitz": lip, "matrix_mode": matrix_mode}


def _lipschitz_from_matrix(P: np.ndarray) -> float:
    """L = max eigenvalue of P^T P."""
    return float(np.linalg.norm(P, 2)) ** 2


def _power_iteration_lipschitz(matvec, rmatvec, n: int, n_iter: int = 30) -> float:
    """幂迭代估计 P^T P 的最大特征值。"""
    v = np.random.randn(n)
    v /= np.linalg.norm(v)
    for _ in range(n_iter):
        v = rmatvec(matvec(v))
        v /= np.linalg.norm(v)
    return float(np.sum(matvec(v) ** 2))


# ---------------------------------------------------------------------------
# FISTA-NNLS solver (Algorithm 1 from the paper)
# ---------------------------------------------------------------------------

def _fista_nnls_solve(
    b: np.ndarray,
    matvec,
    rmatvec,
    lipschitz: float,
    max_iterations: int,
    tolerance: float,
) -> dict:
    """FISTA 求解 NNLS:  min 0.5||Ax - b||^2  s.t. x >= 0.

    Implementation of Algorithm 1 from Liang et al. (JSV 2024),
    with gradient-based adaptive restart to prevent support collapse
    under the non-negativity constraint.
    """
    inv_L = 1.0 / lipschitz
    n = len(b)

    # x^(0) = 0,  y^(1) = 0,  t^(1) = 1
    x = np.zeros(n)
    y = x.copy()
    t = 1.0

    converged = False
    relative_change = float("inf")

    for k in range(max_iterations):
        # Step 1:  x^(k) = max( y^(k) - (1/L) A^T (A y^(k) - b),  0 )
        residual = matvec(y) - b
        grad = rmatvec(residual)
        x_next = np.maximum(0.0, y - inv_L * grad)

        # Step 2:  t^(k+1) = (1 + sqrt(1 + 4 t^(k)^2)) / 2
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))

        # Step 3:  y^(k+1) = x^(k) + (t^(k)-1)/t^(k+1) * (x^(k) - x^(k-1))
        momentum = (t - 1.0) / t_next
        dx = x_next - x
        y_next = x_next + momentum * dx

        # Adaptive restart: if momentum opposes the descent direction,
        # reset to prevent non-negativity clamping from collapsing support.
        if k > 0 and np.dot(y_next - x_next, dx) < 0:
            y_next = x_next
            t_next = 1.0
            momentum = 0.0

        diff = float(np.linalg.norm(dx))
        denom = float(np.linalg.norm(x)) + 1e-15
        relative_change = diff / denom

        x = x_next
        y = y_next
        t = t_next

        if relative_change < tolerance:
            converged = True
            break

    residual_norm = float(np.linalg.norm(matvec(x) - b))

    return {
        "x": x,
        "iterations": k + 1,
        "relative_change": relative_change,
        "residual_norm": residual_norm,
        "converged": converged,
    }
