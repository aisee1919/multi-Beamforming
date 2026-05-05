from __future__ import annotations

import numpy as np

from beamforming_sim.array_geometry import MicrophoneArray
from beamforming_sim.scene import ScanPlane


def conventional_beamforming(
    array: MicrophoneArray,
    plane: ScanPlane,
    signals: np.ndarray,
    sampling_rate_hz: float,
    frequency_hz: float,
    sound_speed_m_s: float = 343.0,
) -> np.ndarray:
    """传统窄带波束形成。

    先显式计算目标频率的交叉谱矩阵 CSM，再用向量化的 P = w^H R w 计算扫描平面能量。
    """

    _validate_cbf_inputs(array, signals, sampling_rate_hz, frequency_hz, sound_speed_m_s)
    csm = compute_cross_spectral_matrix(signals, sampling_rate_hz, frequency_hz)
    return _cbf_from_csm(array, plane, csm, frequency_hz, sound_speed_m_s)


def run_cbf_for_planes(
    array: MicrophoneArray,
    planes: list[ScanPlane],
    signals: np.ndarray,
    sampling_rate_hz: float,
    frequency_hz: float,
    sound_speed_m_s: float = 343.0,
) -> dict[float, np.ndarray]:
    """对多个扫描平面分别执行 CBF。

    CSM 与扫描平面无关，因此只计算一次，后续平面复用，方便扩展 DAMAS、CLEAN-SC。
    """

    _validate_cbf_inputs(array, signals, sampling_rate_hz, frequency_hz, sound_speed_m_s)
    csm = compute_cross_spectral_matrix(signals, sampling_rate_hz, frequency_hz)
    return {
        plane.distance_m: _cbf_from_csm(array, plane, csm, frequency_hz, sound_speed_m_s)
        for plane in planes
    }


def compute_cross_spectral_matrix(
    signals: np.ndarray,
    sampling_rate_hz: float,
    frequency_hz: float,
    block_size: int | None = None,
    overlap: float = 0.5,
) -> np.ndarray:
    """计算目标频率的平均交叉谱矩阵。

    signals 形状为 (通道数, 采样点数)。长时间序列会按 block_size 分块，逐块 FFT 后平均。
    """

    _validate_signal_inputs(signals, sampling_rate_hz, frequency_hz)
    if block_size is None:
        block_size = signals.shape[1]
    if block_size <= 1:
        raise ValueError("block_size must be greater than one")
    if block_size > signals.shape[1]:
        raise ValueError("block_size must not exceed sample count")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")

    step = max(int(round(block_size * (1.0 - overlap))), 1)
    starts = range(0, signals.shape[1] - block_size + 1, step)
    window = np.hanning(block_size)
    frequencies_hz = np.fft.rfftfreq(block_size, d=1.0 / sampling_rate_hz)
    frequency_index = int(np.argmin(np.abs(frequencies_hz - frequency_hz)))

    csm = np.zeros((signals.shape[0], signals.shape[0]), dtype=np.complex128)
    block_count = 0
    for start in starts:
        block = signals[:, start : start + block_size] * window[None, :]
        snapshot = np.fft.rfft(block, axis=1)[:, frequency_index]
        csm += np.outer(snapshot, snapshot.conj())
        block_count += 1

    csm /= block_count
    return (csm + csm.conj().T) / 2.0


def _cbf_from_csm(
    array: MicrophoneArray,
    plane: ScanPlane,
    csm: np.ndarray,
    frequency_hz: float,
    sound_speed_m_s: float,
) -> np.ndarray:
    """用 NumPy 广播一次性计算所有扫描点的 CBF 能量。"""

    steering = _steering_matrix(array.positions_m, plane.points_m, frequency_hz, sound_speed_m_s)
    normalization = np.sum(np.abs(steering) ** 2, axis=1, keepdims=True)
    weights = steering / normalization
    energy = np.einsum("pm,mn,pn->p", weights.conj(), csm, weights, optimize=True).real
    return _normalize_energy(np.maximum(energy, 0.0))


def _steering_matrix(
    microphone_positions_m: np.ndarray,
    scan_points_m: np.ndarray,
    frequency_hz: float,
    sound_speed_m_s: float,
) -> np.ndarray:
    """生成形状为 (扫描点数, 阵元数) 的近场导向矩阵。"""

    wave_number = 2.0 * np.pi * frequency_hz / sound_speed_m_s
    distances_m = np.linalg.norm(scan_points_m[:, None, :] - microphone_positions_m[None, :, :], axis=2)
    return np.exp(-1j * wave_number * distances_m) / distances_m


def _normalize_energy(energy: np.ndarray) -> np.ndarray:
    """归一化能量，避免不同平面之间因绝对幅值差异导致色标不可读。"""

    max_energy = float(np.max(energy))
    if max_energy <= 0.0:
        return energy
    return energy / max_energy


def functional_beamforming(
    array: MicrophoneArray,
    plane: ScanPlane,
    signals: np.ndarray,
    sampling_rate_hz: float,
    frequency_hz: float,
    nu: int = 2,
    sound_speed_m_s: float = 343.0,
) -> np.ndarray:
    """函数波束形成 (Functional Beamforming)。

    B_FB(ν, x) = [w(x)^H · C^(1/ν) · w(x)]^ν

    ν = 1 时退化为 CBF；ν 越大，主瓣越窄，旁瓣抑制越强。
    """

    if nu < 1 or not isinstance(nu, int):
        raise ValueError("nu must be a positive integer")
    _validate_cbf_inputs(array, signals, sampling_rate_hz, frequency_hz, sound_speed_m_s)
    csm = compute_cross_spectral_matrix(signals, sampling_rate_hz, frequency_hz)
    csm_pow = _csm_power_eig(csm, 1.0 / nu)
    return _fb_from_csm(array, plane, csm_pow, nu, frequency_hz, sound_speed_m_s)


def run_fb_for_planes(
    array: MicrophoneArray,
    planes: list[ScanPlane],
    signals: np.ndarray,
    sampling_rate_hz: float,
    frequency_hz: float,
    nu: int = 2,
    sound_speed_m_s: float = 343.0,
) -> dict[float, np.ndarray]:
    """对多个扫描平面分别执行 FB。CSM^(1/ν) 只计算一次。"""

    if nu < 1 or not isinstance(nu, int):
        raise ValueError("nu must be a positive integer")
    _validate_cbf_inputs(array, signals, sampling_rate_hz, frequency_hz, sound_speed_m_s)
    csm = compute_cross_spectral_matrix(signals, sampling_rate_hz, frequency_hz)
    csm_pow = _csm_power_eig(csm, 1.0 / nu)
    return {
        plane.distance_m: _fb_from_csm(array, plane, csm_pow, nu, frequency_hz, sound_speed_m_s)
        for plane in planes
    }


def _csm_power_eig(csm: np.ndarray, exponent: float) -> np.ndarray:
    """通过特征分解计算 CSM 的分数次幂 C^exponent = U · Σ^exponent · U^H。"""

    eigenvalues, eigenvectors = np.linalg.eigh(csm)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    powered_eigenvalues = eigenvalues**exponent
    return (eigenvectors * powered_eigenvalues[None, :]) @ eigenvectors.conj().T


def _fb_from_csm(
    array: MicrophoneArray,
    plane: ScanPlane,
    csm_pow: np.ndarray,
    nu: int,
    frequency_hz: float,
    sound_speed_m_s: float,
) -> np.ndarray:
    """用向量化 einsum 计算所有扫描点的 FB 能量。"""

    steering = _steering_matrix(array.positions_m, plane.points_m, frequency_hz, sound_speed_m_s)
    normalization = np.sum(np.abs(steering) ** 2, axis=1, keepdims=True)
    weights = steering / normalization
    # w^H · C^(1/ν) · w
    raw = np.einsum("pm,mn,pn->p", weights.conj(), csm_pow, weights, optimize=True).real
    raw = np.maximum(raw, 0.0)
    # 升幂到 ν，产生更尖锐的主瓣
    energy = raw**nu
    return _normalize_energy(energy)


def _validate_cbf_inputs(
    array: MicrophoneArray,
    signals: np.ndarray,
    sampling_rate_hz: float,
    frequency_hz: float,
    sound_speed_m_s: float,
) -> None:
    _validate_signal_inputs(signals, sampling_rate_hz, frequency_hz)
    if signals.shape[0] != len(array.positions_m):
        raise ValueError("signals channel count must match microphone count")
    if sound_speed_m_s <= 0:
        raise ValueError("sound_speed_m_s must be positive")


def _validate_signal_inputs(signals: np.ndarray, sampling_rate_hz: float, frequency_hz: float) -> None:
    if signals.ndim != 2:
        raise ValueError("signals must be a 2D array shaped as channels by samples")
    if signals.shape[1] <= 1:
        raise ValueError("signals must contain at least two samples")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    if frequency_hz >= sampling_rate_hz / 2.0:
        raise ValueError("frequency_hz must be below Nyquist frequency")
