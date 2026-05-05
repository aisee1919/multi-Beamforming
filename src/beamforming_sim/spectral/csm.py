from __future__ import annotations

import numpy as np


def compute_cross_spectral_matrix(
    signals: np.ndarray,
    sampling_rate_hz: float,
    frequency_hz: float,
    block_size: int | None = None,
    overlap: float = 0.5,
) -> np.ndarray:
    """计算目标频率处的平均交叉谱矩阵。

    输入信号形状为 (通道数, 采样点数)。长时间序列可按 block_size 分块 FFT 后平均。
    """

    validate_signal_inputs(signals, sampling_rate_hz, frequency_hz)
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


def validate_signal_inputs(signals: np.ndarray, sampling_rate_hz: float, frequency_hz: float) -> None:
    """校验频谱估计所需的基本信号条件。"""

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
