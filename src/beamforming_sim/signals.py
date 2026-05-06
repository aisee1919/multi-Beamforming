from __future__ import annotations

import numpy as np

from beamforming_sim.array_geometry import MicrophoneArray
from beamforming_sim.scene import SourceModel


def simulate_microphone_signals(
    array: MicrophoneArray,
    source_model: SourceModel,
    sampling_rate_hz: float = 192_000.0,
    duration_s: float = 0.01,
    noise_std: float = 0.0,
    random_seed: int | None = None,
    sound_speed_m_s: float = 343.0,
) -> tuple[np.ndarray, np.ndarray]:
    """生成阵列接收信号。

    每个麦克风通道为所有点声源的传播时延正弦叠加；noise_std > 0 时加入加性高斯白噪声。
    """

    sample_count = int(round(sampling_rate_hz * duration_s))
    time_s = np.arange(sample_count, dtype=float) / sampling_rate_hz
    signals = np.zeros((len(array.positions_m), sample_count), dtype=float)

    for source in source_model.sources:
        source_to_mics_m = np.linalg.norm(array.positions_m - source.position_m, axis=1)
        propagation_delay_s = source_to_mics_m / sound_speed_m_s
        phase = 2.0 * np.pi * source.frequency_hz * (time_s[None, :] - propagation_delay_s[:, None])
        # 使用 1/r 衰减保留近远关系，避免把传播模型误写成纯相位平移。
        signals += (source.amplitude / source_to_mics_m[:, None]) * np.sin(phase + source.phase_rad)

    if noise_std > 0:
        rng = np.random.default_rng(random_seed)
        signals += rng.normal(loc=0.0, scale=noise_std, size=signals.shape)

    return time_s, signals
