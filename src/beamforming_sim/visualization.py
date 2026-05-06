from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

from beamforming_sim.array_geometry import MicrophoneArray
from beamforming_sim.scene import ScanPlane, SourceModel


def plot_array_layout(array: MicrophoneArray, output_path: str | Path) -> Path:
    """绘制麦克风阵列的 x-y 平面布局。"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    positions = array.positions_m
    ax.scatter(positions[:, 0], positions[:, 1], s=28)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("Eight-arm spiral microphone array")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_energy_heatmap(
    plane: ScanPlane,
    energy: np.ndarray,
    output_path: str | Path,
    title: str | None = None,
    floor_db: float = -50.0,
    tick_step_m: float = 0.1,
    interpolation: str = "bilinear",
    peak_position: tuple[float, float] | None = None,
) -> Path:
    """绘制 dB 声学热力图。

    energy 按 plane.points_m 顺序排列；图中显示相对峰值 dB，色标范围为 0 到 floor_db。
    可通过 interpolation="nearest" 使 DAMAS 等稀疏结果的网格点可见。
    peak_position 为可选的 (x, y) 坐标，标记峰值位置。
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    db_grid = energy_to_db(np.asarray(energy, dtype=float), floor_db=floor_db).reshape(
        len(plane.y_coordinates_m),
        len(plane.x_coordinates_m),
    )
    masked_db_grid = np.ma.masked_less(db_grid, floor_db)

    fig, ax = plt.subplots(figsize=(4.6, 3.7))
    cmap = colormaps["jet"].copy()
    cmap.set_bad("white")

    image = ax.imshow(
        masked_db_grid,
        origin="lower",
        extent=_image_extent(plane),
        cmap=cmap,
        vmin=floor_db,
        vmax=0.0,
        interpolation=interpolation,
        aspect="equal",
    )

    if peak_position is not None:
        ax.plot(peak_position[0], peak_position[1], "r*", markersize=10, markeredgewidth=0.5)

    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_xticks(tick_values((plane.x_coordinates_m[0], plane.x_coordinates_m[-1]), tick_step_m))
    ax.set_yticks(tick_values((plane.y_coordinates_m[0], plane.y_coordinates_m[-1]), tick_step_m))
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_ticks([0, -10, -20, -30, -40, floor_db])
    colorbar.ax.set_title("dB", fontsize=8, pad=3)
    colorbar.ax.tick_params(labelsize=7)

    caption = title or f"CBF({plane.distance_m:g}m)"
    ax.text(
        0.5,
        -0.33,
        caption,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=14,
        family="serif",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def energy_to_db(energy: np.ndarray, floor_db: float = -50.0) -> np.ndarray:
    """把线性能量转换为参考图使用的相对 dB 标尺。"""

    energy = np.asarray(energy, dtype=float)
    max_energy = float(np.max(energy))
    if max_energy <= 0.0:
        return np.full_like(energy, floor_db - 1.0, dtype=float)

    safe_energy = np.maximum(energy / max_energy, 10.0 ** ((floor_db - 1.0) / 10.0))
    return 10.0 * np.log10(safe_energy)


def tick_values(extent_m: tuple[float, float], step_m: float = 0.1) -> np.ndarray:
    """生成图上显示用的坐标刻度，不影响 CBF 实际扫描步长。"""

    count = int(round((extent_m[1] - extent_m[0]) / step_m)) + 1
    return np.round(extent_m[0] + step_m * np.arange(count, dtype=float), decimals=12)


def plot_microphone_signals(
    time_s: np.ndarray,
    signals: np.ndarray,
    array: MicrophoneArray,
    output_path: str | Path,
    channel_indices: list[int] | None = None,
    title: str | None = None,
) -> Path:
    """绘制选定通道的时域波形，垂直偏移以展示通道间相位差。

    默认每臂取一个阵元（8 路），避免 64 路过密。
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if channel_indices is None:
        elements_per_arm = len(array.positions_m) // array.config.arms
        channel_indices = list(range(0, len(array.positions_m), elements_per_arm))

    sample_count = min(len(time_s), 150)
    t = time_s[:sample_count]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    offset_step = 1.5 * float(np.max(np.abs(signals)))

    for idx, ch in enumerate(channel_indices):
        y = signals[ch, :sample_count] + idx * offset_step
        ax.plot(t * 1e3, y, linewidth=0.6, label=f"ch{ch}")

    ax.set_xlabel("t / ms")
    ax.set_ylabel("amplitude (offset for clarity)")
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Microphone signals (one per arm)")
    ax.legend(loc="upper right", fontsize=6, ncol=2, framealpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_source_signals(
    source_model: SourceModel,
    sampling_rate_hz: float,
    duration_s: float,
    output_path: str | Path,
    noise_std: float = 0.0,
    random_seed: int | None = None,
    title: str | None = None,
) -> Path:
    """绘制声源发射端的原始正弦波形，可选叠加加性高斯白噪声。

    noise_std > 0 时生成三列子图：纯净信号 | AWGN | 叠噪信号。
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_count = int(round(sampling_rate_hz * min(duration_s, 0.002)))
    t = np.arange(sample_count) / sampling_rate_hz
    source = source_model.sources[0]
    clean = source.amplitude * np.sin(2.0 * np.pi * source.frequency_hz * t + source.phase_rad)

    if noise_std > 0:
        rng = np.random.default_rng(random_seed)
        noise = rng.normal(loc=0.0, scale=noise_std, size=sample_count)
        noisy = clean + noise

        fig, axes = plt.subplots(3, 1, figsize=(8, 5.5))
        labels = ("clean signal", "AWGN (σ={:.2f})".format(noise_std), "signal + noise")
        data = (clean, noise, noisy)
        colors = ("#1f77b4", "#d62728", "#9467bd")
        for ax, label, y, color in zip(axes, labels, data, colors):
            ax.plot(t * 1e6, y, linewidth=0.5, color=color)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("t / µs")
            ax.tick_params(labelsize=7)
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 2.8), squeeze=False)
        ax = axes[0, 0]
        ax.plot(t * 1e6, clean, linewidth=0.7)
        ax.set_ylabel(f"{source.frequency_hz / 1000:.0f} kHz" if source.frequency_hz >= 1000 else f"{source.frequency_hz:.0f} Hz")
        ax.tick_params(labelsize=7)
        ax.set_xlabel("t / µs")

    if title:
        fig.suptitle(title, fontsize=12)
    else:
        fig.suptitle("Source waveform (emission point)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _image_extent(plane: ScanPlane) -> tuple[float, float, float, float]:
    """让图像边界与扫描平面坐标一致。"""

    return (
        float(np.min(plane.x_coordinates_m)),
        float(np.max(plane.x_coordinates_m)),
        float(np.min(plane.y_coordinates_m)),
        float(np.max(plane.y_coordinates_m)),
    )
