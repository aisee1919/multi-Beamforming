from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

from beamforming_sim.array_geometry import MicrophoneArray
from beamforming_sim.scene import ScanPlane


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
) -> Path:
    """绘制与参考图匹配的 dB 声学热力图。

    energy 按 plane.points_m 顺序排列；图中显示相对峰值 dB，色标范围为 0 到 floor_db。
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 热力图数据必须和扫描网格一一对应，否则图像会错误映射到空间位置。
    expected_size = len(plane.x_coordinates_m) * len(plane.y_coordinates_m)
    if energy.size != expected_size:
        raise ValueError("energy size must match scan plane grid size")

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
        interpolation="bilinear",
        aspect="equal",
    )
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

    if step_m <= 0:
        raise ValueError("step_m must be positive")
    count = int(round((extent_m[1] - extent_m[0]) / step_m)) + 1
    return np.round(extent_m[0] + step_m * np.arange(count, dtype=float), decimals=12)


def _image_extent(plane: ScanPlane) -> tuple[float, float, float, float]:
    """让图像边界与扫描平面坐标一致。"""

    return (
        float(np.min(plane.x_coordinates_m)),
        float(np.max(plane.x_coordinates_m)),
        float(np.min(plane.y_coordinates_m)),
        float(np.max(plane.y_coordinates_m)),
    )
