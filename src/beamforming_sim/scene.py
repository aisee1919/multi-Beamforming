from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ScanPlane:
    """单个声源扫描平面。

    points_m 为展平后的 (x, y, z) 网格点，后续波束形成算法直接扫描这些点。
    """

    distance_m: float
    x_coordinates_m: np.ndarray
    y_coordinates_m: np.ndarray
    points_m: np.ndarray


@dataclass
class AcousticSource:
    """单个点声源配置。

    position_m 使用三维坐标；二维声源位置会和每个扫描平面距离组合成 (x, y, z)。
    """

    position_m: np.ndarray
    frequency_hz: float = 25_000.0
    amplitude: float = 1.0
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        self.position_m = np.asarray(self.position_m, dtype=float)


@dataclass
class SourceModel:
    """声源模型。"""

    sources: list[AcousticSource] = field(default_factory=list)

    def add_source(self, source: AcousticSource) -> None:
        """向声源模型追加一个外部定义的声源对象。"""

        self.sources.append(source)


def create_default_sources(
    xy_positions_m: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.5, 0.0), (-0.5, 0.0)),
    distances_m: tuple[float, ...] = (1.2, 1.6, 2.0),
    frequency_hz: float = 25_000.0,
) -> SourceModel:
    """在每个扫描平面上创建同一组 25 kHz 点声源。"""

    model = SourceModel()
    for z_m in distances_m:
        for x_m, y_m in xy_positions_m:
            model.add_source(AcousticSource(position_m=np.array([x_m, y_m, z_m]), frequency_hz=frequency_hz))
    return model


def create_scan_planes(
    distances_m: tuple[float, ...] = (1.2, 1.6, 2.0),
    extent_m: tuple[float, float] = (-0.6, 0.6),
    step_m: float = 0.01,
) -> list[ScanPlane]:
    """创建多个与麦克风阵列平行的扫描平面。"""

    # 扫描步长使用 0.01 m，确保 x=+-0.5 m 这类声源位置落在扫描网格上。
    coordinates = _inclusive_grid(extent_m[0], extent_m[1], step_m)
    grid_x, grid_y = np.meshgrid(coordinates, coordinates, indexing="xy")

    planes = []
    for distance in distances_m:
        # 每个扫描平面的 z 坐标固定为其到阵列平面的距离。
        grid_z = np.full_like(grid_x, distance, dtype=float)
        points = np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))
        planes.append(
            ScanPlane(
                distance_m=float(distance),
                x_coordinates_m=coordinates.copy(),
                y_coordinates_m=coordinates.copy(),
                points_m=points,
            )
        )
    return planes


def _inclusive_grid(start: float, stop: float, step: float) -> np.ndarray:
    """生成包含终点的等间隔坐标。"""

    count = int(round((stop - start) / step)) + 1
    values = start + step * np.arange(count, dtype=float)
    return np.round(values, decimals=12)
