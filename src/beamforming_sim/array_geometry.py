from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpiralArrayConfig:
    """八臂螺旋麦克风阵列的几何参数。"""

    # 总阵元数。当前实验按 8 臂 × 每臂 8 个阵元配置。
    elements: int = 64
    # 螺旋臂数量固定为 8，便于和参考实验保持一致。
    arms: int = 8
    # 阵列孔径按最大阵元间距理解，因此最大半径为 aperture_m / 2。
    aperture_m: float = 0.15
    # 每条臂从内到外额外旋转的圈数，用于形成螺旋排布。
    spiral_turns: float = 0.75


@dataclass(frozen=True)
class MicrophoneArray:
    """麦克风阵列坐标，单位为米，阵列平面位于 z=0。"""

    positions_m: np.ndarray
    config: SpiralArrayConfig

    @property
    def aperture_m(self) -> float:
        """由阵元坐标反算孔径，用于验证实际几何是否满足配置。"""

        radii = np.linalg.norm(self.positions_m[:, :2], axis=1)
        return float(2.0 * np.max(radii))


def create_eight_arm_spiral_array(config: SpiralArrayConfig | None = None) -> MicrophoneArray:
    """生成八臂螺旋阵列坐标。

    坐标系约定：阵列中心为原点，阵列位于 x-y 平面，声源扫描平面位于正 z 方向。
    """

    config = config or SpiralArrayConfig()
    _validate_config(config)

    elements_per_arm = config.elements // config.arms
    max_radius = config.aperture_m / 2.0
    # 从中心向外等间隔布置每臂阵元，最外侧阵元落在孔径半径处。
    radial_positions = np.linspace(max_radius / elements_per_arm, max_radius, elements_per_arm)

    positions: list[list[float]] = []
    for arm_index in range(config.arms):
        arm_angle = 2.0 * np.pi * arm_index / config.arms
        for element_index, radius in enumerate(radial_positions):
            progress = element_index / max(elements_per_arm - 1, 1)
            # 基础臂角叠加径向进度产生的旋转角，形成多臂螺旋。
            angle = arm_angle + 2.0 * np.pi * config.spiral_turns * progress
            positions.append([radius * np.cos(angle), radius * np.sin(angle), 0.0])

    return MicrophoneArray(positions_m=np.asarray(positions, dtype=float), config=config)


def _validate_config(config: SpiralArrayConfig) -> None:
    """校验阵列参数，提前暴露不满足实验设定的输入。"""

    if config.elements <= 0:
        raise ValueError("elements must be positive")
    if config.arms <= 0:
        raise ValueError("arms must be positive")
    if config.elements % config.arms != 0:
        raise ValueError("elements must be divisible by arms")
    if config.arms != 8:
        raise ValueError("this setup is fixed to an eight-arm spiral array")
    if config.aperture_m <= 0:
        raise ValueError("aperture_m must be positive")
    if config.spiral_turns < 0:
        raise ValueError("spiral_turns must be non-negative")
