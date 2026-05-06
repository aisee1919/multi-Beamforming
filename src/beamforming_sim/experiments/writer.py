from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from beamforming_sim.array_geometry import MicrophoneArray
from beamforming_sim.domain import BeamformingResult
from beamforming_sim.scene import SourceModel
from beamforming_sim.visualization import plot_array_layout, plot_energy_heatmap, plot_source_signals


@dataclass(frozen=True)
class ResultWriter:
    """集中管理实验输出路径和绘图写入。"""

    output_dir: Path
    tick_step_m: float

    def write_array_layout(self, array: MicrophoneArray) -> Path:
        return plot_array_layout(array, self.output_dir / "array_layout.png")

    def write_source_waveform(
        self,
        source_model: SourceModel,
        sampling_rate_hz: float,
        duration_s: float,
        noise_std: float,
        random_seed: int,
    ) -> Path:
        return plot_source_signals(
            source_model,
            sampling_rate_hz=sampling_rate_hz,
            duration_s=duration_s,
            noise_std=noise_std,
            random_seed=random_seed,
            output_path=self.output_dir / "source_signals.png",
        )

    def write_heatmap(
        self,
        result: BeamformingResult,
        source_index: int = 1,
        source_x_m: float = 0.0,
        source_y_m: float = 0.0,
        interpolation: str = "bilinear",
        mark_peak: bool = False,
    ) -> Path:
        """波束形成热力图写入，不依赖 SourceCase 类型。"""
        algo_key = result.algorithm.lower().replace("-", "_")
        subdir = self.output_dir / f"{algo_key}_single_source"

        plane = result.plane
        z = plane.distance_m

        nx = len(plane.x_coordinates_m)
        step_m = float(plane.x_coordinates_m[1] - plane.x_coordinates_m[0]) if nx > 1 else 0.0

        stem = f"{algo_key}_source_{source_index:02d}_z_{z:.1f}m_step_{step_m:.2f}m"
        title = f"{result.algorithm} (step={step_m:.2f}m, z={z:g}m"

        nu = (result.metadata or {}).get("nu")
        if nu is not None:
            stem += f"_nu{int(nu):02d}"
            title += f", nu={nu}"
        else:
            stem += f"_x_{source_x_m:+.1f}_y_{source_y_m:+.1f}"
            title += f", x={source_x_m:g}m, y={source_y_m:g}m"
        title += ")"

        peak_position = None
        if mark_peak:
            peak_idx = int(np.argmax(result.raw_power))
            peak_pos = plane.points_m[peak_idx]
            peak_position = (float(peak_pos[0]), float(peak_pos[1]))

        return plot_energy_heatmap(
            plane, result.raw_power, subdir / f"{stem}.png",
            title=title, tick_step_m=self.tick_step_m,
            interpolation=interpolation, peak_position=peak_position,
        )
