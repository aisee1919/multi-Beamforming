from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from beamforming_sim.array_geometry import MicrophoneArray
from beamforming_sim.domain import BeamformingResult
from beamforming_sim.experiments.cases import SourceCase
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

    def write_heatmap(self, case: SourceCase, result: BeamformingResult) -> Path:
        """统一的波束形成热力图写入。

        路径和标题由 result.algorithm 与 result.metadata 自动生成，新算法无需额外方法。
        """
        algo_key = result.algorithm.lower().replace("-", "_")
        subdir = self.output_dir / f"{algo_key}_single_source"

        stem = f"{algo_key}_source_{case.index:02d}_z_{case.z_m:.1f}m"
        title = f"{result.algorithm}(z={case.z_m:g}m"

        nu = (result.metadata or {}).get("nu")
        if nu is not None:
            stem += f"_nu{int(nu):02d}"
            title += f", nu={nu}"
        else:
            stem += f"_x_{case.x_m:+.1f}_y_{case.y_m:+.1f}"
            title += f", x={case.x_m:g}m, y={case.y_m:g}m"
        title += ")"

        return plot_energy_heatmap(
            case.plane, result.raw_power, subdir / f"{stem}.png",
            title=title, tick_step_m=self.tick_step_m,
        )
