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

    def write_cbf_heatmap(self, case: SourceCase, result: BeamformingResult) -> Path:
        path = (
            self.output_dir
            / "cbf_single_source"
            / f"cbf_source_{case.index:02d}_z_{case.z_m:.1f}m_x_{case.x_m:+.1f}_y_{case.y_m:+.1f}.png"
        )
        return plot_energy_heatmap(
            case.plane,
            result.raw_power,
            path,
            title=f"CBF(z={case.z_m:g}m, x={case.x_m:g}m, y={case.y_m:g}m)",
            tick_step_m=self.tick_step_m,
        )

    def write_fb_heatmap(self, case: SourceCase, result: BeamformingResult, nu: int) -> Path:
        path = self.output_dir / "fb_single_source" / f"fb_nu{nu:02d}_source_{case.index:02d}_z_{case.z_m:.1f}m.png"
        return plot_energy_heatmap(
            case.plane,
            result.raw_power,
            path,
            title=f"FB(nu={nu}, z={case.z_m:g}m)",
            tick_step_m=self.tick_step_m,
        )

    def write_fft_fista_heatmap(self, case: SourceCase, result: BeamformingResult) -> Path:
        path = (
            self.output_dir
            / "fft_fista_single_source"
            / f"fft_fista_source_{case.index:02d}_z_{case.z_m:.1f}m_x_{case.x_m:+.1f}_y_{case.y_m:+.1f}.png"
        )
        return plot_energy_heatmap(
            case.plane,
            result.raw_power,
            path,
            title=f"FFT-FISTA(z={case.z_m:g}m, x={case.x_m:g}m, y={case.y_m:g}m)",
            tick_step_m=self.tick_step_m,
        )
