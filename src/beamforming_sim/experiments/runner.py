from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from beamforming_sim.algorithms import (
    ConventionalBeamformer,
    FFTFISTABeamformer,
    FunctionalBeamformer,
)
from beamforming_sim.array_geometry import SpiralArrayConfig, create_eight_arm_spiral_array
from beamforming_sim.experiments.cases import SourceCase, build_single_source_cases
from beamforming_sim.experiments.config import ExperimentConfig
from beamforming_sim.experiments.writer import ResultWriter
from beamforming_sim.scene import SourceModel, create_default_sources, create_scan_planes
from beamforming_sim.signals import simulate_microphone_signals
from beamforming_sim.spectral import compute_cross_spectral_matrix


@dataclass(frozen=True)
class ExperimentSummary:
    """实验运行后的输出清单（算法无关的 dict 结构）。"""

    microphone_count: int
    array_aperture_m: float
    scan_step_m: float
    source_case_count: int
    fb_nu_values: tuple[int, ...]
    array_layout_path: Path
    source_waveform_path: Path
    algorithm_paths: dict[str, list[Path]] = field(default_factory=dict)


class BeamformingExperiment:
    """默认单声源 CBF/FB 成像实验流程。

    波束形成器实例可通过 DI 注入，便于测试和参数定制。
    """

    def __init__(
        self,
        config: ExperimentConfig | None = None,
        cbf: ConventionalBeamformer | None = None,
        fft_fista: FFTFISTABeamformer | None = None,
    ) -> None:
        self.config = config or ExperimentConfig()
        self._cbf = cbf or ConventionalBeamformer()
        self._fft_fista = fft_fista or FFTFISTABeamformer()

    def run(self) -> ExperimentSummary:
        config = self.config
        array = create_eight_arm_spiral_array(
            SpiralArrayConfig(
                elements=config.array.elements,
                arms=config.array.arms,
                aperture_m=config.array.aperture_m,
            )
        )
        planes = create_scan_planes(
            distances_m=config.scan.distances_m,
            extent_m=config.scan.extent_m,
            step_m=config.scan.step_m,
        )
        source_model = create_default_sources(
            xy_positions_m=config.source.xy_positions_m,
            distances_m=config.scan.distances_m,
            frequency_hz=config.source.frequency_hz,
        )
        source_cases = build_single_source_cases(source_model, planes)

        writer = ResultWriter(output_dir=config.output_dir, tick_step_m=config.scan.tick_step_m)
        array_layout_path = writer.write_array_layout(array)
        source_waveform_path = writer.write_source_waveform(
            SourceModel([source_model.sources[0]]),
            sampling_rate_hz=config.signal.sampling_rate_hz,
            duration_s=config.signal.source_waveform_duration_s,
            noise_std=config.signal.source_waveform_noise_std,
            random_seed=config.signal.random_seed,
        )

        algorithm_paths: dict[str, list[Path]] = {}

        for case in source_cases:
            _, signals = simulate_microphone_signals(
                array,
                case.source_model,
                sampling_rate_hz=config.signal.sampling_rate_hz,
                duration_s=config.signal.duration_s,
                noise_std=config.signal.microphone_noise_std,
                random_seed=config.signal.random_seed,
            )
            csm = compute_cross_spectral_matrix(signals, config.signal.sampling_rate_hz, config.source.frequency_hz)

            # CBF
            cbf_result = self._cbf.run_from_csm(array, case.plane, csm, config.source.frequency_hz)
            cbf_path = writer.write_heatmap(
                cbf_result,
                source_index=case.index,
                source_x_m=case.x_m,
                source_y_m=case.y_m,
            )
            algorithm_paths.setdefault("CBF", []).append(cbf_path)

            # FB (多 nu 值)
            for nu in config.fb_nu_values:
                fb_result = FunctionalBeamformer(nu=nu).run_from_csm(
                    array, case.plane, csm, config.source.frequency_hz
                )
                fb_path = writer.write_heatmap(
                    fb_result,
                    source_index=case.index,
                    source_x_m=case.x_m,
                    source_y_m=case.y_m,
                )
                algorithm_paths.setdefault("FB", []).append(fb_path)

            # FFT-FISTA
            fft_fista_result = self._fft_fista.run_from_cbf_map(cbf_result, array, config.source.frequency_hz)
            fft_fista_path = writer.write_heatmap(
                fft_fista_result,
                source_index=case.index,
                source_x_m=case.x_m,
                source_y_m=case.y_m,
            )
            algorithm_paths.setdefault("FFT-FISTA", []).append(fft_fista_path)

        return ExperimentSummary(
            microphone_count=len(array.positions_m),
            array_aperture_m=array.aperture_m,
            scan_step_m=config.scan.step_m,
            source_case_count=len(source_cases),
            fb_nu_values=config.fb_nu_values,
            array_layout_path=array_layout_path,
            source_waveform_path=source_waveform_path,
            algorithm_paths=algorithm_paths,
        )
