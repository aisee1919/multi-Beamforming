from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from beamforming_sim.algorithms import ConventionalBeamformer, FFTFISTABeamformer, FunctionalBeamformer
from beamforming_sim.array_geometry import SpiralArrayConfig, create_eight_arm_spiral_array
from beamforming_sim.experiments.cases import SourceCase, build_single_source_cases
from beamforming_sim.experiments.config import ExperimentConfig
from beamforming_sim.experiments.writer import ResultWriter
from beamforming_sim.scene import SourceModel, create_default_sources, create_scan_planes
from beamforming_sim.signals import simulate_microphone_signals
from beamforming_sim.spectral import compute_cross_spectral_matrix


@dataclass(frozen=True)
class ExperimentSummary:
    """实验运行后的输出清单。"""

    microphone_count: int
    array_aperture_m: float
    scan_step_m: float
    source_case_count: int
    fb_nu_values: tuple[int, ...]
    array_layout_path: Path
    source_waveform_path: Path
    cbf_paths: list[Path] = field(default_factory=list)
    fb_paths: list[Path] = field(default_factory=list)
    fft_fista_paths: list[Path] = field(default_factory=list)


class BeamformingExperiment:
    """默认单声源 CBF/FB 成像实验流程。"""

    def __init__(self, config: ExperimentConfig | None = None) -> None:
        self.config = config or ExperimentConfig()

    def run(self) -> ExperimentSummary:
        config = self.config
        array = create_eight_arm_spiral_array(
            SpiralArrayConfig(
                elements=config.array_elements,
                arms=config.array_arms,
                aperture_m=config.array_aperture_m,
            )
        )
        planes = create_scan_planes(
            distances_m=config.scan_distances_m,
            extent_m=config.scan_extent_m,
            step_m=config.scan_step_m,
        )
        source_model = create_default_sources(
            xy_positions_m=config.source_xy_positions_m,
            distances_m=config.scan_distances_m,
            frequency_hz=config.frequency_hz,
        )
        source_cases = build_single_source_cases(source_model, planes)

        writer = ResultWriter(output_dir=config.output_dir, tick_step_m=config.tick_step_m)
        array_layout_path = writer.write_array_layout(array)
        source_waveform_path = writer.write_source_waveform(
            SourceModel([source_model.sources[0]]),
            sampling_rate_hz=config.sampling_rate_hz,
            duration_s=config.source_waveform_duration_s,
            noise_std=config.source_waveform_noise_std,
            random_seed=config.random_seed,
        )

        cbf_paths: list[Path] = []
        fb_paths: list[Path] = []
        fft_fista_paths: list[Path] = []
        cbf = ConventionalBeamformer()
        fft_fista = FFTFISTABeamformer()

        for case in source_cases:
            _, signals = simulate_microphone_signals(
                array,
                case.source_model,
                sampling_rate_hz=config.sampling_rate_hz,
                duration_s=config.duration_s,
                noise_std=config.microphone_noise_std,
                random_seed=config.random_seed,
            )
            csm = compute_cross_spectral_matrix(signals, config.sampling_rate_hz, config.frequency_hz)

            cbf_result = cbf.run_from_csm(array, case.plane, csm, config.frequency_hz)
            cbf_paths.append(writer.write_heatmap(case, cbf_result))

            fb_paths.extend(self._run_fb_cases(array, case, csm, writer))

            fft_fista_result = fft_fista.run_from_cbf_map(cbf_result, array, config.frequency_hz)
            fft_fista_paths.append(writer.write_heatmap(case, fft_fista_result))

        return ExperimentSummary(
            microphone_count=len(array.positions_m),
            array_aperture_m=array.aperture_m,
            scan_step_m=config.scan_step_m,
            source_case_count=len(source_cases),
            fb_nu_values=config.fb_nu_values,
            array_layout_path=array_layout_path,
            source_waveform_path=source_waveform_path,
            cbf_paths=cbf_paths,
            fb_paths=fb_paths,
            fft_fista_paths=fft_fista_paths,
        )

    def _run_fb_cases(self, array, case: SourceCase, csm, writer: ResultWriter) -> list[Path]:
        paths: list[Path] = []
        for nu in self.config.fb_nu_values:
            fb_result = FunctionalBeamformer(nu=nu).run_from_csm(array, case.plane, csm, self.config.frequency_hz)
            paths.append(writer.write_heatmap(case, fb_result))
        return paths
