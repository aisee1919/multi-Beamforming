import numpy as np

from beamforming_sim.algorithms import ConventionalBeamformer, FFTFISTABeamformer, FunctionalBeamformer
from beamforming_sim.array_geometry import create_eight_arm_spiral_array
from beamforming_sim.domain import BeamformingResult
from beamforming_sim.experiments import ExperimentConfig, build_single_source_cases
from beamforming_sim.scene import AcousticSource, SourceModel, create_scan_planes
from beamforming_sim.signals import simulate_microphone_signals


def test_new_algorithm_api_returns_raw_power_result():
    array = create_eight_arm_spiral_array()
    plane = create_scan_planes(distances_m=(1.2,), extent_m=(-0.2, 0.2), step_m=0.2)[0]
    source_model = SourceModel([AcousticSource(position_m=np.array([0.0, 0.0, 1.2]))])
    _, signals = simulate_microphone_signals(array, source_model, duration_s=0.01, noise_std=0.0)

    result = ConventionalBeamformer().run(
        array,
        plane,
        signals,
        sampling_rate_hz=192_000,
        frequency_hz=25_000,
    )

    assert isinstance(result, BeamformingResult)
    assert result.algorithm == "CBF"
    assert result.raw_power.shape == (len(plane.points_m),)
    assert np.max(result.raw_power) > 1.0
    assert np.isclose(np.max(result.normalized_power()), 1.0)


def test_result_normalization_is_explicit_on_result_object():
    array = create_eight_arm_spiral_array()
    plane = create_scan_planes(distances_m=(1.2,), extent_m=(-0.2, 0.2), step_m=0.2)[0]
    source_model = SourceModel([AcousticSource(position_m=np.array([0.0, 0.0, 1.2]))])
    _, signals = simulate_microphone_signals(array, source_model, duration_s=0.01, noise_std=0.0)

    result = ConventionalBeamformer().run(array, plane, signals, sampling_rate_hz=192_000, frequency_hz=25_000)
    energy = result.normalized_power()

    assert isinstance(energy, np.ndarray)
    assert np.isclose(np.max(energy), 1.0)


def test_functional_beamformer_nu1_matches_cbf_raw_power():
    array = create_eight_arm_spiral_array()
    plane = create_scan_planes(distances_m=(1.2,), extent_m=(-0.2, 0.2), step_m=0.2)[0]
    source_model = SourceModel([AcousticSource(position_m=np.array([0.0, 0.0, 1.2]))])
    _, signals = simulate_microphone_signals(array, source_model, duration_s=0.01, noise_std=0.0)

    cbf = ConventionalBeamformer().run(array, plane, signals, sampling_rate_hz=192_000, frequency_hz=25_000)
    fb = FunctionalBeamformer(nu=1).run(array, plane, signals, sampling_rate_hz=192_000, frequency_hz=25_000)

    assert np.allclose(fb.raw_power, cbf.raw_power)


def test_experiment_config_keeps_default_heatmap_signals_noise_free():
    config = ExperimentConfig()

    assert config.signal.microphone_noise_std == 0.0
    assert config.signal.source_waveform_noise_std == 0.0
    assert config.scan.step_m == 0.01
    assert config.scan.tick_step_m == 0.1


def test_single_source_cases_bind_each_candidate_source_to_its_plane():
    planes = create_scan_planes(distances_m=(1.2, 1.6), extent_m=(-0.2, 0.2), step_m=0.2)
    source_model = SourceModel(
        [
            AcousticSource(position_m=np.array([0.0, 0.0, 1.2])),
            AcousticSource(position_m=np.array([0.1, 0.0, 1.6])),
        ]
    )

    cases = build_single_source_cases(source_model, planes)

    assert [case.index for case in cases] == [1, 2]
    assert [case.plane.distance_m for case in cases] == [1.2, 1.6]
    assert all(len(case.source_model.sources) == 1 for case in cases)


def test_fft_fista_returns_beamforming_result_not_ndarray():
    """FFT-FISTA 同样应通过 BeamformingResult 返回原始功率。"""
    array = create_eight_arm_spiral_array()
    plane = create_scan_planes(distances_m=(1.2,), extent_m=(-0.2, 0.2), step_m=0.2)[0]
    source_model = SourceModel([AcousticSource(position_m=np.array([0.0, 0.0, 1.2]))])
    _, signals = simulate_microphone_signals(array, source_model, duration_s=0.01, noise_std=0.0)

    cbf_result = ConventionalBeamformer().run(array, plane, signals, sampling_rate_hz=192_000, frequency_hz=25_000)
    fft_fista_result = FFTFISTABeamformer().run_from_cbf_map(cbf_result, array)

    assert isinstance(fft_fista_result, BeamformingResult)
    assert fft_fista_result.algorithm == "FFT-FISTA"
    assert fft_fista_result.raw_power.shape == cbf_result.raw_power.shape
    assert np.max(fft_fista_result.raw_power) > 0.0
