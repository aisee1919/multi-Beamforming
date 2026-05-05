import numpy as np

from beamforming_sim.array_geometry import create_eight_arm_spiral_array
from beamforming_sim.beamforming import (
    compute_cross_spectral_matrix,
    conventional_beamforming,
    run_cbf_for_planes,
)
from beamforming_sim.scene import AcousticSource, SourceModel, create_default_sources, create_scan_planes
from beamforming_sim.signals import simulate_microphone_signals


def test_cbf_peak_matches_single_source_on_scan_grid():
    array = create_eight_arm_spiral_array()
    plane = create_scan_planes(distances_m=(1.2,), extent_m=(-0.6, 0.6), step_m=0.2)[0]
    source_model = SourceModel([AcousticSource(position_m=np.array([0.0, 0.0, 1.2]))])
    _, signals = simulate_microphone_signals(
        array,
        source_model,
        sampling_rate_hz=192_000,
        duration_s=0.01,
        noise_std=0.0,
    )

    energy = conventional_beamforming(array, plane, signals, sampling_rate_hz=192_000, frequency_hz=25_000)

    peak_point = plane.points_m[int(np.argmax(energy))]
    assert np.allclose(peak_point, [0.0, 0.0, 1.2])
    assert np.isclose(np.max(energy), 1.0)


def test_csm_is_explicit_hermitian_matrix():
    array = create_eight_arm_spiral_array()
    source_model = SourceModel([AcousticSource(position_m=np.array([0.0, 0.0, 1.2]))])
    _, signals = simulate_microphone_signals(array, source_model, duration_s=0.01, noise_std=0.0)

    csm = compute_cross_spectral_matrix(signals, sampling_rate_hz=192_000, frequency_hz=25_000)

    assert csm.shape == (64, 64)
    assert np.allclose(csm, csm.conj().T)
    assert np.all(np.real(np.diag(csm)) >= 0.0)


def test_vectorized_cbf_matches_pointwise_reference_formula():
    array = create_eight_arm_spiral_array()
    plane = create_scan_planes(distances_m=(1.2,), extent_m=(-0.2, 0.2), step_m=0.2)[0]
    source_model = SourceModel([AcousticSource(position_m=np.array([0.0, 0.0, 1.2]))])
    _, signals = simulate_microphone_signals(array, source_model, duration_s=0.01, noise_std=0.0)
    csm = compute_cross_spectral_matrix(signals, sampling_rate_hz=192_000, frequency_hz=25_000)

    actual = conventional_beamforming(array, plane, signals, sampling_rate_hz=192_000, frequency_hz=25_000)
    expected = _pointwise_cbf_reference(array.positions_m, plane.points_m, csm, frequency_hz=25_000)

    assert np.allclose(actual, expected)


def test_cbf_runs_for_all_three_scan_planes():
    array = create_eight_arm_spiral_array()
    planes = create_scan_planes(distances_m=(1.2, 1.6, 2.0), extent_m=(-0.6, 0.6), step_m=0.01)
    source_model = create_default_sources()
    _, signals = simulate_microphone_signals(
        array,
        source_model,
        sampling_rate_hz=192_000,
        duration_s=0.01,
        noise_std=0.0,
        random_seed=42,
    )

    plane_energy = run_cbf_for_planes(array, planes, signals, sampling_rate_hz=192_000, frequency_hz=25_000)

    assert set(plane_energy) == {1.2, 1.6, 2.0}
    for plane in planes:
        energy = plane_energy[plane.distance_m]
        assert energy.shape == (14641,)
        assert np.isfinite(energy).all()
        assert np.max(energy) <= 1.0
        assert np.max(energy) > 0.0


def _pointwise_cbf_reference(
    microphone_positions_m: np.ndarray,
    scan_points_m: np.ndarray,
    csm: np.ndarray,
    frequency_hz: float,
    sound_speed_m_s: float = 343.0,
) -> np.ndarray:
    wave_number = 2.0 * np.pi * frequency_hz / sound_speed_m_s
    energy = np.empty(len(scan_points_m), dtype=float)

    for point_index, point_m in enumerate(scan_points_m):
        distances_m = np.linalg.norm(microphone_positions_m - point_m, axis=1)
        steering = np.exp(-1j * wave_number * distances_m) / distances_m
        weight = steering / np.vdot(steering, steering).real
        energy[point_index] = np.real(np.vdot(weight, csm @ weight))

    return energy / np.max(energy)
