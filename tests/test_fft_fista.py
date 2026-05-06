import numpy as np

from beamforming_sim.algorithms import ConventionalBeamformer, FFTFISTABeamformer
from beamforming_sim.array_geometry import create_eight_arm_spiral_array
from beamforming_sim.domain import BeamformingResult
from beamforming_sim.scene import AcousticSource, SourceModel, create_scan_planes
from beamforming_sim.signals import simulate_microphone_signals


def _run_algorithms(array, plane, source_position, frequency_hz=25_000, sampling_rate_hz=192_000):
    """辅助函数：运行 CBF 和 FFT-FISTA。"""
    source_model = SourceModel([AcousticSource(position_m=np.array(source_position))])
    _, signals = simulate_microphone_signals(array, source_model, duration_s=0.01, noise_std=0.0)
    cbf_result = ConventionalBeamformer().run(array, plane, signals, sampling_rate_hz, frequency_hz)
    fft_fista_result = FFTFISTABeamformer().run_from_cbf_map(cbf_result, array)
    return cbf_result, fft_fista_result


class TestFFTFISTABeamformerResult:
    """验证 FFTFISTABeamformer 返回正确的 BeamformingResult。"""

    def test_returns_beamforming_result(self):
        array = create_eight_arm_spiral_array()
        plane = create_scan_planes(distances_m=(1.2,), extent_m=(-0.2, 0.2), step_m=0.2)[0]
        _, result = _run_algorithms(array, plane, [0.0, 0.0, 1.2])

        assert isinstance(result, BeamformingResult)
        assert result.algorithm == "FFT-FISTA"

    def test_output_shape_matches_scan_points(self):
        array = create_eight_arm_spiral_array()
        plane = create_scan_planes(distances_m=(1.2,), extent_m=(-0.4, 0.4), step_m=0.1)[0]
        _, result = _run_algorithms(array, plane, [0.0, 0.0, 1.2])

        assert result.raw_power.shape == (len(plane.points_m),)

    def test_output_is_nonnegative_and_finite(self):
        array = create_eight_arm_spiral_array()
        plane = create_scan_planes(distances_m=(1.2,), extent_m=(-0.3, 0.3), step_m=0.1)[0]
        _, result = _run_algorithms(array, plane, [0.0, 0.0, 1.2])

        assert np.all(result.raw_power >= 0.0)
        assert np.isfinite(result.raw_power).all()

    def test_peak_near_true_source_center(self):
        array = create_eight_arm_spiral_array()
        plane = create_scan_planes(distances_m=(1.2,), extent_m=(-0.4, 0.4), step_m=0.05)[0]
        source_pos = [0.0, 0.0, 1.2]
        _, result = _run_algorithms(array, plane, source_pos)

        peak_idx = int(np.argmax(result.raw_power))
        peak_point = plane.points_m[peak_idx]
        assert np.allclose(peak_point, source_pos)

    def test_peak_near_true_source_offset(self):
        array = create_eight_arm_spiral_array()
        plane = create_scan_planes(distances_m=(1.2,), extent_m=(-0.6, 0.6), step_m=0.05)[0]
        source_pos = [0.5, 0.0, 1.2]
        _, result = _run_algorithms(array, plane, source_pos)

        peak_idx = int(np.argmax(result.raw_power))
        peak_point = plane.points_m[peak_idx]
        assert np.allclose(peak_point, source_pos)
