import numpy as np
import pytest

from beamforming_sim.algorithms import ConventionalBeamformer, DAMASFISTABeamformer
from beamforming_sim.array_geometry import create_eight_arm_spiral_array
from beamforming_sim.domain import BeamformingResult
from beamforming_sim.scene import AcousticSource, SourceModel, create_scan_planes
from beamforming_sim.signals import simulate_microphone_signals


def _setup_cbf_result(source_position=(0.0, 0.0, 1.2), step_m=0.1):
    """构造 DAMAS-FISTA 所需的 CBF dirty map。"""
    array = create_eight_arm_spiral_array()
    plane = create_scan_planes(distances_m=(1.2,), extent_m=(-0.4, 0.4), step_m=step_m)[0]
    source_model = SourceModel([AcousticSource(position_m=np.array(source_position))])
    _, signals = simulate_microphone_signals(array, source_model, duration_s=0.01, noise_std=0.0)
    cbf_result = ConventionalBeamformer().run(array, plane, signals, sampling_rate_hz=192_000, frequency_hz=25_000)
    return cbf_result, array


class TestDAMASFISTABeamformerInterface:
    """验证 run_from_cbf_map 接口约定。"""

    def test_uses_cbf_result_frequency(self):
        """run_from_cbf_map 默认使用 cbf_result.frequency_hz。"""
        cbf_result, array = _setup_cbf_result()
        # 不传 frequency_hz 参数应正常执行
        result = DAMASFISTABeamformer().run_from_cbf_map(cbf_result, array)
        assert result.frequency_hz == cbf_result.frequency_hz
        assert result.sound_speed_m_s == cbf_result.sound_speed_m_s

    def test_override_sound_speed(self):
        """允许显式覆盖 sound_speed_m_s。"""
        cbf_result, array = _setup_cbf_result()
        result = DAMASFISTABeamformer().run_from_cbf_map(cbf_result, array, sound_speed_m_s=340.0)
        assert result.sound_speed_m_s == 340.0


class TestDAMASFISTABeamformerMetadata:
    """验证 metadata 包含收敛状态等信息。"""

    def test_metadata_contains_converged(self):
        cbf_result, array = _setup_cbf_result()
        result = DAMASFISTABeamformer().run_from_cbf_map(cbf_result, array)
        meta = dict(result.metadata or {})
        assert "converged" in meta
        assert isinstance(meta["converged"], bool)

    def test_metadata_contains_iterations_and_residual(self):
        cbf_result, array = _setup_cbf_result()
        result = DAMASFISTABeamformer(tolerance=1e-4, max_iterations=1000).run_from_cbf_map(cbf_result, array)
        meta = dict(result.metadata or {})
        assert meta["iterations"] >= 1
        assert meta["relative_change"] > 0
        assert meta["residual_norm"] >= 0
        assert meta["point_count"] == len(cbf_result.plane.points_m)
        assert meta["matrix_mode"] in ("dense", "chunked")

    def test_converged_true_when_relative_change_below_tolerance(self):
        cbf_result, array = _setup_cbf_result()
        # 宽松容差确保收敛
        result = DAMASFISTABeamformer(tolerance=1e-2, max_iterations=2000).run_from_cbf_map(cbf_result, array)
        meta = dict(result.metadata or {})
        assert meta["converged"] is True


class TestDAMASFISTABeamformerValidation:
    """验证参数校验和网格保护。"""

    def test_zero_max_iterations_raises(self):
        with pytest.raises(ValueError):
            DAMASFISTABeamformer(max_iterations=0)

    def test_negative_tolerance_raises(self):
        with pytest.raises(ValueError):
            DAMASFISTABeamformer(tolerance=-1e-4)

    def test_max_point_count_exceeded_raises(self):
        cbf_result, array = _setup_cbf_result(step_m=0.1)  # small grid
        damas = DAMASFISTABeamformer(max_point_count=1)  # 1 < actual points
        with pytest.raises(ValueError, match="max_point_count"):
            damas.run_from_cbf_map(cbf_result, array)

    def test_max_point_count_none_allows_large_grid(self):
        """max_point_count=None 时跳过网格大小检查。"""
        cbf_result, array = _setup_cbf_result(step_m=0.1)
        damas = DAMASFISTABeamformer(max_point_count=None)
        result = damas.run_from_cbf_map(cbf_result, array)
        assert result.raw_power is not None


class TestDAMASFISTABeamformerResult:
    """验证 DAMAS-FISTA 输出正确性。"""

    def test_returns_beamforming_result(self):
        cbf_result, array = _setup_cbf_result()
        result = DAMASFISTABeamformer().run_from_cbf_map(cbf_result, array)
        assert isinstance(result, BeamformingResult)
        assert result.algorithm == "DAMAS-FISTA"

    def test_output_shape_matches_scan_points(self):
        cbf_result, array = _setup_cbf_result()
        result = DAMASFISTABeamformer().run_from_cbf_map(cbf_result, array)
        assert result.raw_power.shape == (len(cbf_result.plane.points_m),)

    def test_output_is_nonnegative_and_finite(self):
        cbf_result, array = _setup_cbf_result()
        result = DAMASFISTABeamformer().run_from_cbf_map(cbf_result, array)
        assert np.all(result.raw_power >= 0.0)
        assert np.isfinite(result.raw_power).all()

    def test_peak_near_true_source_center(self):
        source_pos = (0.0, 0.0, 1.2)
        cbf_result, array = _setup_cbf_result(source_position=source_pos)
        result = DAMASFISTABeamformer().run_from_cbf_map(cbf_result, array)
        peak_idx = int(np.argmax(result.raw_power))
        peak_point = cbf_result.plane.points_m[peak_idx]
        assert np.allclose(peak_point, source_pos)

    @pytest.mark.parametrize("source_pos", [
        (0.5, 0.0, 1.2),
        (-0.5, 0.0, 1.2),
    ])
    def test_peak_near_true_source_offset(self, source_pos):
        array = create_eight_arm_spiral_array()
        plane = create_scan_planes(distances_m=(1.2,), extent_m=(-0.6, 0.6), step_m=0.1)[0]
        source_model = SourceModel([AcousticSource(position_m=np.array(source_pos))])
        _, signals = simulate_microphone_signals(array, source_model, duration_s=0.01, noise_std=0.0)
        cbf_result = ConventionalBeamformer().run(array, plane, signals, sampling_rate_hz=192_000, frequency_hz=25_000)
        result = DAMASFISTABeamformer().run_from_cbf_map(cbf_result, array)
        peak_idx = int(np.argmax(result.raw_power))
        peak_point = plane.points_m[peak_idx]
        assert np.allclose(peak_point, source_pos)

    def test_chunked_matrix_mode(self):
        """使用极低 dense_point_limit 触发 chunked 路径。"""
        cbf_result, array = _setup_cbf_result(step_m=0.1)
        damas = DAMASFISTABeamformer(dense_point_limit=1, scan_chunk_size=32)
        result = damas.run_from_cbf_map(cbf_result, array)
        meta = dict(result.metadata or {})
        assert meta["matrix_mode"] == "chunked"
        # chunked 也应得到有效结果
        assert np.all(result.raw_power >= 0.0)
