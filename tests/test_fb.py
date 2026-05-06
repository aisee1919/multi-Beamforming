import numpy as np

from beamforming_sim.algorithms.fb import csm_power_eig
from beamforming_sim.array_geometry import create_eight_arm_spiral_array
from beamforming_sim.algorithms.cbf import conventional_beamforming
from beamforming_sim.algorithms.fb import functional_beamforming, run_fb_for_planes
from beamforming_sim.spectral import compute_cross_spectral_matrix
from beamforming_sim.scene import AcousticSource, SourceModel, create_default_sources, create_scan_planes
from beamforming_sim.signals import simulate_microphone_signals


def test_fb_nu1_equals_cbf():
    """FB(ν=1) 应严格退化到 CBF。"""
    array = create_eight_arm_spiral_array()
    plane = create_scan_planes(distances_m=(1.2,), extent_m=(-0.4, 0.4), step_m=0.1)[0]
    source_model = SourceModel([AcousticSource(position_m=np.array([0.0, 0.0, 1.2]))])
    _, signals = simulate_microphone_signals(array, source_model, duration_s=0.01, noise_std=0.0)

    cbf_energy = conventional_beamforming(array, plane, signals, sampling_rate_hz=192_000, frequency_hz=25_000)
    fb_energy = functional_beamforming(array, plane, signals, sampling_rate_hz=192_000, frequency_hz=25_000, nu=1)

    assert np.allclose(fb_energy, cbf_energy)


def test_fb_peak_matches_source_location():
    """FB 在 ν > 1 时峰值位置仍应对准声源。"""
    array = create_eight_arm_spiral_array()
    plane = create_scan_planes(distances_m=(1.2,), extent_m=(-0.6, 0.6), step_m=0.2)[0]
    source_model = SourceModel([AcousticSource(position_m=np.array([0.0, 0.0, 1.2]))])
    _, signals = simulate_microphone_signals(array, source_model, duration_s=0.01, noise_std=0.0)

    for nu in (2, 4, 8):
        energy = functional_beamforming(array, plane, signals, sampling_rate_hz=192_000, frequency_hz=25_000, nu=nu)
        peak_point = plane.points_m[int(np.argmax(energy))]
        assert np.allclose(peak_point, [0.0, 0.0, 1.2]), f"nu={nu}: peak at {peak_point}"
        assert np.isclose(np.max(energy), 1.0), f"nu={nu}: max energy {np.max(energy)}"


def test_csm_power_is_hermitian():
    """特征分解求得的 C^(1/ν) 应保持 Hermitian 性质。"""
    array = create_eight_arm_spiral_array()
    source_model = SourceModel([AcousticSource(position_m=np.array([0.0, 0.0, 1.2]))])
    _, signals = simulate_microphone_signals(array, source_model, duration_s=0.01, noise_std=0.0)
    csm = compute_cross_spectral_matrix(signals, sampling_rate_hz=192_000, frequency_hz=25_000)

    for exponent in (1.0, 0.5, 0.25):
        csm_pow = csm_power_eig(csm, exponent)
        assert csm_pow.shape == csm.shape
        assert np.allclose(csm_pow, csm_pow.conj().T), f"exponent={exponent}: not Hermitian"


def test_fb_runs_for_multiple_planes():
    """FB 多平面 API 应对所有扫描平面生效。"""
    array = create_eight_arm_spiral_array()
    planes = create_scan_planes(distances_m=(1.2, 1.6, 2.0), extent_m=(-0.6, 0.6), step_m=0.01)
    source_model = create_default_sources()
    _, signals = simulate_microphone_signals(array, source_model, duration_s=0.01, noise_std=0.0, random_seed=42)

    for nu in (2, 4):
        plane_energy = run_fb_for_planes(array, planes, signals, sampling_rate_hz=192_000, frequency_hz=25_000, nu=nu)
        assert set(plane_energy) == {1.2, 1.6, 2.0}
        for plane in planes:
            energy = plane_energy[plane.distance_m]
            assert energy.shape == (14641,)
            assert np.isfinite(energy).all()
            assert np.max(energy) <= 1.0
            assert np.max(energy) > 0.0


def test_higher_nu_produces_sharper_peak():
    """ν 越大，主瓣能量越集中（归一化后旁瓣级更低）。"""
    array = create_eight_arm_spiral_array()
    plane = create_scan_planes(distances_m=(1.2,), extent_m=(-0.3, 0.3), step_m=0.02)[0]
    source_model = SourceModel([AcousticSource(position_m=np.array([0.0, 0.0, 1.2]))])
    _, signals = simulate_microphone_signals(array, source_model, duration_s=0.01, noise_std=0.0)

    fwhm_values = {}
    for nu in (1, 2, 4, 8):
        energy = functional_beamforming(array, plane, signals, sampling_rate_hz=192_000, frequency_hz=25_000, nu=nu)
        half_max = np.max(energy) / 2.0
        above_half = energy >= half_max
        fwhm = float(np.sum(above_half)) * 0.02  # 点数 × 步长 → 宽度(m)
        fwhm_values[nu] = fwhm

    # ν 每翻倍一次，峰宽应显著缩小
    assert fwhm_values[4] < fwhm_values[2], f"ν=4 ({fwhm_values[4]}) should be narrower than ν=2 ({fwhm_values[2]})"
    assert fwhm_values[8] < fwhm_values[4], f"ν=8 ({fwhm_values[8]}) should be narrower than ν=4 ({fwhm_values[4]})"
