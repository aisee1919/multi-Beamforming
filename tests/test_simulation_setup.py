import numpy as np

from beamforming_sim.array_geometry import SpiralArrayConfig, create_eight_arm_spiral_array
from beamforming_sim.scene import SourceModel, create_default_sources, create_scan_planes
from beamforming_sim.signals import simulate_microphone_signals


def test_eight_arm_spiral_array_has_expected_geometry():
    config = SpiralArrayConfig(elements=64, arms=8, aperture_m=0.15)

    array = create_eight_arm_spiral_array(config)

    assert array.positions_m.shape == (64, 3)
    assert np.allclose(array.positions_m[:, 2], 0.0)
    assert np.isclose(array.aperture_m, 0.15)
    assert np.isclose(np.max(np.linalg.norm(array.positions_m[:, :2], axis=1)), 0.075)


def test_scan_planes_cover_requested_grid_and_distances():
    planes = create_scan_planes(
        distances_m=(1.2, 1.6, 2.0),
        extent_m=(-0.6, 0.6),
        step_m=0.01,
    )

    assert [plane.distance_m for plane in planes] == [1.2, 1.6, 2.0]
    for plane in planes:
        assert plane.points_m.shape == (14641, 3)
        assert np.allclose(np.unique(plane.points_m[:, 0]), np.linspace(-0.6, 0.6, 121))
        assert np.allclose(np.unique(plane.points_m[:, 1]), np.linspace(-0.6, 0.6, 121))
        assert np.allclose(plane.points_m[:, 2], plane.distance_m)


def test_source_model_starts_empty_and_accepts_future_sources():
    model = SourceModel()

    assert model.sources == []


def test_default_sources_use_requested_positions_and_frequency():
    model = create_default_sources()

    assert len(model.sources) == 9
    assert [source.frequency_hz for source in model.sources] == [25_000.0] * 9
    assert np.allclose(
        [source.position_m for source in model.sources],
        [
            [0.0, 0.0, 1.2],
            [0.5, 0.0, 1.2],
            [-0.5, 0.0, 1.2],
            [0.0, 0.0, 1.6],
            [0.5, 0.0, 1.6],
            [-0.5, 0.0, 1.6],
            [0.0, 0.0, 2.0],
            [0.5, 0.0, 2.0],
            [-0.5, 0.0, 2.0],
        ],
    )


def test_microphone_signal_simulation_returns_64_channels():
    array = create_eight_arm_spiral_array()
    sources = create_default_sources()

    time_s, signals = simulate_microphone_signals(
        array,
        sources,
        sampling_rate_hz=192_000,
        duration_s=0.001,
        noise_std=0.0,
    )

    assert time_s.shape == (192,)
    assert signals.shape == (64, 192)
    assert np.isfinite(signals).all()
    assert not np.allclose(signals, 0.0)


def test_awgn_is_seeded_and_added_to_microphone_signals():
    array = create_eight_arm_spiral_array()
    sources = create_default_sources()

    _, clean = simulate_microphone_signals(array, sources, duration_s=0.001, noise_std=0.0)
    _, noisy_a = simulate_microphone_signals(array, sources, duration_s=0.001, noise_std=0.05, random_seed=7)
    _, noisy_b = simulate_microphone_signals(array, sources, duration_s=0.001, noise_std=0.05, random_seed=7)

    assert np.allclose(noisy_a, noisy_b)
    assert not np.allclose(clean, noisy_a)
