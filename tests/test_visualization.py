import numpy as np

from beamforming_sim.visualization import energy_to_db, tick_values


def test_energy_to_db_uses_reference_figure_dynamic_range():
    db_values = energy_to_db(np.array([1.0, 0.1, 0.01, 0.0]), floor_db=-50.0)

    assert np.allclose(db_values[:3], [0.0, -10.0, -20.0])
    assert db_values[3] < -50.0


def test_tick_values_use_requested_display_grid_step():
    ticks = tick_values((-0.6, 0.6), step_m=0.1)

    assert np.allclose(ticks, np.linspace(-0.6, 0.6, 13))
