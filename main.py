from __future__ import annotations

from pathlib import Path

from beamforming_sim.array_geometry import SpiralArrayConfig, create_eight_arm_spiral_array
from beamforming_sim.beamforming import conventional_beamforming, functional_beamforming
from beamforming_sim.scene import AcousticSource, SourceModel, create_default_sources, create_scan_planes
from beamforming_sim.signals import simulate_microphone_signals
from beamforming_sim.visualization import plot_array_layout, plot_energy_heatmap


def main() -> None:
    array = create_eight_arm_spiral_array(SpiralArrayConfig(elements=64, arms=8, aperture_m=0.15))
    planes = create_scan_planes(distances_m=(1.2, 1.6, 2.0), extent_m=(-0.6, 0.6), step_m=0.01)
    plane_by_distance = {plane.distance_m: plane for plane in planes}
    source_model = create_default_sources()

    plot_array_layout(array, "outputs/array_layout.png")

    # —— CBF 单声源（基线） ——
    cbf_dir = Path("outputs/cbf_single_source")
    cbf_paths = []
    for source_index, source in enumerate(source_model.sources, start=1):
        single_source_model = SourceModel([source])
        _, signals = simulate_microphone_signals(
            array, single_source_model,
            sampling_rate_hz=192_000, duration_s=0.01, noise_std=0.0, random_seed=42,
        )
        plane = plane_by_distance[float(source.position_m[2])]
        energy = conventional_beamforming(array, plane, signals, sampling_rate_hz=192_000, frequency_hz=25_000)
        path = cbf_dir / f"cbf_source_{source_index:02d}_z_{source.position_m[2]:.1f}m_x_{source.position_m[0]:+.1f}_y_{source.position_m[1]:+.1f}.png"
        plot_energy_heatmap(
            plane, energy, path,
            title=f"CBF(z={source.position_m[2]:g}m, x={source.position_m[0]:g}m, y={source.position_m[1]:g}m)",
            tick_step_m=0.1,
        )
        cbf_paths.append(path)

    # —— FB 单声源（对比不同 ν） ——
    fb_dir = Path("outputs/fb_single_source")
    fb_nu_values = (2, 4, 8, 16)
    fb_paths = []
    for source_index, source in enumerate(source_model.sources, start=1):
        single_source_model = SourceModel([source])
        _, signals = simulate_microphone_signals(
            array, single_source_model,
            sampling_rate_hz=192_000, duration_s=0.01, noise_std=0.0, random_seed=42,
        )
        plane = plane_by_distance[float(source.position_m[2])]
        for nu in fb_nu_values:
            energy = functional_beamforming(
                array, plane, signals,
                sampling_rate_hz=192_000, frequency_hz=25_000, nu=nu,
            )
            path = fb_dir / f"fb_nu{nu:02d}_source_{source_index:02d}_z_{source.position_m[2]:.1f}m.png"
            plot_energy_heatmap(
                plane, energy, path,
                title=f"FB(ν={nu}, z={source.position_m[2]:g}m)",
                tick_step_m=0.1,
            )
            fb_paths.append(path)

    # —— 集中报告 ——
    print(f"microphones: {len(array.positions_m)}")
    print(f"array aperture: {array.aperture_m:.3f} m")
    print(f"scan step: 0.01 m")
    print(f"single-source cases: {len(source_model.sources)}")
    print(f"FB ν values: {fb_nu_values}")
    print(f"\nCBF heatmaps ({len(cbf_paths)}):")
    for p in cbf_paths:
        print(f"  {p}")
    print(f"\nFB heatmaps ({len(fb_paths)}):")
    for p in fb_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
