from __future__ import annotations

from pathlib import Path

from beamforming_sim.array_geometry import SpiralArrayConfig, create_eight_arm_spiral_array
from beamforming_sim.beamforming import conventional_beamforming
from beamforming_sim.scene import SourceModel, create_default_sources, create_scan_planes
from beamforming_sim.signals import simulate_microphone_signals
from beamforming_sim.visualization import plot_array_layout, plot_energy_heatmap


def main() -> None:
    # 阵列平面位于 z=0，孔径 0.15 m，生成 8 臂 × 8 阵元的螺旋阵列。
    array = create_eight_arm_spiral_array(SpiralArrayConfig(elements=64, arms=8, aperture_m=0.15))

    # CBF 实际扫描步长为 0.01 m；图上坐标刻度步长在绘图函数中设为 0.1 m。
    planes = create_scan_planes(distances_m=(1.2, 1.6, 2.0), extent_m=(-0.6, 0.6), step_m=0.01)
    plane_by_distance = {plane.distance_m: plane for plane in planes}

    # 这里仍然生成 9 个候选声源，但后续逐个取出，保证每次只有一个声源发声。
    source_model = create_default_sources()

    plot_array_layout(array, "outputs/array_layout.png")
    output_dir = Path("outputs/cbf_single_source")

    generated_heatmaps = []
    for source_index, source in enumerate(source_model.sources, start=1):
        single_source_model = SourceModel([source])
        time_s, received_signals = simulate_microphone_signals(
            array,
            single_source_model,
            sampling_rate_hz=192_000,
            duration_s=0.01,
            noise_std=0.0,
            random_seed=42,
        )

        source_x, source_y, source_z = source.position_m
        plane = plane_by_distance[float(source_z)]
        energy = conventional_beamforming(
            array,
            plane,
            received_signals,
            sampling_rate_hz=192_000,
            frequency_hz=25_000,
        )
        output_path = output_dir / f"cbf_source_{source_index:02d}_z_{source_z:.1f}m_x_{source_x:+.1f}_y_{source_y:+.1f}.png"
        plot_energy_heatmap(
            plane,
            energy,
            output_path,
            title=f"CBF({source_z:g}m, x={source_x:g}m, y={source_y:g}m)",
            tick_step_m=0.1,
        )
        generated_heatmaps.append(output_path)

    print(f"microphones: {len(array.positions_m)}")
    print(f"array aperture: {array.aperture_m:.3f} m")
    print(f"scan step: 0.01 m")
    print("plot tick step: 0.1 m")
    print(f"single-source cases: {len(source_model.sources)}")
    print("noise std: 0.0")
    print(f"samples per case: {len(time_s)}")
    print(f"cbf heatmaps directory: {output_dir}")
    for path in generated_heatmaps:
        print(path)


if __name__ == "__main__":
    main()
