from __future__ import annotations

from beamforming_sim.experiments import BeamformingExperiment, ExperimentConfig


def main() -> None:
    """运行默认的单声源 CBF/FB 成像实验。"""

    summary = BeamformingExperiment(ExperimentConfig()).run()

    print(f"microphones: {summary.microphone_count}")
    print(f"array aperture: {summary.array_aperture_m:.3f} m")
    print(f"scan step: {summary.scan_step_m:.2f} m")
    print(f"single-source cases: {summary.source_case_count}")
    print(f"FB nu values: {summary.fb_nu_values}")
    print(f"array layout: {summary.array_layout_path}")
    print(f"source waveform: {summary.source_waveform_path}")

    for algo_name in ("CBF", "FB", "FFT-FISTA"):
        paths = summary.algorithm_paths.get(algo_name, [])
        print(f"\n{algo_name} heatmaps ({len(paths)}):")
        for path in paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
