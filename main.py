from __future__ import annotations

import argparse
from pathlib import Path

from beamforming_sim.experiments import (
    BeamformingExperiment,
    ExperimentConfig,
    run_damas_fista_single_source_experiment,
)


def main() -> None:
    """运行单声源波束形成成像实验。"""

    parser = argparse.ArgumentParser(description="多算法声学波束形成单声源实验")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="输出目录")
    parser.add_argument(
        "--include-damas-fista",
        action="store_true",
        help="额外运行 DAMAS-FISTA 单声源实验。默认不运行，避免 0.01m 大网格高成本反卷积。",
    )
    parser.add_argument(
        "--damas-step",
        type=float,
        default=0.1,
        help="DAMAS-FISTA 扫描步长，默认 0.1m；建议正式小规模对比用 0.05m。",
    )
    parser.add_argument("--damas-max-iterations", type=int, default=1000, help="DAMAS-FISTA 最大迭代次数")
    parser.add_argument("--damas-tolerance", type=float, default=1e-4, help="DAMAS-FISTA 收敛容差")
    args = parser.parse_args()

    config = ExperimentConfig(output_dir=args.output_dir)
    summary = BeamformingExperiment(config).run()

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

    if args.include_damas_fista:
        damas_summary = run_damas_fista_single_source_experiment(
            config=config,
            scan_step_m=args.damas_step,
            max_iterations=args.damas_max_iterations,
            tolerance=args.damas_tolerance,
            progress=True,
        )
        print(f"\nDAMAS-FISTA heatmaps ({len(damas_summary.heatmap_paths)}):")
        for path in damas_summary.heatmap_paths:
            print(f"  {path}")
        print(f"DAMAS-FISTA metrics: {damas_summary.metrics_csv_path}")
    else:
        print(
            "\nDAMAS-FISTA is available via --include-damas-fista "
            "(default step 0.1m; use --damas-step 0.05 for a denser small run)."
        )


if __name__ == "__main__":
    main()
