"""DAMAS-FISTA 单声源实验脚本。

对 3 个扫描平面 x 3 个声源位置 = 9 个单声源 case 运行 DAMAS-FISTA，
输出热力图和 metrics.csv。

用法:
    python scripts/run_damas_fista_single_source.py
    python scripts/run_damas_fista_single_source.py --step 0.05
"""

from __future__ import annotations

import argparse
from pathlib import Path

from beamforming_sim.experiments import ExperimentConfig, run_damas_fista_single_source_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="DAMAS-FISTA 单声源实验")
    parser.add_argument(
        "--step",
        type=float,
        default=0.1,
        help="扫描网格步长。默认 0.1m 快速验证；建议 0.05m 正式小规模对比。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="输出根目录。",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="最大迭代次数。",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="收敛容差。",
    )
    args = parser.parse_args()

    run_damas_fista_single_source_experiment(
        config=ExperimentConfig(output_dir=args.output_dir),
        scan_step_m=args.step,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        progress=True,
    )


if __name__ == "__main__":
    main()
