from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from beamforming_sim.algorithms import ConventionalBeamformer, DAMASFISTABeamformer
from beamforming_sim.array_geometry import SpiralArrayConfig, create_eight_arm_spiral_array
from beamforming_sim.domain import BeamformingResult
from beamforming_sim.experiments.cases import build_single_source_cases
from beamforming_sim.experiments.config import ExperimentConfig
from beamforming_sim.experiments.writer import ResultWriter
from beamforming_sim.scene import create_default_sources, create_scan_planes
from beamforming_sim.signals import simulate_microphone_signals
from beamforming_sim.spectral import compute_cross_spectral_matrix


@dataclass(frozen=True)
class DAMASFISTAExperimentSummary:
    """DAMAS-FISTA 单声源实验输出清单。"""

    case_count: int
    scan_step_m: float
    heatmap_paths: list[Path] = field(default_factory=list)
    metrics_csv_path: Path | None = None


def run_damas_fista_single_source_experiment(
    config: ExperimentConfig | None = None,
    scan_step_m: float = 0.1,
    max_iterations: int = 1000,
    tolerance: float = 1e-4,
    progress: bool = False,
) -> DAMASFISTAExperimentSummary:
    """运行 9 个单声源 DAMAS-FISTA case，并输出热力图和 metrics.csv。

    DAMAS-FISTA 构造完整 PSF 算子，默认使用较粗网格，避免主入口在
    0.01 m 默认网格上触发高成本反卷积。
    """

    if scan_step_m <= 0:
        raise ValueError("scan_step_m must be positive")

    config = config or ExperimentConfig()
    tick_step = 0.1 if scan_step_m >= 0.1 else scan_step_m * 2

    array = create_eight_arm_spiral_array(
        SpiralArrayConfig(
            elements=config.array.elements,
            arms=config.array.arms,
            aperture_m=config.array.aperture_m,
        )
    )
    planes = create_scan_planes(
        distances_m=config.scan.distances_m,
        extent_m=config.scan.extent_m,
        step_m=scan_step_m,
    )
    source_model = create_default_sources(
        xy_positions_m=config.source.xy_positions_m,
        distances_m=config.scan.distances_m,
        frequency_hz=config.source.frequency_hz,
    )
    cases = build_single_source_cases(source_model, planes)

    damas = DAMASFISTABeamformer(max_iterations=max_iterations, tolerance=tolerance)
    cbf = ConventionalBeamformer()
    writer = ResultWriter(output_dir=config.output_dir, tick_step_m=tick_step)

    metrics_rows: list[dict] = []
    heatmap_paths: list[Path] = []

    if progress:
        print("DAMAS-FISTA single-source experiment")
        print(f"  grid step: {scan_step_m} m")
        print(f"  cases: {len(cases)}")
        print(f"  max_iterations: {damas.max_iterations}")
        print(f"  tolerance: {damas.tolerance}")
        print()

    for case in cases:
        source_pos = case.source.position_m
        if progress:
            label = (
                f"source {case.index:02d}  "
                f"z={case.z_m:.1f}m  "
                f"x={source_pos[0]:+.1f}  y={source_pos[1]:+.1f}"
            )
            print(f"[{label}]  running ...", end=" ", flush=True)

        _, signals = simulate_microphone_signals(
            array,
            case.source_model,
            sampling_rate_hz=config.signal.sampling_rate_hz,
            duration_s=config.signal.duration_s,
            noise_std=config.signal.microphone_noise_std,
            random_seed=config.signal.random_seed,
        )
        csm = compute_cross_spectral_matrix(
            signals,
            config.signal.sampling_rate_hz,
            config.source.frequency_hz,
        )

        cbf_result = cbf.run_from_csm(array, case.plane, csm, config.source.frequency_hz)

        t0 = time.perf_counter()
        damas_result = damas.run_from_cbf_map(cbf_result, array)
        wall_time_s = time.perf_counter() - t0

        heatmap_path = writer.write_heatmap(
            damas_result,
            source_index=case.index,
            source_x_m=float(source_pos[0]),
            source_y_m=float(source_pos[1]),
            interpolation="nearest",
            mark_peak=True,
        )
        heatmap_paths.append(heatmap_path)

        metrics = compute_damas_fista_metrics(damas_result, source_pos, wall_time_s)
        metrics["source_index"] = case.index
        metrics_rows.append(metrics)

        if progress:
            conv = "converged" if metrics["converged"] else "NOT converged"
            print(
                f"iter={metrics['iterations']}  "
                f"err={metrics['localization_error_m']:.3f}m  "
                f"{conv}  "
                f"({wall_time_s:.1f}s)"
            )

    csv_path = config.output_dir / "damas_fista_single_source" / "metrics.csv"
    write_metrics_csv(metrics_rows, csv_path)

    if progress:
        print(f"\nmetrics -> {csv_path}")
        print(f"done ({len(cases)} cases).")

    return DAMASFISTAExperimentSummary(
        case_count=len(cases),
        scan_step_m=scan_step_m,
        heatmap_paths=heatmap_paths,
        metrics_csv_path=csv_path,
    )


def compute_damas_fista_metrics(
    result: BeamformingResult,
    source_position_m: np.ndarray,
    wall_time_s: float,
) -> dict:
    """从 DAMAS-FISTA 结果提取定位、稀疏度和收敛指标。"""

    plane = result.plane
    power = result.raw_power
    metadata = dict(result.metadata or {})

    peak_idx = int(np.argmax(power))
    peak_point = plane.points_m[peak_idx]
    peak_power = float(power[peak_idx])

    localization_error_m = float(np.linalg.norm(peak_point - source_position_m))

    denom = max(peak_power, 1e-30)
    db_rel = 10.0 * np.log10(np.maximum(power, 1e-30) / denom)
    points_above_minus_3db = int(np.sum(db_rel >= -3.0))
    points_above_minus_10db = int(np.sum(db_rel >= -10.0))
    points_above_minus_20db = int(np.sum(db_rel >= -20.0))

    power_sum = float(np.sum(power))
    if power_sum > 0:
        p = power / power_sum
        renyi_alpha3 = -0.5 * float(np.log(np.sum(p**3)))
    else:
        renyi_alpha3 = float("nan")

    return {
        "source_index": int(metadata.get("source_index", 0)),
        "source_x_m": float(source_position_m[0]),
        "source_y_m": float(source_position_m[1]),
        "source_z_m": float(source_position_m[2]),
        "peak_x_m": float(peak_point[0]),
        "peak_y_m": float(peak_point[1]),
        "peak_z_m": float(peak_point[2]),
        "localization_error_m": localization_error_m,
        "points_above_minus_3db": points_above_minus_3db,
        "points_above_minus_10db": points_above_minus_10db,
        "points_above_minus_20db": points_above_minus_20db,
        "renyi_entropy_alpha3": renyi_alpha3,
        "iterations": int(metadata.get("iterations", 0)),
        "converged": bool(metadata.get("converged", False)),
        "relative_change": float(metadata.get("relative_change", float("nan"))),
        "residual_norm": float(metadata.get("residual_norm", float("nan"))),
        "runtime_s": wall_time_s,
    }


def write_metrics_csv(rows: list[dict], output_path: Path) -> None:
    """将指标写入 CSV 文件。"""

    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
