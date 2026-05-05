# 信号可视化 — 2026-05-05

## 变更：声源信号可视化

新增 `plot_source_signals()`，绘制每个声源发射端的原始正弦波形（传播前）。

## 修改文件

| 文件 | 变更 |
|------|------|
| `src/beamforming_sim/visualization.py` | +`plot_source_signals()`: 为 SourceModel 中每个声源独立绘制时域波形，2 ms 窗口 |
| `main.py` | 调用 `plot_source_signals(source_model, ...)` 生成 `outputs/source_signals.png`；移除不再需要的 `numpy`/`AcousticSource` 导入 |

## 变更 v2：增加 AWGN 叠噪对比

`plot_source_signals()` 新增 `noise_std` 参数，`noise_std > 0` 时输出三列布局：纯净信号 | AWGN | 叠噪信号。仅取首个声源为代表。

## 修改文件

| 文件 | 变更 |
|------|------|
| `src/beamforming_sim/visualization.py` | `plot_source_signals()`: 支持 `noise_std`/`random_seed` 参数，三列对比布局 |
| `main.py` | 传入 `noise_std=0.5, random_seed=42`，生成含噪对比图 |

## 输出

- `outputs/source_signals.png` — 三列：clean signal | AWGN (σ=0.5) | signal + noise
