# FFT-FISTA 算法实现 — 2026-05-05

## 变更摘要

新增 **FFT-FISTA 反卷积声学成像算法**，与 CBF、FB 对比。

**算法**：
1. 以 CBF 原始功率图为 dirty map
2. 构造中心扫描点的点扩散响应作为平移不变近似 PSF（归一化到最大值 1）
3. FFT 加速 FISTA 求解：min 0.5||PSF∗x−b||² + λ||x||₁, s.t. x≥0
4. 使用 ifftshift 对齐 PSF 峰值到 FFT 原点，保证卷积延迟正确

## 修改文件

| 文件 | 变更 |
|------|------|
| `src/beamforming_sim/algorithms/fft_fista.py` | 新建：`FFTFISTABeamformer` (frozen dataclass)，`run_from_cbf_map()` 入口 |
| `src/beamforming_sim/algorithms/__init__.py` | +`FFTFISTABeamformer` 导出 |
| `src/beamforming_sim/experiments/runner.py` | +FFT-FISTA 流程接入，+`ExperimentSummary.fft_fista_paths` |
| `src/beamforming_sim/experiments/writer.py` | +`write_fft_fista_heatmap()` |
| `main.py` | +FFT-FISTA 输出数量打印 |
| `tests/test_fft_fista.py` | 新建：10 个测试（返回类型、shape、非负、峰值、参数校验） |
| `tests/test_architecture.py` | +架构验证：FFT-FISTA 返回 `BeamformingResult` |

## 参数默认值

- `lambda_reg = 0.02`
- `max_iterations = 200`
- `tolerance = 1e-6`

## 验证结果

```
33 passed in 1.44s  (22 原有 + 11 新增)

中心声源:  peak 误差 0.000m, 稀疏度 625→9
偏移声源:  peak 误差 0.000m, 稀疏度 625→9
```

## 输出

- `outputs/fft_fista_single_source/` — 9 张热力图（9声源 × 1）
- FFT-FISTA 显著提升空间分辨率（稀疏重构），主瓣远窄于 CBF
