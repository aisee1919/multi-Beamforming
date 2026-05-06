# DAMAS-FISTA 单声源实验需求

## 当前实现状态

已实现：

1. `DAMASFISTABeamformer` 位于 `src/beamforming_sim/algorithms/damas_fista.py`。
2. 单声源实验编排位于 `src/beamforming_sim/experiments/damas_fista.py`。
3. 专用脚本保留为 `scripts/run_damas_fista_single_source.py`。
4. `main.py` 新增 `--include-damas-fista`，显式运行 DAMAS-FISTA；默认流程仍只跑 CBF、FB、FFT-FISTA。
5. WSL 环境使用 `.venv-wsl`，验证命令为 `.venv-wsl/bin/python -m pytest -q`。

保持显式入口的原因：默认 `0.01 m` 网格有 14641 个扫描点，DAMAS-FISTA 精确 PSF 算子成本明显高于 CBF/FB/FFT-FISTA，不应在无用户确认的默认入口中强制运行。

## 目标

在现有项目中补充 DAMAS-FISTA 算法实验。CBF、FB、FFT-FISTA 已经跑过，本阶段只需要对 9 个候选声源点分别运行 DAMAS-FISTA，并输出热力图和定量指标。

核心实验语义：

```text
9 个候选声源点
每次只激活 1 个声源
共运行 9 个单声源 DAMAS-FISTA case
```

## 实验范围

### 必须做

1. 使用当前八臂螺旋麦克风阵列：
   - 阵元数：64
   - 阵列孔径：0.15 m
   - 阵列平面：z = 0

2. 使用当前三个扫描平面：
   - z = 1.2 m
   - z = 1.6 m
   - z = 2.0 m

3. 每个扫描平面上设置三个候选声源位置：
   - `(0.0, 0.0)`
   - `(0.5, 0.0)`
   - `(-0.5, 0.0)`

4. 共 9 个候选声源点，但每次仿真只使用一个声源：

   ```python
   SourceModel([source])
   ```

5. 声源信号：
   - 频率：25 kHz
   - 信号类型：正弦信号
   - 当前阶段不加高斯白噪声

6. 对每个单声源 case 运行 DAMAS-FISTA。

7. 输出每个 case 的 DAMAS-FISTA 热力图和指标记录。

### 不做

1. 不重新运行 CBF、FB、FFT-FISTA 的完整对比实验。
2. 不把 9 个声源同时放进同一个 `SourceModel`。
3. 不训练 DAMAS-FISTA-Net。
4. 不做网络模型。
5. 不在默认主流程中强制运行大网格 DAMAS-FISTA。

## 推荐实现策略

### 1. DAMAS-FISTA 只作为显式实验运行

DAMAS-FISTA 精确矩阵法计算量大，不应默认绑定到 `main.py` 的常规流程中。

推荐新建独立脚本：

```text
scripts/run_damas_fista_single_source.py
```

该脚本只负责运行 9 个 DAMAS-FISTA 单声源 case。

当前实现采用 `experiments` 层专用 Runner，脚本和 `main.py --include-damas-fista` 都调用同一套函数，避免实验逻辑分叉。

### 2. 先用较粗网格验证

默认 `0.01 m` 网格对应 `14641` 个扫描点，精确 DAMAS-FISTA 运行成本过高。

建议先使用：

```text
extent: -0.6 m 到 0.6 m
step: 0.05 m 或 0.1 m
```

推荐分两阶段：

| 阶段 | 网格步长 | 用途 |
|---|---:|---|
| 快速验证 | 0.1 m | 检查 9 个点是否都能定位正确 |
| 正式小规模对比 | 0.05 m | 查看主瓣压缩、旁瓣抑制和收敛情况 |

暂不建议直接用 `0.01 m` 跑 9 个 DAMAS-FISTA case。

### 3. DAMAS-FISTA 参数

推荐初始参数：

```python
DAMASFISTABeamformer(
    max_iterations=1000,
    tolerance=1e-4,
    dense_point_limit=5000,
    scan_chunk_size=256,
)
```

原因：

- `tolerance=1e-4` 更接近文献建议范围。
- `max_iterations=1000` 避免 200 次迭代未收敛就停止。
- `dense_point_limit=5000` 允许 `0.05 m` 网格使用 dense 矩阵，提高小网格实验速度。
- `scan_chunk_size=256` 保留大网格时的内存保护。

## 算法接口调整需求

### 1. 避免频率手动传错

当前 `run_from_cbf_map()` 如果需要手动传 `frequency_hz`，存在 dirty map 和 PSF 频率不一致的风险。

建议改为：

```python
run_from_cbf_map(
    self,
    cbf_result: BeamformingResult,
    array: MicrophoneArray,
    sound_speed_m_s: float | None = None,
) -> BeamformingResult
```

内部使用：

```python
frequency_hz = cbf_result.frequency_hz
sound_speed_m_s = sound_speed_m_s or cbf_result.sound_speed_m_s
```

### 2. metadata 增加收敛状态

当前 metadata 应至少包含：

```python
{
    "iterations": ...,
    "relative_change": ...,
    "residual_norm": ...,
    "lipschitz": ...,
    "matrix_mode": ...,
    "point_count": ...,
    "converged": ...
}
```

其中：

```python
converged = relative_change < tolerance
```

这样可以区分“跑完最大迭代次数”和“实际收敛”。

### 3. 增加网格保护

建议增加显式保护参数：

```python
max_point_count: int | None = 5000
```

如果扫描点数超过限制，除非用户显式关闭限制，否则报错：

```text
DAMAS-FISTA point count exceeds max_point_count. Use a coarser grid or explicitly override max_point_count.
```

## 实验流程

每个候选声源点执行以下流程：

```text
取第 i 个候选声源
  -> SourceModel([source])
  -> simulate_microphone_signals()
  -> compute_cross_spectral_matrix()
  -> ConventionalBeamformer.run_from_csm()
  -> DAMASFISTABeamformer.run_from_cbf_map()
  -> 保存 DAMAS-FISTA 热力图
  -> 记录定量指标
```

注意：这里使用 CBF 只是为了得到 DAMAS-FISTA 的 dirty map，不需要重新输出 CBF 热力图。

## 输出要求

### 1. 热力图

输出目录：

```text
outputs/damas_fista_single_source/
```

文件名格式：

```text
damas_fista_source_01_z_1.2m_step_0.10m_x_+0.0_y_+0.0.png
damas_fista_source_02_z_1.2m_step_0.10m_x_+0.5_y_+0.0.png
damas_fista_source_03_z_1.2m_step_0.10m_x_-0.5_y_+0.0.png
...
```

共 9 张图。

### 2. 指标表

建议输出：

```text
outputs/damas_fista_single_source/metrics.csv
```

字段：

| 字段 | 含义 |
|---|---|
| `source_index` | 声源编号，1 到 9 |
| `source_x_m` | 真实声源 x 坐标 |
| `source_y_m` | 真实声源 y 坐标 |
| `source_z_m` | 真实声源 z 坐标 |
| `peak_x_m` | DAMAS-FISTA 峰值 x 坐标 |
| `peak_y_m` | DAMAS-FISTA 峰值 y 坐标 |
| `peak_z_m` | DAMAS-FISTA 峰值 z 坐标 |
| `localization_error_m` | 峰值定位误差 |
| `points_above_minus_3db` | -3 dB 以上网格数 |
| `points_above_minus_10db` | -10 dB 以上网格数 |
| `points_above_minus_20db` | -20 dB 以上网格数 |
| `renyi_entropy_alpha3` | 能量集中度指标 |
| `iterations` | 实际迭代次数 |
| `converged` | 是否满足收敛阈值 |
| `relative_change` | 最终相对变化量 |
| `residual_norm` | 最终残差范数 |
| `runtime_s` | 单个 case 运行时间 |

## 验收标准

### 功能验收

1. 只运行 9 个单声源 DAMAS-FISTA case。
2. 每个 case 的 `SourceModel` 中只能有 1 个声源。
3. 输出 9 张 DAMAS-FISTA 热力图。
4. 输出 1 个 `metrics.csv`。
5. 每个结果都返回 `BeamformingResult`，且 `raw_power` 不在算法层归一化。

### 数值验收

在 `step = 0.1 m` 时：

1. 9 个声源的峰值位置应落在真实声源网格点上，或定位误差不超过一个网格步长。
2. `raw_power` 应全部非负、有限。
3. 每个 case 应记录 `converged` 状态。

在 `step = 0.05 m` 时：

1. 单声源主峰应比 CBF 更集中。
2. `points_above_minus_10db` 应明显少于 CBF。
3. 如果未收敛，指标表必须显示 `converged = false`，不能静默当作成功。

### 性能验收

1. `step = 0.1 m` 的 9 个 case 应能在可接受时间内完成。
2. `step = 0.05 m` 可以作为正式小规模实验。
3. `step = 0.01 m` 不作为本阶段验收目标。

## 测试要求

新增或补充测试：

1. `run_from_cbf_map()` 默认使用 `cbf_result.frequency_hz`。
2. DAMAS-FISTA metadata 包含 `converged`。
3. 超过 `max_point_count` 时会报错。
4. 单声源 case 构造时，每个 `SourceModel` 只有一个声源。
5. 小网格单声源 DAMAS-FISTA 峰值位置正确。

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

## 最终交付

1. DAMAS-FISTA 单声源运行脚本或专用 Runner。
2. 9 张 DAMAS-FISTA 热力图。
3. `metrics.csv` 指标表。
4. 对应测试。
5. changelog 记录本次 DAMAS-FISTA 单声源实验变更。
