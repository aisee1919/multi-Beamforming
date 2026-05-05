# 项目架构重构 - 2026-05-05

## 变更摘要

按“低耦合、高内聚”的模式拆分项目职责：

1. `main.py` 只保留程序入口和结果打印。
2. 算法层改为类接口，返回原始功率结果，不在算法内部做显示归一化。
3. CSM 计算从算法文件中拆出，作为后续 DAMAS、CLEAN-SC 等算法复用的频谱中间层。
4. 单声源工况、实验配置、结果写入独立成实验层，避免入口脚本继续堆参数和循环。
5. 保留旧版 `beamforming.py` 函数入口，继续返回归一化数组，兼容已有测试和脚本。

## 新增模块

| 模块 | 职责 |
|------|------|
| `src/beamforming_sim/domain/` | 领域对象和结果对象，新增 `BeamformingResult` |
| `src/beamforming_sim/spectral/` | 频谱计算，当前包含 `compute_cross_spectral_matrix()` |
| `src/beamforming_sim/algorithms/` | CBF、FB 算法实现和统一接口 |
| `src/beamforming_sim/experiments/` | 实验配置、单声源工况、实验运行器、结果写入 |

## 修改文件

| 文件 | 变更 |
|------|------|
| `main.py` | 改为调用 `BeamformingExperiment(ExperimentConfig()).run()` |
| `src/beamforming_sim/beamforming.py` | 改为兼容门面，旧函数继续输出归一化能量图 |
| `src/beamforming_sim/__init__.py` | 导出新架构入口 |
| `tests/test_architecture.py` | 新增架构边界测试 |

## 设计决策

- 算法类返回 `BeamformingResult.raw_power`：保留跨距离、跨算法、跨声源位置比较所需的绝对功率信息。
- 归一化移动到结果对象和兼容门面：绘图仍可使用峰值归一化，但不会污染算法结果。
- CSM 独立：后续高分辨率算法可以直接复用同一频域中间产物。
- `SourceCase` 独立：明确“候选声源集合”和“当前只激活一个声源”的差别。
- `ResultWriter` 独立：输出路径、标题和热力图写入不再散落在入口脚本中。

## 验证结果

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
# 22 passed in 1.34s

.\.venv\Scripts\python.exe main.py
# 生成 9 张 CBF 热力图和 36 张 FB 热力图
```

## 后续建议

下一步如果继续扩展算法，应优先新增 `Algorithm` 类和对应测试，不再往 `main.py` 或兼容门面里加入实验逻辑。
