# 架构优化 — 2026-05-06

## 变更摘要

按架构改进意见执行高/中优先级优化。

## 高优先

| # | 改进 | 文件变更 |
|---|------|----------|
| 1 | 删除 `domain/{array,source,scan}.py` 三个空垫片 | -3 文件, -18 行; `domain/__init__.py` 改从源模块直接导入 |
| 2 | `algorithms/__init__.py` 导出从 13→4 符号 | 只导出 `Beamformer` + 3 个 Beamformer 类 |
| 3 | 移除 `seaborn` 依赖 | `pyproject.toml` / `requirements.txt` 各删 1 行 |

## 中优先

| # | 改进 | 文件变更 |
|---|------|----------|
| 4 | `writer.py` 3 个 write_*_heatmap 合并为 `write_heatmap` | `writer.py` -42 行; 路径/标题由 `result.algorithm` + `result.metadata` 自动生成 |
| 5 | 抽取 `CsmBasedBeamformer` mixin | `base.py` +20 行; `cbf.py` -9 行; `fb.py` -9 行; 新算法只需实现 `run_from_csm()` |
| 6 | 删除 `beamforming.py` 4 个私有别名 | `beamforming.py` -4 行; `test_fb.py` 改为从 `algorithms.fb` 直接导入 |

## 验证

```
33 passed in 1.43s  (零回归)
main.py → 54 张热力图  (CBF 9 + FB 36 + FFT-FISTA 9)
```
