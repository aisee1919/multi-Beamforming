# 实验性项目瘦身 — 2026-05-07

## 变更

1. 明确项目定位为实验性仿真项目，不面向发布，不承诺旧 API 兼容。
2. 删除旧版函数式兼容入口：
   - `conventional_beamforming()`
   - `functional_beamforming()`
   - `run_cbf_for_planes()`
   - `run_fb_for_planes()`
3. 从包根导出中移除旧函数入口，只保留 Beamformer 类和实验所需对象。
4. 删除大部分防御性参数校验，包括配置、信号、CSM、算法参数和结果容器中的显式 `ValueError` 分支。
5. 删除专门验证防御性校验的测试，保留数值正确性、接口和实验语义测试。
6. 更新 README、项目复盘和架构改进意见，反映实验性项目原则。

## 验证

```bash
.venv-wsl/bin/python -m pytest -q
```

结果：39 项测试通过。
