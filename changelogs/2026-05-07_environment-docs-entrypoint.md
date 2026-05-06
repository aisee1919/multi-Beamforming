# WSL 环境、文档与入口对齐 — 2026-05-07

## 变更

1. 新增 WSL 虚拟环境 `.venv-wsl`，通过 `pip install -e .` 注册本地包，并将环境目录加入 `.gitignore`。
2. 新增 `README.md`，说明环境搭建、测试命令、默认实验和 DAMAS-FISTA 显式入口。
3. 将 DAMAS-FISTA 单声源实验逻辑下沉到 `beamforming_sim.experiments.damas_fista`。
4. `main.py` 新增 `--include-damas-fista`，入口可以覆盖当前全部算法；默认仍只跑 CBF、FB、FFT-FISTA。
5. 专用脚本 `scripts/run_damas_fista_single_source.py` 改为调用共享实验函数。
6. 更新项目复盘文档和 DAMAS-FISTA 需求文档，反映当前实现状态。

## 验证

```bash
.venv-wsl/bin/python -m pytest -q
```

结果：49 项测试通过。
