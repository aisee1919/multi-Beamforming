# 初始状态 — 2026-05-05

## 项目基线

- **算法**: 仅实现 CBF（常规波束形成）
- **架构**: 功能性/过程式，无类方法，数据类作为类型化容器
- **阵列**: 8臂 × 8阵元螺旋阵列，孔径 0.15 m
- **声源**: 9个单频点声源 (25 kHz)，3个距离 × 3个位置
- **扫描平面**: 3个平面 (z = 1.2, 1.6, 2.0 m)，步长 0.01 m
- **信号模拟**: 理想自由场，1/r衰减，无噪声

## 文件清单

| 文件 | 用途 |
|------|------|
| `src/beamforming_sim/array_geometry.py` | 螺旋阵列几何生成 |
| `src/beamforming_sim/scene.py` | 扫描平面、声源模型 |
| `src/beamforming_sim/signals.py` | 麦克风信号仿真 |
| `src/beamforming_sim/beamforming.py` | CBF算法 + CSM计算 |
| `src/beamforming_sim/visualization.py` | dB热力图绘制 |
| `src/beamforming_sim/__init__.py` | 包导出 |
| `main.py` | 入口：单声源CBF演示 |
| `tests/test_cbf.py` | CBF正确性测试 |
| `tests/test_simulation_setup.py` | 几何/声源/信号测试 |
| `tests/test_visualization.py` | 可视化工具测试 |
