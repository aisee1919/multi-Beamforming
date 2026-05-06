# beamforming-sim

声学波束形成多算法仿真项目。项目在自由场近场条件下模拟点声源和麦克风阵列接收信号，并生成不同算法的声源定位热力图。

这是实验性项目，不面向发布，不承诺旧 API 兼容。代码默认假设输入来自本项目配置和脚本，避免为错误调用路径堆防御性校验。

## 当前功能

- 阵列：64 阵元八臂螺旋麦克风阵列，默认孔径 0.15 m。
- 声源：25 kHz 点声源，默认 3 个距离平面 x 3 个横向位置，共 9 个单声源 case。
- 信号：按传播时延和 1/r 衰减生成多通道麦克风信号，可配置高斯白噪声。
- 频谱：在目标频率处计算交叉谱矩阵 CSM。
- 算法：CBF、FB、FFT-FISTA、DAMAS-FISTA。
- 输出：阵列图、声源波形图、算法热力图、DAMAS-FISTA 指标 CSV。

## WSL 环境

项目目录中保留一个 WSL 专用虚拟环境：

```bash
python3 -m venv .venv-wsl
.venv-wsl/bin/python -m pip install -r requirements.txt
.venv-wsl/bin/python -m pip install -e .
```

验证：

```bash
.venv-wsl/bin/python -m pytest -q
```

当前验证结果：39 项测试通过。

## 运行默认实验

```bash
.venv-wsl/bin/python main.py
```

默认实验运行 CBF、FB、FFT-FISTA。输出写入 `outputs/`：

- `outputs/array_layout.png`
- `outputs/source_signals.png`
- `outputs/cbf_single_source/`
- `outputs/fb_single_source/`
- `outputs/fft_fista_single_source/`

## 运行 DAMAS-FISTA

DAMAS-FISTA 构造完整 PSF 算子，默认不绑定到 0.01 m 网格的常规流程。需要显式开启：

```bash
.venv-wsl/bin/python main.py --include-damas-fista
```

也可以直接运行专用脚本：

```bash
.venv-wsl/bin/python scripts/run_damas_fista_single_source.py --step 0.1
.venv-wsl/bin/python scripts/run_damas_fista_single_source.py --step 0.05
```

输出写入：

- `outputs/damas_fista_single_source/*.png`
- `outputs/damas_fista_single_source/metrics.csv`

## 代码结构

- `src/beamforming_sim/array_geometry.py`：麦克风阵列几何。
- `src/beamforming_sim/scene.py`：声源模型和扫描平面。
- `src/beamforming_sim/signals.py`：麦克风接收信号仿真。
- `src/beamforming_sim/spectral/csm.py`：CSM 估计。
- `src/beamforming_sim/algorithms/`：CBF、FB、FFT-FISTA、DAMAS-FISTA。
- `src/beamforming_sim/experiments/`：实验配置、编排、结果写出。
- `tests/`：核心数值、接口和架构测试。

## API 原则

- 使用 `ConventionalBeamformer`、`FunctionalBeamformer`、`FFTFISTABeamformer`、`DAMASFISTABeamformer` 类接口。
- 算法层返回 `BeamformingResult.raw_power` 原始功率；绘图或比较时再调用 `normalized_power()` 或 `power_db()`。
- 已移除旧版函数式兼容入口，如 `conventional_beamforming()`、`functional_beamforming()`、`run_cbf_for_planes()`、`run_fb_for_planes()`。
