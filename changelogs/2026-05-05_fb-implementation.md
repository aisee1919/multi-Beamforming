# FB 算法实现 — 2026-05-05

## 变更摘要

新增 **函数波束形成 (Functional Beamforming)** 算法：

**公式**: B_FB(ν, x) = [w(x)^H · C^(1/ν) · w(x)]^ν

**核心逻辑**:
1. 对 CSM 做特征分解：eigenvalues, eigenvectors = eigh(CSM)
2. 计算 C^(1/ν) = U · Σ^(1/ν) · U^H
3. 用向量化 einsum 计算 w^H · C^(1/ν) · w，再升幂到 ν
4. ν = 1 时退化为 CBF；ν 越大主瓣越窄、旁瓣越低

## 修改文件

| 文件 | 变更 |
|------|------|
| `src/beamforming_sim/beamforming.py` | +`functional_beamforming()`, +`run_fb_for_planes()`, +`_csm_power_eig()`, +`_fb_from_csm()` |
| `src/beamforming_sim/__init__.py` | +`functional_beamforming`, +`run_fb_for_planes` 导出 |
| `tests/test_fb.py` | 新建：5 个测试（ν=1 退化为 CBF、峰值位置正确、C^(1/ν) Hermitian、多平面 API、ν 越大峰越窄） |
| `main.py` | 新增 FB 演示：9 声源 × 4 种 ν = 36 张热力图 |

## 测试结果

```
17 passed in 1.24s  (12 原有 + 5 新增)
```

## 回滚方法

```bash
git revert HEAD   # 撤销本次提交（保留历史）
# 或
git reset --hard HEAD~1  # 完全回到初始状态（需确认）
```
