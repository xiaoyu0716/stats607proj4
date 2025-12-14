# 项目中已实现的方法清单

## 📋 总览

项目中实现了 **10 种**后验采样方法，所有方法都支持向量化（vectorized）实现，适用于线性逆问题 `y = Ax + ε`。

---

## ✅ 已实现的方法（按字母顺序）

### 1. **DAPS** - Diffusion Anisotropic Posterior Sampling
- **向量化实现**: `method/algo/daps_vec.py`
- **采样器接口**: `method/samplers/daps.py`
- **图像版本**: 无（仅向量化版本）
- **状态**: ✅ 完整实现

### 2. **DDNM** - Denoising Diffusion Null-Space Model
- **向量化实现**: `method/algo/ddnm_vec.py`
- **采样器接口**: `method/samplers/ddnm.py`
- **图像版本**: `method/algo/ddnm.py`
- **状态**: ✅ 完整实现（向量化 + 图像版本）

### 3. **DDRM** - Denoising Diffusion Restoration Model
- **向量化实现**: `method/algo/ddrm_vec.py`
- **采样器接口**: `method/samplers/ddrm.py`
- **图像版本**: `method/algo/ddrm.py`
- **状态**: ✅ 完整实现（向量化 + 图像版本）

### 4. **DPS** - Diffusion Posterior Sampling
- **向量化实现**: `method/algo/dps_vec.py`
- **采样器接口**: `method/samplers/dps.py`
- **图像版本**: 无（仅向量化版本）
- **状态**: ✅ 完整实现

### 5. **DPS-Paper** - DPS 论文原始版本
- **向量化实现**: `method/algo/dps_vec_paper.py`
- **采样器接口**: `method/samplers/dps_paper.py`
- **图像版本**: 无（仅向量化版本）
- **状态**: ✅ 完整实现

### 6. **MCG** - Monte Carlo Guided Diffusion
- **向量化实现**: 通过粒子滤波实现（`method/particle_filter.py`）
- **采样器接口**: `method/samplers/mcg.py`
- **图像版本**: 无（专为向量化设计）
- **状态**: ✅ 完整实现

### 7. **PiGDM** - Pseudoinverse-Guided Diffusion Model
- **向量化实现**: `method/algo/pigdm_vec.py`
- **采样器接口**: `method/samplers/pigdm.py`
- **图像版本**: `method/algo/pigdm.py`
- **状态**: ✅ 完整实现（向量化 + 图像版本）

### 8. **PnP-DM** - Plug-and-Play Diffusion Model ⭐
- **向量化实现**: `method/algo/pnpdm_vec.py`
- **采样器接口**: `method/samplers/pnpdm.py`
- **图像版本**: `method/algo/pnpdm.py`
- **状态**: ✅ 完整实现（向量化 + 图像版本）
- **特点**: 使用 Langevin 动力学 + Denoiser 的交替更新

### 9. **REDDiff** - RED-diff (InverseBench adapter)
- **向量化实现**: 通过 InverseBench 适配器
- **采样器接口**: `method/samplers/reddiff.py`
- **图像版本**: 无（仅向量化版本）
- **状态**: ✅ 完整实现

### 10. **SCORE-IP** - Score-based Inverse Problem solver
- **向量化实现**: `method/algo/scoreip_vec.py`
- **采样器接口**: `method/samplers/scoreip.py`
- **图像版本**: 无（仅向量化版本）
- **状态**: ✅ 完整实现

---

## 📊 实现统计

### 按实现类型分类

| 类型 | 数量 | 方法 |
|------|------|------|
| **仅向量化版本** | 5 | DAPS, DPS, DPS-Paper, REDDiff, SCORE-IP |
| **向量化 + 图像版本** | 4 | DDNM, DDRM, PiGDM, PnP-DM |
| **特殊实现** | 1 | MCG (粒子滤波) |

### 核心组件

1. **前向算子抽象** (`method/algo/forward_ops.py`):
   - `LinearGaussianForwardOp`: 线性高斯前向算子
   - `ToySVDOp`: SVD 分解的前向算子（用于 DDRM/DDNM）

2. **适配器** (`method/algo/ddrm_adapters.py`):
   - `VPToEDMDenoiser`: VP 到 EDM 的 denoiser 适配器

3. **调度器** (`method/algo/scheduler.py`):
   - 扩散过程的 sigma 调度

---

## 🔍 方法详细说明

### 1. PnP-DM (Plug-and-Play Diffusion Model) ⭐
- **算法**: Langevin 动力学 + Denoiser 交替更新
- **特点**: 
  - Annealing schedule (指数衰减的 sigma)
  - 内层 Langevin 循环 (J 次迭代)
  - 外层 Denoiser 更新
- **配置参数**:
  - `pnp_gamma`: Langevin 步长
  - `pnp_J`: Langevin 迭代次数
  - `pnp_tau`: 噪声参数
  - `pnp_num_anneal`: Annealing 步数
  - `pnp_rho`: 指数衰减率

### 2. DPS (Diffusion Posterior Sampling)
- **算法**: 在扩散过程中添加数据一致性梯度
- **特点**: 
  - 确定性 DDIM 或随机 SDE
  - 可配置的 guidance scale
- **配置参数**:
  - `dps_guidance`: 数据一致性权重
  - `dps_eta`: SDE 噪声水平
  - `dps_det`: 是否使用确定性采样

### 3. DDNM (Denoising Diffusion Null-Space Model)
- **算法**: 在 null space 和 range space 分别处理
- **特点**: 
  - 利用 SVD 分解
  - 支持多次 null-space 更新 (L 参数)
- **配置参数**:
  - `ddnm_eta`: 噪声水平
  - `ddnm_L`: Null-space 更新次数
  - `ddnm_steps`: 扩散步数

### 4. DDRM (Denoising Diffusion Restoration Model)
- **算法**: 基于 SVD 的伪逆引导
- **特点**: 
  - 使用 SVD 分解前向算子
  - 支持不同的调度策略
- **配置参数**:
  - `ddrm_eta_b`: 噪声参数
  - 调度器配置

### 5. DAPS (Diffusion Anisotropic Posterior Sampling)
- **算法**: 各向异性后验采样
- **特点**: 
  - 考虑不同维度的各向异性
  - 自适应步长调整
- **配置参数**:
  - `daps_lambda`: 正则化参数
  - `daps_lr`: 学习率
  - `daps_steps`: 迭代步数

### 6. PiGDM (Pseudoinverse-Guided Diffusion Model)
- **算法**: 伪逆引导的扩散模型
- **特点**: 
  - 使用伪逆算子进行引导
  - 支持噪声和非噪声模式
- **配置参数**:
  - `pigdm_eta`: 噪声参数
  - `pigdm_noisy`: 是否添加噪声

### 7. MCG (Monte Carlo Guided Diffusion)
- **算法**: 粒子滤波 + 扩散模型
- **特点**: 
  - 使用粒子滤波进行后验采样
  - 支持重采样策略
- **配置参数**:
  - `mcg_particles`: 粒子数量
  - `mcg_no_resample`: 是否禁用重采样

### 8. SCORE-IP (Score-based Inverse Problem)
- **算法**: 基于 score 的逆问题求解
- **特点**: 
  - Predictor-corrector 风格
  - 可配置的梯度裁剪
- **配置参数**:
  - `scoreip_guidance`: 引导权重
  - `scoreip_eta`: 噪声水平
  - `scoreip_grad_clip`: 梯度裁剪阈值

### 9. REDDiff
- **算法**: RED-diff 的适配版本
- **特点**: 
  - 来自 InverseBench
  - 使用固定的 lambda 调度
- **配置参数**:
  - `reddiff_steps`: 迭代步数
  - `reddiff_lambda`: 正则化参数
  - `reddiff_lr`: 学习率

### 10. DPS-Paper
- **算法**: DPS 的论文原始实现
- **特点**: 
  - 与标准 DPS 略有不同
  - 更接近原始论文的实现
- **配置参数**: 与 DPS 类似

---

## 🚀 使用方法

所有方法都通过统一的接口调用：

```python
from method.samplers import get_sampler

sampler = get_sampler(
    method="pnpdm",  # 或其他方法名
    score_model=score_model,
    timesteps=timesteps,
    device="cuda",
    A=A,  # [m, d]
    y=y,  # [m] 或 [n, m]
    Sigma_eps_diag=Sigma_eps_diag,  # [m]
    # 方法特定参数...
)

output = sampler.sample(n_samples=200, dim=8)
samples = output.samples  # [200, 8]
```

---

## 📝 代码组织

```
method/
├── algo/                    # 算法核心实现
│   ├── *_vec.py            # 向量化版本
│   ├── *.py                # 图像版本（部分方法）
│   ├── forward_ops.py      # 前向算子
│   ├── scheduler.py        # 调度器
│   └── ddrm_adapters.py    # 适配器
│
└── samplers/               # 采样器接口
    ├── base.py             # 基类和工厂函数
    ├── *.py                # 各方法的采样器包装
    └── mcg.py              # MCG 特殊实现
```

---

## ✅ 总结

- **10 种方法**全部实现
- **所有方法**都有向量化版本
- **4 种方法**同时有图像版本（DDNM, DDRM, PiGDM, PnP-DM）
- **统一接口**便于对比实验
- **完整配置**支持参数调优

**核心贡献**: 将多种扩散模型方法从图像领域扩展到一般线性逆问题，实现了完整的向量化框架。

