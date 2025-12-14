# Uncertainty Quantification (UQ) Simulation Analysis - Design Document

## 概述

本脚本提供了模块化的不确定性量化（UQ）分析框架，用于系统评估多个 diffusion-based inverse algorithms 的 posterior sampling 和 uncertainty quantification 能力。

## 快速开始

### 1. 测试模块是否正常工作
```bash
cd /home/xiaoyuq/InverseBench
python scripts/test_uq_modules.py
```

这会运行一个快速测试，验证所有模块是否正常工作。

**注意**: 如果遇到梯度相关的错误，确保：
- `toy_gausscmog8_diffusion.pt` 文件存在
- 模型文件格式正确（包含 'ema' 或 'net' key）

### 2. 运行单个实验
```bash
# Coverage 分析（最快）
python scripts/uq_simulation_analysis.py --experiment coverage --N 50 --K 20

# Nullspace 方差分析
python scripts/uq_simulation_analysis.py --experiment nullspace --N 50 --K 20

# PSNR vs Uncertainty 分析
python scripts/uq_simulation_analysis.py --experiment tradeoff --N 50 --K 20
```

### 3. 运行完整分析（推荐用于最终报告）
```bash
python scripts/uq_simulation_analysis.py --experiment all --N 200 --K 100
```

## 支持的算法

### Posterior Sampling 方法（支持不确定性量化）
- **MCG_diff**: Monte Carlo guided diffusion
- **PnPDM**: Principled Probabilistic Imaging using Diffusion Models
- **DPS**: Diffusion Posterior Sampling
- **DAPS**: Decoupled Annealing Posterior Sampling

### Point-Estimate 方法（仅提供重建，无不确定性）
- **DDRM**: Denoising Diffusion Restoration Models
- **DDNM**: Denoising Diffusion Null-space Models
- **DiffPIR**: Diffusion Models for Plug-and-Play Image Restoration
- **ReDiff**: Regularized Diffusion

## 模块化架构

脚本按功能分为以下模块，每个模块都可以独立使用：

### 1. 数据生成模块 (`generate_dataset`)

**功能**: 生成测试数据集

**使用示例**:
```python
from scripts.uq_simulation_analysis import generate_dataset

# 生成 A=I 的数据集
dataset_identity = generate_dataset(
    A_type='identity',
    N=200,              # 200个测试样本
    noise_std=0.5,      # 观测噪声标准差
    seed=0,             # 随机种子
)

# 生成 A=MRI-like 的数据集
dataset_mri = generate_dataset(
    A_type='mri_like',
    N=200,
    noise_std=0.5,
    A_seed=1234,        # A矩阵的随机种子
)
```

**返回数据结构**:
```python
{
    'x0': np.array,      # (N, 16) 真实潜在向量
    'y': np.array,       # (N, 16) 观测值
    'A': np.array,       # (16, 16) 前向算子矩阵
    'U': np.array,       # (16, 16) SVD的U矩阵（MRI-like时）
    'S': np.array,       # (16,) 奇异值（MRI-like时）
    'V': np.array,       # (16, 16) SVD的V^T矩阵（MRI-like时）
    'problem': object,   # ToyGausscMoG8Problem 实例
}
```

### 2. 方法执行模块 (`run_method_on_dataset`)

**功能**: 在数据集上运行指定算法，收集 posterior 样本或 point estimate

**使用示例**:
```python
from scripts.uq_simulation_analysis import run_method_on_dataset

# 运行 DPS 方法
result = run_method_on_dataset(
    method_name='DPS',
    dataset=dataset_identity,
    K=100,              # 每个观测收集100个样本
    config_overrides={'guidance_scale': 2.0},  # 可选：覆盖配置
    device='cpu',
    verbose=True,
)

# 结果结构
# result['samples']: (N, K, 16) 或 None
# result['mean']: (N, 16) posterior 均值或 point estimate
# result['meta']: 方法元信息
```

### 3. 实验模块

#### 3.1 实验1: Coverage 分析 (`experiment_coverage_identity`)

**目标**: 评估 A=I 情况下，95% credible interval 的 coverage

**使用示例**:
```python
from scripts.uq_simulation_analysis import experiment_coverage_identity

coverage_results = experiment_coverage_identity(
    methods=['DPS', 'PnPDM', 'MCG_diff', 'DAPS'],
    N=200,
    K=100,
    noise_std=0.5,
    device='cpu',
)

# 结果结构
# coverage_results[method_name] = {
#     'per_dim_coverage': np.array(16,),  # 每维的coverage
#     'global_coverage': float,            # 全局平均coverage
# }
```

**输出**: 
- 每个方法的 per-dimension coverage bar plot
- 全局 coverage 统计（目标值：0.95）

#### 3.2 实验2: Nullspace 方差分析 (`experiment_nullspace_variance_mri`)

**目标**: 评估 A=MRI-like 情况下，nullspace 方向的方差是否更大

**使用示例**:
```python
from scripts.uq_simulation_analysis import experiment_nullspace_variance_mri

nullspace_results, S = experiment_nullspace_variance_mri(
    methods=['DPS', 'PnPDM', 'MCG_diff', 'DAPS'],
    N=200,
    K=100,
    noise_std=0.5,
    device='cpu',
)

# 结果结构
# nullspace_results[method_name] = {
#     'var_per_singular_dim': np.array(16,),  # 每个SVD维度的方差
#     'var_observed_mean': float,             # 观测方向平均方差
#     'var_null_mean': float,                  # nullspace方向平均方差
# }
```

**输出**:
- 每个方法的 per-dimension variance plot（区分 observed/null）
- var_observed_mean vs var_null_mean 对比图

#### 3.3 实验3: PSNR vs Uncertainty Trade-off (`experiment_psnr_vs_uncertainty`)

**目标**: 展示 PSNR 和 uncertainty 的权衡关系

**使用示例**:
```python
from scripts.uq_simulation_analysis import experiment_psnr_vs_uncertainty

tradeoff_result = experiment_psnr_vs_uncertainty(
    method_name='DPS',
    hyperparam_grid={
        'guidance_scale': [1, 2, 4, 8, 10],
        # 可以添加更多超参数
    },
    A_type='identity',
    N=200,
    K=100,
    noise_std=0.5,
    device='cpu',
)

# 结果结构
# tradeoff_result = {
#     'grid': list[dict],           # 超参数配置列表
#     'psnr': np.array,             # 每个配置的PSNR
#     'avg_uncertainty': np.array,  # 每个配置的平均uncertainty
# }
```

**输出**:
- PSNR vs Uncertainty 散点图（不同超参数用不同颜色/标记）

### 4. 可视化模块

#### 4.1 `plot_coverage_bar`
绘制每个方法的 per-dimension coverage bar plot

#### 4.2 `plot_nullspace_variance`
绘制 nullspace variance 分析结果（per-dimension + mean comparison）

#### 4.3 `plot_psnr_vs_uncertainty`
绘制 PSNR vs Uncertainty trade-off 散点图

## 命令行使用

### 运行所有实验
```bash
cd /home/xiaoyuq/InverseBench
python scripts/uq_simulation_analysis.py --experiment all --N 200 --K 100
```

### 只运行 Coverage 分析
```bash
python scripts/uq_simulation_analysis.py --experiment coverage --N 200 --K 100
```

### 只运行 Nullspace 方差分析
```bash
python scripts/uq_simulation_analysis.py --experiment nullspace --N 200 --K 100
```

### 只运行 PSNR vs Uncertainty 分析
```bash
python scripts/uq_simulation_analysis.py --experiment tradeoff --N 200 --K 100
```

### 指定特定方法
```bash
python scripts/uq_simulation_analysis.py --experiment coverage --methods DPS PnPDM --N 200
```

### 完整参数列表
```bash
python scripts/uq_simulation_analysis.py \
    --experiment all \
    --methods DPS PnPDM MCG_diff DAPS \
    --N 200 \
    --K 100 \
    --noise_std 0.5 \
    --output_dir exps/uq_analysis \
    --device cpu
```

## Python API 使用（模块化）

### 示例1: 只生成数据集
```python
from scripts.uq_simulation_analysis import generate_dataset

dataset = generate_dataset('identity', N=100, noise_std=0.5)
print(f"Generated {dataset['x0'].shape[0]} samples")
print(f"x0 shape: {dataset['x0'].shape}")
print(f"y shape: {dataset['y'].shape}")
```

### 示例2: 只运行单个方法
```python
from scripts.uq_simulation_analysis import generate_dataset, run_method_on_dataset

# 生成数据
dataset = generate_dataset('identity', N=50, noise_std=0.5)

# 运行方法
result = run_method_on_dataset('DPS', dataset, K=50, device='cpu', verbose=True)

# 分析结果
if result['samples'] is not None:
    print(f"Posterior samples shape: {result['samples'].shape}")
    print(f"Posterior mean shape: {result['mean'].shape}")
    print(f"Posterior std: {result['samples'].std(axis=1).mean()}")
```

### 示例3: 自定义实验
```python
from scripts.uq_simulation_analysis import (
    generate_dataset, 
    run_method_on_dataset,
    plot_coverage_bar
)
import numpy as np

# 生成数据
dataset = generate_dataset('identity', N=100, noise_std=0.5)

# 运行多个方法
methods = ['DPS', 'PnPDM']
coverage_results = {}

for method in methods:
    result = run_method_on_dataset(method, dataset, K=100, device='cpu')
    
    if result['samples'] is not None:
        samples = result['samples']  # (N, K, 16)
        x0 = dataset['x0']  # (N, 16)
        
        # 计算 coverage
        mu = samples.mean(axis=1)
        std = samples.std(axis=1, ddof=1)
        lower = mu - 1.96 * std
        upper = mu + 1.96 * std
        
        per_dim_coverage = []
        for i in range(16):
            cov_i = np.mean((x0[:, i] >= lower[:, i]) & (x0[:, i] <= upper[:, i]))
            per_dim_coverage.append(cov_i)
        
        coverage_results[method] = {
            'per_dim_coverage': np.array(per_dim_coverage),
            'global_coverage': np.mean(per_dim_coverage),
        }

# 可视化
plot_coverage_bar(coverage_results, save_path='my_coverage.png')
```

## 输出文件结构

运行后会在 `output_dir` 下生成：

```
exps/uq_analysis/
├── coverage_identity.png          # Coverage 分析图
├── nullspace_variance.png         # Nullspace 方差分析图
├── psnr_vs_uncertainty_dps.png   # PSNR vs Uncertainty 图
└── summary.txt                    # 文本总结
```

## 配置说明

### 算法配置文件位置
- MCG_diff: `configs/algorithm/mcgdiff_toy.yaml`
- PnPDM: `configs/algorithm/pnpdm_toy.yaml`
- DPS: `configs/algorithm/dps_toy.yaml`
- DAPS: `configs/algorithm/daps_toy.yaml`
- 其他方法类似

### 超参数覆盖
可以通过 `config_overrides` 参数覆盖配置文件中的超参数：

```python
result = run_method_on_dataset(
    'DPS',
    dataset,
    config_overrides={
        'guidance_scale': 5.0,
        'diffusion_scheduler_config': {
            'num_steps': 50,
            'sigma_max': 5.0,
        }
    }
)
```

## 性能考虑

- **N (样本数)**: 建议 100-500，越大越准确但越慢
- **K (posterior 样本数)**: 建议 50-200，越大越准确但越慢
- **设备**: 默认使用 CPU，如果有 GPU 可以设置 `--device cuda`

## 故障排除

### 1. 模型文件不存在
确保 `toy_gausscmog8_diffusion.pt` 在项目根目录。如果不存在，需要先训练模型：
```bash
python train_toy_gausscmog8_16d.py
```

### 2. 配置文件找不到
确保配置文件路径正确，脚本会自动从项目根目录查找。如果配置文件不存在，检查 `configs/algorithm/` 目录。

### 3. 内存不足
减少 `N` 或 `K` 的值，或者分批处理。对于大型实验，建议：
- 使用 `--N 100` 而不是 `--N 200`
- 使用 `--K 50` 而不是 `--K 100`
- 使用 `--device cpu` 避免 GPU 内存问题

### 4. 方法不支持采样
只有 MCG_diff, PnPDM, DPS, DAPS 支持 posterior sampling。其他方法只能提供 point estimate。

### 5. 运行时间过长
- 对于快速测试，使用 `--N 10 --K 5`
- 对于中等规模，使用 `--N 50 --K 20`
- 对于完整分析，使用 `--N 200 --K 100`

### 6. 模块导入错误
确保在项目根目录运行脚本：
```bash
cd /home/xiaoyuq/InverseBench
python scripts/uq_simulation_analysis.py ...
```

## 扩展指南

### 添加新方法
1. 在 `SUPPORTED_METHODS` 中添加方法信息
2. 确保配置文件存在
3. 确保方法支持 `inference(observation, num_samples=K)` 接口

### 添加新实验
1. 在实验模块部分添加新函数
2. 在 `main()` 中添加调用
3. 添加对应的可视化函数

## 参考文献

- MCG_diff: Monte Carlo guided diffusion for Bayesian linear inverse problems
- PnPDM: Principled Probabilistic Imaging using Diffusion Models as Plug-and-Play Priors
- DPS: Diffusion Posterior Sampling for General Noisy Inverse Problems
- DAPS: Decoupled Annealing Posterior Sampling
