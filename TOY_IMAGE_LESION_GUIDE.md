# 16×16 图像 Toy Example 运行指南

## 概述

这是一个16×16图像的toy example，用于测试逆问题算法。特点：
- **输入尺寸**: 16×16 (256维)
- **观测尺寸**: 8×8 (64维，欠定比例0.25)
- **前向算子**: 高斯模糊 (σ=1.0) + 2×下采样
- **先验**: 两分量高斯混合（正常背景 vs 病灶）

## 快速开始

### 1. 训练 Diffusion Prior

首先需要训练一个diffusion model作为先验：

```bash
python train_toy_image_lesion.py
```

这会：
- 从先验分布生成10000个训练样本
- 训练一个MLP diffusion model
- 保存模型到 `toy_image_lesion_diffusion.pt`

**注意**: 训练时间约几分钟（取决于GPU）

### 2. 运行推理

训练完成后，可以运行推理：

```bash
python main.py problem=toy_image_lesion algorithm=dps pretrain=toy_image_lesion
```

可用的算法：
- `dps` - Diffusion Posterior Sampling
- `daps` - Diffusion Annealed Posterior Sampling  
- `ddnm` - Denoising Diffusion Null-space Model
- `ddrm` - Denoising Diffusion Restoration Models
- 等等...

### 3. 测试基本功能（无需训练）

如果想先测试problem类的基本功能：

```python
from inverse_problems.toy_image_lesion import ToyImageLesionProblem

# 创建problem
op = ToyImageLesionProblem(
    blur_sigma=1.0,
    noise_std=0.03,
    tau=0.2,
    lesion_prior_weight=0.1,
    lesion_amplitude=0.25,
    lesion_radius=3,
    device='cpu'
)

# 生成样本
x0, y = op.generate_sample()
print(f"Ground truth: {x0.shape}, Observation: {y.shape}")

# 计算闭式后验
posterior = op.compute_posterior_variance(y.unsqueeze(0))
print(f"Updated weights: {posterior['updated_component_weights']}")
print(f"Posterior mean shape: {posterior['total_posterior_mean'].shape}")
```

## 配置文件说明

### Problem配置 (`configs/problem/toy_image_lesion.yaml`)

```yaml
model:
  blur_sigma: 1.0      # 高斯模糊sigma（建议0.8-1.2）
  noise_std: 0.03      # 观测噪声（建议0.02-0.04）
  tau: 0.2             # 先验std
  lesion_prior_weight: 0.1   # 病灶先验权重π
  lesion_amplitude: 0.25     # 病灶幅度（建议0.15-0.35）
  lesion_radius: 3           # 病灶半径（像素）
```

### 训练配置 (`configs/pretrain/toy_image_lesion.yaml`)

可以调整：
- `training.total_steps`: 训练步数（默认50000）
- `training.batch_size`: 批次大小（默认64）
- `training.num_samples`: 先验样本数（默认10000）

## 参数调优建议

根据你的需求，可能需要调整以下参数：

1. **病灶幅度** (`lesion_amplitude`): 
   - 范围: 0.15-0.35
   - 目标: 使更新权重 w₁(y) ≈ 0.1-0.4（既"罕见"又"可行"）

2. **模糊程度** (`blur_sigma`):
   - 需要更模糊: 提高到1.2
   - 需要更可分: 降到0.8

3. **观测噪声** (`noise_std`):
   - 范围: 0.02-0.04
   - 影响: 噪声越大，重建越困难

## 验证现象

运行后可以验证以下现象：

1. **两种解释**: 可视化正常/病灶两种后验解释
2. **MAP假阴性**: 统计 k*=1 且 w₀>w₁ 的比例
3. **PSNR惩罚**: 从病灶分量采样时PSNR会显著低于从正常分量采样

## 文件结构

```
inverse_problems/toy_image_lesion.py      # Problem类实现
configs/problem/toy_image_lesion.yaml     # Problem配置
configs/pretrain/toy_image_lesion.yaml    # 训练配置
train_toy_image_lesion.py                 # 训练脚本
toy_image_lesion_prior.pt                 # 先验数据（自动生成）
toy_image_lesion_diffusion.pt             # 训练好的模型
```

## 故障排除

1. **内存不足**: 减少 `training.batch_size` 或 `training.num_samples`
2. **训练太慢**: 减少 `training.total_steps`（至少10000步）
3. **重建质量差**: 检查算法超参数，参考论文Table 12

## 下一步

- 可视化代码：可以添加可视化来展示两种解释
- 扩展到32×32：如果需要更漂亮的图
- 算法对比：运行多个算法并对比结果







