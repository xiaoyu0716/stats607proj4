# 训练16维toy_gausscmog8 Prior模型指南

## 概述

这个指南说明如何训练16维的toy_gausscmog8 diffusion prior模型，并确认所有维度都正确。

## 关键点

### 1. Prior分布是16维的

- **前8维**: Mixture of Gaussians (MoG)
  - 2个分量，均值在第7维分别为-2.0和+2.0
  - 协方差矩阵：Toeplitz结构，rho=0.8
  
- **后8维**: 弱高斯先验
  - 均值为0
  - 方差=5.0（较大的方差，表示弱先验）

### 2. 训练数据生成

训练数据通过 `ToyGausscMoG8Problem.sample_prior(N)` 生成：
- 输入：`N` (样本数量，如50000)
- 输出：`[N, 1, 4, 4]` (16维reshape为图像格式)
- 实际是16维向量，reshape为1x4x4图像

### 3. 模型结构

- **模型类型**: `ToyDiffusionMLP`
- **输入维度**: 16 (reshape为 [batch, 1, 4, 4])
- **输出维度**: 16 (reshape为 [batch, 1, 4, 4])
- **网络结构**: 
  - Linear(17, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, 16)
  - 输入17是因为concat了sigma

## 训练步骤

### 步骤1: 运行训练脚本

```bash
cd /home/xiaoyuq/InverseBench
python train_toy_gausscmog8_16d.py
```

### 步骤2: 检查训练数据

训练脚本会自动生成 `toy_gausscmog8_prior.pt`，包含50000个16维样本。

验证数据维度：
```python
import torch
prior_data = torch.load('toy_gausscmog8_prior.pt')
print(f"训练数据形状: {prior_data.shape}")  # 应该是 [50000, 1, 4, 4]
prior_vec = prior_data.view(50000, -1)  # [50000, 16]
print(f"向量形状: {prior_vec.shape}")  # 应该是 [50000, 16]
print(f"前8维std: {prior_vec[:, :8].std()}")  # 应该约1.0
print(f"后8维std: {prior_vec[:, 8:].std()}")  # 应该约2.24 (sqrt(5.0))
```

### 步骤3: 检查模型

训练完成后，验证模型维度：
```python
import torch
checkpoint = torch.load('toy_gausscmog8_diffusion.pt', map_location='cpu')
model_state = checkpoint['net']
output_dim = model_state['net.4.weight'].shape[0]
print(f"模型输出维度: {output_dim}")  # 应该是 16
```

## 在代码中的接入点

### 1. Prior数据生成（`scripts/gen_toy_gausscmog8_dataset.py` 或训练脚本）

```python
from inverse_problems.toy_gausscmog8 import ToyGausscMoG8Problem

problem = ToyGausscMoG8Problem(
    dim=16,  # ← 这里指定16维
    A_type="random-gaussian",
    # ... 其他参数
)

# 采样16维prior
x0_img = problem.sample_prior(N)  # [N, 1, 4, 4] - 16维
```

**关键**: `problem.sample_prior()` 返回的是16维的MoG样本，reshape为图像格式。

### 2. 模型定义（`configs/pretrain/toy_gausscmog8.yaml`）

```yaml
model:
  _target_: models.toy_mlp_diffusion.ToyDiffusionMLP
  dim: 16  # ← 这里指定16维
  hidden: 128
```

**关键**: `dim=16` 确保模型处理16维数据。

### 3. 训练循环（`train_toy_gausscmog8_16d.py`）

```python
# 加载16维训练数据
x0 = torch.load("toy_gausscmog8_prior.pt")  # [50000, 1, 4, 4]

# 训练时使用图像格式
x = x0[idx]  # [batch_size, 1, 4, 4]
x_t = x + sigma * noise  # 添加噪声
eps_pred = model(x_t, sigma)  # 模型预测
```

**关键**: 训练时使用 `[batch, 1, 4, 4]` 格式，模型内部会reshape为16维处理。

### 4. 推理时使用（`main.py`）

```python
# 加载模型
checkpoint = torch.load('toy_gausscmog8_diffusion.pt')
model = ToyDiffusionMLP(dim=16)  # ← 16维
model.load_state_dict(checkpoint['net'])

# 推理时输入输出都是 [batch, 1, 4, 4]
x = model(x_t, sigma)  # 16维图像格式
```

## 验证16维是否正确

运行以下检查脚本：

```python
import torch
from inverse_problems.toy_gausscmog8 import ToyGausscMoG8Problem

# 创建问题
problem = ToyGausscMoG8Problem(
    dim=16,
    A_type='fixed-full-rank-16x16',
    A_obs_dim=16,
    # ... 其他参数
)

# 检查prior
samples = problem.sample_prior(100)
samples_vec = problem._img_to_vec(samples)  # [100, 16]
print(f"Prior样本形状: {samples_vec.shape}")  # 应该是 [100, 16]
print(f"前8维std: {samples_vec[:, :8].std()}")  # 约1.0
print(f"后8维std: {samples_vec[:, 8:].std()}")  # 约2.24

# 检查forward
x0_img, y_img = problem.generate_sample()
x0_vec = problem._img_to_vec(x0_img.unsqueeze(0))
y_vec = problem._img_to_vec(y_img.unsqueeze(0).unsqueeze(0))
print(f"x0维度: {x0_vec.shape}")  # [1, 16]
print(f"y维度: {y_vec.shape}")  # [1, 16]
print(f"y后8维是否全0: {(y_vec[0, 8:].abs() < 1e-6).all()}")  # False（因为A是16x16）
```

## 常见问题

### Q: 为什么后8维的std较大？

A: 因为后8维使用的是弱高斯先验，方差=5.0，所以std≈2.24。

### Q: y的后8维应该是什么？

A: 如果A是16x16满秩矩阵，y的后8维不应该全为0，因为A会混合所有16维的信息。

### Q: 如何确认训练数据真的是16维的？

A: 检查 `toy_gausscmog8_prior.pt`：
```python
prior_data = torch.load('toy_gausscmog8_prior.pt')
prior_vec = prior_data.view(-1, 16)
print(f"后8维std: {prior_vec[:, 8:].std()}")  # 应该约2.24，不是接近0
```
