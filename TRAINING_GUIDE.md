# 训练16维toy_gausscmog8 Prior模型 - 完整指南

## 问题诊断

**发现的问题**：
1. 当前训练数据 `toy_gausscmog8_prior.pt` 的后8维std只有约1.0，而不是预期的2.24 (sqrt(5.0))
2. 这说明训练数据可能不是从真正的16D prior采样的

**正确的16D prior应该是**：
- 前8维: MoG，std约1.0
- 后8维: 弱高斯先验（方差=5.0），std约2.24

## 训练步骤

### 步骤1: 重新生成正确的16维prior数据

```bash
cd /home/xiaoyuq/InverseBench
python regenerate_prior_data.py
```

这个脚本会：
- 直接构建16D prior（前8维MoG，后8维弱高斯，方差=5.0）
- 采样50000个16维样本
- 保存为 `toy_gausscmog8_prior.pt` (形状: [50000, 1, 4, 4])
- 验证后8维std应该是2.24

**关键接入点**：`regenerate_prior_data.py` 第45-60行
```python
# 构建16D prior协方差
Sigma0_16x16[8:, 8:] = torch.eye(8) * 5.0  # 后8维方差=5.0

# 采样
v_16d = means_samples + z  # [N, 16] - 真正的16维
x_img = v_16d.reshape(N, 1, 4, 4)  # reshape为图像格式
```

### 步骤2: 训练模型

```bash
python train_toy_gausscmog8_16d.py
```

这个脚本会：
1. 检查并加载/生成16维prior数据
2. 构建16维diffusion模型（`ToyDiffusionMLP(dim=16)`）
3. 训练模型
4. 保存到 `toy_gausscmog8_diffusion.pt`

**关键接入点**：

#### 2.1 数据加载（`train_toy_gausscmog8_16d.py` 第38-95行）
```python
# 加载16维训练数据
x0 = torch.load("toy_gausscmog8_prior.pt")  # [50000, 1, 4, 4]
x0_vec = x0.view(50000, -1)  # [50000, 16]
# 验证：后8维std应该约2.24
assert x0_vec[:, 8:].std().item() > 2.0
```

#### 2.2 模型定义（`configs/pretrain/toy_gausscmog8.yaml`）
```yaml
model:
  _target_: models.toy_mlp_diffusion.ToyDiffusionMLP
  dim: 16  # ← 这里指定16维
  hidden: 128
```

#### 2.3 训练循环（`train_toy_gausscmog8_16d.py` 第100-130行）
```python
# 训练时使用图像格式 [batch, 1, 4, 4]
x = x0[idx]  # [batch_size, 1, 4, 4] - 16维图像格式
x_t = x + sigma * noise  # 添加噪声
eps_pred = model(x_t, sigma)  # 模型预测，内部reshape为16维处理
loss = ((eps_pred - noise)**2).mean()
```

**关键**：模型内部会自动将 `[batch, 1, 4, 4]` reshape为 `[batch, 16]` 进行处理。

## 验证训练结果

训练完成后，验证模型维度：

```python
import torch
checkpoint = torch.load('toy_gausscmog8_diffusion.pt', map_location='cpu')
model_state = checkpoint['net']
output_dim = model_state['net.4.weight'].shape[0]
print(f"模型输出维度: {output_dim}")  # 应该是 16
assert output_dim == 16, "模型维度错误！"
```

## 在推理时的接入点

### 1. 加载模型（`main.py` 或算法代码）

```python
from models.toy_mlp_diffusion import ToyDiffusionMLP

checkpoint = torch.load('toy_gausscmog8_diffusion.pt')
model = ToyDiffusionMLP(dim=16)  # ← 16维
model.load_state_dict(checkpoint['net'])
```

### 2. 模型调用（算法代码，如 `algo/dps.py`）

```python
# 输入: x_t [batch, 1, 4, 4] (16维图像格式)
# 模型内部: reshape为 [batch, 16]，处理，再reshape回 [batch, 1, 4, 4]
denoised = model(x_t / scaling_step, sigma)  # [batch, 1, 4, 4]
```

## 关键检查点

1. **训练数据**：`toy_gausscmog8_prior.pt`
   - 形状: `[50000, 1, 4, 4]`
   - 后8维std应该约2.24

2. **模型配置**：`configs/pretrain/toy_gausscmog8.yaml`
   - `dim: 16` ← 确保是16

3. **模型文件**：`toy_gausscmog8_diffusion.pt`
   - 最后一层权重形状: `[16, 128]` ← 输出16维

4. **问题配置**：`configs/problem/toy_gausscmog8.yaml`
   - `dim: 16`
   - `A_type: fixed-full-rank-16x16`
   - `A_obs_dim: 16`

## 总结

- **Prior**: 16维（前8维MoG，后8维弱高斯，方差=5.0）
- **训练数据**: 16维，reshape为 `[N, 1, 4, 4]`
- **模型**: 16维输入输出，内部处理16维向量
- **A矩阵**: 16x16满秩
- **y观测**: 16维（A是16x16，所以y也是16维）

所有维度都应该是16维！
