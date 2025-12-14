# 实验1: PSNR 与不确定性的错位

## 实验目标

展示在同一个观测 y 下，多个"后验可行"的解 x 在 PSNR 上差异巨大，从而说明用 PSNR/accuracy 评判"对/错"是误导性的。

## 实验配置

- **维度**: n ∈ {4, 8, 16}
- **观测维度**: m = ⌊n/2⌋（欠定比例 50%）
- **前向算子 A**: 随机高斯矩阵，列正交化，确保 rank(A) = m
- **先验**: 
  - 简化版：单高斯 x ~ N(0, I)
  - 完整版：使用 toy_gausscmog8 的混合高斯先验

## 实验方法

### 1. 构造等似然解簇

对于真值 x* 和观测 y = Ax* + ε：

- 计算 A 的零空间方向 v ∈ ker(A)
- 构造解簇：x(λ) = x* + λv, λ ∈ ℝ
- 这些解满足 Ax(λ) ≈ Ax*（在零空间中），因此都在可行集 S(y)

### 2. 计算指标

- **PSNR**: PSNR(x(λ), x*) - 衡量与真值的像素级差异
- **后验对数密度**: log p(x(λ)|y) - 衡量后验质量

### 3. 可视化

**图1: PSNR vs λ 曲线**
- 横轴：λ（零空间系数）
- 左纵轴：PSNR (dB)
- 右纵轴：后验对数密度
- 展示：PSNR 大幅波动，但后验密度相对稳定

**图2: 散点图 - 后验对数密度 vs PSNR**
- 横轴：后验对数密度
- 纵轴：PSNR (dB)
- 展示：相同后验质量的点，PSNR 差异巨大

## 运行实验

### 方法1: 使用简化版（单高斯先验）

```bash
python scripts/experiment_psnr_uncertainty_mismatch.py
```

这会运行 n=8, m=4 的实验。

### 方法2: 使用现有 toy_gausscmog8 问题

```bash
python scripts/experiment_psnr_uncertainty_mismatch_toy8d.py
```

这会：
1. 使用 toy_gausscmog8 的混合高斯先验
2. 使用其实际的 A 矩阵（取前 m 行做欠定）
3. 运行 n=8, m=4 的实验
4. 同时运行 n=4, m=2 和 n=16, m=8 的对比实验

## 成功判据

实验成功的标志：

1. ✓ **PSNR 大幅波动**: 同一 y 下，PSNR 范围很大（如 10-40 dB）
2. ✓ **后验密度相对稳定**: 相同后验质量的点，PSNR 差异 > 10 dB
3. ✓ **错位明显**: 散点图显示后验密度与 PSNR 无明显正相关

## 预期结果

从运行结果看：

- **n=8, m=4**: PSNR 变化 10.63 dB，后验密度相似
- **n=16, m=8**: PSNR 变化 16.03 dB，后验密度相似

这证明了：
- 在欠定问题中，PSNR 不能可靠地衡量解的质量
- 后验密度是更合适的评估指标
- 多个"正确"的解可能具有非常不同的 PSNR

## 输出文件

- `exps/experiments/psnr_uncertainty_mismatch/psnr_uncertainty_mismatch_n8_m4.png`
- `exps/experiments/psnr_uncertainty_mismatch/psnr_uncertainty_mismatch_n16_m8.png`
- `exps/experiments/psnr_uncertainty_mismatch/psnr_uncertainty_mismatch_toy8d_n8_m4.png`

## 关键发现

1. **PSNR 的局限性**: 在欠定问题中，PSNR 主要反映零空间方向的差异，而非解的质量
2. **后验密度的优势**: 后验对数密度能更好地反映解的后验质量
3. **评估指标的误导性**: 使用 PSNR 作为唯一评估指标会误导性地认为某些解"更好"，尽管它们在统计上等价
