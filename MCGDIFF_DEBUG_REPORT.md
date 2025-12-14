# MCG-diff调试报告

## 诊断结果

### 1. 降级测试结果

**num_particles=1, 无resampling时**：
- MCG-diff recon norm: 32.25 (target的4.88倍)
- DPS recon norm: 2.59 (target的0.39倍)
- **结论**: ✗ num_particles=1时，MCG-diff和DPS结果差异大 → **问题在SVD空间操作（Ut/V/Vt、M/S、scaling）**

### 2. ESS和prob_sum监控

- ESS始终为1.00（因为num_particles=1）
- prob_sum始终为1.000e+00
- **结论**: 无resampling时，权重分布正常

### 3. 关键发现：observation_t的norm过大

**observation_t的构造**：
- `observation_t = Ut(observation) * (M / S)`
- 由于M全为1，等价于 `Ut(observation) / S`
- obs_ut norm: 48.48
- observation_t norm: **96.24** (放大2倍)

**原因**：
- S的最小值是0.207，最大值是13.82
- 除以最小S会导致放大4.84倍
- 某些维度被大幅放大

**影响**：
- 即使K很小（0.08），K * observation_t的norm仍有7.85
- 对更新有显著影响

### 4. 更新公式分析

MCG-diff的更新公式：
```python
x_masked = K * observation_t + (1 - K) * x_next_t + sqrt(K) * sigma_next * noise
x_unmasked = x_next_t + sqrt(factor) * noise
x_t = M * x_masked + (1 - M) * x_unmasked
```

**问题**：
- 当M全为1时，x_t = x_masked
- x_masked中，K*observation_t的贡献（norm=7.85）与(1-K)*x_next_t的贡献（norm=36.63）相比，虽然较小但仍然显著
- observation_t的norm过大，导致更新被"拉向"observation_t

### 5. 对比其他算法

- **FPS**: `observation_t = Ut(observation) / S` (与MCG-diff相同)
- **DDRM**: `observation_t = Ut(observation) * (M / S)` (与MCG-diff相同)
- **DDNM**: 不使用observation_t，而是使用pseudo_inverse

## 下一步调试计划

### 优先级1: 检查MCG-diff官方实现

1. 下载MCG-diff官方repo
2. 对比observation_t的构造方式
3. 对比K的计算公式
4. 对比更新公式

### 优先级2: 检查SVD分解

1. 验证A的SVD分解是否正确
2. 检查S的最小值是否合理
3. 检查是否需要clip S的最小值

### 优先级3: 尝试修复

1. 对observation_t进行归一化
2. 调整K的计算公式
3. 调整更新公式中的权重

## 当前状态

- ✓ 已修复observation_t的形状不匹配
- ✓ 已添加数值稳定性处理
- ✓ 已添加ESS和prob_sum监控
- ✓ 已确认问题在SVD空间操作，而非resampling
- ⚠️ observation_t的norm过大，可能是根本原因
