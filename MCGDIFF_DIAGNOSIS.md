# MCG-diff诊断总结

## ✅ 已完成步骤

### 0. 承认事实
- ✓ DAPS / RED-diff / DPS 在同一toy problem上表现合理
- ✓ 只有MCG-diff崩（norm大好几倍）
- **结论**: 问题高度集中在MCG-diff的实现上

### 1. 降级测试

#### 1.1 num_particles=1
- ✓ 已设置num_particles=1
- ✓ 结果：MCG-diff recon norm = 32.25 (4.88x)，DPS = 2.59 (0.39x)
- **结论**: ✗ num_particles=1时差异仍大 → **问题在SVD空间操作（Ut/V/Vt、M/S、scaling）**

#### 1.2 关闭resampling
- ✓ 已关闭resampling (ENABLE_RESAMPLING = False)
- ✓ ESS始终为1.00，prob_sum始终为1.0
- **结论**: 问题不在resampling，而在deterministic部分

### 2. 数值调试

#### 2.1 ESS和prob_sum监控
- ✓ 已添加ESS和prob_sum打印
- ✓ ESS=1.00（因为num_particles=1）
- ✓ prob_sum=1.0（正常）

#### 2.2 关键发现：observation_t的norm过大
- observation_t norm: **96.24** (y的norm是48.48，放大2倍)
- 原因：S的最小值是0.207，除以S导致某些维度被放大4.84倍
- 影响：即使K很小（0.08），K * observation_t的norm仍有7.85，对更新有显著影响

### 3. 对比分析

#### 3.1 observation_t的构造
- MCG-diff: `Ut(observation) * (M / S)` (由于M全1，等价于`Ut(observation) / S`)
- FPS: `Ut(observation) / S` (相同)
- DDRM: `Ut(observation) * (M / S)` (相同)
- **结论**: observation_t的构造方式与其他算法一致

#### 3.2 SVD分解验证
- A的条件数: 66.85 (合理)
- S的最小值: 0.207 (A的真实奇异值，不应该被clip)
- **结论**: SVD分解正确，S的值合理

#### 3.3 更新公式分析
```python
x_masked = K * observation_t + (1 - K) * x_next_t + sqrt(K) * sigma_next * noise
x_unmasked = x_next_t + sqrt(factor) * noise
x_t = M * x_masked + (1 - M) * x_unmasked
```
- 当M全为1时，x_t = x_masked
- K * observation_t的norm: 7.85
- (1-K) * x_next_t的norm: 36.63
- **问题**: observation_t的norm过大，导致更新被"拉向"observation_t

## 🔍 下一步（按优先级）

### 优先级1: 检查MCG-diff官方实现
- [ ] 下载MCG-diff官方repo
- [ ] 对比observation_t的构造方式
- [ ] 对比K的计算公式
- [ ] 对比更新公式

### 优先级2: 检查是否需要clip S
- [ ] 检查官方实现是否对S进行clip
- [ ] 如果clip，clip到多少

### 优先级3: 检查更新公式
- [ ] 验证K的计算是否正确
- [ ] 验证masked/unmasked更新是否正确
- [ ] 验证M的使用是否正确

## 📊 当前状态

- ✓ 已确认问题在SVD空间操作
- ✓ 已确认问题不在resampling
- ⚠️ observation_t的norm过大，可能是根本原因
- ⚠️ 需要对比官方实现确认
