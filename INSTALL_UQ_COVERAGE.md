# 安装说明 - UQ Coverage 实验

## 快速开始

### 1. 安装依赖

```bash
# 基础安装（CPU版本）
pip install -r requirements.txt

# 或者使用精简版本（只包含必需包）
pip install -r requirements_uq_coverage.txt
```

### 2. 安装本地包

```bash
# 安装项目本地模块（inverse_problems 等）
pip install -e .
```

### 3. 运行 Coverage 实验

```bash
python scripts/uq_simulation_analysis.py \
  --experiment coverage \
  --methods DPS DAPS \
  --N 200 \
  --K 100
```

## 依赖包说明

### 必需包

| 包名 | 用途 | 版本要求 |
|------|------|----------|
| `numpy` | 数值计算 | >=1.26.0 |
| `torch` | PyTorch 深度学习框架 | >=2.0.0 |
| `torchvision` | 图像处理工具 | >=0.15.0 |
| `omegaconf` | 配置文件管理 | >=2.3.0 |
| `hydra-core` | 配置框架 | >=1.3.0 |
| `pandas` | 数据处理 | >=2.0.0 |
| `matplotlib` | 可视化 | >=3.7.0 |
| `tqdm` | 进度条 | >=4.65.0 |

### GPU 支持（可选）

如果有 NVIDIA GPU，建议安装 CUDA 版本的 PyTorch：

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 验证安装

运行以下命令检查所有包是否已安装：

```bash
python3 << 'EOF'
required = ['numpy', 'torch', 'matplotlib', 'omegaconf', 'hydra', 'tqdm', 'pandas']
missing = []
for pkg in required:
    try:
        __import__(pkg)
        print(f"✅ {pkg}")
    except ImportError:
        print(f"❌ {pkg} - NOT INSTALLED")
        missing.append(pkg)
if missing:
    print(f"\n请安装缺失的包: pip install {' '.join(missing)}")
else:
    print("\n✅ 所有依赖已安装！")
EOF
```

## 常见问题

### 1. ImportError: No module named 'inverse_problems'

**解决方案**：运行 `pip install -e .` 安装本地包

### 2. CUDA out of memory

**解决方案**：减少 `--N` 或 `--K` 参数，或使用 CPU：
```bash
python scripts/uq_simulation_analysis.py --experiment coverage --methods DPS --N 50 --K 20 --device cpu
```

### 3. 配置文件找不到

**解决方案**：确保在项目根目录运行脚本：
```bash
cd /path/to/proj4
python scripts/uq_simulation_analysis.py ...
```

## 文件说明

- `requirements.txt` - 完整依赖列表（包含版本约束）
- `requirements_uq_coverage.txt` - 精简版本（只包含必需包）
- `INSTALL_UQ_COVERAGE.md` - 本文件（安装说明）

