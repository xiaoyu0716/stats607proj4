# 在 Google Colab 上训练 Toy Image Lesion Diffusion Model

## 方法1: 使用 Colab Notebook（推荐）

### 步骤1: 创建新的 Colab Notebook

1. 打开 [Google Colab](https://colab.research.google.com/)
2. 创建新 notebook
3. 设置运行时类型为 GPU: `Runtime` → `Change runtime type` → `GPU`

### 步骤2: 安装依赖和上传代码

在第一个 cell 中运行：

```python
# 安装依赖
!pip install torch torchvision numpy scipy matplotlib tqdm

# 克隆或上传代码
# 方法A: 如果代码在 GitHub
# !git clone https://github.com/your-repo/InverseBench.git
# %cd InverseBench

# 方法B: 手动上传文件（推荐用于快速测试）
# 1. 在 Colab 左侧菜单点击文件夹图标
# 2. 上传以下文件：
#    - models/toy_mlp_diffusion.py
#    - inverse_problems/toy_image_lesion.py
#    - inverse_problems/base.py
#    - scripts/train_toy_image_lesion_colab.py
```

### 步骤3: 设置路径

```python
import sys
import os

# 如果代码在子目录，添加到路径
sys.path.insert(0, '/content/InverseBench')  # 根据实际路径调整
# 或
sys.path.insert(0, '/content')  # 如果文件直接在 content 目录
```

### 步骤4: 运行训练

```python
# 运行训练脚本
exec(open('scripts/train_toy_image_lesion_colab.py').read())
# 或
# !python scripts/train_toy_image_lesion_colab.py
```

### 步骤5: 下载模型

```python
from google.colab import files
files.download('toy_image_lesion_diffusion.pt')
```

## 方法2: 使用压缩包上传

### 步骤1: 在本地创建压缩包

```bash
# 在 InverseBench 目录下
tar -czf colab_files.tar.gz \
  models/toy_mlp_diffusion.py \
  inverse_problems/toy_image_lesion.py \
  inverse_problems/base.py \
  scripts/train_toy_image_lesion_colab.py \
  configs/pretrain/toy_image_lesion.yaml
```

### 步骤2: 在 Colab 中上传和解压

```python
# 上传文件
from google.colab import files
uploaded = files.upload()

# 解压
!tar -xzf colab_files.tar.gz

# 设置路径
import sys
sys.path.insert(0, '/content')
```

### 步骤3: 运行训练

```python
!python scripts/train_toy_image_lesion_colab.py
```

## 方法3: 直接使用 GitHub（如果代码已推送）

```python
# 克隆仓库
!git clone https://github.com/your-username/InverseBench.git
%cd InverseBench

# 安装依赖
!pip install torch torchvision numpy scipy matplotlib tqdm

# 运行训练（使用 Colab 版本）
!python scripts/train_toy_image_lesion_colab.py
```

## 需要的文件清单

最小文件集：
- `models/toy_mlp_diffusion.py` - MLP 模型
- `inverse_problems/toy_image_lesion.py` - 问题定义
- `inverse_problems/base.py` - 基类
- `scripts/train_toy_image_lesion_colab.py` - 训练脚本（Colab 版本）

可选文件：
- `configs/pretrain/toy_image_lesion.yaml` - 配置文件（脚本中已硬编码参数）

## 快速测试脚本

在 Colab 中运行这个来快速测试：

```python
# 安装依赖
!pip install torch torchvision numpy scipy matplotlib tqdm

# 创建必要的目录结构
import os
os.makedirs('models', exist_ok=True)
os.makedirs('inverse_problems', exist_ok=True)
os.makedirs('scripts', exist_ok=True)

# 然后上传文件或使用 !wget/!curl 下载
# 最后运行训练
```

## 注意事项

1. **GPU 限制**: Colab 免费版 GPU 有时间限制（通常 12 小时），Pro 版本更长
2. **文件持久化**: Colab 的文件在会话结束后会丢失，记得下载模型
3. **路径问题**: 确保 Python 路径设置正确
4. **依赖版本**: Colab 可能已有部分包，注意版本兼容性

## 训练完成后

下载文件：
```python
from google.colab import files
files.download('toy_image_lesion_diffusion.pt')
files.download('toy_image_lesion_prior.pt')  # 如果需要
```

## 生成图片（可选）

如果需要生成图片，也可以上传 `scripts/generate_toy_image_lesion.py` 和 `algo/unconditional.py`，然后运行生成脚本。
