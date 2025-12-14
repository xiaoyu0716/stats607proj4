# 修复 algo 模块缺失问题

## 问题描述

运行 `uq_simulation_analysis.py` 时出现错误：
```
ModuleNotFoundError: No module named 'algo'
ImportError: Error loading 'algo.dps.DPS'
```

## 原因

配置文件（如 `configs/algorithm/dps_toy.yaml`）中的 `_target_` 字段指向 `algo.dps.DPS`，但项目中没有 `algo` 模块。

## 解决方案

### 方案 1: 检查是否有其他仓库包含 algo 模块

`algo` 模块可能在其他仓库中。请检查：
1. 是否有其他相关的 GitHub 仓库
2. 是否需要从另一个仓库克隆 `algo` 模块

### 方案 2: 创建 algo 模块的包装（临时方案）

如果 `algo` 模块确实不存在，可以创建一个包装模块：

```bash
# 创建 algo 目录结构
mkdir -p algo/dps
mkdir -p algo/daps
# ... 其他算法目录

# 创建 __init__.py 文件
touch algo/__init__.py
touch algo/dps/__init__.py
touch algo/daps/__init__.py
```

然后需要实现相应的算法类。

### 方案 3: 修改配置文件（如果算法实现在其他位置）

如果算法实现在其他位置（如 `utils` 或其他模块），可以修改配置文件中的 `_target_` 路径。

## 检查步骤

1. 检查是否有其他仓库：
```bash
# 检查 git remote
git remote -v

# 检查是否有其他相关目录
ls -la ../
```

2. 检查项目文档：
- 查看 README.md
- 查看 INSTALL_UQ_COVERAGE.md
- 查看其他安装文档

3. 检查是否需要从另一个仓库克隆：
```bash
# 如果有其他仓库 URL，使用之前创建的脚本
./clone_without_git.sh <仓库URL> ./algo
```

## 建议

**最可能的情况**：`algo` 模块应该在项目的另一个部分，或者需要从另一个仓库获取。

请检查：
1. 项目是否有多个仓库？
2. 是否有安装说明提到需要额外的模块？
3. 是否可以联系项目维护者获取 `algo` 模块？

