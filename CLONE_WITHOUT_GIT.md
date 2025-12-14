# 从 GitHub 仓库克隆内容（不保留 git 历史）

## 方法 1: 使用临时目录克隆（推荐）

```bash
# 1. 克隆到临时目录（只克隆最新版本，不包含历史）
git clone --depth 1 <仓库URL> /tmp/temp_repo

# 2. 复制文件到目标位置（排除 .git 文件夹）
cp -r /tmp/temp_repo/* /Users/xiaoyuq/Documents/607/proj4/
cp -r /tmp/temp_repo/.* /Users/xiaoyuq/Documents/607/proj4/ 2>/dev/null || true

# 3. 删除临时目录
rm -rf /tmp/temp_repo
```

**示例**：
```bash
# 假设要克隆 https://github.com/user/repo.git
git clone --depth 1 https://github.com/user/repo.git /tmp/temp_repo
cp -r /tmp/temp_repo/* /Users/xiaoyuq/Documents/607/proj4/
cp -r /tmp/temp_repo/.* /Users/xiaoyuq/Documents/607/proj4/ 2>/dev/null || true
rm -rf /tmp/temp_repo
```

---

## 方法 2: 直接下载 ZIP 文件（最简单）

```bash
# 进入目标目录
cd /Users/xiaoyuq/Documents/607/proj4

# 下载 ZIP 文件（将 <user>/<repo> 替换为实际的用户名和仓库名）
# 例如：https://github.com/user/repo/archive/refs/heads/main.zip
wget https://github.com/<user>/<repo>/archive/refs/heads/main.zip

# 或者使用 curl
curl -L https://github.com/<user>/<repo>/archive/refs/heads/main.zip -o repo.zip

# 解压
unzip repo.zip

# 移动文件到当前目录
mv <repo>-main/* .
mv <repo>-main/.* . 2>/dev/null || true

# 清理
rm -rf <repo>-main repo.zip
```

**示例**：
```bash
cd /Users/xiaoyuq/Documents/607/proj4
curl -L https://github.com/user/repo/archive/refs/heads/main.zip -o repo.zip
unzip repo.zip
mv repo-main/* .
mv repo-main/.* . 2>/dev/null || true
rm -rf repo-main repo.zip
```

---

## 方法 3: 使用 git archive（如果仓库允许）

```bash
# 直接导出到当前目录
git archive --remote=<仓库URL> --format=tar HEAD | tar -x -C /Users/xiaoyuq/Documents/607/proj4
```

**注意**：这个方法需要仓库支持 `git archive`，GitHub 的 HTTPS 仓库通常不支持。

---

## 方法 4: 克隆到子目录然后移动（最安全）

```bash
# 1. 克隆到子目录
cd /Users/xiaoyuq/Documents/607/proj4
git clone --depth 1 <仓库URL> temp_clone

# 2. 移动所有文件（排除 .git）
rsync -av --exclude='.git' temp_clone/ .

# 3. 删除临时目录
rm -rf temp_clone
```

**示例**：
```bash
cd /Users/xiaoyuq/Documents/607/proj4
git clone --depth 1 https://github.com/user/repo.git temp_clone
rsync -av --exclude='.git' temp_clone/ .
rm -rf temp_clone
```

---

## 方法 5: 使用脚本自动化（最方便）

创建一个脚本 `clone_without_git.sh`：

```bash
#!/bin/bash
# 用法: ./clone_without_git.sh <仓库URL> <目标目录>

REPO_URL=$1
TARGET_DIR=${2:-.}

if [ -z "$REPO_URL" ]; then
    echo "用法: $0 <仓库URL> [目标目录]"
    exit 1
fi

# 创建临时目录
TEMP_DIR=$(mktemp -d)
echo "克隆到临时目录: $TEMP_DIR"

# 克隆（只克隆最新版本）
git clone --depth 1 "$REPO_URL" "$TEMP_DIR"

# 复制文件（排除 .git）
echo "复制文件到: $TARGET_DIR"
rsync -av --exclude='.git' "$TEMP_DIR/" "$TARGET_DIR/"

# 清理
rm -rf "$TEMP_DIR"
echo "完成！"
```

**使用**：
```bash
chmod +x clone_without_git.sh
./clone_without_git.sh https://github.com/user/repo.git /Users/xiaoyuq/Documents/607/proj4
```

---

## 注意事项

1. **备份当前文件**：在操作前建议先备份
   ```bash
   cp -r /Users/xiaoyuq/Documents/607/proj4 /Users/xiaoyuq/Documents/607/proj4_backup
   ```

2. **保留当前 git 仓库**：如果只想添加新文件而不影响当前 git，使用方法 4 或 5

3. **处理冲突**：如果目标目录已有同名文件，需要手动处理

4. **SSH vs HTTPS**：
   - SSH: `git@github.com:user/repo.git`
   - HTTPS: `https://github.com/user/repo.git`

---

## 推荐方案

**如果你想要最简单的方法**：使用方法 2（下载 ZIP）

**如果你想要最安全的方法**：使用方法 4（克隆到子目录然后移动）

**如果你经常需要这样做**：使用方法 5（创建脚本）

