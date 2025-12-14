#!/bin/bash
# 从 GitHub 仓库克隆内容（不保留 git 历史）
# 用法: ./clone_without_git.sh <仓库URL> [目标目录]

REPO_URL=$1
TARGET_DIR=${2:-.}

if [ -z "$REPO_URL" ]; then
    echo "用法: $0 <仓库URL> [目标目录]"
    echo "示例: $0 https://github.com/user/repo.git"
    echo "示例: $0 git@github.com:user/repo.git /path/to/target"
    exit 1
fi

# 创建临时目录
TEMP_DIR=$(mktemp -d)
echo "📥 克隆仓库到临时目录: $TEMP_DIR"

# 克隆（只克隆最新版本，不包含历史）
git clone --depth 1 "$REPO_URL" "$TEMP_DIR" || {
    echo "❌ 克隆失败，请检查仓库 URL"
    rm -rf "$TEMP_DIR"
    exit 1
}

# 复制文件（排除 .git 文件夹）
echo "📋 复制文件到: $TARGET_DIR"
rsync -av --exclude='.git' "$TEMP_DIR/" "$TARGET_DIR/" || {
    echo "❌ 复制失败"
    rm -rf "$TEMP_DIR"
    exit 1
}

# 清理临时目录
rm -rf "$TEMP_DIR"
echo "✅ 完成！文件已复制到 $TARGET_DIR（不包含 .git 历史）"

