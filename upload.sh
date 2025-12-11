#!/bin/bash
# 快速上传脚本
# 使用方法: ./upload.sh "commit message"

# 检查是否有未提交的更改
if [ -z "$(git status --porcelain)" ]; then
    echo "No changes to commit"
    exit 0
fi

# 获取提交信息
COMMIT_MSG=${1:-"Update code"}

# 添加所有更改
git add -A

# 提交
git commit -m "$COMMIT_MSG"

# 推送
git push

echo "Upload complete!"

