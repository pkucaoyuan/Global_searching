@echo off
REM Windows快速上传脚本
REM 使用方法: upload.bat "commit message"

REM 检查参数
if "%1"=="" (
    set COMMIT_MSG=Update code
) else (
    set COMMIT_MSG=%1
)

REM 添加所有更改
git add -A

REM 提交
git commit -m "%COMMIT_MSG%"

REM 推送
git push

echo Upload complete!




