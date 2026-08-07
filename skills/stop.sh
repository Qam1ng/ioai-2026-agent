#!/usr/bin/env bash
# 优雅停止一个无限模式的 run。用法: skills/stop.sh <run-id>
# 控制器会停掉各 lane、收完在飞的提交、写完 RUN_STATUS 再退出。
set -eu
cd ~/jy-agent
RID="${1:?run-id}"
W=$(ls -d workspace/final_system/*-"$RID" 2>/dev/null | head -1) || true
[ -n "${W:-}" ] || { echo "找不到 run-id=$RID 的工作区"; exit 1; }
touch "$W/control/STOP"
echo "已请求停止 $RID — 控制器最多 2 秒内响应；用 ./fswatch $RID 看收尾"
