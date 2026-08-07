#!/usr/bin/env bash
# 标准启动。用法: skills/launch.sh <slug> <run-id> <minutes> <deadline> [--live]
#
# <minutes>  0 = 一直跑到喊停（推荐）。停止: skills/stop.sh <run-id>
# <deadline> 比赛真实截止（HH:MM / ISO / epoch），"-" 表示没有截止
#
# 例:  skills/launch.sh ioai-2026-task-6-xxx t6run1 0 12:00 --live   # 跑到 12:00 或喊停
#      skills/launch.sh ioai-2026-task-6-xxx t6run1 0 -     --live   # 只由你喊停
#
# <deadline> 是**比赛真实截止**（HH:MM / ISO / epoch），不是本轮窗口。必填。
# 缺了它，所有时间闸门都会退回按 run 窗口计算 —— task 5 就是这样让 Broker
# 提前 49 分钟烧光 final 预留然后彻底哑火的。
#
# 续跑同一道题时用 FLOOR_GROUP 复用当天的 floor 组，避免重复交 floor：
#   FLOOR_GROUP=t5day skills/launch.sh <slug> t5run2 30 12:00 --live
set -eu
cd ~/jy-agent
S="${1:?slug}"; RID="${2:?run-id}"; MIN="${3:?minutes}"
DL="${4:?deadline 真实截止(无则填 -)}"; LIVE="${5:-}"
[ "$DL" = "-" ] && DL=""
FG="${FLOOR_GROUP:-${RID}day}"
set -a; . ./.env; set +a
export PATH="$HOME/venvs/ioai312/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export PYTHONUSERBASE=/data/qyan/ws/pyuserbase PIP_CACHE_DIR=/data/qyan/ws/pipcache TMPDIR=/data/qyan/ws/tmp
for f in official_prompts/starter_$S.txt official_prompts/continuation_$S.txt; do
  [ -f "$f" ] || { echo "缺 $f — 先跑 preflight"; exit 1; }
done
[ -d "/data/qyan/ioai/$S" ] || { echo "数据未下载 — 先跑 preflight"; exit 1; }
if ls -d workspace/final_system/*-"$RID" >/dev/null 2>&1; then
  echo "⚠ run-id=$RID 的工作区已存在 — 同名重启会覆盖历史。换个 run-id,或确认后手动删。"; exit 1
fi
nohup ~/venvs/ioai312/bin/python -m final_system run \
  --slug "$S" --assets-dir "/data/qyan/ioai/$S" \
  --duration-minutes "$MIN" --competition-mode practice \
  --run-id "$RID" --kaggle-user hearsayagent \
  --resource-pool-id hearsayagent --floor-group-id "$FG" --day-slugs "$S" \
  ${DL:+--deadline "$DL"} \
  --starter-prompt-file "official_prompts/starter_$S.txt" \
  --continuation-prompt-file "official_prompts/continuation_$S.txt" \
  $LIVE > "fs_$RID.log" 2>&1 &
if [ "$MIN" = "0" ]; then MODE="无限(跑到 skills/stop.sh $RID)"; else MODE="${MIN}min 窗口"; fi
echo "启动 pid=$!  $RID  $MODE  deadline=${DL:-无}  floor-group=$FG  ${LIVE:-dry-run}"
echo "监控: ./fswatch $RID"
