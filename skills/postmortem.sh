#!/usr/bin/env bash
# 每轮结束验尸。用法: skills/postmortem.sh <run-id>
# 教训(2026-08-07 t3run2): NameError 让控制器在探索期结束时崩溃,而它昨天
# dryrun6 就爆过一次 —— 没验尸日志,漏看了一整天。"进程不在了" ≠ "正常收尾"。
set -u
cd ~/jy-agent
RID="${1:?用法: postmortem.sh <run-id>}"
LOG="fs_${RID}.log"
[ -f "$LOG" ] || { echo "  找不到 $LOG"; exit 1; }
echo "══ 验尸 $RID ══"
tb=$(grep -c "Traceback" "$LOG" 2>/dev/null)
if [ "$tb" -gt 0 ]; then
  echo "  ⚠ 发现 $tb 处 Traceback。最后一处末行(真死因):"
  grep -A30 "Traceback" "$LOG" | grep -E "Error:|Error$|Exception" | tail -1 | sed 's/^/    /'
  echo "  → 该 run 不是正常收尾。修复后再判断是否需要续跑。"
else
  echo "  ✅ 无 Traceback"
fi
# 控制器是否走到了正常收尾事件
W=$(ls -d workspace/final_system/*-"$RID" 2>/dev/null | head -1)
if [ -n "$W" ]; then
  last=$(tail -1 "$W/control/controller_events.jsonl" 2>/dev/null | grep -oE '"event": *"[^"]+"' | cut -d'"' -f4)
  echo "  最后事件: ${last:-无}"
fi
