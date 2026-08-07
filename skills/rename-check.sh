#!/usr/bin/env bash
# 改名清零。用法: skills/rename-check.sh <旧变量名> [旧变量名2 ...]
# 教训(2026-08-07): 把 secondary_codex_runner 改名,3 处引用只改了 2 处,
# 漏的那处在探索期结束的清理代码里,正赛崩溃。改名后必须全库清零。
set -u
cd ~/jy-agent
[ $# -ge 1 ] || { echo "  用法: rename-check.sh <旧名> ..."; exit 1; }
fail=0
for name in "$@"; do
  hits=$(grep -rn "$name" final_system/ native/ swarm/ --include=*.py 2>/dev/null \
         | grep -v "def _" | grep -v "# ")
  n=$(echo "$hits" | grep -c "$name")
  if [ -n "$hits" ]; then
    echo "  ❌ '$name' 仍有 $n 处引用:"
    echo "$hits" | sed 's/^/     /'
    fail=1
  else
    echo "  ✅ '$name' 已清零"
  fi
done
[ "$fail" = 0 ] && echo "  全部清零,可提交" || { echo "  有残留,先改完"; exit 1; }
