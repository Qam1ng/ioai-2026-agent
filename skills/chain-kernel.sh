#!/usr/bin/env bash
# Chain-kernel 抢救法:读取已完成 kernel 的 submission.csv,做变换后重新输出。
# 50 秒完成,不重跑计算 —— 对比重跑攻击的 kernel 会撞 600s 超时被 CANCEL。
#
# 用法: skills/chain-kernel.sh <slug> <源kernel-ref> <变换名> [报告文件]
#   变换名: swap-ab   交换第2、3列(task-4 的 delta_a/delta_b 装反)
#           passthru  原样透传(仅用于补技术报告)
#
# 教训来源(2026-08-07 task-4): 前两发榜分精确 0.00,因为 delta_a/delta_b 接反。
# 用本法只交换两列重交 → 98.31。push 免费,不占提交额度。
set -u
cd ~/jy-agent
export PATH="$HOME/venvs/ioai312/bin:$PATH"
SLUG="${1:?slug}"; SRC="${2:?源 kernel ref, 如 hearsayagent/xxx}"; MODE="${3:?swap-ab|passthru}"
REPORT="${4:-}"
OUT="/data/qyan/ws/tmp/chain-$(date +%H%M%S)"
mkdir -p "$OUT"
KID="hearsayagent/chain-$(date +%H%M%S)"

{
  [ -n "$REPORT" ] && [ -f "$REPORT" ] && cat "$REPORT"
  cat <<'PY'
import csv, pathlib
csv.field_size_limit(10**8)
src = next(pathlib.Path("/kaggle/input").rglob("submission.csv"), None)
assert src is not None, "no submission.csv among kernel inputs"
rows = list(csv.reader(src.open()))
PY
  if [ "$MODE" = "swap-ab" ]; then
    echo 'rows = [rows[0]] + [[r[0], r[2], r[1]] for r in rows[1:]]'
  fi
  cat <<'PY'
out = pathlib.Path("/kaggle/working/submission.csv")
with out.open("w", newline="") as fh:
    csv.writer(fh).writerows(rows)
print("wrote", out, "rows:", len(rows) - 1)
PY
} > "$OUT/script.py"

cat > "$OUT/kernel-metadata.json" <<JSON
{
  "id": "$KID", "title": "$(basename $KID)",
  "code_file": "script.py", "language": "python", "kernel_type": "script",
  "is_private": true, "enable_gpu": false, "enable_internet": false,
  "machine_shape": "", "dataset_sources": [],
  "competition_sources": ["$SLUG"],
  "kernel_sources": ["$SRC"], "model_sources": []
}
JSON

echo "  推送 $KID (源: $SRC, 变换: $MODE)"
kaggle kernels push -p "$OUT" 2>&1 | tail -1 | sed 's/^/    /'
for i in $(seq 1 20); do
  s=$(kaggle kernels status "$KID" 2>&1 | grep -oE "KernelWorkerStatus\.[A-Z]+")
  echo "  [$i] $s $(date +%H:%M:%S)"
  if [ "$s" = "KernelWorkerStatus.COMPLETE" ]; then
    kaggle competitions submit "$SLUG" -k "$KID" -v 1 -f submission.csv \
      -m "chain-kernel $MODE from $(basename $SRC)" 2>&1 | tail -1 | sed 's/^/    提交: /'
    exit 0
  fi
  case "$s" in KernelWorkerStatus.ERROR|KernelWorkerStatus.CANCEL)
    echo "    停止: $s"; exit 1;; esac
  sleep 10
done
echo "    超时未完成"; exit 1
