#!/usr/bin/env bash
# 提交归因 + 额度账单。用法: skills/audit.sh <slug>
# 指纹规则: kernel 源码头部有 "Injected by the submission broker" = 我们 broker 推的;
#           提交描述带 [fas:...] = 我们 broker 交的;其余 = agent 自交或其他系统。
set -u
cd ~/jy-agent; set -a; . ./.env; set +a
export PATH="$HOME/venvs/ioai312/bin:$PATH"
S="${1:?slug}"
echo "══ 提交(权威)══"
kaggle competitions submissions -c "$S" --csv 2>/dev/null | ~/venvs/ioai312/bin/python -c "
import csv,sys
rows=list(csv.DictReader(sys.stdin))
print(f'  共 {len(rows)} 次')
for r in rows:
    d=r.get('description','') or ''
    tag='broker' if '[fas:' in d else '自交/他系统'
    print(f\"  {r.get('publicScore') or '-':>9}  {(r.get('date') or '')[11:19]}  {tag:12} {d[:60]}\")"
echo "══ kernel 指纹(最近 12 个)══"
T=$(mktemp -d)
kaggle kernels list --user hearsayagent --csv --page-size 12 2>/dev/null | tail -n +2 | cut -d, -f1 | while read ref; do
  n=$(basename "$ref"); mkdir -p "$T/$n"
  kaggle kernels pull "$ref" -p "$T/$n" >/dev/null 2>&1
  f=$(ls "$T/$n"/*.py 2>/dev/null | head -1)
  [ -z "$f" ] && { echo "  ?          $n"; continue; }
  if head -c 1200 "$f" | grep -q "Injected by the submission broker"; then echo "  ✅ broker   $n"
  else echo "  ❌ 非broker  $n"; fi
done
rm -rf "$T"
