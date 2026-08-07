#!/usr/bin/env bash
# 赛前检查清单。用法: skills/preflight.sh <competition-slug>
# 全部通过才能启动;任何 ❌ 都要先解决。
set -u
cd ~/jy-agent || exit 1
S="${1:?用法: preflight.sh <slug>}"
set -a; . ./.env; set +a
export PATH="$HOME/venvs/ioai312/bin:$PATH"
PY=~/venvs/ioai312/bin/python
ok(){ echo "  ✅ $*"; }; bad(){ echo "  ❌ $*"; FAIL=1; }
FAIL=0
echo "══ 1. 认证 ══"
c=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  "https://prosgrow-aoai-prod.services.ai.azure.com/anthropic/v1/messages" \
  -H "x-api-key: $AZURE_OPENAI_API_KEY" -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-opus-5","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}')
[ "$c" = 200 ] && ok "Azure /anthropic (opus-5): $c" || bad "Azure /anthropic: $c (注意要用 x-api-key 头,不是 api-key)"
c=$(curl -s -o /dev/null -w "%{http_code}" \
  "https://prosgrow-aoai-prod.services.ai.azure.com/openai/models?api-version=2024-10-21" \
  -H "api-key: $AZURE_OPENAI_API_KEY")
[ "$c" = 200 ] && ok "Azure /openai: $c" || bad "Azure /openai: $c"
kaggle competitions submissions -c "$S" >/dev/null 2>&1 && ok "Kaggle CLI 认证+比赛可访问" || bad "Kaggle: 检查 ~/.kaggle/access_token 或是否已在浏览器接受规则"
echo "══ 2. 数据与 prompt ══"
D="/data/qyan/ioai/$S"
[ -d "$D" ] && ok "数据: $D ($(du -sh $D 2>/dev/null|cut -f1))" || bad "数据未下载: kaggle competitions download -c $S -p $D"
for f in starter continuation; do
  P="official_prompts/${f}_$S.txt"
  if [ -f "$P" ]; then
    grep -q "COMPETITION-SLUG" "$P" && bad "$P 有未替换的占位符" || ok "$P"
  else bad "缺 $P(从题目 Overview 逐字复制,替换 slug)"; fi
done
echo "══ 3. 题目参数(每题必查!上题的值≠这题的值)══"
grep -E "^max_submissions|^kernel_timeout_seconds" configs/final_agent_system.toml | sed "s/^/  ⚠ 人工核对: /"
echo "  ⚠ 对照题目 Overview: 版本额度(20?50?每日?)和 --timeout(300?600?1800?)"
echo "== 3.5 子系统开关(调试后常忘恢复!) =="
for sec in search hearsay selection; do
  v=$(grep -A2 "^\[$sec\]" configs/final_agent_system.toml | grep "^enabled" | head -1)
  echo "  ⚠ [$sec] ${v:-enabled=(默认true)}"
done
echo "══ 4. 环境 ══"
$PY -c "
import sys; sys.path.insert(0,'.')
from pathlib import Path
from final_system.config import SystemConfig
c=SystemConfig.load(Path('configs/final_agent_system.toml'), repo_root=Path('.'))
c.validate(); print('  ✅ 配置 validate() 通过')" || bad "配置加载失败"
n=$(pgrep -u "$USER" -f "final_system run" | wc -l)
[ "$n" -le 1 ] && ok "无残留 run" || bad "有 $n 个 final_system run 在跑,先停掉或确认"
df -h /data | awk 'NR==2 {gsub("%","",$5); if ($5+0>97) print "  ❌ /data 已用 "$5"%"; else print "  ✅ /data 可用 "$4}'
busy=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | awk -F", " '$2+0>50 {print $1}' | tr '\n' ' ')
[ -z "$busy" ] && ok "GPU 全部空闲" || echo "  ⚠ GPU $busy 忙(别人的任务?避开)"
echo
[ "$FAIL" = 0 ] && echo "══ 全部通过,可以启动 ══" || echo "══ 有 ❌,先解决 ══"
# 教训 2026-08-06: XINTONG 调试时把 search/hearsay 关了没恢复,dry run 白跑 10 分钟。
