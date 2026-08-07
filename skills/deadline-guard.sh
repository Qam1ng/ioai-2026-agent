#!/usr/bin/env bash
# Deadline 倒排:窗口末尾无条件手动交一发当前最好候选。
# 用法: skills/deadline-guard.sh <slug> <run-id> <deadline-HHMM> [margin-min]
# 教训(2026-08-07 t3): broker 因 kernel 自报估时(10min,实际20s)在 deadline
# 前拒发最后候选,靠人工手动补交才赶上。deadline 前 margin 分钟主动兜底。
set -u
cd ~/jy-agent
export PATH="$HOME/venvs/ioai312/bin:$PATH"
S="${1:?slug}"; RID="${2:?run-id}"; DL="${3:?deadline HHMM}"; MARGIN="${4:-6}"
W=$(ls -d workspace/final_system/*-"$RID" 2>/dev/null | head -1)
[ -n "$W" ] || { echo "  找不到 run $RID"; exit 1; }
# 计算触发时刻
now=$(date +%H%M); trig=$(printf "%04d" $((10#$DL - MARGIN)))
echo "  deadline=$DL  触发=$trig(前${MARGIN}min)  现在=$now"
if [ "$now" -lt "$trig" ]; then
  echo "  未到触发时刻。到 $trig 再运行本脚本(或用 watch)。"; exit 0
fi
# 已提交过的候选(避免重复)
subbed=$(~/venvs/ioai312/bin/python -c "
import json;print(' '.join(s.get('candidate_id','') for s in json.load(open('$W/control/broker_state.json')).get('submissions',[])))" 2>/dev/null)
# 挑本地分最高、带 kernel、未提交的候选
pick=$(~/venvs/ioai312/bin/python - "$W" "$subbed" <<'PY'
import json,glob,os,sys
W,subbed=sys.argv[1],set(sys.argv[2].split())
c=json.load(open(glob.glob(W+'/**/index.json',recursive=True)[0])).get('candidates',{})
best=None
for cid,r in c.items():
    if r.get('status')!='eligible' or cid in subbed: continue
    k=glob.glob(os.path.join(W,'control/candidates/snapshots',cid,'out/kernel/kernel-metadata.json'))
    if not k: continue
    m=(r.get('evaluation') or {}).get('mean') or 0
    if not best or m>best[0]: best=(m,cid,os.path.dirname(k[0]))
if best: print(f"{best[2]}\t{best[1]}\t{best[0]}")
PY
)
[ -z "$pick" ] && { echo "  无未提交的已打包候选(broker 可能已交完)"; exit 0; }
KDIR=$(echo "$pick"|cut -f1); CID=$(echo "$pick"|cut -f2); SC=$(echo "$pick"|cut -f3)
echo "  兜底候选: $CID (local $SC)"
SLUG="hearsayagent/deadline-guard-$(date +%H%M)"
~/venvs/ioai312/bin/python -c "
import json;p='$KDIR/kernel-metadata.json';d=json.load(open(p))
d['id']='$SLUG';d['title']='$SLUG'.split('/')[-1];json.dump(d,open(p,'w'),indent=2)"
kaggle kernels push -p "$KDIR" 2>&1 | tail -1 | sed 's/^/    /'
for i in $(seq 1 10); do
  st=$(kaggle kernels status "$SLUG" 2>&1 | grep -oE "KernelWorkerStatus\.[A-Z]+")
  [ "$st" = "KernelWorkerStatus.COMPLETE" ] && break
  [ "$st" = "KernelWorkerStatus.ERROR" ] && { echo "    kernel ERROR"; exit 1; }
  sleep 8
done
kaggle competitions submit "$S" -k "$SLUG" -v 1 -f submission.csv -m "deadline-guard: $CID local $SC" 2>&1 | tail -1 | sed 's/^/    提交: /'
