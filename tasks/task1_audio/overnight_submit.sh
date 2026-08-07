#!/usr/bin/env bash
# Self-contained: poll kernel -> on COMPLETE submit its output -> read score.
# Pure Kaggle CLI (no Anthropic API). Records everything to result files so the
# agent only needs to wake ONCE (at exit) to report.
export PATH="/home/qing/miniconda3/bin:$PATH"
K="qam1ng/ioai-t1-submit"
SLUG="ioai-2026-ai-models-track-practice-task-1"
DIR="/home/qing/IOAI2026-agent/tasks/task1_audio"
R="$DIR/overnight_result.txt"
: > "$R"; : > "$DIR/overnight_progress.txt"

log(){ echo "$(date +%H:%M:%S) $*" >> "$R"; }

# 1) poll kernel up to ~90 min
FINAL=""
for i in $(seq 1 180); do
  s=$(kaggle kernels status "$K" 2>&1 | tail -1)
  echo "$(date +%H:%M:%S) $s" >> "$DIR/overnight_progress.txt"
  case "$s" in
    *RUNNING*|*QUEUED*) sleep 30 ;;
    *) FINAL="$s"; break ;;
  esac
done
log "kernel final: ${FINAL:-<timed out still running>}"

case "$FINAL" in
  *COMPLETE*)
    log "kernel COMPLETE -> submitting output via competition_submit_code"
    python - >> "$R" 2>&1 <<'PY'
from kaggle.api.kaggle_api_extended import KaggleApi
a = KaggleApi(); a.authenticate()
try:
    r = a.competition_submit_code(
        "submission.csv",
        "agent-loop best: aug feat + frozen-head logits + balanced LogReg (holdout 0.9156)",
        "ioai-2026-ai-models-track-practice-task-1",
        kernel="qam1ng/ioai-t1-submit", kernel_version=3)
    print("SUBMIT OK:", r)
except Exception as e:
    print("SUBMIT ERROR:", type(e).__name__, str(e)[:400])
    resp = getattr(e, "response", None)
    if resp is not None:
        print("BODY:", getattr(resp, "text", "")[:400])
PY
    # 2) poll for the score to appear
    for j in $(seq 1 40); do
      subs=$(kaggle competitions submissions -c "$SLUG" 2>&1)
      echo "$subs" > "$DIR/overnight_subs.txt"
      # stop once a numeric public score shows up
      if echo "$subs" | tail -n +3 | grep -qE '[0-9]+\.[0-9]+'; then break; fi
      sleep 20
    done
    log "=== submissions after scoring ==="
    kaggle competitions submissions -c "$SLUG" >> "$R" 2>&1
    ;;
  *ERROR*|*CANCEL*)
    log "kernel did NOT complete ($FINAL) -> not submitting; downloading log"
    kaggle kernels output "$K" -p "$DIR/kout_overnight" >> "$R" 2>&1
    ;;
  *)
    log "kernel still running after poll window; NOT submitting this pass"
    ;;
esac
log "overnight script DONE"
