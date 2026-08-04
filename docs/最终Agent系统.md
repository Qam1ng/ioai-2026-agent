# IOAI 最终 Agent System

## 结论

系统保留三条故障模式不同的路线，并只在三个机械接口汇合：只读官方资产、公共
验证合同、Submission Broker。Search 的知识不会进入两条直跑线，排行榜反馈也只
返回产生该候选的路线。

```text
                         官方资产（只读 + SHA-256）
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
      Search（分析+4研究）   Codex + GPT-5.6 Sol   Claude Code + Fable 5
              │             独立直跑，不读 Search   独立直跑，不读 Search
       verified Search Bundle     │                   │
              │                   │                   │
       HearSay evaluator + 3 solvers                 │
              │                   │                   │
              └──────── candidate package ───────────┘
                                  │
                 公共 metric.py + folds.json 重评分
                                  │
                    唯一 Submission Broker（脚本）
                  floor / calibration / milestone / final
                                  │
                               Kaggle
```

模块类型：Search Analyst/Research、HearSay evaluator/solver、Codex 和 Claude 直跑
都是 Agent；资产快照、哈希校验、候选快照、公共重评分、额度状态机和 Kaggle 提交
均为确定性脚本。

## 关键边界

- Search 完成后才启动 HearSay；Search `complete/degraded` 且四个必需 MD 与资产哈希
  完整时才传 Bundle，否则 HearSay 只拿原始资产。
- Codex、Claude 在 Search 做研究时就启动。它们只看到自己的 workdir、官方资产、
  公共验证目录及自己路线的 `FEEDBACK.jsonl`。
- HearSay 内部 submitter 与 score watcher 在 `--external-broker-dir` 模式下不启动，
  其 Kaggle 工具也从 MCP 表中删除。三条 Agent 进程使用隔离 HOME，不含 Kaggle
  凭证；只有 controller/Broker 保留 operator 环境。
- `claimed_local_score` 永不参与选择。Broker 只读取 `out/oof.npy`，用第一次冻结的
  `metric.py + folds.json` 重算分数，并把 contract SHA-256 写入证据。
- 候选以内容指纹做不可变快照。同一 submission 哈希只注册一次；Agent 在 READY
  之后改文件会形成新候选，不会偷偷改变已经测过/提交过的候选。
- Kernel push/poll 在最多两个后台槽中运行，不会阻塞新候选收集、公共重评分或反馈
  轮询。Broker 在外部调用前原子预留额度；并发调用不能重复选候选或超出 50 次。

## 50 次提交策略

- 1 次 `floor`：尽早确认整条提交链有效。
- 最多 3 次 `calibration`：每条来源路线最多先拿一个公榜锚点。
- 中间额度为动态 `milestone`：必须比已提交候选的公共本地分数更好；前半场若其他
  路线已有合格候选，单一来源不得占 milestone 的 50% 以上。
- 8 次 `final/recovery` 保留到末段。未使用额度不是必须花完；没有新的不可重复候选
  时，Broker 不会为了凑数重复提交。

`source_lane`（hearsay/codex/claude）与 `submission_class`
（floor/calibration/milestone/final）分开记录，避免把来源和提交目的混为一谈。

## 启动

建议给完整主线与独立 Claude 兜底线各用一个 Max profile（只有一个账号时也可让
两个配置项指向同一路径），并在 shell 中提供 OpenRouter key：

```bash
CLAUDE_CONFIG_DIR="$HOME/.claude-ioai-integrated" claude /login
CLAUDE_CONFIG_DIR="$HOME/.claude-ioai-direct" claude /login
export OPENROUTER_API_KEY='...'
python -m final_system doctor \
  --config configs/final_agent_system.toml \
  --assets-dir /absolute/path/to/official_assets
```

离线排练（不会调用 Kaggle submit）：

```bash
python -m final_system run \
  --config configs/final_agent_system.toml \
  --slug competition-slug \
  --assets-dir /absolute/path/to/official_assets \
  --duration-minutes 30 \
  --competition-mode practice \
  --run-id rehearsal-01
```

正式比赛必须显式增加 `--live`。启动时如果 CLI 不能读取 Kaggle
`Remaining today`，系统 fail closed，不会猜一个新的 50 次额度：

```bash
python -m final_system run \
  --config configs/final_agent_system.toml \
  --slug competition-slug \
  --assets-dir /absolute/path/to/official_assets \
  --duration-minutes 360 \
  --competition-mode formal \
  --kaggle-user YOUR_KAGGLE_USER \
  --run-id day1-task1 \
  --live
```

## 轨迹与恢复

一个 session 下会保存：

```text
search/*/trajectories/                  Search 全部 Agent JSONL
control/trajectories/codex/             Codex 每轮 JSONL
control/trajectories/claude/            Claude 每轮 JSONL
control/hearsay.stdout.log              HearSay 总日志
lanes/hearsay/trace.jsonl               HearSay harness 轨迹
lanes/hearsay/facts.jsonl               HearSay 事实板
control/candidates/snapshots/            所有不可变候选
control/broker_state.json                额度与提交恢复状态
control/broker_events.jsonl              Broker 决策审计
control/feedback/<source>.jsonl          按来源隔离的榜单反馈
RUN_STATUS.json                          最终摘要
```

Broker 会在外部调用前先写 `reserved`；网络调用结果不确定时保留为 `ambiguous`。
这两种记录都不会被自动重试，从而避免“提交已成功但本地来不及落盘”造成重复消耗；
需要先按 `[fas:<submission_id>]` 在 Kaggle submissions 中人工/脚本核对，再处理。
这里的持久化首先是审计和防重复保障；当前 CLI 不提供对整个既有 session 的原地
自动恢复，不能把“状态文件还在”理解为可以无检查地重跑同一个 `--run-id`。

账号切换不是在一个活跃 Claude session 中热替换。可在任务之间停止该路线，把
`integrated_profile_dir` 或 `direct_profile_dir` 可在题目之间指向预先登录的另一个
目录。若比赛中途强行换号，Claude 的推理上下文不能跨账号恢复；当前版本也没有热
切换入口，只能从已落盘 artifact 人工启动一个新的 recovery session。因此应把它当
最后手段，而不是常规调度。
