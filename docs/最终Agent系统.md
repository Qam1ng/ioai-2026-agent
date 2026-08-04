# IOAI 最终 Agent System

## 结论

每道题保留三条故障模式不同的路线，并只在三个机械接口汇合：只读官方资产、公共
验证合同、Submission Broker。第一天三道题同时开放时，启动三套相互独立的题目
Controller；三套系统只在账号级 `DayResourceGate` 汇合。每题仍有自己的 6 小时、
50 次提交和 Selection Manager，只有每周 30 GPU-hours、2 路 GPU/5 路 CPU 并发
跨两天共享；同一天三题另外组成一个 floor group。

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
       公共 metric.py + folds.json + tests + evidence 重评分
                                  │
                 Public-LB 校准器（版本化脚本）◄─────┐
                 E0 冻结；留一排序改善才激活          │
                                  │                   │
              Selection Manager（Claude Code Agent） │
                    Fable 5 / high，只给建议          │
                                  │ candidate / wait  │
                    唯一 Submission Broker（脚本）    │
              校验建议 + 额度策略 + 超时确定性兜底    │
                                  │                   │
                    账号级 DayResourceGate（脚本）   │
            当日三题 floor 优先 + 跨两天并发/30h     │
                                  │                   │
                               Kaggle ────────────────┘
```

模块类型：Search Analyst/Research、HearSay evaluator/solver、Codex 和 Claude 直跑
以及 Selection Manager 都是 Agent；资产快照、哈希校验、候选快照、公共重评分、
Public-LB 校准、额度状态机和 Kaggle 提交均为确定性脚本。Selection Manager 没有
提交权；它的 Claude Code 进程只开放 `Read`，且进程环境不含 Kaggle 凭证。

## 关键边界

- Search 完成后才启动 HearSay；Search `complete/degraded` 且四个必需 MD 与资产哈希
  完整时才传 Bundle，否则 HearSay 只拿原始资产。
- Codex、Claude 在 Search 做研究时就启动。它们只看到自己的 workdir、官方资产、
  公共验证目录及自己路线的 `FEEDBACK.jsonl`。
- HearSay 内部 submitter 与 score watcher 在 `--external-broker-dir` 模式下不启动，
  其 Kaggle 工具也从 MCP 表中删除。隔离 HOME 之外，所有 Agent 及其 Bash 子进程
  还继承 macOS Seatbelt：不能读取 operator 的 Kaggle 凭证文件，也不能执行 Kaggle
  CLI。当前 workspace 的其他路线/control plane 默认不可读，整个仓库、资源池和
  原始资产默认不可写；每个 Agent 只重新开放自己的工作目录和声明过的只读接口。
  Claude/Codex 原生 CLI 会在启动前做实机探针；正式
  live 模式探针失败即停止。只有 controller/Broker 保留 operator 环境。
- `claimed_local_score` 永不参与选择。第一次冻结的公共合同必须同时包含
  `metric.py`、`folds.json`、可重放的 `metric_tests.json` 和来源/方向证据
  `contract_evidence.json`；缺少任何一项或已知答案测试不通过都不启用。HearSay
  evaluator 迟迟失败时，独立 fallback evaluator 只读原始资产生成同一契约。默认
  `metric_direction=auto`，由该证据冻结本题 maximize/minimize，避免三题共用一个
  “越大越好”的错误假设。
- Public-LB 校准层不会修改 E0。至少积累 4 个可配对分数后，它以
  `mean/std/pooled/per-fold` 拟合带正则的校准器；只有留一预测的 Spearman 排序比
  原始 E0 至少改善配置门槛时才激活，而且校准权重上限为 0.5。未通过门禁时排序
  完全退回 E0；通过后只调整排序与 milestone 判断，原始 E0 证据永久保留。
- Selection Manager 只会看到 Broker 已按额度、阶段、格式和反垄断规则过滤后的
  候选。它输出带输入 SHA-256 和过期时间的 `submit/wait` 建议；Broker 再次校验候选
  是否仍然 eligible。模型不匹配、超时、非法 JSON、陈旧输入或无建议超过等待窗口
  时，Broker 使用同一 policy 集合中的确定性最高项，不让 Agent 直接调用 Kaggle。
- 候选以内容指纹做不可变快照。同一实际执行 artifact 哈希只注册一次；Agent 在 READY
  之后改文件会形成新候选，不会偷偷改变已经测过/提交过的候选。
- Kernel 候选必须是 Kaggle 真正会上传的独立单文件；额外 helper 会在本地拒绝，
  metadata 的 dataset/kernel/model sources 会原样保留。远端完成后，Broker 下载并按
  官方 submission 模板校验真实 `submission.csv`，再提交精确 kernel version。
- 每题可有两个 Broker 后台调用，但三个进程共用账号级资源锁。三题先各拿到一个
  floor，之后才放行 milestone；两天所有 GPU lease 合计不超过 2 并发/30 小时，CPU 不
  超过 5 并发。候选的时长声明只能增加、不能低于保守默认值；来不及在截止前完成的
  Kernel 不启动。远端超出本地轮询窗口时保留 lease，并持续对账直至恢复或终态。

## 50 次提交策略

- 1 次 `floor`：尽早确认整条提交链有效。
- 最多 3 次 `calibration`：每条来源路线最多先拿一个公榜锚点。
- 中间额度为动态 `milestone`：必须比已提交候选的公共本地分数更好；前半场若其他
  路线已有合格候选，单一来源不得占 milestone 的 50% 以上。
- 8 次 `final/recovery` 保留到末段。未使用额度不是必须花完；没有新的不可重复候选
  时，Broker 不会为了凑数重复提交。

这些规则先由脚本生成 policy-eligible 集合，Selection Manager 只在该集合内排序或
建议等待。最终“交哪个”的职责分为两层：Agent 做证据判断，Broker 做权限校验、
额度原子预留和真实提交。即使 Agent 故障，Broker 也会在配置的等待窗口后回退。

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

30 分钟离线 smoke test（不会调用 Kaggle submit）。短窗口会把 Search 限制到总时长
的 25%，并及时启动 fallback evaluator，因此可以覆盖候选注册、公共契约和 Broker；
它不是 6 小时吞吐量测试：

```bash
python -m final_system run \
  --config configs/final_agent_system.toml \
  --slug task1-slug \
  --assets-dir /absolute/path/to/task1-official-assets \
  --duration-minutes 30 \
  --competition-mode practice \
  --run-id rehearsal-01
```

正式比赛必须显式增加 `--live`。三道题同时出现后，在三个终端各启动一套；三条
命令的 `--resource-pool-id`、`--floor-group-id` 和 `--day-slugs` 必须完全相同。
第二天继续使用同一个 resource pool（Kaggle GPU 额度按周累计），但换成 day2 floor
group 和第二天三题 slugs。启动时如果 CLI 不能
读取该题 Kaggle `Remaining today`，系统 fail closed，不会猜一个新的 50 次额度。
以下是 task1，task2/task3 只替换 `--slug`、资产目录和 `--run-id`：

```bash
python -m final_system run \
  --config configs/final_agent_system.toml \
  --slug task1-slug \
  --assets-dir /absolute/path/to/task1-official-assets \
  --duration-minutes 360 \
  --competition-mode formal \
  --kaggle-user YOUR_KAGGLE_USER \
  --run-id day1-task1 \
  --resource-pool-id ioai-2026-event-account1 \
  --floor-group-id ioai-2026-day1 \
  --day-slugs task1-slug,task2-slug,task3-slug \
  --live
```

不要让三题各自创建不同 resource pool，也不要在第二天换 pool；前者会重新变成
最多 6 个 GPU Kernel 并发，后者会把同一份 30h/week 误记成 60h。只有
`floor-group-id` 在第二天更换。

## 轨迹与恢复

一个 session 下会保存：

```text
search/*/trajectories/                  Search 全部 Agent JSONL
control/trajectories/codex/             Codex 每轮 JSONL
control/trajectories/claude/            Claude 每轮 JSONL
control/trajectories/selection_manager/  Selection Manager 每次 Claude Code JSONL
control/hearsay.stdout.log              HearSay 总日志
lanes/hearsay/trace.jsonl               HearSay harness 轨迹
lanes/hearsay/facts.jsonl               HearSay 事实板
control/candidates/snapshots/            所有不可变候选
control/calibration/latest.json          当前版本化 Public-LB 校准状态
control/calibration/history/              每个校准版本的不可变记录
control/selection_manager/inputs/         Manager 每次完整输入
control/selection_manager/history/        SHA 绑定的有效建议历史
control/selection_manager/status.json     当前模型、状态与失败原因
control/broker_state.json                额度与提交恢复状态
control/broker_events.jsonl              Broker 决策审计
control/feedback/global.jsonl            全局完整提交反馈，供控制平面审计
control/feedback/<source>.jsonl          路线自己的完整反馈 + 校准摘要
RUN_STATUS.json                          最终摘要
workspace/final_system/_resource_pools/<pool>/resource_state.json
                                         六题共享 lease/GPU-hours 与逐日 floor 状态
```

每次提交会先产生 `submission_result`，公榜可读后再产生 `leaderboard_score`。反馈包含
candidate/submission ID、用途与 parent、完整 E0 mean/std/pooled/per-fold、contract
哈希、执行状态、Public 分数、Local-Public residual、校准版本/预测、选择原因和剩余
额度。Codex、Claude 通过各自的 `FEEDBACK.jsonl` 获取；HearSay 的每个 solver 也会
逐条收到全部未读的 HearSay 路线反馈，不再只截取末尾若干行。Search 完成后自行
结束，因此不会被重新唤醒；跨路线完整历史只进入 Selection Manager，不互相污染
三条独立探索线。

Broker 会在外部调用前先写 `reserved`；已确认未消费额度的 push/mount 瞬时失败最多
退避重试 3 次；格式、运行和截止错误不重试。提交 API 结果不确定时保留为
`ambiguous`，按 `[fas:<submission_id>]` 在 Kaggle submissions 自动对账，确认存在
后才计为已消费。仍在 Kaggle 运行的 Kernel 记为 `external_running`，不会重复 push，
终态后继续下载产物并提交原精确版本；即使原 controller 崩溃，其他两题也会释放其
已终止的共享资源 lease。
这里的持久化首先是审计和防重复保障；当前 CLI 不提供对整个既有 session 的原地
自动恢复，不能把“状态文件还在”理解为可以无检查地重跑同一个 `--run-id`。

账号切换不是在一个活跃 Claude session 中热替换。可在任务之间停止该路线，把
`integrated_profile_dir` 或 `direct_profile_dir` 可在题目之间指向预先登录的另一个
目录。若比赛中途强行换号，Claude 的推理上下文不能跨账号恢复；当前版本也没有热
切换入口，只能从已落盘 artifact 人工启动一个新的 recovery session。因此应把它当
最后手段，而不是常规调度。
