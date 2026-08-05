# 保底 Agent System

`final_system` 的备胎。存在的意义是：比赛当天主系统出问题时，代价是丢名次，不是丢一整天。

设计原则是**少**。没有 Search 模块、没有冻结评估合同、没有校准层、没有跨进程资源池、
没有沙箱。三道题跑在**一个进程**里，共用一把 `asyncio.Semaphore(2)`（Kaggle 账号级
并发上限）。全部代码约 1200 行。

```
                        一个进程
   ┌──────────────┬──────────────┬──────────────┐
   │   题目 1     │   题目 2     │   题目 3     │
   │              │              │              │
   │  codex ─┐    │  codex ─┐    │  codex ─┐    │   ← 共用同一个 work/ 目录
   │         ├─→ candidates/ ─→ select manager  │
   │  claude ┘    │  claude ┘    │  claude ┘    │
   │              │              │              │
   │      board.py（额度/重试/榜分，无模型）     │
   └──────┬───────┴──────┬───────┴──────┬───────┘
          └──────── Semaphore(2) ───────┘
                        │
                     Kaggle
```

## 一次说清的几件事

**两个 agent 共用一个工作目录。** 这是刻意的：对方写出来的东西是你的先验材料。
为了不互相覆盖，约定各自的草稿放 `<agent>/` 子目录，共享结论追加到 `NOTES.md`。

**Agent 不提交。** 它们没有 Kaggle 凭证（`_ENV_ALLOWLIST` 里没有任何 `KAGGLE_*`，
`HOME` 也被改写掉，所以 `~/.kaggle/kaggle.json` 也发现不了）。它们只发布候选。

**Manager 没有工具。** `--tools ""`。它需要的信息全在 prompt 里；一个能跑 Bash 的
manager 就是一个能自己想办法提交的 manager。

**榜分是唯一裁判。** 系统不建本地评估合同。候选可以自报分数，但那只是笔记，
Broker 排序时不看。这是因为这类题上本地 CV 和榜分实测可能反相关。

**第一次提交不问任何人。** 直接把最新候选发出去，把整条链路验证掉——这时候也没什么
可选的。8 分钟还没成功提交过，watchdog 会排一个"原样复制官方模板"的保底候选。

**瞬时失败会重试。** push 失败、产物下载失败、`/kaggle/input` 挂载抖动 → 退避重试，
最多 3 次。格式错误和 kernel 执行错误不重试（重试也不会变好）。

## 跑起来

API key 放在仓库根目录的 `.env`（已 gitignore，系统启动时自动读取）：

```
IOAI_LLM_API_KEY=<你的 key>
```

先体检——它会对三个端点各发一个真实请求：

```bash
python -m backup_system doctor
```

### 正式赛：唯一入口是 Starter prompt

按 IOAI 规则，操作员只被允许做一件事——**原样粘贴官方 Starter prompt**。所以正式
入口就长这样，把每道题的 Starter prompt 存成一个文件（或用 `-` 走 stdin）：

```bash
python -m backup_system run \
  --starter-prompt /path/task1_starter.txt \
  --starter-prompt /path/task2_starter.txt \
  --starter-prompt /path/task3_starter.txt \
  --kaggle-user YOUR_KAGGLE_USER \
  --live
```

slug、kernel 时限、截止时间全部由系统自己从文本里读，数据由系统自己下载。**不需要
`--assets-dir`，不需要 `--slug`，不需要 `--minutes`**（题面里没有绝对截止时间时才
用 `--minutes` 兜底，届时会在日志里标成 `deadline_source=operator_minutes`）。

实测一遍（真 Starter prompt + 真 Kaggle 校验）：

```
slug                     ioai-2026-ai-models-track-practice-task-1   ← 从散文里的 URL 抠出
kernel_timeout_seconds   1800                                        ← "30 minutes" 换算
deadline_source          operator_minutes                            ← 题面只说 "before the round closes"
max_submissions          50                                          ← 来自 Kaggle，不信题面
starter prompt 逐字节保留 True
```

### 离线排练

```bash
python -m backup_system run --task "slug=/abs/path/assets" \
  --kernel-timeout-seconds 1800 --minutes 30
```

`--task` 明确只用于排练：它接受一个别人准备好的数据目录，而规则把下载数据划给
agent，所以 `--task` 配 `--live` 会被直接拒绝。

## 规则合规

| IOAI 规则 | 本系统 |
|---|---|
| 操作员只粘贴 Starter prompt | ✅ `--starter-prompt`，其余全部自解析 |
| agent 自己下载数据 | ✅ `Kaggle.download_data()`，`--live` 时开跑前执行 |
| push 必须带 `--timeout` | ✅ 从题面读出的秒数传给 `kernels push --timeout`；实测 Kaggle 接受 `--timeout 0`，所以非正值在本地就被拒 |
| kernel 跑完后再提交 | ✅ 下载产物→对模板校验→`competition_submit_code` pin 精确 version |
| 提交代码顶部要有 Report | ✅ Starter prompt 原文透传给 solver，该要求随原文到达 |
| 不得 hack 提交系统 / 拿隐藏标签 | ✅ 写进每个 agent prompt 的硬边界 |

**读不到就不跑**：题面里没有可用的 kernel 时限时，系统拒绝启动而不是猜一个值——
少跑一轮总好过整批提交作废。

## LLM 接线（已实测 200）

一个 Azure resource、一把 key、两种 wire format：

| 客户端 | 端点 | 鉴权 | 模型 |
|---|---|---|---|
| Codex | `<host>/openai/v1` + `/responses` | `Authorization: Bearer` | `gpt-5.6-sol` |
| Claude Code | `<host>/anthropic` + `/v1/messages` | `x-api-key` | `claude-fable-5` |

两个 base 携带的路径长度不同，因为客户端各自还会往后拼：Claude Code 的 SDK 追加
`/v1/messages`，Codex 追加 `/responses`。写成 `.../anthropic/v1` 会得到
`/anthropic/v1/v1/messages` → 404。配置里有校验会当场拒掉写反的情况。

端点从 `[run] llm_host` + 每条线的 `runner` **自动派生**，不用手写。换 Azure
resource 只改 `llm_host` 一行。

### 两个踩过的坑

**1. `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` 是必需的，不是卫生习惯。**
不设它时 Claude Code 会带上 `anthropic-beta: advisor-tool-2026-03-01`，Azure 端点
直接 400：`Unexpected value(s) ... for the anthropic-beta header`。runner 里已经写死了。

**2. `--bare` 会把工具集砍到只剩 Bash/Edit/Read**，而且 `--tools` 和 `--allowedTools`
都加不回来。solver 需要 Write 才能发布候选，所以 solver 不能用 `--bare`。
manager 反过来用 `--tools ""`（一个工具都没有）——它需要的信息全在 prompt 里，
一个能跑 Bash 的 manager 就是一个能自己想办法提交的 manager。

### 逃生舱

如果 codex 这条线当天出问题，改 `configs/backup_system.toml` 一行：

```toml
[codex]
runner = "claude_code"      # model 保持 gpt-5.6-sol 不变
```

端点会自动跟着换到 anthropic 那个。这条线仍然跑 gpt-5.6-sol，模型多样性不丢，
只是换 Claude Code 这个客户端去调。不用改别的任何东西。

> 参考：同样的配置打到 OpenRouter 上时，codex-cli 0.144.3 在
> `wire_api="responses"` 下**不发 Authorization 头**（401 "Missing Authentication
> header"，而 `codex doctor` 报告 key 存在）。Azure 没有这个问题。
> `final_system/runners.py` 的 `OpenRouterCodexRunner` 用的正是 OpenRouter 那套配置。

## 产物

```
workspace/backup_system/<run-id>/
├── RUN_STATUS.json
└── <slug>/
    ├── work/                  ← 两个 agent 共用的工作目录
    │   ├── OFFICIAL_ASSETS -> 只读资产
    │   ├── NOTES.md           ← 共享笔记
    │   ├── codex/  claude/    ← 各自的草稿
    ├── candidates/<id>/{kernel.py,meta.json,READY}
    ├── traces/                ← 每个 agent 每轮的完整 JSONL
    ├── feedback.jsonl         ← 榜分回传，两个 agent 都读这个
    ├── events.jsonl
    ├── state.json             ← 额度与重试状态
    └── SUMMARY.json
```

## 测试

```bash
python -m pytest tests/test_backup_system.py -q
```

27 个离线测试，不需要 key、不碰 Kaggle、不调模型。
