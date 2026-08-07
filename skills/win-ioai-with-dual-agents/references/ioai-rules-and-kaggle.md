# IOAI 规则与 Kaggle CLI 参考

## 目录

1. [规则优先级](#规则优先级)
2. [赛制与自主边界](#赛制与自主边界)
3. [Kernel、硬件和资产](#kernel硬件和资产)
4. [Notebook version 与截止时间](#notebook-version-与截止时间)
5. [技术 Report](#技术-report)
6. [CLI 认证](#cli-认证)
7. [CLI 全流程](#cli-全流程)
8. [metadata 模板](#metadata-模板)
9. [输出检查](#输出检查)
10. [同伴代码互通与限制](#同伴代码互通与限制)

## 规则优先级

发生冲突时按以下顺序执行，并优先采用更具体、更晚、更严格的约束：

1. Kaggle Foundational Rules。
2. 当前题目的 Kaggle Rules、Overview、Evaluation、Starter/Continuation/Report prompt。
3. 更晚的 IOAI 组织者公告。
4. IOAI 官网 FAQ 或通用赛制 PDF。
5. Kaggle CLI 通用文档。
6. 本 Skill 和本地历史经验。

不要从上一题复制 timeout、硬件、模型、数据或 push 上限。把官方原文保存到任务资产包，并把最终解释写入机器可读 `TASK_CONTRACT.json`。

权威入口：

- IOAI AI Models Track：<https://ioai-official.org/ai-model-track/>
- IOAI Rules & Format：<https://ioai-official.org/ai-model-track/rules-format/>
- Kaggle CLI kernels：<https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md>
- Kaggle CLI competitions：<https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md>
- Kaggle CLI authentication：<https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md>

## 赛制与自主边界

较新的官方参赛者公告采用以下现场时序：每天 3 场连续 Kaggle competition，每场 2 小时，中间两次各 15 分钟，共约 6.5 小时。旧官网材料可能仍写“6 小时窗口内 3 题”，不要据此设计三题同时运行的系统。

正式比赛中，Agent 自主完成：

- 下载 Kaggle 数据；
- 在预先配置好的硬件上写和迭代 Python；
- 用题目指定的 timeout 执行 `kaggle kernels push`；
- 等 Kernel 完成并检查输出；
- 在 deadline 前执行 `kaggle competitions submit`。

人类只执行组织者明确允许的操作：加入比赛、提供同账号 KGAT、粘贴原样 Starter prompt；系统中断时粘贴原样 Continuation prompt；监控并上报 Kaggle/任务故障；按规则处理缺失 Report。不得基于题目修改后端、临时扩容、写自定义提示、纠正算法或指导提交。

把系统和 Skill 在看题前冻结。练习赛中的人工干预只能用于开发测试，不代表正式赛允许。

## Kernel、硬件和资产

正式题是 Notebook/code competition 时，必须由 Kaggle Kernel 生成 `submission.csv`。不得把本地 CSV 直接上传来绕过 Kernel。

已知 2026 正式题曾允许：

- CPU：metadata 中 `enable_gpu: false`、`machine_shape: ""`。
- GPU：`enable_gpu: true`、`machine_shape: "NvidiaTeslaT4"`。
- Kaggle 可能显示 T4 ×2，但只允许使用 `cuda:0`。
- P100 被明确禁止。
- `enable_internet: false`。

已核验的 Task 2 页面还明确给出账号级 batch 并发上限 CPU=5、GPU=2。并发容量与机型一样是题目专属事实；新题必须重新读取并冻结，不能从 Task 2 继承。

CPU 的具体型号没有在已核验的题目规则中固定，不要编造。若新题规则允许不同硬件，以新题为准。

外部网络研究不等于可以把外部资产带进方案。除题目明确允许外，禁止外部数据、模型、权重、embedding、LoRA、checkpoint、服务/API 生成的预测或特征。官方提供的 wheel 数据集只能用于软件依赖，不得暗藏模型或答案。

Kernel 中先执行官方 environment setup block，再导入会受其影响的 NumPy、PyTorch 等库。技术 Report 必须位于这个 setup block 更上方。

不要硬编码 `/kaggle/input/<猜测目录>`。递归列出 `/kaggle/input`，按必要文件组合定位题目目录，并在找不到或找到多个候选时明确失败。

组织者放在当前 competition/data/model 资产中的带标签 validation、dev 或 test_public 属于官方资产。先保存用途原文和 SHA，再按 `training_use` 执行：可以对 `allowed` split 做本地评估、反复训练与超参搜索；只有方法冻结后才把它并入最终从零训练。`validation_only` 只能评估，`UNKNOWN` 保守不用作训练。无论哪种，最终 Kernel 都不得携带本地权重、标签副本、prediction、ID 映射或派生缓存。官方 validation 也不是隐藏 test；固定 split 查询过多仍会过拟合。

合同还要冻结 `asset-audit.json`：资产发现完成状态、文件 role/SHA、base train、全部带标签 split，以及它们对 train/submission_test 的 ID 和内容重叠。额外 dataset/kernel/model source 必须逐项绑定官方原文授权；本地上传的 cache 不能进入白名单。最终训练 split 同时写入合同和 `.py` 顶部 `IOAI_FINAL_TRAINING_SPLITS` 声明，`validation_only/UNKNOWN` 会被预检拒绝。静态门能挡住明显的大数组、压缩 blob 和预测表，但不能证明任意 Python 语义，所以仍要交叉审计实际读取与训练路径。

## Notebook version 与截止时间

官网通用 FAQ 曾写每题 50 次，但已核验的题目专属 Rules 采用最多 20 个 Kaggle Notebook versions。后者对该题有效。

严格区分：

- `kaggle kernels push` 创建 Notebook version；成功、失败、停止或未提交都可能计入上限。
- `kaggle competitions submit` 把已完成的某个 version 提交到比赛。
- Kaggle UI 的 remaining submissions 只统计 competition submissions，不可靠地反映 Notebook version 上限。
- 同一个 Notebook version 通常最多提交一次。

Kaggle Kernel ref 在账号内跨比赛全局复用。每个 candidate 的 slug 必须同时包含当前 `competition_slug` 的 SHA-256 前 8 位和 `CX07`/`CL08` 候选 ID，例如 `a1b2c3d4-cx07-family`。包装器会先以 competition 列表对账 version 预算，再用账号全局精确 ref 搜索拒绝任何历史 slug 冲突，从而保持“一候选一 slug、只提交 version 1”。一次 push 后等待结果；不要例行 push 两次。网络失败时先查询远端是否已创建 version，再决定是否重试。

push 必须显式传题目要求的精确 timeout。例如历史 Task 1 要求 `--timeout 600`，历史 Task 2 要求 `--timeout 300`。未从新题官方材料解析出 timeout 时，不得 live push。

同一账号的所有题与两条 lane 共享 `$IOAI_ACCOUNT_ROOT/kernel-slot-ledger.jsonl`。包装器按合同 CPU/GPU 容量和每 actor 上限分配 lease；槽满返回 75，不产生 version。final 候选只在真实等待时阻止新 normal 抢下一槽，不静态预留 GPU。无法查询远端状态时 fail closed；终态才释放。所有 Agent、subagent 和人工干预必须走同一个包装器；账号 Kernel 列表分页发现账本外 active/unknown、list/status 冲突或页数超限时包装器会拒绝新 push，但这只是兜底扫描，不是支持旁路写入。active 容量策略不一致时同样拒绝；lease/waiter 使用创建时冻结的 orphan/TTL。

有效提交的关键时间是 competition submit 命令在 Final Submission Deadline 前发出；评分可以稍后完成。仍应留出下载输出、格式验证与 API 抖动的安全余量。deadline 后的普通提交即使出分，也可能不进入排行榜或最终成绩。

## 技术 Report

每个提交的 `.py` 顶部、environment setup block 之前，写 8–10 个短段落的 `#` 注释。为让预检无歧义，把每段标成 `Technical report 1/9` 到 `9/9`，然后再写明确的 `ENVIRONMENT SETUP` 标记。

只描述本文件中的方案：

- 任务建模方式；
- 数据与特征；
- 模型、loss 和解码；
- 关键超参数；
- validation 与实测分数；
- 运行时间和资源；
- 被尝试但淘汰的方法及原因。

不要描述 Agent、subagent、harness、prompt、搜索过程或人类操作。Report 不计分，但用于团队和 Jury 审阅、申诉以及赛后公开透明。

如果比赛结束时 Report 缺失，按组织者给定的 Report Generation Prompt 生成，把它 prepend 到最新 `.py`，并在赛后 30 分钟内 push 和 late-submit。该 late submission 只备案，不改变排行榜成绩。只有合同从本题官方 prompt 冻结了非零 `late_report_window_seconds` 时才可使用下述 `--late-report-only`；description 必须明确包含 `d=late-report`：

```bash
python scripts/guarded_kernel_push.py late-report-kernel \
  --contract "$IOAI_TASK_ROOT/TASK_CONTRACT.json" \
  --candidate CX99 --actor CX \
  --ledger "$IOAI_TASK_ROOT/push-ledger.jsonl" \
  --late-report-only

python scripts/guarded_competition_submit.py \
  --contract "$IOAI_TASK_ROOT/TASK_CONTRACT.json" \
  --candidate CX99 --actor CX \
  --kernel-ref OWNER/TASKTAG-cx99-late-report --version 1 \
  --description 'CX99|p=-|f=report|cv=na|d=late-report|h=abcdef|r=REFHASH10' \
  --push-ledger "$IOAI_TASK_ROOT/push-ledger.jsonl" \
  --submit-ledger "$IOAI_TASK_ROOT/submit-ledger.jsonl" \
  --late-report-only
```

## CLI 认证

Kaggle CLI 使用 API token，不使用账号密码。凭证必须对应参加所有当日题目的同一个、已注册且验证手机号的 Kaggle 账号。

把 token 单独保存为：

```text
~/.kaggle/access_token
```

并设置：

```bash
chmod 600 "$HOME/.kaggle/access_token"
```

不要全局 export token；环境变量会被 Codex、Claude、搜索 subagent 和子进程继承。只在主 Agent执行受信任的 Kaggle I/O 命令时由 CLI 按需读取该文件；研究进程不挂载 `~/.kaggle`，不执行网页下载代码。不要把 `KGAT_...` 写进 Skill、shell history、Git、Kernel、报告、状态文件或 Agent prompt。legacy `~/.kaggle/kaggle.json` 也应为 `0600`，但优先使用 access token。

认证自检只输出用户名和认证方式，不输出 token：

```bash
kaggle config view
kaggle competitions list --search IOAI
```

正式 launcher 还应在赛前固定一个账号级共享目录，例如：

```bash
export IOAI_ACCOUNT_ROOT=/shared/ioai/account-state
```

它只放不含凭证的槽位账本；三道题的 `IOAI_TASK_ROOT` 可以不同，但 `IOAI_ACCOUNT_ROOT` 必须相同。不要把 token 放进这个目录。

## CLI 全流程

以下命令是结构模板。先用当前 CLI 的 `--help` 核对参数，因为 Kaggle CLI 版本会变化。

安装或升级：

```bash
# 若 Kaggle CLI 由 uv 管理（本机当前采用此方式）
uv tool install --upgrade kaggle

# 或在专用 Python 环境中升级
python3 -m pip install --upgrade kaggle
kaggle --version
```

下载比赛文件：

```bash
mkdir -p competition-data
kaggle competitions download -c COMPETITION_SLUG -p competition-data
```

读取页面可用内容：

```bash
kaggle competitions files COMPETITION_SLUG
kaggle competitions leaderboard COMPETITION_SLUG --show
```

部分新版 CLI 支持 `competitions pages`。先运行：

```bash
kaggle competitions pages --help
```

再按帮助读取 `rules`、`overview` 或其他页面；私有题必须先在网页加入并接受规则。

提交前预检：

```bash
python scripts/preflight_kernel.py kernel_dir \
  --contract "$IOAI_TASK_ROOT/TASK_CONTRACT.json"
```

预检只从冻结合同读取 competition、timeout、CPU/GPU、网络、Report、严格文件数和允许资产，不接受命令行自由覆盖，避免新题合同 fail-open。

正式模式使用唯一原子包装器，只 push 一次：

```bash
python scripts/guarded_kernel_push.py kernel_dir \
  --contract "$IOAI_TASK_ROOT/TASK_CONTRACT.json" \
  --candidate CX07 \
  --actor CX \
  --ledger "$IOAI_TASK_ROOT/push-ledger.jsonl" \
  --priority normal
```

包装器内部执行的仍是官方 CLI：

```bash
kaggle kernels push -p kernel_dir --timeout TASK_TIMEOUT_SECONDS
```

但 timeout 来自冻结合同，且 push 前已原子占用账号槽位与 version。退出码 75 表示资源暂缓，按提示重跑同一包装器；决胜候选在 final 窗口改用 `--priority final`。不要绕过包装器直接运行该命令。若 CLI 超时或返回非零，重跑完全相同的包装器命令进行幂等远端对账。

轮询状态和日志：

```bash
kaggle kernels status OWNER/KERNEL_SLUG
kaggle kernels logs OWNER/KERNEL_SLUG
```

完成后下载真实输出并在本地复核：

```bash
mkdir -p kernel-output
kaggle kernels output OWNER/KERNEL_SLUG -p kernel-output
```

先做通用确定性校验，例如：

```bash
python scripts/validate_submission.py kernel-output/submission.csv \
  --sample competition-data/sample_submission.csv \
  --id-column id \
  --integer-column prediction \
  --range prediction:0:5
```

再按当前题 schema 增加排列、概率和分组约束；通用脚本不能替代题目专属断言。

提交已完成的明确 version 时，正式模式使用 Skill 的原子 submit 包装器：

```bash
python scripts/guarded_competition_submit.py \
  --contract "$IOAI_TASK_ROOT/TASK_CONTRACT.json" \
  --candidate CX07 \
  --actor CX \
  --kernel-ref OWNER/TASKTAG-cx07-family \
  --version 1 \
  --description 'CX07|p=-|f=family|cv=.123|d=change|h=abcdef|r=REFHASH10' \
  --push-ledger "$IOAI_TASK_ROOT/push-ledger.jsonl" \
  --submit-ledger "$IOAI_TASK_ROOT/submit-ledger.jsonl"
```

它内部执行的官方 CLI 是：

```bash
kaggle competitions submit COMPETITION_SLUG \
  -k OWNER/KERNEL_SLUG \
  -v VERSION_NUMBER \
  -f submission.csv \
  -m 'CX07|p=-|f=family|cv=.123|d=change|h=abcdef|r=REFHASH10'
```

`REFHASH10` 用下式计算，实际命令不得保留占位符：

```bash
python3 -c 'import hashlib; print(hashlib.sha256("OWNER/KERNEL_SLUG@1".encode()).hexdigest()[:10])'
```

包装器会先确认 Kernel 成功状态，自行下载远端输出，并按冻结的 sample SHA、schema 与题目专属 validator 验证；它不接受调用者传入的本地同名 CSV。`r=` 将 description 绑定到 kernel/version，包装器还要求远端状态为 pending/complete；error 或其他 kernel 的同 description 不会假确认为成功。task validator 的冻结接口是 `python validate_task.py ACTUAL --sample SAMPLE`，应覆盖排列、概率和、分组及序列等当前题约束。不要绕过包装器直接运行该命令；同一 version 的 ambiguous response 应重跑同一包装器做幂等对账，不得直接再次 submit。

需要 GPU 时，第一个有真实预测价值的 GPU floor 就承担硬件冒烟：必须完整通过 push→run→output validate→submit。仅看到 Kernel `COMPLETE` 不能证明合规，P100/metadata 类拒绝可能到 submit 才暴露。包装器用 `submit_with_diagnostics.py` 对同一次 HTTP 请求提取状态码和 `code/status/message/detail/reason/errors` 等白名单字段；submit ledger 只存脱敏字段、正文长度和 SHA，不存 headers、URL、raw HTML 或凭证，也不发送第二次探针提交。发生 4xx 先看该 `diagnostic`，再修 metadata 或向组织者报告平台问题。

查看提交和分数：

```bash
kaggle competitions submissions COMPETITION_SLUG --format json
```

注意：通用 Kaggle 示例可能展示直接 `-f local.csv` 的提交方式；Notebook-only IOAI 题不得使用这种方式。

## metadata 模板

CPU 示例：

```json
{
  "id": "OWNER/TASKTAG-CXNN-UNIQUE_KERNEL_SLUG",
  "title": "UNIQUE CANDIDATE TITLE",
  "code_file": "script.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": false,
  "machine_shape": "",
  "competition_sources": ["COMPETITION_SLUG"],
  "dataset_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

T4 示例：

```json
{
  "id": "OWNER/TASKTAG-CXNN-UNIQUE_KERNEL_SLUG",
  "title": "UNIQUE CANDIDATE TITLE",
  "code_file": "script.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": false,
  "machine_shape": "NvidiaTeslaT4",
  "competition_sources": ["COMPETITION_SLUG"],
  "dataset_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

metadata schema 或题目规则变化时，以 `kaggle kernels init` 生成的当前模板和题目 Rules 为准。

## 输出检查

Kernel 必须把最终文件写到：

```text
/kaggle/working/submission.csv
```

下载后检查：

- 文件名与 Rules 完全一致；
- header 和列顺序与 sample submission 一致；
- 行数一致；
- ID 集合、顺序和唯一性符合要求；
- prediction 类型、范围、排列或概率约束正确；
- 无 NaN、Inf、重复 ID 或额外 index 列；
- 训练日志显示实际使用允许资产和 `cuda:0`；
- 实际墙钟有安全余量。

不要只验证本地模拟输出；必须下载并验证远端 Kernel 的真实 `submission.csv`。

## 同伴代码互通与限制

新 submission 记录通常可提供时间、description、状态、Public/Private score；当前 CLI 的 JSON 不一定返回 Notebook URL。观察器会从 description 取 `CX07`/`CL08`，再在本人该 competition 的 Kernel 列表中寻找包含同一 ID 的唯一 slug。用唯一 Kernel slug 时，可执行：

```bash
kaggle kernels pull OWNER/KERNEL_SLUG -p peer-archive/CANDIDATE -m
```

私有 Kernel 的历史 version 精确 pull 可能返回 403，`kernels status/logs/output` 也通常面向 latest session。因此：

1. 每个 candidate 使用账号全局唯一 slug，把当前比赛 8 位标签和候选 ID 都写入 slug，不在同一 slug 上反复 push 不同实验。
2. 新 submission 出现后立即归档其 latest source 和 metadata；触发深审计时再拉日志和真实输出。
3. 用 submission ID、version、时间和 SHA 绑定归档，避免把后续 latest 当成历史提交。
4. 无法核验 source/version 对应关系时，把证据标为不确定，不盲目继承。

Kaggle token 能代表账号调用 API；它与网页登录密码无关。不要让浏览器密码进入 CLI，也不要在日志中打印认证 header。
