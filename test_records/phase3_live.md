# 阶段 3 · 完整 2 小时 --live  2026-08-04 15:15:22 PDT

pid 2032068 | --duration-minutes 120 --competition-mode practice --run-id live1
--kaggle-user qam1ng --live | GPU 4-7

认证拓扑(已改为单一来源,无 OAuth):
  Search + Claude 直跑 -> ANTHROPIC_API_KEY   (runners.py)
  HearSay             -> ANTHROPIC_API_KEY   (controller.py)
  Codex               -> OPENROUTER_API_KEY  (原有未改)

启动冒烟(三条各出一个 token):
  Claude runner(--bare + API key)  is_error=False cost=$0.0053 "pong"
  anthropic SDK 直调                "pong" 4 tok
  OpenRouter gpt-5.6-sol            "pong"

首次启动失败: ValueError --kaggle-user is required with --live
  -> 她的安全设计:真实提交必须显式声明 Kaggle 账号。补参数后正常。

t+2min: Search 线存活(阶段 2 时 1.4s 即挂),4 个 claude 进程,1307 文件。
起始 Kaggle 配额: Remaining today: 99
对照基线: 我们纯 HearSay 在本题最好成绩 0.82046 (public)

## t+34min 检查

### 1. Search 线 —— 真产出 bundle(阶段2是空的 raw_assets_fallback)
search_handoff @ +1479s，bundle 非空。SEARCH_BUNDLE 内含:
  TASK_ANALYSIS.md  RESEARCH_SYNTHESIS.md  SEARCH_OUTPUT.md  research_raw/
  ASSET_MANIFEST.json  ASSET_MAP.md  SHA256SUMS  ORIGINAL_ASSETS/  stage_a/
→ 认证修复后，1 分析 + 4 研究 的完整链路首次跑通。耗时约 25 分钟。

### 2. Kaggle 真实提交: 0 次
5 次 broker_submission 全部 consumed=False，一次都没到 Kaggle。
最新提交仍是 04:07 我们 HearSay 的 55228840 (0.82046)。

### 5. broker 失败原因分布
  4 次: "kernel push failed: [kaggle CLI not found on PATH]"
        codex-classic-blend-v1 / claude-ast-ft-v1 / claude-ast-ft-v2 /
        codex-classic-multiscale-v2   (local_score 全为 None)
  2 次: "kernel-metadata.json is missing"
        hearsay-solver_a local=0.93043   hearsay-solver_c local=0.53478

#### 故障 A: kaggle CLI not found on PATH
控制器进程 PATH=/usr/local/sbin:...:/snap/bin —— 不含 ~/venvs/ioai312/bin。
kaggle 实际在 /home/qyan/venvs/ioai312/bin/kaggle。
代码里两套写法并存:
  final_system/kaggle.py:10   from agent.tools.registry import _kaggle_bin  ← 正确
  swarm/submit/kernel.py:234  _run(["kaggle", "kernels", "push", ...])      ← 裸名
  swarm/submit/broker.py:431  ["kaggle", "competitions", ...]               ← 裸名
提交路径走的是 swarm/submit/*，用裸名，所以失败。
性质: 与我们 HearSay 早先修过的同一类 bug(_kaggle_bin 解析到 sys.executable
旁边)。她在 final_system/kaggle.py 已经用了正确写法，swarm/submit 下没同步。
责任: 部分是我的启动环境(未 export PATH)，但裸名调用本身是脆弱点。

#### 故障 B: kernel-metadata.json is missing
HearSay 线两个候选(0.930 / 0.535)交到 broker 时缺 kernel-metadata.json。
HearSay 在 --external-broker-dir 模式下不自行提交，候选包需自带 kernel 目录；
solver 未生成。待确认是契约未传达还是打包环节遗漏。

### 4. fold_count 仍是 1(阶段2也是 1)
evaluation_contract_frozen @ +1644s: fold_count=1, source=hearsay_evaluator
2 小时窗口下依然只切 1 折 —— 不是窗口长度导致的。

### 3. 候选与分数
本地分仅 HearSay 线有(0.93043 / 0.53478)；codex 与 claude 直跑线候选
local_score=None —— 未经公共尺子重评分即进入 broker。

## 提交链路修复(3 处)

### 修 1: kaggle 裸名调用 -> _kaggle_bin()
swarm/submit/kernel.py  5 处   swarm/submit/broker.py  1 处
窄 PATH 实测: _kaggle_bin() -> /home/qyan/venvs/ioai312/bin/kaggle
解掉 10/15 次 "kaggle CLI not found on PATH"。

### 修 2: HearSay 候选在 kernel 建好前就被注册(根因)
final_system/controller.py:_register_hearsay
  a) manifest["submission_mode"] 硬编码 "unknown" -> 改为本次运行解析出的真实
     mode(新增 self._resolved_mode，在 detect_mode 后写入)
  b) 触发条件只看 out/submission.csv;solver 先写 CSV 后建 kernel，注册落在
     那个窗口里。实测 solver_a kernel 建于 ...396，broker 尝试于 ...503，
     相差 107 秒。现在 mode==kernel 时必须存在
     out/kernel/kernel-metadata.json 才注册。
证据: LKG 快照里 kernel-metadata.json 确实存在(solver_a/solver_c 都有)，
      但 broker 用的 7 个候选 id 与现存 2 个快照名完全对不上 —— 说明 broker
      读到的是更早、尚无 kernel 的快照。

### 待观察(未修)
- registry.py: status = "eligible" if format_result.valid — 仅校验 CSV。
  修 2 在上游拦住了 HearSay 线，但 codex/claude 线仍可能以无 kernel 的候选
  进入 eligible。若复现再修。
- fold_count=1(两轮皆是)
- codex/claude 候选 local_score=None，未经公共尺子重评分

## Kaggle 账号变更
新 token 属于 hearsayagent(全新账号，0 次提交，配额 100)。
旧账号 qam1ng 上的 0.82046 基线不在此账号下。
--kaggle-user 必须传 hearsayagent，否则 kernel id 前缀不匹配会被拒。

## live2(24 分钟，因磁盘满被停)

背景：跑到 24 分钟时根分区 100% 满(剩 288M)，broker 报 OSError [Errno 28]
No space left on device。按指示先停 agent 再清理，故本轮数据不完整。

### 1. Kaggle 真实提交：0 次
账号已切 hearsayagent(全新，Lifetime 0，配额 100)。

### 2. broker：尝试 3 次，成功 0
  2x  kernel push failed: 400 Client Error (SaveKernel)
  1x  candidate mode csv differs from competition mode kernel
  另有 2 次 broker_background_error: OSError [Errno 28] No space left on device
→ 400 Bad Request 极可能是磁盘满的下游表现(打包 kernel 时写临时目录失败)，
  需在磁盘充裕的条件下复验才能定性。

### 3. 三路线对照：本轮无法比较
只有 claude 直跑线产出了 3 个候选，且 score=None(公共尺子未评分)。
hearsay 线在 +23.4min 才启动(等 Search bundle)，随即被停。
codex 线本轮 0 候选。

### 4. kernel 注册竞态修复：生效
  live1: "kernel-metadata.json is missing" 4 次
  live2: 0 次
注册表中 claude 候选的 submission_mode 已是真实值 kernel/csv，不再是硬编码
"unknown" —— controller.py 的 _resolved_mode 透传生效。

### 遗留问题
- claude-c3 声明 mode=csv，但本题是 kernel-only → 被 broker 正确拒绝。
  说明 claude 直跑线的 prompt 没有把「必须交 kernel」传达到位。
- 三条路线候选的 evaluation score 均为 None：公共 metric+folds 重评分对
  codex/claude 线仍未生效(live1 同样现象，两轮复现)。

## 环境治理(本轮引发)
- 根分区 288M -> 78G
- 两套系统 workspace/archive 全部迁至 /data/qyan/ws/，以符号链接接回
- 沙箱 HOME 的 pip --user 安装(每轮约 5G 的 CUDA/PyTorch，四轮 20G)：
  runners.py + controller.py 增设 PYTHONUSERBASE / PIP_CACHE_DIR 指向 /data
- ~/.cache/pip 亦迁至 /data
- 发现：Search 每轮把 2GB 数据集整份复制进 workspace 并 chmod 只读
  (不可变证据设计)，尚未改动

## live3(35 分钟检查)

### 更正:我此前两次说"公共尺子对 codex/claude 线不生效"是错的
分数一直都在，键名是 evaluation.mean / pooled，不是 evaluation.score。
我的检查脚本和 fswatch 都读了不存在的键，于是把有分数报成 None。已修 fswatch。
—— 又一次拿自己的读取代码当事实源。

### 3. 三条路线在公共尺子上的真实对照(n=184)
  hearsay  0.94022  x5   (solver_a x4, solver_b x1)
  hearsay  0.93478  x4
  hearsay  0.92935  x3
  claude   0.91848       claude-ast29-full-v6
  claude   0.82609       claude-ast29-twohead-v5
  codex    —— 全部 shape_mismatch，无分数
有分数 14/23。

**HearSay(读 Search 情报)以 0.94022 领先 claude 直跑线 0.91848 约 0.022。**
这是三轮以来第一次拿到跨路线可比数字。样本小、单折，不能当定论，但方向明确。

### shape_mismatch 的真正含义
  expected_rows=920   prediction_rows=624
公共尺子按 920 个单元(train 296 + fine_tune 624)编制；codex/claude 直跑线
只对 624 条新类样本产出 OOF —— 它们把题目理解成"只做新类"，与共享折的单元
定义不一致，因此无法比较。这不是评估器故障，是**单元契约没有传达到独立路线**。
9/23 候选因此作废。

### 1. Kaggle 真实提交：仍 0 次
### 2. broker：尝试 7，成功 0
  5x  400 Client Error (SaveKernel)
  1x  Dataset must be specified in the form of {username}/{dataset-slug}
  1x  candidate mode csv differs from competition mode kernel
  后台错误 0 次(磁盘问题已消失)

#### 400 根因已定位并修复：kernel title 超长
逐字段隔离(全部单独通过后才命中最后一个变量)：
  最小 kernel                    OK
  + competition_sources          OK
  + GPU/P100                     OK
  + wheel 数据集                  OK
  + 55 字符 id (title 短)         OK
  + 54 字符 title                 400   <- 唯一失败点
swarm/submit/kernel.py: make_metadata 让 title 直接继承 slug，
MAX_KERNEL_SLUG=44 只约束 slug，title 从不截断。broker id 是四段拼接
(题目缩写-候选名-指纹-提交id)必然超长 => 该路径下没有任何 kernel 能推上去。
已加 MAX_KERNEL_TITLE=44 并截断；用完整真实 metadata 验证推送成功。
注意：live3 加载的是旧代码，本轮剩余时间仍然交不出去。

### 4. 磁盘稳定
  /      77G 可用   /data  1.1T 可用   全程无 ENOSPC

## live4(35 分钟检查)

### 1. Kaggle 真实提交：0 次(hearsayagent, Lifetime 0)
### 2. broker：尝试 1，成功 0
  1x kernel push failed: Your kernel title does not resolve to the specified id
     + Dataset must be specified in the form of {username}/{dataset-slug}
本轮只试了 1 次 —— 公共尺子契约 +30.1min 才冻结(HearSay +23.5min 才启动，
要等 Search 交棒)，此前 10 个候选全是 pending_contract，broker 无从选择。

#### 第 3、4 层错误(本轮暴露，已修，本轮未生效)
kernel 提交是四层错误叠加，每修一层露出下一层：
  1) kaggle CLI 不在 PATH        已修(live3 消失)
  2) kernel title 超长 -> 400    已修(live4 不再报 400)
  3) dataset owner 是字面占位符   本轮暴露 <- 真正致命
  4) title 截断后与 id 不一致     本轮暴露(仅警告)
第 3 层细节：claude 线写出 "REPLACE_OWNER/ioai-2026-wheel-dataset"，
broker 的 final_system/kaggle.py:185 原样照抄：
    dataset_sources=source_metadata.get("dataset_sources") or []
agent 的判断是对的，它在自己的 evidence/NOTES.md 里写明：
    "kernel-metadata.json has REPLACE_USER / REPLACE_OWNER placeholders
     (no Kaggle creds in lane)"
沙箱无 Kaggle 凭证是设计(broker 独占提交权)，所以 agent 解析不出 owner。
错在 broker 不解析 —— 它才是持有凭证的组件。
修复：新增 KaggleAdapter._resolve_datasets()，占位符/裸 slug 一律经
`kaggle datasets list -s` 解析；解析不到则丢弃而非推占位符。实测：
  REPLACE_OWNER/ioai-2026-wheel-dataset -> kamalkhan/ioai-2026-wheel-dataset
  ioai-2026-wheel-dataset               -> kamalkhan/ioai-2026-wheel-dataset
第 4 层修复：title 截断时同步截断 id，两者互相解析(实测一致)。

### 3. 三路线公共尺子对照(n 未变，单折)
  hearsay  0.91739 x2   0.90435 x4
  codex    0.89565      0.88696      0.68696
  claude   —— 8 个候选全部 shape_mismatch
有分数 9/20；另有 3 个 no_oof。

**shape_mismatch 再现且加重**：期望 920 行，实得 624 行，本轮 100% 集中在
claude 线(live3 是 claude+codex 都有)。codex 线本轮学会了产出 920 行并进榜。
=> 直跑线对"单元定义"的理解不稳定：同一套 prompt，codex 这轮对了、claude 全错。
   这不是随机波动，是共享单元契约从未传达给不读 Search 的路线。

### 4. 磁盘稳定：/ 77G、/data 1.1T，无 ENOSPC

## live5(33 分钟，跳过 Search 复用 live4 bundle)

### 突破：kernel 首次成功推上 Kaggle
前五轮 23 次尝试 0 成功；本轮 4 个 kernel 全部推送成功。
四层修复全部验证生效：
  1) kaggle CLI 不在 PATH   2) title 超长 -> 400
  3) dataset owner 占位符    4) title/id 不一致
Kaggle 上可见：
  hearsayagent/ioai-2bfb-hearsay-solver-c-bb9f3b0c75  02:15:23
  hearsayagent/ioai-2bfb-hearsay-solver-a-1d166127df  02:17:03
  hearsayagent/ioai-2bfb-hearsay-solver-b-29f74a76ad  01:55:01
  hearsayagent/ioai-2bfb-claude-ast-ft-rehearsal-v1   01:32:49

### 1/2/3. 仍未提交：两个 kernel 已 RUNNING 21 分钟未 COMPLETE
  broker: floor(solver_c 0.90608) reserved 20.9 分钟
          calibration(solver_a 0.88950) reserved 19.2 分钟
  Kaggle 提交总数：0
  max_inflight_submissions=2 已占满，第三个候选(0.93370)排队等位
kernel 在云端做完整训练(1283 音频抽特征 + 训练 + 推理)，20 分钟未完成属正常
范围，但需继续观察是否有 30 分钟上限风险。

### 4. 三路线分数(公共尺子，单折 n=181)
  hearsay  最高 0.93370   24 个候选
  codex    最高 0.70718    1 个候选
  claude   ——              3 个候选全部 metric_error
候选数 24:1:3 严重失衡，无法据此判定路线优劣。

#### claude 线 metric_error 详情
  ValueError: length mismatch: y_true=181  y_pred=5249
不是 live3/live4 的 shape_mismatch(624 vs 920)，而是 5249 行 —— 疑似输出了
每个滑窗而非每个片段。claude 进程仍活跃。
=> 与前两轮同源：公共尺子的单元定义只存在于 HearSay 的 evaluator 中，
   从未下发给独立路线。codex 猜 624/920、claude 猜 5249，全在盲考。
   而定义标准的 evaluator 属于三方之一，对照实验的公正性不成立。

### 中断：HearSay 线因 API 余额耗尽全线停机
  [solver_a/b/c] giving up: the Anthropic account is out of credit
  ===== RUN ABORTED: the Anthropic account is out of credit =====
  [budget] elapsed=26min cost=$116.08 (no ceiling)
HearSay 一条线 26 分钟烧掉 $116(我们自己的 HearSay 整轮 $50-70)。
原因：[hearsay] rounds=40 max_turns=250 且配置中无任何成本上限字段。
codex 线(OpenRouter)与 claude 直跑线仍在运行。
两个已推送的 kernel 不依赖 LLM，会自行跑完并回传分数。

### 监控盲区(已知，待修)
fswatch 读 controller_events.jsonl，而 broker_submission 是终态事件；
reserved 状态只存在 broker_state.json，故显示"尝试 0 次"实为 2 个在飞。

## Linux 沙箱补齐(她的新提交要求 --live 必须有 OS 沙箱,原实现仅 macOS)
final_system/security.py 新增 _build_bwrap_sandbox()——用系统自带
/usr/bin/bwrap 复刻 Seatbelt 契约:
  --ro-bind / /       整个文件系统只读,"不可写"成为默认而非规则
  自身工作区 --bind 回可写
  ~/.kaggle 等凭证目录 --tmpfs 盖空;单个凭证文件 --ro-bind /dev/null
  kaggle CLI 二进制 --ro-bind /dev/null(禁执行)
  --unshare-pid       私有 PID 命名空间,控制器进程(持有真实 key)在 /proc 不可见
  --die-with-parent
覆盖顺序修正:denied 盖在 writable 之后(bwrap 后写生效),否则落在可写根下的
凭证会露出——探针实测 exit=21 抓到过。
Linux 探针改用 /proc/<controller_pid> 存在性(私有 PID 命名空间下应不存在)。
验证 8 项全 PASS:读凭证/执行 kaggle/写仓库根 均失败;读官方资产/写自己工作区/
网络/读 pyuserbase 均成功;官方 verify_agent_sandbox 通过。
doctor 认 /usr/bin/bwrap,--live 在 Linux 上解锁。

## 补:最优候选晚交 / slot 浪费(争抢层)

问题:day_gate 争抢 GPU slot 时排序键只有 (floor优先, 到达时间)——不看分数。
live5 实况:0.934 在第 15 分钟出现,却排在第 11 分钟入队的 0.906 之后,
整个窗口都在等。broker 选候选时已按分数排序,但那个顺序没带进 slot 争抢。

修 1:分数插队(day_gate.py 排序键)
  (floor优先, -local_score, 到达时间)
  broker.acquire 传入 self._score(record) → request 存 local_score → 排序用它。
  实测:[floor 0.80, 晚到但0.934, 早到0.906, 无分] 正确排为 d,b,a,c。

修 2:floor 延迟(broker.py _policy_options)
  floor 是第一次提交、不比分数,live5 第 11 分钟就把 0.906 交了并占一个 slot。
  现在延迟到"尺子稳定":候选≥3 或 已过 IOAI_FLOOR_SETTLE_S(默认300s)或
  fraction_elapsed≥0.15,三者任一先到即放行。
  三条件保证:①有足够候选就早交 ②不无限等 ③短任务(30分钟)按窗口比例照样早交,
  不会憋到 grace 兜底。
  实测四场景全 PASS。

只改排序与 floor 门,不动调度骨架、不动 external_running/resource_gate。
风险低;短任务下 fraction_elapsed≥0.15 这条保证不会退化。
