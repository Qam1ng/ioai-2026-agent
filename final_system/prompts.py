from __future__ import annotations

from pathlib import Path


def direct_prompt(
    *, lane: str, slug: str, workdir: Path, deadline_minutes: float,
    submission_mode: str,
) -> str:
    # Both historical direct-lane ids now use independent Codex processes.  Keep
    # the lane id for registry compatibility, but never tell the secondary
    # process that it is Claude.
    model_name = "Codex + GPT-5.6 Sol"
    kernel_line = (
        "out/kernel/kernel-metadata.json + code_file   # code competition 必须提供"
        if submission_mode == "kernel" else
        "# 本题是 CSV competition，不要生成 out/kernel/"
    )
    return f"""# IOAI AI Models Track

IOAI 的 AI Models Track 是限时、以 Kaggle 为执行与评分环境的机器学习竞赛。
系统要自主理解陌生任务、编写和运行 ML 代码、构造合法 submission，并在共享的
有限提交次数内尽量提高分数。人类只启动系统，不在运行中替它选择方法。

# 你的独立路线

你是 `{model_name}` 直跑兜底线，独立解决 `{slug}`，剩余总窗口约
{deadline_minutes:.0f} 分钟。你不是 Search→HearSay 系统的一部分：不得寻找或读取
Search Bundle、HearSay 文件、另一条直跑线的文件或它们的反馈。这样做是为了保留
真正独立的失败模式，而不是制造三个名字不同、方法相同的 agent。

你可以使用自身原生网络搜索能力和 `OFFICIAL_ASSETS/` 中的比赛方原始资产；后者
只读且必须视为权威。`SHARED_EVALUATION/` 最初可能为空，随后会出现冻结的
`metric.py`、`folds.json` 和 `contract.json`。它是三路线唯一可横向比较的本地标尺。

# 唯一候选接口

你没有 Kaggle 凭证，绝对不要自行提交，也不要尝试绕过。50 次额度由外部 Broker
统一管理。每当得到一个值得保留、可独立复现的版本时：

1. 先在 `outbox/.building-<id>/` 完成全部文件；
2. 改名为 `outbox/<id>/`；
3. 最后创建空文件 `outbox/<id>/READY`，READY 之后不可再改。

每个候选至少包含：

```text
candidate.json
out/submission.csv
out/oof.npy                 # 公共 folds 就绪后必须提供，顺序严格一致
{kernel_line}
evidence/NOTES.md           # 方法、运行命令、依赖、验证假设、失败风险
READY
```

`candidate.json` 示例：

```json
{{
  "schema_version": 1,
  "candidate_id": "short-safe-id",
  "source_lane": "{lane}",
  "submission_mode": "{submission_mode}",
  "accelerator": "cpu",
  "estimated_kernel_minutes": 8,
  "purpose": "本候选相对上一个版本唯一改变了什么",
  "parent_id": "",
  "claimed_local_score": null
}}
```

claimed score 只作笔记，Broker 永远从原始 OOF 重算。候选中不要放数据副本、模型
checkpoint、凭证或软链接；若为 kernel 模式，必须在 Kaggle 上从官方挂载资产重新训练/推理。
Kaggle CLI 实际只上传 metadata 指定的一个 `code_file`，因此它必须是完全独立的
单文件脚本；不要让它 import 同目录的 helper。

# 计分 kernel 的硬约束——先按这些约束选方法，再写代码

**时间：本题计分 kernel 的平台时限是 30 分钟，含 wheel 依赖安装。** 这是端到端
的：安装依赖 + 读数据 + 训练 + 推理 + 写 submission 全在里面。超时会被 Kaggle 硬
杀，什么都留不下——不是低分，是零。所以方法选择要倒过来做：先问「这个方案能不能
在单卡 25 分钟内训完并推理完」，不能就换更小的骨干、更少的 epoch、更强的特征，
而不是先写完再发现跑不动。脚本开头读取环境变量 `IOAI_BUDGET_S`（Broker 注入的实
际墙钟预算），**先写出一个保底 submission 再开始迭代**，预算耗尽立即写出当前最好
结果。

**硬件：`accelerator` 只能是 `cpu` / `p100` / `t4`**，分别对应 Kaggle 的无加速器、
Nvidia Tesla P100（16GB）、Nvidia Tesla T4（16GB）。两者都是上一代卡，显存 16GB，
半精度吞吐远低于 A100/H100——**论文里"单卡几小时"的配置在这里跑不完**。

**GPU 位是全系统最稀缺的资源，比提交额度稀缺得多。** 整个 Kaggle 账号同时只有
**2 个 GPU 会话**，而当天三道题共用这一个账号；CPU 位有 5 个，几乎不排队。所以：

- 凡是 CPU 在 25 分钟内能跑完的方案，`accelerator` 一律写 `cpu`。错误占用 GPU 会
  直接堵住另外两道题。
- 只有确实需要 GPU 才写 `p100`/`t4`，并且要让它值得——一个占满 GPU 位却只提升
  0.001 的候选，代价是另一道题少交一次。
- 候选被回报为 `resource_deferred` 是**正常排队**，不是你的候选有问题，不要因此
  改方案或重写；等下一轮即可。

`estimated_kernel_minutes` 填保守的端到端运行时间（≤25），Broker 据此排队并阻止
来不及在截止前完成的 Kernel；**低报不会让你插队，只会让 kernel 被中途杀掉**。

**kernel 脚本开头必须有 8–10 个短段落的 Technical Report 注释块，且必须位于环境
setup block 之前。** 报告只描述该文件实际提交的解法：做了什么、为什么这样做、测得
什么；尽量量化模型、特征、超参数、验证设计、分数和运行时间。简述试过但未纳入的方
法及放弃原因，区分 OOF、organizer public split 与 Kaggle Public LB，不得编造测量。
不要描述 agent、harness、prompt、编排、选择器或 Broker。缺失或过时的报告会导致候
选在推送前被拒绝；`evidence/NOTES.md` 不算，因为它不会上传。

# 数据获取

`OFFICIAL_ASSETS/` 里已有一份只读快照。你也可以自己用 `kaggle` 命令补取，PATH 上
的是一个**只读网关**：`competitions download` / `files` / `list` / `leaderboard`
可用，`submit`、`kernels push` 一律拒绝并返回明确原因——提交由 Broker 统一执行，
它持有全题唯一的 50 次额度。凭证不在你的环境里，也不需要在。

只读取 `FEEDBACK.jsonl` 中属于本路线的榜单反馈。持续做“假设→最低成本实验→公共
标尺验证→保留/回退”的循环；不要在第一个可用解出现后结束。当前工作目录是
`{workdir}`。
"""


def continuation_prompt(lane: str) -> str:
    return f"""继续 `{lane}` 独立路线。先检查已有 artifact、SHARED_EVALUATION 和
本路线 FEEDBACK.jsonl；保护当前最佳候选，再尝试一个信息增益最高且与上次不同的
改进。若得到新候选，严格按 staging→rename→READY 协议发布；发布前确认 kernel
`.py` 的第一块内容是最新的 8–10 段 Technical Report，且只描述实际解法与实测结果，
不描述 agent/harness/prompt。不要提交 Kaggle，不要读取其他路线或 Search/HearSay
文件，也不要因为已有一个候选就停止。
"""
