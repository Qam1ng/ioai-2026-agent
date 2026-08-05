from __future__ import annotations

from pathlib import Path


def direct_prompt(
    *, lane: str, slug: str, workdir: Path, deadline_minutes: float,
    submission_mode: str,
) -> str:
    model_name = "Codex + GPT-5.6 Sol" if lane == "codex" else "Claude Code + Fable 5"
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
  "accelerator": "p100",
  "estimated_kernel_minutes": 22,
  "purpose": "本候选相对上一个版本唯一改变了什么",
  "parent_id": "",
  "claimed_local_score": null
}}
```

claimed score 只作笔记，Broker 永远从原始 OOF 重算。候选中不要放数据副本、模型
checkpoint、凭证或软链接；若为 kernel 模式，必须在 Kaggle 上从官方挂载资产重新训练/推理。
Kaggle CLI 实际只上传 metadata 指定的一个 `code_file`，因此它必须是完全独立的
单文件脚本；不要让它 import 同目录的 helper。`estimated_kernel_minutes` 填保守的
端到端运行时间，Broker 会据此阻止来不及在截止前完成的 Kernel。**本题计分 kernel
的平台时限是 30 分钟(含 wheel 依赖安装),声明值必须留出余量(建议 ≤ 25);超过
平台时限的 kernel 会被 Kaggle 硬杀,什么都留不下。** 脚本开头读取环境变量
`IOAI_BUDGET_S`(Broker 注入的实际墙钟预算),先写出保底 submission 再迭代,预算
耗尽立即写出当前最好结果。

只读取 `FEEDBACK.jsonl` 中属于本路线的榜单反馈。持续做“假设→最低成本实验→公共
标尺验证→保留/回退”的循环；不要在第一个可用解出现后结束。当前工作目录是
`{workdir}`。
"""


def continuation_prompt(lane: str) -> str:
    return f"""继续 `{lane}` 独立路线。先检查已有 artifact、SHARED_EVALUATION 和
本路线 FEEDBACK.jsonl；保护当前最佳候选，再尝试一个信息增益最高且与上次不同的
改进。若得到新候选，严格按 staging→rename→READY 协议发布。不要提交 Kaggle，
不要读取其他路线或 Search/HearSay 文件，也不要因为已有一个候选就停止。
"""
