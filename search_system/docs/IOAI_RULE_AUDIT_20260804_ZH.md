# IOAI 2026 Search 模块规则核对（2026-08-04）

结论：没有发现需要改变“题目分析 A → 0–4 路并行研究 → 同一分析会话 B → Search Bundle”架构的信息。发现的是若干会影响 Prompt 精度、时间调度或后续提交模块的规则细节，已把与 Search 直接相关的部分修入共享前言。

## 已核对来源

- 参赛者 Discord `#general`：<https://discord.com/channels/1528040980664684634/1528625630231400589>
- IOAI 官方 Rules & Format：<https://ioai-official.org/wp-content/uploads/2026/06/IOAI%C2%B2-AI-Models-Track_Rules-Format.pdf>
- IOAI 官方 AI Models Track 仓库：<https://github.com/IOAI-official/IOAI-AI-Models-Track>
- 官方 Kaggle CLI 提交 Skill：<https://github.com/IOAI-official/IOAI-AI-Models-Track/blob/main/.agents/skills/kaggle-cli-submission/SKILL.md>
- 官方依赖清单：<https://github.com/IOAI-official/IOAI-AI-Models-Track/blob/main/kaggle-requirements.txt>

## 对现有理解的精确修正

1. 当前比赛安排以主办方给参赛者的正式日程为准：第一天三题同时开放，总窗口 6 小时，因此每题都可使用完整 6 小时。此前 Discord 讨论曾被本文误读为“三道题依次各 2 小时”；该理解已由参赛者确认不适用于当前赛制，不能再用于 Controller 调度。

2. 每题最多 50 次提交。Public Leaderboard 分数可以且应该作为方向性反馈；最终 Private Leaderboard 使用 Kaggle 的既定选择策略，不由人类临场挑选喜欢的提交。这不属于 Search 模块，但后续提交控制器不应把排行榜反馈视为禁用信息。

3. 对正式榜有效的提交，kernel 必须完成并在题目截止前执行 `kaggle competitions submit`；metric scoring 可以在截止后完成。仅在截止前 push、但 kernel 在截止后才完成，不足以进入正式榜。Search 模块不提交，但它的时间预算必须给后续 kernel 运行和 submit 留出余量。

4. 计分 kernel 关闭互联网。正式题除基础环境外，还会挂载组织者提供的固定 wheel 数据集，并要求提交代码运行给定安装脚本；Discord 给出的预期开销约 30–40 秒。Prompt 中“仅预装包”的旧说法过窄，已改为“基础环境 + 题目提供的固定依赖资产”。

5. 上传额外数据集或模型是按题目决定的限制；若当前题面没有明确允许，默认禁止。共享前言已改成这个默认拒绝、题面可例外的精确口径。Search Agent 可以阅读公开方法，但不能因此推导出外部权重或数据可进入最终 kernel。

6. 技术报告位于提交 `.py` 顶部的注释中，约 8–10 段，描述提交的 solution 而不是 Agent system，本身不计分。正式 starter/continuation prompt 会要求 Agent 编写；若遗漏，组织者提供赛后 30 分钟内的 late-submission 补救。它属于最终解法/提交阶段，不应混进 Search Bundle，因此本实现只在 IOAI 背景前言中说明，不新增 Report Agent。

7. 2026-08-04 检查官方 GitHub 仓库时，公开主分支只有 README、`kaggle-requirements.txt` 和 Kaggle CLI `SKILL.md`，没有可见的独立 `RULES.md`。因此 Runner 不假设本地存在某个全局 `RULES.md`；它要求题目分析 Agent直接读取本次任务资产中的官方题面和规则，并以这些资产为最高优先级。

## 对实现的影响

- 赛制背景只维护在 `SEARCH_PROMPTS_ZH.md` 的一份共享前言中。
- 三题各自启动一套 Controller，并共享同一账号级资源池；每次 Agent 调用仍使用 Runner 注入的绝对截止时间，不自行从本文推算预算。
- kernel 可行性门按“离线、单 GPU、基础环境 + 题目固定依赖资产、题面允许的外部资产”判断。
- Search 系统保存研究与分析证据，但不执行 Kaggle、排行榜试探、训练或技术报告提交。
