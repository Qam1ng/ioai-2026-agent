# 两道 IOAI 实战的经验与教训

## 目录

1. [Task 2 Chasing the Robot](#task-2-chasing-the-robot)
2. [Task 1 Find the Order](#task-1-find-the-order)
3. [可泛化的成功经验](#可泛化的成功经验)
4. [必须吸取的教训](#必须吸取的教训)
5. [下一题的开局清单](#下一题的开局清单)

这些记录来自两次真实做题过程，只说明当时证据。不要把具体模型、分数或数据规律硬套到新题。

## Task 2 Chasing the Robot

任务是根据 8×8 网格、英文 mission、机器人身份与状态预测六类下一动作。最有效路线是把局部几何和语义状态显式编码，再用小型 Grid Transformer 学各 robot 的稳定行为。

已观测 Public 分：

| 候选 | 核心变化 | Public |
|---|---|---:|
| V1 | 单 seed Grid Transformer | 0.64861 |
| V2 | 两 seed、各自恢复 best checkpoint、概率平均 | 0.65083 |
| V3 | 三 seed best-checkpoint ensemble | 0.65222 |

V1 的验证峰值 epoch 10 为 0.648611，Public 为 0.64861，说明该 split 在这次任务上具有很强排序价值。epoch 11 验证回落；恢复 best checkpoint 比盲目使用最后一轮更可靠。

有效结构包括：对象/颜色 mask、source/destination、机器人与可通行单元的坐标关系、合法目标邻域、mission 模板语义、robot identity embedding 和 robot-specific output head。这类显式任务结构能让小模型在严格时限内学习。

V3 远端墙钟约 282.75 秒，而 timeout 为 300 秒。虽然成功，但安全余量过小；网络、队列或运行波动都可能让同一方案失败。多 seed 应基于单 seed 真实耗时预算，并预留至少 10% 或题目允许的更大余量。

标签不变量审计发现 carry 状态与 pick/drop 动作存在约束，但 V1 预测已自动满足。没有证据的硬 mask 不一定增益；先测违规率和条件精度，再决定是否加入规则。

这次最大失误是没有充分利用可用版本预算。三次强模型提交只探索了 seed ensemble，几乎没有对架构、loss、robot 分组、数据增强、规则后处理和更强解码做受控探针。应更早拿到 floor，再用若干低成本版本验证彼此正交的假设。

## Task 1 Find the Order

任务是将被打乱的双说话人音频片段恢复完整对话顺序。网络研究快速把问题映射到 AI4Code、sentence ordering、next-utterance selection 和拼图边缘匹配；共同结构是 pairwise 相对顺序/相邻打分加全局解码。

已观测 Public 分：

| 候选 | 核心方法 | Public |
|---|---|---:|
| floor | 简单合法排列 | 0.68969 |
| MFCC | 说话人聚类与交替约束 | 约 0.706–0.709 |
| W2V pairwise | 音频 embedding 相邻排序 | 0.73614 |
| Whisper-Qwen PMI | 转写语义 + beam | 0.78321 |
| seam RankNet | 边界连续性/排序融合 | 0.79645 |
| Qwen full-path alpha4 | 全路径语义重排 | **0.82575** |
| Qwen full-path alpha8 | 过强语义权重 | 0.70931 |

联网研究的直接价值很高：相似竞赛和 sentence ordering 文献提示 pairwise + global decoding；官方提供模型的 README 又明确支持 MFCC 说话人分离、Whisper 转写、Qwen 连贯性和 wav2vec 排序。若只做普通音频分类，很难自然走到最强 full-path 重排。

最强改进来自多层约束互补：MFCC 音色负责说话人，交替约束砍搜索空间，ASR/LLM 语义负责问答连贯，seam 特征提供局部声学证据，beam/full-path reranking 做全局组合。单个模型分数不如结构正确的 ensemble 和 decoder。

Public LB 控制变量实验很有信息量：alpha4 提升到 0.82575，但 alpha8 崩到 0.70931，说明“更依赖语义”并非单调更好。提交反馈适合校准权重和方法族，不适合逐样本猜标签。

本地验证不是万能的。部分 OOF 与 LB 对齐，例如 seam RankNet 约 0.796；另一些方案明显乐观，例如某候选 OOF 约 0.7303、LB 仅 0.7167，另一个研究方案 local 约 0.746、LB 约 0.72235。必须保留多个 split、分组诊断和不同 family 的线上探针。

强语义方案出现得太晚。搜索、转写缓存、全路径打分和远端 Kernel 工程应在开局尽早启动；否则即使最后找到正确方向，也没有时间做权重、beam、融合和 runtime 的充分迭代。

## 可泛化的成功经验

### 1. 先找问题同构，而不是先选模型

把任务映射成排序、匹配、路径、图、控制、结构化预测或检索问题，常比直接换大模型更重要。搜索相似任务、指标和 decoder 可以在前 10 分钟改变整个探索空间。

### 2. floor 与搜索必须并行

尽快获得一条合法 Public 反馈，验证提交格式、数据路径、Kernel 和 metric；同时让只读 subagent 搜方法。不要等研究报告完成才开始跑代码。

### 3. 显式结构与学习模型互补

把确定的约束用于特征、候选剪枝或解码，而不是全部交给模型隐式学习。先验证约束成立率，避免错误规则把上限锁死。

### 4. 训练与解码分开优化

排序题中的 beam/full-path，全局约束；控制题中的合法动作、robot-specific head；这些往往比增加模型参数更有效。

### 5. best checkpoint 与 ensemble 是低风险增益

验证可信时，保存每个 seed 的 best state，并在概率或未离散化分数层 ensemble。不要 ensemble 硬标签。先确认 runtime 和不同 seed 的误差确实有差异。

### 6. 让真实 submission 成为通信对象

分数、代码、metadata、日志和输出比口头汇报更有用。唯一 slug、简短 description 和立即归档能让 Codex/Claude 几乎零额外维护地共享有效工作。

### 7. 主动寻找官方带标签验证集

组织者提供的 validation/test_public 标签能显著提高本地实验吞吐：先用 train-CV 宽搜索，再用它做目标域 promotion gate，方案冻结后若规则允许可并入最终从零训练。它不是隐藏测试，必须记录查询次数、分组分和 local→LB 一致性；最终 Kernel 只能读取官方挂载资产，不能打包本地 checkpoint、标签副本或预测缓存。

## 必须吸取的教训

### 1. 不要浪费 version，也不要为了用满而盲投

20 次级别的预算应该覆盖：格式 floor、不同方法族、关键权重、local→LB 校准、ensemble/final。重复 SHA、没有假设的微调、未预检的失败 Kernel 都是浪费。

### 2. 不要只数 competition submissions

真实限制可能是 Notebook versions。每次 push 都要入账，远端失败也不能自动释放。下载/submit 失败只重试尾段，禁止盲目重新训练。

### 3. 不要把 runtime 推到硬上限

V3 的 282.75/300 秒是警告。提交窗口不仅包含模型运行，还包含 push、队列、输出下载、格式检查和 submit。最强候选必须提前完成。

### 4. 不要让两名主 Agent 写同一份代码

共享工作区会造成冲突、等待和责任不清。两条线独立，Kaggle submission 互通；采用同伴方案时复制已冻结版本并声明 parent。

### 5. 不要让 subagent 成为第三条失控提交线

subagent 适合网络侦察、规则审计、日志诊断、peer code review 和局部实验。默认只读，禁止 Kaggle 写操作；主 Agent 对合并、版本和提交负责。

### 6. 不要根据一条 Public 分数过度归因

一次变化可能同时改变 seed、训练时长、特征和 decoder。坚持单因素候选和可复现 hash；用多个相邻实验判断方向。

### 7. 不要把外部搜索变成外部数据注入

搜索只提供方法论。任何外部样本、标签、模型、embedding、预测或同源映射都不得进入 Kernel，除非当前 Rules 明确允许。

### 8. 不要在最后几分钟才处理 Report 和格式

Report 从首个候选起就写在 `.py` 顶部。每次都下载远端真实输出验证；不要假定本地合法就等于 Kaggle 输出合法。

### 9. Kernel 跑完不代表硬件提交链合规

一次 P100 事件直到 competition submit 才返回 4xx，单看 Kernel `COMPLETE` 无法发现。第一个有用 GPU floor 应尽早走完 push、运行、真实输出校验和 submit；失败时保留脱敏 HTTP 错误字段，不开 verbose、不记录 header，也不为取证重发提交。

### 10. GPU 并发必须由共享锁执行

文字约定“两条线各一张 GPU”不足以防竞态。所有题和 lane 应共享账号级槽位账本；Task 2 的已核验容量是 CPU 5、GPU 2。final 优先采用有等待者才让路的短 TTL 协议，既不让普通实验抢走决胜槽，也不长期空置 GPU。

## 下一题的开局清单

```text
[ ] 逐页读取 Rules/Overview/Evaluation/Data/Starter
[ ] 冻结 TASK_CONTRACT，确认 deadline、timeout、硬件、资产和 version 上限
[ ] 下载并哈希原始资产，验证 sample submission
[ ] 找出所有官方带标签 split，冻结用途、SHA、重叠检查和 training_use
[ ] Codex/Claude 各自启动不同 floor
[ ] 每方默认 2 个只读研究 subagent，全系统同时最多 4 个，10–12 分钟硬截止
[ ] 网络搜索覆盖同构任务、官方模型、global decoder、metric/CV
[ ] 20 分钟内获得至少一个合法远端分数
[ ] 第一个有用 GPU floor 走完 push→run→validate→submit 硬件冒烟
[ ] 两条 lane 共享 IOAI_ACCOUNT_ROOT 槽位账本；确认当前题 CPU/GPU 容量
[ ] 每个候选一个核心变化、一个唯一 slug、一次 push
[ ] 新 submission 触发同伴只读审计，不开会
[ ] 同时保留 best 与 diversity family
[ ] 用实测 p95 runtime 计算 latest_start
[ ] 截止前优先 submit 已完成 best，不等理想 final
```
