# RESEARCH_CHARTERS.md

```yaml
research_plan:
  n_research_agents: 2
  enabled_research_ids: [R1, R2]
  time_allocation_minutes:
    R1: 25
    R2: 25
  rationale: >
    任务族熟悉（AST 音频分类微调），资产自明，模型路线基本锁定（题目提供的 16 类 AST
    checkpoint 几乎是唯一合法底模），因此不需要 4 路。但存在两个高价值缺口：
    (a) bundle 缺题面，metric/规则未知，且历史实测表明音频任务本地 CV 可能与 LB 反号，
    metric/验证轴（轴3）价值最高；(b) 16→29 类扩展 + 混杂采样率 + 长音频 vs 10.24s 窗口
    的工程配方有真实不确定性（轴2）。最新 SOTA 论文轴（轴1）在离线 pinned-wheels kernel
    下价值最低，不启用；经典替代路线并入 R1 的次级问题。故 2 路各 25 分钟。
```

## R1 — 建模与工程配方轴（轴 2 + 轴 4 低算力路线）

- primary_scope：AST（audio-spectrogram-transformer）在几百~千级样本上的类扩展微调实践：如何把 16 类头扩到 29 类并保留旧类知识（扩展分类头保留旧行权重 vs 全新头 vs 旧头蒸馏）；冻结策略、学习率、epoch、batch、SpecAugment/mixup 等在 ESC-50/FSC22/InsectSet 规模数据上的实测配置；混杂采样率(8k–384kHz)→16kHz 重采样与 float32/24bit/extensible wav 的解码工程；长音频（>10.24s）训练裁剪与推理分块 + logits 聚合方案；单 GPU ≤2h 内训练+推理的算力预算；备选低算力路线（冻结 encoder 提特征 + linear probe/kNN/SVM）作为保底。
- 关键问题：1) 扩展 29 类头时保留旧 16 类权重是否实测更好？2) 全微调 vs 冻结底层的分界点（样本量 ~900）？3) 长音频最有效的聚合方式（mean-logits/max-prob/多窗投票）？4) 昆虫类超高采样率音频（能量集中在 >8kHz？）重采样到 16kHz 是否丢失判别信息，有无补救（如降速播放/带通）？
- primary sources：HuggingFace AST 微调官方教程与 issues；ESC-50 榜单方法、FSC22 论文（Forest Sound Classification）、InsectSet32/47 论文（尤其其基线对采样率的处理）；Kaggle 音频分类竞赛 write-up。
- forbidden_overlap：不研究 metric 定义、验证切分设计、LB 探测策略、提交策略（R2 负责）；不追逐无法装进 pinned 离线环境的新模型。
- 越界唯一条件：发现能显著改变题目理解或否定 R2 路线的重要证据（如发现官方题面声明必须冻结 encoder 或禁止修改模型结构）。

## R2 — 题面/metric/验证与泄漏轴（轴 3）

- primary_scope：第一优先：找到本题 Kaggle 竞赛页（slug 疑似 `ioai-2026-ai-models-track-practice-task-1`，或 IOAI 2026 practice task 公告）确认 metric 公式与方向、题面原文、外部资源与模型使用条款、test 类别空间说明。第二优先：小数据+近重复片段场景的验证设计：ESC-50 官方 5-fold 按源录音分组的惯例、FSC22/InsectSet 的推荐 split，同源片段泄漏如何造成 CV 虚高；不均衡（3 样本类）下 accuracy vs macro-F1 的策略差异、类先验/阈值后处理、何时值得用 LB 提交做假设检验（50 次预算的花法）。
- 关键问题：1) 本题 metric 到底是什么？2) train.csv 的旧类数据在评分中扮演什么角色（test 是否含旧类）？3) 怎样的本地验证协议在这种数据上与 LB 相关性最好？4) 若 metric 是 macro-F1，对 3 样本稀有类应采取什么预测偏置？
- primary sources：Kaggle 竞赛页与 discussion/rules 原文（最高优先）；IOAI 官方公告；ESC-50/FSC22/InsectSet 论文的 evaluation protocol 章节；关于 CV-LB 失配与小测试集方差的竞赛经验文。
- forbidden_overlap：不研究模型结构、微调超参、音频解码/重采样工程（R1 负责）。
- 特别禁令：即使找到本题镜像源数据集的公开标签，也只可用于理解 split/类结构，禁止将标签匹配到 test 文件或传递任何逐文件答案。
- 越界唯一条件：发现会显著改变题目理解或否定 R1 路线的重要证据（如题面宣布 metric 按新旧类分别加权、或明确允许外部数据）。
