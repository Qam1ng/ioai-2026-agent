# TASK_ANALYSIS_V1.md

## 1. 任务摘要

音频事件分类（IOAI 2026 AI Models Track practice task 1）。提供一个已在 16 个环境声类别（Dog…Rain, target 0–15）上微调好的 AST（Audio Spectrogram Transformer）checkpoint，外加 13 个新类别（Crackling Fire…4 种鸣虫, target 16–28）的带标签 `fine_tune.csv` 数据。需要对 363 个测试 wav 输出整数类别预测。本质是**把 16 类 AST 分类器扩展/微调为 29 类分类器**（类增量微调），并在离线 Kaggle kernel、单 GPU、pinned wheels 环境内完成训练+推理。

## 2. 输入 / 目标 / 输出 / metric

- 输入：`archive/audio/*.wav`（1283 个，采样率/声道/位深/时长高度混杂，见 ASSET_MAP）。
- 目标：每个测试 wav 一个整数 target。[INFERENCE] 取值空间为 0–28 共 29 类（test 含 ≈5s 短音频和长达 54s 的音频，后者与昆虫类 25–28 的时长特征一致，故 test 很可能同时覆盖旧 16 类和新 13 类；旧类是否出现在 test 未逐条证实，但 train.csv 提供旧类数据 + 模板列名与 train 相同，强烈暗示 29 类全空间）。
- 输出/submission contract：CSV，两列 `path,target`，363 行，path 顺序应与模板一致（保险做法：读入模板、按其 path 顺序填 target）；target 为整数。
- metric：[UNRESOLVED] bundle 无题面。最可能是 accuracy（Kaggle 分类练习题常用），也可能是 macro-F1。这是 Research 需要优先确认的问题（Kaggle 竞赛页 slug 疑似 `ioai-2026-ai-models-track-practice-task-1`）。metric 若为 macro-F1，稀有类（Sheep/Keyboard Typing 各 3 个样本）权重会被放大，策略差异巨大。

## 3. 数据结构与关键统计

- train.csv：296 行，16 旧类，**严重不均衡**（62 vs 3）。fine_tune.csv：624 行，13 新类，较均衡（24–60）。split 列全为 train，无官方验证集。
- 三个 CSV path 两两不相交，并集恰好 = 1283 个磁盘 wav。
- 数据来源特征 [INFERENCE]：0–15 类像 ESC-50 类别子集（5s 片段）；16–24 类（Axe/Chainsaw/Generator…）与公开 FSC22 森林声数据集类别高度重合（5s）；25–28 为鸣虫学名，与 InsectSet32/InsectSet47 类别重合（变长、超高采样率如 256k/384kHz）。即题目可能是多个公开数据集拼合的镜像。
- 音频读取陷阱 [LOCAL_FACT]：存在 float32、24bit、WAVE_FORMAT_EXTENSIBLE 文件，Python `wave` 会失败；须用 soundfile/torchaudio 统一 decode→mono→16kHz。

## 4. 提供模型的作用

- `archive/model/`：ASTForAudioClassification，16 类头，preprocessor 为标准 ASTFeatureExtractor(16kHz, 128 mel, 1024 帧)。position_embeddings [1,1214,768] 与 MIT/ast-finetuned-audioset-10-10-0.4593 结构一致 [INFERENCE]。
- training_args 字符串 `./ast-freeze-enc` [LOCAL_FACT] → 该 checkpoint 大概率是**冻结 encoder、只训头**得到的，encoder 仍接近 AudioSet 预训练权重，作为 29 类微调起点质量应不错。
- 是否强制使用该模型 [UNRESOLVED]（无题面）；在"外部模型默认禁止"的赛制下，它几乎肯定是唯一合法的预训练底模，解题应基于它。

## 5. 验证与泄漏风险

- 无官方 val：需自建分层验证。ESC-50/FSC22 风格数据同一源录音会切成多个 5s 片段 → 近重复样本跨 fold 泄漏风险高，本地 CV 会虚高。
- 系统级历史实测（见第 8 节）：音频任务族上排行榜 +0.096 的改动在本地 5-fold CV 中显示为 −0.127，**本地 CV 与 LB 可能弱相关甚至反号**，排行榜提交（最多 50 次）应作为一等测量资源刻意使用。
- train.csv 极端不均衡（3 个样本的类）+ metric 未知 → 校准/类先验后处理的收益需实测。
- 长音频（最长 120s train / 54s test）与 AST 10.24s 窗口不匹配：训练裁剪策略与推理分块聚合（mean/max logits）是显著自由度。
- 若题为公开数据集镜像，理论上可用公开标签对 test 文件做指纹匹配——这属于标签泄漏，**禁止作为资产使用**（见研究政策）。

## 6. 资源与规则约束

- 计分 kernel：无互联网、单 GPU、kernel 时限内完成训练+推理；必须在脚本顶部先运行 starter 的 `setup_ioai_env()` 从挂载 wheel 数据集离线装依赖（transformers 5.10.2 相关 pinned 版本）。
- 路径不可硬编码：用 starter 的 `find_input()`/rglob 定位 `train.csv` 等。
- 每题最多 50 次提交；本题 Agent 解题窗口最多 2 小时（以运行时注入的绝对截止时间为准）。
- 外部数据/模型默认禁止，除非题面明确允许 [UNRESOLVED：本 bundle 无题面，按默认禁止处理]。

## 7. 网络研究政策

- 允许：查 IOAI practice task 1 的 Kaggle 竞赛页（确认 metric、规则、题面文字）；ESC-50/FSC22/InsectSet 的**方法论**文献与公开代码；AST 微调/类扩展/长音频聚合/不均衡与验证设计的通用资料。本题为 practice（可能是公开题镜像），公开方法复盘合法且高价值。
- 禁止：下载或使用公开数据集的标签去匹配 test 文件（标签泄漏）；使用本题的泄漏标签、他人提交文件、逐字照抄的提交代码；上传原始资产内容。
- 外部资料只能作为方法论参考；任何外部数据/权重不得进入最终解法（除非后续找到题面明确允许）。
- 不明确处：metric、是否强制用提供 checkpoint、test 类别空间——研究优先解决。

## 8. 证据分区

[LOCAL_FACT]（均可由 ASSET_MAP 中路径复现）：三 CSV 行数/类分布/不相交性；1283 wav 与 CSV 完全对应；wav 格式混杂（含 fmt=3/65534，8k–384kHz）；train 全 5s、昆虫类最长 120s、test 最长 54.3s；模型为 16 类 AST，preprocessor 16kHz/1024 帧；training_args 含 `ast-freeze-enc`。

[EXTERNAL_EVIDENCE]：本阶段**未联网**（本地资产已足够界定任务；剩余疑点交给 Research Agent 更划算，且时间盒紧张）。系统 prompt 内置实测教训：音频任务族本地 CV 与 LB 反号案例（+0.096 LB vs −0.127 CV）；无条件注入外部知识为负收益（R&D-Agent 消融 35.1%→32.0%）。prior_lessons 目录本次为空（仅 README 声明无历史教训）。

[INFERENCE]：29 类全空间预测；数据源自 ESC-50 风格 + FSC22 + InsectSet 拼合；底模为 AudioSet AST；checkpoint 为冻结 encoder 训头产物。

[UNRESOLVED]：metric 公式与方向；题面/规则原文；test 类别覆盖；training_args 具体数值；是否强制使用提供模型。

## 9. Research Agent 最需要解决的问题（按优先级）

1. 本题 Kaggle 页面的 metric、题面、规则原文（尤其 accuracy vs macro-F1、外部资源条款）。
2. 小样本+近重复片段下的验证设计：如何切 fold 才能与 LB 相关；ESC-50/FSC22 的官方 fold/来源分组惯例。
3. AST 16→29 类扩展的最佳实践：扩头保留旧权重 vs 重训头；冻结层数、lr、epoch、SpecAugment/mixup 在几百样本规模的实测配置；长音频训练裁剪与推理分块聚合方案；单 GPU 限时内的算力预算。
4. 不均衡（3 样本类）下的损失加权/过采样/先验校正对目标 metric 的影响。
