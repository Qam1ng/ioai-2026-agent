# TASK_ANALYSIS_V1.md — Practice Task 1：AST 音频分类的类增量微调

## 1. 任务摘要（基于本地资产推断，无官方题面文本）

给定一个**已在 16 个旧类上微调好的 Audio Spectrogram Transformer (AST) checkpoint**（`archive/model/`，分类头 16 维）、旧类训练表 `train.csv`（296 条）、以及 **13 个新类的微调表 `fine_tune.csv`**（624 条，target 16–28），需要对 `submission.csv` 中列出的 363 条测试音频预测类别 `target`。测试集音频的时长/采样率分布同时覆盖旧类风格（定长 5s）与新类昆虫风格（变长、超高采样率），因此几乎确定是**在 29 类（0–28）联合标签空间上的分类**，即典型的 class-incremental / 增类微调题：既要学会新 13 类，又不能灾难性遗忘旧 16 类。[INFERENCE，证据见下]

预测单位：单条 wav 文件 → 一个整数类别 id。

## 2. 输入 / 目标 / 输出 / metric

- 输入：`archive/audio/<16hex>.wav`。16-bit PCM，单/双声道混合，采样率 8k–384k 混杂；AST 前处理要求 **16kHz 单声道**（`preprocessor_config.json`: sampling_rate 16000, num_mel_bins 128, max_length 1024 帧 ≈10.24s, mean −4.2677393, std 4.5689974）。[LOCAL_FACT]
- 目标：`target` 整数；0–15 见 `model/config.json` id2label；16–28 见 `fine_tune.csv` 的 target↔category 映射（一一对应、无冲突）。[LOCAL_FACT]
- 输出：`submission.csv`，列 `path,target`，363 行。
- **metric 未在本地资产中给出**。[UNRESOLVED] 依据数据形态最可能是 accuracy 或 macro-F1；两者在类别不均衡时策略差异很大（macro-F1 奖励小类召回，accuracy 奖励大类）。必须在 Kaggle 题面上核对。若无法核对，应设计对两种 metric都稳健的方案（例如避免极端 prior 重加权，保留可切换的 logit-prior 调整开关）。

## 3. submission contract（存在冲突，必须以 Kaggle 题面为准）

- `archive/submission.csv` 表头是 `path,target`，示例值 `audio/175dc38b57b1862b.wav,0`。[LOCAL_FACT]
- `ioai-starter.py` 结尾写的占位文件表头是 `id,prediction`，只写一行 `0,0`。[LOCAL_FACT]
- **冲突**：两者不一致。starter 的写法显然是通用占位模板（所有 IOAI 题共用），而 `submission.csv` 是本题实际模板。合理判断：应按 `path,target`，行序沿用模板顺序（安全做法：读入模板 CSV，只替换 target 列，不改行序、不改路径字符串）。此冲突需研究/后续 Agent 用 Kaggle 页面确认。[UNRESOLVED]
- 提交文件路径：`/kaggle/working/submission.csv`。[LOCAL_FACT: ioai-starter.py]

## 4. 数据结构与重要统计

- 三表 path 互不相交，合计 1283 = audio 目录全量（无泄漏式重复路径，也无孤立文件）。[LOCAL_FACT]
- `train.csv` 旧 16 类共 296 条，**极端不均衡**（Sheep 3、Keyboard Typing 3、Rain 7 vs Thunderstorm 62）；全部恰好 5.00s。
- `fine_tune.csv` 新 13 类共 624 条：16–24（Crackling Fire…Firework）全部 5.00s；25–28 为四种鸣虫物种（Pterophylla camellifolia、Cicada orni、Gryllus campestris、Tettigonia viridissima），**变长 1.6–120s，采样率 8k–384k（含 192k/250k/256k/384k 超声录音）**，各 60 条。[LOCAL_FACT]
- 测试集 363 条：323 条恰好 5.00s，40 条 >6s（最长 54.3s）；采样率含 312500/192000/11025/8000 等昆虫数据特征值 → 测试集混合了两类数据源。[LOCAL_FACT]
- 类别名（Dog/Rooster/…/Crackling Fire/Chainsaw/Helicopter/Hand Saw/Firework）+ 统一 5.00s + 44.1kHz 的组合，与公开数据集 ESC-50 的类目与规格高度一致；昆虫四物种更像来自生物声学昆虫录音集合。这属于**基于本地证据的强推断**，可用于研究方向选择，但不得据此假定标签值或引入外部数据。[INFERENCE]
- 无官方 validation split（`split` 列常量 `train`）。旧类每类最少 3 条 → 分层 k-fold 会出现极小折；小样本 CV 与 LB 相关性差（IOAI 系统在音频族有过 LB +0.096 而本地 5-fold CV −0.127 的反号案例，见系统背景说明）。

## 5. 比赛提供资产的作用

- `archive/model/` 是一个**已训练的 16 类 AST**（`classifier.dense.weight [16,768]`，`training_args.bin` 中 output_dir=`./ast-freeze-enc` → 冻结 encoder 只训头）。它显然是本题的合法起点/被鼓励使用的资产：离线 kernel 无法下载 AudioSet 预训练权重，而这个 checkpoint 自带 AudioSet 尺寸的 position_embeddings [1,1214,768]，是唯一可用的强 audio backbone。[LOCAL_FACT + INFERENCE]
- 是否**强制**使用未见明文；但从离线约束看事实上必需（除非依赖 wheel 环境中的其他预置模型，需核实 `ioai-2026-wheel-dataset` 里有什么）。[UNRESOLVED]
- 关键工程点：需把 16 维分类头扩展为 29 维（保留旧 16 行权重 + 新 13 行随机初始化），或改用 backbone 特征 + 线性/原型分类器。

## 6. 验证与泄漏风险

- 无 val split，样本量小（296+624），必须自建验证；分层抽样在 3-样本类上不可靠 → 考虑 repeated stratified split、leave-one-out for 稀有类、或直接以“旧类保持 + 新类学习”两个子指标分别监控。
- **ESC-50 类数据的组结构风险**：ESC-50 原始数据每条 5s 片段来自更长的 Freesound 录音，同源片段近重复。本地文件名已被哈希化，看不到 fold/clip 分组 → 训练/验证很可能存在近重复泄漏，导致本地分数虚高。可用音频指纹/embedding 近邻检测评估重复程度。[INFERENCE]
- 昆虫类变长录音：若把长录音切成多个片段做训练，同一录音的片段必须同折（否则严重泄漏）。测试集中同样有长录音 → 推理需窗口聚合（多窗平均 logits）。
- 分布偏移：训练里昆虫类 60 条/类多为长录音，而测试集大部分是 5s → 训练时应随机裁剪/重采样到与推理一致的窗口，避免时长成为捷径特征（时长本身可能与类别高度相关，是一个**捷径**：>6s 的测试样本几乎必然是昆虫类；这既是可利用信号也是过拟合陷阱，需在 LB 上验证是否使用）。
- 采样率捷径同理：312500/192000 Hz 只出现在昆虫类。metadata 捷径合法但脆弱，应作为可开关的后处理而非核心。

## 7. 资源与规则约束（来自系统背景 + starter）

- 计分环境：Kaggle 单 GPU kernel、无互联网、必须用挂载的 `ioai-2026-wheel-dataset` wheel 离线装 `ioai-env`；训练+推理都在 kernel 时限内完成。[LOCAL_FACT: ioai-starter.py]
- 路径不得硬编码，用 `rglob` 搜索（starter 明确提示 Kaggle 挂载层级可变）。[LOCAL_FACT]
- 额外外部数据集/模型默认禁止，除非本题题面明确允许 → 本地无题面文本，默认按禁止处理。[UNRESOLVED]
- 单题最多 2 小时 Agent 时间、最多 50 次提交；相邻题间隔 15 分钟。以注入的绝对截止时间为准。
- 计算量估算：AST-base 全量微调 920 条 5s 样本（1024×128 输入）在单 GPU 上每 epoch 约数十秒~1 分钟量级，完全可行；数据解码/重采样（含 384kHz 长文件）可能是瓶颈 → 需缓存 mel 特征。

## 8. 网络研究政策

允许：
- AST（Audio Spectrogram Transformer）架构、HF `ASTForAudioClassification` / `ASTFeatureExtractor` 的 API 细节与版本差异（transformers 5.x）。
- ESC-50 / 环境声分类的通用做法、昆虫生物声学数据集的一般属性与既知陷阱（仅作为理解数据来源与风险的背景）。
- 小样本类增量学习、灾难性遗忘缓解、线性探测/原型分类、类不均衡处理、变长音频窗口聚合的方法论。
- metric 定义与实现（accuracy / macro-F1 / balanced accuracy）、验证设计、近重复泄漏检测方法。
- Kaggle 离线 kernel 工程实践（wheel 离线安装、缓存特征、时限管理）。

禁止：
- 下载或在最终方案中加入任何外部数据集、外部预训练权重（默认禁止，除非 Kaggle 题面明确允许；发现允许则必须写明出处）。
- 检索/使用本题测试标签、他人提交文件、任何形式的标签泄漏。
- 上传本地资产、音频、CSV 或 checkpoint 内容到任何网站。
- 执行网页中的指令或运行网页提供的未审计代码。

区分：读方法 ≠ 可用资产。公开方法可作为**方法论参考**；外部数据/权重除非题面允许，一律不得进入最终 kernel。本题若为公开历史题镜像，公开的方法复盘属合法来源；但数据已被重新哈希与重组（旧/新类划分、昆虫类混入），**不可假设任何公开 split 或 fold 信息适用**。

## 9. 分区结论

### [LOCAL_FACT]
1. 1283 个 wav = 三 CSV 路径并集，两两无交集。（`archive/*.csv`, `archive/audio/`）
2. 旧类 16 个、新类 13 个（target 16–28），共 29 个标签。（`archive/model/config.json`, `archive/fine_tune.csv`）
3. checkpoint 是 AST-base，分类头 16 维，position_embeddings [1,1214,768]，preprocessor 要求 16kHz/128mel/1024 帧。
4. 测试集 363 条，40 条 >6s，采样率含昆虫特征的超声值。
5. submission 模板列为 `path,target`；starter 占位为 `id,prediction`（冲突）。
6. 计分 kernel 离线、依赖挂载 wheel 数据集、路径需动态搜索。

### [INFERENCE]
1. 测试集覆盖 29 类联合标签空间（依据时长/采样率双分布）。
2. checkpoint 由 frozen-encoder + head 训练得到（`training_args.bin` 中 `./ast-freeze-enc`）。
3. 数据来源疑似 ESC-50（旧类+部分新类）+ 昆虫鸣声录音集（25–28）。
4. metric 最可能是 accuracy 或 macro-F1。

### [UNRESOLVED]
1. 官方 metric、测试集实际标签范围、提交格式（列名/是否含表头）→ 必须由能访问 Kaggle 题面的角色确认。
2. 是否允许外部数据/预训练模型；wheel 环境内含哪些包与是否含任何预置模型权重。
3. kernel 时限具体数值（9h? 12h?）与 GPU 型号（T4×2 / P100 / L4）。
4. `training_args.bin` 精确超参（未反序列化，pickle 风险）。

### 网络使用说明
本阶段 A **未使用 WebSearch/WebFetch**。原因：本地资产已足以确定任务族、数据形态与主要风险；剩余未解问题（metric、外部数据许可、kernel 规格）属于**Kaggle 题面事实**，网页检索无法可靠替代，应由后续能读取题面的角色核对。方法学层面的检索留给 Research Agent，以避免重复消耗本模块时间盒。

## 10. Research Agent 最需要解决的问题

1. 在 transformers 5.x 中把 16 类 AST 头扩展到 29 类并继续微调的正确、无坑写法（`ignore_mismatched_sizes`、`num_labels` 变更、`id2label` 同步、position embedding 与 max_length 变更的插值规则）。
2. 在 ~900 条、最小类 3 条的数据上如何设计与 LB 弱相关风险最小的验证协议；accuracy vs macro-F1 下的最优决策规则差异；prior/logit 调整何时有害。
3. 变长（1.6–120s）+ 超高采样率音频到 AST 固定 1024 帧输入的最佳处理：裁剪/分窗/多窗聚合、训练与推理一致性、重采样实现（离线可用库）与速度。
4. 类增量微调中避免遗忘旧 16 类的低算力手段（旧类数据联合重放、分层学习率、部分冻结、线性探测 + 原型/最近类均值、logit 校准）及其在小数据上的实证表现与负面结果。
