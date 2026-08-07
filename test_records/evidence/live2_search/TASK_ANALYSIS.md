# TASK_ANALYSIS.md（最终题目理解，可独立阅读）

## 1. 任务目标

单标签音频事件分类，**29 类**。题目提供：
- 一个已在 16 个环境声类别（target 0–15: Dog…Rain）上微调好的 **AST（Audio Spectrogram Transformer）checkpoint**（16 类头）；
- 这 16 个旧类的 296 条带标签数据（`train.csv`）；
- 13 个**新类**（target 16–28: Crackling Fire / Axe / Chainsaw / Generator / Hand Saw / Vehicle Engine / Helicopter / Gunshot / Firework / 4 种鸣虫）的 624 条带标签数据（`fine_tune.csv`）。

需要产出对 363 个 test wav 的类别预测。本质是**把 16 类分类器扩展成 29 类（带完整旧数据 replay 的类增量微调）**，并让全部训练+推理在离线、单 GPU、pinned-wheels 的 Kaggle 计分 kernel 内跑完。

## 2. 输入 / 目标 / 输出 / 预测单位

- 输入：`archive/audio/<16hex>.wav`。采样率 8k–384kHz、单/双声道、PCM16 / float32 / 24-bit EXTENSIBLE 混杂；旧类与 FSC22 系新类全部恰好 5.0s，昆虫类 3–120s。
- 目标：整数 target ∈ [0, 28]。
- 输出 / **submission contract**：`/kaggle/working/submission.csv`，列 **`path,target`**（与 `archive/submission.csv` 模板一致），363 行，按模板 path 顺序填整数。
  - ⚠ **文档冲突（真实的 0 分风险）**：`ioai-starter.py` 的示例写的是 `id,prediction`。starter 是三题共用样板；**以本题模板 `path,target` 为准**，但下游必须在 kernel 内读取模板确认列名，若评分报 unexpected column 才回退 `id,prediction`。
- 预测单位：**文件级**（不是段级）→ 长音频多窗预测必须聚合成一个类别。

## 3. Metric

**[UNRESOLVED — 本题最大单点不确定性]** bundle 内没有题面/Evaluation 文本；R2 已验证：竞赛页 `kaggle.com/competitions/ioai-2026-ai-models-track-practice-task-1` 返回登录页（slug 存在，对照 `ioai-2026-practice-task-1` 为 404），Kaggle 公共 API 401，Google/Bing/DDG 对该 slug 零命中，IOAI 官网 `/ai-models-track/` 404。→ **公网上不存在本题题面、讨论或 solution**；任何声称"本题 metric 是 X"的外部材料都不可信。

可执行分支（不需要联网）：
- 若人类操作员粘贴的 starter prompt 或 kernel 内竞赛页含 Evaluation 文本，**该文本优先于本文件一切推断**。
- 否则可用 1 次提交做 metric 判别探针：全部预测最大类（Thunderstorm，估 test 占比 ≈0.066）→ accuracy≈0.066 / macro-F1≈0.004 / balanced-accuracy≈0.034，三者量级差 8–15×，一眼可辨（R2-F004）。该探针可与"端到端管道 + 列名契约验证"合并成同一次提交。
- 无论 metric 为何，**推理侧 post-hoc logit adjustment（`logit_c −= τ·log π_c`，τ∈{0,0.5,1}）应实现为可切换开关**：τ=0 近似最优 accuracy，τ≈1 近似最优 balanced-error / macro-F1 友好。

## 4. 数据结构与重要统计（[LOCAL_FACT]，全部在本机复算）

- 行数/类分布：train 296（16 类，62 vs 3 极端不均衡，Sheep 与 Keyboard Typing 各仅 3 条）；fine_tune 624（13 类，24–60，较均衡）；test 363。三者 path 两两不相交，并集 = 1283 wav。
- **无官方 val**（两个 CSV 的 `split` 列全为 `train`）→ 验证协议本身是一个自由度。
- test 时长：323 个 =5.0s；<5.5s 324；5.5–11s 9；≥11s 30；max 54.27s → **仅约 39–40/363（≈11%）超过 AST 单窗 10.24s**。
- test 采样率：44100×242, 48000×80, 24000×26, 96000×11, 11025/8000/312500/192000 各 1。
- **test 覆盖 29 类全空间**[INFERENCE，证据强]：非 5s 文件的采样率签名与 fine_tune 昆虫类一致（昆虫类 ≈40 个）；5s 文件的 (sr,ch) 分布只能由旧 16 类 + FSC22 系新类解释；各 (sr,ch) 签名的 test/train 计数比集中在 0.28–0.45，整体 363/920 = 0.395 → test 像是与训练集同池、按类分层随机抽出的 ~28–40%。**旧 16 类估计占 test 约 45%，绝不能只训新类。**
- 数据集拼接痕迹（阶段 B 逐文件证实）：Hand Saw 45/45 与 Chainsaw 41/45 是 24kHz（24kHz 只出现在这两类）；Wolf Howl 23/23 是 48kHz；Generator 45/45 是 44.1kHz；所有非昆虫类恰好 5.0s（"时长 >6s ⇒ 属 25–28 类"在 train 上 100% 成立）。这是**由题目自带数据推出的元数据先验，不是外部泄漏**，但属伪相关，只宜做 tie-break/诊断，不宜做主特征。
- 来源推测 [INFERENCE]：0–15 类 ≈ ESC-50 类子集；16–24 类 ≈ FSC22（森林声）；25–28 = InsectSet 系鸣虫。
- 解码工程：`wave` 模块在 float32(fmt=3) 上抛错；`soundfile.read(..., always_2d=True).mean(1)` + resample 可读全部 1283 个。**每个文件应包 try/except 并计数打印失败数**（解码崩一个文件可能整 kernel 失败）。

## 5. 比赛提供资产的作用

- `archive/model/`：AST（12 层、hidden 768、128 mel、1024 帧、fstride=tstride=10、position_embeddings [1,1214,768]、含 distillation token），16 类头。`training_args.bin` 中的 `./ast-freeze-enc` 表明该 checkpoint 是**冻结 encoder 只训头**的产物 → encoder 仍接近 AudioSet 预训练权重，**全量微调仍有 headroom；该字符串不是要求我们冻结的规则**。
- 硬性前端约束（阶段 B 在本地 transformers 源码直接验证）：
  1. **静默截断**：`_extract_fbank_features` 中 `difference<0 → fbank[0:max_length,:]`，超过 1024 帧（10.24s）的内容被**无声丢弃、不报错**——最容易被忽略的 silent bug；
  2. **8kHz 硬上限**：mel filter bank `min_frequency=20, max_frequency=sampling_rate//2=8000`（kaldi 尺度）→ 8kHz 以上信息在进入模型前就消失，与用什么库重采样无关；
  3. 不足 1024 帧走 ZeroPad2d 后再整体归一化 `(x−mean)/(2·std)` → padding 区变成常量 **+0.467**。provided checkpoint 是在 5s 文件（≈500 有效帧 + 524 常量帧）上训出来的，**短文件应保持同样的 zero-pad 模式**，改成循环填充属于需重训的 A/B，不是免费改进。
- `ioai-starter.py`：必须置于提交脚本最顶端先执行 `setup_ioai_env()`（离线 wheel 安装），且路径用 `find_input`/rglob 而非硬编码。
- 是否**强制**使用提供的 checkpoint：[UNRESOLVED]（无题面）。在"外部模型默认禁止"的赛制下，它实际上是唯一合法预训练底模。

## 6. 验证与泄漏风险

- **无官方 val**，必须自建。核心张力：源数据集（ESC-50/InsectSet）官方协议要求**按源录音分组**以防同源 5s 片段跨 fold（ESC-50 metadata 有 `src_file`/`take`；InsectSet 把同录音 snippet 视为一个样本），但**本题文件名是 hex 哈希，源信息被抹除**，无法直接复现分组。
- R2 的推断：test 是与训练集同池的分层随机抽样 → **train/fine_tune 与 test 之间同样存在同源近重复**，这份"泄漏"对我们有利且合法；因此**分层随机 CV 比 grouped CV 更接近 LB**，grouped CV 会系统性偏悲观、可能误杀对 LB 有效的改动。这与系统历史教训中"音频任务 LB +0.096 / 本地 5-fold CV −0.127"的符号反转机制一致（CV 协议比 test 更严格）。
- 推荐双协议（详见 RESEARCH_SYNTHESIS 的 MCh）：主 = 按 target 分层的随机 5-fold 或 20% hold-out；辅 = 用 AST embedding 余弦聚类近重复后的 GroupKFold，只作"我的 CV 有多虚"的乐观度诊断。
- 判别信号：若第一次真实提交的 LB **显著低于**本地分层 CV（差 >8%），说明 test 其实是按源录音整组留出的，应切换到 grouped 协议选型。
- 稀有类噪声：Sheep / Keyboard Typing 各仅 3 条训练样本、test 期望各 ~1 个 → 5-fold 中每 fold ≤1 个，per-class F1 方差极大，报告时必须标注"不可信"。
- LB 噪声定量：accuracy 1σ ≈ √(p(1−p)/n)，p≈0.85 时 n=363 → 1.9%，若 public 仅 30%（n≈109）→ 3.4%；macro-F1 因含 ~1 样本类，单样本翻转即 ±3.4%/类。→ **LB 差异 <3–4% 不构成决策依据**；50 次额度应买"大幅度、正交"的对照（metric 探针 / τ=0 vs τ=1 / 是否含旧类训练 / 分块聚合开关），而不是超参微调。
- 新旧类遗忘风险：训练集里旧类只占 32%（296/920），但旧类占 test ~45% → 最常见的失败是新类吞掉旧类，而整体 CV 会掩盖它。必须把验证分数**拆成 old-16 / new-13 两块**分别报告。
- 禁止的泄漏：即使识别出源数据集，也不得用公开标签对 test 文件做指纹/标签匹配。

## 7. 资源、规则与网络边界

- 计分 kernel：无互联网、单 GPU、kernel 时限内完成全部训练+推理；只能用题目挂载的 wheel 资产与基础环境。
- 算力估算 [INFERENCE]：1024×128 输入 → 1214 token（与 checkpoint 位置编码一致），约 ViT-B/16@224 的 6 倍序列长度；amp 下 920 样本 × 10–15 epoch ≈ 15–25 min；363 文件（含长文件约 450 窗）推理 <2 min；1283 文件特征提取 <3 min。→ **单模型训练+推理可安心进 kernel**；batch 48 在 1024 帧下大概率 OOM，用 batch 8–16 + grad accumulation。
- 外部数据集/预训练模型：**默认禁止**（本 bundle 无题面例外条款）→ 按禁止处理。
- Agent 解题窗口以运行时注入的绝对截止时间为准（不是"六小时"）；每题最多 50 次提交，现实上 2h 窗口只能用 5–10 次。
- 网络研究政策（沿用阶段 A，经 R2 复核后收紧）：允许查方法论、官方源码/文档、metric 与验证设计资料；**本题公网无题面/无 solution，不必再在该方向烧时间**；禁止用源数据集公开标签匹配 test、禁止使用他人提交文件；外部资料只作方法论参考，任何外部数据/权重不得进入最终解法。

## 8. 证据分区

**[LOCAL_FACT]**（路径见 ASSET_MAP，均本机复算）：三 CSV 行数/类分布/不相交性；1283 wav 与 CSV 精确对应；test 时长与采样率分布（323 个 5.0s、39–40 个 >10.24s、max 54.27s）；采样率↔类相关（Hand Saw/Chainsaw=24k 等）；非昆虫类全为 5.0s；wav 格式混杂且 `wave` 模块在 fmt=3 上失败、`soundfile` 全部可读；模型 config/preprocessor 全部字段；classifier.dense 为 [16,768]；position_embeddings 1214；`training_args.bin` 含 `./ast-freeze-enc`；starter 示例列名 `id,prediction` 与模板 `path,target` 冲突；`ASTFeatureExtractor` 的静默截断、8kHz mel 上限、pad-then-normalize 常量 +0.467（本地 transformers 源码验证）；test 2 个高采样率文件 >8kHz 能量占比 0.9994/0.9979。

**[EXTERNAL_EVIDENCE]**：官方 AST 仓库 ESC-50 recipe（lr=1e-5 with AudioSet 预训练、epoch 25、freqm 24 / timem 96、mixup=0、bal=none、第 5 epoch 起每 epoch ×0.85）；SpeechCommands recipe（3.5 万条才用 mixup 0.6）；HF audio-classification 文档（lr 3e-5、bs 32、10 epoch）；ESC-50 官方 5-fold 按 src_file 分组；InsectSet（PLOS CB 2023）的 5s/3.75s-overlap 分段、短文件循环填充、同录音视为一个样本、并明确指出 mel 前端对超声昆虫不利；logit adjustment（Menon et al., ICLR 2021）；F1 阈值理论（Lipton et al. 2014）。完整 URL 见 RESEARCH_SYNTHESIS 来源表。

**[INFERENCE]**：test 覆盖 29 类且为同池分层随机抽样、旧类占 ~45%；每类 test 期望数（Thunderstorm ~24、FSC22 各 ~18、昆虫各 ~24、Sheep/Keyboard 各 ~1）；底模为 AudioSet AST；扩头保留旧行是免费强先验；算力预算；分层 CV 比 grouped CV 更贴近 LB。

**[UNRESOLVED]**：① metric 公式与方向（最高优先）；② public/private LB 比例与是否全量计分；③ 是否强制使用提供 checkpoint、是否允许外部资源；④ 扩头保旧行 vs 全新头、全微调 vs 冻结前 N 层的定量差异（无同设定 primary 证据）；⑤ mean-logits vs max-prob vs 投票的优劣；⑥ pinned wheel 是否含 sklearn/librosa/torchaudio/soundfile；⑦ `training_args.bin` 具体数值；⑧ test 是否真有 Sheep/Keyboard Typing 样本（期望 ~1，可能为 0）；⑨ FSC22 类的同源片段比例（论文 403）。

## 9. 阶段 A 被修正的内容与证据

| 阶段 A 说法 | 修正 | 证据 |
|---|---|---|
| test 是否含旧类 = UNRESOLVED | 升级为"证据强的 INFERENCE"：含旧类且占 ~45%，昆虫 ~40 个 | test 时长/采样率签名逐文件统计（R2-F002/F003/F013，阶段 B 复核） |
| "长音频必须裁剪/分块" | 机制更准确：**不处理不会报错，会静默丢弃 10.24s 之后一切**；且只影响 ~39/363 | transformers 源码 `fbank[0:max_length,:]`（阶段 B 本地验证）+ test 时长统计 |
| submission 只写了 `path,target` | 补充 starter 示例写 `id,prediction` 的冲突与回退方案 | `ioai-starter.py` 末尾 vs `archive/submission.csv` 表头 |
| 未提 mel 8kHz 上限 | 新增硬约束：mel max_frequency=8000，超声信息在前端即丢失 | 本地 `ASTFeatureExtractor.__init__` 源码 |
| 昆虫超声只是猜测 | 量化：test 2 个文件 >8kHz 能量 99.8%+；train Tettigonia 部分文件 hi8k=0.998 但并非全部（0.005–0.998 跨度大） | 阶段 B 亲自 FFT 复核 |
| "R1 报告：padding 常量 −0.467" | 符号纠正为 **+0.467** | `normalize = (x−mean)/(2·std)`，(0+4.2677)/9.138 |
| 阶段 A 曾把 CV 设计倾向于按源录音分组 | 反转为"主用分层随机 CV，grouped 仅作诊断" | R2-F003/F005/F006/F010 + 系统历史教训的符号反转机制 |
