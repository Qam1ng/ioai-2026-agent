# RESEARCH_SYNTHESIS.md — 三路研究合并（R2 / R3 / R4）

> 重要声明：以下优先级是**研究建议**，基于外部证据质量与本题适配度，**不是本地实验验证过的模型排名**。除标注 [LOCAL_FACT] 的项外，任何方法在本题上的增益均未被验证。

## 0. 研究覆盖概览

| 路 | 责任轴 | 实际覆盖 | 报告状态 | 未完成项 |
|---|---|---|---|---|
| R2 | HF/AST 实现细节、变长与异常采样率音频工程、离线 kernel 工程 | 源码级深度覆盖（transformers main + 官方 ast 仓库）+ 本地 wav 格式实测 | 完整（146 行，Findings F001–F013，MC-1–MC-6） | SpecAugment/mixup 在 <1k 样本上的定量文献对比；Kaggle AST notebook 逐个复盘 |
| R3 | metric / 验证 / 泄漏 / 校准 / 后处理 | 全部 scope 覆盖 + 大量本地重算（含一个自做的负面实验） | 完整（169 行，F001–F013，MC-1–MC-6） | 未真正跑 AST embedding 近重复（受"不训练/不装依赖"边界限制），只给方案与正对照设计 |
| R4 | 类增量/低算力路线、不均衡、集成、传统兜底、负面结果 | 全部 scope 覆盖，含对 Charter 关键问题的**明确否定结论** | 完整（146 行，F001–F011，MC-R4-01–06） | 音频专用 TTA 的正面定量证据缺失（按"没找到"保守处理） |

三路均未检索本题标签/他人提交/赛后 solution；均未引入外部数据或权重。R2/R3/R4 的本地统计与阶段 A 一致或对其做了有证据的修正；阶段 B 已**独立复验**其中两条最关键的修正（wav fmt tag 分布、238/238 时长门控），结论一致。

历史教训目录本次为空（`context/prior_lessons/README.md`：本次运行没有提供历史教训目录）→ 无法引用具体条目；唯一一等实测教训来自系统背景：**音频族曾出现 LB +0.096 而本地 5-fold CV −0.127 的符号反转**。该教训与 R3-F007（ESC-50 同源片段泄漏使随机 CV 系统性乐观）、R4 负面结果 §6.4 相互印证，合并结论见 §5。

## 1. Canonical Findings（去重后）

**CF-1 [LOCAL_FACT] 本题不是 CIL 问题，而是 920 条小数据 29 类监督微调。**
旧类 296 + 新类 624 全部带标签、路径无交集。外部证据：CIL 综述把 "joint training over all seen data" 定义为一切增量方法的**上界**，最好的 exemplar 方法要到 memory≈1/3 数据才"接近" joint。→ LwF/EWC/iCaRL/蒸馏在本题只增加复杂度与双倍前向成本。
来源：R4-F001, R4-F002。置信度：高。

**CF-2 [EXTERNAL_EVIDENCE] `ignore_mismatched_sizes=True` 是本题最危险的静默陷阱。**
HF AST 头 = `layernorm(768) + dense: Linear(768,num_labels)`，logits 来自 pooler_output。用 `from_pretrained(num_labels=29, ignore_mismatched_sizes=True)` 会把整个 `classifier.dense` 判为 mismatched 并重新随机初始化 29×768 → 丢掉本 checkpoint **唯一被训练过的部分**，只打 warning 不报错。正确做法：手工替换 `classifier.dense`，前 16 行拷旧权重/偏置。`classifier.layernorm` 可完整保留。
来源：R2-F001, R2-F002（transformers main 源码）。置信度：高。

**CF-3 [EXTERNAL_EVIDENCE] 不得改 `max_length`/`num_mel_bins`。**
`position_embeddings` 是纯 `nn.Parameter(zeros(1, num_patches+2, 768))`，forward 只做加法、无插值/切片；`ASTPreTrainedModel._init_weights` 对 `ASTEmbeddings` 执行 `zeros_`（含 cls_token、distillation_token）。→ 改 max_length + ignore_mismatched_sizes 会得到全零位置编码/CLS 的"半毁"模型，loss 仍下降，极易误判正常。若必须缩窗，须按官方 `YuanGongND/ast` 取 `pos_embed[:,2:,:]` reshape (1,768,12,101) 后时间维**中心切片**（或 bilinear 插值）再拼回前 2 token。
来源：R2-F003, R2-F011。置信度：高。

**CF-4 [LOCAL_FACT + EXTERNAL_EVIDENCE] 音频加载必须走 soundfile/torchaudio，且 FE 不会替你重采样。**
本地昆虫文件含 fmt 3（float32）、fmt 0xFFFE 24-bit/16-bit；Python `wave.open` 直接抛 `unknown format: 3`（R2 实测，阶段 B 复现）。`ASTFeatureExtractor` 在 `sampling_rate != 16000` 时 raise，不传则静默用错；归一化为 `(x-mean)/(std*2)`（注意 ×2），pad 在归一化**之前**（零 pad 区归一化后 = 常数 +0.4670）；`is_speech_available()`（torchaudio 是否安装）决定走 `torchaudio.compliance.kaldi.fbank` 还是 numpy 复现路径，**两者数值不等价**。
来源：R2-F004, R2-F005。置信度：高。

**CF-5 [LOCAL_FACT] 时长门控是近确定的标签空间约束（覆盖 11% 测试集）。**
labeled 920 条中非 5.00s 文件 238 条 **100% 属 25–28**（precision 1.000），昆虫 240 条仅 2 条为 5.00s（recall 0.992）。测试集 40 条非 5s。→ 可把这 40 条的 argmax 限制在 {25,26,27,28}，或用软偏置。风险：测试端处理方式未知（有 1 条 11025Hz@5.0s 可疑样本），硬约束最坏损失 40/363≈11%。
来源：R3-F003（全量核验），R2-F009（时长分布），R4-F010（脆弱性警告）；阶段 B 已独立复算。置信度：本地统计极高；测试端有效性未验证。

**CF-6 [LOCAL_FACT] 采样率不是可靠的昆虫标记；24000Hz 是数据源分组键。**
≥192kHz 仅属 T. viridissima（labeled 17、测试 2，收益上限 ≈0.006）；96000 跨旧类(9)/新环境类(26)/昆虫(3)；24000Hz 仅见 16–24（labeled 83、测试 26）。
来源：R3-F005, R2-F008（R2 §2 另给出昆虫内部采样率按物种聚集的细分统计）。置信度：高。

**CF-7 [LOCAL_FACT + INFERENCE] 测试先验 ≠ labeled 先验。**
若分层，昆虫应占 ≈94 条且几乎都非 5s；实测测试仅 40 条非 5s → 测试昆虫占比推测 ≈11% vs labeled 26%。→ 直接反对以 labeled 频率做 balanced-softmax / 反频率加权（方向可能反）。
来源：R3-F004（+R3-F002 稀有类杠杆计算）。置信度：中高（推断，需探针确认）。

**CF-8 [EXTERNAL_EVIDENCE] metric 决定稀有类策略，且两种 metric 在此零和。**
sklearn macro-F1 = 每类 F1 无权重平均，测试集中出现但从未被预测的类 F1=0 仍计入 29 类平均。若测试按 labeled 先验分层，Sheep/Keyboard 期望各 ~1.2 条 → **macro-F1 下一条样本值 ≈0.0345，accuracy 下 0.0028，杠杆差 12.5 倍**。「为稀有类主动留几条预测」在 macro-F1 下期望为正、在 accuracy 下为纯亏损。
来源：R3-F008, R3-F002, R3-F010, R4-F006。置信度：高。

**CF-9 [EXTERNAL_EVIDENCE] 扩头后新类 logit 系统性偏高，需 post-hoc per-class bias；温度缩放无效。**
WA 指出灾难性遗忘的一个重要成因是最后 FC 层权重对新类高度偏置（用权重范数对齐修正）；BiC 把新旧类数据不平衡列为两大失败因素并用小平衡集学两参数偏置。本题正是 16→29 扩头、新类 624 vs 旧类 296、旧类含 3 条样本类，偏置方向可预测。**温度缩放对 accuracy/macro-F1/balanced-acc 全部无效**（argmax(z/T)=argmax(z)，数学恒等式）。
来源：R3-F009, R3-F010, R4-F006。置信度：高（机制）；量级需本地测。

**CF-10 [EXTERNAL_EVIDENCE] frozen 与 fine-tune 的定量分界落在本题类样本数两侧。**
PANNs 在 ESC-50（32 clips/类）：fine-tune 0.947、Freeze_L1 0.918、Freeze_L3 0.908、scratch 0.833；论文明言「<10 clips/类时用作特征提取器最好，样本更多时微调更好」。本题多数类 12–60 条/类（微调应赢 3–4 pts），但 Sheep(3)/Keyboard(3)/Rain(7) 落在 frozen 更优区间。AST 官方 ESC-50 recipe：**lr 1e-5**、batch 48、f/t masking → 95.75%；无 AudioSet 预训练则 ~88.7%（encoder 权重即分数，不能用大 LR 毁掉）。LP-then-FT（Kumar et al.）在强分布偏移下双优（FT: ID +2% / OOD −7%）。
来源：R4-F003, R4-F004, R4-F005。置信度：高（外部定量）；本题量级未验证。

**CF-11 [EXTERNAL_EVIDENCE + LOCAL_FACT] 变长推理必须存在，且多窗 logit 平均是标准做法。**
测试 40 条 5.3–54.3s、训练侧 240 条昆虫录音变长（最长恰 120.0s，上游截断）。训练随机裁 10.24s 窗（天然增强）、推理滑窗（hop≈5s，上限若干窗）对 logits 取算术平均。PSLA 证明模型聚合（checkpoint 权重平均 + ensemble）在音频 tagging 上稳定加分（0.444→0.474 mAP）且权重平均零推理开销。多窗聚合本身缺独立定量论文，属社区标准 [INFERENCE]。
来源：R2-F006, R2-F009, R4-F008, R2-MC-4, R4-MC-04。置信度：多窗高；集成中高。

**CF-12 [LOCAL_FACT + INFERENCE] checkpoint encoder ≈ 未被扭曲的原版 AudioSet AST。**
`./ast-freeze-enc` + 16 维头 + AudioSet 尺寸 pos_emb → 旧类知识几乎全在头权重 + 296 条数据里；给它换新头不会"丢失旧任务知识"，但旧头前 16 行是廉价热启动。推论：本题全量微调 ≈「AudioSet AST 首次下游微调」，可直接套 AST 官方 ESC-50 超参先验。
来源：R4-F009（+ R2 对 config/权重 shape 的核对）。置信度：中高。

**CF-13 [EXTERNAL_EVIDENCE] ESC-50 家族的同源片段近重复使随机 CV 系统性乐观。**
ESC-50 官方 README：clips 由 Freesound 长录音手工切出，且数据集预分 5 折并「确保同一源文件的片段落在同一折」。本题文件名已哈希 → fold/clip/take 信息全丢。边界：本题类计数（Thunderstorm 62 ≠ ESC-50 的 40/类）说明数据已重组，任何公开 fold 清单不可复用（也不允许引入）。
来源：R3-F007, R4-§6.4。置信度：高。

**CF-14 [LOCAL_FACT，负面结果] 廉价频谱指纹不能做近重复检测。**
R3 自做实验（256 维：前 5s、8kHz、8 段×32 频带 log 幅度、L2 归一化，1283 条 CPU ~1min）：labeled LOO 1-NN 仅 **0.363**；cos>0.95 的 labeled 对同标签率仅 **0.257**（>0.9 时 0.208）；测试→labeled 最大相似度中位 0.812。→ 既不能用它建组，也不能据此宣称无泄漏。替代：用题目自带 checkpoint 的 CLS embedding + 正对照（昆虫类同录音簇）/负对照（24000Hz 同源不同录音）校准阈值。
来源：R3-F006（实测）。置信度：高（作为负面结果）。

**CF-15 [INFERENCE] 统计分辨率与噪声底线。**
363 条测试 accuracy 的 95% 二项 CI 半宽 = 0.050(@0.60)/0.045(@0.75)/0.031(@0.90)；920 条 5-fold 单折 val≈184 条 → CI 半宽 0.06–0.07。→ **LB 与本地单折 <0.02 的差异视为噪声**；同一测试集上的成对比较（McNemar / 配对 bootstrap）灵敏度远高，所以 LB A/B 必须"同模型只改一个开关"。
来源：R3-F011。置信度：高（算术）。

**CF-16 [INFERENCE] 5s 音频约 51% 输入是常数 pad，checkpoint 就是在这种分布上训头的。**
5.00s → kaldi fbank ≈498 帧，pad 到 1024。→ 长录音推理若填满 1024 帧真实内容，与旧类训练分布不同。**可测选项**：对长录音也只取 5s 窗以保持 pad 结构一致，与"取满 10.24s"做 A/B。
来源：R2-F006。置信度：中（机制清楚，本题效果未测）。

**CF-17 [EXTERNAL_EVIDENCE] transformers v5 破坏性变更清单（与本题相关）。**
`from_pretrained` 默认 `dtype="auto"`（本 config 写 float32 故仍 fp32）；`Trainer(tokenizer=)` → `processing_class=`；已弃用 TrainingArguments 参数（如 `evaluation_strategy`）被直接移除；`XXXFeatureExtractor` 仅在视觉模型上被移除，`ASTFeatureExtractor` 保留。→ kernel 内用 4.x 写法会 TypeError。
来源：R2-F010。置信度：高。边界：checkpoint 声明 5.10.2、报告读的是 main(5.14.x)，未逐版 diff [UNRESOLVED]。

**CF-18 [LOCAL_FACT] starter 离线安装契约必须整段保留在提交 `.py` 顶部。**
`ioai_env-*.whl` 逐层 glob（depth≤7）；Kaggle 吃掉 `torch-2.13.0+cu126` 的 `+` → symlink 改名 + `UV_SKIP_WHEEL_FILENAME_CHECK=1`；uv 失败回退 pip（40s vs ~6min）；必须在任何相关 import 之前执行。路径一律 `rglob`，不硬编码。
来源：R2-F012, R4-F011。置信度：高。

**CF-19 [EXTERNAL_EVIDENCE] 传统特征的现实分数区间（兜底校准）。**
ESC-50 官方 leaderboard：MFCC+ZCR+RF 44.3%、SVM 39.6%、kNN 32.2%、human 81.3%、简单 CNN 64.5%、AST 95.7%、BEATs 98.1%。→ 本题（29 类、更少数据）传统特征预期 35–55%，远超随机 3.4%；若深度方案本地低于此，几乎必有 bug（sanity check 用途）。
来源：R4-F007。置信度：高（作为量级参考）。

**CF-20 [LOCAL_FACT + EXTERNAL_EVIDENCE] 超声不是主线。**
T. viridissima 鸣声基频 ≈10kHz + 20–60kHz 超声带 → 16kHz 重采样丢掉基频带；但测试集只有 2 条 >100kHz 样本，且该物种另有 41 条常规 44.1/48kHz 训练样本。收益上限 ≈2/363=0.006。
来源：R2-F007, R2-F008。置信度：高。

**CF-21 [INFERENCE] 算力与缓存预算。**
特征缓存后 AST-base 920 条单 GPU 约 1–2 min/epoch；瓶颈是解码/重采样 384kHz×120s 文件 + CPU kaldi fbank（全量 1283 文件预计 3–10 min CPU）。全量 fbank 缓存 fp32 ≈6.7GB / fp16 ≈3.4GB；`/kaggle/working` 有 20GB 上限且最终只应留 submission.csv。折中：每条录音缓存 K=3–5 个固定窗，训练时随机选一个（保留部分裁剪增强）。
来源：R2-F013, R2-MC-6。置信度：中高。

## 2. Canonical Method Cards（全部通过 kernel 可行性门）

按"研究建议"的执行顺序排列。每张卡合并了多路来源。

### CM-A 分类头保序扩展 16→29（前置条件，非可选）
- 机制：`from_pretrained(..., num_labels=16)` 正常加载 → 新建 `nn.Linear(768,29)` → `w[:16]=old.w; b[:16]=old.b`；新 13 行小尺度初始化（或按旧行范数做尺度匹配，避免新类 logit 天生偏小）→ 替换 `model.classifier.dense` → 同步 `config.num_labels/id2label/label2id` 与 `model.num_labels`。
- 解决：CF-2 的静默清零。
- 最小验证：**手术后不训练**，直接在 296 条旧类上评 top-1（限制在前 16 列），应与手术前持平（>0.8 量级）；显著下降 = 手术写错。
- 成本：秒级。失败模式：忘同步 `config.num_labels` → 保存/重载时头被重建为 16；依赖 `ignore_mismatched_sizes` → 静默清零。
- 合并来源：R2-MC-1, R4-F009（旧头热启动）。置信度：高。

### CM-B 统一音频加载/特征管线 + fbank 缓存
- 机制：`soundfile.read(path, dtype='float32', always_2d=True)` → 声道均值转 mono → `sr≠16000` 用 `torchaudio.functional.resample`（fallback `scipy.signal.resample_poly`；≥96kHz 分两级降采样如 384k→48k→16k）→ 交给 `ASTFeatureExtractor(..., sampling_rate=16000)`，保持 `max_length=1024`、`num_mel_bins=128` 不变（CF-3）。一次性把需要的窗口 fbank 算好存 fp16 缓存。
- 解决：CF-4（`wave` 崩溃、FE 不重采样、振幅尺度、后端不等价）、CF-21（CPU 瓶颈）。
- 最小验证：对每种出现过的 (fmt tag, bits, sr) 组合各取 1 文件，打印 mono float 的 max|x|、时长、重采样后长度（应 ≤1.0、时长不变）；并在 kernel 里 `print(transformers.utils.is_speech_available())`，两条后端路径都跑一次「旧类 top-1 探针」。
- 依赖未确认：wheel 内是否有 torchaudio/soundfile/scipy [UNRESOLVED #4]。
- 合并来源：R2-MC-3, R2-MC-6。置信度：高。

### CM-C 常量类 LB 探针（先测 metric 与测试先验，再做一切校准）
- 机制：提交「全部预测类 c」。accuracy → 得分 = n_c/363（反解 n_c）；macro-F1 → ≈(1/29)·2n_c/(363+n_c)；macro-recall/balanced-acc → 恒 0.0345 与 c 无关。建议 c=11（labeled 最大旧类）与 c=25（昆虫）各一次。
- 解决：[UNRESOLVED #1] metric、CF-7 测试先验。这是**唯一能直接观测 metric 的手段**，本地 CV 无论多好都答不了。
- 成本：2–3 次提交；kernel 只需写 CSV（无 GPU/模型）。
- 失败模式：题面已写明 metric（则不必花）；LB 只显示 public 子集（反解的是 public 计数，仍足以判 metric 族）；得分有效位不足时 macro-F1 显示 0.00xx 需注意精度。
- 时机：**必须在还有时间用其结论时花掉**，最迟剩余 60 分钟前完成。
- 合并来源：R3-MC-1。置信度：高（数学恒等式 + 官方 metric 定义）。规则风险：无（自己的提交反馈，不是标签泄漏）。

### CM-D Frozen-encoder 线性探测（第一小时基线，并可占坑提交）
- 机制：encoder 前向一次缓存 920 条 embedding（CLS 或 mean-pool），训 29 类 logistic/线性头（秒级收敛，可 L2 + 类均衡）；embedding 缓存后可免费无限重训头，适合扫 C 与 logit-adjust τ。
- 依据：CF-10（frozen 0.908–0.918 on ESC-50；<10 clips/类时 frozen 最优 → 覆盖本题 Sheep/Keyboard/Rain）、CF-12（encoder 是未扭曲的 AudioSet 特征；出题方自己就是用 LP 得到 16 类模型）。
- 最小验证：5 分钟内 embedding → LogisticRegression → 留出集 acc；应远高于 CF-19 的传统特征区间。
- 成本：前向 920 条数分钟（GPU）+ 训头 <10s。失败模式：昆虫超声降采样后 frozen 特征区分度不足（风险中低，四类各 60 条且种间差异大）。
- 合并来源：R4-MC-02（阶段一）。置信度：高。

### CM-E 旧+新数据联合微调 29 类（主路线）+ LP-then-FT 变体
- 机制：CM-A 扩头后，296 旧 + 624 新 = 920 条一起训；**lr 1e-5**（头可 10×）、3–10 epoch、f/t masking（SpecAugment 式）；长录音训练随机裁 10.24s 窗（与推理一致）。变体 **LP-then-FT**：先用 CM-D 的头初始化，再整体低 LR 解冻。
- 依据：CF-1（joint = CIL 文献上界，旧数据在场即无遗忘机制）、CF-10（AST ESC-50 recipe 95.75%；fine-tune>frozen 约 3–4 pts at ~30 clips/类；LP-FT 在强 OOD 下双优）、CF-12。
- 最小验证：留出每类 ≥1 条（3 条类留 1）快检：旧类 acc 不低于 CM-D 的 frozen 基线，新类 acc>80%。
- 成本：特征缓存 10–20min + 训练 10–30min，单 GPU 时限内绰绰有余。
- 失败模式：LR 过大毁 encoder（AST 官方用 1e-5，比常见 5e-5 低 1 个数量级；PANNs scratch vs FT 差 11.4 pts 说明特征即分数）；3 条样本类被淹没；昆虫 OOD 扭曲特征（若首次全微调后昆虫 LB 差于 LP 基线，应切 LP-FT 或降 encoder LR，**而不是加 epoch**）。
- 合并来源：R4-MC-01, R4-MC-02（阶段二），R4-F003/F004/F005。置信度：高（多来源一致 + 与本地资产对齐）；本题增益未验证。

### CM-F 变长推理：多窗 logit 平均（+ 权重平均/双模集成）
- 机制：>10.24s 样本取 hop≈5s 滑窗（上限如 8 窗），对 **logits 算术平均**后 argmax；训练侧随机裁窗保持一致。集成：同一 run 最后 N epoch 权重平均（SWA 式，零推理开销）+ 可选 LP 头与 FT 模型 logits 平均。
- 依据：CF-11、CF-16。mean vs max：昆虫鸣声通常持续 → 先用 mean；若目标声件稀疏，max 或 top-k mean 可能更优 [INFERENCE]。
- 最小验证（省提交额度的关键）：**只对测试集 40 条变长样本比较"首窗 vs 全窗均值 vs max"的预测差异条数**；若差异 <5 条，不值得花提交额度做 A/B。
- 附加可测项（CF-16）：长录音也只取 5s 窗（保持 51% pad 结构）vs 取满 10.24s。
- 失败模式：随机裁窗裁到静音（可改"选能量最高窗"）；同录音的窗跨折 → 验证虚高（见 CM-H）。
- 合并来源：R2-MC-4, R4-MC-04。置信度：多窗中高。

### CM-G 时长门控标签空间约束（元数据受控后处理）
- 机制：对 `duration>5.02s` 的测试样本，把 argmax 限制在 {25,26,27,28}（等价给 0–24 减一个大常数）。**保守版本**：只对 `dur>8s` 施加（测试 34 条），或用软偏置 δ≈3 而非 ∞。
- 依据：CF-5（labeled precision 1.000 / recall 0.992，全量核验）。
- 成本：≈0 算力 + 1 次提交做单变量 A/B。
- 失败模式：若测试昆虫被裁到 5s 或长文件里混入非昆虫类，硬约束最坏损失 40/363≈11%。R4 独立给出更保守立场：元数据捷径"只可作特征或 tie-break，不可硬覆盖"——两种立场都保留，用 1 次 LB A/B 区分（见 §7 冲突 C-1）。
- 合并来源：R3-MC-2, R2-F009, R4-F010。置信度：本地统计中高；测试端未验证。

### CM-H 泄漏控制：embedding 近重复检测 → group-aware 分折
- 机制：用题目自带 checkpoint 提取每文件（长文件多窗平均）CLS embedding → 余弦相似度建图 → 连通分量作 group → 整分量同折。阈值用正对照（昆虫类同录音簇）与负对照（24000Hz 子集：同数据源不同录音）校准。
- 依据：CF-13（官方 README 明言同源片段必须同折）、CF-14（必须用 embedding 而非粗指纹）。
- 最小验证：相似度直方图 + 连通分量 size 分布。若 size>1 的分量覆盖 >10% 样本 → 泄漏真实存在，随机 CV 绝对值不可信（排序或仍可用）；若几乎全 size=1 → 可放心用 repeated stratified split，把精力转回 CM-C/CM-I。
- 成本：一次全量前向（1283×1–3 窗），几分钟 GPU；仅开发/验证阶段用，不进最终推理路径。
- 失败模式：若 AST embedding 无法分离"同录音不同片段"与"同类不同录音"（正是粗指纹的失败方式），放弃精确分组，改为"保守假设 CV 乐观、只信 LB 与配对比较"。
- 合并来源：R3-MC-4。置信度：中（机制标准，本题可分离性未验证）。

### CM-I argmax 改变型校准：per-class logit bias
- 机制：`z'_y = z_y − τ·log π_y − δ·1[y≥16]`（可加 WA 式：把新类头权重范数缩放到旧类均值）。可选**计数匹配**：搜索偏置使预测类频率接近目标先验（目标先验由 CM-C 探针 + CF-7 估计）。τ 与 δ 是两个不同方向的偏置（长尾 vs 新旧组），**必须分别调，不要合成一个旋钮**。默认 τ=0（accuracy 假设下最优）。
- 依据：CF-8（metric 决定 τ 方向）、CF-9（新类 logit 偏高 + 温度缩放无效）。
- 最小验证：固定一个已训模型，在 repeated holdout 上画 accuracy(τ,δ) 与 macroF1(τ,δ) 两张表；预期两者最优点冲突 → 以 CM-C 判定的 metric 为准；LB 各 1 次提交确认 δ 符号。
- 失败模式：用 labeled 先验做 τ 时先验本身错（CF-7）→ 反向伤害；δ 过大压掉新类；在 3 样本类上 holdout 方差极大 → 选到噪声最优点。
- 合并来源：R3-MC-3, R4-MC-03, R3-F009/F010, R4-F006。置信度：机制高；本题增益不确定。

### CM-J 验证协议与"本地改进是否可信"的判据
- 机制：(1) repeated stratified 5-fold ×3（不同 seed），稀有类**手工**保证每折 val 至多 1 条、3 条类在 3 次重复里各出现一次（不要交给 sklearn 自动 stratify 后再看单折）；(2) 报三个分开的子指标：overall accuracy、旧 16 类 macro-recall、新 13 类 macro-recall（外加 macro-F1 备用）——遗忘只在旧类子指标上可见；(3) 决策：同一折集合上的**配对 bootstrap**，Δ 的 95% CI 含 0 则拒绝；(4) 单折/单 seed 差异 <0.02 一律噪声（CF-15）。
- 最小验证：先跑一次 baseline 的 3×5-fold，记录三个子指标的 seed 间标准差作为后续所有比较的噪声底线。
- 成本：3× 训练时间（可接受）。提交 kernel 只做单次训练；repeated CV 放开发阶段。
- 合并来源：R3-MC-5。置信度：高。

### CM-K LB 与本地 CV 冲突时的决策元规则
- 机制：把改动分两类。**(A) 结构性/低风险**（多窗聚合、group-aware 分折、时长约束、修 bug）：冲突时**信 LB**（本地 CV 在近重复+小样本下有已知方向的系统性乐观偏差，LB 无偏但方差大）。**(B) 高自由度/易过拟合 LB**（逐类阈值手调、按 LB 反推每类配额、只翻转 1–2 条样本）：冲突时**都不采用**，除非 LB 增益 >0.03。任何单变量 A/B 必须与 baseline 只差一处；提交日志表格化（改动 / LB / 本地三子指标），最终按"结构性改动优先 + LB 单调改进链"选提交。
- 依据：CF-15、CF-13、系统背景的 LB +0.096 / CV −0.127 反号实录。
- 合并来源：R3-MC-6, R4-§6.4。置信度：中高。

### CM-L 兜底阶梯（每级都必须能独立产出合法 submission）
- L0：读入 `submission.csv` 模板、target 填众数类，**不改行序/路径**（注意：以模板 `path,target` 为准，不用 starter 的 `id,prediction`）。<1min。
- L1：元数据规则（dur>5.02s → 昆虫类）。脆弱，只作 L2/L3 的特征补充或 tie-break，不单独当方案。
- L2：log-mel/MFCC 统计（mean/std/percentile）+ 时长/采样率元特征 → RandomForest/LogReg/LightGBM。CPU <5min，预期 35–55%（CF-19）。也是 sanity check：深度方案若低于此必有 bug。
- L3：frozen AST embedding + 线性头（=CM-D）或 NCM/kNN（每类 embedding 均值余弦最近原型；3 条样本类也有良定义原型；类内多模态时用 k=5 kNN 补）。
- 合并来源：R4-F010, R4-MC-05, R4-MC-06。置信度：高（作为保险的定位）。

## 3. 历史教训 vs 外部证据

本次运行未提供历史教训条目（目录仅含说明 README）。系统背景中的唯一一等实测教训——**音频族 LB +0.096 / 本地 5-fold CV −0.127 反号**——与两条外部/本地证据相互印证：CF-13（ESC-50 同源片段使随机 CV 系统性乐观）与 CF-14（廉价指纹查不出近重复，所以"看起来没重复"不能反驳）。合并结论：**在本题不得用本地 CV 的 ±0.02 内排序做决策；提交额度就是主要测量仪器**（CM-C/CM-K）。这与"外部论文分数排序"发生冲突时，以教训优先。

## 4. 各方法对本题的适配摘要

| 方法 | 本题适配 | 主要不确定性 |
|---|---|---|
| CM-A 扩头 | 必做前置；秒级 | 无（有源码依据） |
| CM-B 加载/缓存 | 必做；决定训练能否跑通 | wheel 依赖清单未知 |
| CM-C 常量探针 | 高价值、低成本，解决最大未知量 | 题面可能已写明 metric |
| CM-D frozen LP | 快速强基线 + 稀有类更稳 | 昆虫子集特征够不够 |
| CM-E 联合微调 | 主路线，文献上界 | lr/epoch、OOD 扭曲 |
| CM-F 多窗推理 | 只影响 40/363 测试样本，但必须正确 | mean vs max vs 5s窗 |
| CM-G 时长门控 | 覆盖 11% 测试集 | 测试端是否同源处理 |
| CM-H 组泄漏分折 | 决定本地 CV 可信度 | embedding 能否分离 |
| CM-I logit bias | 零成本、可回滚 | metric 与真实先验 |
| CM-J/CM-K 验证与决策 | 元规则，防止把噪声当改进 | — |
| CM-L 兜底 | 保证任何时刻有合法提交 | — |

## 5. metric / validation / leakage / post-processing 结论（汇总）

1. **先测 metric，再做任何 prior/bias 校准**（CM-C）。macro-F1 与 accuracy 在稀有类上零和；不确定时的折中（τ≈0.5）在两个 metric 下都不是最优，只是最小遗憾。
2. 验证：repeated stratified 5-fold ×3 + 三个分开子指标 + 配对 bootstrap；<0.02 差异视为噪声（CM-J、CF-15）。
3. 泄漏：假设存在同源片段近重复（CF-13）；用 embedding 建组（CM-H）；**不要用粗指纹**（CF-14）；长录音切窗必须同折。
4. 后处理：只做改变 argmax 的操作（per-class logit bias、计数匹配、时长门控）；**温度缩放无效**。
5. 提交预算（建议顺序）：2–3 次常量探针（metric + 先验）→ 1 次 baseline → 2 次单变量后处理 A/B（时长约束、δ）→ 1 次多窗聚合 A/B（仅当本地差异条数 ≥5）→ 其余留给配方/seed 迭代 + 最后 2 次保守回退。

## 6. 经典 / 非主流备选

- NCM / 原型分类（frozen embedding，零训练；3 条样本类也良定义；类内多模态时 kNN 补）——CM-L L3。
- MFCC/mel 统计 + RF/GBDT——CM-L L2，兼作 sanity check 下界。
- SWA 式最后 N epoch 权重平均 / LP 头与 FT 模型 logits 平均（PSLA 证据，几乎免费）——CM-F。
- 超声"时间扩展"（把采样率**声明**为 16kHz 等价慢放 1/12–1/24，把 20–60kHz 搬到 1.7–5kHz，或双分支 logit 平均）：机制可靠但**本题收益上限 0.006、训练侧仅 17 条同类样本、极易过拟合，且时间拉伸会改变脉冲率这一判别线索** → **不建议进主线**（R2-MC-5，明确降级）。

## 7. 冲突证据（保留，不投票删除）

**C-1 时长门控的施加强度**：R3 主张硬约束（labeled precision 1.000，覆盖 40 条测试样本）；R4 主张"只可作特征或 tie-break，不可硬覆盖"（测试有 1 条 8kHz、1 条 11025Hz 可疑样本，且规则可能在正式集失效）。证据质量：R3 的本地统计更强（全量 238/238）；R4 的风险论点针对的是**未观测的测试端处理方式**，本地无法反驳。**最小区分实验**：同一模型 ±硬约束各 1 次提交；若 metric 为 accuracy，差值应 ≈(被约束纠正条数−被约束破坏条数)/363，量级可辨。折中路径：先只对 `dur>8s`（34 条）或软偏置 δ=3。

**C-2 长录音的窗口长度**：CF-16 指出 checkpoint 是在"5s+51% pad"分布上训头的，暗示长录音也应只取 5s 窗；CM-F 的标准做法是取满 10.24s 多窗平均。两者未做过 A/B。**最小区分实验**：本地对 240 条昆虫训练录音做留出比较（5s 窗 vs 10.24s 窗 vs 多窗平均），三种都只在推理端切换。

**C-3 frozen vs 全量微调**：CF-10 的 PANNs 数据显示 ~30 clips/类时微调赢 3–4 pts，但 <10 clips/类时 frozen 赢；本题类样本数横跨 3–62，两个区间都占。CF-10 的 LP-FT 证据来自图像任务，量级不可直接搬。**最小区分实验**：CM-D 与 CM-E 各评同一留出集，分开看旧类稀有子集与新类子集。

**C-4 labeled 先验能否用于重加权**：R3-F004/CF-7 用测试时长分布推断昆虫仅占 ~11%，反对以 labeled 频率加权；R4-MC-03 仍把 τ·log π（π 取训练先验）作为可开关后处理。二者可共存：先用 CM-C 探针测真实先验，再决定 π 用 labeled 还是探针估计。**未解**，需探针。

**C-5 transformers 版本**：checkpoint 声明 5.10.2，R2 读的是 main(5.14.x)，未逐版 diff。以 kernel 内 `inspect.getsource(...)` 为准。

**C-6 submission 表头**：`path,target`（模板）vs `id,prediction`（starter 占位）。三方一致认为按模板，但需题面确认。

## 8. 未解决问题（合并去重）

1. 官方 metric（accuracy / macro-F1 / balanced accuracy）→ 优先题面，否则 CM-C。
2. 测试集是否含全部 29 类、public/private 划分与是否分层。
3. 测试昆虫是否被裁到 5s（决定 CM-G 安全性）。
4. wheel 环境依赖清单（torchaudio / soundfile / librosa / scipy / sklearn / LightGBM）与 `is_speech_available()` 结果 → 决定 CM-B 分支与 CM-L 选型。
5. checkpoint 当初的 fbank 后端与归一化细节 → 只能用旧类 top-1 探针反推。
6. kernel 时限与 GPU 型号（影响 repeated-CV 是否能进提交 kernel；建议提交 kernel 只做单次训练）。
7. 新类 logit 偏置 δ 的真实量级（需一次真实微调后测量）。
8. AST embedding 能否分离"同源片段"与"同类不同录音"（CM-H 正对照未跑）。
9. transformers 5.10.2 vs main 的 AST 逐版差异。
10. `training_args.bin` 精确超参（未反序列化）。
11. 音频 TTA 缺乏正面定量证据是"不存在"还是"没找到"（R4 按后者保守处理，结论：不投预算）。
12. SpecAugment/mixup 在 <1k 样本上的定量对比（R2 时间盒内未完成）。

## 9. 来源表（三路合并去重）

| 标题 | URL | 版本/日期 | 类型 | 支持 |
|---|---|---|---|---|
| transformers `modeling_audio_spectrogram_transformer.py` | https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/audio_spectrogram_transformer/modeling_audio_spectrogram_transformer.py | main @2026-08-04 | 官方源码 | CF-2, CF-3 |
| transformers `feature_extraction_audio_spectrogram_transformer.py` | https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/audio_spectrogram_transformer/feature_extraction_audio_spectrogram_transformer.py | main | 官方源码 | CF-4, CF-16 |
| transformers `configuration_audio_spectrogram_transformer.py` | https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/audio_spectrogram_transformer/configuration_audio_spectrogram_transformer.py | main | 官方源码 | CF-3 |
| transformers `utils/import_utils.py`（`is_speech_available`） | https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/utils/import_utils.py | main | 官方源码 | CF-4 |
| transformers v5.0.0 release notes | https://github.com/huggingface/transformers/releases/tag/v5.0.0 | v5.0.0 | 官方 notes | CF-17 |
| transformers releases v5.10.3–v5.14.1 | https://github.com/huggingface/transformers/releases | 2026-06~07 | 官方 notes | CF-17, C-5 |
| 官方 AST 仓库 `ast_models.py`（pos-embed 中心切片/插值） | https://raw.githubusercontent.com/YuanGongND/ast/master/src/models/ast_models.py | master | 官方论文代码 | CF-3 |
| AST 论文（Gong et al.） | https://arxiv.org/abs/2104.01778 | 2021 Interspeech | 原始论文 | CF-10 |
| AST 官方仓库 README（ESC-50 recipe lr1e-5, 95.75%） | https://github.com/YuanGongND/ast | 2022 | 官方仓库 | CF-10 |
| PANNs（Kong et al., Table XIII freeze vs FT vs scratch） | https://arxiv.org/abs/1912.10211 / https://ar5iv.labs.arxiv.org/html/1912.10211 | TASLP'20 | 原始论文 | CF-10 |
| Fine-Tuning can Distort Pretrained Features (LP-FT, Kumar et al.) | https://arxiv.org/abs/2202.10054 | ICLR'22 | 原始论文 | CF-10 |
| CIL survey（Masana et al.，joint = upper bound） | https://arxiv.org/abs/2010.15277 / ar5iv 全文 | TPAMI'22 | 综述 | CF-1 |
| WA: Maintaining Discrimination and Fairness in CIL | http://arxiv.org/abs/1911.07053 | 2019 | 原始论文 | CF-9 |
| BiC: Large Scale Incremental Learning | http://arxiv.org/abs/1905.13260 | 2019 | 原始论文 | CF-9 |
| Long-tail learning via logit adjustment (Menon et al.) | http://arxiv.org/abs/2007.07314 | ICLR'21 | 原始论文 | CF-8, CF-9 |
| PSLA（model aggregation / 权重平均） | https://arxiv.org/abs/2102.01243 | TASLP'21 | 原始论文 | CF-11 |
| ESC-50 官方 README（fold 同源规则 + leaderboard） | https://github.com/karolpiczak/ESC-50 / raw README | master @2026-08-04 | 官方数据集仓库 | CF-13, CF-19 |
| sklearn `f1_score` 文档（macro 语义、zero_division） | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html | stable @2026-08-04 | 官方文档 | CF-8 |
| Bushcricket 高频声传播 / T. viridissima 歌声频谱 | https://www.researchgate.net/publication/226456812_High-frequency_sound_transmission_in_natural_habitats_Implications_for_the_evolution_of_insect_acoustic_communication | 1990s | 研究论文（经摘要检索） | CF-20 |
| Xeno-canto T. viridissima 录音条目（频段旁证，未使用其数据） | https://xeno-canto.org/1155585 | 条目 | 数据库条目 | CF-20 |
| 本题原始资产（三 CSV + 1283 WAV header + model/* + ioai-starter.py） | 本地 `SEARCH_BUNDLE/ORIGINAL_ASSETS/` | 本次运行 | 一级本地证据 | CF-1, CF-4–CF-7, CF-12, CF-14, CF-18, CF-21 |

## 10. Merge Ledger（每个 Finding / Method ID 唯一去向）

### R2（Findings F001–F013，Method Cards MC-1–MC-6）
| ID | 去向 |
|---|---|
| R2-F001 | 合并 → CF-2（AST 头结构）+ CM-A |
| R2-F002 | 合并 → CF-2（ignore_mismatched_sizes 重初始化）|
| R2-F003 | 合并 → CF-3 |
| R2-F004 | 合并 → CF-4（`wave` 崩溃；阶段 B 已独立复现）|
| R2-F005 | 合并 → CF-4（FE 不重采样 / ×2 归一化 / pad 顺序 / 后端不等价）|
| R2-F006 | 合并 → CF-16 + 冲突 C-2 |
| R2-F007 | 合并 → CF-20 |
| R2-F008 | 合并 → CF-6 + CF-20 |
| R2-F009 | 合并 → CF-5（时长分布）+ CF-11 |
| R2-F010 | 合并 → CF-17 |
| R2-F011 | 合并 → CF-3（官方切片做法）+ CM-B |
| R2-F012 | 合并 → CF-18 |
| R2-F013 | 合并 → CF-21 |
| R2-MC-1 | 合并 → CM-A |
| R2-MC-2 | 合并 → CF-3 + CM-B（输入契约不变作为硬约束）|
| R2-MC-3 | 合并 → CM-B |
| R2-MC-4 | 合并 → CM-F |
| R2-MC-5 | 保留于 §6 非主流备选，**明确降级不进主线**（收益上限 0.006）|
| R2-MC-6 | 合并 → CM-B（fbank 缓存）+ CF-21 |

### R3（Findings F001–F013，Method Cards MC-1–MC-6）
| ID | 去向 |
|---|---|
| R3-F001 | 合并 → ASSET_MAP §2/§3 类别计数（与阶段 A 一致，核对通过）|
| R3-F002 | 合并 → CF-8（稀有类杠杆 12.5×）|
| R3-F003 | 合并 → CF-5（238/238，阶段 B 独立复算确认）|
| R3-F004 | 合并 → CF-7 + 冲突 C-4 |
| R3-F005 | 合并 → CF-6 |
| R3-F006 | 保留为 canonical **负面结果** CF-14 |
| R3-F007 | 合并 → CF-13 |
| R3-F008 | 合并 → CF-8 |
| R3-F009 | 合并 → CF-9 |
| R3-F010 | 合并 → CF-8 + CF-9 + CM-I |
| R3-F011 | 合并 → CF-15 |
| R3-F012 | 合并 → §5 提交预算顺序 + CM-C 时机约束 |
| R3-F013 | 合并 → TASK_ANALYSIS §3/§4（无官方 val、模板无标签、行序必须保留）|
| R3-MC-1 | 合并 → CM-C |
| R3-MC-2 | 合并 → CM-G（+ 冲突 C-1）|
| R3-MC-3 | 合并 → CM-I |
| R3-MC-4 | 合并 → CM-H |
| R3-MC-5 | 合并 → CM-J |
| R3-MC-6 | 合并 → CM-K |

### R4（Findings F001–F011，Method Cards MC-R4-01–06）
| ID | 去向 |
|---|---|
| R4-F001 | 合并 → CF-1 |
| R4-F002 | 合并 → CF-1（joint = 上界，**对 Charter 关键问题的否定结论**）|
| R4-F003 | 合并 → CF-10 + 冲突 C-3 |
| R4-F004 | 合并 → CF-10（lr 1e-5 锚点）|
| R4-F005 | 合并 → CF-10（LP-FT）+ 冲突 C-3 |
| R4-F006 | 合并 → CF-8/CF-9 + CM-I（并入"反对硬 class weights"负面结果）|
| R4-F007 | 合并 → CF-19 |
| R4-F008 | 合并 → CF-11 |
| R4-F009 | 合并 → CF-12 + CM-A（旧头热启动）|
| R4-F010 | 合并 → CM-L 兜底阶梯 + 冲突 C-1（元数据捷径的保守立场）|
| R4-F011 | 合并 → C-6（submission 表头冲突）+ CF-18 |
| R4-MC-01 | 合并 → CM-E |
| R4-MC-02 | 合并 → CM-D（阶段一）+ CM-E（LP-then-FT 变体）|
| R4-MC-03 | 合并 → CM-I |
| R4-MC-04 | 合并 → CM-F |
| R4-MC-05 | 合并 → CM-L L3（NCM/kNN）|
| R4-MC-06 | 合并 → CM-L L2（传统特征兜底）|

### 排除项（不进主总结）
- 无。三份报告中未出现纯重复到需删除、与任务无关或违规的 Finding；所有 ID 均有实质去向。R2-MC-5 虽被判定为低价值，但按保全原则保留在 §6 并标注"不建议进主线"。

### 负面结果索引（供后续 Agent 直接避坑）
1. `ignore_mismatched_sizes` 扩类 = 静默清零（CF-2）。
2. 改 max_length/num_mel_bins = position_embeddings/cls_token 被 zeros_（CF-3）。
3. `wave` 标准库在训练集上直接崩（CF-4）。
4. FE 不自动重采样；numpy 回退 ≠ torchaudio kaldi（CF-4）。
5. 温度缩放对硬标签 metric 完全无效（CF-9）。
6. 粗频谱指纹不能做近重复检测（CF-14）。
7. 采样率捷径几乎无收益且易误判 96kHz 的旧类样本（CF-6）。
8. labeled 先验 ≠ 测试先验 → 反频率加权方向可能错（CF-7）。
9. 硬 class weights 在 3-样本类上会让 3 条样本主导梯度（CF-8/CM-I）。
10. 大 LR 全量微调毁掉 AudioSet 特征（CF-10）。
11. 专门 CIL 算法（LwF/EWC/iCaRL）在旧数据全可用时无收益（CF-1）。
12. 对定长 5s 样本做重度 TTA 大概率是噪声，无 primary 定量证据（R4 §6.6 → CF-11 边界）。
13. 本地 CV 在 ±0.02 内的排序不可信（CF-13/CF-15 + 系统 LB/CV 反号实录）。
14. 超声"时间扩展"在本题收益上限 0.006 且会改变脉冲率（CF-20 / §6）。
