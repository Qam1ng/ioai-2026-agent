# R4 Research Report — 经典方法 / 低算力路线 / 反例与负面结果

## 0. Top-5 可行动方向

1. **旧+新数据联合微调 29 类 AST（主路线）**：CIL 文献明确把 joint training on all seen data 定义为一切增量算法的上界（Masana survey §5），本题旧类数据 100% 可用 → 任何专门 CIL 算法（LwF/EWC/iCaRL）都是浪费预算。最小验证：扩头到 29 维（保留旧 16 行权重）、lr≈1e-5、3–5 epoch，先看旧类 holdout 是否保持。
2. **Frozen-encoder 线性探测作为第一小时快速基线**：官方 checkpoint 本身就是 frozen-encoder+头训出来的（`./ast-freeze-enc`）→ encoder 是未被扭曲的 AudioSet 特征。PANNs 实证：ESC-50 上 freeze 90.8–91.8% vs fine-tune 94.7%（≈32 clips/类时差 3–4 pts；<10 clips/类时 freeze 反而最好）。最小验证：缓存 CLS embedding，训 29 类 logistic/线性头，几分钟出分即可提交占坑。
3. **LP-then-FT 顺序**（先训好 29 类头再解冻低 LR 微调）：Kumar et al. 证明直接全量微调会在 OOD 数据上扭曲预训练特征；本题昆虫超声录音相对 AudioSet 是强 OOD。LP-FT 兼得两者，且工程上 = 路线 2 自然升级到路线 1。
4. **变长音频多窗 logits 平均 + 轻量集成/权重平均**：测试集 40 条 >6s（最长 54.3s）必须分窗聚合；PSLA 证明 checkpoint 权重平均与 ensemble 在音频上稳定加分且近乎免费。避免花哨 TTA（见 §6）。
5. **兜底阶梯必须先落袋**：(a) 恒定预测+模板行序合法 submission → (b) MFCC/mel 统计+RF/GBDT（ESC-50 上该类方法 ~44%，能拉开与随机 3.4% 的差距）→ (c) frozen embedding+NCM/kNN。每级都能在训练失败/超时时保住合法提交。

## 1. Charter 与实际覆盖范围

按 Charter 覆盖了：CIL vs 联合微调证据（含否定性结论）、LP vs FT 的定量分界、类不均衡（logit adjustment vs 重加权）、便宜集成/TTA、传统特征兜底与 ESC-50 分数区间、负面结果收集。未覆盖：AST API 细节（R2）、metric/验证协议（R3）。所有本地事实亲自用 csv 复核（类分布与 ASSET_MAP 完全一致）。

## 2. 对题目分析的核对或修正

- **核对通过**：train.csv 296 条（每类 3–62）、fine_tune.csv 624 条（24–60/类）分布与 ASSET_MAP 一致（本地重新统计）。
- **补强而非修正**：TASK_ANALYSIS §5 说 checkpoint 是 frozen-encoder 训头产物 [INFERENCE]。R4 推论其重要推论：**该 checkpoint 的 encoder 权重 ≈ 原版 AudioSet 预训练 AST，从未被下游任务扭曲**。因此 (a) 它的特征质量与 AST 论文/HEAR 报告的 AudioSet 特征等价；(b) 全量微调本题 = 「AudioSet AST 首次下游微调」，可直接套用 AST 官方 ESC-50 recipe 的超参先验（lr 1e-5、batch 48、freq/time masking），该 recipe 5-fold 95.75%。[INFERENCE，证据链见 R4-F004/F009]
- 无发现题目分析错误。

## 3. 检索方向与来源覆盖

- CIL joint-training 上界：Masana et al. 2020 survey（ar5iv 全文检索 "joint … upper bound"）✔
- LP vs FT 定量：PANNs (Kong et al. 2019) ESC-50 迁移表 XIII + few-shot 曲线 ✔；LP-FT 理论 (Kumar et al. 2022) ✔
- AST 在 ESC-50 的官方分数与 recipe：AST 论文 + YuanGongND/ast README ✔
- 类不均衡：logit adjustment (Menon et al. 2020) ✔
- 集成/aggregation：PSLA (Gong et al. 2021) ✔
- 传统特征分数区间：ESC-50 官方 README leaderboard（MFCC+RF 44.3%、人类 81.3%、CNN baseline 64.5%、AST 95.7%、BEATs 98.1%）✔
- 未再扩展：音频专用 TTA 论文检索边际收益低（该方向公开定量证据稀少，已按 [INFERENCE] 标注），把时间留给核对。

## 4. 关键 Findings

- **R4-F001 [LOCAL_FACT]** 旧 16 类训练数据完整可用（296 条），新类 624 条，合计 920 条全部带标签且路径无交集。→ 本题**不是**受内存约束的 CIL 问题，而是普通的小数据 29 类监督微调问题。来源：`archive/train.csv`/`fine_tune.csv` 本地统计。
- **R4-F002 [EXTERNAL_EVIDENCE]** CIL 综述（Masana et al.）把 "joint training over all seen data" 明确作为所有增量方法的 **upper bound**；表现最好的 exemplar 方法（IL2M/EEIL）在 memory 达到 1/3 数据时才"接近"joint。→ 对 Charter 关键问题的**明确否定结论**：没有任何证据表明当旧数据 100% 可用时专门 CIL 算法能胜过联合微调；LwF/蒸馏/EWC 在本题只会增加复杂度与调参风险。边界：该结论基于图像基准，但机制（遗忘源于数据不可得）与模态无关。
- **R4-F003 [EXTERNAL_EVIDENCE]** PANNs（CNN14, AudioSet 预训练）迁移 ESC-50（每折 32 clips/类）：fine-tune **0.947**、Freeze_L3 **0.908**、Freeze_L1 **0.918**、from scratch **0.833**；且论文明确指出 "Using a PANN as a feature extractor achieves the best performance when **fewer than 10 clips per class**; with more training clips, the fine-tuned systems achieve better performance."。→ 本题多数类 12–60 条/类（微调应赢 ~3–4 pts），但 Sheep(3)/Keyboard(3)/Rain(7) 落在 frozen 更优区间 → 支持 LP-then-FT 或"微调后仍保留 frozen 头做对照"。
- **R4-F004 [EXTERNAL_EVIDENCE]** AST 官方 ESC-50 recipe：AudioSet 预训练 + lr **1e-5**、batch 48、f/t masking，5-fold **95.75%**（论文报 95.6±0.4）。→ 本题全量微调的超参锚点；也提示不用 AudioSet 预训练时 AST 掉到 ~88.7%（论文），即 encoder 权重是最大资产、不能用大 LR 毁掉。
- **R4-F005 [EXTERNAL_EVIDENCE]** Kumar et al. (LP-FT)：当预训练特征好且下游有分布偏移时，全量微调 ID 平均 +2% 但 OOD −7%，LP-then-FT 同时优于两者。→ 本题昆虫类（8k–384kHz 重采样后的超声鸣声）相对 AudioSet 分布是强 OOD，测试集又混合两种来源 → 直接激进全微调可能伤及与训练分布不一致的测试子集；LP-FT 是低成本保险。边界：其实验为图像任务，量级不可直接搬。
- **R4-F006 [EXTERNAL_EVIDENCE]** Menon et al.：基于类频率的 **post-hoc logit adjustment**（logits − τ·log π）统一并优于常见 re-weighting/margin 方法，且可训后免费开关。→ 类不均衡处理应首选"训练时均衡采样 or 什么都不做 + 推理时可开关的 logit 调整"，在 LB 上 A/B，而不是硬 class weights（3 条样本的类权重放大 20 倍是典型崩坏源，见 §6）。注意：logit adjustment 假设测试先验均匀——本题测试先验未知（metric 也未知），必须靠提交验证。
- **R4-F007 [EXTERNAL_EVIDENCE]** ESC-50 官方 leaderboard 分数区间：MFCC+ZCR+RF **44.3%**、SVM 39.6%、kNN 32.2%、人类 81.3%、简单 CNN 64.5%、AST 95.7%、BEATs 98.1%。→ 兜底方案的现实预期：传统特征+GBDT 在本题（29 类、更少数据）大约 35–55% 量级，仍远超随机（1/29≈3.4%），是合格的"训练失败保险"和 sanity check（若深度方案本地分低于此，必有 bug）。
- **R4-F008 [EXTERNAL_EVIDENCE]** PSLA：音频 tagging 上 **model aggregation（checkpoint 权重平均 + ensemble）** 是被系统验证的涨分技巧（单模 0.444 → ensemble 0.474 mAP），成本≈0（权重平均不增推理成本）。→ 本题 2h 内可行的集成形式：同一训练 run 最后几个 epoch 权重平均（SWA 风格）、或 LP 头+FT 模型 logits 平均。
- **R4-F009 [LOCAL_FACT+INFERENCE]** `training_args.bin` strings 含 `./ast-freeze-enc`、分类头恰 16 维、position_embeddings 为 AudioSet 尺寸 [1,1214,768] → checkpoint 的 encoder 即原版 AudioSet AST 权重（未被 16 类任务改动）。推论：给它换任何新头都不会"丢失旧任务知识"，旧类知识全部在 296 条数据 + 16 维旧头权重里。旧头权重可用于初始化 29 维新头的前 16 行（廉价热启动）。
- **R4-F010 [INFERENCE]** 兜底阶梯（保证任何时刻有合法 submission）：L0=读入 `submission.csv` 模板、target 填众数类（不改行序/路径）→ L1=时长/采样率元数据规则（>6s 或 SR∈{192k,250k,256k,312.5k,384k,8k,11k} → 昆虫类先验）→ L2=librosa/torchaudio mel 统计量+RF/LogReg → L3=frozen AST embedding+线性头。每级 <5 min 可产出。L1 是脆弱捷径，只作 L2/L3 的特征补充，不单独当方案。
- **R4-F011 [LOCAL_FACT]** 本地复核 ioai-starter.py：确认占位写出为 `id,prediction` 单行，与模板 `path,target` 冲突（同 TASK_ANALYSIS §3）；starter 另一强约束是 wheel 离线安装必须在 import 前执行、路径用 glob 搜索。→ 兜底 L0 必须基于模板 CSV 而非 starter 占位格式。

## 5. Method Cards

### MC-R4-01 联合微调 29 类（full replay joint fine-tune）【主推荐】
- **家族**：标准监督微调（=CIL 文献中的 Joint 上界）。
- **机制**：29 维新头（前 16 行用旧头初始化，后 13 行零/随机初始化），旧 296+新 624 共 920 条一起训；lr 1e-5（头可 10×），3–10 epoch，SpecAugment 式 f/t masking。
- **解决问题**：同时学新 13 类且零遗忘旧 16 类（旧数据在场即无遗忘机制）。
- **证据**：R4-F002（joint=上界）、R4-F004（AST ESC-50 recipe 95.75%）、R4-F003（fine-tune>freeze at ~30/类）。
- **适配**：920×5s 样本、AST-base 单 GPU 每 epoch <1–2 min（特征缓存后）；昆虫长音频训练时随机裁 10.24s 窗与推理一致。
- **与其他方法差异**：vs LwF/蒸馏——无需保留旧模型前向，少一半显存/时间；vs LP——多 3–4 pts 预期但慢且有 OOD 风险。
- **最小验证**：留出每类 ≥1 条（3 条类留 1）做 quick check：旧类 acc 不低于 frozen 头基线、新类 acc>80%。
- **成本**：训练 10–30 min + 特征缓存 10–20 min（384kHz 长文件解码是瓶颈）。依赖：transformers/torch/torchaudio（wheel 内，R2 负责核实）。
- **Kaggle kernel 可行性**：可行（单 GPU、离线、时限内绰绰有余）。
- **失败模式**：LR 过大毁 encoder（负例见 §6）；3 条样本类被淹没；昆虫 OOD 扭曲特征。
- **置信度**：高（多来源一致 + 与本地资产完全对齐）。规则风险：无（只用提供的 checkpoint 与数据）。

### MC-R4-02 Frozen encoder 线性探测 / LP-then-FT
- **家族**：linear probing / 两阶段迁移。
- **机制**：encoder 前向一次缓存 920 条 embedding（CLS 或 mean-pool），训 29 类逻辑回归或单层线性头（秒级收敛，可 L2 正则+类均衡）；可选阶段二：以该头初始化后整体低 LR 微调。
- **解决问题**：第一小时内拿到强基线分（预期比 MC-R4-01 低 3–5 pts，比一切传统特征高 30+ pts）；给稀有类（3/7 条）更稳的决策面；防 OOD 特征扭曲。
- **证据**：R4-F003（freeze 90.8–91.8% on ESC-50；<10 clips/类时 freeze 最优）、R4-F005（LP-FT 双优）、R4-F009（本 checkpoint 本来就是 LP 产物，说明出题方自己用此法达到了可用的 16 类模型）。
- **适配**：embedding 缓存后可无限次免费重训头，适合扫 C/temperature/logit-adjust τ；埋点成本≈0。
- **差异**：vs NCM——有判别式边界，处理不均衡更灵活；vs 全微调——不动 encoder。
- **最小验证**：5 min：缓存 embedding→sklearn LogisticRegression→留出集 acc。
- **成本**：前向 920 条 ≈ 数分钟（GPU）；训头 <10s。
- **Kaggle kernel 可行性**：可行。
- **失败模式**：昆虫超声重采样到 16k 后信息损失使 frozen 特征区分度不足（但四昆虫类各 60 条、种间声学差异大，风险中低）。
- **置信度**：高。规则风险：无。

### MC-R4-03 Post-hoc logit adjustment / 先验校准（开关型后处理）
- **机制**：推理 logits − τ·log(训练类先验)，τ∈[0,1] 用留出集或 LB 提交扫 2–3 个值。
- **解决问题**：类不均衡（3–62/类）下，若 metric 是 macro-F1 或测试先验均匀，未校准模型会系统性压制稀有类。
- **证据**：R4-F006（Menon et al.，优于 re-weighting）。
- **适配**：零训练成本、纯后处理、可提交 A/B（τ=0 vs τ=0.5/1.0 各一次提交），完美匹配"提交是测量资源"的策略。
- **最小验证**：本地留出集上看稀有类召回变化；真验证靠 2 次 LB 提交差分。
- **Kaggle kernel 可行性**：可行（几行代码）。
- **失败模式**：若 metric 是 plain accuracy 且测试先验≈训练先验，τ>0 会掉分 → 必须做成开关，默认 τ=0。
- **置信度**：机制高置信；对本题增益不确定（metric 未知，[UNRESOLVED] 依赖 R3/题面）。

### MC-R4-04 多窗 logits 平均（变长推理）+ 权重平均/双模集成
- **机制**：>10.24s 音频切不重叠/半重叠窗，各窗过模型后 logits（或概率）平均；训练侧对长音频随机裁窗保持一致性。集成：最后 N epoch checkpoint 权重平均（SWA 式）+ 可选 LP 头与 FT 模型 logits 平均。
- **解决问题**：40 条 >6s 测试样本（最长 54.3s）无法单窗覆盖；小数据微调方差大，平均降方差。
- **证据**：PSLA model aggregation 实证（R4-F008）；多窗聚合为 PANNs/AST 系推理标准做法 [INFERENCE，无单独定量论文，属社区标准]。
- **适配**：推理总量 363 条 + 40 条多窗 ≈ 数分钟；权重平均零推理开销。
- **最小验证**：对昆虫长训练样本比较单窗 vs 多窗留出 acc。
- **Kaggle kernel 可行性**：可行。
- **失败模式**：对 5s 定长样本做重度 TTA（时间偏移/通道混合）收益≈0 且烧时间（见 §6）。
- **置信度**：多窗高；集成中高。

### MC-R4-05 NCM/原型 + kNN on embeddings（次级兜底/融合成分）
- **机制**：每类 embedding 均值（L2 归一化后余弦）作原型，最近原型分类；或 k=5 kNN。
- **解决问题**：完全无需训练的分类器，3 条样本类也有定义良好的原型；作为 L3 兜底与 sanity check。
- **证据**：CIL 文献中 NME/NCM 是 iCaRL 等方法的核心组件（R4-F002 来源内），frozen 特征上 NCM 与线性头差距通常 1–3 pts [INFERENCE]。
- **最小验证**：与 MC-R4-02 同一批 embedding，1 min 出结果对照。
- **Kaggle kernel 可行性**：可行。
- **失败模式**：类内多模态（昆虫不同录音设备）使单原型不足 → 用 kNN 补。置信度：中高。

### MC-R4-06 传统特征（mel/MFCC 统计）+ RF/GBDT（最终保险）
- **机制**：每条音频 log-mel/MFCC 的 mean/std/percentile + 时长/采样率元特征 → RandomForest 或 LightGBM。
- **解决问题**：AST 全链路失败（wheel 缺包/显存/超时）时的合法提交；预期 35–55%（参照 ESC-50 传统基线 32–44%，R4-F007；本题元特征捷径会再抬高昆虫类）。
- **证据**：R4-F007、R4-F010。
- **最小验证**：CPU 上 <5 min 全流程。
- **Kaggle kernel 可行性**：可行（纯 CPU 亦可）。
- **失败模式**：分数天花板低，只作保险，不投提交预算调优。置信度：高（作为保险的定位）。

## 6. 负面结果、失败条件与冲突证据

1. **专门 CIL 算法在全数据可用时无收益**（对 Charter 关键问题的否定答案）：Masana survey 中一切方法都以 joint 为上界；LwF/蒸馏还需保留旧模型前向（双倍前向成本）。在本题使用 iCaRL/EWC/LwF = 花更多算力逼近一个你已经可以直接训练的目标。[R4-F002]
2. **大学习率全量微调毁掉 AudioSet 特征**：AST 官方 ESC-50 recipe 用 1e-5（比常见 3e-4/5e-5 低 1–1.5 个数量级）；PANNs from-scratch 対 fine-tune 差 11.4 pts 说明特征即分数。用默认 Trainer lr(5e-5)+短 epoch 可能直接低于 frozen 基线。[R4-F003/F004]
3. **硬 class weights 在 3-样本类上的崩坏**：把 Sheep/Keyboard(3 条) 权重放大 ~20× 等价于让 3 条样本主导梯度 → 大类召回崩塌 + 过拟合 3 条样本。Menon et al. 显示 weighting 系方法整体劣于 logit adjustment；后处理式调整可回滚，训练式加权不可回滚。[R4-F006]
4. **本地 CV 不可信**：(a) ESC-50 源数据同一 Freesound 录音切成多个 5s 片段、本题文件名已哈希 → 随机 split 有近重复泄漏，本地分虚高；(b) 系统在音频族有 LB +0.096 / 本地 CV −0.127 的反号实录。→ 不要按本地 CV 排序微小改动，用 LB 提交额度测量（50 次充足）。[TASK_ANALYSIS §6 + 系统背景]
5. **OOD 上激进微调可能反伤**：昆虫超声重采样后与 AudioSet 分布差异大；Kumar et al. 显示 FT 在大偏移下平均 −7% OOD。若首次全微调后昆虫类 LB 表现差于 LP 基线，应切 LP-FT 或降低 encoder LR，而非加 epoch。[R4-F005]
6. **重度 TTA 对定长 5s 样本大概率是噪声**：没有找到环境声分类上时间偏移/通道混合 TTA 稳定加分的定量 primary source（检索后仍缺失 → 按噪声处理）；唯一有实证的"集成"形式是 PSLA 的权重平均/多模平均与长音频多窗聚合。TTA 每一种变体都消耗推理时间与一次 LB 验证，不值得。[R4-F008 + UNRESOLVED]
7. **元数据捷径（时长/采样率→昆虫类）是双刃剑**：训练集内 100% 有效，但测试集有 1 条 8kHz 与 1 条 11025Hz（可能非昆虫）且规则可能在正式集失效；只可作为特征输入模型或与模型预测一致时的 tie-break，不可硬覆盖。[R4-F010]
8. **冲突记录**：starter 占位 header (`id,prediction`) vs 模板 (`path,target`)（R4-F011，本地内部冲突，非网页冲突）；网页与本地资产无发现冲突。

## 7. 未解决问题

- metric（accuracy vs macro-F1）与测试标签先验 → 决定 MC-R4-03 的 τ 默认值；只能由读题面角色/LB 差分解决。
- wheel 环境是否含 sklearn/LightGBM/librosa（影响 MC-R4-05/06 实现选型）→ R2 范围。
- 音频社区 TTA 的正面定量证据缺失是"不存在"还是"没找到"：22 min 时间盒内按后者保守处理，结论（不投预算）不变。
- LP vs FT 在昆虫超声子集上的真实差距：只能靠本题最小实验（MC-R4-02 阶段一 vs MC-R4-01 各一次留出评估）。

## 8. 给题目分析 Agent 的合并建议

1. 把「本题不是 CIL 问题，而是 920 条小数据 29 类联合微调」写为定论（R4-F001/F002），删除对 LwF/蒸馏路线的任何时间分配。
2. 采纳时间线：0–20min 兜底 L0/L2 落袋 → 20–50min MC-R4-02 LP 基线并提交 → 剩余时间 MC-R4-01 联合微调（lr 1e-5 锚点）+ MC-R4-04 多窗推理 → 末段 MC-R4-03 τ 开关用 2 次提交 A/B。
3. 预期分数区间（29 类、数据比 ESC-50 更少更不均衡，仅供校准不作承诺）：传统特征 35–55%、LP 80–90%、联合微调 88–95%。
4. 提交预算策略：微小改动全部用 LB 差分验证，不信任本地 CV 的 ±2pts 内排序（负面结果 §6.4）。

## 9. 来源表

| 标题 | URL | 日期/版本 | 类型 | 支持 Finding |
|---|---|---|---|---|
| Class-incremental learning: survey and performance evaluation (Masana et al.) | https://arxiv.org/abs/2010.15277 （全文 https://ar5iv.labs.arxiv.org/html/2010.15277） | 2020, TPAMI'22 | 综述论文（原文检索到 joint upper bound 表述） | R4-F002 |
| PANNs: Large-Scale Pretrained Audio Neural Networks (Kong et al.) | https://arxiv.org/abs/1912.10211 （全文 https://ar5iv.labs.arxiv.org/html/1912.10211, Table XIII） | 2019, TASLP'20 | 原始论文（freeze vs fine-tune vs scratch 定量） | R4-F003 |
| AST: Audio Spectrogram Transformer (Gong et al.) | https://arxiv.org/abs/2104.01778 | 2021, Interspeech | 原始论文 | R4-F004 |
| AST 官方仓库 README（ESC-50 recipe, lr1e-5, 95.75%） | https://github.com/YuanGongND/ast （raw README 已核对） | 2022 更新 | 官方代码仓库 | R4-F004 |
| Fine-Tuning can Distort Pretrained Features… (Kumar et al.) | https://arxiv.org/abs/2202.10054 | 2022, ICLR | 原始论文 | R4-F005 |
| Long-tail learning via logit adjustment (Menon et al.) | https://arxiv.org/abs/2007.07314 | 2020, ICLR'21 | 原始论文 | R4-F006 |
| ESC-50 官方仓库 README（leaderboard：MFCC+RF 44.3%、human 81.3%、AST 95.7%、BEATs 98.1%） | https://github.com/karolpiczak/ESC-50 （raw README 已核对） | 持续更新 | 官方数据集仓库 | R4-F007 |
| PSLA: Improving Audio Tagging with … Model Aggregation (Gong et al.) | https://arxiv.org/abs/2102.01243 | 2021, TASLP | 原始论文 | R4-F008 |
| 本地资产：archive/*.csv、model/*、ioai-starter.py | 本地路径见 ASSET_MAP.md | 2026-08-04 快照 | 官方题目资产 | R4-F001/F009/F010/F011 |
