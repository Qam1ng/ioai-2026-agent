# RESEARCH_SYNTHESIS.md

## 0. 研究覆盖概览

| 路 | Charter 主轴 | 状态 | 实际覆盖 | 主要受限 |
|---|---|---|---|---|
| R1 | 建模与工程配方（轴 2 + 轴 4 低算力） | 完整交付（216 行） | AST 官方微调 recipe、扩头设计、SpecAugment/mixup、长音频裁剪与聚合、混杂采样率/位深解码、超声失配与补救、linear probe 保底、算力预算 | 通用搜索引擎被拦截 → 改用 primary source 直取；ESC-50 榜单（302 重定向）与 FSC22 全文（403）未取得 |
| R2 | 题面/metric/验证与泄漏（轴 3） | 完整交付（146 行） | 竞赛页可达性、submission 契约冲突、test 组成推断、ESC-50/InsectSet 官方 split 语义、accuracy vs macro-F1 策略差异、logit adjustment、LB 方差与提交预算 | 竞赛页登录墙、Kaggle API 401、搜索引擎零命中 → **metric 仍未知**；FSC22 论文 403 |

两份报告均按 Charter 工作、无越界、无明显截断。阶段 B 对二者的关键本地数字做了独立复核（见 §1 CF-3/CF-4/CF-7）。

**历史教训目录状态**：`context/prior_lessons/` 只含一个 README，内容为"本次运行没有提供历史教训目录" → **本次无 prior_lessons 条目可引用**。唯一可用的一等实测教训来自系统 prompt 注入：(a) 音频任务族上 LB +0.096 的改动在本地 5-fold CV 上显示为 −0.127（CV/LB 符号反转）；(b) 无条件注入外部知识实测为负收益（R&D-Agent 35.1%→32.0%）。二者均已并入下文结论。

---

## 1. Canonical Findings（去重后）

**CF-1 [UNRESOLVED，一等重要] 本题 metric/rules 在公网不可获得。**
竞赛页 `kaggle.com/competitions/ioai-2026-ai-models-track-practice-task-1` 返回登录页（对照 `ioai-2026-practice-task-1` → 404，说明 slug 真实存在）；Kaggle 公共 API 401；Google/Bing/DDG 对该 slug 零命中；IOAI 官网 `/ai-models-track/` 404。
推论：**"同名题公开 solution"风险为零，因为不存在**；同时任何外部材料声称的 metric 都不可信。若操作员粘贴的 starter prompt 含 Evaluation 文本，该文本优先。
来源：R2-F001。

**CF-2 [LOCAL_FACT] submission 列名存在文档内冲突。** `ioai-starter.py` 示例写 `id,prediction`；本题模板 `archive/submission.csv` 表头是 `path,target`。以模板为准，kernel 内再确认；这是"完美训练拿 0 分"的失败模式。来源：R2-F011。

**CF-3 [LOCAL_FACT + INFERENCE] test 组成（阶段 B 用 soundfile 独立复核，与两报告一致）。**
363 文件：323 个 =5.0s、324 个 <5.5s、9 个 5.5–11s、30 个 ≥11s、max 54.27s；采样率 44100×242 / 48000×80 / 24000×26 / 96000×11 / 11025 / 8000 / 312500 / 192000 各 1（**无 384k**）。
→ 只有 ~39–40/363（≈11%）需要多窗处理；昆虫类约 40 个（≈11%）；旧 16 类估 ~45%，FSC22 系 ~各 18 个。
各 (sr,ch) 签名的 test/train 比：阶段 B 精确复算为 0.28–0.45（R2 报告写 0.31–0.40，略窄），整体 363/920=0.395 → 支持"同池按类分层随机抽样"。补充观察：`(312500,1)` 与 `(11025,1)` **只出现在 test**，`(256000,1)×9`、`(384000,1)×3` 等只出现在 train → 与"从更大源池随机切分"一致，而非严格同签名配对。
来源：R1 §2、R2-F002、R2-F003、R2-F013 + 阶段 B 复核。

**CF-4 [LOCAL_FACT，阶段 B 在本地 transformers 源码直接验证] ASTFeatureExtractor 的三条硬约束。**
① 静默截断：`difference<0 → fbank = fbank[0:max_length, :]`，>1024 帧（10.24s）的内容被无声丢弃、不报错；
② mel 频带 `min_frequency=20, max_frequency=sampling_rate//2 = 8000`（kaldi 尺度、triangularize_in_mel_space、norm=None、frame 400/hop 160/fft 512/preemphasis 0.97）→ 8kHz 以上信息在进入模型前即消失，与用哪个库重采样无关；
③ 不足则 ZeroPad2d 补 0，**之后**才整体 `normalize=(x−mean)/(2·std)` → padding 区为常量 **+0.467**（R1 报告写 −0.467，符号有误，机制不变）。checkpoint 是在 5s 文件（≈500 有效帧 + 524 常量帧）上训出的 → 短文件保持同样 zero-pad 模式与 checkpoint 分布一致。
来源：R1-F003、R1-F004 + 阶段 B 本机复核（transformers 5.14.1 源码；kernel 内 config 声明 5.10.2，行为假定相同 → 建议 kernel 内 assert 一次帧数）。

**CF-5 [EXTERNAL_EVIDENCE] 官方 AST 微调超参先验。**
AST 官方 ESC-50 recipe（AudioSet 预训练起点）：`lr=1e-5`（无预训练则 1e-4）、epoch 25、batch 48、`freqm=24`、`timem=96`、`mixup=0`、`bal=none`、CE loss、第 5 epoch 起每 epoch ×0.85、`audio_length=512`、fstride=tstride=10。SpeechCommands（3.5 万条）才用 mixup=0.6。HF audio-classification 文档：lr 3e-5、bs 32、grad_accum 4、epoch 10、warmup_ratio 0.1、按 accuracy 选模（但其示例是 wav2vec2 + 全新随机头，与"保旧行"情形不同）。
→ 三个独立来源把 lr 收敛到 **{1e-5, 3e-5}**、epoch 到 8–20。边界：`audio_length=512` **不可搬**（本 checkpoint 位置编码 1214 对应 1024 帧）；batch 48 在 1024 帧下大概率 OOM。
来源：R1-F001、R1-F002。

**CF-6 [INFERENCE] provided checkpoint 是"冻结 encoder 训头"的产物。**
`training_args.bin` 含 `./ast-freeze-enc` → encoder 仍非常接近 AudioSet 预训练权重，16 类头与该 encoder 匹配。推论：(a) 扩头后保留旧 16 行与该 encoder 自洽，未训也能给旧类合理 logits（可做数值等价性 sanity check）；(b) encoder 从未被本题数据动过 → **全量微调仍有 headroom，不必因原作者冻结而跟着冻结**；该字符串**不是**规则约束。来源：R1-F009。

**CF-7 [LOCAL_FACT] 超声昆虫文件在 16kHz 前端下信息近乎归零，但影响面极小。**
阶段 B 亲自复核：test 中 2 个高采样率文件 >8kHz 能量占比 **0.99938（sr=312500, 13.1s）与 0.99789（sr=192000, 25.5s）**；11 个 96kHz test 文件全部 ≥92% 能量在 <8kHz（正常声音）。train 侧 Tettigonia 抽查：96kHz 文件 hi8k=0.998，但 384k/256k 文件跨度大（0.005–0.219）→ **超声并非该类的普遍特征，而是部分录音的特征**（这比 R1 的表述更谨慎）。
→ 直接分数上限约 **+0.55%（2/363）**；阈值 0.5 有较大安全边界（96kHz 正常文件 r<0.08）。
来源：R1-F005、R1-F006 + 阶段 B 复核。

**CF-8 [LOCAL_FACT + EXTERNAL_EVIDENCE] 变长音频只出现在昆虫类，且源数据集有成熟分段惯例。**
本地：所有 0–24 类文件恰好 5.0s → "时长 >6s ⇒ 属 25–28 类"在 train 上 100% 成立（可作诊断规则：若模型在长 test 文件上预测非昆虫类，说明分块/超声处理出问题）。
外部（InsectSet, PLOS CB 2023）：5s 段、3.75s overlap（75%）、短文件循环填充到 5s、切到尾部环绕补足（剩余 ≥1.25s 才保留）、**按文件而非按段划分数据**。
来源：R1 §2、R1-F008。

**CF-9 [LOCAL_FACT] 解码工程。** 训练+test 采样率集合 {8k,11.025k,16k,22.05k,24k,25.6k,32k,44k,44.1k,48k,96k,192k,200k,250k,256k,312.5k,384k}，含 float32(fmt=3) 与 EXTENSIBLE(24bit)。`soundfile.read(..., always_2d=True).mean(1)` + `librosa/torchaudio.functional.resample` 可读全部 1283 个；标准库 `wave` 在 fmt=3 抛错。建议逐文件 try/except，失败回退 zeros 并打印计数。来源：R1-F010 + 阶段 A/B 复核。

**CF-10 [INFERENCE] 算力预算充裕。** 1024×128 → 1214 token（与 position_embeddings 一致），约 ViT-B/16@224 的 6 倍序列长度；amp 下估 8–15 样本/s：920×12 epoch ≈ 15–25 min；363 文件推理（含分块约 450 窗）<2 min；1283 文件特征提取 <3 min。→ 单模型训练+推理稳进 kernel，甚至可做 2–3 组配置或 ≤3 fold。边界：不开 amp/bf16 吞吐腰斩；batch 用 8–16 + grad_accum。来源：R1-F011。

**CF-11 [EXTERNAL_EVIDENCE + INFERENCE] 验证协议：源数据集官方是"按源录音分组"，但本题 test 更像分层随机。**
ESC-50 官方：5 folds "making sure that fragments from the same original source file are contained in a single fold"，metadata 有 `src_file`/`take`；InsectSet：按物种切 62.7/15.2/22.1 并把同录音 snippet 视为一个样本。→ 同源近重复在源池中是被官方承认的真实现象。
但本题文件名是 hex 哈希，`src_file`/`take` 被抹除，**无法直接复现分组**；且 CF-3 的比例证据支持 test 是同池分层随机抽样 → train/fine_tune 与 test 之间同样存在同源近重复（对我们**有利且合法**），grouped CV 会系统性偏悲观。
来源：R2-F005、R2-F006、R2-F010、R2-F012。

**CF-12 [INFERENCE] LB 噪声与提交预算。** accuracy 1σ=√(p(1−p)/n)：p≈0.85, n=363 → 1.9%；public 30%（n≈109）→ 3.4%。macro-F1 含 ~1 样本类，单样本翻转 ±3.4%/类。→ **LB 差异 <3–4% 无统计意义**。建议预算（2h 窗口现实可用 5–10 次）：1 次 metric 探针（兼管道/列名验证）→ 1 次 baseline(τ=0) → 1 次 τ=1或0.5 → 1 次最佳+分块聚合开关 → 其余留给失败恢复与明显更强的变体；**每次提交只改一个可解释因子**。来源：R2-F008、R2-F009。

**CF-13 [EXTERNAL_EVIDENCE] 长尾后处理有 primary 依据。** post-hoc logit adjustment（logits 减 τ·log 先验）等价于优化 balanced error（Menon et al., ICLR 2021）；F1 阈值理论：对校准概率最优阈值 = 最优 F1 的一半，无信息分类器在 F1 下最优行为是"全判正"（Lipton et al. 2014）。→ **macro-F1/balanced 类 metric 奖励对稀有类的激进预测，accuracy 惩罚它** → τ 必须可切换。来源：R2-F007。

**CF-14 [INFERENCE] metric 判别探针的期望值。** 全预测最大类（Thunderstorm，估 p≈0.066）：accuracy≈0.066 / macro-F1≈0.0043 / balanced-acc≈0.034，量级差 8–15× → 一次提交可判别 metric 家族。若落在中间值 → 记为"非标准 metric"，再用两次探针（全预测某旧类 vs 某新类）比较。来源：R2-F004。

**CF-15 [LOCAL_FACT，泛化不确定] 数据集拼接痕迹（元数据↔类相关）。** 阶段 B 逐文件证实：Hand Saw 45/45=24kHz、Chainsaw 41/45=24kHz（24kHz 仅出现在这两类，共 83 训练文件；test 有 26 个 24kHz）、Wolf Howl 23/23=48kHz、Generator 45/45=44.1kHz、Tettigonia 独占 256k/384k、所有非昆虫类恰好 5.0s。属合法特征工程（全部来自题目自带数据），但是典型伪相关 → 仅宜作 tie-break/审计信号。来源：R1 §2、MC-6。

**CF-16 [INFERENCE] 主要不均衡问题不是"类不均衡"本身，而是"旧 16 类只有 296 条且极不均衡（62 vs 3），却占 test ~45%"。** 训练集中新类占 68%（624/920），模型天然偏新类；保留旧头行权重是最便宜的旧类知识保护，旧类过采样比调 loss 权重更好控。来源：R1 top-5 #5、R2-F013。

---

## 2. Canonical Method Cards（全部通过 kernel 可行性门）

> 以下是**研究建议的优先级**，不是本地实验验证过的模型排名。任何一条都未在本题上测过分数。

### MC-A 扩展分类头（保留旧 16 行）+ 全量联合微调 ★ 主线
- 家族：迁移学习 / 类增量（带完整 replay）。
- 机制：新建 `nn.Linear(768,29)`，`W[:16]=old_W, b[:16]=old_b`，`W[16:]~N(0,0.02)` 或 0、`b[16:]=0`；同步更新 config 的 `num_labels`/id2label/label2id 到 29；在 `train.csv ∪ fine_tune.csv` = 920 条上 CE 微调整个模型。
- 起点超参（CF-5 + CF-10）：lr 1e-5、batch 8–16 + grad_accum、epoch 10–15、freqm 24 / timem 96、mixup 0、第 5 epoch 起 ×0.85、amp、保持 **1024 帧**输入。
- 证据：CF-5（官方 recipe）、CF-6（checkpoint 自洽性）、CF-16（旧类先验价值）。
- 最小验证：先做 **0-epoch 数值等价性检查**（扩头后旧 16 类 logits 必须与原模型完全相同）；再 3 epoch 快跑，看 old-16 块 recall 是否保持、new-13 是否 >0.8。
- 成本：15–25 min GPU。依赖：torch + transformers（pinned wheel 内含）。
- 失败模式：lr ≥1e-4 → 旧类灾难性遗忘；忘记同步 num_labels → shape 错误；62 条的 Thunderstorm 吸走 3 条的稀有类。
- 置信度：高（结构改动零风险且可数值验证；超参有 primary source）。规则风险：无。
- 溯源：R1-MC-1。

### MC-B 长音频窗口化 + logits 聚合 ★ 必做（防 silent bug）
- 机制：训练时 >10.24s 的文件每 epoch 随机 crop 10.24s（等价时间域增强），或切成 K 段作为 K 个样本并按 1/K 加权；推理用 10.24s 窗 / 5.12s hop（50% overlap），收集各窗 logits → **mean logits**（softmax 前）→ argmax。短文件走原生 zero-pad 路径。
- 解决：CF-4① 静默截断；影响 ~39/363 test 文件。
- 证据：CF-4（源码）、CF-8（InsectSet 75% overlap 惯例）。
- 最小验证：对 39 个 ≥5.5s 的 test 文件比较"首窗 argmax"与"mean-logits argmax"的不一致数（>5 个即说明聚合有真实影响）；再用 train 侧 120s 昆虫文件做 held-out 对比。
- 成本：可忽略（<1 min）。
- 失败模式：切段训练使 4 个昆虫类严重过采样（需按文件加权）；长文件被静音段主导。
- 置信度：中高（截断事实确定；mean vs max vs 投票的相对优劣本题未实测）。
- 溯源：R1-MC-3。

### MC-C Post-hoc logit adjustment（τ 可切换）★ 近零成本、覆盖 metric 不确定性
- 机制：推理时 `logit_c ← logit_c − τ·log π_c`（π 为训练类频率或估计的 test 先验），τ∈{0,0.5,1}；一次推理同时导出三份 submission，按 LB 选。
- 解决：CF-1（metric 未知）+ CF-16（训练分布偏新类）。
- 证据：CF-13（ICLR 2021 + F1 阈值理论）。
- 最小验证：本地分层 CV 上画 accuracy(τ) 与 macro-F1(τ) 曲线（应交叉：accuracy 在 τ≈0 最好，macro-F1 在 τ≈0.5–1 最好）；再用 1 次 LB 提交确认方向。
- 成本：<1s，仅 numpy。失败模式：metric 若为 plain accuracy 且 τ 偏大，稀有类误报吃掉 1–3% 大类正确率。
- 置信度：高。溯源：R2-MC-1。

### MC-D 单类常数提交作为 metric 判别探针 ★ 第一次提交
- 机制：全部预测同一类（建议最大类）→ 由 LB 数值量级反推 metric 家族（CF-14）。**与"端到端管道 + 列名契约（CF-2）验证"合并为同一次提交**，不浪费额度。
- 前提：若竞赛页/starter prompt 已写明 Evaluation，**不要花这次提交**。
- 成本：一次 kernel 运行（无需 GPU/训练）。置信度：高。规则风险：无（只用自己提交的公开反馈）。
- 溯源：R2-MC-2。

### MC-E 双协议验证（分层随机为主 + 近重复分组为诊断）★ 决策基础设施
- 机制：(a) 主协议：按 target 分层的 5-fold 或 20% hold-out（与 test 的 ~28–40% 抽样率同量级）；(b) 诊断协议：用 provided AST encoder 提 embedding → 余弦阈值聚类近重复 → GroupKFold。两者差值 = 泄漏带来的乐观量；差值大则优先信 LB。
- 解决：CF-11、CF-12、无官方 val。
- 报告要求：同时给 accuracy 与 macro-F1；对 3 样本类（Sheep/Keyboard Typing）的 per-class 结果标注"不可信"；`StratifiedKFold` 在 n<k 的类上需回退（稀有类固定放训练集）。
- 判别信号：首次真实提交若 LB 显著低于本地分层 CV（>8%），切换到 grouped 协议选型。
- 最终计分 kernel 只跑 1 次全量训练+推理（或 ≤3 fold 集成），不跑完整 CV。
- 置信度：中高。溯源：R2-MC-3。

### MC-F 新旧类分块诊断视图（不是训练方法，是红线指标）
- 机制：把验证分数拆成 old-16 / new-13 分别报告（并按估计 test 占比 ~45%/~55% 加权），加上跨块混淆率。
- 红线：**若 old-16 块的验证分数明显低于"直接用原始 16 类 checkpoint 在 train.csv 留出集"的分数，说明扩展微调破坏了旧类，必须回退**（冻结更多层 / 降 lr / 旧类过采样）。
- 成本：0 额外。置信度：高。溯源：R2-MC-4。

### MC-G 不均衡处理：旧类过采样（+ 可选 √倒数权重）
- 机制：WeightedRandomSampler 权重 ∝ 1/√n_c，把 296 条旧类上采到与新类可比；稀有类配合 SpecAugment 防死记；**不建议一开始就用 ∝1/n 的权重**（会放大 3 样本类的噪声）。
- 证据：CF-5（官方 AudioSet recipe 才开平衡采样，ESC-50 用 bal=none）、CF-16。
- 最小验证：本地按 fold 看 macro 与 micro 双指标；必要时 1 次 LB A/B。
- 失败模式：3 样本类被过采样 20 倍 → 过拟合到那 3 条录音的背景噪声。置信度：中。溯源：R1-MC-5。

### MC-H 冻结 encoder 特征 + 线性探针 / kNN（保底 & 秒级试验台）
- 机制：用 provided AST 取 pooled/mean-token 768 维特征（1283 文件前向一次，fp16、batch 16，1–3 min，缓存到 `/kaggle/working`），拟合 LogisticRegression(class_weight='balanced') 或 kNN；长文件按窗取特征后 mean-pool。
- 作用：**20 分钟内可提交、几乎不会崩的 baseline 地板**；并作为预处理变体（频移、聚合方式、循环填充）的秒级 A/B 台，不必重训。
- 失败模式：pooled 特征对昆虫细类分辨不足；**pinned 环境可能无 sklearn** → 需 torch 手写 softmax 回归回退。
- 置信度：中高。溯源：R1-MC-2。

### MC-I 超声文件的时间扩展 / 频率下移预处理（低优先，收益上限 ~0.55%）
- 机制：在原生 sr 下算 >8kHz 能量占比 r 与谱质心 c；若 `native_sr>32000 and r>0.5`，取 k=2^ceil(log2(c/2000))（k≤8），把波形**当作 sr/k 采样**（不重采样，只改声明采样率）再重采样到 16kHz → 所有频率 ÷k、时长 ×k，超声进入 20–8000Hz；train/test 用同一函数。必须配合 MC-B（k=4 会把 13s 变 52s）。
- 解决：CF-7 + CF-4②。文献支持"频率信息要保住"（InsectSet 明确指出 mel 前端对超声昆虫不利），但其解法（LEAF 自适应前端、44.1kHz 域）本题不可用。
- 最小验证：**先只在 train 侧超声文件 + MC-H 特征上验证**（下移后最近邻是否变成同类昆虫）；若否直接放弃。必须打印被触发的文件数并人工核对。
- 成本：秒级 CPU（numpy/soundfile）。置信度：中（机制确定，收益上限小且未实测）。溯源：R1-MC-4。

### MC-J 元数据先验（采样率/时长）作为 tie-break —— 谨慎，非主推荐
- 机制：由 train 统计的 `sr→类`、`duration>6s→25..28` 先验，仅在 top-1 与 top-2 概率差 <τ 时打破平局。
- 证据：CF-15（全部来自题目自带数据，合法）。
- 最小验证：本地 CV 上统计"按 sr 先验强制覆盖"会改动多少样本、改对率；<60% 则弃用。
- 失败模式：若组织者对 test 统一转码或刻意打散录音条件，先验失效并伤成绩（过拟合伪相关的典型）。置信度：低。应在 solution 报告中披露。溯源：R1-MC-6。

**建议落地顺序（研究建议，非实测排名）**：MC-D/CF-2 管道与 metric 探针 → MC-H 保底提交 → MC-A（含 MC-B 与 MC-E/MC-F 度量）→ MC-C 的 τ 开关 → MC-G → MC-I → MC-J。

---

## 3. 历史教训 × 外部证据 对照结论

| 教训（一等实测） | 外部证据 | 合并结论 |
|---|---|---|
| 音频任务族：LB +0.096 的改动在本地 5-fold CV 上是 −0.127（符号反转） | ESC-50/InsectSet 官方按源录音分组（CF-11）；本题 test 更像同池分层随机（CF-3） | **机制被解释了**：若 CV 协议比 test 更严格（grouped / 去近重复），CV 会与 LB 反号。→ 主协议用分层随机 CV，grouped 仅作乐观度诊断；LB 与 CV 冲突时以 LB 为准（且 LB 差 <3–4% 视为噪声，CF-12） |
| 无条件注入外部知识为负收益（35.1%→32.0%，仅难题为正） | 本轮外部证据集中在"官方源码硬事实 + 官方 recipe 超参"，而非新颖 SOTA 方法 | 采纳的外部内容限于**可验证的硬约束（CF-4）与官方超参先验（CF-5）**；新颖方法（LwF/蒸馏、LEAF 前端）均被判为低价值或不可用，不进主线 |
| 提交额度是一等测量资源，应刻意花 | CF-12/CF-14 给出定量预算与探针设计 | 第一次提交即买信息（metric 探针 + 管道验证），后续每次只改一个正交因子 |
| （prior_lessons 目录本次为空） | — | 已在 §0 显式声明，无可引用条目；不得伪造 |

---

## 4. Metric / Validation / Leakage / Post-processing 结论

- **Metric**：未知（CF-1）。默认按 accuracy 行事（τ=0），同时把 τ 做成开关（MC-C），并用 1 次常数提交判别（MC-D/CF-14）。若确认为 macro-F1：每个类固定占 1/29≈3.4%，猜对一个 Sheep 的收益 > 把 Thunderstorm 提升 5%；但 Sheep/Keyboard test 期望各 ~1 个，接近抛硬币 → 用 MC-C 低成本拿走期望收益，**不要把整套方案押在稀有类上**。
- **Validation**：主 = 分层随机（CF-11、MC-E）；辅 = 近重复分组诊断；old/new 分块红线（MC-F）；稀有类 per-class 结果标注不可信。
- **Leakage**：train↔test 的同源近重复很可能存在且**合法有利**；禁止的是用源数据集公开标签匹配 test 文件（两份报告均未做，也未提供任何逐文件线索）。
- **Post-processing**：logit adjustment（MC-C）> 元数据 tie-break（MC-J，低置信）；分块聚合用 mean-logits（MC-B）。

## 5. 经典 / 非主流备选

- 冻结 encoder + linear probe / kNN（MC-H）：经典低算力路线，作为地板与试验台。
- 时间扩展 / heterodyne 频移（MC-I）：蝙蝠与昆虫生物声学的经典信号域手段，用来绕过固定 16kHz 前端。
- 元数据指纹 tie-break（MC-J）：非主流、伪相关风险高。
- 两阶段训练（先只训新头再全量解冻）：结构上介于 MC-A 与 MC-H 之间，比 MC-A 多一倍时间，未见 primary 证据支持其更优 → 仅作备选。

## 6. 负面结果与失败条件（去重）

1. **不要盲抄 `audio_length=512`**：官方 ESC-50 用 512 帧，但本 checkpoint 位置编码是 1214（=1024 帧）；改短需插值/裁剪位置编码且偏离 checkpoint 输入分布 → **保持 1024 帧**（本地资产优先于官方 recipe）。
2. **mixup 在本数据规模无官方支持**：ESC-50 recipe mixup=0，只有 3.5 万条的 SpeechCommands 才用 0.6 → 默认不用，最多作最后一次 A/B。
3. **蒸馏 / LwF 大概率浪费**：旧类 296 条**在手**，联合训练已是完整 replay；文献中的蒸馏是为"旧数据不可用"设计的 → 不投时间。
4. **换前端 / LEAF / 更高采样率 / 更强外部模型不可用**：外部模型默认禁止，AST 固定 16kHz/128mel → InsectSet 的最优解法 [INFEASIBLE_FOR_KERNEL / 规则不允许]，仅作思路来源。
5. **不要用 `wave` 模块**（fmt=3 抛错）。
6. **不要以为把长波形交给 feature extractor 就处理完了**（静默截断，CF-4①）。
7. **不要照抄 ESC-50 grouped 5-fold 作为选型协议**（比 test 更严格，会误杀对 LB 有效的改动，CF-11）——这与"照搬源数据集官方协议"的直觉显式冲突。
8. **收益上限被数据量钉死**：分块只影响 39/363，超声只影响 2/363 → 不要把提交额度花在这些小改动的反复 A/B 上，应优先验证影响全部 363 个样本的因素（lr / epoch / 扩头 / τ）。
9. **列名契约**（CF-2）可以让一次完美训练拿 0 分。
10. 未取得 primary 证据的说法：ESC-50 榜单具体数字（paperswithcode 302）、FSC22 论文正文（MDPI 403）→ "AST 在 ESC-50 达 95.6%"一类说法**不作依据**。

## 7. 冲突证据

| # | 冲突 | 各方证据质量 | 处理 |
|---|---|---|---|
| X1 | 官方 recipe `audio_length=512` vs 本地 checkpoint 1024 帧 | 官方代码（强）vs 本地 safetensors 位置编码 1214（最高） | **以 [LOCAL_FACT] 为准，用 1024 帧**（§6.1） |
| X2 | 验证协议：ESC-50/InsectSet 官方"按源录音分组"（R2-F005/F006）vs R2 推荐"分层随机"（R2-F003/F010） | 前者是官方数据集事实但适用于源数据集；后者是本题 test 生成机制的比例推断（强但未逐文件证实） | **双协议并存**（MC-E）：分层随机选型 + grouped 诊断；最小判别实验 = 首次提交后比较 LB 与两套 CV 的差值（LB 低于分层 CV >8% → 切 grouped） |
| X3 | test 长文件计数：R1 "324 个 <5.5s" vs R2 "323 个 ≈5.0s" | 两者都是本地统计 | **非真冲突**：阶段 B 复核为 323 个 =5.0s + 1 个 5.3s = 324 个 <5.5s，两报告一致 |
| X4 | padding 常量符号：R1 写 −0.467 | 阶段 B 本地源码 `(x−mean)/(2·std)` 计算得 **+0.467** | 采用 +0.467；机制（padding 变常量、与 checkpoint 分布相关）不变 |
| X5 | (sr,ch) 签名比例区间：R2 "0.31–0.40" vs 阶段 B 复算 "0.28–0.45"，且 `312500`/`11025` 只在 test、`256000`/`384000` 只在 train | 均为本地统计，阶段 B 更完整 | 采用 0.28–0.45；"同池随机切分"结论不变但**置信度略降**（不是严格同签名配对） |
| X6 | Tettigonia 超声普遍性：R1 表述像该类普遍超声 | 阶段 B 抽查显示 hi8k 从 0.005 到 0.998 跨度极大 | 修正为"部分录音超声"，进一步压低 MC-I 的期望收益 |

## 8. 未解决问题（合并去重）

1. **metric 公式与方向**（CF-1）——只能在竞赛环境内读取或用 MC-D 探针判别。**最高优先。**
2. public/private LB 比例与是否全量计分 → 影响噪声估计（现给 n=363 与 n≈109 两个界）。
3. 是否强制使用提供 checkpoint、是否允许外部数据/权重 → 公网无信息，按默认禁止处理。
4. 扩头保旧行 vs 全新头的定量差异（无同设定 primary 证据）；全微调 vs 冻结前 N 层的最优分界（920 样本）。→ 建议各用 1 次本地 3-fold 或 1 次 LB A/B。
5. mean-logits vs max-prob vs 多窗投票的相对优劣（本题未实测）。
6. 频率下移的最佳因子规则与它是否真能被 AST 正确识别（test 仅 2 个文件，几乎无法统计验证）。
7. pinned wheel 内是否含 sklearn / librosa / torchaudio / soundfile → kernel 内先 import 探测并准备 numpy-only 回退。
8. test 中是否真有 Sheep / Keyboard Typing 样本（期望各 ~1，可能为 0）。
9. FSC22 类（16–24）的同源片段比例（论文 403）→ 近重复泄漏绝对量未知。
10. `training_args.bin` 具体超参数值（未安全解出）。
11. kernel 内 transformers 版本（config 写 5.10.2）与本地验证版本（5.14.1）在 feature extractor 行为上的一致性 → 建议 kernel 内 assert 输出 shape 为 (1024,128)。

## 9. 来源表（合并去重）

| 标题 | URL | 版本/日期 | 类型 | 支持 |
|---|---|---|---|---|
| AST 官方 ESC-50 微调脚本 `run_esc.sh` | https://raw.githubusercontent.com/YuanGongND/ast/master/egs/esc50/run_esc.sh | master, 2026-08-04 取 | 官方代码（primary） | CF-5, MC-A, MC-G, §6.1–6.2 |
| AST 官方 SpeechCommands 脚本 `run_sc.sh` | https://raw.githubusercontent.com/YuanGongND/ast/master/egs/speechcommands/run_sc.sh | master, 2026-08-04 | 官方代码 | CF-5（mixup 对比） |
| AST 官方 ESC-50 数据准备脚本 | https://raw.githubusercontent.com/YuanGongND/ast/master/egs/esc50/prep_esc50.py | master, 2026-08-04 | 官方代码 | CF-5 背景（fold 惯例） |
| transformers `ASTFeatureExtractor` 源码 | https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/audio_spectrogram_transformer/feature_extraction_audio_spectrogram_transformer.py | main, 2026-08-04；阶段 B 另在本机 transformers 5.14.1 复核 | 官方源码（primary） | CF-4, MC-B, MC-I |
| HF 官方任务文档 Audio classification | https://huggingface.co/docs/transformers/tasks/audio_classification | 2026-08-04 | 官方文档 | CF-5 |
| Faiß & Stowell, Adaptive representations of sound for automatic insect recognition | https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011541 | PLOS Comput Biol, 2023-10-04 | 同行评议论文 | CF-7, CF-8, CF-11, MC-I |
| ESC-50 官方数据集 README | https://github.com/karolpiczak/ESC-50 | master, 2026-08-04 | 官方数据集仓库 | CF-11, MC-E |
| Menon et al., Long-tail learning via logit adjustment | https://arxiv.org/abs/2007.07314 | ICLR 2021 | 论文 | CF-13, MC-C |
| Lipton et al., Thresholding Classifiers to Maximize F1 Score | https://arxiv.org/abs/1402.1892 | 2014 | 论文 | CF-13, MC-C |
| Kaggle 竞赛页（本题 slug，返回登录页） | https://www.kaggle.com/competitions/ioai-2026-ai-models-track-practice-task-1 | 2026-08-04 | 官方页（登录墙） | CF-1 |
| Kaggle 公共 API competitions/list（401） | https://www.kaggle.com/api/v1/competitions/list?search=ioai | 2026-08-04 | 官方 API | CF-1 |
| IOAI 官方站点（首页可达，`/ai-models-track/` 404） | https://ioai-official.org/ | 2026-08-04 | 官方公告站 | CF-1 |
| FSC22: Forest Sound Classification Dataset | https://doi.org/10.3390/s23042032 | Sensors 2023-02-10（正文 403 未取得） | 论文（未取得） | §6.10, 未解决 #9 |
| 本地资产（最高优先）：`archive/{train,fine_tune,submission}.csv`、1283 wav 头 + 抽样 FFT、`archive/model/*`、`ioai-starter.py` | `SEARCH_BUNDLE/ORIGINAL_ASSETS/…`（见 ASSET_MAP） | 本次运行 | 官方题目资产 + 本机实测 | CF-2, CF-3, CF-6, CF-7, CF-9, CF-15, CF-16 |

---

## 10. Merge Ledger（每个 Finding ID 唯一去向）

### R1（12 findings + 6 method cards）
| ID | 去向 |
|---|---|
| R1-F001 | → CF-5（canonical finding，官方 recipe）；同时支撑 MC-A/MC-G；`audio_length=512` 部分进 §7 X1 冲突 |
| R1-F002 | → CF-5（与 R1-F001 合并为同一 canonical，保留"wav2vec2 + 随机头"边界差异） |
| R1-F003 | → CF-4②（8kHz mel 上限） |
| R1-F004 | → CF-4①③（静默截断 + pad-then-normalize）；符号问题进 §7 X4 |
| R1-F005 | → CF-7（超声能量），普遍性表述被修正（§7 X6） |
| R1-F006 | → CF-7（test 侧影响面定价 2/363） |
| R1-F007 | → CF-7 + CF-8（文献级 mel 失配警告）；其解法进 §6.4 [INFEASIBLE/规则不允许] |
| R1-F008 | → CF-8（InsectSet 分段与按文件划分惯例）；分段参数进 MC-B |
| R1-F009 | → CF-6（checkpoint 为 freeze-enc 产物，全微调有 headroom） |
| R1-F010 | → CF-9（解码工程） |
| R1-F011 | → CF-10（算力预算） |
| R1-F012 | → §8 未解决 #4（扩头保旧行 vs 全新头无 primary 证据；蒸馏低价值结论进 §6.3） |
| R1-MC-1 | → MC-A |
| R1-MC-2 | → MC-H |
| R1-MC-3 | → MC-B |
| R1-MC-4 | → MC-I |
| R1-MC-5 | → MC-G |
| R1-MC-6 | → MC-J（低置信保留）；其统计事实 → CF-15 |
| R1 §2 test 分布统计 | → CF-3（与 R2-F002/F003 合并，阶段 B 复核） |
| R1 §6 负面结果 | → §6.1/6.2/6.3/6.4/6.5/6.6/6.8/6.10 |
| R1 §7 未解决 1–6 | → §8 #4/#4/#5/#6/#7/#1（逐条并入，无丢弃） |

### R2（13 findings + 4 method cards）
| ID | 去向 |
|---|---|
| R2-F001 | → CF-1（metric/rules 公网不可得） |
| R2-F002 | → CF-3（test 时长/采样率，阶段 B 复核） |
| R2-F003 | → CF-3（分层抽样推断 + 每类期望数）；区间被阶段 B 修正 → §7 X5 |
| R2-F004 | → CF-14（探针期望值）；驱动 MC-D |
| R2-F005 | → CF-11（ESC-50 官方按 src_file 分组）；与 R2-F003 的张力 → §7 X2 |
| R2-F006 | → CF-11（InsectSet split 语义） |
| R2-F007 | → CF-13（logit adjustment + F1 阈值理论） |
| R2-F008 | → CF-12（LB 噪声定量） |
| R2-F009 | → CF-12（提交预算表） |
| R2-F010 | → CF-11（train↔test 同源近重复；CV/LB 反号机制）；进 §3 教训对照表 |
| R2-F011 | → CF-2（列名冲突）；进 §6.9 与 MC-D |
| R2-F012 | → CF-11 / TASK_ANALYSIS §6（无官方 val） |
| R2-F013 | → CF-16 + CF-3（旧类占 test ~45%）；驱动 MC-F |
| R2-MC-1 | → MC-C |
| R2-MC-2 | → MC-D |
| R2-MC-3 | → MC-E |
| R2-MC-4 | → MC-F |
| R2 §6 负面结果 | → §6.7（不照抄 grouped）、§6.9（列名）、§4（macro-F1 稀有类方差）、§7 X2（过度自信风险与判别信号）、§6.10（FSC22 未获证据） |
| R2 §7 未解决 1–5 | → §8 #1/#2/#3/#9/#8 |

### 被排除项（未进主总结）
- R1 关于 "AST 在 ESC-50 达 95.6%" 的常见说法：**排除**——R1 自己声明未取得 primary 证据（paperswithcode 302），不作依据（在 §6.10 留痕）。
- R2 提出的其他 LB 探针（探测类先验、探测 public 比例）：**排除进主线**——R2 自评"价值远低且花费更多提交"，在 MC-D 内留痕。
- 两报告中对 IOAI 通用赛制的复述（提交上限、kernel 离线等）：**排除**——与系统 prompt 重复，无新增信息。
