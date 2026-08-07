# R1 Research Report

（研究窗口 ~25min，网络可用但通用搜索引擎被拦截；改用 primary source 直取：官方 AST 仓库 raw 文件、HF transformers 源码/文档、PLOS/Crossref。所有本地数字均为我本人在 ORIGINAL_ASSETS 上现算，可复现。）

## 0. Top-5 可行动方向

1. **扩头保旧行 + 全量联合微调（MC-1）**：把 `classifier.dense` 从 [16,768] 扩到 [29,768]，前 16 行直接拷贝旧权重、新 13 行小初始化（std≈0.02 或 0）+ bias 置 0；用 `train.csv`(296) ∪ `fine_tune.csv`(624) = 920 条一起训。官方 AST ESC-50 recipe 的 lr=1e-5 / epoch=25 / freqm=24 / timem=96 / mixup=0 / CE / 从第 5 epoch 起每 epoch ×0.85（primary source: `egs/esc50/run_esc.sh`）。最小验证：只训 3 epoch，看旧 16 类在 held-out 上是否保持 ≥ 冻结基线，同时新类是否 >0.8。
2. **长音频必须显式分块（MC-3）**：`ASTFeatureExtractor` 会**静默截断到 1024 帧=10.24s**（HF 源码 `_extract_fbank_features` 的 `fbank[0:max_length]`）。test 有 30 个文件 ≥11s（最长 54.3s），train 侧昆虫类最长 120s。做法：训练时随机 10.24s crop（短文件走原生 zero-pad 路径以匹配 checkpoint 的训练分布），推理时 10.24s 窗 / 5.12s hop，**平均 logits**（softmax 前）。最小验证：只在 39 个 ≥5.5s 的 test 文件上比较「首窗」vs「mean-logits」预测差异数；差异 >5 个即说明聚合有真实影响。
3. **超声昆虫文件的频率下移（时间扩展）预处理（MC-4）**：[LOCAL_FACT] 我实测 Tettigonia Viridissima 的 17 个高采样率文件里，多个把 ~100% 能量放在 8 kHz 以上（如 256 kHz 文件在 <8 kHz 只有 0.2% 能量、峰值 17.7 kHz）；test 里有 2 个同类文件（312500 Hz 峰 20.2 kHz、192000 Hz 峰 10.2 kHz，<8k 能量 0.1–0.2%）。直接重采样到 16 kHz 会把它们变成近乎静音 → 必然错分。对策：若 (native_sr>32k 且 >8kHz 能量占比>0.5)，先按整数因子 k∈{2,4,8} 做时间扩展（把数组当作 sr/k 采样，等价于所有频率 ÷k），使谱质心落到 1–4 kHz 再重采样到 16k；train/test 用完全相同规则。最小验证：对这 17 个 train 文件，比较有/无下移时 provided-checkpoint 特征的类内相似度或微调后 3-fold 上这 4 类的准确率。
4. **保底低算力路线：冻结 encoder + 池化特征 linear probe（MC-2）**：整个 1283 文件前向一次（fp16、10.24s、batch 16）≈ 1–3 分钟，随后 logistic regression / kNN 秒级。它给出一个几乎零风险的可提交 baseline，且能在不消耗微调预算的情况下快速试验（分块聚合方式、频率下移、采样率特征）。最小验证：920 训练样本上 5-fold LR 精度作为地板值。
5. **别把类不平衡当主问题，把「旧 16 类只有 296 条且极不均衡」当主问题**：Sheep/Keyboard Typing 各 3 条。提供的 checkpoint 本身已在旧类上训过（`training_args` 字符串 `./ast-freeze-enc`），因此**保留旧行权重**本身就是最便宜的旧类知识保护；再叠加旧类过采样（把 296 条上采到与新类相当）比调 loss 权重更好控。最小验证：一次提交对比「保留旧行 + 旧类上采样」vs「保留旧行、不上采样」。

## 1. Charter 与实际覆盖范围

覆盖：AST 类扩展头设计、官方微调超参（ESC-50/SpeechCommands recipe）、SpecAugment/mixup 取值、长音频裁剪与聚合、混杂采样率/位深解码、超声信息丢失与补救、低算力保底路线、算力预算。
未覆盖（留给 R2/R3/R4）：metric 定义、CV 切分与 LB 相关性、提交额度策略。
被动放弃：ESC-50 SOTA 榜单（paperswithcode 302）、FSC22 MDPI 全文（403）；这两条不影响上面结论，已用官方 AST recipe + InsectSet 论文替代。

## 2. 对题目分析的核对或修正

- **确认**：`model/config.json` 16 类 id2label、`preprocessor_config.json`（16 kHz/128 mel/1024 帧/mean −4.2677 std 4.5690）、三 CSV 结构与不相交性、长音频存在，全部与 TASK_ANALYSIS_V1 一致。
- **补充/更精确（TASK_ANALYSIS 未量化的关键事实）**：
  - 测试集时长分布（我实测 363 文件）：<5.5s 324 个、5.5–11s 9 个、≥11s 30 个，max 54.27s。**只有约 39/363 需要多窗聚合**——聚合方案最多影响 ~11% 样本，收益上限有限，不要在这上面花掉主要时间。
  - 测试集采样率：44100×242, 48000×80, 24000×26, 96000×11, 11025×1, 8000×1, 312500×1, 192000×1。**没有 384k**，超声只有 2 个文件（占比 0.55%）→ MC-4 的直接分数上限约 +0.005，但它是「几乎确定错 2 个」变「可能对 2 个」的低成本修正。
  - 采样率与类高度相关（train 侧）：Hand Saw 45/45=24 kHz、Chainsaw 38/45=24 kHz、Wolf Howl 23/23=48 kHz、Generator 45/45=44.1 kHz、Tettigonia 独占 ≥192 kHz。test 有 26 个 24 kHz 文件 → 几乎必然是 Chainsaw/Hand Saw。这是**由题目自带训练数据推出的元数据先验**，不是外部泄漏，但属于「数据集拼接痕迹」，泛化风险高，仅建议作为极低置信样本的 tie-break 或后处理审计信号，不建议作为主特征。
  - 昆虫类之外**没有**任何一类是变长的：所有 0–24 类文件都恰好 5.0s（我逐文件核对）。因此「文件时长 >6s ⇒ 属于 25–28」在 train 上是 100% 成立的规则；test 的 39 个长文件很可能全部是昆虫类。此推论可用于诊断（若模型在长文件上预测非昆虫类，说明分块/超声处理出了问题）。
- **修正一处措辞**：ASSET_MAP 说 max_length=1024 帧 ≈10.24s「→ 长音频必须裁剪/分块」。更强的表述是：**不做任何处理也不会报错，会静默丢弃 10.24s 之后的一切**（HF 源码证据，见 R1-F004）。这是最容易被忽略的 silent bug。
- **无冲突**：本轮未发现任何网页证据与本地资产冲突；也未发现任何官方文本要求冻结 encoder（`training_args.bin` 里的 `./ast-freeze-enc` 只是 checkpoint 产出路径，**不是**对我们的约束）。

## 3. 检索方向与来源覆盖

- 官方 AST 仓库 recipe（ESC-50、SpeechCommands）→ 超参 primary source。
- HF transformers AST feature extractor 源码 → 帧数/截断/归一化/mel 上限 8 kHz 的硬事实。
- HF audio-classification 官方任务文档 → 通用微调超参量级（lr 3e-5, bs 32, 10 epoch）。
- PLOS Comput Biol InsectSet 论文全文 → 昆虫超声、mel 前端不适配、5s 分段/3.75s overlap/短文件循环填充、44.1 kHz 标准化。
- Crossref 检索 FSC22 / InsectSet 元数据以定位 primary source。
- 本地实测（FFT 能量分布、采样率/时长普查）作为最高优先级证据。
- 搜索饱和判断：AST 微调超参在三个独立来源上一致收敛到 1e-5~5e-5 区间，停止扩展。

## 4. 关键 Findings

**R1-F001 [EXTERNAL_EVIDENCE] 官方 AST ESC-50 微调 recipe（AudioSet 预训练起点）**：`lr=1e-5`（无 AudioSet 预训练则 1e-4）、`epoch=25`、`batch_size=48`、`freqm=24`、`timem=96`、`mixup=0`、`bal=none`、loss=CE、`lrscheduler_start=5, step=1, decay=0.85`、`audio_length=512`、fstride=tstride=10。
来源：https://raw.githubusercontent.com/YuanGongND/ast/master/egs/esc50/run_esc.sh
相关性：本题起点就是 AudioSet-pretrained AST 且已在 16 类 ESC-50 风格数据上微调过 → 该 recipe 的 lr 量级（1e-5）是最贴近的 primary 先验；mixup=0 说明在 ESC-50 规模（每类 40 条）上作者未从 mixup 获益，而在 SpeechCommands（3.5 万条）上用 mixup=0.6（https://raw.githubusercontent.com/YuanGongND/ast/master/egs/speechcommands/run_sc.sh）。
边界：ESC-50 有 2000 条（每类 40），本题 920 条但类不均衡（3~62）；`audio_length=512` 不可直接搬（见 R1-F004）；batch 48 在 1024 帧输入下显存需求约为 512 帧的 2 倍，Kaggle 单卡应降到 8–16 + grad accumulation。

**R1-F002 [EXTERNAL_EVIDENCE] HF 官方 audio classification 任务文档超参**：lr=3e-5、per_device_train_batch_size=32、grad_accum=4、num_train_epochs=10、warmup_ratio 0.1、load_best_model_at_end + accuracy 选模。
来源：https://huggingface.co/docs/transformers/tasks/audio_classification
相关性：给出「HF Trainer 路线」的默认量级，与 R1-F001 一起把 lr 搜索空间压到 {1e-5, 3e-5}，epoch 压到 8–20。
边界：该教程用 wav2vec2-base 做 16 类意图分类，不是 AST；分类头是全新随机初始化，与我们的「保旧行」情形不同。

**R1-F003 [LOCAL_FACT] ASTFeatureExtractor 的 mel 频带上限 = sampling_rate//2 = 8000 Hz，min 20 Hz，128 mel bins，帧长 25ms/hop 10ms（frame_length=400, hop=160, fft=512, preemphasis 0.97），归一化 `(x-mean)/(2*std)`。
来源（primary，官方源码）：https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/audio_spectrogram_transformer/feature_extraction_audio_spectrogram_transformer.py
相关性：任何 8 kHz 以上的信息在进入 AST 之前就被彻底丢弃，**与是否用 librosa 重采样无关**——这是 MC-4 的根因。

**R1-F004 [LOCAL_FACT/官方源码] 静默截断与常量 padding**：`_extract_fbank_features` 中 `difference<0` 时 `fbank = fbank[0:max_length]`，即超过 1024 帧（10.24s）的部分被无声丢弃；不足则 ZeroPad2d 补 0 **然后**再整体归一化 → padding 区变成常量 −4.2677/(2·4.569) ≈ −0.467。
来源：同 R1-F003。
相关性：(a) 直接把 120s / 54s 波形喂给 extractor 只用到前 10.24s；(b) 提供的 checkpoint 是在 5s 文件（即 ~500 有效帧 + 524 常量帧）上训出来的，所以**短文件应保持同样的 zero-pad 模式**，不要改成循环填充/拼接，否则与 checkpoint 的输入分布不一致。
边界：如果决定改成「循环填充到 10.24s」（InsectSet 论文对短文件的做法），必须整体一致并重新训练，属于可 A/B 的自由度，不是必然更好。

**R1-F005 [LOCAL_FACT，我实测] 昆虫类超声能量分布**：对 fine_tune 中 25–28 类抽样做原生采样率 FFT，>8 kHz 能量占比：Tettigonia Viridissima 最高 0.883（另有 0.451/0.461/0.612）、Pterophylla Camellifolia 最高 0.57、Cicada Orni ≤0.12、Gryllus Campestris ≤0.05。Tettigonia 的 17 个 ≥96 kHz 文件里，有 6 个在 <8 kHz 只剩 0.1%–7% 能量（峰值 10–23 kHz）。
命令可复现：读 `archive/fine_tune.csv` + `soundfile`，对前 8–10s 做 rfft 分带求和。
相关性：结合 R1-F003 → 这些文件重采样到 16 kHz 后信息几乎归零。Gryllus/Cicada 不受影响。

**R1-F006 [LOCAL_FACT，我实测] test 侧受影响样本极少**：363 个 test 文件中只有 2 个 <8 kHz 能量占比 ≈0.001–0.002（sr=312500、192000，时长 13.1s/25.5s）；11 个 96 kHz test 文件全部把 ≥92% 能量放在 <8 kHz（正常声音）。
相关性：为 MC-4 定价——最多影响 2/363 ≈ 0.55% accuracy。**先做便宜且确定的事（分块聚合、扩头保旧、旧类上采样），MC-4 排在其后**。

**R1-F007 [EXTERNAL_EVIDENCE] InsectSet 论文明确指出 mel 前端对昆虫不利**：「insect sounds are much higher in frequency… some up to 150 kHz… the mel-filter bank approach based on human perception is not optimal」「species that produce sounds entirely within the ultrasonic range, which are common in Orthoptera… the lower resolution in high-frequency bands would be increasingly disadvantageous compared to adaptive frontends」；他们**统一到 44.1 kHz**（只收 ≥44.1 kHz 的 WAV，高采样率下采到 44.1k），并用 LEAF 等自适应前端与 mel 对比。
来源：https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011541 （PLOS Comput Biol, 2023-10-04）
相关性：说明「16 kHz mel 前端 + 超声昆虫」是文献已知的失配，不是我在过度推理；同时说明本题原始 InsectSet 数据本来是 44.1 kHz 域的（我们只能在 16 kHz AST 上做，故需要 MC-4 这类频移补救）。
边界：他们的解法（LEAF 自适应前端、更高 sr）在本题不可用（必须用提供的 16 kHz AST checkpoint，且外部模型默认禁止）→ 只能取「频率信息要保住」这一结论，不能取其方法。

**R1-F008 [EXTERNAL_EVIDENCE] InsectSet 的变长音频处理与分段**：「A length of five seconds was chosen… Short files were looped until they reached five seconds. Longer files were sequentially spliced into chunks of five seconds, with an overlap of 3.75 seconds」，切窗到文件尾时环绕回开头补足（剩余 ≥1.25s 才保留）。数据按文件（不是按段）做 60/20/20 划分。
来源：同 R1-F007。
相关性：给出「同源长录音必须按文件分组划分、段级别不能跨 fold」的 primary 依据，以及 75% overlap 的分段惯例可直接迁移到我们的推理分块（我建议 10.24s 窗 / 5.12s hop，overlap 50%，因为 test 只有 39 个长文件、算力不是瓶颈但收益也有限）。
边界：他们是 CNN + 5s 段，段级预测；本题 AST 固定 10.24s，且文件级提交，需要显式聚合。

**R1-F009 [INFERENCE] 提供的 checkpoint 是「冻结 encoder 训头」的产物（`./ast-freeze-enc`，ASSET_MAP 已记录），因此其 encoder 仍非常接近 AudioSet 预训练权重，而其 16 类头是与该 encoder 匹配训出来的。推论：(a) 保留旧 16 行头权重与「未被污染的」encoder 是自洽的，扩头后即使不训也能对旧类给出合理 logits；(b) 因为 encoder 从未被本题数据动过，**全量微调仍有可观 headroom**，不必因为原作者冻结就跟着冻结。
边界：920 条样本全量微调 87M 参数有过拟合风险 → 建议 lr 1e-5、epoch ≤15、SpecAugment(freqm 24/timem 96) 打开，并保留「冻结前 6 层」作为一次 A/B。

**R1-F010 [LOCAL_FACT，我实测] 解码工程细节**：训练+微调集采样率集合 {8k, 11k(仅 test), 16k, 22.05k, 24k, 25.6k, 32k, 44k, 44.1k, 48k, 96k, 192k, 200k, 250k, 256k, 312.5k(仅 test), 384k}；含 float32(fmt=3) 与 WAVE_FORMAT_EXTENSIBLE(24bit)。`soundfile.read(..., always_2d=True).mean(1)` + `librosa.resample`/`torchaudio.functional.resample` 可读全部 1283 个文件（我已全部成功 `sf.info`/部分 `sf.read`），Python 标准库 `wave` 会在 fmt=3 上失败。
相关性：解码失败一个文件就等于随机猜一个 target，且 kernel 可能直接崩；建议对每个文件包 try/except，失败时回退 zeros 并计数打印。

**R1-F011 [INFERENCE] 算力预算**：AST 输入 1024×128 → 1212 patch token + 2 = 1214（与 checkpoint 的 position_embeddings [1,1214,768] 一致），约为 ViT-B/16@224（197 token）的 6 倍序列长度。粗估 Kaggle T4/P100 上 amp 训练吞吐 ~8–15 样本/s：920 样本 × 12 epoch ≈ 11k 前后向 ≈ 15–25 min；推理 363 文件（含长文件分块约 450 窗）< 2 min；特征提取全量 1283 文件 < 3 min。**训练+推理完全能进 kernel 时限，也能进 Agent 的 2h 窗口**，甚至能做 2–3 组配置或 3-fold。
边界：若 GPU 是 T4 且不开 amp/bf16，吞吐可能腰斩；batch 48 在 1024 帧下大概率 OOM，用 batch 8–16 + grad_accum。

**R1-F012 [UNRESOLVED]** 无法从公开 primary source 确认「扩头保旧行 vs 全新头」在本题这种「旧类数据仍然可用」设定下的定量差异；文献（LwF/iCaRL 类）多在旧数据不可用时才需要蒸馏。由于本题旧类数据 296 条**在手**，联合训练已覆盖大部分收益，蒸馏/正则的边际价值低。建议不投时间。

## 5. Method Cards

### MC-1 扩展分类头（保留旧 16 行）+ 全量联合微调
- 家族：迁移学习 / 类扩展（class-incremental with full replay）
- 机制：新建 `nn.Linear(768,29)`，`W[:16]=old_W`, `b[:16]=old_b`，`W[16:]` 用 N(0,0.02) 或 0、`b[16:]=0`；更新 config 的 id2label/label2id 到 29 类；在 920 条（train+fine_tune）上 CE 微调整个模型。
- 解决的具体问题：模型头只有 16 类而 test 空间是 29 类；同时避免重训头导致旧类退化。
- 支持证据：官方 recipe 超参（R1-F001）；HF 文档（R1-F002）；checkpoint 来源自洽性（R1-F009）。
- 适配：一次前向覆盖 29 类，argmax 直接写 `submission.csv` 的 `target` 列（按模板 path 顺序）。
- 与替代方案的结构差异：与「全新 29 类随机头」相比只差 16×768 个数的初始化，但在旧类样本仅 3 条（Sheep/Keyboard Typing）时，保留旧行等价于免费的强先验；与「两阶段（先只训新头再全量）」相比省一半时间。
- 最小验证：3 epoch × 快速 split，观察旧类 recall（尤其 Sheep/Keyboard Typing）与新类 accuracy；或直接 0 epoch 检查扩头后旧类 logits 与原 16 类模型一致（数值等价性 sanity check，必须完全相同）。
- 成本/依赖：~15–25 min GPU；仅需 torch + transformers（pinned wheel 内已含 transformers 5.10.2）。
- Kaggle kernel 可行性：**可行**（离线、单 GPU、全部训练+推理 < 40 min）。
- 失败模式：lr 过大（≥1e-4）导致旧类灾难性遗忘；类不均衡使 62 条的 Thunderstorm 吸走稀有类；忘记同步更新 config.num_labels 导致 shape 错误。
- 置信度：高（超参有 primary source，结构改动零风险且可数值验证）。
- 污染/规则风险：无（只用题目自带 checkpoint 与数据）。

### MC-2 冻结 encoder 特征 + 线性探针/kNN（保底 & 快速实验台）
- 家族：linear probing
- 机制：用 provided AST 取 pooled hidden state（`ASTModel` 输出的 pooler / mean over tokens，768 维），对 920 条拟合 LogisticRegression（class_weight='balanced'）或 kNN。
- 解决的问题：给出一个 20 分钟内可提交、几乎不可能崩的 baseline；并作为「预处理变体（频移、分块聚合、循环填充）」的秒级 A/B 试验台，不必每次重训。
- 支持证据：AST/AudioSet 表征通用性（R1-F001 中 audioset_pretrain=True 使 lr 从 1e-4 降到 1e-5，说明预训练表征已很接近目标任务）；[INFERENCE]。
- 适配：特征只需算一次并缓存到 `/kaggle/working`，长文件按窗取特征后 mean-pool。
- 结构差异：不更新 encoder → 上限低于 MC-1，但方差极小。
- 最小验证：5-fold（按文件分组）LR accuracy；与 MC-1 短训结果对比作为地板。
- 成本：1–3 min GPU 前向 + 秒级 sklearn。
- Kaggle kernel 可行性：**可行**。
- 失败模式：pooled 特征对昆虫细类分辨不足；sklearn 与 pinned 环境版本不匹配（应确认 wheel 内含 scikit-learn，否则用 torch 手写 softmax 回归）。
- 置信度：中高。
- 风险：无。

### MC-3 长音频窗口化 + logits 聚合
- 家族：多实例推理 / test-time aggregation
- 机制：训练：>10.24s 的文件每个 epoch 随机 crop 一段 10.24s（等价于时间域增强）；也可把长文件切成 K 段作为 K 个训练样本（昆虫类样本量从 240 涨到数百，对 4 个新类有帮助）。推理：10.24s 窗、hop 5.12s，收集各窗 logits → **mean logits**（等价 log-prob 平均，比 max-prob 稳），argmax。
- 解决的问题：R1-F004 的静默截断；test 30 个 ≥11s 文件（最长 54.3s）。
- 支持证据：InsectSet 的 5s/3.75s-overlap 分段与短文件循环填充（R1-F008）；HF 源码截断行为（R1-F004）。
- 适配：只对 39 个 ≥5.5s 的 test 文件生效，其余走单窗；推理额外成本 < 1 min。
- 结构差异：mean-logits vs majority-vote vs max-prob：mean-logits 在类不均衡+段内静音时更鲁棒（[INFERENCE]，因为静音段 logits 平坦、对均值贡献小，而投票会让静音段投出一票）。
- 最小验证：对 39 个长 test 文件比较「首窗 argmax」与「mean-logits argmax」的不一致数；再用 train 侧长昆虫文件（120s）做 held-out 精度对比。
- 成本：可忽略。
- Kaggle kernel 可行性：**可行**。
- 失败模式：把长文件切成多段训练会让 4 个昆虫类严重过采样，需按文件加权（每段权重 1/K）；窗口过多导致长文件被静音段主导。
- 置信度：中高（截断事实确定；聚合方式的相对优劣未在本题实测）。
- 风险：无。

### MC-4 超声文件的时间扩展 / 频率下移预处理
- 家族：信号域域适配（heterodyne/time-expansion，蝙蝠与昆虫生物声学惯用）
- 机制：对每个文件在原生 sr 下算 >8 kHz 能量占比 r 与谱质心 c；若 `native_sr>32000 and r>0.5`，取整数因子 k=2^ceil(log2(c/2000))（限 k≤8），把波形**当作 sr/k 采样**（不重采样，仅改声明的采样率）后重采样到 16 kHz → 所有频率 ÷k、时长 ×k，超声内容进入 mel 覆盖的 20–8000 Hz。train 与 test 用同一函数（保证一致）。
- 解决的问题：R1-F005/F006 的「重采样后近乎静音」；R1-F003 的 8 kHz 硬上限；R1-F007 的文献级失配警告。
- 支持证据：R1-F003/F005/F006/F007。
- 适配：时长 ×k 后必须配合 MC-3 分块（k=4 会把 13s 变 52s）；受影响 test 文件仅 2 个（≤+0.55% accuracy）。
- 结构差异：不同于「重采样」（丢信息）与「换前端」（本题不允许换模型）：它把信息搬进模型能看见的频带，不改模型、不引入外部资产。
- 最小验证：先只在 17 个 train 超声文件 + MC-2 特征上验证：下移后这些文件的最近邻是否变成同类昆虫；若否，直接放弃。
- 成本：CPU 侧 O(文件)，秒级。
- Kaggle kernel 可行性：**可行**（纯 numpy/soundfile/torchaudio）。
- 失败模式：阈值误触发把正常声音（如 96 kHz 的烟花/直升机）错误下移 → 我实测 test 的 11 个 96 kHz 文件 r<0.08，阈值 0.5 有很大安全边界，但仍必须打印被触发的文件数并人工核对；k 过大导致 mel 里全是低频糊团；下移后的谱与 AST 训练分布不符，模型可能仍答错（只是从「必错」变「可能对」）。
- 置信度：中（机制确定，收益上限小且未实测）。
- 风险：无外部数据/模型。

### MC-5 不均衡处理：旧类过采样 + （可选）class-balanced CE
- 家族：重采样 / 损失加权
- 机制：把 296 条旧类样本按类上采样到与新类可比（或用 WeightedRandomSampler，权重 ∝ 1/√n_c），Sheep/Keyboard Typing（各 3 条）配合 SpecAugment 防死记；不建议一开始就上 class_weight ∝1/n（对 3 样本类会放大噪声）。
- 解决的问题：train 侧 62:3 的极端不均衡 + 新类 624 条压倒旧类 296 条。
- 支持证据：官方 ESC-50 recipe 用 `bal=none`（ESC-50 本身均衡），AudioSet recipe 才用平衡采样 → 说明平衡采样是针对不均衡数据的官方开关（R1-F001）；[INFERENCE] 本题不均衡程度介于两者之间。
- 适配：不改 metric 假设，纯训练侧；若 metric 是 macro-F1（R2 负责确认），权重应更激进。
- 最小验证：一次提交 A/B（√倒数权重 vs 无权重）；本地按文件分组 CV 同时看 macro 与 micro。
- 成本：0。
- Kaggle kernel 可行性：**可行**。
- 失败模式：3 样本类被过采样 20 倍 → 过拟合到那 3 条录音的背景噪声。
- 置信度：中。

### MC-6 元数据先验（采样率/时长）作为 tie-break —— 谨慎，非主推荐
- 家族：数据集指纹 / 后处理
- 机制：由 train 统计得到 `sr→类` 与 `duration>6s→25..28` 的经验先验，仅在模型 top-1 与 top-2 概率差 < τ 时用于打破平局。
- 解决的问题：拼接数据集的录音条件与类强相关（R1-F002 节以外的本地统计：Hand Saw 45/45=24 kHz、Wolf Howl 23/23=48 kHz、所有非昆虫类恰好 5.0s）。
- 支持证据：全部来自本地 CSV+音频头统计（第 2 节）。
- 适配：只影响少量样本；实现是纯 pandas 后处理。
- 结构差异：不是声学建模，而是利用采集流水线痕迹。
- 最小验证：本地 CV 上统计「若按 sr 先验强制覆盖」会改动多少样本、其中多少改对；改对率 <60% 则弃用。
- 成本：0。
- Kaggle kernel 可行性：可行。
- 失败模式：test 的录音条件分布若被组织者刻意打散（如统一转码），先验失效并伤害成绩；这是**过拟合到伪相关**的典型。
- 置信度：低（机制真实但泛化不确定）。
- 风险：不涉及外部数据，属于合法特征工程；但应在报告中披露。

## 6. 负面结果、失败条件与冲突证据

- **不要盲抄 `audio_length=512`**：官方 ESC-50 用 512 帧（5.12s），但本题 checkpoint 的 position_embeddings 是 1214（对应 1024 帧）。改短需要对位置编码做插值/裁剪，HF 侧不会自动做，且会偏离 checkpoint 的输入分布。判定：**保持 1024 帧**。（冲突：官方 recipe vs 本地 checkpoint → 以本地资产为准。）
- **mixup 在这个数据规模上无官方支持**：ESC-50 recipe mixup=0；只有在 3.5 万条的 SpeechCommands 上才用 0.6。判定：默认不用 mixup，最多作为最后一次 A/B。
- **蒸馏/LwF 大概率是浪费**：旧类数据在手，联合训练即完整 replay（R1-F012）。
- **换更高采样率前端 / LEAF / 更强模型不可用**：外部模型默认禁止，且 AST checkpoint 固定在 16 kHz/128 mel（R1-F003、R1-F007 边界）→ InsectSet 论文的最优解法在本题 **[INFEASIBLE_FOR_KERNEL / 规则不允许]**，仅作思路来源。
- **不要用 `wave` 模块**：float32/24-bit extensible 文件会抛错（R1-F010）。
- **不要把整段长音频交给 feature extractor 就以为处理完了**：静默截断（R1-F004）。
- **聚合与超声修正的收益上限被数据量钉死**：分块只影响 39/363、超声只影响 2/363（R1-F006）。若本地 CV 与 LB 不一致（系统历史教训：音频任务 LB +0.096 对应本地 CV −0.127），不应把提交额度花在这些小改动的反复 A/B 上，而应优先验证 lr/epoch/扩头这类影响全部 363 个样本的因素。
- 未能验证：ESC-50 公开榜单具体数字（paperswithcode 302 重定向）、FSC22 论文正文（MDPI 403）。因此「AST 在 ESC-50 达 95.6%」这一常见说法我**没有**取得 primary 证据，报告中不作为依据。

## 7. 未解决问题

1. 扩头保旧行 vs 全新头在本题的定量差异（未找到同设定 primary 证据；建议用一次提交或一次本地 3-fold 直接测）。
2. 全微调 vs 冻结底部 N 层的最优分界（920 样本）：无 primary source；建议 A/B「全量」vs「冻结 embeddings+前 6 层」。
3. mean-logits vs max-prob vs 投票的相对优劣（本题未实测）。
4. 频率下移的最佳因子选择规则与它是否真能被 AST 正确识别（只有 2 个 test 文件可验证，几乎无法统计验证 → 只能靠 train 侧 17 个文件）。
5. pinned wheel 内是否含 scikit-learn / librosa / torchaudio（本地 bundle 没有 wheel 文件清单，`ioai-starter.py` 只装 meta-package `ioai-env`）→ MC-2/MC-4 的实现需在 kernel 内先 `import` 探测并准备 numpy-only 回退。
6. metric 与验证切分（按 Charter 归 R2）。

## 8. 给题目分析 Agent 的合并建议

- 把「test 时长/采样率分布」的实测数字（第 2 节）并入 TASK_ANALYSIS 的数据结构段：它把「长音频处理」从「重大自由度」降级为「影响 39/363 的次要自由度」，直接改变时间分配。
- 把 R1-F004（静默截断）与 R1-F003（8 kHz mel 上限）作为**必读实现约束**，与 submission 契约同级列出。
- 把 R1-F005/F006（超声昆虫）列为独立风险条目，并注明收益上限 ~0.55%，避免下游过度投入。
- 采用 MC-1 的超参作为默认起点（lr 1e-5、bs 8–16+accum、epoch 10–15、freqm 24/timem 96、mixup 0、第 5 epoch 起 ×0.85），把「lr∈{1e-5,3e-5}」和「冻结/不冻结」列为最优先的两个 A/B。
- 明确记录：`training_args.bin` 中的 `./ast-freeze-enc` 只是 checkpoint 产出路径，**不构成对我们冻结 encoder 的要求**；本轮未发现任何官方文本禁止修改模型结构。
- 建议下游先落地一条「MC-2 保底提交」再做 MC-1，确保任何时刻都有有效 submission。

## 9. 来源表

| 标题 | URL | 日期/版本 | 类型 | 支持 Finding |
|---|---|---|---|---|
| AST 官方仓库 ESC-50 微调脚本 `run_esc.sh` | https://raw.githubusercontent.com/YuanGongND/ast/master/egs/esc50/run_esc.sh | master, 取自 2026-08-04 | 官方代码仓库（primary） | R1-F001, MC-1, MC-5, §6 |
| AST 官方仓库 SpeechCommands 脚本 `run_sc.sh` | https://raw.githubusercontent.com/YuanGongND/ast/master/egs/speechcommands/run_sc.sh | master, 取自 2026-08-04 | 官方代码仓库（primary） | R1-F001（mixup 对比）, §6 |
| AST 官方仓库 ESC-50 数据准备脚本 | https://raw.githubusercontent.com/YuanGongND/ast/master/egs/esc50/prep_esc50.py | master, 取自 2026-08-04 | 官方代码仓库 | R1-F001（5-fold/官方 fold 惯例背景） |
| transformers `ASTFeatureExtractor` 源码 | https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/audio_spectrogram_transformer/feature_extraction_audio_spectrogram_transformer.py | main, 取自 2026-08-04 | 官方源码（primary） | R1-F003, R1-F004, MC-3, MC-4 |
| HF 官方任务文档 Audio classification | https://huggingface.co/docs/transformers/tasks/audio_classification | 取自 2026-08-04 | 官方文档 | R1-F002, MC-1 |
| Faiß & Stowell, Adaptive representations of sound for automatic insect recognition | https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011541 | PLOS Comput Biol, 2023-10-04 | 同行评议论文（primary） | R1-F007, R1-F008, MC-4 |
| 同上 DOI 元数据（Crossref 定位） | https://doi.org/10.1371/journal.pcbi.1011541 | 2023 | 元数据 | R1-F007 |
| FSC22: Forest Sound Classification Dataset | https://doi.org/10.3390/s23042032 | Sensors, 2023-02-10 | 同行评议论文（正文 403 未取得） | §6 未验证项（16–24 类来源假设） |
| 本地资产：`archive/model/config.json` / `preprocessor_config.json` | 本机只读路径 `.../ORIGINAL_ASSETS/archive/model/` | 本次 bundle | 官方题目资产（最高优先） | R1-F009, R1-F011, §2 |
| 本地资产：`archive/{train,fine_tune,submission}.csv` + 1283 个 wav 头/FFT 实测 | 本机只读路径 `.../ORIGINAL_ASSETS/archive/` | 本次 bundle | 官方题目资产 + 本人实测 | R1-F005, R1-F006, R1-F010, §2 |
| 本地资产：`ioai-starter.py`（离线 wheel 安装、`find_input`） | `.../ORIGINAL_ASSETS/ioai-starter.py` | 本次 bundle | 官方题目资产 | §7-5, MC-1/MC-2 依赖判定 |
