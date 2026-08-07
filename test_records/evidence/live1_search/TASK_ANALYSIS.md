# TASK_ANALYSIS.md（最终版，可独立阅读）

## 1. 任务目标

给定题目自带的**已在 16 个旧类上微调好的 Audio Spectrogram Transformer (AST-base) checkpoint**、旧类训练表（296 条）与 **13 个新类的微调表**（624 条），为 `submission.csv` 列出的 **363 条测试音频**各预测一个类别 id。标签空间为 **29 类（0–28）**。本质是「旧数据完整可得」的增类微调 = 普通的 920 条小数据 29 类监督分类问题，**不是内存受限的 class-incremental 问题**。[LOCAL_FACT + EXTERNAL_EVIDENCE]

预测单位：单个 wav 文件 → 一个整数 `target`。

## 2. 输入 / 目标 / 输出

- 输入：`archive/audio/<16hex>.wav`。采样率 8k–384k 混杂，单/双声道，编码含 PCM16 / PCM24 / IEEE-float32 / WAVE_FORMAT_EXTENSIBLE。模型侧要求 **16kHz 单声道 float**（`preprocessor_config.json`）。
- 目标语义：0–15（Dog, Rooster, Pig, Cow, Frog, Cat, Hen, Sheep, Crow, Mouse Click, Keyboard Typing, Thunderstorm, Sea Waves, Bird Chirping, Wolf Howl, Rain）来自 `model/config.json`；16–28（Crackling Fire, Axe, Chainsaw, Generator, Hand Saw, Vehicle Engine, Helicopter, Gunshot, Firework, Pterophylla Camellifolia, Cicada Orni, Gryllus Campestris, Tettigonia Viridissima）来自 `fine_tune.csv`。
- 输出：`/kaggle/working/submission.csv`。

## 3. metric 与 submission contract

- **metric 未在本地任何资产中给出 [UNRESOLVED]**。候选：accuracy / macro-F1 / balanced accuracy。三者的最优决策规则在稀有类上**零和**（见 §7）。
- **可直接测量 metric 的手段（无需模型）**：提交「全部预测类 c」的常量提交。accuracy → 得分严格 = n_c/363（可反解该类计数）；macro-F1 → ≈ (1/29)·2n_c/(363+n_c)（量级小一个数量级以上，例如 n_c=24 时 0.066 vs 0.0043）；balanced accuracy/macro-recall → 恒为 1/29 = 0.0345 与 c 无关。→ 1–2 次提交即可判定 metric 族并顺带测出测试先验。[R3-MC-1]
- submission 格式冲突（必须记录，不得静默取舍）：`archive/submission.csv` 表头 `path,target`；`ioai-starter.py` 占位写 `id,prediction` 单行 `0,0`。三方独立确认。判断：starter 是跨题通用模板，**本题应按 `path,target`，做法是读入模板 CSV、只替换 target 列、不改行序与路径字符串**；仍需 Kaggle 题面确认。[LOCAL_FACT + INFERENCE]

## 4. 数据结构与重要统计

- 三 CSV path 两两无交集，并集 1283 = audio 目录全量。`split` 列常量 `train`（无官方 val）。
- 旧 16 类 296 条，**极端不均衡**：Sheep 3、Keyboard Typing 3、Rain 7 … Thunderstorm 62、Frog 35、Bird Chirping 35。
- 新 13 类 624 条：Crackling Fire 24、17–24 各 45、25–28（四种鸣虫）各 60。
- 时长/编码/采样率全量统计与三条修正见 `ASSET_MAP.md` §3–§4。核心两条：
  - **labeled 中非 5.00s 文件 238 条，100% 属昆虫类 25–28；召回 238/240=0.992**（全量核验）。测试集 40 条非 5s。
  - 测试集 363 条**全部 fmt1/16-bit**（统一重编码），训练侧昆虫子集含 float32/24-bit。
- **测试先验 ≠ labeled 先验**：若分层，昆虫应占 363×240/920≈94 条且几乎都非 5s；实测测试仅 40 条非 5s → 测试昆虫占比推测 ≈11% 而非 26%。这直接削弱一切「用 labeled 频率做 balanced-softmax / 反频率加权」的默认做法。[LOCAL_FACT + INFERENCE]
- 若测试按 labeled 先验分层，稀有类期望条数：Sheep≈1.2、Keyboard Typing≈1.2、Rain≈2.8 → **macro-F1 下一条 Sheep 样本值 ≈1/29=0.0345，accuracy 下值 1/363=0.0028，杠杆差 12.5 倍**。

## 5. 比赛提供资产的作用

- `archive/model/` 是唯一可用的强 audio backbone（离线 kernel 不能下载 AudioSet 权重）。`classifier.dense` 只有 16 维；`training_args.bin` 的 `./ast-freeze-enc` 表明它是**冻结 encoder 只训头**的产物 → encoder ≈ 未被下游扭曲的原版 AudioSet AST，旧类知识集中在 16 维头 + 296 条旧类数据。[LOCAL_FACT + INFERENCE]
- 工程硬约束（源码级证据，R2）：
  1. **禁止依赖 `from_pretrained(num_labels=29, ignore_mismatched_sizes=True)`**：它把 `classifier.dense` 判为 mismatched 并**重新随机初始化整个 29×768**，静默丢掉本 checkpoint 唯一被训练过的部分，只打 warning。正确做法是手工扩头、把旧 16 行权重/偏置拷入。
  2. **禁止改 `max_length`/`num_mel_bins`**：HF AST 的 `position_embeddings` 是纯 `nn.Parameter`，forward 无插值；shape 不匹配时 `_init_weights` 会 `zeros_` 清零 position_embeddings/cls_token/distillation_token，得到"半毁"模型且 loss 仍会下降。若必须缩短窗口，须按官方 `YuanGongND/ast` 做时间维中心切片再手工写回。
  3. `ASTFeatureExtractor` **不会自动重采样**：`sampling_rate != 16000` 直接 ValueError，不传则静默用错。归一化是 `(x-mean)/(std*2)`，pad 发生在归一化**之前**（零 pad 区归一化后是常数 +0.4670）。torchaudio-kaldi 后端与 numpy 回退路径**数值不等价** → 训练/推理必须同一后端，并用「旧类 top-1 探针」确认与 checkpoint 训练分布一致。
- 是否**强制**使用该 checkpoint 无明文；离线约束下事实上必需。

## 6. 规则、资源与网络边界

- 计分 kernel：**无互联网**，必须用挂载的 `ioai-2026-wheel-dataset` 离线安装 `ioai-env`（starter 安装块必须在任何 import 之前，整段保留不改；uv 40s vs pip 回退 ~6min），单 GPU，训练+推理都在 kernel 时限内完成。路径不得硬编码，用 `rglob`。
- 额外外部数据集/预训练模型**默认禁止**（本地无题面允许文字 → 按禁止处理）。
- 单题 ≤2 小时 Agent 时间、**≤50 次提交**。提交额度是一等测量资源，应按策略刻意花。
- 允许的研究/使用边界（本 bundle 遵循）：可读公开方法学（AST 源码、论文、ESC-50 README、sklearn 文档）；**不得**引入外部数据/权重进入 kernel；不得检索测试标签或他人提交文件；不得上传本地资产。
- 历史教训目录本次为空（`context/prior_lessons/README.md` 明示"本次运行没有提供历史教训目录"）→ 唯一可用的一等实测教训是系统背景中的音频族反号案例（LB +0.096 / 本地 5-fold CV −0.127）。

## 7. 验证与泄漏风险

1. **无官方 val**，920 条、最小类 3 条。363 条测试上 accuracy 的 95% 二项 CI 半宽 ≈0.045–0.050；920 条 5-fold 单折 val≈184 条 CI 半宽 ≈0.06–0.07 → **单折数字几乎无信息**。判据：只接受同一批折上配对 bootstrap 95% CI 不含 0 的改动；LB 与本地单折 <0.02 的差异一律视为噪声。
2. **同源片段近重复泄漏**：旧类与部分新类的类目/5s/44.1kHz 规格与 ESC-50 高度一致；ESC-50 官方 README 明言 clips 由 Freesound 长录音切出且「同一源文件的片段必须落在同一折」。本题文件名已哈希 → fold/clip/take 信息全丢 → **随机分层 CV 在这类数据上系统性乐观**。注意类计数（Thunderstorm 62 ≠ ESC-50 的 40/类）说明数据已被重组，任何公开 fold 清单都不可复用（也不允许引入）。
3. **廉价频谱指纹不能做近重复检测（R3 实测负面结果）**：256 维粗指纹 LOO 1-NN 仅 0.363，cos>0.95 的对同标签率仅 0.257。→ 不得据此建组或宣称无泄漏；若要分组，用题目自带 checkpoint 的 CLS embedding + 正/负对照阈值校准。
4. **长录音切窗必须同折**；训练随机裁窗与推理窗口策略必须一致，否则 CV/LB 差异不可归因。
5. **元数据捷径**：时长门控是近确定约束（labeled precision 1.000），但测试端处理方式未知（测试中有 1 条 11025Hz@5.0s 可疑样本）；硬约束最坏损失 40/363≈11%。建议先用 `dur>8s` 或软偏置 δ=3 而非 ∞，并用 1 次 LB 提交做单变量 A/B。
6. **新类 logit 偏置**：扩头后新类样本 2 倍于旧类且新头随机初始化，文献（WA/BiC）报告最后 FC 层对新类系统性偏高 → 需要 post-hoc per-class bias（见 RESEARCH_SYNTHESIS）。**温度缩放对硬标签 metric 完全无效**（argmax 不变），不要花时间。

## 8. 证据分区

### [LOCAL_FACT]（阶段 A/B 亲自核对，多数经三方独立重算）
1. 1283 wav = 三 CSV path 并集，两两无交集；`split` 常量 `train`；`submission.csv` target 全 0（无标签信息）。
2. 29 类标签空间（0–15 来自 config.json，16–28 来自 fine_tune.csv）。
3. checkpoint：AST-base，`classifier.dense [16,768]`，`position_embeddings [1,1214,768]`，preprocessor 16kHz/128mel/1024帧/mean −4.2677393/std 4.5689974。
4. labeled 非 5.00s 文件 238 条 100% 属 25–28；昆虫 240 条中仅 2 条为 5.00s；测试 40 条非 5s（5.3–54.3s）。
5. 测试集 363 条全部 fmt tag 1/16-bit；训练侧昆虫含 fmt 3 ×3、fmt 0xFFFE 24-bit ×3、0xFFFE 16-bit ×17；`wave.open` 在 fmt 3 上抛异常。
6. 采样率：24000Hz 仅见 16–24；96000 跨三类子集；≥192kHz 仅 T. viridissima（labeled 17、测试 2）。
7. submission 模板 `path,target` vs starter 占位 `id,prediction` 冲突。
8. starter 离线 wheel 安装契约（置顶、glob 搜索、symlink 改名 + `UV_SKIP_WHEEL_FILENAME_CHECK=1`、uv 失败回退 pip）。

### [EXTERNAL_EVIDENCE]（来源见 RESEARCH_SYNTHESIS 来源表）
1. HF AST 头结构 = `layernorm(768)+Linear(768,num_labels)`；`ignore_mismatched_sizes` 会重初始化整头；`_init_weights` 对 ASTEmbeddings 执行 `zeros_`；FE 不自动重采样、归一化含 ×2、pad 先于归一化；`is_speech_available()` 决定 kaldi vs numpy 后端。
2. transformers v5 破坏性变更：`Trainer(tokenizer=)`→`processing_class=`、`evaluation_strategy` 等已弃用参数被移除、`from_pretrained` 默认 `dtype="auto"`；`ASTFeatureExtractor` 保留。
3. CIL 综述（Masana et al.）：joint training on all seen data 是所有增量方法的**上界**。
4. PANNs：ESC-50（32 clips/类）fine-tune 0.947 / frozen 0.908–0.918 / scratch 0.833；**<10 clips/类时 frozen 反而最好**。
5. AST 官方 ESC-50 recipe：lr **1e-5**、batch 48、f/t masking → 5-fold 95.75%；无 AudioSet 预训练时掉到 ~88.7%。
6. LP-then-FT（Kumar et al.）：强分布偏移下全微调 ID +2% 但 OOD −7%，LP-FT 双优。
7. Logit adjustment（Menon et al.）：post-hoc `logits − τ·log π` 统一并优于常见 re-weighting。
8. WA / BiC：类增量后最后 FC 层对新类权重/logit 系统性偏高，可用权重对齐或小平衡集学偏置修正。
9. PSLA：checkpoint 权重平均 + ensemble 在音频 tagging 上稳定加分（0.444→0.474 mAP）。
10. ESC-50 官方 README：fold 规则（同源片段同折）+ leaderboard（MFCC+RF 44.3%、human 81.3%、CNN 64.5%、AST 95.7%、BEATs 98.1%）。
11. sklearn macro-F1：不考虑类不均衡；测试集中出现但从未被预测的类 F1=0 仍计入 29 类平均。
12. T. viridissima 鸣声基频 ≈10kHz + 20–60kHz 超声带 → 16kHz 重采样会丢基频带。

### [INFERENCE]
1. 测试集覆盖 29 类联合标签空间（依据时长/采样率双分布 + 测试集混合两个数据源）。
2. checkpoint encoder ≈ 未被扭曲的原版 AudioSet AST。
3. 测试昆虫占比 ≈11%，与 labeled 26% 不同（→ 反对以 labeled 先验做重加权）。
4. 5.00s 音频 → kaldi fbank ≈498 帧，被 pad 到 1024 帧，即约 51% 输入是常数 pad；checkpoint 是在这种分布上训头得到的 → 长录音是否也只取 5s 窗以保持 pad 结构一致，是一个值得 A/B 的开放选项。
5. 计算预算：920 条特征缓存后 AST-base 单 GPU 每 epoch 约 1–2 分钟；瓶颈是解码/重采样长超声文件与 CPU fbank。fp32 全量缓存 ≈6.7GB、fp16 ≈3.4GB。
6. 分数量级参考（不作承诺）：传统特征 35–55%、frozen LP 80–90%、联合微调 88–95%。

### [UNRESOLVED]
1. 官方 metric（accuracy / macro-F1 / balanced accuracy）→ 优先看题面；否则用常量类探针 1–2 次提交解决。
2. 测试集是否含全部 29 类、public/private 划分比例与是否分层。
3. 测试昆虫样本是否被裁剪到 5s（决定时长硬约束的安全性）。
4. wheel 环境内是否含 torchaudio / soundfile / librosa / scipy / sklearn / LightGBM（决定音频加载分支、FE 走哪条后端、兜底方案选型）→ kernel 内 `importlib.util.find_spec` + `transformers.utils.is_speech_available()` 首先确认。
5. checkpoint 当初的 fbank 后端与归一化细节 → 只能用「旧类 top-1 探针」反推。
6. kernel 时限与 GPU 型号。
7. `training_args.bin` 精确超参（未反序列化）。
8. transformers 5.10.2 与 main(5.14.x) 的 AST 逐版差异未核 → kernel 内以 `inspect.getsource` 为准。
9. AST embedding 能否分离「同源片段」与「同类不同录音」（近重复分组的可行性未验证）。

## 9. 阶段 A 被修正的内容

| 阶段 A 表述 | 修正 | 证据 |
|---|---|---|
| wav "全部 16 位 PCM" | 昆虫子集含 float32/24-bit/EXTENSIBLE；测试集全部 fmt1/16bit | R2 实测 + 阶段 B 复现（全量 fmt chunk 统计） |
| "采样率>100kHz ⇒ 昆虫" 视为捷径 | 降级为弱单向信号（≥192kHz 仅 T. viridissima，测试 2 条，上限收益 0.006）；96000 跨三子集 | 全量采样率交叉表 |
| "时长>6s ⇒ 昆虫" 视为捷径/陷阱 | 升级为一级近确定标签空间约束（labeled precision 1.000 / recall 0.992），但施加方式需 LB 验证 | 全量 238/238 核验 |
| 未提及扩头风险 | 新增硬约束：禁止依赖 `ignore_mismatched_sizes`、禁止改 max_length/num_mel_bins | transformers 源码 |
| 未提及测试先验 | 新增：测试先验 ≠ labeled 先验（昆虫 ~11% vs 26%） | 时长/采样率交叉对比 |
| 隐含考虑 CIL 算法族 | 明确否定：旧数据 100% 可用 ⇒ joint fine-tune 即文献上界，LwF/EWC/iCaRL 无收益 | Masana survey |
