# R2 Research Report

## 0. Top-5 可行动方向
1. **手工扩展分类头 16→29（禁用 `ignore_mismatched_sizes` 依赖）**：先 `from_pretrained(..., num_labels=16)` 正常加载，再新建 `nn.Linear(768,29)`，把旧 `classifier.dense.weight/bias` 前 16 行拷入，新 13 行小尺度初始化，替换 `model.classifier.dense` 并同步 `config.num_labels/id2label/label2id`。最小验证：加载后对 `train.csv` 20 条旧类样本前向，检查 argmax(logits[:, :16]) 与旧标签一致率≥旧模型水平（应 >0.8）。（F001/F002/MC-1）
2. **绝不改 `max_length` / `num_mel_bins`**：HF AST 的 position_embeddings 是纯 Parameter，无插值逻辑；shape 不匹配时 `_init_weights` 会把它 **zeros_ 全清零**，静默毁掉 backbone。必须固定 1024×128；若确实要缩短窗口，须照官方 ast 仓库做时间维中心切片再手工赋值。最小验证：改 max_length=512 加载后对同一 batch 前向，比较 logits 与 1024 版是否退化为近乎均匀分布。（F003/MC-2）
3. **音频 I/O 走 `soundfile`/`torchaudio`，绝不用 `wave`；统一到 float[-1,1] 单声道 16 kHz**：本地 fine_tune 昆虫文件含 IEEE-float(fmt 3)、24-bit、WAVE_FORMAT_EXTENSIBLE(0xFFFE)，Python `wave` 直接抛 `unknown format: 3`（我已实测）。振幅尺度不一致会让 log-mel 偏移 ~log(32768)。最小验证：对 3 个 fmt=3 与 3 个 fmt=65534 文件读出后打印 dtype/max abs，确认都在 ~1.0 量级。（F004/F005/MC-3）
4. **训练随机裁剪 10.24 s 窗 + 推理多窗 mean-logit 聚合**：变长昆虫录音（最长 120 s）与测试集 40 条 >5.3 s 样本必须窗口化；同一录音的窗必须同折。最小验证：只对测试集 >5.3 s 的 40 条比较 "首窗" vs "全窗均值" 的预测差异条数；若差异 <5 条则不值得花提交额度。（F006/F009/MC-4）
5. **超声其实几乎不是问题**：>100 kHz 采样率在本地只属于 Tettigonia Viridissima（60 条中 17 条），测试集仅 2 条（312500Hz、192000Hz）。降到 16 kHz 会丢掉 T. viridissima 的 10 kHz 基频与 20–60 kHz 超声带，但该类另有 41 条常规 44.1/48 kHz 录音可学；对 2 条测试样本可用"时间扩展"（按 1/12 速率重采样把超声搬进 <8 kHz）做单独实验，收益上限 2/363≈0.006。（F007/F008/MC-5）

## 1. Charter 与实际覆盖范围
覆盖：HF transformers（main=5.14.x，checkpoint 声明 5.10.2）AST 源码级用法与坑（scope 1，深度覆盖）；变长/异常采样率音频工程路径（scope 2，覆盖 + 本地实测）；ESC-50/昆虫声学工程实践（scope 3，部分覆盖：昆虫频谱学证据到手，增强配方未做穷尽检索）；Kaggle 离线 kernel（scope 4，主要基于 starter 源码 + 推断）。
未展开（按 forbidden_overlap 交给 R3/R4）：metric/验证协议/阈值/泄漏检测方法论；类增量算法族与不均衡损失理论。
时间盒内未完成：SpecAugment/mixup 在 <1k 样本上的定量文献对比、Kaggle AST notebook 逐个复盘。

## 2. 对题目分析的核对或修正
- [LOCAL_FACT 修正] ASSET_MAP 称 wav "全部 16 位 PCM，部分 WAVE_FORMAT_EXTENSIBLE"。实测（手工解析 fmt chunk，路径 `.../ORIGINAL_ASSETS/archive/`）：`fine_tune.csv` 昆虫类含 **fmt tag 3（IEEE float32）3 个文件**、**fmt 0xFFFE 24-bit 3 个**、**0xFFFE 16-bit 17 个**；`train.csv` 全为 fmt 1/16-bit（168 mono + 128 stereo）。测试集 363 条 **全部 fmt 1 / 16-bit**（已统一重编码）。→ 读音频库选择与振幅归一化是真实风险，且"文件格式"不能当测试端捷径。
- [LOCAL_FACT 补充] 超声采样率 **只出现在 Tettigonia Viridissima**：{192000:2, 200000:1, 250000:1, 256000:9, 384000:3, 96000:1, 25600:1, 32000:1, 44100:26, 48000:15}。Pterophylla{32000:1,44100:49,48000:10}、Cicada Orni{8000:2,44100:41,48000:17}、Gryllus{44100:40,48000:18,96000:2}。ASSET_MAP 把 8k–384k 笼统归给"昆虫类"，实际分布高度按物种聚集 → 采样率捷径在昆虫**内部**也带信号（但样本极少，易过拟合；细节归 R3 泄漏轴）。
- [LOCAL_FACT 补充] 昆虫时长中位数 13–16 s，最大恰好 120.0 s（三个类都有 120.0 命中 → 上游按 120 s 截断）。测试集非 5 s 样本 40 条，时长 5.3–54.3 s，其中 38 条采样率是普通 44.1/48 kHz。
- [不变] 1283 文件 = 三 CSV 并集、head 16 维、pos_emb 1214=12×101+2（我用 config 复算 `(128-16)//10+1=12`、`(1024-16)//10+1=101`，与 ASSET_MAP 一致）。

## 3. 检索方向与来源覆盖
- transformers main 分支 AST 源码三件套（modeling / feature_extraction / configuration）+ `utils/import_utils.py` + v5.0.0 与 v5.10–5.14 release notes（GitHub Releases API）。
- 官方 AST 仓库 `YuanGongND/ast` 的 pos-embed 处理源码。
- 昆虫生物声学频谱文献（T. viridissima song spectrum）。
- 未使用任何"同名题 solution/排行榜代码"，未检索任何标签或提交文件。

## 4. 关键 Findings
- **R2-F001 [EXTERNAL_EVIDENCE]** AST 分类头是 `ASTMLPHead`：`layernorm(768) + dense: Linear(768, num_labels)`，logits 来自 `pooler_output`。因此扩类只需替换 `classifier.dense`，`classifier.layernorm` 可完整保留。源：transformers main `modeling_audio_spectrogram_transformer.py` L309-318。相关性：决定"保留旧 16 行"的最小改动面。边界：main 分支 = 5.14.x，checkpoint 写 5.10.2，头结构自 4.x 起未变，风险低。
- **R2-F002 [INFERENCE + EXTERNAL_EVIDENCE]** `from_pretrained(num_labels=29, ignore_mismatched_sizes=True)` 会把 `classifier.dense.weight/bias` 视为 mismatched → 用 `_init_weights` 重新随机初始化 **整个** 29×768，旧 16 类知识（本 checkpoint 唯一被训练过的部分，output_dir=`./ast-freeze-enc`）**全部丢失**，且只打印 warning 不报错。这是本题最容易踩的静默陷阱。缓解见 MC-1。
- **R2-F003 [EXTERNAL_EVIDENCE]** HF `ASTEmbeddings` 的 `position_embeddings = nn.Parameter(zeros(1, num_patches+2, 768))`，`num_patches` 由 `config.max_length`/`num_mel_bins` 决定，**forward 里只有 `embeddings + self.position_embeddings`，没有任何插值/切片**；且 `ASTPreTrainedModel._init_weights` 对 `ASTEmbeddings` 执行 `init.zeros_(position_embeddings)`（含 cls_token、distillation_token）。→ 改 `max_length` 且 `ignore_mismatched_sizes=True` 会得到全零位置编码 + 全零 CLS/distill token 的"半毁"模型，loss 仍能下降，容易误判为正常。源：同文件 L64-99, L246-252。
- **R2-F004 [LOCAL_FACT]** Python 标准库 `wave.open` 在本题昆虫文件上直接抛 `wave.Error: unknown format: 3`（我在 `archive/fine_tune.csv` 遍历时实测）。→ 任何用 `wave`/朴素 header 假设的 dataloader 会在训练集上崩。
- **R2-F005 [EXTERNAL_EVIDENCE]** `ASTFeatureExtractor.__call__` 若传入 `sampling_rate != 16000` 直接 **raise ValueError**（不会自动重采样）；不传则只 warning。特征路径：`is_speech_available()`（=torchaudio 是否安装）为真时用 `torchaudio.compliance.kaldi.fbank(window_type="hanning", num_mel_bins=128)`，否则走 numpy `spectrogram()` 复现路径（frame 400/hop 160/fft 512/preemphasis 0.97/mel_scale kaldi）。**两条路径数值不完全等价**。归一化是 `(x - mean) / (std * 2)`（注意 ×2）；pad 在归一化**之前**，所以零 pad 区归一化后是常数 `+0.4670`。源：`feature_extraction_audio_spectrogram_transformer.py`；`import_utils.py` L1541-1543。相关性：训练与推理必须同一后端；自写 mel 必须复刻 ×2 与 pad 顺序，否则与 checkpoint 的输入分布错位。
- **R2-F006 [INFERENCE]** 5.00 s 音频 → kaldi fbank ≈498 帧，被零 pad 到 1024 帧，即 **约 51% 的输入是常数 pad**。给定 checkpoint 是在这种 5 s+pad 分布上冻结 encoder 训头得到的，推理端对长录音若填满 1024 帧真实内容，分布与旧类训练时不同。→ 一个可测选项：对长录音也只取 5 s 窗（保持 pad 结构一致），与"取满 10.24 s"做 A/B。
- **R2-F007 [EXTERNAL_EVIDENCE]** T. viridissima 鸣声为宽带谱：基频带中心 ≈10 kHz，另有 20–60 kHz 超声带（与其定向听觉最优频率 ~20 kHz 一致）。→ 16 kHz 重采样（Nyquist 8 kHz，AST mel 上限 8000 Hz）会**完全丢掉基频带**；剩余可用信息主要是脉冲串的时间包络与 <8 kHz 的低频泄漏。源：文献摘要（Bailey/Römer 等 bushcricket 高频传播研究）。
- **R2-F008 [LOCAL_FACT + INFERENCE]** 但影响面很小：测试集只有 2 条采样率 >100 kHz（312500、192000）；T. viridissima 在 44.1/48 kHz 常规录音下仍有 41 条训练样本，且这些录音的 <8 kHz 内容与其他三种昆虫可分性未被破坏。→ 不要为超声投入主线预算。
- **R2-F009 [LOCAL_FACT]** 测试集时长：323 条恰 5.00 s，40 条 5.3–54.3 s。训练侧只有昆虫类是变长。→ "时长>5.3 s ⇒ 类 25–28" 是覆盖 11% 测试集的强先验，但它是元数据捷径（是否使用、如何验证属 R3 轴）。工程侧要求：变长样本的推理路径必须存在且正确。
- **R2-F010 [EXTERNAL_EVIDENCE]** transformers v5 破坏性变更中与本题相关的：`from_pretrained` 默认 `dtype="auto"`（按保存 dtype 加载，本 checkpoint config 写 `"dtype": "float32"`，故仍是 fp32）；`Trainer(tokenizer=...)` → `processing_class=...`；`TrainingArguments` 的已弃用参数（如 `evaluation_strategy`）被直接移除；`XXXFeatureExtractor` 仅在**视觉**模型上被移除，`ASTFeatureExtractor` 保留。源：v5.0.0 release notes。相关性：kernel 内若用 Trainer，4.x 写法会 TypeError。
- **R2-F011 [EXTERNAL_EVIDENCE]** 官方 `YuanGongND/ast` 在改变输入长度时的正确做法是：取 pos_embed[:,2:,:] reshape 成 (1,768,12,101)，时间维 **从中间中心切片**（t_dim<101）或 `F.interpolate(mode='bilinear')`（t_dim>101），再与前 2 个 token 的 pos_embed 拼回。源：`src/models/ast_models.py` L93-107, L141-154。→ 若下游确实想用 512 帧窗（省 ~2× 计算），这是唯一有官方先例的实现形态。
- **R2-F012 [LOCAL_FACT]** starter 的离线安装契约：`ioai_env-*.whl` 用 `Path.glob` 逐层深度搜索（depth≤7）；Kaggle 会把 `torch-2.13.0+cu126` 文件名里的 `+` 吃掉导致 pip/uv 认不出，starter 用 symlink 重命名 + `UV_SKIP_WHEEL_FILENAME_CHECK=1` 绕过；uv 失败回退 pip（40 s vs ~6 min）。**必须放在所有 import 之前**。→ 不要改这段；把它整段保留在提交 `.py` 顶部。
- **R2-F013 [INFERENCE]** 计算预算：920 条训练样本、输入 1024×128、AST-base 12 层，单 GPU（T4/P100 级）batch 8–16 全量微调约 1–2 min/epoch；主要瓶颈是解码 + 重采样 384 kHz×120 s 文件与 kaldi fbank（CPU）。→ **一次性预计算 fbank 缓存到 `/kaggle/working/*.npy`（920+363 条 ×1024×128×4B ≈ 6.7 GB fp32，用 fp16 ≈3.4 GB）**，或每类只缓存固定若干裁剪窗以压体积。

## 5. Method Cards

### MC-1 分类头保序扩展（Head surgery / class-incremental 工程形态）
- 家族：参数手术 / 迁移学习。
- 机制：加载 16 类模型后替换 `model.classifier.dense`：`new = nn.Linear(768, 29); new.weight.data[:16] = old.weight.data; new.bias.data[:16] = old.bias.data; new.weight.data[16:].normal_(0, 0.01); new.bias.data[16:].zero_()`（或用旧行的均值/尺度匹配初始化，避免新类 logit 尺度天生偏小）。同步 `model.config.num_labels=29`、`id2label/label2id`、`model.num_labels=29`。
- 解决的问题：唯一可用 backbone 的旧类判别能力不被清零（R2-F002）。
- 证据：F001、F002（源码结构 + `_init_weights` 行为）。
- 适配：训练时旧类 296 条与新类 624 条联合；submission 直接 argmax 29 维。
- 结构差异：与"只训新头 + 旧头拼接（两个 softmax）"不同——单头联合 softmax 让新旧类 logit 可比，但需要旧类样本参与以校准尺度（细节属 R4）。
- 最小验证：手术后**不训练**直接在 296 条旧类上评 top-1（限制在前 16 列）；若显著低于手术前，说明手术写错。
- 成本：秒级。依赖：torch。
- Kernel 可行性：**可行**（纯本地参数操作）。
- 失败模式：忘记同步 `config.num_labels` → 保存/重载后头被重建为 16；用 `ignore_mismatched_sizes` 让 HF 自己扩 → 静默清零。
- 置信度：高（源码直读）。污染风险：无。

### MC-2 固定 1024×128 输入契约（或按官方 ast 切 pos-embed）
- 家族：架构/输入契约。
- 机制：保持 `config.max_length=1024`、`num_mel_bins=128`；如需 512 帧，按 F011 中心切片 pos_embed 后手工写回 `model.audio_spectrogram_transformer.embeddings.position_embeddings`，并同时改 config（否则 shape assert）。
- 解决：避免位置编码被 zeros_ 摧毁（F003）。
- 证据：F003、F011。
- 最小验证：对同一 20 条样本比较 (a) 1024 帧原生、(b) 512 帧 + 中心切片、(c) 512 帧 + ignore_mismatched_sizes 三者旧类 top-1；(c) 应崩到随机水平，可作为陷阱确证实验。
- 成本：分钟级。Kernel 可行性：**可行**。
- 失败模式：切片后忘记拼回前 2 个 token 的 pos_embed；把频率维也切错。
- 置信度：高。

### MC-3 统一音频加载/重采样管线
- 家族：数据工程。
- 机制：`soundfile.read(path, dtype='float32', always_2d=True)` → 声道均值转 mono → 若 sr≠16000 用 `torchaudio.functional.resample`（或 `scipy.signal.resample_poly` 作 fallback）→ 交给 `ASTFeatureExtractor(..., sampling_rate=16000)`。对 sr≥96 kHz 先 `resample_poly` 分两级降采样（如 384k→48k→16k）以控成本。
- 解决：F004（`wave` 崩溃）、F005（FE 不自动重采样、振幅尺度）。
- 证据：F004、F005；FE 源码。
- 适配：训练与推理**同一函数**；缓存 fbank（F013）。
- 最小验证：对每种出现过的 (fmt tag, bits, sr) 组合各取 1 个文件，打印 mono float 的 max|x|、时长、重采样后长度；确认 max|x| 都 ≤1.0 且时长不变。
- 成本：全量 1283 文件解码+重采样+fbank 预计 3–10 min CPU（长超声文件占大头）。依赖：soundfile / torchaudio / scipy（需确认 wheel 环境内存在；torchaudio 缺失会让 FE 走 numpy 路径，见失败模式）。
- Kernel 可行性：**可行**（离线，无外部数据）。
- 失败模式：wheel 环境无 torchaudio → FE 切 numpy 路径，特征与 checkpoint 训练时不一致（若原训练用 torchaudio）→ 旧类精度掉；应在 kernel 里 `print(transformers.utils.is_speech_available())` 并把两条路径的旧类 top-1 都测一遍。24-bit/float 文件用 `scipy.io.wavfile` 读会拿到 int32/float 混杂 dtype。
- 置信度：高。污染风险：无。

### MC-4 训练随机裁剪 + 推理多窗 mean-logit 聚合
- 家族：变长输入处理 / TTA。
- 机制：训练时对每条录音随机取 10.24 s（不足则整段+pad）作为一个 epoch 样本（天然增强，长录音每 epoch 见不同片段）；推理对 >10.24 s 样本取 hop=5 s 的滑窗（上限如 8 窗），**对 logits 取算术平均**后 argmax。
- 解决：F006、F009（长录音训练/推理一致性；测试 40 条变长样本）。
- 证据：F006（pad 占比）、F009（测试时长分布）；聚合方式的通用经验（AudioSet/ESC 类任务惯例）为 [INFERENCE]：mean-logit 更稳，max 对单窗噪声敏感；但若目标声件稀疏（长录音里只有几声鸣叫），max 或 top-k mean 更优——本题昆虫鸣声通常持续，故先用 mean。
- 适配：只影响 40/363 测试样本 + 240 条训练录音。
- 最小验证：对 40 条变长测试样本比较 mean vs max vs 首窗的预测差异条数；差异条数少 ⇒ 不必花提交额度做 A/B。
- 成本：推理增加 <100 次前向，秒级。
- Kernel 可行性：**可行**。
- 失败模式：随机裁剪把无声段裁进来（长录音静音多）→ 可加"选能量最高窗"作为替代；同一录音的窗跨折造成验证虚高（属 R3）。
- 置信度：中高。

### MC-5 超声"时间扩展"（bat-detector 式频移）——低优先
- 家族：信号处理特征工程。
- 机制：对 sr>100 kHz 文件，不做抗混叠降采样，而是把采样率**声明**为 16 kHz（等价 1/12–1/24 慢放），把 20–60 kHz 超声带搬到 1.7–5 kHz，再切窗喂 AST；或双分支（常规重采样 + 时间扩展）取 logit 平均。
- 解决：F007（16 kHz 后 T. viridissima 基频丢失）。
- 证据：F007（频谱学文献）；F008（影响面 2 条测试样本）。
- 适配：本题上限收益 ≈2/363 = 0.006 accuracy，且训练侧只有 17 条同类样本可学这种"被拉伸"的表征 → 极易过拟合。
- 最小验证：把这 17 条超声训练样本按两种处理分别提特征，看它们在 backbone 特征空间里更靠近哪一簇（其余 43 条 T. viridissima）。
- 成本：分钟级。Kernel 可行性：**可行**但收益微小。
- 失败模式：时间拉伸后脉冲率（重要判别线索）被同比改变，可能反而落到其他类的分布里。
- 置信度：低（机制可靠，本题收益/风险比差）。**不建议进主线**。

### MC-6 fbank 预计算缓存 + fp16 存储
- 家族：算力工程。
- 机制：一次性把每条录音的所有需要窗口的 128×1024 fbank 算好存 fp16（内存或 `/kaggle/working`），训练循环只做 tensor 索引 + SpecAugment；避免每 epoch 重复解码 384 kHz 长文件。
- 解决：F013 的 CPU 瓶颈；保证 20+ epoch 在 kernel 时限内。
- 最小验证：计时 50 条昆虫长文件的 "解码+重采样+fbank" 总耗时，外推全量。
- 成本：磁盘 ~1–3.5 GB（fp16，视窗口数）。Kernel 可行性：**可行**（注意 `/kaggle/working` 20 GB 上限与输出目录只应留 submission.csv）。
- 失败模式：随机裁剪与缓存冲突（缓存固定窗口 ⇒ 失去裁剪增强）→ 折中：每条录音缓存 K=3~5 个固定窗，训练时随机选一个。
- 置信度：中高。

## 6. 负面结果、失败条件与冲突证据
- **`ignore_mismatched_sizes=True` 不是"扩类的正确姿势"**：它只是让加载不报错，被判 mismatch 的张量一律重新初始化。对 AST，这同时威胁 `classifier.dense`（丢旧类）与（若改 max_length）`position_embeddings`/`cls_token`/`distillation_token`（被 zeros_）。这是本报告最重要的负面结论。（F002/F003）
- **FeatureExtractor 不会替你重采样**：sampling_rate 传错直接 ValueError，不传则静默用错采样率 → 全部特征错。（F005）
- **numpy 回退路径 ≠ torchaudio kaldi 路径**：同一 wav 两个后端的 fbank 数值不同（不同 preemphasis/remove_dc_offset/energy_floor 细节），因此"本地能跑通"不代表与 checkpoint 训练分布一致。（F005）
- **超声不是主线**：与"384 kHz 会毁掉一个类"的直觉相反，测试集只有 2 条相关样本，且该物种大多数录音是常规采样率。（F007+F008 组合，冲突已解决）
- **`transformers_version: 5.10.2` vs 我读的 main（5.14.x）**：AST 相关代码近年无变更迹象，但我未逐版 diff。若 kernel 内 API 与本报告描述不符，以 kernel 内 `inspect.getsource(...)` 为准。[UNRESOLVED]
- **ASSET_MAP 的 "全部 16 位 PCM" 与实测冲突**（见 §2），以本地 fmt chunk 解析为准。

## 7. 未解决问题
1. wheel 环境是否包含 torchaudio / soundfile / librosa / scipy？（决定 MC-3 的实现分支与 FE 走哪条特征路径）→ kernel 内 `pip list` 或 `importlib.util.find_spec` 首先确认。
2. 给定 checkpoint 当初的 fbank 后端与是否 do_normalize（config 说 true）→ 只能用"旧类 top-1 探针"反推。
3. kernel 时限与 GPU 型号未知 → 预算模型（F013）需按实测调整。
4. 长录音训练是否该保持 5 s+pad 结构（F006）尚无证据，只能 A/B。
5. AST main 与 5.10.2 的逐版差异未核。

## 8. 给题目分析 Agent 的合并建议
- 把 §2 的三条 LOCAL_FACT 修正合并进 ASSET_MAP（wav 格式多样性、超声只属 T. viridissima、120 s 截断）。
- 在 TASK_ANALYSIS 的"关键工程点"里把"扩展分类头"改写为 **显式禁止依赖 `ignore_mismatched_sizes`**，并加"禁止改 max_length/num_mel_bins"的硬约束。
- 把"384 kHz 信息损失"从主要风险降级为边缘风险（2 条测试样本），把节省的注意力转给旧类遗忘与 16/29 logit 尺度校准（R4 轴）。
- 在提交脚本模板中固定：starter 安装块置顶不动 + `soundfile+torchaudio.resample` 统一加载函数 + fbank 缓存 + 多窗 mean-logit。

## 9. 来源表
| 标题 | URL | 日期/版本 | 类型 | 支持 Finding |
|---|---|---|---|---|
| transformers `modeling_audio_spectrogram_transformer.py`（main） | https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/audio_spectrogram_transformer/modeling_audio_spectrogram_transformer.py | main @2026-08-04 | 官方源码 | F001,F002,F003 |
| transformers `feature_extraction_audio_spectrogram_transformer.py`（main） | https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/audio_spectrogram_transformer/feature_extraction_audio_spectrogram_transformer.py | main @2026-08-04 | 官方源码 | F005,F006 |
| transformers `configuration_audio_spectrogram_transformer.py`（main） | https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/audio_spectrogram_transformer/configuration_audio_spectrogram_transformer.py | main | 官方源码 | F003 |
| transformers `utils/import_utils.py`（`is_speech_available`） | https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/utils/import_utils.py | main | 官方源码 | F005,MC-3 |
| transformers v5.0.0 release notes（breaking changes） | https://github.com/huggingface/transformers/releases/tag/v5.0.0 | v5.0.0 | 官方 release notes | F010 |
| transformers releases v5.10.3–v5.14.1（版本时间线） | https://github.com/huggingface/transformers/releases | 2026-06~07 | 官方 release notes | F010, §6 版本冲突 |
| 官方 AST 仓库 `ast_models.py`（pos-embed 中心切片/插值） | https://raw.githubusercontent.com/YuanGongND/ast/master/src/models/ast_models.py | master | 官方论文代码 | F011,MC-2 |
| Bushcricket 高频/超声声传播与 T. viridissima 歌声频谱（10 kHz 基频 + 20–60 kHz 带） | https://www.researchgate.net/publication/226456812_High-frequency_sound_transmission_in_natural_habitats_Implications_for_the_evolution_of_insect_acoustic_communication | 1990s 论文 | 研究论文（经摘要检索获取） | F007 |
| Xeno-canto T. viridissima calling song（录音元数据示例：高频录音、>20 kHz 被 mp3 截掉） | https://xeno-canto.org/1155585 | 录音条目 | 数据库条目（仅作频段旁证，未使用其数据） | F007 |
| 本题原始资产 `archive/{train,fine_tune,submission}.csv` + `archive/audio/*.wav` fmt chunk 实测 | 本地：`.../SEARCH_BUNDLE/ORIGINAL_ASSETS/archive/` | 本次运行 | 官方题目资产 | F004,F008,F009，§2 |
| 本题 `ioai-starter.py`（离线 wheel 安装契约） | 本地：`.../ORIGINAL_ASSETS/ioai-starter.py` | 本次运行 | 官方题目资产 | F012 |
| 本题 `archive/model/{config,preprocessor_config}.json` | 本地：`.../ORIGINAL_ASSETS/archive/model/` | 本次运行 | 官方题目资产 | F003,F005,F013 |
