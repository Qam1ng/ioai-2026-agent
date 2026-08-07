# ASSET_MAP.md（最终版）— IOAI 2026 Practice Task 1：AST 音频分类 / 增类微调

原始资产：`SEARCH_BUNDLE/ORIGINAL_ASSETS/`（只读，本 Agent 未修改/重命名/转换任何文件）。
完整性校验（阶段 B 亲自核对）：`SHA256SUMS` 1291 行，`ORIGINAL_ASSETS` 实际 1291 个文件（其中 `archive/audio/*.wav` 1283 行/个），**逐一抽查 7 个关键文件哈希全部一致**（`ioai-starter.py`、`archive/{train,fine_tune,submission}.csv`、`archive/model/{config.json,model.safetensors}`、`archive/audio/008882a56984cabf.wav`）。未发现缺失或失配。

## 1. 资产清单与用途

| 相对路径 | 类型 | 规模 | 用途 | 读取覆盖 |
|---|---|---|---|---|
| `ioai-starter.py` | Python | 6.0KB | 官方 starter：离线 wheel 安装块（必须置顶、import 之前）+ 动态 `rglob` 找输入 + 占位 submission | 完整阅读（阶段 A、R2、R4 三方） |
| `archive/train.csv` | CSV | 296 行 | 旧类（target 0–15）训练标签 | 完整读，三方独立重算统计 |
| `archive/fine_tune.csv` | CSV | 624 行 | 新类（target 16–28）微调标签 | 完整读，三方独立重算 |
| `archive/submission.csv` | CSV | 363 行 | 测试集清单 + 提交模板（target 列全 0，无标签信息） | 完整读 |
| `archive/audio/*.wav` | WAV | 1283 个，共 ~1.5GB | 全部训练/微调/测试音频；文件名为 16-hex 哈希 | **全量** header（fmt/data chunk）解析：时长、采样率、声道、位深、fmt tag；抽样波形读取 |
| `archive/model/config.json` | JSON | 1.3KB | `ASTForAudioClassification` 配置，16 类 id2label | 完整阅读 |
| `archive/model/preprocessor_config.json` | JSON | 297B | `ASTFeatureExtractor`：sr=16000, num_mel_bins=128, max_length=1024, mean=-4.2677393, std=4.5689974, do_normalize=true | 完整阅读 |
| `archive/model/model.safetensors` | safetensors | 344.8MB | 已微调 16 类 AST checkpoint，204 个 F32 tensor | 仅安全解析 JSON header（未反序列化权重） |
| `archive/model/training_args.bin` | pickle | 5.2KB | HF `TrainingArguments`；**未反序列化**（pickle 不安全），仅 `strings` 检查 | 部分（键名/字符串） |

## 2. Schema、ID 与资产关系

- `train.csv`/`fine_tune.csv`：`path,split,target,category`。`split` 常量 `train`（**官方无 validation 分区**）。
- `submission.csv`：`path,target`，363 行；**行序即提交模板顺序，必须原样保留**。
- 三 CSV 的 `path` 两两无交集，并集 1283 = `archive/audio/` 全量（无缺失、无孤立文件）。
- 标签映射：0–15 由 `model/config.json` 的 `id2label` 定义；16–28 由 `fine_tune.csv` 的 `target↔category` 一一对应（无冲突）。合计 29 类。
- checkpoint 结构（safetensors header）：`ASTForAudioClassification`，hidden 768、12 层 12 头、patch 16、fstride/tstride 10、`position_embeddings [1,1214,768]`（=12×101+2，与 128mel/1024帧 复算一致）、`classifier.layernorm[768]` + `classifier.dense [16,768]`。
- `training_args.bin` strings 含 `output_dir=./ast-freeze-enc`、`adamw_torch_fused`、linear scheduler、`report_to=wandb`。

## 3. 音频统计（全量 header 解析，阶段 B 已独立复验）

| 子集 | 条数 | 时长 | 采样率 | 编码 |
|---|---|---|---|---|
| `train.csv`（旧 16 类） | 296 | **全部恰好 5.00s** | 44100(241)/48000(44)/96000(9)/44000(1)/22050(1) | **全部 fmt tag 1 / 16-bit**（168 mono + 128 stereo） |
| `fine_tune.csv` 16–24（新环境类） | 384 | **全部恰好 5.00s** | 44100/48000/24000(83)/96000(26)/16000(3) | fmt 1 / 16-bit |
| `fine_tune.csv` 25–28（四种鸣虫） | 240 | **238 条非 5.00s**（1.6–120.0s，中位 13–16s，最大恰好 120.0 = 上游截断），仅 2 条为 5.00s | 44100(156)+48000(60) 为主；≥192kHz 仅 17 条且**只属 Tettigonia Viridissima** | fmt 1/16bit 217，**fmt 3 (IEEE float32) 3，fmt 0xFFFE 24-bit 3，fmt 0xFFFE 16-bit 17** |
| `submission.csv`（测试） | 363 | 323 条恰好 5.00s；40 条 5.3–54.3s | 44100(242)/48000(80)/24000(26)/96000(11)/11025(1)/8000(1)/312500(1)/192000(1) | **全部 fmt tag 1 / 16-bit（已统一重编码）** |

关键交叉事实（阶段 B 亲自重算确认）：
- labeled 920 条中非 5.00s 文件共 **238 条，100% 属 25–28**（25:59, 26:60, 27:60, 28:59）；旧类与 16–24 无一条非 5.00s。反向 recall = 238/240 = 0.992。
- 24000Hz 仅出现在 16–24（labeled 83、测试 26）→ 可作数据源分组键，不是类别标记。
- 96000Hz 跨旧类(9)/新环境类(26)/昆虫(3) 三处 → 采样率不是昆虫检测器。

## 4. 相对阶段 A 的修正（均有本地证据）

1. **WAV 编码不是"全部 16-bit PCM"**：昆虫子集含 fmt 3（float32）×3、fmt 0xFFFE 24-bit ×3、0xFFFE 16-bit ×17。Python 标准库 `wave` 在 fmt 3 文件上直接抛 `wave.Error: unknown format: 3`（R2 实测，阶段 B 复现）。→ 必须用 `soundfile`/`torchaudio` 读取。测试集全部 fmt 1/16-bit，故"文件格式"不能当测试端捷径。
2. **"采样率>100kHz ⇒ 昆虫"从捷径降级为弱信号**：≥192kHz 只属 T. viridissima，测试集仅 2 条（收益上限 ≈0.006）。
3. **"时长≠5.00s ⇒ 昆虫"从捷径升级为一级近确定约束**：labeled precision 1.000 / recall 0.992（238 条全量核验，非抽样）。
4. **位置编码 1214 = 12×101+2 已用 config 复算**，与 AudioSet 尺寸一致；此形状与 `max_length=1024`/`num_mel_bins=128` 绑定，不可擅改（见 TASK_ANALYSIS §规则/工程硬约束）。
5. 阶段 A "checkpoint 由 frozen-encoder 训头得到" 的推断被 R4 强化为：**encoder 权重 ≈ 未被下游扭曲的原版 AudioSet AST**，旧类知识几乎全部集中在 16 维头权重 + 296 条旧类数据里。

## 5. 无法解析 / 仍不明确

- `archive/model/training_args.bin` 的具体超参值 [UNRESOLVED]（pickle 反序列化风险，仅取 strings）。保留原文件，未删除。
- 无任何题面/README/规则文本文件 → 官方 metric、测试类别范围、public/private 划分、外部数据许可、kernel 时限/GPU 型号均无本地文字依据 [UNRESOLVED]。
- `ioai-starter.py` 占位 header `id,prediction` 与 `archive/submission.csv` 的 `path,target` 冲突（三方独立确认，未静默取舍；详见 TASK_ANALYSIS §3）。
