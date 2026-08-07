# ASSET_MAP.md — IOAI 2026 Practice Task 1（音频分类 / AST 增类微调）

原始资产根目录（只读，未做任何修改）：
`SEARCH_BUNDLE/ORIGINAL_ASSETS/`（共 1291 个文件，SHA256 由阶段 B `SHA256SUMS` 与 `ASSET_MANIFEST.json` 保证；本 Agent 未修改、重命名或转换任何文件）

## 目录结构与文件清单

| 相对路径 | 类型 | 大小/数量 | 用途 |
|---|---|---|---|
| `ioai-starter.py` | Python 脚本 | ~6KB | 官方 starter：离线 wheel 环境安装 + 占位 submission 写出逻辑（已完整阅读） |
| `archive/train.csv` | CSV | 296 数据行 | 旧类（target 0–15）训练标签表 |
| `archive/fine_tune.csv` | CSV | 624 数据行 | 新类（target 16–28）微调标签表 |
| `archive/submission.csv` | CSV | 363 数据行 | 测试集清单 + 提交格式模板（target 全为占位 0） |
| `archive/audio/*.wav` | WAV 音频 | 1283 个（16 位 PCM，部分 WAVE_FORMAT_EXTENSIBLE） | 全部训练/微调/测试音频，16 进制随机文件名 |
| `archive/model/config.json` | JSON | 1.3KB | ASTForAudioClassification 配置，16 类 id2label（已完整阅读） |
| `archive/model/model.safetensors` | safetensors | 344.8MB | 已微调的 AST checkpoint，204 个 F32 tensor（仅安全解析 header） |
| `archive/model/preprocessor_config.json` | JSON | 297B | ASTFeatureExtractor 配置：16kHz、128 mel、max_length 1024、mean=-4.2677、std=4.5690（已完整阅读） |
| `archive/model/training_args.bin` | pickle 二进制 | 5.2KB | HF TrainingArguments；**未反序列化**（pickle 不安全），仅用 `strings` 检查 |

## 关键 schema

- `train.csv` / `fine_tune.csv` 列：`path,split,target,category`；`split` 全部为 `train`（无 val 分区）；`path` 形如 `audio/<16hex>.wav`。
- `submission.csv` 列：`path,target`；363 行，行序即模板顺序。
- 三个 CSV 的 `path` 两两无交集，合计恰好 1283 条 = `archive/audio/` 全部文件（无缺失、无多余）。[LOCAL_FACT]

## 类别分布

- `train.csv`（旧 16 类，296 条，严重不均衡 3–62/类）：0 Dog(12) 1 Rooster(12) 2 Pig(12) 3 Cow(22) 4 Frog(35) 5 Cat(12) 6 Hen(12) 7 Sheep(3) 8 Crow(12) 9 Mouse Click(22) 10 Keyboard Typing(3) 11 Thunderstorm(62) 12 Sea Waves(12) 13 Bird Chirping(35) 14 Wolf Howl(23) 15 Rain(7)。
- `fine_tune.csv`（新 13 类，624 条）：16 Crackling Fire(24) 17 Axe(45) 18 Chainsaw(45) 19 Generator(45) 20 Hand Saw(45) 21 Vehicle Engine(45) 22 Helicopter(45) 23 Gunshot(45) 24 Firework(45) 25 Pterophylla Camellifolia(60) 26 Cicada Orni(60) 27 Gryllus Campestris(60) 28 Tettigonia Viridissima(60)。

## 音频统计（全量 fmt/data chunk 解析 + 抽样验证）

- `train.csv`：全部恰好 5.00s；采样率 44100(241)/48000(44)/96000(9)/44000(1)/22050(1)；单/双声道混合。
- `fine_tune.csv` 非昆虫类（16–24，384 条）：全部恰好 5.00s；采样率 44100/48000/24000/96000/16000。
- `fine_tune.csv` 昆虫类（25–28，240 条）：**变长 1.6–120s**，采样率极端多样（8000 至 384000Hz，含超声采集 192k/200k/250k/256k/384k）。
- `submission.csv` 测试集：323/363 恰好 5.00s，40 条非 5s（最长 54.3s）；采样率含 44100(242)/48000(80)/24000(26)/96000(11) 及昆虫特征的 312500/192000/11025/8000 各 1。→ 测试集同时包含定长（旧类或新环境类）与变长超声（昆虫类）样本。[LOCAL_FACT]

## 模型 checkpoint 关键信息（safetensors header）

- 架构 `ASTForAudioClassification`（audio-spectrogram-transformer），hidden 768、12 层 12 头、patch 16、fstride/tstride 10、num_mel_bins 128、max_length 1024（≈10.24s 输入窗）、position_embeddings [1,1214,768]（AudioSet 尺寸）。
- `classifier.dense.weight [16,768]`：分类头只有 16 类 = train.csv 的旧类。
- `training_args.bin` strings 显示 `output_dir=./ast-freeze-enc`、optim adamw_torch_fused、linear scheduler、report_to wandb → 推断该 checkpoint 是**冻结 encoder、只训分类头**得到的 16 类模型。[INFERENCE]
- `transformers_version: 5.10.2`（较新版本，注意计分 kernel 固定依赖版本兼容）。

## [UNRESOLVED]

- 无任何题面/README/规则文本文件：官方 metric、测试集类别范围（仅新类 or 全部 29 类）、是否允许外部数据/预训练模型，均无本地文字依据。
- `training_args.bin` 的具体超参数值未解析（pickle 安全约束）。
- starter 尾部写的占位 header 为 `id,prediction`，与 `archive/submission.csv` 的 `path,target` 冲突（见 TASK_ANALYSIS）。
