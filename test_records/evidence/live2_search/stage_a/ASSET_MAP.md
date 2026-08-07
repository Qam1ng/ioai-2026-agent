# ASSET_MAP.md

原始资产根目录（只读，未做任何修改/重命名/转换）：
`SEARCH_BUNDLE/ORIGINAL_ASSETS/`

## 资产总览（共 1291 个文件）

| 相对路径 / 模式 | 类型 | 数量 | 大小 | 用途 |
|---|---|---|---|---|
| `ioai-starter.py` | Python 脚本 | 1 | ~7KB | 官方 starter：离线安装 pinned wheels（`ioai-2026-wheel-dataset` + uv），`find_input()` 辅助函数，示例写出 `/kaggle/working/submission.csv` |
| `archive/train.csv` | CSV | 1 | 296 行数据 | 旧 16 类（target 0–15）的带标签训练数据 |
| `archive/fine_tune.csv` | CSV | 1 | 624 行数据 | 新 13 类（target 16–28）的带标签微调数据 |
| `archive/submission.csv` | CSV | 1 | 363 行数据 | 提交模板 / 测试集清单，列 `path,target`，target 全为占位 0 |
| `archive/audio/*.wav` | WAV 音频 | 1283 | 各不等 | 全部音频（train 296 + fine_tune 624 + test 363 = 1283，与磁盘完全一一对应，无缺失无多余） |
| `archive/model/config.json` | JSON | 1 | 1.3KB | `ASTForAudioClassification` 配置，16 类头（id2label=Dog…Rain），num_mel_bins=128, max_length=1024, transformers 5.10.2 |
| `archive/model/model.safetensors` | safetensors | 1 | 345MB | AST 权重，203 个 tensor，F32；`classifier.dense.weight [16,768]`；position_embeddings [1,1214,768]（仅读 header，未反序列化执行） |
| `archive/model/preprocessor_config.json` | JSON | 1 | 297B | `ASTFeatureExtractor`：sampling_rate=16000, num_mel_bins=128, max_length=1024, mean=-4.2677, std=4.5690 |
| `archive/model/training_args.bin` | pickle(zip) | 1 | 5.2KB | HF TrainingArguments pickle。**未 unpickle**（不安全反序列化），仅提取可打印字符串：output_dir=`./ast-freeze-enc`，adamw_torch_fused，linear scheduler → 提示该 checkpoint 是冻结 encoder 微调产物。[部分UNRESOLVED：具体超参数值未解出] |

## CSV schema 与关系

- `train.csv` / `fine_tune.csv`：列 `path,split,target,category`；`split` 全部为 `train`（没有官方 val 划分）；`path` 形如 `audio/<16hex>.wav`，相对于 `archive/`。
- `submission.csv`：列 `path,target`；363 个唯一 path；即测试集文件清单兼提交格式（预测单位=每个 wav 文件一个整数类别）。
- 三个 CSV 的 path 集合两两不相交，其并集 = 磁盘上 1283 个 wav（已验证）。
- 标签映射：train.csv 的 target 0–15 与 `model/config.json` 的 id2label 完全一致；fine_tune.csv 的 target 16–28 是模型头中不存在的新类。

## 类别分布（[LOCAL_FACT]，逐行统计全部 CSV）

train.csv（296 行，16 类，严重不均衡）：
Thunderstorm 62, Frog 35, Bird Chirping 35, Wolf Howl 23, Cow 22, Mouse Click 22, Dog/Rooster/Pig/Cat/Hen/Crow/Sea Waves 各 12, Rain 7, Sheep 3, Keyboard Typing 3。

fine_tune.csv（624 行，13 类，较均衡）：
16 Crackling Fire 24；17 Axe / 18 Chainsaw / 19 Generator / 20 Hand Saw / 21 Vehicle Engine / 22 Helicopter / 23 Gunshot / 24 Firework 各 45；25 Pterophylla Camellifolia / 26 Cicada Orni / 27 Gryllus Campestris / 28 Tettigonia Viridissima 各 60。

## 音频统计（逐文件解析全部 1283 个 wav 头，时长由文件大小/字节率估算）

- 采样率极其混杂：44.1k(主体)、48k、24k、96k、16k、8k、11k、22.05k、192k、256k、384k 等；声道 1–2；主要 PCM16，少量 float32(fmt=3) 和 WAVE_FORMAT_EXTENSIBLE(fmt=65534, 含 24bit)。**标准 `wave` 模块解析部分文件报错（fmt=3），需用 soundfile/torchaudio/librosa 读取并统一重采样到 16kHz 单声道**。
- 时长：train 全部 ≈5.0s；fine_tune 中 16–24 类全部 ≈5.0s，25–28（昆虫类）3.0–120s 变长；test 大多 ≈5s，最长 ≈54.3s（说明测试集含昆虫类长音频）。
- AST max_length=1024 帧 ≈10.24s@16kHz → 长音频必须裁剪/分块聚合。

## [UNRESOLVED]

1. **题面/规则/metric 文件缺失**：bundle 中没有 README、描述页或 metric 说明；任务目标、评分 metric（accuracy? macro-F1?）只能从资产推断。
2. `training_args.bin` 未安全解出全部超参（仅字符串证据）。
3. `submission.csv` 中 test 是否覆盖全部 29 类未知（时长分布暗示含昆虫类，未证实所有类都出现）。
4. AST 底模来源（推测为 MIT/ast-finetuned-audioset，position_embeddings shape 1214 与 10-10 AudioSet 版一致）未在本地证实。

声明：本阶段所有读取均为只读操作，ORIGINAL_ASSETS 内没有任何文件被修改、重命名、转换或覆盖。
