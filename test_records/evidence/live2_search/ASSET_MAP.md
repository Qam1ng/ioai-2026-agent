# ASSET_MAP.md（最终版）

原始资产：`SEARCH_BUNDLE/ORIGINAL_ASSETS/`（runner 提供的只读副本/挂载，未被任何 Agent 修改）。

## 完整性校验状态（阶段 B 亲自抽查）

- `ORIGINAL_ASSETS` 实际文件数 = **1291**；`SHA256SUMS` 行数 = **1291**（1283 行 `archive/audio/*.wav` + 8 行其他）→ 数量一致。
- 关键文件全部在场并在清单内：`ioai-starter.py`、`archive/{train,fine_tune,submission}.csv`、`archive/model/{config.json,preprocessor_config.json,model.safetensors,training_args.bin}`。
- 抽样 `sha256sum -c` 7 个条目（4 个随机 wav + `ioai-starter.py` + `archive/train.csv` + `archive/model/config.json`）→ **全部 OK**，无失败项。
- 另有 runner 产出的 `ASSET_MANIFEST.json`（244KB，逐文件清单）。
- 结论：**校验通过**，未发现缺失或哈希不一致。

## 资产总览（1291 个文件）

| 相对路径 / 模式 | 类型 | 数量 | 用途 |
|---|---|---|---|
| `ioai-starter.py` | Python | 1 | 官方 starter：`setup_ioai_env()` 从挂载的 `ioai-2026-wheel-dataset` 离线安装 pinned wheels（先 pip 装 uv，再 `uv pip install --system --no-index ioai-env`，失败回退 pip）；`find_input(name)` 用 rglob 定位竞赛文件（禁止硬编码路径）；示例把 `/kaggle/working/submission.csv` 写成 **`id,prediction`** 两列（见下方冲突） |
| `archive/train.csv` | CSV | 1 | 旧 16 类（target 0–15）带标签数据，296 行 |
| `archive/fine_tune.csv` | CSV | 1 | 新 13 类（target 16–28）带标签数据，624 行 |
| `archive/submission.csv` | CSV | 1 | 提交模板兼 test 清单，363 行，列 **`path,target`**，target 全 0 |
| `archive/audio/*.wav` | WAV | 1283 | 全部音频；296+624+363=1283，与三 CSV 并集精确一一对应（无缺失、无多余） |
| `archive/model/config.json` | JSON | 1 | `ASTForAudioClassification`，16 类头 id2label(Dog…Rain)，hidden 768/12 层/12 头，num_mel_bins=128，max_length=1024，fstride=tstride=10，transformers_version 5.10.2 |
| `archive/model/model.safetensors` | safetensors | 1 (345MB) | AST 权重，203 tensor，F32；`classifier.dense.weight [16,768]`、`classifier.dense.bias [16]`、`embeddings.position_embeddings [1,1214,768]`、patch proj [768,1,16,16]、含 `distillation_token`（DeiT 血统）。仅读 header，未反序列化 |
| `archive/model/preprocessor_config.json` | JSON | 1 | `ASTFeatureExtractor`：sampling_rate=16000, num_mel_bins=128, max_length=1024, mean=-4.2677393, std=4.5689974, do_normalize=true, return_attention_mask=false |
| `archive/model/training_args.bin` | zip/pickle | 1 | HF TrainingArguments。**未 unpickle**（避免任意代码执行），仅提字符串：`output_dir=./ast-freeze-enc`、`adamw_torch_fused`、linear scheduler。[部分 UNRESOLVED：具体数值未解出] |

## Schema / ID / 顺序 / 资产关系

- `train.csv` / `fine_tune.csv`：`path,split,target,category`；`split` 全为 `train`（**无官方 val**）；`path` = `audio/<16hex>.wav`（相对 `archive/`）。
- `submission.csv`：`path,target`，363 个唯一 path；**预测单位 = 每个 wav 文件一个整数类别**；下游应按模板 path 顺序原样填 target。
- 三 CSV 的 path 集合两两不相交；并集 = 磁盘 1283 wav（已验证）。
- 标签空间：`train.csv` target 0–15 与 `model/config.json` id2label **完全一致**；`fine_tune.csv` target 16–28 为模型头中不存在的新类 → 总类空间 29。
- 文件名是 16 位 hex 哈希，**源数据集的 `src_file`/`take`/物种/fold 信息被抹除**（这是分组验证无法直接复现的根因，见 TASK_ANALYSIS 泄漏节）。

## 读取与抽样覆盖范围（阶段 A + 阶段 B + 两份 Research 报告，均在本机复算）

- 三个 CSV：**逐行**读取并统计（类分布、split、path 交集）。
- 1283 个 wav：**逐文件**解析头（阶段 A 手工 RIFF 解析；阶段 B 用 `soundfile.info` 复核 test 全部 363 个）。
- FFT 能量分带：抽样（R1 对 25–28 类抽样 + 阶段 B 复核 test 全部 2 个 >100kHz 文件与 6 个 Tettigonia 文件）。**未对全部 1283 文件做频谱统计**。
- 模型：config/preprocessor 全文读取；safetensors 仅 header；training_args 仅字符串。
- starter：全文读取。

## 关键统计（[LOCAL_FACT]，可复现）

类分布：
- train.csv（16 类, 296）：Thunderstorm 62, Frog 35, Bird Chirping 35, Wolf Howl 23, Cow 22, Mouse Click 22, Dog/Rooster/Pig/Cat/Hen/Crow/Sea Waves 各 12, Rain 7, **Sheep 3, Keyboard Typing 3**。
- fine_tune.csv（13 类, 624）：Crackling Fire 24；Axe/Chainsaw/Generator/Hand Saw/Vehicle Engine/Helicopter/Gunshot/Firework 各 45；Pterophylla Camellifolia/Cicada Orni/Gryllus Campestris/Tettigonia Viridissima 各 60。

时长（阶段 B 用 soundfile 精确复核 test）：
- train.csv：全部 = 5.0s。fine_tune.csv：16–24 类全部 5.0s；25–28（昆虫）3.0–120s。
- **test（363）：323 个 =5.0s；<5.5s 共 324；5.5–11s 9 个；≥11s 30 个；max 54.27s** → 只有约 39–40 个文件超过单窗 10.24s 需要多窗处理。

采样率（test，soundfile 精确）：44100×242, 48000×80, 24000×26, 96000×11, 11025×1, 8000×1, 312500×1, 192000×1（**无 384k**）。
train+fine_tune 侧采样率集合含 8k/16k/22.05k/24k/25.6k/32k/44k/44.1k/48k/96k/192k/200k/250k/256k/384k，声道 1–2，含 float32(fmt=3) 与 WAVE_FORMAT_EXTENSIBLE(fmt=65534, 24bit) → **Python 标准库 `wave` 在 fmt=3 上抛错；`soundfile` 可读全部 1283 个**。

采样率↔类相关（阶段 B 逐文件复核，证实 R1 声明）：Hand Saw 45/45 = 24kHz；Chainsaw 41/45 = 24kHz（24k 仅出现在这两类，共 83 个训练文件；test 有 26 个 24kHz 文件）；Wolf Howl 23/23 = 48kHz；Generator 45/45 = 44.1kHz；Tettigonia 独占 256k/384k。

## 与阶段 A 相比的修正

1. **[修正] test 类空间**：阶段 A 列为 UNRESOLVED。现有强证据（323/363 为 5.0s、非 5s 文件的采样率签名与昆虫类一致、各 (sr,ch) 签名 test/train 比例集中在 0.28–0.45 而整体 363/920=0.395）→ **test 覆盖 29 类全空间，旧 16 类 + FSC22 系新类占绝大多数，昆虫类约 40 个（≈11%）**。仍为 [INFERENCE]（未逐类证实）。
2. **[修正] 长音频影响面被量化**：阶段 A 只说"长音频必须裁剪/分块"，实际只有 ~39/363（≈11%）test 文件超过 10.24s；且真正的机制是 **`ASTFeatureExtractor` 静默截断**（超过 1024 帧的部分被无声丢弃，不报错），阶段 B 已在本地 transformers 源码中直接验证（`fbank[0:max_length, :]`）。
3. **[修正] 提交契约存在文档内冲突**：`ioai-starter.py` 示例写 `id,prediction`，而本题模板 `archive/submission.csv` 是 `path,target`。以**本题挂载的模板为准**，但下游必须在 kernel 内再确认（starter 是三题共用样板）。
4. **[新增] 8kHz 硬上限**：`ASTFeatureExtractor` 的 mel 滤波器 min 20Hz / max = sampling_rate//2 = **8000Hz**（阶段 B 本地源码验证），padding 走 ZeroPad2d 后再整体归一化 → 填充区常量 = (0−(−4.2677))/(2×4.569) = **+0.467**（注：R1 报告写作 −0.467，符号有误，机制不变）。
5. **[新增] 超声昆虫文件**：test 中 2 个高采样率文件（312500Hz、192000Hz）>8kHz 能量占比 **0.9994 / 0.9979**（阶段 B 亲自复核）→ 直接重采样到 16kHz 后几乎归零；train 侧 Tettigonia 中同样存在（抽查 96kHz 文件 hi8k=0.998），但并非全部（另有 hi8k=0.005–0.22 的样本）。
6. `training_args.bin` 中的 `./ast-freeze-enc` **只是 checkpoint 的产出目录名，不构成对我们冻结 encoder 的规则约束**（两份报告均未发现任何官方文本要求冻结）。

## 仍无法解析 / 不明确

- `archive/model/training_args.bin`：[UNRESOLVED] 未安全解出全部超参数值（仅字符串证据）。文件已完整保留。
- 题面 / metric / rules 文本：**bundle 中不存在**；R2 已验证公网不可获得（竞赛页登录墙、Kaggle API 401、搜索引擎零命中）。[UNRESOLVED]
- AST 底模确切来源（推测 MIT/ast-finetuned-audioset-10-10-0.4593，position_embeddings=1214 与之一致）未在本地证实。[INFERENCE]
- pinned wheel 内是否含 `scikit-learn`/`librosa`/`torchaudio`/`soundfile`：bundle 无 wheel 清单（starter 只装 meta-package `ioai-env`）→ kernel 内需 import 探测并准备回退。[UNRESOLVED]

声明：阶段 A 与阶段 B 的全部操作均为只读；`ORIGINAL_ASSETS` 内没有任何文件被修改、重命名、转换或覆盖。
