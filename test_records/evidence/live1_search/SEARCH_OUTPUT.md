# SEARCH_OUTPUT.md — IOAI 2026 Practice Task 1（AST 音频分类 / 16→29 增类微调）

## 状态声明

- **完整性：完整交付，无降级。** 三路研究（R2/R3/R4，按阶段 A research_plan 启用）全部产出且完整，均按 Charter 工作、无截断。Merge Ledger 完备（R2 F001–F013 + MC-1–6、R3 F001–F013 + MC-1–6、R4 F001–F011 + MC-R4-01–06，每个 ID 唯一去向，无排除项）。
- **原始资产校验：通过（runner 产出，阶段 B 亲自核对）。** `SHA256SUMS` 1291 行 = `ORIGINAL_ASSETS/` 实际 1291 文件；逐一抽查 7 个关键文件哈希（starter、三个 CSV、config.json、model.safetensors 344.8MB、1 个 wav）**全部一致**。未修改任何原始文件。
- 历史教训目录本次为空（`context/prior_lessons/README.md` 明示未提供）→ 无条目可引用；唯一一等实测教训取自系统背景（音频族 LB +0.096 / 本地 5-fold CV −0.127 反号），已并入结论。
- 阶段 A 文档与三份原始 Research Report 原样保留在 `stage_a/`、`research_raw/`。

## 最重要的 5 条本地事实 [LOCAL_FACT]

1. **标签空间 29 类，checkpoint 头只有 16 维。** `archive/model/` 是 AST-base（`classifier.dense [16,768]`，`position_embeddings [1,1214,768]`，`training_args.bin` 含 `./ast-freeze-enc`）；新类 16–28 见 `fine_tune.csv`。旧类 296 条 + 新类 624 条 = 920 条全部带标签、路径无交集 → **这不是内存受限的 CIL，而是普通小数据 29 类微调**。
2. **时长门控近乎确定**：labeled 920 条中非 5.00s 文件 **238 条，100% 属昆虫类 25–28**；昆虫 240 条仅 2 条为 5.00s（recall 0.992）。测试集 363 条中 40 条非 5s（5.3–54.3s）。阶段 B 已全量独立复算。
3. **测试先验 ≠ labeled 先验**：按 labeled 分层昆虫应占 ~94 条，实测只有 40 条非 5s → 测试昆虫推测 ~11% vs labeled 26%。任何以 labeled 频率做反加权/balanced-softmax 的做法方向可能是错的。
4. **音频编码/采样率高度异质**：训练侧昆虫含 IEEE-float32(fmt 3)×3、24-bit EXTENSIBLE×3、16-bit EXTENSIBLE×17，Python `wave` 直接崩；采样率 8k–384k。**测试集 363 条全部 fmt1/16-bit**（已统一重编码）→ 编码不能当测试端捷径；≥192kHz 仅属 T. viridissima（测试仅 2 条）。
5. **提交契约冲突**：`archive/submission.csv` 表头 `path,target`（363 行，target 全 0）vs `ioai-starter.py` 占位 `id,prediction`。判断按模板 `path,target`、只替换 target 列、不改行序，但**需题面确认**。starter 的离线 wheel 安装块必须原样置顶于任何 import 之前。

## 最值得花提交额度验证的前 3 个方向

1. **常量类 LB 探针定 metric + 测试先验（2–3 次提交，不需要模型）**：全预测类 c 时，accuracy 得分严格 = n_c/363（可反解计数）；macro-F1 ≈ (1/29)·2n_c/(363+n_c)（小一个数量级）；balanced-acc 恒 0.0345。metric 是决定稀有类策略（零和）与 logit 调整方向的前置未知量。必须在还有时间用其结论时花掉（建议剩余 60 分钟前完成）。→ `RESEARCH_SYNTHESIS.md` CM-C。
2. **时长门控标签空间约束（1 次单变量 A/B）**：把 40 条非 5s 测试样本的 argmax 限制在 {25,26,27,28}。本地 precision 1.000，覆盖 11% 测试集；风险是测试端处理方式未知（最坏 −11%）。存在保留冲突（R3 主张硬约束 vs R4 主张只作 tie-break），保守路径：只对 dur>8s 或用软偏置 δ≈3。→ CM-G / 冲突 C-1。
3. **新类 logit 偏置 δ 与长尾 τ 的符号（1–2 次提交）**：扩头后新类样本 2× 旧类且新头随机初始化，文献（WA/BiC）报告新类 logit 系统性偏高；`z' = z − τ·log π − δ·1[y≥16]`，两个旋钮必须分开调。**温度缩放对硬标签 metric 完全无效，不要花时间。** → CM-I。

## 关键风险（验证 / 泄漏 / 规则）

- **静默毁模型的两个陷阱（源码级证据）**：(a) `from_pretrained(num_labels=29, ignore_mismatched_sizes=True)` 会重新随机初始化整个 `classifier.dense`，丢掉本 checkpoint 唯一被训练过的部分，只打 warning → 必须手工扩头并拷入旧 16 行；(b) 改 `max_length`/`num_mel_bins` 会让 `_init_weights` 把 `position_embeddings`/`cls_token`/`distillation_token` **`zeros_` 清零**，loss 仍会下降。
- **`ASTFeatureExtractor` 不会自动重采样**（sr≠16000 直接 raise），归一化是 `(x-mean)/(std*2)` 且 pad 先于归一化；torchaudio-kaldi 与 numpy 回退路径数值不等价 → 训练/推理必须同后端，并用"旧类 top-1 探针"确认与 checkpoint 训练分布一致。
- **本地 CV 系统性乐观**：旧类/部分新类形态与 ESC-50 一致，官方 README 明言同源片段必须同折，而本题文件名已哈希 → 随机分层 CV 有近重复泄漏。R3 实测的廉价频谱指纹**查不出**近重复（LOO 1-NN 仅 0.363），所以"看起来没重复"不能反驳。判据：LB 与本地单折 <0.02 视为噪声；只接受配对 bootstrap CI 不含 0 的本地改进；结构性改动与 LB 冲突时信 LB。
- **规则边界**：计分 kernel 无互联网、只能用挂载 wheel 与题目自带 checkpoint、单 GPU 限时；外部数据/权重默认禁止（本地无允许文字）；≤50 次提交。
- **统计分辨率**：363 条测试 accuracy 95% CI 半宽 ≈0.045–0.050；920 条 5-fold 单折 val≈184 条 → 单折数字几乎无信息。

## 冲突与未解决问题（简表，详见 RESEARCH_SYNTHESIS §7–§8）

| ID | 内容 | 解决方式 |
|---|---|---|
| C-1 | 时长门控：硬约束 vs 仅 tie-break | 1 次 LB 单变量 A/B；折中先用 dur>8s |
| C-2 | 长录音窗口：5s（保持 51% pad 结构）vs 10.24s 多窗 | 本地留出三方比较（仅推理端切换） |
| C-3 | frozen LP vs 全量微调（3–62 条/类横跨两个区间） | CM-D 与 CM-E 同一留出集，分开看稀有旧类/新类 |
| C-4 | 能否用 labeled 先验做 τ 调整 | 先用常量探针测真实先验 |
| C-5 | transformers 5.10.2 vs main(5.14.x) 未逐版 diff | kernel 内 `inspect.getsource` 为准 |
| C-6 | submission 表头 `path,target` vs `id,prediction` | Kaggle 题面确认；默认按模板 |
| U-1 | **官方 metric 未知**（最重要）| 题面优先，否则 CM-C 探针 |
| U-2 | wheel 环境依赖清单（torchaudio/soundfile/scipy/sklearn/LightGBM）与 `is_speech_available()` | kernel 内 `find_spec` 首先打印 |
| U-3 | kernel 时限 / GPU 型号；测试昆虫是否被裁到 5s；public/private 划分 | 题面 / LB 观察 |
| U-4 | `training_args.bin` 精确超参（未反序列化，pickle 风险） | 保持未解 |

## Bundle 内容与阅读顺序

1. **本文件** — 导航与高密度交接。
2. `TASK_ANALYSIS.md` — 最终题目理解（可独立阅读）：目标、输入/输出、metric 与 submission contract、数据结构、提供资产的作用与工程硬约束、规则/资源边界、验证与泄漏、四类证据分区、阶段 A 被修正项。
3. `RESEARCH_SYNTHESIS.md` — 21 条 canonical findings、12 张 canonical method cards（CM-A…CM-L，全部通过 kernel 可行性门）、metric/验证/泄漏/后处理汇总、经典与非主流备选、**14 条负面结果索引**、冲突证据、来源表、完整 Merge Ledger。
4. `ASSET_MAP.md` — 最终资产地图：文件清单、schema/ID/顺序、全量音频统计、相对阶段 A 的 5 项修正、无法解析项、校验状态。
5. `research_raw/research_R2.md`（HF/AST 实现与音频工程）、`research_R3.md`（metric/验证/泄漏/校准）、`research_R4.md`（低算力路线与负面结果）— **原始报告，原样保留**，需要细节或原始 Finding 文本时回查。
6. `stage_a/` — 阶段 A 原始版本（ASSET_MAP.md / TASK_ANALYSIS_V1.md / RESEARCH_CHARTERS.md），保留以便审计阶段 A→B 的修正轨迹。
7. `ORIGINAL_ASSETS/` + `SHA256SUMS` + `ASSET_MANIFEST.json` — 原始题目资产（唯一权威事实来源）与哈希清单。任何与本 bundle 描述冲突之处，**以 ORIGINAL_ASSETS 为准**。

追溯路径：canonical finding（`RESEARCH_SYNTHESIS.md` CF-x）→ Merge Ledger 查原始 Finding ID（R2/R3/R4-Fxxx）→ `research_raw/research_Rx.md` 对应条目 → 其来源表 URL 或 `ORIGINAL_ASSETS/` 相对路径。

## 本模块未做的事

未训练模型、未调参、未生成 submission、未调用 Kaggle、未为后续系统选定最终方案。所有方法优先级均为**研究建议**，不是本地实验验证过的排名。
