# SEARCH_OUTPUT.md — IOAI 2026 Practice Task 1 · Search Bundle 入口

## 状态声明

- **完整性：完整，无降级。** 阶段 A 与阶段 B 均按时交付；Merge Ledger 完备（R1 的 12 findings + 6 method cards、R2 的 13 findings + 4 method cards，全部有唯一去向）。
- **原始资产校验：通过。** `ORIGINAL_ASSETS/` 实际 1291 个文件 = `SHA256SUMS` 1291 行；关键文件（starter、三个 CSV、model/ 四件）全部在场且在清单内；抽样 `sha256sum -c` 7 条（4 随机 wav + starter + train.csv + model/config.json）全部 OK。无失败项、无缺失。
- **Research 报告：2/2 按 research_plan 交付**（R1 建模工程轴、R2 metric/验证轴），均完整、按 Charter 工作、无越界。
- **历史教训目录 `context/prior_lessons/` 本次为空**（README 声明"没有提供历史教训目录"）→ 无条目可引用，未伪造；唯一一等实测教训来自系统 prompt（音频任务 CV/LB 符号反转、外部知识无条件注入负收益），已并入 RESEARCH_SYNTHESIS §3。

## 本题最重要的 5 条本地事实

1. **任务是 16→29 类扩展**：提供 16 类 AST checkpoint（`classifier.dense [16,768]`，id2label Dog…Rain）+ 旧类 296 条（`train.csv`, target 0–15）+ 新类 624 条（`fine_tune.csv`, target 16–28）；test 363 条，**预测单位 = 每个 wav 一个整数 target ∈ [0,28]**。
2. **submission 契约有文档内冲突**：模板 `archive/submission.csv` 是 **`path,target`**，而 `ioai-starter.py` 示例写 `id,prediction`（三题共用样板）。以模板为准，kernel 内再确认——这是"完美训练拿 0 分"的失败模式。
3. **`ASTFeatureExtractor` 三条硬约束（已在本机 transformers 源码验证）**：超过 1024 帧（10.24s）被**静默截断、不报错**；mel 频带上限 = 8000 Hz（8kHz 以上信息在进入模型前即消失）；不足则 zero-pad 后再整体归一化 → padding 区常量 **+0.467**。
4. **test 组成（soundfile 逐文件复核）**：323 个恰好 5.0s、仅 ~39–40 个 >10.24s（max 54.27s）；采样率 44.1k×242 / 48k×80 / 24k×26 / 96k×11 / 11025 / 8000 / 312500 / 192000 各 1。→ 长音频处理只影响 ≈11% 样本；超声修正只影响 2 个文件（≈0.55%）。
5. **无官方 val + 极端不均衡 + test 含旧类**：两 CSV 的 `split` 全为 `train`；Sheep / Keyboard Typing 各仅 3 条；test 按 (sr,ch) 签名比例（0.28–0.45，整体 363/920=0.395）看是与训练集同池的分层随机抽样，**旧 16 类估占 test ~45%** → 绝不能只训新 13 类。

## 最值得花提交额度验证的前 3 个方向

1. **第一次提交买信息，不买分数**：单类常数提交（全预测最大类）同时验证①端到端 kernel 管道、②列名契约、③metric 家族（accuracy≈0.066 / macro-F1≈0.004 / balanced-acc≈0.034，量级差 8–15×）。前提：若 starter prompt 或竞赛页已写明 Evaluation，**不要浪费这次**。（RESEARCH_SYNTHESIS MC-D / CF-14）
2. **主线 MC-A：扩头保留旧 16 行 + 920 条联合全量微调**（lr 1e-5、1024 帧、batch 8–16 + accum、epoch 10–15、freqm 24 / timem 96、mixup 0）。最优先的两个正交 A/B 是 **lr ∈{1e-5, 3e-5}** 与 **全微调 vs 冻结前 6 层**——它们影响全部 363 个样本，价值远高于只影响 39 或 2 个样本的改动。先做 0-epoch 数值等价性 sanity check（扩头后旧类 logits 必须与原模型完全一致）。
3. **推理侧 τ 开关（post-hoc logit adjustment，`logit −= τ·log π`，τ∈{0,0.5,1}）**：零训练成本，一次推理导出三份候选，用 1 次 LB 提交确定方向。它是覆盖"metric 未知"这一最大不确定性的最便宜手段。（MC-C / CF-13）

## 关键风险

- **验证风险（最高）**：源数据集（ESC-50/InsectSet）官方要求按源录音分组，但本题文件名是 hex 哈希、源信息被抹除，且 test 更像同池分层随机抽样 → **grouped CV 会系统性偏悲观，可能正是历史上"LB +0.096 / CV −0.127"符号反转的机制**。建议：主用分层随机 CV/hold-out，grouped 仅作乐观度诊断；LB 与 CV 冲突时以 LB 为准；**LB 差异 <3–4% 视为噪声**（n=363 时 accuracy 1σ≈1.9%，public 30% 时 ≈3.4%）。判别信号：首次真实提交若 LB 低于本地分层 CV >8%，切换 grouped 协议选型。
- **遗忘风险**：训练集里旧类只占 32% 但占 test ~45% → 必须把验证分数拆成 old-16 / new-13 分别报告，红线是 old-16 不得低于"原始 16 类 checkpoint 在留出集上的分数"（MC-F）。
- **工程风险**：Python `wave` 在 float32(fmt=3) wav 上抛错（用 `soundfile`）；长波形直接喂 extractor 会静默丢弃 10.24s 之后一切；`num_labels`/id2label 未同步会 shape 报错；pinned wheel 是否含 sklearn/librosa/torchaudio **未确认**，需 import 探测 + numpy-only 回退。
- **规则风险**：外部数据/预训练模型**默认禁止**（本 bundle 无题面例外条款）；**禁止**用源数据集公开标签对 test 做指纹/标签匹配（两份报告均未做，也未留任何逐文件线索）；采样率↔类的元数据先验来自题目自带数据，属合法但伪相关，仅宜做 tie-break 并在 solution 报告中披露。

## 主要冲突与未解决问题

- **冲突**（详见 RESEARCH_SYNTHESIS §7）：X1 官方 recipe 512 帧 vs 本地 checkpoint 1024 帧（以本地为准）；X2 grouped vs stratified 验证协议（双协议并存 + LB 判别）；X4 R1 的 padding 常量符号（应为 +0.467）；X5 (sr,ch) 比例区间修正为 0.28–0.45；X6 Tettigonia 超声并非全类普遍（hi8k 跨 0.005–0.998）。X3（长文件计数）经复核为非真冲突。
- **未解决（Top 3）**：① **metric 公式与方向未知**（公网确证不可得：竞赛页登录墙、Kaggle API 401、搜索引擎零命中 → 也意味着不存在本题公开 solution）；② public/private LB 比例与是否全量计分；③ 是否强制使用提供 checkpoint / 是否允许外部资源（按默认禁止处理）。完整 11 项见 RESEARCH_SYNTHESIS §8。

## Bundle 内容与阅读顺序

1. **本文件** — 导航与高密度交接。
2. **`TASK_ANALYSIS.md`** — 最终题目理解（可独立阅读）：目标、输入/输出/预测单位、metric 分支、submission contract、数据统计、提供资产的硬约束、验证与泄漏、资源与规则边界、[LOCAL_FACT]/[EXTERNAL_EVIDENCE]/[INFERENCE]/[UNRESOLVED] 分区、阶段 A 被修正项。
3. **`RESEARCH_SYNTHESIS.md`** — 16 条 canonical findings、10 张 method cards（MC-A…MC-J，全部通过 kernel 可行性门；不可行方法只在 §6 作思路来源）、历史教训对照、负面结果、冲突表、未解决问题、来源表、Merge Ledger。**method card 的顺序是研究建议，不是本地实验验证过的排名。**
4. **`ASSET_MAP.md`** — 最终资产地图：1291 个文件的类别/用途/schema/关系、读取与抽样覆盖范围、关键统计、与阶段 A 的 6 处修正、无法解析项、完整性校验状态。
5. **`research_raw/research_R1.md`、`research_raw/research_R2.md`** — 两份原始报告，**原样保留**，用于追溯任何 canonical 条目的原始措辞与 Finding ID。
6. **`stage_a/{ASSET_MAP.md, TASK_ANALYSIS_V1.md, RESEARCH_CHARTERS.md}`** — 阶段 A 原始产物（含 research_plan），原样保留，用于审计判断演化。
7. **`ORIGINAL_ASSETS/` + `SHA256SUMS` + `ASSET_MANIFEST.json`** — runner 生成的原始资产（只读）与哈希清单，**任何数据事实都应回到这里核对**，不要以本 bundle 的摘要替代原始资产。

追溯路径示例：canonical 条目 `CF-4` → RESEARCH_SYNTHESIS §1 → 溯源标注 `R1-F003/R1-F004` → `research_raw/research_R1.md` 对应段落 → 原始依据 `ORIGINAL_ASSETS/archive/model/preprocessor_config.json`。
