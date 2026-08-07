# R2 Research Report

（轴 3：题面/metric/验证与泄漏。时间盒 25 min，联网工具只有 curl；无 Kaggle 凭证。）

## 0. Top-5 可行动方向

1. **metric 不要靠猜：先在 kernel/竞赛页里读，读不到就用 1 次“单一常数类”提交做 metric 判别探针**。全预测最大类（Thunderstorm 估 ~24/363）→ accuracy 期望 ≈0.066，macro-F1 期望 ≈0.0043（差 15 倍，LB 分数一眼可辨）。最小验证：第 1 次提交即为该常数提交，看 LB 数值落在哪个量级（R2-F004/R2-F009）。
2. **验证协议要模仿 test 的生成方式，而不是模仿源数据集论文的 fold**。本地统计强烈支持 test 是对 train+fine_tune 同池按类分层随机抽样 ≈28.3%（test/总=363/1283；按 (sr,ch) 签名分组的 test/train 比例 0.31–0.40 一致）。因此**分层随机 K-fold（分层于类别）比 ESC-50 式“按源录音分组”的 grouped CV 更接近 LB**；grouped CV 会系统性偏悲观。最小验证：同时算 stratified-CV 与 group-CV（近重复聚类），用 2 次 LB 提交校准哪条与 LB 同向（R2-F005/R2-F006/R2-F010）。
3. **稀有类的价值完全由 metric 决定，务必写成一个可切换的开关**：Sheep/Keyboard Typing 训练各 3 个 → test 期望各 ~1 个。accuracy 下每个 ~0.28%（可放弃）；macro-F1 下每类固定占 1/29≈3.4%（猜对一个 Sheep 就 +~3.4 分，远大于把 Thunderstorm 提升 5% 的收益）。实现：训练一份模型，推理时对 logits 加 `-τ·log(prior)`（post-hoc logit adjustment，τ∈{0,0.5,1}），τ 由 metric/LB 选定（R2-F007，Method Card MC-1）。
4. **把 LB 当一等测量资源，按预算表花**：363 个 test、若 public 只占一部分，accuracy 的 1σ 约 1.6%（n=363）~2.9%（n≈109）；macro-F1 因含 1 样本类，1σ 可达 3–5%。**LB 上 <3% 的差异不可作为决策依据**；应用少数几次提交做“大幅度、正交”的对照（metric 探针 / τ=0 vs τ=1 / 旧类是否参与训练），而不是微调超参（R2-F008）。
5. **test 覆盖 29 类且旧类占多数：绝不能只训新 13 类**。test 中仅 40 个文件时长≠5.0s（昆虫类特征），其余 323 个是 5s 片段，(sr,ch) 签名与旧类/FSC22 类池吻合 → 旧 16 类必然在 test 中且占大头（估 ~45% 以上）。最小验证：一次提交“只预测 16–28 类”会使分数上限≈新类占比，可直接量化旧类比重（但更省的做法是直接按 29 类训练，不浪费提交）（R2-F002/R2-F003）。

## 1. Charter 与实际覆盖范围

- 已完成：本题 Kaggle 页可达性验证（结论：**私有/需登录，公网无任何题面、metric、rules 文本**）；submission 契约核对（发现 starter 与模板列名冲突）；test 组成的本地统计推断（类空间、期望每类样本数）；ESC-50 官方 fold 语义（primary source）；InsectSet 官方 split 语义（primary source）；不均衡下 accuracy vs macro-F1 的策略差异与 post-hoc 校正的 primary source；LB 方差与提交预算的定量估计。
- 未完成/受限：无法读取竞赛 Overview/Evaluation/Rules（登录墙）；搜索引擎（Google/Bing/DDG）在本环境全部返回验证码或无结果，无法做关键词发现；FSC22 论文站点 403，其 split 惯例未获 primary 证据。
- 未越界：未研究模型结构/超参/解码工程（R1 范围）；未下载任何源数据集标签，未做任何 test 文件与公开数据集的指纹/标签匹配。

## 2. 对题目分析的核对或修正

- **[修正 1｜submission 列名冲突]** `ORIGINAL_ASSETS/ioai-starter.py` 末尾示例写出的是 `["id","prediction"]` 两列，而 `archive/submission.csv` 表头是 `path,target`。TASK_ANALYSIS 只写了后者。starter 的示例是通用样板（同一 starter 用于三题），**以本题挂载的 `submission.csv` 模板为准（`path,target`）**，但下游必须在 kernel 里再确认一次（若 kernel 报 "unexpected column"，回退到 id/prediction 是唯一备选）。这是一个真实的提交失败风险点。
- **[修正 2｜test 类空间从 INFERENCE 升级为强证据]** ASSET_MAP 把“test 是否含旧类”列为 UNRESOLVED。本地逐文件 WAV 头统计（我自己重算）给出：test 363 个文件中 **323 个时长≈5.0s、40 个非 5s**；非 5s 的采样率集合 {44.1k×25, 48k×11, 8k, 96k, 192k, 312.5k} 与 fine_tune 昆虫类（25–28）签名一致；5s 文件的 (sr,ch) 分布（44.1k/1:158、44.1k/2:84、48k/2:54、24k/2:24…）只能由旧 16 类 + FSC22 系新类（16–24）解释。→ **test 覆盖旧类，且旧类+FSC22 类占绝大多数**。
- **[修正 3｜test 是分层随机抽样，不是“按源录音留出”]** 按 (sr,ch) 签名的 test/train 计数比：24k/2 → 24/78=0.31；整体 363/920=0.395；各签名均落在 0.3–0.4。若组织者做过“同源录音整组留出”，这些比例会更参差且某些签名会全空。→ **本地验证应优先分层随机切分**（详见 R2-F005/F006）。
- **[核对通过]** train.csv 与 fine_tune.csv 的类分布、path 不相交、5s vs 变长的描述与我的独立统计一致。
- **[未能核对]** metric、是否强制使用提供的 checkpoint、public/private LB 比例：公网无法获得（登录墙），仍为 UNRESOLVED。

## 3. 检索方向与来源覆盖

- Kaggle 竞赛页：`/competitions/ioai-2026-ai-models-track-practice-task-1` HTTP 200（对照 `ioai-2026-practice-task-1` 返回 404），说明 slug 存在；但 SPA 外壳无内容，且经 reader 代理解析后标题为 **"Login or Register | Kaggle"** → 私有竞赛，题面不公开。Kaggle 公共 API `/api/v1/competitions/list?search=ioai` 返回 401 Unauthenticated（无凭证）。
- 搜索引擎：Google/Bing/DuckDuckGo 对 `"ioai-2026-ai-models-track"` 均无可用结果（DDG 返回 bot challenge，Bing/Google 结果页无 kaggle.com 命中）→ 没有公开讨论、没有他人 write-up；**"同名题 solution" 风险为零，因为不存在**。
- IOAI 官方站点 ioai-official.org 首页可达，但 `/ai-models-track/` 404；未在时间盒内定位到 AI Models Track 规则页（且其内容已由系统 prompt 概述，边际价值低，停止扩展）。
- 评估协议 primary sources：ESC-50 官方 repo README（fold 语义）、InsectSet/PLOS Comput Biol 2023（split 语义）。FSC22 官方论文 403 未获。
- 方法 primary sources：logit adjustment（ICLR 2021）、F1 阈值理论（Lipton et al. 2014）。饱和后停止。

## 4. 关键 Findings

- **R2-F001 [UNRESOLVED]** 本题 metric、rules、外部数据/模型条款在公网**不可获得**：竞赛页需登录（reader 代理返回 "Login or Register"），Kaggle API 401，搜索引擎零命中。→ 下游不得把"已确认 metric"写进方案；必须在 kernel/竞赛页内读取，或用 §5 MC-2 的 LB 探针判别。边界：若人类操作员粘贴的 starter prompt 里带有题面文本，那份文本优先于本报告的一切推断。
- **R2-F002 [LOCAL_FACT]** test 363 文件：323 个 ≈5.0s，40 个非 5s（时长 5.3–54.3s；含 8k/96k/192k/312.5k 采样率）。命令：读 `archive/submission.csv` 逐 path 解析 WAV fmt/data chunk。→ 昆虫类（25–28）在 test 中约 40 个（≈11%），其余为 5s 类。
- **R2-F003 [INFERENCE，证据强]** test 按类分层抽样，比例 ≈ 363/920=0.395×train 每类数。→ 期望每类 test 数：Thunderstorm ~24、Frog/Bird ~14、Wolf ~9、Cow/Mouse ~9、ESC-50 类各 ~5、Rain ~3、**Sheep ~1、Keyboard Typing ~1**、Crackling Fire ~9、FSC22 新类各 ~18、昆虫各 ~24。边界：这是比例推断，非逐类证实；实际可能有 ±2 抖动，Sheep/Keyboard 也可能为 0 或 2。
- **R2-F004 [INFERENCE]** metric 判别探针的期望值：全预测 Thunderstorm（最大类，p≈0.066）→ accuracy≈0.0661；macro-F1（29 类平均）≈0.0043；balanced accuracy≈1/29=0.034。三者数量级各异 → **一次提交即可判别 metric 家族**。边界：若 public LB 只用部分 test，数值会抖动，但量级差异（15×）足够稳健。
- **R2-F005 [EXTERNAL_EVIDENCE]** ESC-50 官方说明：数据集"prearranged into 5 folds for comparable cross-validation, **making sure that fragments from the same original source file are contained in a single fold**"，metadata 提供 `src_file`/`take` 字段（https://github.com/karolpiczak/ESC-50）。→ 源数据集里同源片段确实存在，是近重复泄漏的真实来源；**但本题文件名是 16-hex 哈希，src_file/take 信息被抹掉**，无法直接复现该分组。
- **R2-F006 [EXTERNAL_EVIDENCE]** InsectSet 官方 split：按每个物种单独切 train/val/test（62.7/15.2/22.1），且"resulting edited snippets from one original recording were treated as one audio example to prevent them from ending up in multiple data sub-sets"（https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011541）。→ 昆虫类的源数据本身已按录音去重；本题昆虫类文件多为完整变长录音（3–120s），同源近重复风险相对低。
- **R2-F007 [EXTERNAL_EVIDENCE]** post-hoc logit adjustment（对 logits 减 τ·log 类先验）是"简单、有统计依据"的长尾修正，等价于优化 balanced error（Menon et al., ICLR 2021, https://arxiv.org/abs/2007.07314）。F1 侧：对校准概率，最优阈值 = 最优 F1 的一半；完全无信息的分类器在 F1 下最优行为是"全部判正"（Lipton et al., https://arxiv.org/abs/1402.1892）。→ 二者共同含义：**macro-F1/balanced 类 metric 会奖励对稀有类的激进预测；accuracy 会惩罚它**。本题必须让 τ 可切换。
- **R2-F008 [INFERENCE]** LB 噪声量化：accuracy 的标准差 √(p(1−p)/n)，p≈0.85、n=363 → 1.9%；若 public 仅 30%（n≈109）→ 3.4%。macro-F1 中 Sheep/Keyboard 只有 ~1 个样本，单个样本翻转就改变该类 F1 从 0→1，即整体 ±3.4%/类。→ **LB 差异 <3–4% 无统计意义**；50 次提交应买"大幅度正交对照"，不是买超参微调。
- **R2-F009 [INFERENCE]** 提交预算建议分配（总 50，2h 窗口现实上只能用 5–10 次）：1 次 metric 探针（常数类）→ 1 次 baseline（29 类微调，τ=0）→ 1 次 τ=1（或 τ=0.5）→ 1 次最佳 + 长音频分块聚合开关 → 其余留给失败恢复/明显更强的变体。任何一次提交都必须与前一次只差一个可解释的因子。
- **R2-F010 [INFERENCE]** 由于 test 是同池分层随机抽样（R2-F003），**train/fine_tune 与 test 之间同样存在同源近重复**（source recording 被切成多个 5s 片段，部分片段落进 test）。这意味着：(a) LB 分数会天然偏高，"泄漏"对我们有利且合法；(b) 用 grouped CV（把近重复聚成一组）估计出的分数会**低于** LB；(c) 用普通 stratified CV 估计出的分数与 LB 更同向。系统历史教训中"LB +0.096 / 5-fold CV −0.127"的符号反转，与"CV 协议比 test 更严格/更保守"这一机制一致。
- **R2-F011 [LOCAL_FACT]** starter 与模板的列名冲突（见 §2 修正 1），路径：`ORIGINAL_ASSETS/ioai-starter.py`（写 `id,prediction`）vs `ORIGINAL_ASSETS/archive/submission.csv`（`path,target`）。
- **R2-F012 [LOCAL_FACT]** train.csv/fine_tune.csv 的 `split` 列全为 `train`，**无官方 val**；因此任何验证集都是我们自己造的，验证协议选择本身是一个自由度，必须被 LB 校准。
- **R2-F013 [INFERENCE]** 旧类数据（train.csv, 296 行）在评分中不是装饰：test 中旧类估计 ≥160 个文件（5s 且签名匹配旧类池）→ **旧 16 类约占 test 的 45%+**。只微调新类、或让新类头压过旧类，会直接损失接近一半的分数。给定的 16 类 checkpoint 对旧类已经很强，"保留旧头行权重 + 只学新类行"因此是高价值先验（结构细节属 R1）。

## 5. Method Cards

### MC-1 Post-hoc 类先验/logit 调整（τ 可切换）
- 家族：长尾学习 / 决策阈值后处理。
- 机制：推理时 `logit_c ← logit_c − τ·log π_c`，π_c 为训练类频率（或估计的 test 先验）。τ=0 为 argmax（近似最优 accuracy）；τ=1 为 balanced-error 最优（近似 macro-recall/macro-F1 友好）。
- 解决的问题：metric 未知 + 极端不均衡（3 样本类 vs 62 样本类），且 train 先验（296:624 且类内 3–62）与 test 先验（分层抽样，比例相同）一致——但**训练时同时用 train+fine_tune 会让新类样本量占 68%**，模型天然偏新类，post-hoc 调整可纠偏。
- 证据：https://arxiv.org/abs/2007.07314 （ICLR 2021）；F1 阈值理论 https://arxiv.org/abs/1402.1892 。
- 本题适配：零训练成本（只改 argmax 前一行），不影响 kernel 时限；可以在同一次推理里输出 τ∈{0,0.5,1} 三份 submission，只交其中一份，LB 对比后再交第二份。
- 与其他方法的结构差异：不同于 loss 加权/过采样（改变训练动态、需重训），这是纯推理侧、可逆、可枚举。
- 最小验证：本地分层 CV 上画 accuracy(τ) 与 macro-F1(τ) 两条曲线（应当交叉：accuracy 在 τ≈0 最好，macro-F1 在 τ≈0.5–1 最好）；然后用 1 次 LB 提交确认方向。
- 成本：<1s。依赖：numpy。
- Kaggle kernel 可行性：**可行**（无额外依赖、无额外训练）。
- 失败模式：若 metric 是 plain accuracy 且 τ 取大，稀有类误报会吃掉大类正确率（估计 −1~3%）；若 test 先验与 train 先验不同（本题证据显示相同，风险低）。
- 置信度：高（机制简单、有 primary source、代价近零）。依据：R2-F003/F007。
- 污染/规则风险：无。

### MC-2 单类常数提交作为 metric 判别探针
- 家族：LB 探测 / 实验设计。
- 机制：提交"全部预测同一个类"的 submission，用 LB 返回值的量级反推 metric（R2-F004：accuracy≈0.066 / macro-F1≈0.004 / balanced-acc≈0.034）。
- 解决的问题：R2-F001（metric 未知）——这是本题最大的单点不确定性，会决定 τ、是否照顾稀有类、是否做类平衡采样。
- 证据：本地类分布统计（R2-F003）+ metric 定义本身（算术，无需外部来源）。
- 本题适配：花 1/50 次提交，换取整条策略分支的确定性；比在本地反复猜 metric 便宜得多。**前提：先确认竞赛页/starter prompt 里没有直接写明 Evaluation；如果写明了，不要浪费这次提交。**
- 与其他方法的差异：不是模型改进，是信息获取；其它 LB 探针（探测类先验、探测 public 比例）价值远低且花费更多提交，不推荐。
- 最小验证：即它本身。可与"必须先跑通一次端到端提交管道"合并（同一次提交同时验证列名契约 R2-F011）。
- 成本：一次 kernel 运行（无需训练，可用纯 CSV 生成，几十秒）。
- Kaggle kernel 可行性：**可行**（不需要 GPU/模型）。
- 失败模式：public LB 子集很小时数值抖动，但量级差异 15× 足以判别；若 metric 是加权/自定义（例如新旧类分别加权），单点值可能落在中间 → 此时记为"非标准 metric"，改用两次探针（全预测某旧类 vs 全预测某新类，比较数值比）。
- 置信度：高。
- 污染/规则风险：无（只使用自己提交的公开反馈；不涉及标签泄漏）。

### MC-3 双协议验证（stratified CV 为主 + 近重复分组 CV 为诊断）
- 家族：验证设计。
- 机制：(a) 主协议：按 target 分层的 5-fold（或 hold-out 20%，与 test 的 28% 抽样率同量级），因为 test 本身就是同池分层随机抽样（R2-F003）；(b) 诊断协议：先用提供的 AST encoder 提 embedding，做余弦相似度阈值聚类，把近重复（同源录音片段）合成 group，跑 GroupKFold；两者的差值 = 泄漏带来的乐观量。若差值很大，说明本地"进步"可能只是记住了近重复 → 优先信任 LB。
- 解决的问题：R2-F010 / 历史 CV-LB 符号反转；无官方 val（R2-F012）。
- 证据：ESC-50 官方 fold 明确按 src_file 分组以防同源片段跨 fold（https://github.com/karolpiczak/ESC-50）；InsectSet 把同录音 snippet 视为一个样本（https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011541）。→ 同源近重复在这两个源池里是被官方承认的真实现象。
- 本题适配：**不要照抄 ESC-50 的 grouped 协议作为选型依据**（它比 test 更严格，会误杀对 LB 有效的改动）；用它只做"我的 CV 有多虚"的量化。稀有类（Sheep/Keyboard 3 样本）在 5-fold 里每 fold 至多 1 个 → 单类 F1 方差极大，**报告 CV 时必须同时给出 accuracy 与 macro-F1，并对 3 样本类的 per-class 结果加"不可信"标注**。
- 与其他方法差异：比单一 CV 多一条诊断线；比纯 LB 驱动省提交。
- 最小验证：一次训练同时输出两套 fold 的分数（复用同一份 embedding/预测），成本≈0 额外训练。
- 成本：分层 CV = k 次训练（时间紧时用单次 80/20 hold-out + 一次全量重训）；embedding 聚类 = 一次前向（1283 文件，几十秒 GPU）。
- Kaggle kernel 可行性：**验证只在探索阶段做；最终计分 kernel 只需 1 次全量训练 + 推理**，可行。若在计分 kernel 里跑完整 5-fold 训练，时限风险上升 → 建议最终提交用全量单模型（或 ≤3 fold 集成，视 R1 的时间预算）。
- 失败模式：embedding 聚类阈值选错会把不同类误合并；分层 CV 在 3 样本类上 fold 不平衡（用 `StratifiedKFold` 会对 n<k 的类报警，需回退到"稀有类固定放进训练集"）。
- 置信度：中高（协议匹配 test 生成机制的推断证据强，但未经 LB 直接验证）。
- 污染/规则风险：无（不使用任何外部标签）。

### MC-4 「新旧类分别评估」的诊断视图（不是训练方法）
- 家族：误差分析。
- 机制：把本地验证分数拆成 old-16 与 new-13 两块分别报告（并按估计的 test 占比 ~45%/~55% 加权），以及"新旧混淆矩阵的跨块错误率"。
- 解决的问题：R2-F013——旧类占 test 近一半，但训练数据里旧类只占 32%；最常见的失败是新类头吞掉旧类（catastrophic forgetting），而整体 CV 分数会掩盖它。
- 证据：本地统计（R2-F002/F003）；无需外部来源。
- 本题适配：给下游一个明确的红线指标：**如果 old-16 块的验证 accuracy 明显低于"直接用原始 16 类 checkpoint 在 train.csv 上留出集"的分数，说明扩展微调破坏了旧类，必须回退（冻结更多层/降 lr/旧类过采样）**。
- 与其他方法差异：纯度量层面，与 R1 的结构选择正交，但为其提供判据。
- 最小验证：一次训练即可产出两块分数。
- 成本：0 额外。Kaggle kernel 可行性：可行（仅日志）。
- 失败模式：留出集太小（Sheep 1 个）导致 old 块噪声大 → 报告 macro 与 micro 两个版本。
- 置信度：高。
- 规则风险：无。

## 6. 负面结果、失败条件与冲突证据

- **公网上不存在本题任何题面/讨论/solution**（Google/Bing/DDG 零命中，竞赛页登录墙）。→ 任何声称"这道题的 metric 是 X"的外部材料都不可信；不要继续在这个方向烧时间。
- **不要照抄 ESC-50 5-fold 作为选型协议**：它是"按源录音分组"的严格协议（R2-F005），而本题 test 是同池分层随机抽样（R2-F003）→ 用它选型会偏保守，可能正是历史上"CV 与 LB 反号"的机制来源（R2-F010）。这是本报告与"照搬源数据集官方协议"这一直觉的**显式冲突**。
- **starter 示例的 `id,prediction` 与模板 `path,target` 冲突**（R2-F011）：这是可以让一次完美训练拿 0 分的失败模式。第一注意力应放在管道正确性上。
- **macro-F1 假设下的 per-class 方差**：Sheep/Keyboard Typing 期望各 ~1 个 test 样本；这意味着即使 metric 是 macro-F1，这两类的贡献也接近抛硬币（±3.4 分/类），**不要把整套方案押在"提升稀有类"上**；用 MC-1 的低成本开关拿走期望收益即可。
- **过度自信风险**：R2-F003 的分层抽样推断基于 (sr,ch) 签名比例一致性，不是逐文件证实。若实际是"按源录音整组留出"，则近重复泄漏消失、grouped CV 才是正确协议，且 LB 会明显低于本地 → **判别信号：第一次真实提交的 LB 若显著低于本地 stratified CV（差 >8%），应立即切换到 grouped 协议做选型**。
- 未获证据：FSC22 官方论文的 split 建议（MDPI 403）。因此对 FSC22 类（16–24，各 45 个）是否存在大量同源 5s 片段，只有"5s 定长切片"这一间接证据。

## 7. 未解决问题

1. metric 公式与方向（accuracy / macro-F1 / 其他）——只能在竞赛环境内读取或用 MC-2 探针判别。
2. public/private LB 比例与 test 是否全量计分——影响 R2-F008 的噪声估计（现给出 n=363 与 n≈109 两个界）。
3. 是否强制使用提供的 checkpoint、是否允许外部数据/权重——公网无信息；按"默认禁止"处理。
4. FSC22 类的同源片段比例（论文 403），影响近重复泄漏的绝对量。
5. test 中是否真的存在 Sheep/Keyboard Typing 样本（期望 ~1，可能为 0）。

## 8. 给题目分析 Agent 的合并建议

- 把 ASSET_MAP 的 UNRESOLVED #3（test 是否覆盖全部 29 类）降级为"已基本解决"：test 含旧类且旧类占 ~45%，昆虫类 ~40 个（R2-F002/F003/F013）。
- 在提交契约一节加入 starter/模板列名冲突警告（R2-F011），并规定：kernel 内以读入的 `submission.csv` 模板列名与 path 顺序为准。
- 把"metric 未知"从待研究项改成**可执行分支**：默认 τ=0（accuracy 友好）+ 一次常数类探针（MC-2）+ τ 开关（MC-1）。
- 验证协议写成：**主 = 分层随机 5-fold/hold-out；辅 = 近重复分组 CV 作为乐观度诊断**；并写明 LB <3–4% 差异视为噪声（R2-F008）。
- 明确提交预算表（R2-F009），并要求每次提交只改变一个可解释因子。
- 保留禁令：不得使用源数据集公开标签做任何 test 匹配（本报告未做，也不提供任何逐文件线索）。

## 9. 来源表

| 标题 | URL | 日期/版本 | 类型 | 支持 Finding |
|---|---|---|---|---|
| Kaggle 竞赛页（本题 slug，返回登录页） | https://www.kaggle.com/competitions/ioai-2026-ai-models-track-practice-task-1 | 2026-08-04 访问 | 官方竞赛页（受登录墙） | R2-F001 |
| Kaggle 公共 API competitions/list（401 Unauthenticated） | https://www.kaggle.com/api/v1/competitions/list?search=ioai | 2026-08-04 访问 | 官方 API | R2-F001 |
| IOAI 官方站点（首页可达，AI Models Track 子页 404） | https://ioai-official.org/ | 2026-08-04 访问 | 官方公告站 | R2-F001 |
| ESC-50 官方数据集 README（5-fold 按 source file 分组；metadata 含 src_file/take） | https://github.com/karolpiczak/ESC-50 | master, 2026-08-04 访问 | 官方数据集仓库 | R2-F005, MC-3 |
| Faiß & Stowell, Adaptive representations of sound for automatic insect recognition（InsectSet32/47/66 split 协议、同录音 snippet 视为一个样本） | https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011541 | PLOS Comput Biol 2023 | 原始论文 | R2-F006, MC-3 |
| Menon et al., Long-tail learning via logit adjustment | https://arxiv.org/abs/2007.07314 | ICLR 2021 (v2) | 原始论文 | R2-F007, MC-1 |
| Lipton et al., Thresholding Classifiers to Maximize F1 Score | https://arxiv.org/abs/1402.1892 | 2014 | 原始论文 | R2-F007, MC-1 |
| 本地资产：`ORIGINAL_ASSETS/archive/{train,fine_tune,submission}.csv` + 1283 个 wav 头逐文件统计（只读） | 本地路径（见 ASSET_MAP） | 本次运行 | 官方题目资产 | R2-F002/F003/F011/F012/F013 |
| 本地资产：`ORIGINAL_ASSETS/ioai-starter.py`（示例写 id,prediction） | 本地路径 | 本次运行 | 官方 starter | R2-F011 |
