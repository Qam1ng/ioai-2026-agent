# R3 Research Report

（axis: metric / 验证设计 / 泄漏 / 校准 / 后处理。所有本地数字均由我自己在 ORIGINAL_ASSETS 上重算，脚本产物在 /tmp，未修改任何资产。）

## 0. Top-5 可行动方向

1. **时长门控标签空间约束（近乎免费的 ~11% 测试集）**：labeled 集中 **每一个时长≠5.00s 的文件都属于昆虫类 25–28（238/238，精度 100%）**，反向 99.2%（240 昆虫里仅 2 条为 5.00s）。测试集 40 条非 5s → 直接把这 40 条的 argmax 限制在 {25,26,27,28}。最小验证：在本地 labeled 上算该规则的 precision/recall（1 分钟，纯 metadata），LB 上用 1 次提交做 A/B（同一模型 ± 约束）。
2. **常量类提交探针（1–3 次提交同时解决 metric 与测试先验两个未知量）**：全预测某一类 c。若 metric=accuracy，得分严格等于 n_c/363（可直接反解该类在测试集的条数）；若 metric=macro-F1，得分 ≈ (1/29)·2n_c/(363+n_c)，量级小一个数量级，两者不可能混淆。建议探针类：11(Thunderstorm，labeled 最大旧类)、25(昆虫)。最小验证：即刻可做，不需模型。
3. **后处理只做「改变 argmax」的操作：per-class logit bias / 预测数配额匹配；不要做温度缩放**。硬标签 metric 下温度缩放对 argmax 恒等无效（数学事实），而 logit 减去 τ·log(prior)（Menon 2020 logit adjustment）、以及新类整体减一个偏置（WA/BiC 观察到增类后新类 FC 权重/logit 系统性偏高）才会改变预测。最小验证：在 held-out 上扫 τ∈{0,0.25,0.5,1.0} 与新类偏置 δ∈{0,0.5,1.0}，看 accuracy 与 macro-F1 两条曲线的符号是否一致；不一致则等 metric 探针结果再定。
4. **验证协议：repeated stratified 5-fold ×3 + 三个分开汇报的子指标（overall acc / 旧 16 类 macro-recall / 新 13 类 macro-recall）+ 配对 bootstrap**。稀有类（Sheep 3、Keyboard Typing 3、Rain 7）用「每折至多 1 条」的手工分配，不要交给 sklearn 自动 stratify 后再看单折。判据：只有在 **同一批折上配对 bootstrap 的 95% CI 不含 0** 时才认为本地改进真实；单次 5-fold 上 <0.02 的差异一律视为噪声。
5. **组结构泄漏：不要相信廉价频谱指纹，用 AST encoder embedding 做近重复连通分量再分折**。我实测的 256 维粗指纹（8 段×32 频带，8kHz）在 labeled 上 LOO 1-NN 只有 0.363，相似度 >0.95 的 labeled 对同标签率仅 0.257 → 该指纹不是重复检测器。最小验证：用题目自带 checkpoint 的 CLS embedding 算余弦，阈值用「昆虫类同类内高相似簇」当正对照，看是否出现 size>1 的紧密簇；再据此建 group-aware 折。

## 1. Charter 与实际覆盖范围

覆盖：metric 语义差异与零和决策（§4 F008/F010、§5 MC-3）、极小样本验证协议与噪声判据（F011、MC-5）、近重复/组泄漏检测（F006/F007、MC-4）、提交额度策略（F012、MC-1）、后处理与校准含 metadata 捷径（F002/F003/F004/F005、MC-2/MC-3）。
未覆盖（按 forbidden_overlap 交给 R2/R4）：torchaudio/HF 前处理实现、具体防遗忘训练配方。
时间盒内未完成：真正跑 AST embedding 近重复（需加载 345MB checkpoint + 依赖，超出「不训练/不装依赖」边界）；只给出方案与正对照设计。

## 2. 对题目分析的核对或修正

- **核对通过**：三 CSV 类别计数、296/624/363 行数、旧 16 类/新 13 类、path 无交集，与 ASSET_MAP 完全一致（我用独立脚本重算）。
- **修正/加强 1（重要）**：TASK_ANALYSIS §6 把「时长>6s ⇒ 昆虫」称为捷径，但强度被低估。实测：labeled 920 条中 **非 5.00s 文件共 238 条，全部是 25–28**；train(296) 与 16–24(384) **无一条非 5.00s**。即规则「dur≠5.00 ⇒ ∈{25..28}」在 labeled 上 precision=1.000, recall=238/240=0.992。这不是弱捷径，而是接近确定的标签空间约束（仍需 LB 验证测试集是否同源处理）。
- **修正 2**：ASSET_MAP 把「采样率>100kHz ⇒ 昆虫」当作捷径。实测昆虫 240 条中 44100(156)+48000(60)=216 条属于常见采样率，仅 ~17 条为 ≥192k 的超声；同时 96000 出现在旧类(9)、16–24(26)、昆虫(3) 三处。**采样率不是可靠的昆虫标记**，只有 ≥192kHz 是单向证据（测试集仅 2 条）。真正有判别力的 metadata 是 **时长**，其次是 24000Hz：24000 只出现在 16–24（labeled 83 条，测试 26 条），是「新环境类数据源」的组标记，可用于 group-aware 分折而非直接预测。
- **修正 3（会改变先验/校准决策）**：如果测试集按 labeled 先验分层，昆虫应占 363×240/920 ≈ 95 条，且几乎都非 5s；实测测试集只有 40 条非 5s。测试中所有 ≥192k/8k/312.5k 的异常采样率文件都是长文件（25.5s/13.1s/15.6s/33.0s），只有一条 11025Hz 是 5.0s。→ **测试类先验与 labeled 先验不同（昆虫占比 ~11% 而非 ~26%），或部分昆虫测试片段被裁到 5s**。任何「用 labeled 先验做 logit prior 调整」的做法都建立在一个已被本地证据削弱的假设上；先用常量类探针测真实先验。
- **保留冲突**：submission 表头 `path,target` vs starter 的 `id,prediction`；官方 metric 与外部资产许可仍无本地文本依据（[UNRESOLVED]，须由能读题面者确认）。

## 3. 检索方向与来源覆盖

已查（primary 优先）：ESC-50 官方仓库 README（fold 与同源片段规则）、scikit-learn `f1_score` 官方文档（macro/zero_division 语义）、arXiv 原文摘要（WA 1911.07053 类增量 FC 权重偏置；BiC 1905.13260 新旧类不平衡+bias correction；logit adjustment 2007.07314）。
本地实测替代检索：类别/时长/采样率交叉表、粗指纹近重复实验、二项 CI 计算——这些问题网页无法回答，且我自己算的数字比任何二手来源都可靠。
主动停止的方向：更多 ESC-50 SOTA 排行（与 metric/验证轴无关）、通用「小数据 CV 技巧」博客（无新增限制或反例）。

## 4. 关键 Findings

- **R3-F001 [LOCAL_FACT]** 类别计数与 ASSET_MAP 一致：旧类 296（Sheep 3、Keyboard Typing 3、Rain 7、Thunderstorm 62），新类 624（16:24、17–24:各 45、25–28:各 60）。来源：`archive/train.csv`,`archive/fine_tune.csv` 独立重算。相关性：所有 metric/验证结论都以此为输入。
- **R3-F002 [INFERENCE，基于 LOCAL_FACT]** 若测试 363 条按 labeled 先验分层，期望每类条数：Sheep 1.2、Keyboard Typing 1.2、Rain 2.8、Dog/Rooster/Pig/Cat/Hen/Crow/Sea Waves 各 4.7、Thunderstorm 24.5、昆虫各 23.7。→ **macro-F1 下一条 Sheep 样本的对错值 ~1/29≈3.45 分，而 accuracy 下值 1/363≈0.28 分，杠杆差 12.5 倍**。边界：F004 显示测试先验可能不成比例，具体数字要靠探针校正。
- **R3-F003 [LOCAL_FACT]** labeled 中非 5.00s 文件 238 条，全部属 25–28；昆虫时长分布（排序抽样）3.0/6.2/8.2/10.1/12.0/14.7/17.0/21.8/30.2/50.0s，最长 120s。测试集非 5s 共 40 条（5.3–54.3s）。→ 时长门控规则见 §0-1。
- **R3-F004 [LOCAL_FACT + INFERENCE]** 测试集 40 条非 5s ≪ 按 labeled 先验期望的 ~94 条；测试异常采样率文件（192000/312500/8000）全部是长文件。推断测试昆虫占比 ≈11%（40/363），远低于 labeled 的 26%。→ 直接反对「用 labeled 类频率做 balanced-softmax/prior 反加权」的默认做法，尤其反对给昆虫类加权。
- **R3-F005 [LOCAL_FACT]** 采样率交叉表：24000Hz 仅见于 16–24（labeled 83，测试 26）；96000 见于旧类/新环境类/昆虫三处；≥192kHz 仅昆虫（labeled 17，测试 2）；测试 11025Hz 1 条为 5.0s。→ SR 不能当昆虫检测器，但可当数据源分组键（group-aware CV）。
- **R3-F006 [LOCAL_FACT，我实测的负面结果]** 廉价频谱指纹（每文件取前 5s，线性插值到 8kHz，8 时间段×32 频带 log 幅度，256 维，L2 归一化；1283 条 CPU 全量约 1 分钟）：labeled LOO 1-NN 准确率 **0.363**（chance 3.4%，多数类 6.7%）；labeled 内 cos>0.95 的对同标签率仅 **0.257**（>0.9 时 0.208）；测试→labeled 最大相似度中位 0.812，仅 13 条 >0.97。结论：该指纹既不足以做近重复检测，也不能证明存在 train/test 近重复。**不要用它构造 group split，也不要据此宣称无泄漏。**
- **R3-F007 [EXTERNAL_EVIDENCE]** ESC-50 官方 README：clips 由 Freesound 长录音手工切出，且「dataset has been prearranged into 5 folds ... making sure that fragments from the same original source file are contained in a single fold」。来源 https://raw.githubusercontent.com/karolpiczak/ESC-50/master/README.md 。相关性：本题旧类/部分新类的类目与 5s/44.1kHz 规格与 ESC-50 高度一致，文件名被哈希后 fold/clip/take 信息全部丢失 → **随机分层 CV 在这类数据上系统性乐观**，官方数据集作者本人认为必须按源录音分折。边界：本题类计数（Thunderstorm 62 ≠ ESC-50 的 40/类）说明数据已被重组，不能假设任何公开 fold 可复用。
- **R3-F008 [EXTERNAL_EVIDENCE]** sklearn `f1_score`：`average='macro'` = 每类 F1 无权重平均（"does not take label imbalance into account"）；`zero_division` 默认 `"warn"`（等价 0）。含义：**测试集中出现但从未被预测的类，其 F1=0 且照样算入 29 类平均**；反之预测一个测试集中不存在的类只损害该类 precision。来源 https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html 。→ macro-F1 下「宁可为每个稀有类留几条预测」是正期望；accuracy 下这是纯亏损。这是 metric 之间真正**零和**的决策点。
- **R3-F009 [EXTERNAL_EVIDENCE]** 类增量学习中最后一层 FC 权重对新类系统性偏高：WA（Zhao et al., arXiv:1911.07053）明确指出 "an important factor causing catastrophic forgetting is that the weights in the last fully connected (FC) layer are highly biased"，并用权重对齐（新类权重范数缩放到旧类水平）修正；BiC（Wu et al., arXiv:1905.13260）把「新旧类数据不平衡」列为两大失败因素之一，用一个小的类平衡验证集学两参数偏置校正。→ 本题正是 16 类头扩到 29 类、新类样本 624 vs 旧类 296，且旧类中 Sheep/Keyboard 各 3 条，偏置方向可预测（新类 logit 偏高）。适配见 MC-3。
- **R3-F010 [EXTERNAL_EVIDENCE]** Logit adjustment（Menon et al., arXiv:2007.07314）：post-hoc 从 logit 中减 τ·log π_y（或训练时加入），"encourages a large relative margin between logits of rare versus dominant labels"。→ τ=0 对应最优 accuracy，τ=1 对应平衡误差/balanced accuracy 视角；τ 是把 accuracy 目标换成 macro-recall 目标的旋钮，**方向随官方 metric 反转**，因此必须先定 metric（MC-1）。
- **R3-F011 [INFERENCE，自算]** 统计分辨率：363 条测试上 accuracy 的 95% 二项 CI 半宽 = 0.050(@0.60) / 0.045(@0.75) / 0.031(@0.90)。→ **LB 上 <0.02 的差异对单次提交没有统计意义**，但同一测试集上的成对比较（McNemar / 配对 bootstrap）灵敏度远高于独立 CI，所以 LB A/B 的正确读法是「同模型只改一个后处理开关」，而不是比较两个全然不同的方案。本地侧：920 条做 5-fold，每折 val≈184 条，单折 accuracy CI 半宽 ≈0.06–0.07 → 单折数字几乎无信息，必须 repeated + 配对。
- **R3-F012 [INFERENCE]** 提交预算分配建议（上限 50）：2–3 次常量类探针（定 metric + 测 2 个类的真实测试计数）；1 次 baseline（模型直出）；2 次单变量后处理 A/B（时长约束、新类偏置 δ）；1 次多窗聚合 A/B；其余留给 seed/配方迭代与最后 2 次保守回退。**探针要在还有时间用其结论时花掉，最迟在剩余 60 分钟前完成。**
- **R3-F013 [LOCAL_FACT]** `split` 列全为 `train`，官方无 val；`submission.csv` 的 target 全为 0（纯模板，无任何标签信息）。→ 任何验证协议都必须自建，且 363 行顺序必须原样保留。

## 5. Method Cards

### MC-1 常量类 LB 探针（metric 识别 + 测试先验测量）
- 家族：主动实验设计 / LB 探测。
- 机制：提交「全部预测类 c」。accuracy → 得分 = n_c/363（可反解 n_c）；macro-F1 → 得分 = (1/29)·2n_c/(363+n_c)（例如 n_c=24 时 accuracy 得 0.066 而 macro-F1 得 0.0043，差 15 倍）；balanced accuracy/macro-recall → 得分恒 = 1/29 = 0.0345，与 c 无关。三者可一次性区分；用第二个类 c' 再确认（macro-recall 下得分不变，accuracy 下随 n_c' 变化）。
- 解决的问题：官方 metric 未知（[UNRESOLVED] #1），以及 F004 揭示的测试先验未知。
- 证据：F008（macro 定义）、F002/F004（先验不确定性）；纯算术，无需外部来源。
- 适配：提交只需读 `submission.csv` 模板并把 target 列写成常量，不需要 GPU/模型；符合 `path,target` 契约。
- 与其他方法的结构差异：这是唯一能**直接观测** metric 的手段；本地 CV 无论多好都无法回答。
- 最小验证实验：即刻提交 1 次 c=11；根据得分量级判定 metric 族；必要时第 2 次 c=25 量化昆虫先验。
- 成本：2–3 次提交额度，~1 分钟 kernel（若 kernel 必须跑训练，可写一个纯写 CSV 的最短 kernel）。
- Kaggle kernel 可行性：可行（离线、无依赖）。
- 失败模式：题面本就写明 metric（则不用花提交）；LB 只显示 public 子集（得分反解的 n_c 是 public 子集计数，仍足以判 metric 族）；得分四舍五入位数不足时 macro-F1 可能显示 0.00xxx 需注意有效位。
- 置信度：高（数学恒等式 + 官方 metric 定义）。
- 污染/规则风险：无（不涉及外部数据，也不是标签泄漏；仅利用自己的提交反馈，属正常 LB 使用）。

### MC-2 时长门控标签空间约束（metadata 受控后处理）
- 家族：约束解码 / 规则后处理。
- 机制：对 duration>5.02s 的测试样本，把 softmax argmax 限制在 {25,26,27,28}（等价给 0–24 的 logit 减 ∞ 或 −10）。可选软版本：对 5.00s 样本给 25–28 减 δ_ins。
- 解决的问题：昆虫类是最不像 AST 预训练域的一族（超声、变长），模型最易混淆；该规则免掉 40 条测试样本的 25 类干扰。
- 证据：F003（238/238 精度、239/240 recall 的本地统计，源自 `archive/*.csv` + WAV header）。
- 适配：只依赖 WAV header（可用 `wave`/soundfile 读 duration），零算力；与任何模型正交。
- 结构差异：与「把时长当特征喂模型」不同，这是硬约束，不需要模型学会且不会被小样本噪声稀释。
- 最小验证：本地对 labeled 全量算 precision/recall（已做）；LB 上同模型 ±约束各 1 次提交。
- 成本：~0；1 次提交额度做 A/B。
- Kaggle kernel 可行性：可行。
- 失败模式：若组织者对测试昆虫做了 5s 裁剪（测试中确实有 1 条 11025Hz@5.0s），长文件里可能混入非昆虫类（例如某些新环境类的原始长录音）→ 硬约束会把它们全判错，最坏损失 40/363=11% accuracy。缓解：先只对 dur>8s（本地昆虫占比更纯、测试 34 条）施加，或用软偏置 δ=3 而非 ∞。
- 置信度：中高（本地统计极强，但测试集处理方式未知，必须 LB 验证）。
- 污染/规则风险：无（使用的是题目自带文件属性，非外部数据）。

### MC-3 argmax 改变型校准：per-class logit bias（prior 调整 + 新旧类偏置 + 计数匹配）
- 家族：post-hoc 校准 / 长尾 logit adjustment / 类增量 bias correction。
- 机制：最终 logit z'_y = z_y − τ·log π_y − δ·1[y≥16]（可加 WA 式：把新类分类头权重范数缩放到旧类均值）。再可选做**计数匹配**：搜索 b 使预测类频率接近目标先验（目标先验由 MC-1 探针 + F004 估计），类似 prior correction / Sinkhorn 式配额。
- 解决的问题：(a) 新类样本多 2 倍 + 新类头随机初始化后过训 → 新类 logit 系统性偏高，旧稀有类被吞（F009）；(b) metric 若为 macro-F1/balanced accuracy 则需为稀有类主动腾出预测（F008/F010）。
- 证据：arXiv:1911.07053（FC 权重对新类高度偏置；权重对齐修正）、arXiv:1905.13260（不平衡 + 小平衡验证集学偏置）、arXiv:2007.07314（post-hoc τ·log π 调整）。
- 适配：全部在推理端 numpy 层面完成，不需重训；τ、δ 用本地 held-out（含旧类稀有类）扫描 + LB 单变量 A/B 确认。**重要：温度缩放 T 对硬标签 metric 完全无效（argmax(z/T)=argmax(z)），不要花时间。**
- 结构差异：与 R4 轴的「训练时防遗忘」互补且更便宜；τ 与 δ 是两个不同方向的偏置（长尾 vs 新旧组），必须分别调，不要合成一个旋钮。
- 最小验证：固定一个已训模型，本地在 repeated holdout 上画 accuracy(τ,δ) 与 macroF1(τ,δ) 两张表；若两者最优点冲突（预期会冲突），以 MC-1 确定的 metric 为准；LB 各 1 次提交确认 δ 的符号。
- 成本：分钟级 CPU；2 次提交额度。
- Kaggle kernel 可行性：可行。
- 失败模式：用 labeled 先验做 τ 调整时先验本身错（F004）→ 反向伤害；δ 过大把新类全部压掉；在 3 样本类上 held-out 估计方差极大导致选到噪声最优点。
- 置信度：中高（机制与证据明确，量级需本地测）。
- 污染/规则风险：无。

### MC-4 embedding 近重复检测 → group-aware 分折
- 家族：泄漏控制 / 数据去重。
- 机制：用题目自带 AST checkpoint 提取每文件（长文件多窗平均）CLS embedding，算余弦相似度，取阈值建图的连通分量作为 group；分折时整个分量同折。阈值用正对照校准：昆虫类每类 60 条大概率来自少数长录音（F003 时长分布 3–120s），若阈值正确应出现明显的紧密簇；同时用 24000Hz 子集（仅 16–24，F005）作为「同数据源但不同录音」的负对照。
- 解决的问题：ESC-50 家族的同源片段近重复会让随机 CV 系统性乐观（F007），而哈希文件名删除了 fold/clip/take 信息（F001）。
- 证据：F007（官方 README 明确同源片段必须同折）；F006（我的负面实验说明必须用 embedding 而非粗指纹）。
- 适配：只用题目提供的 checkpoint，一次前向 1283 条（GPU 上分钟级），产物是折分配表，不进入最终推理路径。
- 结构差异：与「随机 stratified k-fold」的差别是唯一能解释 LB/CV 反号的结构性原因之一。
- 最小验证：算相似度分布直方图 + 连通分量 size 分布；若 size>1 的分量覆盖 >10% 样本，说明泄漏真实存在，随机 CV 的绝对值不可信（但排序可能仍可用）；若几乎全是 size=1，则可放心用 repeated stratified split，把精力转回 MC-1/MC-3。
- 成本：一次全量前向（~1283×(1–3) 窗），几分钟 GPU；无额外依赖。
- Kaggle kernel 可行性：可行（仅在开发/验证阶段使用；不影响提交 kernel 时限）。
- 失败模式：AST embedding 对「同录音不同片段」的相似度可能与「同类不同录音」重叠 → 阈值无法分离（正是 F006 在粗指纹上发生的情况）；此时应放弃精确分组，改为「保守假设 CV 乐观、只信 LB 与配对比较」。
- 置信度：中（机制标准，但本题能否分离未验证）。
- 污染/规则风险：无。

### MC-5 验证协议与「本地改进是否可信」的判据
- 家族：实验设计 / 统计判据。
- 机制：(1) repeated stratified 5-fold ×3（不同 seed），稀有类手工保证每折 val 至多 1 条、且 3 条类在 3 次重复里各出现一次；(2) 报三个数：overall accuracy、旧 16 类 macro-recall、新 13 类 macro-recall（外加 macro-F1 备用），因为遗忘只在旧类子指标上可见；(3) 决策：同一折集合上的配对 bootstrap（对样本重采样，统计 Δ 的 95% CI），CI 含 0 则拒绝该改动；(4) 单折/单 seed 差异 <0.02 一律视为噪声（F011）。
- 解决的问题：<1000 样本、最小类 3 条时局部分数噪声与 LB 弱相关（系统背景中的 LB +0.096 / CV −0.127 反号案例）。
- 证据：F011（自算 CI）、F002（稀有类杠杆）、F007（组泄漏使绝对值乐观）。
- 适配：所有折都在 920 条 labeled 上；推理侧窗口策略必须与训练一致，否则 CV 与 LB 的差异来源不可归因。
- 结构差异：相比「单次 5-fold 比 mean 分数」，这里强调**配对**与**分组子指标**，把噪声从判据中剔除。
- 最小验证：先跑一次 baseline 的 3×5-fold，记录三个子指标的 seed 间标准差，作为后续所有比较的噪声底线。
- 成本：3× 训练时间（AST 920 条 5s，单 GPU 每 epoch 数十秒 → 可接受）。
- Kaggle kernel 可行性：验证阶段用；最终提交 kernel 只需一次训练（[可行]）。
- 失败模式：若组泄漏严重（MC-4 检出），CV 绝对值仍乐观，只能用于排序；若 metric 未定，选模型时可能优化错目标。
- 置信度：高。
- 污染/规则风险：无。

### MC-6 LB 与本地 CV 冲突时的决策规则
- 家族：决策规则（非算法）。
- 机制：把改动分成两类：**(A) 结构性/低风险**（多窗聚合、group-aware 分折、时长约束、修 bug）——LB 与 CV 冲突时**信 LB**，因为本地 CV 在近重复与小样本下有已知的系统性偏差方向（乐观），而 LB 是无偏但方差大的估计；**(B) 高自由度/易过拟合 LB**（逐类阈值手调、按 LB 反推每类配额、只在 1–2 条样本上翻转）——冲突时**都不采用**，除非 LB 增益 >0.03（超过 F011 的噪声尺度）。任何单变量 A/B 都必须与 baseline 只差一处。
- 解决的问题：50 次提交额度下如何避免 LB 过拟合，同时不被虚假本地信号误导。
- 证据：F011（363 条的噪声尺度）、F007（CV 偏差有方向）、系统背景中的反号案例。
- 适配：把提交日志表格化（改动、LB、本地三子指标），最后按「结构性改动优先、LB 单调改进链」选最终提交。
- 结构差异：这是元规则，不占算力。
- 最小验证：不适用。
- 成本：0。
- Kaggle kernel 可行性：不适用（[可行]）。
- 失败模式：public/private 子集划分导致 public LB 本身偏差；对策是最终提交避免依赖仅 LB 支持的、增益 <0.03 的脆弱后处理。
- 置信度：中高。

## 6. 负面结果、失败条件与冲突证据

1. **粗频谱指纹不能做近重复检测（我的实测，F006）**：LOO 1-NN 0.363、>0.95 相似对同标签率 0.257。不要因为「相似度高」就删样本或建组，会引入错误分组。
2. **温度缩放对本题 metric 无效**：accuracy/macro-F1/balanced accuracy 都只看 argmax，单一温度不改变 argmax。任何「校准」时间应全部投给 per-class bias。
3. **采样率捷径弱于 ASSET_MAP 的描述（F005）**：≥192kHz 在测试集只有 2 条；把「SR>100k ⇒ 昆虫」当规则几乎无收益，却容易误判 96000Hz 的旧类/新环境类样本。
4. **labeled 先验 ≠ 测试先验（F004）**：昆虫在 labeled 占 26%、在测试推测占 ~11%。因此 balanced-softmax / 反频率加权若以 labeled 频率为准，方向可能是错的（会进一步抬高本已过多预测的昆虫或反之）。冲突未解，需探针。
5. **ESC-50 类计数不匹配（F007 边界）**：Thunderstorm 62、Frog/Bird 35 与 ESC-50 的 40/类不符 → 数据已重组或混入其他来源，任何公开 fold/清单都不可复用（也不允许引入）。
6. **macro-F1 与 accuracy 的零和点是真实存在的（F008/F010）**：为 Sheep/Keyboard（期望测试 1–2 条）保留预测在 macro-F1 下期望正、在 accuracy 下期望负。不确定 metric 时的折中（τ≈0.5）在两个 metric 下都不是最优，只是最小遗憾。

## 7. 未解决问题

1. 官方 metric（accuracy / macro-F1 / balanced accuracy）——MC-1 可用 1–2 次提交解决；优先看题面。
2. 测试集是否包含全部 29 类、public/private 划分比例与是否分层。
3. 测试昆虫样本是否被裁剪到 5s（决定 MC-2 硬约束的安全性）；本地仅见 1 条 11025Hz@5.0s 的可疑证据。
4. AST embedding 能否分离「同源片段」与「同类不同录音」（MC-4 的正对照未跑）。
5. 新类 logit 偏置 δ 的实际量级（需一次真实微调后测量，落在 R4/实施阶段）。
6. kernel 时限与 GPU 型号，影响 repeated-CV 能否放进提交 kernel（建议提交 kernel 只做单次训练，验证在本地/开发 kernel 做）。

## 8. 给题目分析 Agent 的合并建议

- 把「时长≠5.00s ⇒ 昆虫（labeled 精度 100%、召回 99.2%）」升级为 TASK_ANALYSIS 的一级事实（现文档仅作为「捷径/陷阱」提及），并把「SR>100kHz ⇒ 昆虫」降级为弱信号（仅 ≥192kHz 单向，测试 2 条）。
- 新增一条 [INFERENCE]：测试类先验与 labeled 先验不一致（昆虫 ~11% vs 26%），因此所有基于 labeled 频率的重加权都需 LB 验证。
- 在验证章节写死判据：LB 差 <0.02、本地单折差 <0.02 均视为噪声；只接受配对 bootstrap CI 不含 0 的本地改进。
- 在提交预算章节写死顺序：常量类探针（定 metric+先验） → baseline → 单变量后处理 A/B（时长约束、新类偏置、多窗聚合） → 配方迭代 → 保守回退。
- 记录 R3 的负面结果：粗频谱指纹不可用于近重复检测；温度缩放对硬标签 metric 无效。

## 9. 来源表

| 标题 | URL | 日期/版本 | 类型 | 支持 Finding |
|---|---|---|---|---|
| ESC-50 官方仓库 README（fold 规则、数据来源 Freesound、5s/50 类） | https://raw.githubusercontent.com/karolpiczak/ESC-50/master/README.md | master，2026-08-04 取 | 官方数据集仓库 | R3-F007 |
| sklearn `f1_score` 官方文档（macro 定义、zero_division、labels） | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html | stable（2026-08-04 取） | 官方文档 | R3-F008 |
| Maintaining Discrimination and Fairness in Class Incremental Learning (WA) | http://arxiv.org/abs/1911.07053 | v1, 2019-11 | 原始论文 | R3-F009 |
| Large Scale Incremental Learning (BiC) | http://arxiv.org/abs/1905.13260 | 2019-05 | 原始论文 | R3-F009 |
| Long-tail learning via logit adjustment | http://arxiv.org/abs/2007.07314 | 2020-07 | 原始论文 | R3-F010 |
| 本地资产 `archive/train.csv`,`archive/fine_tune.csv`,`archive/submission.csv` + 1283 个 WAV header（我用独立脚本重算：类计数、时长、采样率交叉表、指纹实验、二项 CI） | `SEARCH_BUNDLE/ORIGINAL_ASSETS/archive/` | 本次运行 | 一级本地证据 | R3-F001–F006, F011–F013 |
