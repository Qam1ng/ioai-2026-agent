# task-5 t5run1 结转（写于 t5run1 结束，供 t5run2 继承）

## 已验证的榜分（基线 28.60）

| 榜分 | 本地 | 残差 | 路线 |
|---|---|---|---|
| 94.08 | 95.16 | -1.1 | hearsay-solver_c-88c2f7eab3-88c2f7eab3 |
| 93.79 | 94.67 | -0.9 | claude-claude-pairft-v3-f494c515eb |
| 93.5 | 94.61 | -1.1 | codex-tfidf-seam-v1-fcd84e95ac |
| 93.1 | 96.16 | -3.1 | codex-stacked-seam-v2-639d8e97d1 |
| 91.43 | 95.51 | -4.1 | hearsay-solver_a-e6bba80eb7-e6bba80eb7 |
| 85.4 | 88.52 | -3.1 | claude-claude-sentlr-v1-d8112039de |
| 72.68 | 45.29 | +27.4 | hearsay-solver_b-20858342a9-20858342a9 |
| 41.52 | 96.48 | -55.0 | hearsay-solver_a-4c2b892d1d-4c2b892d1d |

## 硬教训

1. **残差按路线分层，不是按本地分分层。** TF-IDF/seam 路线残差 -1~-4 且稳定；bge 路线出过 -55 的塌方。本地分高不等于可交。
2. **kernel 里必须有准入门禁。** 模型段只有通过健全性检查（log-odds sd 不接近 0、pos_rate 合理）才允许覆盖上一段的 submission.csv。
3. **41.52 的机制**：broker 注入 `IOAI_BUDGET_S=360`，而脚本守卫要求 `left() > 400`，模型段被静默跳过，只剩位置先验兜底。**写 kernel 时守卫阈值必须低于 300**，否则永远不执行。
4. **self-check 做法**：扣掉最后 200 篇标注做 holdout，两段都在其上用官方 metric 自测后再决定是否覆盖。
5. 官方 metric = `100 * mean(exp(-|pred-true|/100))`，已用基线 28.60 交叉核验。

## t5run1 原始事实摘录

- **[adoption] ?**: solver_a 采纳 solver_c 三条并已落地验证：(1) 按 sample_submission.csv 的 id 顺序 dict 查表写出，缺失填中位数+WARNING；(2) 每段打 SELFCHECK 三行；(3) dry-run 改成 exec 出货文件本身而非副本。强制 T4 fp16 路径实测：kernel 输出 id 顺序==sample_submission，与 out/submission.csv 逐行相同率 0.9895，frac mean 0.5950 vs train 0.5954，与保底一致率 0.324。
- **[adoption] ?**: solver_b 的 40.43 机制(fact-560edb87c6be44ee)暴露了我闸门的缺口，已修：我原来的准入条件是「模型分>保底分」，而保底在我的 holdout 上是 39.42，log-odds 塌缩后是 ~40——差1分就放行了。改成两道：holdout 边际必须>20（健康值 95.26 vs 39.42，边际56），且 test log-odds 必须有限、sd>1.0、pos_rate∈[0.10,0.75]（我的健康值 sd=6.423 pos_rate=0.330，与 solver_b 的 6.29/0.357 一致）。"赢过保底"不是分类器有输出的证据。
- **[env] ?**: 全队指令回报（solver_b）：把 shipped kernel 按出厂设置(EPOCHS=2,SEEDS=[0,1])在模拟挂载下端到端跑完，与本地 out/submission.csv **逐行完全相同率 99.47%**，id 顺序一致，代理一致度 99.645。三段在 t=1s/16s/83s/144s 触发，SELFCHECK 三次全 PASS(lo_sd 5.31/6.57/6.53, pos_rate .352/.360/.358)。A6000 上 144s → T4 约10分钟。我这条线不存在 kernel产出≠本地产出。
- **[adoption] ?**: solver_b 采纳两条：(1) fact-8930adc071ad3af4 —— 改成按 sample_submission.csv 的 id 顺序 dict 查表写出，缺失填中位数+WARNING、多余丢弃+WARNING；取不到该文件则回退 test_public 顺序并打印，不引入新故障点。(2) fact-4c8f7829597d0c8f —— 把我准入门禁的 lo_sd 阈值从 0.5 收到 solver_c 收敛的 1.0（健康值 6.29~6.57，仍极宽松）。正在对出货文件本身做出厂设置 dry-run 复核。
- **[env] ?**: 全队指令回报（solver_c）：shipped kernel 按出厂设置端到端跑完，与本地 out/submission.csv 逐行相同率 100.0%（760/760，id 顺序取自 sample_submission.csv）。两道门全 PASS：holdout(200) model=95.26 vs fallback=39.42（边际56）、test log-odds sd=6.423 pos_rate=0.330 finite=True。纯 CPU 37 秒，无 GPU 依赖、无训练截断失效模式。这条线不存在 kernel产出≠本地产出。
- **[failure] ?**: solver_c 死路索引（CPU 词法侧已到顶，全部同 folds 纯重打分或轻量重训，别再试）：句长加权 lo(两向单调劣)、拼接前一句上下文(w0.25→94.78 单调劣)、篇内 z-score 风格特征(w0.15→95.14 单调劣)、字符距离 decay(最好91.2)、非对称 decay(噪声)、硬窗口W(需W≥12才追平)、Bayes 期望损失解码(与argmax同分)、bge 句向量混合(+0.04)、bge 微调+融合(fold0 95.56<95.67)、frac 窗口收紧(带先验时完全 no-op)。基线 95.158。
- **[env] ?**: solver_b 收尾状态：候选冻结在共享 folds CV 96.2483（bge 单句分类 + 局部衰减 changepoint + 高斯位置先验）。闸门全绿：oof 1601 对齐、submission 760 行且 id 顺序==sample_submission、kernel 无占位符、t4/12min。三段递进 + 准入门禁，失败地板 95.30（CPU TF-IDF 段），出厂设置 dry-run 与本地逐行相同率 99.47%。若选 GPU 路线取本候选，若按 fact-0cf0f6ff9cfa65ea 只信 CPU 残差记录则取 solver_c 的，两者不冲突。
- **[env] ?**: solver_c 收尾状态：冻结在共享 folds OOF 95.158（句级 TF-IDF changepoint + decay0.18 解码 + KDE 先验 pw4.5/bw0.045）。oof 1601 对齐、submission 760 行且 id 顺序==sample_submission、其内容就是 kernel 产物本身、kernel 无占位符、cpu/37 秒。两门 PASS：holdout(200) 95.26 vs 保底 39.42、log-odds sd6.423/pos_rate0.330。同意 solver_b 的分工表述。
- **[failure] ?**: 最终结论（solver_c，供后续接手者）：CPU 纯词法路线在本题的天花板是 OOF 95.2。同一解码器下 bge 微调可到 96.2-96.5，差距约 1.1 分全部来自句分类器本身，与解码/先验/特征工程无关——我在解码与特征侧穷尽的十条路线全为负。若时间预算允许 GPU，直接上编码器；若不允许，95.2 就是终点，不要在词法特征上继续投入。
- **[failure] ?**: solver_b 死路索引（bge 微调侧，同 folds，基线 96.2483，别再试）：pointer network 每篇仅1个监督单元，加码两次(3→8ep,2e-5→5e-5)只从62.5到79.05；KDE 先验 18 格中 8 格低于单高斯，bw 无单调关系，跨度0.24分≈2条样本，是噪声；frac 窗口四档收紧逐位相同 96.2483，带先验时 argmax 从不落窗外；prior_w=8 回落到96.04(w=5 是96.25)。有效的只有 decay(+0.5~0.8) 和高斯先验(+0.2~0.4)，各档位互补。
- **[adoption] ?**: solver_a 采纳 log-odds 准入探针（阈值取 solver_c 收敛的 sd>1.0，pos_rate∈[0.10,0.75]，加有限性）。我的闸门基线是 TF-IDF 的 94.28 而非保底 39.42，塌缩的 bge≈40 本就被以 54 分边际拒绝，故 solver_c 的「差1分放行」缺口不适用于我；探针作为第二道独立检测加在 holdout 之后。实测 shipped test probs: lo_sd=5.942 pos_rate=0.358 PASS；模拟未训练 head sd=0.050 REJECT。语法已校验，无占位符。
- **[result] ?**: 提交 codex-stacked-seam-v2-639d8e97d1（codex 路线，final）榜分 93.1；本地 96.1609387，残差 -3.0609。
- **[result] ?**: 提交 claude-claude-pairft-v3-f494c515eb（claude 路线，final）榜分 93.79；本地 94.67211014，残差 -0.8821。
- **[result] ?**: 提交 hearsay-solver_c-88c2f7eab3-88c2f7eab3（hearsay 路线，final）榜分 94.08；本地 95.15685077，残差 -1.0769。
