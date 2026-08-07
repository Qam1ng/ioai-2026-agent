Kernel-only Kaggle 题目：推送前一定要在本地把 kernel script 端到端跑通一次，代价只要几分钟，能挡住整轮归零的风险。做法：把 `setup_ioai_env()` 那一行 sed 成 `pass` 生成临时副本，脚本里所有 `/kaggle/input`、`/kaggle/working` 都改成读环境变量（默认仍是 Kaggle 路径），epoch/seed 数也用环境变量，然后用 EPOCHS=1 SEEDS=1 跑一遍。

检查两件事：(1) 全链路能写出 submission；(2) kernel 内自带的特征/推理路径产出的预测与本地已计分候选高度一致（我这次探针阶段一致率 97.5%），这才证明「kernel 里的特征 == 本地验证过的特征」，否则本地 CV 再高也可能在云端跑出另一个模型。

另外两条经过验证的 kernel 稳健写法：
- 读到 sample submission 的第一时间就原样写出一份，之后每完成一个阶段（探针→每个 seed）就覆盖一次，任何超时/OOM 都还留着上一阶段的合法结果。
- 墙钟守卫要按「下一段工作的规模」换算，不能直接用上一段的实测耗时：多配方训练里每个 seed 的 epoch 数不同，必须乘 epoch 比例，否则会启动一个注定跑不完的阶段。
