# IOAI Search Agent System 使用说明

这个模块只执行：题目分析阶段 A → 0–4 路并行网络研究 → 恢复同一个题目分析会话执行阶段 B → 生成 Search Bundle。它不训练模型、不生成 submission，也不调用 Kaggle。

运行时直接读取 [`SEARCH_PROMPTS_ZH.md`](SEARCH_PROMPTS_ZH.md) 中的四类代码块，因此审计文档与实际 Prompt 不会维护两份。阶段 A 必须输出机器可解析的 `enabled_research_ids`；Runner 只启动这些研究路，并按 R1、R2、R3、R4 的后端映射选择 Claude Code 或 Codex。

## 运行

全 Claude Code：

```bash
python -m ioai_agent_system.search_cli \
  --assets-dir /absolute/path/to/task_assets \
  --prior-lessons-dir /absolute/path/to/prior_lessons \
  --output-root /absolute/path/to/search_runs \
  --duration-seconds 1500 \
  --competition-mode formal \
  --analyst-backend claude \
  --research-backends claude,claude,claude,claude \
  --claude-model claude-opus-5 \
  --claude-effort high
```

Claude/Codex 混合研究：

```bash
python -m ioai_agent_system.search_cli \
  --assets-dir /absolute/path/to/task_assets \
  --output-root /absolute/path/to/search_runs \
  --duration-seconds 1500 \
  --competition-mode formal \
  --analyst-backend claude \
  --research-backends codex,claude,codex,claude \
  --claude-model claude-opus-5 \
  --codex-model gpt-5.6-codex
```

`--duration-seconds` 必填，没有 30 分钟或正式赛时长的隐藏默认值。使用 Claude 时，程序会在终端中隐藏输入 Anthropic key；也可以用 `--api-key-stdin` 从调用方的安全管道读取。key 只通过临时 Unix socket 提供给 Claude CLI，不进入 Prompt、Markdown、状态文件或 trajectory。Codex 使用本机现有登录态，并启用原生 `--search`。

## 输出

每次运行生成独立目录：

```text
search-<UTC>-<id>/
  RUN_STATUS.json
  controller_trajectory.jsonl
  prompts/
    PROMPT_SPEC.md
    analyst_stage_a.md
    research_R*.md
    analyst_stage_b.md
  trajectories/
    analyst_stage_a.jsonl
    research_R*.jsonl
    analyst_stage_b.jsonl
    *.stderr.log
  work/analyst/SEARCH_BUNDLE/
    SEARCH_OUTPUT.md
    ASSET_MAP.md
    TASK_ANALYSIS.md
    RESEARCH_SYNTHESIS.md
    SHA256SUMS
    ASSET_MANIFEST.json
    ORIGINAL_ASSETS/
    stage_a/
    research_raw/
```

原始资产先复制为逐字节快照，再向 Agent 暴露；源目录路径不会写入 Bundle。Runner 在阶段 B 后同时复核源目录、`ORIGINAL_ASSETS/`、阶段 A 文档和原始研究报告。凭证文件、明显密钥、符号链接和特殊文件会在 Agent 启动前被拒绝。

`RUN_STATUS.json` 的状态含义：

- `complete`：资产复核、同会话恢复、计划内报告、所有核心文件和 Agent 退出状态全部通过。
- `degraded`：Bundle 可读，但发生 fallback、超长/缺失研究材料或某个 Agent 未干净退出；不会伪装为完整成功。
- `failed`：核心阶段、受保护输入或资产完整性失败。
- `timeout`：达到调用方给定的 Search 模块硬截止时间。

阶段 A 和 B 的“同一个 Agent”通过真实 Claude/Codex session ID 恢复；Runner 会检查恢复后的 ID，失败时不会新建总结 Agent 来冒充连续会话。
