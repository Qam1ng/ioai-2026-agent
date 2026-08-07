# 阶段 2 · 30 分钟彩排(无 --live)  2026-08-04 13:43:41 PDT

启动: --duration-minutes 30 --competition-mode practice --run-id rehearsal1
预期窗口切分: 探索止于 T-13.5min (min(120, 30*0.45)), 只提交期 T-6min (min(20, 30*0.2))

## t+0~1min 事件
- official_assets_ready  资产已快照 + SHA-256
- calibration_updated    calibration inactive (feedback_count=0, 预期)
- search_finished        **t+3s 即结束**
- search_handoff         status=raw_assets_fallback, bundle 为空
- hearsay_started        pid 1989825

## 发现 1:Search 线 stage A 挂了,HearSay 线正常
- stage_a_finished: exit_code=1, duration=1.4s, cost=0
  result_text: "Not logged in · Please run /login"
- 同一时刻 HearSay 线的 3 solver + evaluator 全部正常出 token,metric/folds 已冻结。
- config.py:120 用 .expanduser() 在加载期解析,路径是绝对的,不是 ~ 展开问题。
- 判断:**并发 OAuth 刷新竞态**。symlink 让 6-7 个 Claude 进程共用一份
  .credentials.json;refresh token 是轮换制,先刷新者使其余进程手中的副本失效。
- 与昨天"拷贝副本"是同一个根因的不同形态:问题不在拷贝或 symlink,
  而在**一份订阅凭证被多进程并发刷新**。
- 印证:JINGYUAN 配置注释里坚持双账号,不只是为了 usage limit。
  单账号在这个架构下连认证都不稳。

## 更正:发现 1 的判断是错的
"并发 OAuth 竞态"不成立。claude 直跑线 12 轮全部 "Not logged in",均匀分布在
30 分钟里,而非只在 fan-out 瞬间;同期 hearsay 线全程正常。

## 发现 1(更正后):`--bare` 使 Claude Code 跳过凭证读取
ClaudeSubscriptionRunner.argv() 传了 `--bare`。CLI 2.1.220 的说明:
  --bare  Minimal mode: skip hooks, LSP, plugin sync, attribution,
          auto-memory, background prefetches, **keychain reads**, ...
二分验证(同一 profile、同一沙箱 env,只差这一个 flag):
  带 --bare   : Not logged in
  不带 --bare : is_error=false
影响: Search 线(1 analyst + 4 research)与 claude 直跑线全部失效 = 图上三条
路线里有两条从未真正跑起来。hearsay 线不受影响,因为它经 claude_agent_sdk
启动,不带 --bare。
性质: 一行修复(去掉 --bare,或换 --settings 显式指向凭证)。可能在她的
macOS + homebrew CLI 版本上 --bare 语义不同,所以本地没暴露。

## 2. Broker:正确地"准备好但不发"
4 次 broker_submission,全部 result.detail = "dry-run ..."、consumed=True。
Kaggle 权威核对:Submissions today 仍为 1(我手动那次),Remaining 99。
→ 无 --live 时不写 Kaggle,行为正确。
候选与本地分:
  hearsay-solver_b 0.87293 → solver_a 0.89503 → solver_b 0.90608(单调上升)
  codex-classical-rf-v2 0.66298
→ Broker 在跨路线的统一分数上选择,codex 线分数明显更低但仍获得一次名额
  (anti_monopoly_fraction=0.50 生效,避免单路线垄断)。

## 3. 公共 metric.py + folds.json 重评分:发生了
evaluation_contract_frozen @ +150s
  source=hearsay_evaluator  unit_count=920  fold_count=1
  metric_sha256=a83c8fc7...  folds_sha256=cdcc7575...  contract_sha256=d4b09ee1...
shared/evaluation/{metric.py,folds.json,contract.json} 已生成,三条路线共用。
注意: fold_count=1 —— 30 分钟窗口下 evaluator 只切了 1 折,非 5 折。

## 1. 候选产出
  codex   : outbox=5, READY=5
  hearsay : outbox=0 (候选直接进 registry,不走 outbox)
  claude  : outbox=0 (整条线因 --bare 未启动)

## 决策:Claude 路线改走 API,不动 --bare
实测:API key + --bare → is_error=false。--bare 只影响 keychain(订阅凭证),
环境变量注入的 API key 不受影响。
改动(只加不删,已备份 *.orig):
  final_system/runners.py     ClaudeSubscriptionRunner.environment()
  final_system/controller.py  _start_hearsay()
两处都改为:若 os.environ 有 ANTHROPIC_API_KEY 则透传进沙箱 env;没有则
行为完全不变(仍走 CLAUDE_CONFIG_DIR 订阅)。她的 macOS 订阅流程不受影响。
副作用:三条 Claude 路线统一走 API 计费,不再受 Max usage limit 单点影响。
