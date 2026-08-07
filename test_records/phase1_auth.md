# 阶段 1 · 认证真通  2026-08-04 10:16:52 PDT
## OpenRouter (gpt-5.6-sol, 1 token)
  model: openai/gpt-5.6-sol | choice: pong | err: None
## Max profiles (claude -p, 各 1 次)
  integrated: Failed to authenticate: OAuth session expired and could not be refreshed
  direct: Failed to authenticate: OAuth session expired and could not be refreshed
## Kaggle
  Submissions today: 1
  Lifetime submissions: 4
  Remaining today: 99

## 事故记录:凭证拷贝导致全部 OAuth 失效
- 时间线: 昨日将 ~/.claude/.credentials.json 拷入两个 profile 目录 → 主 profile
  在 run16 中正常刷新 → 今日两个副本先后用旧 refresh token 尝试刷新 →
  token 轮换链条被抢占 → 主 profile 的 token 一并作废。
- 结论 1: refresh token 是轮换制,**同一账号的凭证文件不可复制多份**,
  必须 symlink 到同一目录(已改)或使用真正独立的第二账号。
- 结论 2: 这正面验证了 JINGYUAN 双账号设计的动机——单账号不仅有 usage-limit
  单点,连凭证生命周期都是单点。
- 待办: 需要人工重新 claude auth login(浏览器 OAuth,agent 不能代办)。

## 重新登录后复验 10:21:47
  integrated: pong
  direct: pong

## 修复后复验 13:40:50
  integrated: pong
  direct: pong
OK  assets
OK  claude_binary
OK  claude_integrated_profile
OK  claude_direct_profile
OK  codex_binary
OK  search_prompt
OK  openrouter_key_env
OK  claude_integrated_max_login
OK  claude_direct_max_login
NOTE  doctor only checks local wiring; it does not spend model tokens or submit.
