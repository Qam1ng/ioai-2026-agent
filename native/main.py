#!/usr/bin/env python3
"""HearSay launcher — the human's only action.

    python -m native.main --slug <competition> [--solvers 3] [--deadline-min 120]

N independent, full-power Claude Code sessions solve the same problem from
different angles. They do not report to a controller and cannot see each other's
scores. Coordination is one append-only facts board carrying objective
discoveries only, and the truth about who is winning is computed by scripts
after the fact.

The bet: a tuned agent kernel left intact, inside a hard deterministic harness,
beats a constrained kernel inside a smart soft one. What that means concretely:

  * preset system prompt is KEPT and appended to, never replaced
  * setting_sources left at the default so CLAUDE.md and settings load
  * the built-in toolset is left alone, so WebSearch et al. are present
  * budget, submission count, validation split, scoring and promotion are
    enforced by the harness, which the model cannot talk its way past

See docs/DESIGN-HEARSAY.md.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import anyio
from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions,
                             ClaudeSDKClient, HookMatcher, ResultMessage,
                             TextBlock, ToolUseBlock)

from agent.orchestrator import Budget, Tracer
from agent.providers.anthropic_p import _load_dotenv
from agent.tools import registry as R

from . import facts
from . import evidence
from . import supervise as S
from . import tools as T
from .prompts import BOARD_MULTI, BOARD_SOLO, CONTINUE, CONTRACT, FIRST

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "claude-opus-5"
ALL_SOLVERS: list[str] = []
# Agents whose session stopped responding and could not be interrupted.
HUNG: set[str] = set()


def mode_contract(args) -> str:
    if args.mode == "clean-benchmark":
        return (
            "# 证据模式：CLEAN BENCHMARK\n\n"
            "所有 solver 都看不到榜单。不得调用 Kaggle submissions/leaderboard "
            "接口，也不得搜索网络；任何尝试读取审计集的行为都会永久污染该路线"
            "及其全部候选证据。本地结果只能按冻结运行策略披露。harness 即使"
            "观察到公榜分数，也只会在你的上下文之外留存，绝不能将它变成本地"
            "搜索信号。"
        )
    return (
        "# 证据模式：COMPETITION\n\n"
        "冻结候选提交后，harness 可能稀疏报告一次榜单校准。它只能用于诊断，"
        "绝不能被理解为可以围绕公榜连续微调一族相近候选。"
    )

REVIVAL = """
你的上一段 session 在整轮运行结束前中断了，原因可能是卡死或崩溃。现在是新
session。工作目录和事实板仍然保留，但先前推理上下文已经丢失；请读取证据，
不要凭假设续接。

`{cwd}` 中已有：
{artifacts}

你此前在事实板发布过以下内容，这是上一段路线意图的唯一记录：
{mine}

从现有 artifact 继续。如果 `out/oof.npy` 已经表现良好，先保护它；重新推导
已有成果会让这次重启变成净损失。剩余 {left:.0f} 分钟。
"""


def revival_context(ws: Path, solver: str) -> tuple[str, str]:
    out = ws / solver / "out"
    files = []
    if out.exists():
        for f in sorted(out.rglob("*")):
            if f.is_file():
                files.append(f"  {f.relative_to(ws / solver)}  "
                             f"{f.stat().st_size:,} bytes")
    mine = [f"  {r['kind']}: {r['text']}" for r in facts.board()._all()
            if r.get("src") == solver]
    return ("\n".join(files) or "  (nothing yet)",
            "\n".join(mine) or "  (nothing yet)")
# Physical GPUs this run may touch, from the launcher's own environment.
ALLOWED_GPUS = {x.strip() for x in
                os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()}


def reap_workspace_processes(ws: Path) -> int:
    """Kill anything still running out of the workspace.

    Solvers background their training with nohup/&, which survives the agent
    session that started it: run 1's watcher submitted twice more eleven
    minutes after the harness printed "no submission was made", and run 2 left
    six training processes holding GPUs after every agent was gone. Killing the
    agents is not enough — their detached descendants have to be swept too.
    """
    killed = 0
    root = str(ws.resolve())
    if not Path("/proc").is_dir():
        return 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            if not os.path.realpath(f"/proc/{pid}/cwd").startswith(root):
                continue
            os.kill(int(pid), signal.SIGKILL)
            killed += 1
        except (OSError, PermissionError):
            continue
    return killed


def budget_line(b: Budget) -> str:
    return (f"[budget] elapsed={b.elapsed() / 60:.0f}min "
            f"remaining={max(b.remaining(), 0) / 60:.0f}min "
            f"submissions={b.submissions}/{b.max_submissions} "
            f"cost=${b.cost_usd:.2f}/{b.max_cost_usd:.0f}")


def opts(solver: str, ws: Path, args, budget: Budget, trace: Tracer,
         append: str, cost_cap: float) -> ClaudeAgentOptions:
    # The CLI refuses to start if cwd is missing, and it fails as a connection
    # error rather than anything that mentions directories. Create it here so
    # every agent — solvers and the one-shot evaluator alike — is covered by one
    # rule instead of by remembering to pre-create each name.
    (ws / solver).mkdir(parents=True, exist_ok=True)
    server, _ = T.make_server(solver)

    async def gate_submit(input_data, tool_use_id, context):
        # A tripwire, not a budget check. No agent is given a submit tool at
        # all, so this can only fire if one is ever added back to the table —
        # in which case denying it is right, and the log line says why.
        trace.log("submit_tool_appeared", solver=solver)
        return {"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "permissionDecision": "deny",
            "permissionDecisionReason":
                "No agent submits. The evaluator decides which candidate goes "
                "and when; the harness sends it, so that the shared quota is "
                "counted in one place."}}

    async def gate_bash(input_data, tool_use_id, context):
        # A budget the model can step around by shelling out is not a budget.
        # Run 1 spent its allowance through `kaggle competitions submit` in Bash
        # while the harness counter sat at 0/3 and the PreToolUse gate on the
        # MCP tool never fired once.
        cmd = str((input_data.get("tool_input") or {}).get("command", ""))
        if args.mode == "clean-benchmark" and re.search(
                r"kaggle\s+(competitions|c)\s+(submissions|leaderboard)",
                cmd, re.I):
            evidence.taint_process(
                ws, solver,
                "attempted to read Kaggle audit data in clean-benchmark mode",
                attempted_action=cmd,
            )
            trace.log("forbidden_audit_read", solver=solver, tool="Bash")
            return {"hookSpecificOutput": {
                "hookEventName": "PreToolUse", "permissionDecision": "deny",
                "permissionDecisionReason":
                    "Clean-benchmark mode seals Kaggle submissions and the "
                    "leaderboard from solvers. This route is now tainted; use "
                    "only the frozen local evaluator."}}
        if re.search(r"kaggle\s+(competitions|c)\s+submit", cmd):
            return {"hookSpecificOutput": {
                "hookEventName": "PreToolUse", "permissionDecision": "deny",
                "permissionDecisionReason":
                    "You do not submit — the harness does, once the evaluator "
                    "has scored enough candidates to choose between. Write "
                    "out/submission.csv; check mcp__ioai__submission_status for "
                    "where the shared quota stands."}}
        # CUDA_VISIBLE_DEVICES is not hierarchical: a child that sets it
        # overrides the parent's restriction outright, so `CUDA_VISIBLE_DEVICES=0`
        # inside a run launched with 4,5,6,7 means physical GPU 0 — someone
        # else's card. That happened in run 2.
        m = re.search(r"CUDA_VISIBLE_DEVICES\s*=\s*([0-9, ]+)", cmd)
        if m and ALLOWED_GPUS:
            asked = {x.strip() for x in m.group(1).split(",") if x.strip()}
            stray = asked - ALLOWED_GPUS
            if stray:
                return {"hookSpecificOutput": {
                    "hookEventName": "PreToolUse", "permissionDecision": "deny",
                    "permissionDecisionReason":
                        f"GPU {sorted(stray)} is not ours — this machine is "
                        f"shared and other people's jobs run on it. Allowed: "
                        f"{sorted(ALLOWED_GPUS)}. Note that setting "
                        f"CUDA_VISIBLE_DEVICES in a child process replaces the "
                        f"parent's setting rather than indexing into it."}}
        return {}

    async def gate_web(input_data, tool_use_id, context):
        if args.mode != "clean-benchmark":
            return {}
        evidence.taint_process(
            ws, solver,
            "attempted external web access in clean-benchmark mode",
            attempted_action=str(input_data.get("tool_input", "")),
        )
        trace.log("forbidden_audit_read", solver=solver,
                  tool=input_data.get("tool_name"))
        return {"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "permissionDecision": "deny",
            "permissionDecisionReason":
                "Clean-benchmark mode uses only competition materials already "
                "fetched by the harness. This route is now tainted."}}

    async def trace_tool(input_data, tool_use_id, context):
        trace.log("tool", solver=solver, name=input_data.get("tool_name"),
                  input=str(input_data.get("tool_input"))[:200])
        return {}

    return ClaudeAgentOptions(
        model=args.model,
        # Preset KEPT, contract appended. Replacing it was our biggest
        # self-inflicted wound on the previous branch.
        system_prompt={"type": "preset", "preset": "claude_code", "append": append},
        setting_sources=None,   # None => CLI default => user+project settings load
        # `tools` deliberately not passed => full built-in toolset (incl. WebSearch)
        mcp_servers={"ioai": server},
        permission_mode="bypassPermissions",
        cwd=str(ws / solver),
        effort=args.effort,
        max_budget_usd=cost_cap,
        enable_file_checkpointing=True,
        max_turns=args.max_turns,
        hooks={"PreToolUse": [HookMatcher(matcher="mcp__ioai__kaggle_submit",
                                          hooks=[gate_submit]),
                              HookMatcher(matcher="Bash", hooks=[gate_bash]),
                              HookMatcher(matcher="WebSearch", hooks=[gate_web]),
                              HookMatcher(matcher="WebFetch", hooks=[gate_web])],
               "PostToolUse": [HookMatcher(hooks=[trace_tool])]},
    )


async def drain(client, solver: str, budget: Budget, trace: Tracer,
                mark_active: bool = True) -> tuple[float, int]:
    cost = turns = 0
    async for msg in client.receive_response():
        if mark_active:
            # Liveness, sampled where it actually happens. A solver watching a
            # background job emits nothing for minutes; that silence is the
            # signal, and it is only visible from inside this loop.
            S.ACTIVITY[solver] = time.time()
        if isinstance(msg, AssistantMessage):
            for blk in msg.content:
                if isinstance(blk, TextBlock) and blk.text.strip():
                    print(f"  [{solver}] {blk.text.strip()[:280]}")
                    trace.log("say", solver=solver, text=blk.text[:600])
                elif isinstance(blk, ToolUseBlock):
                    print(f"  [{solver}] -> {blk.name}({str(blk.input)[:110]})")
        elif isinstance(msg, ResultMessage):
            cost = msg.total_cost_usd or 0.0
            turns = msg.num_turns
            budget.cost_usd += cost
            u = msg.usage or {}
            budget.tokens_in += u.get("input_tokens", 0)
            budget.tokens_out += u.get("output_tokens", 0)
            trace.log("result", solver=solver, cost=cost, turns=turns,
                      err=msg.is_error)
    return cost, turns


def progress_mark(ws: Path, solver: str) -> tuple:
    """A cheap fingerprint of "did this solver actually produce anything".

    Two observed stalls look identical from the outside — run 1 spent 48% of its
    tool calls re-fighting the same submit syntax, and a solver can sit for ten
    minutes watching a background job — so rather than guess at intent we just
    ask whether the round left a trace: new or changed files under `out/`, or a
    fact on the board.
    """
    out = ws / solver / "out"
    files = sorted((p.name, int(p.stat().st_mtime), p.stat().st_size)
                   for p in out.glob("**/*") if p.is_file()) if out.exists() else []
    posted = 0
    fp = ws / "facts.jsonl"
    if fp.exists():
        posted = sum(1 for line in fp.read_text().splitlines()
                     if f'"src": "{solver}"' in line)
    return (tuple(files), posted)


async def watchdog_turn(client, solver: str, drainer, args, trace: Tracer) -> None:
    """Cut a turn short when the agent has gone quiet inside it.

    Round-boundary checks cannot help here: in run 2 a solver sat watching a
    background job for eight minutes and fifty seconds, which was one turn, so
    the loop had no boundary to notice at. `interrupt()` is the only thing that
    reaches inside a turn — and once it lands, the round ends normally and the
    stall logic takes over from there.
    """
    quiet, tries = args.stall_seconds, 0
    while not drainer.done():
        await asyncio.sleep(15)   # cheap: reads one timestamp
        idle = S.stalled_for(solver)
        if idle > quiet:
            print(f"  [{solver}] silent {idle:.0f}s inside one turn — interrupting",
                  flush=True)
            trace.log("interrupt", solver=solver, idle=round(idle))
            try:
                await client.interrupt()
                return
            except Exception as e:  # noqa: BLE001
                # One attempt used to be all it got, and a failed interrupt left
                # the turn hanging for the rest of the run with nothing watching
                # it any more.
                tries += 1
                trace.log("interrupt_failed", solver=solver, attempt=tries,
                          err=f"{type(e).__name__}: {e}")
                if tries >= 3:
                    trace.log("interrupt_gave_up", solver=solver)
                    HUNG.add(solver)
                    return


NUDGE = """
连续两轮没有在 `out/` 下新增或修改任何文件，也没有向事实板发布新事实。
这是卡住的表现，不是深入思考。

停止当前尝试。如果有长任务仍在运行，就终止它；连续两轮没有提供信息的任务，
现在也不会突然变好。然后选择成本最低、能让脚本读取到结果的一步：
生成可由 `native/scripts/evaluate.py` 计分的 `out/oof.npy`，生成
`out/submission.csv`，或发布一条 `failure`，明确说明什么不工作，避免
别人重复浪费时间。

一个真实存在但较弱的候选，胜过一个只存在于设想中的更强候选。
"""


async def run_solver(solver: str, ws: Path, args, budget: Budget,
                     trace: Tracer, peers_n: int) -> None:
    """One solver, across however many sessions it takes.

    A session that crashes or stops responding used to remove a solver for the
    rest of the run — one FAILED line and it was gone. Its work is in files and
    on the board, not in the context window, so a fresh session can carry on
    from the artifacts. It loses the reasoning that got there, which is worth
    much less than the capacity it gets back.
    """
    n = peers_n
    peers = ("你是这道题唯一的 solver。" if n == 1 else
             f"你的工作目录只属于你；共有 {n} 个 solver 并行解决这道题，"
             "没有任何一个被预先指定路线。")
    append = CONTRACT.format(slug=args.slug, peers=peers,
                             mode_contract=mode_contract(args),
                             board=(BOARD_SOLO if n == 1
                                    else BOARD_MULTI.format(n=n)),
                             budget=budget_line(budget),
                             deadline_min=args.deadline_min)
    cap = args.max_cost_usd / n
    spent = 0.0
    costs: list[float] = []
    stalled = 0
    usage = None
    print(f"[{solver}] start | no assigned angle | cap=${cap:.0f}", flush=True)

    for life in range(1, args.max_lives + 1):
        extra = ""
        if life > 1:
            arts, mine = revival_context(ws, solver)
            extra = REVIVAL.format(cwd=ws / solver, artifacts=arts, mine=mine,
                                   left=max(budget.remaining(), 0) / 60)
            HUNG.discard(solver)
            print(f"[{solver}] revived (life {life}), ${cap - spent:.1f} left "
                  f"of its budget", flush=True)
            trace.log("revived", solver=solver, life=life,
                      budget_left=round(cap - spent, 2))
            facts.board().post(
                "env", f"{solver} lost its session and was restarted — it keeps "
                       f"its files but not its reasoning.", src="harness")

        o = opts(solver, ws, args, budget, trace, append + extra,
                 max(cap - spent, 0.5))
        clean = False
        try:
            async with ClaudeSDKClient(options=o) as client:
                for rnd in range(1, args.rounds + 1):
                    # `max_budget_usd` caps a single query, not the session, so
                    # a loop of cheap-looking rounds walks straight past the cap
                    # — run 1 finished at $19.05 against a $15 limit. Context
                    # only grows, so the next round costs at least about what
                    # the last one did; stop while that still fits.
                    est = S.predict_next_cost(usage, costs)
                    if (budget.remaining() <= 0
                            or spent + est >= cap
                            or budget.cost_usd + est >= budget.max_cost_usd):
                        trace.log("stop", solver=solver, reason="budget",
                                  round=rnd, spent=round(spent, 2),
                                  est_next=round(est, 2))
                        print(f"  [{solver}] stopping: ${spent:.2f} spent, next "
                              f"round ~${est:.2f}, cap ${cap:.0f}", flush=True)
                        clean = True
                        break

                    tmpl = FIRST if rnd == 1 and life == 1 else CONTINUE
                    prompt = tmpl.format(budget=budget_line(budget))
                    if stalled >= 2:
                        prompt += NUDGE
                        trace.log("nudge", solver=solver, round=rnd)
                        print(f"  [{solver}] stalled {stalled} rounds — nudging",
                              flush=True)
                    # Deliver the board here, not only through tool results. The
                    # piggyback rides on MCP calls and solvers barely make any —
                    # they live in Bash. In the radar run solver_c went twelve
                    # minutes with no delivery window and spent a training run
                    # rediscovering what solver_a had already posted. Rounds are
                    # a channel the harness owns, so they are the reliable one.
                    prompt += S.drain_urgent(solver)
                    news = facts.board().render_unseen(solver)
                    if news:
                        prompt += "\n" + news

                    before = progress_mark(ws, solver)
                    await client.query(prompt)
                    drainer = asyncio.create_task(
                        drain(client, solver, budget, trace))
                    watch = asyncio.create_task(
                        watchdog_turn(client, solver, drainer, args, trace))
                    cost, turns = await drainer
                    watch.cancel()
                    spent += cost
                    costs.append(cost)
                    try:
                        usage = await client.get_context_usage()
                    except Exception:  # noqa: BLE001
                        usage = None
                    stalled = (0 if progress_mark(ws, solver) != before
                               else stalled + 1)
                    print(f"  [{solver}] round {rnd}: turns={turns} "
                          f"${cost:.2f} (solver ${spent:.2f}/{cap:.0f})"
                          + (f" [no output x{stalled}]" if stalled else ""),
                          flush=True)

                    promote_all(ws, [solver], trace)
                    publish_results(ws, ALL_SOLVERS, trace,
                                    disclosure=args.result_disclosure)
                    if solver in HUNG:
                        # The watchdog could not reach this session at all.
                        raise RuntimeError("session unreachable — interrupt "
                                           "failed three times")
                else:
                    clean = True
        except asyncio.CancelledError:
            print(f"[{solver}] cancelled at deadline", flush=True)
            raise
        except Exception as e:  # noqa: BLE001 — one solver must not end the run
            print(f"[{solver}] session died: {type(e).__name__}: {e}", flush=True)
            trace.log("solver_error", solver=solver, life=life,
                      err=f"{type(e).__name__}: {e}")

        if clean or budget.remaining() <= 0 or spent >= cap:
            return
    trace.log("lives_exhausted", solver=solver, lives=args.max_lives)
    print(f"[{solver}] out of lives after {args.max_lives} sessions", flush=True)


EVALUATOR = """你只负责本轮的测量体系，不负责其他工作。你不解题，不评价任何
solver 的路线是否优秀，也不为探索本身而浏览数据。

工作分两部分。第一部分只做一次：编译统一标尺。第二部分持续到运行结束：当
harness 将候选送来时，决定提交哪个、何时提交。两部分之间没有任务是正常的。

先尽快完成标尺；在它生成前，solver 都无法统一计分。只研究到足以确定 metric
和可靠 split 的程度。确定性侦察已经完成，结果在 `recon.md` 和事实板中，
不要重复做；寻找泄漏或重复帧不是你的职责。

在 workspace 根目录（{ws}）生成两个 artifact：

1. `metric.py`：`{slug}` 的官方 metric，暴露
   `def score(y_true, y_pred) -> float | dict`。完成前必须用一组手算出正确值的
   合成标签自测，并展示该测试。一个细微错误的 metric 比没有 metric 更危险，
   因为所有下游结果都会被它错误测量而不自知。

   必须交叉验证定义，不能只抄 evaluation 页面。镜像竞赛有时保留 Kaggle 未
   填写的模板页面，其中用示例文字演示另一个 metric 和提交格式；它看起来
   具体自信，却可能完全不属于本题。`kaggle_overview` 会标记可识别的模板，
   但只是启发式检查。无论页面怎么写，都要与 `sample_submission.csv`、实际
   数据和题目显然要求相互核对。页面与数据冲突时，以数据为准，并在事实板说明。

2. `folds.json`：本轮所有候选共用的唯一计分划分。怎么切完全取决于本题，
   因此先加载 playbook，不要从零猜：

       skill_load(name="validation-split")

   它说明文件契约、如何结合训练预算选择 k，以及哪些数据结构会被随机 split
   泄漏。harness 只强制三点：文件合法
   （`{py} -m native.scripts.checkfolds --workspace {ws}`）、恰好覆盖所有评估
   单元、并且一旦有候选在其上计分就永不重切。

   冻结后，把划分方案和理由发布到事实板。本轮所有数字都来自它，因此每个
   solver 都必须知道它的含义。

如果题目材料确实无法唯一确定 metric，不要为了证明它而卡住。实现最可信的
解释，在 `metric.py` 顶部明确写出假设，以 `data` 事实发布到事实板，然后
结束这一阶段。一个冻结、记录完整但略有不确定的共享标尺，胜过没有标尺；
事实板上的未决问题至少能让 solver 采取行动，而沉默不能。

完成后报告你构建的内容并等待。review 请求会作为新消息到达。
"""


REVIEW = """
现在有一个候选等待提交，由你做最终判断。全队每天只有少量且不会恢复的名额，
错误使用一次就永久损失一次。

你适合持有这个闸门，因为你是唯一没有自己候选的 agent。过去让 solver 自己
决定时，三条路线四分钟内都选择“买保险”，而第十二分钟出现的全轮最佳模型
已经没有名额可用。

harness 已检查脚本可以判断的内容：配额、节流，以及该候选是否在共享 folds
上得分最高。脚本无法判断的是：submission 对本竞赛是否真正合法，以及现在
花名额是否优于继续等待改进。

    candidate:  {cand}
    workspace:  {ws}
    files:      {cand}/out/submission.csv, {cand}/out/oof.npy{extra}

机械检查结果如下。它们是在不知道本题规则时写下的启发式检查，因此只能作为
需要你裁决的观察，不能直接当结论：

    fatal（任何规则下都错误）： {fatal}
    warnings（取决于规则）：     {warn}

warning 只有经过你的判断才是缺陷。上轮出现过 "contains negative values" 和
"no sample_submission to compare against"；那道题中 -1 本来就是背景标签，
而镜像确实没有 sample submission，所以两条观察都正确，却都不是问题。

你负责两件事：

VALIDITY：若有主办方 grading code，就直接阅读，并检查所有可能让 submission
被静默判零的条件：数值范围、编码、ID 对齐、行顺序、单位，以及列名是否真的
表达其表面含义。Chicken 题中只要出现一个负像素，整份提交就会被判零，而这点
只有 grading code 写明。

TIMING：剩余 {left} 个名额，{total:.0f} 分钟窗口已过去 {elapsed:.0f} 分钟。
运行早期且仍明显改进时保留名额是合理的；时间将尽时继续保留就不合理，因为
未使用名额得分为零。不要重新评价模型能力：模型已经被测量，这就是当前得分
最高候选。

写入 `review/{cand}.json`，内容必须严格为
`{{"verdict": "APPROVE"|"HOLD"|"REJECT", "reason": "<一句话>"}}`。

    APPROVE  允许发送
    HOLD     文件合法，但值得等待更好候选
    REJECT   文件有问题；明确指出问题，让 solver 修复

然后在事实板发布一条事实：若发现所有路线都应避免的合法性问题，用 `format`；
否则用 `claim` 说明你检查了什么。

请快速完成。competition 模式下若超时，submitter 会退化到纯机械检查；这会让
你的判断失去作用。clean-benchmark 模式不会使用超时降级，必须得到你的明确
APPROVE 才能进入 clean LKG。
"""


async def review_loop(ws: Path, args, budget: Budget, trace: Tracer) -> None:
    """The evaluator's second life: approving submissions, not just measuring.

    It stays alive after compiling the ruler because format validity is a
    judgement a script cannot make. On the chicken task the evaluator found, by
    reading the organisers' code, that one negative pixel voids the whole
    submission — nothing generic would have thought to check that.

    It is watched like any solver. A gate that can hang is a way to score zero,
    so the submitter also has a timeout and a mechanical floor to fall back on.
    """
    client = REVIEWER.get("client")
    if client is None:
        return
    while True:
        await asyncio.sleep(3)
        if budget.remaining() <= 0:
            return
        try:
            cand = REVIEW_Q.get_nowait()
        except asyncio.QueueEmpty:
            continue
        mech = await asyncio.to_thread(format_check, ws, cand)
        extra = (f", {cand}/out/kernel/" if SUBMIT_MODE["mode"] == "kernel" else "")
        (ws / "review").mkdir(exist_ok=True)
        prompt = REVIEW.format(
            cand=cand, ws=ws, extra=extra,
            fatal=json.dumps(mech.get("fatal") or "none"),
            warn=json.dumps(mech.get("warnings") or "none"),
            left=max(budget.max_submissions - budget.submissions, 0),
            elapsed=budget.elapsed() / 60,
            total=budget.deadline_s / 60)
        print(f"  [evaluator] reviewing {cand}", flush=True)
        trace.log("review_start", candidate=cand, mechanical=mech)
        try:
            await client.query(prompt)
            await drain(client, "evaluator", budget, trace)
        except Exception as e:  # noqa: BLE001
            trace.log("review_error", candidate=cand, err=f"{type(e).__name__}: {e}")


def integrity_check(ws: Path, candidate: str, *, require_sample_locality: bool = False) -> dict:
    """Provenance is fatal — a file that moved after scoring makes the score a
    lie. Order-dependence is a warning: ordered data is legitimate, exploiting
    the ordering is not, and only someone who knows the task can tell which."""
    from native.scripts.integrity import (check_order_dependence,
                                          check_provenance,
                                          check_sample_locality)
    try:
        locality_fatal, locality_warn = check_sample_locality(
            ws, candidate, required=require_sample_locality)
        return {"fatal": check_provenance(ws, candidate) + locality_fatal,
                "warnings": check_order_dependence(ws, candidate) + locality_warn}
    except Exception as e:  # noqa: BLE001
        return {"fatal": [], "warnings": [f"integrity check failed: {e}"]}


def format_check(ws: Path, candidate: str) -> dict:
    from native.scripts.check_format import check
    try:
        return check(ws, candidate)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "problems": [f"{type(e).__name__}: {e}"]}


def read_verdict(ws: Path, candidate: str) -> dict | None:
    p = ws / "review" / f"{candidate}.json"
    if not p.exists():
        return None
    try:
        v = json.loads(p.read_text())
        return v if "verdict" in v else None
    except Exception:  # noqa: BLE001
        return None


async def run_evaluator(ws: Path, args, budget: Budget, trace: Tracer) -> None:
    """Compile the ruler, then stay on as the submission gate.

    It used to exit once metric.py and folds.json existed. It stays now because
    validity is a judgement: no generic script would have known to check that a
    negative density pixel voids an entire chicken-counting submission, and the
    evaluator only found that by reading the organisers' grading code.
    """
    o = opts("evaluator", ws, args, budget, trace,
             EVALUATOR.format(ws=ws, slug=args.slug, py=sys.executable),
             cost_cap=args.evaluator_cost)
    print("[evaluator] compiling metric.py + folds.json", flush=True)
    async with ClaudeSDKClient(options=o) as client:
        REVIEWER["client"] = client
        await client.query("现在编译统一标尺；完成后报告结果，并等待 review 请求。")
        drainer = asyncio.create_task(drain(client, "evaluator", budget, trace))
        watch = asyncio.create_task(
            watchdog_turn(client, "evaluator", drainer, args, trace))
        await drainer
        watch.cancel()
        # Now serve reviews until the run ends. The same turn watchdog applies:
        # the gate is watched exactly like the solvers, because a gate that
        # hangs is a way to score zero rather than a way to avoid a bad score.
        await review_loop(ws, args, budget, trace)


def data_present(ws: Path) -> bool:
    inp = ws / "input"
    return inp.exists() and any(p.is_file() for p in inp.rglob("*"))


def submission_mode(slug: str) -> tuple[str, int]:
    """(mode, daily limit) — does this competition require a Kaggle kernel?

    Worth asking rather than assuming. IOAI 2026 proper is kernel-only, but the
    2025 mirrors accept a CSV directly, and telling a solver it must train
    in-kernel when it needn't sends it queueing for a cloud GPU it could have
    skipped — on a 30-minute budget that is most of the budget.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        r = api.competitions_list(search=slug)
        for c in (getattr(r, "competitions", None) or []):
            if getattr(c, "ref", "").endswith(slug) or slug in str(getattr(c, "ref", "")):
                kernels_only = bool(getattr(c, "is_kernels_submissions_only", True))
                limit = int(getattr(c, "max_daily_submissions", 0) or 0)
                return ("kernel" if kernels_only else "csv"), limit
    except Exception:  # noqa: BLE001 — unknown means assume the strict path
        pass
    return "unknown", 0


def submissions_used_today(slug: str) -> int:
    """How much of today's quota is already gone.

    The daily cap is per account and resets on its own clock, but our budget
    used to start at zero every launch — so two runs in one afternoon planned
    for three submissions each against a real allowance of five, and the second
    run discovered the ceiling by hitting it.
    """
    try:
        from datetime import datetime, timezone
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        today = datetime.now(timezone.utc).date()
        n = 0
        for s in (api.competition_submissions(slug) or []):
            d = getattr(s, "date", None)
            if d is not None and getattr(d, "date", lambda: None)() == today:
                n += 1
        return n
    except Exception:  # noqa: BLE001
        return 0


# Small module-level view of the shared quota, so the read-only status tool can
# answer without threading state through every call site.
QUOTA: dict = {"limit": 0, "used_today": 0, "sent": [], "budget": None}


def quota_report(ws: Path) -> str:
    b = QUOTA.get("budget")
    left = "?" if b is None else max(b.max_submissions - b.submissions, 0)
    lines = [f"daily cap {QUOTA['limit']} | already used today "
             f"{QUOTA['used_today']} | this run may still send {left}",
             "The harness submits, not you. Write out/submission.csv; the best "
             "candidate measured by evaluate.py is what goes."]
    for s in QUOTA["sent"]:
        lines.append(f"  sent: {s}")
    return "\n".join(lines)


_SCORE_CACHE: dict[str, tuple[tuple, float]] = {}


def scored_candidates(ws: Path, names: list[str]) -> dict[str, float]:
    """Candidates that have a script-computed score right now.

    Cached on the prediction file's identity. Re-scoring is not free — an oof
    array is one row per training example times however wide a prediction is,
    which for a dense-output task is tens of megabytes — and the submitter asks
    this question on a timer, so without the cache we would reload and re-score
    every candidate several times a minute to learn nothing new, on the same
    machine the solvers are training on.
    """
    from native.scripts.evaluate import evaluate
    out = {}
    for n in names:
        p = ws / n / "out" / "oof.npy"
        if not p.exists():
            continue
        try:
            st = p.stat()
            key = (st.st_mtime_ns, st.st_size)
        except OSError:
            continue
        hit = _SCORE_CACHE.get(n)
        if hit and hit[0] == key:
            out[n] = hit[1]
            continue
        try:
            r = evaluate(ws, n)
        except Exception:  # noqa: BLE001
            continue
        if r.get("status") == "ok":
            _SCORE_CACHE[n] = (key, r["mean"])
            out[n] = r["mean"]
            # Seal here, where the score is taken, so the two always describe
            # the same file. Sealing only at promote made the seal go stale the
            # moment a solver improved its candidate, and the integrity check
            # then blocked that improvement four times over ten minutes for
            # having "changed after it was scored" — it had, and it had been
            # rescored since.
            try:
                from native.scripts.integrity import record as _seal
                _seal(ws, n, r["mean"])
            except Exception:  # noqa: BLE001
                pass
    return out


def should_submit(*, n_scored: int, frac: float, best: dict | None,
                  sent_best: float | None, sent: int, allowance: int,
                  min_candidates: int, fallback_frac: float) -> tuple[bool, str]:
    """Pure decision so it can be tested rather than trusted.

    `frac` is how far through the window we are, 0..1.
    """
    if not best:
        return False, "no scored candidate yet"
    if sent_best is not None and best["mean"] <= sent_best + 1e-6:
        return False, "no measurable improvement on what was already sent"
    # Half the allowance is reserved for the second half of the window; across
    # every run so far the best candidate has arrived late, never first.
    if frac < 0.5 and sent >= max(1, allowance // 2):
        return False, "early-window reserve reached"
    if n_scored >= min_candidates:
        return True, f"{n_scored} candidates scored, best of them"
    if frac >= fallback_frac:
        return True, (f"time override at {frac:.0%} of the window with only "
                      f"{n_scored} scored")
    return False, (f"only {n_scored}/{min_candidates} candidates scored and "
                   f"{frac:.0%} of the window gone")


SUBMIT_MODE = {"mode": "csv"}
REVIEWER: dict = {"client": None}
# Last score announced per candidate, so a re-scored but unchanged candidate
# does not repost the same number every round.
_POSTED: dict[str, float] = {}
# Submissions sent but not yet scored by Kaggle.
PENDING: list[dict] = []


def fetch_scores(slug: str) -> list[dict]:
    """Today's submissions with whatever score Kaggle has attached so far."""
    from datetime import datetime, timezone
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    today = datetime.now(timezone.utc).date()
    out = []
    for s in (api.competition_submissions(slug) or []):
        d = getattr(s, "date", None)
        if d is None or d.date() != today:
            continue
        # snake_case on the object; the CLI's column header is publicScore,
        # which is not the attribute name.
        raw = getattr(s, "public_score", getattr(s, "publicScore", None))
        try:
            score = float(raw) if raw not in (None, "") else None
        except (TypeError, ValueError):
            score = None
        out.append({"description": str(getattr(s, "description", "")),
                    "score": score, "ref": str(getattr(s, "ref", "")),
                    "status": str(getattr(s, "status", ""))})
    return out


async def score_watcher(ws: Path, args, budget: Budget, trace: Tracer) -> None:
    """Collect the leaderboard score for what we sent, and say what it implies.

    Without this the run never learns whether its own measurements mean
    anything. `promote --confirm` has existed the whole time with nothing
    calling it, so the confirmed tier of LKG stayed empty even on a run that
    submitted four times and took first place — and the one number that matters
    most, how far local sits from the leaderboard, was only ever worked out by
    hand afterwards.

    A solver that knows its out-of-fold score has been running just under the
    leaderboard can trust it and keep optimising. One told the gap is large and
    the wrong way round should stop believing its own numbers, which is exactly
    the situation that produced 0.9156 local against 0.78095 on the board.
    """
    seen: set[str] = set()
    while True:
        await asyncio.sleep(90)
        if budget.remaining() <= 0 and not PENDING:
            return
        if not PENDING:
            continue
        try:
            rows = await asyncio.to_thread(fetch_scores, args.slug)
        except Exception:  # noqa: BLE001
            continue
        for p in list(PENDING):
            key = p["message"][:60]
            hit = next((r for r in rows
                        if r["score"] is not None and key[:40] in r["description"]),
                       None)
            if hit is None or key in seen:
                continue
            seen.add(key)
            PENDING.remove(p)
            gap = hit["score"] - p["local"]
            trace.log("confirmed", candidate=p["candidate"], local=p["local"],
                      leaderboard=hit["score"], gap=round(gap, 5))
            await asyncio.to_thread(
                subprocess.run,
                [sys.executable, "-m", "native.scripts.promote", "--confirm",
                 p["candidate"], "--lb", str(hit["score"]),
                 "--submitted-snapshot", p["snapshot_path"],
                 "--submission-sha256", p["submission_sha256"],
                 "--workspace", str(ws)],
                **{"cwd": str(ROOT), "capture_output": True, "text": True})
            manifest = Path("evidence") / "candidates" / f"{p['candidate']}.json"
            reveal = evidence.record_reveal(ws, {
                "candidate": p["candidate"],
                "candidate_manifest": str(manifest),
                "candidate_manifest_sha256": (
                    evidence.sha256_file(ws / manifest)
                    if (ws / manifest).exists() else None
                ),
                "local_score": p["local"],
                "leaderboard_score": hit["score"],
                "gap": round(gap, 8),
                "submission_ref": hit["ref"],
                "submitted_snapshot": p["snapshot_path"],
                "submission_sha256": p["submission_sha256"],
                "mode": args.mode,
                "fed_back_to_solvers": args.mode == "competition",
            })
            verdict = ("local is conservative — trust it" if gap >= -0.002 else
                       "LOCAL IS OPTIMISTIC — stop choosing on out-of-fold "
                       "scores alone")
            print(f"  [confirmed] {p['candidate']} local {p['local']:.5f} -> "
                  f"leaderboard {hit['score']:.5f} (gap {gap:+.5f})", flush=True)
            if args.mode == "competition":
                facts.board().post(
                    "result", f"{p['candidate']}: local {p['local']:.5f} -> "
                              f"leaderboard {hit['score']:.5f}, gap {gap:+.5f}. "
                              f"{verdict}."[:300], src="harness",
                    evidence_refs=[str(reveal.relative_to(ws))])
            else:
                trace.log("reveal_withheld", candidate=p["candidate"],
                          receipt=str(reveal))
REVIEW_Q: asyncio.Queue = asyncio.Queue()


def promote_clean_after_review(ws: Path, cand: str, trace: Tracer) -> tuple[dict | None, str]:
    """Turn an explicit evaluator APPROVE into a hash-bound clean LKG entry."""
    review_path = ws / "review" / f"{cand}.json"
    try:
        review = json.loads(review_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"clean review is unreadable: {exc}"
    if str(review.get("verdict", "")).upper() != "APPROVE":
        return None, "clean promotion requires an explicit evaluator APPROVE"

    post_review = integrity_check(ws, cand, require_sample_locality=True)
    if post_review["fatal"]:
        return None, "post-review integrity failed: " + "; ".join(post_review["fatal"])

    from native.scripts.integrity import evidence_path, record
    previous = {}
    manifest_path = evidence_path(ws, cand)
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = record(ws, cand, previous.get("score"))

    body = {
        "schema_version": 1,
        "candidate": cand,
        "proposer": cand,
        "reviewer": "evaluator",
        "evidence_tier": "clean_oof",
        "decision": "promote",
        "prediction_sha256": manifest.get("oof"),
        "split_sha256": manifest.get("folds"),
        "metric_sha256": manifest.get("metric"),
        "candidate_manifest": str(manifest_path.relative_to(ws)),
        "candidate_manifest_sha256": evidence.sha256_file(manifest_path),
        "evaluator_review": str(review_path.relative_to(ws)),
        "evaluator_review_sha256": evidence.sha256_file(review_path),
        "run_contract_sha256": manifest.get("run_contract"),
        "sample_locality_receipt_sha256": manifest.get("sample_locality_receipt"),
        "reason": " ".join(str(review.get("reason", "")).split())[:500],
    }
    digest = hashlib.sha256(evidence.canonical(body)).hexdigest()
    body["receipt_sha256"] = digest
    receipt = ws / "evidence" / "clean_reviews" / f"{cand}-{digest[:16]}.json"
    if receipt.exists():
        if json.loads(receipt.read_text(encoding="utf-8")) != body:
            return None, f"clean review receipt collision: {receipt}"
    else:
        evidence.write_json(receipt, body)

    run = subprocess.run(
        [sys.executable, "-m", "native.scripts.promote",
         "--candidate", cand, "--tier", "clean",
         "--clean-receipt", str(receipt),
         "--proposer", cand, "--reviewer", "evaluator",
         "--workspace", str(ws)],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    line = (run.stdout or run.stderr).strip().splitlines()[-1:] or [""]
    trace.log("clean_promote", candidate=cand, receipt=str(receipt),
              returncode=run.returncode, out=line[0][:300])
    if run.returncode != 0:
        return None, f"clean promotion failed: {line[0][:240]}"
    state_path = ws / "LKG" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    clean = state.get("clean")
    if not clean or clean.get("candidate") != cand:
        return None, "clean promotion did not select the reviewed candidate"
    return clean, f"clean receipt {receipt.relative_to(ws)}"


async def clear_for_submission(ws: Path, cand: str, args, trace: Tracer,
                               frac: float = 0.0) -> tuple[bool, str]:
    """The evaluator decides; the mechanical checks are evidence handed to it.

    This used to run the other way round, and it cost an entire run. A generic
    check flagged every pixel column of a radar submission for containing -1,
    not knowing that -1 *is* the background label there, and because the check
    was treated as a verdict it blocked the same 0.9816 candidate eighteen times
    over eighteen minutes. The one agent that could have said "that reading is
    wrong" was never asked, because the fallible heuristic was gated ahead of
    the judgement that would have corrected it. A solver even posted the
    diagnosis to the board at minute fourteen; nothing in the harness reads the
    board.

    So a check only blocks on its own when there is nobody to ask, or when the
    finding is fatal under any rules at all — a missing file, an unparseable
    one, NaNs, a shape that contradicts a sample submission that exists.
    """
    mech = await asyncio.to_thread(format_check, ws, cand)
    integ = await asyncio.to_thread(
        integrity_check, ws, cand,
        require_sample_locality=args.mode == "clean-benchmark")
    mech["fatal"] = list(mech.get("fatal", [])) + integ["fatal"]
    mech["warnings"] = list(mech.get("warnings", [])) + integ["warnings"]
    mech["ok"] = not mech["fatal"]

    if REVIEWER.get("client") is None:
        if not mech.get("ok"):
            facts.board().post(
                "format", f"{cand} not submitted — {'; '.join(mech.get('fatal', []))}"
                          [:300], src="harness")
            return False, f"fatal format problem: {mech.get('fatal')}"
        if args.mode == "clean-benchmark":
            return False, "clean benchmark requires independent evaluator approval"
        return True, "mechanical checks only (no evaluator)"

    p = ws / "review" / f"{cand}.json"
    if p.exists():
        p.unlink()
    await REVIEW_Q.put(cand)
    waited = 0.0
    while waited < args.review_timeout:
        await asyncio.sleep(3)
        waited += 3
        v = read_verdict(ws, cand)
        if v is None:
            continue
        trace.log("verdict", candidate=cand, **v)
        verdict = str(v.get("verdict", "")).upper()
        reason = str(v.get("reason", ""))
        if verdict.startswith("APPROVE"):
            return True, f"evaluator approved: {reason}"[:160]
        if verdict.startswith("HOLD"):
            # Valid, but the evaluator would rather wait for something better.
            # Reasonable each time, and fatal if it never stops: eighteen
            # reasonable holds are a zero. Every gate that can refuse needs a
            # point past which it cannot — gate 1 had one, gate 3 did not and
            # cost a run, and this had none either.
            if args.mode == "clean-benchmark":
                return False, f"clean benchmark requires APPROVE; evaluator HOLD: {reason}"[:160]
            if frac >= args.submit_fallback_frac:
                trace.log("hold_overridden", candidate=cand, frac=round(frac, 2))
                facts.board().post(
                    "env", f"{frac:.0%} of the window gone — sending {cand} over "
                           f"the evaluator's hold. An unspent slot scores "
                           f"nothing.", src="harness")
                return True, (f"hold overridden at {frac:.0%} of the window "
                              f"(evaluator wanted to wait: {reason})")[:160]
            return False, f"evaluator holding the slot: {reason}"[:160]
        facts.board().post(
            "format", f"evaluator REJECTED {cand}: {reason}"[:300], src="evaluator")
        return False, f"evaluator rejected: {reason}"

    trace.log("review_timeout", candidate=cand, waited=waited)
    if not mech.get("ok"):
        return False, (f"evaluator silent for {args.review_timeout:.0f}s and the "
                       f"file has a fatal problem: {mech.get('fatal')}")
    if args.mode == "clean-benchmark":
        return False, (f"clean benchmark requires explicit APPROVE; evaluator "
                       f"silent for {args.review_timeout:.0f}s")
    return True, (f"evaluator did not answer in {args.review_timeout:.0f}s; "
                  f"sent on mechanical checks alone")


def send_candidate(ws: Path, candidate: str, csv: Path, msg: str, ctx,
                   trace: Tracer, poll_s: int = 20, waits: int = 30) -> str:
    """Send one candidate, by whichever route this competition actually accepts.

    IOAI proper is kernel-only — the submitted script has to train in-kernel —
    so a submitter that can only upload a CSV cannot submit at all on the day
    that counts. The 2025 mirrors happen to take CSVs, which is what we have
    been testing against, and that difference had gone unnoticed.

    Kernel mode carries a gate that CSV mode does not: a candidate with a fine
    out-of-fold score is still worth nothing if its notebook crashes on Kaggle.
    So push, wait for it to actually finish, and only submit a version that ran.
    """
    if SUBMIT_MODE["mode"] != "kernel":
        return str(R.kaggle_submit({"csv_path": str(csv), "message": msg}, ctx))

    # `csv` lives in the immutable LKG snapshot. The kernel must come from that
    # same snapshot; reading the solver's live directory here could submit code
    # different from the artifact that passed the gates.
    kdir = csv.parent / "kernel"
    if not (kdir / "kernel-metadata.json").exists():
        return f"[skip] {candidate} has no out/kernel/kernel-metadata.json"

    push = str(R.kaggle_push_kernel({"kernel_dir": str(kdir)}, ctx))
    trace.log("kernel_push", candidate=candidate, out=push[:300])
    ref = ver = None
    m = re.search(r"([\w-]+/[\w-]+)", push)
    if m:
        ref = m.group(1)
    m = re.search(r"version\s*(\d+)", push, re.I)
    if m:
        ver = int(m.group(1))

    for _ in range(waits):
        time.sleep(poll_s)
        st = str(R.kaggle_kernel_status({"kernel_ref": ref} if ref else {}, ctx))
        if "COMPLETE" in st.upper():
            break
        if "ERROR" in st.upper():
            log = str(R.kaggle_kernel_log({"kernel_ref": ref} if ref else {}, ctx))
            trace.log("kernel_error", candidate=candidate, log=log[-800:])
            facts.board().post(
                "failure", f"{candidate}'s kernel errored on Kaggle; not "
                           f"submitted. Tail: {log[-160:]}"[:300], src="harness")
            return f"[kernel ERROR] not submitted — {log[-200:]}"
    else:
        return "[kernel timeout] still running; not submitted"

    arg = {"message": msg}
    if ref:
        arg["kernel_ref"] = ref
    if ver:
        arg["kernel_version"] = ver
    return str(R.kaggle_submit(arg, ctx))


async def submitter(ws: Path, args, names: list[str], budget: Budget,
                    trace: Tracer) -> None:
    """The harness owns submission. Solvers have no submit tool.

    Run 2 spent every one of the day's five slots inside four minutes, all of it
    before the metric existed, because the contract told three independent
    agents that scoring zero was the catastrophic outcome and each of them
    bought its own insurance. The best model of the run — OOF 0.877 at twelve
    minutes — then had nothing left to travel on. Allocation of a scarce shared
    resource is not a judgement call to hand to independent agents.

    Gate: the ruler must exist and at least `--min-candidates` candidates must
    carry a real computed score, so the first send is an informed choice between
    alternatives rather than whoever finished first. The time override exists
    because "wait for three" reintroduces the zero it is meant to prevent if one
    solver stalls.
    """
    sent_best: float | None = None
    while True:
        await asyncio.sleep(args.submit_poll_seconds)
        if budget.remaining() <= 0:
            return
        if not budget.can_submit():
            continue
        if not ((ws / "metric.py").exists() and (ws / "folds.json").exists()):
            continue

        scored = scored_candidates(ws, names)
        frac = 1.0 - max(budget.remaining(), 0) / max(budget.deadline_s, 1)
        st = json.loads((ws / "LKG" / "state.json").read_text()) \
            if (ws / "LKG" / "state.json").exists() else {}
        # Candidate search always happens in the development channel. In clean
        # mode the selected candidate enters the clean channel only after an
        # explicit evaluator APPROVE; otherwise clean mode would either submit
        # unreviewed evidence or wait forever on an empty clean LKG.
        best = (st or {}).get("development") or (st or {}).get("local")
        ok, why = should_submit(
            n_scored=len(scored), frac=frac, best=best, sent_best=sent_best,
            sent=budget.submissions, allowance=budget.max_submissions,
            min_candidates=args.min_candidates,
            fallback_frac=args.submit_fallback_frac)
        if not ok:
            continue
        source_snapshot = Path(best.get("path", ""))
        try:
            from native.scripts.promote import verify_snapshot
            snapshot_problems = verify_snapshot(source_snapshot)
        except Exception as exc:  # noqa: BLE001
            snapshot_problems = [f"snapshot audit crashed: {exc}"]
        if snapshot_problems:
            trace.log("submit_blocked", candidate=best.get("candidate"),
                      reason="; ".join(snapshot_problems))
            continue
        if not (source_snapshot / "out" / "submission.csv").exists():
            continue

        cand = best["candidate"]
        ok, note = await clear_for_submission(ws, cand, args, trace, frac)
        if not ok:
            print(f"  [submit] {cand} held back — {note}", flush=True)
            trace.log("submit_blocked", candidate=cand, reason=note)
            continue

        if args.mode == "clean-benchmark":
            clean_best, clean_note = await asyncio.to_thread(
                promote_clean_after_review, ws, cand, trace)
            if clean_best is None:
                print(f"  [submit] {cand} clean promotion blocked — {clean_note}",
                      flush=True)
                trace.log("submit_blocked", candidate=cand, reason=clean_note)
                continue
            best = clean_best
            note = f"{note}; {clean_note}"

        snapshot_path = Path(best.get("path", ""))
        try:
            snapshot_problems = verify_snapshot(snapshot_path)
        except Exception as exc:  # noqa: BLE001
            snapshot_problems = [f"snapshot audit crashed: {exc}"]
        if snapshot_problems:
            trace.log("submit_blocked", candidate=cand,
                      reason="; ".join(snapshot_problems))
            continue
        csv = snapshot_path / "out" / "submission.csv"
        if not csv.exists():
            continue
        submission_sha256 = evidence.sha256_file(csv)
        msg = (f"harness: {best['candidate']} OOF {best['mean']:.5f} "
               f"±{best['std']:.3f} ({len(scored)} scored, {why})")

        ctx = R.Ctx(workspace=ws, slug=args.slug, budget=budget, trace=trace)
        try:
            res = str(send_candidate(ws, cand, csv, f"{msg}; {note}", ctx, trace))
        except Exception as e:  # noqa: BLE001
            res = f"[submit error] {type(e).__name__}: {e}"
        sent_best = best["mean"]
        QUOTA["sent"].append(msg)
        # Remember what we sent and what we thought it was worth, so the score
        # coming back can be compared against it.
        PENDING.append({"candidate": best["candidate"], "local": best["mean"],
                        "message": msg, "at": time.time(),
                        "snapshot_path": str(snapshot_path),
                        "submission_sha256": submission_sha256})
        print(f"  [submit] {msg} -> {res[:120]}", flush=True)
        trace.log("harness_submit", candidate=best["candidate"],
                  mean=best["mean"], scored=len(scored), result=res[:300])
        facts.board().post(
            "format", f"harness submitted {best['candidate']} (OOF "
                      f"{best['mean']:.4f}). Quota left this run: "
                      f"{max(budget.max_submissions - budget.submissions, 0)}.",
            src="harness")


async def reconciler(ws: Path, args, budget: Budget, trace: Tracer) -> None:
    """Watch for the silent failures. Blocks nothing; reports everything.

    Every behavioural fence built for run 1 and run 2 was walked around inside
    one run, and not one of those breaches surfaced while the run was going —
    they were all found the next day by reading logs. Detection is the part the
    harness can actually guarantee, because a solver can bypass a gate but
    cannot stop the harness noticing that Kaggle's numbers disagree with ours.
    """
    rec = S.Reconciler(ws=ws, slug=args.slug, allowed_gpus=ALLOWED_GPUS)
    while True:
        await asyncio.sleep(args.reconcile_seconds)
        if budget.remaining() <= 0:
            return
        drifts = await asyncio.to_thread(
            rec.sweep, budget.submissions, budget.cost_usd, budget.max_cost_usd)
        if S.report(drifts, trace):
            print("!! fatal drift — budget control is lost; stopping the run",
                  flush=True)
            return


_PUBLISHED: dict[str, float] = {}


def claim_of(ws: Path, solver: str) -> str:
    """That solver's most recent claimed direction, if it has posted one."""
    fp = ws / "facts.jsonl"
    if not fp.exists():
        return ""
    latest = ""
    for line in fp.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("kind") == "claim" and r.get("src") == solver:
            latest = r.get("text", "")
    return latest


def publish_results(ws: Path, names: list[str], trace: Tracer,
                    disclosure: str = "live") -> None:
    """Tell everyone what each direction has actually been worth.

    Which approaches have oil in them is too useful to withhold — a solver
    grinding on something the folds have already shown to be flat is the most
    expensive kind of duplicated effort. What matters is where the number comes
    from: these are computed by the evaluator on the shared folds with the
    frozen metric, so they are comparable to each other, which a solver's own
    report would not be.

    Phrased around the approach rather than the person. "Multi-scale buys
    nothing over single-scale" is knowledge about the problem; "solver_a is
    ahead" is a scoreboard, and a scoreboard is what makes independent solvers
    stop being independent.
    """
    scored = scored_candidates(ws, names)
    if not scored:
        return
    if disclosure == "none":
        return
    if disclosure == "after-first" and len(scored) < len(names):
        return
    changed = any(abs(scored[k] - _PUBLISHED.get(k, -1)) > 1e-6 for k in scored)
    if not changed:
        return
    _PUBLISHED.update(scored)
    parts = []
    for k, v in sorted(scored.items(), key=lambda kv: -kv[1]):
        what = (claim_of(ws, k) or k)[:60]
        parts.append(f"{what} → {v:.4f}")
    refs = [str((Path("evidence") / "candidates" / f"{name}.json"))
            for name in scored
            if (ws / "evidence" / "candidates" / f"{name}.json").is_file()]
    facts.board().post("result", "verified on the shared folds: "
                       + " | ".join(parts), src="harness",
                       evidence_refs=refs)
    trace.log("results_published", scores=scored)


def promote_all(ws: Path, names: list[str], trace: Tracer) -> None:
    """Refresh LKG from whatever candidates exist. Local-only and free.

    Deliberately run by the harness on a schedule rather than left to the
    solvers: the safety net is the one thing that must not depend on anybody
    remembering to maintain it. Scoring here touches no Kaggle endpoint, burns
    no GPU quota and joins no queue, so it can run as often as we like.
    """
    if not (ws / "folds.json").exists() or not (ws / "metric.py").exists():
        return
    for n in names:
        if not (ws / n / "out" / "oof.npy").exists():
            continue
        r = subprocess.run(
            [sys.executable, "-m", "native.scripts.promote",
             "--candidate", n, "--workspace", str(ws)],
            cwd=str(ROOT), capture_output=True, text=True)
        line = (r.stdout or r.stderr).strip().splitlines()[-1:] or [""]
        if '"PROMOTED"' in line[0]:
            print(f"  [lkg] {line[0][:160]}", flush=True)
        trace.log("promote", candidate=n, out=line[0][:300])


def run_recon(ws: Path) -> None:
    r = subprocess.run(
        [sys.executable, "-m", "native.scripts.recon",
         "--input", str(ws / "input"), "--workspace", str(ws)],
        cwd=str(ROOT), capture_output=True, text=True)
    print(r.stdout or r.stderr[-1500:], flush=True)


def bootstrap(ws: Path, args, budget: Budget, trace: Tracer) -> None:
    """Deterministic pre-flight: get the data, THEN reconnoitre it.

    The ordering is the whole point, and it is easy to get silently wrong: the
    API download 403s on some mirror competitions, and if we go ahead and
    reconnoitre an empty directory the run continues with an empty facts board
    and nobody notices. So data presence is checked, not assumed, and there is a
    CLI fallback for the 403 case.
    """
    ctx = R.Ctx(workspace=ws, slug=args.slug, budget=None, trace=trace)
    if not data_present(ws):
        try:
            print("[boot]", R.kaggle_download({}, ctx), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[boot] api download failed: {type(e).__name__}: {e}", flush=True)

    if not data_present(ws):
        inp = ws / "input"
        inp.mkdir(parents=True, exist_ok=True)
        print("[boot] api path left input/ empty — trying the CLI", flush=True)
        # Look beside the interpreter first: the launcher is started detached,
        # so the virtualenv's bin is not on PATH and a bare "kaggle" raises
        # FileNotFoundError. And wrap it — a fallback that can kill the run it
        # exists to rescue is worse than no fallback, which is what happened
        # here: the whole launcher died inside the rescue path.
        exe = Path(sys.executable).parent / "kaggle"
        for cmd in ([str(exe)] if exe.exists() else []) + ["kaggle"]:
            try:
                cli = subprocess.run(
                    [cmd, "competitions", "download", "-c", args.slug,
                     "-p", str(inp), "--unzip"],
                    capture_output=True, text=True, timeout=900)
                print(f"[boot] cli({cmd}): {(cli.stdout or cli.stderr)[-300:]}",
                      flush=True)
                if data_present(ws):
                    break
            except Exception as e:  # noqa: BLE001
                print(f"[boot] cli({cmd}) failed: {type(e).__name__}: {e}",
                      flush=True)

    if not data_present(ws):
        msg = (f"could not download {args.slug} — input/ is empty, so "
               f"reconnaissance found nothing. Download it yourself before "
               f"trusting the facts board.")
        print(f"[boot] !! {msg}", flush=True)
        trace.log("bootstrap_no_data", slug=args.slug)
        facts.board().post("env", msg, src="harness")

    # Read the task ONCE, deterministically, before anyone starts. Otherwise
    # every agent fetches the same pages independently — four identical calls —
    # and, worse, each of them has to rediscover on its own that a mirror's
    # pages are Kaggle's unfilled template describing a different competition.
    # That belongs on the board at t=0, not four separate times at minute three.
    try:
        overview = R.kaggle_overview({}, ctx)
        (ws / "TASK.md").write_text(overview)
        print(f"[boot] task statement -> TASK.md ({len(overview)} chars)", flush=True)
        if overview.startswith("!! WARNING"):
            facts.board().post(
                "data", "The Kaggle description/evaluation pages are Kaggle's "
                        "UNFILLED TEMPLATE — any metric or format they state "
                        "belongs to the example, not this competition. Derive "
                        "the metric from the data and shipped code. See TASK.md.",
                src="harness")
        facts.board().post("data", "Task statement saved to TASK.md — read that "
                                   "rather than each fetching it again.",
                           src="harness")
    except Exception as e:  # noqa: BLE001
        print(f"[boot] overview failed: {type(e).__name__}: {e}", flush=True)

    mode, limit = submission_mode(args.slug)
    used = submissions_used_today(args.slug)
    QUOTA.update(limit=limit, used_today=used)
    trace.log("submission_mode", mode=mode, daily_limit=limit, used_today=used)
    if limit:
        allowed = max(0, min(args.max_submissions, limit - used))
        if allowed != args.max_submissions:
            print(f"[boot] daily cap {limit}, {used} already used today -> this "
                  f"run may send {allowed} (asked for {args.max_submissions})",
                  flush=True)
        budget.max_submissions = allowed
        if allowed == 0:
            facts.board().post(
                "env", f"today's {limit} submissions are already used up "
                       f"({used}/{limit}); nothing can reach the leaderboard "
                       f"this run. Local OOF is the only signal available.",
                src="harness")
    if mode == "csv":
        note = (f"This competition accepts a CSV submission directly — a Kaggle "
                f"kernel is NOT required. Train locally and upload; skip the "
                f"push/queue entirely. Daily submission limit: {limit}.")
    elif mode == "kernel":
        note = (f"This competition is kernel-only: submissions must come from a "
                f"notebook that trains in-kernel. Daily limit: {limit}.")
    else:
        note = ("Could not determine whether this competition is kernel-only; "
                "assume it is and produce a kernel.")
    SUBMIT_MODE["mode"] = mode
    print(f"[boot] submission mode: {mode} (daily limit {limit})", flush=True)
    facts.board().post("format", note, src="harness")
    run_recon(ws)


RUNLOG = """# {slug} — {stamp}

## Configuration
solvers {solvers} | deadline {deadline:.0f} min | cost cap ${cap:.0f} | model {model}
effort {effort} | mode {mode} | result disclosure {disclosure}
min_candidates {minc} | submissions allowed {allow}

## What happened
elapsed {elapsed:.0f} min | spent ${spent:.2f} | submissions made {subs}
{scores}
{lb}

## What this does and does not show
This was one run at one budget on one task. It is not a controlled comparison
with anything: no other configuration was run against it under the same
conditions, so nothing here separates the effect of the architecture from the
effect of the model, the budget, or the task. Treat the numbers as evidence
that this configuration produced this result once.

{caveats}
"""


def write_runlog(ws: Path, args, budget: Budget, trace: Tracer) -> None:
    """Record the run the way a teammate's notes do it — numbers first, then
    what the numbers cannot support. Five runs went by here with no controlled
    comparison between any of them and conclusions stated anyway."""
    import datetime
    try:
        scored = scored_candidates(ws, ALL_SOLVERS or [])
        lines = "\n".join(f"- {k}: {v:.6f} (out-of-fold, frozen split)"
                           for k, v in sorted(scored.items(), key=lambda x: -x[1]))
        sent = QUOTA.get("sent") or []
        lb = ("\n### Sent\n" + "\n".join(f"- {m}" for m in sent)) if sent else \
             "\n### Sent\nnothing reached the leaderboard."
        caveats = []
        if budget.submissions == 0:
            caveats.append("- No submission was made, so every score above is "
                           "unvalidated. Local numbers have diverged from the "
                           "leaderboard before.")
        if budget.cost_usd >= budget.max_cost_usd * 0.95:
            caveats.append("- The run stopped on budget, not on the clock, so it "
                           "was still improving when it ended.")
        st = json.loads((ws / "integrity.json").read_text()) if (
            ws / "integrity.json").exists() else {}
        if not st:
            caveats.append("- Nothing was sealed, so the scored files were not "
                           "tied to the submitted ones.")
        (ws / "RUNLOG.md").write_text(RUNLOG.format(
            slug=args.slug, stamp=datetime.datetime.now().isoformat(timespec="minutes"),
            solvers=args.solvers, deadline=args.deadline_min, cap=args.max_cost_usd,
            model=args.model, effort=args.effort, minc=args.min_candidates,
            mode=args.mode, disclosure=args.result_disclosure,
            allow=budget.max_submissions, elapsed=budget.elapsed() / 60,
            spent=budget.cost_usd, subs=budget.submissions,
            scores=lines or "(no candidate was ever scored)", lb=lb,
            caveats="\n".join(caveats) or "- No further caveats recorded."))
        print(f"[runlog] {ws / 'RUNLOG.md'}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[runlog] failed: {type(e).__name__}: {e}", flush=True)


def finalize(ws: Path, args, budget: Budget, trace: Tracer) -> None:
    r = subprocess.run([sys.executable, "-m", "native.scripts.promote",
                        "--show", "--workspace", str(ws)],
                       cwd=str(ROOT), capture_output=True, text=True)
    print("\n===== LKG =====\n" + (r.stdout or r.stderr))
    trace.log("final", lkg=r.stdout[:800], submissions=budget.submissions,
              cost=budget.cost_usd)
    try:
        from native.scripts.coordination import audit as coordination_audit
        coordination = coordination_audit(ws, ALL_SOLVERS)
        trace.log("coordination_audit", valid=coordination["valid"],
                  interpretation=coordination["interpretation"],
                  errors=coordination["errors"])
        print("[coordination] "
              f"{coordination['interpretation']} | valid={coordination['valid']} "
              f"| edges={len(coordination['collaboration_edges'])}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[coordination] audit failed: {type(exc).__name__}: {exc}",
              flush=True)
        trace.log("coordination_audit_failed",
                  err=f"{type(exc).__name__}: {exc}")
    if budget.submissions == 0:
        print("!! no submission was made — LKG above is the fallback to push by hand")
    # Solvers background their training and their watchers, which outlive the
    # sessions that started them: run 1's watcher submitted twice more eleven
    # minutes after this function printed its summary, and run 2 left six
    # processes holding GPUs on a shared machine.
    write_runlog(ws, args, budget, trace)
    n = reap_workspace_processes(ws)
    if n:
        print(f"[cleanup] killed {n} detached process(es) still running out of "
              f"the workspace", flush=True)
        trace.log("reaped", n=n)


async def run(args) -> None:
    _load_dotenv()
    if args.result_disclosure == "auto":
        args.result_disclosure = (
            "after-first" if args.mode == "clean-benchmark" else "live"
        )
    suffix = f"-{args.run_id}" if args.run_id else ""
    ws = ROOT / "workspace" / f"hearsay-{args.slug}{suffix}"
    if args.solvers == 1:
        names = ["solver_solo"]
    else:
        # Plain identities. They are not roles: each solver reads the task and
        # claims its own angle on the board.
        names = [f"solver_{c}" for c in "abcdefgh"[:args.solvers]]
    ALL_SOLVERS[:] = names
    if (ws / evidence.RUN_CONTRACT).exists() or (ws / "trace.jsonl").exists():
        raise FileExistsError(
            f"{ws} has already been started; use a fresh --run-id rather than "
            "overwriting or resuming sealed evidence"
        )
    for n in names:
        (ws / n / "out").mkdir(parents=True, exist_ok=True)

    contract = evidence.init_contract(
        ws, mode=args.mode, slug=args.slug, model=args.model,
        effort=args.effort, solvers=names, max_cost_usd=args.max_cost_usd,
        max_submissions=args.max_submissions, deadline_min=args.deadline_min,
    )

    _, limit = submission_mode(args.slug)
    used = submissions_used_today(args.slug)
    allowance = min(args.max_submissions, max(0, limit - used) if limit else
                    args.max_submissions)
    QUOTA.update({"limit": limit, "used_today": used, "sent": []})
    print(f"[boot] submission quota: {used}/{limit} already used today -> "
          f"this run may send {allowance}", flush=True)

    budget = Budget(deadline_s=args.deadline_min * 60,
                    max_submissions=allowance,
                    max_cost_usd=args.max_cost_usd)
    trace = Tracer(ws / "trace.jsonl")
    facts.init(ws)
    T.init_ctx(ws, args.slug, budget, trace, mode=args.mode)
    trace.log("quota", limit=limit, used_today=used, allowance=allowance)
    if allowance == 0:
        print("[boot] !! no submissions left today — this run cannot reach the "
              "leaderboard.", flush=True)

    print(f"launch hearsay | slug={args.slug} model={args.model} "
          f"solvers={names} effort={args.effort}\nworkspace={ws}")
    trace.log("launch", slug=args.slug, model=args.model, solvers=names,
              mode=args.mode, contract_sha256=contract["contract_sha256"])

    QUOTA["budget"] = budget
    bootstrap(ws, args, budget, trace)

    async def evaluator_task() -> None:
        if args.skip_evaluator:
            return
        try:
            await run_evaluator(ws, args, budget, trace)
        except Exception as e:  # noqa: BLE001
            print(f"[evaluator] FAILED: {type(e).__name__}: {e}", flush=True)
        # Without these two files the entire deterministic layer is inert:
        # nothing can be scored, so nothing can be promoted and LKG stays
        # empty however well the solvers do. That must be loud, not a warning
        # that scrolled past twenty minutes ago.
        # A malformed split is as bad as a missing one, and it fails later and
        # more confusingly — so check the contract, not just the filename.
        if (ws / "folds.json").exists():
            r = subprocess.run(
                [sys.executable, "-m", "native.scripts.checkfolds",
                 "--workspace", str(ws)],
                cwd=str(ROOT), capture_output=True, text=True)
            try:
                v = json.loads(r.stdout or "{}")
            except Exception:  # noqa: BLE001
                v = {}
            if v.get("ok"):
                print(f"[folds] {v.get('summary', '')}", flush=True)
                facts.board().post("data", f"validation split frozen: "
                                           f"{v.get('summary', '')}"[:300],
                                   src="harness")
            else:
                msg = (f"folds.json is malformed: {v.get('problems')} — nothing "
                       f"can be scored until it is fixed.")
                print(f"!! {msg}", flush=True)
                trace.log("folds_invalid", problems=v.get("problems"))
                facts.board().post("env", msg[:300], src="harness")

        missing = [f for f in ("metric.py", "folds.json") if not (ws / f).exists()]
        if missing:
            msg = (f"evaluator did not produce {missing} — scoring, promotion and "
                   f"LKG are all inert until they exist. Build them yourself "
                   f"before relying on any score.")
            print(f"!! {msg}", flush=True)
            trace.log("evaluator_incomplete", missing=missing)
            facts.board().post("env", msg, src="harness")
        else:
            facts.board().post(
                "env", "metric.py and folds.json are frozen and ready — score "
                       "candidates with native/scripts/evaluate.py.", src="harness")

    # The evaluator runs ALONGSIDE the solvers, not in front of them. In run 1 it
    # held the start gate for ten of the twenty-three minutes; nothing a solver
    # does first — reading the data, standing up a baseline, getting a legal
    # submission onto the board — needs the metric to exist yet.
    tasks = [asyncio.create_task(evaluator_task()),
             asyncio.create_task(submitter(ws, args, names, budget, trace)),
             asyncio.create_task(reconciler(ws, args, budget, trace)),
             asyncio.create_task(score_watcher(ws, args, budget, trace))]
    tasks += [asyncio.create_task(run_solver(n, ws, args, budget, trace, len(names)))
              for n in names]
    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True),
                               timeout=max(budget.remaining(), 60))
    except asyncio.TimeoutError:
        print("\n!! deadline reached — stopping solvers")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    print(f"\n===== RUN COMPLETE =====\n{budget_line(budget)}")
    print(f"tokens in={budget.tokens_in} out={budget.tokens_out}")
    finalize(ws, args, budget, trace)
    print(f"workspace: {ws}\ntrace: {trace.path}\nfacts: {ws / 'facts.jsonl'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--run-id", default="",
                    help="fresh immutable workspace suffix; required to repeat "
                         "a slug without overwriting earlier evidence")
    ap.add_argument("--mode", choices=evidence.MODES, default="competition",
                    help="competition may feed sparse leaderboard calibration "
                         "back; clean-benchmark seals it from solvers")
    ap.add_argument("--result-disclosure",
                    choices=["auto", "live", "after-first", "none"],
                    default="auto",
                    help="when evaluator-recomputed local scores reach peers; "
                         "auto=live in competition, after-first in clean mode")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default="high",
                    choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--solvers", type=int, default=3,
                    help="1 = the control (kernel ceiling, no board)")
    ap.add_argument("--deadline-min", type=float, default=120)
    ap.add_argument("--max-submissions", type=int, default=3,
                    help="ceiling for this run; the real allowance is this "
                         "capped by what the daily quota has left")
    ap.add_argument("--min-candidates", type=int, default=3,
                    help="scored candidates required before the harness spends "
                         "the first submission")
    ap.add_argument("--submit-fallback-frac", type=float, default=0.55,
                    help="fraction of the window after which the harness sends "
                         "the best it has even if --min-candidates is unmet; "
                         "waiting for all of them is how a stalled solver turns "
                         "into a zero")
    ap.add_argument("--max-cost-usd", type=float, default=100.0)
    ap.add_argument("--evaluator-cost", type=float, default=5.0)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--max-turns", type=int, default=250)
    ap.add_argument("--skip-evaluator", action="store_true")
    ap.add_argument("--max-lives", type=int, default=3,
                    help="sessions a solver may burn through; a crashed or "
                         "unreachable one is restarted from its files and the "
                         "board rather than lost for the rest of the run")
    ap.add_argument("--stall-seconds", type=float, default=240,
                    help="silence inside a single turn before the harness "
                         "interrupts it (run 2 lost 8m50s to one such turn)")
    ap.add_argument("--submit-poll-seconds", type=float, default=60,
                    help="how often the submitter re-checks its gates; scoring "
                         "is cached on the prediction file so a poll is cheap, "
                         "but not free")
    ap.add_argument("--review-timeout", type=float, default=90,
                    help="how long the submitter waits for the evaluator before "
                         "sending on the mechanical checks alone; a gate that "
                         "can hang is a way to score zero")
    ap.add_argument("--reconcile-seconds", type=float, default=45,
                    help="how often to check the harness's beliefs against what "
                         "Kaggle and the machine actually show")
    anyio.run(run, ap.parse_args())


if __name__ == "__main__":
    main()
