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
from . import supervise as S
from . import tools as T
from .prompts import BOARD_MULTI, BOARD_SOLO, CONTINUE, CONTRACT, FIRST

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "claude-opus-5"
ALL_SOLVERS: list[str] = []
# Agents whose session stopped responding and could not be interrupted.
HUNG: set[str] = set()

REVIVAL = """
Your previous session ended before the run did — it hung, or it crashed. This is
a fresh one. You keep your working directory and the board; you do not keep the
reasoning that got you here, so read rather than assume.

Already in `{cwd}`:
{artifacts}

You posted these to the board earlier, which is the only record of what you were
attempting:
{mine}

Pick up from the artifacts. If `out/oof.npy` already scores well, protect it —
re-deriving what you had is the one thing that would make this restart a net
loss. {left:.0f} minutes remain.
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
        if re.search(r"kaggle\s+(competitions|c)\s+submit", cmd):
            return {"hookSpecificOutput": {
                "hookEventName": "PreToolUse", "permissionDecision": "deny",
                "permissionDecisionReason":
                    "You do not submit — the harness does, once the evaluator "
                    "has scored enough candidates to choose between. Write "
                    "out/submission.csv; check mcp__ioai__submission_status for "
                    "where the shared quota stands."}}
        # Knowledge from earlier runs travels through `memory_recall`, which is
        # distilled and reviewed. Reading a previous run's raw board instead is
        # not the same thing: on the timed-deps rerun a solver found the archive
        # I had made of the attempt before it, replayed its findings onto the
        # board as its own at minute 1.6, and the other two built on them. No
        # rule broken — it was our own run on the same task — but the run stops
        # being a measurement of the system, and nobody else can reproduce it.
        if re.search(r"(IOAI2026-agent/)?archive/\d{4}-\d{2}-\d{2}", cmd):
            return {"hookSpecificOutput": {
                "hookEventName": "PreToolUse", "permissionDecision": "deny",
                "permissionDecisionReason":
                    "That is an archive of a previous run. What earlier runs "
                    "learned reaches you through mcp__ioai__memory_recall, "
                    "which is distilled and checked; replaying a past board "
                    "verbatim makes this run unreproducible for anyone who "
                    "does not have that directory."}}
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

    async def trace_tool(input_data, tool_use_id, context):
        # 200 chars used to be the cap, which put the interesting half of every
        # long Bash command out of reach. The evaluator reads this file to
        # decide whether a kernel was ever pushed; a record that silently drops
        # the end of commands cannot answer that, and it answered "no" while a
        # kernel was running on Kaggle.
        trace.log("tool", solver=solver, name=input_data.get("tool_name"),
                  input=str(input_data.get("tool_input"))[:2000])
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
                              HookMatcher(matcher="Bash", hooks=[gate_bash])],
               "PostToolUse": [HookMatcher(hooks=[trace_tool])]},
    )


# Failures the run cannot work its way out of. A solver that is thinking, or
# stuck on a bad idea, is worth nudging; a solver whose account cannot pay for a
# token will answer the nudge the same way forever. On the second timed-deps
# rerun all three sat at `turns=1 $0.00 [no output]` for forty rounds in one
# minute while the harness printed "stalled — nudging", because nothing
# distinguished "no progress" from "no possible progress".
FATAL_ENV = (
    ("credit balance is too low", "the Anthropic account is out of credit"),
    ("insufficient_quota", "the Anthropic account is out of credit"),
    ("invalid x-api-key", "ANTHROPIC_API_KEY is not valid"),
    ("authentication_error", "authentication was refused"),
    ("oauth token has expired", "the OAuth token has expired — re-run "
                                "`claude setup-token`"),
    ("permission_error", "this key is not allowed to use the requested model"),
)
BROKEN: dict = {}


def note_env(text: str, solver: str) -> None:
    """Record a failure that no amount of retrying will fix."""
    low = text.lower()
    for needle, why in FATAL_ENV:
        if needle in low:
            if not BROKEN:
                print(f"\n!! {why}. Every agent will fail the same way, so "
                      f"there is nothing to wait for.\n!! seen from {solver}: "
                      f"{text.strip()[:200]}\n", flush=True)
            BROKEN.setdefault("why", why)
            BROKEN.setdefault("first", solver)
            BROKEN["seen"] = BROKEN.get("seen", 0) + 1
            return


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
                    note_env(blk.text, solver)
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
Two rounds have now produced no new or changed file under your `out/` and no
fact on the board. That is the shape of being stuck, not of thinking hard.

Drop whatever the current attempt is. If it is a long-running job, kill it — a
run that has told you nothing in two rounds will not start now. Then take the
cheapest step that produces something a script can read: `out/oof.npy` scored by
`native/scripts/evaluate.py`, or `out/submission.csv`, or a `failure` fact
saying concretely what does not work so nobody else spends time on it.

A worse candidate that exists beats a better one that does not.
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
    peers = ("You are the only solver on this problem." if n == 1 else
             f"Your working directory is yours alone; {n} solvers are working "
             f"on this problem in parallel, none of them told what to do.")
    append = CONTRACT.format(slug=args.slug, peers=peers,
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
                    prompt = tmpl.format(slug=args.slug, budget=budget_line(budget))
                    if BROKEN:
                        print(f"  [{solver}] giving up: {BROKEN['why']}",
                              flush=True)
                        trace.log("env_broken", solver=solver, **BROKEN)
                        clean = True
                        break
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


EVALUATOR = """You own measurement for this run, and nothing else. You are not
solving the task, you are not judging whether anyone's approach is any good, and
you are not exploring the data for its own sake.

The job has two parts. First, once: compile the ruler. Then, for the rest of the
run: decide what gets submitted, when the harness queues a candidate for you.
Between the two you have nothing to do — that is expected.

Start with the ruler, and be quick about it: the solvers are working without it
until it exists. Investigate the data only as far as you need to pin down the
metric and a sound split. The deterministic reconnaissance has already run and
its findings are in `recon.md` and on the board, so do not repeat it — hunting
for leaks and duplicate frames is not your job.

Two artifacts, in the workspace root ({ws}):

1. `metric.py` — the official metric of `{slug}`, exposing
   `def score(y_true, y_pred) -> float | dict`. Before you declare it finished,
   self-test it on synthetic labels whose correct value you worked out by hand,
   and show that test. A metric that is subtly wrong is worse than no metric:
   everything downstream is measured with it and nothing will notice.

   Corroborate the definition — do not just transcribe the evaluation page.
   Mirrored competitions sometimes ship Kaggle's unfilled page template, which
   *demonstrates* a metric and a submission format with example text; a page
   like that reads as specific and confident while describing a different
   competition entirely. `kaggle_overview` flags the boilerplate it can detect,
   but the check is a heuristic. Whatever the page says, verify it against
   `sample_submission.csv`, the actual data, and what the task obviously is. If
   the page and the data disagree, the data wins — say so on the facts board.

2. `folds.json` — the split every candidate in this run is scored on. How to cut
   it is your call and depends entirely on this task, so load the playbook
   rather than working from first principles:

       skill_load(name="validation-split")

   It covers the file's contract, how to pick k against your training budget,
   and which structures in the data a shuffled split would leak through. The
   harness insists on three things only: the file is well formed
   (`{py} -m native.scripts.checkfolds --workspace {ws}`), it covers exactly the
   evaluation units, and it is never re-cut once anything has been scored on it.

   Post the scheme and your reasoning to the board when it is frozen. Every
   number in this run comes from it, so everyone should know what it is.

If the shipped materials genuinely do not pin the metric down, do NOT stall
trying to prove it. Implement the most defensible reading, state the assumption
at the top of `metric.py`, post it to the facts board as a `data` fact, and
finish. A frozen, documented, slightly-uncertain ruler that everyone shares beats
no ruler at all — and an unresolved question on the board is something the
solvers can act on, whereas your silence is not.

Then report what you built and wait. Review requests arrive as new messages.
"""


REVIEW = """
A candidate is queued for submission and the decision is yours. The allowance is
a handful for the whole team and does not renew, so a slot spent badly is gone.

You are the right one to hold this because you are the only agent here with no
candidate of its own. When the solvers held it, all three bought insurance
inside four minutes and the best model of the run — twelve minutes later — had
nothing left to travel on.

The harness has already checked what a script can check: quota, throttling, and
that this candidate has the best score on the shared folds. What it cannot check
is whether this submission is *valid for this competition*, and whether spending
a slot now is better than waiting for the run to improve on it.

    candidate:  {cand}
    workspace:  {ws}
    files:      {cand}/out/submission.csv, {cand}/out/oof.npy{extra}

The mechanical checks have run. They are heuristics written without knowing this
competition's rules, so treat them as observations to adjudicate, not findings:

    fatal (bad under any rules):  {fatal}
    warnings (depends on rules):  {warn}

A warning is not a defect until you say it is. The last run's warnings were
"contains negative values" and "no sample_submission to compare against" — on
that task -1 was the background label and the mirror simply shipped no sample,
so both were correct observations and neither was a problem.

ANYTHING ABOUT KAGGLE, ASK KAGGLE. `mcp__ioai__kaggle_kernel_status` and
`mcp__ioai__kaggle_submissions` answer authoritatively in one call. Do not
settle a question about the world by parsing `trace.jsonl`: it records what our
agents asked for, not what happened, it does not see work the harness itself
did, and a `grep` over it will happily match your own earlier commands echoed
back. On the timed-deps run this exact mistake held a candidate for twelve
minutes — "zero push calls in the entire run", re-confirmed three times, each
re-reading raising confidence without adding information — while the kernel sat
COMPLETE on Kaggle, pushed by the harness. Absence of evidence in a log we write
is not evidence of absence. One status call would have settled it.

Two things are yours:

VALIDITY. Read the organisers' own grading code if you have it, and check what
voids a submission silently: value ranges, encoding, id alignment, row order,
units, whether a column means what its name suggests. On the chicken task a
single negative pixel zeroed an entire submission, and only the grading code
said so.

TIMING. {left} slots remain and {elapsed:.0f} of {total:.0f} minutes are gone.
Holding a slot back is right when the run is young and clearly still improving;
it is wrong once time is short, because an unspent slot scores nothing. Do not
re-judge the model — that is measured, and this is already the best-scoring
candidate.

Write `review/{cand}.json` — exactly `{{"verdict": "APPROVE"|"HOLD"|"REJECT",
"reason": "<one line>"}}`.

    APPROVE  send it
    HOLD     valid, but worth waiting for something better
    REJECT   broken; say specifically what, so the solver can fix it

Then post one fact to the board: `format` if you found a validity problem
everyone should avoid, otherwise `claim` describing what you checked.

Be quick — the submitter proceeds on the mechanical checks alone if you take too
long, which wastes your judgement rather than using it.
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
        if budget.remaining() <= 0 or BROKEN:
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


def integrity_check(ws: Path, candidate: str) -> dict:
    """Provenance is fatal — a file that moved after scoring makes the score a
    lie. Order-dependence is a warning: ordered data is legitimate, exploiting
    the ordering is not, and only someone who knows the task can tell which."""
    from native.scripts.integrity import (check_order_dependence,
                                          check_provenance)
    try:
        return {"fatal": check_provenance(ws, candidate),
                "warnings": check_order_dependence(ws, candidate)}
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
        await client.query("Compile the ruler now, then report and wait for review requests.")
        drainer = asyncio.create_task(drain(client, "evaluator", budget, trace))
        watch = asyncio.create_task(
            watchdog_turn(client, "evaluator", drainer, args, trace))
        await drainer
        watch.cancel()
        # Now serve reviews until the run ends. The same turn watchdog applies:
        # the gate is watched exactly like the solvers, because a gate that
        # hangs is a way to score zero rather than a way to avoid a bad score.
        await review_loop(ws, args, budget, trace)


def announce(text: str, kind: str = "env", why: str = "") -> None:
    """Put something on the board, and complain loudly if it does not land.

    The board rejects anything over 300 characters — a rule written for agents,
    to stop status reports arriving dressed as facts. The harness then wrote a
    356-character announcement of its own and never checked the return value, so
    the single most important line of a run — that the competition was
    kernel-only and solvers had to build `out/kernel/` — was dropped in silence.
    Three solvers spent two hours producing candidates that could not be sent.
    """
    text = " ".join(str(text).split())
    if len(text) > facts.MAX_TEXT:
        print(f"!! announcement about {why or kind} is {len(text)} chars, over "
              f"the board's {facts.MAX_TEXT} — truncating. Say it shorter.",
              flush=True)
        text = text[:facts.MAX_TEXT - 1]
    r = facts.board().post(kind, text, src="harness")
    if r.startswith("[rejected]"):
        print(f"!! the board refused a harness announcement about "
              f"{why or kind}: {r}", flush=True)


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


def submissions_remaining(slug: str) -> int | None:
    """Submissions still allowed today, or None if it cannot be established.

    Note this is *remaining*, already net of what has been sent — not a daily
    total to subtract usage from. Mixing the two is easy and would silently
    halve the allowance.

    `competitions_list` cannot see private competitions, so the limit it
    reports for a real task is zero — and a zero limit quietly collapsed the
    run's allowance to whatever `--max-submissions` happened to default to. One
    run planned for three submissions against a true allowance of fifty, and
    nothing in the log said the number was a guess.

    `competitions submission-limits` does answer for private competitions, and
    the task statement names it outright. It is CLI-only — there is no Python
    binding — which is how a probe written against the API missed it.
    """
    import re
    import subprocess
    try:
        r = subprocess.run([R._kaggle_bin(), "competitions",
                            "submission-limits", slug],
                           capture_output=True, text=True, timeout=90)
        m = re.search(r"Remaining today:\s*(\d+)", r.stdout or "")
        if m:
            return int(m.group(1))
        why = ((r.stdout or "") + (r.stderr or "")).strip()[:200]
    except Exception as e:  # noqa: BLE001
        why = f"{type(e).__name__}: {e}"
    print(f"!! could not read today's submission limit ({why}) — falling back "
          f"to --max-submissions, which is a guess, not the quota", flush=True)
    return None


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


# Kaggle runs at most two GPU kernels per account at once, and we run three
# solvers. Without a lease the third push simply queues, and a queued kernel
# still occupies the slot it is waiting on — which is how a team ends up with
# one effective slot instead of two.
GPU_SLOTS = 2
_LEASES: dict[str, float] = {}
LEASE_TTL_S = 40 * 60


def gpu_lease(action: str, who: str) -> str:
    now = time.time()
    for k, t in list(_LEASES.items()):
        if now - t > LEASE_TTL_S:
            del _LEASES[k]          # a holder that died must not hold forever
    if action.startswith("rel"):
        _LEASES.pop(who, None)
        return f"released. {GPU_SLOTS - len(_LEASES)} of {GPU_SLOTS} free."
    if who in _LEASES:
        return f"you already hold a slot ({GPU_SLOTS - len(_LEASES)} free)."
    if len(_LEASES) >= GPU_SLOTS:
        return (f"WAIT — both GPU slots are taken by {sorted(_LEASES)}. Do "
                f"something else and ask again; pushing now would queue and "
                f"tie up a slot without running.")
    _LEASES[who] = now
    return f"granted. {GPU_SLOTS - len(_LEASES)} of {GPU_SLOTS} still free."


SUBMIT_MODE = {"mode": "csv"}
# A Kaggle kernel may run for thirty minutes; leave room to still call submit
# afterwards, because that call is what has to land before the deadline.
KERNEL_MAX_S = 35 * 60
SUBMIT_RESERVE_S = 5 * 60
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
                 "--workspace", str(ws)],
                **{"cwd": str(ROOT), "capture_output": True, "text": True})
            verdict = ("local is conservative — trust it" if gap >= -0.002 else
                       "LOCAL IS OPTIMISTIC — stop choosing on out-of-fold "
                       "scores alone")
            print(f"  [confirmed] {p['candidate']} local {p['local']:.5f} -> "
                  f"leaderboard {hit['score']:.5f} (gap {gap:+.5f})", flush=True)
            facts.board().post(
                "result", f"{p['candidate']}: local {p['local']:.5f} -> "
                          f"leaderboard {hit['score']:.5f}, gap {gap:+.5f}. "
                          f"{verdict}."[:300], src="harness")
REVIEW_Q: asyncio.Queue = asyncio.Queue()


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
    integ = await asyncio.to_thread(integrity_check, ws, cand)
    mech["fatal"] = list(mech.get("fatal", [])) + integ["fatal"]
    mech["warnings"] = list(mech.get("warnings", [])) + integ["warnings"]
    mech["ok"] = not mech["fatal"]

    if REVIEWER.get("client") is None:
        if not mech.get("ok"):
            facts.board().post(
                "format", f"{cand} not submitted — {'; '.join(mech.get('fatal', []))}"
                          [:300], src="harness")
            return False, f"fatal format problem: {mech.get('fatal')}"
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
            if frac >= args.submit_fallback_frac:
                trace.log("hold_overridden", candidate=cand, frac=round(frac, 2))
                facts.board().post(
                    "env", f"{frac:.0%} of the window gone — sending {cand} over "
                           f"the evaluator's hold. An unspent slot scores "
                           f"nothing.", src="harness")
                return True, (f"hold overridden at {frac:.0%} of the window "
                              f"(evaluator wanted to wait: {reason})")[:160]
            return False, f"evaluator holding the slot: {reason}"[:160]
        # Gate 1 had a point past which it could not refuse, and gate 3 was
        # given one after the lack of it cost a run. HOLD got one too. REJECT
        # never did — so an evaluator wrong about *why* it was rejecting could
        # still run the clock out, and on the timed-deps run it spent twelve
        # minutes rejecting a candidate whose kernel had already finished.
        if frac >= args.submit_fallback_frac and mech.get("ok"):
            trace.log("reject_overridden", candidate=cand, frac=round(frac, 2))
            announce(f"{frac:.0%} of the window gone and the mechanical checks "
                     f"pass — sending {cand} over the evaluator's rejection.",
                     kind="env", why="reject overridden")
            return True, (f"reject overridden at {frac:.0%}: mechanical checks "
                          f"pass (evaluator said: {reason})")[:160]
        facts.board().post(
            "format", f"evaluator REJECTED {cand}: {reason}"[:300], src="evaluator")
        return False, f"evaluator rejected: {reason}"

    trace.log("review_timeout", candidate=cand, waited=waited)
    if not mech.get("ok"):
        return False, (f"evaluator silent for {args.review_timeout:.0f}s and the "
                       f"file has a fatal problem: {mech.get('fatal')}")
    return True, (f"evaluator did not answer in {args.review_timeout:.0f}s; "
                  f"sent on mechanical checks alone")


def send_candidate(ws: Path, candidate: str, csv: Path, msg: str, ctx,
                   trace: Tracer, poll_s: int = 20,
                   budget: Budget | None = None) -> str:
    """Send one candidate, by whichever route this competition actually accepts.

    IOAI proper is kernel-only — the submitted script has to train in-kernel —
    so a submitter that can only upload a CSV cannot submit at all on the day
    that counts. The 2025 mirrors happen to take CSVs, which is what we have
    been testing against, and that difference had gone unnoticed.

    Kernel mode carries a gate that CSV mode does not: a candidate with a fine
    out-of-fold score is still worth nothing if its notebook crashes on Kaggle.
    So push, wait for it to actually finish, and only submit a version that ran.
    """
    # Only take the CSV shortcut when we positively established it is allowed.
    # `competitions_list` does not return private competitions, so every task on
    # the day that counts will come back "unknown" — and defaulting that to CSV
    # would mean not submitting at all on a kernel-only competition, which is
    # what IOAI proper is. The docstring above already claimed unknown fell
    # through to the strict path; the condition said otherwise.
    if SUBMIT_MODE["mode"] == "csv":
        return str(R.kaggle_submit({"csv_path": str(csv), "message": msg}, ctx))

    kdir = ws / candidate / "out" / "kernel"
    if not (kdir / "kernel-metadata.json").exists():
        # Tell them. The harness knew for two hours why it could not submit and
        # kept it in a log nobody reads; the solvers went on polishing a score
        # that had no route to the leaderboard.
        announce(f"{candidate} cannot be submitted: this is a code competition "
                 f"and it has no out/kernel/kernel-metadata.json. Build the "
                 f"kernel dir — see skill kaggle-submission.", kind="format",
                 why="missing kernel dir")
        return f"[skip] {candidate} has no out/kernel/kernel-metadata.json"

    # A kernel may run for up to thirty minutes. Waiting only ten, as this did,
    # means abandoning every submission whose training is longer than that.
    # What actually has to happen before the deadline is the `submit` call —
    # the organisers confirmed scoring may finish after it — so wait as long as
    # there is time to still submit, and no longer.
    left = (budget.remaining() if budget else 3600) - SUBMIT_RESERVE_S
    waits = max(3, int(min(KERNEL_MAX_S, left) / poll_s))
    if left <= 0:
        return (f"[skipped] {(budget.remaining() if budget else 0)/60:.0f} min "
                f"left, not enough to push a kernel and still submit in time")

    push = str(R.kaggle_push_kernel({"kernel_dir": str(kdir)}, ctx))
    trace.log("kernel_push", candidate=candidate, out=push[:300])
    ref = ver = None
    # Kaggle answers a push with a sentence and a URL:
    #   "Kernel version 1 successfully pushed.  Please check progress at
    #    https://www.kaggle.com/code/qam1ng/ioai-t1-timeddeps-12-solver-a"
    # A bare `([\w-]+/[\w-]+)` finds `com/code` in that URL long before it
    # reaches the owner and slug, so the poll below asked Kaggle about a kernel
    # that does not exist, never saw COMPLETE, and timed out 35 minutes later
    # while the real kernel had finished in 68 seconds. Read the slug from the
    # path we asked for instead, and use the URL only to confirm it.
    try:
        meta = json.loads((kdir / "kernel-metadata.json").read_text())
        ref = str(meta.get("id") or "").strip() or None
    except Exception:  # noqa: BLE001
        ref = None
    m = re.search(r"kaggle\.com/code/([\w-]+/[\w-]+)", push)
    if m and m.group(1) != ref:
        # The metadata is what we asked for; the URL is what Kaggle made. If
        # they disagree, Kaggle wins — it may have renamed a clashing slug.
        print(f"!! kernel id {ref!r} but Kaggle returned {m.group(1)!r}; "
              f"following Kaggle", flush=True)
        ref = m.group(1)
    if not ref:
        return (f"[skip] pushed {candidate} but could not work out the kernel "
                f"ref from {push[:120]!r} — not polling a guess")
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
        best = (st or {}).get("local")
        ok, why = should_submit(
            n_scored=len(scored), frac=frac, best=best, sent_best=sent_best,
            sent=budget.submissions, allowance=budget.max_submissions,
            min_candidates=args.min_candidates,
            fallback_frac=args.submit_fallback_frac)
        if not ok:
            continue
        csv = ws / "LKG" / "local" / "out" / "submission.csv"
        if not csv.exists():
            continue

        msg = (f"harness: {best['candidate']} OOF {best['mean']:.5f} "
               f"±{best['std']:.3f} ({len(scored)} scored, {why})")
        cand = best["candidate"]
        ok, note = await clear_for_submission(ws, cand, args, trace, frac)
        if not ok:
            print(f"  [submit] {cand} held back — {note}", flush=True)
            trace.log("submit_blocked", candidate=cand, reason=note)
            continue

        ctx = R.Ctx(workspace=ws, slug=args.slug, budget=budget, trace=trace)
        try:
            res = str(send_candidate(ws, cand, csv, f"{msg}; {note}", ctx,
                                     trace, budget=budget))
        except Exception as e:  # noqa: BLE001
            res = f"[submit error] {type(e).__name__}: {e}"
        sent_best = best["mean"]
        QUOTA["sent"].append(msg)
        # Remember what we sent and what we thought it was worth, so the score
        # coming back can be compared against it.
        PENDING.append({"candidate": best["candidate"], "local": best["mean"],
                        "message": msg, "at": time.time()})
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


def publish_results(ws: Path, names: list[str], trace: Tracer) -> None:
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
    changed = any(abs(scored[k] - _PUBLISHED.get(k, -1)) > 1e-6 for k in scored)
    if not changed:
        return
    _PUBLISHED.update(scored)
    parts = []
    for k, v in sorted(scored.items(), key=lambda kv: -kv[1]):
        what = (claim_of(ws, k) or k)[:60]
        parts.append(f"{what} → {v:.4f}")
    facts.board().post("result", "verified on the shared folds: "
                       + " | ".join(parts), src="harness")
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

        # Put the measured number on the board. Which directions have oil in
        # them is the most useful thing anyone here knows, and withholding it
        # only means the others keep drilling dry holes. It goes out as a
        # `result`, which solvers cannot post themselves: a score is comparable
        # only if it came from the shared folds and the frozen metric.
        try:
            d = json.loads(line[0])
        except Exception:  # noqa: BLE001
            continue
        if d.get("mean") is None:
            continue
        prev = _POSTED.get(n)
        if prev is not None and abs(prev - d["mean"]) < 1e-9:
            continue
        _POSTED[n] = d["mean"]
        # Seal the files this number came from. Nothing else ties the scored
        # oof to the submitted csv, so a swap between the two would be silent.
        try:
            from native.scripts.integrity import record as _seal
            _seal(ws, n, d["mean"])
        except Exception:  # noqa: BLE001
            pass
        move = "" if prev is None else f" (was {prev:.4f})"
        facts.board().post(
            "result", f"{n} now scores {d['mean']:.4f} +/-{d.get('std', 0):.3f} "
                      f"on the shared folds{move}. "
                      f"{'best so far' if d.get('decision') == 'PROMOTED' else 'below the current best'}.",
            src="harness")


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
        note = ("Submission mode unknown — the API cannot see private "
                "competitions and every real task is one. Treat this as "
                "kernel-only: build out/kernel/, with the wheel dataset and "
                "setup_ioai_env block the task description specifies.")
    SUBMIT_MODE["mode"] = mode
    print(f"[boot] submission mode: {mode} (daily limit {limit})", flush=True)
    announce(note, kind="format", why="submission mode")
    run_recon(ws)


RUNLOG = """# {slug} — {stamp}

## Configuration
solvers {solvers} | deadline {deadline:.0f} min | cost cap ${cap:.0f} | model {model}
effort {effort} | min_candidates {minc} | submissions allowed {allow}

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
    ws = ROOT / "workspace" / f"hearsay-{args.slug}"
    if args.solvers == 1:
        names = ["solver_solo"]
    else:
        # Plain identities. They are not roles: each solver reads the task and
        # claims its own angle on the board.
        names = [f"solver_{c}" for c in "abcdefgh"[:args.solvers]]
    ALL_SOLVERS[:] = names
    for n in names:
        (ws / n / "out").mkdir(parents=True, exist_ok=True)

    _, limit = submission_mode(args.slug)
    used = submissions_used_today(args.slug)
    # What Kaggle says is left, asked directly. Falling back to the public-API
    # total minus our own count only matters for the 2025 mirrors; for a real
    # (private) task the CLI is the only path that answers at all.
    left = submissions_remaining(args.slug)
    if left is None:
        left = max(0, limit - used) if limit else None
    # --max-submissions is a ceiling we may impose on ourselves, not the quota.
    # It used to be the only number in play, so when the quota probe returned
    # nothing its default silently *became* the quota: a run with fifty
    # submissions available planned for three.
    allowance = left if left is not None else args.max_submissions
    if args.max_submissions:
        allowance = min(allowance, args.max_submissions)
    QUOTA.update({"limit": left, "used_today": used, "sent": []})
    src = "kaggle says" if left is not None else "no quota reading — guessing"
    print(f"[boot] submissions: {src} {left if left is not None else '?'} left "
          f"today ({used} sent already) -> this run may send {allowance}",
          flush=True)

    budget = Budget(deadline_s=args.deadline_min * 60,
                    max_submissions=allowance,
                    max_cost_usd=args.max_cost_usd)
    trace = Tracer(ws / "trace.jsonl")
    facts.init(ws)
    T.init_ctx(ws, args.slug, budget, trace)
    trace.log("quota", limit=limit, used_today=used, allowance=allowance)
    if allowance == 0:
        print("[boot] !! no submissions left today — this run cannot reach the "
              "leaderboard.", flush=True)

    print(f"launch hearsay | slug={args.slug} model={args.model} "
          f"solvers={names} effort={args.effort}\nworkspace={ws}")
    trace.log("launch", slug=args.slug, model=args.model, solvers=names)

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

    if BROKEN:
        # Say it once more at the end. The line that mattered scrolled past
        # forty rounds ago, and a run that ends with "RUN COMPLETE" over an
        # empty workspace reads as a modelling failure rather than an unpaid
        # bill.
        print(f"\n===== RUN ABORTED: {BROKEN['why']} =====")
        print("No agent could run, so nothing here reflects on the system. "
              "Fix the credential and relaunch; the workspace is untouched.")
    print(f"\n===== RUN COMPLETE =====\n{budget_line(budget)}"
          if not BROKEN else budget_line(budget))
    print(f"tokens in={budget.tokens_in} out={budget.tokens_out}")
    finalize(ws, args, budget, trace)
    print(f"workspace: {ws}\ntrace: {trace.path}\nfacts: {ws / 'facts.jsonl'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default="high",
                    choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--solvers", type=int, default=3,
                    help="1 = the control (kernel ceiling, no board)")
    ap.add_argument("--deadline-min", type=float, default=120)
    ap.add_argument("--max-submissions", type=int, default=0,
                    help="optional ceiling on top of the daily quota; 0 (the "
                         "default) means let the quota decide. Only used as "
                         "the allowance itself when the quota cannot be read")
    ap.add_argument("--min-candidates", type=int, default=3,
                    help="scored candidates required before the harness spends "
                         "the first submission")
    ap.add_argument("--submit-fallback-frac", type=float, default=0.55,
                    help="fraction of the window after which the harness sends "
                         "the best it has even if --min-candidates is unmet; "
                         "waiting for all of them is how a stalled solver turns "
                         "into a zero")
    ap.add_argument("--max-cost-usd", type=float, default=40.0)
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
