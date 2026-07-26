#!/usr/bin/env python3
"""multiagent-sdk launcher — the human's ONLY action.

    python -m magent.main --slug <competition> [--brief task.md]
        [--repos 2] [--deadline-min 120] [--max-submissions 3]
        [--max-cost-usd 30] [--rounds 8]

Claude Agent SDK kernel; Manager (main session, Fable 5) orchestrates
setup/designer/coder/tuner/verifier/aggregator/reporter subagents over parallel
solution repos. Harness enforces budgets via hooks; cost is recorded always.
See docs/DESIGN-MULTIAGENT.md.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path

import anyio

from claude_agent_sdk import (ClaudeAgentOptions, ClaudeSDKClient, HookMatcher,
                              AssistantMessage, ResultMessage, TextBlock,
                              ToolUseBlock)

from agent.orchestrator import Budget, Tracer
from agent.providers.anthropic_p import _load_dotenv
from . import tools as T
from .agents import AGENTS
from .prompts import MANAGER_SYSTEM, START_PROMPT, CONTINUE_PROMPT

ROOT = Path(__file__).resolve().parents[1]
MODEL = "claude-fable-5"          # user decision: Fable 5 only on this branch


def budget_line(b: Budget) -> str:
    return (f"[budget] elapsed={b.elapsed()/60:.1f}min "
            f"remaining={max(b.remaining(),0)/60:.1f}min "
            f"submissions={b.submissions}/{b.max_submissions} "
            f"llm_cost=${b.cost_usd:.2f}")


async def run(args) -> None:
    _load_dotenv()  # ANTHROPIC_API_KEY from repo .env
    ws = ROOT / "workspace" / f"magent-{args.slug}"
    ws.mkdir(parents=True, exist_ok=True)
    if args.brief:
        (ws / "BRIEF.md").write_text(Path(args.brief).read_text())

    budget = Budget(deadline_s=args.deadline_min * 60,
                    max_submissions=args.max_submissions,
                    max_cost_usd=args.max_cost_usd)
    trace = Tracer(ws / "trace.jsonl")
    T.init_ctx(ws, args.slug, budget, trace)

    # ---------------- hooks: the harness the LLM cannot bypass ----------------
    async def gate_submit(input_data, tool_use_id, context):
        if not budget.can_submit():
            return {"hookSpecificOutput": {
                "hookEventName": "PreToolUse", "permissionDecision": "deny",
                "permissionDecisionReason":
                    f"submission budget exhausted "
                    f"({budget.submissions}/{budget.max_submissions})"}}
        return {}

    async def trace_tool(input_data, tool_use_id, context):
        trace.log("tool", name=input_data.get("tool_name"),
                  input=str(input_data.get("tool_input"))[:300])
        return {}

    options = ClaudeAgentOptions(
        model=MODEL,
        system_prompt=MANAGER_SYSTEM.format(slug=args.slug, n_repos=args.repos),
        agents=AGENTS,
        mcp_servers={"ioai": T.SERVER},
        allowed_tools=["Task", "Read", "Write", "Edit", "Bash", "Glob", "Grep",
                       "TodoWrite"] + T.TOOL_NAMES,
        permission_mode="bypassPermissions",
        cwd=str(ws),
        setting_sources=[],            # no user/project settings bleed-in
        include_partial_messages=False,
        max_turns=args.max_turns,
        hooks={
            "PreToolUse": [HookMatcher(matcher="mcp__ioai__kaggle_submit",
                                       hooks=[gate_submit])],
            "PostToolUse": [HookMatcher(hooks=[trace_tool])],
        },
    )

    print(f"launch multiagent-sdk | slug={args.slug} model={MODEL} "
          f"repos={args.repos}\nworkspace={ws}")
    trace.log("launch", slug=args.slug, model=MODEL, repos=args.repos)

    async with ClaudeSDKClient(options=options) as client:
        last_turns = 99
        for rnd in range(1, args.rounds + 1):
            if budget.remaining() <= 0 or budget.cost_usd >= budget.max_cost_usd:
                prompt = ("BUDGET EXHAUSTED. Stop all new work NOW. If any valid "
                          "kernel is COMPLETE but unsubmitted, submit it; then "
                          "have reporter write REPORT.md immediately. "
                          + budget_line(budget))
                final_round = True
            else:
                tmpl = START_PROMPT if rnd == 1 else CONTINUE_PROMPT
                prompt = tmpl.format(slug=args.slug, budget=budget_line(budget))
                final_round = False

            print(f"\n===== ROUND {rnd} =====\n{budget_line(budget)}")
            trace.log("round", n=rnd, final=final_round)
            await client.query(prompt)

            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for blk in msg.content:
                        if isinstance(blk, TextBlock) and blk.text.strip():
                            print(f"  [manager] {blk.text.strip()[:400]}")
                            trace.log("manager", text=blk.text[:800])
                        elif isinstance(blk, ToolUseBlock):
                            name = blk.name
                            brief = str(blk.input)[:150]
                            print(f"  [call] {name}({brief})")
                elif isinstance(msg, ResultMessage):
                    cost = msg.total_cost_usd or 0.0
                    budget.cost_usd += cost
                    u = msg.usage or {}
                    budget.tokens_in += u.get("input_tokens", 0)
                    budget.tokens_out += u.get("output_tokens", 0)
                    last_turns = msg.num_turns
                    trace.log("result", round=rnd, cost=cost,
                              turns=msg.num_turns, err=msg.is_error)
                    print(f"  [round {rnd} done] turns={msg.num_turns} "
                          f"cost=${cost:.2f} total=${budget.cost_usd:.2f}")

            done = (ws / "REPORT.md").exists() and budget.submissions > 0
            if done or final_round:
                break
            # Pacing: a short round means the manager is waiting on something.
            # Give real work time to happen instead of burning continue-rounds.
            if last_turns <= 3:
                wait_s = min(120, max(30, budget.remaining() * 0.02))
                print(f"  [pacing] short round -> sleeping {wait_s:.0f}s")
                await asyncio.sleep(wait_s)

    print("\n===== RUN COMPLETE =====")
    print(budget_line(budget))
    print(f"tokens in={budget.tokens_in} out={budget.tokens_out}")
    print(f"workspace: {ws}\ntrace: {trace.path}")
    trace.log("complete", cost=budget.cost_usd, submissions=budget.submissions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--brief", default=None)
    ap.add_argument("--repos", type=int, default=2,
                    help="parallel solution repos (dev: keep small for cost)")
    ap.add_argument("--deadline-min", type=float, default=120)
    ap.add_argument("--max-submissions", type=int, default=3)
    ap.add_argument("--max-cost-usd", type=float, default=30.0)
    ap.add_argument("--rounds", type=int, default=60,
                    help="safety cap on start/continue rounds (wall-clock is "
                         "the real stop; short waiting rounds sleep, see pacing)")
    ap.add_argument("--max-turns", type=int, default=250,
                    help="max agent turns per round")
    args = ap.parse_args()
    anyio.run(run, args)


if __name__ == "__main__":
    main()
