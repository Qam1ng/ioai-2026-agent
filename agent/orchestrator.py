"""Orchestrator: phase state machine + agent loop + budgets + trace.

The harness enforces what must never go wrong (budgets, gates, tracing);
the agent decides everything that needs intelligence (via tools).
See DESIGN.md for the full architecture.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .providers.base import Provider, ToolResult
from .tools.registry import SCHEMAS, FNS, Ctx

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
class Budget:
    def __init__(self, deadline_s: float, max_submissions: int, max_cost_usd: float):
        self.t0 = time.time()
        self.deadline_s = deadline_s
        self.max_submissions = max_submissions
        self.max_cost_usd = max_cost_usd
        self.submissions = 0
        self.cost_usd = 0.0
        self.tokens_in = 0
        self.tokens_out = 0

    def elapsed(self): return time.time() - self.t0
    def remaining(self): return self.deadline_s - self.elapsed()
    def can_submit(self): return self.submissions < self.max_submissions
    def note_submit(self): self.submissions += 1

    def note_llm(self, usage: dict, cost: float):
        self.tokens_in += usage.get("input_tokens", 0)
        self.tokens_out += usage.get("output_tokens", 0)
        self.cost_usd += cost

    def exhausted(self) -> str | None:
        if self.remaining() <= 0: return "wall-clock deadline reached"
        # 0 表示不设费用上限。Claude Max 订阅路线没有可用的逐次 API 费用，
        # 正式比赛也不能因为默认值为 0 而在第一轮前被误判为预算耗尽。
        if self.max_cost_usd and self.cost_usd >= self.max_cost_usd:
            return "LLM cost budget reached"
        return None

    def status_line(self) -> str:
        return (f"[budget] elapsed={self.elapsed()/60:.1f}min "
                f"remaining={max(self.remaining(),0)/60:.1f}min "
                f"submissions={self.submissions}/{self.max_submissions} "
                f"llm=${self.cost_usd:.2f} (in={self.tokens_in} out={self.tokens_out})")


class Tracer:
    def __init__(self, path: Path):
        self.path = path

    def log(self, kind: str, **kw):
        rec = {"t": round(time.time(), 2), "kind": kind, **kw}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


# --------------------------------------------------------------------------- #
@dataclass
class Phase:
    name: str
    goal: str                       # injected as the phase instruction
    gate: "callable"                # (workspace) -> None | error string
    max_steps: int = 40             # LLM turns cap for the phase


def _gate_task_card(ws: Path):
    f = ws / "TASK_CARD.md"
    if not f.exists():
        return "TASK_CARD.md was not created"
    txt = f.read_text().lower()
    for needed in ("metric", "submission"):
        if needed not in txt:
            return f"TASK_CARD.md lacks a '{needed}' section"
    return None


def _gate_plan(ws: Path):
    f = ws / "PLAN.md"
    if not f.exists():
        return "PLAN.md was not created"
    txt = f.read_text().lower()
    if "valid" not in txt and "cv" not in txt:
        return "PLAN.md lacks a verification/CV design section"
    return None


def _gate_scaffold(ws: Path):
    missing = [n for n in ("scaffold",) if not (ws / n).exists()]
    if missing:
        return f"missing scaffold/ directory with pipeline code"
    if not (ws / "scaffold" / "GATE_OK").exists():
        return ("scaffold self-test not passed: create scaffold/GATE_OK by "
                "running your metric self-test + submission format check "
                "successfully (write the file only when both pass)")
    return None


def _gate_baseline(ws: Path):
    if not (ws / "BASELINE.md").exists():
        return ("BASELINE.md missing — record the baseline submission: kernel ref/"
                "version, submission result, and leaderboard score once available")
    return None


def _gate_none(ws: Path):
    return None


PHASES = [
    Phase("INGEST", """Read and understand the task.
- Use kaggle_download to fetch the data (skips if present), then explore input/ (list_dir, read_file, run_python).
- Recall relevant lessons (memory_recall) and list skills (skill_list); load any that fit.
- Produce TASK_CARD.md at workspace root containing at least: task type; the official Metric (exact formula/definition); data files and schema; the Submission format (exact columns/order/values); constraints (budgets, rules); class/label structure if any.
Call phase_complete when TASK_CARD.md is written.""", _gate_task_card),

    Phase("ANALYZE", """Exploratory analysis to ground your plan.
- Use run_python for EDA: sizes, distributions, imbalance, any provided model/checkpoint internals.
- Append an '## EDA findings' section to TASK_CARD.md with concrete numbers and any systematic weaknesses you find.
Call phase_complete when the EDA section is written.""", _gate_task_card),

    Phase("PLAN", """Write PLAN.md:
1. Ranked candidate approaches (strong simple baseline first; note expected risk/cost of each).
2. Validation design: how you'll estimate the official metric locally so it TRACKS the leaderboard (prefer stratified k-fold; state fold count; note variance handling and a significance rule for accepting improvements).
3. Budget allocation: how you'll spend remaining time and submissions (early floor submission is mandatory).
4. Risks and fallbacks.
Call phase_complete when PLAN.md is written.""", _gate_plan),

    Phase("SCAFFOLD", """Build the reusable pipeline under scaffold/ (agent-written, not hand-me-downs):
- data loading; feature/model pipeline; local implementation of the official metric; k-fold CV harness; submission builder (for code competitions: a kernel/ dir with kernel-metadata.json that runs on Kaggle and writes submission.csv).
- Self-test: synthetic-label test proving your metric implementation matches the official definition, AND a submission-format validator against the sample. When BOTH pass, write the file scaffold/GATE_OK (content: brief test output).
Call phase_complete after GATE_OK is written.""", _gate_scaffold),

    Phase("BASELINE", """Produce and actually submit a floor solution EARLY:
- Simplest valid approach through your scaffold; push the kernel (kaggle_push_kernel), wait for COMPLETE (kaggle_kernel_status; poll with run_bash sleep), then kaggle_submit.
- Record kernel ref/version + submission status + score (kaggle_submissions) in BASELINE.md. This establishes the always-valid-submission floor and calibrates local CV vs leaderboard.
Call phase_complete once submitted (score may still be pending; note it).""", _gate_baseline),

    Phase("EXPERIMENT", """Iterate to improve the score. Each iteration:
1. Pick ONE hypothesis grounded in PLAN + your experiment ledger (runs/ledger.json — create/update it: name, code path, k-fold mean±std, notes).
2. Implement + evaluate with the k-fold harness (never a single small holdout).
3. Accept a new best ONLY if the mean gain is significant vs fold variance (your PLAN's rule). Beware phantom failures on classes with 1-2 validation samples.
4. On errors: fix and retry (max 2), then move on.
5. When a significantly better solution exists and submission budget allows, push kernel + submit, then recalibrate local-vs-LB in the ledger.
Respect remaining time (the harness shows budget each turn): stop opening new directions in the last third; consolidate.
Call phase_complete when budget guidance says converge or no promising hypotheses remain.""", _gate_none, 80),

    Phase("FINALIZE", """Converge:
- Retrain the ledger's best on all data via the scaffold; validate submission format; push + submit the final kernel if it beats the last submitted score locally by a significant margin (else keep the existing best submission).
- Write FINAL.md: chosen solution, why, expected vs observed scores.
Call phase_complete when done.""", _gate_none),

    Phase("REPORT", """Write REPORT.md — the technical report required by the rules:
- Approach summary; experiment table (from runs/ledger.json); score trajectory (local CV and leaderboard); tools/resources used (the harness traces all calls; summarize honestly); efficiency (LLM tokens/cost from the budget line, iteration count); what worked / what didn't.
- Save one durable lesson via memory_write for future tasks.
Call phase_complete when REPORT.md is written.""", _gate_none),
]


# --------------------------------------------------------------------------- #
SYSTEM = """You are an autonomous ML engineer competing in a Kaggle competition, running with NO human available — never ask questions; decide and act with tools.

Mission: maximize the official competition metric within budget, then document honestly. You work phase by phase; the current phase's goal is given in the conversation, and the harness enforces gates between phases.

Ground rules:
- Competition slug: {slug}. Workspace is your cwd for all tools; data in input/.
- If BRIEF.md exists in the workspace, it is the official task statement supplied at launch — treat it as authoritative (esp. the metric) and read it FIRST.
- The scoring environment (Kaggle notebook) is likely OFFLINE; anything the kernel needs must come from competition data or be self-contained.
- A valid submission must exist as early as possible and at all times thereafter (floor first, improve later).
- Trust measurements over hunches: k-fold means with variance, never single tiny holdouts; small classes produce phantom 0.0s.
- Budget discipline: the harness prepends a [budget] line each turn — plan around remaining time/submissions/cost. Long trainings: prefer <10 min runs.
- Write durable artifacts (TASK_CARD.md, PLAN.md, scaffold/, runs/ledger.json...) — later phases and the final report depend on them.
- Use memory_recall early; save new lessons with memory_write when you learn something reusable.
When the phase goal is met, call phase_complete with a summary. If truly blocked, call phase_complete explaining the blocker."""


class Orchestrator:
    def __init__(self, provider: Provider, slug: str, workspace: Path,
                 budget: Budget, start_phase: str | None = None):
        self.p = provider
        self.slug = slug
        self.ws = workspace
        self.budget = budget
        self.trace = Tracer(workspace / "trace.jsonl")
        self.ctx = Ctx(workspace=workspace, slug=slug, budget=budget, trace=self.trace)
        self.state_f = workspace / "state.json"
        self.start_phase = start_phase

    # ---------------- state ----------------
    def _load_state(self) -> dict:
        if self.state_f.exists():
            return json.loads(self.state_f.read_text())
        return {"done_phases": []}

    def _save_state(self, st: dict):
        self.state_f.write_text(json.dumps(st, indent=1))

    # ---------------- one phase ----------------
    def run_phase(self, phase: Phase) -> bool:
        print(f"\n===== PHASE {phase.name} =====")
        self.trace.log("phase_start", phase=phase.name)
        history: list = []
        system = SYSTEM.format(slug=self.slug)
        gate_feedback = ""

        for attempt in range(3):  # gate retries
            opener = (f"{self.budget.status_line()}\n\n## Current phase: {phase.name}\n"
                      f"{phase.goal}\n{gate_feedback}")
            self.p.append_user(history, opener)

            for step_i in range(phase.max_steps):
                if (why := self.budget.exhausted()):
                    print(f"[halt] {why}")
                    self.trace.log("budget_halt", why=why, phase=phase.name)
                    return False
                step = self.p.step(system, history, SCHEMAS)
                self.budget.note_llm(step.usage, self.p.cost_usd(step.usage))
                self.trace.log("llm", phase=phase.name, text=step.text[:400],
                               calls=[c.name for c in step.tool_calls],
                               usage=step.usage)
                if step.text.strip():
                    print(f"  [agent] {step.text.strip()[:300]}")
                self.p.append_assistant(history, step)

                if not step.tool_calls:
                    # nudge: must use tools or complete
                    self.p.append_user(history,
                        "(Use tools to act, or call phase_complete. "
                        + self.budget.status_line() + ")")
                    continue

                results, completed = [], False
                for call in step.tool_calls:
                    fn = FNS.get(call.name)
                    print(f"  [tool] {call.name}({str(call.input)[:120]})")
                    try:
                        out = fn(call.input, self.ctx) if fn else f"[unknown tool {call.name}]"
                        err = False
                    except Exception as e:  # noqa: BLE001
                        out, err = f"[tool error] {type(e).__name__}: {e}", True
                    self.trace.log("tool", name=call.name, ok=not err,
                                   out=str(out)[:400])
                    if out == "PHASE_COMPLETE":
                        completed = True
                        out = "acknowledged"
                    results.append(ToolResult(call_id=call.id, output=str(out), is_error=err))
                self.p.append_tool_results(history, results)

                if completed:
                    gate_err = phase.gate(self.ws)
                    if gate_err is None:
                        self.trace.log("phase_done", phase=phase.name)
                        print(f"  [gate] {phase.name} PASSED")
                        return True
                    print(f"  [gate] FAILED: {gate_err}")
                    gate_feedback = f"\nGATE CHECK FAILED: {gate_err}\nFix this before calling phase_complete again."
                    break  # re-open with feedback
            else:
                gate_feedback = "\n(step cap reached; be decisive and finish the phase goal)"
        print(f"[phase {phase.name}] giving up after retries")
        self.trace.log("phase_failed", phase=phase.name)
        return False

    # ---------------- full run ----------------
    def run(self, only_phase: str | None = None) -> None:
        st = self._load_state()
        names = [ph.name for ph in PHASES]
        for ph in PHASES:
            if only_phase and ph.name != only_phase:
                continue
            if ph.name in st["done_phases"] and not only_phase:
                print(f"[skip] {ph.name} (done)")
                continue
            if self.start_phase and names.index(ph.name) < names.index(self.start_phase):
                continue
            ok = self.run_phase(ph)
            if ok:
                st["done_phases"].append(ph.name)
                self._save_state(st)
            else:
                print(f"[stop] phase {ph.name} did not pass; state saved for resume")
                break
        print("\n" + self.budget.status_line())
        print(f"trace: {self.trace.path}")
