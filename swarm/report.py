"""Auto-generated technical report and the operator's status block.

The rules give us **30 minutes after the window closes** to hand in a technical
report per problem (``docs/QA-NOTES.md``), and they require a declaration of the
tools and resources used plus published efficiency metrics. Thirty minutes is
not enough to write three reports by hand, and it is far too little to
reconstruct what happened from memory — so the report is generated from the
blackboard, in milliseconds, at any moment during the run.

Two rules govern everything in this module:

1. **Never invent a number.** Every figure is read from the blackboard, the
   budget, or the GPU quota pool. Anything that was not measured is printed as
   ``unknown``. A plausible-looking fabricated metric in a submitted report is
   worse than an admitted gap.
2. **Declare honestly, including the unflattering parts.** Our role prompts and
   orchestration harness are human-written; the declaration says so plainly.
   The internet was used agent-side; it says that too, together with the fact
   that the submitted kernel itself runs with internet disabled.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from .budget import PodBudget, QuotaPool
from .bus import Blackboard, _atomic_write
from .config import ROLES, SwarmConfig
from .ledger import ExperimentLedger, fit_calibration, predicted_lb, shrunk_estimate
from .schemas import Calibration, TaskCard

UNKNOWN = "unknown"

# --------------------------------------------------------------------------- #
# Tool taxonomy for the declaration. Names come from ``agent/tools/registry.py``;
# prefix matching keeps the mapping stable if a sibling agent renames or adds a
# variant (``web_search_v2``), and anything unrecognised is still listed under
# "other tools" so nothing silently disappears from the declaration.
# --------------------------------------------------------------------------- #
INTERNET_TOOLS = (
    "kaggle_",  # dataset download, kernel push, submission, score polling
    "web_search",
    "web_fetch",
    "http_",
)
CODE_EXEC_TOOLS = ("run_python", "run_bash", "run_shell")
RETRIEVAL_TOOLS = ("web_search", "web_fetch", "memory_recall", "skill_load", "skill_list")

_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)

#: Third-party imports we never want to advertise as "libraries used by the
#: solution" because they belong to the harness, not to the submitted kernel.
_HARNESS_MODULES = {"swarm", "agent", "tasks", "src", "kaggle", "anthropic", "openai", "yaml"}


# --------------------------------------------------------------------------- #
# formatting helpers -- the single place where "unknown" is decided
# --------------------------------------------------------------------------- #
def _num(x: Any, nd: int = 4) -> str:
    try:
        if x is None or isinstance(x, bool):
            return UNKNOWN
        f = float(x)
    except (TypeError, ValueError):
        return UNKNOWN
    if f != f or f in (float("inf"), float("-inf")):  # NaN / inf
        return UNKNOWN
    return f"{f:.{nd}f}"


def _int(x: Any) -> str:
    try:
        if x is None or isinstance(x, bool):
            return UNKNOWN
        return str(int(x))
    except (TypeError, ValueError):
        return UNKNOWN


def _txt(s: Any, default: str = UNKNOWN, limit: int = 400) -> str:
    if s is None:
        return default
    t = str(s).strip()
    if not t:
        return default
    t = t.replace("\n", " ")
    return t if len(t) <= limit else t[: limit - 1] + "…"


def _cell(s: Any, default: str = "-", limit: int = 70) -> str:
    """Table-cell text: pipes escaped so a stray character cannot break markdown."""
    return _txt(s, default=default, limit=limit).replace("|", "/")


def _yes_no(b: bool) -> str:
    return "yes" if b else "no"


def _ts(t: Any) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(t)))
    except (TypeError, ValueError):
        return UNKNOWN


# --------------------------------------------------------------------------- #
# evidence gathering
# --------------------------------------------------------------------------- #
def read_events(bb: Blackboard) -> list[dict]:
    """Every audit event, tolerant of a torn last line (a run may be live)."""
    path = bb.events_path
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def tool_usage(events: Iterable[dict]) -> dict[str, int]:
    """``tool name -> call count``, straight from the event log."""
    counts: dict[str, int] = {}
    for e in events:
        if e.get("kind") != "tool":
            continue
        name = str(e.get("tool") or "").strip()
        if name:
            counts[name] = counts.get(name, 0) + 1
    return counts


def roles_observed(events: Iterable[dict]) -> dict[str, int]:
    """``role -> number of role invocations`` (``role_start`` events)."""
    counts: dict[str, int] = {}
    for e in events:
        if e.get("kind") != "role_start":
            continue
        role = str(e.get("role") or "").strip()
        if role:
            counts[role] = counts.get(role, 0) + 1
    return counts


def _matches(name: str, prefixes: Iterable[str]) -> bool:
    return any(name == p or name.startswith(p) for p in prefixes)


def detect_libraries(bb: Blackboard, max_files: int = 400) -> list[str]:
    """Third-party modules imported by the code the agents actually wrote.

    A static import scan of the Python written into the workspace. This is
    evidence, not a guess: if no solution file exists yet the list is empty and
    the report prints ``unknown`` rather than inventing a plausible stack.
    """
    stdlib = set(getattr(sys, "stdlib_module_names", set()))
    found: set[str] = set()
    files = 0
    for path in sorted(bb.ws.rglob("*.py")):
        if files >= max_files:
            break
        parts = set(path.parts)
        if parts & {".git", "__pycache__", ".venv", "input"}:
            continue
        files += 1
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for mod in _IMPORT_RE.findall(text):
            root = mod.split(".")[0]
            if root and root not in stdlib and root not in _HARNESS_MODULES:
                found.add(root)
    return sorted(found)


def gpu_hours_used(bb: Blackboard, quota: QuotaPool | None) -> float | None:
    """GPU hours consumed, preferring the authoritative shared quota pool."""
    if quota is not None:
        try:
            return float(quota.status().get("gpu_hours_used"))
        except (TypeError, ValueError):
            pass
    secs = sum(float(s.gpu_seconds or 0.0) for s in bb.get_submissions())
    return secs / 3600.0 if secs > 0 else None


# --------------------------------------------------------------------------- #
# declaration
# --------------------------------------------------------------------------- #
def collect_declaration(
    bb: Blackboard,
    budget: PodBudget | None,
    quota: QuotaPool | None,
    cfg: SwarmConfig | None,
) -> dict:
    """The tools/resources declaration the rules require, derived from evidence.

    Every boolean is backed by a count from the event log or by a configuration
    fact, and the backing evidence travels with the answer so a Jury member can
    check it against ``events.jsonl``.
    """
    events = read_events(bb)
    tools = tool_usage(events)
    roles_seen = roles_observed(events)
    st = budget.status() if budget is not None else {}
    cfg = cfg or SwarmConfig()

    internet_tools = {k: v for k, v in tools.items() if _matches(k, INTERNET_TOOLS)}
    exec_tools = {k: v for k, v in tools.items() if _matches(k, CODE_EXEC_TOOLS)}
    retrieval = {k: v for k, v in tools.items() if _matches(k, RETRIEVAL_TOOLS)}
    other_tools = {
        k: v
        for k, v in tools.items()
        if k not in internet_tools and k not in exec_tools and k not in retrieval
    }

    llm_calls = int(st.get("llm_calls", 0) or 0)
    hosted_model = bool(cfg.default_model.backend in ("claude", "openai") or cfg.default_model.base_url)
    # Calling a hosted model IS internet access; declaring only the Kaggle calls
    # would be a technically-true omission.
    internet_used = bool(internet_tools) or (llm_calls > 0 and hosted_model)

    cands = bb.get_candidates()
    subs = bb.get_submissions()
    gpu_h = gpu_hours_used(bb, quota)
    libs = detect_libraries(bb)

    return {
        "generated_at": _ts(time.time()),
        "slug": cfg.slug or (bb.get_task_card().slug if bb.get_task_card() else ""),
        "internet_access": {
            "used": internet_used,
            "detail": (
                "Agent side only. The Kaggle API was used to download data, push "
                "kernels and read leaderboard scores, and the language model runs "
                "behind a hosted API. The SUBMITTED kernel itself runs on Kaggle "
                "with internet disabled and installs nothing."
            ),
            "evidence": {
                "network_tool_calls": internet_tools,
                "llm_api_calls": llm_calls,
                "model_backend": cfg.default_model.backend,
                "model": cfg.default_model.model or UNKNOWN,
            },
        },
        "code_execution": {
            "used": bool(exec_tools),
            "detail": (
                "All solution code was written by the agents and executed locally "
                "in the harness sandbox (timeout-bounded), then re-executed on "
                "Kaggle hardware as the submitted kernel."
            ),
            "evidence": {"tool_calls": exec_tools},
        },
        "retrieval_or_search": {
            "used": bool(retrieval),
            "detail": (
                "Retrieval was limited to this repository's own on-disk memory "
                "(memory/lessons) and skill playbooks unless a web tool appears "
                "in the evidence below."
            ),
            "evidence": {"tool_calls": retrieval},
        },
        "libraries": {
            "detected": libs,
            "source": "static import scan of the Python files written into the workspace",
            "note": (
                "Empty means no solution file had been written when this report was "
                "generated; it does not mean no library was used."
            ),
        },
        "multiple_agents": {
            "used": True,
            "detail": (
                "A role swarm on a shared file blackboard: a Manager takes one "
                "action per step, and specialist roles (profiler, designer, coder, "
                "tuner, verifier, prober, compliance, aggregator, reporter) run as "
                "short-lived bounded loops. Several solution candidates are "
                "developed in parallel."
            ),
            "n_roles_configured": len(ROLES),
            "roles_configured": list(ROLES),
            "roles_observed": sorted(roles_seen),
            "role_invocations": roles_seen,
            "max_parallel_candidates": cfg.parallel.max_candidates,
            "n_candidates_created": len(cands),
        },
        "private_tools": {
            "used": bool(cfg.default_model.base_url),
            "detail": (
                "The harness, prompts and tools are entirely in this repository. "
                + (
                    "The model is served through a private/internal gateway "
                    f"({cfg.default_model.base_url}); no other private tool was used."
                    if cfg.default_model.base_url
                    else "No private or proprietary tooling was used beyond the "
                    "public model API and the Kaggle API."
                )
            ),
            "harness_tools_invoked": sorted(tools),
            "other_tool_calls": other_tools,
        },
        "human_written_prompts_or_scaffolding": {
            "used": True,
            "detail": (
                "Honest answer: yes. The role prompts under prompts/swarm/, the "
                "orchestration harness (budgets, submission gates, blackboard, "
                "this report generator) and the reusable skill playbooks were "
                "written by humans before the competition and are task-agnostic. "
                "Everything task-specific — reading the task, the plan, all model "
                "code, all iteration decisions and the final selection — was "
                "produced by the agents during the window with no human in the loop."
            ),
        },
        "efficiency": {
            "wall_clock_minutes": st.get("elapsed_min", None),
            "wall_clock_remaining_minutes": st.get("remaining_min", None),
            "n_experiments": len(bb.get_experiments()),
            "minutes_per_experiment": (
                round(float(st["elapsed_min"]) / len(bb.get_experiments()), 2)
                if st.get("elapsed_min") is not None and bb.get_experiments()
                else None
            ),
            "cost_usd": st.get("cost_usd", None),
            "tokens_in": st.get("tokens_in", None),
            "tokens_out": st.get("tokens_out", None),
            "llm_calls": st.get("llm_calls", None),
            "n_submissions": len(subs),
            "gpu_hours": gpu_h,
        },
    }


def _render_declaration(dec: dict) -> str:
    """Declaration block. Reads as prose for a human, keyed for a checklist."""
    lines = ["## Declaration of tools and resources", ""]
    lines.append("| Item | Used | Detail |")
    lines.append("|---|---|---|")
    rows = [
        ("Internet access", "internet_access"),
        ("Code execution", "code_execution"),
        ("Retrieval / search", "retrieval_or_search"),
        ("Multiple agents", "multiple_agents"),
        ("Private tools", "private_tools"),
        ("Human-written prompts / scaffolding", "human_written_prompts_or_scaffolding"),
    ]
    for label, key in rows:
        block = dec.get(key, {})
        lines.append(
            f"| {label} | {_yes_no(bool(block.get('used')))} | "
            f"{_cell(block.get('detail'), default=UNKNOWN, limit=800)} |"
        )

    libs = dec.get("libraries", {}).get("detected") or []
    lines.append(
        f"| Libraries | {_yes_no(bool(libs))} | "
        f"{', '.join(libs) if libs else UNKNOWN + ' (no solution file scanned yet)'} |"
    )

    ma = dec.get("multiple_agents", {})
    lines += [
        "",
        "**Multi-agent detail.** "
        f"{_int(ma.get('n_roles_configured'))} specialist roles are configured "
        f"({', '.join(ma.get('roles_configured') or []) or UNKNOWN}); roles actually "
        f"invoked in this run: {', '.join(ma.get('roles_observed') or []) or UNKNOWN}. "
        f"Up to {_int(ma.get('max_parallel_candidates'))} solution candidates run in "
        f"parallel; {_int(ma.get('n_candidates_created'))} candidate(s) were created.",
        "",
        "**Network detail.** "
        + _txt(dec.get("internet_access", {}).get("detail"), limit=600),
        "",
        "**Evidence.** Every entry above is derived from `events.jsonl` "
        "(tool and role call log), `submissions.jsonl` and the persisted budget "
        "files in this workspace. Network tool calls: "
        + (
            ", ".join(
                f"{k}×{v}"
                for k, v in sorted(
                    (dec.get("internet_access", {}).get("evidence", {}) or {})
                    .get("network_tool_calls", {})
                    .items()
                )
            )
            or "none recorded"
        )
        + ".",
    ]
    return "\n".join(lines)


def _render_efficiency(dec: dict, budget: PodBudget | None, quota: QuotaPool | None) -> str:
    eff = dec.get("efficiency", {})
    lines = ["## Efficiency metrics", ""]
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Wall-clock elapsed (min) | {_num(eff.get('wall_clock_minutes'), 1)} |")
    lines.append(
        f"| Wall-clock remaining in window (min) | "
        f"{_num(eff.get('wall_clock_remaining_minutes'), 1)} |"
    )
    lines.append(f"| Experiments run | {_int(eff.get('n_experiments'))} |")
    lines.append(
        f"| Wall-clock per experiment (min) | {_num(eff.get('minutes_per_experiment'), 2)} |"
    )
    lines.append(f"| LLM calls | {_int(eff.get('llm_calls'))} |")
    lines.append(f"| Input tokens | {_int(eff.get('tokens_in'))} |")
    lines.append(f"| Output tokens | {_int(eff.get('tokens_out'))} |")
    lines.append(f"| LLM cost (USD) | {_num(eff.get('cost_usd'), 2)} |")
    lines.append(f"| Kaggle submissions used | {_int(eff.get('n_submissions'))} |")
    lines.append(f"| GPU-hours consumed | {_num(eff.get('gpu_hours'), 2)} |")
    if quota is not None:
        q = quota.status()
        lines.append(f"| GPU-hours remaining in shared pool | {_num(q.get('gpu_hours_remaining'), 2)} |")
    if budget is not None:
        lines.append(f"| Submissions left in budget | {_int(budget.submissions_left())} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# report sections
# --------------------------------------------------------------------------- #
def _render_problem(card: TaskCard | None) -> str:
    lines = ["## 1. Problem", ""]
    if card is None:
        lines.append(
            "No task card was recorded, so the problem statement is `unknown` "
            "in this report. (A run that reaches the report stage normally has "
            "one; its absence is itself a finding.)"
        )
        return "\n".join(lines)
    lines += [
        f"- **Competition**: {_txt(card.slug)}",
        f"- **Title**: {_txt(card.title)}",
        f"- **Task type (as read by the Profiler)**: {_txt(card.task_type)}",
        f"- **Metric**: {_txt(card.metric_name)} ({_txt(card.metric_direction)})",
        f"- **Metric definition**: {_txt(card.metric_description, limit=600)}",
        f"- **Submission format**: {_txt(card.submission_format, limit=400)}",
        f"- **Data**: {_txt(card.data_summary, limit=600)}",
        f"- **Grouping variable for CV**: {_txt(card.grouping_variable, default='none identified')}",
    ]
    if card.constraints:
        lines += ["", "**Constraints taken verbatim from the task statement:**", ""]
        for c in card.constraints:
            lines.append(f"- *{_txt(c.kind)}*: “{_txt(c.quote, limit=300)}”")
    if card.risks:
        lines += ["", "**Risks flagged at read time:**", ""]
        lines += [f"- {_txt(r, limit=300)}" for r in card.risks]
    return "\n".join(lines)


def _render_approach(bb: Blackboard) -> str:
    plans = bb.get_plans()
    cands = bb.get_candidates()
    lines = ["## 2. Approach actually taken", ""]
    if not plans and not cands:
        lines.append("_No plans or candidates were recorded._")
        return "\n".join(lines)

    if plans:
        lines += ["**Plans considered**", "", "| Plan | Family | Approach | Validation | Accel |", "|---|---|---|---|---|"]
        for p in plans:
            lines.append(
                f"| {_cell(p.plan_id)} | {_cell(p.family)} | {_cell(p.title or p.architecture)} | "
                f"{_cell(p.validation_strategy)} | {_cell(p.accelerator)} |"
            )
        lines.append("")

    if cands:
        lines += [
            "**Candidates built in parallel**",
            "",
            "| Candidate | Family | Status | Local CV | ±std | Note |",
            "|---|---|---|---|---|---|",
        ]
        for c in cands:
            note = c.retire_reason or c.last_error or ""
            lines.append(
                f"| {_cell(c.candidate_id)} | {_cell(c.family)} | {_cell(c.status)} | "
                f"{_num(c.local_score)} | {_num(c.local_std)} | {_cell(note)} |"
            )
    return "\n".join(lines)


def _render_trajectory(bb: Blackboard, cal: Calibration) -> str:
    subs = bb.get_submissions()
    lines = ["## 4. Score trajectory (local CV vs leaderboard)", ""]
    if not subs:
        lines.append("_No submissions were made._")
    else:
        lines += [
            "| # | Time | Lane | Purpose | Candidate | Local CV | Leaderboard | Status |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for i, s in enumerate(sorted(subs, key=lambda r: r.submitted_at), start=1):
            lines.append(
                f"| {i} | {_ts(s.submitted_at)} | {_cell(s.lane)} | {_cell(s.purpose)} | "
                f"{_cell(s.candidate_id)} | {_num(s.local_score)} | {_num(s.lb_score)} | "
                f"{_cell(s.status)} |"
            )
        lines.append("")

    lines += ["**Calibration of local CV against the leaderboard**", ""]
    lines += [
        f"- paired points: {_int(cal.n_points)}",
        f"- fitted line: leaderboard ≈ {_num(cal.slope)} × local + {_num(cal.intercept)}",
        f"- residual spread around the line: {_num(cal.residual_std)}",
        f"- **gap (mean local − mean leaderboard): {_num(cal.gap)}**",
        f"- noise band (smallest local difference distinguishable from noise): {_num(cal.noise_band)}",
    ]
    if cal.warning:
        lines += ["", f"> **Calibration warning.** {_txt(cal.warning, limit=1500)}"]
    else:
        lines += [
            "",
            "> No calibration warning was raised: the local↔leaderboard offset "
            "stayed inside the band we consider explainable by sampling noise.",
        ]
    return "\n".join(lines)


def _render_rejected(bb: Blackboard) -> str:
    exps = bb.get_experiments()
    rejected = [e for e in exps if not e.accepted and (e.reject_reason or e.local_score is not None)]
    retired = [c for c in bb.get_candidates() if c.status in ("retired", "failed")]
    probes = bb.get_probes()

    lines = ["## 5. What was tried and rejected, and why", ""]
    if not rejected and not retired and not probes:
        lines.append("_Nothing was rejected or retired; no probes were run._")
        return "\n".join(lines)

    if rejected:
        lines += ["**Experiments rejected by the significance gate or by failure**", "", "| Experiment | Local CV | Reason |", "|---|---|---|"]
        for e in rejected[-25:]:
            lines.append(
                f"| {_cell(e.description or e.exp_id, limit=60)} | {_num(e.local_score)} | "
                f"{_cell(e.reject_reason, default='not recorded', limit=200)} |"
            )
        lines.append("")

    if retired:
        lines += ["**Candidates retired**", "", "| Candidate | Family | Status | Reason |", "|---|---|---|---|"]
        for c in retired:
            lines.append(
                f"| {_cell(c.candidate_id)} | {_cell(c.family)} | {_cell(c.status)} | "
                f"{_cell(c.retire_reason or c.last_error, default='not recorded', limit=200)} |"
            )
        lines.append("")

    if probes:
        lines += ["**Probes (cheap submissions bought purely for information)**", "", "| Probe | Hypothesis | LB | What it told us |", "|---|---|---|---|"]
        for p in probes:
            lines.append(
                f"| {_cell(p.probe_kind)} | {_cell(p.hypothesis, limit=90)} | {_num(p.lb_score)} | "
                f"{_cell(p.inference, default='not recorded', limit=160)} |"
            )
    return "\n".join(lines)


def _render_selection(bb: Blackboard, cal: Calibration) -> str:
    """Final-selection section: raw leaderboard argmax vs the shrunk choice."""
    scored = [s for s in bb.scored_submissions()]
    lines = ["## 6. Final selection (winner's-curse adjustment)", ""]
    if not scored:
        lines.append(
            "_No scored submission exists, so no selection analysis is possible._"
        )
        return "\n".join(lines)

    rows = []
    for s in scored:
        try:
            shrunk = shrunk_estimate(s.lb_score, s.local_score, cal)
        except ValueError:
            shrunk = None
        rows.append((s, predicted_lb(s.local_score, cal), shrunk))

    raw_best = max(scored, key=lambda s: float(s.lb_score))  # type: ignore[arg-type]
    with_shrunk = [r for r in rows if r[2] is not None]
    shrunk_best = (
        max(with_shrunk, key=lambda r: float(r[2]))[0] if with_shrunk else None  # type: ignore[arg-type]
    )

    lines += [
        "The final ranking re-scores our best submission on a *different* private "
        "test set. Picking the raw leaderboard argmax therefore favours whichever "
        "submission got the luckiest scoring noise, and those regress hardest on "
        "the second test set. Each observed score below is shrunk toward what the "
        "calibration predicts from its local CV, which is the estimate with the "
        "best expected score on the second test set.",
        "",
        "| Submission | Candidate | Local CV | Leaderboard | Calibration predicts | Shrunk estimate |",
        "|---|---|---|---|---|---|",
    ]
    for s, pred, shrunk in rows:
        lines.append(
            f"| {_cell(s.sub_id)} | {_cell(s.candidate_id)} | {_num(s.local_score)} | "
            f"{_num(s.lb_score)} | {_num(pred)} | {_num(shrunk)} |"
        )
    lines += [
        "",
        f"- raw leaderboard argmax: **{_cell(raw_best.sub_id)}** "
        f"(leaderboard {_num(raw_best.lb_score)}, local {_num(raw_best.local_score)})",
        f"- shrinkage-adjusted choice: **{_cell(shrunk_best.sub_id) if shrunk_best else UNKNOWN}**",
    ]
    if shrunk_best is not None and shrunk_best.sub_id != raw_best.sub_id:
        lines.append(
            "- the two disagree: the leaderboard leader looks noise-lucky relative "
            "to its local CV, and the shrunk choice is the one we expect to hold up."
        )
    elif shrunk_best is not None:
        lines.append("- the two agree, so the leaderboard leader is also the robust choice.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def render_report(
    bb: Blackboard,
    budget: PodBudget | None = None,
    quota: QuotaPool | None = None,
    cfg: SwarmConfig | None = None,
    task_card: TaskCard | None = None,
) -> str:
    """The full technical report as markdown.

    Pure reads: no LLM call, no network, no recomputation of any model. It is
    safe (and intended) to call this at any point during the run, so a crash at
    T-1min still leaves a submittable report on disk.
    """
    cfg = cfg or SwarmConfig()
    card = task_card or bb.get_task_card()
    ledger = ExperimentLedger(bb)

    # Prefer the persisted calibration; if none was ever written, fit one now
    # from the same data rather than reporting an empty block.
    cal = bb.get_calibration()
    if cal.n_points == 0 and bb.scored_submissions():
        cal = fit_calibration(bb)

    dec = collect_declaration(bb, budget, quota, cfg)
    stats = ledger.summary_stats()
    slug = cfg.slug or (card.slug if card else "") or "unknown-competition"

    header = [
        f"# Technical report — {slug}",
        "",
        f"*Generated automatically at {_ts(time.time())} from the run's own "
        "blackboard (`task_card.json`, `plans.jsonl`, `ledger.jsonl`, "
        "`submissions.jsonl`, `probes.jsonl`, `events.jsonl`) and persisted budget "
        "files. Every number below was measured during the run; anything that was "
        "not measured is printed as `unknown` rather than estimated.*",
        "",
        f"- Best local CV: **{_num(stats.get('best_local'))}** "
        f"(experiment {_cell(stats.get('best_exp_id'), default=UNKNOWN)})",
        f"- Best leaderboard score: **"
        f"{_num(max((s.lb_score for s in bb.scored_submissions()), default=None))}**",
        f"- Experiments: {_int(stats.get('n_experiments'))} "
        f"({_int(stats.get('n_accepted'))} accepted by the significance gate)",
        f"- Submissions: {_int(len(bb.get_submissions()))}",
        "",
    ]

    exp_section = [
        "## 3. Experiment ledger",
        "",
        "Every experiment, in order. `verdict` records the significance gate's "
        "plain-language decision: a gain smaller than the noise band is rejected "
        "on purpose, because spending a submission on noise is how a run loses "
        "time it cannot get back.",
        "",
        ledger.as_table(),
        "",
        f"Summary: {_int(stats.get('n_scored'))} scored experiments across "
        f"{_int(stats.get('n_candidates'))} candidate(s); acceptance rate "
        f"{_num(stats.get('acceptance_rate'), 2)}; median local CV "
        f"{_num(stats.get('median_local'))}; total measured experiment time "
        f"{_num(stats.get('total_experiment_minutes'), 1)} min.",
        "",
    ]

    parts = [
        "\n".join(header),
        _render_problem(card),
        "",
        _render_approach(bb),
        "",
        "\n".join(exp_section),
        _render_trajectory(bb, cal),
        "",
        _render_rejected(bb),
        "",
        _render_selection(bb, cal),
        "",
        _render_declaration(dec),
        "",
        _render_efficiency(dec, budget, quota),
        "",
        "---",
        "",
        "*This report was written by the system itself; no human edited it. "
        "Raw artefacts for audit live beside it in this workspace directory.*",
        "",
    ]
    return "\n".join(parts)


def write_report(
    bb: Blackboard,
    budget: PodBudget | None = None,
    quota: QuotaPool | None = None,
    cfg: SwarmConfig | None = None,
    task_card: TaskCard | None = None,
    filename: str = "REPORT.md",
) -> Path:
    """Render and atomically write ``workspace/<slug>/REPORT.md``.

    Atomic because the report may be regenerated while an operator (or the
    submission script) is reading it; a half-written report at the deadline
    would be worse than a slightly stale one.
    """
    text = render_report(bb, budget, quota, cfg, task_card)
    path = bb.ws / filename
    _atomic_write(path, text)
    bb.event("report_written", path=str(path), chars=len(text))
    return path


def render_run_summary(bb: Blackboard, budget: PodBudget | None = None) -> str:
    """A few lines an operator can read at a glance while the run is going.

    Deliberately different from the report: it answers "is this run healthy right
    now?", not "what did this run do?".
    """
    snap = bb.snapshot()
    cal = bb.get_calibration()
    subs = bb.get_submissions()
    scored = bb.scored_submissions()
    stats = ExperimentLedger(bb).summary_stats()

    lines: list[str] = []
    if budget is not None:
        lines.append(budget.status_line())
    lines.append(
        f"[state] task={_txt(snap.get('task_type'))} metric={_txt(snap.get('metric'), default='-')} "
        f"plans={_int(snap.get('n_plans'))} candidates={len(snap.get('candidates') or [])} "
        f"experiments={_int(stats.get('n_experiments'))} "
        f"(accepted {_int(stats.get('n_accepted'))})"
    )
    lines.append(
        f"[scores] best_local={_num(snap.get('best_local'))} "
        f"best_lb={_num(snap.get('best_lb'))} "
        f"submissions={len(subs)} scored={len(scored)}"
    )
    lines.append(
        f"[calibration] n={_int(cal.n_points)} gap={_num(cal.gap)} "
        f"noise_band={_num(cal.noise_band)}"
        + (f" WARNING: {_txt(cal.warning, limit=160)}" if cal.warning else "")
    )
    for c in snap.get("candidates") or []:
        line = (
            f"  - {_cell(c.get('id'))} [{_cell(c.get('status'))}] "
            f"{_cell(c.get('family'))} local={_num(c.get('local_score'))} "
            f"±{_num(c.get('local_std'))}"
        )
        if c.get("last_error"):
            line += f" err={_cell(c.get('last_error'), limit=80)}"
        lines.append(line)
    findings = snap.get("probe_findings") or []
    if findings:
        lines.append("[probes] " + " | ".join(_txt(f, limit=100) for f in findings[-3:]))
    return "\n".join(lines)
