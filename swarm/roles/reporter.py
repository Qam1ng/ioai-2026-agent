"""Reporter: the technical report, generated from the record.

The organizers require one report per problem within 30 minutes of the window
closing, plus a declaration of tools/internet/multi-agent usage and the
efficiency metrics they publish. That deadline is why this role carries a
deterministic fallback: if the model turn fails or returns no markdown, a report
assembled directly from the ledger, submission log and budget is still delivered.
A thin factual report beats a missed deadline, and every number in it is
traceable — the Jury audits the trajectory against the report.
"""

from __future__ import annotations

import time
from typing import Any

from .base import Role, RoleResult


def _fmt(value: Any, nd: int = 4) -> str:
    """Render a number, or 'not measured' — never a plausible-looking guess."""
    if value is None or value == "":
        return "not measured"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.{nd}g}"
    return str(value)


class ReporterRole(Role):
    """Writes the per-problem technical report and the required declarations."""

    role_name = "reporter"
    prompt_name = "reporter"
    tools = (
        "read_file",
        "list_dir",
        "kaggle_submissions",
        "memory_recall",
        "memory_write",
        "write_file",
    )
    max_steps = 18

    def __init__(self, name: str | None = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(name or self.role_name, *args, **kwargs)

    # ------------------------------------------------------------ evidence
    def evidence(self) -> dict:
        """Everything the report is allowed to be built from."""
        card = self.bb.get_task_card()
        subs = self.bb.get_submissions()
        return {
            "task_card": card.to_dict() if card else {},
            "plans": [p.to_dict() for p in self.bb.get_plans()],
            "candidates": [c.to_dict() for c in self.bb.get_candidates()],
            "ledger": [e.to_dict() for e in self.bb.get_experiments()],
            "submissions": [s.to_dict() for s in subs],
            "probes": [p.to_dict() for p in self.bb.get_probes()],
            "calibration": self.bb.get_calibration().to_dict(),
            "budget": self.budget.status(),
            "tools_available": list(self.tools or ()),
        }

    def fallback_markdown(self, ev: dict) -> str:
        """Deterministic report from the record, used when the model turn fails."""
        card = ev.get("task_card") or {}
        budget = ev.get("budget") or {}
        lines = [
            f"# Technical report: {self.cfg.slug}",
            "",
            "_Generated deterministically from the run record after the reporting "
            "role failed to produce markdown. Numbers are taken verbatim from the "
            "ledger, submission log and budget; nothing is inferred._",
            "",
            "## Task and metric",
            f"- Title: {card.get('title') or 'not recorded'}",
            f"- Task type: {card.get('task_type', 'unknown')}",
            f"- Metric: {card.get('metric_name') or 'not recorded'} "
            f"({card.get('metric_direction', 'maximize')})",
            f"- Submission format: {card.get('submission_format') or 'not recorded'}",
            "",
            "## Candidates",
        ]
        for c in ev.get("candidates") or []:
            lines.append(
                f"- `{c.get('candidate_id')}` family={c.get('family') or 'n/a'} "
                f"status={c.get('status')} local={_fmt(c.get('local_score'))} "
                f"std={_fmt(c.get('local_std'))}"
            )
        if not ev.get("candidates"):
            lines.append("- none recorded")

        lines += ["", "## Experiment ledger", "", "| experiment | local mean | std | accepted | reason |",
                  "| --- | --- | --- | --- | --- |"]
        for e in ev.get("ledger") or []:
            desc = str(e.get("description", ""))[:90].replace("|", "/")
            reason = str(e.get("reject_reason", ""))[:90].replace("|", "/")
            lines.append(
                f"| {desc} | {_fmt(e.get('local_score'))} | {_fmt(e.get('local_std'))} "
                f"| {e.get('accepted')} | {reason} |"
            )
        if not ev.get("ledger"):
            lines.append("| (no experiments recorded) | | | | |")

        lines += ["", "## Submissions", "", "| lane | local | leaderboard | status |",
                  "| --- | --- | --- | --- |"]
        for s in ev.get("submissions") or []:
            lines.append(
                f"| {s.get('lane')} | {_fmt(s.get('local_score'))} "
                f"| {_fmt(s.get('lb_score'))} | {s.get('status')} |"
            )
        if not ev.get("submissions"):
            lines.append("| (no submissions recorded) | | | |")

        cal = ev.get("calibration") or {}
        lines += [
            "",
            "## Local vs leaderboard calibration",
            f"- points: {cal.get('n_points', 0)}, slope: {_fmt(cal.get('slope'))}, "
            f"gap (local-LB): {_fmt(cal.get('gap'))}, noise band: {_fmt(cal.get('noise_band'))}",
            f"- warning: {cal.get('warning') or 'none'}",
            "",
            "## Declaration",
            "- Multi-agent: yes. Roles: profiler, manager, designer, coder, tuner, "
            "verifier, prober, compliance, aggregator, reporter.",
            "- Internet: used on the agent side; disabled inside the scoring Kaggle "
            "kernel, which trains from scratch with no uploaded checkpoint.",
            f"- Tools available to this role: {', '.join(ev.get('tools_available') or []) or 'none'}.",
            "",
            "## Efficiency metrics",
            f"- tokens in/out: {budget.get('tokens_in', 0)}/{budget.get('tokens_out', 0)}",
            f"- LLM calls: {budget.get('llm_calls', 0)}, cost: ${budget.get('cost_usd', 0)}",
            f"- wall clock: {budget.get('elapsed_min', 0)} min",
            f"- submissions: {budget.get('submissions', 0)}",
            f"- GPU hours used: {budget.get('gpu_hours_used', 'not measured')}",
            "",
            f"_Report generated at {time.strftime('%Y-%m-%d %H:%M:%S')}._",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------- parsing
    @staticmethod
    def parse_report(report: dict) -> dict:
        """Normalise the report payload; ``markdown`` may still be empty here."""
        rep = report if isinstance(report, dict) else {}
        md = rep.get("markdown")
        md = md if isinstance(md, str) else ""
        decl = rep.get("declaration")
        eff = rep.get("efficiency")
        return {
            "markdown": md,
            "declaration": decl if isinstance(decl, dict) else {},
            "efficiency": eff if isinstance(eff, dict) else {},
            "unknowns": [str(u) for u in (rep.get("unknowns") or [])],
            "generated": "role",
        }

    # ---------------------------------------------------------------- run
    def write_report(self, instructions: str = "", context: dict | None = None) -> dict:
        """Produce the report dict; always returns non-empty ``markdown``."""
        ev = self.evidence()
        instructions = instructions or (
            "Write the technical report for this problem from the record supplied. "
            "Include the experiment table with its rejections, the score trajectory and "
            "calibration, the verification story, the tool/internet/multi-agent "
            "declaration, and the efficiency metrics. Write 'not measured' for anything "
            "the record does not contain."
        )
        ctx = dict(context or {})
        ctx.update(ev)

        try:
            res: RoleResult = self.run(instructions, ctx)
            out = self.parse_report(res.report)
        except Exception as exc:  # the deadline outranks the exception
            out = {"markdown": "", "declaration": {}, "efficiency": {},
                   "unknowns": [f"reporter step raised {type(exc).__name__}: {exc}"],
                   "generated": "role"}

        if not out["markdown"].strip():
            out["markdown"] = self.fallback_markdown(ev)
            out["generated"] = "fallback"
            out["unknowns"].append("model produced no markdown; deterministic fallback used")

        # Efficiency metrics come from the budget, not from the model's memory.
        budget = ev.get("budget") or {}
        out["efficiency"] = {
            **budget,
            **{k: v for k, v in (out["efficiency"] or {}).items() if k not in budget},
        }
        out["declaration"] = {
            "multi_agent": True,
            "internet_agent_side": True,
            "internet_in_kernel": False,
            **(out["declaration"] or {}),
        }

        path = self.bb.ws / "REPORT.md"
        try:
            path.write_text(out["markdown"])
            out["path"] = str(path)
        except OSError as exc:  # pragma: no cover - disk failure at the deadline
            out["unknowns"].append(f"could not write REPORT.md: {exc}")
        self._emit("report_written", generated=out["generated"], chars=len(out["markdown"]))
        return out
