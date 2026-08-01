"""Per-solver MCP tool servers.

Each solver gets its own server instance so the tools know who is calling. That
is what makes the piggyback work: every tool result comes back with whatever
facts this particular solver has not seen yet, appended at the end. The solver
is already reading that output, so hearing costs it nothing — no polling turn,
no interruption, no tokens spent on "anything new?".

Kaggle/memory/skill functions are reused from agent/tools/registry.py, which is
the part of the old branch that was never the problem.
"""
from __future__ import annotations

from pathlib import Path

from claude_agent_sdk import create_sdk_mcp_server, tool

from agent.tools import registry as R

from . import facts

_CTX: R.Ctx | None = None

_KAGGLE = [
    ("kaggle_overview", R.kaggle_overview, {},
     "Fetch the competition's OFFICIAL pages (description / evaluation / data) "
     "via the Kaggle API. Works for private competitions. READ THIS FIRST."),
    ("kaggle_download", R.kaggle_download, {},
     "Download+unzip this competition's data into input/ (skips if present)."),
    ("kaggle_push_kernel", R.kaggle_push_kernel, {"kernel_dir": str},
     "Push a kernel dir (with kernel-metadata.json) and start a cloud run. "
     "Code competitions score notebooks, not CSVs. Note the version number."),
    ("kaggle_kernel_status", R.kaggle_kernel_status, {"kernel_ref": str},
     "Status of a pushed kernel (default: last pushed). NOTE: Kaggle exposes no "
     "way to cancel a running kernel — a runaway run burns GPU quota until it "
     "finishes, so cap the runtime inside your script instead."),
    ("kaggle_kernel_log", R.kaggle_kernel_log, {"kernel_ref": str},
     "Fetch a finished kernel's run log."),
    ("kaggle_submissions", R.kaggle_submissions, {},
     "List this competition's submissions and their scores."),
    ("memory_recall", R.memory_recall, {"query": str},
     "Recall distilled lessons from past runs. Use early."),
    ("memory_write", R.memory_write, {"name": str, "content": str},
     "Save a durable, task-solving lesson for future runs. No competition logistics."),
    ("skill_list", R.skill_list, {}, "List available skill playbooks."),
    ("skill_load", R.skill_load, {"name": str}, "Load a skill playbook by name."),
]


def init_ctx(workspace: Path, slug: str, budget, trace) -> None:
    global _CTX
    _CTX = R.Ctx(workspace=workspace, slug=slug, budget=budget, trace=trace)


def make_server(solver_id: str):
    """Build an MCP server bound to one agent. Returns (server, tool_names).

    No agent gets a submit tool — `kaggle_submit` is not in the table below at
    all. The harness calls the underlying registry function itself, so that
    quota accounting lives in exactly one place. The daily submission quota is
    small, shared and non-renewable within a run, and three independent agents
    each told that scoring zero is the catastrophic outcome will each buy their
    own insurance — in run 2 all three submitted inside the first four minutes,
    before the metric even existed, and the genuinely better models that
    appeared at minute twelve had nothing left to spend. A scarce shared
    resource has to be allocated by the harness, not by three agents acting
    reasonably in isolation.
    """

    def wrap(fn):
        async def run(args):
            try:
                out = str(fn(args, _CTX))
            except Exception as e:  # noqa: BLE001
                out = f"[tool error] {type(e).__name__}: {e}"
            # Piggyback: the solver hears at a natural pause, for free.
            try:
                out += facts.board().render_unseen(solver_id)
            except Exception:  # noqa: BLE001 — the board must never break a tool
                pass
            return {"content": [{"type": "text", "text": out}]}
        return run

    tools = [tool(name, desc, schema)(wrap(fn))
             for name, fn, schema, desc in _KAGGLE]

    @tool("fact_post",
          "Put ONE thing on the shared board. kind is one of:\n"
          "  claim   — the angle you are taking, so the others can take a "
          "different one. Post this as soon as you have decided.\n"
          "  data    — a structural property of the data: a leak, duplicates, "
          "a label anomaly, a distribution that changes the approach\n"
          "  format  — a submission-format trap\n"
          "  env     — an environment gotcha: an accelerator that OOMs, a "
          "package version, a path that does not exist\n"
          "  failure — something you have CONFIRMED does not work, and why, so "
          "nobody else spends time on it\n"
          "You cannot post `result` — scores reach the board from the evaluator, "
          "computed on the shared folds, because a number measured on your own "
          "split is not comparable to anyone else's. "
          "Max 300 characters. Use layer='day' for facts that "
          "outlive this problem (accelerator availability, package versions, "
          "quota burn rate) so the next problem inherits them. Set urgent=true "
          "ONLY when what you found changes what everyone should be doing right "
          "now — a shared assumption that turns out to be wrong, a format that "
          "voids submissions. Ordinary facts reach the others when they next "
          "call a tool, which is never for someone sitting on a long job; "
          "urgent ones are pushed. Expect to need it once a run, if at all.",
          {"kind": str, "text": str, "layer": str, "urgent": bool})
    async def fact_post(args):
        layer = args.get("layer") or "task"
        out = facts.board().post(args.get("kind", ""), args.get("text", ""),
                                 src=solver_id, layer=layer)
        if args.get("urgent") and out.startswith("[posted]"):
            from . import supervise as S
            S.flag_urgent(args.get("kind", ""), args.get("text", ""), solver_id)
            out += " (urgent — pushed to the others at their next round)"
        return {"content": [{"type": "text", "text": out}]}

    @tool("fact_read", "Read the whole facts board. You normally do not need "
          "this — new facts are appended to every tool result automatically. "
          "Use it when you are stuck and want the full history.", {})
    async def fact_read(args):
        return {"content": [{"type": "text", "text": facts.board().render_all()}]}

    @tool("submission_status",
          "How the shared submission quota stands: how many of today's slots "
          "are left, what has been sent, and what it scored. You do not submit "
          "yourself — write `out/submission.csv` and the harness sends the "
          "best measured candidate.", {})
    async def submission_status(args):
        from .main import quota_report
        return {"content": [{"type": "text", "text": quota_report(_CTX.workspace)}]}

    @tool("recon",
          "Re-run the deterministic reconnaissance over input/ and post what it "
          "finds to the facts board: train/test duplicate overlap, duplicates "
          "within train, label distribution, row-order leakage, and the exact "
          "submission format. It runs once automatically at startup — re-run it "
          "if the data was not downloaded yet at that point, or after you "
          "change what is in input/.", {})
    async def recon(args):
        import subprocess
        import sys
        from .main import ROOT
        r = subprocess.run(
            [sys.executable, "-m", "native.scripts.recon",
             "--input", str(_CTX.workspace / "input"),
             "--workspace", str(_CTX.workspace), "--src", solver_id],
            cwd=str(ROOT), capture_output=True, text=True)
        return {"content": [{"type": "text",
                             "text": (r.stdout or r.stderr)[-4000:]}]}

    tools += [fact_post, fact_read, recon, submission_status]
    server = create_sdk_mcp_server(name="ioai", version="2.1.0", tools=tools)
    names = [f"mcp__ioai__{n}" for n, *_ in _KAGGLE]
    names += ["mcp__ioai__fact_post", "mcp__ioai__fact_read", "mcp__ioai__recon",
              "mcp__ioai__submission_status"]
    return server, names
