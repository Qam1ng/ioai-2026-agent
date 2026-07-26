"""Tool registry: neutral schemas + dispatch.

Every tool = {"name", "description", "input_schema"} + a python function
run(args: dict, ctx: Ctx) -> str. The orchestrator executes calls and feeds
outputs back to the model. All filesystem access is confined to ctx.workspace.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Ctx:
    workspace: Path            # per-run sandbox dir
    slug: str                  # kaggle competition slug
    budget: "object" = None    # Budget (orchestrator.budget)
    trace: "object" = None     # Tracer
    kernel_ref: str = ""       # <user>/<kernel-slug> once created
    extra: dict = field(default_factory=dict)


def _safe(ctx: Ctx, rel: str) -> Path:
    """Resolve a path and require it inside the workspace (or repo memory/skills read-only)."""
    p = (ctx.workspace / rel).resolve()
    if not str(p).startswith(str(ctx.workspace.resolve())):
        raise ValueError(f"path escapes workspace: {rel}")
    return p


def _tail(s: str, n: int = 6000) -> str:
    return s if len(s) <= n else "...[truncated]...\n" + s[-n:]


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
def run_python(args, ctx):
    code = args["code"]
    timeout = int(args.get("timeout_s", 600))
    f = _safe(ctx, f"_exec_{int(time.time()*1000)}.py")
    f.write_text(code)
    try:
        r = subprocess.run(["python", f.name], cwd=ctx.workspace,
                           capture_output=True, text=True, timeout=timeout)
        out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
        return _tail(out or "(no output)") + f"\n[exit={r.returncode}]"
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT after {timeout}s]"
    finally:
        f.unlink(missing_ok=True)


def run_bash(args, ctx):
    timeout = int(args.get("timeout_s", 300))
    try:
        r = subprocess.run(["bash", "-c", args["command"]], cwd=ctx.workspace,
                           capture_output=True, text=True, timeout=timeout)
        out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
        return _tail(out or "(no output)") + f"\n[exit={r.returncode}]"
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT after {timeout}s]"


# --------------------------------------------------------------------------- #
# files
# --------------------------------------------------------------------------- #
def write_file(args, ctx):
    p = _safe(ctx, args["path"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(args["content"])
    return f"wrote {args['path']} ({len(args['content'])} chars)"


def read_file(args, ctx):
    p = _safe(ctx, args["path"])
    if not p.exists():
        return f"[not found: {args['path']}]"
    txt = p.read_text(errors="replace")
    start = int(args.get("offset", 0))
    return _tail(txt[start:start + int(args.get("limit", 20000))], 20000)


def list_dir(args, ctx):
    p = _safe(ctx, args.get("path", "."))
    if not p.exists():
        return "[not found]"
    lines = []
    for c in sorted(p.iterdir()):
        size = c.stat().st_size if c.is_file() else "-"
        lines.append(f"{'d' if c.is_dir() else 'f'} {size:>12} {c.name}")
        if len(lines) >= 200:
            lines.append("...[truncated]")
            break
    return "\n".join(lines) or "(empty)"


# --------------------------------------------------------------------------- #
# kaggle
# --------------------------------------------------------------------------- #
def _kg(cmd: list[str], timeout=1800):
    r = subprocess.run(["kaggle"] + cmd, capture_output=True, text=True,
                       timeout=timeout)
    return _tail((r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else ""))


def kaggle_download(args, ctx):
    dest = ctx.workspace / "input"
    if dest.exists() and any(dest.iterdir()):
        return f"data already present in input/ ({sum(1 for _ in dest.rglob('*'))} entries) — not re-downloading"
    dest.mkdir(parents=True, exist_ok=True)
    out = _kg(["competitions", "download", "-c", ctx.slug, "-p", str(dest)])
    # unzip any archive
    for z in dest.glob("*.zip"):
        subprocess.run(["unzip", "-q", "-o", str(z), "-d", str(dest)], timeout=600)
        z.unlink()
    return out + "\n[downloaded and unzipped into input/]"


def kaggle_push_kernel(args, ctx):
    """Push kernel dir (must contain kernel-metadata.json) and run it on Kaggle."""
    kdir = _safe(ctx, args["kernel_dir"])
    if not (kdir / "kernel-metadata.json").exists():
        return "[error: kernel-metadata.json missing in kernel_dir]"
    out = _kg(["kernels", "push", "-p", str(kdir)])
    import json as _json
    meta = _json.loads((kdir / "kernel-metadata.json").read_text())
    ctx.kernel_ref = meta.get("id", ctx.kernel_ref)
    return out


def kaggle_kernel_status(args, ctx):
    ref = args.get("kernel_ref") or ctx.kernel_ref
    if not ref:
        return "[error: no kernel_ref known]"
    return _kg(["kernels", "status", ref])


def kaggle_kernel_log(args, ctx):
    ref = args.get("kernel_ref") or ctx.kernel_ref
    out_dir = ctx.workspace / "_kernel_out"
    out_dir.mkdir(exist_ok=True)
    _kg(["kernels", "output", ref, "-p", str(out_dir)])
    logs = sorted(out_dir.glob("*.log"))
    if not logs:
        return "[no log file yet]"
    import json as _json
    try:
        entries = _json.loads(logs[-1].read_text())
        txt = "\n".join(e.get("data", "") for e in entries if e.get("data", "").strip())
    except Exception:
        txt = logs[-1].read_text(errors="replace")
    return _tail(txt, 8000)


def kaggle_submit(args, ctx):
    """Submit to the competition (gated). Two modes:
    - code competition: kernel_ref+kernel_version (submits kernel output)
    - normal competition: csv_path (uploads the CSV directly)"""
    if ctx.budget and not ctx.budget.can_submit():
        return "[BLOCKED by submit_gate: submission budget exhausted]"
    msg = args.get("message", "agent submission")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    try:
        if args.get("csv_path"):
            p = _safe(ctx, args["csv_path"])
            r = api.competition_submit(str(p), msg, ctx.slug)
        else:
            ref = args.get("kernel_ref") or ctx.kernel_ref
            r = api.competition_submit_code(
                args.get("file_name", "submission.csv"), msg, ctx.slug,
                kernel=ref, kernel_version=args.get("kernel_version"))
        if ctx.budget:
            ctx.budget.note_submit()
        return f"SUBMITTED: {r}"
    except Exception as e:  # noqa: BLE001
        body = getattr(getattr(e, "response", None), "text", "")
        return f"[submit error] {type(e).__name__}: {str(e)[:300]} {body[:300]}"


def kaggle_submissions(args, ctx):
    return _kg(["competitions", "submissions", "-c", ctx.slug])


# --------------------------------------------------------------------------- #
# memory & skills (repo-level, shared across runs)
# --------------------------------------------------------------------------- #
def memory_recall(args, ctx):
    mdir = ROOT / "memory" / "lessons"
    files = sorted(mdir.glob("*.md"))
    if not files:
        return "(memory empty)"
    q = args.get("query", "").lower()
    out = []
    for f in files:
        txt = f.read_text()
        if not q or q in txt.lower() or q in f.name.lower():
            out.append(f"### {f.name}\n{txt.strip()}")
    return _tail("\n\n".join(out) or "(no matching lessons)", 12000)


def memory_write(args, ctx):
    name = "".join(c if c.isalnum() or c in "-_" else "-" for c in args["name"])[:60]
    p = ROOT / "memory" / "lessons" / f"{name}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(args["content"])
    return f"lesson saved: memory/lessons/{name}.md"


def skill_list(args, ctx):
    sdir = ROOT / "skills"
    out = []
    for d in sorted(sdir.iterdir()):
        f = d / "SKILL.md"
        if f.exists():
            first = f.read_text().strip().splitlines()[0]
            out.append(f"- {d.name}: {first.lstrip('# ')}")
    return "\n".join(out) or "(no skills)"


def skill_load(args, ctx):
    f = ROOT / "skills" / args["name"] / "SKILL.md"
    if not f.exists():
        return f"[skill not found: {args['name']}] — use skill_list"
    return _tail(f.read_text(), 15000)


def phase_complete(args, ctx):
    return "PHASE_COMPLETE"  # sentinel; orchestrator intercepts


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def _t(name, desc, props, req, fn):
    return {"schema": {"name": name, "description": desc,
                       "input_schema": {"type": "object", "properties": props,
                                        "required": req}},
            "fn": fn}


TOOLS = [
    _t("run_python", "Execute Python code in the workspace (cwd=workspace). Returns stdout/stderr tail and exit code. Use for EDA, training, scoring. GPU available.",
       {"code": {"type": "string"}, "timeout_s": {"type": "integer", "description": "default 600"}}, ["code"], run_python),
    _t("run_bash", "Run a bash command in the workspace (ls, unzip, du...).",
       {"command": {"type": "string"}, "timeout_s": {"type": "integer"}}, ["command"], run_bash),
    _t("write_file", "Write a text file at a workspace-relative path.",
       {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"], write_file),
    _t("read_file", "Read a text file (workspace-relative). Optional offset/limit chars.",
       {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, ["path"], read_file),
    _t("list_dir", "List a workspace directory.",
       {"path": {"type": "string"}}, [], list_dir),
    _t("kaggle_download", "Download+unzip this competition's data into input/ (skips if already present).",
       {}, [], kaggle_download),
    _t("kaggle_push_kernel", "Push a kernel dir (with kernel-metadata.json) to Kaggle; starts a cloud run. Code competitions score notebooks, not CSVs.",
       {"kernel_dir": {"type": "string"}}, ["kernel_dir"], kaggle_push_kernel),
    _t("kaggle_kernel_status", "Status of the last pushed kernel (or given ref).",
       {"kernel_ref": {"type": "string"}}, [], kaggle_kernel_status),
    _t("kaggle_kernel_log", "Fetch the last pushed kernel's run log (after it finishes).",
       {"kernel_ref": {"type": "string"}}, [], kaggle_kernel_log),
    _t("kaggle_submit", "Submit the finished kernel's output to the competition. Budget-gated.",
       {"kernel_ref": {"type": "string"}, "kernel_version": {"type": "integer"},
        "message": {"type": "string"}, "file_name": {"type": "string"}}, [], kaggle_submit),
    _t("kaggle_submissions", "List this competition's submissions and scores.",
       {}, [], kaggle_submissions),
    _t("memory_recall", "Recall lessons from past runs/tasks (optionally filter by query).",
       {"query": {"type": "string"}}, [], memory_recall),
    _t("memory_write", "Save a durable lesson for future runs (name + markdown content).",
       {"name": {"type": "string"}, "content": {"type": "string"}}, ["name", "content"], memory_write),
    _t("skill_list", "List available skill playbooks.", {}, [], skill_list),
    _t("skill_load", "Load a skill playbook by name.", {"name": {"type": "string"}}, ["name"], skill_load),
    _t("phase_complete", "Declare the current phase's goal met. Include a short summary of artifacts produced.",
       {"summary": {"type": "string"}}, ["summary"], phase_complete),
]

SCHEMAS = [t["schema"] for t in TOOLS]
FNS = {t["schema"]["name"]: t["fn"] for t in TOOLS}
