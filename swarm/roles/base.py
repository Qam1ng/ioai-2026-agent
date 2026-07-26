"""The Role: a bounded tool-use loop with its own prompt, tools and budget.

Every specialist in the swarm is an instance of this. A role is deliberately
*short-lived* — it is handed a narrow instruction, works with a restricted tool
set, emits a structured JSON report, and exits. That is what keeps context from
exploding (the failure AIBuildAI names as the reason to split a single agent
into sub-agents) and what makes each role's output auditable in isolation.

Roles never call other roles. They read and write the blackboard.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.providers.base import ToolResult, get_provider  # noqa: E402
from agent.tools import registry as toolreg  # noqa: E402

from ..budget import PodBudget  # noqa: E402
from ..bus import Blackboard  # noqa: E402
from ..config import ModelSpec, SwarmConfig  # noqa: E402

PROMPT_DIR = ROOT / "prompts" / "swarm"

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass
class RoleResult:
    """What a role hands back to the pod."""

    role: str
    ok: bool
    text: str = ""
    report: dict = field(default_factory=dict)
    steps: int = 0
    elapsed_s: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "ok": self.ok,
            "report": self.report,
            "steps": self.steps,
            "elapsed_s": round(self.elapsed_s, 1),
            "error": self.error,
            "text": self.text[-4000:],
        }


def extract_json(text: str) -> dict:
    """Pull the role's JSON report out of its final message.

    Roles are told to end with a fenced ```json block. Models drift, so we fall
    back to the last balanced brace span before giving up.
    """
    blocks = _JSON_BLOCK.findall(text or "")
    for blk in reversed(blocks):
        try:
            obj = json.loads(blk)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    # Fallback: scan for the last balanced {...} span.
    depth = 0
    start = -1
    spans: list[tuple[int, int]] = []
    for i, ch in enumerate(text or ""):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append((start, i + 1))
    for s, e in reversed(spans):
        try:
            obj = json.loads(text[s:e])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return {}


def load_prompt(name: str) -> str:
    p = PROMPT_DIR / f"{name}.md"
    if not p.exists():
        raise FileNotFoundError(f"missing role prompt: {p}")
    return p.read_text()


class Role:
    """Runs one specialist to completion."""

    #: Subclasses restrict what the role may touch. ``None`` means all tools.
    tools: tuple[str, ...] | None = None
    #: Step ceiling for the role's internal loop.
    max_steps: int = 24
    #: Prompt file stem under ``prompts/swarm/``.
    prompt_name: str = ""

    def __init__(
        self,
        name: str,
        cfg: SwarmConfig,
        bb: Blackboard,
        budget: PodBudget,
        ctx: Any = None,
        on_event: Callable[[str, dict], None] | None = None,
        sandbox: Path | str | None = None,
    ):
        self.name = name
        self.cfg = cfg
        self.bb = bb
        self.budget = budget
        self.on_event = on_event
        self.spec: ModelSpec = cfg.model_for(name)
        # ``claude_code`` roles drive the CLI, not an API client; constructing a
        # provider for them would demand an API key they never use.
        self.provider = (
            None if self.spec.backend == "claude_code" else self._build_provider(self.spec)
        )
        self.ctx = ctx or toolreg.Ctx(
            workspace=bb.ws, slug=cfg.slug, budget=budget, trace=_TraceShim(bb)
        )
        self.sandbox = Path(sandbox) if sandbox else bb.ws
        self.session_id = ""

    # ------------------------------------------------------------ plumbing
    @staticmethod
    def _build_provider(spec: ModelSpec):
        return get_provider(
            spec.backend,
            spec.model or None,
            base_url=spec.base_url or None,
            api_key_env=spec.api_key_env or None,
        )

    def _schemas(self) -> list[dict]:
        if self.tools is None:
            return toolreg.SCHEMAS
        allow = set(self.tools)
        return [s for s in toolreg.SCHEMAS if s["name"] in allow]

    def _call_tool(self, name: str, args: dict) -> str:
        fn = toolreg.FNS.get(name)
        if fn is None:
            return f"[tool error] unknown tool: {name}"
        try:
            return fn(args, self.ctx)
        except Exception as exc:  # tool errors are information, not crashes
            return f"[tool error] {type(exc).__name__}: {exc}"

    def _emit(self, kind: str, **fields: Any) -> None:
        self.bb.event(kind, role=self.name, **fields)
        if self.on_event:
            try:
                self.on_event(kind, {"role": self.name, **fields})
            except Exception:
                pass

    # ------------------------------------------------------------- system
    def system_prompt(self) -> str:
        base = load_prompt(self.prompt_name or self.name)
        card = self.bb.get_task_card()
        return base.format(
            slug=self.cfg.slug,
            task_type=(card.task_type if card else "unknown"),
            metric=(card.metric_name if card else "unknown"),
        )

    def opening_message(self, instructions: str, context: dict | None) -> str:
        parts = [self.budget.status_line(), ""]
        if context:
            parts += [
                "## Shared state",
                "```json",
                json.dumps(context, indent=2, ensure_ascii=False, default=str)[:12000],
                "```",
                "",
            ]
        parts += ["## Your instruction", instructions]
        return "\n".join(parts)

    # ---------------------------------------------------------------- run
    def run(self, instructions: str, context: dict | None = None) -> RoleResult:
        """Execute the role. The backend is a configuration choice, not a class.

        Any role can run either as a headless Claude Code process (a real
        filesystem and edit-run-fix loop, best for writing code) or on the
        plain API path (controlled, auditable context, best for judgement, and
        able to use any model). Which one is decided per role in the config.
        """
        if self.spec.backend == "claude_code":
            return self._run_claude_code(instructions, context)
        return self._run_api(instructions, context)

    def _run_api(self, instructions: str, context: dict | None = None) -> RoleResult:
        t0 = time.time()
        system = self.system_prompt()
        schemas = self._schemas()
        history: list = []
        self.provider.append_user(history, self.opening_message(instructions, context))
        self._emit("role_start", instruction=instructions[:400])

        last_text = ""
        for step_i in range(self.max_steps):
            halt = self.budget.exhausted()
            if halt:
                return RoleResult(
                    self.name, False, last_text, {}, step_i, time.time() - t0, f"budget: {halt}"
                )
            try:
                step = self.provider.step(system, history, schemas, max_tokens=self.spec.max_tokens)
            except Exception as exc:
                return RoleResult(
                    self.name,
                    False,
                    last_text,
                    {},
                    step_i,
                    time.time() - t0,
                    f"provider error: {type(exc).__name__}: {exc}",
                )

            self.budget.note_llm(step.usage, self.provider.cost_usd(step.usage))
            if step.text:
                last_text = step.text
            self.provider.append_assistant(history, step)
            self._emit(
                "llm",
                step=step_i,
                tokens_in=step.usage.get("input_tokens", 0),
                tokens_out=step.usage.get("output_tokens", 0),
                n_tools=len(step.tool_calls),
            )

            if not step.tool_calls:
                report = extract_json(last_text)
                if report:
                    return RoleResult(
                        self.name, True, last_text, report, step_i + 1, time.time() - t0
                    )
                self.provider.append_user(
                    history,
                    "Finish by emitting your report as a single fenced ```json block, "
                    "or keep working with tools.",
                )
                continue

            results = []
            for call in step.tool_calls:
                out = self._call_tool(call.name, call.input)
                self._emit("tool", tool=call.name, chars=len(out or ""))
                results.append(
                    ToolResult(call.id, out, is_error=str(out).startswith("[tool error]"))
                )
            self.provider.append_tool_results(history, results)

        report = extract_json(last_text)
        return RoleResult(
            self.name,
            bool(report),
            last_text,
            report,
            self.max_steps,
            time.time() - t0,
            "" if report else "step cap reached without a report",
        )

    # ------------------------------------------------------- claude code
    #: Claude Code tool names (not registry tool names). WebFetch/WebSearch are
    #: withheld: the submitted kernel has no internet, so a solution that
    #: quietly depends on a downloaded artifact cannot run on Kaggle.
    cc_tools: tuple[str, ...] = ("Read", "Write", "Edit", "Bash", "Glob", "Grep")
    max_turns: int = 60
    timeout_s: float = 1800.0

    def _run_claude_code(self, instructions: str, context: dict | None = None) -> RoleResult:
        from ..runners import ClaudeCodeUnavailable, run_claude_code

        t0 = time.time()
        self._emit("role_start", instruction=instructions[:400], runner="claude_code")
        try:
            res = run_claude_code(
                self.opening_message(instructions, context),
                cwd=self.sandbox,
                system_append=self.system_prompt(),
                allowed_tools=self.cc_tools,
                model=self.spec.model,
                max_turns=self.max_turns,
                timeout_s=self.timeout_s,
                resume_session=self.session_id,
                add_dirs=(self.bb.ws,) if self.sandbox != self.bb.ws else (),
            )
        except ClaudeCodeUnavailable as exc:
            return RoleResult(self.name, False, "", {}, 0, time.time() - t0, str(exc))

        # Claude Code bills itself; fold its cost into the pod budget so the
        # efficiency figures the rules require us to publish stay complete.
        self.budget.note_llm(res.usage, res.cost_usd)
        if res.session_id:
            self.session_id = res.session_id
        self._emit(
            "claude_code",
            turns=res.num_turns,
            cost_usd=round(res.cost_usd, 4),
            duration_s=round(res.duration_s, 1),
            ok=res.ok,
        )

        report = extract_json(res.text)
        return RoleResult(
            self.name,
            ok=res.ok and bool(report),
            text=res.text,
            report=report,
            steps=res.num_turns,
            elapsed_s=res.duration_s,
            error=res.error or ("" if report else "no JSON report emitted"),
        )


class ClaudeCodeRole(Role):
    """A role pinned to the Claude Code backend regardless of configuration.

    Rarely needed — the backend is normally chosen in ``configs/*.yaml`` — but
    useful when a role only makes sense with a filesystem.
    """

    def run(self, instructions: str, context: dict | None = None) -> RoleResult:
        return self._run_claude_code(instructions, context)


class _TraceShim:
    """Adapts the blackboard to the ``trace.log(kind, **kw)`` shape tools expect."""

    def __init__(self, bb: Blackboard):
        self.bb = bb

    def log(self, kind: str, **kw: Any) -> None:
        self.bb.event(kind, **kw)
