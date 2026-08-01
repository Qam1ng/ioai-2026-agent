"""事实板 — append-only JSONL. No LLM, no heartbeat, no push.

The pool broadcasts *objective discoveries*, never *status*. Status sharing
makes independent solvers converge on whoever is currently ahead, and diversity
is the only reason to run N of them at all.

Enforcement lives here, not in a prompt:
  * ``kind`` is a fixed enum — anything else is rejected
  * ``text`` is capped — a "fact" that needs 300 chars is a progress report
  * there is no score field to fill in

Two layers. ``task`` facts die with the competition; ``day`` facts (which
accelerator works, package versions, quota burn rate, submission latency)
survive across the three problems of a competition day, which is the part no
single-problem architecture has.

Readers hold a cursor. The harness piggybacks unseen facts onto tool results,
so a solver hears at its own natural pauses — never polls, never gets
interrupted, and can be 20 minutes out of sync with no loss.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

# `claim` is how the solvers divide work they were not given a division of: a
# one-line statement of the angle someone is taking, so the others can go
# elsewhere.
#
# `result` carries what an approach was actually worth. Knowing which
# directions have oil in them is too useful to withhold — but only the harness
# and the evaluator may post it, because a number is only comparable if it came
# from the shared folds and the frozen metric. A solver quoting its own score
# would be quoting something unverified, measured on who-knows-what split, and
# would turn the board into a ranking of people rather than a map of the
# problem.
KINDS = ("env", "data", "format", "failure", "claim", "result")
VERIFIED_ONLY = ("result",)
TRUSTED = ("harness", "evaluator", "recon")
MAX_TEXT = 300
LAYERS = ("task", "day")


class Board:
    def __init__(self, task_path: Path, day_path: Path) -> None:
        self.paths = {"task": Path(task_path), "day": Path(day_path)}
        for p in self.paths.values():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch(exist_ok=True)
        self._cursors: dict[str, int] = {}  # reader -> facts already seen

    # ---------------------------------------------------------------- write
    def post(self, kind: str, text: str, src: str, layer: str = "task") -> str:
        if kind not in KINDS:
            return f"[rejected] kind must be one of {KINDS}, got {kind!r}"
        if kind in VERIFIED_ONLY and src not in TRUSTED:
            return (f"[rejected] '{kind}' may only be posted by {TRUSTED}. "
                    f"Scores reach the board through the evaluator, which "
                    f"computes them on the shared folds — a self-reported "
                    f"number is not comparable to anyone else's.")
        if layer not in LAYERS:
            return f"[rejected] layer must be one of {LAYERS}, got {layer!r}"
        text = " ".join(str(text).split())
        if not text:
            return "[rejected] empty text"
        if len(text) > MAX_TEXT:
            return (f"[rejected] {len(text)} chars > {MAX_TEXT}. Post the fact, "
                    "not the narrative. Split it or cut it down.")
        rec = {"ts": int(time.time()), "layer": layer, "kind": kind,
               "src": src, "text": text}
        # Append-only: O_APPEND writes of a single short line are atomic enough
        # for concurrent solvers on a local filesystem.
        with self.paths[layer].open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return f"[posted] {kind}: {text[:80]}"

    # ----------------------------------------------------------------- read
    def _all(self) -> list[dict]:
        out: list[dict] = []
        for layer, p in self.paths.items():
            try:
                for line in p.read_text().splitlines():
                    if line.strip():
                        try:
                            out.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            except FileNotFoundError:
                continue
        out.sort(key=lambda r: r["ts"])
        return out

    def unseen(self, reader: str) -> list[dict]:
        """Facts this reader has not been shown yet, excluding its own."""
        seen = self._cursors.get(reader, 0)
        allf = self._all()
        self._cursors[reader] = len(allf)
        return [r for r in allf[seen:] if r.get("src") != reader]

    def render_unseen(self, reader: str) -> str:
        """The block the harness staples onto a tool result. Empty if nothing."""
        new = self.unseen(reader)
        if not new:
            return ""
        lines = [f"  {r['kind']:<8}({r['src']}) {r['text']}" for r in new]
        return ("\n\n--- facts board: %d new ---\n" % len(new)) + "\n".join(lines)

    def render_all(self) -> str:
        allf = self._all()
        if not allf:
            return "(facts board empty)"
        return "\n".join(f"{r['layer']:<5} {r['kind']:<8}({r['src']}) {r['text']}"
                         for r in allf)


_BOARD: Board | None = None


def init(workspace: Path, day_file: str | None = None) -> Board:
    """Task layer lives in the workspace; day layer is shared across problems."""
    global _BOARD
    day = Path(day_file or os.environ.get(
        "IOAI_DAY_FACTS", Path.home() / ".ioai" / "day_facts.jsonl"))
    _BOARD = Board(Path(workspace) / "facts.jsonl", day)
    return _BOARD


def board() -> Board:
    if _BOARD is None:
        raise RuntimeError("facts.init() not called")
    return _BOARD
