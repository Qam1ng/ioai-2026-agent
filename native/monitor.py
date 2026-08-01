#!/usr/bin/env python3
"""Read a run's artifacts and report what each stage put in and got out.

A run leaves its evidence scattered — trace.jsonl, facts.jsonl, folds.json,
LKG/state.json, review verdicts, Kaggle's own record — and the interesting
questions cut across all of them. Was the first submission made before or after
the ruler existed? Did the solvers claim different angles? Did anything reach
the leaderboard that no script had scored? Answering those by reading raw logs
is how the last run's bugs went unnoticed until the following day.

    python -m native.monitor --slug <slug>            # human-readable
    python -m native.monitor --slug <slug> --json     # for rendering
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def collect(ws: Path) -> dict:
    trace = read_jsonl(ws / "trace.jsonl")
    board = read_jsonl(ws / "facts.jsonl")
    t0 = trace[0]["t"] if trace else 0
    at = lambda e: round((e.get("t", t0) - t0) / 60, 1)

    agents: dict[str, dict] = {}
    for e in trace:
        a = e.get("solver")
        if not a:
            continue
        d = agents.setdefault(a, {"rounds": 0, "cost": 0.0, "tools": 0,
                                  "turns": 0, "first": at(e), "last": at(e),
                                  "nudges": 0, "interrupts": 0})
        d["last"] = at(e)
        k = e.get("kind")
        if k == "tool":
            d["tools"] += 1
        elif k == "result":
            d["rounds"] += 1
            d["cost"] += e.get("cost", 0) or 0
            d["turns"] += e.get("turns", 0) or 0
        elif k == "nudge":
            d["nudges"] += 1
        elif k == "interrupt":
            d["interrupts"] += 1
    for d in agents.values():
        d["cost"] = round(d["cost"], 2)

    # The ordering that mattered most last time: was anything submitted before
    # there was a ruler to measure it with?
    ruler = None
    for f in ("metric.py", "folds.json"):
        p = ws / f
        if p.exists() and trace:
            m = round((p.stat().st_mtime - t0) / 60, 1)
            ruler = m if ruler is None else max(ruler, m)

    submits = [{"at": at(e), "candidate": e.get("candidate"),
                "oof": e.get("mean"), "scored": e.get("scored"),
                "before_ruler": ruler is not None and at(e) < ruler}
               for e in trace if e.get("kind") == "harness_submit"]

    folds = {}
    if (ws / "folds.json").exists():
        try:
            s = json.loads((ws / "folds.json").read_text())
            fold = [int(x) for x in s.get("fold", [])]
            folds = {"scheme": s.get("scheme"), "reason": s.get("reason", ""),
                     "units": len(fold),
                     "scored_folds": len({f for f in fold if f >= 0}),
                     "train_only": sum(1 for f in fold if f < 0)}
        except Exception:  # noqa: BLE001
            folds = {"scheme": "unreadable"}

    lkg = {}
    p = ws / "LKG" / "state.json"
    if p.exists():
        try:
            lkg = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            pass

    # The same block reason over and over is not eighteen events, it is one
    # stuck condition that nobody noticed for eighteen minutes. Surface it as
    # such: a repeated refusal means the gate and the run disagree, and the gate
    # is the thing more likely to be wrong.
    stuck = []
    counts: dict[str, list] = {}
    for e in trace:
        if e.get("kind") != "submit_blocked":
            continue
        key = str(e.get("reason", ""))[:70]
        counts.setdefault(key, []).append(at(e))
    for key, times in counts.items():
        if len(times) >= 3:
            stuck.append({"n": len(times), "from": times[0], "to": times[-1],
                          "reason": key})

    # Candidate scores as the submitter sees them: it scores any candidate with
    # an oof file directly, so the promote log lags behind it and reading only
    # the log makes solvers look later and quieter than they were.
    live = {}
    for d in sorted(ws.glob("solver_*/out/oof.npy")):
        live[d.parts[-3]] = round((d.stat().st_mtime - t0) / 60, 1) if t0 else None

    return {
        "workspace": str(ws),
        "elapsed_min": at(trace[-1]) if trace else 0,
        "stuck_gates": stuck,
        "candidate_files": live,
        "agents": agents,
        "ruler_ready_min": ruler,
        "folds": folds,
        "board": [{"at": round((r["ts"] - t0) / 60, 1) if t0 else 0,
                   "kind": r["kind"], "src": r["src"], "layer": r.get("layer"),
                   "text": r["text"]} for r in board],
        "claims": [r for r in board if r["kind"] == "claim"],
        "results": [r for r in board if r["kind"] == "result"],
        "submits": submits,
        "verdicts": [{"at": at(e), "candidate": e.get("candidate"),
                      "verdict": e.get("verdict"), "reason": e.get("reason")}
                     for e in trace if e.get("kind") == "verdict"],
        "blocked": [{"at": at(e), "candidate": e.get("candidate"),
                     "reason": e.get("reason")}
                    for e in trace if e.get("kind") == "submit_blocked"],
        "drifts": [{"at": at(e), "kind": e.get("kind_"), "detail": e.get("detail")}
                   for e in trace if e.get("kind") == "drift"],
        "lkg": lkg,
        "cost": round(sum(d["cost"] for d in agents.values()), 2),
    }


MARK = {"claim": "->", "result": "##", "failure": "xx", "format": "!!",
        "data": "..", "env": "~~"}


def render_board(d: dict, width: int = 96) -> str:
    """The board first, because it is where everything ends up.

    Agent-by-agent statistics say who was busy; the board says what the run
    actually learned, in the order it learned it, and it is the only place a
    finding by one solver becomes available to the others. Reading it top to
    bottom is reading the run.
    """
    L = ["board — everything the run has learned, in order",
         "  ->claim  ##score  xxfailure  !!format  ..data  ~~env", ""]
    if not d["board"]:
        return "\n".join(L + ["  (empty)"])
    for r in d["board"]:
        head = f"  +{r['at']:>4.1f}m {MARK.get(r['kind'], '  ')} {r['src']:<10}"
        body = r["text"]
        pad = " " * len(head)
        first, rest = body[:width], body[width:]
        L.append(head + first)
        while rest:
            L.append(pad + rest[:width])
            rest = rest[width:]
    return "\n".join(L)


def render(d: dict) -> str:
    L = []
    # Loudest first. A gate refusing the same thing repeatedly is the failure
    # mode that cost a whole run while every individual refusal looked routine.
    for s in d.get("stuck_gates", []):
        L += [f"!! STUCK GATE — refused {s['n']}x between +{s['from']:.0f}m and "
              f"+{s['to']:.0f}m, always for the same reason:",
              f"     {s['reason']}",
              "   A gate that keeps refusing is more likely wrong than the run "
              "is.", ""]
    L += [render_board(d), ""]
    # Claims are the thing the design is betting on: three solvers dividing
    # work nobody divided for them. If they read alike, the division failed.
    claims = d["claims"] or [r for r in d["board"]
                             if "claim" in r["text"][:12].lower()]
    L.append(f"claims on the board: {len(claims)} from "
             f"{len({c['src'] for c in claims})} solver(s)"
             + ("  <- fewer than one each; work may be duplicated"
                if len({c['src'] for c in claims}) < 3 else ""))
    silent = [a for a in d["agents"] if a.startswith("solver")
              and not any(r["src"] == a for r in d["board"])]
    if silent:
        L.append(f"posted nothing yet: {', '.join(silent)}")
    L.append("")
    L += [f"elapsed {d['elapsed_min']}min | cost ${d['cost']} | "
          f"ruler ready at {d['ruler_ready_min']}min"]
    if d["folds"]:
        f = d["folds"]
        L.append(f"split: {f.get('scheme')} | {f.get('units')} units | "
                 f"{f.get('scored_folds')} scored folds"
                 + (f" | {f['train_only']} train-only" if f.get("train_only") else "")
                 + (f"\n  reason: {f['reason']}" if f.get("reason") else ""))
    L.append("\nagents:")
    for a, v in sorted(d["agents"].items()):
        L.append(f"  {a:<12} rounds={v['rounds']} turns={v['turns']} "
                 f"tools={v['tools']} ${v['cost']}"
                 + (f" nudged×{v['nudges']}" if v["nudges"] else "")
                 + (f" interrupted×{v['interrupts']}" if v["interrupts"] else ""))
    if d["claims"]:
        L.append("\nclaims (did they diverge?):")
        for c in d["claims"]:
            L.append(f"  {c['src']}: {c['text'][:110]}")
    if d["results"]:
        L.append("\nscores announced:")
        for r in d["results"][-8:]:
            L.append(f"  {r['text'][:110]}")
    if d["verdicts"]:
        L.append("\nevaluator verdicts:")
        for v in d["verdicts"]:
            L.append(f"  +{v['at']}min {v['candidate']}: {v['verdict']} — "
                     f"{(v.get('reason') or '')[:70]}")
    if d["blocked"]:
        L.append("\nheld back:")
        for b in d["blocked"][-5:]:
            L.append(f"  +{b['at']}min {b['candidate']}: {b['reason'][:80]}")
    if d["submits"]:
        L.append("\nsubmitted:")
        for s in d["submits"]:
            L.append(f"  +{s['at']}min {s['candidate']} oof={s['oof']} "
                     + ("!! BEFORE THE RULER EXISTED" if s["before_ruler"] else "ok"))
    else:
        L.append("\nsubmitted: nothing yet")
    if d["drifts"]:
        L.append("\ndrift detected:")
        for x in d["drifts"]:
            L.append(f"  +{x['at']}min {x['detail'][:100]}")
    if d["lkg"]:
        for tier in ("local", "confirmed"):
            v = d["lkg"].get(tier)
            if v:
                L.append(f"\nLKG {tier}: {v['candidate']} @ "
                         f"{v.get('mean', v.get('lb'))}")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    ws = ROOT / "workspace" / f"hearsay-{a.slug}"
    if not ws.exists():
        print(f"no workspace at {ws}")
        return 1
    d = collect(ws)
    print(json.dumps(d) if a.json else render(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
