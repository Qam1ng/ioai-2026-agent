#!/usr/bin/env python3
"""Self-test for the deterministic layer, on data with a hand-computed answer.

These four scripts are the trust anchor: if evaluate.py is wrong, every
downstream decision is wrong and nothing else in the system can notice. So they
are checked against numbers worked out by hand, not against themselves.

    python -m native.selftest
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAIL: list[str] = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def run(*a: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", *a], cwd=str(ROOT),
                          capture_output=True, text=True)


def test_facts(ws: Path) -> None:
    print("\n[facts board]")
    from native import facts
    b = facts.init(ws, day_file=str(ws / "day.jsonl"))

    check("bad kind rejected", b.post("gossip", "x", "s")[:10], "[rejected]")
    check("empty rejected", b.post("env", "   ", "s")[:10], "[rejected]")
    check("over-long rejected", b.post("env", "x" * 301, "s")[:10], "[rejected]")
    check("valid accepted", b.post("env", "P100 OOMs, T4 works", "solver_a")[:8],
          "[posted]")
    b.post("data", "47 duplicate rows", "solver_b")
    b.post("env", "kaggle image has torch 2.4", "solver_a", layer="day")

    # A reader never hears its own voice, and hears everyone else exactly once.
    first = b.unseen("solver_a")
    check("a sees others only", sorted(r["src"] for r in first), ["solver_b"])
    check("a re-read is empty", b.unseen("solver_a"), [])
    check("b sees a's two", len(b.unseen("solver_b")), 2)

    # Desync is harmless: a solver that has been quiet for a long time simply
    # receives the backlog in one go.
    b.post("failure", "shallow CNN plateaus at 0.71", "solver_b")
    b.post("format", "submission needs header row", "solver_a")
    check("late reader gets backlog", len(b.unseen("solver_c")), 5)

    # Directions are shareable so the solvers can avoid each other; numbers are
    # not, unless the evaluator computed them on the shared folds.
    check("a solver may claim a direction",
          b.post("claim", "attacking the metric's asymmetry", "solver_a")[:8],
          "[posted]")
    check("a solver may not post a score",
          b.post("result", "my OOF is 0.91", "solver_a")[:10], "[rejected]")
    check("the harness may", b.post("result", "approach X -> 0.84", "harness")[:8],
          "[posted]")
    check("so may the evaluator",
          b.post("result", "approach Y -> 0.79", "evaluator")[:8], "[posted]")

    # The cap is there to stop agents narrating, and an agent that trips it can
    # read the rejection and repost shorter. A script cannot. The harness
    # announced "this competition is kernel-only, build out/kernel/" in 356
    # characters, never inspected the rejection it got back, and three solvers
    # spent a full run producing candidates with no route to the leaderboard.
    # For machine sources the cap now truncates, loudly, instead of dropping.
    d = ws / "trunc"
    b2 = facts.init(d, day_file=str(d / "day.jsonl"))
    check("over-long from a solver is still refused",
          b2.post("format", "y" * 400, "solver_a")[:10], "[rejected]")
    check("over-long from the harness lands anyway",
          b2.post("format", "y" * 400, "harness")[:8], "[posted]")
    check("...truncated, not dropped",
          [len(f["text"]) for f in b2.unseen("reader")], [facts.MAX_TEXT - 1])
    check("same for recon", b2.post("data", "z" * 400, "recon")[:8], "[posted]")
    facts.init(ws, day_file=str(ws / "day.jsonl"))  # restore for later tests


def test_pipeline(ws: Path) -> None:
    print("\n[folds -> evaluate -> promote]")
    import numpy as np
    import pandas as pd

    # 40 rows, 4 balanced classes. Hand-checkable.
    n, k = 40, 4
    y = np.repeat(np.arange(4), 10)
    pd.DataFrame({"id": range(n), "label": y}).to_csv(ws / "train.csv", index=False)

    r = run("native.scripts.folds", "--labels", str(ws / "train.csv"),
            "--target", "label", "--id", "id", "--k", str(k),
            "--workspace", str(ws))
    print("   ", (r.stdout or r.stderr).strip().splitlines()[-1][:110])
    spec = json.loads((ws / "folds.json").read_text())
    check("all rows assigned", sorted(set(spec["fold"])), list(range(k)))
    check("folds balanced", sorted(np.bincount(spec["fold"]).tolist()), [10] * k)
    # Stratification: every fold must hold one row of each class.
    per = {(f, l) for f, l in zip(spec["fold"], spec["labels"])}
    check("stratified", len(per), k * 4)

    (ws / "metric.py").write_text(
        "def score(y_true, y_pred):\n"
        "    import numpy as np\n"
        "    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())\n")

    # Candidate A: exactly the first 30 of 40 correct -> pooled accuracy 0.75.
    pred = y.copy()
    pred[30:] = (pred[30:] + 1) % 4
    (ws / "solver_a" / "out").mkdir(parents=True, exist_ok=True)
    np.save(ws / "solver_a" / "out" / "oof.npy", pred)

    r = run("native.scripts.evaluate", "--candidate", "solver_a", "--workspace", str(ws))
    res = json.loads(r.stdout.strip().splitlines()[-1])
    check("pooled accuracy", res["pooled"], 0.75)
    check("fold mean equals pooled (equal folds)", round(res["mean"], 6), 0.75)

    # Candidate B: strictly worse (20/40) -> must be refused by the gate.
    predb = y.copy()
    predb[20:] = (predb[20:] + 1) % 4
    (ws / "solver_b" / "out").mkdir(parents=True, exist_ok=True)
    np.save(ws / "solver_b" / "out" / "oof.npy", predb)
    r = run("native.scripts.evaluate", "--candidate", "solver_b", "--workspace", str(ws))
    check("B scores 0.5", json.loads(r.stdout.strip().splitlines()[-1])["pooled"], 0.5)

    r = run("native.scripts.promote", "--candidate", "solver_a", "--workspace", str(ws))
    check("first candidate promoted",
          json.loads(r.stdout.strip().splitlines()[-1])["decision"], "PROMOTED")
    r = run("native.scripts.promote", "--candidate", "solver_b", "--workspace", str(ws))
    check("no-regression gate holds",
          json.loads(r.stdout.strip().splitlines()[-1])["decision"], "REJECTED")

    # A self-reported score is not a score.
    (ws / "solver_c" / "out").mkdir(parents=True, exist_ok=True)
    (ws / "solver_c" / "out" / "score.json").write_text(
        json.dumps({"status": "ok", "mean": 0.99, "std": 0.0}))
    r = run("native.scripts.promote", "--candidate", "solver_c", "--workspace", str(ws))
    d = json.loads(r.stdout.strip().splitlines()[-1])
    check("hand-written score.json cannot beat LKG", d["decision"], "REJECTED")

    # Kaggle-confirmed tier is independent of the local one.
    r = run("native.scripts.promote", "--confirm", "solver_b", "--lb", "0.61",
            "--workspace", str(ws))
    check("confirmed tier accepts first entry",
          json.loads(r.stdout.strip().splitlines()[-1])["decision"], "PROMOTED")
    st = json.loads((ws / "LKG" / "state.json").read_text())
    check("two tiers tracked separately",
          (st["local"]["candidate"], st["confirmed"]["candidate"]),
          ("solver_a", "solver_b"))


def test_folds_schemes(ws: Path) -> None:
    """Split choice, and the hatch for schemes the built-ins cannot express.

    Restricting everyone to shuffled k-fold would silently mis-score any
    temporal task — a shuffled split on time-ordered rows lets every model see
    its own future. So a declared custom split is allowed; what is not allowed
    is re-cutting the split after candidates have been scored on it, which is
    where the dishonesty would actually live.
    """
    print("\n[folds schemes]")
    import pandas as pd
    g = ws / "fold"
    g.mkdir(exist_ok=True)

    def build(y, *extra):
        (g / "folds.json").unlink(missing_ok=True)
        f = g / "tr.csv"
        pd.DataFrame({"id": range(len(y)), "y": y}).to_csv(f, index=False)
        r = run("native.scripts.folds", "--labels", str(f), "--target", "y",
                "--id", "id", "--k", "5", "--workspace", str(g), *extra)
        return r.stdout + r.stderr

    check("continuous float -> plain kfold",
          "scheme=kfold" in build([i * 0.1 for i in range(50)]), True)
    check("few int classes -> stratified",
          "scheme=stratified" in build([i % 4 for i in range(50)]), True)
    check("floats that repeat as labels -> stratified",
          "scheme=stratified" in build([float(i % 3) for i in range(50)]), True)

    spec = g / "c.json"
    spec.write_text(json.dumps(
        {"fold": [i // 10 for i in range(50)],
         "reason": "temporal: rows are time-ordered"}))
    out = build([i * 0.1 for i in range(50)], "--custom", str(spec))
    check("declared custom split accepted", "scheme=custom" in out, True)
    check("  and its reason is recorded",
          json.loads((g / "folds.json").read_text())["reason"] != "", True)

    # Frozen once written — this is the property that keeps every score in a
    # run comparable, and no restriction on scheme could substitute for it.
    f = g / "tr.csv"
    again = run("native.scripts.folds", "--labels", str(f), "--target", "y",
                "--id", "id", "--k", "5", "--workspace", str(g))
    check("rewriting a written split is refused",
          "frozen" in (again.stdout + again.stderr), True)

    for bad, label in [([0] * 50, "a single fold"),
                       ([0] * 35 + [1] * 15, "one fold holding 70%"),
                       ([0, 1] * 20, "the wrong number of rows")]:
        spec.write_text(json.dumps({"fold": bad, "reason": "x"}))
        out = build([i * 0.1 for i in range(50)], "--custom", str(spec))
        check(f"  rejects {label}", "rejected" in out, True)

    spec.write_text(json.dumps({"fold": [i // 10 for i in range(50)]}))
    out = build([i * 0.1 for i in range(50)], "--custom", str(spec))
    check("  rejects a custom split with no stated reason",
          "requires --reason" in out, True)

    # A single holdout, for tasks where one training run is all the window
    # affords. -1 means train-only: present in the split, never scored.
    spec.write_text(json.dumps(
        {"fold": [-1] * 40 + [0] * 10,
         "reason": "single holdout: one training run fits the window, five do not"}))
    out = build([i * 0.1 for i in range(50)], "--custom", str(spec))
    check("single holdout is expressible", "train_only=40" in out, True)

    from native.scripts.checkfolds import validate_spec
    ok = {"ids": list(range(10)), "fold": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
          "labels": [0] * 10, "scheme": "kfold"}
    check("a plain kfold needs no reason", validate_spec(ok), [])
    check("holding units out does need one",
          any("no stated reason" in p for p in
              validate_spec({**ok, "fold": [-1] * 6 + [0, 0, 1, 1]})), True)
    check("mismatched lengths caught",
          any("entries but" in p for p in validate_spec({**ok, "labels": [0] * 3})),
          True)
    check("everything held out is caught",
          any("no unit is scored" in p for p in
              validate_spec({**ok, "fold": [-1] * 10, "reason": "x"})), True)
    check("duplicate ids caught",
          any("duplicates" in p for p in validate_spec({**ok, "ids": [1] * 10})),
          True)

    # How to cut a split is knowledge, not code. It lives in a skill the
    # evaluator loads, so a task shape nobody anticipated is a page to write
    # rather than another flag on a generator.
    from agent.tools import registry as Rg
    c = Rg.Ctx(workspace=g, slug="x", budget=None, trace=None)
    check("the playbook is listed", "validation-split" in Rg.skill_list({}, c), True)
    sk = Rg.skill_load({"name": "validation-split"}, c)
    for topic in ("temporal", "Repeated subjects", "provided validation set",
                  "not a table row", "checkfolds"):
        check(f"  playbook covers {topic[:24]}", topic in sk, True)
    import native.main as Mn
    check("evaluator prompt sends them to it",
          'skill_load(name="validation-split")' in Mn.EVALUATOR, True)


def test_recon(ws: Path) -> None:
    print("\n[recon]")
    import pandas as pd
    inp = ws / "input"
    inp.mkdir(exist_ok=True)
    # Train ordered by target: recon must flag the ordering leak.
    pd.DataFrame({"id": range(40), "f": range(40),
                  "label": [0] * 20 + [1] * 20}).to_csv(inp / "train.csv", index=False)
    pd.DataFrame({"id": range(5), "f": range(5)}).to_csv(inp / "test.csv", index=False)
    pd.DataFrame({"id": range(5), "label": [0] * 5}).to_csv(
        inp / "sample_submission.csv", index=False)

    r = run("native.scripts.recon", "--input", str(inp), "--workspace", str(ws))
    out = r.stdout + r.stderr
    check("ordering leak detected", "LEAK" in out and "row index" in out, True)
    check("submission format captured", "columns exactly" in out, True)
    check("train/test row overlap found", "test rows are identical" in out, True)


def test_configs() -> None:
    """Nobody arrives with an approach already chosen for them.

    Solvers used to be handed a dimension each — model, data, calibration —
    decided before anyone had read the task. On chicken counting the win was
    that the metric is asymmetric and the target is the density map's sum,
    which is none of those three, so two solvers were aimed at empty ground.
    Division of labour is theirs to negotiate on the board instead.
    """
    print("\n[configurations]")
    from native.prompts import BOARD_MULTI, BOARD_SOLO, CONTRACT

    def render(n: int) -> str:
        peers = ("You are the only solver on this problem." if n == 1 else
                 f"Your working directory is yours alone; {n} solvers are "
                 f"working on this problem in parallel, none of them told "
                 f"what to do.")
        return CONTRACT.format(slug="x", peers=peers,
                               board=(BOARD_SOLO if n == 1
                                      else BOARD_MULTI.format(n=n)),
                               budget="[budget] ...", deadline_min=30)

    solo, multi = render(1), render(3)
    check("no approach is assigned", "Nobody has assigned you an approach" in multi, True)
    for gone in ("Model and representation", "Data and augmentation",
                 "Safe baseline first"):
        check(f"  old brief gone: {gone[:22]}", gone in multi, False)

    check("multi announces peers", "3 solvers are working" in multi, True)
    check("solo claims no peers", "solvers are working" in solo, False)
    check("solo board is the notebook variant", "only solver" in solo, True)

    # Everything a solver could act on is shared, including what each direction
    # turned out to be worth. The line is provenance, not secrecy: a score
    # computed on the shared folds is a measurement, a self-reported one is a
    # boast measured on who-knows-what split.
    check("multi asks for a claim", "post it as a `claim`" in multi, True)
    check("multi warns about duplication", "Duplicated effort" in multi, True)
    check("results are shared", "`result` facts" in multi, True)
    check("solvers cannot post their own scores",
          "cannot post scores yourself" in multi, True)
    check("no unfilled placeholders", "{" in solo or "{" in multi, False)

    from native.facts import KINDS, TRUSTED, VERIFIED_ONLY, Board
    check("claim is postable", "claim" in KINDS, True)
    check("result is postable", "result" in KINDS, True)
    check("result is provenance-gated", VERIFIED_ONLY, ("result",))

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        b = Board(Path(td) / "t.jsonl", Path(td) / "d.jsonl")
        check("a solver cannot post a result",
              b.post("result", "I hit 0.99", "solver_a")[:10], "[rejected]")
        check("the evaluator can", b.post("result", "0.84 on folds", "evaluator")[:8],
              "[posted]")
        check("so can the harness", b.post("result", "0.84 on folds", "harness")[:8],
              "[posted]")
        check("solvers keep the other kinds",
              b.post("failure", "shallow CNN plateaus", "solver_a")[:8], "[posted]")
        check("and can claim a direction",
              b.post("claim", "going after the metric asymmetry", "solver_b")[:8],
              "[posted]")
    check("results reach the board via the evaluator",
          "the evaluator computes those on the shared folds" in multi, True)
    check("solvers cannot post their own numbers",
          "You cannot post scores yourself" in multi, True)


def test_submit_gate() -> None:
    """The rule that cost us the last run: never spend a slot on something no
    script has scored, and never wait so long for perfection that nothing goes
    at all."""
    print("\n[submission gate]")
    from native.main import should_submit

    B = {"mean": 0.87, "std": 0.02, "candidate": "solver_a"}
    kw = dict(min_candidates=3, fallback_frac=0.55, sent=0, allowance=3,
              sent_best=None)

    def ok(**over):
        return should_submit(**{**kw, **over})[0]

    check("nothing scored -> no send", ok(n_scored=0, frac=0.1, best=None), False)
    check("1 of 3 scored, early -> hold", ok(n_scored=1, frac=0.1, best=B), False)
    check("2 of 3 scored, early -> hold", ok(n_scored=2, frac=0.3, best=B), False)
    check("3 scored -> send", ok(n_scored=3, frac=0.3, best=B), True)
    # A stalled solver must not turn "wait for three" into a zero.
    check("time override rescues a stall",
          ok(n_scored=1, frac=0.6, best=B), True)
    check("override needs something scored",
          ok(n_scored=0, frac=0.9, best=None), False)
    # Don't spend a slot to re-send what is already up there.
    check("no improvement -> hold",
          ok(n_scored=3, frac=0.8, best=B, sent_best=0.87), False)
    check("real improvement -> send",
          ok(n_scored=3, frac=0.8, best={**B, "mean": 0.90}, sent_best=0.87), True)
    # Reserve: run 2 spent all five slots in the first four minutes.
    check("early reserve blocks a second early send",
          ok(n_scored=3, frac=0.2, best=B, sent=1, allowance=3), False)
    check("same send allowed once past halfway",
          ok(n_scored=3, frac=0.6, best=B, sent=1, allowance=3), True)


def test_solver_cannot_submit() -> None:
    print("\n[solvers have no submit tool]")
    from native import tools as T
    from native.prompts import CONTRACT
    names = [n for n, *_ in T._KAGGLE]
    check("kaggle_submit not offered to solvers", "kaggle_submit" in names, False)
    check("kaggle_submissions still readable", "kaggle_submissions" in names, True)
    check("contract says the harness submits",
          "You have no submit tool" in CONTRACT, True)


def test_format_gate(ws: Path) -> None:
    """The floor under the evaluator's review.

    The evaluator gates submission because validity needs judgement — it found by
    reading the organisers' code that one negative pixel voids a whole chicken
    submission. But a gate that can hang is a way to score zero, so these
    mechanical checks always run and the submitter proceeds on them alone if the
    review times out.
    """
    print("\n[submission format gate]")
    import pandas as pd
    from native.scripts.check_format import check

    g = ws / "fmt"
    (g / "input").mkdir(parents=True, exist_ok=True)
    (g / "c" / "out").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": range(5), "v": [0.0] * 5}).to_csv(
        g / "input" / "sample_submission.csv", index=False)

    def write(df):
        df.to_csv(g / "c" / "out" / "submission.csv", index=False)

    write(pd.DataFrame({"id": range(5), "v": [1.0, 2, 3, 4, 5]}))
    check_("a well-formed submission passes", check(g, "c")["ok"], True)

    write(pd.DataFrame({"id": range(3), "v": [1.0, 2, 3]}))
    r = check(g, "c")
    check_("wrong row count is fatal", "3 rows, sample has 5" in
           " ".join(r["fatal"]), True)

    write(pd.DataFrame({"v": [1.0] * 5, "id": range(5)}))
    check_("column order caught", "order matters" in
           " ".join(check(g, "c")["problems"]), True)

    write(pd.DataFrame({"id": range(5), "v": [float("nan"), 2, 3, 4, 5]}))
    check_("NaN is fatal", "NaN" in " ".join(check(g, "c")["fatal"]), True)

    # The distinction that cost a run: a warning is an observation about rules
    # this script does not know, and only the evaluator can adjudicate it.
    write(pd.DataFrame({"id": range(5), "v": [-1.0, 0, 1, 2, 3]}))
    r = check(g, "c")
    check_("negative values are a warning, not fatal", r["ok"], True)
    check_("  and are reported", "negative" in " ".join(r["warnings"]), True)
    import shutil
    shutil.move(str(g / "input" / "sample_submission.csv"), str(g / "s.bak"))
    r = check(g, "c")
    check_("a missing sample is a warning too", r["ok"], True)
    check_("  the radar case now passes", r["fatal"], [])
    shutil.move(str(g / "s.bak"), str(g / "input" / "sample_submission.csv"))

    write(pd.DataFrame({"id": [0, 1, 2, 3, 99], "v": [1.0] * 5}))
    check_("unknown id caught", "not in the sample" in
           " ".join(check(g, "c")["problems"]), True)

    check_("a missing file is a failure, not a crash",
           check(g, "nosuch")["ok"], False)


check_ = check


def test_resilience(ws: Path) -> None:
    """Delivery, revival, and the hold override — the three ways a run stalls.

    Each is a gate or a channel that failed silently once: the board reached
    solvers only through MCP calls they rarely made, a dead session removed a
    solver permanently, and a gate that could refuse indefinitely had no point
    past which it could not.
    """
    print("\n[resilience]")
    import inspect
    import native.main as M
    from native import facts

    src = inspect.getsource(M.run_solver)
    check("board is delivered at every round, not only via tools",
          "render_unseen(solver)" in src, True)
    check("a dead session is retried, not fatal", "for life in range" in src, True)
    check("interrupt is retried before giving up",
          "tries >= 3" in inspect.getsource(M.watchdog_turn), True)
    check("an unreachable session ends the life", "HUNG" in src, True)
    check("a hold cannot last forever",
          "hold_overridden" in inspect.getsource(M.clear_for_submission), True)

    g = ws / "revive"
    (g / "solver_a" / "out").mkdir(parents=True, exist_ok=True)
    (g / "solver_a" / "out" / "oof.npy").write_bytes(b"x" * 1234)
    b = facts.init(g, day_file=str(g / "d.jsonl"))
    b.post("claim", "attacking the metric asymmetry", "solver_a")
    b.post("failure", "90 epochs overfits", "solver_a")
    b.post("data", "someone else's finding", "solver_b")
    arts, mine = M.revival_context(g, "solver_a")
    check("a revived session is told what it already built",
          "oof.npy" in arts and "1,234" in arts, True)
    check("  and what it had said", "attacking the metric" in mine, True)
    check("  but not other solvers' posts as its own",
          "someone else's" in mine, False)
    note = M.REVIVAL.format(cwd=g, artifacts=arts, mine=mine, left=42)
    check("  the revival note has no unfilled slots", "{" in note, False)


def test_integrity(ws: Path) -> None:
    """Provenance and order-dependence, both taken from a teammate's notes.

    Their strongest public score on the audio task came from Viterbi-decoding
    the submission ordering — 0.86 to 0.988 with no better model — and they
    caught it with an ordering-invariance test and recorded it as audit-only
    rather than banking it. Separately, nothing here has ever tied the scored
    `oof.npy` to the submitted `submission.csv`.
    """
    print("\n[integrity]")
    import numpy as np
    import pandas as pd
    from native.scripts.integrity import (check_order_dependence,
                                          check_provenance, record)

    g = ws / "integ"
    (g / "solver_a" / "out").mkdir(parents=True, exist_ok=True)
    out = g / "solver_a" / "out"
    np.save(out / "oof.npy", np.arange(50))
    pd.DataFrame({"id": range(50),
                  "y": list(np.random.RandomState(0).randint(0, 4, 50))
                  }).to_csv(out / "submission.csv", index=False)

    record(g, "solver_a", 0.9)
    check("a sealed candidate passes", check_provenance(g, "solver_a"), [])

    np.save(out / "oof.npy", np.arange(50) * 2)
    bad = check_provenance(g, "solver_a")
    check("swapping the scored file is caught", len(bad), 1)
    check("  and says which one", "oof" in bad[0], True)

    record(g, "solver_a", 0.9)
    check("re-sealing clears it", check_provenance(g, "solver_a"), [])

    # The exploit signature: contiguous label blocks over the submission order.
    pd.DataFrame({"id": range(60), "y": [0] * 20 + [1] * 20 + [2] * 20}).to_csv(
        out / "submission.csv", index=False)
    notes = check_order_dependence(g, "solver_a")
    check("contiguous label blocks are flagged",
          any("contiguous" in n for n in notes), True)

    pd.DataFrame({"id": range(60),
                  "y": list(np.random.RandomState(1).randint(0, 3, 60))
                  }).to_csv(out / "submission.csv", index=False)
    check("shuffled predictions are not flagged",
          check_order_dependence(g, "solver_a"), [])

    # A warning, never a verdict — ordered data is legitimate.
    import inspect
    import native.main as M
    src = inspect.getsource(M.integrity_check)
    check("provenance is fatal", "check_provenance" in src, True)
    check("order dependence is only a warning",
          '"warnings": check_order_dependence' in src, True)

    from native.prompts import CONTRACT  # noqa: F401
    sk = (ROOT / "skills" / "validation-split" / "SKILL.md").read_text()
    check("the playbook warns against tuning on the scored folds",
          "chosen on the test of record" in sk, True)
    check("run log template exists", "does and does not show" in M.RUNLOG, True)


def test_submission_mode() -> None:
    """Which route a submission takes, and what happens when we cannot tell.

    `competitions_list` does not return private competitions, and every task on
    the day that counts is private — so the real tasks will all report
    "unknown". Defaulting that to the CSV shortcut means not submitting at all
    on a kernel-only competition, which is what IOAI proper is.
    """
    print("\n[submission mode]")
    import inspect
    import native.main as M
    from native.prompts import CONTRACT

    src = inspect.getsource(M.send_candidate)
    check("CSV taken only when positively established",
          'SUBMIT_MODE["mode"] == "csv"' in src, True)
    check("  so unknown falls through to the kernel path",
          'SUBMIT_MODE["mode"] != "kernel"' in src, False)

    # A kernel may run half an hour; what has to clear the deadline is the
    # submit call, not the scoring.
    check("waits long enough for a real kernel", M.KERNEL_MAX_S >= 30 * 60, True)
    check("reserves time to actually submit", M.SUBMIT_RESERVE_S > 0, True)
    poll = inspect.signature(M.send_candidate).parameters["poll_s"].default
    for left_min, want in ((120, True), (30, True), (3, False)):
        left = left_min * 60 - M.SUBMIT_RESERVE_S
        check(f"  {left_min} min left -> {'push' if want else 'refuse'}",
              left > 0, want)

    # Neither of these is guessable from the CLI docs, and both fail quietly.
    check("contract names the env block", "setup_ioai_env" in CONTRACT, True)
    check("contract names the wheel dataset mount",
          "dataset_sources" in CONTRACT, True)
    check("contract says the submit is what must beat the deadline",
          "submit call, not the scoring" in CONTRACT, True)

    from agent.tools import registry as Rg
    c = Rg.Ctx(workspace=ROOT, slug="x", budget=None, trace=None)
    listing = Rg.skill_list({}, c)
    check("the organisers' own playbook is available",
          "kaggle-cli-official" in listing, True)
    ours = Rg.skill_load({"name": "kaggle-submission"}, c)
    check("  and ours covers what it omits", "setup_ioai_env" in ours, True)
    check("  including not pushing twice by reflex", "Push once" in ours, True)


def test_calibration() -> None:
    """The loop back from the leaderboard, which was never closed.

    `promote --confirm` has existed since the start with nothing calling it, so
    LKG's confirmed tier stayed empty through a run that submitted four times
    and took first place. Worse, the number it exists to produce — how far local
    sits from the leaderboard — is the only defence against the failure already
    on record, 0.9156 out-of-fold against 0.78095 on the board.
    """
    print("\n[leaderboard calibration]")
    import inspect
    import native.main as M

    src = inspect.getsource(M.score_watcher)
    check("something now calls promote --confirm", "--confirm" in src, True)
    check("the gap reaches the board", 'facts.board().post(' in src, True)
    check("an optimistic local score is called out",
          "LOCAL IS OPTIMISTIC" in src, True)
    check("sent submissions are remembered",
          "PENDING.append" in inspect.getsource(M.submitter), True)
    check("the watcher is started with the run",
          "score_watcher(ws" in inspect.getsource(M.run), True)
    # snake_case, not the CLI's column header — this silently returned None.
    fs = inspect.getsource(M.fetch_scores)
    check("reads public_score, not publicScore", '"public_score"' in fs, True)


def test_supervision() -> None:
    """Detection, which is the part the harness can actually guarantee.

    Every behavioural fence in runs 1 and 2 was walked around inside one run,
    and none of the breaches surfaced until the logs were read the next day.
    """
    print("\n[supervision]")
    import time
    from native import supervise as S

    S.URGENT.clear()
    S.flag_urgent("format", "negative density voids the whole submission", "solver_a")
    check("finder is not told its own news", S.drain_urgent("solver_a"), "")
    check("others are pushed it", "URGENT" in S.drain_urgent("solver_b"), True)
    check("and only once", S.drain_urgent("solver_b"), "")

    # Cost tracks context, and context only grows: run 1 went 0.85 -> 1.74 ->
    # 2.65 -> 3.51 -> 4.07 through a $15 cap while each round looked affordable.
    class U:
        percentage = 30.0
    est = S.predict_next_cost(U(), [0.85, 1.74])
    check("next round predicted above the last", est > 1.74, True)
    check("  and above what round 3 really cost ($2.65)", est > 2.65, True)
    check("no history -> no estimate", S.predict_next_cost(U(), []), 0.0)

    S.ACTIVITY["solver_c"] = time.time() - 300
    check("silence is measured", 290 < S.stalled_for("solver_c") < 310, True)
    check("an agent never seen is not called stalled", S.stalled_for("nobody"), 0.0)

    # A directory nothing can be running out of. Pointing this at /tmp made the
    # orphan check count whatever the machine happened to have there, so the
    # suite passed on one box and failed on another.
    import tempfile as _tf
    _iso = Path(_tf.mkdtemp(prefix="recon-iso-"))
    rec = S.Reconciler(ws=_iso, slug="x", allowed_gpus={"4", "5"})
    d = rec.submissions(counted=0)
    check("bypass check tolerates an unreachable API", d, None)
    check("cost overrun caught", rec.cost(19.05, 15.0).kind, "cost-overrun")
    check("cost within cap is quiet", rec.cost(14.0, 15.0), None)
    first = rec.sweep(counted=0, spent=19.05, cap=15.0)
    check("drift reported once", [x.kind for x in first], ["cost-overrun"])
    check("  not repeated every sweep", rec.sweep(0, 19.05, 15.0), [])


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        test_facts(ws)
        test_pipeline(ws)
        test_folds_schemes(ws)
        test_recon(ws)
        test_resilience(ws)
        test_integrity(ws)
        test_format_gate(ws)
    test_configs()
    test_submit_gate()
    test_solver_cannot_submit()
    test_submission_mode()
    test_calibration()
    test_supervision()
    print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
