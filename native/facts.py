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

import hashlib
import json
import os
import re
import time
from pathlib import Path

from . import evidence

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
KINDS = (
    "env", "data", "format", "failure", "claim", "result",
    "conflict", "adoption", "decision",
)
VERIFIED_ONLY = ("result", "decision")
TRUSTED = ("harness", "evaluator", "recon")
# Of the trusted sources, these two are scripts rather than agents: nothing on
# the other end of a rejection can read it and try again. See `post`.
MACHINE = ("harness", "recon")
MAX_TEXT = 300
LAYERS = ("task", "day")
ZERO_SHA256 = "0" * 64


def audit_jsonl(path: Path, expected_layer: str | None = None) -> dict:
    """Verify a board file without silently skipping malformed records."""
    records: list[dict] = []
    errors: list[str] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return {"valid": False, "records": [], "errors": [f"missing {path}"]}
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {number}: invalid JSON: {exc}")
            continue
        if not isinstance(value, dict):
            errors.append(f"line {number}: record is not an object")
            continue
        records.append(value)

    previous = ZERO_SHA256
    ids: set[str] = set()
    claims: dict[str, str] = {}
    for expected_seq, rec in enumerate(records, 1):
        digest = rec.get("record_sha256")
        body = {key: value for key, value in rec.items()
                if key not in ("id", "record_sha256")}
        observed = hashlib.sha256(evidence.canonical(body)).hexdigest()
        if rec.get("seq") != expected_seq:
            errors.append(f"record {expected_seq}: seq is {rec.get('seq')}")
        if expected_layer is not None and rec.get("layer") != expected_layer:
            errors.append(
                f"record {expected_seq}: layer {rec.get('layer')!r} != {expected_layer!r}"
            )
        if rec.get("prev_sha256") != previous:
            errors.append(f"record {expected_seq}: broken previous hash")
        if digest != observed:
            errors.append(f"record {expected_seq}: record hash mismatch")
        if rec.get("id") != f"fact-{observed[:16]}":
            errors.append(f"record {expected_seq}: fact id mismatch")
        fact_id = str(rec.get("id"))
        if fact_id in ids:
            errors.append(f"duplicate fact id {fact_id}")
        ids.add(fact_id)
        previous = str(digest)
        if rec.get("kind") == "claim":
            key = str(rec.get("claim_key", ""))
            owner = claims.setdefault(key, str(rec.get("src", "")))
            if owner != rec.get("src"):
                errors.append(f"claim {key!r} has multiple owners")
    return {
        "valid": not errors,
        "records": records,
        "record_count": len(records),
        "claim_owners": claims,
        "errors": errors,
    }


class Board:
    def __init__(self, task_path: Path, day_path: Path) -> None:
        self.paths = {"task": Path(task_path), "day": Path(day_path)}
        for p in self.paths.values():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch(exist_ok=True)
        self._cursors: dict[str, int] = {}  # reader -> facts already seen

    # ---------------------------------------------------------------- write
    def post(
        self,
        kind: str,
        text: str,
        src: str,
        layer: str = "task",
        *,
        claim_key: str = "",
        evidence_refs: list[str] | tuple[str, ...] | None = None,
        related_ids: list[str] | tuple[str, ...] | None = None,
        tainted: bool = False,
    ) -> str:
        if kind not in KINDS:
            return f"[rejected] kind must be one of {KINDS}, got {kind!r}"
        if kind in VERIFIED_ONLY and src not in TRUSTED:
            return (f"[rejected] '{kind}' may only be posted by {TRUSTED}. "
                    f"Scores reach the board through the evaluator, which "
                    f"computes them on the shared folds — a self-reported "
                    f"number is not comparable to anyone else's.")
        if layer not in LAYERS:
            return f"[rejected] layer must be one of {LAYERS}, got {layer!r}"
        if not str(src).strip():
            return "[rejected] empty source"
        text = " ".join(str(text).split())
        if not text:
            return "[rejected] empty text"
        if len(text) > MAX_TEXT:
            # The cap exists to stop agents narrating, and an agent that trips
            # it can read the rejection and repost shorter. A script cannot, so
            # for those the cap bends rather than drops: the harness announced
            # "this competition is kernel-only, build out/kernel/" in 356
            # characters, got back a rejection string it never inspected, and
            # three solvers spent a whole run producing candidates with no route
            # to the leaderboard. Losing a fact's tail is survivable; losing the
            # fact is not.
            if src in MACHINE:
                print(f"!! {src} posted {len(text)} chars, over the board's "
                      f"{MAX_TEXT} — truncated. Say it shorter: {text[:80]}...",
                      flush=True)
                text = text[:MAX_TEXT - 1]
            else:
                return (f"[rejected] {len(text)} chars > {MAX_TEXT}. Post the "
                        "fact, not the narrative. Split it or cut it down.")
        refs = self._safe_refs(evidence_refs or ())
        related = list(dict.fromkeys(
            str(item).strip() for item in (related_ids or ())
            if str(item).strip()
        ))
        if kind in VERIFIED_ONLY and not refs:
            return (f"[rejected] '{kind}' requires evidence_refs; an official "
                    "result or decision must point to a harness-owned artifact")
        if kind in ("conflict", "adoption") and not related:
            return f"[rejected] '{kind}' requires related_ids"
        if kind == "claim":
            claim_key = self._claim_key(claim_key or text)
            if not claim_key:
                return "[rejected] claim_key is empty after normalization"

        path = self.paths[layer]
        with evidence.locked(path):
            all_records = self._all()
            indexed = {str(item.get("id")): item for item in all_records
                       if item.get("id")}
            if kind in ("conflict", "adoption"):
                missing = [item for item in related if item not in indexed]
                if missing:
                    return f"[rejected] related_ids do not exist: {missing}"
            if kind == "adoption":
                self_refs = [item for item in related
                             if indexed[item].get("src") == src]
                if self_refs:
                    return ("[rejected] adoption must cross routes; cannot "
                            f"adopt your own facts: {self_refs}")
            if kind == "claim":
                owner = next((r.get("src") for r in all_records
                              if r.get("kind") == "claim"
                              and r.get("claim_key") == claim_key), None)
                if owner is not None and owner != src:
                    return (f"[rejected] claim_key {claim_key!r} is already "
                            f"owned by {owner}")
            layer_records = self._read_path(path)
            previous = (layer_records[-1].get("record_sha256", ZERO_SHA256)
                        if layer_records else ZERO_SHA256)
            rec = {
                "seq": len(layer_records) + 1,
                "ts": int(time.time()),
                "layer": layer,
                "kind": kind,
                "src": str(src),
                "text": text,
                "claim_key": claim_key if kind == "claim" else "",
                "evidence_refs": refs,
                "related_ids": related,
                "tainted": bool(tainted),
                "prev_sha256": previous,
            }
            digest = hashlib.sha256(evidence.canonical(rec)).hexdigest()
            rec["id"] = f"fact-{digest[:16]}"
            rec["record_sha256"] = digest
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        return f"[posted] {rec['id']} {kind}: {text[:80]}"

    @staticmethod
    def _claim_key(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")[:80]

    @staticmethod
    def _safe_refs(values) -> list[str]:
        out: list[str] = []
        for raw in values:
            value = str(raw).strip()
            p = Path(value)
            if (not value or p.is_absolute() or ".." in p.parts
                    or len(value) > 240):
                raise ValueError(f"unsafe evidence ref: {value!r}")
            if value not in out:
                out.append(value)
        if len(out) > 12:
            raise ValueError("at most 12 evidence refs are allowed")
        return out

    # ----------------------------------------------------------------- read
    @staticmethod
    def _read_path(path: Path) -> list[dict]:
        out: list[dict] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        return out

    def _all(self) -> list[dict]:
        out: list[dict] = []
        for layer, p in self.paths.items():
            out.extend(self._read_path(p))
        out.sort(key=lambda r: (r.get("ts", 0), r.get("layer", ""),
                                r.get("seq", 0)))
        return out

    def audit(self) -> dict:
        errors: list[str] = []
        ids: set[str] = set()
        claims: dict[str, str] = {}
        for layer, path in self.paths.items():
            audited = audit_jsonl(path, layer)
            errors.extend(f"{layer}: {item}" for item in audited["errors"])
            for rec in audited["records"]:
                if rec.get("id") in ids:
                    errors.append(f"duplicate fact id {rec.get('id')}")
                ids.add(str(rec.get("id")))
                if rec.get("kind") == "claim":
                    key = str(rec.get("claim_key", ""))
                    owner = claims.setdefault(key, str(rec.get("src", "")))
                    if owner != rec.get("src"):
                        errors.append(f"claim {key!r} has multiple owners")
        return {"valid": not errors, "record_count": len(ids),
                "claim_owners": claims, "errors": errors}

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
        lines = [f"  {r['id']} {r['kind']:<8}({r['src']}) {r['text']}"
                 + (" [TAINTED]" if r.get("tainted") else "") for r in new]
        return ("\n\n--- facts board: %d new ---\n" % len(new)) + "\n".join(lines)

    def render_all(self) -> str:
        allf = self._all()
        if not allf:
            return "(facts board empty)"
        return "\n".join(
            f"{r.get('id', '?')} {r['layer']:<5} {r['kind']:<8}"
            f"({r['src']}) {r['text']}"
            + (" [TAINTED]" if r.get("tainted") else "")
            for r in allf
        )


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
