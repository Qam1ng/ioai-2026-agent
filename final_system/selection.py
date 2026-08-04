from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from search_system.ioai_agent_system.redaction import redact_value

from .io import atomic_json, canonical, read_json


def _extract_object(text: str) -> dict:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "decision" in value:
            return value
    raise ValueError("Selection Manager did not return a JSON decision object")


def verify_context(context: dict) -> str:
    value = dict(context)
    expected = str(value.pop("context_sha256", ""))
    actual = hashlib.sha256(canonical(value)).hexdigest()
    if not expected or expected != actual:
        raise ValueError("selection context digest mismatch")
    return expected


class SelectionManager:
    """Claude advisory layer. It can recommend, but it cannot submit."""

    def __init__(
        self, root: Path, *, runner, model: str, effort: str,
        timeout_seconds: float, recommendation_ttl_seconds: float,
    ):
        self.root = Path(root)
        self.runner = runner
        self.model = model
        self.effort = effort
        self.timeout_seconds = timeout_seconds
        self.recommendation_ttl_seconds = recommendation_ttl_seconds
        self.latest_path = self.root / "recommendation.json"
        self.status_path = self.root / "status.json"
        self.inputs = self.root / "inputs"
        self.history = self.root / "history"
        self.traces = self.root.parent / "trajectories" / "selection_manager"
        for path in (self.inputs, self.history, self.traces):
            path.mkdir(parents=True, exist_ok=True)

    def _prompt(self, context: dict) -> str:
        return f"""# IOAI AI Models Track

IOAI AI Models Track 是限时机器学习竞赛。三个相互独立的解题路线会持续产生候选，
但全队只有一个确定性 Submission Broker 可以消耗 Kaggle 提交额度。

# 你的角色：Selection Manager

你使用 Claude Code + {self.model}，effort={self.effort}。你只有建议权，没有 Kaggle
凭证和提交工具。请从本次输入列出的 policy-eligible candidates 中选择一个，或明确
建议等待更多证据。不得推荐列表外候选，不得修改额度策略。

判断时同时考虑：

- 冻结 E0 的 mean/std/per-fold/pooled，而不相信 Agent 自报分数；
- Public LB 反馈及 Local-Public residual；
- 校准器只有 active=true 时才可作为有效调整证据；
- 候选相对 parent 的唯一改动、运行风险和方法多样性；
- Public LB 可能过拟合，final 阶段不能把所有机会押在一种方法上。

只输出一个 JSON 对象，不要 Markdown：

{{
  "decision": "submit" 或 "wait",
  "candidate_id": "submit 时必须填写；wait 时为 null",
  "reason": "不超过 500 字",
  "evidence": ["最多 6 条具体证据"],
  "confidence": 0 到 1,
  "risk_flags": ["风险标签"]
}}

# 本次不可变输入

{json.dumps(context, ensure_ascii=False, sort_keys=True)}
"""

    async def recommend(self, context: dict) -> dict | None:
        digest = verify_context(context)
        status = read_json(self.status_path, {"sequence": 0})
        sequence = int(status.get("sequence", 0)) + 1
        input_path = self.inputs / f"{sequence:04d}-{digest[:12]}.json"
        atomic_json(input_path, context)
        atomic_json(self.status_path, {
            "sequence": sequence, "status": "running",
            "context_sha256": digest, "started_at": time.time(),
            "model": self.model, "effort": self.effort,
        })
        try:
            result = await self.runner.run(
                agent_id=f"selection-manager-{sequence:04d}",
                role="selection_manager", prompt=self._prompt(context),
                workdir=self.root, trace_dir=self.traces,
                timeout_s=self.timeout_seconds, persist_session=False,
            )
        except asyncio.CancelledError:
            atomic_json(self.status_path, {
                "sequence": sequence, "status": "cancelled",
                "context_sha256": digest, "finished_at": time.time(),
                "model": self.model, "effort": self.effort,
            })
            raise
        except Exception as exc:  # noqa: BLE001
            atomic_json(self.status_path, {
                "sequence": sequence, "status": "failed",
                "context_sha256": digest, "finished_at": time.time(),
                "model": self.model, "effort": self.effort,
                "errors": redact_value([f"{type(exc).__name__}: {exc}"]),
            })
            return None
        errors = list(result.errors)
        if result.exit_code != 0:
            errors.append(f"Claude Code exit={result.exit_code}")
        if result.timed_out:
            errors.append("Selection Manager timed out")
        if result.resolved_model != self.model:
            errors.append(
                f"resolved model mismatch: {result.resolved_model!r} != {self.model!r}"
            )
        try:
            raw = _extract_object(result.result_text)
            decision = str(raw.get("decision", ""))
            if decision not in {"submit", "wait"}:
                raise ValueError("decision must be submit or wait")
            candidate_id = raw.get("candidate_id")
            allowed = {
                item["candidate_id"] for item in context.get("candidates", [])
            }
            if decision == "submit" and candidate_id not in allowed:
                raise ValueError("recommended candidate is not policy-eligible")
            if decision == "wait":
                candidate_id = None
            confidence = float(raw.get("confidence", 0.0))
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be in 0..1")
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid recommendation: {exc}")
            raw = {}

        if errors:
            atomic_json(self.status_path, {
                "sequence": sequence, "status": "failed",
                "context_sha256": digest, "finished_at": time.time(),
                "model": self.model, "effort": self.effort,
                "errors": redact_value(errors),
            })
            return None

        now = time.time()
        recommendation = redact_value({
            "schema_version": 1,
            "sequence": sequence,
            "decision": decision,
            "candidate_id": candidate_id,
            "reason": " ".join(str(raw.get("reason", "")).split())[:500],
            "evidence": [
                " ".join(str(item).split())[:300]
                for item in list(raw.get("evidence") or [])[:6]
            ],
            "confidence": confidence,
            "risk_flags": [
                " ".join(str(item).split())[:100]
                for item in list(raw.get("risk_flags") or [])[:8]
            ],
            "context_sha256": digest,
            "model": self.model,
            "effort": self.effort,
            "created_at": now,
            "expires_at": now + self.recommendation_ttl_seconds,
            "submission_authority": "none_advisory_only",
        })
        if not isinstance(recommendation, dict):  # pragma: no cover
            raise TypeError("redacted recommendation must remain an object")
        atomic_json(self.history / f"{sequence:04d}-{digest[:12]}.json", recommendation)
        atomic_json(self.latest_path, recommendation)
        atomic_json(self.status_path, {
            "sequence": sequence, "status": "complete",
            "context_sha256": digest, "finished_at": time.time(),
            "model": self.model, "effort": self.effort,
            "decision": decision, "candidate_id": candidate_id,
        })
        return recommendation
