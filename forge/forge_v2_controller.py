#!/usr/bin/env python3
"""Auditable sidecar search controller for IOAI-FORGE v2 and v3.

The controller ranks and records experiments, but never executes code, writes the
experiment ledger, authorizes a finalist, or submits anything. Completed feedback
must bind to an immutable row written through ``ledger_append.py``.

Version 3 adds interchangeable search-control profiles while preserving every v2
hard gate:

* ``aide_tree`` performs best-first expansion over a validated single-parent tree;
* ``mcgs_graph`` supports multi-parent nodes and bounded ancestor value backup;
* ``pes`` enforces a machine-readable plan -> execute -> summary cycle.

The profile changes experiment scheduling only. It cannot weaken compliance,
leakage, ledger, finalist, or submission custody.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


HEX64 = "0123456789abcdef"
OUTCOMES = {"complete", "failed", "no_op"}
FIDELITIES = {"smoke", "proxy", "full"}
LEGACY_PROFILE = "legacy_ucb"
V3_STRATEGIES = {"aide_tree", "mcgs_graph", "pes"}
PES_DECISIONS = {"continue", "replan", "stop"}


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def write_json_atomic(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        return
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_text(value: datetime | None = None) -> str:
    return (value or utc_now()).isoformat().replace("+00:00", "Z")


def parse_utc_text(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def is_hex64(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in HEX64 for character in value)
    )


def finite(value: Any, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected finite number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"expected finite number, got {value!r}")
    if minimum is not None and result < minimum:
        raise ValueError(f"expected number >= {minimum}, got {result}")
    return result


def probability(value: Any) -> float:
    result = finite(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"expected probability in [0,1], got {result}")
    return result


def positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def policy_schema(policy: dict[str, Any]) -> int:
    schema = policy.get("schema_version")
    if schema not in {1, 2}:
        raise ValueError("unsupported FORGE policy schema")
    return int(schema)


def validate_search_profiles(policy: dict[str, Any]) -> None:
    profiles = policy.get("search_profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("FORGE v3 policy requires search_profiles")
    default = policy.get("default_search_profile")
    if not isinstance(default, str) or default not in profiles:
        raise ValueError("FORGE v3 default_search_profile is unknown")
    observed_strategies: set[str] = set()
    for name, profile in profiles.items():
        if not isinstance(name, str) or not name:
            raise ValueError("search profile names must be non-empty strings")
        if not isinstance(profile, dict):
            raise ValueError(f"search profile {name!r} must be an object")
        strategy = profile.get("strategy")
        if strategy not in V3_STRATEGIES:
            raise ValueError(f"search profile {name!r} has unknown strategy")
        observed_strategies.add(str(strategy))
        if strategy == "aide_tree":
            positive_integer(
                profile.get("maximum_depth"), field=f"{name}.maximum_depth"
            )
            positive_integer(
                profile.get("maximum_children_per_node"),
                field=f"{name}.maximum_children_per_node",
            )
            for field in (
                "parent_value_weight",
                "depth_penalty",
                "branch_exploration_weight",
            ):
                finite(profile.get(field), minimum=0.0)
            if profile.get("require_evaluated_parent") is not True:
                raise ValueError(f"{name}.require_evaluated_parent must be true")
        elif strategy == "mcgs_graph":
            positive_integer(
                profile.get("maximum_parents"), field=f"{name}.maximum_parents"
            )
            positive_integer(
                profile.get("maximum_backpropagation_depth"),
                field=f"{name}.maximum_backpropagation_depth",
            )
            for field in (
                "parent_value_weight",
                "graph_exploration_weight",
                "multi_parent_bonus",
            ):
                finite(profile.get(field), minimum=0.0)
            if profile.get("candidate_signature_required") is not True:
                raise ValueError(
                    f"{name}.candidate_signature_required must be true"
                )
            if profile.get("require_evaluated_parents") is not True:
                raise ValueError(
                    f"{name}.require_evaluated_parents must be true"
                )
        else:
            if profile.get("single_parent_only") is not True:
                raise ValueError(f"{name}.single_parent_only must be true")
            if profile.get("require_evaluated_parent") is not True:
                raise ValueError(f"{name}.require_evaluated_parent must be true")
            if profile.get("stop_decision_blocks_children") is not True:
                raise ValueError(
                    f"{name}.stop_decision_blocks_children must be true"
                )
            required_plan_fields = profile.get("required_plan_fields")
            required_summary_fields = profile.get("required_summary_fields")
            for field, values in (
                ("required_plan_fields", required_plan_fields),
                ("required_summary_fields", required_summary_fields),
            ):
                if (
                    not isinstance(values, list)
                    or not values
                    or any(not isinstance(item, str) or not item for item in values)
                ):
                    raise ValueError(f"{name}.{field} must be a non-empty list")
            finite(profile.get("replan_priority_bonus"), minimum=0.0)
    if observed_strategies != V3_STRATEGIES:
        raise ValueError(
            "FORGE v3 policy must define aide_tree, mcgs_graph, and pes strategies"
        )


def validate_policy(policy: dict[str, Any]) -> None:
    schema = policy_schema(policy)
    trace_ids = policy.get("required_trace_ids")
    count = policy.get("forced_trace_count")
    if (
        not isinstance(count, int)
        or count != 3
        or not isinstance(trace_ids, list)
        or len(trace_ids) != count
        or len(set(map(str, trace_ids))) != count
    ):
        raise ValueError("FORGE v2 requires exactly three distinct traces")
    names = [
        item.get("name")
        for item in policy.get("fidelities", [])
        if isinstance(item, dict)
    ]
    if names != ["smoke", "proxy", "full"]:
        raise ValueError("fidelity order must be smoke, proxy, full")
    for item in policy.get("fidelities", []):
        finite(item.get("maximum_runtime_seconds"), minimum=0.001)
    if policy.get("hard_gates", {}).get(
        "controller_may_write_experiment_ledger"
    ) is not False:
        raise ValueError("controller must never be an experiment-ledger writer")
    if schema == 2:
        validate_search_profiles(policy)


def resolve_search_profile(
    policy: dict[str, Any], requested: str | None
) -> tuple[str, dict[str, Any]]:
    if policy_schema(policy) == 1:
        if requested not in {None, "", LEGACY_PROFILE}:
            raise ValueError("FORGE v2 policy supports only legacy_ucb")
        return LEGACY_PROFILE, {"strategy": LEGACY_PROFILE}
    name = requested or str(policy["default_search_profile"])
    profiles = policy["search_profiles"]
    if name not in profiles:
        raise ValueError(f"unknown search profile {name!r}")
    return str(name), dict(profiles[name])


def fidelity_runtime_seconds(policy: dict[str, Any], fidelity: str) -> float:
    matches = [
        item
        for item in policy.get("fidelities", [])
        if isinstance(item, dict) and item.get("name") == fidelity
    ]
    if len(matches) != 1:
        raise ValueError(f"policy lacks exactly one fidelity {fidelity!r}")
    return finite(matches[0].get("maximum_runtime_seconds"), minimum=0.001)


def task_profile(
    task_config: dict[str, Any], *, allow_clean_benchmark: bool = False
) -> dict[str, Any]:
    execution = task_config.get("execution_policy", {})
    clean_allowed = execution.get("clean_agent_benchmark_allowed")
    if not allow_clean_benchmark and execution.get("submission_allowed") is not False:
        raise ValueError("FORGE v2 offline controller requires submission_allowed=false")
    if not allow_clean_benchmark and clean_allowed is not False:
        raise ValueError(
            "contaminated Task One must keep clean_agent_benchmark_allowed=false"
        )
    if allow_clean_benchmark and not isinstance(clean_allowed, bool):
        raise ValueError("clean_agent_benchmark_allowed must be boolean")
    profile = task_config.get("forge_v3_profile")
    if profile is None:
        profile = task_config.get("forge_v2_profile")
    if not isinstance(profile, dict):
        raise ValueError("task config lacks forge_v2_profile or forge_v3_profile")
    traces = profile.get("traces")
    if not isinstance(traces, list) or len(traces) != 3:
        raise ValueError("task forge_v2_profile must define exactly three traces")
    trace_ids = [str(item.get("trace_id", "")) for item in traces]
    axes = [str(item.get("root_axis", "")) for item in traces]
    families = [str(item.get("mechanism_family", "")) for item in traces]
    if (
        any(not value for value in trace_ids + axes + families)
        or len(set(trace_ids)) != 3
        or len(set(axes)) != 3
        or len(set(families)) != 3
    ):
        raise ValueError("task traces, root axes, and mechanism families must be distinct")
    benchmark_claim_allowed = profile.get("benchmark_claim_allowed")
    if not allow_clean_benchmark and benchmark_claim_allowed is not False:
        raise ValueError("Task One profile must forbid clean benchmark claims")
    if allow_clean_benchmark and benchmark_claim_allowed is not clean_allowed:
        raise ValueError(
            "profile benchmark_claim_allowed must match execution policy"
        )
    return profile


def ledger_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_event(state: dict[str, Any], event_type: str, payload: Any) -> None:
    event = {
        "seq": int(state["event_count"]) + 1,
        "timestamp_utc": utc_text(),
        "type": event_type,
        "prev_event_sha256": state["last_event_sha256"],
        "payload": payload,
    }
    event["event_sha256"] = sha256_bytes(canonical_bytes(event))
    event_path = Path(state["events_path"])
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                event,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    state["event_count"] = event["seq"]
    state["last_event_sha256"] = event["event_sha256"]


def validate_bound_files(state: dict[str, Any]) -> None:
    bindings = (
        ("policy_path", "policy_sha256"),
        ("task_config_path", "task_config_sha256"),
    )
    for path_field, hash_field in bindings:
        path = Path(state[path_field])
        if not path.is_file() or sha256_file(path) != state[hash_field]:
            raise ValueError(f"{path_field} differs from the initialized contract")
    if not Path(state["ledger_path"]).is_file():
        raise ValueError("experiment ledger is missing")


class LockedState:
    def __init__(self, path: Path):
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")
        self.lock_handle: Any = None
        self.state: dict[str, Any] | None = None

    def __enter__(self) -> dict[str, Any]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_handle = self.lock_path.open("a+", encoding="utf-8")
        fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_EX)
        self.state = load_json(self.path)
        validate_bound_files(self.state)
        return self.state

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        assert self.lock_handle is not None
        if exc_type is None and self.state is not None:
            write_json_atomic(self.path, self.state)
        fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_UN)
        self.lock_handle.close()


def acquisition_priority(
    policy: dict[str, Any],
    proposal: dict[str, Any],
    trace_pulls: int,
    total_pulls: int,
) -> float:
    context = proposal["context"]
    expected_gain = finite(context["expected_gain"])
    p_gain = probability(context["probability_of_gain"])
    p_valid = probability(context["probability_of_valid_run"])
    p_finish = probability(context["probability_of_finishing"])
    minutes = max(
        finite(context["expected_minutes"], minimum=0.0),
        finite(policy["selection"]["minimum_expected_minutes"], minimum=0.001),
    )
    uncertainty = probability(context["uncertainty"])
    novelty = probability(context["novelty"])
    risk = probability(context["risk"])
    urgency = probability(context["urgency"])
    alpha = finite(policy["selection"]["uncertainty_alpha"], minimum=0.0)
    exploration = (
        alpha
        * uncertainty
        * math.sqrt(math.log1p(total_pulls + 1) / (trace_pulls + 1))
        / math.sqrt(minutes)
    )
    return (
        expected_gain * p_gain * p_valid * p_finish / minutes
        + exploration
        + finite(policy["selection"]["novelty_weight"]) * novelty
        + finite(policy["selection"]["urgency_weight"]) * urgency
        - finite(policy["selection"]["risk_weight"]) * risk
    )


def proposal_parents(proposal: dict[str, Any]) -> list[str]:
    legacy_parent = str(proposal.get("parent", "")).strip()
    raw = proposal.get("parents")
    if raw is None:
        return [legacy_parent] if legacy_parent else []
    if not isinstance(raw, list):
        raise ValueError("proposal parents must be a list")
    parents = [str(item).strip() for item in raw]
    if any(not item for item in parents) or len(set(parents)) != len(parents):
        raise ValueError("proposal parents must be distinct non-empty node IDs")
    if legacy_parent and (not parents or parents[0] != legacy_parent):
        raise ValueError("legacy parent must equal the first parents entry")
    return parents


def require_nonempty_text_list(value: Any, *, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"{field} must be a non-empty text list")
    return [item.strip() for item in value]


def active_search_profile(
    state: dict[str, Any], policy: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    return resolve_search_profile(policy, str(state.get("search_profile", "")))


def parent_nodes(
    state: dict[str, Any], parents: list[str]
) -> list[dict[str, Any]]:
    graph = state.get("search_graph", {})
    nodes = graph.get("nodes", {})
    if not isinstance(nodes, dict):
        raise ValueError("controller search graph is invalid")
    missing = [parent for parent in parents if parent not in nodes]
    if missing:
        raise ValueError(f"proposal references unknown parent {missing[0]!r}")
    return [nodes[parent] for parent in parents]


def validate_pes_plan(
    proposal: dict[str, Any], profile: dict[str, Any]
) -> None:
    plan = proposal.get("plan")
    if not isinstance(plan, dict):
        raise ValueError("PES proposal requires plan object")
    for field in profile["required_plan_fields"]:
        if field not in plan:
            raise ValueError(f"PES plan.{field} is required")
    if not str(plan.get("objective", "")).strip():
        raise ValueError("PES plan.objective must be non-empty")
    require_nonempty_text_list(plan.get("steps"), field="PES plan.steps")
    require_nonempty_text_list(
        plan.get("expected_artifacts"), field="PES plan.expected_artifacts"
    )
    require_nonempty_text_list(
        plan.get("stop_conditions"), field="PES plan.stop_conditions"
    )


def validate_v3_topology(
    state: dict[str, Any],
    policy: dict[str, Any],
    proposal: dict[str, Any],
) -> list[str]:
    profile_name, profile = active_search_profile(state, policy)
    strategy = profile["strategy"]
    if proposal.get("schema_version") != 2:
        raise ValueError("FORGE v3 search profiles require proposal schema_version=2")
    if proposal.get("search_profile") != profile_name:
        raise ValueError("proposal search_profile differs from controller profile")
    parents = proposal_parents(proposal)
    if proposal["node"] in parents:
        raise ValueError("proposal cannot be its own parent")
    resolved_parents = parent_nodes(state, parents)
    require_evaluated = bool(
        profile.get("require_evaluated_parent")
        or profile.get("require_evaluated_parents")
    )
    if require_evaluated and any(
        parent.get("status") != "evaluated" for parent in resolved_parents
    ):
        raise ValueError("all parents must be evaluated before child registration")
    if any(parent.get("topology_eligible") is not True for parent in resolved_parents):
        raise ValueError("all parents must pass hard gates before child registration")

    graph = state["search_graph"]
    if strategy == "aide_tree":
        if len(parents) > 1:
            raise ValueError("AIDE tree proposals may have at most one parent")
        if parents:
            parent = resolved_parents[0]
            if len(parent["children"]) >= int(profile["maximum_children_per_node"]):
                raise ValueError("AIDE tree parent has reached its child capacity")
            depth = int(parent["depth"]) + 1
            if depth > int(profile["maximum_depth"]):
                raise ValueError("AIDE tree proposal exceeds maximum depth")
    elif strategy == "mcgs_graph":
        if len(parents) > int(profile["maximum_parents"]):
            raise ValueError("MCGS proposal exceeds maximum parent count")
        signature = str(proposal.get("candidate_signature", "")).strip()
        if not signature:
            raise ValueError("MCGS proposal requires candidate_signature")
        if signature in graph["signature_to_node"]:
            raise ValueError("MCGS candidate_signature is already registered")
    else:
        if len(parents) > 1:
            raise ValueError("PES proposals may have at most one parent")
        validate_pes_plan(proposal, profile)
        if parents and profile["stop_decision_blocks_children"]:
            parent = resolved_parents[0]
            if parent.get("pes_decision") == "stop":
                raise ValueError("PES parent decision=stop blocks child registration")
    return parents


def register_search_node(
    state: dict[str, Any],
    policy: dict[str, Any],
    proposal: dict[str, Any],
    parents: list[str],
) -> dict[str, Any]:
    profile_name, profile = active_search_profile(state, policy)
    nodes = state["search_graph"]["nodes"]
    depth = (
        0
        if not parents
        else 1 + max(int(nodes[parent]["depth"]) for parent in parents)
    )
    node = {
        "node": proposal["node"],
        "trace_id": proposal["trace_id"],
        "parents": parents,
        "children": [],
        "depth": depth,
        "status": "planned",
        "visits": 0,
        "value_sum": 0.0,
        "mean_value": None,
        "max_value": None,
        "pes_decision": None,
        "topology_eligible": None,
    }
    signature = str(proposal.get("candidate_signature", "")).strip()
    if signature:
        node["candidate_signature"] = signature
        state["search_graph"]["signature_to_node"][signature] = proposal["node"]
    nodes[proposal["node"]] = node
    for parent in parents:
        nodes[parent]["children"].append(proposal["node"])
    if profile["strategy"] == "pes":
        node["plan"] = proposal["plan"]
    state["search_graph"]["strategy"] = profile["strategy"]
    state["search_graph"]["profile"] = profile_name
    return node


def validate_proposal(
    state: dict[str, Any],
    proposal: dict[str, Any],
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schema = proposal.get("schema_version")
    if schema not in {1, 2}:
        raise ValueError("unsupported proposal schema")
    if proposal.get("task") != state["task"]:
        raise ValueError("proposal task differs from controller task")
    node = str(proposal.get("node", ""))
    if not node or node in state["proposals"] or node in state["results"]:
        raise ValueError("proposal node is missing or already exists")
    trace_id = str(proposal.get("trace_id", ""))
    expected = {
        item["trace_id"]: item for item in state["task_profile"]["traces"]
    }
    if trace_id not in expected:
        raise ValueError(f"unknown trace_id {trace_id!r}")
    trace = expected[trace_id]
    if proposal.get("root_axis") != trace["root_axis"]:
        raise ValueError("proposal root_axis differs from frozen trace assignment")
    if proposal.get("mechanism_family") != trace["mechanism_family"]:
        raise ValueError(
            "proposal mechanism_family differs from frozen trace assignment"
        )
    if proposal.get("fidelity") not in FIDELITIES:
        raise ValueError("proposal fidelity must be smoke, proxy, or full")
    if proposal.get("audit_sensitive_visible") is not False:
        raise ValueError("audit-sensitive information cannot be visible to solver")
    hypothesis = proposal.get("hypothesis")
    if not isinstance(hypothesis, dict):
        raise ValueError("proposal hypothesis must be an object")
    for field in (
        "diagnosis",
        "causal_hypothesis",
        "atomic_change",
        "falsifier",
    ):
        if not str(hypothesis.get(field, "")).strip():
            raise ValueError(f"proposal hypothesis.{field} is required")
    observables = hypothesis.get("expected_observables")
    if (
        not isinstance(observables, list)
        or not observables
        or any(not str(item).strip() for item in observables)
    ):
        raise ValueError("proposal requires non-empty expected_observables")
    context = proposal.get("context")
    if not isinstance(context, dict):
        raise ValueError("proposal context must be an object")
    for field in (
        "expected_gain",
        "probability_of_gain",
        "probability_of_valid_run",
        "probability_of_finishing",
        "expected_minutes",
        "uncertainty",
        "novelty",
        "risk",
        "urgency",
    ):
        if field not in context:
            raise ValueError(f"proposal context.{field} is required")
    finite(context["expected_gain"])
    expected_minutes = finite(context["expected_minutes"], minimum=0.0)
    for field in (
        "probability_of_gain",
        "probability_of_valid_run",
        "probability_of_finishing",
        "uncertainty",
        "novelty",
        "risk",
        "urgency",
    ):
        probability(context[field])
    if state.get("search_profile", LEGACY_PROFILE) == LEGACY_PROFILE:
        if schema != 1:
            raise ValueError("legacy_ucb requires proposal schema_version=1")
    else:
        if policy is None:
            raise ValueError("v3 proposal validation requires the frozen policy")
        if expected_minutes * 60.0 > fidelity_runtime_seconds(
            policy, str(proposal["fidelity"])
        ):
            raise ValueError(
                "proposal expected_minutes exceeds its fidelity runtime cap"
            )
        validate_v3_topology(state, policy, proposal)
    return proposal


def search_priority(
    state: dict[str, Any],
    policy: dict[str, Any],
    proposal: dict[str, Any],
    trace_pulls: int,
    total_pulls: int,
) -> float:
    base = acquisition_priority(policy, proposal, trace_pulls, total_pulls)
    profile_name, profile = active_search_profile(state, policy)
    if profile_name == LEGACY_PROFILE:
        return base
    node = state["search_graph"]["nodes"][proposal["node"]]
    parents = [
        state["search_graph"]["nodes"][parent] for parent in node["parents"]
    ]
    strategy = profile["strategy"]
    if strategy == "aide_tree":
        if not parents:
            return base
        parent = parents[0]
        parent_value = (
            float(parent["mean_value"])
            if parent["mean_value"] is not None
            else 0.0
        )
        branch_exploration = float(profile["branch_exploration_weight"]) / math.sqrt(
            max(1, len(parent["children"]))
        )
        return (
            base
            + float(profile["parent_value_weight"]) * parent_value
            + branch_exploration
            - float(profile["depth_penalty"]) * int(node["depth"])
        )
    if strategy == "mcgs_graph":
        if not parents:
            return base + float(profile["graph_exploration_weight"])
        known_values = [
            float(parent["mean_value"])
            for parent in parents
            if parent["mean_value"] is not None
        ]
        parent_value = max(known_values, default=0.0)
        parent_visits = sum(int(parent["visits"]) for parent in parents)
        graph_visits = max(
            1,
            sum(
                int(item["visits"])
                for item in state["search_graph"]["nodes"].values()
            ),
        )
        exploration = float(profile["graph_exploration_weight"]) * math.sqrt(
            math.log1p(graph_visits + 1) / (parent_visits + 1)
        )
        merge_bonus = float(profile["multi_parent_bonus"]) * max(
            0, len(parents) - 1
        )
        return (
            base
            + float(profile["parent_value_weight"]) * parent_value
            + exploration
            + merge_bonus
        )
    if parents and parents[0].get("pes_decision") == "replan":
        return base + float(profile["replan_priority_bonus"])
    return base


def validate_pes_summary(
    feedback: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    summary = feedback.get("execution_summary")
    if not isinstance(summary, dict):
        raise ValueError("PES feedback requires execution_summary object")
    for field in profile["required_summary_fields"]:
        if field not in summary:
            raise ValueError(f"PES execution_summary.{field} is required")
    require_nonempty_text_list(
        summary.get("executed_steps"),
        field="PES execution_summary.executed_steps",
    )
    require_nonempty_text_list(
        summary.get("observations"),
        field="PES execution_summary.observations",
    )
    deviations = summary.get("deviations")
    if not isinstance(deviations, list) or any(
        not isinstance(item, str) or not item.strip() for item in deviations
    ):
        raise ValueError("PES execution_summary.deviations must be a text list")
    decision = summary.get("decision")
    if decision not in PES_DECISIONS:
        raise ValueError(
            "PES execution_summary.decision must be continue, replan, or stop"
        )
    if decision == "replan" and not str(summary.get("next_plan", "")).strip():
        raise ValueError("PES decision=replan requires non-empty next_plan")
    if decision == "stop" and not str(summary.get("stop_reason", "")).strip():
        raise ValueError("PES decision=stop requires non-empty stop_reason")
    return summary


def validate_v3_feedback(
    state: dict[str, Any],
    policy: dict[str, Any],
    feedback: dict[str, Any],
) -> dict[str, Any] | None:
    profile_name, profile = active_search_profile(state, policy)
    if profile_name == LEGACY_PROFILE:
        if feedback.get("schema_version") != 1:
            raise ValueError("legacy_ucb requires feedback schema_version=1")
        return None
    if feedback.get("schema_version") != 2:
        raise ValueError("FORGE v3 search profiles require feedback schema_version=2")
    if feedback.get("search_profile") != profile_name:
        raise ValueError("feedback search_profile differs from controller profile")
    if profile["strategy"] == "pes":
        return validate_pes_summary(feedback, profile)
    return None


def update_node_statistics(node: dict[str, Any], value: float) -> None:
    node["visits"] = int(node["visits"]) + 1
    node["value_sum"] = float(node["value_sum"]) + value
    node["mean_value"] = float(node["value_sum"]) / int(node["visits"])
    node["max_value"] = (
        value
        if node["max_value"] is None
        else max(float(node["max_value"]), value)
    )


def update_search_after_feedback(
    state: dict[str, Any],
    policy: dict[str, Any],
    proposal: dict[str, Any],
    feedback: dict[str, Any],
    robust_utility: float,
    topology_eligible: bool,
    execution_summary: dict[str, Any] | None,
) -> None:
    profile_name, profile = active_search_profile(state, policy)
    if profile_name == LEGACY_PROFILE:
        return
    nodes = state["search_graph"]["nodes"]
    node = nodes[proposal["node"]]
    node["status"] = "evaluated"
    node["robust_utility"] = robust_utility
    node["topology_eligible"] = topology_eligible
    if not topology_eligible:
        if profile["strategy"] == "pes":
            assert execution_summary is not None
            node["execution_summary"] = execution_summary
            node["pes_decision"] = execution_summary["decision"]
            state["pes_cycles"].append(
                {
                    "node": proposal["node"],
                    "trace_id": proposal["trace_id"],
                    "plan": proposal["plan"],
                    "summary": execution_summary,
                    "robust_utility": robust_utility,
                    "topology_eligible": False,
                }
            )
        return
    if profile["strategy"] == "mcgs_graph":
        frontier: list[tuple[str, int]] = [(proposal["node"], 0)]
        visited: set[str] = set()
        maximum_depth = int(profile["maximum_backpropagation_depth"])
        while frontier:
            current_id, distance = frontier.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            current = nodes[current_id]
            update_node_statistics(current, robust_utility)
            if distance < maximum_depth:
                frontier.extend(
                    (parent, distance + 1) for parent in current["parents"]
                )
    else:
        update_node_statistics(node, robust_utility)
    if profile["strategy"] == "pes":
        assert execution_summary is not None
        node["execution_summary"] = execution_summary
        node["pes_decision"] = execution_summary["decision"]
        state["pes_cycles"].append(
            {
                "node": proposal["node"],
                "trace_id": proposal["trace_id"],
                "plan": proposal["plan"],
                "summary": execution_summary,
                "robust_utility": robust_utility,
                "topology_eligible": True,
            }
        )


def find_bound_ledger_row(
    state: dict[str, Any], feedback: dict[str, Any]
) -> dict[str, str]:
    binding = feedback.get("ledger_binding")
    if not isinstance(binding, dict):
        raise ValueError("feedback lacks ledger_binding")
    matches = [
        row
        for row in ledger_rows(Path(state["ledger_path"]))
        if row.get("task") == binding.get("task")
        and row.get("node") == binding.get("node")
    ]
    if len(matches) != 1:
        raise ValueError("feedback must bind exactly one experiment-ledger row")
    row = matches[0]
    for field in (
        "task",
        "node",
        "code_sha256",
        "split_sha256",
        "task_config_sha256",
        "prediction_sha256",
        "output_sha256",
    ):
        if binding.get(field) != row.get(field):
            raise ValueError(f"feedback ledger binding differs on {field}")
    if row.get("task_config_sha256") != state["task_config_sha256"]:
        raise ValueError("ledger row is bound to a different task config")
    return row


def validate_feedback_against_ledger(
    state: dict[str, Any],
    feedback: dict[str, Any],
    row: dict[str, str],
) -> None:
    if int(state.get("schema_version", 1)) < 2:
        return
    binding = feedback["ledger_binding"]
    required_hashes = (
        "code_sha256",
        "split_sha256",
        "task_config_sha256",
        "prediction_sha256",
        "output_sha256",
        "output_receipt_sha256",
        "evaluator_receipt_sha256",
    )
    for field in required_hashes:
        ledger_value = row.get(field)
        if not is_hex64(ledger_value):
            raise ValueError(f"v3 ledger row requires 64-hex {field}")
        if binding.get(field) != ledger_value:
            raise ValueError(f"feedback ledger binding differs on {field}")
    if row.get("run_status") not in {"complete", "scored"}:
        raise ValueError("v3 ledger run_status must be complete or scored")
    metrics = feedback.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("feedback metrics must be an object")
    for ledger_field, feedback_field in (
        ("primary_metric", "primary"),
        ("secondary_metric", "secondary"),
        ("normalized_utility", "robust_utility"),
        ("score_std", "score_std"),
    ):
        ledger_text = str(row.get(ledger_field, "")).strip()
        if not ledger_text:
            raise ValueError(f"v3 ledger row requires {ledger_field}")
        try:
            ledger_value = float(ledger_text)
        except ValueError as exc:
            raise ValueError(f"ledger {ledger_field} must be numeric") from exc
        feedback_value = finite(
            metrics.get(feedback_field),
            minimum=0.0 if feedback_field == "score_std" else None,
        )
        if not math.isclose(
            feedback_value, ledger_value, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"feedback {feedback_field} differs from ledger {ledger_field}"
            )
    for field in ("compliance_status", "output_contract_status"):
        ledger_value = str(row.get(field, "")).strip()
        if not ledger_value:
            raise ValueError(f"v3 ledger row requires {field}")
        if feedback.get(field) != ledger_value:
            raise ValueError(f"feedback {field} differs from experiment ledger")
    if feedback.get("outcome") != row.get("status"):
        raise ValueError("feedback outcome differs from experiment ledger status")
    runtime_text = str(row.get("runtime_s", "")).strip()
    if not runtime_text:
        raise ValueError("v3 ledger row requires runtime_s")
    try:
        runtime = float(runtime_text)
    except ValueError as exc:
        raise ValueError("ledger runtime_s must be numeric") from exc
    finite(runtime, minimum=0.0)
    policy = load_json(Path(state["policy_path"]))
    if runtime > fidelity_runtime_seconds(
        policy, str(feedback.get("fidelity", ""))
    ):
        raise ValueError("ledger runtime_s exceeds the fidelity runtime cap")


def safe_feedback(feedback: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if feedback.get("audit_sensitive_visible") is not False:
        failures.append("audit_sensitive_visible")
    for field in (
        "compliance_status",
        "output_contract_status",
        "leakage_status",
    ):
        if feedback.get(field) != "pass":
            failures.append(field)
    if feedback.get("duplicate_prediction") is True:
        failures.append("duplicate_prediction")
    return not failures, failures


def trim_archive(state: dict[str, Any], policy: dict[str, Any]) -> None:
    archive = state["archive"]
    archive["challengers"] = sorted(
        archive["challengers"],
        key=lambda item: float(item["robust_utility"]),
        reverse=True,
    )[: int(policy["archive"]["challenger_capacity"])]
    archive["stepping_stones"] = sorted(
        archive["stepping_stones"],
        key=lambda item: (
            int(item["ttl_rounds"]),
            float(item["best_group_delta"]),
        ),
        reverse=True,
    )[: int(policy["archive"]["stepping_stone_capacity"])]


def age_stepping_stones(state: dict[str, Any]) -> None:
    retained: list[dict[str, Any]] = []
    for item in state["archive"]["stepping_stones"]:
        item = dict(item)
        item["ttl_rounds"] = int(item["ttl_rounds"]) - 1
        if item["ttl_rounds"] > 0:
            retained.append(item)
        else:
            state["archive"]["graveyard"].append(
                {"node": item["node"], "reason": "stepping_stone_ttl_expired"}
            )
    state["archive"]["stepping_stones"] = retained


def classify_result(
    state: dict[str, Any],
    policy: dict[str, Any],
    proposal: dict[str, Any],
    feedback: dict[str, Any],
) -> dict[str, Any]:
    safe, hard_failures = safe_feedback(feedback)
    metrics = feedback.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("feedback metrics must be an object")
    robust = finite(metrics.get("robust_utility"))
    finite(metrics.get("primary"))
    finite(metrics.get("secondary"))
    finite(metrics.get("score_std"), minimum=0.0)
    deltas = metrics.get("group_deltas")
    if not isinstance(deltas, dict):
        raise ValueError("metrics.group_deltas must be an object")
    group_deltas = {str(key): finite(value) for key, value in deltas.items()}
    best_group_delta = max(group_deltas.values(), default=-math.inf)
    outcome = feedback.get("outcome")
    if outcome not in OUTCOMES:
        raise ValueError(f"unsupported feedback outcome {outcome!r}")
    fidelity = feedback.get("fidelity")
    if fidelity != proposal["fidelity"]:
        raise ValueError("feedback fidelity differs from proposal")
    gradient = feedback.get("textual_gradient")
    if not isinstance(gradient, dict):
        raise ValueError("feedback lacks textual_gradient")
    for field in (
        "observed_failure",
        "causal_update",
        "next_atomic_change",
        "next_expected_observable",
        "next_falsifier",
    ):
        if not str(gradient.get(field, "")).strip():
            raise ValueError(f"textual_gradient.{field} is required")
    refs = gradient.get("evidence_refs")
    if not isinstance(refs, list) or not refs:
        raise ValueError("textual_gradient.evidence_refs must be non-empty")
    probability(gradient.get("confidence"))
    if gradient.get("memory_visibility") != policy["shared_memory"][
        "allowed_visibility"
    ]:
        raise ValueError("audit-only or unknown memory visibility is forbidden")

    age_stepping_stones(state)
    archive = state["archive"]
    classification = "graveyard"
    champion = archive["champion"]
    champion_robust = (
        float(champion["robust_utility"]) if isinstance(champion, dict) else None
    )
    hard_pass = safe and outcome == "complete"
    if hard_pass and fidelity == "full" and (
        champion_robust is None or robust > champion_robust
    ):
        if champion is not None:
            archive["challengers"].append(champion)
        archive["champion"] = {
            "node": proposal["node"],
            "trace_id": proposal["trace_id"],
            "robust_utility": robust,
            "fidelity": fidelity,
        }
        classification = "champion"
    elif hard_pass and fidelity == "full" and champion_robust is not None and (
        robust
        >= champion_robust
        - float(policy["archive"]["challenger_robust_margin"])
    ):
        archive["challengers"].append(
            {
                "node": proposal["node"],
                "trace_id": proposal["trace_id"],
                "robust_utility": robust,
                "fidelity": fidelity,
            }
        )
        classification = "challenger"
    else:
        reference = champion_robust if champion_robust is not None else robust
        qualifies_stepping_stone = (
            hard_pass
            and best_group_delta
            >= float(policy["archive"]["minimum_failure_group_gain"])
            and robust
            >= reference
            - float(
                policy["archive"]["maximum_stepping_stone_robust_regression"]
            )
        )
        if qualifies_stepping_stone:
            archive["stepping_stones"].append(
                {
                    "node": proposal["node"],
                    "trace_id": proposal["trace_id"],
                    "robust_utility": robust,
                    "best_group_delta": best_group_delta,
                    "ttl_rounds": int(
                        policy["archive"]["stepping_stone_ttl_rounds"]
                    ),
                    "fidelity": fidelity,
                }
            )
            classification = "stepping_stone"
        else:
            archive["graveyard"].append(
                {
                    "node": proposal["node"],
                    "reason": (
                        ",".join(hard_failures)
                        if hard_failures
                        else f"{outcome}_or_not_competitive"
                    ),
                }
            )
    trim_archive(state, policy)

    memory_entry = {
        "node": proposal["node"],
        "trace_id": proposal["trace_id"],
        "classification": classification,
        "robust_utility": robust,
        "observed_failure": gradient["observed_failure"],
        "causal_update": gradient["causal_update"],
        "next_atomic_change": gradient["next_atomic_change"],
        "confidence": gradient["confidence"],
        "visibility": gradient["memory_visibility"],
    }
    if safe:
        state["shared_memory"].append(memory_entry)
        state["shared_memory"] = state["shared_memory"][
            -int(policy["shared_memory"]["maximum_entries"]) :
        ]
    return {
        "classification": classification,
        "hard_gate_pass": hard_pass,
        "topology_eligible": hard_pass,
        "solver_memory_eligible": safe,
        "hard_gate_failures": hard_failures,
        "robust_utility": robust,
        "best_group_delta": best_group_delta,
        "machine_gate_required_for_finalist": fidelity == "full",
    }


def command_init(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.policy, args.task_config, args.ledger):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.state.exists() or args.events.exists():
        raise FileExistsError("controller state/events already exist")
    policy = load_json(args.policy)
    validate_policy(policy)
    task_config = load_json(args.task_config)
    is_v3 = policy_schema(policy) == 2
    profile = task_profile(task_config, allow_clean_benchmark=is_v3)
    requested_profile = getattr(args, "search_profile", None) or profile.get(
        "search_profile"
    )
    search_profile, search_profile_config = resolve_search_profile(
        policy, requested_profile
    )
    if [item["trace_id"] for item in profile["traces"]] != policy[
        "required_trace_ids"
    ]:
        raise ValueError("task trace IDs differ from FORGE policy")
    if task_config.get("competition") != args.competition:
        raise ValueError("competition differs from task config")
    if args.budget_seconds <= 0:
        raise ValueError("budget_seconds must be positive")
    args.events.parent.mkdir(parents=True, exist_ok=True)
    args.events.touch(exist_ok=False)
    start = utc_now()
    state = {
        "schema_version": policy_schema(policy),
        "controller": (
            "ioai-forge-v2"
            if search_profile == LEGACY_PROFILE
            else "ioai-forge-v3"
        ),
        "search_profile": search_profile,
        "search_strategy": search_profile_config["strategy"],
        "task": args.task,
        "competition": args.competition,
        "started_at_utc": utc_text(start),
        "deadline_utc": utc_text(start + timedelta(seconds=args.budget_seconds)),
        "budget_seconds": args.budget_seconds,
        "policy_path": str(args.policy.resolve()),
        "policy_sha256": sha256_file(args.policy),
        "task_config_path": str(args.task_config.resolve()),
        "task_config_sha256": sha256_file(args.task_config),
        "ledger_path": str(args.ledger.resolve()),
        "ledger_sha256_at_init": sha256_file(args.ledger),
        "events_path": str(args.events.resolve()),
        "task_profile": profile,
        "trace_stats": {
            trace_id: {"pulls": 0, "completed": 0}
            for trace_id in policy["required_trace_ids"]
        },
        "proposals": {},
        "results": {},
        "archive": {
            "champion": None,
            "challengers": [],
            "stepping_stones": [],
            "graveyard": [],
        },
        "shared_memory": [],
        "search_graph": {
            "profile": search_profile,
            "strategy": search_profile_config["strategy"],
            "nodes": {},
            "signature_to_node": {},
        },
        "pes_cycles": [],
        "event_count": 0,
        "last_event_sha256": "0" * 64,
        "finalist_authority": "existing_machine_gate_only",
        "submission_allowed": False,
        "clean_benchmark_claim_allowed": bool(
            profile.get("benchmark_claim_allowed", False)
        ),
    }
    append_event(
        state,
        "controller_initialized",
        {
            "policy_sha256": state["policy_sha256"],
            "task_config_sha256": state["task_config_sha256"],
            "ledger_sha256": state["ledger_sha256_at_init"],
            "trace_ids": policy["required_trace_ids"],
            "budget_seconds": args.budget_seconds,
            "search_profile": search_profile,
            "search_strategy": search_profile_config["strategy"],
        },
    )
    write_json_atomic(args.state, state, exclusive=True)
    return {
        "initialized": True,
        "state": str(args.state),
        "events": str(args.events),
        "deadline_utc": state["deadline_utc"],
        "trace_ids": policy["required_trace_ids"],
        "search_profile": search_profile,
        "search_strategy": search_profile_config["strategy"],
        "submission_allowed": False,
        "clean_benchmark_claim_allowed": state[
            "clean_benchmark_claim_allowed"
        ],
    }


def command_register(args: argparse.Namespace) -> dict[str, Any]:
    proposal = load_json(args.proposal)
    with LockedState(args.state) as state:
        policy = load_json(Path(state["policy_path"]))
        validate_policy(policy)
        validate_proposal(state, proposal, policy)
        trace_id = proposal["trace_id"]
        proposal["registered_at_utc"] = utc_text()
        proposal["status"] = "planned"
        parents: list[str] = []
        search_node: dict[str, Any] | None = None
        if state.get("search_profile", LEGACY_PROFILE) != LEGACY_PROFILE:
            parents = proposal_parents(proposal)
            search_node = register_search_node(
                state, policy, proposal, parents
            )
        proposal["priority_at_registration"] = search_priority(
            state,
            policy,
            proposal,
            state["trace_stats"][trace_id]["pulls"],
            sum(item["pulls"] for item in state["trace_stats"].values()),
        )
        state["proposals"][proposal["node"]] = proposal
        append_event(
            state,
            "proposal_registered",
            {
                "node": proposal["node"],
                "trace_id": trace_id,
                "root_axis": proposal["root_axis"],
                "fidelity": proposal["fidelity"],
                "priority": proposal["priority_at_registration"],
                "search_profile": state.get("search_profile", LEGACY_PROFILE),
                "parents": parents,
                "depth": search_node["depth"] if search_node else 0,
            },
        )
        return {
            "registered": True,
            "node": proposal["node"],
            "priority": proposal["priority_at_registration"],
            "search_profile": state.get("search_profile", LEGACY_PROFILE),
            "parents": parents,
            "depth": search_node["depth"] if search_node else 0,
        }


def choose_next(state: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any] | None:
    remaining_seconds = (
        parse_utc_text(state["deadline_utc"], "deadline_utc") - utc_now()
    ).total_seconds()
    if remaining_seconds <= 0:
        return None
    pending = [
        item
        for item in state["proposals"].values()
        if item["status"] == "planned"
        and float(item["context"]["expected_minutes"]) * 60.0
        <= remaining_seconds
    ]
    if not pending:
        return None
    trace_order = policy["required_trace_ids"]
    if policy["selection"]["force_untried_trace_round_robin"]:
        untried = [
            trace_id
            for trace_id in trace_order
            if state["trace_stats"][trace_id]["pulls"] == 0
        ]
        if untried:
            first = untried[0]
            candidates = [item for item in pending if item["trace_id"] == first]
            if candidates:
                pending = candidates
    total = sum(item["pulls"] for item in state["trace_stats"].values())
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for item in pending:
        pulls = state["trace_stats"][item["trace_id"]]["pulls"]
        priority = search_priority(state, policy, item, pulls, total)
        ranked.append((priority, str(item["node"]), item))
    ranked.sort(key=lambda value: (-value[0], value[1]))
    selected = ranked[0][2]
    selected["current_priority"] = ranked[0][0]
    return selected


def command_next(args: argparse.Namespace) -> dict[str, Any]:
    with LockedState(args.state) as state:
        policy = load_json(Path(state["policy_path"]))
        deadline_exhausted = utc_now() >= parse_utc_text(
            state["deadline_utc"], "deadline_utc"
        )
        selected = choose_next(state, policy)
        if selected is None:
            result = {
                "available": False,
                "reason": (
                    "deadline exhausted"
                    if deadline_exhausted
                    else "no planned proposal fits remaining budget"
                ),
            }
        else:
            selected["status"] = "selected"
            selected["selected_at_utc"] = utc_text()
            state["trace_stats"][selected["trace_id"]]["pulls"] += 1
            result = {
                "available": True,
                "node": selected["node"],
                "trace_id": selected["trace_id"],
                "root_axis": selected["root_axis"],
                "fidelity": selected["fidelity"],
                "priority": selected["current_priority"],
                "hypothesis": selected["hypothesis"],
                "shared_memory": state["shared_memory"],
                "finalist_authority": state["finalist_authority"],
                "search_profile": state.get("search_profile", LEGACY_PROFILE),
            }
            if state.get("search_profile", LEGACY_PROFILE) != LEGACY_PROFILE:
                graph_node = state["search_graph"]["nodes"][selected["node"]]
                graph_node["status"] = "selected"
                result["parents"] = graph_node["parents"]
                result["depth"] = graph_node["depth"]
                if "plan" in selected:
                    result["plan"] = selected["plan"]
            append_event(state, "proposal_selected", result)
        if args.receipt:
            write_json_atomic(args.receipt, result, exclusive=True)
        return result


def command_ingest(args: argparse.Namespace) -> dict[str, Any]:
    feedback = load_json(args.feedback)
    with LockedState(args.state) as state:
        policy = load_json(Path(state["policy_path"]))
        validate_policy(policy)
        execution_summary = validate_v3_feedback(state, policy, feedback)
        if feedback.get("task") != state["task"]:
            raise ValueError("feedback task differs from controller task")
        node = str(feedback.get("node", ""))
        if node not in state["proposals"] or node in state["results"]:
            raise ValueError("feedback node is unknown or already ingested")
        proposal = state["proposals"][node]
        if proposal["status"] != "selected":
            raise ValueError("proposal must be selected before feedback ingestion")
        row = find_bound_ledger_row(state, feedback)
        validate_feedback_against_ledger(state, feedback, row)
        outcome = feedback.get("outcome")
        if row.get("status") != outcome:
            raise ValueError("feedback outcome differs from experiment ledger")
        classification = classify_result(
            state, policy, proposal, feedback
        )
        update_search_after_feedback(
            state,
            policy,
            proposal,
            feedback,
            float(classification["robust_utility"]),
            bool(classification["topology_eligible"]),
            execution_summary,
        )
        proposal["status"] = "evaluated"
        state["trace_stats"][proposal["trace_id"]]["completed"] += 1
        state["results"][node] = {
            "feedback_sha256": sha256_file(args.feedback),
            "ledger_row": {
                field: row.get(field, "")
                for field in (
                    "task",
                    "node",
                    "code_sha256",
                    "split_sha256",
                    "task_config_sha256",
                    "prediction_sha256",
                    "output_sha256",
                    "status",
                )
            },
            **classification,
        }
        append_event(
            state,
            "feedback_ingested",
            {
                "node": node,
                "feedback_sha256": sha256_file(args.feedback),
                **classification,
            },
        )
        return {"ingested": True, "node": node, **classification}


def audit_events(state: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    previous = "0" * 64
    count = 0
    with Path(state["events_path"]).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"events line {line_number}: invalid JSON")
                continue
            count += 1
            observed_hash = event.pop("event_sha256", None)
            expected_hash = sha256_bytes(canonical_bytes(event))
            if event.get("seq") != count:
                errors.append(f"events line {line_number}: non-monotonic seq")
            if event.get("prev_event_sha256") != previous:
                errors.append(f"events line {line_number}: previous hash mismatch")
            if observed_hash != expected_hash:
                errors.append(f"events line {line_number}: event hash mismatch")
            previous = str(observed_hash)
    if count != state["event_count"]:
        errors.append("state event_count differs from event log")
    if previous != state["last_event_sha256"]:
        errors.append("state last_event_sha256 differs from event log")
    return errors


def audit_search_graph(
    state: dict[str, Any], policy: dict[str, Any]
) -> list[str]:
    if state.get("search_profile", LEGACY_PROFILE) == LEGACY_PROFILE:
        return []
    errors: list[str] = []
    profile_name, profile = active_search_profile(state, policy)
    graph = state.get("search_graph")
    if not isinstance(graph, dict):
        return ["v3 controller search_graph is missing"]
    if graph.get("profile") != profile_name:
        errors.append("search graph profile differs from controller profile")
    nodes = graph.get("nodes")
    if not isinstance(nodes, dict):
        return errors + ["search graph nodes are invalid"]
    if set(nodes) != set(state["proposals"]):
        errors.append("search graph nodes differ from registered proposals")
    for node_id, node in nodes.items():
        for parent_id in node.get("parents", []):
            parent = nodes.get(parent_id)
            if parent is None or node_id not in parent.get("children", []):
                errors.append(f"search graph edge {parent_id}->{node_id} is invalid")
                continue
            if int(parent["depth"]) >= int(node["depth"]):
                errors.append(f"search graph edge {parent_id}->{node_id} is cyclic")
        if node_id in state["results"] and node.get("status") != "evaluated":
            errors.append(f"evaluated graph node {node_id} has wrong status")
    signatures = graph.get("signature_to_node", {})
    if profile["strategy"] == "mcgs_graph":
        if not isinstance(signatures, dict) or len(signatures) != len(nodes):
            errors.append("MCGS graph signatures are incomplete")
    if profile["strategy"] == "pes":
        completed = {
            node_id
            for node_id, node in nodes.items()
            if node.get("status") == "evaluated"
        }
        summarized = {
            str(item.get("node"))
            for item in state.get("pes_cycles", [])
            if isinstance(item, dict)
        }
        if completed != summarized:
            errors.append("PES evaluated nodes differ from summarized cycles")
    return errors


def command_audit(args: argparse.Namespace) -> dict[str, Any]:
    state = load_json(args.state)
    validate_bound_files(state)
    policy = load_json(Path(state["policy_path"]))
    validate_policy(policy)
    errors = audit_events(state)
    errors.extend(audit_search_graph(state, policy))
    archive = state["archive"]
    if archive["champion"] is not None and archive["champion"].get(
        "fidelity"
    ) != "full":
        errors.append("champion is not full fidelity")
    if len(archive["challengers"]) > policy["archive"]["challenger_capacity"]:
        errors.append("challenger archive exceeds capacity")
    if len(archive["stepping_stones"]) > policy["archive"][
        "stepping_stone_capacity"
    ]:
        errors.append("stepping-stone archive exceeds capacity")
    if any(
        item.get("visibility") != policy["shared_memory"]["allowed_visibility"]
        for item in state["shared_memory"]
    ):
        errors.append("shared memory contains forbidden visibility")
    if state.get("submission_allowed") is not False:
        errors.append("controller state permits submission")
    expected_clean_benchmark = bool(
        state["task_profile"].get("benchmark_claim_allowed", False)
    )
    if state.get("clean_benchmark_claim_allowed") is not expected_clean_benchmark:
        errors.append("controller clean benchmark flag differs from task profile")
    result = {
        "valid": not errors,
        "errors": errors,
        "event_count": state["event_count"],
        "proposal_count": len(state["proposals"]),
        "result_count": len(state["results"]),
        "archive": archive,
        "shared_memory_entries": len(state["shared_memory"]),
        "ledger_sha256_current": sha256_file(Path(state["ledger_path"])),
        "ledger_sha256_at_init": state["ledger_sha256_at_init"],
        "submission_allowed": False,
        "clean_benchmark_claim_allowed": expected_clean_benchmark,
        "search_profile": state.get("search_profile", LEGACY_PROFILE),
        "search_strategy": state.get("search_strategy", LEGACY_PROFILE),
        "search_graph_nodes": len(
            state.get("search_graph", {}).get("nodes", {})
        ),
        "pes_cycles": len(state.get("pes_cycles", [])),
    }
    if errors:
        raise ValueError(json.dumps(result, ensure_ascii=False))
    return result


def emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser("init")
    initialize.add_argument("--policy", type=Path, required=True)
    initialize.add_argument("--state", type=Path, required=True)
    initialize.add_argument("--events", type=Path, required=True)
    initialize.add_argument("--ledger", type=Path, required=True)
    initialize.add_argument("--task-config", type=Path, required=True)
    initialize.add_argument("--task", required=True)
    initialize.add_argument("--competition", required=True)
    initialize.add_argument("--budget-seconds", type=int, required=True)
    initialize.add_argument(
        "--search-profile",
        help=(
            "FORGE v3 profile name; omitted uses the task profile or policy default"
        ),
    )

    register = subparsers.add_parser("register")
    register.add_argument("--state", type=Path, required=True)
    register.add_argument("--proposal", type=Path, required=True)

    next_parser = subparsers.add_parser("next")
    next_parser.add_argument("--state", type=Path, required=True)
    next_parser.add_argument("--receipt", type=Path)

    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("--state", type=Path, required=True)
    ingest.add_argument("--feedback", type=Path, required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--state", type=Path, required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--state", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "init":
            result = command_init(args)
        elif args.command == "register":
            result = command_register(args)
        elif args.command == "next":
            result = command_next(args)
        elif args.command == "ingest":
            result = command_ingest(args)
        elif args.command == "audit":
            result = command_audit(args)
        else:
            state = load_json(args.state)
            validate_bound_files(state)
            result = {
                "task": state["task"],
                "deadline_utc": state["deadline_utc"],
                "trace_stats": state["trace_stats"],
                "archive": state["archive"],
                "shared_memory_entries": len(state["shared_memory"]),
                "submission_allowed": False,
                "clean_benchmark_claim_allowed": state.get(
                    "clean_benchmark_claim_allowed", False
                ),
                "search_profile": state.get("search_profile", LEGACY_PROFILE),
                "search_strategy": state.get(
                    "search_strategy", LEGACY_PROFILE
                ),
                "search_graph_nodes": len(
                    state.get("search_graph", {}).get("nodes", {})
                ),
                "pes_cycles": len(state.get("pes_cycles", [])),
            }
        emit(result)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        emit({"valid": False, "error": str(exc)})
        raise SystemExit(1)


if __name__ == "__main__":
    main()
