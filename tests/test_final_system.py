from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from final_system.broker import SubmissionBroker
from final_system.calibration import FeedbackCalibrator
from final_system.config import SystemConfig
from final_system.controller import FinalController
from final_system.day_gate import DayResourceGate
from final_system.evaluation import (
    freeze_contract,
    validate_kernel_package,
    validate_submission_csv,
    verify_contract,
)
from final_system.io import atomic_json, canonical, read_json
from final_system.kaggle import DryRunAdapter, KaggleAdapter, SubmitResult
from final_system.registry import CandidateRegistry
from final_system.runners import ClaudeSubscriptionRunner, OpenRouterCodexRunner
from final_system.security import build_agent_sandbox, verify_agent_sandbox
from final_system.selection import SelectionManager
from native import main as native_main
from native import tools as native_tools
from search_system.ioai_agent_system.models import AgentResult

ROOT = Path(__file__).resolve().parents[1]


def make_assets(root: Path) -> Path:
    assets = root / "assets"
    assets.mkdir()
    (assets / "sample_submission.csv").write_text(
        "id,pred\na,0\nb,0\n", encoding="utf-8"
    )
    return assets


def make_evaluation(root: Path) -> Path:
    source = root / "evaluator"
    source.mkdir()
    (source / "metric.py").write_text(
        "import numpy as np\n"
        "def score(y_true, y_pred):\n"
        "    return -float(np.mean((np.asarray(y_true)-np.asarray(y_pred))**2))\n",
        encoding="utf-8",
    )
    (source / "folds.json").write_text(
        json.dumps({
            "ids": ["a", "b", "c", "d"],
            "fold": [0, 0, 1, 1],
            "labels": [0.0, 1.0, 0.0, 1.0],
            "scheme": "kfold",
        }),
        encoding="utf-8",
    )
    (source / "metric_tests.json").write_text(
        json.dumps({
            "cases": [{
                "y_true": [0.0, 1.0],
                "y_pred": [0.0, 1.0],
                "expected": 0.0,
                "tolerance": 1e-8,
            }],
        }),
        encoding="utf-8",
    )
    (source / "contract_evidence.json").write_text(
        json.dumps({
            "metric_name": "negative_mse",
            "direction": "maximize",
            "metric_source": "assets/competition_description.md",
            "id_source": "assets/train.csv:id",
            "label_source": "assets/train.csv:target",
            "split_rationale": "deterministic two-fold unit test",
        }),
        encoding="utf-8",
    )
    target = root / "evaluation"
    freeze_contract(source, target, direction="maximize")
    return target


def make_candidate(
    root: Path, *, lane: str, candidate_id: str, predictions: list[float],
    csv_values: tuple[str, str],
) -> Path:
    candidate = root / f"source-{lane}-{candidate_id}"
    (candidate / "out").mkdir(parents=True)
    (candidate / "evidence").mkdir()
    (candidate / "candidate.json").write_text(
        json.dumps({
            "schema_version": 1,
            "candidate_id": candidate_id,
            "source_lane": lane,
            "submission_mode": "csv",
            "accelerator": "cpu",
            "purpose": candidate_id,
        }),
        encoding="utf-8",
    )
    (candidate / "out" / "submission.csv").write_text(
        f"id,pred\na,{csv_values[0]}\nb,{csv_values[1]}\n", encoding="utf-8"
    )
    np.save(candidate / "out" / "oof.npy", np.asarray(predictions))
    (candidate / "evidence" / "NOTES.md").write_text("test", encoding="utf-8")
    return candidate


def test_config_is_formal_duration_agnostic() -> None:
    config = SystemConfig.load(ROOT / "configs/final_agent_system.toml", repo_root=ROOT)
    assert config.run.max_submissions == 50
    assert config.run.final_reserve == 8
    assert config.claude.model == "claude-fable-5"
    assert (
        config.claude.integrated_profile_dir
        != config.claude.direct_profile_dir
    )
    assert config.codex.model == "openai/gpt-5.6-sol"
    assert config.selection.model == "claude-fable-5"
    assert config.selection.effort == "high"
    assert config.run.metric_direction == "auto"
    assert config.resources.gpu_quota_hours == 30
    assert config.resources.gpu_concurrency == 2
    assert config.resources.cpu_concurrency == 5


def test_formal_live_controller_requires_shared_pool_and_floor_group(
    tmp_path: Path,
) -> None:
    config = SystemConfig.load(
        ROOT / "configs/final_agent_system.toml", repo_root=ROOT
    )
    with pytest.raises(ValueError, match="floor-group-id"):
        FinalController(
            config=config, slug="task-a", assets_dir=tmp_path,
            duration_minutes=360, competition_mode="formal",
            kaggle_user="user", live=True, resource_pool_id="event",
            day_slugs=("task-a", "task-b", "task-c"),
        )


def test_evaluation_contract_is_frozen_and_hash_bound(tmp_path: Path) -> None:
    evaluation = make_evaluation(tmp_path)
    contract = verify_contract(evaluation)
    assert contract["unit_count"] == 4
    (evaluation / "folds.json").write_text("{}", encoding="utf-8")
    try:
        verify_contract(evaluation)
    except ValueError as exc:
        assert "changed after freeze" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("mutated contract was accepted")


def test_evaluation_contract_can_replace_controller_precreated_empty_mount(
    tmp_path: Path,
) -> None:
    source = tmp_path / "evaluator"
    source.mkdir()
    (source / "metric.py").write_text(
        "def score(y_true, y_pred): return 0.0\n", encoding="utf-8"
    )
    (source / "folds.json").write_text(
        json.dumps({
            "ids": ["a", "b"], "fold": [0, 1], "labels": [0, 1],
            "scheme": "kfold",
        }),
        encoding="utf-8",
    )
    (source / "metric_tests.json").write_text(
        json.dumps({
            "cases": [{
                "y_true": [0, 1], "y_pred": [1, 0], "expected": 0.0,
            }],
        }),
        encoding="utf-8",
    )
    (source / "contract_evidence.json").write_text(
        json.dumps({
            "metric_name": "constant_test_metric",
            "direction": "maximize",
            "metric_source": "assets/overview.txt",
            "id_source": "assets/train.csv:id",
            "label_source": "assets/train.csv:label",
            "split_rationale": "one deterministic unit per fold",
        }),
        encoding="utf-8",
    )
    target = tmp_path / "evaluation"
    target.mkdir()
    contract = freeze_contract(source, target, direction="maximize")
    assert verify_contract(target) == contract


def test_evaluation_contract_rejects_a_metric_that_fails_its_known_answer(
    tmp_path: Path,
) -> None:
    source = tmp_path / "evaluator"
    source.mkdir()
    (source / "metric.py").write_text(
        "def score(y_true, y_pred): return 0.0\n", encoding="utf-8"
    )
    (source / "folds.json").write_text(json.dumps({
        "ids": ["a", "b"], "fold": [0, 1], "labels": [0, 1],
        "scheme": "kfold",
    }), encoding="utf-8")
    (source / "metric_tests.json").write_text(json.dumps({
        "cases": [{"y_true": [0], "y_pred": [1], "expected": 1.0}],
    }), encoding="utf-8")
    (source / "contract_evidence.json").write_text(json.dumps({
        "metric_name": "broken",
        "direction": "maximize",
        "metric_source": "assets/overview.txt",
        "id_source": "assets/train.csv:id",
        "label_source": "assets/train.csv:label",
        "split_rationale": "deterministic unit test",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="metric test case 0 failed"):
        freeze_contract(source, tmp_path / "evaluation", direction="maximize")


def test_evaluation_contract_resolves_metric_direction_from_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "evaluator"
    source.mkdir()
    (source / "metric.py").write_text(
        "def score(y_true, y_pred): return float((y_true[0]-y_pred[0])**2)\n",
        encoding="utf-8",
    )
    (source / "folds.json").write_text(json.dumps({
        "ids": ["a", "b"], "fold": [0, 1], "labels": [0, 1],
        "scheme": "kfold",
    }), encoding="utf-8")
    (source / "metric_tests.json").write_text(json.dumps({
        "cases": [{"y_true": [0], "y_pred": [1], "expected": 1.0}],
    }), encoding="utf-8")
    (source / "contract_evidence.json").write_text(json.dumps({
        "metric_name": "mse", "direction": "minimize",
        "metric_source": "assets/overview.txt",
        "id_source": "assets/train.csv:id",
        "label_source": "assets/train.csv:label",
        "split_rationale": "deterministic two-fold test",
    }), encoding="utf-8")
    contract = freeze_contract(
        source, tmp_path / "evaluation", direction="auto"
    )
    assert contract["direction"] == "minimize"


def test_submission_csv_uses_official_submission_csv_as_template(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "submission.csv").write_text(
        "id,pred\na,0\nb,0\nc,0\n", encoding="utf-8"
    )
    candidate = tmp_path / "garbage.csv"
    candidate.write_text("id,pred\nfake,1\n", encoding="utf-8")
    result = validate_submission_csv(assets, candidate)
    assert result["valid"] is False
    assert result["sample_submission"].endswith("submission.csv")
    assert any("row count mismatch" in error for error in result["errors"])
    candidate.write_text(
        "id,pred\na,0.1\nb,nan\nc,0.3\n", encoding="utf-8"
    )
    result = validate_submission_csv(assets, candidate)
    assert result["valid"] is False
    assert any("non-finite" in error for error in result["errors"])


def test_kernel_package_rejects_helpers_that_kaggle_would_not_upload(
    tmp_path: Path,
) -> None:
    kernel = tmp_path / "candidate" / "out" / "kernel"
    kernel.mkdir(parents=True)
    (kernel / "main.py").write_text("import helper\n", encoding="utf-8")
    (kernel / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (kernel / "kernel-metadata.json").write_text(json.dumps({
        "code_file": "main.py", "kernel_type": "script", "language": "python",
        "dataset_sources": [], "kernel_sources": [], "model_sources": [],
    }), encoding="utf-8")
    result = validate_kernel_package(tmp_path / "candidate")
    assert result["valid"] is False
    assert any("standalone" in error for error in result["errors"])


def test_registry_recomputes_score_and_rejects_duplicate_prediction_file(
    tmp_path: Path,
) -> None:
    assets = make_assets(tmp_path)
    evaluation = make_evaluation(tmp_path)
    registry = CandidateRegistry(tmp_path / "registry", assets=assets, evaluation=evaluation)
    first = make_candidate(
        tmp_path, lane="codex", candidate_id="c1",
        predictions=[0, 1, 0, 1], csv_values=("0.1", "0.2"),
    )
    second = make_candidate(
        tmp_path, lane="claude", candidate_id="c2",
        predictions=[0, 1, 0, 1], csv_values=("0.1", "0.2"),
    )
    one = registry.register(first, source_lane="codex")
    two = registry.register(second, source_lane="claude")
    assert one["evaluation"]["score_source"] == "common_harness"
    assert one["evaluation"]["mean"] == 0.0
    assert two["status"] == "duplicate"
    assert two["duplicate_of"] == one["candidate_id"]


def test_broker_separates_source_from_class_and_preserves_final_reserve(
    tmp_path: Path,
) -> None:
    assets = make_assets(tmp_path)
    evaluation = make_evaluation(tmp_path)
    registry = CandidateRegistry(tmp_path / "registry", assets=assets, evaluation=evaluation)
    candidates = [
        ("hearsay", "h1", [0, 1, 0, 1], ("0.11", "0.12")),
        ("codex", "c1", [0, 0.9, 0, 0.9], ("0.21", "0.22")),
        ("claude", "a1", [0, 0.8, 0, 0.8], ("0.31", "0.32")),
    ]
    for lane, cid, predictions, values in candidates:
        registry.register(
            make_candidate(
                tmp_path, lane=lane, candidate_id=cid,
                predictions=predictions, csv_values=values,
            ),
            source_lane=lane,
        )
    broker = SubmissionBroker(
        tmp_path / "control", registry=registry,
        adapter=DryRunAdapter("slug", tmp_path), max_submissions=6,
        final_reserve=2, initial_calibrations=3, final_start_fraction=0.8,
        anti_monopoly_fraction=0.5, min_local_gain=0.0, direction="maximize",
    )
    first = broker.tick(fraction_elapsed=0.1)
    second = broker.tick(fraction_elapsed=0.2)
    third = broker.tick(fraction_elapsed=0.3)
    assert first and first["submission_class"] == "floor"
    assert first["source_lane"] == "hearsay"
    assert second and second["submission_class"] == "calibration"
    assert third and third["submission_class"] == "calibration"
    assert broker.remaining() == 3
    assert broker.tick(fraction_elapsed=0.5) is None
    status = read_json(tmp_path / "control" / "BROKER_STATUS.json")
    assert status["final_reserve"] == 2
    assert status["by_source_lane"] == {"claude": 1, "codex": 1, "hearsay": 1}


def test_reserved_attempt_does_not_consume_or_retry_after_restart(tmp_path: Path) -> None:
    assets = make_assets(tmp_path)
    evaluation = make_evaluation(tmp_path)
    registry = CandidateRegistry(tmp_path / "registry", assets=assets, evaluation=evaluation)
    record = registry.register(
        make_candidate(
            tmp_path, lane="codex", candidate_id="c1",
            predictions=[0, 1, 0, 1], csv_values=("0.1", "0.2"),
        ), source_lane="codex",
    )
    control = tmp_path / "control"
    broker = SubmissionBroker(
        control, registry=registry, adapter=DryRunAdapter("slug", tmp_path),
        max_submissions=5, final_reserve=2, initial_calibrations=3,
        final_start_fraction=0.8, anti_monopoly_fraction=0.5,
        min_local_gain=0.0, direction="maximize",
    )
    state = read_json(control / "broker_state.json")
    state["submissions"].append({
        "submission_id": "sub-crash", "candidate_id": record["candidate_id"],
        "source_lane": "codex", "submission_class": "floor",
        "result": {"consumed": False, "status": "reserved"},
    })
    atomic_json(control / "broker_state.json", state)
    resumed = SubmissionBroker(
        control, registry=registry, adapter=DryRunAdapter("slug", tmp_path),
        max_submissions=5, final_reserve=2, initial_calibrations=3,
        final_start_fraction=0.8, anti_monopoly_fraction=0.5,
        min_local_gain=0.0, direction="maximize",
    )
    assert resumed.remaining() == 5
    assert resumed.tick(fraction_elapsed=0.2) is None


def test_codex_openrouter_runner_isolates_home_and_never_inherits_kaggle(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    monkeypatch.setenv("KAGGLE_USERNAME", "must-not-leak")
    runner = OpenRouterCodexRunner(
        binary=Path("/opt/homebrew/bin/codex"), model="openai/gpt-5.6-sol",
        effort="high", provider_id="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY", supports_web_search=True,
    )
    env = runner.environment(tmp_path)
    assert env["HOME"].startswith(str(tmp_path))
    assert env["CODEX_HOME"].startswith(str(tmp_path))
    assert "KAGGLE_USERNAME" not in env
    argv = runner.argv(
        workdir=tmp_path, last_message=tmp_path / "last.md",
        resume_session_id=None, persist_session=True,
    )
    joined = " ".join(argv)
    assert 'model_provider="openrouter"' in joined
    assert "supports_standalone_web_search=true" in joined
    assert "--ignore-rules" in argv
    assert 'shell_environment_policy.inherit="core"' in joined
    assert "shell_environment_policy.ignore_default_excludes=false" in joined
    assert 'filters.OPENROUTER_API_KEY="exclude"' in joined


def test_selection_claude_runner_is_read_only_and_has_no_kaggle_env(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "must-not-leak")
    monkeypatch.setenv("KAGGLE_KEY", "must-not-leak")
    runner = ClaudeSubscriptionRunner(
        binary=Path("/opt/homebrew/bin/claude"), model="claude-fable-5",
        effort="high", profile_dir=tmp_path / "profile", allowed_tools="Read",
    )
    assert runner.allowed_tools == "Read"
    argv = runner.argv(resume_session_id=None, persist_session=False)
    assert argv[argv.index("--model") + 1] == "claude-fable-5"
    assert argv[argv.index("--effort") + 1] == "high"
    assert argv[argv.index("--allowedTools") + 1] == "Read"
    env = runner.environment(tmp_path / "manager")
    assert "KAGGLE_USERNAME" not in env
    assert "KAGGLE_KEY" not in env


def test_feedback_calibrator_versions_without_mutating_e0(tmp_path: Path) -> None:
    records = []
    submissions = []
    for index, public in enumerate((0.1, 0.2, 0.3, 0.4, 0.5, 0.6)):
        candidate_id = f"c{index}"
        records.append({
            "candidate_id": candidate_id,
            "evaluation": {
                "status": "ok", "mean": 0.5,
                "std": abs(public - 0.5), "pooled": 0.5,
                "per_fold": [public, 1.0 - public],
            },
        })
        submissions.append({
            "candidate_id": candidate_id, "source_lane": "codex",
            "leaderboard_score": public,
        })
    frozen_e0 = copy.deepcopy(records)
    calibrator = FeedbackCalibrator(
        tmp_path / "calibration", min_feedback=4, ridge_strength=0.01,
        min_rank_gain=0.05, max_blend_weight=0.5,
    )
    first = calibrator.update(records, submissions)
    second = calibrator.update(records, submissions)
    assert records == frozen_e0
    assert first["version"] == second["version"]
    assert first["base_contract"] == "immutable_E0_metric_and_folds"
    assert first["active"] is True
    assert first["calibrated_loo_spearman"] > first["baseline_spearman"]
    assert 0 < first["blend_weight"] <= 0.5
    assert len(first["candidate_predictions"]) == len(records)
    assert len(list((tmp_path / "calibration" / "history").glob("*.json"))) == 1


def test_selection_manager_persists_digest_bound_advice(tmp_path: Path) -> None:
    class FakeDecisionRunner:
        async def run(self, **kwargs):
            trace = Path(kwargs["trace_dir"]) / "fake.jsonl"
            trace.write_text("{}\n", encoding="utf-8")
            return AgentResult(
                agent_id=kwargs["agent_id"], role=kwargs["role"],
                workdir=Path(kwargs["workdir"]), trace_path=trace,
                exit_code=0, timed_out=False, duration_s=0.01,
                resolved_model="claude-fable-5",
                result_text=json.dumps({
                    "decision": "submit", "candidate_id": "candidate-b",
                    "reason": "折间更稳定", "evidence": ["std 更小"],
                    "confidence": 0.8, "risk_flags": ["public_lb_shift"],
                }),
            )

    core = {
        "schema_version": 1,
        "submission_class": "milestone",
        "candidates": [
            {"candidate_id": "candidate-a"},
            {"candidate_id": "candidate-b"},
        ],
    }
    context = {
        **core,
        "context_sha256": hashlib.sha256(canonical(core)).hexdigest(),
    }
    manager = SelectionManager(
        tmp_path / "selection_manager", runner=FakeDecisionRunner(),
        model="claude-fable-5", effort="high", timeout_seconds=10,
        recommendation_ttl_seconds=60,
    )
    recommendation = asyncio.run(manager.recommend(context))
    assert recommendation
    assert recommendation["candidate_id"] == "candidate-b"
    assert recommendation["context_sha256"] == context["context_sha256"]
    assert recommendation["submission_authority"] == "none_advisory_only"
    assert read_json(manager.latest_path) == recommendation
    assert list(manager.inputs.glob("*.json"))
    assert list(manager.history.glob("*.json"))


def test_broker_obeys_valid_manager_choice_and_emits_complete_route_feedback(
    tmp_path: Path,
) -> None:
    assets = make_assets(tmp_path)
    evaluation = make_evaluation(tmp_path)
    registry = CandidateRegistry(
        tmp_path / "registry", assets=assets, evaluation=evaluation
    )
    for cid, predictions, values in (
        ("c1", [0, 1, 0, 1], ("0.1", "0.2")),
        ("c2", [0, 0.8, 0, 0.8], ("0.3", "0.4")),
    ):
        registry.register(
            make_candidate(
                tmp_path, lane="codex", candidate_id=cid,
                predictions=predictions, csv_values=values,
            ),
            source_lane="codex",
        )

    class ScoringAdapter(DryRunAdapter):
        def __init__(self):
            super().__init__("slug", tmp_path)
            self.submission_id = ""

        def submit(self, record, submission_id, submission_class):
            self.submission_id = submission_id
            return super().submit(record, submission_id, submission_class)

        def scores(self):
            return {self.submission_id: 0.77} if self.submission_id else {}

    recommendation_path = tmp_path / "control" / "selection_manager" / "recommendation.json"
    calibration_path = tmp_path / "control" / "calibration" / "latest.json"
    atomic_json(calibration_path, {
        "version": "cal-v1", "active": False, "candidate_predictions": {},
    })
    adapter = ScoringAdapter()
    broker = SubmissionBroker(
        tmp_path / "control", registry=registry, adapter=adapter,
        max_submissions=3, final_reserve=0, initial_calibrations=1,
        final_start_fraction=0.8, anti_monopoly_fraction=0.5,
        min_local_gain=0.0, direction="maximize", manager_enabled=True,
        recommendation_path=recommendation_path,
        calibration_path=calibration_path, manager_fallback_seconds=60,
    )
    context = broker.manager_context(fraction_elapsed=0.1)
    assert context is not None
    chosen = context["candidates"][-1]["candidate_id"]
    atomic_json(recommendation_path, {
        "decision": "submit", "candidate_id": chosen,
        "context_sha256": context["context_sha256"],
        "expires_at": time.time() + 60,
    })
    submitted = broker.tick(fraction_elapsed=0.1)
    assert submitted and submitted["candidate_id"] == chosen
    assert submitted["selection_reason"] == "selection_manager"
    assert broker.refresh_scores() == 1
    route_lines = [
        json.loads(line) for line in (
            tmp_path / "control" / "feedback" / "codex.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert [line["event"] for line in route_lines] == [
        "submission_result", "leaderboard_score"
    ]
    scored = route_lines[-1]
    assert scored["local_evaluation"]["per_fold"] is not None
    assert scored["local_evaluation"]["contract_sha256"]
    assert scored["leaderboard"]["public_score"] == 0.77
    assert scored["leaderboard"]["local_public_residual"] is not None
    assert scored["selection_reason"] == "selection_manager"


def test_active_calibration_reorders_fallback_without_changing_e0(
    tmp_path: Path,
) -> None:
    assets = make_assets(tmp_path)
    evaluation = make_evaluation(tmp_path)
    registry = CandidateRegistry(
        tmp_path / "registry", assets=assets, evaluation=evaluation
    )
    best_local = registry.register(
        make_candidate(
            tmp_path, lane="codex", candidate_id="local-best",
            predictions=[0, 1, 0, 1], csv_values=("0.1", "0.2"),
        ),
        source_lane="codex",
    )
    calibrated_best = registry.register(
        make_candidate(
            tmp_path, lane="claude", candidate_id="calibrated-best",
            predictions=[0, 0.8, 0, 0.8], csv_values=("0.3", "0.4"),
        ),
        source_lane="claude",
    )
    assert best_local["evaluation"]["mean"] > calibrated_best["evaluation"]["mean"]
    calibration_path = tmp_path / "control" / "calibration" / "latest.json"
    atomic_json(calibration_path, {
        "active": True,
        "candidate_predictions": {
            best_local["candidate_id"]: {"adjusted_score": -0.2},
            calibrated_best["candidate_id"]: {"adjusted_score": -0.1},
        },
    })
    broker = SubmissionBroker(
        tmp_path / "control", registry=registry,
        adapter=DryRunAdapter("slug", tmp_path), max_submissions=2,
        final_reserve=0, initial_calibrations=1, final_start_fraction=0.8,
        anti_monopoly_fraction=0.5, min_local_gain=0.0,
        direction="maximize", calibration_path=calibration_path,
    )
    submitted = broker.tick(fraction_elapsed=0.1)
    assert submitted and submitted["candidate_id"] == calibrated_best["candidate_id"]
    assert registry.get(best_local["candidate_id"])["evaluation"] == best_local["evaluation"]


def test_broker_falls_back_when_manager_recommendation_is_stale(
    tmp_path: Path,
) -> None:
    assets = make_assets(tmp_path)
    evaluation = make_evaluation(tmp_path)
    registry = CandidateRegistry(
        tmp_path / "registry", assets=assets, evaluation=evaluation
    )
    registry.register(
        make_candidate(
            tmp_path, lane="codex", candidate_id="candidate",
            predictions=[0, 1, 0, 1], csv_values=("0.1", "0.2"),
        ),
        source_lane="codex",
    )
    recommendation = tmp_path / "control" / "selection_manager" / "recommendation.json"
    atomic_json(recommendation, {
        "decision": "submit", "candidate_id": "not-eligible",
        "context_sha256": "stale", "expires_at": time.time() + 60,
    })
    broker = SubmissionBroker(
        tmp_path / "control", registry=registry,
        adapter=DryRunAdapter("slug", tmp_path), max_submissions=2,
        final_reserve=0, initial_calibrations=1, final_start_fraction=0.8,
        anti_monopoly_fraction=0.5, min_local_gain=0.0,
        direction="maximize", manager_enabled=True,
        recommendation_path=recommendation, manager_fallback_seconds=0,
    )
    submitted = broker.tick(fraction_elapsed=0.1)
    assert submitted
    assert submitted["selection_reason"] == "deterministic_manager_fallback"


def test_hearsay_each_solver_receives_every_route_feedback_once(tmp_path: Path) -> None:
    control = tmp_path / "control"
    feedback = control / "feedback" / "hearsay.jsonl"
    feedback.parent.mkdir(parents=True)
    feedback.write_text('{"event":"one"}\n{"event":"two"}\n', encoding="utf-8")
    args = SimpleNamespace(external_broker_dir=str(control))
    native_main.EXTERNAL_FEEDBACK_CURSOR.clear()
    first = native_main.external_feedback(args, "solver_a")
    assert '"one"' in first and '"two"' in first
    assert native_main.external_feedback(args, "solver_a") == ""
    assert '"one"' in native_main.external_feedback(args, "solver_b")
    with feedback.open("a", encoding="utf-8") as handle:
        handle.write('{"event":"three"}\n')
    assert '"three"' in native_main.external_feedback(args, "solver_a")


def test_hearsay_external_broker_exposes_no_kaggle_or_gpu_tools(
    tmp_path: Path,
) -> None:
    native_tools.init_ctx(
        tmp_path / "workspace", "slug", budget=object(), trace=object(),
        external_broker=tmp_path / "control",
    )
    _server, names = native_tools.make_server("solver_a")
    assert "mcp__ioai__submission_status" in names
    assert not any("kaggle" in name for name in names)
    assert "mcp__ioai__gpu_kernel_lease" not in names
    native_tools.init_ctx(
        tmp_path / "workspace", "slug", budget=object(), trace=object(),
        external_broker=None,
    )


def test_live_adapter_reads_structured_remaining_quota(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        "final_system.kaggle.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout='{"numTotal": 7, "numAllowedNow": 43}', stderr="",
            returncode=0,
        ),
    )
    assets = make_assets(tmp_path)
    gate = DayResourceGate(
        tmp_path / "resource", pool_id="event", floor_group_id="day-1",
        expected_slugs=("slug",),
        gpu_limit_hours=30, gpu_concurrency=2, cpu_concurrency=5,
    )
    adapter = KaggleAdapter(
        slug="slug", root=tmp_path, kaggle_user="user", submission_mode="auto",
        assets=assets, resource_gate=gate,
        competition_deadline_epoch=time.time() + 3600,
    )
    assert adapter.remaining_today() == 43


def test_kernel_adapter_validates_downloaded_output_before_submission(
    tmp_path: Path, monkeypatch,
) -> None:
    assets = make_assets(tmp_path)
    candidate = tmp_path / "candidate"
    kernel = candidate / "out" / "kernel"
    kernel.mkdir(parents=True)
    (kernel / "solution.py").write_text("print('run')\n", encoding="utf-8")
    source_metadata = {
        "code_file": "solution.py", "kernel_type": "script", "language": "python",
        "dataset_sources": ["owner/dataset"],
        "kernel_sources": ["owner/kernel"],
        "model_sources": ["owner/model/framework/model/1"],
    }
    (kernel / "kernel-metadata.json").write_text(
        json.dumps(source_metadata), encoding="utf-8"
    )
    gate = DayResourceGate(
        tmp_path / "resource", pool_id="event", floor_group_id="day-1",
        expected_slugs=("slug",),
        gpu_limit_hours=30, gpu_concurrency=2, cpu_concurrency=5,
        poll_seconds=0.01,
    )
    adapter = KaggleAdapter(
        slug="slug", root=tmp_path / "control", kaggle_user="user",
        submission_mode="kernel", assets=assets, resource_gate=gate,
        competition_deadline_epoch=time.time() + 3600,
        default_cpu_kernel_minutes=1,
    )
    monkeypatch.setattr("final_system.kaggle.push_kernel", lambda _path: (True, 7, "ok"))
    monkeypatch.setattr(
        "final_system.kaggle.poll_kernel", lambda _ref, timeout_s: ("complete", "ok")
    )
    from kaggle.api.kaggle_api_extended import KaggleApi

    monkeypatch.setattr(KaggleApi, "authenticate", lambda self: None)
    submit_calls: list[dict] = []

    def fake_output(self, kernel_ref, path, **kwargs):
        del self, kernel_ref, kwargs
        Path(path, "submission.csv").write_text(
            "id,pred\nwrong,1\n", encoding="utf-8"
        )

    def fake_submit(self, *args, **kwargs):
        del self, args
        submit_calls.append(kwargs)

    monkeypatch.setattr(KaggleApi, "kernels_output", fake_output)
    monkeypatch.setattr(KaggleApi, "competition_submit_code", fake_submit)
    result = adapter.submit(
        {
            "candidate_id": "candidate", "source_lane": "codex",
            "snapshot_path": str(candidate), "submission_mode": "kernel",
            "accelerator": "cpu", "estimated_kernel_minutes": 0.1,
        },
        "sub-1", "floor",
    )
    assert result.status == "execution_error"
    assert result.consumed is False
    assert submit_calls == []
    assert gate.status()["cpu_inflight"] == 0
    pushed = json.loads(
        (tmp_path / "control" / "kernels" / "sub-1" / "kernel-metadata.json")
        .read_text(encoding="utf-8")
    )
    assert pushed["dataset_sources"] == source_metadata["dataset_sources"]
    assert pushed["kernel_sources"] == source_metadata["kernel_sources"]
    assert pushed["model_sources"] == source_metadata["model_sources"]


def test_kernel_adapter_refuses_to_start_after_its_runtime_deadline(
    tmp_path: Path, monkeypatch,
) -> None:
    assets = make_assets(tmp_path)
    candidate = tmp_path / "candidate"
    kernel = candidate / "out" / "kernel"
    kernel.mkdir(parents=True)
    (kernel / "solution.py").write_text("print('run')\n", encoding="utf-8")
    (kernel / "kernel-metadata.json").write_text(json.dumps({
        "code_file": "solution.py", "kernel_type": "script", "language": "python",
        "dataset_sources": [], "kernel_sources": [], "model_sources": [],
    }), encoding="utf-8")
    gate = DayResourceGate(
        tmp_path / "resource", pool_id="event", floor_group_id="day-1",
        expected_slugs=("slug",),
        gpu_limit_hours=30, gpu_concurrency=2, cpu_concurrency=5,
    )
    adapter = KaggleAdapter(
        slug="slug", root=tmp_path / "control", kaggle_user="user",
        submission_mode="kernel", assets=assets, resource_gate=gate,
        competition_deadline_epoch=time.time() + 30,
        default_cpu_kernel_minutes=1, kernel_start_margin_minutes=1,
    )
    monkeypatch.setattr(
        "final_system.kaggle.push_kernel",
        lambda _path: pytest.fail("late kernel must not be pushed"),
    )
    result = adapter.submit(
        {
            "candidate_id": "candidate", "source_lane": "codex",
            "snapshot_path": str(candidate), "submission_mode": "kernel",
            "accelerator": "cpu", "estimated_kernel_minutes": 0.1,
        },
        "sub-late", "floor",
    )
    assert result.status == "deadline_blocked"
    assert gate.status()["cpu_inflight"] == 0


def test_concurrent_broker_reservation_never_exceeds_quota(tmp_path: Path) -> None:
    assets = make_assets(tmp_path)
    evaluation = make_evaluation(tmp_path)
    registry = CandidateRegistry(
        tmp_path / "registry", assets=assets, evaluation=evaluation
    )
    for lane, cid, values in (
        ("codex", "c1", ("0.1", "0.2")),
        ("claude", "c2", ("0.3", "0.4")),
    ):
        registry.register(
            make_candidate(
                tmp_path, lane=lane, candidate_id=cid,
                predictions=[0, 1, 0, 1], csv_values=values,
            ),
            source_lane=lane,
        )

    class SlowAdapter(DryRunAdapter):
        def __init__(self) -> None:
            super().__init__("slug", tmp_path)
            self.calls = 0
            self.lock = threading.Lock()

        def submit(self, record, submission_id, submission_class):
            with self.lock:
                self.calls += 1
            time.sleep(0.05)
            return super().submit(record, submission_id, submission_class)

    adapter = SlowAdapter()
    broker = SubmissionBroker(
        tmp_path / "control", registry=registry, adapter=adapter,
        max_submissions=1, final_reserve=0, initial_calibrations=1,
        final_start_fraction=0.8, anti_monopoly_fraction=0.5,
        min_local_gain=0.0, direction="maximize",
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda _: broker.tick(fraction_elapsed=0.1), range(2)
        ))
    assert sum(result is not None for result in results) == 1
    assert adapter.calls == 1
    assert broker.remaining() == 0


def test_ambiguous_external_result_holds_slot_until_reconciled(tmp_path: Path) -> None:
    assets = make_assets(tmp_path)
    evaluation = make_evaluation(tmp_path)
    registry = CandidateRegistry(
        tmp_path / "registry", assets=assets, evaluation=evaluation
    )
    registry.register(
        make_candidate(
            tmp_path, lane="codex", candidate_id="c1",
            predictions=[0, 1, 0, 1], csv_values=("0.1", "0.2"),
        ),
        source_lane="codex",
    )

    class AmbiguousAdapter(DryRunAdapter):
        def __init__(self):
            super().__init__("slug", tmp_path)
            self.submission_id = ""
            self.reconciled: list[str] = []

        def submit(self, record, submission_id, submission_class):
            self.submission_id = submission_id
            return SubmitResult(False, "ambiguous", "connection closed")

        def submission_states(self):
            return {
                self.submission_id: {"exists": True, "public_score": None}
            }

        def on_reconciled_submission(self, row):
            self.reconciled.append(row["submission_id"])

    adapter = AmbiguousAdapter()
    broker = SubmissionBroker(
        tmp_path / "control", registry=registry,
        adapter=adapter, max_submissions=2,
        final_reserve=0, initial_calibrations=1, final_start_fraction=0.8,
        anti_monopoly_fraction=0.5, min_local_gain=0.0,
        direction="maximize",
    )
    assert broker.tick(fraction_elapsed=0.1) is not None
    assert broker.remaining() == 2
    assert broker.reservable() == 1
    assert broker.refresh_scores() == 1
    assert broker.remaining() == 1
    assert adapter.reconciled == [adapter.submission_id]


def test_known_non_consuming_failure_retries_without_spending_extra_slot(
    tmp_path: Path,
) -> None:
    assets = make_assets(tmp_path)
    evaluation = make_evaluation(tmp_path)
    registry = CandidateRegistry(
        tmp_path / "registry", assets=assets, evaluation=evaluation
    )
    registry.register(
        make_candidate(
            tmp_path, lane="codex", candidate_id="retry-me",
            predictions=[0, 1, 0, 1], csv_values=("0.1", "0.2"),
        ),
        source_lane="codex",
    )

    class FlakyAdapter(DryRunAdapter):
        def __init__(self) -> None:
            super().__init__("slug", tmp_path)
            self.calls = 0

        def submit(self, record, submission_id, submission_class):
            self.calls += 1
            if self.calls == 1:
                # 旧 Adapter 使用 error；Broker 必须安全归一为有限重试。
                return SubmitResult(False, "error", "temporary 503")
            return super().submit(record, submission_id, submission_class)

    adapter = FlakyAdapter()
    broker = SubmissionBroker(
        tmp_path / "control", registry=registry, adapter=adapter,
        max_submissions=3, final_reserve=0, initial_calibrations=1,
        final_start_fraction=0.8, anti_monopoly_fraction=0.5,
        min_local_gain=0.0, direction="maximize",
        max_retryable_attempts=3, retry_backoff_seconds=0,
    )
    first = broker.tick(fraction_elapsed=0.1)
    second = broker.tick(fraction_elapsed=0.1)
    assert first and first["result"]["status"] == "retryable"
    assert first["result"]["consumed"] is False
    assert second and second["result"]["consumed"] is True
    assert adapter.calls == 2
    assert len(broker.consumed()) == 1
    assert broker.remaining() == 2


def test_external_kernel_is_reconciled_instead_of_holding_slot_forever(
    tmp_path: Path,
) -> None:
    assets = make_assets(tmp_path)
    evaluation = make_evaluation(tmp_path)
    registry = CandidateRegistry(
        tmp_path / "registry", assets=assets, evaluation=evaluation
    )
    registry.register(
        make_candidate(
            tmp_path, lane="codex", candidate_id="remote",
            predictions=[0, 1, 0, 1], csv_values=("0.1", "0.2"),
        ),
        source_lane="codex",
    )

    class ReconcileAdapter(DryRunAdapter):
        def submit(self, record, submission_id, submission_class):
            return SubmitResult(
                False, "external_running", "poll timed out",
                kernel_ref="user/kernel", kernel_version=3,
                resource_lease_id="lease-1",
            )

        def reconcile_external(self, record, row):
            return SubmitResult(
                True, "submitted", "resumed exact version",
                external_ref="submission-ref", kernel_ref="user/kernel",
                kernel_version=3,
            )

        def submission_states(self):
            return {}

    broker = SubmissionBroker(
        tmp_path / "control", registry=registry,
        adapter=ReconcileAdapter("slug", tmp_path), max_submissions=2,
        final_reserve=0, initial_calibrations=1, final_start_fraction=0.8,
        anti_monopoly_fraction=0.5, min_local_gain=0.0,
        direction="maximize",
    )
    pending = broker.tick(fraction_elapsed=0.1)
    assert pending and pending["result"]["status"] == "external_running"
    assert broker.reservable() == 1
    assert broker.refresh_scores() == 1
    assert broker.consumed()[0]["result"]["status"] == "submitted"
    assert broker.remaining() == 1


def test_day_resource_gate_is_shared_across_three_controllers(
    tmp_path: Path,
) -> None:
    kwargs = {
        "pool_id": "event",
        "floor_group_id": "day-1",
        "expected_slugs": ("task-a", "task-b", "task-c"),
        "gpu_limit_hours": 30,
        "gpu_concurrency": 2,
        "cpu_concurrency": 5,
        "poll_seconds": 0.01,
    }
    gate_a = DayResourceGate(tmp_path / "pool", **kwargs)
    gate_b = DayResourceGate(tmp_path / "pool", **kwargs)
    blocked, reason = gate_a.acquire(
        submission_id="too-early", slug="task-a", submission_class="milestone",
        accelerator="p100", estimated_seconds=60,
        deadline_epoch=time.time() + 0.04,
    )
    assert blocked is None
    assert "floor" in reason
    for slug in kwargs["expected_slugs"]:
        gate_a.mark_floor(slug)
    first, _ = gate_a.acquire(
        submission_id="one", slug="task-a", submission_class="milestone",
        accelerator="p100", estimated_seconds=60,
        deadline_epoch=time.time() + 1,
    )
    second, _ = gate_b.acquire(
        submission_id="two", slug="task-b", submission_class="milestone",
        accelerator="t4", estimated_seconds=60,
        deadline_epoch=time.time() + 1,
    )
    third, third_reason = gate_a.acquire(
        submission_id="three", slug="task-c", submission_class="milestone",
        accelerator="p100", estimated_seconds=60,
        deadline_epoch=time.time() + 0.04,
    )
    assert first and second
    assert third is None
    assert "concurrency" in third_reason
    assert gate_b.status()["gpu_inflight"] == 2
    gate_a.settle(first["lease_id"], 30)
    third, _ = gate_b.acquire(
        submission_id="three-retry", slug="task-c", submission_class="milestone",
        accelerator="p100", estimated_seconds=60,
        deadline_epoch=time.time() + 1,
    )
    assert third is not None
    gate_b.settle(second["lease_id"], 20)
    gate_b.settle(third["lease_id"], 10)
    assert gate_a.status()["gpu_hours_used"] == pytest.approx(60 / 3600, abs=1e-4)


def test_account_resource_pool_carries_gpu_usage_into_second_day(
    tmp_path: Path,
) -> None:
    common = {
        "pool_id": "event-account-1", "gpu_limit_hours": 30,
        "gpu_concurrency": 2, "cpu_concurrency": 5, "poll_seconds": 0.01,
    }
    day1 = DayResourceGate(
        tmp_path / "pool", floor_group_id="day1",
        expected_slugs=("a", "b", "c"), **common,
    )
    lease, _ = day1.acquire(
        submission_id="floor-a", slug="a", submission_class="floor",
        accelerator="p100", estimated_seconds=3600,
        deadline_epoch=time.time() + 1,
    )
    assert lease
    day1.settle(lease["lease_id"], 3600)
    day2 = DayResourceGate(
        tmp_path / "pool", floor_group_id="day2",
        expected_slugs=("d", "e", "f"), **common,
    )
    status = day2.status()
    assert status["gpu_hours_used"] == 1.0
    assert status["floors"] == {"d": False, "e": False, "f": False}
    assert set(status["floor_groups"]) == {"day1", "day2"}
    orphan, _ = day2.acquire(
        submission_id="orphan", slug="d", submission_class="floor",
        accelerator="p100", estimated_seconds=60,
        deadline_epoch=time.time() + 1,
    )
    assert orphan
    day2.mark_running(orphan["lease_id"], "user/orphan-kernel")
    state = read_json(day2.state_path)
    state["leases"][orphan["lease_id"]]["pid"] = 999_999_999
    atomic_json(day2.state_path, state)
    unknown = day2.external_unknown_leases()
    assert [item["external_ref"] for item in unknown] == ["user/orphan-kernel"]
    day2.release(orphan["lease_id"])


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS Seatbelt regression")
def test_agent_sandbox_blocks_credentials_but_starts_native_agent_clis(
    tmp_path: Path,
) -> None:
    probe = tmp_path / "operator-secret"
    session = tmp_path / "session"
    lane = session / "lane"
    control = session / "control"
    official_assets = lane / "OFFICIAL_ASSETS"
    for path in (lane, control, official_assets):
        path.mkdir(parents=True, exist_ok=True)
    (control / "broker_state.json").write_text("secret control\n", encoding="utf-8")
    (official_assets / "task.txt").write_text("official\n", encoding="utf-8")
    sandbox = build_agent_sandbox(
        probe_secret=probe,
        protected_write_roots=(session,),
        writable_roots=(lane,),
        read_only_roots=(official_assets,),
        protected_read_roots=(session,),
        readable_roots=(lane, official_assets),
    )
    verify_agent_sandbox(sandbox, probe_secret=probe)
    for path, allowed in (
        (lane / "candidate.txt", True),
        (control / "tamper.json", False),
        (official_assets / "tamper.txt", False),
    ):
        result = subprocess.run(
            sandbox.wrap([
                "/bin/sh", "-c", "/bin/echo test > \"$1\"", "sh", str(path),
            ]),
            capture_output=True, text=True, timeout=10,
        )
        assert path.exists() is allowed
        assert (result.returncode == 0) is allowed
    readable = subprocess.run(
        sandbox.wrap(["/bin/cat", str(official_assets / "task.txt")]),
        capture_output=True, text=True, timeout=10,
    )
    hidden = subprocess.run(
        sandbox.wrap(["/bin/cat", str(control / "broker_state.json")]),
        capture_output=True, text=True, timeout=10,
    )
    assert readable.returncode == 0 and readable.stdout.strip() == "official"
    assert hidden.returncode != 0 and hidden.stdout == ""
    for binary in (Path("/opt/homebrew/bin/claude"), Path("/opt/homebrew/bin/codex")):
        if binary.is_file():
            result = subprocess.run(
                sandbox.wrap([str(binary), "--version"]),
                capture_output=True, text=True, timeout=30,
            )
            assert result.returncode == 0, result.stderr
    kaggle = Path(sys.executable).parent / "kaggle"
    if kaggle.is_file():
        result = subprocess.run(
            sandbox.wrap([str(kaggle), "--version"]),
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0


def test_controller_dry_run_closes_full_wiring_without_model_or_kaggle(
    tmp_path: Path, monkeypatch,
) -> None:
    assets = make_assets(tmp_path)
    config = SystemConfig.load(
        ROOT / "configs/final_agent_system.toml", repo_root=ROOT
    )
    config = replace(
        config,
        run=replace(
            config.run, workspace_root=tmp_path / "runs", poll_seconds=0.05,
            max_submissions=4, final_reserve=1, initial_calibrations=2,
            stop_exploration_minutes_before_end=0.01,
            submission_only_minutes_before_end=0.005,
        ),
        search=replace(config.search, enabled=False),
        codex=replace(config.codex, max_rounds=1, turn_minutes=0.01),
        claude=replace(config.claude, max_rounds=1, turn_minutes=0.01),
        hearsay=replace(config.hearsay, enabled=False),
        selection=replace(
            config.selection, interval_seconds=0.05, timeout_seconds=0.2,
            recommendation_ttl_seconds=1.0, fallback_seconds=0.2,
        ),
    )

    class FakeRunner:
        def __init__(self, lane: str):
            self.lane = lane

        async def run(
            self, *, agent_id, role, prompt, workdir, trace_dir, timeout_s,
            persist_session=False, resume_session_id=None,
        ):
            del prompt, timeout_s, persist_session, resume_session_id
            candidate = workdir / "outbox" / f"{self.lane}-candidate"
            (candidate / "out").mkdir(parents=True, exist_ok=True)
            (candidate / "candidate.json").write_text(json.dumps({
                "schema_version": 1,
                "candidate_id": f"{self.lane}-candidate",
                "source_lane": self.lane,
                "submission_mode": "csv",
                "accelerator": "cpu",
                "purpose": "controller integration test",
            }), encoding="utf-8")
            (candidate / "out" / "submission.csv").write_text(
                "id,pred\na,0.1\nb,0.2\n", encoding="utf-8"
            )
            (candidate / "READY").touch()
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace = trace_dir / f"{agent_id}.jsonl"
            trace.write_text("{}\n", encoding="utf-8")
            return AgentResult(
                agent_id=agent_id, role=role, workdir=workdir,
                trace_path=trace, exit_code=0, timed_out=False,
                duration_s=0.01, session_id=f"{self.lane}-session",
            )

        async def terminate_all(self, **kwargs):
            return None

    class FakeSelectionRunner:
        async def run(
            self, *, agent_id, role, prompt, workdir, trace_dir, timeout_s,
            persist_session=False, resume_session_id=None,
        ):
            del timeout_s, persist_session, resume_session_id
            context = json.loads(prompt.split("# 本次不可变输入\n\n", 1)[1])
            candidate_id = context["candidates"][0]["candidate_id"]
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace = trace_dir / f"{agent_id}.jsonl"
            trace.write_text("{}\n", encoding="utf-8")
            return AgentResult(
                agent_id=agent_id, role=role, workdir=workdir,
                trace_path=trace, exit_code=0, timed_out=False,
                duration_s=0.01, resolved_model="claude-fable-5",
                result_text=json.dumps({
                    "decision": "submit", "candidate_id": candidate_id,
                    "reason": "integration test", "evidence": [],
                    "confidence": 0.9, "risk_flags": [],
                }),
            )

        async def terminate_all(self, **kwargs):
            return None

    controller = FinalController(
        config=config, slug="test-slug", assets_dir=assets,
        duration_minutes=0.03, competition_mode="practice",
        kaggle_user="", live=False, run_id="integration",
    )
    monkeypatch.setattr(
        controller, "_codex_runner", lambda **_kwargs: FakeRunner("codex")
    )

    def fake_claude_runner(_profile, **kwargs):
        return (
            FakeSelectionRunner()
            if kwargs.get("allowed_tools") == "Read" else FakeRunner("claude")
        )

    monkeypatch.setattr(
        controller, "_claude_runner", fake_claude_runner
    )
    session = asyncio.run(controller.run())
    status = read_json(session / "RUN_STATUS.json")
    broker_status = read_json(session / "control" / "BROKER_STATUS.json")
    assert status["status"] == "complete"
    assert status["candidate_count"] == 2
    assert broker_status["consumed"] == 1
    assert broker_status["submission_authority"] == "final_system.SubmissionBroker"
    assert status["selection_manager"]["status"] == "complete"
    assert status["selection_manager"]["model"] == "claude-fable-5"
