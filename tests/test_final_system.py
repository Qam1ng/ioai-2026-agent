from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from final_system.broker import SubmissionBroker
from final_system.config import SystemConfig
from final_system.controller import FinalController
from final_system.evaluation import freeze_contract, verify_contract
from final_system.io import atomic_json, read_json
from final_system.kaggle import DryRunAdapter, KaggleAdapter, SubmitResult
from final_system.registry import CandidateRegistry
from final_system.runners import OpenRouterCodexRunner
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
    target = tmp_path / "evaluation"
    target.mkdir()
    contract = freeze_contract(source, target, direction="maximize")
    assert verify_contract(target) == contract


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
    adapter = KaggleAdapter(
        slug="slug", root=tmp_path, kaggle_user="user", submission_mode="auto"
    )
    assert adapter.remaining_today() == 43


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
        def submit(self, record, submission_id, submission_class):
            return SubmitResult(False, "ambiguous", "connection closed")

    broker = SubmissionBroker(
        tmp_path / "control", registry=registry,
        adapter=AmbiguousAdapter("slug", tmp_path), max_submissions=2,
        final_reserve=0, initial_calibrations=1, final_start_fraction=0.8,
        anti_monopoly_fraction=0.5, min_local_gain=0.0,
        direction="maximize",
    )
    assert broker.tick(fraction_elapsed=0.1) is not None
    assert broker.remaining() == 2
    assert broker.reservable() == 1


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
                "submission_mode": "kernel",
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
                trace_path=trace, exit_code=0, session_id=f"{self.lane}-session",
            )

        async def terminate_all(self, **kwargs):
            return None

    controller = FinalController(
        config=config, slug="test-slug", assets_dir=assets,
        duration_minutes=0.03, competition_mode="practice",
        kaggle_user="", live=False, run_id="integration",
    )
    monkeypatch.setattr(controller, "_codex_runner", lambda: FakeRunner("codex"))
    monkeypatch.setattr(
        controller, "_claude_runner", lambda profile: FakeRunner("claude")
    )
    session = asyncio.run(controller.run())
    status = read_json(session / "RUN_STATUS.json")
    broker_status = read_json(session / "control" / "BROKER_STATUS.json")
    assert status["status"] == "complete"
    assert status["candidate_count"] == 2
    assert broker_status["consumed"] == 1
    assert broker_status["submission_authority"] == "final_system.SubmissionBroker"
