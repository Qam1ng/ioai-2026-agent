"""Tests for the auto-generated technical report and the operator summary.

Offline by construction: a synthetic blackboard is built on ``tmp_path`` with the
real ``Blackboard`` / ``PodBudget`` / ``QuotaPool`` classes, so the report is
rendered from exactly the artefacts a competition run leaves behind.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swarm.budget import PodBudget, QuotaPool  # noqa: E402
from swarm.bus import Blackboard  # noqa: E402
from swarm.config import ROLES, SwarmConfig  # noqa: E402
from swarm.ledger import refresh_calibration  # noqa: E402
from swarm.report import (  # noqa: E402
    collect_declaration,
    detect_libraries,
    read_events,
    render_report,
    render_run_summary,
    roles_observed,
    tool_usage,
    write_report,
)
from swarm.schemas import (  # noqa: E402
    CandidateState,
    Constraint,
    Experiment,
    PlanCard,
    ProbeResult,
    SubmissionRecord,
    TaskCard,
)

SLUG = "ioai-2026-practice-task-1"


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def cfg() -> SwarmConfig:
    c = SwarmConfig(slug=SLUG)
    c.default_model.backend = "claude"
    c.default_model.model = "claude-opus-5"
    return c


@pytest.fixture()
def quota(tmp_path: Path) -> QuotaPool:
    q = QuotaPool(tmp_path / "quota.json", limit_hours=30.0)
    q.settle("sub-a", 1800.0)  # half a GPU-hour actually consumed
    return q


@pytest.fixture()
def budget(tmp_path: Path, quota: QuotaPool) -> PodBudget:
    b = PodBudget(tmp_path / "budget.json", deadline_s=6 * 3600.0, quota=quota)
    b.note_llm({"input_tokens": 12000, "output_tokens": 3000}, 0.42)
    b.note_llm({"input_tokens": 8000, "output_tokens": 1000}, 0.18)
    b.note_submit()
    b.note_submit()
    return b


@pytest.fixture()
def bb(tmp_path: Path) -> Blackboard:
    """A blackboard shaped like a real mid-run pod."""
    board = Blackboard(tmp_path / "workspace" / SLUG)

    board.put_task_card(
        TaskCard(
            slug=SLUG,
            title="Audio class-incremental transfer",
            task_type="supervised",
            metric_name="balanced accuracy",
            metric_description="mean per-class recall over 20 classes",
            metric_direction="maximize",
            submission_format="submission.csv with id,label",
            data_summary="1200 train clips, 400 test clips, 20 classes",
            constraints=[
                Constraint(kind="must", quote="The notebook must run in under 30 minutes."),
                Constraint(kind="must_not", quote="No pretrained checkpoints may be submitted."),
            ],
            grouping_variable="speaker_id",
            risks=["tiny classes (3 samples) make single-holdout validation unstable"],
            routing_confidence=0.8,
        )
    )

    board.add_plan(
        PlanCard(
            title="frozen embeddings + linear probe",
            family="linear",
            architecture="AST embeddings -> logistic regression",
            validation_strategy="5-fold grouped by speaker_id",
            accelerator="cpu",
        )
    )
    board.add_plan(
        PlanCard(
            title="gradient boosting on embeddings",
            family="trees",
            architecture="AST embeddings -> LightGBM",
            validation_strategy="5-fold grouped by speaker_id",
            accelerator="cpu",
        )
    )

    board.put_candidate(
        CandidateState(
            candidate_id="cand-linear",
            family="linear",
            status="ready",
            local_score=0.884,
            local_std=0.021,
            folds=[0.87, 0.90, 0.88, 0.89, 0.88],
            kernel_dir="candidates/cand-linear/kernel",
        )
    )
    board.put_candidate(
        CandidateState(
            candidate_id="cand-trees",
            family="trees",
            status="retired",
            local_score=0.831,
            local_std=0.030,
            retire_reason="never beat the linear probe and cost 4x the runtime",
        )
    )

    for i, (score, std, accepted, reason) in enumerate(
        [
            (0.802, 0.024, True, ""),
            (0.845, 0.022, True, ""),
            (0.848, 0.023, False, "rejected: local gain +0.0030 is inside the noise band"),
            (0.884, 0.021, True, ""),
        ]
    ):
        board.add_experiment(
            Experiment(
                candidate_id="cand-linear",
                role="tuner",
                description=f"linear probe sweep step {i}",
                local_score=score,
                local_std=std,
                folds=[score - 0.01, score + 0.01, score, score + 0.005, score - 0.005],
                runtime_s=180.0,
                accepted=accepted,
                reject_reason=reason,
            )
        )

    board.add_submission(
        SubmissionRecord(
            sub_id="sub-floor",
            candidate_id="cand-linear",
            lane="floor",
            purpose="insurance: establish a valid submission early",
            local_score=0.802,
            lb_score=0.770,
            status="scored",
            accelerator="cpu",
        )
    )
    board.add_submission(
        SubmissionRecord(
            sub_id="sub-milestone",
            candidate_id="cand-linear",
            lane="milestone",
            purpose="first significant local gain",
            local_score=0.884,
            lb_score=0.845,
            status="scored",
            accelerator="p100",
            gpu_seconds=1800.0,
        )
    )
    board.add_submission(
        SubmissionRecord(
            sub_id="sub-pending",
            candidate_id="cand-linear",
            lane="milestone",
            purpose="awaiting score",
            local_score=0.889,
            lb_score=None,
            status="running",
        )
    )

    board.add_probe(
        ProbeResult(
            probe_kind="constant",
            hypothesis="what does the all-majority-class baseline score?",
            lb_score=0.120,
            inference="the hidden test set is class-balanced like train",
        )
    )
    board.add_probe(
        ProbeResult(
            probe_kind="noise_floor",
            hypothesis="two equivalent kernels, how far apart do they score?",
            lb_score=0.770,
            numeric={"noise_floor": 0.004},
            inference="leaderboard noise floor is about 0.004",
        )
    )

    # A realistic slice of the audit log: roles that ran and tools they called.
    for role in ("profiler", "designer", "coder", "tuner", "verifier", "prober"):
        board.event("role_start", role=role, instruction="synthetic")
    for tool, n in (
        ("run_python", 12),
        ("run_bash", 3),
        ("write_file", 5),
        ("kaggle_download", 1),
        ("kaggle_push_kernel", 2),
        ("kaggle_submit", 2),
        ("memory_recall", 2),
        ("skill_load", 1),
    ):
        for _ in range(n):
            board.event("tool", role="coder", tool=tool, chars=10)

    # Code the agents "wrote", so the library declaration has evidence to read.
    kernel = board.candidate_dir("cand-linear") / "kernel"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "solution.py").write_text(
        "import json\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "print(np.zeros(3), pd, LogisticRegression, json)\n"
    )

    refresh_calibration(board)
    return board


# --------------------------------------------------------------------------- #
# evidence gathering
# --------------------------------------------------------------------------- #
def test_read_events_and_tool_usage(bb: Blackboard):
    events = read_events(bb)
    assert events, "the audit log must be readable"
    tools = tool_usage(events)
    assert tools["run_python"] == 12
    assert tools["kaggle_submit"] == 2
    roles = roles_observed(events)
    assert roles["profiler"] == 1
    assert set(roles) == {"profiler", "designer", "coder", "tuner", "verifier", "prober"}


def test_read_events_survives_a_torn_line(bb: Blackboard):
    with open(bb.events_path, "a") as fh:
        fh.write('{"kind": "tool", "tool": "run_py')  # crash mid-write
    events = read_events(bb)
    assert tool_usage(events)["run_python"] == 12


def test_detect_libraries_finds_solution_imports_not_stdlib(bb: Blackboard):
    libs = detect_libraries(bb)
    assert "numpy" in libs and "pandas" in libs and "sklearn" in libs
    assert "json" not in libs, "stdlib modules are not a declarable dependency"


# --------------------------------------------------------------------------- #
# declaration
# --------------------------------------------------------------------------- #
def test_collect_declaration_reports_multi_agent_usage_truthfully(
    bb: Blackboard, budget: PodBudget, quota: QuotaPool, cfg: SwarmConfig
):
    dec = collect_declaration(bb, budget, quota, cfg)
    ma = dec["multiple_agents"]

    assert ma["used"] is True
    assert ma["n_roles_configured"] == len(ROLES)
    assert set(ma["roles_observed"]) == {
        "profiler", "designer", "coder", "tuner", "verifier", "prober",
    }
    assert ma["max_parallel_candidates"] == cfg.parallel.max_candidates
    assert ma["n_candidates_created"] == 2
    assert "parallel" in ma["detail"].lower()


def test_collect_declaration_admits_human_written_scaffolding(
    bb: Blackboard, budget: PodBudget, quota: QuotaPool, cfg: SwarmConfig
):
    dec = collect_declaration(bb, budget, quota, cfg)
    hw = dec["human_written_prompts_or_scaffolding"]
    assert hw["used"] is True
    assert "prompts/swarm" in hw["detail"]
    assert "no human in the loop" in hw["detail"].lower()


def test_collect_declaration_flags_internet_code_and_retrieval_with_evidence(
    bb: Blackboard, budget: PodBudget, quota: QuotaPool, cfg: SwarmConfig
):
    dec = collect_declaration(bb, budget, quota, cfg)

    assert dec["internet_access"]["used"] is True
    assert dec["internet_access"]["evidence"]["network_tool_calls"]["kaggle_submit"] == 2
    assert dec["internet_access"]["evidence"]["llm_api_calls"] == 2
    assert "internet disabled" in dec["internet_access"]["detail"]

    assert dec["code_execution"]["used"] is True
    assert dec["code_execution"]["evidence"]["tool_calls"]["run_python"] == 12

    assert dec["retrieval_or_search"]["used"] is True
    assert dec["retrieval_or_search"]["evidence"]["tool_calls"]["memory_recall"] == 2


def test_collect_declaration_efficiency_matches_the_budget_and_quota(
    bb: Blackboard, budget: PodBudget, quota: QuotaPool, cfg: SwarmConfig
):
    eff = collect_declaration(bb, budget, quota, cfg)["efficiency"]
    assert eff["tokens_in"] == 20000
    assert eff["tokens_out"] == 4000
    assert eff["llm_calls"] == 2
    assert eff["cost_usd"] == pytest.approx(0.60, abs=1e-6)
    assert eff["n_submissions"] == 3
    assert eff["n_experiments"] == 4
    assert eff["gpu_hours"] == pytest.approx(0.5)
    assert eff["wall_clock_minutes"] is not None


def test_collect_declaration_on_an_empty_run_claims_nothing(tmp_path: Path, cfg: SwarmConfig):
    empty = Blackboard(tmp_path / "empty")
    dec = collect_declaration(empty, None, None, cfg)

    assert dec["internet_access"]["used"] is False
    assert dec["code_execution"]["used"] is False
    assert dec["retrieval_or_search"]["used"] is False
    assert dec["libraries"]["detected"] == []
    assert dec["efficiency"]["gpu_hours"] is None
    assert dec["multiple_agents"]["roles_observed"] == []
    # the honest constants stay true even with zero evidence
    assert dec["human_written_prompts_or_scaffolding"]["used"] is True
    assert dec["multiple_agents"]["used"] is True


# --------------------------------------------------------------------------- #
# render_report
# --------------------------------------------------------------------------- #
def test_render_report_contains_every_required_section(
    bb: Blackboard, budget: PodBudget, quota: QuotaPool, cfg: SwarmConfig
):
    text = render_report(bb, budget, quota, cfg, bb.get_task_card())

    for heading in (
        "# Technical report",
        "## 1. Problem",
        "## 2. Approach actually taken",
        "## 3. Experiment ledger",
        "## 4. Score trajectory",
        "## 5. What was tried and rejected",
        "## 6. Final selection",
        "## Declaration of tools and resources",
        "## Efficiency metrics",
    ):
        assert heading in text, f"missing report section: {heading}"


def test_render_report_declares_the_required_fields(
    bb: Blackboard, budget: PodBudget, quota: QuotaPool, cfg: SwarmConfig
):
    text = render_report(bb, budget, quota, cfg, bb.get_task_card())

    for field in (
        "Internet access",
        "Code execution",
        "Retrieval / search",
        "Multiple agents",
        "Private tools",
        "Human-written prompts / scaffolding",
        "Libraries",
    ):
        assert field in text, f"declaration is missing: {field}"

    # efficiency metrics the rules ask to be published
    for metric in (
        "Wall-clock elapsed (min)",
        "LLM calls",
        "Input tokens",
        "Output tokens",
        "LLM cost (USD)",
        "Kaggle submissions used",
        "GPU-hours consumed",
    ):
        assert metric in text, f"efficiency metric is missing: {metric}"

    assert "numpy" in text and "sklearn" in text  # detected libraries are declared
    assert "yes" in text  # at least one declaration answered affirmatively


def test_render_report_has_no_none_looking_artifacts(
    bb: Blackboard, budget: PodBudget, quota: QuotaPool, cfg: SwarmConfig
):
    text = render_report(bb, budget, quota, cfg, bb.get_task_card())
    assert "None" not in text
    assert "null" not in text
    assert "nan" not in text
    # the pending submission has no leaderboard score, and that must READ as unknown
    assert "unknown" in text


def test_render_report_on_an_empty_workspace_still_renders(tmp_path: Path, cfg: SwarmConfig):
    """A crash before any work must still produce a submittable, honest report."""
    empty = Blackboard(tmp_path / "empty")
    text = render_report(empty, None, None, cfg, None)

    assert "# Technical report" in text
    assert "unknown" in text
    assert "None" not in text
    assert "No experiments recorded" in text
    assert "No submissions were made" in text


def test_render_report_quotes_real_numbers_from_the_blackboard(
    bb: Blackboard, budget: PodBudget, quota: QuotaPool, cfg: SwarmConfig
):
    text = render_report(bb, budget, quota, cfg, bb.get_task_card())

    assert "0.8840" in text  # best local CV
    assert "0.8450" in text  # best leaderboard score
    assert "20000" in text and "4000" in text  # token counts
    assert SLUG in text
    assert "balanced accuracy" in text
    # rejected work is reported, not quietly dropped
    assert "inside the noise band" in text
    assert "never beat the linear probe" in text
    assert "leaderboard noise floor is about 0.004" in text


def test_render_report_calls_out_the_calibration_gap(
    bb: Blackboard, budget: PodBudget, quota: QuotaPool, cfg: SwarmConfig
):
    text = render_report(bb, budget, quota, cfg, bb.get_task_card())
    assert "gap (mean local − mean leaderboard)" in text
    cal = bb.get_calibration()
    assert cal.n_points == 2
    assert f"{cal.gap:.4f}" in text
    assert "noise band" in text.lower()


def test_render_report_shows_the_winners_curse_adjustment(
    bb: Blackboard, budget: PodBudget, quota: QuotaPool, cfg: SwarmConfig
):
    text = render_report(bb, budget, quota, cfg, bb.get_task_card())
    assert "Shrunk estimate" in text
    assert "raw leaderboard argmax" in text
    assert "different* private" in text or "different" in text


def test_render_report_falls_back_to_the_stored_task_card(
    bb: Blackboard, budget: PodBudget, quota: QuotaPool, cfg: SwarmConfig
):
    """The caller may omit the task card; the blackboard's copy must be used."""
    text = render_report(bb, budget, quota, cfg)
    assert "Audio class-incremental transfer" in text
    assert "speaker_id" in text


def test_render_report_is_fast_enough_for_the_30_minute_deadline(
    bb: Blackboard, budget: PodBudget, quota: QuotaPool, cfg: SwarmConfig
):
    import time

    t0 = time.time()
    for _ in range(5):
        render_report(bb, budget, quota, cfg, bb.get_task_card())
    assert (time.time() - t0) < 5.0


# --------------------------------------------------------------------------- #
# write_report / run summary
# --------------------------------------------------------------------------- #
def test_write_report_writes_markdown_into_the_workspace(
    bb: Blackboard, budget: PodBudget, quota: QuotaPool, cfg: SwarmConfig
):
    path = write_report(bb, budget, quota, cfg, bb.get_task_card())

    assert path == bb.ws / "REPORT.md"
    assert path.exists()
    text = path.read_text()
    assert text.startswith("# Technical report")
    assert "## Declaration of tools and resources" in text
    # writing the report is itself an auditable event
    assert any(e.get("kind") == "report_written" for e in read_events(bb))


def test_write_report_overwrites_cleanly_on_a_second_call(
    bb: Blackboard, budget: PodBudget, quota: QuotaPool, cfg: SwarmConfig
):
    first = write_report(bb, budget, quota, cfg, bb.get_task_card())
    size_first = first.stat().st_size
    bb.add_experiment(Experiment(description="late idea", local_score=0.9, role="tuner"))
    second = write_report(bb, budget, quota, cfg, bb.get_task_card())

    assert second == first
    assert "late idea" in second.read_text()
    assert second.stat().st_size != size_first or True  # size may coincide; content is the test


def test_render_run_summary_is_a_short_operator_block(bb: Blackboard, budget: PodBudget):
    summary = render_run_summary(bb, budget)

    assert "[budget]" in summary
    assert "[state]" in summary
    assert "[scores]" in summary
    assert "[calibration]" in summary
    assert "cand-linear" in summary
    assert "0.8840" in summary
    assert "None" not in summary
    assert len(summary.splitlines()) <= 12, "the operator block must stay glanceable"


def test_render_run_summary_works_without_a_budget(bb: Blackboard):
    summary = render_run_summary(bb)
    assert "[state]" in summary
    assert "[budget]" not in summary


def test_render_run_summary_on_an_empty_workspace(tmp_path: Path):
    summary = render_run_summary(Blackboard(tmp_path / "empty"))
    assert "[state]" in summary
    assert "unknown" in summary
    assert "None" not in summary
