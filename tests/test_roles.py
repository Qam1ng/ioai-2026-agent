"""Tests for the specialist roles: prompts, allowlists, and parse behaviour.

No network and no model calls. Roles are built with ``object.__new__`` so the
provider is never constructed, and ``run`` is replaced with a stub returning a
canned :class:`RoleResult` — the thing under test is the parsing and gating
logic that sits between the model's JSON and the pod's state.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.tools import registry as toolreg  # noqa: E402
from swarm.budget import PodBudget  # noqa: E402
from swarm.bus import Blackboard  # noqa: E402
from swarm.config import ROLES, SwarmConfig  # noqa: E402
from swarm.roles import ROLE_CLASSES, get_role_class  # noqa: E402
from swarm.roles.base import PROMPT_DIR, RoleResult, load_prompt  # noqa: E402
from swarm.roles.designer import DesignerRole  # noqa: E402
from swarm.roles.manager import ManagerRole  # noqa: E402
from swarm.roles.profiler import ProfilerRole  # noqa: E402
from swarm.roles.verifier import VerifierRole  # noqa: E402
from swarm.schemas import VALID_ACTIONS, CandidateState, ManagerAction, PlanCard  # noqa: E402

ALL_ROLES = sorted(ROLE_CLASSES)

#: Tools no specialist may hold: spending a submission or GPU quota is the
#: broker's and the runner's decision, never a role's.
BROKER_ONLY_TOOLS = {"kaggle_submit", "kaggle_push_kernel"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_role(cls, tmp_path: Path, report: dict, *, ok: bool = True, text: str = ""):
    """Build a role without its provider and stub out its model loop."""
    role = object.__new__(cls)
    role.name = cls.role_name
    role.cfg = SwarmConfig(slug="test-slug", workspace=str(tmp_path))
    role.bb = Blackboard(tmp_path)
    role.budget = PodBudget(tmp_path / "budget.json")
    role.ctx = None
    role.on_event = None
    role.provider = None
    role.spec = role.cfg.model_for(cls.role_name)

    def _stub_run(instructions: str = "", context: dict | None = None, **_: object) -> RoleResult:
        return RoleResult(role.name, ok, text, report, 1, 0.01, "" if ok else "stubbed failure")

    role.run = _stub_run  # type: ignore[method-assign]
    return role


def read_events(tmp_path: Path) -> list[str]:
    p = tmp_path / "events.jsonl"
    return p.read_text().splitlines() if p.exists() else []


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("role_name", ALL_ROLES)
def test_prompt_file_exists(role_name: str) -> None:
    cls = ROLE_CLASSES[role_name]
    assert cls.prompt_name, f"{cls.__name__} has an empty prompt_name"
    assert (PROMPT_DIR / f"{cls.prompt_name}.md").exists(), (
        f"{cls.__name__}.prompt_name points at a missing prompt"
    )


@pytest.mark.parametrize("role_name", ALL_ROLES)
def test_prompt_formats_without_crashing(role_name: str) -> None:
    """Catches unescaped braces: a prompt that breaks .format() is a dead role."""
    cls = ROLE_CLASSES[role_name]
    text = load_prompt(cls.prompt_name)
    out = text.format(slug="titanic", task_type="supervised", metric="macro F1")
    assert "titanic" in out
    assert len(out) > 400, "prompt is suspiciously short"


@pytest.mark.parametrize("role_name", ALL_ROLES)
def test_prompt_states_boundaries_and_output_schema(role_name: str) -> None:
    text = load_prompt(ROLE_CLASSES[role_name].prompt_name).lower()
    assert "must not" in text, "prompt does not state role boundaries"
    assert "```json" in text, "prompt does not show an exact JSON output schema"


def test_manager_prompt_lists_every_valid_action() -> None:
    text = load_prompt("manager")
    for action in VALID_ACTIONS:
        assert action in text, f"manager prompt never mentions the {action!r} action"


def test_verifier_prompt_cites_the_cv_optimism_lesson() -> None:
    """The Verifier exists because of this measured failure; the numbers must be in it."""
    text = load_prompt("verifier")
    assert "0.9156" in text and "0.78095" in text


def test_prober_prompt_states_the_winner_s_curse() -> None:
    text = load_prompt("prober").lower()
    assert "winner's curse" in text
    assert "noise floor" in text


def test_every_prompt_on_disk_survives_formatting() -> None:
    """Guards the whole prompt dir, including prompts added after this test."""
    on_disk = {p.stem for p in PROMPT_DIR.glob("*.md")}
    registered = {cls.prompt_name for cls in ROLE_CLASSES.values()}
    assert registered <= on_disk, f"missing prompts: {sorted(registered - on_disk)}"
    for path in sorted(PROMPT_DIR.glob("*.md")):
        path.read_text().format(slug="s", task_type="supervised", metric="rmse")


# --------------------------------------------------------------------------- #
# registry and allowlists
# --------------------------------------------------------------------------- #
def test_registry_covers_every_configured_role() -> None:
    assert ALL_ROLES == sorted(ROLES)
    assert len(ROLE_CLASSES) == 10


def test_registry_keys_match_role_names() -> None:
    for name, cls in ROLE_CLASSES.items():
        assert cls.role_name == name, f"{cls.__name__}.role_name != registry key"


def test_get_role_class_raises_on_unknown_role() -> None:
    with pytest.raises(KeyError):
        get_role_class("optimiser")


@pytest.mark.parametrize("role_name", ALL_ROLES)
def test_tool_allowlist_is_real_and_least_privilege(role_name: str) -> None:
    cls = ROLE_CLASSES[role_name]
    assert cls.tools, f"{cls.__name__} has no tool allowlist (None means every tool)"
    unknown = sorted(set(cls.tools) - set(toolreg.FNS))
    assert not unknown, f"{cls.__name__} allows tools that do not exist: {unknown}"
    assert not set(cls.tools) & BROKER_ONLY_TOOLS, (
        f"{cls.__name__} may not push kernels or submit"
    )
    assert cls.max_steps > 0


@pytest.mark.parametrize("role_name", ["manager", "compliance", "designer"])
def test_read_only_roles_cannot_execute_or_write(role_name: str) -> None:
    """A gate or a router that can execute is a gate that can change the state it judges."""
    tools = set(ROLE_CLASSES[role_name].tools or ())
    assert not tools & {"run_python", "run_bash", "write_file", "memory_write"}


def test_coder_has_the_tools_it_needs() -> None:
    tools = set(ROLE_CLASSES["coder"].tools or ())
    assert {"run_python", "run_bash", "write_file", "read_file", "list_dir"} <= tools


# --------------------------------------------------------------------------- #
# manager parsing
# --------------------------------------------------------------------------- #
def test_manager_parses_a_valid_action() -> None:
    action = ManagerRole.parse_action(
        {"action": "code", "target": "cand-1", "reason": "impl broken", "instructions": "fix it"}
    )
    assert isinstance(action, ManagerAction)
    assert (action.action, action.target, action.instructions) == ("code", "cand-1", "fix it")


@pytest.mark.parametrize(
    "report",
    [
        {},
        {"reason": "no action key"},
        {"action": "ensemble_everything"},
        {"action": ""},
        {"action": None},
        {"action": 7},
        "not a dict",
    ],
)
def test_manager_degrades_invalid_reports_to_wait(report: object) -> None:
    action = ManagerRole.parse_action(report)  # type: ignore[arg-type]
    assert action.action == "wait"
    assert action.reason, "a degraded action must say why"


def test_manager_normalises_case_and_whitespace() -> None:
    assert ManagerRole.parse_action({"action": "  VERIFY "}).action == "verify"


def test_manager_freeze_gate_blocks_new_directions() -> None:
    for act in ("design", "code"):
        out = ManagerRole.enforce_gates(
            ManagerAction(action=act, reason="promising idea"),
            has_floor_submission=True,
            frozen=True,
        )
        assert out.action == "wait"
        assert "freeze gate" in out.reason
    kept = ManagerRole.enforce_gates(
        ManagerAction(action="aggregate"), has_floor_submission=True, frozen=True
    )
    assert kept.action == "aggregate"


def test_manager_floor_gate_blocks_probing_before_a_floor_submission() -> None:
    out = ManagerRole.enforce_gates(
        ManagerAction(action="probe", reason="curious about base rate"),
        has_floor_submission=False,
        frozen=False,
    )
    assert out.action == "wait"
    assert "floor gate" in out.reason


def test_manager_decide_end_to_end(tmp_path: Path) -> None:
    role = make_role(ManagerRole, tmp_path, {"action": "verify", "target": "cand-2",
                                             "reason": "score unverified"})
    action = role.decide({"submissions_by_lane": {"floor": 1}})
    assert action.action == "verify" and action.target == "cand-2"
    assert any("manager_action" in line for line in read_events(tmp_path))


def test_manager_decide_degrades_when_the_step_fails(tmp_path: Path) -> None:
    role = make_role(ManagerRole, tmp_path, {}, ok=False)
    action = role.decide({"submissions_by_lane": {"floor": 1}})
    assert action.action == "wait"


# --------------------------------------------------------------------------- #
# designer diversity
# --------------------------------------------------------------------------- #
def _plan(family: str, title: str = "p") -> PlanCard:
    return PlanCard(title=title, family=family)


def test_designer_flags_duplicate_families() -> None:
    plans = [_plan("gbdt"), _plan("GBDT"), _plan("nn_mlp"), _plan("gbdt")]
    assert DesignerRole.duplicate_families(plans) == ["gbdt"]


def test_designer_reports_no_duplicates_when_families_differ() -> None:
    plans = [_plan("gbdt"), _plan("linear"), _plan("nn_mlp")]
    assert DesignerRole.duplicate_families(plans) == []


def test_designer_enforce_diversity_keeps_the_first_of_each_family() -> None:
    plans = [_plan("gbdt", "floor"), _plan("gbdt", "second"), _plan("linear", "third")]
    kept = DesignerRole.enforce_diversity(plans)
    assert [p.title for p in kept] == ["floor", "third"]


def test_designer_parse_plans_normalises_and_drops_junk() -> None:
    plans = DesignerRole.parse_plans(
        {
            "plans": [
                {"title": "a", "family": " GBDT ", "accelerator": "TPU",
                 "expected_runtime_min": "12.5", "plan_id": "attacker-supplied"},
                "not a dict",
                {"notes": "no family or title"},
            ]
        }
    )
    assert len(plans) == 1
    assert plans[0].family == "gbdt"
    assert plans[0].accelerator == "cpu"  # unknown accelerator falls back
    assert plans[0].expected_runtime_min == 12.5
    assert plans[0].plan_id != "attacker-supplied"  # ids are minted locally


def test_designer_design_drops_duplicates_and_records_the_violation(tmp_path: Path) -> None:
    report = {
        "plans": [
            {"title": "floor", "family": "linear"},
            {"title": "dupe", "family": "linear"},
            {"title": "trees", "family": "gbdt"},
        ]
    }
    role = make_role(DesignerRole, tmp_path, report)
    plans = role.design(k=3)
    assert [p.family for p in plans] == ["linear", "gbdt"]
    events = "\n".join(read_events(tmp_path))
    assert "plan_diversity_violation" in events
    assert "plan_shortfall" in events
    assert len(role.bb.get_plans()) == 2


# --------------------------------------------------------------------------- #
# profiler / verifier / compliance / aggregator / prober / reporter parsing
# --------------------------------------------------------------------------- #
def test_profiler_parse_card_validates_and_normalises() -> None:
    card = ProfilerRole.parse_card(
        {
            "title": "T",
            "task_type": "wild_guess",
            "metric_name": "macro F1",
            "metric_direction": "up",
            "submission_format": "id,label",
            "constraints": [
                {"kind": "must_not", "quote": "No external data.", "rationale": "rules"},
                {"kind": "must_not", "quote": "   "},
            ],
            "routing_confidence": 5.0,
            "unknown_field": "ignored",
        },
        slug="test-slug",
    )
    assert card.slug == "test-slug"
    assert card.task_type == "unknown"
    assert card.metric_direction == "maximize"
    assert len(card.constraints) == 1  # the unquoted constraint is unenforceable
    assert 0.0 <= card.routing_confidence <= 1.0


@pytest.mark.parametrize(
    "report",
    [{}, {"metric_name": "f1"}, {"submission_format": "id,label"}],
)
def test_profiler_rejects_an_unusable_card(report: dict) -> None:
    with pytest.raises(ValueError):
        ProfilerRole.parse_card(report, slug="test-slug")


def test_verifier_downgrades_a_pass_that_lists_problems() -> None:
    out = VerifierRole.parse_verification(
        {
            "local_score": 0.8,
            "folds": [0.79, 0.81, 0.8, 0.8, 0.8],
            "metric_selftest": {"passed": True},
            "format_valid": True,
            "problems": ["preprocessing fitted before the split"],
            "verdict": "pass",
        }
    )
    assert out["verdict"] == "suspect"


def test_verifier_flags_a_single_holdout_and_computes_std() -> None:
    out = VerifierRole.parse_verification(
        {"folds": [0.9156], "metric_selftest": {"passed": True},
         "format_valid": True, "verdict": "pass"}
    )
    assert out["local_score"] == pytest.approx(0.9156)
    assert out["verdict"] != "pass"
    assert any("fold" in p for p in out["problems"])


def test_verifier_fails_closed_on_an_empty_report() -> None:
    out = VerifierRole.parse_verification({})
    assert out["verdict"] == "fail"
    assert out["problems"]


def test_verifier_verify_writes_state_and_ledger(tmp_path: Path) -> None:
    role = make_role(
        VerifierRole,
        tmp_path,
        {
            "local_score": 0.81,
            "local_std": 0.01,
            "folds": [0.80, 0.81, 0.82, 0.81, 0.81],
            "fold_scheme": "GroupKFold(5) on speaker_id",
            "metric_selftest": {"passed": True},
            "format_valid": True,
            "problems": [],
            "verdict": "pass",
        },
    )
    cand = CandidateState(candidate_id="cand-x", family="gbdt")
    role.bb.put_candidate(cand)
    out = role.verify(cand)
    assert out["verdict"] == "pass"
    assert set(out) >= {"local_score", "local_std", "folds", "problems", "verdict"}
    assert role.bb.get_candidate("cand-x").local_score == pytest.approx(0.81)
    assert len(role.bb.get_experiments()) == 1


def test_compliance_fails_closed_and_requires_quotes(tmp_path: Path) -> None:
    cls = ROLE_CLASSES["compliance"]
    empty = cls.parse_check({})
    assert empty["pass"] is False and empty["violations"]

    quoted = cls.parse_check(
        {"pass": True, "violations": [{"quote": "No external data.", "why": "uses ImageNet"}]}
    )
    assert quoted["pass"] is False  # a listed violation overrides a claimed pass

    paraphrased = cls.parse_check({"pass": True, "violations": [{"why": "feels risky"}]})
    assert paraphrased["pass"] is True  # unquoted findings are notes, not violations
    assert "feels risky" in paraphrased["notes"]


def test_compliance_check_returns_the_contract(tmp_path: Path) -> None:
    role = make_role(ROLE_CLASSES["compliance"], tmp_path, {"pass": True, "violations": []})
    out = role.check(PlanCard(title="floor", family="linear"))
    assert out["pass"] is True
    assert out["violations"] == []


def test_aggregator_normalises_weights_and_flags_overrun() -> None:
    cls = ROLE_CLASSES["aggregator"]
    out = cls.parse_blend(
        {
            "strategy": "rank_blend",
            "members": [{"candidate_id": "a"}, {"candidate_id": "b"}],
            "code": "print('kernel')",
            "expected_runtime_min": 90.0,
        },
        runtime_budget_min=30.0,
    )
    assert out["strategy"] == "rank_blend"
    assert [m["weight"] for m in out["members"]] == [0.5, 0.5]
    assert any("exceeds the kernel budget" in p for p in out["problems"])
    assert out["ok"] is False


def test_aggregator_requires_members_and_code() -> None:
    out = ROLE_CLASSES["aggregator"].parse_blend({"strategy": "prob_average"})
    assert out["ok"] is False
    assert {"strategy", "members", "code"} <= set(out)


def test_prober_forces_cpu_and_requires_a_decision_rule(tmp_path: Path) -> None:
    role = make_role(
        ROLE_CLASSES["prober"],
        tmp_path,
        {
            "probes": [
                {"probe_kind": "noise_floor", "hypothesis": "h", "decision_rule": "d",
                 "priority": 2, "accelerator": "t4"},
                {"probe_kind": "constant", "hypothesis": "base rate", "decision_rule": "d",
                 "priority": 1},
                {"probe_kind": "constant", "hypothesis": "no decision rule"},
            ]
        },
    )
    probes = role.plan_probes(n=5)
    assert [p["probe_kind"] for p in probes] == ["constant", "noise_floor"]  # priority order
    assert all(p["accelerator"] == "cpu" for p in probes)  # GPU quota is for candidates
    assert all(p["lane"] == "probe" for p in probes)


def test_tuner_rejects_a_gain_inside_the_noise_band() -> None:
    cls = ROLE_CLASSES["tuner"]
    assert cls.is_significant(0.805, 0.800, 0.02) is False
    assert cls.is_significant(0.850, 0.800, 0.02) is True
    assert cls.is_significant(0.850, 0.800, 0.02, folds_improved=1, n_folds=5) is False
    assert cls.is_significant(0.790, 0.800, 0.001, direction="minimize") is True


def test_tuner_parse_recomputes_stats_and_flags_unattributable_gains() -> None:
    out = ROLE_CLASSES["tuner"].parse_tuning(
        {
            "folds": [0.80, 0.82, 0.81, 0.83, 0.79],
            "accepted": True,
            "changes": [{"what": "lr", "accepted": True}, {"what": "epochs", "accepted": True}],
        }
    )
    assert out["local_score"] == pytest.approx(0.81)
    assert out["local_std"] > 0
    assert any("unattributable" in p for p in out["problems"])


def test_reporter_falls_back_to_a_ledger_report(tmp_path: Path) -> None:
    """The 30-minute deadline outranks a failed model turn."""
    role = make_role(ROLE_CLASSES["reporter"], tmp_path, {}, ok=False)
    out = role.write_report()
    assert out["generated"] == "fallback"
    assert out["markdown"].startswith("# Technical report: test-slug")
    assert "Efficiency metrics" in out["markdown"]
    assert (tmp_path / "REPORT.md").exists()
    assert out["declaration"]["internet_in_kernel"] is False
    assert "tokens_in" in out["efficiency"]


def test_reporter_keeps_model_markdown_when_present(tmp_path: Path) -> None:
    role = make_role(
        ROLE_CLASSES["reporter"], tmp_path,
        {"markdown": "# Technical report\n\nreal content", "declaration": {"models": ["x"]}},
    )
    out = role.write_report()
    assert out["generated"] == "role"
    assert "real content" in out["markdown"]
    assert out["declaration"]["models"] == ["x"]


def test_coder_downgrades_a_ready_build_without_a_kernel_dir() -> None:
    out = ROLE_CLASSES["coder"].parse_build({"status": "ready", "sanity": {}}, candidate_id="c1")
    assert out["status"] == "failed"
    assert out["candidate_id"] == "c1"
    assert any("kernel_dir" in p for p in out["problems"])


def test_coder_accepts_a_smoke_tested_build() -> None:
    out = ROLE_CLASSES["coder"].parse_build(
        {
            "candidate_id": "c1",
            "kernel_dir": "candidates/c1/kernel",
            "accelerator": "P100",
            "sanity": {"ran_end_to_end": True, "submission_written": True,
                       "format_matches_sample": True},
            "status": "ready",
        }
    )
    assert out["status"] == "ready"
    assert out["accelerator"] == "p100"
    assert out["problems"] == []
