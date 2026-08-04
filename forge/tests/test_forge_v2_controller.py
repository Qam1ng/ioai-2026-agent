from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "forge_v2_controller.py"
SPEC = importlib.util.spec_from_file_location("forge_v2_controller", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


class ForgeV2ControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.policy = ROOT / "config" / "forge_v2_policy.json"
        self.state = self.root / "state.json"
        self.events = self.root / "events.jsonl"
        self.ledger = self.root / "ledger.csv"
        self.task_config = self.root / "task_config.json"
        with (ROOT / "templates" / "experiment_ledger.csv").open(
            encoding="utf-8"
        ) as source:
            self.header = next(csv.reader(source))
        with self.ledger.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(self.header)
        self.config = {
            "schema_version": 1,
            "competition": "task-one-fixture",
            "execution_policy": {
                "submission_allowed": False,
                "clean_agent_benchmark_allowed": False,
            },
            "forge_v2_profile": {
                "benchmark_claim_allowed": False,
                "submission_allowed": False,
                "traces": [
                    {
                        "trace_id": "trace_1",
                        "root_axis": "representation",
                        "mechanism_family": "adapter",
                    },
                    {
                        "trace_id": "trace_2",
                        "root_axis": "geometry",
                        "mechanism_family": "prototype",
                    },
                    {
                        "trace_id": "trace_3",
                        "root_axis": "metadata",
                        "mechanism_family": "headers",
                    },
                ],
            },
        }
        self.task_config.write_text(
            json.dumps(self.config, indent=2) + "\n", encoding="utf-8"
        )
        controller.command_init(
            SimpleNamespace(
                policy=self.policy,
                state=self.state,
                events=self.events,
                ledger=self.ledger,
                task_config=self.task_config,
                task="task1",
                competition="task-one-fixture",
                budget_seconds=3600,
            )
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def proposal(
        self,
        node: str,
        trace_id: str,
        fidelity: str = "smoke",
        *,
        falsifier: str = "metric does not improve",
        minutes: float = 1.0,
    ) -> dict:
        trace = next(
            item
            for item in self.config["forge_v2_profile"]["traces"]
            if item["trace_id"] == trace_id
        )
        return {
            "schema_version": 1,
            "task": "task1",
            "node": node,
            "parent": "",
            "trace_id": trace_id,
            "root_axis": trace["root_axis"],
            "mechanism_family": trace["mechanism_family"],
            "fidelity": fidelity,
            "proposer": trace_id,
            "audit_sensitive_visible": False,
            "hypothesis": {
                "diagnosis": "a measured failure",
                "causal_hypothesis": "the atomic mechanism fixes it",
                "atomic_change": f"change {node}",
                "expected_observables": ["robust score changes"],
                "falsifier": falsifier,
            },
            "context": {
                "expected_gain": 0.02,
                "probability_of_gain": 0.6,
                "probability_of_valid_run": 0.9,
                "probability_of_finishing": 0.9,
                "expected_minutes": minutes,
                "uncertainty": 0.5,
                "novelty": 0.5,
                "risk": 0.1,
                "urgency": 0.5,
            },
        }

    def write_proposal(self, value: dict) -> Path:
        path = self.root / f"{value['node']}-proposal.json"
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        return path

    def register(self, value: dict) -> None:
        controller.command_register(
            SimpleNamespace(
                state=self.state,
                proposal=self.write_proposal(value),
            )
        )

    def append_ledger_row(
        self,
        node: str,
        *,
        status: str = "complete",
        suffix: str = "a",
    ) -> dict[str, str]:
        digest = suffix * 64
        row = {field: "" for field in self.header}
        row.update(
            {
                "timestamp_utc": "2026-07-29T00:00:00Z",
                "competition": "task-one-fixture",
                "task": "task1",
                "node": node,
                "parent": "",
                "operator": "IMPROVE",
                "hypothesis": f"hypothesis {node}",
                "change_axis": "fixture",
                "proposer": "fixture",
                "proposer_agent_id": "fixture-agent",
                "code_sha256": digest,
                "split_sha256": digest,
                "task_config_sha256": controller.sha256_file(self.task_config),
                "prediction_sha256": digest,
                "output_sha256": digest,
                "status": status,
                "run_status": status,
                "decision": "record",
                "submission_backend": "none",
            }
        )
        with self.ledger.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.header).writerow(row)
        return row

    def feedback(
        self,
        node: str,
        row: dict[str, str],
        fidelity: str,
        robust: float,
        group_delta: float,
    ) -> dict:
        return {
            "schema_version": 1,
            "task": "task1",
            "node": node,
            "fidelity": fidelity,
            "outcome": "complete",
            "ledger_binding": {
                field: row[field]
                for field in (
                    "task",
                    "node",
                    "code_sha256",
                    "split_sha256",
                    "task_config_sha256",
                    "prediction_sha256",
                    "output_sha256",
                )
            },
            "metrics": {
                "primary": robust,
                "secondary": robust,
                "robust_utility": robust,
                "score_std": 0.01,
                "group_deltas": {"rare": group_delta},
            },
            "textual_gradient": {
                "observed_failure": "rare group remains weak",
                "evidence_refs": ["fixture.json"],
                "causal_update": "the mechanism helped the rare group",
                "next_atomic_change": "refine the mechanism",
                "next_expected_observable": "rare group rises again",
                "next_falsifier": "rare group does not rise",
                "confidence": 0.7,
                "memory_visibility": "solver_shared",
            },
            "compliance_status": "pass",
            "output_contract_status": "pass",
            "leakage_status": "pass",
            "duplicate_prediction": False,
            "audit_sensitive_visible": False,
        }

    def ingest(self, value: dict) -> dict:
        path = self.root / f"{value['node']}-feedback.json"
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        return controller.command_ingest(
            SimpleNamespace(state=self.state, feedback=path)
        )

    def test_missing_falsifier_and_wrong_axis_are_rejected(self) -> None:
        missing = self.proposal("missing", "trace_1", falsifier="")
        with self.assertRaisesRegex(ValueError, "falsifier"):
            self.register(missing)
        wrong = self.proposal("wrong", "trace_2")
        wrong["root_axis"] = "representation"
        with self.assertRaisesRegex(ValueError, "root_axis"):
            self.register(wrong)

    def test_first_round_forces_all_three_traces(self) -> None:
        for index in range(1, 4):
            self.register(self.proposal(f"n{index}", f"trace_{index}"))
        observed = []
        for _ in range(3):
            result = controller.command_next(
                SimpleNamespace(state=self.state, receipt=None)
            )
            observed.append(result["trace_id"])
        self.assertEqual(observed, ["trace_1", "trace_2", "trace_3"])

    def test_controller_does_not_write_ledger(self) -> None:
        before = controller.sha256_file(self.ledger)
        self.register(self.proposal("n1", "trace_1"))
        controller.command_next(SimpleNamespace(state=self.state, receipt=None))
        self.assertEqual(before, controller.sha256_file(self.ledger))

    def test_full_champion_and_proxy_stepping_stone(self) -> None:
        champion_proposal = self.proposal("champion", "trace_1", "full")
        self.register(champion_proposal)
        controller.command_next(SimpleNamespace(state=self.state, receipt=None))
        champion_row = self.append_ledger_row("champion", suffix="a")
        champion = self.ingest(
            self.feedback("champion", champion_row, "full", 1.0, 0.0)
        )
        self.assertEqual(champion["classification"], "champion")

        stone_proposal = self.proposal("stone", "trace_2", "proxy")
        self.register(stone_proposal)
        controller.command_next(SimpleNamespace(state=self.state, receipt=None))
        stone_row = self.append_ledger_row("stone", suffix="b")
        stone = self.ingest(
            self.feedback("stone", stone_row, "proxy", 0.97, 0.03)
        )
        self.assertEqual(stone["classification"], "stepping_stone")
        audited = controller.command_audit(SimpleNamespace(state=self.state))
        self.assertTrue(audited["valid"])
        self.assertEqual(audited["archive"]["champion"]["node"], "champion")
        self.assertEqual(
            audited["archive"]["stepping_stones"][0]["node"], "stone"
        )

    def test_feedback_must_match_ledger_hashes(self) -> None:
        proposal = self.proposal("bound", "trace_1", "full")
        self.register(proposal)
        controller.command_next(SimpleNamespace(state=self.state, receipt=None))
        row = self.append_ledger_row("bound", suffix="c")
        feedback = self.feedback("bound", row, "full", 0.8, 0.0)
        feedback["ledger_binding"]["code_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "code_sha256"):
            self.ingest(feedback)

    def test_audit_detects_event_tampering(self) -> None:
        self.register(self.proposal("tamper", "trace_1"))
        lines = self.events.read_text(encoding="utf-8").splitlines()
        event = json.loads(lines[-1])
        event["payload"]["node"] = "changed"
        lines[-1] = json.dumps(event, sort_keys=True)
        self.events.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "event hash mismatch"):
            controller.command_audit(SimpleNamespace(state=self.state))

    def test_task_contract_remains_non_submission_postmortem(self) -> None:
        state = controller.load_json(self.state)
        self.assertIs(state["submission_allowed"], False)
        self.assertIs(state["clean_benchmark_claim_allowed"], False)


if __name__ == "__main__":
    unittest.main()
